"""Prefix cache: "how much of this prompt has someone already prefilled?"

Two interchangeable implementations behind one interface:

* ``NativePrefixCache`` -- ctypes binding to the C++ radix tree in ``native/``.
* ``PythonPrefixCache`` -- pure-Python fallback with identical semantics.

The fallback is not a stub. TinyServe is installable with ``pip install`` on a
machine with no compiler, and prefix reuse is a correctness-visible feature
(``n_past`` is set from its answer) -- so "no C++ toolchain" has to mean "no
speedup", never "different behaviour". ``tests/unit/test_prefix_cache.py`` runs
the same suite against both and, when the shared library is present,
differentially fuzzes one against the other.

Note the layering rule from PRD Section 13/14 still holds: nothing here imports
llama.cpp. This module answers a question about token lists. Turning that answer
into ``llama_memory_seq_cp`` calls is the Runtime's job.
"""

from __future__ import annotations

import contextlib
import ctypes
import logging
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# Bumped when the C ABI changes shape. The loader refuses a library whose major
# version it does not recognise rather than calling into a stale layout.
_SUPPORTED_ABI_MAJOR = "1"

NO_SEQ = -1


@dataclass(frozen=True)
class PrefixMatch:
    """A block-aligned prefix hit.

    `n_tokens` is always a multiple of the cache's block size, and `seq_id` is
    the sequence whose KV cells hold it (NO_SEQ when `n_tokens` is 0).
    """

    n_tokens: int
    seq_id: int

    def __bool__(self) -> bool:
        return self.n_tokens > 0


class PrefixCacheBase:
    """Shared clamping logic, so both backends cannot drift on the one rule
    that is TinyServe's rather than the index's."""

    block_size: int
    backend: str

    def match(self, tokens: list[int]) -> PrefixMatch:  # pragma: no cover - interface
        """Longest block-aligned prefix of `tokens` already resident."""
        raise NotImplementedError

    def insert(self, tokens: list[int], seq_id: int) -> int:  # pragma: no cover - interface
        """Publish `tokens` as resident under `seq_id`; returns tokens published."""
        raise NotImplementedError

    def evict(self, seq_id: int) -> None:  # pragma: no cover - interface
        """Drop `seq_id`'s claim."""
        raise NotImplementedError

    def node_count(self) -> int:  # pragma: no cover - interface
        raise NotImplementedError

    def seq_count(self) -> int:  # pragma: no cover - interface
        raise NotImplementedError

    def close(self) -> None:  # pragma: no cover - interface
        raise NotImplementedError

    def match_for_reuse(self, tokens: list[int]) -> PrefixMatch:
        """`match`, clamped so at least one token is left to prefill.

        A sequence whose entire prompt was reused would enter the batch loop
        with no pending tokens: no row in the batch, so no row flagged
        needs_logits, so no sampled token and no way to ever produce one -- it
        would sit in `active` forever, holding a slot. Keeping the last block
        costs one block of redundant prefill and removes that whole class of
        stall.

        This clamp lives here, not in the C++ tree, because it encodes how
        TinyServe's batch loop works rather than anything about prefix indexing.
        """
        usable = ((len(tokens) - 1) // self.block_size) * self.block_size
        if usable <= 0:
            return PrefixMatch(0, NO_SEQ)
        hit = self.match(tokens)
        if hit.n_tokens <= usable:
            return hit
        return PrefixMatch(usable, hit.seq_id)


def _new_node(
    parent: dict[str, Any] | None, key: tuple[int, ...] | None, depth: int
) -> dict[str, Any]:
    return {"children": {}, "owners": [], "parent": parent, "key": key, "depth": depth}


class PythonPrefixCache(PrefixCacheBase):
    """Reference implementation. Mirrors native/src/prefix_cache.cpp exactly --
    same block-keyed tree, same owner-on-every-path-node rule, same
    most-recent-owner donor choice, same deepest-first eviction."""

    backend = "python"

    def __init__(self, block_size: int) -> None:
        if block_size < 1:
            raise ValueError("block_size must be >= 1")
        self.block_size = block_size
        # node = {"children": {key: node}, "owners": [seq_id], "parent", "key", "depth"}
        self._root: dict[str, Any] = _new_node(None, None, 0)
        self._seq_nodes: dict[int, list[dict[str, Any]]] = {}
        self._node_count = 0

    def _blocks(self, tokens: list[int]) -> list[tuple[int, ...]]:
        bs = self.block_size
        n = len(tokens) // bs
        return [tuple(tokens[i * bs : (i + 1) * bs]) for i in range(n)]

    def match(self, tokens: list[int]) -> PrefixMatch:
        if len(tokens) < self.block_size:
            return PrefixMatch(0, NO_SEQ)
        node = self._root
        matched = 0
        for key in self._blocks(tokens):
            child = node["children"].get(key)
            if child is None:
                break
            node = child
            matched += 1
        if matched == 0:
            return PrefixMatch(0, NO_SEQ)
        # Every live node carries an owner; see the ownership note in the C++.
        donor = node["owners"][-1] if node["owners"] else NO_SEQ
        return PrefixMatch(matched * self.block_size, donor)

    def insert(self, tokens: list[int], seq_id: int) -> int:
        if len(tokens) < self.block_size:
            return 0
        self.evict(seq_id)  # one claim per sequence; the newer path supersedes
        node = self._root
        owned: list[dict[str, Any]] = []
        blocks = self._blocks(tokens)
        for key in blocks:
            child = node["children"].get(key)
            if child is None:
                child = _new_node(node, key, node["depth"] + 1)
                node["children"][key] = child
                self._node_count += 1
            node = child
            node["owners"].append(seq_id)
            owned.append(node)
        self._seq_nodes[seq_id] = owned
        return len(blocks) * self.block_size

    def evict(self, seq_id: int) -> None:
        owned = self._seq_nodes.pop(seq_id, None)
        if owned is None:
            return
        # Deepest first. Every ancestor of an owned node is also owned, so this
        # list is closed under "parent of" and no upward cascade is needed --
        # which is also why no node is ever visited twice.
        for node in sorted(owned, key=lambda n: -n["depth"]):
            owners = node["owners"]
            if seq_id in owners:
                owners.remove(seq_id)
            if owners or node["children"]:
                continue
            parent = node["parent"]
            if parent is not None:
                parent["children"].pop(node["key"], None)
            self._node_count -= 1

    def node_count(self) -> int:
        return self._node_count

    def seq_count(self) -> int:
        return len(self._seq_nodes)

    def close(self) -> None:
        self._root = _new_node(None, None, 0)
        self._seq_nodes.clear()
        self._node_count = 0


def _candidate_library_paths() -> list[Path]:
    """Where the shared library might be, most specific first."""
    override = os.environ.get("TINYSERVE_PREFIX_LIB")
    if override:
        return [Path(override)]

    if sys.platform == "win32":
        names = ["tinyserve_prefix.dll"]
    elif sys.platform == "darwin":
        names = ["tinyserve_prefix.dylib", "libtinyserve_prefix.dylib"]
    else:
        names = ["tinyserve_prefix.so", "libtinyserve_prefix.so"]

    native = Path(__file__).resolve().parent.parent.parent / "native"
    roots = [native / "build", native / "build" / "Release", native, Path.cwd()]
    return [root / name for root in roots for name in names]


class NativePrefixCache(PrefixCacheBase):
    """ctypes binding to native/src/prefix_cache.cpp."""

    backend = "native"
    # Process-wide: the library is stateless, so one load serves every cache.
    _shared_lib: ctypes.CDLL | None = None

    @classmethod
    def load_library(cls) -> ctypes.CDLL | None:
        """Locate and bind the shared library, or return None.

        Returning None rather than raising is deliberate: an absent or stale
        library is a normal deployment state (no compiler on the box), and the
        fallback covers it. The one thing that must not happen is calling into
        a library whose ABI does not match this binding.
        """
        if cls._shared_lib is not None:
            return cls._shared_lib
        for path in _candidate_library_paths():
            if not path.exists():
                continue
            try:
                lib = ctypes.CDLL(str(path))
                _bind_signatures(lib)
                version = lib.ts_prefix_cache_abi_version().decode("ascii")
            except (OSError, AttributeError, UnicodeDecodeError) as exc:
                logger.warning("prefix cache: %s is not loadable (%s)", path, exc)
                continue
            if version.split(".")[0] != _SUPPORTED_ABI_MAJOR:
                logger.warning(
                    "prefix cache: %s reports ABI %s, this build needs %s.x -- ignoring",
                    path,
                    version,
                    _SUPPORTED_ABI_MAJOR,
                )
                continue
            logger.info("prefix cache: loaded native backend from %s (ABI %s)", path, version)
            cls._shared_lib = lib
            return lib
        return None

    def __init__(self, block_size: int) -> None:
        if block_size < 1:
            raise ValueError("block_size must be >= 1")
        lib = self.load_library()
        if lib is None:
            raise OSError("tinyserve_prefix shared library not found")
        self.block_size = block_size
        self._lib: ctypes.CDLL = lib
        self._handle: int | None = lib.ts_prefix_cache_create(block_size)
        if not self._handle:
            raise OSError("ts_prefix_cache_create returned NULL")
        self._donor = ctypes.c_int32(NO_SEQ)

    @staticmethod
    def _buffer(tokens: list[int]) -> ctypes.Array[ctypes.c_int32]:
        return (ctypes.c_int32 * len(tokens))(*tokens)

    def match(self, tokens: list[int]) -> PrefixMatch:
        if len(tokens) < self.block_size:
            return PrefixMatch(0, NO_SEQ)
        buf = self._buffer(tokens)
        n = self._lib.ts_prefix_cache_match(
            self._handle, buf, len(tokens), ctypes.byref(self._donor)
        )
        return PrefixMatch(int(n), int(self._donor.value) if n > 0 else NO_SEQ)

    def insert(self, tokens: list[int], seq_id: int) -> int:
        if len(tokens) < self.block_size:
            return 0
        buf = self._buffer(tokens)
        return int(self._lib.ts_prefix_cache_insert(self._handle, buf, len(tokens), seq_id))

    def evict(self, seq_id: int) -> None:
        self._lib.ts_prefix_cache_evict(self._handle, seq_id)

    def node_count(self) -> int:
        return int(self._lib.ts_prefix_cache_node_count(self._handle))

    def seq_count(self) -> int:
        return int(self._lib.ts_prefix_cache_seq_count(self._handle))

    def close(self) -> None:
        if getattr(self, "_handle", None) is not None:
            self._lib.ts_prefix_cache_destroy(self._handle)
            self._handle = None

    def __del__(self) -> None:
        # Interpreter teardown can already have torn down what close() touches.
        with contextlib.suppress(Exception):  # pragma: no cover
            self.close()


def _bind_signatures(lib: ctypes.CDLL) -> None:
    """Pin argtypes/restype. Without this ctypes defaults every return to C int,
    which truncates the 64-bit opaque handle to 32 bits and hands back a pointer
    that segfaults on first use."""
    lib.ts_prefix_cache_create.argtypes = [ctypes.c_int32]
    lib.ts_prefix_cache_create.restype = ctypes.c_void_p
    lib.ts_prefix_cache_destroy.argtypes = [ctypes.c_void_p]
    lib.ts_prefix_cache_destroy.restype = None
    lib.ts_prefix_cache_match.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_int32),
        ctypes.c_int32,
        ctypes.POINTER(ctypes.c_int32),
    ]
    lib.ts_prefix_cache_match.restype = ctypes.c_int32
    lib.ts_prefix_cache_insert.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_int32),
        ctypes.c_int32,
        ctypes.c_int32,
    ]
    lib.ts_prefix_cache_insert.restype = ctypes.c_int32
    lib.ts_prefix_cache_evict.argtypes = [ctypes.c_void_p, ctypes.c_int32]
    lib.ts_prefix_cache_evict.restype = None
    lib.ts_prefix_cache_node_count.argtypes = [ctypes.c_void_p]
    lib.ts_prefix_cache_node_count.restype = ctypes.c_int32
    lib.ts_prefix_cache_seq_count.argtypes = [ctypes.c_void_p]
    lib.ts_prefix_cache_seq_count.restype = ctypes.c_int32
    lib.ts_prefix_cache_abi_version.argtypes = []
    lib.ts_prefix_cache_abi_version.restype = ctypes.c_char_p


def create_prefix_cache(block_size: int, prefer_native: bool = True) -> PrefixCacheBase:
    """Native backend when the shared library is available, Python otherwise.

    Which one ran is reported through the `tinyserve_prefix_cache_backend`
    metric label, so a deployment that silently fell back is visible on the
    dashboard instead of being a mystery in a latency graph.
    """
    if prefer_native:
        try:
            return NativePrefixCache(block_size)
        except OSError as exc:
            logger.info("prefix cache: using Python fallback (%s)", exc)
    return PythonPrefixCache(block_size)
