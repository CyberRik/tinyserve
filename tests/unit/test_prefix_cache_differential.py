"""Differential test: the Python fallback against the C++ tree, via subprocess.

test_prefix_cache.py already fuzzes the two backends against each other through
ctypes -- but only where a *loadable* shared library exists, which requires the
C++ toolchain and the interpreter to share an ABI. This module drops that
requirement: it compiles native/tests/replay_driver.cpp, pipes an operation
script through it, and compares its trace to the Python cache's. Any host with
a working compiler can run it, whatever bitness that compiler targets.

Skips cleanly when no compiler is present.
"""

import random
import shutil
import subprocess
from pathlib import Path

import pytest

from tinyserve.prefix.cache import NO_SEQ, PythonPrefixCache

NATIVE = Path(__file__).resolve().parents[2] / "native"
BLOCK = 16

pytestmark = pytest.mark.slow


def _compiler() -> str | None:
    for name in ("g++", "clang++"):
        if shutil.which(name):
            return name
    return None


@pytest.fixture(scope="module")
def driver(tmp_path_factory) -> Path:
    compiler = _compiler()
    if compiler is None:
        pytest.skip("no C++ compiler on PATH")
    out = tmp_path_factory.mktemp("native") / "replay_driver"
    result = subprocess.run(
        [
            compiler,
            "-std=c++11",
            "-O2",
            "-I",
            str(NATIVE / "include"),
            str(NATIVE / "src" / "prefix_cache.cpp"),
            str(NATIVE / "tests" / "replay_driver.cpp"),
            "-o",
            str(out),
        ],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        pytest.skip(f"replay driver did not build: {result.stderr.strip()[:300]}")
    return out


def _script(rng: random.Random, n_ops: int) -> list[tuple]:
    """Insert/evict/match churn over a small block pool, so prompts genuinely
    collide and the tree develops real sharing instead of a flat fan-out."""
    pool = [[rng.randrange(0, 5) * 1000 + i for i in range(BLOCK)] for _ in range(6)]
    ops: list[tuple] = [("block", BLOCK)]
    for _ in range(n_ops):
        roll = rng.random()
        seq_id = rng.randrange(0, 8)
        prompt = [t for _ in range(rng.randrange(1, 5)) for t in rng.choice(pool)]
        if roll < 0.45:
            ops.append(("insert", seq_id, prompt))
        elif roll < 0.70:
            ops.append(("evict", seq_id))
        else:
            ops.append(("match", prompt))
        ops.append(("stat",))
    return ops


def _render(ops: list[tuple]) -> str:
    lines = []
    for op in ops:
        if op[0] == "insert":
            lines.append(f"insert {op[1]} " + " ".join(str(t) for t in op[2]))
        elif op[0] == "match":
            lines.append("match " + " ".join(str(t) for t in op[1]))
        elif op[0] == "evict":
            lines.append(f"evict {op[1]}")
        elif op[0] == "block":
            lines.append(f"block {op[1]}")
        else:
            lines.append("stat")
    return "\n".join(lines) + "\n"


def _run_python(ops: list[tuple]) -> list[str]:
    cache = PythonPrefixCache(BLOCK)
    out: list[str] = []
    for op in ops:
        if op[0] == "block":
            cache = PythonPrefixCache(op[1])
            out.append(f"block {op[1]}")
        elif op[0] == "insert":
            out.append(f"insert {cache.insert(op[2], op[1])}")
        elif op[0] == "match":
            hit = cache.match(op[1])
            donor = hit.seq_id if hit.n_tokens > 0 else NO_SEQ
            out.append(f"match {hit.n_tokens} {donor}")
        elif op[0] == "evict":
            cache.evict(op[1])
            out.append("evict")
        else:
            out.append(f"stat {cache.node_count()} {cache.seq_count()}")
    return out


def test_python_fallback_matches_the_cpp_tree_op_for_op(driver: Path) -> None:
    rng = random.Random(20260819)
    ops = _script(rng, 4000)

    result = subprocess.run(
        [str(driver)], input=_render(ops), capture_output=True, text=True, timeout=120
    )
    assert result.returncode == 0, result.stderr

    native_lines = result.stdout.splitlines()
    python_lines = _run_python(ops)
    assert len(native_lines) == len(python_lines)

    for index, (native, python) in enumerate(zip(native_lines, python_lines, strict=True)):
        if native != python:
            context = "\n".join(
                f"  {ops[j]!r:.100} -> native={native_lines[j]!r} python={python_lines[j]!r}"
                for j in range(max(0, index - 3), index + 1)
            )
            pytest.fail(f"backends diverged at op {index}:\n{context}")
