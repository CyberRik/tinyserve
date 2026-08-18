"""Prefix cache tests.

Every behavioural test is parametrised over both backends, so the pure-Python
fallback is held to the same contract as the C++ tree rather than being trusted
because it looks similar. When the shared library is absent (no compiler on the
box, which is a supported deployment) the native parameter skips and the Python
parameter still runs -- the suite never silently tests nothing.

The C++ implementation additionally has its own standalone harness in
native/tests/test_prefix_cache.cpp, which covers what ctypes cannot reach from
here: teardown of a tree deep enough to overflow a recursive destructor, and
NULL-pointer arguments.
"""

import random

import pytest

from tinyserve.prefix.cache import (
    NO_SEQ,
    NativePrefixCache,
    PythonPrefixCache,
    create_prefix_cache,
)

BLOCK = 16

_native_available = NativePrefixCache.load_library() is not None
_native_param = pytest.param(
    NativePrefixCache,
    id="native",
    marks=pytest.mark.skipif(
        not _native_available,
        reason="tinyserve_prefix shared library not built (see native/README.md)",
    ),
)
BACKENDS = [pytest.param(PythonPrefixCache, id="python"), _native_param]


@pytest.fixture(params=BACKENDS)
def cache(request):
    instance = request.param(BLOCK)
    yield instance
    instance.close()


def toks(seed: int, n: int) -> list[int]:
    return [seed + i for i in range(n)]


# --------------------------------------------------------------------------
# Matching


def test_empty_cache_never_matches(cache) -> None:
    hit = cache.match(toks(1000, 64))
    assert hit.n_tokens == 0
    assert hit.seq_id == NO_SEQ
    assert not hit
    assert cache.node_count() == 0


def test_identical_prompt_matches_in_full(cache) -> None:
    prompt = toks(1000, 64)
    assert cache.insert(prompt, seq_id=7) == 64
    assert cache.node_count() == 4

    hit = cache.match(prompt)
    assert (hit.n_tokens, hit.seq_id) == (64, 7)


def test_divergent_prompt_matches_only_the_shared_blocks(cache) -> None:
    cache.insert(toks(1000, 64), seq_id=7)

    hit = cache.match(toks(1000, 32) + toks(9000, 32))
    assert (hit.n_tokens, hit.seq_id) == (32, 7)


def test_divergence_inside_the_first_block_matches_nothing(cache) -> None:
    cache.insert(toks(1000, 64), seq_id=7)
    assert cache.match(toks(4242, 64)).n_tokens == 0


def test_matches_are_always_block_aligned(cache) -> None:
    # 40 tokens = 2 whole blocks + an 8-token tail that cannot be published.
    prompt = toks(1000, 40)
    assert cache.insert(prompt, seq_id=1) == 32
    assert cache.match(prompt).n_tokens == 32


def test_sub_block_prompts_are_inert(cache) -> None:
    tiny = toks(1000, BLOCK - 1)
    assert cache.insert(tiny, seq_id=2) == 0
    assert cache.match(tiny).n_tokens == 0
    assert cache.seq_count() == 0


# --------------------------------------------------------------------------
# Sharing and eviction -- the properties the feature actually rests on


def test_shared_prefix_survives_one_owner_leaving(cache) -> None:
    shared = toks(500, 32)
    cache.insert(shared + toks(7000, 32), seq_id=1)
    cache.insert(shared + toks(8000, 32), seq_id=2)
    assert cache.node_count() == 6  # 2 shared + 2 private each

    cache.evict(1)

    assert cache.node_count() == 4  # only seq 1's private blocks went
    hit = cache.match(shared)
    assert (hit.n_tokens, hit.seq_id) == (32, 2)  # donor rolls over


def test_eviction_reclaims_everything_once_the_last_owner_leaves(cache) -> None:
    shared = toks(500, 32)
    cache.insert(shared + toks(7000, 32), seq_id=1)
    cache.insert(shared + toks(8000, 32), seq_id=2)

    cache.evict(1)
    cache.evict(2)

    assert cache.node_count() == 0
    assert cache.seq_count() == 0


def test_evicted_sequences_private_blocks_are_gone(cache) -> None:
    prompt = toks(500, 32) + toks(7000, 32)
    cache.insert(prompt, seq_id=1)
    cache.insert(toks(500, 32) + toks(8000, 32), seq_id=2)

    cache.evict(1)

    assert cache.match(prompt).n_tokens == 32  # shared part only


def test_reinsert_supersedes_rather_than_duplicates(cache) -> None:
    cache.insert(toks(1000, 32), seq_id=5)
    assert cache.node_count() == 2

    cache.insert(toks(1000, 64), seq_id=5)  # same sequence, grown

    assert cache.node_count() == 4
    assert cache.seq_count() == 1
    cache.evict(5)
    assert cache.node_count() == 0  # the superseded claim left nothing behind


def test_donor_is_the_most_recent_owner(cache) -> None:
    prompt = toks(1000, 32)
    for seq_id in (3, 4, 9):
        cache.insert(prompt, seq_id)

    assert cache.match(prompt).seq_id == 9
    cache.evict(9)
    assert cache.match(prompt).seq_id == 4


def test_evicting_an_unknown_sequence_is_a_noop(cache) -> None:
    cache.insert(toks(1000, 32), seq_id=1)
    cache.evict(999)
    assert cache.node_count() == 2


# --------------------------------------------------------------------------
# The clamp that belongs to TinyServe rather than to the index


def test_match_for_reuse_always_leaves_a_block_to_prefill(cache) -> None:
    prompt = toks(1000, 64)
    cache.insert(prompt, seq_id=1)

    # A full 64-token match would leave the sequence with no pending tokens,
    # hence no batch row, hence no sampled token -- a permanent stall.
    assert cache.match(prompt).n_tokens == 64
    assert cache.match_for_reuse(prompt).n_tokens == 48


def test_match_for_reuse_is_transparent_below_the_clamp(cache) -> None:
    cache.insert(toks(1000, 32), seq_id=1)
    prompt = toks(1000, 32) + toks(9000, 32)

    assert cache.match_for_reuse(prompt) == cache.match(prompt)


def test_match_for_reuse_declines_a_single_block_prompt(cache) -> None:
    prompt = toks(1000, BLOCK)
    cache.insert(prompt, seq_id=1)

    # Nothing can be reused without consuming the only block there is.
    assert cache.match_for_reuse(prompt).n_tokens == 0


# --------------------------------------------------------------------------
# Differential fuzz


@pytest.mark.skipif(not _native_available, reason="native backend not built")
def test_python_and_native_backends_agree_under_random_churn() -> None:
    """Random insert/evict/match churn, asserting the two backends stay
    bit-identical. This is the test that makes the fallback trustworthy: the
    cases above encode what I thought to check, this one covers what I didn't.
    """
    rng = random.Random(20260819)
    py = PythonPrefixCache(BLOCK)
    native = NativePrefixCache(BLOCK)
    # A small vocabulary and few distinct blocks, so prompts genuinely collide
    # and the tree gets deep sharing rather than a flat fan-out of misses.
    pool = [toks(rng.randrange(0, 5) * 1000, BLOCK) for _ in range(6)]

    try:
        for step in range(3000):
            op = rng.random()
            seq_id = rng.randrange(0, 8)
            prompt = [t for _ in range(rng.randrange(1, 5)) for t in rng.choice(pool)]

            if op < 0.45:
                assert py.insert(prompt, seq_id) == native.insert(prompt, seq_id), step
            elif op < 0.70:
                py.evict(seq_id)
                native.evict(seq_id)
            else:
                assert py.match(prompt) == native.match(prompt), step
                assert py.match_for_reuse(prompt) == native.match_for_reuse(prompt), step

            assert py.node_count() == native.node_count(), step
            assert py.seq_count() == native.seq_count(), step
    finally:
        py.close()
        native.close()


# --------------------------------------------------------------------------
# Factory


def test_factory_falls_back_to_python_when_native_is_unavailable() -> None:
    cache = create_prefix_cache(BLOCK, prefer_native=False)
    assert cache.backend == "python"
    cache.close()


def test_factory_prefers_native_when_it_is_available() -> None:
    cache = create_prefix_cache(BLOCK)
    assert cache.backend == ("native" if _native_available else "python")
    cache.close()


@pytest.mark.parametrize("backend", BACKENDS)
def test_block_size_must_be_positive(backend) -> None:
    with pytest.raises(ValueError):
        backend(0)
