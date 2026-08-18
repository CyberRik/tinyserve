"""Proves Phase 5's actual claim: a sequence that skips prefill by copying
another sequence's KV cells generates the *same tokens* it would have generated
by prefilling the prompt itself.

This is the test the feature lives or dies on. Every other prefix-cache test is
bookkeeping -- tree shape, node counts, donor selection -- and bookkeeping that
is perfectly self-consistent can still hand the batch loop an n_past that points
at cells holding something else. The failure mode there is not a crash: it is a
sequence that decodes fluent, plausible, *wrong* text. Only a token-for-token
comparison against a full prefill catches it.
"""

from pathlib import Path

import pytest

from tinyserve.batch.builder import ActiveSequence, build_batch
from tinyserve.prefix.cache import create_prefix_cache
from tinyserve.runtime.llama_runtime import LlamaRuntime

MODEL_PATH = Path(__file__).resolve().parents[2] / "models" / "qwen2.5-0.5b-instruct-q4_k_m.gguf"

pytestmark = [
    pytest.mark.slow,
    pytest.mark.skipif(not MODEL_PATH.exists(), reason="test GGUF model not present locally"),
]

SHARED_PROMPT = (
    "You are a careful assistant. Answer in one short sentence. "
    "Be precise and do not speculate. Here is the question: "
)


@pytest.fixture
def runtime():
    """One Runtime per test, closed deterministically.

    Letting these fall to the garbage collector faults: see LlamaRuntime.close()
    for why, and note that it was this file -- the first to build more than one
    Runtime in a session -- that made the latent use-after-free reproducible.
    """
    with LlamaRuntime(str(MODEL_PATH), n_ctx=1024, n_seq_max=3) as rt:
        yield rt


async def _generate(
    runtime: LlamaRuntime, seq_id: int, tokens: list[int], n_past: int, n_new: int
) -> list[int]:
    """Decode `n_new` greedy tokens for one sequence, starting at `n_past`."""
    seq = ActiveSequence(seq_id=seq_id, pending_tokens=tokens[n_past:], n_past=n_past)
    out: list[int] = []
    while len(out) < n_new:
        batch = build_batch([seq], runtime.batch_capacity)
        sampled = await runtime.decode(batch)
        consumed = batch.row_count(seq_id)
        seq.n_past += consumed
        seq.pending_tokens = seq.pending_tokens[consumed:]
        if seq_id in sampled:
            out.append(sampled[seq_id])
            seq.pending_tokens = [sampled[seq_id]]
    return out


async def test_reused_prefix_generates_identical_tokens_to_a_full_prefill(runtime) -> None:
    cache = create_prefix_cache(block_size=16)

    prompt = runtime.tokenize(SHARED_PROMPT + "What is the capital of France?")

    # Sequence 0 prefills the whole prompt the ordinary way, and publishes it.
    baseline = await _generate(runtime, 0, prompt, n_past=0, n_new=12)
    cache.insert(prompt, seq_id=0)

    # Sequence 1 gets the same prompt. The cache should let it skip most of the
    # prefill by copying sequence 0's cells.
    hit = cache.match_for_reuse(prompt)
    assert hit.n_tokens > 0, "identical prompt should hit"
    assert hit.seq_id == 0
    assert hit.n_tokens % 16 == 0
    assert hit.n_tokens < len(prompt), "the clamp must leave something to prefill"

    assert runtime.reuse_prefix(hit.seq_id, 1, hit.n_tokens) is True
    reused = await _generate(runtime, 1, prompt, n_past=hit.n_tokens, n_new=12)

    assert reused == baseline, (
        "reusing KV cells changed the output -- n_past and the copied cells disagree"
    )


async def test_partial_overlap_reuse_matches_a_full_prefill(runtime) -> None:
    """The realistic case: a shared system prompt, different questions. Only the
    common prefix is reused, and the divergent tail still prefills normally."""
    cache = create_prefix_cache(block_size=16)

    first = runtime.tokenize(SHARED_PROMPT + "What is the capital of France?")
    second = runtime.tokenize(SHARED_PROMPT + "Name the largest ocean on Earth.")

    await _generate(runtime, 0, first, n_past=0, n_new=4)
    cache.insert(first, seq_id=0)

    hit = cache.match_for_reuse(second)
    assert 0 < hit.n_tokens < len(second), "the shared system prompt should partly match"

    baseline = await _generate(runtime, 1, second, n_past=0, n_new=12)
    assert runtime.reuse_prefix(0, 2, hit.n_tokens) is True
    reused = await _generate(runtime, 2, second, n_past=hit.n_tokens, n_new=12)

    assert reused == baseline


async def test_reuse_declines_when_the_donor_has_been_freed(runtime) -> None:
    """The stale-donor race, which the batch loop treats as a miss rather than
    an error: the donor can finish between the tree lookup and the copy."""
    prompt = runtime.tokenize(SHARED_PROMPT + "What is the capital of France?")

    await _generate(runtime, 0, prompt, n_past=0, n_new=4)
    assert runtime.reuse_prefix(0, 1, 32) is True

    runtime.free_sequence(0)

    assert runtime.reuse_prefix(0, 2, 32) is False, (
        "copying from a freed donor must be refused, not silently produce empty cells"
    )
