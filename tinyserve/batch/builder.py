"""Batch Builder (PRD Section 6.4): a pure transformation, no persistent state.

Turns the set of in-flight sequences into the row layout llama.cpp's batch
API expects. No llama.cpp import here — only llama_runtime.py is allowed to
touch the bindings (PRD Section 13/14); this module works entirely in plain
ints and dataclasses.
"""

from dataclasses import dataclass


@dataclass
class ActiveSequence:
    """One in-flight sequence's state, owned by the batch loop between ticks."""

    seq_id: int
    pending_tokens: list[int]  # tokens not yet fed to llama_decode this tick
    n_past: int  # positions already committed to this sequence's KV cache


@dataclass(frozen=True)
class BatchRow:
    seq_id: int
    token: int
    pos: int
    needs_logits: bool  # True only for a sequence's last row — where we sample


@dataclass(frozen=True)
class PreparedBatch:
    rows: list[BatchRow]

    def __len__(self) -> int:
        return len(self.rows)


def build_batch(active: list[ActiveSequence], capacity: int) -> PreparedBatch:
    """Fit as many whole sequences' pending tokens as possible into `capacity` rows.

    A sequence is included only if *all* of its pending tokens fit this tick —
    splitting one sequence's tokens across ticks is what chunked prefill
    (Phase 3) formalizes; Phase 2 keeps whole-prompt prefill and just lets a
    prompt that doesn't fit this tick wait for the next one. Decode steps
    (a single pending token) sort first, so in-flight sequences keep making
    progress even when a large new prompt is waiting.
    """
    ordered = sorted(active, key=lambda seq: len(seq.pending_tokens))
    rows: list[BatchRow] = []
    remaining = capacity

    for seq in ordered:
        n = len(seq.pending_tokens)
        if n == 0 or n > remaining:
            continue
        rows.extend(
            BatchRow(seq_id=seq.seq_id, token=token, pos=seq.n_past + i, needs_logits=i == n - 1)
            for i, token in enumerate(seq.pending_tokens)
        )
        remaining -= n

    return PreparedBatch(rows=rows)
