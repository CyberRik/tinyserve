"""Batch Builder (PRD Section 6.4): a pure transformation, no persistent state.

Turns the set of in-flight sequences into the row layout llama.cpp's batch
API expects. No llama.cpp import here — only llama_runtime.py is allowed to
touch the bindings (PRD Section 13/14); this module works entirely in plain
ints and dataclasses.
"""

from dataclasses import dataclass

DEFAULT_CHUNK_SIZE = 512


@dataclass
class ActiveSequence:
    """One in-flight sequence's state, owned by the batch loop between ticks.

    pending_tokens holds *everything* not yet fed to llama_decode — for a
    sequence still in prefill that can be the whole remaining prompt, not
    just this tick's slice; build_batch decides how much of it to consume.
    """

    seq_id: int
    pending_tokens: list[int]
    n_past: int  # positions already committed to this sequence's KV cache


@dataclass(frozen=True)
class BatchRow:
    seq_id: int
    token: int
    pos: int
    needs_logits: bool  # True only on the row that reaches the end of pending_tokens


@dataclass(frozen=True)
class PreparedBatch:
    rows: list[BatchRow]

    def __len__(self) -> int:
        return len(self.rows)

    def row_count(self, seq_id: int) -> int:
        """How many of a sequence's pending tokens this batch consumes —
        the batch loop uses this to trim pending_tokens and advance n_past,
        since a prefill chunk may only be a partial slice."""
        return sum(1 for row in self.rows if row.seq_id == seq_id)


def build_batch(
    active: list[ActiveSequence], capacity: int, chunk_size: int = DEFAULT_CHUNK_SIZE
) -> PreparedBatch:
    """Fit a slice of each sequence's pending tokens into `capacity` rows this tick.

    Chunked prefill: a sequence's prefill is capped at `chunk_size` tokens
    per tick rather than being forced to fit in full — this is what keeps
    one long prompt from stalling every other sequence's decode step for
    an entire tick (PRD Section 7). Decode steps (a single pending token)
    sort first, so in-flight sequences keep making progress even when a
    large new prompt is waiting; whatever capacity remains goes to prefill.
    """
    ordered = sorted(active, key=lambda seq: len(seq.pending_tokens))
    rows: list[BatchRow] = []
    remaining = capacity

    for seq in ordered:
        if not seq.pending_tokens or remaining <= 0:
            continue
        take = min(chunk_size, len(seq.pending_tokens), remaining)
        is_final_slice = take == len(seq.pending_tokens)
        rows.extend(
            BatchRow(
                seq_id=seq.seq_id,
                token=token,
                pos=seq.n_past + i,
                needs_logits=is_final_slice and i == take - 1,
            )
            for i, token in enumerate(seq.pending_tokens[:take])
        )
        remaining -= take

    return PreparedBatch(rows=rows)
