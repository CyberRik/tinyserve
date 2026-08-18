"""The only module allowed to import llama.cpp bindings (PRD Section 13/14).

Phase 2: real multi-sequence continuous batching. The Runtime owns one
llama_context created with kv_unified=True and n_seq_max concurrent
sequence slots sharing that single KV-cache budget (PRD Section 8) — the
batch loop feeds it a PreparedBatch built by tinyserve.batch.builder each
tick, and one llama_decode() call advances every sequence in that batch at
once, however many requests it contains.

Sampling here is plain greedy (argmax) — no temperature/top-k/top-p yet;
that's a real simplification, not an oversight, kept out of scope until a
phase that actually needs configurable sampling.
"""

import asyncio

import numpy as np
from llama_cpp import Llama, llama_cpp
from llama_cpp._internals import LlamaContext

from tinyserve.batch.builder import PreparedBatch


class LlamaRuntime:
    """Loads one GGUF model and runs batched decode ticks against it."""

    def __init__(self, model_path: str, n_ctx: int, n_seq_max: int) -> None:
        # A throwaway, minimal-size context: only used for its tokenizer and
        # loaded model weights. The real decode context is built separately
        # below so we control n_seq_max and kv_unified ourselves.
        self._llm = Llama(model_path=model_path, n_ctx=8, verbose=False)

        params = llama_cpp.llama_context_default_params()
        params.n_ctx = n_ctx
        params.n_batch = n_ctx
        params.n_ubatch = n_ctx
        params.n_seq_max = n_seq_max
        params.kv_unified = True

        self._ctx = LlamaContext(model=self._llm._model, params=params, verbose=False)
        self._n_vocab = self._llm._model.n_vocab()
        self._batch_capacity = int(params.n_batch)
        self._batch = llama_cpp.llama_batch_init(self._batch_capacity, 0, n_seq_max)
        self.eos_token = self._llm.token_eos()
        self._closed = False

    def close(self) -> None:
        """Free the batch, context and model, in that order.

        This is not tidiness. A Runtime owns two objects with overlapping
        lifetimes -- self._llm (which owns the model weights) and self._ctx (a
        context built over self._llm._model) -- and Python guarantees no
        finalization order between them. When the Llama is collected first, the
        LlamaContext destructor then calls llama_free() on a context whose model
        has already been freed, which faults rather than raising: an access
        violation out of __del__ during interpreter teardown, at a point where
        the traceback no longer names anything useful.

        It surfaces when a process builds more than one Runtime -- a test
        session, mainly, since the server builds exactly one and leaks it into
        exit. That makes it easy to dismiss as a test-only artefact; it is not.
        It is a use-after-free that happens to be latency-hidden by the usual
        lifecycle. The llama_batch was also never freed at all.

        Idempotent, so a caller may close early and still use `with`.
        """
        if self._closed:
            return
        self._closed = True
        llama_cpp.llama_batch_free(self._batch)
        # LlamaContext.close is unannotated upstream; the call is correct.
        self._ctx.close()  # type: ignore[no-untyped-call]
        self._llm.close()

    def __enter__(self) -> "LlamaRuntime":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    @property
    def batch_capacity(self) -> int:
        return self._batch_capacity

    def tokenize(self, prompt: str) -> list[int]:
        result: list[int] = self._llm.tokenize(prompt.encode("utf-8"), add_bos=True)
        return result

    def detokenize(self, tokens: list[int]) -> bytes:
        result: bytes = self._llm.detokenize(tokens)
        return result

    def free_sequence(self, seq_id: int) -> None:
        """Release this sequence's KV cells so llama.cpp can reuse them."""
        self._ctx.kv_cache_seq_rm(seq_id, -1, -1)

    def reuse_prefix(self, src_seq_id: int, dst_seq_id: int, n_tokens: int) -> bool:
        """Copy positions [0, n_tokens) of `src_seq_id` onto `dst_seq_id`.

        This is the whole point of the prefix cache: the destination sequence
        starts at n_past = n_tokens without those tokens ever passing through
        llama_decode(). llama.cpp shares the underlying cells rather than
        duplicating them, so the copy is cheap and costs no extra KV budget --
        the block accounting in KVCacheManager charges the destination anyway,
        which is deliberately conservative (see the note at its call site).

        Returns False if the donor no longer holds that range, which is a
        normal race, not an error: the donor can finish and be freed between
        the tree lookup and this call. The caller falls back to a full prefill.
        """
        if n_tokens <= 0 or src_seq_id == dst_seq_id:
            return False
        mem = llama_cpp.llama_get_memory(self._ctx.ctx)
        # Verify the donor still covers the range we were promised. Copying from
        # a freed sequence would silently produce a destination whose n_past
        # claims cells that hold nothing -- garbage output rather than a crash,
        # which is far worse to debug.
        if llama_cpp.llama_memory_seq_pos_max(mem, src_seq_id) < n_tokens - 1:
            return False
        llama_cpp.llama_memory_seq_cp(mem, src_seq_id, dst_seq_id, 0, n_tokens)
        return True

    def perf_snapshot(self) -> dict[str, float]:
        """llama.cpp's own timing counters (llama_perf_context).

        docs/profiling-notes.md flagged this as a scoped-out gap: py-spy can
        only report that 97.2% of wall time is inside llama_decode(), not how
        that splits between prompt-eval and token-eval. This reads the split
        straight from the library, and n_reused is llama.cpp's independent
        count of reused KV tokens -- the cross-check on the prefix cache's own
        prefill_tokens_reused_total.
        """
        data = llama_cpp.llama_perf_context(self._ctx.ctx)
        return {
            "t_p_eval_ms": float(data.t_p_eval_ms),
            "t_eval_ms": float(data.t_eval_ms),
            "n_p_eval": float(data.n_p_eval),
            "n_eval": float(data.n_eval),
            "n_reused": float(data.n_reused),
        }

    async def decode(self, batch: PreparedBatch) -> dict[int, int]:
        """Run one llama_decode() call for `batch`, off the event loop.

        Returns the greedily-sampled next token for every seq_id whose row
        was flagged needs_logits — i.e. every sequence that advanced this tick.
        """
        return await asyncio.to_thread(self._decode_sync, batch)

    def _decode_sync(self, batch: PreparedBatch) -> dict[int, int]:
        rows = batch.rows
        self._batch.n_tokens = len(rows)
        for i, row in enumerate(rows):
            self._batch.token[i] = row.token
            self._batch.pos[i] = row.pos
            self._batch.n_seq_id[i] = 1
            self._batch.seq_id[i][0] = row.seq_id
            self._batch.logits[i] = row.needs_logits

        return_code = llama_cpp.llama_decode(self._ctx.ctx, self._batch)
        if return_code != 0:
            raise RuntimeError(f"llama_decode failed with return code {return_code}")

        sampled: dict[int, int] = {}
        for i, row in enumerate(rows):
            if row.needs_logits:
                sampled[row.seq_id] = self._sample_greedy(i)
        return sampled

    def _sample_greedy(self, batch_row_index: int) -> int:
        logits_ptr = self._ctx.get_logits_ith(batch_row_index)
        logits = np.ctypeslib.as_array(logits_ptr, shape=(self._n_vocab,))
        return int(np.argmax(logits))
