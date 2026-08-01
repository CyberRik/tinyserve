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
