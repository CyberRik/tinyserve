"""The only module allowed to import llama.cpp bindings (PRD Section 13/14).

Phase 1: token-by-token streaming against one in-flight request at a time —
no batching across concurrent requests yet (that's Phase 2's KV Cache
Manager + real batch loop). Everything here still runs one call at a time
off the event loop, via a producer thread bridged through an asyncio.Queue.
"""

import asyncio
import threading
from collections.abc import AsyncIterator, Callable

from llama_cpp import Llama


class LlamaRuntime:
    """Loads one GGUF model and runs completions against it."""

    def __init__(self, model_path: str, n_ctx: int = 2048) -> None:
        self._llm = Llama(model_path=model_path, n_ctx=n_ctx, verbose=False)

    async def stream(
        self,
        prompt: str,
        max_tokens: int,
        is_cancelled: Callable[[], bool],
    ) -> AsyncIterator[bytes]:
        """Yield raw detokenized bytes for each new token as it's sampled.

        Bytes, not str: a single Unicode codepoint can span multiple tokens,
        so UTF-8 reassembly is the caller's job (Stream Manager), not this
        binding's — this module only ever hands back what llama.cpp produced.
        """
        loop = asyncio.get_running_loop()
        out_queue: asyncio.Queue[bytes | None] = asyncio.Queue()

        thread = threading.Thread(
            target=self._stream_worker,
            args=(prompt, max_tokens, is_cancelled, loop, out_queue),
            daemon=True,
        )
        thread.start()

        while True:
            chunk = await out_queue.get()
            if chunk is None:
                break
            yield chunk

    def _stream_worker(
        self,
        prompt: str,
        max_tokens: int,
        is_cancelled: Callable[[], bool],
        loop: asyncio.AbstractEventLoop,
        out_queue: "asyncio.Queue[bytes | None]",
    ) -> None:
        try:
            prompt_tokens = self._llm.tokenize(prompt.encode("utf-8"))
            all_tokens = list(prompt_tokens)
            eos_token = self._llm.token_eos()

            for count, token in enumerate(self._llm.generate(prompt_tokens), start=1):
                if count > max_tokens or token == eos_token or is_cancelled():
                    break
                token_bytes = self._llm.detokenize([token], prev_tokens=all_tokens)
                all_tokens.append(token)
                loop.call_soon_threadsafe(out_queue.put_nowait, token_bytes)
        finally:
            loop.call_soon_threadsafe(out_queue.put_nowait, None)
