"""The only module allowed to import llama.cpp bindings (PRD Section 13/14).

Phase 0: a thin, naive wrapper — one request in, one completion out, no
batching or streaming. Later phases replace the internals (real llama_decode
batch loop, KV cache manager) behind the same async surface.
"""

import asyncio
from dataclasses import dataclass

from llama_cpp import Llama


@dataclass(frozen=True)
class GenerationResult:
    text: str
    prompt_tokens: int
    completion_tokens: int


class LlamaRuntime:
    """Loads one GGUF model and runs completions against it."""

    def __init__(self, model_path: str, n_ctx: int = 2048) -> None:
        self._llm = Llama(model_path=model_path, n_ctx=n_ctx, verbose=False)

    async def generate(self, prompt: str, max_tokens: int) -> GenerationResult:
        """Run a full (non-streaming) completion in a worker thread.

        llama.cpp's decode call is blocking C code, so it runs via
        asyncio.to_thread rather than on the event loop directly.
        """
        return await asyncio.to_thread(self._generate_sync, prompt, max_tokens)

    def _generate_sync(self, prompt: str, max_tokens: int) -> GenerationResult:
        prompt_tokens = self._llm.tokenize(prompt.encode("utf-8"))
        output = self._llm.create_completion(prompt, max_tokens=max_tokens, stream=False)
        assert isinstance(output, dict)  # stream=False rules out the Iterator branch
        text = output["choices"][0]["text"]
        completion_tokens = output["usage"]["completion_tokens"]
        return GenerationResult(
            text=text,
            prompt_tokens=len(prompt_tokens),
            completion_tokens=completion_tokens,
        )
