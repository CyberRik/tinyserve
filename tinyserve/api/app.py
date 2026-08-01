"""FastAPI app. Phase 0: a single /generate endpoint, no streaming, no batching."""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI

from tinyserve.api.schemas import GenerateRequest, GenerateResponse
from tinyserve.config import Settings
from tinyserve.runtime.llama_runtime import LlamaRuntime


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    settings = Settings()  # type: ignore[call-arg]
    app.state.runtime = LlamaRuntime(model_path=settings.model_path, n_ctx=settings.n_ctx)
    yield


app = FastAPI(title="TinyServe", lifespan=lifespan)


@app.get("/healthz")
async def healthz() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/generate")
async def generate(request: GenerateRequest) -> GenerateResponse:
    runtime: LlamaRuntime = app.state.runtime
    result = await runtime.generate(request.prompt, request.max_tokens)
    return GenerateResponse(
        text=result.text,
        prompt_tokens=result.prompt_tokens,
        completion_tokens=result.completion_tokens,
    )
