"""FastAPI app. Phase 1: FIFO queue + single-consumer batch loop + SSE streaming.

The batch loop still runs one request fully before starting the next (real
continuous batching is Phase 2) but is now the sole owner of the Runtime,
so HTTP handlers only ever enqueue and read from the Stream Manager — they
never call the Runtime directly.
"""

import asyncio
import functools
import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse

from tinyserve.api.schemas import GenerateRequest, GenerateResponse
from tinyserve.config import Settings
from tinyserve.queue.request_queue import PendingRequest, RequestQueue
from tinyserve.runtime.llama_runtime import LlamaRuntime
from tinyserve.streaming.stream_manager import StreamManager


async def batch_loop(queue: RequestQueue, runtime: LlamaRuntime, streams: StreamManager) -> None:
    """Pull one request at a time and run it to completion via the Runtime."""
    while True:
        request = await queue.pop()
        async for token_bytes in runtime.stream(
            request.prompt,
            request.max_tokens,
            is_cancelled=functools.partial(streams.is_cancelled, request.id),
        ):
            streams.push_token(request.id, token_bytes)
        streams.close(request.id)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    settings = Settings()  # type: ignore[call-arg]
    app.state.runtime = LlamaRuntime(model_path=settings.model_path, n_ctx=settings.n_ctx)
    app.state.queue = RequestQueue()
    app.state.streams = StreamManager()
    app.state.batch_loop_task = asyncio.create_task(
        batch_loop(app.state.queue, app.state.runtime, app.state.streams)
    )
    yield
    app.state.batch_loop_task.cancel()


app = FastAPI(title="TinyServe", lifespan=lifespan)


@app.get("/healthz")
async def healthz() -> dict[str, str]:
    return {"status": "ok"}


async def _sse_events(
    request_id: str, streams: StreamManager, http_request: Request
) -> AsyncIterator[str]:
    async for text in streams.subscribe(request_id):
        if await http_request.is_disconnected():
            streams.cancel(request_id)
            return
        yield f"data: {text}\n\n"


@app.post("/generate", response_model=None)
async def generate(
    body: GenerateRequest, http_request: Request
) -> GenerateResponse | StreamingResponse:
    request_id = str(uuid.uuid4())
    streams: StreamManager = http_request.app.state.streams
    queue: RequestQueue = http_request.app.state.queue

    streams.create(request_id)
    queue.push(PendingRequest(id=request_id, prompt=body.prompt, max_tokens=body.max_tokens))

    if body.stream:
        return StreamingResponse(
            _sse_events(request_id, streams, http_request), media_type="text/event-stream"
        )

    chunks = [text async for text in streams.subscribe(request_id)]
    return GenerateResponse(text="".join(chunks))
