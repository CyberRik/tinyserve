"""FastAPI app. Phase 3: pluggable scheduling policy, chunked prefill,
queue/generation timeouts on top of Phase 2's continuous batching.

Each tick: expire any waiter that's been queued too long, let the active
SchedulingPolicy choose which waiting requests claim free concurrency
slots, build one batch spanning every active sequence's pending tokens
(chunked prefill included), and run exactly one llama_decode() call for
the whole batch.
"""

import asyncio
import logging
import time
import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import StreamingResponse

from tinyserve.admission.controller import AdmissionController
from tinyserve.api.schemas import GenerateRequest, GenerateResponse
from tinyserve.batch.builder import ActiveSequence, build_batch
from tinyserve.config import Settings
from tinyserve.kv_cache.manager import KVCacheManager
from tinyserve.queue.request_queue import PendingRequest, RequestQueue
from tinyserve.runtime.llama_runtime import LlamaRuntime
from tinyserve.scheduler.base import SchedulingPolicy, create_policy
from tinyserve.streaming.stream_manager import StreamManager

logger = logging.getLogger(__name__)


async def batch_loop(
    queue: RequestQueue,
    runtime: LlamaRuntime,
    streams: StreamManager,
    kv_cache: KVCacheManager,
    policy: SchedulingPolicy,
    n_seq_max: int,
    chunk_size: int,
    queue_timeout_seconds: float,
    generation_timeout_seconds: float,
) -> None:
    active: dict[str, ActiveSequence] = {}  # request_id -> sequence state
    remaining_tokens: dict[str, int] = {}  # request_id -> generations left
    seq_id_of: dict[str, int] = {}  # request_id -> llama seq_id slot
    admitted_at: dict[str, float] = {}  # request_id -> monotonic admission time
    free_slots = list(range(n_seq_max))

    def admit(request: PendingRequest) -> None:
        seq_id = free_slots.pop()
        seq_id_of[request.id] = seq_id
        active[request.id] = ActiveSequence(
            seq_id=seq_id, pending_tokens=request.prompt_tokens, n_past=0
        )
        remaining_tokens[request.id] = request.max_tokens
        admitted_at[request.id] = time.monotonic()

    def finish(request_id: str) -> None:
        seq_id = seq_id_of.pop(request_id)
        runtime.free_sequence(seq_id)
        free_slots.append(seq_id)
        active.pop(request_id)
        remaining_tokens.pop(request_id)
        admitted_at.pop(request_id)
        kv_cache.release(request_id)
        streams.close(request_id)

    def expire_stale_waiters() -> None:
        now = time.monotonic()
        for request in queue.waiting():
            if now - request.arrival_ts > queue_timeout_seconds:
                queue.remove(request.id)
                kv_cache.release(request.id)
                streams.close(request.id)

    while True:
        expire_stale_waiters()
        if not active and queue.depth() == 0:
            await queue.wait_until_nonempty()
            expire_stale_waiters()

        if free_slots:
            for request in policy.select(queue.waiting(), capacity=len(free_slots)):
                queue.remove(request.id)
                admit(request)

        prepared = build_batch(list(active.values()), runtime.batch_capacity, chunk_size)
        if len(prepared) == 0:
            # Every active sequence's next slice is too large to fit this
            # tick's capacity — yield and retry rather than busy-spin.
            await asyncio.sleep(0)
            continue

        try:
            sampled = await runtime.decode(prepared)
        except Exception:
            logger.exception("batch decode failed for %d active sequences", len(active))
            for request_id in list(seq_id_of):
                finish(request_id)
            continue

        request_id_of = {seq.seq_id: request_id for request_id, seq in active.items()}
        for seq_id in {row.seq_id for row in prepared.rows}:
            seq = active[request_id_of[seq_id]]
            consumed = prepared.row_count(seq_id)
            seq.n_past += consumed
            seq.pending_tokens = seq.pending_tokens[consumed:]

        now = time.monotonic()
        for seq_id, token in sampled.items():
            request_id = request_id_of[seq_id]
            seq = active[request_id]
            remaining_tokens[request_id] -= 1

            is_done = (
                token == runtime.eos_token
                or remaining_tokens[request_id] <= 0
                or streams.is_cancelled(request_id)
                or (now - admitted_at[request_id]) > generation_timeout_seconds
            )
            if is_done:
                finish(request_id)
            else:
                streams.push_token(request_id, runtime.detokenize([token]))
                seq.pending_tokens = [token]


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    settings = Settings()  # type: ignore[call-arg]
    runtime = LlamaRuntime(
        model_path=settings.model_path, n_ctx=settings.n_ctx, n_seq_max=settings.n_seq_max
    )
    kv_cache = KVCacheManager(total_tokens=settings.n_ctx, block_size=settings.kv_block_size)
    policy = create_policy(settings.scheduling_policy)

    app.state.runtime = runtime
    app.state.queue = RequestQueue()
    app.state.streams = StreamManager()
    app.state.admission = AdmissionController(kv_cache)
    app.state.batch_loop_task = asyncio.create_task(
        batch_loop(
            app.state.queue,
            runtime,
            app.state.streams,
            kv_cache,
            policy,
            settings.n_seq_max,
            settings.chunk_size,
            settings.queue_timeout_seconds,
            settings.generation_timeout_seconds,
        )
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
    runtime: LlamaRuntime = http_request.app.state.runtime
    admission: AdmissionController = http_request.app.state.admission
    streams: StreamManager = http_request.app.state.streams
    queue: RequestQueue = http_request.app.state.queue

    request_id = str(uuid.uuid4())
    prompt_tokens = runtime.tokenize(body.prompt)

    result = admission.admit(
        request_id, prompt_tokens=len(prompt_tokens), max_tokens=body.max_tokens
    )
    if not result.accepted:
        raise HTTPException(
            status_code=503, detail={"reason": result.reason}, headers={"Retry-After": "1"}
        )

    streams.create(request_id)
    queue.push(
        PendingRequest(
            id=request_id,
            prompt_tokens=prompt_tokens,
            max_tokens=body.max_tokens,
            priority=body.priority,
        )
    )

    if body.stream:
        return StreamingResponse(
            _sse_events(request_id, streams, http_request), media_type="text/event-stream"
        )

    chunks = [text async for text in streams.subscribe(request_id)]
    return GenerateResponse(text="".join(chunks))
