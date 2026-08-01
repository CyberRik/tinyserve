"""FastAPI app. Phase 4: metrics + tracing on top of Phase 3's scheduling.

Each tick: expire any waiter that's been queued too long, let the active
SchedulingPolicy choose which waiting requests claim free concurrency
slots, build one batch spanning every active sequence's pending tokens
(chunked prefill included), and run exactly one llama_decode() call for
the whole batch. Every one of those decisions also emits a metric or a
trace span — "why was my request slow" should be answerable from
Prometheus/Grafana, not from reading logs (PRD Section 10).
"""

import asyncio
import logging
import time
import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass

from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.responses import StreamingResponse
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

from tinyserve.admission.controller import AdmissionController
from tinyserve.api.schemas import GenerateRequest, GenerateResponse
from tinyserve.batch.builder import ActiveSequence, build_batch
from tinyserve.config import Settings
from tinyserve.kv_cache.manager import KVCacheManager
from tinyserve.observability.metrics import Metrics
from tinyserve.observability.tracing import RequestTracer, configure_tracing
from tinyserve.queue.request_queue import PendingRequest, RequestQueue
from tinyserve.runtime.llama_runtime import LlamaRuntime
from tinyserve.scheduler.base import SchedulingPolicy, create_policy
from tinyserve.streaming.stream_manager import StreamManager

logger = logging.getLogger(__name__)


@dataclass
class _RequestState:
    seq_id: int
    remaining_tokens: int
    arrival_ts: float
    admitted_ts: float
    tokens_emitted: int = 0
    last_token_ts: float | None = None


async def batch_loop(
    queue: RequestQueue,
    runtime: LlamaRuntime,
    streams: StreamManager,
    kv_cache: KVCacheManager,
    policy: SchedulingPolicy,
    metrics: Metrics,
    tracer: RequestTracer,
    n_seq_max: int,
    chunk_size: int,
    queue_timeout_seconds: float,
    generation_timeout_seconds: float,
) -> None:
    active: dict[str, ActiveSequence] = {}  # request_id -> sequence state
    requests: dict[str, _RequestState] = {}  # request_id -> timing/budget state
    free_slots = list(range(n_seq_max))
    policy_name = type(policy).__name__

    def admit(request: PendingRequest) -> None:
        now = time.monotonic()
        seq_id = free_slots.pop()
        active[request.id] = ActiveSequence(
            seq_id=seq_id, pending_tokens=request.prompt_tokens, n_past=0
        )
        requests[request.id] = _RequestState(
            seq_id=seq_id,
            remaining_tokens=request.max_tokens,
            arrival_ts=request.arrival_ts,
            admitted_ts=now,
        )
        metrics.queue_wait_seconds.observe(now - request.arrival_ts)
        metrics.scheduler_policy_decision_total.labels(policy=policy_name, outcome="admitted").inc()
        tracer.end_span(request.id, "queue_wait")
        tracer.start_span(request.id, "generation")

    def finish(request_id: str, *, cancel_reason: str | None) -> None:
        state = requests.pop(request_id)
        runtime.free_sequence(state.seq_id)
        free_slots.append(state.seq_id)
        active.pop(request_id)
        kv_cache.release(request_id)
        streams.close(request_id)
        if cancel_reason is not None:
            metrics.cancellations_total.labels(reason=cancel_reason).inc()
        if state.tokens_emitted > 0:
            elapsed = time.monotonic() - state.admitted_ts
            metrics.tokens_per_second.observe(state.tokens_emitted / elapsed)
        tracer.end_span(request_id, "generation")
        tracer.end_request(request_id, completion_tokens=state.tokens_emitted)

    def expire_stale_waiters() -> None:
        now = time.monotonic()
        for request in queue.waiting():
            if now - request.arrival_ts > queue_timeout_seconds:
                queue.remove(request.id)
                kv_cache.release(request.id)
                streams.close(request.id)
                metrics.cancellations_total.labels(reason="queue_timeout").inc()
                tracer.end_span(request.id, "queue_wait")
                tracer.end_request(request.id, completion_tokens=0)

    while True:
        expire_stale_waiters()
        if not active and queue.depth() == 0:
            await queue.wait_until_nonempty()
            expire_stale_waiters()

        metrics.queue_depth.set(queue.depth())
        metrics.kv_blocks_free.set(kv_cache.available_blocks())
        metrics.kv_blocks_used.set(kv_cache.total_blocks() - kv_cache.available_blocks())

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

        distinct_seq_ids = {row.seq_id for row in prepared.rows}
        decode_started = time.monotonic()
        try:
            sampled = await runtime.decode(prepared)
        except Exception:
            logger.exception("batch decode failed for %d active sequences", len(active))
            for request_id in list(requests):
                finish(request_id, cancel_reason="decode_error")
            continue
        metrics.decode_step_duration_seconds.observe(time.monotonic() - decode_started)
        metrics.batch_size_tokens.observe(len(prepared))
        metrics.batch_size_requests.observe(len(distinct_seq_ids))
        metrics.batch_utilization.observe(len(prepared) / runtime.batch_capacity)

        request_id_of = {state.seq_id: request_id for request_id, state in requests.items()}
        for seq_id in distinct_seq_ids:
            seq = active[request_id_of[seq_id]]
            consumed = prepared.row_count(seq_id)
            seq.n_past += consumed
            seq.pending_tokens = seq.pending_tokens[consumed:]

        now = time.monotonic()
        for seq_id, token in sampled.items():
            request_id = request_id_of[seq_id]
            seq = active[request_id]
            state = requests[request_id]
            state.remaining_tokens -= 1

            if token == runtime.eos_token or state.remaining_tokens <= 0:
                finish(request_id, cancel_reason=None)
            elif streams.is_cancelled(request_id):
                finish(request_id, cancel_reason="client_disconnect")
            elif (now - state.admitted_ts) > generation_timeout_seconds:
                finish(request_id, cancel_reason="generation_timeout")
            else:
                if state.tokens_emitted == 0:
                    metrics.ttft_seconds.observe(now - state.arrival_ts)
                elif state.last_token_ts is not None:
                    metrics.inter_token_latency_seconds.observe(now - state.last_token_ts)
                state.tokens_emitted += 1
                state.last_token_ts = now
                streams.push_token(request_id, runtime.detokenize([token]))
                seq.pending_tokens = [token]


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    settings = Settings()  # type: ignore[call-arg]
    tracer_provider = configure_tracing()
    runtime = LlamaRuntime(
        model_path=settings.model_path, n_ctx=settings.n_ctx, n_seq_max=settings.n_seq_max
    )
    kv_cache = KVCacheManager(total_tokens=settings.n_ctx, block_size=settings.kv_block_size)
    policy = create_policy(settings.scheduling_policy)
    metrics = Metrics()
    tracer = RequestTracer()

    app.state.runtime = runtime
    app.state.queue = RequestQueue()
    app.state.streams = StreamManager()
    app.state.admission = AdmissionController(kv_cache)
    app.state.metrics = metrics
    app.state.tracer = tracer
    app.state.batch_loop_task = asyncio.create_task(
        batch_loop(
            app.state.queue,
            runtime,
            app.state.streams,
            kv_cache,
            policy,
            metrics,
            tracer,
            settings.n_seq_max,
            settings.chunk_size,
            settings.queue_timeout_seconds,
            settings.generation_timeout_seconds,
        )
    )
    yield
    app.state.batch_loop_task.cancel()
    tracer_provider.shutdown()


app = FastAPI(title="TinyServe", lifespan=lifespan)


@app.get("/healthz")
async def healthz() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/metrics")
async def metrics_endpoint(http_request: Request) -> Response:
    metrics: Metrics = http_request.app.state.metrics
    return Response(generate_latest(metrics.registry), media_type=CONTENT_TYPE_LATEST)


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
    metrics: Metrics = http_request.app.state.metrics
    tracer: RequestTracer = http_request.app.state.tracer

    request_id = str(uuid.uuid4())
    prompt_tokens = runtime.tokenize(body.prompt)
    tracer.start_request(request_id, prompt_tokens=len(prompt_tokens), max_tokens=body.max_tokens)

    with tracer.span(request_id, "admission"):
        result = admission.admit(
            request_id, prompt_tokens=len(prompt_tokens), max_tokens=body.max_tokens
        )
    if not result.accepted:
        metrics.admission_rejected_total.labels(reason=result.reason).inc()
        tracer.end_request(request_id, rejected=True)
        raise HTTPException(
            status_code=503, detail={"reason": result.reason}, headers={"Retry-After": "1"}
        )
    metrics.admission_accepted_total.inc()

    streams.create(request_id)
    tracer.start_span(request_id, "queue_wait")
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
