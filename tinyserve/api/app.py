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
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.responses import FileResponse, StreamingResponse
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

from tinyserve.admission.controller import AdmissionController
from tinyserve.api.schemas import (
    ConfigResponse,
    GenerateRequest,
    GenerateResponse,
    PolicyRequest,
)
from tinyserve.batch.builder import ActiveSequence, build_batch
from tinyserve.config import Settings
from tinyserve.kv_cache.manager import KVCacheManager
from tinyserve.observability.metrics import Metrics
from tinyserve.observability.tracing import RequestTracer, configure_tracing
from tinyserve.prefix.cache import PrefixCacheBase, create_prefix_cache
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


@dataclass
class ActivePolicy:
    """The scheduling policy in force, behind one level of indirection.

    The batch loop reads this every tick instead of closing over a policy
    object, so POST /config/policy can swap the policy on a running server.
    That exists so the FIFO-vs-WFQ comparison can be driven from one server
    instead of two on different ports.
    Swapping only affects requests admitted after the swap -- sequences
    already holding a slot run to completion under the old policy.
    """

    policy: SchedulingPolicy
    name: str

    def set(self, name: str) -> None:
        self.policy = create_policy(name)
        self.name = name


async def batch_loop(
    queue: RequestQueue,
    runtime: LlamaRuntime,
    streams: StreamManager,
    kv_cache: KVCacheManager,
    active_policy: ActivePolicy,
    prefix_cache: PrefixCacheBase | None,
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

    def _claim_prefix(request: PendingRequest, seq_id: int) -> int:
        """Reuse another sequence's already-prefilled KV cells, if any overlap.

        Returns how many of this prompt's leading tokens are now resident in
        seq_id's cache and must therefore be skipped when building batches.
        Zero means "prefill the whole prompt", i.e. exactly the old behaviour --
        every failure path below falls back to it rather than to an error.
        """
        if prefix_cache is None:
            return 0

        hit = prefix_cache.match_for_reuse(request.prompt_tokens)
        reused = 0
        if hit and runtime.reuse_prefix(hit.seq_id, seq_id, hit.n_tokens):
            reused = hit.n_tokens
            metrics.prefill_tokens_reused_total.inc(reused)
            metrics.prefix_cache_lookups_total.labels(outcome="hit").inc()
        elif hit:
            # The donor was freed between the lookup and the copy. A normal
            # race under churn, not an error -- but worth its own label, since
            # a high stale rate means sequences are finishing faster than the
            # tree is being evicted and points at a real bug.
            metrics.prefix_cache_lookups_total.labels(outcome="stale_donor").inc()
        else:
            metrics.prefix_cache_lookups_total.labels(outcome="miss").inc()

        # Publish after matching, never before: a sequence must not be offered
        # its own cells as a donor. Publishing at admission rather than at
        # completion is deliberate -- these cells are committed the moment this
        # sequence's prefill is scheduled, so later arrivals in the same burst
        # can share them instead of every one of them prefilling in parallel.
        prefix_cache.insert(request.prompt_tokens, seq_id)
        metrics.prefix_cache_nodes.set(prefix_cache.node_count())
        return reused

    def admit(request: PendingRequest) -> None:
        now = time.monotonic()
        seq_id = free_slots.pop()
        n_past = _claim_prefix(request, seq_id)
        active[request.id] = ActiveSequence(
            seq_id=seq_id, pending_tokens=request.prompt_tokens[n_past:], n_past=n_past
        )
        requests[request.id] = _RequestState(
            seq_id=seq_id,
            remaining_tokens=request.max_tokens,
            arrival_ts=request.arrival_ts,
            admitted_ts=now,
        )
        metrics.prefill_tokens_total.inc(len(request.prompt_tokens))
        metrics.queue_wait_seconds.observe(now - request.arrival_ts)
        metrics.scheduler_policy_decision_total.labels(
            policy=active_policy.name, outcome="admitted"
        ).inc()
        tracer.end_span(request.id, "queue_wait")
        tracer.start_span(request.id, "generation")

    def finish(request_id: str, *, cancel_reason: str | None) -> None:
        state = requests.pop(request_id)
        # Evict before the slot returns to the pool. If a later request claimed
        # this seq_id while the tree still advertised the old prompt under it,
        # a match would copy from cells that had been freed and refilled with
        # something else -- silently wrong output, not a crash.
        if prefix_cache is not None:
            prefix_cache.evict(state.seq_id)
            metrics.prefix_cache_nodes.set(prefix_cache.node_count())
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
            for request in active_policy.policy.select(queue.waiting(), capacity=len(free_slots)):
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
    active_policy = ActivePolicy(
        policy=create_policy(settings.scheduling_policy), name=settings.scheduling_policy
    )
    prefix_cache: PrefixCacheBase | None = None
    if settings.prefix_cache_enabled:
        prefix_cache = create_prefix_cache(settings.kv_block_size)
    metrics = Metrics()
    metrics.prefix_cache_backend_info.labels(
        backend=prefix_cache.backend if prefix_cache else "disabled"
    ).set(1)
    tracer = RequestTracer()

    app.state.runtime = runtime
    app.state.queue = RequestQueue()
    app.state.streams = StreamManager()
    app.state.admission = AdmissionController(kv_cache, max_queue_depth=settings.max_queue_depth)
    app.state.metrics = metrics
    app.state.tracer = tracer
    app.state.active_policy = active_policy
    app.state.settings = settings
    app.state.batch_loop_task = asyncio.create_task(
        batch_loop(
            app.state.queue,
            runtime,
            app.state.streams,
            kv_cache,
            active_policy,
            prefix_cache,
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


_DEMO_PAGE = Path(__file__).parent / "static" / "demo.html"


@app.get("/healthz")
async def healthz() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/config")
async def get_config(http_request: Request) -> ConfigResponse:
    """The server's live configuration, so the demo page can display real
    values instead of restating defaults that may not match how it was
    started."""
    settings: Settings = http_request.app.state.settings
    active_policy: ActivePolicy = http_request.app.state.active_policy
    return ConfigResponse(
        scheduling_policy=active_policy.name,
        n_seq_max=settings.n_seq_max,
        n_ctx=settings.n_ctx,
        kv_block_size=settings.kv_block_size,
        max_queue_depth=settings.max_queue_depth,
        prefix_cache_enabled=settings.prefix_cache_enabled,
        model_path=settings.model_path,
    )


@app.post("/config/policy")
async def set_policy(body: PolicyRequest, http_request: Request) -> ConfigResponse:
    """Hot-swap the scheduling policy.

    Deliberately unauthenticated, like every other route here: this runtime
    binds to localhost and is a teaching/demo artifact, not a deployed
    service (see PRD Section 3 non-goals). Do not expose it publicly.
    """
    active_policy: ActivePolicy = http_request.app.state.active_policy
    try:
        active_policy.set(body.policy)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail={"reason": str(exc)}) from None
    return await get_config(http_request)


@app.get("/demo", include_in_schema=False)
async def demo_page() -> FileResponse:
    """Static demo client.

    Served from the app rather than opened as a file:// page purely so it
    shares an origin with /generate -- the alternative was adding permissive
    CORS to the real server for a demo's benefit. Adds no runtime behaviour;
    it drives the same public endpoint curl does.
    """
    # no-store: the page changes while demos are being iterated on, and a cached
    # copy is indistinguishable on screen from the current one until it runs.
    return FileResponse(_DEMO_PAGE, media_type="text/html", headers={"Cache-Control": "no-store"})


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
            request_id,
            prompt_tokens=len(prompt_tokens),
            max_tokens=body.max_tokens,
            queue_depth=queue.depth(),
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
