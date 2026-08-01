# TinyServe — A From-Scratch Inference Runtime

**Status:** Design doc (pre-implementation)
**Author:** [you]
**Companion project:** Ancora (durable workflow runtime, Temporal + Ray)

---

## 0. Relationship to Ancora — read this first

This is the question an interviewer will ask in the first two minutes, so it's answered up front rather than buried.

Ancora and TinyServe look similar on paper — both have queues, both have admission control, both borrow fairness ideas — but they solve problems that live at different points on two axes:

| | Ancora | TinyServe |
|---|---|---|
| Unit of work | workflow step (an "activity") | one inference request (a prompt → token stream) |
| Timescale | seconds to days | milliseconds to seconds |
| Failure model | must survive process/host death; state is durable (Temporal event history) | a crashed request is simply retried by the client; runtime state is disposable |
| Resource being scheduled | worker capacity across a fleet (Ray tasks) | KV-cache memory + GPU/CPU compute *within one process* |
| Correctness concern | exactly-once side effects, idempotency across retries | correct token ordering within a stream, no cache corruption across concurrent requests |
| Scheduling granularity | whole tasks, dispatched to workers | sub-request granularity — chunks of prefill, one decode step per batch tick |

The reused vocabulary (WFQ, admission control, backpressure) is intentional, not incidental: it's the same engineer demonstrating that these are general scheduling primitives, applied correctly to two problems with opposite constraints. Ancora's admission engine rejects work it can't guarantee durable capacity for; TinyServe's admission controller rejects requests it can't guarantee KV-cache memory for. Same shape, different substrate. That's the story to tell in an interview — not "I built two schedulers," but "I understand what changes about a scheduler when durability disappears and the clock speeds up 1000x."

TinyServe has zero dependency on Ancora and doesn't call into it. They stay separate repos.

---

## 1. Vision

Build a small, fully-understood inference server that does what vLLM/SGLang do — continuous batching, KV-cache management, admission control, fair scheduling — but at a scale and clarity where every design decision can be explained on a whiteboard. llama.cpp does the actual matrix multiplication; everything above that line (what runs next, whose tokens go in this batch, when to evict cache, how backpressure propagates to the client) is built from scratch.

The deliverable is not "faster than vLLM." The deliverable is: a runtime where you can point to any latency number in a benchmark and explain, in terms of your own code, exactly which scheduling decision produced it.

## 2. Goals

- Implement continuous batching and chunked prefill against a llama.cpp backend, from scratch, in Python/AsyncIO.
- Implement a simplified block-based KV-cache manager with allocation, reuse, and eviction — not paged attention, but the same *idea* at a scale you can reason about by hand.
- Implement admission control and a pluggable scheduling policy (FIFO / priority / WFQ), reusing the fairness reasoning from Ancora but applied to token budgets instead of workflow budgets.
- Support token streaming (SSE) with correct mid-stream cancellation and timeout propagation all the way down to the batch loop.
- First-class observability: every scheduling decision emits a metric or a trace span. "Why was my request slow" should always be answerable from the metrics, not from reading logs by hand.
- Ship a benchmark suite that produces real latency/throughput/fairness numbers, and use profiling tools to explain *why* those numbers look the way they do.
- Every phase (below) should be independently runnable and demoable — no six-week silent rewrite.

## 3. Non-Goals

Stated explicitly so scope doesn't creep, and so you can say "intentionally out of scope" in an interview instead of "I ran out of time":

- **Not writing a transformer, an attention kernel, or a tokenizer.** llama.cpp/GGUF owns model execution end to end.
- **Not CUDA/Triton kernel work.** GPU execution is llama.cpp's problem; TinyServe only decides *what* to run, not *how* to run a matmul.
- **Not distributed/multi-node serving.** Single process, single model, single machine. Multi-replica load balancing is a different (and less interesting) problem — out of scope.
- **Not full PagedAttention.** A real paged KV-cache with non-contiguous physical blocks requires kernel-level support llama.cpp doesn't expose the same way vLLM's custom CUDA kernels do. TinyServe implements block-based *logical* allocation and documents exactly where it would need to hook into paged attention if llama.cpp ever exposed that primitive.
- **Not multi-model / model-swapping serving.** One model per running instance.
- **Not a training system.** Inference only.
- **Not trying to beat vLLM/SGLang/TensorRT-LLM on throughput.** They have teams and custom CUDA kernels. This project's benchmark story is "here is the latency breakdown and here is why," not "here is a leaderboard win."

## 4. Core Architecture

Single Python process (monolith, as requested — no microservices, no Kubernetes):

```
                         ┌─────────────────────────────────────────┐
                         │              FastAPI app                 │
                         │   POST /generate   GET /stream/{id}      │
                         │   GET /metrics      GET /healthz          │
                         └───────────────────┬───────────────────────┘
                                             │
                         ┌───────────────────▼───────────────────────┐
                         │           Admission Controller             │
                         │  (KV-budget check, queue-depth check,      │
                         │   rate limit, reject-fast on overload)     │
                         └───────────────────┬───────────────────────┘
                                             │ accepted request
                         ┌───────────────────▼───────────────────────┐
                         │              Request Queue                 │
                         │   (per-priority sub-queues, WFQ state)      │
                         └───────────────────┬───────────────────────┘
                                             │
                         ┌───────────────────▼───────────────────────┐
                         │                Scheduler                    │
                         │  runs every batch tick:                     │
                         │  pick next requests → hand to Batch Builder │
                         └───────────────────┬───────────────────────┘
                                             │
                         ┌───────────────────▼───────────────────────┐
                         │              Batch Builder                  │
                         │  merges prefill chunks + decode steps       │
                         │  into one llama.cpp batch call              │
                         └───────────────────┬───────────────────────┘
                                             │
                         ┌───────────────────▼───────────────────────┐
                         │           KV Cache Manager                  │
                         │  allocates/frees blocks per sequence        │
                         └───────────────────┬───────────────────────┘
                                             │
                         ┌───────────────────▼───────────────────────┐
                         │        Model Runtime (llama.cpp binding)    │
                         │   llama_decode() — the only place that      │
                         │   actually touches the GPU/CPU               │
                         └───────────────────┬───────────────────────┘
                                             │ logits
                         ┌───────────────────▼───────────────────────┐
                         │              Stream Manager                 │
                         │  detokenize, push to per-request SSE queue  │
                         └───────────────────┬───────────────────────┘
                                             │
                         ┌───────────────────▼───────────────────────┐
                         │              Metrics / Tracing              │
                         │      (cross-cuts every component above)     │
                         └─────────────────────────────────────────────┘
```

One background asyncio task runs the **batch loop** — a tight `while True` loop that: asks the Scheduler for the next batch → hands it to the Batch Builder → calls the Runtime → distributes results via the Stream Manager → repeats. Everything else (HTTP handlers, admission control) is producer-side; the batch loop is the single consumer that owns GPU access, which sidesteps needing locks around llama.cpp state.

## 5. Request Lifecycle — `POST /generate`

Walking through exactly what happens, stage by stage:

1. **HTTP layer.** FastAPI receives `POST /generate` with `{prompt, max_tokens, priority, stream: bool}`. A `request_id` (UUID) is generated immediately and attached to a trace span (`request.received`).

2. **Admission Controller.** Before anything touches the queue:
   - Estimate KV-cache blocks needed for `len(prompt_tokens) + max_tokens` (worst case).
   - Check against `available_blocks` tracked by the KV Cache Manager.
   - Check current queue depth against a configured max (backpressure — reject fast rather than let latency degrade unboundedly for everyone).
   - If rejected: return `503` immediately with `Retry-After`, emit `admission.rejected{reason}` metric. **This mirrors Ancora's admission engine rejecting work it can't durably guarantee — here it's rejecting work it can't fit in KV-cache memory.**
   - If accepted: `admission.accepted` span closes, request moves to the queue.

3. **Tokenization.** Prompt is tokenized (llama.cpp's tokenizer, called off the batch loop so it doesn't block GPU work). Token count now known exactly — KV reservation is corrected from estimate to exact.

4. **Enqueue.** Request object `{id, tokens, state=PENDING_PREFILL, priority, arrival_time, deadline?}` is pushed into the Request Queue's appropriate priority sub-queue. A `Future`/`asyncio.Queue` is created for this request's outgoing tokens — this is what the HTTP handler will await on for streaming.

5. **HTTP handler blocks on the stream.** For `stream=true`, the handler immediately starts an SSE response and begins consuming from the per-request output queue as tokens arrive. For `stream=false`, it awaits full completion.

6. **Scheduler tick (runs independently, on a timer or event-driven).** Each tick:
   - Ask the KV Cache Manager how many free blocks exist.
   - Pull requests from the queue according to the active policy (Section 7) — some mix of new prefills (possibly chunked) and existing sequences due for their next decode step.
   - Requests that can't get KV blocks this tick stay queued (this is where `queue.wait_time` accumulates — a key fairness/latency metric).

7. **Batch Builder.** Combines: (a) decode steps for sequences already in-flight (1 new token each), and (b) one or more chunked-prefill slices for new/partially-prefilled requests, into a single call shape llama.cpp's batch API accepts (`llama_batch`).

8. **KV Cache Manager** allocates any newly-needed blocks, updates each sequence's block table, and provides the position offsets the Runtime needs.

9. **Runtime executes.** One call to `llama_decode()` for the whole batch. This is the only GPU/CPU-bound step; everything else in this list is Python/AsyncIO overhead the design is trying to keep off the critical path.

10. **Result distribution.** Logits come back per-sequence. For sequences that just finished prefill, nothing is streamed yet (or the first token is, if prefill fully completed). For sequences mid-decode, sample the next token, detokenize incrementally, push onto that request's output queue.

11. **Stream Manager → client.** The SSE generator wakes up, writes `data: {token}\n\n`, flushes.

12. **Completion or cancellation.**
    - Normal: EOS token or `max_tokens` reached → free KV blocks, close the stream, emit final metrics (`ttft`, `total_latency`, `tokens_per_sec`).
    - Client disconnect: FastAPI's `request.is_disconnected()` (polled or via `asyncio.shield`) triggers cancellation — sequence marked `CANCELLED`, removed from the scheduler's active set on the *next* tick, KV blocks freed. (Full cancellation semantics in Section 9.)

13. **Metrics/tracing close out.** The whole lifecycle is one trace with child spans: `admission` → `queue.wait` → `prefill` (possibly multiple chunked spans) → `decode.step` (one per token, or aggregated) → `stream.flush`.

## 6. Runtime Components

### 6.1 Admission Controller
- **Responsibility:** gatekeeper — decide accept/reject before a request consumes any queue or memory resources.
- **State:** current KV block usage, configured max queue depth, a token-bucket rate limiter (reusable pattern — you already built one of these for OpenRouter in Gravton).
- **API:** `async def admit(request: PendingRequest) -> AdmissionResult` (accept / reject-with-reason).
- **Interacts with:** KV Cache Manager (read available blocks), Request Queue (read depth), Metrics.

### 6.2 Request Queue
- **Responsibility:** hold accepted-but-not-yet-scheduled requests; expose them in priority/fairness order.
- **State:** one `asyncio.PriorityQueue`-like structure per priority class, plus WFQ virtual-time bookkeeping (per Ancora's `fairness.py` pattern — virtual finish time per class, newcomer adopts current minimum).
- **Data structures:** `PendingRequest{id, tokens, priority, arrival_ts, virtual_finish_time}`.
- **API:** `push(request)`, `pop_batch(capacity: BatchBudget) -> list[PendingRequest]`.

### 6.3 Scheduler
- **Responsibility:** decide, every tick, *which* requests (new prefill chunks + in-flight decodes) go into the next batch, respecting the active policy and available KV budget.
- **State:** the policy object (FIFO/Priority/WFQ — pluggable, Section 7), a reference to in-flight sequences (`SequenceTable`).
- **API:** `async def tick() -> Batch`.
- **Interacts with:** Request Queue (pull), KV Cache Manager (query capacity), Batch Builder (hand off), Metrics (emit `scheduler.decision` events — this is what makes fairness measurable rather than asserted).

### 6.4 Batch Builder
- **Responsibility:** turn a scheduler decision into the exact tensor/token layout llama.cpp's batch API expects.
- **State:** none persistent — pure transformation per tick.
- **API:** `build(decode_steps, prefill_chunks) -> LlamaBatch`.

### 6.5 KV Cache Manager
- **Responsibility:** own all KV-cache memory; allocate blocks to sequences, free them on completion/cancellation, decide eviction under pressure.
- **State:** `free_blocks: list[BlockId]`, `block_table: dict[seq_id, list[BlockId]]`, `ref_counts` (for future prefix-sharing).
- **API:** `reserve(seq_id, n_blocks) -> bool`, `release(seq_id)`, `available_blocks() -> int`.
- **Design detail in Section 8.**

### 6.6 Runtime (Model Executor)
- **Responsibility:** thin binding around llama.cpp's `llama_decode`. The *only* component allowed to call into llama.cpp.
- **State:** the loaded `llama_context`, `llama_model`.
- **API:** `async def step(batch: LlamaBatch) -> Logits`. Runs in a dedicated thread (llama.cpp is synchronous/blocking C code) via `asyncio.to_thread`, so it doesn't block the event loop that's servicing HTTP/admission concurrently.

### 6.7 Stream Manager
- **Responsibility:** own per-request output queues; translate token IDs to text incrementally (handling multi-token UTF-8 boundaries correctly — a real, easy-to-get-wrong detail); implement cancellation propagation.
- **State:** `dict[request_id, asyncio.Queue[str | Sentinel]]`.
- **API:** `push_token(request_id, token)`, `close(request_id, reason)`, `subscribe(request_id) -> AsyncIterator[str]`.

### 6.8 Metrics / Tracing
- **Responsibility:** cross-cutting; every other component emits into this rather than owning its own logging.
- **Design:** OpenTelemetry (you already have this in your stack) — one trace per request, spans for each lifecycle stage in Section 5; Prometheus-style counters/histograms for aggregate metrics (Section 10).

## 7. Scheduling — policy design and tradeoffs

| Policy | Pros | Cons | Verdict |
|---|---|---|---|
| **FIFO** | Trivial to implement, trivially fair *in arrival order* | A single long prompt (huge prefill) head-of-line blocks every short request behind it — kills tail latency | Baseline for Phase 1, not the shipped default |
| **Strict Priority** | Simple, good for "premium tier gets served first" | Starvation — low-priority requests can wait forever under sustained high-priority load | Available as a policy option, documented starvation risk |
| **Weighted Fair Queuing** | Bounded unfairness even under skewed load; you already understand this from Ancora's `fairness.py` (virtual-time, newcomer-adopts-minimum, charge/refund) | More bookkeeping; "weight" has to be defined per-request (here: per token processed, not per task) | **Recommended default.** Directly reuses proven design from Ancora, adapted so the resource being fairly divided is *decode steps per tick* rather than *worker-seconds*. |
| **Shortest Remaining Processing Time (SRPT)** | Provably minimizes average latency | Requires knowing remaining length in advance — you don't, for LLM generation (you don't know when EOS will fire), so this degrades to an *estimate*-driven heuristic at best; also can starve long requests | Documented, not implemented in Phase 1-3 — noted as a Phase 4+ experiment using `max_tokens` as a length proxy (imperfect, but usable) |
| **Deadline-aware (EDF-style)** | Useful for SLA-bound traffic (e.g., "must finish TTFT within 200ms") | Needs realistic deadline estimation and admission-time rejection when a deadline is already unmeetable — couples tightly to Admission Controller | Phase 4 add-on, built as a policy plugin, not a rewrite |

**Continuous batching** is not a competing policy — it's the batching *mechanism* that any of the above policies sits on top of. Instead of running one request to completion before starting the next (static batching), the batch loop re-forms its batch every tick: finished sequences leave, newly-scheduled sequences join, all mid-flight sequences advance one decode step. This is what turns GPU utilization from "idle between requests" to "always full while any request is in flight," and it's the single highest-leverage mechanism in the whole design — worth its own dedicated benchmark (Section 11).

**Chunked prefill**: a long prompt's prefill (which can dominate a batch tick and stall everyone else's decode step) is split into fixed-size token chunks, and only one chunk is processed per tick, interleaved with other sequences' decode steps. This directly trades a small increase in that one long request's own prefill latency for a large reduction in tail latency for everyone sharing the batch tick. Recommended default: chunk size tunable, default ~512 tokens, benchmarked in Section 11.

**Recommendation:** WFQ + continuous batching + chunked prefill as the Phase-2/3 default, with FIFO and Priority available as togglable policies for comparison benchmarks (this comparison *is* one of your interview artifacts — "here's the P99 latency under bursty load for FIFO vs WFQ, and here's why").

## 8. KV Cache Design

Explicitly **not** copying vLLM's PagedAttention kernel-level implementation — that requires custom CUDA/attention-kernel support to gather non-contiguous physical blocks at attention time, which llama.cpp doesn't expose the same way. Instead: a simplified, fully-Python, block-based **logical** allocator that teaches the same ideas at a scale you can reason about by hand.

- **Allocation unit:** fixed-size block (e.g., 16 tokens' worth of KV per block). A sequence's KV is a list of block IDs — `block_table[seq_id] = [b3, b17, b4, ...]` — not necessarily contiguous in the logical table (this is the "paged" idea), though the physical backing in llama.cpp's own context buffer may not have per-block indirection unless llama.cpp's own KV-cache API supports sequence-relative addressing (it does, via `llama_kv_cache_seq_*` calls) — this project's block manager sits *on top of* that API, tracking logical ownership and reservation, while llama.cpp's own context still does the physical bookkeeping. Document this boundary explicitly: **TinyServe's KV Cache Manager is an allocation/accounting layer over llama.cpp's native per-sequence KV API, not a from-scratch memory allocator.** That's an honest, defensible scope line.
- **Reuse (prefix caching):** when two requests share a prompt prefix (common with system prompts), the block manager can detect matching block hashes (hash of token IDs in that block) and increment a ref-count instead of reallocating — a simplified version of vLLM's prefix caching, without full copy-on-write semantics for now (documented as a Phase 4 stretch goal).
- **Eviction:** LRU over completed-but-not-yet-freed sequences (e.g., keep a finished request's blocks warm for N seconds in case a retry/continuation reuses the prefix) — evict oldest first when the Admission Controller needs blocks and none are free.
- **Fragmentation:** because allocation is block-granular, the only fragmentation is *internal* (a sequence using 17 tokens still reserves 2 full 16-token blocks, wasting up to one block's worth per sequence) — no *external* fragmentation, which is the whole point of block-based allocation over naive contiguous KV buffers. This is worth a dedicated metric (`kv.fragmentation_ratio`) and a benchmark showing block size vs waste vs allocation overhead tradeoff.
- **Future paged attention:** documented as an explicit non-goal with a clear "here's what would need to change" note — namely, a custom attention kernel capable of gathering scattered physical blocks at attention-compute time, which is out of scope for a llama.cpp-backed project without patching llama.cpp itself.

## 9. Streaming

- **Transport:** SSE by default (`text/event-stream`) — simpler than WebSockets for a strictly server-to-client token stream, no need for bidirectional messaging in v1. WebSocket support documented as a Phase 4 option for use cases needing mid-stream client messages (e.g., interrupt-and-redirect), explicitly not needed for the core demo.
- **Token streaming:** Stream Manager pushes detokenized text chunks (careful UTF-8 boundary handling — don't split multi-byte characters across chunks) onto a per-request `asyncio.Queue`; the SSE generator `await`s and flushes as they arrive. `TTFT` (time-to-first-token) and `ITL` (inter-token latency) are both first-class metrics here (Section 10).
- **Cancellation:** client disconnect detected via FastAPI's request-disconnect hook → marks the sequence `CANCELLED` in the `SequenceTable` → the *next* scheduler tick sees this flag and excludes it from the batch, then the KV Cache Manager frees its blocks. Cancellation is cooperative and bounded by one tick's latency, not instant — documented tradeoff (instant cancellation would need to interrupt an in-flight `llama_decode()` call, which is a blocking C call already in flight; not safely interruptible mid-call).
- **Timeouts:** two kinds — (a) queue timeout (a request waiting too long for admission/scheduling gets cancelled before ever running — protects clients from silently waiting forever under overload) and (b) generation timeout (wall-clock cap on total generation time, same cancellation path as client disconnect).

## 10. Observability

Every scheduling decision must be measurable — this is a design requirement, not an afterthought:

**Metrics (Prometheus-style counters/histograms/gauges):**
- `admission_accepted_total`, `admission_rejected_total{reason}`
- `queue_depth{priority}`, `queue_wait_seconds` (histogram)
- `batch_size` (tokens and requests per tick, histogram)
- `batch_utilization` (actual tokens processed / max batch capacity — the single best "is the scheduler doing its job" number)
- `kv_blocks_free`, `kv_blocks_used`, `kv_fragmentation_ratio`
- `ttft_seconds` (time to first token, histogram)
- `inter_token_latency_seconds` (histogram)
- `tokens_per_second` (per request and aggregate)
- `decode_step_duration_seconds` (wall time of each `llama_decode()` call — the ground truth for "is the GPU/CPU the bottleneck or is Python overhead")
- `scheduler_policy_decision_total{policy, outcome}` — lets you literally graph "which requests got starved under which policy" for the fairness benchmark
- `cancellations_total{reason}`

**Tracing:** one OpenTelemetry trace per request, child spans exactly matching Section 5's lifecycle stages. This is what turns "the p99 was bad" into "the p99 was bad because 40% of it was queue wait under WFQ starvation from a priority-0 burst" — a trace waterfall makes that visible without guesswork.

**Logging:** structured (JSON), request-scoped (every log line carries `request_id` and `trace_id`), scheduler-tick-scoped for batch composition logs (what went into this batch and why — invaluable for debugging fairness bugs).

**Profiling:**
- **Python-side overhead** (event loop, admission, scheduling, batch building — everything *not* inside `llama_decode()`): `py-spy` for low-overhead sampling profiles under load, and `asyncio`'s built-in slow-callback logging to catch anything blocking the event loop it shouldn't.
- **Model execution:** since the model runs inside llama.cpp (C++/GGML, not PyTorch), **`torch.profiler` does not apply here** — that's a correction worth making explicitly rather than silently substituting, since the original ask assumed a PyTorch backend. The correct tools are: **Nsight Systems** (if llama.cpp is built with CUDA — gives a real GPU timeline: kernel launches, memory copies, gaps where the GPU sits idle waiting on the Python scheduler) and llama.cpp's own built-in `llama_perf` timing breakdown (prompt eval time vs eval time vs sampling time, exposed via its C API). This distinction — "here's what changes about your profiling story when your model isn't PyTorch" — is itself a good interview point, not a weakness to hide.
- **End-to-end bottleneck attribution:** compare `decode_step_duration_seconds` (from your own metrics) against Nsight's GPU-busy time for the same window. The gap between them *is* your Python/scheduling overhead, quantified — this is the number that proves (or disproves) that your scheduler isn't the bottleneck.

## 11. Performance Benchmarks

What to run, and why each one exists:

1. **Single-request latency** (no contention) — baseline TTFT and tokens/sec with an empty queue. Establishes the floor.
2. **Burst traffic** — send N requests simultaneously, measure queue growth, admission rejection rate, and how TTFT degrades as a function of burst size. This is what validates (or breaks) the Admission Controller's backpressure.
3. **Continuous/sustained load** — fixed request-per-second rate over minutes, watch for KV fragmentation drift, memory leaks, and whether `batch_utilization` stays high (the core continuous-batching claim, empirically verified rather than assumed).
4. **Throughput scaling** — vary batch size cap, plot tokens/sec vs batch size, find the saturation point (where the GPU/CPU becomes the bottleneck rather than scheduling overhead) — this is where the Nsight comparison from Section 10 gets used directly.
5. **Tail latency (P50/P95/P99)** under mixed short+long request workloads — this is the benchmark that makes chunked prefill's value visible: run it with chunking on vs off, show the P99 difference.
6. **Scheduler fairness** — N low-priority + M high-priority concurrent streams, measure per-class throughput and starvation under FIFO vs Priority vs WFQ. Produces the direct "here's the graph proving WFQ prevents starvation, here's FIFO/Priority failing to" artifact.
7. **Batch efficiency** — `batch_utilization` over time under varying request-size distributions; identifies whether small requests are being wastefully batched with large ones.
8. **Memory/KV usage** — blocks used vs blocks free over a sustained run, fragmentation ratio over time, eviction event frequency.

Each benchmark should be a script in `benchmarks/`, producing a CSV + a plot, checked into the repo — this is what makes "performance engineering" a demonstrable artifact rather than a claim (a lesson worth carrying over from the ToolCallLM audit: a single before/after wall-clock number is weaker than a documented methodology with raw data attached).

## 12. Roadmap — phased, each phase usable

**Phase 0 — Bare execution loop (est. 3-5 days)**
Single-request-at-a-time server. `POST /generate` → tokenize → loop `llama_decode()` one token at a time → return full text (no streaming yet, no batching, no scheduler). Goal: prove the llama.cpp binding works end-to-end. *Usable deliverable: a working, if naive, generate endpoint.*

**Phase 1 — Streaming + basic queueing (est. 1 week)**
Add SSE streaming, Stream Manager, a single FIFO Request Queue, and a batch loop that still processes one request fully before the next (no true batching yet) but is now async and non-blocking. *Usable deliverable: streaming server that doesn't block on concurrent requests.*

**Phase 2 — Continuous batching + KV Cache Manager (est. 1.5-2 weeks)**
The core of the project. Real batch loop merging multiple sequences' decode steps per tick, block-based KV allocation/free, admission control wired to KV budget. *Usable deliverable: a server that demonstrably keeps the GPU busy across concurrent requests — first real benchmark numbers.*

**Phase 3 — Scheduling policies + chunked prefill (est. 1-1.5 weeks)**
Pluggable scheduler policies (FIFO/Priority/WFQ), chunked prefill for long prompts, cancellation, timeouts. *Usable deliverable: policy comparison benchmarks (Section 11, #6) — the fairness story.*

**Phase 4 — Observability + profiling pass (est. 1 week)**
Full OpenTelemetry tracing, Prometheus metrics endpoint, Grafana dashboard (reuse whatever you already know from Gravton/Otsuka's observability stack), Nsight/py-spy profiling session with a written findings doc. *Usable deliverable: the dashboard + profiling report — this is the artifact you screen-share in an interview.*

**Phase 5 — Stretch (optional, only if time remains)**
Prefix caching (ref-counted block reuse), deadline-aware scheduling, WebSocket transport, SRPT-with-length-estimate experiment. *Each is independently choosable — do not treat this phase as required.*

No phase requires rewriting a previous phase's architecture — each adds a component or replaces a policy behind an existing interface (this is itself a design property worth defending: the Scheduler and Runtime interfaces are stable from Phase 1 onward; only implementations behind them change).

## 13. Implementation Constraints

- **Language/stack:** Python 3.11+, FastAPI, AsyncIO throughout, llama.cpp via its Python bindings (`llama-cpp-python`) or a thin custom ctypes/cffi binding if the Python package's threading model gets in the way of the batch-loop design (worth prototyping both early — this is a real risk to flag, not hand-wave).
- **No PostgreSQL** unless a genuine need emerges (e.g., persisting request history for analysis) — the runtime itself needs no database; in-memory state is correct and appropriate given no durability requirement (see Section 0).
- **No Kubernetes, no microservices.** One process, one deployable artifact. If horizontal scaling is ever needed, that's a *load balancer in front of N identical TinyServe processes* — explicitly not solved by this project (see Non-Goals).

## 14. Folder Structure

```
tinyserve/
├── PRD.md                          (this doc)
├── pyproject.toml
├── tinyserve/
│   ├── api/
│   │   ├── app.py                  (FastAPI app, routes)
│   │   └── schemas.py              (pydantic request/response models)
│   ├── admission/
│   │   └── controller.py
│   ├── queue/
│   │   ├── request_queue.py
│   │   └── fairness.py             (WFQ virtual-time bookkeeping)
│   ├── scheduler/
│   │   ├── base.py                 (Policy protocol)
│   │   ├── fifo.py
│   │   ├── priority.py
│   │   └── wfq.py
│   ├── batch/
│   │   └── builder.py
│   ├── kv_cache/
│   │   └── manager.py
│   ├── runtime/
│   │   └── llama_runtime.py        (the only file importing llama.cpp bindings)
│   ├── streaming/
│   │   └── stream_manager.py
│   ├── observability/
│   │   ├── metrics.py
│   │   └── tracing.py
│   └── config.py
├── benchmarks/
│   ├── single_request.py
│   ├── burst.py
│   ├── sustained_load.py
│   ├── fairness.py
│   └── results/                    (checked-in CSVs + plots)
├── tests/
│   ├── unit/                       (scheduler policies, KV manager, fairness math — pure functions, fast)
│   └── integration/                (real llama.cpp calls against a tiny GGUF model, slower, marked separately)
├── docs/
│   ├── architecture.md
│   ├── benchmarks.md               (results + interpretation, updated per phase)
│   └── profiling-notes.md          (Nsight/py-spy findings, written up like the ToolCallLM FP8 report)
└── README.md
```

## 15. Coding Standards, Testing, Docs

- **Testing strategy:** unit tests for everything that's pure logic and fast (scheduler policy decisions, WFQ virtual-time math, KV block allocation/eviction) — these should not need a real model loaded, so they run in CI in seconds. A smaller set of integration tests load a tiny (e.g., 0.5B-parameter GGUF) model and exercise the full lifecycle end-to-end, marked `@pytest.mark.slow` and run separately.
- **Coding standards:** type hints throughout (this is a systems project — types document the data structures crossing component boundaries, which matters more here than in typical app code), `ruff`/`black` for formatting, no comments explaining *what* code does — only *why*, matching the discipline already visible in Ancora's `recovery.py`.
- **Benchmark suite:** each benchmark script is runnable standalone (`python benchmarks/fairness.py`), writes results to `benchmarks/results/`, and is referenced from `docs/benchmarks.md` with the actual numbers and a short interpretation — not just raw CSVs.
- **Documentation plan:** `docs/architecture.md` mirrors this PRD's Section 4-6 but kept in sync with actual code (update as you build, not after). `docs/profiling-notes.md` is the most valuable doc for interviews — a written investigation, in the style of the FP8 report, showing a real bottleneck found and fixed.
- **Demo plan:** a short (2-3 minute) terminal recording or script showing: burst traffic hitting admission control and getting graceful 503s, a live dashboard showing batch utilization and queue depth under load, and a side-by-side fairness comparison (WFQ vs FIFO) with the graph from benchmark #6.
- **GitHub milestones:** one milestone per phase (Section 12), each phase's issues tagged, each phase closed out with a short release note summarizing what became usable and what the benchmark numbers showed — this gives the repo a readable history, which is itself part of the interview artifact (a reviewer can read the commit/PR history and see the design evolve deliberately, not appear all at once).

## 16. Self-Critique

Being honest about this design's weaknesses, unprompted:

- **The KV Cache Manager is an accounting layer over llama.cpp's own sequence-KV API, not a from-scratch allocator.** This is the right scope decision (re-implementing llama.cpp's KV storage would be reinventing model execution, which is explicitly out of scope), but it means the "hardest" part of vLLM's actual innovation — the custom attention kernel that makes *non-contiguous* physical KV blocks fast to attend over — is not something this project builds or can build without patching llama.cpp itself. Be precise about this in interviews: you're building the *scheduling and allocation policy* around KV memory, not the *kernel* that makes paged attention fast. That's still a real, substantial, and correctly-scoped systems project — but don't oversell it as "I built PagedAttention."
- **Cooperative (not preemptive) cancellation** means a cancelled request can still consume one extra batch tick's worth of compute before it's actually removed. Documented tradeoff, not a bug, but worth being able to explain why (interrupting a blocking C call safely is a much harder problem than this project needs to solve).
- **SRPT and deadline-aware scheduling are known-imperfect** without true remaining-length knowledge — the design is honest that these are Phase 4+ experiments with heuristic length estimates, not production-grade guarantees.
- **Single-process, single-model** means this project says nothing about multi-replica load balancing, model routing, or fleet-level scheduling — which is a real, large part of what production inference platforms (including ones at Anthropic-scale) actually have to solve. This project is deliberately the *one-box* problem, done well, not the fleet problem.
- **No prefix-sharing copy-on-write** in the base design (only in the Phase 5 stretch) — a real production system would want this from day one for shared system prompts; it's deferred here to keep Phase 2-3 scoped and demoable.

### Comparison to real systems

| | vLLM | SGLang | llama.cpp (server) | TensorRT-LLM | **TinyServe** |
|---|---|---|---|---|---|
| PagedAttention (real, kernel-level) | Yes | Yes | No | Yes (own impl) | No — logical accounting only |
| Continuous batching | Yes | Yes | Partial (server mode) | Yes | Yes (from scratch) |
| Custom scheduling policies | Limited (mostly FCFS-ish + priority) | RadixAttention-aware scheduling | No | No (fixed internal scheduler) | Yes — pluggable, benchmarked head-to-head |
| Structured/constrained generation, RadixAttention prefix cache reuse | No / N/A | Yes (its signature feature) | No | No | Not in scope |
| Multi-GPU/tensor-parallel | Yes | Yes | Limited | Yes (its core strength) | No — single process, single model |
| Kernel-level performance engineering | Yes (custom CUDA) | Yes | Yes (GGML) | Yes (heaviest investment here) | No — relies entirely on llama.cpp for execution |
| Depth of scheduling explainability | Not designed for this | Not designed for this | Not designed for this | Not designed for this | **This is the entire point of the project** |

The honest positioning: TinyServe is not competing with any of these on throughput or feature completeness. Its value is that every one of the mechanisms in that table that it *does* implement, it implements transparently enough to defend line-by-line in a systems interview — which is a different, and for this purpose more useful, kind of achievement than a partial reimplementation that's neither fast nor fully understood.

## 17. What This Project Intentionally Does Not Solve

- Kernel-level attention performance (PagedAttention's actual kernel, FlashAttention-class fused kernels) — llama.cpp's problem, not this project's.
- Multi-GPU or multi-node serving.
- Structured generation / grammar-constrained decoding.
- Prefix-cache-aware routing across multiple model replicas (SGLang's RadixAttention scheduling problem) — single-model, single-process only.
- Quantization strategy or model-format engineering — llama.cpp/GGUF's domain.
- Production hardening: auth, multi-tenancy isolation, rate-limiting-per-customer, TLS — this is a systems-learning project, not a product.

The goal, restated: not to beat vLLM — to be able to stand in front of an Anthropic/NVIDIA systems interviewer, point at a real (if small) running server, and explain every scheduling decision it makes, backed by benchmark data and profiler traces you generated yourself.
