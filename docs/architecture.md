# Architecture

This mirrors PRD Sections 4-6 but describes the code as it actually
exists, kept in sync as the implementation evolves rather than written
after the fact. Where the shipped code diverges from the PRD, that's
called out explicitly — the goal is that this document and `PRD.md`
never quietly drift apart.

## Request flow

```
POST /generate  (tinyserve/api/app.py)
   │
   ▼
tokenize (LlamaRuntime.tokenize, off the batch loop)
   │
   ▼
AdmissionController.admit()  ── reject ──▶ 503 + Retry-After
   │ accept
   ▼
RequestQueue.push()  (waiting pool, not a blind FIFO — SchedulingPolicy
                       reorders it every tick)
   │
   ▼
batch_loop() tick:
   1. expire_stale_waiters()        — queue_timeout_seconds
   2. policy.select(waiting, capacity=free_slots)  — admission into a
      free concurrency slot is the only lever a policy has once
      continuous batching is running (see "Scheduling" below)
   3. build_batch(active, batch_capacity, chunk_size)  — chunked prefill
      + decode steps combined into one llama.cpp batch shape
   4. runtime.decode(batch)          — the only llama_decode() call
   5. sample + detokenize + push to StreamManager per finished sequence
   │
   ▼
StreamManager  ── SSE ──▶ client
```

One background asyncio task (`batch_loop`, `tinyserve/api/app.py`) owns
every call into `LlamaRuntime` — this sidesteps needing locks around
llama.cpp state, matching PRD Section 4's single-consumer design.

## Component map

| Component | File | Notes |
|---|---|---|
| FastAPI app | `tinyserve/api/app.py` | routes + `batch_loop` + `lifespan` |
| Admission Controller | `tinyserve/admission/controller.py` | worst-case `prompt_tokens + max_tokens` KV reservation at accept time |
| Request Queue | `tinyserve/queue/request_queue.py` | reorderable waiting pool keyed by request id, not a plain FIFO |
| Fairness bookkeeping | `tinyserve/queue/fairness.py` | `WeightedFairQueue` virtual-time math |
| Scheduling policies | `tinyserve/scheduler/{base,fifo,priority,wfq}.py` | `SchedulingPolicy` Protocol + `create_policy()` factory |
| Batch Builder | `tinyserve/batch/builder.py` | pure transformation, no llama.cpp import |
| KV Cache Manager | `tinyserve/kv_cache/manager.py` | logical block accounting only — see "KV Cache" below |
| Runtime | `tinyserve/runtime/llama_runtime.py` | **the only file allowed to import llama.cpp bindings** |
| Stream Manager | `tinyserve/streaming/stream_manager.py` | per-request queues, incremental UTF-8 decode, cooperative cancellation |
| Metrics | `tinyserve/observability/metrics.py` | isolated `CollectorRegistry` per instance |
| Tracing | `tinyserve/observability/tracing.py` | one trace per request, spans opened/closed from different call sites |

## Scheduling — what a policy actually controls

Continuous batching means every active sequence advances one decode step
every tick regardless of policy — that part isn't negotiable once a
sequence has a concurrency slot. The only real lever a `SchedulingPolicy`
has is **which waiting requests claim a free slot** when one opens up
(`policy.select(queue.waiting(), capacity=len(free_slots))` in
`batch_loop`). All three shipped policies (`FIFOPolicy`, `PriorityPolicy`,
`WFQPolicy`) are built around that one honest lever — see
`docs/benchmarks.md` #6 for the measured difference this makes.

`WFQPolicy` reuses the virtual-time bookkeeping pattern from the
companion Ancora project's `fairness.py`: priority doubles as both the
fairness-class key and the weight, and a newcomer class adopts the
current system minimum virtual finish time rather than starting at zero
— this is what keeps a bursty high-priority class from being unfairly
penalized just for showing up late.

## KV Cache — logical accounting, not a physical allocator

**This is the most important scope line in the whole project, restated
here because it's easy to oversell by accident:** `KVCacheManager` is a
pure-Python block-counting layer over llama.cpp's own per-sequence KV API
(`llama_kv_cache_seq_rm` etc., called from `LlamaRuntime`, sized by a
single unified `llama_context` created with `kv_unified=True`). It does
not manage physical memory, gather non-contiguous blocks at attention
time, or implement anything resembling vLLM's PagedAttention kernel. It
answers exactly one question — "does this sequence have enough logical
budget reserved?" — via `reserve()` / `release()` / `available_blocks()`.

**Two things PRD Section 8 describes that are *not* implemented, stated
plainly:**
- **No LRU eviction.** `release()` returns blocks to the free pool
  immediately; nothing is kept warm for a possible retry/continuation.
- **No prefix caching / ref-counted block reuse** and correspondingly no
  `kv_fragmentation_ratio` metric. Both were scoped as Phase 5 stretch
  goals in the PRD and were not picked up (see the roadmap decision in
  this repo's history) — they remain real, understood gaps, not
  oversights.

## Runtime — the llama.cpp boundary

`LlamaRuntime` is a thin binding around the low-level ctypes API
(`llama_batch_init`, `llama_decode`, `llama_get_logits_ith`), not the
high-level `Llama` convenience class — the high-level API's own threading
model didn't fit the batch loop's need to drive multi-sequence decode
directly (this was validated with a standalone proof-of-concept before
committing to the design, per PRD Section 13's flagged risk). Sampling is
**greedy-only** (`argmax` over logits) — no temperature, top-k/top-p, or
repetition penalty. This is a real, deliberate simplification: it keeps
the sampling code auditable in a few lines and doesn't touch anything
this project is trying to demonstrate (scheduling, batching, KV
accounting), but it does mean output quality/diversity is not
representative of a production sampler.

## Streaming and cancellation

SSE only (no WebSocket transport — PRD Section 9 flags this as an
explicit non-goal for v1). Cancellation is cooperative: a disconnected
client or an expired timeout marks the sequence for removal, and it's
actually dropped on the *next* batch tick, not instantly — interrupting
an in-flight blocking `llama_decode()` call isn't something this design
attempts, matching PRD Section 9's documented tradeoff.

## Observability

Every scheduling decision in `batch_loop` emits a metric or a trace span
transition — this is enforced by reading the function, not by
convention. One deliberate shape decision: `decode_step_duration_seconds`
is a metric, not a per-request trace span, because a single
`llama_decode()` call can advance multiple requests' sequences at once,
and forcing that into a single-parent span shape would misrepresent what
actually happened. See `docs/profiling-notes.md` for the real profiling
session that cross-checks this metric against `py-spy` sampling data.

## What's out of scope (unchanged from the PRD)

No prefix caching, no deadline-aware scheduling, no WebSocket transport,
no SRPT experiment — all four were explicitly scoped as optional Phase 5
stretch goals (PRD Section 12) and a deliberate decision was made to ship
a fully benchmarked Phase 0-4 project instead of a partially-implemented
Phase 5. No multi-GPU, no distributed serving, no PagedAttention kernel —
these were non-goals from the start (PRD Section 3) and remain so.
