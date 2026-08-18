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
| Admission Controller | `tinyserve/admission/controller.py` | queue-depth backpressure (`max_queue_depth`) checked first, then worst-case `prompt_tokens + max_tokens` KV reservation |
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

## Admission Controller — two checks, not three

`AdmissionController.admit()` runs two independent, fail-fast checks
before a request ever touches the queue: (1) reject if `queue_depth` is
already at the configured `max_queue_depth` (`TINYSERVE_MAX_QUEUE_DEPTH`,
default 128) — this is what stops a burst of many small-footprint
requests from growing the waiting pool unboundedly even when none of them
individually approach the KV budget; (2) reject if the worst-case
`prompt_tokens + max_tokens` reservation doesn't fit in the KV Cache
Manager's free blocks. Both return a `503` with `Retry-After` and a
distinct `reason` label (`queue_full` vs `kv_cache_full`) on
`admission_rejected_total`, so the two failure modes are distinguishable
in metrics, not just in code.

**Not implemented:** PRD Section 6.1 also describes a token-bucket rate
limiter as part of this component's state. There is no rate limiting
anywhere in this codebase — a client can be admitted as fast as queue
depth and KV budget allow, with no per-client throttling. This is a real
gap relative to the PRD, not a deferred stretch goal.

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

**What PRD Section 8 describes that is still *not* implemented, stated
plainly:**
- **No LRU eviction.** `release()` returns blocks to the free pool
  immediately; nothing is kept warm for a possible retry/continuation.
- **No `kv_fragmentation_ratio` metric.**

**Prefix caching, the other Phase 5 gap, is now implemented** — see the
next section. Note it does *not* change the accounting above: a sequence
that reuses 240 tokens of prefix is still charged the full block budget
for its whole prompt. That is deliberately conservative. llama.cpp shares
the underlying cells rather than duplicating them, so the true cost is
lower than the charge, and `KVCacheManager` does not model sharing —
under-charging would let admission over-commit the real cache. Making the
accounting sharing-aware is a genuine follow-up, and it is what the
missing `kv_fragmentation_ratio` metric would measure.

## Prefix cache — reusing another sequence's prefill

Concurrent requests in a chat deployment usually share a long system
preamble and differ only in a short tail. Without reuse, every one of them
prefills that preamble again. `tinyserve/prefix/` removes that work:

1. At admission, the batch loop asks the cache for the longest
   **block-aligned** prefix of the prompt already resident, and which
   sequence holds it.
2. If there is a hit, `LlamaRuntime.reuse_prefix()` calls
   `llama_memory_seq_cp` to copy the donor's cells onto the new sequence,
   which then starts at `n_past = matched` with those tokens already
   trimmed from `pending_tokens`.
3. The new sequence publishes its own prompt, so later arrivals in the
   same burst can share it — publishing happens at **admission**, not at
   completion, which is the only reason a simultaneous wave benefits at
   all.
4. On finish, the sequence is evicted **before** its `seq_id` returns to
   the free pool. Getting that order wrong is the sharpest bug in the
   feature: a later request claiming a recycled `seq_id` would be offered
   cells that had been freed and refilled with something else, producing
   fluent, plausible, *wrong* output rather than a crash.

Two details are load-bearing:

- **Matches are always whole blocks.** `llama_memory_seq_cp` copies a
  position range and `KVCacheManager` accounts in blocks, so a match that
  did not land on a block boundary would be a number the caller could not
  act on.
- **The match is clamped to leave at least one block to prefill.** A
  sequence whose entire prompt was reused would have no pending tokens,
  hence no batch row, hence no row flagged `needs_logits`, hence no
  sampled token — it would hold a slot forever. That clamp lives in
  Python (`match_for_reuse`), not in the tree, because it encodes how the
  batch loop works rather than anything about prefix indexing.

The index itself is a block-aligned radix tree written in C++
(`native/`), behind a flat C ABI with no Python and no llama.cpp in its
dependency set, loaded over ctypes. **That is not a CPU-time
optimisation, and the repo should not be read as claiming it is:**
`docs/profiling-notes.md` measured 97.2% of wall time inside
`llama_decode()`, and the ctypes batch-fill path benchmarks at 0.63 µs/row
— under 2% of a tick even at full batch capacity. The reason it is C++ is
that llama.cpp's own server is C++ and does a per-slot linear
common-prefix scan, so the same translation unit can be linked there
directly. `native/README.md` has the full argument.

A pure-Python implementation with identical semantics ships alongside and
is used whenever the shared library is absent — TinyServe installs and
behaves correctly with no compiler present. The two are held to the same
behaviour by a differential fuzz that does not require an ABI-compatible
build (`tests/unit/test_prefix_cache_differential.py`). Which backend is
live is exposed as `prefix_cache_backend_info{backend=...}`.

Measured on `benchmarks/prefix_reuse.py` against a shared-system-prompt
wave: **47–70% of prompt tokens never reached `llama_decode()`**.

## Runtime — the llama.cpp boundary

`LlamaRuntime` is a thin binding around the low-level ctypes API
(`llama_batch_init`, `llama_decode`, `llama_get_logits_ith`,
`llama_memory_seq_cp`, `llama_perf_context`), not the
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

`LlamaRuntime` owns two objects with overlapping lifetimes — a `Llama`
(which owns the model weights) and a `LlamaContext` built over that
model — and Python guarantees no finalization order between them. If the
`Llama` is collected first, the context's destructor calls `llama_free()`
on a context whose model is already gone, which **faults rather than
raising**: an access violation out of `__del__` during interpreter
teardown, where the traceback names nothing useful. The `llama_batch` was
also never freed. `close()` (and the context-manager protocol) frees all
three in dependency order and is idempotent. The server builds exactly
one Runtime and leaks it into exit, so this only reproduces in a process
that builds several — a test session — which makes it easy to dismiss as
a test-only artefact. It is not; it is a latent use-after-free that the
usual lifecycle happens to hide.

## Streaming and cancellation

SSE only (no WebSocket transport — PRD Section 9 flags this as an
explicit non-goal for v1). Cancellation is cooperative: a disconnected
client or an expired timeout marks the sequence for removal, and it's
actually dropped on the *next* batch tick, not instantly — interrupting
an in-flight blocking `llama_decode()` call isn't something this design
attempts, matching PRD Section 9's documented tradeoff.

**Known gap:** `generation_timeout_seconds` is only checked in
`batch_loop`'s per-sampled-token loop — i.e. only once a sequence has
started producing decode tokens. A sequence still inside a long, multi-
tick chunked prefill (no token sampled yet) is not checked against this
timeout at all. In practice chunked prefill completes in a bounded number
of ticks for any request `AdmissionController` would have accepted, so
this doesn't hang, but the timeout's actual coverage is narrower than PRD
Section 9's "wall-clock cap on total generation time" implies — it caps
decode time, not prefill time.

## Observability

Every scheduling decision in `batch_loop` emits a metric or a trace span
transition — this is enforced by reading the function, not by
convention. One deliberate shape decision: `decode_step_duration_seconds`
is a metric, not a per-request trace span, because a single
`llama_decode()` call can advance multiple requests' sequences at once,
and forcing that into a single-parent span shape would misrepresent what
actually happened. See `docs/profiling-notes.md` for the real profiling
session that cross-checks this metric against `py-spy` sampling data.

**Not implemented:** PRD Section 10 also calls for structured (JSON),
request-scoped logging — every log line carrying `request_id`/`trace_id`,
plus scheduler-tick-scoped batch-composition logs. This codebase has
exactly one logging call site (`logger.exception(...)` in `app.py`'s
decode-error handler), using plain stdlib text logging with no structure
and no request correlation. Metrics and tracing cover the "why was my
request slow" question PRD Section 10 asks for; logging does not.

## What's out of scope (unchanged from the PRD)

No prefix caching, no deadline-aware scheduling, no WebSocket transport,
no SRPT experiment — all four were explicitly scoped as optional Phase 5
stretch goals (PRD Section 12) and a deliberate decision was made to ship
a fully benchmarked Phase 0-4 project instead of a partially-implemented
Phase 5. No multi-GPU, no distributed serving, no PagedAttention kernel —
these were non-goals from the start (PRD Section 3) and remain so.
