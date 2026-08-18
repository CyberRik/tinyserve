# Benchmarks

PRD Section 11 deliverable: nine scripts in `benchmarks/`, each runnable
standalone, each producing a CSV in `benchmarks/results/` (plus a PNG for
the ones with a natural x-axis to sweep). Every number below is from an
actual run against the real server and the real Qwen2.5-0.5B-Instruct
Q4_K_M GGUF model — no estimates, no synthetic numbers.

Unless stated otherwise, runs used `n_ctx=2048`, `n_seq_max=4`,
`chunk_size=512`, `scheduling_policy=wfq` (the shipped defaults) on CPU.

## 1. Single-request latency (`single_request.py`)

Baseline: no contention, one request at a time.

```
n=5/5 succeeded
TTFT     mean=0.1044s  p50=0.0445s  p95=0.2873s
tok/sec  mean=48.78  p50=48.03  p95=58.44
```

This is the floor every other benchmark's numbers should be read against:
~45-60 tokens/sec and well under 300ms TTFT when nothing else is competing
for the batch tick.

## 2. Burst traffic / admission backpressure (`burst.py`)

Sweeping burst size (all requests fired simultaneously) and watching
admission accept/reject counts and TTFT:

| burst_size | accepted | rejected | ttft_p50 (s) | ttft_p95 (s) |
|---|---|---|---|---|
| 1  | 1  | 0  | 0.072 | 0.072 |
| 2  | 2  | 0  | 0.090 | 0.119 |
| 4  | 4  | 0  | 0.172 | 0.241 |
| 8  | 8  | 0  | 0.742 | 1.255 |
| 16 | 16 | 0  | 1.739 | 3.255 |
| 32 | 32 | 0  | 4.517 | 8.136 |
| 64 | 42 | 22 | 5.567 | 9.638 |

**Interpretation:** TTFT grows roughly linearly with burst size while
KV budget still allows admission — that's queue-wait accumulating behind
`n_seq_max=4` concurrency slots, not admission rejecting anything yet.
At burst=64, admission control kicks in for real: 22 of 64 requests get a
`503` with `Retry-After` rather than being queued indefinitely. The
rejection point (42 accepted) matches hand math: `n_ctx=2048` / `block_size=16`
= 128 total blocks; each of these ~15-token-prompt + 24-max-token requests
reserves `ceil(39/16)=3` blocks worst-case, so `128/3≈42` is exactly where
the budget runs out. **This is the Admission Controller's backpressure
claim, verified with a real number, not asserted.**

## 3. Sustained load (`sustained_load.py`)

Fixed request rate over a duration, sampling `/metrics` every second.
A 15s run at 2 req/s produced 16 samples with `kv_blocks_used` moving
between 3 and 6 blocks, tracking demand and returning to baseline as
requests completed, with no runaway growth over the window tested.

**Scope of this claim, stated precisely:** 15 seconds is nowhere near
long enough to claim "no memory leak" in the sense that phrase usually
means — a slow leak (a handful of blocks per hour, say) would be
invisible at this timescale. What this run actually demonstrates is that
occupancy doesn't grow monotonically over *this* window; it is not
evidence against a slow leak over minutes or hours. The mechanism
(interval-local delta of the cumulative `batch_utilization` histogram,
not a since-start average, so short-term drift wouldn't get hidden) is
sound and worth reusing for a real multi-minute soak test — that longer
run is what would actually be needed to make a leak-freedom claim, and it
was not run here.

## 4. Throughput scaling (`throughput_scaling.py`)

**Deviation from the PRD's original framing, documented up front:** the
PRD asks to "vary batch size cap" — but `n_batch` is fixed at server
startup (`LlamaRuntime` derives it from `n_ctx`), not a per-request or
runtime-tunable knob, so it can't be swept without restarting the process.
The honest substitute: vary *concurrent demand* instead and watch
aggregate tokens/sec for the saturation point.

| concurrency | accepted | wall (s) | total tokens | agg tok/sec |
|---|---|---|---|---|
| 1  | 1  | 0.873 | 47  | 53.8 |
| 2  | 2  | 1.473 | 94  | 63.8 |
| 4  | 4  | 2.084 | 188 | 90.2 |
| 8  | 8  | 3.986 | 376 | 94.3 |
| 16 | 16 | 7.822 | 752 | 96.1 |

**Interpretation:** throughput climbs from concurrency 1→4, then flattens
hard from 4→16 (90.2 → 94.3 → 96.1 — a ~6% total gain across a 4x increase
in concurrent demand). That plateau starts almost exactly at
`n_seq_max=4` — the configured concurrency-slot limit — which is the
real, load-bearing saturation point for this model/config, not CPU
decode cost (`py-spy`'s profiling session, `docs/profiling-notes.md`,
already showed `llama_decode()` itself as 97.2% of wall time, so this
plateau is "no more slots available," confirmed structurally rather than
just inferred from the shape of the curve).

## 5. Tail latency: chunked prefill on vs off (`tail_latency.py`)

Two server instances, `n_ctx=768`, `n_seq_max=2`, one with
`chunk_size=32`, one with `chunk_size=8192` (long enough to never
actually chunk a ~450-token prompt in this config — i.e. "off"). 6 short
prompts + 2 long prompts (a ~450-token repeated paragraph) fired
concurrently; TTFT reported for the *short* prompts only:

| label | p50 (s) | p95 (s) | p99 (s) |
|---|---|---|---|
| chunked_32     | 0.511 | 2.666 | 3.136 |
| unchunked_8192 | 0.524 | 2.509 | 2.930 |

**Honest finding: no meaningful difference at this scale, and that's
worth explaining rather than hiding.** `build_batch`
(`tinyserve/batch/builder.py`) sorts active sequences by ascending
pending-token count before packing a tick — decode steps and short
prefills always get greedily packed in *before* a long prefill's slice,
regardless of `chunk_size`. Chunking's actual effect is capping how much
of the long sequence's own remaining prompt gets consumed once it's
already last in line, not whether short requests get starved by it —
short requests were never behind it in the queue to begin with. The
scenario chunking would visibly help is a long prefill occupying the
*only* available concurrency slot (`n_seq_max=1`) with short requests
queued behind it, or several long prefills contending for tick capacity
simultaneously — neither is what this benchmark drove. Recorded as a
real, measured result and a real gap in the initial hypothesis, not
smoothed over.

## 6. Scheduler fairness: FIFO vs Priority vs WFQ (`fairness.py`)

Three server instances (`n_seq_max=2`, one per policy), 12 low-priority +
4 high-priority concurrent requests each:

| policy | low mean TTFT (s) | high mean TTFT (s) | low/high ratio |
|---|---|---|---|
| FIFO     | 2.121 | 4.761 | 0.45x |
| Priority | 3.716 | 0.960 | 3.87x |
| WFQ      | 3.277 | 1.292 | 2.54x |

**Interpretation — this is the fairness story the PRD asks for:**

- **FIFO ignores priority entirely** (as designed) — the ratio being
  *below* 1 here is a real artifact of `asyncio.gather` firing all 16
  requests at effectively the same instant, so "arrival order" among
  them is close to coincidental; the point isn't the exact ratio, it's
  that FIFO's ordering has no relationship to priority at all.
- **Strict Priority strongly favors high-priority** (3.87x) — exactly
  the documented tradeoff: better for the favored class, at real cost to
  the other (PRD Section 7's "starvation risk," visible here as low's
  mean TTFT growing to 3.7s).
  - **WFQ favors high-priority less extremely** (2.54x vs Priority's
  3.87x) while still clearing high-priority requests faster than FIFO
  does — bounded unfairness in practice, not just in theory: same
  high-priority traffic gets better treatment than under FIFO, without
  the low-priority class being starved as hard as it is under strict
  Priority.

## 7. Batch efficiency (`batch_efficiency.py`)

Average `batch_utilization` (tokens processed / `n_batch` capacity) under
three request-size distributions, default config (`n_ctx=2048`, so
`n_batch=2048`):

| distribution | avg batch_utilization |
|---|---|
| all_short | 0.0021 |
| all_long  | 0.0081 |
| mixed     | 0.0051 |

**Interpretation:** utilization is low across the board and scales with
request size, as expected — `n_batch=2048` is sized for worst-case
chunked-prefill capacity, not tuned to this demo's small prompts
(15-100 tokens). This matches `docs/profiling-notes.md`'s independent
~22% utilization finding under a different load shape; both agree
utilization here is a function of "how big are the actual requests
relative to `n_batch`," not a sign the scheduler is doing anything wrong.
Tuning `n_batch`/`n_ctx` to the real expected request-size distribution
is exactly what this metric is for.

## 8. KV / memory usage (`kv_usage.py`)

20s sustained run at 4 req/s, default config (128 total blocks):

```
41 samples over ~20s
kv_blocks_used: min=3 max=39 end=36  (of 128 total)
```

**Documented gap, not glossed over:** PRD Section 8 describes LRU
eviction (keep a finished sequence's blocks warm briefly, evict oldest
first under pressure) as part of the KV Cache Manager's design.
`tinyserve/kv_cache/manager.py` does not implement this — `release()`
returns blocks to the free pool immediately, and there is no
`kv_fragmentation_ratio` metric. This benchmark therefore reports raw
block occupancy only; there's no eviction-event count to show because
there's no eviction. See `docs/architecture.md` for the full list of
places the shipped code trims PRD scope.

## 9. Prefix reuse under a shared system prompt (`prefix_reuse.py`)

The workload prefix caching exists for: many concurrent requests sharing a long
system preamble, differing only in a short question. Three waves of 8
simultaneous requests, `n_seq_max=4`, `max_tokens=24`, run against two servers
differing only in `TINYSERVE_PREFIX_CACHE_ENABLED`. Three paired invocations,
identical prompt-token totals (1908) on both sides each time.

| run | prefix cache | prompt tokens | reused | reuse rate | TTFT p50 | TTFT p95 |
|---|---|---|---|---|---|---|
| 1 | on  | 1908 | 1344 | **70.4%** | 1.040s | 1.807s |
| 1 | off | 1908 | 0 | 0% | 2.001s | 3.865s |
| 2 | on  | 1908 | 1280 | **67.1%** | 1.501s | 3.327s |
| 2 | off | 1908 | 0 | 0% | 2.253s | 4.143s |
| 3 | on  | 1908 | 896 | **47.0%** | 1.636s | 4.554s |
| 3 | off | 1908 | 0 | 0% | 2.439s | 4.155s |

**The headline is the reuse rate, not the latency.** `prefill_tokens_reused_total`
is an exact counter of prompt tokens that never reached `llama_decode()` — work
provably not done, not a sampled estimate. Between **47% and 70%** of prompt
tokens were skipped.

**Why the reuse rate varies so much across identical runs.** The prompts are
identical every time; what differs is how many donors happen to be live at the
moment each request is admitted. With `n_seq_max=4`, a wave of 8 is admitted in
staggered groups, and a sequence's claim is evicted the instant it finishes.
Run 3 also recorded 7 `stale_donor` lookups — donors freed between the tree
lookup and the copy — which the batch loop correctly treats as a miss and
prefills normally. So the spread is a property of *slot churn under this
concurrency*, not of the cache, and it would narrow with more slots or longer
generations.

**TTFT p50 is consistently lower with the cache on** — 1.040/1.501/1.636s against
2.001/2.253/2.439s, with no overlap between the two sets. That is a real effect
and roughly a 1.4–1.9× improvement.

**TTFT p95 supports no claim at all, and is included so that's visible.** On:
1.807/3.327/4.554s. Off: 3.865/4.143/4.155s. The ranges overlap heavily and the
best and worst p95 in the whole table are both from cache-on runs. At three
waves of eight on a CPU build, the tail is dominated by queueing behind
`n_seq_max=4`, and prefill savings do not separate from that noise. Establishing
a p95 effect would need far more samples than this.

**Caveat on scale.** This is a 0.5B model on CPU with a ~60-token preamble. The
reuse *rate* is a property of the workload and would hold anywhere; what it is
worth in latency scales with how expensive prefill actually is, which is much
higher for a larger model or a longer system prompt.

## Reproducing these numbers

```bash
uv run uvicorn tinyserve.api.app:app &   # or the docker compose stack
uv run python benchmarks/single_request.py
uv run python benchmarks/burst.py --sizes 1,2,4,8,16,32,64
uv run python benchmarks/sustained_load.py --rps 2 --duration 15
uv run python benchmarks/throughput_scaling.py --concurrency 1,2,4,8,16
uv run python benchmarks/batch_efficiency.py
uv run python benchmarks/kv_usage.py --rps 4 --duration 20
```

Benchmarks 5 and 6 need multiple differently-configured server instances
— see the docstring at the top of `tail_latency.py` and `fairness.py` for
the exact env-var invocations used to produce the numbers above.
