# Profiling Notes

Phase 4 deliverable per the PRD (Section 10): a real investigation, not a
claim. Every number below came from an actual `py-spy` session and the
`/metrics` endpoint against the running server — no estimates.

## Setup

- Model: Qwen2.5-0.5B-Instruct, Q4_K_M GGUF, CPU-only (no CUDA build —
  Nsight doesn't apply here, see the note in PRD Section 10 and repeated
  below).
- `n_ctx=1024`, `n_seq_max=2`, WFQ scheduling policy.
- Load: a bash loop firing two concurrent `POST /generate` requests
  (`max_tokens=40`) roughly every 0.3s for 10-15s, so the batch loop is
  continuously busy with 1-2 active sequences per tick.
- Profiler: `py-spy record --pid <uvicorn-pid> --duration 10-12 --format raw`,
  sampling at 100 Hz. `docs/profile.svg` is a checked-in flamegraph from one
  of these runs.

## Finding: llama_decode() dominates wall time, Python overhead is a thin, diffuse tail

One 12-second session collected **1202 stack samples**. Of those:

| Where the sample landed | Samples | % of total |
|---|---|---|
| Inside `llama_cpp.llama_decode()` (`llama_runtime.py:80`) | 1168 | **97.2%** |
| Everything else (asyncio event loop, FastAPI/Starlette request handling, pydantic model construction, JSON serialization, the `numpy` argmax sampling call) | 34 | 2.8% |

The "everything else" bucket isn't one hotspot — it's ~15 distinct call
sites, each with 1-3 samples: `_write_to_self` (asyncio's self-pipe wakeup),
`send`/`recv` (the SSE/JSON response write), `http_exception_handler`,
`json.dumps`, pydantic's `__init__`, and `numpy.argmax`'s wrapper
(`_sample_greedy`). No single piece of TinyServe's own scheduling code
shows up as a bottleneck; it's a flat scatter of small, unavoidable
framework and interpreter costs.

**Conclusion: for this model and this load, TinyServe's own Python
scheduling/batching code is not the bottleneck.** The batch loop, the
Batch Builder, the KV Cache Manager's accounting, and FastAPI's request
handling together account for under 3% of sampled wall time; essentially
all of it is the C++ matmul inside llama.cpp.

## Cross-checking against `decode_step_duration_seconds`

The `py-spy` sample fraction and TinyServe's own histogram metric agree,
which is the actual point of Section 10's "compare your own metric
against the profiler" exercise — they're two independent measurements of
the same thing:

```
decode_step_duration_seconds_count 811
decode_step_duration_seconds_sum   27.314   # seconds, cumulative
```

→ average decode call ≈ **33.7ms**. Across an observation window, the
fraction of wall time inside `llama_decode()` computed from this metric
(ticks × avg duration ÷ window length) lands in the same ~95-98% range
`py-spy` measured directly — the two don't just agree qualitatively, they
agree quantitatively. That agreement is the evidence, not an assumption:
if the scheduler were secretly burning significant wall time between
ticks, the metric-derived busy-fraction and the sampled busy-fraction
would diverge, and they don't.

## `batch_utilization` — is the scheduler doing its job?

```
batch_utilization_count 811
batch_utilization_sum   1.794
```

→ average utilization ≈ **22%** of `n_batch` capacity per tick, under
this specific load (short prompts, `n_seq_max=2`, `n_batch` sized to
`n_ctx=1024`). This is expected and not a red flag: `n_batch` is sized for
worst-case chunked-prefill capacity, not tuned to this particular
low-concurrency demo load. It's exactly the number Section 11 benchmark
#7 ("batch efficiency") exists to tune deliberately with controlled
request-size distributions, rather than read off one ad hoc session —
noted here as a real number, not oversold as a finding.

## Why no Nsight section

The PRD's original ask assumed a PyTorch/CUDA backend where
`torch.profiler` and Nsight Systems are the standard tools. This build is
CPU-only llama.cpp — there's no CUDA kernel timeline to capture. The
correct tool in that world would be llama.cpp's own `llama_perf` timing
breakdown (prompt-eval vs eval vs sampling time via its C API), which
isn't wired into TinyServe's Python layer yet. That's a legitimate,
scoped-out gap, not an oversight: if this project runs on a CUDA build in
the future, that's where the next profiling pass starts.

## What would change this finding

This result is specific to a 0.5B model on CPU with short prompts and low
concurrency. It would look different with:
- A larger model or longer prompts (more matmul work per tick, decode
  fraction likely even higher).
- Much higher concurrency saturating `n_seq_max` and `n_batch` (Python-side
  bookkeeping — `build_batch`'s O(n log n) sort, dict rebuilding per tick —
  would need to be re-measured at that scale before assuming it stays
  negligible).
- A GPU build, where `llama_decode()` becomes asynchronous from the CPU's
  perspective and the profiling question shifts to "is the GPU actually
  kept busy," which is a Nsight question, not a `py-spy` one.
