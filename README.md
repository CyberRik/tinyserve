# TinyServe

A small, fully-understood LLM inference runtime built from scratch on top of
[llama.cpp](https://github.com/ggml-org/llama.cpp): continuous batching, block-based
KV-cache management, admission control, and pluggable fair scheduling (FIFO / priority / WFQ).

See [`PRD.md`](./PRD.md) for the full design doc — architecture, scheduling tradeoffs,
KV-cache design, observability plan, and the phased roadmap this project follows.

## Status

Phases 0-4 are done: repo/CI/tooling, streaming, real multi-sequence continuous batching
with a block-based KV Cache Manager, pluggable scheduling (FIFO/Priority/WFQ) with chunked
prefill and timeouts, and Prometheus metrics + OpenTelemetry tracing. The benchmark suite
(PRD Section 11) and architecture docs are also done — see `docs/architecture.md` for what's
actually implemented (including honest deltas from the PRD) and `docs/benchmarks.md` for real
measured numbers. Phase 5 (prefix caching, deadline-aware scheduling, WebSocket transport,
SRPT) is explicitly optional stretch per the PRD and was not picked up. See `PRD.md` Section 12
for the full roadmap.

## Benchmarks

`benchmarks/` has eight standalone scripts (PRD Section 11), each runnable directly against
a running server and each writing a CSV (+ PNG where there's a natural x-axis) to
`benchmarks/results/`:

```bash
uv run python benchmarks/single_request.py       # baseline TTFT / tokens-per-sec
uv run python benchmarks/burst.py                # admission backpressure under burst traffic
uv run python benchmarks/sustained_load.py        # KV usage / batch utilization over time
uv run python benchmarks/throughput_scaling.py    # tokens/sec vs concurrent demand
uv run python benchmarks/tail_latency.py --label x   # chunked prefill's effect on tail latency
uv run python benchmarks/fairness.py --label x       # FIFO vs Priority vs WFQ head-to-head
uv run python benchmarks/batch_efficiency.py      # batch_utilization vs request-size distribution
uv run python benchmarks/kv_usage.py              # KV block occupancy over a sustained run
```

`tail_latency.py` and `fairness.py` compare server configurations (chunk size / scheduling
policy), so they need multiple server instances — see the docstring at the top of each script
for the exact invocations. Real results and interpretation: `docs/benchmarks.md`.

## Development

This project uses [uv](https://docs.astral.sh/uv/) for dependency management.

```bash
uv sync
uv run ruff check .
uv run mypy tinyserve
uv run pytest tests/unit
```

## Running the server

Point `TINYSERVE_MODEL_PATH` at any local GGUF file (a small instruct model, e.g.
[Qwen2.5-0.5B-Instruct-GGUF](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct-GGUF), is
enough for local development):

```bash
TINYSERVE_MODEL_PATH=models/qwen2.5-0.5b-instruct-q4_k_m.gguf uv run uvicorn tinyserve.api.app:app

curl -X POST http://127.0.0.1:8000/generate \
  -H "Content-Type: application/json" \
  -d '{"prompt": "The capital of France is", "max_tokens": 32}'
```

Integration tests that load a real model are marked `@pytest.mark.slow` and skipped
automatically if `models/qwen2.5-0.5b-instruct-q4_k_m.gguf` isn't present:

```bash
uv run pytest tests/integration -m slow
```

## Observability

`GET /metrics` exposes Prometheus-format metrics (admission accept/reject, queue depth and
wait time, batch size/utilization, KV block usage, TTFT, inter-token latency, decode step
duration, scheduler decisions, cancellations). Each request also gets one OpenTelemetry trace
with `admission` / `queue_wait` / `generation` child spans, printed to the console by default.

A local Prometheus + Grafana stack (with a pre-provisioned dashboard) is in `deploy/`:

```bash
cd deploy
docker compose up -d
# Prometheus: http://localhost:9090
# Grafana:    http://localhost:3000  (anonymous admin access, local dev only)
```

Prometheus scrapes TinyServe on the host via `host.docker.internal:8000`, so start the
server on the host first. See `docs/profiling-notes.md` for a real `py-spy` profiling
session and what it found.
