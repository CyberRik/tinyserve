# TinyServe

A small, fully-understood LLM inference runtime built from scratch on top of
[llama.cpp](https://github.com/ggml-org/llama.cpp): continuous batching, block-based
KV-cache management, admission control, and pluggable fair scheduling (FIFO / priority / WFQ).

See [`PRD.md`](./PRD.md) for the full design doc — architecture, scheduling tradeoffs,
KV-cache design, observability plan, and the phased roadmap this project follows.

## Status

Phases 0-4 are done: repo/CI/tooling, streaming, real multi-sequence continuous batching
with a block-based KV Cache Manager, pluggable scheduling (FIFO/Priority/WFQ) with chunked
prefill and timeouts, and Prometheus metrics + OpenTelemetry tracing. See `PRD.md` Section 12
for the full roadmap.

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
