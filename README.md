# TinyServe

A small, fully-understood LLM inference runtime built from scratch on top of
[llama.cpp](https://github.com/ggml-org/llama.cpp): continuous batching, block-based
KV-cache management, admission control, and pluggable fair scheduling (FIFO / priority / WFQ).

See [`PRD.md`](./PRD.md) for the full design doc — architecture, scheduling tradeoffs,
KV-cache design, observability plan, and the phased roadmap this project follows.

## Status

Phase 0 (repo/CI/tooling) and Phase 0's "bare execution loop" milestone are done: a naive,
non-streaming `POST /generate` running against a real llama.cpp-loaded GGUF model. See
`PRD.md` Section 12 for the full roadmap.

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
