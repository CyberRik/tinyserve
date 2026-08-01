# TinyServe

A small, fully-understood LLM inference runtime built from scratch on top of
[llama.cpp](https://github.com/ggml-org/llama.cpp): continuous batching, block-based
KV-cache management, admission control, and pluggable fair scheduling (FIFO / priority / WFQ).

See [`PRD.md`](./PRD.md) for the full design doc — architecture, scheduling tradeoffs,
KV-cache design, observability plan, and the phased roadmap this project follows.

## Status

Phase 0 — repository setup, CI, and tooling. No inference code yet; see `PRD.md` Section 12
for the roadmap.

## Development

This project uses [uv](https://docs.astral.sh/uv/) for dependency management.

```bash
uv sync
uv run ruff check .
uv run mypy tinyserve
uv run pytest tests/unit
```
