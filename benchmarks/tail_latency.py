"""Benchmark 5 (PRD Section 11, #5): tail latency under mixed short+long prompts.

Chunked prefill's whole point is protecting short requests' tail latency
from a long prompt's prefill hogging a batch tick. This script fires a
mix of short and long prompts concurrently and reports P50/P95/P99 TTFT
for the *short* requests specifically — that's where chunking's effect
should show up.

To see chunking's actual effect, run this script twice against two
server instances with different TINYSERVE_CHUNK_SIZE values, e.g.:

    TINYSERVE_CHUNK_SIZE=64   uv run uvicorn tinyserve.api.app:app --port 8001
    TINYSERVE_CHUNK_SIZE=4096 uv run uvicorn tinyserve.api.app:app --port 8002   # effectively "off"

    python benchmarks/tail_latency.py --base-url http://127.0.0.1:8001 --label chunked
    python benchmarks/tail_latency.py --base-url http://127.0.0.1:8002 --label unchunked

Usage: python benchmarks/tail_latency.py [--base-url URL] --label chunked
"""

import argparse
import asyncio
import csv
from pathlib import Path

import httpx
from _common import DEFAULT_BASE_URL, RequestResult, percentile, stream_generate

RESULTS_DIR = Path(__file__).parent / "results"

SHORT_PROMPT = "Say hello in French."
_LONG_PARAGRAPH = (
    "Write a detailed, thorough summary of the causes, major battles, and "
    "political aftermath of the following historical periods, covering "
    "economic factors, key figures, and long-term consequences for each: "
    "the fall of the Western Roman Empire, the Napoleonic Wars, the "
    "Industrial Revolution, the First World War, and the Cold War. "
)


async def run(
    base_url: str, n_short: int, n_long: int, short_tokens: int, long_tokens: int, long_repeat: int
) -> list[RequestResult]:
    long_prompt = _LONG_PARAGRAPH * long_repeat
    async with httpx.AsyncClient(base_url=base_url) as client:
        short_tasks = [
            stream_generate(client, SHORT_PROMPT, short_tokens, priority=1, index=i)
            for i in range(n_short)
        ]
        long_tasks = [
            stream_generate(client, long_prompt, long_tokens, priority=1, index=1000 + i)
            for i in range(n_long)
        ]
        return await asyncio.gather(*short_tasks, *long_tasks)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--label", required=True, help="tag for this run, e.g. 'chunked'")
    parser.add_argument("--n-short", type=int, default=6)
    parser.add_argument("--n-long", type=int, default=2)
    parser.add_argument("--short-tokens", type=int, default=16)
    parser.add_argument("--long-tokens", type=int, default=16)
    parser.add_argument(
        "--long-repeat", type=int, default=1, help="repeat the long paragraph N times"
    )
    args = parser.parse_args()

    results = asyncio.run(
        run(
            args.base_url,
            args.n_short,
            args.n_long,
            args.short_tokens,
            args.long_tokens,
            args.long_repeat,
        )
    )
    short_results = [r for r in results if r.index < 1000 and r.status_code == 200]
    ttfts = [r.ttft_seconds for r in short_results if r.ttft_seconds is not None]
    if not ttfts:
        raise SystemExit("no successful short-prompt results to report")

    p50, p95, p99 = percentile(ttfts, 0.5), percentile(ttfts, 0.95), percentile(ttfts, 0.99)
    print(f"label={args.label}  short-prompt TTFT  p50={p50:.4f}s  p95={p95:.4f}s  p99={p99:.4f}s")

    RESULTS_DIR.mkdir(exist_ok=True)
    out_path = RESULTS_DIR / "tail_latency.csv"
    is_new = not out_path.exists()
    with out_path.open("a", newline="") as f:
        writer = csv.writer(f)
        if is_new:
            writer.writerow(["label", "ttft_p50", "ttft_p95", "ttft_p99", "n_short", "n_long"])
        writer.writerow([args.label, p50, p95, p99, args.n_short, args.n_long])
    print(f"appended to {out_path}")


if __name__ == "__main__":
    main()
