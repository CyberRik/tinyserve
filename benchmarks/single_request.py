"""Benchmark 1 (PRD Section 11, #1): single-request latency, no contention.

Establishes the floor: what TTFT and tokens/sec look like when a request
has the whole batch slot to itself, nothing else in flight. Every other
benchmark's numbers are read relative to this one.

Usage: python benchmarks/single_request.py [--base-url URL] [--n 10]
"""

import argparse
import asyncio
import csv
import statistics
from pathlib import Path

import httpx
from _common import DEFAULT_BASE_URL, RequestResult, percentile, stream_generate

RESULTS_DIR = Path(__file__).parent / "results"

PROMPT = "The history of the Roman Empire began"


async def run(base_url: str, n: int, max_tokens: int) -> list[RequestResult]:
    results: list[RequestResult] = []
    async with httpx.AsyncClient(base_url=base_url) as client:
        for i in range(n):
            result = await stream_generate(client, PROMPT, max_tokens, priority=1, index=i)
            results.append(result)
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--n", type=int, default=10, help="number of sequential requests")
    parser.add_argument("--max-tokens", type=int, default=64)
    args = parser.parse_args()

    results = asyncio.run(run(args.base_url, args.n, args.max_tokens))
    ok = [r for r in results if r.status_code == 200]
    if not ok:
        raise SystemExit(f"all {len(results)} requests failed (non-200 status)")

    ttfts = [r.ttft_seconds for r in ok if r.ttft_seconds is not None]
    tps = [r.tokens_emitted / r.total_seconds for r in ok if r.total_seconds > 0]

    RESULTS_DIR.mkdir(exist_ok=True)
    out_path = RESULTS_DIR / "single_request.csv"
    with out_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["index", "status_code", "ttft_seconds", "total_seconds", "tokens_emitted"])
        for r in results:
            writer.writerow(
                [r.index, r.status_code, r.ttft_seconds, r.total_seconds, r.tokens_emitted]
            )

    print(f"n={len(ok)}/{len(results)} succeeded")
    print(
        f"TTFT     mean={statistics.mean(ttfts):.4f}s  p50={percentile(ttfts, 0.5):.4f}s  "
        f"p95={percentile(ttfts, 0.95):.4f}s"
    )
    print(
        f"tok/sec  mean={statistics.mean(tps):.2f}  p50={percentile(tps, 0.5):.2f}  "
        f"p95={percentile(tps, 0.95):.2f}"
    )
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
