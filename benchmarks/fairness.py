"""Benchmark 6 (PRD Section 11, #6): scheduler fairness under skewed load.

N low-priority + M high-priority concurrent streams, measuring per-class
mean queue-wait and TTFT. Run this script three times, once per server
instance configured with a different TINYSERVE_SCHEDULING_POLICY, to
produce the "here's the graph proving WFQ prevents starvation, here's
FIFO/Priority failing to" artifact the PRD calls for:

    TINYSERVE_SCHEDULING_POLICY=fifo     uv run uvicorn tinyserve.api.app:app --port 8001
    TINYSERVE_SCHEDULING_POLICY=priority uv run uvicorn tinyserve.api.app:app --port 8002
    TINYSERVE_SCHEDULING_POLICY=wfq      uv run uvicorn tinyserve.api.app:app --port 8003

    python benchmarks/fairness.py --base-url http://127.0.0.1:8001 --label fifo
    python benchmarks/fairness.py --base-url http://127.0.0.1:8002 --label priority
    python benchmarks/fairness.py --base-url http://127.0.0.1:8003 --label wfq

Usage: python benchmarks/fairness.py --base-url URL --label wfq
"""

import argparse
import asyncio
import csv
import statistics
from pathlib import Path

import httpx
from _common import DEFAULT_BASE_URL, RequestResult, stream_generate

RESULTS_DIR = Path(__file__).parent / "results"

PROMPT = "Summarize the plot of a mystery novel in two sentences."

LOW_PRIORITY = 1
HIGH_PRIORITY = 5


async def run(
    base_url: str, n_low: int, n_high: int, max_tokens: int
) -> list[RequestResult]:
    async with httpx.AsyncClient(base_url=base_url) as client:
        low_tasks = [
            stream_generate(client, PROMPT, max_tokens, priority=LOW_PRIORITY, index=i)
            for i in range(n_low)
        ]
        high_tasks = [
            stream_generate(client, PROMPT, max_tokens, priority=HIGH_PRIORITY, index=1000 + i)
            for i in range(n_high)
        ]
        return await asyncio.gather(*low_tasks, *high_tasks)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--label", required=True, help="tag for this run, e.g. the policy name")
    parser.add_argument("--n-low", type=int, default=12)
    parser.add_argument("--n-high", type=int, default=4)
    parser.add_argument("--max-tokens", type=int, default=24)
    args = parser.parse_args()

    results = asyncio.run(run(args.base_url, args.n_low, args.n_high, args.max_tokens))
    low = [r for r in results if r.index < 1000 and r.status_code == 200]
    high = [r for r in results if r.index >= 1000 and r.status_code == 200]
    low_ttfts = [r.ttft_seconds for r in low if r.ttft_seconds is not None]
    high_ttfts = [r.ttft_seconds for r in high if r.ttft_seconds is not None]

    low_mean = statistics.mean(low_ttfts) if low_ttfts else float("nan")
    high_mean = statistics.mean(high_ttfts) if high_ttfts else float("nan")
    low_max = max(low_ttfts) if low_ttfts else float("nan")
    high_max = max(high_ttfts) if high_ttfts else float("nan")

    print(f"label={args.label}")
    print(f"  low-priority  (n={len(low)}): mean TTFT={low_mean:.4f}s  max TTFT={low_max:.4f}s")
    print(f"  high-priority (n={len(high)}): mean TTFT={high_mean:.4f}s  max TTFT={high_max:.4f}s")
    print(f"  high/low mean TTFT ratio: {low_mean / high_mean if high_mean else float('nan'):.2f}x")

    RESULTS_DIR.mkdir(exist_ok=True)
    out_path = RESULTS_DIR / "fairness.csv"
    is_new = not out_path.exists()
    with out_path.open("a", newline="") as f:
        writer = csv.writer(f)
        if is_new:
            writer.writerow(
                ["label", "low_mean_ttft", "low_max_ttft", "high_mean_ttft", "high_max_ttft",
                 "n_low", "n_high"]
            )
        writer.writerow(
            [args.label, low_mean, low_max, high_mean, high_max, args.n_low, args.n_high]
        )
    print(f"appended to {out_path}")


if __name__ == "__main__":
    main()
