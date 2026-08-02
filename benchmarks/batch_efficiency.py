"""Benchmark 7 (PRD Section 11, #7): batch efficiency under varying request sizes.

batch_utilization (actual tokens processed / max batch capacity per tick)
is the single best "is the scheduler doing its job" number. This script
drives three request-size distributions (all-short, all-long, mixed) and
reports the average batch_utilization observed under each, to see
whether small requests get wastefully batched alone instead of packed
efficiently alongside others.

Usage: python benchmarks/batch_efficiency.py [--base-url URL]
"""

import argparse
import asyncio
import csv
from pathlib import Path

import httpx
from _common import DEFAULT_BASE_URL, fetch_metrics_text, parse_metric, stream_generate

RESULTS_DIR = Path(__file__).parent / "results"

SHORT_PROMPT = "The sky is"
LONG_PROMPT = (
    "Write a thorough essay covering the economic, political, and social "
    "consequences of the printing press, the steam engine, and the internet. "
) * 3


def read_batch_utilization_delta(
    client: httpx.Client, prev: tuple[float, float]
) -> tuple[float, tuple[float, float]]:
    text = fetch_metrics_text(client)
    util_sum = parse_metric(text, "batch_utilization_sum")
    util_count = parse_metric(text, "batch_utilization_count")
    cur_sum = util_sum[0][1] if util_sum else prev[0]
    cur_count = util_count[0][1] if util_count else prev[1]
    delta_count = cur_count - prev[1]
    avg = (cur_sum - prev[0]) / delta_count if delta_count > 0 else float("nan")
    return avg, (cur_sum, cur_count)


async def run_distribution(base_url: str, prompts: list[str], max_tokens: int) -> None:
    async with httpx.AsyncClient(base_url=base_url) as client:
        tasks = [
            stream_generate(client, p, max_tokens, priority=1, index=i)
            for i, p in enumerate(prompts)
        ]
        await asyncio.gather(*tasks)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--n", type=int, default=8, help="requests per distribution")
    parser.add_argument("--max-tokens", type=int, default=24)
    args = parser.parse_args()

    distributions = {
        "all_short": [SHORT_PROMPT] * args.n,
        "all_long": [LONG_PROMPT] * args.n,
        "mixed": [SHORT_PROMPT if i % 2 == 0 else LONG_PROMPT for i in range(args.n)],
    }

    rows = []
    with httpx.Client(base_url=args.base_url) as sync_client:
        prev = (0.0, 0.0)
        text = fetch_metrics_text(sync_client)
        util_sum = parse_metric(text, "batch_utilization_sum")
        util_count = parse_metric(text, "batch_utilization_count")
        if util_sum and util_count:
            prev = (util_sum[0][1], util_count[0][1])

        for label, prompts in distributions.items():
            asyncio.run(run_distribution(args.base_url, prompts, args.max_tokens))
            avg, prev = read_batch_utilization_delta(sync_client, prev)
            rows.append({"distribution": label, "n": args.n, "avg_batch_utilization": avg})
            print(f"{label:>10}: avg batch_utilization = {avg:.4f}")

    RESULTS_DIR.mkdir(exist_ok=True)
    out_path = RESULTS_DIR / "batch_efficiency.csv"
    with out_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
