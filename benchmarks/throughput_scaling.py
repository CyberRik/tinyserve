"""Benchmark 4 (PRD Section 11, #4): throughput scaling.

PRD's original framing is "vary batch size cap, plot tokens/sec vs batch
size" — but n_batch is fixed at server startup (it's derived from n_ctx
in LlamaRuntime), not a per-request knob, so it can't be swept without
restarting the server. The adapted, honest version of this benchmark
instead varies *concurrent demand* (how many requests are in flight at
once, up to and past n_seq_max) and plots aggregate tokens/sec against
it — this finds the same thing the PRD benchmark wants: the point where
adding more concurrent work stops increasing throughput because either
n_seq_max (concurrency slots) or the CPU decode cost itself has
saturated.

Usage: python benchmarks/throughput_scaling.py [--base-url URL] [--concurrency 1,2,4,8,16]
"""

import argparse
import asyncio
import csv
import time
from pathlib import Path

import httpx
from _common import DEFAULT_BASE_URL, stream_generate

RESULTS_DIR = Path(__file__).parent / "results"

PROMPT = "Describe the water cycle in a few sentences."


async def run_concurrency_level(
    base_url: str, concurrency: int, max_tokens: int
) -> dict[str, float | int]:
    async with httpx.AsyncClient(base_url=base_url) as client:
        start = time.perf_counter()
        tasks = [
            stream_generate(client, PROMPT, max_tokens, priority=1, index=i)
            for i in range(concurrency)
        ]
        results = await asyncio.gather(*tasks)
        elapsed = time.perf_counter() - start

    ok = [r for r in results if r.status_code == 200]
    total_tokens = sum(r.tokens_emitted for r in ok)
    return {
        "concurrency": concurrency,
        "accepted": len(ok),
        "wall_seconds": elapsed,
        "total_tokens": total_tokens,
        "aggregate_tokens_per_second": total_tokens / elapsed if elapsed > 0 else 0.0,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--concurrency", default="1,2,4,8,16")
    parser.add_argument("--max-tokens", type=int, default=48)
    args = parser.parse_args()
    levels = [int(c) for c in args.concurrency.split(",")]

    rows = []
    print(f"{'concurrency':>11} {'accepted':>9} {'wall_s':>8} {'agg_tok/s':>10}")
    for level in levels:
        row = asyncio.run(run_concurrency_level(args.base_url, level, args.max_tokens))
        rows.append(row)
        print(
            f"{row['concurrency']:>11} {row['accepted']:>9} "
            f"{row['wall_seconds']:>8.3f} {row['aggregate_tokens_per_second']:>10.2f}"
        )

    RESULTS_DIR.mkdir(exist_ok=True)
    out_path = RESULTS_DIR / "throughput_scaling.csv"
    with out_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {out_path}")

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(
            [r["concurrency"] for r in rows],
            [r["aggregate_tokens_per_second"] for r in rows],
            marker="o",
        )
        ax.set_xlabel("concurrent requests")
        ax.set_ylabel("aggregate tokens/sec")
        ax.set_title("Throughput scaling vs concurrent demand")
        fig.tight_layout()
        plot_path = RESULTS_DIR / "throughput_scaling.png"
        fig.savefig(plot_path, dpi=120)
        print(f"wrote {plot_path}")
    except ImportError:
        print("matplotlib not installed, skipping plot")


if __name__ == "__main__":
    main()
