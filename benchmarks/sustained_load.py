"""Benchmark 3 (PRD Section 11, #3): continuous/sustained load.

Fires requests at a fixed rate for a configured duration and samples
/metrics periodically, watching whether batch_utilization stays high and
KV usage behaves sanely under continuous load (the core continuous-
batching claim, checked empirically instead of assumed).

Usage: python benchmarks/sustained_load.py [--base-url URL] [--rps 3] [--duration 30]
"""

import argparse
import asyncio
import csv
import time
from pathlib import Path

import httpx
from _common import DEFAULT_BASE_URL, fetch_metrics_text, parse_metric, stream_generate

RESULTS_DIR = Path(__file__).parent / "results"

Sample = dict[str, float | int | None]

PROMPTS = [
    "The capital of France is",
    "Once upon a time in a distant galaxy",
    "def fibonacci(n):",
    "In machine learning, gradient descent",
]


async def load_generator(base_url: str, rps: float, duration: float, max_tokens: int) -> None:
    interval = 1.0 / rps
    async with httpx.AsyncClient(base_url=base_url) as client:
        end = time.monotonic() + duration
        i = 0
        tasks = []
        while time.monotonic() < end:
            prompt = PROMPTS[i % len(PROMPTS)]
            tasks.append(
                asyncio.create_task(
                    stream_generate(client, prompt, max_tokens, priority=1, index=i)
                )
            )
            i += 1
            await asyncio.sleep(interval)
        await asyncio.gather(*tasks, return_exceptions=True)


def sample_loop(base_url: str, duration: float, sample_interval: float, rows: list[Sample]) -> None:
    prev_sum = 0.0
    prev_count = 0.0
    with httpx.Client(base_url=base_url) as client:
        start = time.monotonic()
        while time.monotonic() - start < duration + sample_interval:
            t = time.monotonic() - start
            text = fetch_metrics_text(client)
            free = parse_metric(text, "kv_blocks_free")
            used = parse_metric(text, "kv_blocks_used")
            depth = parse_metric(text, "queue_depth")
            util_sum = parse_metric(text, "batch_utilization_sum")
            util_count = parse_metric(text, "batch_utilization_count")
            cur_sum = util_sum[0][1] if util_sum else prev_sum
            cur_count = util_count[0][1] if util_count else prev_count
            delta_count = cur_count - prev_count
            # interval-local average, not cumulative-since-start — a
            # cumulative average would flatten out and hide any drift.
            interval_avg = (cur_sum - prev_sum) / delta_count if delta_count > 0 else None
            prev_sum, prev_count = cur_sum, cur_count
            rows.append(
                {
                    "t": round(t, 2),
                    "kv_blocks_free": free[0][1] if free else None,
                    "kv_blocks_used": used[0][1] if used else None,
                    "queue_depth": depth[0][1] if depth else None,
                    "batch_utilization_avg": interval_avg,
                }
            )
            time.sleep(sample_interval)


async def run(
    base_url: str, rps: float, duration: float, max_tokens: int, sample_interval: float
) -> list[Sample]:
    rows: list[Sample] = []
    sampler = asyncio.to_thread(sample_loop, base_url, duration, sample_interval, rows)
    await asyncio.gather(
        load_generator(base_url, rps, duration, max_tokens),
        sampler,
    )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--rps", type=float, default=3.0)
    parser.add_argument("--duration", type=float, default=30.0)
    parser.add_argument("--max-tokens", type=int, default=32)
    parser.add_argument("--sample-interval", type=float, default=1.0)
    args = parser.parse_args()

    rows = asyncio.run(
        run(args.base_url, args.rps, args.duration, args.max_tokens, args.sample_interval)
    )

    RESULTS_DIR.mkdir(exist_ok=True)
    out_path = RESULTS_DIR / "sustained_load.csv"
    with out_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    utils = [r["batch_utilization_avg"] for r in rows if r["batch_utilization_avg"] is not None]
    used = [r["kv_blocks_used"] for r in rows if r["kv_blocks_used"] is not None]
    print(f"{len(rows)} samples over ~{args.duration:.0f}s at {args.rps} req/s")
    if utils:
        print(f"batch_utilization: start={utils[0]:.3f} end={utils[-1]:.3f} max={max(utils):.3f}")
    if used:
        print(f"kv_blocks_used: start={used[0]:.0f} end={used[-1]:.0f} max={max(used):.0f}")
    print(f"wrote {out_path}")

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
        ts = [r["t"] for r in rows]
        ax1.plot(ts, [r["kv_blocks_used"] for r in rows], label="used")
        ax1.plot(ts, [r["kv_blocks_free"] for r in rows], label="free")
        ax1.set_xlabel("time (s)")
        ax1.set_ylabel("KV blocks")
        ax1.set_title("KV block usage over sustained load")
        ax1.legend()

        ax2.plot(ts, [r["batch_utilization_avg"] for r in rows])
        ax2.set_xlabel("time (s)")
        ax2.set_ylabel("avg batch_utilization")
        ax2.set_title("Batch utilization over sustained load")

        fig.tight_layout()
        plot_path = RESULTS_DIR / "sustained_load.png"
        fig.savefig(plot_path, dpi=120)
        print(f"wrote {plot_path}")
    except ImportError:
        print("matplotlib not installed, skipping plot")


if __name__ == "__main__":
    main()
