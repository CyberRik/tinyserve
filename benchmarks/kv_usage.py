"""Benchmark 8 (PRD Section 11, #8): memory/KV usage over a sustained run.

Tracks kv_blocks_used vs kv_blocks_free over time under continuous
traffic, and reports how many blocks were ever used concurrently (the
practical "how much of your KV budget does this workload actually need"
number). Note: TinyServe's KV Cache Manager (tinyserve/kv_cache/manager.py)
does not implement the LRU eviction PRD Section 8 describes as a stretch
detail — released blocks return to the free pool immediately, nothing is
kept warm — so there is no fragmentation_ratio metric or eviction-event
count to report here; this benchmark measures block occupancy only, and
that gap is documented in docs/architecture.md, not silently glossed over.

Usage: python benchmarks/kv_usage.py [--base-url URL] [--rps 4] [--duration 20]
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
    "The quick brown fox",
    "In a world where technology",
    "The recipe calls for",
    "According to recent research",
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
    with httpx.Client(base_url=base_url) as client:
        start = time.monotonic()
        while time.monotonic() - start < duration + sample_interval:
            t = time.monotonic() - start
            text = fetch_metrics_text(client)
            free = parse_metric(text, "kv_blocks_free")
            used = parse_metric(text, "kv_blocks_used")
            rows.append(
                {
                    "t": round(t, 2),
                    "kv_blocks_free": free[0][1] if free else None,
                    "kv_blocks_used": used[0][1] if used else None,
                }
            )
            time.sleep(sample_interval)


async def run(
    base_url: str, rps: float, duration: float, max_tokens: int, sample_interval: float
) -> list[Sample]:
    rows: list[Sample] = []
    await asyncio.gather(
        load_generator(base_url, rps, duration, max_tokens),
        asyncio.to_thread(sample_loop, base_url, duration, sample_interval, rows),
    )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--rps", type=float, default=4.0)
    parser.add_argument("--duration", type=float, default=20.0)
    parser.add_argument("--max-tokens", type=int, default=32)
    parser.add_argument("--sample-interval", type=float, default=0.5)
    args = parser.parse_args()

    rows = asyncio.run(
        run(args.base_url, args.rps, args.duration, args.max_tokens, args.sample_interval)
    )

    RESULTS_DIR.mkdir(exist_ok=True)
    out_path = RESULTS_DIR / "kv_usage.csv"
    with out_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    used = [r["kv_blocks_used"] for r in rows if r["kv_blocks_used"] is not None]
    total = None
    free0 = next((r["kv_blocks_free"] for r in rows if r["kv_blocks_free"] is not None), None)
    if used and free0 is not None:
        total = used[0] + free0
    print(f"{len(rows)} samples over ~{args.duration:.0f}s")
    if used:
        print(f"kv_blocks_used: min={min(used):.0f} max={max(used):.0f} end={used[-1]:.0f}"
              + (f"  (of {total:.0f} total)" if total else ""))
    print(f"wrote {out_path}")

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(7, 4))
        ts = [r["t"] for r in rows]
        ax.plot(ts, [r["kv_blocks_used"] for r in rows], label="used")  # type: ignore[arg-type]
        ax.plot(ts, [r["kv_blocks_free"] for r in rows], label="free")  # type: ignore[arg-type]
        ax.set_xlabel("time (s)")
        ax.set_ylabel("KV blocks")
        ax.set_title("KV block occupancy over a sustained run")
        ax.legend()
        fig.tight_layout()
        plot_path = RESULTS_DIR / "kv_usage.png"
        fig.savefig(plot_path, dpi=120)
        print(f"wrote {plot_path}")
    except ImportError:
        print("matplotlib not installed, skipping plot")


if __name__ == "__main__":
    main()
