"""Benchmark 2 (PRD Section 11, #2): burst traffic and admission backpressure.

Fires N requests simultaneously for a sweep of burst sizes and measures:
how many get admitted vs rejected with a 503, and how TTFT for the
admitted ones degrades as burst size grows. This is what validates (or
breaks) the Admission Controller's backpressure story.

Usage: python benchmarks/burst.py [--base-url URL] [--sizes 1,2,4,8,16,32]
"""

import argparse
import asyncio
import csv
import statistics
from pathlib import Path

import httpx
from _common import DEFAULT_BASE_URL, RequestResult, percentile, stream_generate

RESULTS_DIR = Path(__file__).parent / "results"

PROMPT = "Explain the difference between TCP and UDP in one paragraph."


async def run_burst(base_url: str, size: int, max_tokens: int) -> list[RequestResult]:
    async with httpx.AsyncClient(base_url=base_url) as client:
        tasks = [
            stream_generate(client, PROMPT, max_tokens, priority=1, index=i) for i in range(size)
        ]
        return await asyncio.gather(*tasks)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--sizes", default="1,2,4,8,16,32")
    parser.add_argument("--max-tokens", type=int, default=32)
    args = parser.parse_args()
    sizes = [int(s) for s in args.sizes.split(",")]

    RESULTS_DIR.mkdir(exist_ok=True)
    out_path = RESULTS_DIR / "burst.csv"
    rows = []
    print(f"{'burst':>6} {'accepted':>9} {'rejected':>9} {'ttft_p50':>9} {'ttft_p95':>9}")
    for size in sizes:
        results = asyncio.run(run_burst(args.base_url, size, args.max_tokens))
        accepted = [r for r in results if r.status_code == 200]
        rejected = [r for r in results if r.status_code == 503]
        other = [r for r in results if r.status_code not in (200, 503)]
        ttfts = [r.ttft_seconds for r in accepted if r.ttft_seconds is not None]
        p50 = percentile(ttfts, 0.5) if ttfts else float("nan")
        p95 = percentile(ttfts, 0.95) if ttfts else float("nan")
        rows.append(
            {
                "burst_size": size,
                "accepted": len(accepted),
                "rejected": len(rejected),
                "other_status": len(other),
                "ttft_p50": p50,
                "ttft_p95": p95,
                "ttft_mean": statistics.mean(ttfts) if ttfts else float("nan"),
            }
        )
        print(f"{size:>6} {len(accepted):>9} {len(rejected):>9} {p50:>9.4f} {p95:>9.4f}")

    with out_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {out_path}")

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        plot_sizes = [r["burst_size"] for r in rows]
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
        ax1.plot(plot_sizes, [r["accepted"] for r in rows], marker="o", label="accepted")
        ax1.plot(plot_sizes, [r["rejected"] for r in rows], marker="o", label="rejected")
        ax1.set_xlabel("burst size")
        ax1.set_ylabel("request count")
        ax1.set_title("Admission outcome vs burst size")
        ax1.legend()

        ax2.plot(plot_sizes, [r["ttft_p50"] for r in rows], marker="o", label="p50")
        ax2.plot(plot_sizes, [r["ttft_p95"] for r in rows], marker="o", label="p95")
        ax2.set_xlabel("burst size")
        ax2.set_ylabel("TTFT (s)")
        ax2.set_title("TTFT vs burst size (accepted only)")
        ax2.legend()

        fig.tight_layout()
        plot_path = RESULTS_DIR / "burst.png"
        fig.savefig(plot_path, dpi=120)
        print(f"wrote {plot_path}")
    except ImportError:
        print("matplotlib not installed, skipping plot")


if __name__ == "__main__":
    main()
