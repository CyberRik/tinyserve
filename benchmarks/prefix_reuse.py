"""Benchmark 8: prefix reuse under a shared system prompt.

The workload prefix caching exists for, and the one every chat deployment
actually runs: many requests sharing a long system preamble and differing only
in a short tail. Fires a wave of such requests and reports what fraction of
prompt tokens never reached llama_decode(), plus TTFT.

Run it twice against servers started with TINYSERVE_PREFIX_CACHE_ENABLED=true
and =false to get the A/B; the script measures one server per invocation and
writes a row tagged with whichever backend that server reports.

    TINYSERVE_PREFIX_CACHE_ENABLED=true  uvicorn tinyserve.api.app:app
    python benchmarks/prefix_reuse.py --label on

TTFT is reported but is NOT the headline here, and the distinction matters.
docs/profiling-notes.md measured 97.2% of wall time inside llama_decode(), so
on a 0.5B CPU build with a short preamble the saved prefill can be small enough
to sit inside run-to-run noise. The claim this benchmark supports is the token
counter -- work provably not done -- and prefill_tokens_reused_total is exact,
not sampled. TTFT is what that saving is worth on *this* box, which is a
different and much more load-dependent question.

Usage: python benchmarks/prefix_reuse.py [--base-url URL] [--n 16] [--label on]
"""

import argparse
import asyncio
import csv
import statistics
from pathlib import Path

import httpx
from _common import (
    DEFAULT_BASE_URL,
    RequestResult,
    fetch_metrics_text,
    parse_metric,
    percentile,
    stream_generate,
)

RESULTS_DIR = Path(__file__).parent / "results"

# Long enough to span several 16-token KV blocks, so there is a real prefix to
# share. A preamble shorter than one block is unreusable by construction.
SYSTEM_PROMPT = (
    "You are a meticulous technical assistant embedded in a documentation "
    "pipeline. Answer only from what the user provides, never speculate, keep "
    "responses to a single short paragraph, prefer concrete nouns over "
    "abstractions, and never begin a reply with a restatement of the question. "
    "If the question is ambiguous, say so plainly instead of guessing. "
    "Here is the user's question: "
)

QUESTIONS = [
    "What is the difference between TCP and UDP?",
    "Why does a hash map degrade to linear time?",
    "What does fsync actually guarantee?",
    "How does copy-on-write reduce fork cost?",
    "What is head-of-line blocking?",
    "Why is UTF-8 self-synchronising?",
    "What problem does a bloom filter solve?",
    "How does TLS session resumption work?",
]


def _counter(text: str, name: str) -> float:
    return sum(value for _, value in parse_metric(text, name))


def _backend(text: str) -> str:
    for labels, value in parse_metric(text, "prefix_cache_backend_info"):
        if value == 1.0:
            return labels.get("backend", "unknown")
    return "unknown"


async def run_wave(base_url: str, n: int, max_tokens: int) -> list[RequestResult]:
    """One wave of concurrent requests sharing SYSTEM_PROMPT.

    Concurrent rather than sequential on purpose: it is the harder case. Every
    request in a simultaneous wave looks up the prefix before any of them has
    finished, so reuse only happens at all because the cache publishes at
    admission rather than at completion.
    """
    async with httpx.AsyncClient(base_url=base_url) as client:
        tasks = [
            stream_generate(
                client,
                SYSTEM_PROMPT + QUESTIONS[i % len(QUESTIONS)],
                max_tokens,
                priority=1,
                index=i,
            )
            for i in range(n)
        ]
        return await asyncio.gather(*tasks)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--n", type=int, default=16, help="requests per wave")
    parser.add_argument("--waves", type=int, default=3)
    parser.add_argument("--max-tokens", type=int, default=32)
    parser.add_argument("--label", default="", help="tag for this server config, e.g. on/off")
    args = parser.parse_args()

    RESULTS_DIR.mkdir(exist_ok=True)

    with httpx.Client(base_url=args.base_url) as client:
        before = fetch_metrics_text(client)
        backend = _backend(before)
        # Every counter is deltaed against this snapshot: a server that has
        # already served traffic would otherwise fold its whole history into
        # this run's numbers, which is exactly how a second invocation against
        # the same server silently reports a wrong reuse rate.
        prefill_before = _counter(before, "prefill_tokens_total")
        reused_before = _counter(before, "prefill_tokens_reused_total")
        hits_before = _counter(before, 'prefix_cache_lookups_total{outcome="hit"}')
        misses_before = _counter(before, 'prefix_cache_lookups_total{outcome="miss"}')
        stale_before = _counter(before, 'prefix_cache_lookups_total{outcome="stale_donor"}')

    results: list[RequestResult] = []
    for _ in range(args.waves):
        results.extend(asyncio.run(run_wave(args.base_url, args.n, args.max_tokens)))

    with httpx.Client(base_url=args.base_url) as client:
        after = fetch_metrics_text(client)
        prefill = _counter(after, "prefill_tokens_total") - prefill_before
        reused = _counter(after, "prefill_tokens_reused_total") - reused_before
        hits = _counter(after, 'prefix_cache_lookups_total{outcome="hit"}') - hits_before
        misses = _counter(after, 'prefix_cache_lookups_total{outcome="miss"}') - misses_before
        stale = _counter(after, 'prefix_cache_lookups_total{outcome="stale_donor"}') - stale_before
        nodes = _counter(after, "prefix_cache_nodes")

    ok = [r for r in results if r.status_code == 200 and r.ttft_seconds is not None]
    ttfts = [r.ttft_seconds for r in ok if r.ttft_seconds is not None]
    reuse_rate = (reused / prefill * 100.0) if prefill else 0.0

    print(f"backend            {backend}")
    print(f"requests           {len(ok)}/{len(results)} completed")
    print(f"prompt tokens      {prefill:.0f}")
    print(f"  reused           {reused:.0f}  ({reuse_rate:.1f}% never reached llama_decode)")
    print(f"lookups            hit={hits:.0f} miss={misses:.0f} stale_donor={stale:.0f}")
    print(f"tree nodes         {nodes:.0f}")
    if ttfts:
        p50, p95 = statistics.median(ttfts), percentile(ttfts, 0.95)
        print(f"ttft p50/p95       {p50:.3f}s / {p95:.3f}s")

    out_path = RESULTS_DIR / "prefix_reuse.csv"
    write_header = not out_path.exists()
    with out_path.open("a", newline="") as handle:
        writer = csv.writer(handle)
        if write_header:
            writer.writerow(
                [
                    "label",
                    "backend",
                    "requests",
                    "prompt_tokens",
                    "reused_tokens",
                    "reuse_rate_pct",
                    "hits",
                    "misses",
                    "stale_donor",
                    "ttft_p50",
                    "ttft_p95",
                ]
            )
        writer.writerow(
            [
                args.label,
                backend,
                len(ok),
                f"{prefill:.0f}",
                f"{reused:.0f}",
                f"{reuse_rate:.2f}",
                f"{hits:.0f}",
                f"{misses:.0f}",
                f"{stale:.0f}",
                f"{statistics.median(ttfts):.4f}" if ttfts else "",
                f"{percentile(ttfts, 0.95):.4f}" if ttfts else "",
            ]
        )
    print(f"\nappended -> {out_path}")


if __name__ == "__main__":
    main()
