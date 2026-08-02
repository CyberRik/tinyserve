"""Shared HTTP client and Prometheus-text-parsing helpers for benchmark scripts.

Every benchmark in this directory is a standalone script that drives the
*real* running server over HTTP — no mocks, no internal imports of
tinyserve's own modules — because the point of a benchmark is to measure
what a client actually experiences.
"""

import time
from dataclasses import dataclass

import httpx

DEFAULT_BASE_URL = "http://127.0.0.1:8000"


@dataclass(frozen=True)
class RequestResult:
    index: int
    status_code: int
    ttft_seconds: float | None
    total_seconds: float
    tokens_emitted: int
    priority: int


async def stream_generate(
    client: httpx.AsyncClient,
    prompt: str,
    max_tokens: int,
    priority: int = 1,
    index: int = 0,
) -> RequestResult:
    start = time.perf_counter()
    ttft: float | None = None
    tokens = 0
    status = 0
    async with client.stream(
        "POST",
        "/generate",
        json={"prompt": prompt, "max_tokens": max_tokens, "priority": priority, "stream": True},
        timeout=httpx.Timeout(120.0),
    ) as response:
        status = response.status_code
        if status == 200:
            async for line in response.aiter_lines():
                if not line.startswith("data:"):
                    continue
                tokens += 1
                if ttft is None:
                    ttft = time.perf_counter() - start
        else:
            await response.aread()
    total = time.perf_counter() - start
    return RequestResult(index, status, ttft, total, tokens, priority)


def parse_metric(text: str, name: str) -> list[tuple[dict[str, str], float]]:
    """Parse every sample line for one metric family out of /metrics text.

    Prometheus exposition format: `name{label="value",...} 1.23`. Histogram
    families expose `_bucket`, `_sum`, `_count` suffixes as separate metric
    names, so callers pass the exact family name they want (e.g.
    "ttft_seconds_bucket", not "ttft_seconds").
    """
    samples: list[tuple[dict[str, str], float]] = []
    prefix = name
    for line in text.splitlines():
        if line.startswith("#") or not line.startswith(prefix):
            continue
        rest = line[len(prefix) :]
        if rest and rest[0] not in " {":
            continue  # a different metric that happens to share this prefix
        labels: dict[str, str] = {}
        if rest.startswith("{"):
            end = rest.index("}")
            label_str, rest = rest[1:end], rest[end + 1 :]
            for pair in label_str.split(","):
                if not pair:
                    continue
                key, value = pair.split("=", 1)
                labels[key] = value.strip('"')
        samples.append((labels, float(rest.strip())))
    return samples


def fetch_metrics_text(client: httpx.Client) -> str:
    response = client.get("/metrics")
    response.raise_for_status()
    return response.text


def percentile(values: list[float], p: float) -> float:
    if not values:
        return float("nan")
    ordered = sorted(values)
    rank = (len(ordered) - 1) * p
    lower = int(rank)
    upper = min(lower + 1, len(ordered) - 1)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (rank - lower)
