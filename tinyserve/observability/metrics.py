"""Prometheus-style metrics (PRD Section 10).

Every other component emits into this rather than owning its own logging;
this module owns no business logic, only instrument definitions — callers
decide when and what to record.
"""

from prometheus_client import CollectorRegistry, Counter, Gauge, Histogram


class Metrics:
    def __init__(self, registry: CollectorRegistry | None = None) -> None:
        self.registry = registry or CollectorRegistry()

        self.admission_accepted_total = Counter(
            "admission_accepted_total",
            "Requests accepted by the Admission Controller",
            registry=self.registry,
        )
        self.admission_rejected_total = Counter(
            "admission_rejected_total",
            "Requests rejected by the Admission Controller",
            ["reason"],
            registry=self.registry,
        )
        self.queue_depth = Gauge(
            "queue_depth",
            "Requests currently waiting in the Request Queue",
            registry=self.registry,
        )
        self.queue_wait_seconds = Histogram(
            "queue_wait_seconds",
            "Time a request spent waiting before claiming a concurrency slot",
            registry=self.registry,
        )
        self.batch_size_tokens = Histogram(
            "batch_size_tokens", "Tokens processed per batch tick", registry=self.registry
        )
        self.batch_size_requests = Histogram(
            "batch_size_requests",
            "Distinct sequences included per batch tick",
            registry=self.registry,
        )
        self.batch_utilization = Histogram(
            "batch_utilization",
            "Fraction of max batch capacity used per tick",
            registry=self.registry,
        )
        self.kv_blocks_free = Gauge(
            "kv_blocks_free", "Free KV-cache blocks", registry=self.registry
        )
        self.kv_blocks_used = Gauge(
            "kv_blocks_used", "Used KV-cache blocks", registry=self.registry
        )
        self.ttft_seconds = Histogram(
            "ttft_seconds", "Time to first token, from request arrival", registry=self.registry
        )
        self.inter_token_latency_seconds = Histogram(
            "inter_token_latency_seconds",
            "Gap between consecutive tokens of one request",
            registry=self.registry,
        )
        self.tokens_per_second = Histogram(
            "tokens_per_second", "Per-request generation throughput", registry=self.registry
        )
        self.decode_step_duration_seconds = Histogram(
            "decode_step_duration_seconds",
            "Wall time of each llama_decode() call",
            registry=self.registry,
        )
        self.scheduler_policy_decision_total = Counter(
            "scheduler_policy_decision_total",
            "Scheduling admission decisions",
            ["policy", "outcome"],
            registry=self.registry,
        )
        self.cancellations_total = Counter(
            "cancellations_total",
            "Requests cancelled or timed out before normal completion",
            ["reason"],
            registry=self.registry,
        )
