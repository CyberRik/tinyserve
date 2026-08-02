"""Admission Controller (PRD Section 6.1): the gatekeeper before anything
touches the queue or KV memory.

Two independent checks, either of which can reject a request fast rather
than let it degrade latency for everyone else once accepted:

1. Queue-depth backpressure — reject once the waiting pool is already at
   its configured max, rather than let it grow unboundedly under a burst
   of requests too small individually to ever hit the KV budget.
2. KV budget — reserves the worst-case token count (prompt + max_tokens)
   up front so a sequence that's admitted is guaranteed the room to run to
   completion.

Mirrors Ancora's admission engine rejecting work it can't durably
guarantee capacity for (PRD Section 0), applied here to queue capacity and
KV-cache memory instead. A token-bucket rate limiter is PRD Section 6.1's
third piece of Admission Controller state; it is not implemented — see
docs/architecture.md's gap list.
"""

from dataclasses import dataclass

from tinyserve.kv_cache.manager import KVCacheManager


@dataclass(frozen=True)
class AdmissionResult:
    accepted: bool
    reason: str | None = None


class AdmissionController:
    def __init__(self, kv_cache: KVCacheManager, max_queue_depth: int) -> None:
        self._kv_cache = kv_cache
        self._max_queue_depth = max_queue_depth

    def admit(
        self, seq_id: str, prompt_tokens: int, max_tokens: int, queue_depth: int
    ) -> AdmissionResult:
        if queue_depth >= self._max_queue_depth:
            return AdmissionResult(accepted=False, reason="queue_full")
        worst_case_tokens = prompt_tokens + max_tokens
        if self._kv_cache.reserve(seq_id, worst_case_tokens):
            return AdmissionResult(accepted=True)
        return AdmissionResult(accepted=False, reason="kv_cache_full")
