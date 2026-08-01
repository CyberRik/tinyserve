"""Admission Controller (PRD Section 6.1): the KV-budget gatekeeper.

Reserves the worst-case KV budget (prompt + max_tokens) up front so a
sequence that's admitted is guaranteed the room to run to completion —
mirrors Ancora's admission engine rejecting work it can't durably guarantee
capacity for (PRD Section 0), applied here to KV-cache memory instead.
"""

from dataclasses import dataclass

from tinyserve.kv_cache.manager import KVCacheManager


@dataclass(frozen=True)
class AdmissionResult:
    accepted: bool
    reason: str | None = None


class AdmissionController:
    def __init__(self, kv_cache: KVCacheManager) -> None:
        self._kv_cache = kv_cache

    def admit(self, seq_id: str, prompt_tokens: int, max_tokens: int) -> AdmissionResult:
        worst_case_tokens = prompt_tokens + max_tokens
        if self._kv_cache.reserve(seq_id, worst_case_tokens):
            return AdmissionResult(accepted=True)
        return AdmissionResult(accepted=False, reason="kv_cache_full")
