from tinyserve.admission.controller import AdmissionController
from tinyserve.kv_cache.manager import KVCacheManager


def test_admit_accepts_when_kv_budget_available() -> None:
    controller = AdmissionController(KVCacheManager(total_tokens=64, block_size=16))

    result = controller.admit("seq-1", prompt_tokens=10, max_tokens=20)

    assert result.accepted is True
    assert result.reason is None


def test_admit_rejects_when_kv_budget_exhausted() -> None:
    kv_cache = KVCacheManager(total_tokens=32, block_size=16)
    controller = AdmissionController(kv_cache)
    controller.admit("seq-1", prompt_tokens=16, max_tokens=16)  # takes the whole budget

    result = controller.admit("seq-2", prompt_tokens=1, max_tokens=1)

    assert result.accepted is False
    assert result.reason == "kv_cache_full"


def test_admit_reserves_worst_case_prompt_plus_max_tokens() -> None:
    kv_cache = KVCacheManager(total_tokens=32, block_size=16)
    controller = AdmissionController(kv_cache)

    controller.admit("seq-1", prompt_tokens=1, max_tokens=16)  # rounds up to 2 blocks

    assert kv_cache.available_blocks() == 0
