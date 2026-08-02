from tinyserve.admission.controller import AdmissionController
from tinyserve.kv_cache.manager import KVCacheManager


def test_admit_accepts_when_kv_budget_available() -> None:
    kv_cache = KVCacheManager(total_tokens=64, block_size=16)
    controller = AdmissionController(kv_cache, max_queue_depth=10)

    result = controller.admit("seq-1", prompt_tokens=10, max_tokens=20, queue_depth=0)

    assert result.accepted is True
    assert result.reason is None


def test_admit_rejects_when_kv_budget_exhausted() -> None:
    kv_cache = KVCacheManager(total_tokens=32, block_size=16)
    controller = AdmissionController(kv_cache, max_queue_depth=10)
    controller.admit("seq-1", prompt_tokens=16, max_tokens=16, queue_depth=0)  # takes whole budget

    result = controller.admit("seq-2", prompt_tokens=1, max_tokens=1, queue_depth=0)

    assert result.accepted is False
    assert result.reason == "kv_cache_full"


def test_admit_reserves_worst_case_prompt_plus_max_tokens() -> None:
    kv_cache = KVCacheManager(total_tokens=32, block_size=16)
    controller = AdmissionController(kv_cache, max_queue_depth=10)

    controller.admit("seq-1", prompt_tokens=1, max_tokens=16, queue_depth=0)  # rounds to 2 blocks

    assert kv_cache.available_blocks() == 0


def test_admit_rejects_when_queue_depth_at_max() -> None:
    kv_cache = KVCacheManager(total_tokens=1024, block_size=16)  # ample KV budget
    controller = AdmissionController(kv_cache, max_queue_depth=5)

    result = controller.admit("seq-1", prompt_tokens=1, max_tokens=1, queue_depth=5)

    assert result.accepted is False
    assert result.reason == "queue_full"
    # rejected before ever touching the KV budget
    assert kv_cache.available_blocks() == kv_cache.total_blocks()


def test_admit_accepts_when_queue_depth_below_max() -> None:
    kv_cache = KVCacheManager(total_tokens=1024, block_size=16)
    controller = AdmissionController(kv_cache, max_queue_depth=5)

    result = controller.admit("seq-1", prompt_tokens=1, max_tokens=1, queue_depth=4)

    assert result.accepted is True
