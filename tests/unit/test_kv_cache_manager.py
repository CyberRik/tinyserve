import pytest

from tinyserve.kv_cache.manager import KVCacheManager


def test_reserve_allocates_whole_blocks() -> None:
    manager = KVCacheManager(total_tokens=64, block_size=16)

    assert manager.reserve("seq-1", n_tokens=17) is True  # needs 2 blocks (16 + 1)

    assert manager.available_blocks() == manager.total_blocks() - 2


def test_reserve_fails_when_insufficient_blocks() -> None:
    manager = KVCacheManager(total_tokens=32, block_size=16)  # 2 blocks total
    assert manager.reserve("seq-1", n_tokens=32) is True  # uses both blocks

    assert manager.reserve("seq-2", n_tokens=1) is False
    assert manager.available_blocks() == 0


def test_release_returns_blocks_to_the_free_pool() -> None:
    manager = KVCacheManager(total_tokens=32, block_size=16)
    manager.reserve("seq-1", n_tokens=32)

    manager.release("seq-1")

    assert manager.available_blocks() == manager.total_blocks()


def test_release_of_unknown_seq_id_is_a_noop() -> None:
    manager = KVCacheManager(total_tokens=32, block_size=16)

    manager.release("never-reserved")

    assert manager.available_blocks() == manager.total_blocks()


def test_reserve_twice_for_same_seq_id_raises() -> None:
    manager = KVCacheManager(total_tokens=64, block_size=16)
    manager.reserve("seq-1", n_tokens=16)

    with pytest.raises(ValueError, match="seq-1"):
        manager.reserve("seq-1", n_tokens=16)
