import asyncio

from tinyserve.streaming.stream_manager import StreamManager


async def _collect(manager: StreamManager, request_id: str) -> list[str]:
    return [chunk async for chunk in manager.subscribe(request_id)]


def test_push_and_close_delivers_all_text() -> None:
    async def run() -> list[str]:
        manager = StreamManager()
        manager.create("req-1")
        manager.push_token("req-1", b"hello ")
        manager.push_token("req-1", b"world")
        manager.close("req-1")
        return await _collect(manager, "req-1")

    assert asyncio.run(run()) == ["hello ", "world"]


def test_multi_byte_utf8_split_across_tokens_reassembles_correctly() -> None:
    # "café" - the é (U+00E9) encodes to two bytes (0xC3 0xA9); split them
    # across two separate push_token calls, the way two llama.cpp tokens would.
    encoded = "café".encode()
    first, second = encoded[:-1], encoded[-1:]

    async def run() -> list[str]:
        manager = StreamManager()
        manager.create("req-1")
        manager.push_token("req-1", first)
        manager.push_token("req-1", second)
        manager.close("req-1")
        return await _collect(manager, "req-1")

    chunks = asyncio.run(run())
    assert "".join(chunks) == "café"


def test_cancel_is_observable_before_close() -> None:
    manager = StreamManager()
    manager.create("req-1")

    assert manager.is_cancelled("req-1") is False
    manager.cancel("req-1")
    assert manager.is_cancelled("req-1") is True


def test_is_cancelled_true_for_unknown_request() -> None:
    manager = StreamManager()

    assert manager.is_cancelled("never-created") is True


def test_subscribe_removes_stream_after_completion() -> None:
    # is_cancelled() defaults to True for an unknown request_id, so this
    # doubles as a black-box check that subscribe() cleaned up its entry.
    async def run() -> bool:
        manager = StreamManager()
        manager.create("req-1")
        manager.close("req-1")
        async for _ in manager.subscribe("req-1"):
            pass
        return manager.is_cancelled("req-1")

    assert asyncio.run(run()) is True
