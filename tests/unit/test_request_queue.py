import asyncio

from tinyserve.queue.request_queue import PendingRequest, RequestQueue


def test_pop_returns_requests_in_fifo_order() -> None:
    async def run() -> list[str]:
        queue = RequestQueue()
        queue.push(PendingRequest(id="a", prompt="first", max_tokens=8))
        queue.push(PendingRequest(id="b", prompt="second", max_tokens=8))
        queue.push(PendingRequest(id="c", prompt="third", max_tokens=8))

        return [(await queue.pop()).id for _ in range(3)]

    assert asyncio.run(run()) == ["a", "b", "c"]


def test_depth_reflects_unclaimed_requests() -> None:
    async def run() -> tuple[int, int]:
        queue = RequestQueue()
        queue.push(PendingRequest(id="a", prompt="x", max_tokens=8))
        queue.push(PendingRequest(id="b", prompt="y", max_tokens=8))
        before = queue.depth()
        await queue.pop()
        after = queue.depth()
        return before, after

    before, after = asyncio.run(run())
    assert before == 2
    assert after == 1
