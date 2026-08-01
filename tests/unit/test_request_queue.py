import asyncio

from tinyserve.queue.request_queue import PendingRequest, RequestQueue


def test_waiting_returns_all_queued_requests() -> None:
    queue = RequestQueue()
    queue.push(PendingRequest(id="a", prompt_tokens=[1], max_tokens=8))
    queue.push(PendingRequest(id="b", prompt_tokens=[2], max_tokens=8))

    assert {r.id for r in queue.waiting()} == {"a", "b"}


def test_remove_takes_a_request_out_of_the_waiting_pool() -> None:
    queue = RequestQueue()
    queue.push(PendingRequest(id="a", prompt_tokens=[1], max_tokens=8))

    removed = queue.remove("a")

    assert removed.id == "a"
    assert queue.waiting() == []


def test_depth_reflects_unclaimed_requests() -> None:
    queue = RequestQueue()
    queue.push(PendingRequest(id="a", prompt_tokens=[1], max_tokens=8))
    queue.push(PendingRequest(id="b", prompt_tokens=[2], max_tokens=8))

    before = queue.depth()
    queue.remove("a")
    after = queue.depth()

    assert before == 2
    assert after == 1


def test_wait_until_nonempty_blocks_until_a_request_arrives() -> None:
    async def run() -> str:
        queue = RequestQueue()

        async def push_soon() -> None:
            await asyncio.sleep(0.01)
            queue.push(PendingRequest(id="a", prompt_tokens=[1], max_tokens=8))

        asyncio.ensure_future(push_soon())
        await queue.wait_until_nonempty()
        return queue.waiting()[0].id

    assert asyncio.run(run()) == "a"


def test_wait_until_nonempty_returns_immediately_when_already_populated() -> None:
    async def run() -> bool:
        queue = RequestQueue()
        queue.push(PendingRequest(id="a", prompt_tokens=[1], max_tokens=8))

        await asyncio.wait_for(queue.wait_until_nonempty(), timeout=0.1)
        return True

    assert asyncio.run(run()) is True


def test_default_priority_is_one() -> None:
    request = PendingRequest(id="a", prompt_tokens=[1], max_tokens=8)

    assert request.priority == 1
