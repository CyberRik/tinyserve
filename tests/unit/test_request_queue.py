import asyncio

from tinyserve.queue.request_queue import PendingRequest, RequestQueue


def test_pop_batch_returns_requests_in_fifo_order() -> None:
    queue = RequestQueue()
    queue.push(PendingRequest(id="a", prompt_tokens=[1], max_tokens=8))
    queue.push(PendingRequest(id="b", prompt_tokens=[2], max_tokens=8))
    queue.push(PendingRequest(id="c", prompt_tokens=[3], max_tokens=8))

    requests = queue.pop_batch(max_count=3)

    assert [request.id for request in requests] == ["a", "b", "c"]


def test_pop_batch_returns_fewer_than_max_count_when_queue_is_short() -> None:
    queue = RequestQueue()
    queue.push(PendingRequest(id="a", prompt_tokens=[1], max_tokens=8))

    requests = queue.pop_batch(max_count=5)

    assert [request.id for request in requests] == ["a"]


def test_pop_batch_on_empty_queue_returns_empty_list() -> None:
    queue = RequestQueue()

    assert queue.pop_batch(max_count=3) == []


def test_depth_reflects_unclaimed_requests() -> None:
    queue = RequestQueue()
    queue.push(PendingRequest(id="a", prompt_tokens=[1], max_tokens=8))
    queue.push(PendingRequest(id="b", prompt_tokens=[2], max_tokens=8))

    before = queue.depth()
    queue.pop_batch(max_count=1)
    after = queue.depth()

    assert before == 2
    assert after == 1


def test_wait_for_next_blocks_until_a_request_arrives() -> None:
    async def run() -> str:
        queue = RequestQueue()

        async def push_soon() -> None:
            await asyncio.sleep(0.01)
            queue.push(PendingRequest(id="a", prompt_tokens=[1], max_tokens=8))

        asyncio.ensure_future(push_soon())
        request = await queue.wait_for_next()
        return request.id

    assert asyncio.run(run()) == "a"
