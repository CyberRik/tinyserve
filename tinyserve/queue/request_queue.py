"""FIFO Request Queue (PRD Section 6.2). Priority/WFQ sub-queues land in Phase 3."""

import asyncio
import time
from dataclasses import dataclass, field


@dataclass(frozen=True)
class PendingRequest:
    id: str
    prompt_tokens: list[int]
    max_tokens: int
    arrival_ts: float = field(default_factory=time.monotonic)


class RequestQueue:
    """Holds admitted requests in arrival order until a batch-loop tick claims them."""

    def __init__(self) -> None:
        self._queue: asyncio.Queue[PendingRequest] = asyncio.Queue()

    def push(self, request: PendingRequest) -> None:
        self._queue.put_nowait(request)

    def pop_batch(self, max_count: int) -> list[PendingRequest]:
        """Claim up to max_count requests without blocking.

        Used every tick to fill free concurrency slots; returns fewer (or
        none) if the queue doesn't have that many waiting.
        """
        requests: list[PendingRequest] = []
        for _ in range(max_count):
            try:
                requests.append(self._queue.get_nowait())
            except asyncio.QueueEmpty:
                break
        return requests

    async def wait_for_next(self) -> PendingRequest:
        """Block until at least one request arrives — used only when idle,
        so the tick loop doesn't busy-spin with no active sequences."""
        return await self._queue.get()

    def depth(self) -> int:
        return self._queue.qsize()
