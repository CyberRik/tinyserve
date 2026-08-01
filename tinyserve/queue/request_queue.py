"""FIFO Request Queue (PRD Section 6.2). Priority/WFQ sub-queues land in Phase 3."""

import asyncio
import time
from dataclasses import dataclass, field


@dataclass(frozen=True)
class PendingRequest:
    id: str
    prompt: str
    max_tokens: int
    arrival_ts: float = field(default_factory=time.monotonic)


class RequestQueue:
    """Holds accepted requests in arrival order until the batch loop pulls them."""

    def __init__(self) -> None:
        self._queue: asyncio.Queue[PendingRequest] = asyncio.Queue()

    def push(self, request: PendingRequest) -> None:
        self._queue.put_nowait(request)

    async def pop(self) -> PendingRequest:
        return await self._queue.get()

    def depth(self) -> int:
        return self._queue.qsize()
