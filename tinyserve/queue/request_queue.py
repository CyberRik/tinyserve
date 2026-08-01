"""Request Queue (PRD Section 6.2): holds admitted-but-not-yet-running requests.

A reorderable waiting pool rather than a blind FIFO — the active
SchedulingPolicy (tinyserve.scheduler) decides which subset of `waiting()`
claims a free concurrency slot next; this module has no opinion on order.
"""

import asyncio
import time
from dataclasses import dataclass, field


@dataclass(frozen=True)
class PendingRequest:
    id: str
    prompt_tokens: list[int]
    max_tokens: int
    priority: int = 1
    arrival_ts: float = field(default_factory=time.monotonic)


class RequestQueue:
    def __init__(self) -> None:
        self._waiting: dict[str, PendingRequest] = {}
        self._arrived = asyncio.Event()

    def push(self, request: PendingRequest) -> None:
        self._waiting[request.id] = request
        self._arrived.set()

    def waiting(self) -> list[PendingRequest]:
        """All currently-queued requests — a policy chooses among these."""
        return list(self._waiting.values())

    def remove(self, request_id: str) -> PendingRequest:
        return self._waiting.pop(request_id)

    async def wait_until_nonempty(self) -> None:
        """Block only while there's nothing queued, so the tick loop doesn't
        busy-spin when idle; returns as soon as anything arrives."""
        while not self._waiting:
            self._arrived.clear()
            await self._arrived.wait()

    def depth(self) -> int:
        return len(self._waiting)
