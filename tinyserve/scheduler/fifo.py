"""FIFO scheduling policy (PRD Section 7): baseline, trivially fair in arrival order.

Its documented weakness only shows up under sustained priority imbalance —
a low-priority request waits exactly as long as everyone ahead of it,
regardless of how much higher-priority load keeps arriving after it.
"""

from tinyserve.queue.request_queue import PendingRequest


class FIFOPolicy:
    def select(self, waiting: list[PendingRequest], capacity: int) -> list[PendingRequest]:
        ordered = sorted(waiting, key=lambda request: request.arrival_ts)
        return ordered[:capacity]
