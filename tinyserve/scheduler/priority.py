"""Strict-priority scheduling policy (PRD Section 7).

Higher priority always wins a free slot first. Documented starvation risk:
under sustained high-priority load, a low-priority request can wait
forever — this is the failure mode the WFQ policy exists to bound.
"""

from tinyserve.queue.request_queue import PendingRequest


class PriorityPolicy:
    def select(self, waiting: list[PendingRequest], capacity: int) -> list[PendingRequest]:
        ordered = sorted(waiting, key=lambda request: (-request.priority, request.arrival_ts))
        return ordered[:capacity]
