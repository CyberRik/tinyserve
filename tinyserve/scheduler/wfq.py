"""Weighted Fair Queuing scheduling policy (PRD Section 7) — the recommended default.

Each admission goes to whichever backlogged priority class currently has
the smallest virtual finish time (ties broken by arrival order within the
class), then charges that class's virtual time forward. Bounds unfairness
across priority classes under skewed load without the strict-priority
policy's starvation risk.
"""

from tinyserve.queue.fairness import WeightedFairQueue
from tinyserve.queue.request_queue import PendingRequest


class WFQPolicy:
    def __init__(self, fairness: WeightedFairQueue | None = None) -> None:
        self._fairness = fairness or WeightedFairQueue()

    def select(self, waiting: list[PendingRequest], capacity: int) -> list[PendingRequest]:
        pool = list(waiting)
        chosen: list[PendingRequest] = []

        for _ in range(min(capacity, len(pool))):
            backlogged = {request.priority for request in pool}
            best = min(
                pool,
                key=lambda request: (
                    self._fairness.virtual_finish_time(request.priority, backlogged),
                    request.arrival_ts,
                ),
            )
            chosen.append(best)
            pool.remove(best)
            self._fairness.charge(best.priority, backlogged)

        return chosen
