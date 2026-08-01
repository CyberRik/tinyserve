"""Pluggable scheduling policy (PRD Section 6.3/7).

Continuous batching itself isn't a policy choice — every active sequence
advances every tick regardless of policy (PRD Section 7). The one place a
policy actually has leverage in this design is *admission*: which waiting
requests claim a free concurrency slot next, when there are more requests
than slots. FIFO/Priority/WFQ below are all, concretely, different answers
to that one question.
"""

from typing import Protocol

from tinyserve.queue.request_queue import PendingRequest
from tinyserve.scheduler.fifo import FIFOPolicy
from tinyserve.scheduler.priority import PriorityPolicy
from tinyserve.scheduler.wfq import WFQPolicy


class SchedulingPolicy(Protocol):
    def select(self, waiting: list[PendingRequest], capacity: int) -> list[PendingRequest]:
        """Return, in the order they should be admitted, up to `capacity`
        requests to pull from `waiting` this tick."""
        ...


def create_policy(name: str) -> SchedulingPolicy:
    policies: dict[str, type[SchedulingPolicy]] = {
        "fifo": FIFOPolicy,
        "priority": PriorityPolicy,
        "wfq": WFQPolicy,
    }
    try:
        return policies[name]()
    except KeyError:
        raise ValueError(
            f"unknown scheduling policy {name!r}; choose from {sorted(policies)}"
        ) from None
