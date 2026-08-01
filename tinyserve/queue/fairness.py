"""WFQ virtual-time bookkeeping (PRD Section 6.2/7), adapted from Ancora's
fairness.py: there the resource divided fairly is worker-seconds across
workflow classes; here it's concurrency-slot admissions across priority
classes, weighted by `priority`.

One virtual finish time is tracked per priority class. Granting a class a
slot charges 1/weight to its virtual finish time — a higher-weight class
accrues virtual time more slowly, so it keeps winning selection more often.
A class with no prior history adopts the current system virtual time (the
minimum finish time among classes that currently have backlogged work)
instead of starting at zero, so an idle class doesn't get an unfair head
start, and a class returning after inactivity doesn't have to catch up to
one that's been running the whole time. Same newcomer-adopts-minimum rule
Ancora uses to keep bounded unfairness under skewed load.
"""


class WeightedFairQueue:
    def __init__(self) -> None:
        self._virtual_finish: dict[int, float] = {}

    def weight(self, priority: int) -> float:
        return max(priority, 1)

    def virtual_finish_time(self, priority_class: int, backlogged_classes: set[int]) -> float:
        if priority_class not in self._virtual_finish:
            known = [
                self._virtual_finish[c] for c in backlogged_classes if c in self._virtual_finish
            ]
            self._virtual_finish[priority_class] = min(known) if known else 0.0
        return self._virtual_finish[priority_class]

    def charge(self, priority_class: int, backlogged_classes: set[int]) -> None:
        current = self.virtual_finish_time(priority_class, backlogged_classes)
        self._virtual_finish[priority_class] = current + 1.0 / self.weight(priority_class)
