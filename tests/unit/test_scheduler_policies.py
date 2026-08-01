import pytest

from tinyserve.queue.request_queue import PendingRequest
from tinyserve.scheduler.base import create_policy
from tinyserve.scheduler.fifo import FIFOPolicy
from tinyserve.scheduler.priority import PriorityPolicy
from tinyserve.scheduler.wfq import WFQPolicy


def _request(id: str, priority: int = 1, arrival_ts: float = 0.0) -> PendingRequest:
    return PendingRequest(
        id=id, prompt_tokens=[1], max_tokens=8, priority=priority, arrival_ts=arrival_ts
    )


class TestCreatePolicy:
    def test_creates_each_known_policy_by_name(self) -> None:
        assert isinstance(create_policy("fifo"), FIFOPolicy)
        assert isinstance(create_policy("priority"), PriorityPolicy)
        assert isinstance(create_policy("wfq"), WFQPolicy)

    def test_unknown_policy_name_raises(self) -> None:
        with pytest.raises(ValueError, match="unknown scheduling policy"):
            create_policy("not-a-real-policy")


class TestFIFOPolicy:
    def test_selects_in_arrival_order(self) -> None:
        waiting = [
            _request("c", arrival_ts=3),
            _request("a", arrival_ts=1),
            _request("b", arrival_ts=2),
        ]

        selected = FIFOPolicy().select(waiting, capacity=2)

        assert [r.id for r in selected] == ["a", "b"]

    def test_capacity_caps_selection(self) -> None:
        waiting = [_request("a", arrival_ts=1), _request("b", arrival_ts=2)]

        selected = FIFOPolicy().select(waiting, capacity=1)

        assert len(selected) == 1


class TestPriorityPolicy:
    def test_higher_priority_wins_over_earlier_arrival(self) -> None:
        waiting = [
            _request("low", priority=1, arrival_ts=0),
            _request("high", priority=5, arrival_ts=10),
        ]

        selected = PriorityPolicy().select(waiting, capacity=1)

        assert selected[0].id == "high"

    def test_ties_within_a_priority_class_broken_by_arrival_order(self) -> None:
        waiting = [
            _request("second", priority=1, arrival_ts=2),
            _request("first", priority=1, arrival_ts=1),
        ]

        selected = PriorityPolicy().select(waiting, capacity=1)

        assert selected[0].id == "first"

    def test_sustained_high_priority_load_starves_low_priority(self) -> None:
        # Documents the known tradeoff: as long as high-priority requests
        # keep arriving, a low-priority request never wins a slot.
        low = _request("low", priority=1, arrival_ts=0)
        policy = PriorityPolicy()

        for tick in range(50):
            high = _request(f"high-{tick}", priority=5, arrival_ts=tick + 1)
            selected = policy.select([low, high], capacity=1)
            assert selected[0].id == high.id


class TestWFQPolicy:
    def test_ties_within_the_same_priority_class_broken_by_arrival_order(self) -> None:
        # priority is the fairness class key here, so "a" and "b" sharing
        # priority=1 means they're the same class — within a class, WFQ
        # falls back to arrival order (still no starvation, just no extra
        # fine-grained fairness beneath the class level; a real scope line,
        # not an oversight).
        policy = WFQPolicy()
        a = _request("a", priority=1, arrival_ts=1)
        b = _request("b", priority=1, arrival_ts=2)

        first = policy.select([a, b], capacity=1)[0]

        assert first.id == "a"

    def test_higher_priority_class_wins_proportionally_more_slots(self) -> None:
        policy = WFQPolicy()
        low = _request("low", priority=1, arrival_ts=0)
        high = _request("high", priority=3, arrival_ts=0)

        picks = [policy.select([low, high], capacity=1)[0].id for _ in range(8)]

        # Weight 3 vs weight 1 -> roughly 3x as many slots for "high".
        assert picks.count("high") > picks.count("low")

    def test_low_priority_is_not_starved_unlike_strict_priority(self) -> None:
        # Same sustained-high-priority-arrivals scenario as the Priority
        # starvation test above, but WFQ must still eventually pick "low".
        policy = WFQPolicy()
        low = _request("low", priority=1, arrival_ts=0)

        picked_low_at_least_once = False
        for tick in range(50):
            high = _request(f"high-{tick}", priority=5, arrival_ts=tick + 1)
            selected = policy.select([low, high], capacity=1)
            if selected[0].id == "low":
                picked_low_at_least_once = True
                break

        assert picked_low_at_least_once

    def test_newcomer_class_adopts_current_minimum_not_zero(self) -> None:
        policy = WFQPolicy()
        a = _request("a", priority=1, arrival_ts=0)

        # Priority class 1 runs alone for a while, accruing virtual time.
        for _ in range(10):
            policy.select([a], capacity=1)

        # A brand-new priority class (2) arrives later. If it started at
        # virtual time 0 it would monopolize every tick just for being new;
        # if it had to start from class 1's current virtual time exactly
        # without adopting it as its *own* baseline, the bookkeeping would
        # be wrong in the other direction. Adopting the current minimum
        # means it's immediately competitive — neither starved nor favored.
        b = _request("b", priority=2, arrival_ts=100)
        picks = [policy.select([a, b], capacity=1)[0].id for _ in range(2)]

        assert "b" in picks
