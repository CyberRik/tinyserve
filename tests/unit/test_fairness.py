from tinyserve.queue.fairness import WeightedFairQueue


def test_weight_matches_priority_value() -> None:
    fairness = WeightedFairQueue()

    assert fairness.weight(3) == 3


def test_weight_floors_at_one_for_non_positive_priority() -> None:
    fairness = WeightedFairQueue()

    assert fairness.weight(0) == 1
    assert fairness.weight(-5) == 1


def test_unknown_class_with_no_backlogged_history_starts_at_zero() -> None:
    fairness = WeightedFairQueue()

    assert fairness.virtual_finish_time(1, backlogged_classes={1}) == 0.0


def test_charge_advances_virtual_finish_time_by_inverse_weight() -> None:
    fairness = WeightedFairQueue()

    fairness.charge(priority_class=2, backlogged_classes={2})

    assert fairness.virtual_finish_time(2, backlogged_classes={2}) == 0.5


def test_newcomer_class_adopts_minimum_of_backlogged_known_classes() -> None:
    fairness = WeightedFairQueue()
    fairness.charge(priority_class=1, backlogged_classes={1})  # class 1 -> vft 1.0
    fairness.charge(priority_class=1, backlogged_classes={1})  # class 1 -> vft 2.0

    # Class 2 is new; among backlogged classes {1, 2}, the known minimum is 2.0.
    vft = fairness.virtual_finish_time(2, backlogged_classes={1, 2})

    assert vft == 2.0


def test_repeated_charges_accumulate() -> None:
    fairness = WeightedFairQueue()

    for _ in range(4):
        fairness.charge(priority_class=1, backlogged_classes={1})

    assert fairness.virtual_finish_time(1, backlogged_classes={1}) == 4.0
