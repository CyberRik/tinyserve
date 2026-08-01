from tinyserve.batch.builder import ActiveSequence, build_batch


def test_single_decode_step_produces_one_row_flagged_for_logits() -> None:
    active = [ActiveSequence(seq_id=1, pending_tokens=[99], n_past=5)]

    batch = build_batch(active, capacity=8)

    assert len(batch) == 1
    row = batch.rows[0]
    assert row.seq_id == 1
    assert row.token == 99
    assert row.pos == 5
    assert row.needs_logits is True


def test_prefill_only_flags_the_last_row_for_logits() -> None:
    active = [ActiveSequence(seq_id=1, pending_tokens=[1, 2, 3], n_past=0)]

    batch = build_batch(active, capacity=8)

    assert [row.needs_logits for row in batch.rows] == [False, False, True]
    assert [row.pos for row in batch.rows] == [0, 1, 2]


def test_multiple_sequences_are_merged_into_one_batch() -> None:
    active = [
        ActiveSequence(seq_id=1, pending_tokens=[10], n_past=3),
        ActiveSequence(seq_id=2, pending_tokens=[20], n_past=7),
    ]

    batch = build_batch(active, capacity=8)

    seq_ids = {row.seq_id for row in batch.rows}
    assert seq_ids == {1, 2}
    assert len(batch) == 2


def test_decode_steps_are_prioritized_over_a_large_prefill_under_capacity() -> None:
    active = [
        ActiveSequence(seq_id=1, pending_tokens=list(range(10)), n_past=0),  # big prefill
        ActiveSequence(seq_id=2, pending_tokens=[42], n_past=4),  # single decode step
    ]

    batch = build_batch(active, capacity=5)

    # Only the small decode step fits given a lower capacity than the prefill needs.
    assert len(batch) == 1
    assert batch.rows[0].seq_id == 2


def test_sequence_excluded_entirely_when_it_does_not_fit_this_tick() -> None:
    active = [ActiveSequence(seq_id=1, pending_tokens=list(range(10)), n_past=0)]

    batch = build_batch(active, capacity=5)

    assert len(batch) == 0


def test_empty_pending_tokens_are_skipped() -> None:
    active = [ActiveSequence(seq_id=1, pending_tokens=[], n_past=10)]

    batch = build_batch(active, capacity=8)

    assert len(batch) == 0
