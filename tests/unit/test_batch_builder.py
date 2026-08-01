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


def test_prefill_that_fits_whole_flags_only_the_last_row_for_logits() -> None:
    active = [ActiveSequence(seq_id=1, pending_tokens=[1, 2, 3], n_past=0)]

    batch = build_batch(active, capacity=8, chunk_size=512)

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


def test_decode_steps_are_prioritized_over_prefill_when_capacity_is_tight() -> None:
    active = [
        ActiveSequence(seq_id=1, pending_tokens=list(range(10)), n_past=0),  # big prefill
        ActiveSequence(seq_id=2, pending_tokens=[42], n_past=4),  # single decode step
    ]

    batch = build_batch(active, capacity=5)

    # Decode step goes first and gets its row; the prefill gets whatever's left (4).
    assert batch.row_count(seq_id=2) == 1
    assert batch.row_count(seq_id=1) == 4
    assert len(batch) == 5


def test_long_prefill_is_chunked_across_ticks_instead_of_excluded() -> None:
    active = [ActiveSequence(seq_id=1, pending_tokens=list(range(20)), n_past=0)]

    batch = build_batch(active, capacity=100, chunk_size=8)

    assert len(batch) == 8  # capped by chunk_size, not by capacity
    assert [row.pos for row in batch.rows] == list(range(8))
    assert all(row.needs_logits is False for row in batch.rows)  # not the final chunk


def test_final_chunk_of_a_prefill_flags_last_row_for_logits() -> None:
    # Simulates the last remaining slice of a prompt after earlier ticks
    # already consumed the first two 8-token chunks (n_past=16 reflects that).
    active = [ActiveSequence(seq_id=1, pending_tokens=list(range(4)), n_past=16)]

    batch = build_batch(active, capacity=100, chunk_size=8)

    assert len(batch) == 4
    assert [row.needs_logits for row in batch.rows] == [False, False, False, True]


def test_prefill_capped_by_remaining_capacity_not_just_chunk_size() -> None:
    active = [ActiveSequence(seq_id=1, pending_tokens=list(range(20)), n_past=0)]

    batch = build_batch(active, capacity=3, chunk_size=512)

    assert len(batch) == 3
    assert batch.rows[-1].needs_logits is False  # only a partial slice, not the final one


def test_empty_pending_tokens_are_skipped() -> None:
    active = [ActiveSequence(seq_id=1, pending_tokens=[], n_past=10)]

    batch = build_batch(active, capacity=8)

    assert len(batch) == 0


def test_row_count_reports_rows_consumed_per_sequence() -> None:
    active = [
        ActiveSequence(seq_id=1, pending_tokens=[1, 2, 3], n_past=0),
        ActiveSequence(seq_id=2, pending_tokens=[9], n_past=0),
    ]

    batch = build_batch(active, capacity=8)

    assert batch.row_count(seq_id=1) == 3
    assert batch.row_count(seq_id=2) == 1
    assert batch.row_count(seq_id=999) == 0
