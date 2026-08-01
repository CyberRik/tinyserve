from tinyserve.observability.metrics import Metrics


def test_each_instance_gets_an_isolated_registry() -> None:
    # Two Metrics() instances must not collide on Prometheus's global
    # default registry — this is what makes the module safe to instantiate
    # once per test (and once per app instance) without name clashes.
    first = Metrics()
    second = Metrics()

    first.admission_accepted_total.inc()

    assert first.registry.get_sample_value("admission_accepted_total") == 1.0
    assert second.registry.get_sample_value("admission_accepted_total") == 0.0


def test_admission_counters_track_accept_and_reject_separately() -> None:
    metrics = Metrics()

    metrics.admission_accepted_total.inc()
    metrics.admission_accepted_total.inc()
    metrics.admission_rejected_total.labels(reason="kv_cache_full").inc()

    assert metrics.registry.get_sample_value("admission_accepted_total") == 2.0
    assert (
        metrics.registry.get_sample_value("admission_rejected_total", {"reason": "kv_cache_full"})
        == 1.0
    )


def test_queue_depth_gauge_reflects_last_set_value() -> None:
    metrics = Metrics()

    metrics.queue_depth.set(3)
    metrics.queue_depth.set(1)

    assert metrics.registry.get_sample_value("queue_depth") == 1.0


def test_histogram_observations_are_counted() -> None:
    metrics = Metrics()

    metrics.decode_step_duration_seconds.observe(0.01)
    metrics.decode_step_duration_seconds.observe(0.02)

    assert metrics.registry.get_sample_value("decode_step_duration_seconds_count") == 2.0


def test_cancellations_total_tracks_reason_label() -> None:
    metrics = Metrics()

    metrics.cancellations_total.labels(reason="queue_timeout").inc()
    metrics.cancellations_total.labels(reason="client_disconnect").inc(2)

    assert (
        metrics.registry.get_sample_value("cancellations_total", {"reason": "queue_timeout"}) == 1.0
    )
    assert (
        metrics.registry.get_sample_value("cancellations_total", {"reason": "client_disconnect"})
        == 2.0
    )
