from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from tinyserve.observability.tracing import RequestTracer


def _tracer_with_memory_exporter() -> tuple[RequestTracer, InMemorySpanExporter]:
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))

    tracer = RequestTracer(tracer=provider.get_tracer("test"))
    return tracer, exporter


def test_start_and_end_request_produces_one_span() -> None:
    tracer, exporter = _tracer_with_memory_exporter()

    tracer.start_request("req-1", prompt_tokens=5)
    tracer.end_request("req-1", completion_tokens=10)

    spans = exporter.get_finished_spans()
    assert len(spans) == 1
    assert spans[0].name == "request"
    assert spans[0].attributes["prompt_tokens"] == 5
    assert spans[0].attributes["completion_tokens"] == 10


def test_child_span_is_nested_under_the_request_root() -> None:
    tracer, exporter = _tracer_with_memory_exporter()

    tracer.start_request("req-1")
    tracer.start_span("req-1", "admission")
    tracer.end_span("req-1", "admission")
    tracer.end_request("req-1")

    spans = {span.name: span for span in exporter.get_finished_spans()}
    assert spans["admission"].parent.span_id == spans["request"].context.span_id


def test_span_context_manager_opens_and_closes() -> None:
    tracer, exporter = _tracer_with_memory_exporter()

    tracer.start_request("req-1")
    with tracer.span("req-1", "queue_wait"):
        pass
    tracer.end_request("req-1")

    names = {span.name for span in exporter.get_finished_spans()}
    assert "queue_wait" in names


def test_ending_an_unknown_span_is_a_noop() -> None:
    tracer, _ = _tracer_with_memory_exporter()

    tracer.end_span("never-started", "admission")  # must not raise


def test_ending_an_unknown_request_is_a_noop() -> None:
    tracer, _ = _tracer_with_memory_exporter()

    tracer.end_request("never-started")  # must not raise
