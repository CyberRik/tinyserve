"""OpenTelemetry tracing (PRD Section 10): one trace per request, with child
spans matching the Section 5 lifecycle stages (admission, queue.wait,
generation).

One real simplification, stated rather than hidden: continuous batching
means a single llama_decode() call can advance many different requests'
sequences at once, so "decode.step" has no single natural parent trace —
that per-tick cost is reported as a metric (decode_step_duration_seconds)
instead of a span, rather than forcing a many-parents span shape OpenTelemetry
isn't built for.

Spans here are opened and closed from different call sites (the HTTP
handler and the batch loop both touch the same request's lifecycle), so
this wraps the raw API in a small per-request span registry instead of
relying on context-manager nesting, which needs one call site owning the
whole span lifetime.
"""

from collections.abc import Iterator
from contextlib import contextmanager

from opentelemetry import trace
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter, SpanExporter
from opentelemetry.trace import Span, Tracer

_TRACER_NAME = "tinyserve"


def configure_tracing(exporter: SpanExporter | None = None) -> TracerProvider:
    provider = TracerProvider(resource=Resource.create({"service.name": "tinyserve"}))
    provider.add_span_processor(BatchSpanProcessor(exporter or ConsoleSpanExporter()))
    trace.set_tracer_provider(provider)
    return provider


class RequestTracer:
    """One root span per request_id, with child spans opened/closed as the
    request moves through admission, queueing, and generation."""

    def __init__(self, tracer: Tracer | None = None) -> None:
        self._tracer = tracer or trace.get_tracer(_TRACER_NAME)
        self._roots: dict[str, Span] = {}
        self._open_children: dict[tuple[str, str], Span] = {}

    def start_request(self, request_id: str, **attributes: str | int) -> None:
        span = self._tracer.start_span("request", attributes=attributes)
        self._roots[request_id] = span

    def start_span(self, request_id: str, name: str) -> None:
        root = self._roots.get(request_id)
        context = trace.set_span_in_context(root) if root is not None else None
        self._open_children[request_id, name] = self._tracer.start_span(name, context=context)

    def end_span(self, request_id: str, name: str) -> None:
        span = self._open_children.pop((request_id, name), None)
        if span is not None:
            span.end()

    @contextmanager
    def span(self, request_id: str, name: str) -> Iterator[None]:
        self.start_span(request_id, name)
        try:
            yield
        finally:
            self.end_span(request_id, name)

    def end_request(self, request_id: str, **attributes: str | int) -> None:
        span = self._roots.pop(request_id, None)
        if span is None:
            return
        for key, value in attributes.items():
            span.set_attribute(key, value)
        span.end()
