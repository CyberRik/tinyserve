"""Owns per-request output queues and incremental UTF-8 reassembly (PRD Section 6.7)."""

import asyncio
import codecs
from collections.abc import AsyncIterator
from dataclasses import dataclass, field


class _EndOfStream:
    __slots__ = ()


_END = _EndOfStream()


@dataclass
class _Stream:
    queue: "asyncio.Queue[str | _EndOfStream]" = field(default_factory=asyncio.Queue)
    decoder: codecs.IncrementalDecoder = field(
        default_factory=lambda: codecs.getincrementaldecoder("utf-8")("replace")
    )
    # Set from the producer thread and read from the SSE consumer task; a plain
    # bool is fine here since CPython's GIL makes single-attribute read/write atomic
    # and this flag only ever moves one way (False -> True).
    cancelled: bool = False


class StreamManager:
    """Bridges token bytes coming out of the Runtime to per-request SSE consumers."""

    def __init__(self) -> None:
        self._streams: dict[str, _Stream] = {}

    def create(self, request_id: str) -> None:
        self._streams[request_id] = _Stream()

    def push_token(self, request_id: str, token_bytes: bytes) -> None:
        # A disconnected client's stream can already be gone by the time the
        # producer checks is_cancelled() again (cancellation is checked once
        # per token, not instantly) — that's a normal race, not an error.
        stream = self._streams.get(request_id)
        if stream is None:
            return
        text = stream.decoder.decode(token_bytes)
        if text:
            stream.queue.put_nowait(text)

    def close(self, request_id: str) -> None:
        stream = self._streams.get(request_id)
        if stream is None:
            return
        tail = stream.decoder.decode(b"", final=True)
        if tail:
            stream.queue.put_nowait(tail)
        stream.queue.put_nowait(_END)

    def cancel(self, request_id: str) -> None:
        stream = self._streams.get(request_id)
        if stream is not None:
            stream.cancelled = True

    def is_cancelled(self, request_id: str) -> bool:
        stream = self._streams.get(request_id)
        return stream.cancelled if stream is not None else True

    async def subscribe(self, request_id: str) -> AsyncIterator[str]:
        stream = self._streams[request_id]
        try:
            while True:
                item = await stream.queue.get()
                if isinstance(item, _EndOfStream):
                    break
                yield item
        finally:
            self._streams.pop(request_id, None)
