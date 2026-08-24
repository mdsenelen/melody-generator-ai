from __future__ import annotations

import asyncio
import sys
from pathlib import Path
from typing import Any

import pytest

sys.path.append(str(Path(__file__).resolve().parents[1]))

from app.request_limits import MaxBodySizeMiddleware, RequestBodyTooLarge  # noqa: E402

MAX_BYTES = 100


def _http_scope(headers: dict[str, str]) -> dict[str, Any]:
    return {
        "type": "http",
        "headers": [(k.lower().encode(), v.encode()) for k, v in headers.items()],
    }


class _RecordingApp:
    """Stand-in downstream ASGI app: reads the body to completion via
    whatever `receive` it's handed, and records whether it ran at all --
    lets tests assert the fast path never invokes it."""

    def __init__(self):
        self.called = False
        self.received_bodies: list[bytes] = []

    async def __call__(self, scope, receive, send):
        self.called = True
        while True:
            message = await receive()
            self.received_bodies.append(message.get("body", b""))
            if not message.get("more_body", False):
                break
        await send({"type": "http.response.start", "status": 200, "headers": []})
        await send({"type": "http.response.body", "body": b"ok"})


def _collect_send():
    sent = []

    async def send(message):
        sent.append(message)

    return sent, send


def test_content_length_over_cap_rejects_without_calling_inner_app():
    inner = _RecordingApp()
    middleware = MaxBodySizeMiddleware(inner, max_bytes=MAX_BYTES)
    scope = _http_scope({"content-length": str(MAX_BYTES + 1)})
    sent, send = _collect_send()

    async def receive():
        raise AssertionError("receive() must not be called once Content-Length fails the check")

    asyncio.run(middleware(scope, receive, send))

    assert inner.called is False
    start = next(m for m in sent if m["type"] == "http.response.start")
    assert start["status"] == 413


def test_content_length_under_cap_calls_inner_app_normally():
    inner = _RecordingApp()
    middleware = MaxBodySizeMiddleware(inner, max_bytes=MAX_BYTES)
    body = b"x" * 10
    scope = _http_scope({"content-length": str(len(body))})
    sent, send = _collect_send()

    async def receive():
        return {"type": "http.request", "body": body, "more_body": False}

    asyncio.run(middleware(scope, receive, send))

    assert inner.called is True
    assert inner.received_bodies == [body]
    start = next(m for m in sent if m["type"] == "http.response.start")
    assert start["status"] == 200


def test_streamed_body_without_content_length_raises_once_cap_crossed():
    inner = _RecordingApp()
    middleware = MaxBodySizeMiddleware(inner, max_bytes=MAX_BYTES)
    scope = _http_scope({})  # no content-length at all (e.g. chunked transfer)

    chunks = [b"x" * 60, b"y" * 60]  # cumulative 120 > MAX_BYTES (100)
    chunk_iter = iter(chunks)

    async def receive():
        chunk = next(chunk_iter, None)
        if chunk is None:
            return {"type": "http.request", "body": b"", "more_body": False}
        return {"type": "http.request", "body": chunk, "more_body": True}

    _, send = _collect_send()

    with pytest.raises(RequestBodyTooLarge) as exc_info:
        asyncio.run(middleware(scope, receive, send))

    assert exc_info.value.max_bytes == MAX_BYTES
    # Only the first (60-byte) chunk was ever handed to the inner app --
    # the cap trips on the second chunk, before the full 120-byte body is
    # ever assembled.
    assert inner.received_bodies == [chunks[0]]


def test_non_http_scope_passes_through_untouched():
    inner = _RecordingApp()
    middleware = MaxBodySizeMiddleware(inner, max_bytes=MAX_BYTES)
    scope = {"type": "lifespan"}

    async def receive():
        return {"type": "lifespan.startup"}

    sent, send = _collect_send()
    asyncio.run(middleware(scope, receive, send))

    assert inner.called is True
