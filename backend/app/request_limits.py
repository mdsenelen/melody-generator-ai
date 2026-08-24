from __future__ import annotations

from starlette.datastructures import Headers
from starlette.responses import JSONResponse
from starlette.types import ASGIApp, Message, Receive, Scope, Send


class RequestBodyTooLarge(Exception):
    """Raised mid-stream when a request body exceeds the cap but arrived
    without a trustworthy Content-Length (chunked encoding, a stripped
    header, or a lying client). Caught by the exception handler registered
    in main.py -- see MaxBodySizeMiddleware below for why this has to be an
    exception rather than a response sent directly."""

    def __init__(self, max_bytes: int):
        self.max_bytes = max_bytes


def _too_large_response(max_bytes: int) -> JSONResponse:
    return JSONResponse(
        status_code=413,
        content={"detail": f"Request body exceeds the {max_bytes // (1024 * 1024)}MB upload limit"},
    )


class MaxBodySizeMiddleware:
    """Rejects oversized request bodies before they're read into memory.

    Content-Length, when present, lets us reject with zero bytes read --
    the fast path FastAPI's `await file.read()` skips entirely today. A
    missing/understated Content-Length (chunked transfer, a stripped
    header, or a client that just lies) can't be caught that way, so the
    streaming fallback counts bytes as they arrive and raises once the
    running total crosses the cap -- still well short of the full body
    being materialized. The raise (rather than sending a response
    directly) is necessary there: by the time we're mid-stream, the
    downstream app already owns `send` and is mid-flight parsing the body,
    so this middleware can't hijack the response itself. Because this
    middleware is added outermost (app.add_middleware inserts at position
    0, ahead of CORSMiddleware/ExceptionMiddleware), the exception
    propagates up through routing/dependency-resolution -- which is what's
    actually calling `receive()` while parsing multipart data -- into
    ExceptionMiddleware, where the handler registered in main.py turns it
    into a clean response.
    """

    def __init__(self, app: ASGIApp, max_bytes: int):
        self.app = app
        self.max_bytes = max_bytes

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        headers = Headers(scope=scope)
        content_length = headers.get("content-length")
        if content_length is not None:
            try:
                declared_size = int(content_length)
            except ValueError:
                declared_size = None
            if declared_size is not None and declared_size > self.max_bytes:
                await _too_large_response(self.max_bytes)(scope, receive, send)
                return

        total = 0

        async def limited_receive() -> Message:
            nonlocal total
            message = await receive()
            if message["type"] == "http.request":
                total += len(message.get("body", b""))
                if total > self.max_bytes:
                    raise RequestBodyTooLarge(self.max_bytes)
            return message

        await self.app(scope, limited_receive, send)
