from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import Any

import orjson


class RequestBodyTooLarge(RuntimeError):
    pass


class RequestBodyLimitMiddleware:
    """Bound selected ASGI request bodies before Pydantic materializes JSON."""

    def __init__(
        self,
        app: Callable[..., Awaitable[None]],
        *,
        max_bytes: int,
        paths: tuple[str, ...],
    ) -> None:
        if max_bytes <= 0 or not paths:
            raise ValueError("request body limit configuration is invalid")
        self.app = app
        self.max_bytes = max_bytes
        self.paths = frozenset(paths)

    async def __call__(
        self,
        scope: dict[str, Any],
        receive: Callable[[], Awaitable[dict[str, Any]]],
        send: Callable[[dict[str, Any]], Awaitable[None]],
    ) -> None:
        if (
            scope.get("type") != "http"
            or scope.get("path") not in self.paths
        ):
            await self.app(scope, receive, send)
            return
        headers = dict(scope.get("headers", ()))
        raw_length = headers.get(b"content-length")
        if raw_length is not None:
            try:
                content_length = int(raw_length)
            except ValueError:
                await _send_error(send, 400, "invalid Content-Length")
                return
            if content_length < 0:
                await _send_error(send, 400, "invalid Content-Length")
                return
            if content_length > self.max_bytes:
                await _send_error(send, 413, "request body exceeds its byte limit")
                return

        received = 0
        response_started = False

        async def limited_receive() -> dict[str, Any]:
            nonlocal received
            message = await receive()
            if message.get("type") == "http.request":
                received += len(message.get("body", b""))
                if received > self.max_bytes:
                    raise RequestBodyTooLarge
            return message

        async def tracked_send(message: dict[str, Any]) -> None:
            nonlocal response_started
            if message.get("type") == "http.response.start":
                response_started = True
            await send(message)

        try:
            await self.app(scope, limited_receive, tracked_send)
        except RequestBodyTooLarge:
            if response_started:
                raise
            await _send_error(send, 413, "request body exceeds its byte limit")


async def _send_error(
    send: Callable[[dict[str, Any]], Awaitable[None]],
    status: int,
    detail: str,
) -> None:
    body = orjson.dumps({"detail": detail})
    await send(
        {
            "type": "http.response.start",
            "status": status,
            "headers": (
                (b"content-type", b"application/json"),
                (b"content-length", str(len(body)).encode()),
            ),
        }
    )
    await send({"type": "http.response.body", "body": body})


__all__ = ["RequestBodyLimitMiddleware", "RequestBodyTooLarge"]
