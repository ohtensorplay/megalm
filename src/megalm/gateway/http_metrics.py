"""Central HTTP RED metrics for the Gateway.

A thin ASGI middleware observes every response status (including errors
returned directly by middleware, which never raise ``HTTPException``) and
records it with route and status-class labels. Route templates rather than raw
paths keep label cardinality bounded.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Any

from megalm.observability import labeled_sample, observe_latency
from megalm.observability_otel import (
    end_server_span,
    record_http_response,
    start_server_span,
)


@dataclass(slots=True)
class GatewayMetrics:
    responses: Counter[tuple[str, str]] = field(default_factory=Counter)
    duration_ms_histogram: dict[int, int] = field(default_factory=dict)
    duration_total_ms: float = 0.0
    duration_samples: int = 0

    def record(self, status: int, route: str, duration_ms: float) -> None:
        status_class = f"{status // 100}xx" if status else "unknown"
        self.responses[(str(status), status_class, route)] += 1
        observe_latency(self.duration_ms_histogram, duration_ms)
        self.duration_total_ms += duration_ms
        self.duration_samples += 1

    def snapshot(self) -> dict[str, object]:
        samples = [
            labeled_sample(
                count,
                status=status,
                status_class=status_class,
                route=route,
            )
            for (status, status_class, route), count in sorted(
                self.responses.items()
            )
        ]
        return {
            "http_responses": samples,
            "http_request_duration_ms_histogram": {
                **self.duration_ms_histogram,
                "__sum__": self.duration_total_ms,
                "__count__": self.duration_samples,
            },
        }


class RequestMetricsMiddleware:
    """Pure-ASGI response observer; wraps the whole app including middleware."""

    def __init__(self, app: Any, metrics: GatewayMetrics) -> None:
        self.app = app
        self.metrics = metrics

    async def __call__(self, scope: dict, receive: Any, send: Any) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        import time

        started = time.perf_counter_ns()
        status = 0
        span, token = start_server_span(
            "http.request",
            scope.get("headers", ()),
            **{
                "http.request.method": scope.get("method", "UNKNOWN"),
                "http.route": scope.get("path", "unmatched"),
            },
        )

        async def send_wrapper(message: dict) -> None:
            nonlocal status
            if message["type"] == "http.response.start":
                status = int(message["status"])
            await send(message)

        error: BaseException | None = None
        try:
            await self.app(scope, receive, send_wrapper)
        except BaseException as exc:
            error = exc
            raise
        finally:
            # FastAPI resolves the matched route after routing; fall back to a
            # constant for unmatched requests to avoid path-based cardinality.
            route = "unmatched"
            matched = scope.get("route")
            if matched is not None and getattr(matched, "path", None):
                route = matched.path
            if span is not None:
                span.set_attribute("http.route", route)
            end_server_span(span, token, status=status, error=error)
            duration_ms = (time.perf_counter_ns() - started) / 1_000_000
            self.metrics.record(status, route, duration_ms)
            record_http_response(status, route, duration_ms)


__all__ = ["GatewayMetrics", "RequestMetricsMiddleware"]
