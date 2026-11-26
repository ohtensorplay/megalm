"""Compatibility shim for the shared ``tp_observability.otel`` API.

The OpenTelemetry implementation lives in the TensorPlay observability package.
This module re-exports it with megaLM-specific defaults (prefix ``megalm``,
service name ``megalm``, version ``0.2.0``) so existing call sites keep emitting
identical metric names and resource attributes.
"""

from __future__ import annotations

import sys
from types import ModuleType
from typing import Any

from tp_observability import otel as _shared

DEFAULT_PREFIX = "megalm"

# Direct re-exports: these have no service-specific defaults.
available = _shared.available
configured = _shared.configured
shutdown = _shared.shutdown
get_current_trace_fields = _shared.get_current_trace_fields
extract_traceparent = _shared.extract_traceparent
record_request_latency = _shared.record_request_latency
record_http_response = _shared.record_http_response
start_server_span = _shared.start_server_span
end_server_span = _shared.end_server_span
start_child_span = _shared.start_child_span


def configure_telemetry(
    *,
    service_name: str = "megalm",
    service_version: str = "0.2.0",
    replica_mode: str = "embedded",
    process_kind: str = "gateway",
    prefix: str = DEFAULT_PREFIX,
    metrics_endpoint: str | None = None,
    traces_endpoint: str | None = None,
) -> bool:
    """Install OTLP providers with megaLM's default names and prefix."""

    return _shared.configure_telemetry(
        service_name=service_name,
        service_version=service_version,
        replica_mode=replica_mode,
        process_kind=process_kind,
        prefix=prefix,
        metrics_endpoint=metrics_endpoint,
        traces_endpoint=traces_endpoint,
    )


def get_tracer(name: str = DEFAULT_PREFIX) -> Any:
    """Return a tracer named after the megaLM service."""

    return _shared.get_tracer(name)


def mirror_snapshot(snapshot: dict[str, Any], *, prefix: str = DEFAULT_PREFIX) -> None:
    """Mirror a Prometheus snapshot into ``megalm.*`` observable gauges."""

    _shared.mirror_snapshot(snapshot, prefix=prefix)


# Preserve the original module's private test hooks: patching state such as
# ``_configured`` or ``_meter`` on this module must reach the shared
# implementation, otherwise the patch would silently do nothing.
_STATE_NAMES = frozenset(
    {
        "_LOGGER",
        "_OTEL_AVAILABLE",
        "_configured",
        "_prefix",
        "_lock",
        "_meter",
        "_tracer",
        "_instruments",
        "_snapshot_values",
        "_export_lock",
        "_providers",
        "_SpanAttributesContext",
        "_register_sli_instruments",
        "_flatten_numbers",
    }
)


def __getattr__(name: str) -> Any:
    if name in _STATE_NAMES:
        return getattr(_shared, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


class _ShimModule(ModuleType):
    def __setattr__(self, name: str, value: Any) -> None:
        if name in _STATE_NAMES:
            setattr(_shared, name, value)
        else:
            super().__setattr__(name, value)


sys.modules[__name__].__class__ = _ShimModule


__all__ = [
    "available",
    "configured",
    "configure_telemetry",
    "shutdown",
    "get_tracer",
    "start_server_span",
    "end_server_span",
    "start_child_span",
    "get_current_trace_fields",
    "extract_traceparent",
    "record_request_latency",
    "record_http_response",
    "mirror_snapshot",
]
