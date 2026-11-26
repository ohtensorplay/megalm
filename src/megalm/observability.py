"""Compatibility shim for the shared ``tp_observability.jsonlog`` API.

megaLM's logging helpers now live in the TensorPlay observability package. The
public names and behaviour below are unchanged; only the implementation moved.
"""

from __future__ import annotations

import logging

from tp_observability.jsonlog import (
    JsonLogFormatter as _JsonLogFormatter,
    LATENCY_BUCKETS_MS,
    configure_logging as _configure_logging,
    histogram_totals,
    labeled_sample,
    log_event,
    observe_latency,
)

_LEGACY_FIELDS_KEY = "_megalm_fields"


class JsonLogFormatter(_JsonLogFormatter):
    """Shared JSON formatter that also honours legacy ``_megalm_fields``."""

    def format(self, record: logging.LogRecord) -> str:
        # Older callers attach extra fields under megaLM's original record key.
        # The shared formatter reads ``_tp_fields``; bridge the two so existing
        # records keep rendering their structured fields.
        if not hasattr(record, "_tp_fields"):
            fields = getattr(record, _LEGACY_FIELDS_KEY, None)
            if isinstance(fields, dict):
                record._tp_fields = fields  # type: ignore[attr-defined]
        return super().format(record)


def configure_logging(level: str = "INFO") -> None:
    """Configure the ``megalm`` logger with shared JSON log formatting."""

    _configure_logging(level, service="megalm")


__all__ = [
    "JsonLogFormatter",
    "configure_logging",
    "log_event",
    "LATENCY_BUCKETS_MS",
    "observe_latency",
    "labeled_sample",
    "histogram_totals",
]
