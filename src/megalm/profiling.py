"""Opt-in request-level profiling with stage/hop breakdown views.

This module is the Python counterpart of the Rust ``Profiler`` in
``rust/crates/server/src/profiling.rs``. Both write the same JSONL schema so a
single directory (or a glob over per-process files) can be merged by request::

    {"ts_ns": ..., "pid": ..., "run_id": ..., "request_id": ...,
     "event": ..., "stage": ..., "extra": {...}}

When ``MEGALM_PROFILE_FILE`` is unset the profiler is a no-op: :meth:`record`
is a single ``None`` check with no allocation, file I/O, or task overhead, so
enabling the plugin on the serving path costs nothing when it is disabled.

Active stage binding uses both a :class:`contextvars.ContextVar` (asyncio
requests) and :class:`threading.ThreadLocal` (replica worker / reader threads),
because a ContextVar set in an HTTP task is not visible in a separate OS thread.
"""

from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from glob import glob
import json
import logging
import os
from pathlib import Path
import threading
import time
from typing import Any, Iterator, Sequence
from uuid import uuid4

_LOGGER = logging.getLogger("megalm.profiling")

_ENV_FILE = "MEGALM_PROFILE_FILE"
_ENV_RUN_ID = "MEGALM_PROFILE_RUN_ID"

# Milestones from GAP-PLAN.md §11. New event names are allowed (record() does
# not reject them) so a caller can add its own without touching this module.
ADMISSION = "admission"
PREFILL_START = "prefill_start"
FIRST_TOKEN = "first_token"
HOP_SENT = "hop_sent"
HOP_RECEIVED = "hop_received"
TERMINAL_RESPONSE = "terminal_response"

MILESTONES: tuple[str, ...] = (
    ADMISSION,
    PREFILL_START,
    FIRST_TOKEN,
    HOP_SENT,
    HOP_RECEIVED,
    TERMINAL_RESPONSE,
)

_SCALARS = (str, int, float, bool)
_MAX_STRING_CHARS = 512

# The Rust frontend records ``admitted`` while the shared milestone name is
# ``admission``; normalize on load so both JSONL streams merge by request.
_EVENT_ALIASES = {"admitted": ADMISSION}


# ---------------------------------------------------------------------------
# Active-stage dual binding
# ---------------------------------------------------------------------------

_STAGE_CONTEXT: ContextVar[str] = ContextVar("megalm_profile_stage", default="")
_THREAD = threading.local()


def current_stage() -> str:
    """Return the active stage, preferring the thread binding over asyncio."""

    thread_stage = getattr(_THREAD, "stage", "")
    return thread_stage or _STAGE_CONTEXT.get()


@contextmanager
def active_stage(stage: str) -> Iterator[None]:
    """Bind ``stage`` for the duration of the block in both contexts."""

    token = _STAGE_CONTEXT.set(stage)
    previous_thread = getattr(_THREAD, "stage", "")
    _THREAD.stage = stage
    try:
        yield
    finally:
        _STAGE_CONTEXT.reset(token)
        _THREAD.stage = previous_thread


# ---------------------------------------------------------------------------
# Event value object
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class ProfileEvent:
    """One JSONL milestone line."""

    ts_ns: int
    pid: int
    run_id: str
    request_id: str
    event: str
    stage: str = ""
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "ts_ns": self.ts_ns,
            "pid": self.pid,
            "run_id": self.run_id,
            "request_id": self.request_id,
            "event": self.event,
            "stage": self.stage,
            "extra": self.extra,
        }

    def to_line(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, separators=(",", ":"))

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> ProfileEvent:
        event = str(payload.get("event", ""))
        return cls(
            ts_ns=int(payload["ts_ns"]),
            pid=int(payload["pid"]),
            run_id=str(payload.get("run_id", "")),
            request_id=str(payload.get("request_id", "")),
            event=_EVENT_ALIASES.get(event, event),
            stage=str(payload.get("stage", "")),
            extra=dict(payload.get("extra") or {}),
        )

    @classmethod
    def from_line(cls, line: str) -> ProfileEvent | None:
        try:
            payload = json.loads(line)
        except (TypeError, ValueError):
            return None
        if not isinstance(payload, dict) or "ts_ns" not in payload:
            return None
        try:
            return cls.from_dict(payload)
        except (KeyError, TypeError, ValueError):
            return None


def _clean_extra(values: dict[str, object]) -> tuple[dict[str, Any], int]:
    """Keep only small JSON scalars; drop tensors/numpy/containers.

    Returns the cleaned mapping and the number of dropped metadata fields.
    """

    clean: dict[str, Any] = {}
    dropped = 0
    for key, value in values.items():
        if value is None or isinstance(value, bool):
            clean[str(key)] = value
        elif isinstance(value, (int, float)):
            clean[str(key)] = value
        elif isinstance(value, str):
            clean[str(key)] = value[:_MAX_STRING_CHARS]
        else:
            # Tensors, numpy arrays, nested containers and other objects are
            # forbidden in profile metadata; skip them without touching them.
            dropped += 1
    return clean, dropped


# ---------------------------------------------------------------------------
# Recorder
# ---------------------------------------------------------------------------


class Profiler:
    """Append-only JSONL milestone recorder. No-op when no file is configured."""

    def __init__(
        self,
        path: str | os.PathLike[str] | None = None,
        *,
        run_id: str | None = None,
        stage: str = "",
    ) -> None:
        self._file: Any = None
        self._lock = threading.Lock()
        self._path = os.fspath(path) if path else ""
        self._stage = stage
        self._run_id = run_id or os.environ.get(_ENV_RUN_ID) or uuid4().hex
        self._warned = False
        self.events_recorded = 0
        self.dropped_writes = 0
        self.dropped_metadata = 0
        if self._path:
            try:
                Path(self._path).parent.mkdir(parents=True, exist_ok=True)
                self._file = open(self._path, "a", encoding="utf-8")
            except OSError as exc:  # pragma: no cover - filesystem dependent
                self._file = None
                self._warn("cannot open profile file", error=str(exc))

    @classmethod
    def from_env(cls, *, stage: str = "") -> Profiler:
        """Build a profiler from ``MEGALM_PROFILE_FILE`` / ``MEGALM_PROFILE_RUN_ID``."""

        return cls(os.environ.get(_ENV_FILE), stage=stage)

    @property
    def enabled(self) -> bool:
        return self._file is not None

    @property
    def run_id(self) -> str:
        return self._run_id

    @property
    def path(self) -> str:
        return self._path

    def _warn(self, message: str, **fields: object) -> None:
        if self._warned:
            return
        self._warned = True
        _LOGGER.warning(message, extra=fields)

    def record(
        self,
        event: str,
        request_id: str,
        *,
        stage: str | None = None,
        **extra: object,
    ) -> None:
        """Append one milestone. Failures are swallowed and counted."""

        if self._file is None:
            return
        clean, dropped = _clean_extra(extra)
        milestone = ProfileEvent(
            ts_ns=time.time_ns(),
            pid=os.getpid(),
            run_id=self._run_id,
            request_id=str(request_id),
            event=event,
            stage=stage if stage is not None else (current_stage() or self._stage),
            extra=clean,
        )
        try:
            line = milestone.to_line()
        except (TypeError, ValueError) as exc:  # pragma: no cover
            self.dropped_writes += 1
            self._warn("cannot serialize profile event", error=str(exc))
            return
        try:
            with self._lock:
                self._file.write(line + "\n")
                self._file.flush()
        except (OSError, ValueError) as exc:  # pragma: no cover
            self.dropped_writes += 1
            self._warn("cannot write profile event", error=str(exc))
            return
        self.events_recorded += 1
        self.dropped_metadata += dropped

    def snapshot(self) -> dict[str, Any]:
        """Cold-path metrics snapshot for /metrics."""

        return {
            "enabled": self.enabled,
            "events_recorded": self.events_recorded,
            "dropped_writes": self.dropped_writes,
            "dropped_metadata": self.dropped_metadata,
            "run_id": self._run_id,
        }

    def close(self) -> None:
        with self._lock:
            if self._file is not None:
                try:
                    self._file.close()
                except OSError:  # pragma: no cover
                    pass
                self._file = None

    def __enter__(self) -> Profiler:
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()


# ---------------------------------------------------------------------------
# Process-wide accessor (single owner: this module)
# ---------------------------------------------------------------------------

_PROFILER: Profiler | None = None
_DISABLED = Profiler()


def configure_profiler(profiler: Profiler | None) -> None:
    """Install the process profiler. ``None`` restores the no-op singleton."""

    global _PROFILER
    _PROFILER = profiler


def get_profiler() -> Profiler:
    """Return the configured profiler, or the shared no-op singleton."""

    return _PROFILER if _PROFILER is not None else _DISABLED


def record(event: str, request_id: str, *, stage: str | None = None, **extra: object) -> None:
    """Record a milestone through the process profiler (no-op by default)."""

    get_profiler().record(event, request_id, stage=stage, **extra)


# ---------------------------------------------------------------------------
# Views
# ---------------------------------------------------------------------------


def load_events(*paths: str | os.PathLike[str]) -> tuple[ProfileEvent, ...]:
    """Load and sort events from files, directories, or glob patterns."""

    resolved: list[str] = []
    for raw in paths:
        text = os.fspath(raw)
        if any(ch in text for ch in "*?["):
            resolved.extend(glob(text))
        elif os.path.isdir(text):
            resolved.extend(str(p) for p in Path(text).glob("*.jsonl"))
        else:
            resolved.append(text)
    events: list[ProfileEvent] = []
    seen: set[str] = set()
    for path in resolved:
        real = os.path.realpath(path)
        if real in seen or not os.path.isfile(real):
            continue
        seen.add(real)
        try:
            with open(real, encoding="utf-8") as handle:
                for line in handle:
                    event = ProfileEvent.from_line(line)
                    if event is not None:
                        events.append(event)
        except OSError:  # pragma: no cover - filesystem dependent
            continue
    events.sort(key=lambda event: (event.ts_ns, event.pid, event.event))
    return tuple(events)


def _request_events(
    events: Sequence[ProfileEvent], request_id: str
) -> tuple[ProfileEvent, ...]:
    return tuple(event for event in events if event.request_id == request_id)


def timeline(
    events: Sequence[ProfileEvent], request_id: str
) -> tuple[dict[str, Any], ...]:
    """Return one request's milestones with elapsed milliseconds from the first."""

    selected = _request_events(events, request_id)
    if not selected:
        return ()
    origin = selected[0].ts_ns
    return tuple(
        {
            "event": event.event,
            "stage": event.stage,
            "pid": event.pid,
            "run_id": event.run_id,
            "ts_ns": event.ts_ns,
            "elapsed_ms": (event.ts_ns - origin) / 1_000_000,
            "extra": event.extra,
        }
        for event in selected
    )


def stage_breakdown(
    events: Sequence[ProfileEvent], request_id: str
) -> dict[str, float]:
    """Milliseconds attributed to each stage, from one milestone to the next."""

    selected = _request_events(events, request_id)
    breakdown: dict[str, float] = {}
    for index, event in enumerate(selected[:-1]):
        duration = (selected[index + 1].ts_ns - event.ts_ns) / 1_000_000
        stage = event.stage or "unknown"
        breakdown[stage] = breakdown.get(stage, 0.0) + max(duration, 0.0)
    return breakdown


def hop_breakdown(events: Sequence[ProfileEvent]) -> tuple[dict[str, Any], ...]:
    """Pair ``hop_sent``/``hop_received`` by (request_id, src, dst, chunk_id)."""

    def hop_key(event: ProfileEvent) -> tuple[str, str, str, str] | None:
        if event.event not in (HOP_SENT, HOP_RECEIVED):
            return None
        extra = event.extra
        if not all(key in extra for key in ("src", "dst", "chunk_id")):
            return None
        return (
            event.request_id,
            str(extra["src"]),
            str(extra["dst"]),
            str(extra["chunk_id"]),
        )

    sent: dict[tuple[str, str, str, str], ProfileEvent] = {}
    received: dict[tuple[str, str, str, str], ProfileEvent] = {}
    for event in events:
        key = hop_key(event)
        if key is None:
            continue
        target = sent if event.event == HOP_SENT else received
        # A chunk may be retried; keep the earliest observation for each side.
        if key not in target or event.ts_ns < target[key].ts_ns:
            target[key] = event

    pairs: list[dict[str, Any]] = []
    for key, send_event in sorted(sent.items()):
        recv_event = received.get(key)
        pairs.append(
            {
                "request_id": key[0],
                "src": key[1],
                "dst": key[2],
                "chunk_id": key[3],
                "sent_ns": send_event.ts_ns,
                "received_ns": recv_event.ts_ns if recv_event else None,
                "latency_ms": (
                    (recv_event.ts_ns - send_event.ts_ns) / 1_000_000
                    if recv_event is not None
                    else None
                ),
                "matched": recv_event is not None,
            }
        )
    return tuple(pairs)


def view(
    *paths: str | os.PathLike[str], request_id: str | None = None
) -> dict[str, Any]:
    """Build a combined profiling view for all events (or one request)."""

    events = load_events(*paths)
    request_ids = (
        (request_id,) if request_id else tuple(sorted({e.request_id for e in events}))
    )
    return {
        "run_ids": sorted({e.run_id for e in events if e.run_id}),
        "timelines": {
            rid: timeline(events, rid) for rid in request_ids
        },
        "stage_breakdowns": {
            rid: stage_breakdown(events, rid) for rid in request_ids
        },
        "hops": hop_breakdown(events),
        "event_count": len(events),
    }


__all__ = [
    "ADMISSION",
    "FIRST_TOKEN",
    "HOP_RECEIVED",
    "HOP_SENT",
    "MILESTONES",
    "PREFILL_START",
    "TERMINAL_RESPONSE",
    "ProfileEvent",
    "Profiler",
    "active_stage",
    "current_stage",
    "hop_breakdown",
    "load_events",
    "stage_breakdown",
    "timeline",
    "view",
]
