import json
import logging

import pytest

from megalm.contracts.plan import PlanContractError
from megalm.observability import JsonLogFormatter
from megalm.startup import StartupPhase, StartupTracker


def test_structured_log_formatter_preserves_event_fields() -> None:
    record = logging.LogRecord(
        "megalm.test",
        logging.INFO,
        __file__,
        1,
        "startup_phase",
        (),
        None,
    )
    record._megalm_fields = {  # type: ignore[attr-defined]
        "phase": "ready",
        "plan_digest": "a" * 64,
    }

    payload = json.loads(JsonLogFormatter().format(record))

    assert payload["event"] == "startup_phase"
    assert payload["phase"] == "ready"
    assert payload["plan_digest"] == "a" * 64
    assert payload["level"] == "info"


def test_startup_tracker_only_allows_the_cold_start_order() -> None:
    tracker = StartupTracker(model_path="model")
    tracker.advance(StartupPhase.DRAFT)
    tracker.advance(StartupPhase.LOAD_PACK)

    with pytest.raises(PlanContractError, match="invalid startup transition"):
        tracker.advance(StartupPhase.SEAL)


def test_startup_tracker_reports_phase_progress_and_timing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[dict[str, object]] = []

    def capture(
        _logger: logging.Logger,
        _level: int,
        event: str,
        **fields: object,
    ) -> None:
        events.append({"event": event, **fields})

    monkeypatch.setattr("megalm.startup.log_event", capture)

    tracker = StartupTracker(model_path="model")
    tracker.advance(StartupPhase.DRAFT, dtype="float16")

    assert events[0]["phase"] == "discover"
    assert events[0]["phase_index"] == 1
    assert events[0]["phase_total"] == 7
    assert events[1]["phase"] == "draft"
    assert events[1]["previous_phase"] == "discover"
    assert events[1]["dtype"] == "float16"
    assert float(events[1]["previous_phase_duration_ms"]) >= 0
    assert float(events[1]["elapsed_ms"]) >= 0
