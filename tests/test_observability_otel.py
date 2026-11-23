"""OpenTelemetry integration tests.

The suite constructs Engine/Gateway objects directly and never runs main(),
so telemetry must remain a strict no-op until explicitly configured. These
tests guard that contract plus trace-context helpers.
"""

from __future__ import annotations

import pytest

from megalm import observability_otel as otel


@pytest.fixture(autouse=True)
def _ensure_unconfigured():
    """Never let a test accidentally leave global OTel providers installed."""

    was = otel.configured()
    yield
    assert otel.configured() is was


def test_telemetry_is_noop_until_configured() -> None:
    assert otel.configured() is False
    # All of these must silently do nothing.
    otel.record_request_latency(ttft_ms=12.0, tpot_ms=3.0, e2e_ms=20.0)
    otel.record_http_response(200, "/v1/chat/completions", 25.0)
    otel.mirror_snapshot({"engine": {"steps": 5}})
    otel.shutdown()


def test_child_span_context_manager_is_safe_when_unconfigured() -> None:
    with otel.start_child_span("test.span", key="value") as span:
        assert span is None


def test_trace_fields_are_empty_without_an_active_span() -> None:
    assert otel.get_current_trace_fields() == {}


def test_extract_traceparent_reads_w3c_headers() -> None:
    traceparent = "00-0af7651916cd43dd8448eb211c80319c-b7ad6b7169203331-01"
    carrier = otel.extract_traceparent(
        [("traceparent", traceparent), ("content-type", "application/json")]
    )
    assert carrier.get("traceparent") == traceparent
    assert "content-type" not in carrier


def test_extract_traceparent_accepts_mapping_and_bytes() -> None:
    carrier = otel.extract_traceparent(
        {b"traceparent": b"00-0123456789abcdef0123456789abcdef-0123456789abcdef-01"}
    )
    assert carrier["traceparent"].startswith("00-")


def test_server_span_helpers_are_safe_when_unconfigured() -> None:
    span, token = otel.start_server_span("http.request", [])
    assert span is None and token is None
    otel.end_server_span(span, token, status=200)


def test_mirror_snapshot_observable_gauge_uses_observation_api(
    monkeypatch,
) -> None:
    """OTel 1.44 callbacks return Observations, not options.observe()."""

    otel_sdk = pytest.importorskip("opentelemetry.sdk.metrics")
    from opentelemetry.sdk.metrics.export import InMemoryMetricReader

    reader = InMemoryMetricReader()
    provider = otel_sdk.MeterProvider(metric_readers=[reader])
    meter = provider.get_meter("test")

    monkeypatch.setattr(otel, "_configured", True)
    monkeypatch.setattr(otel, "_meter", meter)
    monkeypatch.setattr(otel, "_instruments", {})
    monkeypatch.setattr(otel, "_snapshot_values", {})

    otel.mirror_snapshot({"engine": {"steps": 5}})
    # Force a collection; before the fix this raised AttributeError in the
    # callback because CallbackOptions has no `observe` method.
    reader.get_metrics_data()  # type: ignore[attr-defined]
    provider.shutdown()


def test_flatten_numbers_handles_descriptors_and_nesting() -> None:
    out: dict[str, float] = {}
    otel._flatten_numbers(
        {
            "engine": {"steps": 5, "flag": True},
            "rate": {
                "value": {
                    "__value__": 3,
                    "__labels__": {"status": "200"},
                }
            },
            "ignored": "text",
        },
        [],
        out,
    )
    assert out["engine.steps"] == 5.0
    assert out["engine.flag"] == 1.0
    assert out["rate.value"] == 3.0
    assert "ignored" not in out
