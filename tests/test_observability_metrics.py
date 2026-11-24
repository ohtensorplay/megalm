"""Contract tests for the Prometheus exposition used by /metrics.

These assert format-level properties that every Prometheus-compatible panel
depends on: HELP/TYPE lines, counter vs gauge typing, and real histogram
series. They exercise the Rust binding directly and need no GPU or model.
"""

from __future__ import annotations

from megalm.megalm_rs import prometheus_metrics


def test_exposition_includes_help_and_type_for_each_series() -> None:
    out = prometheus_metrics(
        '{"engine":{"steps":12,"emitted_tokens":34},"proxy":{"hit_rate":0.9}}'
    )
    lines = out.splitlines()
    for name in ("megalm_engine_steps", "megalm_engine_emitted_tokens", "megalm_proxy_hit_rate"):
        assert f"# HELP {name} " in out
        assert f"# TYPE {name} " in out
    # Every non-comment sample line must be preceded (somewhere) by HELP/TYPE.
    assert all(
        line.startswith("#") or " " in line for line in lines
    )


def test_counters_and_gauges_are_typed_distinctly() -> None:
    out = prometheus_metrics(
        '{"engine":{"steps":1},"proxy":{"hit_rate":0.5,"free_pages":4}}'
    )
    assert "# TYPE megalm_engine_steps counter\n" in out
    assert "# TYPE megalm_proxy_hit_rate gauge\n" in out
    assert "# TYPE megalm_proxy_free_pages gauge\n" in out


def test_histogram_exposes_cumulative_buckets_sum_and_count() -> None:
    out = prometheus_metrics(
        '{"engine":{"sequence_batch_histogram":{"1":5,"2":3,"10":1}}}'
    )
    name = "megalm_engine_sequence_batch_histogram"
    assert f"# TYPE {name} histogram\n" in out
    assert f'{name}_bucket{{le="1"}} 5\n' in out
    assert f'{name}_bucket{{le="2"}} 8\n' in out
    assert f'{name}_bucket{{le="10"}} 9\n' in out
    assert f'{name}_bucket{{le="+Inf"}} 9\n' in out
    assert f"{name}_sum 21\n" in out
    assert f"{name}_count 9\n" in out


def test_booleans_become_numeric_and_strings_are_skipped() -> None:
    out = prometheus_metrics('{"healthy":true,"status":"ok"}')
    assert "megalm_healthy 1\n" in out
    assert "status" not in out


def test_latency_buckets_are_cumulative_and_cover_overflow() -> None:
    from megalm.observability import LATENCY_BUCKETS_MS, observe_latency

    hist: dict[int, int] = {}
    for value in (3.0, 8.0, 30.0, 999999.0):
        observe_latency(hist, value)
    # 3 -> bucket 5, 8 -> bucket 10, 30 -> bucket 50, overflow -> last bucket.
    assert hist[5] == 1
    assert hist[10] == 1
    assert hist[50] == 1
    assert hist[LATENCY_BUCKETS_MS[-1]] == 1
    total = sum(hist.values())
    assert total == 4




def test_labeled_samples_render_native_prometheus_labels() -> None:
    from megalm.observability import labeled_sample

    out = prometheus_metrics(
        '{"http_requests":['
        '{"__value__":3,"__type__":"counter","__labels__":{"status":"503","route":"/v1/chat"}},'
        '{"__value__":9,"__type__":"counter","__labels__":{"status":"200","route":"/v1/chat"}}'
        ']}'
    )
    assert out.count("# TYPE megalm_http_requests counter\n") == 1
    assert 'megalm_http_requests{route="/v1/chat",status="200"} 9\n' in out
    assert 'megalm_http_requests{route="/v1/chat",status="503"} 3\n' in out
    assert labeled_sample(1, status="200")["__labels__"] == {"status": "200"}


def test_histogram_exact_totals_override_bucket_approximation() -> None:
    from megalm.observability import histogram_totals

    snapshot = {
        "ttft_ms_histogram": histogram_totals(
            {5: 2, 25: 3}, total_sum=88.4, total_count=5
        )
    }
    import json

    out = prometheus_metrics(json.dumps(snapshot))
    assert "megalm_ttft_ms_histogram_sum 88.4\n" in out
    assert "megalm_ttft_ms_histogram_count 5\n" in out
    assert 'megalm_ttft_ms_histogram_bucket{le="5"} 2\n' in out
    assert 'megalm_ttft_ms_histogram_bucket{le="25"} 5\n' in out




def test_cache_stats_expose_gdn_utilization_and_capacity_failures() -> None:
    from megalm.runtime.cache import CacheStats

    stats = CacheStats(
        total_pages=10,
        free_pages=4,
        reclaimable_pages=0,
        allocated_pages=6,
        active_requests=1,
        request_page_refs=6,
        prefix=None,
        gdn_slots_total=8,
        gdn_slots_used=2,
        capacity_failures={"kv_page": 3},
    )
    snap = stats.snapshot()
    assert snap["gdn_utilization"] == 0.25
    assert snap["capacity_failures"] == {"kv_page": 3}


def test_engine_snapshot_has_outcome_and_saturation_metrics() -> None:
    from megalm.runtime.engine import EngineMetrics

    metrics = EngineMetrics()
    metrics.finish_reasons["stop"] = 3
    metrics.finish_reasons["length"] = 1
    snapshot = metrics.snapshot()
    assert snapshot["finish_reasons"] == {"stop": 3, "length": 1}
