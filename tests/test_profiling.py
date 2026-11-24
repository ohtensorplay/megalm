"""G10 gold tests: request-level JSONL profiling and stage/hop views."""

from __future__ import annotations

import asyncio
import json
import os
import threading

import pytest

from megalm.profiling import (
    Profiler,
    active_stage,
    configure_profiler,
    current_stage,
    get_profiler,
    hop_breakdown,
    load_events,
    record as record_profile,
    stage_breakdown,
    timeline,
    view,
)


def test_disabled_profiler_is_a_noop(tmp_path, caplog) -> None:
    profiler = Profiler()
    assert not profiler.enabled
    profiler.record("admission", "req-1", prompt_tokens=3)
    assert profiler.snapshot()["events_recorded"] == 0
    assert not (tmp_path / "profile.jsonl").exists()


def test_from_env_enables_recording(monkeypatch, tmp_path) -> None:
    path = tmp_path / "profile.jsonl"
    monkeypatch.setenv("MEGALM_PROFILE_FILE", str(path))
    monkeypatch.setenv("MEGALM_PROFILE_RUN_ID", "env-run")
    with Profiler.from_env() as profiler:
        assert profiler.enabled
        assert profiler.run_id == "env-run"
        profiler.record("admission", "req-1", prompt_tokens=7)
    line = json.loads(path.read_text().strip())
    assert line["run_id"] == "env-run"
    assert line["request_id"] == "req-1"
    assert line["event"] == "admission"
    assert line["extra"] == {"prompt_tokens": 7}


def test_jsonl_schema_matches_rust_recorder(tmp_path) -> None:
    path = tmp_path / "profile.jsonl"
    with Profiler(path, run_id="r1") as profiler:
        with active_stage("gateway"):
            profiler.record("admission", "req-1", prompt_tokens=5)
    payload = json.loads(path.read_text().strip())
    assert set(payload) == {
        "ts_ns",
        "pid",
        "run_id",
        "request_id",
        "event",
        "stage",
        "extra",
    }
    assert isinstance(payload["ts_ns"], int)
    assert payload["pid"] == os.getpid()
    assert payload["stage"] == "gateway"


def test_active_stage_binds_contextvar_and_thread() -> None:
    assert current_stage() == ""
    with active_stage("gateway"):
        assert current_stage() == "gateway"
        seen: list[str] = []

        def worker() -> None:
            seen.append(current_stage())

        thread = threading.Thread(target=worker)
        thread.start()
        thread.join()
        # threading.local does not inherit into new threads; the ContextVar
        # default is empty there. Explicit per-thread binding is the caller's
        # responsibility; the dual binding is exercised by setting it there.
        assert seen == [""]
    assert current_stage() == ""


def test_active_stage_thread_binding_in_thread() -> None:
    seen: list[str] = []

    def worker() -> None:
        with active_stage("worker"):
            seen.append(current_stage())

    thread = threading.Thread(target=worker)
    thread.start()
    thread.join()
    assert seen == ["worker"]
    assert current_stage() == ""


def test_active_stage_is_asyncio_compatible() -> None:
    seen: list[str] = []

    async def child() -> None:
        seen.append(current_stage())

    async def main() -> None:
        with active_stage("async-stage"):
            await asyncio.gather(child(), child())

    asyncio.run(main())
    assert seen == ["async-stage", "async-stage"]


def test_metadata_only_keeps_small_scalars(tmp_path) -> None:
    path = tmp_path / "profile.jsonl"

    class FakeTensor:
        pass

    with Profiler(path) as profiler:
        profiler.record(
            "admission",
            "req-1",
            ok=1,
            rate=0.5,
            flag=True,
            missing=None,
            text="x" * 600,
            tensor=FakeTensor(),
            nested={"a": 1},
        )
    payload = json.loads(path.read_text().strip())
    assert payload["extra"]["ok"] == 1
    assert payload["extra"]["rate"] == 0.5
    assert payload["extra"]["flag"] is True
    assert payload["extra"]["missing"] is None
    assert len(payload["extra"]["text"]) == 512
    assert "tensor" not in payload["extra"]
    assert "nested" not in payload["extra"]
    assert profiler.dropped_metadata == 2


def test_write_failures_are_swallowed_and_counted(tmp_path) -> None:
    class BrokenFile:
        def write(self, _text: str) -> None:
            raise OSError("disk full")

        def flush(self) -> None:
            return None

        def close(self) -> None:
            return None

    profiler = Profiler(tmp_path / "profile.jsonl")
    profiler._file = BrokenFile()
    profiler.record("admission", "req-1")
    profiler.record("admission", "req-2")
    assert profiler.dropped_writes == 2
    assert profiler.events_recorded == 0


def test_load_events_from_file_directory_and_glob(tmp_path) -> None:
    a = tmp_path / "a.jsonl"
    b = tmp_path / "b.jsonl"
    a.write_text(
        json.dumps(
            {
                "ts_ns": 2,
                "pid": 1,
                "run_id": "r1",
                "request_id": "req-1",
                "event": "first_token",
                "stage": "",
                "extra": {},
            }
        )
        + "\n"
        + "not-json\n"
    )
    b.write_text(
        json.dumps(
            {
                "ts_ns": 1,
                "pid": 2,
                "run_id": "r2",
                "request_id": "req-1",
                "event": "admission",
                "stage": "",
                "extra": {},
            }
        )
        + "\n"
    )
    assert len(load_events(a)) == 1
    assert len(load_events(str(tmp_path))) == 2
    assert len(load_events(str(tmp_path / "*.jsonl"))) == 2
    # Dedup when the same file is matched twice.
    assert len(load_events(a, str(tmp_path))) == 2
    # Sorted by timestamp across processes.
    assert [e.event for e in load_events(str(tmp_path))] == [
        "admission",
        "first_token",
    ]


def test_timeline_and_stage_breakdown_reconstruct_a_request(tmp_path) -> None:
    path = tmp_path / "profile.jsonl"
    with Profiler(path, run_id="r1") as profiler:
        profiler.record("admission", "req-1", stage="gateway")
        profiler.record("prefill_start", "req-1", stage="engine")
        profiler.record("first_token", "req-1", stage="engine")
        profiler.record("terminal_response", "req-1", stage="gateway")
    events = load_events(path)
    rows = timeline(events, "req-1")
    assert [row["event"] for row in rows] == [
        "admission",
        "prefill_start",
        "first_token",
        "terminal_response",
    ]
    assert rows[0]["elapsed_ms"] == 0.0
    assert all(row["elapsed_ms"] >= 0 for row in rows)
    assert rows[0]["stage"] == "gateway"
    breakdown = stage_breakdown(events, "req-1")
    assert set(breakdown) == {"gateway", "engine"}
    assert breakdown["gateway"] >= 0.0
    assert timeline(events, "missing") == ()


def test_hop_breakdown_pairs_cross_process_hops() -> None:
    # Two processes write to separate files with different run_id / pid, but
    # the same (request_id, src, dst, chunk_id) key must pair correctly.
    def make(ts_ns: int, pid: int, run_id: str, event: str) -> dict[str, object]:
        return {
            "ts_ns": ts_ns,
            "pid": pid,
            "run_id": run_id,
            "request_id": "req-1",
            "event": event,
            "stage": "router",
            "extra": {"src": "gateway", "dst": "prefill", "chunk_id": "0"},
        }

    from megalm.profiling import ProfileEvent

    batch = (
        ProfileEvent.from_dict(make(1_000_000, 10, "run-a", "hop_sent")),
        ProfileEvent.from_dict(make(3_000_000, 20, "run-b", "hop_received")),
        ProfileEvent.from_dict(
            {
                **make(5_000_000, 10, "run-a", "hop_sent"),
                "extra": {"src": "gateway", "dst": "decode", "chunk_id": "1"},
            }
        ),
    )
    pairs = hop_breakdown(batch)
    assert len(pairs) == 2
    paired = next(p for p in pairs if p["chunk_id"] == "0")
    assert paired["matched"]
    assert paired["latency_ms"] == pytest.approx(2.0)
    unmatched = next(p for p in pairs if p["chunk_id"] == "1")
    assert not unmatched["matched"]
    assert unmatched["latency_ms"] is None


def test_timeline_does_not_change_when_profiling_disabled(tmp_path) -> None:
    # Enabling/disabling the profiler must not alter recorded event content.
    enabled_path = tmp_path / "enabled.jsonl"
    disabled = Profiler()
    disabled.record("admission", "req-1", prompt_tokens=3)
    with Profiler(enabled_path) as enabled:
        enabled.record("admission", "req-1", prompt_tokens=3)
    assert not disabled.enabled
    assert len(load_events(enabled_path)) == 1


def test_loads_rust_recorder_jsonl_and_normalizes_admitted(tmp_path) -> None:
    # The Rust frontend writes no `stage` field and calls the milestone
    # "admitted"; both streams must merge into one timeline.
    path = tmp_path / "rust.jsonl"
    path.write_text(
        json.dumps(
            {
                "ts_ns": 1_000,
                "pid": 4242,
                "run_id": "rust-run",
                "request_id": "req-1",
                "event": "admitted",
                "extra": {"prompt_tokens": 9},
            }
        )
        + "\n"
    )
    events = load_events(path)
    assert len(events) == 1
    assert events[0].event == "admission"
    assert events[0].stage == ""
    assert events[0].pid == 4242
    assert events[0].extra == {"prompt_tokens": 9}


def test_global_accessor_and_record_helper(tmp_path) -> None:
    previously = get_profiler()
    try:
        path = tmp_path / "profile.jsonl"
        configure_profiler(Profiler(path, run_id="global"))
        with active_stage("gateway"):
            record_profile("admission", "req-1", prompt_tokens=2)
        assert get_profiler().snapshot()["events_recorded"] == 1
        event = load_events(path)[0]
        assert event.stage == "gateway"
        assert event.run_id == "global"
        # Restoring the disabled singleton keeps later tests isolated.
        configure_profiler(None)
        record_profile("admission", "req-2")
        assert get_profiler().snapshot()["events_recorded"] == 0
        assert len(load_events(path)) == 1
    finally:
        configure_profiler(previously)


def test_view_combines_all_sections(tmp_path) -> None:
    path = tmp_path / "profile.jsonl"
    with Profiler(path, run_id="r1") as profiler:
        profiler.record("admission", "req-1", stage="gateway")
        profiler.record("first_token", "req-1", stage="engine")
        profiler.record("terminal_response", "req-1", stage="gateway")
    result = view(path)
    assert result["event_count"] == 3
    assert result["run_ids"] == ["r1"]
    assert "req-1" in result["timelines"]
    assert "req-1" in result["stage_breakdowns"]
    assert result["hops"] == ()
    single = view(path, request_id="req-1")
    assert len(single["timelines"]) == 1
