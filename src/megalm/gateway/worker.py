from __future__ import annotations

import argparse
import base64
from collections import Counter, deque
from dataclasses import asdict, dataclass, field
import logging
import os
import queue
import socket
import threading
import time
from typing import Any

import orjson

from megalm.contracts.request import (
    AudioOutputOptions,
    EngineEvent,
    EventKind,
    OutputModality,
)
from megalm.gateway.media_ipc import media_inputs_from_fds
from megalm.gateway.protocol import (
    FrameType,
    exception_to_wire,
    lora_from_wire,
    outputs_to_wire,
    receive_frame_with_fds,
    sampling_from_wire,
    send_frame,
)
from megalm.observability import configure_logging
from megalm.observability_otel import configure_telemetry
from megalm.profiling import (
    PREFILL_START,
    Profiler,
    configure_profiler,
    record as record_profile,
)
from megalm.runtime.embedding_engine import EmbeddingEngine, EmbeddingEngineConfig
from megalm.runtime.engine import (
    Engine,
    EngineConfig,
    MultimodalAdmission,
    PendingMultimodalBatch,
)
from megalm.runtime.scheduler import Completion

_LOGGER = logging.getLogger("megalm.worker")


def _template_kwargs_from_wire(payload: dict[str, Any]) -> dict[str, Any] | None:
    value = payload.get("chat_template_kwargs")
    if value is None:
        return None
    if not isinstance(value, dict):
        raise RuntimeError("chat template kwargs must be an object")
    return value


def _output_settings_from_wire(
    payload: dict[str, Any],
) -> tuple[tuple[OutputModality, ...], AudioOutputOptions | None]:
    raw_modalities = payload.get("output_modalities", ["text"])
    if not isinstance(raw_modalities, list):
        raise RuntimeError("output modalities are invalid")
    try:
        modalities = tuple(
            OutputModality(str(item))
            for item in raw_modalities
        )
    except ValueError as exc:
        raise RuntimeError("output modalities are invalid") from exc
    raw_audio = payload.get("audio_output")
    if raw_audio is None:
        audio_output = None
    elif isinstance(raw_audio, dict):
        audio_output = AudioOutputOptions(
            voice=str(raw_audio.get("voice", "Chelsie")),
            format=str(raw_audio.get("format", "wav")),
        )
    else:
        raise RuntimeError("audio output options are invalid")
    return modalities, audio_output


@dataclass(frozen=True, slots=True)
class _ReaderFailure:
    error: BaseException


@dataclass(slots=True)
class _PendingMultimodalCommand:
    command_id: int
    raw_submissions: tuple[dict[str, Any], ...]
    parsed: tuple[
        tuple[int, str, int, MultimodalAdmission] | BaseException,
        ...,
    ]
    valid: tuple[tuple[int, str, int, MultimodalAdmission], ...]
    engine_batch: PendingMultimodalBatch


@dataclass(slots=True)
class WorkerMetrics:
    command_frames: int = 0
    submit_batches: int = 0
    multimodal_batches: int = 0
    multimodal_submissions: int = 0
    admitted_requests: int = 0
    event_frames: int = 0
    emitted_events: int = 0
    sent_bytes: int = 0
    admission_wait_ns: int = 0
    idle_coalesced_batches: int = 0
    idle_target_shortfalls: int = 0
    video_admission_deferrals: int = 0
    video_admission_wait_ns: int = 0
    async_media_batches: int = 0
    async_media_completions: int = 0
    max_pending_media_batches: int = 0
    decode_steps_with_pending_media: int = 0
    submit_batch_histogram: Counter[int] = field(default_factory=Counter)
    multimodal_batch_histogram: Counter[int] = field(
        default_factory=Counter
    )

    def snapshot(self) -> dict[str, object]:
        return {
            "command_frames": self.command_frames,
            "submit_batches": self.submit_batches,
            "multimodal_batches": self.multimodal_batches,
            "multimodal_submissions": self.multimodal_submissions,
            "admitted_requests": self.admitted_requests,
            "event_frames": self.event_frames,
            "emitted_events": self.emitted_events,
            "sent_bytes": self.sent_bytes,
            "admission_wait_ms": self.admission_wait_ns / 1_000_000,
            "idle_coalesced_batches": self.idle_coalesced_batches,
            "idle_target_shortfalls": self.idle_target_shortfalls,
            "video_admission_deferrals": (
                self.video_admission_deferrals
            ),
            "video_admission_wait_ms": (
                self.video_admission_wait_ns / 1_000_000
            ),
            "async_media_batches": self.async_media_batches,
            "async_media_completions": self.async_media_completions,
            "max_pending_media_batches": self.max_pending_media_batches,
            "decode_steps_with_pending_media": (
                self.decode_steps_with_pending_media
            ),
            "submit_batch_histogram": dict(
                sorted(self.submit_batch_histogram.items())
            ),
            "multimodal_batch_histogram": dict(
                sorted(self.multimodal_batch_histogram.items())
            ),
        }


class ReplicaWorker:
    """GPU-owning process endpoint for the versioned local data plane."""

    def __init__(
        self,
        channel: socket.socket,
        config: EngineConfig,
        *,
        admission_max_wait_us: int = 0,
        admission_quiet_us: int = 0,
    ) -> None:
        if min(admission_max_wait_us, admission_quiet_us) < 0:
            raise ValueError("admission coalescing windows cannot be negative")
        if admission_quiet_us > admission_max_wait_us:
            raise ValueError(
                "admission quiet window cannot exceed the maximum window"
            )
        self.channel = channel
        self.config = config
        self.admission_max_wait_ns = admission_max_wait_us * 1_000
        self.admission_quiet_ns = admission_quiet_us * 1_000
        self.video_admission_max_defer_ns = (
            config.video_admission_max_defer_ms * 1_000_000
        )
        self.video_admission_defer_min_target = (
            config.video_admission_defer_min_target
        )
        self.commands: queue.Queue[
            tuple[FrameType, dict[str, Any], tuple[int, ...]]
            | _ReaderFailure
            | None
        ] = queue.Queue()
        self.metrics = WorkerMetrics()
        self.engine: Engine | None = None
        self.embedding_engine: EmbeddingEngine | None = None
        self.stopping = False
        self._admission_target = 1
        self._stream_ids: dict[str, int] = {}
        self._request_ids: dict[int, str] = {}
        self._pending_request_ids: set[str] = set()
        self._pending_stream_ids: set[int] = set()
        self._deferred_video_commands: deque[
            tuple[
                tuple[FrameType, dict[str, Any], tuple[int, ...]],
                int,
            ]
        ] = deque()
        self._pending_multimodal_commands: deque[
            _PendingMultimodalCommand
        ] = deque()
        self._reader = threading.Thread(
            target=self._read_commands,
            name="megalm-replica-ipc-reader",
            daemon=True,
        )

    def run(self) -> None:
        configure_profiler(Profiler.from_env(stage="engine"))
        if self.config.task == "embed":
            self._run_embedding()
            return
        try:
            self.engine = Engine(
                self.config,
                emit_engine_events=False,
            )
            self.metrics.sent_bytes += send_frame(
                self.channel,
                FrameType.READY,
                {"descriptor": self.engine.descriptor()},
            )
        except BaseException as exc:
            self._send_failure(exc)
            raise

        self._reader.start()
        pending_events: list[EngineEvent] = []
        pending_completions: list[Completion] = []
        try:
            while not self.stopping:
                engine = self._engine
                pending_events.extend(engine.drain_output_events())
                pending_events.extend(
                    self._finish_ready_multimodal_commands()
                )
                if engine.scheduler.is_idle:
                    self._send_events(tuple(pending_events))
                    pending_events.clear()
                    for completion in pending_completions:
                        self._send_completion(completion)
                    pending_completions.clear()
                    if not self._pending_multimodal_commands:
                        self._admission_target = 1
                    if self._deferred_video_commands:
                        command, deferred_ns = (
                            self._deferred_video_commands.popleft()
                        )
                        self.metrics.video_admission_wait_ns += (
                            time.perf_counter_ns() - deferred_ns
                        )
                    else:
                        try:
                            command = self.commands.get(
                                timeout=(
                                    0.001
                                    if (
                                        self._pending_multimodal_commands
                                        or engine.has_pending_output
                                    )
                                    else 0.05
                                )
                            )
                        except queue.Empty:
                            continue
                    pending_events.extend(self._handle(command))
                    if (
                        isinstance(command, tuple)
                        and command[0] is FrameType.SUBMIT_BATCH
                        and not self.stopping
                    ):
                        pending_events.extend(
                            self._coalesce_idle_admissions()
                        )
                while not self.stopping:
                    try:
                        command = self.commands.get_nowait()
                    except queue.Empty:
                        break
                    if (
                        not engine.scheduler.is_idle
                        and self._video_admission_defer_budget_ns() > 0
                        and self._is_video_admission(command)
                    ):
                        self._deferred_video_commands.append(
                            (command, time.perf_counter_ns())
                        )
                        self.metrics.video_admission_deferrals += 1
                        continue
                    pending_events.extend(self._handle(command))
                if (
                    not engine.scheduler.is_idle
                    and self._deferred_video_commands
                    and time.perf_counter_ns()
                    - self._deferred_video_commands[0][1]
                    >= self._video_admission_defer_budget_ns()
                    and not self.stopping
                ):
                    command, deferred_ns = (
                        self._deferred_video_commands.popleft()
                    )
                    self.metrics.video_admission_wait_ns += (
                        time.perf_counter_ns() - deferred_ns
                    )
                    pending_events.extend(self._handle(command))
                if not engine.scheduler.is_idle and not self.stopping:
                    if engine.tp_group.enabled and engine.tp_group.rank == 0:
                        engine.tp_group.broadcast_object({"kind": "step"})
                    if self._pending_multimodal_commands:
                        self.metrics.decode_steps_with_pending_media += 1
                    pending = engine.begin_step()
                    self._send_events(tuple(pending_events))
                    pending_events.clear()
                    for completion in pending_completions:
                        self._send_completion(completion)
                    pending_completions.clear()
                    pending_completions.append(
                        engine.finish_step_compact(pending)
                    )
        except BaseException as exc:
            self._send_failure(exc)
            raise
        finally:
            try:
                pending_events.extend(
                    self._engine.drain_output_events()
                )
                self._send_events(tuple(pending_events))
                for completion in pending_completions:
                    self._send_completion(completion)
            finally:
                self._engine.close()
                while self._deferred_video_commands:
                    command, _ = self._deferred_video_commands.popleft()
                    _close_fds(command[2])
                while self._pending_multimodal_commands:
                    pending = self._pending_multimodal_commands.popleft()
                    self._engine.cancel_multimodal_batch(
                        pending.engine_batch
                    )
                    for _, request_id, stream_id, _ in pending.valid:
                        self._pending_request_ids.discard(request_id)
                        self._pending_stream_ids.discard(stream_id)
                self.stopping = True
                try:
                    self.channel.shutdown(socket.SHUT_RDWR)
                except OSError:
                    pass
                self.channel.close()

    def _run_embedding(self) -> None:
        """Embedding workers never construct a generation scheduler."""

        def dtype_name() -> str:
            if self.config.dtype in {"float16", "bfloat16", "float32"}:
                return self.config.dtype
            return "float16"

        try:
            self.embedding_engine = EmbeddingEngine(
                EmbeddingEngineConfig(
                    model_path=str(self.config.model_path),
                    device=self.config.device,
                    dtype=dtype_name(),
                    max_batch_size=self.config.embedding_max_batch_size,
                    max_sequence_length=self.config.embedding_max_sequence_length,
                )
            )
            self.metrics.sent_bytes += send_frame(
                self.channel,
                FrameType.READY,
                {"descriptor": self.embedding_engine.descriptor()},
            )
        except BaseException as exc:
            self._send_failure(exc)
            raise
        self._reader.start()
        try:
            while not self.stopping:
                command = self.commands.get()
                if command is None:
                    self.stopping = True
                    break
                if isinstance(command, _ReaderFailure):
                    raise command.error
                frame_type, payload, fds = command
                self.metrics.command_frames += 1
                if fds:
                    _close_fds(fds)
                if frame_type is FrameType.STOP:
                    self.stopping = True
                elif frame_type is FrameType.SUBMIT_EMBEDDINGS:
                    payloads, deferred = self._drain_embedding_frames(payload)
                    self._handle_embedding_batch(payloads)
                    for deferred_type, deferred_payload in deferred:
                        if deferred_type is FrameType.INSPECT:
                            self.metrics.sent_bytes += send_frame(
                                self.channel,
                                FrameType.SNAPSHOT,
                                {
                                    "command_id": int(deferred_payload["command_id"]),
                                    "snapshot": {
                                        "embedding": self.embedding_engine.descriptor(),
                                        "metrics": self.metrics.snapshot(),
                                    },
                                },
                            )
                elif frame_type is FrameType.INSPECT:
                    self.metrics.sent_bytes += send_frame(
                        self.channel,
                        FrameType.SNAPSHOT,
                        {
                            "command_id": int(payload["command_id"]),
                            "snapshot": {
                                "embedding": self.embedding_engine.descriptor(),
                                "metrics": self.metrics.snapshot(),
                            },
                        },
                    )
                else:
                    raise RuntimeError(
                        f"embedding worker received unexpected {frame_type.name}"
                    )
        except BaseException as exc:
            self._send_failure(exc)
            raise
        finally:
            try:
                self.channel.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass
            self.channel.close()

    def _drain_embedding_frames(
        self, first: dict[str, Any]
    ) -> tuple[list[dict[str, Any]], list[tuple[FrameType, dict[str, Any]]]]:
        """Coalesce independent embedding frames into one encoder forward."""

        payloads = [first]
        deferred: list[tuple[FrameType, dict[str, Any]]] = []
        deadline = time.perf_counter_ns() + self.config.embedding_coalesce_wait_us * 1_000
        limit = self.config.embedding_max_batch_size
        while len(payloads) < limit and time.perf_counter_ns() < deadline:
            try:
                command = self.commands.get_nowait()
            except queue.Empty:
                break
            if command is None:
                self.stopping = True
                break
            if isinstance(command, _ReaderFailure):
                raise command.error
            frame_type, payload, fds = command
            self.metrics.command_frames += 1
            if fds:
                _close_fds(fds)
            if frame_type is FrameType.SUBMIT_EMBEDDINGS:
                payloads.append(payload)
            else:
                deferred.append((frame_type, payload))
        return payloads, deferred

    def _handle_embedding(self, payload: dict[str, Any]) -> None:
        self._handle_embedding_batch([payload])

    def _handle_embedding_batch(self, payloads: list[dict[str, Any]]) -> None:
        # Requests with different pooling options cannot share one forward's
        # output; group them by the option tuple while preserving input order.
        groups: dict[tuple[object, ...], list[tuple[int, int, dict[str, Any]]]] = {}
        for frame_index, payload in enumerate(payloads):
            submissions = payload.get("submissions")
            if not isinstance(submissions, list):
                raise RuntimeError("embedding submissions must be a list")
            key = (
                payload.get("dimensions"),
                payload.get("normalize"),
                payload.get("truncate_prompt_tokens"),
                str(payload.get("truncation_side", "right")),
            )
            bucket = groups.setdefault(key, [])
            for item_index, raw in enumerate(submissions):
                bucket.append((frame_index, item_index, raw))
        vectors: dict[tuple[int, int], list[float]] = {}
        failures: dict[tuple[object, ...], dict[str, Any]] = {}
        for key, items in groups.items():
            try:
                encoded = self.embedding_engine.encode(
                    [list(raw["token_ids"]) for _, _, raw in items],
                    dimensions=key[0],
                    normalize=key[1],
                    truncate_prompt_tokens=key[2],
                    truncation_side=str(key[3]),
                )
                for (frame_index, item_index, _), vector in zip(items, encoded):
                    vectors[(frame_index, item_index)] = vector
            except BaseException as exc:
                failures[key] = exception_to_wire(exc)
        for frame_index, payload in enumerate(payloads):
            results: list[dict[str, Any]] = []
            submissions = payload.get("submissions", [])
            key = (
                payload.get("dimensions"),
                payload.get("normalize"),
                payload.get("truncate_prompt_tokens"),
                str(payload.get("truncation_side", "right")),
            )
            for item_index, raw in enumerate(submissions):
                position = (frame_index, item_index)
                if position in vectors:
                    results.append(
                        {
                            "request_id": str(raw["request_id"]),
                            "ok": True,
                            "embedding": vectors[position],
                        }
                    )
                else:
                    results.append(
                        {
                            "request_id": str(raw.get("request_id", "")),
                            "ok": False,
                            "error": failures.get(
                                key,
                                {"message": "embedding request failed"},
                            ),
                        }
                    )
            self.metrics.sent_bytes += send_frame(
                self.channel,
                FrameType.EMBEDDING_RESULT,
                {"command_id": int(payload["command_id"]), "results": results},
            )

    def _video_admission_defer_budget_ns(self) -> int:
        target = self._admission_target
        minimum = self.video_admission_defer_min_target
        if (
            self.video_admission_max_defer_ns <= 0
            or target < minimum
        ):
            return 0
        full_target = minimum * 2
        return (
            self.video_admission_max_defer_ns
            * min(target, full_target)
            // full_target
        )

    @staticmethod
    def _is_video_admission(
        command: (
            tuple[FrameType, dict[str, Any], tuple[int, ...]]
            | _ReaderFailure
            | None
        ),
    ) -> bool:
        if not isinstance(command, tuple):
            return False
        frame_type, payload, _ = command
        # A low-priority media stream prevents host blocking, but a long video
        # tower can still consume enough SM bandwidth to perturb decode. Keep
        # the existing bounded video QoS gate for both submission shapes.
        if frame_type not in {
            FrameType.SUBMIT_MULTIMODAL,
            FrameType.SUBMIT_MULTIMODAL_BATCH,
        }:
            return False
        descriptors = payload.get("media")
        return isinstance(descriptors, list) and any(
            isinstance(item, dict) and item.get("kind") == "video"
            for item in descriptors
        )

    def _coalesce_idle_admissions(self) -> tuple[EngineEvent, ...]:
        """Fill a known concurrent wave before its first GPU prefill."""

        engine = self._engine
        target = min(
            self.config.max_num_sequences,
            self._admission_target,
        )
        if (
            self.admission_max_wait_ns == 0
            or target <= 1
            or engine.scheduler.active_request_count >= target
        ):
            return ()
        started_ns = time.perf_counter_ns()
        deadline_ns = started_ns + self.admission_max_wait_ns
        events: list[EngineEvent] = []
        extra_batches = 0
        while (
            not self.stopping
            and engine.scheduler.active_request_count < target
        ):
            now_ns = time.perf_counter_ns()
            if now_ns >= deadline_ns:
                break
            wait_ns = min(
                deadline_ns - now_ns,
                max(self.admission_quiet_ns, 1_000),
            )
            try:
                command = self.commands.get(
                    timeout=wait_ns / 1_000_000_000
                )
            except queue.Empty:
                continue
            is_submit = (
                isinstance(command, tuple)
                and command[0] is FrameType.SUBMIT_BATCH
            )
            events.extend(self._handle(command))
            if is_submit:
                extra_batches += 1
                target = min(
                    self.config.max_num_sequences,
                    max(target, self._admission_target),
                )
        self.metrics.admission_wait_ns += (
            time.perf_counter_ns() - started_ns
        )
        self.metrics.idle_coalesced_batches += extra_batches
        if engine.scheduler.active_request_count < target:
            self.metrics.idle_target_shortfalls += 1
        return tuple(events)

    @property
    def _engine(self) -> Engine:
        if self.engine is None:
            raise RuntimeError("Replica worker Engine is not initialized")
        return self.engine

    def _handle(
        self,
        command: (
            tuple[FrameType, dict[str, Any], tuple[int, ...]]
            | _ReaderFailure
            | None
        ),
    ) -> tuple[EngineEvent, ...]:
        if command is None:
            self.stopping = True
            return ()
        if isinstance(command, _ReaderFailure):
            raise command.error
        frame_type, payload, fds = command
        self.metrics.command_frames += 1
        if frame_type is FrameType.SUBMIT_MULTIMODAL:
            self._handle_submit_multimodal(payload, fds)
            return self._engine.scheduler.drain_events()
        if frame_type is FrameType.SUBMIT_MULTIMODAL_BATCH:
            self._handle_submit_multimodal_batch(payload, fds)
            return self._engine.scheduler.drain_events()
        if fds:
            _close_fds(fds)
            raise RuntimeError(
                f"{frame_type.name} cannot carry file descriptors"
            )
        if frame_type is FrameType.STOP:
            self.stopping = True
            return ()
        if frame_type is FrameType.SUBMIT_BATCH:
            self._handle_submit_batch(payload)
            return self._engine.scheduler.drain_events()
        if frame_type is FrameType.CANCEL:
            request_id = str(payload["request_id"])
            command_id = int(payload["command_id"])
            cancelled = False
            try:
                self._engine.cancel(request_id)
                cancelled = True
            except BaseException:
                pass
            self.metrics.sent_bytes += send_frame(
                self.channel,
                FrameType.SNAPSHOT,
                {
                    "command_id": command_id,
                    "snapshot": {"cancelled": cancelled},
                },
            )
            return self._engine.scheduler.drain_events()
        if frame_type is FrameType.INSPECT:
            command_id = int(payload["command_id"])
            op = payload.get("op")
            if op == "update_weights":
                # Same versioned INSPECT frame; the admin coordinator pauses
                # admissions and verifies digests around this rank-local swap.
                try:
                    result = self._engine.update_weights(str(payload["path"]))
                    snapshot = {"ok": True, "weights": result}
                except BaseException as exc:
                    snapshot = {
                        "ok": False,
                        "weights": {"error": str(exc) or type(exc).__name__},
                    }
                self.metrics.sent_bytes += send_frame(
                    self.channel,
                    FrameType.SNAPSHOT,
                    {"command_id": command_id, "snapshot": snapshot},
                )
                return ()
            if op in {"load_lora", "unload_lora"}:
                # Reuse the versioned INSPECT frame so no wire-format change is
                # needed for this control operation.
                try:
                    if op == "load_lora":
                        result = self._engine.load_lora(
                            str(payload["path"]),
                            name=payload.get("name"),
                            adapter_id=payload.get("id"),
                        )
                    else:
                        reference = (
                            payload.get("id")
                            if payload.get("id") is not None
                            else payload.get("name")
                        )
                        result = self._engine.unload_lora(reference)
                    manager = self._engine.lora_manager
                    snapshot = {
                        "ok": True,
                        "lora": {
                            **manager.stats(),
                            "adapters": manager.describe(),
                            "result": result,
                        },
                    }
                except BaseException as exc:
                    error = exception_to_wire(exc)
                    # Preserve stable client codes across the process boundary
                    # so load/unload endpoints keep their 4xx semantics.
                    code = getattr(exc, "code", None)
                    if isinstance(code, str):
                        error["code"] = code
                        error["retryable"] = bool(
                            getattr(exc, "retryable", True)
                        )
                    snapshot = {
                        "ok": False,
                        "lora": {"error": error},
                    }
            else:
                snapshot = self._runtime_snapshot()
            self.metrics.sent_bytes += send_frame(
                self.channel,
                FrameType.SNAPSHOT,
                {"command_id": command_id, "snapshot": snapshot},
            )
            return ()
        raise RuntimeError(f"unexpected command frame {frame_type.name}")

    def _handle_submit_batch(self, payload: dict[str, Any]) -> None:
        command_id = int(payload["command_id"])
        raw_submissions = payload.get("submissions")
        if not isinstance(raw_submissions, list) or not raw_submissions:
            raise RuntimeError("submit batch must contain requests")
        target_active_requests = int(
            payload.get("target_active_requests", len(raw_submissions))
        )
        if target_active_requests <= 0:
            raise RuntimeError("submit batch target must be positive")
        self._admission_target = min(
            self.config.max_num_sequences,
            max(self._admission_target, target_active_requests),
        )
        self.metrics.submit_batches += 1
        self.metrics.submit_batch_histogram[len(raw_submissions)] += 1
        results: list[dict[str, Any]] = []
        for raw in raw_submissions:
            request_id = str(raw["request_id"])
            stream_id = int(raw["stream_id"])
            if (
                stream_id <= 0
                or stream_id in self._request_ids
                or stream_id in self._pending_stream_ids
                or request_id in self._pending_request_ids
            ):
                raise RuntimeError("submit batch contains a duplicate stream id")
            try:
                output_modalities, audio_output = _output_settings_from_wire(
                    raw
                )
                self._engine.add_request(
                    tuple(int(token) for token in raw["token_ids"]),
                    max_new_tokens=int(raw["max_new_tokens"]),
                    sampling=sampling_from_wire(raw["sampling"]),
                    request_id=request_id,
                    output_modalities=output_modalities,
                    audio_output=audio_output,
                    lora_request=lora_from_wire(raw.get("lora")),
                )
            except BaseException as exc:
                results.append(
                    {
                        "request_id": request_id,
                        "ok": False,
                        "error": exception_to_wire(exc),
                    }
                )
            else:
                self._stream_ids[request_id] = stream_id
                self._request_ids[stream_id] = request_id
                self.metrics.admitted_requests += 1
                record_profile(PREFILL_START, request_id, stage="engine")
                results.append({"request_id": request_id, "ok": True})
        self.metrics.sent_bytes += send_frame(
            self.channel,
            FrameType.SUBMIT_RESULT,
            {"command_id": command_id, "results": results},
        )

    def _handle_submit_multimodal(
        self,
        payload: dict[str, Any],
        fds: tuple[int, ...],
    ) -> None:
        command_id = int(payload["command_id"])
        request_id = str(payload.get("request_id", ""))
        result: dict[str, Any] = {
            "command_id": command_id,
            "request_id": request_id,
        }
        admitted = False
        try:
            media = media_inputs_from_fds(payload.get("media"), fds)
            stream_id = int(payload["stream_id"])
            if (
                not request_id
                or stream_id <= 0
                or stream_id in self._request_ids
                or stream_id in self._pending_stream_ids
                or request_id in self._stream_ids
                or request_id in self._pending_request_ids
            ):
                raise RuntimeError(
                    "multimodal request or stream id is invalid"
                )
            raw_messages = payload.get("messages")
            if not isinstance(raw_messages, list) or not raw_messages:
                raise RuntimeError(
                    "multimodal request messages are invalid"
                )
            messages = tuple(
                dict(message)
                for message in raw_messages
                if isinstance(message, dict)
            )
            if len(messages) != len(raw_messages):
                raise RuntimeError(
                    "multimodal request contains an invalid message"
                )
            self._engine.add_multimodal_chat(
                messages,
                media.images,
                videos=media.videos,
                audios=media.audios,
                max_new_tokens=int(payload["max_new_tokens"]),
                sampling=sampling_from_wire(payload["sampling"]),
                request_id=request_id,
                open_thinking=bool(payload.get("open_thinking", False)),
                output_modalities=_output_settings_from_wire(
                    payload
                )[0],
                audio_output=_output_settings_from_wire(payload)[1],
                chat_template_kwargs=_template_kwargs_from_wire(payload),
            )
            admitted = True
            prompt_tokens = self._engine.scheduler.prompt_length(request_id)
        except BaseException as exc:
            if admitted:
                try:
                    self._engine.cancel(request_id)
                    self._engine.scheduler.drain_events()
                except BaseException:
                    pass
            result.update(
                {
                    "ok": False,
                    "error": exception_to_wire(exc),
                }
            )
        else:
            self._stream_ids[request_id] = stream_id
            self._request_ids[stream_id] = request_id
            self.metrics.multimodal_submissions += 1
            self.metrics.admitted_requests += 1
            result.update(
                {
                    "ok": True,
                    "prompt_tokens": prompt_tokens,
                }
            )
        self.metrics.sent_bytes += send_frame(
            self.channel,
            FrameType.SUBMIT_RESULT,
            result,
        )

    def _handle_submit_multimodal_batch(
        self,
        payload: dict[str, Any],
        fds: tuple[int, ...],
    ) -> None:
        command_id = int(payload["command_id"])
        raw_submissions = payload.get("submissions")
        descriptors = payload.get("media")
        if (
            not isinstance(raw_submissions, list)
            or not raw_submissions
            or not isinstance(descriptors, list)
            or len(descriptors) != len(fds)
        ):
            _close_fds(fds)
            raise RuntimeError(
                "multimodal batch payload or descriptors are invalid"
            )
        target_active_requests = int(
            payload.get("target_active_requests", len(raw_submissions))
        )
        if target_active_requests <= 0:
            _close_fds(fds)
            raise RuntimeError(
                "multimodal batch target must be positive"
            )
        self._admission_target = min(
            self.config.max_num_sequences,
            max(self._admission_target, target_active_requests),
        )
        ranges: list[tuple[int, int]] = []
        cursor = 0
        try:
            for raw in raw_submissions:
                if not isinstance(raw, dict):
                    raise RuntimeError(
                        "multimodal batch request is invalid"
                    )
                start = int(raw.get("media_start", -1))
                count = int(raw.get("media_count", 0))
                if start != cursor or count <= 0:
                    raise RuntimeError(
                        "multimodal batch media ranges are invalid"
                    )
                cursor += count
                if cursor > len(fds):
                    raise RuntimeError(
                        "multimodal batch media range exceeds descriptors"
                    )
                ranges.append((start, cursor))
            if cursor != len(fds):
                raise RuntimeError(
                    "multimodal batch leaves unused media descriptors"
                )
        except BaseException:
            _close_fds(fds)
            raise

        parsed: list[
            tuple[int, str, int, MultimodalAdmission] | BaseException
        ] = []
        seen_requests: set[str] = set()
        seen_streams: set[int] = set()
        for index, (raw, (start, end)) in enumerate(
            zip(raw_submissions, ranges, strict=True)
        ):
            assert isinstance(raw, dict)
            request_id = str(raw.get("request_id", ""))
            try:
                media = media_inputs_from_fds(
                    descriptors[start:end],
                    fds[start:end],
                )
                stream_id = int(raw["stream_id"])
                if (
                    not request_id
                    or stream_id <= 0
                    or request_id in seen_requests
                    or stream_id in seen_streams
                    or stream_id in self._request_ids
                    or stream_id in self._pending_stream_ids
                    or request_id in self._stream_ids
                    or request_id in self._pending_request_ids
                ):
                    raise RuntimeError(
                        "multimodal batch request or stream id is invalid"
                    )
                raw_messages = raw.get("messages")
                if not isinstance(raw_messages, list) or not raw_messages:
                    raise RuntimeError(
                        "multimodal request messages are invalid"
                    )
                messages = tuple(
                    dict(message)
                    for message in raw_messages
                    if isinstance(message, dict)
                )
                if len(messages) != len(raw_messages):
                    raise RuntimeError(
                        "multimodal request contains an invalid message"
                    )
                admission = MultimodalAdmission(
                    messages=messages,
                    images=media.images,
                    videos=media.videos,
                    audios=media.audios,
                    max_new_tokens=int(raw["max_new_tokens"]),
                    sampling=sampling_from_wire(raw["sampling"]),
                    request_id=request_id,
                    open_thinking=bool(
                        raw.get("open_thinking", False)
                    ),
                    output_modalities=_output_settings_from_wire(raw)[0],
                    audio_output=_output_settings_from_wire(raw)[1],
                    tools=tuple(
                        tool
                        for tool in raw.get("tools", ())
                        if isinstance(tool, dict)
                    ),
                    chat_template_kwargs=_template_kwargs_from_wire(raw),
                )
            except BaseException as exc:
                parsed.append(exc)
                continue
            seen_requests.add(request_id)
            seen_streams.add(stream_id)
            parsed.append((index, request_id, stream_id, admission))

        valid = tuple(
            entry
            for entry in parsed
            if not isinstance(entry, BaseException)
        )
        if valid:
            engine_batch = self._engine.begin_multimodal_batch(
                tuple(entry[3] for entry in valid)
            )
            self._pending_multimodal_commands.append(
                _PendingMultimodalCommand(
                    command_id=command_id,
                    raw_submissions=tuple(raw_submissions),
                    parsed=tuple(parsed),
                    valid=valid,
                    engine_batch=engine_batch,
                )
            )
            self._pending_request_ids.update(
                entry[1] for entry in valid
            )
            self._pending_stream_ids.update(
                entry[2] for entry in valid
            )
            self.metrics.async_media_batches += 1
            self.metrics.max_pending_media_batches = max(
                self.metrics.max_pending_media_batches,
                len(self._pending_multimodal_commands),
            )
            return
        else:
            engine_results = ()
        self._send_multimodal_batch_result(
            command_id=command_id,
            raw_submissions=tuple(raw_submissions),
            parsed=tuple(parsed),
            valid=valid,
            engine_results=engine_results,
        )

    def _finish_ready_multimodal_commands(
        self,
    ) -> tuple[EngineEvent, ...]:
        while self._pending_multimodal_commands:
            command = self._pending_multimodal_commands[0]
            if not self._engine.multimodal_batch_ready(
                command.engine_batch
            ):
                break
            self._pending_multimodal_commands.popleft()
            try:
                engine_results = self._engine.finish_multimodal_batch(
                    command.engine_batch
                )
                self._send_multimodal_batch_result(
                    command_id=command.command_id,
                    raw_submissions=command.raw_submissions,
                    parsed=command.parsed,
                    valid=command.valid,
                    engine_results=engine_results,
                )
            finally:
                for _, request_id, stream_id, _ in command.valid:
                    self._pending_request_ids.discard(request_id)
                    self._pending_stream_ids.discard(stream_id)
            self.metrics.async_media_completions += 1
        return self._engine.scheduler.drain_events()

    def _send_multimodal_batch_result(
        self,
        *,
        command_id: int,
        raw_submissions: tuple[dict[str, Any], ...],
        parsed: tuple[
            tuple[int, str, int, MultimodalAdmission] | BaseException,
            ...,
        ],
        valid: tuple[tuple[int, str, int, MultimodalAdmission], ...],
        engine_results: tuple[str | BaseException, ...],
    ) -> None:
        results_by_index: dict[int, str | BaseException] = {}
        for entry, result in zip(valid, engine_results, strict=True):
            results_by_index[entry[0]] = result

        results: list[dict[str, Any]] = []
        admitted = 0
        for index, raw in enumerate(raw_submissions):
            assert isinstance(raw, dict)
            request_id = str(raw.get("request_id", ""))
            parsed_entry = parsed[index]
            result = (
                parsed_entry
                if isinstance(parsed_entry, BaseException)
                else results_by_index[index]
            )
            if isinstance(result, BaseException):
                results.append(
                    {
                        "request_id": request_id,
                        "ok": False,
                        "error": exception_to_wire(result),
                    }
                )
                continue
            _, admitted_id, stream_id, _ = parsed_entry
            prompt_tokens = self._engine.scheduler.prompt_length(
                admitted_id
            )
            self._stream_ids[admitted_id] = stream_id
            self._request_ids[stream_id] = admitted_id
            admitted += 1
            results.append(
                {
                    "request_id": admitted_id,
                    "ok": True,
                    "prompt_tokens": prompt_tokens,
                }
            )
        self._admission_target = min(
            self.config.max_num_sequences,
            max(self._admission_target, admitted),
        )
        self.metrics.multimodal_batches += 1
        self.metrics.multimodal_submissions += admitted
        self.metrics.multimodal_batch_histogram[
            len(raw_submissions)
        ] += 1
        self.metrics.admitted_requests += admitted
        self.metrics.sent_bytes += send_frame(
            self.channel,
            FrameType.SUBMIT_RESULT,
            {
                "command_id": command_id,
                "results": results,
            },
        )

    def _send_events(self, events: tuple[EngineEvent, ...]) -> None:
        relayed = tuple(
            event
            for event in events
            if event.kind in {
                EventKind.TOKEN,
                EventKind.AUDIO,
                EventKind.TERMINAL,
            }
            and event.request_id in self._stream_ids
        )
        if not relayed:
            return
        outputs = outputs_to_wire(relayed, self._stream_ids)
        self.metrics.sent_bytes += send_frame(
            self.channel,
            FrameType.OUTPUTS,
            {"outputs": outputs},
        )
        self.metrics.event_frames += 1
        self.metrics.emitted_events += len(relayed)
        for event in relayed:
            if event.kind is EventKind.TERMINAL:
                stream_id = self._stream_ids.pop(event.request_id)
                self._request_ids.pop(stream_id)
                self._engine.scheduler.forget(event.request_id)

    def _send_completion(self, completion: Completion) -> None:
        grouped: dict[str, list[Any]] = {}
        emitted_events = 0
        terminal_request_ids: list[str] = []
        label_ids = completion.output_label_ids
        label_logprobs = completion.output_label_logprobs
        for index, (request_id, token_id, finish_reason, logprob) in enumerate(
            zip(
                completion.output_request_ids,
                completion.output_token_ids,
                completion.finish_reasons,
                completion.output_logprobs,
                strict=True,
            )
        ):
            stream_id = self._stream_ids.get(request_id)
            if stream_id is None:
                continue
            output = grouped.setdefault(
                request_id,
                [stream_id, [], [], None, None, None, None],
            )
            top_rows = (
                completion.output_top_logprobs[index]
                if index < len(completion.output_top_logprobs)
                else ()
            )
            if top_rows:
                if output[6] is None:
                    output[6] = []
                output[6].extend(
                    [[token, value] for token, value in top_rows]
                )
            read_ids = label_ids[index] if index < len(label_ids) else ()
            read_logprobs = (
                label_logprobs[index] if index < len(label_logprobs) else ()
            )
            if read_ids:
                # Decision reads reuse the token slot to carry the label ids
                # and the logprob slot to carry their raw log-probabilities.
                output[1].extend(read_ids)
                if read_logprobs:
                    if output[5] is None:
                        output[5] = []
                    output[5].extend(read_logprobs)
            else:
                output[1].append(token_id)
                if logprob is not None:
                    if output[5] is None:
                        output[5] = []
                    output[5].append(logprob)
            emitted_events += 1 + int(finish_reason is not None)
            if finish_reason is not None:
                if output[3] is not None:
                    raise RuntimeError(
                        "completion repeats a terminal request"
                    )
                output[3] = finish_reason.value
                terminal_request_ids.append(request_id)
        outputs = list(grouped.values())
        if not outputs:
            return
        self.metrics.sent_bytes += send_frame(
            self.channel,
            FrameType.OUTPUTS,
            {"outputs": outputs},
        )
        self.metrics.event_frames += 1
        self.metrics.emitted_events += emitted_events
        for request_id in terminal_request_ids:
            stream_id = self._stream_ids.pop(request_id)
            self._request_ids.pop(stream_id)
            self._engine.scheduler.forget(request_id)

    def _runtime_snapshot(self) -> dict[str, Any]:
        engine = self._engine
        graph = engine.graph_runner
        vision_stats = engine.vision_cache.stats()
        media_cache = {
            **asdict(vision_stats),
            "hit_rate": vision_stats.hit_rate,
        }
        return {
            "engine": engine.metrics.snapshot(),
            "worker": self.metrics.snapshot(),
            "scheduler": engine.scheduler.metrics_snapshot(),
            "cache": engine.cache_manager.stats().snapshot(),
            "media_cache": media_cache,
            "vision_cache": media_cache,
            "cuda_graph": (
                None
                if graph is None
                else {
                    "buckets": graph.batch_sizes,
                    "captures": graph.metrics.captures,
                    "greedy_captures": graph.metrics.greedy_captures,
                    "speculative_captures": (
                        graph.metrics.speculative_captures
                    ),
                    "replays": graph.metrics.replays,
                    "model_replays": graph.metrics.model_replays,
                    "greedy_replays": graph.metrics.greedy_replays,
                    "speculative_replays": (
                        graph.metrics.speculative_replays
                    ),
                    "speculative_lengths": graph.speculative_lengths,
                    "eager_fallbacks": graph.metrics.eager_fallbacks,
                    "capture_failures": graph.metrics.capture_failures,
                    "capture_capable": graph.metrics.capture_capable,
                    "memory_bytes": graph.memory_bytes,
                }
            ),
            "compile": engine.compile_report.as_dict(),
        }

    def _send_failure(self, error: BaseException) -> None:
        try:
            send_frame(
                self.channel,
                FrameType.FAILURE,
                {"error": exception_to_wire(error)},
            )
        except BaseException:
            pass

    def _read_commands(self) -> None:
        try:
            while True:
                frame = receive_frame_with_fds(self.channel)
                if frame is None:
                    self.commands.put(None)
                    return
                self.commands.put(frame)
        except BaseException as exc:
            self.commands.put(_ReaderFailure(exc))


def _close_fds(fds: tuple[int, ...]) -> None:
    for fd in fds:
        try:
            os.close(fd)
        except OSError:
            pass


def _decode_config(encoded: str) -> EngineConfig:
    try:
        payload = orjson.loads(base64.urlsafe_b64decode(encoded.encode()))
    except BaseException as exc:
        raise RuntimeError("invalid Replica worker configuration") from exc
    if not isinstance(payload, dict):
        raise RuntimeError("Replica worker configuration must be an object")
    return EngineConfig(**payload)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run one MegaLM GPU Replica worker.")
    parser.add_argument("--fd", type=int, required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--admission-max-wait-us", type=int, default=0)
    parser.add_argument("--admission-quiet-us", type=int, default=0)
    parser.add_argument(
        "--log-level",
        choices=("DEBUG", "INFO", "WARNING", "ERROR"),
        default="INFO",
    )
    args = parser.parse_args()
    configure_logging(args.log_level)
    configure_telemetry(
        service_name="megalm-worker",
        replica_mode="process",
        process_kind="replica_worker",
    )
    channel = socket.socket(fileno=args.fd)
    channel.set_inheritable(False)
    worker = ReplicaWorker(
        channel,
        _decode_config(args.config),
        admission_max_wait_us=args.admission_max_wait_us,
        admission_quiet_us=args.admission_quiet_us,
    )
    try:
        worker.run()
    except BaseException:
        _LOGGER.exception("replica_worker_failed")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
