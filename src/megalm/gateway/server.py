from __future__ import annotations

import argparse
import asyncio
import base64
from collections import Counter
from concurrent.futures import Future, ThreadPoolExecutor
from contextlib import asynccontextmanager
from dataclasses import asdict, dataclass, field
import logging
import os
import queue
from pathlib import Path
import socket
import subprocess
import sys
import threading
import time
from typing import Annotated, Any, AsyncIterator, Literal, Mapping, Sequence
from uuid import uuid4

import orjson
from fastapi import FastAPI, HTTPException
from fastapi.responses import Response, StreamingResponse
from pydantic import BaseModel, ConfigDict, Field

from megalm.contracts.request import (
    AudioInput,
    AudioOutputOptions,
    EngineError,
    EngineEvent,
    EventKind,
    FinishReason,
    ImageInput,
    OutputModality,
    SamplingParams,
    VideoInput,
)
from megalm.gateway.body_limit import RequestBodyLimitMiddleware
from megalm.gateway.http_metrics import (
    GatewayMetrics,
    RequestMetricsMiddleware,
)
from megalm.observability_otel import configure_telemetry, mirror_snapshot
from megalm.gateway.chat_media import MAX_CHAT_REQUEST_BYTES
from megalm.gateway.media_fetch import MEDIA_FETCH_CONFIG, MEDIA_FETCH_STATS
from megalm.gateway.media_ipc import MAX_MEDIA_FDS, seal_media_inputs
from megalm.gateway.omni_admission import (
    AsyncOmniAdmissionBatcher,
    MultimodalReplicaInput,
)
from megalm.gateway.protocol import (
    FrameType,
    output_from_wire,
    receive_frame,
    sampling_to_wire,
    send_frame,
    send_frame_with_fds,
)
from megalm.gateway.replica_types import (
    GatewayError,
    ReplicaFailure,
    ReplicaOutput,
    StreamBatch,
)
from megalm.gateway.responses import MegaLMJSONResponse
from megalm.megalm_rs import prometheus_metrics
from megalm.gateway.detokenize import IncrementalDecoder
from megalm.observability import (
    configure_logging,
    histogram_totals,
    log_event,
)
from megalm.runtime.engine import (
    Engine,
    EngineConfig,
    MultimodalAdmission,
)
from megalm.runtime.scheduler import SchedulerOverloaded
from megalm.runtime.tokenizer import FrontendError, ModelTokenizer

_LOGGER = logging.getLogger("megalm.gateway")


class CompletionRequest(BaseModel):
    """Supported OpenAI Completions request surface.

    Unknown extension fields are accepted so clients can share a request
    schema with vLLM and SGLang. Unsupported semantic features are rejected
    explicitly below instead of being silently ignored.
    """

    model_config = ConfigDict(extra="allow")

    model: str
    prompt: str | list[int]
    max_tokens: Annotated[int, Field(gt=0)] = 16
    stream: bool = False
    temperature: Annotated[float, Field(ge=0)] = 0.0
    top_p: Annotated[float, Field(gt=0, le=1)] = 1.0
    top_k: Annotated[int, Field(ge=0)] = 0
    min_p: Annotated[float, Field(ge=0, le=1)] = 0.0
    repetition_penalty: Annotated[float, Field(gt=0)] = 1.0
    seed: int | None = None
    stop_token_ids: list[int] = Field(default_factory=list)
    ignore_eos: bool = False
    n: Annotated[int, Field(gt=0)] = 1
    best_of: Annotated[int, Field(gt=0)] = 1


@dataclass(frozen=True, slots=True)
class _TokenizeJob:
    text: str
    result: asyncio.Future[tuple[int, ...]]


@dataclass(slots=True)
class MicrobatchMetrics:
    batches: int = 0
    jobs: int = 0
    batch_histogram: Counter[int] = field(default_factory=Counter)

    def record(self, batch_size: int) -> None:
        self.batches += 1
        self.jobs += batch_size
        self.batch_histogram[batch_size] += 1

    def snapshot(self) -> dict[str, object]:
        return {
            "batches": self.batches,
            "jobs": self.jobs,
            "batch_histogram": dict(sorted(self.batch_histogram.items())),
        }


class AsyncTokenizerBatcher:
    """Bounded Gateway-owned tokenizer microbatcher.

    One event-loop yield coalesces sibling HTTP tasks without a fixed latency
    timer. The Rust tokenizer then processes the whole burst through its native
    encode_batch path; the GPU Replica still receives only token IDs.
    """

    def __init__(
        self,
        tokenizer: ModelTokenizer,
        *,
        max_batch_size: int = 256,
        max_pending: int = 4096,
    ) -> None:
        if min(max_batch_size, max_pending) <= 0:
            raise GatewayError("tokenizer batch limits must be positive")
        self.tokenizer = tokenizer
        self.max_batch_size = max_batch_size
        self._jobs: asyncio.Queue[_TokenizeJob | None] = asyncio.Queue(
            maxsize=max_pending
        )
        self._task: asyncio.Task[None] | None = None
        self._failure: BaseException | None = None
        self._executor = ThreadPoolExecutor(
            max_workers=1,
            thread_name_prefix="megalm-tokenizer",
        )
        self.metrics = MicrobatchMetrics()

    def start(self) -> None:
        if self._task is not None:
            raise GatewayError("tokenizer batcher is already running")
        self._task = asyncio.create_task(
            self._run(),
            name="megalm-tokenizer-batcher",
        )

    async def stop(self) -> None:
        task = self._task
        if task is None:
            return
        await self._jobs.put(None)
        try:
            await task
        finally:
            self._task = None
            self._executor.shutdown(wait=True, cancel_futures=True)

    async def encode(self, text: str) -> tuple[int, ...]:
        if self._failure is not None:
            raise GatewayError("tokenizer batcher is unavailable") from self._failure
        loop = asyncio.get_running_loop()
        result: asyncio.Future[tuple[int, ...]] = loop.create_future()
        try:
            self._jobs.put_nowait(_TokenizeJob(text=text, result=result))
        except asyncio.QueueFull as exc:
            raise GatewayError("tokenizer input queue is overloaded") from exc
        return await result

    async def _run(self) -> None:
        try:
            while True:
                first = await self._jobs.get()
                if first is None:
                    return
                jobs = [first]
                await asyncio.sleep(0)
                while len(jobs) < self.max_batch_size:
                    try:
                        job = self._jobs.get_nowait()
                    except asyncio.QueueEmpty:
                        break
                    if job is None:
                        self._jobs.put_nowait(None)
                        break
                    jobs.append(job)
                self.metrics.record(len(jobs))
                try:
                    encoded = await asyncio.get_running_loop().run_in_executor(
                        self._executor,
                        self.tokenizer.encode_batch,
                        tuple(job.text for job in jobs),
                    )
                except BaseException as exc:
                    for job in jobs:
                        if not job.result.done():
                            job.result.set_exception(exc)
                    continue
                for job, token_ids in zip(jobs, encoded, strict=True):
                    if not job.result.done():
                        job.result.set_result(token_ids)
        except BaseException as exc:
            self._failure = exc
            while True:
                try:
                    job = self._jobs.get_nowait()
                except asyncio.QueueEmpty:
                    break
                if job is not None and not job.result.done():
                    job.result.set_exception(exc)
            raise

@dataclass(frozen=True, slots=True)
class RequestStream:
    request_id: str
    prompt_tokens: int
    events: asyncio.Queue[ReplicaOutput]


@dataclass(frozen=True, slots=True)
class _ReplicaInput:
    token_ids: tuple[int, ...]
    max_new_tokens: int
    sampling: SamplingParams
    request_id: str
    output_modalities: tuple[OutputModality, ...] = (OutputModality.TEXT,)
    audio_output: AudioOutputOptions | None = None


@dataclass(frozen=True, slots=True)
class _AdmissionJob:
    submission: _ReplicaInput
    result: asyncio.Future[RequestStream]


@dataclass(frozen=True, slots=True)
class _Submit:
    token_ids: tuple[int, ...]
    max_new_tokens: int
    sampling: SamplingParams
    request_id: str
    output_modalities: tuple[OutputModality, ...]
    audio_output: AudioOutputOptions | None
    loop: asyncio.AbstractEventLoop
    events: asyncio.Queue[ReplicaOutput]
    accepted: Future[str]


@dataclass(frozen=True, slots=True)
class _SubmitBatch:
    submissions: tuple[_Submit, ...]


@dataclass(frozen=True, slots=True)
class _SubmitMultimodal:
    messages: tuple[Mapping[str, Any], ...]
    images: tuple[ImageInput, ...]
    videos: tuple[VideoInput, ...]
    audios: tuple[AudioInput, ...]
    max_new_tokens: int
    sampling: SamplingParams
    request_id: str
    open_thinking: bool
    output_modalities: tuple[OutputModality, ...]
    audio_output: AudioOutputOptions | None
    loop: asyncio.AbstractEventLoop
    events: asyncio.Queue[ReplicaOutput]
    accepted: Future[tuple[str, int]]
    tools: tuple[dict[str, Any], ...] = ()


@dataclass(frozen=True, slots=True)
class _SubmitMultimodalBatch:
    submissions: tuple[_SubmitMultimodal, ...]


@dataclass(frozen=True, slots=True)
class _Cancel:
    request_id: str
    ack: Future[bool] = field(default_factory=Future)


@dataclass(frozen=True, slots=True)
class _Stop:
    pass


@dataclass(frozen=True, slots=True)
class _Inspect:
    result: Future[dict[str, Any]]


_Command = (
    _Submit
    | _SubmitBatch
    | _SubmitMultimodal
    | _SubmitMultimodalBatch
    | _Cancel
    | _Stop
    | _Inspect
)


@dataclass(frozen=True, slots=True)
class _Subscriber:
    request_id: str
    loop: asyncio.AbstractEventLoop
    events: asyncio.Queue[ReplicaOutput]
    stream_id: int | None = None


@dataclass(slots=True)
class ReplicaMetrics:
    admitted_requests: int = 0
    admission_wait_ns: int = 0
    admission_burst_histogram: Counter[int] = field(default_factory=Counter)
    event_batches: int = 0
    delivered_events: int = 0
    delivered_output_batches: int = 0
    relay_step_histogram: Counter[int] = field(default_factory=Counter)
    peak_frontend_inflight: int = 0
    ttft_ms_total: float = 0.0
    ttft_samples: int = 0
    tpot_ms_total: float = 0.0
    tpot_samples: int = 0
    ttft_ms_histogram: Counter[int] = field(default_factory=Counter)
    tpot_ms_histogram: Counter[int] = field(default_factory=Counter)
    e2e_ms_histogram: Counter[int] = field(default_factory=Counter)
    finish_reasons: Counter[str] = field(default_factory=Counter)
    # Rust parser counters, folded in by chat_api._record_parser_telemetry.
    parser_pushes: int = 0
    parser_finishes: int = 0
    parser_calls: int = 0
    parser_diagnostics: int = 0
    parser_bytes: int = 0

    def snapshot(self) -> dict[str, object]:
        return {
            "admitted_requests": self.admitted_requests,
            "admission_wait_ms": self.admission_wait_ns / 1_000_000,
            "admission_burst_histogram": dict(
                sorted(self.admission_burst_histogram.items())
            ),
            "event_batches": self.event_batches,
            "delivered_events": self.delivered_events,
            "delivered_output_batches": self.delivered_output_batches,
            "relay_step_histogram": dict(
                sorted(self.relay_step_histogram.items())
            ),
            "peak_frontend_inflight": self.peak_frontend_inflight,
            "ttft_ms_mean": (
                self.ttft_ms_total / self.ttft_samples
                if self.ttft_samples
                else 0.0
            ),
            "ttft_samples": self.ttft_samples,
            "tpot_ms_mean": (
                self.tpot_ms_total / self.tpot_samples
                if self.tpot_samples
                else 0.0
            ),
            "tpot_samples": self.tpot_samples,
            "ttft_ms_histogram": histogram_totals(
                dict(sorted(self.ttft_ms_histogram.items())),
                total_sum=self.ttft_ms_total,
                total_count=self.ttft_samples,
            ),
            "tpot_ms_histogram": histogram_totals(
                dict(sorted(self.tpot_ms_histogram.items())),
                total_sum=self.tpot_ms_total,
                total_count=self.tpot_samples,
            ),
            "e2e_ms_histogram": dict(sorted(self.e2e_ms_histogram.items())),
            "finish_reasons": dict(sorted(self.finish_reasons.items())),
            "parser": {
                "pushes": self.parser_pushes,
                "finishes": self.parser_finishes,
                "calls": self.parser_calls,
                "diagnostics": self.parser_diagnostics,
                "bytes": self.parser_bytes,
            },
        }


class EmbeddedReplica:
    """One-owner TP=1 Replica loop behind a thread-safe token/event boundary."""

    replica_mode = "embedded"

    def __init__(
        self,
        config: EngineConfig,
        *,
        max_buffered_events: int = 512,
        admission_max_wait_us: int = 0,
        admission_quiet_us: int = 0,
        stream_flush_steps: int = 1,
        stream_flush_min_batch: int = 8,
    ) -> None:
        if min(
            max_buffered_events,
            stream_flush_steps,
            stream_flush_min_batch,
        ) <= 0:
            raise GatewayError("Replica buffer and relay limits must be positive")
        if min(admission_max_wait_us, admission_quiet_us) < 0:
            raise GatewayError("admission coalescing windows cannot be negative")
        if admission_quiet_us > admission_max_wait_us:
            raise GatewayError(
                "admission quiet window cannot exceed the maximum window"
            )
        self.config = config
        self.max_buffered_events = max_buffered_events
        self.admission_max_wait_ns = admission_max_wait_us * 1_000
        self.admission_quiet_ns = admission_quiet_us * 1_000
        self.stream_flush_steps = stream_flush_steps
        self.stream_flush_min_batch = stream_flush_min_batch
        self._commands: queue.Queue[_Command] = queue.Queue()
        self._ready = threading.Event()
        self._thread = threading.Thread(
            target=self._run,
            name="megalm-replica",
            daemon=True,
        )
        self._engine: Engine | None = None
        self._failure: BaseException | None = None
        self._subscribers: dict[str, _Subscriber] = {}
        self._deferred_events: list[EngineEvent] = []
        self._deferred_event_steps = 0
        self._started_streams: set[str] = set()
        self._stopping = False
        self.metrics = ReplicaMetrics()
        self._frontend_lock = threading.Lock()
        self._frontend_inflight = 0

    @property
    def engine(self) -> Engine:
        engine = self._engine
        if engine is None:
            if self._failure is not None:
                raise GatewayError("Replica initialization failed") from self._failure
            raise GatewayError("Replica is not ready")
        return engine

    def start(self, *, timeout: float = 600.0) -> None:
        self._thread.start()
        if not self._ready.wait(timeout):
            raise GatewayError("Replica initialization timed out")
        if self._failure is not None:
            raise GatewayError("Replica initialization failed") from self._failure

    def stop(self, *, timeout: float = 30.0) -> None:
        if not self._thread.is_alive():
            return
        self._commands.put(_Stop())
        self._thread.join(timeout)
        if self._thread.is_alive():
            raise GatewayError("Replica did not stop within the shutdown deadline")

    async def submit(
        self,
        token_ids: tuple[int, ...],
        *,
        max_new_tokens: int,
        sampling: SamplingParams,
        request_id: str | None = None,
        output_modalities: tuple[OutputModality, ...] = (
            OutputModality.TEXT,
        ),
        audio_output: AudioOutputOptions | None = None,
    ) -> RequestStream:
        resolved_id = request_id or uuid4().hex
        result = (
            await self.submit_batch(
                (
                    _ReplicaInput(
                        token_ids=token_ids,
                        max_new_tokens=max_new_tokens,
                        sampling=sampling,
                        request_id=resolved_id,
                        output_modalities=output_modalities,
                        audio_output=audio_output,
                    ),
                )
            )
        )[0]
        if isinstance(result, BaseException):
            raise result
        return result

    async def submit_batch(
        self,
        submissions: tuple[_ReplicaInput, ...],
    ) -> tuple[RequestStream | BaseException, ...]:
        if not submissions:
            return ()
        if self._failure is not None:
            raise GatewayError("Replica is unavailable") from self._failure
        loop = asyncio.get_running_loop()
        commands = tuple(
            _Submit(
                token_ids=submission.token_ids,
                max_new_tokens=submission.max_new_tokens,
                sampling=submission.sampling,
                request_id=submission.request_id,
                output_modalities=submission.output_modalities,
                audio_output=submission.audio_output,
                loop=loop,
                events=asyncio.Queue(maxsize=self.max_buffered_events),
                accepted=Future(),
            )
            for submission in submissions
        )
        self._commands.put(_SubmitBatch(commands))
        accepted = await asyncio.gather(
            *(
                asyncio.wrap_future(command.accepted)
                for command in commands
            ),
            return_exceptions=True,
        )
        outputs: list[RequestStream | BaseException] = []
        for command, accepted_id in zip(commands, accepted, strict=True):
            if isinstance(accepted_id, BaseException):
                outputs.append(accepted_id)
            else:
                outputs.append(
                    RequestStream(
                        request_id=accepted_id,
                        prompt_tokens=len(command.token_ids),
                        events=command.events,
                    )
                )
        return tuple(outputs)

    async def submit_multimodal(
        self,
        messages: Sequence[Mapping[str, Any]],
        images: Sequence[ImageInput],
        *,
        videos: Sequence[VideoInput] = (),
        audios: Sequence[AudioInput] = (),
        max_new_tokens: int,
        sampling: SamplingParams,
        request_id: str | None = None,
        open_thinking: bool = False,
        output_modalities: tuple[OutputModality, ...] = (
            OutputModality.TEXT,
        ),
        audio_output: AudioOutputOptions | None = None,
        tools: tuple[dict[str, Any], ...] = (),
    ) -> RequestStream:
        output = (
            await self.submit_multimodal_batch(
                (
                    MultimodalReplicaInput(
                        messages=tuple(messages),
                        images=tuple(images),
                        videos=tuple(videos),
                        audios=tuple(audios),
                        max_new_tokens=max_new_tokens,
                        sampling=sampling,
                        request_id=request_id or uuid4().hex,
                        open_thinking=open_thinking,
                        output_modalities=output_modalities,
                        audio_output=audio_output,
                        tools=tools,
                    ),
                )
            )
        )[0]
        if isinstance(output, BaseException):
            raise output
        return output

    async def submit_multimodal_batch(
        self,
        submissions: tuple[MultimodalReplicaInput, ...],
    ) -> tuple[RequestStream | BaseException, ...]:
        if not submissions:
            return ()
        if self._failure is not None:
            raise GatewayError("Replica is unavailable") from self._failure
        loop = asyncio.get_running_loop()
        commands = tuple(
            _SubmitMultimodal(
                messages=submission.messages,
                images=submission.images,
                videos=submission.videos,
                audios=submission.audios,
                max_new_tokens=submission.max_new_tokens,
                sampling=submission.sampling,
                request_id=submission.request_id,
                open_thinking=submission.open_thinking,
                output_modalities=submission.output_modalities,
                audio_output=submission.audio_output,
                tools=submission.tools,
                loop=loop,
                events=asyncio.Queue(
                    maxsize=self.max_buffered_events
                ),
                accepted=Future(),
            )
            for submission in submissions
        )
        self._commands.put(_SubmitMultimodalBatch(commands))
        accepted = await asyncio.gather(
            *(
                asyncio.wrap_future(command.accepted)
                for command in commands
            ),
            return_exceptions=True,
        )
        outputs: list[RequestStream | BaseException] = []
        for command, result in zip(commands, accepted, strict=True):
            if isinstance(result, BaseException):
                outputs.append(result)
            else:
                accepted_id, prompt_tokens = result
                outputs.append(
                    RequestStream(
                        request_id=accepted_id,
                        prompt_tokens=prompt_tokens,
                        events=command.events,
                    )
                )
        return tuple(outputs)

    def cancel(self, request_id: str) -> Future[bool]:
        command = _Cancel(request_id)
        self._commands.put(command)
        return command.ack

    def frontend_request_started(self) -> None:
        with self._frontend_lock:
            self._frontend_inflight += 1
            self.metrics.peak_frontend_inflight = max(
                self.metrics.peak_frontend_inflight,
                self._frontend_inflight,
            )

    def frontend_request_finished(self) -> None:
        with self._frontend_lock:
            if self._frontend_inflight <= 0:
                raise GatewayError("frontend request accounting underflow")
            self._frontend_inflight -= 1

    @property
    def frontend_inflight(self) -> int:
        with self._frontend_lock:
            return self._frontend_inflight

    async def snapshot(self) -> dict[str, Any]:
        if self._failure is not None:
            raise GatewayError("Replica is unavailable") from self._failure
        result: Future[dict[str, Any]] = Future()
        self._commands.put(_Inspect(result))
        return await asyncio.wrap_future(result)

    async def describe(self) -> dict[str, Any]:
        return self.engine.descriptor()

    def _run(self) -> None:
        try:
            self._engine = Engine(self.config)
        except BaseException as exc:
            self._failure = exc
            self._ready.set()
            return
        self._ready.set()
        try:
            while not self._stopping:
                engine = self.engine
                output_events = engine.drain_output_events()
                if output_events:
                    self._deferred_events.extend(output_events)
                if engine.scheduler.is_idle:
                    self._relay_deferred_events(force=True)
                admission_burst = 0
                if engine.scheduler.is_idle:
                    try:
                        command = self._commands.get(
                            timeout=(
                                0.005
                                if engine.has_pending_output
                                else 0.05
                            )
                        )
                    except queue.Empty:
                        continue
                    admission_burst += self._handle(command)
                    if admission_burst:
                        admission_burst += self._coalesce_admissions()
                admission_burst += self._drain_commands()
                if admission_burst:
                    self.metrics.admission_burst_histogram[admission_burst] += 1
                if not engine.scheduler.is_idle:
                    pending = engine.begin_step()
                    self._relay_deferred_events()
                    completed_events = engine.finish_step(pending)
                    if completed_events:
                        self._deferred_events.extend(completed_events)
                        if any(
                            event.kind is EventKind.TOKEN
                            for event in completed_events
                        ):
                            self._deferred_event_steps += 1
        except BaseException as exc:
            self._failure = exc
            self._fail_all(exc)
        finally:
            self._relay_deferred_events(force=True)
            if self._subscribers:
                self._fail_all(GatewayError("Replica stopped before completion"))
            self.engine.close()
            self._stopping = True

    def _coalesce_admissions(self) -> int:
        if self.admission_max_wait_ns == 0:
            return 0
        started_ns = time.perf_counter_ns()
        deadline_ns = started_ns + self.admission_max_wait_ns
        quiet_deadline_ns = started_ns + self.admission_quiet_ns
        admitted = 0
        target_requests = max(1, self.frontend_inflight)
        while (
            not self._stopping
            and self.engine.scheduler.active_request_count
            < self.config.max_num_sequences
        ):
            now_ns = time.perf_counter_ns()
            target_requests = min(
                self.config.max_num_sequences,
                max(target_requests, self.frontend_inflight),
            )
            active_requests = self.engine.scheduler.active_request_count
            if active_requests >= target_requests and target_requests > 1:
                break
            if active_requests < target_requests:
                wait_until_ns = min(
                    deadline_ns,
                    now_ns + max(self.admission_quiet_ns, 1_000),
                )
            else:
                wait_until_ns = min(deadline_ns, quiet_deadline_ns)
            if now_ns >= wait_until_ns:
                break
            try:
                command = self._commands.get(
                    timeout=(wait_until_ns - now_ns) / 1_000_000_000
                )
            except queue.Empty:
                if (
                    self.engine.scheduler.active_request_count < target_requests
                    and time.perf_counter_ns() < deadline_ns
                ):
                    continue
                break
            accepted = self._handle(command)
            admitted += accepted
            if accepted:
                quiet_deadline_ns = min(
                    deadline_ns,
                    time.perf_counter_ns() + self.admission_quiet_ns,
                )
        self.metrics.admission_wait_ns += time.perf_counter_ns() - started_ns
        return admitted

    def _drain_commands(self) -> int:
        admitted = 0
        while not self._stopping:
            try:
                command = self._commands.get_nowait()
            except queue.Empty:
                return admitted
            admitted += self._handle(command)
        return admitted

    def _handle(self, command: _Command) -> int:
        if isinstance(command, _Stop):
            self._stopping = True
            return 0
        engine = self.engine
        if isinstance(command, _Inspect):
            command.result.set_result(self._runtime_snapshot())
            return 0
        if isinstance(command, _SubmitBatch):
            return sum(
                self._handle(submission)
                for submission in command.submissions
            )
        if isinstance(command, _SubmitMultimodalBatch):
            return self._handle_multimodal_batch(command.submissions)
        if isinstance(command, _SubmitMultimodal):
            return self._handle_multimodal_batch((command,))
        if isinstance(command, _Submit):
            if self._stopping:
                command.accepted.set_exception(GatewayError("Replica is stopping"))
                return 0
            self._subscribers[command.request_id] = _Subscriber(
                request_id=command.request_id,
                loop=command.loop,
                events=command.events,
            )
            try:
                engine.add_request(
                    command.token_ids,
                    max_new_tokens=command.max_new_tokens,
                    sampling=command.sampling,
                    request_id=command.request_id,
                    output_modalities=command.output_modalities,
                    audio_output=command.audio_output,
                )
            except BaseException as exc:
                self._subscribers.pop(command.request_id, None)
                command.accepted.set_exception(exc)
                return 0
            else:
                command.accepted.set_result(command.request_id)
                self.metrics.admitted_requests += 1
                return 1
        if isinstance(command, _Cancel):
            if command.request_id not in self._subscribers:
                command.ack.set_result(False)
                return 0
            try:
                engine.cancel(command.request_id)
            except BaseException:
                command.ack.set_result(False)
                return 0
            command.ack.set_result(True)
            self._deferred_events.extend(engine.scheduler.drain_events())
        return 0

    def _handle_multimodal_batch(
        self,
        commands: tuple[_SubmitMultimodal, ...],
    ) -> int:
        if self._stopping:
            error = GatewayError("Replica is stopping")
            for command in commands:
                command.accepted.set_exception(error)
            return 0
        engine = self.engine
        admissions: list[MultimodalAdmission] = []
        accepted_commands: list[_SubmitMultimodal] = []
        for command in commands:
            if command.request_id in self._subscribers:
                command.accepted.set_exception(
                    GatewayError(
                        f"duplicate request_id {command.request_id!r}"
                    )
                )
                continue
            self._subscribers[command.request_id] = _Subscriber(
                request_id=command.request_id,
                loop=command.loop,
                events=command.events,
            )
            accepted_commands.append(command)
            admissions.append(
                MultimodalAdmission(
                    messages=command.messages,
                    images=command.images,
                    videos=command.videos,
                    audios=command.audios,
                    max_new_tokens=command.max_new_tokens,
                    sampling=command.sampling,
                    request_id=command.request_id,
                    open_thinking=command.open_thinking,
                    output_modalities=command.output_modalities,
                    audio_output=command.audio_output,
                    tools=command.tools,
                )
            )
        if not admissions:
            return 0
        try:
            results = engine.add_multimodal_batch(tuple(admissions))
        except BaseException as exc:
            results = (exc,) * len(admissions)
        admitted = 0
        for command, result in zip(
            accepted_commands,
            results,
            strict=True,
        ):
            if isinstance(result, BaseException):
                self._subscribers.pop(command.request_id, None)
                command.accepted.set_exception(result)
                continue
            prompt_tokens = engine.scheduler.prompt_length(result)
            command.accepted.set_result((result, prompt_tokens))
            self.metrics.admitted_requests += 1
            admitted += 1
        return admitted

    def _relay_deferred_events(self, *, force: bool = False) -> None:
        if not self._deferred_events:
            return
        contains_terminal = any(
            event.kind is EventKind.TERMINAL
            for event in self._deferred_events
        )
        contains_first_token = any(
            event.kind is EventKind.TOKEN
            and event.request_id not in self._started_streams
            for event in self._deferred_events
        )
        contains_audio = any(
            event.kind is EventKind.AUDIO
            for event in self._deferred_events
        )
        enough_pressure = (
            self.engine.scheduler.active_request_count
            >= self.stream_flush_min_batch
        )
        if (
            not force
            and not contains_terminal
            and not contains_audio
            and not contains_first_token
            and enough_pressure
            and self._deferred_event_steps < self.stream_flush_steps
        ):
            return
        events = tuple(self._deferred_events)
        self._deferred_events.clear()
        self.metrics.relay_step_histogram[
            max(1, self._deferred_event_steps)
        ] += 1
        self._deferred_event_steps = 0
        self._dispatch(events)

    def _runtime_snapshot(self) -> dict[str, Any]:
        engine = self.engine
        graph = engine.graph_runner
        vision_stats = engine.vision_cache.stats()
        media_cache = {
            **asdict(vision_stats),
            "hit_rate": vision_stats.hit_rate,
        }
        return {
            "engine": engine.metrics.snapshot(),
            "replica": self.metrics.snapshot(),
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
        }

    def _dispatch(self, events: tuple[EngineEvent, ...]) -> None:
        engine = self.engine
        terminal_ids: list[str] = []
        grouped_deliveries: dict[
            asyncio.AbstractEventLoop,
            dict[str, tuple[asyncio.Queue[ReplicaOutput], list[EngineEvent]]],
        ] = {}
        for event in events:
            if event.kind not in {
                EventKind.TOKEN,
                EventKind.AUDIO,
                EventKind.TERMINAL,
            }:
                continue
            subscriber = self._subscribers.get(event.request_id)
            if subscriber is None:
                continue
            loop_deliveries = grouped_deliveries.setdefault(
                subscriber.loop,
                {},
            )
            _, request_events = loop_deliveries.setdefault(
                event.request_id,
                (subscriber.events, []),
            )
            request_events.append(event)
            if event.kind is EventKind.TOKEN:
                self._started_streams.add(event.request_id)
            if event.kind is EventKind.TERMINAL:
                terminal_ids.append(event.request_id)
        for loop, request_deliveries in grouped_deliveries.items():
            loop_deliveries: list[
                tuple[str, asyncio.Queue[ReplicaOutput], ReplicaOutput]
            ] = []
            delivered_events = 0
            for request_id, (output, request_events) in request_deliveries.items():
                delivered_events += len(request_events)
                token_ids = tuple(
                    token_id
                    for event in request_events
                    if event.kind is EventKind.TOKEN
                    for token_id in event.token_ids
                )
                audio = tuple(
                    event.audio
                    for event in request_events
                    if event.kind is EventKind.AUDIO
                    and event.audio is not None
                )
                logprobs = tuple(
                    value
                    for event in request_events
                    if event.kind is EventKind.TOKEN
                    for value in event.logprobs
                )
                terminal = next(
                    (
                        event
                        for event in request_events
                        if event.kind is EventKind.TERMINAL
                    ),
                    None,
                )
                delivery = StreamBatch(
                    token_ids=token_ids,
                    audio=audio,
                    finish_reason=(
                        None if terminal is None else terminal.finish_reason
                    ),
                    error=None if terminal is None else terminal.error,
                    logprobs=logprobs,
                )
                loop_deliveries.append((request_id, output, delivery))
            self.metrics.event_batches += 1
            self.metrics.delivered_events += delivered_events
            self.metrics.delivered_output_batches += len(loop_deliveries)
            loop.call_soon_threadsafe(
                self._deliver_batch,
                tuple(loop_deliveries),
            )
        for request_id in terminal_ids:
            self._subscribers.pop(request_id, None)
            self._started_streams.discard(request_id)
            engine.scheduler.forget(request_id)

    def _deliver_batch(
        self,
        deliveries: tuple[
            tuple[str, asyncio.Queue[ReplicaOutput], ReplicaOutput],
            ...,
        ],
    ) -> None:
        for request_id, output, event in deliveries:
            self._deliver(request_id, output, event)

    def _deliver(
        self,
        request_id: str,
        output: asyncio.Queue[ReplicaOutput],
        event: ReplicaOutput,
    ) -> None:
        try:
            output.put_nowait(event)
        except asyncio.QueueFull:
            while True:
                try:
                    output.get_nowait()
                except asyncio.QueueEmpty:
                    break
            output.put_nowait(
                ReplicaFailure(
                    GatewayError(
                        "client output buffer exceeded its production limit"
                    )
                )
            )
            self.cancel(request_id)

    def _fail_all(self, error: BaseException) -> None:
        subscribers = tuple(self._subscribers.items())
        self._subscribers.clear()
        for request_id, subscriber in subscribers:
            subscriber.loop.call_soon_threadsafe(
                self._deliver,
                request_id,
                subscriber.events,
                ReplicaFailure(error),
            )


@dataclass(slots=True)
class ProcessReplicaMetrics:
    # Proxy wire-level counters (topology-specific).
    sent_frames: int = 0
    sent_bytes: int = 0
    received_frames: int = 0
    received_events: int = 0
    delivered_output_batches: int = 0
    admitted_requests: int = 0
    peak_frontend_inflight: int = 0
    # Request-facing SLI state (same names as ReplicaMetrics so /metrics is
    # topology-independent: megalm_replica_* exists in both modes).
    ttft_ms_total: float = 0.0
    ttft_samples: int = 0
    tpot_ms_total: float = 0.0
    tpot_samples: int = 0
    ttft_ms_histogram: Counter[int] = field(default_factory=Counter)
    tpot_ms_histogram: Counter[int] = field(default_factory=Counter)
    e2e_ms_histogram: Counter[int] = field(default_factory=Counter)
    finish_reasons: Counter[str] = field(default_factory=Counter)
    parser_pushes: int = 0
    parser_finishes: int = 0
    parser_calls: int = 0
    parser_diagnostics: int = 0
    parser_bytes: int = 0

    def proxy_snapshot(self) -> dict[str, object]:
        """Wire-level counters, exposed under the `proxy` prefix."""

        return {
            "sent_frames": self.sent_frames,
            "sent_bytes": self.sent_bytes,
            "received_frames": self.received_frames,
            "received_events": self.received_events,
            "delivered_output_batches": self.delivered_output_batches,
            "admitted_requests": self.admitted_requests,
            "peak_frontend_inflight": self.peak_frontend_inflight,
        }

    def snapshot(self) -> dict[str, object]:
        """Request-facing SLI snapshot, identical across replica topologies."""

        return {
            "admitted_requests": self.admitted_requests,
            "ttft_ms_mean": (
                self.ttft_ms_total / self.ttft_samples
                if self.ttft_samples
                else 0.0
            ),
            "ttft_samples": self.ttft_samples,
            "tpot_ms_mean": (
                self.tpot_ms_total / self.tpot_samples
                if self.tpot_samples
                else 0.0
            ),
            "tpot_samples": self.tpot_samples,
            "ttft_ms_histogram": histogram_totals(
                dict(sorted(self.ttft_ms_histogram.items())),
                total_sum=self.ttft_ms_total,
                total_count=self.ttft_samples,
            ),
            "tpot_ms_histogram": histogram_totals(
                dict(sorted(self.tpot_ms_histogram.items())),
                total_sum=self.tpot_ms_total,
                total_count=self.tpot_samples,
            ),
            "e2e_ms_histogram": dict(sorted(self.e2e_ms_histogram.items())),
            "finish_reasons": dict(sorted(self.finish_reasons.items())),
            "parser": {
                "pushes": self.parser_pushes,
                "finishes": self.parser_finishes,
                "calls": self.parser_calls,
                "diagnostics": self.parser_diagnostics,
                "bytes": self.parser_bytes,
            },
        }


class ProcessReplica:
    """Stateless-Gateway proxy to one GPU-owning Unix-socket worker."""

    replica_mode = "process"

    def __init__(
        self,
        config: EngineConfig,
        *,
        max_buffered_events: int = 512,
        admission_max_wait_us: int = 0,
        admission_quiet_us: int = 0,
        stream_flush_steps: int = 1,
        stream_flush_min_batch: int = 8,
    ) -> None:
        if min(
            max_buffered_events,
            stream_flush_steps,
            stream_flush_min_batch,
        ) <= 0:
            raise GatewayError("Replica buffer and relay limits must be positive")
        if stream_flush_steps != 1:
            raise GatewayError(
                "isolated Replica currently requires stream_flush_steps=1"
            )
        if min(admission_max_wait_us, admission_quiet_us) < 0:
            raise GatewayError("admission coalescing windows cannot be negative")
        if admission_quiet_us > admission_max_wait_us:
            raise GatewayError(
                "admission quiet window cannot exceed the maximum window"
            )
        self.config = config
        self.max_buffered_events = max_buffered_events
        self.admission_max_wait_ns = admission_max_wait_us * 1_000
        self.admission_quiet_ns = admission_quiet_us * 1_000
        self.stream_flush_steps = stream_flush_steps
        self.stream_flush_min_batch = stream_flush_min_batch
        self.metrics = ProcessReplicaMetrics()
        self._channel: socket.socket | None = None
        self._process: subprocess.Popen[bytes] | None = None
        self._reader: threading.Thread | None = None
        self._ready = threading.Event()
        self._stopping = False
        self._failure: BaseException | None = None
        self._descriptor: dict[str, Any] | None = None
        self._ipc_lock = threading.Lock()
        self._subscriber_lock = threading.Lock()
        self._frontend_lock = threading.Lock()
        self._frontend_inflight = 0
        self._next_command_id = 1
        self._next_stream_id = 1
        self._pending: dict[int, Future[dict[str, Any]]] = {}
        self._subscribers: dict[str, _Subscriber] = {}
        self._streams: dict[int, _Subscriber] = {}

    def start(self, *, timeout: float = 600.0) -> None:
        if self._process is not None:
            raise GatewayError("isolated Replica is already started")
        parent_channel, child_channel = socket.socketpair(
            socket.AF_UNIX,
            socket.SOCK_STREAM,
        )
        config_payload = asdict(self.config)
        config_payload["model_path"] = str(self.config.model_path)
        encoded_config = base64.urlsafe_b64encode(
            orjson.dumps(config_payload)
        ).decode()
        command = (
            sys.executable,
            "-m",
            "megalm.gateway.worker",
            "--fd",
            str(child_channel.fileno()),
            "--config",
            encoded_config,
            "--admission-max-wait-us",
            str(self.admission_max_wait_ns // 1_000),
            "--admission-quiet-us",
            str(self.admission_quiet_ns // 1_000),
            "--log-level",
            logging.getLevelName(
                logging.getLogger("megalm").getEffectiveLevel()
            ),
        )
        try:
            self._process = subprocess.Popen(
                command,
                pass_fds=(child_channel.fileno(),),
                close_fds=True,
                cwd=os.getcwd(),
                env=os.environ.copy(),
                stdin=subprocess.DEVNULL,
                start_new_session=True,
            )
        except BaseException:
            parent_channel.close()
            child_channel.close()
            raise
        child_channel.close()
        self._channel = parent_channel
        self._reader = threading.Thread(
            target=self._read_frames,
            name="megalm-gateway-ipc-reader",
            daemon=True,
        )
        self._reader.start()
        if not self._ready.wait(timeout):
            self.stop()
            raise GatewayError("isolated Replica initialization timed out")
        if self._failure is not None:
            failure = self._failure
            self.stop()
            raise GatewayError("isolated Replica initialization failed") from failure

    def stop(self, *, timeout: float = 30.0) -> None:
        process = self._process
        if process is None:
            return
        self._stopping = True
        if process.poll() is None:
            try:
                self._send(FrameType.STOP, {})
            except BaseException:
                pass
            try:
                process.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                process.terminate()
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=5)
        channel = self._channel
        if channel is not None:
            try:
                channel.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass
            channel.close()
            self._channel = None
        reader = self._reader
        if reader is not None and reader is not threading.current_thread():
            reader.join(timeout=5)
        self._reader = None
        self._process = None
        self._fail_pending(GatewayError("isolated Replica stopped"))
        self._fail_all(GatewayError("Replica stopped before completion"))

    async def submit(
        self,
        token_ids: tuple[int, ...],
        *,
        max_new_tokens: int,
        sampling: SamplingParams,
        request_id: str | None = None,
        output_modalities: tuple[OutputModality, ...] = (
            OutputModality.TEXT,
        ),
        audio_output: AudioOutputOptions | None = None,
    ) -> RequestStream:
        result = (
            await self.submit_batch(
                (
                    _ReplicaInput(
                        token_ids=token_ids,
                        max_new_tokens=max_new_tokens,
                        sampling=sampling,
                        request_id=request_id or uuid4().hex,
                        output_modalities=output_modalities,
                        audio_output=audio_output,
                    ),
                )
            )
        )[0]
        if isinstance(result, BaseException):
            raise result
        return result

    async def submit_batch(
        self,
        submissions: tuple[_ReplicaInput, ...],
    ) -> tuple[RequestStream | BaseException, ...]:
        if not submissions:
            return ()
        if self._failure is not None:
            raise GatewayError("Replica is unavailable") from self._failure
        loop = asyncio.get_running_loop()
        streams: dict[str, RequestStream] = {}
        stream_ids: dict[str, int] = {}
        with self._subscriber_lock:
            for submission in submissions:
                if submission.request_id in self._subscribers:
                    raise GatewayError(
                        f"duplicate request_id {submission.request_id!r}"
                    )
                stream_id = self._next_stream_id
                self._next_stream_id += 1
                events: asyncio.Queue[ReplicaOutput] = asyncio.Queue(
                    maxsize=self.max_buffered_events
                )
                subscriber = _Subscriber(
                    request_id=submission.request_id,
                    loop=loop,
                    events=events,
                    stream_id=stream_id,
                )
                self._subscribers[submission.request_id] = subscriber
                self._streams[stream_id] = subscriber
                stream_ids[submission.request_id] = stream_id
                streams[submission.request_id] = RequestStream(
                    request_id=submission.request_id,
                    prompt_tokens=len(submission.token_ids),
                    events=events,
                )
        with self._subscriber_lock:
            subscriber_count = len(self._subscribers)
        target_active_requests = min(
            self.config.max_num_sequences,
            subscriber_count
            + self.frontend_inflight
            - len(submissions),
        )
        payload = {
            "target_active_requests": max(
                len(submissions),
                target_active_requests,
            ),
            "submissions": [
                {
                    "request_id": submission.request_id,
                    "stream_id": stream_ids[submission.request_id],
                    "token_ids": submission.token_ids,
                    "max_new_tokens": submission.max_new_tokens,
                    "sampling": sampling_to_wire(submission.sampling),
                    "output_modalities": [
                        item.value
                        for item in submission.output_modalities
                    ],
                    "audio_output": (
                        None
                        if submission.audio_output is None
                        else {
                            "voice": submission.audio_output.voice,
                            "format": submission.audio_output.format,
                        }
                    ),
                }
                for submission in submissions
            ]
        }
        try:
            response = await asyncio.wrap_future(
                self._request(FrameType.SUBMIT_BATCH, payload)
            )
        except BaseException:
            with self._subscriber_lock:
                for submission in submissions:
                    self._remove_subscriber_locked(submission.request_id)
            raise
        raw_results = response.get("results")
        if not isinstance(raw_results, list):
            raise GatewayError("Replica returned an invalid submit result")
        by_id = {
            str(result["request_id"]): result
            for result in raw_results
        }
        outputs: list[RequestStream | BaseException] = []
        for submission in submissions:
            result = by_id.get(submission.request_id)
            if result is None or not bool(result.get("ok")):
                with self._subscriber_lock:
                    self._remove_subscriber_locked(submission.request_id)
                raw_error = {} if result is None else result.get("error", {})
                outputs.append(
                    GatewayError(
                        str(raw_error.get("message", "Replica rejected request"))
                    )
                )
            else:
                self.metrics.admitted_requests += 1
                outputs.append(streams[submission.request_id])
        return tuple(outputs)

    async def submit_multimodal(
        self,
        messages: Sequence[Mapping[str, Any]],
        images: Sequence[ImageInput],
        *,
        videos: Sequence[VideoInput] = (),
        audios: Sequence[AudioInput] = (),
        max_new_tokens: int,
        sampling: SamplingParams,
        request_id: str | None = None,
        open_thinking: bool = False,
        output_modalities: tuple[OutputModality, ...] = (
            OutputModality.TEXT,
        ),
        audio_output: AudioOutputOptions | None = None,
        tools: tuple[dict[str, Any], ...] = (),
    ) -> RequestStream:
        output = (
            await self.submit_multimodal_batch(
                (
                    MultimodalReplicaInput(
                        messages=tuple(messages),
                        images=tuple(images),
                        videos=tuple(videos),
                        audios=tuple(audios),
                        max_new_tokens=max_new_tokens,
                        sampling=sampling,
                        request_id=request_id or uuid4().hex,
                        open_thinking=open_thinking,
                        output_modalities=output_modalities,
                        audio_output=audio_output,
                        tools=tools,
                    ),
                )
            )
        )[0]
        if isinstance(output, BaseException):
            raise output
        return output

    async def submit_multimodal_batch(
        self,
        submissions: tuple[MultimodalReplicaInput, ...],
    ) -> tuple[RequestStream | BaseException, ...]:
        if not submissions:
            return ()
        if self._failure is not None:
            raise GatewayError("Replica is unavailable") from self._failure
        groups: list[tuple[MultimodalReplicaInput, ...]] = []
        current: list[MultimodalReplicaInput] = []
        media_count = 0
        for submission in submissions:
            count = submission.media_count
            if count <= 0 or count > MAX_MEDIA_FDS:
                raise GatewayError(
                    "multimodal request exceeds the media transport limit"
                )
            if current and media_count + count > MAX_MEDIA_FDS:
                groups.append(tuple(current))
                current = []
                media_count = 0
            current.append(submission)
            media_count += count
        if current:
            groups.append(tuple(current))
        grouped = await asyncio.gather(
            *(
                self._submit_multimodal_chunk(group)
                for group in groups
            )
        )
        return tuple(
            output
            for outputs in grouped
            for output in outputs
        )

    async def _submit_multimodal_chunk(
        self,
        submissions: tuple[MultimodalReplicaInput, ...],
    ) -> tuple[RequestStream | BaseException, ...]:
        loop = asyncio.get_running_loop()
        request_ids = tuple(
            submission.request_id for submission in submissions
        )
        if len(set(request_ids)) != len(request_ids):
            raise GatewayError(
                "multimodal batch contains duplicate request ids"
            )
        streams: dict[str, RequestStream] = {}
        stream_ids: dict[str, int] = {}
        with self._subscriber_lock:
            if any(
                request_id in self._subscribers
                for request_id in request_ids
            ):
                raise GatewayError(
                    "multimodal batch contains an active request id"
                )
            for submission in submissions:
                stream_id = self._next_stream_id
                self._next_stream_id += 1
                events: asyncio.Queue[ReplicaOutput] = asyncio.Queue(
                    maxsize=self.max_buffered_events
                )
                subscriber = _Subscriber(
                    request_id=submission.request_id,
                    loop=loop,
                    events=events,
                    stream_id=stream_id,
                )
                self._subscribers[submission.request_id] = subscriber
                self._streams[stream_id] = subscriber
                stream_ids[submission.request_id] = stream_id
                streams[submission.request_id] = RequestStream(
                    request_id=submission.request_id,
                    prompt_tokens=0,
                    events=events,
                )
        with self._subscriber_lock:
            subscriber_count = len(self._subscribers)
        target_active_requests = min(
            self.config.max_num_sequences,
            subscriber_count
            + self.frontend_inflight
            - len(submissions),
        )
        sealed = []
        try:
            descriptors: list[dict[str, Any]] = []
            fds: list[int] = []
            raw_submissions: list[dict[str, Any]] = []
            for submission in submissions:
                media = seal_media_inputs(
                    submission.images,
                    submission.videos,
                    submission.audios,
                )
                sealed.append(media)
                fd_start = len(fds)
                descriptors.extend(media.descriptors)
                fds.extend(media.fds)
                raw_submissions.append(
                    {
                        "request_id": submission.request_id,
                        "stream_id": stream_ids[submission.request_id],
                        "messages": submission.messages,
                        "media_start": fd_start,
                        "media_count": len(media.fds),
                        "max_new_tokens": submission.max_new_tokens,
                        "sampling": sampling_to_wire(
                            submission.sampling
                        ),
                        "open_thinking": submission.open_thinking,
                        "tools": submission.tools,
                        "output_modalities": [
                            item.value
                            for item in submission.output_modalities
                        ],
                        "audio_output": (
                            None
                            if submission.audio_output is None
                            else {
                                "voice": submission.audio_output.voice,
                                "format": submission.audio_output.format,
                            }
                        ),
                    }
                )
            response_future = self._request_with_fds(
                FrameType.SUBMIT_MULTIMODAL_BATCH,
                {
                    "target_active_requests": max(
                        len(submissions),
                        target_active_requests,
                    ),
                    "submissions": raw_submissions,
                    "media": descriptors,
                },
                tuple(fds),
            )
        except BaseException:
            with self._subscriber_lock:
                for request_id in request_ids:
                    self._remove_subscriber_locked(request_id)
            raise
        finally:
            for media in sealed:
                media.close()
        try:
            response = await asyncio.wrap_future(response_future)
        except BaseException:
            with self._subscriber_lock:
                for request_id in request_ids:
                    self._remove_subscriber_locked(request_id)
            raise
        raw_results = response.get("results")
        if not isinstance(raw_results, list):
            with self._subscriber_lock:
                for request_id in request_ids:
                    self._remove_subscriber_locked(request_id)
            raise GatewayError(
                "Replica returned an invalid multimodal batch result"
            )
        by_id = {
            str(result["request_id"]): result
            for result in raw_results
            if isinstance(result, dict) and "request_id" in result
        }
        outputs: list[RequestStream | BaseException] = []
        for submission in submissions:
            result = by_id.get(submission.request_id)
            prompt_tokens = (
                None if result is None else result.get("prompt_tokens")
            )
            if (
                result is None
                or not bool(result.get("ok"))
                or isinstance(prompt_tokens, bool)
                or not isinstance(prompt_tokens, int)
                or prompt_tokens <= 0
            ):
                with self._subscriber_lock:
                    self._remove_subscriber_locked(
                        submission.request_id
                    )
                raw_error = (
                    {}
                    if result is None
                    else result.get("error", {})
                )
                outputs.append(
                    GatewayError(
                        str(
                            raw_error.get(
                                "message",
                                "Replica rejected multimodal request",
                            )
                        )
                    )
                )
                continue
            stream = streams[submission.request_id]
            outputs.append(
                RequestStream(
                    request_id=stream.request_id,
                    prompt_tokens=prompt_tokens,
                    events=stream.events,
                )
            )
            self.metrics.admitted_requests += 1
        return tuple(outputs)

    def cancel(self, request_id: str) -> Future[bool]:
        ack: Future[bool] = Future()

        def resolve(done: Future[dict[str, Any]]) -> None:
            try:
                result = done.result()
                ack.set_result(bool(result.get("cancelled", False)))
            except BaseException:
                ack.set_result(False)

        try:
            self._request(FrameType.CANCEL, {"request_id": request_id}).add_done_callback(
                resolve
            )
        except BaseException:
            ack.set_result(False)
        return ack

    def frontend_request_started(self) -> None:
        with self._frontend_lock:
            self._frontend_inflight += 1
            self.metrics.peak_frontend_inflight = max(
                self.metrics.peak_frontend_inflight,
                self._frontend_inflight,
            )

    def frontend_request_finished(self) -> None:
        with self._frontend_lock:
            if self._frontend_inflight <= 0:
                raise GatewayError("frontend request accounting underflow")
            self._frontend_inflight -= 1

    @property
    def frontend_inflight(self) -> int:
        with self._frontend_lock:
            return self._frontend_inflight

    async def snapshot(self) -> dict[str, Any]:
        if self._failure is not None:
            raise GatewayError("Replica is unavailable") from self._failure
        snapshot = await asyncio.wrap_future(
            self._request(FrameType.INSPECT, {})
        )
        snapshot["proxy"] = self.metrics.proxy_snapshot()
        # Expose the same request-facing `megalm_replica_*` series as the
        # embedded topology so dashboards work in either deployment mode.
        snapshot["replica"] = self.metrics.snapshot()
        return snapshot

    async def describe(self) -> dict[str, Any]:
        descriptor = self._descriptor
        if descriptor is None:
            raise GatewayError("Replica descriptor is unavailable")
        return descriptor

    def _request(
        self,
        frame_type: FrameType,
        payload: dict[str, Any],
    ) -> Future[dict[str, Any]]:
        result: Future[dict[str, Any]] = Future()
        with self._ipc_lock:
            command_id = self._next_command_id
            self._next_command_id += 1
            payload = {**payload, "command_id": command_id}
            self._pending[command_id] = result
            try:
                self._send_locked(frame_type, payload)
            except BaseException as exc:
                self._pending.pop(command_id, None)
                result.set_exception(exc)
        return result

    def _request_with_fds(
        self,
        frame_type: FrameType,
        payload: dict[str, Any],
        fds: tuple[int, ...],
    ) -> Future[dict[str, Any]]:
        result: Future[dict[str, Any]] = Future()
        with self._ipc_lock:
            command_id = self._next_command_id
            self._next_command_id += 1
            payload = {**payload, "command_id": command_id}
            self._pending[command_id] = result
            try:
                channel = self._channel
                if channel is None:
                    raise GatewayError("Replica channel is closed")
                self.metrics.sent_bytes += send_frame_with_fds(
                    channel,
                    frame_type,
                    payload,
                    fds,
                )
                self.metrics.sent_frames += 1
            except BaseException as exc:
                self._pending.pop(command_id, None)
                result.set_exception(exc)
        return result

    def _send(self, frame_type: FrameType, payload: dict[str, Any]) -> None:
        with self._ipc_lock:
            self._send_locked(frame_type, payload)

    def _send_locked(
        self,
        frame_type: FrameType,
        payload: dict[str, Any],
    ) -> None:
        channel = self._channel
        if channel is None:
            raise GatewayError("Replica channel is closed")
        self.metrics.sent_bytes += send_frame(channel, frame_type, payload)
        self.metrics.sent_frames += 1

    def _read_frames(self) -> None:
        channel = self._channel
        assert channel is not None
        try:
            while True:
                frame = receive_frame(channel)
                if frame is None:
                    if not self._stopping:
                        raise GatewayError("Replica channel closed unexpectedly")
                    return
                frame_type, payload = frame
                self.metrics.received_frames += 1
                if frame_type is FrameType.READY:
                    descriptor = payload.get("descriptor")
                    if not isinstance(descriptor, dict):
                        raise GatewayError("Replica READY frame is invalid")
                    self._descriptor = descriptor
                    self._ready.set()
                elif frame_type is FrameType.OUTPUTS:
                    raw_outputs = payload.get("outputs")
                    if not isinstance(raw_outputs, list):
                        raise GatewayError("Replica OUTPUTS frame is invalid")
                    self._dispatch_outputs(raw_outputs)
                elif frame_type in {
                    FrameType.SUBMIT_RESULT,
                    FrameType.SNAPSHOT,
                }:
                    command_id = int(payload["command_id"])
                    with self._ipc_lock:
                        result = self._pending.pop(command_id, None)
                    if result is None:
                        raise GatewayError(
                            f"Replica response has unknown command_id {command_id}"
                        )
                    value = (
                        payload
                        if frame_type is FrameType.SUBMIT_RESULT
                        else payload["snapshot"]
                    )
                    result.set_result(value)
                elif frame_type is FrameType.FAILURE:
                    raw_error = payload.get("error", {})
                    raise GatewayError(
                        str(raw_error.get("message", "Replica worker failed"))
                    )
                else:
                    raise GatewayError(
                        f"unexpected Replica response {frame_type.name}"
                    )
        except BaseException as exc:
            if not self._stopping:
                self._record_failure(exc)
        finally:
            self._ready.set()

    def _dispatch_outputs(self, raw_outputs: list[Any]) -> None:
        grouped: dict[
            asyncio.AbstractEventLoop,
            list[tuple[str, asyncio.Queue[ReplicaOutput], ReplicaOutput]],
        ] = {}
        seen_streams: set[int] = set()
        with self._subscriber_lock:
            for raw_output in raw_outputs:
                (
                    stream_id,
                    token_ids,
                    audio,
                    finish_reason,
                    error,
                    logprobs,
                ) = output_from_wire(raw_output)
                if stream_id in seen_streams:
                    raise GatewayError(
                        "Replica OUTPUTS frame repeats a stream id"
                    )
                seen_streams.add(stream_id)
                self.metrics.received_events += (
                    int(bool(token_ids))
                    + len(audio)
                    + int(finish_reason is not None)
                )
                subscriber = self._streams.get(stream_id)
                if subscriber is None:
                    continue
                grouped.setdefault(subscriber.loop, []).append(
                    (
                        subscriber.request_id,
                        subscriber.events,
                        StreamBatch(
                            token_ids=token_ids,
                            audio=audio,
                            finish_reason=finish_reason,
                            error=error,
                            logprobs=logprobs,
                        ),
                    )
                )
                if finish_reason is not None:
                    self._remove_subscriber_locked(subscriber.request_id)
        for loop, deliveries in grouped.items():
            self.metrics.delivered_output_batches += len(deliveries)
            loop.call_soon_threadsafe(
                self._deliver_batch,
                tuple(deliveries),
            )

    def _remove_subscriber_locked(self, request_id: str) -> None:
        subscriber = self._subscribers.pop(request_id, None)
        if subscriber is not None and subscriber.stream_id is not None:
            self._streams.pop(subscriber.stream_id, None)

    def _deliver_batch(
        self,
        deliveries: tuple[
            tuple[str, asyncio.Queue[ReplicaOutput], ReplicaOutput],
            ...,
        ],
    ) -> None:
        for request_id, output, event in deliveries:
            self._deliver(request_id, output, event)

    def _deliver(
        self,
        request_id: str,
        output: asyncio.Queue[ReplicaOutput],
        event: ReplicaOutput,
    ) -> None:
        try:
            output.put_nowait(event)
        except asyncio.QueueFull:
            while True:
                try:
                    output.get_nowait()
                except asyncio.QueueEmpty:
                    break
            output.put_nowait(
                ReplicaFailure(
                    GatewayError(
                        "client output buffer exceeded its production limit"
                    )
                )
            )
            self.cancel(request_id)

    def _record_failure(self, error: BaseException) -> None:
        if self._failure is None:
            self._failure = error
        self._fail_pending(error)
        self._fail_all(error)

    def _fail_pending(self, error: BaseException) -> None:
        with self._ipc_lock:
            pending = tuple(self._pending.values())
            self._pending.clear()
        for result in pending:
            if not result.done():
                result.set_exception(error)

    def _fail_all(self, error: BaseException) -> None:
        with self._subscriber_lock:
            subscribers = tuple(self._subscribers.items())
            self._subscribers.clear()
            self._streams.clear()
        for request_id, subscriber in subscribers:
            subscriber.loop.call_soon_threadsafe(
                self._deliver,
                request_id,
                subscriber.events,
                ReplicaFailure(error),
            )


class AsyncAdmissionBatcher:
    """Event-loop microbatcher for one cross-thread Replica submission."""

    def __init__(
        self,
        replica: EmbeddedReplica | ProcessReplica,
        *,
        max_batch_size: int,
        max_pending: int = 4096,
    ) -> None:
        if min(max_batch_size, max_pending) <= 0:
            raise GatewayError("admission batch limits must be positive")
        self.replica = replica
        self.max_batch_size = max_batch_size
        self._jobs: asyncio.Queue[_AdmissionJob | None] = asyncio.Queue(
            maxsize=max_pending
        )
        self._task: asyncio.Task[None] | None = None
        self._failure: BaseException | None = None
        self.metrics = MicrobatchMetrics()

    def start(self) -> None:
        if self._task is not None:
            raise GatewayError("admission batcher is already running")
        self._task = asyncio.create_task(
            self._run(),
            name="megalm-admission-batcher",
        )

    async def stop(self) -> None:
        task = self._task
        if task is None:
            return
        await self._jobs.put(None)
        await task
        self._task = None

    async def submit(
        self,
        token_ids: tuple[int, ...],
        *,
        max_new_tokens: int,
        sampling: SamplingParams,
        request_id: str | None = None,
        output_modalities: tuple[OutputModality, ...] = (
            OutputModality.TEXT,
        ),
        audio_output: AudioOutputOptions | None = None,
    ) -> RequestStream:
        if self._failure is not None:
            raise GatewayError("admission batcher is unavailable") from self._failure
        loop = asyncio.get_running_loop()
        result: asyncio.Future[RequestStream] = loop.create_future()
        submission = _ReplicaInput(
            token_ids=token_ids,
            max_new_tokens=max_new_tokens,
            sampling=sampling,
            request_id=request_id or uuid4().hex,
            output_modalities=output_modalities,
            audio_output=audio_output,
        )
        try:
            self._jobs.put_nowait(
                _AdmissionJob(submission=submission, result=result)
            )
        except asyncio.QueueFull as exc:
            raise GatewayError("admission input queue is overloaded") from exc
        return await result

    async def _run(self) -> None:
        try:
            while True:
                first = await self._jobs.get()
                if first is None:
                    return
                jobs = [first]
                await asyncio.sleep(0)
                while len(jobs) < self.max_batch_size:
                    try:
                        job = self._jobs.get_nowait()
                    except asyncio.QueueEmpty:
                        break
                    if job is None:
                        self._jobs.put_nowait(None)
                        break
                    jobs.append(job)
                jobs = [job for job in jobs if not job.result.cancelled()]
                if not jobs:
                    continue
                self.metrics.record(len(jobs))
                try:
                    outputs = await self.replica.submit_batch(
                        tuple(job.submission for job in jobs)
                    )
                except BaseException as exc:
                    outputs = (exc,) * len(jobs)
                for job, output in zip(jobs, outputs, strict=True):
                    if job.result.cancelled():
                        if isinstance(output, RequestStream):
                            self.replica.cancel(output.request_id)
                    elif isinstance(output, BaseException):
                        job.result.set_exception(output)
                    else:
                        job.result.set_result(output)
        except BaseException as exc:
            self._failure = exc
            while True:
                try:
                    job = self._jobs.get_nowait()
                except asyncio.QueueEmpty:
                    break
                if job is not None and not job.result.done():
                    job.result.set_exception(exc)
            raise


def _json_bytes(payload: Any) -> bytes:
    return orjson.dumps(payload)


def _completion_chunk(
    *,
    request_id: str,
    model: str,
    created: int,
    text: str,
    finish_reason: str | None,
    usage: dict[str, int] | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "id": f"cmpl-{request_id}",
        "object": "text_completion",
        "created": created,
        "model": model,
        "choices": [
            {
                "index": 0,
                "text": text,
                "logprobs": None,
                "finish_reason": finish_reason,
            }
        ],
    }
    if usage is not None:
        payload["usage"] = usage
    return payload


def _openai_finish_reason(reason: FinishReason | None) -> str | None:
    if reason is None:
        return None
    return {
        FinishReason.STOP: "stop",
        FinishReason.LENGTH: "length",
        FinishReason.CANCELLED: "stop",
        FinishReason.REJECTED: "error",
        FinishReason.ERROR: "error",
    }[reason]


def create_app(
    *,
    replica: EmbeddedReplica | ProcessReplica,
    tokenizer: ModelTokenizer,
    served_model_name: str,
) -> FastAPI:
    tokenizer_batcher = AsyncTokenizerBatcher(
        tokenizer,
        max_batch_size=replica.config.max_num_sequences,
    )
    admission_batcher = AsyncAdmissionBatcher(
        replica,
        max_batch_size=replica.config.max_num_sequences,
    )
    omni_admission_batcher = AsyncOmniAdmissionBatcher(
        replica,
        max_batch_size=replica.config.max_num_sequences,
        coalesce_wait_us=getattr(
            replica.config,
            "omni_admission_coalesce_wait_us",
            5_000,
        ),
        max_light_batch_size=getattr(
            replica.config,
            "omni_light_admission_batch_size",
            2,
        ),
    )

    @asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncIterator[None]:
        tokenizer_batcher.start()
        admission_batcher.start()
        omni_admission_batcher.start()
        try:
            yield
        finally:
            await omni_admission_batcher.stop()
            await admission_batcher.stop()
            await tokenizer_batcher.stop()
            replica.stop()

    gateway_metrics = GatewayMetrics()
    app = FastAPI(
        title="MegaLM OpenAI Gateway",
        version="0.2.0",
        default_response_class=MegaLMJSONResponse,
        lifespan=lifespan,
    )
    # Added last => outermost: observes responses from the body-limit
    # middleware too (400/413) as well as every endpoint.
    app.add_middleware(RequestMetricsMiddleware, metrics=gateway_metrics)
    app.add_middleware(
        RequestBodyLimitMiddleware,
        max_bytes=MAX_CHAT_REQUEST_BYTES,
        paths=(
            "/v1/chat/completions",
            "/v1/systemone",
            "/v1/decisions",
        ),
    )

    @app.get("/health")
    async def health() -> dict[str, str]:
        if replica._failure is not None:
            raise HTTPException(status_code=503, detail="Replica is unavailable")
        return {"status": "ok"}

    @app.get("/metrics", include_in_schema=False)
    async def metrics() -> Response:
        try:
            snapshot = await replica.snapshot()
        except GatewayError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        # Frontend batcher metrics were previously only on /server_info.
        # Fold them in so batching saturation appears on Prometheus too.
        frontend = {
            "tokenizer_batcher": tokenizer_batcher.metrics.snapshot(),
            "admission_batcher": admission_batcher.metrics.snapshot(),
        }
        if omni_admission_batcher is not None:
            frontend["omni_admission_batcher"] = (
                omni_admission_batcher.metrics.snapshot()
            )
        snapshot["frontend"] = frontend
        snapshot["gateway"] = gateway_metrics.snapshot()
        mirror_snapshot(snapshot)
        payload = orjson.dumps(snapshot).decode("utf-8")
        return Response(
            content=prometheus_metrics(payload),
            media_type="text/plain; version=0.0.4; charset=utf-8",
        )

    @app.get("/v1/models")
    async def models() -> dict[str, Any]:
        return {
            "object": "list",
            "data": [
                {
                    "id": served_model_name,
                    "object": "model",
                    "created": 0,
                    "owned_by": "megalm",
                }
            ],
        }

    async def studio_model_state() -> dict[str, Any]:
        descriptor = await replica.describe()
        input_modalities = tuple(descriptor.get("input_modalities", ()))
        max_model_len = descriptor.get("max_model_len")
        speculative_type = descriptor.get("speculative_mode", "off")
        return {
            "descriptor": descriptor,
            "is_vision": any(
                modality in input_modalities
                for modality in ("image", "video")
            ),
            "has_audio_input": "audio" in input_modalities,
            "context_length": (
                max_model_len if isinstance(max_model_len, int) else None
            ),
            "speculative_type": (
                speculative_type
                if isinstance(speculative_type, str)
                else "off"
            ),
        }

    @app.get("/api/health", include_in_schema=False)
    async def studio_health() -> dict[str, Any]:
        if replica._failure is not None:
            raise HTTPException(status_code=503, detail="Replica is unavailable")
        return {
            "status": "ok",
            "backend": "megalm",
            "device_type": "linux",
            "chat_only": True,
        }

    @app.get("/api/models/list", include_in_schema=False)
    async def studio_models() -> dict[str, Any]:
        state = await studio_model_state()
        return {
            "models": [
                {
                    "id": served_model_name,
                    "name": served_model_name,
                    "is_vision": state["is_vision"],
                    "is_lora": False,
                    "is_gguf": False,
                    "is_audio": state["has_audio_input"],
                    "audio_type": (
                        "audio_vlm" if state["has_audio_input"] else None
                    ),
                    "has_audio_input": state["has_audio_input"],
                }
            ],
            "default_models": [served_model_name],
        }

    @app.get("/api/models/loras", include_in_schema=False)
    async def studio_loras() -> dict[str, Any]:
        return {"loras": [], "outputs_dir": ""}

    @app.get("/api/inference/status", include_in_schema=False)
    async def studio_inference_status() -> dict[str, Any]:
        state = await studio_model_state()
        descriptor = state["descriptor"]
        return {
            "active_model": served_model_name,
            "model_identifier": served_model_name,
            "is_vision": state["is_vision"],
            "is_gguf": False,
            "gguf_variant": None,
            "is_audio": state["has_audio_input"],
            "audio_type": (
                "audio_vlm" if state["has_audio_input"] else None
            ),
            "has_audio_input": state["has_audio_input"],
            "loading": [],
            "loaded": [served_model_name],
            "inference": {
                "temperature": 0.6,
                "top_p": 0.95,
                "top_k": 20,
                "min_p": 0.0,
                "presence_penalty": 0.0,
                "trust_remote_code": False,
            },
            "requires_trust_remote_code": False,
            "supports_reasoning": True,
            "reasoning_style": "enable_thinking",
            "reasoning_always_on": False,
            "supports_preserve_thinking": False,
            "supports_tools": False,
            "chat_template": None,
            "context_length": state["context_length"],
            "max_context_length": state["context_length"],
            "native_context_length": state["context_length"],
            "cache_type_kv": descriptor.get("dtype"),
            "chat_template_override": None,
            "speculative_type": state["speculative_type"],
            "spec_draft_n_max": descriptor.get("num_speculative_tokens"),
        }

    @app.post("/api/inference/validate", include_in_schema=False)
    async def studio_validate_model(payload: dict[str, Any]) -> dict[str, Any]:
        model_path = payload.get("model_path")
        valid = model_path == served_model_name
        return {
            "valid": valid,
            "message": (
                "MegaLM Replica is ready"
                if valid
                else "MegaLM serves one immutable model per Replica"
            ),
            "identifier": served_model_name if valid else None,
            "display_name": served_model_name if valid else None,
            "is_gguf": False,
            "is_lora": False,
            "requires_trust_remote_code": False,
        }

    @app.post("/api/inference/load", include_in_schema=False)
    async def studio_load_model(payload: dict[str, Any]) -> dict[str, Any]:
        if payload.get("model_path") != served_model_name:
            raise HTTPException(
                status_code=409,
                detail="MegaLM serves one immutable model per Replica",
            )
        status = await studio_inference_status()
        return {
            "status": "ready",
            "model": served_model_name,
            "display_name": served_model_name,
            **{
                key: status[key]
                for key in (
                    "is_vision",
                    "is_gguf",
                    "is_audio",
                    "audio_type",
                    "has_audio_input",
                    "inference",
                    "requires_trust_remote_code",
                    "supports_reasoning",
                    "reasoning_style",
                    "reasoning_always_on",
                    "supports_preserve_thinking",
                    "supports_tools",
                    "chat_template",
                    "context_length",
                    "max_context_length",
                    "native_context_length",
                    "cache_type_kv",
                    "speculative_type",
                    "spec_draft_n_max",
                )
            },
        }

    @app.post("/api/inference/cancel", include_in_schema=False)
    async def studio_cancel(payload: dict[str, Any]) -> dict[str, Any]:
        cancel_id = payload.get("cancel_id")
        if not isinstance(cancel_id, str) or not cancel_id:
            raise HTTPException(status_code=400, detail="cancel_id is required")
        cancelled = await asyncio.wrap_future(replica.cancel(cancel_id))
        if not cancelled:
            raise HTTPException(
                status_code=404,
                detail=f"active request {cancel_id!r} was not found",
            )
        return {"status": "cancelled", "cancel_id": cancel_id}

    @app.get("/server_info")
    async def server_info() -> dict[str, Any]:
        descriptor = await replica.describe()
        runtime = await replica.snapshot()
        scheduler = dict(descriptor["scheduler"])
        scheduler.update(
            {
                "admission_max_wait_us": (
                    replica.admission_max_wait_ns // 1_000
                ),
                "admission_quiet_us": replica.admission_quiet_ns // 1_000,
            }
        )
        return {
            **descriptor,
            "cache": runtime.get(
                "cache",
                descriptor.get("cache", {}),
            ),
            "media_cache": runtime.get(
                "media_cache",
                descriptor.get("media_cache", {}),
            ),
            "vision_cache": runtime.get(
                "vision_cache",
                descriptor.get("vision_cache", {}),
            ),
            "served_model_name": served_model_name,
            "capabilities": {
                "chat": True,
                "input_modalities": list(
                    descriptor.get("input_modalities", ())
                ),
                "output_modalities": list(
                    descriptor.get("output_modalities", ("text",))
                ),
                "audio": descriptor.get("audio_output"),
                "thinking": True,
                "reasoning": False,
            },
            "replica_transport": (
                "unix_socket_v1"
                if isinstance(replica, ProcessReplica)
                else "in_process"
            ),
            "frontend": {
                "tokenizer_batcher": tokenizer_batcher.metrics.snapshot(),
                "admission_batcher": admission_batcher.metrics.snapshot(),
                "omni_admission_batcher": (
                    omni_admission_batcher.metrics.snapshot()
                ),
                "media_fetch": {
                    "config": MEDIA_FETCH_CONFIG.snapshot(),
                    **MEDIA_FETCH_STATS.snapshot(),
                },
            },
            "scheduler": scheduler,
            "stream_relay": {
                "flush_steps": replica.stream_flush_steps,
                "min_batch_size": replica.stream_flush_min_batch,
                "first_token_immediate": True,
                "terminal_immediate": True,
            },
            "runtime": runtime,
        }

    @app.post("/v1/completions", response_model=None)
    async def completions(
        payload: CompletionRequest,
    ) -> MegaLMJSONResponse | StreamingResponse:
        if payload.model != served_model_name:
            raise HTTPException(status_code=404, detail="Unknown model")
        if payload.n != 1 or payload.best_of != 1:
            raise HTTPException(
                status_code=400,
                detail="MegaLM v1 supports n=1 and best_of=1",
            )
        replica.frontend_request_started()
        try:
            await asyncio.sleep(0)
            token_ids = (
                await tokenizer_batcher.encode(payload.prompt)
                if isinstance(payload.prompt, str)
                else tuple(payload.prompt)
            )
            sampling = SamplingParams(
                temperature=payload.temperature,
                top_p=payload.top_p,
                top_k=payload.top_k,
                min_p=payload.min_p,
                repetition_penalty=payload.repetition_penalty,
                seed=payload.seed,
                stop_token_ids=tuple(payload.stop_token_ids),
                ignore_eos=payload.ignore_eos,
            )
            stream = await admission_batcher.submit(
                token_ids,
                max_new_tokens=payload.max_tokens,
                sampling=sampling,
            )
        except SchedulerOverloaded as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        except (FrontendError, ValueError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except GatewayError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        finally:
            replica.frontend_request_finished()

        created = int(time.time())
        if payload.stream:
            return StreamingResponse(
                _stream_completion(
                    stream=stream,
                    replica=replica,
                    tokenizer=tokenizer,
                    model=served_model_name,
                    created=created,
                ),
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "X-Accel-Buffering": "no",
                },
            )
        response = await _collect_completion(
            stream=stream,
            replica=replica,
            tokenizer=tokenizer,
            model=served_model_name,
            created=created,
        )
        return MegaLMJSONResponse(response)

    from megalm.gateway.chat_api import install_chat_routes

    install_chat_routes(
        app,
        replica=replica,
        tokenizer=tokenizer,
        tokenizer_batcher=tokenizer_batcher,
        admission_batcher=admission_batcher,
        omni_admission_batcher=omni_admission_batcher,
        served_model_name=served_model_name,
    )

    from megalm.gateway.decision_api import install_decision_routes

    install_decision_routes(
        app,
        replica=replica,
        tokenizer=tokenizer,
        tokenizer_batcher=tokenizer_batcher,
        admission_batcher=admission_batcher,
        served_model_name=served_model_name,
    )
    return app


async def _stream_completion(
    *,
    stream: RequestStream,
    replica: EmbeddedReplica | ProcessReplica,
    tokenizer: ModelTokenizer,
    model: str,
    created: int,
) -> AsyncIterator[bytes]:
    decoder = IncrementalDecoder(tokenizer)
    terminal = False
    try:
        while True:
            item = await stream.events.get()
            if isinstance(item, ReplicaFailure):
                raise GatewayError("Replica stream failed") from item.error
            if item.token_ids:
                delta = decoder.push(item.token_ids)
                if delta:
                    chunk = _completion_chunk(
                        request_id=stream.request_id,
                        model=model,
                        created=created,
                        text=delta,
                        finish_reason=None,
                    )
                    yield b"data: " + _json_bytes(chunk) + b"\n\n"
            if item.finish_reason is not None:
                tail = decoder.finish()
                if tail:
                    chunk = _completion_chunk(
                        request_id=stream.request_id,
                        model=model,
                        created=created,
                        text=tail,
                        finish_reason=None,
                    )
                    yield b"data: " + _json_bytes(chunk) + b"\n\n"
                terminal = True
                usage = {
                    "prompt_tokens": stream.prompt_tokens,
                    "completion_tokens": decoder.token_count,
                    "total_tokens": stream.prompt_tokens + decoder.token_count,
                }
                chunk = _completion_chunk(
                    request_id=stream.request_id,
                    model=model,
                    created=created,
                    text="",
                    finish_reason=_openai_finish_reason(
                        item.finish_reason
                    ),
                    usage=usage,
                )
                yield b"data: " + _json_bytes(chunk) + b"\n\n"
                yield b"data: [DONE]\n\n"
                return
    finally:
        if not terminal:
            replica.cancel(stream.request_id)


async def _collect_completion(
    *,
    stream: RequestStream,
    replica: EmbeddedReplica | ProcessReplica,
    tokenizer: ModelTokenizer,
    model: str,
    created: int,
) -> dict[str, Any]:
    token_ids: list[int] = []
    terminal = False
    try:
        while True:
            item = await stream.events.get()
            if isinstance(item, ReplicaFailure):
                raise GatewayError("Replica request failed") from item.error
            if item.token_ids:
                token_ids.extend(item.token_ids)
            if item.finish_reason is not None:
                terminal = True
                text = tokenizer.decode(
                    token_ids,
                    skip_special_tokens=False,
                )
                return _completion_chunk(
                    request_id=stream.request_id,
                    model=model,
                    created=created,
                    text=text,
                    finish_reason=_openai_finish_reason(
                        item.finish_reason
                    ),
                    usage={
                        "prompt_tokens": stream.prompt_tokens,
                        "completion_tokens": len(token_ids),
                        "total_tokens": stream.prompt_tokens + len(token_ids),
                    },
                )
    finally:
        if not terminal:
            replica.cancel(stream.request_id)


def _parse_graph_sizes(value: str) -> tuple[int, ...]:
    try:
        sizes = tuple(int(item) for item in value.split(",") if item)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "CUDA Graph buckets must be comma-separated integers"
        ) from exc
    if not sizes or any(size <= 0 for size in sizes):
        raise argparse.ArgumentTypeError(
            "CUDA Graph buckets must contain positive integers"
        )
    return sizes


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the stateless MegaLM OpenAI Gateway over a TP=1 Replica."
    )
    parser.add_argument("model", type=Path)
    parser.add_argument("--served-model-name", default="megalm")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--dtype", choices=("float16", "bfloat16"), default="float16")
    parser.add_argument("--max-model-len", type=int, default=32_768)
    parser.add_argument(
        "--max-cache-tokens",
        type=int,
        default=0,
        help="page-aligned KV token cap; 0 profiles capacity from free GPU memory",
    )
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    parser.add_argument("--min-runtime-memory-mib", type=int, default=512)
    parser.add_argument("--max-num-seqs", type=int, default=256)
    parser.add_argument("--max-num-batched-tokens", type=int, default=16384)
    parser.add_argument("--prefill-chunk-size", type=int, default=2048)
    parser.add_argument(
        "--mixed-prefill-chunk-size",
        type=int,
        default=2048,
        help="maximum prefill tokens admitted beside active decode",
    )
    parser.add_argument(
        "--cpu-intraop-threads",
        type=int,
        default=16,
        help="fixed TensorPlay CPU worker count for native media preprocessing",
    )
    parser.add_argument(
        "--vision-max-encode-patches",
        type=int,
        default=8192,
        help="maximum packed patches in one shared image/video tower launch",
    )
    parser.add_argument(
        "--omni-admission-coalesce-wait-us",
        type=int,
        default=5000,
        help="bounded wait for coalescing sibling multimodal requests",
    )
    parser.add_argument(
        "--omni-light-admission-batch-size",
        type=int,
        default=2,
        help="fairness cap for joint image/audio tower admission",
    )
    parser.add_argument(
        "--video-admission-max-defer-ms",
        type=int,
        default=300,
        help="bounded video admission deferral while lighter requests run",
    )
    parser.add_argument(
        "--video-admission-defer-min-target",
        type=int,
        default=8,
        help="minimum target concurrency that enables video deferral",
    )
    parser.add_argument("--page-size", type=int, default=16)
    parser.add_argument("--triton-moe-max-tokens", type=int, default=64)
    parser.add_argument("--tensorcore-moe-min-tokens", type=int, default=8)
    parser.add_argument("--grouped-moe-min-tokens", type=int, default=8)
    parser.add_argument(
        "--speculative-mode",
        choices=("off", "ngram", "mtp"),
        default="off",
    )
    parser.add_argument(
        "--num-speculative-tokens",
        type=int,
        default=7,
    )
    parser.add_argument(
        "--prompt-lookup-min-ngram",
        type=int,
        default=2,
    )
    parser.add_argument(
        "--prompt-lookup-max-ngram",
        type=int,
        default=4,
    )
    parser.add_argument(
        "--prompt-lookup-min-candidate-support",
        type=int,
        default=3,
    )
    parser.add_argument(
        "--prompt-lookup-min-candidate-agreement",
        type=float,
        default=0.0,
    )
    parser.add_argument(
        "--prompt-lookup-probe-interval",
        type=int,
        default=4,
    )
    parser.add_argument(
        "--cuda-graph-batches",
        type=_parse_graph_sizes,
        default=(1, 2, 4, 8, 16, 32, 64, 128, 256),
    )
    parser.add_argument("--no-prefix-cache", action="store_true")
    parser.add_argument("--kv-host-cache-bytes", type=int, default=0)
    parser.add_argument("--kv-disk-cache-dir", default=None)
    parser.add_argument("--kv-disk-cache-bytes", type=int, default=0)
    parser.add_argument("--kv-disk-read-threads", type=int, default=4)
    parser.add_argument("--kv-disk-write-threads", type=int, default=2)
    parser.add_argument("--no-cuda-graph", action="store_true")
    parser.add_argument("--admission-max-wait-us", type=int, default=0)
    parser.add_argument("--admission-quiet-us", type=int, default=0)
    parser.add_argument("--stream-flush-steps", type=int, default=1)
    parser.add_argument("--stream-flush-min-batch", type=int, default=8)
    parser.add_argument(
        "--replica-mode",
        choices=("process", "embedded"),
        default="process",
        help="HTTP defaults to an isolated GPU worker; SDK can stay embedded",
    )
    parser.add_argument("--access-log", action="store_true")
    parser.add_argument(
        "--no-omni-audio-output",
        action="store_true",
        help="do not load Qwen2.5-Omni Talker and Token2Wav stages",
    )
    parser.add_argument(
        "--omni-talker-device",
        default=None,
        help="optional Talker device, for example cuda:1",
    )
    parser.add_argument(
        "--omni-token2wav-device",
        default=None,
        help="optional Token2Wav device, for example cuda:2",
    )
    parser.add_argument(
        "--omni-output-max-pending",
        type=int,
        default=32,
        help="bounded Talker/Token2Wav request queue",
    )
    parser.add_argument(
        "--log-level",
        choices=("DEBUG", "INFO", "WARNING", "ERROR"),
        default="INFO",
    )
    parser.add_argument(
        "--otlp-metrics-endpoint",
        default=None,
        help="OTLP/HTTP metrics endpoint (default $OTEL_EXPORTER_OTLP_ENDPOINT)",
    )
    parser.add_argument(
        "--otlp-traces-endpoint",
        default=None,
        help="OTLP/HTTP traces endpoint (default $OTEL_EXPORTER_OTLP_ENDPOINT)",
    )
    parser.add_argument(
        "--service-name",
        default="megalm",
        help="OpenTelemetry service.name",
    )
    args = parser.parse_args()
    configure_logging(args.log_level)
    configure_telemetry(
        service_name=args.service_name,
        metrics_endpoint=args.otlp_metrics_endpoint,
        traces_endpoint=args.otlp_traces_endpoint,
    )

    config = EngineConfig(
        model_path=args.model,
        device="cuda",
        dtype=args.dtype,
        page_size=args.page_size,
        max_cache_tokens=(
            None if args.max_cache_tokens == 0 else args.max_cache_tokens
        ),
        gpu_memory_utilization=args.gpu_memory_utilization,
        min_runtime_memory_bytes=args.min_runtime_memory_mib * 1024**2,
        max_model_len=args.max_model_len,
        max_num_sequences=args.max_num_seqs,
        max_num_batched_tokens=args.max_num_batched_tokens,
        prefill_chunk_size=args.prefill_chunk_size,
        mixed_prefill_chunk_size=args.mixed_prefill_chunk_size,
        cpu_intraop_threads=args.cpu_intraop_threads,
        vision_max_encode_patches=args.vision_max_encode_patches,
        omni_admission_coalesce_wait_us=(
            args.omni_admission_coalesce_wait_us
        ),
        omni_light_admission_batch_size=(
            args.omni_light_admission_batch_size
        ),
        video_admission_max_defer_ms=(
            args.video_admission_max_defer_ms
        ),
        video_admission_defer_min_target=(
            args.video_admission_defer_min_target
        ),
        enable_prefix_cache=not args.no_prefix_cache,
        kv_host_cache_bytes=args.kv_host_cache_bytes,
        kv_disk_cache_dir=args.kv_disk_cache_dir,
        kv_disk_cache_bytes=args.kv_disk_cache_bytes,
        kv_disk_read_threads=args.kv_disk_read_threads,
        kv_disk_write_threads=args.kv_disk_write_threads,
        use_triton=True,
        triton_moe_max_tokens=args.triton_moe_max_tokens,
        tensorcore_moe_min_tokens=args.tensorcore_moe_min_tokens,
        grouped_moe_min_tokens=args.grouped_moe_min_tokens,
        speculative_mode=args.speculative_mode,
        num_speculative_tokens=args.num_speculative_tokens,
        prompt_lookup_min_ngram=args.prompt_lookup_min_ngram,
        prompt_lookup_max_ngram=args.prompt_lookup_max_ngram,
        prompt_lookup_min_candidate_support=(
            args.prompt_lookup_min_candidate_support
        ),
        prompt_lookup_min_candidate_agreement=(
            args.prompt_lookup_min_candidate_agreement
        ),
        prompt_lookup_probe_interval=args.prompt_lookup_probe_interval,
        enable_cuda_graph=not args.no_cuda_graph,
        cuda_graph_batch_sizes=args.cuda_graph_batches,
        strict_cuda_graph=True,
        enable_omni_audio_output=not args.no_omni_audio_output,
        omni_talker_device=args.omni_talker_device,
        omni_token2wav_device=args.omni_token2wav_device,
        omni_output_max_pending=args.omni_output_max_pending,
    )
    replica_type = (
        ProcessReplica
        if args.replica_mode == "process"
        else EmbeddedReplica
    )
    replica = replica_type(
        config,
        admission_max_wait_us=args.admission_max_wait_us,
        admission_quiet_us=args.admission_quiet_us,
        stream_flush_steps=args.stream_flush_steps,
        stream_flush_min_batch=args.stream_flush_min_batch,
    )
    startup_started_ns = time.perf_counter_ns()
    log_event(
        _LOGGER,
        logging.INFO,
        "gateway_starting",
        model_path=str(args.model),
        served_model_name=args.served_model_name,
        host=args.host,
        port=args.port,
        replica_mode=args.replica_mode,
        dtype=args.dtype,
        max_model_len=args.max_model_len,
        max_num_sequences=args.max_num_seqs,
        max_num_batched_tokens=args.max_num_batched_tokens,
        prefix_cache=not args.no_prefix_cache,
        cuda_graph=not args.no_cuda_graph,
        cuda_graph_buckets=args.cuda_graph_batches,
    )
    replica.start()
    try:
        tokenizer = ModelTokenizer(args.model)
    except BaseException:
        replica.stop()
        raise
    app = create_app(
        replica=replica,
        tokenizer=tokenizer,
        served_model_name=args.served_model_name,
    )
    log_event(
        _LOGGER,
        logging.INFO,
        "gateway_ready",
        startup_ms=(
            time.perf_counter_ns() - startup_started_ns
        ) / 1_000_000,
        served_model_name=args.served_model_name,
        replica_mode=args.replica_mode,
        api_base=f"http://{args.host}:{args.port}/v1",
    )

    import uvicorn

    uvicorn.run(
        app,
        host=args.host,
        port=args.port,
        access_log=args.access_log,
        loop="uvloop",
        http="httptools",
    )


if __name__ == "__main__":
    main()
