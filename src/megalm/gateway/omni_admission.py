from __future__ import annotations

import asyncio
from collections import Counter
from dataclasses import dataclass, field
from time import perf_counter_ns
from typing import Any, Mapping, Sequence
from uuid import uuid4

from megalm.contracts.request import (
    AudioInput,
    AudioOutputOptions,
    ImageInput,
    OutputModality,
    SamplingParams,
    VideoInput,
)
from megalm.gateway.media_ipc import MAX_MEDIA_FDS
from megalm.gateway.replica_types import GatewayError


@dataclass(frozen=True, slots=True)
class MultimodalReplicaInput:
    messages: tuple[Mapping[str, Any], ...]
    images: tuple[ImageInput, ...]
    videos: tuple[VideoInput, ...]
    audios: tuple[AudioInput, ...]
    max_new_tokens: int
    sampling: SamplingParams
    request_id: str
    open_thinking: bool = False
    output_modalities: tuple[OutputModality, ...] = (OutputModality.TEXT,)
    audio_output: AudioOutputOptions | None = None
    tools: tuple[dict[str, Any], ...] = ()

    @property
    def media_count(self) -> int:
        return len(self.images) + len(self.videos) + len(self.audios)


@dataclass(frozen=True, slots=True)
class _OmniAdmissionJob:
    submission: MultimodalReplicaInput
    result: asyncio.Future[Any]


@dataclass(slots=True)
class OmniAdmissionMetrics:
    batches: int = 0
    requests: int = 0
    media_items: int = 0
    coalesce_waits: int = 0
    coalesce_wait_ns: int = 0
    image_items: int = 0
    video_items: int = 0
    audio_items: int = 0
    batch_histogram: Counter[int] = field(default_factory=Counter)

    def record_coalesce_wait(self, elapsed_ns: int) -> None:
        self.coalesce_waits += 1
        self.coalesce_wait_ns += elapsed_ns

    def record(
        self,
        submissions: Sequence[MultimodalReplicaInput],
    ) -> None:
        batch_size = len(submissions)
        self.batches += 1
        self.requests += batch_size
        self.media_items += sum(item.media_count for item in submissions)
        self.image_items += sum(len(item.images) for item in submissions)
        self.video_items += sum(len(item.videos) for item in submissions)
        self.audio_items += sum(len(item.audios) for item in submissions)
        self.batch_histogram[batch_size] += 1

    def snapshot(self) -> dict[str, object]:
        return {
            "batches": self.batches,
            "requests": self.requests,
            "media_items": self.media_items,
            "image_items": self.image_items,
            "video_items": self.video_items,
            "audio_items": self.audio_items,
            "coalesce_waits": self.coalesce_waits,
            "coalesce_wait_ms": self.coalesce_wait_ns / 1_000_000,
            "batch_histogram": dict(sorted(self.batch_histogram.items())),
        }


class AsyncOmniAdmissionBatcher:
    """Coalesce sibling Omni requests within a bounded latency window."""

    def __init__(
        self,
        replica: Any,
        *,
        max_batch_size: int,
        max_pending: int = 1024,
        max_media_items: int = MAX_MEDIA_FDS,
        coalesce_wait_us: int = 5_000,
        max_light_batch_size: int = 2,
    ) -> None:
        if min(
            max_batch_size,
            max_pending,
            max_media_items,
            max_light_batch_size,
        ) <= 0:
            raise GatewayError("Omni admission limits must be positive")
        if coalesce_wait_us < 0:
            raise GatewayError(
                "Omni admission coalesce wait must be non-negative"
            )
        self.replica = replica
        self.max_batch_size = max_batch_size
        self.max_media_items = max_media_items
        self.max_light_batch_size = min(
            max_batch_size,
            max_light_batch_size,
        )
        self.coalesce_wait_s = coalesce_wait_us / 1_000_000
        self._jobs: asyncio.Queue[_OmniAdmissionJob | None] = (
            asyncio.Queue(maxsize=max_pending)
        )
        self._task: asyncio.Task[None] | None = None
        self._failure: BaseException | None = None
        self.metrics = OmniAdmissionMetrics()

    def start(self) -> None:
        if self._task is not None:
            raise GatewayError(
                "Omni admission batcher is already running"
            )
        self._task = asyncio.create_task(
            self._run(),
            name="megalm-omni-admission-batcher",
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
    ) -> Any:
        if self._failure is not None:
            raise GatewayError(
                "Omni admission batcher is unavailable"
            ) from self._failure
        loop = asyncio.get_running_loop()
        result: asyncio.Future[Any] = loop.create_future()
        submission = MultimodalReplicaInput(
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
        )
        try:
            self._jobs.put_nowait(
                _OmniAdmissionJob(
                    submission=submission,
                    result=result,
                )
            )
        except asyncio.QueueFull as exc:
            raise GatewayError(
                "Omni admission input queue is overloaded"
            ) from exc
        return await result

    async def _run(self) -> None:
        try:
            while True:
                first = await self._jobs.get()
                if first is None:
                    return
                jobs = [first]
                wait_started_ns = perf_counter_ns()
                if (
                    self.coalesce_wait_s > 0
                    and self.max_batch_size > 1
                ):
                    timed_out = False
                    try:
                        job = await asyncio.wait_for(
                            self._jobs.get(),
                            timeout=self.coalesce_wait_s,
                        )
                    except TimeoutError:
                        timed_out = True
                        job = None
                    if timed_out:
                        pass
                    elif job is None:
                        self._jobs.put_nowait(None)
                    else:
                        jobs.append(job)
                    self.metrics.record_coalesce_wait(
                        perf_counter_ns() - wait_started_ns
                    )
                else:
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
                groups = _partition_jobs(
                    jobs,
                    max_batch_size=self.max_batch_size,
                    max_light_batch_size=self.max_light_batch_size,
                    max_media_items=self.max_media_items,
                )
                await asyncio.gather(
                    *(self._dispatch(group) for group in groups)
                )
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

    async def _dispatch(
        self,
        jobs: tuple[_OmniAdmissionJob, ...],
    ) -> None:
        submissions = tuple(job.submission for job in jobs)
        self.metrics.record(submissions)
        try:
            submit_batch = getattr(
                self.replica,
                "submit_multimodal_batch",
                None,
            )
            if submit_batch is not None:
                outputs = await submit_batch(submissions)
            else:
                outputs = await asyncio.gather(
                    *(
                        self.replica.submit_multimodal(
                            submission.messages,
                            submission.images,
                            videos=submission.videos,
                            audios=submission.audios,
                            max_new_tokens=(
                                submission.max_new_tokens
                            ),
                            sampling=submission.sampling,
                            open_thinking=(
                                submission.open_thinking
                            ),
                            tools=submission.tools,
                        )
                        for submission in submissions
                    ),
                    return_exceptions=True,
                )
        except BaseException as exc:
            outputs = (exc,) * len(jobs)
        for job, output in zip(jobs, outputs, strict=True):
            if job.result.cancelled():
                if not isinstance(output, BaseException):
                    self.replica.cancel(output.request_id)
            elif isinstance(output, BaseException):
                job.result.set_exception(output)
            else:
                job.result.set_result(output)


def _partition_jobs(
    jobs: Sequence[_OmniAdmissionJob],
    *,
    max_batch_size: int,
    max_light_batch_size: int,
    max_media_items: int,
) -> tuple[tuple[_OmniAdmissionJob, ...], ...]:
    light = tuple(job for job in jobs if not job.submission.videos)
    video = tuple(job for job in jobs if job.submission.videos)
    return (
        *_partition_homogeneous_jobs(
            light,
            max_batch_size=max_light_batch_size,
            max_media_items=max_media_items,
        ),
        *_partition_homogeneous_jobs(
            video,
            max_batch_size=max_batch_size,
            max_media_items=max_media_items,
        ),
    )


def _partition_homogeneous_jobs(
    jobs: Sequence[_OmniAdmissionJob],
    *,
    max_batch_size: int,
    max_media_items: int,
) -> tuple[tuple[_OmniAdmissionJob, ...], ...]:
    groups: list[tuple[_OmniAdmissionJob, ...]] = []
    current: list[_OmniAdmissionJob] = []
    current_media = 0
    for job in jobs:
        media_count = job.submission.media_count
        if media_count <= 0 or media_count > max_media_items:
            raise GatewayError(
                "Omni request media count exceeds the batch transport limit"
            )
        if current and (
            len(current) >= max_batch_size
            or current_media + media_count > max_media_items
        ):
            groups.append(tuple(current))
            current = []
            current_media = 0
        current.append(job)
        current_media += media_count
    if current:
        groups.append(tuple(current))
    return tuple(groups)


__all__ = [
    "AsyncOmniAdmissionBatcher",
    "MultimodalReplicaInput",
    "OmniAdmissionMetrics",
]
