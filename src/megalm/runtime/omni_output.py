from __future__ import annotations

import queue
import threading
import time
from dataclasses import dataclass
from typing import Protocol

import tensorplay

from megalm.contracts.request import (
    AudioChunk,
    EngineError as ContractEngineError,
    FinishReason,
)


class OmniOutputError(RuntimeError):
    """Raised when the optional Omni speech stages cannot run correctly."""


@dataclass(frozen=True, slots=True)
class OmniSpeechTask:
    request_id: str
    prompt_token_ids: tuple[int, ...]
    output_token_ids: tuple[int, ...]
    thinker_features: tensorplay.Tensor
    special_embeddings: tensorplay.Tensor
    voice: str
    finish_reason: FinishReason
    deadline_ns: int | None = None


@dataclass(frozen=True, slots=True)
class OmniSpeechResult:
    request_id: str
    chunks: tuple[AudioChunk, ...]
    finish_reason: FinishReason
    error: ContractEngineError | None = None


class OmniSpeechBackend(Protocol):
    @property
    def voices(self) -> tuple[str, ...]: ...

    def text_embedding_ids(self, voice: str) -> tuple[int, int, int]: ...

    def synthesize(
        self,
        task: OmniSpeechTask,
        cancelled: threading.Event,
    ) -> bytes: ...


class OmniSpeechStage:
    """One independently scheduled Talker/Token2Wav stage pair."""

    def __init__(
        self,
        backend: OmniSpeechBackend,
        *,
        max_pending: int = 32,
        audio_chunk_bytes: int = 48 * 1024,
    ) -> None:
        if max_pending <= 0 or audio_chunk_bytes <= 0:
            raise OmniOutputError("Omni output queue limits must be positive")
        if audio_chunk_bytes % 3:
            raise OmniOutputError(
                "audio chunk size must preserve concatenated base64 framing"
            )
        self.backend = backend
        self.audio_chunk_bytes = audio_chunk_bytes
        self._jobs: queue.Queue[
            tuple[OmniSpeechTask, threading.Event] | None
        ] = queue.Queue(maxsize=max_pending)
        self._results: queue.SimpleQueue[OmniSpeechResult] = (
            queue.SimpleQueue()
        )
        self._cancel: dict[str, threading.Event] = {}
        self._lock = threading.Lock()
        self._thread = threading.Thread(
            target=self._run,
            name="megalm-omni-output",
            daemon=True,
        )
        self._started = False

    @property
    def voices(self) -> tuple[str, ...]:
        return self.backend.voices

    @property
    def pending(self) -> bool:
        with self._lock:
            return bool(self._cancel)

    def start(self) -> None:
        if self._started:
            raise OmniOutputError("Omni output stage is already running")
        self._started = True
        self._thread.start()

    def close(self) -> None:
        if not self._started:
            return
        with self._lock:
            for cancelled in self._cancel.values():
                cancelled.set()
        self._jobs.put(None)
        self._thread.join()
        self._started = False

    def text_embedding_ids(self, voice: str) -> tuple[int, int, int]:
        return self.backend.text_embedding_ids(voice)

    def submit(self, task: OmniSpeechTask) -> None:
        if not self._started:
            raise OmniOutputError("Omni output stage is not running")
        cancelled = threading.Event()
        with self._lock:
            if task.request_id in self._cancel:
                raise OmniOutputError(
                    f"duplicate Omni output request {task.request_id!r}"
                )
            self._cancel[task.request_id] = cancelled
        try:
            self._jobs.put_nowait((task, cancelled))
        except queue.Full as exc:
            with self._lock:
                self._cancel.pop(task.request_id, None)
            raise OmniOutputError("Omni output queue is overloaded") from exc

    def cancel(self, request_id: str) -> bool:
        with self._lock:
            cancelled = self._cancel.get(request_id)
        if cancelled is None:
            return False
        cancelled.set()
        return True

    def drain(self) -> tuple[OmniSpeechResult, ...]:
        results: list[OmniSpeechResult] = []
        while True:
            try:
                results.append(self._results.get_nowait())
            except queue.Empty:
                return tuple(results)

    def _run(self) -> None:
        while True:
            item = self._jobs.get()
            if item is None:
                return
            task, cancelled = item
            try:
                if _stopped(task, cancelled):
                    result = OmniSpeechResult(
                        request_id=task.request_id,
                        chunks=(),
                        finish_reason=FinishReason.CANCELLED,
                    )
                else:
                    encoded = self.backend.synthesize(task, cancelled)
                    if _stopped(task, cancelled):
                        result = OmniSpeechResult(
                            request_id=task.request_id,
                            chunks=(),
                            finish_reason=FinishReason.CANCELLED,
                        )
                    else:
                        result = OmniSpeechResult(
                            request_id=task.request_id,
                            chunks=tuple(
                                AudioChunk(data=chunk)
                                for chunk in _chunk_bytes(
                                    encoded,
                                    self.audio_chunk_bytes,
                                )
                            ),
                            finish_reason=task.finish_reason,
                        )
            except BaseException as exc:
                result = OmniSpeechResult(
                    request_id=task.request_id,
                    chunks=(),
                    finish_reason=FinishReason.ERROR,
                    error=ContractEngineError(
                        code="omni_output_failed",
                        message=str(exc) or type(exc).__name__,
                        retryable=False,
                    ),
                )
            finally:
                with self._lock:
                    self._cancel.pop(task.request_id, None)
            self._results.put(result)


class TransformersOmniSpeechBackend:
    """Disabled on TensorPlay.

    The Qwen2.5-Omni Talker/Token2Wav stages were previously delegated to the
    Transformers implementation, which binds itself to a foreign tensor
    runtime. The TensorPlay migration removes that dependency, so audio output
    is disabled rather than silently mixing two incompatible tensor stacks.
    The native Thinker, image and audio *input* paths remain fully available.
    """

    def __init__(self, *args: object, **kwargs: object) -> None:
        del args, kwargs
        raise OmniOutputError(
            "Omni audio output is disabled: the Transformers Talker/Token2Wav "
            "backend is not part of the TensorPlay runtime"
        )


def _chunk_bytes(payload: bytes, size: int) -> tuple[bytes, ...]:
    if not payload:
        raise OmniOutputError("Omni output stage produced empty audio")
    return tuple(
        payload[offset : offset + size]
        for offset in range(0, len(payload), size)
    )


def _stopped(
    task: OmniSpeechTask,
    cancelled: threading.Event,
) -> bool:
    return cancelled.is_set() or (
        task.deadline_ns is not None
        and time.monotonic_ns() >= task.deadline_ns
    )


__all__ = [
    "OmniOutputError",
    "OmniSpeechBackend",
    "OmniSpeechResult",
    "OmniSpeechStage",
    "OmniSpeechTask",
    "TransformersOmniSpeechBackend",
]