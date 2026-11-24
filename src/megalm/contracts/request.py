from __future__ import annotations

import hashlib
import json
import math
import time
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from enum import Enum

from megalm.contracts.lora import LoRARequest


class RequestContractError(ValueError):
    """Raised when a request or lifecycle operation violates the contract."""


class RequestState(str, Enum):
    CREATED = "created"
    ROUTING = "routing"
    LEASED = "leased"
    ADMITTED = "admitted"
    WAIT_MEDIA = "wait_media"
    READY = "ready"
    PREFILL = "prefill"
    DECODE = "decode"
    OUTPUT_BLOCKED = "output_blocked"
    PREEMPTED = "preempted"
    FINISHING = "finishing"
    FINISHED = "finished"
    CANCELLED = "cancelled"
    REJECTED = "rejected"
    FAILED = "failed"

    @property
    def is_terminal(self) -> bool:
        return self in {
            RequestState.FINISHED,
            RequestState.CANCELLED,
            RequestState.REJECTED,
            RequestState.FAILED,
        }

    @property
    def has_started_execution(self) -> bool:
        return self in {
            RequestState.PREFILL,
            RequestState.DECODE,
            RequestState.OUTPUT_BLOCKED,
            RequestState.PREEMPTED,
            RequestState.FINISHING,
            RequestState.FINISHED,
        }


class EventKind(str, Enum):
    STATE_CHANGED = "state_changed"
    TOKEN = "token"
    AUDIO = "audio"
    TERMINAL = "terminal"


class FinishReason(str, Enum):
    STOP = "stop"
    LENGTH = "length"
    CANCELLED = "cancelled"
    REJECTED = "rejected"
    ERROR = "error"


class OutputModality(str, Enum):
    TEXT = "text"
    AUDIO = "audio"


@dataclass(frozen=True, slots=True)
class AudioOutputOptions:
    """Protocol-neutral speech output controls understood by Omni stages."""

    voice: str = "Chelsie"
    format: str = "wav"

    def __post_init__(self) -> None:
        if not self.voice:
            raise RequestContractError("audio output voice must not be empty")
        if self.format != "wav":
            raise RequestContractError("MegaLM Omni audio output requires wav")


@dataclass(frozen=True, slots=True)
class AudioChunk:
    """One ordered byte range of an encoded audio response."""

    data: bytes = field(repr=False)
    format: str = "wav"
    sample_rate: int = 24_000

    def __post_init__(self) -> None:
        object.__setattr__(self, "data", bytes(self.data))
        if not self.data:
            raise RequestContractError("audio chunk must not be empty")
        if self.format != "wav":
            raise RequestContractError("audio chunk format must be wav")
        if self.sample_rate <= 0:
            raise RequestContractError("audio sample rate must be positive")


@dataclass(frozen=True, slots=True)
class StructuredOutput:
    """Engine-level constrained-decoding request, independent of any HTTP API.

    ``kind`` is ``"json_object"`` (any JSON object) or ``"json_schema"``
    (validated against ``schema_json``). The schema is kept as its canonical
    JSON text so the request stays immutable and hashable; the grammar manager
    compiles and caches grammars by this exact text.
    """

    kind: str
    schema_json: str | None = None

    def __post_init__(self) -> None:
        if self.kind not in {"json_object", "json_schema"}:
            raise RequestContractError(
                f"unsupported structured output kind {self.kind!r}"
            )
        if self.kind == "json_schema":
            if not isinstance(self.schema_json, str) or not self.schema_json:
                raise RequestContractError(
                    "json_schema structured output requires schema_json"
                )
            try:
                parsed = json.loads(self.schema_json)
            except json.JSONDecodeError as exc:
                raise RequestContractError(
                    f"schema_json is not valid JSON: {exc}"
                ) from exc
            if not isinstance(parsed, dict):
                raise RequestContractError("schema_json must be a JSON object")
            # Canonicalize so equivalent schemas share one compiled grammar.
            object.__setattr__(
                self,
                "schema_json",
                json.dumps(parsed, sort_keys=True, separators=(",", ":")),
            )
        elif self.schema_json is not None:
            raise RequestContractError(
                "json_object structured output must not carry schema_json"
            )


@dataclass(frozen=True, slots=True)
class SamplingParams:
    """Sampling inputs understood by the engine, independent of any HTTP API."""

    temperature: float = 0.0
    top_p: float = 1.0
    top_k: int = 0
    min_p: float = 0.0
    repetition_penalty: float = 1.0
    frequency_penalty: float = 0.0
    presence_penalty: float = 0.0
    seed: int | None = None
    stop_token_ids: tuple[int, ...] = ()
    ignore_eos: bool = False
    logprobs: bool = False
    # OpenAI-compatible top-k logprobs per sampled token. Requires ``logprobs``.
    top_logprobs: int = 0
    structured_output: StructuredOutput | None = None
    # Decision-model read: when non-empty, the engine reads the raw
    # next-token log-probabilities of exactly these token ids at the first
    # sampled position instead of taking a generative step. It is a free
    # function of the logits, so it composes with the existing scheduler,
    # KV arena and prefix cache without a second execution path.
    score_labels: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "stop_token_ids", tuple(self.stop_token_ids))
        object.__setattr__(self, "score_labels", tuple(self.score_labels))
        if not math.isfinite(self.temperature) or self.temperature < 0:
            raise RequestContractError("temperature must be non-negative")
        if not 0 < self.top_p <= 1:
            raise RequestContractError("top_p must be in (0, 1]")
        if (
            isinstance(self.top_k, bool)
            or not isinstance(self.top_k, int)
            or self.top_k < 0
        ):
            raise RequestContractError("top_k must be non-negative")
        if not 0 <= self.min_p <= 1:
            raise RequestContractError("min_p must be in [0, 1]")
        if (
            not math.isfinite(self.repetition_penalty)
            or self.repetition_penalty <= 0
        ):
            raise RequestContractError("repetition_penalty must be positive")
        for name in ("frequency_penalty", "presence_penalty"):
            value = getattr(self, name)
            if not math.isfinite(value) or not -2 <= value <= 2:
                raise RequestContractError(
                    f"{name} must be finite in [-2, 2]"
                )
        if self.seed is not None and (
            isinstance(self.seed, bool)
            or not isinstance(self.seed, int)
        ):
            raise RequestContractError("seed must be an integer")
        if any(
            isinstance(token_id, bool)
            or not isinstance(token_id, int)
            or token_id < 0
            for token_id in self.stop_token_ids
        ):
            raise RequestContractError("stop token ids must be non-negative")
        if not isinstance(self.ignore_eos, bool):
            raise RequestContractError("ignore_eos must be a boolean")
        if not isinstance(self.logprobs, bool):
            raise RequestContractError("logprobs must be a boolean")
        if (
            isinstance(self.top_logprobs, bool)
            or not isinstance(self.top_logprobs, int)
            or not 0 <= self.top_logprobs <= 20
        ):
            raise RequestContractError("top_logprobs must be in [0, 20]")
        if self.top_logprobs and not self.logprobs:
            raise RequestContractError("top_logprobs requires logprobs=True")
        if any(
            isinstance(token_id, bool)
            or not isinstance(token_id, int)
            or token_id < 0
            for token_id in self.score_labels
        ):
            raise RequestContractError("score label ids must be non-negative")
        if len(set(self.score_labels)) != len(self.score_labels):
            raise RequestContractError("score label ids must be unique")


IMAGE_DETAIL_VALUES = frozenset({"auto", "low", "high"})


@dataclass(frozen=True, slots=True)
class ImageInput:
    """An image delivered to a Replica; the Controller only receives its digest.

    ``detail`` is the OpenAI-style client hint. It is carried as a fact, never
    applied to the payload at the gateway: the vision processor chooses the
    effective pixel budget. Payload bytes and digest therefore stay identical
    across detail levels, while the vision cache key separates them.
    """

    content_digest: str
    mime_type: str
    payload: bytes = field(repr=False)
    detail: str = "auto"

    @classmethod
    def from_bytes(
        cls,
        payload: bytes,
        mime_type: str,
        *,
        detail: str = "auto",
    ) -> ImageInput:
        immutable_payload = bytes(payload)
        return cls(
            content_digest=hashlib.sha256(immutable_payload).hexdigest(),
            mime_type=mime_type,
            payload=immutable_payload,
            detail=detail,
        )

    def __post_init__(self) -> None:
        object.__setattr__(self, "payload", bytes(self.payload))
        if len(self.content_digest) != 64:
            raise RequestContractError("image content_digest must be a SHA-256 hex digest")
        try:
            int(self.content_digest, 16)
        except ValueError as exc:
            raise RequestContractError(
                "image content_digest must be a SHA-256 hex digest"
            ) from exc
        if hashlib.sha256(self.payload).hexdigest() != self.content_digest.lower():
            raise RequestContractError("image payload does not match content_digest")
        if not self.mime_type.startswith("image/"):
            raise RequestContractError("image mime_type must start with 'image/'")
        if self.detail not in IMAGE_DETAIL_VALUES:
            raise RequestContractError(
                "image detail must be one of 'auto', 'low' or 'high'"
            )


@dataclass(frozen=True, slots=True)
class VideoSampling:
    """Deterministic sampling policy included in the video-cache identity."""

    target_fps: float | None = 2.0
    num_frames: int | None = None
    start_seconds: float = 0.0
    end_seconds: float | None = None

    def __post_init__(self) -> None:
        if self.target_fps is not None and (
            not math.isfinite(self.target_fps) or self.target_fps <= 0
        ):
            raise RequestContractError("video target_fps must be positive")
        if self.num_frames is not None and (
            isinstance(self.num_frames, bool)
            or not isinstance(self.num_frames, int)
            or self.num_frames <= 0
        ):
            raise RequestContractError("video num_frames must be positive")
        if self.target_fps is not None and self.num_frames is not None:
            raise RequestContractError(
                "video target_fps and num_frames are mutually exclusive"
            )
        if (
            not math.isfinite(self.start_seconds)
            or self.start_seconds < 0
        ):
            raise RequestContractError(
                "video start_seconds must be non-negative"
            )
        if self.end_seconds is not None and (
            not math.isfinite(self.end_seconds)
            or self.end_seconds <= self.start_seconds
        ):
            raise RequestContractError(
                "video end_seconds must follow start_seconds"
            )

    @property
    def digest(self) -> str:
        payload = {
            "target_fps": self.target_fps,
            "num_frames": self.num_frames,
            "start_seconds": self.start_seconds,
            "end_seconds": self.end_seconds,
        }
        return hashlib.sha256(
            json.dumps(
                payload,
                sort_keys=True,
                separators=(",", ":"),
            ).encode()
        ).hexdigest()


@dataclass(frozen=True, slots=True)
class VideoInput:
    """An encoded video and its deterministic frame-sampling policy."""

    content_digest: str
    mime_type: str
    payload: bytes = field(repr=False)
    sampling: VideoSampling = field(default_factory=VideoSampling)

    @classmethod
    def from_bytes(
        cls,
        payload: bytes,
        mime_type: str,
        *,
        sampling: VideoSampling | None = None,
    ) -> VideoInput:
        immutable_payload = bytes(payload)
        return cls(
            content_digest=hashlib.sha256(immutable_payload).hexdigest(),
            mime_type=mime_type,
            payload=immutable_payload,
            sampling=sampling or VideoSampling(),
        )

    def __post_init__(self) -> None:
        object.__setattr__(self, "payload", bytes(self.payload))
        _validate_media_digest(
            self.content_digest,
            self.payload,
            modality="video",
        )
        if not self.mime_type.startswith("video/"):
            raise RequestContractError(
                "video mime_type must start with 'video/'"
            )


@dataclass(frozen=True, slots=True)
class AudioInput:
    """Encoded audio accepted by native Omni model adapters."""

    content_digest: str
    mime_type: str
    payload: bytes = field(repr=False)

    @classmethod
    def from_bytes(cls, payload: bytes, mime_type: str) -> AudioInput:
        immutable_payload = bytes(payload)
        return cls(
            content_digest=hashlib.sha256(immutable_payload).hexdigest(),
            mime_type=mime_type,
            payload=immutable_payload,
        )

    def __post_init__(self) -> None:
        object.__setattr__(self, "payload", bytes(self.payload))
        _validate_media_digest(
            self.content_digest,
            self.payload,
            modality="audio",
        )
        if not self.mime_type.startswith("audio/"):
            raise RequestContractError(
                "audio mime_type must start with 'audio/'"
            )


def _validate_media_digest(
    content_digest: str,
    payload: bytes,
    *,
    modality: str,
) -> None:
    if len(content_digest) != 64:
        raise RequestContractError(
            f"{modality} content_digest must be a SHA-256 hex digest"
        )
    try:
        int(content_digest, 16)
    except ValueError as exc:
        raise RequestContractError(
            f"{modality} content_digest must be a SHA-256 hex digest"
        ) from exc
    if hashlib.sha256(payload).hexdigest() != content_digest.lower():
        raise RequestContractError(
            f"{modality} payload does not match content_digest"
        )


@dataclass(frozen=True, slots=True)
class InferenceRequest:
    """Immutable request accepted by the engine data plane."""

    request_id: str
    model: str
    input_token_ids: tuple[int, ...]
    max_new_tokens: int
    sampling: SamplingParams = field(default_factory=SamplingParams)
    images: tuple[ImageInput, ...] = ()
    videos: tuple[VideoInput, ...] = ()
    audios: tuple[AudioInput, ...] = ()
    output_modalities: tuple[OutputModality, ...] = (OutputModality.TEXT,)
    audio_output: AudioOutputOptions | None = None
    priority: int = 0
    deadline_ns: int | None = None
    prefix_fingerprint: str | None = None
    # Per-request LoRA adapter. None means the base model, and is the only
    # value that ever reaches the model when LoRA is not enabled.
    lora_request: LoRARequest | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "input_token_ids", tuple(self.input_token_ids))
        object.__setattr__(self, "images", tuple(self.images))
        object.__setattr__(self, "videos", tuple(self.videos))
        object.__setattr__(self, "audios", tuple(self.audios))
        modalities = tuple(OutputModality(item) for item in self.output_modalities)
        object.__setattr__(self, "output_modalities", modalities)
        if not self.request_id:
            raise RequestContractError("request_id must not be empty")
        if not self.model:
            raise RequestContractError("model must not be empty")
        if not self.input_token_ids:
            raise RequestContractError("input_token_ids must not be empty")
        if any(token_id < 0 for token_id in self.input_token_ids):
            raise RequestContractError("input token ids must be non-negative")
        if self.max_new_tokens <= 0:
            raise RequestContractError("max_new_tokens must be positive")
        if not modalities or len(set(modalities)) != len(modalities):
            raise RequestContractError(
                "output modalities must be unique and non-empty"
            )
        if (
            OutputModality.AUDIO in modalities
        ) != (self.audio_output is not None):
            raise RequestContractError(
                "audio output options must match the audio output modality"
            )
        if self.deadline_ns is not None and self.deadline_ns <= 0:
            raise RequestContractError("deadline_ns must be a positive monotonic time")
        if self.prefix_fingerprint is not None:
            if len(self.prefix_fingerprint) != 64:
                raise RequestContractError(
                    "prefix_fingerprint must be a SHA-256 hex digest"
                )
            try:
                int(self.prefix_fingerprint, 16)
            except ValueError as exc:
                raise RequestContractError(
                    "prefix_fingerprint must be a SHA-256 hex digest"
                ) from exc
        if self.lora_request is not None and not isinstance(
            self.lora_request, LoRARequest
        ):
            raise RequestContractError(
                "lora_request must be a LoRARequest or None"
            )

    @property
    def media_hashes(self) -> tuple[str, ...]:
        return (
            *(
                f"image:{image.content_digest}:{image.detail}"
                for image in self.images
            ),
            *(
                f"video:{video.content_digest}:{video.sampling.digest}"
                for video in self.videos
            ),
            *(f"audio:{audio.content_digest}" for audio in self.audios),
        )


@dataclass(frozen=True, slots=True)
class EngineError:
    code: str
    message: str
    retryable: bool = False

    def __post_init__(self) -> None:
        if not self.code:
            raise RequestContractError("engine error code must not be empty")
        if not self.message:
            raise RequestContractError("engine error message must not be empty")


@dataclass(frozen=True, slots=True)
class EngineEvent:
    """Versioned Replica event. It intentionally carries token ids, not HTTP text."""

    request_id: str
    sequence_no: int
    kind: EventKind
    state: RequestState
    monotonic_ns: int
    token_ids: tuple[int, ...] = ()
    audio: AudioChunk | None = None
    finish_reason: FinishReason | None = None
    error: EngineError | None = None
    logprobs: tuple[float, ...] = ()
    # Per-token top-k (token_id, logprob) pairs, aligned with ``token_ids``.
    top_logprobs: tuple[tuple[tuple[int, float], ...], ...] = ()
    schema_version: int = 1

    def __post_init__(self) -> None:
        object.__setattr__(self, "token_ids", tuple(self.token_ids))
        object.__setattr__(self, "logprobs", tuple(self.logprobs))
        if self.top_logprobs:
            object.__setattr__(
                self,
                "top_logprobs",
                tuple(
                    tuple(
                        (int(token_id), float(value))
                        for token_id, value in row
                    )
                    for row in self.top_logprobs
                ),
            )
        if self.schema_version not in {1, 2}:
            raise RequestContractError("unsupported EngineEvent schema_version")
        if not self.request_id:
            raise RequestContractError("event request_id must not be empty")
        if self.sequence_no < 0:
            raise RequestContractError("event sequence_no must be non-negative")
        if self.kind is EventKind.TOKEN:
            if self.state is not RequestState.DECODE or not self.token_ids:
                raise RequestContractError(
                    "token events require DECODE state and at least one token"
                )
            if self.audio is not None:
                raise RequestContractError(
                    "token events cannot carry audio"
                )
            if self.logprobs and len(self.logprobs) != len(self.token_ids):
                raise RequestContractError(
                    "token logprobs must align with token ids"
                )
            if (
                self.top_logprobs
                and len(self.top_logprobs) != len(self.token_ids)
            ):
                raise RequestContractError(
                    "token top_logprobs must align with token ids"
                )
        elif self.kind is EventKind.AUDIO:
            if (
                self.schema_version != 2
                or self.state is not RequestState.OUTPUT_BLOCKED
                or self.audio is None
            ):
                raise RequestContractError(
                    "audio events require schema v2 OUTPUT_BLOCKED audio"
                )
            if self.token_ids:
                raise RequestContractError(
                    "audio events cannot carry token ids"
                )
        elif self.token_ids or self.top_logprobs:
            raise RequestContractError(
                "only token events may carry token ids or top_logprobs"
            )
        elif self.audio is not None:
            raise RequestContractError("only audio events may carry audio")
        if self.kind is EventKind.TERMINAL and not self.state.is_terminal:
            raise RequestContractError("terminal events require a terminal state")
        if self.state in {RequestState.REJECTED, RequestState.FAILED}:
            if self.error is None:
                raise RequestContractError("rejected/failed events require an error")
        elif self.error is not None:
            raise RequestContractError("only rejected/failed events may carry an error")


_TERMINAL_FROM_ANY = frozenset(
    {RequestState.CANCELLED, RequestState.FAILED}
)
_TRANSITIONS: dict[RequestState, frozenset[RequestState]] = {
    RequestState.CREATED: frozenset(
        {RequestState.ROUTING, RequestState.REJECTED}
    )
    | _TERMINAL_FROM_ANY,
    RequestState.ROUTING: frozenset(
        {RequestState.LEASED, RequestState.REJECTED}
    )
    | _TERMINAL_FROM_ANY,
    RequestState.LEASED: frozenset(
        {RequestState.ADMITTED, RequestState.REJECTED}
    )
    | _TERMINAL_FROM_ANY,
    RequestState.ADMITTED: frozenset(
        {RequestState.WAIT_MEDIA, RequestState.READY, RequestState.REJECTED}
    )
    | _TERMINAL_FROM_ANY,
    RequestState.WAIT_MEDIA: frozenset(
        {RequestState.READY, RequestState.REJECTED}
    )
    | _TERMINAL_FROM_ANY,
    RequestState.READY: frozenset(
        {RequestState.PREFILL, RequestState.REJECTED}
    )
    | _TERMINAL_FROM_ANY,
    RequestState.PREFILL: frozenset(
        {
            RequestState.DECODE,
            RequestState.PREEMPTED,
            RequestState.FINISHING,
        }
    )
    | _TERMINAL_FROM_ANY,
    RequestState.DECODE: frozenset(
        {
            RequestState.OUTPUT_BLOCKED,
            RequestState.PREEMPTED,
            RequestState.FINISHING,
        }
    )
    | _TERMINAL_FROM_ANY,
    RequestState.OUTPUT_BLOCKED: frozenset(
        {RequestState.DECODE, RequestState.PREEMPTED}
    )
    | _TERMINAL_FROM_ANY,
    RequestState.PREEMPTED: frozenset({RequestState.PREFILL})
    | _TERMINAL_FROM_ANY,
    RequestState.FINISHING: frozenset({RequestState.FINISHED})
    | _TERMINAL_FROM_ANY,
    RequestState.FINISHED: frozenset(),
    RequestState.CANCELLED: frozenset(),
    RequestState.REJECTED: frozenset(),
    RequestState.FAILED: frozenset(),
}


class RequestStateMachine:
    """Leader-owned lifecycle state with ordered event generation."""

    __slots__ = ("_first_token_emitted", "_request_id", "_sequence_no", "_state")

    def __init__(self, request_id: str) -> None:
        if not request_id:
            raise RequestContractError("request_id must not be empty")
        self._request_id = request_id
        self._state = RequestState.CREATED
        self._sequence_no = 0
        self._first_token_emitted = False

    @property
    def state(self) -> RequestState:
        return self._state

    @property
    def first_token_emitted(self) -> bool:
        return self._first_token_emitted

    @property
    def can_retry_after_replica_failure(self) -> bool:
        return not self._first_token_emitted and not self._state.is_terminal

    def transition(
        self,
        new_state: RequestState,
        *,
        finish_reason: FinishReason | None = None,
        error: EngineError | None = None,
        now_ns: int | None = None,
    ) -> EngineEvent:
        kind, finish_reason = self._advance(
            new_state,
            finish_reason=finish_reason,
            error=error,
        )
        return self._event(
            kind=kind,
            now_ns=now_ns,
            finish_reason=finish_reason,
            error=error,
        )

    def transition_silent(
        self,
        new_state: RequestState,
        *,
        finish_reason: FinishReason | None = None,
        error: EngineError | None = None,
    ) -> FinishReason | None:
        """Advance an internal lifecycle without materializing a wire event."""

        _, finish_reason = self._advance(
            new_state,
            finish_reason=finish_reason,
            error=error,
        )
        self._sequence_no += 1
        return finish_reason

    def _advance(
        self,
        new_state: RequestState,
        *,
        finish_reason: FinishReason | None,
        error: EngineError | None,
    ) -> tuple[EventKind, FinishReason | None]:
        if new_state not in _TRANSITIONS[self._state]:
            raise RequestContractError(
                f"invalid request transition: {self._state.value} -> "
                f"{new_state.value}"
            )
        if new_state is RequestState.REJECTED and self._state.has_started_execution:
            raise RequestContractError("a request cannot be rejected after execution starts")
        kind = (
            EventKind.TERMINAL
            if new_state.is_terminal
            else EventKind.STATE_CHANGED
        )
        if new_state in {RequestState.REJECTED, RequestState.FAILED}:
            if error is None:
                raise RequestContractError(f"{new_state.value} requires an error")
        elif error is not None:
            raise RequestContractError("errors are only valid for rejected/failed states")
        if kind is EventKind.TERMINAL and finish_reason is None:
            finish_reason = {
                RequestState.FINISHED: FinishReason.STOP,
                RequestState.CANCELLED: FinishReason.CANCELLED,
                RequestState.REJECTED: FinishReason.REJECTED,
                RequestState.FAILED: FinishReason.ERROR,
            }[new_state]
        elif kind is not EventKind.TERMINAL and finish_reason is not None:
            raise RequestContractError("finish_reason is only valid for terminal states")
        if kind is EventKind.TERMINAL:
            allowed_reasons = {
                RequestState.FINISHED: {FinishReason.STOP, FinishReason.LENGTH},
                RequestState.CANCELLED: {FinishReason.CANCELLED},
                RequestState.REJECTED: {FinishReason.REJECTED},
                RequestState.FAILED: {FinishReason.ERROR},
            }[new_state]
            if finish_reason not in allowed_reasons:
                raise RequestContractError(
                    f"{finish_reason.value} is invalid for {new_state.value}"
                )

        self._state = new_state
        return kind, finish_reason

    def emit_tokens(
        self,
        token_ids: Iterable[int],
        *,
        now_ns: int | None = None,
        logprobs: Sequence[float] = (),
        top_logprobs: Sequence[Sequence[tuple[int, float]]] = (),
    ) -> EngineEvent:
        tokens = tuple(token_ids)
        if self._state is not RequestState.DECODE:
            raise RequestContractError("tokens may only be emitted while decoding")
        if not tokens or any(token_id < 0 for token_id in tokens):
            raise RequestContractError("token event requires non-negative token ids")
        self._first_token_emitted = True
        return self._event(
            kind=EventKind.TOKEN,
            now_ns=now_ns,
            token_ids=tokens,
            logprobs=tuple(logprobs),
            top_logprobs=tuple(
                tuple(row) for row in top_logprobs
            ),
        )

    def emit_token_silent(self, token_id: int) -> None:
        """Record one scheduler-owned token without allocating EngineEvent."""

        if self._state is not RequestState.DECODE:
            raise RequestContractError("tokens may only be emitted while decoding")
        if token_id < 0:
            raise RequestContractError("token event requires a non-negative token id")
        self._first_token_emitted = True
        self._sequence_no += 1

    def _event(
        self,
        *,
        kind: EventKind,
        now_ns: int | None,
        token_ids: tuple[int, ...] = (),
        finish_reason: FinishReason | None = None,
        error: EngineError | None = None,
        logprobs: tuple[float, ...] = (),
        top_logprobs: tuple[tuple[tuple[int, float], ...], ...] = (),
    ) -> EngineEvent:
        event = EngineEvent(
            request_id=self._request_id,
            sequence_no=self._sequence_no,
            kind=kind,
            state=self._state,
            monotonic_ns=time.monotonic_ns() if now_ns is None else now_ns,
            token_ids=token_ids,
            finish_reason=finish_reason,
            error=error,
            logprobs=logprobs,
            top_logprobs=top_logprobs,
        )
        self._sequence_no += 1
        return event
