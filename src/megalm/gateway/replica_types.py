from __future__ import annotations

from dataclasses import dataclass

from megalm.contracts.request import AudioChunk, EngineError, FinishReason


class GatewayError(RuntimeError):
    """Raised when the protocol frontend cannot safely relay a request."""


@dataclass(frozen=True, slots=True)
class ReplicaFailure:
    error: BaseException


@dataclass(frozen=True, slots=True)
class StreamBatch:
    token_ids: tuple[int, ...]
    audio: tuple[AudioChunk, ...] = ()
    finish_reason: FinishReason | None = None
    error: EngineError | None = None
    logprobs: tuple[float, ...] = ()


ReplicaOutput = StreamBatch | ReplicaFailure


__all__ = [
    "GatewayError",
    "ReplicaFailure",
    "ReplicaOutput",
    "StreamBatch",
]
