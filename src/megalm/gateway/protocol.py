from __future__ import annotations

import array
import base64
import os
import socket
import struct
from collections.abc import Mapping
from enum import IntEnum
from typing import Any

import orjson

from megalm.contracts.lora import LoRARequest
from megalm.contracts.request import (
    AudioChunk,
    EngineError,
    EngineEvent,
    EventKind,
    FinishReason,
    SamplingParams,
)
from megalm.megalm_rs import (
    decode_replica_frame_header as _rust_decode_frame_header,
)
from megalm.megalm_rs import (
    encode_replica_frame as _rust_encode_frame,
)


class ReplicaProtocolError(RuntimeError):
    """Raised when the local Replica channel violates its wire contract."""


class FrameType(IntEnum):
    READY = 1
    SUBMIT_BATCH = 2
    SUBMIT_RESULT = 3
    CANCEL = 4
    INSPECT = 5
    SNAPSHOT = 6
    OUTPUTS = 7
    STOP = 8
    FAILURE = 9
    SUBMIT_MULTIMODAL = 10
    SUBMIT_MULTIMODAL_BATCH = 11
    SUBMIT_EMBEDDINGS = 12
    EMBEDDING_RESULT = 13


_MAGIC = b"MGLM"
_VERSION = 6
_HEADER = struct.Struct("!4sBBI")
_MAX_FRAME_BYTES = 64 * 1024**2
_MAX_FRAME_FDS = 24
_FD_ITEM_SIZE = array.array("i").itemsize


def send_frame(
    channel: socket.socket,
    frame_type: FrameType,
    payload: dict[str, Any],
) -> int:
    frame = _encode_frame(frame_type, payload)
    channel.sendall(frame)
    return len(frame)


def send_frame_with_fds(
    channel: socket.socket,
    frame_type: FrameType,
    payload: dict[str, Any],
    fds: tuple[int, ...],
) -> int:
    """Send one control frame with sealed media descriptors out-of-band."""

    if not fds:
        return send_frame(channel, frame_type, payload)
    if len(fds) > _MAX_FRAME_FDS:
        raise ReplicaProtocolError(
            f"Replica frame exceeds {_MAX_FRAME_FDS} file descriptors"
        )
    if any(fd < 0 for fd in fds):
        raise ReplicaProtocolError("Replica frame contains an invalid descriptor")
    frame = _encode_frame(frame_type, payload)
    rights = array.array("i", fds)
    sent = channel.sendmsg(
        [frame],
        [(socket.SOL_SOCKET, socket.SCM_RIGHTS, rights)],
    )
    if sent <= 0:
        raise ReplicaProtocolError("Replica channel accepted no frame bytes")
    if sent < len(frame):
        channel.sendall(memoryview(frame)[sent:])
    return len(frame)


def _encode_frame(
    frame_type: FrameType,
    payload: dict[str, Any],
) -> bytes:
    body = orjson.dumps(payload, option=orjson.OPT_NON_STR_KEYS)
    try:
        return _rust_encode_frame(
            int(frame_type),
            body.decode("utf-8"),
        )
    except ValueError as exc:
        raise ReplicaProtocolError(str(exc)) from exc


def receive_frame(
    channel: socket.socket,
) -> tuple[FrameType, dict[str, Any]] | None:
    frame = receive_frame_with_fds(channel)
    if frame is None:
        return None
    frame_type, payload, fds = frame
    if fds:
        _close_fds(fds)
        raise ReplicaProtocolError(
            "Replica frame unexpectedly contains file descriptors"
        )
    return frame_type, payload


def receive_frame_with_fds(
    channel: socket.socket,
) -> tuple[FrameType, dict[str, Any], tuple[int, ...]] | None:
    """Receive one frame and preserve any Unix descriptor rights.

    ``recvmsg`` must read the first byte of every stream frame: descriptor
    rights are attached to that byte and would be discarded by ``recv``.
    """

    flags = getattr(socket, "MSG_CMSG_CLOEXEC", 0)
    first, ancillary, message_flags, _ = channel.recvmsg(
        _HEADER.size,
        socket.CMSG_SPACE(_MAX_FRAME_FDS * _FD_ITEM_SIZE),
        flags,
    )
    if not first:
        return None
    received_fds: list[int] = []
    try:
        if message_flags & getattr(socket, "MSG_CTRUNC", 0):
            raise ReplicaProtocolError(
                "Replica frame descriptor rights were truncated"
            )
        if message_flags & getattr(socket, "MSG_TRUNC", 0):
            raise ReplicaProtocolError("Replica frame header was truncated")
        for level, kind, raw_data in ancillary:
            if level != socket.SOL_SOCKET or kind != socket.SCM_RIGHTS:
                raise ReplicaProtocolError(
                    "Replica frame contains unsupported ancillary data"
                )
            if len(raw_data) % _FD_ITEM_SIZE:
                raise ReplicaProtocolError(
                    "Replica frame descriptor payload is malformed"
                )
            rights = array.array("i")
            rights.frombytes(raw_data)
            received_fds.extend(rights)
        if len(received_fds) > _MAX_FRAME_FDS:
            raise ReplicaProtocolError(
                f"Replica frame exceeds {_MAX_FRAME_FDS} file descriptors"
            )
        if not flags:
            for fd in received_fds:
                os.set_inheritable(fd, False)
        if len(first) < _HEADER.size:
            remainder = _receive_exact(
                channel,
                _HEADER.size - len(first),
                allow_clean_eof=False,
            )
            assert remainder is not None
            header = first + remainder
        else:
            header = first
        frame_type, payload = _decode_frame(channel, header)
        return frame_type, payload, tuple(received_fds)
    except BaseException:
        _close_fds(received_fds)
        raise


def _decode_frame(
    channel: socket.socket,
    header: bytes,
) -> tuple[FrameType, dict[str, Any]]:
    try:
        raw_type, payload_size = _rust_decode_frame_header(bytes(header))
        frame_type = FrameType(raw_type)
    except ValueError as exc:
        raise ReplicaProtocolError(str(exc)) from exc
    body = _receive_exact(channel, payload_size, allow_clean_eof=False)
    assert body is not None
    try:
        payload = orjson.loads(body)
    except orjson.JSONDecodeError as exc:
        raise ReplicaProtocolError("Replica frame contains invalid JSON") from exc
    if not isinstance(payload, dict):
        raise ReplicaProtocolError("Replica frame payload must be an object")
    return frame_type, payload


def sampling_to_wire(sampling: SamplingParams) -> dict[str, Any]:
    return {
        "temperature": sampling.temperature,
        "top_p": sampling.top_p,
        "top_k": sampling.top_k,
        "min_p": sampling.min_p,
        "repetition_penalty": sampling.repetition_penalty,
        "seed": sampling.seed,
        "stop_token_ids": sampling.stop_token_ids,
        "ignore_eos": sampling.ignore_eos,
        "logprobs": sampling.logprobs,
        "top_logprobs": sampling.top_logprobs,
        "score_labels": sampling.score_labels,
    }


def sampling_from_wire(payload: dict[str, Any]) -> SamplingParams:
    return SamplingParams(
        temperature=float(payload["temperature"]),
        top_p=float(payload["top_p"]),
        top_k=int(payload["top_k"]),
        min_p=float(payload["min_p"]),
        repetition_penalty=float(payload["repetition_penalty"]),
        seed=(None if payload.get("seed") is None else int(payload["seed"])),
        stop_token_ids=tuple(int(token) for token in payload["stop_token_ids"]),
        ignore_eos=bool(payload["ignore_eos"]),
        logprobs=bool(payload.get("logprobs", False)),
        top_logprobs=int(payload.get("top_logprobs", 0)),
        score_labels=tuple(
            int(token) for token in payload.get("score_labels", ())
        ),
    )


def lora_to_wire(lora: LoRARequest | None) -> dict[str, Any] | None:
    return None if lora is None else lora.to_wire()


def lora_from_wire(payload: Mapping[str, Any] | None) -> LoRARequest | None:
    return LoRARequest.from_wire(payload)


def outputs_to_wire(
    events: tuple[EngineEvent, ...],
    stream_ids: dict[str, int],
) -> list[list[Any]]:
    """Collapse Engine events into request-slot keyed streaming outputs.

    Lifecycle sequence numbers, states and timestamps stay Leader-owned.  The
    stateless Gateway only needs token IDs plus terminal status, so forwarding
    the full ``EngineEvent`` per token duplicates cold-path contract metadata
    on the hottest process boundary.
    """

    grouped: dict[int, list[Any]] = {}
    for event in events:
        if event.kind not in {
            EventKind.TOKEN,
            EventKind.AUDIO,
            EventKind.TERMINAL,
        }:
            continue
        stream_id = stream_ids.get(event.request_id)
        if stream_id is None:
            continue
        output = grouped.setdefault(
            stream_id,
            [stream_id, [], [], None, None, None, None],
        )
        if event.kind is EventKind.TOKEN:
            output[1].extend(event.token_ids)
            if event.logprobs:
                if output[5] is None:
                    output[5] = []
                output[5].extend(event.logprobs)
            if event.top_logprobs:
                if output[6] is None:
                    output[6] = []
                output[6].extend(
                    [
                        [[token_id, value] for token_id, value in row]
                        for row in event.top_logprobs
                    ]
                )
            continue
        if event.kind is EventKind.AUDIO:
            assert event.audio is not None
            output[2].append(
                [
                    base64.b64encode(event.audio.data).decode("ascii"),
                    event.audio.format,
                    event.audio.sample_rate,
                ]
            )
            continue
        if output[3] is not None:
            raise ReplicaProtocolError("duplicate terminal output in one frame")
        assert event.finish_reason is not None
        output[3] = event.finish_reason.value
        if event.error is not None:
            output[4] = [
                event.error.code,
                event.error.message,
                event.error.retryable,
            ]
    return list(grouped.values())


def output_from_wire(
    payload: list[Any],
) -> tuple[
    int,
    tuple[int, ...],
    tuple[AudioChunk, ...],
    FinishReason | None,
    EngineError | None,
    tuple[float, ...],
    tuple[tuple[tuple[int, float], ...], ...],
]:
    if not isinstance(payload, list) or len(payload) != 7:
        raise ReplicaProtocolError("Replica output payload is invalid")
    stream_id = payload[0]
    if isinstance(stream_id, bool) or not isinstance(stream_id, int) or stream_id <= 0:
        raise ReplicaProtocolError("Replica output stream id is invalid")
    raw_tokens = payload[1]
    if not isinstance(raw_tokens, list) or any(
        isinstance(token, bool) or not isinstance(token, int) or token < 0
        for token in raw_tokens
    ):
        raise ReplicaProtocolError("Replica output token payload is invalid")
    raw_audio = payload[2]
    if not isinstance(raw_audio, list):
        raise ReplicaProtocolError("Replica output audio payload is invalid")
    audio: list[AudioChunk] = []
    try:
        for item in raw_audio:
            if not isinstance(item, list) or len(item) != 3:
                raise ReplicaProtocolError(
                    "Replica output audio chunk is invalid"
                )
            audio.append(
                AudioChunk(
                    data=base64.b64decode(
                        str(item[0]),
                        validate=True,
                    ),
                    format=str(item[1]),
                    sample_rate=int(item[2]),
                )
            )
    except (ValueError, TypeError) as exc:
        raise ReplicaProtocolError(
            "Replica output audio chunk is invalid"
        ) from exc
    raw_reason = payload[3]
    try:
        finish_reason = (
            None if raw_reason is None else FinishReason(str(raw_reason))
        )
    except ValueError as exc:
        raise ReplicaProtocolError(
            "Replica output finish reason is invalid"
        ) from exc
    raw_error = payload[4]
    if raw_error is not None and (
        not isinstance(raw_error, list) or len(raw_error) != 3
    ):
        raise ReplicaProtocolError("Replica output error payload is invalid")
    error = (
        None
        if raw_error is None
        else EngineError(
            code=str(raw_error[0]),
            message=str(raw_error[1]),
            retryable=bool(raw_error[2]),
        )
    )
    if not raw_tokens and not audio and finish_reason is None:
        raise ReplicaProtocolError(
            "Replica output contains no token, audio, or terminal"
        )
    failed = finish_reason in {FinishReason.REJECTED, FinishReason.ERROR}
    if failed != (error is not None):
        raise ReplicaProtocolError(
            "Replica output error does not match its finish reason"
        )
    raw_logprobs = payload[5]
    if raw_logprobs is None:
        logprobs: tuple[float, ...] = ()
    else:
        if not isinstance(raw_logprobs, list) or len(raw_logprobs) != len(
            raw_tokens
        ):
            raise ReplicaProtocolError(
                "Replica output logprobs do not align with tokens"
            )
        if any(
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or value != value  # NaN check
            for value in raw_logprobs
        ):
            raise ReplicaProtocolError(
                "Replica output logprob payload is invalid"
            )
        logprobs = tuple(float(value) for value in raw_logprobs)
    raw_top_logprobs = payload[6]
    if raw_top_logprobs is None:
        top_logprobs: tuple[tuple[tuple[int, float], ...], ...] = ()
    else:
        if not isinstance(raw_top_logprobs, list) or len(
            raw_top_logprobs
        ) != len(raw_tokens):
            raise ReplicaProtocolError(
                "Replica output top logprobs do not align with tokens"
            )
        rows: list[tuple[tuple[int, float], ...]] = []
        for row in raw_top_logprobs:
            if not isinstance(row, list):
                raise ReplicaProtocolError(
                    "Replica output top logprob row is invalid"
                )
            entries: list[tuple[int, float]] = []
            for pair in row:
                if (
                    not isinstance(pair, list)
                    or len(pair) != 2
                    or isinstance(pair[0], bool)
                    or not isinstance(pair[0], int)
                    or pair[0] < 0
                    or isinstance(pair[1], bool)
                    or not isinstance(pair[1], (int, float))
                    or pair[1] != pair[1]
                ):
                    raise ReplicaProtocolError(
                        "Replica output top logprob pair is invalid"
                    )
                entries.append((pair[0], float(pair[1])))
            rows.append(tuple(entries))
        top_logprobs = tuple(rows)
    return (
        stream_id,
        tuple(raw_tokens),
        tuple(audio),
        finish_reason,
        error,
        logprobs,
        top_logprobs,
    )


def exception_to_wire(error: BaseException) -> dict[str, Any]:
    return {
        "type": type(error).__name__,
        "message": str(error) or type(error).__name__,
    }


def _receive_exact(
    channel: socket.socket,
    size: int,
    *,
    allow_clean_eof: bool,
) -> bytes | None:
    chunks: list[bytes] = []
    remaining = size
    while remaining:
        chunk = channel.recv(remaining)
        if not chunk:
            if allow_clean_eof and remaining == size:
                return None
            raise ReplicaProtocolError("Replica channel closed mid-frame")
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def _close_fds(fds: list[int] | tuple[int, ...]) -> None:
    for fd in fds:
        try:
            os.close(fd)
        except OSError:
            pass
