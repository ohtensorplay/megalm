from __future__ import annotations

import base64
import binascii
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from megalm.contracts.request import (
    IMAGE_DETAIL_VALUES,
    AudioInput,
    ImageInput,
    VideoInput,
    VideoSampling,
)

MAX_HTTP_IMAGES = 8
MAX_HTTP_VIDEOS = 1
MAX_HTTP_AUDIOS = 4
MAX_HTTP_IMAGE_BYTES = 20 * 1024**2
MAX_HTTP_VIDEO_BYTES = 64 * 1024**2
MAX_HTTP_AUDIO_BYTES = 32 * 1024**2
MAX_HTTP_MEDIA_BYTES = 64 * 1024**2
MAX_CHAT_REQUEST_BYTES = (
    ((MAX_HTTP_MEDIA_BYTES + 2) // 3) * 4
    + 2 * 1024**2
)

_IMAGE_TYPES = {
    "image/gif",
    "image/jpeg",
    "image/png",
    "image/webp",
}
_VIDEO_TYPES = {
    "video/mp4",
    "video/quicktime",
    "video/webm",
    "video/x-matroska",
    "video/x-msvideo",
}
_AUDIO_TYPES = {
    "audio/flac",
    "audio/mp4",
    "audio/mpeg",
    "audio/ogg",
    "audio/wav",
    "audio/x-wav",
}
_INPUT_AUDIO_FORMATS = {
    "flac": "audio/flac",
    "mp3": "audio/mpeg",
    "mpeg": "audio/mpeg",
    "ogg": "audio/ogg",
    "wav": "audio/wav",
}


class MediaRequestError(ValueError):
    """Raised when HTTP media is unsafe or outside the supported contract."""


@dataclass(frozen=True, slots=True)
class ParsedMediaChat:
    messages: tuple[dict[str, Any], ...]
    images: tuple[ImageInput, ...]
    videos: tuple[VideoInput, ...]
    audios: tuple[AudioInput, ...]


def parse_media_chat(
    messages: Sequence[Mapping[str, Any]],
) -> ParsedMediaChat:
    normalized_messages: list[dict[str, Any]] = []
    images: list[ImageInput] = []
    videos: list[VideoInput] = []
    audios: list[AudioInput] = []
    total_bytes = 0

    for message in messages:
        content = message.get("content")
        normalized = dict(message)
        if isinstance(content, str):
            normalized_messages.append(normalized)
            continue
        if not isinstance(content, list) or not content:
            raise MediaRequestError(
                "message content must be text or a non-empty content-part list"
            )
        parts: list[dict[str, Any]] = []
        for part in content:
            if not isinstance(part, dict):
                raise MediaRequestError("message content part must be an object")
            kind = part.get("type")
            if kind == "text":
                text = part.get("text")
                if not isinstance(text, str):
                    raise MediaRequestError("text content part is invalid")
                parts.append({"type": "text", "text": text})
                continue
            if kind == "image_url":
                if len(images) >= MAX_HTTP_IMAGES:
                    raise MediaRequestError(
                        f"at most {MAX_HTTP_IMAGES} images are allowed"
                    )
                source = _source_object(part.get("image_url"), "image_url")
                detail = _validate_image_options(source)
                mime_type, payload = _decode_data_url(
                    source["url"],
                    modality="image",
                    limit=MAX_HTTP_IMAGE_BYTES,
                )
                total_bytes += len(payload)
                _validate_total(total_bytes)
                _validate_image_signature(mime_type, payload)
                images.append(
                    ImageInput.from_bytes(payload, mime_type, detail=detail)
                )
                parts.append({"type": "image"})
                continue
            if kind == "video_url":
                if len(videos) >= MAX_HTTP_VIDEOS:
                    raise MediaRequestError(
                        f"at most {MAX_HTTP_VIDEOS} video is allowed"
                    )
                source = _source_object(part.get("video_url"), "video_url")
                sampling = _video_sampling(source)
                mime_type, payload = _decode_data_url(
                    source["url"],
                    modality="video",
                    limit=MAX_HTTP_VIDEO_BYTES,
                )
                total_bytes += len(payload)
                _validate_total(total_bytes)
                _validate_video_signature(mime_type, payload)
                videos.append(
                    VideoInput.from_bytes(
                        payload,
                        mime_type,
                        sampling=sampling,
                    )
                )
                parts.append({"type": "video"})
                continue
            if kind in {"audio_url", "input_audio"}:
                if len(audios) >= MAX_HTTP_AUDIOS:
                    raise MediaRequestError(
                        f"at most {MAX_HTTP_AUDIOS} audio inputs are allowed"
                    )
                if kind == "audio_url":
                    source = _source_object(
                        part.get("audio_url"),
                        "audio_url",
                    )
                    if set(source) != {"url"}:
                        raise MediaRequestError(
                            "audio_url only supports the url field"
                        )
                    mime_type, payload = _decode_data_url(
                        source["url"],
                        modality="audio",
                        limit=MAX_HTTP_AUDIO_BYTES,
                    )
                else:
                    mime_type, payload = _decode_input_audio(
                        part.get("input_audio")
                    )
                total_bytes += len(payload)
                _validate_total(total_bytes)
                _validate_audio_signature(mime_type, payload)
                audios.append(AudioInput.from_bytes(payload, mime_type))
                parts.append({"type": "audio"})
                continue
            if kind == "audio":
                raise MediaRequestError(
                    "audio content must use audio_url or input_audio"
                )
            raise MediaRequestError(f"unsupported content part type {kind!r}")
        normalized["content"] = parts
        normalized_messages.append(normalized)

    if not images and not videos and not audios:
        raise MediaRequestError("multimodal chat contains no media")
    return ParsedMediaChat(
        messages=tuple(normalized_messages),
        images=tuple(images),
        videos=tuple(videos),
        audios=tuple(audios),
    )


def _source_object(value: object, name: str) -> dict[str, Any]:
    if isinstance(value, str):
        return {"url": value}
    if not isinstance(value, dict) or not isinstance(value.get("url"), str):
        raise MediaRequestError(f"{name} must contain a string URL")
    return value


def _validate_image_options(source: dict[str, Any]) -> str:
    """Validate image_url options and return the requested detail level."""

    unknown = set(source) - {"url", "detail"}
    if unknown:
        raise MediaRequestError(
            f"unsupported image options: {', '.join(sorted(unknown))}"
        )
    detail = source.get("detail", "auto")
    if detail not in IMAGE_DETAIL_VALUES:
        raise MediaRequestError(
            "image detail must be one of 'auto', 'low' or 'high'"
        )
    return detail


def _video_sampling(source: dict[str, Any]) -> VideoSampling:
    unknown = set(source) - {
        "url",
        "fps",
        "target_fps",
        "num_frames",
        "start_seconds",
        "end_seconds",
    }
    if unknown:
        raise MediaRequestError(
            f"unsupported video options: {', '.join(sorted(unknown))}"
        )
    if "fps" in source and "target_fps" in source:
        raise MediaRequestError("video fps and target_fps are aliases")
    explicit_fps = source.get("target_fps", source.get("fps"))
    num_frames = source.get("num_frames")
    target_fps = (
        None
        if num_frames is not None and explicit_fps is None
        else 2.0
        if explicit_fps is None
        else explicit_fps
    )
    try:
        return VideoSampling(
            target_fps=target_fps,
            num_frames=num_frames,
            start_seconds=source.get("start_seconds", 0.0),
            end_seconds=source.get("end_seconds"),
        )
    except (TypeError, ValueError) as exc:
        raise MediaRequestError(str(exc)) from exc


def _decode_data_url(
    value: object,
    *,
    modality: str,
    limit: int,
) -> tuple[str, bytes]:
    if not isinstance(value, str) or not value.startswith("data:"):
        raise MediaRequestError(
            f"{modality} URL must be an inline base64 data URL; "
            "remote and filesystem URLs are disabled"
        )
    header, separator, encoded = value.partition(",")
    if not separator or not header.endswith(";base64"):
        raise MediaRequestError(f"{modality} data URL must use base64")
    mime_type = header[5:-7].lower()
    allowed = {
        "image": _IMAGE_TYPES,
        "video": _VIDEO_TYPES,
        "audio": _AUDIO_TYPES,
    }.get(modality)
    if allowed is None:
        raise MediaRequestError(f"unsupported media modality {modality!r}")
    if mime_type not in allowed:
        raise MediaRequestError(
            f"unsupported {modality} MIME type {mime_type!r}"
        )
    if not encoded or len(encoded) > ((limit + 2) // 3) * 4:
        raise MediaRequestError(f"{modality} payload exceeds its byte limit")
    try:
        payload = base64.b64decode(encoded, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise MediaRequestError(
            f"{modality} data URL contains invalid base64"
        ) from exc
    if not payload or len(payload) > limit:
        raise MediaRequestError(f"{modality} payload exceeds its byte limit")
    return mime_type, payload


def _decode_input_audio(value: object) -> tuple[str, bytes]:
    if not isinstance(value, dict):
        raise MediaRequestError("input_audio must be an object")
    if set(value) != {"data", "format"}:
        raise MediaRequestError(
            "input_audio requires only data and format fields"
        )
    encoded = value.get("data")
    audio_format = str(value.get("format", "")).lower()
    mime_type = _INPUT_AUDIO_FORMATS.get(audio_format)
    if mime_type is None:
        raise MediaRequestError(
            f"unsupported input_audio format {audio_format!r}"
        )
    if not isinstance(encoded, str) or not encoded:
        raise MediaRequestError("input_audio data must be base64")
    if len(encoded) > ((MAX_HTTP_AUDIO_BYTES + 2) // 3) * 4:
        raise MediaRequestError("audio payload exceeds its byte limit")
    try:
        payload = base64.b64decode(encoded, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise MediaRequestError(
            "input_audio data contains invalid base64"
        ) from exc
    if not payload or len(payload) > MAX_HTTP_AUDIO_BYTES:
        raise MediaRequestError("audio payload exceeds its byte limit")
    return mime_type, payload


def _validate_total(total_bytes: int) -> None:
    if total_bytes > MAX_HTTP_MEDIA_BYTES:
        raise MediaRequestError("request media exceeds its total byte limit")


def _validate_image_signature(mime_type: str, payload: bytes) -> None:
    valid = {
        "image/png": payload.startswith(b"\x89PNG\r\n\x1a\n"),
        "image/jpeg": payload.startswith(b"\xff\xd8\xff"),
        "image/gif": payload.startswith((b"GIF87a", b"GIF89a")),
        "image/webp": (
            len(payload) >= 12
            and payload.startswith(b"RIFF")
            and payload[8:12] == b"WEBP"
        ),
    }[mime_type]
    if not valid:
        raise MediaRequestError("image bytes do not match the declared MIME type")


def _validate_video_signature(mime_type: str, payload: bytes) -> None:
    if mime_type in {"video/mp4", "video/quicktime"}:
        valid = len(payload) >= 12 and payload[4:8] == b"ftyp"
    elif mime_type in {"video/webm", "video/x-matroska"}:
        valid = payload.startswith(b"\x1a\x45\xdf\xa3")
    else:
        valid = (
            len(payload) >= 12
            and payload.startswith(b"RIFF")
            and payload[8:12] == b"AVI "
        )
    if not valid:
        raise MediaRequestError("video bytes do not match the declared MIME type")


def _validate_audio_signature(mime_type: str, payload: bytes) -> None:
    if mime_type in {"audio/wav", "audio/x-wav"}:
        valid = (
            len(payload) >= 12
            and payload.startswith(b"RIFF")
            and payload[8:12] == b"WAVE"
        )
    elif mime_type == "audio/flac":
        valid = payload.startswith(b"fLaC")
    elif mime_type == "audio/ogg":
        valid = payload.startswith(b"OggS")
    elif mime_type == "audio/mpeg":
        valid = payload.startswith(b"ID3") or (
            len(payload) >= 2
            and payload[0] == 0xFF
            and payload[1] & 0xE0 == 0xE0
        )
    else:
        valid = len(payload) >= 12 and payload[4:8] == b"ftyp"
    if not valid:
        raise MediaRequestError("audio bytes do not match the declared MIME type")


# Formats that share a magic-byte signature. A remote server may declare
# either member; accepting the alias keeps those responses usable without
# weakening the byte-vs-type check against unrelated formats.
_MIME_ALIASES = {
    "video/quicktime": "video/mp4",
    "audio/x-wav": "audio/wav",
}
_GENERIC_MIMES = frozenset({"application/octet-stream", "binary/octet-stream"})


def sniff_media_mime(payload: bytes, modality: str) -> str | None:
    """Detect a supported media MIME type from magic bytes only."""

    if modality == "image":
        if payload.startswith(b"\x89PNG\r\n\x1a\n"):
            return "image/png"
        if payload.startswith(b"\xff\xd8\xff"):
            return "image/jpeg"
        if payload.startswith((b"GIF87a", b"GIF89a")):
            return "image/gif"
        if (
            len(payload) >= 12
            and payload.startswith(b"RIFF")
            and payload[8:12] == b"WEBP"
        ):
            return "image/webp"
        return None
    if modality == "video":
        if len(payload) >= 12 and payload[4:8] == b"ftyp":
            return "video/mp4"
        if payload.startswith(b"\x1a\x45\xdf\xa3"):
            return "video/webm"
        if (
            len(payload) >= 12
            and payload.startswith(b"RIFF")
            and payload[8:12] == b"AVI "
        ):
            return "video/x-msvideo"
        return None
    if modality == "audio":
        if (
            len(payload) >= 12
            and payload.startswith(b"RIFF")
            and payload[8:12] == b"WAVE"
        ):
            return "audio/wav"
        if payload.startswith(b"fLaC"):
            return "audio/flac"
        if payload.startswith(b"OggS"):
            return "audio/ogg"
        if payload.startswith(b"ID3") or (
            len(payload) >= 2
            and payload[0] == 0xFF
            and payload[1] & 0xE0 == 0xE0
        ):
            return "audio/mpeg"
        if len(payload) >= 12 and payload[4:8] == b"ftyp":
            return "audio/mp4"
        return None
    return None


def resolve_media_mime(
    declared: str | None,
    payload: bytes,
    modality: str,
    allowed: frozenset[str],
) -> str:
    """Validate a fetched payload against its declared type and magic bytes.

    A missing or generic ``Content-Type`` is resolved purely from bytes. A
    specific declared type must agree with the bytes; a declared alias of the
    detected type is preserved so the caller sees the server's choice.
    """

    detected = sniff_media_mime(payload, modality)
    if detected is None:
        raise MediaRequestError(
            f"{modality} bytes are not a recognized media format"
        )
    normalized = declared.lower().split(";", 1)[0].strip() if declared else ""
    if normalized and normalized not in _GENERIC_MIMES:
        if normalized not in allowed:
            raise MediaRequestError(
                f"unsupported {modality} MIME type {normalized!r}"
            )
        validator = {
            "image": _validate_image_signature,
            "video": _validate_video_signature,
            "audio": _validate_audio_signature,
        }.get(modality)
        if validator is not None:
            try:
                validator(normalized, payload)
            except MediaRequestError:
                if _MIME_ALIASES.get(normalized, normalized) != _MIME_ALIASES.get(
                    detected, detected
                ):
                    raise
        return normalized
    if detected not in allowed:
        raise MediaRequestError(f"unsupported {modality} MIME type {detected!r}")
    return detected


__all__ = [
    "MAX_CHAT_REQUEST_BYTES",
    "MAX_HTTP_AUDIOS",
    "MAX_HTTP_AUDIO_BYTES",
    "MAX_HTTP_IMAGES",
    "MAX_HTTP_IMAGE_BYTES",
    "MAX_HTTP_MEDIA_BYTES",
    "MAX_HTTP_VIDEOS",
    "MAX_HTTP_VIDEO_BYTES",
    "MediaRequestError",
    "ParsedMediaChat",
    "parse_media_chat",
    "resolve_media_mime",
    "sniff_media_mime",
]
