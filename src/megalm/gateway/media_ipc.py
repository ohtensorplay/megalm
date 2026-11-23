from __future__ import annotations

import fcntl
import hashlib
import os
import stat
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from megalm.contracts.request import (
    IMAGE_DETAIL_VALUES,
    AudioInput,
    ImageInput,
    VideoInput,
    VideoSampling,
)
from megalm.gateway.protocol import ReplicaProtocolError

MAX_MEDIA_FDS = 24
MAX_IMAGES = 16
MAX_VIDEOS = 4
MAX_AUDIOS = 8
MAX_IMAGE_BYTES = 32 * 1024**2
MAX_VIDEO_BYTES = 512 * 1024**2
MAX_AUDIO_BYTES = 64 * 1024**2
MAX_MEDIA_BYTES = 768 * 1024**2

_F_ADD_SEALS = getattr(fcntl, "F_ADD_SEALS", 1033)
_F_GET_SEALS = getattr(fcntl, "F_GET_SEALS", 1034)
_F_SEAL_SEAL = getattr(fcntl, "F_SEAL_SEAL", 0x0001)
_F_SEAL_SHRINK = getattr(fcntl, "F_SEAL_SHRINK", 0x0002)
_F_SEAL_GROW = getattr(fcntl, "F_SEAL_GROW", 0x0004)
_F_SEAL_WRITE = getattr(fcntl, "F_SEAL_WRITE", 0x0008)
_REQUIRED_SEALS = (
    _F_SEAL_WRITE
    | _F_SEAL_GROW
    | _F_SEAL_SHRINK
    | _F_SEAL_SEAL
)


@dataclass(slots=True)
class SealedMediaBatch:
    descriptors: tuple[dict[str, Any], ...]
    fds: tuple[int, ...]

    def close(self) -> None:
        for fd in self.fds:
            try:
                os.close(fd)
            except OSError:
                pass
        self.fds = ()

    def __enter__(self) -> SealedMediaBatch:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()


@dataclass(frozen=True, slots=True)
class MediaInputs:
    images: tuple[ImageInput, ...]
    videos: tuple[VideoInput, ...]
    audios: tuple[AudioInput, ...]


def seal_media_inputs(
    images: Sequence[ImageInput],
    videos: Sequence[VideoInput],
    audios: Sequence[AudioInput] = (),
) -> SealedMediaBatch:
    """Copy immutable request media into sealed anonymous Linux files."""

    ordered_images = tuple(images)
    ordered_videos = tuple(videos)
    ordered_audios = tuple(audios)
    _validate_counts(
        len(ordered_images),
        len(ordered_videos),
        len(ordered_audios),
    )
    descriptors: list[dict[str, Any]] = []
    fds: list[int] = []
    total_bytes = 0
    try:
        for image in ordered_images:
            total_bytes += len(image.payload)
            _validate_size("image", len(image.payload), total_bytes)
            fds.append(_sealed_memfd(image.payload))
            descriptors.append(
                {
                    "kind": "image",
                    "mime_type": image.mime_type,
                    "size": len(image.payload),
                    "sha256": image.content_digest,
                    "detail": image.detail,
                }
            )
        for video in ordered_videos:
            total_bytes += len(video.payload)
            _validate_size("video", len(video.payload), total_bytes)
            fds.append(_sealed_memfd(video.payload))
            descriptors.append(
                {
                    "kind": "video",
                    "mime_type": video.mime_type,
                    "size": len(video.payload),
                    "sha256": video.content_digest,
                    "sampling": {
                        "target_fps": video.sampling.target_fps,
                        "num_frames": video.sampling.num_frames,
                        "start_seconds": video.sampling.start_seconds,
                        "end_seconds": video.sampling.end_seconds,
                    },
                }
            )
        for audio in ordered_audios:
            total_bytes += len(audio.payload)
            _validate_size("audio", len(audio.payload), total_bytes)
            fds.append(_sealed_memfd(audio.payload))
            descriptors.append(
                {
                    "kind": "audio",
                    "mime_type": audio.mime_type,
                    "size": len(audio.payload),
                    "sha256": audio.content_digest,
                }
            )
        return SealedMediaBatch(tuple(descriptors), tuple(fds))
    except BaseException:
        for fd in fds:
            try:
                os.close(fd)
            except OSError:
                pass
        raise


def media_inputs_from_fds(
    descriptors: object,
    fds: tuple[int, ...],
) -> MediaInputs:
    """Validate untrusted descriptors and reconstruct immutable inputs."""

    try:
        if not isinstance(descriptors, list):
            raise ReplicaProtocolError("media descriptors must be a list")
        if len(descriptors) != len(fds) or not descriptors:
            raise ReplicaProtocolError(
                "media descriptors and file descriptors differ"
            )
        if len(fds) > MAX_MEDIA_FDS:
            raise ReplicaProtocolError("too many media file descriptors")
        kinds = [
            descriptor.get("kind")
            if isinstance(descriptor, dict)
            else None
            for descriptor in descriptors
        ]
        _validate_counts(
            kinds.count("image"),
            kinds.count("video"),
            kinds.count("audio"),
        )
        if any(kind not in {"image", "video", "audio"} for kind in kinds):
            raise ReplicaProtocolError("media descriptor kind is invalid")

        images: list[ImageInput] = []
        videos: list[VideoInput] = []
        audios: list[AudioInput] = []
        total_bytes = 0
        for descriptor, fd in zip(descriptors, fds, strict=True):
            assert isinstance(descriptor, dict)
            kind = str(descriptor["kind"])
            size = _wire_positive_int(descriptor.get("size"), "media size")
            total_bytes += size
            _validate_size(kind, size, total_bytes)
            digest = str(descriptor.get("sha256", "")).lower()
            mime_type = str(descriptor.get("mime_type", "")).lower()
            _validate_fd(fd, size)
            payload = _pread_exact(fd, size)
            if hashlib.sha256(payload).hexdigest() != digest:
                raise ReplicaProtocolError(
                    "media file does not match its SHA-256 descriptor"
                )
            if kind == "image":
                detail = descriptor.get("detail", "auto")
                if not isinstance(detail, str) or detail not in IMAGE_DETAIL_VALUES:
                    raise ReplicaProtocolError("image detail descriptor is invalid")
                images.append(
                    ImageInput(
                        content_digest=digest,
                        mime_type=mime_type,
                        payload=payload,
                        detail=detail,
                    )
                )
            elif kind == "video":
                raw_sampling = descriptor.get("sampling")
                if not isinstance(raw_sampling, dict):
                    raise ReplicaProtocolError(
                        "video sampling descriptor is invalid"
                    )
                videos.append(
                    VideoInput(
                        content_digest=digest,
                        mime_type=mime_type,
                        payload=payload,
                        sampling=VideoSampling(
                            target_fps=_optional_float(
                                raw_sampling.get("target_fps")
                            ),
                            num_frames=_optional_int(
                                raw_sampling.get("num_frames")
                            ),
                            start_seconds=float(
                                raw_sampling.get("start_seconds", 0.0)
                            ),
                            end_seconds=_optional_float(
                                raw_sampling.get("end_seconds")
                            ),
                        ),
                    )
                )
            else:
                audios.append(
                    AudioInput(
                        content_digest=digest,
                        mime_type=mime_type,
                        payload=payload,
                    )
                )
        return MediaInputs(
            tuple(images),
            tuple(videos),
            tuple(audios),
        )
    except (ReplicaProtocolError, ValueError):
        raise
    except BaseException as exc:
        raise ReplicaProtocolError("invalid media descriptor") from exc
    finally:
        for fd in fds:
            try:
                os.close(fd)
            except OSError:
                pass


def _sealed_memfd(payload: bytes) -> int:
    try:
        fd = os.memfd_create(
            "megalm-media",
            os.MFD_CLOEXEC | os.MFD_ALLOW_SEALING,
        )
    except (AttributeError, OSError) as exc:
        raise ReplicaProtocolError(
            "sealed memfd media transport is unavailable"
        ) from exc
    try:
        view = memoryview(payload)
        written = 0
        while written < len(view):
            count = os.write(fd, view[written:])
            if count <= 0:
                raise ReplicaProtocolError("media memfd write made no progress")
            written += count
        fcntl.fcntl(fd, _F_ADD_SEALS, _REQUIRED_SEALS)
        if fcntl.fcntl(fd, _F_GET_SEALS) & _REQUIRED_SEALS != _REQUIRED_SEALS:
            raise ReplicaProtocolError("media memfd seals were not applied")
        return fd
    except BaseException:
        os.close(fd)
        raise


def _validate_fd(fd: int, expected_size: int) -> None:
    details = os.fstat(fd)
    if not stat.S_ISREG(details.st_mode):
        raise ReplicaProtocolError("media descriptor is not a regular file")
    if details.st_size != expected_size:
        raise ReplicaProtocolError("media descriptor size mismatch")
    seals = fcntl.fcntl(fd, _F_GET_SEALS)
    if seals & _REQUIRED_SEALS != _REQUIRED_SEALS:
        raise ReplicaProtocolError("media descriptor is not fully sealed")


def _pread_exact(fd: int, size: int) -> bytes:
    chunks: list[bytes] = []
    offset = 0
    while offset < size:
        chunk = os.pread(fd, min(size - offset, 8 * 1024**2), offset)
        if not chunk:
            raise ReplicaProtocolError("media descriptor ended early")
        chunks.append(chunk)
        offset += len(chunk)
    return b"".join(chunks)


def _validate_counts(
    image_count: int,
    video_count: int,
    audio_count: int,
) -> None:
    if image_count < 0 or image_count > MAX_IMAGES:
        raise ReplicaProtocolError(f"at most {MAX_IMAGES} images are allowed")
    if video_count < 0 or video_count > MAX_VIDEOS:
        raise ReplicaProtocolError(f"at most {MAX_VIDEOS} videos are allowed")
    if audio_count < 0 or audio_count > MAX_AUDIOS:
        raise ReplicaProtocolError(f"at most {MAX_AUDIOS} audios are allowed")
    total_count = image_count + video_count + audio_count
    if total_count == 0:
        raise ReplicaProtocolError("a media batch cannot be empty")
    if total_count > MAX_MEDIA_FDS:
        raise ReplicaProtocolError("too many media inputs")


def _validate_size(kind: str, size: int, total_bytes: int) -> None:
    try:
        limit = {
            "image": MAX_IMAGE_BYTES,
            "video": MAX_VIDEO_BYTES,
            "audio": MAX_AUDIO_BYTES,
        }[kind]
    except KeyError as exc:
        raise ReplicaProtocolError("media kind is invalid") from exc
    if size <= 0 or size > limit:
        raise ReplicaProtocolError(f"{kind} payload exceeds its byte limit")
    if total_bytes > MAX_MEDIA_BYTES:
        raise ReplicaProtocolError("media batch exceeds its byte limit")


def _wire_positive_int(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ReplicaProtocolError(f"{name} is invalid")
    return value


def _optional_int(value: object) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int):
        raise ReplicaProtocolError("video num_frames is invalid")
    return value


def _optional_float(value: object) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ReplicaProtocolError("video sampling value is invalid")
    return float(value)


__all__ = [
    "MAX_AUDIOS",
    "MAX_AUDIO_BYTES",
    "MAX_IMAGES",
    "MAX_IMAGE_BYTES",
    "MAX_MEDIA_BYTES",
    "MAX_MEDIA_FDS",
    "MAX_VIDEOS",
    "MAX_VIDEO_BYTES",
    "MediaInputs",
    "SealedMediaBatch",
    "media_inputs_from_fds",
    "seal_media_inputs",
]
