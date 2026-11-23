from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
import math
from pathlib import Path
import tempfile

import tensorplay
from PIL import Image

from megalm.contracts.request import VideoInput


class VideoDecodeError(ValueError):
    """Raised when bounded random-access video decoding fails."""


@dataclass(frozen=True, slots=True)
class VideoMetadata:
    total_frames: int
    source_fps: float
    width: int
    height: int
    start_frame: int
    end_frame: int

    @property
    def available_frames(self) -> int:
        return self.end_frame - self.start_frame

    @property
    def duration_seconds(self) -> float:
        return self.available_frames / self.source_fps


@dataclass(frozen=True, slots=True)
class DecodedVideo:
    frames: tuple[tensorplay.Tensor, ...]
    frame_indices: tuple[int, ...]
    metadata: VideoMetadata


FrameSelector = Callable[[VideoMetadata], tuple[int, ...]]


def decode_video(
    video: VideoInput,
    *,
    max_encoded_bytes: int,
    max_duration_seconds: float,
    select_frames: FrameSelector,
) -> DecodedVideo:
    """Decode selected RGB frames while bounding bytes, duration and geometry."""

    if len(video.payload) > max_encoded_bytes:
        raise VideoDecodeError("encoded video exceeds the byte limit")
    try:
        import cv2
    except ImportError as exc:
        raise VideoDecodeError(
            "the OpenCV video decoder backend is unavailable"
        ) from exc
    path: Path | None = None
    capture = None
    try:
        with tempfile.NamedTemporaryFile(
            prefix="megalm-video-",
            suffix=_video_suffix(video.mime_type),
            delete=False,
        ) as temporary:
            temporary.write(video.payload)
            path = Path(temporary.name)
        capture = cv2.VideoCapture(str(path))
        if not capture.isOpened():
            raise VideoDecodeError("cannot open encoded video")
        total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        source_fps = float(capture.get(cv2.CAP_PROP_FPS))
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if (
            total_frames <= 0
            or not math.isfinite(source_fps)
            or source_fps <= 0
            or min(width, height) <= 0
        ):
            raise VideoDecodeError("video metadata is missing or invalid")
        if total_frames / source_fps > max_duration_seconds:
            raise VideoDecodeError(
                "video duration exceeds the configured limit"
            )
        if max(width, height) / min(width, height) > 200:
            raise VideoDecodeError(
                "absolute video aspect ratio exceeds 200"
            )
        start = min(
            total_frames - 1,
            int(math.floor(video.sampling.start_seconds * source_fps)),
        )
        end = total_frames
        if video.sampling.end_seconds is not None:
            end = min(
                end,
                int(math.ceil(video.sampling.end_seconds * source_fps)),
            )
        metadata = VideoMetadata(
            total_frames=total_frames,
            source_fps=source_fps,
            width=width,
            height=height,
            start_frame=start,
            end_frame=end,
        )
        if metadata.available_frames <= 0:
            raise VideoDecodeError("video sampling window is empty")
        indices = tuple(int(index) for index in select_frames(metadata))
        if (
            not indices
            or any(
                index < start or index >= end
                for index in indices
            )
            or any(
                left > right
                for left, right in zip(indices, indices[1:])
            )
        ):
            raise VideoDecodeError(
                "video frame selector returned invalid indices"
            )
        frames: list[tensorplay.Tensor] = []
        for frame_index in indices:
            capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
            ok, bgr = capture.read()
            if not ok or bgr is None:
                raise VideoDecodeError(
                    f"cannot decode video frame {frame_index}"
                )
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            frames.append(tensorplay.from_numpy(rgb.copy()))
        return DecodedVideo(
            frames=tuple(frames),
            frame_indices=indices,
            metadata=metadata,
        )
    finally:
        if capture is not None:
            capture.release()
        if path is not None:
            path.unlink(missing_ok=True)


def linspace_indices(
    start: int,
    end: int,
    count: int,
) -> tuple[int, ...]:
    if count <= 0 or end < start:
        raise VideoDecodeError("linspace frame geometry is invalid")
    if count == 1:
        return (start,)
    span = end - start
    return tuple(
        start + round(index * span / (count - 1))
        for index in range(count)
    )


def resize_rgb_frame(
    frame: tensorplay.Tensor,
    height: int,
    width: int,
) -> tensorplay.Tensor:
    if (
        frame.ndim != 3
        or frame.shape[2] != 3
        or frame.dtype != tensorplay.uint8
        or min(height, width) <= 0
    ):
        raise VideoDecodeError("RGB frame geometry is invalid")
    image = Image.fromarray(frame.numpy(), mode="RGB")
    image = image.resize(
        (width, height),
        resample=Image.Resampling.BICUBIC,
    )
    pixels = tensorplay.frombuffer(
        bytearray(image.tobytes()),
        dtype=tensorplay.uint8,
    ).clone()
    return pixels.view(height, width, 3).permute(2, 0, 1)


def _video_suffix(mime_type: str) -> str:
    return {
        "video/mp4": ".mp4",
        "video/webm": ".webm",
        "video/quicktime": ".mov",
        "video/x-msvideo": ".avi",
        "video/mpeg": ".mpeg",
    }.get(mime_type.lower(), ".video")


__all__ = [
    "DecodedVideo",
    "VideoDecodeError",
    "VideoMetadata",
    "decode_video",
    "linspace_indices",
    "resize_rgb_frame",
]
