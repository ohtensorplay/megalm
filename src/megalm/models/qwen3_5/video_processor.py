from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass

import tensorplay

from megalm.contracts.request import VideoInput
from megalm.models.components import pack_video_patches
from megalm.models.components.video_decode import (
    DecodedVideo,
    VideoDecodeError,
    VideoMetadata,
    decode_video,
    linspace_indices,
    resize_rgb_frame,
)
from megalm.models.qwen3_5.config import Qwen35VisionConfig


class VideoProcessingError(ValueError):
    """Raised before GPU admission when a video is unsafe or malformed."""


@dataclass(frozen=True, slots=True)
class PreparedVideos:
    pixel_values: tensorplay.Tensor
    grid_thw: tensorplay.Tensor
    feature_counts: tuple[int, ...]
    frame_feature_counts: tuple[tuple[int, ...], ...]
    timestamps: tuple[tuple[float, ...], ...]
    seconds_per_grid: tuple[tuple[float, ...], ...]
    content_digests: tuple[str, ...]


class Qwen35VideoProcessor:
    """Bounded Qwen3.5 video sampling and temporal-patch preprocessing."""

    def __init__(
        self,
        config: Qwen35VisionConfig,
        *,
        min_pixels: int = 4_096,
        max_pixels: int = 25_165_824,
        max_encoded_bytes: int = 512 * 1024**2,
        max_videos: int = 4,
        min_frames: int = 4,
        max_frames: int = 256,
        max_duration_seconds: float = 3_600.0,
    ) -> None:
        if min(
            min_pixels,
            max_pixels,
            max_encoded_bytes,
            max_videos,
            min_frames,
            max_frames,
        ) <= 0:
            raise VideoProcessingError("video limits must be positive")
        if min_pixels > max_pixels or min_frames > max_frames:
            raise VideoProcessingError("video limits are inverted")
        if (
            not math.isfinite(max_duration_seconds)
            or max_duration_seconds <= 0
        ):
            raise VideoProcessingError(
                "maximum video duration must be positive"
            )
        self.config = config
        self.min_pixels = int(min_pixels)
        self.max_pixels = int(max_pixels)
        self.max_encoded_bytes = int(max_encoded_bytes)
        self.max_videos = int(max_videos)
        self.min_frames = int(min_frames)
        self.max_frames = int(max_frames)
        self.max_duration_seconds = float(max_duration_seconds)
        self.factor = config.patch_size * config.spatial_merge_size
        payload = {
            "abi": "megalm.qwen3_5.video_processor.v1",
            "backend": "opencv-random-access",
            "config": asdict(config),
            "min_pixels": self.min_pixels,
            "max_pixels": self.max_pixels,
            "max_encoded_bytes": self.max_encoded_bytes,
            "max_videos": self.max_videos,
            "min_frames": self.min_frames,
            "max_frames": self.max_frames,
            "max_duration_seconds": self.max_duration_seconds,
            "resample": "bicubic",
            "normalization": "rgb_half",
        }
        self.digest = hashlib.sha256(
            json.dumps(
                payload,
                sort_keys=True,
                separators=(",", ":"),
            ).encode()
        ).hexdigest()

    def cache_digest(self, video: VideoInput) -> str:
        return hashlib.sha256(
            f"{self.digest}:{video.sampling.digest}".encode()
        ).hexdigest()

    def prepare(
        self,
        videos: tuple[VideoInput, ...],
    ) -> PreparedVideos:
        if not videos or len(videos) > self.max_videos:
            raise VideoProcessingError(
                f"video count must be in [1, {self.max_videos}]"
            )
        patches: list[tensorplay.Tensor] = []
        grids: list[tuple[int, int, int]] = []
        feature_counts: list[int] = []
        frame_feature_counts: list[tuple[int, ...]] = []
        timestamps: list[tuple[float, ...]] = []
        seconds_per_grid: list[tuple[float, ...]] = []
        for video in videos:
            prepared = self._prepare_one(video)
            patches.append(prepared[0])
            grids.append(prepared[1])
            feature_counts.append(prepared[2])
            frame_feature_counts.append(prepared[3])
            timestamps.append(prepared[4])
            seconds_per_grid.append(prepared[5])
        return PreparedVideos(
            pixel_values=tensorplay.cat(patches, dim=0),
            grid_thw=tensorplay.tensor(grids, dtype=tensorplay.long),
            feature_counts=tuple(feature_counts),
            frame_feature_counts=tuple(frame_feature_counts),
            timestamps=tuple(timestamps),
            seconds_per_grid=tuple(seconds_per_grid),
            content_digests=tuple(video.content_digest for video in videos),
        )

    def _prepare_one(
        self,
        video: VideoInput,
    ) -> tuple[
        tensorplay.Tensor,
        tuple[int, int, int],
        int,
        tuple[int, ...],
        tuple[float, ...],
        tuple[float, ...],
    ]:
        decoded = self._decode(video)
        temporal = self.config.temporal_patch_size
        frames = list(decoded.frames)
        indices = list(decoded.frame_indices)
        if pad := -len(frames) % temporal:
            frames.extend(frames[-1:] * pad)
            indices.extend(indices[-1:] * pad)
        height, width = frames[0].shape[:2]
        resized_height, resized_width = smart_video_resize(
            len(frames),
            height,
            width,
            temporal_factor=temporal,
            factor=self.factor,
            min_pixels=self.min_pixels,
            max_pixels=self.max_pixels,
        )
        pixels = tensorplay.stack(
            [
                resize_rgb_frame(frame, resized_height, resized_width)
                for frame in frames
            ]
        )
        pixels = pixels.float().div_(255.0).sub_(0.5).div_(0.5)
        merge = self.config.spatial_merge_size
        patches, grid = pack_video_patches(
            pixels,
            in_channels=self.config.in_channels,
            temporal_patch_size=temporal,
            patch_size=self.config.patch_size,
            spatial_merge_size=merge,
        )
        grid_t, grid_h, grid_w = grid
        per_frame = grid_h * grid_w // merge**2
        timestamps = tuple(
            (
                indices[offset]
                + indices[offset + temporal - 1]
            )
            / (2.0 * decoded.metadata.source_fps)
            for offset in range(0, len(indices), temporal)
        )
        sampled_fps = len(indices) / decoded.metadata.duration_seconds
        seconds_per_grid = (
            temporal / sampled_fps,
        ) * grid_t
        return (
            patches,
            grid,
            grid_t * per_frame,
            (per_frame,) * grid_t,
            timestamps,
            seconds_per_grid,
        )

    def _decode(self, video: VideoInput) -> DecodedVideo:
        def select_frames(metadata: VideoMetadata) -> tuple[int, ...]:
            available = metadata.available_frames
            if video.sampling.num_frames is not None:
                sample_count = video.sampling.num_frames
            else:
                target_fps = video.sampling.target_fps
                assert target_fps is not None
                sample_count = int(
                    metadata.duration_seconds * target_fps
                )
            sample_count = min(
                max(sample_count, self.min_frames),
                self.max_frames,
                available,
            )
            return linspace_indices(
                metadata.start_frame,
                metadata.end_frame - 1,
                sample_count,
            )

        try:
            return decode_video(
                video,
                max_encoded_bytes=self.max_encoded_bytes,
                max_duration_seconds=self.max_duration_seconds,
                select_frames=select_frames,
            )
        except VideoDecodeError as exc:
            raise VideoProcessingError(str(exc)) from exc


def smart_video_resize(
    num_frames: int,
    height: int,
    width: int,
    *,
    temporal_factor: int,
    factor: int,
    min_pixels: int,
    max_pixels: int,
) -> tuple[int, int]:
    if min(
        num_frames,
        height,
        width,
        temporal_factor,
        factor,
    ) <= 0:
        raise VideoProcessingError("video dimensions must be positive")
    if height < factor or width < factor:
        raise VideoProcessingError(
            "video dimensions are smaller than one merged patch"
        )
    if max(height, width) / min(height, width) > 200:
        raise VideoProcessingError(
            "absolute video aspect ratio exceeds 200"
        )
    resized_height = round(height / factor) * factor
    resized_width = round(width / factor) * factor
    padded_frames = math.ceil(num_frames / temporal_factor) * temporal_factor
    volume = padded_frames * resized_height * resized_width
    if volume > max_pixels:
        scale = math.sqrt(num_frames * height * width / max_pixels)
        resized_height = max(
            factor,
            math.floor(height / scale / factor) * factor,
        )
        resized_width = max(
            factor,
            math.floor(width / scale / factor) * factor,
        )
    elif volume < min_pixels:
        scale = math.sqrt(min_pixels / (num_frames * height * width))
        resized_height = math.ceil(height * scale / factor) * factor
        resized_width = math.ceil(width * scale / factor) * factor
    return resized_height, resized_width


__all__ = [
    "PreparedVideos",
    "Qwen35VideoProcessor",
    "VideoProcessingError",
    "smart_video_resize",
]
