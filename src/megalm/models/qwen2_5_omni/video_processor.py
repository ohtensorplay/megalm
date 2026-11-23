from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass

import tensorplay

from megalm.contracts.request import VideoInput
from megalm.models.components import pack_video_patches
from megalm.models.components.media_resize import (
    MediaResizeError,
    smart_resize,
)
from megalm.models.components.video_decode import (
    DecodedVideo,
    VideoDecodeError,
    VideoMetadata,
    decode_video,
    resize_rgb_frame,
)
from megalm.models.qwen2_5_omni.config import OmniVisionConfig


class OmniVideoProcessingError(ValueError):
    """Raised before admission when an Omni video is unsafe or malformed."""


@dataclass(frozen=True, slots=True)
class PreparedOmniVideos:
    pixel_values: tensorplay.Tensor
    grid_thw: tensorplay.Tensor
    feature_counts: tuple[int, ...]
    frame_feature_counts: tuple[tuple[int, ...], ...]
    timestamps: tuple[tuple[float, ...], ...]
    seconds_per_grid: tuple[tuple[float, ...], ...]
    content_digests: tuple[str, ...]


class Qwen25OmniVideoProcessor:
    """Official Qwen2-VL video packing with bounded native decoding."""

    _MEAN = (0.48145466, 0.4578275, 0.40821073)
    _STD = (0.26862954, 0.26130258, 0.27577711)

    def __init__(
        self,
        config: OmniVisionConfig,
        *,
        min_pixels: int = 128 * 28 * 28,
        max_pixels: int = 768 * 28 * 28,
        max_encoded_bytes: int = 512 * 1024**2,
        max_videos: int = 4,
        min_frames: int = 4,
        max_frames: int = 768,
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
            raise OmniVideoProcessingError(
                "video limits must be positive"
            )
        if min_pixels > max_pixels or min_frames > max_frames:
            raise OmniVideoProcessingError("video limits are inverted")
        if (
            not math.isfinite(max_duration_seconds)
            or max_duration_seconds <= 0
        ):
            raise OmniVideoProcessingError(
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
        self._mean = tensorplay.tensor(self._MEAN).view(1, 3, 1, 1)
        self._std = tensorplay.tensor(self._STD).view(1, 3, 1, 1)
        payload = {
            "abi": "megalm.qwen2_5_omni.video_processor.v1",
            "backend": "opencv-random-access",
            "config": asdict(config),
            "min_pixels": self.min_pixels,
            "max_pixels": self.max_pixels,
            "max_encoded_bytes": self.max_encoded_bytes,
            "max_videos": self.max_videos,
            "min_frames": self.min_frames,
            "max_frames": self.max_frames,
            "max_duration_seconds": self.max_duration_seconds,
            "sampling": "qwen2vl-aligned-uniform",
            "resample": "bicubic",
            "normalization": "openai_clip",
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
    ) -> PreparedOmniVideos:
        if not videos or len(videos) > self.max_videos:
            raise OmniVideoProcessingError(
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
        return PreparedOmniVideos(
            pixel_values=tensorplay.cat(patches, dim=0),
            grid_thw=tensorplay.tensor(grids, dtype=tensorplay.long),
            feature_counts=tuple(feature_counts),
            frame_feature_counts=tuple(frame_feature_counts),
            timestamps=tuple(timestamps),
            seconds_per_grid=tuple(seconds_per_grid),
            content_digests=tuple(
                video.content_digest for video in videos
            ),
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
        height, width = decoded.frames[0].shape[:2]
        try:
            resized_height, resized_width = smart_resize(
                height,
                width,
                factor=self.factor,
                min_pixels=self.min_pixels,
                max_pixels=self.max_pixels,
            )
        except MediaResizeError as exc:
            raise OmniVideoProcessingError(str(exc)) from exc
        pixels = tensorplay.stack(
            [
                resize_rgb_frame(
                    frame,
                    resized_height,
                    resized_width,
                )
                for frame in decoded.frames
            ]
        )
        pixels = (
            pixels.float()
            .div_(255.0)
            .sub_(self._mean)
            .div_(self._std)
        )
        patches, grid = pack_video_patches(
            pixels,
            in_channels=self.config.in_channels,
            temporal_patch_size=self.config.temporal_patch_size,
            patch_size=self.config.patch_size,
            spatial_merge_size=self.config.spatial_merge_size,
        )
        grid_t, grid_h, grid_w = grid
        merge = self.config.spatial_merge_size
        per_grid = grid_h * grid_w // merge**2
        temporal = self.config.temporal_patch_size
        indices = decoded.frame_indices
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
            grid_t * per_grid,
            (per_grid,) * grid_t,
            timestamps,
            seconds_per_grid,
        )

    def _decode(self, video: VideoInput) -> DecodedVideo:
        def select_frames(metadata: VideoMetadata) -> tuple[int, ...]:
            temporal = self.config.temporal_patch_size
            available = metadata.available_frames
            if video.sampling.num_frames is not None:
                requested = video.sampling.num_frames
                sample_count = round(requested / temporal) * temporal
            else:
                target_fps = video.sampling.target_fps
                assert target_fps is not None
                maximum = (
                    min(self.max_frames, available) // temporal
                ) * temporal
                inferred = metadata.duration_seconds * target_fps
                sample_count = math.floor(
                    min(
                        max(inferred, self.min_frames),
                        maximum,
                        available,
                    )
                    / temporal
                ) * temporal
            if sample_count <= 0 or sample_count > available:
                raise VideoDecodeError(
                    "video sampling cannot satisfy temporal alignment"
                )
            step = available / sample_count
            return tuple(
                metadata.start_frame + int(index * step)
                for index in range(sample_count)
            )

        try:
            return decode_video(
                video,
                max_encoded_bytes=self.max_encoded_bytes,
                max_duration_seconds=self.max_duration_seconds,
                select_frames=select_frames,
            )
        except VideoDecodeError as exc:
            raise OmniVideoProcessingError(str(exc)) from exc


__all__ = [
    "OmniVideoProcessingError",
    "PreparedOmniVideos",
    "Qwen25OmniVideoProcessor",
]
