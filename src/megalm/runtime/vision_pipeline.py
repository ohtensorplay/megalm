from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Protocol
from uuid import uuid4

import tensorplay

from megalm.contracts.request import ImageInput, VideoInput
from megalm.runtime.vision_cache import (
    GPUVisionCache,
    VisionCacheKey,
    VisionCacheValue,
)


class VisionPipelineError(RuntimeError):
    """Raised when cached and newly encoded image features disagree."""


EncodeImages = Callable[[tensorplay.Tensor, tensorplay.Tensor], tensorplay.Tensor]


class _PreparedImages(Protocol):
    pixel_values: tensorplay.Tensor
    grid_thw: tensorplay.Tensor
    feature_counts: tuple[int, ...]


class _PreparedVideos(_PreparedImages, Protocol):
    frame_feature_counts: tuple[tuple[int, ...], ...]
    timestamps: tuple[tuple[float, ...], ...]
    seconds_per_grid: tuple[tuple[float, ...], ...]


@dataclass(frozen=True, slots=True)
class _PreparedVisionUnit:
    key: VisionCacheKey
    pixel_values: tensorplay.Tensor
    grid_thw: tuple[int, int, int]
    feature_count: int
    frame_feature_counts: tuple[int, ...] = ()
    timestamps: tuple[float, ...] = ()
    seconds_per_grid: tuple[float, ...] = ()

    @property
    def patch_count(self) -> int:
        return int(self.pixel_values.shape[0])


class ImageProcessor(Protocol):
    digest: str

    def prepare(
        self,
        images: tuple[ImageInput, ...],
    ) -> _PreparedImages: ...


class VideoProcessor(Protocol):
    def cache_digest(self, video: VideoInput) -> str: ...

    def prepare(
        self,
        videos: tuple[VideoInput, ...],
    ) -> _PreparedVideos: ...


@dataclass(frozen=True, slots=True)
class ResolvedImages:
    values: tuple[VisionCacheValue, ...]

    @property
    def feature_counts(self) -> tuple[int, ...]:
        return tuple(value.feature_count for value in self.values)

    @property
    def content_grids(self) -> tuple[tuple[int, int, int], ...]:
        return tuple(value.grid_thw for value in self.values)

    @property
    def embeddings(self) -> tuple[tensorplay.Tensor, ...]:
        return tuple(value.embedding for value in self.values)


@dataclass(frozen=True, slots=True)
class ResolvedVideos:
    values: tuple[VisionCacheValue, ...]

    @property
    def feature_counts(self) -> tuple[int, ...]:
        return tuple(value.feature_count for value in self.values)

    @property
    def content_grids(self) -> tuple[tuple[int, int, int], ...]:
        return tuple(value.grid_thw for value in self.values)

    @property
    def embeddings(self) -> tuple[tensorplay.Tensor, ...]:
        return tuple(value.embedding for value in self.values)

    @property
    def frame_feature_counts(self) -> tuple[tuple[int, ...], ...]:
        return tuple(value.frame_feature_counts for value in self.values)

    @property
    def timestamps(self) -> tuple[tuple[float, ...], ...]:
        return tuple(value.timestamps for value in self.values)

    @property
    def seconds_per_grid(self) -> tuple[tuple[float, ...], ...]:
        return tuple(value.seconds_per_grid for value in self.values)


class VisionPipeline:
    """Resolve per-image GPU embeddings through one bounded hot cache."""

    DEFAULT_MAX_ENCODE_PATCHES = 8_192

    def __init__(
        self,
        *,
        processor: ImageProcessor,
        video_processor: VideoProcessor | None,
        cache: GPUVisionCache,
        model_digest: str,
        encode_images: EncodeImages,
        device: tensorplay.device | str,
        max_encode_patches: int = DEFAULT_MAX_ENCODE_PATCHES,
    ) -> None:
        if max_encode_patches <= 0:
            raise VisionPipelineError(
                "vision tower patch batch limit must be positive"
            )
        self.processor = processor
        self.video_processor = video_processor
        self.cache = cache
        self.model_digest = model_digest
        self.encode_images = encode_images
        self.device = tensorplay.device(device)
        self.max_encode_patches = int(max_encode_patches)

    @tensorplay.inference_mode()
    def resolve(
        self,
        request_id: str,
        images: Sequence[ImageInput],
    ) -> ResolvedImages:
        ordered_images = tuple(images)
        if not ordered_images:
            raise VisionPipelineError("multimodal request has no images")
        keys = tuple(self._image_key(image) for image in ordered_images)
        resolved: dict[VisionCacheKey, VisionCacheValue] = {}
        missing: dict[VisionCacheKey, ImageInput] = {}
        try:
            for key, image in zip(keys, ordered_images):
                if key in resolved or key in missing:
                    continue
                value = self.cache.acquire(key, request_id)
                if value is None:
                    missing[key] = image
                else:
                    resolved[key] = value
            if missing:
                self._encode_missing(request_id, missing, resolved)
            occurrence: dict[VisionCacheKey, int] = {}
            values: list[VisionCacheValue] = []
            for key in keys:
                count = occurrence.get(key, 0)
                value = resolved[key]
                if count and value.cached:
                    acquired = self.cache.acquire(key, request_id)
                    if acquired is None:
                        raise VisionPipelineError(
                            "vision cache entry disappeared while pinned"
                        )
                    value = acquired
                occurrence[key] = count + 1
                values.append(value)
            return ResolvedImages(tuple(values))
        except BaseException:
            self.cache.release_request(request_id)
            raise

    @tensorplay.inference_mode()
    def resolve_batch(
        self,
        requests: Sequence[tuple[str, Sequence[ImageInput]]],
    ) -> tuple[ResolvedImages, ...]:
        """Encode the unique cold images from one admission wave together."""

        images, _ = self.resolve_media_batch(image_requests=requests)
        return images

    @tensorplay.inference_mode()
    def resolve_videos(
        self,
        request_id: str,
        videos: Sequence[VideoInput],
    ) -> ResolvedVideos:
        processor = self.video_processor
        if processor is None:
            raise VisionPipelineError(
                "the loaded model has no video processor"
            )
        ordered_videos = tuple(videos)
        if not ordered_videos:
            raise VisionPipelineError("multimodal request has no videos")
        keys = tuple(
            self._video_key(video) for video in ordered_videos
        )
        resolved: dict[VisionCacheKey, VisionCacheValue] = {}
        missing: dict[VisionCacheKey, VideoInput] = {}
        try:
            for key, video in zip(keys, ordered_videos):
                if key in resolved or key in missing:
                    continue
                value = self.cache.acquire(key, request_id)
                if value is None:
                    missing[key] = video
                else:
                    resolved[key] = value
            if missing:
                self._encode_missing_videos(
                    request_id,
                    missing,
                    resolved,
                )
            occurrence: dict[VisionCacheKey, int] = {}
            values: list[VisionCacheValue] = []
            for key in keys:
                count = occurrence.get(key, 0)
                value = resolved[key]
                if count and value.cached:
                    acquired = self.cache.acquire(key, request_id)
                    if acquired is None:
                        raise VisionPipelineError(
                            "vision cache entry disappeared while pinned"
                        )
                    value = acquired
                occurrence[key] = count + 1
                values.append(value)
            return ResolvedVideos(tuple(values))
        except BaseException:
            self.cache.release_request(request_id)
            raise

    @tensorplay.inference_mode()
    def resolve_videos_batch(
        self,
        requests: Sequence[tuple[str, Sequence[VideoInput]]],
    ) -> tuple[ResolvedVideos, ...]:
        """Encode the unique cold videos from one admission wave together."""

        _, videos = self.resolve_media_batch(video_requests=requests)
        return videos

    @tensorplay.inference_mode()
    def resolve_media_batch(
        self,
        *,
        image_requests: Sequence[
            tuple[str, Sequence[ImageInput]]
        ] = (),
        video_requests: Sequence[
            tuple[str, Sequence[VideoInput]]
        ] = (),
    ) -> tuple[tuple[ResolvedImages, ...], tuple[ResolvedVideos, ...]]:
        """Resolve one mixed visual wave through bounded tower launches."""

        ordered_images = tuple(
            (request_id, tuple(images))
            for request_id, images in image_requests
        )
        ordered_videos = tuple(
            (request_id, tuple(videos))
            for request_id, videos in video_requests
        )
        if not ordered_images and not ordered_videos:
            return (), ()
        if ordered_images:
            self._validate_batch_requests(
                ordered_images,
                media_name="images",
            )
        if ordered_videos:
            if self.video_processor is None:
                raise VisionPipelineError(
                    "the loaded model has no video processor"
                )
            self._validate_batch_requests(
                ordered_videos,
                media_name="videos",
            )

        image_keys = tuple(
            tuple(self._image_key(image) for image in images)
            for _, images in ordered_images
        )
        video_keys = tuple(
            tuple(self._video_key(video) for video in videos)
            for _, videos in ordered_videos
        )
        unique_images: dict[VisionCacheKey, ImageInput] = {}
        unique_videos: dict[VisionCacheKey, VideoInput] = {}
        for (_, images), keys in zip(
            ordered_images,
            image_keys,
            strict=True,
        ):
            for key, image in zip(keys, images, strict=True):
                unique_images.setdefault(key, image)
        for (_, videos), keys in zip(
            ordered_videos,
            video_keys,
            strict=True,
        ):
            for key, video in zip(keys, videos, strict=True):
                unique_videos.setdefault(key, video)

        staging_id = f"__megalm_visual_batch_{uuid4().hex}"
        request_ids = tuple(
            dict.fromkeys(
                request_id
                for request_id, _ in (
                    *ordered_images,
                    *ordered_videos,
                )
            )
        )
        resolved: dict[VisionCacheKey, VisionCacheValue] = {}
        missing_images: dict[VisionCacheKey, ImageInput] = {}
        missing_videos: dict[VisionCacheKey, VideoInput] = {}
        try:
            for key, image in unique_images.items():
                value = self.cache.acquire(key, staging_id)
                if value is None:
                    missing_images[key] = image
                else:
                    resolved[key] = value
            for key, video in unique_videos.items():
                value = self.cache.acquire(key, staging_id)
                if value is None:
                    missing_videos[key] = video
                else:
                    resolved[key] = value
            if missing_images or missing_videos:
                self._encode_missing_media(
                    staging_id,
                    missing_images,
                    missing_videos,
                    resolved,
                )
            images = tuple(
                ResolvedImages(
                    tuple(
                        self._retain_resolved(
                            key,
                            resolved[key],
                            request_id,
                        )
                        for key in keys
                    )
                )
                for (request_id, _), keys in zip(
                    ordered_images,
                    image_keys,
                    strict=True,
                )
            )
            videos = tuple(
                ResolvedVideos(
                    tuple(
                        self._retain_resolved(
                            key,
                            resolved[key],
                            request_id,
                        )
                        for key in keys
                    )
                )
                for (request_id, _), keys in zip(
                    ordered_videos,
                    video_keys,
                    strict=True,
                )
            )
            return images, videos
        except BaseException:
            for request_id in request_ids:
                self.cache.release_request(request_id)
            raise
        finally:
            self.cache.release_request(staging_id)

    def release(self, request_id: str) -> None:
        self.cache.release_request(request_id)

    def _retain_resolved(
        self,
        key: VisionCacheKey,
        value: VisionCacheValue,
        request_id: str,
    ) -> VisionCacheValue:
        if not value.cached:
            return value
        retained = self.cache.retain(key, request_id)
        if retained is None:
            raise VisionPipelineError(
                "vision cache entry disappeared while batch-pinned"
            )
        return retained

    @staticmethod
    def _validate_batch_requests(
        requests: Sequence[tuple[str, Sequence[object]]],
        *,
        media_name: str,
    ) -> None:
        request_ids = tuple(request_id for request_id, _ in requests)
        if (
            any(not request_id for request_id in request_ids)
            or len(set(request_ids)) != len(request_ids)
        ):
            raise VisionPipelineError(
                "vision batch request ids must be unique and non-empty"
            )
        if any(not media for _, media in requests):
            raise VisionPipelineError(
                f"vision batch request has no {media_name}"
            )

    def _encode_missing(
        self,
        request_id: str,
        missing: dict[VisionCacheKey, ImageInput],
        resolved: dict[VisionCacheKey, VisionCacheValue],
    ) -> None:
        keys = tuple(missing)
        prepared = self.processor.prepare(
            tuple(missing[key] for key in keys)
        )
        encoded = self.encode_images(
            prepared.pixel_values,
            prepared.grid_thw,
        )
        if encoded.ndim != 2 or encoded.shape[0] != sum(
            prepared.feature_counts
        ):
            raise VisionPipelineError(
                "vision tower output differs from merged token count"
            )
        offset = 0
        for key, grid, feature_count in zip(
            keys,
            prepared.grid_thw.tolist(),
            prepared.feature_counts,
        ):
            end = offset + feature_count
            embedding = encoded[offset:end].clone()
            resolved[key] = self.cache.insert_and_acquire(
                key,
                embedding,
                tuple(int(value) for value in grid),
                request_id,
            )
            offset = end

    def _encode_missing_media(
        self,
        request_id: str,
        missing_images: dict[VisionCacheKey, ImageInput],
        missing_videos: dict[VisionCacheKey, VideoInput],
        resolved: dict[VisionCacheKey, VisionCacheValue],
    ) -> None:
        units = [
            *self._prepare_image_units(missing_images),
            *self._prepare_video_units(missing_videos),
        ]
        groups: list[list[_PreparedVisionUnit]] = []
        current: list[_PreparedVisionUnit] = []
        current_patches = 0
        for unit in units:
            if current and (
                current_patches + unit.patch_count
                > self.max_encode_patches
            ):
                groups.append(current)
                current = []
                current_patches = 0
            current.append(unit)
            current_patches += unit.patch_count
        if current:
            groups.append(current)

        for group in groups:
            pixels = tensorplay.cat(
                [unit.pixel_values for unit in group],
                dim=0,
            )
            grids = tensorplay.tensor(
                [unit.grid_thw for unit in group],
                dtype=tensorplay.long,
            )
            encoded = self.encode_images(pixels, grids)
            expected = sum(unit.feature_count for unit in group)
            if encoded.ndim != 2 or encoded.shape[0] != expected:
                raise VisionPipelineError(
                    "vision tower output differs from merged token count"
                )
            offset = 0
            for unit in group:
                end = offset + unit.feature_count
                embedding = encoded[offset:end].clone()
                resolved[unit.key] = self.cache.insert_and_acquire(
                    unit.key,
                    embedding,
                    unit.grid_thw,
                    request_id,
                    frame_feature_counts=unit.frame_feature_counts,
                    timestamps=unit.timestamps,
                    seconds_per_grid=unit.seconds_per_grid,
                )
                offset = end

    def _prepare_image_units(
        self,
        missing: dict[VisionCacheKey, ImageInput],
    ) -> list[_PreparedVisionUnit]:
        if not missing:
            return []
        keys = tuple(missing)
        maximum = int(
            getattr(self.processor, "max_images", len(keys))
        )
        if maximum <= 0:
            raise VisionPipelineError(
                "image processor batch limit must be positive"
            )
        units: list[_PreparedVisionUnit] = []
        for start in range(0, len(keys), maximum):
            chunk = keys[start : start + maximum]
            prepared = self.processor.prepare(
                tuple(missing[key] for key in chunk)
            )
            units.extend(
                self._split_prepared_units(
                    chunk,
                    prepared,
                )
            )
        return units

    def _prepare_video_units(
        self,
        missing: dict[VisionCacheKey, VideoInput],
    ) -> list[_PreparedVisionUnit]:
        if not missing:
            return []
        processor = self.video_processor
        assert processor is not None
        keys = tuple(missing)
        maximum = int(
            getattr(processor, "max_videos", len(keys))
        )
        if maximum <= 0:
            raise VisionPipelineError(
                "video processor batch limit must be positive"
            )
        units: list[_PreparedVisionUnit] = []
        for start in range(0, len(keys), maximum):
            chunk = keys[start : start + maximum]
            prepared = processor.prepare(
                tuple(missing[key] for key in chunk)
            )
            units.extend(
                self._split_prepared_units(
                    chunk,
                    prepared,
                    frame_feature_counts=(
                        prepared.frame_feature_counts
                    ),
                    timestamps=prepared.timestamps,
                    seconds_per_grid=prepared.seconds_per_grid,
                )
            )
        return units

    @staticmethod
    def _split_prepared_units(
        keys: tuple[VisionCacheKey, ...],
        prepared: _PreparedImages,
        *,
        frame_feature_counts: tuple[tuple[int, ...], ...] | None = None,
        timestamps: tuple[tuple[float, ...], ...] | None = None,
        seconds_per_grid: tuple[tuple[float, ...], ...] | None = None,
    ) -> list[_PreparedVisionUnit]:
        grids = tuple(
            tuple(int(value) for value in grid)
            for grid in prepared.grid_thw.tolist()
        )
        if (
            len(keys) != len(grids)
            or len(keys) != len(prepared.feature_counts)
        ):
            raise VisionPipelineError(
                "processor output differs from visual input count"
            )
        patch_counts = tuple(
            grid[0] * grid[1] * grid[2] for grid in grids
        )
        if sum(patch_counts) != int(prepared.pixel_values.shape[0]):
            raise VisionPipelineError(
                "processor patch rows differ from visual grids"
            )
        empty = ((),) * len(keys)
        frame_counts = (
            empty
            if frame_feature_counts is None
            else frame_feature_counts
        )
        frame_times = empty if timestamps is None else timestamps
        grid_seconds = (
            empty
            if seconds_per_grid is None
            else seconds_per_grid
        )
        if not (
            len(frame_counts)
            == len(frame_times)
            == len(grid_seconds)
            == len(keys)
        ):
            raise VisionPipelineError(
                "video metadata differs from visual input count"
            )
        units: list[_PreparedVisionUnit] = []
        offset = 0
        for (
            key,
            grid,
            patch_count,
            feature_count,
            per_frame,
            per_timestamp,
            per_seconds,
        ) in zip(
            keys,
            grids,
            patch_counts,
            prepared.feature_counts,
            frame_counts,
            frame_times,
            grid_seconds,
            strict=True,
        ):
            end = offset + patch_count
            units.append(
                _PreparedVisionUnit(
                    key=key,
                    pixel_values=prepared.pixel_values[offset:end],
                    grid_thw=grid,
                    feature_count=feature_count,
                    frame_feature_counts=per_frame,
                    timestamps=per_timestamp,
                    seconds_per_grid=per_seconds,
                )
            )
            offset = end
        return units

    def _encode_missing_videos(
        self,
        request_id: str,
        missing: dict[VisionCacheKey, VideoInput],
        resolved: dict[VisionCacheKey, VisionCacheValue],
    ) -> None:
        processor = self.video_processor
        assert processor is not None
        keys = tuple(missing)
        maximum = int(
            getattr(processor, "max_videos", len(keys))
        )
        if maximum <= 0:
            raise VisionPipelineError(
                "video processor batch limit must be positive"
            )
        for start in range(0, len(keys), maximum):
            chunk = keys[start : start + maximum]
            self._encode_missing_video_chunk(
                request_id,
                chunk,
                missing,
                resolved,
            )

    def _encode_missing_video_chunk(
        self,
        request_id: str,
        keys: tuple[VisionCacheKey, ...],
        missing: dict[VisionCacheKey, VideoInput],
        resolved: dict[VisionCacheKey, VisionCacheValue],
    ) -> None:
        processor = self.video_processor
        assert processor is not None
        prepared = processor.prepare(
            tuple(missing[key] for key in keys)
        )
        encoded = self.encode_images(
            prepared.pixel_values,
            prepared.grid_thw,
        )
        if encoded.ndim != 2 or encoded.shape[0] != sum(
            prepared.feature_counts
        ):
            raise VisionPipelineError(
                "vision tower output differs from merged video token count"
            )
        offset = 0
        for (
            key,
            grid,
            feature_count,
            frame_feature_counts,
            timestamps,
            seconds_per_grid,
        ) in zip(
            keys,
            prepared.grid_thw.tolist(),
            prepared.feature_counts,
            prepared.frame_feature_counts,
            prepared.timestamps,
            prepared.seconds_per_grid,
        ):
            end = offset + feature_count
            embedding = encoded[offset:end].clone()
            resolved[key] = self.cache.insert_and_acquire(
                key,
                embedding,
                tuple(int(value) for value in grid),
                request_id,
                frame_feature_counts=frame_feature_counts,
                timestamps=timestamps,
                seconds_per_grid=seconds_per_grid,
            )
            offset = end

    def _image_key(self, image: ImageInput) -> VisionCacheKey:
        return VisionCacheKey(
            model_digest=self.model_digest,
            processor_digest=self.processor.digest,
            content_digest=image.content_digest,
            modality="image",
            detail=image.detail,
        )

    def _video_key(self, video: VideoInput) -> VisionCacheKey:
        processor = self.video_processor
        assert processor is not None
        return VisionCacheKey(
            model_digest=self.model_digest,
            processor_digest=processor.cache_digest(video),
            content_digest=video.content_digest,
            modality="video",
        )


__all__ = [
    "ResolvedImages",
    "ResolvedVideos",
    "VisionPipeline",
    "VisionPipelineError",
]
