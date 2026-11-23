from __future__ import annotations

import itertools
import math

import tensorplay

from megalm.models.components.mrope import vision_position_ids


class OmniMRoPEPositionError(ValueError):
    """Raised when Omni media tokens and temporal grids disagree."""


def build_omni_mrope_positions(
    token_type_ids: tensorplay.Tensor,
    image_grid_thw: tensorplay.Tensor,
    *,
    video_grid_thw: tensorplay.Tensor | None = None,
    video_seconds_per_grid: tuple[tuple[float, ...], ...] = (),
    audio_feature_counts: tuple[int, ...] = (),
    spatial_merge_size: int,
    position_ids_per_second: int,
) -> tuple[tensorplay.Tensor, int]:
    """Build official Qwen2.5-Omni text/image/video position IDs."""

    if token_type_ids.ndim != 1 or token_type_ids.dtype not in {
        tensorplay.int32,
        tensorplay.int64,
    }:
        raise OmniMRoPEPositionError(
            "token_type_ids must be one-dimensional integers"
        )
    _validate_grids(image_grid_thw, name="image_grid_thw")
    if video_grid_thw is None:
        video_grid_thw = tensorplay.empty(
            (0, 3),
            dtype=tensorplay.long,
            device=token_type_ids.device,
        )
    _validate_grids(video_grid_thw, name="video_grid_thw")
    if spatial_merge_size <= 0 or position_ids_per_second <= 0:
        raise OmniMRoPEPositionError(
            "Omni position scales must be positive"
        )
    if len(video_seconds_per_grid) != video_grid_thw.shape[0]:
        raise OmniMRoPEPositionError(
            "video temporal intervals differ from video grids"
        )
    intervals: list[tensorplay.Tensor] = []
    for grid, values in zip(video_grid_thw, video_seconds_per_grid):
        temporal = int(grid[0].item())
        if (
            len(values) != temporal
            or any(not math.isfinite(value) or value <= 0 for value in values)
        ):
            raise OmniMRoPEPositionError(
                "video temporal intervals are invalid"
            )
        intervals.append(
            tensorplay.tensor(
                values,
                dtype=tensorplay.float32,
                device=token_type_ids.device,
            )
        )
    type_values = tuple(int(value) for value in token_type_ids.tolist())
    if any(value not in {0, 1, 2, 3} for value in type_values):
        raise OmniMRoPEPositionError(
            "unknown Qwen2.5-Omni multimodal token type"
        )

    image_grids = iter(image_grid_thw)
    video_grids = iter(zip(video_grid_thw, intervals))
    audio_counts = iter(audio_feature_counts)
    media_counts = {1: 0, 2: 0, 3: 0}
    current_position = 0
    sections: list[tensorplay.Tensor] = []
    for modality, grouped in itertools.groupby(
        type_values,
        key=lambda value: value,
    ):
        length = sum(1 for _ in grouped)
        if modality == 0:
            positions = tensorplay.arange(
                length,
                dtype=tensorplay.long,
                device=token_type_ids.device,
            )
            section = (
                positions.unsqueeze(0).expand(3, -1)
                + current_position
            )
        elif modality == 1:
            try:
                grid = next(image_grids)
            except StopIteration as exc:
                raise OmniMRoPEPositionError(
                    "image token group has no matching vision grid"
                ) from exc
            temporal_positions = (
                tensorplay.arange(
                    int(grid[0].item()),
                    dtype=tensorplay.long,
                    device=token_type_ids.device,
                )
                * position_ids_per_second
            )
            section = vision_position_ids(
                current_position,
                grid,
                spatial_merge_size=spatial_merge_size,
                device=token_type_ids.device,
                temporal_positions=temporal_positions,
            )
            media_counts[1] += 1
        elif modality == 2:
            try:
                grid, seconds_per_grid = next(video_grids)
            except StopIteration as exc:
                raise OmniMRoPEPositionError(
                    "video token group has no matching temporal grid"
                ) from exc
            elapsed = tensorplay.cat(
                (
                    tensorplay.zeros(
                        1,
                        dtype=tensorplay.float32,
                        device=token_type_ids.device,
                    ),
                    tensorplay.cumsum(seconds_per_grid[:-1], dim=0),
                )
            )
            temporal_positions = (
                elapsed * position_ids_per_second
            ).long()
            section = vision_position_ids(
                current_position,
                grid,
                spatial_merge_size=spatial_merge_size,
                device=token_type_ids.device,
                temporal_positions=temporal_positions,
            )
            media_counts[2] += 1
        else:
            try:
                expected = int(next(audio_counts))
            except StopIteration as exc:
                raise OmniMRoPEPositionError(
                    "audio token group has no matching feature count"
                ) from exc
            if expected <= 0 or expected != length:
                raise OmniMRoPEPositionError(
                    "audio token count differs from encoded features"
                )
            positions = tensorplay.arange(
                length,
                dtype=tensorplay.long,
                device=token_type_ids.device,
            )
            section = (
                positions.unsqueeze(0).expand(3, -1)
                + current_position
            )
            media_counts[3] += 1
        if section.shape[1] != length:
            raise OmniMRoPEPositionError(
                "media token count differs from its merged vision grid"
            )
        sections.append(section)
        current_position = int(section.max().item()) + 1

    _assert_exhausted(image_grids, name="image")
    _assert_exhausted(video_grids, name="video")
    _assert_exhausted(audio_counts, name="audio")
    if media_counts[1] != image_grid_thw.shape[0]:
        raise OmniMRoPEPositionError(
            "image count differs from vision grids"
        )
    if media_counts[2] != video_grid_thw.shape[0]:
        raise OmniMRoPEPositionError(
            "video count differs from temporal grids"
        )
    if media_counts[3] != len(audio_feature_counts):
        raise OmniMRoPEPositionError(
            "audio count differs from feature metadata"
        )
    positions = tensorplay.cat(sections, dim=1)
    delta = int(positions.max().item()) + 1 - token_type_ids.numel()
    return positions, delta


def _validate_grids(grid_thw: tensorplay.Tensor, *, name: str) -> None:
    if (
        grid_thw.ndim != 2
        or grid_thw.shape[1] != 3
        or grid_thw.dtype not in {tensorplay.int32, tensorplay.int64}
        or bool((grid_thw <= 0).any().item())
    ):
        raise OmniMRoPEPositionError(
            f"{name} must be positive integer [N, 3]"
        )


def _assert_exhausted(iterator: object, *, name: str) -> None:
    try:
        next(iterator)  # type: ignore[arg-type]
    except StopIteration:
        return
    raise OmniMRoPEPositionError(
        f"{name} grid has no matching media token group"
    )


__all__ = [
    "OmniMRoPEPositionError",
    "build_omni_mrope_positions",
]
