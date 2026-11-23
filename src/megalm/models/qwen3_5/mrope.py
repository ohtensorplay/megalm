from __future__ import annotations

import itertools

import tensorplay

from megalm.models.components.mrope import vision_position_ids


class MRoPEPositionError(ValueError):
    """Raised when media token groups and vision grids do not describe one prompt."""


def build_mrope_positions(
    token_type_ids: tensorplay.Tensor,
    image_grid_thw: tensorplay.Tensor,
    *,
    video_grid_thw: tensorplay.Tensor | None = None,
    spatial_merge_size: int,
) -> tuple[tensorplay.Tensor, int]:
    """Build Qwen3.5 text/image/video positions and the decode delta."""

    if token_type_ids.ndim != 1:
        raise MRoPEPositionError("token_type_ids must be one-dimensional")
    if token_type_ids.dtype not in {tensorplay.int32, tensorplay.int64}:
        raise MRoPEPositionError("token_type_ids must be integer")
    _validate_grids(image_grid_thw, name="image_grid_thw")
    if video_grid_thw is None:
        video_grid_thw = tensorplay.empty(
            (0, 3),
            dtype=tensorplay.long,
            device=token_type_ids.device,
        )
    _validate_grids(video_grid_thw, name="video_grid_thw")
    if spatial_merge_size <= 0:
        raise MRoPEPositionError("spatial merge size must be positive")
    type_values = tuple(int(value) for value in token_type_ids.tolist())
    if any(value not in {0, 1, 2} for value in type_values):
        raise MRoPEPositionError("unknown Qwen3.5 multimodal token type")

    expanded_video_grids = tensorplay.repeat_interleave(
        video_grid_thw,
        video_grid_thw[:, 0],
        dim=0,
    )
    if expanded_video_grids.numel():
        expanded_video_grids = expanded_video_grids.clone()
        expanded_video_grids[:, 0] = 1
    grids = {
        1: iter(image_grid_thw),
        2: iter(expanded_video_grids),
    }
    current_position = 0
    media_counts = {1: 0, 2: 0}
    sections: list[tensorplay.Tensor] = []
    for modality, grouped in itertools.groupby(
        enumerate(type_values),
        key=lambda item: item[1],
    ):
        rows = tuple(grouped)
        length = len(rows)
        if modality == 0:
            positions = tensorplay.arange(
                length,
                dtype=tensorplay.long,
                device=token_type_ids.device,
            )
            sections.append(
                positions.unsqueeze(0).expand(3, -1) + current_position
            )
            current_position += length
            continue
        try:
            grid = next(grids[modality])
        except StopIteration as exc:
            raise MRoPEPositionError(
                "media token group has no matching vision grid"
            ) from exc
        vision = vision_position_ids(
            current_position,
            grid,
            spatial_merge_size=spatial_merge_size,
            device=token_type_ids.device,
        )
        if vision.shape[1] != length:
            raise MRoPEPositionError(
                "media token count differs from its merged vision grid"
            )
        sections.append(vision)
        current_position += (
            max(int(grid[1].item()), int(grid[2].item()))
            // spatial_merge_size
        )
        media_counts[modality] += 1

    for modality, grid_iterator in grids.items():
        try:
            next(grid_iterator)
        except StopIteration:
            pass
        else:
            raise MRoPEPositionError(
                "vision grid has no matching media token group"
            )
    if media_counts[1] != image_grid_thw.shape[0]:
        raise MRoPEPositionError("image count differs from vision grids")
    if media_counts[2] != expanded_video_grids.shape[0]:
        raise MRoPEPositionError("video frame count differs from vision grids")
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
        raise MRoPEPositionError(f"{name} must be positive integer [N, 3]")


__all__ = [
    "MRoPEPositionError",
    "build_mrope_positions",
    "vision_position_ids",
]
