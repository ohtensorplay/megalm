from __future__ import annotations

import tensorplay


def vision_position_ids(
    start_position: int,
    grid_thw: tensorplay.Tensor,
    *,
    spatial_merge_size: int,
    device: tensorplay.device | str,
    temporal_positions: tensorplay.Tensor | None = None,
) -> tensorplay.Tensor:
    """Build one merged vision item's temporal/height/width positions."""

    if grid_thw.shape != (3,):
        raise ValueError("one vision grid must have shape [3]")
    temporal = int(grid_thw[0].item())
    height = int(grid_thw[1].item())
    width = int(grid_thw[2].item())
    if min(temporal, height, width, spatial_merge_size) <= 0:
        raise ValueError("vision grid dimensions must be positive")
    if height % spatial_merge_size or width % spatial_merge_size:
        raise ValueError("vision grid is not merge aligned")
    if temporal_positions is None:
        temporal_ids = tensorplay.arange(
            temporal,
            dtype=tensorplay.long,
            device=device,
        )
    else:
        if (
            temporal_positions.shape != (temporal,)
            or temporal_positions.dtype not in {tensorplay.int32, tensorplay.int64}
        ):
            raise ValueError(
                "temporal positions must be one integer value per grid"
            )
        temporal_ids = temporal_positions.to(device=device, dtype=tensorplay.long)
    llm_height = height // spatial_merge_size
    llm_width = width // spatial_merge_size
    width_ids = tensorplay.arange(llm_width, device=device) + start_position
    height_ids = tensorplay.arange(llm_height, device=device) + start_position
    width_ids = width_ids.repeat(llm_height * temporal)
    height_ids = height_ids.repeat_interleave(llm_width).repeat(temporal)
    temporal_ids = (
        temporal_ids.repeat_interleave(llm_height * llm_width)
        + start_position
    )
    return tensorplay.stack((temporal_ids, height_ids, width_ids), dim=0)


__all__ = ["vision_position_ids"]
