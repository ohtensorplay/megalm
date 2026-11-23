from __future__ import annotations

import tensorplay


def apply_vision_rope(
    hidden_states: tensorplay.Tensor,
    frequencies: tensorplay.Tensor,
) -> tensorplay.Tensor:
    source_dtype = hidden_states.dtype
    values = hidden_states.float()
    angles = tensorplay.cat((frequencies, frequencies), dim=-1)
    cos = angles.cos()[:, None, :]
    sin = angles.sin()[:, None, :]
    first, second = values.chunk(2, dim=-1)
    rotated = tensorplay.cat((-second, first), dim=-1)
    return (values * cos + rotated * sin).to(source_dtype)


def merged_vision_rotary(
    grid_thw: tensorplay.Tensor,
    inverse_frequency: tensorplay.Tensor,
    *,
    spatial_merge_size: int,
    device: tensorplay.device,
) -> tensorplay.Tensor:
    maximum = int(grid_thw[:, 1:].max().item())
    frequencies = (
        tensorplay.arange(
            maximum,
            dtype=tensorplay.float32,
            device=device,
        )[:, None]
        * inverse_frequency[None, :]
    )
    outputs: list[tensorplay.Tensor] = []
    merge = spatial_merge_size
    for temporal, height, width in grid_thw.tolist():
        rows = tensorplay.arange(height, device=device)
        columns = tensorplay.arange(width, device=device)
        row_ids = (
            rows.view(height // merge, merge, 1, 1)
            .expand(-1, -1, width // merge, merge)
            .permute(0, 2, 1, 3)
            .reshape(-1)
        )
        column_ids = (
            columns.view(1, 1, width // merge, merge)
            .expand(height // merge, merge, -1, -1)
            .permute(0, 2, 1, 3)
            .reshape(-1)
        )
        coordinates = tensorplay.stack((row_ids, column_ids), dim=-1)
        outputs.append(
            frequencies[coordinates].flatten(1).repeat(temporal, 1)
        )
    return tensorplay.cat(outputs, dim=0)


__all__ = ["apply_vision_rope", "merged_vision_rotary"]
