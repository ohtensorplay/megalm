from __future__ import annotations

import tensorplay
import tensorplay.nn.functional as F


def frame_cu_seqlens(grid_thw: tensorplay.Tensor) -> tensorplay.Tensor:
    frame_lengths = tensorplay.repeat_interleave(
        grid_thw[:, 1] * grid_thw[:, 2],
        grid_thw[:, 0],
    )
    return F.pad(
        frame_lengths.cumsum(dim=0, dtype=tensorplay.int32),
        (1, 0),
    )


def merged_window_index(
    grid_thw: tensorplay.Tensor,
    *,
    spatial_merge_size: int,
    patch_size: int,
    window_size: int,
    device: tensorplay.device,
) -> tuple[tensorplay.Tensor, tensorplay.Tensor]:
    window = window_size // spatial_merge_size // patch_size
    if window <= 0:
        raise ValueError("vision attention window is smaller than a patch")
    indices: list[tensorplay.Tensor] = []
    cumulative = [0]
    offset = 0
    merge_unit = spatial_merge_size**2
    for temporal, height, width in grid_thw.tolist():
        llm_h = height // spatial_merge_size
        llm_w = width // spatial_merge_size
        index = tensorplay.arange(
            temporal * llm_h * llm_w,
            device=device,
        ).view(temporal, llm_h, llm_w)
        pad_h = window - llm_h % window
        pad_w = window - llm_w % window
        padded = F.pad(index, (0, pad_w, 0, pad_h), value=-100)
        windows_h = (llm_h + pad_h) // window
        windows_w = (llm_w + pad_w) // window
        padded = (
            padded.view(
                temporal,
                windows_h,
                window,
                windows_w,
                window,
            )
            .permute(0, 1, 3, 2, 4)
            .reshape(temporal, windows_h * windows_w, window, window)
        )
        lengths = (padded != -100).sum(dim=(2, 3)).reshape(-1)
        flattened = padded.reshape(-1)
        indices.append(flattened[flattened != -100] + offset)
        endpoints = lengths.cumsum(0) * merge_unit + cumulative[-1]
        cumulative.extend(int(value) for value in endpoints.tolist())
        offset += temporal * llm_h * llm_w
    cu_seqlens = tensorplay.tensor(
        cumulative,
        dtype=tensorplay.int32,
        device=device,
    )
    return (
        tensorplay.cat(indices).long(),
        tensorplay.unique_consecutive(cu_seqlens),
    )


__all__ = ["frame_cu_seqlens", "merged_window_index"]
