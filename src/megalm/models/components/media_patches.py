from __future__ import annotations

import tensorplay


def pack_image_patches(
    pixels: tensorplay.Tensor,
    *,
    in_channels: int,
    temporal_patch_size: int,
    patch_size: int,
    spatial_merge_size: int,
) -> tensorplay.Tensor:
    """Pack CHW pixels in the Qwen spatiotemporal/merge-major order."""

    if pixels.ndim != 3 or pixels.shape[0] != in_channels:
        raise ValueError("pixels must have shape [C, H, W]")
    height, width = pixels.shape[1:]
    factor = patch_size * spatial_merge_size
    if height % factor or width % factor:
        raise ValueError("image dimensions are not merge-patch aligned")
    grid_h = height // patch_size
    grid_w = width // patch_size
    patches = pixels.reshape(
        in_channels,
        grid_h // spatial_merge_size,
        spatial_merge_size,
        patch_size,
        grid_w // spatial_merge_size,
        spatial_merge_size,
        patch_size,
    )
    patches = patches.permute(1, 4, 2, 5, 0, 3, 6)
    return (
        patches.unsqueeze(5)
        .expand(
            -1,
            -1,
            -1,
            -1,
            -1,
            temporal_patch_size,
            -1,
            -1,
        )
        .reshape(
            grid_h * grid_w,
            in_channels * temporal_patch_size * patch_size**2,
        )
    )


def pack_video_patches(
    pixels: tensorplay.Tensor,
    *,
    in_channels: int,
    temporal_patch_size: int,
    patch_size: int,
    spatial_merge_size: int,
) -> tuple[tensorplay.Tensor, tuple[int, int, int]]:
    """Pack TCHW pixels in Qwen temporal/merge-major order."""

    if (
        pixels.ndim != 4
        or pixels.shape[1] != in_channels
        or pixels.shape[0] % temporal_patch_size
    ):
        raise ValueError(
            "video pixels must be aligned TCHW values"
        )
    frames, _, height, width = pixels.shape
    factor = patch_size * spatial_merge_size
    if height % factor or width % factor:
        raise ValueError("video dimensions are not merge-patch aligned")
    grid_t = frames // temporal_patch_size
    grid_h = height // patch_size
    grid_w = width // patch_size
    patches = pixels.reshape(
        grid_t,
        temporal_patch_size,
        in_channels,
        grid_h // spatial_merge_size,
        spatial_merge_size,
        patch_size,
        grid_w // spatial_merge_size,
        spatial_merge_size,
        patch_size,
    )
    patches = patches.permute(0, 3, 6, 4, 7, 2, 1, 5, 8)
    return (
        patches.reshape(
            grid_t * grid_h * grid_w,
            in_channels * temporal_patch_size * patch_size**2,
        ),
        (grid_t, grid_h, grid_w),
    )


__all__ = ["pack_image_patches", "pack_video_patches"]
