"""Pure media geometry arithmetic for Gemma-4 (no TensorPlay dependency).

The gateway needs to know exactly how many placeholder tokens each image or
audio clip expands to, and the vision tower needs the ``(x, y)`` grid of each
patch. Both are derived here from the same formulas as the reference
``Gemma4Processor`` / ``Gemma4ImageProcessor`` so that prompt-side expansion
and encoder-side output length can never disagree.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class ImageGeometry:
    """Resolved geometry for one resized image."""

    target_h: int
    target_w: int
    patch_h: int
    patch_w: int
    num_soft_tokens: int

    def positions(self) -> list[tuple[int, int]]:
        """Return ``(x, y)`` coordinates in row-major patch order.

        Matches the reference image processor's
        ``torch.meshgrid(arange(patch_w), arange(patch_h))`` stacked on the
        last axis and flattened to ``(num_patches, 2)``.
        """
        return [
            (x, y)
            for y in range(self.patch_h)
            for x in range(self.patch_w)
        ]


def aspect_ratio_preserving_size(
    height: int,
    width: int,
    *,
    patch_size: int,
    max_soft_tokens: int,
    pooling_kernel_size: int,
) -> tuple[int, int]:
    """Largest aspect-preserving size within the patch budget.

    Both dimensions are rounded down to multiples of
    ``patch_size * pooling_kernel_size``. Degenerate aspect ratios are
    handled by pinning the short side to one unit and clamping the long side,
    exactly as the reference image processor does.
    """
    if height <= 0 or width <= 0:
        raise ValueError("image dimensions must be positive")
    max_patches = max_soft_tokens * pooling_kernel_size**2
    total_pixels = height * width
    target_pixels = max_patches * patch_size**2
    factor = math.sqrt(target_pixels / total_pixels)
    side_mult = pooling_kernel_size * patch_size
    target_h = int(math.floor(factor * height / side_mult)) * side_mult
    target_w = int(math.floor(factor * width / side_mult)) * side_mult
    if target_h == 0 and target_w == 0:
        raise ValueError("image resizes to 0 x 0")
    max_side = (max_patches // pooling_kernel_size**2) * side_mult
    if target_h == 0:
        target_h = side_mult
        target_w = min(int(math.floor(width / height)) * side_mult, max_side)
    elif target_w == 0:
        target_w = side_mult
        target_h = min(int(math.floor(height / width)) * side_mult, max_side)
    if target_h * target_w > target_pixels:
        raise ValueError("image exceeds the patch budget after resizing")
    return target_h, target_w


def image_geometry(
    height: int,
    width: int,
    *,
    patch_size: int,
    max_soft_tokens: int,
    pooling_kernel_size: int,
) -> ImageGeometry:
    """Resolve one image's target size and exact soft-token count."""
    target_h, target_w = aspect_ratio_preserving_size(
        height,
        width,
        patch_size=patch_size,
        max_soft_tokens=max_soft_tokens,
        pooling_kernel_size=pooling_kernel_size,
    )
    patch_h = target_h // patch_size
    patch_w = target_w // patch_size
    num_tokens = min(
        (patch_h * patch_w) // pooling_kernel_size**2,
        max_soft_tokens,
    )
    return ImageGeometry(target_h, target_w, patch_h, patch_w, num_tokens)


def num_audio_soft_tokens(
    num_samples: int,
    *,
    sampling_rate: int,
    audio_seq_length: int,
    frame_length: int,
    hop_length: int,
) -> int:
    """Soft-token count produced by the audio tower for one waveform.

    Mirrors semicausal mel framing followed by two stride-2 subsampling
    convolutions (kernel 3, padding 1), capped at ``audio_seq_length``.
    """
    if num_samples <= 0:
        return 0
    frame_size_for_unfold = frame_length + 1
    pad_left = frame_length // 2
    num_frames = (
        num_samples + pad_left - frame_size_for_unfold
    ) // hop_length + 1
    if num_frames <= 0:
        return 0
    for _ in range(2):
        num_frames = (num_frames + 2 - 3) // 2 + 1
    return min(num_frames, audio_seq_length)
