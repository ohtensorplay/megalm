from __future__ import annotations

import math


class MediaResizeError(ValueError):
    """Raised when bounded patch-aligned media resizing is impossible."""


# OpenAI's low-detail budget is a 512x512 square. The processor, not the
# gateway, applies it so the original encoded bytes and their digest stay
# intact and the model's own patch-aligned smart_resize runs once.
LOW_DETAIL_MAX_PIXELS = 512 * 512


def detail_pixel_limits(
    detail: str,
    min_pixels: int,
    max_pixels: int,
) -> tuple[int, int]:
    """Resolve effective resize limits from an image's detail hint.

    ``auto`` and ``high`` keep the processor's configured budget. ``low`` caps
    the budget at 512x512 (aligned by the caller's smart_resize) and lowers the
    minimum to match, so a large image is downscaled while a small one is not
    upscaled. The returned pair is always ordered.
    """

    if detail not in {"auto", "low", "high"}:
        raise MediaResizeError(f"unknown image detail {detail!r}")
    if detail != "low":
        return min_pixels, max_pixels
    low_max = min(max_pixels, LOW_DETAIL_MAX_PIXELS)
    low_min = min(min_pixels, low_max)
    return low_min, low_max


def smart_resize(
    height: int,
    width: int,
    *,
    factor: int,
    min_pixels: int,
    max_pixels: int,
) -> tuple[int, int]:
    """Match Qwen's aspect-preserving, patch-aligned image resize."""

    if min(height, width, factor, min_pixels, max_pixels) <= 0:
        raise MediaResizeError("media resize dimensions must be positive")
    if min_pixels > max_pixels:
        raise MediaResizeError("media resize pixel limits are inverted")
    if max(height, width) / min(height, width) > 200:
        raise MediaResizeError("absolute media aspect ratio exceeds 200")
    resized_height = round(height / factor) * factor
    resized_width = round(width / factor) * factor
    if resized_height * resized_width > max_pixels:
        scale = math.sqrt(height * width / max_pixels)
        resized_height = max(
            factor,
            math.floor(height / scale / factor) * factor,
        )
        resized_width = max(
            factor,
            math.floor(width / scale / factor) * factor,
        )
    elif resized_height * resized_width < min_pixels:
        scale = math.sqrt(min_pixels / (height * width))
        resized_height = math.ceil(height * scale / factor) * factor
        resized_width = math.ceil(width * scale / factor) * factor
    return resized_height, resized_width


__all__ = [
    "LOW_DETAIL_MAX_PIXELS",
    "MediaResizeError",
    "detail_pixel_limits",
    "smart_resize",
]
