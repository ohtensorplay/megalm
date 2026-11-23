"""detail is a passthrough fact; the processor owns the pixel budget."""

from __future__ import annotations

import pytest

from megalm.contracts.request import ImageInput, RequestContractError
from megalm.gateway.media_ipc import (
    media_inputs_from_fds,
    seal_media_inputs,
)
from megalm.models.components.media_resize import (
    LOW_DETAIL_MAX_PIXELS,
    MediaResizeError,
    detail_pixel_limits,
    smart_resize,
)


def _image(detail: str = "auto") -> ImageInput:
    return ImageInput.from_bytes(b"\x89PNG\r\n\x1a\n" + b"payload", "image/png", detail=detail)


def test_detail_pixel_limits_keep_auto_and_high() -> None:
    assert detail_pixel_limits("auto", 65_536, 16_777_216) == (
        65_536,
        16_777_216,
    )
    assert detail_pixel_limits("high", 65_536, 16_777_216) == (
        65_536,
        16_777_216,
    )


def test_detail_pixel_limits_cap_low_at_512_square() -> None:
    low_min, low_max = detail_pixel_limits("low", 65_536, 16_777_216)
    assert low_max == LOW_DETAIL_MAX_PIXELS
    assert low_min == 65_536
    assert low_min <= low_max

    # A processor whose minimum already exceeds the low budget still gets an
    # ordered pair instead of an inverted range.
    assert detail_pixel_limits("low", 400_000, 900_000) == (
        LOW_DETAIL_MAX_PIXELS,
        LOW_DETAIL_MAX_PIXELS,
    )

    with pytest.raises(MediaResizeError):
        detail_pixel_limits("ultra", 65_536, 16_777_216)


def test_low_detail_shrinks_large_image_without_pillow_resave() -> None:
    height, width = 1024, 1024
    factor = 32
    auto_h, auto_w = smart_resize(
        height, width, factor=factor, min_pixels=65_536, max_pixels=16_777_216
    )
    min_pixels, max_pixels = detail_pixel_limits("low", 65_536, 16_777_216)
    low_h, low_w = smart_resize(
        height, width, factor=factor, min_pixels=min_pixels, max_pixels=max_pixels
    )
    assert (auto_h, auto_w) == (1024, 1024)
    assert low_h * low_w <= LOW_DETAIL_MAX_PIXELS + factor * factor
    assert low_h % factor == 0 and low_w % factor == 0


def test_image_input_rejects_unknown_detail() -> None:
    with pytest.raises(RequestContractError, match="detail"):
        ImageInput.from_bytes(b"x" * 8, "image/png", detail="ultra")


def test_media_ipc_round_trips_detail() -> None:
    images = (_image("low"), _image("high"))
    with seal_media_inputs(images, ()) as sealed:
        restored = media_inputs_from_fds(list(sealed.descriptors), sealed.fds)
    assert [image.detail for image in restored.images] == ["low", "high"]
    assert [image.payload for image in restored.images] == [
        image.payload for image in images
    ]


def test_media_hashes_separate_detail_levels() -> None:
    request_kwargs = {
        "request_id": "r",
        "model": "m",
        "input_token_ids": (1,),
        "max_new_tokens": 1,
    }
    from megalm.contracts.request import InferenceRequest

    low = InferenceRequest(**request_kwargs, images=(_image("low"),))
    high = InferenceRequest(**request_kwargs, images=(_image("high"),))
    assert low.media_hashes != high.media_hashes
