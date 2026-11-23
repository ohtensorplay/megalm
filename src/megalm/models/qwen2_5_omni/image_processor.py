from __future__ import annotations

import hashlib
import io
import json
import math
from dataclasses import asdict, dataclass

import tensorplay
from PIL import Image, UnidentifiedImageError

from megalm.contracts.request import ImageInput
from megalm.models.components import pack_image_patches
from megalm.models.components.media_resize import (
    MediaResizeError,
    detail_pixel_limits,
    smart_resize,
)
from megalm.models.qwen2_5_omni.config import OmniVisionConfig


class OmniImageProcessingError(ValueError):
    """Raised before admission when an Omni image cannot be processed safely."""


@dataclass(frozen=True, slots=True)
class PreparedOmniImages:
    pixel_values: tensorplay.Tensor
    grid_thw: tensorplay.Tensor
    feature_counts: tuple[int, ...]
    content_digests: tuple[str, ...]


class Qwen25OmniImageProcessor:
    """Bounded Qwen2-VL preprocessing without a Transformers runtime."""

    _MEAN = (0.48145466, 0.4578275, 0.40821073)
    _STD = (0.26862954, 0.26130258, 0.27577711)

    def __init__(
        self,
        config: OmniVisionConfig,
        *,
        min_pixels: int = 3_136,
        max_pixels: int = 12_845_056,
        max_encoded_bytes: int = 32 * 1024**2,
        max_images: int = 16,
    ) -> None:
        if min(
            min_pixels,
            max_pixels,
            max_encoded_bytes,
            max_images,
        ) <= 0:
            raise OmniImageProcessingError("image limits must be positive")
        if min_pixels > max_pixels:
            raise OmniImageProcessingError("image pixel limits are inverted")
        self.config = config
        self.min_pixels = int(min_pixels)
        self.max_pixels = int(max_pixels)
        self.max_encoded_bytes = int(max_encoded_bytes)
        self.max_images = int(max_images)
        self.factor = config.patch_size * config.spatial_merge_size
        self._mean = tensorplay.tensor(self._MEAN).view(3, 1, 1)
        self._std = tensorplay.tensor(self._STD).view(3, 1, 1)
        payload = {
            "abi": "megalm.qwen2_5_omni.image_processor.v1",
            "config": asdict(config),
            "min_pixels": self.min_pixels,
            "max_pixels": self.max_pixels,
            "max_encoded_bytes": self.max_encoded_bytes,
            "max_images": self.max_images,
            "resample": "bicubic",
            "normalization": "openai_clip",
        }
        self.digest = hashlib.sha256(
            json.dumps(
                payload,
                sort_keys=True,
                separators=(",", ":"),
            ).encode()
        ).hexdigest()

    def prepare(
        self,
        images: tuple[ImageInput, ...],
    ) -> PreparedOmniImages:
        if not images or len(images) > self.max_images:
            raise OmniImageProcessingError(
                f"image count must be in [1, {self.max_images}]"
            )
        patches: list[tensorplay.Tensor] = []
        grids: list[tuple[int, int, int]] = []
        feature_counts: list[int] = []
        for image in images:
            patch_values, grid = self._prepare_one(image)
            patches.append(patch_values)
            grids.append(grid)
            feature_counts.append(
                math.prod(grid) // self.config.spatial_merge_size**2
            )
        return PreparedOmniImages(
            pixel_values=tensorplay.cat(patches, dim=0),
            grid_thw=tensorplay.tensor(grids, dtype=tensorplay.long),
            feature_counts=tuple(feature_counts),
            content_digests=tuple(image.content_digest for image in images),
        )

    def _prepare_one(
        self,
        image_input: ImageInput,
    ) -> tuple[tensorplay.Tensor, tuple[int, int, int]]:
        if len(image_input.payload) > self.max_encoded_bytes:
            raise OmniImageProcessingError(
                "encoded image exceeds the byte limit"
            )
        try:
            with Image.open(io.BytesIO(image_input.payload)) as source:
                source.load()
                image = source.convert("RGB")
        except (UnidentifiedImageError, OSError, ValueError) as exc:
            raise OmniImageProcessingError(
                "cannot decode image payload"
            ) from exc
        width, height = image.size
        min_pixels, max_pixels = detail_pixel_limits(
            image_input.detail,
            self.min_pixels,
            self.max_pixels,
        )
        try:
            resized_height, resized_width = smart_resize(
                height,
                width,
                factor=self.factor,
                min_pixels=min_pixels,
                max_pixels=max_pixels,
            )
        except MediaResizeError as exc:
            raise OmniImageProcessingError(str(exc)) from exc
        image = image.resize(
            (resized_width, resized_height),
            resample=Image.Resampling.BICUBIC,
        )
        pixels = tensorplay.frombuffer(
            bytearray(image.tobytes()),
            dtype=tensorplay.uint8,
        ).clone()
        pixels = (
            pixels.view(resized_height, resized_width, 3)
            .permute(2, 0, 1)
            .float()
            .div_(255.0)
        )
        pixels = pixels.sub(self._mean).div(self._std)
        grid_h = resized_height // self.config.patch_size
        grid_w = resized_width // self.config.patch_size
        patches = pack_image_patches(
            pixels,
            in_channels=self.config.in_channels,
            temporal_patch_size=self.config.temporal_patch_size,
            patch_size=self.config.patch_size,
            spatial_merge_size=self.config.spatial_merge_size,
        )
        return patches, (1, grid_h, grid_w)


__all__ = [
    "OmniImageProcessingError",
    "PreparedOmniImages",
    "Qwen25OmniImageProcessor",
]
