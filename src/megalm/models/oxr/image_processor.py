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
from megalm.models.components.media_resize import detail_pixel_limits
from megalm.models.oxr.config import OXRVisionConfig


class OXRImageProcessingError(ValueError):
    """Raised before GPU admission when OXR media is malformed or too large."""


@dataclass(frozen=True, slots=True)
class PreparedImages:
    pixel_values: tensorplay.Tensor
    grid_thw: tensorplay.Tensor
    feature_counts: tuple[int, ...]
    content_digests: tuple[str, ...]


def smart_resize(
    height: int,
    width: int,
    *,
    factor: int,
    min_pixels: int,
    max_pixels: int,
) -> tuple[int, int]:
    """OXR's aspect-preserving, patch-aligned resize (ported from upstream)."""

    if min(height, width) <= 0:
        raise OXRImageProcessingError("image dimensions must be positive")
    if height < factor or width < factor:
        if height < width:
            width = round(factor / height * width)
            height = factor
        else:
            height = round(factor / width * height)
            width = factor
    elif max(height, width) / min(height, width) > 200:
        if height > width:
            height = 200 * width
        else:
            width = 200 * height
    resized_height = round(height / factor) * factor
    resized_width = round(width / factor) * factor
    if resized_height * resized_width > max_pixels:
        beta = math.sqrt((height * width) / max_pixels)
        resized_height = math.floor(height / beta / factor) * factor
        resized_width = math.floor(width / beta / factor) * factor
    elif resized_height * resized_width < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        resized_height = math.ceil(height * beta / factor) * factor
        resized_width = math.ceil(width * beta / factor) * factor
    return max(factor, resized_height), max(factor, resized_width)


class OXRImageProcessor:
    """Bounded, Transformers-free preprocessing for OXR document images."""

    def __init__(
        self,
        config: OXRVisionConfig,
        *,
        min_pixels: int = 448 * 448,
        max_pixels: int = 1344 * 1792,
        max_encoded_bytes: int = 64 * 1024**2,
        max_images: int = 8,
    ) -> None:
        if min(min_pixels, max_pixels, max_encoded_bytes, max_images) <= 0:
            raise OXRImageProcessingError("image limits must be positive")
        if min_pixels > max_pixels:
            raise OXRImageProcessingError("image pixel limits are inverted")
        self.config = config
        self.min_pixels = min_pixels
        self.max_pixels = max_pixels
        self.max_encoded_bytes = max_encoded_bytes
        self.max_images = max_images
        self.factor = config.patch_size * config.hidden_stride
        payload = {
            "abi": "megalm.oxr.image_processor.v1",
            "config": asdict(config),
            "min_pixels": min_pixels,
            "max_pixels": max_pixels,
            "max_encoded_bytes": max_encoded_bytes,
            "max_images": max_images,
            "resample": "bilinear",
            "normalization": "rgb_half",
        }
        self.digest = hashlib.sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()

    def prepare(self, images: tuple[ImageInput, ...]) -> PreparedImages:
        if not images or len(images) > self.max_images:
            raise OXRImageProcessingError(
                f"image count must be in [1, {self.max_images}]"
            )
        patches: list[tensorplay.Tensor] = []
        grids: list[tuple[int, int, int]] = []
        feature_counts: list[int] = []
        for image_input in images:
            patch, grid = self._prepare_one(image_input)
            patches.append(patch)
            grids.append(grid)
            feature_counts.append(
                math.prod(grid) // self.config.spatial_merge_size**2
            )
        return PreparedImages(
            pixel_values=tensorplay.cat(patches, dim=0),
            grid_thw=tensorplay.tensor(grids, dtype=tensorplay.long),
            feature_counts=tuple(feature_counts),
            content_digests=tuple(
                image.content_digest for image in images
            ),
        )

    def _prepare_one(
        self,
        image_input: ImageInput,
    ) -> tuple[tensorplay.Tensor, tuple[int, int, int]]:
        if len(image_input.payload) > self.max_encoded_bytes:
            raise OXRImageProcessingError("encoded image exceeds the byte limit")
        try:
            with Image.open(io.BytesIO(image_input.payload)) as source:
                source.load()
                image = source.convert("RGB")
        except (UnidentifiedImageError, OSError, ValueError) as exc:
            raise OXRImageProcessingError("cannot decode image payload") from exc
        width, height = image.size
        min_pixels, max_pixels = detail_pixel_limits(
            image_input.detail,
            self.min_pixels,
            self.max_pixels,
        )
        resized_height, resized_width = smart_resize(
            height,
            width,
            factor=self.factor,
            min_pixels=min_pixels,
            max_pixels=max_pixels,
        )
        image = image.resize(
            (resized_width, resized_height),
            resample=Image.Resampling.BILINEAR,
        )
        pixels = (
            tensorplay.frombuffer(
                bytearray(image.tobytes()), dtype=tensorplay.uint8
            )
            .clone()
            .view(resized_height, resized_width, 3)
            .permute(2, 0, 1)
            .float()
            .div_(255.0)
            .sub_(0.5)
            .div_(0.5)
        )
        grid_h = resized_height // self.config.patch_size
        grid_w = resized_width // self.config.patch_size
        patches = pack_image_patches(
            pixels,
            in_channels=self.config.num_channels,
            temporal_patch_size=self.config.temporal_patch_size,
            patch_size=self.config.patch_size,
            spatial_merge_size=self.config.spatial_merge_size,
        )
        return patches, (1, grid_h, grid_w)


__all__ = ["OXRImageProcessingError", "OXRImageProcessor", "PreparedImages"]
