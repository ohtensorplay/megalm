"""Transformers-free Gemma-4 media frontends for gateway admission.

Both processors produce the exact geometry the native towers expect and a
stable digest so cached tower outputs cannot be reused across a config or
frontend change.
"""

from __future__ import annotations

import hashlib
import io
import json
import math
from dataclasses import dataclass
from typing import Sequence

import numpy as np
import tensorplay
from PIL import Image, UnidentifiedImageError

from megalm.contracts.request import AudioInput, ImageInput
from megalm.models.gemma4.config import Gemma4AudioConfig, Gemma4VisionConfig
from megalm.models.gemma4.media import image_geometry, num_audio_soft_tokens


class Gemma4MediaError(ValueError):
    """Raised before GPU admission when media is malformed or too large."""


@dataclass(frozen=True, slots=True)
class PreparedImages:
    pixel_values: tensorplay.Tensor
    grid_thw: tensorplay.Tensor
    feature_counts: tuple[int, ...]
    content_digests: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class PreparedAudios:
    input_features: tensorplay.Tensor
    input_features_mask: tensorplay.Tensor
    feature_lengths: tuple[int, ...]
    output_lengths: tuple[int, ...]


def _digest(payload: dict[str, object]) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


class Gemma4ImageProcessor:
    """Aspect-preserving resize, patchify and rescale (no normalisation)."""

    max_encoded_bytes = 32 * 1024**2
    max_images = 16

    def __init__(self, config: Gemma4VisionConfig) -> None:
        self.config = config
        self.digest = _digest(
            {
                "abi": "megalm.gemma4.image_processor.v1",
                "patch_size": config.patch_size,
                "pooling_kernel_size": config.pooling_kernel_size,
                "max_soft_tokens": config.default_output_length,
                "resample": "bicubic",
                "normalization": "rescale_255",
                "max_encoded_bytes": self.max_encoded_bytes,
            }
        )

    def prepare(self, images: tuple[ImageInput, ...]) -> PreparedImages:
        if not images or len(images) > self.max_images:
            raise Gemma4MediaError(
                f"image count must be in [1, {self.max_images}]"
            )
        patches: list[tensorplay.Tensor] = []
        grids: list[tuple[int, int, int]] = []
        counts: list[int] = []
        for image_input in images:
            patch, grid, count = self._prepare_one(image_input)
            patches.append(patch)
            grids.append(grid)
            counts.append(count)
        return PreparedImages(
            pixel_values=tensorplay.cat(patches, dim=0),
            grid_thw=tensorplay.tensor(grids, dtype=tensorplay.long),
            feature_counts=tuple(counts),
            content_digests=tuple(image.content_digest for image in images),
        )

    def _prepare_one(
        self, image_input: ImageInput
    ) -> tuple[tensorplay.Tensor, tuple[int, int, int], int]:
        if len(image_input.payload) > self.max_encoded_bytes:
            raise Gemma4MediaError("encoded image exceeds the byte limit")
        try:
            with Image.open(io.BytesIO(image_input.payload)) as source:
                source.load()
                image = source.convert("RGB")
        except (UnidentifiedImageError, OSError, ValueError) as exc:
            raise Gemma4MediaError("cannot decode image payload") from exc
        config = self.config
        geometry = image_geometry(
            image.height,
            image.width,
            patch_size=config.patch_size,
            max_soft_tokens=config.default_output_length,
            pooling_kernel_size=config.pooling_kernel_size,
        )
        resized = image.resize(
            (geometry.target_w, geometry.target_h),
            resample=Image.Resampling.BICUBIC,
        )
        pixels = tensorplay.frombuffer(
            bytearray(resized.tobytes()), dtype=tensorplay.uint8
        ).clone()
        pixels = (
            pixels.view(geometry.target_h, geometry.target_w, 3)
            .permute(2, 0, 1)
            .float()
            .div_(255.0)
        )
        patch_size = config.patch_size
        patch = (
            pixels.view(3, geometry.patch_h, patch_size, geometry.patch_w, patch_size)
            .permute(1, 3, 2, 4, 0)
            .reshape(geometry.patch_h * geometry.patch_w, patch_size * patch_size * 3)
        )
        return (
            patch.contiguous(),
            (1, geometry.patch_h, geometry.patch_w),
            geometry.num_soft_tokens,
        )


class Gemma4AudioProcessor:
    """Semicausal 16 kHz log-mel frontend matching Gemma4AudioFeatureExtractor."""

    max_encoded_bytes = 64 * 1024**2
    maximum_seconds = 300

    def __init__(self, config: Gemma4AudioConfig) -> None:
        self.config = config
        self.frame_length = config.frame_length
        self.hop_length = config.hop_length
        self.fft_length = config.fft_length
        self.sampling_rate = config.sampling_rate
        self.maximum_samples = self.sampling_rate * self.maximum_seconds
        self._mel_filters = _htk_mel_filter_bank(
            frequency_bins=self.fft_length // 2 + 1,
            mel_bins=config.num_mel_bins,
            sampling_rate=self.sampling_rate,
            min_frequency=config.min_frequency,
            max_frequency=config.max_frequency,
        )
        # Periodic Hann window, matching HF ``window_function``.
        self._window = (
            0.5 - 0.5 * np.cos(2 * np.pi * np.arange(self.frame_length) / self.frame_length)
        ).astype(np.float32)
        self.digest = _digest(
            {
                "abi": "megalm.gemma4.audio_processor.v1",
                "sampling_rate": self.sampling_rate,
                "frame_length": self.frame_length,
                "hop_length": self.hop_length,
                "fft_length": self.fft_length,
                "mel_bins": config.num_mel_bins,
                "mel_floor": config.mel_floor,
                "min_frequency": config.min_frequency,
                "max_frequency": config.max_frequency,
                "max_audio_tokens": config.audio_seq_length,
            }
        )

    def prepare(self, audios: Sequence[AudioInput]) -> PreparedAudios:
        ordered = tuple(audios)
        if not ordered:
            raise Gemma4MediaError("audio batch is empty")
        features: list[np.ndarray] = []
        masks: list[np.ndarray] = []
        feature_lengths: list[int] = []
        output_lengths: list[int] = []
        for audio in ordered:
            waveform = self._decode(audio)
            mel, mask = self._log_mel(waveform)
            length = num_audio_soft_tokens(
                waveform.numel(),
                sampling_rate=self.sampling_rate,
                audio_seq_length=self.config.audio_seq_length,
                frame_length=self.frame_length,
                hop_length=self.hop_length,
            )
            # Subsample the frame mask exactly as the encoder's two stride-2
            # convolutions do; the surviving frame count must equal the
            # placeholder count derived independently from sample length.
            surviving = len(mask)
            for _ in range(2):
                surviving = (surviving + 2 - 3) // 2 + 1
            if length <= 0 or length != surviving:
                raise Gemma4MediaError(
                    "audio mel framing disagrees with the encoder token count"
                )
            features.append(mel)
            masks.append(mask)
            feature_lengths.append(len(mask))
            output_lengths.append(length)
        width = max(feature.shape[0] for feature in features)
        padded = np.zeros((len(features), self.config.num_mel_bins, width), dtype=np.float32)
        padded_mask = np.zeros((len(features), width), dtype=bool)
        for index, (feature, mask) in enumerate(zip(features, masks, strict=True)):
            padded[index, :, : feature.shape[0]] = feature.T
            padded_mask[index, : len(mask)] = mask
        return PreparedAudios(
            input_features=tensorplay.from_numpy(padded).contiguous(),
            input_features_mask=tensorplay.from_numpy(padded_mask).contiguous(),
            feature_lengths=tuple(feature_lengths),
            output_lengths=tuple(output_lengths),
        )

    def _decode(self, audio: AudioInput) -> tensorplay.Tensor:
        if len(audio.payload) > self.max_encoded_bytes:
            raise Gemma4MediaError("encoded audio exceeds the byte limit")
        try:
            import soundfile
            values, source_rate = soundfile.read(
                io.BytesIO(audio.payload), dtype="float32", always_2d=True
            )
        except Exception as exc:
            raise Gemma4MediaError(f"cannot decode {audio.mime_type} audio") from exc
        if (
            source_rate <= 0
            or values.ndim != 2
            or values.shape[0] == 0
            or not np.isfinite(values).all()
        ):
            raise Gemma4MediaError("decoded audio geometry is invalid")
        waveform = tensorplay.from_numpy(values).mean(dim=1)
        if source_rate != self.sampling_rate:
            try:
                waveform = tensorplay.audio.functional.resample(
                    waveform, source_rate, self.sampling_rate
                )
            except Exception as exc:
                raise Gemma4MediaError("audio resampling failed") from exc
        waveform = waveform.to(dtype=tensorplay.float32).contiguous()
        if (
            waveform.numel() == 0
            or waveform.numel() > self.maximum_samples
            or not bool(tensorplay.isfinite(waveform).all().item())
        ):
            raise Gemma4MediaError("decoded audio duration is invalid")
        return waveform

    def _log_mel(self, waveform: tensorplay.Tensor) -> tuple[np.ndarray, np.ndarray]:
        samples = waveform.numpy()
        pad_left = self.frame_length // 2
        padded = np.pad(samples, (pad_left, 0))
        frame_size = self.frame_length + 1
        count = (len(padded) - frame_size) // self.hop_length + 1
        if count <= 0:
            raise Gemma4MediaError("audio is too short for the Gemma-4 encoder")
        frames = np.stack(
            [
                padded[offset : offset + frame_size]
                for offset in range(0, count * self.hop_length, self.hop_length)
            ]
        )[:count, :-1]
        frames = frames * self._window
        spectrum = np.abs(np.fft.rfft(frames, n=self.fft_length, axis=-1))
        mel = spectrum @ self._mel_filters
        log_mel = np.log(mel + self.config.mel_floor).astype(np.float32)
        ends = np.arange(count) * self.hop_length + frame_size - 1
        mask = ends < (pad_left + len(samples))
        if not bool(mask.any()):
            raise Gemma4MediaError("audio has no valid mel frames")
        return log_mel, mask


def _htk_mel_filter_bank(
    *,
    frequency_bins: int,
    mel_bins: int,
    sampling_rate: int,
    min_frequency: float,
    max_frequency: float,
) -> np.ndarray:
    """Unnormalised HTK mel filter bank (HF ``mel_filter_bank`` equivalent)."""

    def to_mel(frequency: float) -> float:
        return 2595.0 * math.log10(1.0 + frequency / 700.0)

    def from_mel(mel: float) -> float:
        return 700.0 * (10 ** (mel / 2595.0) - 1.0)

    mel_min, mel_max = to_mel(min_frequency), to_mel(max_frequency)
    points = np.linspace(mel_min, mel_max, mel_bins + 2)
    frequencies = from_mel(points)
    bin_frequencies = (
        np.arange(frequency_bins) * (sampling_rate / 2) / (frequency_bins - 1)
    )
    filters = np.zeros((frequency_bins, mel_bins), dtype=np.float32)
    for index in range(mel_bins):
        left, center, right = frequencies[index : index + 3]
        below = (bin_frequencies >= left) & (bin_frequencies <= center)
        above = (bin_frequencies > center) & (bin_frequencies <= right)
        filters[below, index] = (bin_frequencies[below] - left) / (center - left)
        filters[above, index] = (right - bin_frequencies[above]) / (right - center)
    return filters


__all__ = [
    "Gemma4AudioProcessor",
    "Gemma4ImageProcessor",
    "Gemma4MediaError",
    "PreparedAudios",
    "PreparedImages",
]
