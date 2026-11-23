from __future__ import annotations

import hashlib
import io
import json
import math
from dataclasses import dataclass
from typing import Sequence

import numpy as np
import tensorplay
import tensorplay.nn.functional as F

from megalm.contracts.request import AudioInput
from megalm.models.qwen2_5_omni.config import OmniAudioConfig


class OmniAudioProcessingError(ValueError):
    """Raised when encoded audio cannot enter the native Omni frontend."""


@dataclass(frozen=True, slots=True)
class PreparedOmniAudios:
    input_features: tensorplay.Tensor
    feature_lengths: tuple[int, ...]
    output_lengths: tuple[int, ...]
    durations_seconds: tuple[float, ...]


class Qwen25OmniAudioProcessor:
    """Whisper-compatible 16 kHz, 128-bin log-mel frontend."""

    sample_rate = 16_000
    fft_size = 400
    hop_length = 160
    maximum_samples = 300 * sample_rate

    def __init__(self, config: OmniAudioConfig) -> None:
        if config.num_mel_bins != 128:
            raise OmniAudioProcessingError(
                "Qwen2.5-Omni requires 128 mel bins"
            )
        self.config = config
        self._mel_filters = tensorplay.from_numpy(
            _mel_filter_bank(
                frequency_bins=1 + self.fft_size // 2,
                mel_bins=config.num_mel_bins,
                sample_rate=self.sample_rate,
            )
        )
        self._window = tensorplay.hann_window(
            self.fft_size,
            periodic=True,
            dtype=tensorplay.float32,
        )
        contract = {
            "implementation": "megalm-whisper-logmel-v1",
            "sample_rate": self.sample_rate,
            "fft_size": self.fft_size,
            "hop_length": self.hop_length,
            "mel_bins": config.num_mel_bins,
            "maximum_samples": self.maximum_samples,
            "mel_scale": "slaney",
            "mel_norm": "slaney",
        }
        self.digest = hashlib.sha256(
            json.dumps(
                contract,
                sort_keys=True,
                separators=(",", ":"),
            ).encode()
        ).hexdigest()

    def prepare(
        self,
        audios: Sequence[AudioInput],
    ) -> PreparedOmniAudios:
        ordered = tuple(audios)
        if not ordered:
            raise OmniAudioProcessingError("audio batch is empty")
        features: list[tensorplay.Tensor] = []
        feature_lengths: list[int] = []
        output_lengths: list[int] = []
        durations: list[float] = []
        for audio in ordered:
            waveform = self._decode(audio)
            length = (waveform.numel() - 1) // self.hop_length + 1
            after_conv = (length - 1) // 2 + 1
            output_length = (after_conv - 2) // 2 + 1
            if output_length <= 0:
                raise OmniAudioProcessingError(
                    "audio is too short for the Omni encoder"
                )
            features.append(self._log_mel(waveform, length))
            feature_lengths.append(length)
            output_lengths.append(output_length)
            durations.append(waveform.numel() / self.sample_rate)
        return PreparedOmniAudios(
            input_features=tensorplay.cat(features, dim=1).contiguous(),
            feature_lengths=tuple(feature_lengths),
            output_lengths=tuple(output_lengths),
            durations_seconds=tuple(durations),
        )

    def _decode(self, audio: AudioInput) -> tensorplay.Tensor:
        try:
            import soundfile

            values, source_rate = soundfile.read(
                io.BytesIO(audio.payload),
                dtype="float32",
                always_2d=True,
            )
        except Exception as exc:
            raise OmniAudioProcessingError(
                f"cannot decode {audio.mime_type} audio"
            ) from exc
        if (
            source_rate <= 0
            or values.ndim != 2
            or values.shape[0] == 0
            or not np.isfinite(values).all()
        ):
            raise OmniAudioProcessingError(
                "decoded audio geometry is invalid"
            )
        waveform = tensorplay.from_numpy(values).mean(dim=1)
        if source_rate != self.sample_rate:
            try:
                waveform = tensorplay.audio.functional.resample(
                    waveform,
                    source_rate,
                    self.sample_rate,
                )
            except Exception as exc:
                raise OmniAudioProcessingError(
                    "audio resampling failed"
                ) from exc
        waveform = waveform.to(dtype=tensorplay.float32).contiguous()
        if (
            waveform.numel() == 0
            or waveform.numel() > self.maximum_samples
            or not bool(tensorplay.isfinite(waveform).all().item())
        ):
            raise OmniAudioProcessingError(
                "decoded audio duration is invalid"
            )
        return waveform

    def _log_mel(
        self,
        waveform: tensorplay.Tensor,
        feature_length: int,
    ) -> tensorplay.Tensor:
        padded = F.pad(waveform, (0, self.fft_size))
        spectrum = tensorplay.stft(
            padded,
            self.fft_size,
            self.hop_length,
            window=self._window,
            center=True,
            pad_mode="reflect",
            return_complex=True,
        )
        magnitudes = spectrum[:, :-1].abs().square()
        mel = self._mel_filters.transpose(0, 1) @ magnitudes
        log_mel = mel.clamp_min(1e-10).log10()
        log_mel = tensorplay.maximum(log_mel, log_mel.max() - 8.0)
        normalized = (log_mel + 4.0) / 4.0
        if normalized.shape[1] < feature_length:
            raise OmniAudioProcessingError(
                "log-mel frontend produced too few frames"
            )
        return normalized[:, :feature_length].contiguous()


def _mel_filter_bank(
    *,
    frequency_bins: int,
    mel_bins: int,
    sample_rate: int,
) -> np.ndarray:
    mel_min = _hertz_to_mel(np.array(0.0))
    mel_max = _hertz_to_mel(np.array(sample_rate / 2))
    mel_points = np.linspace(mel_min, mel_max, mel_bins + 2)
    filter_points = _mel_to_hertz(mel_points)
    fft_points = np.linspace(0, sample_rate // 2, frequency_bins)
    differences = np.diff(filter_points)
    slopes = filter_points[None, :] - fft_points[:, None]
    down = -slopes[:, :-2] / differences[:-1]
    up = slopes[:, 2:] / differences[1:]
    filters = np.maximum(0.0, np.minimum(down, up))
    normalization = 2.0 / (
        filter_points[2:] - filter_points[:-2]
    )
    filters *= normalization[None, :]
    return filters.astype(np.float32)


def _hertz_to_mel(frequency: np.ndarray) -> np.ndarray:
    frequency = np.asarray(frequency, dtype=np.float64)
    mels = 3.0 * frequency / 200.0
    logarithmic = frequency >= 1000.0
    return np.where(
        logarithmic,
        15.0
        + np.log(np.maximum(frequency, 1000.0) / 1000.0)
        * (27.0 / math.log(6.4)),
        mels,
    )


def _mel_to_hertz(mels: np.ndarray) -> np.ndarray:
    mels = np.asarray(mels, dtype=np.float64)
    linear = 200.0 * mels / 3.0
    logarithmic = mels >= 15.0
    return np.where(
        logarithmic,
        1000.0
        * np.exp((mels - 15.0) * (math.log(6.4) / 27.0)),
        linear,
    )


__all__ = [
    "OmniAudioProcessingError",
    "PreparedOmniAudios",
    "Qwen25OmniAudioProcessor",
]
