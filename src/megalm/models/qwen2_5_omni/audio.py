from __future__ import annotations

import math
from typing import Sequence

import tensorplay
import tensorplay.nn as nn
import tensorplay.nn.functional as F

from megalm.kernels.vision_attention import (
    build_grouped_varlen_attention_layout,
)
from megalm.models.components import (
    FrozenConv1d,
    FrozenLayerNorm,
    FrozenLinear,
    frozen_parameter,
)
from megalm.models.qwen2_5_omni.audio_layers import (
    OmniAudioEncoderLayer,
)
from megalm.models.qwen2_5_omni.config import OmniAudioConfig


class Qwen25OmniAudioEncoder(nn.Module):
    """Native packed Qwen2.5-Omni audio encoder."""

    def __init__(
        self,
        config: OmniAudioConfig,
        *,
        device: tensorplay.device | str,
        dtype: tensorplay.dtype,
    ) -> None:
        super().__init__()
        self.config = config
        self.device = tensorplay.device(device)
        self.dtype = dtype
        self.conv1 = FrozenConv1d(
            config.num_mel_bins,
            config.hidden_size,
            3,
            padding=1,
            device=self.device,
            dtype=dtype,
        )
        self.conv2 = FrozenConv1d(
            config.hidden_size,
            config.hidden_size,
            3,
            stride=2,
            padding=1,
            device=self.device,
            dtype=dtype,
        )
        self.audio_bos_eos_token = nn.Module()
        self.audio_bos_eos_token.register_parameter(
            "weight",
            frozen_parameter(
                (2, config.output_hidden_size),
                device=self.device,
                dtype=dtype,
            ),
        )
        self.layers = nn.ModuleList(
            OmniAudioEncoderLayer(
                config,
                device=self.device,
                dtype=dtype,
            )
            for _ in range(config.num_layers)
        )
        self.ln_post = FrozenLayerNorm(
            config.hidden_size,
            eps=config.layer_norm_eps,
            device=self.device,
            dtype=dtype,
        )
        self.proj = FrozenLinear(
            config.hidden_size,
            config.output_hidden_size,
            bias=True,
            device=self.device,
            dtype=dtype,
        )
        self.register_buffer(
            "positional_embedding",
            _sinusoids(
                config.max_source_positions,
                config.hidden_size,
                device=self.device,
            ),
            persistent=False,
        )

    @staticmethod
    def output_lengths(
        feature_lengths: tensorplay.Tensor,
    ) -> tuple[tensorplay.Tensor, tensorplay.Tensor]:
        after_conv = (feature_lengths - 1) // 2 + 1
        output = (after_conv - 2) // 2 + 1
        return after_conv, output

    def forward(
        self,
        input_features: tensorplay.Tensor,
        feature_lengths: Sequence[int] | tensorplay.Tensor,
    ) -> tuple[tensorplay.Tensor, tuple[int, ...]]:
        lengths = tensorplay.as_tensor(
            feature_lengths,
            dtype=tensorplay.long,
            device=self.device,
        )
        if (
            input_features.ndim != 2
            or input_features.shape[0] != self.config.num_mel_bins
            or lengths.ndim != 1
            or lengths.numel() == 0
            or bool((lengths < 3).any().item())
            or int(lengths.sum().item()) != input_features.shape[1]
        ):
            raise ValueError("audio features or lengths are invalid")
        after_conv, output_lengths = self.output_lengths(lengths)
        if bool((output_lengths <= 0).any().item()):
            raise ValueError("audio is too short for the encoder")

        chunks = _chunk_lengths(
            lengths,
            maximum=self.config.window_size * 2,
        )
        features = input_features.to(
            device=self.device,
            dtype=self.dtype,
        )
        split = features.split(chunks.tolist(), dim=1)
        padded = tensorplay.nn.utils.rnn.pad_sequence(
            tuple(value.transpose(0, 1) for value in split),
            batch_first=True,
        ).transpose(1, 2)
        positions = tensorplay.arange(
            padded.shape[2],
            device=self.device,
        )
        valid = positions[None, :] < chunks[:, None]
        hidden_states = F.gelu(
            self.conv1(padded),
            approximate="none",
        )
        hidden_states = hidden_states * valid[:, None, :]
        hidden_states = F.gelu(
            self.conv2(hidden_states),
            approximate="none",
        ).transpose(1, 2)

        chunk_after_conv = (chunks - 1) // 2 + 1
        after_positions = tensorplay.arange(
            hidden_states.shape[1],
            device=self.device,
        )
        after_valid = (
            after_positions[None, :] < chunk_after_conv[:, None]
        )
        hidden_states = hidden_states + self.positional_embedding[
            : hidden_states.shape[1]
        ].to(dtype=self.dtype)[None, :, :]
        packed = hidden_states[after_valid]
        cu_seqlens = tensorplay.cat(
            (
                tensorplay.zeros(1, dtype=tensorplay.int32, device=self.device),
                chunk_after_conv.cumsum(0).to(tensorplay.int32),
            )
        )
        attention_layout = build_grouped_varlen_attention_layout(
            cu_seqlens
        )
        for layer in self.layers:
            packed = layer(
                packed,
                attention_layout=attention_layout,
            )

        encoded: list[tensorplay.Tensor] = []
        for states in packed.split(after_conv.tolist(), dim=0):
            pooled = F.avg_pool1d(
                states.transpose(0, 1).unsqueeze(0),
                kernel_size=2,
                stride=2,
            ).squeeze(0).transpose(0, 1)
            encoded.append(self.proj(self.ln_post(pooled)))
        output = tensorplay.cat(encoded, dim=0).contiguous()
        counts = tuple(int(value) for value in output_lengths.tolist())
        if output.shape[0] != sum(counts):
            raise ValueError("audio encoder output length is inconsistent")
        return output, counts


def _chunk_lengths(
    feature_lengths: tensorplay.Tensor,
    *,
    maximum: int,
) -> tensorplay.Tensor:
    values: list[int] = []
    for length in feature_lengths.tolist():
        whole, tail = divmod(int(length), maximum)
        values.extend([maximum] * whole)
        if tail:
            values.append(tail)
    return tensorplay.tensor(
        values,
        dtype=tensorplay.long,
        device=feature_lengths.device,
    )


def _sinusoids(
    length: int,
    channels: int,
    *,
    device: tensorplay.device,
) -> tensorplay.Tensor:
    if channels % 2:
        raise ValueError("audio position width must be even")
    increment = math.log(10_000.0) / (channels // 2 - 1)
    inverse = tensorplay.exp(
        -increment
        * tensorplay.arange(channels // 2, device=device, dtype=tensorplay.float32)
    )
    scaled = (
        tensorplay.arange(length, device=device, dtype=tensorplay.float32)[:, None]
        * inverse[None, :]
    )
    return tensorplay.cat((scaled.sin(), scaled.cos()), dim=1)


__all__ = ["Qwen25OmniAudioEncoder"]
