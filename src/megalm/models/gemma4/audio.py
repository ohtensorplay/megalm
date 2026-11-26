"""Gemma-4 audio (Conformer-style) tower.

A compact but faithful implementation of the reference audio encoder: two
strided Conv2d subsampling layers, relative sinusoidal position encoding,
chunked bidirectional attention with a logit cap, feed-forward / depthwise
conv / feed-forward blocks with residual weighting, and an output projection.
"""

from __future__ import annotations

import math

import tensorplay
import tensorplay.nn as nn
import tensorplay.nn.functional as F

from megalm.kernels.bindings import KernelBindings
from megalm.models.components.parameters import frozen_parameter
from megalm.models.gemma4.config import Gemma4AudioConfig
from megalm.models.components.norms import RMSNorm as Gemma4RMSNorm


def _glu(value: tensorplay.Tensor) -> tensorplay.Tensor:
    first, second = value.chunk(2, dim=-1)
    return first * F.silu(second)


class AudioFeedForward(nn.Module):
    def __init__(
        self,
        config: Gemma4AudioConfig,
        *,
        device: tensorplay.device,
        dtype: tensorplay.dtype,
        kernel_bindings: KernelBindings,
    ) -> None:
        super().__init__()
        hidden = config.hidden_size
        self.layer1 = frozen_parameter((4 * hidden, hidden), device=device, dtype=dtype)
        self.layer2 = frozen_parameter((hidden, 4 * hidden), device=device, dtype=dtype)
        self.pre_norm = Gemma4RMSNorm(
            hidden, config.rms_norm_eps,
            device=device, dtype=dtype, kernel_bindings=kernel_bindings,
        )
        self.post_norm = Gemma4RMSNorm(
            hidden, config.rms_norm_eps,
            device=device, dtype=dtype, kernel_bindings=kernel_bindings,
        )
        self.scale = config.residual_weight
        self.clip = config.gradient_clipping

    def forward(self, value: tensorplay.Tensor) -> tensorplay.Tensor:
        residual = value
        value = value.clamp(-self.clip, self.clip)
        value = self.pre_norm(value)
        value = F.linear(value, self.layer1)
        value = F.silu(value)
        value = F.linear(value, self.layer2).clamp(-self.clip, self.clip)
        value = self.post_norm(value) * self.scale
        return value + residual


class AudioLightConv(nn.Module):
    def __init__(
        self,
        config: Gemma4AudioConfig,
        *,
        device: tensorplay.device,
        dtype: tensorplay.dtype,
        kernel_bindings: KernelBindings,
    ) -> None:
        super().__init__()
        hidden = config.hidden_size
        self.linear_start = frozen_parameter((2 * hidden, hidden), device=device, dtype=dtype)
        self.linear_end = frozen_parameter((hidden, hidden), device=device, dtype=dtype)
        self.conv = nn.Conv1d(
            hidden, hidden, kernel_size=config.conv_kernel_size,
            groups=hidden, bias=False, device=device, dtype=dtype,
        )
        self.pre_norm = Gemma4RMSNorm(
            hidden, config.rms_norm_eps,
            device=device, dtype=dtype, kernel_bindings=kernel_bindings,
        )
        self.conv_norm = Gemma4RMSNorm(
            hidden, config.rms_norm_eps,
            device=device, dtype=dtype, kernel_bindings=kernel_bindings,
        )
        self.clip = config.gradient_clipping
        self.left_pad = config.conv_kernel_size - 1

    def forward(self, value: tensorplay.Tensor) -> tensorplay.Tensor:
        residual = value
        value = self.pre_norm(value)
        value = _glu(F.linear(value, self.linear_start))
        value = value.transpose(1, 2)
        if self.left_pad:
            value = F.pad(value, (self.left_pad, 0))
        value = self.conv(value).transpose(1, 2)
        value = value.clamp(-self.clip, self.clip)
        value = F.silu(self.conv_norm(value))
        return F.linear(value, self.linear_end) + residual


class AudioRelPositionalEncoding(nn.Module):
    """Sinusoidal relative position table for the blocked audio attention."""

    def __init__(
        self,
        config: Gemma4AudioConfig,
        *,
        device: tensorplay.device,
        dtype: tensorplay.dtype,
    ) -> None:
        super().__init__()
        self.context_size = (
            config.attention_chunk_size
            + config.attention_context_left
            - 1
            + config.attention_context_right
        )
        num_timescales = config.hidden_size // 2
        increment = math.log(10_000.0) / max(num_timescales - 1, 1)
        inv = tensorplay.exp(
            -tensorplay.arange(num_timescales, device=device, dtype=tensorplay.float32)
            * increment
        )
        self.register_buffer("inv_timescales", inv[None, None], persistent=False)
        self.dtype = dtype

    def forward(self) -> tensorplay.Tensor:
        positions = tensorplay.arange(
            self.context_size // 2, -1, -1,
            device=self.inv_timescales.device, dtype=tensorplay.float32,
        )[:, None]
        scaled = positions * self.inv_timescales
        return tensorplay.cat((scaled.sin(), scaled.cos()), dim=-1).to(self.dtype)


class AudioAttention(nn.Module):
    """Blocked local attention with learned relative-key bias and soft-cap."""

    def __init__(
        self,
        config: Gemma4AudioConfig,
        *,
        device: tensorplay.device,
        dtype: tensorplay.dtype,
        kernel_bindings: KernelBindings,
    ) -> None:
        super().__init__()
        hidden = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = hidden // self.num_heads
        self.chunk_size = config.attention_chunk_size
        self.max_past = config.attention_context_left - 1
        self.max_future = config.attention_context_right
        self.context_size = (
            self.chunk_size + self.max_past + self.max_future
        )
        self.q_scale = (self.head_dim**-0.5) / math.log(2)
        self.k_scale = math.log(1 + math.e) / math.log(2)
        for name in ("q_proj", "k_proj", "v_proj"):
            setattr(
                self, name,
                frozen_parameter((hidden, hidden), device=device, dtype=dtype),
            )
        self.post = frozen_parameter((hidden, hidden), device=device, dtype=dtype)
        self.relative_k = frozen_parameter(
            (hidden, hidden), device=device, dtype=dtype
        )
        self.per_dim_scale = frozen_parameter(
            (self.head_dim,), device=device, dtype=dtype
        )
        self.config = config

    @staticmethod
    def _to_blocks(value: tensorplay.Tensor, chunk: int) -> tensorplay.Tensor:
        batch, tokens, heads, dim = value.shape
        count = (tokens + chunk - 1) // chunk
        pad = count * chunk - tokens
        if pad:
            value = tensorplay.nn.functional.pad(value, (0, 0, 0, 0, 0, pad))
        return value.reshape(batch, count, chunk, heads, dim)

    def _context(self, value: tensorplay.Tensor) -> tensorplay.Tensor:
        batch, tokens, heads, dim = value.shape
        value = tensorplay.nn.functional.pad(
            value,
            (0, 0, 0, 0, self.max_past, self.max_future + self.chunk_size - 1),
        )
        return value.unfold(1, self.context_size, self.chunk_size).movedim(-1, 2)

    @staticmethod
    def _rel_shift(value: tensorplay.Tensor, context: int) -> tensorplay.Tensor:
        batch, heads, blocks, block_size, positions = value.shape
        pad = context + 1 - positions
        if pad > 0:
            value = tensorplay.nn.functional.pad(value, (0, pad))
        value = value.reshape(batch, heads, blocks, block_size * (context + 1))
        return value[..., : block_size * context].reshape(
            batch, heads, blocks, block_size, context
        )

    def forward(
        self,
        hidden_states: tensorplay.Tensor,
        *,
        position_embeddings: tensorplay.Tensor,
        attention_mask: tensorplay.Tensor | None = None,
    ) -> tensorplay.Tensor:
        batch, tokens, _ = hidden_states.shape
        heads, dim = self.num_heads, self.head_dim
        shape = (batch, tokens, heads, dim)
        q = F.linear(hidden_states, self.q_proj).float().view(shape)
        k = F.linear(hidden_states, self.k_proj).float().view(shape)
        v = F.linear(hidden_states, self.v_proj).float().view(shape)
        q = q * (self.q_scale * F.softplus(self.per_dim_scale))
        k = k * self.k_scale
        q_blocks = self._to_blocks(q, self.chunk_size)
        k_ctx = self._context(k)
        v_ctx = self._context(v)
        blocks = q_blocks.shape[1]
        queries = q_blocks.permute(0, 3, 1, 2, 4)
        matrix_ac = queries @ k_ctx.permute(0, 3, 1, 4, 2)
        flat = queries.reshape(batch, heads, -1, dim)
        relative_k = F.linear(
            position_embeddings.float(), self.relative_k.float()
        ).view(-1, heads, dim)
        matrix_bd = flat @ relative_k.permute(1, 2, 0)
        matrix_bd = matrix_bd.reshape(
            batch, heads, blocks, self.chunk_size, -1
        )
        matrix_bd = self._rel_shift(matrix_bd, self.context_size)
        weights = matrix_ac + matrix_bd
        cap = self.config.attention_logit_cap
        weights = cap * tensorplay.tanh(weights / cap)
        if attention_mask is not None:
            weights = tensorplay.where(
                attention_mask[None, None, None, :, :],
                weights,
                self.config.attention_invalid_logits_value,
            )
        probabilities = tensorplay.softmax(weights, dim=-1).to(v.dtype)
        attended = probabilities @ v_ctx.permute(0, 3, 1, 2, 4)
        attended = attended.permute(0, 2, 3, 1, 4).reshape(
            batch, blocks * self.chunk_size, -1
        )[:, :tokens]
        return F.linear(attended.to(hidden_states.dtype), self.post)


class AudioLayer(nn.Module):
    def __init__(
        self,
        config: Gemma4AudioConfig,
        *,
        device: tensorplay.device,
        dtype: tensorplay.dtype,
        kernel_bindings: KernelBindings,
    ) -> None:
        super().__init__()
        self.ff1 = AudioFeedForward(
            config, device=device, dtype=dtype, kernel_bindings=kernel_bindings
        )
        self.attn = AudioAttention(
            config, device=device, dtype=dtype, kernel_bindings=kernel_bindings
        )
        self.lconv = AudioLightConv(
            config, device=device, dtype=dtype, kernel_bindings=kernel_bindings
        )
        self.ff2 = AudioFeedForward(
            config, device=device, dtype=dtype, kernel_bindings=kernel_bindings
        )
        self.norm_pre_attn = Gemma4RMSNorm(
            config.hidden_size, config.rms_norm_eps,
            device=device, dtype=dtype, kernel_bindings=kernel_bindings,
        )
        self.norm_post_attn = Gemma4RMSNorm(
            config.hidden_size, config.rms_norm_eps,
            device=device, dtype=dtype, kernel_bindings=kernel_bindings,
        )
        self.norm_out = Gemma4RMSNorm(
            config.hidden_size, config.rms_norm_eps,
            device=device, dtype=dtype, kernel_bindings=kernel_bindings,
        )
        self.clip = config.gradient_clipping

    def forward(
        self,
        value: tensorplay.Tensor,
        *,
        position_embeddings: tensorplay.Tensor,
        attention_mask: tensorplay.Tensor | None = None,
    ) -> tensorplay.Tensor:
        value = self.ff1(value)
        residual = value
        attended = self.attn(
            self.norm_pre_attn(value.clamp(-self.clip, self.clip)),
            position_embeddings=position_embeddings,
            attention_mask=attention_mask,
        )
        value = residual + self.norm_post_attn(attended.clamp(-self.clip, self.clip))
        value = self.lconv(value)
        value = self.ff2(value)
        return self.norm_out(value.clamp(-self.clip, self.clip))


class Gemma4AudioModel(nn.Module):
    def __init__(
        self,
        config: Gemma4AudioConfig,
        *,
        device: tensorplay.device,
        dtype: tensorplay.dtype,
        kernel_bindings: KernelBindings,
    ) -> None:
        super().__init__()
        self.config = config
        channels = config.subsampling_conv_channels
        self.subsample1 = nn.Conv2d(
            1, channels[0], kernel_size=3, stride=2, padding=1,
            bias=False, device=device, dtype=dtype,
        )
        self.subsample2 = nn.Conv2d(
            channels[0], channels[1], kernel_size=3, stride=2, padding=1,
            bias=False, device=device, dtype=dtype,
        )
        self.subsample_norm1 = nn.LayerNorm(
            channels[0], eps=config.rms_norm_eps, bias=False,
            device=device, dtype=dtype,
        )
        self.subsample_norm2 = nn.LayerNorm(
            channels[1], eps=config.rms_norm_eps, bias=False,
            device=device, dtype=dtype,
        )
        freq = math.ceil(math.ceil(config.num_mel_bins / 2) / 2)
        flat_size = channels[1] * freq
        self.input_proj = frozen_parameter(
            (config.hidden_size, flat_size), device=device, dtype=dtype
        )
        self.rel_pos = AudioRelPositionalEncoding(
            config, device=device, dtype=dtype
        )
        self.layers = nn.ModuleList(
            [
                AudioLayer(
                    config, device=device, dtype=dtype,
                    kernel_bindings=kernel_bindings,
                )
                for _ in range(config.num_hidden_layers)
            ]
        )
        self.output_proj = frozen_parameter(
            (config.output_proj_dims, config.hidden_size),
            device=device, dtype=dtype,
        )

    def _subsample(
        self, features: tensorplay.Tensor, mask: tensorplay.Tensor | None
    ) -> tuple[tensorplay.Tensor, tensorplay.Tensor | None]:
        # features: [batch, frames, freq]
        value = features.unsqueeze(1)
        value = F.relu(
            self.subsample_norm1(
                self.subsample1(value).permute(0, 2, 3, 1)
            ).permute(0, 3, 1, 2).contiguous()
        )
        value = F.relu(
            self.subsample_norm2(
                self.subsample2(value).permute(0, 2, 3, 1)
            ).permute(0, 3, 1, 2).contiguous()
        )
        batch, channels, frames, freq = value.shape
        value = value.permute(0, 2, 1, 3).reshape(batch, frames, channels * freq)
        value = F.linear(value, self.input_proj)
        if mask is not None:
            # Two stride-2 subsampling stages reduce the time axis by 4.
            mask = mask[:, ::4]
        return value, mask

    def forward(
        self,
        input_features: tensorplay.Tensor,
        attention_mask: tensorplay.Tensor | None = None,
    ) -> tuple[tensorplay.Tensor, tensorplay.Tensor | None]:
        value, mask = self._subsample(input_features, attention_mask)
        position_embeddings = self.rel_pos()
        for layer in self.layers:
            value = layer(
                value, position_embeddings=position_embeddings,
                attention_mask=None,
            )
        output = F.linear(value, self.output_proj)
        return output, mask


__all__ = ["Gemma4AudioModel"]
