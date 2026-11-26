"""Gemma-4 vision tower and multimodal connector.

The tower follows the reference encoder: a patch projection with additive 2-D
position tables, QK-normalised bidirectional attention with axial RoPE, a
sandwich-norm MLP stack and a position-based pooling head.  The connector is
the shared ``Gemma4MultimodalEmbedder`` used by both the vision and audio
paths.
"""

from __future__ import annotations

import math

import tensorplay
import tensorplay.nn.functional as F
from tensorplay import nn

from megalm.kernels.bindings import KernelBindings
from megalm.models.components.parameters import frozen_parameter
from megalm.models.gemma4.attention import gelu_tanh
from megalm.models.gemma4.config import Gemma4VisionConfig
from megalm.models.components.norms import RMSNorm
from megalm.models.components.norms import RMSNorm as Gemma4RMSNorm


class MultimodalEmbedder(nn.Module):
    """Project tower output into the language model's embedding width."""

    def __init__(
        self,
        *,
        input_size: int,
        output_size: int,
        eps: float,
        device: tensorplay.device,
        dtype: tensorplay.dtype,
        kernel_bindings: KernelBindings,
    ) -> None:
        super().__init__()
        self.norm = RMSNorm(
            input_size, eps, with_scale=False,
            device=device, dtype=dtype, kernel_bindings=kernel_bindings,
        )
        self.projection = frozen_parameter(
            (output_size, input_size), device=device, dtype=dtype
        )

    def forward(self, hidden_states: tensorplay.Tensor) -> tensorplay.Tensor:
        return F.linear(self.norm(hidden_states), self.projection)


class Gemma4VisionPatchEmbedder(nn.Module):
    """Flatten patches, project them and add 2-D position embeddings."""

    def __init__(
        self,
        config: Gemma4VisionConfig,
        *,
        device: tensorplay.device,
        dtype: tensorplay.dtype,
    ) -> None:
        super().__init__()
        self.patch_size = config.patch_size
        self.input_size = 3 * config.patch_size**2
        self.hidden_size = config.hidden_size
        self.input_proj = frozen_parameter(
            (config.hidden_size, self.input_size), device=device, dtype=dtype
        )
        self.position_x = frozen_parameter(
            (config.position_embedding_size, config.hidden_size),
            device=device, dtype=dtype,
        )
        self.position_y = frozen_parameter(
            (config.position_embedding_size, config.hidden_size),
            device=device, dtype=dtype,
        )

    def forward(
        self,
        pixel_values: tensorplay.Tensor,
        pixel_position_ids: tensorplay.Tensor,
    ) -> tensorplay.Tensor:
        # Reference convention: pixel values are centred in [-1, 1].
        values = 2.0 * (pixel_values.float() - 0.5)
        hidden = F.linear(values.to(pixel_values.dtype), self.input_proj)
        ids = pixel_position_ids.long()
        valid = (ids >= 0).all(dim=-1)
        safe = ids.clamp_min(0).clamp_max(self.position_x.shape[0] - 1)
        hidden = (
            hidden
            + self.position_x.index_select(0, safe[:, 0])
            + self.position_y.index_select(0, safe[:, 1])
        )
        return tensorplay.where(valid[:, None], hidden, tensorplay.zeros_like(hidden))


def _vision_rope(
    hidden: tensorplay.Tensor,
    positions: tensorplay.Tensor,
    head_dim: int,
    *,
    rope_theta: float = 100.0,
) -> tensorplay.Tensor:
    """Axial 2-D RoPE with the reference's H-H-W-W frequency layout."""

    spatial = head_dim // 2
    inv_freq = 1.0 / (
        rope_theta
        ** (
            tensorplay.arange(
                0, spatial, 2, dtype=tensorplay.float32, device=positions.device
            )
            / spatial
        )
    )
    coordinates = positions.float()
    pieces: list[tensorplay.Tensor] = []
    for axis in range(coordinates.shape[-1]):
        angles = coordinates[:, axis : axis + 1] * inv_freq
        cos = tensorplay.cat((angles.cos(), angles.cos()), dim=-1).to(hidden.dtype)
        sin = tensorplay.cat((angles.sin(), angles.sin()), dim=-1).to(hidden.dtype)
        while cos.ndim < hidden.ndim:
            cos = cos.unsqueeze(-2)
            sin = sin.unsqueeze(-2)
        part = hidden[..., axis * spatial : (axis + 1) * spatial]
        half = part.shape[-1] // 2
        rotated = tensorplay.cat((-part[..., half:], part[..., :half]), dim=-1)
        pieces.append(part * cos + rotated * sin)
    return tensorplay.cat(pieces, dim=-1)


class _ClippedLinearMixin:
    """Per-linear input/output clamping from checkpoint calibration buffers."""

    def _init_clips(self) -> None:
        self.clip_bounds: dict[str, list[tensorplay.Tensor | None]] = {}

    def _set_clip(self, name: str, kind: str, value: tensorplay.Tensor) -> None:
        index = {
            "input_min": 0,
            "input_max": 1,
            "output_min": 2,
            "output_max": 3,
        }[kind]
        bounds = self.clip_bounds.setdefault(name, [None, None, None, None])
        bounds[index] = value

    def _linear(
        self,
        name: str,
        value: tensorplay.Tensor,
        weight: tensorplay.Tensor,
    ) -> tensorplay.Tensor:
        bounds = self.clip_bounds.get(name)
        if bounds is not None and all(bound is not None for bound in bounds):
            low, high, out_low, out_high = (
                bound.to(device=value.device, dtype=value.dtype)
                for bound in bounds
            )
            value = value.clamp(low, high)
            return F.linear(value, weight).clamp(out_low, out_high)
        return F.linear(value, weight)


class Gemma4VisionAttention(_ClippedLinearMixin, nn.Module):
    """QK-normalised bidirectional attention with 2-D axial RoPE."""

    def __init__(
        self,
        config: Gemma4VisionConfig,
        *,
        device: tensorplay.device,
        dtype: tensorplay.dtype,
        kernel_bindings: KernelBindings,
    ) -> None:
        super().__init__()
        self.config = config
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.head_dim = config.head_dim
        self.query_width = self.num_heads * self.head_dim
        self.kv_width = self.num_kv_heads * self.head_dim
        for name, width in (
            ("q_proj", self.query_width),
            ("k_proj", self.kv_width),
            ("v_proj", self.kv_width),
            ("o_proj", config.hidden_size),
        ):
            in_features = (
                config.hidden_size if name != "o_proj" else self.query_width
            )
            setattr(
                self,
                name,
                frozen_parameter((width, in_features), device=device, dtype=dtype),
            )
        self.q_norm = Gemma4RMSNorm(
            self.head_dim, config.rms_norm_eps,
            device=device, dtype=dtype, kernel_bindings=kernel_bindings,
        )
        self.k_norm = Gemma4RMSNorm(
            self.head_dim, config.rms_norm_eps,
            device=device, dtype=dtype, kernel_bindings=kernel_bindings,
        )
        self.v_norm = RMSNorm(
            self.head_dim, config.rms_norm_eps, with_scale=False,
            device=device, dtype=dtype, kernel_bindings=kernel_bindings,
        )
        self._init_clips()

    def _norm(self, value: tensorplay.Tensor, norm: nn.Module) -> tensorplay.Tensor:
        shape = (*value.shape[:-1], -1, self.head_dim)
        return norm(value.view(*shape)).view(value.shape)

    def forward(
        self,
        hidden_states: tensorplay.Tensor,
        *,
        positions: tensorplay.Tensor,
        attention_mask: tensorplay.Tensor,
    ) -> tensorplay.Tensor:
        tokens = hidden_states.shape[0]
        query = self._linear("self_attn.q_proj", hidden_states, self.q_proj).view(
            tokens, self.num_heads, self.head_dim
        )
        key = self._linear("self_attn.k_proj", hidden_states, self.k_proj).view(
            tokens, self.num_kv_heads, self.head_dim
        )
        value = self._linear("self_attn.v_proj", hidden_states, self.v_proj).view(
            tokens, self.num_kv_heads, self.head_dim
        )
        query = self._norm(query, self.q_norm)
        key = self._norm(key, self.k_norm)
        value = self._norm(value, self.v_norm)
        # Axial RoPE: the two halves of each head rotate with the x and y
        # grid coordinates respectively.
        rope_theta = float(getattr(self.config, "rope_theta", 100.0))
        query = _vision_rope(query, positions, self.head_dim, rope_theta=rope_theta)
        key = _vision_rope(key, positions, self.head_dim, rope_theta=rope_theta)
        repeat = self.num_heads // self.num_kv_heads
        key = key.repeat_interleave(repeat, dim=1)
        value = value.repeat_interleave(repeat, dim=1)
        q = query.float().transpose(0, 1).unsqueeze(0)
        k = key.float().transpose(0, 1).unsqueeze(0)
        v = value.to(hidden_states.dtype).transpose(0, 1).unsqueeze(0)
        scores = tensorplay.matmul(q, k.transpose(-1, -2))
        mask = attention_mask[None, None, None, :]
        scores = scores.masked_fill(~mask, float("-inf"))
        weights = tensorplay.softmax(scores, dim=-1).to(hidden_states.dtype)
        attended = tensorplay.matmul(weights, v).squeeze(0).transpose(0, 1)
        return self._linear(
            "self_attn.o_proj", attended.reshape(tokens, -1), self.o_proj
        )


class Gemma4VisionEncoderLayer(_ClippedLinearMixin, nn.Module):
    def __init__(
        self,
        config: Gemma4VisionConfig,
        *,
        device: tensorplay.device,
        dtype: tensorplay.dtype,
        kernel_bindings: KernelBindings,
    ) -> None:
        super().__init__()
        self.input_norm = Gemma4RMSNorm(
            config.hidden_size, config.rms_norm_eps,
            device=device, dtype=dtype, kernel_bindings=kernel_bindings,
        )
        self.post_attention_norm = Gemma4RMSNorm(
            config.hidden_size, config.rms_norm_eps,
            device=device, dtype=dtype, kernel_bindings=kernel_bindings,
        )
        self.pre_mlp_norm = Gemma4RMSNorm(
            config.hidden_size, config.rms_norm_eps,
            device=device, dtype=dtype, kernel_bindings=kernel_bindings,
        )
        self.post_mlp_norm = Gemma4RMSNorm(
            config.hidden_size, config.rms_norm_eps,
            device=device, dtype=dtype, kernel_bindings=kernel_bindings,
        )
        self.self_attn = Gemma4VisionAttention(
            config, device=device, dtype=dtype, kernel_bindings=kernel_bindings
        )
        # Separate gate/up weights (rather than a merged gate_up_weight) so
        # each keeps its own checkpoint input/output clipping bounds.
        self.gate_weight = frozen_parameter(
            (config.intermediate_size, config.hidden_size),
            device=device, dtype=dtype,
        )
        self.up_weight = frozen_parameter(
            (config.intermediate_size, config.hidden_size),
            device=device, dtype=dtype,
        )
        self.down_weight = frozen_parameter(
            (config.hidden_size, config.intermediate_size),
            device=device, dtype=dtype,
        )
        self._init_clips()

    def forward(
        self,
        hidden_states: tensorplay.Tensor,
        *,
        positions: tensorplay.Tensor,
        attention_mask: tensorplay.Tensor,
    ) -> tensorplay.Tensor:
        residual = hidden_states
        hidden_states = self.input_norm(hidden_states)
        hidden_states = self.self_attn(
            hidden_states, positions=positions, attention_mask=attention_mask
        )
        hidden_states = self.post_attention_norm(hidden_states) + residual
        residual = hidden_states
        hidden_states = self.pre_mlp_norm(hidden_states)
        gate = self._linear("mlp.gate_proj", hidden_states, self.gate_weight)
        up = self._linear("mlp.up_proj", hidden_states, self.up_weight)
        hidden_states = self._linear(
            "mlp.down_proj", gelu_tanh(gate) * up, self.down_weight
        )
        return self.post_mlp_norm(hidden_states) + residual


class Gemma4VisionModel(nn.Module):
    """Full vision tower: patch embed, encoder stack and pooled features."""

    def __init__(
        self,
        config: Gemma4VisionConfig,
        *,
        device: tensorplay.device,
        dtype: tensorplay.dtype,
        kernel_bindings: KernelBindings,
    ) -> None:
        super().__init__()
        self.config = config
        self.patch_embedder = Gemma4VisionPatchEmbedder(
            config, device=device, dtype=dtype
        )
        self.layers = nn.ModuleList(
            [
                Gemma4VisionEncoderLayer(
                    config, device=device, dtype=dtype,
                    kernel_bindings=kernel_bindings,
                )
                for _ in range(config.num_hidden_layers)
            ]
        )

    def forward(
        self,
        pixel_values: tensorplay.Tensor,
        pixel_position_ids: tensorplay.Tensor,
        output_length: int | None = None,
    ) -> tuple[tensorplay.Tensor, tensorplay.Tensor]:
        hidden = self.patch_embedder(pixel_values, pixel_position_ids)
        valid = (pixel_position_ids.long() >= 0).all(dim=-1)
        for layer in self.layers:
            hidden = layer(
                hidden, positions=pixel_position_ids, attention_mask=valid
            )
        kernel = self.config.pooling_kernel_size
        positions = pixel_position_ids.long().clamp_min(0)
        max_x = int(positions[:, 0].max().item()) + 1
        length = output_length or hidden.shape[0] // kernel**2
        if length <= 0 or length > hidden.shape[0]:
            raise ValueError("invalid vision pooling output length")
        hidden = tensorplay.where(valid[:, None], hidden, tensorplay.zeros_like(hidden))
        # One row per k x k spatial cluster, indexed by (x // k, y // k).
        # The x stride is ceil(max_x / k), matching the HF reference pooler.
        cluster = positions[:, 0] // kernel + (max_x // kernel) * (
            positions[:, 1] // kernel
        )
        weights = tensorplay.nn.functional.one_hot(cluster, length).to(hidden.dtype) / (
            kernel * kernel
        )
        pooled = weights.transpose(0, 1) @ hidden
        present = (weights.sum(dim=0) > 0)
        pooled = pooled * (hidden.shape[-1] ** 0.5)
        return pooled, present


__all__ = [
    "Gemma4VisionModel",
    "MultimodalEmbedder",
]
