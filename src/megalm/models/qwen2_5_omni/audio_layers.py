from __future__ import annotations

import tensorplay
import tensorplay.nn as nn
import tensorplay.nn.functional as F

from megalm.kernels.vision_attention import (
    GroupedVarlenAttentionLayout,
    grouped_varlen_attention,
)
from megalm.models.components import (
    FrozenLayerNorm,
    FrozenLinear,
    frozen_parameter,
)
from megalm.models.qwen2_5_omni.config import OmniAudioConfig


class PackedAudioAttention(nn.Module):
    """Official Qwen audio attention with one packed QKV projection."""

    def __init__(
        self,
        config: OmniAudioConfig,
        *,
        device: tensorplay.device,
        dtype: tensorplay.dtype,
    ) -> None:
        super().__init__()
        width = config.hidden_size
        self.num_heads = config.num_heads
        self.head_dim = width // config.num_heads
        self.qkv_weight = frozen_parameter(
            (3 * width, width),
            device=device,
            dtype=dtype,
        )
        self.qkv_bias = frozen_parameter(
            (3 * width,),
            device=device,
            dtype=dtype,
        )
        with tensorplay.no_grad():
            self.qkv_bias.zero_()
        self.out_proj = FrozenLinear(
            width,
            width,
            bias=True,
            device=device,
            dtype=dtype,
        )

    def forward(
        self,
        hidden_states: tensorplay.Tensor,
        *,
        attention_layout: GroupedVarlenAttentionLayout,
    ) -> tensorplay.Tensor:
        token_count = hidden_states.shape[0]
        query, key, value = (
            F.linear(
                hidden_states,
                self.qkv_weight,
                self.qkv_bias,
            )
            .view(token_count, 3, self.num_heads, self.head_dim)
            .unbind(dim=1)
        )
        attended = grouped_varlen_attention(
            query,
            key,
            value,
            attention_layout,
            scale=self.head_dim**-0.5,
        )
        return self.out_proj(attended.reshape(token_count, -1))


class OmniAudioEncoderLayer(nn.Module):
    def __init__(
        self,
        config: OmniAudioConfig,
        *,
        device: tensorplay.device,
        dtype: tensorplay.dtype,
    ) -> None:
        super().__init__()
        width = config.hidden_size
        self.self_attn = PackedAudioAttention(
            config,
            device=device,
            dtype=dtype,
        )
        self.self_attn_layer_norm = FrozenLayerNorm(
            width,
            eps=config.layer_norm_eps,
            device=device,
            dtype=dtype,
        )
        self.fc1 = FrozenLinear(
            width,
            config.intermediate_size,
            bias=True,
            device=device,
            dtype=dtype,
        )
        self.fc2 = FrozenLinear(
            config.intermediate_size,
            width,
            bias=True,
            device=device,
            dtype=dtype,
        )
        self.final_layer_norm = FrozenLayerNorm(
            width,
            eps=config.layer_norm_eps,
            device=device,
            dtype=dtype,
        )

    def forward(
        self,
        hidden_states: tensorplay.Tensor,
        *,
        attention_layout: GroupedVarlenAttentionLayout,
    ) -> tensorplay.Tensor:
        hidden_states = hidden_states + self.self_attn(
            self.self_attn_layer_norm(hidden_states),
            attention_layout=attention_layout,
        )
        hidden_states = hidden_states + self.fc2(
            F.gelu(
                self.fc1(self.final_layer_norm(hidden_states)),
                approximate="none",
            )
        )
        if hidden_states.dtype == tensorplay.float16:
            limit = tensorplay.finfo(tensorplay.float16).max - 1000
            hidden_states = hidden_states.clamp(-limit, limit)
        return hidden_states


__all__ = ["OmniAudioEncoderLayer", "PackedAudioAttention"]
