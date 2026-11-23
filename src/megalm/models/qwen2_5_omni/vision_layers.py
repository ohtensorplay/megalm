from __future__ import annotations

import tensorplay
import tensorplay.nn as nn
import tensorplay.nn.functional as F

from megalm.kernels.bindings import KernelBindings
from megalm.kernels.vision_attention import (
    GroupedVarlenAttentionLayout,
)
from megalm.models.components import (
    PackedSwiGLU,
    PackedVisionAttention,
    frozen_parameter,
)
from megalm.models.qwen2_5_omni.config import OmniVisionConfig
from megalm.models.qwen3 import RMSNorm


class OmniVisionBlock(nn.Module):
    def __init__(
        self,
        config: OmniVisionConfig,
        *,
        device: tensorplay.device,
        dtype: tensorplay.dtype,
        kernels: KernelBindings,
    ) -> None:
        super().__init__()
        self.norm1 = RMSNorm(
            config.hidden_size,
            1e-6,
            device=device,
            dtype=dtype,
            kernel_bindings=kernels,
        )
        self.norm2 = RMSNorm(
            config.hidden_size,
            1e-6,
            device=device,
            dtype=dtype,
            kernel_bindings=kernels,
        )
        self.attn = PackedVisionAttention(
            config.hidden_size,
            config.num_heads,
            device=device,
            dtype=dtype,
        )
        self.mlp = PackedSwiGLU(
            config.hidden_size,
            config.intermediate_size,
            bias=True,
            device=device,
            dtype=dtype,
            kernels=kernels,
        )

    def forward(
        self,
        hidden_states: tensorplay.Tensor,
        *,
        attention_layout: GroupedVarlenAttentionLayout,
        rotary: tensorplay.Tensor,
    ) -> tensorplay.Tensor:
        hidden_states = hidden_states + self.attn(
            self.norm1(hidden_states),
            attention_layout=attention_layout,
            rotary=rotary,
        )
        return hidden_states + self.mlp(self.norm2(hidden_states))


class OmniVisionMerger(nn.Module):
    def __init__(
        self,
        config: OmniVisionConfig,
        *,
        device: tensorplay.device,
        dtype: tensorplay.dtype,
        kernels: KernelBindings,
    ) -> None:
        super().__init__()
        self.merged_width = (
            config.hidden_size * config.spatial_merge_size**2
        )
        self.norm = RMSNorm(
            config.hidden_size,
            1e-6,
            device=device,
            dtype=dtype,
            kernel_bindings=kernels,
        )
        self.linear1_weight = frozen_parameter(
            (self.merged_width, self.merged_width),
            device=device,
            dtype=dtype,
        )
        self.linear1_bias = frozen_parameter(
            (self.merged_width,),
            device=device,
            dtype=dtype,
        )
        self.linear2_weight = frozen_parameter(
            (config.output_hidden_size, self.merged_width),
            device=device,
            dtype=dtype,
        )
        self.linear2_bias = frozen_parameter(
            (config.output_hidden_size,),
            device=device,
            dtype=dtype,
        )

    def forward(self, hidden_states: tensorplay.Tensor) -> tensorplay.Tensor:
        merged = self.norm(hidden_states).view(-1, self.merged_width)
        merged = F.gelu(
            F.linear(
                merged,
                self.linear1_weight,
                self.linear1_bias,
            ),
            approximate="none",
        )
        return F.linear(
            merged,
            self.linear2_weight,
            self.linear2_bias,
        )


__all__ = ["OmniVisionBlock", "OmniVisionMerger"]
