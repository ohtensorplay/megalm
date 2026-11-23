from __future__ import annotations

import tensorplay
import tensorplay.nn as nn

from megalm.kernels.vision_attention import (
    GroupedVarlenAttentionLayout,
    grouped_varlen_attention,
)
from megalm.models.components.vision_rope import apply_vision_rope


def _freeze(module: nn.Module) -> nn.Module:
    module.requires_grad_(False)
    return module


class PackedVisionAttention(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        *,
        device: tensorplay.device,
        dtype: tensorplay.dtype,
    ) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.qkv = _freeze(
            nn.Linear(
                hidden_size,
                3 * hidden_size,
                bias=True,
                device=device,
                dtype=dtype,
            )
        )
        self.proj = _freeze(
            nn.Linear(
                hidden_size,
                hidden_size,
                bias=True,
                device=device,
                dtype=dtype,
            )
        )

    def forward(
        self,
        hidden_states: tensorplay.Tensor,
        *,
        attention_layout: GroupedVarlenAttentionLayout,
        rotary: tensorplay.Tensor,
    ) -> tensorplay.Tensor:
        token_count = hidden_states.shape[0]
        query, key, value = (
            self.qkv(hidden_states)
            .view(token_count, 3, self.num_heads, self.head_dim)
            .unbind(dim=1)
        )
        query = apply_vision_rope(query, rotary)
        key = apply_vision_rope(key, rotary)
        attended = grouped_varlen_attention(
            query,
            key,
            value,
            attention_layout,
            scale=self.head_dim**-0.5,
        )
        return self.proj(attended.reshape(token_count, -1))


__all__ = ["PackedVisionAttention"]
