from __future__ import annotations

import tensorplay
import tensorplay.nn as nn

from megalm.kernels.bindings import KernelBindings
from megalm.models.components import PackedSwiGLU
from megalm.models.qwen2_5_omni.config import OmniTextConfig
from megalm.models.qwen2_5_omni.text_attention import OmniAttention
from megalm.models.qwen2_5_omni.text_rope import OmniMRoPE
from megalm.models.qwen3 import RMSNorm
from megalm.runtime.batch import ExecutionBatch
from megalm.runtime.cache import PagedKVCache


class OmniMLP(PackedSwiGLU):
    def __init__(
        self,
        config: OmniTextConfig,
        *,
        device: tensorplay.device,
        dtype: tensorplay.dtype,
        kernels: KernelBindings,
    ) -> None:
        super().__init__(
            config.hidden_size,
            config.intermediate_size,
            bias=False,
            device=device,
            dtype=dtype,
            kernels=kernels,
            linear=kernels.core.linear,
        )


class OmniDecoderLayer(nn.Module):
    def __init__(
        self,
        config: OmniTextConfig,
        layer_index: int,
        *,
        device: tensorplay.device,
        dtype: tensorplay.dtype,
        rotary: OmniMRoPE,
        kernels: KernelBindings,
    ) -> None:
        super().__init__()
        self.self_attn = OmniAttention(
            config,
            layer_index,
            device=device,
            dtype=dtype,
            rotary=rotary,
            kernels=kernels,
        )
        self.mlp = OmniMLP(
            config,
            device=device,
            dtype=dtype,
            kernels=kernels,
        )
        self.input_layernorm = RMSNorm(
            config.hidden_size,
            config.rms_norm_eps,
            device=device,
            dtype=dtype,
            kernel_bindings=kernels,
        )
        self.post_attention_layernorm = RMSNorm(
            config.hidden_size,
            config.rms_norm_eps,
            device=device,
            dtype=dtype,
            kernel_bindings=kernels,
        )

    def forward(
        self,
        hidden_states: tensorplay.Tensor,
        residual: tensorplay.Tensor | None,
        *,
        batch: ExecutionBatch,
        cache: PagedKVCache,
        position_embeddings: tuple[tensorplay.Tensor, tensorplay.Tensor],
    ) -> tuple[tensorplay.Tensor, tensorplay.Tensor]:
        if residual is None:
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)
        else:
            hidden_states, residual = self.input_layernorm.with_residual(
                hidden_states,
                residual,
            )
        hidden_states = self.self_attn(
            hidden_states,
            batch=batch,
            cache=cache,
            position_embeddings=position_embeddings,
        )
        hidden_states, residual = (
            self.post_attention_layernorm.with_residual(
                hidden_states,
                residual,
            )
        )
        return self.mlp(hidden_states), residual


__all__ = ["OmniDecoderLayer", "OmniMLP"]
