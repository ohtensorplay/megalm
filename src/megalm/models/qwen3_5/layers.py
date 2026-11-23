from __future__ import annotations

import tensorplay
import tensorplay.nn as nn
import tensorplay.nn.functional as F

from megalm.kernels.bindings import KernelBindings
from megalm.models.qwen3_5.attention import MRoPE, Qwen35Attention
from megalm.models.qwen3_5.config import Qwen35TextConfig
from megalm.models.qwen3_5.gdn import Qwen35GatedDeltaNet
from megalm.models.qwen3_5.moe import Qwen35SparseMoeBlock
from megalm.models.qwen3_5.norms import OffsetRMSNorm, frozen_parameter
from megalm.runtime.batch import ExecutionBatch
from megalm.runtime.hybrid_cache import HybridCache


class PackedDenseMLP(nn.Module):
    def __init__(
        self,
        config: Qwen35TextConfig,
        *,
        device: tensorplay.device,
        dtype: tensorplay.dtype,
        kernel_bindings: KernelBindings,
    ) -> None:
        super().__init__()
        self.gate_up_weight = frozen_parameter(
            (2 * config.intermediate_size, config.hidden_size),
            device=device,
            dtype=dtype,
        )
        self.down_proj_weight = frozen_parameter(
            (config.hidden_size, config.intermediate_size),
            device=device,
            dtype=dtype,
        )
        self._silu_and_mul = kernel_bindings.core.silu_and_mul

    def forward(self, hidden_states: tensorplay.Tensor) -> tensorplay.Tensor:
        gate_up = F.linear(hidden_states, self.gate_up_weight)
        return F.linear(
            self._silu_and_mul(gate_up),
            self.down_proj_weight,
        )


class Qwen35DecoderLayer(nn.Module):
    def __init__(
        self,
        config: Qwen35TextConfig,
        *,
        layer_index: int,
        rotary: MRoPE,
        device: tensorplay.device,
        dtype: tensorplay.dtype,
        kernel_bindings: KernelBindings,
        layer_type: str | None = None,
    ) -> None:
        super().__init__()
        layer_type = (
            config.layer_types[layer_index]
            if layer_type is None
            else layer_type
        )
        if layer_type not in {"full_attention", "linear_attention"}:
            raise ValueError(f"unsupported Qwen3.5 layer type {layer_type!r}")
        self.layer_type = layer_type
        self.self_attn: Qwen35Attention | None = None
        self.linear_attn: Qwen35GatedDeltaNet | None = None
        if layer_type == "full_attention":
            self.self_attn = Qwen35Attention(
                config,
                layer_index=layer_index,
                rotary=rotary,
                device=device,
                dtype=dtype,
                kernel_bindings=kernel_bindings,
            )
        else:
            self.linear_attn = Qwen35GatedDeltaNet(
                config,
                layer_index=layer_index,
                device=device,
                dtype=dtype,
                kernel_bindings=kernel_bindings,
            )
        self.mlp: PackedDenseMLP | Qwen35SparseMoeBlock
        if config.is_moe:
            self.mlp = Qwen35SparseMoeBlock(
                config,
                device=device,
                dtype=dtype,
                kernel_bindings=kernel_bindings,
            )
        else:
            self.mlp = PackedDenseMLP(
                config,
                device=device,
                dtype=dtype,
                kernel_bindings=kernel_bindings,
            )
        self.input_layernorm = OffsetRMSNorm(
            config.hidden_size,
            config.rms_norm_eps,
            device=device,
            dtype=dtype,
            kernel_bindings=kernel_bindings,
        )
        self.post_attention_layernorm = OffsetRMSNorm(
            config.hidden_size,
            config.rms_norm_eps,
            device=device,
            dtype=dtype,
            kernel_bindings=kernel_bindings,
        )

    def forward(
        self,
        hidden_states: tensorplay.Tensor,
        *,
        batch: ExecutionBatch,
        cache: HybridCache,
        position_embeddings: tuple[tensorplay.Tensor, tensorplay.Tensor],
    ) -> tensorplay.Tensor:
        hidden_states, residual = self.forward_with_residual(
            hidden_states,
            None,
            batch=batch,
            cache=cache,
            position_embeddings=position_embeddings,
        )
        return residual + hidden_states

    def forward_with_residual(
        self,
        hidden_states: tensorplay.Tensor,
        residual: tensorplay.Tensor | None,
        *,
        batch: ExecutionBatch,
        cache: HybridCache,
        position_embeddings: tuple[tensorplay.Tensor, tensorplay.Tensor],
    ) -> tuple[tensorplay.Tensor, tensorplay.Tensor]:
        if residual is None:
            residual = hidden_states
            normalized = self.input_layernorm(hidden_states)
        else:
            normalized, residual = self.input_layernorm.with_residual(
                hidden_states,
                residual,
            )
        if self.self_attn is not None:
            hidden_states = self.self_attn(
                normalized,
                batch=batch,
                cache=cache,
                position_embeddings=position_embeddings,
            )
        else:
            assert self.linear_attn is not None
            hidden_states = self.linear_attn(
                normalized,
                batch=batch,
                cache=cache,
            )
        hidden_states, residual = (
            self.post_attention_layernorm.with_residual(
                hidden_states,
                residual,
            )
        )
        if isinstance(self.mlp, Qwen35SparseMoeBlock):
            hidden_states = self.mlp(hidden_states, decode=batch.all_decode)
        else:
            hidden_states = self.mlp(hidden_states)
        return hidden_states, residual
