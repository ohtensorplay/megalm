from __future__ import annotations

import tensorplay
import tensorplay.nn as nn

from megalm.kernels.bindings import KernelBindings
from megalm.models.qwen2_5_omni.config import OmniTextConfig
from megalm.models.qwen2_5_omni.text_attention import (
    OmniAttention,
    OmniAttentionError,
)
from megalm.models.qwen2_5_omni.text_layers import OmniDecoderLayer
from megalm.models.qwen2_5_omni.text_errors import OmniTextError
from megalm.models.qwen2_5_omni.text_rope import (
    OmniMRoPE,
    OmniTextPositionError,
)
from megalm.models.qwen3 import RMSNorm
from megalm.runtime.batch import ExecutionBatch
from megalm.runtime.cache import PagedKVCache


class OmniTextModel(nn.Module):
    def __init__(
        self,
        config: OmniTextConfig,
        *,
        device: tensorplay.device,
        dtype: tensorplay.dtype,
        kernels: KernelBindings,
    ) -> None:
        super().__init__()
        self.embed_tokens = nn.Embedding(
            config.vocab_size,
            config.hidden_size,
            device=device,
            dtype=dtype,
        )
        self.embed_tokens.requires_grad_(False)
        self.rotary = OmniMRoPE(config, device=device)
        self.layers = nn.ModuleList(
            [
                OmniDecoderLayer(
                    config,
                    index,
                    device=device,
                    dtype=dtype,
                    rotary=self.rotary,
                    kernels=kernels,
                )
                for index in range(config.num_hidden_layers)
            ]
        )
        self.norm = RMSNorm(
            config.hidden_size,
            config.rms_norm_eps,
            device=device,
            dtype=dtype,
            kernel_bindings=kernels,
        )
        self._last_input_embeddings: tensorplay.Tensor | None = None

    def forward(
        self,
        batch: ExecutionBatch,
        cache: PagedKVCache,
    ) -> tensorplay.Tensor:
        hidden_states = (
            self.embed_tokens(batch.input_ids)
            if batch.input_embeddings is None
            else batch.input_embeddings
        )
        self._last_input_embeddings = hidden_states
        positions = (
            batch.positions.unsqueeze(0).expand(3, -1)
            if batch.mrope_positions is None
            else batch.mrope_positions
        )
        position_embeddings = self.rotary(
            positions,
            dtype=hidden_states.dtype,
        )
        residual: tensorplay.Tensor | None = None
        for layer in self.layers:
            hidden_states, residual = layer(
                hidden_states,
                residual,
                batch=batch,
                cache=cache,
                position_embeddings=position_embeddings,
            )
        assert residual is not None
        hidden_states, _ = self.norm.with_residual(
            hidden_states,
            residual,
        )
        return hidden_states

    @property
    def last_input_embeddings(self) -> tensorplay.Tensor:
        if self._last_input_embeddings is None:
            raise OmniTextError("Omni text model has not executed")
        return self._last_input_embeddings

__all__ = [
    "OmniAttention",
    "OmniMRoPE",
    "OmniTextError",
    "OmniTextModel",
]
