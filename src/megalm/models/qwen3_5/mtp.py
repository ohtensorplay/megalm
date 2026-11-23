from __future__ import annotations

import tensorplay
import tensorplay.nn as nn
import tensorplay.nn.functional as F

from megalm.kernels.bindings import KernelBindings
from megalm.models.qwen3_5.attention import MRoPE
from megalm.models.qwen3_5.config import Qwen35TextConfig
from megalm.models.qwen3_5.layers import Qwen35DecoderLayer
from megalm.models.qwen3_5.norms import OffsetRMSNorm, frozen_parameter
from megalm.runtime.batch import ExecutionBatch
from megalm.runtime.cache import PagedKVCache


class Qwen35MultiTokenPredictor(nn.Module):
    """The checkpoint-native one-layer Qwen3.5 MTP draft model.

    The token embedding and LM head are owned by the target model.  Only the
    checkpoint's ``mtp.*`` parameters live here, so disabling MTP allocates no
    draft weights and adds no target-model hot-path branch.
    """

    def __init__(
        self,
        config: Qwen35TextConfig,
        *,
        embed_tokens: nn.Embedding,
        rotary: MRoPE,
        device: tensorplay.device,
        dtype: tensorplay.dtype,
        kernel_bindings: KernelBindings,
    ) -> None:
        super().__init__()
        if config.mtp_num_hidden_layers != 1:
            raise ValueError(
                "Qwen3.5 native MTP requires mtp_num_hidden_layers=1"
            )
        self.config = config
        self.embed_tokens = embed_tokens
        self.fc_weight = frozen_parameter(
            (config.hidden_size, 2 * config.hidden_size),
            device=device,
            dtype=dtype,
        )
        self.layers = nn.ModuleList(
            [
                Qwen35DecoderLayer(
                    config,
                    layer_index=0,
                    layer_type="full_attention",
                    rotary=rotary,
                    device=device,
                    dtype=dtype,
                    kernel_bindings=kernel_bindings,
                )
            ]
        )
        self.norm = OffsetRMSNorm(
            config.hidden_size,
            config.rms_norm_eps,
            device=device,
            dtype=dtype,
            kernel_bindings=kernel_bindings,
        )
        self.pre_fc_norm_embedding = OffsetRMSNorm(
            config.hidden_size,
            config.rms_norm_eps,
            device=device,
            dtype=dtype,
            kernel_bindings=kernel_bindings,
        )
        self.pre_fc_norm_hidden = OffsetRMSNorm(
            config.hidden_size,
            config.rms_norm_eps,
            device=device,
            dtype=dtype,
            kernel_bindings=kernel_bindings,
        )
        self.rotary = rotary

    def forward(
        self,
        batch: ExecutionBatch,
        target_hidden_states: tensorplay.Tensor,
        cache: PagedKVCache,
    ) -> tensorplay.Tensor:
        if target_hidden_states.shape != (
            batch.input_ids.shape[0],
            self.config.hidden_size,
        ):
            raise ValueError(
                "MTP target hidden states must align one-to-one with input ids"
            )
        embeddings = self.pre_fc_norm_embedding(
            self.embed_tokens(batch.input_ids)
        )
        target_hidden_states = self.pre_fc_norm_hidden(target_hidden_states)
        hidden_states = F.linear(
            tensorplay.cat((embeddings, target_hidden_states), dim=-1),
            self.fc_weight,
        )
        positions = (
            batch.positions
            if batch.mrope_positions is None
            else batch.mrope_positions
        )
        position_embeddings = self.rotary(
            positions,
            dtype=hidden_states.dtype,
        )
        hidden_states = self.layers[0](
            hidden_states,
            batch=batch,
            cache=cache,  # type: ignore[arg-type]
            position_embeddings=position_embeddings,
        )
        return self.norm(hidden_states)

    def compute_logits(
        self,
        hidden_states: tensorplay.Tensor,
        lm_head_weight: tensorplay.Tensor,
    ) -> tensorplay.Tensor:
        return F.linear(hidden_states, lm_head_weight)
