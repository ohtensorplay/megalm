from __future__ import annotations

import tensorplay
import tensorplay.nn as nn
import tensorplay.nn.functional as F

from megalm.kernels.bindings import KernelBindings
from megalm.models.qwen3 import RMSNorm
from megalm.models.qwen3_5.attention import MRoPE, Qwen35Attention
from megalm.models.qwen3_5.norms import frozen_parameter
from megalm.models.qwen3_vl.config import Qwen3VLTextConfig
from megalm.runtime.batch import ExecutionBatch
from megalm.runtime.cache import PagedKVCache


class Qwen3VLAttention(Qwen35Attention):
    """Qwen3-VL full attention: MRoPE with standard QK-Norm and no output gate."""

    def __init__(
        self,
        config: Qwen3VLTextConfig,
        *,
        layer_index: int,
        rotary: MRoPE,
        device: tensorplay.device,
        dtype: tensorplay.dtype,
        kernel_bindings: KernelBindings,
    ) -> None:
        nn.Module.__init__(self)
        self.layer_index = layer_index
        self.num_query_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.head_dim = config.head_dim
        self.query_width = self.num_query_heads * self.head_dim
        self.kv_width = self.num_kv_heads * self.head_dim
        self.query_per_kv = self.num_query_heads // self.num_kv_heads
        self.qkv_proj_weight = frozen_parameter(
            (
                self.query_width + 2 * self.kv_width,
                config.hidden_size,
            ),
            device=device,
            dtype=dtype,
        )
        self.o_proj_weight = frozen_parameter(
            (config.hidden_size, self.query_width),
            device=device,
            dtype=dtype,
        )
        self.q_norm = RMSNorm(
            self.head_dim,
            config.rms_norm_eps,
            device=device,
            dtype=dtype,
            kernel_bindings=kernel_bindings,
        )
        self.k_norm = RMSNorm(
            self.head_dim,
            config.rms_norm_eps,
            device=device,
            dtype=dtype,
            kernel_bindings=kernel_bindings,
        )
        self.rotary = rotary
        self._paged_decode = kernel_bindings.core.paged_decode_attention
        self._paged_prefill = kernel_bindings.core.paged_prefill_attention
        self._active_kv_write = (
            kernel_bindings.core.qwen35_active_kv_write
        )

    def forward(
        self,
        hidden_states: tensorplay.Tensor,
        *,
        batch: ExecutionBatch,
        cache: PagedKVCache,
        position_embeddings: tuple[tensorplay.Tensor, tensorplay.Tensor],
    ) -> tensorplay.Tensor:
        projected = F.linear(hidden_states, self.qkv_proj_weight)
        query, key, value = projected.split(
            (self.query_width, self.kv_width, self.kv_width),
            dim=-1,
        )
        query = self.q_norm(
            query.view(-1, self.num_query_heads, self.head_dim)
        )
        key = self.k_norm(
            key.view(-1, self.num_kv_heads, self.head_dim)
        )
        value = value.view(-1, self.num_kv_heads, self.head_dim)
        query, key = self.rotary.apply(query, key, position_embeddings)
        key_cache, value_cache = cache.layer(self.layer_index)
        flat_key = key_cache.view(-1, self.num_kv_heads, self.head_dim)
        flat_value = value_cache.view(
            -1,
            self.num_kv_heads,
            self.head_dim,
        )
        active = batch.token_active
        if self._active_kv_write is not None:
            self._active_kv_write(
                key,
                value,
                flat_key,
                flat_value,
                batch.slot_mapping,
                active,
            )
        else:
            active_slots = batch.slot_mapping[active]
            flat_key.index_copy_(0, active_slots, key[active])
            flat_value.index_copy_(0, active_slots, value[active])

        if batch.all_decode:
            attended = self._paged_decode(
                query,
                key_cache,
                value_cache,
                batch.block_tables,
                batch.sequence_lengths,
                batch.page_size,
            )
        elif self._paged_prefill is not None:
            attended = self._mixed_attention_triton(
                query,
                key_cache,
                value_cache,
                batch,
            )
        else:
            attended = self._mixed_attention_tensorplay(
                query,
                key_cache,
                value_cache,
                batch,
            )
        return F.linear(
            attended.reshape(hidden_states.shape[0], self.query_width),
            self.o_proj_weight,
        )


__all__ = ["Qwen3VLAttention"]
