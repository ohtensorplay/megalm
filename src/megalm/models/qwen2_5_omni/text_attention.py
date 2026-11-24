from __future__ import annotations

import math

import tensorplay
import tensorplay.nn as nn
import tensorplay.nn.functional as F

from megalm.kernels.bindings import KernelBindings
from megalm.kernels.fallback import (
    dequantize_cache_to_dtype,
    quantize_cache_rows,
)
from megalm.models.components import frozen_parameter
from megalm.models.qwen2_5_omni.config import OmniTextConfig
from megalm.models.qwen2_5_omni.text_errors import OmniTextError
from megalm.models.qwen2_5_omni.text_rope import OmniMRoPE
from megalm.runtime.batch import BatchPhase, ExecutionBatch
from megalm.runtime.cache import PagedKVCache
from megalm.runtime.kv_quant import (
    attach_kv_quant,
    attach_kv_scale,
    kv_quant_of,
    kv_scale_of,
)


class OmniAttentionError(OmniTextError):
    """Raised when packed text attention receives invalid batch geometry."""


class OmniAttention(nn.Module):
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
        self.query_width = config.num_attention_heads * config.head_dim
        self.kv_width = config.num_key_value_heads * config.head_dim
        packed_width = self.query_width + 2 * self.kv_width
        self.qkv_weight = frozen_parameter(
            (packed_width, config.hidden_size),
            device=device,
            dtype=dtype,
        )
        self.qkv_bias = frozen_parameter(
            (packed_width,),
            device=device,
            dtype=dtype,
        )
        self.o_proj_weight = frozen_parameter(
            (config.hidden_size, self.query_width),
            device=device,
            dtype=dtype,
        )
        self.num_query_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.head_dim = config.head_dim
        self.query_per_kv = self.num_query_heads // self.num_kv_heads
        self.layer_index = layer_index
        self.rotary = rotary
        self._linear = kernels.core.linear
        self._active_kv_write = kernels.core.qwen35_active_kv_write
        self._paged_decode = kernels.core.paged_decode_attention
        self._paged_prefill = kernels.core.paged_prefill_attention

    def forward(
        self,
        hidden_states: tensorplay.Tensor,
        *,
        batch: ExecutionBatch,
        cache: PagedKVCache,
        position_embeddings: tuple[tensorplay.Tensor, tensorplay.Tensor],
    ) -> tensorplay.Tensor:
        projected = self._linear(
            hidden_states,
            self.qkv_weight,
            self.qkv_bias,
        )
        query, key, value = projected.split(
            (self.query_width, self.kv_width, self.kv_width),
            dim=-1,
        )
        query = query.view(-1, self.num_query_heads, self.head_dim)
        key = key.view(-1, self.num_kv_heads, self.head_dim)
        value = value.view(-1, self.num_kv_heads, self.head_dim)
        query, key = self.rotary.apply(
            query,
            key,
            position_embeddings,
        )
        key_cache, value_cache = cache.layer(self.layer_index)
        quant = kv_quant_of(key_cache)
        content_dim = (
            quant.content_dim(self.head_dim) if quant else self.head_dim
        )
        flat_key = attach_kv_quant(
            key_cache.view(-1, self.num_kv_heads, content_dim), quant
        )
        flat_value = attach_kv_quant(
            value_cache.view(-1, self.num_kv_heads, content_dim), quant
        )
        attach_kv_scale(flat_key, kv_scale_of(key_cache))
        attach_kv_scale(flat_value, kv_scale_of(value_cache))
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
            slots = batch.slot_mapping[active]
            page_size = key_cache.shape[1]
            quantize_cache_rows(
                key[active],
                key_cache,
                quant,
                "key",
                slots,
                page_size=page_size,
            )
            quantize_cache_rows(
                value[active],
                value_cache,
                quant,
                "value",
                slots,
                page_size=page_size,
            )
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
        return self._linear(
            attended.reshape(-1, self.query_width),
            self.o_proj_weight,
        )

    def _mixed_attention_triton(
        self,
        query: tensorplay.Tensor,
        key_cache: tensorplay.Tensor,
        value_cache: tensorplay.Tensor,
        batch: ExecutionBatch,
    ) -> tensorplay.Tensor:
        query_start_loc = batch.query_start_loc_device
        assert query_start_loc is not None
        assert self._paged_prefill is not None
        decode_count = 0
        for phase in batch.phases:
            if phase is not BatchPhase.DECODE:
                break
            decode_count += 1
        if any(
            phase is BatchPhase.DECODE
            for phase in batch.phases[decode_count:]
        ):
            raise OmniAttentionError(
                "mixed batch must keep decode rows contiguous"
            )
        output = tensorplay.empty_like(query)
        if decode_count:
            self._paged_decode(
                query[:decode_count],
                key_cache,
                value_cache,
                batch.block_tables[:decode_count],
                batch.sequence_lengths[:decode_count],
                batch.page_size,
                output=output[:decode_count],
            )
        if decode_count < batch.num_sequences:
            self._paged_prefill(
                query,
                key_cache,
                value_cache,
                batch.block_tables[decode_count:],
                query_start_loc[decode_count:],
                batch.positions,
                query_lengths=batch.query_lengths[decode_count:],
                context_lengths=batch.context_lengths[decode_count:],
                page_size=batch.page_size,
                output=output,
            )
        return output

    def _mixed_attention_tensorplay(
        self,
        query: tensorplay.Tensor,
        key_cache: tensorplay.Tensor,
        value_cache: tensorplay.Tensor,
        batch: ExecutionBatch,
    ) -> tensorplay.Tensor:
        output = tensorplay.empty_like(query)
        for sequence_index, phase in enumerate(batch.phases):
            start = batch.query_start_loc[sequence_index]
            end = batch.query_start_loc[sequence_index + 1]
            if phase is BatchPhase.DECODE:
                attended = self._paged_decode(
                    query[start:end],
                    key_cache,
                    value_cache,
                    batch.block_tables[
                        sequence_index : sequence_index + 1
                    ],
                    batch.sequence_lengths[
                        sequence_index : sequence_index + 1
                    ],
                    batch.page_size,
                )
                output[start:end] = attended
                continue
            context_length = int(batch.context_lengths[sequence_index])
            sequence_length = context_length + end - start
            num_pages = math.ceil(sequence_length / batch.page_size)
            pages = batch.block_tables[
                sequence_index,
                :num_pages,
            ].long()
            quant = kv_quant_of(key_cache)
            source_keys = dequantize_cache_to_dtype(
                key_cache, quant, "key", query.dtype
            ).reshape(-1, self.num_kv_heads, self.head_dim)
            source_values = dequantize_cache_to_dtype(
                value_cache, quant, "value", query.dtype
            ).reshape(-1, self.num_kv_heads, self.head_dim)
            keys = source_keys.index_select(0, pages)[:sequence_length]
            values = source_values.index_select(0, pages)[:sequence_length]
            keys = keys.repeat_interleave(self.query_per_kv, dim=1)
            values = values.repeat_interleave(self.query_per_kv, dim=1)
            query_states = query[start:end].transpose(0, 1).unsqueeze(0)
            key_states = keys.transpose(0, 1).unsqueeze(0)
            value_states = values.transpose(0, 1).unsqueeze(0)
            query_positions = tensorplay.arange(
                context_length,
                sequence_length,
                device=query.device,
            )
            key_positions = tensorplay.arange(
                sequence_length,
                device=query.device,
            )
            causal = (
                key_positions[None, :] <= query_positions[:, None]
            )[None, None]
            attended = F.scaled_dot_product_attention(
                query_states,
                key_states,
                value_states,
                attn_mask=causal,
                dropout_p=0.0,
                scale=self.head_dim**-0.5,
            )
            output[start:end] = attended.squeeze(0).transpose(0, 1)
        return output


__all__ = ["OmniAttention", "OmniAttentionError"]
