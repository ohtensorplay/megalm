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
from megalm.models.qwen3_5.config import Qwen35TextConfig
from megalm.models.qwen3_5.norms import OffsetRMSNorm, frozen_parameter
from megalm.runtime.batch import BatchPhase, ExecutionBatch
from megalm.runtime.hybrid_cache import HybridCache
from megalm.runtime.kv_quant import (
    attach_kv_quant,
    attach_kv_scale,
    kv_quant_of,
    kv_scale_of,
)


class MRoPE(nn.Module):
    """Qwen3.5 interleaved 3-axis partial rotary embedding."""

    def __init__(
        self,
        config: Qwen35TextConfig,
        *,
        device: tensorplay.device,
    ) -> None:
        super().__init__()
        rotary_dim = int(config.head_dim * config.partial_rotary_factor)
        inverse_frequency = 1.0 / (
            config.rope_theta
            ** (
                tensorplay.arange(
                    0,
                    rotary_dim,
                    2,
                    dtype=tensorplay.float32,
                    device=device,
                )
                / rotary_dim
            )
        )
        axis = tensorplay.zeros(
            rotary_dim // 2,
            dtype=tensorplay.long,
            device=device,
        )
        if config.mrope_interleaved:
            height, width = config.mrope_sections[1:]
            axis[1 : height * 3 : 3] = 1
            axis[2 : width * 3 : 3] = 2
        else:
            start = config.mrope_sections[0]
            axis[start : start + config.mrope_sections[1]] = 1
            axis[start + config.mrope_sections[1] :] = 2
        self.register_buffer(
            "inverse_frequency",
            inverse_frequency,
            persistent=False,
        )
        self.register_buffer("axis", axis, persistent=False)
        self.rotary_dim = rotary_dim

    def forward(
        self,
        positions: tensorplay.Tensor,
        *,
        dtype: tensorplay.dtype,
    ) -> tuple[tensorplay.Tensor, tensorplay.Tensor]:
        if positions.ndim == 1:
            positions = positions.unsqueeze(0).expand(3, -1)
        if positions.shape[0] != 3:
            raise RuntimeError("Qwen3.5 MRoPE positions must have three axes")
        frequencies = (
            positions.float().unsqueeze(-1)
            * self.inverse_frequency[None, None, :]
        )
        token_indices = tensorplay.arange(
            positions.shape[1],
            device=positions.device,
        )[:, None]
        frequency_indices = tensorplay.arange(
            self.inverse_frequency.numel(),
            device=positions.device,
        )[None, :]
        selected = frequencies[
            self.axis[None, :],
            token_indices,
            frequency_indices,
        ]
        embeddings = tensorplay.cat((selected, selected), dim=-1)
        return embeddings.cos().to(dtype), embeddings.sin().to(dtype)

    def apply(
        self,
        query: tensorplay.Tensor,
        key: tensorplay.Tensor,
        position_embeddings: tuple[tensorplay.Tensor, tensorplay.Tensor],
    ) -> tuple[tensorplay.Tensor, tensorplay.Tensor]:
        cos, sin = position_embeddings
        query_rotary, query_pass = (
            query[..., : self.rotary_dim],
            query[..., self.rotary_dim :],
        )
        key_rotary, key_pass = (
            key[..., : self.rotary_dim],
            key[..., self.rotary_dim :],
        )
        query_rotated = _rotate_half(query_rotary)
        key_rotated = _rotate_half(key_rotary)
        cos = cos[:, None, :]
        sin = sin[:, None, :]
        query = tensorplay.cat(
            (
                query_rotary * cos + query_rotated * sin,
                query_pass,
            ),
            dim=-1,
        )
        key = tensorplay.cat(
            (
                key_rotary * cos + key_rotated * sin,
                key_pass,
            ),
            dim=-1,
        )
        return query, key


def _rotate_half(hidden_states: tensorplay.Tensor) -> tensorplay.Tensor:
    first, second = hidden_states.chunk(2, dim=-1)
    return tensorplay.cat((-second, first), dim=-1)


class Qwen35Attention(nn.Module):
    def __init__(
        self,
        config: Qwen35TextConfig,
        *,
        layer_index: int,
        rotary: MRoPE,
        device: tensorplay.device,
        dtype: tensorplay.dtype,
        kernel_bindings: KernelBindings,
    ) -> None:
        super().__init__()
        self.layer_index = layer_index
        self.num_query_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.head_dim = config.head_dim
        self.query_width = self.num_query_heads * self.head_dim
        self.kv_width = self.num_kv_heads * self.head_dim
        self.query_per_kv = self.num_query_heads // self.num_kv_heads
        self.qkv_proj_weight = frozen_parameter(
            (
                2 * self.query_width + 2 * self.kv_width,
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
        self.q_norm = OffsetRMSNorm(
            self.head_dim,
            config.rms_norm_eps,
            device=device,
            dtype=dtype,
            kernel_bindings=kernel_bindings,
        )
        self.k_norm = OffsetRMSNorm(
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
        cache: HybridCache,
        position_embeddings: tuple[tensorplay.Tensor, tensorplay.Tensor],
    ) -> tensorplay.Tensor:
        projected = F.linear(hidden_states, self.qkv_proj_weight)
        q_gate, key, value = projected.split(
            (2 * self.query_width, self.kv_width, self.kv_width),
            dim=-1,
        )
        query, gate = q_gate.view(
            -1,
            self.num_query_heads,
            2 * self.head_dim,
        ).chunk(2, dim=-1)
        key = key.view(-1, self.num_kv_heads, self.head_dim)
        value = value.view(-1, self.num_kv_heads, self.head_dim)
        query = self.q_norm(query)
        key = self.k_norm(key)
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
            active_slots = batch.slot_mapping[active]
            page_size = key_cache.shape[1]
            quantize_cache_rows(
                key[active],
                key_cache,
                quant,
                "key",
                active_slots,
                page_size=page_size,
            )
            quantize_cache_rows(
                value[active],
                value_cache,
                quant,
                "value",
                active_slots,
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
        attended = attended * tensorplay.sigmoid(gate)
        return F.linear(
            attended.reshape(hidden_states.shape[0], self.query_width),
            self.o_proj_weight,
        )

    def _mixed_attention_triton(
        self,
        query: tensorplay.Tensor,
        key_cache: tensorplay.Tensor,
        value_cache: tensorplay.Tensor,
        batch: ExecutionBatch,
    ) -> tensorplay.Tensor:
        query_starts = batch.query_start_loc_device
        if query_starts is None or self._paged_prefill is None:
            raise RuntimeError("Qwen3.5 prefill metadata is incomplete")
        decode_count = 0
        for phase in batch.phases:
            if phase is not BatchPhase.DECODE:
                break
            decode_count += 1
        if any(
            phase is BatchPhase.DECODE
            for phase in batch.phases[decode_count:]
        ):
            raise RuntimeError("mixed batch must keep decode rows contiguous")
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
                query_starts[decode_count:],
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
        decode_rows = [
            index
            for index, phase in enumerate(batch.phases)
            if phase is BatchPhase.DECODE
        ]
        if decode_rows:
            row_indices = tensorplay.tensor(
                decode_rows,
                dtype=tensorplay.long,
                device=query.device,
            )
            token_indices = tensorplay.tensor(
                [batch.query_start_loc[index] for index in decode_rows],
                dtype=tensorplay.long,
                device=query.device,
            )
            decoded = self._paged_decode(
                query.index_select(0, token_indices),
                key_cache,
                value_cache,
                batch.block_tables.index_select(0, row_indices),
                batch.sequence_lengths.index_select(0, row_indices),
                batch.page_size,
            )
            output.index_copy_(0, token_indices, decoded)
        for sequence_index, phase in enumerate(batch.phases):
            if phase is BatchPhase.DECODE:
                continue
            start = batch.query_start_loc[sequence_index]
            end = batch.query_start_loc[sequence_index + 1]
            context_length = batch.context_lengths[sequence_index]
            sequence_length = context_length + end - start
            page_count = math.ceil(sequence_length / batch.page_size)
            page_ids = batch.block_tables[
                sequence_index,
                :page_count,
            ].long()
            quant = kv_quant_of(key_cache)
            source_keys = dequantize_cache_to_dtype(
                key_cache, quant, "key", query.dtype
            )
            source_values = dequantize_cache_to_dtype(
                value_cache, quant, "value", query.dtype
            )
            # Gather whole pages first, then flatten page/token into rows.
            # Reshaping before the gather would index token rows with page ids.
            keys = (
                source_keys.index_select(0, page_ids)
                .reshape(-1, self.num_kv_heads, self.head_dim)[:sequence_length]
            )
            values = (
                source_values.index_select(0, page_ids)
                .reshape(-1, self.num_kv_heads, self.head_dim)[:sequence_length]
            )
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
            )[None, None, :, :]
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
