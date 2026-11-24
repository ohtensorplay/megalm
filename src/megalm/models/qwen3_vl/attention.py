from __future__ import annotations

import tensorplay
import tensorplay.nn as nn
import tensorplay.nn.functional as F

from megalm.kernels.bindings import KernelBindings
from megalm.kernels.qk_rope import (
    FUSED_QK_NORM_ROPE,
    FUSED_QK_ROPE_KV,
    qk_norm_rope,
    qk_norm_rope_kv,
)
from megalm.kernels.fallback import quantize_cache_rows
from megalm.models.components.int4_linear import Int4GroupLinear
from megalm.models.qwen3 import RMSNorm
from megalm.models.qwen3_5.attention import MRoPE, Qwen35Attention
from megalm.models.qwen3_5.norms import frozen_parameter
from megalm.models.qwen3_vl.config import (
    Qwen3VLQuantConfig,
    Qwen3VLTextConfig,
)
from megalm.runtime.batch import ExecutionBatch
from megalm.runtime.cache import PagedKVCache
from megalm.runtime.kv_quant import (
    attach_kv_quant,
    attach_kv_scale,
    kv_quant_of,
    kv_scale_of,
)


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
        quantization: Qwen3VLQuantConfig | None = None,
    ) -> None:
        nn.Module.__init__(self)
        self.layer_index = layer_index
        self.num_query_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.head_dim = config.head_dim
        self.query_width = self.num_query_heads * self.head_dim
        self.kv_width = self.num_kv_heads * self.head_dim
        self.query_per_kv = self.num_query_heads // self.num_kv_heads
        self.qkv_int4: Int4GroupLinear | None = None
        self.o_proj_int4: Int4GroupLinear | None = None
        if quantization is None:
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
        else:
            self.qkv_proj_weight = None
            self.o_proj_weight = None
            self.qkv_int4 = Int4GroupLinear(
                config.hidden_size,
                self.query_width + 2 * self.kv_width,
                group_size=quantization.group_size,
                device=device,
                dtype=dtype,
            )
            self.o_proj_int4 = Int4GroupLinear(
                self.query_width,
                config.hidden_size,
                group_size=quantization.group_size,
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
        self.dtype = dtype
        self.rotary = rotary
        self._paged_decode = kernel_bindings.core.paged_decode_attention
        self._paged_prefill = kernel_bindings.core.paged_prefill_attention
        self._active_kv_write = (
            kernel_bindings.core.qwen35_active_kv_write
        )
        # One fused launch replaces the per-layer norm/rotate/cat chain, which
        # is launch-latency bound on 4 KB tensors during decode.
        self._fused_qk_norm_rope = FUSED_QK_NORM_ROPE and (
            tensorplay.device(device).type == "cuda"
        ) and dtype in {tensorplay.bfloat16, tensorplay.float16}
        self._fused_qk_rope_kv = FUSED_QK_ROPE_KV and (
            self._fused_qk_norm_rope
        )

    def forward(
        self,
        hidden_states: tensorplay.Tensor,
        *,
        batch: ExecutionBatch,
        cache: PagedKVCache,
        position_embeddings: tuple[tensorplay.Tensor, tensorplay.Tensor],
    ) -> tensorplay.Tensor:
        projected = (
            self.qkv_int4(hidden_states)
            if self.qkv_int4 is not None
            else F.linear(hidden_states, self.qkv_proj_weight)
        )
        query, key, value = projected.split(
            (self.query_width, self.kv_width, self.kv_width),
            dim=-1,
        )
        key_cache, value_cache = cache.layer(self.layer_index)
        quant = kv_quant_of(key_cache)
        content_dim = (
            quant.content_dim(self.head_dim) if quant else self.head_dim
        )
        active = batch.token_active
        # One launch does both rotations plus the paged-cache store.  The
        # three separate launches it replaces each run on a few-KB tensor, so
        # the per-launch latency costs more than the arithmetic.
        fused_kv_write = (
            self._fused_qk_rope_kv
            and quant is None
            and self._active_kv_write is not None
            and int(query.shape[0]) > 0
        )
        if fused_kv_write:
            query, key = qk_norm_rope_kv(
                query,
                key,
                value,
                self.q_norm.weight,
                self.k_norm.weight,
                position_embeddings,
                q_heads=self.num_query_heads,
                k_heads=self.num_kv_heads,
                head_dim=self.head_dim,
                rotary_dim=self.rotary.rotary_dim,
                q_eps=self.q_norm.eps,
                k_eps=self.k_norm.eps,
                dtype=self.dtype,
                key_cache=key_cache.view(-1, self.num_kv_heads, content_dim),
                value_cache=value_cache.view(
                    -1, self.num_kv_heads, content_dim
                ),
                slots=batch.slot_mapping,
                active=active,
            )
        elif self._fused_qk_norm_rope:
            query = qk_norm_rope(
                query,
                self.q_norm.weight,
                position_embeddings,
                heads=self.num_query_heads,
                head_dim=self.head_dim,
                rotary_dim=self.rotary.rotary_dim,
                eps=self.q_norm.eps,
                dtype=self.dtype,
            )
            key = qk_norm_rope(
                key,
                self.k_norm.weight,
                position_embeddings,
                heads=self.num_kv_heads,
                head_dim=self.head_dim,
                rotary_dim=self.rotary.rotary_dim,
                eps=self.k_norm.eps,
                dtype=self.dtype,
            )
            value = value.view(-1, self.num_kv_heads, self.head_dim)
        else:
            query = self.q_norm(
                query.view(-1, self.num_query_heads, self.head_dim)
            )
            key = self.k_norm(
                key.view(-1, self.num_kv_heads, self.head_dim)
            )
            value = value.view(-1, self.num_kv_heads, self.head_dim)
            query, key = self.rotary.apply(query, key, position_embeddings)
        if not fused_kv_write:
            flat_key = attach_kv_quant(
                key_cache.view(-1, self.num_kv_heads, content_dim), quant
            )
            flat_value = attach_kv_quant(
                value_cache.view(-1, self.num_kv_heads, content_dim), quant
            )
            attach_kv_scale(flat_key, kv_scale_of(key_cache))
            attach_kv_scale(flat_value, kv_scale_of(value_cache))
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
        attended = attended.reshape(
            hidden_states.shape[0], self.query_width
        )
        if self.o_proj_int4 is not None:
            return self.o_proj_int4(attended)
        return F.linear(attended, self.o_proj_weight)


__all__ = ["Qwen3VLAttention"]
