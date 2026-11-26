from __future__ import annotations

import tensorplay
import triton
import triton.language as tl

from megalm.kernels.kv_quant_triton import (
    load_dequant_kv,
    rht_enabled,
    rht_output,
    rht_query,
)
from megalm.runtime.kv_quant import (
    dequant_kernel_kwargs,
    head_scale_tensor,
    kv_quant_of,
    kv_scale_of,
    quantize_tensor,
    scale_values,
)


@triton.jit
def _paged_prefill_attention_kernel(
    query_ptr,
    key_cache_ptr,
    value_cache_ptr,
    key_scale_ptr,
    value_scale_ptr,
    head_scale_k_ptr,
    head_scale_v_ptr,
    block_tables_ptr,
    query_start_loc_ptr,
    positions_ptr,
    output_ptr,
    max_blocks,
    page_size: tl.constexpr,
    num_query_heads: tl.constexpr,
    num_kv_heads: tl.constexpr,
    head_dim: tl.constexpr,
    query_per_kv: tl.constexpr,
    scale: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
    stride_slot,
    stride_head,
    ss_page,
    ss_pos,
    ss_head,
    MODE_K: tl.constexpr,
    MODE_V: tl.constexpr,
    SCALAR_K,
    SCALAR_V,
    GLOBAL_SCALE,
    FP8_DTYPE: tl.constexpr,
    NATIVE: tl.constexpr,
    BMM1_SCALE,
    V_SCALE,
    PER_HEAD_NATIVE: tl.constexpr,
    SLIDING_WINDOW: tl.constexpr = 0,
):
    sequence_index = tl.program_id(0)
    query_head = tl.program_id(1)
    query_block = tl.program_id(2)
    kv_head = query_head // query_per_kv

    query_start = tl.load(query_start_loc_ptr + sequence_index)
    query_end = tl.load(query_start_loc_ptr + sequence_index + 1)
    query_offsets = query_block * BLOCK_M + tl.arange(0, BLOCK_M)
    query_tokens = query_start + query_offsets
    query_mask = query_tokens < query_end
    dimensions = tl.arange(0, BLOCK_D)
    dimension_mask = dimensions < head_dim
    absolute_positions = tl.load(
        positions_ptr + query_tokens,
        mask=query_mask,
        other=-1,
    )
    final_position = tl.load(positions_ptr + query_end - 1)
    sequence_length = final_position + 1

    query_base = (
        query_tokens[:, None] * num_query_heads * head_dim
        + query_head * head_dim
        + dimensions[None, :]
    )
    query = tl.load(
        query_ptr + query_base,
        mask=query_mask[:, None] & dimension_mask[None, :],
        other=0.0,
    )

    score_scale = scale
    value_scale = 1.0
    if NATIVE:
        if PER_HEAD_NATIVE:
            score_scale = scale * BMM1_SCALE * tl.load(head_scale_k_ptr + kv_head)
            value_scale = tl.load(head_scale_v_ptr + kv_head)
        else:
            score_scale = scale * BMM1_SCALE
            value_scale = V_SCALE

    running_max = tl.full((BLOCK_M,), -float("inf"), tl.float32)
    running_sum = tl.zeros((BLOCK_M,), tl.float32)
    accumulator = tl.zeros((BLOCK_M, BLOCK_D), tl.float32)
    key_start = 0
    while key_start < sequence_length:
        key_offsets = key_start + tl.arange(0, BLOCK_N)
        key_mask = key_offsets < sequence_length
        logical_pages = key_offsets // page_size
        page_offsets = key_offsets % page_size
        physical_pages = tl.load(
            block_tables_ptr
            + sequence_index * max_blocks
            + logical_pages,
            mask=key_mask,
            other=0,
        )
        slots = physical_pages * page_size + page_offsets
        cache_offsets = (
            slots[:, None] * stride_slot
            + kv_head * stride_head
            + dimensions[None, :]
        )
        cache_mask = key_mask[:, None] & dimension_mask[None, :]

        if NATIVE:
            keys = tl.load(
                key_cache_ptr + cache_offsets,
                mask=cache_mask,
                other=0.0,
            )
        else:
            keys = load_dequant_kv(
                key_cache_ptr,
                key_scale_ptr,
                head_scale_k_ptr,
                slots,
                kv_head,
                dimensions,
                stride_slot,
                stride_head,
                ss_page,
                ss_pos,
                ss_head,
                cache_mask,
                page_size,
                head_dim,
                MODE_K,
                SCALAR_K,
                GLOBAL_SCALE,
                FP8_DTYPE,
            ).to(query.dtype)
        scores = tl.dot(query, tl.trans(keys)) * score_scale
        causal_mask = (
            query_mask[:, None]
            & key_mask[None, :]
            & (key_offsets[None, :] <= absolute_positions[:, None])
        )
        if SLIDING_WINDOW > 0:
            # Gemma-4 local layers attend only to keys within a trailing
            # window: q_pos - k_pos < window.  Matches the HF overlay.
            causal_mask = causal_mask & (
                key_offsets[None, :]
                > absolute_positions[:, None] - SLIDING_WINDOW
            )
        scores = tl.where(causal_mask, scores, -float("inf"))
        block_max = tl.max(scores, axis=1)
        new_max = tl.maximum(running_max, block_max)
        new_max = tl.where(query_mask, new_max, 0.0)
        old_scale = tl.where(
            query_mask,
            tl.exp(running_max - new_max),
            0.0,
        )
        probabilities = tl.where(
            causal_mask,
            tl.exp(scores - new_max[:, None]),
            0.0,
        )

        if NATIVE:
            values = tl.load(
                value_cache_ptr + cache_offsets,
                mask=cache_mask,
                other=0.0,
            )
            # Probabilities are in [0, 1]; cast to the fp8 operand type so the
            # second GEMM stays fp8 (same folding vLLM's Triton backend does).
            probabilities_fp8 = probabilities.to(values.dtype)
            accumulator = (
                accumulator * old_scale[:, None]
                + tl.dot(probabilities_fp8, values)
            )
        else:
            values = load_dequant_kv(
                value_cache_ptr,
                value_scale_ptr,
                head_scale_v_ptr,
                slots,
                kv_head,
                dimensions,
                stride_slot,
                stride_head,
                ss_page,
                ss_pos,
                ss_head,
                cache_mask,
                page_size,
                head_dim,
                MODE_V,
                SCALAR_V,
                GLOBAL_SCALE,
                FP8_DTYPE,
            ).to(query.dtype)
            accumulator = (
                accumulator * old_scale[:, None]
                + tl.dot(probabilities.to(values.dtype), values)
            )
        running_sum = (
            running_sum * old_scale
            + tl.sum(probabilities, axis=1)
        )
        running_max = new_max
        key_start += BLOCK_N

    output_base = (
        query_tokens[:, None] * num_query_heads * head_dim
        + query_head * head_dim
        + dimensions[None, :]
    )
    result = accumulator / running_sum[:, None]
    if NATIVE:
        result = result * value_scale
    tl.store(
        output_ptr + output_base,
        result,
        mask=query_mask[:, None] & dimension_mask[None, :],
    )


def paged_prefill_attention(
    query: tensorplay.Tensor,
    key_cache: tensorplay.Tensor,
    value_cache: tensorplay.Tensor,
    block_tables: tensorplay.Tensor,
    query_start_loc: tensorplay.Tensor,
    positions: tensorplay.Tensor,
    *,
    query_lengths: tuple[int, ...],
    context_lengths: tuple[int, ...],
    page_size: int,
    output: tensorplay.Tensor | None = None,
    scale: float | None = None,
    sliding_window: int = 0,
) -> tensorplay.Tensor:
    """Paged prefill attention.

    ``scale`` overrides the default ``head_dim**-0.5`` score scale.  Models
    with unit-normalised Q/K (for example Gemma-4) pass ``scale=1.0``.
    ``sliding_window`` (0 disables) keeps only keys with ``q_pos - k_pos <
    window``, matching HuggingFace's local-attention overlay.
    """
    if not query_lengths or len(query_lengths) != len(context_lengths):
        raise ValueError("packed prefill metadata must contain every sequence")
    if query_start_loc.dtype != tensorplay.int32:
        raise ValueError("query_start_loc must use int32")
    query = query.contiguous()
    block_tables = block_tables.contiguous()
    positions = positions.contiguous()
    if output is None:
        output = tensorplay.empty_like(query)
    elif (
        output.shape != query.shape
        or output.device != query.device
        or output.dtype != query.dtype
        or not output.is_contiguous()
    ):
        raise ValueError("paged prefill output must match contiguous query")
    num_query_heads = query.shape[1]
    num_kv_heads = key_cache.shape[-2]
    head_dim = query.shape[-1]
    quant = kv_quant_of(key_cache)
    use_rht = rht_enabled(quant)
    if use_rht:
        query = rht_query(query, quant, head_dim)
    base_scale = head_dim**-0.5 if scale is None else scale
    softmax_scale = base_scale / head_dim if use_rht else base_scale
    key_scale = kv_scale_of(key_cache)
    value_scale = kv_scale_of(value_cache)
    if key_scale is None:
        key_scale = key_cache
        value_scale = value_cache
    head_scale_k = head_scale_tensor(
        quant, "key", query.device, key_cache
    )
    head_scale_v = head_scale_tensor(
        quant, "value", query.device, value_cache
    )
    k_kwargs = dequant_kernel_kwargs(quant, "key")
    v_kwargs = dequant_kernel_kwargs(quant, "value")

    native = bool(quant and quant.native_fp8_dot)
    if native:
        query = quantize_tensor(query, quant, "q")
        q_scale = scale_values(quant, "q")
        per_head = quant.q_group == "per_kv_head" or isinstance(
            scale_values(quant, "key"), tuple
        )
        bmm1 = (
            1.0
            if per_head
            else float(q_scale) * float(scale_values(quant, "key"))
        )
        value_scale_scalar = (
            1.0
            if per_head
            else float(scale_values(quant, "value"))
        )
    else:
        bmm1 = 1.0
        value_scale_scalar = 1.0
        per_head = False

    max_query_length = max(query_lengths)
    block_m = 16
    block_n = 32
    block_d = triton.next_power_of_2(head_dim)
    grid = (
        len(query_lengths),
        num_query_heads,
        triton.cdiv(max_query_length, block_m),
    )
    _paged_prefill_attention_kernel[grid](
        query_ptr=query,
        key_cache_ptr=key_cache,
        value_cache_ptr=value_cache,
        key_scale_ptr=key_scale,
        value_scale_ptr=value_scale,
        head_scale_k_ptr=head_scale_k,
        head_scale_v_ptr=head_scale_v,
        block_tables_ptr=block_tables,
        query_start_loc_ptr=query_start_loc,
        positions_ptr=positions,
        output_ptr=output,
        max_blocks=block_tables.shape[1],
        page_size=page_size,
        num_query_heads=num_query_heads,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        query_per_kv=num_query_heads // num_kv_heads,
        scale=softmax_scale,
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        BLOCK_D=block_d,
        stride_slot=key_cache.stride(1),
        stride_head=key_cache.stride(2),
        ss_page=key_scale.stride(0),
        ss_pos=key_scale.stride(1),
        ss_head=key_scale.stride(2),
        MODE_K=k_kwargs["MODE"],
        MODE_V=v_kwargs["MODE"],
        SCALAR_K=k_kwargs["SCALAR_SCALE"],
        SCALAR_V=v_kwargs["SCALAR_SCALE"],
        GLOBAL_SCALE=k_kwargs["GLOBAL_SCALE"],
        FP8_DTYPE=k_kwargs["FP8_DTYPE"],
        NATIVE=native,
        BMM1_SCALE=bmm1,
        V_SCALE=value_scale_scalar,
        PER_HEAD_NATIVE=per_head,
        SLIDING_WINDOW=int(sliding_window),
        num_warps=4,
        num_stages=2,
    )
    return rht_output(output, quant, head_dim)
