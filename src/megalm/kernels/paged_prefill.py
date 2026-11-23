from __future__ import annotations

import tensorplay
import triton
import triton.language as tl


@triton.jit
def _paged_prefill_attention_kernel(
    query_ptr,
    key_cache_ptr,
    value_cache_ptr,
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
            slots[:, None] * num_kv_heads * head_dim
            + kv_head * head_dim
            + dimensions[None, :]
        )
        cache_mask = key_mask[:, None] & dimension_mask[None, :]
        keys = tl.load(
            key_cache_ptr + cache_offsets,
            mask=cache_mask,
            other=0.0,
        )
        scores = tl.dot(query, tl.trans(keys)) * scale
        causal_mask = (
            query_mask[:, None]
            & key_mask[None, :]
            & (key_offsets[None, :] <= absolute_positions[:, None])
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
        values = tl.load(
            value_cache_ptr + cache_offsets,
            mask=cache_mask,
            other=0.0,
        )
        accumulator = (
            accumulator * old_scale[:, None]
            + tl.dot(
                probabilities.to(values.dtype),
                values,
            )
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
    tl.store(
        output_ptr + output_base,
        accumulator / running_sum[:, None],
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
) -> tensorplay.Tensor:
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
        query,
        key_cache,
        value_cache,
        block_tables,
        query_start_loc,
        positions,
        output,
        block_tables.shape[1],
        page_size=page_size,
        num_query_heads=num_query_heads,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        query_per_kv=num_query_heads // num_kv_heads,
        scale=head_dim**-0.5,
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        BLOCK_D=block_d,
        num_warps=4,
        num_stages=2,
    )
    return output
