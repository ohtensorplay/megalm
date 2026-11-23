from __future__ import annotations

import tensorplay
import triton
import triton.language as tl

@triton.jit
def _paged_decode_single_kernel(
    query_ptr,
    key_cache_ptr,
    value_cache_ptr,
    block_tables_ptr,
    sequence_lengths_ptr,
    output_ptr,
    max_blocks,
    page_size: tl.constexpr,
    num_query_heads: tl.constexpr,
    num_kv_heads: tl.constexpr,
    head_dim: tl.constexpr,
    query_per_kv: tl.constexpr,
    scale: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    batch_index = tl.program_id(0)
    query_head = tl.program_id(1)
    kv_head = query_head // query_per_kv
    sequence_length = tl.load(sequence_lengths_ptr + batch_index)
    dimensions = tl.arange(0, BLOCK_D)
    dimension_mask = dimensions < head_dim
    query_base = (
        batch_index * num_query_heads * head_dim + query_head * head_dim
    )
    query = tl.load(
        query_ptr + query_base + dimensions,
        mask=dimension_mask,
        other=0.0,
    ).to(tl.float32)

    running_max = tl.full((), -float("inf"), tl.float32)
    running_sum = tl.zeros((), tl.float32)
    accumulator = tl.zeros((BLOCK_D,), tl.float32)
    token_start = 0
    while token_start < sequence_length:
        token_offsets = token_start + tl.arange(0, BLOCK_N)
        token_mask = token_offsets < sequence_length
        logical_pages = token_offsets // page_size
        page_offsets = token_offsets % page_size
        physical_pages = tl.load(
            block_tables_ptr
            + batch_index * max_blocks
            + logical_pages,
            mask=token_mask,
            other=0,
        )
        slots = physical_pages * page_size + page_offsets
        cache_offsets = (
            slots[:, None] * num_kv_heads * head_dim
            + kv_head * head_dim
            + dimensions[None, :]
        )
        matrix_mask = token_mask[:, None] & dimension_mask[None, :]
        keys = tl.load(
            key_cache_ptr + cache_offsets,
            mask=matrix_mask,
            other=0.0,
        ).to(tl.float32)
        scores = tl.sum(keys * query[None, :], axis=1) * scale
        scores = tl.where(token_mask, scores, -float("inf"))
        block_max = tl.max(scores, axis=0)
        new_max = tl.maximum(running_max, block_max)
        old_scale = tl.exp(running_max - new_max)
        probabilities = tl.where(
            token_mask,
            tl.exp(scores - new_max),
            0.0,
        )
        values = tl.load(
            value_cache_ptr + cache_offsets,
            mask=matrix_mask,
            other=0.0,
        ).to(tl.float32)
        accumulator = (
            accumulator * old_scale
            + tl.sum(probabilities[:, None] * values, axis=0)
        )
        running_sum = (
            running_sum * old_scale + tl.sum(probabilities, axis=0)
        )
        running_max = new_max
        token_start += BLOCK_N

    tl.store(
        output_ptr + query_base + dimensions,
        accumulator / running_sum,
        mask=dimension_mask,
    )


@triton.jit
def _paged_decode_split_kernel(
    query_ptr,
    key_cache_ptr,
    value_cache_ptr,
    block_tables_ptr,
    sequence_lengths_ptr,
    partial_max_ptr,
    partial_sum_ptr,
    partial_output_ptr,
    max_blocks,
    page_size: tl.constexpr,
    num_query_heads: tl.constexpr,
    num_kv_heads: tl.constexpr,
    head_dim: tl.constexpr,
    query_per_kv: tl.constexpr,
    num_splits: tl.constexpr,
    scale: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    batch_index = tl.program_id(0)
    query_head = tl.program_id(1)
    split_index = tl.program_id(2)
    kv_head = query_head // query_per_kv
    sequence_length = tl.load(sequence_lengths_ptr + batch_index)
    tokens_per_split = (sequence_length + num_splits - 1) // num_splits
    split_start = split_index * tokens_per_split
    split_end = tl.minimum(split_start + tokens_per_split, sequence_length)
    dimensions = tl.arange(0, BLOCK_D)
    dimension_mask = dimensions < head_dim
    query_base = (
        batch_index * num_query_heads * head_dim + query_head * head_dim
    )
    query = tl.load(
        query_ptr + query_base + dimensions,
        mask=dimension_mask,
        other=0.0,
    ).to(tl.float32)

    running_max = tl.full((), -float("inf"), tl.float32)
    running_sum = tl.zeros((), tl.float32)
    accumulator = tl.zeros((BLOCK_D,), tl.float32)
    token_start = split_start
    while token_start < split_end:
        token_offsets = token_start + tl.arange(0, BLOCK_N)
        token_mask = token_offsets < split_end
        logical_pages = token_offsets // page_size
        page_offsets = token_offsets % page_size
        physical_pages = tl.load(
            block_tables_ptr + batch_index * max_blocks + logical_pages,
            mask=token_mask,
            other=0,
        )
        slots = physical_pages * page_size + page_offsets
        cache_offsets = (
            slots[:, None] * num_kv_heads * head_dim
            + kv_head * head_dim
            + dimensions[None, :]
        )
        matrix_mask = token_mask[:, None] & dimension_mask[None, :]
        keys = tl.load(
            key_cache_ptr + cache_offsets,
            mask=matrix_mask,
            other=0.0,
        ).to(tl.float32)
        scores = tl.sum(keys * query[None, :], axis=1) * scale
        scores = tl.where(token_mask, scores, -float("inf"))
        block_max = tl.max(scores, axis=0)
        new_max = tl.maximum(running_max, block_max)
        old_scale = tl.exp(running_max - new_max)
        probabilities = tl.exp(scores - new_max)
        probabilities = tl.where(token_mask, probabilities, 0.0)
        values = tl.load(
            value_cache_ptr + cache_offsets,
            mask=matrix_mask,
            other=0.0,
        ).to(tl.float32)
        accumulator = (
            accumulator * old_scale
            + tl.sum(probabilities[:, None] * values, axis=0)
        )
        running_sum = running_sum * old_scale + tl.sum(probabilities, axis=0)
        running_max = new_max
        token_start += BLOCK_N

    partial_index = (
        (batch_index * num_query_heads + query_head) * num_splits
        + split_index
    )
    tl.store(partial_max_ptr + partial_index, running_max)
    tl.store(partial_sum_ptr + partial_index, running_sum)
    tl.store(
        partial_output_ptr + partial_index * head_dim + dimensions,
        accumulator,
        mask=dimension_mask,
    )


@triton.jit
def _paged_decode_reduce_kernel(
    partial_max_ptr,
    partial_sum_ptr,
    partial_output_ptr,
    output_ptr,
    num_query_heads: tl.constexpr,
    head_dim: tl.constexpr,
    num_splits: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_S: tl.constexpr,
):
    batch_index = tl.program_id(0)
    query_head = tl.program_id(1)
    split_offsets = tl.arange(0, BLOCK_S)
    split_mask = split_offsets < num_splits
    partial_base = (
        (batch_index * num_query_heads + query_head) * num_splits
    )
    partial_max = tl.load(
        partial_max_ptr + partial_base + split_offsets,
        mask=split_mask,
        other=-float("inf"),
    ).to(tl.float32)
    global_max = tl.max(partial_max, axis=0)
    rescale = tl.exp(partial_max - global_max)
    partial_sum = tl.load(
        partial_sum_ptr + partial_base + split_offsets,
        mask=split_mask,
        other=0.0,
    ).to(tl.float32)
    denominator = tl.sum(partial_sum * rescale, axis=0)

    dimensions = tl.arange(0, BLOCK_D)
    dimension_mask = dimensions < head_dim
    partial_offsets = (
        (partial_base + split_offsets[:, None]) * head_dim
        + dimensions[None, :]
    )
    partial_values = tl.load(
        partial_output_ptr + partial_offsets,
        mask=split_mask[:, None] & dimension_mask[None, :],
        other=0.0,
    ).to(tl.float32)
    numerator = tl.sum(partial_values * rescale[:, None], axis=0)
    output_base = (
        batch_index * num_query_heads * head_dim + query_head * head_dim
    )
    tl.store(
        output_ptr + output_base + dimensions,
        numerator / denominator,
        mask=dimension_mask,
    )


def paged_decode_attention(
    query: tensorplay.Tensor,
    key_cache: tensorplay.Tensor,
    value_cache: tensorplay.Tensor,
    block_tables: tensorplay.Tensor,
    sequence_lengths: tensorplay.Tensor,
    page_size: int,
    *,
    output: tensorplay.Tensor | None = None,
) -> tensorplay.Tensor:
    query = query.contiguous()
    block_tables = block_tables.contiguous()
    sequence_lengths = sequence_lengths.contiguous()
    if output is None:
        output = tensorplay.empty_like(query)
    elif (
        output.shape != query.shape
        or output.device != query.device
        or output.dtype != query.dtype
        or not output.is_contiguous()
    ):
        raise ValueError("paged decode output must match contiguous query")
    num_query_heads = query.shape[1]
    num_kv_heads = key_cache.shape[-2]
    head_dim = query.shape[-1]
    block_d = triton.next_power_of_2(head_dim)
    max_sequence_length = block_tables.shape[1] * page_size
    if max_sequence_length <= 512:
        # B1 has only one CTA per query head.  A full-context tile avoids
        # repeatedly serializing the online-softmax loop for the common
        # 128--256 token latency workload.  Wider batches retain the smaller
        # tile to preserve occupancy.
        if query.shape[0] == 1 and max_sequence_length > 128:
            block_n = 256
            num_warps = 8
        elif query.shape[0] <= 16 and max_sequence_length > 128:
            block_n = 256
            num_warps = 4
        elif query.shape[0] == 1 and max_sequence_length > 64:
            block_n = 128
            num_warps = 2
        else:
            block_n = 64
            num_warps = 2
        _paged_decode_single_kernel[(query.shape[0], num_query_heads)](
            query,
            key_cache,
            value_cache,
            block_tables,
            sequence_lengths,
            output,
            block_tables.shape[1],
            page_size=page_size,
            num_query_heads=num_query_heads,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            query_per_kv=num_query_heads // num_kv_heads,
            scale=head_dim**-0.5,
            BLOCK_D=block_d,
            BLOCK_N=block_n,
            num_warps=num_warps,
        )
        return output
    num_splits = 8
    partial_shape = (query.shape[0], num_query_heads, num_splits)
    partial_max = tensorplay.empty(
        partial_shape,
        dtype=tensorplay.float32,
        device=query.device,
    )
    partial_sum = tensorplay.empty_like(partial_max)
    partial_output = tensorplay.empty(
        (*partial_shape, head_dim),
        dtype=tensorplay.float32,
        device=query.device,
    )
    _paged_decode_split_kernel[
        (query.shape[0], num_query_heads, num_splits)
    ](
        query,
        key_cache,
        value_cache,
        block_tables,
        sequence_lengths,
        partial_max,
        partial_sum,
        partial_output,
        block_tables.shape[1],
        page_size=page_size,
        num_query_heads=num_query_heads,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        query_per_kv=num_query_heads // num_kv_heads,
        num_splits=num_splits,
        scale=head_dim**-0.5,
        BLOCK_D=block_d,
        BLOCK_N=32,
        num_warps=4,
    )
    _paged_decode_reduce_kernel[(query.shape[0], num_query_heads)](
        partial_max,
        partial_sum,
        partial_output,
        output,
        num_query_heads=num_query_heads,
        head_dim=head_dim,
        num_splits=num_splits,
        BLOCK_D=block_d,
        BLOCK_S=triton.next_power_of_2(num_splits),
        num_warps=4,
    )
    return output


