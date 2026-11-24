"""Grouped-query decode attention: one program per K/V head.

The per-query-head decode kernel re-reads the K/V cache once per query head, so
a 4:1 GQA model streams the cache four times.  At batch 8 with a 520-token
context that is ~0.6 GB of avoidable traffic per step.

This kernel serves every query head of one K/V head inside one program, so the
cache is read once, and it uses tensor-core dots for the score and value
products (an explicit ``[tokens, heads, dim]`` multiply would materialise a
16 K-element tile and spill).  Split-K over the context keeps the launch wide
enough for small batches; :func:`_gqa_reduce_kernel` merges the FP32 partials.

When the split count is one there is nothing to merge.  :func:`_gqa_direct_decode_kernel`
normalises and stores the result inside the attention kernel, removing the
partial workspace, its two full-memory passes and one launch.  Keeping it as a
separate ``@triton.jit`` function (rather than a ``DIRECT`` constexpr branch in
the split kernel) avoids inflating the split kernel's register footprint.

Only the plain (unquantised) bf16/fp16 cache uses this path.
"""

from __future__ import annotations

import tensorplay
import triton
import triton.language as tl

__all__ = ["gqa_decode_attention", "gqa_decode_supported"]


@triton.jit
def _gqa_decode_kernel(
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
    head_dim: tl.constexpr,
    QPKV: tl.constexpr,
    QPKV_PAD: tl.constexpr,
    scale: tl.constexpr,
    num_splits: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_N: tl.constexpr,
    stride_slot,
    stride_head,
):
    batch_index = tl.program_id(0)
    kv_head = tl.program_id(1)
    split_index = tl.program_id(2)

    sequence_length = tl.load(sequence_lengths_ptr + batch_index)
    dimensions = tl.arange(0, BLOCK_D)
    dimension_mask = dimensions < head_dim
    query_offsets = tl.arange(0, QPKV_PAD)
    query_mask = query_offsets < QPKV

    query_head = kv_head * QPKV + query_offsets
    query = tl.load(
        query_ptr
        + (batch_index * num_query_heads + query_head[:, None]) * head_dim
        + dimensions[None, :],
        mask=query_mask[:, None] & dimension_mask[None, :],
        other=0.0,
    )

    running_max = tl.full((QPKV_PAD,), -float("inf"), tl.float32)
    running_sum = tl.zeros((QPKV_PAD,), tl.float32)
    accumulator = tl.zeros((QPKV_PAD, BLOCK_D), tl.float32)

    tokens_per_split = (sequence_length + num_splits - 1) // num_splits
    split_start = split_index * tokens_per_split
    split_end = tl.minimum(split_start + tokens_per_split, sequence_length)
    token_start = split_start
    while token_start < split_end:
        token_offsets = token_start + tl.arange(0, BLOCK_N)
        token_mask = token_offsets < split_end
        logical_pages = token_offsets // page_size
        slot_offsets = token_offsets % page_size
        physical_pages = tl.load(
            block_tables_ptr + batch_index * max_blocks + logical_pages,
            mask=token_mask,
            other=0,
        )
        slots = physical_pages * page_size + slot_offsets
        matrix_mask = token_mask[:, None] & dimension_mask[None, :]
        slot_base = slots[:, None] * stride_slot + kv_head * stride_head
        keys = tl.load(
            key_cache_ptr + slot_base + dimensions[None, :],
            mask=matrix_mask,
            other=0.0,
        )
        scores = tl.dot(query, tl.trans(keys)) * scale
        scores = tl.where(token_mask[None, :], scores, -float("inf"))
        block_max = tl.max(scores, axis=1)
        new_max = tl.maximum(running_max, block_max)
        correction = tl.exp(running_max - new_max)
        probabilities = tl.exp(scores - new_max[:, None])
        values = tl.load(
            value_cache_ptr + slot_base + dimensions[None, :],
            mask=matrix_mask,
            other=0.0,
        )
        accumulator = accumulator * correction[:, None] + tl.dot(
            probabilities.to(values.dtype), values
        )
        running_sum = running_sum * correction + tl.sum(probabilities, axis=1)
        running_max = new_max
        token_start += BLOCK_N

    partial_index = (
        (batch_index * tl.num_programs(1) + kv_head) * QPKV_PAD
        + query_offsets
    )
    partial_offsets = (
        split_index * tl.num_programs(0) * tl.num_programs(1) * QPKV_PAD
        + partial_index
    )
    tl.store(partial_max_ptr + partial_offsets, running_max, mask=query_mask)
    tl.store(partial_sum_ptr + partial_offsets, running_sum, mask=query_mask)
    tl.store(
        partial_output_ptr
        + partial_offsets[:, None] * BLOCK_D
        + dimensions[None, :],
        accumulator,
        mask=query_mask[:, None] & dimension_mask[None, :],
    )


@triton.jit
def _gqa_direct_decode_kernel(
    query_ptr,
    key_cache_ptr,
    value_cache_ptr,
    block_tables_ptr,
    sequence_lengths_ptr,
    output_ptr,
    max_blocks,
    page_size: tl.constexpr,
    num_query_heads: tl.constexpr,
    head_dim: tl.constexpr,
    QPKV: tl.constexpr,
    QPKV_PAD: tl.constexpr,
    scale: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_N: tl.constexpr,
    stride_slot,
    stride_head,
):
    """Same attention as ``_gqa_decode_kernel`` but writes final output.

    Used only when a single split covers the sequence, so online-softmax state
    is already the global one and no reduce launch is needed.
    """

    batch_index = tl.program_id(0)
    kv_head = tl.program_id(1)

    sequence_length = tl.load(sequence_lengths_ptr + batch_index)
    dimensions = tl.arange(0, BLOCK_D)
    dimension_mask = dimensions < head_dim
    query_offsets = tl.arange(0, QPKV_PAD)
    query_mask = query_offsets < QPKV

    query_head = kv_head * QPKV + query_offsets
    query = tl.load(
        query_ptr
        + (batch_index * num_query_heads + query_head[:, None]) * head_dim
        + dimensions[None, :],
        mask=query_mask[:, None] & dimension_mask[None, :],
        other=0.0,
    )

    running_max = tl.full((QPKV_PAD,), -float("inf"), tl.float32)
    running_sum = tl.zeros((QPKV_PAD,), tl.float32)
    accumulator = tl.zeros((QPKV_PAD, BLOCK_D), tl.float32)

    token_start = 0
    while token_start < sequence_length:
        token_offsets = token_start + tl.arange(0, BLOCK_N)
        token_mask = token_offsets < sequence_length
        logical_pages = token_offsets // page_size
        slot_offsets = token_offsets % page_size
        physical_pages = tl.load(
            block_tables_ptr + batch_index * max_blocks + logical_pages,
            mask=token_mask,
            other=0,
        )
        slots = physical_pages * page_size + slot_offsets
        matrix_mask = token_mask[:, None] & dimension_mask[None, :]
        slot_base = slots[:, None] * stride_slot + kv_head * stride_head
        keys = tl.load(
            key_cache_ptr + slot_base + dimensions[None, :],
            mask=matrix_mask,
            other=0.0,
        )
        scores = tl.dot(query, tl.trans(keys)) * scale
        scores = tl.where(token_mask[None, :], scores, -float("inf"))
        block_max = tl.max(scores, axis=1)
        new_max = tl.maximum(running_max, block_max)
        correction = tl.exp(running_max - new_max)
        probabilities = tl.exp(scores - new_max[:, None])
        values = tl.load(
            value_cache_ptr + slot_base + dimensions[None, :],
            mask=matrix_mask,
            other=0.0,
        )
        accumulator = accumulator * correction[:, None] + tl.dot(
            probabilities.to(values.dtype), values
        )
        running_sum = running_sum * correction + tl.sum(probabilities, axis=1)
        running_max = new_max
        token_start += BLOCK_N

    # Reciprocal per row (16 values) then a multiply over the tile avoids a
    # full-tile fp32 divide and keeps the register footprint small.
    denominator = tl.where(running_sum > 0.0, running_sum, 1.0)
    inv_denominator = 1.0 / denominator
    result = accumulator * inv_denominator[:, None]
    tl.store(
        output_ptr
        + (batch_index * num_query_heads + query_head[:, None]) * head_dim
        + dimensions[None, :],
        result.to(output_ptr.dtype.element_ty),
        mask=query_mask[:, None] & dimension_mask[None, :],
    )


@triton.jit
def _gqa_reduce_kernel(
    partial_max_ptr,
    partial_sum_ptr,
    partial_output_ptr,
    output_ptr,
    batch_size,
    num_kv_heads: tl.constexpr,
    num_query_heads: tl.constexpr,
    head_dim: tl.constexpr,
    QPKV: tl.constexpr,
    QPKV_PAD: tl.constexpr,
    num_splits: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_S: tl.constexpr,
):
    batch_index = tl.program_id(0)
    kv_head = tl.program_id(1)
    dimensions = tl.arange(0, BLOCK_D)
    dimension_mask = dimensions < head_dim
    query_offsets = tl.arange(0, QPKV_PAD)
    query_mask = query_offsets < QPKV
    split_offsets = tl.arange(0, BLOCK_S)
    split_mask = split_offsets < num_splits

    total = num_kv_heads * QPKV_PAD
    base = (batch_index * num_kv_heads + kv_head) * QPKV_PAD + query_offsets
    indices = split_offsets[:, None] * batch_size * total + base[None, :]

    maxima = tl.load(
        partial_max_ptr + indices,
        mask=split_mask[:, None] & query_mask[None, :],
        other=-float("inf"),
    )
    global_max = tl.max(maxima, axis=0)
    weights = tl.exp(maxima - global_max[None, :])
    weights = tl.where(split_mask[:, None], weights, 0.0)
    sums = tl.load(
        partial_sum_ptr + indices,
        mask=split_mask[:, None] & query_mask[None, :],
        other=0.0,
    )
    denominators = tl.sum(weights * sums, axis=0)
    values = tl.load(
        partial_output_ptr
        + indices[:, :, None] * BLOCK_D
        + dimensions[None, None, :],
        mask=split_mask[:, None, None]
        & query_mask[None, :, None]
        & dimension_mask[None, None, :],
        other=0.0,
    )
    numerator = tl.sum(weights[:, :, None] * values, axis=0)
    result = numerator / denominators[:, None]

    query_head = kv_head * QPKV + query_offsets
    tl.store(
        output_ptr
        + (batch_index * num_query_heads + query_head[:, None]) * head_dim
        + dimensions[None, :],
        result.to(output_ptr.dtype.element_ty),
        mask=query_mask[:, None] & dimension_mask[None, :],
    )


def gqa_decode_supported(
    query: tensorplay.Tensor,
    key_cache: tensorplay.Tensor,
    num_kv_heads: int,
    quant: object | None,
) -> bool:
    """Whether the grouped-query kernel covers this cache configuration."""

    if quant is not None:
        return False
    if key_cache.dtype not in {tensorplay.bfloat16, tensorplay.float16}:
        return False
    if int(query.shape[1]) % num_kv_heads:
        return False
    head_dim = int(query.shape[-1])
    return head_dim in {64, 128, 256} and int(query.shape[1]) > num_kv_heads


def gqa_decode_attention(
    query: tensorplay.Tensor,
    key_cache: tensorplay.Tensor,
    value_cache: tensorplay.Tensor,
    block_tables: tensorplay.Tensor,
    sequence_lengths: tensorplay.Tensor,
    page_size: int,
    *,
    output: tensorplay.Tensor,
    scale: float,
) -> None:
    batch = int(query.shape[0])
    num_query_heads = int(query.shape[1])
    num_kv_heads = int(key_cache.shape[-2])
    head_dim = int(query.shape[-1])
    qpkv = num_query_heads // num_kv_heads
    # tl.dot needs 16 rows; the surplus rows are masked on store.
    qpkv_pad = max(16, triton.next_power_of_2(qpkv))
    block_d = triton.next_power_of_2(head_dim)
    programs = batch * num_kv_heads
    # Split-K exists to fill the GPU when there are few (batch x kv-head)
    # programs; with enough of them it only adds a partial buffer and a second
    # launch.
    splits = 1
    if programs < 128:
        splits = min(8, triton.next_power_of_2(max(1, 256 // programs)))
    block_n = 64 if block_d * qpkv_pad <= 2048 else 32
    # A [pages, page_size, kv, dim] cache addresses slots through stride(1);
    # a flattened [slots, kv, dim] view uses stride(0).
    stride_slot = key_cache.stride(1) if key_cache.dim() == 4 else key_cache.stride(0)
    stride_head = key_cache.stride(-2)

    common = {
        "max_blocks": block_tables.shape[1],
        "page_size": page_size,
        "num_query_heads": num_query_heads,
        "head_dim": head_dim,
        "QPKV": qpkv,
        "QPKV_PAD": qpkv_pad,
        "scale": scale,
        "BLOCK_D": block_d,
        "BLOCK_N": block_n,
        "stride_slot": stride_slot,
        "stride_head": stride_head,
    }
    grid = (batch, num_kv_heads)

    if splits == 1:
        # No reduce needed: write final output from the attention kernel.
        # With >= 256 CTAs the 4-warp variant becomes occupancy-limited;
        # 2 warps double resident CTAs/SM and win measurably on RTX 3090
        # (35.6us vs 45.6us at batch 32, head_dim 128).  Keep the larger tile
        # configurations on 4 warps to avoid register spills.
        direct_warps = 2 if programs >= 256 and qpkv_pad * block_d <= 2048 else 4
        _gqa_direct_decode_kernel[grid](
            query,
            key_cache,
            value_cache,
            block_tables,
            sequence_lengths,
            output,
            num_warps=direct_warps,
            **common,
        )
        return

    slot_count = splits * batch * num_kv_heads * qpkv_pad
    partial_max = tensorplay.empty(
        (slot_count,), dtype=tensorplay.float32, device=query.device
    )
    partial_sum = tensorplay.empty_like(partial_max)
    partial_output = tensorplay.empty(
        (slot_count, block_d), dtype=tensorplay.float32, device=query.device
    )

    _gqa_decode_kernel[(batch, num_kv_heads, splits)](
        query,
        key_cache,
        value_cache,
        block_tables,
        sequence_lengths,
        partial_max,
        partial_sum,
        partial_output,
        num_warps=4,
        num_splits=splits,
        **common,
    )
    _gqa_reduce_kernel[grid](
        partial_max,
        partial_sum,
        partial_output,
        output,
        batch,
        num_kv_heads=num_kv_heads,
        num_query_heads=num_query_heads,
        head_dim=head_dim,
        QPKV=qpkv,
        QPKV_PAD=qpkv_pad,
        num_splits=splits,
        BLOCK_D=block_d,
        BLOCK_S=triton.next_power_of_2(splits),
        num_warps=4,
    )
