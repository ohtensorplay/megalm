from __future__ import annotations

import tensorplay
import triton
import triton.language as tl

@triton.jit(do_not_specialize=["token_capacity"])
def _grouped_moe_down_kernel(
    intermediate_ptr,
    down_weight_ptr,
    sorted_indices_ptr,
    expert_counts_ptr,
    output_ptr,
    route_output_ptr,
    token_capacity,
    num_experts: tl.constexpr,
    hidden_size: tl.constexpr,
    intermediate_size: tl.constexpr,
    TOP_K: tl.constexpr,
    STORE_ROUTES: tl.constexpr,
    COMPACT: tl.constexpr,
    DYNAMIC_BLOCKS: tl.constexpr,
    EXPERT_BLOCK: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    if DYNAMIC_BLOCKS:
        global_row_block = tl.program_id(0)
        column_block = tl.program_id(1)
        expert_offsets = tl.arange(0, EXPERT_BLOCK)
        expert_mask = expert_offsets < num_experts
        expert_counts = tl.load(
            expert_counts_ptr + expert_offsets,
            mask=expert_mask,
            other=0,
        )
        expert_blocks = (expert_counts + BLOCK_M - 1) // BLOCK_M
        block_ends = tl.cumsum(expert_blocks, axis=0)
        block_starts = block_ends - expert_blocks
        owns_block = (
            expert_mask
            & (global_row_block >= block_starts)
            & (global_row_block < block_ends)
        )
        expert_index = tl.sum(
            tl.where(owns_block, expert_offsets, 0),
            axis=0,
        )
        row_block = global_row_block - tl.sum(
            tl.where(owns_block, block_starts, 0),
            axis=0,
        )
        expert_count = tl.sum(
            tl.where(owns_block, expert_counts, 0),
            axis=0,
        )
        if COMPACT:
            token_ends = tl.cumsum(expert_counts, axis=0)
            token_starts = token_ends - expert_counts
            sorted_row_base = tl.sum(
                tl.where(owns_block, token_starts, 0),
                axis=0,
            )
        else:
            sorted_row_base = expert_index * token_capacity
        row_start = row_block * BLOCK_M
        active = tl.sum(owns_block.to(tl.int32), axis=0) != 0
    else:
        expert_index = tl.program_id(0)
        row_block = tl.program_id(1)
        column_block = tl.program_id(2)
        expert_count = tl.load(expert_counts_ptr + expert_index)
        sorted_row_base = expert_index * token_capacity
        row_start = row_block * BLOCK_M
        active = row_start < expert_count
    if active:
        row_offsets = row_start + tl.arange(0, BLOCK_M)
        column_offsets = (
            column_block * BLOCK_N + tl.arange(0, BLOCK_N)
        )
        row_mask = row_offsets < expert_count
        column_mask = column_offsets < hidden_size
        accumulator = tl.zeros((BLOCK_M, BLOCK_N), tl.float32)
        k_offsets = tl.arange(0, BLOCK_K)
        k_start = 0
        while k_start < intermediate_size:
            current_k = k_start + k_offsets
            k_mask = current_k < intermediate_size
            intermediate_offsets = (
                (sorted_row_base + row_offsets[:, None])
                * intermediate_size
                + current_k[None, :]
            )
            values = tl.load(
                intermediate_ptr + intermediate_offsets,
                mask=row_mask[:, None] & k_mask[None, :],
                other=0.0,
            )
            weight_offsets = (
                expert_index * hidden_size * intermediate_size
                + column_offsets[None, :] * intermediate_size
                + current_k[:, None]
            )
            weights = tl.load(
                down_weight_ptr + weight_offsets,
                mask=k_mask[:, None] & column_mask[None, :],
                other=0.0,
            )
            accumulator += tl.dot(values, weights)
            k_start += BLOCK_K
        route_indices = tl.load(
            sorted_indices_ptr
            + sorted_row_base
            + row_offsets,
            mask=row_mask,
            other=0,
        )
        if STORE_ROUTES:
            # One partial result per (token, slot) route; a deterministic
            # reduction sums them after this kernel finishes.
            output_offsets = (
                route_indices[:, None] * hidden_size
                + column_offsets[None, :]
            )
            tl.store(
                route_output_ptr + output_offsets,
                accumulator,
                mask=row_mask[:, None] & column_mask[None, :],
            )
        else:
            token_indices = route_indices // TOP_K
            output_offsets = (
                token_indices[:, None] * hidden_size
                + column_offsets[None, :]
            )
            tl.store(
                output_ptr + output_offsets,
                accumulator,
                mask=row_mask[:, None] & column_mask[None, :],
            )


@triton.jit
def _moe_topk_reduce_kernel(
    route_output_ptr,
    output_ptr,
    top_k: tl.constexpr,
    hidden_size: tl.constexpr,
    BLOCK_H: tl.constexpr,
):
    """Sum the ``top_k`` route partials of each token in slot order."""

    token_index = tl.program_id(0)
    hidden_offsets = tl.arange(0, BLOCK_H)
    hidden_mask = hidden_offsets < hidden_size
    accumulator = tl.zeros((BLOCK_H,), tl.float32)
    for slot in range(top_k):
        values = tl.load(
            route_output_ptr
            + (token_index * top_k + slot) * hidden_size
            + hidden_offsets,
            mask=hidden_mask,
            other=0.0,
        )
        accumulator += values.to(tl.float32)
    tl.store(
        output_ptr + token_index * hidden_size + hidden_offsets,
        accumulator.to(output_ptr.dtype.element_ty),
        mask=hidden_mask,
    )


