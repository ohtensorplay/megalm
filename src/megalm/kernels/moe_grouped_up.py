from __future__ import annotations

import tensorplay
import triton
import triton.language as tl

@triton.jit(do_not_specialize=["token_capacity"])
def _grouped_moe_up_kernel(
    hidden_ptr,
    sorted_hidden_ptr,
    sorted_indices_ptr,
    gate_up_weight_ptr,
    sorted_weights_ptr,
    expert_counts_ptr,
    intermediate_ptr,
    token_capacity,
    num_experts: tl.constexpr,
    hidden_size: tl.constexpr,
    intermediate_size: tl.constexpr,
    TOP_K: tl.constexpr,
    COMPACT: tl.constexpr,
    DYNAMIC_BLOCKS: tl.constexpr,
    APPLY_ROUTE_WEIGHT: tl.constexpr,
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
        column_mask = column_offsets < intermediate_size
        gate_accumulator = tl.zeros((BLOCK_M, BLOCK_N), tl.float32)
        up_accumulator = tl.zeros((BLOCK_M, BLOCK_N), tl.float32)
        if COMPACT:
            route_indices = tl.load(
                sorted_indices_ptr + sorted_row_base + row_offsets,
                mask=row_mask,
                other=0,
            )
            if TOP_K > 1:
                original_indices = route_indices // TOP_K
            else:
                original_indices = route_indices
        k_offsets = tl.arange(0, BLOCK_K)
        k_start = 0
        while k_start < hidden_size:
            current_k = k_start + k_offsets
            k_mask = current_k < hidden_size
            if COMPACT:
                hidden_offsets = (
                    original_indices[:, None] * hidden_size
                    + current_k[None, :]
                )
                hidden = tl.load(
                    hidden_ptr + hidden_offsets,
                    mask=row_mask[:, None] & k_mask[None, :],
                    other=0.0,
                )
            else:
                hidden_offsets = (
                    (sorted_row_base + row_offsets[:, None])
                    * hidden_size
                    + current_k[None, :]
                )
                hidden = tl.load(
                    sorted_hidden_ptr + hidden_offsets,
                    mask=row_mask[:, None] & k_mask[None, :],
                    other=0.0,
                )
            expert_base = (
                expert_index * 2 * intermediate_size * hidden_size
            )
            gate_offsets = (
                expert_base
                + column_offsets[None, :] * hidden_size
                + current_k[:, None]
            )
            up_offsets = (
                expert_base
                + (intermediate_size + column_offsets[None, :])
                * hidden_size
                + current_k[:, None]
            )
            weight_mask = k_mask[:, None] & column_mask[None, :]
            gate_weight = tl.load(
                gate_up_weight_ptr + gate_offsets,
                mask=weight_mask,
                other=0.0,
            )
            up_weight = tl.load(
                gate_up_weight_ptr + up_offsets,
                mask=weight_mask,
                other=0.0,
            )
            gate_accumulator += tl.dot(hidden, gate_weight)
            up_accumulator += tl.dot(hidden, up_weight)
            k_start += BLOCK_K
        activated = (
            gate_accumulator
            * tl.sigmoid(gate_accumulator)
            * up_accumulator
        )
        if APPLY_ROUTE_WEIGHT:
            route_weights = tl.load(
                sorted_weights_ptr
                + sorted_row_base
                + row_offsets,
                mask=row_mask,
                other=0.0,
            ).to(tl.float32)
        output_offsets = (
            (sorted_row_base + row_offsets[:, None])
            * intermediate_size
            + column_offsets[None, :]
        )
        if APPLY_ROUTE_WEIGHT:
            tl.store(
                intermediate_ptr + output_offsets,
                activated * route_weights[:, None],
                mask=row_mask[:, None] & column_mask[None, :],
            )
        else:
            tl.store(
                intermediate_ptr + output_offsets,
                activated,
                mask=row_mask[:, None] & column_mask[None, :],
            )


