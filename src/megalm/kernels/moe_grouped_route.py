from __future__ import annotations

import tensorplay
import triton
import triton.language as tl

@triton.jit(do_not_specialize=["token_capacity"])
def _moe_route_scatter_kernel(
    hidden_ptr,
    expert_ids_ptr,
    route_weights_ptr,
    expert_counts_ptr,
    sorted_hidden_ptr,
    sorted_indices_ptr,
    sorted_weights_ptr,
    hidden_size: tl.constexpr,
    token_capacity,
    BLOCK_H: tl.constexpr,
):
    token_index = tl.program_id(0)
    expert_index = tl.load(expert_ids_ptr + token_index)
    expert_slot = tl.atomic_add(expert_counts_ptr + expert_index, 1)
    sorted_row = expert_index * token_capacity + expert_slot
    hidden_offsets = tl.arange(0, BLOCK_H)
    hidden_mask = hidden_offsets < hidden_size
    values = tl.load(
        hidden_ptr + token_index * hidden_size + hidden_offsets,
        mask=hidden_mask,
        other=0.0,
    )
    tl.store(
        sorted_hidden_ptr + sorted_row * hidden_size + hidden_offsets,
        values,
        mask=hidden_mask,
    )
    tl.store(sorted_indices_ptr + sorted_row, token_index)
    route_weight = tl.load(route_weights_ptr + token_index)
    tl.store(sorted_weights_ptr + sorted_row, route_weight)


@triton.jit(do_not_specialize=["token_capacity"])
def _moe_topk_route_scatter_kernel(
    hidden_ptr,
    expert_ids_ptr,
    route_weights_ptr,
    expert_counts_ptr,
    sorted_hidden_ptr,
    sorted_indices_ptr,
    sorted_weights_ptr,
    hidden_size: tl.constexpr,
    token_capacity,
    TOP_K: tl.constexpr,
    BLOCK_H: tl.constexpr,
):
    """Scatter flattened ``[token, slot]`` routes grouped by expert.

    ``sorted_indices`` stores the flat route id (``token * TOP_K + slot``),
    which lets the down kernel write one partial result per route and a later
    reduction sum the ``TOP_K`` contributions of each token deterministically.
    """

    route_index = tl.program_id(0)
    token_index = route_index // TOP_K
    expert_index = tl.load(expert_ids_ptr + route_index)
    expert_slot = tl.atomic_add(expert_counts_ptr + expert_index, 1)
    sorted_row = expert_index * token_capacity + expert_slot
    hidden_offsets = tl.arange(0, BLOCK_H)
    hidden_mask = hidden_offsets < hidden_size
    values = tl.load(
        hidden_ptr + token_index * hidden_size + hidden_offsets,
        mask=hidden_mask,
        other=0.0,
    )
    tl.store(
        sorted_hidden_ptr + sorted_row * hidden_size + hidden_offsets,
        values,
        mask=hidden_mask,
    )
    tl.store(sorted_indices_ptr + sorted_row, route_index)
    route_weight = tl.load(route_weights_ptr + route_index)
    tl.store(sorted_weights_ptr + sorted_row, route_weight)


@triton.jit(do_not_specialize=["token_capacity"])
def _moe_top1_route_scatter_from_hidden_kernel(
    hidden_ptr,
    gate_weight_ptr,
    expert_counts_ptr,
    sorted_hidden_ptr,
    sorted_indices_ptr,
    sorted_weights_ptr,
    token_capacity,
    hidden_size: tl.constexpr,
    num_experts: tl.constexpr,
    normalize_top1: tl.constexpr,
    BLOCK_H: tl.constexpr,
    BLOCK_E: tl.constexpr,
    BLOCK_COPY_H: tl.constexpr,
):
    token_index = tl.program_id(0)
    expert_offsets = tl.arange(0, BLOCK_E)
    expert_mask = expert_offsets < num_experts
    logits = tl.zeros((BLOCK_E,), tl.float32)
    hidden_offsets = tl.arange(0, BLOCK_H)
    hidden_start = 0
    while hidden_start < hidden_size:
        current_hidden = hidden_start + hidden_offsets
        hidden_mask = current_hidden < hidden_size
        hidden = tl.load(
            hidden_ptr + token_index * hidden_size + current_hidden,
            mask=hidden_mask,
            other=0.0,
        ).to(tl.float32)
        gate_offsets = (
            expert_offsets[:, None] * hidden_size
            + current_hidden[None, :]
        )
        gate = tl.load(
            gate_weight_ptr + gate_offsets,
            mask=expert_mask[:, None] & hidden_mask[None, :],
            other=0.0,
        ).to(tl.float32)
        logits += tl.sum(gate * hidden[None, :], axis=1)
        hidden_start += BLOCK_H
    logits = logits.to(hidden_ptr.dtype.element_ty).to(tl.float32)
    logits = tl.where(expert_mask, logits, -float("inf"))
    expert_index = tl.argmax(logits, axis=0)
    if normalize_top1:
        route_weight = 1.0
    else:
        maximum = tl.max(logits, axis=0)
        denominator = tl.sum(
            tl.where(expert_mask, tl.exp(logits - maximum), 0.0),
            axis=0,
        )
        route_weight = 1.0 / denominator

    expert_slot = tl.atomic_add(expert_counts_ptr + expert_index, 1)
    sorted_row = expert_index * token_capacity + expert_slot
    copy_offsets = tl.arange(0, BLOCK_COPY_H)
    copy_mask = copy_offsets < hidden_size
    values = tl.load(
        hidden_ptr + token_index * hidden_size + copy_offsets,
        mask=copy_mask,
        other=0.0,
    )
    tl.store(
        sorted_hidden_ptr + sorted_row * hidden_size + copy_offsets,
        values,
        mask=copy_mask,
    )
    tl.store(sorted_indices_ptr + sorted_row, token_index)
    if not normalize_top1:
        tl.store(sorted_weights_ptr + sorted_row, route_weight)


@triton.jit(do_not_specialize=["token_count"])
def _moe_route_count_kernel(
    expert_ids_ptr,
    expert_counts_ptr,
    token_count,
    BLOCK: tl.constexpr,
):
    token_offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    token_mask = token_offsets < token_count
    expert_ids = tl.load(
        expert_ids_ptr + token_offsets,
        mask=token_mask,
        other=0,
    )
    tl.atomic_add(
        expert_counts_ptr + expert_ids,
        1,
        mask=token_mask,
    )


@triton.jit
def _moe_compact_route_scatter_kernel(
    expert_ids_ptr,
    route_weights_ptr,
    expert_counts_ptr,
    expert_cursors_ptr,
    sorted_indices_ptr,
    sorted_weights_ptr,
    num_experts: tl.constexpr,
    EXPERT_BLOCK: tl.constexpr,
):
    token_index = tl.program_id(0)
    expert_index = tl.load(expert_ids_ptr + token_index)
    expert_offsets = tl.arange(0, EXPERT_BLOCK)
    expert_mask = expert_offsets < num_experts
    expert_counts = tl.load(
        expert_counts_ptr + expert_offsets,
        mask=expert_mask,
        other=0,
    )
    expert_start = tl.sum(
        tl.where(
            expert_mask & (expert_offsets < expert_index),
            expert_counts,
            0,
        ),
        axis=0,
    )
    expert_slot = tl.atomic_add(expert_cursors_ptr + expert_index, 1)
    sorted_row = expert_start + expert_slot
    tl.store(sorted_indices_ptr + sorted_row, token_index)
    route_weight = tl.load(route_weights_ptr + token_index)
    tl.store(sorted_weights_ptr + sorted_row, route_weight)


