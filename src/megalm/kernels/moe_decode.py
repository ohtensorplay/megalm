from __future__ import annotations

import tensorplay
import triton
import triton.language as tl

@triton.jit
def _routed_moe_up_kernel(
    hidden_ptr,
    gate_up_weight_ptr,
    expert_ids_ptr,
    route_weights_ptr,
    intermediate_ptr,
    hidden_size: tl.constexpr,
    intermediate_size: tl.constexpr,
    top_k: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    token_route = tl.program_id(0)
    block_index = tl.program_id(1)
    token_index = token_route // top_k
    route_index = token_route % top_k
    expert_index = tl.load(expert_ids_ptr + token_route)
    route_weight = tl.load(route_weights_ptr + token_route).to(tl.float32)
    output_offsets = block_index * BLOCK_N + tl.arange(0, BLOCK_N)
    output_mask = output_offsets < intermediate_size
    gate_accumulator = tl.zeros((BLOCK_N,), tl.float32)
    up_accumulator = tl.zeros((BLOCK_N,), tl.float32)
    hidden_offsets = tl.arange(0, BLOCK_K)
    hidden_start = 0
    while hidden_start < hidden_size:
        current_hidden = hidden_start + hidden_offsets
        hidden_mask = current_hidden < hidden_size
        values = tl.load(
            hidden_ptr + token_index * hidden_size + current_hidden,
            mask=hidden_mask,
            other=0.0,
        ).to(tl.float32)
        expert_base = (
            expert_index * 2 * intermediate_size * hidden_size
        )
        gate_offsets = (
            expert_base
            + output_offsets[:, None] * hidden_size
            + current_hidden[None, :]
        )
        up_offsets = (
            expert_base
            + (intermediate_size + output_offsets[:, None]) * hidden_size
            + current_hidden[None, :]
        )
        weight_mask = output_mask[:, None] & hidden_mask[None, :]
        gate_weight = tl.load(
            gate_up_weight_ptr + gate_offsets,
            mask=weight_mask,
            other=0.0,
        ).to(tl.float32)
        up_weight = tl.load(
            gate_up_weight_ptr + up_offsets,
            mask=weight_mask,
            other=0.0,
        ).to(tl.float32)
        gate_accumulator += tl.sum(gate_weight * values[None, :], axis=1)
        up_accumulator += tl.sum(up_weight * values[None, :], axis=1)
        hidden_start += BLOCK_K
    activated = (
        gate_accumulator
        * tl.sigmoid(gate_accumulator)
        * up_accumulator
        * route_weight
    )
    tl.store(
        intermediate_ptr
        + token_route * intermediate_size
        + output_offsets,
        activated,
        mask=output_mask,
    )


@triton.jit
def _routed_moe_down_kernel(
    intermediate_ptr,
    down_weight_ptr,
    expert_ids_ptr,
    output_ptr,
    hidden_size: tl.constexpr,
    intermediate_size: tl.constexpr,
    top_k: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    token_index = tl.program_id(0)
    block_index = tl.program_id(1)
    output_offsets = block_index * BLOCK_N + tl.arange(0, BLOCK_N)
    output_mask = output_offsets < hidden_size
    output_accumulator = tl.zeros((BLOCK_N,), tl.float32)
    intermediate_offsets = tl.arange(0, BLOCK_K)
    for route_index in range(top_k):
        token_route = token_index * top_k + route_index
        expert_index = tl.load(expert_ids_ptr + token_route)
        intermediate_start = 0
        while intermediate_start < intermediate_size:
            current_intermediate = intermediate_start + intermediate_offsets
            intermediate_mask = current_intermediate < intermediate_size
            values = tl.load(
                intermediate_ptr
                + token_route * intermediate_size
                + current_intermediate,
                mask=intermediate_mask,
                other=0.0,
            ).to(tl.float32)
            weight_offsets = (
                expert_index * hidden_size * intermediate_size
                + output_offsets[:, None] * intermediate_size
                + current_intermediate[None, :]
            )
            weight_mask = output_mask[:, None] & intermediate_mask[None, :]
            weight = tl.load(
                down_weight_ptr + weight_offsets,
                mask=weight_mask,
                other=0.0,
            ).to(tl.float32)
            output_accumulator += tl.sum(weight * values[None, :], axis=1)
            intermediate_start += BLOCK_K
    tl.store(
        output_ptr + token_index * hidden_size + output_offsets,
        output_accumulator,
        mask=output_mask,
    )


def routed_moe_decode(
    hidden_states: tensorplay.Tensor,
    gate_up_weight: tensorplay.Tensor,
    down_weight: tensorplay.Tensor,
    topk_ids: tensorplay.Tensor,
    topk_weights: tensorplay.Tensor,
) -> tensorplay.Tensor:
    hidden_states = hidden_states.contiguous()
    topk_ids = topk_ids.contiguous()
    topk_weights = topk_weights.contiguous()
    token_count, hidden_size = hidden_states.shape
    top_k = topk_ids.shape[1]
    intermediate_size = gate_up_weight.shape[1] // 2
    intermediate = tensorplay.empty(
        (token_count, top_k, intermediate_size),
        dtype=hidden_states.dtype,
        device=hidden_states.device,
    )
    # A single decode token needs more reduction work per CTA and narrower
    # output tiles to expose enough independent expert-weight streams.  This
    # is a geometry heuristic, not a device-specific autotune result: it keeps
    # the same kernel on every CUDA backend while avoiding the 24 long-lived
    # down-projection CTAs produced by the generic 32x32 tile.
    single_token = token_count == 1
    up_block_n = 32
    up_block_k = 64 if single_token else 32
    up_num_warps = 8 if single_token else 4
    _routed_moe_up_kernel[
        (
            token_count * top_k,
            triton.cdiv(intermediate_size, up_block_n),
        )
    ](
        hidden_states,
        gate_up_weight,
        topk_ids,
        topk_weights,
        intermediate,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        top_k=top_k,
        BLOCK_N=up_block_n,
        BLOCK_K=up_block_k,
        num_warps=up_num_warps,
    )
    output = tensorplay.empty_like(hidden_states)
    down_block_n = 8 if single_token else 32
    down_block_k = 128 if single_token else 32
    down_num_warps = 8 if single_token else 4
    _routed_moe_down_kernel[
        (token_count, triton.cdiv(hidden_size, down_block_n))
    ](
        intermediate,
        down_weight,
        topk_ids,
        output,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        top_k=top_k,
        BLOCK_N=down_block_n,
        BLOCK_K=down_block_k,
        num_warps=down_num_warps,
    )
    return output


