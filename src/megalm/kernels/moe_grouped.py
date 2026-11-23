from __future__ import annotations

from typing import TYPE_CHECKING

import tensorplay
import triton

from megalm.kernels.moe_grouped_down import (
    _grouped_moe_down_kernel,
    _moe_topk_reduce_kernel,
)
from megalm.kernels.moe_grouped_route import (
    _moe_compact_route_scatter_kernel,
    _moe_route_count_kernel,
    _moe_route_scatter_kernel,
    _moe_top1_route_scatter_from_hidden_kernel,
    _moe_topk_route_scatter_kernel,
)
from megalm.kernels.moe_grouped_up import _grouped_moe_up_kernel

if TYPE_CHECKING:
    from megalm.kernels.ops import GroupedMoEWorkspace

def routed_moe_grouped_tensorcore(
    hidden_states: tensorplay.Tensor,
    gate_up_weight: tensorplay.Tensor,
    down_weight: tensorplay.Tensor,
    topk_ids: tensorplay.Tensor | None,
    topk_weights: tensorplay.Tensor | None,
    *,
    workspace: GroupedMoEWorkspace | None = None,
    router_weight: tensorplay.Tensor | None = None,
    normalize_router: bool = False,
) -> tensorplay.Tensor:
    hidden_states = hidden_states.contiguous()
    token_count, hidden_size = hidden_states.shape
    num_experts = gate_up_weight.shape[0]
    intermediate_size = gate_up_weight.shape[1] // 2
    if down_weight.shape != (num_experts, hidden_size, intermediate_size):
        raise ValueError("grouped MoE weight layout is invalid")
    fused_router = router_weight is not None
    if fused_router:
        assert router_weight is not None
        router_weight = router_weight.contiguous()
        if router_weight.shape != (num_experts, hidden_size):
            raise ValueError("grouped MoE router weight layout is invalid")
        if num_experts <= 0 or num_experts > 16:
            raise ValueError("fused grouped router supports 1--16 experts")
        route_dtype = tensorplay.float32
    else:
        if topk_ids is None or topk_weights is None:
            raise ValueError("grouped MoE requires routing metadata")
        topk_ids = topk_ids.contiguous()
        topk_weights = topk_weights.contiguous()
        if topk_ids.ndim != 2 or not 1 <= topk_ids.shape[1] <= 32:
            raise ValueError("grouped Tensor Core MoE requires 1 <= top-k <= 32")
        if topk_weights.shape != topk_ids.shape:
            raise ValueError("MoE routing ids and weights must align")
        route_dtype = topk_weights.dtype
    top_k = 1 if fused_router else topk_ids.shape[1]
    route_count = token_count * top_k

    if workspace is not None:
        if token_count > workspace.max_tokens:
            raise ValueError("MoE batch exceeds the configured workspace")
        if top_k > workspace.max_top_k:
            raise ValueError("MoE batch exceeds the configured top-k")
        if (
            workspace.num_experts != num_experts
            or workspace.hidden_size != hidden_size
            or workspace.intermediate_size != intermediate_size
        ):
            raise ValueError("MoE workspace geometry does not match the weights")
        if (
            workspace.output.device != hidden_states.device
            or workspace.output.dtype != hidden_states.dtype
            or workspace.sorted_weights.dtype != route_dtype
        ):
            raise ValueError("MoE workspace dtype or device does not match")
        if top_k > 1 and workspace.route_output is None:
            raise ValueError("MoE workspace lacks a route reduction buffer")
        compact = token_count >= workspace.compact_min_tokens
    else:
        compact = token_count >= 512
    if fused_router and compact:
        raise ValueError("fused grouped router requires a noncompact route")
    expert_capacity = route_count if compact else num_experts * route_count
    if workspace is None:
        expert_counts = tensorplay.zeros(
            num_experts,
            dtype=tensorplay.int32,
            device=hidden_states.device,
        )
        sorted_hidden = (
            hidden_states
            if compact
            else tensorplay.empty(
                (expert_capacity, hidden_size),
                dtype=hidden_states.dtype,
                device=hidden_states.device,
            )
        )
        sorted_indices = tensorplay.empty(
            (expert_capacity,),
            dtype=tensorplay.int64,
            device=hidden_states.device,
        )
        sorted_weights = tensorplay.empty(
            (expert_capacity,),
            dtype=route_dtype,
            device=hidden_states.device,
        )
        intermediate = tensorplay.empty(
            (expert_capacity, intermediate_size),
            dtype=hidden_states.dtype,
            device=hidden_states.device,
        )
        output = tensorplay.empty_like(hidden_states)
        route_output = (
            tensorplay.empty(
                (route_count, hidden_size),
                dtype=tensorplay.float32,
                device=hidden_states.device,
            )
            if top_k > 1
            else output
        )
        expert_cursors = (
            tensorplay.zeros_like(expert_counts) if compact else None
        )
    else:
        if expert_capacity > workspace.expert_capacity:
            raise ValueError("MoE route exceeds the configured workspace")
        expert_counts = workspace.expert_counts
        expert_counts.zero_()
        sorted_hidden = (
            hidden_states
            if compact
            else workspace.sorted_hidden[:expert_capacity]
        )
        sorted_indices = workspace.sorted_indices[:expert_capacity]
        sorted_weights = workspace.sorted_weights[:expert_capacity]
        intermediate = workspace.intermediate[:expert_capacity]
        output = workspace.output[:token_count]
        route_output = (
            workspace.route_output[:route_count] if top_k > 1 else output
        )
        expert_cursors = workspace.expert_cursors if compact else None
        if expert_cursors is not None:
            expert_cursors.zero_()

    if not compact and sorted_hidden.shape[0] < expert_capacity:
        raise ValueError("MoE padded route exceeds the configured workspace")

    if compact:
        assert topk_ids is not None
        assert topk_weights is not None
        assert expert_cursors is not None
        count_block = 256
        flat_ids = topk_ids.view(-1)
        flat_weights = topk_weights.view(-1)
        _moe_route_count_kernel[(triton.cdiv(route_count, count_block),)](
            flat_ids,
            expert_counts,
            route_count,
            BLOCK=count_block,
            num_warps=4,
        )
        _moe_compact_route_scatter_kernel[(route_count,)](
            flat_ids,
            flat_weights,
            expert_counts,
            expert_cursors,
            sorted_indices,
            sorted_weights,
            num_experts=num_experts,
            EXPERT_BLOCK=triton.next_power_of_2(num_experts),
            num_warps=1,
        )
    elif fused_router:
        assert router_weight is not None
        _moe_top1_route_scatter_from_hidden_kernel[(token_count,)](
            hidden_states,
            router_weight,
            expert_counts,
            sorted_hidden,
            sorted_indices,
            sorted_weights,
            token_count,
            hidden_size=hidden_size,
            num_experts=num_experts,
            normalize_top1=normalize_router,
            BLOCK_H=min(256, triton.next_power_of_2(hidden_size)),
            BLOCK_E=triton.next_power_of_2(num_experts),
            BLOCK_COPY_H=triton.next_power_of_2(hidden_size),
            num_warps=4,
        )
    elif top_k > 1:
        assert topk_ids is not None
        assert topk_weights is not None
        _moe_topk_route_scatter_kernel[(route_count,)](
            hidden_states,
            topk_ids.view(-1),
            topk_weights.view(-1),
            expert_counts,
            sorted_hidden,
            sorted_indices,
            sorted_weights,
            hidden_size=hidden_size,
            token_capacity=route_count,
            TOP_K=top_k,
            BLOCK_H=triton.next_power_of_2(hidden_size),
            num_warps=4,
        )
    else:
        assert topk_ids is not None
        assert topk_weights is not None
        _moe_route_scatter_kernel[(token_count,)](
            hidden_states,
            topk_ids,
            topk_weights,
            expert_counts,
            sorted_hidden,
            sorted_indices,
            sorted_weights,
            hidden_size=hidden_size,
            token_capacity=token_count,
            BLOCK_H=triton.next_power_of_2(hidden_size),
            num_warps=4,
        )
    expert_block = triton.next_power_of_2(num_experts)
    dynamic_blocks = compact or route_count > 16
    apply_route_weight = not (fused_router and normalize_router)
    if compact:
        large_compact = route_count > 4096
        up_block_m = 128 if large_compact else 64
        up_block_n = 64 if large_compact else 128
        up_block_k = 64
        up_warps = 4
        up_stages = 3
        down_block_m = 64
        down_block_n = 128
        down_block_k = 32
        down_warps = 4
        down_stages = 3
    else:
        up_block_m = 16
        up_block_n = 64 if route_count <= 16 else 32
        up_block_k = 64
        up_warps = 8 if route_count <= 16 else 4
        up_stages = 3
        down_block_m = 16 if route_count <= 64 else 32
        down_block_n = 32
        down_block_k = 128
        down_warps = 4
        down_stages = 3
    if dynamic_blocks:
        up_grid = (
            triton.cdiv(route_count, up_block_m) + num_experts - 1,
            triton.cdiv(intermediate_size, up_block_n),
        )
        down_grid = (
            triton.cdiv(route_count, down_block_m) + num_experts - 1,
            triton.cdiv(hidden_size, down_block_n),
        )
    else:
        up_grid = (
            num_experts,
            triton.cdiv(route_count, up_block_m),
            triton.cdiv(intermediate_size, up_block_n),
        )
        down_grid = (
            num_experts,
            triton.cdiv(route_count, down_block_m),
            triton.cdiv(hidden_size, down_block_n),
        )
    _grouped_moe_up_kernel[
        up_grid
    ](
        hidden_states,
        sorted_hidden,
        sorted_indices,
        gate_up_weight,
        sorted_weights,
        expert_counts,
        intermediate,
        token_capacity=1 if compact else route_count,
        num_experts=num_experts,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        TOP_K=top_k,
        COMPACT=compact,
        DYNAMIC_BLOCKS=dynamic_blocks,
        APPLY_ROUTE_WEIGHT=apply_route_weight,
        EXPERT_BLOCK=expert_block,
        BLOCK_M=up_block_m,
        BLOCK_N=up_block_n,
        BLOCK_K=up_block_k,
        num_warps=up_warps,
        num_stages=up_stages,
    )
    _grouped_moe_down_kernel[
        down_grid
    ](
        intermediate,
        down_weight,
        sorted_indices,
        expert_counts,
        output,
        route_output,
        token_capacity=1 if compact else route_count,
        num_experts=num_experts,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        TOP_K=top_k,
        STORE_ROUTES=top_k > 1,
        COMPACT=compact,
        DYNAMIC_BLOCKS=dynamic_blocks,
        EXPERT_BLOCK=expert_block,
        BLOCK_M=down_block_m,
        BLOCK_N=down_block_n,
        BLOCK_K=down_block_k,
        num_warps=down_warps,
        num_stages=down_stages,
    )
    if top_k > 1:
        _moe_topk_reduce_kernel[(token_count,)](
            route_output,
            output,
            top_k=top_k,
            hidden_size=hidden_size,
            BLOCK_H=triton.next_power_of_2(hidden_size),
            num_warps=4,
        )
    return output


