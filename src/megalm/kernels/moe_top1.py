from __future__ import annotations

import tensorplay
import triton
import triton.language as tl

@triton.jit
def _moe_top1_kernel(
    router_logits_ptr,
    topk_weights_ptr,
    topk_ids_ptr,
    num_experts: tl.constexpr,
    normalize_top1: tl.constexpr,
    BLOCK_E: tl.constexpr,
):
    token_index = tl.program_id(0)
    expert_offsets = tl.arange(0, BLOCK_E)
    expert_mask = expert_offsets < num_experts
    logits = tl.load(
        router_logits_ptr
        + token_index * num_experts
        + expert_offsets,
        mask=expert_mask,
        other=-float("inf"),
    ).to(tl.float32)
    expert_id = tl.argmax(logits, axis=0)
    maximum = tl.max(logits, axis=0)
    if normalize_top1:
        route_weight = 1.0
    else:
        denominator = tl.sum(
            tl.where(expert_mask, tl.exp(logits - maximum), 0.0),
            axis=0,
        )
        route_weight = 1.0 / denominator
    tl.store(topk_ids_ptr + token_index, expert_id)
    tl.store(topk_weights_ptr + token_index, route_weight)


def moe_top1(
    router_logits: tensorplay.Tensor,
    *,
    normalize: bool,
    output: tuple[tensorplay.Tensor, tensorplay.Tensor] | None = None,
) -> tuple[tensorplay.Tensor, tensorplay.Tensor]:
    if router_logits.ndim != 2:
        raise ValueError("router logits must have shape [tokens, experts]")
    router_logits = router_logits.contiguous()
    token_count, num_experts = router_logits.shape
    if num_experts > 128:
        raise ValueError("fused top-1 router supports at most 128 experts")
    if output is None:
        topk_weights = tensorplay.empty(
            (token_count, 1),
            dtype=tensorplay.float32,
            device=router_logits.device,
        )
        topk_ids = tensorplay.empty(
            (token_count, 1),
            dtype=tensorplay.int64,
            device=router_logits.device,
        )
    else:
        topk_weights, topk_ids = output
        if (
            topk_weights.shape != (token_count, 1)
            or topk_weights.dtype != tensorplay.float32
            or topk_weights.device != router_logits.device
            or topk_ids.shape != (token_count, 1)
            or topk_ids.dtype != tensorplay.int64
            or topk_ids.device != router_logits.device
        ):
            raise ValueError("preallocated top-1 output has an invalid layout")
    _moe_top1_kernel[(token_count,)](
        router_logits,
        topk_weights,
        topk_ids,
        num_experts=num_experts,
        normalize_top1=normalize,
        BLOCK_E=triton.next_power_of_2(num_experts),
        num_warps=1,
    )
    return topk_weights, topk_ids


@triton.jit
def _moe_top1_from_hidden_kernel(
    hidden_ptr,
    gate_weight_ptr,
    topk_weights_ptr,
    topk_ids_ptr,
    hidden_size: tl.constexpr,
    num_experts: tl.constexpr,
    normalize_top1: tl.constexpr,
    BLOCK_H: tl.constexpr,
    BLOCK_E: tl.constexpr,
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
    expert_id = tl.argmax(logits, axis=0)
    maximum = tl.max(logits, axis=0)
    if normalize_top1:
        route_weight = 1.0
    else:
        denominator = tl.sum(
            tl.where(expert_mask, tl.exp(logits - maximum), 0.0),
            axis=0,
        )
        route_weight = 1.0 / denominator
    tl.store(topk_ids_ptr + token_index, expert_id)
    tl.store(topk_weights_ptr + token_index, route_weight)


def moe_top1_from_hidden(
    hidden_states: tensorplay.Tensor,
    gate_weight: tensorplay.Tensor,
    *,
    normalize: bool,
    output: tuple[tensorplay.Tensor, tensorplay.Tensor] | None = None,
) -> tuple[tensorplay.Tensor, tensorplay.Tensor]:
    """Fuse a small-expert router projection with top-1 selection."""

    if hidden_states.ndim != 2 or hidden_states.shape[0] <= 0:
        raise ValueError("fused hidden router requires one or more tokens")
    hidden_states = hidden_states.contiguous()
    gate_weight = gate_weight.contiguous()
    if (
        gate_weight.ndim != 2
        or gate_weight.shape[1] != hidden_states.shape[1]
    ):
        raise ValueError("router weight does not match hidden states")
    num_experts = gate_weight.shape[0]
    if num_experts <= 0 or num_experts > 16:
        raise ValueError("fused hidden router supports 1--16 experts")
    token_count = hidden_states.shape[0]
    output_shape = (token_count, 1)
    if output is None:
        topk_weights = tensorplay.empty(
            output_shape,
            dtype=tensorplay.float32,
            device=hidden_states.device,
        )
        topk_ids = tensorplay.empty(
            output_shape,
            dtype=tensorplay.int64,
            device=hidden_states.device,
        )
    else:
        topk_weights, topk_ids = output
        if (
            topk_weights.shape != output_shape
            or topk_weights.dtype != tensorplay.float32
            or topk_weights.device != hidden_states.device
            or topk_ids.shape != output_shape
            or topk_ids.dtype != tensorplay.int64
            or topk_ids.device != hidden_states.device
        ):
            raise ValueError("preallocated top-1 output has an invalid layout")
    block_h = min(256, triton.next_power_of_2(hidden_states.shape[1]))
    num_warps = (
        4
        if (
            num_experts > 8
            or (num_experts <= 4 and token_count <= 32)
        )
        else 1
    )
    _moe_top1_from_hidden_kernel[(token_count,)](
        hidden_states,
        gate_weight,
        topk_weights,
        topk_ids,
        hidden_size=hidden_states.shape[1],
        num_experts=num_experts,
        normalize_top1=normalize,
        BLOCK_H=block_h,
        BLOCK_E=triton.next_power_of_2(num_experts),
        num_warps=num_warps,
    )
    return topk_weights, topk_ids


