from __future__ import annotations

import tensorplay
import tensorplay.nn.functional as F

from megalm.kernels import fallback
from megalm.kernels.ops import (
    GroupedMoEWorkspace,
    routed_moe_tensorcore,
)


def fused_qkv_norm_rope_cache_write(
    qkv: tensorplay.Tensor,
    q_norm_weight: tensorplay.Tensor,
    k_norm_weight: tensorplay.Tensor,
    eps: float,
    positions: tensorplay.Tensor,
    slot_mapping: tensorplay.Tensor,
    cos_cache: tensorplay.Tensor,
    sin_cache: tensorplay.Tensor,
    key_cache: tensorplay.Tensor,
    value_cache: tensorplay.Tensor,
    active: tensorplay.Tensor,
    *,
    num_query_heads: int,
    num_kv_heads: int,
) -> tensorplay.Tensor:
    head_dim = q_norm_weight.numel()
    query_width = num_query_heads * head_dim
    kv_width = num_kv_heads * head_dim
    query, key, value = qkv.split(
        (query_width, kv_width, kv_width),
        dim=-1,
    )
    query = fallback.rms_norm(
        query.view(-1, num_query_heads, head_dim),
        q_norm_weight,
        eps,
    )
    key = fallback.rms_norm(
        key.view(-1, num_kv_heads, head_dim),
        k_norm_weight,
        eps,
    )
    return fallback.rope_and_cache_write(
        query,
        key,
        value.view(-1, num_kv_heads, head_dim),
        positions,
        slot_mapping,
        cos_cache,
        sin_cache,
        key_cache,
        value_cache,
        active,
    )


def moe_top1(
    router_logits: tensorplay.Tensor,
    *,
    normalize: bool,
    output: tuple[tensorplay.Tensor, tensorplay.Tensor] | None = None,
) -> tuple[tensorplay.Tensor, tensorplay.Tensor]:
    if output is not None:
        raise ValueError("preallocated top-1 output requires Triton")
    routing_weights = tensorplay.softmax(router_logits.float(), dim=-1)
    topk_weights, topk_ids = tensorplay.topk(routing_weights, 1, dim=-1)
    if normalize:
        topk_weights = tensorplay.ones_like(topk_weights)
    return topk_weights, topk_ids


def moe_top1_from_hidden(
    hidden_states: tensorplay.Tensor,
    gate_weight: tensorplay.Tensor,
    *,
    normalize: bool,
    output: tuple[tensorplay.Tensor, tensorplay.Tensor] | None = None,
) -> tuple[tensorplay.Tensor, tensorplay.Tensor]:
    return moe_top1(
        F.linear(hidden_states, gate_weight),
        normalize=normalize,
        output=output,
    )


def routed_moe_grouped(
    hidden_states: tensorplay.Tensor,
    gate_up_weight: tensorplay.Tensor,
    down_weight: tensorplay.Tensor,
    topk_ids: tensorplay.Tensor,
    topk_weights: tensorplay.Tensor,
    *,
    workspace: GroupedMoEWorkspace | None = None,
) -> tensorplay.Tensor:
    del workspace
    return routed_moe_tensorcore(
        hidden_states,
        gate_up_weight,
        down_weight,
        topk_ids,
        topk_weights,
    )


def routed_moe_grouped_from_hidden(
    hidden_states: tensorplay.Tensor,
    router_weight: tensorplay.Tensor,
    gate_up_weight: tensorplay.Tensor,
    down_weight: tensorplay.Tensor,
    *,
    normalize: bool,
    workspace: GroupedMoEWorkspace | None = None,
) -> tensorplay.Tensor:
    topk_weights, topk_ids = moe_top1(
        F.linear(hidden_states, router_weight),
        normalize=normalize,
    )
    return routed_moe_grouped(
        hidden_states,
        gate_up_weight,
        down_weight,
        topk_ids,
        topk_weights,
        workspace=workspace,
    )


__all__ = [
    "fused_qkv_norm_rope_cache_write",
    "moe_top1",
    "moe_top1_from_hidden",
    "routed_moe_grouped",
    "routed_moe_grouped_from_hidden",
]
