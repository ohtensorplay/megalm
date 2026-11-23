from __future__ import annotations

import tensorplay
import tensorplay.nn.functional as F


def rms_norm(
    x: tensorplay.Tensor,
    weight: tensorplay.Tensor,
    eps: float,
) -> tensorplay.Tensor:
    variance = x.float().pow(2).mean(dim=-1, keepdim=True)
    normalized = x.float() * tensorplay.rsqrt(variance + eps)
    return (normalized * weight.float()).to(dtype=x.dtype)


def fused_add_rms_norm(
    x: tensorplay.Tensor,
    residual: tensorplay.Tensor,
    weight: tensorplay.Tensor,
    eps: float,
) -> tuple[tensorplay.Tensor, tensorplay.Tensor]:
    updated_residual = residual + x
    return rms_norm(updated_residual, weight, eps), updated_residual


def offset_rms_norm(
    x: tensorplay.Tensor,
    weight: tensorplay.Tensor,
    eps: float,
) -> tensorplay.Tensor:
    variance = x.float().pow(2).mean(dim=-1, keepdim=True)
    normalized = x.float() * tensorplay.rsqrt(variance + eps)
    return (normalized * (1.0 + weight.float())).to(dtype=x.dtype)


def fused_add_offset_rms_norm(
    x: tensorplay.Tensor,
    residual: tensorplay.Tensor,
    weight: tensorplay.Tensor,
    eps: float,
) -> tuple[tensorplay.Tensor, tensorplay.Tensor]:
    updated_residual = residual + x
    return (
        offset_rms_norm(updated_residual, weight, eps),
        updated_residual,
    )


def silu_and_mul(gate_up: tensorplay.Tensor) -> tensorplay.Tensor:
    gate, up = gate_up.chunk(2, dim=-1)
    return F.silu(gate) * up


def moe_topk_softmax(
    router_logits: tensorplay.Tensor,
    top_k: int,
    *,
    normalize: bool,
    gating: str = "softmax",
) -> tuple[tensorplay.Tensor, tensorplay.Tensor]:
    """Reference routing: ``softmax/sigmoid -> topk -> optional renorm``."""

    logits = router_logits.float()
    if gating == "sigmoid":
        scores = tensorplay.sigmoid(logits)
        topk_weights, topk_ids = tensorplay.topk(scores, top_k, dim=-1)
        if normalize:
            topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)
    elif gating == "softmax":
        topk_weights, topk_ids = tensorplay.topk(logits, top_k, dim=-1)
        if normalize:
            topk_weights = tensorplay.softmax(topk_weights, dim=-1)
        else:
            full_probabilities = tensorplay.softmax(logits, dim=-1)
            topk_weights = tensorplay.gather(
                full_probabilities,
                -1,
                topk_ids,
            )
    else:
        raise ValueError("gating must be softmax or sigmoid")
    return topk_weights.float(), topk_ids.long()


def rope_and_cache_write(
    query: tensorplay.Tensor,
    key: tensorplay.Tensor,
    value: tensorplay.Tensor,
    positions: tensorplay.Tensor,
    slot_mapping: tensorplay.Tensor,
    cos_cache: tensorplay.Tensor,
    sin_cache: tensorplay.Tensor,
    key_cache: tensorplay.Tensor,
    value_cache: tensorplay.Tensor,
    active: tensorplay.Tensor | None = None,
) -> tensorplay.Tensor:
    if active is None:
        active = tensorplay.ones(
            query.shape[0],
            dtype=tensorplay.bool,
            device=query.device,
        )
    half = query.shape[-1] // 2
    cos = cos_cache.index_select(0, positions.long()).to(query.dtype)[:, None, :]
    sin = sin_cache.index_select(0, positions.long()).to(query.dtype)[:, None, :]

    query_first, query_second = query[..., :half], query[..., half:]
    key_first, key_second = key[..., :half], key[..., half:]
    query_rotated = tensorplay.cat(
        (
            query_first * cos - query_second * sin,
            query_second * cos + query_first * sin,
        ),
        dim=-1,
    )
    key_rotated = tensorplay.cat(
        (
            key_first * cos - key_second * sin,
            key_second * cos + key_first * sin,
        ),
        dim=-1,
    )
    flat_key_cache = key_cache.view(
        -1,
        key_cache.shape[-2],
        key_cache.shape[-1],
    )
    flat_value_cache = value_cache.view_as(flat_key_cache)
    active_slots = slot_mapping[active].long()
    flat_key_cache[active_slots] = key_rotated[active]
    flat_value_cache[active_slots] = value[active]
    return tensorplay.where(active[:, None, None], query_rotated, 0)


def paged_decode_attention(
    query: tensorplay.Tensor,
    key_cache: tensorplay.Tensor,
    value_cache: tensorplay.Tensor,
    block_tables: tensorplay.Tensor,
    sequence_lengths: tensorplay.Tensor,
    page_size: int,
) -> tensorplay.Tensor:
    batch_size, num_query_heads, head_dim = query.shape
    num_kv_heads = key_cache.shape[-2]
    query_per_kv = num_query_heads // num_kv_heads
    pages = int(block_tables.shape[1])
    width = pages * page_size
    offsets = tensorplay.arange(
        width,
        device=query.device,
    )
    valid = offsets[None, :] < sequence_lengths[:, None]
    slots = (
        block_tables.long().unsqueeze(-1) * page_size
        + tensorplay.arange(
            page_size,
            device=query.device,
        )[None, None, :]
    ).reshape(batch_size, width)
    safe_slots = tensorplay.where(valid, slots, 0)
    flat_keys = key_cache.reshape(-1, num_kv_heads, head_dim)
    flat_values = value_cache.reshape(-1, num_kv_heads, head_dim)
    keys = (
        flat_keys.index_select(0, safe_slots.reshape(-1))
        .reshape(batch_size, width, num_kv_heads, head_dim)
        .repeat_interleave(query_per_kv, dim=2)
    )
    values = (
        flat_values.index_select(0, safe_slots.reshape(-1))
        .reshape(batch_size, width, num_kv_heads, head_dim)
        .repeat_interleave(query_per_kv, dim=2)
    )
    scores = tensorplay.einsum(
        "bhd,blhd->bhl",
        query.float(),
        keys.float(),
    ).masked_fill(~valid[:, None, :], float("-inf"))
    probabilities = tensorplay.softmax(
        scores * (head_dim**-0.5),
        dim=-1,
    ).to(values.dtype)
    return tensorplay.einsum(
        "bhl,blhd->bhd",
        probabilities,
        values,
    )


def routed_moe_decode(
    hidden_states: tensorplay.Tensor,
    gate_up_weight: tensorplay.Tensor,
    down_weight: tensorplay.Tensor,
    topk_ids: tensorplay.Tensor,
    topk_weights: tensorplay.Tensor,
) -> tensorplay.Tensor:
    output = tensorplay.zeros_like(hidden_states)
    num_experts = gate_up_weight.shape[0]
    for expert_index in range(num_experts):
        token_indices, route_indices = tensorplay.where(topk_ids == expert_index)
        selected = hidden_states.index_select(0, token_indices)
        gate_up = F.linear(selected, gate_up_weight[expert_index])
        expert_output = F.linear(
            silu_and_mul(gate_up),
            down_weight[expert_index],
        )
        route_weights = topk_weights[token_indices, route_indices].to(
            expert_output.dtype
        )
        output.index_add_(
            0,
            token_indices,
            expert_output * route_weights[:, None],
        )
    return output


def greedy_argmax(logits: tensorplay.Tensor) -> tensorplay.Tensor:
    return tensorplay.argmax(logits, dim=-1)
