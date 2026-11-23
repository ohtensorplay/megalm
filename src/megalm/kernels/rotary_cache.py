from __future__ import annotations

import tensorplay
import triton
import triton.language as tl

@triton.jit
def _rope_cache_write_kernel(
    query_ptr,
    key_ptr,
    value_ptr,
    positions_ptr,
    slots_ptr,
    active_ptr,
    cos_ptr,
    sin_ptr,
    query_output_ptr,
    key_cache_ptr,
    value_cache_ptr,
    num_query_heads: tl.constexpr,
    num_kv_heads: tl.constexpr,
    head_dim: tl.constexpr,
    half_dim: tl.constexpr,
    BLOCK_HALF: tl.constexpr,
):
    token_index = tl.program_id(0)
    head_index = tl.program_id(1)
    offsets = tl.arange(0, BLOCK_HALF)
    mask = offsets < half_dim
    position = tl.load(positions_ptr + token_index)
    slot = tl.load(slots_ptr + token_index)
    active = tl.load(active_ptr + token_index).to(tl.int1)
    cos = tl.load(cos_ptr + position * half_dim + offsets, mask=mask)
    sin = tl.load(sin_ptr + position * half_dim + offsets, mask=mask)

    if head_index < num_query_heads:
        query_base = (
            token_index * num_query_heads * head_dim + head_index * head_dim
        )
        query_first = tl.load(query_ptr + query_base + offsets, mask=mask)
        query_second = tl.load(
            query_ptr + query_base + half_dim + offsets,
            mask=mask,
        )
        query_first_f32 = query_first.to(tl.float32)
        query_second_f32 = query_second.to(tl.float32)
        rotated_first = query_first_f32 * cos - query_second_f32 * sin
        rotated_second = query_second_f32 * cos + query_first_f32 * sin
        tl.store(
            query_output_ptr + query_base + offsets,
            tl.where(active, rotated_first, 0.0),
            mask=mask,
        )
        tl.store(
            query_output_ptr + query_base + half_dim + offsets,
            tl.where(active, rotated_second, 0.0),
            mask=mask,
        )

    if head_index < num_kv_heads:
        kv_base = token_index * num_kv_heads * head_dim + head_index * head_dim
        key_first = tl.load(key_ptr + kv_base + offsets, mask=mask)
        key_second = tl.load(
            key_ptr + kv_base + half_dim + offsets,
            mask=mask,
        )
        key_first_f32 = key_first.to(tl.float32)
        key_second_f32 = key_second.to(tl.float32)
        rotated_key_first = key_first_f32 * cos - key_second_f32 * sin
        rotated_key_second = key_second_f32 * cos + key_first_f32 * sin
        cache_base = slot * num_kv_heads * head_dim + head_index * head_dim
        write_mask = mask & active
        tl.store(
            key_cache_ptr + cache_base + offsets,
            rotated_key_first,
            mask=write_mask,
        )
        tl.store(
            key_cache_ptr + cache_base + half_dim + offsets,
            rotated_key_second,
            mask=write_mask,
        )
        value_first = tl.load(value_ptr + kv_base + offsets, mask=mask)
        value_second = tl.load(
            value_ptr + kv_base + half_dim + offsets,
            mask=mask,
        )
        tl.store(
            value_cache_ptr + cache_base + offsets,
            value_first,
            mask=write_mask,
        )
        tl.store(
            value_cache_ptr + cache_base + half_dim + offsets,
            value_second,
            mask=write_mask,
        )


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
    active: tensorplay.Tensor,
) -> tensorplay.Tensor:
    query = query.contiguous()
    key = key.contiguous()
    value = value.contiguous()
    output = tensorplay.empty_like(query)
    head_dim = query.shape[-1]
    half_dim = head_dim // 2
    block_half = triton.next_power_of_2(half_dim)
    grid = (query.shape[0], max(query.shape[1], key.shape[1]))
    _rope_cache_write_kernel[grid](
        query,
        key,
        value,
        positions,
        slot_mapping,
        active,
        cos_cache,
        sin_cache,
        output,
        key_cache,
        value_cache,
        num_query_heads=query.shape[1],
        num_kv_heads=key.shape[1],
        head_dim=head_dim,
        half_dim=half_dim,
        BLOCK_HALF=block_half,
        num_warps=1,
    )
    return output


@triton.jit
def _fused_qkv_norm_rope_cache_write_kernel(
    qkv_ptr,
    q_norm_weight_ptr,
    k_norm_weight_ptr,
    positions_ptr,
    slots_ptr,
    active_ptr,
    cos_ptr,
    sin_ptr,
    query_output_ptr,
    key_cache_ptr,
    value_cache_ptr,
    qkv_width: tl.constexpr,
    num_query_heads: tl.constexpr,
    num_kv_heads: tl.constexpr,
    head_dim: tl.constexpr,
    half_dim: tl.constexpr,
    eps: tl.constexpr,
    BLOCK_HALF: tl.constexpr,
):
    token_index = tl.program_id(0)
    head_index = tl.program_id(1)
    offsets = tl.arange(0, BLOCK_HALF)
    mask = offsets < half_dim
    position = tl.load(positions_ptr + token_index)
    slot = tl.load(slots_ptr + token_index)
    active = tl.load(active_ptr + token_index).to(tl.int1)
    cos = tl.load(cos_ptr + position * half_dim + offsets, mask=mask)
    sin = tl.load(sin_ptr + position * half_dim + offsets, mask=mask)
    row_base = token_index * qkv_width

    if head_index < num_query_heads:
        query_base = row_base + head_index * head_dim
        query_first = tl.load(
            qkv_ptr + query_base + offsets,
            mask=mask,
            other=0.0,
        ).to(tl.float32)
        query_second = tl.load(
            qkv_ptr + query_base + half_dim + offsets,
            mask=mask,
            other=0.0,
        ).to(tl.float32)
        query_variance = (
            tl.sum(query_first * query_first, axis=0)
            + tl.sum(query_second * query_second, axis=0)
        ) / head_dim
        query_inverse_rms = tl.rsqrt(query_variance + eps)
        query_weight_first = tl.load(
            q_norm_weight_ptr + offsets,
            mask=mask,
            other=0.0,
        ).to(tl.float32)
        query_weight_second = tl.load(
            q_norm_weight_ptr + half_dim + offsets,
            mask=mask,
            other=0.0,
        ).to(tl.float32)
        query_first = (
            query_first * query_inverse_rms * query_weight_first
        ).to(qkv_ptr.dtype.element_ty)
        query_second = (
            query_second * query_inverse_rms * query_weight_second
        ).to(qkv_ptr.dtype.element_ty)
        query_first = query_first.to(tl.float32)
        query_second = query_second.to(tl.float32)
        output_base = (
            token_index * num_query_heads * head_dim
            + head_index * head_dim
        )
        tl.store(
            query_output_ptr + output_base + offsets,
            tl.where(
                active,
                query_first * cos - query_second * sin,
                0.0,
            ),
            mask=mask,
        )
        tl.store(
            query_output_ptr + output_base + half_dim + offsets,
            tl.where(
                active,
                query_second * cos + query_first * sin,
                0.0,
            ),
            mask=mask,
        )

    if head_index < num_kv_heads:
        key_offset = num_query_heads * head_dim
        key_base = row_base + key_offset + head_index * head_dim
        key_first = tl.load(
            qkv_ptr + key_base + offsets,
            mask=mask,
            other=0.0,
        ).to(tl.float32)
        key_second = tl.load(
            qkv_ptr + key_base + half_dim + offsets,
            mask=mask,
            other=0.0,
        ).to(tl.float32)
        key_variance = (
            tl.sum(key_first * key_first, axis=0)
            + tl.sum(key_second * key_second, axis=0)
        ) / head_dim
        key_inverse_rms = tl.rsqrt(key_variance + eps)
        key_weight_first = tl.load(
            k_norm_weight_ptr + offsets,
            mask=mask,
            other=0.0,
        ).to(tl.float32)
        key_weight_second = tl.load(
            k_norm_weight_ptr + half_dim + offsets,
            mask=mask,
            other=0.0,
        ).to(tl.float32)
        key_first = (
            key_first * key_inverse_rms * key_weight_first
        ).to(qkv_ptr.dtype.element_ty)
        key_second = (
            key_second * key_inverse_rms * key_weight_second
        ).to(qkv_ptr.dtype.element_ty)
        key_first = key_first.to(tl.float32)
        key_second = key_second.to(tl.float32)
        cache_base = slot * num_kv_heads * head_dim + head_index * head_dim
        write_mask = mask & active
        tl.store(
            key_cache_ptr + cache_base + offsets,
            key_first * cos - key_second * sin,
            mask=write_mask,
        )
        tl.store(
            key_cache_ptr + cache_base + half_dim + offsets,
            key_second * cos + key_first * sin,
            mask=write_mask,
        )
        value_base = (
            row_base
            + (num_query_heads + num_kv_heads) * head_dim
            + head_index * head_dim
        )
        value_first = tl.load(
            qkv_ptr + value_base + offsets,
            mask=mask,
            other=0.0,
        )
        value_second = tl.load(
            qkv_ptr + value_base + half_dim + offsets,
            mask=mask,
            other=0.0,
        )
        tl.store(
            value_cache_ptr + cache_base + offsets,
            value_first,
            mask=write_mask,
        )
        tl.store(
            value_cache_ptr + cache_base + half_dim + offsets,
            value_second,
            mask=write_mask,
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
    qkv = qkv.contiguous()
    if qkv.ndim != 2:
        raise ValueError("packed QKV must have shape [tokens, width]")
    head_dim = q_norm_weight.numel()
    if (
        head_dim <= 0
        or head_dim % 2
        or k_norm_weight.numel() != head_dim
    ):
        raise ValueError("Q/K norm geometry is invalid")
    expected_width = (
        num_query_heads + 2 * num_kv_heads
    ) * head_dim
    if qkv.shape[1] != expected_width:
        raise ValueError("packed QKV width does not match the head geometry")
    output = tensorplay.empty(
        (qkv.shape[0], num_query_heads, head_dim),
        dtype=qkv.dtype,
        device=qkv.device,
    )
    half_dim = head_dim // 2
    block_half = triton.next_power_of_2(half_dim)
    _fused_qkv_norm_rope_cache_write_kernel[
        (qkv.shape[0], max(num_query_heads, num_kv_heads))
    ](
        qkv,
        q_norm_weight,
        k_norm_weight,
        positions,
        slot_mapping,
        active,
        cos_cache,
        sin_cache,
        output,
        key_cache,
        value_cache,
        qkv_width=expected_width,
        num_query_heads=num_query_heads,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        half_dim=half_dim,
        eps=eps,
        BLOCK_HALF=block_half,
        num_warps=1,
    )
    return output


