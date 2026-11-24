from __future__ import annotations

import tensorplay
import triton
import triton.language as tl

from megalm.kernels.kv_quant_triton import (
    get_rht_matrices,
    quantize_and_store_kv,
    rht_enabled,
)
from megalm.runtime.kv_quant import (
    dequant_kernel_kwargs,
    head_scale_tensor,
    kv_quant_of,
    kv_scale_of,
)


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
    key_scale_ptr,
    value_scale_ptr,
    head_scale_k_ptr,
    head_scale_v_ptr,
    num_query_heads: tl.constexpr,
    num_kv_heads: tl.constexpr,
    head_dim: tl.constexpr,
    half_dim: tl.constexpr,
    BLOCK: tl.constexpr,
    BLOCK_HALF: tl.constexpr,
    stride_slot,
    stride_head,
    ss_pos,
    ss_head,
    MODE: tl.constexpr,
    SCALAR_K,
    SCALAR_V,
    GLOBAL_SCALE,
    FP8_DTYPE: tl.constexpr,
    QUANT_MAX_K: tl.constexpr,
    QUANT_MAX_V: tl.constexpr,
    USE_RHT: tl.constexpr,
    rht_h_ptr,
    rht_sign_ptr,
):
    token_index = tl.program_id(0)
    head_index = tl.program_id(1)
    position = tl.load(positions_ptr + token_index)
    slot = tl.load(slots_ptr + token_index).to(tl.int64)
    active = tl.load(active_ptr + token_index).to(tl.int1)

    dims = tl.arange(0, BLOCK)
    full_mask = dims < head_dim
    cos_index = dims % half_dim
    cos = tl.load(
        cos_ptr + position * half_dim + cos_index,
        mask=cos_index < half_dim,
        other=0.0,
    )
    sin = tl.load(
        sin_ptr + position * half_dim + cos_index,
        mask=cos_index < half_dim,
        other=0.0,
    )
    partner = tl.where(dims < half_dim, dims + half_dim, dims - half_dim)

    if head_index < num_query_heads:
        query_base = (
            token_index * num_query_heads * head_dim + head_index * head_dim
        )
        query_first = tl.load(
            query_ptr + query_base + dims, mask=full_mask, other=0.0
        ).to(tl.float32)
        query_partner = tl.load(
            query_ptr + query_base + partner, mask=full_mask, other=0.0
        ).to(tl.float32)
        rotated_query = tl.where(
            dims < half_dim,
            query_first * cos - query_partner * sin,
            query_first * cos + query_partner * sin,
        )
        tl.store(
            query_output_ptr + query_base + dims,
            tl.where(active, rotated_query, 0.0),
            mask=full_mask,
        )

    if head_index < num_kv_heads:
        kv_base = token_index * num_kv_heads * head_dim + head_index * head_dim
        key_first = tl.load(
            key_ptr + kv_base + dims, mask=full_mask, other=0.0
        ).to(tl.float32)
        key_partner = tl.load(
            key_ptr + kv_base + partner, mask=full_mask, other=0.0
        ).to(tl.float32)
        rotated_key = tl.where(
            dims < half_dim,
            key_first * cos - key_partner * sin,
            key_first * cos + key_partner * sin,
        )
        value = tl.load(
            value_ptr + kv_base + dims, mask=full_mask, other=0.0
        ).to(tl.float32)

        if MODE == 4 or MODE == 5:
            data_dims = tl.arange(0, BLOCK // 2)
            data_mask = (data_dims < head_dim // 2) & active
        else:
            data_dims = dims
            data_mask = full_mask & active
        key_target = slot * stride_slot + head_index * stride_head + data_dims
        value_target = slot * stride_slot + head_index * stride_head + data_dims
        # The scale view is (pages, page_size, heads) with strides (PS*H, H, 1),
        # so a global slot maps directly through ``slot * ss_pos``.
        scale_offset = slot * ss_pos + head_index * ss_head
        pack_scale_base = slot * stride_slot + head_index * stride_head + head_dim // 2

        quantize_and_store_kv(
            rotated_key,
            key_cache_ptr,
            key_target,
            data_mask,
            key_scale_ptr,
            scale_offset,
            head_scale_k_ptr,
            head_index,
            pack_scale_base,
            16,
            MODE,
            SCALAR_K,
            GLOBAL_SCALE,
            FP8_DTYPE,
            QUANT_MAX_K,
            USE_RHT,
            BLOCK,
            rht_h_ptr,
            rht_sign_ptr,
        )
        quantize_and_store_kv(
            value,
            value_cache_ptr,
            value_target,
            data_mask,
            value_scale_ptr,
            scale_offset,
            head_scale_v_ptr,
            head_index,
            pack_scale_base,
            16,
            MODE,
            SCALAR_V,
            GLOBAL_SCALE,
            FP8_DTYPE,
            QUANT_MAX_V,
            USE_RHT,
            BLOCK,
            rht_h_ptr,
            rht_sign_ptr,
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
    block = triton.next_power_of_2(head_dim)
    grid = (query.shape[0], max(query.shape[1], key.shape[1]))
    quant = kv_quant_of(key_cache)
    key_scale = kv_scale_of(key_cache)
    value_scale = kv_scale_of(value_cache)
    if key_scale is None:
        key_scale = key_cache
        value_scale = value_cache
    head_scale_k = head_scale_tensor(quant, "key", key.device, key_cache)
    head_scale_v = head_scale_tensor(quant, "value", value.device, value_cache)
    k_kwargs = dequant_kernel_kwargs(quant, "key")
    v_kwargs = dequant_kernel_kwargs(quant, "value")
    use_rht = rht_enabled(quant)
    if use_rht:
        rht_h, rht_signs = get_rht_matrices(head_dim, key.device)
    else:
        rht_h = rht_signs = key_cache
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
        key_scale_ptr=key_scale,
        value_scale_ptr=value_scale,
        head_scale_k_ptr=head_scale_k,
        head_scale_v_ptr=head_scale_v,
        num_query_heads=query.shape[1],
        num_kv_heads=key.shape[1],
        head_dim=head_dim,
        half_dim=half_dim,
        BLOCK=block,
        BLOCK_HALF=block_half,
        stride_slot=key_cache.stride(-3),
        stride_head=key_cache.stride(-2),
        ss_pos=key_scale.stride(1),
        ss_head=key_scale.stride(2),
        MODE=0 if quant is None else quant.triton_mode,
        SCALAR_K=k_kwargs["SCALAR_SCALE"],
        SCALAR_V=v_kwargs["SCALAR_SCALE"],
        GLOBAL_SCALE=k_kwargs["GLOBAL_SCALE"],
        FP8_DTYPE=k_kwargs["FP8_DTYPE"],
        QUANT_MAX_K=int(quant.quant_max) if quant else 448,
        QUANT_MAX_V=int(quant.quant_max) if quant else 448,
        USE_RHT=use_rht,
        rht_h_ptr=rht_h,
        rht_sign_ptr=rht_signs,
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
    key_scale_ptr,
    value_scale_ptr,
    head_scale_k_ptr,
    head_scale_v_ptr,
    qkv_width: tl.constexpr,
    num_query_heads: tl.constexpr,
    num_kv_heads: tl.constexpr,
    head_dim: tl.constexpr,
    half_dim: tl.constexpr,
    eps: tl.constexpr,
    BLOCK: tl.constexpr,
    BLOCK_HALF: tl.constexpr,
    stride_slot,
    stride_head,
    ss_pos,
    ss_head,
    MODE: tl.constexpr,
    SCALAR_K,
    SCALAR_V,
    GLOBAL_SCALE,
    FP8_DTYPE: tl.constexpr,
    QUANT_MAX_K: tl.constexpr,
    QUANT_MAX_V: tl.constexpr,
    USE_RHT: tl.constexpr,
    rht_h_ptr,
    rht_sign_ptr,
):
    token_index = tl.program_id(0)
    head_index = tl.program_id(1)
    position = tl.load(positions_ptr + token_index)
    slot = tl.load(slots_ptr + token_index).to(tl.int64)
    active = tl.load(active_ptr + token_index).to(tl.int1)

    dims = tl.arange(0, BLOCK)
    full_mask = dims < head_dim
    cos_index = dims % half_dim
    cos = tl.load(
        cos_ptr + position * half_dim + cos_index,
        mask=cos_index < half_dim,
        other=0.0,
    )
    sin = tl.load(
        sin_ptr + position * half_dim + cos_index,
        mask=cos_index < half_dim,
        other=0.0,
    )
    partner = tl.where(dims < half_dim, dims + half_dim, dims - half_dim)
    row_base = token_index * qkv_width

    if head_index < num_query_heads:
        query_base = row_base + head_index * head_dim
        query_first = tl.load(
            qkv_ptr + query_base + dims, mask=full_mask, other=0.0
        ).to(tl.float32)
        query_partner = tl.load(
            qkv_ptr + query_base + partner, mask=full_mask, other=0.0
        ).to(tl.float32)
        query_variance = tl.sum(query_first * query_first, axis=0) / head_dim
        query_inverse_rms = tl.rsqrt(query_variance + eps)
        query_weight = tl.load(
            q_norm_weight_ptr + dims, mask=full_mask, other=0.0
        ).to(tl.float32)
        partner_weight = tl.load(
            q_norm_weight_ptr + partner, mask=full_mask, other=0.0
        ).to(tl.float32)
        query_first = (
            query_first * query_inverse_rms * query_weight
        ).to(qkv_ptr.dtype.element_ty).to(tl.float32)
        query_partner = (
            query_partner * query_inverse_rms * partner_weight
        ).to(qkv_ptr.dtype.element_ty).to(tl.float32)
        rotated_query = tl.where(
            dims < half_dim,
            query_first * cos - query_partner * sin,
            query_first * cos + query_partner * sin,
        )
        output_base = (
            token_index * num_query_heads * head_dim
            + head_index * head_dim
        )
        tl.store(
            query_output_ptr + output_base + dims,
            tl.where(active, rotated_query, 0.0),
            mask=full_mask,
        )

    if head_index < num_kv_heads:
        key_base = row_base + num_query_heads * head_dim + head_index * head_dim
        key_first = tl.load(
            qkv_ptr + key_base + dims, mask=full_mask, other=0.0
        ).to(tl.float32)
        key_partner = tl.load(
            qkv_ptr + key_base + partner, mask=full_mask, other=0.0
        ).to(tl.float32)
        key_variance = tl.sum(key_first * key_first, axis=0) / head_dim
        key_inverse_rms = tl.rsqrt(key_variance + eps)
        key_weight = tl.load(
            k_norm_weight_ptr + dims, mask=full_mask, other=0.0
        ).to(tl.float32)
        key_partner_weight = tl.load(
            k_norm_weight_ptr + partner, mask=full_mask, other=0.0
        ).to(tl.float32)
        key_first = (
            key_first * key_inverse_rms * key_weight
        ).to(qkv_ptr.dtype.element_ty).to(tl.float32)
        key_partner = (
            key_partner * key_inverse_rms * key_partner_weight
        ).to(qkv_ptr.dtype.element_ty).to(tl.float32)
        rotated_key = tl.where(
            dims < half_dim,
            key_first * cos - key_partner * sin,
            key_first * cos + key_partner * sin,
        )
        value_base = (
            row_base
            + (num_query_heads + num_kv_heads) * head_dim
            + head_index * head_dim
        )
        value = tl.load(
            qkv_ptr + value_base + dims, mask=full_mask, other=0.0
        ).to(tl.float32)

        if MODE == 4 or MODE == 5:
            data_dims = tl.arange(0, BLOCK // 2)
            data_mask = (data_dims < head_dim // 2) & active
        else:
            data_dims = dims
            data_mask = full_mask & active
        key_target = slot * stride_slot + head_index * stride_head + data_dims
        value_target = slot * stride_slot + head_index * stride_head + data_dims
        # The scale view is (pages, page_size, heads) with strides (PS*H, H, 1),
        # so a global slot maps directly through ``slot * ss_pos``.
        scale_offset = slot * ss_pos + head_index * ss_head
        pack_scale_base = slot * stride_slot + head_index * stride_head + head_dim // 2

        quantize_and_store_kv(
            rotated_key,
            key_cache_ptr,
            key_target,
            data_mask,
            key_scale_ptr,
            scale_offset,
            head_scale_k_ptr,
            head_index,
            pack_scale_base,
            16,
            MODE,
            SCALAR_K,
            GLOBAL_SCALE,
            FP8_DTYPE,
            QUANT_MAX_K,
            USE_RHT,
            BLOCK,
            rht_h_ptr,
            rht_sign_ptr,
        )
        quantize_and_store_kv(
            value,
            value_cache_ptr,
            value_target,
            data_mask,
            value_scale_ptr,
            scale_offset,
            head_scale_v_ptr,
            head_index,
            pack_scale_base,
            16,
            MODE,
            SCALAR_V,
            GLOBAL_SCALE,
            FP8_DTYPE,
            QUANT_MAX_V,
            USE_RHT,
            BLOCK,
            rht_h_ptr,
            rht_sign_ptr,
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
    block = triton.next_power_of_2(head_dim)
    quant = kv_quant_of(key_cache)
    key_scale = kv_scale_of(key_cache)
    value_scale = kv_scale_of(value_cache)
    if key_scale is None:
        key_scale = key_cache
        value_scale = value_cache
    head_scale_k = head_scale_tensor(quant, "key", qkv.device, key_cache)
    head_scale_v = head_scale_tensor(quant, "value", qkv.device, value_cache)
    k_kwargs = dequant_kernel_kwargs(quant, "key")
    v_kwargs = dequant_kernel_kwargs(quant, "value")
    use_rht = rht_enabled(quant)
    if use_rht:
        rht_h, rht_signs = get_rht_matrices(head_dim, qkv.device)
    else:
        rht_h = rht_signs = key_cache
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
        key_scale_ptr=key_scale,
        value_scale_ptr=value_scale,
        head_scale_k_ptr=head_scale_k,
        head_scale_v_ptr=head_scale_v,
        qkv_width=expected_width,
        num_query_heads=num_query_heads,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        half_dim=half_dim,
        eps=eps,
        BLOCK=block,
        BLOCK_HALF=block_half,
        stride_slot=key_cache.stride(-3),
        stride_head=key_cache.stride(-2),
        ss_pos=key_scale.stride(1),
        ss_head=key_scale.stride(2),
        MODE=0 if quant is None else quant.triton_mode,
        SCALAR_K=k_kwargs["SCALAR_SCALE"],
        SCALAR_V=v_kwargs["SCALAR_SCALE"],
        GLOBAL_SCALE=k_kwargs["GLOBAL_SCALE"],
        FP8_DTYPE=k_kwargs["FP8_DTYPE"],
        QUANT_MAX_K=int(quant.quant_max) if quant else 448,
        QUANT_MAX_V=int(quant.quant_max) if quant else 448,
        USE_RHT=use_rht,
        rht_h_ptr=rht_h,
        rht_sign_ptr=rht_signs,
        num_warps=1,
    )
    return output


__all__ = ["fused_qkv_norm_rope_cache_write", "rope_and_cache_write"]
