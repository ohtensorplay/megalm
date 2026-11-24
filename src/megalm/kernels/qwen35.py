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
def _active_kv_write_kernel(
    key_ptr,
    value_ptr,
    key_cache_ptr,
    value_cache_ptr,
    key_scale_ptr,
    value_scale_ptr,
    head_scale_k_ptr,
    head_scale_v_ptr,
    slots_ptr,
    active_ptr,
    key_token_stride: tl.constexpr,
    key_head_stride: tl.constexpr,
    value_token_stride: tl.constexpr,
    value_head_stride: tl.constexpr,
    heads: tl.constexpr,
    width: tl.constexpr,
    stride_slot,
    stride_head,
    ss_pos,
    ss_head,
    BLOCK: tl.constexpr,
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
    token = tl.program_id(0)
    head = tl.program_id(1)
    active = tl.load(active_ptr + token)
    slot = tl.load(slots_ptr + token).to(tl.int64)

    key_source = token * key_token_stride + head * key_head_stride
    value_source = token * value_token_stride + head * value_head_stride
    dims = tl.arange(0, BLOCK)
    dim_mask = dims < width
    key = tl.load(key_ptr + key_source + dims, mask=dim_mask & active, other=0.0)
    value = tl.load(
        value_ptr + value_source + dims, mask=dim_mask & active, other=0.0
    )

    if MODE == 4 or MODE == 5:
        data_dims = tl.arange(0, BLOCK // 2)
        data_mask = data_dims < width // 2
        key_target = slot * stride_slot + head * stride_head + data_dims
        value_target = slot * stride_slot + head * stride_head + data_dims
    else:
        data_dims = dims
        data_mask = dim_mask
        key_target = slot * stride_slot + head * stride_head + data_dims
        value_target = slot * stride_slot + head * stride_head + data_dims

    # The scale view is (pages, page_size, heads) with strides (PS*H, H, 1),
    # so a global slot maps directly to ``slot * H``.
    scale_offset = slot * ss_pos + head * ss_head
    pack_scale_base = slot * stride_slot + head * stride_head + width // 2

    quantize_and_store_kv(
        key.to(tl.float32),
        key_cache_ptr,
        key_target,
        data_mask,
        key_scale_ptr,
        scale_offset,
        head_scale_k_ptr,
        head,
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
        value.to(tl.float32),
        value_cache_ptr,
        value_target,
        data_mask,
        value_scale_ptr,
        scale_offset,
        head_scale_v_ptr,
        head,
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


def active_kv_write(
    key: tensorplay.Tensor,
    value: tensorplay.Tensor,
    flat_key_cache: tensorplay.Tensor,
    flat_value_cache: tensorplay.Tensor,
    slots: tensorplay.Tensor,
    active: tensorplay.Tensor,
) -> None:
    if (
        key.shape != value.shape
        or key.ndim != 3
        or flat_key_cache.shape != flat_value_cache.shape
        or flat_key_cache.ndim != 3
    ):
        raise ValueError("Qwen3.5 KV write geometry differs")
    tokens, heads, width = key.shape
    if (
        flat_key_cache.shape[1:] != (heads, flat_key_cache.shape[-1])
        or slots.shape != (tokens,)
        or active.shape != (tokens,)
    ):
        raise ValueError("Qwen3.5 KV write metadata differs")
    quant = kv_quant_of(flat_key_cache)
    key_scale = kv_scale_of(flat_key_cache)
    value_scale = kv_scale_of(flat_value_cache)
    if key_scale is None:
        key_scale = flat_key_cache
        value_scale = flat_value_cache
    head_scale_k = head_scale_tensor(
        quant, "key", key.device, flat_key_cache
    )
    head_scale_v = head_scale_tensor(
        quant, "value", value.device, flat_value_cache
    )
    k_kwargs = dequant_kernel_kwargs(quant, "key")
    v_kwargs = dequant_kernel_kwargs(quant, "value")
    block = triton.next_power_of_2(width)
    use_rht = rht_enabled(quant)
    if use_rht:
        rht_h, rht_signs = get_rht_matrices(width, key.device)
    else:
        rht_h = rht_signs = flat_key_cache  # unused when USE_RHT is False
    if key.stride(-1) != 1:
        key = key.contiguous()
    if value.stride(-1) != 1:
        value = value.contiguous()
    _active_kv_write_kernel[(tokens, heads)](
        key_ptr=key,
        value_ptr=value,
        key_cache_ptr=flat_key_cache,
        value_cache_ptr=flat_value_cache,
        key_scale_ptr=key_scale,
        value_scale_ptr=value_scale,
        head_scale_k_ptr=head_scale_k,
        head_scale_v_ptr=head_scale_v,
        slots_ptr=slots,
        active_ptr=active,
        key_token_stride=key.stride(0),
        key_head_stride=key.stride(1),
        value_token_stride=value.stride(0),
        value_head_stride=value.stride(1),
        heads=heads,
        width=width,
        stride_slot=flat_key_cache.stride(0),
        stride_head=flat_key_cache.stride(1),
        ss_pos=key_scale.stride(1),
        ss_head=key_scale.stride(2),
        BLOCK=block,
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
        num_warps=4,
    )


__all__ = ["active_kv_write"]
