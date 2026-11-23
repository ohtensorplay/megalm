from __future__ import annotations

import tensorplay
import triton
import triton.language as tl


@triton.jit
def _active_kv_write_kernel(
    key_ptr,
    value_ptr,
    key_cache_ptr,
    value_cache_ptr,
    slots_ptr,
    active_ptr,
    key_token_stride: tl.constexpr,
    key_head_stride: tl.constexpr,
    value_token_stride: tl.constexpr,
    value_head_stride: tl.constexpr,
    heads: tl.constexpr,
    width: tl.constexpr,
    BLOCK: tl.constexpr,
):
    token = tl.program_id(0)
    head = tl.program_id(1)
    offsets = tl.arange(0, BLOCK)
    mask = offsets < width
    active = tl.load(active_ptr + token)
    slot = tl.load(slots_ptr + token).to(tl.int64)
    key_source = (
        token * key_token_stride + head * key_head_stride + offsets
    )
    value_source = (
        token * value_token_stride + head * value_head_stride + offsets
    )
    target = (slot * heads + head) * width + offsets
    key = tl.load(
        key_ptr + key_source,
        mask=mask & active,
        other=0.0,
    )
    value = tl.load(
        value_ptr + value_source,
        mask=mask & active,
        other=0.0,
    )
    tl.store(key_cache_ptr + target, key, mask=mask & active)
    tl.store(value_cache_ptr + target, value, mask=mask & active)


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
        flat_key_cache.shape[1:] != (heads, width)
        or slots.shape != (tokens,)
        or active.shape != (tokens,)
    ):
        raise ValueError("Qwen3.5 KV write metadata differs")
    block = triton.next_power_of_2(width)
    if key.stride(-1) != 1:
        key = key.contiguous()
    if value.stride(-1) != 1:
        value = value.contiguous()
    _active_kv_write_kernel[(tokens, heads)](
        key,
        value,
        flat_key_cache,
        flat_value_cache,
        slots,
        active,
        key_token_stride=key.stride(0),
        key_head_stride=key.stride(1),
        value_token_stride=value.stride(0),
        value_head_stride=value.stride(1),
        heads=heads,
        width=width,
        BLOCK=block,
        num_warps=4,
    )


__all__ = ["active_kv_write"]
