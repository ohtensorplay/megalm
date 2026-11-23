"""Fused Inkling-2 short convolution for one decode token per sequence."""

from __future__ import annotations

import tensorplay
import triton
import triton.language as tl


@triton.jit
def _short_convolution_decode_kernel(
    x_ptr,
    taps_ptr,
    dynamic_ptr,
    state_ptr,
    slots_ptr,
    positions_ptr,
    active_ptr,
    output_ptr,
    CHANNELS: tl.constexpr,
    KERNEL: tl.constexpr,
    DYNAMIC: tl.constexpr,
    SCALE: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    channels = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    in_bounds = channels < CHANNELS
    active = tl.load(active_ptr + row)
    output_address = output_ptr + row * CHANNELS + channels
    if active:
        slot = tl.load(slots_ptr + row)
        has_history = tl.load(positions_ptr + row) > 0
        history = state_ptr + (slot * CHANNELS + channels) * (KERNEL - 1)
        value = tl.load(x_ptr + row * CHANNELS + channels, in_bounds, 0)
        static_taps = taps_ptr + channels * KERNEL
        total = tl.full((BLOCK,), 0.0, tl.float32)
        for tap in tl.static_range(KERNEL):
            if tap < KERNEL - 1:
                sample = tl.load(history + tap, in_bounds & has_history, 0)
            else:
                sample = value
            if tap > 0:
                tl.store(history + tap - 1, sample, in_bounds)
            if DYNAMIC:
                lag = KERNEL - 1 - tap
                dynamic_taps = dynamic_ptr + (row * CHANNELS + channels) * KERNEL
                dynamic = tl.load(dynamic_taps + lag, in_bounds, 0)
                static = tl.load(static_taps + lag, in_bounds, 0)
                # The reference rounds the sum to the tap dtype before
                # multiplying it by the FP32 convolution sample.
                coefficient = (static.to(tl.float32) + dynamic.to(tl.float32)).to(
                    static.dtype
                )
            else:
                coefficient = tl.load(static_taps + tap, in_bounds, 0)
            total += sample.to(tl.float32) * coefficient.to(tl.float32)
        tl.store(output_address, (total + value.to(tl.float32)) * SCALE, in_bounds)
    else:
        tl.store(output_address, 0, in_bounds)


def short_convolution_decode(
    x: tensorplay.Tensor,
    taps: tensorplay.Tensor,
    state: tensorplay.Tensor,
    slots: tensorplay.Tensor,
    positions: tensorplay.Tensor,
    active: tensorplay.Tensor,
    dynamic: tensorplay.Tensor | None,
) -> tensorplay.Tensor:
    """Read and advance active request histories on the current CUDA stream."""

    rows, channels = x.shape
    kernel = taps.shape[1]
    output = tensorplay.empty_like(x)
    block = 128
    _short_convolution_decode_kernel[(rows, triton.cdiv(channels, block))](
        x,
        taps,
        dynamic if dynamic is not None else taps,
        state,
        slots,
        positions,
        active,
        output,
        CHANNELS=channels,
        KERNEL=kernel,
        DYNAMIC=dynamic is not None,
        SCALE=(kernel * (2 if dynamic is not None else 1) + 1) ** -0.5,
        BLOCK=block,
        num_warps=4,
        enable_fp_fusion=False,
    )
    return output
