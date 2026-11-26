"""Fused residual/scaling elementwise kernels for decode."""

from __future__ import annotations

import tensorplay
import triton
import triton.language as tl


@triton.jit
def _add_scaled_kernel(
    a_ptr,
    b_ptr,
    scale_ptr,
    output_ptr,
    width: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    offsets = tl.arange(0, BLOCK)
    mask = offsets < width
    base = row * width
    a = tl.load(a_ptr + base + offsets, mask=mask, other=0.0).to(tl.float32)
    b = tl.load(b_ptr + base + offsets, mask=mask, other=0.0).to(tl.float32)
    scale = tl.load(scale_ptr).to(tl.float32)
    tl.store(
        output_ptr + base + offsets,
        ((a + b) * scale).to(output_ptr.dtype.element_ty),
        mask=mask,
    )


def add_scaled(
    a: tensorplay.Tensor,
    b: tensorplay.Tensor,
    scale: tensorplay.Tensor,
) -> tensorplay.Tensor:
    """Compute ``(a + b) * scale`` where ``scale`` has a single element."""

    width = a.shape[-1]
    rows = a.numel() // width
    output = tensorplay.empty(a.shape, dtype=a.dtype, device=a.device)
    block = triton.next_power_of_2(width)
    _add_scaled_kernel[(rows,)](
        a,
        b,
        scale,
        output,
        width,
        block,
        num_warps=4,
    )
    return output
