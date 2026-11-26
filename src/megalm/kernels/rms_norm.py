from __future__ import annotations

import tensorplay
import triton
import triton.language as tl

@triton.jit
def _rms_norm_kernel(
    x_ptr,
    weight_ptr,
    output_ptr,
    row_stride,
    n_cols: tl.constexpr,
    eps: tl.constexpr,
    WEIGHT_OFFSET: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    offsets = tl.arange(0, BLOCK)
    mask = offsets < n_cols
    values = tl.load(x_ptr + row * row_stride + offsets, mask=mask, other=0.0)
    values_f32 = values.to(tl.float32)
    variance = tl.sum(values_f32 * values_f32, axis=0) / n_cols
    inverse_rms = tl.rsqrt(variance + eps)
    weight = tl.load(weight_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    if WEIGHT_OFFSET:
        weight += 1.0
    result = values_f32 * inverse_rms * weight
    tl.store(output_ptr + row * row_stride + offsets, result, mask=mask)


def rms_norm(
    x: tensorplay.Tensor,
    weight: tensorplay.Tensor,
    eps: float,
) -> tensorplay.Tensor:
    x_contiguous = x.contiguous()
    output = tensorplay.empty_like(x_contiguous)
    n_cols = x_contiguous.shape[-1]
    block = triton.next_power_of_2(n_cols)
    if block > 65536:
        raise ValueError(f"RMSNorm width {n_cols} exceeds the Triton guard")
    rows = x_contiguous.numel() // n_cols
    num_warps = 4 if block <= 2048 else 8
    _rms_norm_kernel[(rows,)](
        x_contiguous,
        weight,
        output,
        n_cols,
        n_cols=n_cols,
        eps=eps,
        WEIGHT_OFFSET=False,
        BLOCK=block,
        num_warps=num_warps,
    )
    return output.view_as(x)


def offset_rms_norm(
    x: tensorplay.Tensor,
    weight: tensorplay.Tensor,
    eps: float,
) -> tensorplay.Tensor:
    """Gemma-style RMSNorm whose checkpoint scale is stored as ``weight - 1``."""

    x_contiguous = x.contiguous()
    output = tensorplay.empty_like(x_contiguous)
    n_cols = x_contiguous.shape[-1]
    block = triton.next_power_of_2(n_cols)
    if block > 65536:
        raise ValueError(f"RMSNorm width {n_cols} exceeds the Triton guard")
    rows = x_contiguous.numel() // n_cols
    num_warps = 4 if block <= 2048 else 8
    _rms_norm_kernel[(rows,)](
        x_contiguous,
        weight,
        output,
        n_cols,
        n_cols=n_cols,
        eps=eps,
        WEIGHT_OFFSET=True,
        BLOCK=block,
        num_warps=num_warps,
    )
    return output.view_as(x)


@triton.jit
def _rms_norm_add_kernel(
    x_ptr,
    residual_ptr,
    weight_ptr,
    output_ptr,
    n_cols: tl.constexpr,
    eps: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    offsets = tl.arange(0, BLOCK)
    mask = offsets < n_cols
    values = tl.load(x_ptr + row * n_cols + offsets, mask=mask, other=0.0).to(
        tl.float32
    )
    variance = tl.sum(values * values, axis=0) / n_cols
    inverse_rms = tl.rsqrt(variance + eps)
    weight = tl.load(weight_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    residual = tl.load(
        residual_ptr + row * n_cols + offsets, mask=mask, other=0.0
    ).to(tl.float32)
    result = values * inverse_rms * weight + residual
    tl.store(
        output_ptr + row * n_cols + offsets,
        result.to(output_ptr.dtype.element_ty),
        mask=mask,
    )


def rms_norm_add(
    x: tensorplay.Tensor,
    weight: tensorplay.Tensor,
    eps: float,
    residual: tensorplay.Tensor,
) -> tensorplay.Tensor:
    """Fuse ``rms_norm(x) * weight + residual`` in one kernel."""

    x_contiguous = x.contiguous()
    n_cols = x_contiguous.shape[-1]
    block = triton.next_power_of_2(n_cols)
    rows = x_contiguous.numel() // n_cols
    output = tensorplay.empty_like(x_contiguous)
    _rms_norm_add_kernel[(rows,)](
        x_contiguous,
        residual,
        weight,
        output,
        n_cols=n_cols,
        eps=eps,
        BLOCK=block,
        num_warps=4 if block <= 2048 else 8,
    )
    return output.view_as(x)


@triton.jit
def _fused_add_rms_norm_kernel(
    x_ptr,
    residual_ptr,
    weight_ptr,
    row_stride,
    n_cols: tl.constexpr,
    eps: tl.constexpr,
    WEIGHT_OFFSET: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    offsets = tl.arange(0, BLOCK)
    mask = offsets < n_cols
    row_offsets = row * row_stride + offsets
    x = tl.load(x_ptr + row_offsets, mask=mask, other=0.0)
    residual = tl.load(
        residual_ptr + row_offsets,
        mask=mask,
        other=0.0,
    )
    updated = (x + residual).to(residual_ptr.dtype.element_ty)
    tl.store(residual_ptr + row_offsets, updated, mask=mask)
    updated_f32 = updated.to(tl.float32)
    variance = tl.sum(updated_f32 * updated_f32, axis=0) / n_cols
    inverse_rms = tl.rsqrt(variance + eps)
    weight = tl.load(
        weight_ptr + offsets,
        mask=mask,
        other=0.0,
    ).to(tl.float32)
    if WEIGHT_OFFSET:
        weight += 1.0
    tl.store(
        x_ptr + row_offsets,
        updated_f32 * inverse_rms * weight,
        mask=mask,
    )


def fused_add_rms_norm(
    x: tensorplay.Tensor,
    residual: tensorplay.Tensor,
    weight: tensorplay.Tensor,
    eps: float,
) -> tuple[tensorplay.Tensor, tensorplay.Tensor]:
    if x.shape != residual.shape:
        raise ValueError("residual and hidden-state shapes must match")
    if x.dtype != residual.dtype or x.device != residual.device:
        raise ValueError("residual and hidden-state placement must match")
    x_contiguous = x.contiguous()
    residual_contiguous = residual.contiguous()
    n_cols = x_contiguous.shape[-1]
    block = triton.next_power_of_2(n_cols)
    if block > 65536:
        raise ValueError(f"RMSNorm width {n_cols} exceeds the Triton guard")
    rows = x_contiguous.numel() // n_cols
    num_warps = 4 if block <= 2048 else 8
    _fused_add_rms_norm_kernel[(rows,)](
        x_contiguous,
        residual_contiguous,
        weight,
        n_cols,
        n_cols=n_cols,
        eps=eps,
        WEIGHT_OFFSET=False,
        BLOCK=block,
        num_warps=num_warps,
    )
    return (
        x_contiguous.view_as(x),
        residual_contiguous.view_as(residual),
    )


def fused_add_offset_rms_norm(
    x: tensorplay.Tensor,
    residual: tensorplay.Tensor,
    weight: tensorplay.Tensor,
    eps: float,
) -> tuple[tensorplay.Tensor, tensorplay.Tensor]:
    """Add the BF16/FP16 residual and apply offset RMSNorm in one kernel."""

    if x.shape != residual.shape:
        raise ValueError("residual and hidden-state shapes must match")
    if x.dtype != residual.dtype or x.device != residual.device:
        raise ValueError("residual and hidden-state placement must match")
    x_contiguous = x.contiguous()
    residual_contiguous = residual.contiguous()
    n_cols = x_contiguous.shape[-1]
    block = triton.next_power_of_2(n_cols)
    if block > 65536:
        raise ValueError(f"RMSNorm width {n_cols} exceeds the Triton guard")
    rows = x_contiguous.numel() // n_cols
    num_warps = 4 if block <= 2048 else 8
    _fused_add_rms_norm_kernel[(rows,)](
        x_contiguous,
        residual_contiguous,
        weight,
        n_cols,
        n_cols=n_cols,
        eps=eps,
        WEIGHT_OFFSET=True,
        BLOCK=block,
        num_warps=num_warps,
    )
    return (
        x_contiguous.view_as(x),
        residual_contiguous.view_as(residual),
    )


__all__ = [
    "fused_add_offset_rms_norm",
    "fused_add_rms_norm",
    "offset_rms_norm",
    "rms_norm",
]
