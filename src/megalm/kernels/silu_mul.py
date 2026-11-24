from __future__ import annotations

import tensorplay
import triton
import triton.language as tl
from triton.language.extra.cuda import libdevice


@triton.jit
def _silu_and_mul_kernel(
    gate_up_ptr,
    output_ptr,
    intermediate_size: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    offsets = tl.arange(0, BLOCK)
    mask = offsets < intermediate_size
    row_base = row * intermediate_size * 2
    gate = tl.load(gate_up_ptr + row_base + offsets, mask=mask, other=0.0)
    up = tl.load(
        gate_up_ptr + row_base + intermediate_size + offsets,
        mask=mask,
        other=0.0,
    )
    gate_f32 = gate.to(tl.float32)
    activated = gate_f32 * tl.sigmoid(gate_f32) * up.to(tl.float32)
    tl.store(output_ptr + row * intermediate_size + offsets, activated, mask=mask)


@triton.jit
def _silu_and_mul_pairs_kernel(
    gate_up_ptr,
    output_ptr,
    intermediate_size: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    offsets = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < intermediate_size
    base = row * intermediate_size * 2 + offsets * 2
    gate = tl.load(gate_up_ptr + base, mask=mask, other=0.0).to(tl.float32)
    up = tl.load(gate_up_ptr + base + 1, mask=mask, other=0.0).to(tl.float32)
    tl.store(
        output_ptr + row * intermediate_size + offsets,
        gate * tl.sigmoid(gate) * up,
        mask=mask,
    )


def silu_and_mul_pairs(gate_up: tensorplay.Tensor) -> tensorplay.Tensor:
    """``silu(gate) * up`` over an element-interleaved ``[g0, u0, g1, u1, ...]``.

    The gate/up projection is repacked at load so both operands of the
    activation share a column tile; the decode path folds this into the GEMM
    store, so this kernel covers the dense prefill path.
    """

    gate_up_contiguous = gate_up.contiguous()
    intermediate_size = gate_up_contiguous.shape[-1] // 2
    output = tensorplay.empty(
        (*gate_up_contiguous.shape[:-1], intermediate_size),
        dtype=gate_up_contiguous.dtype,
        device=gate_up_contiguous.device,
    )
    rows = output.numel() // intermediate_size
    block = 2048 if intermediate_size >= 2048 else triton.next_power_of_2(
        intermediate_size
    )
    _silu_and_mul_pairs_kernel[(rows, triton.cdiv(intermediate_size, block))](
        gate_up_contiguous,
        output,
        intermediate_size=intermediate_size,
        BLOCK=block,
        num_warps=8 if block >= 2048 else 4,
    )
    return output


def silu_and_mul(gate_up: tensorplay.Tensor) -> tensorplay.Tensor:
    gate_up_contiguous = gate_up.contiguous()
    intermediate_size = gate_up_contiguous.shape[-1] // 2
    output = tensorplay.empty(
        (*gate_up_contiguous.shape[:-1], intermediate_size),
        dtype=gate_up_contiguous.dtype,
        device=gate_up_contiguous.device,
    )
    rows = output.numel() // intermediate_size
    block = triton.next_power_of_2(intermediate_size)
    _silu_and_mul_kernel[(rows,)](
        gate_up_contiguous,
        output,
        intermediate_size=intermediate_size,
        BLOCK=block,
        num_warps=8 if block >= 2048 else 4,
    )
    return output


@triton.jit
def _capped_swiglu_kernel(
    gate,
    up,
    output,
    WIDTH: tl.constexpr,
    GATE_ROW: tl.constexpr,
    GATE_COL: tl.constexpr,
    UP_ROW: tl.constexpr,
    UP_COL: tl.constexpr,
    GATE_CAP: tl.constexpr,
    UP_CAP: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    column = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    valid = column < WIDTH
    gate_value = tl.load(gate + row * GATE_ROW + column * GATE_COL, valid, 0)
    gate_value = gate_value.to(tl.float32)
    up_value = tl.load(up + row * UP_ROW + column * UP_COL, valid, 0)
    up_value = up_value.to(tl.float32)
    sigmoid = tl.div_rn(1.0, 1.0 + libdevice.exp(-gate_value))
    if GATE_CAP > 0:
        gate_value = libdevice.tanh(tl.div_rn(gate_value, GATE_CAP)) * GATE_CAP
    if UP_CAP > 0:
        up_value = libdevice.tanh(tl.div_rn(up_value, UP_CAP)) * UP_CAP
    tl.store(output + row * WIDTH + column, gate_value * sigmoid * up_value, valid)


def capped_swiglu_cuda(
    gate: tensorplay.Tensor,
    up: tensorplay.Tensor,
    gate_cap: float,
    up_cap: float,
) -> tensorplay.Tensor:
    """Apply Inkling's capped SwiGLU to separate or interleaved expert rows."""
    rows, width = gate.shape
    output = tensorplay.empty(gate.shape, dtype=gate.dtype, device=gate.device)
    block = 128
    _capped_swiglu_kernel[(rows, triton.cdiv(width, block))](
        gate,
        up,
        output,
        width,
        *gate.stride(),
        *up.stride(),
        gate_cap,
        up_cap,
        block,
        num_warps=4,
        enable_fp_fusion=False,
    )
    return output
