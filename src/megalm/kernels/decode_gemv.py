"""Fused single-token GEMV for the small-model decode path.

For batch-1 decode, cuBLAS pads the M dimension to 16 and wastes most of its
tensor-core work.  For the OXR Qwen3-0.6B projections the weight matrices are
small enough that a direct bf16 GEMV (fp32 accumulate) is memory-bandwidth
bound and measurably faster than the padded cuBLAS GEMM:

    shape           cuBLAS   triton gemv
    [2688, 1024]    13.6 us    8.4 us     (qkv projection)
    [6144, 1024]    19.1 us   16.9 us     (gate/up projection)

The kernel is graph-capture safe: all tiling constants are compile-time
specializations and no host synchronization happens on the hot path.
"""

from __future__ import annotations

import tensorplay
import triton
import triton.language as tl


@triton.jit
def _gemv_kernel(
    x_ptr,
    w_ptr,
    y_ptr,
    N: tl.constexpr,
    K: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid = tl.program_id(0)
    rows = pid * BLOCK_N + tl.arange(0, BLOCK_N)
    cols = tl.arange(0, BLOCK_K)
    acc = tl.zeros((BLOCK_N, BLOCK_K), dtype=tl.float32)
    for k in range(0, K, BLOCK_K):
        x = tl.load(x_ptr + k + cols, mask=k + cols < K, other=0).to(tl.float32)
        w = tl.load(
            w_ptr + rows[:, None] * K + k + cols[None, :],
            mask=(rows[:, None] < N) & (k + cols[None, :] < K),
            other=0,
        ).to(tl.float32)
        acc += w * x[None, :]
    result = tl.sum(acc, axis=1)
    tl.store(y_ptr + rows, result.to(tl.bfloat16), mask=rows < N)


@triton.jit
def _gate_up_silu_gemv_kernel(
    x_ptr,
    w_ptr,
    y_ptr,
    INTER: tl.constexpr,
    K: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """Compute silu(gate) * up for a single token in one fused launch."""

    pid = tl.program_id(0)
    rows = pid * BLOCK_N + tl.arange(0, BLOCK_N)
    cols = tl.arange(0, BLOCK_K)

    gate_acc = tl.zeros((BLOCK_N, BLOCK_K), dtype=tl.float32)
    up_acc = tl.zeros((BLOCK_N, BLOCK_K), dtype=tl.float32)
    for k in range(0, K, BLOCK_K):
        x = tl.load(x_ptr + k + cols, mask=k + cols < K, other=0).to(tl.float32)
        gate_w = tl.load(
            w_ptr + rows[:, None] * K + k + cols[None, :],
            mask=(rows[:, None] < INTER) & (k + cols[None, :] < K),
            other=0,
        ).to(tl.float32)
        up_w = tl.load(
            w_ptr + (INTER + rows[:, None]) * K + k + cols[None, :],
            mask=(rows[:, None] < INTER) & (k + cols[None, :] < K),
            other=0,
        ).to(tl.float32)
        gate_acc += gate_w * x[None, :]
        up_acc += up_w * x[None, :]

    gate = tl.sum(gate_acc, axis=1)
    up = tl.sum(up_acc, axis=1)
    activated = (gate * tl.sigmoid(gate)) * up
    tl.store(y_ptr + rows, activated.to(tl.bfloat16), mask=rows < INTER)


def gate_up_silu_gemv(
    x: tensorplay.Tensor,
    gate_up_weight: tensorplay.Tensor,
    *,
    block_n: int = 16,
    block_k: int = 128,
    num_warps: int = 8,
) -> tensorplay.Tensor:
    """Fused single-token gate/up GEMV + SiLU activation.

    ``gate_up_weight`` has shape ``[2 * INTER, K]``; the returned tensor has
    shape ``[INTER]`` and equals ``silu(gate) * up``.
    """

    if x.ndim != 1 or gate_up_weight.ndim != 2:
        raise ValueError("gate_up_silu_gemv expects x [K] and weight [2*INTER, K]")
    packed, k = gate_up_weight.shape
    if packed % 2 or x.shape[0] != k:
        raise ValueError("gate_up_silu_gemv has invalid shapes")
    inter = packed // 2
    if not x.is_contiguous() or not gate_up_weight.is_contiguous():
        raise ValueError("gate_up_silu_gemv requires contiguous inputs")

    output = tensorplay.empty((inter,), dtype=x.dtype, device=x.device)
    grid = (triton.cdiv(inter, block_n),)
    _gate_up_silu_gemv_kernel[grid](
        x,
        gate_up_weight,
        output,
        inter,
        k,
        block_n,
        block_k,
        num_warps=num_warps,
    )
    return output


def decode_gemv(
    x: tensorplay.Tensor,
    weight: tensorplay.Tensor,
    *,
    block_n: int = 16,
    block_k: int = 256,
    num_warps: int = 4,
) -> tensorplay.Tensor:
    """Compute ``x @ weight.T`` for a single token ``x`` of shape ``[K]``.

    ``weight`` must be a 2-D row-major ``[N, K]`` tensor and ``x`` a
    contiguous 1-D ``[K]`` tensor. The returned tensor has shape ``[N]``.
    """

    if x.ndim != 1 or weight.ndim != 2:
        raise ValueError("decode_gemv expects x [K] and weight [N, K]")
    n, k = weight.shape
    if x.shape[0] != k:
        raise ValueError("decode_gemv input/weight width mismatch")
    if not x.is_contiguous() or not weight.is_contiguous():
        raise ValueError("decode_gemv requires contiguous inputs")

    # The gate/up projection (N=6144) benefits from a wider warp count and a
    # shorter K tile; the qkv projection (N=2688) is fastest with the defaults.
    if n > 4096:
        block_k = 128
        num_warps = 8

    output = tensorplay.empty((n,), dtype=x.dtype, device=x.device)
    grid = (triton.cdiv(n, block_n),)
    _gemv_kernel[grid](
        x,
        weight,
        output,
        n,
        k,
        block_n,
        block_k,
        num_warps=num_warps,
    )
    return output
