# Copyright 2026 TensorPlay Authors
# SPDX-License-Identifier: Apache-2.0
"""Fused NF4 weight-only GEMM for QLoRA training (Triton).

The QLoRA recipe (TRL / torchao) keeps the frozen base in packed 4-bit and
unpacks nibbles *inside* the GEMM tile, so a step streams only packed bytes and
never materialises a dense bf16 weight.  This module follows that scheduling
structure:

* ``nf4_linear`` is a TensorPlay ``Function``: forward streams P/Q/S/C into one
  tiled matmul; backward recomputes the dequantised tile on the fly for
  ``grad_x = grad_out @ W``.  No dense weight is saved for backward, so
  gradients reach every LoRA target (q/k/v/gate/up included) while peak VRAM
  stays at the packed-weight level.
* Both paths are one kernel launch per projection, replacing the old
  ``dequantize_nf4`` (several elementwise kernels) + ``F.linear`` pair.

Storage contract (identical to ``megatrain.lora.qlora.quantize_nf4``):

* ``packed``: ``[N, K // 2]`` uint8, high nibble = code of even k, low = odd k;
* ``qscalers``: ``[N, K // 64]`` int8, double-quantised block absmax;
* ``qfactor``: ``[N, K // (64 * SB)]`` fp32 factor (ones if not double quant);
* ``mean``: scalar fp32 offset;
* ``codebook``: ``[16]`` fp32 NF4 values indexed by the nibble.

Tiling constants are compile-time specialisations; no host sync on the hot path.
"""

from __future__ import annotations

import tensorplay
import triton
import triton.language as tl
from tensorplay.autograd import Function

__all__ = [
    "nf4_linear", "nf4_gemm_forward", "nf4_gemm_backward", "nf4_linear_scratch",
]


@triton.jit
def _dequant_tile(
    P, QS, QF, MEAN, CB,
    n_offs, k_start,
    K: tl.constexpr, NB: tl.constexpr, NB_SB: tl.constexpr,
    BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    GROUP: tl.constexpr, SB: tl.constexpr, DOUBLE_QUANT: tl.constexpr,
    n_mask,
):
    """Return the dequantised ``[BLOCK_N, BLOCK_K]`` weight tile at ``k_start``."""
    half: tl.constexpr = BLOCK_K // 2
    n_groups: tl.constexpr = BLOCK_K // GROUP
    k_off = tl.arange(0, BLOCK_K)

    byte_tile = tl.load(
        P + n_offs[:, None] * (K // 2) + k_start // 2 + tl.arange(0, half)[None, :],
        mask=n_mask[:, None],
        other=0,
    )
    high = ((byte_tile >> 4) & 0x0F).to(tl.int32)
    low = (byte_tile & 0x0F).to(tl.int32)
    # packed byte holds [code(2j), code(2j+1)]; interleave -> k-major order.
    codes = tl.interleave(high, low)
    values = tl.load(CB + codes, mask=n_mask[:, None], other=0.0).to(tl.float32)

    groups = k_start // GROUP + tl.arange(0, n_groups)
    qs = tl.load(
        QS + n_offs[:, None] * NB + groups[None, :],
        mask=n_mask[:, None],
        other=0.0,
    ).to(tl.float32)
    if DOUBLE_QUANT:
        qf = tl.load(
            QF + n_offs[:, None] * NB_SB + (groups // SB)[None, :],
            mask=n_mask[:, None],
            other=1.0,
        ).to(tl.float32)
        absmax = qs / qf + tl.load(MEAN).to(tl.float32)
    else:
        absmax = qs
    absmax = tl.maximum(absmax, 1.0e-8)

    weight = tl.reshape(
        tl.reshape(values, (BLOCK_N, n_groups, GROUP)) * absmax[:, :, None],
        (BLOCK_N, BLOCK_K),
    )
    # Dequantise straight to the activation dtype, mirroring int4_gemm's
    # DEQUANT_BF16 path: (codebook value * absmax) rounds once to bf16 and
    # halves the tiled weight's shared-memory footprint.
    return weight.to(tl.bfloat16), k_off


@triton.jit
def _nf4_fwd_kernel(
    X, P, QS, QF, MEAN, CB, Y,
    m_size,
    M: tl.constexpr, N: tl.constexpr, K: tl.constexpr,
    NB: tl.constexpr, NB_SB: tl.constexpr,
    GROUP: tl.constexpr, SB: tl.constexpr, DOUBLE_QUANT: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid_n = tl.program_id(0)
    pid_m = tl.program_id(1)
    n_offs = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    m_offs = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    n_mask = n_offs < N
    m_mask = m_offs < m_size

    acc = tl.zeros((BLOCK_M, BLOCK_N), tl.float32)
    for k_start in range(0, K, BLOCK_K):
        x = tl.load(
            X + m_offs[:, None] * K + k_start + tl.arange(0, BLOCK_K)[None, :],
            mask=m_mask[:, None],
            other=0.0,
        )
        weight, _ = _dequant_tile(
            P, QS, QF, MEAN, CB, n_offs, k_start,
            K, NB, NB_SB, BLOCK_N, BLOCK_K, GROUP, SB, DOUBLE_QUANT, n_mask,
        )
        acc = tl.dot(x, tl.trans(weight), acc)
    tl.store(
        Y + m_offs[:, None] * N + n_offs[None, :],
        acc.to(Y.dtype.element_ty),
        mask=m_mask[:, None] & n_mask[None, :],
    )


@triton.jit
def _nf4_bwd_kernel(
    DY, P, QS, QF, MEAN, CB, DX,
    m_size,
    M: tl.constexpr, N: tl.constexpr, K: tl.constexpr,
    NB: tl.constexpr, NB_SB: tl.constexpr,
    GROUP: tl.constexpr, SB: tl.constexpr, DOUBLE_QUANT: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    """``grad_x[M,K] = grad_out[M,N] @ W[N,K]`` with on-the-fly dequant."""
    pid_k = tl.program_id(0)
    pid_m = tl.program_id(1)
    k_offs = pid_k * BLOCK_K + tl.arange(0, BLOCK_K)
    m_offs = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    k_mask = k_offs < K
    m_mask = m_offs < m_size

    acc = tl.zeros((BLOCK_M, BLOCK_K), tl.float32)
    for n_start in range(0, N, BLOCK_N):
        dy = tl.load(
            DY + m_offs[:, None] * N + n_start + tl.arange(0, BLOCK_N)[None, :],
            mask=m_mask[:, None],
            other=0.0,
        )
        n_offs = n_start + tl.arange(0, BLOCK_N)
        weight, _ = _dequant_tile(
            P, QS, QF, MEAN, CB, n_offs, pid_k * BLOCK_K,
            K, NB, NB_SB, BLOCK_N, BLOCK_K, GROUP, SB, DOUBLE_QUANT,
            n_offs < N,
        )
        acc = tl.dot(dy, weight, acc)
    tl.store(
        DX + m_offs[:, None] * K + k_offs[None, :],
        acc.to(DX.dtype.element_ty),
        mask=m_mask[:, None] & k_mask[None, :],
    )


def _validate(x, packed, qscalers, qfactor, codebook, block, sb, double_quant):
    if x.ndim != 2 or packed.ndim != 2:
        raise ValueError("nf4 GEMM expects 2-D activations and packed weights")
    m, k = int(x.shape[0]), int(x.shape[1])
    n = int(packed.shape[0])
    if int(packed.shape[1]) * 2 != k:
        raise ValueError("packed NF4 width must be K / 2")
    if k % block or (double_quant and (k // block) % sb):
        raise ValueError("NF4 block/scaler geometry does not match K")
    if tuple(qscalers.shape) != (n, k // block):
        raise ValueError("qscalers shape does not match [N, K / block]")
    if tuple(qfactor.shape) != (n, k // block // sb if double_quant else k // block):
        raise ValueError("qfactor shape does not match [N, K / (block*sb)]")
    if tuple(codebook.shape) != (16,):
        raise ValueError("NF4 codebook must hold 16 values")
    return m, n, k


def nf4_gemm_forward(
    x, packed, qscalers, qfactor, mean, codebook,
    *, block: int = 64, sb: int = 16, double_quant: bool = True,
):
    """``x @ W.T`` with the frozen weight resident as packed NF4."""
    m, n, k = _validate(x, packed, qscalers, qfactor, codebook, block, sb, double_quant)
    if x.device.type != "cuda" or k % 128:
        from megatrain.lora.qlora import dequantize_nf4
        import tensorplay.nn.functional as F
        state = dict(
            packed=packed, quantized_scalers=qscalers, quantization_factor=qfactor,
            scaler_mean=mean, in_features=k, out_features=n, block_size=block,
            scaler_block_size=sb, double_quant=double_quant,
        )
        return F.linear(x, dequantize_nf4(state, dtype=x.dtype))
    out = tensorplay.empty((m, n), dtype=x.dtype, device=x.device)
    bm, bn, bk = 128, 128, 64
    _nf4_fwd_kernel[(triton.cdiv(n, bn), triton.cdiv(m, bm))](
        x, packed, qscalers, qfactor, mean, codebook, out,
        m, M=m, N=n, K=k, NB=k // block, NB_SB=k // block // sb if double_quant else k // block,
        GROUP=block, SB=sb, DOUBLE_QUANT=double_quant,
        BLOCK_M=bm, BLOCK_N=bn, BLOCK_K=bk,
        num_warps=8, num_stages=3,
    )
    return out


def nf4_gemm_backward(
    grad_out, packed, qscalers, qfactor, mean, codebook,
    *, block: int = 64, sb: int = 16, double_quant: bool = True,
):
    """``grad_out @ W`` with the frozen weight dequantised inside the tile."""
    n = int(packed.shape[0])
    k = int(packed.shape[1]) * 2
    m = int(grad_out.shape[0])
    if tuple(grad_out.shape) != (m, n):
        raise ValueError("grad_out shape must be [M, N] of the forward pass")
    if int(packed.shape[1]) * 2 != k:
        raise ValueError("packed NF4 width must be K / 2")
    if k % block or (double_quant and (k // block) % sb):
        raise ValueError("NF4 block/scaler geometry does not match K")
    if tuple(qscalers.shape) != (n, k // block):
        raise ValueError("qscalers shape does not match [N, K / block]")
    if tuple(qfactor.shape) != (n, k // block // sb if double_quant else k // block):
        raise ValueError("qfactor shape does not match [N, K / (block*sb)]")
    if tuple(codebook.shape) != (16,):
        raise ValueError("NF4 codebook must hold 16 values")
    if grad_out.device.type != "cuda" or k % 128:
        from megatrain.lora.qlora import dequantize_nf4
        state = dict(
            packed=packed, quantized_scalers=qscalers, quantization_factor=qfactor,
            scaler_mean=mean, in_features=k, out_features=n, block_size=block,
            scaler_block_size=sb, double_quant=double_quant,
        )
        return grad_out @ dequantize_nf4(state, dtype=grad_out.dtype)
    dx = tensorplay.empty((m, k), dtype=grad_out.dtype, device=grad_out.device)
    bm, bn, bk = 64, 64, 64
    _nf4_bwd_kernel[(triton.cdiv(k, bk), triton.cdiv(m, bm))](
        grad_out, packed, qscalers, qfactor, mean, codebook, dx,
        m, M=m, N=n, K=k, NB=k // block, NB_SB=k // block // sb if double_quant else k // block,
        GROUP=block, SB=sb, DOUBLE_QUANT=double_quant,
        BLOCK_M=bm, BLOCK_N=bn, BLOCK_K=bk,
        num_warps=4, num_stages=3,
    )
    return dx


class _NF4LinearFn(Function):
    """Autograd wrapper: fused NF4 matmul in both directions, no dense weight."""

    @staticmethod
    def forward(ctx, x, packed, qscalers, qfactor, mean, codebook, block, sb, dq):
        ctx.save_for_backward(packed, qscalers, qfactor, mean, codebook)
        ctx.block, ctx.sb, ctx.dq = block, sb, dq
        return nf4_gemm_forward(
            x, packed, qscalers, qfactor, mean, codebook,
            block=block, sb=sb, double_quant=dq,
        )

    @staticmethod
    def backward(ctx, grad_output):
        packed, qscalers, qfactor, mean, codebook = ctx.saved_tensors
        grad_x = nf4_gemm_backward(
            grad_output.contiguous(), packed, qscalers, qfactor, mean, codebook,
            block=ctx.block, sb=ctx.sb, double_quant=ctx.dq,
        )
        return grad_x, None, None, None, None, None, None, None, None


def nf4_linear(
    x, packed, qscalers, qfactor, mean, codebook,
    *, block: int = 64, sb: int = 16, double_quant: bool = True,
):
    """Differentiable ``x @ W.T`` with ``W`` held in packed NF4 form."""
    return _NF4LinearFn.apply(
        x, packed, qscalers, qfactor, mean, codebook, block, sb, double_quant
    )


# ---------------------------------------------------------------------------
# Shared-scratch path: dequantise into one grow-only buffer, then use cuBLAS.
# ---------------------------------------------------------------------------
@triton.jit
def _nf4_dequant_kernel(
    P, QS, QF, MEAN, CB, OUT,
    N: tl.constexpr, K: tl.constexpr, NB: tl.constexpr, NB_SB: tl.constexpr,
    GROUP: tl.constexpr, SB: tl.constexpr, DOUBLE_QUANT: tl.constexpr,
    BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid_n = tl.program_id(0)
    pid_k = tl.program_id(1)
    n_offs = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    k_offs = pid_k * BLOCK_K + tl.arange(0, BLOCK_K)
    weight, _ = _dequant_tile(
        P, QS, QF, MEAN, CB, n_offs, pid_k * BLOCK_K,
        K, NB, NB_SB, BLOCK_N, BLOCK_K, GROUP, SB, DOUBLE_QUANT,
        n_offs < N,
    )
    tl.store(
        OUT + n_offs[:, None] * K + k_offs[None, :],
        weight,
        mask=(n_offs[:, None] < N),
    )


_SCRATCH: dict[str, tensorplay.Tensor] = {}


def _scratch(numel: int, device, dtype):
    """One grow-only dense buffer per (device, dtype), reused by every layer."""
    key = f"{device}|{dtype}"
    buf = _SCRATCH.get(key)
    if buf is None or int(buf.numel()) < numel:
        buf = tensorplay.empty((numel,), dtype=dtype, device=device)
        _SCRATCH[key] = buf
    return buf[:numel]


def _dequant_into(packed, qscalers, qfactor, mean, codebook, out,
                  *, block, sb, double_quant):
    n = int(packed.shape[0])
    k = int(packed.shape[1]) * 2
    if out.numel() < n * k:
        raise ValueError("NF4 scratch buffer is too small")
    if packed.device.type != "cuda":
        from megatrain.lora.qlora import dequantize_nf4

        state = dict(
            packed=packed, quantized_scalers=qscalers, quantization_factor=qfactor,
            scaler_mean=mean, in_features=k, out_features=n, block_size=block,
            scaler_block_size=sb, double_quant=double_quant,
        )
        out[: n * k] = dequantize_nf4(state, dtype=out.dtype).reshape(-1)
        return out[: n * k].view(n, k)
    bn, bk = 64, 256
    _nf4_dequant_kernel[(triton.cdiv(n, bn), triton.cdiv(k, bk))](
        packed, qscalers, qfactor, mean, codebook, out,
        N=n, K=k, NB=k // block, NB_SB=k // block // sb if double_quant else k // block,
        GROUP=block, SB=sb, DOUBLE_QUANT=double_quant,
        BLOCK_N=bn, BLOCK_K=bk,
        num_warps=4, num_stages=2,
    )
    return out[: n * k].view(n, k)


class _NF4ScratchFn(Function):
    """Dequantise into shared scratch + cuBLAS GEMM; re-dequantise in backward.

    ``Function.forward`` runs without graph tracking, so the scratch tensor is
    never captured by autograd: peak VRAM holds only the packed base plus one
    grow-only dense buffer (the largest projection), not 168 dense weights.
    """

    @staticmethod
    def forward(ctx, x, packed, qscalers, qfactor, mean, codebook, block, sb, dq):
        import tensorplay.nn.functional as F

        ctx.save_for_backward(packed, qscalers, qfactor, mean, codebook)
        ctx.block, ctx.sb, ctx.dq = block, sb, dq
        n = int(packed.shape[0])
        k = int(packed.shape[1]) * 2
        scratch = _scratch(n * k, packed.device, x.dtype)
        weight = _dequant_into(
            packed, qscalers, qfactor, mean, codebook, scratch,
            block=block, sb=sb, double_quant=dq,
        )
        return F.linear(x, weight)

    @staticmethod
    def backward(ctx, grad_output):
        packed, qscalers, qfactor, mean, codebook = ctx.saved_tensors
        n = int(packed.shape[0])
        k = int(packed.shape[1]) * 2
        scratch = _scratch(n * k, packed.device, grad_output.dtype)
        weight = _dequant_into(
            packed, qscalers, qfactor, mean, codebook, scratch,
            block=ctx.block, sb=ctx.sb, double_quant=ctx.dq,
        )
        return grad_output @ weight, None, None, None, None, None, None, None, None


def nf4_linear_scratch(
    x, packed, qscalers, qfactor, mean, codebook,
    *, block: int = 64, sb: int = 16, double_quant: bool = True,
):
    """``x @ W.T`` via shared dense scratch + cuBLAS, differentiable."""
    return _NF4ScratchFn.apply(
        x, packed, qscalers, qfactor, mean, codebook, block, sb, double_quant
    )
