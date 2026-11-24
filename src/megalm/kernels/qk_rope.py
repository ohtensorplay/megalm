"""Fused QK-Norm + RoPE for the Qwen3-VL decode path.

The reference path per layer is: two RMSNorm launches, ``chunk`` copies plus
``cat`` copies inside ``_rotate_half``, four elementwise multiply/add launches,
and two more ``cat`` launches that splice the pass-through channels back on.
Each of those kernels runs on a few-KB tensor, so the chain is launch-latency
bound instead of bandwidth bound.

This kernel emits the final rotated ``q`` and ``k`` with one launch per tensor.
It normalises per head in FP32, rounds to the activation dtype, and reproduces
the reference's per-operation rounding.
"""

from __future__ import annotations

import tensorplay
import triton
import triton.language as tl

# Module switch so tests can pin the reference path for bit-exact comparisons.
FUSED_QK_NORM_ROPE = True
# Merge the two rotations and the paged-cache store into one launch.  Honoured
# only on the plain bf16/fp16 cache, where no quantisation step intervenes.
FUSED_QK_ROPE_KV = True


@triton.jit
def _qk_norm_rope_kernel(
    x_ptr,
    weight_ptr,
    cos_ptr,
    sin_ptr,
    out_ptr,
    row_stride,
    HEADS: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    ROT: tl.constexpr,
    EPS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    width: tl.constexpr = HEADS * HEAD_DIM
    offs = tl.arange(0, BLOCK)
    mask = offs < width
    head = offs // HEAD_DIM
    d = offs % HEAD_DIM
    base = row * row_stride

    values = tl.load(x_ptr + base + offs, mask=mask, other=0.0)
    values_f32 = values.to(tl.float32)
    shaped = tl.reshape(values_f32, (HEADS, HEAD_DIM))
    variance = tl.sum(shaped * shaped, axis=1) / HEAD_DIM
    inverse_rms = tl.rsqrt(variance + EPS)
    weight = tl.load(weight_ptr + tl.arange(0, HEAD_DIM)).to(tl.float32)
    normed = tl.reshape(
        shaped * inverse_rms[:, None] * weight[None, :], (BLOCK,)
    ).to(tl.bfloat16)

    # _rotate_half: the first half rotates onto -second, the second onto +first.
    partner_d = tl.where(d < ROT // 2, d + ROT // 2, d - ROT // 2)
    partner_d = tl.where(d < ROT, partner_d, d)
    partner = tl.load(
        x_ptr + base + head * HEAD_DIM + partner_d, mask=mask, other=0.0
    ).to(tl.float32)
    partner_shaped = tl.reshape(partner, (HEADS, HEAD_DIM))
    # The partner channel carries its own RMSNorm weight, so gather it there.
    partner_weight = tl.reshape(
        tl.load(weight_ptr + partner_d), (HEADS, HEAD_DIM)
    ).to(tl.float32)
    partner_normed = tl.reshape(
        partner_shaped * inverse_rms[:, None] * partner_weight, (BLOCK,)
    ).to(tl.bfloat16)

    sign = tl.where(d < ROT // 2, -1.0, 1.0)
    rotated = partner_normed.to(tl.float32) * sign
    cos = tl.load(cos_ptr + row * ROT + d, mask=mask & (d < ROT), other=0.0)
    sin = tl.load(sin_ptr + row * ROT + d, mask=mask & (d < ROT), other=0.0)
    # Keep every product and the final sum in the activation dtype: the
    # reference dispatches one kernel per operator, so each intermediate is
    # rounded to bf16 and the multiply cannot contract into an FMA.
    rotated_term = (rotated.to(tl.bfloat16).to(tl.float32) * sin.to(
        tl.float32
    )).to(tl.bfloat16)
    direct_term = (normed.to(tl.float32) * cos.to(tl.float32)).to(tl.bfloat16)
    combined = (direct_term.to(tl.float32) + rotated_term.to(tl.float32)).to(
        tl.bfloat16
    )
    tl.store(
        out_ptr + row * width + offs,
        tl.where(d < ROT, combined, normed),
        mask=mask,
    )


@triton.jit
def _qk_norm_rope_kv_kernel(
    q_ptr,
    q_weight_ptr,
    k_ptr,
    k_weight_ptr,
    cos_ptr,
    sin_ptr,
    q_out_ptr,
    k_out_ptr,
    v_ptr,
    key_cache_ptr,
    value_cache_ptr,
    slots_ptr,
    active_ptr,
    q_stride,
    k_stride,
    v_stride,
    cache_stride_slot,
    Q_HEADS: tl.constexpr,
    K_HEADS: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    ROT: tl.constexpr,
    EPS_Q: tl.constexpr,
    EPS_K: tl.constexpr,
    WRITE_CACHE: tl.constexpr,
):
    """QK-norm+RoPE for query and key plus the KV-cache write, in one launch.

    The reference path used three launches per layer for this (two rotations and
    one cache write) on few-KB tensors, and each launch costs more than the
    arithmetic inside it.
    """

    row = tl.program_id(0)

    QB: tl.constexpr = Q_HEADS * HEAD_DIM
    oq = tl.arange(0, QB)
    hq = oq // HEAD_DIM
    dq = oq % HEAD_DIM
    bq = row * q_stride
    q_values = tl.load(q_ptr + bq + oq)
    q_f32 = q_values.to(tl.float32)
    q_shaped = tl.reshape(q_f32, (Q_HEADS, HEAD_DIM))
    q_inverse_rms = tl.rsqrt(
        tl.sum(q_shaped * q_shaped, axis=1) / HEAD_DIM + EPS_Q
    )
    q_weight = tl.load(q_weight_ptr + tl.arange(0, HEAD_DIM)).to(tl.float32)
    q_normed = tl.reshape(
        q_shaped * q_inverse_rms[:, None] * q_weight[None, :], (QB,)
    ).to(tl.bfloat16)
    q_partner_d = tl.where(dq < ROT // 2, dq + ROT // 2, dq - ROT // 2)
    q_partner_d = tl.where(dq < ROT, q_partner_d, dq)
    q_partner = tl.load(q_ptr + bq + hq * HEAD_DIM + q_partner_d).to(tl.float32)
    q_partner_shaped = tl.reshape(q_partner, (Q_HEADS, HEAD_DIM))
    q_partner_weight = tl.reshape(
        tl.load(q_weight_ptr + q_partner_d), (Q_HEADS, HEAD_DIM)
    ).to(tl.float32)
    q_partner_normed = tl.reshape(
        q_partner_shaped * q_inverse_rms[:, None] * q_partner_weight, (QB,)
    ).to(tl.bfloat16)
    q_sign = tl.where(dq < ROT // 2, -1.0, 1.0)
    q_rotated = q_partner_normed.to(tl.float32) * q_sign
    q_cos = tl.load(cos_ptr + row * ROT + dq, mask=dq < ROT, other=0.0)
    q_sin = tl.load(sin_ptr + row * ROT + dq, mask=dq < ROT, other=0.0)
    q_rotated_term = (
        q_rotated.to(tl.bfloat16).to(tl.float32) * q_sin.to(tl.float32)
    ).to(tl.bfloat16)
    q_direct_term = (q_normed.to(tl.float32) * q_cos.to(tl.float32)).to(
        tl.bfloat16
    )
    q_combined = (
        q_direct_term.to(tl.float32) + q_rotated_term.to(tl.float32)
    ).to(tl.bfloat16)
    tl.store(
        q_out_ptr + row * QB + oq,
        tl.where(dq < ROT, q_combined, q_normed),
    )

    KB: tl.constexpr = K_HEADS * HEAD_DIM
    ok = tl.arange(0, KB)
    hk = ok // HEAD_DIM
    dk = ok % HEAD_DIM
    bk = row * k_stride
    k_values = tl.load(k_ptr + bk + ok)
    k_f32 = k_values.to(tl.float32)
    k_shaped = tl.reshape(k_f32, (K_HEADS, HEAD_DIM))
    k_inverse_rms = tl.rsqrt(
        tl.sum(k_shaped * k_shaped, axis=1) / HEAD_DIM + EPS_K
    )
    k_weight = tl.load(k_weight_ptr + tl.arange(0, HEAD_DIM)).to(tl.float32)
    k_normed = tl.reshape(
        k_shaped * k_inverse_rms[:, None] * k_weight[None, :], (KB,)
    ).to(tl.bfloat16)
    k_partner_d = tl.where(dk < ROT // 2, dk + ROT // 2, dk - ROT // 2)
    k_partner_d = tl.where(dk < ROT, k_partner_d, dk)
    k_partner = tl.load(k_ptr + bk + hk * HEAD_DIM + k_partner_d).to(tl.float32)
    k_partner_shaped = tl.reshape(k_partner, (K_HEADS, HEAD_DIM))
    k_partner_weight = tl.reshape(
        tl.load(k_weight_ptr + k_partner_d), (K_HEADS, HEAD_DIM)
    ).to(tl.float32)
    k_partner_normed = tl.reshape(
        k_partner_shaped * k_inverse_rms[:, None] * k_partner_weight, (KB,)
    ).to(tl.bfloat16)
    k_sign = tl.where(dk < ROT // 2, -1.0, 1.0)
    k_rotated = k_partner_normed.to(tl.float32) * k_sign
    k_cos = tl.load(cos_ptr + row * ROT + dk, mask=dk < ROT, other=0.0)
    k_sin = tl.load(sin_ptr + row * ROT + dk, mask=dk < ROT, other=0.0)
    k_rotated_term = (
        k_rotated.to(tl.bfloat16).to(tl.float32) * k_sin.to(tl.float32)
    ).to(tl.bfloat16)
    k_direct_term = (k_normed.to(tl.float32) * k_cos.to(tl.float32)).to(
        tl.bfloat16
    )
    k_combined = (
        k_direct_term.to(tl.float32) + k_rotated_term.to(tl.float32)
    ).to(tl.bfloat16)
    key_out = tl.where(dk < ROT, k_combined, k_normed)
    tl.store(k_out_ptr + row * KB + ok, key_out)

    if WRITE_CACHE:
        live = tl.load(active_ptr + row).to(tl.int32) != 0
        slot = tl.load(slots_ptr + row).to(tl.int64)
        cache_base = slot * cache_stride_slot
        tl.store(key_cache_ptr + cache_base + ok, key_out, mask=live)
        value = tl.load(v_ptr + row * v_stride + ok, mask=live, other=0.0)
        tl.store(value_cache_ptr + cache_base + ok, value, mask=live)


def qk_norm_rope_kv(
    query: tensorplay.Tensor,
    key: tensorplay.Tensor,
    value: tensorplay.Tensor,
    q_weight: tensorplay.Tensor,
    k_weight: tensorplay.Tensor,
    position_embeddings: tuple[tensorplay.Tensor, tensorplay.Tensor],
    *,
    q_heads: int,
    k_heads: int,
    head_dim: int,
    rotary_dim: int,
    q_eps: float,
    k_eps: float,
    dtype: tensorplay.dtype,
    key_cache: tensorplay.Tensor,
    value_cache: tensorplay.Tensor,
    slots: tensorplay.Tensor,
    active: tensorplay.Tensor,
    write_cache: bool = True,
) -> tuple[tensorplay.Tensor, tensorplay.Tensor]:
    """Rotate ``query``/``key`` and store ``key``/``value`` into the paged cache."""

    rows = int(query.shape[0])
    if int(query.shape[1]) != q_heads * head_dim:
        raise ValueError("qk_norm_rope_kv query width does not match the heads")
    if int(key.shape[1]) != k_heads * head_dim:
        raise ValueError("qk_norm_rope_kv key width does not match the heads")
    q_width = q_heads * head_dim
    k_width = k_heads * head_dim
    if q_width & (q_width - 1) or k_width & (k_width - 1):
        raise ValueError("qk_norm_rope_kv needs power-of-two head widths")
    if write_cache:
        if (
            key_cache.stride(1) != head_dim
            or value_cache.stride(1) != head_dim
            or key_cache.stride(0) != k_width
            or value_cache.stride(0) != k_width
            or key_cache.shape[1:] != (k_heads, head_dim)
        ):
            raise ValueError("qk_norm_rope_kv cache geometry differs")
    if key.stride(-1) != 1 or value.stride(-1) != 1:
        raise ValueError("qk_norm_rope_kv needs unit last strides")
    q_out = tensorplay.empty(
        (rows, q_heads, head_dim), dtype=dtype, device=query.device
    )
    k_out = tensorplay.empty(
        (rows, k_heads, head_dim), dtype=dtype, device=key.device
    )
    cos, sin = position_embeddings
    _qk_norm_rope_kv_kernel[(rows,)](
        query,
        q_weight,
        key,
        k_weight,
        cos,
        sin,
        q_out,
        k_out,
        value,
        key_cache,
        value_cache,
        slots,
        active,
        query.stride(0),
        key.stride(0),
        value.stride(0),
        key_cache.stride(0),
        Q_HEADS=q_heads,
        K_HEADS=k_heads,
        HEAD_DIM=head_dim,
        ROT=rotary_dim,
        EPS_Q=q_eps,
        EPS_K=k_eps,
        WRITE_CACHE=write_cache,
        num_warps=4 if max(q_width, k_width) <= 4096 else 8,
    )
    return q_out, k_out


def qk_norm_rope(
    x: tensorplay.Tensor,
    weight: tensorplay.Tensor,
    position_embeddings: tuple[tensorplay.Tensor, tensorplay.Tensor],
    *,
    heads: int,
    head_dim: int,
    rotary_dim: int,
    eps: float,
    dtype: tensorplay.dtype,
) -> tensorplay.Tensor:
    """Return RMS-normalised, RoPE-rotated ``x`` shaped ``[rows, heads, dim]``.

    ``x`` may be a strided view of a fused QKV projection, so its row stride is
    passed explicitly instead of assuming contiguity.
    """

    rows = int(x.shape[0])
    if int(x.shape[1]) != heads * head_dim:
        raise ValueError("qk_norm_rope input width does not match the heads")
    out = tensorplay.empty(
        (rows, heads, head_dim), dtype=dtype, device=x.device
    )
    cos, sin = position_embeddings
    block = heads * head_dim
    _qk_norm_rope_kernel[(rows,)](
        x,
        weight,
        cos,
        sin,
        out,
        x.stride(0),
        HEADS=heads,
        HEAD_DIM=head_dim,
        ROT=rotary_dim,
        EPS=eps,
        BLOCK=block,
        num_warps=4 if block <= 4096 else 8,
    )
    return out