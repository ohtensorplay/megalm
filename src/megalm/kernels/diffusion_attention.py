"""Fused attention for contiguous DiffusionGemma prefix and canvas tensors."""

from __future__ import annotations

import tensorplay
import triton
import triton.language as tl


@triton.jit(do_not_specialize=["query_len", "key_len"])
def _attention(
    q_ptr,
    k_ptr,
    v_ptr,
    out_ptr,
    query_len,
    key_len,
    HEADS: tl.constexpr,
    KV_HEADS: tl.constexpr,
    DIM: tl.constexpr,
    WINDOW: tl.constexpr,
    CAUSAL: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
):
    head = tl.program_id(1)
    kv_head = head // (HEADS // KV_HEADS)
    q_start = tl.program_id(0) * BM
    qi = q_start + tl.arange(0, BM)
    q_position = key_len - query_len + qi
    d = tl.arange(0, DIM)
    query_offset = (qi[:, None] * HEADS + head) * DIM + d[None, :]
    query = tl.load(q_ptr + query_offset, qi[:, None] < query_len, 0)

    key_start = 0
    if WINDOW:
        earliest = tl.maximum(key_len - query_len + q_start - WINDOW + 1, 0)
        key_start = earliest // BN * BN
    key_end = key_len
    if CAUSAL:
        key_end = tl.minimum(key_len, key_len - query_len + q_start + BM)

    running_max = tl.full((BM,), -float("inf"), tl.float32)
    running_sum = tl.full((BM,), 0.0, tl.float32)
    acc = tl.full((BM, DIM), 0.0, tl.float32)
    for start in range(key_start, key_end, BN):
        ki = start + tl.arange(0, BN)
        kv_offset = (ki[:, None] * KV_HEADS + kv_head) * DIM + d[None, :]
        keys = tl.load(k_ptr + kv_offset, ki[:, None] < key_len, 0)
        values = tl.load(v_ptr + kv_offset, ki[:, None] < key_len, 0)

        scores = tl.dot(query, tl.trans(keys)).to(query.dtype).to(tl.float32)
        visible = (ki[None, :] < key_len) & (qi[:, None] < query_len)
        if CAUSAL:
            visible &= ki[None, :] <= q_position[:, None]
        if WINDOW:
            visible &= ki[None, :] > q_position[:, None] - WINDOW
        scores = tl.where(visible, scores, -float("inf"))
        maximum = tl.maximum(running_max, tl.max(scores, axis=1))
        finite_max = tl.where(maximum == -float("inf"), 0.0, maximum)
        alpha = tl.exp(running_max - finite_max)
        weights = tl.exp(scores - finite_max[:, None])
        running_sum = running_sum * alpha + tl.sum(weights, axis=1)
        acc = tl.dot(weights.to(query.dtype), values, acc * alpha[:, None])
        running_max = maximum

    result = acc / tl.maximum(running_sum[:, None], 1e-20)
    tl.store(out_ptr + query_offset, result, qi[:, None] < query_len)


def diffusion_attention(
    query: tensorplay.Tensor,
    keys: tensorplay.Tensor,
    values: tensorplay.Tensor,
    *,
    causal: bool,
    sliding_window: int,
) -> tensorplay.Tensor:
    query_len, heads, head_dim = query.shape
    key_len, kv_heads, _ = keys.shape
    output = tensorplay.empty(
        (query_len, heads * head_dim), device=query.device, dtype=query.dtype
    )
    block_m, block_n = (32, 32) if head_dim <= 256 else (16, 32)
    _attention[(triton.cdiv(query_len, block_m), heads)](
        query,
        keys,
        values,
        output,
        query_len,
        key_len,
        heads,
        kv_heads,
        head_dim,
        sliding_window,
        causal,
        block_m,
        block_n,
        num_warps=4,
        num_stages=2,
    )
    return output


@triton.jit
def _norm_rope(
    src,
    weight,
    cos,
    sin,
    dst,
    token,
    head,
    stride,
    HEADS: tl.constexpr,
    HALF: tl.constexpr,
    EPS: tl.constexpr,
):
    i = tl.arange(0, HALF)
    offset = token * stride + head * (2 * HALF) + i
    first = tl.load(src + offset).to(tl.float32)
    second = tl.load(src + offset + HALF).to(tl.float32)
    variance = tl.sum(first * first + second * second, axis=0) / (2 * HALF)
    inverse = tl.rsqrt(variance + EPS)
    dtype: tl.constexpr = src.dtype.element_ty
    first = (
        (first * inverse * tl.load(weight + i).to(tl.float32)).to(dtype).to(tl.float32)
    )
    second = (
        (second * inverse * tl.load(weight + i + HALF).to(tl.float32))
        .to(dtype)
        .to(tl.float32)
    )
    c = tl.load(cos + token * HALF + i).to(dtype).to(tl.float32)
    s = tl.load(sin + token * HALF + i).to(dtype).to(tl.float32)
    # Preserve the eager BF16 rounding at each product before combining halves.
    y0 = (first * c).to(dtype).to(tl.float32) - (second * s).to(dtype).to(tl.float32)
    y1 = (second * c).to(dtype).to(tl.float32) + (first * s).to(dtype).to(tl.float32)
    output_offset = (token * HEADS + head) * (2 * HALF) + i
    tl.store(dst + output_offset, y0)
    tl.store(dst + output_offset + HALF, y1)


@triton.jit
def _prepare_qkv(
    q,
    k,
    v,
    q_weight,
    k_weight,
    cos,
    sin,
    q_out,
    k_out,
    v_out,
    q_stride,
    k_stride,
    v_stride,
    Q_HEADS: tl.constexpr,
    KV_HEADS: tl.constexpr,
    DIM: tl.constexpr,
    EPS: tl.constexpr,
):
    token = tl.program_id(0)
    head = tl.program_id(1)
    if head < Q_HEADS:
        _norm_rope(
            q, q_weight, cos, sin, q_out, token, head, q_stride, Q_HEADS, DIM // 2, EPS
        )
    else:
        kv_head = head - Q_HEADS
        _norm_rope(
            k,
            k_weight,
            cos,
            sin,
            k_out,
            token,
            kv_head,
            k_stride,
            KV_HEADS,
            DIM // 2,
            EPS,
        )
        i = tl.arange(0, DIM)
        raw = tl.load(v + token * v_stride + kv_head * DIM + i).to(tl.float32)
        value = raw * tl.rsqrt(tl.sum(raw * raw, axis=0) / DIM + EPS)
        tl.store(v_out + (token * KV_HEADS + kv_head) * DIM + i, value)


def prepare_diffusion_qkv(
    query: tensorplay.Tensor,
    key: tensorplay.Tensor,
    value: tensorplay.Tensor,
    q_norm: tensorplay.Tensor,
    k_norm: tensorplay.Tensor,
    cos: tensorplay.Tensor,
    sin: tensorplay.Tensor,
    eps: float,
) -> tuple[tensorplay.Tensor, tensorplay.Tensor, tensorplay.Tensor]:
    tokens, heads, head_dim = query.shape
    kv_heads = int(key.shape[1])
    outputs = tuple(
        tensorplay.empty(t.shape, device=t.device, dtype=t.dtype)
        for t in (query, key, value)
    )
    _prepare_qkv[(tokens, heads + kv_heads)](
        query,
        key,
        value,
        q_norm,
        k_norm,
        cos,
        sin,
        *outputs,
        query.stride(0),
        key.stride(0),
        value.stride(0),
        heads,
        kv_heads,
        head_dim,
        eps,
        num_warps=4,
        enable_fp_fusion=False,
    )
    return outputs
