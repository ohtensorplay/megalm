"""Fused clipping and normalization of the Inkling-2 attention heads."""

from __future__ import annotations

import tensorplay
import triton
import triton.language as tl


@triton.jit
def _clip_l2(values, threshold, eps):
    length = tl.sqrt_rn(tl.sum(values * values, axis=0))
    coefficient = tl.minimum(tl.div_rn(threshold, length + eps), 1.0)
    return values * coefficient


@triton.jit
def _clip_projected_head(
    projection,
    gain,
    tau,
    output,
    row,
    head,
    lanes,
    HEADS: tl.constexpr,
    WIDTH: tl.constexpr,
    THRESHOLD: tl.constexpr,
    EPS: tl.constexpr,
    SCALE_BY_TAU: tl.constexpr,
):
    valid = lanes < WIDTH
    offset = (row * HEADS + head) * WIDTH + lanes
    values = tl.load(projection + offset, valid, 0)
    per_channel = tl.load(gain + head * WIDTH + lanes, valid, 0)
    # The checkpoint applies the channel gain in the model dtype before clipping.
    gained = (values.to(tl.float32) * per_channel.to(tl.float32)).to(values.dtype)
    clipped = _clip_l2(gained.to(tl.float32), THRESHOLD, EPS).to(values.dtype)
    if SCALE_BY_TAU:
        clipped = (clipped.to(tl.float32) * tl.load(tau + row)).to(values.dtype)
    tl.store(output + offset, clipped, valid)


@triton.jit
def _prepare_attention_heads(
    q,
    k,
    v,
    relative,
    query_gain,
    relative_gain,
    tau,
    out_q,
    out_k,
    out_v,
    out_relative,
    HEADS: tl.constexpr,
    KV_HEADS: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    REL_DIM: tl.constexpr,
    QK_THRESHOLD: tl.constexpr,
    REL_THRESHOLD: tl.constexpr,
    EPS: tl.constexpr,
    SCALE_BY_TAU: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    stream_head = tl.program_id(1)
    lanes = tl.arange(0, BLOCK)
    if stream_head < HEADS:
        _clip_projected_head(
            q,
            query_gain,
            tau,
            out_q,
            row,
            stream_head,
            lanes,
            HEADS,
            HEAD_DIM,
            QK_THRESHOLD,
            EPS,
            SCALE_BY_TAU,
        )
    elif stream_head < HEADS + KV_HEADS:
        kv_head = stream_head - HEADS
        offset = (row * KV_HEADS + kv_head) * HEAD_DIM + lanes
        valid = lanes < HEAD_DIM
        keys = tl.load(k + offset, valid, 0).to(tl.float32)
        tl.store(out_k + offset, _clip_l2(keys, QK_THRESHOLD, EPS), valid)
    elif stream_head < HEADS + 2 * KV_HEADS:
        kv_head = stream_head - HEADS - KV_HEADS
        offset = (row * KV_HEADS + kv_head) * HEAD_DIM + lanes
        valid = lanes < HEAD_DIM
        values = tl.load(v + offset, valid, 0).to(tl.float32)
        inverse = tl.rsqrt(tl.sum(values * values, axis=0) / HEAD_DIM + EPS)
        tl.store(out_v + offset, values * inverse, valid)
    else:
        relative_head = stream_head - HEADS - 2 * KV_HEADS
        _clip_projected_head(
            relative,
            relative_gain,
            tau,
            out_relative,
            row,
            relative_head,
            lanes,
            HEADS,
            REL_DIM,
            REL_THRESHOLD,
            EPS,
            SCALE_BY_TAU,
        )


def prepare_inkling2_heads(
    q: tensorplay.Tensor,
    k: tensorplay.Tensor,
    v: tensorplay.Tensor,
    relative: tensorplay.Tensor,
    query_gain: tensorplay.Tensor,
    relative_gain: tensorplay.Tensor,
    tau: tensorplay.Tensor | None,
    *,
    qk_threshold: float,
    rel_threshold: float,
    eps: float,
) -> tuple[tensorplay.Tensor, tensorplay.Tensor, tensorplay.Tensor, tensorplay.Tensor]:
    """Apply per-head clipping, value RMSNorm, and optional query log scaling."""

    rows, heads, head_dim = q.shape
    kv_heads, rel_dim = k.shape[1], relative.shape[-1]
    out_q = tensorplay.empty_like(q)
    out_k = tensorplay.empty_like(k)
    out_v = tensorplay.empty_like(v)
    out_relative = tensorplay.empty_like(relative)
    _prepare_attention_heads[(rows, 2 * (heads + kv_heads))](
        q,
        k,
        v,
        relative,
        query_gain,
        relative_gain,
        tau if tau is not None else q,
        out_q,
        out_k,
        out_v,
        out_relative,
        heads,
        kv_heads,
        head_dim,
        rel_dim,
        qk_threshold,
        rel_threshold,
        eps,
        tau is not None,
        triton.next_power_of_2(max(head_dim, rel_dim)),
        num_warps=4,
        enable_fp_fusion=False,
    )
    return out_q, out_k, out_v, out_relative
