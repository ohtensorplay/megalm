from __future__ import annotations

import tensorplay
import tensorplay.nn.functional as F

from megalm.kernels.kv_quant_triton import (
    get_rht_matrices,
    rht_enabled,
)
from megalm.runtime.kv_quant import (
    KVQuantMode,
    dequantize_tensor,
    kv_quant_of,
    kv_scale_of,
    quantize_tensor,
)


def _rht_forward(values, quant, head_dim: int):
    if not rht_enabled(quant):
        return values
    matrix, signs = get_rht_matrices(head_dim, values.device)
    # vLLM forward RHT is H @ (signs ⊙ x); in row form (x ⊙ signs) @ H.
    return (values.float() * signs) @ matrix


def _rht_inverse(values, quant, head_dim: int):
    if not rht_enabled(quant):
        return values
    matrix, signs = get_rht_matrices(head_dim, values.device)
    return ((values.float() @ matrix) * signs) / head_dim

# ---------------------------------------------------------------------------
# Generic (Triton-free) quantized KV cache helpers
#
# These operate on the raw cache tensor produced by ``PagedKVCache.layer``:
# shape ``(pages, page_size, heads, content_dim)`` with quantization metadata
# (and, for the dynamic modes, an inline fp32 scale view) attached.  The
# helpers restore the logical ``head_dim`` so a plain ``index_select`` can be
# used to gather K/V afterwards.
# ---------------------------------------------------------------------------

_E2M1_CODES = (
    0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
    0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0,
)


def _cache_head_dim(quant, content_dim: int) -> int:
    """Recover the logical head_dim from a packed cache row width."""
    if quant.mode == KVQuantMode.INT4_PER_TOKEN_HEAD:
        return (content_dim - 4) * 2
    if quant.mode == KVQuantMode.NVFP4:
        return content_dim * 16 // 9
    if quant.mode == KVQuantMode.FP8_PER_TOKEN_HEAD:
        return content_dim - 4
    return content_dim


def _unpack_nibbles(packed):
    """Expand packed uint8 bytes into interleaved low/high 4-bit codes."""
    values = packed.to(tensorplay.int32)
    lo = values & 0xF
    hi = (values >> 4) & 0xF
    return tensorplay.stack((lo, hi), dim=-1).reshape(
        *packed.shape[:-1], packed.shape[-1] * 2
    )


def _pack_nibbles(codes) -> tensorplay.Tensor:
    """Pack two consecutive 4-bit codes into one uint8 byte."""
    lo = codes[..., 0::2].to(tensorplay.int32)
    hi = codes[..., 1::2].to(tensorplay.int32)
    return ((lo & 0xF) | ((hi & 0xF) << 4)).to(tensorplay.uint8)


def _decode_e2m1(codes):
    table = tensorplay.tensor(
        _E2M1_CODES, dtype=tensorplay.float32, device=codes.device
    )
    return table[codes.long()]


def _encode_e2m1(values):
    v = values.to(tensorplay.float32)
    magnitudes = tensorplay.tensor(
        (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0),
        dtype=tensorplay.float32,
        device=v.device,
    ).reshape((1,) * (v.ndim - 1) + (8,))
    distance = tensorplay.abs(magnitudes - tensorplay.abs(v).unsqueeze(-1))
    best = tensorplay.argmin(distance, dim=-1)
    return best + (v < 0).to(tensorplay.int32) * 8


def _decode_e4m3(byte_tensor):
    bits = byte_tensor.to(tensorplay.int32)
    sign = 1.0 - 2.0 * (((bits >> 7) & 1).to(tensorplay.float32))
    exponent = (bits >> 3) & 0xF
    mantissa_f = (bits & 0x7).to(tensorplay.float32) / 8.0
    subnormal = mantissa_f / 64.0  # 2^-6 * mant / 8
    normal = tensorplay.exp2(exponent.to(tensorplay.float32) - 7.0) * (
        1.0 + mantissa_f
    )
    special = tensorplay.where(
        (bits & 0x7) == 0,
        tensorplay.full_like(normal, float("inf")),
        tensorplay.full_like(normal, float("nan")),
    )
    magnitude = tensorplay.where(
        exponent == 0,
        subnormal,
        tensorplay.where(exponent == 15, special, normal),
    )
    return sign * magnitude


def _encode_e4m3(values):
    v = values.to(tensorplay.float32)
    magnitude = tensorplay.abs(v)
    exponent = tensorplay.floor(
        tensorplay.log2(tensorplay.clamp(magnitude, min=1e-12))
    ).to(tensorplay.int32)
    exponent = tensorplay.clamp(exponent + 7, 1, 14)
    mantissa_f = magnitude / tensorplay.exp2((exponent - 7).to(tensorplay.float32))
    mantissa = tensorplay.floor((mantissa_f - 1.0) * 8.0 + 0.5).to(tensorplay.int32)
    mantissa = tensorplay.clamp(mantissa, 0, 7)
    bits = (exponent << 3) | mantissa
    bits = bits | ((v < 0).to(tensorplay.int32) << 7)
    return bits.to(tensorplay.uint8)


def dequantize_cache_to_dtype(
    cache_tensor: tensorplay.Tensor,
    quant,
    which: str,
    output_dtype: tensorplay.dtype,
    scale: tensorplay.Tensor | None = None,
) -> tensorplay.Tensor:
    """Decode a whole quantized KV cache (or a gathered tile) to ``output_dtype``.

    The returned tensor has the same leading dimensions as ``cache_tensor`` and
    the logical ``head_dim`` as its last dimension.  ``scale`` is the optional
    inline scale for dynamic modes; when omitted it is read from
    ``kv_scale_of(cache_tensor)``.
    """
    if quant is None:
        return cache_tensor.to(output_dtype)
    mode = quant.mode
    content_dim = cache_tensor.shape[-1]
    head_dim = _cache_head_dim(quant, content_dim)
    data = cache_tensor[..., :head_dim]
    if mode in (KVQuantMode.FP8_PER_TENSOR, KVQuantMode.FP8_PER_HEAD):
        return dequantize_tensor(data, quant, which, output_dtype)
    if mode == KVQuantMode.FP8_PER_TOKEN_HEAD:
        inline = kv_scale_of(cache_tensor) if scale is None else scale
        if inline is None:
            raise RuntimeError("fp8_per_token_head cache has no inline scale view")
        return tensorplay.dequantize_affine_float8(
            data, inline[..., None], output_dtype
        )
    if mode == KVQuantMode.INT4_PER_TOKEN_HEAD:
        codes = _unpack_nibbles(
            cache_tensor[..., : head_dim // 2]
        ).to(tensorplay.float32)
        inline = kv_scale_of(cache_tensor) if scale is None else scale
        if inline is None:
            raise RuntimeError("int4_per_token_head cache has no inline scale view")
        bits = inline.view(tensorplay.int32)
        zero_point = (bits & 0xF).to(tensorplay.float32)
        values = (bits & -16).view(tensorplay.float32)
        decoded = (codes - zero_point[..., None]) * values[..., None]
        # The Triton INT4 write kernel stores RHT-rotated values; undo the
        # rotation so Python gather/SDPA paths see the original K/V space.
        return _rht_inverse(decoded, quant, head_dim).to(output_dtype)
    if mode == KVQuantMode.NVFP4:
        codes = _unpack_nibbles(cache_tensor[..., : head_dim // 2])
        scale_bytes = cache_tensor[
            ..., head_dim // 2 : head_dim // 2 + head_dim // 16
        ]
        block_scale = _decode_e4m3(scale_bytes).repeat_interleave(16, dim=-1)
        values = (
            _decode_e2m1(codes)
            * block_scale
            * float(quant.nvfp4_global_scale)
        )
        return values.to(output_dtype)
    raise RuntimeError(f"unsupported KV quantization mode: {quant.mode}")


def _write_inline_scale(cache_tensor, slots, values, page_size: int) -> None:
    view = kv_scale_of(cache_tensor)
    if view is None:
        raise RuntimeError("dynamic KV cache has no inline scale view")
    view[slots // page_size, slots % page_size] = values.to(view.dtype)


def quantize_cache_rows(
    values: tensorplay.Tensor,
    cache_tensor: tensorplay.Tensor,
    quant,
    which: str,
    slots: tensorplay.Tensor,
    *,
    page_size: int,
) -> None:
    """Quantize ``[tokens, heads, head_dim]`` values into cache rows at ``slots``.

    The Python implementation covers every mode (including the dynamic ones)
    without requiring Triton, and writes the inline scale where applicable.
    """
    heads = cache_tensor.shape[-2]
    content_dim = cache_tensor.shape[-1]
    head_dim = values.shape[-1]
    flat = cache_tensor.view(-1, heads, content_dim)
    if quant is None:
        flat[slots] = values.to(cache_tensor.dtype)
        return
    mode = quant.mode
    if mode == KVQuantMode.FP8_PER_TOKEN_HEAD:
        v = values.float()
        scale = tensorplay.clamp(v.abs().amax(dim=-1) / quant.quant_max, min=1e-6)
        # Cast through the native fp8 op: a plain ``to(fp8)`` is not supported
        # on every device (e.g. CUDA in this build).
        q = tensorplay.quantize_affine_float8(
            v, scale[..., None], cache_tensor.dtype
        )
        flat[slots, :, :head_dim] = q
        _write_inline_scale(cache_tensor, slots, scale, page_size)
        return
    if mode == KVQuantMode.INT4_PER_TOKEN_HEAD:
        # vLLM rotates before INT4 quantization to gaussianize the data.
        v = _rht_forward(values.float(), quant, head_dim)
        lo = v.amin(dim=-1)
        hi = v.amax(dim=-1)
        scale = tensorplay.clamp((hi - lo) / 15.0, min=1e-6)
        negative = -lo < 0
        zero_point = tensorplay.where(
            negative,
            tensorplay.floor(-lo / scale - 0.5),
            tensorplay.floor(-lo / scale + 0.5),
        )
        zero_point = tensorplay.clamp(zero_point, 0, 15).to(tensorplay.int32)
        codes = v / scale[..., None] + zero_point[..., None].to(tensorplay.float32)
        codes = tensorplay.where(
            codes >= 0,
            tensorplay.floor(codes + 0.5),
            tensorplay.floor(codes - 0.5),
        )
        codes = tensorplay.clamp(codes, 0, 15).to(tensorplay.int32)
        flat[slots, :, : head_dim // 2] = _pack_nibbles(codes)
        bits = scale.view(tensorplay.int32)
        packed_scale = ((bits & -16) | (zero_point & 0xF)).view(tensorplay.float32)
        _write_inline_scale(cache_tensor, slots, packed_scale, page_size)
        return
    if mode == KVQuantMode.NVFP4:
        v = values.float() / float(quant.nvfp4_global_scale)
        blocks = v.reshape(*v.shape[:-1], head_dim // 16, 16)
        block_scale = tensorplay.clamp(blocks.abs().amax(dim=-1) / 6.0, min=1e-6)
        scale_bytes = _encode_e4m3(block_scale)
        decoded = _decode_e4m3(scale_bytes).repeat_interleave(16, dim=-1)
        codes = _encode_e2m1(v / decoded)
        flat[slots, :, : head_dim // 2] = _pack_nibbles(codes)
        flat[slots, :, head_dim // 2 : head_dim // 2 + head_dim // 16] = scale_bytes
        return
    stored = quantize_tensor(values, quant, which)
    flat[slots, :, :head_dim] = stored


def rms_norm(
    x: tensorplay.Tensor,
    weight: tensorplay.Tensor,
    eps: float,
) -> tensorplay.Tensor:
    variance = x.float().pow(2).mean(dim=-1, keepdim=True)
    normalized = x.float() * tensorplay.rsqrt(variance + eps)
    return (normalized * weight.float()).to(dtype=x.dtype)


def fused_add_rms_norm(
    x: tensorplay.Tensor,
    residual: tensorplay.Tensor,
    weight: tensorplay.Tensor,
    eps: float,
) -> tuple[tensorplay.Tensor, tensorplay.Tensor]:
    updated_residual = residual + x
    return rms_norm(updated_residual, weight, eps), updated_residual


def offset_rms_norm(
    x: tensorplay.Tensor,
    weight: tensorplay.Tensor,
    eps: float,
) -> tensorplay.Tensor:
    variance = x.float().pow(2).mean(dim=-1, keepdim=True)
    normalized = x.float() * tensorplay.rsqrt(variance + eps)
    return (normalized * (1.0 + weight.float())).to(dtype=x.dtype)


def fused_add_offset_rms_norm(
    x: tensorplay.Tensor,
    residual: tensorplay.Tensor,
    weight: tensorplay.Tensor,
    eps: float,
) -> tuple[tensorplay.Tensor, tensorplay.Tensor]:
    updated_residual = residual + x
    return (
        offset_rms_norm(updated_residual, weight, eps),
        updated_residual,
    )


def silu_and_mul(gate_up: tensorplay.Tensor) -> tensorplay.Tensor:
    gate, up = gate_up.chunk(2, dim=-1)
    return F.silu(gate) * up


def moe_topk_softmax(
    router_logits: tensorplay.Tensor,
    top_k: int,
    *,
    normalize: bool,
    gating: str = "softmax",
) -> tuple[tensorplay.Tensor, tensorplay.Tensor]:
    """Reference routing: ``softmax/sigmoid -> topk -> optional renorm``."""

    logits = router_logits.float()
    if gating == "sigmoid":
        scores = tensorplay.sigmoid(logits)
        topk_weights, topk_ids = tensorplay.topk(scores, top_k, dim=-1)
        if normalize:
            topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)
    elif gating == "softmax":
        topk_weights, topk_ids = tensorplay.topk(logits, top_k, dim=-1)
        if normalize:
            topk_weights = tensorplay.softmax(topk_weights, dim=-1)
        else:
            full_probabilities = tensorplay.softmax(logits, dim=-1)
            topk_weights = tensorplay.gather(
                full_probabilities,
                -1,
                topk_ids,
            )
    else:
        raise ValueError("gating must be softmax or sigmoid")
    return topk_weights.float(), topk_ids.long()


def rope_and_cache_write(
    query: tensorplay.Tensor,
    key: tensorplay.Tensor,
    value: tensorplay.Tensor,
    positions: tensorplay.Tensor,
    slot_mapping: tensorplay.Tensor,
    cos_cache: tensorplay.Tensor,
    sin_cache: tensorplay.Tensor,
    key_cache: tensorplay.Tensor,
    value_cache: tensorplay.Tensor,
    active: tensorplay.Tensor | None = None,
) -> tensorplay.Tensor:
    if active is None:
        active = tensorplay.ones(
            query.shape[0],
            dtype=tensorplay.bool,
            device=query.device,
        )
    half = query.shape[-1] // 2
    cos = cos_cache.index_select(0, positions.long()).to(query.dtype)[:, None, :]
    sin = sin_cache.index_select(0, positions.long()).to(query.dtype)[:, None, :]

    query_first, query_second = query[..., :half], query[..., half:]
    key_first, key_second = key[..., :half], key[..., half:]
    query_rotated = tensorplay.cat(
        (
            query_first * cos - query_second * sin,
            query_second * cos + query_first * sin,
        ),
        dim=-1,
    )
    key_rotated = tensorplay.cat(
        (
            key_first * cos - key_second * sin,
            key_second * cos + key_first * sin,
        ),
        dim=-1,
    )
    active_slots = slot_mapping[active].long()
    quant = kv_quant_of(key_cache)
    page_size = key_cache.shape[1]
    quantize_cache_rows(
        key_rotated[active],
        key_cache,
        quant,
        "key",
        active_slots,
        page_size=page_size,
    )
    quantize_cache_rows(
        value[active],
        value_cache,
        quant,
        "value",
        active_slots,
        page_size=page_size,
    )
    return tensorplay.where(active[:, None, None], query_rotated, 0)


def paged_decode_attention(
    query: tensorplay.Tensor,
    key_cache: tensorplay.Tensor,
    value_cache: tensorplay.Tensor,
    block_tables: tensorplay.Tensor,
    sequence_lengths: tensorplay.Tensor,
    page_size: int,
) -> tensorplay.Tensor:
    batch_size, num_query_heads, head_dim = query.shape
    num_kv_heads = key_cache.shape[-2]
    query_per_kv = num_query_heads // num_kv_heads
    pages = int(block_tables.shape[1])
    width = pages * page_size
    offsets = tensorplay.arange(
        width,
        device=query.device,
    )
    valid = offsets[None, :] < sequence_lengths[:, None]
    slots = (
        block_tables.long().unsqueeze(-1) * page_size
        + tensorplay.arange(
            page_size,
            device=query.device,
        )[None, None, :]
    ).reshape(batch_size, width)
    safe_slots = tensorplay.where(valid, slots, 0)
    quant = kv_quant_of(key_cache)
    # Dequantize/unpack the whole cache first: index_select cannot handle fp8
    # or packed 4-bit rows, and the packed width differs from ``head_dim``.
    flat_keys = dequantize_cache_to_dtype(
        key_cache, quant, "key", query.dtype
    ).reshape(-1, num_kv_heads, head_dim)
    flat_values = dequantize_cache_to_dtype(
        value_cache, quant, "value", query.dtype
    ).reshape(-1, num_kv_heads, head_dim)
    keys = (
        flat_keys.index_select(0, safe_slots.reshape(-1))
        .reshape(batch_size, width, num_kv_heads, head_dim)
        .repeat_interleave(query_per_kv, dim=2)
    )
    values = (
        flat_values.index_select(0, safe_slots.reshape(-1))
        .reshape(batch_size, width, num_kv_heads, head_dim)
        .repeat_interleave(query_per_kv, dim=2)
    )
    scores = tensorplay.einsum(
        "bhd,blhd->bhl",
        query.float(),
        keys.float(),
    ).masked_fill(~valid[:, None, :], float("-inf"))
    probabilities = tensorplay.softmax(
        scores * (head_dim**-0.5),
        dim=-1,
    ).to(values.dtype)
    return tensorplay.einsum(
        "bhl,blhd->bhd",
        probabilities,
        values,
    )


def routed_moe_decode(
    hidden_states: tensorplay.Tensor,
    gate_up_weight: tensorplay.Tensor,
    down_weight: tensorplay.Tensor,
    topk_ids: tensorplay.Tensor,
    topk_weights: tensorplay.Tensor,
) -> tensorplay.Tensor:
    output = tensorplay.zeros_like(hidden_states)
    num_experts = gate_up_weight.shape[0]
    for expert_index in range(num_experts):
        token_indices, route_indices = tensorplay.where(topk_ids == expert_index)
        selected = hidden_states.index_select(0, token_indices)
        gate_up = F.linear(selected, gate_up_weight[expert_index])
        expert_output = F.linear(
            silu_and_mul(gate_up),
            down_weight[expert_index],
        )
        route_weights = topk_weights[token_indices, route_indices].to(
            expert_output.dtype
        )
        output.index_add_(
            0,
            token_indices,
            expert_output * route_weights[:, None],
        )
    return output


def greedy_argmax(logits: tensorplay.Tensor) -> tensorplay.Tensor:
    return tensorplay.argmax(logits, dim=-1)
