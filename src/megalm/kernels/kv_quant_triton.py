"""Triton helpers for quantized KV caches.

Covers the production vLLM schemes:

* FP8 static: ``q = fp8(x / scale)`` / ``x = q * scale``.
* FP8 per-token-head: ``scale = max(absmax / QMAX, 1e-6)``.
* INT4 per-token-head: asymmetric [0, 15] codes with the zero point steganographically
  stored in the low 4 bits of the fp32 scale (as vLLM does).
* NVFP4: E2M1 4-bit codes with one E4M3 block scale per 16 elements.

The fp8 variant is a compile-time constant so a kernel only specializes on the
dtype it actually uses (``fp8e4nv`` needs SM89+, ``fp8e5`` works on SM86+).
"""

from __future__ import annotations

import tensorplay
import triton
import triton.language as tl

# ---------------------------------------------------------------------------
# Randomized Hadamard transform (INT4 accuracy, mirrors vLLM's single_rht)
# ---------------------------------------------------------------------------

_RHT_CACHE: dict[tuple[int, str], tuple[tensorplay.Tensor, tensorplay.Tensor]] = {}


def get_rht_matrices(
    head_dim: int,
    device: tensorplay.device | str,
) -> tuple[tensorplay.Tensor, tensorplay.Tensor]:
    """Return ``(H, signs)`` for the randomized Hadamard transform.

    ``H`` is the unnormalized Sylvester matrix (``H @ H = d * I``) and ``signs``
    is a deterministic ``±1`` diagonal.  The forward transform is
    ``H @ (signs * x)``; the inverse is ``(H @ y) * signs / d``.  Both are
    cached per ``(head_dim, device)`` so no per-call allocation is needed.
    ``head_dim`` must be a power of two, matching vLLM's constraint.
    """
    if head_dim <= 0 or head_dim & (head_dim - 1):
        raise ValueError(f"RHT requires a power-of-two head_dim, got {head_dim}")
    key = (head_dim, str(device))
    cached = _RHT_CACHE.get(key)
    if cached is not None:
        return cached
    hadamard = tensorplay.ones((1, 1), dtype=tensorplay.float32, device=device)
    while hadamard.shape[0] < head_dim:
        hadamard = tensorplay.cat(
            [
                tensorplay.cat([hadamard, hadamard], dim=1),
                tensorplay.cat([hadamard, -hadamard], dim=1),
            ],
            dim=0,
        )
    generator = tensorplay.Generator(seed=0x9E3779B9)
    signs = (
        2.0
        * tensorplay.bernoulli(
            tensorplay.full((head_dim,), 0.5, device="cpu"),
            generator=generator,
        )
        - 1.0
    ).to(device)
    result = (hadamard.contiguous(), signs.contiguous())
    _RHT_CACHE[key] = result
    return result


def rht_enabled(quant) -> bool:
    """Whether INT4 Hadamard rotation is active for ``quant``."""
    from megalm.runtime.kv_quant import KVQuantMode

    return bool(
        quant is not None
        and quant.mode == KVQuantMode.INT4_PER_TOKEN_HEAD
        and quant.int4_use_rht
    )


def rht_query(query: tensorplay.Tensor, quant, head_dim: int) -> tensorplay.Tensor:
    """Rotate Q into the same RHT space as the INT4 K/V cache."""
    if not rht_enabled(quant):
        return query
    matrix, signs = get_rht_matrices(head_dim, query.device)
    flat = query.float().reshape(-1, head_dim)
    # vLLM forward RHT is H @ (signs ⊙ x); in row form that is
    # (x ⊙ signs) @ H (H is symmetric).
    rotated = (flat * signs) @ matrix
    return rotated.to(query.dtype).reshape(query.shape)


def rht_output(output: tensorplay.Tensor, quant, head_dim: int) -> tensorplay.Tensor:
    """Undo the RHT on an attention result (``single_rht(out, inverse=True)/d``)."""
    if not rht_enabled(quant):
        return output
    matrix, signs = get_rht_matrices(head_dim, output.device)
    flat = output.float().reshape(-1, head_dim)
    restored = ((flat @ matrix) * signs) / head_dim
    output.copy_(restored.to(output.dtype).reshape(output.shape))
    return output


@triton.jit
def rht_forward_1d(values, h_ptr, sign_ptr, D: tl.constexpr):
    """Forward RHT of a length-D fp32 tile: ``H @ (signs * values)``."""
    dims = tl.arange(0, D)
    matrix = tl.load(h_ptr + dims[:, None] * D + dims[None, :])
    signs = tl.load(sign_ptr + dims)
    return tl.sum(matrix * (values * signs)[None, :], axis=1)


@triton.jit
def rht_inverse_2d(values, h_ptr, sign_ptr, N: tl.constexpr, D: tl.constexpr):
    """Inverse RHT of an ``[N, D]`` fp32 tile: ``(H @ y) * signs / D``."""
    dims = tl.arange(0, D)
    matrix = tl.load(h_ptr + dims[:, None] * D + dims[None, :])
    signs = tl.load(sign_ptr + dims)
    rotated = tl.sum(matrix[None, :, :] * values[:, None, :], axis=2)
    return rotated * signs[None, :] / D


# ---------------------------------------------------------------------------
# FP8
# ---------------------------------------------------------------------------


@triton.jit
def quantize_fp8(value, scale, FP8_DTYPE: tl.constexpr):
    """Quantize a value to the configured fp8 cache dtype."""
    scaled = value.to(tl.float32) / scale
    if FP8_DTYPE == "e4m3":
        return scaled.to(tl.float8e4nv)
    return scaled.to(tl.float8e5)


@triton.jit
def dequantize_fp8(value, scale):
    """Dequantize an fp8 cache value back to fp32 with its static scale."""
    return value.to(tl.float32) * scale


@triton.jit
def quantize_fp8_dynamic(values, QUANT_MAX: tl.constexpr, FP8_DTYPE: tl.constexpr):
    """Return ``(q, scale)`` with a per-vector absmax scale.

    ``values`` is a 1-D fp32 tile covering one (token, head) row.  ``scale``
    mirrors vLLM's ``max(absmax / QMAX, 1e-6)``.
    """
    absmax = tl.max(tl.abs(values), axis=0)
    scale = tl.maximum(absmax / QUANT_MAX, 1e-6)
    scaled = values / scale
    scaled = tl.minimum(tl.maximum(scaled, -QUANT_MAX), QUANT_MAX)
    if FP8_DTYPE == "e4m3":
        return scaled.to(tl.float8e4nv), scale
    return scaled.to(tl.float8e5), scale


# ---------------------------------------------------------------------------
# INT4 (asymmetric [0, 15] with zero-point steganography)
# ---------------------------------------------------------------------------


@triton.jit
def pack_int4(lo, hi):
    """Pack two unsigned 4-bit codes into one byte (low nibble first)."""
    return ((lo.to(tl.int32) & 0xF) | ((hi.to(tl.int32) & 0xF) << 4)).to(tl.uint8)


@triton.jit
def unpack_int4(byte):
    """Return the low and high unsigned 4-bit codes of a byte as fp32."""
    byte = byte.to(tl.int32)
    return (byte & 0xF).to(tl.float32), ((byte >> 4) & 0xF).to(tl.float32)


@triton.jit
def stego_scale(scale, zero_point):
    """Hide a 4-bit zero point in the low mantissa bits of an fp32 scale."""
    bits = scale.to(tl.int32, bitcast=True)
    packed = (bits & -16) | (zero_point.to(tl.int32) & 0xF)
    return packed.to(tl.float32, bitcast=True)


@triton.jit
def unstego_scale(packed):
    """Return ``(scale, zero_point)`` from a steganographically packed scale."""
    bits = packed.to(tl.int32, bitcast=True)
    zero_point = (bits & 0xF).to(tl.float32)
    scale = (bits & -16).to(tl.float32, bitcast=True)
    return scale, zero_point


@triton.jit
def quantize_int4_dynamic(values):
    """Quantize a fp32 tile to packed INT4 with a stego-packed scale.

    ``values`` is a 1-D tile covering one (token, head) row (length head_dim).
    Returns ``(packed, packed_scale)`` where ``packed`` has head_dim/2 uint8
    elements and ``packed_scale`` is one fp32 value.
    """
    vmin = tl.min(values, axis=0)
    vmax = tl.max(values, axis=0)
    scale = tl.maximum((vmax - vmin) / 15.0, 1e-6)
    zero_point = tl.where(
        -vmin >= 0.0,
        tl.floor(-vmin / scale + 0.5),
        tl.floor(-vmin / scale - 0.5),
    )
    zero_point = tl.minimum(tl.maximum(zero_point, 0.0), 15.0)
    quantized = values / scale + zero_point
    quantized = tl.where(
        quantized >= 0.0,
        tl.floor(quantized + 0.5),
        tl.floor(quantized - 0.5),
    )
    quantized = tl.minimum(tl.maximum(quantized, 0.0), 15.0).to(tl.int32)
    # Even/odd interleave (low nibble first).  Triton has no strided slicing,
    # so reshape pairs and split along the trailing dimension of size two.
    pairs = tl.reshape(quantized, (quantized.shape[0] // 2, 2))
    even, odd = tl.split(pairs)
    packed = pack_int4(even, odd)
    return packed, stego_scale(scale, zero_point)


# ---------------------------------------------------------------------------
# NVFP4 (E2M1 codes, one E4M3 block scale per 16 elements)
# ---------------------------------------------------------------------------


@triton.jit
def decode_e2m1(code):
    """Decode an E2M1 nibble (0..15) to fp32.

    Values: 0, ±0.5, ±1, ±1.5, ±2, ±3, ±4, ±6 (code 8 is -0).
    """
    code = code.to(tl.int32)
    sign = tl.where((code & 8) != 0, -1.0, 1.0)
    magnitude = code & 7
    # exp = magnitude >> 1, mant = magnitude & 1
    exp = magnitude >> 1
    mant = magnitude & 1
    value = tl.where(
        exp == 0,
        0.5 * mant.to(tl.float32),
        (1.0 + 0.5 * mant.to(tl.float32)) * tl.exp2((exp - 1).to(tl.float32)),
    )
    return tl.where(magnitude == 0, 0.0, sign * value)


@triton.jit
def decode_e4m3(byte):
    """Decode a raw E4M3FN byte (as uint8) to fp32 (software bit decode).

    A software decoder is used so NVFP4 also works on SM86 where Triton
    cannot construct ``fp8e4nv`` values.
    """
    bits = byte.to(tl.int32)
    sign = tl.where((bits & 0x80) != 0, -1.0, 1.0)
    exponent = (bits >> 3) & 0xF
    mantissa = bits & 0x7
    subnormal = tl.exp2(-6.0) * (mantissa.to(tl.float32) / 8.0)
    normal = tl.exp2((exponent - 7).to(tl.float32)) * (
        1.0 + mantissa.to(tl.float32) / 8.0
    )
    value = tl.where(exponent == 0, subnormal, normal)
    value = tl.where(
        exponent == 15,
        tl.where(mantissa == 0, float("inf"), float("nan")),
        value,
    )
    return sign * value


@triton.jit
def encode_e2m1(value):
    """Encode fp32 values to the nearest E2M1 nibble (0..15).

    Magnitudes are 0, 0.5, 1, 1.5, 2, 3, 4, 6 with midpoints as thresholds.
    """
    magnitude = tl.abs(value)
    code = tl.where(
        magnitude <= 0.25,
        0,
        tl.where(
            magnitude <= 0.75,
            1,
            tl.where(
                magnitude <= 1.25,
                2,
                tl.where(
                    magnitude <= 1.75,
                    3,
                    tl.where(
                        magnitude <= 2.5,
                        4,
                        tl.where(
                            magnitude <= 3.5,
                            5,
                            tl.where(magnitude <= 5.0, 6, 7),
                        ),
                    ),
                ),
            ),
        ),
    )
    return tl.where(value < 0.0, code + 8, code).to(tl.int32)


@triton.jit
def encode_e4m3(value):
    """Software E4M3FN encoder for NVFP4 block scales (returns raw byte)."""
    abs_value = tl.abs(value)
    exponent = tl.floor(tl.log2(tl.maximum(abs_value, 1e-12))).to(tl.int32)
    exponent = tl.minimum(tl.maximum(exponent + 7, 1), 14)
    mantissa_float = abs_value / tl.exp2((exponent - 7).to(tl.float32))
    mantissa = tl.floor((mantissa_float - 1.0) * 8.0 + 0.5).to(tl.int32)
    mantissa = tl.minimum(tl.maximum(mantissa, 0), 7)
    bits = (exponent << 3) | mantissa
    bits = tl.where(value < 0.0, bits | 0x80, bits)
    return bits.to(tl.uint8)


@triton.jit
def dequant_int4(code, packed_scale):
    """Dequantize one INT4 code with a stego-packed scale."""
    scale, zero_point = unstego_scale(packed_scale)
    return (code - zero_point) * scale


@triton.jit
def dequant_nvfp4(code, scale_byte, global_scale):
    """Dequantize one E2M1 code with an E4M3 block scale and global scale."""
    return decode_e2m1(code) * decode_e4m3(scale_byte) * global_scale


@triton.jit
def inline_scale_offset(
    slots,
    kv_head,
    page_size: tl.constexpr,
    stride_page,
    stride_pos,
    stride_head,
):
    """Flat offset into a ``(pages, page_size, heads)`` scale view."""
    pages = slots // page_size
    positions = slots % page_size
    return pages * stride_page + positions * stride_pos + kv_head * stride_head


# Quantization modes shared by all attention kernels.
MODE_NONE = 0
MODE_FP8_SCALAR = 1
MODE_FP8_PER_HEAD = 2
MODE_FP8_PTH = 3
MODE_INT4_PTH = 4
MODE_NVFP4 = 5


@triton.jit
def quantize_and_store_kv(
    values,
    data_ptr,
    data_offsets,
    mask,
    scale_ptr,
    scale_offset,
    head_scale_ptr,
    kv_head,
    pack_scale_base,
    PAGE_SIZE: tl.constexpr,
    MODE: tl.constexpr,
    SCALAR_SCALE,
    GLOBAL_SCALE,
    FP8_DTYPE: tl.constexpr,
    QUANT_MAX: tl.constexpr,
    USE_RHT: tl.constexpr,
    D: tl.constexpr,
    rht_h_ptr,
    rht_sign_ptr,
):
    """Quantize a one-head fp32 tile and store it into a KV cache row.

    ``values`` is a 1-D tile of length ``head_dim`` and ``data_offsets`` are
    the precomputed byte/element addresses for that row.  ``scale_offset`` is
    the flat fp32 scale address for a ``(pages, page_size, heads)`` view.
    """
    if MODE == 0:
        tl.store(data_ptr + data_offsets, values, mask=mask)
        return
    if MODE == 1 or MODE == 2:
        if MODE == 1:
            scale = SCALAR_SCALE
        else:
            scale = tl.load(head_scale_ptr + kv_head)
        tl.store(
            data_ptr + data_offsets,
            quantize_fp8(values, scale, FP8_DTYPE),
            mask=mask,
        )
        return
    if MODE == 3:
        q, scale = quantize_fp8_dynamic(values, QUANT_MAX, FP8_DTYPE)
        tl.store(data_ptr + data_offsets, q, mask=mask)
        tl.store(scale_ptr + scale_offset, scale)
        return
    if MODE == 4:
        if USE_RHT:
            values = rht_forward_1d(values, rht_h_ptr, rht_sign_ptr, D)
        packed, packed_scale = quantize_int4_dynamic(values)
        tl.store(
            data_ptr + data_offsets,
            packed,
            mask=tl.arange(0, values.shape[0] // 2) < values.shape[0] // 2,
        )
        tl.store(scale_ptr + scale_offset, packed_scale)
        return
    if MODE == 5:
        # NVFP4: one E4M3 block scale per 16 elements, packed E2M1 codes.
        blocks = tl.reshape(values, (values.shape[0] // 16, 16))
        block_max = tl.max(tl.abs(blocks), axis=1)
        block_scale = tl.maximum(block_max / (6.0 * GLOBAL_SCALE), 1e-6)
        scaled = blocks / (block_scale[:, None] * GLOBAL_SCALE)
        codes = encode_e2m1(tl.reshape(scaled, (values.shape[0],)))
        pairs = tl.reshape(codes, (values.shape[0] // 2, 2))
        even, odd = tl.split(pairs)
        tl.store(data_ptr + data_offsets, pack_int4(even, odd))
        scale_index = tl.arange(0, values.shape[0] // 16)
        tl.store(
            data_ptr + pack_scale_base + scale_index,
            encode_e4m3(block_scale),
        )
        return


@triton.jit
def load_dequant_kv(
    data_ptr,
    scale_ptr,
    head_scale_ptr,
    slots,
    kv_head,
    dims,
    stride_slot,
    stride_head,
    scale_stride_page,
    scale_stride_pos,
    scale_stride_head,
    mask,
    PAGE_SIZE: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    MODE: tl.constexpr,
    SCALAR_SCALE,
    GLOBAL_SCALE,
    FP8_DTYPE: tl.constexpr,
):
    """Load a ``[tokens, head_dim]`` K/V tile and dequantize to fp32.

    ``slots`` is 1-D of length N and ``dims`` is 1-D of length D.  Data
    strides are expressed in elements of the arena dtype (one byte for all
    quantized modes).  Scale views are ``(pages, page_size, heads)`` fp32.
    """
    if MODE == 0:
        offsets = (
            slots[:, None] * stride_slot
            + kv_head * stride_head
            + dims[None, :]
        )
        return tl.load(data_ptr + offsets, mask=mask, other=0.0).to(tl.float32)

    if MODE == 1 or MODE == 2 or MODE == 3:
        offsets = (
            slots[:, None] * stride_slot
            + kv_head * stride_head
            + dims[None, :]
        )
        raw = tl.load(data_ptr + offsets, mask=mask, other=0.0)
        if MODE == 1:
            scale = SCALAR_SCALE
        elif MODE == 2:
            scale = tl.load(head_scale_ptr + kv_head)
        else:
            scale_offsets = inline_scale_offset(
                slots,
                kv_head,
                PAGE_SIZE,
                scale_stride_page,
                scale_stride_pos,
                scale_stride_head,
            )
            scale = tl.load(scale_ptr + scale_offsets)[:, None]
        return dequantize_fp8(raw, scale)

    if MODE == 4:
        packed_dims = dims // 2
        offsets = (
            slots[:, None] * stride_slot
            + kv_head * stride_head
            + packed_dims[None, :]
        )
        byte = tl.load(data_ptr + offsets, mask=mask, other=0)
        lo, hi = unpack_int4(byte)
        code = tl.where((dims[None, :] % 2) == 0, lo, hi)
        scale_offsets = inline_scale_offset(
            slots,
            kv_head,
            PAGE_SIZE,
            scale_stride_page,
            scale_stride_pos,
            scale_stride_head,
        )
        packed_scale = tl.load(scale_ptr + scale_offsets)[:, None]
        return dequant_int4(code, packed_scale)

    if MODE == 5:
        packed_dims = dims // 2
        offsets = (
            slots[:, None] * stride_slot
            + kv_head * stride_head
            + packed_dims[None, :]
        )
        byte = tl.load(data_ptr + offsets, mask=mask, other=0)
        lo, hi = unpack_int4(byte)
        code = tl.where((dims[None, :] % 2) == 0, lo, hi)
        scale_offsets = (
            slots[:, None] * stride_slot
            + kv_head * stride_head
            + (HEAD_DIM // 2 + dims // 16)[None, :]
        )
        scale_byte = tl.load(data_ptr + scale_offsets, mask=mask, other=0)
        return dequant_nvfp4(code, scale_byte, GLOBAL_SCALE)

    return tl.full((slots.shape[0], dims.shape[0]), 0.0, tl.float32)


