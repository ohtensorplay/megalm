"""NVFP4 / MXFP8 block codecs for Inkling-2 expert weights.

The checkpoint stores routed experts as packed E2M1 (NVFP4) or E4M3 (MXFP8)
values with per-16 / per-32 block scales.  Production inference only decodes;
:func:`encode_nvfp4` and :func:`roundtrip_mxfp8` exist so the codec can be
tested against the checkpoint's own arithmetic.

All ops are vectorised tensorplay.  The nibble order is ``low then high`` and
each 16-value block shares one scale, so a row of ``B`` bytes carries ``B``
scales of ``K // 16`` entries in total.
"""

from __future__ import annotations

import math

import tensorplay

Tensor = tensorplay.Tensor

_E2M1_GRID = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)
# Ascending thresholds; a magnitude at or above ``k`` midpoints rounds to
# level ``k`` of the grid.
_E2M1_MIDPOINTS = (0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0)
_FP32_MAX = 3.4028234663852886e38
_E4M3_MAX = 448.0
# ModelOpt NVFP4's outer scale: the per-16 E4M3 block scales reach the E4M3
# maximum 448 and the codes reach 6, so the per-expert outer factor is
# ``amax / (448 * 6) = amax / 2688`` (the checkpoint's
# ``*_outer_scale_denominator``).
_NVFP4_DENOMINATOR = _E4M3_MAX * 6.0


def e2m1_lut(device: tensorplay.device | str = "cpu") -> Tensor:
    """The 16 E2M1 codes, low nibble first (0..7 positive, 8..15 negative)."""

    values: list[float] = []
    for code in range(16):
        magnitude = _E2M1_GRID[code & 0x7]
        values.append(-magnitude if code & 0x8 else magnitude)
    return tensorplay.tensor(values, dtype=tensorplay.float32, device=device)


def _low_nibbles(flat: Tensor) -> Tensor:
    """Low nibble of an int64 vector without the uint8 ``and`` dtype quirk."""

    return flat - (flat.bitwise_right_shift(4) * 16)


def unpack_nvfp4(packed: Tensor, scales: Tensor, *, lut: Tensor | None = None) -> Tensor:
    """Decode packed E2M1 bytes ``[..., K // 2]`` with per-16 E4M3 scales.

    Args:
        packed: U8 tensor holding two nibbles per byte, low nibble first.
        scales: FP32 tensor ``[..., K // 16]`` of E4M3 block scales.
        lut: Optional cached :func:`e2m1_lut` result.

    Returns:
        FP32 ``[..., K]`` code-times-scale values.  The row's outer factor
        ``amax / denominator`` (ModelOpt's ``amax / 2688``) is applied by the
        caller, not here.
    """

    if packed.shape[-1] * 2 != scales.shape[-1] * 16:
        raise ValueError("NVFP4 scales must cover 16 values per entry")
    shape = tuple(packed.shape)
    table = e2m1_lut(packed.device) if lut is None else lut
    flat = packed.reshape(packed.numel())
    if flat.dtype != tensorplay.int32:
        flat = flat.to(tensorplay.int32)
    high_codes = flat.bitwise_right_shift(4)
    low = table.index_select(0, flat - high_codes * 16)
    high = table.index_select(0, high_codes)
    codes = tensorplay.stack((low, high), dim=1).reshape(
        *shape[:-1], shape[-1] * 2
    )
    # One scale covers 16 consecutive codes: broadcast instead of
    # materialising a repeat_interleave expansion (a full extra pass).
    blocks = codes.reshape(*shape[:-1], codes.shape[-1] // 16, 16)
    return (blocks * scales.float().unsqueeze(-1)).reshape(codes.shape)


def nvfp4_outer_scale(amax: float, denominator: float = _NVFP4_DENOMINATOR) -> float:
    """The per-expert ``amax / denominator`` factor that completes NVFP4 decoding."""

    return amax / denominator


def nvfp4_outer_reciprocal(amax: float, denominator: float) -> float:
    """Deprecated alias of :func:`nvfp4_outer_scale`."""

    return nvfp4_outer_scale(amax, denominator)


def _e4m3(value: Tensor) -> Tensor:
    """Round an FP32 tensor through FP8 E4M3, as the block scales are stored."""

    return value.to(tensorplay.float8_e4m3fn).float()


def _nearest_e2m1(scaled: Tensor) -> Tensor:
    """Round an FP32 tensor to signed E2M1 levels."""

    magnitude = tensorplay.abs(scaled)
    index = tensorplay.zeros_like(magnitude)
    for midpoint in _E2M1_MIDPOINTS:
        index = index + (magnitude >= midpoint).float()
    index = index.long()
    levels = tensorplay.tensor(
        list(_E2M1_GRID),
        dtype=tensorplay.float32,
        device=scaled.device,
    )
    grid = levels.index_select(0, index.reshape(index.numel())).reshape(
        scaled.shape
    )
    return tensorplay.where(scaled < 0, -grid, grid)


def encode_nvfp4(values: Tensor, amax: float) -> tuple[Tensor, Tensor, Tensor]:
    """Encode ``values`` (last axis a multiple of 16) as NVFP4.

    Returns ``(packed, scales, outer)``: U8 ``[..., K // 2]`` codes, FP32
    ``[..., K // 16]`` E4M3 scales, and the FP32 row factor
    ``amax / 2688`` that reconstructs the original magnitudes.
    """

    width = values.shape[-1]
    if width % 16:
        raise ValueError("NVFP4 rows must hold whole 16-value blocks")
    shape = tuple(values.shape)
    blocks = values.float().reshape(*shape[:-1], width // 16, 16)
    encoder = min(_NVFP4_DENOMINATOR / amax, _FP32_MAX)
    outer = amax * (1.0 / _NVFP4_DENOMINATOR)
    maximum = tensorplay.amax(tensorplay.abs(blocks), dim=-1, keepdim=True)

    def candidate(multiplier: float) -> tuple[Tensor, Tensor, Tensor]:
        scale = _e4m3(
            tensorplay.minimum(
                maximum * (encoder * multiplier),
                tensorplay.full_like(maximum, _E4M3_MAX),
            )
        )
        inverse = tensorplay.minimum(
            1.0 / (scale * (1.0 / encoder)),
            tensorplay.full_like(scale, _FP32_MAX),
        )
        codes = _nearest_e2m1(blocks * inverse)
        restored = codes * scale * outer
        error = tensorplay.mean(tensorplay.abs(restored - blocks), dim=-1)
        return scale, codes, error

    scale6, code6, error6 = candidate(1.0 / 6.0)
    scale4, code4, error4 = candidate(0.25)
    choose4 = error4 < error6
    scale = tensorplay.where(choose4, scale4, scale6)
    code = tensorplay.where(choose4, code4, code6)
    nibbles = tensorplay.where(code < 0, -code, code)
    # Rebuild the signed nibble index: grid level + 8 for negatives.
    index = tensorplay.zeros_like(nibbles)
    for level, grid_value in enumerate(_E2M1_GRID):
        index = tensorplay.where(
            tensorplay.abs(nibbles - grid_value) < 1e-6,
            tensorplay.full_like(index, float(level)),
            index,
        )
    index = index + tensorplay.where(code < 0, 8.0, 0.0)
    index = index.long().reshape(*shape[:-1], width)
    low = index[..., 0::2].bitwise_and(0x0F).to(tensorplay.uint8)
    high = tensorplay.bitwise_left_shift(index[..., 1::2], 4).to(tensorplay.uint8)
    packed = tensorplay.bitwise_or(low, high)
    return (
        packed,
        scale.reshape(*shape[:-1], width // 16),
        tensorplay.full(
            shape[:-1],
            outer,
            dtype=tensorplay.float32,
            device=values.device,
        ),
    )


def roundtrip_nvfp4(values: Tensor, amax: float, denominator: float = 1.0) -> Tensor:
    """Encode then decode ``values``; the result differs by E2M1 rounding."""

    packed, scales, outer = encode_nvfp4(values, amax)
    factor = float(outer.reshape(-1)[0].item()) / denominator
    decoded = unpack_nvfp4(packed, scales)
    return tensorplay.tensor(
        [value * factor for value in decoded.reshape(decoded.numel()).tolist()],
        dtype=decoded.dtype,
        device=decoded.device,
    ).reshape(decoded.shape)


def mxfp8_scale(exponents: Tensor) -> Tensor:
    """UE8M0 scale bytes as FP32: ``2 ** (byte - 127)``, 255 decoding to NaN."""

    raw = exponents.float()
    return tensorplay.where(
        exponents.long() == 255,
        tensorplay.full_like(raw, float("nan")),
        tensorplay.pow(2.0, raw - 127.0),
    )


def unpack_mxfp8(values: Tensor, scales: Tensor) -> Tensor:
    """Decode E4M3 values ``[..., K]`` with per-32 UE8M0 scales ``[..., K // 32]``."""

    if values.shape[-1] != scales.shape[-1] * 32:
        raise ValueError("MXFP8 scales must cover 32 values per entry")
    shape = tuple(values.shape)
    blocks = values.float().reshape(*shape[:-1], shape[-1] // 32, 32)
    return (blocks * mxfp8_scale(scales).float().unsqueeze(-1)).reshape(shape)


def roundtrip_mxfp8(values: Tensor) -> Tensor:
    """Encode then decode ``values`` per 32 values with UE8M0 scales.

    Lossy: the per-32 UE8M0 exponent is a power of two.
    """

    width = values.shape[-1]
    if width % 32:
        raise ValueError("MXFP8 rows must hold whole 32-value blocks")
    shape = tuple(values.shape)
    blocks = values.float().reshape(*shape[:-1], width // 32, 32)
    maximum = tensorplay.amax(tensorplay.abs(blocks), dim=-1, keepdim=True)
    floor = tensorplay.full_like(maximum, 2.0**-127)
    exponent = tensorplay.ceil(
        tensorplay.log2(tensorplay.maximum(maximum, floor))
    )
    exponent = tensorplay.minimum(
        tensorplay.maximum(exponent, tensorplay.full_like(exponent, -127.0)),
        tensorplay.full_like(exponent, 254.0),
    )
    scale = tensorplay.pow(2.0, exponent)
    codes = (blocks / scale).to(tensorplay.float8_e4m3fn)
    bytes_ = (exponent + 127.0).to(tensorplay.uint8)
    return unpack_mxfp8(
        codes.reshape(*shape[:-1], width),
        bytes_.reshape(*shape[:-1], width // 32),
    )


def pack_int4_weight(
    weight: Tensor, group_size: int = 128
) -> tuple[Tensor, Tensor]:
    """Quantise a ``[N, K]`` weight into tensorplay's int4 weight-only form.

    Returns the packed tensor-core B fragments plus the ``[K/groups, N, 2]``
    scale-and-zero metadata consumed by :func:`int4_weight_mm`.
    """

    from tensorplay.ao.quantization.int4_weight_only import _pack_int4_weight

    return _pack_int4_weight(weight, group_size)


def int4_weight_mm(
    x: Tensor,
    packed: Tensor,
    scale_zero: Tensor,
    group_size: int = 128,
) -> Tensor:
    """``x @ W.T`` with the int4 group dequantisation fused into the GEMM.

    This is tensorplay's native weight-only int4 path: the packed codes are
    dequantised to the activation dtype inside the kernel, so no decoded copy
    of the expert ever materialises in Python.
    """

    if x.device.type == "cuda":
        return tensorplay._C._weight_int4pack_mm(x, packed, group_size, scale_zero)
    return tensorplay._C._weight_int4pack_mm_for_cpu(
        x, packed, group_size, scale_zero
    )


__all__ = [
    "encode_nvfp4",
    "e2m1_lut",
    "int4_weight_mm",
    "mxfp8_scale",
    "nvfp4_outer_reciprocal",
    "nvfp4_outer_scale",
    "pack_int4_weight",
    "roundtrip_mxfp8",
    "roundtrip_nvfp4",
    "unpack_mxfp8",
    "unpack_nvfp4",
]