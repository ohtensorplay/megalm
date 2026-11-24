"""Quantization schemes for the paged KV cache.

The schemes mirror the production ``--kv-cache-dtype`` choices of vLLM:

* ``FP8_PER_TENSOR`` — one static scalar per K/V tensor, ``q = fp8(x/scale)``
  on write and ``x ≈ q*scale`` on read (the vLLM uncalibrated default).
* ``FP8_PER_HEAD`` — a calibrated static scale per KV head, loaded from a
  checkpoint (llm-compressor ``attn_head`` strategy).
* ``FP8_PER_TOKEN_HEAD`` — a dynamic scale per (token, KV head) computed on
  write and stored inline in the cache block (vLLM's per-token-head mode).
* ``INT4_PER_TOKEN_HEAD`` — signed 4-bit KV with an inline per-token-head
  scale, two 4-bit values packed per byte.
* ``NVFP4`` — E2M1 4-bit KV with an inline E8M0 block scale per 16 elements.

The configuration is attached to cache tensors so attention kernels can read
it without threading a new argument through every model forward call.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum

import tensorplay


class KVQuantError(RuntimeError):
    """Raised when a quantized KV configuration is invalid."""


class KVQuantMode(str, Enum):
    """Scale granularity / packing scheme of a quantized KV arena."""

    FP8_PER_TENSOR = "fp8_per_tensor"
    FP8_PER_HEAD = "fp8_per_head"
    FP8_PER_TOKEN_HEAD = "fp8_per_token_head"
    INT4_PER_TOKEN_HEAD = "int4_per_token_head"
    NVFP4 = "nvfp4"

    @property
    def is_fp8(self) -> bool:
        return self in (
            KVQuantMode.FP8_PER_TENSOR,
            KVQuantMode.FP8_PER_HEAD,
            KVQuantMode.FP8_PER_TOKEN_HEAD,
        )

    @property
    def is_dynamic(self) -> bool:
        return self in (
            KVQuantMode.FP8_PER_TOKEN_HEAD,
            KVQuantMode.INT4_PER_TOKEN_HEAD,
            KVQuantMode.NVFP4,
        )

    @property
    def is_4bit(self) -> bool:
        return self in (
            KVQuantMode.INT4_PER_TOKEN_HEAD,
            KVQuantMode.NVFP4,
        )


# Triton constexpr codes for the scale-indexing mode used inside kernels.
SCALE_SCALAR = 0   # one Python float for the whole tensor
SCALE_PER_HEAD = 1  # fp32 scale tensor indexed by KV head
SCALE_PER_TOKEN_HEAD = 2  # fp32 scale embedded in the cache block


@dataclass(frozen=True, slots=True)
class KVQuantConfig:
    """Quantization for one paged KV arena.

    ``key_scale`` / ``value_scale`` are scalars for ``FP8_PER_TENSOR`` and
    sequences of ``num_kv_heads`` floats for ``FP8_PER_HEAD``.  Dynamic modes
    keep the fields at their defaults because the scale lives in the cache.
    """

    dtype: tensorplay.dtype
    key_scale: float | tuple[float, ...] = 1.0
    value_scale: float | tuple[float, ...] = 1.0
    mode: KVQuantMode = KVQuantMode.FP8_PER_TENSOR
    num_kv_heads: int = 0
    # --- Q quantization / native fp8 attention (vLLM QuantFP8) ---
    q_dtype: tensorplay.dtype | None = None
    q_scale: float | tuple[float, ...] = 1.0
    q_group: str = "per_tensor"  # "per_tensor" | "per_kv_head"
    native_fp8_dot: bool = False
    # --- 4-bit specifics ---
    nvfp4_global_scale: float = 1.0  # checkpoint global scale (divisor)
    int4_use_rht: bool = False  # randomized Hadamard rotation for INT4 accuracy

    def __post_init__(self) -> None:
        if not isinstance(self.mode, KVQuantMode):
            object.__setattr__(self, "mode", KVQuantMode(self.mode))
        if self.mode.is_fp8:
            if self.dtype not in (
                tensorplay.float8_e4m3fn,
                tensorplay.float8_e5m2,
            ):
                raise KVQuantError(
                    f"fp8 KV mode requires an fp8 dtype, got {self.dtype}"
                )
        elif self.mode == KVQuantMode.INT4_PER_TOKEN_HEAD:
            if self.dtype not in (tensorplay.uint8, tensorplay.int8):
                raise KVQuantError("int4 KV mode uses a packed uint8 arena")
        elif (
            self.mode == KVQuantMode.NVFP4
            and self.dtype is not tensorplay.uint8
        ):
            raise KVQuantError("nvfp4 KV mode uses a packed uint8 arena")
        self._validate_scale("key_scale", self.key_scale)
        self._validate_scale("value_scale", self.value_scale)
        self._validate_scale("q_scale", self.q_scale)
        if self.q_group not in ("per_tensor", "per_kv_head"):
            raise KVQuantError("q_group must be per_tensor or per_kv_head")
        if self.num_kv_heads < 0:
            raise KVQuantError("num_kv_heads cannot be negative")
        if self.mode == KVQuantMode.FP8_PER_HEAD:
            if self.num_kv_heads == 0:
                raise KVQuantError("per-head scale requires num_kv_heads")
            # A single scalar means "same scale for every KV head"; expand it
            # so the kernel-side head-scale vector is always well-formed.
            for name in ("key_scale", "value_scale"):
                value = getattr(self, name)
                if isinstance(value, (int, float)):
                    object.__setattr__(
                        self, name, (float(value),) * self.num_kv_heads
                    )
        for name in ("key_scale", "value_scale", "q_scale"):
            value = getattr(self, name)
            if isinstance(value, tuple) and len(value) != self.num_kv_heads:
                raise KVQuantError(
                    f"{name} must have num_kv_heads entries"
                )
        if self.native_fp8_dot:
            if not self.mode.is_fp8:
                raise KVQuantError("native fp8 dot requires an fp8 KV mode")
            if self.dtype != tensorplay.float8_e4m3fn:
                raise KVQuantError("native fp8 dot requires e4m3fn KV")
            if self.q_dtype not in (None, tensorplay.float8_e4m3fn):
                raise KVQuantError("native fp8 dot requires e4m3fn Q")
            if self.q_group == "per_kv_head" and self.num_kv_heads == 0:
                raise KVQuantError("per-head Q scale requires num_kv_heads")

    def _validate_scale(self, name: str, value: object) -> None:
        values = value if isinstance(value, tuple) else (value,)
        for entry in values:
            if not isinstance(entry, (int, float)) or not math.isfinite(entry) or entry <= 0:
                raise KVQuantError(f"{name} must be positive and finite")

    @property
    def enabled(self) -> bool:
        return True

    @property
    def scale_mode(self) -> int:
        """Triton scale-index code for K/V."""
        if self.mode in (
            KVQuantMode.FP8_PER_TOKEN_HEAD,
            KVQuantMode.INT4_PER_TOKEN_HEAD,
            KVQuantMode.NVFP4,
        ):
            return SCALE_PER_TOKEN_HEAD
        if self.mode == KVQuantMode.FP8_PER_HEAD:
            return SCALE_PER_HEAD
        return SCALE_SCALAR

    @property
    def triton_mode(self) -> int:
        """Numeric mode dispatch code shared with the Triton kernels."""
        return {
            KVQuantMode.FP8_PER_TENSOR: 1,
            KVQuantMode.FP8_PER_HEAD: 2,
            KVQuantMode.FP8_PER_TOKEN_HEAD: 3,
            KVQuantMode.INT4_PER_TOKEN_HEAD: 4,
            KVQuantMode.NVFP4: 5,
        }[self.mode]

    @property
    def triton_dtype(self) -> str:
        """Triton fp8 type name for the cache store/load cast."""
        if self.dtype == tensorplay.float8_e4m3fn:
            return "e4m3"
        if self.dtype == tensorplay.float8_e5m2:
            return "e5m2"
        return "u8"  # packed 4-bit modes

    @property
    def element_size(self) -> int:
        return int(tensorplay.empty((), dtype=self.dtype).element_size())

    @property
    def quant_max(self) -> float:
        """Largest finite representable magnitude for the cache dtype."""
        if self.mode == KVQuantMode.INT4_PER_TOKEN_HEAD:
            return 7.0
        if self.dtype == tensorplay.float8_e5m2:
            return 57344.0
        return 448.0  # e4m3 / nvfp4 block element range (scaled by block scale)

    @property
    def data_bytes_per_element(self) -> float:
        """Bytes consumed by one logical KV element (4-bit modes pack two)."""
        return self.element_size / (2.0 if self.mode.is_4bit else 1.0)

    @property
    def scale_bytes_per_token_head(self) -> int:
        """Inline fp32 scale bytes stored with each (token, KV head)."""
        if self.mode in (
            KVQuantMode.FP8_PER_TOKEN_HEAD,
            KVQuantMode.INT4_PER_TOKEN_HEAD,
        ):
            return 4
        return 0  # NVFP4 block scales are inline in the data region

    def data_dim(self, head_dim: int) -> int:
        """Packed data elements stored per (token, KV head)."""
        if self.mode == KVQuantMode.INT4_PER_TOKEN_HEAD:
            return head_dim // 2
        if self.mode == KVQuantMode.NVFP4:
            return head_dim // 2 + head_dim // 16  # packed E2M1 + E4M3 scales
        return head_dim

    def content_dim(self, head_dim: int) -> int:
        """Total elements per (token, KV head) in the raw arena.

        Dynamic fp8/int4 append a 4-byte fp32 scale after the data; NVFP4
        already carries its block scales inside ``data_dim``.
        """
        return self.data_dim(head_dim) + self.scale_bytes_per_token_head

    @property
    def has_inline_scale(self) -> bool:
        """Whether a per-token-head fp32 scale is appended to the data."""
        return self.scale_bytes_per_token_head > 0

    def scale_signature(self) -> str:
        """Stable string of all quantization settings for cache identity."""
        parts = [self.mode.value, str(self.dtype).rsplit(".", 1)[-1]]
        for name in ("key_scale", "value_scale", "q_scale"):
            value = getattr(self, name)
            if isinstance(value, tuple):
                parts.append(name + "=" + ",".join(format(v, ".9g") for v in value))
            else:
                parts.append(f"{name}={value:.9g}")
        parts.extend(
            (
                f"q_group={self.q_group}",
                f"q_dtype={None if self.q_dtype is None else str(self.q_dtype).rsplit('.', 1)[-1]}",
                f"native={self.native_fp8_dot}",
                f"nvfp4_global_scale={self.nvfp4_global_scale:.9g}",
                f"int4_use_rht={self.int4_use_rht}",
            )
        )
        return "|".join(parts)

    def describe(self) -> dict[str, object]:
        payload: dict[str, object] = {
            "mode": self.mode.value,
            "dtype": str(self.dtype).rsplit(".", 1)[-1],
        }
        for name in ("key_scale", "value_scale", "q_scale"):
            value = getattr(self, name)
            payload[name] = list(value) if isinstance(value, tuple) else value
        payload.update(
            {
                "q_group": self.q_group,
                "q_dtype": (
                    None
                    if self.q_dtype is None
                    else str(self.q_dtype).rsplit(".", 1)[-1]
                ),
                "native_fp8_dot": self.native_fp8_dot,
            }
        )
        return payload


_KV_QUANT_ATTR = "_megalm_kv_quant"


def attach_kv_quant(
    tensor: tensorplay.Tensor,
    quant: KVQuantConfig | None,
) -> tensorplay.Tensor:
    """Bind quantization metadata to a cache tensor or one-layer view."""
    setattr(tensor, _KV_QUANT_ATTR, quant)
    return tensor


def kv_quant_of(tensor: tensorplay.Tensor) -> KVQuantConfig | None:
    """Read the quantization metadata attached to a cache tensor."""
    return getattr(tensor, _KV_QUANT_ATTR, None)


_KV_SCALE_ATTR = "_megalm_kv_scale"


def attach_kv_scale(
    tensor: tensorplay.Tensor,
    scale: tensorplay.Tensor | None,
) -> tensorplay.Tensor:
    """Bind an inline fp32 scale view to a cache data tensor."""
    setattr(tensor, _KV_SCALE_ATTR, scale)
    return tensor


def kv_scale_of(tensor: tensorplay.Tensor) -> tensorplay.Tensor | None:
    """Read the inline fp32 scale view attached to a cache data tensor."""
    return getattr(tensor, _KV_SCALE_ATTR, None)


def scale_values(
    quant: KVQuantConfig,
    which: str,
) -> float | tuple[float, ...]:
    """Return the configured scale(s) for K or V (or Q)."""
    name = f"{which}_scale"
    return getattr(quant, name)


def _scale_tensor(
    quant: KVQuantConfig,
    which: str,
    device: tensorplay.device | str,
) -> tensorplay.Tensor:
    value = scale_values(quant, which)
    if isinstance(value, tuple):
        return tensorplay.tensor(value, dtype=tensorplay.float32, device=device)
    return tensorplay.tensor(float(value), dtype=tensorplay.float32, device=device)


def quantize_tensor(
    values: tensorplay.Tensor,
    quant: KVQuantConfig,
    which: str,
) -> tensorplay.Tensor:
    """Quantize K/V/Q via TensorPlay's native affine fp8 op.

    For ``FP8_PER_HEAD`` the scale is a ``(num_kv_heads,)`` vector and the
    values are reshaped so the scale broadcasts across each head's rows.
    """
    scale = _scale_tensor(quant, which, values.device)
    dtype = quant.q_dtype if which == "q" and quant.q_dtype else quant.dtype
    if scale.ndim == 0:
        return tensorplay.quantize_affine_float8(values, scale, dtype)
    # Per-head: the head axis is the second-to-last one (values are laid out
    # as [..., num_kv_heads, head_dim]), so reshape the scale to broadcast
    # along that axis only.
    shape = [1] * values.ndim
    shape[-2] = scale.numel()
    scale = scale.view(shape)
    return tensorplay.quantize_affine_float8(values, scale, dtype)


def dequantize_tensor(
    values: tensorplay.Tensor,
    quant: KVQuantConfig,
    which: str,
    output_dtype: tensorplay.dtype,
) -> tensorplay.Tensor:
    """Dequantize K/V via TensorPlay's native affine fp8 op."""
    scale = _scale_tensor(quant, which, values.device)
    if scale.ndim:
        shape = [1] * values.ndim
        shape[-2] = scale.numel()
        scale = scale.view(shape)
    return tensorplay.dequantize_affine_float8(
        values, scale, output_dtype
    )


def head_scale_tensor(
    quant: KVQuantConfig,
    which: str,
    device: tensorplay.device | str,
    fallback: tensorplay.Tensor,
) -> tensorplay.Tensor:
    """Return a device scale vector for per-head mode, or a dummy tensor."""
    if quant is None or quant.mode != KVQuantMode.FP8_PER_HEAD:
        return fallback
    return _scale_tensor(quant, which, device)


def dequant_kernel_kwargs(
    quant: KVQuantConfig | None,
    which: str,
) -> dict[str, object]:
    """Scalar constexpr/runtime kwargs for ``load_dequant_kv``."""
    if quant is None:
        return {
            "MODE": 0,
            "SCALAR_SCALE": 1.0,
            "GLOBAL_SCALE": 1.0,
            "FP8_DTYPE": "e4m3",
        }
    scale = scale_values(quant, which)
    return {
        "MODE": quant.triton_mode,
        "SCALAR_SCALE": float(scale) if isinstance(scale, (int, float)) else 1.0,
        "GLOBAL_SCALE": quant.nvfp4_global_scale,
        "FP8_DTYPE": quant.triton_dtype,
    }
