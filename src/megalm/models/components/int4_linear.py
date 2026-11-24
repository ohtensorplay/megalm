"""Weight-only int4 projections for ``compressed-tensors`` checkpoints.

llm-compressor's ``pack-quantized`` W4A16 format stores an ``[N, K // 8]``
int32 tensor where each word holds eight 4-bit codes (low nibble first), an
``[N, K // group_size]`` scale tensor, and the pre-pack ``[N, K]`` shape.

On CUDA the checkpoint form is used verbatim: the int32 words are viewed as
``[N, K // 2]`` uint8 and handed to :mod:`megalm.kernels.int4_gemm`, which
unpacks the nibbles inside the tile and applies the checkpoint's own scales.
TensorPlay's native ``_weight_int4pack_mm`` instead dequantises the whole
weight matrix to bf16 on every call, which costs a 3.5 GB write plus read per
decode step for an 8B model, so it is only used as the non-CUDA fallback.
"""

from __future__ import annotations

import os

import tensorplay
import tensorplay.nn as nn
import tensorplay.nn.functional as F

from megalm.kernels.int4_gemm import (
    dequantize_int4_weight_into,
    dense_weight_buffer,
    int4_dequant_gemm,
    reserve_split_counters,
)
from megalm.kernels.quant_fp import int4_weight_mm

# Opt-in Marlin-class W4A16 GEMM (self-contained CUDA kernel in
# ``megalm.kernels.csrc``).  Off by default so the shared engine path is
# unchanged; set MEGALM_MARLIN=1 to route the plain int4 projections through it.
_USE_MARLIN = os.environ.get("MEGALM_MARLIN", "0") == "1"

# When Marlin owns a projection it serves decode *and* prefill, so the
# checkpoint-format copy is dead weight.  Dropping it keeps the resident
# footprint at one int4 layout per projection, matching vLLM.  Set
# MEGALM_MARLIN_KEEP_CHECKPOINT=1 to retain both for A/B debugging.
_KEEP_CHECKPOINT = os.environ.get("MEGALM_MARLIN_KEEP_CHECKPOINT", "0") == "1"

__all__ = [
    "Int4GroupLinear",
    "pack_compressed_int4",
    "unpack_packed_int4",
]

# Above this many activation rows the projection is compute bound, and a
# single dense re-materialisation plus a plain tensor-core GEMM is ~25 % faster
# than unpacking the weight inside every GEMM tile.
DENSE_PREFILL_MIN_ROWS = 1024


def unpack_packed_int4(packed: tensorplay.Tensor) -> tensorplay.Tensor:
    """Expand ``[N, K // 8]`` int32 words into ``[N, K]`` codes in ``[0, 15]``.

    ``compressed-tensors`` stores the signed int4 value plus eight, and packs
    consecutive inputs into increasing nibble positions.
    """

    if packed.dim() != 2:
        raise ValueError("packed int4 weights must be a 2-D word tensor")
    if packed.dtype == tensorplay.int32:
        words = packed.contiguous().view(tensorplay.uint8)
        rows = int(packed.shape[0])
        width = int(packed.shape[1]) * 8
    elif packed.dtype == tensorplay.uint8:
        words = packed.contiguous()
        rows = int(packed.shape[0])
        width = int(packed.shape[1]) * 2
    else:
        raise ValueError(
            f"packed int4 weights must be int32 or uint8, got {packed.dtype}"
        )
    low = words.bitwise_and(0x0F)
    high = words.bitwise_right_shift(4)
    interleaved = tensorplay.stack((low, high), dim=-1)
    return (
        interleaved.reshape(rows, width).to(tensorplay.int32).contiguous()
    )


def pack_compressed_int4(
    codes: tensorplay.Tensor,
    scale: tensorplay.Tensor,
    *,
    group_size: int,
    dtype: tensorplay.dtype,
) -> tuple[tensorplay.Tensor, tensorplay.Tensor]:
    """Pack ``[N, K]`` unsigned codes plus ``[N, K // group]`` scales.

    Returns the tensor-core B fragment and ``[K // group, N, 2]`` metadata
    whose second column is the symmetric zero point.
    """

    if codes.dim() != 2:
        raise ValueError("int4 codes must be a 2-D tensor")
    out_features, in_features = codes.shape
    if in_features % group_size:
        raise ValueError(
            "int4 input width must be divisible by the group size"
        )
    if tuple(scale.shape) != (out_features, in_features // group_size):
        raise ValueError(
            "int4 scales must cover one entry per output row and group"
        )
    if codes.device.type == "cuda":
        packed = tensorplay._C._convert_weight_to_int4pack(codes, 2)
    else:
        packed = tensorplay._C._convert_weight_to_int4pack_for_cpu(codes, 2)
    scale_typed = scale.to(device=codes.device, dtype=dtype).t().contiguous()
    scale_zero = tensorplay.stack(
        (scale_typed, tensorplay.zeros_like(scale_typed)),
        dim=-1,
    )
    return packed, scale_zero


class Int4GroupLinear(nn.Module):
    """A dense projection stored as int4 codes with group-wise scales.

    ``packed`` and ``scale`` stay ``None`` until :meth:`load_compressed`
    receives the checkpoint tensors, which keeps model construction free of
    transient full-width code buffers.  ``packed_tc``/``scale_zero`` are only
    populated for the non-CUDA fallback through TensorPlay's native int4 GEMM.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        *,
        group_size: int,
        device: tensorplay.device,
        dtype: tensorplay.dtype,
        pair_interleaved: bool = False,
    ) -> None:
        super().__init__()
        if in_features <= 0 or out_features <= 0:
            raise ValueError("int4 projection dimensions must be positive")
        if in_features % group_size:
            raise ValueError(
                "int4 input width must be divisible by the group size"
            )
        if pair_interleaved and out_features % 2:
            raise ValueError(
                "paired gate/up interleaving needs an even output width"
            )
        self.in_features = in_features
        self.out_features = out_features
        self.group_size = group_size
        self.pair_interleaved = pair_interleaved
        self.device = tensorplay.device(device)
        self.dtype = dtype
        self.register_buffer("packed", None)
        self.register_buffer("scale", None)
        self.register_buffer("packed_tc", None)
        self.register_buffer("scale_zero", None)
        self.register_buffer("marlin_q", None)
        self.register_buffer("marlin_s", None)
        self._loaded = False
        if self.device.type == "cuda":
            # Reserve the self-reducing GEMM counters now, so CUDA graph
            # capture never has to allocate them mid-capture.
            reserve_split_counters(self.device)

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    @property
    def uses_triton(self) -> bool:
        return self.packed_tc is None

    def load_compressed(
        self,
        packed: tensorplay.Tensor,
        scale: tensorplay.Tensor,
        *,
        in_features: int | None = None,
    ) -> None:
        """Install checkpoint ``weight_packed`` / ``weight_scale`` tensors."""

        if packed.dim() != 2 or scale.dim() != 2:
            raise ValueError("int4 packed weights and scales must be 2-D")
        rows = int(packed.shape[0])
        width = int(packed.shape[1]) * 8
        if rows != self.out_features:
            raise ValueError(
                f"int4 packed rows {rows} do not match {self.out_features}"
            )
        if width != self.in_features:
            raise ValueError(
                f"int4 packed width {width} does not match {self.in_features}"
            )
        if in_features is not None and in_features != width:
            raise ValueError(
                f"checkpoint declares width {in_features}, packed holds {width}"
            )
        if tuple(scale.shape) != (rows, width // self.group_size):
            raise ValueError(
                f"int4 scale shape {tuple(scale.shape)} does not match "
                f"({rows}, {width // self.group_size})"
            )
        # Canonical storage is the checkpoint's own packed form: two codes per
        # byte, low nibble = even k.  The CUDA path stops here and lets the
        # fused Triton GEMM unpack inside the tile.
        codes_u8 = packed.to(device=self.device).view(tensorplay.uint8)
        packed_u8 = codes_u8.reshape(rows, width // 2).contiguous()
        scale_dev = scale.to(device=self.device, dtype=self.dtype)
        if self.pair_interleaved:
            # Repack ``[gate_0..gate_H-1, up_0..up_H-1]`` into element pairs
            # ``[gate_0, up_0, gate_1, up_1, ...]`` so one column tile owns both
            # operands of the SwiGLU activation and the GEMM store can fold it.
            # Values and scales are untouched: this is a pure row permutation.
            half = rows // 2
            column = tensorplay.arange(half, device=self.device)
            pairs = tensorplay.stack(
                (column, column + half), dim=-1
            ).reshape(rows)
            packed_u8 = packed_u8.index_select(0, pairs).contiguous()
            scale_dev = scale_dev.index_select(0, pairs).contiguous()
        self.packed = packed_u8
        self.scale = scale_dev
        self._loaded = True
        if self.device.type == "cuda" and self.dtype in {
            tensorplay.bfloat16,
            tensorplay.float16,
        }:
            self.packed_tc = None
            self.scale_zero = None
            if _USE_MARLIN and self.dtype == tensorplay.bfloat16:
                from megalm.kernels import marlin_int4

                if marlin_int4.available():
                    self.marlin_q, self.marlin_s = marlin_int4.pack_marlin(
                        self.packed, self.scale, self.group_size
                    )
                    if not _KEEP_CHECKPOINT:
                        # Marlin covers every row count, so the checkpoint
                        # layout has no remaining reader.
                        self.packed = None
                        self.scale = None
            return
        # Non-CUDA fallback keeps TensorPlay's native int4 GEMM, which needs
        # the tensor-core B fragment plus the transposed scale/zero metadata.
        packed_fragment, scale_zero = pack_compressed_int4(
            unpack_packed_int4(self.packed),
            self.scale,
            group_size=self.group_size,
            dtype=self.dtype,
        )
        self.packed_tc = packed_fragment
        self.scale_zero = scale_zero

    @property
    def fuse_silu_pairs(self) -> bool:
        """Whether this projection can fold ``silu(gate) * up`` into its store."""

        return (
            self.pair_interleaved
            and self.packed_tc is None
            and self.packed is not None
            and self.scale is not None
        )

    def forward_silu_pairs(
        self, hidden_states: tensorplay.Tensor
    ) -> tensorplay.Tensor:
        """``gate_up`` then ``silu(gate) * up`` in one launch, half the columns."""

        if not self.fuse_silu_pairs:
            raise RuntimeError("paired silu fusion is not available here")
        return int4_dequant_gemm(
            hidden_states,
            self.packed,
            self.scale,
            self.group_size,
            fuse_silu_pairs=True,
        )

    def forward(self, hidden_states: tensorplay.Tensor) -> tensorplay.Tensor:
        if self.packed_tc is not None:
            return int4_weight_mm(
                hidden_states,
                self.packed_tc,
                self.scale_zero,
                self.group_size,
            )
        if (
            self.marlin_q is not None
            and hidden_states.dtype == tensorplay.bfloat16
            and hidden_states.dim() == 2
        ):
            from megalm.kernels import marlin_int4

            rows = int(hidden_states.shape[0])
            return marlin_int4.marlin_linear(
                hidden_states,
                self.marlin_q,
                self.marlin_s,
                self.group_size,
                prefill=rows > 8,
            )
        if self.packed is None or self.scale is None:
            raise RuntimeError("int4 projection has no loaded weight")
        rows = int(hidden_states.shape[0]) if hidden_states.dim() > 1 else 1
        if (
            rows >= DENSE_PREFILL_MIN_ROWS
            and hidden_states.dim() == 2
            and hidden_states.dtype in {
                tensorplay.bfloat16,
                tensorplay.float16,
            }
        ):
            # Compute-bound prefill: one linear unpack pass into scratch, then
            # the hardware GEMM.  Numerically this is the same dequantisation
            # the checkpoint defines, just hoisted out of the tile loop.
            weight = dequantize_int4_weight_into(
                self.packed,
                self.scale,
                self.group_size,
                dense_weight_buffer(
                    self.out_features * self.in_features,
                    self.packed.device,
                    self.dtype,
                ),
            )
            return F.linear(hidden_states, weight)
        return int4_dequant_gemm(
            hidden_states,
            self.packed,
            self.scale,
            self.group_size,
        )