"""Fused int4 weight-only GEMMs for ``compressed-tensors`` AWQ checkpoints.

TensorPlay's native ``_weight_int4pack_mm`` dequantises the *entire* weight
matrix to the activation dtype before every GEMM, which is a 3.5 GB write plus
read per decode step for an 8B model.  These kernels keep the packed codes
resident, unpack the nibbles inside the tile, apply the checkpoint's
group scales in FP32, and feed the tile to tensor cores, so a decode step only
ever streams the int4 bytes.

Storage contract (identical to the checkpoint):

* ``packed``: ``[N, K // 2]`` uint8, two codes per byte, low nibble = even k;
* ``scale``: ``[N, K // group]`` activation-dtype group scales, symmetric
  around code 8, so ``w[n, k] = (code - 8) * scale[n, k // group]``.

The kernels are capture-safe: every tiling constant is a compile-time
specialisation and no host sync happens on the hot path.
"""

from __future__ import annotations

import tensorplay
import triton
import triton.language as tl

__all__ = [
    "int4_dequant_gemm",
    "int4_gemm_tuning",
    "reserve_split_counters",
]

# Module switch so tests can pin the two-pass split-K path when comparing bits.
SELF_REDUCE_ENABLED = True

# Dequantise the nibbles straight into bf16 instead of routing the scale
# multiply through fp32.  ``(code - 8)`` is exactly representable in bf16 and
# every group scale is a bf16 value, so ``code * scale`` needs at most 12
# mantissa bits: both forms produce the same real number and round it to bf16
# exactly once, but the bf16 form skips one convert per element (the fp32
# convert is 16% of the prefill GEMM).  Measured -2.5% at 512 rows and -7.8% at
# 2048 rows on the Triton path; neutral at decode, where the tile is
# bandwidth-bound; a bitwise test pins the equivalence.
DEQUANT_BF16 = True


@triton.jit
def _int4_store_tile(
    out_ptr,
    acc,
    m_offs,
    m_mask,
    n_out_offs,
    n_out_mask,
    N_OUT: tl.constexpr,
    BLOCK_M: tl.constexpr,
    OUT_N: tl.constexpr,
    FUSE_SILU_PAIRS: tl.constexpr,
):
    """Write one accumulator tile, optionally folding SwiGLU into the store.

    With ``FUSE_SILU_PAIRS`` the column block holds element-interleaved
    ``[gate_i, up_i]`` pairs from the repacked gate/up weight, so the tile owns
    both operands of the activation: half the columns are written back.
    """

    if FUSE_SILU_PAIRS:
        paired = tl.reshape(acc, (BLOCK_M, OUT_N, 2))
        gate, up = tl.split(paired)
        # The stand-alone path rounds the projection to the activation dtype on
        # store and reads it back, so replay that rounding before the activation.
        gate_f32 = gate.to(tl.bfloat16).to(tl.float32)
        activated = (
            gate_f32 * tl.sigmoid(gate_f32) * up.to(tl.bfloat16).to(tl.float32)
        )
        tl.store(
            out_ptr + m_offs[:, None] * N_OUT + n_out_offs[None, :],
            activated,
            mask=m_mask[:, None] & n_out_mask[None, :],
        )
    else:
        tl.store(
            out_ptr + m_offs[:, None] * N_OUT + n_out_offs[None, :],
            acc.to(out_ptr.dtype.element_ty),
            mask=m_mask[:, None] & n_out_mask[None, :],
        )


@triton.jit
def _int4_gemm_kernel(
    x_ptr,
    w_ptr,
    s_ptr,
    out_ptr,
    partial_ptr,
    counter_ptr,
    m_size,
    K: tl.constexpr,
    N: tl.constexpr,
    GROUP: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    K_PER_SLICE: tl.constexpr,
    SPLIT_K: tl.constexpr,
    NIBBLE_MAJOR: tl.constexpr,
    NUM_N_TILES: tl.constexpr,
    SELF_REDUCE: tl.constexpr,
    N_OUT: tl.constexpr,
    OUT_N: tl.constexpr,
    FUSE_SILU_PAIRS: tl.constexpr,
    DEQUANT_BF16: tl.constexpr,
    M_TILES: tl.constexpr,
):
    pid_n = tl.program_id(0)
    pid_m = tl.program_id(1)
    pid_k = tl.program_id(2)

    n_offs = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    n_mask = n_offs < N
    m_offs = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    m_mask = m_offs < m_size
    # Row of this M tile inside its own BLOCK_M block.  Partial rows are indexed
    # by (K slice, M tile, local row): with more than one M tile a global row
    # offset makes different (slice, tile) pairs collide.
    local_m = tl.arange(0, BLOCK_M)
    n_out_offs = pid_n * OUT_N + tl.arange(0, OUT_N)
    n_out_mask = n_out_offs < N_OUT

    acc = tl.zeros((BLOCK_M, BLOCK_N), tl.float32)
    k_end = tl.minimum((pid_k + 1) * K_PER_SLICE, K)
    k_start = pid_k * K_PER_SLICE
    NG: tl.constexpr = BLOCK_K // GROUP
    half: tl.constexpr = BLOCK_K // 2
    while k_start < k_end:
        x = tl.load(
            x_ptr + m_offs[:, None] * K + k_start + tl.arange(0, BLOCK_K)[None, :],
            mask=m_mask[:, None],
            other=0.0,
        )
        byte_tile = tl.load(
            w_ptr
            + n_offs[:, None] * (K // 2)
            + k_start // 2
            + tl.arange(0, half)[None, :],
            mask=n_mask[:, None],
            other=0,
        )
        low = (byte_tile & 0x0F).to(tl.uint8)
        high = ((byte_tile >> 4) & 0x0F).to(tl.uint8)
        if NIBBLE_MAJOR:
            codes = tl.interleave(low, high)
        else:
            codes = tl.join(low, high).reshape(BLOCK_N, BLOCK_K)
        scale = tl.load(
            s_ptr
            + n_offs[:, None] * (K // GROUP)
            + k_start // GROUP
            + tl.arange(0, NG)[None, :],
            mask=n_mask[:, None],
            other=0.0,
        )
        if DEQUANT_BF16:
            # (code - 8) is exact in bf16 and the scale is a bf16 value, so the
            # product needs at most 12 mantissa bits and rounds exactly once -
            # the same bf16 the fp32 form yields, without the fp32 round trip.
            # Multiply in the (n, group, k-in-group) shape so the scale stays a
            # broadcast instead of a materialised tile.
            weight = tl.reshape(
                tl.reshape(
                    codes.to(tl.bfloat16) - 8.0, (BLOCK_N, NG, GROUP)
                )
                * scale[:, :, None],
                (BLOCK_N, BLOCK_K),
            )
        else:
            scale_f32 = scale.to(tl.float32)
            weight = tl.reshape(
                tl.reshape(
                    codes.to(tl.float32) - 8.0, (BLOCK_N, NG, GROUP)
                )
                * scale_f32[:, :, None],
                (BLOCK_N, BLOCK_K),
            )
        acc = tl.dot(x, tl.trans(weight.to(x_ptr.dtype.element_ty)), acc)
        k_start += BLOCK_K

    if SPLIT_K > 1:
        tile_id = pid_m * NUM_N_TILES + pid_n
        tl.store(
            partial_ptr
            + ((pid_k * M_TILES + pid_m) * BLOCK_M + local_m)[:, None] * N
            + n_offs[None, :],
            acc,
            mask=m_mask[:, None] & n_mask[None, :],
        )
        if SELF_REDUCE:
            # The partials of one column tile stay hot in L2, so the last K slice
            # to arrive folds them itself instead of paying a second launch.
            # Summing in slice order keeps the result bit-identical to the
            # stand-alone reduce pass, and exactly one program resets the count.
            arrived = tl.atomic_add(
                counter_ptr + tile_id, 1, sem="acq_rel", scope="gpu"
            )
            if arrived == SPLIT_K - 1:
                total = tl.zeros((BLOCK_M, BLOCK_N), tl.float32)
                for split in tl.static_range(SPLIT_K):
                    total += tl.load(
                        partial_ptr
                        + ((split * M_TILES + pid_m) * BLOCK_M + local_m)[
                            :, None
                        ] * N
                        + n_offs[None, :],
                        mask=m_mask[:, None] & n_mask[None, :],
                        other=0.0,
                        volatile=True,
                    )
                _int4_store_tile(
                    out_ptr,
                    total,
                    m_offs,
                    m_mask,
                    n_out_offs,
                    n_out_mask,
                    N_OUT,
                    BLOCK_M,
                    OUT_N,
                    FUSE_SILU_PAIRS,
                )
                tl.store(counter_ptr + tile_id, 0)
    else:
        _int4_store_tile(
            out_ptr,
            acc,
            m_offs,
            m_mask,
            n_out_offs,
            n_out_mask,
            N_OUT,
            BLOCK_M,
            OUT_N,
            FUSE_SILU_PAIRS,
        )


@triton.jit
def _int4_reduce_kernel(
    partial_ptr,
    out_ptr,
    m_size,
    N: tl.constexpr,
    SPLIT_K: tl.constexpr,
    PAD_M: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    M_TILES: tl.constexpr,
):
    pid = tl.program_id(0)
    m_offs = pid * BLOCK_M + tl.arange(0, BLOCK_M)
    n_offs = tl.program_id(1) * BLOCK_N + tl.arange(0, BLOCK_N)
    mask = (m_offs[:, None] < m_size) & (n_offs[None, :] < N)
    local_m = tl.arange(0, BLOCK_M)
    acc = tl.zeros((BLOCK_M, BLOCK_N), tl.float32)
    for split in range(SPLIT_K):
        acc += tl.load(
            partial_ptr
            + ((split * M_TILES + pid) * PAD_M + local_m)[:, None] * N
            + n_offs[None, :],
            mask=mask,
            other=0.0,
        )
    tl.store(
        out_ptr + m_offs[:, None] * N + n_offs[None, :],
        acc.to(out_ptr.dtype.element_ty),
        mask=mask,
    )


def _split_k(tuning: dict, n_size: int, k_size: int) -> int:
    return max(1, min(tuning["split_k"], k_size // tuning["block_k"]))


# One zeroed atomic counter per output tile, reused by every self-reducing
# launch.  Capacity covers the widest decode projection (24576 / 32 = 768
# column tiles) with room for multi-row tiles; larger shapes fall back to the
# separate reduce pass instead of growing this buffer at run time.
_SPLIT_COUNTER_CAPACITY = 8192
_SPLIT_COUNTERS: dict[str, tensorplay.Tensor] = {}


def reserve_split_counters(device: object) -> tensorplay.Tensor:
    """Return the per-device counter buffer, creating it if needed.

    Callers that capture CUDA graphs should touch this during warm-up so the
    allocation cannot land inside a capture.
    """

    key = str(device)
    buffer = _SPLIT_COUNTERS.get(key)
    if buffer is None:
        buffer = tensorplay.zeros(
            (_SPLIT_COUNTER_CAPACITY,), dtype=tensorplay.int32, device=device
        )
        _SPLIT_COUNTERS[key] = buffer
    return buffer


def _pick_tuning(m: int, n: int, k: int) -> dict:
    """Static tile heuristic; CUDA graphs recompile once per bucket shape.

    Decode is DRAM bound on the int4 bytes, so the tiles are chosen to keep
    every SM streaming: narrow column blocks give more concurrent programs
    than wide ones, and one K slice per launch avoids partial-sum traffic.
    """

    if m <= 32:
        # Decode is DRAM bound on the int4 bytes.  Every projection has too few
        # column tiles to fill 82 SMs on its own (N / 32 is 128-768 programs),
        # so K slices supply the missing waves and a 256-wide K block keeps the
        # loop short.  The wide MLP output already has 9 waves and needs no
        # split; the long-K down projection needs the most.
        if k >= 8192:
            block_n, block_k, split_k, warps, stages = 32, 256, 4, 4, 2
        elif n <= 4608:
            block_n, block_k, split_k, warps, stages = 64, 256, 2, 8, 3
        else:
            block_n, block_k, split_k, warps, stages = 32, 256, 2, 4, 2
        return {
            "block_m": 16,
            "block_n": block_n,
            "block_k": block_k,
            "split_k": split_k,
            "num_warps": warps,
            "num_stages": stages,
        }
    if m <= 128:
        return {
            "block_m": 32,
            "block_n": 64,
            "block_k": 64,
            "split_k": 1,
            "num_warps": 4,
            "num_stages": 3,
        }
    # Prefill at a single short prompt: the row tile is wide enough to amortise
    # the unpack but too few M tiles exist to hide latency behind wide column
    # blocks, so each projection gets its own shape-dependent tiling.
    if m <= 768:
        if k >= 8192:
            block_m, block_n, block_k, warps, stages = 64, 64, 64, 4, 5
        elif n >= 8192:
            block_m, block_n, block_k, warps, stages = 128, 64, 32, 4, 5
        elif n <= 4608:
            block_m, block_n, block_k, warps, stages = 64, 32, 64, 4, 4
        else:
            block_m, block_n, block_k, warps, stages = 64, 128, 64, 4, 5
        return {
            "block_m": block_m,
            "block_n": block_n,
            "block_k": block_k,
            "split_k": 1,
            "num_warps": warps,
            "num_stages": stages,
        }
    # Long prefill: the unpack cost is amortised over BLOCK_M rows, so wide
    # row tiles keep the tensor cores fed.
    return {
        "block_m": 128 if m >= 256 else 64,
        "block_n": 64,
        "block_k": 64,
        "split_k": 1,
        "num_warps": 4,
        "num_stages": 2,
    }


def int4_gemm_tuning(m: int, n: int, k: int) -> dict:
    return _pick_tuning(m, n, k)


def int4_dequant_gemm(
    x: tensorplay.Tensor,
    packed: tensorplay.Tensor,
    scale: tensorplay.Tensor,
    group_size: int,
    *,
    tuning: dict | None = None,
    fuse_silu_pairs: bool = False,
) -> tensorplay.Tensor:
    """``x @ W.T`` with ``W`` resident as int4 ``packed`` codes.

    ``fuse_silu_pairs`` consumes a projection whose rows were repacked as
    element-interleaved ``[gate_i, up_i]`` pairs and returns
    ``silu(gate) * up`` with half the columns, folding the activation into the
    store.
    """

    if x.dim() != 2:
        raise ValueError("int4 GEMM expects 2-D activations")
    if packed.dim() != 2:
        raise ValueError("int4 GEMM expects 2-D packed weights")
    m, k = int(x.shape[0]), int(x.shape[1])
    n = int(packed.shape[0])
    if fuse_silu_pairs and n % 2:
        raise ValueError("paired silu fusion needs an even output width")
    if k != int(scale.shape[1]) * group_size:
        raise ValueError("int4 scales do not cover the activation width")
    if int(packed.shape[1]) * 2 != k:
        raise ValueError("packed int4 width must be K / 2")

    plan = dict(tuning) if tuning else _pick_tuning(m, n, k)
    block_m = plan["block_m"]
    if block_m < 16:
        block_m = 16
    plan["block_m"] = block_m
    splits = _split_k(plan, n, k)
    k_per_slice = triton.cdiv(k, splits)
    k_per_slice = triton.cdiv(k_per_slice, plan["block_k"]) * plan["block_k"]

    n_tiles = triton.cdiv(n, plan["block_n"])
    m_tiles = triton.cdiv(m, block_m)
    out_cols = n // 2 if fuse_silu_pairs else n
    out = tensorplay.empty(
        (m, out_cols), dtype=x.dtype, device=x.device
    )
    if splits > 1:
        # Partial rows are addressed as ``pid_k * BLOCK_M + m_offs`` with the
        # global row offset, so K slice 0 of the second M tile already starts at
        # row BLOCK_M: the buffer holds ``splits + m_tiles - 1`` blocks, not
        # ``splits``.  Short prompts (m = 17..32) take the decode plan, which
        # splits K with m_tiles = 2, and used to run off the end of this buffer.
        partial = tensorplay.empty(
            (splits * m_tiles * block_m, n),
            dtype=tensorplay.float32,
            device=x.device,
        )
    else:
        partial = out
    counters = reserve_split_counters(x.device)
    # The in-kernel fold needs one counter per output tile; when a caller asks
    # for a shape larger than the reserved slots, keep the separate reduce pass.
    self_reduce = (
        SELF_REDUCE_ENABLED
        and splits > 1
        and m_tiles * n_tiles <= int(counters.shape[0])
    )
    if fuse_silu_pairs and splits > 1 and not self_reduce:
        # The stand-alone reduce pass writes a full-width tile and cannot apply
        # the paired activation, so the fused path needs the in-kernel fold.
        raise ValueError("paired silu fusion needs the in-kernel split fold")

    grid = (n_tiles, m_tiles, splits)
    _int4_gemm_kernel[grid](
        x,
        packed,
        scale,
        out,
        partial,
        counters,
        m,
        K=k,
        N=n,
        GROUP=group_size,
        BLOCK_M=block_m,
        BLOCK_N=plan["block_n"],
        BLOCK_K=plan["block_k"],
        K_PER_SLICE=k_per_slice,
        SPLIT_K=splits,
        NIBBLE_MAJOR=True,
        NUM_N_TILES=n_tiles,
        SELF_REDUCE=self_reduce,
        N_OUT=out_cols,
        OUT_N=plan["block_n"] // 2 if fuse_silu_pairs else plan["block_n"],
        FUSE_SILU_PAIRS=fuse_silu_pairs,
        DEQUANT_BF16=(
            DEQUANT_BF16 and x.dtype == tensorplay.bfloat16
        ),
        M_TILES=m_tiles,
        num_warps=plan["num_warps"],
        num_stages=plan["num_stages"],
    )
    if splits > 1 and not self_reduce:
        _int4_reduce_kernel[
            (triton.cdiv(m, block_m), triton.cdiv(n, 256))
        ](
            partial,
            out,
            m,
            N=n,
            SPLIT_K=splits,
            PAD_M=block_m,
            BLOCK_M=block_m,
            BLOCK_N=256,
            M_TILES=m_tiles,
            num_warps=4,
        )
    return out

@triton.jit
def _int4_dequantize_kernel(
    w_ptr,
    s_ptr,
    out_ptr,
    K: tl.constexpr,
    N: tl.constexpr,
    GROUP: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_n = tl.program_id(0)
    pid_k = tl.program_id(1)
    n_offs = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    n_mask = n_offs < N
    k_offs = pid_k * BLOCK_K + tl.arange(0, BLOCK_K)
    NG: tl.constexpr = BLOCK_K // GROUP
    byte_tile = tl.load(
        w_ptr
        + n_offs[:, None] * (K // 2)
        + pid_k * (BLOCK_K // 2)
        + tl.arange(0, BLOCK_K // 2)[None, :],
        mask=n_mask[:, None],
        other=0,
    )
    low = (byte_tile & 0x0F).to(tl.uint8)
    high = ((byte_tile >> 4) & 0x0F).to(tl.uint8)
    weight = tl.interleave(low, high).to(tl.float32) - 8.0
    scale = tl.load(
        s_ptr
        + n_offs[:, None] * (K // GROUP)
        + (pid_k * NG + tl.arange(0, NG))[None, :],
        mask=n_mask[:, None],
        other=0.0,
    ).to(tl.float32)
    weight = tl.reshape(
        tl.reshape(weight, (BLOCK_N, NG, GROUP)) * scale[:, :, None],
        (BLOCK_N, BLOCK_K),
    )
    tl.store(
        out_ptr + n_offs[:, None] * K + k_offs[None, :],
        weight.to(out_ptr.dtype.element_ty),
        mask=n_mask[:, None],
    )


# One grow-only scratch buffer per device holds the dense weight while a
# prefill GEMM runs; each module re-materialises into it in turn.  Peak extra
# memory is the largest projection (200 MB for the MLP gate/up stack).
_DENSE_SCRATCH: dict[str, tensorplay.Tensor] = {}


def dense_weight_buffer(
    numel: int,
    device: tensorplay.device,
    dtype: tensorplay.dtype,
) -> tensorplay.Tensor:
    """Return a contiguous scratch vector of at least ``numel`` elements."""

    key = f"{device}|{dtype}"
    buffer = _DENSE_SCRATCH.get(key)
    if buffer is None or buffer.numel() < numel:
        buffer = tensorplay.empty((numel,), dtype=dtype, device=device)
        _DENSE_SCRATCH[key] = buffer
    return buffer[:numel]


def dequantize_int4_weight_into(
    packed: tensorplay.Tensor,
    scale: tensorplay.Tensor,
    group_size: int,
    out: tensorplay.Tensor,
) -> tensorplay.Tensor:
    """Unpack ``packed`` into the ``[N, K]`` dense buffer ``out``."""

    n = int(packed.shape[0])
    k = int(packed.shape[1]) * 2
    if int(out.numel()) < n * k:
        raise ValueError("dense weight buffer is too small")
    weight = out[: n * k].view(n, k)
    _int4_dequantize_kernel[(triton.cdiv(n, 64), k // 256)](
        packed,
        scale,
        weight,
        K=k,
        N=n,
        GROUP=group_size,
        BLOCK_N=64,
        BLOCK_K=256,
        num_warps=4,
    )
    return weight


def dequantize_int4_weight(
    packed: tensorplay.Tensor,
    scale: tensorplay.Tensor,
    group_size: int,
    dtype: tensorplay.dtype,
) -> tensorplay.Tensor:
    """Materialise the int4 weight as a dense ``[N, K]`` matrix once.

    Prefill is compute bound, so one linear re-materialisation pass followed by
    a plain tensor-core GEMM beats unpacking every weight tile once per
    prefill chunk.
    """

    n = int(packed.shape[0])
    k = int(packed.shape[1]) * 2
    if k % 256 or k % group_size:
        return _dequant_reference(packed, scale, group_size, dtype)
    out = tensorplay.empty((n, k), dtype=dtype, device=packed.device)
    return dequantize_int4_weight_into(packed, scale, group_size, out)


def _dequant_reference(
    packed: tensorplay.Tensor,
    scale: tensorplay.Tensor,
    group_size: int,
    dtype: tensorplay.dtype,
) -> tensorplay.Tensor:
    from megalm.models.components.int4_linear import unpack_packed_int4

    codes = unpack_packed_int4(packed).to(tensorplay.float32) - 8.0
    rows, width = int(codes.shape[0]), int(codes.shape[1])
    shaped = codes.reshape(rows, width // group_size, group_size)
    shaped = shaped * scale.to(tensorplay.float32)[:, :, None]
    return shaped.reshape(rows, width).to(dtype)
