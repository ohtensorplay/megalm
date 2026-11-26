"""Marlin-class W4A16 int4 GEMM for MegaLM (decode + prefill).

The CUDA source in ``csrc/marlin_int4.cu`` is a self-contained lift of vLLM's
Apache-2.0 Marlin kernel (headers vendored under ``csrc/marlin/``).  It is
compiled once with ``nvcc`` into a cached shared object and driven through a
plain C ABI, so nothing here depends on the TensorPlay native library or on
vLLM's torch extension.

Weights are consumed in Marlin's own packed layout.  :func:`pack_marlin`
converts MegaLM's checkpoint form (``[N, K//2]`` uint8, low nibble = even k,
plus ``[N, K//group]`` bf16 scales) into that layout at load time.  The
dequantisation contract is identical to the checkpoint (``(code - 8) * scale``,
single bf16 rounding), so outputs match the Triton path up to float
accumulation order.
"""

from __future__ import annotations

import ctypes
import hashlib
import os
import subprocess
import tempfile
from pathlib import Path

import numpy as np

import tensorplay

_HERE = Path(__file__).resolve().parent
_CSRC = _HERE / "csrc"
_SRC = _CSRC / "marlin_int4.cu"
_LIB = None


# --------------------------------------------------------------------------
# build / load
# --------------------------------------------------------------------------
def _nvcc() -> str:
    for cand in ("nvcc", "/usr/local/cuda/bin/nvcc"):
        try:
            subprocess.run([cand, "--version"], capture_output=True, check=True)
            return cand
        except (OSError, subprocess.CalledProcessError):
            continue
    raise RuntimeError("nvcc not found; marlin int4 GEMM needs CUDA toolkit")


def _build() -> str:
    digest = hashlib.sha1(_SRC.read_bytes()).hexdigest()[:16]
    for hdr in _CSRC.joinpath("marlin").rglob("*"):
        if hdr.is_file():
            digest += hashlib.sha1(hdr.read_bytes()).hexdigest()[:8]
    cache = Path(tempfile.gettempdir()) / "megalm_marlin_int4"
    cache.mkdir(exist_ok=True)
    so = cache / f"marlin_int4_{digest}.so"
    if not so.exists():
        tmp = so.with_suffix(".so.tmp")
        cmd = [
            _nvcc(), "-arch=sm_86", "-O3", "-std=c++20",
            "--expt-relaxed-constexpr", "-shared", "-Xcompiler", "-fPIC",
            "-I", str(_CSRC), "-o", str(tmp), str(_SRC),
        ]
        subprocess.run(cmd, check=True, capture_output=True)
        os.replace(tmp, so)
    return str(so)


def _lib():
    global _LIB
    if _LIB is None:
        _LIB = ctypes.CDLL(_build())
        _LIB.marlin_decode.restype = ctypes.c_int
        _LIB.marlin_prefill.restype = ctypes.c_int
        common = [ctypes.c_void_p] * 6 + [ctypes.c_int] * 5 + [
            ctypes.c_int, ctypes.c_int, ctypes.c_void_p]
        _LIB.marlin_decode.argtypes = common
        _LIB.marlin_prefill.argtypes = common
    return _LIB


_STREAM = None
_LOCKS = {}
_C_TMP = {}


def _c_tmp(device) -> tensorplay.Tensor:
    """FP32 split-K reduction workspace, as in vLLM's ``marlin_gemm``.

    Without it the kernel reduces directly in the output buffer using fp16
    bit patterns, which corrupts bf16 outputs whenever more than one block
    shares a K slice.  vLLM defaults this on for exactly that reason.
    """

    buf = _C_TMP.get(device)
    if buf is None:
        # Worst case: one reduction slot per launched block, 64 rows and
        # 128 output columns per tile (vLLM's ``c_size`` formula).
        buf = tensorplay.empty(
            (2048 * 64 * 128,), dtype=tensorplay.float32, device=device
        )
        _C_TMP[device] = buf
    return buf


def _stream() -> ctypes.c_void_p:
    # TensorPlay's current stream, which is the capture stream during graph
    # capture (tensorplay.cuda.graph(stream=...) pins it).  Launching on it
    # records the kernel into the graph.
    _STREAM = ctypes.c_void_p(tensorplay.cuda.current_stream().cuda_stream)
    return _STREAM


def _locks(device) -> tensorplay.Tensor:
    buf = _LOCKS.get(device)
    if buf is None:
        buf = tensorplay.zeros((1024,), dtype=tensorplay.int32, device=device)
        _LOCKS[device] = buf
    return buf


def available() -> bool:
    if tensorplay is None:
        return False
    try:
        _lib()
        return True
    except Exception:
        return False


# --------------------------------------------------------------------------
# weight packing  (engine form -> marlin form)
# --------------------------------------------------------------------------
def _get_weight_perm() -> np.ndarray:
    perm_list = []
    for i in range(32):
        col = i // 4
        perm1 = []
        for block in [0, 1]:
            for row in [2*(i % 4), 2*(i % 4)+1, 2*(i % 4+4), 2*(i % 4+4)+1]:
                perm1.append(16 * row + col + 8 * block)
        for j in range(4):
            perm_list.extend([p + 256 * j for p in perm1])
    interleave = np.array([0, 2, 4, 6, 1, 3, 5, 7])
    return np.array(perm_list).reshape((-1, 8))[:, interleave].ravel()


def _scale_perm() -> np.ndarray:
    return np.array([i + 8 * j for i in range(8) for j in range(8)])


def pack_marlin(packed: tensorplay.Tensor, scale: tensorplay.Tensor,
                group_size: int) -> tuple[tensorplay.Tensor, tensorplay.Tensor]:
    """``[N, K//2]`` uint8 + ``[N, K//g]`` bf16  ->  marlin ``Q`` (int32) + ``S`` (bf16).

    ``Q`` has shape ``[K//16, N*2]``; ``S`` has shape ``[K//g, N]``.
    """

    n = int(packed.shape[0])
    k = int(packed.shape[1]) * 2
    p_np = packed.detach().to("cpu").contiguous().view(tensorplay.uint8).numpy()
    s_np = (scale.detach().to("cpu").float().numpy())
    lo = (p_np & 0x0F).astype(np.int32)
    hi = (p_np >> 4).astype(np.int32)
    codes = np.empty((n, k), dtype=np.int32)
    codes[:, 0::2] = lo
    codes[:, 1::2] = hi
    q_w = codes.T.astype(np.int32)  # [K, N]

    tile = 16
    q_w = q_w.reshape((k // tile, tile, n // tile, tile))
    q_w = q_w.transpose((0, 2, 1, 3)).reshape((k // tile, n * tile))
    perm = _get_weight_perm()
    q_w = q_w.reshape((-1, perm.size))[:, perm].reshape(q_w.shape).astype(np.uint32)
    packed_q = np.zeros((q_w.shape[0], q_w.shape[1] // 8), dtype=np.uint32)
    for i in range(8):
        packed_q |= q_w[:, i::8] << (4 * i)

    sp = _scale_perm()
    s2 = s_np.T.copy()  # [K//g, N]
    s2 = s2.reshape((-1, sp.size))[:, sp].reshape((-1, n))

    dev = packed.device
    Q = tensorplay.from_numpy(packed_q.astype(np.int32)).to(dev)
    S = tensorplay.from_numpy(np.ascontiguousarray(s2)).to(
        tensorplay.bfloat16).to(dev)
    return Q, S


_CODE_INDEX_CACHE: dict[tuple[int, int], np.ndarray] = {}
_SCALE_INDEX_CACHE: dict[tuple[int, int, int], np.ndarray] = {}


def _code_gather_index(k: int, n: int) -> np.ndarray:
    """Flat source indices for the permuted ``[K/16, N*16]`` code layout."""

    key = (k, n)
    cached = _CODE_INDEX_CACHE.get(key)
    if cached is not None:
        return cached
    perm = _get_weight_perm()
    # Indices into a [K, N] row-major matrix; convert afterwards to the
    # checkpoint's [N, K] row-major layout.
    flat = np.arange(k * n, dtype=np.int32).reshape(
        k // 16, 16, n // 16, 16
    )
    flat = flat.transpose((0, 2, 1, 3)).reshape(k // 16, n * 16)
    flat = flat.reshape(-1, perm.size)[:, perm].reshape(
        k // 16, n * 16
    ).ravel()
    indices = ((flat % n) * k + flat // n).astype(np.int32)
    _CODE_INDEX_CACHE[key] = indices
    return indices


def _scale_gather_index(k: int, n: int, group_size: int) -> np.ndarray:
    """Flat source indices for the permuted ``[K/g, N]`` scale layout."""

    key = (k, n, group_size)
    cached = _SCALE_INDEX_CACHE.get(key)
    if cached is not None:
        return cached
    groups = k // group_size
    sp = _scale_perm()
    flat = np.arange(groups * n, dtype=np.int32).reshape(groups, n)
    flat = flat.reshape(-1, sp.size)[:, sp].reshape(groups, n).ravel()
    indices = ((flat % n) * groups + flat // n).astype(np.int32)
    _SCALE_INDEX_CACHE[key] = indices
    return indices


def pack_marlin_batch(
    packed: tensorplay.Tensor,
    scale: tensorplay.Tensor,
    group_size: int,
) -> tuple[tensorplay.Tensor, tensorplay.Tensor]:
    """Stacked-expert variant of :func:`pack_marlin`.

    ``packed`` is ``[E, N, K//2]`` uint8 and ``scale`` is ``[E, N, K//g]``
    bf16.  The permutation is identical for every expert, so the whole stack
    is reshaped and permuted in one vectorised NumPy pass instead of running
    the per-expert Python packer E times.  The returned ``Q``/``S`` keep the
    expert as the leading dimension and are bitwise identical to stacking
    ``pack_marlin`` outputs.
    """

    experts, n, half_k = (int(value) for value in packed.shape)
    k = half_k * 2
    group_count = int(scale.shape[2])
    if int(scale.shape[0]) != experts or int(scale.shape[1]) != n:
        raise ValueError("packed and scale expert dimensions do not match")
    if k % 16 or n % 16 or k % group_size:
        raise ValueError("Marlin packing needs K%16 == N%16 == K%g == 0")

    p_np = packed.detach().to("cpu").contiguous().view(tensorplay.uint8).numpy()
    s_np = scale.detach().to("cpu").float().numpy()

    # Expand the two nibbles into one uint8 code per K element.  The source
    # stays in checkpoint order [E, N, K]; both permutations below are applied
    # by a single gather with a cached index, so no intermediate [E, K, N]
    # copy is ever materialised.
    codes = np.empty((experts, n, k), dtype=np.uint8)
    codes[:, :, 0::2] = p_np & 0x0F
    codes[:, :, 1::2] = p_np >> 4
    q_gathered = codes.reshape(experts, n * k)[
        :, _code_gather_index(k, n)
    ].reshape(experts, k // 16, n * 2, 8)
    # Two codes per byte, little-endian: code i occupies bits 4*i .. 4*i+3.
    pairs = q_gathered[:, :, :, 0::2] | (
        q_gathered[:, :, :, 1::2] << np.uint8(4)
    )
    packed_q = np.ascontiguousarray(pairs).view(np.uint32).reshape(
        (experts, k // 16, n * 2)
    )

    s2 = s_np.reshape(experts, n * group_count)[
        :, _scale_gather_index(k, n, group_size)
    ].reshape(experts, group_count, n)

    dev = packed.device
    Q = tensorplay.from_numpy(packed_q.astype(np.int32)).to(dev)
    S = tensorplay.from_numpy(np.ascontiguousarray(s2)).to(
        tensorplay.bfloat16
    ).to(dev)
    return Q, S



# --------------------------------------------------------------------------
# launches
# --------------------------------------------------------------------------
def _max_smem() -> int:
    return 101376


_SMS = None


def _num_sms() -> int:
    global _SMS
    if _SMS is None:
        _SMS = int(tensorplay.cuda.get_device_properties(
            tensorplay.cuda.current_device()).multi_processor_count)
    return _SMS


def _grid_for(m: int, n: int, k: int, prefill: bool) -> int:
    """Smallest power-of-two grid covering every output tile.

    Marlin's split-K part2 path misreduces when there are more tiles than
    blocks, so the grid must cover ``m_tiles * n_tiles * k_tiles``.  A fixed
    2048-block grid was correct but launched ~4x more blocks than most expert
    GEMMs need.
    """

    override = os.environ.get("MEGALM_MARLIN_GRID")
    if override is not None:
        return int(override)
    row_tile = 16 if prefill else 8
    tiles = (
        ((m + row_tile - 1) // row_tile)
        * (n // 128)
        * ((k + 63) // 64)
    )
    return min(2048, max(128, 1 << max(tiles - 1, 0).bit_length()))


def marlin_linear(x: tensorplay.Tensor, Q: tensorplay.Tensor, S: tensorplay.Tensor,
                  group_size: int, *, prefill: bool,
                  out: tensorplay.Tensor | None = None) -> tensorplay.Tensor:
    """``x`` [M, K] bf16 @ dequant(Q, S) -> [M, N] bf16 (marlin packed GEMM)."""

    rows = int(x.shape[0]) if x.dim() > 1 else 1
    k = int(x.shape[1])
    n = int(S.shape[1])
    dev = x.device
    # Marlin's row tile is 8 (decode) or 16 (prefill) rows.  Without padding,
    # a partial tile writes past the output allocation and corrupts adjacent
    # tensors, which shows up as random NaNs on later GEMMs.
    tile = 8 if not prefill else 16
    m = ((rows + tile - 1) // tile) * tile
    if x.dim() == 1:
        x = x.reshape(1, k)
    if m != rows:
        padded = tensorplay.zeros((m, k), dtype=x.dtype, device=dev)
        padded[:rows] = x
        x = padded
    if out is None:
        out = tensorplay.empty((m, n), dtype=tensorplay.bfloat16, device=dev)
    locks = _locks(dev)
    # vLLM zero-inits its Marlin workspace on every GEMM; the barriers assume
    # a clean slate.  Reusing a dirty buffer corrupts split-K reductions.
    locks.zero_()
    fn = _lib().marlin_prefill if prefill else _lib().marlin_decode
    rc = fn(
        ctypes.c_void_p(x.data_ptr()), ctypes.c_void_p(Q.data_ptr()),
        ctypes.c_void_p(out.data_ptr()),
        ctypes.c_void_p(_c_tmp(dev).data_ptr()),
        ctypes.c_void_p(locks.data_ptr()), ctypes.c_void_p(S.data_ptr()),
        k // group_size, m, n, k, k,
        # vLLM gives each SM two blocks for these small tiles; with only one
        # block per SM the kernel takes its part2 path, whose split-K barriers
        # misreduce whenever there is more than one N tile.
        _max_smem(), _grid_for(m, n, k, prefill), _stream())
    if rc:
        raise RuntimeError(f"marlin launch failed (rc={rc})")
    return out if rows == m else out[:rows]


def release_locks() -> None:
    _LOCKS.clear()
