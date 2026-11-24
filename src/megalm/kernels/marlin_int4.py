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
        common = [ctypes.c_void_p] * 5 + [ctypes.c_int] * 5 + [
            ctypes.c_int, ctypes.c_int, ctypes.c_void_p]
        _LIB.marlin_decode.argtypes = common
        _LIB.marlin_prefill.argtypes = common
    return _LIB


_STREAM = None
_LOCKS = {}


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


def marlin_linear(x: tensorplay.Tensor, Q: tensorplay.Tensor, S: tensorplay.Tensor,
                  group_size: int, *, prefill: bool,
                  out: tensorplay.Tensor | None = None) -> tensorplay.Tensor:
    """``x`` [M, K] bf16 @ dequant(Q, S) -> [M, N] bf16 (marlin packed GEMM)."""

    m = int(x.shape[0]) if x.dim() > 1 else 1
    k = int(x.shape[1])
    n = int(S.shape[1])
    dev = x.device
    if out is None:
        out = tensorplay.empty((m, n), dtype=tensorplay.bfloat16, device=dev)
    locks = _locks(dev)
    fn = _lib().marlin_prefill if prefill else _lib().marlin_decode
    rc = fn(
        ctypes.c_void_p(x.data_ptr()), ctypes.c_void_p(Q.data_ptr()),
        ctypes.c_void_p(out.data_ptr()), ctypes.c_void_p(locks.data_ptr()),
        ctypes.c_void_p(S.data_ptr()), k // group_size, m, n, k, k,
        _max_smem(), _num_sms(), _stream())
    if rc:
        raise RuntimeError(f"marlin launch failed (rc={rc})")
    return out


def release_locks() -> None:
    _LOCKS.clear()
