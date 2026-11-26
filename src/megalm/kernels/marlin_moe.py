"""Grouped W4A16 Marlin for symmetric group-32 BF16 experts."""

from __future__ import annotations

import ctypes
import hashlib
import os
import subprocess
import tempfile
from functools import lru_cache
from pathlib import Path

import tensorplay

from megalm.kernels.marlin_int4 import _nvcc

_CSRC = Path(__file__).resolve().parent / "csrc"
_SOURCE = _CSRC / "marlin_moe.cu"
_GROUP = 32
_TILE_N = 128
_CONFIG_COUNT = 7


@lru_cache(maxsize=None)
def _library(major: int, minor: int) -> ctypes.CDLL:
    arch = f"sm_{major}{minor}"
    digest = hashlib.sha256(arch.encode() + _SOURCE.read_bytes())
    for header in sorted(
        path for path in (_CSRC / "marlin").rglob("*") if path.is_file()
    ):
        digest.update(header.read_bytes())
    cache = Path(tempfile.gettempdir()) / "megalm_marlin_moe"
    cache.mkdir(exist_ok=True)
    library = cache / f"{arch}_{digest.hexdigest()[:24]}.so"
    if not library.exists():
        with tempfile.TemporaryDirectory(dir=cache) as build:
            artifact = Path(build) / library.name
            try:
                subprocess.run(
                    [
                        _nvcc(),
                        f"-arch={arch}",
                        "-O3",
                        "-std=c++20",
                        "--expt-relaxed-constexpr",
                        "-shared",
                        "-Xcompiler",
                        "-fPIC",
                        "-I",
                        str(_CSRC),
                        "-o",
                        str(artifact),
                        str(_SOURCE),
                    ],
                    capture_output=True,
                    text=True,
                    check=True,
                )
            except subprocess.CalledProcessError as exc:
                raise RuntimeError(
                    f"Failed to compile grouped Marlin: {exc.stderr}"
                ) from exc
            os.replace(artifact, library)
    lib = ctypes.CDLL(str(library))
    lib.megalm_marlin_moe_prepare.argtypes = [
        ctypes.c_int,
        ctypes.POINTER(ctypes.c_int),
    ]
    lib.megalm_marlin_moe_prepare.restype = ctypes.c_int
    lib.megalm_marlin_moe_pack.argtypes = (
        [ctypes.c_void_p] * 4 + [ctypes.c_int] * 3 + [ctypes.c_void_p]
    )
    lib.megalm_marlin_moe_pack.restype = ctypes.c_int
    lib.megalm_marlin_project.argtypes = (
        [ctypes.c_void_p] * 9 + [ctypes.c_int] * 6 + [ctypes.c_void_p]
    )
    lib.megalm_marlin_project.restype = ctypes.c_int
    lib.megalm_marlin_moe.argtypes = (
        [ctypes.c_void_p] * 15 + [ctypes.c_int] * 10 + [ctypes.c_void_p]
    )
    lib.megalm_marlin_moe.restype = ctypes.c_int
    return lib


@lru_cache(maxsize=None)
def _prepare(device_index: int) -> tuple[ctypes.CDLL, int, tuple[int, ...]]:
    properties = tensorplay.cuda.get_device_properties(device_index)
    lib = _library(int(properties.major), int(properties.minor))
    grids = (ctypes.c_int * _CONFIG_COUNT)()
    with tensorplay.cuda.device(device_index):
        error = lib.megalm_marlin_moe_prepare(device_index, grids)
    if error:
        raise RuntimeError(f"Grouped Marlin setup failed with CUDA error {error}")
    return lib, int(properties.multi_processor_count), tuple(grids)


def supports_marlin_moe(
    device: tensorplay.device,
    dtype: tensorplay.dtype,
    hidden: int,
    intermediate: int,
    experts: int,
    group_size: int,
) -> bool:
    if (
        device.type != "cuda"
        or dtype != tensorplay.bfloat16
        or group_size != _GROUP
        or hidden % _TILE_N
        or (2 * intermediate) % _TILE_N
        or not 0 < experts <= 256
    ):
        return False
    properties = tensorplay.cuda.get_device_properties(device)
    return int(properties.major) >= 8


def prepare_marlin_moe(device: tensorplay.device) -> None:
    """Prepare Marlin before checkpoint loading or CUDA graph capture."""
    index = device.index if device.index is not None else tensorplay.cuda.current_device()
    _prepare(index)


def pack_experts(
    packed: tensorplay.Tensor, scale_zero: tensorplay.Tensor
) -> tuple[tensorplay.Tensor, tensorplay.Tensor]:
    """Repack checkpoint codes and BF16 scales on their owning CUDA stream."""
    experts, n, half_k = packed.shape
    k = half_k * 2
    device = packed.device
    lib, _, _ = _prepare(device.index)
    q = tensorplay.empty(
        (experts, k // 16, n * 2), device=device, dtype=tensorplay.int32
    )
    s = tensorplay.empty(
        (experts, k // _GROUP, n), device=device, dtype=tensorplay.bfloat16
    )
    with tensorplay.cuda.device(device.index):
        error = lib.megalm_marlin_moe_pack(
            packed.data_ptr(),
            scale_zero.data_ptr(),
            q.data_ptr(),
            s.data_ptr(),
            experts,
            n,
            k,
            tensorplay.cuda.current_stream(device).cuda_stream,
        )
    if error:
        raise RuntimeError(f"Grouped Marlin pack failed with CUDA error {error}")
    return q, s


def _launch_config(
    block_rows: int,
    k: int,
    n: int,
    routes: int,
    sms: int,
    grids: tuple[int, ...],
) -> tuple[int, int]:
    if block_rows <= 16:
        config = 0 if block_rows == 8 else 2
        if k % 128 == 0 and grids[config + 1] >= grids[config]:
            config += 1
    else:
        config = block_rows // 16 + 2
    # For small dispatches, avoid more split-K stripes than there is work.
    work_grid = sms * max(1, n // _TILE_N * routes * 4 // sms)
    return config, min(grids[config], work_grid)


def marlin_project(
    hidden_states: tensorplay.Tensor,
    packed: tensorplay.Tensor,
    scales: tensorplay.Tensor,
) -> tensorplay.Tensor:
    """One dense projection using the grouped kernel's one-expert schedule."""
    rows, k = hidden_states.shape
    n = int(scales.shape[-1])
    device = hidden_states.device
    if not rows:
        return tensorplay.empty((0, n), device=device, dtype=hidden_states.dtype)
    lib, sms, grids = _prepare(device.index)
    block_rows = next(size for size in (8, 16, 32, 64) if rows <= size or size == 64)
    config, grid = _launch_config(block_rows, k, n, rows, sms, grids)
    blocks = (rows + block_rows - 1) // block_rows
    sorted_ids = tensorplay.empty(
        (blocks * block_rows,), device=device, dtype=tensorplay.int32
    )
    expert_ids = tensorplay.empty((blocks,), device=device, dtype=tensorplay.int32)
    padded = tensorplay.empty((1,), device=device, dtype=tensorplay.int32)
    locks = tensorplay.empty((grid,), device=device, dtype=tensorplay.int32)
    temporary = tensorplay.empty(
        (grid * max(16, block_rows) * _TILE_N,), device=device, dtype=tensorplay.float32
    )
    output = tensorplay.empty((rows, n), device=device, dtype=tensorplay.bfloat16)
    inputs = (
        hidden_states.contiguous(),
        packed,
        scales,
        sorted_ids,
        expert_ids,
        padded,
        locks,
        temporary,
        output,
    )
    with tensorplay.cuda.device(device.index):
        error = lib.megalm_marlin_project(
            *(item.data_ptr() for item in inputs),
            rows,
            n,
            k,
            block_rows,
            config,
            grid,
            tensorplay.cuda.current_stream(device).cuda_stream,
        )
    if error:
        raise RuntimeError(f"Marlin projection failed with CUDA error {error}")
    return output


def grouped_marlin_moe(
    hidden_states: tensorplay.Tensor,
    topk_ids: tensorplay.Tensor,
    topk_weights: tensorplay.Tensor,
    q13: tensorplay.Tensor,
    s13: tensorplay.Tensor,
    q2: tensorplay.Tensor,
    s2: tensorplay.Tensor,
) -> tensorplay.Tensor:
    """Compute the two expert projections and sum weighted routes in FP32.

    Inputs come from the local top-k router and the Marlin checkpoint packer.
    All scratch is stream-local and may be reclaimed in stream order as soon
    as the forward returns. The first call per CUDA device prepares the kernels.
    """
    tokens, hidden = hidden_states.shape
    if tokens == 0:
        return tensorplay.empty_like(hidden_states)
    top_k = int(topk_ids.shape[1])
    routes = tokens * top_k
    experts = int(q13.shape[0])
    intermediate = int(q2.shape[1]) * 16
    device = hidden_states.device
    lib, sms, grids = _prepare(device.index)
    for block_rows in (8, 16, 32, 48, 64):
        if routes * 10 < experts * block_rows * 9:
            break
    up_config, up_grid = _launch_config(
        block_rows, hidden, 2 * intermediate, routes, sms, grids
    )
    down_config, down_grid = _launch_config(
        block_rows, intermediate, hidden, routes, sms, grids
    )

    grid = max(up_grid, down_grid)
    max_blocks = (routes + experts * (block_rows - 1)) // block_rows
    sorted_ids = tensorplay.empty(
        (max_blocks * block_rows,), device=device, dtype=tensorplay.int32
    )
    expert_ids = tensorplay.empty((max_blocks,), device=device, dtype=tensorplay.int32)
    padded = tensorplay.empty((1,), device=device, dtype=tensorplay.int32)
    locks = tensorplay.empty((grid,), device=device, dtype=tensorplay.int32)
    temporary = tensorplay.empty(
        (grid * max(16, block_rows) * _TILE_N,), device=device, dtype=tensorplay.float32
    )
    stage = tensorplay.empty(
        (routes * max(hidden, 2 * intermediate),),
        device=device,
        dtype=tensorplay.bfloat16,
    )
    activation = tensorplay.empty(
        (routes * intermediate,), device=device, dtype=tensorplay.bfloat16
    )
    output = tensorplay.empty(
        (tokens, hidden), device=device, dtype=tensorplay.bfloat16
    )

    inputs = (
        hidden_states.contiguous(),
        topk_ids.long().contiguous(),
        topk_weights.float().contiguous(),
        q13,
        s13,
        q2,
        s2,
        sorted_ids,
        expert_ids,
        padded,
        locks,
        temporary,
        stage,
        activation,
        output,
    )
    with tensorplay.cuda.device(device.index):
        error = lib.megalm_marlin_moe(
            *(item.data_ptr() for item in inputs),
            tokens,
            top_k,
            experts,
            hidden,
            intermediate,
            block_rows,
            up_config,
            down_config,
            up_grid,
            down_grid,
            tensorplay.cuda.current_stream(device).cuda_stream,
        )
    if error:
        raise RuntimeError(f"Grouped Marlin launch failed with CUDA error {error}")
    return output
