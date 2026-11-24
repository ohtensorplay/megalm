"""Tests for the single-token decode GEMV kernels."""

from __future__ import annotations

import pytest

import tensorplay
import tensorplay.nn.functional as F

pytest.importorskip("triton")

from megalm.kernels.decode_gemv import decode_gemv, gate_up_silu_gemv  # noqa: E402


def _close(a, b, atol: float) -> bool:
    return bool((a.float() - b.float()).abs().max().item() <= atol)


@pytest.mark.skipif(tensorplay.cuda.device_count() == 0, reason="requires CUDA")
def test_decode_gemv_qkv_shape_and_values() -> None:
    x = tensorplay.randn((1024,), dtype=tensorplay.bfloat16, device="cuda")
    weight = tensorplay.randn((2688, 1024), dtype=tensorplay.bfloat16, device="cuda")
    got = decode_gemv(x, weight)
    expected = F.linear(x.unsqueeze(0), weight)[0]
    assert tuple(got.shape) == (2688,)
    assert _close(got, expected, atol=0.25)


@pytest.mark.skipif(tensorplay.cuda.device_count() == 0, reason="requires CUDA")
def test_decode_gemv_rejects_bad_shapes() -> None:
    x = tensorplay.randn((8,), dtype=tensorplay.bfloat16, device="cuda")
    w = tensorplay.randn((16, 16), dtype=tensorplay.bfloat16, device="cuda")
    with pytest.raises(ValueError):
        decode_gemv(x, w)


@pytest.mark.skipif(tensorplay.cuda.device_count() == 0, reason="requires CUDA")
def test_gate_up_silu_gemv_matches_reference() -> None:
    x = tensorplay.randn((1024,), dtype=tensorplay.bfloat16, device="cuda") * 0.1
    gate_up = tensorplay.randn((6144, 1024), dtype=tensorplay.bfloat16, device="cuda") * 0.1
    got = gate_up_silu_gemv(x, gate_up)
    gate, up = gate_up.chunk(2, dim=0)
    expected = F.silu(F.linear(x.unsqueeze(0), gate)[0]) * F.linear(x.unsqueeze(0), up)[0]
    assert tuple(got.shape) == (3072,)
    assert _close(got, expected, atol=0.25)


@pytest.mark.skipif(tensorplay.cuda.device_count() == 0, reason="requires CUDA")
def test_gate_up_silu_gemv_rejects_odd_pack() -> None:
    x = tensorplay.randn((8,), dtype=tensorplay.bfloat16, device="cuda")
    w = tensorplay.randn((15, 8), dtype=tensorplay.bfloat16, device="cuda")
    with pytest.raises(ValueError):
        gate_up_silu_gemv(x, w)


@pytest.mark.skipif(tensorplay.cuda.device_count() == 0, reason="requires CUDA")
def test_decode_gemv_preserves_fp16_without_bf16_round_trip() -> None:
    """The store must convert fp32 directly to the caller's dtype.

    A previous implementation rounded through bf16, which costs ~0.25 abs at
    a magnitude of 120 (several fp16 ULPs) for fp16 callers.
    """
    x = tensorplay.randn((1024,), dtype=tensorplay.float16, device="cuda")
    weight = tensorplay.randn((2688, 1024), dtype=tensorplay.float16, device="cuda")
    got = decode_gemv(x, weight)
    expected = F.linear(x.unsqueeze(0), weight)[0]
    assert got.dtype == tensorplay.float16
    assert _close(got, expected, atol=0.0625)


@pytest.mark.skipif(tensorplay.cuda.device_count() == 0, reason="requires CUDA")
def test_gate_up_silu_gemv_preserves_fp16() -> None:
    x = tensorplay.randn((1024,), dtype=tensorplay.float16, device="cuda") * 0.1
    gate_up = tensorplay.randn((6144, 1024), dtype=tensorplay.float16, device="cuda") * 0.1
    got = gate_up_silu_gemv(x, gate_up)
    gate, up = gate_up.chunk(2, dim=0)
    expected = F.silu(F.linear(x.unsqueeze(0), gate)[0]) * F.linear(x.unsqueeze(0), up)[0]
    assert got.dtype == tensorplay.float16
    assert _close(got, expected, atol=1e-3)
