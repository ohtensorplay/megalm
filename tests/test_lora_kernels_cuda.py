"""Numerical parity between the Triton multi-LoRA kernel and its reference."""

from __future__ import annotations

from typing import Any

import pytest

tensorplay = pytest.importorskip("tensorplay")

from megalm.kernels.lora import lora_add

pytestmark = pytest.mark.skipif(
    not tensorplay.cuda.is_available(),
    reason="LoRA Triton kernel parity requires CUDA",
)

SLOT_PATTERNS = (
    (0, -1, 2, 1, -1, 0, 2),
    (-1, -1, -1, -1),
    (1, 1, 1, 1, 1, 1),
    (2, 0, 0, 2, 1, -1, 0, 1),
)


@pytest.mark.parametrize(
    ("tokens", "in_features", "out_features", "rank", "dtype", "scale", "pattern"),
    [
        (7, 64, 48, 16, tensorplay.bfloat16, 1.0, SLOT_PATTERNS[0]),
        (1, 16, 17, 1, tensorplay.float16, 1.0, SLOT_PATTERNS[1][:1]),
        (5, 129, 33, 7, tensorplay.float16, 2.5, SLOT_PATTERNS[0][:5]),
        (4, 64, 100, 32, tensorplay.bfloat16, 0.25, SLOT_PATTERNS[3][:4]),
        (3, 32, 16, 8, tensorplay.float32, 1.0, SLOT_PATTERNS[0][:3]),
        # Packed qkv/gate-up footprint: three logical r=16 slices stacked.
        (6, 80, 64, 48, tensorplay.float16, 1.0, SLOT_PATTERNS[0]),
    ],
)
def test_lora_add_matches_reference_across_geometries(
    tokens: int,
    in_features: int,
    out_features: int,
    rank: int,
    dtype: Any,
    scale: float,
    pattern: tuple[int, ...],
) -> None:
    tensorplay.manual_seed(101 + rank)
    slots_count = max(1, max(pattern) + 1)
    x = tensorplay.randn(tokens, in_features, device="cuda", dtype=dtype)
    a = tensorplay.randn(
        slots_count, rank, in_features, device="cuda", dtype=dtype
    )
    b = tensorplay.randn(
        slots_count, out_features, rank, device="cuda", dtype=dtype
    )
    slot_ids = tensorplay.tensor(
        pattern[:tokens], dtype=tensorplay.int32, device="cuda"
    )
    expected = tensorplay.zeros(tokens, out_features, device="cuda", dtype=dtype)
    actual = expected.clone()
    lora_add(expected, x, a, b, slot_ids, scale=scale, use_triton=False)
    lora_add(actual, x, a, b, slot_ids, scale=scale, use_triton=True)
    # Triton accumulates the shrink in fp32 while the reference uses the
    # checkpoint dtype, so low-precision deltas can differ by a few ULPs.
    tolerance = (
        (1e-3, 1e-3)
        if dtype == tensorplay.float32
        else (0.6, 0.08)
    )
    tensorplay.testing.assert_close(
        actual, expected, atol=tolerance[0], rtol=tolerance[1]
    )


def test_lora_add_no_slots_is_a_no_op() -> None:
    tensorplay.manual_seed(12)
    output = tensorplay.ones((2, 4), device="cuda", dtype=tensorplay.bfloat16)
    hidden = tensorplay.zeros((2, 8), device="cuda", dtype=tensorplay.bfloat16)
    a = tensorplay.zeros((1, 4, 8), device="cuda", dtype=tensorplay.bfloat16)
    b = tensorplay.zeros((1, 4, 4), device="cuda", dtype=tensorplay.bfloat16)
    before = output.clone()
    lora_add(output, hidden, a, b, None, use_triton=True)
    tensorplay.testing.assert_close(output, before, atol=0, rtol=0)


def test_lora_add_skips_negative_slots() -> None:
    tensorplay.manual_seed(13)
    tokens = 4
    x = tensorplay.randn(tokens, 32, device="cuda", dtype=tensorplay.bfloat16)
    a = tensorplay.randn(2, 8, 32, device="cuda", dtype=tensorplay.bfloat16)
    b = tensorplay.randn(2, 16, 8, device="cuda", dtype=tensorplay.bfloat16)
    slot_ids = tensorplay.full((tokens,), -1, dtype=tensorplay.int32, device="cuda")
    expected = tensorplay.zeros(tokens, 16, device="cuda", dtype=tensorplay.bfloat16)
    actual = expected.clone()
    lora_add(expected, x, a, b, slot_ids, use_triton=False)
    lora_add(actual, x, a, b, slot_ids, use_triton=True)
    tensorplay.testing.assert_close(actual, expected, atol=0, rtol=0)


def test_lora_add_strided_output_falls_back_safely() -> None:
    """A strided view must not be written through the flat Triton kernel."""

    tensorplay.manual_seed(14)
    tokens, in_features, out_features, rank = 5, 40, 24, 8
    x = tensorplay.randn(tokens, in_features, device="cuda", dtype=tensorplay.float16)
    a = tensorplay.randn(
        2, rank, in_features, device="cuda", dtype=tensorplay.float16
    )
    b = tensorplay.randn(
        2, out_features, rank, device="cuda", dtype=tensorplay.float16
    )
    slot_ids = tensorplay.tensor(
        [0, -1, 1, 0, 1], dtype=tensorplay.int32, device="cuda"
    )
    triton_storage = tensorplay.zeros(
        tokens, out_features * 2, device="cuda", dtype=tensorplay.float16
    )
    reference_storage = tensorplay.zeros(
        tokens, out_features * 2, device="cuda", dtype=tensorplay.float16
    )
    actual = triton_storage[:, ::2]
    expected = reference_storage[:, ::2]
    assert not actual.is_contiguous()
    lora_add(actual, x, a, b, slot_ids, use_triton=True)
    lora_add(expected, x, a, b, slot_ids, use_triton=False)
    tensorplay.testing.assert_close(actual, expected, atol=4e-2, rtol=4e-2)
