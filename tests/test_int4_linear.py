import pytest
import tensorplay
import tensorplay.nn.functional as F

from megalm.models.components.int4_linear import (
    Int4GroupLinear,
    unpack_packed_int4,
)


def _pack_codes(codes: tensorplay.Tensor) -> tensorplay.Tensor:
    """Compressed-tensors layout: eight codes per int32, low nibble first."""

    rows, width = codes.shape
    grouped = codes.reshape(rows, width // 8, 8).to(tensorplay.int32)
    packed = tensorplay.zeros((rows, width // 8), dtype=tensorplay.int32)
    for index in range(8):
        packed = packed + grouped[:, :, index].bitwise_left_shift(4 * index)
    return packed


def _dequantized(
    codes: tensorplay.Tensor,
    scale: tensorplay.Tensor,
    group_size: int,
) -> tensorplay.Tensor:
    rows, width = codes.shape
    groups = codes.to(tensorplay.float32).reshape(
        rows, width // group_size, group_size
    )
    return (
        (groups - 8.0) * scale.to(tensorplay.float32).unsqueeze(-1)
    ).reshape(rows, width)


def test_unpack_packed_int4_round_trips_nibble_order() -> None:
    codes = tensorplay.tensor(
        [[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]],
        dtype=tensorplay.int32,
    )
    assert (
        unpack_packed_int4(_pack_codes(codes)).to(tensorplay.int32)
        == codes
    ).all()


def test_int4_projection_matches_dequantized_matmul() -> None:
    group_size = 32
    out_features, in_features = 64, 128
    tensorplay.manual_seed(7)
    codes = tensorplay.randint(
        0,
        16,
        (out_features, in_features),
        dtype=tensorplay.int32,
    )
    scale = (
        tensorplay.rand((out_features, in_features // group_size))
        * 0.05
    ).to(tensorplay.bfloat16)
    linear = Int4GroupLinear(
        in_features,
        out_features,
        group_size=group_size,
        device="cpu",
        dtype=tensorplay.bfloat16,
    )
    linear.load_compressed(_pack_codes(codes), scale, in_features=in_features)

    hidden = (
        tensorplay.rand((3, in_features), dtype=tensorplay.bfloat16) - 0.5
    )
    actual = linear(hidden)
    weight = _dequantized(codes, scale, group_size).to(tensorplay.bfloat16)
    expected = F.linear(hidden, weight)
    assert tensorplay.allclose(actual, expected, atol=0.02, rtol=0.02)


def test_int4_projection_rejects_shape_drift() -> None:
    linear = Int4GroupLinear(
        64,
        32,
        group_size=32,
        device="cpu",
        dtype=tensorplay.bfloat16,
    )
    codes = tensorplay.zeros((32, 64), dtype=tensorplay.int32)
    packed = _pack_codes(codes)
    scale = tensorplay.ones((32, 2), dtype=tensorplay.bfloat16)
    with pytest.raises(ValueError, match="width"):
        linear.load_compressed(packed, scale, in_features=32)