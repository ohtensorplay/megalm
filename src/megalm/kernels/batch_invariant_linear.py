from __future__ import annotations

import tensorplay
import triton
import triton.language as tl


@triton.jit
def _batch_invariant_linear_kernel(
    input_ptr,
    weight_ptr,
    bias_ptr,
    output_ptr,
    rows,
    output_width,
    input_width: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    output_block = tl.program_id(0)
    output_blocks = tl.cdiv(output_width, BLOCK_N)
    row_block = output_block // output_blocks
    column_block = output_block % output_blocks
    row_offsets = row_block * BLOCK_M + tl.arange(0, BLOCK_M)
    column_offsets = column_block * BLOCK_N + tl.arange(0, BLOCK_N)
    accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for input_start in range(0, input_width, BLOCK_K):
        input_offsets = input_start + tl.arange(0, BLOCK_K)
        activations = tl.load(
            input_ptr
            + row_offsets[:, None] * input_width
            + input_offsets[None, :],
            mask=(
                (row_offsets[:, None] < rows)
                & (input_offsets[None, :] < input_width)
            ),
            other=0.0,
        )
        weights = tl.load(
            weight_ptr
            + column_offsets[None, :] * input_width
            + input_offsets[:, None],
            mask=(
                (column_offsets[None, :] < output_width)
                & (input_offsets[:, None] < input_width)
            ),
            other=0.0,
        )
        accumulator = tl.dot(activations, weights, accumulator)

    if HAS_BIAS:
        bias = tl.load(
            bias_ptr + column_offsets,
            mask=column_offsets < output_width,
            other=0.0,
        )
        accumulator += bias[None, :]
    tl.store(
        output_ptr
        + row_offsets[:, None] * output_width
        + column_offsets[None, :],
        accumulator,
        mask=(
            (row_offsets[:, None] < rows)
            & (column_offsets[None, :] < output_width)
        ),
    )


def batch_invariant_linear(
    inputs: tensorplay.Tensor,
    weight: tensorplay.Tensor,
    bias: tensorplay.Tensor | None = None,
) -> tensorplay.Tensor:
    """Apply one fixed-schedule Tensor Core linear kernel for every batch size."""

    if inputs.ndim < 2 or weight.ndim != 2:
        raise ValueError("batch-invariant linear expects matrix-backed inputs")
    if inputs.shape[-1] != weight.shape[1]:
        raise ValueError("batch-invariant linear input width does not match weight")
    if inputs.device.type != "cuda" or inputs.device != weight.device:
        raise ValueError("batch-invariant linear requires one CUDA device")
    if inputs.dtype != weight.dtype or inputs.dtype not in {
        tensorplay.float16,
        tensorplay.bfloat16,
    }:
        raise ValueError("batch-invariant linear requires one FP16 or BF16 dtype")
    if bias is not None and (
        bias.ndim != 1
        or bias.shape[0] != weight.shape[0]
        or bias.device != inputs.device
        or bias.dtype != inputs.dtype
    ):
        raise ValueError("batch-invariant linear bias does not match output")

    contiguous_inputs = inputs.contiguous()
    contiguous_weight = weight.contiguous()
    rows = contiguous_inputs.numel() // contiguous_inputs.shape[-1]
    if rows <= 0:
        raise ValueError("batch-invariant linear requires at least one row")
    input_width = contiguous_inputs.shape[-1]
    output_width = contiguous_weight.shape[0]
    output = tensorplay.empty(
        (rows, output_width),
        dtype=inputs.dtype,
        device=inputs.device,
    )
    block_m = (
        16
        if rows <= 16 or output_width >= 65536
        else 64
    )
    latency_shape = rows <= 16 and output_width < 65536
    block_n = 32 if latency_shape else 64
    block_k = 64
    grid = (
        triton.cdiv(rows, block_m)
        * triton.cdiv(output_width, block_n),
    )
    _batch_invariant_linear_kernel[grid](
        contiguous_inputs,
        contiguous_weight,
        bias,
        output,
        rows,
        output_width,
        input_width=input_width,
        HAS_BIAS=bias is not None,
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        BLOCK_K=block_k,
        num_warps=4,
        num_stages=4 if latency_shape else 3,
    )
    return output.view(*inputs.shape[:-1], output_width)


__all__ = ["batch_invariant_linear"]
