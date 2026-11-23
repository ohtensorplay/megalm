from __future__ import annotations

import tensorplay
import triton
import triton.language as tl

@triton.jit
def _prepare_sampling_logits_kernel(
    logits_ptr,
    row_indices_ptr,
    temperatures_ptr,
    output_ptr,
    vocab_size: tl.constexpr,
    BLOCK: tl.constexpr,
):
    output_row = tl.program_id(0)
    block_index = tl.program_id(1)
    source_row = tl.load(row_indices_ptr + output_row)
    temperature = tl.load(temperatures_ptr + output_row).to(tl.float32)
    temperature = tl.where(temperature > 0, temperature, 1.0)
    offsets = block_index * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < vocab_size
    logits = tl.load(
        logits_ptr + source_row * vocab_size + offsets,
        mask=mask,
        other=-float("inf"),
    ).to(tl.float32)
    tl.store(
        output_ptr + output_row * vocab_size + offsets,
        logits / temperature,
        mask=mask,
    )


def prepare_sampling_logits(
    logits: tensorplay.Tensor,
    row_indices: tensorplay.Tensor,
    temperatures: tensorplay.Tensor,
    output: tensorplay.Tensor | None = None,
) -> tensorplay.Tensor:
    if logits.ndim != 2:
        raise ValueError("sampling logits must have shape [batch, vocabulary]")
    if row_indices.ndim != 1 or temperatures.shape != row_indices.shape:
        raise ValueError("sampling row metadata has an invalid shape")
    expected_shape = (row_indices.numel(), logits.shape[1])
    if output is None:
        output = tensorplay.empty(
            expected_shape,
            dtype=tensorplay.float32,
            device=logits.device,
        )
    elif output.shape != expected_shape or output.dtype != tensorplay.float32:
        raise ValueError("sampling score workspace has an invalid layout")
    block = 256
    _prepare_sampling_logits_kernel[
        (row_indices.numel(), triton.cdiv(logits.shape[1], block))
    ](
        logits,
        row_indices,
        temperatures,
        output,
        vocab_size=logits.shape[1],
        BLOCK=block,
        num_warps=4,
    )
    return output


