from __future__ import annotations

import tensorplay
import triton
import triton.language as tl

@triton.jit
def _greedy_argmax_kernel(
    logits_ptr,
    sample_indices_ptr,
    output_ptr,
    vocab_size: tl.constexpr,
    HAS_SAMPLE_INDICES: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    active = True
    if HAS_SAMPLE_INDICES:
        active = tl.load(sample_indices_ptr + row) >= 0
    offsets = tl.arange(0, BLOCK)
    values = tl.load(
        logits_ptr + row * vocab_size + offsets,
        mask=(offsets < vocab_size) & active,
        other=-float("inf"),
    ).to(tl.float32)
    token = tl.argmax(values, axis=0)
    tl.store(output_ptr + row, tl.where(active, token, -1))


@triton.jit
def _greedy_argmax_partials_kernel(
    logits_ptr,
    sample_indices_ptr,
    partial_values_ptr,
    partial_indices_ptr,
    vocab_size,
    num_chunks,
    HAS_SAMPLE_INDICES: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    chunk = tl.program_id(1)
    active = True
    if HAS_SAMPLE_INDICES:
        active = tl.load(sample_indices_ptr + row) >= 0
    offsets = chunk * BLOCK + tl.arange(0, BLOCK)
    mask = (offsets < vocab_size) & active
    values = tl.load(
        logits_ptr + row * vocab_size + offsets,
        mask=mask,
        other=-float("inf"),
    ).to(tl.float32)
    maximum = tl.max(values, axis=0)
    local_index = tl.argmax(values, axis=0)
    partial_offset = row * num_chunks + chunk
    tl.store(partial_values_ptr + partial_offset, maximum)
    tl.store(partial_indices_ptr + partial_offset, chunk * BLOCK + local_index)


@triton.jit
def _greedy_argmax_reduce_kernel(
    sample_indices_ptr,
    partial_values_ptr,
    partial_indices_ptr,
    output_ptr,
    num_chunks: tl.constexpr,
    HAS_SAMPLE_INDICES: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    active = True
    if HAS_SAMPLE_INDICES:
        active = tl.load(sample_indices_ptr + row) >= 0
    offsets = tl.arange(0, BLOCK)
    mask = offsets < num_chunks
    values = tl.load(
        partial_values_ptr + row * num_chunks + offsets,
        mask=mask,
        other=-float("inf"),
    )
    indices = tl.load(
        partial_indices_ptr + row * num_chunks + offsets,
        mask=mask,
        other=0x7FFFFFFF,
    )
    maximum = tl.max(values, axis=0)
    candidates = tl.where(values == maximum, indices, 0x7FFFFFFF)
    token = tl.min(candidates, axis=0)
    tl.store(output_ptr + row, tl.where(active, token, -1))


def greedy_argmax(
    logits: tensorplay.Tensor,
    *,
    sample_indices: tensorplay.Tensor | None = None,
    output: tensorplay.Tensor | None = None,
    partial_values: tensorplay.Tensor | None = None,
    partial_indices: tensorplay.Tensor | None = None,
) -> tensorplay.Tensor:
    if logits.ndim != 2 or logits.shape[0] <= 0 or logits.shape[1] <= 0:
        raise ValueError("greedy logits must have shape [batch, vocabulary]")
    logits = logits.contiguous()
    batch_size = logits.shape[0]
    if sample_indices is not None:
        if (
            sample_indices.shape != (batch_size,)
            or sample_indices.device != logits.device
            or sample_indices.dtype not in {tensorplay.int32, tensorplay.int64}
        ):
            raise ValueError(
                "sample indices must be an int32/int64 tensor with shape [batch]"
            )
        sample_indices = sample_indices.contiguous()
    if output is None:
        output = tensorplay.empty(
            batch_size,
            dtype=tensorplay.int64,
            device=logits.device,
        )
    elif (
        output.shape != (batch_size,)
        or output.device != logits.device
        or output.dtype != tensorplay.int64
    ):
        raise ValueError(
            "greedy output must be an int64 tensor with shape [batch]"
        )
    sample_indices_ptr = logits if sample_indices is None else sample_indices
    vocab_size = logits.shape[-1]
    if vocab_size <= 65536:
        block = triton.next_power_of_2(vocab_size)
        _greedy_argmax_kernel[(batch_size,)](
            logits,
            sample_indices_ptr,
            output,
            vocab_size=vocab_size,
            HAS_SAMPLE_INDICES=sample_indices is not None,
            BLOCK=block,
            num_warps=8,
        )
    else:
        chunk_size = 16384
        num_chunks = triton.cdiv(vocab_size, chunk_size)
        partial_shape = (batch_size, num_chunks)
        if partial_values is None:
            partial_values = tensorplay.empty(
                partial_shape,
                dtype=tensorplay.float32,
                device=logits.device,
            )
        elif (
            partial_values.shape != partial_shape
            or partial_values.device != logits.device
            or partial_values.dtype != tensorplay.float32
        ):
            raise ValueError(
                "greedy partial values do not match the required workspace"
            )
        if partial_indices is None:
            partial_indices = tensorplay.empty(
                partial_shape,
                dtype=tensorplay.int32,
                device=logits.device,
            )
        elif (
            partial_indices.shape != partial_shape
            or partial_indices.device != logits.device
            or partial_indices.dtype != tensorplay.int32
        ):
            raise ValueError(
                "greedy partial indices do not match the required workspace"
            )
        _greedy_argmax_partials_kernel[(batch_size, num_chunks)](
            logits,
            sample_indices_ptr,
            partial_values,
            partial_indices,
            vocab_size,
            num_chunks,
            HAS_SAMPLE_INDICES=sample_indices is not None,
            BLOCK=chunk_size,
            num_warps=8,
        )
        _greedy_argmax_reduce_kernel[(batch_size,)](
            sample_indices_ptr,
            partial_values,
            partial_indices,
            output,
            num_chunks=num_chunks,
            HAS_SAMPLE_INDICES=sample_indices is not None,
            BLOCK=triton.next_power_of_2(num_chunks),
            num_warps=4,
        )
    return output
