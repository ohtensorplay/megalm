from __future__ import annotations

import tensorplay
import triton
import triton.language as tl


@triton.jit
def _relay_speculative_chain_kernel(
    source_input_ids_ptr,
    source_positions_ptr,
    source_slots_ptr,
    source_blocks_ptr,
    source_mrope_ptr,
    target_input_ids_ptr,
    target_positions_ptr,
    target_slots_ptr,
    target_blocks_ptr,
    target_sequence_lengths_ptr,
    target_mrope_ptr,
    source_mrope_row_stride,
    source_mrope_col_stride,
    target_mrope_row_stride,
    target_mrope_col_stride,
    chain_length: tl.constexpr,
    block_count: tl.constexpr,
    HAS_MROPE: tl.constexpr,
    BLOCK: tl.constexpr,
):
    offsets = tl.arange(0, BLOCK)
    token_mask = offsets < chain_length
    positions = tl.load(
        source_positions_ptr + offsets,
        mask=token_mask,
        other=0,
    )
    tl.store(
        target_input_ids_ptr + offsets,
        tl.load(
            source_input_ids_ptr + offsets,
            mask=token_mask,
            other=0,
        ),
        mask=token_mask,
    )
    tl.store(
        target_positions_ptr + offsets,
        positions,
        mask=token_mask,
    )
    tl.store(
        target_slots_ptr + offsets,
        tl.load(
            source_slots_ptr + offsets,
            mask=token_mask,
            other=0,
        ),
        mask=token_mask,
    )
    tl.store(
        target_sequence_lengths_ptr + offsets,
        positions + 1,
        mask=token_mask,
    )
    block_mask = offsets < block_count
    tl.store(
        target_blocks_ptr + offsets,
        tl.load(
            source_blocks_ptr + offsets,
            mask=block_mask,
            other=-1,
        ),
        mask=block_mask,
    )
    if HAS_MROPE:
        mrope_offsets = offsets
        mrope_mask = mrope_offsets < 3 * chain_length
        mrope_rows = mrope_offsets // chain_length
        mrope_cols = mrope_offsets % chain_length
        tl.store(
            target_mrope_ptr
            + mrope_rows * target_mrope_row_stride
            + mrope_cols * target_mrope_col_stride,
            tl.load(
                source_mrope_ptr
                + mrope_rows * source_mrope_row_stride
                + mrope_cols * source_mrope_col_stride,
                mask=mrope_mask,
                other=0,
            ),
            mask=mrope_mask,
        )


def relay_speculative_chain(
    source_input_ids: tensorplay.Tensor,
    source_positions: tensorplay.Tensor,
    source_slots: tensorplay.Tensor,
    source_blocks: tensorplay.Tensor,
    target_input_ids: tensorplay.Tensor,
    target_positions: tensorplay.Tensor,
    target_slots: tensorplay.Tensor,
    target_blocks: tensorplay.Tensor,
    target_sequence_lengths: tensorplay.Tensor,
    *,
    source_mrope: tensorplay.Tensor | None = None,
    target_mrope: tensorplay.Tensor | None = None,
) -> None:
    """Relay one fixed-length speculative chain with a single GPU launch."""

    chain_length = source_input_ids.numel()
    if (
        chain_length < 2
        or source_input_ids.shape != (chain_length,)
        or source_positions.shape != (chain_length,)
        or source_slots.shape != (chain_length,)
        or target_input_ids.shape != (chain_length,)
        or target_positions.shape != (chain_length,)
        or target_slots.shape != (chain_length,)
        or target_sequence_lengths.shape != (chain_length,)
        or source_blocks.shape != target_blocks.shape
        or source_blocks.ndim != 2
        or source_blocks.shape[0] != 1
    ):
        raise ValueError("speculative relay geometry is invalid")
    tensors = (
        source_input_ids,
        source_positions,
        source_slots,
        source_blocks,
        target_input_ids,
        target_positions,
        target_slots,
        target_blocks,
        target_sequence_lengths,
    )
    if (
        source_input_ids.device.type != "cuda"
        or any(tensor.device != source_input_ids.device for tensor in tensors)
        or any(
            tensor.dtype != tensorplay.int64
            for tensor in (
                source_input_ids,
                source_positions,
                source_slots,
                target_input_ids,
                target_positions,
                target_slots,
            )
        )
        or source_blocks.dtype != tensorplay.int32
        or target_blocks.dtype != tensorplay.int32
        or target_sequence_lengths.dtype != tensorplay.int32
    ):
        raise ValueError("speculative relay buffer dtype/device is invalid")
    has_mrope = source_mrope is not None or target_mrope is not None
    if has_mrope and (
        source_mrope is None
        or target_mrope is None
        or source_mrope.shape != (3, chain_length)
        or target_mrope.shape != (3, chain_length)
        or source_mrope.dtype != tensorplay.int64
        or target_mrope.dtype != tensorplay.int64
        or source_mrope.device != source_input_ids.device
        or target_mrope.device != source_input_ids.device
    ):
        raise ValueError("speculative MRoPE relay geometry is invalid")
    block_count = source_blocks.numel()
    relay_width = max(
        chain_length,
        block_count,
        3 * chain_length if has_mrope else 0,
    )
    block = triton.next_power_of_2(relay_width)
    mrope_source = (
        source_input_ids if source_mrope is None else source_mrope
    )
    mrope_target = (
        target_input_ids if target_mrope is None else target_mrope
    )
    _relay_speculative_chain_kernel[(1,)](
        source_input_ids,
        source_positions,
        source_slots,
        source_blocks,
        mrope_source,
        target_input_ids,
        target_positions,
        target_slots,
        target_blocks,
        target_sequence_lengths,
        mrope_target,
        mrope_source.stride(0) if has_mrope else 0,
        mrope_source.stride(1) if has_mrope else 0,
        mrope_target.stride(0) if has_mrope else 0,
        mrope_target.stride(1) if has_mrope else 0,
        chain_length=chain_length,
        block_count=block_count,
        HAS_MROPE=has_mrope,
        BLOCK=block,
        num_warps=1,
    )


__all__ = ["relay_speculative_chain"]
