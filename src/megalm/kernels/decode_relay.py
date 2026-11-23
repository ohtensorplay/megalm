from __future__ import annotations

import tensorplay
import triton
import triton.language as tl


@triton.jit
def _decode_batch_relay_kernel(
    source_input_ids_ptr,
    source_positions_ptr,
    source_slots_ptr,
    source_block_tables_ptr,
    source_sequence_lengths_ptr,
    source_active_ptr,
    source_gdn_state_indices_ptr,
    source_mrope_positions_ptr,
    target_input_ids_ptr,
    target_positions_ptr,
    target_slots_ptr,
    target_block_tables_ptr,
    target_sequence_lengths_ptr,
    target_active_ptr,
    target_gdn_state_indices_ptr,
    target_mrope_positions_ptr,
    active_rows,
    source_mrope_axis_stride,
    source_mrope_row_stride,
    target_mrope_axis_stride,
    target_mrope_row_stride,
    bucket_size: tl.constexpr,
    max_blocks: tl.constexpr,
    BLOCK_ROWS: tl.constexpr,
    BLOCK_BLOCKS: tl.constexpr,
    HAS_GDN_STATE: tl.constexpr,
    HAS_MROPE: tl.constexpr,
):
    rows = tl.program_id(0) * BLOCK_ROWS + tl.arange(0, BLOCK_ROWS)
    target_mask = rows < bucket_size
    source_mask = rows < active_rows
    input_ids = tl.load(
        source_input_ids_ptr + rows,
        mask=source_mask,
        other=0,
    )
    positions = tl.load(
        source_positions_ptr + rows,
        mask=source_mask,
        other=0,
    )
    slots = tl.load(
        source_slots_ptr + rows,
        mask=source_mask,
        other=0,
    )
    sequence_lengths = tl.load(
        source_sequence_lengths_ptr + rows,
        mask=source_mask,
        other=1,
    )
    active = tl.load(
        source_active_ptr + rows,
        mask=source_mask,
        other=0,
    )
    tl.store(target_input_ids_ptr + rows, input_ids, mask=target_mask)
    tl.store(target_positions_ptr + rows, positions, mask=target_mask)
    tl.store(target_slots_ptr + rows, slots, mask=target_mask)
    tl.store(
        target_sequence_lengths_ptr + rows,
        sequence_lengths,
        mask=target_mask,
    )
    tl.store(target_active_ptr + rows, active, mask=target_mask)

    if HAS_GDN_STATE:
        columns = tl.arange(0, 2)
        offsets = rows[:, None] * 2 + columns[None, :]
        values = tl.load(
            source_gdn_state_indices_ptr + offsets,
            mask=source_mask[:, None],
            other=0,
        )
        tl.store(
            target_gdn_state_indices_ptr + offsets,
            values,
            mask=target_mask[:, None],
        )

    if HAS_MROPE:
        axes = tl.arange(0, 4)
        axis_mask = axes < 3
        source_offsets = (
            axes[:, None] * source_mrope_axis_stride
            + rows[None, :] * source_mrope_row_stride
        )
        target_offsets = (
            axes[:, None] * target_mrope_axis_stride
            + rows[None, :] * target_mrope_row_stride
        )
        values = tl.load(
            source_mrope_positions_ptr + source_offsets,
            mask=axis_mask[:, None] & source_mask[None, :],
            other=0,
        )
        tl.store(
            target_mrope_positions_ptr + target_offsets,
            values,
            mask=axis_mask[:, None] & target_mask[None, :],
        )

    block_offsets = tl.arange(0, BLOCK_BLOCKS)
    block_mask = block_offsets < max_blocks
    source_matrix_mask = source_mask[:, None] & block_mask[None, :]
    target_matrix_mask = target_mask[:, None] & block_mask[None, :]
    offsets = rows[:, None] * max_blocks + block_offsets[None, :]
    blocks = tl.load(
        source_block_tables_ptr + offsets,
        mask=source_matrix_mask,
        other=0,
    )
    tl.store(
        target_block_tables_ptr + offsets,
        blocks,
        mask=target_matrix_mask,
    )


def relay_decode_batch(
    source_input_ids: tensorplay.Tensor,
    source_positions: tensorplay.Tensor,
    source_slots: tensorplay.Tensor,
    source_block_tables: tensorplay.Tensor,
    source_sequence_lengths: tensorplay.Tensor,
    source_active: tensorplay.Tensor,
    target_input_ids: tensorplay.Tensor,
    target_positions: tensorplay.Tensor,
    target_slots: tensorplay.Tensor,
    target_block_tables: tensorplay.Tensor,
    target_sequence_lengths: tensorplay.Tensor,
    target_active: tensorplay.Tensor,
    source_gdn_state_indices: tensorplay.Tensor | None = None,
    source_mrope_positions: tensorplay.Tensor | None = None,
    target_gdn_state_indices: tensorplay.Tensor | None = None,
    target_mrope_positions: tensorplay.Tensor | None = None,
) -> None:
    active_rows = source_input_ids.numel()
    bucket_size = target_input_ids.numel()
    if active_rows > bucket_size:
        raise ValueError("decode relay source exceeds its graph bucket")
    if (
        source_block_tables.shape[1] != target_block_tables.shape[1]
        or source_block_tables.shape[0] != active_rows
        or target_block_tables.shape[0] != bucket_size
    ):
        raise ValueError("decode relay block table geometry differs")
    has_gdn = _paired(
        source_gdn_state_indices,
        target_gdn_state_indices,
        name="GDN state",
    )
    has_mrope = _paired(
        source_mrope_positions,
        target_mrope_positions,
        name="MRoPE",
    )
    if has_gdn and (
        source_gdn_state_indices.shape != (active_rows, 2)
        or target_gdn_state_indices.shape != (bucket_size, 2)
    ):
        raise ValueError("decode relay GDN geometry differs")
    if has_mrope and (
        source_mrope_positions.shape != (3, active_rows)
        or target_mrope_positions.shape != (3, bucket_size)
    ):
        raise ValueError("decode relay MRoPE geometry differs")
    source_gdn = (
        source_positions
        if source_gdn_state_indices is None
        else source_gdn_state_indices
    )
    target_gdn = (
        target_positions
        if target_gdn_state_indices is None
        else target_gdn_state_indices
    )
    source_mrope = (
        source_positions
        if source_mrope_positions is None
        else source_mrope_positions
    )
    target_mrope = (
        target_positions
        if target_mrope_positions is None
        else target_mrope_positions
    )
    max_blocks = target_block_tables.shape[1]
    block_rows = 16
    _decode_batch_relay_kernel[
        (triton.cdiv(bucket_size, block_rows),)
    ](
        source_input_ids,
        source_positions,
        source_slots,
        source_block_tables,
        source_sequence_lengths,
        source_active,
        source_gdn,
        source_mrope,
        target_input_ids,
        target_positions,
        target_slots,
        target_block_tables,
        target_sequence_lengths,
        target_active,
        target_gdn,
        target_mrope,
        active_rows,
        source_mrope.stride(0),
        source_mrope.stride(1) if source_mrope.ndim > 1 else 0,
        target_mrope.stride(0),
        target_mrope.stride(1) if target_mrope.ndim > 1 else 0,
        bucket_size=bucket_size,
        max_blocks=max_blocks,
        BLOCK_ROWS=block_rows,
        BLOCK_BLOCKS=triton.next_power_of_2(max_blocks),
        HAS_GDN_STATE=has_gdn,
        HAS_MROPE=has_mrope,
        num_warps=4,
    )


def _paired(
    source: tensorplay.Tensor | None,
    target: tensorplay.Tensor | None,
    *,
    name: str,
) -> bool:
    if (source is None) != (target is None):
        raise ValueError(f"decode relay {name} metadata is incomplete")
    return source is not None


__all__ = ["relay_decode_batch"]
