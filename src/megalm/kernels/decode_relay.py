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


@triton.jit
def _decode_self_loop_kernel(
    input_ids_ptr,
    positions_ptr,
    slot_mapping_ptr,
    block_tables_ptr,
    sequence_lengths_ptr,
    token_active_ptr,
    sampled_ptr,
    record_ptr,
    counter_ptr,
    mrope_positions_ptr,
    page_size,
    mrope_axis_stride,
    bucket_size: tl.constexpr,
    max_blocks: tl.constexpr,
    record_stride: tl.constexpr,
    RECORD_STEPS: tl.constexpr,
    HAS_MROPE: tl.constexpr,
    BLOCK_ROWS: tl.constexpr,
):
    """Fold one sampled token back into the graph's static decode inputs.

    Captured inside the greedy graph, so a chunk of decode replays needs no
    host round trip: the next replay reads the token this one produced.  Slot
    addresses are recomputed from the block table rather than added to, which
    keeps page crossings exact.  A single program covers the bucket, so the
    record index comes from one uncontended atomic and every row lands in the
    same record row.
    """

    rows = tl.arange(0, BLOCK_ROWS)
    mask = rows < bucket_size
    active = tl.load(token_active_ptr + rows, mask=mask, other=0)
    live = mask & (active != 0)
    step = tl.atomic_add(counter_ptr, 1)
    # A sequence of replays must never write past the record; clamp so a stale
    # counter recycles the last row instead of corrupting memory.
    step = tl.minimum(step, RECORD_STEPS - 1)
    sampled = tl.load(sampled_ptr + rows, mask=mask, other=0)
    tl.store(record_ptr + step * record_stride + rows, sampled, mask=mask)
    positions = tl.load(positions_ptr + rows, mask=mask, other=0)
    lengths = tl.load(sequence_lengths_ptr + rows, mask=mask, other=1)
    # The next replay consumes position+1, so the slot it writes must be
    # addressed by that position, not by this one.
    next_position = positions + 1
    block = next_position // page_size
    pages = tl.load(
        block_tables_ptr + rows * max_blocks + block,
        mask=mask & (block < max_blocks),
        other=0,
    )
    tl.store(input_ids_ptr + rows, sampled, mask=live)
    tl.store(positions_ptr + rows, next_position, mask=live)
    tl.store(
        slot_mapping_ptr + rows,
        tl.where(pages < 0, 0, pages) * page_size + next_position % page_size,
        mask=live,
    )
    tl.store(sequence_lengths_ptr + rows, lengths + 1, mask=live)
    if HAS_MROPE:
        axes = tl.arange(0, 4)
        axis_mask = axes < 3
        offsets = axes[:, None] * mrope_axis_stride + rows[None, :]
        values = tl.load(
            mrope_positions_ptr + offsets,
            mask=axis_mask[:, None] & mask[None, :],
            other=0,
        )
        tl.store(
            mrope_positions_ptr + offsets,
            values + 1,
            mask=axis_mask[:, None] & live[None, :],
        )


def advance_decode_state(
    *,
    input_ids: tensorplay.Tensor,
    positions: tensorplay.Tensor,
    slot_mapping: tensorplay.Tensor,
    block_tables: tensorplay.Tensor,
    sequence_lengths: tensorplay.Tensor,
    token_active: tensorplay.Tensor,
    sampled: tensorplay.Tensor,
    record: tensorplay.Tensor,
    counter: tensorplay.Tensor,
    mrope_positions: tensorplay.Tensor | None,
    page_size: int,
) -> None:
    """Advance a captured decode bucket by one token, on the device."""

    bucket_size = input_ids.numel()
    _decode_self_loop_kernel[(1,)](
        input_ids,
        positions,
        slot_mapping,
        block_tables,
        sequence_lengths,
        token_active,
        sampled,
        record,
        counter,
        positions if mrope_positions is None else mrope_positions,
        page_size,
        0 if mrope_positions is None else mrope_positions.stride(0),
        bucket_size=bucket_size,
        max_blocks=block_tables.shape[1],
        record_stride=record.shape[1],
        RECORD_STEPS=int(record.shape[0]),
        HAS_MROPE=mrope_positions is not None,
        BLOCK_ROWS=triton.next_power_of_2(max(16, bucket_size)),
        num_warps=4,
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


__all__ = ["advance_decode_state", "relay_decode_batch"]
