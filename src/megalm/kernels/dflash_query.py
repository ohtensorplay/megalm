from __future__ import annotations

import tensorplay
import triton
import triton.language as tl


@triton.jit
def _dflash_query_setup_kernel(
    last_token_ids_ptr,
    last_positions_ptr,
    block_tables_ptr,
    block_table_stride: tl.constexpr,
    input_ids_ptr,
    positions_ptr,
    slot_mapping_ptr,
    sample_indices_ptr,
    mask_token_id,
    page_size: tl.constexpr,
    num_speculative_tokens: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    offsets = tl.arange(0, BLOCK)
    query_tokens = num_speculative_tokens + 1
    active = offsets < query_tokens
    base = row * query_tokens

    last_token = tl.load(last_token_ids_ptr + row)
    last_position = tl.load(last_positions_ptr + row)
    query_positions = last_position + 1 + offsets
    block_indices = query_positions // page_size
    page_ids = tl.load(
        block_tables_ptr + row * block_table_stride + block_indices,
        mask=active,
        other=-1,
    ).to(tl.int64)
    slots = page_ids * page_size + query_positions % page_size
    token_ids = tl.where(offsets == 0, last_token, mask_token_id)

    tl.store(input_ids_ptr + base + offsets, token_ids, mask=active)
    tl.store(positions_ptr + base + offsets, query_positions, mask=active)
    tl.store(slot_mapping_ptr + base + offsets, slots, mask=active)

    draft = (offsets > 0) & active
    sample_offset = row * num_speculative_tokens + offsets - 1
    tl.store(
        sample_indices_ptr + sample_offset,
        base + offsets,
        mask=draft,
    )


def dflash_query_setup_into(
    last_token_ids: tensorplay.Tensor,
    last_positions: tensorplay.Tensor,
    block_tables: tensorplay.Tensor,
    input_ids: tensorplay.Tensor,
    positions: tensorplay.Tensor,
    slot_mapping: tensorplay.Tensor,
    sample_indices: tensorplay.Tensor,
    *,
    mask_token_id: int,
    page_size: int,
    num_speculative_tokens: int,
) -> None:
    """Build graph-stable DFlash bonus/mask query metadata in one launch."""

    batch = last_token_ids.numel()
    query_tokens = num_speculative_tokens + 1
    if (
        last_token_ids.ndim != 1
        or last_positions.shape != last_token_ids.shape
        or block_tables.ndim != 2
        or block_tables.shape[0] != batch
        or input_ids.shape != (batch * query_tokens,)
        or positions.shape != input_ids.shape
        or slot_mapping.shape != input_ids.shape
        or sample_indices.shape != (batch * num_speculative_tokens,)
    ):
        raise ValueError("DFlash query buffer geometry is invalid")
    tensors = (
        last_token_ids,
        last_positions,
        block_tables,
        input_ids,
        positions,
        slot_mapping,
        sample_indices,
    )
    if any(tensor.device != last_token_ids.device for tensor in tensors):
        raise ValueError("DFlash query buffers must share one device")
    if last_token_ids.device.type != "cuda":
        raise ValueError("DFlash query setup requires CUDA")
    if (
        last_token_ids.dtype not in {tensorplay.int32, tensorplay.int64}
        or last_positions.dtype not in {tensorplay.int32, tensorplay.int64}
        or block_tables.dtype not in {tensorplay.int32, tensorplay.int64}
        or input_ids.dtype != last_token_ids.dtype
        or positions.dtype != last_positions.dtype
        or slot_mapping.dtype != tensorplay.int64
        or sample_indices.dtype != tensorplay.int64
    ):
        raise ValueError("DFlash query buffer dtype is invalid")
    if (
        page_size <= 0
        or num_speculative_tokens <= 0
        or num_speculative_tokens > 32
    ):
        raise ValueError("DFlash query limits are invalid")
    block = triton.next_power_of_2(query_tokens)
    _dflash_query_setup_kernel[(batch,)](
        last_token_ids,
        last_positions,
        block_tables,
        block_tables.stride(0),
        input_ids,
        positions,
        slot_mapping,
        sample_indices,
        mask_token_id,
        page_size=page_size,
        num_speculative_tokens=num_speculative_tokens,
        BLOCK=block,
        num_warps=1,
    )


__all__ = ["dflash_query_setup_into"]
