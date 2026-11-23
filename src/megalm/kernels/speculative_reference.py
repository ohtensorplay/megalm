from __future__ import annotations

import tensorplay


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
    batch = last_token_ids.numel()
    offsets = tensorplay.arange(
        num_speculative_tokens + 1,
        device=last_token_ids.device,
        dtype=last_positions.dtype,
    )
    query_positions = last_positions[:, None] + 1 + offsets
    pages = block_tables.gather(
        1,
        tensorplay.div(query_positions, page_size, rounding_mode="floor").long(),
    )
    input_ids.view(batch, -1).fill_(mask_token_id)
    input_ids.view(batch, -1)[:, 0].copy_(last_token_ids)
    positions.copy_(query_positions.reshape(-1))
    slot_mapping.copy_(
        (
            pages * page_size
            + tensorplay.remainder(query_positions, page_size)
        ).reshape(-1)
    )
    sample_indices.copy_(
        (
            tensorplay.arange(
                batch,
                device=last_token_ids.device,
                dtype=tensorplay.int64,
            )[:, None]
            * (num_speculative_tokens + 1)
            + tensorplay.arange(
                1,
                num_speculative_tokens + 1,
                device=last_token_ids.device,
                dtype=tensorplay.int64,
            )
        ).reshape(-1)
    )


def greedy_chain_verify_into(
    draft_token_ids: tensorplay.Tensor,
    target_token_ids: tensorplay.Tensor,
    output_token_ids: tensorplay.Tensor,
    accepted_draft_counts: tensorplay.Tensor,
) -> None:
    output_token_ids.fill_(-1)
    for row in range(draft_token_ids.shape[0]):
        accepted = 0
        for index in range(draft_token_ids.shape[1]):
            draft_id = draft_token_ids[row, index]
            target_id = target_token_ids[row, index]
            if draft_id != target_id:
                output_token_ids[row, index] = target_id
                break
            output_token_ids[row, index] = draft_id
            accepted += 1
        else:
            output_token_ids[row, -1] = target_token_ids[row, -1]
        accepted_draft_counts[row] = accepted


__all__ = [
    "dflash_query_setup_into",
    "greedy_chain_verify_into",
]
