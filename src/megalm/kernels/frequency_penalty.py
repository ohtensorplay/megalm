from __future__ import annotations

import tensorplay
import triton
import triton.language as tl


@triton.jit
def _frequency_presence_kernel(
    scores_ptr,
    history_rows_ptr,
    history_tokens_ptr,
    history_counts_ptr,
    presence_ptr,
    frequency_ptr,
    vocab_size,
    entry_count,
    BLOCK: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < entry_count
    rows = tl.load(history_rows_ptr + offsets, mask=mask, other=0)
    tokens = tl.load(history_tokens_ptr + offsets, mask=mask, other=0)
    counts = tl.load(history_counts_ptr + offsets, mask=mask, other=0)
    presence = tl.load(presence_ptr + rows, mask=mask).to(tl.float32)
    frequency = tl.load(frequency_ptr + rows, mask=mask).to(tl.float32)
    score_offsets = rows * vocab_size + tokens
    values = tl.load(scores_ptr + score_offsets, mask=mask).to(tl.float32)
    penalized = values - (presence + frequency * counts.to(tl.float32))
    tl.store(scores_ptr + score_offsets, penalized, mask=mask)


def apply_frequency_presence_penalty_(
    scores: tensorplay.Tensor,
    history_rows: tensorplay.Tensor,
    history_tokens: tensorplay.Tensor,
    history_counts: tensorplay.Tensor,
    presence: tensorplay.Tensor,
    frequency: tensorplay.Tensor,
) -> None:
    if history_rows.numel() == 0:
        return
    if not (history_rows.shape == history_tokens.shape == history_counts.shape):
        raise ValueError("packed frequency/presence metadata shapes differ")
    block = 256
    _frequency_presence_kernel[(triton.cdiv(history_rows.numel(), block),)](
        scores,
        history_rows,
        history_tokens,
        history_counts,
        presence,
        frequency,
        scores.shape[1],
        history_rows.numel(),
        BLOCK=block,
        num_warps=4,
    )
