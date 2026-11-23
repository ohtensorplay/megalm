from __future__ import annotations

import tensorplay
import triton
import triton.language as tl

@triton.jit
def _repetition_penalty_kernel(
    scores_ptr,
    history_rows_ptr,
    history_tokens_ptr,
    penalties_ptr,
    vocab_size,
    entry_count,
    BLOCK: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < entry_count
    rows = tl.load(history_rows_ptr + offsets, mask=mask, other=0)
    tokens = tl.load(history_tokens_ptr + offsets, mask=mask, other=0)
    score_offsets = rows * vocab_size + tokens
    values = tl.load(scores_ptr + score_offsets, mask=mask).to(tl.float32)
    penalties = tl.load(penalties_ptr + rows, mask=mask).to(tl.float32)
    adjusted = tl.where(values < 0, values * penalties, values / penalties)
    tl.store(scores_ptr + score_offsets, adjusted, mask=mask)


def apply_repetition_penalty_(
    scores: tensorplay.Tensor,
    history_rows: tensorplay.Tensor,
    history_tokens: tensorplay.Tensor,
    penalties: tensorplay.Tensor,
) -> None:
    if history_rows.numel() == 0:
        return
    if history_rows.shape != history_tokens.shape:
        raise ValueError("packed repetition metadata shapes differ")
    block = 256
    _repetition_penalty_kernel[(triton.cdiv(history_rows.numel(), block),)](
        scores,
        history_rows,
        history_tokens,
        penalties,
        scores.shape[1],
        history_rows.numel(),
        BLOCK=block,
        num_warps=4,
    )


