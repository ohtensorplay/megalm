from __future__ import annotations

import tensorplay
import triton
import triton.language as tl


@triton.jit
def _greedy_chain_verify_kernel(
    draft_ids_ptr,
    draft_stride: tl.constexpr,
    target_ids_ptr,
    target_stride: tl.constexpr,
    output_ids_ptr,
    output_stride: tl.constexpr,
    accepted_drafts_ptr,
    num_speculative_tokens: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    offsets = tl.arange(0, BLOCK)
    output_mask = offsets < num_speculative_tokens + 1
    tl.store(
        output_ids_ptr + row * output_stride + offsets,
        -1,
        mask=output_mask,
    )

    alive = tl.full((), True, tl.int1)
    accepted = tl.zeros((), tl.int32)
    for index in range(num_speculative_tokens):
        draft_id = tl.load(draft_ids_ptr + row * draft_stride + index)
        target_id = tl.load(target_ids_ptr + row * target_stride + index)
        matches = draft_id == target_id
        accept = alive & matches
        reject = alive & ~matches
        tl.store(
            output_ids_ptr + row * output_stride + index,
            draft_id,
            mask=accept,
        )
        tl.store(
            output_ids_ptr + row * output_stride + index,
            target_id,
            mask=reject,
        )
        accepted += accept
        alive &= matches

    bonus_id = tl.load(
        target_ids_ptr
        + row * target_stride
        + num_speculative_tokens
    )
    tl.store(
        output_ids_ptr
        + row * output_stride
        + num_speculative_tokens,
        bonus_id,
        mask=alive,
    )
    tl.store(accepted_drafts_ptr + row, accepted)


def greedy_chain_verify_into(
    draft_token_ids: tensorplay.Tensor,
    target_token_ids: tensorplay.Tensor,
    output_token_ids: tensorplay.Tensor,
    accepted_draft_counts: tensorplay.Tensor,
) -> None:
    """Verify parallel draft chains and emit the first target correction."""

    if (
        draft_token_ids.ndim != 2
        or target_token_ids.ndim != 2
        or target_token_ids.shape[0] != draft_token_ids.shape[0]
        or target_token_ids.shape[1] != draft_token_ids.shape[1] + 1
        or output_token_ids.shape != target_token_ids.shape
        or accepted_draft_counts.shape != (draft_token_ids.shape[0],)
    ):
        raise ValueError("speculative verify buffer geometry is invalid")
    tensors = (
        draft_token_ids,
        target_token_ids,
        output_token_ids,
        accepted_draft_counts,
    )
    if any(tensor.device != draft_token_ids.device for tensor in tensors):
        raise ValueError("speculative verify buffers must share one device")
    if draft_token_ids.device.type != "cuda":
        raise ValueError("speculative verify requires CUDA")
    if (
        draft_token_ids.dtype not in {tensorplay.int32, tensorplay.int64}
        or target_token_ids.dtype != draft_token_ids.dtype
        or output_token_ids.dtype != draft_token_ids.dtype
        or accepted_draft_counts.dtype != tensorplay.int32
    ):
        raise ValueError("speculative verify buffer dtype is invalid")
    num_speculative_tokens = draft_token_ids.shape[1]
    if num_speculative_tokens <= 0 or num_speculative_tokens > 32:
        raise ValueError("speculative verify token count is invalid")
    block = triton.next_power_of_2(num_speculative_tokens + 1)
    _greedy_chain_verify_kernel[(draft_token_ids.shape[0],)](
        draft_token_ids,
        draft_token_ids.stride(0),
        target_token_ids,
        target_token_ids.stride(0),
        output_token_ids,
        output_token_ids.stride(0),
        accepted_draft_counts,
        num_speculative_tokens=num_speculative_tokens,
        BLOCK=block,
        num_warps=1,
    )


__all__ = ["greedy_chain_verify_into"]
