from __future__ import annotations

import tensorplay
import triton
import triton.language as tl

@triton.jit
def _topk_radix_histogram_kernel(
    scores_ptr,
    topk_rows_ptr,
    prefixes_ptr,
    histogram_ptr,
    vocab_size: tl.constexpr,
    LEVEL: tl.constexpr,
    BLOCK: tl.constexpr,
):
    topk_index = tl.program_id(0)
    chunk_index = tl.program_id(1)
    row = tl.load(topk_rows_ptr + topk_index)
    prefix = tl.load(prefixes_ptr + row)
    token_offsets = chunk_index * BLOCK + tl.arange(0, BLOCK)
    mask = token_offsets < vocab_size
    values = tl.load(
        scores_ptr + row * vocab_size + token_offsets,
        mask=mask,
        other=-float("inf"),
    ).to(tl.float32)
    raw = values.to(tl.int32, bitcast=True)
    raw_unsigned = raw.to(tl.uint32, bitcast=True)
    sign_mask = tl.full((), 0x80000000, tl.uint32)
    all_bits = tl.full((), 0xFFFFFFFF, tl.uint32)
    ordered = raw_unsigned ^ tl.where(
        raw < 0,
        all_bits,
        sign_mask,
    )
    ordered_i64 = ordered.to(tl.int64)
    shift: tl.constexpr = 24 - LEVEL * 8
    if LEVEL == 0:
        matches_prefix = mask
    else:
        matches_prefix = mask & (
            (ordered_i64 >> (shift + 8)) == prefix
        )
    digits = ((ordered_i64 >> shift) & 0xFF).to(tl.int32)
    local_histogram = tl.histogram(
        digits,
        256,
        mask=matches_prefix,
    )
    bins = tl.arange(0, 256)
    tl.atomic_add(
        histogram_ptr + row * 256 + bins,
        local_histogram,
    )


@triton.jit
def _topk_radix_update_kernel(
    topk_rows_ptr,
    top_k_ptr,
    prefixes_ptr,
    remaining_ptr,
    histogram_ptr,
    cutoff_ptr,
    vocab_size: tl.constexpr,
    FINAL: tl.constexpr,
):
    topk_index = tl.program_id(0)
    row = tl.load(topk_rows_ptr + topk_index)
    bins_descending = tl.arange(0, 256)
    counts = tl.load(
        histogram_ptr + row * 256 + (255 - bins_descending)
    )
    cumulative = tl.cumsum(counts, axis=0)
    remaining = tl.load(remaining_ptr + row)
    selected = tl.min(
        tl.where(cumulative >= remaining, bins_descending, 255),
        axis=0,
    )
    higher_count = tl.sum(
        tl.where(bins_descending < selected, counts, 0),
        axis=0,
    )
    digit = 255 - selected
    prefix = tl.load(prefixes_ptr + row) * 256 + digit
    tl.store(prefixes_ptr + row, prefix)
    tl.store(remaining_ptr + row, remaining - higher_count)
    tl.store(
        histogram_ptr + row * 256 + bins_descending,
        0,
    )
    if FINAL:
        prefix_u32 = prefix.to(tl.uint32)
        sign_mask = tl.full((), 0x80000000, tl.uint32)
        all_bits = tl.full((), 0xFFFFFFFF, tl.uint32)
        raw_cutoff = prefix_u32 ^ tl.where(
            (prefix_u32 & sign_mask) != 0,
            sign_mask,
            all_bits,
        )
        cutoff = raw_cutoff.to(tl.int32, bitcast=True).to(
            tl.float32,
            bitcast=True,
        )
        tl.store(cutoff_ptr + row, cutoff)


def topk_cutoffs(
    scores: tensorplay.Tensor,
    topk_rows: tensorplay.Tensor,
    top_k: tensorplay.Tensor,
    *,
    max_top_k: int,
    cutoffs: tensorplay.Tensor | None = None,
    prefixes: tensorplay.Tensor | None = None,
    remaining: tensorplay.Tensor | None = None,
    histogram: tensorplay.Tensor | None = None,
) -> tensorplay.Tensor:
    batch_size = scores.shape[0]
    if cutoffs is None:
        cutoffs = tensorplay.empty(
            batch_size,
            dtype=tensorplay.float32,
            device=scores.device,
        )
    cutoffs.fill_(-tensorplay.inf)
    if topk_rows.numel() == 0:
        return cutoffs
    del max_top_k
    if prefixes is None:
        prefixes = tensorplay.empty(
            batch_size,
            dtype=tensorplay.int64,
            device=scores.device,
        )
    if remaining is None:
        remaining = tensorplay.empty_like(top_k)
    if histogram is None:
        histogram = tensorplay.empty(
            (batch_size, 256),
            dtype=tensorplay.int32,
            device=scores.device,
        )
    prefixes.zero_()
    remaining.copy_(top_k)
    histogram.zero_()
    block = 4096
    grid = (
        topk_rows.numel(),
        triton.cdiv(scores.shape[1], block),
    )
    for level in range(4):
        _topk_radix_histogram_kernel[grid](
            scores,
            topk_rows,
            prefixes,
            histogram,
            vocab_size=scores.shape[1],
            LEVEL=level,
            BLOCK=block,
            num_warps=8,
        )
        _topk_radix_update_kernel[(topk_rows.numel(),)](
            topk_rows,
            top_k,
            prefixes,
            remaining,
            histogram,
            cutoffs,
            vocab_size=scores.shape[1],
            FINAL=level == 3,
            num_warps=4,
        )
    return cutoffs


