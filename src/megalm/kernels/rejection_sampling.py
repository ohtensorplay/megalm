from __future__ import annotations

import tensorplay
import triton
import triton.language as tl

@triton.jit
def _filtered_rejection_sampling_kernel(
    scores_ptr,
    stochastic_rows_ptr,
    top_k_ptr,
    topk_cutoff_ptr,
    top_p_ptr,
    min_p_ptr,
    seeds_ptr,
    counters_ptr,
    output_ptr,
    valid_ptr,
    vocab_size: tl.constexpr,
    BLOCK: tl.constexpr,
    MAX_ROUNDS: tl.constexpr,
):
    stochastic_index = tl.program_id(0)
    row = tl.load(stochastic_rows_ptr + stochastic_index)
    top_k = tl.load(top_k_ptr + row)
    topk_cutoff = tl.load(topk_cutoff_ptr + row).to(tl.float32)
    top_p = tl.load(top_p_ptr + row).to(tl.float32)
    min_p = tl.load(min_p_ptr + row).to(tl.float32)
    seed = tl.load(seeds_ptr + row).to(tl.uint32)
    counter = tl.load(counters_ptr + row).to(tl.uint32)
    offsets = tl.arange(0, BLOCK)

    maximum = tl.full((), -float("inf"), tl.float32)
    token_start = 0
    while token_start < vocab_size:
        token_offsets = token_start + offsets
        mask = token_offsets < vocab_size
        values = tl.load(
            scores_ptr + row * vocab_size + token_offsets,
            mask=mask,
            other=-float("inf"),
        ).to(tl.float32)
        maximum = tl.maximum(maximum, tl.max(values, axis=0))
        token_start += BLOCK

    denominator = tl.zeros((), tl.float32)
    proposal_mass = tl.zeros((), tl.float32)
    last_qualified = tl.zeros((), tl.int32)
    token_start = 0
    while token_start < vocab_size:
        token_offsets = token_start + offsets
        mask = token_offsets < vocab_size
        values = tl.load(
            scores_ptr + row * vocab_size + token_offsets,
            mask=mask,
            other=-float("inf"),
        ).to(tl.float32)
        weights = tl.exp(values - maximum)
        weights = tl.where(mask, weights, 0.0)
        qualified = (
            mask
            & (weights >= min_p)
            & ((top_k <= 0) | (values >= topk_cutoff))
        )
        denominator += tl.sum(weights, axis=0)
        proposal_mass += tl.sum(tl.where(qualified, weights, 0.0), axis=0)
        last_qualified = tl.maximum(
            last_qualified,
            tl.max(
                tl.where(qualified, token_offsets, 0),
                axis=0,
            ),
        )
        token_start += BLOCK

    target_mass = top_p * denominator
    low = tl.zeros((), tl.float32)
    high = tl.full((), 1.0, tl.float32)
    q = proposal_mass
    sampled_id = last_qualified
    accepted = tl.zeros((), tl.int1)
    round_index = tl.zeros((), tl.int32)

    while (round_index < MAX_ROUNDS) & (~accepted):
        random_offset = (
            counter
            + round_index.to(tl.uint32) * 0x27D4EB2D
        )
        uniform = tl.rand(seed, random_offset)
        threshold = uniform * q
        aggregate = tl.zeros((), tl.float32)
        current_sample = tl.full((), vocab_size, tl.int32)
        current_last = last_qualified
        token_start = 0
        while token_start < vocab_size:
            token_offsets = token_start + offsets
            mask = token_offsets < vocab_size
            values = tl.load(
                scores_ptr + row * vocab_size + token_offsets,
                mask=mask,
                other=-float("inf"),
            ).to(tl.float32)
            weights = tl.exp(values - maximum)
            qualified = (
                mask
                & (weights >= min_p)
                & ((top_k <= 0) | (values >= topk_cutoff))
                & (weights > low)
            )
            proposal = tl.where(qualified, weights, 0.0)
            prefix = tl.cumsum(proposal, axis=0)
            crosses = (
                qualified
                & (current_sample == vocab_size)
                & (aggregate + prefix >= threshold)
            )
            candidate = tl.min(
                tl.where(crosses, token_offsets, vocab_size),
                axis=0,
            )
            current_sample = tl.minimum(current_sample, candidate)
            current_last = tl.maximum(
                current_last,
                tl.max(
                    tl.where(qualified, token_offsets, 0),
                    axis=0,
                ),
            )
            aggregate += tl.sum(proposal, axis=0)
            token_start += BLOCK
        current_sample = tl.where(
            current_sample < vocab_size,
            current_sample,
            current_last,
        )
        sampled_id = current_sample
        pivot_score = tl.load(
            scores_ptr + row * vocab_size + sampled_id
        ).to(tl.float32)
        pivot = tl.exp(pivot_score - maximum)

        rank_filter = top_p < 1.0
        if rank_filter:
            midpoint = (pivot + high) * 0.5
            mass_above_pivot = tl.zeros((), tl.float32)
            mass_above_midpoint = tl.zeros((), tl.float32)
            token_start = 0
            while token_start < vocab_size:
                token_offsets = token_start + offsets
                mask = token_offsets < vocab_size
                values = tl.load(
                    scores_ptr + row * vocab_size + token_offsets,
                    mask=mask,
                    other=-float("inf"),
                ).to(tl.float32)
                weights = tl.exp(values - maximum)
                above_pivot = mask & (weights > pivot)
                above_midpoint = mask & (weights > midpoint)
                mass_above_pivot += tl.sum(
                    tl.where(above_pivot, weights, 0.0),
                    axis=0,
                )
                mass_above_midpoint += tl.sum(
                    tl.where(above_midpoint, weights, 0.0),
                    axis=0,
                )
                token_start += BLOCK

            pivot_accepted = (
                mass_above_pivot < target_mass
            )
            midpoint_accepted = (
                mass_above_midpoint < target_mass
            )
            use_midpoint_bound = (~pivot_accepted) & midpoint_accepted
            accepted = pivot_accepted
            low = tl.where(
                use_midpoint_bound,
                pivot,
                tl.where(pivot_accepted, low, midpoint),
            )
            high = tl.where(use_midpoint_bound, midpoint, high)
            q = tl.where(
                use_midpoint_bound,
                mass_above_pivot,
                tl.where(pivot_accepted, q, mass_above_midpoint),
            )
        else:
            accepted = tl.full((), True, tl.int1)
        round_index += 1

    tl.store(output_ptr + row, tl.where(accepted, sampled_id, -1))
    tl.store(valid_ptr + row, accepted)


def filtered_rejection_sample(
    scores: tensorplay.Tensor,
    stochastic_rows: tensorplay.Tensor,
    top_k: tensorplay.Tensor,
    topk_cutoff: tensorplay.Tensor,
    top_p: tensorplay.Tensor,
    min_p: tensorplay.Tensor,
    seeds: tensorplay.Tensor,
    counters: tensorplay.Tensor,
    output: tensorplay.Tensor,
    valid: tensorplay.Tensor | None = None,
) -> tensorplay.Tensor:
    if stochastic_rows.numel() == 0:
        return tensorplay.ones(
            scores.shape[0],
            dtype=tensorplay.bool,
            device=scores.device,
        )
    if valid is None:
        valid = tensorplay.empty(
            scores.shape[0],
            dtype=tensorplay.bool,
            device=scores.device,
        )
    valid.fill_(True)
    _filtered_rejection_sampling_kernel[(stochastic_rows.numel(),)](
        scores,
        stochastic_rows,
        top_k,
        topk_cutoff,
        top_p,
        min_p,
        seeds,
        counters,
        output,
        valid,
        vocab_size=scores.shape[1],
        BLOCK=8192,
        MAX_ROUNDS=64,
        num_warps=8,
    )
    return valid


