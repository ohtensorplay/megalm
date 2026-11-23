"""Decode-time attention over Inkling-2 residual blocks and loop messages."""

from __future__ import annotations

import tensorplay
import triton
import triton.language as tl


@triton.jit
def _score_residual_sources(
    blocks,
    term0,
    term1,
    term2,
    candidate,
    queries,
    values,
    logits,
    WIDTH: tl.constexpr,
    SOURCES: tl.constexpr,
    TOKENS: tl.constexpr,
    QUERIES: tl.constexpr,
    TERMS: tl.constexpr,
    STORE_CANDIDATE: tl.constexpr,
    NORMALIZE_VALUES: tl.constexpr,
    EPS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    token = tl.program_id(0)
    source = tl.program_id(1)
    hidden = tl.arange(0, BLOCK)
    valid = hidden < WIDTH
    token_offset = token * WIDTH + hidden
    if source == SOURCES - 1:
        row = tl.load(term0 + token_offset, valid, 0).to(tl.float32)
        if TERMS > 1:
            row += tl.load(term1 + token_offset, valid, 0).to(tl.float32)
        if TERMS > 2:
            row += tl.load(term2 + token_offset, valid, 0).to(tl.float32)
        if STORE_CANDIDATE:
            # Preserve the reference's round-and-store at a block boundary.
            rounded = row.to(candidate.dtype.element_ty)
            tl.store(candidate + token_offset, rounded, valid)
            row = rounded.to(tl.float32)
    else:
        row = tl.load(
            blocks + source * TOKENS * WIDTH + token_offset, valid, 0
        ).to(tl.float32)
    inverse = tl.rsqrt(tl.sum(row * row, axis=0) / WIDTH + EPS)
    normalized = row * inverse
    tl.store(
        values + source * TOKENS * WIDTH + token_offset,
        normalized if NORMALIZE_VALUES else row,
        valid,
    )
    for index in tl.static_range(QUERIES):
        q = tl.load(queries + index * WIDTH + hidden, valid, 0).to(tl.float32)
        tl.store(
            logits + (token * QUERIES + index) * SOURCES + source,
            tl.sum(q * normalized, axis=0),
        )


@triton.jit
def _mix_residual_sources(
    values,
    logits,
    norm0,
    norm1,
    norm2,
    output,
    WIDTH: tl.constexpr,
    SOURCES: tl.constexpr,
    TOKENS: tl.constexpr,
    QUERIES: tl.constexpr,
    APPLY_NORM: tl.constexpr,
    EPS: tl.constexpr,
    BLOCK: tl.constexpr,
    SOURCE_BLOCK: tl.constexpr,
):
    token = tl.program_id(0)
    query = tl.program_id(1)
    source = tl.arange(0, SOURCE_BLOCK)
    hidden = tl.arange(0, BLOCK)
    valid = hidden < WIDTH
    logits = tl.load(
        logits + (token * QUERIES + query) * SOURCES + source,
        source < SOURCES,
        -float("inf"),
    )
    exponents = tl.exp(logits - tl.max(logits, axis=0))
    inverse = 1.0 / tl.sum(exponents, axis=0)
    mixed = tl.full((BLOCK,), 0.0, tl.float32)
    for index in tl.static_range(SOURCES):
        weight = tl.sum(tl.where(source == index, exponents, 0.0), axis=0) * inverse
        row = tl.load(values + (index * TOKENS + token) * WIDTH + hidden, valid, 0)
        mixed = tl.fma(weight, row, mixed)
    if APPLY_NORM:
        gain = norm0
        if query == 1:
            gain = norm1
        elif query == 2:
            gain = norm2
        inverse_rms = tl.rsqrt(tl.sum(mixed * mixed, axis=0) / WIDTH + EPS)
        mixed = mixed * inverse_rms * tl.load(gain + hidden, valid, 0).to(tl.float32)
    tl.store(output + (query * TOKENS + token) * WIDTH + hidden, mixed, valid)


def depth_residual_mix(
    blocks: tensorplay.Tensor,
    queries: tensorplay.Tensor,
    terms: tuple[tensorplay.Tensor, ...],
    candidate_out: tensorplay.Tensor | None,
    norms: tuple[tensorplay.Tensor, ...] | None,
    *,
    normalize_values: bool,
    eps: float,
) -> tensorplay.Tensor | None:
    """Mix a small decode batch's residual sources, or return None for fallback."""

    device = terms[0].device
    tokens = terms[0].shape[0]
    query_count, width = queries.shape
    sources = blocks.shape[0] + 1
    if (
        device.type != "cuda"
        or not 0 < tokens <= 4
        or not 0 < width <= 4096
        or not 0 < query_count <= (3 if norms is not None else 4)
        or sources > 8
        or len(terms) > 3
        or not blocks.is_contiguous()
        or not queries.is_contiguous()
        or not all(term.is_contiguous() for term in terms)
        or (candidate_out is not None and not candidate_out.is_contiguous())
        or (
            norms is not None
            and not all(
                norm.is_contiguous() and norm.dtype == norms[0].dtype for norm in norms
            )
        )
    ):
        return None

    values = tensorplay.empty(
        (sources, tokens, width), device=device, dtype=tensorplay.float32
    )
    logits = tensorplay.empty(
        (tokens, query_count, sources), device=device, dtype=tensorplay.float32
    )
    output = tensorplay.empty(
        (query_count, tokens, width),
        device=device,
        dtype=norms[0].dtype if norms is not None else tensorplay.float32,
    )
    placeholder = terms[0]
    term_ptrs = terms + (placeholder,) * (3 - len(terms))
    norm_ptrs = norms or ()
    norm_ptrs += (placeholder,) * (3 - len(norm_ptrs))
    block = triton.next_power_of_2(width)
    _score_residual_sources[(tokens, sources)](
        blocks,
        *term_ptrs,
        candidate_out if candidate_out is not None else placeholder,
        queries,
        values,
        logits,
        width,
        sources,
        tokens,
        query_count,
        len(terms),
        candidate_out is not None,
        normalize_values,
        eps,
        block,
        num_warps=8,
        enable_fp_fusion=False,
    )
    _mix_residual_sources[(tokens, query_count)](
        values,
        logits,
        *norm_ptrs,
        output,
        width,
        sources,
        tokens,
        query_count,
        norms is not None,
        eps,
        block,
        triton.next_power_of_2(sources),
        num_warps=8,
        enable_fp_fusion=False,
    )
    return output
