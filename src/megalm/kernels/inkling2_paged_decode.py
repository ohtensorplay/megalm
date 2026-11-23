"""Paged relative attention with a zero-value sink for Inkling-2 decode."""

from __future__ import annotations

import tensorplay
import triton
import triton.language as tl


@triton.jit
def _attention_scores(
    query,
    keys,
    relative,
    tables,
    lengths,
    positions,
    active,
    scores,
    HEADS: tl.constexpr,
    KV_HEADS: tl.constexpr,
    DIM: tl.constexpr,
    PAGE: tl.constexpr,
    TABLE_WIDTH: tl.constexpr,
    CAPACITY: tl.constexpr,
    REL_EXTENT: tl.constexpr,
    WINDOW: tl.constexpr,
    SCALE: tl.constexpr,
    BLOCK_T: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    row = tl.program_id(0)
    head = tl.program_id(1)
    first = tl.program_id(2) * BLOCK_T
    length = tl.load(lengths + row)
    start = tl.maximum(0, length - WINDOW) if WINDOW else 0
    count = length - start
    if tl.load(active + row) & (first < count):
        offsets = first + tl.arange(0, BLOCK_T)
        valid = offsets < count
        logical = start + offsets
        pages = tl.load(tables + row * TABLE_WIDTH + logical // PAGE, valid, 0)
        slots = pages * PAGE + logical % PAGE
        dim = tl.arange(0, BLOCK_D)
        kv_head = head // (HEADS // KV_HEADS)
        query_row = (row * HEADS + head) * DIM
        q = tl.load(query + query_row + dim, dim < DIM, 0).to(tl.float32)
        addresses = (slots[:, None] * KV_HEADS + kv_head) * DIM + dim[None, :]
        k = tl.load(keys + addresses, valid[:, None] & (dim[None, :] < DIM), 0)
        # The unfused path stores both the dot and scaled score in the model
        # dtype before adding the relative bias.
        dtype: tl.constexpr = query.dtype.element_ty
        dot = tl.sum(k.to(tl.float32) * q[None, :], axis=1).to(dtype)
        scaled = (dot.to(tl.float32) * SCALE).to(dtype)
        distance = tl.load(positions + row) - logical
        bias = tl.load(
            relative + (row * HEADS + head) * REL_EXTENT + distance,
            valid & (distance >= 0) & (distance < REL_EXTENT),
            0,
        )
        tl.store(
            scores + (row * HEADS + head) * CAPACITY + offsets,
            scaled.to(tl.float32) + bias.to(tl.float32),
            valid,
        )


@triton.jit
def _attention_probabilities(
    scores,
    sink,
    lengths,
    active,
    probabilities,
    HEADS: tl.constexpr,
    CAPACITY: tl.constexpr,
    WINDOW: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    head = tl.program_id(1)
    if tl.load(active + row):
        length = tl.load(lengths + row)
        count = tl.minimum(length, WINDOW) if WINDOW else length
        offsets = tl.arange(0, BLOCK)
        addresses = (row * HEADS + head) * CAPACITY + offsets
        sink_score = tl.load(sink + row * HEADS + head).to(tl.float32)
        logits = tl.load(scores + addresses, offsets < count, -float("inf"))
        logits = logits.to(tl.float32)
        maximum = tl.maximum(tl.max(logits, axis=0), sink_score)
        exponents = tl.exp(logits - maximum)
        denominator = tl.sum(exponents, axis=0) + tl.exp(sink_score - maximum)
        # Store rounded probabilities, including the sink in the denominator.
        tl.store(
            probabilities + addresses,
            tl.div_rn(exponents, denominator),
            offsets < count,
        )


@triton.jit
def _attention_values(
    values,
    tables,
    lengths,
    active,
    probabilities,
    output,
    HEADS: tl.constexpr,
    KV_HEADS: tl.constexpr,
    DIM: tl.constexpr,
    PAGE: tl.constexpr,
    TABLE_WIDTH: tl.constexpr,
    CAPACITY: tl.constexpr,
    WINDOW: tl.constexpr,
    BLOCK_T: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    row = tl.program_id(0)
    head = tl.program_id(1)
    dims = tl.program_id(2) * BLOCK_D + tl.arange(0, BLOCK_D)
    length = tl.where(tl.load(active + row), tl.load(lengths + row), 0)
    start = tl.maximum(0, length - WINDOW) if WINDOW else 0
    count = length - start
    kv_head = head // (HEADS // KV_HEADS)
    accumulator = tl.full((BLOCK_T, BLOCK_D), 0.0, tl.float32)
    for first in range(0, count, BLOCK_T):
        offsets = first + tl.arange(0, BLOCK_T)
        valid = offsets < count
        logical = start + offsets
        pages = tl.load(tables + row * TABLE_WIDTH + logical // PAGE, valid, 0)
        slots = pages * PAGE + logical % PAGE
        addresses = (slots[:, None] * KV_HEADS + kv_head) * DIM + dims[None, :]
        value = tl.load(values + addresses, valid[:, None] & (dims[None, :] < DIM), 0)
        weight = tl.load(
            probabilities + (row * HEADS + head) * CAPACITY + offsets,
            valid,
            0,
        )
        accumulator += weight[:, None].to(tl.float32) * value.to(tl.float32)
    tl.store(
        output + (row * HEADS + head) * DIM + dims,
        tl.sum(accumulator, axis=0),
        dims < DIM,
    )


# The single CTA softmax uses one block per head. Longer global contexts
# use the model's original attention path until they have a split reduction.
MAX_PAGED_DECODE_TOKENS = 4096


def inkling2_paged_decode(
    query: tensorplay.Tensor,
    keys: tensorplay.Tensor,
    values: tensorplay.Tensor,
    relative: tensorplay.Tensor,
    sink: tensorplay.Tensor,
    tables: tensorplay.Tensor,
    lengths: tensorplay.Tensor,
    positions: tensorplay.Tensor,
    active: tensorplay.Tensor,
    *,
    window: int,
    scale: float,
) -> tensorplay.Tensor:
    """Attend directly over cache pages for one query per sequence."""

    rows, heads, dim = query.shape
    _, page_size, kv_heads, _ = keys.shape
    table_width = tables.shape[1]
    capacity = table_width * page_size
    if window:
        capacity = min(capacity, window)
    scores = tensorplay.empty(
        (rows, heads, capacity), device=query.device, dtype=query.dtype
    )
    weights = tensorplay.empty_like(scores)
    output = tensorplay.empty_like(query)
    token_block = 32
    dim_block = 32
    _attention_scores[(rows, heads, triton.cdiv(capacity, token_block))](
        query,
        keys,
        relative,
        tables,
        lengths,
        positions,
        active,
        scores,
        heads,
        kv_heads,
        dim,
        page_size,
        table_width,
        capacity,
        relative.shape[-1],
        window,
        scale,
        token_block,
        triton.next_power_of_2(dim),
        num_warps=4,
        enable_fp_fusion=False,
    )
    _attention_probabilities[(rows, heads)](
        scores,
        sink,
        lengths,
        active,
        weights,
        heads,
        capacity,
        window,
        triton.next_power_of_2(capacity),
        num_warps=4,
    )
    _attention_values[(rows, heads, triton.cdiv(dim, dim_block))](
        values,
        tables,
        lengths,
        active,
        weights,
        output,
        heads,
        kv_heads,
        dim,
        page_size,
        table_width,
        capacity,
        window,
        token_block,
        dim_block,
        num_warps=4,
        enable_fp_fusion=False,
    )
    return output
