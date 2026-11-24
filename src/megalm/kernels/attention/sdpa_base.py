"""Shared SDPA paged-attention helpers for reference attention backends.

These backends are correctness-first typed wrappers: they gather physical pages,
build an explicit attention mask and call TensorPlay SDPA. They keep the exact
``paged_prefill_attention`` / ``paged_decode_attention`` ABI so a cold bind can
swap them without touching any model file.
"""

from __future__ import annotations

import tensorplay

from megalm.kernels.attention.types import (
    AttentionConfigError,
    _gather_pages,
    _repeat_kv,
    _sdpa,
)
from megalm.kernels.fallback import dequantize_cache_to_dtype
from megalm.runtime.kv_quant import kv_quant_of


def prepare_caches(
    key_cache: tensorplay.Tensor,
    value_cache: tensorplay.Tensor,
    query_dtype: tensorplay.dtype,
) -> tuple[tensorplay.Tensor, tensorplay.Tensor]:
    """Return K/V in the query dtype, undoing any cache quantization."""

    quant = kv_quant_of(key_cache)
    if quant is None:
        if key_cache.dtype != query_dtype or value_cache.dtype != query_dtype:
            raise AttentionConfigError("attention caches must share the query dtype")
        return key_cache, value_cache
    return (
        dequantize_cache_to_dtype(key_cache, quant, "key", query_dtype),
        dequantize_cache_to_dtype(value_cache, quant, "value", query_dtype),
    )


def check_output(
    query: tensorplay.Tensor,
    output: tensorplay.Tensor | None,
) -> tensorplay.Tensor:
    if output is None:
        return tensorplay.empty_like(query)
    if (
        output.shape != query.shape
        or output.device != query.device
        or output.dtype != query.dtype
    ):
        raise AttentionConfigError("attention output must match the query")
    return output


def prefill_rows(
    query: tensorplay.Tensor,
    key_cache: tensorplay.Tensor,
    value_cache: tensorplay.Tensor,
    block_tables: tensorplay.Tensor,
    positions: tensorplay.Tensor,
    query_lengths: tuple[int, ...],
    context_lengths: tuple[int, ...],
    page_size: int,
    output: tensorplay.Tensor,
    *,
    causal: bool,
    selected_pages: tensorplay.Tensor | None = None,
) -> tensorplay.Tensor:
    """Fill packed prefill rows. Returns ``output`` for chaining.

    ``selected_pages`` is ``[B, page_count]``; when present, only those physical
    pages participate (sparse attention). Otherwise the sequence block table is
    used in full.
    """

    total_query = sum(query_lengths)
    if total_query > query.shape[0]:
        raise AttentionConfigError("prefill lengths exceed the packed query")
    start = query.shape[0] - total_query
    num_query_heads = query.shape[1]
    num_kv_heads = key_cache.shape[2]
    query_per_kv = num_query_heads // num_kv_heads
    head_dim = query.shape[2]
    scale = head_dim**-0.5
    keys_cache, values_cache = prepare_caches(
        key_cache, value_cache, query.dtype
    )
    for sequence_index, (q_len, ctx_len) in enumerate(
        zip(query_lengths, context_lengths, strict=True)
    ):
        end = start + q_len
        if selected_pages is None:
            page_count = (ctx_len + q_len + page_size - 1) // page_size
            pages = block_tables[sequence_index, :page_count]
        else:
            pages = selected_pages[sequence_index]
        keys = _gather_pages(keys_cache, pages.unsqueeze(0), 0, pages.shape[0])
        values = _gather_pages(values_cache, pages.unsqueeze(0), 0, pages.shape[0])
        key_len = min(keys.shape[0], ctx_len + q_len)
        keys = keys[:key_len]
        values = values[:key_len]
        keys = _repeat_kv(keys, query_per_kv)
        values = _repeat_kv(values, query_per_kv)
        query_rows = query[start:end]
        if causal:
            query_pos = tensorplay.arange(
                ctx_len, ctx_len + q_len, device=query.device
            )
            key_pos = tensorplay.arange(key_len, device=query.device)
            mask = (key_pos[None, :] <= query_pos[:, None])[None, None]
        else:
            mask = None
        output[start:end] = _sdpa(query_rows, keys, values, mask, scale)
        start = end
    return output


def decode_rows(
    query: tensorplay.Tensor,
    key_cache: tensorplay.Tensor,
    value_cache: tensorplay.Tensor,
    block_tables: tensorplay.Tensor,
    sequence_lengths: tensorplay.Tensor,
    page_size: int,
    output: tensorplay.Tensor,
    *,
    causal: bool,
) -> tensorplay.Tensor:
    """Fill all decode rows with paged SDPA (one sequence per row)."""

    num_query_heads = query.shape[1]
    num_kv_heads = key_cache.shape[2]
    query_per_kv = num_query_heads // num_kv_heads
    head_dim = query.shape[2]
    scale = head_dim**-0.5
    keys_cache, values_cache = prepare_caches(
        key_cache, value_cache, query.dtype
    )
    # Use the full block table and a device-side length mask. Reading a length
    # with int() would synchronize the host and break CUDA Graph capture.
    page_count = block_tables.shape[1]
    for row in range(query.shape[0]):
        keys = _gather_pages(keys_cache, block_tables, row, page_count)
        values = _gather_pages(values_cache, block_tables, row, page_count)
        keys = _repeat_kv(keys, query_per_kv)
        values = _repeat_kv(values, query_per_kv)
        key_pos = tensorplay.arange(
            keys.shape[0], dtype=sequence_lengths.dtype, device=query.device
        )
        allowed = key_pos[None, :] < sequence_lengths[row : row + 1].reshape(-1, 1)
        mask = allowed[:, None, None, :]
        output[row : row + 1] = _sdpa(
            query[row : row + 1], keys, values, mask, scale
        )
    return output
