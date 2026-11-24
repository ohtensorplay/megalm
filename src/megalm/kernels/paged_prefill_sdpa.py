from __future__ import annotations

import tensorplay
import tensorplay.nn.functional as F

from megalm.kernels.fallback import dequantize_cache_to_dtype
from megalm.runtime.kv_quant import kv_quant_of


def paged_prefill_attention(
    query: tensorplay.Tensor,
    key_cache: tensorplay.Tensor,
    value_cache: tensorplay.Tensor,
    block_tables: tensorplay.Tensor,
    query_start_loc: tensorplay.Tensor,
    positions: tensorplay.Tensor,
    *,
    query_lengths: tuple[int, ...],
    context_lengths: tuple[int, ...],
    page_size: int,
    output: tensorplay.Tensor | None = None,
) -> tensorplay.Tensor:
    """Reference-stable paged prefill backed by CUDA SDPA."""

    sequence_count = len(query_lengths)
    if not query_lengths or sequence_count != len(context_lengths):
        raise ValueError("packed prefill metadata must contain every sequence")
    if any(length <= 0 for length in query_lengths):
        raise ValueError("paged prefill query lengths must be positive")
    if any(length < 0 for length in context_lengths):
        raise ValueError("paged prefill context lengths must be non-negative")
    if page_size <= 0 or key_cache.shape[1] != page_size:
        raise ValueError("paged prefill page size does not match the KV cache")
    if query_start_loc.dtype != tensorplay.int32:
        raise ValueError("query_start_loc must use int32")
    if query_start_loc.ndim != 1 or query_start_loc.numel() != sequence_count + 1:
        raise ValueError("query_start_loc must delimit every prefill sequence")
    if positions.ndim != 1 or positions.numel() != query.shape[0]:
        raise ValueError("positions must contain every packed query token")
    if block_tables.ndim != 2 or block_tables.shape[0] != sequence_count:
        raise ValueError("block tables must contain every prefill sequence")
    if query.ndim != 3 or key_cache.ndim != 4 or value_cache.shape != key_cache.shape:
        raise ValueError("paged prefill query or KV cache shape is invalid")
    if query.device != key_cache.device or value_cache.device != query.device:
        raise ValueError("paged prefill tensors must share one device")
    quant = kv_quant_of(key_cache)
    if quant is None and (
        query.dtype != key_cache.dtype
        or value_cache.dtype != query.dtype
    ):
        raise ValueError("paged prefill tensors must share one dtype")

    query = query.contiguous()
    total_prefill_tokens = sum(query_lengths)
    output_start = query.shape[0] - total_prefill_tokens
    if output_start < 0:
        raise ValueError("prefill query lengths exceed the packed query")
    if output is None:
        if output_start:
            raise ValueError("mixed decode/prefill requires a shared output buffer")
        output = tensorplay.empty_like(query)
    elif (
        output.shape != query.shape
        or output.device != query.device
        or output.dtype != query.dtype
        or not output.is_contiguous()
    ):
        raise ValueError("paged prefill output must match contiguous query")

    num_query_heads = query.shape[1]
    num_kv_heads = key_cache.shape[2]
    head_dim = query.shape[2]
    if key_cache.shape[3] != head_dim or num_query_heads % num_kv_heads:
        raise ValueError("paged prefill GQA geometry is invalid")
    query_per_kv = num_query_heads // num_kv_heads

    start = output_start
    for sequence_index, (query_length, context_length) in enumerate(
        zip(query_lengths, context_lengths, strict=True)
    ):
        end = start + query_length
        sequence_length = context_length + query_length
        page_count = (sequence_length + page_size - 1) // page_size
        if page_count > block_tables.shape[1]:
            raise ValueError("block table cannot address the prefill sequence")
        page_ids = block_tables[sequence_index, :page_count].long()
        # index_select cannot gather fp8/packed rows; decode the full cache
        # first so the logical head_dim is restored before gathering.
        source_keys = dequantize_cache_to_dtype(
            key_cache, quant, "key", query.dtype
        )
        source_values = dequantize_cache_to_dtype(
            value_cache, quant, "value", query.dtype
        )
        # page_ids are physical *page* indices (the cache's first dimension),
        # so select pages before flattening into slots.
        keys = source_keys.index_select(0, page_ids).reshape(
            -1, num_kv_heads, head_dim
        )[:sequence_length]
        values = source_values.index_select(0, page_ids).reshape(
            -1, num_kv_heads, head_dim
        )[:sequence_length]
        keys = keys.repeat_interleave(query_per_kv, dim=1)
        values = values.repeat_interleave(query_per_kv, dim=1)

        query_states = query[start:end].transpose(0, 1).unsqueeze(0)
        key_states = keys.transpose(0, 1).unsqueeze(0)
        value_states = values.transpose(0, 1).unsqueeze(0)
        query_positions = tensorplay.arange(
            context_length,
            sequence_length,
            device=query.device,
        )
        key_positions = tensorplay.arange(sequence_length, device=query.device)
        causal = (
            key_positions[None, :] <= query_positions[:, None]
        )[None, None]
        attended = F.scaled_dot_product_attention(
            query_states,
            key_states,
            value_states,
            attn_mask=causal,
            dropout_p=0.0,
            scale=head_dim**-0.5,
        )
        output[start:end] = attended.squeeze(0).transpose(0, 1)
        start = end
    return output


__all__ = ["paged_prefill_attention"]
