"""Bidirectional (non-causal) paged attention backend.

Every query token attends to the full sequence rather than only prior tokens.
This is the text-side counterpart of the vision tower's ``is_causal=False``
path and is selected cold by ``AttentionConfig(mode="non_causal")``.
"""

from __future__ import annotations

import tensorplay

from megalm.kernels.attention.sdpa_base import (
    check_output,
    decode_rows,
    prefill_rows,
)
from megalm.kernels.attention.types import (
    NON_CAUSAL_BACKEND,
    AttentionConfigError,
    _validate_common,
)


class NonCausalAttention:
    backend_id = NON_CAUSAL_BACKEND
    decode_backend_id = NON_CAUSAL_BACKEND

    def prefill(
        self,
        query,
        key_cache,
        value_cache,
        block_tables,
        query_start_loc,
        positions,
        *,
        query_lengths,
        context_lengths,
        page_size,
        output=None,
    ):
        _validate_common(query, key_cache, value_cache, block_tables, page_size)
        if len(query_lengths) != len(context_lengths):
            raise AttentionConfigError("prefill metadata length mismatch")
        if query_start_loc.numel() != len(query_lengths) + 1:
            raise AttentionConfigError("query_start_loc length mismatch")
        return prefill_rows(
            query,
            key_cache,
            value_cache,
            block_tables,
            positions,
            query_lengths,
            context_lengths,
            page_size,
            check_output(query, output),
            causal=False,
        )

    def decode(
        self,
        query,
        key_cache,
        value_cache,
        block_tables,
        sequence_lengths,
        page_size,
        *,
        output=None,
    ):
        _validate_common(query, key_cache, value_cache, block_tables, page_size)
        return decode_rows(
            query,
            key_cache,
            value_cache,
            block_tables,
            sequence_lengths,
            page_size,
            check_output(query, output),
            causal=False,
        )


def build_non_causal_attention() -> NonCausalAttention:
    return NonCausalAttention()


__all__ = ["NonCausalAttention", "build_non_causal_attention"]
