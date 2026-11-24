"""Attention-sink backend (StreamingLLM style).

A sink is a learnable per-head scalar logit added to the softmax denominator;
no sink KV is stored. SDPA cannot add a bias to the softmax directly, so this
backend appends one zero-valued key/value row and places the sink logit in the
last column of a float attention mask. The extra key contributes no value,
only the bias, which is exactly the sink semantics.
"""

from __future__ import annotations

import tensorplay

from megalm.kernels.attention.sdpa_base import (
    check_output,
    prepare_caches,
)
from megalm.kernels.attention.types import (
    SINK_BACKEND,
    AttentionConfigError,
    _gather_pages,
    _repeat_kv,
    _sdpa,
    _validate_common,
)


def _sink_mask(
    base_mask: tensorplay.Tensor | None,
    *,
    query_length: int,
    key_length: int,
    sink_logits: tensorplay.Tensor,
) -> tensorplay.Tensor:
    """Extend a causal mask by one sink column carrying the per-head bias."""

    heads = sink_logits.shape[0]
    shape = (1, heads, query_length, key_length)
    if base_mask is None:
        base = tensorplay.zeros(
            shape, dtype=tensorplay.float32, device=sink_logits.device
        )
    else:
        condition = base_mask.expand(shape)
        base = tensorplay.where(
            condition,
            tensorplay.zeros(shape, dtype=tensorplay.float32, device=condition.device),
            tensorplay.full(shape, float("-inf"), dtype=tensorplay.float32, device=condition.device),
        )
    sink_column = sink_logits.reshape(1, heads, 1, 1).expand(
        1, heads, query_length, 1
    )
    return tensorplay.cat((base, sink_column), dim=-1)


class SinkAttention:
    backend_id = SINK_BACKEND
    decode_backend_id = SINK_BACKEND

    def __init__(self, sink_logits: tuple[float, ...]) -> None:
        if not sink_logits:
            raise AttentionConfigError("sink attention needs per-head sink logits")
        self.sink_logits = tuple(float(value) for value in sink_logits)
        self._logit_cache: dict[tuple[int, str], tensorplay.Tensor] = {}

    def _logits(self, num_heads: int, device) -> tensorplay.Tensor:
        key = (num_heads, str(device))
        cached = self._logit_cache.get(key)
        if cached is not None:
            return cached
        if len(self.sink_logits) == 1:
            values = self.sink_logits * num_heads
        elif len(self.sink_logits) == num_heads:
            values = self.sink_logits
        else:
            raise AttentionConfigError(
                "sink logits must be one value or one per query head"
            )
        tensor = tensorplay.tensor(
            values, dtype=tensorplay.float32, device=device
        )
        self._logit_cache[key] = tensor
        return tensor

    def _run(
        self,
        query,
        key_cache,
        value_cache,
        block_tables,
        query_lengths,
        context_lengths,
        page_size,
        output,
        *,
        decode: bool,
    ) -> tensorplay.Tensor:
        num_heads = query.shape[1]
        head_dim = query.shape[2]
        scale = head_dim**-0.5
        query_per_kv = num_heads // key_cache.shape[2]
        sink = self._logits(num_heads, query.device)
        keys_cache, values_cache = prepare_caches(
            key_cache, value_cache, query.dtype
        )
        if decode:
            starts = [0] * query.shape[0]
            ends = [1] * query.shape[0]
        else:
            total = sum(query_lengths)
            start = query.shape[0] - total
            starts = list(range(start, query.shape[0]))
            ends = [index + length for index, length in zip(starts, query_lengths)]

        page_count = block_tables.shape[1]
        lengths = context_lengths
        if not isinstance(lengths, tensorplay.Tensor):
            lengths = tensorplay.tensor(
                lengths, dtype=tensorplay.int32, device=query.device
            )
        for sequence_index, (row_start, row_end) in enumerate(zip(starts, ends)):
            pages = block_tables[sequence_index, :page_count]
            keys = _gather_pages(keys_cache, pages.unsqueeze(0), 0, pages.shape[0])
            values = _gather_pages(
                values_cache, pages.unsqueeze(0), 0, pages.shape[0]
            )
            keys = _repeat_kv(keys, query_per_kv)
            values = _repeat_kv(values, query_per_kv)
            key_count = keys.shape[0]
            # Append a zero key/value row so the sink column has a real slot.
            padding = tensorplay.zeros(
                (1, keys.shape[1], keys.shape[2]),
                dtype=keys.dtype,
                device=keys.device,
            )
            keys = tensorplay.cat((keys, padding), dim=0)
            values = tensorplay.cat((values, padding), dim=0)
            q_len = row_end - row_start
            key_pos = tensorplay.arange(
                key_count, dtype=lengths.dtype, device=query.device
            )
            seq_len = lengths[sequence_index]
            if not decode:
                seq_len = seq_len + (row_end - row_start)
            valid = key_pos < seq_len
            if decode:
                allowed = valid
            else:
                query_pos = seq_len - q_len + tensorplay.arange(
                    q_len, dtype=lengths.dtype, device=query.device
                )
                allowed = valid[None, :] & (key_pos[None, :] <= query_pos[:, None])
            mask = allowed[None, None] if decode else allowed[None, None]
            mask = _sink_mask(
                mask,
                query_length=q_len,
                key_length=key_count,
                sink_logits=sink,
            )
            output[row_start:row_end] = _sdpa(
                query[row_start:row_end], keys, values, mask, scale
            )
        return output

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
        return self._run(
            query,
            key_cache,
            value_cache,
            block_tables,
            query_lengths,
            context_lengths,
            page_size,
            check_output(query, output),
            decode=False,
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
        return self._run(
            query,
            key_cache,
            value_cache,
            block_tables,
            (1,) * query.shape[0],
            sequence_lengths,
            page_size,
            check_output(query, output),
            decode=True,
        )


def build_sink_attention(sink_logits: tuple[float, ...]) -> SinkAttention:
    return SinkAttention(sink_logits)


__all__ = ["SinkAttention", "build_sink_attention"]
