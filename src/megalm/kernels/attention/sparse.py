"""Sparse attention backend: score, select top-k pages, attend.

The indexer keeps a compact per-page side cache of index keys
(``SparseIndexCache``). When no side cache has been written, it falls back to
mean-pooling the KV cache heads, so the backend is usable without any model
change; a model can opt into a learned indexer by attaching a side cache to
the key cache tensor via :func:`attach_index_cache`.

Selected pages are ordered by their position in the sequence. Decode keeps a
fixed number of slots and masks unused slots on the device. The first page
(attention sink) and the last occupied page (local window) are forced.
"""

from __future__ import annotations

from dataclasses import dataclass

import tensorplay

from megalm.kernels.attention.sdpa_base import (
    check_output,
    prepare_caches,
)
from megalm.kernels.attention.sparse_select import SparseTopKSelector
from megalm.kernels.attention.types import (
    SPARSE_BACKEND,
    AttentionConfigError,
    _gather_pages,
    _repeat_kv,
    _sdpa,
    _validate_common,
)

_INDEX_ATTR = "_megalm_sparse_index"


@dataclass(slots=True)
class SparseIndexCache:
    """Per-page index keys used by the sparse indexer.

    Shape matches ``[pages, page_size, index_dim]`` so the same page ids from
    ``block_tables`` address both the KV arena and the side cache.
    """

    keys: tensorplay.Tensor

    def __post_init__(self) -> None:
        if self.keys.ndim != 3:
            raise AttentionConfigError("sparse index cache must be 3-D")


def attach_index_cache(
    key_cache: tensorplay.Tensor, index_cache: SparseIndexCache
) -> None:
    """Bind a side cache to a KV cache tensor without changing kernel ABIs."""

    if index_cache.keys.shape[:2] != key_cache.shape[:2]:
        raise AttentionConfigError("sparse index cache page geometry differs")
    setattr(key_cache, _INDEX_ATTR, index_cache)


def index_cache_of(key_cache: tensorplay.Tensor) -> SparseIndexCache | None:
    return getattr(key_cache, _INDEX_ATTR, None)


class SparseAttention:
    backend_id = SPARSE_BACKEND
    decode_backend_id = SPARSE_BACKEND

    def __init__(self, top_k: int, block_size: int = 16) -> None:
        if top_k <= 0 or block_size <= 0:
            raise AttentionConfigError("sparse top-k and block size must be positive")
        self.top_k = top_k
        self.block_size = block_size
        self._selector = SparseTopKSelector(top_k)
        # Static device buffers for host-derived metadata. Allocated on the
        # warmup pass so CUDA Graph capture never performs a host->device copy.
        self._score_indices: dict[tuple[tuple[int, ...], str], tensorplay.Tensor] = {}
        self._lengths: dict[tuple[tuple[int, ...], str], tensorplay.Tensor] = {}

    def _device_indices(
        self,
        cache: dict[tuple[tuple[int, ...], str], tensorplay.Tensor],
        values: tuple[int, ...],
        device,
        dtype,
    ) -> tensorplay.Tensor:
        key = (values, str(device))
        tensor = cache.get(key)
        if tensor is None:
            tensor = tensorplay.tensor(values, dtype=dtype, device=device)
            cache[key] = tensor
        return tensor

    def _index_keys(
        self, key_cache: tensorplay.Tensor, query: tensorplay.Tensor
    ) -> tensorplay.Tensor:
        attached = index_cache_of(key_cache)
        if attached is not None:
            return attached.keys.to(query.dtype)
        return key_cache.to(query.dtype).mean(dim=2)

    def _page_scores(
        self,
        query: tensorplay.Tensor,
        index_keys: tensorplay.Tensor,
        physical: tensorplay.Tensor,
        page_size: int,
    ) -> tensorplay.Tensor:
        batch, pages = physical.shape
        gathered = index_keys.index_select(0, physical.reshape(-1))
        if gathered.ndim == 4:
            gathered = gathered.mean(dim=2)
        return tensorplay.bmm(
            gathered.reshape(batch, pages * page_size, index_keys.shape[-1]),
            query.mean(dim=1).unsqueeze(-1),
        ).reshape(batch, pages, page_size)

    def _run_prefill(
        self,
        query,
        key_cache,
        value_cache,
        block_tables,
        query_lengths,
        context_lengths,
        page_size,
        output,
    ) -> tensorplay.Tensor:
        num_heads = query.shape[1]
        query_per_kv = num_heads // key_cache.shape[2]
        scale = query.shape[2] ** -0.5
        keys_cache, values_cache = prepare_caches(
            key_cache, value_cache, query.dtype
        )
        index_keys = self._index_keys(key_cache, query)

        start = query.shape[0] - sum(query_lengths)
        starts = range(start, query.shape[0])
        ends = [index + length for index, length in zip(starts, query_lengths)]
        score_indices = self._device_indices(
            self._score_indices,
            tuple(end - 1 for end in ends),
            query.device,
            tensorplay.int64,
        )
        physical = block_tables.index_select(0, score_indices).long()
        scores = self._page_scores(
            query.index_select(0, score_indices), index_keys, physical, page_size
        )
        selected = self._selector(scores.amax(dim=2), physical)

        lengths = context_lengths
        if not isinstance(lengths, tensorplay.Tensor):
            lengths = self._device_indices(
                self._lengths,
                tuple(int(value) for value in lengths),
                query.device,
                tensorplay.int32,
            )
        for sequence_index, (row_start, row_end) in enumerate(zip(starts, ends)):
            seq_len = lengths[sequence_index] + (row_end - row_start)
            physical = selected[sequence_index]
            keys = _gather_pages(
                keys_cache, physical.unsqueeze(0), 0, physical.shape[0]
            )
            values = _gather_pages(
                values_cache, physical.unsqueeze(0), 0, physical.shape[0]
            )
            keys = _repeat_kv(keys, query_per_kv)
            values = _repeat_kv(values, query_per_kv)
            kept_len = keys.shape[0]
            key_pos = tensorplay.arange(
                kept_len, dtype=lengths.dtype, device=query.device
            )
            valid = key_pos < seq_len
            query_pos = seq_len - (row_end - row_start) + tensorplay.arange(
                row_end - row_start, dtype=lengths.dtype, device=query.device
            )
            mask = (valid[None, :] & (key_pos[None, :] <= query_pos[:, None]))[
                None, None
            ]
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
        return self._run_prefill(
            query,
            key_cache,
            value_cache,
            block_tables,
            query_lengths,
            context_lengths,
            page_size,
            check_output(query, output),
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
        output = check_output(query, output)
        if not isinstance(sequence_lengths, tensorplay.Tensor):
            sequence_lengths = self._device_indices(
                self._lengths, tuple(sequence_lengths), query.device, tensorplay.int32
            )
        batch, page_count = block_tables.shape
        if query.shape[0] != batch or sequence_lengths.shape != (batch,):
            raise AttentionConfigError("sparse decode requires a length and block table per query row")
        if batch == 0 or page_count == 0:
            return output.zero_()

        keys_cache, values_cache = prepare_caches(key_cache, value_cache, query.dtype)
        query_per_kv = query.shape[1] // key_cache.shape[2]
        scale = query.shape[2] ** -0.5
        page_slots = tensorplay.arange(page_count, device=query.device)
        token_offsets = tensorplay.arange(page_size, device=query.device)
        token_positions = page_slots[:, None] * page_size + token_offsets[None, :]
        lengths = sequence_lengths.reshape(batch, 1, 1)
        present = (token_positions[:, 0] < lengths.reshape(batch, 1)) & (block_tables >= 0)
        allowed = present.unsqueeze(-1) & (token_positions < lengths)
        pages = block_tables.masked_fill(~present, 0).long()

        capacity = min(self.top_k + 2, page_count)
        if capacity < page_count:
            tail = ((sequence_lengths - 1) // page_size).clamp(0, page_count - 1).long()
            interior = present & (page_slots > 0) & (page_slots < tail.unsqueeze(-1))
            attached = index_cache_of(key_cache)
            index_keys = keys_cache if attached is None else attached.keys.to(query.dtype)
            scores = self._page_scores(query, index_keys, pages, page_size)
            per_page = scores.masked_fill(~allowed, float("-inf")).amax(dim=-1)
            per_page = per_page.clamp_min(tensorplay.finfo(per_page.dtype).min)
            top = tensorplay.topk(
                per_page.masked_fill(~interior, float("-inf")), self.top_k, sorted=False
            ).indices
            # A row with fewer eligible pages may have spare top-k slots.
            # Give those slots a sentinel so duplicates and padding can be masked.
            top = top.masked_fill(~interior.gather(1, top), page_count)
            candidates = tensorplay.cat((top, page_slots[:1].expand(batch, 1), tail[:, None]), dim=1)

        for row in range(batch):
            row_pages, row_allowed = pages[row], allowed[row]
            if capacity < page_count:
                selected, _ = tensorplay.unique_static(
                    candidates[row], size=capacity, fill_value=page_count
                )
                valid = selected < page_count
                selected = selected.masked_fill(~valid, 0)
                row_pages = row_pages.index_select(0, selected)
                row_allowed = row_allowed.index_select(0, selected) & valid[:, None]

            row_allowed = row_allowed.reshape(-1)
            invalid = ~row_allowed[:, None, None]
            keys = _gather_pages(keys_cache, row_pages.unsqueeze(0), 0, row_pages.shape[0])
            values = _gather_pages(values_cache, row_pages.unsqueeze(0), 0, row_pages.shape[0])
            keys = _repeat_kv(keys.masked_fill(invalid, 0), query_per_kv)
            values = _repeat_kv(values.masked_fill(invalid, 0), query_per_kv)
            output[row : row + 1] = _sdpa(
                query[row : row + 1], keys, values, row_allowed[None, None, None, :], scale
            )
        return output


def build_sparse_attention(
    top_k: int, block_size: int = 16
) -> SparseAttention:
    return SparseAttention(top_k, block_size)


__all__ = [
    "SparseAttention",
    "SparseIndexCache",
    "attach_index_cache",
    "build_sparse_attention",
    "index_cache_of",
]
