"""Typed ABI for attention backends.

The signatures here are deliberately identical to the existing
``CoreKernelBindings.paged_prefill_attention`` / ``paged_decode_attention``
callables. A backend is chosen once during DISCOVER/DRAFT and bound as a plain
callable, so the request path performs no backend lookup or capability test.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

import tensorplay

# Backend ids are sealed in ExecutionPlan.kernels and declared in the cold
# capability matrix (kernels/capabilities.py).
CAUSAL_BACKEND = "megalm.attention.causal_paged"
NON_CAUSAL_BACKEND = "megalm.attention.non_causal_sdpa"
SINK_BACKEND = "megalm.attention.sink_sdpa"
SPARSE_BACKEND = "megalm.attention.sparse_index_sdpa"
MLA_PREFILL_BACKEND = "megalm.attention.mla_latent_prefill"
MLA_DECODE_BACKEND = "megalm.attention.mla_latent_decode"

ATTENTION_MODES = ("causal", "non_causal", "sink", "sparse", "mla")


class AttentionConfigError(ValueError):
    """Raised when an attention backend configuration is invalid."""


@dataclass(frozen=True, slots=True)
class AttentionConfig:
    """Cold decisions that select and parameterize an attention backend."""

    mode: str = "causal"
    sink_tokens: int = 0
    sparse_top_k: int = 0
    sparse_block_size: int = 16
    mla_kv_lora_rank: int = 0
    mla_qk_rope_head_dim: int = 0
    mla_qk_nope_head_dim: int = 0

    def __post_init__(self) -> None:
        if self.mode not in ATTENTION_MODES:
            raise AttentionConfigError(f"unknown attention mode {self.mode!r}")
        if self.mode == "sink" and self.sink_tokens <= 0:
            raise AttentionConfigError("sink attention requires sink_tokens > 0")
        if self.mode == "sparse" and self.sparse_top_k <= 0:
            raise AttentionConfigError("sparse attention requires sparse_top_k > 0")
        if self.sparse_block_size <= 0:
            raise AttentionConfigError("sparse block size must be positive")
        if self.mode == "mla":
            if min(
                self.mla_kv_lora_rank,
                self.mla_qk_rope_head_dim,
                self.mla_qk_nope_head_dim,
            ) <= 0:
                raise AttentionConfigError("MLA geometry must be positive")

    @property
    def backend_id(self) -> str:
        return {
            "causal": CAUSAL_BACKEND,
            "non_causal": NON_CAUSAL_BACKEND,
            "sink": SINK_BACKEND,
            "sparse": SPARSE_BACKEND,
            "mla": MLA_PREFILL_BACKEND,
        }[self.mode]

    @property
    def decode_backend_id(self) -> str:
        return MLA_DECODE_BACKEND if self.mode == "mla" else self.backend_id

    @property
    def capabilities(self) -> tuple[str, ...]:
        return {
            "causal": ("causal", "gqa"),
            "non_causal": ("non_causal", "gqa"),
            "sink": ("causal", "gqa", "sink"),
            "sparse": ("causal", "gqa", "sparse"),
            "mla": ("causal", "gqa", "mla"),
        }[self.mode]


class AttentionBackend(Protocol):
    """A bound backend: two callables with the stable paged-attention ABI."""

    backend_id: str
    decode_backend_id: str

    def prefill(
        self,
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
    ) -> tensorplay.Tensor: ...

    def decode(
        self,
        query: tensorplay.Tensor,
        key_cache: tensorplay.Tensor,
        value_cache: tensorplay.Tensor,
        block_tables: tensorplay.Tensor,
        sequence_lengths: tensorplay.Tensor,
        page_size: int,
        *,
        output: tensorplay.Tensor | None = None,
    ) -> tensorplay.Tensor: ...


def _validate_common(
    query: tensorplay.Tensor,
    key_cache: tensorplay.Tensor,
    value_cache: tensorplay.Tensor,
    block_tables: tensorplay.Tensor,
    page_size: int,
) -> None:
    if query.ndim != 3 or key_cache.ndim != 4:
        raise AttentionConfigError("attention query/cache geometry is invalid")
    if key_cache.shape != value_cache.shape:
        raise AttentionConfigError("key and value caches must match")
    if key_cache.shape[1] != page_size or page_size <= 0:
        raise AttentionConfigError("page size does not match the cache")
    if query.shape[2] != key_cache.shape[3]:
        raise AttentionConfigError("query and cache head dimensions differ")
    num_q = query.shape[1]
    num_kv = key_cache.shape[2]
    if num_kv == 0 or num_q % num_kv:
        raise AttentionConfigError("GQA head geometry is invalid")
    if block_tables.ndim != 2:
        raise AttentionConfigError("block tables must be 2-D")


def _gather_pages(
    cache: tensorplay.Tensor,
    block_tables: tensorplay.Tensor,
    sequence_index: int,
    page_count: int,
) -> tensorplay.Tensor:
    """Gather physical pages for one sequence into a flat token-major tensor."""

    page_ids = block_tables[sequence_index, :page_count].long()
    return cache.index_select(0, page_ids).reshape(-1, *cache.shape[2:])


def _repeat_kv(
    tensor: tensorplay.Tensor, query_per_kv: int
) -> tensorplay.Tensor:
    if query_per_kv == 1:
        return tensor
    return tensor.repeat_interleave(query_per_kv, dim=1)


def _sdpa(
    query: tensorplay.Tensor,
    key: tensorplay.Tensor,
    value: tensorplay.Tensor,
    mask: tensorplay.Tensor | None,
    scale: float,
) -> tensorplay.Tensor:
    return tensorplay.nn.functional.scaled_dot_product_attention(
        query.transpose(0, 1).unsqueeze(0),
        key.transpose(0, 1).unsqueeze(0),
        value.transpose(0, 1).unsqueeze(0),
        attn_mask=mask,
        dropout_p=0.0,
        scale=scale,
    ).squeeze(0).transpose(0, 1)


__all__ = [
    "ATTENTION_MODES",
    "CAUSAL_BACKEND",
    "NON_CAUSAL_BACKEND",
    "SINK_BACKEND",
    "SPARSE_BACKEND",
    "MLA_PREFILL_BACKEND",
    "MLA_DECODE_BACKEND",
    "AttentionBackend",
    "AttentionConfig",
    "AttentionConfigError",
    "_gather_pages",
    "_repeat_kv",
    "_sdpa",
    "_validate_common",
]
