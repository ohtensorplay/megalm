"""Cold-selected attention backends with a stable typed ABI.

Backends are chosen during DISCOVER/DRAFT and bound as plain callables. The
request path never queries this module; it only calls the bound callable.
"""

from __future__ import annotations

from typing import Any

from megalm.kernels.attention.mla import (
    MLAAttention,
    MLACache,
    MLAWeights,
    build_mla_attention,
)
from megalm.kernels.attention.non_causal import (
    NonCausalAttention,
    build_non_causal_attention,
)
from megalm.kernels.attention.sink import SinkAttention, build_sink_attention
from megalm.kernels.attention.sparse import (
    SparseAttention,
    SparseIndexCache,
    attach_index_cache,
    build_sparse_attention,
    index_cache_of,
)
from megalm.kernels.attention.types import (
    ATTENTION_MODES,
    CAUSAL_BACKEND,
    MLA_DECODE_BACKEND,
    MLA_PREFILL_BACKEND,
    NON_CAUSAL_BACKEND,
    SINK_BACKEND,
    SPARSE_BACKEND,
    AttentionBackend,
    AttentionConfig,
    AttentionConfigError,
)


class CausalAttention:
    """Adapter around the existing causal paged kernels."""

    def __init__(self, prefill: Any, decode: Any) -> None:
        if prefill is None or decode is None:
            raise AttentionConfigError("causal attention needs both kernels")
        self.prefill = prefill
        self.decode = decode
        self.backend_id = CAUSAL_BACKEND
        self.decode_backend_id = CAUSAL_BACKEND


def build_attention_backend(
    config: AttentionConfig,
    *,
    causal_prefill: Any = None,
    causal_decode: Any = None,
    sink_logits: tuple[float, ...] = (),
    mla_weights: Any = None,
) -> AttentionBackend:
    """Build the backend selected by ``config`` (cold path only)."""

    if config.mode == "causal":
        return CausalAttention(causal_prefill, causal_decode)
    if config.mode == "non_causal":
        return build_non_causal_attention()
    if config.mode == "sink":
        return build_sink_attention(sink_logits)
    if config.mode == "sparse":
        return build_sparse_attention(
            config.sparse_top_k, config.sparse_block_size
        )
    if config.mode == "mla":
        if mla_weights is None:
            raise AttentionConfigError("MLA attention requires up-projection weights")
        return build_mla_attention(
            kv_lora_rank=config.mla_kv_lora_rank,
            qk_rope_head_dim=config.mla_qk_rope_head_dim,
            qk_nope_head_dim=config.mla_qk_nope_head_dim,
            weights=mla_weights,
        )
    raise AttentionConfigError(f"unsupported attention mode {config.mode!r}")


__all__ = [
    "ATTENTION_MODES",
    "CAUSAL_BACKEND",
    "MLA_DECODE_BACKEND",
    "MLA_PREFILL_BACKEND",
    "NON_CAUSAL_BACKEND",
    "SINK_BACKEND",
    "SPARSE_BACKEND",
    "AttentionBackend",
    "AttentionConfig",
    "AttentionConfigError",
    "CausalAttention",
    "MLAAttention",
    "MLACache",
    "MLAWeights",
    "NonCausalAttention",
    "SinkAttention",
    "SparseAttention",
    "SparseIndexCache",
    "attach_index_cache",
    "build_attention_backend",
    "index_cache_of",
]
