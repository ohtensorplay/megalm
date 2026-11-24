"""Multi-head Latent Attention (MLA) backend.

Only the compressed KV latent ``kv_c`` and the decoupled RoPE key ``k_pe`` are
cached, in one tensor per layer of shape
``[pages, page_size, 1, kv_lora_rank + qk_rope_head_dim]``. The per-head K and V
are reconstructed on the fly with ``W_UK`` / ``W_UV`` (the non-absorbed,
compute-friendly prefill form). Decode uses the same math with one query row;
the absorbed-Q form is an optional optimization left to a future Triton kernel.

The backend keeps the standard paged-attention ABI by treating the latent
arena as ``key_cache``; ``value_cache`` is ignored.
"""

from __future__ import annotations

from dataclasses import dataclass

import tensorplay

from megalm.kernels.attention.sdpa_base import prepare_caches
from megalm.kernels.attention.types import (
    MLA_DECODE_BACKEND,
    MLA_PREFILL_BACKEND,
    AttentionConfigError,
    _gather_pages,
    _sdpa,
    _validate_common,
)


@dataclass(frozen=True, slots=True)
class MLAWeights:
    """Up-projection weights shared by every layer using this backend."""

    w_uk: tensorplay.Tensor  # [kv_lora_rank, heads, qk_nope_head_dim]
    w_uv: tensorplay.Tensor  # [kv_lora_rank, heads, v_head_dim]

    def __post_init__(self) -> None:
        if self.w_uk.ndim != 3 or self.w_uv.ndim != 3:
            raise AttentionConfigError("MLA up-projection weights must be 3-D")
        if self.w_uk.shape[0] != self.w_uv.shape[0]:
            raise AttentionConfigError("MLA latent widths disagree")


class MLACache:
    """Per-layer latent arena with the same page/slot addressing as KV cache."""

    def __init__(
        self,
        num_layers: int,
        num_pages: int,
        page_size: int,
        kv_lora_rank: int,
        qk_rope_head_dim: int,
        *,
        dtype,
        device,
    ) -> None:
        if min(num_layers, num_pages, page_size, kv_lora_rank) <= 0:
            raise AttentionConfigError("MLA cache geometry must be positive")
        self.latent = tensorplay.zeros(
            (num_layers, num_pages, page_size, 1, kv_lora_rank + qk_rope_head_dim),
            dtype=dtype,
            device=device,
        )

    def layer(self, index: int) -> tensorplay.Tensor:
        return self.latent[index]

    @property
    def nbytes(self) -> int:
        return int(self.latent.numel() * self.latent.element_size())


class MLAAttention:
    backend_id = MLA_PREFILL_BACKEND
    decode_backend_id = MLA_DECODE_BACKEND

    def __init__(
        self,
        *,
        kv_lora_rank: int,
        qk_rope_head_dim: int,
        qk_nope_head_dim: int,
        weights: MLAWeights,
    ) -> None:
        if min(kv_lora_rank, qk_rope_head_dim, qk_nope_head_dim) <= 0:
            raise AttentionConfigError("MLA geometry must be positive")
        self.kv_lora_rank = kv_lora_rank
        self.qk_rope_head_dim = qk_rope_head_dim
        self.qk_nope_head_dim = qk_nope_head_dim
        self.weights = weights

    def _expand_latent(
        self, latent: tensorplay.Tensor, num_heads: int
    ) -> tuple[tensorplay.Tensor, tensorplay.Tensor]:
        """Return per-head key [L,H,P+R] and value [L,H,V]."""

        if latent.ndim != 3 or latent.shape[1] != 1:
            raise AttentionConfigError("MLA latent cache must have one kv head")
        kv_c = latent[:, 0, : self.kv_lora_rank]  # [tokens, rank]
        k_pe = latent[:, 0, self.kv_lora_rank :]  # [tokens, rope]
        k_nope = tensorplay.einsum("lr,rhd->lhd", kv_c, self.weights.w_uk)
        value = tensorplay.einsum("lr,rhd->lhd", kv_c, self.weights.w_uv)
        if k_nope.shape[1] != num_heads:
            raise AttentionConfigError("MLA projection head count differs from query")
        k_pe = k_pe.unsqueeze(1).expand(-1, num_heads, -1)
        return tensorplay.cat((k_nope, k_pe), dim=-1), value

    def _output(self, query, output):
        width = self.weights.w_uv.shape[2]
        shape = (query.shape[0], query.shape[1], width)
        if output is None:
            return tensorplay.empty(
                shape, dtype=query.dtype, device=query.device
            )
        if tuple(output.shape) != shape:
            raise AttentionConfigError("MLA attention output must use the value width")
        return output

    def _run(
        self,
        query,
        latent_cache,
        block_tables,
        lengths,
        page_size,
        output,
        *,
        decode: bool,
    ) -> tensorplay.Tensor:
        num_heads = query.shape[1]
        expected_dim = self.qk_nope_head_dim + self.qk_rope_head_dim
        if query.shape[2] != expected_dim:
            raise AttentionConfigError("MLA query width does not match geometry")
        if latent_cache.shape[2] != 1:
            raise AttentionConfigError("MLA latent cache needs one kv head")
        if latent_cache.shape[3] != self.kv_lora_rank + self.qk_rope_head_dim:
            raise AttentionConfigError("MLA latent cache width is invalid")
        scale = expected_dim**-0.5
        if decode:
            starts = [0] * query.shape[0]
            ends = [1] * query.shape[0]
        else:
            start = query.shape[0] - sum(lengths)
            starts = list(range(start, query.shape[0]))
            ends = [index + length for index, length in zip(starts, lengths)]
        page_count = block_tables.shape[1]
        if not isinstance(lengths, tensorplay.Tensor):
            lengths = tensorplay.tensor(
                lengths, dtype=tensorplay.int32, device=query.device
            )
        for sequence_index, (row_start, row_end) in enumerate(zip(starts, ends)):
            seq_len = lengths[sequence_index]
            if not decode:
                seq_len = seq_len + (row_end - row_start)
            pages = block_tables[sequence_index, :page_count]
            latent = _gather_pages(latent_cache, pages.unsqueeze(0), 0, pages.shape[0])
            latent, _ = prepare_caches(latent, latent, query.dtype)
            keys, values = self._expand_latent(latent, num_heads)
            q_len = row_end - row_start
            key_pos = tensorplay.arange(
                keys.shape[0], dtype=lengths.dtype, device=query.device
            )
            valid = key_pos < seq_len
            if decode:
                mask = valid[None, None]
            else:
                query_pos = seq_len - q_len + tensorplay.arange(
                    q_len, dtype=lengths.dtype, device=query.device
                )
                mask = (valid[None, :] & (key_pos[None, :] <= query_pos[:, None]))[
                    None, None
                ]
            # TensorPlay SDPA requires identical key/value widths. MLA's value
            # head is narrower, so pad it and drop the padding afterwards.
            value_width = values.shape[-1]
            if value_width < keys.shape[-1]:
                padding = tensorplay.zeros(
                    (*values.shape[:-1], keys.shape[-1] - value_width),
                    dtype=values.dtype,
                    device=values.device,
                )
                values = tensorplay.cat((values, padding), dim=-1)
            attended = _sdpa(
                query[row_start:row_end], keys, values, mask, scale
            )
            output[row_start:row_end] = attended[..., :value_width]
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
        lengths = tuple(
            ctx + q_len for ctx, q_len in zip(context_lengths, query_lengths)
        )
        return self._run(
            query,
            key_cache,
            block_tables,
            lengths,
            page_size,
            self._output(query, output),
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
        return self._run(
            query,
            key_cache,
            block_tables,
            sequence_lengths,
            page_size,
            self._output(query, output),
            decode=True,
        )


def build_mla_attention(
    *,
    kv_lora_rank: int,
    qk_rope_head_dim: int,
    qk_nope_head_dim: int,
    weights: MLAWeights,
) -> MLAAttention:
    return MLAAttention(
        kv_lora_rank=kv_lora_rank,
        qk_rope_head_dim=qk_rope_head_dim,
        qk_nope_head_dim=qk_nope_head_dim,
        weights=weights,
    )


__all__ = ["MLAAttention", "MLACache", "MLAWeights", "build_mla_attention"]
