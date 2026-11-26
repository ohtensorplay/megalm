"""Reusable scaffolding for checkpoint-native multi-token prediction.

Target models and draft predictors share a small, model-agnostic contract with
the engine: a graph-stable hidden-state buffer, a ``compute_logits`` method and
an optional centroid-masked logit head.  Keeping that contract here means a new
model only supplies its own layers and weight names instead of reimplementing
the buffer/lm-head plumbing for every adaptation.
"""

from __future__ import annotations

from typing import Any

import tensorplay
import tensorplay.nn.functional as F
from tensorplay import nn

from megalm.models.components.parameters import frozen_parameter


class MTPTargetMixin:
    """Hidden-state buffer contract implemented by MTP-enabled target models.

    Mix this into a target model, call ``_init_mtp_state`` during construction
    and set ``self.mtp`` to the predictor.  Models with a non-standard feature
    layout can override :meth:`mtp_buffer_shape`.
    """

    mtp: Any
    dtype: tensorplay.dtype
    device: tensorplay.device

    def _init_mtp_state(
        self,
        *,
        dtype: tensorplay.dtype,
        device: tensorplay.device,
    ) -> None:
        self.register_buffer("_mtp_hidden_buffer", None, persistent=False)

    def mtp_buffer_shape(self, max_tokens: int) -> tuple[int, ...]:
        hidden_size = getattr(self, "config", None)
        hidden_size = getattr(
            getattr(hidden_size, "text_config", hidden_size),
            "hidden_size",
            None,
        )
        if hidden_size is None:
            raise ValueError("MTP target must expose a hidden_size")
        return (max_tokens, int(hidden_size))

    def configure_mtp_hidden_buffer(self, max_tokens: int) -> int:
        if getattr(self, "mtp", None) is None:
            self._mtp_hidden_buffer = None
            return 0
        if max_tokens <= 0:
            raise ValueError("MTP hidden buffer capacity must be positive")
        self._mtp_hidden_buffer = tensorplay.zeros(
            self.mtp_buffer_shape(max_tokens),
            dtype=self.dtype,
            device=self.device,
        )
        return (
            self._mtp_hidden_buffer.numel()
            * self._mtp_hidden_buffer.element_size()
        )

    @property
    def last_hidden_states(self) -> tensorplay.Tensor | None:
        return self._mtp_hidden_buffer

    def _capture_mtp_hidden(self, hidden_states: tensorplay.Tensor) -> None:
        buffer = self._mtp_hidden_buffer
        if getattr(self, "mtp", None) is None or buffer is None:
            return
        if hidden_states.shape[0] > buffer.shape[0]:
            raise ValueError("MTP hidden buffer is smaller than the token count")
        buffer[: hidden_states.shape[0]].copy_(hidden_states)


class MaskedLogitsEmbedder(nn.Module):
    """Centroid-masked sparse logits for ordered-embedding draft heads.

    The vocabulary is partitioned into contiguous centroid buckets (via a
    token-ordering buffer).  Only the buckets selected by the top-k centroid
    projection receive scores; all other logits are filled with a sentinel.
    """

    def __init__(
        self,
        *,
        hidden_size: int,
        vocab_size: int,
        num_centroids: int,
        top_k: int,
        device: tensorplay.device,
        dtype: tensorplay.dtype,
    ) -> None:
        super().__init__()
        if num_centroids <= 0 or top_k <= 0:
            raise ValueError("centroid geometry must be positive")
        if vocab_size % num_centroids:
            raise ValueError("vocab size must be divisible by the centroid count")
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.num_centroids = num_centroids
        self.top_k = top_k
        self.vocab_size_per_centroid = vocab_size // num_centroids
        self.num_selected = top_k * self.vocab_size_per_centroid
        self.centroids = frozen_parameter(
            (num_centroids, hidden_size), device=device, dtype=dtype
        )
        self.register_buffer(
            "token_ordering",
            tensorplay.zeros((vocab_size,), dtype=tensorplay.int64, device=device),
        )

    def _select_and_score(
        self,
        hidden_states: tensorplay.Tensor,
        lm_head_weight: tensorplay.Tensor,
    ) -> tuple[tensorplay.Tensor, tensorplay.Tensor]:
        centroid_logits = F.linear(hidden_states, self.centroids)
        _, top_indices = tensorplay.topk(centroid_logits, self.top_k, dim=-1)
        clusters = self.token_ordering.view(
            self.num_centroids, self.vocab_size_per_centroid
        )
        selected = clusters.index_select(0, top_indices.reshape(-1)).view(
            hidden_states.shape[0], self.num_selected
        )
        embeddings = lm_head_weight.index_select(0, selected.reshape(-1)).view(
            hidden_states.shape[0], self.num_selected, self.hidden_size
        )
        scores = (
            hidden_states.unsqueeze(-2) @ embeddings.transpose(-1, -2)
        ).squeeze(-2)
        return scores, selected

    def compute_logits(
        self,
        hidden_states: tensorplay.Tensor,
        lm_head_weight: tensorplay.Tensor,
    ) -> tensorplay.Tensor:
        scores, selected = self._select_and_score(hidden_states, lm_head_weight)
        fill = float(scores.min().item()) - 1.0
        output = tensorplay.full(
            (hidden_states.shape[0], self.vocab_size),
            fill,
            dtype=hidden_states.dtype,
            device=hidden_states.device,
        )
        return output.scatter(-1, selected, scores)

    def get_top_tokens(
        self,
        hidden_states: tensorplay.Tensor,
        lm_head_weight: tensorplay.Tensor,
    ) -> tensorplay.Tensor:
        scores, selected = self._select_and_score(hidden_states, lm_head_weight)
        best = tensorplay.argmax(scores, dim=-1, keepdim=True)
        return selected.gather(-1, best).squeeze(-1)


class MTPProjector(nn.Module):
    """Shared head projection + logit path for single-stage draft predictors.

    Subclasses supply ``project(hidden)`` and ``layers``; this class owns the
    embedding normaliser, the optional masked logit head and the common
    ``compute_logits`` implementation.
    """

    def __init__(
        self,
        *,
        backbone_embed_tokens: tensorplay.Tensor,
        backbone_hidden_size: int,
        dtype: tensorplay.dtype,
        device: tensorplay.device,
        masked_embedder: MaskedLogitsEmbedder | None = None,
    ) -> None:
        super().__init__()
        self.embed_tokens = backbone_embed_tokens
        self.backbone_hidden_size = backbone_hidden_size
        self.normalizer = tensorplay.tensor(
            backbone_hidden_size**0.5, dtype=dtype, device=device
        )
        self.masked_embedder = masked_embedder

    def embed_input_ids(self, input_ids: tensorplay.Tensor) -> tensorplay.Tensor:
        return F.embedding(input_ids, self.embed_tokens) * self.normalizer

    def compute_logits(
        self,
        hidden_states: tensorplay.Tensor,
        lm_head_weight: tensorplay.Tensor,
    ) -> tensorplay.Tensor:
        if self.masked_embedder is not None:
            return self.masked_embedder.compute_logits(hidden_states, lm_head_weight)
        return F.linear(hidden_states, lm_head_weight)


__all__ = [
    "MaskedLogitsEmbedder",
    "MTPProjector",
    "MTPTargetMixin",
]
