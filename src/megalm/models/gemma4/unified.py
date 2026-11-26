"""Encoder-free multimodal embedders used by Gemma-4 unified checkpoints.

Unified checkpoints have no vision/audio towers: image patches and waveform
frames are projected directly into the language model's embedding space by a
small stack of norms, a dense patch projection and a factorised 2-D position
embedding, followed by the shared multimodal connector.
"""

from __future__ import annotations

import tensorplay
import tensorplay.nn.functional as F
from tensorplay import nn

from megalm.kernels.bindings import KernelBindings
from megalm.models.components.norms import RMSNorm
from megalm.models.components.parameters import frozen_parameter
from megalm.models.gemma4.config import (
    Gemma4AudioConfig,
    Gemma4TextConfig,
    Gemma4VisionConfig,
)
from megalm.models.gemma4.vision import MultimodalEmbedder


class Gemma4UnifiedVisionEmbedder(nn.Module):
    """Direct image-patch projection for unified checkpoints."""

    def __init__(
        self,
        vision: Gemma4VisionConfig,
        text: Gemma4TextConfig,
        *,
        device: tensorplay.device,
        dtype: tensorplay.dtype,
        kernel_bindings: KernelBindings,
    ) -> None:
        super().__init__()
        self.patch_size = vision.patch_size
        self.mm_embed_dim = vision.mm_embed_dim or text.hidden_size
        patch_dim = 3 * vision.patch_size**2
        self.patch_ln1 = RMSNorm(
            patch_dim, vision.rms_norm_eps,
            device=device, dtype=dtype, kernel_bindings=kernel_bindings,
        )
        self.patch_dense = frozen_parameter(
            (self.mm_embed_dim, patch_dim), device=device, dtype=dtype
        )
        self.patch_ln2 = RMSNorm(
            self.mm_embed_dim, vision.rms_norm_eps,
            device=device, dtype=dtype, kernel_bindings=kernel_bindings,
        )
        self.pos_embedding = frozen_parameter(
            (vision.mm_posemb_size or 1024, 2, self.mm_embed_dim),
            device=device, dtype=dtype,
        )
        self.pos_norm = RMSNorm(
            self.mm_embed_dim, vision.rms_norm_eps,
            device=device, dtype=dtype, kernel_bindings=kernel_bindings,
        )
        self.multimodal_embedder = MultimodalEmbedder(
            input_size=self.mm_embed_dim,
            output_size=text.hidden_size,
            eps=text.rms_norm_eps,
            device=device, dtype=dtype, kernel_bindings=kernel_bindings,
        )

    def forward(
        self,
        patches: tensorplay.Tensor,
        positions: tensorplay.Tensor,
    ) -> tensorplay.Tensor:
        values = 2.0 * (patches.float() - 0.5)
        values = values.to(patches.dtype)
        hidden = F.linear(self.patch_ln1(values), self.patch_dense)
        hidden = self.patch_ln2(hidden)
        ids = positions.long().clamp_min(0).clamp_max(self.pos_embedding.shape[0] - 1)
        pos = self.pos_embedding[:, 0].index_select(
            0, ids[:, 0]
        ) + self.pos_embedding[:, 1].index_select(0, ids[:, 1])
        valid = (positions.long() >= 0).all(dim=-1)
        hidden = self.pos_norm(hidden + pos)
        hidden = tensorplay.where(valid[:, None], hidden, tensorplay.zeros_like(hidden))
        return self.multimodal_embedder(hidden)


class Gemma4UnifiedAudioEmbedder(nn.Module):
    """Direct waveform-frame projection for unified checkpoints."""

    def __init__(
        self,
        audio: Gemma4AudioConfig,
        text: Gemma4TextConfig,
        *,
        device: tensorplay.device,
        dtype: tensorplay.dtype,
        kernel_bindings: KernelBindings,
    ) -> None:
        super().__init__()
        self.multimodal_embedder = MultimodalEmbedder(
            input_size=audio.output_proj_dims,
            output_size=text.hidden_size,
            eps=text.rms_norm_eps,
            device=device, dtype=dtype, kernel_bindings=kernel_bindings,
        )

    def forward(self, frames: tensorplay.Tensor) -> tensorplay.Tensor:
        return self.multimodal_embedder(frames)


__all__ = ["Gemma4UnifiedAudioEmbedder", "Gemma4UnifiedVisionEmbedder"]
