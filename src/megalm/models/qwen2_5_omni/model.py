from __future__ import annotations

from pathlib import Path
from typing import Sequence

import tensorplay
import tensorplay.nn as nn

from megalm.kernels.bindings import KernelBindings, bind_kernel_plan
from megalm.kernels.ops import KernelPolicy
from megalm.loading import WeightLoadReport
from megalm.models.components import frozen_parameter
from megalm.runtime.batch import ExecutionBatch
from megalm.runtime.cache import PagedKVCache

from megalm.models.qwen2_5_omni.config import Qwen25OmniConfig
from megalm.models.qwen2_5_omni.audio import Qwen25OmniAudioEncoder
from megalm.models.qwen2_5_omni.loading import load_thinker_weights
from megalm.models.qwen2_5_omni.text import OmniTextModel
from megalm.models.qwen2_5_omni.vision import Qwen25OmniVisionEncoder


class Qwen25OmniThinker(nn.Module):
    """Native Thinker with text, visual, and audio input towers."""

    uses_mrope = True

    def __init__(
        self,
        config: Qwen25OmniConfig,
        *,
        device: tensorplay.device | str,
        dtype: tensorplay.dtype,
        kernel_policy: KernelPolicy,
        kernel_bindings: KernelBindings | None = None,
    ) -> None:
        super().__init__()
        self.config = config
        self.device = tensorplay.device(device)
        self.dtype = dtype
        self.kernel_policy = kernel_policy
        if kernel_bindings is None:
            kernel_bindings = bind_kernel_plan(
                kernel_policy,
                device=self.device,
                dtype=dtype,
                model_type="qwen2_5_omni",
            )
        if kernel_bindings.policy != kernel_policy:
            raise ValueError(
                "kernel bindings do not match the Omni kernel policy"
            )
        self.kernel_bindings = kernel_bindings
        self._linear = kernel_bindings.core.linear
        self.model = OmniTextModel(
            config.text_config,
            device=self.device,
            dtype=dtype,
            kernels=kernel_bindings,
        )
        self.visual = Qwen25OmniVisionEncoder(
            config.vision_config,
            device=self.device,
            dtype=dtype,
            kernels=kernel_bindings,
        )
        self.audio_tower = Qwen25OmniAudioEncoder(
            config.audio_config,
            device=self.device,
            dtype=dtype,
        )
        self.lm_head_weight = frozen_parameter(
            (config.vocab_size, config.hidden_size),
            device=self.device,
            dtype=dtype,
        )
        self._last_hidden_states: tensorplay.Tensor | None = None

    @classmethod
    def from_pretrained(
        cls,
        model_path: str | Path,
        *,
        device: tensorplay.device | str,
        dtype: tensorplay.dtype,
        kernel_policy: KernelPolicy,
        kernel_bindings: KernelBindings | None = None,
    ) -> tuple[Qwen25OmniThinker, WeightLoadReport]:
        model = cls(
            Qwen25OmniConfig.from_path(model_path),
            device=device,
            dtype=dtype,
            kernel_policy=kernel_policy,
            kernel_bindings=kernel_bindings,
        )
        report = load_thinker_weights(model, model_path)
        model.eval()
        return model, report

    def forward(
        self,
        batch: ExecutionBatch,
        cache: PagedKVCache,
    ) -> tensorplay.Tensor:
        hidden_states = self.model(batch, cache)
        self._last_hidden_states = hidden_states
        if batch.all_decode:
            return self._linear(hidden_states, self.lm_head_weight)
        indices = batch.sample_indices.clamp_min(0).long()
        logits = self._linear(
            hidden_states.index_select(0, indices),
            self.lm_head_weight,
        )
        return tensorplay.where(
            (batch.sample_indices >= 0)[:, None],
            logits,
            0,
        )

    @property
    def last_hidden_states(self) -> tensorplay.Tensor:
        if self._last_hidden_states is None:
            raise RuntimeError("Qwen2.5-Omni Thinker has not executed")
        return self._last_hidden_states

    @property
    def last_input_embeddings(self) -> tensorplay.Tensor:
        return self.model.last_input_embeddings

    def input_embeddings(self, input_ids: tensorplay.Tensor) -> tensorplay.Tensor:
        return self.model.embed_tokens(input_ids)

    @tensorplay.inference_mode()
    def encode_images(
        self,
        pixel_values: tensorplay.Tensor,
        grid_thw: tensorplay.Tensor,
    ) -> tensorplay.Tensor:
        return self.visual(pixel_values, grid_thw)

    @tensorplay.inference_mode()
    def encode_audios(
        self,
        input_features: tensorplay.Tensor,
        feature_lengths: Sequence[int],
    ) -> tuple[tensorplay.Tensor, tuple[int, ...]]:
        return self.audio_tower(input_features, feature_lengths)

    def configure_moe_workspace(self, max_batched_tokens: int) -> int:
        if max_batched_tokens <= 0:
            raise ValueError("token budget must be positive")
        return 0

    @tensorplay.inference_mode()
    def warmup_prefill_kernels(
        self,
        *,
        max_batched_tokens: int,
    ) -> tuple[int, ...]:
        if max_batched_tokens <= 0:
            raise ValueError("token budget must be positive")
        return ()


__all__ = ["Qwen25OmniThinker"]
