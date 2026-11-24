"""Nanbeige 4.2 native execution on the TensorPlay hot path.

The released Nanbeige4.2-3B checkpoint is a standard dense RoPE/GQA decoder
whose 22 physical layers are replayed ``num_loops`` (=2) times with shared
weights.  It therefore reuses the generic Llama-family kernels and loader
unchanged; only the forward schedule differs:

* every loop visits the same physical layers with a fresh residual stream;
* the ``i``-th loop writes KV into cache layers ``[i*L, (i+1)*L)`` so the two
  passes never overwrite each other;
* a final RMSNorm is folded into the residual stream at each loop boundary
  (matching ``skip_loop_final_norm=false`` in the reference implementation).
"""

from __future__ import annotations

from pathlib import Path

import tensorplay
import tensorplay.nn.functional as F

from megalm.kernels.bindings import KernelBindings
from megalm.kernels.ops import KernelPolicy
from megalm.loading import WeightLoadReport
from megalm.models.generic_causal_lm import GenericCausalLM
from megalm.models.nanbeige.config import NanbeigeConfig
from megalm.runtime.batch import ExecutionBatch
from megalm.runtime.cache import PagedKVCache


class NanbeigeForCausalLM(GenericCausalLM):
    """Generic dense decoder with a full-stack replay schedule."""

    def __init__(
        self,
        config: NanbeigeConfig,
        *,
        device: tensorplay.device | str,
        dtype: tensorplay.dtype,
        kernel_policy: KernelPolicy,
        kernel_bindings: KernelBindings | None = None,
    ) -> None:
        if config.num_loops < 1:
            raise ValueError("Nanbeige requires at least one loop")
        # Physical layer weights are shared across loops, but every visit owns
        # a distinct KV cache layer (22 x 2 = 44 for Nanbeige4.2-3B).
        self.num_kv_cache_layers = config.num_hidden_layers * config.num_loops
        super().__init__(
            config,
            device=device,
            dtype=dtype,
            kernel_policy=kernel_policy,
            kernel_bindings=kernel_bindings,
        )

    @classmethod
    def from_pretrained(
        cls,
        model_path: str | Path,
        *,
        device: tensorplay.device | str = "cuda",
        dtype: tensorplay.dtype = tensorplay.bfloat16,
        kernel_policy: KernelPolicy,
        kernel_bindings: KernelBindings | None = None,
    ) -> tuple[NanbeigeForCausalLM, WeightLoadReport]:
        config = NanbeigeConfig.from_path(model_path)
        model = cls(
            config,
            device=device,
            dtype=dtype,
            kernel_policy=kernel_policy,
            kernel_bindings=kernel_bindings,
        )
        report = model.load_weights(model_path)
        model.eval()
        return model, report

    def lm_head(self) -> tensorplay.Tensor:
        """Exposed for the engine's shared logits helpers."""

        head = (
            self.model.embed_tokens.weight
            if self.config.tie_word_embeddings
            else self.lm_head_weight
        )
        assert head is not None
        return head

    def forward(
        self,
        batch: ExecutionBatch,
        cache: PagedKVCache,
    ) -> tensorplay.Tensor:
        config = self.config
        model = self.model
        hidden_states = model.embed_tokens(batch.input_ids)
        if config.embedding_multiplier != 1.0:
            hidden_states = hidden_states * config.embedding_multiplier

        num_layers = len(model.layers)
        for loop_index in range(config.num_loops):
            residual: tensorplay.Tensor | None = None
            kv_offset = loop_index * num_layers
            for layer in model.layers:
                hidden_states, residual = layer(
                    hidden_states,
                    residual,
                    batch=batch,
                    cache=cache,
                    layer_index=layer.layer_index + kv_offset,
                )
            assert residual is not None
            if config.skip_loop_final_norm:
                # Keep the fused residual raw between loops; the reference
                # applies a single norm after the last loop in this mode.
                hidden_states = residual + hidden_states
            else:
                # Default Nanbeige4.2 layout: normalize every loop output,
                # including the last one, before the lm_head.
                hidden_states, _ = model.norm.with_residual(hidden_states, residual)

        if config.skip_loop_final_norm:
            hidden_states = model.norm(hidden_states)

        head = self.lm_head()
        scale = config.logit_scale
        if batch.all_decode:
            logits = F.linear(hidden_states, head)
            return logits * scale if scale != 1.0 else logits
        indices = batch.sample_indices.clamp_min(0).long()
        logits = F.linear(hidden_states.index_select(0, indices), head)
        if scale != 1.0:
            logits = logits * scale
        return tensorplay.where(
            (batch.sample_indices >= 0)[:, None],
            logits,
            0,
        )


__all__ = ["NanbeigeForCausalLM"]
