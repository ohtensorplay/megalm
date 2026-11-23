from __future__ import annotations

from pathlib import Path

import tensorplay
import tensorplay.nn as nn
import tensorplay.nn.functional as F

from megalm.kernels.bindings import KernelBindings, bind_kernel_plan
from megalm.kernels.ops import GroupedMoEWorkspace, KernelPolicy
from megalm.loading import WeightLoadReport
from megalm.models.qwen3_5.attention import MRoPE
from megalm.models.qwen3_5.config import Qwen35Config, Qwen35TextConfig
from megalm.models.qwen3_5.layers import Qwen35DecoderLayer
from megalm.models.qwen3_5.loading import load_qwen35_weights
from megalm.models.qwen3_5.moe import Qwen35SparseMoeBlock
from megalm.models.qwen3_5.mtp import Qwen35MultiTokenPredictor
from megalm.models.qwen3_5.norms import OffsetRMSNorm, frozen_parameter
from megalm.models.qwen3_5.vision import Qwen35VisionTransformer
from megalm.runtime.batch import ExecutionBatch
from megalm.runtime.hybrid_cache import HybridCache


class Qwen35TextModel(nn.Module):
    def __init__(
        self,
        config: Qwen35TextConfig,
        *,
        device: tensorplay.device,
        dtype: tensorplay.dtype,
        kernel_bindings: KernelBindings,
    ) -> None:
        super().__init__()
        self.config = config
        self.embed_tokens = nn.Embedding(
            config.vocab_size,
            config.hidden_size,
            device=device,
            dtype=dtype,
        )
        self.embed_tokens.requires_grad_(False)
        self.rotary = MRoPE(config, device=device)
        self.layers = nn.ModuleList(
            [
                Qwen35DecoderLayer(
                    config,
                    layer_index=layer_index,
                    rotary=self.rotary,
                    device=device,
                    dtype=dtype,
                    kernel_bindings=kernel_bindings,
                )
                for layer_index in range(config.num_hidden_layers)
            ]
        )
        self.norm = OffsetRMSNorm(
            config.hidden_size,
            config.rms_norm_eps,
            device=device,
            dtype=dtype,
            kernel_bindings=kernel_bindings,
        )

    def forward(
        self,
        batch: ExecutionBatch,
        cache: HybridCache,
    ) -> tensorplay.Tensor:
        hidden_states = (
            self.embed_tokens(batch.input_ids)
            if batch.input_embeddings is None
            else batch.input_embeddings
        )
        positions = (
            batch.positions
            if batch.mrope_positions is None
            else batch.mrope_positions
        )
        position_embeddings = self.rotary(
            positions,
            dtype=hidden_states.dtype,
        )
        residual: tensorplay.Tensor | None = None
        for layer in self.layers:
            hidden_states, residual = layer.forward_with_residual(
                hidden_states,
                residual,
                batch=batch,
                cache=cache,
                position_embeddings=position_embeddings,
            )
        if residual is None:
            raise RuntimeError("Qwen3.5 text model has no decoder layers")
        hidden_states, _ = self.norm.with_residual(
            hidden_states,
            residual,
        )
        return hidden_states


class Qwen35Model(nn.Module):
    def __init__(
        self,
        config: Qwen35Config,
        *,
        device: tensorplay.device,
        dtype: tensorplay.dtype,
        kernel_bindings: KernelBindings,
    ) -> None:
        super().__init__()
        self.visual = Qwen35VisionTransformer(
            config.vision_config,
            device=device,
            dtype=dtype,
        )
        self.language_model = Qwen35TextModel(
            config.text_config,
            device=device,
            dtype=dtype,
            kernel_bindings=kernel_bindings,
        )


class Qwen35ForConditionalGeneration(nn.Module):
    """Native Qwen3.5 image-text generation model; no Transformers runtime."""

    uses_mrope = True

    def __init__(
        self,
        config: Qwen35Config,
        *,
        device: tensorplay.device | str,
        dtype: tensorplay.dtype,
        kernel_policy: KernelPolicy,
        kernel_bindings: KernelBindings | None = None,
        enable_mtp: bool = False,
    ) -> None:
        super().__init__()
        if enable_mtp and config.text_config.is_moe:
            raise NotImplementedError(
                "native MTP is available only for dense Qwen3.5 checkpoints"
            )
        self.config = config
        self.device = tensorplay.device(device)
        self.dtype = dtype
        self.kernel_policy = kernel_policy
        if kernel_bindings is None:
            kernel_bindings = bind_kernel_plan(
                kernel_policy,
                device=self.device,
                dtype=dtype,
                model_type=config.model_type,
            )
        self.kernel_bindings = kernel_bindings
        self.model = Qwen35Model(
            config,
            device=self.device,
            dtype=dtype,
            kernel_bindings=kernel_bindings,
        )
        self.mtp = (
            Qwen35MultiTokenPredictor(
                config.text_config,
                embed_tokens=self.model.language_model.embed_tokens,
                rotary=self.model.language_model.rotary,
                device=self.device,
                dtype=dtype,
                kernel_bindings=kernel_bindings,
            )
            if enable_mtp
            else None
        )
        self.register_buffer(
            "_mtp_hidden_buffer",
            None,
            persistent=False,
        )
        if config.tie_word_embeddings:
            self.register_parameter("lm_head_weight", None)
            self.lm_head_weight = None
        else:
            self.lm_head_weight = frozen_parameter(
                (config.vocab_size, config.text_config.hidden_size),
                device=self.device,
                dtype=dtype,
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
        enable_mtp: bool = False,
    ) -> tuple[Qwen35ForConditionalGeneration, WeightLoadReport]:
        config = Qwen35Config.from_path(model_path)
        model = cls(
            config,
            device=device,
            dtype=dtype,
            kernel_policy=kernel_policy,
            kernel_bindings=kernel_bindings,
            enable_mtp=enable_mtp,
        )
        report = load_qwen35_weights(model, model_path)
        model.eval()
        return model, report

    def forward(
        self,
        batch: ExecutionBatch,
        cache: HybridCache,
    ) -> tensorplay.Tensor:
        hidden_states = self.forward_hidden(batch, cache)
        if self.mtp is not None:
            if (
                self._mtp_hidden_buffer is None
                or hidden_states.shape[0]
                > self._mtp_hidden_buffer.shape[0]
            ):
                raise RuntimeError("Qwen3.5 MTP hidden buffer is not configured")
            self._mtp_hidden_buffer[: hidden_states.shape[0]].copy_(
                hidden_states
            )
        return self.compute_logits(hidden_states, batch)

    @property
    def last_hidden_states(self) -> tensorplay.Tensor | None:
        """Return the optional graph-stable MTP feature buffer."""

        return self._mtp_hidden_buffer

    def configure_mtp_hidden_buffer(self, max_tokens: int) -> int:
        if self.mtp is None:
            return 0
        if max_tokens <= 0:
            raise ValueError("MTP hidden buffer capacity must be positive")
        self._mtp_hidden_buffer = tensorplay.empty(
            (max_tokens, self.config.text_config.hidden_size),
            dtype=self.dtype,
            device=self.device,
        )
        return (
            self._mtp_hidden_buffer.numel()
            * self._mtp_hidden_buffer.element_size()
        )

    def forward_hidden(
        self,
        batch: ExecutionBatch,
        cache: HybridCache,
    ) -> tensorplay.Tensor:
        return self.model.language_model(batch, cache)

    def lm_head(self) -> tensorplay.Tensor:
        lm_head = (
            self.model.language_model.embed_tokens.weight
            if self.config.tie_word_embeddings
            else self.lm_head_weight
        )
        assert lm_head is not None
        return lm_head

    def compute_logits(
        self,
        hidden_states: tensorplay.Tensor,
        batch: ExecutionBatch,
    ) -> tensorplay.Tensor:
        lm_head = self.lm_head()
        if batch.all_decode:
            return F.linear(hidden_states, lm_head)
        indices = batch.sample_indices.clamp_min(0).long()
        logits = F.linear(hidden_states.index_select(0, indices), lm_head)
        return tensorplay.where(
            (batch.sample_indices >= 0)[:, None],
            logits,
            0,
        )

    def encode_images(
        self,
        pixel_values: tensorplay.Tensor,
        grid_thw: tensorplay.Tensor,
    ) -> tensorplay.Tensor:
        return self.model.visual(pixel_values, grid_thw)

    def input_embeddings(self, input_ids: tensorplay.Tensor) -> tensorplay.Tensor:
        return self.model.language_model.embed_tokens(input_ids)

    def merge_image_embeddings(
        self,
        input_ids: tensorplay.Tensor,
        image_embeddings: tensorplay.Tensor,
    ) -> tensorplay.Tensor:
        mask = input_ids == self.config.image_token_id
        if int(mask.sum().item()) != image_embeddings.shape[0]:
            raise ValueError(
                "image placeholder count differs from vision features"
            )
        embeddings = self.input_embeddings(input_ids)
        embeddings[mask] = image_embeddings.to(embeddings.dtype)
        return embeddings

    def _moe_layers(self) -> list[Qwen35SparseMoeBlock]:
        return [
            layer.mlp
            for layer in self.model.language_model.layers
            if isinstance(layer.mlp, Qwen35SparseMoeBlock)
        ]

    def configure_moe_workspace(self, max_batched_tokens: int) -> int:
        if max_batched_tokens <= 0:
            raise ValueError("workspace token budget must be positive")
        moe_layers = self._moe_layers()
        if not self.kernel_policy.use_triton or not moe_layers:
            return 0
        first = moe_layers[0]
        geometry = (
            first.gate_up_weight.shape[0],
            first.gate_up_weight.shape[2],
            first.gate_up_weight.shape[1] // 2,
        )
        if any(
            (
                mlp.gate_up_weight.shape[0],
                mlp.gate_up_weight.shape[2],
                mlp.gate_up_weight.shape[1] // 2,
            )
            != geometry
            for mlp in moe_layers
        ):
            raise ValueError(
                "shared MoE workspace requires one expert geometry"
            )
        num_experts, hidden_size, intermediate_size = geometry
        workspace = GroupedMoEWorkspace.allocate(
            max_tokens=max_batched_tokens,
            num_experts=num_experts,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            dtype=first.gate_up_weight.dtype,
            device=first.gate_up_weight.device,
            max_top_k=max(mlp.top_k for mlp in moe_layers),
        )
        for mlp in moe_layers:
            mlp.workspace = workspace
        return workspace.nbytes

    @tensorplay.inference_mode()
    def warmup_prefill_kernels(
        self,
        *,
        max_batched_tokens: int,
    ) -> tuple[int, ...]:
        if max_batched_tokens <= 0:
            raise ValueError("warmup token budget must be positive")
        moe_layers = self._moe_layers()
        if not self.kernel_policy.use_triton or not moe_layers:
            return ()
        grouped_min = self.kernel_policy.grouped_moe_min_tokens
        if grouped_min > max_batched_tokens:
            return ()
        token_counts = [max(grouped_min, min(64, max_batched_tokens))]
        large_count = max(512, grouped_min)
        if large_count <= max_batched_tokens:
            token_counts.append(large_count)
        warmed_geometries: set[tuple[int, int, int, int, object]] = set()
        for mlp in moe_layers:
            geometry = (
                mlp.gate_up_weight.shape[0],
                mlp.gate_up_weight.shape[2],
                mlp.gate_up_weight.shape[1] // 2,
                mlp.top_k,
                mlp.gate_up_weight.dtype,
            )
            if geometry in warmed_geometries:
                continue
            warmed_geometries.add(geometry)
            hidden_size = mlp.gate_up_weight.shape[2]
            for token_count in token_counts:
                hidden_states = tensorplay.zeros(
                    (token_count, hidden_size),
                    dtype=mlp.gate_up_weight.dtype,
                    device=mlp.gate_up_weight.device,
                )
                mlp(hidden_states, decode=False)
        return tuple(token_counts)
