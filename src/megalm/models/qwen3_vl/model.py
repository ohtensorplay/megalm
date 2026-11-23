from __future__ import annotations

import re
from pathlib import Path

import tensorplay
import tensorplay.nn as nn
import tensorplay.nn.functional as F

from megalm.kernels.bindings import KernelBindings, bind_kernel_plan
from megalm.kernels.ops import GroupedMoEWorkspace, KernelPolicy
from megalm.loading import SafetensorsSource, WeightLoadReport
from megalm.models.qwen3 import RMSNorm, SparseMoE, TokenEmbedding
from megalm.models.qwen3_5.attention import MRoPE
from megalm.models.qwen3_5.layers import PackedDenseMLP
from megalm.models.qwen3_5.norms import frozen_parameter
from megalm.models.qwen3_vl.attention import Qwen3VLAttention
from megalm.models.qwen3_vl.config import (
    Qwen3VLConfig,
    Qwen3VLTextConfig,
)
from megalm.models.qwen3_vl.vision import Qwen3VLVisionTransformer
from megalm.runtime.batch import ExecutionBatch
from megalm.runtime.cache import PagedKVCache


class Qwen3VLModelError(RuntimeError):
    """Raised when a Qwen3-VL checkpoint violates the native contract."""


class Qwen3VLSparseMoE(SparseMoE):
    def __init__(
        self,
        config: Qwen3VLTextConfig,
        *,
        device: tensorplay.device,
        dtype: tensorplay.dtype,
        kernel_bindings: KernelBindings,
    ) -> None:
        super().__init__(
            config,
            device=device,
            dtype=dtype,
            kernel_bindings=kernel_bindings,
        )


class Qwen3VLDecoderLayer(nn.Module):
    def __init__(
        self,
        config: Qwen3VLTextConfig,
        layer_index: int,
        *,
        rotary: MRoPE,
        device: tensorplay.device,
        dtype: tensorplay.dtype,
        kernel_bindings: KernelBindings,
    ) -> None:
        super().__init__()
        self.self_attn = Qwen3VLAttention(
            config,
            layer_index=layer_index,
            rotary=rotary,
            device=device,
            dtype=dtype,
            kernel_bindings=kernel_bindings,
        )
        self.mlp: PackedDenseMLP | Qwen3VLSparseMoE
        if config.is_moe:
            self.mlp = Qwen3VLSparseMoE(
                config,
                device=device,
                dtype=dtype,
                kernel_bindings=kernel_bindings,
            )
        else:
            self.mlp = PackedDenseMLP(
                config,
                device=device,
                dtype=dtype,
                kernel_bindings=kernel_bindings,
            )
        self.input_layernorm = RMSNorm(
            config.hidden_size,
            config.rms_norm_eps,
            device=device,
            dtype=dtype,
            kernel_bindings=kernel_bindings,
        )
        self.post_attention_layernorm = RMSNorm(
            config.hidden_size,
            config.rms_norm_eps,
            device=device,
            dtype=dtype,
            kernel_bindings=kernel_bindings,
        )

    def forward_with_residual(
        self,
        hidden_states: tensorplay.Tensor,
        residual: tensorplay.Tensor | None,
        *,
        batch: ExecutionBatch,
        cache: PagedKVCache,
        position_embeddings: tuple[tensorplay.Tensor, tensorplay.Tensor],
    ) -> tuple[tensorplay.Tensor, tensorplay.Tensor]:
        if residual is None:
            residual = hidden_states
            normalized = self.input_layernorm(hidden_states)
        else:
            normalized, residual = self.input_layernorm.with_residual(
                hidden_states,
                residual,
            )
        hidden_states = self.self_attn(
            normalized,
            batch=batch,
            cache=cache,
            position_embeddings=position_embeddings,
        )
        hidden_states, residual = (
            self.post_attention_layernorm.with_residual(
                hidden_states,
                residual,
            )
        )
        if isinstance(self.mlp, Qwen3VLSparseMoE):
            hidden_states = self.mlp(hidden_states, decode=batch.all_decode)
        else:
            hidden_states = self.mlp(hidden_states)
        return hidden_states, residual


class Qwen3VLTextModel(nn.Module):
    def __init__(
        self,
        config: Qwen3VLTextConfig,
        *,
        device: tensorplay.device,
        dtype: tensorplay.dtype,
        kernel_bindings: KernelBindings,
    ) -> None:
        super().__init__()
        self.config = config
        self.embed_tokens = TokenEmbedding(
            config.vocab_size,
            config.hidden_size,
            device=device,
            dtype=dtype,
        )
        self.rotary = MRoPE(config, device=device)
        self.layers = nn.ModuleList(
            [
                Qwen3VLDecoderLayer(
                    config,
                    layer_index,
                    rotary=self.rotary,
                    device=device,
                    dtype=dtype,
                    kernel_bindings=kernel_bindings,
                )
                for layer_index in range(config.num_hidden_layers)
            ]
        )
        self.norm = RMSNorm(
            config.hidden_size,
            config.rms_norm_eps,
            device=device,
            dtype=dtype,
            kernel_bindings=kernel_bindings,
        )

    def forward(
        self,
        batch: ExecutionBatch,
        cache: PagedKVCache,
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
        for layer_index, layer in enumerate(self.layers):
            hidden_states, residual = layer.forward_with_residual(
                hidden_states,
                residual,
                batch=batch,
                cache=cache,
                position_embeddings=position_embeddings,
            )
            if layer_index < len(batch.deepstack_embeddings):
                hidden_states = (
                    hidden_states
                    + batch.deepstack_embeddings[layer_index]
                )
        if residual is None:
            raise Qwen3VLModelError("Qwen3-VL has no decoder layers")
        hidden_states, _ = self.norm.with_residual(hidden_states, residual)
        return hidden_states


class Qwen3VLModel(nn.Module):
    def __init__(
        self,
        config: Qwen3VLConfig,
        *,
        device: tensorplay.device,
        dtype: tensorplay.dtype,
        kernel_bindings: KernelBindings,
    ) -> None:
        super().__init__()
        self.visual = Qwen3VLVisionTransformer(
            config.vision_config,
            device=device,
            dtype=dtype,
        )
        self.language_model = Qwen3VLTextModel(
            config.text_config,
            device=device,
            dtype=dtype,
            kernel_bindings=kernel_bindings,
        )


class Qwen3VLForConditionalGeneration(nn.Module):
    uses_mrope = True

    def __init__(
        self,
        config: Qwen3VLConfig,
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
                model_type=config.model_type,
            )
        self.kernel_bindings = kernel_bindings
        self.model = Qwen3VLModel(
            config,
            device=self.device,
            dtype=dtype,
            kernel_bindings=kernel_bindings,
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

    @property
    def deepstack_depth(self) -> int:
        return self.model.visual.deepstack_depth

    @property
    def hidden_size(self) -> int:
        return self.config.text_config.hidden_size

    @classmethod
    def from_pretrained(
        cls,
        model_path: str | Path,
        *,
        device: tensorplay.device | str = "cuda",
        dtype: tensorplay.dtype = tensorplay.bfloat16,
        kernel_policy: KernelPolicy,
        kernel_bindings: KernelBindings | None = None,
    ) -> tuple[Qwen3VLForConditionalGeneration, WeightLoadReport]:
        config = Qwen3VLConfig.from_path(model_path)
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

    def forward(
        self,
        batch: ExecutionBatch,
        cache: PagedKVCache,
    ) -> tensorplay.Tensor:
        return self.compute_logits(
            self.model.language_model(batch, cache),
            batch,
        )

    def compute_logits(
        self,
        hidden_states: tensorplay.Tensor,
        batch: ExecutionBatch,
    ) -> tensorplay.Tensor:
        lm_head = (
            self.model.language_model.embed_tokens.weight
            if self.config.tie_word_embeddings
            else self.lm_head_weight
        )
        assert lm_head is not None
        if batch.all_decode:
            return F.linear(hidden_states, lm_head)
        indices = batch.sample_indices.clamp_min(0).long()
        logits = F.linear(hidden_states.index_select(0, indices), lm_head)
        return tensorplay.where(
            (batch.sample_indices >= 0)[:, None],
            logits,
            0,
        )

    def lm_head(self) -> tensorplay.Tensor:
        lm_head = (
            self.model.language_model.embed_tokens.weight
            if self.config.tie_word_embeddings
            else self.lm_head_weight
        )
        assert lm_head is not None
        return lm_head

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
        hidden = self.hidden_size
        if image_embeddings.shape[-1] < hidden:
            raise Qwen3VLModelError("vision feature width is incomplete")
        main = image_embeddings[..., :hidden]
        mask = input_ids == self.config.image_token_id
        if int(mask.sum().item()) != main.shape[0]:
            raise Qwen3VLModelError(
                "image placeholder count differs from vision features"
            )
        embeddings = self.input_embeddings(input_ids)
        embeddings[mask] = main.to(embeddings.dtype)
        return embeddings

    def configure_moe_workspace(self, max_batched_tokens: int) -> int:
        if max_batched_tokens <= 0:
            raise Qwen3VLModelError("workspace token budget must be positive")
        moe_layers = [
            layer.mlp
            for layer in self.model.language_model.layers
            if isinstance(layer.mlp, Qwen3VLSparseMoE)
        ]
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
            raise Qwen3VLModelError(
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
            raise Qwen3VLModelError("warmup token budget must be positive")
        moe_layers = [
            layer.mlp
            for layer in self.model.language_model.layers
            if isinstance(layer.mlp, Qwen3VLSparseMoE)
        ]
        if not self.kernel_policy.use_triton or not moe_layers:
            return ()
        grouped_min = self.kernel_policy.grouped_moe_min_tokens
        if grouped_min > max_batched_tokens:
            return ()
        token_counts = [max(grouped_min, min(64, max_batched_tokens))]
        large_count = max(512, grouped_min)
        if large_count <= max_batched_tokens:
            token_counts.append(large_count)
        warmed: set[tuple[int, int, int, int, object]] = set()
        for mlp in moe_layers:
            geometry = (
                mlp.gate_up_weight.shape[0],
                mlp.gate_up_weight.shape[2],
                mlp.gate_up_weight.shape[1] // 2,
                mlp.top_k,
                mlp.gate_up_weight.dtype,
            )
            if geometry in warmed:
                continue
            warmed.add(geometry)
            for token_count in token_counts:
                hidden_states = tensorplay.zeros(
                    (token_count, mlp.gate_up_weight.shape[2]),
                    dtype=mlp.gate_up_weight.dtype,
                    device=mlp.gate_up_weight.device,
                )
                mlp(hidden_states, decode=False)
        return tuple(token_counts)

    def load_weights(self, model_path: str | Path) -> WeightLoadReport:
        config = self.config
        parameters = dict(self.named_parameters())
        loaded: set[str] = set()
        packed_parts: dict[str, set[str]] = {}
        text_config = config.text_config
        qkv_pattern = re.compile(
            r"model\.language_model\.layers\.(?P<layer>\d+)\."
            r"self_attn\.(?P<projection>q_proj|k_proj|v_proj)\.weight"
        )
        dense_pattern = re.compile(
            r"model\.language_model\.layers\.(?P<layer>\d+)\."
            r"mlp\.(?P<projection>gate_proj|up_proj)\.weight"
        )
        expert_pattern = re.compile(
            r"model\.language_model\.layers\.(?P<layer>\d+)\."
            r"mlp\.experts\.(?P<expert>\d+)\."
            r"(?P<projection>gate_proj|up_proj|down_proj)\.weight"
        )
        visual_qkv_pattern = re.compile(
            r"model\.visual\.blocks\.(?P<block>\d+)\.attn\."
            r"(?P<projection>q|k|v)\.(?P<field>weight|bias)"
        )

        def copy(target_name: str, tensor: tensorplay.Tensor) -> None:
            try:
                target = parameters[target_name]
            except KeyError as exc:
                raise Qwen3VLModelError(
                    f"checkpoint tensor has no native target: {target_name}"
                ) from exc
            if tuple(target.shape) != tuple(tensor.shape):
                raise Qwen3VLModelError(
                    f"shape mismatch for {target_name}: expected "
                    f"{tuple(target.shape)}, got {tuple(tensor.shape)}"
                )
            target.copy_(tensor.to(device=self.device, dtype=self.dtype))
            loaded.add(target_name)

        def copy_slice(
            target: tensorplay.Tensor,
            tensor: tensorplay.Tensor,
            source_name: str,
        ) -> None:
            if tuple(target.shape) != tuple(tensor.shape):
                raise Qwen3VLModelError(
                    f"shape mismatch for packed {source_name}"
                )
            target.copy_(tensor.to(device=self.device, dtype=self.dtype))

        @tensorplay.no_grad()
        def consume(source_name: str, tensor: tensorplay.Tensor) -> None:
            if source_name == "lm_head.weight" and config.tie_word_embeddings:
                return
            match = qkv_pattern.fullmatch(source_name)
            if match is not None:
                layer_index = int(match.group("layer"))
                projection = match.group("projection")
                attention = self.model.language_model.layers[
                    layer_index
                ].self_attn
                offsets = {
                    "q_proj": (0, attention.query_width),
                    "k_proj": (
                        attention.query_width,
                        attention.query_width + attention.kv_width,
                    ),
                    "v_proj": (
                        attention.query_width + attention.kv_width,
                        attention.query_width + 2 * attention.kv_width,
                    ),
                }
                start, end = offsets[projection]
                target_name = (
                    f"model.language_model.layers.{layer_index}."
                    "self_attn.qkv_proj_weight"
                )
                copy_slice(
                    attention.qkv_proj_weight[start:end],
                    tensor,
                    source_name,
                )
                packed_parts.setdefault(target_name, set()).add(projection)
                if packed_parts[target_name] == {
                    "q_proj",
                    "k_proj",
                    "v_proj",
                }:
                    loaded.add(target_name)
                return

            match = dense_pattern.fullmatch(source_name)
            if match is not None:
                layer_index = int(match.group("layer"))
                projection = match.group("projection")
                mlp = self.model.language_model.layers[layer_index].mlp
                if isinstance(mlp, Qwen3VLSparseMoE):
                    raise Qwen3VLModelError(
                        f"dense MLP weight supplied for MoE layer {layer_index}"
                    )
                target_name = (
                    f"model.language_model.layers.{layer_index}."
                    "mlp.gate_up_weight"
                )
                start = (
                    0
                    if projection == "gate_proj"
                    else text_config.intermediate_size
                )
                copy_slice(
                    mlp.gate_up_weight[
                        start : start + text_config.intermediate_size
                    ],
                    tensor,
                    source_name,
                )
                packed_parts.setdefault(target_name, set()).add(projection)
                if packed_parts[target_name] == {"gate_proj", "up_proj"}:
                    loaded.add(target_name)
                return

            match = expert_pattern.fullmatch(source_name)
            if match is not None:
                layer_index = int(match.group("layer"))
                expert_index = int(match.group("expert"))
                projection = match.group("projection")
                mlp = self.model.language_model.layers[layer_index].mlp
                if not isinstance(mlp, Qwen3VLSparseMoE):
                    raise Qwen3VLModelError(
                        f"expert weight supplied for dense layer {layer_index}"
                    )
                if projection == "down_proj":
                    target_name = (
                        f"model.language_model.layers.{layer_index}."
                        "mlp.down_weight"
                    )
                    target = mlp.down_weight[expert_index]
                    expected_parts = text_config.num_experts
                else:
                    target_name = (
                        f"model.language_model.layers.{layer_index}."
                        "mlp.gate_up_weight"
                    )
                    offset = (
                        0
                        if projection == "gate_proj"
                        else text_config.moe_intermediate_size
                    )
                    target = mlp.gate_up_weight[
                        expert_index,
                        offset : offset + text_config.moe_intermediate_size,
                    ]
                    expected_parts = text_config.num_experts * 2
                copy_slice(target, tensor, source_name)
                packed_parts.setdefault(target_name, set()).add(
                    f"{expert_index}:{projection}"
                )
                relevant = {
                    part
                    for part in packed_parts[target_name]
                    if (
                        projection == "down_proj"
                        and part.endswith(":down_proj")
                    )
                    or (
                        projection != "down_proj"
                        and not part.endswith(":down_proj")
                    )
                }
                if len(relevant) == expected_parts:
                    loaded.add(target_name)
                return

            match = visual_qkv_pattern.fullmatch(source_name)
            if match is not None:
                block_index = int(match.group("block"))
                projection = match.group("projection")
                field = match.group("field")
                visual = self.model.visual.blocks[block_index].attn
                hidden = config.vision_config.hidden_size
                offsets = {
                    "q": (0, hidden),
                    "k": (hidden, 2 * hidden),
                    "v": (2 * hidden, 3 * hidden),
                }
                start, end = offsets[projection]
                target_name = (
                    f"model.visual.blocks.{block_index}.attn.qkv.{field}"
                )
                copy_slice(getattr(visual.qkv, field)[start:end], tensor, source_name)
                packed_parts.setdefault(target_name, set()).add(projection)
                if packed_parts[target_name] == {"q", "k", "v"}:
                    loaded.add(target_name)
                return

            target_name = source_name
            replacements = (
                ("self_attn.o_proj.weight", "self_attn.o_proj_weight"),
                ("mlp.down_proj.weight", "mlp.down_proj_weight"),
            )
            for source_suffix, target_suffix in replacements:
                if target_name.endswith(source_suffix):
                    target_name = (
                        target_name[: -len(source_suffix)] + target_suffix
                    )
                    break
            if source_name == "lm_head.weight":
                target_name = "lm_head_weight"
            copy(target_name, tensor)

        source = SafetensorsSource(model_path)
        report = source.stream(consume)
        missing = sorted(set(parameters) - loaded)
        if missing:
            raise Qwen3VLModelError(
                f"checkpoint did not initialize parameters: {missing}"
            )
        return report
