"""Native TensorPlay executor for Gemma-4 text models.

This implements the checkpoint semantics end to end: scaled word embeddings,
per-layer embeddings (PLE), sandwich RMSNorms, layer scalars, 5:1 sliding/full
attention with proportional RoPE on the global layers, YOCO KV sharing, and
the optional dense+MoE double feed-forward block.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Sequence

import tensorplay
import tensorplay.nn.functional as F
from tensorplay import nn

from megalm.kernels.bindings import KernelBindings, bind_kernel_plan
from megalm.kernels.decode_gemv import gate_up_gelu_tanh_gemv
from megalm.kernels.ops import KernelPolicy
from megalm.kernels.residual_ops import add_scaled
from megalm.kernels.rms_norm import rms_norm_add
from megalm.kernels.silu_mul import gelu_tanh_and_mul
from megalm.models.components.mtp import MTPTargetMixin
from megalm.models.gemma4.audio import Gemma4AudioModel
from megalm.models.gemma4.vision import Gemma4VisionModel, MultimodalEmbedder
from megalm.loading import SafetensorsSource, WeightLoadReport
from megalm.models.gemma4.attention import (
    Gemma4Attention,
    Gemma4RotaryCache,
    gelu_tanh,
)
from megalm.models.gemma4.config import (
    FULL_ATTENTION,
    SLIDING_ATTENTION,
    Gemma4Config,
    Gemma4TextConfig,
)
from megalm.runtime.batch import ExecutionBatch
from megalm.runtime.cache import PagedKVCache


class Gemma4Error(RuntimeError):
    """Raised when a Gemma-4 checkpoint violates its native contract."""


def _parameter(
    shape: tuple[int, ...],
    *,
    device: tensorplay.device,
    dtype: tensorplay.dtype,
) -> nn.Parameter:
    return nn.Parameter(
        tensorplay.empty(shape, device=device, dtype=dtype),
        requires_grad=False,
    )


class Gemma4RMSNorm(nn.Module):
    def __init__(
        self,
        width: int,
        eps: float,
        *,
        device: tensorplay.device,
        dtype: tensorplay.dtype,
        kernel_bindings: KernelBindings,
    ) -> None:
        super().__init__()
        self.weight = _parameter((width,), device=device, dtype=dtype)
        self.eps = eps
        self._rms_norm = kernel_bindings.core.rms_norm
        self._fused_add_rms_norm = kernel_bindings.core.fused_add_rms_norm

    def forward(self, value: tensorplay.Tensor) -> tensorplay.Tensor:
        return self._rms_norm(value, self.weight, self.eps)

    def with_residual(
        self,
        value: tensorplay.Tensor,
        residual: tensorplay.Tensor,
    ) -> tuple[tensorplay.Tensor, tensorplay.Tensor]:
        return self._fused_add_rms_norm(value, residual, self.weight, self.eps)


class Gemma4SparseMoE(nn.Module):
    """The optional second (sparse) feed-forward branch of Gemma-4.

    Routing is softmax over *all* experts, top-k, renormalise over the
    selected experts, then multiply by the learned per-expert scale.  The
    branch is added to the dense MLP output, it does not replace it.
    """

    def __init__(
        self,
        config: Gemma4TextConfig,
        *,
        device: tensorplay.device,
        dtype: tensorplay.dtype,
    ) -> None:
        super().__init__()
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.moe_intermediate_size
        self.num_experts = config.num_experts
        self.top_k = config.top_k_experts
        self.eps = config.rms_norm_eps
        self.router_scale = _parameter(
            (config.hidden_size,), device=device, dtype=dtype
        )
        self.router_proj = _parameter(
            (config.num_experts, config.hidden_size),
            device=device,
            dtype=dtype,
        )
        self.per_expert_scale = _parameter(
            (config.num_experts,), device=device, dtype=dtype
        )
        self.gate_up_weight = _parameter(
            (config.num_experts, 2 * self.intermediate_size, config.hidden_size),
            device=device,
            dtype=dtype,
        )
        self.down_weight = _parameter(
            (config.num_experts, config.hidden_size, self.intermediate_size),
            device=device,
            dtype=dtype,
        )

    def forward(
        self,
        expert_states: tensorplay.Tensor,
        *,
        routing_states: tensorplay.Tensor | None = None,
    ) -> tensorplay.Tensor:
        # HF routes on the raw residual but feeds the experts the
        # pre_feedforward_layernorm_2-projected residual.
        if routing_states is None:
            routing_states = expert_states
        mean = routing_states.pow(2).mean(-1, keepdim=True) + self.eps
        routed = routing_states * mean.rsqrt()
        routed = routed * (self.hidden_size**-0.5) * self.router_scale
        logits = F.linear(routed, self.router_proj)
        probabilities = tensorplay.softmax(logits.float(), dim=-1)
        top_weights, top_ids = tensorplay.topk(probabilities, self.top_k, dim=-1)
        top_weights = top_weights / top_weights.sum(-1, keepdim=True)
        top_weights = top_weights * self.per_expert_scale.index_select(
            0, top_ids.reshape(-1)
        ).reshape(top_ids.shape)
        result = tensorplay.zeros_like(expert_states)
        for expert in range(self.num_experts):
            mask = top_ids == expert
            if not bool(mask.any()):
                continue
            weight = (top_weights * mask).sum(-1)
            active = weight > 0
            tokens = expert_states[active]
            if tokens.shape[0] == 0:
                continue
            gate_up = F.linear(
                tokens, self.gate_up_weight[expert]
            )
            gate, up = gate_up.chunk(2, dim=-1)
            activated = gelu_tanh(gate) * up
            expert_output = F.linear(activated, self.down_weight[expert])
            update = tensorplay.zeros_like(expert_states)
            update[active] = expert_output * weight[active].unsqueeze(-1).to(
                expert_output.dtype
            )
            result = result + update
        return result


class Gemma4DecoderLayer(nn.Module):
    def __init__(
        self,
        config: Gemma4TextConfig,
        layer_index: int,
        *,
        kernel_bindings: KernelBindings,
        device: tensorplay.device,
        dtype: tensorplay.dtype,
        rotary: dict[str, Gemma4RotaryCache],
    ) -> None:
        super().__init__()
        self.layer_index = layer_index
        self.config = config
        self._use_triton = kernel_bindings.policy.use_triton
        first_shared = config.num_hidden_layers - config.num_kv_shared_layers
        owns_kv = layer_index < first_shared
        kind = config.layer_types[layer_index]
        head_dim = config.layer_head_dims[layer_index]
        num_kv_heads = config.layer_num_key_value_heads[layer_index]
        self.self_attn = Gemma4Attention(
            layer_index=layer_index,
            head_dim=head_dim,
            num_heads=config.num_attention_heads,
            num_kv_heads=num_kv_heads,
            owns_kv=owns_kv,
            use_k_eq_v=config.attention_k_eq_v and kind == FULL_ATTENTION,
            sliding_window=(
                config.sliding_window if kind == SLIDING_ATTENTION else None
            ),
            softcap=config.attn_logit_softcapping,
            hidden_size=config.hidden_size,
            eps=config.rms_norm_eps,
            rotary=rotary[kind],
            kernel_bindings=kernel_bindings,
            device=device,
            dtype=dtype,
        )
        self.input_layernorm = Gemma4RMSNorm(
            config.hidden_size, config.rms_norm_eps,
            device=device, dtype=dtype, kernel_bindings=kernel_bindings,
        )
        self.post_attention_layernorm = Gemma4RMSNorm(
            config.hidden_size, config.rms_norm_eps,
            device=device, dtype=dtype, kernel_bindings=kernel_bindings,
        )
        self.pre_feedforward_layernorm = Gemma4RMSNorm(
            config.hidden_size, config.rms_norm_eps,
            device=device, dtype=dtype, kernel_bindings=kernel_bindings,
        )
        self.post_feedforward_layernorm = Gemma4RMSNorm(
            config.hidden_size, config.rms_norm_eps,
            device=device, dtype=dtype, kernel_bindings=kernel_bindings,
        )
        intermediate = config.mlp_intermediate(layer_index)
        self.gate_up_weight = _parameter(
            (2 * intermediate, config.hidden_size), device=device, dtype=dtype
        )
        self.down_weight = _parameter(
            (config.hidden_size, intermediate), device=device, dtype=dtype
        )
        self.mlp_intermediate = intermediate

        if config.enable_moe_block:
            self.moe = Gemma4SparseMoE(
                config, device=device, dtype=dtype
            )
            self.post_feedforward_layernorm_1 = Gemma4RMSNorm(
                config.hidden_size, config.rms_norm_eps,
                device=device, dtype=dtype, kernel_bindings=kernel_bindings,
            )
            self.post_feedforward_layernorm_2 = Gemma4RMSNorm(
                config.hidden_size, config.rms_norm_eps,
                device=device, dtype=dtype, kernel_bindings=kernel_bindings,
            )
            self.pre_feedforward_layernorm_2 = Gemma4RMSNorm(
                config.hidden_size, config.rms_norm_eps,
                device=device, dtype=dtype, kernel_bindings=kernel_bindings,
            )
        else:
            self.moe = None

        if config.uses_ple:
            ple_dim = config.hidden_size_per_layer_input
            self.per_layer_input_gate = _parameter(
                (ple_dim, config.hidden_size), device=device, dtype=dtype
            )
            self.per_layer_projection = _parameter(
                (config.hidden_size, ple_dim), device=device, dtype=dtype
            )
            self.post_per_layer_input_norm = Gemma4RMSNorm(
                config.hidden_size, config.rms_norm_eps,
                device=device, dtype=dtype, kernel_bindings=kernel_bindings,
            )
        else:
            self.per_layer_input_gate = None
            self.per_layer_projection = None
            self.post_per_layer_input_norm = None
        self.register_buffer(
            "layer_scalar",
            tensorplay.ones((1,), dtype=dtype, device=device),
        )

    def forward(
        self,
        hidden_states: tensorplay.Tensor,
        *,
        batch: ExecutionBatch,
        cache: PagedKVCache,
        per_layer_input: tensorplay.Tensor | None = None,
    ) -> tensorplay.Tensor:
        config = self.config
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn(
            hidden_states,
            batch=batch,
            cache=cache,
            kv_slot=config.kv_slot(self.layer_index),
        )
        # Gemma-4 applies its post norm to the sublayer output only, then adds
        # the residual.  (The kernel's fused-add norm normalises the *sum*,
        # which is a different contract used by Qwen-style layers.)
        triton_path = self._use_triton and hidden_states.device.type == "cuda"
        if triton_path:
            norm = self.post_attention_layernorm
            hidden_states = rms_norm_add(
                hidden_states, norm.weight, norm.eps, residual
            )
        else:
            hidden_states = self.post_attention_layernorm(hidden_states)
            hidden_states = hidden_states + residual
        residual = hidden_states
        hidden_states = self.pre_feedforward_layernorm(hidden_states)
        if triton_path and hidden_states.shape[0] == 1:
            activated = gate_up_gelu_tanh_gemv(
                hidden_states[0].contiguous(), self.gate_up_weight
            ).unsqueeze(0)
            hidden_states = F.linear(activated, self.down_weight)
        elif triton_path:
            gate_up = F.linear(hidden_states, self.gate_up_weight)
            hidden_states = F.linear(
                gelu_tanh_and_mul(gate_up), self.down_weight
            )
        else:
            gate_up = F.linear(hidden_states, self.gate_up_weight)
            gate, up = gate_up.chunk(2, dim=-1)
            hidden_states = F.linear(gelu_tanh(gate) * up, self.down_weight)
        if self.moe is not None:
            hidden_states = self.post_feedforward_layernorm_1(hidden_states)
            moe_input = self.pre_feedforward_layernorm_2(residual)
            moe_output = self.moe(moe_input, routing_states=residual)
            moe_output = self.post_feedforward_layernorm_2(moe_output)
            hidden_states = hidden_states + moe_output
        hidden_states = self.post_feedforward_layernorm(hidden_states)
        fused = self._use_triton and hidden_states.device.type == "cuda"
        if per_layer_input is not None and self.per_layer_input_gate is not None:
            residual = hidden_states + residual
            hidden_states = residual
            gated = gelu_tanh(
                F.linear(hidden_states, self.per_layer_input_gate)
            )
            gated = gated * per_layer_input
            projected = F.linear(gated, self.per_layer_projection)
            projected = self.post_per_layer_input_norm(projected)
            if fused:
                return add_scaled(
                    residual, projected, self.layer_scalar
                )
            hidden_states = residual + projected
        elif fused:
            return add_scaled(hidden_states, residual, self.layer_scalar)
        else:
            hidden_states = hidden_states + residual
        return hidden_states * self.layer_scalar


class Gemma4TextModel(nn.Module):
    def __init__(
        self,
        config: Gemma4TextConfig,
        *,
        kernel_bindings: KernelBindings,
        device: tensorplay.device,
        dtype: tensorplay.dtype,
    ) -> None:
        super().__init__()
        self.config = config
        self.embed_tokens = _parameter(
            (config.vocab_size, config.hidden_size), device=device, dtype=dtype
        )
        # HF casts sqrt(hidden_size) to the checkpoint dtype before scaling.
        self.embed_scale_value = math.sqrt(config.hidden_size)
        self.embed_scale = tensorplay.tensor(
            self.embed_scale_value, dtype=dtype, device=device
        )
        self.rotary: dict[str, Gemma4RotaryCache] = {}
        for kind in dict.fromkeys(config.layer_types):
            self.rotary[kind] = Gemma4RotaryCache(
                head_dim=next(
                    dim
                    for dim, layer_kind in zip(config.layer_head_dims, config.layer_types)
                    if layer_kind == kind
                ),
                partial=config.rope_partial_factor[kind],
                theta=config.rope_theta[kind],
                rope_type=config.rope_type[kind],
                max_positions=config.max_position_embeddings,
                device=device,
            )
        self.layers = nn.ModuleList(
            [
                Gemma4DecoderLayer(
                    config,
                    index,
                    kernel_bindings=kernel_bindings,
                    device=device,
                    dtype=dtype,
                    rotary=self.rotary,
                )
                for index in range(config.num_hidden_layers)
            ]
        )
        self.norm = Gemma4RMSNorm(
            config.hidden_size, config.rms_norm_eps,
            device=device, dtype=dtype, kernel_bindings=kernel_bindings,
        )
        if config.uses_ple:
            ple_dim = config.hidden_size_per_layer_input
            total = config.num_hidden_layers * ple_dim
            self.embed_tokens_per_layer = _parameter(
                (config.vocab_size_per_layer_input, total),
                device=device, dtype=dtype,
            )
            self.embed_scale_per_layer = tensorplay.tensor(
                math.sqrt(ple_dim), dtype=dtype, device=device
            )
            self.per_layer_model_projection = _parameter(
                (total, config.hidden_size), device=device, dtype=dtype
            )
            self.per_layer_projection_norm = Gemma4RMSNorm(
                ple_dim, config.rms_norm_eps,
                device=device, dtype=dtype, kernel_bindings=kernel_bindings,
            )
            self.per_layer_input_scale = 2.0**-0.5
            self.per_layer_projection_scale = config.hidden_size**-0.5
        else:
            self.embed_tokens_per_layer = None

    def _per_layer_inputs(
        self, batch: ExecutionBatch, hidden: tensorplay.Tensor
    ) -> tensorplay.Tensor | None:
        if self.embed_tokens_per_layer is None:
            return None
        config = self.config
        ple_dim = config.hidden_size_per_layer_input
        tokens = batch.input_ids
        # Decode batches may carry stale/sentinel ids in padding rows; clamp
        # both ends so the embedding gather is always in range.
        safe_ids = tokens.long().clamp(0, config.vocab_size_per_layer_input - 1)
        token_part = F.embedding(
            safe_ids, self.embed_tokens_per_layer
        ) * self.embed_scale_per_layer
        context = F.linear(hidden, self.per_layer_model_projection)
        context = context.view(
            hidden.shape[0], config.num_hidden_layers, ple_dim
        ) * self.per_layer_projection_scale
        context = self.per_layer_projection_norm(context)
        combined = (context + token_part.view(context.shape)) * self.per_layer_input_scale
        return combined

    def forward(
        self,
        batch: ExecutionBatch,
        cache: PagedKVCache,
    ) -> tensorplay.Tensor:
        # Batched decode staging rows can carry stale ids; keep the embedding
        # gather in range (valid token ids are unchanged by this clamp).
        safe_ids = batch.input_ids.long().clamp(
            0, self.embed_tokens.shape[0] - 1
        )
        hidden = F.embedding(safe_ids, self.embed_tokens)
        if self.embed_scale_value != 1.0:
            hidden = hidden * self.embed_scale
        per_layer = self._per_layer_inputs(batch, hidden)
        for index, layer in enumerate(self.layers):
            ple = None if per_layer is None else per_layer[:, index, :]
            hidden = layer(
                hidden, batch=batch, cache=cache, per_layer_input=ple
            )
        return self.norm(hidden)


class Gemma4ForCausalLM(MTPTargetMixin, nn.Module):
    """The first-party Gemma-4 text executor (also used by MM/unified)."""

    def __init__(
        self,
        config: Gemma4TextConfig,
        *,
        device: tensorplay.device | str,
        dtype: tensorplay.dtype,
        kernel_policy: KernelPolicy,
        kernel_bindings: KernelBindings | None = None,
        vision_config: Any = None,
        audio_config: Any = None,
        token_ids: dict[str, int] | None = None,
        enable_mtp: bool = False,
        mtp_config: Any | None = None,
    ) -> None:
        super().__init__()
        self.text_config = config
        self.device = tensorplay.device(device)
        self.dtype = dtype
        self.kernel_policy = kernel_policy
        if kernel_bindings is None:
            kernel_bindings = bind_kernel_plan(
                kernel_policy,
                device=self.device,
                dtype=dtype,
                model_type="gemma4",
            )
        if kernel_bindings.policy != kernel_policy:
            raise Gemma4Error("kernel bindings do not match the model kernel policy")
        self.kernel_bindings = kernel_bindings
        self.model = Gemma4TextModel(
            config,
            kernel_bindings=kernel_bindings,
            device=self.device,
            dtype=dtype,
        )
        self.vision_tower: Gemma4VisionModel | None = None
        self.audio_tower: Gemma4AudioModel | None = None
        self.embed_vision: MultimodalEmbedder | None = None
        self.embed_audio: MultimodalEmbedder | None = None
        if vision_config is not None:
            self.vision_tower = Gemma4VisionModel(
                vision_config,
                device=self.device,
                dtype=dtype,
                kernel_bindings=kernel_bindings,
            )
            self.embed_vision = MultimodalEmbedder(
                input_size=vision_config.hidden_size,
                output_size=config.hidden_size,
                eps=vision_config.rms_norm_eps,
                device=self.device,
                dtype=dtype,
                kernel_bindings=kernel_bindings,
            )
        if audio_config is not None:
            self.audio_tower = Gemma4AudioModel(
                audio_config,
                device=self.device,
                dtype=dtype,
                kernel_bindings=kernel_bindings,
            )
            self.embed_audio = MultimodalEmbedder(
                input_size=audio_config.output_proj_dims,
                output_size=config.hidden_size,
                eps=audio_config.rms_norm_eps,
                device=self.device,
                dtype=dtype,
                kernel_bindings=kernel_bindings,
            )
        if config.tie_word_embeddings:
            self.register_parameter("lm_head_weight", None)
            self.lm_head_weight = None
        else:
            self.lm_head_weight = _parameter(
                (config.vocab_size, config.hidden_size),
                device=self.device,
                dtype=dtype,
            )

        self.mtp = None
        if enable_mtp:
            from megalm.models.gemma4.mtp import Gemma4MultiTokenPredictor
            from megalm.models.gemma4.config import Gemma4AssistantConfig

            if mtp_config is None:
                from dataclasses import replace

                assistant_text = replace(
                    config,
                    num_kv_shared_layers=config.num_hidden_layers,
                    hidden_size_per_layer_input=0,
                    vocab_size_per_layer_input=0,
                )
                mtp_config = Gemma4AssistantConfig(
                    text_config=assistant_text,
                    backbone_hidden_size=config.hidden_size,
                )
            self.mtp = Gemma4MultiTokenPredictor(
                mtp_config,
                backbone_embed_tokens=self.model.embed_tokens,
                kernel_bindings=kernel_bindings,
                device=self.device,
                dtype=dtype,
            )
        self._init_mtp_state(dtype=dtype, device=self.device)

        # Heterogeneous per-layer KV geometry is normalised to the largest
        # geometry for the shared paged arena; each layer takes a view.
        self.cache_head_dim = config.max_head_dim
        self.cache_num_kv_heads = config.max_kv_heads
        self.num_kv_cache_layers = config.num_kv_cache_layers
        self.kv_cache_layer_indices = tuple(range(config.num_kv_cache_layers))
        token_ids = token_ids or {}
        self.config = SimpleNamespace(
            model_type=config.model_type,
            vocab_size=config.vocab_size,
            hidden_size=config.hidden_size,
            num_hidden_layers=config.num_hidden_layers,
            head_dim=self.cache_head_dim,
            num_key_value_heads=self.cache_num_kv_heads,
            max_position_embeddings=config.max_position_embeddings,
            text_config=SimpleNamespace(
                max_position_embeddings=config.max_position_embeddings
            ),
            vision_config=vision_config,
            audio_config=audio_config,
            image_token_id=token_ids.get("image_token_id"),
            video_token_id=token_ids.get("video_token_id"),
            audio_token_id=token_ids.get("audio_token_id"),
        )

    def _load_vision_tensor(
        self,
        name: str,
        tensor: tensorplay.Tensor,
        copy: Any,
        copy_slice: Any,
        loaded: set[str],
    ) -> None:
        tower = self.vision_tower
        assert tower is not None
        if name.endswith((".input_max", ".input_min", ".output_max", ".output_min")):
            # Checkpoint per-linear calibration bounds.  Leaf name is e.g.
            # "self_attn.q_proj.input_min" / "mlp.gate_proj.output_max".
            parts = name.split(".")
            if len(parts) < 5 or parts[0] != "encoder" or parts[1] != "layers":
                return
            index = int(parts[2])
            kind = parts[-1]
            module_name = ".".join(parts[3:-1])
            layer = tower.layers[index]
            target = (
                layer.self_attn
                if module_name.startswith("self_attn.")
                else layer
            )
            target._set_clip(module_name, kind, tensor)
            return
        if name == "patch_embedder.input_proj.weight":
            copy("vision_tower.patch_embedder.input_proj", tensor)
            return
        if name == "patch_embedder.position_embedding_table":
            copy("vision_tower.patch_embedder.position_x", tensor[0])
            copy("vision_tower.patch_embedder.position_y", tensor[1])
            return
        parts = name.split(".")
        if len(parts) < 5 or parts[0] != "encoder" or parts[1] != "layers":
            return
        index = int(parts[2])
        leaf = ".".join(parts[3:])
        prefix = f"vision_tower.layers.{index}"
        renames = {
            "input_layernorm.weight": "input_norm.weight",
            "post_attention_layernorm.weight": "post_attention_norm.weight",
            "pre_feedforward_layernorm.weight": "pre_mlp_norm.weight",
            "post_feedforward_layernorm.weight": "post_mlp_norm.weight",
        }
        if leaf in renames:
            copy(f"{prefix}.{renames[leaf]}", tensor)
            return
        if leaf.startswith("self_attn."):
            field = leaf.split(".")[1]
            if field in {"q_norm", "k_norm"}:
                copy(f"{prefix}.self_attn.{field}.weight", tensor)
            elif field in {"q_proj", "k_proj", "v_proj", "o_proj"}:
                copy(f"{prefix}.self_attn.{field}", tensor)
            return
        if leaf == "mlp.gate_proj.linear.weight":
            copy(f"{prefix}.gate_weight", tensor)
            return
        if leaf == "mlp.up_proj.linear.weight":
            copy(f"{prefix}.up_weight", tensor)
            return
        if leaf == "mlp.down_proj.linear.weight":
            copy(f"{prefix}.down_weight", tensor)

    def _load_audio_tensor(
        self,
        name: str,
        tensor: tensorplay.Tensor,
        copy: Any,
    ) -> None:
        tower = self.audio_tower
        assert tower is not None
        if name.endswith((".input_max", ".input_min", ".output_max", ".output_min")):
            return
        renames = {
            "subsample_conv_projection.input_proj_linear.weight": "input_proj",
            "subsample_conv_projection.layer0.conv.weight": "subsample1.weight",
            "subsample_conv_projection.layer0.norm.weight": "subsample_norm1.weight",
            "subsample_conv_projection.layer1.conv.weight": "subsample2.weight",
            "subsample_conv_projection.layer1.norm.weight": "subsample_norm2.weight",
            "output_proj.weight": "output_proj",
        }
        if name in renames:
            copy(f"audio_tower.{renames[name]}", tensor)
            return
        parts = name.split(".")
        if len(parts) < 4 or parts[0] != "layers":
            return
        index = int(parts[1])
        leaf = ".".join(parts[2:])
        prefix = f"audio_tower.layers.{index}"
        renames = {
            "feed_forward1.ffw_layer_1.linear.weight": "ff1.layer1",
            "feed_forward1.ffw_layer_2.linear.weight": "ff1.layer2",
            "feed_forward1.pre_layer_norm.weight": "ff1.pre_norm.weight",
            "feed_forward1.post_layer_norm.weight": "ff1.post_norm.weight",
            "feed_forward2.ffw_layer_1.linear.weight": "ff2.layer1",
            "feed_forward2.ffw_layer_2.linear.weight": "ff2.layer2",
            "feed_forward2.pre_layer_norm.weight": "ff2.pre_norm.weight",
            "feed_forward2.post_layer_norm.weight": "ff2.post_norm.weight",
            "lconv1d.linear_start.linear.weight": "lconv.linear_start",
            "lconv1d.linear_end.linear.weight": "lconv.linear_end",
            "lconv1d.depthwise_conv1d.weight": "lconv.conv.weight",
            "lconv1d.pre_layer_norm.weight": "lconv.pre_norm.weight",
            "lconv1d.conv_norm.weight": "lconv.conv_norm.weight",
            "self_attn.q_proj.linear.weight": "attn.q_proj",
            "self_attn.k_proj.linear.weight": "attn.k_proj",
            "self_attn.v_proj.linear.weight": "attn.v_proj",
            "self_attn.post.linear.weight": "attn.post",
            "self_attn.relative_k_proj.weight": "attn.relative_k",
            "self_attn.per_dim_scale": "attn.per_dim_scale",
            "norm_pre_attn.weight": "norm_pre_attn.weight",
            "norm_post_attn.weight": "norm_post_attn.weight",
            "norm_out.weight": "norm_out.weight",
        }
        target = renames.get(leaf)
        if target is not None:
            copy(f"{prefix}.{target}", tensor)

    def input_embeddings(self, token_ids: tensorplay.Tensor) -> tensorplay.Tensor:
        """Look up text embeddings (used by multimodal row replacement)."""

        return self.model.embed_tokens(token_ids)

    def encode_images(
        self,
        pixel_values: tensorplay.Tensor,
        grid_thw: tensorplay.Tensor,
    ) -> tensorplay.Tensor:
        """Encode a concatenated patch batch into one row per soft token."""

        tower = self.vision_tower
        connector = self.embed_vision
        if tower is None or connector is None:
            raise Gemma4Error("this checkpoint has no vision tower")
        pixel_values = pixel_values.to(device=self.device, dtype=self.dtype)
        grid_thw = grid_thw.to(device=self.device)
        kernel = tower.config.pooling_kernel_size
        outputs: list[tensorplay.Tensor] = []
        offset = 0
        for grid in grid_thw.tolist():
            _, patch_h, patch_w = (int(value) for value in grid)
            rows = patch_h * patch_w
            patches = pixel_values[offset : offset + rows]
            offset += rows
            positions_y, positions_x = tensorplay.meshgrid(
                tensorplay.arange(patch_h, device=self.device),
                tensorplay.arange(patch_w, device=self.device),
                indexing="ij",
            )
            positions = tensorplay.stack(
                (positions_x.reshape(-1), positions_y.reshape(-1)), dim=-1
            )
            features, _ = tower(
                patches,
                positions,
                output_length=(patch_h // kernel) * (patch_w // kernel),
            )
            outputs.append(connector(features))
        if offset != int(pixel_values.shape[0]):
            raise Gemma4Error("vision patch rows disagree with grid metadata")
        return tensorplay.cat(outputs, dim=0)

    def encode_audios(
        self,
        input_features: tensorplay.Tensor,
        feature_lengths: Sequence[int],
    ) -> tuple[tensorplay.Tensor, tuple[int, ...]]:
        """Encode batched mel features into projected audio soft tokens."""

        tower = self.audio_tower
        connector = self.embed_audio
        if tower is None or connector is None:
            raise Gemma4Error("this checkpoint has no audio tower")
        # Processors batch mel banks as [batch, bins, frames]; the tower takes
        # [batch, frames, bins].
        input_features = input_features.to(device=self.device, dtype=self.dtype)
        frames = input_features.transpose(1, 2)
        outputs: list[tensorplay.Tensor] = []
        counts: list[int] = []
        for index, length in enumerate(feature_lengths):
            mask = tensorplay.arange(frames.shape[1], device=self.device) < length
            encoded, surviving = tower(
                frames[index : index + 1],
                mask.unsqueeze(0),
            )
            count = int(surviving.sum().item()) if surviving is not None else 0
            if count <= 0:
                raise Gemma4Error("audio encoder produced no valid frames")
            outputs.append(connector(encoded[0, :count]))
            counts.append(count)
        return tensorplay.cat(outputs, dim=0), tuple(counts)

    @classmethod
    def from_pretrained(
        cls,
        model_path: str | Path,
        *,
        device: tensorplay.device | str = "cuda",
        dtype: tensorplay.dtype = tensorplay.float16,
        kernel_policy: KernelPolicy,
        kernel_bindings: KernelBindings | None = None,
        enable_mtp: bool = False,
        mtp_model_path: str | Path | None = None,
    ) -> tuple[Gemma4ForCausalLM, WeightLoadReport]:
        parsed = Gemma4Config.from_path(model_path)
        mtp_config = None
        if enable_mtp:
            from megalm.models.gemma4.config import Gemma4AssistantConfig

            assistant_path = Path(mtp_model_path) if mtp_model_path else None
            if assistant_path is not None:
                assistant_payload = json.loads(
                    (assistant_path / "config.json").read_text(encoding="utf-8")
                )
                mtp_config = Gemma4AssistantConfig.from_dict(assistant_payload)
        token_ids = {
            "image_token_id": parsed.image_token_id,
            "video_token_id": parsed.video_token_id,
            "audio_token_id": parsed.audio_token_id,
        }
        model = cls(
            parsed.text,
            device=device,
            dtype=dtype,
            kernel_policy=kernel_policy,
            kernel_bindings=kernel_bindings,
            vision_config=parsed.vision,
            audio_config=parsed.audio,
            token_ids=token_ids,
            enable_mtp=enable_mtp,
            mtp_config=mtp_config,
        )
        report = model.load_weights(model_path)
        if model.mtp is not None and assistant_path is not None:
            from megalm.models.gemma4.mtp import load_gemma4_mtp_weights

            load_gemma4_mtp_weights(model.mtp, str(assistant_path))
            model.bind_mtp_target_kv_slots()
        model.eval()
        return model, report

    def load_weights(
        self,
        model_path: str | Path,
        *,
        source: SafetensorsSource | None = None,
        key_prefix: str = "",
        strict: bool = True,
    ) -> WeightLoadReport:
        parameters = dict(self.named_parameters())
        buffers = dict(self.named_buffers())
        loaded: set[str] = set()
        packed: dict[str, set[str]] = {}

        def copy(target_name: str, tensor: tensorplay.Tensor) -> None:
            target = parameters.get(target_name)
            if target is None:
                target = buffers.get(target_name)
            if target is None:
                raise Gemma4Error(
                    f"checkpoint tensor has no native target: {target_name}"
                )
            if tuple(target.shape) != tuple(tensor.shape):
                raise Gemma4Error(
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
                raise Gemma4Error(f"shape mismatch for packed {source_name}")
            target.copy_(tensor.to(device=self.device, dtype=self.dtype))

        @tensorplay.no_grad()
        def consume(name: str, tensor: tensorplay.Tensor) -> None:
            if key_prefix:
                if not name.startswith(key_prefix):
                    return
                name = name[len(key_prefix) :]
            # Full multimodal checkpoints wrap the text model under
            # ``model.language_model``; text-only checkpoints use ``model``.
            if name.startswith("model.language_model."):
                name = name[len("language_model.") :]
            elif name.startswith("language_model."):
                name = name[len("language_model.") :]
            if name.startswith("model.vision_tower.") and self.vision_tower is not None:
                self._load_vision_tensor(
                    name[len("model.vision_tower.") :], tensor, copy, copy_slice, loaded
                )
                return
            if name.startswith("model.audio_tower.") and self.audio_tower is not None:
                self._load_audio_tensor(
                    name[len("model.audio_tower.") :], tensor, copy
                )
                return
            if name == "model.embed_vision.embedding_projection.weight" and self.embed_vision is not None:
                copy("embed_vision.projection", tensor)
                return
            if name == "model.embed_audio.embedding_projection.weight" and self.embed_audio is not None:
                copy("embed_audio.projection", tensor)
                return
            if name == "lm_head.weight" and self.text_config.tie_word_embeddings:
                return
            if name == "lm_head.weight":
                copy("lm_head_weight", tensor)
                return
            if name.startswith(("vision_tower.", "audio_tower.", "embed_vision.", "embed_audio.")):
                return
            if "confidence_head" in name:
                return
            if name == "model.embed_tokens.weight":
                copy("model.embed_tokens", tensor)
                return
            if name == "model.embed_tokens_per_layer.weight":
                copy("model.embed_tokens_per_layer", tensor)
                return
            if name == "model.per_layer_model_projection.weight":
                copy("model.per_layer_model_projection", tensor)
                return
            if name == "model.per_layer_projection_norm.weight":
                copy("model.per_layer_projection_norm.weight", tensor)
                return
            if name == "model.norm.weight":
                copy("model.norm.weight", tensor)
                return
            parts = name.split(".")
            if len(parts) < 4 or parts[0] != "model" or parts[1] != "layers":
                # KV scale metadata and other non-parameter tensors are handled
                # by the loader before the consumer sees them.
                return
            try:
                layer_index = int(parts[2])
            except ValueError:
                return
            leaf = ".".join(parts[3:])
            if leaf.startswith("mlp."):
                leaf = leaf[len("mlp.") :]
            layer = self.model.layers[layer_index]
            attention = layer.self_attn
            if leaf.startswith("self_attn."):
                field = leaf[len("self_attn.") :]
                target = f"model.layers.{layer_index}.self_attn.{field[:-7]}"
                if attention.qkv_proj is not None and field in {
                    "q_proj.weight", "k_proj.weight", "v_proj.weight",
                }:
                    part = field.split(".")[0]
                    if part == "v_proj" and attention.use_k_eq_v:
                        return
                    qw = attention.query_width
                    kw = attention.kv_width
                    if part == "q_proj":
                        segment = attention.qkv_proj[:qw]
                    elif part == "k_proj":
                        segment = attention.qkv_proj[qw : qw + kw]
                    else:
                        segment = attention.qkv_proj[qw + kw : qw + 2 * kw]
                    copy_slice(segment, tensor, name)
                    qkv_key = f"model.layers.{layer_index}.self_attn.qkv_proj"
                    packed.setdefault(qkv_key, set()).add(part)
                    expected = (
                        {"q_proj", "k_proj"} if attention.use_k_eq_v
                        else {"q_proj", "k_proj", "v_proj"}
                    )
                    if packed[qkv_key] == expected:
                        loaded.add(qkv_key)
                    return
                if field == "q_proj.weight":
                    copy(target, tensor)
                elif field == "o_proj.weight":
                    copy(target, tensor)
                elif field == "q_norm.weight":
                    copy(target, tensor)
                elif field == "k_norm.weight" and attention.k_norm is not None:
                    copy(target, tensor)
                elif field == "k_proj.weight" and attention.k_proj is not None:
                    copy(target, tensor)
                elif field == "v_proj.weight" and attention.v_proj is not None:
                    copy(target, tensor)
                return
            if leaf in {
                "input_layernorm.weight",
                "post_attention_layernorm.weight",
                "pre_feedforward_layernorm.weight",
                "post_feedforward_layernorm.weight",
                "post_per_layer_input_norm.weight",
                "post_feedforward_layernorm_1.weight",
                "post_feedforward_layernorm_2.weight",
                "pre_feedforward_layernorm_2.weight",
                "layer_scalar",
            }:
                copy(f"model.layers.{layer_index}.{leaf}", tensor)
                return
            if leaf in {"per_layer_input_gate.weight", "per_layer_projection.weight"}:
                copy(
                    f"model.layers.{layer_index}.{leaf[: -len('.weight')]}",
                    tensor,
                )
                return
            if leaf in {"gate_proj.weight", "up_proj.weight"}:
                intermediate = layer.mlp_intermediate
                offset = 0 if leaf.startswith("gate") else intermediate
                copy_slice(
                    layer.gate_up_weight[offset : offset + intermediate],
                    tensor,
                    name,
                )
                key = f"model.layers.{layer_index}.gate_up_weight"
                packed.setdefault(key, set()).add(leaf.split(".")[0])
                if packed[key] == {"gate_proj", "up_proj"}:
                    loaded.add(key)
                return
            if leaf == "down_proj.weight":
                copy(f"model.layers.{layer_index}.down_weight", tensor)
                return
            if layer.moe is not None:
                moe = layer.moe
                if leaf == "router.proj.weight":
                    copy(f"model.layers.{layer_index}.moe.router_proj", tensor)
                    return
                if leaf == "router.scale":
                    copy(f"model.layers.{layer_index}.moe.router_scale", tensor)
                    return
                if leaf in {"router.per_expert_scale", "moe.per_expert_scale"}:
                    copy(f"model.layers.{layer_index}.moe.per_expert_scale", tensor)
                    return
                if leaf == "experts.gate_up_proj":
                    copy(f"model.layers.{layer_index}.moe.gate_up_weight", tensor)
                    return
                if leaf == "experts.down_proj":
                    copy(f"model.layers.{layer_index}.moe.down_weight", tensor)
                    return

        if source is None:
            source = SafetensorsSource(model_path)
        report = source.stream(consume)
        if strict:
            # ``state_dict`` excludes non-persistent buffers (RoPE tables),
            # which are computed locally rather than restored.
            persistent = set(self.state_dict())
            missing = sorted(set(parameters) - loaded)
            missing_buffer = sorted(
                name for name in set(buffers) & persistent if name not in loaded
            )
            if missing or missing_buffer:
                raise Gemma4Error(
                    "checkpoint did not initialize parameters: "
                    f"{missing + missing_buffer}"
                )
        return report

    def lm_head(self) -> tensorplay.Tensor:
        lm_head = (
            self.model.embed_tokens
            if self.text_config.tie_word_embeddings
            else self.lm_head_weight
        )
        assert lm_head is not None
        return lm_head

    def bind_mtp_target_kv_slots(self) -> None:
        """Map each assistant layer onto the target's physical KV slot."""
        if self.mtp is None:
            return
        config = self.text_config
        first_shared = config.num_hidden_layers - config.num_kv_shared_layers
        owners: dict[str, int] = {}
        for index in range(first_shared):
            owners[config.layer_types[index]] = index
        slots = []
        for kind in self.mtp.text_config.layer_types:
            if kind not in owners:
                raise Gemma4Error(f"no target KV owner for assistant layer type {kind!r}")
            slots.append(config.kv_slot(owners[kind]))
        self.mtp.bind_target_kv_slots(tuple(slots))

    def forward(
        self,
        batch: ExecutionBatch,
        cache: PagedKVCache,
    ) -> tensorplay.Tensor:
        hidden_states = self.model(batch, cache)
        self._capture_mtp_hidden(hidden_states)
        lm_head = self.lm_head()
        if batch.all_decode:
            logits = F.linear(hidden_states, lm_head)
            return self._softcap(logits)
        indices = batch.sample_indices.clamp_min(0).long()
        logits = F.linear(hidden_states.index_select(0, indices), lm_head)
        logits = self._softcap(logits)
        return tensorplay.where(
            (batch.sample_indices >= 0)[:, None], logits, 0
        )

    def _softcap(self, logits: tensorplay.Tensor) -> tensorplay.Tensor:
        cap = self.text_config.final_logit_softcapping
        if cap is None:
            return logits
        return cap * tensorplay.tanh(logits / cap)

    def warmup_prefill_kernels(self, *, max_batched_tokens: int) -> tuple[int, ...]:
        if max_batched_tokens <= 0:
            raise Gemma4Error("prefill warmup token budget must be positive")
        return ()

    def configure_moe_workspace(self, max_batched_tokens: int) -> int:
        if max_batched_tokens <= 0:
            raise Gemma4Error("workspace token budget must be positive")
        return 0


__all__ = [
    "Gemma4Attention",
    "Gemma4DecoderLayer",
    "Gemma4Error",
    "Gemma4ForCausalLM",
    "Gemma4TextModel",
]
