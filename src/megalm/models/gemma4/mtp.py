"""Gemma-4 multi-token prediction (assistant) model.

The assistant is a lightweight decoder that reads K/V state from the target
model's paged cache.  Every assistant layer is KV-shared: it owns only a Q
projection, a Q norm and an MLP.  At runtime each layer is bound to the target
cache slot of the last non-shared target layer of the same attention type.

Only the Gemma-specific layers and weight names live here; the hidden buffer,
embedding scaling and masked-logit head are shared components.
"""

from __future__ import annotations

import tensorplay
import tensorplay.nn.functional as F
from tensorplay import nn

from megalm.kernels.bindings import KernelBindings
from megalm.models.components.mtp import MaskedLogitsEmbedder, MTPProjector
from megalm.models.components.parameters import frozen_parameter
from megalm.models.gemma4.attention import Gemma4Attention, Gemma4RotaryCache, gelu_tanh
from megalm.models.gemma4.config import (
    FULL_ATTENTION,
    Gemma4AssistantConfig,
    Gemma4TextConfig,
)
from megalm.models.gemma4.model import Gemma4RMSNorm
from megalm.runtime.batch import ExecutionBatch
from megalm.runtime.cache import PagedKVCache


class Gemma4MTPDecoderLayer(nn.Module):
    """One Q-only assistant decoder layer."""

    def __init__(
        self,
        config: Gemma4TextConfig,
        layer_index: int,
        *,
        rotary: dict[str, Gemma4RotaryCache],
        kernel_bindings: KernelBindings,
        device: tensorplay.device,
        dtype: tensorplay.dtype,
    ) -> None:
        super().__init__()
        self.layer_index = layer_index
        kind = config.layer_types[layer_index]
        self.self_attn = Gemma4Attention(
            layer_index=layer_index,
            head_dim=config.layer_head_dims[layer_index],
            num_heads=config.num_attention_heads,
            num_kv_heads=config.layer_num_key_value_heads[layer_index],
            owns_kv=False,
            use_k_eq_v=False,
            sliding_window=config.sliding_window if kind != FULL_ATTENTION else None,
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
        intermediate = config.intermediate_size
        self.gate_up_weight = frozen_parameter(
            (2 * intermediate, config.hidden_size), device=device, dtype=dtype
        )
        self.down_weight = frozen_parameter(
            (config.hidden_size, intermediate), device=device, dtype=dtype
        )
        self.register_buffer(
            "layer_scalar", tensorplay.ones((1,), dtype=dtype, device=device)
        )

    def forward(
        self,
        hidden_states: tensorplay.Tensor,
        *,
        batch: ExecutionBatch,
        cache: PagedKVCache,
        kv_slot: int,
    ) -> tensorplay.Tensor:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn(
            hidden_states, batch=batch, cache=cache, kv_slot=kv_slot
        )
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = hidden_states + residual
        residual = hidden_states
        hidden_states = self.pre_feedforward_layernorm(hidden_states)
        gate_up = F.linear(hidden_states, self.gate_up_weight)
        gate, up = gate_up.chunk(2, dim=-1)
        hidden_states = F.linear(gelu_tanh(gate) * up, self.down_weight)
        hidden_states = self.post_feedforward_layernorm(hidden_states)
        hidden_states = hidden_states + residual
        return hidden_states * self.layer_scalar


class Gemma4MultiTokenPredictor(MTPProjector):
    """Checkpoint-native Gemma-4 assistant.

    ``forward`` returns ``(draft_hidden, backbone_hidden)``: the first is in
    the assistant's hidden size and is used for draft logits; the second is
    projected back to the backbone width and fed to the next draft step.
    """

    def __init__(
        self,
        config: Gemma4AssistantConfig,
        *,
        backbone_embed_tokens: tensorplay.Tensor,
        kernel_bindings: KernelBindings,
        device: tensorplay.device,
        dtype: tensorplay.dtype,
    ) -> None:
        text = config.text_config
        masked = (
            MaskedLogitsEmbedder(
                hidden_size=text.hidden_size,
                vocab_size=text.vocab_size,
                num_centroids=config.num_centroids,
                top_k=config.centroid_intermediate_top_k,
                device=device,
                dtype=dtype,
            )
            if config.use_ordered_embeddings
            else None
        )
        super().__init__(
            backbone_embed_tokens=backbone_embed_tokens,
            backbone_hidden_size=config.backbone_hidden_size,
            dtype=dtype,
            device=device,
            masked_embedder=masked,
        )
        self.config = config
        self.text_config = text
        self.hidden_size = text.hidden_size
        self.pre_projection = frozen_parameter(
            (self.hidden_size, 2 * self.backbone_hidden_size),
            device=device, dtype=dtype,
        )
        self.post_projection = frozen_parameter(
            (self.backbone_hidden_size, self.hidden_size),
            device=device, dtype=dtype,
        )
        self.rotary: dict[str, Gemma4RotaryCache] = {}
        for kind in dict.fromkeys(text.layer_types):
            self.rotary[kind] = Gemma4RotaryCache(
                head_dim=next(
                    dim
                    for dim, layer_kind in zip(text.layer_head_dims, text.layer_types)
                    if layer_kind == kind
                ),
                partial=text.rope_partial_factor[kind],
                theta=text.rope_theta[kind],
                rope_type=text.rope_type[kind],
                max_positions=text.max_position_embeddings,
                device=device,
            )
        self.layers = nn.ModuleList(
            [
                Gemma4MTPDecoderLayer(
                    text, index,
                    rotary=self.rotary,
                    kernel_bindings=kernel_bindings,
                    device=device,
                    dtype=dtype,
                )
                for index in range(text.num_hidden_layers)
            ]
        )
        self.norm = Gemma4RMSNorm(
            text.hidden_size, text.rms_norm_eps,
            device=device, dtype=dtype, kernel_bindings=kernel_bindings,
        )
        # Physical target cache slot for each assistant layer.  All assistant
        # layers read K/V from the target; this is populated by
        # ``bind_target_kv_slots`` once the target model is known.
        self.target_kv_slots: tuple[int, ...] = tuple(
            range(text.num_hidden_layers)
        )

    def bind_target_kv_slots(self, slots: tuple[int, ...]) -> None:
        if len(slots) != len(self.layers):
            raise ValueError("one target KV slot is required per assistant layer")
        self.target_kv_slots = tuple(slots)

    def forward(
        self,
        batch: ExecutionBatch,
        target_hidden_states: tensorplay.Tensor,
        cache: PagedKVCache,
    ) -> tuple[tensorplay.Tensor, tensorplay.Tensor]:
        embeddings = self.embed_input_ids(batch.input_ids)
        if embeddings.shape[-1] != self.backbone_hidden_size:
            raise ValueError("assistant embedding width must match the backbone")
        if target_hidden_states.shape != embeddings.shape:
            raise ValueError("target hidden states must align one-to-one with tokens")
        combined = tensorplay.cat((embeddings, target_hidden_states), dim=-1)
        hidden = F.linear(combined, self.pre_projection)
        for layer, slot in zip(self.layers, self.target_kv_slots):
            hidden = layer(hidden, batch=batch, cache=cache, kv_slot=slot)
        draft_hidden = self.norm(hidden)
        backbone_hidden = F.linear(draft_hidden, self.post_projection)
        return draft_hidden, backbone_hidden


def load_gemma4_mtp_weights(
    predictor: Gemma4MultiTokenPredictor,
    model_path: str,
    *,
    prefix: str = "mtp",
) -> set[str]:
    """Load an assistant checkpoint into a native predictor.

    The assistant's ``model.embed_tokens`` is ignored: the predictor shares the
    target model's backbone embedding, as the reference proposer does.
    Returns the set of populated parameter names.
    """

    from megalm.loading import SafetensorsSource

    parameters = dict(predictor.named_parameters())
    buffers = dict(predictor.named_buffers())
    loaded: set[str] = set()
    packed: dict[str, set[str]] = {}

    def copy(target_name: str, tensor: tensorplay.Tensor) -> None:
        target = parameters.get(target_name)
        if target is None:
            target = buffers.get(target_name)
        if target is None:
            raise ValueError(f"assistant tensor has no native target: {target_name}")
        if tuple(target.shape) != tuple(tensor.shape):
            raise ValueError(
                f"shape mismatch for {target_name}: expected {tuple(target.shape)}, "
                f"got {tuple(tensor.shape)}"
            )
        target.copy_(tensor.to(device=target.device, dtype=target.dtype))
        loaded.add(target_name)

    @tensorplay.no_grad()
    def consume(name: str, tensor: tensorplay.Tensor) -> None:
        if name == "model.embed_tokens.weight":
            return
        target = name
        if name.startswith("model.layers."):
            parts = name.split(".")
            try:
                layer = int(parts[2])
            except ValueError:
                return
            leaf = ".".join(parts[3:])
            if leaf.startswith("self_attn."):
                field = leaf[len("self_attn.") :]
                if field in {"q_proj.weight", "o_proj.weight", "q_norm.weight"}:
                    target = f"{prefix}.layers.{layer}.self_attn.{field[:-7]}"
                else:
                    return
            elif leaf in {
                "input_layernorm.weight",
                "post_attention_layernorm.weight",
                "pre_feedforward_layernorm.weight",
                "post_feedforward_layernorm.weight",
                "layer_scalar",
            }:
                target = f"{prefix}.layers.{layer}.{leaf}"
            elif leaf in {"gate_proj.weight", "up_proj.weight"}:
                intermediate = predictor.text_config.intermediate_size
                offset = 0 if leaf.startswith("gate") else intermediate
                target = f"{prefix}.layers.{layer}.gate_up_weight"
                packed_target = parameters[target][offset : offset + intermediate]
                if tuple(packed_target.shape) != tuple(tensor.shape):
                    raise ValueError(f"shape mismatch for {name}")
                packed_target.copy_(
                    tensor.to(device=packed_target.device, dtype=packed_target.dtype)
                )
                packed.setdefault(target, set()).add(leaf.split(".")[0])
                if packed[target] == {"gate_proj", "up_proj"}:
                    loaded.add(target)
                return
            elif leaf == "down_proj.weight":
                target = f"{prefix}.layers.{layer}.down_weight"
            else:
                return
        elif name in {"pre_projection.weight", "post_projection.weight"}:
            target = f"{prefix}.{name[:-7]}"
        elif name == "model.norm.weight":
            target = f"{prefix}.norm.weight"
        elif name in {
            "masked_embedding.centroids.weight",
            "masked_embedding.token_ordering",
        }:
            target = f"{prefix}.masked_embedder."
            target += name[len("masked_embedding.") :]
        else:
            return
        copy(target, tensor)

    SafetensorsSource(model_path).stream(consume)
    return loaded


__all__ = [
    "Gemma4MTPDecoderLayer",
    "Gemma4MultiTokenPredictor",
    "load_gemma4_mtp_weights",
]
