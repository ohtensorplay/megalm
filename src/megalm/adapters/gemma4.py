"""Gemma-4 adapter: checkpoint facts only, no scheduling policy."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from megalm.contracts.model import (
    AttentionKind,
    FeedForwardKind,
    LayerSpec,
    ModelContractError,
    ModelSpec,
    ParallelConstraints,
    PositionKind,
    PositionSchema,
    StateComponentSpec,
    StateKind,
    VisionSpec,
    WeightMappingRule,
)
from megalm.models.gemma4.config import (
    FULL_ATTENTION,
    SLIDING_ATTENTION,
    Gemma4Config,
    Gemma4ConfigError,
)

_MODEL_TYPES = {"gemma4", "gemma4_unified", "gemma4_text", "gemma4_unified_text"}
_ARCHITECTURES = {
    "gemma4": "Gemma4ForConditionalGeneration",
    "gemma4_unified": "Gemma4UnifiedForConditionalGeneration",
    "gemma4_text": "Gemma4ForCausalLM",
    "gemma4_unified_text": "Gemma4UnifiedForCausalLM",
}


def load_gemma4_model_spec(model_path: str | Path) -> ModelSpec:
    path = Path(model_path)
    config_path = path / "config.json" if path.is_dir() else path
    try:
        payload: Any = json.loads(config_path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ModelContractError(f"missing model config: {config_path}") from exc
    except json.JSONDecodeError as exc:
        raise ModelContractError(f"invalid JSON model config: {config_path}") from exc
    if not isinstance(payload, dict):
        raise ModelContractError("model config must be a JSON object")
    return gemma4_model_spec_from_config(payload)


def gemma4_model_spec_from_config(config: dict[str, Any]) -> ModelSpec:
    try:
        parsed = Gemma4Config.from_dict(config)
    except Gemma4ConfigError as exc:
        raise ModelContractError(str(exc)) from exc
    text = parsed.text
    layers: list[LayerSpec] = []
    for index in range(text.num_hidden_layers):
        full = text.layer_types[index] == FULL_ATTENTION
        layers.append(
            LayerSpec(
                index=index,
                attention=AttentionKind.FULL,
                feed_forward=(
                    FeedForwardKind.MOE if text.enable_moe_block else FeedForwardKind.DENSE
                ),
                hidden_size=text.hidden_size,
                intermediate_size=text.mlp_intermediate(index),
                num_attention_heads=text.num_attention_heads,
                num_key_value_heads=text.layer_num_key_value_heads[index],
                head_dim=text.layer_head_dims[index],
                num_experts=text.num_experts if text.enable_moe_block else 0,
                experts_per_token=text.top_k_experts if text.enable_moe_block else 0,
                sliding_window=None if full else text.sliding_window,
            )
        )
    tp_facts = [
        text.hidden_size,
        text.num_attention_heads,
        text.max_kv_heads,
        text.max_head_dim,
        text.vocab_size,
        text.intermediate_size,
    ]
    if text.enable_moe_block:
        tp_facts.extend((text.num_experts, text.moe_intermediate_size))
    supported_tp = tuple(
        degree
        for degree in (1, 2, 4, 8)
        if all(value % degree == 0 for value in tp_facts)
    )
    architectures = config.get("architectures", ())
    architecture = (
        str(architectures[0])
        if isinstance(architectures, list) and architectures
        else _ARCHITECTURES[text.model_type]
    )
    # The native model keeps two RoPE flavours; the contract advertises the
    # global one, which is the one used by the KV-owning full layers.
    rope_kind = FULL_ATTENTION if FULL_ATTENTION in text.rope_theta else SLIDING_ATTENTION
    vision = None
    if parsed.vision is not None:
        vision = VisionSpec(
            processor_type="gemma4",
            patch_size=parsed.vision.patch_size * parsed.vision.pooling_kernel_size,
            merge_size=parsed.vision.pooling_kernel_size,
            hidden_size=parsed.vision.hidden_size,
            intermediate_size=parsed.vision.intermediate_size,
            depth=parsed.vision.num_hidden_layers,
            num_heads=parsed.vision.num_attention_heads,
            output_hidden_size=parsed.vision.mm_embed_dim or text.hidden_size,
            num_position_embeddings=parsed.vision.position_embedding_size,
            image_token_id=parsed.image_token_id,
            vision_start_token_id=parsed.boi_token_id,
            vision_end_token_id=parsed.eoi_token_id,
        )
    return ModelSpec(
        architecture=architecture,
        model_type=str(config.get("model_type") or text.model_type),
        vocab_size=text.vocab_size,
        hidden_size=text.hidden_size,
        layers=tuple(layers),
        position=PositionSchema(
            kind=PositionKind.ROPE,
            max_positions=text.max_position_embeddings,
            rope_theta=text.rope_theta[rope_kind],
            head_dim=text.max_head_dim,
        ),
        state_schema=(
            StateComponentSpec(
                kind=StateKind.PAGED_KV,
                layer_indices=tuple(text.kv_layer_indices),
                checkpoint_alignment=1,
            ),
        ),
        weight_mapping=(
            WeightMappingRule(
                source_pattern=r"(?P<name>.+)",
                target_template=r"\g<name>",
            ),
        ),
        parallel=ParallelConstraints(
            supported_tp_degrees=supported_tp,
            minimum_compute_capability=(8, 0),
        ),
        supported_dtypes=("float16", "bfloat16"),
        tie_word_embeddings=text.tie_word_embeddings,
        vision=vision,
        final_logit_softcapping=text.final_logit_softcapping,
        qk_norm=True,
        embedding_multiplier=text.hidden_size**0.5,
    )


__all__ = ["gemma4_model_spec_from_config", "load_gemma4_model_spec"]
