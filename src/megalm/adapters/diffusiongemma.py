"""DiffusionGemma adapter: checkpoint facts for the block-diffusion executor."""

from __future__ import annotations

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
    WeightMappingRule,
)
from megalm.models.diffusiongemma.config import DiffusionGemmaConfig
from megalm.models.gemma4.config import FULL_ATTENTION, Gemma4ConfigError


def diffusion_gemma_model_spec_from_config(
    config: dict[str, Any],
) -> ModelSpec:
    try:
        parsed = DiffusionGemmaConfig.from_dict(config)
    except Gemma4ConfigError as exc:
        raise ModelContractError(str(exc)) from exc
    return _spec(parsed)


def load_diffusion_gemma_model_spec(model_path: str | Path) -> ModelSpec:
    """Load the spec from a directory, inferring the KV-sharing layout."""

    try:
        parsed = DiffusionGemmaConfig.from_path(model_path)
    except Gemma4ConfigError as exc:
        raise ModelContractError(str(exc)) from exc
    return _spec(parsed)


def _spec(parsed: DiffusionGemmaConfig) -> ModelSpec:
    text = parsed.text
    layers: list[LayerSpec] = []
    for index in range(text.num_hidden_layers):
        full = text.layer_types[index] == FULL_ATTENTION
        layers.append(
            LayerSpec(
                index=index,
                attention=AttentionKind.FULL,
                feed_forward=FeedForwardKind.MOE,
                hidden_size=text.hidden_size,
                intermediate_size=text.moe_intermediate_size,
                num_attention_heads=text.num_attention_heads,
                num_key_value_heads=text.layer_num_key_value_heads[index],
                head_dim=text.layer_head_dims[index],
                num_experts=text.num_experts,
                experts_per_token=text.top_k_experts,
                sliding_window=None if full else text.sliding_window,
            )
        )
    tp_facts = [
        text.hidden_size,
        text.num_attention_heads,
        text.max_kv_heads,
        text.max_head_dim,
        text.vocab_size,
        text.num_experts,
        text.moe_intermediate_size,
    ]
    supported_tp = tuple(
        degree
        for degree in (1, 2, 4, 8)
        if all(value % degree == 0 for value in tp_facts)
    )
    # Every layer owns its own KV: DiffusionGemma has no YOCO sharing, so the
    # encoder cache is a plain per-layer paged KV.
    state = StateComponentSpec(
        kind=StateKind.PAGED_KV,
        layer_indices=tuple(range(text.num_hidden_layers)),
        checkpoint_alignment=1,
    )
    return ModelSpec(
        architecture="DiffusionGemmaForBlockDiffusion",
        model_type="diffusion_gemma",
        vocab_size=text.vocab_size,
        hidden_size=text.hidden_size,
        layers=tuple(layers),
        position=PositionSchema(
            kind=PositionKind.ROPE,
            max_positions=text.max_position_embeddings,
            rope_theta=text.rope_theta[FULL_ATTENTION],
            head_dim=text.max_head_dim,
            partial_rotary_factor=text.rope_partial_factor[FULL_ATTENTION],
        ),
        state_schema=(state,),
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
        final_logit_softcapping=text.final_logit_softcapping,
        qk_norm=True,
        embedding_multiplier=text.hidden_size**0.5,
    )


__all__ = [
    "diffusion_gemma_model_spec_from_config",
    "load_diffusion_gemma_model_spec",
]
