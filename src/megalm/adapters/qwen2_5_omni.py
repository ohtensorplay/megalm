from __future__ import annotations

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


def qwen2_5_omni_model_spec_from_config(
    config: dict[str, Any],
) -> ModelSpec:
    if config.get("model_type") != "qwen2_5_omni":
        raise ModelContractError("checkpoint is not Qwen2.5-Omni")
    thinker = _object(config, "thinker_config")
    text = _object(thinker, "text_config")
    vision = _object(thinker, "vision_config")
    if thinker.get("model_type") != "qwen2_5_omni_thinker":
        raise ModelContractError("Qwen2.5-Omni thinker config is invalid")
    hidden_size = _integer(text, "hidden_size")
    num_layers = _integer(text, "num_hidden_layers")
    attention_heads = _integer(text, "num_attention_heads")
    kv_heads = _integer(text, "num_key_value_heads")
    head_dim = hidden_size // attention_heads
    intermediate_size = _integer(text, "intermediate_size")
    if hidden_size != attention_heads * head_dim:
        raise ModelContractError("Omni text attention geometry is invalid")
    rope = text.get("rope_scaling", {})
    if not isinstance(rope, dict):
        raise ModelContractError("Omni rope_scaling must be an object")
    sections = rope.get("mrope_section")
    if not isinstance(sections, list):
        raise ModelContractError("Omni requires MRoPE sections")
    layers = tuple(
        LayerSpec(
            index=index,
            attention=AttentionKind.FULL,
            feed_forward=FeedForwardKind.DENSE,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            num_attention_heads=attention_heads,
            num_key_value_heads=kv_heads,
            head_dim=head_dim,
        )
        for index in range(num_layers)
    )
    vocab_size = _integer(text, "vocab_size")
    supported_tp = tuple(
        degree
        for degree in (1, 2, 4, 8)
        if all(
            value % degree == 0
            for value in (
                hidden_size,
                intermediate_size,
                attention_heads,
                kv_heads,
                vocab_size,
            )
        )
    )
    return ModelSpec(
        architecture="Qwen2_5OmniThinkerForConditionalGeneration",
        model_type="qwen2_5_omni",
        vocab_size=vocab_size,
        hidden_size=hidden_size,
        layers=layers,
        position=PositionSchema(
            kind=PositionKind.MROPE,
            max_positions=_integer(text, "max_position_embeddings"),
            rope_theta=float(text.get("rope_theta", 1_000_000.0)),
            head_dim=head_dim,
            mrope_sections=tuple(int(value) for value in sections),
            mrope_interleaved=False,
        ),
        state_schema=(
            StateComponentSpec(
                StateKind.PAGED_KV,
                tuple(range(num_layers)),
            ),
            StateComponentSpec(StateKind.VISION_EMBEDDING),
            StateComponentSpec(StateKind.MEDIA_POSITION),
        ),
        weight_mapping=(
            WeightMappingRule(
                source_pattern=r"thinker\.(?P<name>.+)",
                target_template=r"\g<name>",
            ),
        ),
        parallel=ParallelConstraints(
            supported_tp_degrees=supported_tp,
            minimum_compute_capability=(8, 0),
        ),
        supported_dtypes=("float16", "bfloat16"),
        tie_word_embeddings=bool(text.get("tie_word_embeddings", False)),
        qkv_bias=True,
        vision=VisionSpec(
            processor_type="qwen2_5_omni",
            patch_size=_integer(vision, "patch_size"),
            merge_size=_integer(vision, "spatial_merge_size"),
            temporal_patch_size=_integer(
                vision,
                "temporal_patch_size",
            ),
            image_token_id=_integer(
                thinker,
                "image_token_index",
                allow_zero=True,
            ),
            vision_start_token_id=_integer(
                thinker,
                "vision_start_token_id",
                allow_zero=True,
            ),
            vision_end_token_id=_integer(
                thinker,
                "vision_end_token_id",
                allow_zero=True,
            ),
        ),
    )


def _object(config: dict[str, Any], key: str) -> dict[str, Any]:
    value = config.get(key)
    if not isinstance(value, dict):
        raise ModelContractError(f"Omni config requires object {key!r}")
    return value


def _integer(
    config: dict[str, Any],
    key: str,
    *,
    allow_zero: bool = False,
) -> int:
    try:
        value = int(config[key])
    except (KeyError, TypeError, ValueError) as exc:
        raise ModelContractError(
            f"Omni config requires integer {key!r}"
        ) from exc
    if value < (0 if allow_zero else 1):
        raise ModelContractError(f"Omni config {key!r} is invalid")
    return value


__all__ = ["qwen2_5_omni_model_spec_from_config"]
