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

_MODEL_TYPES = {"qwen3_vl", "qwen3_vl_moe"}
_TEXT_MODEL_TYPES = {"qwen3_vl_text", "qwen3_vl_moe_text"}


def load_qwen3_vl_model_spec(model_path: str | Path) -> ModelSpec:
    path = Path(model_path)
    config_path = path / "config.json" if path.is_dir() else path
    try:
        payload = json.loads(config_path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ModelContractError(f"missing model config: {config_path}") from exc
    except json.JSONDecodeError as exc:
        raise ModelContractError(f"invalid JSON model config: {config_path}") from exc
    if not isinstance(payload, dict):
        raise ModelContractError("model config must be a JSON object")
    return qwen3_vl_model_spec_from_config(payload)


def qwen3_vl_model_spec_from_config(config: dict[str, Any]) -> ModelSpec:
    model_type = _required_str(config, "model_type")
    if model_type not in _MODEL_TYPES:
        raise ModelContractError(
            f"Qwen3-VL adapter does not support model_type={model_type!r}"
        )
    text = _required_object(config, "text_config")
    vision = _required_object(config, "vision_config")
    text_model_type = _required_str(text, "model_type")
    expected_text_type = (
        "qwen3_vl_moe_text" if model_type == "qwen3_vl_moe" else "qwen3_vl_text"
    )
    if text_model_type not in _TEXT_MODEL_TYPES or text_model_type != expected_text_type:
        raise ModelContractError(
            f"{model_type} requires text_config.model_type={expected_text_type!r}"
        )

    hidden_size = _required_int(text, "hidden_size")
    num_layers = _required_int(text, "num_hidden_layers")
    attention_heads = _required_int(text, "num_attention_heads")
    kv_heads = _required_int(text, "num_key_value_heads")
    head_dim = _required_int(text, "head_dim")
    if hidden_size != attention_heads * head_dim:
        raise ModelContractError(
            "hidden_size must equal num_attention_heads * head_dim"
        )
    if attention_heads % kv_heads:
        raise ModelContractError("full-attention head counts are not divisible")

    is_moe = model_type == "qwen3_vl_moe"
    dense_intermediate = 1 if is_moe else _required_int(text, "intermediate_size")
    moe_intermediate = _required_int(text, "moe_intermediate_size") if is_moe else 0
    num_experts = _required_int(text, "num_experts") if is_moe else 0
    experts_per_token = (
        _required_int(text, "num_experts_per_tok") if is_moe else 0
    )
    if is_moe and experts_per_token > num_experts:
        raise ModelContractError("num_experts_per_tok exceeds num_experts")

    layers = tuple(
        LayerSpec(
            index=index,
            attention=AttentionKind.FULL,
            feed_forward=FeedForwardKind.MOE if is_moe else FeedForwardKind.DENSE,
            hidden_size=hidden_size,
            intermediate_size=(
                moe_intermediate if is_moe else dense_intermediate
            ),
            num_attention_heads=attention_heads,
            num_key_value_heads=kv_heads,
            head_dim=head_dim,
            num_experts=num_experts,
            experts_per_token=experts_per_token,
        )
        for index in range(num_layers)
    )

    rope = text.get("rope_parameters", text.get("rope_scaling", {}))
    if not isinstance(rope, dict):
        raise ModelContractError("Qwen3-VL rope parameters must be an object")
    sections = rope.get("mrope_section")
    if not isinstance(sections, list):
        raise ModelContractError("Qwen3-VL requires mrope_section")
    sections = tuple(int(value) for value in sections)
    if len(sections) != 3:
        raise ModelContractError(
            "Qwen3-VL MRoPE requires exactly three sections"
        )
    rotary_dim = sum(sections) * 2
    if rotary_dim > head_dim:
        raise ModelContractError("MRoPE sections exceed head_dim")
    partial = float(rope.get("partial_rotary_factor", rotary_dim / head_dim))
    if abs(partial * head_dim - rotary_dim) > 1e-6:
        raise ModelContractError(
            "MRoPE sections do not cover the partial rotary dimension"
        )

    vision_spec = VisionSpec(
        processor_type="qwen3_vl",
        patch_size=_required_int(vision, "patch_size"),
        merge_size=_required_int(vision, "spatial_merge_size"),
        temporal_patch_size=_required_int(vision, "temporal_patch_size"),
        in_channels=_required_int(vision, "in_channels"),
        depth=_required_int(vision, "depth"),
        hidden_size=_required_int(vision, "hidden_size"),
        intermediate_size=_required_int(vision, "intermediate_size"),
        num_heads=_required_int(vision, "num_heads"),
        output_hidden_size=_required_int(vision, "out_hidden_size"),
        num_position_embeddings=_required_int(
            vision, "num_position_embeddings"
        ),
        image_token_id=_required_int(config, "image_token_id", allow_zero=True),
        vision_start_token_id=_required_int(
            config, "vision_start_token_id", allow_zero=True
        ),
        vision_end_token_id=_required_int(
            config, "vision_end_token_id", allow_zero=True
        ),
        hidden_act=str(vision.get("hidden_act", "silu")),
    )
    if vision_spec.output_hidden_size != hidden_size:
        raise ModelContractError(
            "vision output width must equal the text hidden size"
        )

    tp_facts = [
        hidden_size,
        attention_heads,
        kv_heads,
        head_dim,
        _required_int(text, "vocab_size"),
        dense_intermediate,
    ]
    if is_moe:
        tp_facts.extend((num_experts, moe_intermediate))
    supported_tp = tuple(
        degree
        for degree in (1, 2, 4, 8)
        if all(value % degree == 0 for value in tp_facts)
    )
    architectures = config.get("architectures", ())
    architecture = (
        str(architectures[0])
        if isinstance(architectures, list) and architectures
        else (
            "Qwen3VLMoeForConditionalGeneration"
            if is_moe
            else "Qwen3VLForConditionalGeneration"
        )
    )
    return ModelSpec(
        architecture=architecture,
        model_type=model_type,
        vocab_size=_required_int(text, "vocab_size"),
        hidden_size=hidden_size,
        layers=layers,
        position=PositionSchema(
            kind=PositionKind.MROPE,
            max_positions=_required_int(text, "max_position_embeddings"),
            rope_theta=float(rope.get("rope_theta", 1_000_000.0)),
            head_dim=head_dim,
            mrope_sections=sections,
            partial_rotary_factor=partial,
            mrope_interleaved=bool(rope.get("mrope_interleaved", True)),
        ),
        state_schema=(
            StateComponentSpec(
                kind=StateKind.PAGED_KV,
                layer_indices=tuple(range(num_layers)),
                checkpoint_alignment=1,
            ),
            StateComponentSpec(StateKind.VISION_EMBEDDING),
            StateComponentSpec(StateKind.MEDIA_POSITION),
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
        tie_word_embeddings=bool(
            config.get("tie_word_embeddings", text.get("tie_word_embeddings", False))
        ),
        qk_norm=True,
        vision=vision_spec,
    )


def _required_object(config: dict[str, Any], key: str) -> dict[str, Any]:
    value = config.get(key)
    if not isinstance(value, dict):
        raise ModelContractError(f"Qwen3-VL config requires object {key!r}")
    return value


def _required_str(config: dict[str, Any], key: str) -> str:
    value = config.get(key)
    if not isinstance(value, str) or not value:
        raise ModelContractError(f"Qwen3-VL config requires string {key!r}")
    return value


def _required_int(
    config: dict[str, Any],
    key: str,
    *,
    allow_zero: bool = False,
) -> int:
    try:
        value = int(config[key])
    except (KeyError, TypeError, ValueError) as exc:
        raise ModelContractError(
            f"Qwen3-VL config requires integer {key!r}"
        ) from exc
    if value < (0 if allow_zero else 1):
        raise ModelContractError(f"Qwen3-VL config {key!r} has an invalid value")
    return value
