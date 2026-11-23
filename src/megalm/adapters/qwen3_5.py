from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from megalm.contracts.model import (
    AttentionKind,
    FeedForwardKind,
    GatedDeltaNetSpec,
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

_MODEL_TYPES = frozenset({"qwen3_5", "qwen3_5_moe"})
_TEXT_MODEL_TYPES = frozenset({"qwen3_5_text", "qwen3_5_moe_text"})


def load_qwen3_5_model_spec(model_path: str | Path) -> ModelSpec:
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
    return qwen3_5_model_spec_from_config(payload)


def qwen3_5_model_spec_from_config(config: dict[str, Any]) -> ModelSpec:
    model_type = _required_str(config, "model_type")
    if model_type not in _MODEL_TYPES:
        raise ModelContractError(
            f"Qwen3.5 adapter does not support model_type={model_type!r}"
        )
    text = _required_object(config, "text_config")
    vision = _required_object(config, "vision_config")
    text_model_type = _required_str(text, "model_type")
    expected_text_type = (
        "qwen3_5_moe_text" if model_type == "qwen3_5_moe" else "qwen3_5_text"
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
    if attention_heads % kv_heads:
        raise ModelContractError("full-attention head counts are not divisible")

    layer_types = text.get("layer_types")
    if layer_types is None:
        interval = _positive_int(text.get("full_attention_interval", 4), "full_attention_interval")
        layer_types = [
            "full_attention" if (index + 1) % interval == 0 else "linear_attention"
            for index in range(num_layers)
        ]
    if (
        not isinstance(layer_types, list)
        or len(layer_types) != num_layers
        or any(
            layer_type not in {"full_attention", "linear_attention"}
            for layer_type in layer_types
        )
    ):
        raise ModelContractError(
            "Qwen3.5 layer_types must describe every decoder layer"
        )

    gdn = GatedDeltaNetSpec(
        num_key_heads=_required_int(text, "linear_num_key_heads"),
        num_value_heads=_required_int(text, "linear_num_value_heads"),
        key_head_dim=_required_int(text, "linear_key_head_dim"),
        value_head_dim=_required_int(text, "linear_value_head_dim"),
        conv_kernel_dim=_required_int(text, "linear_conv_kernel_dim"),
        state_dtype=str(text.get("mamba_ssm_dtype", "float32")),
    )
    is_moe = model_type == "qwen3_5_moe"
    dense_intermediate = (
        _required_int(text, "intermediate_size") if not is_moe else 1
    )
    moe_intermediate = (
        _required_int(text, "moe_intermediate_size") if is_moe else 0
    )
    num_experts = _required_int(text, "num_experts") if is_moe else 0
    experts_per_token = (
        _required_int(text, "num_experts_per_tok") if is_moe else 0
    )
    if is_moe and experts_per_token > num_experts:
        raise ModelContractError(
            "Qwen3.5-MoE num_experts_per_tok exceeds num_experts"
        )
    shared_expert_intermediate = (
        _required_int(text, "shared_expert_intermediate_size", allow_zero=True)
        if is_moe
        else 0
    )
    layers = tuple(
        LayerSpec(
            index=index,
            attention=(
                AttentionKind.FULL
                if layer_type == "full_attention"
                else AttentionKind.GATED_DELTA_NET
            ),
            feed_forward=(
                FeedForwardKind.MOE if is_moe else FeedForwardKind.DENSE
            ),
            hidden_size=hidden_size,
            intermediate_size=(
                moe_intermediate if is_moe else dense_intermediate
            ),
            num_attention_heads=attention_heads,
            num_key_value_heads=kv_heads,
            head_dim=head_dim,
            num_experts=num_experts,
            experts_per_token=experts_per_token,
            gated_delta_net=(
                None if layer_type == "full_attention" else gdn
            ),
        )
        for index, layer_type in enumerate(layer_types)
    )
    full_layers = tuple(
        layer.index for layer in layers if layer.attention is AttentionKind.FULL
    )
    gdn_layers = tuple(
        layer.index
        for layer in layers
        if layer.attention is AttentionKind.GATED_DELTA_NET
    )
    if not full_layers or not gdn_layers:
        raise ModelContractError(
            "Qwen3.5 requires both full-attention and Gated DeltaNet layers"
        )

    rope = text.get("rope_parameters", text.get("rope_scaling", {}))
    if rope is None:
        rope = {}
    if not isinstance(rope, dict):
        raise ModelContractError("Qwen3.5 rope parameters must be an object")
    sections = rope.get("mrope_section", text.get("mrope_section"))
    if not isinstance(sections, list):
        raise ModelContractError("Qwen3.5 requires MRoPE section sizes")
    partial_rotary_factor = float(
        rope.get(
            "partial_rotary_factor",
            text.get("partial_rotary_factor", 0.25),
        )
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
        hidden_act=str(vision.get("hidden_act", "gelu_pytorch_tanh")),
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
        gdn.num_key_heads,
        gdn.num_value_heads,
        _required_int(text, "vocab_size"),
    ]
    if is_moe:
        tp_facts.append(num_experts)
        tp_facts.append(moe_intermediate)
        if shared_expert_intermediate:
            tp_facts.append(shared_expert_intermediate)
    else:
        tp_facts.append(dense_intermediate)
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
            "Qwen3_5MoeForConditionalGeneration"
            if is_moe
            else "Qwen3_5ForConditionalGeneration"
        )
    )
    state_schema = (
        StateComponentSpec(StateKind.PAGED_KV, full_layers),
        StateComponentSpec(StateKind.GDN_CONV, gdn_layers),
        StateComponentSpec(StateKind.GDN_RECURRENT, gdn_layers),
        StateComponentSpec(StateKind.PREFIX_CHECKPOINT, tuple(range(num_layers))),
        StateComponentSpec(StateKind.VISION_EMBEDDING),
        StateComponentSpec(StateKind.MEDIA_POSITION),
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
            rope_theta=float(
                rope.get("rope_theta", text.get("rope_theta", 10_000_000.0))
            ),
            head_dim=head_dim,
            mrope_sections=tuple(int(value) for value in sections),
            partial_rotary_factor=partial_rotary_factor,
            mrope_interleaved=bool(rope.get("mrope_interleaved", True)),
        ),
        state_schema=state_schema,
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
        qk_norm=True,
        tie_word_embeddings=bool(
            config.get(
                "tie_word_embeddings",
                text.get("tie_word_embeddings", False),
            )
        ),
        vision=vision_spec,
    )


def _required_object(config: dict[str, Any], key: str) -> dict[str, Any]:
    value = config.get(key)
    if not isinstance(value, dict):
        raise ModelContractError(f"Qwen3.5 config requires object {key!r}")
    return value


def _required_str(config: dict[str, Any], key: str) -> str:
    value = config.get(key)
    if not isinstance(value, str) or not value:
        raise ModelContractError(f"Qwen3.5 config requires string {key!r}")
    return value


def _positive_int(value: Any, key: str, *, allow_zero: bool = False) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise ModelContractError(
            f"Qwen3.5 config requires integer {key!r}"
        ) from exc
    minimum = 0 if allow_zero else 1
    if parsed < minimum:
        qualifier = "non-negative" if allow_zero else "positive"
        raise ModelContractError(f"Qwen3.5 config {key!r} must be {qualifier}")
    return parsed


def _required_int(
    config: dict[str, Any],
    key: str,
    *,
    allow_zero: bool = False,
) -> int:
    if key not in config:
        raise ModelContractError(f"Qwen3.5 config requires integer {key!r}")
    return _positive_int(config[key], key, allow_zero=allow_zero)
