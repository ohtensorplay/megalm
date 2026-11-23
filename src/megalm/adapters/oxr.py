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

_MODEL_TYPES = frozenset({"oxr"})


def load_oxr_model_spec(model_path: str | Path) -> ModelSpec:
    """Translate an OXR-1.0 checkpoint config into scheduler-free model facts."""

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
    return oxr_model_spec_from_config(payload)


def _object(config: dict[str, Any], key: str) -> dict[str, Any]:
    value = config.get(key)
    if not isinstance(value, dict):
        raise ModelContractError(f"OXR config requires object {key!r}")
    return value


def _integer(config: dict[str, Any], key: str) -> int:
    try:
        value = int(config[key])
    except (KeyError, TypeError, ValueError) as exc:
        raise ModelContractError(f"OXR config requires integer {key!r}") from exc
    if value <= 0:
        raise ModelContractError(f"OXR config {key!r} must be positive")
    return value


def oxr_model_spec_from_config(config: dict[str, Any]) -> ModelSpec:
    model_type = config.get("model_type")
    if not isinstance(model_type, str) or model_type not in _MODEL_TYPES:
        raise ModelContractError(
            f"OXR adapter does not support model_type={model_type!r}"
        )
    llm = _object(config, "llm_config")
    vit = _object(config, "vit_config")

    hidden_size = _integer(llm, "hidden_size")
    num_layers = _integer(llm, "num_hidden_layers")
    attention_heads = _integer(llm, "num_attention_heads")
    kv_heads = _integer(llm, "num_key_value_heads")
    head_dim = int(llm.get("head_dim", hidden_size // attention_heads))
    intermediate_size = _integer(llm, "intermediate_size")
    vocab_size = _integer(llm, "vocab_size")
    max_positions = _integer(llm, "max_position_embeddings")
    # Qwen3-0.6B uses 16 heads x 128 dims with a 1024-wide residual, so the
    # query width intentionally differs from hidden_size.
    if attention_heads % kv_heads:
        raise ModelContractError("OXR llm_config head counts are not divisible")

    layer_types = llm.get("layer_types")
    if isinstance(layer_types, list) and layer_types and (
        len(layer_types) != num_layers
        or any(value != "full_attention" for value in layer_types)
    ):
        raise ModelContractError("OXR requires all-full-attention LLM layers")

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

    visual_vocab_size = _integer(config, "visual_vocab_size")
    if visual_vocab_size <= 4:
        raise ModelContractError("OXR visual_vocab_size must exceed indicator count")
    patch_size = _integer(vit, "patch_size")
    hidden_stride = _integer(vit, "hidden_stride")
    vision_depth = _integer(vit, "num_hidden_layers")
    vision_hidden = _integer(vit, "hidden_size")
    vision_intermediate = _integer(vit, "intermediate_size")
    vision_heads = _integer(vit, "num_attention_heads")
    if vision_hidden % vision_heads:
        raise ModelContractError("OXR ViT head count does not divide hidden size")
    if vit.get("fullatt_block_indexes") is not None:
        raise ModelContractError(
            "OXR partial full-attention ViT blocks are not supported"
        )

    vision = VisionSpec(
        processor_type="oxr",
        patch_size=patch_size,
        merge_size=hidden_stride,
        temporal_patch_size=_integer(vit, "temporal_patch_size"),
        in_channels=_integer(vit, "num_channels"),
        depth=vision_depth,
        hidden_size=vision_hidden,
        intermediate_size=vision_intermediate,
        num_heads=vision_heads,
        output_hidden_size=hidden_size,
        num_position_embeddings=int(vit.get("num_patches", -1))
        if int(vit.get("num_patches", -1)) > 0
        else (_integer(vit, "image_size") // patch_size) ** 2,
        image_token_id=151655,
        vision_start_token_id=151652,
        vision_end_token_id=151653,
        hidden_act=str(vit.get("hidden_act", "gelu_pytorch_tanh")),
    )

    tp_facts = (
        hidden_size,
        attention_heads,
        kv_heads,
        intermediate_size,
        vocab_size,
    )
    supported_tp = tuple(
        degree
        for degree in (1, 2, 4, 8)
        if all(value % degree == 0 for value in tp_facts)
    )
    architecture_values = config.get("architectures", ())
    architecture = (
        str(architecture_values[0])
        if isinstance(architecture_values, list) and architecture_values
        else "OXR"
    )
    return ModelSpec(
        architecture=architecture,
        model_type=model_type,
        vocab_size=vocab_size,
        hidden_size=hidden_size,
        layers=layers,
        position=PositionSchema(
            kind=PositionKind.ROPE,
            max_positions=max_positions,
            rope_theta=float(llm.get("rope_theta", 1_000_000.0)),
            head_dim=head_dim,
        ),
        state_schema=(
            StateComponentSpec(
                kind=StateKind.PAGED_KV,
                layer_indices=tuple(range(num_layers)),
                checkpoint_alignment=1,
            ),
            StateComponentSpec(StateKind.VISION_EMBEDDING),
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
        tie_word_embeddings=bool(llm.get("tie_word_embeddings", True)),
        qk_norm=True,
        vision=vision,
    )


__all__ = ["load_oxr_model_spec", "oxr_model_spec_from_config"]
