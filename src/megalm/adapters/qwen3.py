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
    WeightMappingRule,
)

_QWEN3_MODEL_TYPES = frozenset({"qwen3", "qwen3_moe"})


def load_qwen3_model_spec(model_path: str | Path) -> ModelSpec:
    """Translate an HF-compatible Qwen3 config into scheduler-free model facts."""

    path = Path(model_path)
    config_path = path / "config.json" if path.is_dir() else path
    try:
        config = json.loads(config_path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ModelContractError(f"missing model config: {config_path}") from exc
    except json.JSONDecodeError as exc:
        raise ModelContractError(f"invalid JSON model config: {config_path}") from exc
    if not isinstance(config, dict):
        raise ModelContractError("model config must be a JSON object")
    return qwen3_model_spec_from_config(config)


def qwen3_model_spec_from_config(config: dict[str, Any]) -> ModelSpec:
    model_type = _required_str(config, "model_type")
    if model_type not in _QWEN3_MODEL_TYPES:
        raise ModelContractError(
            f"Qwen3 adapter does not support model_type={model_type!r}"
        )

    hidden_size = _required_int(config, "hidden_size")
    num_layers = _required_int(config, "num_hidden_layers")
    attention_heads = _required_int(config, "num_attention_heads")
    kv_heads = _required_int(config, "num_key_value_heads")
    head_dim = int(config.get("head_dim", hidden_size // attention_heads))
    dense_intermediate = _required_int(config, "intermediate_size")
    max_positions = _required_int(config, "max_position_embeddings")
    vocab_size = _required_int(config, "vocab_size")

    if hidden_size != attention_heads * head_dim:
        raise ModelContractError(
            "hidden_size must equal num_attention_heads * head_dim"
        )

    is_moe_model = model_type == "qwen3_moe"
    num_experts = int(config.get("num_experts", 0)) if is_moe_model else 0
    experts_per_token = (
        int(config.get("num_experts_per_tok", 0)) if is_moe_model else 0
    )
    moe_intermediate = (
        _required_int(config, "moe_intermediate_size")
        if is_moe_model
        else dense_intermediate
    )
    sparse_step = int(config.get("decoder_sparse_step", 1))
    mlp_only_layers = frozenset(int(index) for index in config.get("mlp_only_layers", []))
    if sparse_step <= 0:
        raise ModelContractError("decoder_sparse_step must be positive")
    if not mlp_only_layers.issubset(set(range(num_layers))):
        raise ModelContractError("mlp_only_layers contains an invalid layer index")

    layers: list[LayerSpec] = []
    for layer_index in range(num_layers):
        is_moe_layer = (
            is_moe_model
            and num_experts > 0
            and layer_index not in mlp_only_layers
            and (layer_index + 1) % sparse_step == 0
        )
        layers.append(
            LayerSpec(
                index=layer_index,
                attention=AttentionKind.FULL,
                feed_forward=(
                    FeedForwardKind.MOE
                    if is_moe_layer
                    else FeedForwardKind.DENSE
                ),
                hidden_size=hidden_size,
                intermediate_size=(
                    moe_intermediate if is_moe_layer else dense_intermediate
                ),
                num_attention_heads=attention_heads,
                num_key_value_heads=kv_heads,
                head_dim=head_dim,
                num_experts=num_experts if is_moe_layer else 0,
                experts_per_token=experts_per_token if is_moe_layer else 0,
            )
        )

    tp_divisors = (
        hidden_size,
        attention_heads,
        dense_intermediate,
        vocab_size,
        moe_intermediate,
    )
    supported_tp = tuple(
        degree
        for degree in (1, 2, 4, 8)
        if all(value % degree == 0 for value in tp_divisors)
    )
    architecture_values = config.get("architectures", ())
    architecture = (
        str(architecture_values[0])
        if isinstance(architecture_values, list) and architecture_values
        else ("Qwen3MoeForCausalLM" if is_moe_model else "Qwen3ForCausalLM")
    )

    return ModelSpec(
        architecture=architecture,
        model_type=model_type,
        vocab_size=vocab_size,
        hidden_size=hidden_size,
        layers=tuple(layers),
        position=PositionSchema(
            kind=PositionKind.ROPE,
            max_positions=max_positions,
            rope_theta=float(config.get("rope_theta", 1_000_000.0)),
            head_dim=head_dim,
        ),
        state_schema=(
            StateComponentSpec(
                kind=StateKind.PAGED_KV,
                layer_indices=tuple(range(num_layers)),
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
        tie_word_embeddings=bool(config.get("tie_word_embeddings", False)),
        qk_norm=True,
    )


def _required_int(config: dict[str, Any], key: str) -> int:
    try:
        value = int(config[key])
    except (KeyError, TypeError, ValueError) as exc:
        raise ModelContractError(f"Qwen3 config requires integer {key!r}") from exc
    if value <= 0:
        raise ModelContractError(f"Qwen3 config {key!r} must be positive")
    return value


def _required_str(config: dict[str, Any], key: str) -> str:
    value = config.get(key)
    if not isinstance(value, str) or not value:
        raise ModelContractError(f"Qwen3 config requires string {key!r}")
    return value

