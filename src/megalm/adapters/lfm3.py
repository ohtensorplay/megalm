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


def load_lfm3_model_spec(model_path: str | Path) -> ModelSpec:
    path = Path(model_path)
    config_path = path / "config.json" if path.is_dir() else path
    payload = json.loads(config_path.read_text(encoding="utf-8"))
    return lfm3_model_spec_from_config(payload)


def lfm3_model_spec_from_config(config: dict[str, Any]) -> ModelSpec:
    if str(config.get("model_type", "")) != "lfm3":
        raise ModelContractError("LFM3 adapter requires model_type='lfm3'")
    hidden_size = int(config["hidden_size"])
    num_heads = int(config["num_attention_heads"])
    num_kv_heads = int(config["num_key_value_heads"])
    head_dim = int(config.get("head_dim") or hidden_size // num_heads)
    layer_types = tuple(str(value) for value in config["layer_types"])
    rope = config.get("rope_parameters", {})
    if not isinstance(rope, dict):
        raise ModelContractError("LFM3 rope_parameters must be an object")

    layers: list[LayerSpec] = []
    for index, kind in enumerate(layer_types):
        if kind == "conv":
            attention = AttentionKind.GATED_SHORT_CONV
            sliding_window = None
        elif kind == "sliding_attention":
            attention = AttentionKind.FULL
            sliding_window = int(config.get("sliding_window", 256))
        elif kind == "full_attention":
            attention = AttentionKind.FULL
            sliding_window = None
        else:
            raise ModelContractError(f"unsupported LFM3 layer type {kind!r}")
        layers.append(
            LayerSpec(
                index=index,
                attention=attention,
                feed_forward=FeedForwardKind.DENSE,
                hidden_size=hidden_size,
                intermediate_size=int(config["intermediate_size"]),
                num_attention_heads=num_heads,
                num_key_value_heads=num_kv_heads,
                head_dim=head_dim,
                sliding_window=sliding_window,
            )
        )

    full_rope = rope.get("full_attention", {})
    theta = float(full_rope.get("rope_theta", 1e6)) if full_rope else 1e6
    position = PositionSchema(
        kind=PositionKind.ROPE,
        max_positions=int(config["max_position_embeddings"]),
        rope_theta=theta,
        head_dim=head_dim,
        partial_rotary_factor=float(full_rope.get("partial_rotary_factor", 1.0)),
    )
    attention_indices = tuple(
        index for index, kind in enumerate(layer_types) if kind != "conv"
    )
    return ModelSpec(
        architecture="Lfm3ForCausalLM",
        model_type="lfm3",
        vocab_size=int(config["vocab_size"]),
        hidden_size=hidden_size,
        layers=tuple(layers),
        position=position,
        state_schema=(
            StateComponentSpec(
                kind=StateKind.PAGED_KV, layer_indices=attention_indices
            ),
            StateComponentSpec(kind=StateKind.PREFIX_CHECKPOINT),
        ),
        weight_mapping=(
            WeightMappingRule(source_pattern=".*", target_template="\\g<0>"),
        ),
        parallel=ParallelConstraints(supported_tp_degrees=(1,)),
        tie_word_embeddings=bool(config.get("tie_word_embeddings", True)),
    )
