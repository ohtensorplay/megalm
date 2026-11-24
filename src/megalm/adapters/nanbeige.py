"""Nanbeige 4.2 config -> scheduler-free model facts.

Nanbeige4.2-3B is a dense RoPE/GQA transformer whose *whole* decoder stack is
replayed ``num_loops`` times with shared weights.  Every loop owns its own KV
cache layers, and a final RMSNorm is applied at the end of every loop (unless
``skip_loop_final_norm`` is set).  The optional features in the reference
``modeling_nanbeige.py`` (n-gram embeddings, hyper connections, depth attention,
LoopSplit) are not present in the released checkpoint and are rejected here so
an unsupported checkpoint can never be silently served with wrong logits.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from megalm.contracts.model import (
    AttentionKind,
    FeedForwardKind,
    LayerSpec,
    LoopSpec,
    ModelContractError,
    ModelSpec,
    ParallelConstraints,
    PositionKind,
    PositionSchema,
    StateComponentSpec,
    StateKind,
    WeightMappingRule,
)

_MODEL_TYPES = frozenset({"nanbeige"})
_ARCHITECTURE = "NanbeigeForCausalLM"


def load_nanbeige_model_spec(model_path: str | Path) -> ModelSpec:
    """Translate a Nanbeige4.2 ``config.json`` into :class:`ModelSpec`."""

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
    return nanbeige_model_spec_from_config(config)


def nanbeige_model_spec_from_config(config: dict[str, Any]) -> ModelSpec:
    """Build the contract for the released Nanbeige4.2 checkpoint layout."""

    model_type = _required_str(config, "model_type")
    if model_type not in _MODEL_TYPES:
        raise ModelContractError(
            f"Nanbeige adapter does not support model_type={model_type!r}"
        )
    _reject_unsupported_features(config)

    hidden_size = _required_int(config, "hidden_size")
    intermediate_size = _required_int(config, "intermediate_size")
    num_layers = _required_int(config, "num_hidden_layers")
    attention_heads = _required_int(config, "num_attention_heads")
    kv_heads = _required_int(config, "num_key_value_heads")
    head_dim = _required_int(config, "head_dim")
    vocab_size = _required_int(config, "vocab_size")
    max_positions = _required_int(config, "max_position_embeddings")
    num_loops = int(config.get("num_loops", 1))
    if num_loops < 1:
        raise ModelContractError("Nanbeige num_loops must be positive")
    if attention_heads % kv_heads:
        raise ModelContractError("Nanbeige query heads must be divisible by KV heads")

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

    # The released checkpoint replays all 22 physical layers. ``gate_rank`` is
    # only meaningful for Inkling-style gated loops; Nanbeige has no loop gate
    # weights, so the contract uses the smallest legal positive value.
    loop = LoopSpec(
        core_start=0,
        core_end=num_layers,
        num_passes=num_loops,
        gate_rank=1,
        is_varloop=False,
    )

    state_schema = (
        StateComponentSpec(StateKind.PAGED_KV, tuple(range(num_layers))),
    )

    tp_facts = (
        hidden_size,
        intermediate_size,
        attention_heads,
        kv_heads,
        head_dim,
        vocab_size,
    )
    supported_tp = tuple(
        degree
        for degree in (1, 2, 4, 8)
        if all(value % degree == 0 for value in tp_facts)
    )

    architectures = config.get("architectures", ())
    architecture = (
        str(architectures[0])
        if isinstance(architectures, list) and architectures
        else _ARCHITECTURE
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
            rope_theta=float(config.get("rope_theta", 10_000.0)),
            head_dim=head_dim,
            rotary_style="neox",
        ),
        state_schema=state_schema,
        weight_mapping=(
            WeightMappingRule(source_pattern=r"(?P<name>.+)", target_template=r"\g<name>"),
        ),
        parallel=ParallelConstraints(
            supported_tp_degrees=supported_tp,
            minimum_compute_capability=(8, 0),
        ),
        supported_dtypes=("bfloat16", "float16"),
        tie_word_embeddings=bool(config.get("tie_word_embeddings", False)),
        loop=loop,
    )


def _reject_unsupported_features(config: dict[str, Any]) -> None:
    """Guard against Nanbeige variants the native hot path cannot execute."""

    if config.get("emb_neighbor_num") not in (None, 0):
        raise ModelContractError("Nanbeige n-gram embeddings are not supported")
    for flag in (
        "enable_hyper_connection",
        "enable_mhc",
        "enable_h_res_identity",
        "enable_depth_attention",
        "enable_double_loop_split",
        "loop_share_kv",
        "qk_layernorm",
        "attention_bias",
        "mlp_bias",
    ):
        if config.get(flag):
            raise ModelContractError(
                f"Nanbeige {flag} is not supported by the native hot path"
            )
    if config.get("rope_scaling") not in (None, {}):
        raise ModelContractError("Nanbeige rope_scaling is not supported")
    if config.get("loop_loss_weights"):
        raise ModelContractError(
            "Nanbeige loop_loss_weights are training-only and not supported"
        )
    if int(config.get("pretraining_tp", 1)) != 1:
        raise ModelContractError("Nanbeige pretraining_tp != 1 is not supported")
    if str(config.get("hidden_act", "silu")) != "silu":
        raise ModelContractError("Nanbeige requires silu hidden_act")


def _required_str(config: dict[str, Any], key: str) -> str:
    value = config.get(key)
    if not isinstance(value, str) or not value:
        raise ModelContractError(f"Nanbeige config requires string {key!r}")
    return value


def _required_int(config: dict[str, Any], key: str) -> int:
    if key not in config:
        raise ModelContractError(f"Nanbeige config requires integer {key!r}")
    try:
        parsed = int(config[key])
    except (TypeError, ValueError) as exc:
        raise ModelContractError(f"Nanbeige config {key!r} must be an integer") from exc
    if parsed <= 0:
        raise ModelContractError(f"Nanbeige config {key!r} must be positive")
    return parsed


__all__ = ["load_nanbeige_model_spec", "nanbeige_model_spec_from_config"]
