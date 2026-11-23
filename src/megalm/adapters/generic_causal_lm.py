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

_MODEL_TYPES = {
    "llama": "LlamaForCausalLM",
    "mistral": "MistralForCausalLM",
    "qwen2": "Qwen2ForCausalLM",
    "qwen2_moe": "Qwen2MoeForCausalLM",
    "mixtral": "MixtralForCausalLM",
    "yi": "YiForCausalLM",
    "deepseek": "DeepSeekForCausalLM",
    "phi3": "Phi3ForCausalLM",
    "internlm2": "InternLM2ForCausalLM",
    "internlm3": "InternLM3ForCausalLM",
    "seed_oss": "SeedOssForCausalLM",
    "granite": "GraniteForCausalLM",
    "ernie4_5": "Ernie4_5ForCausalLM",
}
_MOE_TYPES = frozenset({"qwen2_moe", "mixtral"})
_QKV_BIAS_TYPES = frozenset({"qwen2", "qwen2_moe", "seed_oss"})


def load_generic_causal_lm_spec(model_path: str | Path) -> ModelSpec:
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
    return generic_causal_lm_spec_from_config(payload)


def generic_causal_lm_spec_from_config(config: dict[str, Any]) -> ModelSpec:
    model_type = _string(config, "model_type")
    if model_type not in _MODEL_TYPES:
        raise ModelContractError(
            f"generic causal LM adapter does not support {model_type!r}"
        )
    _reject_unsupported_rope(config)
    if config.get("is_causal", True) is False:
        raise ModelContractError(
            "generic causal LM adapter requires causal attention"
        )
    hidden_size = _positive_int(config, "hidden_size")
    num_layers = _positive_int(config, "num_hidden_layers")
    attention_heads = _positive_int(config, "num_attention_heads")
    kv_heads = _positive_int(config, "num_key_value_heads", default=attention_heads)
    if attention_heads % kv_heads:
        raise ModelContractError("full-attention head counts are not divisible")
    head_dim = _positive_int(
        config,
        "head_dim",
        default=hidden_size // attention_heads,
    )
    if hidden_size != attention_heads * head_dim:
        raise ModelContractError(
            "hidden_size must equal num_attention_heads * head_dim"
        )
    scales = _model_scales(config, model_type, head_dim)
    rope_facts = _rope_facts(config, model_type, head_dim)
    vocab_size = _positive_int(config, "vocab_size")
    max_positions = _positive_int(config, "max_position_embeddings")
    sliding_window = config.get("sliding_window")
    if sliding_window is not None:
        window = _positive_int(config, "sliding_window")
        max_positions = min(max_positions, window)

    is_moe = model_type in _MOE_TYPES
    sparse_step = 1
    mlp_only_layers: frozenset[int] = frozenset()
    if is_moe:
        if model_type == "mixtral":
            num_experts = _positive_int(config, "num_local_experts")
            moe_intermediate = _positive_int(config, "intermediate_size")
            dense_intermediate = 1
        else:
            num_experts = _positive_int(config, "num_experts")
            moe_intermediate = _positive_int(config, "moe_intermediate_size")
            dense_intermediate = _positive_int(config, "intermediate_size")
            sparse_step = _positive_int(
                config,
                "decoder_sparse_step",
                default=1,
            )
            mlp_only_layers = frozenset(
                int(index) for index in config.get("mlp_only_layers", ())
            )
            if not mlp_only_layers.issubset(set(range(num_layers))):
                raise ModelContractError(
                    "mlp_only_layers contains an invalid layer index"
                )
        experts_per_token = _positive_int(config, "num_experts_per_tok")
        if experts_per_token > num_experts:
            raise ModelContractError("num_experts_per_tok exceeds num_experts")
    else:
        num_experts = 0
        experts_per_token = 0
        moe_intermediate = 0
        dense_intermediate = _positive_int(config, "intermediate_size")

    layers: list[LayerSpec] = []
    for index in range(num_layers):
        layer_is_moe = is_moe and (
            model_type == "mixtral"
            or (
                index not in mlp_only_layers
                and (index + 1) % sparse_step == 0
            )
        )
        layers.append(
            LayerSpec(
                index=index,
                attention=AttentionKind.FULL,
                feed_forward=(
                    FeedForwardKind.MOE if layer_is_moe else FeedForwardKind.DENSE
                ),
                hidden_size=hidden_size,
                intermediate_size=(
                    moe_intermediate if layer_is_moe else dense_intermediate
                ),
                num_attention_heads=attention_heads,
                num_key_value_heads=kv_heads,
                head_dim=head_dim,
                num_experts=num_experts if layer_is_moe else 0,
                experts_per_token=experts_per_token if layer_is_moe else 0,
            )
        )

    tp_facts = [
        hidden_size,
        attention_heads,
        kv_heads,
        head_dim,
        vocab_size,
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
        else _MODEL_TYPES[model_type]
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
            rope_theta=float(
                config.get(
                    "rope_theta",
                    1_000_000.0 if model_type == "seed_oss" else 10_000.0,
                )
            ),
            head_dim=head_dim,
            **rope_facts,
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
        qk_norm=bool(config.get("qk_norm", False)),
        qkv_bias=model_type in _QKV_BIAS_TYPES,
        **scales,
    )


def _rope_facts(
    config: dict[str, Any],
    model_type: str,
    head_dim: int,
) -> dict[str, object]:
    defaults: dict[str, object] = {
        "partial_rotary_factor": 1.0,
        "rotary_style": "neox",
    }
    if model_type != "ernie4_5":
        return defaults
    if bool(config.get("qk_norm", False)):
        raise ModelContractError(
            "Ernie 4.5 interleaved RoPE does not support QK-Norm"
        )
    rope = config.get("rope_parameters", config.get("rope_scaling", {}))
    if not isinstance(rope, dict):
        raise ModelContractError("Ernie 4.5 rope parameters must be an object")
    try:
        partial = float(rope.get("partial_rotary_factor", 0.5))
    except (TypeError, ValueError) as exc:
        raise ModelContractError(
            "Ernie 4.5 partial_rotary_factor must be numeric"
        ) from exc
    rotary_dim = head_dim * partial
    if not 0 < partial <= 1 or not rotary_dim.is_integer() or int(rotary_dim) % 2:
        raise ModelContractError("Ernie 4.5 rotary_dim must be even and positive")
    return {
        "partial_rotary_factor": partial,
        "rotary_style": "interleaved",
    }


def _model_scales(
    config: dict[str, Any],
    model_type: str,
    head_dim: int,
) -> dict[str, float]:
    defaults = {
        "embedding_multiplier": 1.0,
        "residual_multiplier": 1.0,
        "attention_multiplier": 0.0,
        "logit_scale": 1.0,
    }
    if model_type != "granite":
        return defaults
    if any(
        config.get(key, False) is True
        for key in ("bias", "attention_bias", "mlp_bias")
    ):
        raise ModelContractError(
            "Granite bias variants require an explicit first-party implementation"
        )
    try:
        embedding = float(config.get("embedding_multiplier", 1.0))
        residual = float(config.get("residual_multiplier", 1.0))
        attention = float(config.get("attention_multiplier", 0.0))
        logit_scale = float(config.get("logit_scale", 1.0))
        logits_scaling = float(config.get("logits_scaling", 1.0))
    except (TypeError, ValueError) as exc:
        raise ModelContractError(
            "Granite scaling factors must be numeric"
        ) from exc
    if min(embedding, residual, logits_scaling) <= 0:
        raise ModelContractError("Granite scaling factors must be positive")
    expected_attention = head_dim**-0.5
    if attention and abs(attention - expected_attention) > 1e-6:
        raise ModelContractError(
            "Granite attention_multiplier requires a dedicated attention kernel"
        )
    return {
        "embedding_multiplier": embedding,
        "residual_multiplier": residual,
        "attention_multiplier": attention or expected_attention,
        "logit_scale": logit_scale / logits_scaling,
    }


def _reject_unsupported_rope(config: dict[str, Any]) -> None:
    for key in ("rope_scaling", "rope_parameters"):
        value = config.get(key)
        if value in (None, {}):
            continue
        if not isinstance(value, dict):
            raise ModelContractError(f"config {key!r} must be an object")
        rope_type = value.get("rope_type", value.get("type", "default"))
        if rope_type not in {"default", "base"}:
            raise ModelContractError(
                f"unsupported rope type {rope_type!r} in {key!r}"
            )


def _string(config: dict[str, Any], key: str) -> str:
    value = config.get(key)
    if not isinstance(value, str) or not value:
        raise ModelContractError(f"config requires string {key!r}")
    return value


def _positive_int(
    config: dict[str, Any],
    key: str,
    *,
    default: int | None = None,
) -> int:
    value = config.get(key, default)
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise ModelContractError(f"config requires integer {key!r}") from exc
    if parsed <= 0:
        raise ModelContractError(f"config {key!r} must be positive")
    return parsed
