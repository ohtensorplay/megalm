from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


class GenericCausalLMConfigError(ValueError):
    """Raised when a Llama-family checkpoint declares unsupported geometry."""


@dataclass(frozen=True, slots=True)
class GenericCausalLMConfig:
    model_type: str
    vocab_size: int
    hidden_size: int
    intermediate_size: int
    num_hidden_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int
    max_position_embeddings: int
    rms_norm_eps: float
    rope_theta: float
    tie_word_embeddings: bool
    qkv_bias: bool
    qk_norm: bool
    num_experts: int
    experts_per_token: int
    moe_intermediate_size: int
    norm_topk_prob: bool
    decoder_sparse_step: int
    mlp_only_layers: tuple[int, ...]
    embedding_multiplier: float = 1.0
    residual_multiplier: float = 1.0
    attention_multiplier: float = 0.0
    logit_scale: float = 1.0
    partial_rotary_factor: float = 1.0
    rotary_style: str = "neox"

    @classmethod
    def from_path(cls, model_path: str | Path) -> GenericCausalLMConfig:
        path = Path(model_path)
        config_path = path / "config.json" if path.is_dir() else path
        try:
            payload = json.loads(config_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise GenericCausalLMConfigError(
                f"cannot load generic causal LM config from {config_path}"
            ) from exc
        if not isinstance(payload, dict):
            raise GenericCausalLMConfigError("model config must be an object")
        model_type = str(payload.get("model_type", ""))
        supported = {
            "llama",
            "mistral",
            "qwen2",
            "qwen2_moe",
            "mixtral",
            "yi",
            "deepseek",
            "phi3",
            "internlm2",
            "internlm3",
            "seed_oss",
            "granite",
            "ernie4_5",
        }
        if model_type not in supported:
            raise GenericCausalLMConfigError(
                f"unsupported generic model_type {model_type!r}"
            )
        _reject_unsupported_rope(payload)
        if payload.get("is_causal", True) is False:
            raise GenericCausalLMConfigError(
                "generic causal LM config requires causal attention"
            )
        hidden_size = _positive(payload, "hidden_size")
        attention_heads = _positive(payload, "num_attention_heads")
        kv_heads = _positive(
            payload,
            "num_key_value_heads",
            default=attention_heads,
        )
        head_dim = _positive(
            payload,
            "head_dim",
            default=hidden_size // attention_heads,
        )
        if hidden_size != attention_heads * head_dim:
            raise GenericCausalLMConfigError(
                "hidden_size and attention heads disagree"
            )
        if attention_heads % kv_heads:
            raise GenericCausalLMConfigError("GQA head counts are not divisible")
        max_positions = _positive(payload, "max_position_embeddings")
        sliding_window = payload.get("sliding_window")
        if sliding_window is not None:
            max_positions = min(
                max_positions,
                _positive(payload, "sliding_window"),
            )
        scales = _model_scales(payload, model_type, head_dim)
        rope_facts = _rope_facts(payload, model_type, head_dim)
        is_moe = model_type in {"qwen2_moe", "mixtral"}
        sparse_step = 1
        mlp_only_layers: tuple[int, ...] = ()
        if model_type == "mixtral":
            num_experts = _positive(payload, "num_local_experts")
            moe_intermediate = _positive(payload, "intermediate_size")
            dense_intermediate = 1
        elif model_type == "qwen2_moe":
            num_experts = _positive(payload, "num_experts")
            moe_intermediate = _positive(payload, "moe_intermediate_size")
            dense_intermediate = _positive(payload, "intermediate_size")
            sparse_step = _positive(payload, "decoder_sparse_step", default=1)
            mlp_only_layers = tuple(
                int(index) for index in payload.get("mlp_only_layers", ())
            )
        else:
            num_experts = 0
            moe_intermediate = 0
            dense_intermediate = _positive(payload, "intermediate_size")
        experts_per_token = (
            _positive(payload, "num_experts_per_tok") if is_moe else 0
        )
        config = cls(
            model_type=model_type,
            vocab_size=_positive(payload, "vocab_size"),
            hidden_size=hidden_size,
            intermediate_size=dense_intermediate,
            num_hidden_layers=_positive(payload, "num_hidden_layers"),
            num_attention_heads=attention_heads,
            num_key_value_heads=kv_heads,
            head_dim=head_dim,
            max_position_embeddings=max_positions,
            rms_norm_eps=float(payload.get("rms_norm_eps", 1e-6)),
            rope_theta=float(
                payload.get(
                    "rope_theta",
                    1_000_000.0 if model_type == "seed_oss" else 10_000.0,
                )
            ),
            tie_word_embeddings=bool(
                payload.get("tie_word_embeddings", False)
            ),
            qkv_bias=model_type in {"qwen2", "qwen2_moe", "seed_oss"},
            qk_norm=bool(payload.get("qk_norm", False)),
            num_experts=num_experts,
            experts_per_token=experts_per_token,
            moe_intermediate_size=moe_intermediate,
            norm_topk_prob=bool(payload.get("norm_topk_prob", False)),
            decoder_sparse_step=sparse_step,
            mlp_only_layers=mlp_only_layers,
            **scales,
            **rope_facts,
        )
        config.validate()
        return config

    def validate(self) -> None:
        if self.head_dim % 2:
            raise GenericCausalLMConfigError("RoPE head_dim must be even")
        rotary_dim = self.head_dim * self.partial_rotary_factor
        if (
            not 0 < self.partial_rotary_factor <= 1
            or not rotary_dim.is_integer()
            or int(rotary_dim) % 2
        ):
            raise GenericCausalLMConfigError("partial rotary_dim must be even")
        if self.rotary_style not in {"neox", "interleaved"}:
            raise GenericCausalLMConfigError("unsupported rotary_style")
        if self.qk_norm and (
            self.partial_rotary_factor != 1
            or self.rotary_style == "interleaved"
        ):
            raise GenericCausalLMConfigError(
                "partial or interleaved RoPE does not support QK-Norm"
            )
        if self.rms_norm_eps <= 0 or self.rope_theta <= 0:
            raise GenericCausalLMConfigError("norm and RoPE facts must be positive")
        if any(
            index < 0 or index >= self.num_hidden_layers
            for index in self.mlp_only_layers
        ):
            raise GenericCausalLMConfigError(
                "mlp_only_layers contains an invalid layer index"
            )
        if self.num_experts:
            if min(
                self.experts_per_token,
                self.moe_intermediate_size,
            ) <= 0:
                raise GenericCausalLMConfigError("MoE expert facts are missing")
            if self.experts_per_token > self.num_experts:
                raise GenericCausalLMConfigError(
                    "top-k exceeds the expert count"
                )

    def is_moe_layer(self, layer_index: int) -> bool:
        return (
            self.num_experts > 0
            and layer_index not in self.mlp_only_layers
            and (layer_index + 1) % self.decoder_sparse_step == 0
        )

    @property
    def expert_projection_names(self) -> tuple[str, str, str]:
        if self.model_type == "mixtral":
            return ("w1", "w3", "w2")
        return ("gate_proj", "up_proj", "down_proj")


def _rope_facts(
    payload: dict[str, Any],
    model_type: str,
    head_dim: int,
) -> dict[str, object]:
    defaults: dict[str, object] = {
        "partial_rotary_factor": 1.0,
        "rotary_style": "neox",
    }
    if model_type != "ernie4_5":
        return defaults
    rope = payload.get("rope_parameters", payload.get("rope_scaling", {}))
    if not isinstance(rope, dict):
        raise GenericCausalLMConfigError(
            "Ernie 4.5 rope parameters must be an object"
        )
    try:
        partial = float(rope.get("partial_rotary_factor", 0.5))
    except (TypeError, ValueError) as exc:
        raise GenericCausalLMConfigError(
            "Ernie 4.5 partial_rotary_factor must be numeric"
        ) from exc
    rotary_dim = head_dim * partial
    if not 0 < partial <= 1 or not rotary_dim.is_integer() or int(rotary_dim) % 2:
        raise GenericCausalLMConfigError(
            "Ernie 4.5 rotary_dim must be even and positive"
        )
    return {
        "partial_rotary_factor": partial,
        "rotary_style": "interleaved",
    }


def _model_scales(
    payload: dict[str, Any],
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
        payload.get(key, False) is True
        for key in ("bias", "attention_bias", "mlp_bias")
    ):
        raise GenericCausalLMConfigError(
            "Granite bias variants require an explicit first-party implementation"
        )
    try:
        embedding = float(payload.get("embedding_multiplier", 1.0))
        residual = float(payload.get("residual_multiplier", 1.0))
        attention = float(payload.get("attention_multiplier", 0.0))
        logit_scale = float(payload.get("logit_scale", 1.0))
        logits_scaling = float(payload.get("logits_scaling", 1.0))
    except (TypeError, ValueError) as exc:
        raise GenericCausalLMConfigError(
            "Granite scaling factors must be numeric"
        ) from exc
    if min(embedding, residual, logits_scaling) <= 0:
        raise GenericCausalLMConfigError(
            "Granite scaling factors must be positive"
        )
    expected_attention = head_dim**-0.5
    if attention and abs(attention - expected_attention) > 1e-6:
        raise GenericCausalLMConfigError(
            "Granite attention_multiplier requires a dedicated attention kernel"
        )
    return {
        "embedding_multiplier": embedding,
        "residual_multiplier": residual,
        "attention_multiplier": attention or expected_attention,
        "logit_scale": logit_scale / logits_scaling,
    }


def _reject_unsupported_rope(payload: dict[str, Any]) -> None:
    for key in ("rope_scaling", "rope_parameters"):
        value = payload.get(key)
        if value in (None, {}):
            continue
        if not isinstance(value, dict):
            raise GenericCausalLMConfigError(
                f"config {key!r} must be an object"
            )
        rope_type = value.get("rope_type", value.get("type", "default"))
        if rope_type not in {"default", "base"}:
            raise GenericCausalLMConfigError(
                f"unsupported rope type {rope_type!r} in {key!r}"
            )


def _positive(
    payload: dict[str, Any],
    key: str,
    *,
    default: int | None = None,
) -> int:
    value = payload.get(key, default)
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise GenericCausalLMConfigError(
            f"config requires integer {key!r}"
        ) from exc
    if parsed <= 0:
        raise GenericCausalLMConfigError(f"config {key!r} must be positive")
    return parsed
