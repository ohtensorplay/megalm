from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


class Qwen35ConfigError(ValueError):
    """Raised when a Qwen3.5 checkpoint declares an unsupported geometry."""


@dataclass(frozen=True, slots=True)
class Qwen35TextConfig:
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
    hidden_act: str
    attention_bias: bool
    attn_output_gate: bool
    tie_word_embeddings: bool
    layer_types: tuple[str, ...]
    linear_conv_kernel_dim: int
    linear_key_head_dim: int
    linear_value_head_dim: int
    linear_num_key_heads: int
    linear_num_value_heads: int
    mamba_ssm_dtype: str
    rope_theta: float
    partial_rotary_factor: float
    mrope_sections: tuple[int, ...]
    mrope_interleaved: bool
    mtp_num_hidden_layers: int = 0
    num_experts: int = 0
    num_experts_per_tok: int = 0
    moe_intermediate_size: int = 0
    shared_expert_intermediate_size: int = 0
    norm_topk_prob: bool = True

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> Qwen35TextConfig:
        model_type = str(payload.get("model_type", ""))
        if model_type not in {"qwen3_5_text", "qwen3_5_moe_text"}:
            raise Qwen35ConfigError(
                f"unsupported Qwen3.5 text model_type {model_type!r}"
            )
        num_layers = _int(payload, "num_hidden_layers")
        layer_types = payload.get("layer_types")
        if layer_types is None:
            interval = int(payload.get("full_attention_interval", 4))
            if interval <= 0:
                raise Qwen35ConfigError(
                    "full_attention_interval must be positive"
                )
            layer_types = [
                (
                    "full_attention"
                    if (index + 1) % interval == 0
                    else "linear_attention"
                )
                for index in range(num_layers)
            ]
        if (
            not isinstance(layer_types, list)
            or len(layer_types) != num_layers
            or any(
                value not in {"linear_attention", "full_attention"}
                for value in layer_types
            )
        ):
            raise Qwen35ConfigError(
                "layer_types must describe every Qwen3.5 decoder layer"
            )
        rope = payload.get("rope_parameters", {})
        if not isinstance(rope, dict):
            raise Qwen35ConfigError("rope_parameters must be an object")
        sections = rope.get("mrope_section")
        if not isinstance(sections, list):
            raise Qwen35ConfigError("Qwen3.5 requires mrope_section")
        is_moe = model_type == "qwen3_5_moe_text"
        config = cls(
            model_type=model_type,
            vocab_size=_int(payload, "vocab_size"),
            hidden_size=_int(payload, "hidden_size"),
            intermediate_size=(
                int(payload.get("intermediate_size", 0)) if is_moe else _int(
                    payload, "intermediate_size"
                )
            ),
            num_hidden_layers=num_layers,
            num_attention_heads=_int(payload, "num_attention_heads"),
            num_key_value_heads=_int(payload, "num_key_value_heads"),
            head_dim=_int(payload, "head_dim"),
            max_position_embeddings=_int(payload, "max_position_embeddings"),
            rms_norm_eps=float(payload.get("rms_norm_eps", 1e-6)),
            hidden_act=str(payload.get("hidden_act", "silu")),
            attention_bias=bool(payload.get("attention_bias", False)),
            attn_output_gate=bool(payload.get("attn_output_gate", True)),
            tie_word_embeddings=bool(
                payload.get("tie_word_embeddings", False)
            ),
            layer_types=tuple(str(value) for value in layer_types),
            linear_conv_kernel_dim=_int(
                payload, "linear_conv_kernel_dim"
            ),
            linear_key_head_dim=_int(payload, "linear_key_head_dim"),
            linear_value_head_dim=_int(payload, "linear_value_head_dim"),
            linear_num_key_heads=_int(payload, "linear_num_key_heads"),
            linear_num_value_heads=_int(payload, "linear_num_value_heads"),
            mamba_ssm_dtype=str(
                payload.get("mamba_ssm_dtype", "float32")
            ),
            rope_theta=float(rope.get("rope_theta", 10_000_000.0)),
            partial_rotary_factor=float(
                rope.get("partial_rotary_factor", 0.25)
            ),
            mrope_sections=tuple(int(value) for value in sections),
            mrope_interleaved=bool(
                rope.get("mrope_interleaved", True)
            ),
            mtp_num_hidden_layers=int(
                payload.get("mtp_num_hidden_layers", 0)
            ),
            num_experts=(
                _int(payload, "num_experts") if is_moe else 0
            ),
            num_experts_per_tok=(
                _int(payload, "num_experts_per_tok") if is_moe else 0
            ),
            moe_intermediate_size=(
                _int(payload, "moe_intermediate_size") if is_moe else 0
            ),
            shared_expert_intermediate_size=(
                _int(
                    payload,
                    "shared_expert_intermediate_size",
                    allow_zero=True,
                )
                if is_moe
                else 0
            ),
            norm_topk_prob=bool(payload.get("norm_topk_prob", True)),
        )
        config.validate()
        return config

    def validate(self) -> None:
        if self.hidden_act != "silu":
            raise Qwen35ConfigError("Qwen3.5 requires SiLU")
        if self.attention_bias:
            raise Qwen35ConfigError(
                "Qwen3.5 attention bias is not supported"
            )
        if not self.attn_output_gate:
            raise Qwen35ConfigError(
                "Qwen3.5 full attention requires output gating"
            )
        if self.num_attention_heads % self.num_key_value_heads:
            raise Qwen35ConfigError(
                "full-attention head counts are not divisible"
            )
        if self.linear_num_value_heads % self.linear_num_key_heads:
            raise Qwen35ConfigError("GDN head counts are not divisible")
        if self.linear_conv_kernel_dim < 2:
            raise Qwen35ConfigError("GDN convolution requires retained state")
        rotary_dim = self.head_dim * self.partial_rotary_factor
        if (
            not rotary_dim.is_integer()
            or int(rotary_dim) % 2
            or sum(self.mrope_sections) * 2 != int(rotary_dim)
        ):
            raise Qwen35ConfigError(
                "MRoPE sections do not cover the rotary dimension"
            )
        if self.mamba_ssm_dtype != "float32":
            raise Qwen35ConfigError(
                "production GDN state currently requires float32"
            )
        if self.mtp_num_hidden_layers not in {0, 1}:
            raise Qwen35ConfigError(
                "native Qwen3.5 MTP currently requires exactly one draft layer"
            )
        if self.is_moe:
            if min(
                self.num_experts,
                self.num_experts_per_tok,
                self.moe_intermediate_size,
            ) <= 0:
                raise Qwen35ConfigError(
                    "Qwen3.5-MoE expert facts are missing"
                )
            if self.num_experts_per_tok > self.num_experts:
                raise Qwen35ConfigError(
                    "Qwen3.5-MoE top-k exceeds the expert count"
                )
        if not any(value == "full_attention" for value in self.layer_types):
            raise Qwen35ConfigError("Qwen3.5 requires full-attention layers")
        if not any(value == "linear_attention" for value in self.layer_types):
            raise Qwen35ConfigError("Qwen3.5 requires GDN layers")

    @property
    def full_layer_indices(self) -> tuple[int, ...]:
        return tuple(
            index
            for index, value in enumerate(self.layer_types)
            if value == "full_attention"
        )

    @property
    def gdn_layer_indices(self) -> tuple[int, ...]:
        return tuple(
            index
            for index, value in enumerate(self.layer_types)
            if value == "linear_attention"
        )

    @property
    def is_moe(self) -> bool:
        return self.model_type == "qwen3_5_moe_text"


@dataclass(frozen=True, slots=True)
class Qwen35VisionConfig:
    depth: int
    hidden_size: int
    intermediate_size: int
    num_heads: int
    in_channels: int
    patch_size: int
    spatial_merge_size: int
    temporal_patch_size: int
    out_hidden_size: int
    num_position_embeddings: int
    hidden_act: str

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> Qwen35VisionConfig:
        config = cls(
            depth=_int(payload, "depth"),
            hidden_size=_int(payload, "hidden_size"),
            intermediate_size=_int(payload, "intermediate_size"),
            num_heads=_int(payload, "num_heads"),
            in_channels=_int(payload, "in_channels"),
            patch_size=_int(payload, "patch_size"),
            spatial_merge_size=_int(payload, "spatial_merge_size"),
            temporal_patch_size=_int(payload, "temporal_patch_size"),
            out_hidden_size=_int(payload, "out_hidden_size"),
            num_position_embeddings=_int(
                payload, "num_position_embeddings"
            ),
            hidden_act=str(
                payload.get("hidden_act", "gelu_pytorch_tanh")
            ),
        )
        if config.hidden_size % config.num_heads:
            raise Qwen35ConfigError(
                "vision hidden size must be divisible by heads"
            )
        if config.hidden_act != "gelu_pytorch_tanh":
            raise Qwen35ConfigError(
                "unsupported Qwen3.5 vision activation"
            )
        return config


@dataclass(frozen=True, slots=True)
class Qwen35Config:
    model_type: str
    architecture: str
    text_config: Qwen35TextConfig
    vision_config: Qwen35VisionConfig
    image_token_id: int
    video_token_id: int
    vision_start_token_id: int
    vision_end_token_id: int
    tie_word_embeddings: bool

    @classmethod
    def from_path(cls, model_path: str | Path) -> Qwen35Config:
        path = Path(model_path)
        config_path = path / "config.json" if path.is_dir() else path
        try:
            payload = json.loads(config_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise Qwen35ConfigError(
                f"cannot load Qwen3.5 config from {config_path}"
            ) from exc
        if not isinstance(payload, dict):
            raise Qwen35ConfigError("Qwen3.5 config must be an object")
        model_type = str(payload.get("model_type", ""))
        if model_type not in {"qwen3_5", "qwen3_5_moe"}:
            raise Qwen35ConfigError(
                f"unsupported Qwen3.5 model_type {model_type!r}"
            )
        text_payload = payload.get("text_config")
        vision_payload = payload.get("vision_config")
        if not isinstance(text_payload, dict) or not isinstance(
            vision_payload, dict
        ):
            raise Qwen35ConfigError(
                "Qwen3.5 requires text_config and vision_config"
            )
        text = Qwen35TextConfig.from_dict(text_payload)
        vision = Qwen35VisionConfig.from_dict(vision_payload)
        if vision.out_hidden_size != text.hidden_size:
            raise Qwen35ConfigError(
                "vision output width differs from text hidden size"
            )
        expected_text_type = (
            "qwen3_5_moe_text"
            if model_type == "qwen3_5_moe"
            else "qwen3_5_text"
        )
        if text.model_type != expected_text_type:
            raise Qwen35ConfigError(
                f"{model_type} requires {expected_text_type}"
            )
        architectures = payload.get("architectures")
        default_architecture = (
            "Qwen3_5MoeForConditionalGeneration"
            if model_type == "qwen3_5_moe"
            else "Qwen3_5ForConditionalGeneration"
        )
        architecture = (
            str(architectures[0])
            if isinstance(architectures, list) and architectures
            else default_architecture
        )
        return cls(
            model_type=model_type,
            architecture=architecture,
            text_config=text,
            vision_config=vision,
            image_token_id=_int(payload, "image_token_id", allow_zero=True),
            video_token_id=_int(payload, "video_token_id", allow_zero=True),
            vision_start_token_id=_int(
                payload, "vision_start_token_id", allow_zero=True
            ),
            vision_end_token_id=_int(
                payload, "vision_end_token_id", allow_zero=True
            ),
            tie_word_embeddings=bool(
                payload.get(
                    "tie_word_embeddings",
                    text.tie_word_embeddings,
                )
            ),
        )

    @property
    def vocab_size(self) -> int:
        return self.text_config.vocab_size

    @property
    def num_hidden_layers(self) -> int:
        return self.text_config.num_hidden_layers

    @property
    def num_key_value_heads(self) -> int:
        return self.text_config.num_key_value_heads

    @property
    def head_dim(self) -> int:
        return self.text_config.head_dim

    @property
    def max_position_embeddings(self) -> int:
        return self.text_config.max_position_embeddings


def _int(
    payload: dict[str, Any],
    key: str,
    *,
    allow_zero: bool = False,
) -> int:
    try:
        value = int(payload[key])
    except (KeyError, TypeError, ValueError) as exc:
        raise Qwen35ConfigError(
            f"Qwen3.5 config requires integer {key!r}"
        ) from exc
    if value < (0 if allow_zero else 1):
        raise Qwen35ConfigError(
            f"Qwen3.5 config {key!r} has an invalid value"
        )
    return value
