from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


class Qwen3VLConfigError(ValueError):
    """Raised when a Qwen3-VL checkpoint declares unsupported geometry."""


@dataclass(frozen=True, slots=True)
class Qwen3VLTextConfig:
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
    partial_rotary_factor: float
    mrope_sections: tuple[int, ...]
    mrope_interleaved: bool
    tie_word_embeddings: bool
    num_experts: int = 0
    num_experts_per_tok: int = 0
    moe_intermediate_size: int = 0
    norm_topk_prob: bool = False

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> Qwen3VLTextConfig:
        model_type = str(payload.get("model_type", ""))
        expected = {"qwen3_vl_text", "qwen3_vl_moe_text"}
        if model_type not in expected:
            raise Qwen3VLConfigError(
                f"unsupported Qwen3-VL text model_type {model_type!r}"
            )
        is_moe = model_type == "qwen3_vl_moe_text"
        rope = payload.get("rope_parameters", payload.get("rope_scaling", {}))
        if not isinstance(rope, dict):
            raise Qwen3VLConfigError("Qwen3-VL rope parameters must be an object")
        sections = rope.get("mrope_section")
        if not isinstance(sections, list) or not sections:
            raise Qwen3VLConfigError("Qwen3-VL requires mrope_section")
        sections = tuple(int(value) for value in sections)
        if len(sections) != 3:
            raise Qwen3VLConfigError(
                "Qwen3-VL MRoPE requires exactly three sections"
            )
        head_dim = _positive(payload, "head_dim")
        rotary_dim = sum(sections) * 2
        if rotary_dim > head_dim:
            raise Qwen3VLConfigError(
                "MRoPE sections exceed the attention head dimension"
            )
        partial = float(
            rope.get("partial_rotary_factor", rotary_dim / head_dim)
        )
        # Transformers 4.57 moved ``rope_theta`` out of ``rope_scaling`` and
        # keeps it on the text config itself, while the hub checkpoints still
        # ship a ``rope_scaling`` block.  Read the outer key as the fallback so
        # the RoPE base is not silently reset to 1e6.
        rope_theta = float(
            rope.get("rope_theta", payload.get("rope_theta", 1_000_000.0))
        )
        if abs(partial * head_dim - rotary_dim) > 1e-6:
            raise Qwen3VLConfigError(
                "MRoPE sections do not cover the partial rotary dimension"
            )
        config = cls(
            model_type=model_type,
            vocab_size=_positive(payload, "vocab_size"),
            hidden_size=_positive(payload, "hidden_size"),
            intermediate_size=(
                _positive(payload, "intermediate_size")
                if not is_moe
                else 1
            ),
            num_hidden_layers=_positive(payload, "num_hidden_layers"),
            num_attention_heads=_positive(payload, "num_attention_heads"),
            num_key_value_heads=_positive(payload, "num_key_value_heads"),
            head_dim=head_dim,
            max_position_embeddings=_positive(
                payload, "max_position_embeddings"
            ),
            rms_norm_eps=float(payload.get("rms_norm_eps", 1e-6)),
            rope_theta=rope_theta,
            partial_rotary_factor=partial,
            mrope_sections=sections,
            mrope_interleaved=bool(rope.get("mrope_interleaved", True)),
            tie_word_embeddings=bool(
                payload.get("tie_word_embeddings", False)
            ),
            num_experts=_positive(payload, "num_experts") if is_moe else 0,
            num_experts_per_tok=(
                _positive(payload, "num_experts_per_tok") if is_moe else 0
            ),
            moe_intermediate_size=(
                _positive(payload, "moe_intermediate_size") if is_moe else 0
            ),
            norm_topk_prob=bool(payload.get("norm_topk_prob", False)),
        )
        config.validate()
        return config

    def validate(self) -> None:
        if self.hidden_size != self.num_attention_heads * self.head_dim:
            raise Qwen3VLConfigError(
                "hidden_size and attention heads disagree"
            )
        if self.num_attention_heads % self.num_key_value_heads:
            raise Qwen3VLConfigError("GQA head counts are not divisible")
        if self.head_dim % 2:
            raise Qwen3VLConfigError("RoPE head_dim must be even")
        if self.rms_norm_eps <= 0 or self.rope_theta <= 0:
            raise Qwen3VLConfigError("norm and RoPE facts must be positive")
        if self.num_experts:
            if min(
                self.num_experts_per_tok,
                self.moe_intermediate_size,
            ) <= 0:
                raise Qwen3VLConfigError("MoE expert facts are missing")
            if self.num_experts_per_tok > self.num_experts:
                raise Qwen3VLConfigError(
                    "Qwen3-VL-MoE top-k exceeds the expert count"
                )

    @property
    def is_moe(self) -> bool:
        return self.model_type == "qwen3_vl_moe_text"


@dataclass(frozen=True, slots=True)
class Qwen3VLVisionConfig:
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
    deepstack_visual_indexes: tuple[int, ...]

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> Qwen3VLVisionConfig:
        config = cls(
            depth=_positive(payload, "depth"),
            hidden_size=_positive(payload, "hidden_size"),
            intermediate_size=_positive(payload, "intermediate_size"),
            num_heads=_positive(payload, "num_heads"),
            in_channels=_positive(payload, "in_channels"),
            patch_size=_positive(payload, "patch_size"),
            spatial_merge_size=_positive(payload, "spatial_merge_size"),
            temporal_patch_size=_positive(
                payload, "temporal_patch_size"
            ),
            out_hidden_size=_positive(payload, "out_hidden_size"),
            num_position_embeddings=_positive(
                payload, "num_position_embeddings"
            ),
            hidden_act=str(payload.get("hidden_act", "silu")),
            deepstack_visual_indexes=tuple(
                int(index)
                for index in payload.get("deepstack_visual_indexes", ())
            ),
        )
        if config.hidden_size % config.num_heads:
            raise Qwen3VLConfigError(
                "vision hidden size must be divisible by heads"
            )
        if config.hidden_act not in {"silu", "gelu_pytorch_tanh"}:
            raise Qwen3VLConfigError(
                f"unsupported Qwen3-VL vision activation {config.hidden_act!r}"
            )
        if any(
            index < 0 or index >= config.depth
            for index in config.deepstack_visual_indexes
        ):
            raise Qwen3VLConfigError(
                "deepstack_visual_indexes contains an invalid block index"
            )
        return config

    @property
    def deepstack_depth(self) -> int:
        return len(self.deepstack_visual_indexes)


@dataclass(frozen=True, slots=True)
class Qwen3VLQuantConfig:
    """Weight-only int4 metadata for a compressed-tensors checkpoint.

    Only the ``pack-quantized`` W4A16 form emitted by llm-compressor / AWQ is
    accepted: one int32 word carries eight 4-bit codes (low nibble first) and
    each ``group_size`` consecutive inputs share a bf16 scale.
    """

    quant_method: str
    format: str
    num_bits: int
    group_size: int
    symmetric: bool

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> Qwen3VLQuantConfig:
        quant_method = str(payload.get("quant_method", ""))
        if quant_method != "compressed-tensors":
            raise Qwen3VLConfigError(
                f"unsupported Qwen3-VL quant_method {quant_method!r}"
            )
        fmt = str(payload.get("format", ""))
        if fmt != "pack-quantized":
            raise Qwen3VLConfigError(
                f"unsupported Qwen3-VL quant format {fmt!r}"
            )
        groups = payload.get("config_groups")
        if not isinstance(groups, dict) or len(groups) != 1:
            raise Qwen3VLConfigError(
                "Qwen3-VL weight-only quantization needs exactly one group"
            )
        group = next(iter(groups.values()))
        if not isinstance(group, dict):
            raise Qwen3VLConfigError(
                "Qwen3-VL quantization group must be an object"
            )
        if group.get("input_activations") not in (None, {}):
            raise Qwen3VLConfigError(
                "Qwen3-VL activation quantization is not supported"
            )
        weights = group.get("weights")
        if not isinstance(weights, dict):
            raise Qwen3VLConfigError(
                "Qwen3-VL quantization group has no weight scheme"
            )
        targets = group.get("targets")
        if not isinstance(targets, list) or "Linear" not in targets:
            raise Qwen3VLConfigError(
                "Qwen3-VL weight-only quantization must target Linear"
            )
        num_bits = int(weights.get("num_bits", 0))
        group_size = int(weights.get("group_size", 0))
        if num_bits != 4:
            raise Qwen3VLConfigError(
                f"only int4 Qwen3-VL weights are supported, got {num_bits}"
            )
        if group_size not in {32, 64, 128}:
            raise Qwen3VLConfigError(
                f"unsupported int4 group_size {group_size}"
            )
        if not bool(weights.get("symmetric", True)):
            raise Qwen3VLConfigError(
                "asymmetric int4 Qwen3-VL weights are not supported"
            )
        return cls(
            quant_method=quant_method,
            format=fmt,
            num_bits=num_bits,
            group_size=group_size,
            symmetric=bool(weights.get("symmetric", True)),
        )


@dataclass(frozen=True, slots=True)
class Qwen3VLConfig:
    model_type: str
    architecture: str
    text_config: Qwen3VLTextConfig
    vision_config: Qwen3VLVisionConfig
    image_token_id: int
    video_token_id: int
    vision_start_token_id: int
    vision_end_token_id: int
    tie_word_embeddings: bool
    quantization: Qwen3VLQuantConfig | None = None

    @classmethod
    def from_path(cls, model_path: str | Path) -> Qwen3VLConfig:
        path = Path(model_path)
        config_path = path / "config.json" if path.is_dir() else path
        try:
            payload = json.loads(config_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise Qwen3VLConfigError(
                f"cannot load Qwen3-VL config from {config_path}"
            ) from exc
        if not isinstance(payload, dict):
            raise Qwen3VLConfigError("Qwen3-VL config must be an object")
        model_type = str(payload.get("model_type", ""))
        if model_type not in {"qwen3_vl", "qwen3_vl_moe"}:
            raise Qwen3VLConfigError(
                f"unsupported Qwen3-VL model_type {model_type!r}"
            )
        text_payload = payload.get("text_config")
        vision_payload = payload.get("vision_config")
        if not isinstance(text_payload, dict) or not isinstance(
            vision_payload, dict
        ):
            raise Qwen3VLConfigError(
                "Qwen3-VL requires text_config and vision_config"
            )
        text = Qwen3VLTextConfig.from_dict(text_payload)
        vision = Qwen3VLVisionConfig.from_dict(vision_payload)
        expected = (
            "qwen3_vl_moe_text" if model_type == "qwen3_vl_moe" else "qwen3_vl_text"
        )
        if text.model_type != expected:
            raise Qwen3VLConfigError(f"{model_type} requires {expected}")
        if vision.out_hidden_size != text.hidden_size:
            raise Qwen3VLConfigError(
                "vision output width differs from the text hidden size"
            )
        quant_payload = payload.get("quantization_config")
        if quant_payload is not None and not isinstance(quant_payload, dict):
            raise Qwen3VLConfigError(
                "Qwen3-VL quantization_config must be an object"
            )
        quantization = (
            Qwen3VLQuantConfig.from_dict(quant_payload)
            if quant_payload is not None
            else None
        )
        if quantization is not None and model_type == "qwen3_vl_moe":
            raise Qwen3VLConfigError(
                "quantized Qwen3-VL-MoE experts are not supported"
            )
        architectures = payload.get("architectures")
        architecture = (
            str(architectures[0])
            if isinstance(architectures, list) and architectures
            else (
                "Qwen3VLMoeForConditionalGeneration"
                if model_type == "qwen3_vl_moe"
                else "Qwen3VLForConditionalGeneration"
            )
        )
        return cls(
            model_type=model_type,
            architecture=architecture,
            text_config=text,
            vision_config=vision,
            image_token_id=_nonnegative(payload, "image_token_id"),
            video_token_id=_nonnegative(payload, "video_token_id"),
            vision_start_token_id=_nonnegative(
                payload, "vision_start_token_id"
            ),
            vision_end_token_id=_nonnegative(payload, "vision_end_token_id"),
            tie_word_embeddings=bool(
                payload.get(
                    "tie_word_embeddings",
                    text.tie_word_embeddings,
                )
            ),
            quantization=quantization,
        )

    @property
    def vocab_size(self) -> int:
        return self.text_config.vocab_size

    # The engine and runtime address the decoder geometry directly; Qwen3-VL
    # keeps it inside ``text_config``, so expose the same flat names as the
    # text-only Qwen configs.
    @property
    def num_hidden_layers(self) -> int:
        return self.text_config.num_hidden_layers

    @property
    def num_attention_heads(self) -> int:
        return self.text_config.num_attention_heads

    @property
    def num_key_value_heads(self) -> int:
        return self.text_config.num_key_value_heads

    @property
    def head_dim(self) -> int:
        return self.text_config.head_dim

    @property
    def hidden_size(self) -> int:
        return self.text_config.hidden_size

    @property
    def max_position_embeddings(self) -> int:
        return self.text_config.max_position_embeddings


def _positive(payload: dict[str, Any], key: str) -> int:
    try:
        value = int(payload[key])
    except (KeyError, TypeError, ValueError) as exc:
        raise Qwen3VLConfigError(
            f"Qwen3-VL config requires integer {key!r}"
        ) from exc
    if value <= 0:
        raise Qwen3VLConfigError(f"Qwen3-VL config {key!r} must be positive")
    return value


def _nonnegative(payload: dict[str, Any], key: str) -> int:
    try:
        value = int(payload[key])
    except (KeyError, TypeError, ValueError) as exc:
        raise Qwen3VLConfigError(
            f"Qwen3-VL config requires integer {key!r}"
        ) from exc
    if value < 0:
        raise Qwen3VLConfigError(
            f"Qwen3-VL config {key!r} must be non-negative"
        )
    return value
