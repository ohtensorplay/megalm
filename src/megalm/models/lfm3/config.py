from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


class Lfm3ConfigError(ValueError):
    """Raised when an LFM3 checkpoint declares unsupported geometry."""


@dataclass(frozen=True, slots=True)
class Lfm3Config:
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
    rope_theta_full: float
    rope_theta_sliding: float
    partial_rotary_factor: float
    sliding_window: int
    conv_L_cache: int
    conv_bias: bool
    use_sandwich_norm: bool
    layer_types: tuple[str, ...]
    bos_token_id: int
    eos_token_id: int
    pad_token_id: int
    tie_word_embeddings: bool

    @classmethod
    def from_path(cls, model_path: str | Path) -> Lfm3Config:
        root = Path(model_path)
        payload = json.loads((root / "config.json").read_text(encoding="utf-8"))
        return cls.from_dict(payload)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> Lfm3Config:
        if str(payload.get("model_type", "")) != "lfm3":
            raise Lfm3ConfigError("config model_type must be 'lfm3'")
        architectures = payload.get("architectures", ())
        if not architectures or str(architectures[0]) != "Lfm3ForCausalLM":
            raise Lfm3ConfigError("unsupported LFM3 architecture")

        hidden_size = int(payload["hidden_size"])
        num_heads = int(payload["num_attention_heads"])
        head_dim = int(payload.get("head_dim") or hidden_size // num_heads)
        layer_types = tuple(str(value) for value in payload["layer_types"])
        if len(layer_types) != int(payload["num_hidden_layers"]):
            raise Lfm3ConfigError("layer_types length must match num_hidden_layers")
        allowed = {"conv", "full_attention", "sliding_attention"}
        if any(kind not in allowed for kind in layer_types):
            raise Lfm3ConfigError("unsupported LFM3 layer type")
        if "conv" not in layer_types or "full_attention" not in layer_types:
            raise Lfm3ConfigError("LFM3 requires conv and full_attention layers")

        rope = payload.get("rope_parameters", {})
        if not isinstance(rope, dict):
            raise Lfm3ConfigError("rope_parameters must be an object")
        full_rope = rope.get("full_attention", {})
        sliding_rope = rope.get("sliding_attention", {})
        if "full_attention" in layer_types and not full_rope:
            raise Lfm3ConfigError("full_attention rope parameters are required")
        if "sliding_attention" in layer_types and not sliding_rope:
            raise Lfm3ConfigError("sliding_attention rope parameters are required")
        for params in (full_rope, sliding_rope):
            if params and str(params.get("rope_type", "default")) != "default":
                raise Lfm3ConfigError("unsupported LFM3 rope type")
        partial = float(full_rope.get("partial_rotary_factor", 1.0))
        if not 0 < partial <= 1 or (head_dim * partial) % 2:
            raise Lfm3ConfigError("invalid LFM3 partial_rotary_factor")

        eos = payload.get("eos_token_id", 124900)
        if isinstance(eos, (list, tuple)):
            eos = int(eos[0]) if eos else 124900
        return cls(
            model_type="lfm3",
            vocab_size=int(payload["vocab_size"]),
            hidden_size=hidden_size,
            intermediate_size=int(payload["intermediate_size"]),
            num_hidden_layers=int(payload["num_hidden_layers"]),
            num_attention_heads=num_heads,
            num_key_value_heads=int(payload["num_key_value_heads"]),
            head_dim=head_dim,
            max_position_embeddings=int(payload["max_position_embeddings"]),
            rms_norm_eps=float(payload.get("norm_eps", 1e-5)),
            rope_theta_full=float(full_rope.get("rope_theta", 1e6)),
            rope_theta_sliding=float(sliding_rope.get("rope_theta", 1e4)),
            partial_rotary_factor=partial,
            sliding_window=int(payload.get("sliding_window") or 256),
            conv_L_cache=int(payload.get("conv_L_cache", 7)),
            conv_bias=bool(payload.get("conv_bias", False)),
            use_sandwich_norm=bool(payload.get("use_sandwich_norm", True)),
            layer_types=layer_types,
            bos_token_id=int(payload.get("bos_token_id", 124894)),
            eos_token_id=int(eos),
            pad_token_id=int(payload.get("pad_token_id", 124893)),
            tie_word_embeddings=bool(payload.get("tie_word_embeddings", True)),
        )

    @property
    def attention_layers(self) -> tuple[int, ...]:
        return tuple(
            index
            for index, kind in enumerate(self.layer_types)
            if kind != "conv"
        )

    @property
    def conv_layers(self) -> tuple[int, ...]:
        return tuple(
            index for index, kind in enumerate(self.layer_types) if kind == "conv"
        )
