from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


class OmniConfigError(ValueError):
    """Raised when an official Qwen2.5-Omni config is inconsistent."""


@dataclass(frozen=True, slots=True)
class OmniTextConfig:
    vocab_size: int
    hidden_size: int
    intermediate_size: int
    num_hidden_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    max_position_embeddings: int
    rms_norm_eps: float
    rope_theta: float
    mrope_sections: tuple[int, int, int]
    tie_word_embeddings: bool

    @property
    def head_dim(self) -> int:
        return self.hidden_size // self.num_attention_heads


@dataclass(frozen=True, slots=True)
class OmniVisionConfig:
    depth: int
    hidden_size: int
    intermediate_size: int
    num_heads: int
    in_channels: int
    patch_size: int
    spatial_merge_size: int
    temporal_patch_size: int
    window_size: int
    output_hidden_size: int
    full_attention_layers: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class OmniAudioConfig:
    num_mel_bins: int
    num_layers: int
    num_heads: int
    intermediate_size: int
    hidden_size: int
    max_source_positions: int
    window_size: int
    output_hidden_size: int
    layer_norm_eps: float
    activation_function: str


@dataclass(frozen=True, slots=True)
class Qwen25OmniConfig:
    model_type: str
    text_config: OmniTextConfig
    vision_config: OmniVisionConfig
    audio_config: OmniAudioConfig
    image_token_id: int
    video_token_id: int
    audio_token_id: int
    vision_start_token_id: int
    vision_end_token_id: int
    audio_start_token_id: int
    audio_end_token_id: int
    eos_token_id: int
    pad_token_id: int
    position_id_per_seconds: int
    seconds_per_chunk: int
    enable_audio_output: bool

    @classmethod
    def from_path(cls, model_path: str | Path) -> Qwen25OmniConfig:
        root = Path(model_path)
        path = root / "config.json" if root.is_dir() else root
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise OmniConfigError(
                f"cannot load Qwen2.5-Omni config from {path}"
            ) from exc
        if not isinstance(payload, dict):
            raise OmniConfigError("Omni config must be an object")
        if payload.get("model_type") != "qwen2_5_omni":
            raise OmniConfigError("checkpoint is not Qwen2.5-Omni")
        thinker = _object(payload, "thinker_config")
        text = _object(thinker, "text_config")
        vision = _object(thinker, "vision_config")
        audio = _object(thinker, "audio_config")
        rope = text.get("rope_scaling", {})
        if not isinstance(rope, dict):
            raise OmniConfigError("Omni rope scaling must be an object")
        sections = rope.get("mrope_section")
        if (
            not isinstance(sections, list)
            or len(sections) != 3
        ):
            raise OmniConfigError("Omni MRoPE sections are invalid")
        config = cls(
            model_type="qwen2_5_omni",
            text_config=OmniTextConfig(
                vocab_size=_integer(text, "vocab_size"),
                hidden_size=_integer(text, "hidden_size"),
                intermediate_size=_integer(text, "intermediate_size"),
                num_hidden_layers=_integer(text, "num_hidden_layers"),
                num_attention_heads=_integer(text, "num_attention_heads"),
                num_key_value_heads=_integer(text, "num_key_value_heads"),
                max_position_embeddings=_integer(
                    text,
                    "max_position_embeddings",
                ),
                rms_norm_eps=float(text.get("rms_norm_eps", 1e-6)),
                rope_theta=float(text.get("rope_theta", 1_000_000.0)),
                mrope_sections=tuple(int(value) for value in sections),
                tie_word_embeddings=bool(
                    text.get("tie_word_embeddings", False)
                ),
            ),
            vision_config=OmniVisionConfig(
                depth=_integer(vision, "depth"),
                hidden_size=_integer(vision, "hidden_size"),
                intermediate_size=_integer(vision, "intermediate_size"),
                num_heads=_integer(vision, "num_heads"),
                in_channels=_integer(vision, "in_channels"),
                patch_size=_integer(vision, "patch_size"),
                spatial_merge_size=_integer(
                    vision,
                    "spatial_merge_size",
                ),
                temporal_patch_size=_integer(
                    vision,
                    "temporal_patch_size",
                ),
                window_size=_integer(vision, "window_size"),
                output_hidden_size=_integer(
                    vision,
                    "out_hidden_size",
                ),
                full_attention_layers=tuple(
                    int(index)
                    for index in vision.get(
                        "fullatt_block_indexes",
                        (),
                    )
                ),
            ),
            audio_config=OmniAudioConfig(
                num_mel_bins=_integer(audio, "num_mel_bins"),
                num_layers=_integer(audio, "encoder_layers"),
                num_heads=_integer(audio, "encoder_attention_heads"),
                intermediate_size=_integer(audio, "encoder_ffn_dim"),
                hidden_size=_integer(audio, "d_model"),
                max_source_positions=_integer(
                    audio,
                    "max_source_positions",
                ),
                window_size=_integer(audio, "n_window"),
                output_hidden_size=_integer(audio, "output_dim"),
                layer_norm_eps=float(
                    audio.get("layer_norm_eps", 1e-5)
                ),
                activation_function=str(
                    audio.get("activation_function", "gelu")
                ),
            ),
            image_token_id=_integer(
                thinker,
                "image_token_index",
                allow_zero=True,
            ),
            video_token_id=_integer(
                thinker,
                "video_token_index",
                allow_zero=True,
            ),
            audio_token_id=_integer(
                thinker,
                "audio_token_index",
                allow_zero=True,
            ),
            vision_start_token_id=_integer(
                thinker,
                "vision_start_token_id",
                allow_zero=True,
            ),
            vision_end_token_id=_integer(
                thinker,
                "vision_end_token_id",
                allow_zero=True,
            ),
            audio_start_token_id=_integer(
                thinker,
                "audio_start_token_id",
                allow_zero=True,
            ),
            audio_end_token_id=_integer(
                thinker,
                "audio_end_token_id",
                allow_zero=True,
            ),
            eos_token_id=_integer(
                thinker,
                "eos_token_id",
                allow_zero=True,
            ),
            pad_token_id=_integer(
                thinker,
                "pad_token_id",
                allow_zero=True,
            ),
            position_id_per_seconds=_integer(
                thinker,
                "position_id_per_seconds",
            ),
            seconds_per_chunk=_integer(
                thinker,
                "seconds_per_chunk",
            ),
            enable_audio_output=bool(
                payload.get("enable_audio_output", False)
            ),
        )
        config._validate()
        return config

    @property
    def vocab_size(self) -> int:
        return self.text_config.vocab_size

    @property
    def hidden_size(self) -> int:
        return self.text_config.hidden_size

    @property
    def intermediate_size(self) -> int:
        return self.text_config.intermediate_size

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
    def max_position_embeddings(self) -> int:
        return self.text_config.max_position_embeddings

    @property
    def tie_word_embeddings(self) -> bool:
        return self.text_config.tie_word_embeddings

    def _validate(self) -> None:
        text = self.text_config
        if text.hidden_size % text.num_attention_heads:
            raise OmniConfigError("Omni hidden size is not head aligned")
        if text.num_attention_heads % text.num_key_value_heads:
            raise OmniConfigError("Omni GQA head counts are invalid")
        if sum(text.mrope_sections) * 2 != text.head_dim:
            raise OmniConfigError("Omni MRoPE sections do not cover a head")
        if self.vision_config.output_hidden_size != text.hidden_size:
            raise OmniConfigError(
                "Omni vision output width differs from text width"
            )
        if self.audio_config.output_hidden_size != text.hidden_size:
            raise OmniConfigError(
                "Omni audio output width differs from text width"
            )
        audio = self.audio_config
        if audio.hidden_size % audio.num_heads:
            raise OmniConfigError("Omni audio hidden size is not head aligned")
        if (
            audio.layer_norm_eps <= 0
            or audio.activation_function != "gelu"
        ):
            raise OmniConfigError(
                "Omni audio normalization or activation is unsupported"
            )


def _object(config: dict[str, Any], key: str) -> dict[str, Any]:
    value = config.get(key)
    if not isinstance(value, dict):
        raise OmniConfigError(f"Omni config requires object {key!r}")
    return value


def _integer(
    config: dict[str, Any],
    key: str,
    *,
    allow_zero: bool = False,
) -> int:
    try:
        value = int(config[key])
    except (KeyError, TypeError, ValueError) as exc:
        raise OmniConfigError(
            f"Omni config requires integer {key!r}"
        ) from exc
    if value < (0 if allow_zero else 1):
        raise OmniConfigError(f"Omni config {key!r} is invalid")
    return value


__all__ = [
    "OmniAudioConfig",
    "OmniConfigError",
    "OmniTextConfig",
    "OmniVisionConfig",
    "Qwen25OmniConfig",
]
