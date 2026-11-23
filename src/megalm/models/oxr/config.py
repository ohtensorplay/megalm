from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from megalm.models.qwen3 import Qwen3Config


class OXRConfigError(ValueError):
    """Raised when an OXR checkpoint config is incomplete or inconsistent."""


@dataclass(frozen=True, slots=True)
class OXRVisionConfig:
    """SigLIP2-NAViT tower geometry used by OXR-1.0."""

    patch_size: int
    hidden_stride: int
    temporal_patch_size: int
    num_channels: int
    image_size: int
    hidden_size: int
    intermediate_size: int
    num_attention_heads: int
    num_hidden_layers: int
    layer_norm_eps: float
    hidden_act: str
    window_size: int
    use_rope: bool
    preserve_original_pe: bool
    num_patches: int
    fullatt_block_indexes: tuple[int, ...] | None
    # megaLM media-admission contract, derived from hidden_stride.
    spatial_merge_size: int

    @property
    def head_dim(self) -> int:
        return self.hidden_size // self.num_attention_heads

    @property
    def position_embedding_size(self) -> int:
        if self.num_patches > 0:
            return int(self.num_patches**0.5)
        return self.image_size // self.patch_size

    @property
    def num_position_embeddings(self) -> int:
        return self.position_embedding_size**2

    @property
    def rotary_dim(self) -> int:
        return self.head_dim // 2

    @property
    def vision_tokens_per_merge(self) -> int:
        return self.hidden_stride * self.hidden_stride


@dataclass(frozen=True, slots=True)
class OXRConfig:
    """Top-level OXR config plus the embedded Qwen3 LLM config."""

    model_type: str
    architectures: tuple[str, ...]
    vocab_size: int
    hidden_size: int
    visual_vocab_size: int
    indicator_count: int
    max_position_embeddings: int
    image_token_id: int
    vision_start_token_id: int
    vision_end_token_id: int
    visual_atom_token_id: int
    image_start_token_id: int
    image_end_token_id: int
    video_start_token_id: int
    video_end_token_id: int
    llm_config: Qwen3Config
    vision_config: OXRVisionConfig

    @property
    def num_indicators(self) -> int:
        return self.indicator_count

    @property
    def num_hidden_layers(self) -> int:
        return self.llm_config.num_hidden_layers

    @property
    def num_key_value_heads(self) -> int:
        return self.llm_config.num_key_value_heads

    @property
    def head_dim(self) -> int:
        return self.llm_config.head_dim

    @property
    def visual_head_dim(self) -> int:
        return self.visual_vocab_size - self.indicator_count

    @classmethod
    def from_path(cls, model_path: str | Path) -> OXRConfig:
        path = Path(model_path)
        config_path = path / "config.json" if path.is_dir() else path
        try:
            payload = json.loads(config_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise OXRConfigError(
                f"cannot load OXR config from {config_path}"
            ) from exc
        return cls.from_dict(payload)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> OXRConfig:
        if not isinstance(payload, dict):
            raise OXRConfigError("OXR config must be a JSON object")
        model_type = str(payload.get("model_type", ""))
        if model_type != "oxr":
            raise OXRConfigError(f"unsupported OXR model_type {model_type!r}")

        llm_payload = payload.get("llm_config")
        vit_payload = payload.get("vit_config")
        if not isinstance(llm_payload, dict) or not isinstance(vit_payload, dict):
            raise OXRConfigError("OXR config requires llm_config and vit_config objects")
        llm = Qwen3Config.from_dict(llm_payload)
        if llm.model_type != "qwen3" or llm.num_experts:
            raise OXRConfigError("OXR requires a dense Qwen3 LLM")
        if int(payload.get("hidden_size", llm.hidden_size)) != llm.hidden_size:
            raise OXRConfigError("OXR hidden size disagrees with its LLM")
        if int(payload.get("vocab_size", llm.vocab_size)) != llm.vocab_size:
            raise OXRConfigError("OXR vocab size disagrees with its LLM")

        visual_vocab_size = int(payload["visual_vocab_size"])
        indicator_count = 4
        if visual_vocab_size <= indicator_count:
            raise OXRConfigError("OXR visual_vocab_size is too small")
        hidden_stride = int(vit_payload["hidden_stride"])
        if hidden_stride <= 0:
            raise OXRConfigError("OXR hidden_stride must be positive")
        fullatt = vit_payload.get("fullatt_block_indexes")
        fullatt_blocks = (
            None
            if fullatt is None
            else tuple(int(value) for value in str(fullatt).split("|") if value)
        )
        vision = OXRVisionConfig(
            patch_size=int(vit_payload["patch_size"]),
            hidden_stride=hidden_stride,
            temporal_patch_size=int(vit_payload.get("temporal_patch_size", 1)),
            num_channels=int(vit_payload.get("num_channels", 3)),
            image_size=int(vit_payload.get("image_size", 512)),
            hidden_size=int(vit_payload["hidden_size"]),
            intermediate_size=int(vit_payload["intermediate_size"]),
            num_attention_heads=int(vit_payload["num_attention_heads"]),
            num_hidden_layers=int(vit_payload["num_hidden_layers"]),
            layer_norm_eps=float(vit_payload.get("layer_norm_eps", 1e-6)),
            hidden_act=str(vit_payload.get("hidden_act", "gelu_pytorch_tanh")),
            window_size=int(vit_payload.get("window_size", 112)),
            use_rope=bool(vit_payload.get("use_rope", True)),
            preserve_original_pe=bool(
                vit_payload.get("preserve_original_pe", True)
            ),
            num_patches=int(vit_payload.get("num_patches", -1)),
            fullatt_block_indexes=fullatt_blocks,
            spatial_merge_size=hidden_stride,
        )
        if vision.head_dim * vision.num_attention_heads != vision.hidden_size:
            raise OXRConfigError("OXR ViT hidden size and heads disagree")
        if fullatt_blocks is not None:
            raise OXRConfigError(
                "OXR partial full-attention ViT blocks are not supported"
            )
        # Slots reserved in the text vocabulary for visual placeholder rows.
        # 151655 is the tokenizer's <|image_pad|>; the remaining ids are
        # unused by the OXR tokenizer and never produced by generation.
        return cls(
            model_type=model_type,
            architectures=tuple(payload.get("architectures", ()) or ()),
            vocab_size=llm.vocab_size,
            hidden_size=llm.hidden_size,
            visual_vocab_size=visual_vocab_size,
            indicator_count=indicator_count,
            max_position_embeddings=llm.max_position_embeddings,
            image_token_id=151655,
            vision_start_token_id=151652,
            vision_end_token_id=151653,
            visual_atom_token_id=151930,
            image_start_token_id=151931,
            image_end_token_id=151932,
            video_start_token_id=151933,
            video_end_token_id=151934,
            llm_config=llm,
            vision_config=vision,
        )


__all__ = ["OXRConfig", "OXRConfigError", "OXRVisionConfig"]
