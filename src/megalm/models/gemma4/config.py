"""Configuration objects for the Gemma-4 family.

The Gemma-4 checkpoints are *nested*: the file on disk is a ``Gemma4Config``
whose ``text_config`` carries the decoder geometry.  A few checkpoints ship a
bare text config (``gemma4_text`` / ``gemma4_unified_text``).  Both shapes are
accepted here so the rest of the runtime only ever sees ``Gemma4TextConfig``.

All geometry that varies per layer is resolved once, at config-load time.  In
particular the full-attention layers may have a different head dimension and
number of KV heads than the sliding-attention layers, and the trailing YOCO
layers share their KV state with an earlier layer of the same type.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

FULL_ATTENTION = "full_attention"
SLIDING_ATTENTION = "sliding_attention"
_LAYER_TYPES = (FULL_ATTENTION, SLIDING_ATTENTION)


class Gemma4ConfigError(ValueError):
    """Raised when a Gemma-4 checkpoint declares unsupported geometry."""


@dataclass(frozen=True, slots=True)
class Gemma4VisionConfig:
    """Vision-tower geometry used by the tower and unified variants."""

    hidden_size: int = 768
    intermediate_size: int = 3072
    num_hidden_layers: int = 16
    num_attention_heads: int = 12
    num_key_value_heads: int = 12
    head_dim: int = 64
    rope_theta: float = 100.0
    patch_size: int = 16
    pooling_kernel_size: int = 3
    position_embedding_size: int = 10_240
    default_output_length: int = 280
    rms_norm_eps: float = 1e-6
    hidden_act: str = "gelu_pytorch_tanh"
    max_position_embeddings: int = 131_072
    use_clipped_linears: bool = False
    standardize: bool = False
    mm_embed_dim: int = 0
    mm_posemb_size: int = 0
    num_soft_tokens: int = 0

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> Gemma4VisionConfig:
        if "global_head_dim" in payload:
            # vLLM configs may carry a global override; the encoder itself is
            # always uniform and uses head_dim.
            payload = {key: value for key, value in payload.items() if key != "global_head_dim"}
        return cls(
            hidden_size=int(payload.get("hidden_size", 768)),
            intermediate_size=int(payload.get("intermediate_size", 3072)),
            num_hidden_layers=int(payload.get("num_hidden_layers", 16)),
            num_attention_heads=int(payload.get("num_attention_heads", 12)),
            num_key_value_heads=int(payload.get("num_key_value_heads", 12)),
            head_dim=int(payload.get("head_dim", 64)),
            rope_theta=float(
                (payload.get("rope_parameters") or {}).get("rope_theta", 100.0)
            ),
            patch_size=int(payload.get("patch_size", 16)),
            pooling_kernel_size=int(payload.get("pooling_kernel_size", 3)),
            position_embedding_size=int(payload.get("position_embedding_size", 10_240)),
            default_output_length=int(payload.get("default_output_length", 280)),
            rms_norm_eps=float(payload.get("rms_norm_eps", 1e-6)),
            hidden_act=str(payload.get("hidden_activation", "gelu_pytorch_tanh")),
            max_position_embeddings=int(payload.get("max_position_embeddings", 131_072)),
            use_clipped_linears=bool(payload.get("use_clipped_linears", False)),
            standardize=bool(payload.get("standardize", False)),
            mm_embed_dim=int(payload.get("mm_embed_dim", 0)),
            mm_posemb_size=int(payload.get("mm_posemb_size", 0)),
            num_soft_tokens=int(payload.get("num_soft_tokens", 0)),
        )


@dataclass(frozen=True, slots=True)
class Gemma4AudioConfig:
    """Conformer audio-tower geometry (only built when the checkpoint has it)."""

    hidden_size: int = 1024
    num_hidden_layers: int = 12
    num_attention_heads: int = 8
    num_mel_bins: int = 128
    sampling_rate: int = 16_000
    frame_length: int = 320
    hop_length: int = 160
    fft_length: int = 512
    min_frequency: float = 0.0
    max_frequency: float = 8_000.0
    mel_floor: float = 1e-3
    audio_seq_length: int = 750
    hidden_act: str = "silu"
    conv_kernel_size: int = 5
    subsampling_conv_channels: tuple[int, ...] = (128, 32)
    attention_chunk_size: int = 12
    attention_context_left: int = 13
    attention_context_right: int = 0
    attention_logit_cap: float = 50.0
    attention_invalid_logits_value: float = -1e9
    residual_weight: float = 0.5
    output_proj_dims: int = 1536
    rms_norm_eps: float = 1e-6
    gradient_clipping: float = 1e10
    use_clipped_linears: bool = True

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> Gemma4AudioConfig:
        channels = payload.get("subsampling_conv_channels", (128, 32))
        if isinstance(channels, (list, tuple)):
            channels = tuple(int(value) for value in channels)
        else:
            channels = (128, 32)
        return cls(
            hidden_size=int(payload.get("hidden_size", 1024)),
            num_hidden_layers=int(payload.get("num_hidden_layers", 12)),
            num_attention_heads=int(payload.get("num_attention_heads", 8)),
            num_mel_bins=int(payload.get("num_mel_bins", 128)),
            sampling_rate=int(payload.get("sampling_rate", 16_000)),
            frame_length=int(payload.get("frame_length", 320)),
            hop_length=int(payload.get("hop_length", 160)),
            fft_length=int(payload.get("fft_length", 512)),
            min_frequency=float(payload.get("min_frequency", 0.0)),
            max_frequency=float(payload.get("max_frequency", 8_000.0)),
            mel_floor=float(payload.get("mel_floor", 1e-3)),
            audio_seq_length=int(payload.get("audio_seq_length", 750)),
            hidden_act=str(payload.get("hidden_act", "silu")),
            conv_kernel_size=int(payload.get("conv_kernel_size", 5)),
            subsampling_conv_channels=channels,
            attention_chunk_size=int(payload.get("attention_chunk_size", 12)),
            attention_context_left=int(payload.get("attention_context_left", 13)),
            attention_context_right=int(payload.get("attention_context_right", 0)),
            attention_logit_cap=float(payload.get("attention_logit_cap", 50.0)),
            attention_invalid_logits_value=float(
                payload.get("attention_invalid_logits_value", -1e9)
            ),
            residual_weight=float(payload.get("residual_weight", 0.5)),
            output_proj_dims=int(payload.get("output_proj_dims", 1536)),
            rms_norm_eps=float(payload.get("rms_norm_eps", 1e-6)),
            gradient_clipping=float(payload.get("gradient_clipping", 1e10)),
            use_clipped_linears=bool(payload.get("use_clipped_linears", True)),
        )


@dataclass(frozen=True, slots=True)
class Gemma4TextConfig:
    """Resolved, layer-aware text configuration."""

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
    hidden_activation: str
    tie_word_embeddings: bool
    attention_bias: bool
    sliding_window: int
    layer_types: tuple[str, ...]
    layer_head_dims: tuple[int, ...]
    layer_num_key_value_heads: tuple[int, ...]
    rope_theta: dict[str, float]
    rope_partial_factor: dict[str, float]
    rope_type: dict[str, str]
    final_logit_softcapping: float | None
    attn_logit_softcapping: float | None
    use_bidirectional_attention: str | None
    vocab_size_per_layer_input: int
    hidden_size_per_layer_input: int
    attention_k_eq_v: bool
    num_kv_shared_layers: int
    enable_moe_block: bool
    use_double_wide_mlp: bool
    num_experts: int
    top_k_experts: int
    moe_intermediate_size: int
    # Derived fields -----------------------------------------------------
    kv_slots: tuple[int, ...] = ()
    kv_layer_indices: tuple[int, ...] = ()
    kv_share_sources: tuple[int, ...] = ()
    vision_config: Gemma4VisionConfig | None = None
    audio_config: Gemma4AudioConfig | None = None

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> Gemma4TextConfig:
        model_type = str(payload.get("model_type") or "")
        if model_type not in {"gemma4_text", "gemma4_unified_text"}:
            raise Gemma4ConfigError(f"unsupported Gemma-4 text model_type {model_type!r}")

        hidden_size = _positive(payload, "hidden_size")
        num_layers = _positive(payload, "num_hidden_layers")
        num_heads = _positive(payload, "num_attention_heads")
        kv_heads = _positive(
            payload, "num_key_value_heads", default=num_heads
        )
        head_dim = _positive(
            payload, "head_dim", default=hidden_size // num_heads
        )
        global_head_dim = _positive(
            payload, "global_head_dim", default=head_dim
        )
        attention_k_eq_v = bool(payload.get("attention_k_eq_v", False))
        global_kv_heads = payload.get("num_global_key_value_heads")
        global_kv_heads = int(global_kv_heads) if global_kv_heads else kv_heads

        layer_types = _layer_types(payload, num_layers)
        layer_head_dims = tuple(
            global_head_dim if kind == FULL_ATTENTION else head_dim
            for kind in layer_types
        )
        # HF only applies the global KV-head override for k_eq_v checkpoints.
        layer_kv_heads = tuple(
            global_kv_heads
            if (kind == FULL_ATTENTION and attention_k_eq_v)
            else kv_heads
            for kind in layer_types
        )

        rope_parameters = payload.get("rope_parameters")
        if rope_parameters is None or isinstance(rope_parameters, dict) and (
            FULL_ATTENTION not in rope_parameters
            and SLIDING_ATTENTION not in rope_parameters
        ):
            # Legacy single-dict rope config: treat it as the sliding flavour
            # and synthesise the proportional full-attention flavour.
            legacy = dict(rope_parameters or {})
            rope_parameters = {
                SLIDING_ATTENTION: {
                    "rope_type": "default",
                    "rope_theta": float(legacy.get("rope_theta", 10_000.0)),
                },
                FULL_ATTENTION: {
                    "rope_type": "proportional",
                    "partial_rotary_factor": 0.25,
                    "rope_theta": 1_000_000.0,
                },
            }
        if not isinstance(rope_parameters, dict):
            raise Gemma4ConfigError("rope_parameters must be an object")
        rope_theta: dict[str, float] = {}
        rope_partial: dict[str, float] = {}
        rope_type: dict[str, str] = {}
        for kind in dict.fromkeys(layer_types):
            params = rope_parameters.get(kind)
            if not isinstance(params, dict):
                raise Gemma4ConfigError(
                    f"rope_parameters[{kind!r}] must be an object"
                )
            rope_type_value = str(params.get("rope_type", "default"))
            if rope_type_value not in {"default", "proportional"}:
                raise Gemma4ConfigError(
                    f"unsupported Gemma-4 rope_type {rope_type_value!r}"
                )
            rope_type[kind] = rope_type_value
            rope_theta[kind] = float(
                params.get("rope_theta", 10_000.0 if kind == SLIDING_ATTENTION else 1_000_000.0)
            )
            partial = params.get("partial_rotary_factor")
            if rope_type_value == "proportional":
                try:
                    partial = float(partial)
                except (TypeError, ValueError) as exc:
                    raise Gemma4ConfigError(
                        "proportional RoPE requires partial_rotary_factor"
                    ) from exc
                if not 0 < partial <= 1:
                    raise Gemma4ConfigError("partial_rotary_factor must be in (0, 1]")
            else:
                partial = 1.0
            rope_partial[kind] = partial

        eos = payload.get("eos_token_id", 1)
        if isinstance(eos, (list, tuple)):
            eos = eos[0] if eos else 1
        sliding_window = _positive(payload, "sliding_window", default=512)
        bidirectional = payload.get("use_bidirectional_attention")
        if bidirectional is not None and bidirectional not in {"all", "vision"}:
            raise Gemma4ConfigError(
                "use_bidirectional_attention must be 'all', 'vision' or null"
            )
        num_kv_shared = int(payload.get("num_kv_shared_layers", 0) or 0)
        if not 0 <= num_kv_shared < num_layers:
            raise Gemma4ConfigError("num_kv_shared_layers is out of range")
        enable_moe = bool(payload.get("enable_moe_block", False))
        num_experts = int(payload.get("num_experts") or 0)
        top_k = int(
            payload.get("top_k_experts")
            or payload.get("num_experts_per_tok")
            or 0
        )
        moe_intermediate = int(
            payload.get("moe_intermediate_size")
            or payload.get("expert_intermediate_size")
            or 0
        )
        if enable_moe and min(num_experts, top_k, moe_intermediate) <= 0:
            raise Gemma4ConfigError("MoE geometry is incomplete")
        if top_k > num_experts and num_experts:
            raise Gemma4ConfigError("top_k_experts exceeds num_experts")

        softcap = payload.get("final_logit_softcapping")
        attn_softcap = payload.get("attn_logit_softcapping")
        if attn_softcap is None:
            attn_softcap = payload.get("attn_logit_soft_cap")
        config = cls(
            model_type=model_type,
            vocab_size=_positive(payload, "vocab_size"),
            hidden_size=hidden_size,
            intermediate_size=_positive(payload, "intermediate_size"),
            num_hidden_layers=num_layers,
            num_attention_heads=num_heads,
            num_key_value_heads=kv_heads,
            head_dim=head_dim,
            max_position_embeddings=_positive(payload, "max_position_embeddings"),
            rms_norm_eps=float(payload.get("rms_norm_eps", 1e-6)),
            hidden_activation=str(
                payload.get("hidden_activation", "gelu_pytorch_tanh")
            ),
            tie_word_embeddings=bool(payload.get("tie_word_embeddings", False)),
            attention_bias=bool(payload.get("attention_bias", False)),
            sliding_window=sliding_window,
            layer_types=layer_types,
            layer_head_dims=layer_head_dims,
            layer_num_key_value_heads=layer_kv_heads,
            rope_theta=rope_theta,
            rope_partial_factor=rope_partial,
            rope_type=rope_type,
            final_logit_softcapping=float(softcap) if softcap is not None else None,
            attn_logit_softcapping=float(attn_softcap) if attn_softcap is not None else None,
            use_bidirectional_attention=bidirectional,
            vocab_size_per_layer_input=int(
                payload.get("vocab_size_per_layer_input", 0) or 0
            ),
            hidden_size_per_layer_input=int(
                payload.get("hidden_size_per_layer_input", 0) or 0
            ),
            attention_k_eq_v=attention_k_eq_v,
            num_kv_shared_layers=num_kv_shared,
            enable_moe_block=enable_moe,
            use_double_wide_mlp=bool(payload.get("use_double_wide_mlp", False)),
            num_experts=num_experts,
            top_k_experts=top_k,
            moe_intermediate_size=moe_intermediate,
        )
        config = config._with_kv_layout()
        config.validate()
        return config

    def _with_kv_layout(self) -> Gemma4TextConfig:
        """Map every layer onto the KV slot that owns its keys and values."""
        first_shared = self.num_hidden_layers - self.num_kv_shared_layers
        owners: dict[str, int] = {}
        sources: list[int] = []
        for index, kind in enumerate(self.layer_types):
            if index < first_shared:
                owners[kind] = index
            elif kind not in owners:
                raise Gemma4ConfigError(
                    f"YOCO layer {index} has no earlier {kind} KV owner"
                )
            sources.append(owners[kind])
        kv_layers = tuple(sorted(set(sources)))
        slots = {layer: slot for slot, layer in enumerate(kv_layers)}
        return replace(
            self,
            kv_slots=tuple(slots[layer] for layer in sources),
            kv_layer_indices=kv_layers,
            kv_share_sources=tuple(sources),
        )

    def validate(self) -> None:
        if self.hidden_activation != "gelu_pytorch_tanh":
            raise Gemma4ConfigError(
                "Gemma-4 requires the gelu_pytorch_tanh activation"
            )
        if self.num_attention_heads % max(self.layer_num_key_value_heads):
            raise Gemma4ConfigError("GQA head counts are not divisible")
        for kind, dim in zip(self.layer_types, self.layer_head_dims):
            if dim <= 0 or dim % 2:
                raise Gemma4ConfigError("Gemma-4 head dimensions must be even")
            rotary_dim = int(dim * self.rope_partial_factor[kind])
            if rotary_dim % 2:
                raise Gemma4ConfigError("Gemma-4 rotary dimension must be even")
        if self.attention_bias:
            raise Gemma4ConfigError(
                "Gemma-4 attention bias requires an explicit implementation"
            )
        if self.vocab_size_per_layer_input and not self.hidden_size_per_layer_input:
            raise Gemma4ConfigError("per-layer embeddings are incomplete")
        if self.hidden_size_per_layer_input and not self.vocab_size_per_layer_input:
            raise Gemma4ConfigError("per-layer embeddings are incomplete")
        if len(self.kv_share_sources) != self.num_hidden_layers:
            raise Gemma4ConfigError("KV sharing layout is incomplete")
        if len(self.kv_slots) != self.num_hidden_layers:
            raise Gemma4ConfigError("KV slot layout is incomplete")
        if max(self.kv_layer_indices) >= self.num_hidden_layers:
            raise Gemma4ConfigError("KV layer index is out of range")
        if self.final_logit_softcapping is not None and self.final_logit_softcapping <= 0:
            raise Gemma4ConfigError("final_logit_softcapping must be positive")
        if self.attn_logit_softcapping is not None and self.attn_logit_softcapping <= 0:
            raise Gemma4ConfigError("attn_logit_softcapping must be positive")

    @property
    def max_head_dim(self) -> int:
        return max(self.layer_head_dims)

    @property
    def max_kv_heads(self) -> int:
        return max(self.layer_num_key_value_heads)

    @property
    def uses_ple(self) -> bool:
        return self.hidden_size_per_layer_input > 0

    @property
    def num_kv_cache_layers(self) -> int:
        return len(self.kv_layer_indices)

    def is_full(self, layer_index: int) -> bool:
        return self.layer_types[layer_index] == FULL_ATTENTION

    def kv_layer(self, layer_index: int) -> int:
        return self.kv_share_sources[layer_index]

    def kv_slot(self, layer_index: int) -> int:
        return self.kv_slots[layer_index]

    def mlp_intermediate(self, layer_index: int) -> int:
        first_shared = self.num_hidden_layers - self.num_kv_shared_layers
        if self.use_double_wide_mlp and layer_index >= first_shared > 0:
            return self.intermediate_size * 2
        return self.intermediate_size


@dataclass(frozen=True, slots=True)
class Gemma4AssistantConfig:
    """Assistant (MTP) checkpoint config."""

    text_config: Gemma4TextConfig
    backbone_hidden_size: int
    num_centroids: int = 0
    centroid_intermediate_top_k: int = 0
    use_ordered_embeddings: bool = False
    tie_word_embeddings: bool = True

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> Gemma4AssistantConfig:
        text_payload = dict(payload.get("text_config") or {})
        if not text_payload:
            raise Gemma4ConfigError("assistant config requires text_config")
        # The assistant checkpoint marks every layer as KV-shared.  The native
        # predictor reads K/V from the target cache explicitly, so resolve its
        # geometry with independent KV slots and no per-layer inputs.
        text_payload["num_kv_shared_layers"] = 0
        text_payload["hidden_size_per_layer_input"] = 0
        text_payload["vocab_size_per_layer_input"] = 0
        text = Gemma4TextConfig.from_dict(text_payload)
        backbone = int(payload.get("backbone_hidden_size") or text.hidden_size)
        use_ordered = bool(payload.get("use_ordered_embeddings", False))
        centroids = int(payload.get("num_centroids") or 0)
        top_k = int(payload.get("centroid_intermediate_top_k") or 0)
        if use_ordered and min(centroids, top_k) <= 0:
            raise Gemma4ConfigError("ordered embedding geometry is incomplete")
        return cls(
            text_config=text,
            backbone_hidden_size=backbone,
            num_centroids=centroids,
            centroid_intermediate_top_k=top_k,
            use_ordered_embeddings=use_ordered,
            tie_word_embeddings=bool(payload.get("tie_word_embeddings", True)),
        )


@dataclass(frozen=True, slots=True)
class Gemma4Config:
    """Top-level config: dereferences the nested text configuration."""

    text: Gemma4TextConfig
    vision: Gemma4VisionConfig | None = None
    audio: Gemma4AudioConfig | None = None
    boi_token_id: int = 255_999
    eoi_token_id: int = 258_882
    image_token_id: int = 258_880
    video_token_id: int = 258_884
    boa_token_id: int = 256_000
    eoa_token_id: int = 258_883
    audio_token_id: int = 258_881

    @classmethod
    def from_path(cls, model_path: str | Path) -> Gemma4Config:
        path = Path(model_path)
        config_path = path / "config.json" if path.is_dir() else path
        payload = json.loads(config_path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise Gemma4ConfigError("model config must be an object")
        # Feature-extractor geometry lives in processor_config.json, not in
        # config.json. Merge it so the media frontend and the tower share one
        # source of truth.
        processor_path = path / "processor_config.json"
        if processor_path.is_file():
            try:
                processor = json.loads(processor_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                processor = {}
            if isinstance(processor, dict):
                extractor = processor.get("feature_extractor")
                if isinstance(extractor, dict):
                    audio = payload.get("audio_config")
                    if isinstance(audio, dict):
                        for key in (
                            "sampling_rate",
                            "frame_length",
                            "hop_length",
                            "fft_length",
                            "min_frequency",
                            "max_frequency",
                            "mel_floor",
                            "feature_size",
                        ):
                            if key in extractor:
                                target = "num_mel_bins" if key == "feature_size" else key
                                audio.setdefault(target, extractor[key])
                if "audio_seq_length" in processor:
                    audio = payload.get("audio_config")
                    if isinstance(audio, dict):
                        audio.setdefault(
                            "audio_seq_length", processor["audio_seq_length"]
                        )
                image = processor.get("image_processor")
                vision = payload.get("vision_config")
                if isinstance(image, dict) and isinstance(vision, dict):
                    if "max_soft_tokens" in image:
                        vision.setdefault(
                            "default_output_length", image["max_soft_tokens"]
                        )
        return cls.from_dict(payload)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> Gemma4Config:
        model_type = str(payload.get("model_type") or "")
        if model_type in {"gemma4_text", "gemma4_unified_text"}:
            text_payload = payload
        elif model_type in {"gemma4", "gemma4_unified", "gemma4_assistant"}:
            text_payload = payload.get("text_config")
            if not isinstance(text_payload, dict):
                raise Gemma4ConfigError("nested text_config is required")
        else:
            raise Gemma4ConfigError(f"unsupported Gemma-4 model_type {model_type!r}")
        text = Gemma4TextConfig.from_dict(text_payload)
        vision_payload = payload.get("vision_config")
        audio_payload = payload.get("audio_config")
        return cls(
            text=text,
            vision=(
                Gemma4VisionConfig.from_dict(vision_payload)
                if isinstance(vision_payload, dict)
                else None
            ),
            audio=(
                Gemma4AudioConfig.from_dict(audio_payload)
                if isinstance(audio_payload, dict)
                else None
            ),
            boi_token_id=int(payload.get("boi_token_id", 255_999)),
            eoi_token_id=int(payload.get("eoi_token_id", 258_882)),
            image_token_id=int(payload.get("image_token_id", 258_880)),
            video_token_id=int(payload.get("video_token_id", 258_884)),
            boa_token_id=int(payload.get("boa_token_id", 256_000)),
            eoa_token_id=int(
                payload.get("eoa_token_id", payload.get("eoa_token_index", 258_883))
            ),
            audio_token_id=int(payload.get("audio_token_id", 258_881)),
        )

    @property
    def is_unified(self) -> bool:
        return self.text.model_type == "gemma4_unified_text"


def _layer_types(payload: dict[str, Any], num_layers: int) -> tuple[str, ...]:
    declared = payload.get("layer_types")
    if declared is None:
        pattern = 6  # HF default: five sliding layers per full layer
        kinds = tuple(
            SLIDING_ATTENTION if (index + 1) % pattern else FULL_ATTENTION
            for index in range(num_layers)
        )
    elif isinstance(declared, (list, tuple)) and len(declared) == num_layers:
        kinds = tuple(str(value) for value in declared)
    else:
        raise Gemma4ConfigError("layer_types length must match num_hidden_layers")
    if any(kind not in _LAYER_TYPES for kind in kinds):
        raise Gemma4ConfigError("unsupported Gemma-4 layer type")
    if kinds[-1] != FULL_ATTENTION:
        kinds = (*kinds[:-1], FULL_ATTENTION)
    return kinds


def _positive(payload: dict[str, Any], key: str, *, default: int | None = None) -> int:
    value = payload.get(key, default)
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise Gemma4ConfigError(f"config requires integer {key!r}") from exc
    if parsed <= 0:
        raise Gemma4ConfigError(f"config {key!r} must be positive")
    return parsed


__all__ = [
    "FULL_ATTENTION",
    "SLIDING_ATTENTION",
    "Gemma4AssistantConfig",
    "Gemma4AudioConfig",
    "Gemma4Config",
    "Gemma4ConfigError",
    "Gemma4TextConfig",
    "Gemma4VisionConfig",
]
