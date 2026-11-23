from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum

from megalm.contracts._canonical import canonical_digest


class ModelContractError(ValueError):
    """Raised when model facts are incomplete or internally inconsistent."""


class AttentionKind(str, Enum):
    FULL = "full_attention"
    GATED_DELTA_NET = "gated_delta_net"


class FeedForwardKind(str, Enum):
    DENSE = "dense_ffn"
    MOE = "moe"


class PositionKind(str, Enum):
    ROPE = "rope"
    MROPE = "mrope"
    LEARNED_RELATIVE = "learned_relative"


class StateKind(str, Enum):
    PAGED_KV = "paged_kv"
    GDN_CONV = "gdn_conv"
    GDN_RECURRENT = "gdn_recurrent"
    PREFIX_CHECKPOINT = "prefix_checkpoint"
    VISION_EMBEDDING = "vision_embedding"
    MEDIA_POSITION = "media_position"


class WeightTransform(str, Enum):
    IDENTITY = "identity"
    TRANSPOSE = "transpose"
    PACK_QKV = "pack_qkv"
    PACK_GATE_UP = "pack_gate_up"


@dataclass(frozen=True, slots=True)
class GatedDeltaNetSpec:
    num_key_heads: int
    num_value_heads: int
    key_head_dim: int
    value_head_dim: int
    conv_kernel_dim: int
    state_dtype: str = "float32"

    def __post_init__(self) -> None:
        if min(
            self.num_key_heads,
            self.num_value_heads,
            self.key_head_dim,
            self.value_head_dim,
            self.conv_kernel_dim,
        ) <= 0:
            raise ModelContractError("Gated DeltaNet dimensions must be positive")
        if self.num_value_heads % self.num_key_heads:
            raise ModelContractError(
                "Gated DeltaNet value heads must be divisible by key heads"
            )
        if self.conv_kernel_dim < 2:
            raise ModelContractError(
                "Gated DeltaNet convolution kernel must retain state"
            )
        if self.state_dtype not in {"float32", "float16", "bfloat16"}:
            raise ModelContractError("unsupported Gated DeltaNet state dtype")


@dataclass(frozen=True, slots=True)
class LoopSpec:
    """Looped-transformer schedule: a contiguous core range is replayed N times.

    Layers outside ``[core_start, core_end)`` run once; the core layers run
    ``num_passes`` times with shared weights (e.g. Inkling-2 core 6..18 x3).
    """

    core_start: int
    core_end: int
    num_passes: int
    gate_rank: int
    is_varloop: bool = False

    def __post_init__(self) -> None:
        if self.core_start < 0 or self.core_end <= self.core_start:
            raise ModelContractError("loop core range must be non-empty and ordered")
        if self.num_passes < 1:
            raise ModelContractError("loop num_passes must be positive")
        if self.gate_rank <= 0:
            raise ModelContractError("loop gate_rank must be positive")


@dataclass(frozen=True, slots=True)
class AttentionResidualSpec:
    """Learned cross-layer residual mixing over blocks of decoder layers."""

    n_layers_per_block: int
    normalize_values: bool = True

    def __post_init__(self) -> None:
        if self.n_layers_per_block <= 0:
            raise ModelContractError("attention residual block size must be positive")


@dataclass(frozen=True, slots=True)
class ShortConvolutionSpec:
    """Depthwise short-convolution applied to attention and MLP outputs."""

    kernel_size: int

    def __post_init__(self) -> None:
        if self.kernel_size < 2:
            raise ModelContractError("short convolution must retain state")


@dataclass(frozen=True, slots=True)
class RelativeAttentionSpec:
    """Learned relative-bias attention with QK/relative L2 clipping and log scaling."""

    d_rel: int
    rel_extent: int
    log_scaling_n_floor: int | None = None
    log_scaling_alpha: float = 0.1
    qk_l2_clip_threshold: float = 0.0
    rel_l2_clip_threshold: float = 0.0
    q_bias: bool = False
    o_bias: bool = False
    use_value_norm_no_gamma: bool = False

    def __post_init__(self) -> None:
        if self.d_rel <= 0 or self.rel_extent <= 0:
            raise ModelContractError("relative attention dimensions must be positive")
        if self.log_scaling_n_floor is not None and self.log_scaling_n_floor <= 0:
            raise ModelContractError("log scaling floor must be positive")
        if self.qk_l2_clip_threshold < 0 or self.rel_l2_clip_threshold < 0:
            raise ModelContractError("L2 clip thresholds must be nonnegative")


@dataclass(frozen=True, slots=True)
class QuantizationSpec:
    """Checkpoint precision metadata (e.g. ModelOpt NVFP4 + MXFP8)."""

    method: str
    algo: str | None = None
    config_file: str = "hf_quant_config.json"
    nvfp4_tensors: int = 0
    mxfp8_tensors: int = 0

    def __post_init__(self) -> None:
        if not self.method:
            raise ModelContractError("quantization method must not be empty")
        if self.nvfp4_tensors < 0 or self.mxfp8_tensors < 0:
            raise ModelContractError("quantized tensor counts must be nonnegative")


@dataclass(frozen=True, slots=True)
class LayerSpec:
    index: int
    attention: AttentionKind
    feed_forward: FeedForwardKind
    hidden_size: int
    intermediate_size: int
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int
    num_experts: int = 0
    experts_per_token: int = 0
    gated_delta_net: GatedDeltaNetSpec | None = None
    sliding_window: int | None = None

    def __post_init__(self) -> None:
        positive = {
            "hidden_size": self.hidden_size,
            "intermediate_size": self.intermediate_size,
            "num_attention_heads": self.num_attention_heads,
            "num_key_value_heads": self.num_key_value_heads,
            "head_dim": self.head_dim,
        }
        if self.index < 0:
            raise ModelContractError("layer index must be non-negative")
        if self.sliding_window is not None and self.sliding_window <= 0:
            raise ModelContractError("sliding_window must be positive when set")
        for name, value in positive.items():
            if value <= 0:
                raise ModelContractError(f"{name} must be positive")
        if self.feed_forward is FeedForwardKind.MOE:
            if self.num_experts <= 0 or self.experts_per_token <= 0:
                raise ModelContractError("MoE layers require expert counts")
            if self.experts_per_token > self.num_experts:
                raise ModelContractError("experts_per_token exceeds num_experts")
        elif self.num_experts or self.experts_per_token:
            raise ModelContractError("dense layers cannot declare experts")
        if self.attention is AttentionKind.GATED_DELTA_NET:
            if self.gated_delta_net is None:
                raise ModelContractError(
                    "Gated DeltaNet layers require recurrent geometry"
                )
        elif self.gated_delta_net is not None:
            raise ModelContractError(
                "full-attention layers cannot declare Gated DeltaNet geometry"
            )


@dataclass(frozen=True, slots=True)
class PositionSchema:
    kind: PositionKind
    max_positions: int
    rope_theta: float
    head_dim: int
    mrope_sections: tuple[int, ...] = ()
    partial_rotary_factor: float = 1.0
    rotary_style: str = "neox"
    mrope_interleaved: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "mrope_sections", tuple(self.mrope_sections))
        if self.max_positions <= 0 or self.head_dim <= 0:
            raise ModelContractError("position dimensions must be positive")
        if (
            self.kind in (PositionKind.ROPE, PositionKind.MROPE)
            and self.rope_theta <= 0
        ):
            raise ModelContractError("rope_theta must be positive for RoPE positions")
        if not 0 < self.partial_rotary_factor <= 1:
            raise ModelContractError("partial_rotary_factor must be in (0, 1]")
        if self.rotary_style not in {"neox", "interleaved"}:
            raise ModelContractError("unsupported rotary_style")
        if self.kind is PositionKind.MROPE and not self.mrope_sections:
            raise ModelContractError("MRoPE requires section sizes")
        if self.kind is PositionKind.ROPE and self.mrope_sections:
            raise ModelContractError("RoPE cannot declare MRoPE sections")
        rotary_dim = self.head_dim * self.partial_rotary_factor
        if self.kind is PositionKind.MROPE and (
            not rotary_dim.is_integer()
            or int(rotary_dim) % 2
            or sum(self.mrope_sections) * 2 != int(rotary_dim)
        ):
            raise ModelContractError(
                "MRoPE sections must cover the partial rotary dimension"
            )
        if self.kind is PositionKind.ROPE and self.mrope_interleaved:
            raise ModelContractError("interleaving is only defined for MRoPE")


@dataclass(frozen=True, slots=True)
class StateComponentSpec:
    kind: StateKind
    layer_indices: tuple[int, ...] = ()
    checkpoint_alignment: int = 1

    def __post_init__(self) -> None:
        object.__setattr__(self, "layer_indices", tuple(self.layer_indices))
        if len(set(self.layer_indices)) != len(self.layer_indices):
            raise ModelContractError("state layer indices must be unique")
        if any(index < 0 for index in self.layer_indices):
            raise ModelContractError("state layer indices must be non-negative")
        if self.checkpoint_alignment <= 0:
            raise ModelContractError("checkpoint_alignment must be positive")


@dataclass(frozen=True, slots=True)
class WeightMappingRule:
    source_pattern: str
    target_template: str
    transform: WeightTransform = WeightTransform.IDENTITY

    def __post_init__(self) -> None:
        try:
            re.compile(self.source_pattern)
        except re.error as exc:
            raise ModelContractError(
                f"invalid weight mapping regex: {self.source_pattern!r}"
            ) from exc

    def map_name(self, source_name: str) -> str | None:
        match = re.fullmatch(self.source_pattern, source_name)
        if match is None:
            return None
        return match.expand(self.target_template)


@dataclass(frozen=True, slots=True)
class ParallelConstraints:
    supported_tp_degrees: tuple[int, ...] = (1,)
    allow_cross_node_tp: bool = False
    allow_pipeline_parallel: bool = False
    allow_expert_parallel: bool = False
    minimum_compute_capability: tuple[int, int] = (8, 0)

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "supported_tp_degrees",
            tuple(sorted(set(self.supported_tp_degrees))),
        )
        if not self.supported_tp_degrees or self.supported_tp_degrees[0] != 1:
            raise ModelContractError("TP=1 must always be supported")
        if any(degree <= 0 for degree in self.supported_tp_degrees):
            raise ModelContractError("TP degrees must be positive")
        if len(self.minimum_compute_capability) != 2:
            raise ModelContractError("compute capability must be a (major, minor) pair")


@dataclass(frozen=True, slots=True)
class VisionSpec:
    processor_type: str
    patch_size: int
    merge_size: int
    temporal_patch_size: int = 1
    in_channels: int = 3
    depth: int = 0
    hidden_size: int = 0
    intermediate_size: int = 0
    num_heads: int = 0
    output_hidden_size: int = 0
    num_position_embeddings: int = 0
    image_token_id: int = -1
    vision_start_token_id: int = -1
    vision_end_token_id: int = -1
    hidden_act: str = "gelu_pytorch_tanh"

    def __post_init__(self) -> None:
        if not self.processor_type:
            raise ModelContractError("vision processor_type must not be empty")
        if min(self.patch_size, self.merge_size, self.temporal_patch_size) <= 0:
            raise ModelContractError("vision dimensions must be positive")
        geometry = (
            self.depth,
            self.hidden_size,
            self.intermediate_size,
            self.num_heads,
            self.output_hidden_size,
            self.num_position_embeddings,
        )
        if any(geometry) and any(value <= 0 for value in geometry):
            raise ModelContractError("vision tower geometry must be complete")
        token_ids = (
            self.image_token_id,
            self.vision_start_token_id,
            self.vision_end_token_id,
        )
        if any(token_id >= 0 for token_id in token_ids) and any(
            token_id < 0 for token_id in token_ids
        ):
            raise ModelContractError("vision token ids must be declared together")


@dataclass(frozen=True, slots=True)
class ModelSpec:
    """Model facts consumed by planners; it contains no scheduling policy."""

    architecture: str
    model_type: str
    vocab_size: int
    hidden_size: int
    layers: tuple[LayerSpec, ...]
    position: PositionSchema
    state_schema: tuple[StateComponentSpec, ...]
    weight_mapping: tuple[WeightMappingRule, ...]
    parallel: ParallelConstraints
    supported_dtypes: tuple[str, ...] = ("float16", "bfloat16")
    tie_word_embeddings: bool = False
    vision: VisionSpec | None = None
    loop: LoopSpec | None = None
    attention_residual: AttentionResidualSpec | None = None
    relative_attention: RelativeAttentionSpec | None = None
    short_convolution: ShortConvolutionSpec | None = None
    final_logit_softcapping: float | None = None
    quantization: QuantizationSpec | None = None
    qk_norm: bool = False
    qkv_bias: bool = False
    embedding_multiplier: float = 1.0
    residual_multiplier: float = 1.0
    attention_multiplier: float = 0.0
    logit_scale: float = 1.0
    schema_version: int = 1

    def __post_init__(self) -> None:
        object.__setattr__(self, "layers", tuple(self.layers))
        object.__setattr__(self, "state_schema", tuple(self.state_schema))
        object.__setattr__(self, "weight_mapping", tuple(self.weight_mapping))
        object.__setattr__(self, "supported_dtypes", tuple(self.supported_dtypes))
        if self.schema_version != 1:
            raise ModelContractError("unsupported ModelSpec schema_version")
        if not self.architecture or not self.model_type:
            raise ModelContractError("architecture and model_type must not be empty")
        if self.vocab_size <= 0 or self.hidden_size <= 0:
            raise ModelContractError("model dimensions must be positive")
        if min(
            self.embedding_multiplier,
            self.residual_multiplier,
            abs(self.logit_scale),
        ) <= 0:
            raise ModelContractError("model scaling factors must be positive")
        if self.attention_multiplier < 0:
            raise ModelContractError("attention scale must be non-negative")
        if not self.layers:
            raise ModelContractError("ModelSpec requires at least one layer")
        if tuple(layer.index for layer in self.layers) != tuple(range(len(self.layers))):
            raise ModelContractError("layers must be contiguous and zero-indexed")
        valid_layer_indices = set(range(len(self.layers)))
        for component in self.state_schema:
            if not set(component.layer_indices).issubset(valid_layer_indices):
                raise ModelContractError(
                    f"{component.kind.value} references a missing layer"
                )
        if not self.weight_mapping:
            raise ModelContractError("ModelSpec requires weight mapping rules")
        if not self.supported_dtypes:
            raise ModelContractError("ModelSpec requires at least one dtype")
        layer_count = len(self.layers)
        if self.loop is not None and self.loop.core_end > layer_count:
            raise ModelContractError("loop core range exceeds the layer count")
        if (
            self.attention_residual is not None
            and self.attention_residual.n_layers_per_block > layer_count
        ):
            raise ModelContractError(
                "attention residual block exceeds the layer count"
            )

    @property
    def digest(self) -> str:
        return canonical_digest(self)

    @property
    def num_layers(self) -> int:
        return len(self.layers)

    def map_weight_name(self, source_name: str) -> tuple[str, WeightTransform]:
        for rule in self.weight_mapping:
            target = rule.map_name(source_name)
            if target is not None:
                return target, rule.transform
        raise ModelContractError(f"no weight mapping rule for {source_name!r}")
