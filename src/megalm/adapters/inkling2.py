from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from megalm.contracts.model import (
    AttentionKind,
    AttentionResidualSpec,
    FeedForwardKind,
    LayerSpec,
    LoopSpec,
    ModelContractError,
    ModelSpec,
    ParallelConstraints,
    PositionKind,
    PositionSchema,
    QuantizationSpec,
    RelativeAttentionSpec,
    ShortConvolutionSpec,
    StateComponentSpec,
    StateKind,
    VisionSpec,
    WeightMappingRule,
)

_MODEL_TYPES = frozenset({"inkling2_loop", "inkling2"})


def load_inkling2_model_spec(model_path: str | Path) -> ModelSpec:
    """Translate an Inkling-2 ``config.json`` into scheduler-free model facts."""

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
    nvfp4_tensors, mxfp8_tensors = _count_quantized_tensors(config_path.parent)
    return inkling2_model_spec_from_config(
        config,
        nvfp4_tensors=nvfp4_tensors,
        mxfp8_tensors=mxfp8_tensors,
    )


def inkling2_model_spec_from_config(
    config: dict[str, Any],
    *,
    nvfp4_tensors: int = 0,
    mxfp8_tensors: int = 0,
) -> ModelSpec:
    model_type = _required_str(config, "model_type")
    if model_type not in _MODEL_TYPES:
        raise ModelContractError(
            f"Inkling2 adapter does not support model_type={model_type!r}"
        )

    text = _required_object(config, "text_config")
    text_model_type = str(text.get("model_type", ""))
    if not text_model_type.startswith("inkling2"):
        raise ModelContractError(
            f"Inkling2 requires an inkling2 text_config, got {text_model_type!r}"
        )

    hidden_size = _required_int(text, "hidden_size")
    num_layers = _required_int(text, "num_hidden_layers")
    attention_heads = _required_int(text, "num_attention_heads")
    kv_heads = _required_int(text, "num_key_value_heads")
    head_dim = _required_int(text, "head_dim")
    vocab_size = _required_int(text, "vocab_size")
    max_positions = _first_int(
        text, ("max_position_embeddings", "model_max_length")
    )
    if attention_heads % kv_heads:
        raise ModelContractError("Inkling2 query heads must be divisible by KV heads")
    # Inkling's hidden size is independent of heads*head_dim: the qkvr projection
    # emits q (heads*head_dim) + k + v + relative + gate + sinks from hidden_size.

    num_experts = _required_int(text, "n_routed_experts")
    experts_per_token = _required_int(text, "num_experts_per_tok")
    moe_intermediate = _required_int(text, "intermediate_size")
    if experts_per_token > num_experts:
        raise ModelContractError("Inkling2 experts_per_token exceeds the expert count")

    local_ids = text.get("local_layer_ids")
    if not isinstance(local_ids, list) or not all(
        isinstance(index, int) for index in local_ids
    ):
        raise ModelContractError("Inkling2 requires an integer local_layer_ids list")
    if len(set(local_ids)) != len(local_ids) or any(
        index < 0 or index >= num_layers for index in local_ids
    ):
        raise ModelContractError(
            "Inkling2 local_layer_ids must be unique valid layer indices"
        )
    local_layer_ids = frozenset(local_ids)
    sliding_window = _required_int(text, "sliding_window_size")

    layers = tuple(
        LayerSpec(
            index=index,
            attention=AttentionKind.FULL,
            feed_forward=FeedForwardKind.MOE,
            hidden_size=hidden_size,
            intermediate_size=moe_intermediate,
            num_attention_heads=attention_heads,
            num_key_value_heads=kv_heads,
            head_dim=head_dim,
            num_experts=num_experts,
            experts_per_token=experts_per_token,
            sliding_window=sliding_window if index in local_layer_ids else None,
        )
        for index in range(num_layers)
    )

    loop_config = _required_object(text, "loop_config")
    loop = LoopSpec(
        core_start=_required_int(loop_config, "core_start"),
        core_end=_required_int(loop_config, "core_end"),
        num_passes=_required_int(loop_config, "num_passes"),
        gate_rank=_required_int(loop_config, "gate_rank"),
        is_varloop=bool(loop_config.get("is_varloop", False)),
    )
    if loop.core_end > num_layers:
        raise ModelContractError("Inkling2 loop core_end exceeds the layer count")

    attention_residual = AttentionResidualSpec(
        n_layers_per_block=_required_int(text, "n_layers_per_attn_res_block"),
        normalize_values=bool(text.get("attn_res_normalize_values", True)),
    )

    relative_attention = RelativeAttentionSpec(
        d_rel=_required_int(text, "d_rel"),
        rel_extent=_required_int(text, "rel_extent"),
        log_scaling_n_floor=_optional_int(text, "log_scaling_n_floor"),
        log_scaling_alpha=float(text.get("log_scaling_alpha", 0.1)),
        qk_l2_clip_threshold=float(text.get("qk_l2_clip_threshold", 0.0)),
        rel_l2_clip_threshold=float(text.get("rel_l2_clip_threshold", 0.0)),
        q_bias=bool(text.get("q_bias", False)),
        o_bias=bool(text.get("o_bias", False)),
        use_value_norm_no_gamma=bool(text.get("use_value_norm_no_gamma", False)),
    )

    short_convolution = ShortConvolutionSpec(
        kernel_size=_required_int(text, "sconv_kernel_size")
    )

    vision_spec = _vision_spec_from_config(config.get("vision_config"), hidden_size)

    state_schema: list[StateComponentSpec] = [
        StateComponentSpec(StateKind.PAGED_KV, tuple(range(num_layers))),
        StateComponentSpec(StateKind.PREFIX_CHECKPOINT, tuple(range(num_layers))),
    ]
    if vision_spec is not None:
        state_schema.append(StateComponentSpec(StateKind.VISION_EMBEDDING))
        state_schema.append(StateComponentSpec(StateKind.MEDIA_POSITION))

    tp_facts = [
        hidden_size,
        attention_heads,
        kv_heads,
        head_dim,
        vocab_size,
        num_experts,
        moe_intermediate,
    ]
    supported_tp = tuple(
        degree
        for degree in (1, 2, 4, 8)
        if all(value % degree == 0 for value in tp_facts)
    )

    architectures = config.get("architectures", ())
    architecture = (
        str(architectures[0])
        if isinstance(architectures, list) and architectures
        else "Inkling2LoopForConditionalGeneration"
    )

    return ModelSpec(
        architecture=architecture,
        model_type=model_type,
        vocab_size=vocab_size,
        hidden_size=hidden_size,
        layers=layers,
        position=PositionSchema(
            kind=PositionKind.LEARNED_RELATIVE,
            max_positions=max_positions,
            rope_theta=1.0,  # unused: Inkling-2 uses learned relative bias, not RoPE
            head_dim=head_dim,
        ),
        state_schema=tuple(state_schema),
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
        supported_dtypes=("bfloat16",),
        tie_word_embeddings=False,
        vision=vision_spec,
        loop=loop,
        attention_residual=attention_residual,
        relative_attention=relative_attention,
        short_convolution=short_convolution,
        final_logit_softcapping=_optional_float(text, "final_logit_softcapping"),
        quantization=_quantization_spec(
            config.get("quantization_config"),
            nvfp4_tensors=nvfp4_tensors,
            mxfp8_tensors=mxfp8_tensors,
        ),
    )


def _vision_spec_from_config(
    vision: Any,
    text_hidden_size: int,
) -> VisionSpec | None:
    if not isinstance(vision, dict) or not vision:
        return None
    hidden = _required_int(vision, "hidden_size")
    grid = _required_int(vision, "posemb_grid_size")
    mlp_ratio = float(vision.get("mlp_ratio", 4.0))
    return VisionSpec(
        processor_type="inkling2_vision",
        patch_size=_required_int(vision, "native_patch_size"),
        merge_size=_required_int(vision, "adapter_merge_size"),
        temporal_patch_size=_required_int(vision, "temporal_patch_size"),
        in_channels=int(vision.get("in_channels", 3)),
        depth=_required_int(vision, "num_hidden_layers"),
        hidden_size=hidden,
        intermediate_size=int(hidden * mlp_ratio),
        num_heads=_required_int(vision, "num_attention_heads"),
        output_hidden_size=_required_int(vision, "text_hidden_size"),
        num_position_embeddings=grid * grid,
        hidden_act=str(vision.get("hidden_act", "gelu_pytorch_tanh")),
    )


def _quantization_spec(
    quantization_config: Any,
    *,
    nvfp4_tensors: int,
    mxfp8_tensors: int,
) -> QuantizationSpec | None:
    if not isinstance(quantization_config, dict):
        return None
    method = str(quantization_config.get("quant_method", "")).strip()
    if not method:
        return None
    algo = quantization_config.get("quant_algo")
    return QuantizationSpec(
        method=method,
        algo=str(algo) if algo else None,
        config_file="hf_quant_config.json",
        nvfp4_tensors=nvfp4_tensors,
        mxfp8_tensors=mxfp8_tensors,
    )


def _count_quantized_tensors(root: Path) -> tuple[int, int]:
    """Count NVFP4 / MXFP8 tensors from the checkpoint precision manifest."""

    quant_path = root / "hf_quant_config.json"
    if not quant_path.is_file():
        return (0, 0)
    try:
        payload = json.loads(quant_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return (0, 0)
    nvfp4 = mxfp8 = 0
    if isinstance(payload, dict):
        sections = payload.values()
    else:
        sections = ()
    for section in sections:
        if not isinstance(section, dict):
            continue
        for group in section.get("parameter_groups", []):
            if not isinstance(group, dict):
                continue
            count = len(group.get("tensors", []))
            compute = group.get("compute_dtype")
            if compute == "nvfp4":
                nvfp4 += count
            elif compute == "mxfp8":
                mxfp8 += count
    return (nvfp4, mxfp8)


def _required_object(config: dict[str, Any], key: str) -> dict[str, Any]:
    value = config.get(key)
    if not isinstance(value, dict):
        raise ModelContractError(f"Inkling2 config requires object {key!r}")
    return value


def _required_str(config: dict[str, Any], key: str) -> str:
    value = config.get(key)
    if not isinstance(value, str) or not value:
        raise ModelContractError(f"Inkling2 config requires string {key!r}")
    return value


def _positive_int(value: Any, key: str, *, allow_zero: bool = False) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise ModelContractError(f"Inkling2 config requires integer {key!r}") from exc
    minimum = 0 if allow_zero else 1
    if parsed < minimum:
        qualifier = "non-negative" if allow_zero else "positive"
        raise ModelContractError(f"Inkling2 config {key!r} must be {qualifier}")
    return parsed


def _required_int(
    config: dict[str, Any],
    key: str,
    *,
    allow_zero: bool = False,
) -> int:
    if key not in config:
        raise ModelContractError(f"Inkling2 config requires integer {key!r}")
    return _positive_int(config[key], key, allow_zero=allow_zero)


def _optional_int(config: dict[str, Any], key: str) -> int | None:
    value = config.get(key)
    if value is None:
        return None
    return _positive_int(value, key)


def _optional_float(config: dict[str, Any], key: str) -> float | None:
    value = config.get(key)
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise ModelContractError(f"Inkling2 config {key!r} must be numeric") from exc


def _first_int(config: dict[str, Any], keys: tuple[str, ...]) -> int:
    for key in keys:
        if key in config:
            return _required_int(config, key)
    raise ModelContractError(
        f"Inkling2 config requires one of {keys!r}"
    )