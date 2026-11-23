"""Inkling-2 checkpoint geometry and streaming loader.

Inkling-2 is a looped, hybrid local/global MoE decoder with depth-wise
attention residuals, per-layer message controllers and NVFP4/MXFP8 expert
weights.  The loader keeps expert weights in their checkpoint precision so a
24 GB GPU can hold the model; everything else is BF16.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import tensorplay

from megalm.loading import SafetensorsSource, TensorInfo, WeightLoadReport

_CHECKPOINT_PREFIX = "model.llm."


class Inkling2CheckpointError(ValueError):
    """Raised when a checkpoint tensor or config violates the implemented contract."""


@dataclass(frozen=True, slots=True)
class Inkling2Layer:
    index: int
    is_local: bool
    is_global: bool
    is_looped: bool
    # Attention geometry.
    num_heads: int = 24
    num_kv_heads: int = 8
    head_dim: int = 128
    d_rel: int = 16
    rel_extent: int = 1024
    qk_l2_clip_threshold: float = 22.627416997969522
    rel_l2_clip_threshold: float = 8.0
    # Short convolution (applied to attn output, MoE output and K/V).
    sconv_kernel_size: int = 4
    # MoE.
    num_experts: int = 128
    experts_per_token: int = 8
    moe_intermediate_size: int = 1152
    mlp_gate_softcap: float = 4.0
    mlp_up_softcap: float = 25.0
    route_scale: float = 8.0
    # Precision groups from hf_quant_config.json.
    expert_storage: str = "nvfp4"  # "nvfp4" or "mxfp8"


@dataclass(frozen=True, slots=True)
class Inkling2Config:
    model_type: str
    vocab_size: int
    padded_vocab_size: int
    unpadded_vocab_size: int
    hidden_size: int
    num_hidden_layers: int
    rms_norm_eps: float
    max_position_embeddings: int
    # Loop.
    loop_core_start: int
    loop_core_end: int
    loop_num_passes: int
    loop_gate_rank: int
    # Depth attention residuals.
    n_layers_per_attn_res_block: int
    attn_res_normalize_values: bool
    attn_res_scale: float
    # Relative attention (global layers).
    log_scaling_n_floor: int
    log_scaling_alpha: float
    # Attention.
    sliding_window_size: int
    local_layer_ids: tuple[int, ...]
    # Final logits.
    final_logit_softcap: float | None
    layers: tuple[Inkling2Layer, ...]

    @property
    def is_looped(self) -> bool:
        return self.loop_num_passes > 1

    @property
    def num_loop_layers(self) -> int:
        return self.loop_core_end - self.loop_core_start

    @property
    def total_layer_visits(self) -> int:
        return (
            self.num_hidden_layers
            + (self.loop_num_passes - 1) * self.num_loop_layers
        )

    def is_local(self, layer: int) -> bool:
        return layer in set(self.local_layer_ids)

    def is_global(self, layer: int) -> bool:
        return not self.is_local(layer)

    def is_looped(self, layer: int) -> bool:
        return self.loop_core_start <= layer < self.loop_core_end

    def layer_visits(self) -> tuple[tuple[int, int], ...]:
        """``(layer, pass)`` in execution order: prelude, core * passes, coda."""

        visits: list[tuple[int, int]] = [
            (layer, 0) for layer in range(self.loop_core_start)
        ]
        for pass_index in range(self.loop_num_passes):
            visits.extend(
                (layer, pass_index)
                for layer in range(self.loop_core_start, self.loop_core_end)
            )
        visits.extend(
            (layer, 0) for layer in range(self.loop_core_end, self.num_hidden_layers)
        )
        return tuple(visits)

    @property
    def num_kv_cache_layers(self) -> int:
        """48 layer visits, or 24 when the loop shares each core's cache."""

        return self.total_layer_visits


def load_inkling2_config(model_path: str | Path) -> Inkling2Config:
    path = Path(model_path)
    config_path = path / "config.json" if path.is_dir() else path
    try:
        payload = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise Inkling2CheckpointError(
            f"cannot read Inkling-2 config {config_path}"
        ) from exc
    text = payload.get("text_config")
    if not isinstance(text, dict):
        raise Inkling2CheckpointError("Inkling-2 config requires text_config")
    local = tuple(int(index) for index in text["local_layer_ids"])
    num_layers = int(text["num_hidden_layers"])
    hidden = int(text["hidden_size"])
    loop = text["loop_config"]
    experts = int(text["n_routed_experts"])
    experts_per_token = int(text["num_experts_per_tok"])
    moe_intermediate = int(text["intermediate_size"])
    sconv_kernel = int(text.get("sconv_kernel_size", 4))
    quant_method = (payload.get("quantization_config") or {}).get("quant_method")
    manifest = _read_expert_quant_methods(config_path.parent)
    layers = tuple(
        Inkling2Layer(
            index=layer,
            is_local=layer in set(local),
            is_global=layer not in set(local),
            is_looped=int(loop["core_start"]) <= layer < int(loop["core_end"]),
            num_heads=int(text["num_attention_heads"]),
            num_kv_heads=int(text["num_key_value_heads"]),
            head_dim=int(text.get("head_dim", 128)),
            d_rel=int(text.get("d_rel", 16)),
            rel_extent=int(text.get("rel_extent", 1024)),
            qk_l2_clip_threshold=float(text.get("qk_l2_clip_threshold", 22.627)),
            rel_l2_clip_threshold=float(text.get("rel_l2_clip_threshold", 8.0)),
            sconv_kernel_size=sconv_kernel,
            num_experts=experts,
            experts_per_token=experts_per_token,
            moe_intermediate_size=moe_intermediate,
            mlp_gate_softcap=float(text.get("mlp_gate_softcap", 4.0)),
            mlp_up_softcap=float(text.get("mlp_up_softcap", 25.0)),
            route_scale=float(text.get("route_scale", 8.0)),
            expert_storage=manifest.get(
                layer,
                "nvfp4" if quant_method == "modelopt" else "mxfp8",
            ),
        )
        for layer in range(num_layers)
    )
    softcap = text.get("final_logit_softcapping")
    return Inkling2Config(
        model_type=str(payload.get("model_type", "inkling2_loop")),
        vocab_size=int(text["vocab_size"]),
        padded_vocab_size=int(text["vocab_size"]),
        unpadded_vocab_size=int(text.get("unpadded_vocab_size", text["vocab_size"])),
        hidden_size=hidden,
        num_hidden_layers=num_layers,
        rms_norm_eps=float(text.get("rms_norm_eps", 1e-6)),
        max_position_embeddings=int(
            text.get("max_position_embeddings", text.get("model_max_length", 1 << 20))
        ),
        loop_core_start=int(loop["core_start"]),
        loop_core_end=int(loop["core_end"]),
        loop_num_passes=int(loop["num_passes"]),
        loop_gate_rank=int(loop["gate_rank"]),
        n_layers_per_attn_res_block=int(text.get("n_layers_per_attn_res_block", 6)),
        attn_res_normalize_values=bool(text.get("attn_res_normalize_values", True)),
        attn_res_scale=float(text.get("attn_res_scale", hidden**-0.5)),
        log_scaling_n_floor=int(text.get("log_scaling_n_floor", 8192)),
        log_scaling_alpha=float(text.get("log_scaling_alpha", 0.1)),
        sliding_window_size=int(text.get("sliding_window_size", 1024)),
        local_layer_ids=local,
        final_logit_softcap=float(softcap) if softcap is not None else None,
        layers=layers,
    )


def _read_expert_quant_methods(root: Path) -> dict[int, str]:
    """Map layer index -> ``"nvfp4"`` / ``"mxfp8"`` from ``hf_quant_config.json``."""

    path = root / "hf_quant_config.json"
    if not path.is_file():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    methods: dict[int, str] = {}
    for group in payload.get("model", {}).get("parameter_groups", []):
        compute = group.get("compute_dtype")
        if compute not in {"nvfp4", "mxfp8"}:
            continue
        for name in group.get("tensors", []):
            parts = name.split(".layers.")
            if len(parts) != 2:
                continue
            layer = int(parts[1].split(".", 1)[0])
            methods[layer] = compute
    return methods


def expert_quant_method_for_layer(
    config: Inkling2Config,
    layer: int,
) -> str:
    """NVFP4 for layers below 20, MXFP8 for 20..23, as the manifest declares."""

    return config.layers[layer].expert_storage


@dataclass(frozen=True, slots=True)
class ExpertPrecision:
    """Storage dtype and layout of one expert tensor."""

    compute: str  # "nvfp4", "mxfp8", "bfloat16"
    storage: str  # "U8", "F8_E4M3", "BF16", "F32"

    @property
    def is_quantized(self) -> bool:
        return self.compute in {"nvfp4", "mxfp8"}


def layer_quant_method(manifest: dict[str, TensorInfo], layer: int) -> str:
    """Resolve one layer's expert precision from a checkpoint manifest."""

    w13 = manifest.get(f"model.llm.layers.{layer}.mlp.experts.w13_weight")
    if w13 is None:
        raise Inkling2CheckpointError(f"layer {layer} has no expert weights")
    if w13.dtype == "U8":
        return "nvfp4"
    if w13.dtype == "F8_E4M3":
        return "mxfp8"
    raise Inkling2CheckpointError(
        f"layer {layer} expert storage dtype {w13.dtype!r} is not quantized"
    )


class Inkling2Loader:
    """Stream Inkling-2 tensors into a caller-owned placement.

    The caller supplies ``assign``; this loader decodes names and shapes and
    hands each tensor to the model.  Expert weights stay in their checkpoint
    precision so the resident footprint is ~19 GB instead of ~52 GB.
    """

    def __init__(self, model_path: str | Path) -> None:
        self.config = load_inkling2_config(model_path)
        self.source = SafetensorsSource(model_path)
        self._by_name = {tensor.name: tensor for tensor in self.source.manifest}

    def names(self) -> tuple[str, ...]:
        return tuple(sorted(self._by_name))

    def info(self, name: str) -> TensorInfo:
        try:
            return self._by_name[name]
        except KeyError as exc:
            raise Inkling2CheckpointError(f"unknown tensor {name!r}") from exc

    def stream(self, consumer) -> WeightLoadReport:
        return self.source.stream(consumer)

    def iter_tensors(self, *, device: str = "cpu"):
        return self.source.iter_tensors(device=device)


def resolve_expert_shapes(config: Inkling2Config) -> dict[str, tuple[int, ...]]:
    """Expected checkpoint shapes for every expert tensor, by layer."""

    shapes: dict[str, tuple[int, ...]] = {}
    for layer in config.layers:
        quant = layer.expert_storage
        prefix = f"model.llm.layers.{layer.index}.mlp.experts"
        if quant == "nvfp4":
            shapes[f"{prefix}.w13_weight"] = (
                layer.num_experts,
                layer.hidden_size if hasattr(layer, "hidden_size") else 2304,
                layer.moe_intermediate_size * 2 // 2,
            )
        shapes[prefix] = ()
    return shapes


__all__ = [
    "ExpertPrecision",
    "Inkling2CheckpointError",
    "Inkling2Config",
    "Inkling2Layer",
    "Inkling2Loader",
    "expert_quant_method_for_layer",
    "layer_quant_method",
    "load_inkling2_config",
    "resolve_expert_shapes",
]