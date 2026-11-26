"""Load PEFT LoRA adapters from a local safetensors checkpoint.

The loader follows vLLM's PEFT conventions (``refs/vllm/vllm/lora``): rank is
``alpha / r`` and MegaLM packs q/k/v and gate/up into the native fused
projections.  Every logical module path (including the layer index) is kept
separate, so a real multi-layer adapter is bound per layer instead of being
collapsed into one bank shared by all layers.
"""

from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import tensorplay

from megalm.contracts.lora import (
    LORA_TARGET_SLICES,
    LoRAAdapterSpec,
    LoRAContractError,
    LoRALayerWeights,
    parse_lora_tensor_name,
    resolve_lora_targets,
)
from megalm.loading.safetensors import SafetensorsSource

_LAYER_RE = re.compile(r"layers\.(\d+)\.")


@dataclass(frozen=True, slots=True)
class LoadedLoRAAdapter:
    spec: LoRAAdapterSpec
    weights: dict[str, LoRALayerWeights]
    """Backward-compatible flat view, populated only for single-layer adapters."""

    layers: dict[int, dict[str, LoRALayerWeights]]
    """Per-layer packed weights: ``layer index -> native target -> weights``."""

    source_tensors: int
    source_bytes: int
    skipped_modules: tuple[str, ...]

    @property
    def layer_count(self) -> int:
        return len(self.layers)

    @property
    def rank_footprint(self) -> int:
        return max(
            (weight.rank for targets in self.layers.values() for weight in targets.values()),
            default=0,
        )


def _target_slices(target: str) -> tuple[str, ...]:
    if target not in LORA_TARGET_SLICES:
        raise LoRAContractError(f"unsupported LoRA target {target!r}")
    if target == "qkv_proj":
        return ("q_proj", "k_proj", "v_proj")
    if target == "gate_up_proj":
        return ("gate_proj", "up_proj")
    return (target,)


def _pack_pairs(
    pairs: dict[str, dict[str, Any]],
    spec: LoRAAdapterSpec,
    targets: tuple[str, ...],
) -> dict[str, LoRALayerWeights]:
    """Pack the logical pairs of one module into native target banks."""
    weights: dict[str, LoRALayerWeights] = {}
    for target in targets:
        parts: list[tuple[Any, Any]] = []
        for logical in _target_slices(target):
            pair = pairs.get(logical)
            if pair is None or "lora_a" not in pair or "lora_b" not in pair:
                raise LoRAContractError(
                    f"adapter is missing the {logical!r} LoRA pair "
                    f"required by packed target {target!r}"
                )
            a, b = pair["lora_a"], pair["lora_b"]
            if a.ndim != 2 or b.ndim != 2:
                raise LoRAContractError(f"LoRA tensors for {logical!r} must be 2-D")
            if a.shape[0] != spec.r or b.shape[1] != spec.r:
                raise LoRAContractError(
                    f"LoRA rank for {logical!r} does not match adapter_config.json"
                )
            parts.append((a, b))

        in_features = tuple(parts[0][0].shape)[1]
        if any(tuple(a.shape)[1] != in_features for a, _ in parts):
            raise LoRAContractError(
                f"LoRA A input dimensions differ within packed target {target!r}"
            )
        out_sizes = [b.shape[0] for _, b in parts]
        packed_a = tensorplay.cat([a for a, _ in parts], dim=0)
        packed_b = tensorplay.zeros(
            (sum(out_sizes), spec.r * len(parts)),
            dtype=parts[0][1].dtype,
            device=parts[0][1].device,
        )
        row = 0
        for index, (_, b) in enumerate(parts):
            rows = slice(row, row + out_sizes[index])
            columns = slice(index * spec.r, (index + 1) * spec.r)
            packed_b[rows, columns] = b
            row += out_sizes[index]
        # vLLM folds alpha / rank (or alpha / sqrt(rank)) into B at load time.
        packed_b = packed_b * spec.scaling
        weights[target] = LoRALayerWeights(
            module_name=target,
            lora_a=packed_a.contiguous(),
            lora_b=packed_b.contiguous(),
            rank=spec.r * len(parts),
        )
    return weights


def load_lora_adapter(
    path: str | Path,
    *,
    max_lora_rank: int,
    dtype: Any = None,
    device: Any = None,
) -> LoadedLoRAAdapter:
    """Read one PEFT adapter and pack it per layer."""
    root = Path(path).expanduser()
    source = SafetensorsSource(root / "adapter_model.safetensors")
    spec = LoRAAdapterSpec.from_path(root, max_lora_rank=max_lora_rank)
    targets = resolve_lora_targets(spec)
    if not targets:
        raise LoRAContractError(
            "adapter targets no supported module; expected one of "
            "q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj"
        )

    # Full module path -> logical leaf -> A/B pair. Keeping the path prevents
    # layers from overwriting each other's tensors.
    grouped: dict[int, dict[str, dict[str, Any]]] = defaultdict(
        lambda: defaultdict(dict)
    )
    skipped: set[str] = set()
    source_tensors = 0
    no_layer: dict[str, dict[str, Any]] = defaultdict(dict)
    for info, tensor in source.iter_tensors(device="cpu"):
        source_tensors += 1
        try:
            module_name, is_a = parse_lora_tensor_name(info.name)
        except LoRAContractError:
            skipped.add(info.name)
            continue
        logical = module_name.rsplit(".", 1)[-1]
        match = _LAYER_RE.search(module_name)
        slot = int(match.group(1)) if match else -1
        bucket = grouped[slot] if slot >= 0 else no_layer
        key = "lora_a" if is_a else "lora_b"
        if key in bucket[logical]:
            raise LoRAContractError(
                f"duplicate {key} tensor for LoRA module {module_name!r}"
            )
        bucket[logical][key] = tensor

    if no_layer:
        if grouped:
            raise LoRAContractError("adapter mixes layered and unlayered LoRA modules")
        grouped[0] = no_layer

    layers: dict[int, dict[str, LoRALayerWeights]] = {}
    for index in sorted(grouped):
        packed = _pack_pairs(grouped[index], spec, targets)
        for target, weight in packed.items():
            if dtype is not None:
                weight = LoRALayerWeights(
                    module_name=weight.module_name,
                    lora_a=weight.lora_a.to(dtype=dtype),
                    lora_b=weight.lora_b.to(dtype=dtype),
                    rank=weight.rank,
                )
            if device is not None:
                weight = LoRALayerWeights(
                    module_name=weight.module_name,
                    lora_a=weight.lora_a.to(device=device),
                    lora_b=weight.lora_b.to(device=device),
                    rank=weight.rank,
                )
            packed[target] = weight
        layers[index] = packed

    used = {name for target in targets for name in _target_slices(target)}
    for logical in list(grouped.values()):
        for name in logical:
            if name not in used:
                skipped.add(name)
    flat = layers[0] if len(layers) == 1 else {}
    return LoadedLoRAAdapter(
        spec=spec,
        weights=flat,
        layers=layers,
        source_tensors=source_tensors,
        source_bytes=source.total_bytes,
        skipped_modules=tuple(sorted(skipped)),
    )


__all__ = ["LoadedLoRAAdapter", "load_lora_adapter"]
