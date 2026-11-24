"""Load PEFT LoRA adapters from a local safetensors checkpoint.

The loader intentionally mirrors vLLM's PEFT conventions (``refs/vllm/vllm/lora``):
``adapter_config.json`` supplies the rank, alpha, rsLoRA flag and target
modules, while ``adapter_model.safetensors`` supplies ``lora_A`` / ``lora_B``
pairs.  The adapter scale is folded into ``lora_B`` once, so the hot path only
computes ``y += (x @ A.T) @ B.T``.

Packed native projections need one extra step.  MegaLM's generic causal LM
executes q/k/v as one ``qkv_proj`` and gate/up as one ``gate_up_proj``.  For
those targets the loader stacks the per-logical-module A matrices vertically
and places the B matrices on a block diagonal.  A missing slice stays zero,
which makes the additive delta correct without any per-token branch.
"""

from __future__ import annotations

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


@dataclass(frozen=True, slots=True)
class LoadedLoRAAdapter:
    spec: LoRAAdapterSpec
    weights: dict[str, LoRALayerWeights]
    source_tensors: int
    source_bytes: int
    skipped_modules: tuple[str, ...]

    @property
    def rank_footprint(self) -> int:
        return max(
            (weight.rank for weight in self.weights.values()),
            default=0,
        )


def load_lora_adapter(
    path: str | Path,
    *,
    max_lora_rank: int,
    dtype: Any = None,
    device: Any = None,
) -> LoadedLoRAAdapter:
    """Read and pack one PEFT adapter.

    ``dtype``/``device`` are passed to ``Tensor.to``; when omitted the tensors
    stay on CPU in their checkpoint dtype (used by the CPU registration path).
    """

    root = Path(path).expanduser()
    source = SafetensorsSource(root / "adapter_model.safetensors")
    spec = LoRAAdapterSpec.from_path(root, max_lora_rank=max_lora_rank)
    targets = resolve_lora_targets(spec)
    if not targets:
        raise LoRAContractError(
            "adapter targets no supported module; expected one of "
            "q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj"
        )

    pairs: dict[str, dict[str, Any]] = {}
    skipped: set[str] = set()
    source_tensors = 0
    for info, tensor in source.iter_tensors(device="cpu"):
        source_tensors += 1
        try:
            module_name, is_a = parse_lora_tensor_name(info.name)
        except LoRAContractError:
            skipped.add(info.name)
            continue
        # PEFT keys contain a full module path; pack by the logical leaf name.
        pair = pairs.setdefault(module_name.rsplit(".", 1)[-1], {})
        key = "lora_a" if is_a else "lora_b"
        if key in pair:
            raise LoRAContractError(
                f"duplicate {key} tensor for LoRA module {module_name!r}"
            )
        pair[key] = tensor

    weights: dict[str, LoRALayerWeights] = {}
    for target in targets:
        slices = _target_slices(target)
        parts: list[tuple[Any, Any]] = []
        for logical in slices:
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
        if dtype is not None:
            packed_a = packed_a.to(dtype=dtype)
            packed_b = packed_b.to(dtype=dtype)
        if device is not None:
            packed_a = packed_a.to(device=device)
            packed_b = packed_b.to(device=device)
        weights[target] = LoRALayerWeights(
            module_name=target,
            lora_a=packed_a.contiguous(),
            lora_b=packed_b.contiguous(),
            rank=spec.r * len(parts),
        )

    used = {name for target in targets for name in _target_slices(target)}
    for module_name in pairs:
        if module_name not in used:
            skipped.add(module_name)
    return LoadedLoRAAdapter(
        spec=spec,
        weights=weights,
        source_tensors=source_tensors,
        source_bytes=source.total_bytes,
        skipped_modules=tuple(sorted(skipped)),
    )


def _target_slices(target: str) -> tuple[str, ...]:
    if target not in LORA_TARGET_SLICES:
        raise LoRAContractError(f"unsupported LoRA target {target!r}")
    if target == "qkv_proj":
        return ("q_proj", "k_proj", "v_proj")
    if target == "gate_up_proj":
        return ("gate_proj", "up_proj")
    return (target,)


__all__ = ["LoadedLoRAAdapter", "load_lora_adapter"]
