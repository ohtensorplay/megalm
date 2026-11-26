"""Immutable LoRA contracts shared by the gateway and the engine.

The shapes and scaling rules here follow the PEFT layout used by vLLM's LoRA
stack (``refs/vllm/vllm/lora``): ``lora_A`` is ``[rank, in]``, ``lora_B`` is
``[out, rank]`` and the adapter scale is ``alpha / rank`` (or
``alpha / sqrt(rank)`` for rsLoRA).  Nothing in this module imports TensorPlay
so contract validation stays GPU-free.
"""

from __future__ import annotations

import json
import math
import re
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any


class LoRAContractError(ValueError):
    """Raised when a LoRA request or PEFT adapter config violates the contract."""


# Machine-readable error codes shared by the engine, scheduler and gateway.
# The status mapping is the client-facing contract: unknown adapters are a
# 404, exhausted slots are a retryable 429, and configuration mistakes are 400.
LORA_ERROR_STATUS: dict[str, int] = {
    "lora_disabled": 400,
    "lora_invalid": 400,
    "lora_adapter_not_found": 404,
    "lora_adapter_busy": 409,
    "lora_slots_exhausted": 429,
}
_LORA_RETRYABLE_CODES: frozenset[str] = frozenset(
    {"lora_adapter_busy", "lora_slots_exhausted", "lora_unavailable"}
)


def lora_http_status(code: str) -> int:
    """Map a LoRA error code onto an HTTP status (503 for unknown codes)."""

    return LORA_ERROR_STATUS.get(code, 503)


def lora_retryable(code: str) -> bool:
    """Whether retrying a request that failed with ``code`` can succeed."""

    return code in _LORA_RETRYABLE_CODES


# Logical target names understood by the native generic-causal-LM model.
# Packed projections (q/k/v and gate/up) are represented by one bank whose
# Lora-A slices are stacked vertically and whose Lora-B is block diagonal.
LORA_TARGET_MODULES: tuple[str, ...] = (
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
)

# Mapping from the PEFT / HuggingFace logical module to the native packed
# projection that the generic model actually executes.
LORA_PACKED_TARGETS: dict[str, str] = {
    "q_proj": "qkv_proj",
    "k_proj": "qkv_proj",
    "v_proj": "qkv_proj",
    "o_proj": "o_proj",
    "gate_proj": "gate_up_proj",
    "up_proj": "gate_up_proj",
    "down_proj": "down_proj",
}

# Number of logical LoRA slices per packed native target.
LORA_TARGET_SLICES: dict[str, int] = {
    "qkv_proj": 3,
    "o_proj": 1,
    "gate_up_proj": 2,
    "down_proj": 1,
}

# Dense generic-causal-LM families whose packed QKV/gate-up geometry is known
# to the LoRA bank binder.  MoE-only families are excluded: expert LoRA needs a
# grouped kernel and is not part of this release.
GENERIC_LORA_MODEL_TYPES: frozenset[str] = frozenset(
    {
        "llama",
        "mistral",
        "qwen2",
        "yi",
        "deepseek",
        "phi3",
        "internlm2",
        "internlm3",
        "seed_oss",
        "granite",
        "ernie4_5",
        "olmo2",
        "minicpm",
    }
)

_ALLOWED_RANKS = frozenset({1, 2, 4, 8, 16, 32, 64, 128, 256})
_LORA_NAME_RE = re.compile(
    r"^(?:base_model\.model\.)?(?P<module>.+?)\."
    r"lora_(?P<kind>[AB])(?:\.default)?\.weight$"
)


@dataclass(frozen=True, slots=True)
class LoRARequest:
    """A per-request adapter reference (vLLM-compatible field names).

    ``lora_int_id`` is the globally unique adapter id; ``0`` means "resolve by
    name" and is filled in by the engine when the adapter is registered.  A
    request with no LoRA is represented by ``None``, never by a sentinel id.
    """

    lora_name: str = ""
    lora_int_id: int = 0
    lora_path: str = ""
    scale: float = 1.0

    def __post_init__(self) -> None:
        object.__setattr__(self, "lora_name", str(self.lora_name).strip())
        object.__setattr__(self, "lora_path", str(self.lora_path).strip())
        if not self.lora_name and self.lora_int_id <= 0:
            raise LoRAContractError(
                "LoRA request requires a name or a positive adapter id"
            )
        if (
            isinstance(self.lora_int_id, bool)
            or not isinstance(self.lora_int_id, int)
            or self.lora_int_id < 0
        ):
            raise LoRAContractError("LoRA adapter id must be a non-negative integer")
        if not math.isfinite(self.scale) or self.scale <= 0:
            raise LoRAContractError("LoRA scale must be positive and finite")

    @property
    def name(self) -> str:
        return self.lora_name

    @property
    def adapter_id(self) -> int:
        return self.lora_int_id

    @property
    def path(self) -> str:
        return self.lora_path

    @classmethod
    def from_payload(cls, value: Any) -> LoRARequest:
        """Parse an OpenAI-style ``lora_request`` extension field.

        Accepts a vLLM ``LoRARequest``-shaped object, a mapping with any of the
        common aliases, or a bare adapter name.
        """
        if value is None:
            raise LoRAContractError("LoRA request payload is empty")
        if isinstance(value, cls):
            return value
        if isinstance(value, str):
            return cls(lora_name=value)
        if isinstance(value, LoRARequest):
            return value
        if not isinstance(value, Mapping):
            raise LoRAContractError("lora_request must be a name or an object")
        name = (
            value.get("lora_name")
            or value.get("name")
            or value.get("adapter_name")
            or ""
        )
        raw_id = (
            value.get("lora_int_id")
            if value.get("lora_int_id") is not None
            else (
                value.get("lora_id")
                if value.get("lora_id") is not None
                else value.get("id")
            )
        )
        path = (
            value.get("lora_path")
            or value.get("path")
            or value.get("adapter_path")
            or ""
        )
        scale = value.get("scale", value.get("lora_scale", 1.0))
        if raw_id is None:
            raw_id = 0
        if isinstance(raw_id, bool) or not isinstance(raw_id, int):
            try:
                raw_id = int(raw_id)
            except (TypeError, ValueError) as exc:
                raise LoRAContractError("LoRA adapter id must be an integer") from exc
        try:
            scale_value = float(scale)
        except (TypeError, ValueError) as exc:
            raise LoRAContractError("LoRA scale must be a number") from exc
        return cls(
            lora_name=str(name),
            lora_int_id=raw_id,
            lora_path=str(path),
            scale=scale_value,
        )

    def to_wire(self) -> dict[str, Any]:
        return {
            "lora_name": self.lora_name,
            "lora_int_id": self.lora_int_id,
            "lora_path": "",
            "scale": self.scale,
        }

    @classmethod
    def from_wire(cls, payload: Mapping[str, Any] | None) -> LoRARequest | None:
        if payload is None:
            return None
        return cls.from_payload(payload)


@dataclass(frozen=True, slots=True)
class LoRAConfig:
    """Static LoRA deployment configuration, sealed in the engine plan."""

    enable_lora: bool = False
    lora_paths: tuple[str, ...] = ()
    max_loras: int = 4
    max_lora_rank: int = 16
    lora_dtype: str = "auto"
    allowed_roots: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "lora_paths", tuple(self.lora_paths))
        object.__setattr__(self, "allowed_roots", tuple(self.allowed_roots))
        if not isinstance(self.enable_lora, bool):
            raise LoRAContractError("enable_lora must be a boolean")
        if self.max_loras <= 0:
            raise LoRAContractError("max_loras must be positive")
        if self.max_lora_rank not in _ALLOWED_RANKS:
            raise LoRAContractError(
                "max_lora_rank must be one of "
                + ", ".join(str(rank) for rank in sorted(_ALLOWED_RANKS))
            )
        if self.lora_dtype not in {"auto", "float16", "bfloat16", "float32"}:
            raise LoRAContractError("unsupported LoRA dtype")
        if self.lora_paths and not self.enable_lora:
            raise LoRAContractError("lora_paths requires enable_lora")
        if len(set(self.lora_paths)) != len(self.lora_paths):
            raise LoRAContractError("lora_paths must be unique")


@dataclass(frozen=True, slots=True)
class LoRAAdapterSpec:
    """Validated subset of a PEFT ``adapter_config.json``."""

    r: int
    lora_alpha: float
    target_modules: tuple[str, ...]
    use_rslora: bool = False
    bias: str = "none"
    use_dora: bool = False
    modules_to_save: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "target_modules", tuple(self.target_modules))
        object.__setattr__(self, "modules_to_save", tuple(self.modules_to_save))
        if self.r <= 0:
            raise LoRAContractError("adapter rank must be positive")
        if not math.isfinite(self.lora_alpha) or self.lora_alpha <= 0:
            raise LoRAContractError("lora_alpha must be positive and finite")
        if not self.target_modules:
            raise LoRAContractError("adapter target_modules must not be empty")

    @property
    def scaling(self) -> float:
        if self.use_rslora:
            return self.lora_alpha / math.sqrt(self.r)
        return self.lora_alpha / self.r

    @classmethod
    def from_dict(
        cls,
        data: Mapping[str, Any],
        *,
        max_lora_rank: int,
    ) -> LoRAAdapterSpec:
        if not isinstance(data, Mapping):
            raise LoRAContractError("adapter_config.json must be an object")
        try:
            rank = int(data["r"])
        except (KeyError, TypeError, ValueError) as exc:
            raise LoRAContractError("adapter_config.json is missing rank 'r'") from exc
        if rank > max_lora_rank:
            raise LoRAContractError(
                f"adapter rank {rank} exceeds max_lora_rank {max_lora_rank}"
            )
        try:
            alpha = float(data["lora_alpha"])
        except (KeyError, TypeError, ValueError) as exc:
            raise LoRAContractError("adapter_config.json is missing lora_alpha") from exc
        targets = data.get("target_modules") or ()
        if isinstance(targets, str):
            targets = (targets,)
        try:
            targets = tuple(str(item) for item in targets)
        except TypeError as exc:
            raise LoRAContractError("target_modules must be a list of strings") from exc
        bias = str(data.get("bias", "none"))
        use_dora = bool(data.get("use_dora", False))
        modules_to_save = data.get("modules_to_save") or ()
        if isinstance(modules_to_save, str):
            modules_to_save = (modules_to_save,)
        if bias != "none" or use_dora or modules_to_save:
            raise LoRAContractError(
                "LoRA bias, DoRA and modules_to_save are not supported"
            )
        return cls(
            r=rank,
            lora_alpha=alpha,
            target_modules=targets,
            use_rslora=bool(data.get("use_rslora", False)),
            bias=bias,
            use_dora=use_dora,
            modules_to_save=tuple(str(item) for item in modules_to_save),
        )

    @classmethod
    def from_path(cls, path: str | Path, *, max_lora_rank: int) -> LoRAAdapterSpec:
        config_path = Path(path) / "adapter_config.json"
        try:
            data = json.loads(config_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise LoRAContractError(f"cannot read {config_path}: {exc}") from exc
        return cls.from_dict(data, max_lora_rank=max_lora_rank)


@dataclass(frozen=True, slots=True)
class LoRALayerWeights:
    """One logical PEFT LoRA pair, already scaled and on the target device.

    ``lora_a`` has shape ``[rank, in_features]`` and ``lora_b`` has shape
    ``[out_features, rank]``.  ``scaling`` from the adapter config is folded
    into ``lora_b`` when loading, so kernels use a unit scale at runtime.
    """

    module_name: str
    lora_a: Any
    lora_b: Any
    rank: int

    @property
    def in_features(self) -> int:
        return tuple(self.lora_a.shape)[-1]

    @property
    def out_features(self) -> int:
        return next(iter(self.lora_b.shape))


def parse_lora_tensor_name(name: str) -> tuple[str, bool]:
    """Return ``(module_name, is_lora_a)`` for a PEFT safetensors key."""

    match = _LORA_NAME_RE.fullmatch(name)
    if match is None:
        raise LoRAContractError(f"unrecognized LoRA tensor name: {name!r}")
    return match.group("module"), match.group("kind") == "A"


def resolve_lora_targets(spec: LoRAAdapterSpec) -> tuple[str, ...]:
    """Map a PEFT target list onto native packed target names, in fixed order."""

    resolved: set[str] = set()
    for module in spec.target_modules:
        leaf = module.rsplit(".", 1)[-1]
        if leaf not in LORA_PACKED_TARGETS:
            continue
        resolved.add(LORA_PACKED_TARGETS[leaf])
    return tuple(
        target for target in LORA_TARGET_SLICES if target in resolved
    )


__all__ = [
    "GENERIC_LORA_MODEL_TYPES",
    "LORA_ERROR_STATUS",
    "LORA_PACKED_TARGETS",
    "LORA_TARGET_MODULES",
    "LORA_TARGET_SLICES",
    "LoRAAdapterSpec",
    "LoRAConfig",
    "LoRAContractError",
    "LoRALayerWeights",
    "LoRARequest",
    "lora_http_status",
    "lora_retryable",
    "parse_lora_tensor_name",
    "resolve_lora_targets",
]
