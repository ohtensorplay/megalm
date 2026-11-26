"""Runtime LoRA slot manager and the fixed-shape banks bound to a model.

Adapters are registered on CPU and activated into a small number of GPU slots,
as in vLLM's ``LoRAModelManager``.  A slot is pinned for the lifetime of every
request that uses it, so an in-flight request can never lose its weights to
LRU eviction; exhausting the slots is an explicit admission error, not an OOM.

Production properties of this module:

* every failure carries a stable machine code and an HTTP status hint so the
  gateway can answer 400/404/409/429 instead of a generic 503;
* the registry and slot table are protected by a re-entrant lock, because the
  engine may be driven from a replica thread while control frames arrive;
* activation validates geometry before touching a bank and restores an evicted
  adapter if a bank write fails, so a failed ``pin`` never corrupts slot state;
* counters for loads, activations, evictions and rejections are exported for
  the engine/Prometheus snapshot.

When LoRA is disabled, no banks are allocated, no model hook is installed and
``lora_add`` is never called.  LoRA requests also skip CUDA-graph replay and
speculative paths, so the captured graphs keep their base-weight semantics.
"""

from __future__ import annotations

import threading
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import tensorplay

from megalm.contracts.lora import (
    GENERIC_LORA_MODEL_TYPES,
    LoRAAdapterSpec,
    LoRAConfig,
    LoRALayerWeights,
    LoRARequest,
)
from megalm.kernels.lora import lora_add
from megalm.loading.lora import load_lora_adapter


class LoRAError(RuntimeError):
    """Base failure for adapter registration, activation and release."""

    code: str = "lora_unavailable"
    retryable: bool = True

    def __init__(self, message: str) -> None:
        super().__init__(message)
        # Instance attributes make the class attributes overridable per error
        # without leaking state between instances.
        self.code = type(self).code
        self.retryable = type(self).retryable


class LoRADisabled(LoRAError):
    """LoRA was referenced but the replica was started without it."""

    code = "lora_disabled"
    retryable = False


class LoRAValidationError(LoRAError):
    """The request, config or adapter geometry is invalid."""

    code = "lora_invalid"
    retryable = False


class LoRAAdapterNotFound(LoRAError):
    """No adapter with the requested id or name is registered."""

    code = "lora_adapter_not_found"
    retryable = False


class LoRAAdapterBusy(LoRAError):
    """The adapter is pinned by in-flight requests and cannot be unloaded."""

    code = "lora_adapter_busy"
    retryable = True


class LoRASlotsExhausted(LoRAError):
    """Every GPU slot is pinned; retry once an in-flight request completes."""

    code = "lora_slots_exhausted"
    retryable = True


@dataclass(frozen=True, slots=True)
class LoRABank:
    a: Any  # [slots, rank, in_features]
    b: Any  # [slots, out_features, rank]
    in_features: int
    out_features: int
    rank: int


@dataclass(slots=True)
class LoRAMetrics:
    """Monotonic counters describing the lifetime of the LoRA subsystem."""

    loads: int = 0
    failed_loads: int = 0
    unloads: int = 0
    activations: int = 0
    evictions: int = 0
    pins: int = 0
    rejections: int = 0

    def snapshot(self) -> dict[str, int]:
        return {
            "loads": self.loads,
            "failed_loads": self.failed_loads,
            "unloads": self.unloads,
            "activations": self.activations,
            "evictions": self.evictions,
            "pins": self.pins,
            "rejections": self.rejections,
        }


class LoRAState:
    """GPU-resident banks: one bank per (layer, target) pair.

    The bank shapes are fixed at startup; changing the active adapter only
    copies values into a slot, which is safe under CUDA Graph capture.
    ``banks`` remains a view of layer 0 for callers written for the old
    single-layer contract.
    """

    def __init__(
        self,
        banks: Mapping[str, LoRABank] | Mapping[int, Mapping[str, LoRABank]],
        *,
        max_loras: int,
        use_triton: bool,
    ) -> None:
        if not banks:
            raise LoRAError("LoRA state requires at least one target bank")
        if all(isinstance(key, int) for key in banks):
            self.layers: dict[int, dict[str, LoRABank]] = {
                int(index): dict(targets) for index, targets in banks.items()  # type: ignore[union-attr]
            }
        else:
            self.layers = {0: dict(banks)}  # type: ignore[arg-type]
        if not self.layers or not any(self.layers.values()):
            raise LoRAError("LoRA state requires at least one target bank")
        self.banks: dict[str, LoRABank] = self.layers[min(self.layers)]
        self.max_loras = max_loras
        self.use_triton = use_triton

    @staticmethod
    def _as_layers(
        weights: Mapping[str, LoRALayerWeights]
        | Mapping[int, Mapping[str, LoRALayerWeights]]
        | None,
    ) -> dict[int, Mapping[str, LoRALayerWeights]] | None:
        if weights is None:
            return None
        if all(isinstance(key, int) for key in weights):
            return {int(index): dict(targets) for index, targets in weights.items()}  # type: ignore[union-attr]
        return {0: dict(weights)}  # type: ignore[arg-type]

    def set_lora(
        self,
        slot: int,
        weights: Mapping[str, LoRALayerWeights]
        | Mapping[int, Mapping[str, LoRALayerWeights]]
        | None,
        *,
        scale: float = 1.0,
    ) -> None:
        if not 0 <= slot < self.max_loras:
            raise LoRAError(f"LoRA slot {slot} is out of range")
        layers = self._as_layers(weights)
        for index, targets in self.layers.items():
            for target, bank in targets.items():
                weight = None if layers is None else layers.get(index, {}).get(target)
                if weight is None:
                    bank.a[slot].zero_()
                    bank.b[slot].zero_()
                    continue
                if weight.rank > bank.rank:
                    raise LoRAError(
                        f"LoRA rank {weight.rank} exceeds bank capacity {bank.rank} "
                        f"for layer {index} / {target!r}"
                    )
                if (
                    weight.lora_a.shape[1] != bank.in_features
                    or weight.lora_b.shape[0] != bank.out_features
                ):
                    raise LoRAError(
                        f"LoRA geometry for layer {index} / {target!r} does not match the model"
                    )
                bank.a[slot, : weight.rank].copy_(weight.lora_a)
                bank.b[slot, :, : weight.rank].copy_(weight.lora_b * scale)
                if weight.rank < bank.rank:
                    bank.a[slot, weight.rank :].zero_()
                    bank.b[slot, :, weight.rank :].zero_()

    def apply(
        self,
        target: str,
        hidden: Any,
        output: Any,
        slots: Any | None,
        *,
        layer_index: int = 0,
    ) -> None:
        bank = self.layers.get(layer_index, {}).get(target)
        if bank is None or slots is None:
            return
        lora_add(
            output,
            hidden,
            bank.a,
            bank.b,
            slots,
            use_triton=self.use_triton,
        )


@dataclass(slots=True)
class _RegisteredAdapter:
    adapter_id: int
    name: str
    path: str
    spec: LoRAAdapterSpec
    weights: dict[str, LoRALayerWeights]
    """Flat view for single-layer adapters (backward compatibility)."""

    layers: dict[int, dict[str, LoRALayerWeights]]
    """Per-layer packed weights consumed by :class:`LoRAState`."""

    slot: int = -1
    refs: int = 0
    touched: int = 0


class LoRAManager:
    """Registry + slot allocator for dynamically loaded LoRA adapters."""

    def __init__(
        self,
        config: LoRAConfig,
        *,
        device: Any,
        dtype: Any,
        use_triton: bool = True,
    ) -> None:
        self.config = config
        self.device = device
        self.dtype = dtype
        self.use_triton = use_triton
        self.state: LoRAState | None = None
        self.metrics = LoRAMetrics()
        self._lock = threading.RLock()
        self._adapters: dict[int, _RegisteredAdapter] = {}
        self._by_name: dict[str, int] = {}
        self._slots: list[int | None] = [None] * config.max_loras
        self._clock = 0

    @property
    def enabled(self) -> bool:
        return self.config.enable_lora and self.state is not None

    def bind_model(self, model: Any) -> None:
        """Allocate banks and install hooks on a generic causal LM."""

        if not self.config.enable_lora:
            return
        model_type = str(getattr(model.config, "model_type", ""))
        if model_type not in GENERIC_LORA_MODEL_TYPES:
            raise LoRAValidationError(
                f"LoRA is not supported for model_type {model_type!r}; "
                "the generic causal LM family is the supported target"
            )
        with self._lock:
            max_rank = self.config.max_lora_rank
            layer_banks: dict[int, dict[str, LoRABank]] = {}
            for layer_index, layer in enumerate(model.model.layers):
                attention = layer.self_attn
                mlp = layer.mlp
                geometry = {
                    "qkv_proj": (
                        attention.qkv_proj.weight.shape[1],
                        attention.qkv_proj.weight.shape[0],
                        3 * max_rank,
                    ),
                    "o_proj": (
                        attention.o_proj.weight.shape[1],
                        attention.o_proj.weight.shape[0],
                        max_rank,
                    ),
                    "gate_up_proj": (
                        mlp.gate_up_weight.shape[1],
                        mlp.gate_up_weight.shape[0],
                        2 * max_rank,
                    ),
                    "down_proj": (
                        mlp.down_proj.weight.shape[1],
                        mlp.down_proj.weight.shape[0],
                        max_rank,
                    ),
                }
                banks: dict[str, LoRABank] = {}
                for target, (in_features, out_features, rank) in geometry.items():
                    banks[target] = LoRABank(
                        a=tensorplay.zeros(
                            (self.config.max_loras, rank, in_features),
                            dtype=self.dtype,
                            device=self.device,
                        ),
                        b=tensorplay.zeros(
                            (self.config.max_loras, out_features, rank),
                            dtype=self.dtype,
                            device=self.device,
                        ),
                        in_features=int(in_features),
                        out_features=int(out_features),
                        rank=rank,
                    )
                layer_banks[layer_index] = banks
            self.state = LoRAState(
                layer_banks,
                max_loras=self.config.max_loras,
                use_triton=self.use_triton,
            )
            bind_lora = getattr(model, "bind_lora", None)
            if bind_lora is None:
                raise LoRAValidationError(
                    "the native model does not expose bind_lora()"
                )
            bind_lora(self.state)

    def load(
        self,
        path: str | Path,
        *,
        name: str | None = None,
        adapter_id: int | None = None,
        activate: bool = True,
    ) -> dict[str, Any]:
        """Register one PEFT adapter; optionally activate it into a GPU slot."""

        if not self.config.enable_lora:
            raise LoRADisabled("LoRA is disabled; start with --enable-lora")
        if self.state is None:
            raise LoRAError("LoRA banks are not bound to a model yet")
        adapter_path = str(Path(path).expanduser())
        self._check_path(adapter_path)
        with self._lock:
            if adapter_id is None:
                adapter_id = max(self._adapters, default=0) + 1
            if adapter_id <= 0:
                raise LoRAValidationError("LoRA adapter id must be positive")
            if adapter_id in self._adapters:
                raise LoRAAdapterBusy(
                    f"LoRA adapter id {adapter_id} is already registered"
                )
            resolved_name = name or Path(adapter_path).name
            if not resolved_name:
                raise LoRAValidationError("LoRA adapter name must not be empty")
            if resolved_name in self._by_name:
                raise LoRAAdapterBusy(
                    f"LoRA adapter {resolved_name!r} is already registered"
                )
            try:
                adapter = load_lora_adapter(
                    adapter_path,
                    max_lora_rank=self.config.max_lora_rank,
                    dtype=self.dtype,
                    device="cpu",
                )
            except Exception as exc:
                self.metrics.failed_loads += 1
                if isinstance(exc, LoRAError):
                    raise
                raise LoRAValidationError(str(exc)) from exc
            entry = _RegisteredAdapter(
                adapter_id=adapter_id,
                name=resolved_name,
                path=adapter_path,
                spec=adapter.spec,
                weights=adapter.weights,
                layers=adapter.layers,
            )
            try:
                self._validate_weights(entry.layers)
            except LoRAError:
                self.metrics.failed_loads += 1
                raise
            self._adapters[adapter_id] = entry
            self._by_name[resolved_name] = adapter_id
            try:
                if activate:
                    self._activate(entry)
                    self._clock += 1
                    entry.touched = self._clock
            except BaseException:
                self._adapters.pop(adapter_id, None)
                self._by_name.pop(resolved_name, None)
                self.metrics.failed_loads += 1
                # Rolling back an activation must not leave a pinned adapter.
                if entry.slot >= 0:
                    self.state.set_lora(entry.slot, None)
                    self._slots[entry.slot] = None
                    entry.slot = -1
                raise
            self.metrics.loads += 1
            return self.describe_adapter(entry)

    def unload(self, reference: str | int) -> dict[str, Any]:
        adapter_id = self._resolve_id(reference)
        with self._lock:
            entry = self._adapters[adapter_id]
            if entry.refs:
                raise LoRAAdapterBusy(
                    f"LoRA adapter {entry.name!r} is still used by "
                    f"{entry.refs} request(s)"
                )
            if entry.slot >= 0:
                assert self.state is not None
                self.state.set_lora(entry.slot, None)
                self._slots[entry.slot] = None
            del self._adapters[adapter_id]
            self._by_name.pop(entry.name, None)
            self.metrics.unloads += 1
            return {"id": adapter_id, "name": entry.name, "unloaded": True}

    def pin(self, request: LoRARequest | None) -> int:
        """Resolve, activate and pin the adapter for one request. -1 = base."""

        if request is None:
            return -1
        if not self.enabled:
            raise LoRADisabled(
                "request referenced a LoRA adapter but LoRA is disabled"
            )
        if request.scale != 1.0:
            raise LoRAValidationError(
                "per-request LoRA scale is not supported in multi-LoRA mode"
            )
        with self._lock:
            adapter_id = self._resolve_id(
                request.lora_int_id if request.lora_int_id > 0 else request.lora_name
            )
            assert adapter_id is not None
            entry = self._adapters[adapter_id]
            slot = self._activate(entry)
            entry.refs += 1
            self._clock += 1
            entry.touched = self._clock
            self.metrics.pins += 1
            return slot

    def unpin(self, reference: str | int | None) -> None:
        if reference is None:
            return
        with self._lock:
            adapter_id = self._resolve_id(reference, missing_ok=True)
            if adapter_id is None:
                return
            entry = self._adapters[adapter_id]
            if entry.refs > 0:
                entry.refs -= 1

    def _check_path(self, path: str) -> None:
        """Restrict runtime loads to configured roots when an allowlist is set."""

        roots = self.config.allowed_roots
        if not roots:
            return
        candidate = Path(path).resolve()
        for root in roots:
            try:
                if candidate.is_relative_to(Path(root).expanduser().resolve()):
                    return
            except OSError:
                continue
        raise LoRAValidationError(
            f"LoRA path {path!r} is outside the configured allowed roots"
        )

    def _resolve_id(
        self,
        reference: str | int,
        *,
        missing_ok: bool = False,
    ) -> int | None:
        adapter_id: int | None
        if isinstance(reference, int) and not isinstance(reference, bool):
            adapter_id = reference
        else:
            adapter_id = self._by_name.get(str(reference))
        if adapter_id is None or adapter_id not in self._adapters:
            if missing_ok:
                return None
            raise LoRAAdapterNotFound(f"unknown LoRA adapter {reference!r}")
        return adapter_id

    def _validate_weights(
        self,
        weights: Mapping[str, LoRALayerWeights]
        | Mapping[int, Mapping[str, LoRALayerWeights]],
    ) -> None:
        """Reject geometrically invalid adapters before any bank is written."""

        if self.state is None:
            return
        layers = LoRAState._as_layers(weights) or {}
        for index, targets in self.state.layers.items():
            source = layers.get(index, {})
            for target, bank in targets.items():
                weight = source.get(target)
                if weight is None:
                    continue
                if weight.rank > bank.rank:
                    raise LoRAValidationError(
                        f"LoRA rank {weight.rank} exceeds bank capacity {bank.rank} "
                        f"for layer {index} / {target!r}"
                    )
                if (
                    weight.lora_a.ndim != 2
                    or weight.lora_b.ndim != 2
                    or tuple(weight.lora_a.shape) != (weight.rank, bank.in_features)
                    or tuple(weight.lora_b.shape) != (bank.out_features, weight.rank)
                ):
                    raise LoRAValidationError(
                        f"LoRA geometry for layer {index} / {target!r} does not match the model"
                    )

    def _activate(self, entry: _RegisteredAdapter) -> int:
        """Activate ``entry`` into a slot. The caller must hold ``_lock``."""

        if entry.slot >= 0:
            return entry.slot
        # Validate every target before evicting anything: a malformed adapter
        # must not disturb an LRU victim that is still in use by other banks.
        self._validate_weights(entry.layers)
        slot: int | None = None
        victim: _RegisteredAdapter | None = None
        for index, adapter_id in enumerate(self._slots):
            if adapter_id is None:
                slot = index
                break
        if slot is None:
            candidates = [
                index
                for index, adapter_id in enumerate(self._slots)
                if adapter_id is not None
                and self._adapters[adapter_id].refs == 0
            ]
            if not candidates:
                self.metrics.rejections += 1
                raise LoRASlotsExhausted(
                    f"all {self.config.max_loras} LoRA slots are pinned by "
                    "in-flight requests"
                )
            slot = min(
                candidates,
                key=lambda index: self._adapters[self._slots[index]].touched,
            )
            victim = self._adapters[self._slots[slot]]
        assert slot is not None and self.state is not None
        try:
            self.state.set_lora(slot, entry.layers)
        except BaseException:
            # Restore the evicted adapter so the slot is never half-written.
            if victim is not None:
                self.state.set_lora(slot, victim.layers)
            raise
        if victim is not None:
            victim.slot = -1
            self.metrics.evictions += 1
        entry.slot = slot
        self._slots[slot] = entry.adapter_id
        self.metrics.activations += 1
        return slot

    def describe_adapter(self, entry: _RegisteredAdapter) -> dict[str, Any]:
        return {
            "id": entry.adapter_id,
            "name": entry.name,
            "path": entry.path,
            "rank": entry.spec.r,
            "alpha": entry.spec.lora_alpha,
            "scaling": entry.spec.scaling,
            "target_modules": list(entry.spec.target_modules),
            "slot": entry.slot,
            "refs": entry.refs,
            "error_code": None,
        }

    def describe(self) -> list[dict[str, Any]]:
        with self._lock:
            return [self.describe_adapter(entry) for entry in self._adapters.values()]

    def stats(self) -> dict[str, Any]:
        with self._lock:
            result: dict[str, Any] = {
                "enabled": self.enabled,
                "registered": len(self._adapters),
                "max_loras": self.config.max_loras,
                "max_lora_rank": self.config.max_lora_rank,
                "active_slots": sum(1 for slot in self._slots if slot is not None),
                "pinned_slots": sum(
                    1 for entry in self._adapters.values() if entry.refs
                ),
            }
            result.update(self.metrics.snapshot())
            return result


__all__ = [
    "LoRAAdapterBusy",
    "LoRAAdapterNotFound",
    "LoRADisabled",
    "LoRAError",
    "LoRAManager",
    "LoRAMetrics",
    "LoRASlotsExhausted",
    "LoRAState",
    "LoRAValidationError",
]
