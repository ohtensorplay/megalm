"""Hardware provider registry and per-provider capability evidence lock.

This is the cold-path counterpart of ``kernels/capabilities.py``: the latter
guards *backend* selection inside one hardware family, while this module owns
*which hardware provider* is discovered, what dtype/backend set it declares,
and a digest that DISCOVER and DRAFT can compare to detect drift.

Design constraints from ``devdoc/GAP-PLAN.md`` §6 (G5):

* No universal hardware abstraction layer — a provider is just a name, a
  probe, and a declaration of supported dtypes / backend ids.
* **An unselected provider is neither imported nor allocated.** Providers are
  registered as factories; ``tensorplay`` is imported inside
  :meth:`CUDAProvider.probe`, never at module import.
* Disabled providers (``MEGALM_DISABLE_PROVIDERS=cpu,cuda``) are not even
  constructed; ``select`` rejects them with a cold error.
* Unsupported hardware / dtype / backend is rejected before serving, never
  through a runtime kernel exception.
* Each provider has an independent evidence digest, sealed into the plan by
  comparing DISCOVER and DRAFT evidence.

The CPU provider produces a *portable plan* (tensorplay / sdpa backends only);
full CPU inference serving is a later step. The CUDA provider reproduces the
existing Ampere probe exactly so enabling this registry does not change the
current NVIDIA path.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from typing import Callable, Iterable, Mapping, Protocol

_PROVIDER_ENV = "MEGALM_DISABLE_PROVIDERS"

# Backend ids that need a CUDA/Triton provider. Kept as a declaration here so a
# new kernel must be admitted before it can be selected for a hardware family.
_CUDA_ONLY_BACKENDS = frozenset(
    {
        "megalm.triton.batch_invariant_linear",
        "megalm.triton.varlen_paged_flash_prefill",
        "megalm.triton.single_pass_paged_online_softmax",
        "megalm.triton.split_kv_online_softmax",
    }
)

_PORTABLE_BACKENDS = frozenset(
    {
        "tensorplay.cublas.linear",
        "tensorplay.sdpa.paged_gather",
        "megalm.tensorplay.paged_decode",
        "megalm.attention.non_causal_sdpa",
        "megalm.attention.sink_sdpa",
        "megalm.attention.sparse_index_sdpa",
        "megalm.attention.mla_latent_prefill",
        "megalm.attention.mla_latent_decode",
    }
)

_CUDA_BACKENDS = _PORTABLE_BACKENDS | _CUDA_ONLY_BACKENDS
_DTYPES = ("float16", "bfloat16")


class ProviderError(RuntimeError):
    """A hardware provider is missing, disabled, or cannot satisfy a request."""


@dataclass(frozen=True, slots=True)
class ProviderEvidence:
    """Immutable probe result plus its capability-evidence digest."""

    provider: str
    accelerator: str
    device_name: str
    compute_capability: tuple[int, int]
    total_memory_bytes: int
    free_memory_bytes: int
    supported_dtypes: tuple[str, ...]
    backends: tuple[str, ...]
    digest: str

    @classmethod
    def create(
        cls,
        *,
        provider: str,
        accelerator: str,
        device_name: str,
        compute_capability: tuple[int, int],
        total_memory_bytes: int,
        free_memory_bytes: int,
        supported_dtypes: Iterable[str],
        backends: Iterable[str],
    ) -> ProviderEvidence:
        dtypes = tuple(sorted(supported_dtypes))
        ids = tuple(sorted(backends))
        if not provider or not accelerator or not device_name:
            raise ProviderError("provider evidence identity is incomplete")
        if total_memory_bytes <= 0 or free_memory_bytes < 0:
            raise ProviderError("provider memory evidence must be non-negative")
        payload = {
            "provider": provider,
            "accelerator": accelerator,
            "device_name": device_name,
            "compute_capability": list(compute_capability),
            "total_memory_bytes": total_memory_bytes,
            "supported_dtypes": list(dtypes),
            "backends": list(ids),
        }
        digest = hashlib.sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
        return cls(
            provider=provider,
            accelerator=accelerator,
            device_name=device_name,
            compute_capability=tuple(compute_capability),
            total_memory_bytes=total_memory_bytes,
            free_memory_bytes=free_memory_bytes,
            supported_dtypes=dtypes,
            backends=ids,
            digest=digest,
        )

    def verify(self, expected: ProviderEvidence) -> None:
        """Reject drift between a DISCOVER probe and a DRAFT probe."""

        if self.digest != expected.digest:
            raise ProviderError(
                f"provider evidence drift for {self.provider}: "
                f"{expected.digest[:12]} -> {self.digest[:12]}"
            )

    def describe(self) -> dict[str, object]:
        return {
            "provider": self.provider,
            "accelerator": self.accelerator,
            "device_name": self.device_name,
            "compute_capability": self.compute_capability,
            "total_memory_bytes": self.total_memory_bytes,
            "supported_dtypes": self.supported_dtypes,
            "backends": self.backends,
            "digest": self.digest,
        }


class HardwareProvider(Protocol):
    """A hardware family's probe and declarations (no universal HAL)."""

    name: str
    accelerator: str

    def probe(
        self, *, device: str, total_memory_bytes: int | None = None
    ) -> ProviderEvidence: ...

    def supports(self, *, dtype: str, backends: Iterable[str]) -> bool: ...

    def validate(self, *, dtype: str, backends: Iterable[str]) -> None: ...


@dataclass(frozen=True, slots=True)
class _ProviderBase:
    supported_dtypes: tuple[str, ...]
    backends: frozenset[str]

    def supports(self, *, dtype: str, backends: Iterable[str]) -> bool:
        return dtype in self.supported_dtypes and set(backends) <= self.backends

    def validate(self, *, dtype: str, backends: Iterable[str]) -> None:
        if dtype not in self.supported_dtypes:
            raise ProviderError(
                f"{self.name} provider does not support dtype {dtype!r}"
            )
        unknown = sorted(set(backends) - self.backends)
        if unknown:
            raise ProviderError(
                f"{self.name} provider does not declare backends {unknown}"
            )


@dataclass(frozen=True, slots=True)
class CPUProvider(_ProviderBase):
    """Portable reference provider. Imports no CUDA / Triton module."""

    name: str = "cpu"
    accelerator: str = "cpu"
    supported_dtypes: tuple[str, ...] = _DTYPES + ("float32",)
    backends: frozenset[str] = _PORTABLE_BACKENDS

    def probe(
        self, *, device: str = "cpu", total_memory_bytes: int | None = None
    ) -> ProviderEvidence:
        memory = total_memory_bytes or (1 << 30)
        return ProviderEvidence.create(
            provider=self.name,
            accelerator=self.accelerator,
            device_name="cpu-reference",
            compute_capability=(0, 0),
            total_memory_bytes=memory,
            free_memory_bytes=memory,
            supported_dtypes=self.supported_dtypes,
            backends=self.backends,
        )


@dataclass(frozen=True, slots=True)
class CUDAProvider(_ProviderBase):
    """NVIDIA provider; imports tensorplay only when probed."""

    name: str = "cuda"
    accelerator: str = "cuda"
    supported_dtypes: tuple[str, ...] = _DTYPES
    backends: frozenset[str] = _CUDA_BACKENDS

    def probe(
        self, *, device: str = "cuda", total_memory_bytes: int | None = None
    ) -> ProviderEvidence:
        import tensorplay  # lazy: an unselected provider never imports this

        tensorplay.cuda.synchronize(device)
        free, total = tensorplay.cuda.mem_get_info(device)
        props = tensorplay.cuda.get_device_properties(device)
        if (props.major, props.minor) < (8, 0):
            raise ProviderError("MegaLM v1 requires NVIDIA Ampere or newer")
        return ProviderEvidence.create(
            provider=self.name,
            accelerator=self.accelerator,
            device_name=props.name,
            compute_capability=(int(props.major), int(props.minor)),
            total_memory_bytes=int(total),
            free_memory_bytes=int(free),
            supported_dtypes=self.supported_dtypes,
            backends=self.backends,
        )


class ProviderRegistry:
    """Factories + disabled set; selects exactly one provider cold."""

    def __init__(
        self,
        factories: Mapping[str, Callable[[], HardwareProvider]],
        *,
        disabled: Iterable[str] = (),
    ) -> None:
        self._factories = dict(factories)
        self._disabled = frozenset(disabled)
        unknown = self._disabled - set(self._factories)
        if unknown:
            raise ProviderError(f"unknown disabled providers {sorted(unknown)}")

    @property
    def disabled(self) -> frozenset[str]:
        return self._disabled

    @property
    def available(self) -> tuple[str, ...]:
        return tuple(sorted(set(self._factories) - self._disabled))

    def select(
        self, *, device: str = "cuda", provider: str | None = None
    ) -> HardwareProvider:
        name = provider or ("cuda" if device.startswith("cuda") else "cpu")
        if name in self._disabled:
            raise ProviderError(f"hardware provider {name!r} is disabled")
        factory = self._factories.get(name)
        if factory is None:
            raise ProviderError(f"unknown hardware provider {name!r}")
        return factory()

    def for_accelerator(self, accelerator: str) -> HardwareProvider:
        for name in self.available:
            instance = self._factories[name]()
            if instance.accelerator == accelerator:
                return instance
        raise ProviderError(f"no provider for accelerator {accelerator!r}")

    def validate(
        self, *, accelerator: str, dtype: str, backends: Iterable[str]
    ) -> None:
        self.for_accelerator(accelerator).validate(dtype=dtype, backends=backends)

    def snapshot(self) -> dict[str, object]:
        return {"available": list(self.available), "disabled": list(sorted(self._disabled))}


def _disabled_from_env() -> tuple[str, ...]:
    raw = os.environ.get(_PROVIDER_ENV, "")
    return tuple(item.strip() for item in raw.split(",") if item.strip())


DEFAULT_REGISTRY = ProviderRegistry(
    {"cpu": CPUProvider, "cuda": CUDAProvider},
    disabled=_disabled_from_env(),
)


def probe_hardware(
    *, device: str = "cuda", provider: str | None = None
) -> ProviderEvidence:
    """Select one provider and probe its capability evidence."""

    return DEFAULT_REGISTRY.select(device=device, provider=provider).probe(device=device)


__all__ = [
    "CPUProvider",
    "CUDAProvider",
    "DEFAULT_REGISTRY",
    "HardwareProvider",
    "ProviderError",
    "ProviderEvidence",
    "ProviderRegistry",
    "probe_hardware",
]
