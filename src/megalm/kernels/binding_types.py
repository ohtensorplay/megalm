from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from megalm.kernels.ops import KernelPolicy

KernelCallable = Callable[..., Any]
TENSORPLAY_CUBLAS_LINEAR_BACKEND = "tensorplay.cublas.linear"
TRITON_BATCH_INVARIANT_LINEAR_BACKEND = (
    "megalm.triton.batch_invariant_linear"
)
TRITON_PAGED_PREFILL_BACKEND = (
    "megalm.triton.varlen_paged_flash_prefill"
)
TENSORPLAY_SDPA_PAGED_PREFILL_BACKEND = "tensorplay.sdpa.paged_gather"
_LINEAR_BACKENDS = frozenset(
    {
        TENSORPLAY_CUBLAS_LINEAR_BACKEND,
        TRITON_BATCH_INVARIANT_LINEAR_BACKEND,
    }
)
_PAGED_PREFILL_BACKENDS = frozenset(
    {
        TRITON_PAGED_PREFILL_BACKEND,
        TENSORPLAY_SDPA_PAGED_PREFILL_BACKEND,
    }
)


class KernelBindingError(RuntimeError):
    """Raised when a cold-selected provider cannot be bound exactly."""


@dataclass(frozen=True, slots=True)
class KernelArtifact:
    provider: str
    variant: str
    abi_version: int
    digest: str
    weight_layout: str

    def __post_init__(self) -> None:
        if not self.provider or not self.variant or not self.weight_layout:
            raise KernelBindingError("kernel artifact identity is incomplete")
        if self.abi_version <= 0 or len(self.digest) != 64:
            raise KernelBindingError("kernel artifact digest or ABI is invalid")


@dataclass(frozen=True, slots=True)
class CoreKernelBindings:
    linear: KernelCallable
    rms_norm: KernelCallable
    fused_add_rms_norm: KernelCallable
    offset_rms_norm: KernelCallable
    fused_add_offset_rms_norm: KernelCallable
    silu_and_mul: KernelCallable
    fused_qkv_norm_rope_cache_write: KernelCallable
    rope_cache_write: KernelCallable
    paged_decode_attention: KernelCallable
    paged_prefill_attention: KernelCallable | None
    gdn_causal_conv_update: KernelCallable | None
    gdn_causal_conv_prefill: KernelCallable | None
    gdn_packed_recurrent_decode: KernelCallable | None
    gdn_packed_recurrent_prefill: KernelCallable | None
    gdn_gated_rms_norm: KernelCallable | None
    qwen35_active_kv_write: KernelCallable | None


@dataclass(frozen=True, slots=True)
class MoEKernelBindings:
    top1: KernelCallable
    top1_from_hidden: KernelCallable
    topk: KernelCallable
    routed_small: KernelCallable
    routed_fallback: KernelCallable
    routed_grouped: KernelCallable
    routed_grouped_from_hidden: KernelCallable


@dataclass(frozen=True, slots=True)
class SamplingKernelBindings:
    use_triton: bool
    greedy_argmax: KernelCallable
    prepare_logits: KernelCallable | None = None
    apply_repetition_penalty: KernelCallable | None = None
    apply_frequency_presence_penalty: KernelCallable | None = None
    topk_cutoffs: KernelCallable | None = None
    filtered_rejection_sample: KernelCallable | None = None


@dataclass(frozen=True, slots=True)
class SpeculativeKernelBindings:
    dflash_query_setup: KernelCallable
    greedy_chain_verify: KernelCallable


@dataclass(frozen=True, slots=True)
class KernelBindingSpec:
    policy: KernelPolicy
    linear_backend: str
    prefill_backend: str
    artifacts: tuple[KernelArtifact, ...]
    digest: str

    def __post_init__(self) -> None:
        if self.linear_backend not in _LINEAR_BACKENDS:
            raise KernelBindingError("linear backend is invalid")
        if (
            self.linear_backend == TRITON_BATCH_INVARIANT_LINEAR_BACKEND
            and not self.policy.use_triton
        ):
            raise KernelBindingError("Triton linear requires Triton policy")
        if self.prefill_backend not in _PAGED_PREFILL_BACKENDS:
            raise KernelBindingError("paged prefill backend is invalid")
        if (
            self.prefill_backend == TRITON_PAGED_PREFILL_BACKEND
            and not self.policy.use_triton
        ):
            raise KernelBindingError("Triton paged prefill requires Triton policy")
        if not self.artifacts or len(self.digest) != 64:
            raise KernelBindingError("kernel binding spec is incomplete")


@dataclass(frozen=True, slots=True)
class KernelBindings:
    """Process-local direct call targets created once during cold start."""

    policy: KernelPolicy
    linear_backend: str
    prefill_backend: str
    core: CoreKernelBindings
    moe: MoEKernelBindings
    sampling: SamplingKernelBindings
    speculative: SpeculativeKernelBindings
    artifacts: tuple[KernelArtifact, ...]
    digest: str

    def __post_init__(self) -> None:
        if self.linear_backend not in _LINEAR_BACKENDS:
            raise KernelBindingError("bound linear backend is invalid")
        if (
            self.linear_backend == TRITON_BATCH_INVARIANT_LINEAR_BACKEND
            and not self.policy.use_triton
        ):
            raise KernelBindingError("Triton linear requires Triton policy")
        if self.prefill_backend not in _PAGED_PREFILL_BACKENDS:
            raise KernelBindingError("bound paged prefill backend is invalid")
        if (
            self.prefill_backend == TRITON_PAGED_PREFILL_BACKEND
            and not self.policy.use_triton
        ):
            raise KernelBindingError("Triton paged prefill requires Triton policy")
        if not self.artifacts or len(self.digest) != 64:
            raise KernelBindingError("kernel binding identity is incomplete")

    def artifact_for_backend(self, backend: str) -> KernelArtifact:
        provider = "triton" if ".triton." in backend else "tensorplay"
        for artifact in self.artifacts:
            if artifact.provider == provider:
                return artifact
        raise KernelBindingError(
            f"no {provider!r} artifact is bound for backend {backend!r}"
        )


__all__ = [
    "CoreKernelBindings",
    "KernelArtifact",
    "KernelBindingError",
    "KernelBindingSpec",
    "KernelBindings",
    "KernelCallable",
    "MoEKernelBindings",
    "SamplingKernelBindings",
    "SpeculativeKernelBindings",
    "TENSORPLAY_CUBLAS_LINEAR_BACKEND",
    "TENSORPLAY_SDPA_PAGED_PREFILL_BACKEND",
    "TRITON_BATCH_INVARIANT_LINEAR_BACKEND",
    "TRITON_PAGED_PREFILL_BACKEND",
]
