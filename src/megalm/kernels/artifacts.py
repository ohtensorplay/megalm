from __future__ import annotations

import hashlib
import json
from dataclasses import asdict
from pathlib import Path

import tensorplay

from megalm.kernels import fallback, tensorplay_backend
from megalm.kernels.binding_types import (
    KernelArtifact,
    KernelBindingError,
    KernelBindingSpec,
    TENSORPLAY_CUBLAS_LINEAR_BACKEND,
    TENSORPLAY_SDPA_PAGED_PREFILL_BACKEND,
    TRITON_BATCH_INVARIANT_LINEAR_BACKEND,
    TRITON_PAGED_PREFILL_BACKEND,
)
from megalm.kernels.ops import KernelPolicy

KERNEL_ABI_VERSION = 1


def discover_kernel_plan(
    policy: KernelPolicy,
    *,
    device: tensorplay.device | str,
    dtype: tensorplay.dtype,
    model_type: str | None = None,
) -> KernelBindingSpec:
    """Inspect selected provider artifacts without creating model tensors."""

    target_device = tensorplay.device(device)
    if policy.use_triton and target_device.type != "cuda":
        raise KernelBindingError("the Triton plan requires a CUDA device")
    prefill_backend = (
        TRITON_PAGED_PREFILL_BACKEND
        if policy.use_triton
        else TENSORPLAY_SDPA_PAGED_PREFILL_BACKEND
    )
    linear_backend = (
        TRITON_BATCH_INVARIANT_LINEAR_BACKEND
        if policy.use_triton and model_type == "qwen2_5_omni"
        else TENSORPLAY_CUBLAS_LINEAR_BACKEND
    )
    tensorplay_modules: tuple[object, ...] = (fallback, tensorplay_backend)
    artifacts = [
        _artifact(
            provider="tensorplay",
            variant="eager-cublas-sdpa",
            device=target_device,
            dtype=dtype,
            modules=tensorplay_modules,
        )
    ]
    if policy.use_triton:
        from megalm.kernels import (
            batch_invariant_linear,
            decode_relay,
            dflash_query,
            gdn,
            gdn_prefill,
            greedy_argmax,
            moe_decode,
            moe_grouped,
            moe_grouped_down,
            moe_grouped_route,
            moe_grouped_up,
            moe_top1,
            moe_topk,
            paged_decode,
            qwen35,
            rejection_sampling,
            repetition_penalty,
            rms_norm,
            rotary_cache,
            sampling_logits,
            silu_mul,
            topk_radix,
            speculative_verify,
        )

        triton_modules = (
            *((batch_invariant_linear,) if (
                linear_backend == TRITON_BATCH_INVARIANT_LINEAR_BACKEND
            ) else ()),
            rms_norm,
            silu_mul,
            rotary_cache,
            paged_decode,
            moe_decode,
            moe_grouped_route,
            moe_grouped_up,
            moe_grouped_down,
            moe_grouped,
            moe_top1,
            moe_topk,
            sampling_logits,
            repetition_penalty,
            topk_radix,
            rejection_sampling,
            greedy_argmax,
            decode_relay,
            dflash_query,
            speculative_verify,
            gdn,
            gdn_prefill,
            qwen35,
        )
        if prefill_backend == TRITON_PAGED_PREFILL_BACKEND:
            from megalm.kernels import paged_prefill

            triton_modules += (paged_prefill,)
        artifacts.append(
            _artifact(
                provider="triton",
                variant="jit-fixed-dispatch",
                device=target_device,
                dtype=dtype,
                modules=triton_modules,
            )
        )
    artifact_tuple = tuple(artifacts)
    digest_payload = {
        "abi": KERNEL_ABI_VERSION,
        "policy": asdict(policy),
        "linear_backend": linear_backend,
        "prefill_backend": prefill_backend,
        "artifacts": [asdict(artifact) for artifact in artifact_tuple],
    }
    binding_digest = hashlib.sha256(
        json.dumps(
            digest_payload,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    return KernelBindingSpec(
        policy=policy,
        linear_backend=linear_backend,
        prefill_backend=prefill_backend,
        artifacts=artifact_tuple,
        digest=binding_digest,
    )


def _artifact(
    *,
    provider: str,
    variant: str,
    device: tensorplay.device,
    dtype: tensorplay.dtype,
    modules: tuple[object, ...],
) -> KernelArtifact:
    digest = hashlib.sha256()
    identity = (
        provider,
        variant,
        str(KERNEL_ABI_VERSION),
        tensorplay.__version__,
        str(tensorplay.version.cuda),
        str(dtype),
        _device_target(device),
    )
    for item in identity:
        digest.update(item.encode("utf-8"))
        digest.update(b"\0")
    source_files = (Path(__file__), Path(__file__).with_name("bindings.py"))
    for path in source_files:
        try:
            digest.update(path.read_bytes())
        except OSError as exc:
            raise KernelBindingError(
                f"cannot hash kernel binding source {path}"
            ) from exc
    for module in modules:
        module_path = getattr(module, "__file__", None)
        if not module_path:
            raise KernelBindingError(
                f"cannot resolve source artifact for {module!r}"
            )
        path = Path(module_path)
        try:
            digest.update(path.read_bytes())
        except OSError as exc:
            raise KernelBindingError(
                f"cannot hash kernel artifact {path}"
            ) from exc
    return KernelArtifact(
        provider=provider,
        variant=variant,
        abi_version=KERNEL_ABI_VERSION,
        digest=digest.hexdigest(),
        weight_layout="qwen3.native-packed.v1",
    )


def _device_target(device: tensorplay.device) -> str:
    if device.type != "cuda":
        return device.type
    properties = tensorplay.cuda.get_device_properties(device)
    return f"sm_{properties.major}{properties.minor}:{properties.name}"


__all__ = ["KERNEL_ABI_VERSION", "discover_kernel_plan"]
