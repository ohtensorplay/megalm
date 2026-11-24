from __future__ import annotations

import hashlib
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

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
    attention_config: Any = None,
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
    attention_mode = "causal" if attention_config is None else attention_config.mode
    attention_backend = (
        "" if attention_config is None else attention_config.backend_id
    )
    if attention_config is not None and attention_mode != "causal":
        # The typed backend id is sealed instead of the Triton/SDPA default.
        prefill_backend = attention_backend
    tensorplay_modules: tuple[object, ...] = (fallback, tensorplay_backend)
    if attention_mode != "causal":
        from megalm.kernels import attention
        from megalm.kernels.attention import (
            mla,
            non_causal,
            sdpa_base,
            sink,
            sparse,
            sparse_select,
            types,
        )

        # Seal every typed backend source so any change invalidates the digest.
        tensorplay_modules += (
            attention,
            types,
            sdpa_base,
            non_causal,
            sink,
            sparse,
            sparse_select,
            mla,
        )
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
        if attention_mode != "causal":
            from megalm.kernels.attention import sparse_select

            triton_modules += (sparse_select,)
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
        "attention_mode": attention_mode,
        "attention_backend": attention_backend,
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
        attention_mode=attention_mode,
        attention_backend=attention_backend,
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
