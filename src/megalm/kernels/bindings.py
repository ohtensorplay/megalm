from __future__ import annotations

from dataclasses import replace
from functools import partial
from typing import Any

import tensorplay
import tensorplay.nn.functional as F

from megalm.kernels import fallback, tensorplay_backend
from megalm.kernels.artifacts import discover_kernel_plan
from megalm.kernels.binding_types import (
    CoreKernelBindings,
    KernelArtifact,
    KernelBindingError,
    KernelBindingSpec,
    KernelBindings,
    KernelCallable,
    MoEKernelBindings,
    SamplingKernelBindings,
    SpeculativeKernelBindings,
    TRITON_BATCH_INVARIANT_LINEAR_BACKEND,
    TENSORPLAY_SDPA_PAGED_PREFILL_BACKEND,
    TRITON_PAGED_PREFILL_BACKEND,
)
from megalm.kernels.ops import (
    KernelPolicy,
    rope_and_cache_write as dispatch_rope_cache_write,
)

def bind_kernel_plan(
    policy: KernelPolicy,
    *,
    device: tensorplay.device | str,
    dtype: tensorplay.dtype,
    expected: KernelBindingSpec | None = None,
    model_type: str | None = None,
    attention_config: Any = None,
    sink_logits: tuple[float, ...] = (),
) -> KernelBindings:
    """Resolve every provider import and callable before model construction."""

    target_device = tensorplay.device(device)
    if policy.use_triton and target_device.type != "cuda":
        raise KernelBindingError("the Triton plan requires a CUDA device")
    discovered = discover_kernel_plan(
        policy,
        device=target_device,
        dtype=dtype,
        model_type=model_type,
        attention_config=attention_config,
    )
    if expected is not None and discovered != expected:
        raise KernelBindingError(
            "kernel artifacts changed between discovery and binding"
        )
    linear: KernelCallable = F.linear
    if discovered.linear_backend == TRITON_BATCH_INVARIANT_LINEAR_BACKEND:
        from megalm.kernels.batch_invariant_linear import (
            batch_invariant_linear,
        )

        linear = batch_invariant_linear
    paged_prefill_attention: KernelCallable | None = None
    if discovered.prefill_backend == TRITON_PAGED_PREFILL_BACKEND:
        from megalm.kernels.paged_prefill import paged_prefill_attention
    elif (
        discovered.prefill_backend == TENSORPLAY_SDPA_PAGED_PREFILL_BACKEND
        and policy.use_triton
        and model_type == "qwen2_5_omni"
    ):
        from megalm.kernels.paged_prefill_sdpa import (
            paged_prefill_attention,
        )
    if policy.use_triton:
        from megalm.kernels import gdn, gdn_prefill, qwen35
        from megalm.kernels.greedy_argmax import greedy_argmax
        from megalm.kernels.moe_decode import routed_moe_decode
        from megalm.kernels.moe_grouped import (
            routed_moe_grouped_tensorcore,
        )
        from megalm.kernels.moe_top1 import (
            moe_top1,
            moe_top1_from_hidden,
        )
        from megalm.kernels.moe_topk import moe_topk_softmax
        from megalm.kernels.paged_decode import paged_decode_attention
        from megalm.kernels.rejection_sampling import (
            filtered_rejection_sample,
        )
        from megalm.kernels.frequency_penalty import (
            apply_frequency_presence_penalty_,
        )
        from megalm.kernels.repetition_penalty import (
            apply_repetition_penalty_,
        )
        from megalm.kernels.rms_norm import (
            fused_add_offset_rms_norm,
            fused_add_rms_norm,
            offset_rms_norm,
            rms_norm,
        )
        from megalm.kernels.rotary_cache import (
            fused_qkv_norm_rope_cache_write,
        )
        from megalm.kernels.sampling_logits import (
            prepare_sampling_logits,
        )
        from megalm.kernels.silu_mul import silu_and_mul
        from megalm.kernels.topk_radix import topk_cutoffs
        from megalm.kernels.dflash_query import dflash_query_setup_into
        from megalm.kernels.speculative_verify import (
            greedy_chain_verify_into,
        )

        core = CoreKernelBindings(
            linear=linear,
            rms_norm=rms_norm,
            fused_add_rms_norm=fused_add_rms_norm,
            offset_rms_norm=offset_rms_norm,
            fused_add_offset_rms_norm=fused_add_offset_rms_norm,
            silu_and_mul=silu_and_mul,
            fused_qkv_norm_rope_cache_write=(
                fused_qkv_norm_rope_cache_write
            ),
            rope_cache_write=partial(
                dispatch_rope_cache_write,
                policy=policy,
            ),
            paged_decode_attention=paged_decode_attention,
            paged_prefill_attention=paged_prefill_attention,
            gdn_causal_conv_update=gdn.causal_conv_update,
            gdn_causal_conv_prefill=gdn_prefill.causal_conv_prefill,
            gdn_packed_recurrent_decode=gdn.packed_recurrent_decode,
            gdn_packed_recurrent_prefill=(
                gdn_prefill.packed_recurrent_prefill
            ),
            gdn_gated_rms_norm=gdn.gated_rms_norm,
            qwen35_active_kv_write=qwen35.active_kv_write,
        )
        moe = MoEKernelBindings(
            top1=moe_top1,
            top1_from_hidden=moe_top1_from_hidden,
            topk=moe_topk_softmax,
            routed_small=routed_moe_decode,
            routed_fallback=fallback.routed_moe_decode,
            routed_grouped=routed_moe_grouped_tensorcore,
            routed_grouped_from_hidden=_bind_triton_grouped_from_hidden(
                routed_moe_grouped_tensorcore
            ),
        )
        sampling = SamplingKernelBindings(
            use_triton=True,
            greedy_argmax=greedy_argmax,
            prepare_logits=prepare_sampling_logits,
            apply_repetition_penalty=apply_repetition_penalty_,
            apply_frequency_presence_penalty=(
                apply_frequency_presence_penalty_
            ),
            topk_cutoffs=topk_cutoffs,
            filtered_rejection_sample=filtered_rejection_sample,
        )
        speculative = SpeculativeKernelBindings(
            dflash_query_setup=dflash_query_setup_into,
            greedy_chain_verify=greedy_chain_verify_into,
        )
    else:
        core = CoreKernelBindings(
            linear=linear,
            rms_norm=fallback.rms_norm,
            fused_add_rms_norm=fallback.fused_add_rms_norm,
            offset_rms_norm=fallback.offset_rms_norm,
            fused_add_offset_rms_norm=(
                fallback.fused_add_offset_rms_norm
            ),
            silu_and_mul=fallback.silu_and_mul,
            fused_qkv_norm_rope_cache_write=(
                tensorplay_backend.fused_qkv_norm_rope_cache_write
            ),
            rope_cache_write=partial(
                dispatch_rope_cache_write,
                policy=policy,
            ),
            paged_decode_attention=fallback.paged_decode_attention,
            paged_prefill_attention=paged_prefill_attention,
            gdn_causal_conv_update=None,
            gdn_causal_conv_prefill=None,
            gdn_packed_recurrent_decode=None,
            gdn_packed_recurrent_prefill=None,
            gdn_gated_rms_norm=None,
            qwen35_active_kv_write=None,
        )
        moe = MoEKernelBindings(
            top1=tensorplay_backend.moe_top1,
            top1_from_hidden=tensorplay_backend.moe_top1_from_hidden,
            topk=fallback.moe_topk_softmax,
            routed_small=fallback.routed_moe_decode,
            routed_fallback=fallback.routed_moe_decode,
            routed_grouped=tensorplay_backend.routed_moe_grouped,
            routed_grouped_from_hidden=(
                tensorplay_backend.routed_moe_grouped_from_hidden
            ),
        )
        sampling = SamplingKernelBindings(
            use_triton=False,
            greedy_argmax=fallback.greedy_argmax,
        )
        from megalm.kernels import speculative_reference

        speculative = SpeculativeKernelBindings(
            dflash_query_setup=(
                speculative_reference.dflash_query_setup_into
            ),
            greedy_chain_verify=(
                speculative_reference.greedy_chain_verify_into
            ),
        )

    artifact_tuple = discovered.artifacts
    if attention_config is not None and attention_config.mode != "causal":
        from megalm.kernels.attention import build_attention_backend

        if attention_config.mode == "mla":
            raise KernelBindingError(
                "MLA attention requires checkpoint up-projection weights; "
                "bind MLAAttention directly until model wiring lands"
            )
        backend = build_attention_backend(
            attention_config,
            sink_logits=sink_logits,
        )
        core = replace(
            core,
            paged_prefill_attention=backend.prefill,
            paged_decode_attention=backend.decode,
        )
    return KernelBindings(
        policy=policy,
        linear_backend=discovered.linear_backend,
        prefill_backend=discovered.prefill_backend,
        core=core,
        moe=moe,
        sampling=sampling,
        speculative=speculative,
        artifacts=artifact_tuple,
        digest=discovered.digest,
    )


def _bind_triton_grouped_from_hidden(
    grouped_impl: KernelCallable,
) -> KernelCallable:
    def routed_from_hidden(
        hidden_states: tensorplay.Tensor,
        router_weight: tensorplay.Tensor,
        gate_up_weight: tensorplay.Tensor,
        down_weight: tensorplay.Tensor,
        *,
        normalize: bool,
        workspace: object | None = None,
    ) -> tensorplay.Tensor:
        return grouped_impl(
            hidden_states,
            gate_up_weight,
            down_weight,
            None,
            None,
            workspace=workspace,
            router_weight=router_weight,
            normalize_router=normalize,
        )

    return routed_from_hidden


__all__ = [
    "CoreKernelBindings",
    "KernelArtifact",
    "KernelBindingSpec",
    "KernelBindingError",
    "KernelBindings",
    "MoEKernelBindings",
    "SamplingKernelBindings",
    "SpeculativeKernelBindings",
    "bind_kernel_plan",
    "discover_kernel_plan",
]
