from __future__ import annotations

import hashlib
from dataclasses import replace

from megalm.contracts.model import AttentionKind, FeedForwardKind, ModelSpec
from megalm.contracts.plan import (
    CachePlan,
    CollectiveKind,
    CompiledPlan,
    CollectiveSite,
    ExecutionPhase,
    ExecutionPlan,
    HardwareSpec,
    KernelSelection,
    MemoryPlan,
    PlanContractError,
    ShapeGuard,
    TopologySpec,
    WeightPlan,
    WorkloadProfile,
)
from megalm.kernels.binding_types import (
    KernelArtifact,
    TENSORPLAY_CUBLAS_LINEAR_BACKEND,
    TENSORPLAY_SDPA_PAGED_PREFILL_BACKEND,
    TRITON_BATCH_INVARIANT_LINEAR_BACKEND,
    TRITON_PAGED_PREFILL_BACKEND,
)
from megalm.kernels.ops import KernelPolicy
from megalm.providers.registry import (
    DEFAULT_REGISTRY as HARDWARE_PROVIDERS,
    ProviderError,
)

_MIB = 1024**2
_DTYPE_BYTES = {"float16": 2, "bfloat16": 2, "float32": 4}


def build_execution_plan(
    *,
    model_spec: ModelSpec,
    hardware: HardwareSpec,
    resident_model_bytes: int,
    source_dtype: str,
    dtype: str,
    workload: WorkloadProfile,
    cache_bytes: int,
    graph_bytes: int,
    workspace_bytes: int,
    page_size: int,
    max_batched_tokens: int,
    cuda_graph_buckets: tuple[int, ...],
    kernel_policy: KernelPolicy,
    linear_backend: str | None = None,
    prefill_backend: str | None = None,
    kernel_artifacts: tuple[KernelArtifact, ...] = (),
    kernel_binding_digest: str = "",
    enable_prefix_cache: bool = True,
    topology: TopologySpec = TopologySpec(),
    compiled: CompiledPlan = CompiledPlan.disabled(),
) -> ExecutionPlan:
    """Freeze every backend, sharding and memory decision before serving."""

    use_triton = kernel_policy.use_triton
    if linear_backend is None:
        linear_backend = (
            TRITON_BATCH_INVARIANT_LINEAR_BACKEND
            if use_triton and model_spec.model_type == "qwen2_5_omni"
            else TENSORPLAY_CUBLAS_LINEAR_BACKEND
        )
    if linear_backend not in {
        TENSORPLAY_CUBLAS_LINEAR_BACKEND,
        TRITON_BATCH_INVARIANT_LINEAR_BACKEND,
    }:
        raise PlanContractError("unsupported linear backend")
    if (
        linear_backend == TRITON_BATCH_INVARIANT_LINEAR_BACKEND
        and not use_triton
    ):
        raise PlanContractError("Triton linear requires Triton kernels")
    if prefill_backend is None:
        prefill_backend = (
            TRITON_PAGED_PREFILL_BACKEND
            if use_triton
            else TENSORPLAY_SDPA_PAGED_PREFILL_BACKEND
        )
    if prefill_backend not in {
        TRITON_PAGED_PREFILL_BACKEND,
        TENSORPLAY_SDPA_PAGED_PREFILL_BACKEND,
        "megalm.attention.non_causal_sdpa",
        "megalm.attention.sink_sdpa",
        "megalm.attention.sparse_index_sdpa",
        "megalm.attention.mla_latent_prefill",
    }:
        raise PlanContractError("unsupported paged prefill backend")
    if prefill_backend == TRITON_PAGED_PREFILL_BACKEND and not use_triton:
        raise PlanContractError("Triton paged prefill requires Triton kernels")
    # The selected hardware provider must declare the dtype and both backend
    # ids. This is a cold check: an unsupported combination is rejected before
    # any kernel is imported or any device memory is allocated.
    try:
        HARDWARE_PROVIDERS.validate(
            accelerator=hardware.accelerator,
            dtype=dtype,
            backends=(linear_backend, prefill_backend),
        )
    except ProviderError as exc:
        raise PlanContractError(str(exc)) from exc
    triton_moe_max_tokens = kernel_policy.moe_max_tokens
    tensorcore_moe_min_tokens = kernel_policy.moe_tensorcore_min_tokens
    grouped_moe_min_tokens = kernel_policy.grouped_moe_min_tokens
    if dtype not in model_spec.supported_dtypes:
        raise PlanContractError(
            f"{dtype!r} is not supported by {model_spec.architecture}"
        )
    if hardware.compute_capability < model_spec.parallel.minimum_compute_capability:
        raise PlanContractError(
            f"compute capability {hardware.compute_capability} is below "
            f"{model_spec.parallel.minimum_compute_capability}"
        )
    if min(
        resident_model_bytes,
        cache_bytes,
        page_size,
        max_batched_tokens,
        triton_moe_max_tokens,
        tensorcore_moe_min_tokens,
        grouped_moe_min_tokens,
    ) <= 0:
        raise PlanContractError("execution-plan sizes must be positive")
    if graph_bytes < 0:
        raise PlanContractError("graph memory cannot be negative")
    if workspace_bytes < 0:
        raise PlanContractError("workspace memory cannot be negative")
    normalized_buckets = tuple(sorted(set(cuda_graph_buckets)))
    if normalized_buckets != cuda_graph_buckets:
        raise PlanContractError("graph buckets must be sorted and unique")
    if any(bucket > workload.max_batch_size for bucket in normalized_buckets):
        raise PlanContractError("graph bucket exceeds the maximum batch size")
    if normalized_buckets and not use_triton:
        raise PlanContractError("decode CUDA Graphs require fixed Triton kernels")
    if tensorcore_moe_min_tokens > workload.max_batch_size:
        tensorcore_moe_min_tokens = workload.max_batch_size + 1
    if grouped_moe_min_tokens < tensorcore_moe_min_tokens:
        raise PlanContractError(
            "grouped MoE guard cannot precede Tensor Core MoE"
        )
    decode_grouped_moe_min_tokens = grouped_moe_min_tokens
    if decode_grouped_moe_min_tokens > workload.max_batch_size:
        decode_grouped_moe_min_tokens = workload.max_batch_size + 1

    all_layers = tuple(layer.index for layer in model_spec.layers)
    full_attention_layers = tuple(
        layer.index
        for layer in model_spec.layers
        if layer.attention is AttentionKind.FULL
    )
    gdn_layers = tuple(
        layer.index
        for layer in model_spec.layers
        if layer.attention is AttentionKind.GATED_DELTA_NET
    )
    is_qwen35 = model_spec.model_type in {"qwen3_5", "qwen3_5_moe"}
    rope_kv_backend = (
        (
            "megalm.triton.fused_qk_norm_rope_kv_write"
            if use_triton
            else "megalm.tensorplay.qk_norm_rope_kv_write"
        )
        if model_spec.qk_norm
        else (
            "megalm.triton.rope_kv_write"
            if use_triton
            else "megalm.tensorplay.rope_kv_write"
        )
    )
    dense_layers = tuple(
        layer.index
        for layer in model_spec.layers
        if layer.feed_forward is FeedForwardKind.DENSE
    )
    moe_layers = tuple(
        layer.index
        for layer in model_spec.layers
        if layer.feed_forward is FeedForwardKind.MOE
    )
    tensor_parallel = topology.tensor_parallel
    if tensor_parallel > 1:
        _validate_tensor_parallel(
            model_spec,
            tensor_parallel,
            gdn_layers=gdn_layers,
            moe_layers=moe_layers,
        )
    top1_moe_layers = tuple(
        layer.index
        for layer in model_spec.layers
        if (
            layer.feed_forward is FeedForwardKind.MOE
            and layer.experts_per_token == 1
        )
    )
    small_top1_moe_layers = tuple(
        layer.index
        for layer in model_spec.layers
        if (
            layer.feed_forward is FeedForwardKind.MOE
            and layer.experts_per_token == 1
            and layer.num_experts <= 16
        )
    )
    large_top1_moe_layers = tuple(
        index
        for index in top1_moe_layers
        if index not in small_top1_moe_layers
    )
    compact_top1_router_layers = tuple(
        layer.index
        for layer in model_spec.layers
        if (
            layer.index in small_top1_moe_layers
            and model_spec.hidden_size * layer.num_experts <= 8192
        )
    )
    wide_top1_router_layers = tuple(
        index
        for index in small_top1_moe_layers
        if index not in compact_top1_router_layers
    )
    topk_moe_layers = tuple(
        layer.index
        for layer in model_spec.layers
        if (
            layer.feed_forward is FeedForwardKind.MOE
            and layer.experts_per_token != 1
        )
    )
    batch_guard = ShapeGuard(
        min_batch_size=1,
        max_batch_size=workload.max_batch_size,
        min_tokens=1,
        max_tokens=max_batched_tokens,
    )
    context_guard = ShapeGuard(
        min_batch_size=1,
        max_batch_size=workload.max_batch_size,
        min_tokens=1,
        max_tokens=workload.max_sequence_length,
    )
    kernels: list[KernelSelection] = [
        KernelSelection(
            operator="linear_and_embedding",
            phases=(ExecutionPhase.PREFILL, ExecutionPhase.DECODE),
            backend=linear_backend,
            layer_indices=all_layers,
            guard=batch_guard,
        ),
        KernelSelection(
            operator="rms_norm",
            phases=(ExecutionPhase.PREFILL, ExecutionPhase.DECODE),
            backend=(
                (
                    "megalm.triton.offset_rms_norm"
                    if use_triton
                    else "megalm.tensorplay.offset_rms_norm"
                )
                if is_qwen35
                else (
                "megalm.triton.rms_norm"
                if use_triton
                else "megalm.tensorplay.rms_norm"
                )
            ),
            layer_indices=all_layers,
            guard=batch_guard,
        ),
        KernelSelection(
            operator="qk_norm_rope_and_kv_write",
            phases=(ExecutionPhase.PREFILL, ExecutionPhase.DECODE),
            backend=rope_kv_backend,
            layer_indices=full_attention_layers,
            guard=batch_guard,
        ),
        KernelSelection(
            operator="prefill_attention",
            phases=(ExecutionPhase.PREFILL,),
            backend=prefill_backend,
            layer_indices=full_attention_layers,
            guard=context_guard,
        ),
        KernelSelection(
            operator="paged_decode_attention",
            phases=(ExecutionPhase.DECODE,),
            backend=_decode_backend(prefill_backend, use_triton, workload.max_sequence_length),
            layer_indices=full_attention_layers,
            guard=context_guard,
        ),
        KernelSelection(
            operator="greedy_sampling_single",
            phases=(ExecutionPhase.DECODE,),
            backend="tensorplay.argmax",
            layer_indices=(),
            guard=ShapeGuard(
                min_batch_size=1,
                max_batch_size=1,
                min_tokens=1,
                max_tokens=max_batched_tokens,
            ),
        ),
        KernelSelection(
            operator="sampling_logits_prepare",
            phases=(ExecutionPhase.DECODE,),
            backend=(
                "megalm.triton.temperature_repetition"
                if use_triton
                else "tensorplay.temperature_repetition"
            ),
            layer_indices=(),
            guard=batch_guard,
        ),
        KernelSelection(
            operator="filtered_sampling",
            phases=(ExecutionPhase.DECODE,),
            backend=(
                "megalm.triton.radix_rejection_sampling"
                if use_triton
                else "tensorplay.filtered_multinomial"
            ),
            layer_indices=(),
            guard=batch_guard,
        ),
    ]
    if gdn_layers:
        kernels.extend(
            (
                KernelSelection(
                    operator="gdn_input_projection",
                    phases=(ExecutionPhase.PREFILL, ExecutionPhase.DECODE),
                    backend="tensorplay.cublas.linear",
                    layer_indices=gdn_layers,
                    guard=batch_guard,
                ),
                KernelSelection(
                    operator="gdn_causal_conv_prefill",
                    phases=(ExecutionPhase.PREFILL,),
                    backend=(
                        "megalm.triton.packed_causal_conv_prefill"
                        if use_triton
                        else "tensorplay.grouped_conv1d"
                    ),
                    layer_indices=gdn_layers,
                    guard=context_guard,
                ),
                KernelSelection(
                    operator="gdn_causal_conv_decode",
                    phases=(ExecutionPhase.DECODE,),
                    backend=(
                        "megalm.triton.causal_conv_update"
                        if use_triton
                        else "megalm.tensorplay.vectorized_causal_conv_update"
                    ),
                    layer_indices=gdn_layers,
                    guard=batch_guard,
                ),
                KernelSelection(
                    operator="gdn_delta_rule_prefill",
                    phases=(ExecutionPhase.PREFILL,),
                    backend=(
                        "megalm.triton.packed_recurrent_delta_rule_prefill"
                        if use_triton
                        else "megalm.tensorplay.sequential_delta_rule_reference"
                    ),
                    layer_indices=gdn_layers,
                    guard=context_guard,
                ),
                KernelSelection(
                    operator="gdn_delta_rule_decode",
                    phases=(ExecutionPhase.DECODE,),
                    backend=(
                        "megalm.triton.packed_recurrent_delta_rule"
                        if use_triton
                        else "megalm.tensorplay.vectorized_delta_rule"
                    ),
                    layer_indices=gdn_layers,
                    guard=batch_guard,
                ),
                KernelSelection(
                    operator="gdn_gated_norm_output_prefill",
                    phases=(ExecutionPhase.PREFILL,),
                    backend=(
                        "megalm.triton.gated_rms_norm+tensorplay.cublas.linear"
                        if use_triton
                        else (
                            "megalm.tensorplay.gated_rms_norm+"
                            "tensorplay.cublas.linear"
                        )
                    ),
                    layer_indices=gdn_layers,
                    guard=context_guard,
                ),
                KernelSelection(
                    operator="gdn_gated_norm_output_decode",
                    phases=(ExecutionPhase.DECODE,),
                    backend=(
                        "megalm.triton.gated_rms_norm+tensorplay.cublas.linear"
                        if use_triton
                        else (
                            "megalm.tensorplay.gated_rms_norm+"
                            "tensorplay.cublas.linear"
                        )
                    ),
                    layer_indices=gdn_layers,
                    guard=batch_guard,
                ),
            )
        )
    if model_spec.vision is not None:
        kernels.extend(
            (
                KernelSelection(
                    operator="vision_patch_embedding",
                    phases=(ExecutionPhase.PREFILL,),
                    backend="tensorplay.cudnn.conv3d",
                    layer_indices=(),
                    guard=context_guard,
                ),
                KernelSelection(
                    operator="vision_attention",
                    phases=(ExecutionPhase.PREFILL,),
                    backend="tensorplay.sdpa.flash_attention",
                    layer_indices=(),
                    guard=context_guard,
                ),
                KernelSelection(
                    operator="vision_mlp_and_merger",
                    phases=(ExecutionPhase.PREFILL,),
                    backend="tensorplay.cublas.linear",
                    layer_indices=(),
                    guard=context_guard,
                ),
            )
        )
    if workload.max_batch_size > 1:
        kernels.append(
            KernelSelection(
                operator="greedy_sampling_batched",
                phases=(ExecutionPhase.DECODE,),
                backend=(
                    "megalm.triton.hierarchical_argmax"
                    if use_triton
                    else "tensorplay.argmax"
                ),
                layer_indices=(),
                guard=ShapeGuard(
                    min_batch_size=2,
                    max_batch_size=workload.max_batch_size,
                    min_tokens=1,
                    max_tokens=max_batched_tokens,
                ),
            )
        )
    if dense_layers:
        kernels.append(
            KernelSelection(
                operator="dense_silu_and_mul",
                phases=(ExecutionPhase.PREFILL, ExecutionPhase.DECODE),
                backend=(
                    "megalm.triton.silu_and_mul"
                    if use_triton
                    else "tensorplay.silu_and_mul"
                ),
                layer_indices=dense_layers,
                guard=batch_guard,
            )
        )
    if moe_layers:
        if top1_moe_layers:
            kernels.append(
                KernelSelection(
                    operator="moe_router_top1_prefill",
                    phases=(ExecutionPhase.PREFILL,),
                    backend=(
                        "tensorplay.cublas_gate+megalm.triton.top1"
                        if use_triton
                        else "tensorplay.softmax_topk"
                    ),
                    layer_indices=top1_moe_layers,
                    guard=batch_guard,
                )
            )
            if use_triton and small_top1_moe_layers:
                kernels.append(
                    KernelSelection(
                        operator="moe_router_top1_single",
                        phases=(ExecutionPhase.DECODE,),
                        backend="megalm.triton.fused_gate_projection_top1",
                        layer_indices=small_top1_moe_layers,
                        guard=ShapeGuard(
                            min_batch_size=1,
                            max_batch_size=1,
                            min_tokens=1,
                            max_tokens=1,
                        ),
                    )
                )
            b1_generic_layers = (
                large_top1_moe_layers if use_triton else top1_moe_layers
            )
            if b1_generic_layers:
                kernels.append(
                    KernelSelection(
                        operator="moe_router_top1_single",
                        phases=(ExecutionPhase.DECODE,),
                        backend=(
                            "tensorplay.cublas_gate+megalm.triton.top1"
                            if use_triton
                            else "tensorplay.softmax_topk"
                        ),
                        layer_indices=b1_generic_layers,
                        guard=ShapeGuard(
                            min_batch_size=1,
                            max_batch_size=1,
                            min_tokens=1,
                            max_tokens=1,
                        ),
                    )
                )
            if workload.max_batch_size > 1:
                if use_triton:
                    if compact_top1_router_layers:
                        kernels.append(
                            KernelSelection(
                                operator="moe_router_top1_batched",
                                phases=(ExecutionPhase.DECODE,),
                                backend=(
                                    "megalm.triton."
                                    "fused_gate_projection_top1"
                                ),
                                layer_indices=compact_top1_router_layers,
                                guard=ShapeGuard(
                                    min_batch_size=2,
                                    max_batch_size=workload.max_batch_size,
                                    min_tokens=2,
                                    max_tokens=workload.max_batch_size,
                                ),
                            )
                        )
                    fused_wide_limit = min(32, workload.max_batch_size)
                    if wide_top1_router_layers and fused_wide_limit >= 2:
                        kernels.append(
                            KernelSelection(
                                operator="moe_router_top1_batched",
                                phases=(ExecutionPhase.DECODE,),
                                backend=(
                                    "megalm.triton."
                                    "fused_gate_projection_top1"
                                ),
                                layer_indices=wide_top1_router_layers,
                                guard=ShapeGuard(
                                    min_batch_size=2,
                                    max_batch_size=fused_wide_limit,
                                    min_tokens=2,
                                    max_tokens=fused_wide_limit,
                                ),
                            )
                        )
                    if (
                        wide_top1_router_layers
                        and workload.max_batch_size > 32
                    ):
                        kernels.append(
                            KernelSelection(
                                operator="moe_router_top1_batched",
                                phases=(ExecutionPhase.DECODE,),
                                backend=(
                                    "tensorplay.cublas_gate+"
                                    "megalm.triton.top1"
                                ),
                                layer_indices=wide_top1_router_layers,
                                guard=ShapeGuard(
                                    min_batch_size=33,
                                    max_batch_size=workload.max_batch_size,
                                    min_tokens=33,
                                    max_tokens=workload.max_batch_size,
                                ),
                            )
                        )
                    if large_top1_moe_layers:
                        kernels.append(
                            KernelSelection(
                                operator="moe_router_top1_batched",
                                phases=(ExecutionPhase.DECODE,),
                                backend=(
                                    "tensorplay.cublas_gate+"
                                    "megalm.triton.top1"
                                ),
                                layer_indices=large_top1_moe_layers,
                                guard=ShapeGuard(
                                    min_batch_size=2,
                                    max_batch_size=workload.max_batch_size,
                                    min_tokens=2,
                                    max_tokens=workload.max_batch_size,
                                ),
                            )
                        )
                else:
                    kernels.append(
                        KernelSelection(
                            operator="moe_router_top1_batched",
                            phases=(ExecutionPhase.DECODE,),
                            backend="tensorplay.softmax_topk",
                            layer_indices=top1_moe_layers,
                            guard=ShapeGuard(
                                min_batch_size=2,
                                max_batch_size=workload.max_batch_size,
                                min_tokens=2,
                                max_tokens=workload.max_batch_size,
                            ),
                        )
                    )
        if topk_moe_layers:
            kernels.append(
                KernelSelection(
                    operator="moe_router_topk",
                    phases=(ExecutionPhase.PREFILL, ExecutionPhase.DECODE),
                    backend=(
                        "tensorplay.cublas_gate+megalm.triton.topk"
                        if use_triton
                        else "tensorplay.softmax_topk"
                    ),
                    layer_indices=topk_moe_layers,
                    guard=batch_guard,
                )
            )
        triton_token_limit = min(
            triton_moe_max_tokens,
            max_batched_tokens,
        )
        small_decode_limit = min(
            triton_token_limit,
            tensorcore_moe_min_tokens - 1,
        )
        if small_decode_limit >= 1:
            kernels.append(
                KernelSelection(
                    operator="routed_expert_gemv_small",
                    phases=(ExecutionPhase.DECODE,),
                    backend=(
                        "megalm.triton.routed_moe_fixed_tiles"
                        if use_triton
                        else "tensorplay.expert_linear_loop"
                    ),
                    layer_indices=moe_layers,
                    guard=ShapeGuard(
                        min_batch_size=1,
                        max_batch_size=small_decode_limit,
                        min_tokens=1,
                        max_tokens=small_decode_limit,
                    ),
                )
            )
        grouped_enabled = (
            use_triton
            and bool(moe_layers)
            and decode_grouped_moe_min_tokens <= workload.max_batch_size
        )
        top1_tensorcore_limit = (
            decode_grouped_moe_min_tokens - 1
            if grouped_enabled
            else workload.max_batch_size
        )
        if (
            top1_moe_layers
            and tensorcore_moe_min_tokens <= top1_tensorcore_limit
        ):
            kernels.append(
                KernelSelection(
                    operator="routed_expert_tensorcore_decode",
                    phases=(ExecutionPhase.DECODE,),
                    backend="tensorplay.cublas.all_expert_gemm_bmm",
                    layer_indices=top1_moe_layers,
                    guard=ShapeGuard(
                        min_batch_size=tensorcore_moe_min_tokens,
                        max_batch_size=top1_tensorcore_limit,
                        min_tokens=tensorcore_moe_min_tokens,
                        max_tokens=top1_tensorcore_limit,
                    ),
                )
            )
        topk_grouped_min = (
            decode_grouped_moe_min_tokens
            if grouped_enabled
            else workload.max_batch_size + 1
        )
        if topk_moe_layers and small_decode_limit + 1 < topk_grouped_min:
            kernels.append(
                KernelSelection(
                    operator="routed_expert_gemv_decode_topk",
                    phases=(ExecutionPhase.DECODE,),
                    backend=(
                        "megalm.triton.routed_moe"
                        if use_triton
                        else "tensorplay.expert_linear_loop"
                    ),
                    layer_indices=topk_moe_layers,
                    guard=ShapeGuard(
                        min_batch_size=small_decode_limit + 1,
                        max_batch_size=topk_grouped_min - 1,
                        min_tokens=small_decode_limit + 1,
                        max_tokens=topk_grouped_min - 1,
                    ),
                )
            )
        if grouped_enabled:
            kernels.append(
                KernelSelection(
                    operator="routed_expert_grouped_decode",
                    phases=(ExecutionPhase.DECODE,),
                    backend="megalm.triton.grouped_tensorcore_moe",
                    layer_indices=top1_moe_layers,
                    guard=ShapeGuard(
                        min_batch_size=decode_grouped_moe_min_tokens,
                        max_batch_size=workload.max_batch_size,
                        min_tokens=decode_grouped_moe_min_tokens,
                        max_tokens=workload.max_batch_size,
                    ),
                )
            )
            if topk_moe_layers:
                kernels.append(
                    KernelSelection(
                        operator="routed_expert_grouped_decode_topk",
                        phases=(ExecutionPhase.DECODE,),
                        backend="megalm.triton.grouped_tensorcore_moe",
                        layer_indices=topk_moe_layers,
                        guard=ShapeGuard(
                            min_batch_size=decode_grouped_moe_min_tokens,
                            max_batch_size=workload.max_batch_size,
                            min_tokens=decode_grouped_moe_min_tokens,
                            max_tokens=workload.max_batch_size,
                        ),
                    )
                )
        top1_small_limit = min(
            max_batched_tokens,
            grouped_moe_min_tokens - 1,
        )
        top1_triton_limit = min(top1_small_limit, triton_token_limit)
        if top1_moe_layers and top1_triton_limit >= 1:
            kernels.append(
                KernelSelection(
                    operator="routed_expert_gemv_prefill_top1_small",
                    phases=(ExecutionPhase.PREFILL,),
                    backend=(
                        "megalm.triton.routed_moe"
                        if use_triton
                        else "tensorplay.expert_linear_loop"
                    ),
                    layer_indices=top1_moe_layers,
                    guard=ShapeGuard(
                        min_batch_size=1,
                        max_batch_size=workload.max_batch_size,
                        min_tokens=1,
                        max_tokens=top1_triton_limit,
                    ),
                )
            )
        if (
            top1_moe_layers
            and top1_triton_limit < top1_small_limit
        ):
            kernels.append(
                KernelSelection(
                    operator="routed_expert_gemm_prefill_top1_fallback",
                    phases=(ExecutionPhase.PREFILL,),
                    backend="tensorplay.expert_linear_loop",
                    layer_indices=top1_moe_layers,
                    guard=ShapeGuard(
                        min_batch_size=1,
                        max_batch_size=workload.max_batch_size,
                        min_tokens=top1_triton_limit + 1,
                        max_tokens=top1_small_limit,
                    ),
                )
            )
        if (
            top1_moe_layers
            and grouped_moe_min_tokens <= max_batched_tokens
        ):
            kernels.append(
                KernelSelection(
                    operator="routed_expert_grouped_prefill",
                    phases=(ExecutionPhase.PREFILL,),
                    backend=(
                        "megalm.triton.compact_grouped_tensorcore_moe"
                        if use_triton
                        else "tensorplay.cublas.all_expert_gemm_bmm"
                    ),
                    layer_indices=top1_moe_layers,
                    guard=ShapeGuard(
                        min_batch_size=1,
                        max_batch_size=workload.max_batch_size,
                        min_tokens=grouped_moe_min_tokens,
                        max_tokens=max_batched_tokens,
                    ),
                )
            )
        if topk_moe_layers:
            kernels.append(
                KernelSelection(
                    operator="routed_expert_gemm_prefill_topk_small",
                    phases=(ExecutionPhase.PREFILL,),
                    backend=(
                        "megalm.triton.routed_moe"
                        if use_triton
                        else "tensorplay.expert_linear_loop"
                    ),
                    layer_indices=topk_moe_layers,
                    guard=ShapeGuard(
                        min_batch_size=1,
                        max_batch_size=workload.max_batch_size,
                        min_tokens=1,
                        max_tokens=triton_token_limit,
                    ),
                )
            )
        topk_fallback_limit = (
            min(max_batched_tokens, grouped_moe_min_tokens - 1)
            if use_triton
            else max_batched_tokens
        )
        if topk_moe_layers and triton_token_limit < topk_fallback_limit:
            kernels.append(
                KernelSelection(
                    operator="routed_expert_gemm_prefill_topk_large",
                    phases=(ExecutionPhase.PREFILL,),
                    backend="tensorplay.expert_linear_loop",
                    layer_indices=topk_moe_layers,
                    guard=ShapeGuard(
                        min_batch_size=1,
                        max_batch_size=workload.max_batch_size,
                        min_tokens=triton_token_limit + 1,
                        max_tokens=topk_fallback_limit,
                    ),
                )
            )
        if (
            use_triton
            and topk_moe_layers
            and grouped_moe_min_tokens <= max_batched_tokens
        ):
            kernels.append(
                KernelSelection(
                    operator="routed_expert_grouped_prefill_topk",
                    phases=(ExecutionPhase.PREFILL,),
                    backend="megalm.triton.grouped_tensorcore_moe",
                    layer_indices=topk_moe_layers,
                    guard=ShapeGuard(
                        min_batch_size=1,
                        max_batch_size=workload.max_batch_size,
                        min_tokens=grouped_moe_min_tokens,
                        max_tokens=max_batched_tokens,
                    ),
                )
            )

    reserved_bytes = max(hardware.total_memory_bytes // 20, 512 * _MIB)
    tensor_parallel = topology.tensor_parallel
    per_rank_weights = -(-resident_model_bytes // tensor_parallel)
    per_rank_cache = cache_bytes // tensor_parallel
    collective_workspace_bytes = 0
    if tensor_parallel > 1:
        first_attention = next(
            layer
            for layer in model_spec.layers
            if layer.attention is AttentionKind.FULL
        )
        hidden_size = first_attention.head_dim * first_attention.num_attention_heads
        message_bytes = (
            hidden_size
            * max_batched_tokens
            * _DTYPE_BYTES.get(dtype, 4)
        )
        # Send and receive staging buffers for one in-flight all-reduce.
        collective_workspace_bytes = 2 * message_bytes
        committed = (
            per_rank_weights
            + graph_bytes
            + workspace_bytes
            + collective_workspace_bytes
            + per_rank_cache
            + reserved_bytes
        )
        if committed > hardware.total_memory_bytes:
            raise PlanContractError(
                "sharded weights, graph pool, workspace, KV cache and "
                "safety reserve do not fit the GPU"
            )
        collectives = (
            CollectiveSite(
                after_operator="linear_and_embedding",
                kind=CollectiveKind.ALL_REDUCE,
                ranks=tuple(range(topology.world_size)),
            ),
        )
    else:
        collectives = ()
        committed = (
            resident_model_bytes
            + graph_bytes
            + workspace_bytes
            + cache_bytes
            + reserved_bytes
        )
        if committed > hardware.total_memory_bytes:
            raise PlanContractError(
                "weights, graph pool, workspace, KV cache and safety reserve "
                "do not fit the GPU"
            )
    memory = MemoryPlan(
        total_bytes=hardware.total_memory_bytes,
        weights_bytes=per_rank_weights,
        graph_bytes=graph_bytes,
        workspace_bytes=workspace_bytes + collective_workspace_bytes,
        elastic_cache_bytes=per_rank_cache,
        reserved_bytes=reserved_bytes,
    )
    prefix_cache_enabled = enable_prefix_cache and not gdn_layers
    cache = CachePlan(
        state_kinds=tuple(
            component.kind for component in model_spec.state_schema
        ),
        kv_page_size_tokens=page_size,
        checkpoint_alignment_tokens=page_size,
        capacity_bytes=per_rank_cache,
        prefix_backend=(
            "megalm.radix.page_aligned.v1"
            if prefix_cache_enabled
            else "disabled"
        ),
        copy_on_write=prefix_cache_enabled,
    )
    native_weight_layout = (
        "qwen3_5.native-packed.v1"
        if is_qwen35
        else "qwen3.native-packed.v1"
    )
    kernels = [
        replace(kernel, weight_layout=native_weight_layout)
        for kernel in kernels
    ]
    materialized_kernels = _materialize_kernel_resources(
        tuple(kernels),
        kernel_artifacts=kernel_artifacts,
        workspace_bytes=workspace_bytes,
        weight_layout=native_weight_layout,
    )
    materialized_kernels = _declare_capabilities(materialized_kernels)
    return ExecutionPlan.create(
        model_digest=model_spec.digest,
        hardware=hardware,
        topology=topology,
        dtype=dtype,
        workload=workload,
        weights=WeightPlan(
            source_format="safetensors",
            source_dtype=source_dtype,
            target_dtype=dtype,
            loader="megalm.safetensors.streaming.v1",
            estimated_bytes=per_rank_weights,
            tensor_parallel=tensor_parallel,
            packing_rules=(
                "attention_qkv_contiguous",
                "dense_gate_up_contiguous",
                "expert_gate_up_contiguous",
                "expert_down_contiguous",
                *(("tensor_parallel_sharded.v1",) if tensor_parallel > 1 else ()),
            ),
        ),
        kernels=materialized_kernels,
        memory=memory,
        cache=cache,
        compiled=compiled,
        kernel_binding_digest=kernel_binding_digest,
        collectives=collectives,
        cuda_graph_buckets=normalized_buckets,
        vision_backend=(
            "megalm.qwen3_5.native_vision.v1"
            if model_spec.vision is not None
            else None
        ),
    )


def _validate_tensor_parallel(
    model_spec: ModelSpec,
    tensor_parallel: int,
    *,
    gdn_layers: tuple[int, ...],
    moe_layers: tuple[int, ...],
) -> None:
    """Check that every sharded dimension divides by the tensor degree.

    Megatron-style sharding requires the attention head count, the KV head
    count and the dense MLP width to divide evenly. Gated DeltaNet, MoE
    expert grids and the vision tower use disjoint sharding semantics
    (state/expert/tensor) and stay out of scope for this plan version.
    """
    if tensor_parallel <= 0:
        raise PlanContractError("tensor parallel degree must be positive")
    if gdn_layers:
        raise PlanContractError(
            "Gated DeltaNet layers do not support tensor parallelism yet"
        )
    if moe_layers:
        raise PlanContractError(
            "MoE layers do not support tensor parallelism yet"
        )
    if model_spec.vision is not None:
        raise PlanContractError(
            "vision towers do not support tensor parallelism yet"
        )
    for layer in model_spec.layers:
        if layer.num_attention_heads % tensor_parallel:
            raise PlanContractError(
                f"layer {layer.index}: {layer.num_attention_heads} attention "
                f"heads do not divide by TP={tensor_parallel}"
            )
        if layer.num_key_value_heads % tensor_parallel:
            raise PlanContractError(
                f"layer {layer.index}: {layer.num_key_value_heads} KV heads "
                f"do not divide by TP={tensor_parallel}"
            )
        if layer.intermediate_size % tensor_parallel:
            raise PlanContractError(
                f"layer {layer.index}: intermediate size "
                f"{layer.intermediate_size} does not divide by "
                f"TP={tensor_parallel}"
            )


def _decode_backend(
    prefill_backend: str | None,
    use_triton: bool,
    max_sequence_length: int,
) -> str:
    """Fix the decode backend id cold: typed backend or the hand-written path."""

    if prefill_backend in {
        "megalm.attention.non_causal_sdpa",
        "megalm.attention.sink_sdpa",
        "megalm.attention.sparse_index_sdpa",
    }:
        return prefill_backend
    if prefill_backend == "megalm.attention.mla_latent_prefill":
        return "megalm.attention.mla_latent_decode"
    if use_triton:
        return (
            "megalm.triton.single_pass_paged_online_softmax"
            if max_sequence_length <= 512
            else "megalm.triton.split_kv_online_softmax"
        )
    return "megalm.tensorplay.paged_decode"


def _declare_capabilities(
    kernels: tuple[KernelSelection, ...],
) -> tuple[KernelSelection, ...]:
    """Attach each backend's declared capability set from the cold registry.

    The matrix is a set of per-backend guard declarations, not a priority
    table. Unknown backends keep an empty set: they are existing providers
    whose attention capabilities are not part of the MLA/sparse/sink matrix.
    """

    from megalm.kernels.capabilities import CapabilityError, DEFAULT_MATRIX

    declared: list[KernelSelection] = []
    for kernel in kernels:
        capabilities: tuple[str, ...] = ()
        for phase in kernel.phases:
            try:
                entry = DEFAULT_MATRIX.entry(
                    kernel.operator, phase.value, kernel.backend
                )
            except CapabilityError:
                continue
            capabilities = tuple(sorted(set(capabilities) | set(entry.capabilities)))
        declared.append(replace(kernel, capabilities=capabilities))
    return tuple(declared)


def _materialize_kernel_resources(
    kernels: tuple[KernelSelection, ...],
    *,
    kernel_artifacts: tuple[KernelArtifact, ...],
    workspace_bytes: int,
    weight_layout: str,
) -> tuple[KernelSelection, ...]:
    if not kernel_artifacts:
        return kernels
    artifacts = {artifact.provider: artifact for artifact in kernel_artifacts}
    if len(artifacts) != len(kernel_artifacts):
        raise PlanContractError("kernel artifact providers must be unique")

    materialized: list[KernelSelection] = []
    for kernel in kernels:
        providers = (
            ("tensorplay", "triton")
            if "triton" in kernel.backend and "tensorplay" in kernel.backend
            else (
                ("triton",)
                if "triton" in kernel.backend
                else ("tensorplay",)
            )
        )
        try:
            selected = tuple(artifacts[item] for item in providers)
        except KeyError as exc:
            raise PlanContractError(
                f"missing bound artifact for provider {exc.args[0]!r}"
            ) from exc
        abi_versions = {artifact.abi_version for artifact in selected}
        if len(abi_versions) != 1:
            raise PlanContractError(
                f"kernel ABI mismatch for {kernel.backend!r}"
            )
        artifact_hash = hashlib.sha256()
        artifact_hash.update(kernel.backend.encode("utf-8"))
        for artifact in selected:
            artifact_hash.update(b"\0")
            artifact_hash.update(artifact.digest.encode("ascii"))
        owns_workspace = (
            kernel.backend.startswith("megalm.triton.")
            and kernel.operator
            in {
                "routed_expert_grouped_decode",
                "routed_expert_grouped_decode_topk",
                "routed_expert_grouped_prefill",
                "routed_expert_grouped_prefill_topk",
            }
        )
        materialized.append(
            replace(
                kernel,
                variant="+".join(
                    artifact.variant for artifact in selected
                ),
                abi_version=abi_versions.pop(),
                artifact_digest=artifact_hash.hexdigest(),
                weight_layout=weight_layout,
                workspace=(
                    "moe.shared.top1.v1" if owns_workspace else "none"
                ),
                workspace_bytes=workspace_bytes if owns_workspace else 0,
            )
        )
    return tuple(materialized)


__all__ = ["build_execution_plan"]
