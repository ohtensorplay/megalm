from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
import logging
import os
import time

import tensorplay

from megalm.contracts.model import AttentionKind, FeedForwardKind, ModelSpec
from megalm.contracts.plan import (
    CompiledPlan,
    ExecutionPlan,
    HardwareSpec,
    KernelSelection,
    PlanContractError,
    TopologySpec,
    WorkloadProfile,
)
from megalm.kernels.bindings import KernelBindingSpec
from megalm.kernels.moe_grouped_int4 import Int4MoEWorkspace
from megalm.kernels.ops import GroupedMoEWorkspace
from megalm.observability import log_event
from megalm.planning import build_execution_plan

_LOGGER = logging.getLogger("megalm.startup")


class StartupPhase(str, Enum):
    DISCOVER = "discover"
    DRAFT = "draft"
    LOAD_PACK = "load_pack"
    ALLOCATE_BIND = "allocate_bind"
    WARM_CAPTURE = "warm_capture"
    SEAL = "seal"
    READY = "ready"


_PHASE_ORDER = tuple(StartupPhase)


@dataclass(slots=True)
class StartupTracker:
    model_path: str
    phase: StartupPhase = StartupPhase.DISCOVER
    _started_ns: int = field(init=False, repr=False)
    _phase_started_ns: int = field(init=False, repr=False)

    def __post_init__(self) -> None:
        now_ns = time.perf_counter_ns()
        self._started_ns = now_ns
        self._phase_started_ns = now_ns
        log_event(
            _LOGGER,
            logging.INFO,
            "startup_phase",
            phase=self.phase.value,
            phase_index=1,
            phase_total=len(_PHASE_ORDER),
            elapsed_ms=0.0,
            model_path=self.model_path,
        )

    def advance(
        self,
        phase: StartupPhase,
        **fields: object,
    ) -> None:
        current_index = _PHASE_ORDER.index(self.phase)
        if (
            current_index + 1 >= len(_PHASE_ORDER)
            or _PHASE_ORDER[current_index + 1] is not phase
        ):
            raise PlanContractError(
                f"invalid startup transition {self.phase.value}->{phase.value}"
            )
        now_ns = time.perf_counter_ns()
        previous_phase = self.phase
        self.phase = phase
        log_event(
            _LOGGER,
            logging.INFO,
            "startup_phase",
            phase=phase.value,
            phase_index=current_index + 2,
            phase_total=len(_PHASE_ORDER),
            previous_phase=previous_phase.value,
            previous_phase_duration_ms=(
                now_ns - self._phase_started_ns
            ) / 1_000_000,
            elapsed_ms=(now_ns - self._started_ns) / 1_000_000,
            model_path=self.model_path,
            **fields,
        )
        self._phase_started_ns = now_ns


@dataclass(frozen=True, slots=True)
class PlanDraft:
    """Immutable kernel and resource decisions made before weight loading."""

    model_digest: str
    hardware: HardwareSpec
    source_dtype: str
    dtype: str
    workload: WorkloadProfile
    estimated_model_bytes: int
    estimated_workspace_bytes: int
    page_size: int
    max_batched_tokens: int
    cuda_graph_buckets: tuple[int, ...]
    kernels: tuple[KernelSelection, ...]
    kernel_binding_digest: str
    topology: TopologySpec = TopologySpec()
    compiled: CompiledPlan = CompiledPlan.disabled()

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "cuda_graph_buckets",
            tuple(self.cuda_graph_buckets),
        )
        object.__setattr__(self, "kernels", tuple(self.kernels))
        if len(self.model_digest) != 64:
            raise PlanContractError("draft model digest is invalid")
        if min(
            self.estimated_model_bytes,
            self.page_size,
            self.max_batched_tokens,
        ) <= 0:
            raise PlanContractError("draft resource sizes must be positive")
        if self.estimated_workspace_bytes < 0:
            raise PlanContractError("draft workspace size cannot be negative")
        if not self.kernels or len(self.kernel_binding_digest) != 64:
            raise PlanContractError("draft kernel identity is incomplete")

    def verify_sealed(self, plan: ExecutionPlan) -> None:
        """Reject any provider, guard or resource drift during initialization."""

        checks = (
            (plan.model_digest, self.model_digest, "model"),
            (plan.hardware, self.hardware, "hardware"),
            (plan.topology, self.topology, "topology"),
            (plan.dtype, self.dtype, "dtype"),
            (plan.workload, self.workload, "workload"),
            (
                plan.weights.source_dtype,
                self.source_dtype,
                "source dtype",
            ),
            (plan.kernels, self.kernels, "kernel selection"),
            (
                plan.kernel_binding_digest,
                self.kernel_binding_digest,
                "kernel binding",
            ),
        )
        for actual, expected, name in checks:
            if actual != expected:
                raise PlanContractError(
                    f"{name} changed between PlanDraft and seal"
                )
        expected_workspace = (
            self.estimated_workspace_bytes + plan.compiled.artifact_bytes
        )
        if plan.memory.workspace_bytes != expected_workspace:
            raise PlanContractError("workspace changed between PlanDraft and seal")
        drafted_compile = self.compiled
        sealed_compile = plan.compiled
        if (
            sealed_compile.backend != drafted_compile.backend
            or sealed_compile.digest != drafted_compile.digest
        ):
            raise PlanContractError("compiled artifact changed between PlanDraft and seal")
        if not set(sealed_compile.regions) <= set(drafted_compile.regions):
            raise PlanContractError("sealed compiled regions exceed the draft")
        if not set(sealed_compile.buckets) <= set(drafted_compile.buckets):
            raise PlanContractError("sealed compiled buckets exceed the draft")
        if sealed_compile.artifact_bytes > drafted_compile.max_artifact_bytes:
            raise PlanContractError("compiled artifacts exceed the drafted budget")
        # Capturing fewer buckets than drafted is a legitimate degradation
        # (capability opt-out or failed capture in non-strict mode); only
        # expanding beyond the draft violates the plan.
        if not set(plan.cuda_graph_buckets) <= set(self.cuda_graph_buckets):
            raise PlanContractError(
                "sealed CUDA Graph buckets exceed the drafted buckets"
            )


def build_plan_draft(
    *,
    model_spec: ModelSpec,
    hardware: HardwareSpec,
    source_dtype: str,
    dtype: str,
    workload: WorkloadProfile,
    estimated_model_bytes: int,
    page_size: int,
    max_batched_tokens: int,
    cuda_graph_buckets: tuple[int, ...],
    binding_spec: KernelBindingSpec,
    enable_prefix_cache: bool,
    topology: TopologySpec = TopologySpec(),
    compiled: CompiledPlan = CompiledPlan.disabled(),
) -> PlanDraft:
    """Select exact backends using manifest/config facts only."""

    tensorplay_dtype = {
        "float16": tensorplay.float16,
        "bfloat16": tensorplay.bfloat16,
    }.get(dtype)
    if tensorplay_dtype is None:
        raise PlanContractError(f"unsupported draft dtype {dtype!r}")
    workspace_bytes = estimate_moe_workspace_bytes(
        model_spec,
        policy_use_triton=binding_spec.policy.use_triton,
        max_batched_tokens=max_batched_tokens,
        dtype=tensorplay_dtype,
    )
    page_bytes = estimate_kv_page_bytes(
        model_spec,
        page_size=page_size,
        dtype=tensorplay_dtype,
    )
    fixed_state_bytes = estimate_gdn_state_bytes(
        model_spec,
        num_slots=workload.max_batch_size,
        dtype=tensorplay_dtype,
    )
    provisional = build_execution_plan(
        model_spec=model_spec,
        hardware=hardware,
        resident_model_bytes=estimated_model_bytes,
        source_dtype=source_dtype,
        dtype=dtype,
        workload=workload,
        cache_bytes=page_bytes + fixed_state_bytes,
        graph_bytes=0,
        workspace_bytes=workspace_bytes,
        page_size=page_size,
        max_batched_tokens=max_batched_tokens,
        cuda_graph_buckets=cuda_graph_buckets,
        kernel_policy=binding_spec.policy,
        linear_backend=binding_spec.linear_backend,
        prefill_backend=binding_spec.prefill_backend,
        kernel_artifacts=binding_spec.artifacts,
        kernel_binding_digest=binding_spec.digest,
        enable_prefix_cache=enable_prefix_cache,
        topology=topology,
        compiled=compiled,
    )
    return PlanDraft(
        model_digest=model_spec.digest,
        hardware=hardware,
        source_dtype=source_dtype,
        dtype=dtype,
        workload=workload,
        estimated_model_bytes=estimated_model_bytes,
        estimated_workspace_bytes=provisional.memory.workspace_bytes,
        page_size=page_size,
        max_batched_tokens=max_batched_tokens,
        cuda_graph_buckets=cuda_graph_buckets,
        kernels=provisional.kernels,
        kernel_binding_digest=provisional.kernel_binding_digest,
        topology=topology,
        compiled=compiled,
    )


def estimate_moe_workspace_bytes(
    model_spec: ModelSpec,
    *,
    policy_use_triton: bool,
    max_batched_tokens: int,
    dtype: tensorplay.dtype,
) -> int:
    # Omni keeps its own MoE state and owns no grouped arena.
    if model_spec.model_type == "qwen2_5_omni":
        return 0
    # Inkling-2 uses the int4 grouped kernel with its own reusable arena.
    if model_spec.model_type in {"inkling2", "inkling2_loop"}:
        if (
            not policy_use_triton
            or os.environ.get("MEGALM_INT4_EXPERTS", "1") == "0"
        ):
            return 0
        moe_layers = tuple(
            layer
            for layer in model_spec.layers
            if layer.feed_forward is FeedForwardKind.MOE
        )
        if not moe_layers:
            return 0
        layer = moe_layers[0]
        return Int4MoEWorkspace.estimate_nbytes(
            max_tokens=max_batched_tokens,
            num_experts=layer.num_experts,
            hidden_size=layer.hidden_size,
            out_width13=2 * layer.intermediate_size,
            top_k=layer.experts_per_token,
            dtype=dtype,
        )
    moe_layers = tuple(
        layer
        for layer in model_spec.layers
        if layer.feed_forward is FeedForwardKind.MOE
    )
    if not policy_use_triton or not moe_layers:
        return 0
    geometry = (
        moe_layers[0].num_experts,
        moe_layers[0].hidden_size,
        moe_layers[0].intermediate_size,
    )
    if any(
        (
            layer.num_experts,
            layer.hidden_size,
            layer.intermediate_size,
        )
        != geometry
        for layer in moe_layers
    ):
        raise PlanContractError(
            "shared MoE workspace requires one expert geometry"
        )
    num_experts, hidden_size, intermediate_size = geometry
    return GroupedMoEWorkspace.estimate_nbytes(
        max_tokens=max_batched_tokens,
        num_experts=num_experts,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        dtype=dtype,
        max_top_k=max(layer.experts_per_token for layer in moe_layers),
    )


def kv_cache_layer_count(model_spec: ModelSpec) -> int:
    """Physical KV cache layers: one per attention layer, or one per loop visit."""

    full = [layer for layer in model_spec.layers if layer.attention is AttentionKind.FULL]
    if model_spec.loop is not None:
        # Any looped model replays a core with shared weights; each visit owns
        # its own KV cache layer (Inkling-2 and Nanbeige4.2 alike).
        loop = model_spec.loop
        core_layers = loop.core_end - loop.core_start
        return model_spec.num_layers + (loop.num_passes - 1) * core_layers
    return len(full)


def estimate_kv_page_bytes(
    model_spec: ModelSpec,
    *,
    page_size: int,
    dtype: tensorplay.dtype,
) -> int:
    element_size = tensorplay.empty((), dtype=dtype).element_size()
    per_layer = sum(
        2
        * page_size
        * layer.num_key_value_heads
        * layer.head_dim
        * element_size
        for layer in model_spec.layers
        if layer.attention is AttentionKind.FULL
    )
    full = [
        layer
        for layer in model_spec.layers
        if layer.attention is AttentionKind.FULL
    ]
    if not full:
        return 0
    if model_spec.loop is not None:
        # Every visit of a replayed core reuses the same geometry.
        geometries = {(layer.num_key_value_heads, layer.head_dim) for layer in full}
        if len(geometries) != 1:
            raise PlanContractError(
                "a looped model requires one uniform KV geometry"
            )
        return per_layer * kv_cache_layer_count(model_spec) // max(len(full), 1)
    return per_layer


def estimate_gdn_state_bytes(
    model_spec: ModelSpec,
    *,
    num_slots: int,
    dtype: tensorplay.dtype,
) -> int:
    """Return the fixed-address conv and FP32 recurrent arenas."""

    if num_slots <= 0:
        raise PlanContractError("GDN state slots must be positive")
    element_size = tensorplay.empty((), dtype=dtype).element_size()
    total = 0
    for layer in model_spec.layers:
        if layer.attention is not AttentionKind.GATED_DELTA_NET:
            continue
        spec = layer.gated_delta_net
        if spec is None:
            raise PlanContractError(
                f"GDN layer {layer.index} has no recurrent geometry"
            )
        key_width = spec.num_key_heads * spec.key_head_dim
        value_width = spec.num_value_heads * spec.value_head_dim
        conv_width = 2 * key_width + value_width
        total += (
            num_slots
            * conv_width
            * (spec.conv_kernel_dim - 1)
            * element_size
        )
        total += (
            num_slots
            * spec.num_value_heads
            * spec.key_head_dim
            * spec.value_head_dim
            * tensorplay.empty((), dtype=tensorplay.float32).element_size()
        )
    return total


__all__ = [
    "PlanDraft",
    "StartupPhase",
    "StartupTracker",
    "build_plan_draft",
    "estimate_gdn_state_bytes",
    "estimate_kv_page_bytes",
    "estimate_moe_workspace_bytes",
]
