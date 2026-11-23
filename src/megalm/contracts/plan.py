from __future__ import annotations

import hashlib
from dataclasses import dataclass, replace
from enum import Enum

from megalm.contracts._canonical import canonical_digest, canonical_value
from megalm.contracts.model import StateKind


class PlanContractError(ValueError):
    """Raised when an execution plan is invalid or has a mismatched digest."""


class ExecutionPhase(str, Enum):
    PREFILL = "prefill"
    DECODE = "decode"
    VISION = "vision"


class CollectiveKind(str, Enum):
    ALL_REDUCE = "all_reduce"
    ALL_GATHER = "all_gather"
    REDUCE_SCATTER = "reduce_scatter"


@dataclass(frozen=True, slots=True)
class HardwareSpec:
    accelerator: str
    device_name: str
    compute_capability: tuple[int, int]
    total_memory_bytes: int

    def __post_init__(self) -> None:
        if not self.accelerator or not self.device_name:
            raise PlanContractError("hardware identity must not be empty")
        if len(self.compute_capability) != 2:
            raise PlanContractError("compute capability must be a pair")
        if self.total_memory_bytes <= 0:
            raise PlanContractError("total GPU memory must be positive")


@dataclass(frozen=True, slots=True)
class TopologySpec:
    tensor_parallel: int = 1
    data_parallel: int = 1
    node_count: int = 1
    gpus_per_node: int = 1

    def __post_init__(self) -> None:
        if min(
            self.tensor_parallel,
            self.data_parallel,
            self.node_count,
            self.gpus_per_node,
        ) <= 0:
            raise PlanContractError("topology degrees must be positive")
        if self.tensor_parallel > self.gpus_per_node:
            raise PlanContractError("v1 does not allow cross-node tensor parallelism")
        if self.tensor_parallel * self.data_parallel > (
            self.node_count * self.gpus_per_node
        ):
            raise PlanContractError("topology requires more GPUs than it declares")

    @property
    def world_size(self) -> int:
        return self.tensor_parallel * self.data_parallel


@dataclass(frozen=True, slots=True)
class WorkloadProfile:
    max_batch_size: int
    max_sequence_length: int
    typical_prefill_tokens: int
    typical_decode_tokens: int

    def __post_init__(self) -> None:
        if min(
            self.max_batch_size,
            self.max_sequence_length,
            self.typical_prefill_tokens,
            self.typical_decode_tokens,
        ) <= 0:
            raise PlanContractError("workload dimensions must be positive")
        if self.typical_prefill_tokens > self.max_sequence_length:
            raise PlanContractError("typical prefill exceeds max sequence length")


@dataclass(frozen=True, slots=True)
class ShapeGuard:
    min_batch_size: int
    max_batch_size: int
    min_tokens: int
    max_tokens: int

    def __post_init__(self) -> None:
        if min(
            self.min_batch_size,
            self.max_batch_size,
            self.min_tokens,
            self.max_tokens,
        ) <= 0:
            raise PlanContractError("shape guard bounds must be positive")
        if self.min_batch_size > self.max_batch_size:
            raise PlanContractError("invalid batch shape guard")
        if self.min_tokens > self.max_tokens:
            raise PlanContractError("invalid token shape guard")


@dataclass(frozen=True, slots=True)
class KernelSelection:
    operator: str
    phases: tuple[ExecutionPhase, ...]
    backend: str
    layer_indices: tuple[int, ...]
    guard: ShapeGuard
    variant: str = "reference"
    abi_version: int = 1
    artifact_digest: str = ""
    weight_layout: str = "qwen3.native-packed.v1"
    workspace: str = "none"
    workspace_bytes: int = 0

    def __post_init__(self) -> None:
        object.__setattr__(self, "phases", tuple(self.phases))
        object.__setattr__(self, "layer_indices", tuple(self.layer_indices))
        if not self.operator or not self.backend:
            raise PlanContractError("kernel operator/backend must not be empty")
        if not self.phases:
            raise PlanContractError("kernel selection requires an execution phase")
        if any(index < 0 for index in self.layer_indices):
            raise PlanContractError("kernel layer indices must be non-negative")
        if not self.variant or not self.weight_layout or not self.workspace:
            raise PlanContractError("kernel resource identity must not be empty")
        if self.abi_version <= 0 or self.workspace_bytes < 0:
            raise PlanContractError("kernel ABI or workspace size is invalid")
        if not self.artifact_digest:
            fallback_digest = hashlib.sha256(
                (
                    f"{self.backend}\0{self.variant}\0{self.abi_version}\0"
                    f"{self.weight_layout}"
                ).encode("utf-8")
            ).hexdigest()
            object.__setattr__(
                self,
                "artifact_digest",
                fallback_digest,
            )
        if len(self.artifact_digest) != 64:
            raise PlanContractError(
                "kernel artifact_digest must be a SHA-256 digest"
            )


@dataclass(frozen=True, slots=True)
class WeightPlan:
    source_format: str
    source_dtype: str
    target_dtype: str
    loader: str
    estimated_bytes: int
    tensor_parallel: int
    packing_rules: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "packing_rules", tuple(self.packing_rules))
        if not all(
            (self.source_format, self.source_dtype, self.target_dtype, self.loader)
        ):
            raise PlanContractError("weight plan strings must not be empty")
        if self.estimated_bytes <= 0 or self.tensor_parallel <= 0:
            raise PlanContractError("weight plan sizes must be positive")


@dataclass(frozen=True, slots=True)
class MemoryPlan:
    total_bytes: int
    weights_bytes: int
    graph_bytes: int
    workspace_bytes: int
    elastic_cache_bytes: int
    reserved_bytes: int

    def __post_init__(self) -> None:
        values = (
            self.total_bytes,
            self.weights_bytes,
            self.graph_bytes,
            self.workspace_bytes,
            self.elastic_cache_bytes,
            self.reserved_bytes,
        )
        if any(value < 0 for value in values):
            raise PlanContractError("memory plan values must be non-negative")
        committed = (
            self.weights_bytes
            + self.graph_bytes
            + self.workspace_bytes
            + self.elastic_cache_bytes
            + self.reserved_bytes
        )
        if committed > self.total_bytes:
            raise PlanContractError("memory plan exceeds total device memory")


@dataclass(frozen=True, slots=True)
class CachePlan:
    state_kinds: tuple[StateKind, ...]
    kv_page_size_tokens: int
    checkpoint_alignment_tokens: int
    capacity_bytes: int
    prefix_backend: str = "disabled"
    copy_on_write: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "state_kinds", tuple(self.state_kinds))
        if not self.prefix_backend:
            raise PlanContractError("cache prefix backend must not be empty")
        if not self.state_kinds:
            raise PlanContractError("cache plan requires at least one state kind")
        if min(
            self.kv_page_size_tokens,
            self.checkpoint_alignment_tokens,
            self.capacity_bytes,
        ) <= 0:
            raise PlanContractError("cache plan sizes must be positive")
        if self.prefix_backend != "disabled":
            if self.checkpoint_alignment_tokens != self.kv_page_size_tokens:
                raise PlanContractError(
                    "v1 prefix checkpoints must be KV-page aligned"
                )
            if not self.copy_on_write:
                raise PlanContractError(
                    "a shared prefix cache requires copy-on-write"
                )


@dataclass(frozen=True, slots=True)
class CollectiveSite:
    after_operator: str
    kind: CollectiveKind
    ranks: tuple[int, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "ranks", tuple(self.ranks))
        if not self.after_operator or len(self.ranks) < 2:
            raise PlanContractError("collective requires an operator and multiple ranks")


@dataclass(frozen=True, slots=True)
class ExecutionPlan:
    """An immutable, rank-verifiable result of all cold-path decisions."""

    model_digest: str
    hardware: HardwareSpec
    topology: TopologySpec
    dtype: str
    workload: WorkloadProfile
    weights: WeightPlan
    kernels: tuple[KernelSelection, ...]
    memory: MemoryPlan
    cache: CachePlan
    kernel_binding_digest: str = ""
    collectives: tuple[CollectiveSite, ...] = ()
    cuda_graph_buckets: tuple[int, ...] = ()
    vision_backend: str | None = None
    schema_version: int = 3
    plan_digest: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "kernels", tuple(self.kernels))
        object.__setattr__(self, "collectives", tuple(self.collectives))
        object.__setattr__(
            self, "cuda_graph_buckets", tuple(self.cuda_graph_buckets)
        )
        if self.schema_version != 3:
            raise PlanContractError("unsupported ExecutionPlan schema_version")
        if len(self.model_digest) != 64:
            raise PlanContractError("model_digest must be a SHA-256 digest")
        if not self.dtype:
            raise PlanContractError("plan dtype must not be empty")
        if not self.kernels:
            raise PlanContractError("execution plan requires selected kernels")
        if not self.kernel_binding_digest:
            binding_hash = hashlib.sha256()
            for kernel in self.kernels:
                binding_hash.update(kernel.backend.encode("utf-8"))
                binding_hash.update(b"\0")
                binding_hash.update(kernel.artifact_digest.encode("ascii"))
            object.__setattr__(
                self,
                "kernel_binding_digest",
                binding_hash.hexdigest(),
            )
        if len(self.kernel_binding_digest) != 64:
            raise PlanContractError(
                "kernel_binding_digest must be a SHA-256 digest"
            )
        if any(bucket <= 0 for bucket in self.cuda_graph_buckets):
            raise PlanContractError("CUDA graph buckets must be positive")
        if tuple(sorted(set(self.cuda_graph_buckets))) != self.cuda_graph_buckets:
            raise PlanContractError("CUDA graph buckets must be sorted and unique")
        if self.topology.tensor_parallel != self.weights.tensor_parallel:
            raise PlanContractError("weight sharding and topology TP degree differ")
        world_size = self.topology.world_size
        for site in self.collectives:
            if any(rank < 0 or rank >= world_size for rank in site.ranks):
                raise PlanContractError(
                    "collective ranks must stay within the topology world"
                )
        if self.memory.elastic_cache_bytes != self.cache.capacity_bytes:
            raise PlanContractError("cache and memory budgets differ")
        if self.plan_digest and self.plan_digest != self.computed_digest:
            raise PlanContractError("ExecutionPlan digest does not match its contents")

    @classmethod
    def create(
        cls,
        *,
        model_digest: str,
        hardware: HardwareSpec,
        topology: TopologySpec,
        dtype: str,
        workload: WorkloadProfile,
        weights: WeightPlan,
        kernels: tuple[KernelSelection, ...],
        memory: MemoryPlan,
        cache: CachePlan,
        kernel_binding_digest: str = "",
        collectives: tuple[CollectiveSite, ...] = (),
        cuda_graph_buckets: tuple[int, ...] = (),
        vision_backend: str | None = None,
    ) -> ExecutionPlan:
        provisional = cls(
            model_digest=model_digest,
            hardware=hardware,
            topology=topology,
            dtype=dtype,
            workload=workload,
            weights=weights,
            kernels=kernels,
            memory=memory,
            cache=cache,
            kernel_binding_digest=kernel_binding_digest,
            collectives=collectives,
            cuda_graph_buckets=cuda_graph_buckets,
            vision_backend=vision_backend,
        )
        return replace(provisional, plan_digest=provisional.computed_digest)

    @property
    def computed_digest(self) -> str:
        return canonical_digest(self, omit=frozenset({"plan_digest"}))

    def verify_digest(self, expected_digest: str | None = None) -> None:
        actual = self.computed_digest
        expected = self.plan_digest if expected_digest is None else expected_digest
        if actual != expected:
            raise PlanContractError(
                f"plan digest mismatch: expected {expected}, computed {actual}"
            )

    def to_dict(self) -> dict[str, object]:
        return canonical_value(self)
