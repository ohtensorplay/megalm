"""Stable, protocol-independent contracts shared by MegaLM components."""

from megalm.contracts.cache import (
    CacheAllocation,
    CacheHandle,
    CacheKind,
    LogicalCacheDirectory,
)
from megalm.contracts.model import (
    AttentionKind,
    FeedForwardKind,
    GatedDeltaNetSpec,
    ModelSpec,
    StateKind,
)
from megalm.contracts.lora import (
    LoRAAdapterSpec,
    LoRAConfig,
    LoRARequest,
)
from megalm.contracts.plan import ExecutionPlan
from megalm.contracts.request import (
    EngineEvent,
    InferenceRequest,
    RequestState,
    RequestStateMachine,
)

__all__ = [
    "AttentionKind",
    "CacheAllocation",
    "CacheHandle",
    "CacheKind",
    "EngineEvent",
    "ExecutionPlan",
    "FeedForwardKind",
    "GatedDeltaNetSpec",
    "InferenceRequest",
    "LoRAAdapterSpec",
    "LoRAConfig",
    "LoRARequest",
    "LogicalCacheDirectory",
    "ModelSpec",
    "RequestState",
    "RequestStateMachine",
    "StateKind",
]
