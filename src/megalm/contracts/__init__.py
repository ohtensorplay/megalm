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
    "LogicalCacheDirectory",
    "ModelSpec",
    "RequestState",
    "RequestStateMachine",
    "StateKind",
]
