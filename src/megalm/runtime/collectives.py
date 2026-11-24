"""Tensor-parallel collective abstraction.

The plan fixes collective sites at seal time (``ExecutionPlan.collectives``).
This module supplies the runtime half: a single, typed object that model
layers call at those fixed sites.

Design rules from ``devdoc/GAP-PLAN.md`` §2 and §13:

* TP=1 is a no-op singleton. No group lookup, allocation, or synchronization
  is performed on the hot path, and no extra process is started.
* TP>1 uses TensorPlay's distributed process group (NCCL on CUDA, Gloo on CPU).
* The group is created once during startup and shared by every layer. Layers
  bind the callable at construction time so the forward path does no dispatch.
* A recording backend supports unit tests on single-GPU hosts.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import timedelta
from typing import Any, Protocol, runtime_checkable

import tensorplay.distributed as dist


class CollectiveError(RuntimeError):
    """Raised when a tensor-parallel collective group is unusable."""


@runtime_checkable
class CollectiveGroup(Protocol):
    """The surface model layers and the engine need from a TP group."""

    rank: int
    world_size: int
    enabled: bool

    def all_reduce(self, tensor: Any) -> None:
        """Sum ``tensor`` across ranks, in place."""

    def broadcast_object(self, obj: Any, *, src: int = 0) -> Any:
        """Pickle-and-broadcast a host object (used for BatchPlan/tokens)."""

    def barrier(self) -> None:
        """Block until every rank reaches this point."""

    def check_health(self) -> None:
        """Raise if any rank has left the group."""

    def close(self) -> None:
        """Tear down the group; idempotent."""


class _NoopCollectiveGroup:
    """TP=1 singleton. Every method is a genuine no-op."""

    rank = 0
    world_size = 1
    enabled = False

    def all_reduce(self, tensor: Any) -> None:
        return None

    def broadcast_object(self, obj: Any, *, src: int = 0) -> Any:
        return obj

    def barrier(self) -> None:
        return None

    def check_health(self) -> None:
        return None

    def close(self) -> None:
        return None


NOOP_COLLECTIVE: CollectiveGroup = _NoopCollectiveGroup()


@dataclass(frozen=True, slots=True)
class TPGroupConfig:
    """How to form the tensor-parallel process group."""

    rank: int
    world_size: int
    init_method: str = "env://"
    backend: str = "nccl"
    timeout_seconds: float = 600.0

    def __post_init__(self) -> None:
        if self.world_size <= 0:
            raise CollectiveError("TP world size must be positive")
        if not 0 <= self.rank < self.world_size:
            raise CollectiveError("TP rank is outside the world")
        if self.world_size > 1 and not self.init_method:
            raise CollectiveError("TP init method must not be empty")
        if self.timeout_seconds <= 0:
            raise CollectiveError("TP timeout must be positive")


class DistributedCollectiveGroup:
    """A real TP group backed by ``tensorplay.distributed``."""

    def __init__(self, config: TPGroupConfig) -> None:
        self._config = config
        self._closed = False
        if config.world_size == 1:
            # Still a distinct object so callers can close it uniformly, but
            # no process group is created and the hot path stays dispatchable.
            self.rank = 0
            self.world_size = 1
            self.enabled = False
            return
        if not dist.is_available():
            raise CollectiveError("TensorPlay distributed is not available")
        try:
            dist.init_process_group(
                backend=config.backend,
                init_method=config.init_method,
                world_size=config.world_size,
                rank=config.rank,
                timeout=timedelta(seconds=config.timeout_seconds),
            )
        except Exception as exc:  # pragma: no cover - backend specific
            raise CollectiveError(f"failed to initialize TP group: {exc}") from exc
        self.rank = dist.get_rank()
        self.world_size = dist.get_world_size()
        if self.world_size != config.world_size:
            raise CollectiveError("TP world size disagree with process group")
        self.enabled = True

    def all_reduce(self, tensor: Any) -> None:
        if not self.enabled:
            return
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)

    def broadcast_object(self, obj: Any, *, src: int = 0) -> Any:
        if not self.enabled:
            return obj
        objects = [obj] if self.rank == src else [None]
        dist.broadcast_object_list(objects, src=src)
        return objects[0]

    def barrier(self) -> None:
        if self.enabled:
            dist.barrier()

    def check_health(self) -> None:
        if self.enabled:
            # A barrier is the only reliable liveness signal for NCCL/Gloo:
            # a dead or wedged rank makes this raise on the configured timeout.
            self.barrier()

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        if self.enabled:
            self.enabled = False
            try:
                dist.destroy_process_group()
            except Exception:  # pragma: no cover - teardown best effort
                pass


@dataclass
class RecordingCollectiveGroup:
    """In-process fake for TP logic tests (no group, no GPU required).

    ``reduce`` maps a local tensor to the value the rank would see after a
    real reduction. Tests use an identity by default and can inject a sum or
    a counter when they need to assert reduction behaviour.
    """

    rank: int = 0
    world_size: int = 2
    reduce: Any = None
    all_reduce_calls: list[int] = field(default_factory=list)
    broadcasts: list[Any] = field(default_factory=list)
    barriers: int = 0
    closed: bool = False
    fail_health: bool = False

    @property
    def enabled(self) -> bool:
        return True

    def all_reduce(self, tensor: Any) -> None:
        self.all_reduce_calls.append(self.rank)
        if self.reduce is not None:
            reduced = self.reduce(tensor, self.rank)
            # In-place update keeps captured graphs and callers valid.
            tensor.copy_(reduced)

    def broadcast_object(self, obj: Any, *, src: int = 0) -> Any:
        self.broadcasts.append(obj)
        return obj

    def barrier(self) -> None:
        self.barriers += 1

    def check_health(self) -> None:
        if self.fail_health:
            raise CollectiveError("simulated rank failure")

    def close(self) -> None:
        self.closed = True


def bind_all_reduce(group: CollectiveGroup):
    """Return the exact callable a layer should store on its instance.

    For TP=1 this returns a function that does nothing; layers still call it
    unconditionally so the compiled/captured graph shape is identical across
    TP degrees. There is no branch or group lookup in the forward path.
    """

    if not group.enabled:
        return _noop_all_reduce
    return group.all_reduce


def _noop_all_reduce(tensor: Any) -> None:
    return None


def require_tp_group(group: CollectiveGroup | None) -> CollectiveGroup:
    if group is None:  # defensive: layers must be constructed with one
        raise CollectiveError("tensor-parallel group was not bound")
    return group
