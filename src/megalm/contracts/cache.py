from __future__ import annotations

import heapq
import threading
import uuid
from dataclasses import dataclass
from enum import Enum
from types import MappingProxyType
from typing import Iterable, Mapping


class CacheContractError(RuntimeError):
    """Base error for logical cache contract violations."""


class CacheCapacityError(CacheContractError):
    """Raised before execution when a transactional reservation cannot fit."""


class CacheKind(str, Enum):
    KV_PAGE = "kv_page"
    GDN_CONV_STATE = "gdn_conv_state"
    GDN_RECURRENT_STATE = "gdn_recurrent_state"
    PREFIX_CHECKPOINT = "prefix_checkpoint"
    VISION_EMBEDDING = "vision_embedding"
    MEDIA_POSITION = "media_position"
    REQUEST_SLOT = "request_slot"


@dataclass(frozen=True, slots=True, order=True)
class CacheAllocation:
    kind: CacheKind
    arena_id: str
    index: int

    def __post_init__(self) -> None:
        if not self.arena_id:
            raise CacheContractError("cache arena_id must not be empty")
        if self.index < 0:
            raise CacheContractError("cache allocation index must be non-negative")


@dataclass(frozen=True, slots=True)
class CacheHandle:
    """The single logical cache identity for one admitted request."""

    handle_id: str
    request_id: str
    allocations: tuple[CacheAllocation, ...]
    generation: int = 0

    def __post_init__(self) -> None:
        object.__setattr__(self, "allocations", tuple(self.allocations))
        if not self.handle_id or not self.request_id:
            raise CacheContractError("cache handle identity must not be empty")
        if self.generation < 0:
            raise CacheContractError("cache handle generation must be non-negative")
        request_slots = sum(
            allocation.kind is CacheKind.REQUEST_SLOT
            for allocation in self.allocations
        )
        if request_slots != 1:
            raise CacheContractError(
                "cache handle requires exactly one request table slot"
            )
        if len(set(self.allocations)) != len(self.allocations):
            raise CacheContractError("cache handle contains duplicate allocations")

    def refs(self, kind: CacheKind) -> tuple[CacheAllocation, ...]:
        return tuple(
            allocation
            for allocation in self.allocations
            if allocation.kind is kind
        )


class LogicalCacheDirectory:
    """Leader-owned transactional directory for physical cache-arena refs."""

    def __init__(
        self,
        capacities: Mapping[CacheKind, int],
        *,
        arena_id: str = "logical-0",
    ) -> None:
        if not arena_id:
            raise CacheContractError("arena_id must not be empty")
        normalized = {kind: int(capacities.get(kind, 0)) for kind in CacheKind}
        if any(capacity < 0 for capacity in normalized.values()):
            raise CacheContractError("cache capacities must be non-negative")
        if normalized[CacheKind.REQUEST_SLOT] <= 0:
            raise CacheContractError("at least one request table slot is required")
        self._arena_id = arena_id
        self.capacity_failures: dict[str, int] = {}
        self._capacities = MappingProxyType(normalized)
        self._free: dict[CacheKind, list[int]] = {
            kind: list(range(capacity)) for kind, capacity in normalized.items()
        }
        for indices in self._free.values():
            heapq.heapify(indices)
        self._handles: dict[str, CacheHandle] = {}
        self._refcounts: dict[CacheAllocation, int] = {}
        self._pending_requests: set[str] = set()
        self._lock = threading.RLock()

    @property
    def capacities(self) -> Mapping[CacheKind, int]:
        return self._capacities

    @property
    def active_request_ids(self) -> tuple[str, ...]:
        with self._lock:
            return tuple(self._handles)

    def available(self, kind: CacheKind) -> int:
        with self._lock:
            return len(self._free[kind])

    def get(self, request_id: str) -> CacheHandle | None:
        with self._lock:
            return self._handles.get(request_id)

    def allocation(self, kind: CacheKind, index: int) -> CacheAllocation:
        allocation = CacheAllocation(kind, self._arena_id, index)
        self._validate_allocation(allocation)
        return allocation

    def refcount(self, allocation: CacheAllocation) -> int:
        with self._lock:
            self._validate_allocation(allocation)
            return self._refcounts.get(allocation, 0)

    def retain(self, allocations: Iterable[CacheAllocation]) -> None:
        """Atomically retain live allocations for a non-request cache owner."""

        refs = tuple(allocations)
        if len(set(refs)) != len(refs):
            raise CacheContractError("cannot retain duplicate cache allocations")
        if any(ref.kind is CacheKind.REQUEST_SLOT for ref in refs):
            raise CacheContractError("request slots cannot have shared owners")
        with self._lock:
            for allocation in refs:
                self._validate_allocation(allocation)
                if self._refcounts.get(allocation, 0) <= 0:
                    raise CacheContractError(
                        f"cannot retain non-live allocation {allocation}"
                    )
            for allocation in refs:
                self._refcounts[allocation] += 1

    def release_refs(self, allocations: Iterable[CacheAllocation]) -> None:
        """Atomically release refs retained outside a request handle."""

        refs = tuple(allocations)
        if len(set(refs)) != len(refs):
            raise CacheContractError("cannot release duplicate cache allocations")
        if any(ref.kind is CacheKind.REQUEST_SLOT for ref in refs):
            raise CacheContractError("request slots cannot have shared owners")
        with self._lock:
            for allocation in refs:
                self._validate_allocation(allocation)
                if self._refcounts.get(allocation, 0) <= 0:
                    raise CacheContractError(
                        f"cannot release non-live allocation {allocation}"
                    )
            self._release_live_allocations(refs)

    def transaction(self, request_id: str) -> CacheTransaction:
        if not request_id:
            raise CacheContractError("request_id must not be empty")
        with self._lock:
            if request_id in self._pending_requests:
                raise CacheContractError(
                    f"request {request_id!r} already has a pending cache transaction"
                )
            self._pending_requests.add(request_id)
            return CacheTransaction(
                directory=self,
                request_id=request_id,
                base=self._handles.get(request_id),
            )

    def release(self, request_id: str) -> CacheHandle:
        with self._lock:
            if request_id in self._pending_requests:
                raise CacheContractError(
                    f"request {request_id!r} has a pending cache transaction"
                )
            try:
                handle = self._handles.pop(request_id)
            except KeyError as exc:
                raise CacheContractError(
                    f"request {request_id!r} has no cache handle"
                ) from exc
            self._release_live_allocations(handle.allocations)
            return handle

    def _reserve(
        self,
        transaction: CacheTransaction,
        kind: CacheKind,
        count: int,
    ) -> tuple[CacheAllocation, ...]:
        if count < 0:
            raise CacheContractError("reservation count must be non-negative")
        if count == 0:
            return ()
        with self._lock:
            free = self._free[kind]
            if len(free) < count:
                label = kind.value
                self.capacity_failures[label] = (
                    self.capacity_failures.get(label, 0) + 1
                )
                raise CacheCapacityError(
                    f"insufficient {kind.value}: requested {count}, "
                    f"available {len(free)}"
                )
            allocations = tuple(
                CacheAllocation(kind, self._arena_id, heapq.heappop(free))
                for _ in range(count)
            )
            transaction._reserved.extend(allocations)
            return allocations

    def _commit(self, transaction: CacheTransaction) -> CacheHandle:
        with self._lock:
            self._assert_pending(transaction)
            base = transaction._base
            if base is None:
                if self._handles.get(transaction.request_id) is not None:
                    raise CacheContractError(
                        f"request {transaction.request_id!r} already has a handle"
                    )
                self._reserve(transaction, CacheKind.REQUEST_SLOT, 1)
                new_allocations = (
                    tuple(transaction._attached)
                    + tuple(transaction._reserved)
                )
                handle = CacheHandle(
                    handle_id=uuid.uuid4().hex,
                    request_id=transaction.request_id,
                    allocations=new_allocations,
                )
            else:
                current = self._handles.get(transaction.request_id)
                if current != base:
                    raise CacheContractError("cache handle changed during transaction")
                new_allocations = (
                    tuple(transaction._attached)
                    + tuple(transaction._reserved)
                )
                handle = CacheHandle(
                    handle_id=base.handle_id,
                    request_id=base.request_id,
                    allocations=(
                        tuple(
                            allocation
                            for allocation in base.allocations
                            if allocation not in transaction._detached
                        )
                        + new_allocations
                    ),
                    generation=base.generation + 1,
                )
            for allocation in transaction._attached:
                current_refcount = self._refcounts.get(allocation, 0)
                if current_refcount <= 0:
                    raise CacheContractError(
                        f"cannot attach non-live allocation {allocation}"
                    )
            for allocation in transaction._reserved:
                if allocation in self._refcounts:
                    raise CacheContractError(
                        f"reserved allocation is already live: {allocation}"
                    )
            for allocation in transaction._detached:
                if (
                    base is None
                    or allocation not in base.allocations
                    or self._refcounts.get(allocation, 0) <= 0
                ):
                    raise CacheContractError(
                        f"cannot detach non-owned allocation {allocation}"
                    )
            for allocation in transaction._attached:
                self._refcounts[allocation] += 1
            for allocation in transaction._reserved:
                self._refcounts[allocation] = 1
            self._handles[transaction.request_id] = handle
            self._release_live_allocations(tuple(transaction._detached))
            self._pending_requests.remove(transaction.request_id)
            transaction._closed = True
            transaction._committed = True
            return handle

    def _rollback(self, transaction: CacheTransaction) -> None:
        with self._lock:
            if transaction._closed:
                return
            self._assert_pending(transaction)
            self._return_allocations(tuple(transaction._reserved))
            self._pending_requests.remove(transaction.request_id)
            transaction._closed = True

    def _return_allocations(
        self, allocations: tuple[CacheAllocation, ...]
    ) -> None:
        for allocation in allocations:
            if allocation in self._refcounts:
                raise CacheContractError(
                    f"cannot return live allocation {allocation}"
                )
            heapq.heappush(self._free[allocation.kind], allocation.index)

    def _release_live_allocations(
        self,
        allocations: tuple[CacheAllocation, ...],
    ) -> None:
        for allocation in allocations:
            refcount = self._refcounts[allocation] - 1
            if refcount == 0:
                del self._refcounts[allocation]
                heapq.heappush(self._free[allocation.kind], allocation.index)
            else:
                self._refcounts[allocation] = refcount

    def _validate_allocation(self, allocation: CacheAllocation) -> None:
        if allocation.arena_id != self._arena_id:
            raise CacheContractError(
                f"allocation belongs to arena {allocation.arena_id!r}, "
                f"expected {self._arena_id!r}"
            )
        if allocation.index >= self._capacities[allocation.kind]:
            raise CacheContractError(
                f"allocation index {allocation.index} exceeds "
                f"{allocation.kind.value} capacity"
            )

    def assert_integrity(self) -> None:
        with self._lock:
            if self._pending_requests:
                raise CacheContractError(
                    "cannot validate cache directory during a transaction"
                )
            handle_counts: dict[CacheAllocation, int] = {}
            for handle in self._handles.values():
                for allocation in handle.allocations:
                    self._validate_allocation(allocation)
                    handle_counts[allocation] = (
                        handle_counts.get(allocation, 0) + 1
                    )
            for allocation, count in handle_counts.items():
                if self._refcounts.get(allocation, 0) < count:
                    raise CacheContractError(
                        f"allocation {allocation} has fewer refs than owners"
                    )
            for kind, capacity in self._capacities.items():
                free = self._free[kind]
                if len(free) != len(set(free)):
                    raise CacheContractError(
                        f"{kind.value} free list contains duplicates"
                    )
                if any(index < 0 or index >= capacity for index in free):
                    raise CacheContractError(
                        f"{kind.value} free list contains an invalid index"
                    )
                live = {
                    allocation.index
                    for allocation in self._refcounts
                    if allocation.kind is kind
                }
                if live.intersection(free):
                    raise CacheContractError(
                        f"{kind.value} allocation is both live and free"
                    )
                if len(live) + len(free) != capacity:
                    raise CacheContractError(
                        f"{kind.value} accounting does not match capacity"
                    )

    def _assert_pending(self, transaction: CacheTransaction) -> None:
        if transaction.request_id not in self._pending_requests:
            raise CacheContractError("cache transaction is no longer pending")


class CacheTransaction:
    __slots__ = (
        "_base",
        "_closed",
        "_committed",
        "_directory",
        "_attached",
        "_detached",
        "_reserved",
        "request_id",
    )

    def __init__(
        self,
        *,
        directory: LogicalCacheDirectory,
        request_id: str,
        base: CacheHandle | None,
    ) -> None:
        self._directory = directory
        self.request_id = request_id
        self._base = base
        self._attached: list[CacheAllocation] = []
        self._detached: list[CacheAllocation] = []
        self._reserved: list[CacheAllocation] = []
        self._closed = False
        self._committed = False

    def reserve(
        self,
        kind: CacheKind,
        count: int = 1,
    ) -> tuple[CacheAllocation, ...]:
        if self._closed:
            raise CacheContractError("cache transaction is closed")
        if kind is CacheKind.REQUEST_SLOT:
            raise CacheContractError("request slots are reserved automatically")
        try:
            return self._directory._reserve(self, kind, count)
        except Exception:
            self._directory._rollback(self)
            raise

    def attach(
        self,
        allocations: Iterable[CacheAllocation],
    ) -> tuple[CacheAllocation, ...]:
        """Attach already-live shared allocations to this request atomically."""

        if self._closed:
            raise CacheContractError("cache transaction is closed")
        refs = tuple(allocations)
        if any(ref.kind is CacheKind.REQUEST_SLOT for ref in refs):
            self._directory._rollback(self)
            raise CacheContractError(
                "request slots cannot be attached to another request"
            )
        existing = set(self._attached)
        existing.update(self._reserved)
        if self._base is not None:
            existing.update(self._base.allocations)
        if len(set(refs)) != len(refs) or existing.intersection(refs):
            self._directory._rollback(self)
            raise CacheContractError(
                "cache transaction would attach duplicate allocations"
            )
        try:
            with self._directory._lock:
                for allocation in refs:
                    self._directory._validate_allocation(allocation)
                    if self._directory._refcounts.get(allocation, 0) <= 0:
                        raise CacheContractError(
                            f"cannot attach non-live allocation {allocation}"
                        )
                self._attached.extend(refs)
            return refs
        except Exception:
            self._directory._rollback(self)
            raise

    def detach(
        self,
        allocations: Iterable[CacheAllocation],
    ) -> tuple[CacheAllocation, ...]:
        """Atomically drop request-owned refs when a speculative tail rolls back."""

        if self._closed:
            raise CacheContractError("cache transaction is closed")
        refs = tuple(allocations)
        base = self._base
        invalid = (
            base is None
            or not refs
            or len(set(refs)) != len(refs)
            or any(ref.kind is CacheKind.REQUEST_SLOT for ref in refs)
            or not set(refs).issubset(base.allocations)
            or set(refs).intersection(self._detached)
            or set(refs).intersection(self._attached)
            or set(refs).intersection(self._reserved)
        )
        if invalid:
            self._directory._rollback(self)
            raise CacheContractError(
                "cache transaction cannot detach these allocations"
            )
        self._detached.extend(refs)
        return refs

    def commit(self) -> CacheHandle:
        if self._closed:
            raise CacheContractError("cache transaction is closed")
        try:
            return self._directory._commit(self)
        except Exception:
            self._directory._rollback(self)
            raise

    def rollback(self) -> None:
        self._directory._rollback(self)

    def __enter__(self) -> CacheTransaction:
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> bool:
        if not self._closed:
            self.rollback()
        return False
