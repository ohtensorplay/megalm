from __future__ import annotations

import math
import threading
from collections import Counter, OrderedDict
from dataclasses import dataclass

import tensorplay

from megalm.contracts.request import IMAGE_DETAIL_VALUES


class VisionCacheError(RuntimeError):
    """Raised when GPU vision-cache ownership or geometry is invalid."""


@dataclass(frozen=True, slots=True)
class VisionCacheKey:
    model_digest: str
    processor_digest: str
    content_digest: str
    modality: str = "image"
    detail: str = "auto"

    def __post_init__(self) -> None:
        if (
            len(self.model_digest) != 64
            or len(self.processor_digest) != 64
            or len(self.content_digest) != 64
        ):
            raise VisionCacheError("vision cache keys require SHA-256 digests")
        if self.modality not in {"image", "video", "audio"}:
            raise VisionCacheError("media cache modality is invalid")
        if self.detail not in IMAGE_DETAIL_VALUES:
            raise VisionCacheError("vision cache detail is invalid")


@dataclass(frozen=True, slots=True)
class VisionCacheValue:
    embedding: tensorplay.Tensor
    grid_thw: tuple[int, int, int]
    cached: bool
    frame_feature_counts: tuple[int, ...] = ()
    timestamps: tuple[float, ...] = ()
    seconds_per_grid: tuple[float, ...] = ()

    @property
    def feature_count(self) -> int:
        return self.embedding.shape[0]

    @property
    def nbytes(self) -> int:
        return self.embedding.numel() * self.embedding.element_size()


@dataclass(frozen=True, slots=True)
class VisionCacheStats:
    capacity_bytes: int
    used_bytes: int
    pinned_bytes: int
    entries: int
    pinned_entries: int
    requests: int
    queries: int
    hits: int
    misses: int
    insertions: int
    bypasses: int
    evictions: int
    evicted_bytes: int

    @property
    def hit_rate(self) -> float:
        return 0.0 if self.queries == 0 else self.hits / self.queries


@dataclass(slots=True)
class _Entry:
    value: VisionCacheValue
    ref_count: int = 0


class GPUVisionCache:
    """Byte-bounded global GPU cache for image/video/audio embeddings."""

    def __init__(
        self,
        *,
        capacity_bytes: int,
        width: int,
        dtype: tensorplay.dtype,
        device: tensorplay.device | str,
    ) -> None:
        if capacity_bytes < 0 or width <= 0:
            raise VisionCacheError("vision cache limits are invalid")
        self.capacity_bytes = int(capacity_bytes)
        self.width = int(width)
        self.dtype = dtype
        target_device = tensorplay.device(device)
        if target_device.type == "cuda" and target_device.index is None:
            target_device = tensorplay.device(
                "cuda",
                tensorplay.cuda.current_device(),
            )
        self.device = target_device
        self._entries: OrderedDict[VisionCacheKey, _Entry] = OrderedDict()
        self._request_refs: dict[
            str, Counter[VisionCacheKey]
        ] = {}
        self._used_bytes = 0
        self._queries = 0
        self._hits = 0
        self._misses = 0
        self._insertions = 0
        self._bypasses = 0
        self._evictions = 0
        self._evicted_bytes = 0
        self._lock = threading.RLock()

    def acquire(
        self,
        key: VisionCacheKey,
        request_id: str,
    ) -> VisionCacheValue | None:
        if not request_id:
            raise VisionCacheError("vision cache request id is empty")
        with self._lock:
            self._queries += 1
            entry = self._entries.get(key)
            if entry is None:
                self._misses += 1
                return None
            self._hits += 1
            self._retain(key, entry, request_id)
            return entry.value

    def retain(
        self,
        key: VisionCacheKey,
        request_id: str,
    ) -> VisionCacheValue | None:
        """Pin an already-resolved entry without recording another lookup."""

        if not request_id:
            raise VisionCacheError("vision cache request id is empty")
        with self._lock:
            entry = self._entries.get(key)
            if entry is None:
                return None
            self._retain(key, entry, request_id)
            return entry.value

    def insert_and_acquire(
        self,
        key: VisionCacheKey,
        embedding: tensorplay.Tensor,
        grid_thw: tuple[int, int, int],
        request_id: str,
        *,
        frame_feature_counts: tuple[int, ...] = (),
        timestamps: tuple[float, ...] = (),
        seconds_per_grid: tuple[float, ...] = (),
    ) -> VisionCacheValue:
        self._validate_value(
            key,
            embedding,
            grid_thw,
            frame_feature_counts,
            timestamps,
            seconds_per_grid,
        )
        value = VisionCacheValue(
            embedding=embedding.detach(),
            grid_thw=tuple(int(value) for value in grid_thw),
            cached=True,
            frame_feature_counts=tuple(
                int(value) for value in frame_feature_counts
            ),
            timestamps=tuple(float(value) for value in timestamps),
            seconds_per_grid=tuple(
                float(value) for value in seconds_per_grid
            ),
        )
        with self._lock:
            existing = self._entries.get(key)
            if existing is not None:
                self._retain(key, existing, request_id)
                return existing.value
            if (
                self.capacity_bytes == 0
                or value.nbytes > self.capacity_bytes
                or not self._evict_until_fits(value.nbytes)
            ):
                self._bypasses += 1
                return VisionCacheValue(
                    embedding=value.embedding,
                    grid_thw=value.grid_thw,
                    cached=False,
                    frame_feature_counts=value.frame_feature_counts,
                    timestamps=value.timestamps,
                    seconds_per_grid=value.seconds_per_grid,
                )
            entry = _Entry(value=value)
            self._entries[key] = entry
            self._used_bytes += value.nbytes
            self._insertions += 1
            self._retain(key, entry, request_id)
            return value

    def release_request(self, request_id: str) -> None:
        with self._lock:
            refs = self._request_refs.pop(request_id, None)
            if refs is None:
                return
            for key, count in refs.items():
                entry = self._entries.get(key)
                if entry is None or entry.ref_count < count:
                    raise VisionCacheError(
                        "vision cache request reference is inconsistent"
                    )
                entry.ref_count -= count

    def evict_bytes(self, target_bytes: int) -> int:
        if target_bytes < 0:
            raise VisionCacheError("vision eviction target is negative")
        with self._lock:
            released = 0
            for key in tuple(self._entries):
                if released >= target_bytes:
                    break
                entry = self._entries[key]
                if entry.ref_count:
                    continue
                released += self._evict(key)
            return released

    def reset(self) -> None:
        with self._lock:
            if any(entry.ref_count for entry in self._entries.values()):
                raise VisionCacheError(
                    "cannot reset vision cache with active request references"
                )
            for key in tuple(self._entries):
                self._evict(key)
            self._request_refs.clear()

    def stats(self) -> VisionCacheStats:
        with self._lock:
            pinned = tuple(
                entry
                for entry in self._entries.values()
                if entry.ref_count
            )
            return VisionCacheStats(
                capacity_bytes=self.capacity_bytes,
                used_bytes=self._used_bytes,
                pinned_bytes=sum(
                    entry.value.nbytes for entry in pinned
                ),
                entries=len(self._entries),
                pinned_entries=len(pinned),
                requests=len(self._request_refs),
                queries=self._queries,
                hits=self._hits,
                misses=self._misses,
                insertions=self._insertions,
                bypasses=self._bypasses,
                evictions=self._evictions,
                evicted_bytes=self._evicted_bytes,
            )

    def assert_integrity(self) -> None:
        with self._lock:
            expected_refs: Counter[VisionCacheKey] = Counter()
            for refs in self._request_refs.values():
                expected_refs.update(refs)
            for key, entry in self._entries.items():
                if entry.ref_count != expected_refs[key]:
                    raise VisionCacheError(
                        "vision cache reference count is inconsistent"
                    )
            if set(expected_refs) - set(self._entries):
                raise VisionCacheError(
                    "vision cache request references an evicted entry"
                )
            used = sum(
                entry.value.nbytes for entry in self._entries.values()
            )
            if used != self._used_bytes or used > self.capacity_bytes:
                raise VisionCacheError(
                    "vision cache byte accounting is inconsistent"
                )

    def _retain(
        self,
        key: VisionCacheKey,
        entry: _Entry,
        request_id: str,
    ) -> None:
        entry.ref_count += 1
        self._request_refs.setdefault(request_id, Counter())[key] += 1
        self._entries.move_to_end(key)

    def _evict_until_fits(self, required_bytes: int) -> bool:
        while self._used_bytes + required_bytes > self.capacity_bytes:
            candidate = next(
                (
                    key
                    for key, entry in self._entries.items()
                    if entry.ref_count == 0
                ),
                None,
            )
            if candidate is None:
                return False
            self._evict(candidate)
        return True

    def _evict(self, key: VisionCacheKey) -> int:
        entry = self._entries.pop(key)
        if entry.ref_count:
            raise VisionCacheError("cannot evict a pinned vision embedding")
        released = entry.value.nbytes
        self._used_bytes -= released
        self._evictions += 1
        self._evicted_bytes += released
        return released

    def _validate_value(
        self,
        key: VisionCacheKey,
        embedding: tensorplay.Tensor,
        grid_thw: tuple[int, int, int],
        frame_feature_counts: tuple[int, ...],
        timestamps: tuple[float, ...],
        seconds_per_grid: tuple[float, ...],
    ) -> None:
        if (
            embedding.ndim != 2
            or embedding.shape[0] <= 0
            or embedding.shape[1] != self.width
            or embedding.dtype != self.dtype
            or embedding.device != self.device
            or not embedding.is_contiguous()
            or len(grid_thw) != 3
            or any(int(value) <= 0 for value in grid_thw)
        ):
            raise VisionCacheError("vision embedding geometry is invalid")
        if key.modality in {"image", "audio"}:
            if frame_feature_counts or timestamps or seconds_per_grid:
                raise VisionCacheError(
                    f"{key.modality} cache entries cannot carry video metadata"
                )
            return
        if (
            not frame_feature_counts
            or len(frame_feature_counts) != int(grid_thw[0])
            or len(timestamps) != int(grid_thw[0])
            or any(int(value) <= 0 for value in frame_feature_counts)
            or sum(int(value) for value in frame_feature_counts)
            != embedding.shape[0]
            or any(
                not math.isfinite(float(value)) or float(value) < 0
                for value in timestamps
            )
            or len(seconds_per_grid) != int(grid_thw[0])
            or any(
                not math.isfinite(float(value)) or float(value) <= 0
                for value in seconds_per_grid
            )
        ):
            raise VisionCacheError("video cache metadata is invalid")


__all__ = [
    "GPUVisionCache",
    "VisionCacheError",
    "VisionCacheKey",
    "VisionCacheStats",
    "VisionCacheValue",
]
