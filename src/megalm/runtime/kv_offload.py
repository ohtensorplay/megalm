"""Hierarchical KV cache offloading.

Three tiers:

* GPU: the paged arena owned by :class:`megalm.runtime.cache.PagedKVCache`.
  It is the only tier the attention kernels read.
* Host: a page-granular, pinned-memory store.  It is the GPU's DMA peer and
  the mandatory staging tier for the filesystem tier.
* Filesystem: content-addressed pages on local disk.  Disk pages are always
  staged through the host tier; GPU never touches them directly.

Pages are addressed by a chained content digest over the exact token pages
(the same chain used by the radix prefix cache) plus the runtime identity
(model digest, dtype, KV geometry).  Entries therefore deduplicate across
requests and can never alias across models, dtypes or TP layouts.

Ownership rules, mirroring the reference implementations:

* A host entry is immutable once ready.  ``refs`` pins it against eviction
  for the duration of a DMA transfer or a disk read/write.
* Disk entries are immutable files.  Removal only happens under the disk
  byte budget; a read that lost the race degrades to a cache miss.
* GPU pages are never released while a host store is reading them or while a
  promotion is writing them.  :class:`TieredKVOffload` hands the retention
  decision to the cache manager through store tickets and promotion
  outcomes.
* Any failure degrades to a miss.  Offload must never make a request fail.
"""

from __future__ import annotations

import hashlib
import json
import os
import queue
import struct
import threading
import time
import zlib
from collections import OrderedDict
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import tensorplay

from megalm.runtime.radix_cache import RadixKey

_DISK_MAGIC = b"MGLMKV1\0"
_DISK_VERSION = 1
_DISK_HEADER = struct.Struct("<8sIIIIII32s32sQII")
_DISK_CONFIG = "config.json"
_DISK_SUFFIX = ".bin"


class KVOffloadError(RuntimeError):
    """Raised for offload configuration errors that must fail at startup."""


def _dtype_size(dtype: tensorplay.dtype) -> int:
    return int(tensorplay.empty((), dtype=dtype).element_size())


# ---------------------------------------------------------------------------
# Content-addressed page keys
# ---------------------------------------------------------------------------


class KVKeyChain:
    """Chained content digests for an append-only sequence of token pages.

    ``digest(i)`` covers pages ``0..i`` (inclusive) of one namespace, seeded
    with the runtime identity.  Extending is incremental, so publishing a
    decode page costs O(page) rather than O(prompt).
    """

    __slots__ = (
        "_digests",
        "_hasher",
        "_pages",
        "identity_digest",
        "namespace",
        "page_size",
    )

    def __init__(
        self,
        *,
        identity_digest: bytes,
        namespace: str,
        page_size: int,
        pages: Sequence[Sequence[int]] = (),
    ) -> None:
        if page_size <= 0:
            raise KVOffloadError("offload page size must be positive")
        if len(identity_digest) != 32:
            raise KVOffloadError("offload identity digest must be 32 bytes")
        self.identity_digest = identity_digest
        self.namespace = namespace
        self.page_size = page_size
        self._hasher = hashlib.blake2b(digest_size=32)
        self._hasher.update(identity_digest)
        self._hasher.update(namespace.encode())
        self._hasher.update(struct.pack("<q", page_size))
        self._digests: list[bytes] = []
        self._pages: list[tuple[int, ...]] = []
        self.extend(pages)

    @property
    def pages(self) -> tuple[tuple[int, ...], ...]:
        return tuple(self._pages)

    def __len__(self) -> int:
        return len(self._digests)

    def digest(self, index: int) -> bytes:
        return self._digests[index]

    def tokens(self, index: int) -> tuple[int, ...]:
        return self._pages[index]

    def extend(self, pages: Sequence[Sequence[int]]) -> None:
        for page in pages:
            tokens = tuple(int(token) for token in page)
            if len(tokens) != self.page_size:
                raise KVOffloadError("offload key contains a partial page")
            self._hasher.update(
                struct.pack(f"<{self.page_size}q", *tokens)
            )
            self._digests.append(self._hasher.digest())
            self._pages.append(tokens)


# ---------------------------------------------------------------------------
# Copy transport
# ---------------------------------------------------------------------------


class CopyTask:
    """A submitted DMA batch; readiness is polled by the leader."""

    __slots__ = ("event", "stream")

    def __init__(
        self,
        event: object | None = None,
        stream: object | None = None,
    ) -> None:
        self.event = event
        self.stream = stream

    def ready(self) -> bool:
        if self.event is None:
            return True
        return bool(self.event.query())  # type: ignore[attr-defined]

    def wait(self) -> None:
        if self.event is not None:
            self.event.synchronize()  # type: ignore[attr-defined]

    def attach(self) -> None:
        """Order the compute stream after this transfer before it is read."""

        if self.event is None or self.stream is None:
            return
        tensorplay.cuda.current_stream().wait_event(self.event)


class _PairTask:
    __slots__ = ("_key", "_value")

    def __init__(self, key: CopyTask, value: CopyTask) -> None:
        self._key = key
        self._value = value

    def ready(self) -> bool:
        return self._key.ready() and self._value.ready()

    def wait(self) -> None:
        self._key.wait()
        self._value.wait()

    def attach(self) -> None:
        self._key.attach()
        self._value.attach()


class TensorCopier:
    """Page-granular D2H/H2D copies on a dedicated CUDA stream.

    On a CPU-only engine the copies run synchronously and complete
    immediately; the ownership and promotion state machine stays identical,
    which keeps the failure paths testable without a GPU.
    """

    __slots__ = ("_stream", "asynchronous")

    def __init__(self, *, asynchronous: bool) -> None:
        self.asynchronous = bool(asynchronous)
        self._stream = tensorplay.cuda.Stream() if self.asynchronous else None

    def to_host(
        self,
        host: tensorplay.Tensor,
        slots: Sequence[int],
        device: tensorplay.Tensor,
        pages: Sequence[int],
        *,
        rows: int,
    ) -> CopyTask:
        return self._copy(host, slots, device, pages, rows=rows, to_host=True)

    def to_device(
        self,
        host: tensorplay.Tensor,
        slots: Sequence[int],
        device: tensorplay.Tensor,
        pages: Sequence[int],
        *,
        rows: int,
    ) -> CopyTask:
        return self._copy(host, slots, device, pages, rows=rows, to_host=False)

    def _copy(
        self,
        host: tensorplay.Tensor,
        slots: Sequence[int],
        device: tensorplay.Tensor,
        pages: Sequence[int],
        *,
        rows: int,
        to_host: bool,
    ) -> CopyTask:
        if len(slots) != len(pages):
            raise KVOffloadError("copy batch does not pair slots with pages")
        stream = self._stream
        if stream is None:
            for slot, page in zip(slots, pages, strict=True):
                for row in range(rows):
                    target = host[slot, row] if to_host else device[row, page]
                    source = device[row, page] if to_host else host[slot, row]
                    target.copy_(source)
            return CopyTask()
        with tensorplay.cuda.stream(stream):
            # Device pages must not be read while earlier kernels still write
            # them, and host slots must not be reused before an earlier H2D
            # copy finishes.  A stream dependency covers both directions.
            stream.wait_stream(tensorplay.cuda.current_stream())
            for slot, page in zip(slots, pages, strict=True):
                for row in range(rows):
                    target = host[slot, row] if to_host else device[row, page]
                    source = device[row, page] if to_host else host[slot, row]
                    target.copy_(source, non_blocking=True)
            event = tensorplay.cuda.Event()
            event.record(stream)
        return CopyTask(event, stream)


# ---------------------------------------------------------------------------
# Host tier
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class _HostEntry:
    digest: bytes
    slot: int
    tokens: tuple[int, ...]
    ready: bool
    refs: int


@dataclass(frozen=True, slots=True)
class HostTierStats:
    capacity_pages: int
    used_pages: int
    free_pages: int
    pinned_pages: int
    inflight_pages: int
    pages_bytes: int
    used_bytes: int
    stores: int
    store_pages: int
    loads: int
    load_pages: int
    evictions: int
    evicted_pages: int
    hits: int
    misses: int
    lookup_pages: int
    hit_pages: int
    store_failures: int
    load_failures: int

    @property
    def hit_rate(self) -> float:
        return 0.0 if self.lookup_pages == 0 else self.hit_pages / self.lookup_pages

    def snapshot(self) -> dict[str, object]:
        return {
            "capacity_pages": self.capacity_pages,
            "used_pages": self.used_pages,
            "free_pages": self.free_pages,
            "pinned_pages": self.pinned_pages,
            "inflight_pages": self.inflight_pages,
            "used_bytes": self.used_bytes,
            "pages_bytes": self.pages_bytes,
            "stores": self.stores,
            "store_pages": self.store_pages,
            "loads": self.loads,
            "load_pages": self.load_pages,
            "evictions": self.evictions,
            "evicted_pages": self.evicted_pages,
            "hits": self.hits,
            "misses": self.misses,
            "lookup_pages": self.lookup_pages,
            "hit_pages": self.hit_pages,
            "hit_rate": self.hit_rate,
            "store_failures": self.store_failures,
            "load_failures": self.load_failures,
        }


class HostKVTier:
    """Pinned host page store with LRU eviction and reference counting."""

    def __init__(
        self,
        *,
        identity_digest: bytes,
        rows: int,
        page_size: int,
        num_kv_heads: int,
        head_dim: int,
        dtype: tensorplay.dtype,
        capacity_bytes: int,
        copier: TensorCopier,
        pin_memory: bool,
    ) -> None:
        if min(rows, page_size, num_kv_heads, head_dim) <= 0:
            raise KVOffloadError("host tier geometry must be positive")
        element = _dtype_size(dtype)
        self.page_bytes = (
            2 * rows * page_size * num_kv_heads * head_dim * element
        )
        slots = int(capacity_bytes) // self.page_bytes
        if slots <= 0:
            raise KVOffloadError(
                "host KV cache budget is smaller than one page"
            )
        self.identity_digest = identity_digest
        self.rows = rows
        self.page_size = page_size
        self.capacity = slots
        self.copier = copier
        shape = (slots, rows, page_size, num_kv_heads, head_dim)
        key = tensorplay.empty(shape, dtype=dtype, device="cpu")
        value = tensorplay.empty(shape, dtype=dtype, device="cpu")
        if pin_memory:
            key = key.pin_memory()
            value = value.pin_memory()
        self.key = key
        self.value = value
        self._lock = threading.RLock()
        self._entries: OrderedDict[bytes, _HostEntry] = OrderedDict()
        self._free: list[int] = list(range(slots - 1, -1, -1))
        self._pending_stores: list[tuple[tuple[bytes, ...], CopyTask]] = []
        self._stores = 0
        self._store_pages = 0
        self._loads = 0
        self._load_pages = 0
        self._evictions = 0
        self._evicted_pages = 0
        self._hits = 0
        self._misses = 0
        self._lookup_pages = 0
        self._hit_pages = 0
        self._store_failures = 0
        self._load_failures = 0

    @property
    def nbytes(self) -> int:
        return self.capacity * self.page_bytes

    def __len__(self) -> int:
        return len(self._entries)

    # -- lookup ---------------------------------------------------------

    def lookup_chain(self, chain: KVKeyChain, start: int) -> int:
        """Return the first index after the longest ready contiguous prefix."""

        with self._lock:
            end = start
            while end < len(chain):
                entry = self._entries.get(chain.digest(end))
                if (
                    entry is None
                    or not entry.ready
                    or entry.tokens != chain.tokens(end)
                ):
                    break
                end += 1
            self._lookup_pages += end - start
            self._hit_pages += end - start
            if end == start:
                self._misses += 1
            else:
                self._hits += 1
                for index in range(start, end):
                    self._entries.move_to_end(chain.digest(index))
            return end

    # -- stores ---------------------------------------------------------

    def submit_store(
        self,
        chain: KVKeyChain,
        start: int,
        end: int,
        *,
        device_key: tensorplay.Tensor,
        device_value: tensorplay.Tensor,
        device_pages: Sequence[int],
    ) -> tuple[int, tuple[bytes, ...]]:
        """Schedule GPU→host copies for newly published pages.

        The caller must keep the GPU pages alive until :meth:`poll` reports
        the store completion.  Only the contiguous subset that fits in the
        host tier is staged.  Returns the first page index staged together
        with its digests.
        """

        with self._lock:
            inflight = {
                digest
                for digests, _task in self._pending_stores
                for digest in digests
            }
            first = start
            while first < end:
                digest = chain.digest(first)
                if digest not in self._entries and digest not in inflight:
                    break
                first += 1
            if first >= end:
                return first, ()
            needed = end - first
            available = self._make_room(needed)
            if available <= 0:
                self._store_failures += 1
                return first, ()
            needed = min(needed, available, len(device_pages) - first)
            digests: list[bytes] = []
            slots: list[int] = []
            for index in range(first, first + needed):
                slot = self._free.pop()
                digest = chain.digest(index)
                self._entries[digest] = _HostEntry(
                    digest=digest,
                    slot=slot,
                    tokens=chain.tokens(index),
                    ready=False,
                    refs=0,
                )
                digests.append(digest)
                slots.append(slot)
            if not digests:
                return first, ()
            pages = list(device_pages[first : first + len(digests)])
            key_task = self.copier.to_host(
                self.key,
                slots,
                device_key,
                pages,
                rows=self.rows,
            )
            value_task = self.copier.to_host(
                self.value,
                slots,
                device_value,
                pages,
                rows=self.rows,
            )
            self._pending_stores.append(
                (tuple(digests), _PairTask(key_task, value_task))
            )
            self._stores += 1
            self._store_pages += len(digests)
            return first, tuple(digests)

    # -- promotions -----------------------------------------------------

    def prepare_load(self, chain: KVKeyChain, start: int, end: int) -> list[int]:
        """Allocate and pin slots that disk loads will fill."""

        with self._lock:
            needed = end - start
            if needed <= 0:
                return []
            if self._make_room(needed) < needed:
                self._load_failures += 1
                return []
            slots: list[int] = []
            for index in range(start, end):
                slot = self._free.pop()
                digest = chain.digest(index)
                self._entries[digest] = _HostEntry(
                    digest=digest,
                    slot=slot,
                    tokens=chain.tokens(index),
                    ready=False,
                    refs=1,
                )
                slots.append(slot)
            self._loads += 1
            self._load_pages += needed
            return slots

    def complete_load(self, digest: bytes, success: bool) -> None:
        with self._lock:
            entry = self._entries.get(digest)
            if entry is None:
                return
            if not success:
                # A failed load may leave a slot with garbage; only release
                # it when no other transfer is pinning the same content.
                if entry.refs == 0:
                    self._drop_entry_locked(entry)
                self._load_failures += 1
                return
            entry.ready = True

    def pin(self, digests: Sequence[bytes]) -> bool:
        with self._lock:
            for digest in digests:
                entry = self._entries.get(digest)
                if entry is None or not entry.ready:
                    return False
            for digest in digests:
                self._entries[digest].refs += 1
            return True

    def unpin(self, digests: Sequence[bytes]) -> None:
        with self._lock:
            for digest in digests:
                entry = self._entries.get(digest)
                if entry is not None and entry.refs > 0:
                    entry.refs -= 1

    def copy_to_device(
        self,
        digests: Sequence[bytes],
        device_key: tensorplay.Tensor,
        device_value: tensorplay.Tensor,
        device_pages: Sequence[int],
    ) -> _PairTask:
        with self._lock:
            slots = []
            for digest in digests:
                entry = self._entries.get(digest)
                if entry is None or not entry.ready:
                    raise KVOffloadError("host page vanished before promotion")
                slots.append(entry.slot)
        return _PairTask(
            self.copier.to_device(
                self.key,
                slots,
                device_key,
                device_pages,
                rows=self.rows,
            ),
            self.copier.to_device(
                self.value,
                slots,
                device_value,
                device_pages,
                rows=self.rows,
            ),
        )

    def entry_location(self, digest: bytes) -> tuple[int, tuple[int, ...]] | None:
        """Slot and token page for a ready entry, used by disk stores."""

        with self._lock:
            entry = self._entries.get(digest)
            if entry is None or not entry.ready:
                return None
            return entry.slot, entry.tokens

    # -- polling --------------------------------------------------------

    def poll(self) -> tuple[bytes, ...]:
        """Return digests whose D2H batch completed since the last call."""

        completed: list[bytes] = []
        with self._lock:
            remaining: list[tuple[tuple[bytes, ...], CopyTask]] = []
            for digests, task in self._pending_stores:
                if not task.ready():
                    remaining.append((digests, task))
                    continue
                for digest in digests:
                    entry = self._entries.get(digest)
                    if entry is None or entry.ready:
                        continue
                    entry.ready = True
                    self._entries.move_to_end(digest)
                    completed.append(digest)
            self._pending_stores = remaining
        return tuple(completed)

    # -- capacity -------------------------------------------------------

    def _make_room(self, needed: int) -> int:
        free = len(self._free)
        if free >= needed:
            return free
        for digest, entry in list(self._entries.items()):
            if free >= needed:
                break
            if entry.refs or not entry.ready:
                continue
            del self._entries[digest]
            self._free.append(entry.slot)
            free += 1
            self._evictions += 1
            self._evicted_pages += 1
        return free

    def _drop_entry_locked(self, entry: _HostEntry) -> None:
        current = self._entries.get(entry.digest)
        if current is not entry:
            return
        del self._entries[entry.digest]
        self._free.append(entry.slot)

    def stats(self) -> HostTierStats:
        with self._lock:
            pinned = sum(1 for entry in self._entries.values() if entry.refs)
            inflight = sum(
                1
                for digests, _task in self._pending_stores
                for digest in digests
                if digest in self._entries
                and not self._entries[digest].ready
            )
            used = len(self._entries)
            return HostTierStats(
                capacity_pages=self.capacity,
                used_pages=used,
                free_pages=len(self._free),
                pinned_pages=pinned,
                inflight_pages=inflight,
                pages_bytes=self.page_bytes,
                used_bytes=used * self.page_bytes,
                stores=self._stores,
                store_pages=self._store_pages,
                loads=self._loads,
                load_pages=self._load_pages,
                evictions=self._evictions,
                evicted_pages=self._evicted_pages,
                hits=self._hits,
                misses=self._misses,
                lookup_pages=self._lookup_pages,
                hit_pages=self._hit_pages,
                store_failures=self._store_failures,
                load_failures=self._load_failures,
            )


# ---------------------------------------------------------------------------
# Filesystem tier
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class DiskTierStats:
    capacity_bytes: int
    used_bytes: int
    entries: int
    stores: int
    store_pages: int
    store_bytes: int
    evictions: int
    evicted_pages: int
    hits: int
    misses: int
    lookup_pages: int
    hit_pages: int
    store_failures: int
    load_failures: int
    inflight_jobs: int

    @property
    def hit_rate(self) -> float:
        return 0.0 if self.lookup_pages == 0 else self.hit_pages / self.lookup_pages

    def snapshot(self) -> dict[str, object]:
        return {
            "capacity_bytes": self.capacity_bytes,
            "used_bytes": self.used_bytes,
            "entries": self.entries,
            "stores": self.stores,
            "store_pages": self.store_pages,
            "bytes_written": self.store_bytes,
            "evictions": self.evictions,
            "evicted_pages": self.evicted_pages,
            "hits": self.hits,
            "misses": self.misses,
            "lookup_pages": self.lookup_pages,
            "hit_pages": self.hit_pages,
            "hit_rate": self.hit_rate,
            "store_failures": self.store_failures,
            "load_failures": self.load_failures,
            "inflight_jobs": self.inflight_jobs,
        }


@dataclass(slots=True)
class _DiskEntry:
    digest: bytes
    path: str
    nbytes: int
    accessed_ns: int


@dataclass(frozen=True, slots=True)
class _DiskJob:
    job_id: int
    kind: str
    digest: bytes
    tokens: tuple[int, ...]
    slot: int


@dataclass(frozen=True, slots=True)
class DiskCompletion:
    kind: str
    job_id: int
    success: bool
    digest: bytes


class DiskKVTier:
    """Content-addressed filesystem tier with atomic writes and LRU."""

    def __init__(
        self,
        *,
        root: str | Path,
        identity: str,
        identity_digest: bytes,
        page_size: int,
        rows: int,
        num_kv_heads: int,
        head_dim: int,
        dtype: tensorplay.dtype,
        capacity_bytes: int,
        host: HostKVTier,
        read_threads: int = 4,
        write_threads: int = 2,
    ) -> None:
        if capacity_bytes <= 0:
            raise KVOffloadError("disk KV cache budget must be positive")
        if read_threads <= 0 or write_threads <= 0:
            raise KVOffloadError("disk KV cache thread counts must be positive")
        element = _dtype_size(dtype)
        self.page_bytes = (
            2 * rows * page_size * num_kv_heads * head_dim * element
        )
        if capacity_bytes < self.page_bytes:
            raise KVOffloadError("disk KV cache budget is smaller than one page")
        self.root = Path(root).expanduser()
        self.capacity_bytes = int(capacity_bytes)
        self.identity = identity
        self.identity_digest = identity_digest
        self.rows = rows
        self.page_size = page_size
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.dtype = dtype
        self.host = host
        self._dir = self.root / _identity_directory(identity)
        self._lock = threading.RLock()
        self._entries: OrderedDict[bytes, _DiskEntry] = OrderedDict()
        self._bytes = 0
        self._jobs = 0
        self._stores = 0
        self._store_pages = 0
        self._store_bytes = 0
        self._evictions = 0
        self._evicted_pages = 0
        self._hits = 0
        self._misses = 0
        self._lookup_pages = 0
        self._hit_pages = 0
        self._store_failures = 0
        self._load_failures = 0
        self._inflight: set[int] = set()
        self._completions: queue.Queue[DiskCompletion] = queue.Queue()
        self._condition = threading.Condition()
        self._stop = threading.Event()
        self._load_queue: queue.Queue[_DiskJob] = queue.Queue(maxsize=1024)
        self._store_queue: queue.Queue[_DiskJob] = queue.Queue(maxsize=1024)
        self._threads: list[threading.Thread] = []
        self._prepare_layout()
        self._scan()
        for index in range(read_threads):
            thread = threading.Thread(
                target=self._read_loop,
                name=f"megalm-kv-read-{index}",
                daemon=True,
            )
            thread.start()
            self._threads.append(thread)
        for index in range(write_threads):
            thread = threading.Thread(
                target=self._write_loop,
                name=f"megalm-kv-write-{index}",
                daemon=True,
            )
            thread.start()
            self._threads.append(thread)

    @property
    def directory(self) -> Path:
        return self._dir

    # -- layout / recovery ---------------------------------------------

    def _prepare_layout(self) -> None:
        self._dir.mkdir(parents=True, exist_ok=True)
        config_path = self._dir / _DISK_CONFIG
        record = {
            "format": _DISK_VERSION,
            "identity": self.identity,
            "identity_digest": self.identity_digest.hex(),
            "page_bytes": self.page_bytes,
            "page_size": self.page_size,
            "rows": self.rows,
            "num_kv_heads": self.num_kv_heads,
            "head_dim": self.head_dim,
            "dtype": str(self.dtype),
        }
        if config_path.exists():
            try:
                current = json.loads(config_path.read_text("utf-8"))
            except (OSError, ValueError) as exc:
                raise KVOffloadError(
                    f"disk KV cache config is unreadable: {config_path}"
                ) from exc
            if current != record:
                raise KVOffloadError(
                    "disk KV cache directory belongs to a different runtime "
                    f"configuration: {self._dir}"
                )
        else:
            temporary = config_path.with_suffix(".tmp")
            temporary.write_text(json.dumps(record, sort_keys=True), "utf-8")
            os.replace(temporary, config_path)

    def _scan(self) -> None:
        """Rebuild the LRU index from immutable page files."""

        entries: list[tuple[int, bytes, str, int]] = []
        for current_root, _dirs, files in os.walk(self._dir):
            for name in files:
                if name.endswith(".tmp"):
                    # Atomic writes leave temp files only after a crash.
                    _unlink_quietly(os.path.join(current_root, name))
                    continue
                if not name.endswith(_DISK_SUFFIX):
                    continue
                path = os.path.join(current_root, name)
                try:
                    stat = os.stat(path)
                    header = _read_header(
                        path,
                        self.page_size,
                        self.identity_digest,
                    )
                except (OSError, ValueError):
                    continue
                if header is None:
                    _unlink_quietly(path)
                    continue
                digest, _tokens, nbytes = header
                if (
                    nbytes != self.page_bytes
                    or stat.st_size
                    != self.page_bytes + _header_size(self.page_size)
                    or name[: -len(_DISK_SUFFIX)] != digest.hex()
                ):
                    _unlink_quietly(path)
                    continue
                entries.append(
                    (
                        max(stat.st_atime_ns, stat.st_mtime_ns),
                        digest,
                        path,
                        nbytes,
                    )
                )
        entries.sort()
        for accessed, digest, path, nbytes in entries:
            self._entries[digest] = _DiskEntry(
                digest=digest,
                path=path,
                nbytes=nbytes,
                accessed_ns=accessed,
            )
            self._bytes += nbytes
        while self._bytes > self.capacity_bytes and self._entries:
            self._evict_locked()

    # -- lookup ---------------------------------------------------------

    def lookup_chain(self, chain: KVKeyChain, start: int) -> int:
        end = start
        with self._lock:
            while end < len(chain):
                if chain.digest(end) not in self._entries:
                    break
                end += 1
            now = time.monotonic_ns()
            for index in range(start, end):
                digest = chain.digest(index)
                self._entries.move_to_end(digest)
                self._entries[digest].accessed_ns = now
        self._lookup_pages += end - start
        self._hit_pages += end - start
        if end == start:
            self._misses += 1
        else:
            self._hits += 1
        return end

    # -- reads ----------------------------------------------------------

    def submit_load(
        self,
        chain: KVKeyChain,
        start: int,
        end: int,
        slots: Sequence[int],
    ) -> dict[int, int]:
        """Submit disk→host page loads; returns ``job_id → page index``."""

        if end - start != len(slots):
            raise KVOffloadError("disk load jobs must pair pages with slots")
        jobs: dict[int, int] = {}
        for index, slot in zip(range(start, end), slots, strict=True):
            digest = chain.digest(index)
            tokens = chain.tokens(index)
            with self._lock:
                entry = self._entries.get(digest)
                if entry is None:
                    continue
                entry.accessed_ns = time.monotonic_ns()
                self._entries.move_to_end(digest)
            job_id = self._next_job()
            try:
                self._load_queue.put_nowait(
                    _DiskJob(job_id, "load", digest, tokens, slot)
                )
            except queue.Full:
                self._load_failures += 1
                self._completions.put(
                    DiskCompletion("load", job_id, False, digest)
                )
                self._notify()
                continue
            with self._lock:
                self._inflight.add(job_id)
            jobs[job_id] = index
        return jobs

    # -- writes ---------------------------------------------------------

    def submit_store(self, digests: Sequence[bytes]) -> dict[int, bytes]:
        """Submit host→disk page writes; returns ``job_id → digest``."""

        jobs: dict[int, bytes] = {}
        for digest in digests:
            with self._lock:
                if digest in self._entries:
                    self._entries.move_to_end(digest)
                    self.host.unpin((digest,))
                    continue
                location = self.host.entry_location(digest)
                if location is None:
                    self.host.unpin((digest,))
                    continue
                slot, tokens = location
            job_id = self._next_job()
            try:
                self._store_queue.put_nowait(
                    _DiskJob(job_id, "store", digest, tokens, slot)
                )
            except queue.Full:
                self._store_failures += 1
                self.host.unpin((digest,))
                continue
            with self._lock:
                self._inflight.add(job_id)
            jobs[job_id] = digest
        return jobs

    # -- polling --------------------------------------------------------

    def poll(self) -> tuple[DiskCompletion, ...]:
        completed: list[DiskCompletion] = []
        while True:
            try:
                completed.append(self._completions.get_nowait())
            except queue.Empty:
                break
        return tuple(completed)

    def wait_for_completion(self, timeout: float) -> None:
        with self._condition:
            self._condition.wait(timeout)

    def _notify(self) -> None:
        with self._condition:
            self._condition.notify_all()

    def _next_job(self) -> int:
        with self._lock:
            self._jobs += 1
            return self._jobs

    # -- workers --------------------------------------------------------

    def _read_loop(self) -> None:
        while not self._stop.is_set():
            try:
                job = self._load_queue.get(timeout=0.05)
            except queue.Empty:
                continue
            success = False
            try:
                success = self._load(job)
            except Exception:  # noqa: BLE001 - a daemon worker must survive
                success = False
            with self._lock:
                self._inflight.discard(job.job_id)
            if not success:
                self._load_failures += 1
            self._completions.put(
                DiskCompletion("load", job.job_id, success, job.digest)
            )
            self._notify()
            self._load_queue.task_done()

    def _write_loop(self) -> None:
        while not self._stop.is_set():
            try:
                job = self._store_queue.get(timeout=0.05)
            except queue.Empty:
                continue
            success = False
            try:
                success = self._store(job)
            except Exception:  # noqa: BLE001 - a daemon worker must survive
                success = False
            with self._lock:
                self._inflight.discard(job.job_id)
            if success:
                with self._lock:
                    self._stores += 1
                    self._store_pages += 1
                    self._store_bytes += self.page_bytes
            else:
                self._store_failures += 1
            self._completions.put(
                DiskCompletion("store", job.job_id, success, job.digest)
            )
            self._notify()
            self._store_queue.task_done()

    def _load(self, job: _DiskJob) -> bool:
        with self._lock:
            entry = self._entries.get(job.digest)
        if entry is None:
            return False
        try:
            with open(entry.path, "rb"):
                pass
        except OSError:
            self._drop_disk_entry(entry)
            return False
        header = _read_header(entry.path, self.page_size, self.identity_digest)
        if header is None:
            self._drop_disk_entry(entry)
            return False
        digest, tokens, nbytes = header
        if (
            digest != job.digest
            or tokens != job.tokens
            or nbytes != self.page_bytes
        ):
            self._drop_disk_entry(entry)
            return False
        try:
            with open(entry.path, "rb") as handle:
                handle.seek(_header_size(self.page_size))
                payload = handle.read()
        except OSError:
            return False
        if (
            len(payload) != nbytes
            or zlib.crc32(payload)
            != _read_crc(entry.path, self.page_size)
        ):
            self._drop_disk_entry(entry)
            return False
        key_payload = payload[: nbytes // 2]
        value_payload = payload[nbytes // 2 :]
        key_view = self.host.key[job.slot].view(tensorplay.uint8)
        key_view.copy_(_uint8_tensor(key_payload, key_view.shape))
        value_view = self.host.value[job.slot].view(tensorplay.uint8)
        value_view.copy_(_uint8_tensor(value_payload, value_view.shape))
        self.host.complete_load(job.digest, True)
        return True

    def _store(self, job: _DiskJob) -> bool:
        with self._lock:
            if job.digest in self._entries:
                return True
            self._make_room_locked(self.page_bytes)
            if self._bytes + self.page_bytes > self.capacity_bytes:
                return False
        header = _read_header(
            self._path_for(job.digest).as_posix(),
            self.page_size,
            self.identity_digest,
        )
        if header is not None:
            return True
        location = self.host.entry_location(job.digest)
        if location is None or location[0] != job.slot:
            return False
        key_payload = (
            self.host.key[job.slot].view(tensorplay.uint8).numpy().tobytes()
        )
        value_payload = (
            self.host.value[job.slot].view(tensorplay.uint8).numpy().tobytes()
        )
        payload = key_payload + value_payload
        if len(payload) != self.page_bytes:
            return False
        blob = _build_header(
            identity_digest=self.identity_digest,
            digest=job.digest,
            tokens=job.tokens,
            page_size=self.page_size,
            rows=self.rows,
            num_kv_heads=self.num_kv_heads,
            head_dim=self.head_dim,
            nbytes=self.page_bytes,
            crc=zlib.crc32(payload),
        ) + payload
        path = self._path_for(job.digest)
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(
            f".{job.digest.hex()}.{os.getpid()}.{threading.get_ident()}.tmp"
        )
        try:
            with open(temporary, "wb") as handle:
                handle.write(blob)
            os.replace(temporary, path)
        except OSError:
            _unlink_quietly(temporary)
            return False
        with self._lock:
            if job.digest not in self._entries:
                self._entries[job.digest] = _DiskEntry(
                    digest=job.digest,
                    path=str(path),
                    nbytes=self.page_bytes,
                    accessed_ns=time.monotonic_ns(),
                )
                self._bytes += self.page_bytes
            self._trim_locked()
        return True

    # -- capacity -------------------------------------------------------

    def _path_for(self, digest: bytes) -> Path:
        hexed = digest.hex()
        return self._dir / hexed[:2] / hexed[2:4] / f"{hexed}{_DISK_SUFFIX}"

    def _make_room_locked(self, nbytes: int) -> None:
        while self._bytes + nbytes > self.capacity_bytes and self._entries:
            if not self._evict_locked():
                break

    def _evict_locked(self) -> bool:
        for digest, entry in list(self._entries.items()):
            del self._entries[digest]
            self._bytes -= entry.nbytes
            self._evictions += 1
            self._evicted_pages += 1
            _unlink_quietly(entry.path)
            return True
        return False

    def _trim_locked(self) -> None:
        while self._bytes > self.capacity_bytes and self._entries:
            if not self._evict_locked():
                break

    def _drop_disk_entry(self, entry: _DiskEntry) -> None:
        with self._lock:
            current = self._entries.get(entry.digest)
            if current is entry:
                del self._entries[entry.digest]
                self._bytes -= entry.nbytes
        _unlink_quietly(entry.path)

    # -- lifecycle / stats ---------------------------------------------

    def close(self) -> None:
        self._stop.set()
        for thread in self._threads:
            thread.join(timeout=1.0)

    def stats(self) -> DiskTierStats:
        with self._lock:
            return DiskTierStats(
                capacity_bytes=self.capacity_bytes,
                used_bytes=self._bytes,
                entries=len(self._entries),
                stores=self._stores,
                store_pages=self._store_pages,
                store_bytes=self._store_bytes,
                evictions=self._evictions,
                evicted_pages=self._evicted_pages,
                hits=self._hits,
                misses=self._misses,
                lookup_pages=self._lookup_pages,
                hit_pages=self._hit_pages,
                store_failures=self._store_failures,
                load_failures=self._load_failures,
                inflight_jobs=len(self._inflight),
            )


def _identity_directory(identity: str) -> str:
    digest = hashlib.blake2b(identity.encode(), digest_size=8).hexdigest()
    safe = "".join(
        char if char.isascii() and char.isalnum() else "_"
        for char in identity[:64]
    )
    return f"{safe}_{digest}"


def _header_size(page_size: int) -> int:
    return _DISK_HEADER.size + page_size * 4


def _build_header(
    *,
    identity_digest: bytes,
    digest: bytes,
    tokens: Sequence[int],
    page_size: int,
    rows: int,
    num_kv_heads: int,
    head_dim: int,
    nbytes: int,
    crc: int,
) -> bytes:
    fixed = _DISK_HEADER.pack(
        _DISK_MAGIC,
        _DISK_VERSION,
        rows,
        page_size,
        num_kv_heads,
        head_dim,
        page_size,
        identity_digest,
        digest,
        nbytes,
        crc,
        len(tokens),
    )
    return fixed + struct.pack(f"<{page_size}i", *tokens)


def _read_header(
    path: str,
    page_size: int,
    identity_digest: bytes,
) -> tuple[bytes, tuple[int, ...], int] | None:
    try:
        with open(path, "rb") as handle:
            raw = handle.read(_header_size(page_size))
    except OSError:
        return None
    if len(raw) != _header_size(page_size):
        return None
    (
        magic,
        version,
        rows,
        header_page_size,
        num_kv_heads,
        head_dim,
        token_stride,
        header_identity,
        digest,
        nbytes,
        _crc,
        token_count,
    ) = _DISK_HEADER.unpack(raw[: _DISK_HEADER.size])
    if (
        magic != _DISK_MAGIC
        or version != _DISK_VERSION
        or header_page_size != page_size
        or token_stride != page_size
        or token_count != page_size
        or rows <= 0
        or num_kv_heads <= 0
        or head_dim <= 0
        or header_identity != identity_digest
        or len(digest) != 32
    ):
        return None
    tokens = struct.unpack(f"<{page_size}i", raw[_DISK_HEADER.size :])
    return digest, tuple(int(token) for token in tokens), int(nbytes)


def _read_crc(path: str, page_size: int) -> int:
    try:
        with open(path, "rb") as handle:
            handle.seek(_DISK_HEADER.size - 8)
            raw = handle.read(4)
    except OSError:
        return -1
    return struct.unpack("<I", raw)[0] if len(raw) == 4 else -1


def _uint8_tensor(payload: bytes, shape: object) -> tensorplay.Tensor:
    array = np.frombuffer(payload, dtype=np.uint8).copy()
    array = array.reshape(tuple(shape))  # type: ignore[arg-type]
    return tensorplay.from_numpy(array)


def _unlink_quietly(path: str) -> None:
    try:
        os.unlink(path)
    except OSError:
        pass


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class StoreTicket:
    """GPU pages retained while their host copies are in flight."""

    ticket_id: int
    pages: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class PromotionOutcome:
    request_id: str
    cached_tokens: int
    ready: bool
    failure: str | None = None


@dataclass(frozen=True, slots=True)
class OffloadStats:
    host: HostTierStats | None
    disk: DiskTierStats | None
    promotions: int
    promotion_pages: int
    promotion_failures: int
    pending_promotions: int
    pending_stores: int
    promotion_wait_ns: int

    def snapshot(self) -> dict[str, object]:
        return {
            "host": None if self.host is None else self.host.snapshot(),
            "disk": None if self.disk is None else self.disk.snapshot(),
            "promotions": self.promotions,
            "promotion_pages": self.promotion_pages,
            "promotion_failures": self.promotion_failures,
            "pending_promotions": self.pending_promotions,
            "pending_stores": self.pending_stores,
            "promotion_wait_ms": self.promotion_wait_ns / 1_000_000,
        }


@dataclass(slots=True)
class _Promotion:
    request_id: str
    chain: KVKeyChain
    start: int
    end: int
    device_pages: tuple[int, ...]
    device_key: tensorplay.Tensor
    device_value: tensorplay.Tensor
    host_end: int
    state: str = "init"
    slots: list[int] = field(default_factory=list)
    load_jobs: dict[int, int] = field(default_factory=dict)
    load_results: dict[int, bool] = field(default_factory=dict)
    load_reserved: set[int] = field(default_factory=set)
    host_digests: list[bytes] = field(default_factory=list)
    copy: _PairTask | None = None
    failure: str | None = None
    cancelled: bool = False


class TieredKVOffload:
    """Coordinates GPU, host and disk tiers for one replica.

    Leader-owned: it runs on the scheduler thread; only the disk workers
    touch the disk queues.  GPU work is submitted to one dedicated copy
    stream.
    """

    def __init__(
        self,
        *,
        identity: str,
        host: HostKVTier,
        disk: DiskKVTier | None = None,
    ) -> None:
        self.identity = identity
        self.identity_digest = host.identity_digest
        self.host = host
        self.disk = disk
        self._lock = threading.RLock()
        self._promotions: dict[str, _Promotion] = {}
        self._promotion_by_load_job: dict[int, str] = {}
        self._tickets: dict[int, tuple[tuple[int, ...], tuple[bytes, ...], float]] = {}
        self._completed_tickets: list[StoreTicket] = []
        self._pending_disk_stores: dict[int, bytes] = {}
        self._next_ticket = 0
        self._promotions_total = 0
        self._promotion_pages = 0
        self._promotion_failures = 0
        self._promotion_wait_ns = 0

    # -- keys -----------------------------------------------------------

    def chain(
        self,
        *,
        namespace: str,
        page_size: int,
        pages: Sequence[Sequence[int]],
    ) -> KVKeyChain:
        return KVKeyChain(
            identity_digest=self.identity_digest,
            namespace=namespace,
            page_size=page_size,
            pages=pages,
        )

    def chain_from_key(self, key: RadixKey) -> KVKeyChain:
        return self.chain(
            namespace=key.namespace,
            page_size=key.page_size,
            pages=key.pages,
        )

    # -- lookup ---------------------------------------------------------

    def lookup(self, chain: KVKeyChain, start: int) -> int:
        """Longest contiguous prefix available from the host and disk tiers."""

        host_end = self.host.lookup_chain(chain, start)
        if self.disk is None or host_end >= len(chain):
            return host_end
        return self.disk.lookup_chain(chain, host_end)

    # -- stores ---------------------------------------------------------

    def store_prefix(
        self,
        *,
        chain: KVKeyChain,
        start: int,
        end: int,
        device_key: tensorplay.Tensor,
        device_value: tensorplay.Tensor,
        device_pages: Sequence[int],
    ) -> StoreTicket | None:
        if end <= start or end > len(device_pages):
            return None
        first, digests = self.host.submit_store(
            chain,
            start,
            end,
            device_key=device_key,
            device_value=device_value,
            device_pages=device_pages,
        )
        if not digests:
            return None
        pages = tuple(device_pages[first : first + len(digests)])
        with self._lock:
            self._next_ticket += 1
            ticket = StoreTicket(self._next_ticket, pages)
            self._tickets[ticket.ticket_id] = (
                pages,
                tuple(digests),
                time.monotonic(),
            )
        return ticket

    def pop_completed_tickets(self) -> tuple[StoreTicket, ...]:
        with self._lock:
            tickets, self._completed_tickets = self._completed_tickets, []
        return tuple(tickets)

    # -- promotions -----------------------------------------------------

    def begin_promotion(
        self,
        *,
        request_id: str,
        chain: KVKeyChain,
        start: int,
        end: int,
        device_pages: Sequence[int],
        device_key: tensorplay.Tensor,
        device_value: tensorplay.Tensor,
    ) -> bool:
        """Start promoting pages ``start..end`` into the request's pages."""

        if end <= start:
            return False
        with self._lock:
            if request_id in self._promotions:
                raise KVOffloadError(
                    "request already has a pending promotion"
                )
        job = _Promotion(
            request_id=request_id,
            chain=chain,
            start=start,
            end=min(end, len(chain)),
            device_pages=tuple(
                device_pages[: min(end, len(chain)) - start]
            ),
            device_key=device_key,
            device_value=device_value,
            host_end=start,
        )
        if len(job.device_pages) != job.end - job.start:
            return False
        with self._lock:
            self._promotions[request_id] = job
            self._promotions_total += 1
            self._promotion_pages += job.end - job.start
        try:
            self._start(job)
        except Exception:  # noqa: BLE001 - any setup error is a cache miss
            self._fail(job, "promotion setup failed")
        return True

    def _start(self, job: _Promotion) -> None:
        job.host_end = self.host.lookup_chain(job.chain, job.start)
        if job.host_end < job.end:
            if self.disk is None:
                job.end = job.host_end
            else:
                disk_end = self.disk.lookup_chain(job.chain, job.host_end)
                job.end = min(job.end, disk_end)
                if job.end > job.host_end:
                    slots = self.host.prepare_load(
                        job.chain,
                        job.host_end,
                        job.end,
                    )
                    if slots:
                        job.slots = slots
                        job.load_reserved = set(
                            range(job.host_end, job.end)
                        )
                        job.load_jobs = self.disk.submit_load(
                            job.chain,
                            job.host_end,
                            job.end,
                            slots,
                        )
                        expected = job.end - job.host_end
                        if len(job.load_jobs) < expected:
                            # A full queue or a lost file shortened the
                            # batch; promote only the contiguous submitted
                            # prefix.
                            submitted = set(job.load_jobs.values())
                            first_missing = next(
                                index
                                for index in range(job.host_end, job.end)
                                if index not in submitted
                            )
                            for index in range(first_missing, job.end):
                                self.host.complete_load(
                                    job.chain.digest(index),
                                    False,
                                )
                                job.load_reserved.discard(index)
                            job.end = first_missing
                        with self._lock:
                            for job_id in job.load_jobs:
                                self._promotion_by_load_job[job_id] = (
                                    job.request_id
                                )
                        if not job.load_jobs:
                            job.end = job.host_end
                        else:
                            job.state = "disk"
                            return
                    else:
                        job.end = job.host_end
        if job.end <= job.start:
            self._fail(job, "no page available for promotion")
            return
        self._start_device_copy(job)

    def _start_device_copy(self, job: _Promotion) -> None:
        digests = [
            job.chain.digest(index) for index in range(job.start, job.end)
        ]
        if not self.host.pin(digests):
            self._fail(job, "host page vanished")
            return
        job.host_digests = digests
        job.copy = self.host.copy_to_device(
            digests,
            job.device_key,
            job.device_value,
            job.device_pages,
        )
        job.state = "gpu"

    # -- polling --------------------------------------------------------

    def poll(
        self,
        *,
        device_key: tensorplay.Tensor,
        device_value: tensorplay.Tensor,
        block: bool = False,
        timeout_s: float = 0.005,
    ) -> tuple[tuple[PromotionOutcome, ...], tuple[StoreTicket, ...]]:
        """Advance all offload work and return terminal outcomes/tickets."""

        started = time.monotonic_ns()
        if block:
            deadline = started + int(timeout_s * 1_000_000_000)
            while True:
                outcomes, tickets = self._poll_once(
                    device_key,
                    device_value,
                )
                if outcomes or tickets:
                    break
                if time.monotonic_ns() >= deadline:
                    break
                if self.disk is not None:
                    with self._lock:
                        pending_loads = bool(self._promotion_by_load_job)
                    if pending_loads:
                        self.disk.wait_for_completion(0.001)
                    else:
                        time.sleep(0.0002)
                else:
                    time.sleep(0.0002)
            self._promotion_wait_ns += time.monotonic_ns() - started
            return outcomes, tickets
        outcomes, tickets = self._poll_once(device_key, device_value)
        self._promotion_wait_ns += time.monotonic_ns() - started
        return outcomes, tickets

    def _poll_once(
        self,
        device_key: tensorplay.Tensor,
        device_value: tensorplay.Tensor,
    ) -> tuple[tuple[PromotionOutcome, ...], tuple[StoreTicket, ...]]:
        stored = self.host.poll()
        self._complete_tickets(stored)
        if stored and self.disk is not None and self.host.pin(stored):
            jobs = self.disk.submit_store(stored)
            with self._lock:
                self._pending_disk_stores.update(jobs)
        if self.disk is not None:
            self._route_disk_completions(self.disk.poll())
        outcomes: list[PromotionOutcome] = []
        with self._lock:
            jobs = tuple(self._promotions.values())
        for job in jobs:
            if job.state == "gpu":
                assert job.copy is not None
                if not job.copy.ready():
                    continue
                job.copy.attach()
                outcomes.append(self._finish(job, ready=not job.cancelled))
            elif job.state in {"failed", "done"}:
                outcomes.append(self._finish(job, ready=False))
        with self._lock:
            completed = tuple(self._completed_tickets)
            self._completed_tickets.clear()
        return tuple(outcomes), completed

    def _route_disk_completions(
        self,
        completions: Sequence[DiskCompletion],
    ) -> None:
        for completion in completions:
            if completion.kind == "store":
                with self._lock:
                    digest = self._pending_disk_stores.pop(
                        completion.job_id, None
                    )
                if digest is not None:
                    self.host.unpin((digest,))
                continue
            with self._lock:
                request_id = self._promotion_by_load_job.pop(
                    completion.job_id, None
                )
                job = (
                    None
                    if request_id is None
                    else self._promotions.get(request_id)
                )
                index = (
                    None
                    if job is None
                    else job.load_jobs.get(completion.job_id)
                )
            if job is None or index is None:
                if completion.success:
                    self.host.complete_load(completion.digest, True)
                else:
                    self.host.complete_load(completion.digest, False)
                continue
            self.host.complete_load(completion.digest, completion.success)
            job.load_results[index] = completion.success
            needed = job.end - job.host_end
            if len(job.load_results) < needed:
                continue
            if job.cancelled:
                self._fail(job, "promotion cancelled")
                job.load_jobs.clear()
                continue
            self._after_disk_loads(job)

    def _after_disk_loads(self, job: _Promotion) -> None:
        if job.cancelled:
            self._fail(job, "promotion cancelled")
            return
        failed = [
            index
            for index in range(job.host_end, job.end)
            if not job.load_results.get(index)
        ]
        if failed:
            job.end = min(failed)
        if job.end <= job.start:
            self._fail(job, "disk load failed")
            return
        for index in sorted(job.load_reserved):
            if index >= job.end:
                if job.load_results.get(index):
                    # A successfully loaded page beyond the promoted prefix
                    # is released from the load reservation but stays cached.
                    self.host.unpin((job.chain.digest(index),))
                job.load_reserved.discard(index)
        job.state = "init"
        job.slots = []
        job.load_jobs = {}
        job.load_results = {}
        # Host pages in [start, end) are ready; re-enter the copy phase.
        self._start_device_copy(job)

    # -- bookkeeping ----------------------------------------------------

    def _complete_tickets(self, stored: Sequence[bytes]) -> tuple[StoreTicket, ...]:
        if not stored:
            return ()
        stored_set = set(stored)
        completed: list[StoreTicket] = []
        with self._lock:
            for ticket_id, (pages, digests, _queued) in tuple(
                self._tickets.items()
            ):
                if stored_set.issuperset(digests):
                    del self._tickets[ticket_id]
                    completed.append(StoreTicket(ticket_id, pages))
            self._completed_tickets.extend(completed)
        return tuple(completed)

    def _finish(self, job: _Promotion, *, ready: bool) -> PromotionOutcome:
        with self._lock:
            if self._promotions.pop(job.request_id, None) is None:
                return PromotionOutcome(
                    request_id=job.request_id,
                    cached_tokens=job.start * job.chain.page_size,
                    ready=False,
                )
            for job_id in job.load_jobs:
                self._promotion_by_load_job.pop(job_id, None)
        if job.copy is not None:
            job.copy.wait()
        if job.host_digests:
            self.host.unpin(job.host_digests)
        if not ready:
            # Free slots reserved for disk loads that never became part of
            # the promoted prefix.
            for index in sorted(job.load_reserved):
                self.host.complete_load(job.chain.digest(index), False)
            job.load_reserved.clear()
        if ready:
            return PromotionOutcome(
                request_id=job.request_id,
                cached_tokens=job.end * job.chain.page_size,
                ready=True,
            )
        self._promotion_failures += 1
        return PromotionOutcome(
            request_id=job.request_id,
            cached_tokens=job.start * job.chain.page_size,
            ready=False,
            failure=job.failure,
        )

    def _fail(self, job: _Promotion, reason: str) -> None:
        job.failure = reason
        job.state = "failed"

    def cancel(self, request_id: str, *, wait: bool = True) -> None:
        """Cancel a promotion; GPU writes are drained before returning."""

        with self._lock:
            job = self._promotions.get(request_id)
            if job is None:
                return
            job.cancelled = True
        if wait:
            if job.copy is not None:
                job.copy.wait()
            if job.state == "disk" and self.disk is not None:
                deadline = time.monotonic() + 30.0
                while (
                    time.monotonic() < deadline
                    and job.load_jobs
                    and job.state == "disk"
                ):
                    self._route_disk_completions(self.disk.poll())
                    if job.load_jobs:
                        self.disk.wait_for_completion(0.001)
        self._finish(job, ready=False)

    def pending(self, request_id: str) -> bool:
        with self._lock:
            return request_id in self._promotions

    @property
    def has_pending(self) -> bool:
        with self._lock:
            return bool(self._promotions)

    # -- stats / lifecycle ---------------------------------------------

    def stats(self) -> OffloadStats:
        with self._lock:
            pending_promotions = len(self._promotions)
            pending_stores = len(self._tickets)
        return OffloadStats(
            host=self.host.stats(),
            disk=None if self.disk is None else self.disk.stats(),
            promotions=self._promotions_total,
            promotion_pages=self._promotion_pages,
            promotion_failures=self._promotion_failures,
            pending_promotions=pending_promotions,
            pending_stores=pending_stores,
            promotion_wait_ns=self._promotion_wait_ns,
        )

    def close(self) -> None:
        if self.disk is not None:
            self.disk.close()


__all__ = [
    "CopyTask",
    "DiskKVTier",
    "DiskTierStats",
    "HostKVTier",
    "HostTierStats",
    "KVKeyChain",
    "KVOffloadError",
    "OffloadStats",
    "PromotionOutcome",
    "StoreTicket",
    "TensorCopier",
    "TieredKVOffload",
]