from __future__ import annotations

import hashlib
import math
import struct
import threading
from collections import OrderedDict
from collections.abc import Callable
from dataclasses import asdict, dataclass, field

import tensorplay

from megalm.contracts.cache import (
    CacheAllocation,
    CacheCapacityError,
    CacheHandle,
    CacheKind,
    LogicalCacheDirectory,
)
from megalm.runtime.kv_offload import (
    KVKeyChain,
    PromotionOutcome,
    TieredKVOffload,
)
from megalm.runtime.radix_cache import (
    RadixCacheHandle,
    RadixCacheStats,
    RadixInsertResult,
    RadixKey,
    RadixKeyInput,
    RadixPrefixCache,
)


class KVCacheError(RuntimeError):
    """Raised when logical page ownership and physical cache metadata diverge."""


@dataclass(frozen=True, slots=True)
class CacheGeometry:
    num_layers: int
    num_pages: int
    page_size: int
    num_kv_heads: int
    head_dim: int

    @property
    def token_capacity(self) -> int:
        return self.num_pages * self.page_size


@dataclass(frozen=True, slots=True)
class CacheAdmission:
    handle: CacheHandle
    cached_tokens: int
    page_ids: tuple[int, ...]
    pending_promotion: bool = False


@dataclass(frozen=True, slots=True)
class CacheStats:
    total_pages: int
    free_pages: int
    reclaimable_pages: int
    allocated_pages: int
    active_requests: int
    request_page_refs: int
    prefix: RadixCacheStats | None
    offload: dict[str, object] | None = None
    # GDN hybrid-state slot utilization (0 when the model has no GDN layers).
    gdn_slots_total: int = 0
    gdn_slots_used: int = 0
    capacity_failures: dict[str, int] = field(default_factory=dict)

    @property
    def gdn_utilization(self) -> float:
        return (
            0.0
            if self.gdn_slots_total == 0
            else self.gdn_slots_used / self.gdn_slots_total
        )

    @property
    def utilization(self) -> float:
        return (
            0.0
            if self.total_pages == 0
            else self.allocated_pages / self.total_pages
        )

    def snapshot(self) -> dict[str, object]:
        """Serialize raw counters and computed cache health metrics."""

        payload: dict[str, object] = {
            **asdict(self),
            "utilization": self.utilization,
            "gdn_utilization": self.gdn_utilization,
        }
        if self.prefix is not None:
            payload["prefix"] = self.prefix.snapshot()
        return payload


class PrefixStateStore:
    """Bounded recurrent-state checkpoints keyed by exact token pages.

    A radix hit covers K/V pages only.  Inkling-2 also carries short
    convolution history that never enters the paged arena, so a page hit is
    only safe when the convolution state at the matched boundary was captured.
    ``capture(request_id, token_count)`` returns an opaque payload (or
    ``None`` when the state cannot be taken); ``restore(request_id,
    token_count, payload, slot)`` writes it back into a request state slot.

    Entries are content addressed by the exact pages of the prefix, so a
    checkpoint stays valid across radix evictions and re-insertions, and two
    requests that share a prefix always resolve to the same entry.
    """

    __slots__ = (
        "_bytes",
        "_capture",
        "_entries",
        "_restore",
        "hits",
        "max_bytes",
        "misses",
    )

    def __init__(
        self,
        capture: Callable[[str, int], object | None],
        restore: Callable[[str, int, object, int], None],
        *,
        max_bytes: int = 1 << 30,
    ) -> None:
        if max_bytes <= 0:
            raise KVCacheError("prefix state budget must be positive")
        self._capture = capture
        self._restore = restore
        self._entries: OrderedDict[
            tuple[str, int, bytes], tuple[object, int]
        ] = OrderedDict()
        self._bytes = 0
        self.max_bytes = int(max_bytes)
        self.hits = 0
        self.misses = 0

    @property
    def nbytes(self) -> int:
        return self._bytes

    def __len__(self) -> int:
        return len(self._entries)

    @staticmethod
    def _digests(key: RadixKey, pages: int) -> list[bytes]:
        digest = hashlib.blake2b(digest_size=16)
        digest.update(key.namespace.encode())
        digest.update(struct.pack("<q", key.page_size))
        chain: list[bytes] = []
        for page in key.pages[:pages]:
            digest.update(struct.pack(f"<{len(page)}q", *page))
            chain.append(digest.copy().digest())
        return chain

    @classmethod
    def _key(
        cls,
        key: RadixKey,
        token_count: int,
    ) -> tuple[str, int, bytes]:
        if token_count % key.page_size:
            raise KVCacheError("prefix state must align with full pages")
        pages = token_count // key.page_size
        if pages <= 0 or pages > len(key.pages):
            raise KVCacheError("prefix state token count is out of range")
        return (key.namespace, token_count, cls._digests(key, pages)[-1])

    @staticmethod
    def _payload_bytes(payload: object) -> int:
        if isinstance(payload, dict):
            total = 0
            for item in payload.values():
                try:
                    total += int(item.numel()) * int(item.element_size())
                except (AttributeError, TypeError):
                    total += 1024
            return total
        try:
            return int(payload.numel()) * int(payload.element_size())  # type: ignore[attr-defined]
        except (AttributeError, TypeError):
            return 1024

    def snapshot(
        self,
        request_id: str,
        key: RadixKey,
        token_count: int,
    ) -> bool:
        """Capture the recurrent state at ``token_count`` and retain it."""

        payload = self._capture(request_id, token_count)
        if payload is None:
            self.misses += 1
            return False
        entry_key = self._key(key, token_count)
        size = self._payload_bytes(payload)
        previous = self._entries.pop(entry_key, None)
        if previous is not None:
            self._bytes -= previous[1]
        self._entries[entry_key] = (payload, size)
        self._bytes += size
        while self._bytes > self.max_bytes and len(self._entries) > 1:
            _, (_, evicted) = self._entries.popitem(last=False)
            self._bytes -= evicted
        return True

    def has(self, key: RadixKey, token_count: int) -> bool:
        return self._key(key, token_count) in self._entries

    def longest(self, key: RadixKey, *, below: int | None = None) -> int:
        """Longest page-aligned checkpoint not exceeding ``below``."""

        limit = key.token_count if below is None else min(below - 1, key.token_count)
        pages = limit // key.page_size
        if pages <= 0 or not self._entries:
            return 0
        chain = self._digests(key, pages)
        for count in range(pages, 0, -1):
            probe = (key.namespace, count * key.page_size, chain[count - 1])
            if probe in self._entries:
                return count * key.page_size
        return 0

    def restore(
        self,
        request_id: str,
        key: RadixKey,
        token_count: int,
        slot: int,
    ) -> bool:
        entry = self._entries.get(self._key(key, token_count))
        if entry is None:
            self.misses += 1
            return False
        self._entries.move_to_end(self._key(key, token_count))
        self._restore(request_id, token_count, entry[0], slot)
        self.hits += 1
        return True


class PagedKVCache:
    """Stable-address physical K/V arena owned by one GPU executor."""

    def __init__(
        self,
        *,
        num_layers: int,
        num_pages: int,
        page_size: int,
        num_kv_heads: int,
        head_dim: int,
        dtype: tensorplay.dtype,
        device: tensorplay.device | str,
        layer_indices: tuple[int, ...] | None = None,
    ) -> None:
        values = (num_layers, num_pages, page_size, num_kv_heads, head_dim)
        if any(value <= 0 for value in values):
            raise KVCacheError("all paged-cache dimensions must be positive")
        self.geometry = CacheGeometry(
            num_layers=num_layers,
            num_pages=num_pages,
            page_size=page_size,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
        )
        physical_layers = (
            tuple(range(num_layers))
            if layer_indices is None
            else tuple(layer_indices)
        )
        if (
            not physical_layers
            or len(set(physical_layers)) != len(physical_layers)
            or any(
                index < 0 or index >= num_layers
                for index in physical_layers
            )
        ):
            raise KVCacheError("paged-cache layer mapping is invalid")
        self.layer_indices = physical_layers
        self._layer_rows = {
            layer_index: row
            for row, layer_index in enumerate(physical_layers)
        }
        shape = (
            len(physical_layers),
            num_pages,
            page_size,
            num_kv_heads,
            head_dim,
        )
        self.key = tensorplay.zeros(shape, dtype=dtype, device=device)
        self.value = tensorplay.zeros_like(self.key)

    @property
    def nbytes(self) -> int:
        return (
            self.key.numel() * self.key.element_size()
            + self.value.numel() * self.value.element_size()
        )

    @property
    def device(self) -> tensorplay.device:
        return self.key.device

    def layer(self, layer_index: int) -> tuple[tensorplay.Tensor, tensorplay.Tensor]:
        try:
            row = self._layer_rows[layer_index]
        except KeyError as exc:
            raise KVCacheError(f"invalid cache layer {layer_index}")
        return self.key[row], self.value[row]

    def register_conv_ring(
        self,
        layer_index: int,
        *,
        channels: int,
        kernel_size: int,
    ) -> None:
        """Allocate one short-convolution ring; called by the Inkling-2 executor.

        Inkling-2 keeps ``kernel - 1`` previous inputs per channel per
        (layer, pass).  The rings are not part of the paged KV arena.
        """

        if not hasattr(self, "_conv_rings"):
            object.__setattr__(self, "_conv_rings", {})
        self._conv_rings[layer_index] = tensorplay.zeros(
            (
                self.geometry.num_pages * self.geometry.page_size,
                channels,
                kernel_size - 1,
            ),
            dtype=self.key.dtype,
            device=self.key.device,
        )

    def conv_state(self, layer_index: int) -> tensorplay.Tensor:
        """The ring registered by :meth:`register_conv_ring`."""

        rings = getattr(self, "_conv_rings", {})
        try:
            return rings[layer_index]
        except KeyError as exc:
            raise KVCacheError(
                f"no convolution ring configured for layer {layer_index}"
            ) from exc


class KVCacheManager:
    """Leader-owned transactional page directory plus optional Radix Cache."""

    def __init__(
        self,
        *,
        num_pages: int,
        page_size: int,
        max_sequences: int,
        max_model_len: int,
        device: tensorplay.device | str,
        enable_prefix_cache: bool = True,
        gdn_state_slots: int = 0,
        gdn_states_per_request: int = 1,
        vision_embedding_slots: int = 0,
        media_position_slots: int = 0,
        prefix_state: PrefixStateStore | None = None,
        offload: TieredKVOffload | None = None,
        offload_key: tensorplay.Tensor | None = None,
        offload_value: tensorplay.Tensor | None = None,
    ) -> None:
        if min(num_pages, page_size, max_sequences, max_model_len) <= 0:
            raise KVCacheError("cache-manager dimensions must be positive")
        if (offload is None) != (offload_key is None and offload_value is None):
            raise KVCacheError("offload requires both GPU K and V arenas")
        self.page_size = page_size
        self.max_model_len = max_model_len
        self.max_blocks_per_sequence = math.ceil(max_model_len / page_size)
        self.device = tensorplay.device(device)
        self.enable_prefix_cache = bool(enable_prefix_cache)
        if offload is not None and not self.enable_prefix_cache:
            raise KVCacheError("offload requires the radix prefix cache")
        if min(
            gdn_state_slots,
            vision_embedding_slots,
            media_position_slots,
        ) < 0:
            raise KVCacheError("state-cache capacities cannot be negative")
        if gdn_states_per_request not in {1, 2}:
            raise KVCacheError(
                "GDN state copies per request must be one or two"
            )
        if (
            gdn_state_slots
            and gdn_state_slots
            < max_sequences * gdn_states_per_request
        ):
            raise KVCacheError(
                "GDN state capacity must cover every request slot"
            )
        if gdn_state_slots and self.enable_prefix_cache:
            raise KVCacheError(
                "hybrid prefix cache requires aligned state checkpoints"
            )
        self._uses_gdn_state = gdn_state_slots > 0
        self._gdn_states_per_request = gdn_states_per_request
        self._gdn_active_copy: dict[str, int] = {}
        self._lock = threading.RLock()
        self._directory = LogicalCacheDirectory(
            {
                CacheKind.KV_PAGE: num_pages,
                CacheKind.GDN_CONV_STATE: gdn_state_slots,
                CacheKind.GDN_RECURRENT_STATE: gdn_state_slots,
                CacheKind.PREFIX_CHECKPOINT: 0,
                CacheKind.VISION_EMBEDDING: vision_embedding_slots,
                CacheKind.MEDIA_POSITION: media_position_slots,
                CacheKind.REQUEST_SLOT: max_sequences,
            },
            arena_id="kv-pages",
        )
        self._request_pages: dict[str, tuple[int, ...]] = {}
        self._page_allocation_table = tuple(
            self._directory.allocation(CacheKind.KV_PAGE, page)
            for page in range(num_pages)
        )
        self._prefix_handles: dict[str, RadixCacheHandle] = {}
        self._published_prefix_tokens: dict[str, int] = {}
        self._shared_prefix_tokens: dict[str, int] = {}
        self._prefix_cache = (
            RadixPrefixCache(
                page_size=page_size,
                retain_pages=self._retain_page_ids,
                release_pages=self._release_page_ids,
                trust_page_ownership=True,
            )
            if self.enable_prefix_cache
            else None
        )
        if prefix_state is not None and self._prefix_cache is None:
            raise KVCacheError(
                "prefix state checkpoints require the radix prefix cache"
            )
        if prefix_state is not None and self._uses_gdn_state:
            raise KVCacheError(
                "GDN hybrid state already owns aligned slots"
            )
        self._prefix_state = prefix_state
        self._offload = offload
        self._offload_key = offload_key
        self._offload_value = offload_value
        self._offload_chains: dict[str, KVKeyChain] = {}
        self._promotions: dict[
            str, tuple[int, int, tuple[int, ...], RadixKey]
        ] = {}
        self._store_refs: dict[
            int, tuple[CacheAllocation, ...]
        ] = {}

    @property
    def offload(self) -> TieredKVOffload | None:
        return self._offload

    @property
    def has_pending_promotions(self) -> bool:
        return bool(self._promotions)

    def promotion_pending(self, request_id: str) -> bool:
        return request_id in self._promotions

    @property
    def prefix_state(self) -> PrefixStateStore | None:
        return self._prefix_state

    @property
    def free_pages(self) -> int:
        return self._directory.available(CacheKind.KV_PAGE)

    @property
    def reclaimable_pages(self) -> int:
        prefix = self._prefix_cache
        return 0 if prefix is None else prefix.evictable_pages

    @property
    def available_pages(self) -> int:
        return self.free_pages + self.reclaimable_pages

    @property
    def total_pages(self) -> int:
        return self._directory.capacities[CacheKind.KV_PAGE]

    @property
    def prefix_cache(self) -> RadixPrefixCache | None:
        return self._prefix_cache

    @property
    def uses_gdn_state(self) -> bool:
        return self._uses_gdn_state

    def handle(self, request_id: str) -> CacheHandle | None:
        return self._directory.get(request_id)

    def admit(self, request_id: str) -> CacheHandle:
        with self._lock:
            current = self._directory.get(request_id)
            if current is not None:
                return current
            with self._directory.transaction(request_id) as transaction:
                self._reserve_fixed_state(transaction)
                handle = transaction.commit()
            self._request_pages[request_id] = ()
            self._published_prefix_tokens[request_id] = 0
            self._shared_prefix_tokens[request_id] = 0
            if self._uses_gdn_state:
                self._gdn_active_copy[request_id] = 0
            return handle

    def _offload_target(
        self,
        request_id: str,
        key_or_tokens: RadixKeyInput,
        start_pages: int,
    ) -> tuple[KVKeyChain | None, int]:
        """Longest lower-tier prefix that can safely extend a radix hit."""

        offload = self._offload
        if offload is None or not isinstance(key_or_tokens, RadixKey):
            return None, start_pages
        chain = self._offload_chains.get(request_id)
        if chain is None:
            chain = offload.chain_from_key(key_or_tokens)
            self._offload_chains[request_id] = chain
        target = offload.lookup(chain, start_pages)
        if self._prefix_state is not None:
            # KV alone cannot recreate recurrent/convolution state. Only a
            # page boundary with a matching checkpoint is usable.
            longest = self._prefix_state.longest(
                key_or_tokens,
                below=target * self.page_size + 1,
            )
            target = longest // self.page_size
        return chain, max(start_pages, min(target, len(chain)))

    def _sync_offload_chain(
        self,
        request_id: str,
        key: RadixKey,
    ) -> KVKeyChain | None:
        if self._offload is None:
            return None
        chain = self._offload_chains.get(request_id)
        if chain is None:
            chain = self._offload.chain_from_key(key)
            self._offload_chains[request_id] = chain
        elif len(chain) < len(key.pages):
            chain.extend(key.pages[len(chain) :])
        return chain

    def admit_with_prefix(
        self,
        request_id: str,
        key_or_tokens: RadixKeyInput,
    ) -> CacheAdmission:
        """Atomically pin, attach and expose the longest safe radix hit."""

        with self._lock:
            current = self._directory.get(request_id)
            if current is not None:
                cached_tokens = self._shared_prefix_tokens.get(request_id, 0)
                pages = self._request_pages[request_id][
                    : cached_tokens // self.page_size
                ]
                return CacheAdmission(current, cached_tokens, pages)

            prefix = self._prefix_cache
            state_key = (
                key_or_tokens
                if isinstance(key_or_tokens, RadixKey)
                else None
            )
            if self._prefix_state is not None and state_key is None:
                # A state-carrying model cannot reuse KV without the matching
                # recurrent checkpoint, so raw token lists disable the hit.
                prefix = None
            search_key = key_or_tokens
            match = None
            while prefix is not None:
                match = prefix.match_prefix(
                    search_key,
                    reserve_tokens=1,
                    pin=True,
                )
                matched = 0 if match is None else match.cached_tokens
                if match is None or matched <= 0:
                    match = None
                    break
                if self._prefix_state is None:
                    break
                if self._prefix_state.has(state_key, matched):
                    break
                # The radix hit runs past the newest checkpoint (a longer
                # prefix was evicted or published before the state existed):
                # fall back to the longest shorter boundary that has one.
                prefix.unpin(match.handle)
                match = None
                shorter = self._prefix_state.longest(state_key, below=matched)
                if shorter <= 0:
                    break
                search_key = state_key.prefix(shorter + 1)
            matched_handle = None if match is None else match.handle
            page_ids = () if match is None else match.page_ids
            cached_tokens = 0 if match is None else match.cached_tokens
            start_pages = cached_tokens // self.page_size
            chain, target_pages = self._offload_target(
                request_id,
                key_or_tokens,
                start_pages,
            )
            extension: tuple[int, ...] = ()
            if target_pages > start_pages:
                needed = target_pages - start_pages
                self._evict_for_allocation(needed)
            try:
                with self._directory.transaction(request_id) as transaction:
                    if page_ids:
                        transaction.attach(self._page_allocations(page_ids))
                    if target_pages > start_pages:
                        allocations = transaction.reserve(
                            CacheKind.KV_PAGE,
                            target_pages - start_pages,
                        )
                        extension = tuple(
                            allocation.index for allocation in allocations
                        )
                    self._reserve_fixed_state(transaction)
                    handle = transaction.commit()
            except Exception:
                if prefix is not None and matched_handle is not None:
                    prefix.unpin(matched_handle)
                raise
            if self._prefix_state is not None and cached_tokens > 0:
                try:
                    restored = self._prefix_state.restore(
                        request_id,
                        state_key,
                        cached_tokens,
                        self.request_slot(request_id),
                    )
                except Exception:
                    # Never leave a half-admitted request holding a slot,
                    # a pin, or pages when the state cannot be written back.
                    if matched_handle is not None:
                        prefix.unpin(matched_handle)
                    self._directory.release(request_id)
                    raise
                if not restored:
                    if matched_handle is not None:
                        prefix.unpin(matched_handle)
                    self._directory.release(request_id)
                    raise KVCacheError(
                        "prefix checkpoint vanished during admission"
                    )
            if matched_handle is not None:
                self._prefix_handles[request_id] = matched_handle
            self._request_pages[request_id] = page_ids + extension
            self._published_prefix_tokens[request_id] = cached_tokens
            self._shared_prefix_tokens[request_id] = cached_tokens
            if self._uses_gdn_state:
                self._gdn_active_copy[request_id] = 0
            if extension and chain is not None:
                try:
                    started = self._offload.begin_promotion(
                        request_id=request_id,
                        chain=chain,
                        start=start_pages,
                        end=target_pages,
                        device_pages=extension,
                        device_key=self._offload_key,
                        device_value=self._offload_value,
                    )
                except Exception:  # noqa: BLE001 - offload is best effort
                    started = False
                if started:
                    self._promotions[request_id] = (
                        cached_tokens,
                        target_pages * self.page_size,
                        extension,
                        key_or_tokens,
                    )
                else:
                    self.trim_tokens(request_id, cached_tokens)
            return CacheAdmission(
                handle,
                cached_tokens,
                page_ids,
                pending_promotion=request_id in self._promotions,
            )

    def ensure_tokens(self, request_id: str, total_tokens: int) -> CacheHandle:
        if total_tokens <= 0:
            raise KVCacheError("a cache reservation requires at least one token")
        if total_tokens > self.max_model_len:
            raise KVCacheError(
                f"request needs {total_tokens} tokens, max is {self.max_model_len}"
            )
        with self._lock:
            required_pages = math.ceil(total_tokens / self.page_size)
            current = self._directory.get(request_id)
            if current is None:
                current = self.admit(request_id)
            page_ids = self._request_pages[request_id]
            current_pages = len(page_ids)
            if current_pages >= required_pages:
                return current
            needed_pages = required_pages - current_pages
            self._evict_for_allocation(needed_pages)
            with self._directory.transaction(request_id) as transaction:
                allocations = transaction.reserve(
                    CacheKind.KV_PAGE,
                    needed_pages,
                )
                handle = transaction.commit()
            self._request_pages[request_id] = page_ids + tuple(
                allocation.index
                for allocation in allocations
            )
            return handle

    def has_reserved_tokens(self, request_id: str, total_tokens: int) -> bool:
        """Return whether the Leader already owns every page for the range.

        The scheduler is the sole mutator of logical cache ownership.  This
        lock-free read keeps the common decode step out of the transactional
        allocator; ``ensure_tokens`` remains the checked slow path at a page
        boundary.
        """

        if total_tokens <= 0 or total_tokens > self.max_model_len:
            return False
        pages = self._request_pages.get(request_id)
        return (
            pages is not None
            and len(pages) * self.page_size >= total_tokens
        )

    def trim_tokens(self, request_id: str, total_tokens: int) -> CacheHandle:
        """Release whole speculative tail pages beyond the committed length."""

        if total_tokens < 0 or total_tokens > self.max_model_len:
            raise KVCacheError("cache trim token count is invalid")
        with self._lock:
            handle = self._directory.get(request_id)
            if handle is None:
                raise KVCacheError(
                    f"request {request_id!r} has no cache handle"
                )
            protected_tokens = max(
                self._published_prefix_tokens.get(request_id, 0),
                self._shared_prefix_tokens.get(request_id, 0),
            )
            if total_tokens < protected_tokens:
                raise KVCacheError(
                    "cache trim would release a protected prefix"
                )
            required_pages = math.ceil(total_tokens / self.page_size)
            pages = self._request_pages[request_id]
            if required_pages >= len(pages):
                return handle
            released_pages = pages[required_pages:]
            with self._directory.transaction(request_id) as transaction:
                transaction.detach(
                    self._page_allocations(released_pages)
                )
                updated = transaction.commit()
            self._request_pages[request_id] = pages[:required_pages]
            return updated

    def release(self, request_id: str) -> CacheHandle:
        with self._lock:
            if request_id in self._promotions and self._offload is not None:
                self._offload.cancel(request_id)
                self._promotions.pop(request_id, None)
            self._offload_chains.pop(request_id, None)
            prefix_handle = self._prefix_handles.pop(request_id, None)
            if prefix_handle is not None:
                assert self._prefix_cache is not None
                self._prefix_cache.unpin(prefix_handle)
            self._request_pages.pop(request_id, None)
            self._published_prefix_tokens.pop(request_id, None)
            self._shared_prefix_tokens.pop(request_id, None)
            self._gdn_active_copy.pop(request_id, None)
            return self._directory.release(request_id)

    def publish_prefix(
        self,
        request_id: str,
        key_or_tokens: RadixKeyInput,
        *,
        cached_tokens: int,
    ) -> RadixInsertResult | None:
        """Publish newly computed full pages and move the request's radix pin."""

        with self._lock:
            prefix = self._prefix_cache
            if prefix is None:
                return None
            handle = self._directory.get(request_id)
            if handle is None:
                raise KVCacheError(
                    f"request {request_id!r} has no cache handle"
                )
            input_tokens = (
                key_or_tokens.token_count
                if isinstance(key_or_tokens, RadixKey)
                else len(key_or_tokens)
            )
            valid_tokens = min(cached_tokens, input_tokens)
            aligned_tokens = (
                valid_tokens // self.page_size
            ) * self.page_size
            previous_tokens = self._published_prefix_tokens.get(request_id, 0)
            if aligned_tokens <= previous_tokens:
                return None
            pages = self._request_pages[request_id]
            prefix_key = (
                key_or_tokens.prefix(aligned_tokens)
                if isinstance(key_or_tokens, RadixKey)
                else RadixKey.from_tokens(
                    key_or_tokens[:aligned_tokens],
                    page_size=self.page_size,
                )
            )
            if self._prefix_state is not None:
                # The recurrent state only exists at exact page ends.  A
                # partial page would checkpoint history that does not match
                # the published boundary, so defer to the next full page.
                if valid_tokens != aligned_tokens:
                    return None
                if not self._prefix_state.snapshot(
                    request_id,
                    prefix_key,
                    aligned_tokens,
                ):
                    return None
            result = prefix.insert_prefix(
                prefix_key,
                pages[: aligned_tokens // self.page_size],
                pin=True,
                replace_handle=self._prefix_handles.get(request_id),
            )
            if result.inserted_pages > 0 and self._offload is not None:
                chain = self._sync_offload_chain(request_id, prefix_key)
                page_count = aligned_tokens // self.page_size
                try:
                    ticket = self._offload.store_prefix(
                        chain=chain,
                        start=page_count - result.inserted_pages,
                        end=page_count,
                        device_key=self._offload_key,
                        device_value=self._offload_value,
                        device_pages=pages[:page_count],
                    )
                except Exception:  # noqa: BLE001 - offload is best effort
                    ticket = None
                if ticket is not None:
                    allocations = self._page_allocations(ticket.pages)
                    self._directory.retain(allocations)
                    self._store_refs[ticket.ticket_id] = allocations
            if result.handle is not None:
                self._prefix_handles[request_id] = result.handle
            self._published_prefix_tokens[request_id] = aligned_tokens
            return result

    def shared_prefix_tokens(self, request_id: str) -> int:
        return self._shared_prefix_tokens.get(request_id, 0)

    def published_prefix_tokens(self, request_id: str) -> int:
        return self._published_prefix_tokens.get(request_id, 0)

    def page_ids(self, request_id: str) -> tuple[int, ...]:
        try:
            return self._request_pages[request_id]
        except KeyError as exc:
            raise KVCacheError(
                f"request {request_id!r} has no cache handle"
            ) from exc

    def request_slot(self, request_id: str) -> int:
        handle = self._directory.get(request_id)
        if handle is None:
            raise KVCacheError(f"request {request_id!r} has no cache handle")
        return handle.refs(CacheKind.REQUEST_SLOT)[0].index

    def gdn_state_indices(
        self,
        request_id: str,
        *,
        shadow: bool = False,
    ) -> tuple[int, int]:
        handle = self._directory.get(request_id)
        if handle is None:
            raise KVCacheError(f"request {request_id!r} has no cache handle")
        conv = handle.refs(CacheKind.GDN_CONV_STATE)
        recurrent = handle.refs(CacheKind.GDN_RECURRENT_STATE)
        if not conv and not recurrent:
            return (-1, -1)
        if (
            len(conv) != self._gdn_states_per_request
            or len(recurrent) != self._gdn_states_per_request
        ):
            raise KVCacheError("request GDN state ownership is incomplete")
        active = self._gdn_active_copy.get(request_id, 0)
        if shadow:
            if self._gdn_states_per_request != 2:
                raise KVCacheError(
                    "request has no transactional GDN shadow state"
                )
            selected = 1 - active
        else:
            selected = active
        return conv[selected].index, recurrent[selected].index

    def swap_gdn_state(self, request_id: str) -> None:
        if self._gdn_states_per_request != 2:
            raise KVCacheError(
                "request has no transactional GDN shadow state"
            )
        if self._directory.get(request_id) is None:
            raise KVCacheError(f"request {request_id!r} has no cache handle")
        self._gdn_active_copy[request_id] = (
            1 - self._gdn_active_copy[request_id]
        )

    def slot_mapping(
        self,
        request_id: str,
        start_position: int,
        token_count: int,
    ) -> tuple[int, ...]:
        if start_position < 0 or token_count <= 0:
            raise KVCacheError("invalid slot-mapping range")
        protected_tokens = self._published_prefix_tokens.get(request_id, 0)
        if start_position < protected_tokens:
            raise KVCacheError(
                "copy-on-write violation: execution would overwrite shared KV"
            )
        end_position = start_position + token_count
        self.ensure_tokens(request_id, end_position)
        return self.slot_mapping_reserved(
            request_id,
            start_position,
            token_count,
        )

    def slot_mapping_reserved(
        self,
        request_id: str,
        start_position: int,
        token_count: int,
    ) -> tuple[int, ...]:
        """Map a range whose pages were reserved by the current schedule."""

        if start_position < 0 or token_count <= 0:
            raise KVCacheError("invalid slot-mapping range")
        protected_tokens = self._published_prefix_tokens.get(request_id, 0)
        if start_position < protected_tokens:
            raise KVCacheError(
                "copy-on-write violation: execution would overwrite shared KV"
            )
        end_position = start_position + token_count
        if not self.has_reserved_tokens(request_id, end_position):
            raise KVCacheError(
                "slot mapping range has not been reserved transactionally"
            )
        pages = self.page_ids(request_id)
        return tuple(
            pages[position // self.page_size] * self.page_size
            + position % self.page_size
            for position in range(start_position, end_position)
        )

    def block_table(self, request_id: str) -> tuple[int, ...]:
        pages = self.page_ids(request_id)
        if len(pages) > self.max_blocks_per_sequence:
            raise KVCacheError("request block table exceeds the configured maximum")
        return pages

    def build_block_tables(
        self,
        request_ids: tuple[str, ...],
    ) -> tensorplay.Tensor:
        host_table: list[list[int]] = []
        for request_id in request_ids:
            pages = self.block_table(request_id)
            host_table.append(
                list(pages)
                + [-1] * (self.max_blocks_per_sequence - len(pages))
            )
        return tensorplay.tensor(
            host_table,
            dtype=tensorplay.int32,
            device=self.device,
        )

    def assert_integrity(self) -> None:
        with self._lock:
            self._directory.assert_integrity()
            prefix_pages: tuple[int, ...] = ()
            if self._prefix_cache is not None:
                self._prefix_cache.assert_integrity()
                prefix_pages = self._prefix_cache.cached_page_ids()
            expected_refs = [0] * self.total_pages
            for request_id in self._directory.active_request_ids:
                pages = self.page_ids(request_id)
                if len(set(pages)) != len(pages):
                    raise KVCacheError(f"duplicate page within {request_id!r}")
                for page in pages:
                    expected_refs[page] += 1
            for page in prefix_pages:
                expected_refs[page] += 1
            for allocations in self._store_refs.values():
                for allocation in allocations:
                    if allocation.kind is CacheKind.KV_PAGE:
                        expected_refs[allocation.index] += 1
            for page, expected in enumerate(expected_refs):
                actual = self._directory.refcount(
                    self._directory.allocation(CacheKind.KV_PAGE, page)
                )
                if actual != expected:
                    raise KVCacheError(
                        f"KV page {page} has refcount {actual}, expected {expected}"
                    )
            allocated = sum(refcount > 0 for refcount in expected_refs)
            if allocated != self.total_pages - self.free_pages:
                raise KVCacheError(
                    "logical page accounting does not match capacity"
                )
            active_ids = set(self._directory.active_request_ids)
            if not set(self._prefix_handles).issubset(active_ids):
                raise KVCacheError("radix handle outlived its request")
            if set(self._published_prefix_tokens) != active_ids:
                raise KVCacheError("published-prefix directory is inconsistent")
            if set(self._shared_prefix_tokens) != active_ids:
                raise KVCacheError("shared-prefix directory is inconsistent")
            if set(self._request_pages) != active_ids:
                raise KVCacheError("request page directory is inconsistent")
            expected_gdn_ids = active_ids if self._uses_gdn_state else set()
            if set(self._gdn_active_copy) != expected_gdn_ids:
                raise KVCacheError("GDN active-state directory is inconsistent")
            if not set(self._promotions).issubset(active_ids):
                raise KVCacheError("promotion outlived its request")
            if not set(self._offload_chains).issubset(active_ids):
                raise KVCacheError("offload key chain outlived its request")
            for ticket, allocations in self._store_refs.items():
                if ticket <= 0 or not allocations:
                    raise KVCacheError("offload store ticket is invalid")
                for allocation in allocations:
                    if self._directory.refcount(allocation) <= 0:
                        raise KVCacheError(
                            "offload store lost a retained KV page"
                        )
            for request_id in active_ids:
                handle = self._directory.get(request_id)
                assert handle is not None
                handle_pages = tuple(
                    allocation.index
                    for allocation in handle.refs(CacheKind.KV_PAGE)
                )
                if handle_pages != self._request_pages[request_id]:
                    raise KVCacheError(
                        f"request page index diverged for {request_id!r}"
                    )

    def schedulable_tokens(
        self,
        request_id: str,
        *,
        cached_tokens: int,
    ) -> int:
        owned_pages = len(self._request_pages.get(request_id, ()))
        owned_slack = owned_pages * self.page_size - cached_tokens
        return (
            max(0, owned_slack)
            + self.available_pages * self.page_size
        )

    def poll_offload(
        self,
        *,
        block: bool = False,
        timeout_s: float = 0.005,
    ) -> tuple[PromotionOutcome, ...]:
        """Advance lower-tier work; returns terminal promotion outcomes."""

        if self._offload is None:
            return ()
        with self._lock:
            outcomes, tickets = self._offload.poll(
                device_key=self._offload_key,
                device_value=self._offload_value,
                block=block,
                timeout_s=timeout_s,
            )
        for ticket in tickets:
            allocations = self._store_refs.pop(ticket.ticket_id, None)
            if allocations:
                self._directory.release_refs(allocations)
        for outcome in outcomes:
            if outcome.ready:
                self._apply_promotion(outcome)
            else:
                self._fail_promotion(outcome)
        return outcomes

    def _apply_promotion(self, outcome: PromotionOutcome) -> None:
        pending = self._promotions.pop(outcome.request_id, None)
        if pending is None:
            return
        radix_tokens, _target_tokens, _extension, key = pending
        tokens = outcome.cached_tokens
        if tokens <= radix_tokens:
            self._trim_to_radix(outcome.request_id, radix_tokens)
            return
        prefix = self._prefix_cache
        if prefix is None:
            self._trim_to_radix(outcome.request_id, radix_tokens)
            return
        try:
            state_key = key.prefix(tokens)
            if self._prefix_state is not None:
                restored = self._prefix_state.restore(
                    outcome.request_id,
                    state_key,
                    tokens,
                    self.request_slot(outcome.request_id),
                )
                if not restored:
                    self._trim_to_radix(outcome.request_id, radix_tokens)
                    return
            pages = self._request_pages[outcome.request_id][
                : tokens // self.page_size
            ]
            result = prefix.insert_prefix(
                state_key,
                pages,
                pin=True,
                replace_handle=self._prefix_handles.get(
                    outcome.request_id
                ),
            )
        except Exception:  # noqa: BLE001 - publishing is best effort
            self._trim_to_radix(outcome.request_id, radix_tokens)
            return
        if result.handle is not None:
            self._prefix_handles[outcome.request_id] = result.handle
        self._published_prefix_tokens[outcome.request_id] = tokens
        self._shared_prefix_tokens[outcome.request_id] = tokens

    def _fail_promotion(self, outcome: PromotionOutcome) -> None:
        pending = self._promotions.pop(outcome.request_id, None)
        if pending is None:
            return
        self._trim_to_radix(outcome.request_id, pending[0])

    def _trim_to_radix(self, request_id: str, radix_tokens: int) -> None:
        if self._directory.get(request_id) is None:
            return
        self.trim_tokens(request_id, radix_tokens)

    def evict_prefix_pages(self, target_pages: int) -> tuple[int, ...]:
        with self._lock:
            if self._prefix_cache is None:
                return ()
            return self._prefix_cache.evict_pages(target_pages)

    def stats(self) -> CacheStats:
        with self._lock:
            request_page_refs = sum(
                len(pages)
                for pages in self._request_pages.values()
            )
            gdn_total = self._directory.capacities.get(
                CacheKind.GDN_CONV_STATE, 0
            )
            gdn_free = self._directory.available(CacheKind.GDN_CONV_STATE)
            return CacheStats(
                total_pages=self.total_pages,
                free_pages=self.free_pages,
                reclaimable_pages=self.reclaimable_pages,
                allocated_pages=self.total_pages - self.free_pages,
                active_requests=len(self._directory.active_request_ids),
                request_page_refs=request_page_refs,
                gdn_slots_total=gdn_total,
                gdn_slots_used=gdn_total - gdn_free,
                capacity_failures=dict(self._directory.capacity_failures),
                prefix=(
                    None
                    if self._prefix_cache is None
                    else self._prefix_cache.stats()
                ),
                offload=(
                    None
                    if self._offload is None
                    else self._offload.stats().snapshot()
                ),
            )

    def _page_allocations(
        self,
        page_ids: tuple[int, ...],
    ) -> tuple[CacheAllocation, ...]:
        return tuple(
            self._page_allocation_table[page]
            for page in page_ids
        )

    def _reserve_fixed_state(self, transaction: object) -> None:
        if not self._uses_gdn_state:
            return
        # CacheTransaction is intentionally kept out of the public runtime
        # surface; this helper only relies on its stable reserve ABI.
        transaction.reserve(  # type: ignore[attr-defined]
            CacheKind.GDN_CONV_STATE,
            self._gdn_states_per_request,
        )
        transaction.reserve(  # type: ignore[attr-defined]
            CacheKind.GDN_RECURRENT_STATE,
            self._gdn_states_per_request,
        )

    def _retain_page_ids(self, page_ids: tuple[int, ...]) -> None:
        self._directory.retain(self._page_allocations(page_ids))

    def _release_page_ids(self, page_ids: tuple[int, ...]) -> None:
        self._directory.release_refs(self._page_allocations(page_ids))

    def _evict_for_allocation(self, needed_pages: int) -> None:
        missing = needed_pages - self.free_pages
        if missing <= 0 or self._prefix_cache is None:
            return
        self._prefix_cache.evict_pages(missing)


__all__ = [
    "CacheAdmission",
    "CacheCapacityError",
    "CacheGeometry",
    "CacheStats",
    "KVCacheError",
    "KVCacheManager",
    "PagedKVCache",
    "PrefixStateStore",
]
