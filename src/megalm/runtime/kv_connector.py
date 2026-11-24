"""KV transfer connector plugin for prefill/decode disaggregation (G3).

This module owns only the *transfer channel and staged-promotion semantics*;
it does not change the Engine or Scheduler. A prefill instance publishes full
KV pages behind a small handle; a decode instance applies them into its own
paged arena. Unpublished tail pages are deliberately excluded — the decode
side recomputes them, so a partial page can never decide request correctness.

The default :class:`LocalKVConnector` copies pages between two caches in one
process (CPU or CUDA). Multi-instance transport (CUDA IPC / RDMA) plugs in via
the :class:`KVConnector` protocol without touching this module's callers.

Full acceptance (bit-identical outputs across two live engines) requires two
engine instances and a transport channel and is not locally verifiable here.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from typing import Protocol

from megalm.runtime.cache import PagedKVCache


class KVConnectorError(RuntimeError):
    """A KV transfer handle is incompatible or corrupted."""


@dataclass(frozen=True, slots=True)
class KVTransferParams:
    """Typed prefill/decode handshake carried in ``kv_transfer_params``.

    This is the engine-side contract for the handle that ``PDRouter`` relays.
    The wire representation stays an opaque JSON object (see ``pd_router``), so
    adding this typed view does not change the HTTP protocol. The decode side
    consumes ``remote_engine_id`` / ``remote_request_id`` to locate the source
    instance when applying a :class:`KVTransferHandle`.
    """

    do_remote_prefill: bool = False
    do_remote_decode: bool = False
    remote_engine_id: str | None = None
    remote_request_id: str | None = None
    connector: str = "local"
    handle: tuple[tuple[str, str], ...] = ()

    @classmethod
    def from_wire(cls, params: object) -> KVTransferParams:
        if not isinstance(params, dict) or not params:
            raise KVConnectorError("kv_transfer_params must be a non-empty object")
        known = {
            "do_remote_prefill",
            "do_remote_decode",
            "remote_engine_id",
            "remote_request_id",
            "connector",
        }
        if not any(key in params for key in known):
            raise KVConnectorError("kv_transfer_params has no recognized field")
        return cls(
            do_remote_prefill=bool(params.get("do_remote_prefill", False)),
            do_remote_decode=bool(params.get("do_remote_decode", False)),
            remote_engine_id=(
                None
                if params.get("remote_engine_id") is None
                else str(params["remote_engine_id"])
            ),
            remote_request_id=(
                None
                if params.get("remote_request_id") is None
                else str(params["remote_request_id"])
            ),
            connector=str(params.get("connector", "local")),
            handle=tuple(
                (str(key), str(value))
                for key, value in params.items()
                if key not in known
            ),
        )

    def decode_params(self) -> dict[str, object]:
        """Wire dict forwarded to the decode upstream."""

        body: dict[str, object] = {"do_remote_decode": True}
        if self.remote_engine_id is not None:
            body["remote_engine_id"] = self.remote_engine_id
        if self.remote_request_id is not None:
            body["remote_request_id"] = self.remote_request_id
        if self.connector != "local":
            body["connector"] = self.connector
        body.update(self.handle)
        return body

    def source_instance(self) -> str | None:
        return self.remote_engine_id


@dataclass(frozen=True, slots=True)
class KVCacheGeometry:
    num_layers: int
    page_size: int
    num_kv_heads: int
    head_dim: int
    dtype: str
    layer_indices: tuple[int, ...]

    @classmethod
    def from_cache(cls, cache: PagedKVCache) -> KVCacheGeometry:
        return cls(
            num_layers=cache.geometry.num_layers,
            page_size=cache.geometry.page_size,
            num_kv_heads=cache.geometry.num_kv_heads,
            head_dim=cache.geometry.head_dim,
            dtype=str(cache.key.dtype),
            layer_indices=tuple(cache.layer_indices),
        )


@dataclass(frozen=True, slots=True)
class KVTransferHandle:
    request_id: str
    src_instance: str
    dst_instance: str
    geometry: KVCacheGeometry
    page_ids: tuple[int, ...]
    published_tokens: int
    tail_start: int
    digest: str
    schema_version: int = 1

    def __post_init__(self) -> None:
        if not self.request_id or not self.src_instance:
            raise KVConnectorError("transfer handle identity is incomplete")
        if self.page_size <= 0 or self.published_tokens < 0:
            raise KVConnectorError("transfer handle sizes are invalid")

    @property
    def page_size(self) -> int:
        return self.geometry.page_size

    def compatible_with(self, dst: KVCacheGeometry) -> bool:
        return self.geometry == dst


class KVConnector(Protocol):
    def publish(
        self,
        cache: PagedKVCache,
        *,
        request_id: str,
        src_instance: str,
        dst_instance: str,
        published_tokens: int,
        page_ids: tuple[int, ...],
    ) -> KVTransferHandle: ...

    def verify(self, handle: KVTransferHandle, src_cache: PagedKVCache) -> bool: ...

    def apply(
        self,
        handle: KVTransferHandle,
        src_cache: PagedKVCache,
        dst_cache: PagedKVCache,
        dst_page_ids: tuple[int, ...],
    ) -> int: ...

    def release(self, handle: KVTransferHandle) -> None: ...


class LocalKVConnector:
    """In-process connector: copies published pages into a decoder arena."""

    def publish(
        self,
        cache: PagedKVCache,
        *,
        request_id: str,
        src_instance: str,
        dst_instance: str,
        published_tokens: int,
        page_ids: tuple[int, ...],
    ) -> KVTransferHandle:
        geometry = KVCacheGeometry.from_cache(cache)
        full_pages = published_tokens // geometry.page_size
        if full_pages < 0 or full_pages > len(page_ids):
            raise KVConnectorError("published tokens exceed the page table")
        selected = tuple(page_ids[:full_pages])
        digest = _pages_digest(cache, selected)
        return KVTransferHandle(
            request_id=request_id,
            src_instance=src_instance,
            dst_instance=dst_instance,
            geometry=geometry,
            page_ids=selected,
            published_tokens=full_pages * geometry.page_size,
            tail_start=published_tokens,
            digest=digest,
        )

    def verify(
        self,
        handle: KVTransferHandle,
        src_cache: PagedKVCache,
    ) -> bool:
        """Check the source pages still match the published handle digest."""

        return _pages_digest(src_cache, handle.page_ids) == handle.digest

    def apply(
        self,
        handle: KVTransferHandle,
        src_cache: PagedKVCache,
        dst_cache: PagedKVCache,
        dst_page_ids: tuple[int, ...],
    ) -> int:
        if not handle.compatible_with(KVCacheGeometry.from_cache(dst_cache)):
            raise KVConnectorError("KV cache geometry mismatch")
        if not handle.compatible_with(KVCacheGeometry.from_cache(src_cache)):
            raise KVConnectorError("KV cache geometry mismatch")
        if len(dst_page_ids) < len(handle.page_ids):
            raise KVConnectorError("destination page table is too short")
        if not self.verify(handle, src_cache):
            raise KVConnectorError("source pages drifted since publish")
        src_pages = handle.page_ids
        dst_pages = dst_page_ids[: len(src_pages)]
        for src_page, dst_page in zip(src_pages, dst_pages):
            dst_cache.key[:, dst_page].copy_(src_cache.key[:, src_page])
            dst_cache.value[:, dst_page].copy_(src_cache.value[:, src_page])
        return handle.published_tokens

    def release(self, handle: KVTransferHandle) -> None:
        # In-process pages are owned by their cache; nothing to free.
        return None


def _pages_digest(cache: PagedKVCache, page_ids: tuple[int, ...]) -> str:
    digest = hashlib.sha256()
    for page in page_ids:
        for tensor in (cache.key, cache.value):
            payload = tensor[:, page]
            digest.update(str(payload.dtype).encode())
            digest.update(str(tuple(payload.shape)).encode())
            # ``tolist`` works on both CPU and CUDA tensors without a dedicated
            # byte-export API in the tensor runtime. Transport-level checksums
            # remain the responsibility of remote connectors.
            digest.update(repr(payload.tolist()).encode())
    return digest.hexdigest()


__all__ = [
    "KVConnector",
    "KVConnectorError",
    "KVCacheGeometry",
    "KVTransferHandle",
    "KVTransferParams",
    "LocalKVConnector",
]
