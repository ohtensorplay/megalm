"""G3 gold tests: KV transfer handle and staged page promotion."""

from __future__ import annotations

import tensorplay
import pytest

from megalm.runtime.cache import PagedKVCache
from megalm.runtime.kv_connector import (
    KVCacheGeometry,
    KVConnectorError,
    KVTransferHandle,
    KVTransferParams,
    LocalKVConnector,
)


def _cache(*, num_pages: int = 4, page_size: int = 4, num_layers: int = 2):
    return PagedKVCache(
        num_layers=num_layers,
        num_pages=num_pages,
        page_size=page_size,
        num_kv_heads=2,
        head_dim=8,
        dtype=tensorplay.float16,
        device="cpu",
    )


def _fill(cache: PagedKVCache, page: int, value: float) -> None:
    shape = cache.key[:, page].shape
    cache.key[:, page].copy_(tensorplay.full(shape, value, dtype=cache.key.dtype))
    cache.value[:, page].copy_(tensorplay.full(shape, value + 1, dtype=cache.value.dtype))


def test_publish_promotes_only_full_pages() -> None:
    src = _cache()
    _fill(src, 0, 1.0)
    _fill(src, 1, 2.0)
    connector = LocalKVConnector()

    handle = connector.publish(
        src,
        request_id="req-1",
        src_instance="prefill",
        dst_instance="decode",
        published_tokens=9,  # 2 full pages + 1 tail token
        page_ids=(0, 1, 2),
    )
    assert handle.page_ids == (0, 1)
    assert handle.published_tokens == 8
    assert handle.tail_start == 9
    assert handle.page_size == 4
    assert len(handle.digest) == 64
    assert connector.verify(handle, src) is True


def test_apply_copies_exact_pages_and_preserves_untouched_pages() -> None:
    src = _cache()
    dst = _cache()
    _fill(src, 0, 3.0)
    _fill(src, 1, 4.0)
    _fill(dst, 2, 99.0)
    connector = LocalKVConnector()
    handle = connector.publish(
        src,
        request_id="req-2",
        src_instance="prefill",
        dst_instance="decode",
        published_tokens=8,
        page_ids=(0, 1),
    )

    applied = connector.apply(handle, src, dst, (2, 0))
    assert applied == 8
    assert dst.key[0, 2].tolist() == src.key[0, 0].tolist()
    assert dst.value[0, 2].tolist() == src.value[0, 0].tolist()
    assert dst.key[0, 0].tolist() == src.key[0, 1].tolist()
    # The tail page was never imported.
    assert dst.key[0, 3].tolist() == src.key[0, 3].tolist()


def test_apply_rejects_geometry_mismatch_and_short_page_table() -> None:
    src = _cache(num_layers=2)
    other = _cache(num_layers=3)
    connector = LocalKVConnector()
    handle = connector.publish(
        src,
        request_id="req-3",
        src_instance="prefill",
        dst_instance="decode",
        published_tokens=4,
        page_ids=(0,),
    )
    with pytest.raises(KVConnectorError, match="geometry mismatch"):
        connector.apply(handle, src, other, (0,))
    with pytest.raises(KVConnectorError, match="page table is too short"):
        connector.apply(handle, src, src, ())


def test_digest_detects_source_drift_after_publish() -> None:
    src = _cache()
    _fill(src, 0, 5.0)
    connector = LocalKVConnector()
    handle = connector.publish(
        src,
        request_id="req-4",
        src_instance="prefill",
        dst_instance="decode",
        published_tokens=4,
        page_ids=(0,),
    )
    assert connector.verify(handle, src)
    _fill(src, 0, 6.0)
    assert not connector.verify(handle, src)


def test_handle_validation_and_compatibility() -> None:
    geometry = KVCacheGeometry.from_cache(_cache())
    handle = KVTransferHandle(
        request_id="req-5",
        src_instance="prefill",
        dst_instance="decode",
        geometry=geometry,
        page_ids=(0,),
        published_tokens=4,
        tail_start=6,
        digest="a" * 64,
    )
    assert handle.compatible_with(geometry)
    with pytest.raises(KVConnectorError, match="identity is incomplete"):
        KVTransferHandle(
            request_id="",
            src_instance="prefill",
            dst_instance="decode",
            geometry=geometry,
            page_ids=(0,),
            published_tokens=4,
            tail_start=6,
            digest="a" * 64,
        )
    with pytest.raises(KVConnectorError, match="sizes are invalid"):
        KVTransferHandle(
            request_id="req-6",
            src_instance="prefill",
            dst_instance="decode",
            geometry=geometry,
            page_ids=(0,),
            published_tokens=-1,
            tail_start=6,
            digest="a" * 64,
        )


def test_transfer_params_round_trip_and_decode_forwarding() -> None:
    params = KVTransferParams.from_wire(
        {
            "do_remote_prefill": True,
            "remote_engine_id": "engine-1",
            "remote_request_id": "req-1",
        }
    )
    assert params.do_remote_prefill is True
    assert params.source_instance() == "engine-1"
    assert params.decode_params() == {
        "do_remote_decode": True,
        "remote_engine_id": "engine-1",
        "remote_request_id": "req-1",
    }


def test_transfer_params_keeps_unknown_handle_fields() -> None:
    params = KVTransferParams.from_wire(
        {"remote_engine_id": "e", "connector": "rdma", "opaque": "xyz"}
    )
    forwarded = params.decode_params()
    assert forwarded["connector"] == "rdma"
    assert forwarded["opaque"] == "xyz"


def test_transfer_params_rejects_malformed_wire_objects() -> None:
    with pytest.raises(KVConnectorError, match="non-empty object"):
        KVTransferParams.from_wire({})
    with pytest.raises(KVConnectorError, match="non-empty object"):
        KVTransferParams.from_wire("not-an-object")
    with pytest.raises(KVConnectorError, match="no recognized field"):
        KVTransferParams.from_wire({"arbitrary": 1})


def test_publish_rejects_tokens_beyond_page_table() -> None:
    src = _cache()
    connector = LocalKVConnector()
    with pytest.raises(KVConnectorError, match="exceed the page table"):
        connector.publish(
            src,
            request_id="req-7",
            src_instance="prefill",
            dst_instance="decode",
            published_tokens=16,
            page_ids=(0, 1),
        )
