import time
from pathlib import Path

import pytest
import tensorplay

from megalm.contracts.request import InferenceRequest, SamplingParams
from megalm.runtime.cache import KVCacheManager, PagedKVCache
from megalm.runtime.engine import EngineConfig
from megalm.runtime.kv_offload import (
    CopyTask,
    DiskKVTier,
    HostKVTier,
    KVOffloadError,
    TensorCopier,
    TieredKVOffload,
)
from megalm.runtime.radix_cache import RadixKey
from megalm.runtime.scheduler import ContinuousBatchScheduler

PAGE = 4
TOKENS = tuple(range(8))
IDENTITY = "megalm-kv-test"


def _identity_digest() -> bytes:
    import hashlib

    return hashlib.blake2b(IDENTITY.encode(), digest_size=32).digest()


def _arena(pages: int = 8) -> PagedKVCache:
    return PagedKVCache(
        num_layers=2,
        num_pages=pages,
        page_size=PAGE,
        num_kv_heads=1,
        head_dim=2,
        dtype=tensorplay.float16,
        device="cpu",
    )


def _host(pages: int = 8, copier: TensorCopier | None = None) -> HostKVTier:
    return HostKVTier(
        identity_digest=_identity_digest(),
        rows=2,
        page_size=PAGE,
        num_kv_heads=1,
        head_dim=2,
        dtype=tensorplay.float16,
        capacity_bytes=pages * 64,
        copier=copier or TensorCopier(asynchronous=False),
        pin_memory=False,
    )


def _offload(host: HostKVTier, disk: DiskKVTier | None = None):
    return TieredKVOffload(identity=IDENTITY, host=host, disk=disk)


def _manager(
    offload: TieredKVOffload,
    arena: PagedKVCache,
    *,
    pages: int = 8,
) -> KVCacheManager:
    return KVCacheManager(
        num_pages=pages,
        page_size=PAGE,
        max_sequences=4,
        max_model_len=32,
        device="cpu",
        enable_prefix_cache=True,
        offload=offload,
        offload_key=arena.key,
        offload_value=arena.value,
    )


def _fill(arena: PagedKVCache, page: int, value: float) -> None:
    for row in range(2):
        arena.key[row, page].fill_(value)
        arena.value[row, page].fill_(value + 10)


def _store_one_page(
    arena: PagedKVCache,
    offload: TieredKVOffload,
    *,
    page_tokens: tuple[int, ...] | None = None,
    page: int = 0,
) -> tuple[RadixKey, object]:
    tokens = page_tokens or TOKENS[:PAGE]
    key = RadixKey.from_tokens(tokens, page_size=PAGE)
    chain = offload.chain_from_key(key)
    _fill(arena, page, 1.0)
    ticket = offload.store_prefix(
        chain=chain,
        start=0,
        end=1,
        device_key=arena.key,
        device_value=arena.value,
        device_pages=(page,),
    )
    offload.poll(device_key=arena.key, device_value=arena.value)
    return key, ticket


def _wait_for(predicate, timeout: float = 5.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(0.01)
    raise AssertionError("offload job did not finish in time")


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


def test_offload_config_requires_a_host_staging_tier(tmp_path: Path) -> None:
    base = {"model_path": tmp_path / "model", "enable_cuda_graph": False}
    with pytest.raises(Exception, match="host staging tier"):
        EngineConfig(  # type: ignore[arg-type]
            **base,
            kv_disk_cache_dir=str(tmp_path),
            kv_disk_cache_bytes=1024,
        )
    with pytest.raises(Exception, match="byte budget requires a directory"):
        EngineConfig(  # type: ignore[arg-type]
            **base,
            kv_host_cache_bytes=1024,
            kv_disk_cache_bytes=1024,
        )
    config = EngineConfig(  # type: ignore[arg-type]
        **base,
        kv_host_cache_bytes=1024,
        kv_disk_cache_dir=str(tmp_path),
        kv_disk_cache_bytes=1024,
    )
    assert config.kv_host_cache_bytes == 1024


# ---------------------------------------------------------------------------
# Key chain and host tier
# ---------------------------------------------------------------------------


def test_key_chain_is_incremental_and_identity_scoped() -> None:
    offload = _offload(_host())
    chain = offload.chain(
        namespace="ns", page_size=PAGE, pages=[TOKENS[:PAGE]]
    )
    first = chain.digest(0)
    chain.extend([TOKENS[PAGE:]])
    assert len(chain) == 2
    assert chain.digest(0) == first
    assert chain.digest(1) != first
    other = offload.chain(
        namespace="other", page_size=PAGE, pages=[TOKENS[:PAGE]]
    )
    assert other.digest(0) != first
    with pytest.raises(KVOffloadError):
        offload.chain(namespace="ns", page_size=PAGE, pages=[(1, 2)])


def test_host_tier_lru_eviction_pins_and_deduplication() -> None:
    arena = _arena()
    offload = _offload(_host(pages=2))
    chains = [
        offload.chain(
            namespace="",
            page_size=PAGE,
            pages=[tuple(range(i, i + PAGE))],
        )
        for i in range(3)
    ]
    for index, chain in enumerate(chains):
        assert offload.store_prefix(
            chain=chain,
            start=0,
            end=1,
            device_key=arena.key,
            device_value=arena.value,
            device_pages=(index,),
        )
        offload.poll(device_key=arena.key, device_value=arena.value)
    stats = offload.stats().host
    assert stats is not None
    assert stats.used_pages == 2
    assert stats.evictions == 1
    assert offload.lookup(chains[0], 0) == 0
    assert offload.lookup(chains[1], 0) == 1
    assert offload.lookup(chains[2], 0) == 1

    # A pinned entry survives eviction pressure.
    assert offload.host.pin([chains[2].digest(0)])
    third = offload.chain(
        namespace="",
        page_size=PAGE,
        pages=[tuple(range(20, 20 + PAGE))],
    )
    offload.store_prefix(
        chain=third,
        start=0,
        end=1,
        device_key=arena.key,
        device_value=arena.value,
        device_pages=(3,),
    )
    offload.poll(device_key=arena.key, device_value=arena.value)
    assert offload.lookup(chains[2], 0) == 1
    offload.host.unpin([chains[2].digest(0)])

    # Re-storing an identical page does not consume a second slot.
    used = offload.stats().host.used_pages
    assert offload.store_prefix(
        chain=chains[2],
        start=0,
        end=1,
        device_key=arena.key,
        device_value=arena.value,
        device_pages=(4,),
    ) is None
    assert offload.stats().host.used_pages == used


def test_host_tier_verifies_exact_token_pages() -> None:
    arena = _arena()
    offload = _offload(_host())
    key, _ = _store_one_page(arena, offload)
    chain = offload.chain_from_key(key)
    entry = next(iter(offload.host._entries.values()))
    entry.tokens = tuple(-1 for _ in range(PAGE))
    assert offload.lookup(chain, 0) == 0


def test_host_promotion_restores_page_contents() -> None:
    arena = _arena()
    offload = _offload(_host())
    key, _ = _store_one_page(arena, offload)
    chain = offload.chain_from_key(key)
    _fill(arena, 0, 0.0)
    assert offload.begin_promotion(
        request_id="r",
        chain=chain,
        start=0,
        end=1,
        device_pages=(0,),
        device_key=arena.key,
        device_value=arena.value,
    )
    assert offload.pending("r")
    outcomes, _ = offload.poll(
        device_key=arena.key, device_value=arena.value
    )
    assert len(outcomes) == 1 and outcomes[0].ready
    assert outcomes[0].cached_tokens == PAGE
    assert float(arena.key[0, 0, 0, 0, 0]) == 1.0
    assert float(arena.value[0, 0, 0, 0, 0]) == 11.0
    assert not offload.pending("r")


# ---------------------------------------------------------------------------
# Filesystem tier
# ---------------------------------------------------------------------------


def _disk(host: HostKVTier, root: Path, *, pages: int = 8) -> DiskKVTier:
    return DiskKVTier(
        root=root,
        identity=IDENTITY,
        identity_digest=_identity_digest(),
        rows=2,
        page_size=PAGE,
        num_kv_heads=1,
        head_dim=2,
        dtype=tensorplay.float16,
        capacity_bytes=pages * 64,
        host=host,
        read_threads=2,
        write_threads=2,
    )


def test_disk_tier_cascades_survives_restart_and_rejects_corruption(
    tmp_path: Path,
) -> None:
    arena = _arena()
    host = _host()
    disk = _disk(host, tmp_path)
    offload = _offload(host, disk)
    try:
        key, _ = _store_one_page(arena, offload)
        chain = offload.chain_from_key(key)
        _wait_for(lambda: disk.stats().entries == 1)
        stored = disk.directory / chain.digest(0).hex()[:2]
        assert stored.is_dir()

        # A fresh manager with an emptied host tier still finds the disk page.
        host2 = _host()
        disk2 = _disk(host2, tmp_path)
        offload2 = _offload(host2, disk2)
        try:
            assert offload2.lookup(chain, 0) == 1
            _fill(arena, 0, 0.0)
            assert offload2.begin_promotion(
                request_id="disk-hit",
                chain=chain,
                start=0,
                end=1,
                device_pages=(0,),
                device_key=arena.key,
                device_value=arena.value,
            )
            outcomes: tuple = ()
            deadline = time.monotonic() + 5.0
            while time.monotonic() < deadline:
                ready, _ = offload2.poll(
                    device_key=arena.key, device_value=arena.value
                )
                outcomes += ready
                if outcomes:
                    break
                time.sleep(0.01)
            assert outcomes and outcomes[0].ready
            assert float(arena.key[0, 0, 0, 0, 0]) == 1.0
        finally:
            offload2.close()

        # A corrupt payload is dropped rather than trusted.
        path = next(disk2.directory.rglob("*.bin"))
        with path.open("r+b") as handle:
            handle.seek(-1, 2)
            original = handle.read(1)
            handle.seek(-1, 2)
            handle.write(bytes([original[0] ^ 0xFF]))
        host3 = _host()
        disk3 = _disk(host3, tmp_path)
        offload3 = _offload(host3, disk3)
        try:
            assert offload3.lookup(chain, 0) == 1
            assert offload3.begin_promotion(
                request_id="corrupt",
                chain=chain,
                start=0,
                end=1,
                device_pages=(0,),
                device_key=arena.key,
                device_value=arena.value,
            )
            outcomes = ()
            deadline = time.monotonic() + 5.0
            while time.monotonic() < deadline:
                outcomes, _ = offload3.poll(
                    device_key=arena.key, device_value=arena.value
                )
                if outcomes:
                    break
                time.sleep(0.01)
            assert outcomes and not outcomes[0].ready
            assert disk3.stats().load_failures >= 1
        finally:
            offload3.close()
    finally:
        offload.close()


def test_disk_tier_enforces_capacity_and_identity(tmp_path: Path) -> None:
    host = _host()
    arena = _arena()
    disk = _disk(host, tmp_path, pages=1)
    offload = _offload(host, disk)
    try:
        for index in range(3):
            chain = offload.chain(
                namespace="",
                page_size=PAGE,
                pages=[tuple(range(index * PAGE, (index + 1) * PAGE))],
            )
            offload.store_prefix(
                chain=chain,
                start=0,
                end=1,
                device_key=arena.key,
                device_value=arena.value,
                device_pages=(index,),
            )
            offload.poll(device_key=arena.key, device_value=arena.value)
        _wait_for(lambda: disk.stats().store_pages == 3)
        assert disk.stats().entries == 1
        assert disk.stats().evictions >= 2
        assert disk.stats().used_bytes == 64
        with pytest.raises(KVOffloadError):
            DiskKVTier(
                root=tmp_path,
                identity=IDENTITY,
                identity_digest=b"x" * 32,
                rows=2,
                page_size=PAGE * 2,
                num_kv_heads=1,
                head_dim=2,
                dtype=tensorplay.float16,
                capacity_bytes=8 * 64,
                host=host,
            )
    finally:
        offload.close()


# ---------------------------------------------------------------------------
# Cache manager and scheduler integration
# ---------------------------------------------------------------------------


def _publish(manager: KVCacheManager, arena: PagedKVCache, key: RadixKey, request_id: str) -> None:
    manager.admit(request_id)
    slots = manager.slot_mapping(request_id, 0, key.token_count)
    for page in sorted({slot // PAGE for slot in slots}):
        _fill(arena, page, float(page + 1))
    manager.publish_prefix(request_id, key, cached_tokens=key.token_count)
    manager.poll_offload()
    manager.release(request_id)


def test_manager_promotes_evicted_prefix_from_host() -> None:
    arena = _arena()
    offload = _offload(_host())
    manager = _manager(offload, arena)
    key = RadixKey.from_tokens(TOKENS, page_size=PAGE)
    _publish(manager, arena, key, "author")
    assert manager.evict_prefix_pages(2) == (0, 1)
    assert manager.prefix_cache is not None
    assert manager.prefix_cache.cached_pages == 0

    admission = manager.admit_with_prefix("reader", key)
    assert admission.cached_tokens == 0
    assert admission.pending_promotion
    assert manager.promotion_pending("reader")
    outcomes = manager.poll_offload()
    assert len(outcomes) == 1 and outcomes[0].ready
    assert outcomes[0].cached_tokens == len(TOKENS)
    assert manager.published_prefix_tokens("reader") == len(TOKENS)
    assert manager.shared_prefix_tokens("reader") == len(TOKENS)
    assert manager.prefix_cache.cached_pages == 2
    for index, page in enumerate(manager.page_ids("reader")):
        assert float(arena.key[0, page, 0, 0, 0]) == float(index + 1)
    manager.assert_integrity()


def test_manager_extends_a_radix_hit_from_host() -> None:
    arena = _arena()
    offload = _offload(_host())
    manager = _manager(offload, arena)
    tokens = tuple(range(16))
    full = RadixKey.from_tokens(tokens, page_size=PAGE)
    _publish(manager, arena, full, "author")

    # Hold the first two pages in the GPU radix tree.
    holder = manager.admit_with_prefix(
        "holder", RadixKey.from_tokens(tokens[:12], page_size=PAGE)
    )
    assert holder.cached_tokens == 8
    if holder.pending_promotion:
        outcomes = manager.poll_offload()
        assert outcomes and outcomes[0].ready
    assert manager.prefix_cache is not None
    assert manager.prefix_cache.cached_pages == 4

    admission = manager.admit_with_prefix("reader", full)
    assert admission.cached_tokens == 12
    assert admission.pending_promotion
    outcomes = manager.poll_offload()
    assert outcomes and outcomes[0].ready
    assert outcomes[0].cached_tokens == 16
    assert manager.shared_prefix_tokens("reader") == 16
    manager.assert_integrity()


def test_manager_release_cancels_pending_promotion() -> None:
    class ManualTask(CopyTask):
        def __init__(self) -> None:
            super().__init__()
            self.done = False

        def ready(self) -> bool:
            return self.done

        def wait(self) -> None:
            self.done = True

    class DeferredCopier(TensorCopier):
        def __init__(self) -> None:
            super().__init__(asynchronous=False)
            self.tasks: list[ManualTask] = []

        def to_device(self, host, slots, device, pages, *, rows):
            super().to_device(host, slots, device, pages, rows=rows)
            task = ManualTask()
            self.tasks.append(task)
            return task

    copier = DeferredCopier()
    arena = _arena()
    offload = _offload(_host(copier=copier))
    manager = _manager(offload, arena)
    key = RadixKey.from_tokens(TOKENS, page_size=PAGE)
    _publish(manager, arena, key, "author")  # uses to_host only
    manager.evict_prefix_pages(2)
    admission = manager.admit_with_prefix("reader", key)
    assert admission.pending_promotion
    assert copier.tasks
    manager.release("reader")
    assert all(task.done for task in copier.tasks)
    assert not manager.promotion_pending("reader")
    manager.assert_integrity()


def test_scheduler_waits_until_promotion_completes() -> None:
    class ManualTask(CopyTask):
        def __init__(self) -> None:
            super().__init__()
            self.done = False

        def ready(self) -> bool:
            return self.done

        def wait(self) -> None:
            self.done = True

    class DeferredCopier(TensorCopier):
        def __init__(self) -> None:
            super().__init__(asynchronous=False)
            self.tasks: list[ManualTask] = []

        def to_device(self, host, slots, device, pages, *, rows):
            super().to_device(host, slots, device, pages, rows=rows)
            task = ManualTask()
            self.tasks.append(task)
            return task

    copier = DeferredCopier()
    arena = _arena()
    host = _host(pages=1, copier=copier)
    offload = _offload(host)
    manager = _manager(offload, arena)
    key = RadixKey.from_tokens(TOKENS, page_size=PAGE)
    _publish(manager, arena, key, "author")
    manager.evict_prefix_pages(2)
    scheduler = ContinuousBatchScheduler(
        manager,
        eos_token_id=0,
        max_num_sequences=2,
        max_num_batched_tokens=32,
        prefill_chunk_size=16,
        mixed_prefill_chunk_size=16,
    )
    scheduler.submit(
        InferenceRequest(
            request_id="reader",
            model="qwen3",
            input_token_ids=TOKENS,
            max_new_tokens=2,
            sampling=SamplingParams(),
        )
    )
    assert scheduler.schedule() is None
    assert scheduler.is_waiting_on_offload
    assert scheduler.offload_waits == 1
    assert copier.tasks

    for task in copier.tasks:
        task.done = True
    batch = scheduler.schedule()
    assert batch is not None
    assert batch.request_ids == ("reader",)
    assert batch.scheduled_tokens == (PAGE,)
    assert not scheduler.is_waiting_on_offload
