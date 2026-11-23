from megalm.runtime.cache import CacheStats
from megalm.runtime.radix_cache import RadixCacheStats


def test_cache_snapshot_includes_computed_utilization_and_hit_rates() -> None:
    prefix = RadixCacheStats(
        nodes=3,
        cached_pages=4,
        flattened_page_refs=4,
        pinned_pages=1,
        evictable_pages=3,
        lookups=4,
        hits=3,
        misses=1,
        lookup_pages=10,
        hit_pages=8,
        insertions=2,
        inserted_pages=4,
        evictions=0,
        evicted_pages=0,
    )
    stats = CacheStats(
        total_pages=20,
        free_pages=12,
        reclaimable_pages=3,
        allocated_pages=8,
        active_requests=2,
        request_page_refs=5,
        prefix=prefix,
    )

    snapshot = stats.snapshot()
    prefix_snapshot = snapshot["prefix"]

    assert snapshot["utilization"] == 0.4
    assert isinstance(prefix_snapshot, dict)
    assert prefix_snapshot["request_hit_rate"] == 0.75
    assert prefix_snapshot["token_hit_rate"] == 0.8
    assert prefix_snapshot["hits"] == 3


def test_cache_snapshot_preserves_disabled_prefix_cache() -> None:
    snapshot = CacheStats(
        total_pages=4,
        free_pages=4,
        reclaimable_pages=0,
        allocated_pages=0,
        active_requests=0,
        request_page_refs=0,
        prefix=None,
    ).snapshot()

    assert snapshot["utilization"] == 0.0
    assert snapshot["prefix"] is None
