import random

import pytest

from megalm.contracts.cache import CacheCapacityError
from megalm.runtime.cache import KVCacheError, KVCacheManager
from megalm.runtime.radix_cache import (
    RadixCacheError,
    RadixKey,
    RadixKeyBuilder,
    RadixPrefixCache,
)


class _PageRefs:
    def __init__(self) -> None:
        self.refs: dict[int, int] = {}

    def retain(self, pages: tuple[int, ...]) -> None:
        for page in pages:
            self.refs[page] = self.refs.get(page, 0) + 1

    def release(self, pages: tuple[int, ...]) -> None:
        for page in pages:
            count = self.refs[page] - 1
            if count:
                self.refs[page] = count
            else:
                del self.refs[page]


def test_radix_match_splits_at_pages_and_protects_only_matched_path() -> None:
    refs = _PageRefs()
    cache = RadixPrefixCache(
        page_size=2,
        retain_pages=refs.retain,
        release_pages=refs.release,
    )
    inserted = cache.insert_prefix((1, 2, 3, 4, 5, 6), (10, 11, 12))
    assert inserted.inserted_pages == 3
    assert refs.refs == {10: 1, 11: 1, 12: 1}

    match = cache.match_prefix((1, 2, 3, 4, 9, 9, 7))
    assert match.cached_tokens == 4
    assert match.page_ids == (10, 11)
    assert cache.stats().pinned_pages == 2
    assert cache.stats().evictable_pages == 1
    cache.assert_integrity()

    assert cache.evict_pages(1) == (12,)
    assert refs.refs == {10: 1, 11: 1}
    assert cache.evict_pages(1) == ()
    assert match.handle is not None
    cache.unpin(match.handle)
    assert set(cache.evict_pages(2)) == {10, 11}
    assert refs.refs == {}
    cache.assert_integrity()


def test_radix_branch_reuses_canonical_page_instead_of_duplicate() -> None:
    refs = _PageRefs()
    cache = RadixPrefixCache(
        page_size=2,
        retain_pages=refs.retain,
        release_pages=refs.release,
    )
    first = cache.insert_prefix((1, 2, 3, 4), (0, 1))
    second = cache.insert_prefix((1, 2, 8, 9), (5, 6))

    assert first.handle is not None
    assert first.handle.page_ids == (0, 1)
    assert second.handle is not None
    assert second.handle.page_ids == (0, 6)
    assert second.already_cached_pages == 1
    assert second.inserted_pages == 1
    assert refs.refs == {0: 1, 1: 1, 6: 1}
    cache.assert_integrity()


def _manager(*, pages: int = 6) -> KVCacheManager:
    return KVCacheManager(
        num_pages=pages,
        page_size=2,
        max_sequences=4,
        max_model_len=16,
        device="cpu",
        enable_prefix_cache=True,
    )


def test_manager_radix_hit_shares_pages_and_enforces_page_cow() -> None:
    manager = _manager()
    tokens = (1, 2, 3, 4, 5)
    first = manager.admit_with_prefix("first", tokens)
    assert first.cached_tokens == 0
    manager.ensure_tokens("first", 4)
    first_pages = manager.page_ids("first")
    manager.publish_prefix("first", tokens, cached_tokens=4)
    manager.release("first")

    assert manager.free_pages == 4
    assert manager.reclaimable_pages == 2
    second = manager.admit_with_prefix("second", tokens)
    assert second.cached_tokens == 4
    assert second.page_ids == first_pages
    assert manager.page_ids("second") == first_pages
    with pytest.raises(KVCacheError, match="copy-on-write"):
        manager.slot_mapping("second", 0, 1)

    next_slot = manager.slot_mapping("second", 4, 1)
    assert next_slot[0] // manager.page_size not in first_pages
    manager.assert_integrity()
    manager.release("second")
    assert manager.available_pages == manager.total_pages
    assert manager.stats().prefix is not None
    assert manager.stats().prefix.hits == 1

    assert set(manager.evict_prefix_pages(2)) == set(first_pages)
    assert manager.free_pages == manager.total_pages
    manager.assert_integrity()


def test_allocation_pressure_evicts_unpinned_radix_pages_before_overload() -> None:
    manager = _manager(pages=4)
    cached_tokens = (1, 2, 3, 4, 5)
    manager.admit_with_prefix("cached", cached_tokens)
    manager.ensure_tokens("cached", 4)
    manager.publish_prefix("cached", cached_tokens, cached_tokens=4)
    manager.release("cached")
    assert manager.free_pages == 2

    manager.admit_with_prefix("active", (8, 9, 10, 11, 12, 13, 14))
    manager.ensure_tokens("active", 4)
    assert manager.free_pages == 0
    manager.ensure_tokens("active", 6)
    assert len(manager.page_ids("active")) == 3
    assert manager.stats().prefix is not None
    assert manager.stats().prefix.evicted_pages == 2
    manager.assert_integrity()
    manager.release("active")
    assert manager.free_pages == manager.total_pages


def test_precomputed_key_builder_only_materializes_new_complete_pages() -> None:
    tokens = [1, 2, 3]
    builder = RadixKeyBuilder(page_size=2, token_ids=tokens)
    assert builder.complete_pages == 1
    assert builder.snapshot().pages == ((1, 2),)

    tokens.extend((4, 5, 6, 7))
    builder.sync(tokens)
    key = builder.snapshot()
    assert key.token_count == 7
    assert key.pages == ((1, 2), (3, 4), (5, 6))
    assert builder.snapshot(token_count=4).pages == ((1, 2), (3, 4))

    refs = _PageRefs()
    cache = RadixPrefixCache(
        page_size=2,
        retain_pages=refs.retain,
        release_pages=refs.release,
    )
    inserted = cache.insert_prefix(key, (10, 11, 12))
    assert inserted.inserted_pages == 3
    match = cache.match_prefix(key, reserve_tokens=0)
    assert match.cached_tokens == 6
    assert match.handle is not None
    cache.unpin(match.handle)

    with pytest.raises(RadixCacheError, match="page size"):
        cache.match_prefix(
            RadixKey.from_tokens(tokens, page_size=1),
            reserve_tokens=0,
        )


def test_radix_namespace_isolates_media_kv_and_reclaims_empty_roots() -> None:
    refs = _PageRefs()
    cache = RadixPrefixCache(
        page_size=2,
        retain_pages=refs.retain,
        release_pages=refs.release,
    )
    tokens = (1, 2, 3, 4)
    media_a = RadixKey.from_tokens(
        tokens,
        page_size=2,
        namespace="media-a",
    )
    media_b = RadixKey.from_tokens(
        tokens,
        page_size=2,
        namespace="media-b",
    )

    cache.insert_prefix(media_a, (10, 11))
    different_media = cache.match_prefix(media_b, reserve_tokens=0)
    assert different_media.cached_tokens == 0
    assert set(cache._roots) == {"", "media-a"}

    same_media = cache.match_prefix(media_a, reserve_tokens=0)
    assert same_media.cached_tokens == 4
    assert same_media.page_ids == (10, 11)
    assert same_media.handle is not None
    cache.unpin(same_media.handle)
    assert cache.evict_pages(2) == (10, 11)
    assert set(cache._roots) == {""}
    cache.assert_integrity()


def test_incremental_lru_evicts_the_cold_unpinned_leaf() -> None:
    refs = _PageRefs()
    cache = RadixPrefixCache(
        page_size=2,
        retain_pages=refs.retain,
        release_pages=refs.release,
    )
    cache.insert_prefix((1, 2, 3, 4), (10, 11))
    cache.insert_prefix((8, 9, 10, 11), (20, 21))

    hot = cache.match_prefix((1, 2, 3, 4), reserve_tokens=0)
    assert hot.handle is not None
    cache.unpin(hot.handle)

    assert cache.evict_pages(2) == (20, 21)
    assert refs.refs == {10: 1, 11: 1}
    cache.assert_integrity()


def test_replacement_handle_cannot_be_reused_for_another_prefix() -> None:
    refs = _PageRefs()
    cache = RadixPrefixCache(
        page_size=2,
        retain_pages=refs.retain,
        release_pages=refs.release,
    )
    cache.insert_prefix((1, 2, 3, 4), (10, 11))
    match = cache.match_prefix((1, 2, 3, 4), reserve_tokens=0)
    assert match.handle is not None

    with pytest.raises(RadixCacheError, match="another prefix"):
        cache.insert_prefix(
            (8, 9, 10, 11),
            (20, 21),
            pin=True,
            replace_handle=match.handle,
        )

    cache.unpin(match.handle)
    cache.assert_integrity()


def test_failed_physical_release_keeps_the_leaf_retriable() -> None:
    class FailingReleaseRefs(_PageRefs):
        fail_release = True

        def release(self, pages: tuple[int, ...]) -> None:
            if self.fail_release:
                raise RuntimeError("injected release failure")
            super().release(pages)

    refs = FailingReleaseRefs()
    cache = RadixPrefixCache(
        page_size=2,
        retain_pages=refs.retain,
        release_pages=refs.release,
    )
    cache.insert_prefix((1, 2, 3, 4), (10, 11))

    with pytest.raises(RuntimeError, match="injected"):
        cache.evict_pages(2)
    assert cache.cached_pages == 2
    assert refs.refs == {10: 1, 11: 1}
    cache.assert_integrity()

    refs.fail_release = False
    assert cache.evict_pages(2) == (10, 11)
    assert refs.refs == {}
    cache.assert_integrity()


def test_failed_admission_rolls_back_the_radix_pin_and_shared_refs() -> None:
    manager = KVCacheManager(
        num_pages=4,
        page_size=2,
        max_sequences=1,
        max_model_len=16,
        device="cpu",
        enable_prefix_cache=True,
    )
    tokens = (1, 2, 3, 4, 5)
    manager.admit_with_prefix("seed", tokens)
    manager.ensure_tokens("seed", 4)
    manager.publish_prefix("seed", tokens, cached_tokens=4)
    manager.release("seed")

    manager.admit_with_prefix("occupier", (9, 8, 7))
    with pytest.raises(CacheCapacityError):
        manager.admit_with_prefix("blocked", tokens)

    assert manager.handle("blocked") is None
    assert manager.prefix_cache is not None
    assert manager.prefix_cache.pinned_pages == 0
    manager.assert_integrity()
    manager.release("occupier")


def test_radix_randomized_branch_pin_and_full_reclamation_stress() -> None:
    rng = random.Random(0)
    refs = _PageRefs()
    cache = RadixPrefixCache(
        page_size=2,
        retain_pages=refs.retain,
        release_pages=refs.release,
    )
    prefixes: list[tuple[int, ...]] = []
    next_page = 0
    for branch in range(200):
        common_pages = rng.randrange(0, min(3, len(prefixes)) + 1)
        common = (
            prefixes[rng.randrange(len(prefixes))][: common_pages * 2]
            if prefixes and common_pages
            else ()
        )
        suffix = tuple(
            branch * 100 + token
            for token in range(2, 2 + rng.randrange(1, 7) * 2)
        )
        tokens = common + suffix
        page_count = len(tokens) // 2
        pages = tuple(range(next_page, next_page + page_count))
        next_page += page_count
        cache.insert_prefix(tokens, pages)
        prefixes.append(tokens)

    pinned = []
    for tokens in rng.sample(prefixes, 32):
        match = cache.match_prefix(tokens, reserve_tokens=0)
        assert match.cached_tokens == len(tokens)
        assert match.handle is not None
        pinned.append(match.handle)
    cache.assert_integrity()
    cache.evict_pages(cache.evictable_pages)
    cache.assert_integrity()

    for handle in pinned:
        cache.unpin(handle)
    cache.evict_pages(cache.cached_pages)
    assert cache.cached_pages == 0
    assert refs.refs == {}
    cache.assert_integrity()
