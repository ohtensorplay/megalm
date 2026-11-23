import pytest

from megalm.contracts.cache import (
    CacheCapacityError,
    CacheContractError,
    CacheKind,
    LogicalCacheDirectory,
)


def _directory() -> LogicalCacheDirectory:
    return LogicalCacheDirectory(
        {
            CacheKind.KV_PAGE: 3,
            CacheKind.GDN_CONV_STATE: 1,
            CacheKind.GDN_RECURRENT_STATE: 1,
            CacheKind.PREFIX_CHECKPOINT: 1,
            CacheKind.VISION_EMBEDDING: 1,
            CacheKind.REQUEST_SLOT: 2,
        }
    )


def test_transaction_commits_one_handle_and_extends_same_identity() -> None:
    directory = _directory()
    with directory.transaction("req-1") as transaction:
        transaction.reserve(CacheKind.KV_PAGE, 2)
        transaction.reserve(CacheKind.GDN_CONV_STATE)
        handle = transaction.commit()

    assert directory.get("req-1") == handle
    assert len(handle.refs(CacheKind.KV_PAGE)) == 2
    assert len(handle.refs(CacheKind.REQUEST_SLOT)) == 1

    with directory.transaction("req-1") as transaction:
        transaction.reserve(CacheKind.KV_PAGE)
        updated = transaction.commit()

    assert updated.handle_id == handle.handle_id
    assert updated.generation == handle.generation + 1
    assert len(updated.refs(CacheKind.KV_PAGE)) == 3
    assert len(directory.get("req-1").refs(CacheKind.REQUEST_SLOT)) == 1  # type: ignore[union-attr]


def test_partial_reservation_rolls_back_on_capacity_failure() -> None:
    directory = _directory()
    initial_vision = directory.available(CacheKind.VISION_EMBEDDING)
    initial_kv = directory.available(CacheKind.KV_PAGE)

    with pytest.raises(CacheCapacityError, match="insufficient kv_page"):
        with directory.transaction("req-1") as transaction:
            transaction.reserve(CacheKind.VISION_EMBEDDING)
            transaction.reserve(CacheKind.KV_PAGE, initial_kv + 1)

    assert directory.get("req-1") is None
    assert directory.available(CacheKind.VISION_EMBEDDING) == initial_vision
    assert directory.available(CacheKind.KV_PAGE) == initial_kv


def test_capacity_error_closes_transaction_even_when_caller_catches_it() -> None:
    directory = _directory()
    transaction = directory.transaction("req-1")
    transaction.reserve(CacheKind.VISION_EMBEDDING)

    with pytest.raises(CacheCapacityError):
        transaction.reserve(CacheKind.KV_PAGE, 4)
    with pytest.raises(CacheContractError, match="closed"):
        transaction.commit()

    assert directory.get("req-1") is None
    assert directory.available(CacheKind.VISION_EMBEDDING) == 1


def test_failed_extension_keeps_existing_handle_unchanged() -> None:
    directory = _directory()
    with directory.transaction("req-1") as transaction:
        transaction.reserve(CacheKind.KV_PAGE, 3)
        original = transaction.commit()

    with pytest.raises(CacheCapacityError):
        with directory.transaction("req-1") as transaction:
            transaction.reserve(CacheKind.KV_PAGE)

    assert directory.get("req-1") == original
    assert original.generation == 0


def test_transaction_detaches_speculative_tail_without_changing_identity() -> None:
    directory = _directory()
    with directory.transaction("req-1") as transaction:
        pages = transaction.reserve(CacheKind.KV_PAGE, 3)
        original = transaction.commit()

    with directory.transaction("req-1") as transaction:
        transaction.detach(pages[-2:])
        trimmed = transaction.commit()

    assert trimmed.handle_id == original.handle_id
    assert trimmed.generation == 1
    assert trimmed.refs(CacheKind.KV_PAGE) == pages[:1]
    assert directory.available(CacheKind.KV_PAGE) == 2
    directory.assert_integrity()


def test_rolled_back_detach_keeps_every_request_ref() -> None:
    directory = _directory()
    with directory.transaction("req-1") as transaction:
        pages = transaction.reserve(CacheKind.KV_PAGE, 2)
        original = transaction.commit()

    with directory.transaction("req-1") as transaction:
        transaction.detach((pages[-1],))

    assert directory.get("req-1") == original
    assert directory.available(CacheKind.KV_PAGE) == 1
    directory.assert_integrity()


def test_release_returns_every_allocation_to_its_arena() -> None:
    directory = _directory()
    with directory.transaction("req-1") as transaction:
        transaction.reserve(CacheKind.KV_PAGE, 2)
        transaction.reserve(CacheKind.PREFIX_CHECKPOINT)
        transaction.commit()

    directory.release("req-1")
    assert directory.available(CacheKind.KV_PAGE) == 3
    assert directory.available(CacheKind.PREFIX_CHECKPOINT) == 1
    assert directory.available(CacheKind.REQUEST_SLOT) == 2


def test_shared_allocation_refs_survive_independent_owner_release() -> None:
    directory = _directory()
    with directory.transaction("producer") as transaction:
        page = transaction.reserve(CacheKind.KV_PAGE)[0]
        transaction.commit()

    directory.retain((page,))
    directory.release("producer")
    assert directory.available(CacheKind.KV_PAGE) == 2
    assert directory.refcount(page) == 1

    with directory.transaction("consumer") as transaction:
        transaction.attach((page,))
        consumer = transaction.commit()
    assert consumer.refs(CacheKind.KV_PAGE) == (page,)
    assert directory.refcount(page) == 2

    directory.release_refs((page,))
    assert directory.refcount(page) == 1
    directory.release("consumer")
    assert directory.available(CacheKind.KV_PAGE) == 3
    directory.assert_integrity()


def test_request_slots_can_never_be_shared_or_attached() -> None:
    directory = _directory()
    with directory.transaction("producer") as transaction:
        producer = transaction.commit()
    slot = producer.refs(CacheKind.REQUEST_SLOT)[0]

    with pytest.raises(CacheContractError, match="shared owners"):
        directory.retain((slot,))
    with pytest.raises(CacheContractError, match="attached"):
        with directory.transaction("consumer") as transaction:
            transaction.attach((slot,))

    assert directory.get("consumer") is None
    directory.release("producer")
    directory.assert_integrity()
