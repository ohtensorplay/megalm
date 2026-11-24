"""Prefix-cache reuse for models whose recurrent state lives outside K/V."""

import pytest

from megalm.contracts.request import InferenceRequest, SamplingParams
from megalm.runtime.cache import KVCacheError, KVCacheManager, PrefixStateStore
from megalm.runtime.radix_cache import RadixKey
from megalm.runtime.scheduler import ContinuousBatchScheduler


def _hooks() -> tuple[object, object, dict[int, str], list[tuple]]:
    captured: dict[int, str] = {}
    restored: list[tuple] = []

    def capture(request_id: str, token_count: int) -> dict[str, str]:
        payload = {"ring": f"{request_id}:{token_count}"}
        captured[token_count] = payload["ring"]
        return payload

    def restore(
        request_id: str,
        token_count: int,
        payload: object,
        slot: int,
    ) -> None:
        restored.append((request_id, token_count, payload, slot))

    return capture, restore, captured, restored


def _manager(state: PrefixStateStore | None) -> KVCacheManager:
    return KVCacheManager(
        num_pages=8,
        page_size=4,
        max_sequences=2,
        max_model_len=64,
        device="cpu",
        enable_prefix_cache=True,
        prefix_state=state,
    )


def _key(tokens: tuple[int, ...]) -> RadixKey:
    return RadixKey.from_tokens(tokens, page_size=4)


def test_snapshot_is_content_addressed_and_namespaced() -> None:
    capture, restore, _captured, restored = _hooks()
    store = PrefixStateStore(capture, restore)
    tokens = tuple(range(1, 13))
    key = _key(tokens)
    other = RadixKey.from_tokens(tokens, page_size=4, namespace="tenant-b")

    assert store.snapshot("req", key, 8)
    assert store.has(key, 8)
    assert not store.has(other, 8)
    assert store.longest(key) == 8
    assert store.longest(key, below=8) == 0

    assert store.restore("req", key, 8, slot=3)
    assert restored == [("req", 8, {"ring": "req:8"}, 3)]
    assert store.hits == 1


def test_snapshot_evicts_oldest_entry_by_byte_budget() -> None:
    capture, restore, _, _ = _hooks()
    store = PrefixStateStore(capture, restore, max_bytes=1500)
    tokens = tuple(range(1, 21))
    key = _key(tokens)

    assert store.snapshot("req", key, 4)
    assert store.snapshot("req", key, 8)
    assert len(store) == 1
    assert store.has(key, 8)
    assert not store.has(key, 4)


def test_publish_defers_partial_pages_and_admits_with_a_checkpoint() -> None:
    capture, restore, _, restored = _hooks()
    store = PrefixStateStore(capture, restore)
    manager = _manager(store)
    tokens = tuple(range(1, 22))
    key = _key(tokens)

    manager.admit_with_prefix("first", key)
    manager.ensure_tokens("first", 12)
    # The recurrent state advances with the token stream, so a partial page
    # must not be published under the completed-page boundary.
    assert manager.publish_prefix("first", key, cached_tokens=10) is None
    assert not store.has(key, 8)
    assert manager.publish_prefix("first", key, cached_tokens=12) is not None
    assert store.has(key, 12)
    manager.release("first")

    second = manager.admit_with_prefix("second", key)
    assert second.cached_tokens == 12
    assert [entry[1] for entry in restored] == [12]
    assert restored[0][3] == manager.request_slot("second")
    manager.release("second")
    manager.assert_integrity()


def test_admit_falls_back_to_the_shorter_checkpointed_prefix() -> None:
    capture, restore, _, restored = _hooks()
    store = PrefixStateStore(capture, restore)
    manager = _manager(store)
    tokens = tuple(range(1, 22))
    key = _key(tokens)

    manager.admit_with_prefix("first", key)
    manager.ensure_tokens("first", 12)
    assert manager.publish_prefix("first", key, cached_tokens=8) is not None
    assert manager.publish_prefix("first", key, cached_tokens=12) is not None
    manager.release("first")

    # Simulate the LRU dropping the newest checkpoint while its pages survive.
    store._entries.pop(
        PrefixStateStore._key(key, 12),
    )

    second = manager.admit_with_prefix("second", key)
    assert second.cached_tokens == 8
    assert [entry[1] for entry in restored] == [8]
    manager.release("second")


def test_reuse_without_any_checkpoint_is_refused() -> None:
    capture, restore, _, _ = _hooks()
    manager = _manager(PrefixStateStore(capture, restore))
    tokens = tuple(range(1, 13))
    key = _key(tokens)

    manager.admit_with_prefix("first", key)
    manager.ensure_tokens("first", 8)
    assert manager.publish_prefix("first", key, cached_tokens=8) is not None
    manager.release("first")

    empty = _manager(PrefixStateStore(capture, restore))
    admission = empty.admit_with_prefix("third", key)
    assert admission.cached_tokens == 0


def test_raw_token_lists_cannot_bypass_the_state_checkpoint() -> None:
    capture, restore, _, _ = _hooks()
    store = PrefixStateStore(capture, restore)

    with pytest.raises(KVCacheError, match="radix prefix cache"):
        KVCacheManager(
            num_pages=4,
            page_size=4,
            max_sequences=1,
            max_model_len=16,
            device="cpu",
            enable_prefix_cache=False,
            prefix_state=store,
        )

    manager = _manager(store)
    tokens = (1, 2, 3, 4, 5)
    manager.admit_with_prefix("first", tokens)
    manager.ensure_tokens("first", 4)
    assert manager.publish_prefix("first", tokens, cached_tokens=4) is not None
    manager.release("first")
    assert manager.admit_with_prefix("second", tokens).cached_tokens == 0


def test_failed_restore_releases_the_admission() -> None:
    """A restore failure must not strand a slot, pin or page attachment."""

    capture, _, _, _ = _hooks()

    def exploding_restore(
        request_id: str,
        token_count: int,
        payload: object,
        slot: int,
    ) -> None:
        raise RuntimeError("device copy failed")

    store = PrefixStateStore(capture, exploding_restore)
    manager = _manager(store)
    tokens = tuple(range(1, 13))
    key = _key(tokens)

    manager.admit_with_prefix("first", key)
    manager.ensure_tokens("first", 8)
    assert manager.publish_prefix("first", key, cached_tokens=8) is not None
    manager.release("first")

    with pytest.raises(RuntimeError, match="device copy failed"):
        manager.admit_with_prefix("second", key)
    manager.assert_integrity()

    # A clean store on the same manager can still serve the same request id,
    # so nothing from the failed admission was left attached to it.
    store._restore = lambda *args: None
    admission = manager.admit_with_prefix("second", key)
    assert admission.cached_tokens == 8
    manager.release("second")
    manager.assert_integrity()


def _request(request_id: str, token_ids: tuple[int, ...]) -> InferenceRequest:
    return InferenceRequest(
        request_id=request_id,
        model="inkling2",
        input_token_ids=token_ids,
        max_new_tokens=1,
        sampling=SamplingParams(),
    )


def test_scheduler_splits_the_final_prompt_chunk_on_a_page_boundary() -> None:
    manager = KVCacheManager(
        num_pages=16,
        page_size=4,
        max_sequences=2,
        max_model_len=64,
        device="cpu",
        enable_prefix_cache=False,
    )
    scheduler = ContinuousBatchScheduler(
        manager,
        eos_token_id=99,
        max_num_sequences=1,
        max_num_batched_tokens=32,
        prefill_chunk_size=32,
        emit_events=False,
        align_prefill_for_state=True,
    )
    scheduler.submit(_request("tail", tuple(range(1, 23))))

    first = scheduler.schedule()
    assert first is not None
    assert first.scheduled_tokens == (20,)
    scheduler.complete_host(first, [0])
    second = scheduler.schedule()
    assert second is not None
    assert second.scheduled_tokens == (2,)


def test_scheduler_keeps_the_plain_chunk_without_state_alignment() -> None:
    manager = KVCacheManager(
        num_pages=16,
        page_size=4,
        max_sequences=2,
        max_model_len=64,
        device="cpu",
        enable_prefix_cache=False,
    )
    scheduler = ContinuousBatchScheduler(
        manager,
        eos_token_id=99,
        max_num_sequences=1,
        max_num_batched_tokens=32,
        prefill_chunk_size=32,
        emit_events=False,
    )
    scheduler.submit(_request("flat", tuple(range(1, 23))))
    first = scheduler.schedule()
    assert first is not None
    assert first.scheduled_tokens == (22,)


def test_cache_hit_checkpoints_extended_prompt_boundary() -> None:
    capture, restore, _captured, _restored = _hooks()

    manager = KVCacheManager(
        num_pages=16,
        page_size=4,
        max_sequences=2,
        max_model_len=64,
        device="cpu",
        enable_prefix_cache=True,
        prefix_state=PrefixStateStore(capture, restore),
    )
    scheduler = ContinuousBatchScheduler(
        manager,
        eos_token_id=99,
        max_num_sequences=1,
        max_num_batched_tokens=32,
        prefill_chunk_size=32,
        emit_events=False,
        align_prefill_for_state=True,
    )
    scheduler.submit(_request("seed", tuple(range(1, 23))))
    first = scheduler.schedule()
    assert first is not None
    assert first.scheduled_tokens == (20,)
    scheduler.complete_host(first, [0])
    second = scheduler.schedule()
    assert second is not None
    assert second.scheduled_tokens == (2,)
    scheduler.complete_host(second, [99])

    extended = tuple(range(1, 21)) + (88, 77, 76, 75)
    scheduler.submit(_request("reuse", extended + (74, 73)))
    sequence = scheduler._sequences["reuse"]
    assert sequence.prefix_hit_tokens == 20
    step = scheduler.schedule()
    assert step is not None
    assert step.scheduled_tokens == (4,)
    scheduler.complete_host(step, [0])
    tail = scheduler.schedule()
    assert tail is not None
    assert tail.scheduled_tokens == (2,)
    scheduler.complete_host(tail, [99])

    scheduler.submit(_request("next", extended + (72, 71)))
    assert scheduler._sequences["next"].prefix_hit_tokens == 24
    step = scheduler.schedule()
    assert step is not None
    assert step.scheduled_tokens == (2,)
