import time

import pytest
import tensorplay

from megalm.contracts.request import (
    EventKind,
    FinishReason,
    InferenceRequest,
    SamplingParams,
)
from megalm.runtime.batch import BatchContractError, BatchPhase, ExecutionBatch
from megalm.runtime.cache import KVCacheManager, PagedKVCache
from megalm.runtime.scheduler import (
    ContinuousBatchScheduler,
    SchedulerOverloaded,
)
from megalm.runtime.speculative import prompt_lookup_draft


def _request(
    request_id: str,
    token_ids: tuple[int, ...],
    *,
    max_new_tokens: int = 2,
    priority: int = 0,
    deadline_ns: int | None = None,
    sampling: SamplingParams | None = None,
) -> InferenceRequest:
    return InferenceRequest(
        request_id=request_id,
        model="qwen3",
        input_token_ids=token_ids,
        max_new_tokens=max_new_tokens,
        sampling=sampling or SamplingParams(),
        priority=priority,
        deadline_ns=deadline_ns,
    )


def _manager(*, pages: int = 8, sequences: int = 4) -> KVCacheManager:
    return KVCacheManager(
        num_pages=pages,
        page_size=4,
        max_sequences=sequences,
        max_model_len=32,
        device="cpu",
        enable_prefix_cache=False,
    )


def test_physical_cache_geometry_and_transactional_page_growth() -> None:
    physical = PagedKVCache(
        num_layers=2,
        num_pages=4,
        page_size=4,
        num_kv_heads=2,
        head_dim=8,
        dtype=tensorplay.float16,
        device="cpu",
    )
    manager = _manager(pages=4)
    assert physical.geometry.token_capacity == 16
    assert physical.key.shape == (2, 4, 4, 2, 8)
    assert physical.nbytes == physical.key.nbytes + physical.value.nbytes

    manager.admit("r1")
    assert manager.free_pages == 4
    assert manager.slot_mapping("r1", 0, 5) == (0, 1, 2, 3, 4)
    assert manager.page_ids("r1") == (0, 1)
    assert manager.free_pages == 2
    manager.assert_integrity()
    manager.release("r1")
    assert manager.free_pages == manager.total_pages


def test_speculative_tail_trim_releases_only_whole_uncommitted_pages() -> None:
    manager = _manager(pages=6)
    manager.admit("spec")
    original = manager.ensure_tokens("spec", 13)
    assert manager.page_ids("spec") == (0, 1, 2, 3)
    assert manager.free_pages == 2

    trimmed = manager.trim_tokens("spec", 5)

    assert trimmed.handle_id == original.handle_id
    assert trimmed.generation == original.generation + 1
    assert manager.page_ids("spec") == (0, 1)
    assert manager.free_pages == 4
    assert manager.slot_mapping_reserved("spec", 0, 5) == (0, 1, 2, 3, 4)
    manager.assert_integrity()


def test_prompt_lookup_uses_longest_repeated_suffix() -> None:
    assert prompt_lookup_draft(
        (1, 2, 3, 1, 2, 3),
        max_draft_tokens=3,
        min_ngram_size=2,
        max_ngram_size=3,
    ) == (1, 2, 3)
    assert prompt_lookup_draft(
        (1, 2, 3, 4),
        max_draft_tokens=3,
        min_ngram_size=2,
        max_ngram_size=3,
    ) == ()
    assert prompt_lookup_draft(
        (9, 9, 9, 9, 9, 9, 9),
        max_draft_tokens=3,
        min_ngram_size=2,
        max_ngram_size=4,
    ) == (9, 9, 9)
    assert prompt_lookup_draft(
        (1, 2, 3, 1, 2, 3),
        max_draft_tokens=3,
        min_ngram_size=2,
        max_ngram_size=3,
        excluded_drafts={(1, 2)},
    ) == ()


def test_prompt_lookup_full_agreement_checks_the_entire_draft() -> None:
    tokens = (1, 2, 3, 4, 1, 2, 3, 5, 1)
    assert prompt_lookup_draft(
        tokens,
        max_draft_tokens=3,
        min_ngram_size=1,
        max_ngram_size=1,
        min_candidate_support=2,
    ) in {(2, 3, 4), (2, 3, 5)}
    assert prompt_lookup_draft(
        tokens,
        max_draft_tokens=3,
        min_ngram_size=1,
        max_ngram_size=1,
        min_candidate_support=2,
        min_candidate_agreement=1.0,
    ) == ()


def test_scheduler_commits_verified_ngram_prefix_and_trims_tail() -> None:
    manager = _manager(pages=8)
    scheduler = ContinuousBatchScheduler(
        manager,
        eos_token_id=99,
        max_num_sequences=1,
        max_num_batched_tokens=8,
        prefill_chunk_size=8,
        emit_events=False,
    )
    scheduler.submit(
        _request(
            "spec",
            (1, 2, 3, 1, 2),
            max_new_tokens=6,
        )
    )
    prefill = scheduler.schedule()
    assert prefill is not None
    scheduler.complete_host(prefill, [3])
    decode = scheduler.schedule()
    assert decode is not None

    speculative = scheduler.prepare_ngram_speculative(
        decode,
        num_speculative_tokens=3,
        min_ngram_size=2,
        max_ngram_size=3,
        min_candidate_support=1,
        probe_interval=1,
    )

    assert speculative.is_speculative
    assert speculative.draft_token_ids == ((1, 2),)
    assert speculative.execution.input_ids.tolist() == [3, 1, 2]
    completion = scheduler.complete_speculative_host(
        speculative,
        ((1, 2, 50),),
        accepted_drafts=(2,),
    )
    assert completion.output_token_ids == (1, 2, 50)
    assert scheduler.result_tokens("spec") == (3, 1, 2, 50)
    assert not scheduler.is_finished("spec")
    assert manager.page_ids("spec") == (0, 1)
    manager.assert_integrity()


def test_scheduler_builds_batched_ngram_chains_with_uniform_length() -> None:
    manager = _manager(pages=16, sequences=4)
    scheduler = ContinuousBatchScheduler(
        manager,
        eos_token_id=99,
        max_num_sequences=4,
        max_num_batched_tokens=16,
        prefill_chunk_size=8,
        emit_events=False,
    )
    # Both prompts end in a repeated (1, 2, 3) tail: each decode step
    # re-derives a (1, 2, 3) prompt-lookup draft, so chains verify in one
    # batched step.
    scheduler.submit(
        _request("spec-a", (1, 2, 3, 1, 2), max_new_tokens=6)
    )
    scheduler.submit(
        _request("spec-b", (9, 1, 2, 3, 1, 2), max_new_tokens=6)
    )
    prefill = scheduler.schedule()
    assert prefill is not None
    scheduler.complete_host(prefill, [3, 3])
    decode = scheduler.schedule()
    assert decode is not None
    assert decode.execution.num_sequences == 2

    speculative = scheduler.prepare_ngram_speculative(
        decode,
        num_speculative_tokens=3,
        min_ngram_size=2,
        max_ngram_size=3,
        min_candidate_support=1,
        min_candidate_agreement=0.0,
        probe_interval=1,
    )

    assert speculative.is_speculative
    assert speculative.execution.num_sequences == 2
    assert speculative.draft_token_ids == ((1, 2), (1, 2))
    # Each chain packs (context token, *draft) into one contiguous span.
    assert speculative.execution.input_ids.tolist() == [3, 1, 2, 3, 1, 2]
    assert speculative.execution.query_start_loc == (0, 3, 6)
    completion = scheduler.complete_speculative_host(
        speculative,
        ((1, 2, 50), (1, 2, 60)),
        accepted_drafts=(2, 2),
    )
    assert completion.output_request_ids == (
        "spec-a",
        "spec-a",
        "spec-a",
        "spec-b",
        "spec-b",
        "spec-b",
    )
    assert completion.output_token_ids == (1, 2, 50, 1, 2, 60)
    assert scheduler.result_tokens("spec-a") == (3, 1, 2, 50)
    assert scheduler.result_tokens("spec-b") == (3, 1, 2, 60)
    assert not scheduler.is_finished("spec-a")
    assert not scheduler.is_finished("spec-b")
    manager.assert_integrity()


def test_scheduler_prepares_ngram_chain_for_sampled_decoding() -> None:
    manager = _manager(pages=8)
    scheduler = ContinuousBatchScheduler(
        manager,
        eos_token_id=99,
        max_num_sequences=1,
        max_num_batched_tokens=8,
        prefill_chunk_size=8,
        emit_events=False,
    )
    scheduler.submit(
        _request(
            "spec",
            (1, 2, 3, 1, 2),
            max_new_tokens=6,
            sampling=SamplingParams(temperature=0.7, seed=7),
        )
    )
    prefill = scheduler.schedule()
    assert prefill is not None
    scheduler.complete_host(prefill, [3])
    decode = scheduler.schedule()
    assert decode is not None

    speculative = scheduler.prepare_ngram_speculative(
        decode,
        num_speculative_tokens=3,
        min_ngram_size=2,
        max_ngram_size=3,
        min_candidate_support=1,
        probe_interval=1,
    )

    assert speculative.is_speculative
    assert speculative.draft_token_ids == ((1, 2),)
    assert speculative.sampling[0].temperature == 0.7
    assert speculative.histories == ((),)


def test_scheduler_reserves_native_mtp_chain_on_gdn_shadow_state() -> None:
    manager = KVCacheManager(
        num_pages=8,
        page_size=4,
        max_sequences=1,
        max_model_len=32,
        device="cpu",
        enable_prefix_cache=False,
        gdn_state_slots=2,
        gdn_states_per_request=2,
    )
    scheduler = ContinuousBatchScheduler(
        manager,
        eos_token_id=99,
        max_num_sequences=1,
        max_num_batched_tokens=8,
        prefill_chunk_size=8,
        emit_events=False,
    )
    scheduler.submit(
        _request("mtp", (1, 2, 3), max_new_tokens=5)
    )
    prefill = scheduler.schedule()
    assert prefill is not None
    scheduler.complete_host(prefill, [4])
    decode = scheduler.schedule()
    assert decode is not None

    speculative = scheduler.prepare_mtp_speculative(
        decode,
        draft_token_id=5,
    )

    assert speculative.is_speculative
    assert speculative.draft_token_ids == ((5,),)
    assert speculative.execution.input_ids.tolist() == [4, 5]
    assert speculative.execution.context_lengths == (3,)
    assert speculative.execution.gdn_state_indices is not None
    assert tuple(
        speculative.execution.gdn_state_indices[0].tolist()
    ) == manager.gdn_state_indices("mtp", shadow=True)
    manager.assert_integrity()


def test_scheduler_builds_decode_first_mixed_chunked_batch() -> None:
    manager = _manager()
    scheduler = ContinuousBatchScheduler(
        manager,
        eos_token_id=99,
        max_num_sequences=4,
        max_num_batched_tokens=8,
        prefill_chunk_size=4,
    )
    scheduler.submit(_request("decode", (1, 2, 3)))
    first = scheduler.schedule()
    assert first is not None
    assert first.execution.phases == (BatchPhase.PREFILL,)
    assert first.should_sample == (True,)
    scheduler.complete(first, tensorplay.tensor([10]))

    scheduler.submit(_request("prefill", (4, 5, 6, 7, 8, 9), max_new_tokens=1))
    mixed = scheduler.schedule()
    assert mixed is not None
    assert mixed.request_ids == ("decode", "prefill")
    assert mixed.execution.phases == (
        BatchPhase.DECODE,
        BatchPhase.PREFILL,
    )
    assert mixed.scheduled_tokens == (1, 4)
    assert mixed.should_sample == (True, False)
    scheduler.complete(mixed, tensorplay.tensor([11, -1]))
    assert scheduler.is_finished("decode")
    assert scheduler.result_finish_reason("decode") is FinishReason.LENGTH

    final_prefill = scheduler.schedule()
    assert final_prefill is not None
    assert final_prefill.request_ids == ("prefill",)
    assert final_prefill.scheduled_tokens == (2,)
    assert final_prefill.should_sample == (True,)
    scheduler.complete(final_prefill, tensorplay.tensor([12]))
    assert scheduler.is_finished("prefill")
    assert manager.free_pages == manager.total_pages
    manager.assert_integrity()


def test_mixed_batch_caps_total_prefill_to_one_latency_chunk() -> None:
    scheduler = ContinuousBatchScheduler(
        _manager(pages=16, sequences=4),
        eos_token_id=99,
        max_num_sequences=4,
        max_num_batched_tokens=16,
        prefill_chunk_size=2,
    )
    scheduler.submit(_request("decode", (1,), max_new_tokens=2))
    first = scheduler.schedule()
    assert first is not None
    scheduler.complete(first, tensorplay.tensor([10]))
    scheduler.submit(_request("prefill-a", (2, 3, 4, 5)))
    scheduler.submit(_request("prefill-b", (6, 7, 8, 9)))

    mixed = scheduler.schedule()

    assert mixed is not None
    assert mixed.execution.phases[0] is BatchPhase.DECODE
    assert sum(
        token_count
        for token_count, phase in zip(
            mixed.scheduled_tokens,
            mixed.execution.phases,
            strict=True,
        )
        if phase is BatchPhase.PREFILL
    ) == 2


def test_mixed_batch_uses_independent_decode_sla_chunk() -> None:
    scheduler = ContinuousBatchScheduler(
        _manager(pages=16, sequences=4),
        eos_token_id=99,
        max_num_sequences=4,
        max_num_batched_tokens=16,
        prefill_chunk_size=8,
        mixed_prefill_chunk_size=2,
    )
    scheduler.submit(_request("decode", (1,), max_new_tokens=2))
    first = scheduler.schedule()
    assert first is not None
    scheduler.complete(first, tensorplay.tensor([10]))
    scheduler.submit(_request("prefill", tuple(range(2, 10))))

    mixed = scheduler.schedule()

    assert mixed is not None
    assert mixed.scheduled_tokens == (1, 2)


def test_scheduler_priority_does_not_duplicate_a_chunk_in_one_round() -> None:
    scheduler = ContinuousBatchScheduler(
        _manager(),
        eos_token_id=99,
        max_num_sequences=4,
        max_num_batched_tokens=8,
        prefill_chunk_size=4,
    )
    scheduler.submit(_request("high", tuple(range(12)), priority=10))
    scheduler.submit(_request("low", tuple(range(12)), priority=0))
    batch = scheduler.schedule()
    assert batch is not None
    assert batch.request_ids == ("high", "low")
    assert len(set(batch.request_ids)) == len(batch.request_ids)


def test_partial_prefill_capacity_deadlock_reclaims_recompute_victim() -> None:
    manager = _manager(pages=4, sequences=2)
    scheduler = ContinuousBatchScheduler(
        manager,
        eos_token_id=99,
        max_num_sequences=2,
        max_num_batched_tokens=16,
        prefill_chunk_size=8,
    )
    scheduler.submit(_request("a", tuple(range(12)), max_new_tokens=1))
    scheduler.submit(_request("b", tuple(range(20, 32)), max_new_tokens=1))

    split = scheduler.schedule()
    assert split is not None
    assert split.request_ids == ("a", "b")
    assert split.scheduled_tokens == (8, 8)
    scheduler.complete(split, tensorplay.tensor([-1, -1]))
    assert manager.free_pages == 0

    recovered = scheduler.schedule()

    assert recovered is not None
    assert recovered.request_ids[0] == "a"
    assert recovered.scheduled_tokens[0] == 4
    assert scheduler.prefill_deadlock_recoveries == 1
    assert scheduler.prefill_recomputed_tokens == 8
    scheduler.complete_host(
        recovered,
        [
            10 if should_sample else -1
            for should_sample in recovered.should_sample
        ],
    )
    while not scheduler.is_finished("b"):
        batch = scheduler.schedule()
        assert batch is not None
        samples = [
            11 if should_sample else -1
            for should_sample in batch.should_sample
        ]
        scheduler.complete_host(batch, samples)
    assert manager.free_pages == manager.total_pages
    manager.assert_integrity()


def test_cancel_and_deadline_release_the_single_cache_handle() -> None:
    manager = _manager()
    scheduler = ContinuousBatchScheduler(
        manager,
        eos_token_id=99,
        max_num_sequences=4,
        max_num_batched_tokens=8,
        prefill_chunk_size=4,
    )
    scheduler.submit(_request("cancel", (1, 2, 3)))
    event = scheduler.cancel("cancel")
    assert event.kind is EventKind.TERMINAL
    assert event.finish_reason is FinishReason.CANCELLED
    assert manager.handle("cancel") is None

    scheduler.submit(
        _request(
            "expired",
            (1, 2, 3),
            deadline_ns=time.monotonic_ns() - 1,
        )
    )
    assert scheduler.schedule() is None
    assert scheduler.is_finished("expired")
    assert scheduler.result_finish_reason("expired") is FinishReason.CANCELLED
    assert manager.free_pages == manager.total_pages
    assert scheduler.metrics_snapshot()["deadline_expirations"] == 1


def test_admission_overload_waits_instead_of_rejecting() -> None:
    # A saturated KV arena is transient: a valid request is kept alive and
    # retried on a later tick rather than being rejected with a 503.
    scheduler = ContinuousBatchScheduler(
        _manager(sequences=1),
        eos_token_id=99,
        max_num_sequences=1,
        max_num_batched_tokens=4,
        prefill_chunk_size=4,
    )
    scheduler.submit(_request("first", (1,)))
    scheduler.submit(_request("second", (2,)))
    assert scheduler.metrics_snapshot()["cache_capacity_waits"] == 1
    assert scheduler.metrics_snapshot()["overload_rejections"] == 0
    assert not scheduler.is_finished("second")


def test_scheduler_reuses_radix_prefix_and_prefills_only_the_tail() -> None:
    manager = KVCacheManager(
        num_pages=8,
        page_size=2,
        max_sequences=4,
        max_model_len=32,
        device="cpu",
        enable_prefix_cache=True,
    )
    scheduler = ContinuousBatchScheduler(
        manager,
        eos_token_id=99,
        max_num_sequences=4,
        max_num_batched_tokens=16,
        prefill_chunk_size=16,
    )
    prompt = (1, 2, 3, 4, 5)
    scheduler.submit(_request("first", prompt, max_new_tokens=1))
    first = scheduler.schedule()
    assert first is not None
    assert first.scheduled_tokens == (5,)
    scheduler.complete(first, tensorplay.tensor([10]))
    assert scheduler.is_finished("first")

    scheduler.submit(_request("second", prompt, max_new_tokens=1))
    second = scheduler.schedule()
    assert second is not None
    assert second.scheduled_tokens == (1,)
    assert second.execution.context_lengths == (4,)
    scheduler.complete(second, tensorplay.tensor([10]))
    assert scheduler.is_finished("second")
    assert manager.available_pages == manager.total_pages
    manager.assert_integrity()


def test_ignore_eos_generates_until_the_explicit_length_limit() -> None:
    scheduler = ContinuousBatchScheduler(
        _manager(),
        eos_token_id=99,
        max_num_sequences=1,
        max_num_batched_tokens=4,
        prefill_chunk_size=4,
    )
    request = InferenceRequest(
        request_id="ignore-eos",
        model="qwen3",
        input_token_ids=(1,),
        max_new_tokens=2,
        sampling=SamplingParams(ignore_eos=True),
    )
    scheduler.submit(request)
    prefill = scheduler.schedule()
    assert prefill is not None
    scheduler.complete(prefill, tensorplay.tensor([99]))
    assert not scheduler.is_finished("ignore-eos")

    decode = scheduler.schedule()
    assert decode is not None
    scheduler.complete(decode, tensorplay.tensor([99]))
    assert scheduler.is_finished("ignore-eos")
    assert scheduler.result_finish_reason("ignore-eos") is FinishReason.LENGTH


def test_structurally_complete_request_stops_without_eos() -> None:
    """A terminal output grammar finishes the request with STOP."""

    scheduler = ContinuousBatchScheduler(
        _manager(),
        eos_token_id=99,
        max_num_sequences=1,
        max_num_batched_tokens=4,
        prefill_chunk_size=4,
    )
    scheduler.submit(_request("structured", (1, 2), max_new_tokens=10))
    batch = scheduler.schedule()
    assert batch is not None
    completion = scheduler.complete_host(
        batch,
        [10],
        structurally_complete=frozenset({"structured"}),
    )
    assert completion.finish_reasons == (FinishReason.STOP,)
    assert completion.finished_request_ids == ("structured",)


def test_scheduler_accepts_multiple_eos_ids() -> None:
    """MiniCPM5 declares two EOS ids; either one terminates the request."""

    scheduler = ContinuousBatchScheduler(
        _manager(),
        eos_token_id=frozenset({99, 130073}),
        max_num_sequences=1,
        max_num_batched_tokens=4,
        prefill_chunk_size=4,
    )
    scheduler.submit(_request("multi-eos", (1, 2), max_new_tokens=10))
    batch = scheduler.schedule()
    assert batch is not None

    completion = scheduler.complete_host(batch, [130073])

    assert completion.finish_reasons == (FinishReason.STOP,)
    assert completion.finished_request_ids == ("multi-eos",)


def test_compact_scheduler_completion_avoids_engine_events() -> None:
    scheduler = ContinuousBatchScheduler(
        _manager(),
        eos_token_id=99,
        max_num_sequences=1,
        max_num_batched_tokens=4,
        prefill_chunk_size=4,
        emit_events=False,
    )
    scheduler.submit(_request("compact", (1, 2), max_new_tokens=1))
    assert scheduler.drain_events() == ()
    batch = scheduler.schedule()
    assert batch is not None

    completion = scheduler.complete_host(batch, [10])

    assert completion.events == ()
    assert completion.output_request_ids == ("compact",)
    assert completion.output_token_ids == (10,)
    assert completion.finish_reasons == (FinishReason.LENGTH,)
    assert completion.finished_request_ids == ("compact",)
    assert scheduler.drain_events() == ()


def test_scheduler_emits_top_logprobs_on_token_event() -> None:
    scheduler = ContinuousBatchScheduler(
        _manager(pages=8),
        eos_token_id=99,
        max_num_sequences=1,
        max_num_batched_tokens=8,
        prefill_chunk_size=8,
        emit_events=True,
    )
    scheduler.submit(
        _request(
            "top",
            (1, 2, 3),
            max_new_tokens=2,
            sampling=SamplingParams(logprobs=True, top_logprobs=2),
        )
    )
    batch = scheduler.schedule()
    assert batch is not None
    completion = scheduler.complete_host(
        batch,
        [5],
        sampled_logprobs=[-0.05],
        top_logprobs=(((5, -0.05), (7, -1.5)),),
    )
    expected = ((5, -0.05), (7, -1.5))
    assert completion.output_top_logprobs == (expected,)
    token_events = [
        event for event in completion.events if event.kind is EventKind.TOKEN
    ]
    assert len(token_events) == 1
    assert token_events[0].top_logprobs == (expected,)


def test_batch_publishes_host_state_slots_for_prefill_and_mixed() -> None:
    manager = _manager()
    scheduler = ContinuousBatchScheduler(
        manager,
        eos_token_id=99,
        max_num_sequences=4,
        max_num_batched_tokens=8,
        prefill_chunk_size=4,
    )
    scheduler.submit(_request("host-a", (1, 2, 3)))
    first = scheduler.schedule()
    assert first is not None
    slots = first.execution.host_state_slots
    assert slots == (manager.request_slot("host-a"),)
    assert first.execution.state_slots is not None
    assert tuple(int(slot) for slot in first.execution.state_slots) == slots
    scheduler.complete(first, tensorplay.tensor([10]))

    scheduler.submit(_request("host-b", (4, 5, 6, 7, 8, 9), max_new_tokens=1))
    mixed = scheduler.schedule()
    assert mixed is not None
    expected = tuple(
        manager.request_slot(request_id) for request_id in mixed.request_ids
    )
    assert mixed.execution.host_state_slots == expected
    assert mixed.execution.state_slots is not None
    assert tuple(int(slot) for slot in mixed.execution.state_slots) == expected


def test_batch_rejects_host_state_slot_mismatch() -> None:
    with pytest.raises(BatchContractError, match="host_state_slots"):
        ExecutionBatch(
            input_ids=tensorplay.tensor((10, 11), dtype=tensorplay.int64),
            positions=tensorplay.tensor((0, 1), dtype=tensorplay.int64),
            slot_mapping=tensorplay.tensor((0, 1), dtype=tensorplay.int64),
            block_tables=tensorplay.zeros((1, 2), dtype=tensorplay.int32),
            sequence_lengths=tensorplay.tensor((2,), dtype=tensorplay.int32),
            token_active=tensorplay.ones(2, dtype=tensorplay.bool),
            query_start_loc=(0, 2),
            query_lengths=(2,),
            context_lengths=(0,),
            phases=(BatchPhase.PREFILL,),
            sample_indices=tensorplay.tensor((1,), dtype=tensorplay.int64),
            page_size=1,
            host_state_slots=(0, 1),
        )
