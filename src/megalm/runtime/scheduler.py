from __future__ import annotations

from collections import OrderedDict, deque
from dataclasses import dataclass, field
from enum import Enum
import time
from typing import Any

import tensorplay

from megalm.contracts.cache import CacheCapacityError
from megalm.contracts.lora import lora_retryable
from megalm.contracts.request import (
    EngineError,
    EngineEvent,
    FinishReason,
    InferenceRequest,
    RequestState,
    RequestStateMachine,
    SamplingParams,
)
from megalm.runtime.batch import BatchPhase, ExecutionBatch
from megalm.runtime.cache import KVCacheError, KVCacheManager
from megalm.runtime.lora import LoRAError
from megalm.runtime.radix_cache import RadixKeyBuilder
from megalm.runtime.speculative import DraftSource, PromptLookupIndex


class SchedulerError(RuntimeError):
    """Base scheduler failure."""


class SchedulerOverloaded(SchedulerError):
    """Admission failed before execution because no cache page is available."""


class SchedulerLoRAError(SchedulerError):
    """A LoRA admission failure with a stable, client-facing error code."""

    def __init__(self, error: EngineError) -> None:
        super().__init__(error.message)
        self.error = error
        self.code = error.code
        self.retryable = error.retryable


class SequenceStatus(str, Enum):
    WAITING = "waiting"
    RUNNING = "running"
    FINISHED = "finished"


@dataclass(slots=True)
class Sequence:
    request: InferenceRequest
    lifecycle: RequestStateMachine
    token_ids: list[int]
    prompt_length: int
    radix_key: RadixKeyBuilder | None
    cached_tokens: int = 0
    published_tokens: int = 0
    # Tokens reused from the radix cache when this request was admitted.
    # Such a request already starts on a published page boundary, so the
    # scheduler does not split its final prefill chunk to manufacture one.
    prefix_hit_tokens: int = 0
    # Resolved GPU slot of this request's LoRA adapter; -1 means base weights.
    lora_slot: int = -1
    page_table: tuple[int, ...] = ()
    generated_tokens: int = 0
    status: SequenceStatus = SequenceStatus.WAITING
    enqueue_tick: int = 0
    finish_reason: FinishReason | None = None
    rejected_speculative_drafts: set[tuple[int, ...]] = field(
        default_factory=set
    )
    prompt_lookup_index: PromptLookupIndex | None = None
    prompt_lookup_probe_cooldown: int = 0
    # Incrementally maintained unique-token counts for repetition penalty.
    # Rebuilding `sorted(set(token_ids))` per decode step costs O(L log L);
    # tracking counts keeps each committed token O(1).
    penalty_counts: dict[int, int] = field(default_factory=dict)
    _penalty_unique: tuple[int, ...] = ()
    _penalty_dirty: bool = True

    def record_generated_token(self, token_id: int) -> None:
        counts = self.penalty_counts
        counts[token_id] = counts.get(token_id, 0) + 1
        self._penalty_dirty = True

    def penalty_history(self) -> tuple[int, ...]:
        if self._penalty_dirty:
            self._penalty_unique = tuple(sorted(self.penalty_counts))
            self._penalty_dirty = False
        return self._penalty_unique

    def penalty_tokens_and_counts(
        self,
    ) -> tuple[tuple[int, ...], tuple[int, ...]]:
        """Unique tokens sorted by id, aligned with their occurrence counts."""

        unique = self.penalty_history()
        counts = self.penalty_counts
        return unique, tuple(counts[token] for token in unique)

    def has_logit_penalty(self) -> bool:
        sampling = self.sampling
        return (
            sampling.repetition_penalty != 1.0
            or sampling.frequency_penalty != 0.0
            or sampling.presence_penalty != 0.0
        )

    @property
    def request_id(self) -> str:
        return self.request.request_id

    @property
    def needs_prefill(self) -> bool:
        return self.cached_tokens < len(self.token_ids)

    @property
    def sampling(self) -> SamplingParams:
        return self.request.sampling


@dataclass(frozen=True, slots=True)
class ScheduledBatch:
    execution: ExecutionBatch
    request_ids: tuple[str, ...]
    scheduled_tokens: tuple[int, ...]
    should_sample: tuple[bool, ...]
    sampling: tuple[SamplingParams | None, ...]
    histories: tuple[tuple[int, ...], ...]
    draft_token_ids: tuple[tuple[int, ...], ...] = ()
    history_counts: tuple[tuple[int, ...], ...] = ()

    @property
    def is_speculative(self) -> bool:
        return bool(self.draft_token_ids)


@dataclass(frozen=True, slots=True)
class Completion:
    events: tuple[EngineEvent, ...]
    finished_request_ids: tuple[str, ...]
    output_request_ids: tuple[str, ...]
    output_token_ids: tuple[int, ...]
    finish_reasons: tuple[FinishReason | None, ...]
    output_logprobs: tuple[float | None, ...] = ()
    # Per sampled token: top-k (token_id, logprob) pairs. Empty when the
    # request did not ask for top_logprobs.
    output_top_logprobs: tuple[tuple[tuple[int, float], ...], ...] = ()
    # Decision reads carry their label ids and label log-probabilities in the
    # output slot instead of one sampled token. Empty for normal generation.
    output_label_ids: tuple[tuple[int, ...], ...] = ()
    output_label_logprobs: tuple[tuple[float, ...], ...] = ()


class _CUDABatchStaging:
    """Fixed-address pinned/device metadata buffers for the Leader hot path."""

    def __init__(
        self,
        *,
        device: tensorplay.device,
        max_tokens: int,
        max_sequences: int,
        max_blocks_per_sequence: int,
        uses_gdn_state: bool,
    ) -> None:
        self.device = device
        self.max_blocks_per_sequence = max_blocks_per_sequence
        self.input_ids_host = tensorplay.empty(
            max_tokens,
            dtype=tensorplay.int64,
            pin_memory=True,
        )
        self.positions_host = tensorplay.empty(
            self.input_ids_host.shape,
            dtype=self.input_ids_host.dtype,
            pin_memory=True,
        )
        self.slots_host = tensorplay.empty(
            self.input_ids_host.shape,
            dtype=self.input_ids_host.dtype,
            pin_memory=True,
        )
        self.sequence_lengths_host = tensorplay.empty(
            max_sequences,
            dtype=tensorplay.int32,
            pin_memory=True,
        )
        self.sample_indices_host = tensorplay.empty(
            max_sequences,
            dtype=tensorplay.int64,
            pin_memory=True,
        )
        self.query_starts_host = tensorplay.empty(
            max_sequences + 1,
            dtype=tensorplay.int32,
            pin_memory=True,
        )
        self.gdn_state_indices_host = (
            tensorplay.empty(
                (max_sequences, 2),
                dtype=tensorplay.int32,
                pin_memory=True,
            )
            if uses_gdn_state
            else None
        )
        self.state_slots_host = tensorplay.empty(
            max_sequences,
            dtype=tensorplay.int32,
            pin_memory=True,
        )
        self.block_tables_host = tensorplay.empty(
            (max_sequences, max_blocks_per_sequence),
            dtype=tensorplay.int32,
            pin_memory=True,
        )
        self.input_ids = tensorplay.empty(
            max_tokens,
            dtype=tensorplay.int64,
            device=device,
        )
        self.positions = tensorplay.empty_like(self.input_ids)
        self.slots = tensorplay.empty_like(self.input_ids)
        self.sequence_lengths = tensorplay.empty(
            max_sequences,
            dtype=tensorplay.int32,
            device=device,
        )
        self.sample_indices = tensorplay.empty(
            max_sequences,
            dtype=tensorplay.int64,
            device=device,
        )
        self.query_starts = tensorplay.empty(
            max_sequences + 1,
            dtype=tensorplay.int32,
            device=device,
        )
        self.gdn_state_indices = (
            tensorplay.empty(
                (max_sequences, 2),
                dtype=tensorplay.int32,
                device=device,
            )
            if uses_gdn_state
            else None
        )
        self.state_slots = tensorplay.empty(
            max_sequences,
            dtype=tensorplay.int32,
            device=device,
        )
        # Pure decode has one token and one sampled logit per sequence. These
        # tensors are immutable for every batch size and keep their addresses,
        # so the Leader does not rebuild/copy identical metadata every token.
        self.decode_sample_indices = tensorplay.arange(
            max_sequences,
            dtype=tensorplay.int64,
            device=device,
        )
        self.speculative_sample_indices = tensorplay.arange(
            max_tokens,
            dtype=tensorplay.int64,
            device=device,
        )
        self.decode_query_starts = tensorplay.arange(
            max_sequences + 1,
            dtype=tensorplay.int32,
            device=device,
        )
        self.block_tables = tensorplay.empty(
            (max_sequences, max_blocks_per_sequence),
            dtype=tensorplay.int32,
            device=device,
        )
        # Every row produced by Scheduler._build_batch is real. Graph bucket
        # padding is applied later by the relay kernel.
        self.token_active = tensorplay.ones(
            max_tokens,
            dtype=tensorplay.bool,
            device=device,
        )
        self._input_ids_array = self.input_ids_host.numpy()
        self._positions_array = self.positions_host.numpy()
        self._slots_array = self.slots_host.numpy()
        self._sequence_lengths_array = self.sequence_lengths_host.numpy()
        self._sample_indices_array = self.sample_indices_host.numpy()
        self._query_starts_array = self.query_starts_host.numpy()
        self._gdn_state_indices_array = (
            None
            if self.gdn_state_indices_host is None
            else self.gdn_state_indices_host.numpy()
        )
        self._state_slots_array = self.state_slots_host.numpy()
        self._block_tables_array = self.block_tables_host.numpy()
        self._last_page_tables: tuple[tuple[int, ...], ...] | None = None

    def stage(
        self,
        *,
        input_ids: list[int],
        positions: list[int],
        slots: list[int],
        sequence_lengths: list[int],
        sample_indices: list[int],
        query_starts: list[int],
        page_tables: tuple[tuple[int, ...], ...],
        all_decode: bool,
        gdn_state_indices: list[tuple[int, int]] | None = None,
        state_slots: list[int] | None = None,
    ) -> tuple[
        tensorplay.Tensor,
        tensorplay.Tensor,
        tensorplay.Tensor,
        tensorplay.Tensor,
        tensorplay.Tensor,
        tensorplay.Tensor,
        tensorplay.Tensor,
        tensorplay.Tensor,
        tensorplay.Tensor | None,
        tensorplay.Tensor,
    ]:
        num_tokens = len(input_ids)
        num_sequences = len(sequence_lengths)
        query_count = num_sequences + 1
        self._input_ids_array[:num_tokens] = input_ids
        self._positions_array[:num_tokens] = positions
        self._slots_array[:num_tokens] = slots
        self._sequence_lengths_array[:num_sequences] = sequence_lengths
        if state_slots is None:
            state_slots = list(range(num_sequences))
        if len(state_slots) != num_sequences:
            raise SchedulerError("state slot metadata length differs")
        self._state_slots_array[:num_sequences] = state_slots
        if gdn_state_indices is not None:
            if self._gdn_state_indices_array is None:
                raise SchedulerError(
                    "GDN metadata supplied to a KV-only staging arena"
                )
            self._gdn_state_indices_array[:num_sequences] = (
                gdn_state_indices
            )
        if not all_decode:
            self._sample_indices_array[:num_sequences] = sample_indices
            self._query_starts_array[:query_count] = query_starts

        page_tables_changed = page_tables != self._last_page_tables
        if page_tables_changed:
            block_tables = self._block_tables_array[:num_sequences]
            block_tables.fill(-1)
            for row, pages in enumerate(page_tables):
                if len(pages) > self.max_blocks_per_sequence:
                    raise SchedulerError(
                        "request block table exceeds staging geometry"
                    )
                block_tables[row, : len(pages)] = pages

        self.input_ids[:num_tokens].copy_(
            self.input_ids_host[:num_tokens],
            non_blocking=True,
        )
        self.positions[:num_tokens].copy_(
            self.positions_host[:num_tokens],
            non_blocking=True,
        )
        self.slots[:num_tokens].copy_(
            self.slots_host[:num_tokens],
            non_blocking=True,
        )
        self.sequence_lengths[:num_sequences].copy_(
            self.sequence_lengths_host[:num_sequences],
            non_blocking=True,
        )
        if gdn_state_indices is not None:
            assert self.gdn_state_indices is not None
            assert self.gdn_state_indices_host is not None
            self.gdn_state_indices[:num_sequences].copy_(
                self.gdn_state_indices_host[:num_sequences],
                non_blocking=True,
            )
        self.state_slots[:num_sequences].copy_(
            self.state_slots_host[:num_sequences],
            non_blocking=True,
        )
        if all_decode:
            sample_indices_device = self.decode_sample_indices[:num_sequences]
            query_starts_device = self.decode_query_starts[:query_count]
        else:
            self.sample_indices[:num_sequences].copy_(
                self.sample_indices_host[:num_sequences],
                non_blocking=True,
            )
            self.query_starts[:query_count].copy_(
                self.query_starts_host[:query_count],
                non_blocking=True,
            )
            sample_indices_device = self.sample_indices[:num_sequences]
            query_starts_device = self.query_starts[:query_count]
        if page_tables_changed:
            self.block_tables[:num_sequences].copy_(
                self.block_tables_host[:num_sequences],
                non_blocking=True,
            )
            self._last_page_tables = page_tables
        return (
            self.input_ids[:num_tokens],
            self.positions[:num_tokens],
            self.slots[:num_tokens],
            self.block_tables[:num_sequences],
            self.sequence_lengths[:num_sequences],
            self.token_active[:num_tokens],
            sample_indices_device,
            query_starts_device,
            (
                None
                if self.gdn_state_indices is None
                else self.gdn_state_indices[:num_sequences]
            ),
            self.state_slots[:num_sequences],
        )


class ContinuousBatchScheduler:
    """Leader-only decode-first scheduler with chunked prefill in spare budget."""

    _AGING_INTERVAL = 8

    def __init__(
        self,
        cache_manager: KVCacheManager,
        *,
        eos_token_id: int | frozenset[int],
        max_num_sequences: int,
        max_num_batched_tokens: int,
        prefill_chunk_size: int,
        mixed_prefill_chunk_size: int | None = None,
        emit_events: bool = True,
        align_prefill_for_state: bool = False,
        lora_manager: Any | None = None,
    ) -> None:
        if min(
            max_num_sequences,
            max_num_batched_tokens,
            prefill_chunk_size,
        ) <= 0:
            raise SchedulerError("scheduler limits must be positive")
        self.cache_manager = cache_manager
        self.lora_manager = lora_manager
        self.eos_token_id = eos_token_id
        self.eos_token_ids = (
            frozenset(eos_token_id)
            if isinstance(eos_token_id, (frozenset, set, tuple, list))
            else frozenset((eos_token_id,))
        )
        self.max_num_sequences = max_num_sequences
        self.max_num_batched_tokens = max_num_batched_tokens
        self.prefill_chunk_size = prefill_chunk_size
        self.mixed_prefill_chunk_size = min(
            prefill_chunk_size,
            (
                prefill_chunk_size
                if mixed_prefill_chunk_size is None
                else mixed_prefill_chunk_size
            ),
        )
        if self.mixed_prefill_chunk_size <= 0:
            raise SchedulerError("mixed prefill chunk size must be positive")
        self.emit_events = bool(emit_events)
        # A model whose recurrent state lives outside the paged K/V arena can
        # only publish a cacheable prefix when the step ends exactly on a page
        # boundary; the final prompt chunk is split to make that true.
        self.align_prefill_for_state = bool(align_prefill_for_state)
        self._sequences: dict[str, Sequence] = {}
        self._waiting: deque[str] = deque()
        self._running: OrderedDict[str, None] = OrderedDict()
        self._events: deque[EngineEvent] = deque()
        self._deadline_request_ids: set[str] = set()
        self._tick = 0
        self._prefill_recovery_owner: str | None = None
        self.prefill_deadlock_recoveries = 0
        self.prefill_recomputed_tokens = 0
        # Saturation / error signals for production observability.
        self.overload_rejections = 0
        self.lora_rejections = 0
        self.preemptions = 0
        self.deadline_expirations = 0
        self.cache_capacity_waits = 0
        self.offload_waits = 0
        self._cuda_staging = (
            _CUDABatchStaging(
                device=self.cache_manager.device,
                max_tokens=max_num_batched_tokens,
                max_sequences=max_num_sequences,
                max_blocks_per_sequence=(
                    self.cache_manager.max_blocks_per_sequence
                ),
                uses_gdn_state=self.cache_manager.uses_gdn_state,
            )
            if self.cache_manager.device.type == "cuda"
            else None
        )

    @property
    def is_idle(self) -> bool:
        return not self._waiting and not self._running

    @property
    def active_request_count(self) -> int:
        return sum(
            sequence.status is not SequenceStatus.FINISHED
            for sequence in self._sequences.values()
        )

    def submit(self, request: InferenceRequest) -> tuple[EngineEvent, ...]:
        if request.request_id in self._sequences:
            raise SchedulerError(f"duplicate request_id {request.request_id!r}")
        lifecycle = RequestStateMachine(request.request_id)
        if self.emit_events:
            events = [
                lifecycle.transition(RequestState.ROUTING),
                lifecycle.transition(RequestState.LEASED),
                lifecycle.transition(RequestState.ADMITTED),
                lifecycle.transition(RequestState.READY),
            ]
        else:
            events = []
            lifecycle.transition_silent(RequestState.ROUTING)
            lifecycle.transition_silent(RequestState.LEASED)
            lifecycle.transition_silent(RequestState.ADMITTED)
            lifecycle.transition_silent(RequestState.READY)
        lora_request = request.lora_request
        lora_slot = -1
        if self.lora_manager is not None and lora_request is not None:
            try:
                lora_slot = self.lora_manager.pin(lora_request)
            except LoRAError as exc:
                error = EngineError(
                    code=exc.code,
                    message=str(exc),
                    retryable=lora_retryable(exc.code),
                )
                if self.emit_events:
                    events.append(
                        lifecycle.transition(RequestState.REJECTED, error=error)
                    )
                else:
                    lifecycle.transition_silent(
                        RequestState.REJECTED, error=error
                    )
                self.lora_rejections += 1
                self._events.extend(events)
                # A dedicated exception keeps LoRA client errors out of the
                # generic 503 overload path so the gateway can answer 4xx.
                raise SchedulerLoRAError(error) from exc
        # LoRA changes K/V projections, so a prefix computed with another
        # adapter is not a valid hit.  Adapter identity is part of the radix
        # namespace; base requests keep the original namespace.
        namespace = request.prefix_fingerprint or ""
        if lora_request is not None:
            adapter_key = (
                f"id:{lora_request.lora_int_id}"
                if lora_request.lora_int_id > 0
                else f"name:{lora_request.lora_name}"
            )
            namespace = f"{namespace}|lora:{adapter_key}"
        radix_key = (
            RadixKeyBuilder(
                page_size=self.cache_manager.page_size,
                token_ids=request.input_token_ids,
                namespace=namespace,
            )
            if self.cache_manager.enable_prefix_cache
            else None
        )
        try:
            admission = self.cache_manager.admit_with_prefix(
                request.request_id,
                (
                    radix_key.snapshot()
                    if radix_key is not None
                    else request.input_token_ids
                ),
            )
        except CacheCapacityError:
            # A saturated KV arena is transient. Mirror schedule()'s waiting
            # path: keep the sequence alive with no pages and retry admission
            # on a later tick instead of rejecting a valid request.
            self.cache_capacity_waits += 1
            admission = None
        cached_tokens = 0 if admission is None else admission.cached_tokens
        sequence = Sequence(
            request=request,
            lifecycle=lifecycle,
            token_ids=list(request.input_token_ids),
            prompt_length=len(request.input_token_ids),
            radix_key=radix_key,
            cached_tokens=cached_tokens,
            published_tokens=cached_tokens,
            lora_slot=lora_slot,
            page_table=() if admission is None else admission.page_ids,
            prefix_hit_tokens=cached_tokens,
            enqueue_tick=self._tick,
        )
        for token_id in request.input_token_ids:
            sequence.record_generated_token(token_id)
        self._tick += 1
        self._sequences[request.request_id] = sequence
        if request.deadline_ns is not None:
            self._deadline_request_ids.add(request.request_id)
        self._waiting.append(request.request_id)
        self._events.extend(events)
        return tuple(events)

    def drain_events(self) -> tuple[EngineEvent, ...]:
        events = tuple(self._events)
        self._events.clear()
        return events

    def _poll_offload(self) -> None:
        outcomes = self.cache_manager.poll_offload()
        for outcome in outcomes:
            sequence = self._sequences.get(outcome.request_id)
            if sequence is None or sequence.status is SequenceStatus.FINISHED:
                continue
            sequence.cached_tokens = outcome.cached_tokens
            sequence.published_tokens = outcome.cached_tokens
            sequence.page_table = self.cache_manager.block_table(
                outcome.request_id
            )

    @property
    def is_waiting_on_offload(self) -> bool:
        return self.cache_manager.has_pending_promotions

    def schedule(self) -> ScheduledBatch | None:
        self._tick += 1
        self._poll_offload()
        self._expire_deadlines()
        if self.is_idle:
            return None
        entries: list[tuple[Sequence, int, BatchPhase, bool]] = []
        token_budget = self.max_num_batched_tokens

        for request_id in tuple(self._running):
            if len(entries) >= self.max_num_sequences or token_budget == 0:
                break
            sequence = self._sequences[request_id]
            if len(sequence.token_ids) != sequence.cached_tokens + 1:
                raise SchedulerError(
                    f"decode invariant failed for {sequence.request_id!r}"
                )
            required_tokens = sequence.cached_tokens + 1
            if sequence.cached_tokens % self.cache_manager.page_size == 0:
                try:
                    self.cache_manager.ensure_tokens(
                        sequence.request_id,
                        required_tokens,
                    )
                except CacheCapacityError:
                    self._preempt(sequence)
                    continue
                sequence.page_table = self.cache_manager.block_table(
                    sequence.request_id
                )
            entries.append((sequence, 1, BatchPhase.DECODE, True))
            token_budget -= 1

        # Decode owns the latency budget. Pure prefill may fill the global
        # token envelope, while a mixed step admits at most one configured
        # prefill chunk in total so new prompts cannot create a long stream
        # stall for already-running sequences.
        prefill_token_budget = (
            min(token_budget, self.mixed_prefill_chunk_size)
            if entries
            else token_budget
        )
        waiting_round = len(self._waiting)
        visited_waiting: set[str] = set()
        while (
            self._waiting
            and waiting_round > 0
            and len(entries) < self.max_num_sequences
            and prefill_token_budget > 0
        ):
            waiting_round -= 1
            request_id = self._pop_next_waiting(visited_waiting)
            if request_id is None:
                break
            visited_waiting.add(request_id)
            sequence = self._sequences[request_id]
            if self.cache_manager.handle(request_id) is None:
                try:
                    prefix_input = sequence.token_ids
                    if sequence.radix_key is not None:
                        sequence.radix_key.sync(sequence.token_ids)
                        prefix_input = sequence.radix_key.snapshot()
                    admission = self.cache_manager.admit_with_prefix(
                        request_id,
                        prefix_input,
                    )
                except CacheCapacityError:
                    self.cache_capacity_waits += 1
                    self._waiting.append(request_id)
                    continue
                sequence.cached_tokens = admission.cached_tokens
                sequence.published_tokens = admission.cached_tokens
                sequence.prefix_hit_tokens = admission.cached_tokens
                sequence.page_table = admission.page_ids
            if self.cache_manager.promotion_pending(request_id):
                # Lower-tier pages are still being copied into this
                # request's GPU pages.  Do not prefill them yet: the copy
                # would race with attention writing the same slots.
                self._waiting.append(request_id)
                self.offload_waits += 1
                continue
            remaining = len(sequence.token_ids) - sequence.cached_tokens
            if remaining <= 0:
                raise SchedulerError(
                    f"prefill invariant failed for {sequence.request_id!r}"
                )
            scheduled = min(
                remaining,
                token_budget,
                prefill_token_budget,
                self.prefill_chunk_size,
                self.cache_manager.schedulable_tokens(
                    sequence.request_id,
                    cached_tokens=sequence.cached_tokens,
                ),
            )
            if scheduled <= 0:
                self._waiting.append(request_id)
                continue
            if self.align_prefill_for_state:
                scheduled = self._align_prefill_tail(sequence, scheduled)
                if scheduled <= 0:
                    self._waiting.append(request_id)
                    continue
            try:
                self.cache_manager.ensure_tokens(
                    sequence.request_id,
                    sequence.cached_tokens + scheduled,
                )
            except CacheCapacityError:
                self.cache_capacity_waits += 1
                self._waiting.append(request_id)
                continue
            sequence.page_table = self.cache_manager.block_table(request_id)
            final_chunk = scheduled == remaining
            entries.append(
                (sequence, scheduled, BatchPhase.PREFILL, final_chunk)
            )
            token_budget -= scheduled
            prefill_token_budget -= scheduled
            if not final_chunk:
                self._waiting.append(request_id)

        if not entries and self._recover_partial_prefill_capacity():
            return self.schedule()
        if not entries:
            return None
        return self._build_batch(entries)

    def _align_prefill_tail(
        self,
        sequence: Sequence,
        scheduled: int,
    ) -> int:
        """End the final prompt chunk on a full page boundary.

        State-carrying models can only checkpoint their recurrent history at a
        page-aligned publish.  Splitting the last chunk moves the final partial
        page into a tiny follow-up step, so the prompt boundary itself becomes
        cacheable while the bulk prefill keeps its full chunk size.

        This also records the end of an extended prompt after a cache hit.
        A later turn can reuse that boundary even if its next tokens diverge
        before decoding reaches a full page.
        """

        page_size = self.cache_manager.page_size
        total = len(sequence.token_ids)
        start = sequence.cached_tokens
        if start + scheduled < total or total % page_size == 0:
            return scheduled
        aligned = (total // page_size) * page_size - start
        if aligned <= 0 or aligned >= scheduled:
            return scheduled
        return aligned

    def metrics_snapshot(self) -> dict[str, int]:
        return {
            "active_requests": self.active_request_count,
            "waiting_requests": len(self._waiting),
            "running_requests": len(self._running),
            "prefill_deadlock_recoveries": (
                self.prefill_deadlock_recoveries
            ),
            "prefill_recomputed_tokens": self.prefill_recomputed_tokens,
            "overload_rejections": self.overload_rejections,
            "lora_rejections": self.lora_rejections,
            "preemptions": self.preemptions,
            "deadline_expirations": self.deadline_expirations,
            "cache_capacity_waits": self.cache_capacity_waits,
            "offload_waits": self.offload_waits,
        }

    def complete(
        self,
        scheduled: ScheduledBatch,
        sampled_tokens: tensorplay.Tensor,
        sampled_logprobs: tensorplay.Tensor | None = None,
        top_logprobs: tuple[
            tuple[tuple[int, float], ...] | None, ...
        ]
        | None = None,
    ) -> Completion:
        return self.complete_host(
            scheduled,
            sampled_tokens.tolist(),
            (
                None
                if sampled_logprobs is None
                else sampled_logprobs.tolist()
            ),
            top_logprobs=top_logprobs,
        )

    def reserve_decode_tokens(
        self,
        scheduled: ScheduledBatch,
        steps: int,
    ) -> int:
        """Extend a pure-decode batch's KV reservation by up to ``steps`` tokens.

        A device-side decode loop consumes the whole chunk before the host
        sees a token, so every page the chunk touches must already be owned.
        Returns the granted chunk length; 1 means the caller must fall back to
        a normal single-token step.
        """

        if steps <= 1 or scheduled.is_speculative:
            return 1
        execution = scheduled.execution
        if not execution.all_decode:
            return 1
        grant = steps
        for request_id in scheduled.request_ids:
            sequence = self._sequences[request_id]
            grant = min(
                grant,
                sequence.request.max_new_tokens - sequence.generated_tokens,
                self.cache_manager.max_model_len - sequence.cached_tokens,
            )
        if grant <= 1:
            return 1
        # Shrink until the whole chunk fits: a partial reservation would leave
        # the loop writing past the last owned page.
        while grant > 1:
            try:
                for request_id in scheduled.request_ids:
                    sequence = self._sequences[request_id]
                    self.cache_manager.ensure_tokens(
                        request_id,
                        sequence.cached_tokens + grant,
                    )
                    # The trusted decode builder reads this cached tuple instead
                    # of the cache manager, so a grown reservation must land in
                    # it before the next schedule.
                    sequence.page_table = self.cache_manager.block_table(request_id)
            except (KVCacheError, CacheCapacityError):
                grant -= 1
                continue
            break
        if grant <= 1:
            return 1
        tables = self.cache_manager.build_block_tables(scheduled.request_ids)
        if tables.shape != execution.block_tables.shape:
            return 1
        execution.block_tables.copy_(tables)
        return grant

    def complete_host(
        self,
        scheduled: ScheduledBatch,
        sampled_tokens: list[int],
        sampled_logprobs: list[float] | None = None,
        structurally_complete: frozenset[str] = frozenset(),
        score_logprobs: tuple[tuple[float, ...] | None, ...] | None = None,
        top_logprobs: tuple[
            tuple[tuple[int, float], ...] | None, ...
        ]
        | None = None,
    ) -> Completion:
        """Commit tokens already transferred to the Leader host."""

        if len(sampled_tokens) != len(scheduled.request_ids):
            raise SchedulerError("sampler result does not match scheduled batch")
        if sampled_logprobs is not None and len(sampled_logprobs) != len(
            scheduled.request_ids
        ):
            raise SchedulerError(
                "sampler logprobs do not match scheduled batch"
            )
        if score_logprobs is not None and len(score_logprobs) != len(
            scheduled.request_ids
        ):
            raise SchedulerError(
                "score logprobs do not match scheduled batch"
            )
        events: list[EngineEvent] = []
        finished: list[str] = []
        output_request_ids: list[str] = []
        output_token_ids: list[int] = []
        finish_reasons: list[FinishReason | None] = []
        output_logprobs: list[float | None] = []
        output_top_logprobs: list[tuple[tuple[int, float], ...]] = []
        output_label_ids: list[tuple[int, ...]] = []
        output_label_logprobs: list[tuple[float, ...]] = []
        for index, request_id in enumerate(scheduled.request_ids):
            sequence = self._sequences[request_id]
            sequence.cached_tokens += scheduled.scheduled_tokens[index]
            if (
                sequence.cached_tokens
                // self.cache_manager.page_size
                > sequence.published_tokens
                // self.cache_manager.page_size
            ):
                prefix_input = sequence.token_ids
                if sequence.radix_key is not None:
                    sequence.radix_key.sync(sequence.token_ids)
                    prefix_input = sequence.radix_key.snapshot(
                        token_count=sequence.cached_tokens,
                    )
                self.cache_manager.publish_prefix(
                    request_id,
                    prefix_input,
                    cached_tokens=sequence.cached_tokens,
                )
                sequence.published_tokens = (
                    sequence.cached_tokens // self.cache_manager.page_size
                ) * self.cache_manager.page_size
            phase = scheduled.execution.phases[index]
            if phase is BatchPhase.PREFILL:
                if sequence.lifecycle.state is RequestState.READY:
                    if self.emit_events:
                        events.append(
                            sequence.lifecycle.transition(RequestState.PREFILL)
                        )
                    else:
                        sequence.lifecycle.transition_silent(
                            RequestState.PREFILL
                        )
                elif sequence.lifecycle.state is RequestState.PREEMPTED:
                    if self.emit_events:
                        events.append(
                            sequence.lifecycle.transition(RequestState.PREFILL)
                        )
                    else:
                        sequence.lifecycle.transition_silent(
                            RequestState.PREFILL
                        )
            if not scheduled.should_sample[index]:
                continue
            if phase is BatchPhase.PREFILL:
                if self.emit_events:
                    events.append(
                        sequence.lifecycle.transition(RequestState.DECODE)
                    )
                else:
                    sequence.lifecycle.transition_silent(RequestState.DECODE)
                sequence.status = SequenceStatus.RUNNING
                self._running[request_id] = None
            token_id = int(sampled_tokens[index])
            if token_id < 0:
                raise SchedulerError("sampler omitted a required token")
            sequence.token_ids.append(token_id)
            sequence.record_generated_token(token_id)
            sequence.generated_tokens += 1
            score_labels = sequence.sampling.score_labels
            emitted_top: tuple[tuple[int, float], ...] = ()
            if score_labels:
                # Decision reads surface the label ids and their raw
                # log-probabilities in the output slot instead of a generated
                # token. The sampled token is still committed to the sequence
                # so the existing lifecycle/limit accounting stays valid.
                label_logprobs = (
                    tuple(score_logprobs[index])
                    if score_logprobs is not None
                    and score_logprobs[index] is not None
                    else ()
                )
                if len(label_logprobs) != len(score_labels):
                    raise SchedulerError(
                        "score logprobs do not match the label ids"
                    )
                emitted_ids = score_labels
                emitted_logprobs = label_logprobs
                output_label_ids.append(score_labels)
                output_label_logprobs.append(label_logprobs)
            else:
                emitted_ids = (token_id,)
                row_logprob = (
                    float(sampled_logprobs[index])
                    if sampled_logprobs is not None
                    and sequence.sampling.logprobs
                    else None
                )
                emitted_logprobs = (
                    (row_logprob,) if row_logprob is not None else ()
                )
                top_row = (
                    top_logprobs[index]
                    if top_logprobs is not None
                    and index < len(top_logprobs)
                    else None
                )
                if top_row:
                    entries = list(top_row)
                    if not any(tid == token_id for tid, _ in entries):
                        # vLLM-style: the sampled token is always reported
                        # even when it falls outside the top-k cutoff.
                        sampled_lp = (
                            sampled_logprobs[index]
                            if sampled_logprobs is not None
                            and index < len(sampled_logprobs)
                            else None
                        )
                        if sampled_lp is not None:
                            entries.append((token_id, float(sampled_lp)))
                            entries.sort(
                                key=lambda item: item[1], reverse=True
                            )
                    emitted_top = tuple(entries)
                output_label_ids.append(())
                output_label_logprobs.append(())
            output_top_logprobs.append(emitted_top)
            if self.emit_events:
                events.append(
                    sequence.lifecycle.emit_tokens(
                        emitted_ids,
                        logprobs=emitted_logprobs,
                        top_logprobs=(
                            (emitted_top,) if emitted_top else ()
                        ),
                    )
                )
            else:
                sequence.lifecycle.emit_token_silent(token_id)
            reached_eos = (
                (
                    token_id in self.eos_token_ids
                    and not sequence.sampling.ignore_eos
                )
                or token_id in sequence.sampling.stop_token_ids
            )
            reached_limit = (
                sequence.generated_tokens >= sequence.request.max_new_tokens
            )
            # A constrained request whose grammar reached a terminal state is
            # complete even if the model never emits EOS (e.g. ignore_eos).
            reached_structural_end = request_id in structurally_complete
            finish_reason: FinishReason | None = None
            if reached_eos or reached_limit or reached_structural_end:
                finish_reason = (
                    FinishReason.STOP
                    if reached_eos or reached_structural_end
                    else FinishReason.LENGTH
                )
                if self.emit_events:
                    events.append(
                        sequence.lifecycle.transition(RequestState.FINISHING)
                    )
                    events.append(
                        sequence.lifecycle.transition(
                            RequestState.FINISHED,
                            finish_reason=finish_reason,
                        )
                    )
                else:
                    sequence.lifecycle.transition_silent(
                        RequestState.FINISHING
                    )
                    sequence.lifecycle.transition_silent(
                        RequestState.FINISHED,
                        finish_reason=finish_reason,
                    )
                sequence.status = SequenceStatus.FINISHED
                sequence.finish_reason = finish_reason
                self._deadline_request_ids.discard(request_id)
                self._running.pop(request_id, None)
                self.cache_manager.release(request_id)
                finished.append(request_id)
            output_request_ids.append(request_id)
            output_token_ids.append(token_id)
            finish_reasons.append(finish_reason)
            if sampled_logprobs is not None and sequence.sampling.logprobs:
                output_logprobs.append(float(sampled_logprobs[index]))
            else:
                output_logprobs.append(None)
        self._events.extend(events)
        return Completion(
            events=tuple(events),
            finished_request_ids=tuple(finished),
            output_request_ids=tuple(output_request_ids),
            output_token_ids=tuple(output_token_ids),
            finish_reasons=tuple(finish_reasons),
            output_logprobs=tuple(output_logprobs),
            output_top_logprobs=tuple(output_top_logprobs),
            output_label_ids=tuple(output_label_ids),
            output_label_logprobs=tuple(output_label_logprobs),
        )

    def result_tokens(self, request_id: str) -> tuple[int, ...]:
        sequence = self._sequences.get(request_id)
        if sequence is None:
            raise SchedulerError(f"unknown request_id {request_id!r}")
        return tuple(sequence.token_ids[sequence.prompt_length :])

    def prompt_length(self, request_id: str) -> int:
        sequence = self._sequences.get(request_id)
        if sequence is None:
            raise SchedulerError(f"unknown request_id {request_id!r}")
        return sequence.prompt_length

    def is_finished(self, request_id: str) -> bool:
        sequence = self._sequences.get(request_id)
        if sequence is None:
            raise SchedulerError(f"unknown request_id {request_id!r}")
        return sequence.status is SequenceStatus.FINISHED

    def result_finish_reason(self, request_id: str) -> FinishReason:
        sequence = self._sequences.get(request_id)
        if sequence is None:
            raise SchedulerError(f"unknown request_id {request_id!r}")
        if sequence.status is not SequenceStatus.FINISHED:
            raise SchedulerError("request has not finished")
        assert sequence.finish_reason is not None
        return sequence.finish_reason

    def forget(self, request_id: str) -> None:
        sequence = self._sequences.get(request_id)
        if sequence is None:
            return
        if sequence.status is not SequenceStatus.FINISHED:
            raise SchedulerError("cannot forget an active request")
        self._deadline_request_ids.discard(request_id)
        del self._sequences[request_id]

    def cancel(self, request_id: str) -> EngineEvent:
        sequence = self._sequences.get(request_id)
        if sequence is None:
            raise SchedulerError(f"unknown request_id {request_id!r}")
        if sequence.status is SequenceStatus.FINISHED:
            raise SchedulerError("request is already terminal")
        self._running.pop(request_id, None)
        self._waiting = deque(
            queued_id
            for queued_id in self._waiting
            if queued_id != request_id
        )
        if self.cache_manager.handle(request_id) is not None:
            self.cache_manager.release(request_id)
        event = sequence.lifecycle.transition(RequestState.CANCELLED)
        sequence.status = SequenceStatus.FINISHED
        sequence.finish_reason = FinishReason.CANCELLED
        self._deadline_request_ids.discard(request_id)
        self._events.append(event)
        return event

    def sampling_metadata(
        self,
        scheduled: ScheduledBatch,
    ) -> tuple[
        tuple[SamplingParams | None, ...],
        tuple[str, ...],
        tuple[tuple[int, ...], ...],
        tuple[tuple[int, ...], ...],
    ]:
        return (
            scheduled.sampling,
            scheduled.request_ids,
            scheduled.histories,
            scheduled.history_counts,
        )

    def prepare_ngram_speculative(
        self,
        scheduled: ScheduledBatch,
        *,
        num_speculative_tokens: int,
        min_ngram_size: int,
        max_ngram_size: int,
        min_candidate_support: int,
        probe_interval: int,
        min_candidate_agreement: float = 0.0,
        draft_source: DraftSource | None = None,
    ) -> ScheduledBatch:
        """Reserve parallel greedy prompt-lookup chains for target verification.

        When ``draft_source`` is provided (suffix mode), proposals come from the
        shared plugin source. The default ``None`` path is the original
        per-request ``PromptLookupIndex``, byte-for-byte unchanged.
        """

        if (
            scheduled.is_speculative
            or not scheduled.execution.all_decode
            or num_speculative_tokens <= 0
        ):
            return scheduled
        # Collect one verified draft chain per eligible running sequence.
        # Chains are truncated to a uniform length so the verifier can
        # consume a [batch, chain] draft matrix; the shortest eligible
        # chain bounds the step. Sampling modes verify by target
        # resampling: a draft token is accepted when the target
        # distribution samples the same token, so any sampling params
        # work on the eager verification path.
        candidates: list[tuple[Sequence, tuple[int, ...]]] = []
        for request_id in scheduled.request_ids:
            sequence = self._sequences[request_id]
            if sequence.status is not SequenceStatus.RUNNING:
                continue
            if sequence.prompt_lookup_probe_cooldown > 0:
                sequence.prompt_lookup_probe_cooldown -= 1
                continue
            remaining = (
                sequence.request.max_new_tokens - sequence.generated_tokens
            )
            draft_limit = min(
                num_speculative_tokens,
                remaining - 1,
                self.max_num_batched_tokens - 1,
                self.cache_manager.max_model_len
                - sequence.cached_tokens
                - 1,
            )
            if draft_limit <= 0:
                continue
            if draft_source is None:
                if sequence.prompt_lookup_index is None:
                    sequence.prompt_lookup_index = PromptLookupIndex(
                        max_ngram_size=max_ngram_size
                    )
                draft = sequence.prompt_lookup_index.propose(
                    sequence.token_ids,
                    max_draft_tokens=draft_limit,
                    min_ngram_size=min_ngram_size,
                    max_ngram_size=max_ngram_size,
                    excluded_drafts=sequence.rejected_speculative_drafts,
                    min_candidate_support=min_candidate_support,
                    min_candidate_agreement=min_candidate_agreement,
                )
            else:
                draft = draft_source.propose(
                    request_id,
                    sequence.token_ids,
                    max_draft_tokens=draft_limit,
                    min_ngram_size=min_ngram_size,
                    max_ngram_size=max_ngram_size,
                    excluded_drafts=sequence.rejected_speculative_drafts,
                    min_candidate_support=min_candidate_support,
                    min_candidate_agreement=min_candidate_agreement,
                )
            # K exact decode evaluations can validate at most K outputs
            # without an unconditional K+1 "bonus" forward. The final
            # proposal is still useful: the Kth target prediction either
            # confirms it or becomes the correction token, while only the
            # preceding K-1 drafts need KV input.
            if len(draft) < 2:
                sequence.prompt_lookup_probe_cooldown = max(
                    0,
                    probe_interval - 1,
                )
                continue
            sequence.prompt_lookup_probe_cooldown = 0
            candidates.append((sequence, tuple(draft[:-1])))
        if not candidates:
            return scheduled
        chain_length = min(len(verified) for _, verified in candidates)
        if chain_length < 1:
            return scheduled
        rows: list[tuple[Sequence, tuple[int, ...], int, int]] = []
        for sequence, verified in candidates:
            truncated = verified[:chain_length]
            try:
                self.cache_manager.ensure_tokens(
                    sequence.request_id,
                    sequence.cached_tokens + chain_length + 1,
                )
            except CacheCapacityError:
                continue
            rows.append(
                (
                    sequence,
                    truncated,
                    chain_length + 1,
                    sequence.cached_tokens + chain_length + 1,
                )
            )
        if not rows:
            return scheduled

        packed_tokens: list[int] = []
        positions: list[int] = []
        slots: list[int] = []
        query_starts = [0]
        page_tables: list[tuple[int, ...]] = []
        sequence_lengths: list[int] = []
        context_lengths: list[int] = []
        request_ids: list[str] = []
        samplings: list[SamplingParams] = []
        histories_rows: list[tuple[int, ...]] = []
        history_counts_rows: list[tuple[int, ...]] = []
        drafts: list[tuple[int, ...]] = []
        state_slots: list[int] = []
        gdn_state_indices: list[tuple[int, int]] | None = (
            [] if self.cache_manager.uses_gdn_state else None
        )
        for sequence, verified, query_length, end in rows:
            request_id = sequence.request_id
            start = sequence.cached_tokens
            packed_tokens.append(sequence.token_ids[start])
            packed_tokens.extend(verified)
            positions.extend(range(start, end))
            slots.extend(
                self.cache_manager.slot_mapping_reserved(
                    request_id,
                    start,
                    query_length,
                )
            )
            sequence.page_table = self.cache_manager.block_table(request_id)
            page_tables.append(sequence.page_table)
            sequence_lengths.append(end)
            context_lengths.append(start)
            query_starts.append(query_starts[-1] + query_length)
            request_ids.append(request_id)
            samplings.append(sequence.sampling)
            state_slots.append(self.cache_manager.request_slot(request_id))
            histories_rows.append(
                sequence.penalty_history()
                if sequence.has_logit_penalty()
                else ()
            )
            history_counts_rows.append(
                sequence.penalty_tokens_and_counts()[1]
                if sequence.has_logit_penalty()
                else ()
            )
            drafts.append(verified)
            if gdn_state_indices is not None:
                gdn_state_indices.append(
                    self.cache_manager.gdn_state_indices(
                        request_id,
                        shadow=True,
                    )
                )
        device = self.cache_manager.device
        staging = self._cuda_staging
        total_tokens = len(packed_tokens)
        if staging is None:
            input_ids_device = tensorplay.tensor(
                packed_tokens,
                dtype=tensorplay.int64,
                device=device,
            )
            positions_device = tensorplay.tensor(
                positions,
                dtype=tensorplay.int64,
                device=device,
            )
            slots_device = tensorplay.tensor(
                slots,
                dtype=tensorplay.int64,
                device=device,
            )
            block_tables_device = self.cache_manager.build_block_tables(
                tuple(request_ids)
            )
            sequence_lengths_device = tensorplay.tensor(
                sequence_lengths,
                dtype=tensorplay.int32,
                device=device,
            )
            token_active_device = tensorplay.ones(
                total_tokens,
                dtype=tensorplay.bool,
                device=device,
            )
            sample_indices_device = tensorplay.arange(
                total_tokens,
                dtype=tensorplay.int64,
                device=device,
            )
            query_starts_device = tensorplay.tensor(
                tuple(query_starts),
                dtype=tensorplay.int32,
                device=device,
            )
            gdn_state_indices_device = (
                None
                if gdn_state_indices is None
                else tensorplay.tensor(
                    gdn_state_indices,
                    dtype=tensorplay.int32,
                    device=device,
                )
            )
            state_slots_device = tensorplay.tensor(
                state_slots,
                dtype=tensorplay.int32,
                device=device,
            )
        else:
            (
                input_ids_device,
                positions_device,
                slots_device,
                block_tables_device,
                sequence_lengths_device,
                token_active_device,
                _,
                query_starts_device,
                gdn_state_indices_device,
                state_slots_device,
            ) = staging.stage(
                input_ids=packed_tokens,
                positions=positions,
                slots=slots,
                sequence_lengths=sequence_lengths,
                sample_indices=[row[3] - 1 for row in rows],
                query_starts=query_starts,
                page_tables=tuple(page_tables),
                all_decode=False,
                gdn_state_indices=gdn_state_indices,
                state_slots=state_slots,
            )
            sample_indices_device = (
                staging.speculative_sample_indices[:total_tokens]
            )
        execution = ExecutionBatch(
            input_ids=input_ids_device,
            positions=positions_device,
            slot_mapping=slots_device,
            block_tables=block_tables_device,
            sequence_lengths=sequence_lengths_device,
            token_active=token_active_device,
            query_start_loc=tuple(query_starts),
            query_lengths=(chain_length + 1,) * len(rows),
            context_lengths=tuple(context_lengths),
            phases=(BatchPhase.PREFILL,) * len(rows),
            sample_indices=sample_indices_device,
            page_size=self.cache_manager.page_size,
            query_start_loc_device=query_starts_device,
            gdn_state_indices=gdn_state_indices_device,
            state_slots=state_slots_device,
            is_speculative=True,
        )
        return ScheduledBatch(
            execution=execution,
            request_ids=tuple(request_ids),
            scheduled_tokens=(chain_length + 1,) * len(rows),
            should_sample=(True,) * len(rows),
            sampling=tuple(samplings),
            histories=tuple(histories_rows),
            history_counts=tuple(history_counts_rows),
            draft_token_ids=tuple(drafts),
        )

    def prepare_mtp_speculative(
        self,
        scheduled: ScheduledBatch,
        *,
        draft_token_id: int | tuple[int, ...],
    ) -> ScheduledBatch:
        """Reserve a native-MTP draft chain plus its target bonus position."""

        if (
            scheduled.is_speculative
            or not scheduled.execution.all_decode
            or scheduled.execution.num_sequences != 1
        ):
            return scheduled
        draft_tokens = (
            (int(draft_token_id),)
            if isinstance(draft_token_id, int)
            else tuple(int(token) for token in draft_token_id)
        )
        if not draft_tokens or any(token < 0 for token in draft_tokens):
            raise SchedulerError("MTP draft token ids are invalid")
        request_id = scheduled.request_ids[0]
        sequence = self._sequences[request_id]
        sampling = sequence.sampling
        remaining = (
            sequence.request.max_new_tokens - sequence.generated_tokens
        )
        count = len(draft_tokens)
        if (
            sequence.status is not SequenceStatus.RUNNING
            or sampling.temperature != 0
            or sampling.repetition_penalty != 1.0
            or remaining < count + 1
            or sequence.cached_tokens + count + 1 > self.cache_manager.max_model_len
            or self.max_num_batched_tokens < count + 1
        ):
            return scheduled

        start = sequence.cached_tokens
        end = start + count + 1
        try:
            self.cache_manager.ensure_tokens(request_id, end)
        except CacheCapacityError:
            return scheduled
        sequence.page_table = self.cache_manager.block_table(request_id)
        query_tokens = (sequence.token_ids[start],) + draft_tokens
        if len(query_tokens) != count + 1:
            return scheduled
        positions = tuple(range(start, end))
        slots = self.cache_manager.slot_mapping_reserved(
            request_id,
            start,
            count + 1,
        )
        hybrid = self.cache_manager.uses_gdn_state
        shadow_state = (
            self.cache_manager.gdn_state_indices(request_id, shadow=True)
            if hybrid
            else None
        )
        device = self.cache_manager.device
        staging = self._cuda_staging
        if staging is None:
            input_ids_device = tensorplay.tensor(
                query_tokens, dtype=tensorplay.int64, device=device
            )
            positions_device = tensorplay.tensor(
                positions, dtype=tensorplay.int64, device=device
            )
            slots_device = tensorplay.tensor(
                slots, dtype=tensorplay.int64, device=device
            )
            block_tables_device = self.cache_manager.build_block_tables(
                (request_id,)
            )
            sequence_lengths_device = tensorplay.tensor(
                (end,), dtype=tensorplay.int32, device=device
            )
            token_active_device = tensorplay.ones(
                count + 1, dtype=tensorplay.bool, device=device
            )
            sample_indices_device = tensorplay.arange(
                count + 1, dtype=tensorplay.int64, device=device
            )
            query_starts_device = tensorplay.tensor(
                (0, count + 1), dtype=tensorplay.int32, device=device
            )
            gdn_state_indices_device = (
                tensorplay.tensor((shadow_state,), dtype=tensorplay.int32, device=device)
                if hybrid
                else None
            )
            state_slots_device = tensorplay.tensor(
                (self.cache_manager.request_slot(request_id),),
                dtype=tensorplay.int32,
                device=device,
            )
        else:
            (
                input_ids_device,
                positions_device,
                slots_device,
                block_tables_device,
                sequence_lengths_device,
                token_active_device,
                _,
                query_starts_device,
                gdn_state_indices_device,
                state_slots_device,
            ) = staging.stage(
                input_ids=list(query_tokens),
                positions=list(positions),
                slots=list(slots),
                sequence_lengths=[end],
                sample_indices=[count],
                query_starts=[0, count + 1],
                page_tables=(sequence.page_table,),
                all_decode=False,
                gdn_state_indices=([shadow_state] if hybrid else None),
                state_slots=[self.cache_manager.request_slot(request_id)],
            )
            sample_indices_device = (
                staging.speculative_sample_indices[: count + 1]
            )
        execution = ExecutionBatch(
            input_ids=input_ids_device,
            positions=positions_device,
            slot_mapping=slots_device,
            block_tables=block_tables_device,
            sequence_lengths=sequence_lengths_device,
            token_active=token_active_device,
            query_start_loc=(0, count + 1),
            query_lengths=(count + 1,),
            context_lengths=(start,),
            phases=(BatchPhase.PREFILL,),
            sample_indices=sample_indices_device,
            page_size=self.cache_manager.page_size,
            query_start_loc_device=query_starts_device,
            gdn_state_indices=gdn_state_indices_device,
            state_slots=state_slots_device,
            is_speculative=True,
        )
        return ScheduledBatch(
            execution=execution,
            request_ids=(request_id,),
            scheduled_tokens=(count + 1,),
            should_sample=(True,),
            sampling=(sampling,),
            histories=((),),
            history_counts=((),),
            draft_token_ids=(draft_tokens,),
        )

    def complete_speculative_host(
        self,
        scheduled: ScheduledBatch,
        verified_tokens: tuple[tuple[int, ...], ...],
        *,
        accepted_drafts: tuple[int, ...],
    ) -> Completion:
        """Commit verified prompt-lookup chains and roll back their KV tails."""

        if (
            not scheduled.is_speculative
            or len(scheduled.request_ids) != len(scheduled.draft_token_ids)
            or len(verified_tokens) != len(scheduled.request_ids)
            or len(accepted_drafts) != len(scheduled.request_ids)
        ):
            raise SchedulerError("speculative completion batch is invalid")
        finished: list[str] = []
        output_request_ids: list[str] = []
        output_token_ids: list[int] = []
        finish_reasons: list[FinishReason | None] = []
        for row, request_id in enumerate(scheduled.request_ids):
            draft = scheduled.draft_token_ids[row]
            row_accepted = accepted_drafts[row]
            row_tokens = verified_tokens[row]
            if (
                row_accepted < 0
                or row_accepted > len(draft)
                or len(row_tokens) != row_accepted + 1
                or tuple(row_tokens[:row_accepted]) != draft[:row_accepted]
            ):
                raise SchedulerError(
                    "speculative verification output is invalid"
                )
            sequence = self._sequences[request_id]
            if row_accepted < len(draft):
                if len(sequence.rejected_speculative_drafts) >= 16:
                    sequence.rejected_speculative_drafts.clear()
                sequence.rejected_speculative_drafts.add(draft)
            sequence.cached_tokens += row_accepted + 1
            emitted: list[int] = []
            finish_reason: FinishReason | None = None
            for raw_token in row_tokens:
                token_id = int(raw_token)
                if token_id < 0:
                    raise SchedulerError(
                        "speculative verifier omitted a required token"
                    )
                sequence.token_ids.append(token_id)
                sequence.record_generated_token(token_id)
                sequence.generated_tokens += 1
                emitted.append(token_id)
                reached_eos = (
                    (
                        token_id in self.eos_token_ids
                        and not sequence.sampling.ignore_eos
                    )
                    or token_id in sequence.sampling.stop_token_ids
                )
                reached_limit = (
                    sequence.generated_tokens
                    >= sequence.request.max_new_tokens
                )
                if reached_eos or reached_limit:
                    finish_reason = (
                        FinishReason.STOP
                        if reached_eos
                        else FinishReason.LENGTH
                    )
                finish_reasons.append(finish_reason)
                if finish_reason is not None:
                    break
            if self.emit_events:
                self._events.append(
                    sequence.lifecycle.emit_tokens(tuple(emitted))
                )
            else:
                for token_id in emitted:
                    sequence.lifecycle.emit_token_silent(token_id)
            if finish_reason is not None:
                if self.emit_events:
                    self._events.append(
                        sequence.lifecycle.transition(RequestState.FINISHING)
                    )
                    self._events.append(
                        sequence.lifecycle.transition(
                            RequestState.FINISHED,
                            finish_reason=finish_reason,
                        )
                    )
                else:
                    sequence.lifecycle.transition_silent(
                        RequestState.FINISHING
                    )
                    sequence.lifecycle.transition_silent(
                        RequestState.FINISHED,
                        finish_reason=finish_reason,
                    )
                sequence.status = SequenceStatus.FINISHED
                sequence.finish_reason = finish_reason
                self._deadline_request_ids.discard(request_id)
                self._running.pop(request_id, None)
                self.cache_manager.release(request_id)
                finished.append(request_id)
            else:
                if len(sequence.token_ids) != sequence.cached_tokens + 1:
                    raise SchedulerError(
                        "speculative commit violated the decode invariant"
                    )
                self.cache_manager.trim_tokens(
                    request_id,
                    sequence.cached_tokens,
                )
                sequence.page_table = self.cache_manager.block_table(
                    request_id
                )
                if (
                    sequence.cached_tokens
                    // self.cache_manager.page_size
                    > sequence.published_tokens
                    // self.cache_manager.page_size
                ):
                    prefix_input = sequence.token_ids
                    if sequence.radix_key is not None:
                        sequence.radix_key.sync(sequence.token_ids)
                        prefix_input = sequence.radix_key.snapshot(
                            token_count=sequence.cached_tokens,
                        )
                    self.cache_manager.publish_prefix(
                        request_id,
                        prefix_input,
                        cached_tokens=sequence.cached_tokens,
                    )
                    sequence.published_tokens = (
                        sequence.cached_tokens
                        // self.cache_manager.page_size
                    ) * self.cache_manager.page_size
            output_request_ids.extend([request_id] * len(emitted))
            output_token_ids.extend(emitted)
        return Completion(
            events=(),
            finished_request_ids=tuple(finished),
            output_request_ids=tuple(output_request_ids),
            output_token_ids=tuple(output_token_ids),
            finish_reasons=tuple(finish_reasons),
            output_logprobs=(None,) * len(output_token_ids),
        )

    def _build_batch(
        self,
        entries: list[tuple[Sequence, int, BatchPhase, bool]],
    ) -> ScheduledBatch:
        all_decode = all(
            phase is BatchPhase.DECODE for _, _, phase, _ in entries
        )
        if all_decode and self._cuda_staging is not None:
            return self._build_decode_batch(entries)
        request_ids = tuple(entry[0].request_id for entry in entries)
        packed_tokens: list[int] = []
        positions: list[int] = []
        slots: list[int] = []
        query_starts = [0]
        query_lengths: list[int] = []
        context_lengths: list[int] = []
        phases: list[BatchPhase] = []
        sequence_lengths: list[int] = []
        sample_indices: list[int] = []
        sampling: list[SamplingParams | None] = []
        histories: list[tuple[int, ...]] = []
        history_counts: list[tuple[int, ...]] = []
        state_slots: list[int] = []
        lora_slots: list[int] = []
        gdn_state_indices: list[tuple[int, int]] | None = (
            [] if self.cache_manager.uses_gdn_state else None
        )

        for sequence, scheduled, phase, should_sample in entries:
            start = sequence.cached_tokens
            end = start + scheduled
            input_tokens = sequence.token_ids[start:end]
            if len(input_tokens) != scheduled:
                raise SchedulerError("scheduled token slice is incomplete")
            packed_tokens.extend(input_tokens)
            positions.extend(range(start, end))
            slots.extend(
                self.cache_manager.slot_mapping_reserved(
                    sequence.request_id,
                    start,
                    scheduled,
                )
            )
            lora_slots.extend([sequence.lora_slot] * scheduled)
            query_lengths.append(scheduled)
            context_lengths.append(start)
            phases.append(phase)
            sequence_lengths.append(end)
            query_starts.append(query_starts[-1] + scheduled)
            sample_indices.append(
                query_starts[-1] - 1 if should_sample else -1
            )
            sampling.append(sequence.sampling if should_sample else None)
            histories.append(
                sequence.penalty_history()
                if sequence.has_logit_penalty()
                else ()
            )
            history_counts.append(
                sequence.penalty_tokens_and_counts()[1]
                if sequence.has_logit_penalty()
                else ()
            )
            state_slots.append(
                self.cache_manager.request_slot(sequence.request_id)
            )
            if gdn_state_indices is not None:
                gdn_state_indices.append(
                    self.cache_manager.gdn_state_indices(
                        sequence.request_id
                    )
                )

        device = self.cache_manager.device
        staging = self._cuda_staging
        if staging is None:
            input_ids_device = tensorplay.tensor(
                packed_tokens,
                dtype=tensorplay.long,
                device=device,
            )
            positions_device = tensorplay.tensor(
                positions,
                dtype=tensorplay.long,
                device=device,
            )
            slots_device = tensorplay.tensor(
                slots,
                dtype=tensorplay.long,
                device=device,
            )
            block_tables_device = self.cache_manager.build_block_tables(
                request_ids
            )
            sequence_lengths_device = tensorplay.tensor(
                sequence_lengths,
                dtype=tensorplay.int32,
                device=device,
            )
            token_active_device = tensorplay.ones(
                len(packed_tokens),
                dtype=tensorplay.bool,
                device=device,
            )
            sample_indices_device = tensorplay.tensor(
                sample_indices,
                dtype=tensorplay.long,
                device=device,
            )
            query_starts_device = None
            gdn_state_indices_device = (
                None
                if gdn_state_indices is None
                else tensorplay.tensor(
                    gdn_state_indices,
                    dtype=tensorplay.int32,
                    device=device,
                )
            )
            state_slots_device = tensorplay.tensor(
                state_slots,
                dtype=tensorplay.int32,
                device=device,
            )
        else:
            (
                input_ids_device,
                positions_device,
                slots_device,
                block_tables_device,
                sequence_lengths_device,
                token_active_device,
                sample_indices_device,
                query_starts_device,
                gdn_state_indices_device,
                state_slots_device,
            ) = staging.stage(
                input_ids=packed_tokens,
                positions=positions,
                slots=slots,
                sequence_lengths=sequence_lengths,
                sample_indices=sample_indices,
                query_starts=query_starts,
                page_tables=tuple(entry[0].page_table for entry in entries),
                all_decode=all_decode,
                gdn_state_indices=gdn_state_indices,
                state_slots=state_slots,
            )
        # A batch with only base requests must not launch the LoRA kernels at
        # all; the model hooks treat a None slot tensor as a cheap no-op.
        lora_slots_device = (
            tensorplay.tensor(
                lora_slots,
                dtype=tensorplay.int32,
                device=device,
            )
            if any(slot >= 0 for slot in lora_slots)
            else None
        )
        execution = ExecutionBatch(
            input_ids=input_ids_device,
            positions=positions_device,
            slot_mapping=slots_device,
            block_tables=block_tables_device,
            sequence_lengths=sequence_lengths_device,
            token_active=token_active_device,
            query_start_loc=tuple(query_starts),
            query_lengths=tuple(query_lengths),
            context_lengths=tuple(context_lengths),
            phases=tuple(phases),
            sample_indices=sample_indices_device,
            page_size=self.cache_manager.page_size,
            query_start_loc_device=query_starts_device,
            gdn_state_indices=gdn_state_indices_device,
            state_slots=state_slots_device,
            lora_slots=lora_slots_device,
            validate_contract=False,
            host_state_slots=tuple(state_slots),
        )
        return ScheduledBatch(
            execution=execution,
            request_ids=request_ids,
            scheduled_tokens=tuple(query_lengths),
            should_sample=tuple(
                should_sample for _, _, _, should_sample in entries
            ),
            sampling=tuple(sampling),
            histories=tuple(histories),
            history_counts=tuple(history_counts),
        )

    def _build_decode_batch(
        self,
        entries: list[tuple[Sequence, int, BatchPhase, bool]],
    ) -> ScheduledBatch:
        """Build the scheduler-owned one-token path without generic metadata work."""

        num_sequences = len(entries)
        query_starts = tuple(range(num_sequences + 1))
        query_lengths = (1,) * num_sequences
        phases = (BatchPhase.DECODE,) * num_sequences
        request_ids: list[str] = []
        packed_tokens: list[int] = []
        positions: list[int] = []
        slots: list[int] = []
        sequence_lengths: list[int] = []
        page_tables: list[tuple[int, ...]] = []
        sampling: list[SamplingParams] = []
        histories: list[tuple[int, ...]] = []
        history_counts: list[tuple[int, ...]] = []
        state_slots: list[int] = []
        lora_slots: list[int] = []
        gdn_state_indices: list[tuple[int, int]] | None = (
            [] if self.cache_manager.uses_gdn_state else None
        )
        page_size = self.cache_manager.page_size

        for sequence, scheduled, phase, should_sample in entries:
            if scheduled != 1 or phase is not BatchPhase.DECODE or not should_sample:
                raise SchedulerError("invalid entry in trusted decode batch")
            position = sequence.cached_tokens
            pages = sequence.page_table
            request_ids.append(sequence.request_id)
            packed_tokens.append(sequence.token_ids[position])
            positions.append(position)
            slots.append(
                pages[position // page_size] * page_size
                + position % page_size
            )
            sequence_lengths.append(position + 1)
            page_tables.append(pages)
            sampling.append(sequence.sampling)
            state_slots.append(
                self.cache_manager.request_slot(sequence.request_id)
            )
            lora_slots.append(sequence.lora_slot)
            histories.append(
                sequence.penalty_history()
                if sequence.has_logit_penalty()
                else ()
            )
            history_counts.append(
                sequence.penalty_tokens_and_counts()[1]
                if sequence.has_logit_penalty()
                else ()
            )
            if gdn_state_indices is not None:
                gdn_state_indices.append(
                    self.cache_manager.gdn_state_indices(
                        sequence.request_id
                    )
                )

        staging = self._cuda_staging
        if staging is None:
            raise SchedulerError("trusted decode batching requires CUDA staging")
        (
            input_ids_device,
            positions_device,
            slots_device,
            block_tables_device,
            sequence_lengths_device,
            token_active_device,
            sample_indices_device,
            query_starts_device,
            gdn_state_indices_device,
            state_slots_device,
        ) = staging.stage(
            input_ids=packed_tokens,
            positions=positions,
            slots=slots,
            sequence_lengths=sequence_lengths,
            sample_indices=[],
            query_starts=[],
            page_tables=tuple(page_tables),
            all_decode=True,
            gdn_state_indices=gdn_state_indices,
            state_slots=state_slots,
        )
        lora_slots_device = (
            tensorplay.tensor(
                lora_slots,
                dtype=tensorplay.int32,
                device=self.cache_manager.device,
            )
            if any(slot >= 0 for slot in lora_slots)
            else None
        )
        execution = ExecutionBatch(
            input_ids=input_ids_device,
            positions=positions_device,
            slot_mapping=slots_device,
            block_tables=block_tables_device,
            sequence_lengths=sequence_lengths_device,
            token_active=token_active_device,
            query_start_loc=query_starts,
            query_lengths=query_lengths,
            context_lengths=tuple(positions),
            phases=phases,
            sample_indices=sample_indices_device,
            page_size=page_size,
            query_start_loc_device=query_starts_device,
            gdn_state_indices=gdn_state_indices_device,
            state_slots=state_slots_device,
            lora_slots=lora_slots_device,
            validate_contract=False,
        )
        return ScheduledBatch(
            execution=execution,
            request_ids=tuple(request_ids),
            scheduled_tokens=query_lengths,
            should_sample=(True,) * num_sequences,
            sampling=tuple(sampling),
            histories=tuple(histories),
            history_counts=tuple(history_counts),
        )

    def _preempt(self, sequence: Sequence) -> None:
        self.preemptions += 1
        self._running.pop(sequence.request_id, None)
        self.cache_manager.release(sequence.request_id)
        sequence.cached_tokens = 0
        sequence.published_tokens = 0
        sequence.page_table = ()
        sequence.status = SequenceStatus.WAITING
        self._waiting.appendleft(sequence.request_id)
        # A sequence may already be PREEMPTED (e.g. reclaimed by partial-prefill
        # recovery and then evicted again here).  The lifecycle forbids a
        # PREEMPTED -> PREEMPTED transition, so only emit it on a real change.
        if sequence.lifecycle.state is not RequestState.PREEMPTED:
            if self.emit_events:
                self._events.append(
                    sequence.lifecycle.transition(RequestState.PREEMPTED)
                )
            else:
                sequence.lifecycle.transition_silent(RequestState.PREEMPTED)

    def _pop_next_waiting(self, excluded: set[str]) -> str | None:
        recovery_owner = self._prefill_recovery_owner
        if (
            recovery_owner is not None
            and recovery_owner not in excluded
            and recovery_owner in self._waiting
        ):
            self._waiting.remove(recovery_owner)
            self._prefill_recovery_owner = None
            return recovery_owner
        best_index: int | None = None
        best_score: int | None = None
        for index, request_id in enumerate(self._waiting):
            if request_id in excluded:
                continue
            sequence = self._sequences[request_id]
            age = max(0, self._tick - sequence.enqueue_tick)
            score = sequence.request.priority + age // self._AGING_INTERVAL
            if best_score is None or score > best_score:
                best_index = index
                best_score = score
        if best_index is None:
            return None
        request_id = self._waiting[best_index]
        del self._waiting[best_index]
        return request_id

    def _recover_partial_prefill_capacity(self) -> bool:
        """Reclaim split partial-prefill pages when no request can advance."""

        partials = tuple(
            self._sequences[request_id]
            for request_id in dict.fromkeys(self._waiting)
            if (
                (sequence := self._sequences[request_id]).needs_prefill
                and sequence.cached_tokens > 0
                and sequence.lifecycle.state
                in {RequestState.PREFILL, RequestState.PREEMPTED}
                and self.cache_manager.handle(request_id) is not None
            )
        )
        if len(partials) < 2:
            return False

        def owner_score(sequence: Sequence) -> tuple[int, int]:
            age = max(0, self._tick - sequence.enqueue_tick)
            return (
                sequence.request.priority
                + age // self._AGING_INTERVAL,
                sequence.cached_tokens,
            )

        owner = max(partials, key=owner_score)
        reclaimed = 0
        for sequence in partials:
            if sequence is owner:
                continue
            reclaimed += sequence.cached_tokens
            self.cache_manager.release(sequence.request_id)
            sequence.cached_tokens = 0
            sequence.published_tokens = 0
            sequence.page_table = ()
            sequence.status = SequenceStatus.WAITING
            if sequence.lifecycle.state is not RequestState.PREEMPTED:
                if self.emit_events:
                    self._events.append(
                        sequence.lifecycle.transition(RequestState.PREEMPTED)
                    )
                else:
                    sequence.lifecycle.transition_silent(
                        RequestState.PREEMPTED
                    )
        if reclaimed <= 0:
            return False
        self._prefill_recovery_owner = owner.request_id
        self.prefill_deadlock_recoveries += 1
        self.prefill_recomputed_tokens += reclaimed
        return True

    def _expire_deadlines(self) -> None:
        if not self._deadline_request_ids:
            return
        now_ns = time.monotonic_ns()
        expired = tuple(
            request_id
            for request_id in self._deadline_request_ids
            if (sequence := self._sequences.get(request_id)) is not None
            if sequence.status is not SequenceStatus.FINISHED
            and sequence.request.deadline_ns <= now_ns  # type: ignore[operator]
        )
        for request_id in expired:
            self.deadline_expirations += 1
            self.cancel(request_id)
