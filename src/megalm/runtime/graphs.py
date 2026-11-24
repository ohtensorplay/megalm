from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Callable

import tensorplay

from megalm.kernels.decode_relay import advance_decode_state, relay_decode_batch
from megalm.kernels.speculative_relay import relay_speculative_chain
from megalm.runtime.batch import BatchPhase, ExecutionBatch
from megalm.runtime.cache import PagedKVCache
from megalm.runtime.hybrid_cache import HybridCache


@contextmanager
def _capture_mode(model: Any):
    """Signal the model that a CUDA graph capture is in progress.

    Models whose hot path needs host synchronization (e.g. data-dependent
    MoE dispatch) switch to a capture-safe equivalent for the duration.
    """

    set_active = getattr(model, "set_capture_active", None)
    if set_active is not None:
        set_active(True)
    try:
        yield
    finally:
        if set_active is not None:
            set_active(False)


class CUDAGraphError(RuntimeError):
    """Raised when a requested production graph bucket cannot be captured."""


@dataclass(slots=True)
class GraphReplayMetrics:
    captures: int = 0
    greedy_captures: int = 0
    speculative_captures: int = 0
    replays: int = 0
    model_replays: int = 0
    greedy_replays: int = 0
    speculative_replays: int = 0
    greedy_loop_steps: int = 0
    eager_fallbacks: int = 0
    # Capture health (USE "errors" signal).
    capture_failures: int = 0
    capture_capable: int = 0


@dataclass(slots=True)
class _GraphBucket:
    size: int
    batch: ExecutionBatch
    model_graph: tensorplay.cuda.CUDAGraph
    logits: tensorplay.Tensor
    model_input_embeddings: tensorplay.Tensor | None
    model_hidden_states: tensorplay.Tensor | None
    greedy_graph: tensorplay.cuda.CUDAGraph
    sampled: tensorplay.Tensor
    greedy_input_embeddings: tensorplay.Tensor | None
    greedy_hidden_states: tensorplay.Tensor | None
    # Optional device-side self-loop: model -> argmax -> advance static inputs.
    loop_graph: tensorplay.cuda.CUDAGraph | None = None
    loop_record: tensorplay.Tensor | None = None
    loop_counter: tensorplay.Tensor | None = None


@dataclass(slots=True)
class _SpeculativeGraphBucket:
    length: int
    input_ids: tensorplay.Tensor
    positions: tensorplay.Tensor
    slot_mapping: tensorplay.Tensor
    block_tables: tensorplay.Tensor
    sequence_lengths: tensorplay.Tensor
    gdn_state_indices: tensorplay.Tensor | None
    mrope_positions: tensorplay.Tensor | None
    batches: tuple[ExecutionBatch, ...]
    graph: tensorplay.cuda.CUDAGraph
    sampled: tensorplay.Tensor
    target_hidden_states: tensorplay.Tensor | None


@dataclass(slots=True)
class _MTPUpdateGraphBucket:
    length: int
    batch: ExecutionBatch
    target_hidden_states: tensorplay.Tensor
    graph: tensorplay.cuda.CUDAGraph
    sampled: tensorplay.Tensor


class MTPUpdateGraphRunner:
    """Captured one- and two-token native MTP cache updates."""

    def __init__(
        self,
        predictor: Any,
        lm_head_weight: tensorplay.Tensor,
        cache: PagedKVCache,
        *,
        max_blocks_per_sequence: int,
        page_size: int,
        hidden_size: int,
        dtype: tensorplay.dtype,
        stream_priority: int = 0,
        strict: bool = True,
    ) -> None:
        self.predictor = predictor
        self.lm_head_weight = lm_head_weight
        self.cache = cache
        self.max_blocks_per_sequence = max_blocks_per_sequence
        self.page_size = page_size
        self.device = cache.device
        self.hidden_size = hidden_size
        self.dtype = dtype
        self.stream_priority = int(stream_priority)
        self.strict = strict
        self.replays = 0
        allocated_before = tensorplay.cuda.memory_allocated(self.device)
        self._buckets: dict[int, _MTPUpdateGraphBucket] = {}
        self._capture((1, 2))
        self.memory_bytes = max(
            0,
            tensorplay.cuda.memory_allocated(self.device) - allocated_before,
        )

    def run(
        self,
        batch: ExecutionBatch,
        target_hidden_states: tensorplay.Tensor,
    ) -> tensorplay.Tensor:
        try:
            bucket = self._buckets[batch.num_tokens]
        except KeyError as exc:
            raise CUDAGraphError(
                "no MTP update graph for the requested token count"
            ) from exc
        if batch.num_tokens == 1:
            relay_decode_batch(
                batch.input_ids,
                batch.positions,
                batch.slot_mapping,
                batch.block_tables,
                batch.sequence_lengths,
                batch.token_active,
                bucket.batch.input_ids,
                bucket.batch.positions,
                bucket.batch.slot_mapping,
                bucket.batch.block_tables,
                bucket.batch.sequence_lengths,
                bucket.batch.token_active,
                None,
                None,
                None,
                None,
            )
        else:
            bucket.batch.input_ids.copy_(batch.input_ids)
            bucket.batch.positions.copy_(batch.positions)
            bucket.batch.slot_mapping.copy_(batch.slot_mapping)
            bucket.batch.block_tables.copy_(batch.block_tables)
            bucket.batch.sequence_lengths.copy_(batch.sequence_lengths)
        bucket.target_hidden_states.copy_(target_hidden_states)
        bucket.graph.replay()
        self.replays += 1
        return bucket.sampled

    def _capture(self, lengths: tuple[int, ...]) -> None:
        capture_stream = tensorplay.cuda.Stream(
            device=self.device,
            priority=self.stream_priority,
        )
        current_stream = tensorplay.cuda.current_stream(self.device)
        capture_stream.wait_stream(current_stream)
        shared_pool = tensorplay.cuda.graph_pool_handle()
        for length in lengths:
            positions = tensorplay.arange(
                length,
                dtype=tensorplay.int64,
                device=self.device,
            )
            query_starts = tensorplay.tensor(
                (0, length),
                dtype=tensorplay.int32,
                device=self.device,
            )
            phase = (
                BatchPhase.DECODE
                if length == 1
                else BatchPhase.PREFILL
            )
            batch = ExecutionBatch(
                input_ids=tensorplay.zeros(
                    length,
                    dtype=tensorplay.int64,
                    device=self.device,
                ),
                positions=positions,
                slot_mapping=positions.clone(),
                block_tables=tensorplay.zeros(
                    (1, self.max_blocks_per_sequence),
                    dtype=tensorplay.int32,
                    device=self.device,
                ),
                sequence_lengths=tensorplay.tensor(
                    (length,),
                    dtype=tensorplay.int32,
                    device=self.device,
                ),
                token_active=tensorplay.ones(
                    length,
                    dtype=tensorplay.bool,
                    device=self.device,
                ),
                query_start_loc=(0, length),
                query_lengths=(length,),
                context_lengths=(0,),
                phases=(phase,),
                sample_indices=tensorplay.tensor(
                    (length - 1,),
                    dtype=tensorplay.int64,
                    device=self.device,
                ),
                page_size=self.page_size,
                query_start_loc_device=query_starts,
                validate_contract=False,
            )
            target_hidden_states = tensorplay.zeros(
                (length, self.hidden_size),
                dtype=self.dtype,
                device=self.device,
            )

            def run_update() -> tensorplay.Tensor:
                hidden_states = self.predictor(
                    batch,
                    target_hidden_states,
                    self.cache,
                )
                logits = self.predictor.compute_logits(
                    hidden_states,
                    self.lm_head_weight,
                )
                return tensorplay.argmax(logits, dim=-1)

            try:
                with tensorplay.cuda.stream(capture_stream), tensorplay.inference_mode():
                    for _ in range(2):
                        sampled = run_update()
                capture_stream.synchronize()
                current_stream.wait_stream(capture_stream)
                graph = tensorplay.cuda.CUDAGraph()
                with _capture_mode(self.model), tensorplay.cuda.graph(
                    graph,
                    pool=shared_pool,
                    stream=capture_stream,
                ), tensorplay.inference_mode():
                    sampled = run_update()
                with tensorplay.cuda.stream(capture_stream):
                    graph.replay()
                capture_stream.synchronize()
            except Exception as exc:
                if self.strict:
                    raise CUDAGraphError(
                        f"failed to capture MTP update length {length}"
                    ) from exc
                continue
            self._buckets[length] = _MTPUpdateGraphBucket(
                length=length,
                batch=batch,
                target_hidden_states=target_hidden_states,
                graph=graph,
                sampled=sampled,
            )
        if len(self._buckets) != len(lengths):
            raise CUDAGraphError("not every MTP update graph was captured")


class DecodeGraphRunner:
    """Static-address CUDA Graph capture/replay for decode batch buckets."""

    def __init__(
        self,
        model: Any,
        cache: PagedKVCache | HybridCache,
        *,
        batch_sizes: tuple[int, ...],
        max_blocks_per_sequence: int,
        page_size: int,
        greedy_argmax: Callable[[tensorplay.Tensor], tensorplay.Tensor],
        speculative_lengths: tuple[int, ...] = (),
        stream_priority: int = 0,
        strict: bool = True,
        loop_steps: int = 1,
    ) -> None:
        if not model.device.type == "cuda":
            raise CUDAGraphError("CUDA Graphs require a CUDA model")
        normalized = tuple(sorted(set(batch_sizes)))
        if not normalized or any(size <= 0 for size in normalized):
            raise CUDAGraphError("graph batch buckets must be positive")
        self.model = model
        self.cache = cache
        self.max_blocks_per_sequence = max_blocks_per_sequence
        self.page_size = page_size
        self.greedy_argmax = greedy_argmax
        self.stream_priority = int(stream_priority)
        self.strict = strict
        # Chunked decode: how many replays a device-side self-loop may chain
        # before the host has to commit.  1 disables the loop entirely.
        self.loop_steps = max(1, int(loop_steps))
        self.hybrid_state = isinstance(cache, HybridCache)
        self.uses_mrope = bool(getattr(model, "uses_mrope", False))
        if self.hybrid_state and not self.uses_mrope:
            raise CUDAGraphError(
                "hybrid-state models must declare MRoPE graph metadata"
            )
        self.metrics = GraphReplayMetrics()
        allocated_before = tensorplay.cuda.memory_allocated(model.device)
        self._buckets: dict[int, _GraphBucket] = {}
        self._speculative_buckets: dict[
            int,
            _SpeculativeGraphBucket,
        ] = {}
        self._last_features: tuple[tensorplay.Tensor, tensorplay.Tensor] | None = None
        shared_pool = tensorplay.cuda.graph_pool_handle()
        # Models whose hot path needs data-dependent host dispatch (e.g.
        # grouped int4 MoE without a fused kernel) opt out of capture; decode
        # then runs eager instead of failing capture after loading weights.
        self.capture_capable = bool(getattr(model, "graph_capable", True))
        self.metrics.capture_capable = int(self.capture_capable)
        if self.capture_capable:
            self._capture(normalized, shared_pool)
            normalized_speculative = tuple(
                sorted(set(speculative_lengths))
            )
            if any(length < 2 or length > 32 for length in normalized_speculative):
                raise CUDAGraphError(
                    "speculative graph lengths must be in [2, 32]"
                )
            if normalized_speculative and getattr(
                getattr(self.model, "config", None), "model_type", None
            ) not in {"inkling2", "inkling2_loop"}:
                self._capture_speculative(
                    normalized_speculative,
                    shared_pool,
                )
            self._warm_relay()
        self.memory_bytes = max(
            0,
            tensorplay.cuda.memory_allocated(model.device) - allocated_before,
        )

    @property
    def batch_sizes(self) -> tuple[int, ...]:
        return tuple(self._buckets)

    def can_replay(self, batch: ExecutionBatch) -> bool:
        return (
            batch.all_decode
            and any(size >= batch.num_sequences for size in self._buckets)
        )

    def bucket_size_for(self, batch: ExecutionBatch) -> int | None:
        if not batch.all_decode:
            return None
        return next(
            (size for size in self._buckets if size >= batch.num_sequences),
            None,
        )

    def run(self, batch: ExecutionBatch) -> tensorplay.Tensor:
        bucket, count = self._relay(batch)
        bucket.model_graph.replay()
        self._last_features = _graph_features(
            bucket.model_input_embeddings,
            bucket.model_hidden_states,
            count,
        )
        self.metrics.replays += 1
        self.metrics.model_replays += 1
        return bucket.logits[:count]

    def run_greedy(self, batch: ExecutionBatch) -> tensorplay.Tensor:
        """Replay one model-plus-greedy graph without a second host launch."""

        bucket, count = self._relay(batch)
        bucket.greedy_graph.replay()
        self._last_features = _graph_features(
            bucket.greedy_input_embeddings,
            bucket.greedy_hidden_states,
            count,
        )
        self.metrics.replays += 1
        self.metrics.greedy_replays += 1
        return bucket.sampled[:count]

    def loop_capacity_for(self, batch: ExecutionBatch) -> int:
        """Chunk length this bucket can chain without a host round trip."""

        if self.loop_steps <= 1 or self.hybrid_state:
            return 1
        bucket_size = self.bucket_size_for(batch)
        if bucket_size is None:
            return 1
        bucket = self._buckets[bucket_size]
        if bucket.loop_graph is None or bucket.loop_record is None:
            return 1
        return min(self.loop_steps, int(bucket.loop_record.shape[0]))

    def run_greedy_loop(
        self,
        batch: ExecutionBatch,
        steps: int,
    ) -> tensorplay.Tensor:
        """Replay ``steps`` chained decode iterations and return their tokens.

        The graph folds each argmax back into its own static inputs, so the
        whole chunk is one host launch.  The returned tensor is
        ``[steps, live_rows]`` in replay order.
        """

        bucket, count = self._relay(batch)
        if (
            bucket.loop_graph is None
            or bucket.loop_record is None
            or bucket.loop_counter is None
        ):
            raise CUDAGraphError("decode bucket has no self-loop graph")
        capacity = int(bucket.loop_record.shape[0])
        if steps < 1 or steps > capacity:
            raise CUDAGraphError(
                f"decode self-loop chunk {steps} exceeds capacity {capacity}"
            )
        bucket.loop_counter.zero_()
        for _ in range(steps):
            bucket.loop_graph.replay()
        self.metrics.replays += steps
        self.metrics.greedy_replays += steps
        self.metrics.greedy_loop_steps += steps
        return bucket.loop_record[:steps, :count]

    def last_features(self) -> tuple[tensorplay.Tensor, tensorplay.Tensor]:
        if self._last_features is None:
            raise CUDAGraphError("decode graph has not produced model features")
        return self._last_features

    @property
    def speculative_lengths(self) -> tuple[int, ...]:
        return tuple(self._speculative_buckets)

    def can_replay_speculative(self, batch: ExecutionBatch) -> bool:
        return (
            batch.is_speculative
            and batch.num_tokens in self._speculative_buckets
        )

    def run_speculative_greedy(
        self,
        batch: ExecutionBatch,
    ) -> tensorplay.Tensor:
        """Replay one captured chain of exact B1 decode forwards."""

        if not self.can_replay_speculative(batch):
            raise CUDAGraphError(
                "no exact speculative graph for the requested length"
            )
        bucket = self._speculative_buckets[batch.num_tokens]
        relay_speculative_chain(
            batch.input_ids,
            batch.positions,
            batch.slot_mapping,
            batch.block_tables,
            bucket.input_ids,
            bucket.positions,
            bucket.slot_mapping,
            bucket.block_tables,
            bucket.sequence_lengths,
            source_mrope=batch.mrope_positions,
            target_mrope=bucket.mrope_positions,
        )
        if self.hybrid_state:
            if (
                batch.gdn_state_indices is None
                or bucket.gdn_state_indices is None
            ):
                raise CUDAGraphError(
                    "hybrid speculative graph has no GDN state indices"
                )
            bucket.gdn_state_indices.copy_(batch.gdn_state_indices)
        bucket.graph.replay()
        self.metrics.replays += 1
        self.metrics.greedy_replays += 1
        self.metrics.speculative_replays += 1
        return bucket.sampled

    def speculative_hidden_states(
        self,
        length: int,
    ) -> tensorplay.Tensor | None:
        try:
            return self._speculative_buckets[length].target_hidden_states
        except KeyError as exc:
            raise CUDAGraphError(
                "no speculative graph for the requested hidden states"
            ) from exc

    def _relay(self, batch: ExecutionBatch) -> tuple[_GraphBucket, int]:
        if not batch.all_decode:
            raise CUDAGraphError("only decode batches can use graph replay")
        bucket_size = self.bucket_size_for(batch)
        if bucket_size is None:
            self.metrics.eager_fallbacks += 1
            raise CUDAGraphError(
                f"no graph bucket for batch size {batch.num_sequences}"
            )
        bucket = self._buckets[bucket_size]
        static = bucket.batch
        count = batch.num_sequences
        relay_decode_batch(
            batch.input_ids,
            batch.positions,
            batch.slot_mapping,
            batch.block_tables,
            batch.sequence_lengths,
            batch.token_active,
            static.input_ids,
            static.positions,
            static.slot_mapping,
            static.block_tables,
            static.sequence_lengths,
            static.token_active,
            batch.gdn_state_indices if self.hybrid_state else None,
            batch.mrope_positions if self.uses_mrope else None,
            static.gdn_state_indices,
            static.mrope_positions,
        )
        if batch.state_slots is not None:
            static.state_slots[:count].copy_(batch.state_slots)
            if count < bucket_size:
                static.state_slots[count:].zero_()
        return bucket, count

    def _capture(
        self,
        batch_sizes: tuple[int, ...],
        shared_pool: tuple[int, int],
    ) -> None:
        capture_stream = tensorplay.cuda.Stream(
            device=self.model.device,
            priority=self.stream_priority,
        )
        current_stream = tensorplay.cuda.current_stream(self.model.device)
        capture_stream.wait_stream(current_stream)
        for batch_size in batch_sizes:
            static_batch = self._static_batch(batch_size)
            try:
                with tensorplay.cuda.stream(capture_stream), tensorplay.inference_mode():
                    for _ in range(2):
                        self.model(static_batch, self.cache)
                capture_stream.synchronize()
                current_stream.wait_stream(capture_stream)
                model_graph = tensorplay.cuda.CUDAGraph()
                with _capture_mode(self.model), tensorplay.cuda.graph(
                    model_graph,
                    pool=shared_pool,
                    stream=capture_stream,
                ), tensorplay.inference_mode():
                    logits = self.model(static_batch, self.cache)
                    model_input_embeddings = getattr(
                        self.model,
                        "last_input_embeddings",
                        None,
                    )
                    model_hidden_states = getattr(
                        self.model,
                        "last_hidden_states",
                        None,
                    )
                with tensorplay.cuda.stream(capture_stream), tensorplay.inference_mode():
                    model_graph.replay()
                    for _ in range(2):
                        greedy_logits = self.model(static_batch, self.cache)
                        self._greedy_sample(greedy_logits)
                capture_stream.synchronize()
                greedy_graph = tensorplay.cuda.CUDAGraph()
                with _capture_mode(self.model), tensorplay.cuda.graph(
                    greedy_graph,
                    pool=shared_pool,
                    stream=capture_stream,
                ), tensorplay.inference_mode():
                    greedy_logits = self.model(static_batch, self.cache)
                    sampled = self._greedy_sample(greedy_logits)
                    greedy_input_embeddings = getattr(
                        self.model,
                        "last_input_embeddings",
                        None,
                    )
                    greedy_hidden_states = getattr(
                        self.model,
                        "last_hidden_states",
                        None,
                    )
                with tensorplay.cuda.stream(capture_stream):
                    greedy_graph.replay()
                capture_stream.synchronize()
                loop_graph = None
                loop_record = None
                loop_counter = None
                if self.loop_steps > 1 and not self.hybrid_state:
                    loop_record = tensorplay.zeros(
                        (self.loop_steps, batch_size),
                        dtype=tensorplay.long,
                        device=self.model.device,
                    )
                    loop_counter = tensorplay.zeros(
                        1,
                        dtype=tensorplay.int32,
                        device=self.model.device,
                    )
                    warm_sampled = tensorplay.zeros(
                        batch_size,
                        dtype=tensorplay.long,
                        device=self.model.device,
                    )
                    with tensorplay.cuda.stream(capture_stream):
                        advance_decode_state(
                            input_ids=static_batch.input_ids,
                            positions=static_batch.positions,
                            slot_mapping=static_batch.slot_mapping,
                            block_tables=static_batch.block_tables,
                            sequence_lengths=static_batch.sequence_lengths,
                            token_active=static_batch.token_active,
                            sampled=warm_sampled,
                            record=loop_record,
                            counter=loop_counter,
                            mrope_positions=(
                                static_batch.mrope_positions
                                if self.uses_mrope
                                else None
                            ),
                            page_size=self.page_size,
                        )
                    capture_stream.synchronize()
                    current_stream.wait_stream(capture_stream)
                    loop_graph = tensorplay.cuda.CUDAGraph()
                    with _capture_mode(self.model), tensorplay.cuda.graph(
                        loop_graph,
                        pool=shared_pool,
                        stream=capture_stream,
                    ), tensorplay.inference_mode():
                        loop_logits = self.model(static_batch, self.cache)
                        loop_sampled = self._greedy_sample(loop_logits)
                        advance_decode_state(
                            input_ids=static_batch.input_ids,
                            positions=static_batch.positions,
                            slot_mapping=static_batch.slot_mapping,
                            block_tables=static_batch.block_tables,
                            sequence_lengths=static_batch.sequence_lengths,
                            token_active=static_batch.token_active,
                            sampled=loop_sampled,
                            record=loop_record,
                            counter=loop_counter,
                            mrope_positions=(
                                static_batch.mrope_positions
                                if self.uses_mrope
                                else None
                            ),
                            page_size=self.page_size,
                        )
                    with tensorplay.cuda.stream(capture_stream):
                        loop_graph.replay()
                    capture_stream.synchronize()
                    loop_counter.zero_()
            except Exception as exc:
                self.metrics.capture_failures += 1
                if self.strict:
                    raise CUDAGraphError(
                        f"failed to capture decode bucket {batch_size}"
                    ) from exc
                continue
            self._buckets[batch_size] = _GraphBucket(
                size=batch_size,
                batch=static_batch,
                model_graph=model_graph,
                logits=logits,
                model_input_embeddings=model_input_embeddings,
                model_hidden_states=model_hidden_states,
                greedy_graph=greedy_graph,
                sampled=sampled,
                greedy_input_embeddings=greedy_input_embeddings,
                greedy_hidden_states=greedy_hidden_states,
                loop_graph=loop_graph,
                loop_record=loop_record,
                loop_counter=loop_counter,
            )
            self.metrics.captures += 1
            self.metrics.greedy_captures += 1
        if not self._buckets:
            raise CUDAGraphError("no CUDA Graph decode bucket was captured")
    def _capture_speculative(
        self,
        lengths: tuple[int, ...],
        shared_pool: tuple[int, int],
    ) -> None:
        """Capture each exact-decode chain as one host replay."""

        capture_stream = tensorplay.cuda.Stream(
            device=self.model.device,
            priority=self.stream_priority,
        )
        current_stream = tensorplay.cuda.current_stream(self.model.device)
        capture_stream.wait_stream(current_stream)
        for length in lengths:
            device = self.model.device
            input_ids = tensorplay.zeros(
                length,
                dtype=tensorplay.int64,
                device=device,
            )
            positions = tensorplay.zeros_like(input_ids)
            slot_mapping = tensorplay.zeros_like(input_ids)
            block_tables = tensorplay.zeros(
                (1, self.max_blocks_per_sequence),
                dtype=tensorplay.int32,
                device=device,
            )
            sequence_lengths = tensorplay.ones(
                length,
                dtype=tensorplay.int32,
                device=device,
            )
            token_active = tensorplay.ones(
                length,
                dtype=tensorplay.bool,
                device=device,
            )
            sample_indices = tensorplay.zeros(
                length,
                dtype=tensorplay.int64,
                device=device,
            )
            mrope_positions = (
                tensorplay.zeros(
                    (3, length),
                    dtype=tensorplay.int64,
                    device=device,
                )
                if self.uses_mrope
                else None
            )
            gdn_state_indices = (
                tensorplay.zeros(
                    (1, 2),
                    dtype=tensorplay.int32,
                    device=device,
                )
                if self.hybrid_state
                else None
            )
            batches = tuple(
                ExecutionBatch(
                    input_ids=input_ids[index : index + 1],
                    positions=positions[index : index + 1],
                    slot_mapping=slot_mapping[index : index + 1],
                    block_tables=block_tables,
                    sequence_lengths=sequence_lengths[
                        index : index + 1
                    ],
                    token_active=token_active[index : index + 1],
                    query_start_loc=(0, 1),
                    query_lengths=(1,),
                    context_lengths=(index,),
                    phases=(BatchPhase.DECODE,),
                    sample_indices=sample_indices[index : index + 1],
                    page_size=self.page_size,
                    gdn_state_indices=gdn_state_indices,
                    mrope_positions=(
                        None
                        if mrope_positions is None
                        else mrope_positions[:, index : index + 1]
                    ),
                    validate_contract=False,
                )
                for index in range(length)
            )
            sampled = tensorplay.empty(
                length,
                dtype=tensorplay.int64,
                device=device,
            )
            target_hidden_states = (
                tensorplay.empty(
                    (
                        length,
                        self.model.config.text_config.hidden_size,
                    ),
                    dtype=self.model.dtype,
                    device=device,
                )
                if getattr(self.model, "mtp", None) is not None
                else None
            )

            def run_chain() -> None:
                for index, decode_batch in enumerate(batches):
                    logits = self.model(decode_batch, self.cache)
                    if target_hidden_states is not None:
                        target_hidden_states[index].copy_(
                            self.model.last_hidden_states[0]
                        )
                    token = self._greedy_sample(logits)
                    sampled[index : index + 1].copy_(token)

            try:
                with tensorplay.cuda.stream(capture_stream), tensorplay.inference_mode():
                    for _ in range(2):
                        run_chain()
                capture_stream.synchronize()
                current_stream.wait_stream(capture_stream)
                graph = tensorplay.cuda.CUDAGraph()
                with _capture_mode(self.model), tensorplay.cuda.graph(
                    graph,
                    pool=shared_pool,
                    stream=capture_stream,
                ), tensorplay.inference_mode():
                    run_chain()
                with tensorplay.cuda.stream(capture_stream):
                    graph.replay()
                capture_stream.synchronize()
            except Exception as exc:
                self.metrics.capture_failures += 1
                if self.strict:
                    raise CUDAGraphError(
                        "failed to capture speculative decode chain "
                        f"{length}"
                    ) from exc
                continue
            self._speculative_buckets[length] = _SpeculativeGraphBucket(
                length=length,
                input_ids=input_ids,
                positions=positions,
                slot_mapping=slot_mapping,
                block_tables=block_tables,
                sequence_lengths=sequence_lengths,
                gdn_state_indices=gdn_state_indices,
                mrope_positions=mrope_positions,
                batches=batches,
                graph=graph,
                sampled=sampled,
                target_hidden_states=target_hidden_states,
            )
            self.metrics.captures += 1
            self.metrics.greedy_captures += 1
            self.metrics.speculative_captures += 1
        if len(self._speculative_buckets) != len(lengths):
            raise CUDAGraphError(
                "not every speculative decode chain was captured"
            )

    def _warm_relay(self) -> None:
        """Compile the one metadata relay before the Replica becomes ready."""

        static = next(iter(self._buckets.values())).batch
        relay_decode_batch(
            static.input_ids,
            static.positions,
            static.slot_mapping,
            static.block_tables,
            static.sequence_lengths,
            static.token_active,
            static.input_ids,
            static.positions,
            static.slot_mapping,
            static.block_tables,
            static.sequence_lengths,
            static.token_active,
            static.gdn_state_indices,
            static.mrope_positions,
            static.gdn_state_indices,
            static.mrope_positions,
        )
        for speculative in self._speculative_buckets.values():
            relay_speculative_chain(
                speculative.input_ids,
                speculative.positions,
                speculative.slot_mapping,
                speculative.block_tables,
                speculative.input_ids,
                speculative.positions,
                speculative.slot_mapping,
                speculative.block_tables,
                speculative.sequence_lengths,
                source_mrope=speculative.mrope_positions,
                target_mrope=speculative.mrope_positions,
            )
            if speculative.gdn_state_indices is not None:
                speculative.gdn_state_indices.copy_(
                    speculative.gdn_state_indices
                )

    def _greedy_sample(self, logits: tensorplay.Tensor) -> tensorplay.Tensor:
        if logits.shape[0] == 1:
            return tensorplay.argmax(logits, dim=-1)
        return self.greedy_argmax(logits)

    def _static_batch(self, batch_size: int) -> ExecutionBatch:
        device = self.model.device
        return ExecutionBatch(
            input_ids=tensorplay.zeros(
                batch_size,
                dtype=tensorplay.long,
                device=device,
            ),
            positions=tensorplay.zeros(
                batch_size,
                dtype=tensorplay.long,
                device=device,
            ),
            slot_mapping=tensorplay.zeros(
                batch_size,
                dtype=tensorplay.long,
                device=device,
            ),
            block_tables=tensorplay.zeros(
                (batch_size, self.max_blocks_per_sequence),
                dtype=tensorplay.int32,
                device=device,
            ),
            sequence_lengths=tensorplay.ones(
                batch_size,
                dtype=tensorplay.int32,
                device=device,
            ),
            token_active=tensorplay.zeros(
                batch_size,
                dtype=tensorplay.bool,
                device=device,
            ),
            query_start_loc=tuple(range(batch_size + 1)),
            query_lengths=(1,) * batch_size,
            context_lengths=(1,) * batch_size,
            phases=(BatchPhase.DECODE,) * batch_size,
            sample_indices=tensorplay.arange(
                batch_size,
                dtype=tensorplay.long,
                device=device,
            ),
            page_size=self.page_size,
            gdn_state_indices=(
                tensorplay.zeros(
                    (batch_size, 2),
                    dtype=tensorplay.int32,
                    device=device,
                )
                if self.hybrid_state
                else None
            ),
            state_slots=tensorplay.zeros(
                batch_size,
                dtype=tensorplay.int32,
                device=device,
            ),
            mrope_positions=(
                tensorplay.zeros(
                    (3, batch_size),
                    dtype=tensorplay.long,
                    device=device,
                )
                if self.uses_mrope
                else None
            ),
        )


def _graph_features(
    inputs: tensorplay.Tensor | None,
    hidden: tensorplay.Tensor | None,
    count: int,
) -> tuple[tensorplay.Tensor, tensorplay.Tensor] | None:
    if inputs is None or hidden is None:
        return None
    return inputs[:count], hidden[:count]
