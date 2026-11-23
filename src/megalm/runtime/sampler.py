from __future__ import annotations

import secrets
from collections import OrderedDict
from dataclasses import dataclass

import tensorplay

from megalm.contracts.request import SamplingParams
from megalm.kernels import fallback
from megalm.kernels.bindings import SamplingKernelBindings


class SamplingError(ValueError):
    """Raised when sampling metadata, filters or logits do not align."""


def _mutable_empty(*shape: object, **options: object) -> tensorplay.Tensor:
    """Allocate reusable output storage even when first touched in inference mode."""

    with tensorplay.inference_mode(False):
        return tensorplay.empty(*shape, **options)


@dataclass(slots=True)
class _DeviceSamplingPlan:
    request_ids: tuple[str, ...]
    active_rows: tensorplay.Tensor
    temperatures: tensorplay.Tensor
    penalties: tensorplay.Tensor
    presence: tensorplay.Tensor
    frequency: tensorplay.Tensor
    top_k: tensorplay.Tensor
    top_p: tensorplay.Tensor
    min_p: tensorplay.Tensor
    stochastic_rows: tensorplay.Tensor
    greedy_rows: tensorplay.Tensor
    topk_rows: tensorplay.Tensor
    seeds: tensorplay.Tensor


@dataclass(slots=True)
class _SamplingWorkspace:
    scores: tensorplay.Tensor
    sampled: tensorplay.Tensor
    counters: tensorplay.Tensor
    cutoffs: tensorplay.Tensor
    prefixes: tensorplay.Tensor
    remaining: tensorplay.Tensor
    histogram: tensorplay.Tensor
    valid: tensorplay.Tensor


@dataclass(slots=True)
class _GreedyWorkspace:
    sampled: tensorplay.Tensor
    partial_values: tensorplay.Tensor | None
    partial_indices: tensorplay.Tensor | None


class Sampler:
    """Leader-owned deterministic sampling state with a full Triton fast path."""

    def __init__(
        self,
        *,
        use_triton: bool = True,
        kernel_bindings: SamplingKernelBindings | None = None,
    ) -> None:
        if kernel_bindings is None:
            if use_triton:
                from megalm.kernels.frequency_penalty import (
                    apply_frequency_presence_penalty_,
                )
                from megalm.kernels.greedy_argmax import greedy_argmax
                from megalm.kernels.rejection_sampling import (
                    filtered_rejection_sample,
                )
                from megalm.kernels.repetition_penalty import (
                    apply_repetition_penalty_,
                )
                from megalm.kernels.sampling_logits import (
                    prepare_sampling_logits,
                )
                from megalm.kernels.topk_radix import topk_cutoffs

                kernel_bindings = SamplingKernelBindings(
                    use_triton=True,
                    greedy_argmax=greedy_argmax,
                    prepare_logits=prepare_sampling_logits,
                    apply_repetition_penalty=apply_repetition_penalty_,
                    apply_frequency_presence_penalty=apply_frequency_presence_penalty_,
                    topk_cutoffs=topk_cutoffs,
                    filtered_rejection_sample=filtered_rejection_sample,
                )
            else:
                kernel_bindings = SamplingKernelBindings(
                    use_triton=False,
                    greedy_argmax=fallback.greedy_argmax,
                )
        if kernel_bindings.use_triton != use_triton:
            raise SamplingError(
                "sampling bindings do not match the selected provider"
            )
        self._greedy_argmax = kernel_bindings.greedy_argmax
        if kernel_bindings.use_triton:
            triton_callables = (
                kernel_bindings.prepare_logits,
                kernel_bindings.apply_repetition_penalty,
                kernel_bindings.apply_frequency_presence_penalty,
                kernel_bindings.topk_cutoffs,
                kernel_bindings.filtered_rejection_sample,
            )
            if any(item is None for item in triton_callables):
                raise SamplingError("Triton sampling bindings are incomplete")
            (
                self._prepare_logits,
                self._apply_repetition_penalty,
                self._apply_frequency_presence,
                self._topk_cutoffs,
                self._filtered_rejection_sample,
            ) = triton_callables
            self._sample_impl = self._sample_triton
        else:
            self._prepare_logits = None
            self._apply_repetition_penalty = None
            self._apply_frequency_presence = None
            self._topk_cutoffs = None
            self._filtered_rejection_sample = None
            self._sample_impl = self._sample_tensorplay
        self._generators: dict[str, tensorplay.Generator] = {}
        self._seeds: dict[str, int] = {}
        self._draw_counts: dict[str, int] = {}
        self._device_plans: OrderedDict[tuple[object, ...], _DeviceSamplingPlan] = (
            OrderedDict()
        )
        self._workspaces: OrderedDict[
            tuple[object, ...],
            _SamplingWorkspace,
        ] = OrderedDict()
        self._greedy_workspaces: OrderedDict[
            tuple[object, ...],
            _GreedyWorkspace,
        ] = OrderedDict()

    def sample(
        self,
        logits: tensorplay.Tensor,
        params: tuple[SamplingParams | None, ...],
        request_ids: tuple[str, ...],
        histories: tuple[tuple[int, ...], ...],
        *,
        history_counts: tuple[tuple[int, ...], ...] = (),
        sample_indices: tensorplay.Tensor | None = None,
        return_logprobs: bool = False,
    ) -> tensorplay.Tensor | tuple[tensorplay.Tensor, tensorplay.Tensor]:
        batch_size = logits.shape[0]
        if logits.ndim != 2 or not (
            len(params)
            == len(request_ids)
            == len(histories)
            == batch_size
        ):
            raise SamplingError("sampling metadata does not match logits batch")
        active = tuple(
            index for index, item in enumerate(params) if item is not None
        )
        if not active:
            tokens = tensorplay.full(
                (batch_size,),
                -1,
                dtype=tensorplay.int64,
                device=logits.device,
            )
            if return_logprobs:
                return tokens, tensorplay.zeros(
                    (batch_size,),
                    dtype=tensorplay.float32,
                    device=logits.device,
                )
            return tokens
        sampled = self._sample_impl(
            logits,
            params,
            request_ids,
            histories,
            active,
            history_counts,
            sample_indices,
        )
        if return_logprobs:
            return (
                sampled,
                self._selected_logprobs(
                    logits,
                    sampled,
                    params,
                    histories,
                    history_counts,
                    active,
                ),
            )
        return sampled

    def _selected_logprobs(
        self,
        logits: tensorplay.Tensor,
        sampled: tensorplay.Tensor,
        params: tuple[SamplingParams | None, ...],
        histories: tuple[tuple[int, ...], ...],
        history_counts: tuple[tuple[int, ...], ...],
        active: tuple[int, ...],
    ) -> tensorplay.Tensor:
        """Log-probability of each sampled token under its request's
        penalty-adjusted, temperature-scaled distribution.

        Deterministically recomputed from the raw logits, so both sampling
        providers produce identical values without touching the kernels.
        """
        output = tensorplay.zeros(
            (logits.shape[0],),
            dtype=tensorplay.float32,
            device=logits.device,
        )
        vocab_size = logits.shape[1]
        for index in active:
            sampling = params[index]
            assert sampling is not None
            token = int(sampled[index])
            if token < 0 or token >= vocab_size:
                continue
            row = logits[index].float().clone()
            history = histories[index]
            if sampling.repetition_penalty != 1.0 and history:
                seen = tensorplay.tensor(
                    sorted(set(history)),
                    dtype=tensorplay.long,
                    device=row.device,
                )
                values = row.index_select(0, seen)
                penalized = tensorplay.where(
                    values < 0,
                    values * sampling.repetition_penalty,
                    values / sampling.repetition_penalty,
                )
                row.index_copy_(0, seen, penalized)
            if (
                sampling.frequency_penalty != 0.0
                or sampling.presence_penalty != 0.0
            ) and history:
                counts = (
                    history_counts[index]
                    if index < len(history_counts)
                    else ()
                )
                seen = tensorplay.tensor(
                    sorted(set(history)),
                    dtype=tensorplay.long,
                    device=row.device,
                )
                if len(counts) != len(seen):
                    raise SamplingError(
                        "frequency/presence penalty counts do not match"
                        " the history"
                    )
                count_values = tensorplay.tensor(
                    counts,
                    dtype=tensorplay.float32,
                    device=row.device,
                )
                values = row.index_select(0, seen)
                adjusted = values - (
                    sampling.presence_penalty
                    + sampling.frequency_penalty * count_values
                )
                row.index_copy_(0, seen, adjusted)
            temperature = sampling.temperature if sampling.temperature > 0 else 1.0
            if temperature != 1.0:
                row = row / temperature
            output[index] = tensorplay.log_softmax(row, dim=-1)[token]
        return output

    @staticmethod
    def can_use_greedy_graph(
        params: tuple[SamplingParams | None, ...],
    ) -> bool:
        """Whether every row has the exact allocation-free greedy semantics."""

        return bool(params) and all(
            item is not None
            and item.temperature == 0
            and item.repetition_penalty == 1.0
            for item in params
        )

    def release(self, request_id: str) -> None:
        self._generators.pop(request_id, None)
        self._seeds.pop(request_id, None)
        self._draw_counts.pop(request_id, None)
        stale = tuple(
            key
            for key, plan in self._device_plans.items()
            if request_id in plan.request_ids
        )
        for key in stale:
            self._device_plans.pop(key, None)

    def _sample_triton(
        self,
        logits: tensorplay.Tensor,
        params: tuple[SamplingParams | None, ...],
        request_ids: tuple[str, ...],
        histories: tuple[tuple[int, ...], ...],
        active: tuple[int, ...],
        history_counts: tuple[tuple[int, ...], ...],
        sample_indices: tensorplay.Tensor | None,
    ) -> tensorplay.Tensor:
        resolved_params = tuple(
            params[index] for index in active
        )
        assert all(item is not None for item in resolved_params)
        resolved_params = tuple(
            item for item in resolved_params if item is not None
        )
        pure_greedy = all(
            item.temperature == 0 and item.repetition_penalty == 1.0
            for item in resolved_params
        )
        if pure_greedy:
            all_rows = active == tuple(range(logits.shape[0]))
            workspace = self._greedy_workspace(logits)
            if sample_indices is not None:
                if (
                    sample_indices.shape != (logits.shape[0],)
                    or sample_indices.device != logits.device
                    or sample_indices.dtype not in {tensorplay.int32, tensorplay.int64}
                ):
                    raise SamplingError(
                        "sample indices do not match greedy logits"
                    )
                return self._greedy_argmax(
                    logits,
                    sample_indices=sample_indices,
                    output=workspace.sampled,
                    partial_values=workspace.partial_values,
                    partial_indices=workspace.partial_indices,
                )
            if all_rows:
                if logits.shape[0] == 1:
                    return tensorplay.argmax(
                        logits,
                        dim=-1,
                        out=workspace.sampled,
                    )
                return self._greedy_argmax(
                    logits,
                    output=workspace.sampled,
                    partial_values=workspace.partial_values,
                    partial_indices=workspace.partial_indices,
                )
            active_rows = (
                None
                if all_rows
                else tensorplay.tensor(
                    active,
                    dtype=tensorplay.long,
                    device=logits.device,
                )
            )
            selected = (
                logits
                if all_rows
                else logits.index_select(0, active_rows)
            )
            sampled = (
                tensorplay.argmax(selected, dim=-1)
                if selected.shape[0] == 1
                else self._greedy_argmax(selected)
            )
            if all_rows:
                return sampled
            output = tensorplay.full(
                (logits.shape[0],),
                -1,
                dtype=tensorplay.int64,
                device=logits.device,
            )
            assert active_rows is not None
            output.index_copy_(0, active_rows, sampled)
            return output

        plan = self._device_plan(
            logits,
            active,
            tuple(request_ids[index] for index in active),
            resolved_params,
        )
        workspace = self._workspace(logits, len(active))
        assert self._prepare_logits is not None
        scores = self._prepare_logits(
            logits,
            plan.active_rows,
            plan.temperatures,
            workspace.scores,
        )
        history_rows: list[int] = []
        history_tokens: list[int] = []
        penalty_rows: list[int] = []
        penalty_tokens: list[int] = []
        penalty_counts: list[int] = []
        for active_index, source_index in enumerate(active):
            item = resolved_params[active_index]
            needs_counts = (
                item.frequency_penalty != 0.0 or item.presence_penalty != 0.0
            )
            if item.repetition_penalty == 1.0 and not needs_counts:
                continue
            unique_tokens = sorted(set(histories[source_index]))
            if any(
                token < 0 or token >= logits.shape[1]
                for token in unique_tokens
            ):
                raise SamplingError("history token is outside the vocabulary")
            history_rows.extend([active_index] * len(unique_tokens))
            history_tokens.extend(unique_tokens)
            if needs_counts:
                counts = (
                    history_counts[source_index]
                    if source_index < len(history_counts)
                    else ()
                )
                if len(counts) != len(unique_tokens) or any(
                    count < 1 for count in counts
                ):
                    raise SamplingError(
                        "frequency/presence penalty counts do not match"
                        " the history"
                    )
                penalty_rows.extend([active_index] * len(unique_tokens))
                penalty_tokens.extend(unique_tokens)
                penalty_counts.extend(counts)
        assert self._apply_repetition_penalty is not None
        self._apply_repetition_penalty(
            scores,
            tensorplay.tensor(
                history_rows,
                dtype=tensorplay.long,
                device=logits.device,
            ),
            tensorplay.tensor(
                history_tokens,
                dtype=tensorplay.long,
                device=logits.device,
            ),
            plan.penalties,
        )
        if penalty_rows:
            assert self._apply_frequency_presence is not None
            self._apply_frequency_presence(
                scores,
                tensorplay.tensor(
                    penalty_rows,
                    dtype=tensorplay.long,
                    device=logits.device,
                ),
                tensorplay.tensor(
                    penalty_tokens,
                    dtype=tensorplay.long,
                    device=logits.device,
                ),
                tensorplay.tensor(
                    penalty_counts,
                    dtype=tensorplay.float32,
                    device=logits.device,
                ),
                plan.presence,
                plan.frequency,
            )

        sampled_active = workspace.sampled
        stochastic_rows_host = tuple(
            index
            for index, item in enumerate(resolved_params)
            if item.temperature > 0
        )
        greedy_rows_host = tuple(
            index
            for index, item in enumerate(resolved_params)
            if item.temperature == 0
        )
        if greedy_rows_host:
            greedy_scores = scores.index_select(0, plan.greedy_rows)
            greedy_tokens = (
                tensorplay.argmax(greedy_scores, dim=-1)
                if len(greedy_rows_host) == 1
                else self._greedy_argmax(greedy_scores)
            )
            sampled_active.index_copy_(
                0,
                plan.greedy_rows,
                greedy_tokens,
            )
        if stochastic_rows_host:
            counter_values = [
                (
                    (
                        self._seed_for(
                            request_ids[source_index],
                            resolved_params[active_index].seed,
                        )
                        >> 32
                    )
                    + self._draw_counts.get(
                        request_ids[source_index],
                        0,
                    )
                    * 0x27D4EB2D
                )
                & 0xFFFFFFFF
                for active_index, source_index in enumerate(active)
            ]
            workspace.counters.copy_(
                tensorplay.tensor(
                    counter_values,
                    dtype=tensorplay.int64,
                    device=logits.device,
                )
            )
            assert self._topk_cutoffs is not None
            topk_cutoff = self._topk_cutoffs(
                scores,
                plan.topk_rows,
                plan.top_k,
                max_top_k=max(
                    (
                        resolved_params[index].top_k
                        for index in range(len(resolved_params))
                        if 0 < resolved_params[index].top_k < logits.shape[1]
                    ),
                    default=1,
                ),
                cutoffs=workspace.cutoffs,
                prefixes=workspace.prefixes,
                remaining=workspace.remaining,
                histogram=workspace.histogram,
            )
            assert self._filtered_rejection_sample is not None
            self._filtered_rejection_sample(
                scores,
                plan.stochastic_rows,
                plan.top_k,
                topk_cutoff,
                plan.top_p,
                plan.min_p,
                plan.seeds,
                workspace.counters,
                sampled_active,
                workspace.valid,
            )
            for active_index in stochastic_rows_host:
                request_id = request_ids[active[active_index]]
                self._draw_counts[request_id] = (
                    self._draw_counts.get(request_id, 0) + 1
                )
        output = tensorplay.full(
            (logits.shape[0],),
            -1,
            dtype=tensorplay.int64,
            device=logits.device,
        )
        output.index_copy_(0, plan.active_rows, sampled_active)
        return output

    def _device_plan(
        self,
        logits: tensorplay.Tensor,
        active: tuple[int, ...],
        active_request_ids: tuple[str, ...],
        params: tuple[SamplingParams, ...],
    ) -> _DeviceSamplingPlan:
        key: tuple[object, ...] = (
            logits.device.type,
            logits.device.index,
            active,
            active_request_ids,
            params,
        )
        cached = self._device_plans.get(key)
        if cached is not None:
            self._device_plans.move_to_end(key)
            return cached
        device = logits.device
        stochastic_rows = tuple(
            index for index, item in enumerate(params) if item.temperature > 0
        )
        plan = _DeviceSamplingPlan(
            request_ids=active_request_ids,
            active_rows=tensorplay.tensor(
                active,
                dtype=tensorplay.long,
                device=device,
            ),
            temperatures=tensorplay.tensor(
                [item.temperature for item in params],
                dtype=tensorplay.float32,
                device=device,
            ),
            penalties=tensorplay.tensor(
                [item.repetition_penalty for item in params],
                dtype=tensorplay.float32,
                device=device,
            ),
            presence=tensorplay.tensor(
                [item.presence_penalty for item in params],
                dtype=tensorplay.float32,
                device=device,
            ),
            frequency=tensorplay.tensor(
                [item.frequency_penalty for item in params],
                dtype=tensorplay.float32,
                device=device,
            ),
            top_k=tensorplay.tensor(
                [item.top_k for item in params],
                dtype=tensorplay.int32,
                device=device,
            ),
            top_p=tensorplay.tensor(
                [item.top_p for item in params],
                dtype=tensorplay.float32,
                device=device,
            ),
            min_p=tensorplay.tensor(
                [item.min_p for item in params],
                dtype=tensorplay.float32,
                device=device,
            ),
            stochastic_rows=tensorplay.tensor(
                stochastic_rows,
                dtype=tensorplay.long,
                device=device,
            ),
            greedy_rows=tensorplay.tensor(
                [
                    index
                    for index, item in enumerate(params)
                    if item.temperature == 0
                ],
                dtype=tensorplay.long,
                device=device,
            ),
            topk_rows=tensorplay.tensor(
                [
                    index
                    for index in stochastic_rows
                    if 0 < params[index].top_k < logits.shape[1]
                ],
                dtype=tensorplay.long,
                device=device,
            ),
            seeds=tensorplay.tensor(
                [
                    self._seed_for(request_id, item.seed) & 0xFFFFFFFF
                    for request_id, item in zip(active_request_ids, params)
                ],
                dtype=tensorplay.int64,
                device=device,
            ),
        )
        self._device_plans[key] = plan
        self._device_plans.move_to_end(key)
        while len(self._device_plans) > 32:
            self._device_plans.popitem(last=False)
        return plan

    def _workspace(
        self,
        logits: tensorplay.Tensor,
        active_count: int,
    ) -> _SamplingWorkspace:
        key: tuple[object, ...] = (
            logits.device.type,
            logits.device.index,
            active_count,
            logits.shape[1],
        )
        cached = self._workspaces.get(key)
        if cached is not None:
            self._workspaces.move_to_end(key)
            return cached
        device = logits.device
        workspace = _SamplingWorkspace(
            scores=_mutable_empty(
                (active_count, logits.shape[1]),
                dtype=tensorplay.float32,
                device=device,
            ),
            sampled=_mutable_empty(
                active_count,
                dtype=tensorplay.int64,
                device=device,
            ),
            counters=_mutable_empty(
                active_count,
                dtype=tensorplay.int64,
                device=device,
            ),
            cutoffs=_mutable_empty(
                active_count,
                dtype=tensorplay.float32,
                device=device,
            ),
            prefixes=_mutable_empty(
                active_count,
                dtype=tensorplay.int64,
                device=device,
            ),
            remaining=_mutable_empty(
                active_count,
                dtype=tensorplay.int32,
                device=device,
            ),
            histogram=_mutable_empty(
                (active_count, 256),
                dtype=tensorplay.int32,
                device=device,
            ),
            valid=_mutable_empty(
                active_count,
                dtype=tensorplay.bool,
                device=device,
            ),
        )
        self._workspaces[key] = workspace
        self._workspaces.move_to_end(key)
        while len(self._workspaces) > 8:
            self._workspaces.popitem(last=False)
        return workspace

    def _greedy_workspace(self, logits: tensorplay.Tensor) -> _GreedyWorkspace:
        key: tuple[object, ...] = (
            logits.device.type,
            logits.device.index,
            logits.shape[0],
            logits.shape[1],
        )
        cached = self._greedy_workspaces.get(key)
        if cached is not None:
            self._greedy_workspaces.move_to_end(key)
            return cached
        num_chunks = (logits.shape[1] + 16383) // 16384
        needs_partials = logits.shape[1] > 65536
        workspace = _GreedyWorkspace(
            sampled=_mutable_empty(
                logits.shape[0],
                dtype=tensorplay.int64,
                device=logits.device,
            ),
            partial_values=(
                _mutable_empty(
                    (logits.shape[0], num_chunks),
                    dtype=tensorplay.float32,
                    device=logits.device,
                )
                if needs_partials
                else None
            ),
            partial_indices=(
                _mutable_empty(
                    (logits.shape[0], num_chunks),
                    dtype=tensorplay.int32,
                    device=logits.device,
                )
                if needs_partials
                else None
            ),
        )
        self._greedy_workspaces[key] = workspace
        self._greedy_workspaces.move_to_end(key)
        while len(self._greedy_workspaces) > 8:
            self._greedy_workspaces.popitem(last=False)
        return workspace

    def _sample_tensorplay(
        self,
        logits: tensorplay.Tensor,
        params: tuple[SamplingParams | None, ...],
        request_ids: tuple[str, ...],
        histories: tuple[tuple[int, ...], ...],
        active: tuple[int, ...],
        history_counts: tuple[tuple[int, ...], ...],
        sample_indices: tensorplay.Tensor | None,
    ) -> tensorplay.Tensor:
        del sample_indices
        output = tensorplay.full(
            (logits.shape[0],),
            -1,
            dtype=tensorplay.int64,
            device=logits.device,
        )
        for index in active:
            sampling = params[index]
            assert sampling is not None
            row = logits[index].float().clone()
            history = histories[index]
            if sampling.repetition_penalty != 1.0 and history:
                seen = tensorplay.tensor(
                    sorted(set(history)),
                    dtype=tensorplay.long,
                    device=row.device,
                )
                values = row.index_select(0, seen)
                penalized = tensorplay.where(
                    values < 0,
                    values * sampling.repetition_penalty,
                    values / sampling.repetition_penalty,
                )
                row.index_copy_(0, seen, penalized)
            if (
                sampling.frequency_penalty != 0.0
                or sampling.presence_penalty != 0.0
            ) and history:
                counts = (
                    history_counts[index]
                    if index < len(history_counts)
                    else ()
                )
                seen = tensorplay.tensor(
                    sorted(set(history)),
                    dtype=tensorplay.long,
                    device=row.device,
                )
                if len(counts) != len(seen):
                    raise SamplingError(
                        "frequency/presence penalty counts do not match"
                        " the history"
                    )
                count_values = tensorplay.tensor(
                    counts,
                    dtype=tensorplay.float32,
                    device=row.device,
                )
                values = row.index_select(0, seen)
                adjusted = values - (
                    sampling.presence_penalty
                    + sampling.frequency_penalty * count_values
                )
                row.index_copy_(0, seen, adjusted)
            if sampling.temperature == 0:
                output[index] = tensorplay.argmax(row)
                continue
            row /= sampling.temperature
            if sampling.top_k:
                cutoff = tensorplay.topk(
                    row,
                    min(sampling.top_k, row.numel()),
                ).values[-1]
                row = tensorplay.where(row < cutoff, -tensorplay.inf, row)
            probabilities = tensorplay.softmax(row, dim=-1)
            if sampling.min_p:
                threshold = probabilities.max() * sampling.min_p
                probabilities = tensorplay.where(
                    probabilities >= threshold,
                    probabilities,
                    0,
                )
            if sampling.top_p < 1:
                sorted_probabilities, sorted_indices = tensorplay.sort(
                    probabilities,
                    descending=True,
                )
                cumulative = tensorplay.cumsum(sorted_probabilities, dim=-1)
                remove = cumulative - sorted_probabilities > sampling.top_p
                sorted_probabilities = sorted_probabilities.masked_fill(remove, 0)
                probabilities.zero_().scatter_(
                    0,
                    sorted_indices,
                    sorted_probabilities,
                )
            total = probabilities.sum()
            if not tensorplay.isfinite(total) or total <= 0:
                raise SamplingError("sampling filters removed every token")
            probabilities /= total
            generator = self._generator(
                request_ids[index],
                sampling.seed,
                logits.device,
            )
            output[index] = tensorplay.multinomial(
                probabilities,
                num_samples=1,
                generator=generator,
            )
        return output

    def _seed_for(self, request_id: str, seed: int | None) -> int:
        current = self._seeds.get(request_id)
        if current is not None:
            return current
        resolved = secrets.randbits(63) if seed is None else int(seed)
        resolved &= 0x7FFFFFFFFFFFFFFF
        self._seeds[request_id] = resolved
        return resolved

    def _generator(
        self,
        request_id: str,
        seed: int | None,
        device: tensorplay.device,
    ) -> tensorplay.Generator | None:
        if seed is None:
            return None
        generator = self._generators.get(request_id)
        if generator is None:
            # The Generator constructor only accepts a device on CUDA
            # backends; CPU generators are device-implicit.
            generator = (
                tensorplay.Generator()
                if device.type == "cpu"
                else tensorplay.Generator(device=device)
            )
            generator.manual_seed(seed)
            self._generators[request_id] = generator
        return generator
