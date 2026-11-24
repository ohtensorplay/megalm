from __future__ import annotations

from dataclasses import InitVar, dataclass, field
from enum import IntEnum

import tensorplay


class BatchContractError(ValueError):
    """Raised before GPU execution when compact batch metadata is inconsistent."""


class BatchPhase(IntEnum):
    PREFILL = 0
    DECODE = 1


@dataclass(frozen=True, slots=True)
class ExecutionBatch:
    """Fixed, replayable executor input; it never contains Python requests."""

    input_ids: tensorplay.Tensor
    positions: tensorplay.Tensor
    slot_mapping: tensorplay.Tensor
    block_tables: tensorplay.Tensor
    sequence_lengths: tensorplay.Tensor
    token_active: tensorplay.Tensor
    query_start_loc: tuple[int, ...]
    query_lengths: tuple[int, ...]
    context_lengths: tuple[int, ...]
    phases: tuple[BatchPhase, ...]
    sample_indices: tensorplay.Tensor
    page_size: int
    query_start_loc_device: tensorplay.Tensor | None = None
    gdn_state_indices: tensorplay.Tensor | None = None
    state_slots: tensorplay.Tensor | None = None
    # One int32 slot index per packed token; -1 selects the base model.
    # None means the batch carries no LoRA references at all.
    lora_slots: tensorplay.Tensor | None = None
    mrope_positions: tensorplay.Tensor | None = None
    input_embeddings: tensorplay.Tensor | None = None
    deepstack_embeddings: tuple[tensorplay.Tensor, ...] = ()
    is_speculative: bool = False
    validate_contract: InitVar[bool] = True
    # Eager prefill can address state directly without synchronizing CUDA.
    host_state_slots: tuple[int, ...] | None = None
    _all_decode: bool = field(init=False, repr=False, compare=False)

    def __post_init__(self, validate_contract: bool) -> None:
        all_decode = all(
            phase is BatchPhase.DECODE for phase in self.phases
        )
        object.__setattr__(self, "_all_decode", all_decode)
        if not validate_contract:
            return
        object.__setattr__(self, "query_start_loc", tuple(self.query_start_loc))
        object.__setattr__(self, "query_lengths", tuple(self.query_lengths))
        object.__setattr__(self, "context_lengths", tuple(self.context_lengths))
        object.__setattr__(self, "phases", tuple(self.phases))
        if self.query_start_loc_device is None:
            object.__setattr__(
                self,
                "query_start_loc_device",
                tensorplay.tensor(
                    self.query_start_loc,
                    dtype=tensorplay.int32,
                    device=self.input_ids.device,
                ),
            )
        sequence_count = len(self.query_lengths)
        if sequence_count == 0:
            raise BatchContractError("ExecutionBatch must contain a sequence")
        if len(self.query_start_loc) != sequence_count + 1:
            raise BatchContractError("query_start_loc must have B + 1 entries")
        if (
            self.host_state_slots is not None
            and len(self.host_state_slots) != sequence_count
        ):
            raise BatchContractError("host_state_slots must have one entry per sequence")
        if (
            len(self.context_lengths) != sequence_count
            or len(self.phases) != sequence_count
        ):
            raise BatchContractError("per-sequence metadata lengths differ")
        if self.query_start_loc[0] != 0:
            raise BatchContractError("query_start_loc must start at zero")
        if any(length <= 0 for length in self.query_lengths):
            raise BatchContractError("query lengths must be positive")
        expected_starts = [0]
        for length in self.query_lengths:
            expected_starts.append(expected_starts[-1] + length)
        if tuple(expected_starts) != self.query_start_loc:
            raise BatchContractError("query_start_loc does not match query lengths")
        token_count = expected_starts[-1]
        if self.input_ids.ndim != 1 or self.input_ids.numel() != token_count:
            raise BatchContractError("input_ids must be a packed one-dimensional tensor")
        if self.input_ids.dtype != tensorplay.int64:
            raise BatchContractError("input_ids must use int64")
        for name, tensor in (
            ("positions", self.positions),
            ("slot_mapping", self.slot_mapping),
            ("token_active", self.token_active),
        ):
            if tensor.ndim != 1 or tensor.numel() != token_count:
                raise BatchContractError(f"{name} must have one entry per token")
            if tensor.device != self.input_ids.device:
                raise BatchContractError(f"{name} is on a different device")
        if self.positions.dtype != tensorplay.int64:
            raise BatchContractError("positions must use int64")
        if self.slot_mapping.dtype != tensorplay.int64:
            raise BatchContractError("slot_mapping must use int64")
        if self.token_active.dtype != tensorplay.bool:
            raise BatchContractError("token_active must use bool")
        if self.block_tables.ndim != 2 or self.block_tables.shape[0] != sequence_count:
            raise BatchContractError("block_tables must have shape [B, max_blocks]")
        if self.block_tables.device != self.input_ids.device:
            raise BatchContractError("block_tables is on a different device")
        if self.block_tables.dtype != tensorplay.int32:
            raise BatchContractError("block_tables must use int32")
        if (
            self.sequence_lengths.ndim != 1
            or self.sequence_lengths.numel() != sequence_count
        ):
            raise BatchContractError("sequence_lengths must have shape [B]")
        if self.sequence_lengths.device != self.input_ids.device:
            raise BatchContractError("sequence_lengths is on a different device")
        if self.sequence_lengths.dtype != tensorplay.int32:
            raise BatchContractError("sequence_lengths must use int32")
        expected_sample_rows = (
            token_count if self.is_speculative else sequence_count
        )
        if (
            self.sample_indices.ndim != 1
            or self.sample_indices.numel() != expected_sample_rows
        ):
            raise BatchContractError(
                "sample_indices do not match the execution mode"
            )
        if self.sample_indices.device != self.input_ids.device:
            raise BatchContractError("sample_indices is on a different device")
        if self.sample_indices.dtype != tensorplay.int64:
            raise BatchContractError("sample_indices must use int64")
        query_start_loc_device = self.query_start_loc_device
        assert query_start_loc_device is not None
        if (
            query_start_loc_device.ndim != 1
            or query_start_loc_device.numel() != sequence_count + 1
            or query_start_loc_device.dtype != tensorplay.int32
            or query_start_loc_device.device != self.input_ids.device
        ):
            raise BatchContractError(
                "query_start_loc_device must be int32 [B + 1] on the batch device"
            )
        if self.gdn_state_indices is not None and (
            self.gdn_state_indices.shape != (sequence_count, 2)
            or self.gdn_state_indices.dtype not in {tensorplay.int32, tensorplay.int64}
            or self.gdn_state_indices.device != self.input_ids.device
        ):
            raise BatchContractError(
                "gdn_state_indices must be integer [B, 2] on the batch device"
            )
        if self.state_slots is not None and (
            self.state_slots.ndim != 1
            or self.state_slots.numel() != sequence_count
            or self.state_slots.dtype not in {tensorplay.int32, tensorplay.int64}
            or self.state_slots.device != self.input_ids.device
        ):
            raise BatchContractError(
                "state_slots must be integer [B] on the batch device"
            )
        if self.lora_slots is not None and (
            self.lora_slots.ndim != 1
            or self.lora_slots.numel() != token_count
            or self.lora_slots.dtype != tensorplay.int32
            or self.lora_slots.device != self.input_ids.device
        ):
            raise BatchContractError(
                "lora_slots must be int32 [T] on the batch device"
            )
        if self.mrope_positions is not None and (
            self.mrope_positions.shape != (3, token_count)
            or self.mrope_positions.dtype != tensorplay.int64
            or self.mrope_positions.device != self.input_ids.device
        ):
            raise BatchContractError(
                "mrope_positions must be int64 [3, T] on the batch device"
            )
        if self.input_embeddings is not None and (
            self.input_embeddings.ndim != 2
            or self.input_embeddings.shape[0] != token_count
            or self.input_embeddings.device != self.input_ids.device
        ):
            raise BatchContractError(
                "input_embeddings must have one row per packed token"
            )
        for index, embeddings in enumerate(self.deepstack_embeddings):
            if embeddings.ndim != 2 or embeddings.shape[0] != token_count:
                raise BatchContractError(
                    f"deepstack_embeddings[{index}] must have one row per token"
                )
            if embeddings.device != self.input_ids.device:
                raise BatchContractError(
                    f"deepstack_embeddings[{index}] is on a different device"
                )
        if self.page_size <= 0:
            raise BatchContractError("page_size must be positive")
        expected_sequence_lengths: list[int] = []
        for query_length, context_length, phase in zip(
            self.query_lengths,
            self.context_lengths,
            self.phases,
        ):
            expected_sequence_length = query_length + context_length
            expected_sequence_lengths.append(expected_sequence_length)
            if phase is BatchPhase.DECODE and query_length != 1:
                raise BatchContractError("decode entries must contain one query token")
        if self.is_speculative and (
            sequence_count < 1
            or any(phase is not BatchPhase.PREFILL for phase in self.phases)
            or any(query_length <= 1 for query_length in self.query_lengths)
            or not bool(tensorplay.all(self.token_active).item())
        ):
            raise BatchContractError(
                "speculative verification requires active prefill chains"
            )
        if self.sequence_lengths.device.type == "cpu" and tuple(
            self.sequence_lengths.tolist()
        ) != tuple(expected_sequence_lengths):
            raise BatchContractError("sequence length metadata differs")

    @property
    def num_sequences(self) -> int:
        return len(self.query_lengths)

    @property
    def num_tokens(self) -> int:
        return self.input_ids.numel()

    @property
    def all_decode(self) -> bool:
        return self._all_decode

    @property
    def all_active(self) -> bool:
        return bool(tensorplay.all(self.token_active).item())
