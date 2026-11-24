"""Tensor-parallel weight sharding for dense checkpoint tensors.

The sharder runs inside a model's weight ``consume`` callback, before the
existing QKV/gate-up packing logic. Every rank receives the same checkpoint
tensor and keeps only its slice, so the loader contract (one source tensor,
one callback) stays unchanged.

Sharding rules (Megatron-style, matching ``planning._validate_tensor_parallel``):

* query/key/value and gate/up projections: column-parallel, split on dim 0;
* attention output and MLP down projections: row-parallel, split on dim 1;
* LM head: column-parallel on the vocabulary axis; logits are all-reduced;
* everything else (norms, embeddings, RoPE tables): replicated.

Packed QKV and gate-up checkpoints are split into their logical halves first,
each half is sliced, then the rank-local halves are concatenated. This keeps
per-rank head groups intact.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

import tensorplay


class ShardError(ValueError):
    """Raised when a checkpoint tensor cannot be sharded for TP."""


@dataclass(frozen=True, slots=True)
class ShardGeometry:
    """Full (unsharded) widths needed to shard packed checkpoint tensors."""

    hidden_size: int
    query_width: int
    kv_width: int
    intermediate_size: int
    vocab_size: int


_QKV_SPLIT = re.compile(
    r"(self_attn\.(?:q_proj|k_proj|v_proj|qkv_proj)|attention\.wqkv)"
)
_O_PROJ = re.compile(r"(self_attn\.o_proj|attention\.wo)\.weight$")
_DOWN_PROJ = re.compile(r"mlp\.down_proj\.weight$")
_GATE_UP = re.compile(r"mlp\.(?:gate_proj|up_proj|gate_up_proj|gate_up_weight)")
_LM_HEAD = re.compile(r"(?:lm_head|output)\.weight$")


def _split_evenly(width: int, parts: int) -> int:
    if width % parts:
        raise ShardError(f"width {width} is not divisible by TP={parts}")
    return width // parts


class TPWeightSharder:
    """Slice a checkpoint tensor to one rank's local geometry."""

    def __init__(self, geometry: ShardGeometry, tp_size: int, tp_rank: int) -> None:
        if tp_size <= 0:
            raise ShardError("TP size must be positive")
        if not 0 <= tp_rank < tp_size:
            raise ShardError("TP rank is outside the world")
        self.geometry = geometry
        self.tp_size = tp_size
        self.tp_rank = tp_rank

    @property
    def enabled(self) -> bool:
        return self.tp_size > 1

    def shard(self, source_name: str, tensor: tensorplay.Tensor) -> tensorplay.Tensor:
        if not self.enabled:
            return tensor
        if _QKV_SPLIT.search(source_name):
            return self._shard_qkv(source_name, tensor)
        if _O_PROJ.search(source_name):
            return self._slice(tensor, axis=1, full=self.geometry.hidden_size)
        if _DOWN_PROJ.search(source_name):
            return self._slice(tensor, axis=1, full=self.geometry.intermediate_size)
        if _GATE_UP.search(source_name):
            return self._shard_gate_up(source_name, tensor)
        if _LM_HEAD.search(source_name):
            return self._slice(tensor, axis=0, full=self.geometry.vocab_size)
        return tensor  # replicated parameter

    def _slice(self, tensor: tensorplay.Tensor, *, axis: int, full: int) -> tensorplay.Tensor:
        local = _split_evenly(full, self.tp_size)
        start = self.tp_rank * local
        if axis == 0:
            return tensor[start : start + local]
        return tensor[:, start : start + local]

    def _shard_qkv(self, source_name: str, tensor: tensorplay.Tensor) -> tensorplay.Tensor:
        geo = self.geometry
        if "wqkv" in source_name:
            raise ShardError("packed wqkv sharding requires explicit head geometry")
        query_width = geo.query_width
        kv_width = geo.kv_width
        # Check packed names first: "qkv_proj.weight" also ends with
        # "v_proj.weight", and "gate_up_proj" ends with "up_proj".
        if source_name.endswith("qkv_proj.weight"):
            total = query_width + 2 * kv_width
            if tensor.shape[0] != total:
                raise ShardError(f"packed QKV width {tensor.shape[0]} != {total}")
            query = tensor[:query_width]
            key = tensor[query_width : query_width + kv_width]
            value = tensor[query_width + kv_width :]
            return tensorplay.cat(
                (
                    self._slice(query, axis=0, full=query_width),
                    self._slice(key, axis=0, full=kv_width),
                    self._slice(value, axis=0, full=kv_width),
                ),
                dim=0,
            )
        if source_name.endswith("q_proj.weight") or source_name.endswith("q_proj.bias"):
            return self._slice(tensor, axis=0, full=query_width)
        if source_name.endswith("k_proj.weight") or source_name.endswith("k_proj.bias"):
            return self._slice(tensor, axis=0, full=kv_width)
        if source_name.endswith("v_proj.weight") or source_name.endswith("v_proj.bias"):
            return self._slice(tensor, axis=0, full=kv_width)
        raise ShardError(f"unsupported QKV checkpoint tensor: {source_name}")

    def _shard_gate_up(self, source_name: str, tensor: tensorplay.Tensor) -> tensorplay.Tensor:
        geo = self.geometry
        width = geo.intermediate_size
        if source_name.endswith("gate_up_proj.weight") or source_name.endswith("gate_up_weight"):
            if tensor.shape[0] != 2 * width:
                raise ShardError(f"packed gate-up width {tensor.shape[0]} != {2 * width}")
            gate = tensor[:width]
            up = tensor[width:]
            return tensorplay.cat(
                (
                    self._slice(gate, axis=0, full=width),
                    self._slice(up, axis=0, full=width),
                ),
                dim=0,
            )
        if source_name.endswith("gate_proj.weight") or source_name.endswith("up_proj.weight"):
            return self._slice(tensor, axis=0, full=width)
        raise ShardError(f"unsupported gate/up checkpoint tensor: {source_name}")
