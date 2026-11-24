from __future__ import annotations

import json
import math
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterator

import tensorplay
from tensorplay.serialization import inspect_checkpoint, load

from megalm.contracts.model import ModelSpec, WeightTransform


class WeightFormatError(ValueError):
    """Raised when a safetensors model is incomplete, ambiguous, or unsafe."""


# llm-compressor / ModelOpt KV scale names end in one of these suffixes.  They
# are quantization metadata, not model weights, so weight loading skips them.
KV_SCALE_SUFFIXES = (".k_scale", ".v_scale", ".q_scale")
_KV_SCALE_WHICH = {
    ".k_scale": "key",
    ".v_scale": "value",
    ".q_scale": "q",
}


def is_kv_scale_tensor(name: str) -> bool:
    """Whether ``name`` names a KV quantization scale rather than a weight."""
    return name.endswith(KV_SCALE_SUFFIXES)


def read_kv_scale_vectors(
    model_path: str | Path,
    *,
    num_kv_heads: int,
    num_query_heads: int,
    include_prefixes: tuple[str, ...] = (),
) -> dict[str, tuple[float, ...]]:
    """Read calibrated KV scales from a checkpoint and reduce them per KV head.

    llm-compressor stores ``k_scale``/``v_scale`` as ``[num_kv_heads, 1, 1]``
    and ``q_scale`` as ``[num_query_heads, 1, 1]``.  Following vLLM, query
    scales are reduced by taking the amax over each GQA group; K/V scales are
    already per-KV-head.  With multiple transformer layers the elementwise
    amax across layers is used, matching MegaLM's engine-wide
    :class:`~megalm.runtime.kv_quant.KVQuantConfig`.

    Returns a mapping with ``key_scale``/``value_scale``/``q_scale`` tuples;
    entries are absent when the checkpoint does not provide them.
    """
    if min(num_kv_heads, num_query_heads) <= 0:
        raise WeightFormatError("KV scale reduction needs positive head counts")
    if num_query_heads % num_kv_heads:
        raise WeightFormatError("num_query_heads must be divisible by num_kv_heads")
    query_per_kv = num_query_heads // num_kv_heads
    source = SafetensorsSource(model_path, include_prefixes=include_prefixes)
    by_shard: dict[Path, list[tuple[TensorInfo, str]]] = defaultdict(list)
    for info in source.manifest:
        for suffix, which in _KV_SCALE_WHICH.items():
            if info.name.endswith(suffix):
                by_shard[info.shard].append((info, which))
                break
    if not by_shard:
        return {}
    collected: dict[str, list[tensorplay.Tensor]] = defaultdict(list)
    for shard, entries in sorted(by_shard.items()):
        tensors = load(shard, map_location="cpu", mmap=True)
        for info, which in entries:
            raw = tensors[info.name].float().reshape(-1)
            if raw.numel() == 1:
                vector = raw.expand(num_kv_heads)
            elif which == "q" and raw.numel() == num_query_heads:
                vector = raw.reshape(num_kv_heads, query_per_kv).amax(dim=1)
            elif which != "q" and raw.numel() == num_kv_heads:
                vector = raw
            else:
                raise WeightFormatError(
                    f"unexpected {info.name!r} scale shape {tuple(info.shape)}"
                )
            collected[f"{which}_scale"].append(vector)
    result: dict[str, tuple[float, ...]] = {}
    for name, vectors in collected.items():
        stacked = tensorplay.stack(vectors, dim=0)
        reduced = stacked.amax(dim=0)
        result[name] = tuple(float(value) for value in reduced.tolist())
    return result


_DTYPE_BYTES = {
    "BOOL": 1,
    "U8": 1,
    "I8": 1,
    "F8_E4M3": 1,
    "F8_E5M2": 1,
    "F8_E8M0": 1,
    "U16": 2,
    "I16": 2,
    "F16": 2,
    "BF16": 2,
    "U32": 4,
    "I32": 4,
    "F32": 4,
    "U64": 8,
    "I64": 8,
    "F64": 8,
}


@dataclass(frozen=True, slots=True)
class TensorInfo:
    name: str
    shard: Path
    shape: tuple[int, ...]
    dtype: str

    @property
    def numel(self) -> int:
        return math.prod(self.shape)

    @property
    def nbytes(self) -> int:
        try:
            item_size = _DTYPE_BYTES[self.dtype]
        except KeyError as exc:
            raise WeightFormatError(
                f"unsupported safetensors dtype {self.dtype!r} for {self.name!r}"
            ) from exc
        return self.numel * item_size


@dataclass(frozen=True, slots=True)
class WeightLoadReport:
    tensor_count: int
    total_bytes: int
    source_names: tuple[str, ...]
    target_names: tuple[str, ...]


class SafetensorsSource:
    """Memory-map safetensors and hand one materialized tensor to a consumer at a time."""

    def __init__(
        self,
        model_path: str | Path,
        *,
        include_prefixes: tuple[str, ...] = (),
        exclude_suffixes: tuple[str, ...] = (),
    ) -> None:
        self._input_path = Path(model_path)
        self._root, self._weight_map = self._discover(self._input_path)
        self._include_prefixes = tuple(include_prefixes)
        self._exclude_suffixes = tuple(exclude_suffixes)
        if any(not prefix for prefix in self._include_prefixes):
            raise WeightFormatError("weight prefixes must not be empty")
        self._manifest: tuple[TensorInfo, ...] | None = None

    @property
    def root(self) -> Path:
        return self._root

    @property
    def manifest(self) -> tuple[TensorInfo, ...]:
        if self._manifest is None:
            self._manifest = self._inspect()
        return self._manifest

    @property
    def total_bytes(self) -> int:
        return sum(tensor.nbytes for tensor in self.manifest)

    def iter_tensors(
        self,
        *,
        device: str = "cpu",
    ) -> Iterator[tuple[TensorInfo, tensorplay.Tensor]]:
        """Yield tensors in deterministic order without accumulating a state dict."""

        by_shard: dict[Path, list[TensorInfo]] = defaultdict(list)
        for tensor in self.manifest:
            by_shard[tensor.shard].append(tensor)
        for shard in sorted(by_shard, key=lambda path: path.name):
            # TensorPlay's safetensors reader with mmap=True returns zero-copy
            # views over one private mapping; moving each tensor to the target
            # device happens here so a whole shard never lands on the GPU.
            tensors = load(shard, map_location="cpu", mmap=True)
            for info in sorted(by_shard[shard], key=lambda item: item.name):
                tensor = tensors[info.name]
                if device != "cpu":
                    tensor = tensor.to(device=device)
                yield info, tensor
                del tensor, tensors[info.name]

    def stream(
        self,
        consumer: Callable[[str, tensorplay.Tensor], None],
        *,
        model_spec: ModelSpec | None = None,
        device: str = "cpu",
    ) -> WeightLoadReport:
        """Load through a non-retaining consumer; transformations stay explicit."""

        source_names: list[str] = []
        target_names: list[str] = []
        total_bytes = 0
        for info, tensor in self.iter_tensors(device=device):
            target_name = info.name
            transform = WeightTransform.IDENTITY
            if model_spec is not None:
                target_name, transform = model_spec.map_weight_name(info.name)
            if transform is not WeightTransform.IDENTITY:
                raise WeightFormatError(
                    f"loader cannot apply {transform.value!r} to {info.name!r}"
                )
            consumer(target_name, tensor)
            source_names.append(info.name)
            target_names.append(target_name)
            total_bytes += info.nbytes
            del tensor
        return WeightLoadReport(
            tensor_count=len(source_names),
            total_bytes=total_bytes,
            source_names=tuple(source_names),
            target_names=tuple(target_names),
        )

    def _inspect(self) -> tuple[TensorInfo, ...]:
        by_shard: dict[Path, set[str] | None] = {}
        if self._weight_map is not None:
            grouped: dict[Path, set[str]] = defaultdict(set)
            for name, shard in self._weight_map.items():
                grouped[shard].add(name)
            by_shard = dict(grouped)
        else:
            for shard in self._shards_without_index():
                by_shard[shard] = None

        seen: set[str] = set()
        manifest: list[TensorInfo] = []
        for shard in sorted(by_shard, key=lambda path: path.name):
            expected_names = by_shard[shard]
            description = inspect_checkpoint(shard)
            tensor_specs = description.get("tensors")
            if not isinstance(tensor_specs, dict):
                raise WeightFormatError(
                    f"safetensors manifest is missing for {shard.name}"
                )
            actual_names = set(tensor_specs)
            if expected_names is not None and actual_names != expected_names:
                missing = sorted(expected_names - actual_names)
                unexpected = sorted(actual_names - expected_names)
                raise WeightFormatError(
                    f"index mismatch for {shard.name}: "
                    f"missing={missing}, unexpected={unexpected}"
                )
            for name in sorted(actual_names):
                if name in seen:
                    raise WeightFormatError(f"duplicate tensor name {name!r}")
                seen.add(name)
                spec = tensor_specs[name]
                info = TensorInfo(
                    name=name,
                    shard=shard,
                    shape=tuple(int(dimension) for dimension in spec["shape"]),
                    dtype=str(spec["dtype"]),
                )
                _ = info.nbytes
                manifest.append(info)
        if not manifest:
            raise WeightFormatError("safetensors source contains no tensors")
        if self._weight_map is not None and seen != set(self._weight_map):
            raise WeightFormatError("safetensors index does not cover all tensors")
        selected = tuple(
            sorted(manifest, key=lambda tensor: tensor.name)
        )
        if self._include_prefixes:
            selected = tuple(
                tensor
                for tensor in selected
                if tensor.name.startswith(self._include_prefixes)
            )
            if not selected:
                raise WeightFormatError(
                    "weight prefix filter selected no tensors"
                )
        if self._exclude_suffixes:
            selected = tuple(
                tensor
                for tensor in selected
                if not tensor.name.endswith(self._exclude_suffixes)
            )
        return selected

    def _shards_without_index(self) -> tuple[Path, ...]:
        if self._input_path.is_file():
            return (self._input_path.resolve(),)
        shards = tuple(sorted(self._root.glob("*.safetensors")))
        if not shards:
            raise WeightFormatError(f"no safetensors files under {self._root}")
        if len(shards) > 1:
            raise WeightFormatError(
                "multiple safetensors files require a versioned index"
            )
        return tuple(shard.resolve() for shard in shards)

    @staticmethod
    def _discover(
        input_path: Path,
    ) -> tuple[Path, dict[str, Path] | None]:
        if input_path.is_file():
            if input_path.suffix != ".safetensors":
                raise WeightFormatError("weight file must use the .safetensors format")
            return input_path.resolve().parent, None
        if not input_path.is_dir():
            raise WeightFormatError(f"model path does not exist: {input_path}")

        root = input_path.resolve()
        indices = tuple(sorted(root.glob("*.safetensors.index.json")))
        if len(indices) > 1:
            raise WeightFormatError(
                f"ambiguous safetensors indices: {[path.name for path in indices]}"
            )
        if not indices:
            return root, None
        try:
            payload = json.loads(indices[0].read_text(encoding="utf-8"))
            raw_weight_map = payload["weight_map"]
        except (OSError, json.JSONDecodeError, KeyError, TypeError) as exc:
            raise WeightFormatError(
                f"invalid safetensors index: {indices[0]}"
            ) from exc
        if not isinstance(raw_weight_map, dict) or not raw_weight_map:
            raise WeightFormatError("safetensors index weight_map must be non-empty")

        weight_map: dict[str, Path] = {}
        for name, shard_name in raw_weight_map.items():
            if not isinstance(name, str) or not isinstance(shard_name, str):
                raise WeightFormatError("safetensors weight_map entries must be strings")
            shard = (root / shard_name).resolve()
            try:
                shard.relative_to(root)
            except ValueError as exc:
                raise WeightFormatError(
                    f"safetensors shard escapes the model directory: {shard_name}"
                ) from exc
            if not shard.is_file():
                raise WeightFormatError(f"missing safetensors shard: {shard_name}")
            weight_map[name] = shard
        return root, weight_map
