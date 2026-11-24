import json
from pathlib import Path

import pytest
import tensorplay
from tensorplay.serialization import save

from megalm.loading.safetensors import (
    KV_SCALE_SUFFIXES,
    SafetensorsSource,
    WeightFormatError,
    is_kv_scale_tensor,
    read_kv_scale_vectors,
)


def _save_file(tensors, path):
    save(dict(tensors), str(path))


def test_kv_scale_vectors_reduce_gqa_and_layers(tmp_path: Path) -> None:
    _save_file(
        {
            "model.layers.0.self_attn.q_proj.weight": tensorplay.ones(2, 2),
            "model.layers.0.self_attn.k_scale": tensorplay.tensor(
                [[1.0], [2.0], [3.0], [4.0]]
            ).reshape(4, 1, 1),
            "model.layers.1.self_attn.k_scale": tensorplay.tensor(
                [[2.0], [1.0], [4.0], [3.0]]
            ).reshape(4, 1, 1),
            "model.layers.0.self_attn.v_scale": tensorplay.tensor(
                [[0.5], [0.25], [0.75], [1.0]]
            ).reshape(4, 1, 1),
            "model.layers.0.self_attn.q_scale": tensorplay.tensor(
                [[1.0], [3.0], [2.0], [4.0], [1.5], [2.5], [3.5], [0.5]]
            ).reshape(8, 1, 1),
        },
        tmp_path / "model.safetensors",
    )
    scales = read_kv_scale_vectors(
        tmp_path, num_kv_heads=4, num_query_heads=8
    )
    # Per-layer key scales reduce by elementwise amax -> [2,2,4,4].
    assert scales["key_scale"] == (2.0, 2.0, 4.0, 4.0)
    assert scales["value_scale"] == (0.5, 0.25, 0.75, 1.0)
    # Query scales reduce by GQA group amax: [max(1,3), max(2,4),
    # max(1.5,2.5), max(3.5,0.5)] -> [3,4,2.5,3.5].
    assert scales["q_scale"] == (3.0, 4.0, 2.5, 3.5)
    assert is_kv_scale_tensor("model.layers.0.self_attn.k_scale")
    assert not is_kv_scale_tensor("model.layers.0.self_attn.q_proj.weight")

    # Weight loading can exclude the metadata so model consumers never see it.
    weights = SafetensorsSource(
        tmp_path, exclude_suffixes=KV_SCALE_SUFFIXES
    )
    assert tuple(info.name for info in weights.manifest) == (
        "model.layers.0.self_attn.q_proj.weight",
    )


def test_streams_a_sharded_manifest_in_deterministic_tensor_order(
    tmp_path: Path,
) -> None:
    _save_file({"z.weight": tensorplay.ones(2, 3)}, tmp_path / "part-2.safetensors")
    _save_file({"a.weight": tensorplay.arange(4)}, tmp_path / "part-1.safetensors")
    (tmp_path / "model.safetensors.index.json").write_text(
        json.dumps(
            {
                "metadata": {"total_size": 40},
                "weight_map": {
                    "z.weight": "part-2.safetensors",
                    "a.weight": "part-1.safetensors",
                },
            }
        ),
        encoding="utf-8",
    )

    source = SafetensorsSource(tmp_path)
    seen: list[tuple[str, tuple[int, ...], float]] = []
    report = source.stream(
        lambda name, tensor: seen.append(
            (name, tuple(tensor.shape), float(tensor.sum().item()))
        )
    )

    assert tuple(info.name for info in source.manifest) == ("a.weight", "z.weight")
    assert [name for name, _, _ in seen] == ["a.weight", "z.weight"]
    assert report.tensor_count == 2
    assert report.total_bytes == source.total_bytes
    assert report.source_names == report.target_names


def test_index_mismatch_fails_before_materializing_weights(tmp_path: Path) -> None:
    _save_file({"actual": tensorplay.ones(1)}, tmp_path / "part.safetensors")
    (tmp_path / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {"declared": "part.safetensors"}}),
        encoding="utf-8",
    )

    with pytest.raises(WeightFormatError, match="index mismatch"):
        _ = SafetensorsSource(tmp_path).manifest


def test_index_cannot_escape_model_directory(tmp_path: Path) -> None:
    outside = tmp_path.parent / "outside.safetensors"
    _save_file({"weight": tensorplay.ones(1)}, outside)
    (tmp_path / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {"weight": "../outside.safetensors"}}),
        encoding="utf-8",
    )

    with pytest.raises(WeightFormatError, match="escapes"):
        SafetensorsSource(tmp_path)


def test_manifest_is_inspected_without_a_state_dict(tmp_path: Path) -> None:
    # A tiny synthetic shard exercises the same header-only path as a full
    # checkpoint: no model assets are needed to inspect names and byte counts.
    _save_file(
        {
            "model.llm.attn_res.attn_queries.0": tensorplay.ones(2, 2),
            "model.llm.attn_res.attn_keys.0": tensorplay.ones(2, 3),
        },
        tmp_path / "model.safetensors",
    )
    source = SafetensorsSource(tmp_path)

    assert tuple(info.name for info in source.manifest) == (
        "model.llm.attn_res.attn_keys.0",
        "model.llm.attn_res.attn_queries.0",
    )
    assert source.total_bytes == (2 * 3 + 2 * 2) * 4

