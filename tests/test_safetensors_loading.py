import json
from pathlib import Path

import pytest
import tensorplay
from tensorplay.serialization import save

from megalm.loading.safetensors import SafetensorsSource, WeightFormatError


def _save_file(tensors, path):
    save(dict(tensors), str(path))


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


def test_bundled_model_manifest_is_inspected_without_a_state_dict() -> None:
    project_root = Path(__file__).resolve().parents[1]
    source = SafetensorsSource(project_root / "model")

    assert len(source.manifest) == 1244
    assert source.manifest[0].name == "model.llm.attn_res.attn_queries.0"
    assert source.total_bytes > 18 * 1024**3

