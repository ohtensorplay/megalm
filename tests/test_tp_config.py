from pathlib import Path

import pytest

from megalm.runtime.engine import EngineConfig, EngineError


def _config(**overrides: object) -> EngineConfig:
    payload = dict(model_path=Path("/tmp/model"))
    payload.update(overrides)
    return EngineConfig(**payload)  # type: ignore[arg-type]


def test_tp1_config_is_default_and_valid():
    config = _config()
    assert config.tensor_parallel == 1
    assert config.tp_rank == 0
    assert config.tp_backend == "nccl"


def test_tp_config_validates_rank_and_backend():
    with pytest.raises(EngineError, match="tensor_parallel"):
        _config(tensor_parallel=0)
    with pytest.raises(EngineError, match="tp_rank"):
        _config(tensor_parallel=2, tp_rank=2)
    with pytest.raises(EngineError, match="tp_rank"):
        _config(tensor_parallel=1, tp_rank=1)
    with pytest.raises(EngineError, match="tp_backend"):
        _config(tensor_parallel=2, tp_backend="mpi")
    with pytest.raises(EngineError, match="tp_init_method"):
        _config(tensor_parallel=2, tp_init_method="")


def test_tp_config_accepts_a_valid_rank():
    config = _config(
        tensor_parallel=2,
        tp_rank=1,
        tp_backend="gloo",
        tp_init_method="tcp://127.0.0.1:29500",
    )
    assert config.tensor_parallel == 2
    assert config.tp_rank == 1
