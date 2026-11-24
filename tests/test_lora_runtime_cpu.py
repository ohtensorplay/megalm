"""CPU tests for PEFT adapter loading, packed banks and slot management."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import pytest

tensorplay = pytest.importorskip("tensorplay")
from tensorplay.serialization import save

from megalm.contracts.lora import (
    LoRAConfig,
    LoRARequest,
    lora_http_status,
    lora_retryable,
)
from megalm.loading.lora import load_lora_adapter
from megalm.runtime.lora import (
    LoRAAdapterNotFound,
    LoRAError,
    LoRAManager,
    LoRASlotsExhausted,
    LoRAValidationError,
)

HIDDEN = 16
Q_WIDTH = 16
KV_WIDTH = 8
INTERMEDIATE = 32
RANK = 4
TARGETS = (
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
)
GEOMETRY = {
    "q_proj": (Q_WIDTH, HIDDEN),
    "k_proj": (KV_WIDTH, HIDDEN),
    "v_proj": (KV_WIDTH, HIDDEN),
    "o_proj": (HIDDEN, Q_WIDTH),
    "gate_proj": (INTERMEDIATE, HIDDEN),
    "up_proj": (INTERMEDIATE, HIDDEN),
    "down_proj": (HIDDEN, INTERMEDIATE),
}


def _write_adapter(root: Path) -> dict[str, tuple[tensorplay.Tensor, tensorplay.Tensor]]:
    raw: dict[str, tuple[tensorplay.Tensor, tensorplay.Tensor]] = {}
    tensors: dict[str, tensorplay.Tensor] = {}
    for target in TARGETS:
        out, in_features = GEOMETRY[target]
        a = tensorplay.arange(1, RANK * in_features + 1, dtype=tensorplay.float32)
        a = (a / 100.0).reshape(RANK, in_features)
        b = tensorplay.arange(1, out * RANK + 1, dtype=tensorplay.float32)
        b = (b / 100.0).reshape(out, RANK)
        stem = f"base_model.model.model.layers.0.{target}"
        tensors[f"{stem}.lora_A.weight"] = a
        tensors[f"{stem}.lora_B.weight"] = b
        raw[target] = (a, b)
    root.mkdir(parents=True, exist_ok=True)
    (root / "adapter_config.json").write_text(
        json.dumps(
            {
                "r": RANK,
                "lora_alpha": 8,
                "target_modules": list(TARGETS),
                "bias": "none",
                "use_rslora": False,
            }
        ),
        encoding="utf-8",
    )
    save(tensors, str(root / "adapter_model.safetensors"))
    return raw


def test_packed_adapter_shapes_and_scaling(tmp_path: Path) -> None:
    raw = _write_adapter(tmp_path / "adapter")
    adapter = load_lora_adapter(
        tmp_path / "adapter",
        max_lora_rank=16,
        dtype=tensorplay.float32,
        device="cpu",
    )
    assert tuple(adapter.weights["qkv_proj"].lora_a.shape) == (3 * RANK, HIDDEN)
    assert tuple(adapter.weights["qkv_proj"].lora_b.shape) == (
        Q_WIDTH + 2 * KV_WIDTH,
        3 * RANK,
    )
    assert tuple(adapter.weights["gate_up_proj"].lora_b.shape) == (
        2 * INTERMEDIATE,
        2 * RANK,
    )
    assert tuple(adapter.weights["down_proj"].lora_b.shape) == (
        HIDDEN,
        RANK,
    )
    # alpha / rank is folded into B exactly once.
    q_raw_b = raw["q_proj"][1]
    assert tensorplay.allclose(
        adapter.weights["qkv_proj"].lora_b[:Q_WIDTH, :RANK],
        q_raw_b * 2.0,
        atol=1e-6,
    )
    # Block-diagonal placement keeps q/k/v slices out of each other's columns.
    assert bool(
        tensorplay.all(
            adapter.weights["qkv_proj"].lora_b[Q_WIDTH:, :RANK] == 0
        )
    )


@dataclass
class _FakeModel:
    config: SimpleNamespace
    model: SimpleNamespace
    state: object | None = None

    def bind_lora(self, state: object) -> None:
        self.state = state


def _fake_model() -> _FakeModel:
    def linear(out: int, in_features: int) -> SimpleNamespace:
        return SimpleNamespace(
            weight=tensorplay.zeros(out, in_features, dtype=tensorplay.float32)
        )

    attention = SimpleNamespace(
        qkv_proj=linear(Q_WIDTH + 2 * KV_WIDTH, HIDDEN),
        o_proj=linear(HIDDEN, Q_WIDTH),
    )
    mlp = SimpleNamespace(
        gate_up_weight=tensorplay.zeros(2 * INTERMEDIATE, HIDDEN),
        down_proj=linear(HIDDEN, INTERMEDIATE),
    )
    return _FakeModel(
        config=SimpleNamespace(model_type="llama"),
        model=SimpleNamespace(
            layers=[SimpleNamespace(self_attn=attention, mlp=mlp)]
        ),
    )


def _manager(max_loras: int = 2) -> LoRAManager:
    return LoRAManager(
        LoRAConfig(enable_lora=True, max_loras=max_loras, max_lora_rank=16),
        device=tensorplay.device("cpu"),
        dtype=tensorplay.float32,
        use_triton=False,
    )


def test_manager_slot_lifecycle_and_lru(tmp_path: Path) -> None:
    _write_adapter(tmp_path / "a")
    _write_adapter(tmp_path / "b")
    _write_adapter(tmp_path / "c")
    manager = _manager(max_loras=2)
    manager.bind_model(_fake_model())
    a = manager.load(tmp_path / "a", name="a")
    b = manager.load(tmp_path / "b", name="b")
    assert a["slot"] == 0 and b["slot"] == 1
    assert manager.pin(LoRARequest(lora_name="a", lora_int_id=a["id"])) == 0
    with pytest.raises(LoRAError, match="still used"):
        manager.unload("a")
    manager.unpin("a")
    # A third adapter evicts the least-recently active unpinned slot.
    c = manager.load(tmp_path / "c", name="c")
    assert c["slot"] in {0, 1}
    assert manager.stats()["active_slots"] == 2
    assert manager.stats()["registered"] == 3
    manager.unload("b")
    assert manager.stats()["registered"] == 2


def test_manager_rejects_unknown_and_pinned_adapters(tmp_path: Path) -> None:
    _write_adapter(tmp_path / "a")
    manager = _manager(max_loras=1)
    manager.bind_model(_fake_model())
    with pytest.raises(LoRAError, match="unknown"):
        manager.pin(LoRARequest(lora_name="missing"))
    adapter = manager.load(tmp_path / "a", name="a")
    manager.pin(LoRARequest(lora_name="a", lora_int_id=adapter["id"]))
    with pytest.raises(LoRAError, match="pinned"):
        manager.load(tmp_path / "a", name="another")


def test_state_bank_geometry_and_apply() -> None:
    manager = _manager()
    manager.bind_model(_fake_model())
    state = manager.state
    assert state is not None
    assert tuple(state.banks["qkv_proj"].a.shape) == (
        2,
        3 * 16,
        HIDDEN,
    )
    assert tuple(state.banks["o_proj"].b.shape) == (2, HIDDEN, 16)
    # An empty slot contributes nothing and never allocates a new tensor.
    hidden = tensorplay.ones(3, HIDDEN, dtype=tensorplay.float32)
    output = tensorplay.zeros(3, Q_WIDTH + 2 * KV_WIDTH, dtype=tensorplay.float32)
    slots = tensorplay.full((3,), -1, dtype=tensorplay.int32)
    state.apply("qkv_proj", hidden, output, slots)
    assert bool(tensorplay.all(output == 0))


def test_lora_errors_carry_stable_client_codes() -> None:
    assert LoRAAdapterNotFound("x").code == "lora_adapter_not_found"
    assert lora_retryable("lora_adapter_not_found") is False
    assert lora_http_status("lora_adapter_not_found") == 404
    assert LoRASlotsExhausted("x").code == "lora_slots_exhausted"
    assert lora_retryable("lora_slots_exhausted") is True
    assert lora_http_status("lora_slots_exhausted") == 429
    assert lora_http_status("unexpected") == 503


def test_validation_failure_does_not_evict_lru_victim(tmp_path: Path) -> None:
    _write_adapter(tmp_path / "a")
    _write_adapter(tmp_path / "b")
    manager = _manager(max_loras=1)
    manager.bind_model(_fake_model())
    a = manager.load(tmp_path / "a", name="a")
    # Register but do not activate B, then give it an impossible rank.
    manager.load(tmp_path / "b", name="b", activate=False)
    b_id = manager._by_name["b"]
    bad_weight = manager._adapters[b_id].weights["qkv_proj"]
    object.__setattr__(bad_weight, "rank", bad_weight.rank + 1)
    with pytest.raises(LoRAValidationError):
        manager.pin(LoRARequest(lora_name="b", lora_int_id=b_id))
    assert manager._slots[0] == a["id"]
    assert manager._adapters[a["id"]].slot == 0
    assert manager._adapters[b_id].slot == -1


def test_failed_bank_write_restores_evicted_adapter(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _write_adapter(tmp_path / "a")
    _write_adapter(tmp_path / "b")
    manager = _manager(max_loras=1)
    manager.bind_model(_fake_model())
    a = manager.load(tmp_path / "a", name="a")
    manager.load(tmp_path / "b", name="b", activate=False)
    b_id = manager._by_name["b"]
    state = manager.state
    assert state is not None
    original_set_lora = state.set_lora
    b_weights = manager._adapters[b_id].weights

    def failing_set_lora(slot: int, weights: object, **kwargs: object) -> None:
        if weights is b_weights:
            raise RuntimeError("simulated bank write failure")
        original_set_lora(slot, weights, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(state, "set_lora", failing_set_lora)
    with pytest.raises(RuntimeError, match="simulated bank write failure"):
        manager.pin(LoRARequest(lora_name="b", lora_int_id=b_id))
    assert manager._slots[0] == a["id"]
    assert manager._adapters[a["id"]].slot == 0
    assert manager._adapters[b_id].slot == -1
    assert manager._adapters[b_id].refs == 0
    expected = manager._adapters[a["id"]].weights["qkv_proj"].lora_b
    bank_slot = state.banks["qkv_proj"].b[0]
    assert bool(
        tensorplay.allclose(bank_slot[:, : expected.shape[1]], expected)
    )
    assert bool(tensorplay.all(bank_slot[:, expected.shape[1] :] == 0))


def test_manager_exports_monotonic_counters(tmp_path: Path) -> None:
    _write_adapter(tmp_path / "a")
    manager = _manager(max_loras=1)
    manager.bind_model(_fake_model())
    a = manager.load(tmp_path / "a", name="a")
    manager.pin(LoRARequest(lora_name="a", lora_int_id=a["id"]))
    with pytest.raises(LoRASlotsExhausted):
        manager.load(tmp_path / "a", name="b", activate=True)
    manager.unpin("a")
    manager.unload("a")
    stats = manager.stats()
    assert stats["loads"] == 1
    assert stats["activations"] == 1
    assert stats["pins"] == 1
    assert stats["rejections"] == 1
    assert stats["unloads"] == 1
    assert stats["failed_loads"] == 1




def test_manager_restricts_runtime_loads_to_allowed_roots(tmp_path: Path) -> None:
    allowed = tmp_path / "allowed"
    outside = tmp_path / "outside"
    _write_adapter(allowed / "a")
    _write_adapter(outside / "b")
    manager = LoRAManager(
        LoRAConfig(
            enable_lora=True,
            max_loras=1,
            max_lora_rank=16,
            allowed_roots=(str(allowed),),
        ),
        device=tensorplay.device("cpu"),
        dtype=tensorplay.float32,
        use_triton=False,
    )
    manager.bind_model(_fake_model())
    with pytest.raises(LoRAValidationError, match="allowed roots"):
        manager.load(outside / "b", name="b")
    assert manager.load(allowed / "a", name="a")["name"] == "a"
