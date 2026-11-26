"""Multi-layer PEFT adapters must stay layer-local when packed and activated."""

from __future__ import annotations

import json

import tensorplay as tp

from megalm.contracts.lora import GENERIC_LORA_MODEL_TYPES, LoRAAdapterSpec
from megalm.loading.lora import load_lora_adapter
from megalm.runtime.lora import LoRABank, LoRAState

RANK = 4
TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


def _write_multilayer_adapter(root, layers: int) -> None:
    root.mkdir(parents=True, exist_ok=True)
    (root / "adapter_config.json").write_text(
        json.dumps(
            {"peft_type": "LORA", "r": RANK, "lora_alpha": RANK, "target_modules": TARGETS}
        )
    )
    tensors: dict[str, object] = {}
    for layer in range(layers):
        for target in TARGETS:
            if target in ("q_proj", "o_proj", "gate_proj", "down_proj"):
                out, width = 8, 8
            else:
                out, width = 8, 8
            base = f"base_model.model.model.layers.{layer}.self_attn.{target}"
            if target in ("gate_proj", "up_proj", "down_proj"):
                base = f"base_model.model.model.layers.{layer}.mlp.{target}"
            tensors[f"{base}.lora_A.default.weight"] = tp.ones(
                (RANK, width), dtype=tp.float32
            ) * (layer + 1)
            tensors[f"{base}.lora_B.default.weight"] = tp.ones(
                (out, RANK), dtype=tp.float32
            ) * (layer + 1)
    tp.save(tensors, root / "adapter_model.safetensors")


def test_multilayer_loader_keeps_each_layer_distinct(tmp_path) -> None:
    _write_multilayer_adapter(tmp_path / "adapter", layers=3)
    loaded = load_lora_adapter(
        tmp_path / "adapter", max_lora_rank=16, dtype=tp.float32, device="cpu"
    )
    assert loaded.layer_count == 3
    assert set(loaded.layers) == {0, 1, 2}
    assert set(loaded.layers[0]) == {"qkv_proj", "o_proj", "gate_up_proj", "down_proj"}
    assert loaded.weights == {}  # flat view is only for single-layer adapters
    first = loaded.layers[0]["o_proj"].lora_b
    last = loaded.layers[2]["o_proj"].lora_b
    assert float((last - first).abs().sum()) > 0
    assert LoRAAdapterSpec.from_path(tmp_path / "adapter", max_lora_rank=16).r == RANK


def test_multilayer_state_applies_per_layer_bank(tmp_path) -> None:
    _write_multilayer_adapter(tmp_path / "adapter", layers=2)
    loaded = load_lora_adapter(
        tmp_path / "adapter", max_lora_rank=16, dtype=tp.float32, device="cpu"
    )
    state = LoRAState(
        {
            index: {
                target: LoRABank(
                    a=tp.zeros((1, weight.rank, weight.lora_a.shape[1]), dtype=tp.float32),
                    b=tp.zeros((1, weight.lora_b.shape[0], weight.rank), dtype=tp.float32),
                    in_features=weight.lora_a.shape[1],
                    out_features=weight.lora_b.shape[0],
                    rank=weight.rank,
                )
                for target, weight in targets.items()
            }
            for index, targets in loaded.layers.items()
        },
        max_loras=1,
        use_triton=False,
    )
    state.set_lora(0, loaded.layers)
    hidden = tp.ones((2, 8), dtype=tp.float32)
    out0, out1 = tp.zeros((2, 8), dtype=tp.float32), tp.zeros((2, 8), dtype=tp.float32)
    slots = tp.zeros((2,), dtype=tp.int32)
    state.apply("o_proj", hidden, out0, slots, layer_index=0)
    state.apply("o_proj", hidden, out1, slots, layer_index=1)
    assert float((out1 - out0).abs().sum()) > 0


def test_minicpm_is_a_supported_lora_family() -> None:
    assert "minicpm" in GENERIC_LORA_MODEL_TYPES
