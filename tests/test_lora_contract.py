"""GPU-free LoRA contract tests: request parsing, PEFT config and packing."""

from __future__ import annotations

import math

import pytest

from megalm.contracts.lora import (
    LoRAAdapterSpec,
    LoRAConfig,
    LoRAContractError,
    LoRARequest,
    parse_lora_tensor_name,
    resolve_lora_targets,
)
from megalm.contracts.request import InferenceRequest, RequestContractError


def test_lora_request_accepts_vllm_style_payload() -> None:
    request = LoRARequest.from_payload(
        {
            "lora_name": "sql",
            "lora_int_id": 7,
            "lora_path": "/adapters/sql",
        }
    )
    assert request.name == "sql"
    assert request.adapter_id == 7
    assert request.path == "/adapters/sql"
    assert request.scale == 1.0
    assert request.to_wire()["lora_int_id"] == 7


def test_lora_request_accepts_aliases_and_name_only() -> None:
    assert LoRARequest.from_payload("style").lora_name == "style"
    assert LoRARequest.from_payload({"name": "style", "id": 3}).adapter_id == 3
    assert LoRARequest.from_wire({"lora_name": "x", "lora_int_id": 1}) is not None
    assert LoRARequest.from_wire(None) is None


@pytest.mark.parametrize(
    "payload",
    [
        {},
        {"lora_name": "", "lora_int_id": 0},
        {"lora_name": "x", "lora_int_id": -1},
        {"lora_name": "x", "scale": 0},
        {"lora_name": "x", "scale": float("nan")},
    ],
)
def test_lora_request_rejects_invalid_payloads(payload: object) -> None:
    with pytest.raises(LoRAContractError):
        LoRARequest.from_payload(payload)


def test_lora_config_validation() -> None:
    config = LoRAConfig(
        enable_lora=True,
        lora_paths=("/adapters/a", "/adapters/b"),
        max_loras=8,
        max_lora_rank=32,
    )
    assert len(config.lora_paths) == 2
    with pytest.raises(LoRAContractError):
        LoRAConfig(lora_paths=("/adapters/a",))
    with pytest.raises(LoRAContractError):
        LoRAConfig(max_lora_rank=3)
    with pytest.raises(LoRAContractError):
        LoRAConfig(lora_paths=("/a", "/a"), enable_lora=True)


def test_adapter_spec_scaling_and_validation() -> None:
    spec = LoRAAdapterSpec.from_dict(
        {
            "r": 8,
            "lora_alpha": 16,
            "target_modules": ["q_proj", "k_proj", "v_proj"],
        },
        max_lora_rank=16,
    )
    assert spec.scaling == pytest.approx(2.0)
    rslora = LoRAAdapterSpec.from_dict(
        {
            "r": 16,
            "lora_alpha": 32,
            "target_modules": ["q_proj"],
            "use_rslora": True,
        },
        max_lora_rank=32,
    )
    assert rslora.scaling == pytest.approx(32 / math.sqrt(16))
    with pytest.raises(LoRAContractError):
        LoRAAdapterSpec.from_dict(
            {"r": 32, "lora_alpha": 1, "target_modules": ["q_proj"]},
            max_lora_rank=16,
        )
    for unsupported in (
        {"bias": "all"},
        {"use_dora": True},
        {"modules_to_save": ["lm_head"]},
    ):
        with pytest.raises(LoRAContractError):
            LoRAAdapterSpec.from_dict(
                {
                    "r": 8,
                    "lora_alpha": 8,
                    "target_modules": ["q_proj"],
                    **unsupported,
                },
                max_lora_rank=16,
            )


def test_parse_peft_tensor_names() -> None:
    cases = {
        "base_model.model.model.layers.0.self_attn.q_proj.lora_A.weight": (
            "model.layers.0.self_attn.q_proj",
            True,
        ),
        "base_model.model.model.layers.1.mlp.down_proj.lora_B.default.weight": (
            "model.layers.1.mlp.down_proj",
            False,
        ),
    }
    for name, expected in cases.items():
        assert parse_lora_tensor_name(name) == expected
    with pytest.raises(LoRAContractError):
        parse_lora_tensor_name("model.layers.0.weight")


def test_target_mapping_preserves_fixed_order() -> None:
    spec = LoRAAdapterSpec(
        r=8,
        lora_alpha=8,
        target_modules=("down_proj", "v_proj", "q_proj", "gate_proj", "up_proj"),
    )
    assert resolve_lora_targets(spec) == (
        "qkv_proj",
        "gate_up_proj",
        "down_proj",
    )
    unrelated = LoRAAdapterSpec(
        r=8,
        lora_alpha=8,
        target_modules=("embed_tokens",),
    )
    assert resolve_lora_targets(unrelated) == ()


def test_inference_request_carries_lora_reference() -> None:
    lora = LoRARequest(lora_name="sql", lora_int_id=1)
    request = InferenceRequest(
        request_id="r1",
        model="megalm",
        input_token_ids=(1, 2, 3),
        max_new_tokens=4,
        lora_request=lora,
    )
    assert request.lora_request is lora
    with pytest.raises(RequestContractError):
        InferenceRequest(
            request_id="r2",
            model="megalm",
            input_token_ids=(1,),
            max_new_tokens=1,
            lora_request="sql",  # type: ignore[arg-type]
        )
