"""DiffusionGemma config, sampler and stopping-rule unit tests."""

from __future__ import annotations

import json

import pytest
import tensorplay

from megalm.models.diffusiongemma.config import DiffusionGemmaConfig
from megalm.models.diffusiongemma.sampler import (
    DiffusionSamplingConfig,
    EntropyBoundSampler,
    StableAndConfidentStopping,
    entropy_bound_mask,
    temperature_for_step,
    token_entropy,
)


def _text_payload() -> dict:
    return {
        "model_type": "gemma4_text",
        "vocab_size": 64,
        "hidden_size": 32,
        "intermediate_size": 64,
        "moe_intermediate_size": 32,
        "num_hidden_layers": 2,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "head_dim": 8,
        "global_head_dim": 8,
        "max_position_embeddings": 128,
        "layer_types": ["sliding_attention", "full_attention"],
        "sliding_window": 8,
        "tie_word_embeddings": True,
        "num_experts": 4,
        "top_k_experts": 2,
        "rms_norm_eps": 1e-6,
        "hidden_activation": "gelu_pytorch_tanh",
        "rope_parameters": {
            "sliding_attention": {
                "rope_theta": 10000.0,
                "rope_type": "default",
            },
            "full_attention": {
                "rope_theta": 1000000.0,
                "rope_type": "proportional",
                "partial_rotary_factor": 0.25,
            },
        },
    }


def _payload(**overrides: object) -> dict:
    payload = {
        "model_type": "diffusion_gemma",
        "canvas_length": 16,
        "eos_token_id": [1, 2],
        "pad_token_id": 0,
        "text_config": _text_payload(),
    }
    payload.update(overrides)
    return payload


def _quant_payload(ignore: list[str]) -> dict:
    return {
        "quant_method": "compressed-tensors",
        "format": "pack-quantized",
        "config_groups": {
            "group_0": {
                "weights": {
                    "num_bits": 4,
                    "group_size": 32,
                    "strategy": "group",
                    "symmetric": True,
                    "type": "int",
                }
            }
        },
        "ignore": ignore,
    }


def test_config_resolves_diffusion_knobs() -> None:
    config = DiffusionGemmaConfig.from_dict(_payload())
    assert config.canvas_length == 16
    assert config.eos_token_ids == (1, 2)
    assert config.pad_token_id == 0
    assert config.text.hidden_size == 32
    assert config.text.attention_k_eq_v is True
    assert config.quant is None


def test_quantized_module_selection_uses_ignore_prefixes() -> None:
    payload = _payload(
        quantization_config=_quant_payload(
            ["model.decoder.layers.0.mlp", "model.decoder.layers.0.router"]
        )
    )
    config = DiffusionGemmaConfig.from_dict(payload)
    assert config.quant is not None
    assert config.quant.group_size == 32
    assert config.module_is_quantized("layers.0.self_attn.q_proj")
    assert config.module_is_quantized("layers.1.experts.3.gate_proj")
    assert not config.module_is_quantized("layers.0.mlp.gate_proj")
    assert not config.module_is_quantized("layers.0.router.proj")


@pytest.mark.parametrize("weights", [
    {"num_bits": 8, "group_size": 32},
    {"num_bits": 4, "group_size": 32, "symmetric": False},
])
def test_config_rejects_bad_quantization(weights: dict) -> None:
    payload = _payload(
        quantization_config={
            "quant_method": "compressed-tensors",
            "format": "pack-quantized",
            "config_groups": {"g": {"weights": weights}},
        }
    )
    with pytest.raises(Exception):
        DiffusionGemmaConfig.from_dict(payload)


def test_attention_k_eq_v_is_inferred_from_the_checkpoint(tmp_path) -> None:
    (tmp_path / "config.json").write_text(json.dumps(_payload()), encoding="utf-8")
    # Full layer 1 has no v_proj: the value is the key projection.
    (tmp_path / "model.safetensors.index.json").write_text(
        json.dumps(
            {
                "weight_map": {
                    "model.decoder.layers.1.self_attn.q_proj.weight": "a",
                    "model.decoder.layers.1.self_attn.k_proj.weight": "a",
                    "model.decoder.layers.1.self_attn.o_proj.weight": "a",
                }
            }
        ),
        encoding="utf-8",
    )
    assert DiffusionGemmaConfig.from_path(tmp_path).text.attention_k_eq_v is True

    (tmp_path / "model.safetensors.index.json").write_text(
        json.dumps(
            {
                "weight_map": {
                    "model.decoder.layers.1.self_attn.q_proj.weight": "a",
                    "model.decoder.layers.1.self_attn.k_proj.weight": "a",
                    "model.decoder.layers.1.self_attn.v_proj.weight": "a",
                    "model.decoder.layers.1.self_attn.o_proj.weight": "a",
                }
            }
        ),
        encoding="utf-8",
    )
    assert DiffusionGemmaConfig.from_path(tmp_path).text.attention_k_eq_v is False


def test_temperature_schedule_counts_down() -> None:
    assert temperature_for_step(4, 4, 0.4, 0.8) == pytest.approx(0.8)
    assert temperature_for_step(1, 4, 0.4, 0.8) == pytest.approx(0.5)


def test_token_entropy_matches_closed_form() -> None:
    uniform = tensorplay.zeros((1, 3, 8), dtype=tensorplay.float32)
    entropy = token_entropy(uniform)
    assert entropy.shape == (1, 3)
    assert tensorplay.allclose(
        entropy, tensorplay.full((1, 3), 2.0794415, dtype=tensorplay.float32), atol=1e-5
    )


def test_entropy_bound_mask_uses_cumulative_bound() -> None:
    logits = tensorplay.zeros((1, 4, 8), dtype=tensorplay.float32)
    entropy = tensorplay.tensor(
        [[0.05, 0.2, 0.5, 1.0]], dtype=tensorplay.float32
    )
    mask = entropy_bound_mask(logits, 0.1, entropy=entropy)
    assert mask.tolist() == [[True, True, False, False]]


def test_sampler_keeps_accepted_and_renoises_the_rest() -> None:
    config = DiffusionSamplingConfig(
        entropy_bound=0.1, eos_token_ids=(1,), pad_token_id=0
    )
    sampler = EntropyBoundSampler(config, vocab_size=8)
    current = tensorplay.tensor([[10, 11, 12]], dtype=tensorplay.int64)
    denoiser = tensorplay.tensor([[20, 21, 22]], dtype=tensorplay.int64)
    logits = tensorplay.zeros((1, 3, 8), dtype=tensorplay.float32)
    # The lowest-entropy token is always accepted; the second is accepted
    # only when the smallest entropy itself is already inside the bound.
    entropy = tensorplay.tensor([[5.0, 0.2, 5.0]], dtype=tensorplay.float32)
    mask, canvas = sampler.accept_and_renoise(
        current, denoiser, logits, entropy=entropy
    )
    assert mask.tolist() == [[False, True, False]]
    assert 0 <= int(canvas[0, 0]) < 8
    assert int(canvas[0, 1]) == 21
    assert 0 <= int(canvas[0, 2]) < 8

    # Two near-zero entropies plus a high one: the bound (which excludes the
    # largest member) accepts both, and adding the third token breaks it.
    entropy = tensorplay.tensor([[0.0, 5.0, 5.0]], dtype=tensorplay.float32)
    mask, _ = sampler.accept_and_renoise(
        current, denoiser, logits, entropy=entropy
    )
    assert mask.tolist() == [[True, True, False]]


def test_stopper_requires_stability_and_confidence() -> None:
    stopper = StableAndConfidentStopping(
        stability_threshold=1, confidence_threshold=0.5
    )
    argmax = tensorplay.tensor([[1, 2, 3]], dtype=tensorplay.int64)
    logits = tensorplay.zeros((1, 3, 8), dtype=tensorplay.float32)
    confident = tensorplay.zeros((1, 3), dtype=tensorplay.float32)
    assert stopper(argmax, logits, entropy=confident).tolist() == [False]
    assert stopper(argmax, logits, entropy=confident).tolist() == [True]

    stopper.reset()
    unsure = tensorplay.full((1, 3), 2.0, dtype=tensorplay.float32)
    assert stopper(argmax, logits, entropy=unsure).tolist() == [False]
    assert stopper(argmax, logits, entropy=unsure).tolist() == [False]


def test_sampling_config_rejects_invalid_schedule() -> None:
    config = DiffusionSamplingConfig(t_min=0.8, t_max=0.4)
    with pytest.raises(ValueError):
        config.validate(canvas_length=8)
