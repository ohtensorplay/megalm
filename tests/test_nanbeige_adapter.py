"""Contract tests for the Nanbeige4.2 looped-transformer adapter."""

from __future__ import annotations

import copy

import pytest

from megalm.adapters.nanbeige import (
    load_nanbeige_model_spec,
    nanbeige_model_spec_from_config,
)
from megalm.contracts.model import (
    AttentionKind,
    FeedForwardKind,
    ModelContractError,
    PositionKind,
)

# The released Nanbeige/Nanbeige4.2-3B config.json.
_RELEASE_CONFIG = {
    "architectures": ["NanbeigeForCausalLM"],
    "model_type": "nanbeige",
    "hidden_size": 3072,
    "intermediate_size": 10752,
    "num_hidden_layers": 22,
    "num_attention_heads": 48,
    "num_key_value_heads": 8,
    "head_dim": 128,
    "vocab_size": 166144,
    "max_position_embeddings": 262144,
    "rope_theta": 70000000.0,
    "rms_norm_eps": 1e-05,
    "num_loops": 2,
    "hidden_act": "silu",
    "tie_word_embeddings": False,
    "attention_bias": False,
    "rope_scaling": None,
    "loop_loss_weights": [],
    "skip_loop_final_norm": False,
}


def test_released_config_builds_looped_spec() -> None:
    spec = nanbeige_model_spec_from_config(copy.deepcopy(_RELEASE_CONFIG))
    assert spec.model_type == "nanbeige"
    assert spec.architecture == "NanbeigeForCausalLM"
    assert spec.hidden_size == 3072
    assert spec.vocab_size == 166144
    assert spec.num_layers == 22
    assert spec.position.kind is PositionKind.ROPE
    assert spec.position.rope_theta == 70_000_000.0
    assert spec.position.max_positions == 262_144
    assert all(layer.attention is AttentionKind.FULL for layer in spec.layers)
    assert all(layer.feed_forward is FeedForwardKind.DENSE for layer in spec.layers)

    loop = spec.loop
    assert loop is not None
    assert (loop.core_start, loop.core_end, loop.num_passes) == (0, 22, 2)
    # Full-stack replay: 22 physical layers become 44 KV cache visits.
    kv_layers = spec.num_layers + (loop.num_passes - 1) * (
        loop.core_end - loop.core_start
    )
    assert kv_layers == 44


def test_config_file_loading(tmp_path) -> None:
    import json

    (tmp_path / "config.json").write_text(
        json.dumps(_RELEASE_CONFIG), encoding="utf-8"
    )
    spec = load_nanbeige_model_spec(tmp_path)
    assert spec.loop is not None and spec.loop.num_passes == 2


@pytest.mark.parametrize(
    "key,value",
    [
        ("enable_mhc", True),
        ("enable_hyper_connection", True),
        ("enable_depth_attention", True),
        ("enable_double_loop_split", True),
        ("loop_share_kv", True),
        ("qk_layernorm", True),
        ("attention_bias", True),
        ("mlp_bias", True),
        ("emb_neighbor_num", 3),
        ("rope_scaling", {"type": "linear", "factor": 2.0}),
        ("loop_loss_weights", [1.0]),
        ("pretraining_tp", 2),
        ("hidden_act", "gelu"),
    ],
)
def test_unsupported_variants_are_rejected(key, value) -> None:
    config = copy.deepcopy(_RELEASE_CONFIG)
    config[key] = value
    with pytest.raises(ModelContractError):
        nanbeige_model_spec_from_config(config)


def test_invalid_model_type_is_rejected() -> None:
    config = copy.deepcopy(_RELEASE_CONFIG)
    config["model_type"] = "llama"
    with pytest.raises(ModelContractError):
        nanbeige_model_spec_from_config(config)


def test_native_config_parses_loop_schedule() -> None:
    pytest.importorskip("tensorplay")
    from megalm.models.nanbeige.config import NanbeigeConfig

    config = NanbeigeConfig.from_path("model/nanbeige42")
    assert config.model_type == "nanbeige"
    assert config.hidden_size == 3072
    assert config.head_dim == 128
    assert config.num_attention_heads == 48
    assert config.num_key_value_heads == 8
    assert config.num_hidden_layers == 22
    assert config.num_loops == 2
    assert config.skip_loop_final_norm is False
    assert config.rope_theta == 70_000_000.0
