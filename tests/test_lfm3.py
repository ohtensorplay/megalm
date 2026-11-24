import json
import math
from pathlib import Path

import pytest
import tensorplay

from megalm.adapters.lfm3 import lfm3_model_spec_from_config
from megalm.contracts.model import AttentionKind, StateKind
from megalm.models.lfm3.config import Lfm3Config
from megalm.models.lfm3.model import Lfm3RotaryCache


def _config(**overrides: object) -> dict:
    payload = {
        "model_type": "lfm3",
        "architectures": ["Lfm3ForCausalLM"],
        "vocab_size": 1024,
        "hidden_size": 64,
        "intermediate_size": 128,
        "num_hidden_layers": 6,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "head_dim": 16,
        "max_position_embeddings": 128,
        "norm_eps": 1e-5,
        "conv_L_cache": 7,
        "conv_bias": False,
        "use_sandwich_norm": True,
        "sliding_window": 32,
        "layer_types": [
            "conv",
            "full_attention",
            "conv",
            "sliding_attention",
            "conv",
            "full_attention",
        ],
        "rope_parameters": {
            "full_attention": {
                "rope_type": "default",
                "rope_theta": 1000000.0,
                "partial_rotary_factor": 0.5,
            },
            "sliding_attention": {
                "rope_type": "default",
                "rope_theta": 10000.0,
            },
        },
        "bos_token_id": 1,
        "eos_token_id": 2,
        "pad_token_id": 3,
        "tie_word_embeddings": True,
    }
    payload.update(overrides)
    return payload


def test_lfm3_spec_declares_hybrid_layers() -> None:
    spec = lfm3_model_spec_from_config(_config())

    assert spec.architecture == "Lfm3ForCausalLM"
    assert spec.model_type == "lfm3"
    assert spec.num_layers == 6
    assert [layer.attention for layer in spec.layers] == [
        AttentionKind.GATED_SHORT_CONV,
        AttentionKind.FULL,
        AttentionKind.GATED_SHORT_CONV,
        AttentionKind.FULL,
        AttentionKind.GATED_SHORT_CONV,
        AttentionKind.FULL,
    ]
    assert spec.layers[3].sliding_window == 32
    assert spec.layers[1].sliding_window is None
    assert spec.position.partial_rotary_factor == 0.5
    assert spec.tie_word_embeddings is True
    assert spec.parallel.supported_tp_degrees == (1,)
    kv = next(c for c in spec.state_schema if c.kind is StateKind.PAGED_KV)
    assert kv.layer_indices == (1, 3, 5)


def test_lfm3_native_config_from_path(tmp_path: Path) -> None:
    (tmp_path / "config.json").write_text(
        json.dumps(_config()), encoding="utf-8"
    )

    config = Lfm3Config.from_path(tmp_path)

    assert config.head_dim == 16
    assert config.rope_theta_full == 1e6
    assert config.rope_theta_sliding == 1e4
    assert config.partial_rotary_factor == 0.5
    assert config.conv_layers == (0, 2, 4)
    assert config.attention_layers == (1, 3, 5)


def test_lfm3_full_rotary_zero_tail() -> None:
    config = Lfm3Config.from_dict(_config())
    cache = Lfm3RotaryCache(config, kind="full_attention", device=tensorplay.device("cpu"))

    assert tuple(cache.cos.shape) == (128, 8)
    theta = config.rope_theta_full
    head_dim = config.head_dim
    angles = int(0.5 * head_dim) // 2
    positions = (0, 1, 7, 31)
    for position in positions:
        expected = [
            math.cos(position / (theta ** (2 * index / head_dim)))
            for index in range(angles)
        ]
        actual = cache.cos[position].tolist()
        assert actual[:angles] == pytest.approx(expected, abs=1e-4)
        assert actual[angles:] == pytest.approx([1.0] * (8 - angles), abs=1e-4)
        assert cache.sin[position].tolist()[angles:] == pytest.approx(
            [0.0] * (8 - angles), abs=1e-4
        )


def test_lfm3_conv_state_advances_like_reference() -> None:
    """The width-7 window must shift by one token after each decode step."""

    config = Lfm3Config.from_dict(_config())
    hidden = config.hidden_size
    kernel = config.conv_L_cache
    weights = tensorplay.arange(1, kernel + 1, dtype=tensorplay.float32).reshape(
        1, kernel
    )
    state = tensorplay.zeros((1, hidden, kernel))
    token = tensorplay.ones((1, hidden)) * 2

    window = tensorplay.cat((state, token.unsqueeze(-1)), dim=-1)
    values = window[..., 1:]
    result = (values * weights).sum(dim=-1)
    assert result[0, 0].item() == pytest.approx(2 * kernel)

    new_state = values
    state.copy_(new_state)
    assert state[0, 0, -1].item() == pytest.approx(2)
    assert state[0, 0, 0].item() == pytest.approx(0)


def test_lfm3_sliding_rotary_rotates_all_channels() -> None:
    config = Lfm3Config.from_dict(_config())
    cache = Lfm3RotaryCache(
        config, kind="sliding_attention", device=tensorplay.device("cpu")
    )

    assert tuple(cache.cos.shape) == (128, 8)
    position = 5
    expected = [
        math.cos(position / (1e4 ** (2 * index / config.head_dim)))
        for index in range(8)
    ]
    assert cache.cos[position].tolist() == pytest.approx(expected, abs=1e-4)
