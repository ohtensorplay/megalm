import copy
import json
from pathlib import Path

import pytest

from megalm.adapters.qwen3_5 import qwen3_5_model_spec_from_config
from megalm.contracts.model import (
    AttentionKind,
    FeedForwardKind,
    ModelContractError,
    StateKind,
)
from megalm.models.qwen3_5.config import (
    Qwen35Config,
    Qwen35ConfigError,
    Qwen35TextConfig,
)


def _config() -> dict:
    return {
        "model_type": "qwen3_5_moe",
        "architectures": ["Qwen3_5MoeForConditionalGeneration"],
        "image_token_id": 10,
        "video_token_id": 11,
        "vision_start_token_id": 12,
        "vision_end_token_id": 13,
        "text_config": {
            "model_type": "qwen3_5_moe_text",
            "vocab_size": 1000,
            "hidden_size": 64,
            "num_hidden_layers": 4,
            "num_attention_heads": 4,
            "num_key_value_heads": 2,
            "head_dim": 16,
            "max_position_embeddings": 1024,
            "layer_types": [
                "linear_attention",
                "full_attention",
                "linear_attention",
                "full_attention",
            ],
            "linear_conv_kernel_dim": 4,
            "linear_key_head_dim": 16,
            "linear_value_head_dim": 32,
            "linear_num_key_heads": 2,
            "linear_num_value_heads": 2,
            "mamba_ssm_dtype": "float32",
            "num_experts": 8,
            "num_experts_per_tok": 2,
            "moe_intermediate_size": 32,
            "shared_expert_intermediate_size": 16,
            "norm_topk_prob": True,
            "rope_parameters": {
                "mrope_section": [1, 1],
                "partial_rotary_factor": 0.25,
            },
        },
        "vision_config": {
            "depth": 2,
            "hidden_size": 16,
            "intermediate_size": 32,
            "num_heads": 4,
            "in_channels": 3,
            "patch_size": 2,
            "spatial_merge_size": 2,
            "temporal_patch_size": 2,
            "out_hidden_size": 64,
            "num_position_embeddings": 1024,
        },
    }


def test_qwen35_moe_spec_declares_hybrid_moe_layers() -> None:
    spec = qwen3_5_model_spec_from_config(_config())

    assert spec.model_type == "qwen3_5_moe"
    assert spec.architecture == "Qwen3_5MoeForConditionalGeneration"
    assert all(
        layer.feed_forward is FeedForwardKind.MOE for layer in spec.layers
    )
    assert all(layer.experts_per_token == 2 for layer in spec.layers)
    assert all(layer.num_experts == 8 for layer in spec.layers)
    assert tuple(
        layer.attention for layer in spec.layers
    ) == (
        AttentionKind.GATED_DELTA_NET,
        AttentionKind.FULL,
        AttentionKind.GATED_DELTA_NET,
        AttentionKind.FULL,
    )
    assert {
        component.kind for component in spec.state_schema
    } >= {
        StateKind.PAGED_KV,
        StateKind.GDN_CONV,
        StateKind.GDN_RECURRENT,
        StateKind.VISION_EMBEDDING,
    }
    assert 2 in spec.parallel.supported_tp_degrees


def test_qwen35_moe_config_loads_default_architecture(tmp_path: Path) -> None:
    payload = _config()
    payload.pop("architectures")
    (tmp_path / "config.json").write_text(
        json.dumps(payload), encoding="utf-8"
    )

    config = Qwen35Config.from_path(tmp_path)

    assert config.architecture == "Qwen3_5MoeForConditionalGeneration"
    assert config.text_config.is_moe
    assert config.text_config.num_experts_per_tok == 2
    assert config.text_config.shared_expert_intermediate_size == 16
    assert config.text_config.full_layer_indices == (1, 3)
    assert config.text_config.gdn_layer_indices == (0, 2)


def test_qwen35_moe_rejects_top_k_larger_than_expert_count() -> None:
    payload = _config()
    payload["text_config"]["num_experts_per_tok"] = 9

    with pytest.raises(ModelContractError, match="exceeds"):
        qwen3_5_model_spec_from_config(payload)

    with pytest.raises(Qwen35ConfigError, match="top-k"):
        Qwen35TextConfig.from_dict(payload["text_config"])


def test_qwen35_moe_allows_zero_width_shared_expert() -> None:
    payload = copy.deepcopy(_config())
    payload["text_config"]["shared_expert_intermediate_size"] = 0

    spec = qwen3_5_model_spec_from_config(payload)
    config = Qwen35TextConfig.from_dict(payload["text_config"])

    assert spec.model_type == "qwen3_5_moe"
    assert config.shared_expert_intermediate_size == 0
