from __future__ import annotations

import json
from pathlib import Path

import pytest

from megalm.adapters.oxr import (
    load_oxr_model_spec,
    oxr_model_spec_from_config,
)
from megalm.contracts.model import ModelContractError, PositionKind, StateKind
from megalm.models.oxr.config import OXRConfig
from megalm.runtime.tokenizer import ModelTokenizer

OXR_CONFIG = Path("/home/bluemoon/projects/OXR/models/OXR-1.0")


def _config() -> dict:
    return json.loads((OXR_CONFIG / "config.json").read_text(encoding="utf-8"))


def test_oxr_model_spec_facts():
    spec = load_oxr_model_spec(OXR_CONFIG)
    assert spec.model_type == "oxr"
    assert spec.architecture == "OXR"
    assert spec.num_layers == 28
    assert spec.hidden_size == 1024
    assert spec.vocab_size == 151936
    assert spec.position.kind is PositionKind.ROPE
    assert spec.position.max_positions == 32768
    assert spec.layers[0].num_attention_heads == 16
    assert spec.layers[0].num_key_value_heads == 8
    assert spec.layers[0].head_dim == 128
    assert spec.layers[0].intermediate_size == 3072
    assert spec.qk_norm is True
    assert spec.tie_word_embeddings is True
    assert spec.vision is not None
    assert spec.vision.processor_type == "oxr"
    assert spec.vision.patch_size == 16
    assert spec.vision.merge_size == 2
    assert spec.vision.depth == 27
    assert StateKind.VISION_EMBEDDING in {
        component.kind for component in spec.state_schema
    }


def test_oxr_config_parses_nested_llm_and_vit():
    config = OXRConfig.from_path(OXR_CONFIG)
    assert config.llm_config.num_hidden_layers == 28
    assert config.llm_config.head_dim == 128
    assert config.hidden_size == config.llm_config.hidden_size
    assert config.max_position_embeddings == 32768
    assert config.visual_vocab_size == 65536
    assert config.vision_config.spatial_merge_size == 2
    assert config.vision_config.num_position_embeddings == 1024
    assert config.image_start_token_id == 151931
    assert config.image_end_token_id == 151932


def test_oxr_model_spec_rejects_partial_vit_attention():
    payload = _config()
    payload["vit_config"]["fullatt_block_indexes"] = "0|1"
    with pytest.raises(ModelContractError, match="full-attention"):
        oxr_model_spec_from_config(payload)


def test_oxr_model_spec_rejects_foreign_model_type():
    payload = _config()
    payload["model_type"] = "qwen3"
    with pytest.raises(ModelContractError):
        oxr_model_spec_from_config(payload)


@pytest.mark.skipif(not OXR_CONFIG.is_dir(), reason="OXR checkpoint not present")
def test_oxr_tokenizer_expands_visual_indicators():
    tokenizer = ModelTokenizer(OXR_CONFIG)
    encoding = tokenizer.encode_multimodal_chat(
        [
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": "data:image/png"}},
                    {"type": "text", "text": "transcribe"},
                ],
            }
        ],
        image_feature_counts=(9,),
    )
    assert encoding.token_type_ids.count(1) == 9
    assert encoding.input_ids.count(151931) == 1
    assert encoding.input_ids.count(151932) == 1
    assert encoding.input_ids.count(151655) == 9
