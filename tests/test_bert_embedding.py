from __future__ import annotations

import json
import math
from dataclasses import replace
from pathlib import Path

import pytest
import tensorplay

from megalm.models.bert_embedding import (
    BertEmbeddingConfig,
    BertEmbeddingModel,
    PoolingMode,
    resolve_pooling_config,
)
from megalm.runtime.embedding_engine import EmbeddingEngineConfig, EmbeddingEngineError


def _config(pooling: PoolingMode = PoolingMode.CLS) -> BertEmbeddingConfig:
    return BertEmbeddingConfig(
        model_type="bert",
        architecture="BertModel",
        vocab_size=64,
        hidden_size=16,
        num_hidden_layers=2,
        num_attention_heads=4,
        intermediate_size=32,
        max_position_embeddings=32,
        layer_norm_eps=1e-12,
        hidden_act="gelu",
        pad_token_id=0,
        type_vocab_size=2,
        pooling_mode=pooling,
        normalize=True,
        use_pooler_dense=False,
        embedding_size=16,
    )


def test_forward_is_batched_masked_and_normalized() -> None:
    model = BertEmbeddingModel(_config(), device="cpu", dtype=tensorplay.float32)
    ids = tensorplay.tensor([[1, 2, 3, 0], [4, 5, 6, 7]])
    mask = tensorplay.tensor([[1, 1, 1, 0], [1, 1, 1, 1]])
    output = model(ids, mask).tolist()
    assert len(output) == 2
    assert all(len(row) == 16 for row in output)
    for row in output:
        assert math.isclose(math.sqrt(sum(value * value for value in row)), 1.0, abs_tol=1e-5)


def test_mean_pooling_ignores_padding() -> None:
    config = replace(_config(PoolingMode.MEAN), normalize=False)
    model = BertEmbeddingModel(config, device="cpu", dtype=tensorplay.float32)
    hidden = tensorplay.tensor([[[2.0, 4.0], [6.0, 8.0], [100.0, 100.0]]])
    mask = tensorplay.tensor([[1, 1, 0]])
    pooled = model._pool(hidden, mask).tolist()
    assert pooled == [[4.0, 6.0]]


def test_sentence_transformers_pooling_config(tmp_path: Path) -> None:
    (tmp_path / "modules.json").write_text(
        json.dumps(
            [
                {"type": "sentence_transformers.models.Transformer", "path": ""},
                {"type": "sentence_transformers.models.Pooling", "path": "1_Pooling"},
                {"type": "sentence_transformers.models.Normalize", "path": "2_Normalize"},
            ]
        )
    )
    pooling_dir = tmp_path / "1_Pooling"
    pooling_dir.mkdir()
    (pooling_dir / "config.json").write_text(
        json.dumps(
            {
                "pooling_mode_cls_token": False,
                "pooling_mode_mean_tokens": True,
            }
        )
    )
    mode, normalize = resolve_pooling_config(tmp_path)
    assert mode is PoolingMode.MEAN
    assert normalize is True


def test_embedding_engine_config_rejects_invalid_sizes() -> None:
    with pytest.raises(EmbeddingEngineError):
        EmbeddingEngineConfig(model_path="unused", max_batch_size=0)
    with pytest.raises(EmbeddingEngineError):
        EmbeddingEngineConfig(model_path="unused", dtype="float64")
