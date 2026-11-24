"""G4 admission gold tests for the OLMo-2 dense family.

OLMo-2 is admitted purely by declarations: a flat Llama-style RoPE/GQA geometry
with standard HF weight names and QK-norm. No model block, media processor, or
stage runtime is copied, and Scheduler / Cache / Gateway need no changes.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path

import tensorplay

from megalm.adapters.generic_causal_lm import (
    generic_causal_lm_spec_from_config,
)
from megalm.adapters.registry import load_model_spec
from megalm.contracts.lora import GENERIC_LORA_MODEL_TYPES
from megalm.contracts.model import (
    AttentionKind,
    FeedForwardKind,
    ModelContractError,
    PositionKind,
    StateKind,
)
from megalm.kernels.ops import KernelPolicy
from megalm.models.generic_causal_lm.config import GenericCausalLMConfig
from megalm.models.generic_causal_lm.model import GenericCausalLM
from megalm.runtime.batch import BatchPhase, ExecutionBatch
from megalm.runtime.cache import PagedKVCache
from tensorplay.serialization import save

def test_olmo2_is_declared_for_generic_lora() -> None:
    # OLMo-2 uses the same standard q/k/v + gate/up HF weight names as the
    # other generic dense families, so the packed LoRA bank binder applies.
    assert "olmo2" in GENERIC_LORA_MODEL_TYPES


OLMO2_CONFIG = {
    "model_type": "olmo2",
    "architectures": ["OLMo2ForCausalLM"],
    "hidden_size": 64,
    "num_hidden_layers": 2,
    "num_attention_heads": 4,
    "num_key_value_heads": 2,
    "head_dim": 16,
    "vocab_size": 128,
    "max_position_embeddings": 512,
    "intermediate_size": 32,
    "rope_theta": 500000,
    "rms_norm_eps": 1e-6,
    "qk_norm": True,
    "tie_word_embeddings": False,
}


def _tiny_olmo2() -> GenericCausalLM:
    config = GenericCausalLMConfig(
        model_type="olmo2",
        vocab_size=32,
        hidden_size=16,
        intermediate_size=16,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=2,
        head_dim=8,
        max_position_embeddings=64,
        rms_norm_eps=1e-6,
        rope_theta=10_000,
        tie_word_embeddings=False,
        qkv_bias=False,
        qk_norm=True,
        num_experts=0,
        experts_per_token=0,
        moe_intermediate_size=0,
        norm_topk_prob=False,
        decoder_sparse_step=1,
        mlp_only_layers=(),
    )
    return GenericCausalLM(
        config,
        device="cpu",
        dtype=tensorplay.float32,
        kernel_policy=KernelPolicy(use_triton=False),
    )


def test_olmo2_config_declares_the_generic_dense_contract() -> None:
    spec = generic_causal_lm_spec_from_config(OLMO2_CONFIG)

    assert spec.model_type == "olmo2"
    assert spec.architecture == "OLMo2ForCausalLM"
    assert len(spec.layers) == 2
    assert all(layer.attention is AttentionKind.FULL for layer in spec.layers)
    assert all(
        layer.feed_forward is FeedForwardKind.DENSE for layer in spec.layers
    )
    assert all(
        layer.num_attention_heads == 4
        and layer.num_key_value_heads == 2
        and layer.head_dim == 16
        for layer in spec.layers
    )
    assert spec.position.kind is PositionKind.ROPE
    assert spec.position.rope_theta == 500000.0
    assert tuple(component.kind for component in spec.state_schema) == (
        StateKind.PAGED_KV,
    )
    assert spec.state_schema[0].layer_indices == (0, 1)
    assert spec.qk_norm is True
    assert spec.qkv_bias is False
    assert spec.tie_word_embeddings is False
    assert 1 in spec.parallel.supported_tp_degrees


def test_olmo2_config_file_loads_through_the_adapter_registry(
    tmp_path: Path,
) -> None:
    (tmp_path / "config.json").write_text(
        json.dumps(OLMO2_CONFIG), encoding="utf-8"
    )
    spec = load_model_spec(tmp_path)
    assert spec.model_type == "olmo2"
    assert spec.architecture == "OLMo2ForCausalLM"


def test_olmo2_rejects_non_causal_config() -> None:
    config = copy.deepcopy(OLMO2_CONFIG)
    config["is_causal"] = False
    try:
        generic_causal_lm_spec_from_config(config)
    except ModelContractError:
        pass
    else:
        raise AssertionError("non-causal OLMo-2 config must be rejected")


def test_olmo2_kv_layer_layout_matches_the_spec() -> None:
    spec = generic_causal_lm_spec_from_config(OLMO2_CONFIG)
    kv_layers = [
        index
        for index, layer in enumerate(spec.layers)
        if layer.attention is AttentionKind.FULL
    ]
    cache = PagedKVCache(
        num_layers=len(kv_layers),
        num_pages=2,
        page_size=4,
        num_kv_heads=spec.layers[0].num_key_value_heads,
        head_dim=spec.layers[0].head_dim,
        dtype=tensorplay.float32,
        device=tensorplay.device("cpu"),
    )
    for layer_index in range(len(kv_layers)):
        key, value = cache.layer(layer_index)
        assert tuple(key.shape) == (2, 4, 2, 16)
        assert key.shape == value.shape


def test_olmo2_synthetic_checkpoint_and_reference_logits(
    tmp_path: Path,
) -> None:
    """Standard HF names load, and a zero-residual model matches reference logits."""

    model = _tiny_olmo2()
    sources: dict[str, tensorplay.Tensor] = {}
    for name, parameter in model.named_parameters():
        if name == "model.embed_tokens.weight":
            values = tensorplay.arange(
                parameter.numel(), dtype=tensorplay.float32
            ).reshape(parameter.shape)
            sources[name] = values * 0.05
        elif name == "lm_head_weight":
            values = tensorplay.arange(
                parameter.numel(), dtype=tensorplay.float32
            ).reshape(parameter.shape)
            sources[name] = (
                (values % 7).to(tensorplay.float32) - 3.0
            ) * 0.1
        elif name == "model.norm.weight":
            sources[name] = tensorplay.ones_like(parameter)
        else:
            sources[name] = tensorplay.zeros_like(parameter)

    # q/k/v and gate/up are packed in the native model; the loader accepts the
    # standard separate HF names and assembles them.
    attention = model.model.layers[0].self_attn
    qkv = sources.pop("model.layers.0.self_attn.qkv_proj.weight")
    query_width = attention.query_width
    kv_width = attention.kv_width
    sources["model.layers.0.self_attn.q_proj.weight"] = qkv[:query_width]
    sources["model.layers.0.self_attn.k_proj.weight"] = qkv[
        query_width : query_width + kv_width
    ]
    sources["model.layers.0.self_attn.v_proj.weight"] = qkv[
        query_width + kv_width :
    ]
    gate_up = sources.pop("model.layers.0.mlp.gate_up_weight")
    intermediate = model.config.intermediate_size
    gate, up = gate_up.split((intermediate, intermediate), dim=0)
    sources["model.layers.0.mlp.gate_proj.weight"] = gate
    sources["model.layers.0.mlp.up_proj.weight"] = up
    sources["model.layers.0.mlp.down_proj.weight"] = sources.pop(
        "model.layers.0.mlp.down_proj.weight"
    )
    sources["lm_head.weight"] = sources.pop("lm_head_weight")

    save(sources, str(tmp_path / "model.safetensors"))
    report = model.load_weights(tmp_path)
    assert report.tensor_count == len(sources)

    positions = tensorplay.tensor([0, 1], dtype=tensorplay.int64)
    batch = ExecutionBatch(
        input_ids=tensorplay.tensor([3, 5], dtype=tensorplay.int64),
        positions=positions,
        slot_mapping=positions,
        block_tables=tensorplay.tensor([[0]], dtype=tensorplay.int32),
        sequence_lengths=tensorplay.tensor([2], dtype=tensorplay.int32),
        token_active=tensorplay.ones(2, dtype=tensorplay.bool),
        query_start_loc=(0, 2),
        query_lengths=(2,),
        context_lengths=(0,),
        phases=(BatchPhase.PREFILL,),
        sample_indices=tensorplay.tensor([1], dtype=tensorplay.int64),
        page_size=4,
    )
    cache = PagedKVCache(
        num_layers=1,
        num_pages=1,
        page_size=4,
        num_kv_heads=2,
        head_dim=8,
        dtype=tensorplay.float32,
        device=tensorplay.device("cpu"),
    )
    logits = model(batch, cache)
    assert tuple(logits.shape) == (1, 32)

    # With every attention/MLP projection zero and the final norm weight one,
    # the only path to the logits is RMSNorm(embed[token]) @ lm_head.T.
    embedding = sources["model.embed_tokens.weight"][5]
    rms = tensorplay.sqrt((embedding * embedding).mean() + 1e-6)
    normed = embedding / rms
    expected = tensorplay.matmul(
        normed, sources["lm_head.weight"].transpose(0, 1)
    )
    assert tensorplay.allclose(logits[0], expected, atol=1e-4)
