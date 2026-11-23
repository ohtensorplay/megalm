import json
from pathlib import Path

import pytest
import tensorplay
from tensorplay.serialization import save

from megalm.adapters.generic_causal_lm import (
    generic_causal_lm_spec_from_config,
)
from megalm.kernels.ops import KernelPolicy
from megalm.contracts.model import (
    FeedForwardKind,
    ModelContractError,
    StateKind,
)
from megalm.models.generic_causal_lm.config import GenericCausalLMConfig
from megalm.models.generic_causal_lm.model import (
    GenericCausalLM,
    GenericRotaryCache,
)

_DENSE_TYPES = (
    "llama",
    "mistral",
    "qwen2",
    "yi",
    "deepseek",
    "phi3",
    "internlm2",
    "internlm3",
    "seed_oss",
    "granite",
    "ernie4_5",
)
_MOE_TYPES = ("qwen2_moe", "mixtral")


def _config(model_type: str, **overrides: object) -> dict:
    payload = {
        "model_type": model_type,
        "hidden_size": 64,
        "num_hidden_layers": 2,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "head_dim": 16,
        "vocab_size": 1000,
        "max_position_embeddings": 512,
        "intermediate_size": 32,
        "rope_theta": 10000,
        "rms_norm_eps": 1e-6,
    }
    if model_type == "qwen2_moe":
        payload.update(
            num_experts=8,
            num_experts_per_tok=2,
            moe_intermediate_size=32,
        )
    elif model_type == "mixtral":
        payload.update(num_local_experts=8, num_experts_per_tok=2)
    payload.update(overrides)
    return payload


@pytest.mark.parametrize("model_type", _DENSE_TYPES)
def test_dense_families_are_sealed_as_full_attention_models(
    model_type: str,
) -> None:
    spec = generic_causal_lm_spec_from_config(_config(model_type))

    assert spec.model_type == model_type
    assert all(
        layer.feed_forward is FeedForwardKind.DENSE for layer in spec.layers
    )
    assert tuple(component.kind for component in spec.state_schema) == (
        StateKind.PAGED_KV,
    )
    assert spec.qkv_bias is (model_type in {"qwen2", "seed_oss"})
    assert spec.qk_norm is False


@pytest.mark.parametrize("model_type", _MOE_TYPES)
def test_moe_families_are_sealed_as_top2_models(model_type: str) -> None:
    spec = generic_causal_lm_spec_from_config(_config(model_type))

    assert spec.model_type == model_type
    assert all(
        layer.feed_forward is FeedForwardKind.MOE for layer in spec.layers
    )
    assert all(layer.num_experts == 8 for layer in spec.layers)
    assert all(layer.experts_per_token == 2 for layer in spec.layers)
    assert spec.qkv_bias is (model_type == "qwen2_moe")


def test_qk_norm_and_bias_are_recorded_in_the_contract() -> None:
    spec = generic_causal_lm_spec_from_config(
        _config("qwen2", qk_norm=True)
    )

    assert spec.qk_norm is True
    assert spec.qkv_bias is True
    assert spec.digest != generic_causal_lm_spec_from_config(
        _config("qwen2")
    ).digest


def test_sliding_window_limits_the_advertised_context() -> None:
    spec = generic_causal_lm_spec_from_config(
        _config("mistral", sliding_window=128)
    )

    assert spec.position.max_positions == 128


def test_qwen2_moe_sparse_step_keeps_dense_layers() -> None:
    spec = generic_causal_lm_spec_from_config(
        _config(
            "qwen2_moe",
            decoder_sparse_step=2,
            mlp_only_layers=[1],
        )
    )

    assert tuple(layer.feed_forward for layer in spec.layers) == (
        FeedForwardKind.DENSE,
        FeedForwardKind.DENSE,
    )


def test_generic_config_loads_mixtral_expert_names(tmp_path: Path) -> None:
    (tmp_path / "config.json").write_text(
        json.dumps(_config("mixtral")),
        encoding="utf-8",
    )

    config = GenericCausalLMConfig.from_path(tmp_path)

    assert config.model_type == "mixtral"
    assert config.expert_projection_names == ("w1", "w3", "w2")
    assert config.num_experts == 8
    assert config.experts_per_token == 2


def test_unsupported_family_is_rejected() -> None:
    with pytest.raises(ModelContractError, match="does not support"):
        generic_causal_lm_spec_from_config(_config("gemma"))


def test_seed_oss_defaults_to_one_million_rope_theta() -> None:
    payload = _config("seed_oss")
    payload.pop("rope_theta")
    spec = generic_causal_lm_spec_from_config(payload)

    assert spec.position.rope_theta == 1_000_000.0


def test_ernie45_declares_partial_interleaved_rope() -> None:
    spec = generic_causal_lm_spec_from_config(_config("ernie4_5"))

    assert spec.position.partial_rotary_factor == 0.5
    assert spec.position.rotary_style == "interleaved"
    assert spec.qkv_bias is False


def test_ernie45_rejects_qk_norm() -> None:
    with pytest.raises(ModelContractError, match="QK-Norm"):
        generic_causal_lm_spec_from_config(
            _config("ernie4_5", qk_norm=True)
        )


def test_generic_rotary_cache_builds_interleaved_permutation() -> None:
    config = GenericCausalLMConfig(
        model_type="ernie4_5",
        vocab_size=32,
        hidden_size=16,
        intermediate_size=16,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=8,
        max_position_embeddings=64,
        rms_norm_eps=1e-6,
        rope_theta=10_000,
        tie_word_embeddings=False,
        qkv_bias=False,
        qk_norm=False,
        num_experts=0,
        experts_per_token=0,
        moe_intermediate_size=0,
        norm_topk_prob=False,
        decoder_sparse_step=1,
        mlp_only_layers=(),
        partial_rotary_factor=0.5,
        rotary_style="interleaved",
    )

    cache = GenericRotaryCache(config, device=tensorplay.device("cpu"))

    assert tuple(cache.cos.shape) == (64, 4)
    assert tuple(cache.sin.shape) == (64, 4)
    assert tuple(cache.permutation.shape) == (8,)


def test_granite_scaling_facts_are_recorded() -> None:
    spec = generic_causal_lm_spec_from_config(
        _config(
            "granite",
            embedding_multiplier=2.0,
            residual_multiplier=0.5,
            attention_multiplier=0.25,
            logit_scale=2.0,
            logits_scaling=4.0,
        )
    )

    assert spec.embedding_multiplier == 2.0
    assert spec.residual_multiplier == 0.5
    assert spec.attention_multiplier == 0.25
    assert spec.logit_scale == 0.5


def test_granite_rejects_unsupported_attention_scale() -> None:
    with pytest.raises(ModelContractError, match="attention_multiplier"):
        generic_causal_lm_spec_from_config(
            _config("granite", attention_multiplier=0.5)
        )


def test_granite_rejects_bias_variants() -> None:
    with pytest.raises(ModelContractError, match="bias variants"):
        generic_causal_lm_spec_from_config(
            _config("granite", mlp_bias=True)
        )


def test_non_causal_generic_config_is_rejected() -> None:
    with pytest.raises(ModelContractError, match="causal"):
        generic_causal_lm_spec_from_config(
            _config("seed_oss", is_causal=False)
        )


def test_non_default_rope_scaling_is_rejected() -> None:
    with pytest.raises(ModelContractError, match="rope type"):
        generic_causal_lm_spec_from_config(
            _config("yi", rope_scaling={"rope_type": "yarn"})
        )


def _tiny_generic_model(model_type: str) -> GenericCausalLM:
    config = GenericCausalLMConfig(
        model_type=model_type,
        vocab_size=32,
        hidden_size=16,
        intermediate_size=16,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=8,
        max_position_embeddings=64,
        rms_norm_eps=1e-6,
        rope_theta=10_000,
        tie_word_embeddings=False,
        qkv_bias=False,
        qk_norm=False,
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


def _synthetic_generic_sources(
    model: GenericCausalLM,
    model_type: str,
) -> dict[str, tensorplay.Tensor]:
    sources: dict[str, tensorplay.Tensor] = {}
    for name, parameter in model.named_parameters():
        if name == "model.layers.0.self_attn.qkv_proj.weight":
            if model_type == "phi3":
                sources[name] = tensorplay.ones_like(parameter)
            else:
                attention = model.model.layers[0].self_attn
                groups = attention.query_width // attention.kv_width
                query = parameter[: attention.query_width].view(
                    attention.num_kv_heads,
                    groups,
                    attention.head_dim,
                    model.config.hidden_size,
                )
                key = parameter[
                    attention.query_width :
                    attention.query_width + attention.kv_width
                ].view(
                    attention.num_kv_heads,
                    1,
                    attention.head_dim,
                    model.config.hidden_size,
                )
                value = parameter[
                    attention.query_width + attention.kv_width :
                ].view(
                    attention.num_kv_heads,
                    1,
                    attention.head_dim,
                    model.config.hidden_size,
                )
                interleaved = tensorplay.cat(
                    (query, key, value),
                    dim=1,
                ).reshape(parameter.shape)
                sources[
                    "model.layers.0.attention.wqkv.weight"
                ] = tensorplay.ones_like(interleaved)
            continue
        if name == "model.layers.0.mlp.gate_up_weight":
            intermediate = model.config.intermediate_size
            gate, up = parameter.split(
                (intermediate, intermediate),
                dim=0,
            )
            if model_type == "phi3":
                sources[
                    "model.layers.0.mlp.gate_up_proj.weight"
                ] = tensorplay.ones_like(parameter)
            else:
                sources[
                    "model.layers.0.feed_forward.w1.weight"
                ] = tensorplay.ones_like(gate)
                sources[
                    "model.layers.0.feed_forward.w3.weight"
                ] = tensorplay.ones_like(up)
            continue
        source_name = name
        if model_type == "internlm2":
            source_name = source_name.replace(
                "self_attn.o_proj.weight",
                "attention.wo.weight",
            )
            source_name = source_name.replace(
                "mlp.down_proj.weight",
                "feed_forward.w2.weight",
            )
            source_name = source_name.replace(
                "input_layernorm.weight",
                "attention_norm.weight",
            )
            source_name = source_name.replace(
                "post_attention_layernorm.weight",
                "ffn_norm.weight",
            )
            source_name = source_name.replace(
                "model.embed_tokens.weight",
                "model.tok_embeddings.weight",
            )
            source_name = source_name.replace(
                "lm_head_weight",
                "output.weight",
            )
        elif source_name == "lm_head_weight":
            source_name = "lm_head.weight"
        sources[source_name] = tensorplay.ones_like(parameter)
    return sources


@pytest.mark.parametrize("model_type", ("phi3", "internlm2"))
def test_phi3_and_internlm2_synthetic_checkpoints_load(
    model_type: str,
    tmp_path: Path,
) -> None:
    model = _tiny_generic_model(model_type)
    sources = _synthetic_generic_sources(model, model_type)
    save(sources, str(tmp_path / "model.safetensors"))

    report = model.load_weights(tmp_path)

    assert report.tensor_count == len(sources)

