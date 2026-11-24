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
    "minicpm",
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


def test_minicpm5_decoupled_head_dim_is_accepted() -> None:
    spec = generic_causal_lm_spec_from_config(
        _config(
            "llama",
            hidden_size=1536,
            intermediate_size=4608,
            num_hidden_layers=24,
            num_attention_heads=16,
            num_key_value_heads=2,
            head_dim=128,
            max_position_embeddings=131072,
            rope_theta=5_000_000,
        )
    )

    assert spec.position.head_dim == 128
    assert spec.position.rope_theta == 5_000_000
    assert spec.position.max_positions == 131072
    assert spec.hidden_size == 1536
    assert all(layer.num_attention_heads == 16 for layer in spec.layers)


def test_implicit_head_dim_still_requires_matching_hidden_size() -> None:
    payload = _config("llama", hidden_size=1000, num_attention_heads=16)
    payload.pop("head_dim")

    with pytest.raises(ModelContractError, match="head_dim is declared explicitly"):
        generic_causal_lm_spec_from_config(payload)


def test_minicpm5_native_config_keeps_declared_head_dim(tmp_path: Path) -> None:
    (tmp_path / "config.json").write_text(
        json.dumps(
            _config(
                "llama",
                hidden_size=1536,
                intermediate_size=4608,
                num_attention_heads=16,
                num_key_value_heads=2,
                head_dim=128,
                max_position_embeddings=131072,
                rope_theta=5_000_000,
            )
        ),
        encoding="utf-8",
    )

    config = GenericCausalLMConfig.from_path(tmp_path)

    assert config.head_dim == 128
    assert config.hidden_size == 1536
    assert config.rope_theta == 5_000_000
    assert config.max_position_embeddings == 131072


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


def _minicpm4_config(**overrides: object) -> dict:
    factors = [1.0 + 0.5 * index for index in range(8)]
    payload = _config(
        "minicpm",
        num_hidden_layers=4,
        max_position_embeddings=128,
        scale_emb=12,
        scale_depth=1.4,
        dim_model_base=16,
        rope_scaling={
            "rope_type": "longrope",
            "short_factor": factors,
            "long_factor": factors,
            "original_max_position_embeddings": 128,
        },
    )
    payload.update(overrides)
    return payload


def test_minicpm4_scaling_and_longrope_spec() -> None:
    spec = generic_causal_lm_spec_from_config(_minicpm4_config())

    assert spec.architecture == "MiniCPMForCausalLM"
    assert spec.model_type == "minicpm"
    assert spec.embedding_multiplier == 12.0
    assert spec.residual_multiplier == pytest.approx(1.4 / (4**0.5))
    assert spec.logit_scale == pytest.approx(16 / 64)
    assert spec.tie_word_embeddings is True
    assert len(spec.position.rope_short_factor) == 8
    assert len(spec.position.rope_long_factor) == 8
    assert spec.position.rope_original_max_positions == 128


def test_minicpm4_model_type_inferred_from_architecture() -> None:
    payload = _minicpm4_config(
        architectures=["MiniCPMForCausalLM"],
    )
    payload.pop("model_type")

    spec = generic_causal_lm_spec_from_config(payload)

    assert spec.model_type == "minicpm"


def test_minicpm4_native_config_from_path(tmp_path: Path) -> None:
    (tmp_path / "config.json").write_text(
        json.dumps(_minicpm4_config()), encoding="utf-8"
    )

    config = GenericCausalLMConfig.from_path(tmp_path)

    assert config.model_type == "minicpm"
    assert config.tie_word_embeddings is True
    assert config.embedding_multiplier == 12.0
    assert config.residual_multiplier == pytest.approx(1.4 / (4**0.5))
    assert config.logit_scale == pytest.approx(0.25)
    assert len(config.rope_short_factor) == 8


def test_minicpm4_longrope_cache_matches_reference(tmp_path: Path) -> None:
    import math

    (tmp_path / "config.json").write_text(
        json.dumps(_minicpm4_config()), encoding="utf-8"
    )
    config = GenericCausalLMConfig.from_path(tmp_path)
    cache = GenericRotaryCache(config, device=tensorplay.device("cpu"))

    factors = config.rope_short_factor
    original = config.rope_original_max_positions
    scale = math.sqrt(
        1.0
        + math.log(config.max_position_embeddings / original)
        / math.log(original)
    )
    inverse = [
        1.0 / (config.rope_theta ** (index / config.head_dim))
        for index in range(0, config.head_dim, 2)
    ]
    assert tuple(cache.cos.shape) == (config.max_position_embeddings, config.head_dim // 2)
    for position in (0, 1, 7, 64, 127):
        cos_row = cache.cos[position].tolist()[: len(inverse)]
        sin_row = cache.sin[position].tolist()[: len(inverse)]
        for index, base in enumerate(inverse):
            phase = position * base / factors[index]
            assert cos_row[index] == pytest.approx(
                math.cos(phase) * scale, abs=1e-4
            )
            assert sin_row[index] == pytest.approx(
                math.sin(phase) * scale, abs=1e-4
            )


def test_minicpm4_rejects_wrong_longrope_factor_count() -> None:
    config = _minicpm4_config()
    config["rope_scaling"]["short_factor"] = [1.0, 2.0]

    with pytest.raises(ModelContractError, match="factor count"):
        generic_causal_lm_spec_from_config(config)


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

