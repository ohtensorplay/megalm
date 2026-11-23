import json
from pathlib import Path

import pytest

from megalm.adapters.inkling2 import (
    inkling2_model_spec_from_config,
    load_inkling2_model_spec,
)
from megalm.adapters.registry import load_model_spec
from megalm.contracts.model import (
    AttentionKind,
    AttentionResidualSpec,
    FeedForwardKind,
    LoopSpec,
    ModelContractError,
    PositionKind,
    RelativeAttentionSpec,
    ShortConvolutionSpec,
)

# The real thinkingmachines/Inkling-2-Mini config.json (text_config/vision_config/
# mtp_config/quantization_config), trimmed to the fields the adapter consumes.
INKLING2_CONFIG = {
    "dtype": "bfloat16",
    "architectures": ["Inkling2LoopForConditionalGeneration"],
    "model_type": "inkling2_loop",
    "eos_token_id": 200006,
    "quantization_config": {"quant_method": "modelopt", "quant_algo": "NVFP4"},
    "text_config": {
        "model_type": "inkling2_loop_text",
        "model_max_length": 1048576,
        "hidden_size": 2304,
        "num_hidden_layers": 24,
        "vocab_size": 201024,
        "num_attention_heads": 24,
        "num_key_value_heads": 8,
        "head_dim": 128,
        "d_rel": 16,
        "rel_extent": 1024,
        "q_bias": False,
        "o_bias": False,
        "log_scaling_n_floor": 8192,
        "log_scaling_alpha": 0.1,
        "rms_norm_eps": 1e-06,
        "use_value_norm_no_gamma": True,
        "qk_l2_clip_threshold": 22.627416997969522,
        "rel_l2_clip_threshold": 8.0,
        "local_layer_ids": [
            0, 1, 2, 3, 4, 6, 7, 8, 9, 10, 12, 13, 14, 15, 16, 18, 19, 20, 21, 22,
        ],
        "sconv_kernel_size": 4,
        "unpadded_vocab_size": 200058,
        "final_logit_softcapping": 30.0,
        "swa_head_dim": 128,
        "swa_num_attention_heads": 24,
        "swa_num_key_value_heads": 8,
        "sliding_window_size": 1024,
        "n_layers_per_attn_res_block": 6,
        "attn_res_normalize_values": True,
        "n_routed_experts": 128,
        "num_experts_per_tok": 8,
        "n_shared_experts": 0,
        "intermediate_size": 1152,
        "mlp_gate_softcap": 4.0,
        "mlp_up_softcap": 25.0,
        "route_scale": 8.0,
        "use_gate_bias": True,
        "loop_config": {
            "core_start": 6,
            "core_end": 18,
            "num_passes": 3,
            "gate_rank": 64,
            "is_varloop": False,
        },
    },
    "vision_config": {
        "model_type": "inkling2_vision",
        "text_hidden_size": 2304,
        "temporal_patch_size": 2,
        "native_patch_size": 20,
        "adapter_merge_size": 2,
        "projection_hidden_dim": 4096,
        "projector_bias": False,
        "hidden_size": 1024,
        "num_hidden_layers": 23,
        "num_attention_heads": 16,
        "mlp_ratio": 4.0,
        "posemb_grid_size": 32,
        "layer_norm_eps": 1e-05,
    },
    "mtp_config": {"hidden_size": 2304, "num_hidden_layers": 3},
}

LOCAL_IDS = frozenset(INKLING2_CONFIG["text_config"]["local_layer_ids"])


def test_inkling2_spec_matches_the_real_checkpoint() -> None:
    spec = inkling2_model_spec_from_config(INKLING2_CONFIG)

    assert spec.model_type == "inkling2_loop"
    assert spec.architecture == "Inkling2LoopForConditionalGeneration"
    assert spec.num_layers == 24
    assert spec.hidden_size == 2304
    assert spec.vocab_size == 201024
    assert spec.supported_dtypes == ("bfloat16",)
    assert spec.tie_word_embeddings is False
    assert spec.position.kind is PositionKind.LEARNED_RELATIVE
    assert spec.position.max_positions == 1048576

    assert all(layer.attention is AttentionKind.FULL for layer in spec.layers)
    assert all(layer.feed_forward is FeedForwardKind.MOE for layer in spec.layers)
    assert all(layer.num_experts == 128 for layer in spec.layers)
    assert all(layer.experts_per_token == 8 for layer in spec.layers)
    assert all(layer.intermediate_size == 1152 for layer in spec.layers)

    sliding = {
        layer.index: layer.sliding_window
        for layer in spec.layers
        if layer.sliding_window is not None
    }
    assert set(sliding) == set(LOCAL_IDS)
    assert set(sliding.values()) == {1024}


def test_inkling2_loop_and_attention_residual_specs() -> None:
    spec = inkling2_model_spec_from_config(INKLING2_CONFIG)

    assert spec.loop == LoopSpec(
        core_start=6, core_end=18, num_passes=3, gate_rank=64, is_varloop=False
    )
    assert spec.attention_residual == AttentionResidualSpec(
        n_layers_per_block=6, normalize_values=True
    )
    assert spec.short_convolution == ShortConvolutionSpec(kernel_size=4)
    assert spec.final_logit_softcapping == 30.0


def test_inkling2_relative_attention_spec() -> None:
    spec = inkling2_model_spec_from_config(INKLING2_CONFIG)

    assert spec.relative_attention == RelativeAttentionSpec(
        d_rel=16,
        rel_extent=1024,
        log_scaling_n_floor=8192,
        log_scaling_alpha=0.1,
        qk_l2_clip_threshold=22.627416997969522,
        rel_l2_clip_threshold=8.0,
        q_bias=False,
        o_bias=False,
        use_value_norm_no_gamma=True,
    )


def test_inkling2_vision_and_quantization_specs() -> None:
    spec = inkling2_model_spec_from_config(INKLING2_CONFIG)

    assert spec.vision is not None
    assert spec.vision.processor_type == "inkling2_vision"
    assert spec.vision.patch_size == 20
    assert spec.vision.merge_size == 2
    assert spec.vision.output_hidden_size == 2304
    assert spec.vision.num_position_embeddings == 32 * 32

    assert spec.quantization is not None
    assert spec.quantization.method == "modelopt"
    assert spec.quantization.algo == "NVFP4"
    assert spec.quantization.config_file == "hf_quant_config.json"


def test_inkling2_spec_digest_is_stable_and_complete() -> None:
    spec = inkling2_model_spec_from_config(INKLING2_CONFIG)
    assert len(spec.digest) == 64
    assert spec.digest == inkling2_model_spec_from_config(INKLING2_CONFIG).digest


def test_registry_dispatches_inkling2(tmp_path: Path) -> None:
    (tmp_path / "config.json").write_text(
        json.dumps(INKLING2_CONFIG), encoding="utf-8"
    )
    spec = load_model_spec(tmp_path)
    assert spec.model_type == "inkling2_loop"
    assert spec.num_layers == 24


def test_quantized_tensor_counts_from_manifest(tmp_path: Path) -> None:
    (tmp_path / "config.json").write_text(
        json.dumps(INKLING2_CONFIG), encoding="utf-8"
    )
    manifest = {
        "model": {
            "parameter_groups": [
                {"compute_dtype": "nvfp4", "tensors": ["a.w13_weight", "a.w2_weight"]},
                {"compute_dtype": "mxfp8", "tensors": ["b.w13_weight"]},
                {"compute_dtype": "bfloat16", "tensors": ["norm.weight"]},
            ]
        }
    }
    (tmp_path / "hf_quant_config.json").write_text(
        json.dumps(manifest), encoding="utf-8"
    )
    spec = load_inkling2_model_spec(tmp_path)
    assert spec.quantization is not None
    assert spec.quantization.nvfp4_tensors == 2
    assert spec.quantization.mxfp8_tensors == 1


def test_inkling2_rejects_foreign_model_type() -> None:
    config = dict(INKLING2_CONFIG)
    config["model_type"] = "qwen3_moe"
    with pytest.raises(ModelContractError, match="does not support"):
        inkling2_model_spec_from_config(config)


def test_inkling2_rejects_expert_count_mismatch() -> None:
    config = json.loads(json.dumps(INKLING2_CONFIG))
    config["text_config"]["num_experts_per_tok"] = 200
    with pytest.raises(ModelContractError, match="exceeds the expert count"):
        inkling2_model_spec_from_config(config)


def test_inkling2_rejects_invalid_local_layer_ids() -> None:
    config = json.loads(json.dumps(INKLING2_CONFIG))
    config["text_config"]["local_layer_ids"] = [0, 0, 999]
    with pytest.raises(ModelContractError, match="local_layer_ids"):
        inkling2_model_spec_from_config(config)


def test_inkling2_rejects_loop_beyond_layer_count() -> None:
    config = json.loads(json.dumps(INKLING2_CONFIG))
    config["text_config"]["loop_config"]["core_end"] = 999
    with pytest.raises(ModelContractError, match="core_end exceeds"):
        inkling2_model_spec_from_config(config)


def test_inkling2_rejects_missing_text_config() -> None:
    config = dict(INKLING2_CONFIG)
    config.pop("text_config")
    with pytest.raises(ModelContractError, match="requires object 'text_config'"):
        inkling2_model_spec_from_config(config)