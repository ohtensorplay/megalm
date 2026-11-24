import copy
import json
import re
from pathlib import Path

import pytest
import tensorplay
from tensorplay.serialization import save

from megalm.adapters.qwen3_vl import qwen3_vl_model_spec_from_config
from megalm.contracts.model import (
    AttentionKind,
    FeedForwardKind,
    ModelContractError,
    PositionKind,
    StateKind,
)
from megalm.kernels.ops import KernelPolicy
from megalm.models.qwen3_vl.config import (
    Qwen3VLConfig,
    Qwen3VLConfigError,
    Qwen3VLTextConfig,
    Qwen3VLVisionConfig,
)
from megalm.models.qwen3_vl.model import Qwen3VLForConditionalGeneration
from megalm.runtime.batch import BatchContractError, BatchPhase, ExecutionBatch


def _config(model_type: str = "qwen3_vl") -> dict:
    is_moe = model_type == "qwen3_vl_moe"
    text = {
        "model_type": "qwen3_vl_moe_text" if is_moe else "qwen3_vl_text",
        "vocab_size": 1000,
        "hidden_size": 64,
        "num_hidden_layers": 2,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "head_dim": 16,
        "max_position_embeddings": 512,
        "intermediate_size": 32,
        "rms_norm_eps": 1e-6,
        "rope_parameters": {
            "rope_type": "default",
            "rope_theta": 1_000_000,
            "mrope_section": [1, 1, 1],
            "mrope_interleaved": True,
            "partial_rotary_factor": 0.375,
        },
    }
    if is_moe:
        text.update(
            num_experts=8,
            num_experts_per_tok=2,
            moe_intermediate_size=32,
        )
    return {
        "model_type": model_type,
        "image_token_id": 10,
        "video_token_id": 11,
        "vision_start_token_id": 12,
        "vision_end_token_id": 13,
        "text_config": text,
        "vision_config": {
            "depth": 4,
            "hidden_size": 16,
            "intermediate_size": 32,
            "num_heads": 4,
            "in_channels": 3,
            "patch_size": 2,
            "spatial_merge_size": 2,
            "temporal_patch_size": 2,
            "out_hidden_size": 64,
            "num_position_embeddings": 1024,
            "hidden_act": "silu",
            "deepstack_visual_indexes": [1, 3],
        },
    }


def test_qwen3_vl_dense_spec_declares_mrope_and_deepstack_media() -> None:
    spec = qwen3_vl_model_spec_from_config(_config())

    assert spec.model_type == "qwen3_vl"
    assert spec.architecture == "Qwen3VLForConditionalGeneration"
    assert all(
        layer.attention is AttentionKind.FULL
        for layer in spec.layers
    )
    assert all(
        layer.feed_forward is FeedForwardKind.DENSE
        for layer in spec.layers
    )
    assert spec.position.kind is PositionKind.MROPE
    assert spec.position.mrope_sections == (1, 1, 1)
    assert spec.qk_norm is True
    assert {
        component.kind for component in spec.state_schema
    } >= {
        StateKind.PAGED_KV,
        StateKind.VISION_EMBEDDING,
        StateKind.MEDIA_POSITION,
    }
    assert spec.vision is not None
    assert spec.vision.processor_type == "qwen3_vl"
    assert spec.vision.hidden_act == "silu"


def test_qwen3_vl_moe_spec_declares_top2_experts() -> None:
    spec = qwen3_vl_model_spec_from_config(_config("qwen3_vl_moe"))

    assert spec.model_type == "qwen3_vl_moe"
    assert spec.architecture == "Qwen3VLMoeForConditionalGeneration"
    assert all(
        layer.feed_forward is FeedForwardKind.MOE
        for layer in spec.layers
    )
    assert all(layer.num_experts == 8 for layer in spec.layers)
    assert all(layer.experts_per_token == 2 for layer in spec.layers)


def test_qwen3_vl_config_loads_deepstack_defaults(tmp_path: Path) -> None:
    payload = _config()
    payload.pop("architectures", None)
    (tmp_path / "config.json").write_text(
        json.dumps(payload),
        encoding="utf-8",
    )

    config = Qwen3VLConfig.from_path(tmp_path)

    assert config.architecture == "Qwen3VLForConditionalGeneration"
    assert config.vision_config.deepstack_depth == 2
    assert config.vision_config.deepstack_visual_indexes == (1, 3)
    assert config.text_config.mrope_sections == (1, 1, 1)
    assert config.text_config.partial_rotary_factor == 0.375


def test_qwen3_vl_rejects_invalid_mrope_sections() -> None:
    payload = _config()
    payload["text_config"]["rope_parameters"]["mrope_section"] = [9, 9]

    with pytest.raises(ModelContractError, match="sections"):
        qwen3_vl_model_spec_from_config(payload)

    with pytest.raises(Qwen3VLConfigError, match="sections"):
        Qwen3VLTextConfig.from_dict(payload["text_config"])


def _batch(
    deepstack_embeddings: tuple[tensorplay.Tensor, ...] = (),
) -> ExecutionBatch:
    return ExecutionBatch(
        input_ids=tensorplay.tensor((10, 11), dtype=tensorplay.int64),
        positions=tensorplay.tensor((0, 1), dtype=tensorplay.int64),
        slot_mapping=tensorplay.tensor((0, 1), dtype=tensorplay.int64),
        block_tables=tensorplay.zeros((1, 2), dtype=tensorplay.int32),
        sequence_lengths=tensorplay.tensor((2,), dtype=tensorplay.int32),
        token_active=tensorplay.ones(2, dtype=tensorplay.bool),
        query_start_loc=(0, 2),
        query_lengths=(2,),
        context_lengths=(0,),
        phases=(BatchPhase.PREFILL,),
        sample_indices=tensorplay.tensor((1,), dtype=tensorplay.int64),
        page_size=1,
        deepstack_embeddings=deepstack_embeddings,
    )


def test_deepstack_embeddings_are_validated() -> None:
    batch = _batch((tensorplay.zeros((2, 16)),))
    assert len(batch.deepstack_embeddings) == 1

    with pytest.raises(BatchContractError, match="deepstack"):
        _batch((tensorplay.zeros((1, 16)),))


def test_qwen3_vl_moe_rejects_top_k_larger_than_experts() -> None:
    payload = copy.deepcopy(_config("qwen3_vl_moe"))
    payload["text_config"]["num_experts_per_tok"] = 9

    with pytest.raises(ModelContractError, match="exceeds"):
        qwen3_vl_model_spec_from_config(payload)


def _tiny_model() -> Qwen3VLForConditionalGeneration:
    text = Qwen3VLTextConfig(
        model_type="qwen3_vl_text",
        vocab_size=32,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=8,
        max_position_embeddings=64,
        rms_norm_eps=1e-6,
        rope_theta=10_000,
        partial_rotary_factor=0.375,
        mrope_sections=(1, 1, 1),
        mrope_interleaved=True,
        tie_word_embeddings=False,
    )
    vision = Qwen3VLVisionConfig(
        depth=1,
        hidden_size=8,
        intermediate_size=16,
        num_heads=2,
        in_channels=3,
        patch_size=2,
        spatial_merge_size=2,
        temporal_patch_size=2,
        out_hidden_size=16,
        num_position_embeddings=4,
        hidden_act="silu",
        deepstack_visual_indexes=(0,),
    )
    config = Qwen3VLConfig(
        model_type="qwen3_vl",
        architecture="Qwen3VLForConditionalGeneration",
        text_config=text,
        vision_config=vision,
        image_token_id=10,
        video_token_id=11,
        vision_start_token_id=12,
        vision_end_token_id=13,
        tie_word_embeddings=False,
    )
    return Qwen3VLForConditionalGeneration(
        config,
        device="cpu",
        dtype=tensorplay.float32,
        kernel_policy=KernelPolicy(use_triton=False),
    )


def _synthetic_source_tensors(
    model: Qwen3VLForConditionalGeneration,
) -> dict[str, tensorplay.Tensor]:
    sources: dict[str, tensorplay.Tensor] = {}
    text_qkv = re.compile(
        r"model\.language_model\.layers\.(?P<layer>\d+)\."
        r"self_attn\.qkv_proj_weight"
    )
    text_gate_up = re.compile(
        r"model\.language_model\.layers\.(?P<layer>\d+)\."
        r"mlp\.gate_up_weight"
    )
    visual_qkv = re.compile(
        r"model\.visual\.blocks\.(?P<block>\d+)\.attn\.qkv\.(?P<field>weight|bias)"
    )
    for name, parameter in model.named_parameters():
        match = text_qkv.fullmatch(name)
        if match is not None:
            layer = model.model.language_model.layers[int(match.group("layer"))]
            attention = layer.self_attn
            parts = parameter.split(
                (
                    attention.query_width,
                    attention.kv_width,
                    attention.kv_width,
                ),
                dim=0,
            )
            for projection, part in zip(("q_proj", "k_proj", "v_proj"), parts):
                sources[
                    f"model.language_model.layers.{match.group('layer')}."
                    f"self_attn.{projection}.weight"
                ] = tensorplay.ones_like(part)
            continue
        match = text_gate_up.fullmatch(name)
        if match is not None:
            intermediate = model.config.text_config.intermediate_size
            gate, up = parameter.split((intermediate, intermediate), dim=0)
            prefix = (
                f"model.language_model.layers.{match.group('layer')}.mlp."
            )
            sources[f"{prefix}gate_proj.weight"] = tensorplay.ones_like(gate)
            sources[f"{prefix}up_proj.weight"] = tensorplay.ones_like(up)
            continue
        match = visual_qkv.fullmatch(name)
        if match is not None:
            field = match.group("field")
            width = model.config.vision_config.hidden_size
            q, k, v = parameter.split((width, width, width), dim=0)
            prefix = f"model.visual.blocks.{match.group('block')}.attn."
            for projection, part in zip(("q", "k", "v"), (q, k, v)):
                sources[f"{prefix}{projection}.{field}"] = tensorplay.ones_like(
                    part
                )
            continue
        source_name = name
        if source_name.endswith("self_attn.o_proj_weight"):
            source_name = source_name.replace(
                "o_proj_weight",
                "o_proj.weight",
            )
        elif source_name.endswith("mlp.down_proj_weight"):
            source_name = source_name.replace(
                "down_proj_weight",
                "down_proj.weight",
            )
        elif source_name == "lm_head_weight":
            source_name = "lm_head.weight"
        sources[source_name] = tensorplay.ones_like(parameter)
    return sources


def test_qwen3_vl_synthetic_checkpoint_loads(tmp_path: Path) -> None:
    model = _tiny_model()
    sources = _synthetic_source_tensors(model)
    save(sources, str(tmp_path / "model.safetensors"))

    report = model.load_weights(tmp_path)

    assert report.tensor_count == len(sources)



def _quantization_payload() -> dict:
    return {
        "config_groups": {
            "group_0": {
                "format": "pack-quantized",
                "input_activations": None,
                "output_activations": None,
                "targets": ["Linear"],
                "weights": {
                    "num_bits": 4,
                    "type": "int",
                    "symmetric": True,
                    "group_size": 32,
                    "strategy": "group",
                    "dynamic": False,
                },
            }
        },
        "format": "pack-quantized",
        "quant_method": "compressed-tensors",
        "quantization_status": "compressed",
    }


def _packed_words(codes: tensorplay.Tensor) -> tensorplay.Tensor:
    rows, width = codes.shape
    grouped = codes.reshape(rows, width // 8, 8)
    packed = tensorplay.zeros((rows, width // 8), dtype=tensorplay.int32)
    for index in range(8):
        packed = packed + grouped[:, :, index].bitwise_left_shift(4 * index)
    return packed


def _add_quantized_projection(
    sources: dict[str, tensorplay.Tensor],
    prefix: str,
    out_features: int,
    in_features: int,
    group_size: int = 32,
) -> None:
    codes = tensorplay.randint(
        0,
        16,
        (out_features, in_features),
        dtype=tensorplay.int32,
    )
    scale = (tensorplay.rand((out_features, in_features // group_size)) * 0.02).to(
        tensorplay.bfloat16
    )
    sources[f"{prefix}.weight_packed"] = _packed_words(codes)
    sources[f"{prefix}.weight_scale"] = scale
    sources[f"{prefix}.weight_shape"] = tensorplay.tensor(
        [out_features, in_features],
        dtype=tensorplay.long,
    )


def test_qwen3_vl_rope_theta_falls_back_to_text_config(tmp_path: Path) -> None:
    payload = _config()
    payload["text_config"].pop("rope_parameters")
    payload["text_config"]["rope_theta"] = 5_000_000
    payload["text_config"]["rope_scaling"] = {
        "rope_type": "default",
        "mrope_section": [1, 1, 1],
        "mrope_interleaved": True,
    }
    (tmp_path / "config.json").write_text(json.dumps(payload), encoding="utf-8")

    config = Qwen3VLConfig.from_path(tmp_path)

    assert config.text_config.rope_theta == 5_000_000


def test_qwen3_vl_config_reads_compressed_tensors(tmp_path: Path) -> None:
    payload = _config()
    payload["quantization_config"] = _quantization_payload()
    (tmp_path / "config.json").write_text(json.dumps(payload), encoding="utf-8")

    config = Qwen3VLConfig.from_path(tmp_path)

    assert config.quantization is not None
    assert config.quantization.group_size == 32
    assert config.quantization.num_bits == 4
    assert config.quantization.symmetric is True


def test_qwen3_vl_rejects_activation_quantization() -> None:
    from megalm.models.qwen3_vl.config import Qwen3VLQuantConfig

    payload = _quantization_payload()
    payload["config_groups"]["group_0"]["input_activations"] = {"num_bits": 8}

    with pytest.raises(Qwen3VLConfigError, match="activation"):
        Qwen3VLQuantConfig.from_dict(payload)


def test_qwen3_vl_quantized_checkpoint_loads(tmp_path: Path) -> None:
    from megalm.models.components.int4_linear import Int4GroupLinear
    from megalm.models.qwen3_vl.config import Qwen3VLQuantConfig

    payload = _config()
    payload["quantization_config"] = _quantization_payload()
    (tmp_path / "config.json").write_text(json.dumps(payload), encoding="utf-8")
    config = Qwen3VLConfig.from_path(tmp_path)
    model = Qwen3VLForConditionalGeneration(
        config,
        device="cpu",
        dtype=tensorplay.float32,
        kernel_policy=KernelPolicy(use_triton=False),
    )
    sources = _synthetic_source_tensors(model)
    hidden = config.text_config.hidden_size
    intermediate = config.text_config.intermediate_size
    for layer in range(config.text_config.num_hidden_layers):
        prefix = f"model.language_model.layers.{layer}"
        attention = model.model.language_model.layers[layer].self_attn
        for projection, out_features in (
            ("q_proj", attention.query_width),
            ("k_proj", attention.kv_width),
            ("v_proj", attention.kv_width),
            ("o_proj", hidden),
        ):
            _add_quantized_projection(
                sources,
                f"{prefix}.self_attn.{projection}",
                out_features if projection != "o_proj" else hidden,
                attention.query_width if projection == "o_proj" else hidden,
            )
        _add_quantized_projection(
            sources, f"{prefix}.mlp.gate_proj", intermediate, hidden
        )
        _add_quantized_projection(
            sources, f"{prefix}.mlp.up_proj", intermediate, hidden
        )
        _add_quantized_projection(
            sources, f"{prefix}.mlp.down_proj", hidden, intermediate
        )
    save(sources, str(tmp_path / "model.safetensors"))

    report = model.load_weights(tmp_path)

    assert report.tensor_count == len(sources)
    layer0 = model.model.language_model.layers[0]
    assert isinstance(layer0.self_attn.qkv_int4, Int4GroupLinear)
    assert layer0.self_attn.qkv_int4.is_loaded
    assert layer0.self_attn.qkv_int4.out_features == (
        layer0.self_attn.query_width + 2 * layer0.self_attn.kv_width
    )
    assert tuple(layer0.self_attn.qkv_int4.packed.shape) == (
        layer0.self_attn.query_width + 2 * layer0.self_attn.kv_width,
        hidden // 2,
    )
    assert layer0.mlp.gate_up_int4.is_loaded
    assert tuple(layer0.mlp.gate_up_int4.packed.shape) == (
        2 * intermediate,
        hidden // 2,
    )
    assert layer0.mlp.down_proj_int4.is_loaded
    assert tuple(layer0.mlp.down_proj_int4.packed.shape) == (
        hidden,
        intermediate // 2,
    )
    assert isinstance(config.quantization, Qwen3VLQuantConfig)
    assert not any(
        name.endswith(("qkv_proj_weight", "gate_up_weight"))
        for name, _ in model.named_parameters()
    )
