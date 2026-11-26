"""Native Gemma-4 tests: config geometry, adapter layout and HF parity."""

from __future__ import annotations

import math
from typing import Any

import pytest
import tensorplay

from megalm.adapters.gemma4 import gemma4_model_spec_from_config
from megalm.contracts.request import SamplingParams
from megalm.kernels.ops import KernelPolicy
from megalm.models.gemma4.config import Gemma4AssistantConfig, Gemma4Config
from megalm.models.gemma4.model import Gemma4ForCausalLM
from megalm.models.components.mtp import MaskedLogitsEmbedder
from megalm.models.gemma4.mtp import Gemma4MultiTokenPredictor
from megalm.runtime.batch import BatchPhase, ExecutionBatch
from megalm.runtime.cache import PagedKVCache

H, D, S = 32, 8, 6
LAYER_TYPES = [
    "sliding_attention",
    "sliding_attention",
    "full_attention",
    "sliding_attention",
    "sliding_attention",
    "full_attention",
]


def _payload(**overrides: object) -> dict:
    payload = {
        "model_type": "gemma4_text",
        "architectures": ["Gemma4ForCausalLM"],
        "vocab_size": 64,
        "hidden_size": H,
        "intermediate_size": 64,
        "num_hidden_layers": S,
        "num_attention_heads": 4,
        "num_key_value_heads": 1,
        "head_dim": D,
        "global_head_dim": 2 * D,
        "max_position_embeddings": 64,
        "hidden_activation": "gelu_pytorch_tanh",
        "tie_word_embeddings": True,
        "sliding_window": 4,
        "layer_types": list(LAYER_TYPES),
        "final_logit_softcapping": 30.0,
        "hidden_size_per_layer_input": 8,
        "vocab_size_per_layer_input": 64,
        "num_kv_shared_layers": 3,
        "use_double_wide_mlp": True,
    }
    payload.update(overrides)
    return payload


def _parsed(**overrides: object) -> Gemma4Config:
    return Gemma4Config.from_dict(
        {"model_type": "gemma4", "text_config": _payload(**overrides)}
    )


def _batch(ids: list[int]) -> ExecutionBatch:
    n = len(ids)
    return ExecutionBatch(
        input_ids=tensorplay.tensor(ids, dtype=tensorplay.int64),
        positions=tensorplay.tensor(list(range(n)), dtype=tensorplay.int64),
        slot_mapping=tensorplay.tensor(list(range(n)), dtype=tensorplay.int64),
        block_tables=tensorplay.tensor([[0, 1]], dtype=tensorplay.int32),
        sequence_lengths=tensorplay.tensor([n], dtype=tensorplay.int32),
        token_active=tensorplay.ones(n, dtype=tensorplay.bool),
        query_start_loc=(0, n),
        query_lengths=(n,),
        context_lengths=(0,),
        phases=(BatchPhase.PREFILL,),
        sample_indices=tensorplay.tensor([n - 1], dtype=tensorplay.int64),
        page_size=4,
    )


def _cache(model: Gemma4ForCausalLM) -> PagedKVCache:
    return PagedKVCache(
        num_layers=model.num_kv_cache_layers,
        num_pages=4,
        page_size=4,
        num_kv_heads=model.cache_num_kv_heads,
        head_dim=model.cache_head_dim,
        dtype=tensorplay.float32,
        device="cpu",
        layer_indices=model.kv_cache_layer_indices,
    )


def _make_model(**overrides: object) -> Gemma4ForCausalLM:
    cfg = _parsed(**overrides)
    model = Gemma4ForCausalLM(
        cfg.text,
        device="cpu",
        dtype=tensorplay.float32,
        kernel_policy=KernelPolicy(use_triton=False),
    ).eval()
    # Parameters are intentionally allocated uninitialised; give the synthetic
    # model small deterministic weights so a forward pass is well-defined.
    with tensorplay.no_grad():
        for index, (_, param) in enumerate(model.named_parameters()):
            values = tensorplay.randn(param.shape, dtype=tensorplay.float32)
            param.copy_((values * 0.02).to(dtype=param.dtype))
    return model


def test_config_geometry() -> None:
    cfg = _parsed().text
    assert cfg.layer_head_dims == (D, D, 2 * D, D, D, 2 * D)
    assert cfg.layer_num_key_value_heads == (1,) * S
    assert cfg.layer_types == tuple(LAYER_TYPES)
    assert cfg.max_head_dim == 2 * D
    assert cfg.max_kv_heads == 1
    assert cfg.num_kv_cache_layers == 3  # 6 - 3 shared layers
    # Shared layers reuse the last owning layer of the same type.
    assert cfg.kv_slots == (0, 1, 2, 1, 1, 2)
    assert cfg.uses_ple
    assert math.isclose(cfg.final_logit_softcapping or 0.0, 30.0)


def test_config_rejects_bad_moe() -> None:
    with pytest.raises(Exception):
        _parsed(enable_moe_block=True, num_experts=0)


def test_adapter_layout() -> None:
    config = {"model_type": "gemma4", "text_config": _payload()}
    spec = gemma4_model_spec_from_config(config)
    assert len(spec.layers) == S
    assert [layer.sliding_window for layer in spec.layers] == [
        4, 4, None, 4, 4, None
    ]
    assert [layer.num_key_value_heads for layer in spec.layers] == [1] * S
    assert [layer.head_dim for layer in spec.layers] == [
        D, D, 2 * D, D, D, 2 * D
    ]
    assert spec.final_logit_softcapping == 30.0
    assert spec.qk_norm
    assert spec.state_schema[0].layer_indices == (0, 1, 2)


def test_forward_shape_and_finite() -> None:
    model = _make_model()
    out = model(_batch([1, 2, 3, 4, 5]), _cache(model))
    assert out.shape == (1, _parsed().text.vocab_size)
    assert tensorplay.isfinite(out).all().item()


def test_moe_forward_shape_and_finite() -> None:
    model = _make_model(
        enable_moe_block=True,
        num_experts=4,
        top_k_experts=2,
        moe_intermediate_size=32,
    )
    out = model(_batch([1, 2, 3, 4, 5]), _cache(model))
    assert out.shape == (1, _parsed().text.vocab_size)
    assert tensorplay.isfinite(out).all().item()


def _assistant_config() -> Gemma4AssistantConfig:
    text = _payload(
        num_hidden_layers=2,
        layer_types=["sliding_attention", "full_attention"],
        hidden_size_per_layer_input=0,
        vocab_size_per_layer_input=0,
        num_kv_shared_layers=2,  # assistant checkpoints mark all layers shared
    )
    return Gemma4AssistantConfig.from_dict(
        {
            "model_type": "gemma4_assistant",
            "text_config": text,
            "backbone_hidden_size": H,
        }
    )


def _mtp_model() -> Gemma4MultiTokenPredictor:
    from megalm.kernels.bindings import bind_kernel_plan

    config = _assistant_config()
    bindings = bind_kernel_plan(
        KernelPolicy(use_triton=False),
        device=tensorplay.device("cpu"),
        dtype=tensorplay.float32,
        model_type="gemma4",
    )
    model = Gemma4MultiTokenPredictor(
        config,
        backbone_embed_tokens=tensorplay.empty(
            (config.text_config.vocab_size, H), dtype=tensorplay.float32
        ),
        kernel_bindings=bindings,
        device=tensorplay.device("cpu"),
        dtype=tensorplay.float32,
    )
    with tensorplay.no_grad():
        for _, param in model.named_parameters():
            values = tensorplay.randn(param.shape, dtype=tensorplay.float32)
            param.copy_((values * 0.02).to(dtype=param.dtype))
    return model


def _vision_model():
    from megalm.kernels.bindings import bind_kernel_plan
    from megalm.models.gemma4.config import Gemma4VisionConfig
    from megalm.models.gemma4.vision import Gemma4VisionModel, MultimodalEmbedder

    cfg = Gemma4VisionConfig(
        hidden_size=32, intermediate_size=64, num_hidden_layers=2,
        num_attention_heads=4, num_key_value_heads=2, head_dim=8,
        patch_size=2, pooling_kernel_size=2, position_embedding_size=16,
    )
    bindings = bind_kernel_plan(
        KernelPolicy(use_triton=False),
        device=tensorplay.device("cpu"),
        dtype=tensorplay.float32,
        model_type="gemma4",
    )
    tower = Gemma4VisionModel(
        cfg, device=tensorplay.device("cpu"), dtype=tensorplay.float32,
        kernel_bindings=bindings,
    )
    connector = MultimodalEmbedder(
        input_size=cfg.hidden_size, output_size=H, eps=cfg.rms_norm_eps,
        device=tensorplay.device("cpu"), dtype=tensorplay.float32,
        kernel_bindings=bindings,
    )
    with tensorplay.no_grad():
        for module in (tower, connector):
            for _, param in module.named_parameters():
                values = tensorplay.randn(param.shape, dtype=tensorplay.float32)
                param.copy_((values * 0.02).to(dtype=param.dtype))
    return tower, connector


def test_image_media_geometry_matches_reference() -> None:
    from megalm.models.gemma4.media import image_geometry

    geom = image_geometry(
        640,
        480,
        patch_size=16,
        max_soft_tokens=280,
        pooling_kernel_size=3,
    )
    # Matches transformers' get_aspect_ratio_preserving_size(640, 480).
    assert (geom.patch_h, geom.patch_w) == (57, 42)
    assert geom.num_soft_tokens == 266
    assert len(geom.positions()) == geom.patch_h * geom.patch_w
    assert geom.positions()[0] == (0, 0)
    assert geom.positions()[-1] == (geom.patch_w - 1, geom.patch_h - 1)

    # Extreme aspect ratio must stay capped at the configured budget.
    wide = image_geometry(
        3,
        900,
        patch_size=16,
        max_soft_tokens=280,
        pooling_kernel_size=3,
    )
    assert wide.num_soft_tokens <= 280


def test_audio_media_token_count_matches_reference() -> None:
    from megalm.models.gemma4.media import num_audio_soft_tokens

    count = num_audio_soft_tokens(
        16_000,
        sampling_rate=16_000,
        audio_seq_length=750,
        frame_length=320,
        hop_length=160,
    )
    # 100 mel frames -> 50 -> 25 after the two stride-2 layers.
    assert count == 25
    assert num_audio_soft_tokens(
        0,
        sampling_rate=16_000,
        audio_seq_length=750,
        frame_length=320,
        hop_length=160,
    ) == 0


def test_vision_tower_and_connector() -> None:
    tower, connector = _vision_model()
    pixels = tensorplay.rand((8, 12), dtype=tensorplay.float32)
    positions = tensorplay.tensor(
        [[x, y] for x in range(4) for y in range(2)], dtype=tensorplay.int64
    )
    features, valid = tower(pixels, positions)
    assert valid.shape == (2,)
    assert features.shape == (2, 32)
    assert tensorplay.isfinite(features).all().item()
    embedded = connector(features)
    assert embedded.shape == (2, H)
    assert tensorplay.isfinite(embedded).all().item()


def test_audio_tower() -> None:
    from megalm.kernels.bindings import bind_kernel_plan
    from megalm.models.gemma4.audio import Gemma4AudioModel
    from megalm.models.gemma4.config import Gemma4AudioConfig

    cfg = Gemma4AudioConfig(
        hidden_size=32, num_hidden_layers=1, num_attention_heads=4,
        num_mel_bins=16, subsampling_conv_channels=(8, 8),
        output_proj_dims=32, attention_chunk_size=4,
        attention_context_left=3,
    )
    bindings = bind_kernel_plan(
        KernelPolicy(use_triton=False),
        device=tensorplay.device("cpu"), dtype=tensorplay.float32,
        model_type="gemma4",
    )
    model = Gemma4AudioModel(
        cfg, device=tensorplay.device("cpu"), dtype=tensorplay.float32,
        kernel_bindings=bindings,
    )
    with tensorplay.no_grad():
        for _, param in model.named_parameters():
            values = tensorplay.randn(param.shape, dtype=tensorplay.float32)
            param.copy_((values * 0.02).to(dtype=param.dtype))
    features = tensorplay.rand((1, 8, 16), dtype=tensorplay.float32)
    output, mask = model(features)
    assert output.shape[-1] == 32
    assert tensorplay.isfinite(output).all().item()


def test_unified_embedders() -> None:
    from megalm.kernels.bindings import bind_kernel_plan
    from megalm.models.gemma4.config import Gemma4VisionConfig
    from megalm.models.gemma4.unified import (
        Gemma4UnifiedAudioEmbedder,
        Gemma4UnifiedVisionEmbedder,
    )

    bindings = bind_kernel_plan(
        KernelPolicy(use_triton=False),
        device=tensorplay.device("cpu"), dtype=tensorplay.float32,
        model_type="gemma4",
    )
    text = _parsed().text
    vision = Gemma4VisionConfig(
        patch_size=2, mm_embed_dim=16, mm_posemb_size=8
    )
    vision_model = Gemma4UnifiedVisionEmbedder(
        vision, text, device=tensorplay.device("cpu"),
        dtype=tensorplay.float32, kernel_bindings=bindings,
    )
    from megalm.models.gemma4.config import Gemma4AudioConfig

    audio_model = Gemma4UnifiedAudioEmbedder(
        Gemma4AudioConfig(output_proj_dims=32),
        text, device=tensorplay.device("cpu"), dtype=tensorplay.float32,
        kernel_bindings=bindings,
    )
    for module in (vision_model, audio_model):
        with tensorplay.no_grad():
            for _, param in module.named_parameters():
                values = tensorplay.randn(param.shape, dtype=tensorplay.float32)
                param.copy_((values * 0.02).to(dtype=param.dtype))
    patches = tensorplay.rand((4, 12), dtype=tensorplay.float32)
    positions = tensorplay.tensor([[0, 0], [0, 1], [1, 0], [1, 1]])
    vision_out = vision_model(patches, positions)
    assert vision_out.shape == (4, H)
    frames = tensorplay.rand((3, 32), dtype=tensorplay.float32)
    audio_out = audio_model(frames)
    assert audio_out.shape == (3, H)
    assert tensorplay.isfinite(vision_out).all().item()
    assert tensorplay.isfinite(audio_out).all().item()


def test_mtp_config_and_forward() -> None:
    config = _assistant_config()
    # Native predictor resolves geometry with independent KV slots.
    assert config.text_config.num_kv_shared_layers == 0
    assert config.text_config.layer_head_dims == (D, 2 * D)
    model = _mtp_model()
    assert len(model.layers) == 2
    ids = [1, 2, 3]
    target_hidden = tensorplay.randn((len(ids), H), dtype=tensorplay.float32) * 0.02
    draft, backbone = model(_batch(ids), target_hidden, _cache(_make_model()))
    assert draft.shape == (len(ids), H)
    assert backbone.shape == (len(ids), H)
    assert tensorplay.isfinite(draft).all().item()
    lm_head = tensorplay.randn((64, H), dtype=tensorplay.float32) * 0.02
    logits = model.compute_logits(draft[-1:], lm_head)
    assert logits.shape == (1, 64)
    assert tensorplay.isfinite(logits).all().item()


def test_mtp_masked_embedder() -> None:
    config = _assistant_config()
    from dataclasses import replace

    config = replace(
        config,
        num_centroids=8,
        centroid_intermediate_top_k=2,
        use_ordered_embeddings=True,
    )
    embedder = MaskedLogitsEmbedder(
        hidden_size=H,
        vocab_size=64,
        num_centroids=8,
        top_k=2,
        device=tensorplay.device("cpu"),
        dtype=tensorplay.float32,
    )
    with tensorplay.no_grad():
        for _, param in embedder.named_parameters():
            values = tensorplay.randn(param.shape, dtype=tensorplay.float32)
            param.copy_((values * 0.02).to(dtype=param.dtype))
    hidden = tensorplay.randn((2, H), dtype=tensorplay.float32) * 0.02
    lm_head = tensorplay.randn((64, H), dtype=tensorplay.float32) * 0.02
    logits = embedder.compute_logits(hidden, lm_head)
    assert logits.shape == (2, 64)
    top = embedder.get_top_tokens(hidden, lm_head)
    assert top.shape == (2,)


def test_tokenizer_expands_gemma4_media_placeholders() -> None:
    from pathlib import Path

    from megalm.runtime.tokenizer import ModelTokenizer

    roots = sorted(
        Path("/home/bluemoon/.cache/huggingface/hub").glob(
            "models--google--gemma-4-*/snapshots/*"
        )
    )
    if not roots:
        pytest.skip("no downloaded Gemma-4 checkpoint")
    tokenizer = ModelTokenizer(roots[0])
    encoding = tokenizer.encode_multimodal_chat(
        [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": "what is this"},
                ],
            }
        ],
        image_feature_counts=(266,),
    )
    assert encoding.image_count == 1
    assert encoding.token_type_ids.count(1) == 266
    ids = encoding.input_ids
    start = tokenizer.vision_start_token_id
    end = tokenizer.vision_end_token_id
    assert start in ids and end in ids
    assert ids.index(start) < ids.index(end)


def test_media_admission_uses_plain_positions_for_gemma4() -> None:
    from types import SimpleNamespace

    from megalm.contracts.request import ImageInput
    from megalm.runtime.media_admission import (
        MultimodalAdmission,
        build_media_admission,
    )

    class _Tokenizer:
        def encode_multimodal_chat(self, *args: object, **kwargs: object) -> Any:
            return SimpleNamespace(
                input_ids=(1, 2, 3, 4),
                token_type_ids=(0, 1, 1, 0),
            )

    admission = MultimodalAdmission(
        messages=(),
        images=(ImageInput.from_bytes(b"image", "image/png"),),
        videos=(),
        audios=(),
        max_new_tokens=8,
        sampling=SamplingParams(),
        request_id="gemma4-media",
    )
    resolved_images = SimpleNamespace(
        feature_counts=(2,),
        content_grids=((1, 1, 2),),
        embeddings=(tensorplay.zeros((2, 8)),),
    )
    prepared = build_media_admission(
        admission,
        tokenizer=_Tokenizer(),
        resolved_images=resolved_images,
        resolved_videos=None,
        resolved_audios=None,
        model_type="gemma4",
        spatial_merge_size=1,
        position_ids_per_second=None,
        device=tensorplay.device("cpu"),
    )
    assert prepared.state.rope_delta == 0
    assert prepared.state.mrope_positions.shape == (3, 4)
    assert prepared.state.media_positions == (1, 2)


def test_tokenizer_detects_gemma4_grammar() -> None:
    from pathlib import Path

    from megalm.runtime.tokenizer import ModelTokenizer

    roots = sorted(
        Path("/home/bluemoon/.cache/huggingface/hub").glob(
            "models--google--gemma-4-*/snapshots/*"
        )
    )
    if not roots:
        pytest.skip("no downloaded Gemma-4 checkpoint")
    tokenizer = ModelTokenizer(roots[0])
    assert tokenizer.output_grammar() == "gemma4"
    name, preserve = tokenizer.tool_call_parser()
    assert name == "gemma4"
    assert preserve


def test_hf_logits_parity() -> None:
    torch = pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")
    hf_cls = getattr(transformers, "Gemma4ForCausalLM", None)
    hf_cfg_cls = getattr(transformers, "Gemma4TextConfig", None)
    if hf_cls is None or hf_cfg_cls is None:
        pytest.skip("installed transformers has no Gemma-4 support")

    hf_cfg = hf_cfg_cls(**_payload())
    hf = hf_cls(hf_cfg).eval()
    model = _make_model()

    hf_sd = hf.state_dict()
    params = dict(model.named_parameters())
    bufs = dict(model.named_buffers())
    for name, param in params.items():
        if name == "lm_head_weight":
            continue
        if name.endswith(".gate_up_weight"):
            base = name[: -len(".gate_up_weight")]
            src = torch.cat(
                [
                    hf_sd[base + ".mlp.gate_proj.weight"],
                    hf_sd[base + ".mlp.up_proj.weight"],
                ],
                dim=0,
            )
        elif name.endswith(".down_weight"):
            src = hf_sd[
                name[: -len(".down_weight")] + ".mlp.down_proj.weight"
            ]
        elif name in hf_sd:
            src = hf_sd[name]
        else:
            src = hf_sd[name + ".weight"]
        assert tuple(src.shape) == tuple(param.shape), name
        param.copy_(
            tensorplay.tensor(
                src.flatten().tolist(), dtype=tensorplay.float32
            ).reshape(param.shape)
        )
    for name, buf in bufs.items():
        if name.endswith("layer_scalar"):
            src = hf_sd[name]
            buf.copy_(
                tensorplay.tensor(
                    src.flatten().tolist(), dtype=tensorplay.float32
                ).reshape(buf.shape)
            )

    ids = [1, 2, 3, 4, 5]
    with torch.no_grad():
        expected = hf(torch.tensor([ids]), use_cache=False).logits[0, -1]
    actual = model(_batch(ids), _cache(model))[0]
    diff = (actual - tensorplay.tensor(expected.tolist())).abs().max().item()
    assert diff < 1e-4, f"max logit diff {diff}"


def test_hf_moe_logits_parity() -> None:
    torch = pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")
    hf_cls = getattr(transformers, "Gemma4ForCausalLM", None)
    hf_cfg_cls = getattr(transformers, "Gemma4TextConfig", None)
    if hf_cls is None or hf_cfg_cls is None:
        pytest.skip("installed transformers has no Gemma-4 support")

    overrides = {
        "enable_moe_block": True,
        "num_experts": 4,
        "top_k_experts": 2,
        "moe_intermediate_size": 32,
    }
    hf_cfg = hf_cfg_cls(**_payload(**overrides))
    hf = hf_cls(hf_cfg).eval()
    model = _make_model(**overrides)

    hf_sd = hf.state_dict()
    params = dict(model.named_parameters())
    bufs = dict(model.named_buffers())
    moe_map = {
        ".moe.router_scale": ".router.scale",
        ".moe.router_proj": ".router.proj.weight",
        ".moe.per_expert_scale": ".router.per_expert_scale",
        ".moe.gate_up_weight": ".experts.gate_up_proj",
        ".moe.down_weight": ".experts.down_proj",
    }
    for name, param in params.items():
        if name == "lm_head_weight":
            continue
        hf_name = name if name.endswith(".weight") else name + ".weight"
        for suffix, source_suffix in moe_map.items():
            if name.endswith(suffix):
                hf_name = name[: -len(suffix)] + source_suffix
                break
        if name.endswith(".gate_up_weight") and ".moe." not in name:
            base = name[: -len(".gate_up_weight")]
            src = torch.cat(
                [
                    hf_sd[base + ".mlp.gate_proj.weight"],
                    hf_sd[base + ".mlp.up_proj.weight"],
                ],
                dim=0,
            )
        elif name.endswith(".down_weight") and ".moe." not in name:
            src = hf_sd[
                name[: -len(".down_weight")] + ".mlp.down_proj.weight"
            ]
        else:
            src = hf_sd[hf_name]
        assert tuple(src.shape) == tuple(param.shape), (name, hf_name)
        param.copy_(
            tensorplay.tensor(
                src.flatten().tolist(), dtype=tensorplay.float32
            ).reshape(param.shape)
        )
    for name, buf in bufs.items():
        if name.endswith("layer_scalar"):
            src = hf_sd[name]
            buf.copy_(
                tensorplay.tensor(
                    src.flatten().tolist(), dtype=tensorplay.float32
                ).reshape(buf.shape)
            )

    ids = [1, 2, 3, 4, 5]
    with torch.no_grad():
        expected = hf(torch.tensor([ids]), use_cache=False).logits[0, -1]
    actual = model(_batch(ids), _cache(model))[0]
    diff = (actual - tensorplay.tensor(expected.tolist())).abs().max().item()
    assert diff < 1e-4, f"max logit diff {diff}"
