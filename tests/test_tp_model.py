import tensorplay

from megalm.kernels.ops import KernelPolicy
from megalm.models.generic_causal_lm.config import GenericCausalLMConfig
from megalm.models.generic_causal_lm.model import (
    GenericCausalLM,
    GenericDenseMLP,
)
from megalm.runtime.collectives import RecordingCollectiveGroup


def _config(**overrides: object) -> GenericCausalLMConfig:
    payload = dict(
        model_type="llama",
        vocab_size=256,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        max_position_embeddings=128,
        rms_norm_eps=1e-6,
        rope_theta=10000.0,
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
    payload.update(overrides)
    return GenericCausalLMConfig(**payload)  # type: ignore[arg-type]


def test_tp1_model_keeps_full_geometry():
    model = GenericCausalLM(
        _config(),
        device="cpu",
        dtype=tensorplay.float32,
        kernel_policy=KernelPolicy(use_triton=False),
    )
    layer = model.model.layers[0]
    assert layer.self_attn.qkv_proj.weight.shape == (64, 32)  # q32 + k16 + v16
    assert layer.self_attn.o_proj.weight.shape == (32, 32)
    assert layer.mlp.gate_up_weight.shape == (128, 32)
    assert layer.mlp.down_proj.weight.shape == (32, 64)
    assert model.lm_head_weight is not None
    assert model.lm_head_weight.shape == (256, 32)
    assert model.tp_size == 1


def test_tp2_model_builds_rank_local_geometry():
    fake = RecordingCollectiveGroup(rank=1, world_size=2)
    model = GenericCausalLM(
        _config(),
        device="cpu",
        dtype=tensorplay.float32,
        kernel_policy=KernelPolicy(use_triton=False),
        tp_size=2,
        tp_rank=1,
        collective=fake,
    )
    layer = model.model.layers[0]
    assert layer.self_attn.num_query_heads == 2
    assert layer.self_attn.num_kv_heads == 1
    assert layer.self_attn.qkv_proj.weight.shape == (32, 32)  # q16 + k8 + v8
    assert layer.self_attn.o_proj.weight.shape == (32, 16)
    assert layer.mlp.gate_up_weight.shape == (64, 32)
    assert layer.mlp.down_proj.weight.shape == (32, 32)
    assert model.lm_head_weight is not None
    assert model.lm_head_weight.shape == (128, 32)
    assert model._shard_lm_head is True
    assert model._all_reduce == fake.all_reduce


def test_tied_head_is_not_sharded():
    model = GenericCausalLM(
        _config(tie_word_embeddings=True),
        device="cpu",
        dtype=tensorplay.float32,
        kernel_policy=KernelPolicy(use_triton=False),
        tp_size=2,
        tp_rank=0,
    )
    assert model.lm_head_weight is None
    assert model._shard_lm_head is False


def test_dense_mlp_calls_all_reduce_after_down_projection():
    fake = RecordingCollectiveGroup(rank=0, world_size=2)
    model = GenericCausalLM(
        _config(),
        device="cpu",
        dtype=tensorplay.float32,
        kernel_policy=KernelPolicy(use_triton=False),
        tp_size=2,
        tp_rank=0,
        collective=fake,
    )
    mlp = model.model.layers[0].mlp
    assert isinstance(mlp, GenericDenseMLP)
    hidden = tensorplay.zeros((3, 32), dtype=tensorplay.float32)
    mlp(hidden, decode=False)
    assert fake.all_reduce_calls == [0]
