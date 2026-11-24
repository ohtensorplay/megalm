import pytest
import tensorplay

from megalm.runtime.tp_sharding import (
    ShardError,
    ShardGeometry,
    TPWeightSharder,
)


def _values(shape):
    count = 1
    for dim in shape:
        count *= dim
    return tensorplay.arange(count, dtype=tensorplay.float32).reshape(shape)


GEO = ShardGeometry(
    hidden_size=8,
    query_width=16,
    kv_width=8,
    intermediate_size=12,
    vocab_size=32,
)


def test_tp1_is_identity():
    sharder = TPWeightSharder(GEO, 1, 0)
    tensor = _values((16, 8))
    assert sharder.shard("model.layers.0.self_attn.q_proj.weight", tensor) is tensor
    assert sharder.enabled is False


def test_column_and_row_slices():
    source = _values((16, 8))
    r0 = TPWeightSharder(GEO, 2, 0).shard("model.layers.0.self_attn.q_proj.weight", source)
    r1 = TPWeightSharder(GEO, 2, 1).shard("model.layers.0.self_attn.q_proj.weight", source)
    assert r0.shape == (8, 8)
    assert r1.shape == (8, 8)
    assert r0.tolist() == source[:8].tolist()
    assert r1.tolist() == source[8:].tolist()

    down = _values((8, 12))
    d0 = TPWeightSharder(GEO, 2, 0).shard("model.layers.0.mlp.down_proj.weight", down)
    d1 = TPWeightSharder(GEO, 2, 1).shard("model.layers.0.mlp.down_proj.weight", down)
    assert d0.tolist() == down[:, :6].tolist()
    assert d1.tolist() == down[:, 6:].tolist()


def test_packed_qkv_keeps_head_groups():
    source = _values((32, 8))  # q(16) + k(8) + v(8)
    r0 = TPWeightSharder(GEO, 2, 0).shard("model.layers.0.self_attn.qkv_proj.weight", source)
    r1 = TPWeightSharder(GEO, 2, 1).shard("model.layers.0.self_attn.qkv_proj.weight", source)
    assert r0.tolist() == tensorplay.cat((source[:8], source[16:20], source[24:28])).tolist()
    assert r1.tolist() == tensorplay.cat((source[8:16], source[20:24], source[28:])).tolist()


def test_packed_gate_up_keeps_halves():
    source = _values((24, 8))  # gate(12) + up(12)
    r0 = TPWeightSharder(GEO, 2, 0).shard("model.layers.0.mlp.gate_up_proj.weight", source)
    r1 = TPWeightSharder(GEO, 2, 1).shard("model.layers.0.mlp.gate_up_proj.weight", source)
    assert r0.tolist() == tensorplay.cat((source[:6], source[12:18])).tolist()
    assert r1.tolist() == tensorplay.cat((source[6:12], source[18:])).tolist()


def test_lm_head_shards_vocabulary():
    source = _values((32, 8))
    r0 = TPWeightSharder(GEO, 4, 2).shard("lm_head.weight", source)
    assert r0.tolist() == source[16:24].tolist()


def test_replicated_tensor_is_untouched():
    source = _values((8, 3))
    assert TPWeightSharder(GEO, 2, 1).shard(
        "model.layers.0.input_layernorm.weight", source
    ).tolist() == source.tolist()


def test_rejects_undivisible_width_and_bad_rank():
    with pytest.raises(ShardError, match="divisible"):
        TPWeightSharder(GEO, 3, 0).shard("model.layers.0.self_attn.q_proj.weight", _values((16, 8)))
    with pytest.raises(ShardError):
        TPWeightSharder(GEO, 2, 2)
