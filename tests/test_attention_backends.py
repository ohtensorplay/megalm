import math

import pytest
import tensorplay

from megalm.kernels.attention import (
    AttentionConfig,
    build_attention_backend,
)
from megalm.kernels.attention.sink import SinkAttention
from megalm.kernels.attention.sparse import (
    SparseIndexCache,
    attach_index_cache,
)
from megalm.kernels.attention.types import AttentionConfigError


def _values(shape, offset=0.0):
    count = 1
    for dim in shape:
        count *= dim
    values = tensorplay.arange(count, dtype=tensorplay.float32)
    return (values.reshape(shape) + offset) * 0.1


def _gather(cache, pages, seq_len):
    page_ids = pages[: math.ceil(seq_len / cache.shape[1])].long()
    return cache.index_select(0, page_ids).reshape(-1, *cache.shape[2:])[:seq_len]


def _repeat(tensor, factor):
    return tensor if factor == 1 else tensor.repeat_interleave(factor, dim=1)


def _dense_reference(query, keys, values, *, causal, sink=None):
    """Independent softmax reference: [T,H,D] x [L,H,D] -> [T,H,D]."""

    num_heads = query.shape[1]
    factor = num_heads // keys.shape[1]
    keys = _repeat(keys, factor)
    values = _repeat(values, factor)
    scale = query.shape[2] ** -0.5
    # scores[h, t, l]
    scores = (tensorplay.einsum("thd,lhd->htl", query, keys) * scale).unsqueeze(0)
    q_len = query.shape[0]
    k_len = keys.shape[0]
    if causal:
        q_pos = tensorplay.arange(k_len - q_len, k_len)
        k_pos = tensorplay.arange(k_len)
        allowed = k_pos[None, :] <= q_pos[:, None]
        scores = tensorplay.where(allowed[None], scores, tensorplay.full_like(scores, float("-inf")))
    if sink is not None:
        sink_col = sink.reshape(1, num_heads, 1, 1).expand(
            1, num_heads, q_len, 1
        )
        scores = tensorplay.cat((scores, sink_col), dim=-1)
    probs = tensorplay.softmax(scores, dim=-1)
    if sink is not None:
        probs = probs[:, :, :, :k_len]
    return tensorplay.einsum("htl,lhd->thd", probs.squeeze(0), values)


def _cache():
    # 4 pages x 4 tokens, 2 kv heads, head_dim 4
    key = _values((4, 4, 2, 4), 1.0)
    value = _values((4, 4, 2, 4), 9.0)
    block_tables = tensorplay.tensor([[0, 1], [2, 3]], dtype=tensorplay.int32)
    return key, value, block_tables


def test_config_validation_and_backend_ids():
    assert AttentionConfig(mode="causal").backend_id == "megalm.attention.causal_paged"
    assert AttentionConfig(mode="non_causal").capabilities == ("non_causal", "gqa")
    assert "sink" in AttentionConfig(mode="sink", sink_tokens=4).capabilities
    assert "sparse" in AttentionConfig(mode="sparse", sparse_top_k=8).capabilities
    with pytest.raises(AttentionConfigError):
        AttentionConfig(mode="sink")
    with pytest.raises(AttentionConfigError):
        AttentionConfig(mode="sparse")
    with pytest.raises(AttentionConfigError):
        AttentionConfig(mode="mla")


def test_non_causal_prefill_matches_dense_reference():
    key, value, tables = _cache()
    query = _values((4, 4, 4), 3.0)  # 2 sequences x 2 query tokens
    backend = build_attention_backend(AttentionConfig(mode="non_causal"))
    output = backend.prefill(
        query,
        key,
        value,
        tables,
        tensorplay.tensor([0, 2, 4], dtype=tensorplay.int32),
        tensorplay.zeros(4, dtype=tensorplay.int64),
        query_lengths=(2, 2),
        context_lengths=(2, 2),
        page_size=4,
    )
    for index, (start, ctx) in enumerate(((0, 2), (2, 2))):
        keys = _gather(key, tables[index], ctx + 2)
        values = _gather(value, tables[index], ctx + 2)
        expected = _dense_reference(query[start : start + 2], keys, values, causal=False)
        assert tensorplay.allclose(
            output[start : start + 2], expected, atol=1e-5
        )


def test_non_causal_decode_matches_dense_reference():
    key, value, tables = _cache()
    query = _values((1, 4, 4), 5.0)
    backend = build_attention_backend(AttentionConfig(mode="non_causal"))
    out = backend.decode(
        query,
        key,
        value,
        tables[:1],
        tensorplay.tensor([6], dtype=tensorplay.int32),
        page_size=4,
    )
    keys = _gather(key, tables[0], 6)
    values = _gather(value, tables[0], 6)
    expected = _dense_reference(query, keys, values, causal=False)
    assert tensorplay.allclose(out, expected, atol=1e-5)


def test_sink_matches_dense_reference_with_bias():
    key, value, tables = _cache()
    query = _values((2, 4, 4), 7.0)
    sink = (0.5, -0.25, 0.75, 0.0)
    backend = SinkAttention(sink)
    out = backend.prefill(
        query,
        key,
        value,
        tables[:1],
        tensorplay.tensor([0, 2], dtype=tensorplay.int32),
        tensorplay.zeros(2, dtype=tensorplay.int64),
        query_lengths=(2,),
        context_lengths=(2,),
        page_size=4,
    )
    keys = _gather(key, tables[0], 4)
    values = _gather(value, tables[0], 4)
    sink_tensor = tensorplay.tensor(sink, dtype=tensorplay.float32)
    expected = _dense_reference(query, keys, values, causal=True, sink=sink_tensor)
    assert tensorplay.allclose(out, expected, atol=1e-5)


def test_default_causal_path_binds_identical_providers_and_callables():
    """Performance guard: the default path must be byte-for-byte unchanged.

    Cold attention wiring must not add a wrapper, rebind, or extra branch to
    the existing causal kernels. Only non-default modes may swap callables.
    """

    import tensorplay

    from megalm.kernels.bindings import bind_kernel_plan
    from megalm.kernels.ops import KernelPolicy

    policy = KernelPolicy(use_triton=False)
    kwargs = dict(
        policy=policy,
        device=tensorplay.device("cpu"),
        dtype=tensorplay.float32,
    )
    baseline = bind_kernel_plan(**kwargs)
    wired = bind_kernel_plan(
        **kwargs,
        attention_config=AttentionConfig(mode="causal"),
        sink_logits=(0.5,),
    )
    assert baseline.linear_backend == wired.linear_backend
    assert baseline.prefill_backend == wired.prefill_backend
    for name in (
        "linear",
        "rms_norm",
        "fused_qkv_norm_rope_cache_write",
        "paged_prefill_attention",
        "paged_decode_attention",
    ):
        assert getattr(baseline.core, name) is getattr(wired.core, name), name


def test_non_default_mode_swaps_only_attention_callables():
    import tensorplay

    from megalm.kernels.bindings import bind_kernel_plan
    from megalm.kernels.ops import KernelPolicy

    policy = KernelPolicy(use_triton=False)
    kwargs = dict(
        policy=policy,
        device=tensorplay.device("cpu"),
        dtype=tensorplay.float32,
    )
    baseline = bind_kernel_plan(**kwargs)
    special = bind_kernel_plan(
        **kwargs, attention_config=AttentionConfig(mode="non_causal")
    )
    assert special.core.paged_decode_attention is not baseline.core.paged_decode_attention
    assert special.core.linear is baseline.core.linear
    assert special.core.rms_norm is baseline.core.rms_norm


def test_cold_discovery_and_binding_select_the_backend():
    from megalm.kernels.artifacts import discover_kernel_plan
    from megalm.kernels.bindings import bind_kernel_plan
    from megalm.kernels.ops import KernelPolicy

    policy = KernelPolicy(use_triton=False)
    config = AttentionConfig(mode="sink", sink_tokens=1)
    spec = discover_kernel_plan(
        policy,
        device=tensorplay.device("cpu"),
        dtype=tensorplay.float32,
        attention_config=config,
    )
    assert spec.attention_mode == "sink"
    assert spec.prefill_backend == "megalm.attention.sink_sdpa"
    bindings = bind_kernel_plan(
        policy,
        device=tensorplay.device("cpu"),
        dtype=tensorplay.float32,
        expected=spec,
        attention_config=config,
        sink_logits=(0.25,),
    )
    assert bindings.core.paged_prefill_attention is not None
    assert bindings.core.paged_decode_attention is not None
    assert bindings.digest == spec.digest


def test_planned_decode_backend_and_capabilities_follow_the_mode():
    from tests.test_model_and_plan import _tp_plan_kwargs

    from megalm.planning import build_execution_plan

    plan = build_execution_plan(
        **_tp_plan_kwargs(
            prefill_backend="megalm.attention.non_causal_sdpa",
        )
    )
    attention = {
        kernel.operator: kernel
        for kernel in plan.kernels
        if kernel.operator in {"prefill_attention", "paged_decode_attention"}
    }
    assert attention["prefill_attention"].backend == "megalm.attention.non_causal_sdpa"
    assert attention["paged_decode_attention"].backend == "megalm.attention.non_causal_sdpa"
    assert "non_causal" in attention["prefill_attention"].capabilities


def test_mla_decode_matches_expanded_latent_reference():
    from megalm.kernels.attention import MLACache, MLAWeights, build_mla_attention

    rank, rope, nope, heads = 4, 2, 3, 2
    w_uk = _values((rank, heads, nope), 1.0)
    w_uv = _values((rank, heads, nope), 5.0)
    weights = MLAWeights(w_uk, w_uv)
    backend = build_mla_attention(
        kv_lora_rank=rank,
        qk_rope_head_dim=rope,
        qk_nope_head_dim=nope,
        weights=weights,
    )
    cache = MLACache(
        1, 1, 4, rank, rope, dtype=tensorplay.float32, device=tensorplay.device("cpu")
    )
    latent = _values((4, 1, rank + rope), 2.0)
    cache.layer(0)[:] = latent
    query = _values((1, heads, nope + rope), 3.0)
    out = backend.decode(
        query,
        cache.layer(0),
        cache.layer(0),
        tensorplay.tensor([[0]], dtype=tensorplay.int32),
        tensorplay.tensor([4], dtype=tensorplay.int32),
        page_size=4,
    )
    kv_c = latent[:, 0, :rank]
    k_pe = latent[:, 0, rank:]
    keys = tensorplay.cat(
        (
            tensorplay.einsum("lr,rhd->lhd", kv_c, w_uk),
            k_pe.unsqueeze(1).expand(-1, heads, -1),
        ),
        dim=-1,
    )
    values = tensorplay.einsum("lr,rhd->lhd", kv_c, w_uv)
    expected = _dense_reference(query, keys, values, causal=False)
    assert tensorplay.allclose(out, expected, atol=1e-5)


def test_sparse_topk_selector_is_fixed_shape_with_forced_pages():
    from megalm.kernels.attention.sparse_select import SparseTopKSelector

    scores = tensorplay.tensor(
        [[0.1, 0.9, 0.2, 0.8, 0.3, 0.7]], dtype=tensorplay.float32
    )
    tables = tensorplay.tensor([[10, 11, 12, 13, 14, 15]], dtype=tensorplay.int32)
    selected = SparseTopKSelector(top_k=2)(scores, tables)
    # top_k + 2 = 4, with sink (10) and tail (15) forced and results sorted.
    assert tuple(selected.shape) == (1, 4)
    assert selected[0].tolist() == [10, 11, 13, 15]


def test_sparse_attends_only_to_selected_pages():
    key, value, tables = _cache()
    query = _values((1, 4, 4), 11.0)
    cache = SparseIndexCache(_values((4, 4, 4), 13.0))
    attach_index_cache(key, cache)
    backend = build_attention_backend(
        AttentionConfig(mode="sparse", sparse_top_k=1)
    )
    out = backend.decode(
        query,
        key,
        value,
        tables[:1],
        tensorplay.tensor([8], dtype=tensorplay.int32),
        page_size=4,
    )
    assert out.shape == query.shape
    assert bool(tensorplay.isfinite(out).all())
