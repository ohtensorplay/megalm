from types import SimpleNamespace

import pytest
import tensorplay

from megalm.kernels.inkling2 import (
    block_attention_mix,
    l2_clip,
    log_scale_tau,
    message_gate,
    residual_sum,
    rms_norm,
)
from megalm.kernels.inkling2_depth_residual import depth_residual_mix
from megalm.kernels.inkling2_head_prep import prepare_inkling2_heads
from megalm.kernels.inkling2_paged_decode import inkling2_paged_decode
from megalm.kernels.inkling2_sconv import short_convolution_decode
from megalm.kernels.quant_fp import (
    e2m1_lut,
    encode_nvfp4,
    nvfp4_outer_scale,
    roundtrip_mxfp8,
    roundtrip_nvfp4,
    unpack_mxfp8,
    unpack_nvfp4,
)
from megalm.models.inkling2.model import Inkling2ForCausalLM
from megalm.models.inkling2.model import l2_clip as model_l2_clip
from megalm.models.inkling2.model import rms_norm as model_rms_norm
from megalm.runtime.batch import BatchPhase, ExecutionBatch


def test_residual_sum_rounds_once_to_first_dtype():
    a = tensorplay.tensor([[1.0, 2.0]], dtype=tensorplay.bfloat16)
    b = tensorplay.tensor([[0.5, 0.25]], dtype=tensorplay.bfloat16)
    out = residual_sum((a, b))
    assert out.dtype == tensorplay.bfloat16
    expected = (a.float() + b.float()).to(tensorplay.bfloat16)
    assert out.reshape(-1).tolist() == expected.reshape(-1).tolist()


def test_rms_norm_matches_unweighted_definition():
    x = tensorplay.tensor([[3.0, 4.0], [1.0, 1.0]], dtype=tensorplay.float32)
    out = rms_norm(x, None, 1e-6, output_dtype=tensorplay.float32)
    # Rows are [3,4] -> rms 3.5355 and [1,1] -> rms 1.
    assert abs(out.reshape(-1)[0].item() - 3.0 / 3.5355) < 1e-3
    assert abs(out.reshape(-1)[3].item() - 1.0) < 1e-3


def test_l2_clip_scales_only_long_rows():
    x = tensorplay.tensor([[3.0, 4.0], [0.3, 0.4]], dtype=tensorplay.float32)
    out = l2_clip(x, 1.0)
    rows = out.reshape(2, 2).tolist()
    norms = [sum(value * value for value in row) ** 0.5 for row in rows]
    assert abs(norms[0] - 1.0) < 1e-5
    assert abs(norms[1] - 0.5) < 1e-5


def test_log_scale_tau_is_one_below_floor():
    positions = tensorplay.tensor([0, 1, 8191, 8192, 16383], dtype=tensorplay.int64)
    tau = log_scale_tau(positions, 8192, 0.1)
    values = tau.reshape(-1).tolist()
    assert abs(values[0] - 1.0) < 1e-6
    assert abs(values[2] - 1.0) < 1e-6
    # (8192 + 1) / 8192 is just above 1, so tau is a hair above 1.
    assert abs(values[3] - 1.0) < 1e-4
    assert values[4] > values[3]


def test_block_attention_mix_shapes_and_normalisation():
    # Source 0 is [1000, 0] per row (RMS ~ 707 -> normalised [1, 0]).
    # Source 1 (the partial) is [0, 1000] per row (normalised [0, 1]).
    # The query dot product strongly favours source 0.
    blocks = tensorplay.tensor(
        [[[1000.0, 0.0], [1000.0, 0.0]]],
        dtype=tensorplay.float32,
    )
    partial = tensorplay.tensor([[0.0, 1000.0]], dtype=tensorplay.float32)
    queries = tensorplay.tensor([[1000.0, 0.0]], dtype=tensorplay.float32)
    out = block_attention_mix(blocks, partial, queries)
    assert tuple(out.shape) == (1, 2, 2)
    # The query selects source 0; RMS-normalised values are [1, 0].
    values = out.reshape(-1).tolist()
    # RMSNorm of [1000, 0] is sqrt(2) * [1, 0] (no gain), so the mix is
    # [1.414, 0] for both token rows.
    assert values[0] > 1.4
    assert abs(values[1]) < 1e-3
    assert abs(values[2] - 1.4142) < 1e-3
    assert abs(values[3]) < 1e-3


def test_message_gate_shapes_and_convexity():
    destination = tensorplay.tensor([[1.0, 2.0]], dtype=tensorplay.float32)
    message = tensorplay.tensor([[3.0, 4.0]], dtype=tensorplay.float32)
    down = tensorplay.zeros((4, 4), dtype=tensorplay.bfloat16)
    forget = tensorplay.zeros((2, 4), dtype=tensorplay.bfloat16)
    gain = tensorplay.zeros((2, 4), dtype=tensorplay.bfloat16)
    out = message_gate(
        destination,
        message,
        down_weight=down,
        forget_weight=forget,
        gain_weight=gain,
    )
    # tanh(0) = 0 -> gate is identity on the destination.
    assert out.reshape(-1).tolist() == pytest.approx([1.0, 2.0], abs=1e-5)


def test_e2m1_lut_order():
    lut = e2m1_lut()
    assert lut.reshape(-1).tolist() == [
        0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
        -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0,
    ]


def test_nvfp4_roundtrip_within_one_step():
    values = tensorplay.tensor(
        [[0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
          -0.5, -1.5, -6.0, 5.0, -3.0, 0.0, 1.0, 2.0]],
        dtype=tensorplay.float32,
    )
    amax = 6.0
    packed, scales, outer = encode_nvfp4(values, amax)
    assert tuple(packed.shape) == (1, 8)
    assert tuple(scales.shape) == (1, 1)
    # The checkpoint's block scales reach the E4M3 maximum 448, so
    # amax = 6 gives scale 448 exactly (ModelOpt's 448 * 6 = 2688 outer).
    assert scales.reshape(-1)[0].item() == 448.0
    assert abs(float(outer.reshape(-1)[0].item()) - amax / 2688.0) < 1e-9
    assert abs(nvfp4_outer_scale(amax, 2688.0) - amax / 2688.0) < 1e-9
    decoded = unpack_nvfp4(packed, scales)
    # Code values (before the outer factor) are the E2M1 grid times 448.
    assert decoded.reshape(-1).tolist()[1] == 224.0
    restored = roundtrip_nvfp4(values, amax)
    assert (restored - values).abs().max().item() <= 1.0 + 1e-5


def test_mxfp8_roundtrip_is_exact_for_representable_values():
    values = tensorplay.tensor(
        [[float(index) for index in range(32)]],
        dtype=tensorplay.float32,
    )
    restored = roundtrip_mxfp8(values)
    # UE8M0 scales are powers of two, so integers 0..15 are exact.
    assert (restored.reshape(-1)[:16] - values.reshape(-1)[:16]).abs().max().item() == 0.0
    # 16..31 quantise to a coarser grid; they stay within one E4M3 step.
    assert (restored.reshape(-1)[16:] - values.reshape(-1)[16:]).abs().max().item() <= 1.0


def test_mxfp8_scale_decode_special_case():
    scales = tensorplay.tensor([[127, 128, 255]], dtype=tensorplay.uint8)
    values = tensorplay.zeros((1, 96), dtype=tensorplay.float32)
    decoded = unpack_mxfp8(values, scales)
    assert tuple(decoded.shape) == (1, 96)


@pytest.mark.skipif(not tensorplay.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("dtype", [tensorplay.bfloat16, tensorplay.float16])
@pytest.mark.parametrize("use_dynamic", [False, True])
def test_short_convolution_decode_graph_preserves_request_histories(dtype, use_dynamic):
    tensorplay.manual_seed(12)
    options = {"device": "cuda", "dtype": dtype}
    channels, taps_count = 131, 4
    x = tensorplay.randn((3, channels), **options)
    taps = tensorplay.randn((channels, taps_count), **options)
    dynamic = (
        tensorplay.randn((3, channels, taps_count), **options)
        if use_dynamic
        else None
    )
    slots = tensorplay.zeros(3, device=x.device, dtype=tensorplay.int32)
    positions = tensorplay.zeros(3, device=x.device, dtype=tensorplay.int64)
    active = tensorplay.zeros(3, device=x.device, dtype=tensorplay.bool)
    state = tensorplay.randn((3, channels, taps_count - 1), **options)
    expected_state = state.clone()

    short_convolution_decode(x, taps, state, slots, positions, active, dynamic)
    graph = tensorplay.cuda.CUDAGraph()
    with tensorplay.cuda.graph(graph):
        output = short_convolution_decode(
            x, taps, state, slots, positions, active, dynamic
        )

    for slot_ids, offsets in (((2, 0, 0), (4, 0, 7)), ((0, 2, 2), (1, 5, 5))):
        slots.copy_(tensorplay.tensor(slot_ids, device=x.device, dtype=slots.dtype))
        positions.copy_(tensorplay.tensor(offsets, device=x.device))
        active[:2].fill_(True)
        graph.replay()

        for row, slot in enumerate(slot_ids[:2]):
            expected_history = expected_state[slot : slot + 1]
            if offsets[row] == 0:
                expected_history.zero_()
            expected_output = Inkling2ForCausalLM._sconv_segment(
                None,
                x[row : row + 1],
                taps,
                state=expected_history,
                dynamic=None if dynamic is None else dynamic[row : row + 1],
            )
            assert tensorplay.equal(output[row : row + 1], expected_output)
        assert tensorplay.equal(state, expected_state)
        assert tensorplay.count_nonzero(output[2]).item() == 0


@pytest.mark.skipif(not tensorplay.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize(
    "window,scale,num_pages,kv_heads,lengths_to_test",
    [
        (0, 128**-0.5, 4, 2, (27, 52)),
        (19, 1 / 128, 4, 2, (27, 52)),
        (0, 128**-0.5, 2048, 8, (8193, 16379, 32761)),
    ],
)
def test_paged_relative_decode_graph_matches_dense_attention(
    window, scale, num_pages, kv_heads, lengths_to_test
):
    tensorplay.manual_seed(9)
    options = {"device": "cuda", "dtype": tensorplay.bfloat16}
    heads = 3 * kv_heads
    rel_extent = 16 if num_pages == 4 else 1024
    query = tensorplay.randn((2, heads, 128), **options)
    keys = tensorplay.randn((num_pages, 16, kv_heads, 128), **options)
    values = tensorplay.randn_like(keys)
    relative = tensorplay.randn((2, heads, rel_extent), **options)
    sink = tensorplay.randn((2, heads), **options)
    page_order = tensorplay.arange(num_pages, device=query.device, dtype=tensorplay.int32)
    tables = tensorplay.stack((page_order.flip((0,)), page_order))
    length = lengths_to_test[0]
    lengths = tensorplay.tensor([length, 1], device=query.device, dtype=tensorplay.int32)
    positions = tensorplay.tensor([length - 1, 0], device=query.device)
    active = tensorplay.tensor([True, False], device=query.device)
    args = (query, keys, values, relative, sink, tables, lengths, positions, active)
    inkling2_paged_decode(*args, window=window, scale=scale)
    graph = tensorplay.cuda.CUDAGraph()
    with tensorplay.cuda.graph(graph):
        result = inkling2_paged_decode(*args, window=window, scale=scale)

    pages = tables[0].long()
    page_offsets = tensorplay.arange(16, device=query.device)
    slots = (pages[:, None] * 16 + page_offsets[None, :]).reshape(-1)
    logical = tensorplay.arange(num_pages * 16, device=query.device)

    def dense_cache(cache):
        dense = cache.reshape(-1, kv_heads, 128).index_select(0, slots)
        return dense.transpose(0, 1).repeat_interleave(3, dim=0)

    dense_keys = dense_cache(keys)
    dense_values = dense_cache(values)
    for length in lengths_to_test:
        lengths[0].fill_(length)
        positions[0].fill_(length - 1)
        graph.replay()
        logits = (query[:1].transpose(0, 1) @ dense_keys.transpose(1, 2)) * scale
        distance = length - 1 - logical
        bias = relative[0].index_select(1, distance.clamp(0, rel_extent - 1))
        bias = tensorplay.where((distance >= 0) & (distance < rel_extent), bias, 0)
        start = max(0, length - window) if window else 0
        mask = (logical < length) & (logical >= start)
        logits = (logits + bias[:, None, :]).masked_fill(~mask, float("-inf"))
        with_sink = tensorplay.cat(
            (logits.float(), sink[0, :, None, None].float()), dim=-1
        )
        probabilities = tensorplay.softmax(with_sink, dim=-1)[..., :-1].to(query.dtype)
        expected = (probabilities @ dense_values).transpose(0, 1)

        tensorplay.testing.assert_close(result[:1], expected, atol=1e-3, rtol=1e-2)
        assert tensorplay.count_nonzero(result[1]).item() == 0


@pytest.mark.skipif(not tensorplay.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize(
    "tokens,queries_count,round_candidate,normalize_values,apply_norm",
    [
        (1, 1, False, True, True),
        (1, 3, True, True, True),
        (1, 4, True, False, False),
        (3, 3, True, True, True),
        (4, 4, False, True, False),
    ],
)
def test_fused_depth_residual_replays_and_preserves_boundary(
    tokens, queries_count, round_candidate, normalize_values, apply_norm
):
    tensorplay.manual_seed(12)
    options = {"device": "cuda", "dtype": tensorplay.bfloat16}
    width = 2304
    blocks = tensorplay.randn((3, tokens, width), **options)
    terms = tuple(tensorplay.randn((tokens, width), **options) for _ in range(3))
    queries = tensorplay.randn((queries_count, width), device="cuda") * 0.02
    candidate = tensorplay.empty_like(terms[0]) if round_candidate else None
    norms = (
        tuple(tensorplay.randn(width, **options) for _ in range(queries_count))
        if apply_norm
        else None
    )
    args = (blocks, queries, terms, candidate, norms)
    kwargs = {"normalize_values": normalize_values, "eps": 1e-6}
    depth_residual_mix(*args, **kwargs)
    graph = tensorplay.cuda.CUDAGraph()
    with tensorplay.cuda.graph(graph):
        actual = depth_residual_mix(*args, **kwargs)
    assert actual is not None

    for _ in range(2):
        terms[0].mul_(0.75)
        graph.replay()
        partial = terms[0].float() + terms[1].float() + terms[2].float()
        if candidate is not None:
            rounded = partial.to(candidate.dtype)
            assert tensorplay.equal(candidate, rounded)
            partial = rounded.float()
        sources = tensorplay.cat((blocks.float(), partial.unsqueeze(0)), dim=0)
        normalized = model_rms_norm(sources)
        logits = tensorplay.einsum("qh,sth->qst", queries, normalized)
        weights = tensorplay.softmax(logits, dim=1)
        values = normalized if normalize_values else sources
        expected = tensorplay.einsum("qst,sth->qth", weights, values)
        if norms is not None:
            expected = tensorplay.stack(
                [
                    model_rms_norm(expected[i], norm, output_dtype=norm.dtype)
                    for i, norm in enumerate(norms)
                ]
            )
            tensorplay.testing.assert_close(actual, expected, atol=2e-5, rtol=8e-3)
        else:
            tensorplay.testing.assert_close(actual, expected, atol=2e-6, rtol=2e-6)



@pytest.mark.skipif(not tensorplay.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("use_tau", [False, True])
@pytest.mark.parametrize("dtype", [tensorplay.bfloat16, tensorplay.float16])
def test_fused_attention_heads_match_the_rounded_reference(use_tau, dtype):
    tensorplay.manual_seed(61)
    options = {"device": "cuda", "dtype": dtype}
    q = tensorplay.randn((1, 24, 128), **options) * 2.5
    k = tensorplay.randn((1, 8, 128), **options) * 2.5
    v = tensorplay.randn((1, 8, 128), **options) * 2.5
    relative = tensorplay.randn((1, 24, 16), **options) * 5
    query_gain = tensorplay.rand((24, 128), **options) + 0.8
    relative_gain = tensorplay.rand((24, 16), **options) + 0.8
    tau = tensorplay.tensor([1.12], device="cuda") if use_tau else None
    qk_threshold = 22.627416997969522
    rel_threshold = 8.0

    expected_q = model_l2_clip(q * query_gain, qk_threshold)
    expected_k = model_l2_clip(k, qk_threshold)
    expected_v = model_rms_norm(v)
    expected_relative = model_l2_clip(relative * relative_gain, rel_threshold)
    if tau is not None:
        expected_q = (expected_q.float() * tau.reshape(1, 1, 1)).to(dtype)
        expected_relative = (expected_relative.float() * tau.reshape(1, 1, 1)).to(dtype)

    actual = prepare_inkling2_heads(
        q, k, v, relative, query_gain, relative_gain, tau,
        qk_threshold=qk_threshold, rel_threshold=rel_threshold, eps=1e-6,
    )
    for result, expected in zip(
        actual, (expected_q, expected_k, expected_v, expected_relative), strict=True
    ):
        assert tensorplay.equal(result, expected)


@pytest.mark.skipif(not tensorplay.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("local", [False, True])
def test_inkling_prefill_prefix_does_not_read_future_keys(local):
    width = 16
    dtype = tensorplay.bfloat16
    device = "cuda"
    identity = tensorplay.eye(width, device=device, dtype=dtype)
    layer_spec = SimpleNamespace(
        num_heads=1,
        num_kv_heads=1,
        head_dim=width,
        d_rel=4,
        rel_extent=8,
        is_local=local,
        sconv_kernel_size=4,
        qk_l2_clip_threshold=32.0,
        rel_l2_clip_threshold=8.0,
    )
    gain = tensorplay.ones((1, width), device=device, dtype=dtype)
    layer = SimpleNamespace(
        index=0,
        wq=identity,
        wk=identity,
        wv=identity,
        wr=identity[:4],
        wo=identity,
        gate=identity,
        sink=identity[:1],
        q_clip=gain,
        rel_clip=gain[:, :4],
        rel_proj=tensorplay.zeros((4, 8), device=device, dtype=dtype),
        k_sconv=None,
        v_sconv=None,
        k_sconv_proj=None,
    )
    model = SimpleNamespace(
        inkling=SimpleNamespace(layers=(layer_spec,), sliding_window_size=2),
        eps=1e-6,
        device=tensorplay.device(device),
        attn_scale="sqrt",
        _sconv=lambda x, *_args, **_kwargs: x,
    )
    positions = tensorplay.arange(3, device=device)
    batch = ExecutionBatch(
        input_ids=positions,
        positions=positions,
        slot_mapping=positions,
        block_tables=tensorplay.tensor([[0, 1, -1]], device=device, dtype=tensorplay.int32),
        sequence_lengths=tensorplay.tensor([3], device=device, dtype=tensorplay.int32),
        token_active=tensorplay.ones(3, device=device, dtype=tensorplay.bool),
        query_start_loc=(0, 3),
        query_lengths=(3,),
        context_lengths=(0,),
        phases=(BatchPhase.PREFILL,),
        sample_indices=positions[2:],
        page_size=2,
    )
    keys = tensorplay.zeros((2, 2, 1, width), device=device, dtype=dtype)
    values = tensorplay.zeros_like(keys)
    cache = SimpleNamespace(layer=lambda _visit: (keys, values))
    hidden = identity[:3].contiguous()
    different_suffix = hidden.clone()
    different_suffix[2].fill_(2)
    reference = Inkling2ForCausalLM._attention(
        model, layer, (hidden,), visit=0, batch=batch, cache=cache, tau=None
    )
    dense_rows = {}
    planned = Inkling2ForCausalLM._attention(
        model,
        layer,
        (hidden,),
        visit=0,
        batch=batch,
        cache=cache,
        tau=None,
        dense_rows=dense_rows,
    )
    assert tensorplay.equal(reference, planned)
    rows = next(iter(dense_rows.values()))
    changed = Inkling2ForCausalLM._attention(
        model,
        layer,
        (different_suffix,),
        visit=0,
        batch=batch,
        cache=cache,
        tau=None,
        dense_rows=dense_rows,
    )
    assert next(iter(dense_rows.values())) is rows
    assert tensorplay.equal(reference[:2], changed[:2])
    assert not tensorplay.equal(reference[2], changed[2])
