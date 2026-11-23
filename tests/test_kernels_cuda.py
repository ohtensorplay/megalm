import pytest
import tensorplay
import tensorplay.nn.functional as F

from megalm.kernels import fallback
from megalm.kernels.batch_invariant_linear import batch_invariant_linear
from megalm.kernels.binding_types import (
    TENSORPLAY_SDPA_PAGED_PREFILL_BACKEND,
    TRITON_BATCH_INVARIANT_LINEAR_BACKEND,
    TRITON_PAGED_PREFILL_BACKEND,
)
from megalm.kernels.bindings import bind_kernel_plan, discover_kernel_plan
from megalm.kernels.dflash_query import dflash_query_setup_into
from megalm.kernels.greedy_argmax import greedy_argmax
from megalm.kernels.gdn import causal_conv_update
from megalm.kernels.moe_decode import routed_moe_decode
from megalm.kernels.moe_grouped import routed_moe_grouped_tensorcore
from megalm.kernels.moe_top1 import moe_top1, moe_top1_from_hidden
from megalm.kernels.moe_topk import moe_topk_softmax
from megalm.kernels.paged_decode import paged_decode_attention
from megalm.kernels.paged_prefill import paged_prefill_attention
from megalm.kernels.paged_prefill_sdpa import (
    paged_prefill_attention as sdpa_paged_prefill_attention,
)
from megalm.kernels.rms_norm import (
    fused_add_offset_rms_norm,
    fused_add_rms_norm,
    offset_rms_norm,
    rms_norm,
)
from megalm.kernels.rotary_cache import (
    fused_qkv_norm_rope_cache_write,
    rope_and_cache_write,
)
from megalm.kernels.silu_mul import silu_and_mul
from megalm.kernels.speculative_reference import (
    dflash_query_setup_into as dflash_query_setup_reference,
)
from megalm.kernels.speculative_reference import (
    greedy_chain_verify_into as greedy_chain_verify_reference,
)
from megalm.kernels.speculative_verify import greedy_chain_verify_into
from megalm.kernels.ops import (
    GroupedMoEWorkspace,
    KernelPolicy,
    routed_moe_tensorcore,
)


pytestmark = pytest.mark.skipif(
    not tensorplay.cuda.is_available(),
    reason="handwritten kernel tests require CUDA",
)


def test_rms_norm_and_silu_match_fallback() -> None:
    tensorplay.manual_seed(1)
    x = tensorplay.randn(7, 768, device="cuda", dtype=tensorplay.float16)
    weight = tensorplay.randn(768, device="cuda", dtype=tensorplay.float16)
    tensorplay.testing.assert_close(
        rms_norm(x, weight, 1e-6),
        fallback.rms_norm(x, weight, 1e-6),
        atol=2e-3,
        rtol=2e-3,
    )
    residual = tensorplay.randn_like(x)
    expected_residual = residual + x
    fused_input = x.clone()
    fused_residual = residual.clone()
    fused, updated_residual = fused_add_rms_norm(
        fused_input,
        fused_residual,
        weight,
        1e-6,
    )
    assert fused.data_ptr() == fused_input.data_ptr()
    assert updated_residual.data_ptr() == fused_residual.data_ptr()
    tensorplay.testing.assert_close(
        updated_residual,
        expected_residual,
        atol=2e-3,
        rtol=2e-3,
    )
    tensorplay.testing.assert_close(
        fused,
        fallback.rms_norm(expected_residual, weight, 1e-6),
        atol=2e-3,
        rtol=2e-3,
    )
    gate_up = tensorplay.randn(7, 4864, device="cuda", dtype=tensorplay.float16)
    tensorplay.testing.assert_close(
        silu_and_mul(gate_up),
        fallback.silu_and_mul(gate_up),
        atol=2e-3,
        rtol=2e-3,
    )


def test_gdn_causal_conv_update_tiles_channels_and_masks_padding() -> None:
    tensorplay.manual_seed(43)
    batch = 3
    channels = 6_144
    history = 3
    mixed = tensorplay.randn(
        batch,
        channels,
        device="cuda",
        dtype=tensorplay.bfloat16,
    )
    weight = tensorplay.randn(
        channels,
        1,
        history + 1,
        device="cuda",
        dtype=tensorplay.bfloat16,
    )
    initial = tensorplay.randn(
        5,
        channels,
        history,
        device="cuda",
        dtype=tensorplay.bfloat16,
    )
    state_indices = tensorplay.tensor(
        [3, 1, 4],
        device="cuda",
        dtype=tensorplay.int32,
    )
    active = tensorplay.tensor(
        [True, False, True],
        device="cuda",
        dtype=tensorplay.bool,
    )

    expected_state = initial.clone()
    selected = initial.index_select(0, state_indices.long())
    combined = tensorplay.cat((selected, mixed.unsqueeze(-1)), dim=-1)
    expected = tensorplay.nn.functional.silu(
        (
            combined
            * weight[:, 0, :].unsqueeze(0)
        ).sum(dim=-1)
    )
    expected[~active] = 0
    expected_state[state_indices[active].long()] = combined[active, :, 1:]

    actual_state = initial.clone()
    actual = causal_conv_update(
        mixed,
        actual_state,
        weight,
        state_indices,
        active,
    )

    tensorplay.testing.assert_close(actual, expected, atol=2e-2, rtol=2e-2)
    tensorplay.testing.assert_close(actual_state, expected_state, atol=0, rtol=0)


def test_offset_rms_norm_and_fused_residual_match_fallback() -> None:
    tensorplay.manual_seed(44)
    hidden = tensorplay.randn(
        7,
        1024,
        device="cuda",
        dtype=tensorplay.bfloat16,
    )
    residual = tensorplay.randn_like(hidden)
    weight = tensorplay.randn(
        1024,
        device="cuda",
        dtype=tensorplay.bfloat16,
    )
    expected = fallback.offset_rms_norm(hidden, weight, 1e-6)
    tensorplay.testing.assert_close(
        offset_rms_norm(hidden, weight, 1e-6),
        expected,
        atol=2e-2,
        rtol=2e-2,
    )

    expected_residual = residual + hidden
    expected_normalized = fallback.offset_rms_norm(
        expected_residual,
        weight,
        1e-6,
    )
    actual_hidden = hidden.clone()
    actual_residual = residual.clone()
    actual_normalized, actual_updated = fused_add_offset_rms_norm(
        actual_hidden,
        actual_residual,
        weight,
        1e-6,
    )
    tensorplay.testing.assert_close(actual_updated, expected_residual, atol=0, rtol=0)
    tensorplay.testing.assert_close(
        actual_normalized,
        expected_normalized,
        atol=2e-2,
        rtol=2e-2,
    )


def test_batch_invariant_linear_is_exact_across_row_geometry() -> None:
    tensorplay.manual_seed(41)
    row = tensorplay.randn(1, 2048, device="cuda", dtype=tensorplay.float16)
    prefix = tensorplay.randn(7, 2048, device="cuda", dtype=tensorplay.float16)
    suffix = tensorplay.randn(19, 2048, device="cuda", dtype=tensorplay.float16)
    packed = tensorplay.cat((prefix, row, suffix), dim=0)
    weight = tensorplay.randn(513, 2048, device="cuda", dtype=tensorplay.float16)
    bias = tensorplay.randn(513, device="cuda", dtype=tensorplay.float16)

    single = batch_invariant_linear(row, weight, bias)
    packed_output = batch_invariant_linear(packed, weight, bias)

    tensorplay.testing.assert_close(single, packed_output[7:8], atol=0, rtol=0)
    tensorplay.testing.assert_close(
        packed_output,
        F.linear(packed, weight, bias),
        atol=2e-2,
        rtol=2e-2,
    )


def test_dflash_query_setup_matches_reference() -> None:
    batch = 3
    speculative_tokens = 4
    query_tokens = speculative_tokens + 1
    last_ids = tensorplay.tensor([7, 11, 13], device="cuda", dtype=tensorplay.int64)
    last_positions = tensorplay.tensor(
        [15, 30, 47],
        device="cuda",
        dtype=tensorplay.int64,
    )
    block_tables = tensorplay.tensor(
        [
            [10, 11, 12, 13, 14, 15],
            [20, 21, 22, 23, 24, 25],
            [30, 31, 32, 33, 34, 35],
        ],
        device="cuda",
        dtype=tensorplay.int32,
    )
    actual = (
        tensorplay.empty(batch * query_tokens, device="cuda", dtype=tensorplay.int64),
        tensorplay.empty(batch * query_tokens, device="cuda", dtype=tensorplay.int64),
        tensorplay.empty(batch * query_tokens, device="cuda", dtype=tensorplay.int64),
        tensorplay.empty(
            batch * speculative_tokens,
            device="cuda",
            dtype=tensorplay.int64,
        ),
    )
    expected = tuple(tensorplay.empty_like(tensor) for tensor in actual)

    dflash_query_setup_into(
        last_ids,
        last_positions,
        block_tables,
        *actual,
        mask_token_id=151665,
        page_size=16,
        num_speculative_tokens=speculative_tokens,
    )
    dflash_query_setup_reference(
        last_ids,
        last_positions,
        block_tables,
        *expected,
        mask_token_id=151665,
        page_size=16,
        num_speculative_tokens=speculative_tokens,
    )

    for actual_tensor, expected_tensor in zip(actual, expected, strict=True):
        tensorplay.testing.assert_close(actual_tensor, expected_tensor)


def test_greedy_chain_verify_matches_reference_and_is_graph_safe() -> None:
    draft = tensorplay.tensor(
        [
            [1, 2, 3, 4],
            [5, 6, 7, 8],
            [9, 10, 11, 12],
        ],
        device="cuda",
        dtype=tensorplay.int64,
    )
    target = tensorplay.tensor(
        [
            [1, 2, 3, 4, 99],
            [50, 60, 70, 80, 90],
            [9, 10, 111, 120, 130],
        ],
        device="cuda",
        dtype=tensorplay.int64,
    )
    actual_tokens = tensorplay.empty_like(target)
    actual_counts = tensorplay.empty(3, device="cuda", dtype=tensorplay.int32)
    expected_tokens = tensorplay.empty_like(target)
    expected_counts = tensorplay.empty_like(actual_counts)

    greedy_chain_verify_into(
        draft,
        target,
        actual_tokens,
        actual_counts,
    )
    greedy_chain_verify_reference(
        draft,
        target,
        expected_tokens,
        expected_counts,
    )
    tensorplay.testing.assert_close(actual_tokens, expected_tokens)
    tensorplay.testing.assert_close(actual_counts, expected_counts)

    graph = tensorplay.cuda.CUDAGraph()
    with tensorplay.cuda.graph(graph):
        greedy_chain_verify_into(
            draft,
            target,
            actual_tokens,
            actual_counts,
        )
    target[1, 0] = 5
    graph.replay()
    assert actual_counts.tolist() == [4, 1, 2]
    assert actual_tokens[1].tolist() == [5, 60, -1, -1, -1]


def test_fused_rope_cache_write_matches_fallback_and_masks_padding() -> None:
    tensorplay.manual_seed(2)
    query = tensorplay.randn(3, 8, 96, device="cuda", dtype=tensorplay.float16)
    key = tensorplay.randn(3, 4, 96, device="cuda", dtype=tensorplay.float16)
    value = tensorplay.randn_like(key)
    positions = tensorplay.tensor([0, 7, 11], device="cuda")
    slots = tensorplay.tensor([0, 5, 13], device="cuda")
    active = tensorplay.tensor([True, False, True], device="cuda")
    inverse = 1.0 / (
        1_000_000
        ** (tensorplay.arange(0, 96, 2, device="cuda", dtype=tensorplay.float32) / 96)
    )
    frequencies = tensorplay.outer(
        tensorplay.arange(32, device="cuda", dtype=tensorplay.float32),
        inverse,
    )
    cos = frequencies.cos()
    sin = frequencies.sin()
    triton_key = tensorplay.zeros(2, 8, 4, 96, device="cuda", dtype=tensorplay.float16)
    triton_value = tensorplay.zeros_like(triton_key)
    tensorplay_key = tensorplay.zeros_like(triton_key)
    tensorplay_value = tensorplay.zeros_like(triton_key)
    actual = rope_and_cache_write(
        query,
        key,
        value,
        positions,
        slots,
        cos,
        sin,
        triton_key,
        triton_value,
        active,
    )
    expected = fallback.rope_and_cache_write(
        query,
        key,
        value,
        positions,
        slots,
        cos,
        sin,
        tensorplay_key,
        tensorplay_value,
        active,
    )
    tensorplay.testing.assert_close(actual, expected, atol=2e-3, rtol=2e-3)
    tensorplay.testing.assert_close(triton_key, tensorplay_key, atol=2e-3, rtol=2e-3)
    tensorplay.testing.assert_close(triton_value, tensorplay_value, atol=0, rtol=0)
    assert tensorplay.count_nonzero(triton_key.view(-1, 4, 96)[5]) == 0

    q_weight = tensorplay.randn(96, device="cuda", dtype=tensorplay.float16)
    k_weight = tensorplay.randn(96, device="cuda", dtype=tensorplay.float16)
    packed_qkv = tensorplay.cat(
        (
            query.flatten(1),
            key.flatten(1),
            value.flatten(1),
        ),
        dim=-1,
    )
    fused_key = tensorplay.zeros_like(triton_key)
    fused_value = tensorplay.zeros_like(triton_value)
    fused_query = fused_qkv_norm_rope_cache_write(
        packed_qkv,
        q_weight,
        k_weight,
        1e-6,
        positions,
        slots,
        cos,
        sin,
        fused_key,
        fused_value,
        active,
        num_query_heads=8,
        num_kv_heads=4,
    )
    normalized_query = fallback.rms_norm(query, q_weight, 1e-6)
    normalized_key = fallback.rms_norm(key, k_weight, 1e-6)
    expected_key = tensorplay.zeros_like(triton_key)
    expected_value = tensorplay.zeros_like(triton_value)
    expected_query = fallback.rope_and_cache_write(
        normalized_query,
        normalized_key,
        value,
        positions,
        slots,
        cos,
        sin,
        expected_key,
        expected_value,
        active,
    )
    tensorplay.testing.assert_close(
        fused_query,
        expected_query,
        atol=2e-3,
        rtol=2e-3,
    )
    tensorplay.testing.assert_close(
        fused_key,
        expected_key,
        atol=2e-3,
        rtol=2e-3,
    )
    tensorplay.testing.assert_close(fused_value, expected_value, atol=0, rtol=0)


def test_split_kv_paged_decode_matches_fallback_for_variable_lengths() -> None:
    tensorplay.manual_seed(3)
    query = tensorplay.randn(2, 8, 96, device="cuda", dtype=tensorplay.float16)
    key = tensorplay.randn(8, 16, 4, 96, device="cuda", dtype=tensorplay.float16)
    value = tensorplay.randn_like(key)
    block_tables = tensorplay.tensor(
        [[0, 1, 2, -1], [3, 4, 5, 6]],
        device="cuda",
        dtype=tensorplay.int32,
    )
    lengths = tensorplay.tensor([37, 63], device="cuda", dtype=tensorplay.int32)
    actual = paged_decode_attention(
        query,
        key,
        value,
        block_tables,
        lengths,
        16,
    )
    expected = fallback.paged_decode_attention(
        query,
        key,
        value,
        block_tables,
        lengths,
        16,
    )
    tensorplay.testing.assert_close(actual, expected, atol=2e-3, rtol=2e-3)
    decode_output = tensorplay.empty_like(query)
    reused_decode = paged_decode_attention(
        query,
        key,
        value,
        block_tables,
        lengths,
        16,
        output=decode_output,
    )
    assert reused_decode.data_ptr() == decode_output.data_ptr()
    tensorplay.testing.assert_close(
        reused_decode,
        expected,
        atol=2e-3,
        rtol=2e-3,
    )


def test_b1_full_context_paged_decode_tile_matches_fallback() -> None:
    tensorplay.manual_seed(30)
    query = tensorplay.randn(1, 8, 96, device="cuda", dtype=tensorplay.float16)
    key = tensorplay.randn(16, 16, 4, 96, device="cuda", dtype=tensorplay.float16)
    value = tensorplay.randn_like(key)
    block_tables = tensorplay.arange(
        16,
        device="cuda",
        dtype=tensorplay.int32,
    ).reshape(1, -1)
    lengths = tensorplay.tensor([129], device="cuda", dtype=tensorplay.int32)
    actual = paged_decode_attention(
        query,
        key,
        value,
        block_tables,
        lengths,
        16,
    )
    expected = fallback.paged_decode_attention(
        query,
        key,
        value,
        block_tables,
        lengths,
        16,
    )
    tensorplay.testing.assert_close(actual, expected, atol=2e-3, rtol=2e-3)


def test_b16_wide_tile_paged_decode_matches_fallback() -> None:
    tensorplay.manual_seed(3016)
    batch_size = 16
    pages_per_sequence = 9
    query = tensorplay.randn(
        batch_size,
        8,
        96,
        device="cuda",
        dtype=tensorplay.float16,
    )
    key = tensorplay.randn(
        batch_size * pages_per_sequence,
        16,
        4,
        96,
        device="cuda",
        dtype=tensorplay.float16,
    )
    value = tensorplay.randn_like(key)
    block_tables = tensorplay.arange(
        batch_size * pages_per_sequence,
        device="cuda",
        dtype=tensorplay.int32,
    ).reshape(batch_size, pages_per_sequence)
    lengths = tensorplay.full(
        (batch_size,),
        129,
        device="cuda",
        dtype=tensorplay.int32,
    )
    actual = paged_decode_attention(
        query,
        key,
        value,
        block_tables,
        lengths,
        16,
    )
    expected = fallback.paged_decode_attention(
        query,
        key,
        value,
        block_tables,
        lengths,
        16,
    )
    tensorplay.testing.assert_close(actual, expected, atol=2e-3, rtol=2e-3)


def test_varlen_paged_flash_prefill_matches_sdpa_with_context() -> None:
    tensorplay.manual_seed(31)
    query = tensorplay.randn(8, 8, 96, device="cuda", dtype=tensorplay.float16)
    key = tensorplay.randn(4, 16, 4, 96, device="cuda", dtype=tensorplay.float16)
    value = tensorplay.randn_like(key)
    block_tables = tensorplay.tensor(
        [[0, -1], [1, -1]],
        device="cuda",
        dtype=tensorplay.int32,
    )
    query_start_loc = tensorplay.tensor(
        [0, 5, 8],
        device="cuda",
        dtype=tensorplay.int32,
    )
    positions = tensorplay.tensor(
        [0, 1, 2, 3, 4, 4, 5, 6],
        device="cuda",
        dtype=tensorplay.long,
    )
    actual = paged_prefill_attention(
        query,
        key,
        value,
        block_tables,
        query_start_loc,
        positions,
        query_lengths=(5, 3),
        context_lengths=(0, 4),
        page_size=16,
    )
    prefill_output = tensorplay.empty_like(query)
    reused_prefill = paged_prefill_attention(
        query,
        key,
        value,
        block_tables,
        query_start_loc,
        positions,
        query_lengths=(5, 3),
        context_lengths=(0, 4),
        page_size=16,
        output=prefill_output,
    )
    assert reused_prefill.data_ptr() == prefill_output.data_ptr()

    expected = tensorplay.empty_like(query)
    for sequence_index, (start, end, context) in enumerate(
        ((0, 5, 0), (5, 8, 4))
    ):
        sequence_length = context + end - start
        keys = key[sequence_index, :sequence_length].repeat_interleave(2, dim=1)
        values = value[sequence_index, :sequence_length].repeat_interleave(
            2, dim=1
        )
        query_states = query[start:end].transpose(0, 1).unsqueeze(0)
        key_states = keys.transpose(0, 1).unsqueeze(0)
        value_states = values.transpose(0, 1).unsqueeze(0)
        query_positions = tensorplay.arange(
            context,
            sequence_length,
            device="cuda",
        )
        key_positions = tensorplay.arange(sequence_length, device="cuda")
        causal_mask = (
            key_positions[None, :] <= query_positions[:, None]
        )[None, None, :, :]
        attended = F.scaled_dot_product_attention(
            query_states,
            key_states,
            value_states,
            attn_mask=causal_mask,
            dropout_p=0.0,
            scale=96**-0.5,
        )
        expected[start:end] = attended.squeeze(0).transpose(0, 1)
    tensorplay.testing.assert_close(actual, expected, atol=3e-3, rtol=3e-3)
    tensorplay.testing.assert_close(
        reused_prefill,
        expected,
        atol=3e-3,
        rtol=3e-3,
    )


def test_sdpa_paged_prefill_preserves_leading_decode_output() -> None:
    tensorplay.manual_seed(32)
    query = tensorplay.randn(10, 8, 128, device="cuda", dtype=tensorplay.float16)
    key = tensorplay.randn(2, 16, 4, 128, device="cuda", dtype=tensorplay.float16)
    value = tensorplay.randn_like(key)
    block_tables = tensorplay.tensor(
        [[0], [1]],
        device="cuda",
        dtype=tensorplay.int32,
    )
    query_start_loc = tensorplay.tensor(
        [2, 7, 10],
        device="cuda",
        dtype=tensorplay.int32,
    )
    positions = tensorplay.arange(10, device="cuda")
    output = tensorplay.full_like(query, 7.0)

    actual = sdpa_paged_prefill_attention(
        query,
        key,
        value,
        block_tables,
        query_start_loc,
        positions,
        query_lengths=(5, 3),
        context_lengths=(0, 4),
        page_size=16,
        output=output,
    )

    assert actual.data_ptr() == output.data_ptr()
    tensorplay.testing.assert_close(actual[:2], tensorplay.full_like(actual[:2], 7.0))
    expected = tensorplay.empty_like(query[2:])
    for sequence_index, (start, end, context) in enumerate(
        ((2, 7, 0), (7, 10, 4))
    ):
        sequence_length = context + end - start
        keys = key[sequence_index, :sequence_length].repeat_interleave(2, dim=1)
        values = value[sequence_index, :sequence_length].repeat_interleave(
            2, dim=1
        )
        query_states = query[start:end].transpose(0, 1).unsqueeze(0)
        key_states = keys.transpose(0, 1).unsqueeze(0)
        value_states = values.transpose(0, 1).unsqueeze(0)
        query_positions = tensorplay.arange(
            context,
            sequence_length,
            device="cuda",
        )
        key_positions = tensorplay.arange(sequence_length, device="cuda")
        causal = (
            key_positions[None, :] <= query_positions[:, None]
        )[None, None]
        attended = F.scaled_dot_product_attention(
            query_states,
            key_states,
            value_states,
            attn_mask=causal,
            dropout_p=0.0,
            scale=128**-0.5,
        )
        packed_start = start - 2
        packed_end = end - 2
        expected[packed_start:packed_end] = (
            attended.squeeze(0).transpose(0, 1)
        )
    tensorplay.testing.assert_close(actual[2:], expected, atol=0, rtol=0)


def test_qwen25_omni_triton_plan_binds_triton_prefill() -> None:
    policy = KernelPolicy(use_triton=True)
    binding_spec = discover_kernel_plan(
        policy,
        device="cuda",
        dtype=tensorplay.float16,
        model_type="qwen2_5_omni",
    )
    bindings = bind_kernel_plan(
        policy,
        device="cuda",
        dtype=tensorplay.float16,
        expected=binding_spec,
        model_type="qwen2_5_omni",
    )

    assert binding_spec.prefill_backend == TRITON_PAGED_PREFILL_BACKEND
    assert bindings.prefill_backend == TRITON_PAGED_PREFILL_BACKEND
    assert (
        binding_spec.linear_backend
        == TRITON_BATCH_INVARIANT_LINEAR_BACKEND
    )
    assert (
        bindings.linear_backend
        == TRITON_BATCH_INVARIANT_LINEAR_BACKEND
    )
    assert bindings.core.linear is batch_invariant_linear
    assert bindings.core.paged_prefill_attention is paged_prefill_attention
    assert bindings.sampling.use_triton


def test_routed_moe_and_hierarchical_argmax_match_fallback() -> None:
    tensorplay.manual_seed(4)
    hidden = tensorplay.randn(5, 64, device="cuda", dtype=tensorplay.float16)
    gate_up = (
        tensorplay.randn(4, 192, 64, device="cuda", dtype=tensorplay.float16) * 0.05
    )
    down = tensorplay.randn(4, 64, 96, device="cuda", dtype=tensorplay.float16) * 0.05
    expert_ids = tensorplay.tensor(
        [[0, 1], [1, 3], [2, 0], [3, 1], [0, 2]],
        device="cuda",
    )
    route_weights = tensorplay.softmax(
        tensorplay.randn(5, 2, device="cuda"),
        dim=-1,
    )
    actual = routed_moe_decode(
        hidden,
        gate_up,
        down,
        expert_ids,
        route_weights,
    )
    expected = fallback.routed_moe_decode(
        hidden,
        gate_up,
        down,
        expert_ids,
        route_weights,
    )
    tensorplay.testing.assert_close(actual, expected, atol=3e-3, rtol=3e-3)
    tensorcore = routed_moe_tensorcore(
        hidden,
        gate_up,
        down,
        expert_ids,
        route_weights,
    )
    tensorplay.testing.assert_close(tensorcore, expected, atol=3e-3, rtol=3e-3)
    top1_ids = expert_ids[:, :1]
    top1_weights = route_weights[:, :1]
    workspace = GroupedMoEWorkspace.allocate(
        max_tokens=hidden.shape[0],
        num_experts=gate_up.shape[0],
        hidden_size=hidden.shape[1],
        intermediate_size=down.shape[2],
        dtype=hidden.dtype,
        device=hidden.device,
    )
    grouped = routed_moe_grouped_tensorcore(
        hidden,
        gate_up,
        down,
        top1_ids,
        top1_weights,
        workspace=workspace,
    )
    assert grouped.data_ptr() == workspace.output.data_ptr()
    grouped_expected = fallback.routed_moe_decode(
        hidden,
        gate_up,
        down,
        top1_ids,
        top1_weights,
    )
    tensorplay.testing.assert_close(
        grouped,
        grouped_expected,
        atol=3e-3,
        rtol=3e-3,
    )
    router_logits = tensorplay.randn(17, 4, device="cuda", dtype=tensorplay.float16)
    top1_weights, top1_ids = moe_top1(
        router_logits,
        normalize=False,
    )
    expected_weights = tensorplay.softmax(
        router_logits.float(),
        dim=-1,
    )
    expected_top1_weights, expected_top1_ids = tensorplay.topk(
        expected_weights,
        1,
        dim=-1,
    )
    assert tensorplay.equal(top1_ids, expected_top1_ids)
    tensorplay.testing.assert_close(
        top1_weights,
        expected_top1_weights,
        atol=1e-6,
        rtol=1e-6,
    )
    normalized_weights, normalized_ids = moe_top1(
        router_logits,
        normalize=True,
    )
    assert tensorplay.equal(normalized_ids, expected_top1_ids)
    assert tensorplay.equal(normalized_weights, tensorplay.ones_like(normalized_weights))

    logits = tensorplay.randn(
        5,
        151936,
        device="cuda",
        dtype=tensorplay.float16,
    )
    logits[0, 3] = 100
    logits[0, -1] = 100
    assert tensorplay.equal(
        greedy_argmax(logits),
        tensorplay.argmax(logits, dim=-1),
    )
    sample_indices = tensorplay.tensor([0, -1, 2, -1, 4], device="cuda")
    output = tensorplay.empty(5, device="cuda", dtype=tensorplay.int64)
    masked = greedy_argmax(
        logits,
        sample_indices=sample_indices,
        output=output,
    )
    expected = tensorplay.argmax(logits, dim=-1)
    expected[sample_indices < 0] = -1
    assert masked.data_ptr() == output.data_ptr()
    assert tensorplay.equal(masked, expected)


def test_fused_topk_router_matches_reference() -> None:
    tensorplay.manual_seed(71)
    num_experts = 32
    offsets = tensorplay.arange(num_experts, device="cuda").float() * 1e-3
    logits = tensorplay.randn(19, num_experts, device="cuda") + offsets
    for gating in ("softmax", "sigmoid"):
        for normalize in (False, True):
            actual_weights, actual_ids = moe_topk_softmax(
                logits,
                4,
                normalize=normalize,
                gating=gating,
            )
            expected_weights, expected_ids = fallback.moe_topk_softmax(
                logits,
                4,
                normalize=normalize,
                gating=gating,
            )
            assert tensorplay.equal(actual_ids, expected_ids)
            tensorplay.testing.assert_close(
                actual_weights,
                expected_weights,
                atol=1e-6,
                rtol=1e-6,
            )


def test_grouped_topk_matches_fallback() -> None:
    tensorplay.manual_seed(72)
    hidden_size = 64
    intermediate_size = 96
    for num_experts, token_count in ((4, 5), (128, 32)):
        hidden = tensorplay.randn(
            token_count,
            hidden_size,
            device="cuda",
            dtype=tensorplay.float16,
        )
        gate_up = (
            tensorplay.randn(
                num_experts,
                2 * intermediate_size,
                hidden_size,
                device="cuda",
                dtype=tensorplay.float16,
            )
            * 0.05
        )
        down = (
            tensorplay.randn(
                num_experts,
                hidden_size,
                intermediate_size,
                device="cuda",
                dtype=tensorplay.float16,
            )
            * 0.05
        )
        expert_ids = tensorplay.randint(
            0,
            num_experts,
            (token_count, 2),
            device="cuda",
        )
        route_weights = tensorplay.softmax(
            tensorplay.randn(token_count, 2, device="cuda"),
            dim=-1,
        )
        workspace = GroupedMoEWorkspace.allocate(
            max_tokens=token_count,
            num_experts=num_experts,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            dtype=hidden.dtype,
            device=hidden.device,
            max_top_k=2,
        )
        grouped = routed_moe_grouped_tensorcore(
            hidden,
            gate_up,
            down,
            expert_ids,
            route_weights,
            workspace=workspace,
        )
        expected = fallback.routed_moe_decode(
            hidden,
            gate_up,
            down,
            expert_ids,
            route_weights,
        )
        assert grouped.data_ptr() == workspace.output.data_ptr()
        tensorplay.testing.assert_close(
            grouped,
            expected,
            atol=3e-3,
            rtol=3e-3,
        )


def test_compact_grouped_moe_tiles_match_fallback() -> None:
    tensorplay.manual_seed(4512)
    token_count = 512
    hidden_size = 768
    intermediate_size = 2432
    num_experts = 4
    hidden = tensorplay.randn(
        token_count,
        hidden_size,
        device="cuda",
        dtype=tensorplay.float16,
    )
    gate_up = (
        tensorplay.randn(
            num_experts,
            intermediate_size * 2,
            hidden_size,
            device="cuda",
            dtype=tensorplay.float16,
        )
        * 0.02
    )
    down = (
        tensorplay.randn(
            num_experts,
            hidden_size,
            intermediate_size,
            device="cuda",
            dtype=tensorplay.float16,
        )
        * 0.02
    )
    top1_ids = tensorplay.randint(
        num_experts,
        (token_count, 1),
        device="cuda",
    )
    top1_weights = tensorplay.rand(
        token_count,
        1,
        device="cuda",
        dtype=tensorplay.float32,
    )
    workspace = GroupedMoEWorkspace.allocate(
        max_tokens=token_count,
        num_experts=num_experts,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        dtype=hidden.dtype,
        device=hidden.device,
    )

    actual = routed_moe_grouped_tensorcore(
        hidden,
        gate_up,
        down,
        top1_ids,
        top1_weights,
        workspace=workspace,
    )
    expected = fallback.routed_moe_decode(
        hidden,
        gate_up,
        down,
        top1_ids,
        top1_weights,
    )
    tensorplay.testing.assert_close(actual, expected, atol=3e-3, rtol=3e-3)


def test_dynamic_block_grouped_moe_matches_fallback() -> None:
    tensorplay.manual_seed(4064)
    token_count = 64
    hidden_size = 64
    intermediate_size = 96
    num_experts = 4
    hidden = tensorplay.randn(
        token_count,
        hidden_size,
        device="cuda",
        dtype=tensorplay.float16,
    )
    gate_up = (
        tensorplay.randn(
            num_experts,
            intermediate_size * 2,
            hidden_size,
            device="cuda",
            dtype=tensorplay.float16,
        )
        * 0.05
    )
    down = (
        tensorplay.randn(
            num_experts,
            hidden_size,
            intermediate_size,
            device="cuda",
            dtype=tensorplay.float16,
        )
        * 0.05
    )
    top1_ids = tensorplay.tensor(
        [0] * 50 + [2] * 10 + [3] * 4,
        device="cuda",
        dtype=tensorplay.int64,
    ).reshape(token_count, 1)
    top1_weights = tensorplay.rand(
        token_count,
        1,
        device="cuda",
        dtype=tensorplay.float32,
    )
    workspace = GroupedMoEWorkspace.allocate(
        max_tokens=token_count,
        num_experts=num_experts,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        dtype=hidden.dtype,
        device=hidden.device,
    )

    actual = routed_moe_grouped_tensorcore(
        hidden,
        gate_up,
        down,
        top1_ids,
        top1_weights,
        workspace=workspace,
    )
    expected = fallback.routed_moe_decode(
        hidden,
        gate_up,
        down,
        top1_ids,
        top1_weights,
    )
    tensorplay.testing.assert_close(actual, expected, atol=3e-3, rtol=3e-3)


def test_fused_top1_router_grouped_moe_matches_fallback() -> None:
    tensorplay.manual_seed(4065)
    token_count = 64
    hidden_size = 64
    intermediate_size = 96
    num_experts = 4
    hidden = tensorplay.randn(
        token_count,
        hidden_size,
        device="cuda",
        dtype=tensorplay.float16,
    )
    router = (
        tensorplay.randn(
            num_experts,
            hidden_size,
            device="cuda",
            dtype=tensorplay.float16,
        )
        * 0.05
    )
    gate_up = (
        tensorplay.randn(
            num_experts,
            intermediate_size * 2,
            hidden_size,
            device="cuda",
            dtype=tensorplay.float16,
        )
        * 0.05
    )
    down = (
        tensorplay.randn(
            num_experts,
            hidden_size,
            intermediate_size,
            device="cuda",
            dtype=tensorplay.float16,
        )
        * 0.05
    )
    workspace = GroupedMoEWorkspace.allocate(
        max_tokens=token_count,
        num_experts=num_experts,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        dtype=hidden.dtype,
        device=hidden.device,
    )

    actual = routed_moe_grouped_tensorcore(
        hidden,
        gate_up,
        down,
        None,
        None,
        workspace=workspace,
        router_weight=router,
        normalize_router=False,
    )
    router_logits = F.linear(hidden, router)
    routing_weights = tensorplay.softmax(router_logits.float(), dim=-1)
    top1_weights, top1_ids = tensorplay.topk(routing_weights, 1, dim=-1)
    expected = fallback.routed_moe_decode(
        hidden,
        gate_up,
        down,
        top1_ids,
        top1_weights,
    )
    tensorplay.testing.assert_close(actual, expected, atol=3e-3, rtol=3e-3)


def test_hidden_router_and_fixed_tile_moe_match_fallback() -> None:
    tensorplay.manual_seed(40)
    hidden = tensorplay.randn(1, 768, device="cuda", dtype=tensorplay.float16)
    gate = (
        tensorplay.randn(4, 768, device="cuda", dtype=tensorplay.float16) * 0.05
    )
    actual_weights, actual_ids = moe_top1_from_hidden(
        hidden,
        gate,
        normalize=False,
    )
    expected_logits = F.linear(hidden, gate)
    expected_weights = tensorplay.softmax(expected_logits.float(), dim=-1)
    expected_weights, expected_ids = tensorplay.topk(expected_weights, 1, dim=-1)
    assert tensorplay.equal(actual_ids, expected_ids)
    tensorplay.testing.assert_close(
        actual_weights,
        expected_weights,
        atol=3e-5,
        rtol=3e-5,
    )

    gate_up = (
        tensorplay.randn(4, 4864, 768, device="cuda", dtype=tensorplay.float16)
        * 0.02
    )
    down = (
        tensorplay.randn(4, 768, 2432, device="cuda", dtype=tensorplay.float16)
        * 0.02
    )
    actual = routed_moe_decode(
        hidden,
        gate_up,
        down,
        actual_ids,
        actual_weights,
    )
    expected = fallback.routed_moe_decode(
        hidden,
        gate_up,
        down,
        actual_ids,
        actual_weights,
    )
    tensorplay.testing.assert_close(actual, expected, atol=3e-3, rtol=3e-3)


def test_batched_hidden_router_matches_fallback_with_preallocated_output() -> None:
    tensorplay.manual_seed(41)
    hidden = tensorplay.randn(64, 768, device="cuda", dtype=tensorplay.float16)
    gate = tensorplay.randn(4, 768, device="cuda", dtype=tensorplay.float16) * 0.05
    output = (
        tensorplay.empty(64, 1, device="cuda", dtype=tensorplay.float32),
        tensorplay.empty(64, 1, device="cuda", dtype=tensorplay.int64),
    )

    actual_weights, actual_ids = moe_top1_from_hidden(
        hidden,
        gate,
        normalize=False,
        output=output,
    )
    expected_logits = F.linear(hidden, gate)
    expected_weights = tensorplay.softmax(expected_logits.float(), dim=-1)
    expected_weights, expected_ids = tensorplay.topk(expected_weights, 1, dim=-1)

    assert actual_weights.data_ptr() == output[0].data_ptr()
    assert actual_ids.data_ptr() == output[1].data_ptr()
    assert tensorplay.equal(actual_ids, expected_ids)
    tensorplay.testing.assert_close(
        actual_weights,
        expected_weights,
        atol=3e-5,
        rtol=3e-5,
    )


def test_grouped_int4_moe_matches_per_expert_reference() -> None:
    """The fused grouped int4 MoE matches the eager per-expert path.

    The reference runs ``int4_weight_mm`` once per (token, slot) route with
    the runtime's native int4 packing, exactly like the segmented eager
    path; the fused kernel consumes the stacked row-major packing the CUDA
    loader produces.  Agreement bounds the Triton dequantisation against the
    native op, which also pins the down-stage row mapping: any gather
    mismatch shows up as a large per-route error.
    """

    from megalm.kernels.moe_grouped_int4 import (
        pack_int4_expert_rowmajor,
        routed_moe_grouped_int4,
    )
    from megalm.kernels.quant_fp import int4_weight_mm, pack_int4_weight

    device = "cuda"
    num_experts, tokens, top = 4, 5, 2
    hidden, inter = 64, 32
    out13 = inter * 2
    group = 32

    def randn(*shape):
        return tensorplay.randn(*shape, dtype=tensorplay.float32, device=device)

    hidden_states = randn(tokens, hidden).to(tensorplay.bfloat16)
    packed13, scale13 = [], []
    packed2, scale2 = [], []
    native13, native_sz13 = [], []
    native2, native_sz2 = [], []
    for _ in range(num_experts):
        w13 = randn(out13, hidden)
        packed, scales = pack_int4_expert_rowmajor(w13, group)
        packed13.append(packed)
        scale13.append(scales.to(tensorplay.bfloat16))
        native, native_scales = pack_int4_weight(w13, group)
        native13.append(native)
        native_sz13.append(native_scales.to(tensorplay.bfloat16))
        w2 = randn(hidden, inter)
        packed, scales = pack_int4_expert_rowmajor(w2, group)
        packed2.append(packed)
        scale2.append(scales.to(tensorplay.bfloat16))
        native, native_scales = pack_int4_weight(w2, group)
        native2.append(native)
        native_sz2.append(native_scales.to(tensorplay.bfloat16))

    expert_ids = tensorplay.tensor(
        [[0, 1], [1, 2], [2, 3], [3, 0], [0, 2]],
        dtype=tensorplay.int64,
        device=device,
    )
    route_weights = tensorplay.tensor(
        [[0.6, 0.4], [0.5, 0.5], [0.7, 0.3], [0.2, 0.8], [1.0, 0.0]],
        dtype=tensorplay.float32,
        device=device,
    )

    def silu_cap(gate, up, gate_softcap, up_softcap):
        return F.silu(gate) * up

    actual = routed_moe_grouped_int4(
        hidden_states,
        tensorplay.stack(packed13, dim=0),
        tensorplay.stack(scale13, dim=0),
        tensorplay.stack(packed2, dim=0),
        tensorplay.stack(scale2, dim=0),
        expert_ids,
        route_weights,
        group_size=group,
        num_experts=num_experts,
        gate_softcap=0.0,
        up_softcap=0.0,
        interleaved=True,
        cap_fn=silu_cap,
    )

    expected = tensorplay.zeros(
        tokens, hidden, dtype=tensorplay.float32, device=device
    )
    for token in range(tokens):
        acc = tensorplay.zeros(hidden, dtype=tensorplay.float32, device=device)
        for slot in range(top):
            expert = int(expert_ids[token][slot])
            gate_up = int4_weight_mm(
                hidden_states[token : token + 1],
                native13[expert],
                native_sz13[expert],
                group,
            ).float()
            act = (F.silu(gate_up[:, 0::2]) * gate_up[:, 1::2]).to(
                tensorplay.bfloat16
            )
            down = int4_weight_mm(
                act, native2[expert], native_sz2[expert], group
            ).float()
            acc += down[0] * float(route_weights[token][slot])
        expected[token] = acc

    assert actual.shape == expected.shape
    # Tolerance covers the bf16 activation rounding (the fused kernel keeps
    # the swiglu input in bf16 like the eager path, the reference rounds a
    # float32 activation down instead).
    error = (actual.float() - expected).abs().max().item()
    scale = expected.abs().max().item()
    assert error <= max(4.0, scale * 0.01), (error, scale)
