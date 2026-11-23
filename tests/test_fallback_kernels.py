import math

import tensorplay

from megalm.kernels.fallback import (
    moe_topk_softmax,
    paged_decode_attention,
    routed_moe_decode,
)


def _reference_paged_decode(
    query,
    key_cache,
    value_cache,
    block_tables,
    sequence_lengths,
    page_size,
):
    batch_size, num_query_heads, head_dim = query.shape
    num_kv_heads = key_cache.shape[-2]
    query_per_kv = num_query_heads // num_kv_heads
    output = tensorplay.empty_like(query)
    scale = head_dim**-0.5
    for batch_index in range(batch_size):
        sequence_length = int(sequence_lengths[batch_index].item())
        page_ids = block_tables[batch_index, : math.ceil(sequence_length / page_size)].long()
        keys = (
            key_cache.index_select(0, page_ids)
            .reshape(-1, num_kv_heads, head_dim)[:sequence_length]
            .repeat_interleave(query_per_kv, dim=1)
        )
        values = (
            value_cache.index_select(0, page_ids)
            .reshape(-1, num_kv_heads, head_dim)[:sequence_length]
            .repeat_interleave(query_per_kv, dim=1)
        )
        scores = tensorplay.einsum(
            "hd,lhd->hl",
            query[batch_index].float(),
            keys.float(),
        )
        probabilities = tensorplay.softmax(scores * scale, dim=-1).to(values.dtype)
        output[batch_index] = tensorplay.einsum(
            "hl,lhd->hd",
            probabilities,
            values,
        )
    return output


def test_vectorized_paged_decode_matches_reference() -> None:
    tensorplay.manual_seed(7)
    query = tensorplay.randn(3, 4, 8)
    key_cache = tensorplay.randn(6, 3, 2, 8)
    value_cache = tensorplay.randn(6, 3, 2, 8)
    block_tables = tensorplay.tensor(
        [
            [0, 1, 2],
            [3, 4, 5],
            [2, 0, 1],
        ],
        dtype=tensorplay.long,
    )
    sequence_lengths = tensorplay.tensor((7, 4, 9), dtype=tensorplay.long)

    actual = paged_decode_attention(
        query,
        key_cache,
        value_cache,
        block_tables,
        sequence_lengths,
        page_size=3,
    )
    expected = _reference_paged_decode(
        query,
        key_cache,
        value_cache,
        block_tables,
        sequence_lengths,
        page_size=3,
    )

    assert tensorplay.allclose(actual, expected, atol=1e-5, rtol=1e-5)


def test_routed_moe_decode_handles_empty_expert_routes() -> None:
    hidden_states = tensorplay.randn(4, 8)
    gate_up_weight = tensorplay.randn(4, 16, 8)
    down_weight = tensorplay.randn(4, 8, 8)
    topk_ids = tensorplay.tensor(
        ((0, 1), (0, 1), (3, 3), (2, 3)),
        dtype=tensorplay.long,
    )
    topk_weights = tensorplay.full((4, 2), 0.5)

    output = routed_moe_decode(
        hidden_states,
        gate_up_weight,
        down_weight,
        topk_ids,
        topk_weights,
    )

    assert tuple(output.shape) == (4, 8)
    assert bool(tensorplay.isfinite(output).all().item())


def test_moe_topk_softmax_matches_manual_chains() -> None:
    logits = tensorplay.tensor(
        (
            (1.0, 3.0, 2.0, 0.0),
            (-1.0, 0.0, 4.0, 2.0),
        ),
        dtype=tensorplay.float32,
    )
    expected_ids = tensorplay.tensor(((1, 2), (2, 3)))

    normalized, ids = moe_topk_softmax(logits, 2, normalize=True)
    assert tensorplay.equal(ids, expected_ids)
    selected = tensorplay.gather(logits, 1, expected_ids)
    expected = tensorplay.softmax(selected, dim=-1)
    assert tensorplay.allclose(normalized, expected, atol=1e-6, rtol=1e-6)

    full_probabilities, ids = moe_topk_softmax(logits, 2, normalize=False)
    assert tensorplay.equal(ids, expected_ids)
    expected = tensorplay.gather(
        tensorplay.softmax(logits, dim=-1), 1, expected_ids
    )
    assert tensorplay.allclose(full_probabilities, expected, atol=1e-6, rtol=1e-6)

    sigmoid_scores = tensorplay.sigmoid(logits)
    _, ids = moe_topk_softmax(
        logits, 2, normalize=True, gating="sigmoid"
    )
    assert tensorplay.equal(ids, expected_ids)
    selected = tensorplay.gather(sigmoid_scores, 1, expected_ids)
    expected = selected / selected.sum(dim=-1, keepdim=True)
    weights, _ = moe_topk_softmax(
        logits, 2, normalize=True, gating="sigmoid"
    )
    assert tensorplay.allclose(weights, expected, atol=1e-6, rtol=1e-6)
