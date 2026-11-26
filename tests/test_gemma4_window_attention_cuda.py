"""Windowed fast-attention kernels match explicit sliding-window masking.

Gemma-4 local layers (sliding_window=512) used to fall back to a Python loop
because the fast Triton kernels had no window support.  This verifies the fast
kernel path now implements the same ``q_pos - k_pos < window`` mask as the
HuggingFace overlay and the reference fallback.
"""

from __future__ import annotations

import math

import pytest
import tensorplay

pytestmark = pytest.mark.skipif(
    not tensorplay.cuda.is_available(), reason="CUDA windowed-attention test"
)

DEV = tensorplay.device("cuda:0")
PAGE = 4
H, KV, HD = 4, 2, 64


def _reference(
    query: tensorplay.Tensor,
    key_flat: tensorplay.Tensor,
    value_flat: tensorplay.Tensor,
    query_positions: list[int],
    window: int,
    query_length: int,
) -> tensorplay.Tensor:
    sequence = key_flat.shape[0]
    output = tensorplay.empty_like(query)
    for index in range(query_length):
        position = query_positions[index]
        scores = tensorplay.full((H, sequence), -1e9, device=DEV)
        for key_position in range(sequence):
            if key_position <= position and position - key_position < window:
                key = key_flat[key_position].repeat_interleave(H // KV, dim=0)
                scores[:, key_position] = (query[index] * key).sum(-1)
        weights = tensorplay.softmax(scores, dim=-1)
        accumulator = tensorplay.zeros((H, HD), device=DEV)
        for key_position in range(sequence):
            value = value_flat[key_position].repeat_interleave(H // KV, dim=0)
            accumulator = accumulator + weights[:, key_position : key_position + 1] * value
        output[index] = accumulator
    return output


def _setup(sequence: int):
    pages = math.ceil(sequence / PAGE)
    key = tensorplay.randn(pages, PAGE, KV, HD, device=DEV) * 0.1
    value = tensorplay.randn(pages, PAGE, KV, HD, device=DEV) * 0.1
    tables = tensorplay.tensor(
        [list(range(pages))], dtype=tensorplay.int32, device=DEV
    )
    lengths = tensorplay.tensor([sequence], dtype=tensorplay.int32, device=DEV)
    return key, value, key.reshape(-1, KV, HD), value.reshape(-1, KV, HD), tables, lengths


def test_windowed_decode_matches_reference() -> None:
    from megalm.kernels.paged_decode import paged_decode_attention

    sequence, window = 14, 6
    key, value, key_flat, value_flat, tables, lengths = _setup(sequence)
    query = tensorplay.randn(1, H, HD, device=DEV) * 0.5
    fast = paged_decode_attention(
        query, key, value, tables, lengths, PAGE, scale=1.0,
        sliding_window=window,
    )
    expected = _reference(query, key_flat, value_flat, [sequence - 1], window, 1)
    assert tensorplay.allclose(fast, expected, atol=1e-5)


def test_windowed_prefill_matches_reference() -> None:
    from megalm.kernels.paged_prefill import paged_prefill_attention

    sequence, window, query_length = 14, 6, 3
    key, value, key_flat, value_flat, tables, _ = _setup(sequence)
    query = tensorplay.randn(query_length, H, HD, device=DEV) * 0.5
    starts = tensorplay.tensor([0, query_length], dtype=tensorplay.int32, device=DEV)
    positions = tensorplay.tensor(
        list(range(sequence - query_length, sequence)), dtype=tensorplay.int64,
        device=DEV,
    )
    fast = paged_prefill_attention(
        query, key, value, tables, starts, positions,
        query_lengths=(query_length,), context_lengths=(sequence - query_length,),
        page_size=PAGE, scale=1.0, sliding_window=window,
    )
    expected = _reference(
        query, key_flat, value_flat,
        list(range(sequence - query_length, sequence)), window, query_length,
    )
    assert tensorplay.allclose(fast, expected, atol=1e-3)


def test_window_larger_than_context_is_noop() -> None:
    from megalm.kernels.paged_decode import paged_decode_attention

    sequence = 14
    key, value, _, _, tables, lengths = _setup(sequence)
    query = tensorplay.randn(1, H, HD, device=DEV) * 0.5
    full = paged_decode_attention(query, key, value, tables, lengths, PAGE, scale=1.0)
    windowed = paged_decode_attention(
        query, key, value, tables, lengths, PAGE, scale=1.0,
        sliding_window=sequence + 10,
    )
    assert tensorplay.allclose(full, windowed, atol=1e-6)
