"""Gold tests for the grouped-query paged decode attention kernel.

The general paged-decode tests use ``head_dim=96``, which is intentionally
outside :func:`gqa_decode_supported`; this module covers the GQA path directly
at production-shaped head dims (128) with ``page_size > 1`` and both split and
direct (single-split) launches.
"""

from __future__ import annotations

import pytest

import tensorplay as tp

pytest.importorskip("triton")

from megalm.kernels import paged_decode_gqa as gqa  # noqa: E402


def _reference(
    query: tp.Tensor,
    key_cache: tp.Tensor,
    value_cache: tp.Tensor,
    block_tables: tp.Tensor,
    sequence_lengths: tp.Tensor,
    page_size: int,
    scale: float,
) -> tp.Tensor:
    """Independent dense softmax attention over gathered paged KV (fp32)."""

    batch, num_query_heads, head_dim = query.shape
    num_kv_heads = key_cache.shape[-2]
    qpkv = num_query_heads // num_kv_heads
    reference = tp.empty((batch, num_query_heads, head_dim), dtype=query.dtype, device=query.device)

    for b in range(batch):
        length = int(sequence_lengths[b].item())
        num_pages = (length + page_size - 1) // page_size
        pages = block_tables[b, :num_pages].long()
        keys = key_cache[pages].reshape(-1, num_kv_heads, head_dim).float()[:length]
        values = value_cache[pages].reshape(-1, num_kv_heads, head_dim).float()[:length]
        q = query[b].float().reshape(num_kv_heads, qpkv, head_dim)
        scores = tp.einsum("hqd,htd->hqt", q, keys.transpose(0, 1)) * scale
        probabilities = tp.softmax(scores, dim=-1)
        out = tp.einsum("hqt,htd->hqd", probabilities, values.transpose(0, 1))
        reference[b] = out.reshape(num_query_heads, head_dim).to(query.dtype)
    return reference


def _inputs(
    batch: int,
    dtype: object,
    lengths: list[int] | None = None,
    head_dim: int = 128,
    qpkv: int = 4,
    page_size: int = 16,
):
    num_kv_heads = 8
    num_query_heads = num_kv_heads * qpkv
    lengths = lengths or [512] * batch
    max_len = max(lengths)
    num_pages = (max_len + page_size - 1) // page_size
    query = tp.randn((batch, num_query_heads, head_dim), dtype=dtype, device="cuda") * 0.1
    key_cache = tp.randn(
        (num_pages + 2, page_size, num_kv_heads, head_dim), dtype=dtype, device="cuda"
    ) * 0.1
    value_cache = tp.randn_like(key_cache) * 0.1
    block_tables = tp.stack(
        [
            tp.arange(1, num_pages + 3, dtype=tp.int32, device="cuda")
            for _ in range(batch)
        ]
    )
    sequence_lengths = tp.tensor(lengths, dtype=tp.int32, device="cuda")
    output = tp.empty_like(query)
    return (
        query,
        key_cache,
        value_cache,
        block_tables,
        sequence_lengths,
        page_size,
        output,
        head_dim**-0.5,
    )


@pytest.mark.skipif(tp.cuda.device_count() == 0, reason="requires CUDA")
@pytest.mark.parametrize("dtype", [tp.bfloat16, tp.float16])
def test_gqa_decode_split_path_matches_reference(dtype) -> None:
    """batch=1 -> 8 programs -> split-K + reduce path."""
    args = _inputs(1, dtype, lengths=[517])
    query, key_cache, value_cache, tables, lengths, page_size, output, scale = args
    gqa.gqa_decode_attention(
        query, key_cache, value_cache, tables, lengths, page_size, output=output, scale=scale
    )
    expected = _reference(query, key_cache, value_cache, tables, lengths, page_size, scale)
    assert float((output.float() - expected.float()).abs().max().item()) <= 3e-3


@pytest.mark.skipif(tp.cuda.device_count() == 0, reason="requires CUDA")
@pytest.mark.parametrize("lengths", [[512] * 16, [37, 129, 511, 512] * 4])
def test_gqa_decode_direct_path_matches_reference(lengths) -> None:
    """batch=16 -> 128 programs -> single split -> direct output, no reduce."""
    args = _inputs(16, tp.bfloat16, lengths=lengths)
    query, key_cache, value_cache, tables, seqs, page_size, output, scale = args
    gqa.gqa_decode_attention(
        query, key_cache, value_cache, tables, seqs, page_size, output=output, scale=scale
    )
    expected = _reference(query, key_cache, value_cache, tables, seqs, page_size, scale)
    assert float((output.float() - expected.float()).abs().max().item()) <= 3e-3


@pytest.mark.skipif(tp.cuda.device_count() == 0, reason="requires CUDA")
def test_direct_path_skips_partial_workspace_and_reduce() -> None:
    """Structural guard: the single-split path must not allocate or reduce."""
    args = _inputs(16, tp.bfloat16)
    query, key_cache, value_cache, tables, seqs, page_size, output, scale = args
    calls: list[str] = []

    def spy(original, name):
        class Wrapper:
            def __getitem__(self, grid):
                calls.append(name)
                return original[grid]

        return Wrapper()

    direct = gqa._gqa_direct_decode_kernel
    reduce_kernel = gqa._gqa_reduce_kernel
    gqa._gqa_direct_decode_kernel = spy(direct, "direct")
    gqa._gqa_reduce_kernel = spy(reduce_kernel, "reduce")
    try:
        gqa.gqa_decode_attention(
            query, key_cache, value_cache, tables, seqs, page_size, output=output, scale=scale
        )
    finally:
        gqa._gqa_direct_decode_kernel = direct
        gqa._gqa_reduce_kernel = reduce_kernel
    assert calls == ["direct"]


@pytest.mark.skipif(tp.cuda.device_count() == 0, reason="requires CUDA")
def test_gqa_decode_cuda_graph_capture_replay() -> None:
    """The direct path must be capture-safe and allocation-free during capture."""
    args = _inputs(16, tp.bfloat16)
    query, key_cache, value_cache, tables, seqs, page_size, output, scale = args
    # Warm the JIT cache before capture.
    for _ in range(2):
        gqa.gqa_decode_attention(
            query, key_cache, value_cache, tables, seqs, page_size, output=output, scale=scale
        )
    tp.cuda.synchronize()

    graph = tp.cuda.CUDAGraph()
    with tp.cuda.graph(graph):
        gqa.gqa_decode_attention(
            query, key_cache, value_cache, tables, seqs, page_size, output=output, scale=scale
        )
    graph.replay()
    tp.cuda.synchronize()

    expected = _reference(query, key_cache, value_cache, tables, seqs, page_size, scale)
    assert float((output.float() - expected.float()).abs().max().item()) <= 3e-3


@pytest.mark.skipif(tp.cuda.device_count() == 0, reason="requires CUDA")
def test_gqa_decode_supported_guard() -> None:
    args = _inputs(1, tp.bfloat16)
    query, key_cache, *_ = args
    assert gqa.gqa_decode_supported(query, key_cache, 8, None)
    # head_dim 96 is outside the tensor-core tile set.
    odd = tp.empty((1, 32, 96), dtype=tp.bfloat16, device="cuda")
    assert not gqa.gqa_decode_supported(odd, key_cache, 8, None)
    # Quantised caches use the dequantising kernel instead.
    assert not gqa.gqa_decode_supported(query, key_cache, 8, object())
