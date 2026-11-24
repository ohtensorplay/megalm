import pytest
import tensorplay

from megalm.kernels.attention import (
    AttentionConfig,
    build_attention_backend,
)
from megalm.kernels.attention.sparse import SparseIndexCache, attach_index_cache


pytestmark = pytest.mark.skipif(
    not tensorplay.cuda.is_available(), reason="CUDA attention backend test"
)


def _cuda_tensors():
    device = tensorplay.device("cuda:0")
    pages = 8
    key = tensorplay.arange(
        pages * 4 * 2 * 8, dtype=tensorplay.float32, device=device
    ).reshape(pages, 4, 2, 8) * 0.01
    value = key + 1.0
    query = tensorplay.randn(1, 4, 8, dtype=tensorplay.float32, device=device)
    tables = tensorplay.tensor(
        [list(range(pages))], dtype=tensorplay.int32, device=device
    )
    lengths = tensorplay.tensor([pages * 4], dtype=tensorplay.int32, device=device)
    return query, key, value, tables, lengths


@pytest.mark.parametrize(
    "config",
    [
        AttentionConfig(mode="non_causal"),
        AttentionConfig(mode="sink", sink_tokens=1),
        AttentionConfig(mode="sparse", sparse_top_k=2),
    ],
    ids=lambda c: c.mode,
)
def test_decode_capture_replay_matches_eager(config):
    query, key, value, tables, lengths = _cuda_tensors()
    backend = build_attention_backend(
        config, sink_logits=(0.5,) if config.mode == "sink" else ()
    )
    expected = backend.decode(query, key, value, tables, lengths, 4).clone()
    graph = tensorplay.cuda.CUDAGraph()
    with tensorplay.cuda.graph(graph):
        captured = backend.decode(query, key, value, tables, lengths, 4)
    graph.replay()
    assert tensorplay.allclose(captured, expected, atol=1e-5)


def test_sparse_decode_replay_updates_pages_lengths_and_queries():
    device = tensorplay.device("cuda:0")
    key = tensorplay.zeros((6, 2, 1, 8), device=device)
    value = tensorplay.arange(12, dtype=key.dtype, device=device).reshape(6, 2, 1, 1)
    value = value.expand_as(key).contiguous()
    index = tensorplay.arange(6, dtype=key.dtype, device=device).reshape(6, 1, 1)
    attach_index_cache(key, SparseIndexCache(index.expand(6, 2, 8).contiguous()))
    query = tensorplay.ones((2, 2, 8), device=device)
    tables = tensorplay.zeros((2, 5), dtype=tensorplay.int32, device=device)
    lengths = tensorplay.ones((2,), dtype=tensorplay.int32, device=device)
    output = tensorplay.empty_like(query)
    backend = build_attention_backend(AttentionConfig(mode="sparse", sparse_top_k=1))
    graph = tensorplay.cuda.CUDAGraph()
    with tensorplay.cuda.graph(graph):
        assert backend.decode(query, key, value, tables, lengths, 2, output=output) is output

    replays = [
        (1, [[4, 1, 5, 3, -1], [0, 4, 1, -1, -1]], [7, 5], [8.8, 4.0]),
        (-1, [[1, 5, 2, 0, -1], [3, -1, -1, -1, -1]], [8, 1], [2.5, 6.0]),
        (1, [[4, 1, 5, 3, -1], [-1] * 5], [7, 0], [8.8, 0.0]),
    ]
    for sign, page_ids, seq_lens, averages in replays:
        query.fill_(sign)
        tables.copy_(tensorplay.tensor(page_ids, dtype=tables.dtype, device=device))
        lengths.copy_(tensorplay.tensor(seq_lens, dtype=lengths.dtype, device=device))
        graph.replay()
        expected = tensorplay.tensor(averages, device=device)[:, None, None].expand_as(output)
        assert tensorplay.allclose(output, expected, atol=1e-5)
        assert tensorplay.allclose(output, backend.decode(query, key, value, tables, lengths, 2))


def test_sparse_selector_cuda_matches_cpu_reference():
    import tensorplay

    from megalm.kernels.attention.sparse_select import SparseTopKSelector

    device = tensorplay.device("cuda:0")
    scores = tensorplay.randn(2, 8, device=device)
    tables = tensorplay.tensor(
        [list(range(8)), list(range(8, 16))],
        dtype=tensorplay.int32,
        device=device,
    )
    selected = SparseTopKSelector(top_k=2)(scores, tables)
    tensorplay.cuda.synchronize()
    assert tuple(selected.shape) == (2, 4)
    rows = selected.tolist()
    assert rows[0][0] == 0 and rows[0][-1] == 7
    assert rows[1][0] == 8 and rows[1][-1] == 15
    assert rows[0] == sorted(rows[0]) and rows[1] == sorted(rows[1])
