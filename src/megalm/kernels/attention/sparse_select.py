"""CUDA-Graph-safe top-k page selection for sparse attention.

The fixed-shape Triton kernel uses one program per sequence. Each program
performs ``top_k`` argmax eliminations over its page scores, forces the
first (sink) and last (local-window) pages, and writes the selected **physical**
page ids in ascending logical order into a preallocated ``[batch, K]`` buffer.

All shapes (including ``K``) are fixed at plan time; the kernel never allocates,
synchronizes, or reads a host tensor. The caller keeps one output buffer per
shape so capture and replay use the same address.
"""

from __future__ import annotations

import triton
import triton.language as tl

import tensorplay

from megalm.kernels.attention.types import AttentionConfigError


def _select_cpu(
    scores: tensorplay.Tensor,
    block_tables: tensorplay.Tensor,
    top_k: int,
) -> tensorplay.Tensor:
    """Reference selection for CPU gold tests (not graph-capturable)."""

    batch, pages = scores.shape
    k = min(top_k + 2, pages)
    selected = tensorplay.empty(
        (batch, k), dtype=tensorplay.int32, device=block_tables.device
    )
    for row in range(batch):
        if k == pages:
            positions = tensorplay.arange(
                pages, dtype=tensorplay.int64, device=scores.device
            )
        else:
            interior = tensorplay.arange(
                1, pages - 1, dtype=tensorplay.int64, device=scores.device
            )
            interior_scores = scores[row].index_select(0, interior)
            best = tensorplay.topk(interior_scores, min(top_k, pages - 2)).indices
            best = interior.index_select(0, best.reshape(-1))
            forced = tensorplay.tensor(
                (0, pages - 1), dtype=tensorplay.int64, device=scores.device
            )
            positions = tensorplay.unique(
                tensorplay.cat((best, forced)), sorted=True
            )
        selected[row] = block_tables[row].index_select(0, positions).int()
    return selected


@triton.jit
def _sparse_topk_select_kernel(
    scores_ptr,
    tables_ptr,
    out_ptr,
    P: tl.constexpr,
    K: tl.constexpr,
    TOP_K: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    offsets = tl.arange(0, BLOCK)
    valid = offsets < P

    scores = tl.load(
        scores_ptr + row * P + offsets,
        mask=valid,
        other=-float("inf"),
    ).to(tl.float32)
    tables = tl.load(
        tables_ptr + row * P + offsets,
        mask=valid,
        other=0,
    ).to(tl.int32)

    # Exclude the forced sink/tail pages from the argmax pool; they are added
    # back explicitly so they can never be dropped or duplicated.
    work = tl.where(
        (offsets == 0) | (offsets == P - 1), -float("inf"), scores
    )
    selected = tl.zeros((BLOCK,), tl.int32)
    for _ in range(TOP_K):
        best = tl.argmax(work, 0)
        selected = tl.where(offsets == best, 1, selected)
        work = tl.where(offsets == best, -float("inf"), work)

    selected = tl.where(offsets == 0, 1, selected)
    selected = tl.where(offsets == P - 1, 1, selected)

    # rank = position of this page in the ascending selected order. Storing
    # tables[offsets] at out[rank] yields physical page ids sorted by logical
    # position, which preserves causal order across the selected blocks.
    rank = tl.cumsum(selected, 0) - 1
    tl.store(out_ptr + row * K + rank, tables, mask=selected > 0)


class SparseTopKSelector:
    """Stateful, shape-keyed selector with persistent device buffers."""

    def __init__(self, top_k: int) -> None:
        if top_k <= 0:
            raise AttentionConfigError("sparse top-k must be positive")
        self.top_k = top_k
        self._buffers: dict[tuple[int, int, int, str], tensorplay.Tensor] = {}

    def __call__(
        self,
        scores: tensorplay.Tensor,
        block_tables: tensorplay.Tensor,
    ) -> tensorplay.Tensor:
        """Return ``[batch, K]`` selected physical page ids (ascending)."""

        if scores.ndim != 2 or scores.shape != block_tables.shape:
            raise AttentionConfigError(
                "page scores and block tables must have the same 2-D shape"
            )
        batch, pages = scores.shape
        if batch == 0 or pages == 0:
            raise AttentionConfigError("sparse selection needs nonempty inputs")
        k = min(self.top_k + 2, pages)
        top_k = max(k - 2, 0)
        if block_tables.device.type != "cuda":
            return _select_cpu(scores, block_tables, self.top_k)
        key = (batch, pages, k, str(block_tables.device))
        out = self._buffers.get(key)
        if out is None:
            out = tensorplay.empty(
                (batch, k), dtype=tensorplay.int32, device=block_tables.device
            )
            self._buffers[key] = out
        block = triton.next_power_of_2(pages)
        _sparse_topk_select_kernel[(batch,)](
            scores,
            block_tables,
            out,
            P=pages,
            K=k,
            TOP_K=top_k,
            BLOCK=block,
            num_warps=4,
        )
        return out


__all__ = ["SparseTopKSelector", "_sparse_topk_select_kernel"]
