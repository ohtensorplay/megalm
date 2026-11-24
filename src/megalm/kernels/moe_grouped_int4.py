"""Grouped int4 MoE: one Triton launch per GEMM for every routed expert.

The Inkling-2 executor stores experts as torchao-style int4 weight-only
tensors (``[N, K/2]`` packed codes plus ``[K/group, N, 2]`` scale-and-zero
metadata).  The segmented eager path dequantises through the native
``_weight_int4pack_mm`` op once per expert, which is a host loop that cannot
be captured into a CUDA graph.  These kernels keep the packed weights
resident and dequantise inline inside a single grouped launch, so the whole
MoE becomes a fixed op sequence with static shapes — capture-safe.

Layouts mirror :func:`int4_weight_mm` exactly:

* packed: ``[N, K/2]`` unsigned bytes, low nibble = even k, high nibble =
  odd k;
* scale-zero: ``[K/group, N, 2]`` bf16, last column index 0 = scale,
  1 = zero point;
* dequantised weight: ``w[n, k] = (code - zero) * scale``.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import tensorplay
import triton
import triton.language as tl

from megalm.kernels.moe_grouped_route import _moe_route_count_kernel


@triton.jit
def _small_grouped_int4_tiles(
    expert_counts_ptr,
    tiles_ptr,
    CAPACITY: tl.constexpr,
    NUM_EXPERTS: tl.constexpr,
    EXPERT_BLOCK: tl.constexpr,
    BLOCK_M: tl.constexpr,
    ADAPTIVE: tl.constexpr,
):
    """Build one tile map for both projections in a small grouped batch."""
    tile = tl.program_id(0)
    experts = tl.arange(0, EXPERT_BLOCK)
    counts = tl.load(expert_counts_ptr + experts, experts < NUM_EXPERTS, 0)
    if ADAPTIVE:
        # Group only when there is enough reuse to repay the sort and padding.
        active_experts = tl.sum((counts > 0).to(tl.int32), axis=0)
        use_grouped = active_experts * 3 <= CAPACITY
        if tile == 0:
            tl.store(tiles_ptr, use_grouped.to(tl.int32))

    expert_tiles = tl.cdiv(counts, BLOCK_M)
    tile_ends = tl.cumsum(expert_tiles, axis=0)
    tile_starts = tile_ends - expert_tiles
    owns_tile = (tile >= tile_starts) & (tile < tile_ends)
    row_starts = tl.cumsum(counts, axis=0) - counts
    expert = tl.sum(tl.where(owns_tile, experts + 1, 0), axis=0) - 1
    tile_offset = (tile - tile_starts) * BLOCK_M
    row_start = tl.sum(tl.where(owns_tile, row_starts + tile_offset, 0), axis=0)
    row_count = tl.sum(tl.where(owns_tile, counts - tile_offset, 0), axis=0)
    # Reserve the first row for the dispatch flag shared by both GEMMs.
    tile_ptr = tiles_ptr + (tile + 1) * 3
    tl.store(tile_ptr, expert)
    tl.store(tile_ptr + 1, row_start)
    tl.store(tile_ptr + 2, tl.minimum(row_count, BLOCK_M))


@triton.jit(do_not_specialize=["token_capacity"])
def _grouped_int4_gemm_kernel(
    x_ptr,
    sorted_token_ptr,
    expert_counts_ptr,
    tiles_ptr,
    packed_ptr,
    scale_zero_ptr,
    out_ptr,
    partial_ptr,
    token_capacity,
    SPLIT_K: tl.constexpr,
    K_PER_SLICE: tl.constexpr,
    group_size: tl.constexpr,
    num_experts: tl.constexpr,
    k_size: tl.constexpr,
    n_size: tl.constexpr,
    COMPACT: tl.constexpr,
    EXPERT_BLOCK: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    HALF_K: tl.constexpr,
    ROUTE_MAJOR: tl.constexpr = False,
    ROUTE_INPUT: tl.constexpr = False,
    BROADCAST_SCALE: tl.constexpr = False,
    USE_TILES: tl.constexpr = False,
    ADAPTIVE: tl.constexpr = False,
):
    """Grouped ``x @ W.T`` over every expert of one stacked int4 weight.

    ``packed_ptr`` stacks ``[E, N, K/2]`` codes and ``scale_zero_ptr``
    stacks ``[E, K/group, N, 2]``.  Row blocks over the sorted
    (expert-major) token order are derived on device from ``expert_counts``.
    For one token, each row block is one route and ``sorted_token_ptr``
    contains the expert IDs in slot order. The down GEMM reads its input
    in that same order.
    """

    global_row_block = tl.program_id(0)
    column_block = tl.program_id(1)
    split_k_index = tl.program_id(2)
    if ROUTE_MAJOR:
        active = True
    elif USE_TILES:
        tile_ptr = tiles_ptr + (global_row_block + 1) * 3
        expert_index = tl.load(tile_ptr)
        active = expert_index >= 0
        if ADAPTIVE:
            active &= tl.load(tiles_ptr) != 0
    else:
        expert_offsets = tl.arange(0, EXPERT_BLOCK)
        expert_mask = expert_offsets < num_experts
        expert_counts = tl.load(
            expert_counts_ptr + expert_offsets,
            mask=expert_mask,
            other=0,
        )
        expert_blocks = (expert_counts + BLOCK_M - 1) // BLOCK_M
        block_ends = tl.cumsum(expert_blocks, axis=0)
        block_starts = block_ends - expert_blocks
        owns_block = (
            expert_mask
            & (global_row_block >= block_starts)
            & (global_row_block < block_ends)
        )
        active = tl.sum(owns_block.to(tl.int32), axis=0) != 0
    if active:
        if ROUTE_MAJOR:
            expert_index = tl.load(sorted_token_ptr + global_row_block)
            sorted_row_base = global_row_block
            row_offsets = tl.arange(0, BLOCK_M)
            rows_valid = row_offsets == 0
            token_ids = tl.full((BLOCK_M,), 0, dtype=tl.int64)
            if ROUTE_INPUT:
                token_ids += global_row_block
        elif USE_TILES:
            sorted_row_base = tl.load(tile_ptr + 1)
            row_offsets = tl.arange(0, BLOCK_M)
            rows_valid = row_offsets < tl.load(tile_ptr + 2)
            token_ids = sorted_row_base + row_offsets
            if not ROUTE_INPUT:
                token_ids = tl.load(
                    sorted_token_ptr + token_ids, mask=rows_valid, other=0
                )
        else:
            expert_index = tl.sum(tl.where(owns_block, expert_offsets, 0), axis=0)
            row_block = global_row_block - tl.sum(
                tl.where(owns_block, block_starts, 0), axis=0
            )
            if COMPACT:
                token_starts = tl.cumsum(expert_counts, axis=0) - expert_counts
            else:
                token_starts = expert_offsets * token_capacity
            sorted_row_base = tl.sum(tl.where(owns_block, token_starts, 0), axis=0)
            expert_count = tl.sum(tl.where(owns_block, expert_counts, 0), axis=0)
            row_offsets = row_block * BLOCK_M + tl.arange(0, BLOCK_M)
            rows_valid = row_offsets < expert_count
            token_ids = tl.load(
                sorted_token_ptr + sorted_row_base + row_offsets,
                mask=rows_valid,
                other=0,
            )

        n_offsets = column_block * BLOCK_N + tl.arange(0, BLOCK_N)
        n_valid = n_offsets < n_size
        packed_base = (
            packed_ptr + expert_index * n_size * (k_size // 2)
        )
        scale_base = (
            scale_zero_ptr
            + expert_index * (k_size // group_size) * n_size * 2
        )
        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        k_begin = split_k_index * K_PER_SLICE
        k_end = tl.minimum(k_begin + K_PER_SLICE, k_size)
        k_start = k_begin
        while k_start < k_end:
            # Even-k codes live in the low nibbles, odd-k codes in the high
            # nibbles of the same bytes.  Running one dot per nibble plane
            # keeps the k order consistent between x and W without any
            # interleave.
            k_evens = k_start + 2 * tl.arange(0, HALF_K)
            k_odds = k_evens + 1
            byte_tile = tl.load(
                packed_base
                + n_offsets[:, None] * (k_size // 2)
                + k_evens[None, :] // 2,
                mask=n_valid[:, None],
                other=0,
            )
            if BROADCAST_SCALE and group_size % BLOCK_K == 0:
                meta = scale_base + (k_start // group_size) * n_size * 2 + n_offsets * 2
                scale = tl.load(meta, mask=n_valid, other=0.0).to(tl.float32)[:, None]
                zero = tl.load(meta + 1, mask=n_valid, other=0.0).to(tl.float32)[:, None]
                weight_even = ((byte_tile & 0xF).to(tl.float32) - zero) * scale
                weight_odd = (((byte_tile >> 4) & 0xF).to(tl.float32) - zero) * scale
            else:
                meta_even = (
                    scale_base
                    + (k_evens // group_size)[None, :] * n_size * 2
                    + n_offsets[:, None] * 2
                )
                meta_odd = (
                    scale_base
                    + (k_odds // group_size)[None, :] * n_size * 2
                    + n_offsets[:, None] * 2
                )
                scale_even = tl.load(meta_even, mask=n_valid[:, None], other=0.0)
                zero_even = tl.load(meta_even + 1, mask=n_valid[:, None], other=0.0)
                scale_odd = tl.load(meta_odd, mask=n_valid[:, None], other=0.0)
                zero_odd = tl.load(meta_odd + 1, mask=n_valid[:, None], other=0.0)
                weight_even = (
                    (byte_tile & 0xF).to(tl.float32) - zero_even.to(tl.float32)
                ) * scale_even.to(tl.float32)
                weight_odd = (
                    ((byte_tile >> 4) & 0xF).to(tl.float32) - zero_odd.to(tl.float32)
                ) * scale_odd.to(tl.float32)
            x_even = tl.load(
                x_ptr
                + token_ids[:, None] * k_size
                + k_evens[None, :],
                mask=rows_valid[:, None],
                other=0.0,
            )
            x_odd = tl.load(
                x_ptr
                + token_ids[:, None] * k_size
                + k_odds[None, :],
                mask=rows_valid[:, None],
                other=0.0,
            )
            acc = tl.dot(x_even, tl.trans(weight_even.to(x_even.dtype)), acc)
            acc = tl.dot(x_odd, tl.trans(weight_odd.to(x_odd.dtype)), acc)
            k_start += BLOCK_K
        if SPLIT_K > 1:
            partial_ptrs = (
                partial_ptr
                + (split_k_index * token_capacity + sorted_row_base + row_offsets)[
                    :, None
                ]
                * n_size
                + n_offsets[None, :]
            )
            tl.store(
                partial_ptrs,
                acc,
                mask=rows_valid[:, None] & n_valid[None, :],
            )
        else:
            out_ptrs = (
                out_ptr
                + (sorted_row_base + row_offsets)[:, None] * n_size
                + n_offsets[None, :]
            )
            tl.store(
                out_ptrs,
                acc.to(out_ptr.dtype.element_ty),
                mask=rows_valid[:, None] & n_valid[None, :],
            )


@triton.jit
def _route_major_int4_gemm(
    x_ptr,
    expert_ids_ptr,
    tiles_ptr,
    packed_ptr,
    scale_zero_ptr,
    out_ptr,
    ROUTES: tl.constexpr,
    TOP_K: tl.constexpr,
    K: tl.constexpr,
    N: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
    ROUTE_INPUT: tl.constexpr,
    K_TILES: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    ADAPTIVE: tl.constexpr,
):
    """Run one tensor-core tile for each routed expert and output block."""
    run_route = True
    if ADAPTIVE:
        run_route = tl.load(tiles_ptr) == 0
    if run_route:
        route = tl.program_id(0)
        split = tl.program_id(2)
        expert = tl.load(expert_ids_ptr + route).to(tl.int32)
        columns = tl.program_id(1) * BLOCK_N + tl.arange(0, BLOCK_N)
        packed = packed_ptr + expert * N * (K // 2)
        metadata = scale_zero_ptr + expert * (K // GROUP_SIZE) * N * 2
        first_row = tl.arange(0, 16) == 0
        pairs = tl.arange(0, BLOCK_K // 2)
        row = route if ROUTE_INPUT else route // TOP_K
        accumulator = tl.full((16, BLOCK_N), 0.0, tl.float32)
        for tile in tl.range(0, K_TILES):
            k_start = (split * K_TILES + tile) * BLOCK_K
            valid_k = k_start < K
            valid_n = columns < N
            scales = metadata + ((k_start // GROUP_SIZE) * N + columns) * 2
            scale = tl.load(scales, valid_n & valid_k, 0).to(tl.float32)
            zero = tl.load(scales + 1, valid_n & valid_k, 0).to(tl.float32)
            codes = tl.load(
                packed + columns[:, None] * (K // 2) + k_start // 2 + pairs[None, :],
                valid_n[:, None] & valid_k,
                0,
            )
            weight_even = ((codes & 15).to(tl.float32) - zero[:, None]) * scale[:, None]
            weight_odd = ((codes >> 4).to(tl.float32) - zero[:, None]) * scale[:, None]
            k_even = k_start + 2 * pairs
            x_even = tl.load(
                x_ptr + row * K + k_even[None, :], first_row[:, None] & valid_k, 0
            )
            x_odd = tl.load(
                x_ptr + row * K + k_even[None, :] + 1, first_row[:, None] & valid_k, 0
            )
            accumulator = tl.dot(x_even, tl.trans(weight_even.to(x_even.dtype)), accumulator)
            accumulator = tl.dot(x_odd, tl.trans(weight_odd.to(x_odd.dtype)), accumulator)
        addresses = (split * ROUTES + route) * N + columns[None, :]
        addresses = tl.broadcast_to(addresses, (16, BLOCK_N))
        tl.store(
            out_ptr + addresses,
            accumulator,
            first_row[:, None] & (columns[None, :] < N),
        )


@triton.jit
def _int4_route_index_kernel(
    flat_ids_ptr,
    expert_counts_ptr,
    expert_cursors_ptr,
    sorted_token_ptr,
    route_to_row_ptr,
    tiles_ptr,
    num_experts: tl.constexpr,
    top_k: tl.constexpr,
    EXPERT_BLOCK: tl.constexpr,
    ADAPTIVE: tl.constexpr,
):
    """Place flat route ids into compact, expert-major rows on device."""

    route_index = tl.program_id(0)
    use_grouped = True
    if ADAPTIVE:
        use_grouped = tl.load(tiles_ptr) != 0
    if use_grouped:
        expert_index = tl.load(flat_ids_ptr + route_index)
        expert_offsets = tl.arange(0, EXPERT_BLOCK)
        expert_mask = expert_offsets < num_experts
        expert_counts = tl.load(
            expert_counts_ptr + expert_offsets,
            mask=expert_mask,
            other=0,
        )
        expert_start = tl.sum(
            tl.where(
                expert_mask & (expert_offsets < expert_index),
                expert_counts,
                0,
            ),
            axis=0,
        )
        expert_slot = tl.atomic_add(expert_cursors_ptr + expert_index, 1)
        sorted_row = expert_start + expert_slot
        # The up GEMM indexes x directly by this value, so store the token id;
        # the reduction needs the flat route id to recover its weight.
        tl.store(sorted_token_ptr + sorted_row, route_index // top_k)
        tl.store(route_to_row_ptr + route_index, sorted_row)
    else:
        tl.store(route_to_row_ptr + route_index, route_index)


@triton.jit
def _moe_int4_reduce_kernel(
    contribution_ptr,
    route_to_row_ptr,
    route_weights_ptr,
    output_ptr,
    top_k: tl.constexpr,
    hidden_size: tl.constexpr,
    BLOCK_H: tl.constexpr,
    ROUTE_MAJOR: tl.constexpr = False,
):
    """Deterministically sum the weighted routes of each token in FP32."""

    token_index = tl.program_id(0)
    hidden_offsets = tl.arange(0, BLOCK_H)
    hidden_mask = hidden_offsets < hidden_size
    accumulator = tl.zeros((BLOCK_H,), tl.float32)
    for slot in range(top_k):
        route_index = token_index * top_k + slot
        weight = tl.load(route_weights_ptr + route_index).to(tl.float32)
        if ROUTE_MAJOR:
            sorted_row = route_index
        else:
            sorted_row = tl.load(route_to_row_ptr + route_index)
        values = tl.load(
            contribution_ptr + sorted_row * hidden_size + hidden_offsets,
            mask=hidden_mask,
            other=0.0,
        ).to(tl.float32)
        accumulator += values * weight
    tl.store(
        output_ptr + token_index * hidden_size + hidden_offsets,
        accumulator.to(output_ptr.dtype.element_ty),
        mask=hidden_mask,
    )


@triton.jit
def _split_k_reduce_kernel(
    partial_ptr,
    out_ptr,
    rows,
    n_size: tl.constexpr,
    SPLIT_K: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """Sum the K slices of one GEMM in FP32 before the nonlinear epilogue."""

    row = tl.program_id(0)
    n_offsets = tl.program_id(1) * BLOCK_N + tl.arange(0, BLOCK_N)
    valid = n_offsets < n_size
    accumulator = tl.zeros((BLOCK_N,), tl.float32)
    for split in range(SPLIT_K):
        accumulator += tl.load(
            partial_ptr + (split * rows + row) * n_size + n_offsets,
            mask=valid,
            other=0.0,
        ).to(tl.float32)
    tl.store(
        out_ptr + row * n_size + n_offsets,
        accumulator.to(out_ptr.dtype.element_ty),
        mask=valid,
    )


@dataclass(slots=True)
class Int4MoEWorkspace:
    """Reusable scratch for :func:`routed_moe_grouped_int4`.

    The grouped int4 MoE runs once per layer visit; keeping the sort/count
    metadata and both GEMM outputs outside ``forward`` removes per-layer
    allocations and gives CUDA Graphs stable addresses.
    """

    max_tokens: int
    num_experts: int
    hidden_size: int
    out_width13: int
    top_k: int
    capacity: int
    expert_counts: tensorplay.Tensor
    expert_cursors: tensorplay.Tensor
    sorted_token: tensorplay.Tensor
    route_to_row: tensorplay.Tensor
    identity_rows: tensorplay.Tensor
    tiles: tensorplay.Tensor
    gate_up: tensorplay.Tensor
    contribution: tensorplay.Tensor
    output: tensorplay.Tensor
    up_partial: tensorplay.Tensor
    down_partial: tensorplay.Tensor

    @classmethod
    def allocate(
        cls,
        *,
        max_tokens: int,
        num_experts: int,
        hidden_size: int,
        out_width13: int,
        top_k: int,
        dtype: tensorplay.dtype,
        device: tensorplay.device | str,
    ) -> Int4MoEWorkspace:
        if min(max_tokens, num_experts, hidden_size, out_width13, top_k) <= 0:
            raise ValueError("int4 MoE workspace dimensions must be positive")
        capacity = max_tokens * top_k
        small_capacity = min(max_tokens, _SMALL_GROUPED_MAX_TOKENS) * top_k
        target_device = tensorplay.device(device)
        return cls(
            max_tokens=max_tokens,
            num_experts=num_experts,
            hidden_size=hidden_size,
            out_width13=out_width13,
            top_k=top_k,
            capacity=capacity,
            expert_counts=tensorplay.zeros(
                num_experts,
                dtype=tensorplay.int32,
                device=target_device,
            ),
            expert_cursors=tensorplay.zeros(
                num_experts,
                dtype=tensorplay.int32,
                device=target_device,
            ),
            sorted_token=tensorplay.empty(
                capacity,
                dtype=tensorplay.int64,
                device=target_device,
            ),
            route_to_row=tensorplay.empty(
                capacity,
                dtype=tensorplay.int32,
                device=target_device,
            ),
            identity_rows=tensorplay.arange(
                capacity,
                dtype=tensorplay.int64,
                device=target_device,
            ),
            tiles=tensorplay.empty(
                (_small_tile_capacity(small_capacity, num_experts) + 1, 3),
                dtype=tensorplay.int32,
                device=target_device,
            ),
            gate_up=tensorplay.empty(
                (capacity, out_width13),
                dtype=dtype,
                device=target_device,
            ),
            contribution=tensorplay.empty(
                (capacity, hidden_size),
                dtype=dtype,
                device=target_device,
            ),
            output=tensorplay.empty(
                (max_tokens, hidden_size),
                dtype=dtype,
                device=target_device,
            ),
            up_partial=tensorplay.empty(
                (_MAX_SPLIT_K, top_k, out_width13),
                dtype=tensorplay.float32,
                device=target_device,
            ),
            down_partial=tensorplay.empty(
                (_MAX_SPLIT_K, top_k, hidden_size),
                dtype=tensorplay.float32,
                device=target_device,
            ),
        )

    @staticmethod
    def estimate_nbytes(
        *,
        max_tokens: int,
        num_experts: int,
        hidden_size: int,
        out_width13: int,
        top_k: int,
        dtype: tensorplay.dtype,
    ) -> int:
        if min(max_tokens, num_experts, hidden_size, out_width13, top_k) <= 0:
            raise ValueError("int4 MoE workspace dimensions must be positive")
        capacity = max_tokens * top_k
        small_capacity = min(max_tokens, _SMALL_GROUPED_MAX_TOKENS) * top_k
        element_size = tensorplay.empty((), dtype=dtype).element_size()
        return (
            2 * num_experts * 4
            + capacity * 8
            + capacity * 8
            + capacity * 4
            + (_small_tile_capacity(small_capacity, num_experts) + 1) * 3 * 4
            + capacity * out_width13 * element_size
            + capacity * hidden_size * element_size
            + max_tokens * hidden_size * element_size
            + _MAX_SPLIT_K * top_k * (out_width13 + hidden_size) * 4
        )

    @property
    def nbytes(self) -> int:
        tensors = (
            self.expert_counts,
            self.expert_cursors,
            self.sorted_token,
            self.route_to_row,
            self.identity_rows,
            self.tiles,
            self.gate_up,
            self.contribution,
            self.output,
            self.up_partial,
            self.down_partial,
        )
        return sum(
            tensor.numel() * tensor.element_size() for tensor in tensors
        )


def pack_int4_expert_rowmajor(
    weight: tensorplay.Tensor, group_size: int
) -> tuple[tensorplay.Tensor, tensorplay.Tensor]:
    """Quantise ``[N, K]`` into the layout :data:`_grouped_int4_gemm_kernel` reads.

    Returns ``(packed, scale_zero)`` with ``packed`` ``[N, K/2]`` uint8
    (low nibble = even k) and ``scale_zero`` ``[K/group, N, 2]``, where the
    second column stores ``8 + zero`` so the kernel's
    ``(code - zero) * scale`` matches the symmetric quantiser directly.
    """

    from tensorplay.ao.quantization.int4_weight_only import (
        _quantize_int4_groupwise,
    )

    codes, scale, zero = _quantize_int4_groupwise(weight, group_size)
    low = codes[:, 0::2]
    high = codes[:, 1::2]
    packed = (low | (high << 4)).to(tensorplay.uint8)
    scale_zero = tensorplay.stack(
        [scale.t().contiguous(), (zero.t() + 8).contiguous()],
        dim=-1,
    )
    return packed, scale_zero


def stack_int4_experts(
    packed_list: list[tensorplay.Tensor],
    scale_zero_list: list[tensorplay.Tensor],
) -> tuple[tensorplay.Tensor, tensorplay.Tensor]:
    """Stack per-expert int4 tensors into one contiguous device tensor.

    The stacked tensors replace the per-expert lists on the layer (slicing
    the result keeps the eager segmented path working unchanged), so total
    resident memory does not grow beyond a transient copy.
    """

    packed = tensorplay.stack(
        [item.contiguous() for item in packed_list], dim=0
    )
    scale_zero = tensorplay.stack(
        [item.contiguous() for item in scale_zero_list], dim=0
    )
    return packed, scale_zero


def _project_int4(
    x: tensorplay.Tensor,
    route_rows: tensorplay.Tensor,
    flat_ids: tensorplay.Tensor,
    expert_counts: tensorplay.Tensor,
    tiles: tensorplay.Tensor,
    packed: tensorplay.Tensor,
    scale_zero: tensorplay.Tensor,
    output: tensorplay.Tensor,
    partial: tensorplay.Tensor,
    *,
    capacity: int,
    group_size: int,
    num_experts: int,
    grid_rows: int,
    split_count: int,
    block_k: int,
    top_k: int,
    route_major: bool,
    use_tiles: bool,
    adaptive: bool,
    route_input: bool = False,
) -> None:
    k_size = x.shape[1]
    n_size = output.shape[1]
    k_per_slice = _k_per_slice(k_size, split_count)
    fast_route = (
        (route_major or adaptive)
        and k_size % _UP_BLOCK_K == 0
        and group_size % _UP_BLOCK_K == 0
        and packed.numel() < 2**31
        and x.dtype in (tensorplay.bfloat16, tensorplay.float16)
    )
    if fast_route:
        # Keep the reference partition boundaries and ignore wholly empty
        # slices, which would only contribute zero to the reduction.
        split_count = triton.cdiv(k_size, k_per_slice)
        _route_major_int4_gemm[
            (capacity, triton.cdiv(n_size, _ROUTE_MAJOR_BLOCK_N), split_count)
        ](
            x,
            flat_ids,
            tiles,
            packed,
            scale_zero,
            partial if split_count > 1 else output,
            capacity,
            top_k,
            k_size,
            n_size,
            group_size,
            route_input,
            k_per_slice // _UP_BLOCK_K,
            _ROUTE_MAJOR_BLOCK_N,
            _UP_BLOCK_K,
            adaptive,
            num_warps=4,
            num_stages=3,
        )
    if not fast_route or adaptive:
        grid_cols = triton.cdiv(n_size, _UP_BLOCK_N)
        _grouped_int4_gemm_kernel[(grid_rows, grid_cols, split_count)](
            x,
            route_rows,
            expert_counts,
            tiles,
            packed,
            scale_zero,
            output,
            partial,
            token_capacity=capacity,
            SPLIT_K=split_count,
            K_PER_SLICE=k_per_slice,
            group_size=group_size,
            num_experts=num_experts,
            k_size=k_size,
            n_size=n_size,
            COMPACT=True,
            EXPERT_BLOCK=triton.next_power_of_2(num_experts),
            BLOCK_M=_SMALL_BLOCK_M if use_tiles else _UP_BLOCK_M,
            BLOCK_N=_UP_BLOCK_N,
            BLOCK_K=block_k,
            HALF_K=block_k // 2,
            ROUTE_MAJOR=route_major,
            ROUTE_INPUT=route_input and (route_major or use_tiles),
            BROADCAST_SCALE=group_size % block_k == 0,
            USE_TILES=use_tiles,
            ADAPTIVE=adaptive,
        )
    if split_count > 1:
        _split_k_reduce_kernel[(capacity, triton.cdiv(n_size, 256))](
            partial, output, capacity, n_size, split_count, 256, num_warps=4
        )


def routed_moe_grouped_int4(
    hidden_states: tensorplay.Tensor,
    w13_packed: tensorplay.Tensor,
    w13_scale_zero: tensorplay.Tensor,
    w2_packed: tensorplay.Tensor,
    w2_scale_zero: tensorplay.Tensor,
    topk_ids: tensorplay.Tensor,
    topk_weights: tensorplay.Tensor,
    *,
    group_size: int,
    num_experts: int,
    gate_softcap: float,
    up_softcap: float,
    interleaved: bool,
    cap_fn,
    workspace: Int4MoEWorkspace | None = None,
) -> tensorplay.Tensor:
    """Capture-safe grouped int4 MoE forward for one layer.

    ``topk_ids``/``topk_weights`` are the routing decision of shape
    ``[T, top]``; every stage stays on device with static shapes so the
    sequence can be captured into a CUDA graph.  A reusable ``workspace``
    keeps the sorted-route metadata and both GEMM outputs at stable addresses.
    """

    token_count, top = topk_ids.shape
    hidden_size = hidden_states.shape[1]
    intermediate_size = w2_packed.shape[2] * 2
    # ``w13_packed`` stacks ``[E, N, K/2]`` codes, so ``shape[1]`` is already
    # the full (interleaved gate+up) output width.
    out_width13 = w13_packed.shape[1]
    device = hidden_states.device
    flat_ids = topk_ids.reshape(-1).long()
    flat_gate = topk_weights.reshape(-1)
    if flat_gate.dtype != tensorplay.float32:
        flat_gate = flat_gate.float()
    capacity = flat_ids.shape[0]
    if workspace is not None:
        if (
            workspace.max_tokens < token_count
            or workspace.top_k != top
            or workspace.num_experts != num_experts
            or workspace.hidden_size != hidden_size
            or workspace.out_width13 != out_width13
            or workspace.capacity < capacity
        ):
            raise ValueError("int4 MoE workspace geometry does not match")
        expert_counts = workspace.expert_counts
        expert_cursors = workspace.expert_cursors
        tiles = workspace.tiles
        sorted_token = workspace.sorted_token[:capacity]
        route_to_row = workspace.route_to_row[:capacity]
        gate_up = workspace.gate_up[:capacity]
        contribution = workspace.contribution[:capacity]
        output = workspace.output[:token_count]
    else:
        small_capacity = min(token_count, _SMALL_GROUPED_MAX_TOKENS) * top
        expert_counts = tensorplay.zeros(
            num_experts,
            dtype=tensorplay.int32,
            device=device,
        )
        expert_cursors = tensorplay.zeros(
            num_experts,
            dtype=tensorplay.int32,
            device=device,
        )
        tiles = tensorplay.empty(
            (_small_tile_capacity(small_capacity, num_experts) + 1, 3),
            dtype=tensorplay.int32,
            device=device,
        )
        sorted_token = tensorplay.empty(
            capacity,
            dtype=tensorplay.int64,
            device=device,
        )
        route_to_row = tensorplay.empty(
            capacity,
            dtype=tensorplay.int32,
            device=device,
        )
        gate_up = tensorplay.empty(
            (capacity, out_width13),
            dtype=hidden_states.dtype,
            device=device,
        )
        contribution = tensorplay.empty(
            (capacity, hidden_size),
            dtype=hidden_states.dtype,
            device=device,
        )
        output = tensorplay.empty(
            (token_count, hidden_size),
            dtype=hidden_states.dtype,
            device=device,
        )
    single_token = token_count == 1
    route_eligible = (
        group_size % _UP_BLOCK_K == 0
        and hidden_size % _UP_BLOCK_K == 0
        and intermediate_size % _UP_BLOCK_K == 0
        and max(w13_packed.numel(), w2_packed.numel()) < 2**31
        and hidden_states.dtype in (tensorplay.bfloat16, tensorplay.float16)
    )
    route_major = single_token or (
        token_count <= _ROUTE_MAJOR_MAX_TOKENS and route_eligible
    )
    use_tiles = not route_major and token_count <= _SMALL_GROUPED_MAX_TOKENS
    # Both alternatives can be captured with fixed grids. A device flag
    # disables the unused projection without reading the router on the host.
    adaptive = use_tiles and route_eligible and _ROUTE_MAJOR_OVERRIDE is None
    split_up = _SINGLE_TOKEN_SPLIT_UP if single_token else 1
    split_down = _SINGLE_TOKEN_SPLIT_DOWN if single_token else 1
    if split_up > 1 or split_down > 1:
        if workspace is not None:
            # Split-K only runs for one token and has exactly top_k routes.
            up_partial = workspace.up_partial
            down_partial = workspace.down_partial
        else:
            up_partial = tensorplay.empty(
                (split_up, capacity, out_width13),
                dtype=tensorplay.float32,
                device=device,
            )
            down_partial = tensorplay.empty(
                (split_down, capacity, hidden_size),
                dtype=tensorplay.float32,
                device=device,
            )
    else:
        up_partial = gate_up
        down_partial = contribution
    if route_major:
        # For a few tokens, direct routes avoid sorting and empty expert tiles.
        # Both GEMMs and the final reduction retain the original slot order.
        grid_rows = capacity
        up_rows = down_rows = flat_ids
    else:
        expert_counts.zero_()
        expert_cursors.zero_()
        count_block = 256
        _moe_route_count_kernel[(triton.cdiv(capacity, count_block),)](
            flat_ids,
            expert_counts,
            capacity,
            BLOCK=count_block,
            num_warps=4,
        )
        expert_block = triton.next_power_of_2(num_experts)
        if use_tiles:
            grid_rows = _small_tile_capacity(capacity, num_experts)
            _small_grouped_int4_tiles[(grid_rows,)](
                expert_counts,
                tiles,
                capacity,
                num_experts,
                expert_block,
                _SMALL_BLOCK_M,
                adaptive,
                num_warps=4,
            )
        else:
            grid_rows = triton.cdiv(capacity, _UP_BLOCK_M) + num_experts - 1
        _int4_route_index_kernel[(capacity,)](
            flat_ids,
            expert_counts,
            expert_cursors,
            sorted_token,
            route_to_row,
            tiles,
            num_experts=num_experts,
            top_k=top,
            EXPERT_BLOCK=expert_block,
            ADAPTIVE=adaptive,
            num_warps=1,
        )
        up_rows = sorted_token
        down_rows = sorted_token if use_tiles else (
            workspace.identity_rows[:capacity]
            if workspace is not None
            else tensorplay.arange(capacity, device=device)
        )

    block_k = min(_UP_BLOCK_K, hidden_size, intermediate_size)
    _project_int4(
        hidden_states.contiguous(),
        up_rows,
        flat_ids,
        expert_counts,
        tiles,
        w13_packed,
        w13_scale_zero,
        gate_up,
        up_partial,
        capacity=capacity,
        group_size=group_size,
        num_experts=num_experts,
        grid_rows=grid_rows,
        split_count=split_up,
        block_k=block_k,
        top_k=top,
        route_major=route_major,
        use_tiles=use_tiles,
        adaptive=adaptive,
    )
    if interleaved:
        gate = gate_up[:, 0::2]
        up = gate_up[:, 1::2]
    else:
        half = out_width13 // 2
        gate = gate_up[:, :half]
        up = gate_up[:, half:]
    hidden_act = cap_fn(gate, up, gate_softcap, up_softcap).to(
        hidden_states.dtype
    )

    # Only the up projection gathers inputs by token. The down projection
    # reads the activation rows in the order established by the up pass.
    _project_int4(
        hidden_act,
        down_rows,
        flat_ids,
        expert_counts,
        tiles,
        w2_packed,
        w2_scale_zero,
        contribution,
        down_partial,
        capacity=capacity,
        group_size=group_size,
        num_experts=num_experts,
        grid_rows=grid_rows,
        split_count=split_down,
        block_k=block_k,
        top_k=top,
        route_major=route_major,
        use_tiles=use_tiles,
        adaptive=adaptive,
        route_input=True,
    )
    # One deterministic FP32 reduction replaces ``zeros_like + index_add``;
    # the route weights stay in (token, slot) order, so no extra gather of
    # the sorted gate values is needed.
    _moe_int4_reduce_kernel[(token_count,)](
        contribution,
        route_to_row,
        flat_gate,
        output,
        top_k=top,
        hidden_size=hidden_size,
        BLOCK_H=triton.next_power_of_2(hidden_size),
        ROUTE_MAJOR=route_major,
        num_warps=4,
    )
    return output


_BLOCK_M_OVERRIDE = os.environ.get("MEGALM_MOE_BLOCK_M")
_UP_BLOCK_M = 32 if _BLOCK_M_OVERRIDE is None else int(_BLOCK_M_OVERRIDE)
_SMALL_BLOCK_M = 16 if _BLOCK_M_OVERRIDE is None else _UP_BLOCK_M
_SMALL_GROUPED_MAX_TOKENS = 32
_UP_BLOCK_N = int(os.environ.get("MEGALM_MOE_BLOCK_N", "128"))
_UP_BLOCK_K = 128
# Decode and small-prefill keep the narrow column tile: the route-major grid
# is latency-bound and the wider tile measured ~6% slower there, while the
# grouped prefill grid is bandwidth-bound and prefers the wide tile.
_ROUTE_MAJOR_BLOCK_N = int(os.environ.get("MEGALM_MOE_ROUTE_BLOCK_N", "64"))
# Small prefills are latency-bound, not throughput-bound: below this many
# tokens the expert-grouped grid pads a BLOCK_M row tile per routed expert
# (128 experts on Inkling-2), so most launched tiles only read a counter and
# exit.  llama.cpp answers the same regime with mat-vec kernels; the
# route-major path below is the equivalent here and launches exactly
# ``token_count * top_k`` row tiles instead.
_ROUTE_MAJOR_OVERRIDE = os.environ.get("MEGALM_ROUTE_MAJOR_MAX_TOKENS")
_ROUTE_MAJOR_MAX_TOKENS = 8 if _ROUTE_MAJOR_OVERRIDE is None else int(_ROUTE_MAJOR_OVERRIDE)
# Tuned K-split for inkling2 decode: single-token GEMMs have only top_k CTAs,
# so splitting K fills the SMs. See the shape sweep in
# tests/test_inkling2_int4_cuda.py.
_MAX_SPLIT_K = 8
_SINGLE_TOKEN_SPLIT_UP = 1
_SINGLE_TOKEN_SPLIT_DOWN = 8


def _small_tile_capacity(capacity: int, num_experts: int) -> int:
    # The first tile of an expert needs one route. Each additional tile needs
    # a full block of additional routes. The worst case maximizes used experts.
    active = min(num_experts, capacity)
    return active + (capacity - active) // _SMALL_BLOCK_M


def _k_per_slice(k_size: int, split: int) -> int:
    return triton.cdiv(triton.cdiv(k_size, split), _UP_BLOCK_K) * _UP_BLOCK_K
