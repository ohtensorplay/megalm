from __future__ import annotations

import tensorplay
import triton
import triton.language as tl

from megalm.kernels.kv_quant_triton import (
    load_dequant_kv,
    rht_enabled,
    rht_output,
    rht_query,
)
from megalm.kernels.paged_decode_gqa import (
    gqa_decode_attention,
    gqa_decode_supported,
)
from megalm.runtime.kv_quant import (
    dequant_kernel_kwargs,
    head_scale_tensor,
    kv_quant_of,
    kv_scale_of,
)


@triton.jit
def _paged_decode_single_kernel(
    query_ptr,
    key_cache_ptr,
    value_cache_ptr,
    key_scale_ptr,
    value_scale_ptr,
    head_scale_k_ptr,
    head_scale_v_ptr,
    block_tables_ptr,
    sequence_lengths_ptr,
    output_ptr,
    max_blocks,
    page_size: tl.constexpr,
    num_query_heads: tl.constexpr,
    num_kv_heads: tl.constexpr,
    head_dim: tl.constexpr,
    query_per_kv: tl.constexpr,
    scale: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_N: tl.constexpr,
    stride_slot,
    stride_head,
    ss_page,
    ss_pos,
    ss_head,
    MODE_K: tl.constexpr,
    MODE_V: tl.constexpr,
    SCALAR_K,
    SCALAR_V,
    GLOBAL_SCALE,
    FP8_DTYPE: tl.constexpr,
    SLIDING_WINDOW: tl.constexpr = 0,
):
    batch_index = tl.program_id(0)
    query_head = tl.program_id(1)
    kv_head = query_head // query_per_kv
    sequence_length = tl.load(sequence_lengths_ptr + batch_index)
    dimensions = tl.arange(0, BLOCK_D)
    dimension_mask = dimensions < head_dim
    query_base = (
        batch_index * num_query_heads * head_dim + query_head * head_dim
    )
    query = tl.load(
        query_ptr + query_base + dimensions,
        mask=dimension_mask,
        other=0.0,
    ).to(tl.float32)

    running_max = tl.full((), -float("inf"), tl.float32)
    running_sum = tl.zeros((), tl.float32)
    accumulator = tl.zeros((BLOCK_D,), tl.float32)
    if SLIDING_WINDOW > 0:
        # Skip blocks entirely outside the window instead of loading and
        # masking them; the trailing partial block is still masked below.
        token_start = tl.maximum(0, sequence_length - SLIDING_WINDOW)
    else:
        token_start = 0
    while token_start < sequence_length:
        token_offsets = token_start + tl.arange(0, BLOCK_N)
        token_mask = token_offsets < sequence_length
        if SLIDING_WINDOW > 0:
            # Decode query sits at the last position: keep the trailing window.
            token_mask = token_mask & (
                token_offsets >= sequence_length - SLIDING_WINDOW
            )
        logical_pages = token_offsets // page_size
        page_offsets = token_offsets % page_size
        physical_pages = tl.load(
            block_tables_ptr
            + batch_index * max_blocks
            + logical_pages,
            mask=token_mask,
            other=0,
        )
        slots = physical_pages * page_size + page_offsets
        matrix_mask = token_mask[:, None] & dimension_mask[None, :]
        keys = load_dequant_kv(
            key_cache_ptr,
            key_scale_ptr,
            head_scale_k_ptr,
            slots,
            kv_head,
            dimensions,
            stride_slot,
            stride_head,
            ss_page,
            ss_pos,
            ss_head,
            matrix_mask,
            page_size,
            head_dim,
            MODE_K,
            SCALAR_K,
            GLOBAL_SCALE,
            FP8_DTYPE,
        )
        scores = tl.sum(keys * query[None, :], axis=1) * scale
        scores = tl.where(token_mask, scores, -float("inf"))
        block_max = tl.max(scores, axis=0)
        new_max = tl.maximum(running_max, block_max)
        # A fully window-masked block has max -inf; clamp so exp(-inf - -inf)
        # cannot produce nan. Such blocks contribute zero probability anyway.
        new_max = tl.where(new_max > -float("inf") / 2, new_max, 0.0)
        old_scale = tl.exp(running_max - new_max)
        probabilities = tl.where(
            token_mask,
            tl.exp(scores - new_max),
            0.0,
        )
        values = load_dequant_kv(
            value_cache_ptr,
            value_scale_ptr,
            head_scale_v_ptr,
            slots,
            kv_head,
            dimensions,
            stride_slot,
            stride_head,
            ss_page,
            ss_pos,
            ss_head,
            matrix_mask,
            page_size,
            head_dim,
            MODE_V,
            SCALAR_V,
            GLOBAL_SCALE,
            FP8_DTYPE,
        )
        accumulator = (
            accumulator * old_scale
            + tl.sum(probabilities[:, None] * values, axis=0)
        )
        running_sum = (
            running_sum * old_scale + tl.sum(probabilities, axis=0)
        )
        running_max = new_max
        token_start += BLOCK_N

    tl.store(
        output_ptr + query_base + dimensions,
        accumulator / running_sum,
        mask=dimension_mask,
    )


@triton.jit
def _paged_decode_split_kernel(
    query_ptr,
    key_cache_ptr,
    value_cache_ptr,
    key_scale_ptr,
    value_scale_ptr,
    head_scale_k_ptr,
    head_scale_v_ptr,
    block_tables_ptr,
    sequence_lengths_ptr,
    partial_max_ptr,
    partial_sum_ptr,
    partial_output_ptr,
    max_blocks,
    page_size: tl.constexpr,
    num_query_heads: tl.constexpr,
    num_kv_heads: tl.constexpr,
    head_dim: tl.constexpr,
    query_per_kv: tl.constexpr,
    num_splits: tl.constexpr,
    scale: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_N: tl.constexpr,
    stride_slot,
    stride_head,
    ss_page,
    ss_pos,
    ss_head,
    MODE_K: tl.constexpr,
    MODE_V: tl.constexpr,
    SCALAR_K,
    SCALAR_V,
    GLOBAL_SCALE,
    FP8_DTYPE: tl.constexpr,
    SLIDING_WINDOW: tl.constexpr = 0,
):
    batch_index = tl.program_id(0)
    query_head = tl.program_id(1)
    split_index = tl.program_id(2)
    kv_head = query_head // query_per_kv
    sequence_length = tl.load(sequence_lengths_ptr + batch_index)
    if SLIDING_WINDOW > 0:
        split_origin = tl.maximum(0, sequence_length - SLIDING_WINDOW)
        split_length = sequence_length - split_origin
    else:
        split_origin = 0
        split_length = sequence_length
    tokens_per_split = (split_length + num_splits - 1) // num_splits
    split_start = split_origin + split_index * tokens_per_split
    split_end = tl.minimum(split_start + tokens_per_split, sequence_length)
    dimensions = tl.arange(0, BLOCK_D)
    dimension_mask = dimensions < head_dim
    query_base = (
        batch_index * num_query_heads * head_dim + query_head * head_dim
    )
    query = tl.load(
        query_ptr + query_base + dimensions,
        mask=dimension_mask,
        other=0.0,
    ).to(tl.float32)

    running_max = tl.full((), -float("inf"), tl.float32)
    running_sum = tl.zeros((), tl.float32)
    accumulator = tl.zeros((BLOCK_D,), tl.float32)
    token_start = split_start
    while token_start < split_end:
        token_offsets = token_start + tl.arange(0, BLOCK_N)
        token_mask = token_offsets < split_end
        if SLIDING_WINDOW > 0:
            token_mask = token_mask & (
                token_offsets >= sequence_length - SLIDING_WINDOW
            )
        logical_pages = token_offsets // page_size
        page_offsets = token_offsets % page_size
        physical_pages = tl.load(
            block_tables_ptr + batch_index * max_blocks + logical_pages,
            mask=token_mask,
            other=0,
        )
        slots = physical_pages * page_size + page_offsets
        matrix_mask = token_mask[:, None] & dimension_mask[None, :]
        keys = load_dequant_kv(
            key_cache_ptr,
            key_scale_ptr,
            head_scale_k_ptr,
            slots,
            kv_head,
            dimensions,
            stride_slot,
            stride_head,
            ss_page,
            ss_pos,
            ss_head,
            matrix_mask,
            page_size,
            head_dim,
            MODE_K,
            SCALAR_K,
            GLOBAL_SCALE,
            FP8_DTYPE,
        )
        scores = tl.sum(keys * query[None, :], axis=1) * scale
        scores = tl.where(token_mask, scores, -float("inf"))
        block_max = tl.max(scores, axis=0)
        new_max = tl.maximum(running_max, block_max)
        # A fully window-masked block has max -inf; clamp so exp(-inf - -inf)
        # cannot produce nan. Such blocks contribute zero probability anyway.
        new_max = tl.where(new_max > -float("inf") / 2, new_max, 0.0)
        old_scale = tl.exp(running_max - new_max)
        probabilities = tl.exp(scores - new_max)
        probabilities = tl.where(token_mask, probabilities, 0.0)
        values = load_dequant_kv(
            value_cache_ptr,
            value_scale_ptr,
            head_scale_v_ptr,
            slots,
            kv_head,
            dimensions,
            stride_slot,
            stride_head,
            ss_page,
            ss_pos,
            ss_head,
            matrix_mask,
            page_size,
            head_dim,
            MODE_V,
            SCALAR_V,
            GLOBAL_SCALE,
            FP8_DTYPE,
        )
        accumulator = (
            accumulator * old_scale
            + tl.sum(probabilities[:, None] * values, axis=0)
        )
        running_sum = running_sum * old_scale + tl.sum(probabilities, axis=0)
        running_max = new_max
        token_start += BLOCK_N

    partial_index = (
        (batch_index * num_query_heads + query_head) * num_splits
        + split_index
    )
    tl.store(partial_max_ptr + partial_index, running_max)
    tl.store(partial_sum_ptr + partial_index, running_sum)
    tl.store(
        partial_output_ptr + partial_index * head_dim + dimensions,
        accumulator,
        mask=dimension_mask,
    )


@triton.jit
def _paged_decode_reduce_kernel(
    partial_max_ptr,
    partial_sum_ptr,
    partial_output_ptr,
    output_ptr,
    num_query_heads: tl.constexpr,
    head_dim: tl.constexpr,
    num_splits: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_S: tl.constexpr,
):
    batch_index = tl.program_id(0)
    query_head = tl.program_id(1)
    split_offsets = tl.arange(0, BLOCK_S)
    split_mask = split_offsets < num_splits
    partial_base = (
        (batch_index * num_query_heads + query_head) * num_splits
    )
    partial_max = tl.load(
        partial_max_ptr + partial_base + split_offsets,
        mask=split_mask,
        other=-float("inf"),
    ).to(tl.float32)
    global_max = tl.max(partial_max, axis=0)
    rescale = tl.exp(partial_max - global_max)
    partial_sum = tl.load(
        partial_sum_ptr + partial_base + split_offsets,
        mask=split_mask,
        other=0.0,
    ).to(tl.float32)
    denominator = tl.sum(partial_sum * rescale, axis=0)

    dimensions = tl.arange(0, BLOCK_D)
    dimension_mask = dimensions < head_dim
    partial_offsets = (
        (partial_base + split_offsets[:, None]) * head_dim
        + dimensions[None, :]
    )
    partial_values = tl.load(
        partial_output_ptr + partial_offsets,
        mask=split_mask[:, None] & dimension_mask[None, :],
        other=0.0,
    ).to(tl.float32)
    numerator = tl.sum(partial_values * rescale[:, None], axis=0)
    output_base = (
        batch_index * num_query_heads * head_dim + query_head * head_dim
    )
    tl.store(
        output_ptr + output_base + dimensions,
        numerator / denominator,
        mask=dimension_mask,
    )


def paged_decode_attention(
    query: tensorplay.Tensor,
    key_cache: tensorplay.Tensor,
    value_cache: tensorplay.Tensor,
    block_tables: tensorplay.Tensor,
    sequence_lengths: tensorplay.Tensor,
    page_size: int,
    *,
    output: tensorplay.Tensor | None = None,
    scale: float | None = None,
    sliding_window: int = 0,
) -> tensorplay.Tensor:
    """Paged decode attention.

    ``scale`` overrides the default ``head_dim**-0.5`` score scale.  Models
    with unit-normalised Q/K (for example Gemma-4) pass ``scale=1.0``.
    ``sliding_window`` (0 disables) keeps only the trailing window of keys.
    """
    query = query.contiguous()
    block_tables = block_tables.contiguous()
    sequence_lengths = sequence_lengths.contiguous()
    if output is None:
        output = tensorplay.empty_like(query)
    elif (
        output.shape != query.shape
        or output.device != query.device
        or output.dtype != query.dtype
        or not output.is_contiguous()
    ):
        raise ValueError("paged decode output must match contiguous query")
    num_query_heads = query.shape[1]
    num_kv_heads = key_cache.shape[-2]
    head_dim = query.shape[-1]
    quant = kv_quant_of(key_cache)
    use_rht = rht_enabled(quant)
    if use_rht:
        query = rht_query(query, quant, head_dim)
    base_scale = head_dim**-0.5 if scale is None else scale
    softmax_scale = base_scale / head_dim if use_rht else base_scale
    key_scale = kv_scale_of(key_cache)
    value_scale = kv_scale_of(value_cache)
    head_scale_k = head_scale_tensor(
        quant, "key", query.device, key_cache
    )
    head_scale_v = head_scale_tensor(
        quant, "value", query.device, value_cache
    )
    if key_scale is None:
        key_scale = key_cache
        value_scale = value_cache
    k_kwargs = dequant_kernel_kwargs(quant, "key")
    v_kwargs = dequant_kernel_kwargs(quant, "value")
    common = {
        "stride_slot": key_cache.stride(1),
        "stride_head": key_cache.stride(2),
        "ss_page": key_scale.stride(0),
        "ss_pos": key_scale.stride(1),
        "ss_head": key_scale.stride(2),
        "MODE_K": k_kwargs["MODE"],
        "MODE_V": v_kwargs["MODE"],
        "SCALAR_K": k_kwargs["SCALAR_SCALE"],
        "SCALAR_V": v_kwargs["SCALAR_SCALE"],
        "GLOBAL_SCALE": k_kwargs["GLOBAL_SCALE"],
        "FP8_DTYPE": k_kwargs["FP8_DTYPE"],
    }
    pointer_kwargs = {
        "key_scale_ptr": key_scale,
        "value_scale_ptr": value_scale,
        "head_scale_k_ptr": head_scale_k,
        "head_scale_v_ptr": head_scale_v,
    }
    block_d = triton.next_power_of_2(head_dim)
    if gqa_decode_supported(query, key_cache, num_kv_heads, quant):
        # One program per K/V head so the cache is streamed once, not once per
        # query head.
        gqa_decode_attention(
            query,
            key_cache,
            value_cache,
            block_tables,
            sequence_lengths,
            page_size,
            output=output,
            scale=softmax_scale,
            sliding_window=sliding_window,
        )
        return output
    max_sequence_length = block_tables.shape[1] * page_size
    # Split-K exists to fill the GPU when there are few (batch x head)
    # programs; with enough of them it only adds partial buffers and a second
    # launch.  Aim for ~256 concurrent programs.
    programs = query.shape[0] * num_query_heads
    num_splits = 1
    if programs < 256:
        wanted = triton.next_power_of_2(max(1, 256 // programs))
        num_splits = min(8, wanted)
    if num_splits <= 1:
        if max_sequence_length <= 512:
            if query.shape[0] == 1 and max_sequence_length > 128:
                block_n = 256
                num_warps = 8
            elif query.shape[0] <= 16 and max_sequence_length > 128:
                block_n = 256
                num_warps = 4
            elif query.shape[0] == 1 and max_sequence_length > 64:
                block_n = 128
                num_warps = 2
            else:
                block_n = 64
                num_warps = 2
        else:
            # Long loops: a 256-token block of FP32 keys/values would spill,
            # so keep the resident tile small and iterate more.
            block_n = 64
            num_warps = 4
        _paged_decode_single_kernel[(query.shape[0], num_query_heads)](
            query_ptr=query,
            key_cache_ptr=key_cache,
            value_cache_ptr=value_cache,
            **pointer_kwargs,
            block_tables_ptr=block_tables,
            sequence_lengths_ptr=sequence_lengths,
            output_ptr=output,
            max_blocks=block_tables.shape[1],
            page_size=page_size,
            num_query_heads=num_query_heads,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            query_per_kv=num_query_heads // num_kv_heads,
            scale=softmax_scale,
            BLOCK_D=block_d,
            BLOCK_N=block_n,
            SLIDING_WINDOW=int(sliding_window),
            **common,
            num_warps=num_warps,
        )
        return rht_output(output, quant, head_dim)
    partial_shape = (query.shape[0], num_query_heads, num_splits)
    partial_max = tensorplay.empty(
        partial_shape,
        dtype=tensorplay.float32,
        device=query.device,
    )
    partial_sum = tensorplay.empty_like(partial_max)
    partial_output = tensorplay.empty(
        (*partial_shape, head_dim),
        dtype=tensorplay.float32,
        device=query.device,
    )
    _paged_decode_split_kernel[
        (query.shape[0], num_query_heads, num_splits)
    ](
        query_ptr=query,
        key_cache_ptr=key_cache,
        value_cache_ptr=value_cache,
        **pointer_kwargs,
        block_tables_ptr=block_tables,
        sequence_lengths_ptr=sequence_lengths,
        partial_max_ptr=partial_max,
        partial_sum_ptr=partial_sum,
        partial_output_ptr=partial_output,
        max_blocks=block_tables.shape[1],
        page_size=page_size,
        num_query_heads=num_query_heads,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        query_per_kv=num_query_heads // num_kv_heads,
        num_splits=num_splits,
        scale=softmax_scale,
        BLOCK_D=block_d,
        BLOCK_N=32,
        SLIDING_WINDOW=int(sliding_window),
        **common,
        num_warps=4,
    )
    _paged_decode_reduce_kernel[(query.shape[0], num_query_heads)](
        partial_max,
        partial_sum,
        partial_output,
        output,
        num_query_heads=num_query_heads,
        head_dim=head_dim,
        num_splits=num_splits,
        BLOCK_D=block_d,
        BLOCK_S=triton.next_power_of_2(num_splits),
        num_warps=4,
    )
    return rht_output(output, quant, head_dim)
