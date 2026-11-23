from __future__ import annotations

import tensorplay
import triton
import triton.language as tl


@triton.jit
def _causal_conv_prefill_kernel(
    x_ptr,
    state_ptr,
    weight_ptr,
    query_starts_ptr,
    positions_ptr,
    state_indices_ptr,
    output_ptr,
    x_row_stride: tl.constexpr,
    channels: tl.constexpr,
    history: tl.constexpr,
    MAX_TOKENS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    sequence = tl.program_id(0)
    channel_block = tl.program_id(1)
    channels_offset = channel_block * BLOCK + tl.arange(0, BLOCK)
    channel_mask = channels_offset < channels
    start = tl.load(query_starts_ptr + sequence)
    end = tl.load(query_starts_ptr + sequence + 1)
    state_index = tl.load(state_indices_ptr + sequence * 2).to(tl.int64)
    state_base = (
        state_ptr
        + state_index * channels * history
        + channels_offset * history
    )
    reset = tl.load(positions_ptr + start) == 0
    for index in range(history):
        prior = tl.load(
            state_base + index,
            mask=channel_mask,
            other=0.0,
        )
        tl.store(
            state_base + index,
            tl.where(reset, 0.0, prior),
            mask=channel_mask,
        )

    weight_base = weight_ptr + channels_offset * (history + 1)
    for step in range(MAX_TOKENS):
        token = start + step
        token_active = token < end
        x = tl.load(
            x_ptr + token * x_row_stride + channels_offset,
            mask=channel_mask & token_active,
            other=0.0,
        ).to(tl.float32)
        accumulator = x * tl.load(
            weight_base + history,
            mask=channel_mask,
            other=0.0,
        ).to(tl.float32)
        for index in range(history):
            prior = tl.load(
                state_base + index,
                mask=channel_mask,
                other=0.0,
            ).to(tl.float32)
            coefficient = tl.load(
                weight_base + index,
                mask=channel_mask,
                other=0.0,
            ).to(tl.float32)
            accumulator += prior * coefficient
            if index + 1 < history:
                shifted = tl.load(
                    state_base + index + 1,
                    mask=channel_mask,
                    other=0.0,
                )
                tl.store(
                    state_base + index,
                    shifted,
                    mask=channel_mask & token_active,
                )
        tl.store(
            state_base + history - 1,
            x,
            mask=channel_mask & token_active,
        )
        activated = accumulator * tl.sigmoid(accumulator)
        tl.store(
            output_ptr + token * channels + channels_offset,
            activated,
            mask=channel_mask & token_active,
        )


def causal_conv_prefill(
    mixed_qkv: tensorplay.Tensor,
    state: tensorplay.Tensor,
    weight: tensorplay.Tensor,
    query_starts: tensorplay.Tensor,
    positions: tensorplay.Tensor,
    state_indices: tensorplay.Tensor,
    *,
    max_query_tokens: int,
) -> tensorplay.Tensor:
    if mixed_qkv.ndim != 2 or state.ndim != 3 or weight.ndim != 3:
        raise ValueError("GDN prefill convolution geometry is invalid")
    if mixed_qkv.stride(1) != 1:
        mixed_qkv = mixed_qkv.contiguous()
    tokens, channels = mixed_qkv.shape
    sequences = query_starts.numel() - 1
    if (
        state.shape[1] != channels
        or weight.shape != (channels, 1, state.shape[2] + 1)
        or positions.shape != (tokens,)
        or state_indices.shape != (sequences, 2)
        or max_query_tokens <= 0
    ):
        raise ValueError("GDN prefill convolution metadata differs")
    output = tensorplay.empty_like(mixed_qkv)
    block = 256
    max_tokens = triton.next_power_of_2(max_query_tokens)
    _causal_conv_prefill_kernel[
        (sequences, triton.cdiv(channels, block))
    ](
        mixed_qkv,
        state,
        weight,
        query_starts,
        positions,
        state_indices,
        output,
        x_row_stride=mixed_qkv.stride(0),
        channels=channels,
        history=state.shape[2],
        MAX_TOKENS=max_tokens,
        BLOCK=block,
        num_warps=4,
    )
    return output


@triton.jit
def _packed_recurrent_prefill_kernel(
    mixed_qkv_ptr,
    a_ptr,
    b_ptr,
    A_log_ptr,
    dt_bias_ptr,
    state_ptr,
    query_starts_ptr,
    positions_ptr,
    state_indices_ptr,
    output_ptr,
    mixed_row_stride: tl.constexpr,
    a_row_stride: tl.constexpr,
    b_row_stride: tl.constexpr,
    key_heads: tl.constexpr,
    value_heads: tl.constexpr,
    key_dim: tl.constexpr,
    value_dim: tl.constexpr,
    key_width: tl.constexpr,
    conv_width: tl.constexpr,
    scale: tl.constexpr,
    MAX_TOKENS: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
):
    value_block = tl.program_id(0)
    sequence_head = tl.program_id(1)
    sequence = sequence_head // value_heads
    value_head = sequence_head % value_heads
    key_head = value_head // (value_heads // key_heads)
    start = tl.load(query_starts_ptr + sequence)
    end = tl.load(query_starts_ptr + sequence + 1)
    state_index = tl.load(
        state_indices_ptr + sequence * 2 + 1
    ).to(tl.int64)
    reset = tl.load(positions_ptr + start) == 0

    key_offsets = tl.arange(0, BK)
    value_offsets = value_block * BV + tl.arange(0, BV)
    key_mask = key_offsets < key_dim
    value_mask = value_offsets < value_dim
    state_mask = key_mask[:, None] & value_mask[None, :]
    state_offsets = (
        state_index * value_heads * key_dim * value_dim
        + value_head * key_dim * value_dim
        + key_offsets[:, None] * value_dim
        + value_offsets[None, :]
    )
    recurrent = tl.load(
        state_ptr + state_offsets,
        mask=state_mask,
        other=0.0,
    ).to(tl.float32)
    recurrent = tl.where(reset, 0.0, recurrent)
    A_log = tl.load(A_log_ptr + value_head).to(tl.float32)
    dt_bias = tl.load(dt_bias_ptr + value_head).to(tl.float32)

    for step in range(MAX_TOKENS):
        token = start + step
        token_active = token < end
        packed = mixed_qkv_ptr + token * mixed_row_stride
        query = tl.load(
            packed + key_head * key_dim + key_offsets,
            mask=key_mask & token_active,
            other=0.0,
        ).to(tl.float32)
        key = tl.load(
            packed + key_width + key_head * key_dim + key_offsets,
            mask=key_mask & token_active,
            other=0.0,
        ).to(tl.float32)
        value = tl.load(
            packed
            + 2 * key_width
            + value_head * value_dim
            + value_offsets,
            mask=value_mask & token_active,
            other=0.0,
        ).to(tl.float32)
        query *= tl.rsqrt(
            tl.sum(query * query, axis=0) + 1e-6
        ) * scale
        key *= tl.rsqrt(tl.sum(key * key, axis=0) + 1e-6)
        a_value = tl.load(
            a_ptr + token * a_row_stride + value_head,
            mask=token_active,
            other=0.0,
        ).to(tl.float32)
        b_value = tl.load(
            b_ptr + token * b_row_stride + value_head,
            mask=token_active,
            other=0.0,
        ).to(tl.float32)
        softplus_input = a_value + dt_bias
        softplus = tl.where(
            softplus_input <= 20.0,
            tl.log(1.0 + tl.exp(softplus_input)),
            softplus_input,
        )
        decay = tl.exp(-tl.exp(A_log) * softplus)
        beta = tl.sigmoid(b_value)
        updated = recurrent * decay
        memory = tl.sum(updated * key[:, None], axis=0)
        delta = (value - memory) * beta
        updated += key[:, None] * delta[None, :]
        recurrent = tl.where(token_active, updated, recurrent)
        result = tl.sum(recurrent * query[:, None], axis=0)
        output_offsets = (
            (token * value_heads + value_head) * value_dim
            + value_offsets
        )
        tl.store(
            output_ptr + output_offsets,
            result,
            mask=value_mask & token_active,
        )
    tl.store(
        state_ptr + state_offsets,
        recurrent,
        mask=state_mask,
    )


def packed_recurrent_prefill(
    mixed_qkv: tensorplay.Tensor,
    a: tensorplay.Tensor,
    b: tensorplay.Tensor,
    A_log: tensorplay.Tensor,
    dt_bias: tensorplay.Tensor,
    state: tensorplay.Tensor,
    query_starts: tensorplay.Tensor,
    positions: tensorplay.Tensor,
    state_indices: tensorplay.Tensor,
    *,
    num_key_heads: int,
    scale: float,
    max_query_tokens: int,
) -> tensorplay.Tensor:
    tokens = mixed_qkv.shape[0]
    sequences = query_starts.numel() - 1
    if (
        mixed_qkv.ndim != 2
        or state.ndim != 4
        or a.shape != b.shape
        or a.shape[0] != tokens
        or positions.shape != (tokens,)
        or state_indices.shape != (sequences, 2)
        or max_query_tokens <= 0
    ):
        raise ValueError("packed GDN prefill metadata differs")
    if mixed_qkv.stride(1) != 1:
        mixed_qkv = mixed_qkv.contiguous()
    if a.stride(1) != 1:
        a = a.contiguous()
    if b.stride(1) != 1:
        b = b.contiguous()
    value_heads, key_dim, value_dim = state.shape[1:]
    key_width = num_key_heads * key_dim
    conv_width = 2 * key_width + value_heads * value_dim
    if (
        mixed_qkv.shape[1] != conv_width
        or a.shape[1] != value_heads
        or A_log.shape != (value_heads,)
        or dt_bias.shape != (value_heads,)
        or value_heads % num_key_heads
    ):
        raise ValueError("packed GDN prefill head geometry differs")
    output = tensorplay.empty(
        (tokens, value_heads, value_dim),
        dtype=mixed_qkv.dtype,
        device=mixed_qkv.device,
    )
    block_k = triton.next_power_of_2(key_dim)
    block_v = 32 if value_dim >= 32 else triton.next_power_of_2(value_dim)
    max_tokens = triton.next_power_of_2(max_query_tokens)
    _packed_recurrent_prefill_kernel[
        (triton.cdiv(value_dim, block_v), sequences * value_heads)
    ](
        mixed_qkv,
        a,
        b,
        A_log,
        dt_bias,
        state,
        query_starts,
        positions,
        state_indices,
        output,
        mixed_row_stride=mixed_qkv.stride(0),
        a_row_stride=a.stride(0),
        b_row_stride=b.stride(0),
        key_heads=num_key_heads,
        value_heads=value_heads,
        key_dim=key_dim,
        value_dim=value_dim,
        key_width=key_width,
        conv_width=conv_width,
        scale=scale,
        MAX_TOKENS=max_tokens,
        BK=block_k,
        BV=block_v,
        num_warps=4,
    )
    return output


__all__ = ["causal_conv_prefill", "packed_recurrent_prefill"]
