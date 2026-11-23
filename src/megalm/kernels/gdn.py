from __future__ import annotations

import tensorplay
import triton
import triton.language as tl


@triton.jit
def _causal_conv_update_kernel(
    x_ptr,
    state_ptr,
    weight_ptr,
    state_indices_ptr,
    active_ptr,
    output_ptr,
    x_row_stride: tl.constexpr,
    channels: tl.constexpr,
    history: tl.constexpr,
    BLOCK: tl.constexpr,
):
    sequence = tl.program_id(0)
    channel_block = tl.program_id(1)
    offsets = channel_block * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < channels
    state_index = tl.load(state_indices_ptr + sequence).to(tl.int64)
    active = tl.load(active_ptr + sequence)
    x = tl.load(
        x_ptr + sequence * x_row_stride + offsets,
        mask=mask & active,
        other=0.0,
    ).to(tl.float32)
    state_base = (
        state_ptr
        + state_index * channels * history
        + offsets * history
    )
    weight_base = weight_ptr + offsets * (history + 1)
    accumulator = x * tl.load(
        weight_base + history,
        mask=mask & active,
        other=0.0,
    ).to(tl.float32)
    for index in range(history):
        prior = tl.load(
            state_base + index,
            mask=mask & active,
            other=0.0,
        ).to(tl.float32)
        coefficient = tl.load(
            weight_base + index,
            mask=mask & active,
            other=0.0,
        ).to(tl.float32)
        accumulator += prior * coefficient
        if index + 1 < history:
            shifted = tl.load(
                state_base + index + 1,
                mask=mask & active,
                other=0.0,
            )
            tl.store(
                state_base + index,
                shifted,
                mask=mask & active,
            )
    tl.store(
        state_base + history - 1,
        x,
        mask=mask & active,
    )
    activated = accumulator * tl.sigmoid(accumulator)
    tl.store(
        output_ptr + sequence * channels + offsets,
        activated,
        mask=mask,
    )


def causal_conv_update(
    mixed_qkv: tensorplay.Tensor,
    state: tensorplay.Tensor,
    weight: tensorplay.Tensor,
    state_indices: tensorplay.Tensor,
    active: tensorplay.Tensor,
) -> tensorplay.Tensor:
    if mixed_qkv.ndim != 2:
        raise ValueError("GDN convolution input must be [B, QKV]")
    if mixed_qkv.stride(1) != 1:
        mixed_qkv = mixed_qkv.contiguous()
    if state.ndim != 3 or weight.ndim != 3:
        raise ValueError("GDN convolution state and weight geometry is invalid")
    batch, channels = mixed_qkv.shape
    if (
        state.shape[1] != channels
        or weight.shape[0] != channels
        or weight.shape[1] != 1
        or state.shape[2] + 1 != weight.shape[2]
        or state_indices.shape != (batch,)
        or active.shape != (batch,)
    ):
        raise ValueError("GDN convolution decode metadata differs")
    output = tensorplay.empty_like(mixed_qkv)
    # Qwen3.5-0.8B has 6,144 convolution channels. A single power-of-two
    # program would pad that to 8,192 lanes and leave the rest of the GPU idle
    # for every one of its 18 GDN layers. Channel tiles are independent, so
    # spread them across CTAs while retaining one program dimension per
    # sequence for graph-stable decode batches.
    block = min(256, triton.next_power_of_2(channels))
    _causal_conv_update_kernel[
        (batch, triton.cdiv(channels, block))
    ](
        mixed_qkv,
        state,
        weight,
        state_indices,
        active,
        output,
        x_row_stride=mixed_qkv.stride(0),
        channels=channels,
        history=state.shape[2],
        BLOCK=block,
        num_warps=4,
    )
    return output


@triton.jit
def _packed_recurrent_decode_kernel(
    mixed_qkv_ptr,
    a_ptr,
    b_ptr,
    A_log_ptr,
    dt_bias_ptr,
    state_ptr,
    state_indices_ptr,
    active_ptr,
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
    BK: tl.constexpr,
    BV: tl.constexpr,
):
    value_block = tl.program_id(0)
    sequence_head = tl.program_id(1)
    sequence = sequence_head // value_heads
    value_head = sequence_head % value_heads
    key_head = value_head // (value_heads // key_heads)
    active = tl.load(active_ptr + sequence)
    state_index = tl.load(state_indices_ptr + sequence).to(tl.int64)

    key_offsets = tl.arange(0, BK)
    value_offsets = value_block * BV + tl.arange(0, BV)
    key_mask = key_offsets < key_dim
    value_mask = value_offsets < value_dim
    state_mask = key_mask[:, None] & value_mask[None, :] & active
    packed = mixed_qkv_ptr + sequence * mixed_row_stride
    query = tl.load(
        packed + key_head * key_dim + key_offsets,
        mask=key_mask & active,
        other=0.0,
    ).to(tl.float32)
    key = tl.load(
        packed + key_width + key_head * key_dim + key_offsets,
        mask=key_mask & active,
        other=0.0,
    ).to(tl.float32)
    value = tl.load(
        packed + 2 * key_width + value_head * value_dim + value_offsets,
        mask=value_mask & active,
        other=0.0,
    ).to(tl.float32)
    query *= tl.rsqrt(tl.sum(query * query, axis=0) + 1e-6) * scale
    key *= tl.rsqrt(tl.sum(key * key, axis=0) + 1e-6)

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
    a_value = tl.load(
        a_ptr + sequence * a_row_stride + value_head,
        mask=active,
        other=0.0,
    ).to(tl.float32)
    b_value = tl.load(
        b_ptr + sequence * b_row_stride + value_head,
        mask=active,
        other=0.0,
    ).to(tl.float32)
    A_log = tl.load(A_log_ptr + value_head).to(tl.float32)
    dt_bias = tl.load(dt_bias_ptr + value_head).to(tl.float32)
    softplus_input = a_value + dt_bias
    softplus = tl.where(
        softplus_input <= 20.0,
        tl.log(1.0 + tl.exp(softplus_input)),
        softplus_input,
    )
    decay = tl.exp(-tl.exp(A_log) * softplus)
    beta = tl.sigmoid(b_value)
    recurrent *= decay
    memory = tl.sum(recurrent * key[:, None], axis=0)
    delta = (value - memory) * beta
    recurrent += key[:, None] * delta[None, :]
    result = tl.sum(recurrent * query[:, None], axis=0)
    tl.store(
        state_ptr + state_offsets,
        recurrent,
        mask=state_mask,
    )
    output_offsets = (
        (sequence * value_heads + value_head) * value_dim
        + value_offsets
    )
    tl.store(
        output_ptr + output_offsets,
        result,
        mask=value_mask,
    )


def packed_recurrent_decode(
    mixed_qkv: tensorplay.Tensor,
    a: tensorplay.Tensor,
    b: tensorplay.Tensor,
    A_log: tensorplay.Tensor,
    dt_bias: tensorplay.Tensor,
    state: tensorplay.Tensor,
    state_indices: tensorplay.Tensor,
    active: tensorplay.Tensor,
    *,
    num_key_heads: int,
    scale: float,
) -> tensorplay.Tensor:
    if mixed_qkv.ndim != 2:
        raise ValueError("packed GDN input must be [B, QKV]")
    if mixed_qkv.stride(1) != 1:
        mixed_qkv = mixed_qkv.contiguous()
    if a.stride(1) != 1:
        a = a.contiguous()
    if b.stride(1) != 1:
        b = b.contiguous()
    batch = mixed_qkv.shape[0]
    if (
        state.ndim != 4
        or a.shape != b.shape
        or a.shape[0] != batch
        or state_indices.shape != (batch,)
        or active.shape != (batch,)
    ):
        raise ValueError("packed GDN recurrent metadata differs")
    value_heads, key_dim, value_dim = state.shape[1:]
    if (
        a.shape[1] != value_heads
        or A_log.shape != (value_heads,)
        or dt_bias.shape != (value_heads,)
        or value_heads % num_key_heads
    ):
        raise ValueError("packed GDN recurrent head geometry differs")
    key_width = num_key_heads * key_dim
    conv_width = 2 * key_width + value_heads * value_dim
    if mixed_qkv.shape[1] != conv_width:
        raise ValueError("packed GDN QKV width differs")
    output = tensorplay.empty(
        (batch, value_heads, value_dim),
        dtype=mixed_qkv.dtype,
        device=mixed_qkv.device,
    )
    block_k = triton.next_power_of_2(key_dim)
    block_v = 32 if value_dim >= 32 else triton.next_power_of_2(value_dim)
    _packed_recurrent_decode_kernel[
        (triton.cdiv(value_dim, block_v), batch * value_heads)
    ](
        mixed_qkv,
        a,
        b,
        A_log,
        dt_bias,
        state,
        state_indices,
        active,
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
        BK=block_k,
        BV=block_v,
        num_warps=4,
    )
    return output


@triton.jit
def _gated_rms_norm_kernel(
    x_ptr,
    gate_ptr,
    weight_ptr,
    output_ptr,
    x_token_stride: tl.constexpr,
    x_head_stride: tl.constexpr,
    gate_token_stride: tl.constexpr,
    gate_head_stride: tl.constexpr,
    heads: tl.constexpr,
    width: tl.constexpr,
    eps: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    token = row // heads
    head = row % heads
    offsets = tl.arange(0, BLOCK)
    mask = offsets < width
    source = tl.load(
        x_ptr
        + token * x_token_stride
        + head * x_head_stride
        + offsets,
        mask=mask,
        other=0.0,
    ).to(tl.float32)
    gate = tl.load(
        gate_ptr
        + token * gate_token_stride
        + head * gate_head_stride
        + offsets,
        mask=mask,
        other=0.0,
    ).to(tl.float32)
    weight = tl.load(
        weight_ptr + offsets,
        mask=mask,
        other=0.0,
    ).to(tl.float32)
    inverse_rms = tl.rsqrt(
        tl.sum(source * source, axis=0) / width + eps
    )
    result = source * inverse_rms * weight
    result *= gate * tl.sigmoid(gate)
    tl.store(
        output_ptr + row * width + offsets,
        result,
        mask=mask,
    )


def gated_rms_norm(
    hidden_states: tensorplay.Tensor,
    gate: tensorplay.Tensor,
    weight: tensorplay.Tensor,
    eps: float,
) -> tensorplay.Tensor:
    if hidden_states.shape != gate.shape or hidden_states.ndim != 3:
        raise ValueError("GDN gated norm expects matching [B, H, V] tensors")
    if hidden_states.stride(-1) != 1:
        hidden_states = hidden_states.contiguous()
    if gate.stride(-1) != 1:
        gate = gate.contiguous()
    width = hidden_states.shape[-1]
    output = tensorplay.empty_like(hidden_states)
    block = triton.next_power_of_2(width)
    _gated_rms_norm_kernel[
        (hidden_states.numel() // width,)
    ](
        hidden_states,
        gate,
        weight,
        output,
        x_token_stride=hidden_states.stride(0),
        x_head_stride=hidden_states.stride(1),
        gate_token_stride=gate.stride(0),
        gate_head_stride=gate.stride(1),
        heads=hidden_states.shape[1],
        width=width,
        eps=eps,
        BLOCK=block,
        num_warps=4,
    )
    return output


__all__ = [
    "causal_conv_update",
    "gated_rms_norm",
    "packed_recurrent_decode",
]
