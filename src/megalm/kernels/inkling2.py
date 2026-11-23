"""Inkling-2 block arithmetic: depth attention, residual sums, RMSNorm, gates."""

from __future__ import annotations

import math

import tensorplay

Tensor = tensorplay.Tensor


def residual_sum(inputs: tuple[Tensor, ...]) -> Tensor:
    """Sum same-shaped sources in FP32, then round once to the first dtype."""

    if not inputs:
        raise ValueError("residual sum needs at least one input")
    first = inputs[0]
    total = first.float()
    for term in inputs[1:]:
        if term.shape != first.shape:
            raise ValueError("residual sum inputs must share shape")
        total = total + term.float()
    return total.to(first.dtype)


def rms_norm(
    x: Tensor,
    weight: Tensor | None = None,
    eps: float = 1e-6,
    *,
    output_dtype: tensorplay.dtype | None = None,
) -> Tensor:
    """Unweighted or learned RMSNorm over the last axis."""

    if eps <= 0 or not math.isfinite(eps):
        raise ValueError("RMSNorm needs a positive finite epsilon")
    values = x.float()
    inverse = tensorplay.rsqrt(
        tensorplay.mean(values * values, dim=-1, keepdim=True) + eps
    )
    normed = values * inverse
    if weight is not None:
        normed = normed * weight.float()
    return normed.to(output_dtype or x.dtype)


def l2_clip(x: Tensor, threshold: float, eps: float = 1e-6) -> Tensor:
    """Scale each row down to ``threshold`` when its L2 norm exceeds it."""

    values = x.float()
    norm = tensorplay.sqrt(
        tensorplay.sum(values * values, dim=-1, keepdim=True)
    )
    coefficient = tensorplay.minimum(
        threshold / (norm + eps),
        tensorplay.ones_like(norm),
    )
    return (values * coefficient).to(x.dtype)


def log_scale_tau(
    positions: Tensor,
    floor: int,
    alpha: float,
) -> Tensor:
    """``1 + alpha * log(max((position + 1) / floor, 1))`` per position."""

    ratio = tensorplay.maximum(
        (positions.float() + 1.0) / float(floor),
        tensorplay.ones_like(positions.float()),
    )
    return 1.0 + alpha * tensorplay.log(ratio)


def block_attention_mix(
    blocks: Tensor,
    partial: Tensor,
    queries: Tensor,
    *,
    eps: float = 1e-6,
    normalize_values: bool = True,
) -> Tensor:
    """Depth pseudo-attention over completed residual blocks.

    ``blocks`` is ``[sources, tokens, hidden]`` (source 0 first), ``partial``
    is ``[tokens, hidden]`` (the running block sum), and ``queries`` is
    ``[q, hidden]`` of FP32 score weights already scaled by ``hidden**-0.5``.
    Both sources and values are RMS-normalised (without a gain) before the
    softmax when ``normalize_values`` is set.

    Returns ``[q, tokens, hidden]`` FP32 mixtures.
    """

    sources = tensorplay.cat(
        (blocks.float(), tensorplay.unsqueeze(partial.float(), 0).expand_as(blocks[:1])),
        dim=0,
    )
    # Normalise each source row in FP32, then stack back to
    # [sources, tokens, hidden].
    normalized = tensorplay.stack(
        [
            rms_norm(sources[index], None, eps, output_dtype=tensorplay.float32)
            for index in range(sources.shape[0])
        ],
        dim=0,
    )
    # logits[q, s, t] = sum_h weights[q, h] * normalized[s, t, h]
    logits = tensorplay.einsum("qh,sth->qst", queries.float(), normalized)
    probabilities = tensorplay.softmax(logits, dim=1)
    values = normalized if normalize_values else sources
    return tensorplay.einsum("qst,sth->qth", probabilities, values)


def message_gate(
    destination: Tensor,
    message: Tensor,
    *,
    down_weight: Tensor,
    forget_weight: Tensor,
    gain_weight: Tensor,
    eps: float = 1e-6,
) -> Tensor:
    """Gate ``message`` into one FP32 ``[tokens, hidden]`` residual mixture.

    ``h = [rms_norm(x), rms_norm(msg)] @ down.T`` in BF16, then
    ``x' = (1 - tanh(h @ forget.T)) * x + tanh(h @ gain.T) * message`` in FP32.
    """

    hidden = destination.shape[-1]
    stacked = tensorplay.stack((destination.float(), message.float()), dim=1)
    normalized = tensorplay.stack(
        [
            rms_norm(stacked[:, 0], None, eps, output_dtype=tensorplay.bfloat16),
            rms_norm(stacked[:, 1], None, eps, output_dtype=tensorplay.bfloat16),
        ],
        dim=1,
    ).reshape(-1, 2 * hidden)
    gate_hidden = tensorplay.matmul(
        normalized.to(tensorplay.bfloat16),
        down_weight.to(tensorplay.bfloat16).transpose(0, 1),
    ).float()
    forget = tensorplay.tanh(
        tensorplay.matmul(gate_hidden, forget_weight.float().transpose(0, 1))
    )
    gain = tensorplay.tanh(
        tensorplay.matmul(gate_hidden, gain_weight.float().transpose(0, 1))
    )
    return (1.0 - forget) * destination.float() + gain * message.float()


def short_convolution(
    x: Tensor,
    weight: Tensor,
    state: Tensor,
    *,
    dynamic: Tensor | None = None,
    use_residual: bool = True,
) -> tuple[Tensor, Tensor]:
    """Causal depthwise FIR with a ``[tokens, channels, kernel]`` state cache.

    Args:
        x: ``[tokens, channels]`` BF16 input.
        weight: ``[channels, kernel]`` static taps.
        state: ``[slots, channels, kernel - 1]`` cache; ``slots >= tokens``.
        dynamic: Optional ``[tokens, channels, kernel]`` per-token taps added
            to ``weight`` (full-attention K/V only).
        use_residual: Add the input to the convolution output.

    Returns:
        ``(output, state)`` where output is ``[tokens, channels]``.
    """

    tokens, channels = x.shape
    kernel = weight.shape[-1]
    if state.shape[1:] != (channels, kernel - 1):
        raise ValueError("short convolution state geometry mismatch")
    if dynamic is not None and tuple(dynamic.shape) != (tokens, channels, kernel):
        raise ValueError("dynamic taps must match the input and kernel")
    values = x.float()
    output = tensorplay.zeros_like(values)
    window = state[:tokens].float()
    for tap in range(kernel):
        if tap == 0:
            sample = values
        else:
            sample = window[:, :, tap - 1]
        taps = weight.float() if dynamic is None else (
            weight.float()[None, :, :]
            + dynamic.float()[:, :, tap]
        )
        if dynamic is None:
            output = output + sample * taps
        else:
            output = output + sample * taps
    if use_residual:
        output = output + values
    scale = (kernel * (2 if dynamic is not None else 1) + int(use_residual)) ** -0.5
    output = output * scale
    shifted = tensorplay.cat(
        (
            values.reshape(tokens, channels, 1),
            window[:, :, : kernel - 2],
        ),
        dim=2,
    )
    state[:tokens].copy_(shifted)
    return output.to(x.dtype), state


__all__ = [
    "block_attention_mix",
    "l2_clip",
    "log_scale_tau",
    "message_gate",
    "residual_sum",
    "rms_norm",
    "short_convolution",
]