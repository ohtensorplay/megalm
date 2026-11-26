"""Full-vocabulary categorical sampling and entropy for block diffusion."""

from __future__ import annotations

import tensorplay
import triton
import triton.language as tl
from triton.language.extra.cuda import libdevice


@triton.jit
def _scale_logits(x, inv_temperature, SOFTCAP: tl.constexpr):
    x = x.to(tl.float32)
    if SOFTCAP != 0.0:
        x = SOFTCAP * libdevice.tanh(x / SOFTCAP)
    return x * inv_temperature


@triton.jit
def _summarize_logits(
    logits,
    conditioning,
    parts,
    winners,
    inv_temperature,
    VOCAB: tl.constexpr,
    CHUNKS: tl.constexpr,
    BLOCK: tl.constexpr,
    SOFTCAP: tl.constexpr,
):
    row = tl.program_id(0)
    chunk = tl.program_id(1)
    indices = chunk * BLOCK + tl.arange(0, BLOCK)
    valid = indices < VOCAB
    x = tl.load(logits + row * VOCAB + indices, valid, other=0.0)
    x = tl.where(valid, _scale_logits(x, inv_temperature, SOFTCAP), -float("inf"))
    dtype: tl.constexpr = conditioning.dtype.element_ty
    rounded = x.to(dtype).to(tl.float32)
    tl.store(conditioning + row * VOCAB + indices, rounded, valid)

    maximum = tl.max(x, axis=0)
    shifted = tl.where(maximum == -float("inf"), -float("inf"), x - maximum)
    weight = tl.exp(shifted)
    entropy_mass = weight * tl.where(shifted == -float("inf"), 0.0, -shifted)
    winner = tl.min(
        tl.where((indices < VOCAB) & (x == maximum), indices, VOCAB), axis=0
    )
    round_max = maximum.to(dtype).to(tl.float32)
    round_mass = tl.exp(
        tl.where(round_max == -float("inf"), -float("inf"), rounded - round_max)
    )
    base = row * CHUNKS * 4 + chunk
    tl.store(parts + base, maximum)
    tl.store(parts + base + CHUNKS, tl.sum(weight, axis=0))
    tl.store(parts + base + CHUNKS * 2, tl.sum(entropy_mass, axis=0))
    tl.store(parts + base + CHUNKS * 3, tl.sum(round_mass, axis=0))
    tl.store(winners + row * CHUNKS + chunk, winner)


@triton.jit
def _sample_logits(
    logits,
    conditioning,
    parts,
    winners,
    uniforms,
    sampled,
    argmax,
    entropy,
    norms,
    inv_temperature,
    VOCAB: tl.constexpr,
    CHUNKS: tl.constexpr,
    BLOCK: tl.constexpr,
    BLOCK_CHUNKS: tl.constexpr,
    SOFTCAP: tl.constexpr,
):
    row = tl.program_id(0)
    chunk = tl.arange(0, BLOCK_CHUNKS)
    valid = chunk < CHUNKS
    base = row * CHUNKS * 4 + chunk
    maxima = tl.load(parts + base, valid, other=-float("inf"))
    maximum = tl.max(maxima, axis=0)
    scale = tl.exp(maxima - maximum)
    masses = tl.load(parts + base + CHUNKS, valid, other=0.0) * scale
    total = tl.sum(masses, axis=0)
    terms = tl.load(parts + base + CHUNKS * 2, valid, other=0.0)
    offsets = tl.where(maxima == -float("inf"), 0.0, maximum - maxima)
    ent = tl.log(total) + tl.sum(terms * scale + masses * offsets, axis=0) / total

    ids = tl.load(winners + row * CHUNKS + chunk, valid, other=VOCAB)
    tl.store(argmax + row, tl.min(tl.where(maxima == maximum, ids, VOCAB), axis=0))
    tl.store(entropy + row, ent)

    dtype: tl.constexpr = conditioning.dtype.element_ty
    round_max = maximum.to(dtype).to(tl.float32)
    round_maxima = maxima.to(dtype).to(tl.float32)
    round_mass = tl.load(parts + base + CHUNKS * 3, valid, other=0.0)
    denominator = tl.sum(round_mass * tl.exp(round_maxima - round_max), axis=0)
    tl.store(norms + row * 2, round_max)
    tl.store(norms + row * 2 + 1, 1.0 / denominator)

    # Draw from the entire vocabulary. Locate the tile containing the uniform
    # quantile, then scan only that tile to obtain the selected token.
    prefix = tl.cumsum(masses, axis=0)
    target = tl.load(uniforms + row) * total
    selected = tl.min(tl.where((prefix > target) & valid, chunk, CHUNKS), axis=0)
    last_chunk = tl.max(tl.where(masses > 0, chunk, 0), axis=0)
    selected = tl.minimum(selected, last_chunk)
    lower = tl.max(tl.where(chunk < selected, prefix, 0.0), axis=0)
    local_max = tl.sum(tl.where(chunk == selected, maxima, 0.0), axis=0)
    local_target = tl.maximum(target - lower, 0.0) * tl.exp(maximum - local_max)

    indices = selected * BLOCK + tl.arange(0, BLOCK)
    x = tl.load(logits + row * VOCAB + indices, indices < VOCAB, other=0.0)
    scaled = _scale_logits(x, inv_temperature, SOFTCAP)
    weights = tl.where(indices < VOCAB, tl.exp(scaled - local_max), 0.0)
    local_prefix = tl.cumsum(weights, axis=0)
    token = tl.min(
        tl.where((local_prefix > local_target) & (indices < VOCAB), indices, VOCAB),
        axis=0,
    )
    last_token = tl.max(tl.where(weights > 0, indices, 0), axis=0)
    tl.store(sampled + row, tl.minimum(token, last_token))


@triton.jit
def _normalize_conditioning(
    conditioning,
    norms,
    VOCAB: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    indices = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    offset = row * VOCAB + indices
    maximum = tl.load(norms + row * 2)
    inverse = tl.load(norms + row * 2 + 1)
    x = tl.load(conditioning + offset, indices < VOCAB, other=0.0).to(tl.float32)
    tl.store(conditioning + offset, tl.exp(x - maximum) * inverse, indices < VOCAB)


def sample_diffusion_logits(
    logits: tensorplay.Tensor,
    uniforms: tensorplay.Tensor,
    temperature: float,
    conditioning_dtype: tensorplay.dtype,
    *,
    softcap: float = 0.0,
) -> tuple[tensorplay.Tensor, tensorplay.Tensor, tensorplay.Tensor, tensorplay.Tensor]:
    """Return sampled IDs, argmax IDs, FP32 entropy and conditioning probabilities."""
    shape = logits.shape[:-1]
    vocab = int(logits.shape[-1])
    rows = logits.numel() // vocab
    block = min(4096, triton.next_power_of_2(vocab))
    chunks = triton.cdiv(vocab, block)
    device = logits.device
    parts = tensorplay.empty((rows, 4, chunks), dtype=tensorplay.float32, device=device)
    winners = tensorplay.empty((rows, chunks), dtype=tensorplay.int32, device=device)
    norms = tensorplay.empty((rows, 2), dtype=tensorplay.float32, device=device)
    conditioning = tensorplay.empty(
        logits.shape, dtype=conditioning_dtype, device=device
    )
    sampled = tensorplay.empty(shape, dtype=tensorplay.int64, device=device)
    argmax = tensorplay.empty_like(sampled)
    entropy = tensorplay.empty(shape, dtype=tensorplay.float32, device=device)
    inverse = 1.0 / temperature
    _summarize_logits[(rows, chunks)](
        logits,
        conditioning,
        parts,
        winners,
        inverse,
        vocab,
        chunks,
        block,
        softcap,
        num_warps=8,
    )
    _sample_logits[(rows,)](
        logits,
        conditioning,
        parts,
        winners,
        uniforms,
        sampled,
        argmax,
        entropy,
        norms,
        inverse,
        vocab,
        chunks,
        block,
        triton.next_power_of_2(chunks),
        softcap,
        num_warps=8,
    )
    _normalize_conditioning[(rows, chunks)](
        conditioning,
        norms,
        vocab,
        block,
        num_warps=8,
    )
    return sampled, argmax, entropy, conditioning
