"""Sampling primitives for DiffusionGemma block diffusion.

The sampler mirrors the reference *entropy-bound* scheme: a denoising step
predicts every canvas position, but only a subset of positions whose joint
mutual information stays below ``entropy_bound`` is committed.  Rejected
positions are re-noised with fresh uniform tokens, which keeps the block a
valid sample from the model's reverse process instead of freezing early
mistakes.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import tensorplay

_NEGATIVE_INF = float("-inf")


@dataclass(frozen=True, slots=True)
class DiffusionSamplingConfig:
    """Generation knobs, matching ``generation_config.json`` defaults."""

    max_new_tokens: int = 256
    max_denoising_steps: int = 48
    entropy_bound: float = 0.1
    t_min: float = 0.4
    t_max: float = 0.8
    stability_threshold: int = 1
    confidence_threshold: float = 0.005
    eos_token_ids: tuple[int, ...] = (1,)
    pad_token_id: int = 0
    seed: int | None = None

    def validate(self, canvas_length: int) -> None:
        if self.max_new_tokens <= 0:
            raise ValueError("max_new_tokens must be positive")
        if self.max_denoising_steps <= 0:
            raise ValueError("max_denoising_steps must be positive")
        if self.entropy_bound <= 0:
            raise ValueError("entropy_bound must be positive")
        if not 0 < self.t_min <= self.t_max:
            raise ValueError("temperature schedule must satisfy 0 < t_min <= t_max")
        if self.stability_threshold < 0:
            raise ValueError("stability_threshold cannot be negative")
        if self.confidence_threshold <= 0:
            raise ValueError("confidence_threshold must be positive")
        if canvas_length <= 0:
            raise ValueError("canvas_length must be positive")

    @classmethod
    def from_generation_config(
        cls,
        payload: dict,
        *,
        canvas_length: int,
    ) -> "DiffusionSamplingConfig":
        """Build from a HF ``generation_config.json`` payload."""

        sampler = payload.get("sampler_config") or {}
        eos = payload.get("eos_token_id", (1,))
        if isinstance(eos, int):
            eos = (eos,)
        config = cls(
            max_new_tokens=int(payload.get("max_new_tokens") or canvas_length),
            max_denoising_steps=int(payload.get("max_denoising_steps") or 48),
            entropy_bound=float(sampler.get("entropy_bound", 0.1)),
            t_min=float(payload.get("t_min", 0.4)),
            t_max=float(payload.get("t_max", 0.8)),
            stability_threshold=int(payload.get("stability_threshold", 1)),
            confidence_threshold=float(payload.get("confidence_threshold", 0.005)),
            eos_token_ids=tuple(int(value) for value in eos),
            pad_token_id=int(payload.get("pad_token_id", 0)),
        )
        config.validate(canvas_length)
        return config


def temperature_for_step(
    cur_step: int, max_denoising_steps: int, t_min: float, t_max: float
) -> float:
    """Linear schedule, ``cur_step`` counting *down* to the final step."""

    return t_min + (t_max - t_min) * (cur_step / max_denoising_steps)


def token_entropy(logits: tensorplay.Tensor) -> tensorplay.Tensor:
    """Entropy of ``softmax(logits)`` in nats, computed in fp32."""

    probabilities = tensorplay.softmax(logits.float(), dim=-1)
    return -(
        probabilities
        * tensorplay.log(probabilities.clamp_min(1e-30))
    ).sum(-1)


def entropy_bound_mask(
    logits: tensorplay.Tensor,
    entropy_bound: float,
    *,
    entropy: tensorplay.Tensor | None = None,
) -> tensorplay.Tensor:
    """Accept the lowest-entropy positions whose joint bound fits.

    Sorting entropies ascending gives the reference criterion
    ``sum(entropy[:k]) - max(entropy[:k]) <= bound``; the running sum minus
    the current value is exactly the cumulative entropy *excluding* the
    current token.  Callers that already computed :func:`token_entropy` can
    pass it in to avoid a second full-vocabulary softmax.
    """

    if entropy is None:
        entropy = token_entropy(logits)
    sorted_entropy, sorted_indices = tensorplay.sort(entropy, dim=-1)
    cumulative = tensorplay.cumsum(sorted_entropy, dim=-1)
    selection = (cumulative - sorted_entropy) <= entropy_bound
    return tensorplay.scatter(
        tensorplay.zeros_like(selection),
        -1,
        sorted_indices,
        selection,
    )


class EntropyBoundSampler:
    """Stateful canvas initialiser / acceptor / re-noiser."""

    def __init__(self, config: DiffusionSamplingConfig, *, vocab_size: int) -> None:
        self.entropy_bound = config.entropy_bound
        self.vocab_size = vocab_size

    def initialize_canvas(
        self,
        batch_size: int,
        canvas_length: int,
        *,
        device: tensorplay.device,
    ) -> tensorplay.Tensor:
        return tensorplay.randint(
            low=0,
            high=self.vocab_size,
            size=(batch_size, canvas_length),
            device=device,
            dtype=tensorplay.int64,
        )

    def accept_and_renoise(
        self,
        current_canvas: tensorplay.Tensor,
        denoiser_canvas: tensorplay.Tensor,
        logits: tensorplay.Tensor,
        *,
        entropy: tensorplay.Tensor | None = None,
    ) -> tuple[tensorplay.Tensor, tensorplay.Tensor]:
        """Commit accepted positions and re-noise the rest.

        Returns ``(accepted_mask, next_canvas)``.  Accepted positions take the
        denoiser token; every other position is replaced by a fresh uniform
        draw, not by its previous value.
        """

        mask = entropy_bound_mask(logits, self.entropy_bound, entropy=entropy)
        accepted = tensorplay.where(mask, denoiser_canvas, current_canvas)
        random_canvas = self.initialize_canvas(
            accepted.shape[0],
            accepted.shape[1],
            device=accepted.device,
        )
        return mask, tensorplay.where(mask, accepted, random_canvas)


class StableAndConfidentStopping:
    """Stop when the argmax canvas is stable and the block is confident."""

    def __init__(
        self,
        *,
        stability_threshold: int,
        confidence_threshold: float,
    ) -> None:
        self.stability_threshold = stability_threshold
        self.confidence_threshold = confidence_threshold
        self.history: tensorplay.Tensor | None = None

    def reset(self) -> None:
        self.history = None

    def __call__(
        self,
        argmax_canvas: tensorplay.Tensor,
        logits: tensorplay.Tensor,
        *,
        entropy: tensorplay.Tensor | None = None,
    ) -> tensorplay.Tensor:
        batch = argmax_canvas.shape[0]
        if self.stability_threshold == 0:
            stable = tensorplay.ones(
                (batch,), dtype=tensorplay.bool, device=argmax_canvas.device
            )
        else:
            if self.history is None:
                self.history = tensorplay.full(
                    (
                        self.stability_threshold,
                        argmax_canvas.shape[0],
                        argmax_canvas.shape[1],
                    ),
                    -1,
                    dtype=argmax_canvas.dtype,
                    device=argmax_canvas.device,
                )
            stable = (
                (self.history == argmax_canvas.unsqueeze(0))
                .all(dim=-1)
                .all(dim=0)
            )
            self.history = tensorplay.cat(
                (self.history[1:], argmax_canvas.unsqueeze(0)), dim=0
            )
        if entropy is None:
            entropy = token_entropy(logits)
        confident = entropy.mean(-1) < self.confidence_threshold
        return stable & confident


__all__ = [
    "DiffusionSamplingConfig",
    "EntropyBoundSampler",
    "StableAndConfidentStopping",
    "entropy_bound_mask",
    "temperature_for_step",
    "token_entropy",
]
