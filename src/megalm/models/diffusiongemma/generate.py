"""End-to-end block-diffusion generation loop.

One request performs an encoder prefill, then repeatedly denoises a canvas of
``canvas_length`` tokens.  Each canvas costs ``max_denoising_steps`` decoder
passes, after which its argmax tokens are committed and encoded into the
prefix cache for the next block.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field

import tensorplay

from megalm.models.diffusiongemma.model import (
    DiffusionGemmaError,
    DiffusionGemmaForBlockDiffusion,
    DiffusionKVCache,
)
from megalm.models.diffusiongemma.sampler import (
    DiffusionSamplingConfig,
    EntropyBoundSampler,
    StableAndConfidentStopping,
    temperature_for_step,
    token_entropy,
)


@dataclass(slots=True)
class DiffusionGenerationResult:
    """Generated tokens plus the counters diffusion papers report."""

    sequences: list[list[int]]
    new_tokens: list[int]
    denoising_steps: list[int]
    decoder_forward_passes: list[int]
    elapsed_seconds: float
    tokens_per_second: float
    tokens_per_forward: list[float] = field(default_factory=list)


class DiffusionGemmaGenerator:
    """Batched sampler for one loaded DiffusionGemma checkpoint."""

    def __init__(
        self,
        model: DiffusionGemmaForBlockDiffusion,
        config: DiffusionSamplingConfig | None = None,
    ) -> None:
        self.model = model
        self.config = config or DiffusionSamplingConfig(
            eos_token_ids=model.config.eos_token_ids,
            pad_token_id=model.config.pad_token_id,
        )
        self.config.validate(model.canvas_length)

    # -- public API -----------------------------------------------------

    @tensorplay.inference_mode()
    def generate(
        self,
        prompts: list[list[int]],
        *,
        config: DiffusionSamplingConfig | None = None,
    ) -> DiffusionGenerationResult:
        if not prompts:
            raise DiffusionGemmaError("at least one prompt is required")
        sampling = config or self.config
        sampling.validate(self.model.canvas_length)
        if sampling.seed is not None:
            tensorplay.manual_seed(sampling.seed)
        canvas_length = self.model.canvas_length
        batch = len(prompts)
        eos = set(sampling.eos_token_ids)
        max_canvases = math.ceil(sampling.max_new_tokens / canvas_length)
        pad = sampling.pad_token_id
        started = time.perf_counter()

        caches: list[DiffusionKVCache] = []
        sequences: list[list[int]] = []
        for prompt in prompts:
            if not prompt:
                raise DiffusionGemmaError("prompts must be non-empty")
            capacity = (
                len(prompt)
                + max_canvases * canvas_length
                + canvas_length
                + 1
            )
            cache = self.model.allocate_cache(capacity)
            tokens = tensorplay.tensor(
                prompt,
                dtype=tensorplay.int64,
                device=self.model.device,
            )
            self.model.encode(tokens, cache)
            caches.append(cache)
            sequences.append(list(prompt))

        self.model.configure_moe_workspace(batch * canvas_length)

        sampler = EntropyBoundSampler(
            sampling, vocab_size=self.model.config.vocab_size
        )
        stopper = StableAndConfidentStopping(
            stability_threshold=sampling.stability_threshold,
            confidence_threshold=sampling.confidence_threshold,
        )
        vocab_size = self.model.config.vocab_size
        device = self.model.device
        sample_cuda = device.type == "cuda" and self.model.kernel_policy.use_triton
        if sample_cuda:
            from megalm.kernels.diffusion_sample import sample_diffusion_logits

            softcap = self.model.config.text.final_logit_softcapping or 0.0
        denoising_steps = [0] * batch
        finished = [False] * batch
        # The committed canvas is encoded at the start of the *next* block,
        # matching the reference loop: the last canvas never pays an encoder
        # pass it cannot use.
        pending: list[list[int] | None] = [None] * batch

        for _canvas in range(max_canvases):
            for row in range(batch):
                if finished[row] or pending[row] is None:
                    continue
                committed = tensorplay.tensor(
                    pending[row],
                    dtype=tensorplay.int64,
                    device=device,
                )
                self.model.encode(committed, caches[row])
                pending[row] = None
            if all(finished):
                break
            canvas = sampler.initialize_canvas(batch, canvas_length, device=device)
            argmax_canvas = canvas
            self_conditioning: tensorplay.Tensor | None = None
            stopped = tensorplay.zeros((batch,), dtype=tensorplay.bool, device=device)
            stopped_rows = [False] * batch
            stopper.reset()

            for step in reversed(range(1, sampling.max_denoising_steps + 1)):
                temperature = temperature_for_step(
                    step,
                    sampling.max_denoising_steps,
                    sampling.t_min,
                    sampling.t_max,
                )
                if sample_cuda:
                    logits = self.model.denoise(
                        canvas, caches, self_conditioning_probs=self_conditioning,
                        return_raw_logits=True,
                    )
                    uniforms = tensorplay.rand(
                        (batch, canvas_length), device=device, dtype=tensorplay.float32
                    )
                    denoiser, new_argmax, entropy, processed = sample_diffusion_logits(
                        logits, uniforms, temperature, self.model.dtype, softcap=softcap,
                    )
                else:
                    logits = self.model.denoise(
                        canvas, caches, self_conditioning_logits=self_conditioning,
                    )
                    processed = logits / temperature
                    probabilities = tensorplay.softmax(processed, dim=-1)
                    denoiser = tensorplay.multinomial(
                        probabilities.view(-1, vocab_size), 1
                    ).view(batch, canvas_length)
                    del probabilities
                    new_argmax = tensorplay.argmax(processed, dim=-1)
                    entropy = token_entropy(processed)
                del logits
                _, next_canvas = sampler.accept_and_renoise(
                    canvas, denoiser, processed, entropy=entropy
                )

                stop_now = stopper(new_argmax, processed, entropy=entropy)
                if any(stopped_rows):
                    keep = stopped.unsqueeze(-1)
                    next_canvas = tensorplay.where(keep, canvas, next_canvas)
                    new_argmax = tensorplay.where(keep, argmax_canvas, new_argmax)
                    if self_conditioning is not None:
                        processed = tensorplay.where(
                            keep.unsqueeze(-1), self_conditioning, processed
                        )
                stopped = stopped | stop_now
                canvas = next_canvas
                argmax_canvas = new_argmax
                self_conditioning = processed.to(self.model.dtype)

                for row, is_done in enumerate(stopped_rows):
                    if not finished[row] and not is_done:
                        denoising_steps[row] += 1
                stopped_rows = stopped.tolist()
                if all(stopped_rows):
                    break

            generated = argmax_canvas.tolist()
            for row in range(batch):
                if finished[row]:
                    continue
                remaining = sampling.max_new_tokens - (len(sequences[row]) - len(prompts[row]))
                tokens = generated[row][:remaining]
                eos_at = next(
                    (index for index, token in enumerate(tokens) if token in eos),
                    None,
                )
                if eos_at is not None:
                    sequences[row].extend(tokens[: eos_at + 1])
                    finished[row] = True
                    continue
                sequences[row].extend(tokens)
                pending[row] = tokens

        elapsed = time.perf_counter() - started
        new_tokens = [len(sequence) - len(prompt) for sequence, prompt in zip(sequences, prompts)]
        total_new = sum(new_tokens)
        return DiffusionGenerationResult(
            sequences=sequences,
            new_tokens=new_tokens,
            denoising_steps=denoising_steps,
            decoder_forward_passes=denoising_steps,
            elapsed_seconds=elapsed,
            tokens_per_second=total_new / elapsed if elapsed > 0 else 0.0,
            tokens_per_forward=[
                tokens / passes if passes else 0.0
                for tokens, passes in zip(new_tokens, denoising_steps)
            ],
        )


__all__ = ["DiffusionGemmaGenerator", "DiffusionGenerationResult"]
