"""Exact block-loop parity against the HF tiny DiffusionGemma checkpoint.

The two optimisers under test are the block loop itself (canvas scheduling,
KV-cache growth, self-conditioning hand-off, stopping) rather than sampling
noise: both engines are fed the same synthetic canvas stream and a
deterministic token selector, so identical output ids mean identical loops.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest
import tensorplay

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")

from transformers.models.diffusion_gemma.generation_diffusion_gemma import (  # noqa: E402
    DiffusionGemmaGenerationConfig,
    EntropyBoundSampler as HFEntropyBoundSampler,
    EntropyBoundSamplerConfig,
)
from transformers.models.diffusion_gemma.modeling_diffusion_gemma import (  # noqa: E402
    DiffusionGemmaForBlockDiffusion as HFDiffusionGemma,
)

from megalm.kernels.ops import KernelPolicy  # noqa: E402
from megalm.models.diffusiongemma import (  # noqa: E402
    DiffusionGemmaForBlockDiffusion,
)
from megalm.models.diffusiongemma.generate import DiffusionGemmaGenerator  # noqa: E402
from megalm.models.diffusiongemma.sampler import (  # noqa: E402
    DiffusionSamplingConfig,
    EntropyBoundSampler as NativeEntropyBoundSampler,
)


def _tiny_checkpoint() -> str:
    env = os.environ.get("MEGALM_TINY_DIFFUSIONGEMMA")
    if env:
        return env
    cache = Path.home() / ".cache/huggingface/hub"
    pattern = "models--trl-internal-testing--tiny-DiffusionGemmaForBlockDiffusion"
    snapshots = sorted((cache / pattern / "snapshots").glob("*"))
    if not snapshots:
        pytest.skip("tiny DiffusionGemma checkpoint is not cached")
    return str(snapshots[0])


TINY = None
PROMPT = [100, 200, 300, 400]
CANVAS_LENGTH = 32
STEPS = 3
CANVASES = 2
VOCAB = 262144


class _CanvasStream:
    def __init__(self) -> None:
        self.index = 0

    def _row(self, length: int) -> list[int]:
        base = self.index
        self.index += 1
        return [
            (base * 37 + position * 11 + 5) % (VOCAB - 8) + 4
            for position in range(length)
        ]

    def hf(self, batch_size: int, device: torch.device) -> torch.Tensor:
        row = torch.tensor(self._row(CANVAS_LENGTH), dtype=torch.long, device=device)
        return row.unsqueeze(0).expand(batch_size, -1).clone()

    def native(
        self, batch_size: int, canvas_length: int, *, device: tensorplay.device
    ) -> tensorplay.Tensor:
        row = tensorplay.tensor(
            self._row(canvas_length), dtype=tensorplay.int64, device=device
        )
        return row.unsqueeze(0).expand(batch_size, -1).clone()


def _run_hf(path: str, stream: _CanvasStream) -> list[list[int]]:
    model = HFDiffusionGemma.from_pretrained(path, dtype=torch.float32)
    model.eval()
    config = DiffusionGemmaGenerationConfig(
        max_new_tokens=CANVASES * CANVAS_LENGTH,
        max_denoising_steps=STEPS,
        sampler_config=EntropyBoundSamplerConfig(entropy_bound=1e9),
        t_min=0.4,
        t_max=0.8,
        stability_threshold=1,
        confidence_threshold=1e-12,
        eos_token_id=None,
        pad_token_id=0,
    )
    stream.index = 0
    with torch.no_grad():
        out = model.generate(
            input_ids=torch.tensor([PROMPT], dtype=torch.long),
            generation_config=config,
        )
    return out.sequences.tolist()


def _run_native(path: str, stream: _CanvasStream) -> list[list[int]]:
    model, _ = DiffusionGemmaForBlockDiffusion.from_pretrained(
        path,
        device="cpu",
        dtype=tensorplay.float32,
        kernel_policy=KernelPolicy(use_triton=False),
    )
    generator = DiffusionGemmaGenerator(
        model,
        DiffusionSamplingConfig(
            max_new_tokens=CANVASES * CANVAS_LENGTH,
            max_denoising_steps=STEPS,
            entropy_bound=1e9,
            t_min=0.4,
            t_max=0.8,
            stability_threshold=1,
            confidence_threshold=1e-12,
            eos_token_ids=(),
            pad_token_id=0,
        ),
    )
    stream.index = 0
    return generator.generate([PROMPT]).sequences


def test_generation_matches_hf_tiny(monkeypatch: pytest.MonkeyPatch) -> None:
    path = _tiny_checkpoint()

    def torch_multinomial(probs, num_samples=1, **kwargs):
        del num_samples, kwargs
        return probs.argmax(dim=-1, keepdim=True)

    def native_multinomial(probs, num_samples=1):
        del num_samples
        return tensorplay.argmax(probs, dim=-1).unsqueeze(-1)

    monkeypatch.setattr(torch, "multinomial", torch_multinomial)
    monkeypatch.setattr(tensorplay, "multinomial", native_multinomial)
    stream = _CanvasStream()
    monkeypatch.setattr(HFEntropyBoundSampler, "initialize_canvas", stream.hf)
    monkeypatch.setattr(
        NativeEntropyBoundSampler, "initialize_canvas", stream.native
    )

    hf = _run_hf(path, stream)
    native = _run_native(path, stream)
    assert hf[0][: len(PROMPT)] == native[0][: len(PROMPT)] == PROMPT
    assert hf[0][len(PROMPT) :] == native[0][len(PROMPT) :]
