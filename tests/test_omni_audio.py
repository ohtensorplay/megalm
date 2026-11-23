from __future__ import annotations

import hashlib
from pathlib import Path

import tensorplay

from megalm.contracts.request import AudioInput
from megalm.models.qwen2_5_omni.audio_processor import (
    PreparedOmniAudios,
    Qwen25OmniAudioProcessor,
)
from megalm.models.qwen2_5_omni.config import OmniAudioConfig
from megalm.runtime.audio_pipeline import AudioPipeline
from megalm.runtime.vision_cache import GPUVisionCache


def _config() -> OmniAudioConfig:
    return OmniAudioConfig(
        num_mel_bins=128,
        num_layers=2,
        num_heads=4,
        intermediate_size=64,
        hidden_size=32,
        max_source_positions=1500,
        window_size=100,
        output_hidden_size=16,
        layer_norm_eps=1e-5,
        activation_function="gelu",
    )


def _digest(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def test_native_audio_frontend_produces_official_feature_lengths() -> None:
    path = Path(
        "tests/assets/qwen25_omni_suite/audio_00.wav"
    )
    audio = AudioInput.from_bytes(path.read_bytes(), "audio/wav")

    prepared = Qwen25OmniAudioProcessor(_config()).prepare((audio,))

    assert prepared.input_features.shape == (128, 55)
    assert prepared.feature_lengths == (55,)
    assert prepared.output_lengths == (14,)
    assert prepared.durations_seconds == (0.55,)
    assert prepared.input_features.is_contiguous()
    assert bool(tensorplay.isfinite(prepared.input_features).all())


def test_audio_pipeline_reuses_replica_global_gpu_features() -> None:
    audio = AudioInput.from_bytes(b"same audio", "audio/wav")
    cache = GPUVisionCache(
        capacity_bytes=1024,
        width=4,
        dtype=tensorplay.float32,
        device="cpu",
    )

    class Processor:
        digest = _digest("processor")

        def prepare(
            self,
            audios: tuple[AudioInput, ...],
        ) -> PreparedOmniAudios:
            assert audios == (audio,)
            return PreparedOmniAudios(
                input_features=tensorplay.ones((2, 7)),
                feature_lengths=(7,),
                output_lengths=(2,),
                durations_seconds=(1.0,),
            )

    calls = 0

    def encode(
        features: tensorplay.Tensor,
        lengths: tuple[int, ...],
    ) -> tuple[tensorplay.Tensor, tuple[int, ...]]:
        nonlocal calls
        calls += 1
        assert features.shape == (2, 7)
        assert lengths == (7,)
        return tensorplay.arange(8, dtype=tensorplay.float32).view(2, 4), (2,)

    pipeline = AudioPipeline(
        processor=Processor(),
        cache=cache,
        model_digest=_digest("model"),
        encode_audios=encode,
    )
    first = pipeline.resolve("first", (audio,))
    first_pointer = first.embeddings[0].data_ptr()
    pipeline.release("first")
    second = pipeline.resolve("second", (audio,))

    assert calls == 1
    assert second.embeddings[0].data_ptr() == first_pointer
    stats = cache.stats()
    assert stats.misses == 1
    assert stats.hits == 1
    pipeline.release("second")
    cache.assert_integrity()
