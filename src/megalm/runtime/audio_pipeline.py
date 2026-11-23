from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Protocol, Sequence
from uuid import uuid4

import tensorplay

from megalm.contracts.request import AudioInput
from megalm.runtime.vision_cache import (
    GPUVisionCache,
    VisionCacheKey,
    VisionCacheValue,
)


class AudioPipelineError(RuntimeError):
    """Raised when cached and encoded audio features disagree."""


class _PreparedAudios(Protocol):
    input_features: tensorplay.Tensor
    feature_lengths: tuple[int, ...]
    output_lengths: tuple[int, ...]


class AudioProcessor(Protocol):
    digest: str

    def prepare(
        self,
        audios: Sequence[AudioInput],
    ) -> _PreparedAudios: ...


EncodeAudios = Callable[
    [tensorplay.Tensor, Sequence[int]],
    tuple[tensorplay.Tensor, tuple[int, ...]],
]


@dataclass(frozen=True, slots=True)
class ResolvedAudios:
    values: tuple[VisionCacheValue, ...]

    @property
    def feature_counts(self) -> tuple[int, ...]:
        return tuple(value.feature_count for value in self.values)

    @property
    def embeddings(self) -> tuple[tensorplay.Tensor, ...]:
        return tuple(value.embedding for value in self.values)


class AudioPipeline:
    """Resolve audio tower outputs through the Replica-global media cache."""

    def __init__(
        self,
        *,
        processor: AudioProcessor,
        cache: GPUVisionCache,
        model_digest: str,
        encode_audios: EncodeAudios,
    ) -> None:
        self.processor = processor
        self.cache = cache
        self.model_digest = model_digest
        self.encode_audios = encode_audios

    @tensorplay.inference_mode()
    def resolve(
        self,
        request_id: str,
        audios: Sequence[AudioInput],
    ) -> ResolvedAudios:
        ordered = tuple(audios)
        if not ordered:
            raise AudioPipelineError("multimodal request has no audio")
        keys = tuple(self._key(audio) for audio in ordered)
        resolved: dict[VisionCacheKey, VisionCacheValue] = {}
        missing: dict[VisionCacheKey, AudioInput] = {}
        try:
            for key, audio in zip(keys, ordered):
                if key in resolved or key in missing:
                    continue
                value = self.cache.acquire(key, request_id)
                if value is None:
                    missing[key] = audio
                else:
                    resolved[key] = value
            if missing:
                self._encode_missing(request_id, missing, resolved)
            occurrences: dict[VisionCacheKey, int] = {}
            values: list[VisionCacheValue] = []
            for key in keys:
                count = occurrences.get(key, 0)
                value = resolved[key]
                if count and value.cached:
                    acquired = self.cache.acquire(key, request_id)
                    if acquired is None:
                        raise AudioPipelineError(
                            "audio cache entry disappeared while pinned"
                        )
                    value = acquired
                occurrences[key] = count + 1
                values.append(value)
            return ResolvedAudios(tuple(values))
        except BaseException:
            self.cache.release_request(request_id)
            raise

    @tensorplay.inference_mode()
    def resolve_batch(
        self,
        requests: Sequence[tuple[str, Sequence[AudioInput]]],
    ) -> tuple[ResolvedAudios, ...]:
        """Encode unique cold audio items from one admission wave together."""

        ordered = tuple(
            (request_id, tuple(audios))
            for request_id, audios in requests
        )
        if not ordered:
            return ()
        request_ids = tuple(request_id for request_id, _ in ordered)
        if (
            any(not request_id for request_id in request_ids)
            or len(set(request_ids)) != len(request_ids)
        ):
            raise AudioPipelineError(
                "audio batch request ids must be unique and non-empty"
            )
        if any(not audios for _, audios in ordered):
            raise AudioPipelineError(
                "audio batch request has no audio"
            )
        keys_by_request = tuple(
            tuple(self._key(audio) for audio in audios)
            for _, audios in ordered
        )
        unique: dict[VisionCacheKey, AudioInput] = {}
        for (_, audios), keys in zip(ordered, keys_by_request, strict=True):
            for key, audio in zip(keys, audios, strict=True):
                unique.setdefault(key, audio)
        staging_id = f"__megalm_audio_batch_{uuid4().hex}"
        try:
            staged = self.resolve(staging_id, tuple(unique.values()))
            by_key = dict(zip(unique, staged.values, strict=True))
            outputs = tuple(
                ResolvedAudios(
                    tuple(
                        self._retain_resolved(key, by_key[key], request_id)
                        for key in keys
                    )
                )
                for request_id, keys in zip(
                    request_ids,
                    keys_by_request,
                    strict=True,
                )
            )
        except BaseException:
            for request_id in request_ids:
                self.cache.release_request(request_id)
            raise
        finally:
            self.cache.release_request(staging_id)
        return outputs

    def release(self, request_id: str) -> None:
        self.cache.release_request(request_id)

    def _retain_resolved(
        self,
        key: VisionCacheKey,
        value: VisionCacheValue,
        request_id: str,
    ) -> VisionCacheValue:
        if not value.cached:
            return value
        retained = self.cache.retain(key, request_id)
        if retained is None:
            raise AudioPipelineError(
                "audio cache entry disappeared while batch-pinned"
            )
        return retained

    def _encode_missing(
        self,
        request_id: str,
        missing: dict[VisionCacheKey, AudioInput],
        resolved: dict[VisionCacheKey, VisionCacheValue],
    ) -> None:
        keys = tuple(missing)
        prepared = self.processor.prepare(
            tuple(missing[key] for key in keys)
        )
        encoded, counts = self.encode_audios(
            prepared.input_features,
            prepared.feature_lengths,
        )
        if (
            counts != prepared.output_lengths
            or encoded.ndim != 2
            or encoded.shape[0] != sum(counts)
        ):
            raise AudioPipelineError(
                "audio tower output differs from placeholder count"
            )
        offset = 0
        for key, count in zip(keys, counts):
            end = offset + count
            embedding = encoded[offset:end].clone().contiguous()
            resolved[key] = self.cache.insert_and_acquire(
                key,
                embedding,
                (count, 1, 1),
                request_id,
            )
            offset = end

    def _key(self, audio: AudioInput) -> VisionCacheKey:
        return VisionCacheKey(
            model_digest=self.model_digest,
            processor_digest=self.processor.digest,
            content_digest=audio.content_digest,
            modality="audio",
        )


__all__ = [
    "AudioPipeline",
    "AudioPipelineError",
    "ResolvedAudios",
]
