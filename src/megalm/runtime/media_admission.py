from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import tensorplay

from megalm.contracts.request import (
    AudioInput,
    AudioOutputOptions,
    ImageInput,
    OutputModality,
    SamplingParams,
    VideoInput,
)
from megalm.models.qwen2_5_omni import build_omni_mrope_positions
from megalm.models.qwen3_5 import build_mrope_positions
from megalm.runtime.audio_pipeline import ResolvedAudios
from megalm.runtime.tokenizer import ModelTokenizer
from megalm.runtime.vision_pipeline import (
    ResolvedImages,
    ResolvedVideos,
)


class MediaAdmissionError(RuntimeError):
    """Raised when resolved media cannot be bound to one prompt."""


@dataclass(frozen=True, slots=True)
class MultimodalAdmission:
    messages: tuple[Mapping[str, Any], ...]
    images: tuple[ImageInput, ...]
    videos: tuple[VideoInput, ...]
    audios: tuple[AudioInput, ...]
    max_new_tokens: int
    sampling: SamplingParams
    request_id: str
    priority: int = 0
    deadline_ns: int | None = None
    open_thinking: bool = False
    output_modalities: tuple[OutputModality, ...] = (OutputModality.TEXT,)
    audio_output: AudioOutputOptions | None = None
    tools: tuple[dict[str, Any], ...] = ()
    chat_template_kwargs: Mapping[str, Any] | None = None


@dataclass(frozen=True, slots=True)
class MediaState:
    token_type_ids: tuple[int, ...]
    mrope_positions: tensorplay.Tensor
    rope_delta: int
    media_positions: tuple[int, ...]
    media_sources: tuple[tuple[int, int], ...]
    media_embeddings: tuple[tensorplay.Tensor, ...]
    content_digests: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class PreparedMediaAdmission:
    input_ids: tuple[int, ...]
    state: MediaState
    prefix_fingerprint: str


def build_media_admission(
    admission: MultimodalAdmission,
    *,
    tokenizer: ModelTokenizer,
    resolved_images: ResolvedImages | None,
    resolved_videos: ResolvedVideos | None,
    resolved_audios: ResolvedAudios | None,
    model_type: str,
    spatial_merge_size: int,
    position_ids_per_second: int | None,
    device: tensorplay.device,
) -> PreparedMediaAdmission:
    """Bind cached tower outputs to prompt rows without a device sync."""

    encoding = tokenizer.encode_multimodal_chat(
        admission.messages,
        image_feature_counts=(
            ()
            if resolved_images is None
            else resolved_images.feature_counts
        ),
        video_frame_feature_counts=(
            ()
            if resolved_videos is None
            else resolved_videos.frame_feature_counts
        ),
        video_timestamps=(
            ()
            if resolved_videos is None
            else resolved_videos.timestamps
        ),
        audio_feature_counts=(
            ()
            if resolved_audios is None
            else resolved_audios.feature_counts
        ),
        open_thinking=admission.open_thinking,
        tools=admission.tools or None,
        extras=admission.chat_template_kwargs,
        video_timestamp_tokens=(
            model_type not in {"qwen2_5_omni", "gemma4", "gemma4_unified"}
        ),
    )
    image_grid_thw = _grid_tensor(resolved_images)
    video_grid_thw = _grid_tensor(resolved_videos)
    token_types = tensorplay.tensor(
        encoding.token_type_ids,
        dtype=tensorplay.long,
    )
    if model_type in {"gemma4", "gemma4_unified"}:
        # Gemma-4 uses ordinary 1-D RoPE over the expanded token sequence.
        # Soft-token expansion already accounts for every media row, so the
        # generated-token delta is zero.
        positions = tensorplay.arange(len(encoding.input_ids)).unsqueeze(0).expand(3, -1)
        rope_delta = 0
    elif model_type == "qwen2_5_omni":
        if position_ids_per_second is None:
            raise MediaAdmissionError(
                "Omni admission requires a temporal position scale"
            )
        positions, rope_delta = build_omni_mrope_positions(
            token_types,
            image_grid_thw,
            video_grid_thw=video_grid_thw,
            video_seconds_per_grid=(
                ()
                if resolved_videos is None
                else resolved_videos.seconds_per_grid
            ),
            audio_feature_counts=(
                ()
                if resolved_audios is None
                else resolved_audios.feature_counts
            ),
            spatial_merge_size=spatial_merge_size,
            position_ids_per_second=position_ids_per_second,
        )
    else:
        positions, rope_delta = build_mrope_positions(
            token_types,
            image_grid_thw,
            video_grid_thw=video_grid_thw,
            spatial_merge_size=spatial_merge_size,
        )

    media_embeddings = (
        (() if resolved_images is None else resolved_images.embeddings)
        + (() if resolved_videos is None else resolved_videos.embeddings)
        + (() if resolved_audios is None else resolved_audios.embeddings)
    )
    image_sources = iter(
        (image_index, feature_index)
        for image_index, feature_count in enumerate(
            ()
            if resolved_images is None
            else resolved_images.feature_counts
        )
        for feature_index in range(feature_count)
    )
    video_offset = len(admission.images)
    video_sources = iter(
        (video_offset + video_index, feature_index)
        for video_index, feature_count in enumerate(
            ()
            if resolved_videos is None
            else resolved_videos.feature_counts
        )
        for feature_index in range(feature_count)
    )
    audio_offset = len(admission.images) + len(admission.videos)
    audio_sources = iter(
        (audio_offset + audio_index, feature_index)
        for audio_index, feature_count in enumerate(
            ()
            if resolved_audios is None
            else resolved_audios.feature_counts
        )
        for feature_index in range(feature_count)
    )
    media_positions: list[int] = []
    media_sources: list[tuple[int, int]] = []
    for position, token_type in enumerate(encoding.token_type_ids):
        if token_type == 1:
            source = next(image_sources, None)
        elif token_type == 2:
            source = next(video_sources, None)
        elif token_type == 3:
            source = next(audio_sources, None)
        else:
            continue
        if source is None:
            raise MediaAdmissionError(
                "media placeholders exceed encoded features"
            )
        media_positions.append(position)
        media_sources.append(source)
    if (
        next(image_sources, None) is not None
        or next(video_sources, None) is not None
        or next(audio_sources, None) is not None
    ):
        raise MediaAdmissionError(
            "encoded media features exceed prompt placeholders"
        )

    content_digests = (
        *(
            f"image:{image.content_digest}:{image.detail}"
            for image in admission.images
        ),
        *(
            f"video:{video.content_digest}:{video.sampling.digest}"
            for video in admission.videos
        ),
        *(
            f"audio:{audio.content_digest}"
            for audio in admission.audios
        ),
    )
    state = MediaState(
        token_type_ids=encoding.token_type_ids,
        mrope_positions=positions.to(device=device, non_blocking=True),
        rope_delta=rope_delta,
        media_positions=tuple(media_positions),
        media_sources=tuple(media_sources),
        media_embeddings=media_embeddings,
        content_digests=content_digests,
    )
    return PreparedMediaAdmission(
        input_ids=encoding.input_ids,
        state=state,
        prefix_fingerprint=media_prefix_fingerprint(content_digests),
    )


def media_prefix_fingerprint(content_digests: Sequence[str]) -> str:
    """Build a framed, domain-separated namespace for multimodal prompt KV."""

    digest = hashlib.sha256()
    digest.update(b"megalm.multimodal-prefix.v1\0")
    for content_digest in content_digests:
        encoded = content_digest.encode("ascii")
        digest.update(len(encoded).to_bytes(4, "little"))
        digest.update(encoded)
    return digest.hexdigest()


def _grid_tensor(
    resolved: ResolvedImages | ResolvedVideos | None,
) -> tensorplay.Tensor:
    return tensorplay.tensor(
        () if resolved is None else resolved.content_grids,
        dtype=tensorplay.long,
    ).reshape(-1, 3)


__all__ = [
    "MediaAdmissionError",
    "MediaState",
    "MultimodalAdmission",
    "PreparedMediaAdmission",
    "build_media_admission",
    "media_prefix_fingerprint",
]
