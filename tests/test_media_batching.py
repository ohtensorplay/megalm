from __future__ import annotations

import asyncio
from contextlib import nullcontext
import hashlib
from types import SimpleNamespace

import pytest
import tensorplay

from megalm.contracts.request import (
    AudioInput,
    ImageInput,
    SamplingParams,
    VideoInput,
    VideoSampling,
)
from megalm.gateway.omni_admission import AsyncOmniAdmissionBatcher
from megalm.runtime.audio_pipeline import AudioPipeline
from megalm.runtime.engine import Engine, EngineError, EngineMetrics
from megalm.runtime.media_admission import (
    MultimodalAdmission,
    build_media_admission,
)
from megalm.runtime.tokenizer import MultimodalEncoding
from megalm.runtime.vision_cache import (
    GPUVisionCache,
    VisionCacheValue,
)
from megalm.runtime.vision_pipeline import (
    ResolvedImages,
    ResolvedVideos,
    VisionPipeline,
)


def _digest(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def test_engine_media_stream_defers_commit_until_cuda_event(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class Event:
        def __init__(self) -> None:
            self.ready = False
            self.recorded_stream: object | None = None
            self.synchronized = False

        def record(self, stream: object) -> None:
            self.recorded_stream = stream

        def query(self) -> bool:
            return self.ready

        def synchronize(self) -> None:
            assert self.ready
            self.synchronized = True

    event = Event()
    monkeypatch.setattr(tensorplay.cuda, "Event", lambda: event)
    monkeypatch.setattr(tensorplay.cuda, "stream", lambda _: nullcontext())

    engine = Engine.__new__(Engine)
    engine.media_stream = object()
    engine.metrics = EngineMetrics()
    engine._media_requests = {}
    engine.vision_cache = SimpleNamespace(
        release_request=lambda _: None
    )
    resolved_calls: list[tuple[str, ...]] = []
    committed: list[str] = []
    engine._validate_multimodal = lambda _: None
    engine._resolve_multimodal_batch = lambda admissions: (
        resolved_calls.append(
            tuple(admission.request_id for admission in admissions)
        )
        or {
            admission.request_id: (None, None, None)
            for admission in admissions
        }
    )
    engine._commit_multimodal = (
        lambda admission, *_: (
            committed.append(admission.request_id)
            or admission.request_id
        )
    )
    admission = MultimodalAdmission(
        messages=({"role": "user", "content": "image"},),
        images=(),
        videos=(),
        audios=(),
        max_new_tokens=1,
        sampling=SamplingParams(),
        request_id="media-stream",
    )

    pending = engine.begin_multimodal_batch((admission,))

    assert resolved_calls == [("media-stream",)]
    assert committed == []
    assert event.recorded_stream is engine.media_stream
    assert not engine.multimodal_batch_ready(pending)

    event.ready = True
    assert engine.multimodal_batch_ready(pending)
    assert engine.finish_multimodal_batch(pending) == ("media-stream",)
    assert committed == ["media-stream"]
    assert event.synchronized
    assert engine.metrics.media_stream_batches == 1
    assert engine.metrics.media_admission_batches == 1
    with pytest.raises(EngineError, match="already been consumed"):
        engine.finish_multimodal_batch(pending)


def test_vision_batch_deduplicates_tower_work_and_preserves_pins() -> None:
    first = ImageInput.from_bytes(b"first", "image/png")
    second = ImageInput.from_bytes(b"second", "image/png")
    cache = GPUVisionCache(
        capacity_bytes=4096,
        width=4,
        dtype=tensorplay.float32,
        device="cpu",
    )
    prepared_batches: list[tuple[ImageInput, ...]] = []
    tower_batches: list[int] = []

    class Processor:
        digest = _digest("processor")

        def prepare(
            self,
            images: tuple[ImageInput, ...],
        ) -> SimpleNamespace:
                prepared_batches.append(images)
                count = len(images)
                return SimpleNamespace(
                    pixel_values=tensorplay.arange(
                        count * 4 * 3,
                        dtype=tensorplay.float32,
                    ).view(count * 4, 3),
                grid_thw=tensorplay.tensor(
                    [(1, 2, 2)] * count,
                    dtype=tensorplay.long,
                ),
                feature_counts=(1,) * count,
            )

    def encode(
        pixels: tensorplay.Tensor,
        grids: tensorplay.Tensor,
    ) -> tensorplay.Tensor:
        tower_batches.append(grids.shape[0])
        assert grids.device.type == "cpu"
        return tensorplay.arange(
            grids.shape[0] * 4,
            dtype=tensorplay.float32,
        ).view(grids.shape[0], 4)

    pipeline = VisionPipeline(
        processor=Processor(),
        video_processor=None,
        cache=cache,
        model_digest=_digest("model"),
        encode_images=encode,
        device="cpu",
    )
    resolved = pipeline.resolve_batch(
        (
            ("request-a", (first, first)),
            ("request-b", (second,)),
        )
    )

    assert prepared_batches == [(first, second)]
    assert tower_batches == [2]
    assert (
        resolved[0].embeddings[0].data_ptr()
        == resolved[0].embeddings[1].data_ptr()
    )
    stats = cache.stats()
    assert stats.misses == 2
    assert stats.hits == 0
    assert stats.requests == 2
    pipeline.release("request-a")
    pipeline.release("request-b")

    warm = pipeline.resolve_batch(
        (
            ("request-c", (first,)),
            ("request-d", (second,)),
        )
    )
    assert len(warm) == 2
    assert tower_batches == [2]
    assert cache.stats().hits == 2
    pipeline.release("request-c")
    pipeline.release("request-d")
    cache.assert_integrity()


def test_audio_batch_uses_one_encoder_launch() -> None:
    first = AudioInput.from_bytes(b"first", "audio/wav")
    second = AudioInput.from_bytes(b"second", "audio/wav")
    cache = GPUVisionCache(
        capacity_bytes=4096,
        width=4,
        dtype=tensorplay.float32,
        device="cpu",
    )
    calls: list[tuple[int, ...]] = []

    class Processor:
        digest = _digest("audio-processor")

        def prepare(
            self,
            audios: tuple[AudioInput, ...],
        ) -> SimpleNamespace:
            assert audios == (first, second)
            return SimpleNamespace(
                input_features=tensorplay.ones((2, 12)),
                feature_lengths=(5, 7),
                output_lengths=(1, 2),
            )

    def encode(
        features: tensorplay.Tensor,
        lengths: tuple[int, ...],
    ) -> tuple[tensorplay.Tensor, tuple[int, ...]]:
        calls.append(lengths)
        return tensorplay.arange(12, dtype=tensorplay.float32).view(3, 4), (1, 2)

    pipeline = AudioPipeline(
        processor=Processor(),
        cache=cache,
        model_digest=_digest("model"),
        encode_audios=encode,
    )
    resolved = pipeline.resolve_batch(
        (
            ("request-a", (first,)),
            ("request-b", (second,)),
        )
    )

    assert calls == [(5, 7)]
    assert tuple(value.feature_counts for value in resolved) == (
        (1,),
        (2,),
    )
    pipeline.release("request-a")
    pipeline.release("request-b")
    cache.assert_integrity()


def test_video_batch_chunks_processor_work_but_merges_tower_launch() -> None:
    first = VideoInput.from_bytes(b"first", "video/mp4")
    second = VideoInput.from_bytes(b"second", "video/mp4")
    cache = GPUVisionCache(
        capacity_bytes=4096,
        width=4,
        dtype=tensorplay.float32,
        device="cpu",
    )
    prepared_batches: list[tuple[VideoInput, ...]] = []
    tower_batches: list[int] = []

    class ImageProcessor:
        digest = _digest("image-processor")

        def prepare(self, _: object) -> None:
            raise AssertionError("image processor must not run")

    class VideoProcessor:
        max_videos = 1

        def cache_digest(self, _: VideoInput) -> str:
            return _digest("video-processor")

        def prepare(
            self,
            videos: tuple[VideoInput, ...],
        ) -> SimpleNamespace:
                prepared_batches.append(videos)
                return SimpleNamespace(
                    pixel_values=tensorplay.ones((len(videos) * 4, 3)),
                grid_thw=tensorplay.tensor(
                    [(1, 2, 2)] * len(videos),
                    dtype=tensorplay.long,
                ),
                feature_counts=(1,) * len(videos),
                frame_feature_counts=((1,),) * len(videos),
                timestamps=((0.0,),) * len(videos),
                seconds_per_grid=((0.5,),) * len(videos),
            )

    def encode(
        _: tensorplay.Tensor,
        grids: tensorplay.Tensor,
    ) -> tensorplay.Tensor:
        tower_batches.append(grids.shape[0])
        return tensorplay.ones((grids.shape[0], 4))

    pipeline = VisionPipeline(
        processor=ImageProcessor(),
        video_processor=VideoProcessor(),
        cache=cache,
        model_digest=_digest("model"),
        encode_images=encode,
        device="cpu",
    )
    resolved = pipeline.resolve_videos_batch(
        (
            ("request-a", (first,)),
            ("request-b", (second,)),
        )
    )

    assert prepared_batches == [(first,), (second,)]
    assert tower_batches == [2]
    assert tuple(value.feature_counts for value in resolved) == (
        (1,),
        (1,),
    )
    pipeline.release("request-a")
    pipeline.release("request-b")
    cache.assert_integrity()


def test_mixed_visual_batch_uses_one_bounded_tower_launch() -> None:
    image = ImageInput.from_bytes(b"image", "image/png")
    video = VideoInput.from_bytes(b"video", "video/mp4")
    cache = GPUVisionCache(
        capacity_bytes=4096,
        width=4,
        dtype=tensorplay.float32,
        device="cpu",
    )
    tower_batches: list[tuple[str, ...]] = []

    class ImageProcessor:
        digest = _digest("image-processor")
        max_images = 1

        def prepare(
            self,
            images: tuple[ImageInput, ...],
        ) -> SimpleNamespace:
            assert images == (image,)
            return SimpleNamespace(
                pixel_values=tensorplay.ones((4, 3)),
                grid_thw=tensorplay.tensor([(1, 2, 2)]),
                feature_counts=(1,),
            )

    class VideoProcessor:
        max_videos = 1

        def cache_digest(self, _: VideoInput) -> str:
            return _digest("video-processor")

        def prepare(
            self,
            videos: tuple[VideoInput, ...],
        ) -> SimpleNamespace:
            assert videos == (video,)
            return SimpleNamespace(
                pixel_values=tensorplay.full((4, 3), 2.0),
                grid_thw=tensorplay.tensor([(1, 2, 2)]),
                feature_counts=(1,),
                frame_feature_counts=((1,),),
                timestamps=((0.0,),),
                seconds_per_grid=((0.5,),),
            )

    def encode(
        pixels: tensorplay.Tensor,
        grids: tensorplay.Tensor,
    ) -> tensorplay.Tensor:
        tower_batches.append(
            tuple(str(value.item()) for value in pixels[:, 0])
        )
        assert grids.shape == (2, 3)
        return tensorplay.ones((2, 4))

    pipeline = VisionPipeline(
        processor=ImageProcessor(),
        video_processor=VideoProcessor(),
        cache=cache,
        model_digest=_digest("model"),
        encode_images=encode,
        device="cpu",
        max_encode_patches=8,
    )
    images, videos = pipeline.resolve_media_batch(
        image_requests=(("request-image", (image,)),),
        video_requests=(("request-video", (video,)),),
    )

    assert tower_batches == [("1.0",) * 4 + ("2.0",) * 4]
    assert images[0].feature_counts == (1,)
    assert videos[0].feature_counts == (1,)
    assert videos[0].frame_feature_counts == ((1,),)
    pipeline.release("request-image")
    pipeline.release("request-video")
    cache.assert_integrity()


def test_mixed_media_binding_preserves_prompt_modality_order() -> None:
    image = ImageInput.from_bytes(b"image", "image/png")
    video = VideoInput.from_bytes(
        b"video",
        "video/mp4",
        sampling=VideoSampling(target_fps=None, num_frames=2),
    )
    audio = AudioInput.from_bytes(b"audio", "audio/wav")
    image_value = VisionCacheValue(
        embedding=tensorplay.ones((1, 4)),
        grid_thw=(1, 2, 2),
        cached=False,
    )
    video_value = VisionCacheValue(
        embedding=tensorplay.full((1, 4), 2.0),
        grid_thw=(1, 2, 2),
        cached=False,
        frame_feature_counts=(1,),
        timestamps=(0.0,),
        seconds_per_grid=(0.5,),
    )
    audio_value = VisionCacheValue(
        embedding=tensorplay.full((2, 4), 3.0),
        grid_thw=(2, 1, 1),
        cached=False,
    )

    class Tokenizer:
        def encode_multimodal_chat(
            self,
            *_: object,
            **__: object,
        ) -> MultimodalEncoding:
                return MultimodalEncoding(
                    input_ids=(10, 11, 12, 13, 13, 14, 15),
                    token_type_ids=(0, 1, 0, 3, 3, 0, 2),
                    image_count=1,
                    video_count=1,
                    audio_count=1,
                )

    admission = MultimodalAdmission(
        messages=({"role": "user", "content": "mixed"},),
        images=(image,),
        videos=(video,),
        audios=(audio,),
        max_new_tokens=4,
        sampling=SamplingParams(),
        request_id="mixed",
    )
    prepared = build_media_admission(
        admission,
        tokenizer=Tokenizer(),
        resolved_images=ResolvedImages((image_value,)),
        resolved_videos=ResolvedVideos((video_value,)),
        resolved_audios=SimpleNamespace(
            feature_counts=(2,),
            embeddings=(audio_value.embedding,),
        ),
        model_type="qwen2_5_omni",
        spatial_merge_size=2,
        position_ids_per_second=25,
        device=tensorplay.device("cpu"),
    )

    assert prepared.state.media_positions == (1, 3, 4, 6)
    assert prepared.state.media_sources == (
        (0, 0),
        (2, 0),
        (2, 1),
        (1, 0),
    )
    assert prepared.state.mrope_positions.shape == (3, 7)


def test_omni_admission_coalesces_one_event_loop_wave() -> None:
    class Replica:
        def __init__(self) -> None:
            self.batch_sizes: list[int] = []

        async def submit_multimodal_batch(
            self,
            submissions: tuple[object, ...],
        ) -> tuple[SimpleNamespace, ...]:
            self.batch_sizes.append(len(submissions))
            return tuple(
                SimpleNamespace(request_id=item.request_id)
                for item in submissions
            )

        def cancel(self, _: str) -> None:
            raise AssertionError("no request should be cancelled")

    async def run() -> None:
        replica = Replica()
        batcher = AsyncOmniAdmissionBatcher(
            replica,
            max_batch_size=8,
        )
        batcher.start()
        image = ImageInput.from_bytes(b"image", "image/png")
        outputs = await asyncio.gather(
            *(
                batcher.submit(
                    ({"role": "user", "content": "image"},),
                    (image,),
                    max_new_tokens=1,
                    sampling=SamplingParams(),
                    request_id=f"request-{index}",
                )
                for index in range(4)
            )
        )
        await batcher.stop()
        assert replica.batch_sizes == [2, 2]
        assert [item.request_id for item in outputs] == [
            f"request-{index}" for index in range(4)
        ]
        assert batcher.metrics.snapshot()["batch_histogram"] == {2: 2}

    asyncio.run(run())


def test_omni_admission_dispatches_light_media_before_video() -> None:
    class Replica:
        def __init__(self) -> None:
            self.batch_kinds: list[tuple[bool, ...]] = []

        async def submit_multimodal_batch(
            self,
            submissions: tuple[object, ...],
        ) -> tuple[SimpleNamespace, ...]:
            self.batch_kinds.append(
                tuple(bool(item.videos) for item in submissions)
            )
            return tuple(
                SimpleNamespace(request_id=item.request_id)
                for item in submissions
            )

        def cancel(self, _: str) -> None:
            raise AssertionError("no request should be cancelled")

    async def run() -> None:
        replica = Replica()
        batcher = AsyncOmniAdmissionBatcher(
            replica,
            max_batch_size=8,
        )
        batcher.start()
        image = ImageInput.from_bytes(b"image", "image/png")
        video = VideoInput.from_bytes(b"video", "video/mp4")
        await asyncio.gather(
            batcher.submit(
                ({"role": "user", "content": "video"},),
                (),
                videos=(video,),
                max_new_tokens=1,
                sampling=SamplingParams(),
                request_id="video",
            ),
            batcher.submit(
                ({"role": "user", "content": "image"},),
                (image,),
                max_new_tokens=1,
                sampling=SamplingParams(),
                request_id="image",
            ),
        )
        await batcher.stop()

        assert replica.batch_kinds == [(False,), (True,)]

    asyncio.run(run())


def test_omni_admission_coalesces_staggered_siblings() -> None:
    class Replica:
        def __init__(self) -> None:
            self.batch_sizes: list[int] = []

        async def submit_multimodal_batch(
            self,
            submissions: tuple[object, ...],
        ) -> tuple[SimpleNamespace, ...]:
            self.batch_sizes.append(len(submissions))
            return tuple(
                SimpleNamespace(request_id=item.request_id)
                for item in submissions
            )

        def cancel(self, _: str) -> None:
            raise AssertionError("no request should be cancelled")

    async def run() -> None:
        replica = Replica()
        batcher = AsyncOmniAdmissionBatcher(
            replica,
            max_batch_size=8,
            coalesce_wait_us=20_000,
        )
        batcher.start()
        image = ImageInput.from_bytes(b"image", "image/png")
        first = asyncio.create_task(
            batcher.submit(
                ({"role": "user", "content": "image"},),
                (image,),
                max_new_tokens=1,
                sampling=SamplingParams(),
                request_id="first",
            )
        )
        await asyncio.sleep(0.001)
        second = asyncio.create_task(
            batcher.submit(
                ({"role": "user", "content": "image"},),
                (image,),
                max_new_tokens=1,
                sampling=SamplingParams(),
                request_id="second",
            )
        )
        outputs = await asyncio.gather(first, second)
        await batcher.stop()

        assert replica.batch_sizes == [2]
        assert [item.request_id for item in outputs] == [
            "first",
            "second",
        ]
        snapshot = batcher.metrics.snapshot()
        assert snapshot["coalesce_waits"] == 1
        assert snapshot["coalesce_wait_ms"] > 0

    asyncio.run(run())
