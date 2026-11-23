from __future__ import annotations

import threading
import time
from types import SimpleNamespace

import tensorplay

from megalm.contracts.request import AudioOutputOptions, FinishReason
from megalm.runtime.batch import BatchPhase, ExecutionBatch
from megalm.runtime.engine import (
    Engine,
    PendingEngineStep,
    _OmniFeatureCapture,
)
from megalm.runtime.omni_output import OmniSpeechStage, OmniSpeechTask
from megalm.runtime.scheduler import ScheduledBatch


class _Backend:
    voices = ("Chelsie",)

    def __init__(self) -> None:
        self.called = threading.Event()

    def text_embedding_ids(self, voice: str) -> tuple[int, int, int]:
        assert voice == "Chelsie"
        return 1, 2, 3

    def synthesize(
        self,
        task: OmniSpeechTask,
        cancelled: threading.Event,
    ) -> bytes:
        assert task.prompt_token_ids == (10, 11)
        assert task.output_token_ids == (12, 13)
        assert not cancelled.is_set()
        self.called.set()
        return b"abcdefghijkl"


def test_omni_speech_stage_preserves_audio_chunk_order() -> None:
    backend = _Backend()
    stage = OmniSpeechStage(
        backend,
        max_pending=1,
        audio_chunk_bytes=6,
    )
    stage.start()
    try:
        stage.submit(
            OmniSpeechTask(
                request_id="request",
                prompt_token_ids=(10, 11),
                output_token_ids=(12, 13),
                thinker_features=tensorplay.zeros(3, 4),
                special_embeddings=tensorplay.zeros(3, 4),
                voice="Chelsie",
                finish_reason=FinishReason.STOP,
            )
        )
        assert backend.called.wait(1)
        deadline = time.monotonic() + 1
        results = ()
        while not results and time.monotonic() < deadline:
            results = stage.drain()
        assert len(results) == 1
        assert tuple(chunk.data for chunk in results[0].chunks) == (
            b"abcdef",
            b"ghijkl",
        )
        assert results[0].finish_reason is FinishReason.STOP
        assert results[0].error is None
    finally:
        stage.close()


def test_engine_captures_talker_features_and_masks_media_embeddings() -> None:
    engine = Engine.__new__(Engine)
    engine.model = SimpleNamespace(
        config=SimpleNamespace(
            image_token_id=99,
            video_token_id=98,
            audio_token_id=97,
        )
    )
    engine._omni_captures = {
        "audio": _OmniFeatureCapture(
            prompt_token_ids=(99, 2),
            audio_output=AudioOutputOptions(),
        )
    }
    batch = ExecutionBatch(
        input_ids=tensorplay.tensor((99, 2, 3), dtype=tensorplay.int64),
        positions=tensorplay.tensor((0, 1, 0), dtype=tensorplay.int64),
        slot_mapping=tensorplay.tensor((0, 1, 2), dtype=tensorplay.int64),
        block_tables=tensorplay.zeros((2, 1), dtype=tensorplay.int32),
        sequence_lengths=tensorplay.tensor((2, 1), dtype=tensorplay.int32),
        token_active=tensorplay.ones(3, dtype=tensorplay.bool),
        query_start_loc=(0, 2, 3),
        query_lengths=(2, 1),
        context_lengths=(0, 0),
        phases=(BatchPhase.PREFILL, BatchPhase.PREFILL),
        sample_indices=tensorplay.tensor((1, 2), dtype=tensorplay.int64),
        page_size=1,
    )
    scheduled = ScheduledBatch(
        execution=batch,
        request_ids=("audio", "text"),
        scheduled_tokens=(2, 1),
        should_sample=(True, True),
        sampling=(None, None),
        histories=((), ()),
    )
    pending = PendingEngineStep(
        prior_events=(),
        scheduled=scheduled,
        sampled=tensorplay.tensor((4, 5)),
        sample_ready_event=None,
        step_started_ns=0,
        scheduled_ns=0,
        model_launched_ns=0,
        sampler_launched_ns=0,
        omni_input_embeddings=tensorplay.ones((3, 4)),
        omni_hidden_states=tensorplay.full((3, 4), 2.0),
    )

    engine._capture_omni_step(pending)

    captured = engine._omni_captures["audio"].features
    assert len(captured) == 1
    assert tensorplay.equal(captured[0][0], tensorplay.full((4,), 2.0))
    assert tensorplay.equal(captured[0][1], tensorplay.full((4,), 3.0))
