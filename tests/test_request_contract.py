from dataclasses import FrozenInstanceError

import pytest

from megalm.contracts.request import (
    AudioOutputOptions,
    EngineError,
    EventKind,
    FinishReason,
    ImageInput,
    InferenceRequest,
    OutputModality,
    RequestContractError,
    RequestState,
    RequestStateMachine,
    SamplingParams,
)


def test_audio_output_options_are_tied_to_audio_modality() -> None:
    request = InferenceRequest(
        request_id="audio",
        model="qwen2_5_omni",
        input_token_ids=(1, 2),
        max_new_tokens=3,
        output_modalities=(
            OutputModality.TEXT,
            OutputModality.AUDIO,
        ),
        audio_output=AudioOutputOptions(voice="Chelsie"),
    )

    assert request.audio_output == AudioOutputOptions(voice="Chelsie")


def test_request_is_immutable_and_media_digest_is_verified() -> None:
    image = ImageInput.from_bytes(b"image-payload", "image/png")
    request = InferenceRequest(
        request_id="req-1",
        model="smoke",
        input_token_ids=[1, 2, 3],
        max_new_tokens=4,
        images=[image],
    )

    assert request.input_token_ids == (1, 2, 3)
    # Detail is part of the vision cache key, not of the payload digest.
    assert request.media_hashes == (f"image:{image.content_digest}:{image.detail}",)
    with pytest.raises(FrozenInstanceError):
        request.priority = 7  # type: ignore[misc]
    with pytest.raises(RequestContractError, match="does not match"):
        ImageInput(
            content_digest=image.content_digest,
            mime_type="image/png",
            payload=b"different",
        )


def test_prefix_fingerprint_requires_a_sha256_digest() -> None:
    with pytest.raises(RequestContractError, match="SHA-256"):
        InferenceRequest(
            request_id="bad-prefix",
            model="smoke",
            input_token_ids=(1,),
            max_new_tokens=1,
            prefix_fingerprint="not-a-digest",
        )

    fingerprint = "ab" * 32
    request = InferenceRequest(
        request_id="valid-prefix",
        model="smoke",
        input_token_ids=(1,),
        max_new_tokens=1,
        prefix_fingerprint=fingerprint,
    )
    assert request.prefix_fingerprint == fingerprint


def test_lifecycle_orders_events_and_closes_retry_boundary_at_first_token() -> None:
    lifecycle = RequestStateMachine("req-1")
    events = [
        lifecycle.transition(RequestState.ROUTING, now_ns=1),
        lifecycle.transition(RequestState.LEASED, now_ns=2),
        lifecycle.transition(RequestState.ADMITTED, now_ns=3),
        lifecycle.transition(RequestState.READY, now_ns=4),
        lifecycle.transition(RequestState.PREFILL, now_ns=5),
        lifecycle.transition(RequestState.DECODE, now_ns=6),
    ]
    assert [event.sequence_no for event in events] == list(range(6))
    assert lifecycle.can_retry_after_replica_failure

    token = lifecycle.emit_tokens([42], now_ns=7)
    assert token.kind is EventKind.TOKEN
    assert token.sequence_no == 6
    assert not lifecycle.can_retry_after_replica_failure

    lifecycle.transition(RequestState.FINISHING, now_ns=8)
    terminal = lifecycle.transition(
        RequestState.FINISHED,
        finish_reason=FinishReason.LENGTH,
        now_ns=9,
    )
    assert terminal.kind is EventKind.TERMINAL
    assert terminal.finish_reason is FinishReason.LENGTH
    assert terminal.sequence_no == 8


def test_silent_lifecycle_preserves_state_and_event_sequence() -> None:
    lifecycle = RequestStateMachine("req-compact")
    for state in (
        RequestState.ROUTING,
        RequestState.LEASED,
        RequestState.ADMITTED,
        RequestState.READY,
        RequestState.PREFILL,
        RequestState.DECODE,
    ):
        lifecycle.transition_silent(state)
    lifecycle.emit_token_silent(42)
    lifecycle.transition_silent(RequestState.FINISHING)
    terminal = lifecycle.transition(
        RequestState.FINISHED,
        finish_reason=FinishReason.LENGTH,
        now_ns=9,
    )

    assert terminal.sequence_no == 8
    assert terminal.finish_reason is FinishReason.LENGTH
    assert not lifecycle.can_retry_after_replica_failure


def test_invalid_transition_and_post_execution_rejection_are_forbidden() -> None:
    lifecycle = RequestStateMachine("req-1")
    with pytest.raises(RequestContractError, match="invalid request transition"):
        lifecycle.transition(RequestState.DECODE)

    for state in (
        RequestState.ROUTING,
        RequestState.LEASED,
        RequestState.ADMITTED,
        RequestState.READY,
        RequestState.PREFILL,
    ):
        lifecycle.transition(state)
    with pytest.raises(RequestContractError, match="invalid request transition"):
        lifecycle.transition(
            RequestState.REJECTED,
            error=EngineError("overloaded", "capacity unavailable", retryable=True),
        )


def test_rejection_requires_structured_error_before_execution() -> None:
    lifecycle = RequestStateMachine("req-1")
    lifecycle.transition(RequestState.ROUTING)
    event = lifecycle.transition(
        RequestState.REJECTED,
        error=EngineError("no_capacity", "replica is full", retryable=True),
    )
    assert event.finish_reason is FinishReason.REJECTED
    assert event.error is not None and event.error.retryable


@pytest.mark.parametrize(
    "kwargs",
    (
        {"temperature": float("nan")},
        {"repetition_penalty": float("inf")},
        {"top_k": 1.5},
        {"seed": "123"},
        {"frequency_penalty": 3.0},
        {"frequency_penalty": float("nan")},
        {"presence_penalty": -2.5},
        {"presence_penalty": float("inf")},
    ),
)
def test_sampling_rejects_non_finite_or_ambiguous_values(
    kwargs: dict[str, object],
) -> None:
    with pytest.raises(RequestContractError):
        SamplingParams(**kwargs)  # type: ignore[arg-type]


def test_frequency_and_presence_penalties_apply_on_cpu_path() -> None:
    import tensorplay

    from megalm.runtime.sampler import Sampler

    sampler = Sampler(use_triton=False)
    logits = tensorplay.tensor([[0.0, 0.0, 0.0, 0.0]])

    sampled = sampler.sample(
        tensorplay.tensor([[0.0, 0.0, 0.0, 0.0]]),
        (SamplingParams(temperature=0.0, frequency_penalty=1.0),),
        ("r1",),
        ((0, 0, 1),),
        history_counts=((2, 1),),
    )
    # tokens 0/1 shift by -(presence + frequency * count) = -2 and -1.
    assert sampled.tolist() == [2]

    sampled = sampler.sample(
        tensorplay.tensor([[0.0, 0.0, 0.0, 0.0]]),
        (SamplingParams(temperature=0.0, presence_penalty=1.0),),
        ("r1",),
        ((0, 0, 1),),
        history_counts=((2, 1),),
    )
    # presence shifts every seen token by exactly -1.
    assert sampled.tolist() == [2]
