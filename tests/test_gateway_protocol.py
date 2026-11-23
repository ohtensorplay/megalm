import asyncio
import fcntl
import os
import socket
from pathlib import Path
from types import SimpleNamespace

import orjson
import pytest

from megalm.contracts.request import (
    AudioChunk,
    AudioInput,
    EngineEvent,
    EventKind,
    FinishReason,
    ImageInput,
    OutputModality,
    RequestState,
    SamplingParams,
    VideoInput,
    VideoSampling,
)
from megalm.gateway.media_ipc import (
    media_inputs_from_fds,
    seal_media_inputs,
)
from megalm.gateway.protocol import (
    FrameType,
    ReplicaProtocolError,
    output_from_wire,
    outputs_to_wire,
    receive_frame,
    receive_frame_with_fds,
    sampling_from_wire,
    sampling_to_wire,
    send_frame,
    send_frame_with_fds,
)
from megalm.gateway.detokenize import IncrementalDecoder
from megalm.runtime.tokenizer import ModelTokenizer


def test_replica_frame_round_trip_supports_metric_histograms() -> None:
    sender, receiver = socket.socketpair()
    try:
        sent = send_frame(
            sender,
            FrameType.SNAPSHOT,
            {"histogram": {1: 4, 64: 128}},
        )
        frame = receive_frame(receiver)
    finally:
        sender.close()
        receiver.close()

    assert sent > 0
    assert frame == (
        FrameType.SNAPSHOT,
        {"histogram": {"1": 4, "64": 128}},
    )


def test_replica_sampling_and_output_wire_contracts_round_trip() -> None:
    sampling = SamplingParams(
        temperature=0.7,
        top_p=0.9,
        top_k=32,
        min_p=0.05,
        repetition_penalty=1.1,
        seed=7,
        stop_token_ids=(8, 9),
        ignore_eos=True,
    )
    assert sampling_from_wire(sampling_to_wire(sampling)) == sampling

    token_event = EngineEvent(
        request_id="request",
        sequence_no=5,
        kind=EventKind.TOKEN,
        state=RequestState.DECODE,
        monotonic_ns=123,
        token_ids=(11, 12),
    )
    terminal_event = EngineEvent(
        request_id="request",
        sequence_no=6,
        kind=EventKind.TERMINAL,
        state=RequestState.FINISHED,
        monotonic_ns=124,
        finish_reason=FinishReason.LENGTH,
    )
    outputs = outputs_to_wire(
        (token_event, terminal_event),
        {"request": 7},
    )
    assert len(outputs) == 1
    assert output_from_wire(outputs[0]) == (
        7,
        (11, 12),
        (),
        FinishReason.LENGTH,
        None,
        (),
    )


def test_replica_audio_output_wire_round_trip() -> None:
    audio_event = EngineEvent(
        request_id="request",
        sequence_no=7,
        kind=EventKind.AUDIO,
        state=RequestState.OUTPUT_BLOCKED,
        monotonic_ns=125,
        audio=AudioChunk(data=b"RIFF-wave"),
        schema_version=2,
    )
    terminal_event = EngineEvent(
        request_id="request",
        sequence_no=8,
        kind=EventKind.TERMINAL,
        state=RequestState.FINISHED,
        monotonic_ns=126,
        finish_reason=FinishReason.STOP,
        schema_version=2,
    )

    outputs = outputs_to_wire(
        (audio_event, terminal_event),
        {"request": 9},
    )

    assert output_from_wire(outputs[0]) == (
        9,
        (),
        (AudioChunk(data=b"RIFF-wave"),),
        FinishReason.STOP,
        None,
        (),
    )


def test_replica_media_fds_are_sealed_and_digest_verified() -> None:
    image = ImageInput.from_bytes(b"image-payload", "image/png")
    video = VideoInput.from_bytes(
        b"video-payload",
        "video/mp4",
        sampling=VideoSampling(
            target_fps=None,
            num_frames=8,
            start_seconds=1.0,
            end_seconds=4.0,
        ),
    )
    audio = AudioInput.from_bytes(b"audio-payload", "audio/wav")
    sender, receiver = socket.socketpair()
    try:
        with seal_media_inputs((image,), (video,), (audio,)) as media:
            sent = send_frame_with_fds(
                sender,
                FrameType.SUBMIT_MULTIMODAL,
                {"media": media.descriptors},
                media.fds,
            )
        frame = receive_frame_with_fds(receiver)
        assert frame is not None
        frame_type, payload, fds = frame
        assert all(not os.get_inheritable(fd) for fd in fds)
        assert all(
            fcntl.fcntl(fd, getattr(fcntl, "F_GET_SEALS", 1034))
            & getattr(fcntl, "F_SEAL_WRITE", 0x0008)
            for fd in fds
        )
        restored = media_inputs_from_fds(payload["media"], fds)
    finally:
        sender.close()
        receiver.close()

    assert sent > 0
    assert frame_type is FrameType.SUBMIT_MULTIMODAL
    assert restored.images == (image,)
    assert restored.videos == (video,)
    assert restored.audios == (audio,)
    for fd in fds:
        try:
            os.fstat(fd)
        except OSError:
            pass
        else:
            raise AssertionError("received media descriptor leaked")


def test_replica_multimodal_batch_frame_preserves_media_ranges() -> None:
    first = ImageInput.from_bytes(b"first-image", "image/png")
    second = AudioInput.from_bytes(b"second-audio", "audio/wav")
    sender, receiver = socket.socketpair()
    try:
        with (
            seal_media_inputs((first,), ()) as first_media,
            seal_media_inputs((), (), (second,)) as second_media,
        ):
            descriptors = (
                first_media.descriptors + second_media.descriptors
            )
            fds = first_media.fds + second_media.fds
            send_frame_with_fds(
                sender,
                FrameType.SUBMIT_MULTIMODAL_BATCH,
                {
                    "media": descriptors,
                    "submissions": [
                        {"media_start": 0, "media_count": 1},
                        {"media_start": 1, "media_count": 1},
                    ],
                },
                fds,
            )
        frame = receive_frame_with_fds(receiver)
        assert frame is not None
        frame_type, payload, received_fds = frame
        restored_first = media_inputs_from_fds(
            payload["media"][0:1],
            received_fds[0:1],
        )
        restored_second = media_inputs_from_fds(
            payload["media"][1:2],
            received_fds[1:2],
        )
    finally:
        sender.close()
        receiver.close()

    assert frame_type is FrameType.SUBMIT_MULTIMODAL_BATCH
    assert restored_first.images == (first,)
    assert restored_second.audios == (second,)


def test_replica_rejects_unsealed_media_descriptor_and_closes_it() -> None:
    sender, receiver = socket.socketpair()
    raw_fd = os.memfd_create("unsealed", os.MFD_CLOEXEC)
    os.write(raw_fd, b"payload")
    try:
        send_frame_with_fds(
            sender,
            FrameType.SUBMIT_MULTIMODAL,
            {
                "media": [
                    {
                        "kind": "image",
                        "mime_type": "image/png",
                        "size": 7,
                        "sha256": "0" * 64,
                    }
                ]
            },
            (raw_fd,),
        )
        frame = receive_frame_with_fds(receiver)
        assert frame is not None
        _, payload, received_fds = frame
        try:
            media_inputs_from_fds(payload["media"], received_fds)
        except ReplicaProtocolError as exc:
            assert "sealed" in str(exc)
        else:
            raise AssertionError("unsealed media descriptor was accepted")
        for fd in received_fds:
            try:
                os.fstat(fd)
            except OSError:
                pass
            else:
                raise AssertionError("rejected media descriptor leaked")
    finally:
        os.close(raw_fd)
        sender.close()
        receiver.close()


def test_chat_request_carries_frequency_and_presence_penalties() -> None:
    from pydantic import ValidationError

    from megalm.gateway.chat_api import ChatCompletionRequest

    request = ChatCompletionRequest(
        model="m",
        messages=[{"role": "user", "content": "hi"}],
        frequency_penalty=1.5,
        presence_penalty=-0.5,
    )
    sampling = request.sampling()
    assert sampling.frequency_penalty == 1.5
    assert sampling.presence_penalty == -0.5

    with pytest.raises(ValidationError):
        ChatCompletionRequest(
            model="m",
            messages=[{"role": "user", "content": "hi"}],
            frequency_penalty=2.5,
        )


def test_collect_chat_multi_merges_choices_with_indices() -> None:
    from megalm.gateway.chat_api import _collect_chat_multi

    tokenizer = ModelTokenizer(Path("model"))
    first, second = _split_for_stop(tokenizer, "hello world!", "world")
    events = [_event(first), _event(second, finish=FinishReason.LENGTH)]
    streams = [
        _FakeStream(events, request_id="req-0"),
        _FakeStream(events, request_id="req-1"),
    ]
    replica = _FakeReplica()

    response = asyncio.run(
        _collect_chat_multi(
            streams=streams,
            replica=replica,
            tokenizer=tokenizer,
            model="test",
            created=0,
            modalities=(OutputModality.TEXT,),
            stop_sequences=("world",),
        )
    )

    assert [choice["index"] for choice in response["choices"]] == [0, 1]
    assert all(
        choice["message"]["content"] == "hello "
        for choice in response["choices"]
    )
    assert response["usage"]["prompt_tokens"] == 6
    assert replica.cancelled == ["req-0", "req-1"]


def test_stream_chat_multi_interleaves_choices_with_usage() -> None:
    import orjson

    from megalm.gateway.chat_api import _stream_chat_multi

    tokenizer = ModelTokenizer(Path("model"))
    first, second = _split_for_stop(tokenizer, "hello world!", "world")
    events = [_event(first), _event(second, finish=FinishReason.LENGTH)]
    streams = [
        _FakeStream(events, request_id="req-0"),
        _FakeStream(events, request_id="req-1"),
    ]
    replica = _FakeReplica()

    async def run() -> list[dict]:
        chunks: list[dict] = []
        async for payload in _stream_chat_multi(
            streams=streams,
            replica=replica,
            tokenizer=tokenizer,
            model="test",
            created=0,
            modalities=(OutputModality.TEXT,),
            stop_sequences=("world",),
            include_usage=True,
        ):
            text = payload.decode("utf-8")
            assert text.startswith("data: ")
            body = text[len("data: "):].strip()
            if body == "[DONE]":
                chunks.append({"done": True})
                continue
            chunks.append(orjson.loads(body))
        return chunks

    chunks = asyncio.run(run())
    assert sum(1 for chunk in chunks if chunk.get("done")) == 1
    role_chunks = [
        chunk
        for chunk in chunks
        if chunk.get("choices")
        and chunk["choices"][0]["delta"].get("role")
    ]
    assert sorted(
        chunk["choices"][0]["index"] for chunk in role_chunks
    ) == [0, 1]
    content_by_index: dict[int, str] = {}
    usage_trailer = None
    for chunk in chunks:
        if chunk.get("choices"):
            index = chunk["choices"][0]["index"]
            content_by_index.setdefault(index, "")
            content_by_index[index] += chunk["choices"][0]["delta"].get(
                "content", ""
            )
        if chunk.get("choices") == [] and "usage" in chunk:
            usage_trailer = chunk["usage"]
    assert content_by_index == {0: "hello ", 1: "hello "}
    assert usage_trailer is not None
    assert usage_trailer["total_tokens"] > 0
    assert replica.cancelled == ["req-0", "req-1"]


def test_incremental_decoder_preserves_cross_token_utf8() -> None:
    tokenizer = ModelTokenizer(Path("model"))
    token_ids = tokenizer.encode("你好，world! café 🚀")
    decoder = IncrementalDecoder(tokenizer)

    streamed = "".join(decoder.push((token_id,)) for token_id in token_ids)
    streamed += decoder.finish()

    expected = tokenizer.decode(token_ids, skip_special_tokens=False)
    assert streamed == expected
    assert decoder.text == expected
    assert decoder.token_count == len(token_ids)


class _FakeStream:
    def __init__(self, events: list, request_id: str = "req-1") -> None:
        self.request_id = request_id
        self.prompt_tokens = 3
        self.events = asyncio.Queue()
        for event in events:
            self.events.put_nowait(event)

    async def get(self):
        return await self.events.get()


class _FakeReplica:
    def __init__(self) -> None:
        self.cancelled: list[str] = []

    def cancel(self, request_id: str) -> None:
        self.cancelled.append(request_id)


def _event(token_ids: tuple[int, ...], *, finish=None, logprobs=()):
    from types import SimpleNamespace

    return SimpleNamespace(
        token_ids=token_ids,
        audio=(),
        finish_reason=finish,
        error=None,
        logprobs=tuple(logprobs),
    )


def _split_for_stop(tokenizer, text: str, stop: str) -> list[tuple[int, ...]]:
    """Split tokens so `stop` only completes after the delta boundary."""

    token_ids = tokenizer.encode(text)
    for completed in range(1, len(token_ids) + 1):
        if stop in tokenizer.decode(token_ids[:completed]):
            split = completed - 1
            break
    else:
        split = len(token_ids) - 1
    if split == 0:
        pytest.skip("stop sequence completes within the first token")
    return token_ids[:split], token_ids[split:]


def test_collect_chat_truncates_at_stop_sequence_and_cancels() -> None:
    from megalm.gateway.chat_api import _collect_chat

    tokenizer = ModelTokenizer(Path("model"))
    first, second = _split_for_stop(tokenizer, "hello world!", "world")
    stream = _FakeStream(
        [
            _event(first),
            _event(second, finish=FinishReason.LENGTH),
        ],
    )
    replica = _FakeReplica()

    response = asyncio.run(
        _collect_chat(
            stream=stream,
            replica=replica,
            tokenizer=tokenizer,
            model="test",
            created=0,
            modalities=(OutputModality.TEXT,),
            stop_sequences=("world",),
        )
    )

    assert response["choices"][0]["finish_reason"] == "stop"
    assert response["choices"][0]["message"]["content"] == "hello "
    assert response["usage"]["completion_tokens"] == len(first) + len(second)
    assert replica.cancelled == ["req-1"]


def test_stream_chat_honors_stop_and_include_usage() -> None:
    from megalm.gateway.chat_api import _stream_chat

    tokenizer = ModelTokenizer(Path("model"))
    first, second = _split_for_stop(tokenizer, "hello world!", "world")
    replica = _FakeReplica()

    async def run(include_usage: bool) -> list[dict]:
        stream = _FakeStream(
            [
                _event(first),
                _event(second, finish=FinishReason.LENGTH),
            ],
        )
        chunks: list[dict] = []
        async for payload in _stream_chat(
            stream=stream,
            replica=replica,
            tokenizer=tokenizer,
            model="test",
            created=0,
            modalities=(OutputModality.TEXT,),
            stop_sequences=("world",),
            include_usage=include_usage,
        ):
            text = payload.decode("utf-8")
            assert text.startswith("data: ")
            body = text[len("data: "):].strip()
            if body == "[DONE]":
                continue
            chunks.append(orjson.loads(body))
        return chunks

    chunks = asyncio.run(run(include_usage=True))
    contents = "".join(
        chunk["choices"][0]["delta"].get("content", "")
        for chunk in chunks
        if chunk.get("choices")
    )
    assert contents == "hello "
    assert chunks[-1]["choices"] == []
    assert chunks[-1]["usage"]["total_tokens"] > 0
    assert replica.cancelled == ["req-1"]

    replica = _FakeReplica()
    chunks = asyncio.run(run(include_usage=False))
    assert all("usage" not in chunk for chunk in chunks)
    assert replica.cancelled == ["req-1"]


def test_normalized_tools_validates_openai_inputs() -> None:
    from megalm.gateway.chat_api import _normalized_tools

    tools, forced = _normalized_tools(
        [
            {
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "description": "weather",
                    "parameters": {"type": "object"},
                },
            }
        ],
        {"type": "function", "function": {"name": "get_weather"}},
    )
    assert forced == "get_weather"
    assert tools[0]["type"] == "function"
    assert tools[0]["function"]["name"] == "get_weather"

    assert _normalized_tools(None, None) == ((), None)
    assert _normalized_tools([{"type": "function", "function": {"name": "f"}}], "none") == ((), None)

    with pytest.raises(ValueError):
        _normalized_tools(None, "auto")
    with pytest.raises(ValueError):
        _normalized_tools([{"type": "function", "function": {"name": "f"}}], "required_missing")
    with pytest.raises(ValueError):
        _normalized_tools([{"type": "function", "function": {"name": "f"}}], {"type": "function", "function": {"name": "missing"}})
    with pytest.raises(ValueError):
        _normalized_tools([{"type": "function", "function": {"name": "f"}}, {"type": "function", "function": {"name": "f"}}], None)


def test_collect_chat_extracts_tool_calls() -> None:
    from megalm.gateway.chat_api import _collect_chat

    tokenizer = ModelTokenizer(Path("model"))
    text = 'Sure. {"name":"get_weather","arguments":{"city":"SF"}}'
    token_ids = tokenizer.encode(text)
    stream = _FakeStream([_event(tuple(token_ids), finish=FinishReason.STOP)])
    tools = ({"type": "function", "function": {"name": "get_weather"}},)

    response = asyncio.run(
        _collect_chat(
            stream=stream,
            replica=_FakeReplica(),
            tokenizer=tokenizer,
            model="test",
            created=0,
            modalities=(OutputModality.TEXT,),
            tools=tools,
        )
    )

    choice = response["choices"][0]
    message = choice["message"]
    assert choice["finish_reason"] == "tool_calls"
    assert "get_weather" not in message["content"]
    assert len(message["tool_calls"]) == 1
    call = message["tool_calls"][0]
    assert call["type"] == "function"
    assert call["function"]["name"] == "get_weather"
    assert orjson.loads(call["function"]["arguments"]) == {"city": "SF"}
    assert call["id"].startswith("req-1")


def test_stream_chat_emits_tool_call_chunks() -> None:
    from megalm.gateway.chat_api import _stream_chat

    tokenizer = ModelTokenizer(Path("model"))
    text = 'get_weather{"name":"get_weather","args":{"city":"SF"}}'
    token_ids = tokenizer.encode(text)
    stream = _FakeStream([_event(tuple(token_ids), finish=FinishReason.STOP)])
    tools = ({"type": "function", "function": {"name": "get_weather"}},)

    async def run() -> list[dict]:
        chunks: list[dict] = []
        async for payload in _stream_chat(
            stream=stream,
            replica=_FakeReplica(),
            tokenizer=tokenizer,
            model="test",
            created=0,
            modalities=(OutputModality.TEXT,),
            tools=tools,
            include_usage=True,
        ):
            body = payload.decode("utf-8")[len("data: "):].strip()
            if body == "[DONE]":
                continue
            chunks.append(orjson.loads(body))
        return chunks

    chunks = asyncio.run(run())
    tool_chunks = [
        chunk
        for chunk in chunks
        if chunk.get("choices")
        and chunk["choices"][0]["delta"].get("tool_calls")
    ]
    assert len(tool_chunks) == 1
    delta_call = tool_chunks[0]["choices"][0]["delta"]["tool_calls"][0]
    assert delta_call["index"] == 0
    assert delta_call["function"]["name"] == "get_weather"
    assert orjson.loads(delta_call["function"]["arguments"]) == {"city": "SF"}
    finish = [c for c in chunks if c.get("choices") and c["choices"][0].get("finish_reason")]
    assert finish[-1]["choices"][0]["finish_reason"] == "tool_calls"
    assert chunks[-1]["choices"] == []


def test_stream_chat_fragments_tool_arguments_when_opted_in(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from megalm.gateway.chat_api import _stream_chat

    monkeypatch.setenv("MEGALM_STREAM_TOOL_ARGUMENTS", "1")
    tokenizer = ModelTokenizer(Path("model"))
    arguments = '{"city":"San Francisco","days":3}'
    text = 'get_weather{"name":"get_weather","args":' + arguments + "}"
    stream = _FakeStream([_event(tuple(tokenizer.encode(text)), finish=FinishReason.STOP)])
    tools = ({"type": "function", "function": {"name": "get_weather"}},)

    async def run() -> list[dict]:
        chunks: list[dict] = []
        async for payload in _stream_chat(
            stream=stream,
            replica=_FakeReplica(),
            tokenizer=tokenizer,
            model="test",
            created=0,
            modalities=(OutputModality.TEXT,),
            tools=tools,
        ):
            body = payload.decode("utf-8")[len("data: "):].strip()
            if body != "[DONE]":
                chunks.append(orjson.loads(body))
        return chunks

    chunks = asyncio.run(run())
    tool_chunks = [
        chunk
        for chunk in chunks
        if chunk.get("choices")
        and chunk["choices"][0]["delta"].get("tool_calls")
    ]
    assert len(tool_chunks) > 1
    first = tool_chunks[0]["choices"][0]["delta"]["tool_calls"][0]
    assert first["id"].startswith("req-1")
    assert first["type"] == "function"
    assert first["function"]["name"] == "get_weather"
    assembled = first["function"]["arguments"]
    for chunk in tool_chunks[1:]:
        delta_call = chunk["choices"][0]["delta"]["tool_calls"][0]
        assert delta_call["index"] == first["index"] == 0
        assert "id" not in delta_call
        assert "function" in delta_call
        assert "name" not in delta_call["function"]
        assembled += delta_call["function"]["arguments"]
    assert assembled == arguments
    assert orjson.loads(assembled) == {"city": "San Francisco", "days": 3}
    finish = [
        chunk
        for chunk in chunks
        if chunk.get("choices") and chunk["choices"][0].get("finish_reason")
    ]
    assert finish[-1]["choices"][0]["finish_reason"] == "tool_calls"
