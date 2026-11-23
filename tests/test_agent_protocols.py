"""Route-level tests for the native Anthropic and Responses protocols."""

import asyncio
from pathlib import Path
from types import SimpleNamespace

import orjson
from fastapi import FastAPI
from fastapi.testclient import TestClient

from megalm.contracts.request import FinishReason
from megalm.gateway.chat_api import install_chat_routes
from megalm.runtime.tokenizer import ModelTokenizer


class _FakeReplica:
    async def describe(self):
        return {
            "output_modalities": ("text",),
            "max_model_len": 4096,
            "audio_output": None,
        }

    def frontend_request_started(self):
        pass

    def frontend_request_finished(self):
        pass

    def cancel(self, request_id):
        pass


class _TokenizerBatcher:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.last_rendered = None

    async def encode(self, text):
        self.last_rendered = text
        return tuple(self.tokenizer.encode(text))


class _AdmissionBatcher:
    def __init__(self, tokenizer, output):
        self.tokenizer = tokenizer
        self.output = output

    async def submit(self, token_ids, *, max_new_tokens, sampling, **kwargs):
        stream = SimpleNamespace(
            request_id="req-1",
            prompt_tokens=len(token_ids),
            events=asyncio.Queue(),
        )
        stream.events.put_nowait(
            SimpleNamespace(
                token_ids=tuple(self.tokenizer.encode(self.output)),
                audio=(),
                finish_reason=FinishReason.STOP,
                error=None,
                logprobs=(),
            )
        )
        return stream


_TOOL_OUTPUT = (
    '<|message_model|><|content_invoke_tool_json|>'
    '{"name":"get_weather","args":{"city":"SF"}}'
    '<|end_message|>'
)


class _FakeOmniBatcher:
    def __init__(self, tokenizer, output):
        self.tokenizer = tokenizer
        self.output = output
        self.last_tools = None

    async def submit(self, messages, images, **kwargs):
        self.last_tools = kwargs.get("tools")
        stream = SimpleNamespace(
            request_id="req-omni",
            prompt_tokens=len(messages),
            events=asyncio.Queue(),
        )
        stream.events.put_nowait(
            SimpleNamespace(
                token_ids=tuple(self.tokenizer.encode(self.output)),
                audio=(),
                finish_reason=FinishReason.STOP,
                error=None,
                logprobs=(),
            )
        )
        return stream


def _make_client(
    output=_TOOL_OUTPUT, omni_batcher=None, builtin_tool_executor=None
):
    app = FastAPI()
    tokenizer = ModelTokenizer(Path("model"))
    tokenizer_batcher = _TokenizerBatcher(tokenizer)
    install_chat_routes(
        app,
        replica=_FakeReplica(),
        tokenizer=tokenizer,
        tokenizer_batcher=tokenizer_batcher,
        admission_batcher=_AdmissionBatcher(tokenizer, output),
        omni_admission_batcher=omni_batcher or SimpleNamespace(),
        served_model_name="test-model",
        builtin_tool_executor=builtin_tool_executor,
    )
    app.state.tokenizer_batcher = tokenizer_batcher
    return TestClient(app)


_ANTHROPIC_TOOL = {
    "name": "get_weather",
    "description": "Get the weather",
    "input_schema": {
        "type": "object",
        "properties": {"city": {"type": "string"}},
    },
}


def _anthropic_payload(**overrides):
    payload = {
        "model": "test-model",
        "max_tokens": 128,
        "messages": [{"role": "user", "content": "weather in SF?"}],
    }
    payload.update(overrides)
    return payload


def _parse_sse(response):
    events = []
    for line in response.text.split("\n\n"):
        event_type = None
        data = None
        for row in line.splitlines():
            if row.startswith("event: "):
                event_type = row[len("event: ") :]
            elif row.startswith("data: "):
                data = orjson.loads(row[len("data: ") :])
        if event_type is not None:
            events.append((event_type, data))
    return events


def test_anthropic_messages_non_streaming_tool_use():
    with _make_client() as client:
        response = client.post(
            "/v1/messages",
            json=_anthropic_payload(
                system="You are helpful.",
                tools=[_ANTHROPIC_TOOL],
                tool_choice={"type": "tool", "name": "get_weather"},
            ),
        )
    assert response.status_code == 200
    body = response.json()
    assert body["type"] == "message"
    assert body["role"] == "assistant"
    assert body["stop_reason"] == "tool_use"
    assert body["usage"]["input_tokens"] > 0
    tool_blocks = [block for block in body["content"] if block["type"] == "tool_use"]
    assert len(tool_blocks) == 1
    assert tool_blocks[0]["name"] == "get_weather"
    assert tool_blocks[0]["input"] == {"city": "SF"}


def test_anthropic_messages_streaming_tool_use():
    with _make_client() as client:
        response = client.post(
            "/v1/messages",
            json=_anthropic_payload(
                stream=True,
                tools=[_ANTHROPIC_TOOL],
                tool_choice={"type": "any"},
            ),
        )
    assert response.status_code == 200
    events = _parse_sse(response)
    types = [event_type for event_type, _ in events]
    assert types[0] == "message_start"
    assert "content_block_start" in types
    assert "content_block_delta" in types
    assert "content_block_stop" in types
    assert types[-2] == "message_delta"
    assert types[-1] == "message_stop"
    starts = [data for kind, data in events if kind == "content_block_start"]
    assert any(
        block["content_block"]["type"] == "tool_use"
        and block["content_block"]["name"] == "get_weather"
        for block in starts
    )
    deltas = [data for kind, data in events if kind == "content_block_delta"]
    assert any(
        data["delta"]["type"] == "input_json_delta"
        and orjson.loads(data["delta"]["partial_json"])["city"] == "SF"
        for data in deltas
    )
    message_delta = [data for kind, data in events if kind == "message_delta"][-1]
    assert message_delta["delta"]["stop_reason"] == "tool_use"


def test_anthropic_tool_result_round_trip():
    messages = [
        {"role": "user", "content": "weather in SF?"},
        {
            "role": "assistant",
            "content": [
                {
                    "type": "tool_use",
                    "id": "toolu_1",
                    "name": "get_weather",
                    "input": {"city": "SF"},
                }
            ],
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "tool_result",
                    "tool_use_id": "toolu_1",
                    "content": "sunny",
                }
            ],
        },
    ]
    with _make_client() as client:
        response = client.post("/v1/messages", json=_anthropic_payload(messages=messages))
    assert response.status_code == 200
    rendered = client.app.state.tokenizer_batcher.last_rendered
    assert "sunny" in rendered
    assert "get_weather" in rendered


def test_anthropic_error_shape():
    with _make_client() as client:
        response = client.post(
            "/v1/messages",
            json=_anthropic_payload(model="unknown"),
        )
    assert response.status_code == 404
    body = response.json()
    assert body["type"] == "error"
    assert body["error"]["type"] == "invalid_request_error"


_RESPONSES_TOOL = {
    "type": "function",
    "name": "get_weather",
    "description": "Get the weather",
    "parameters": {"type": "object", "properties": {"city": {"type": "string"}}},
}


def _responses_payload(**overrides):
    payload = {
        "model": "test-model",
        "input": "weather in SF?",
        "tools": [_RESPONSES_TOOL],
        "tool_choice": {"type": "function", "name": "get_weather"},
    }
    payload.update(overrides)
    return payload


def test_responses_non_streaming_function_call():
    with _make_client() as client:
        response = client.post("/v1/responses", json=_responses_payload())
    assert response.status_code == 200
    body = response.json()
    assert body["object"] == "response"
    assert body["status"] == "completed"
    assert body["usage"]["total_tokens"] > 0
    calls = [item for item in body["output"] if item["type"] == "function_call"]
    assert len(calls) == 1
    assert calls[0]["name"] == "get_weather"
    assert orjson.loads(calls[0]["arguments"]) == {"city": "SF"}


def test_responses_streaming_function_call():
    with _make_client() as client:
        response = client.post(
            "/v1/responses", json=_responses_payload(stream=True)
        )
    assert response.status_code == 200
    events = _parse_sse(response)
    types = [event_type for event_type, _ in events]
    assert types[0] == "response.created"
    assert types[1] == "response.in_progress"
    assert "response.output_item.added" in types
    assert "response.function_call_arguments.delta" in types
    assert "response.function_call_arguments.done" in types
    assert types[-1] == "response.completed"
    completed = [data for kind, data in events if kind == "response.completed"][-1]
    assert completed["response"]["status"] == "completed"
    calls = [
        item
        for item in completed["response"]["output"]
        if item["type"] == "function_call"
    ]
    assert calls and orjson.loads(calls[0]["arguments"]) == {"city": "SF"}


def test_responses_function_call_output_input():
    items = [
        {"role": "user", "content": "weather in SF?"},
        {
            "type": "function_call",
            "call_id": "call-1",
            "name": "get_weather",
            "arguments": '{"city":"SF"}',
        },
        {
            "type": "function_call_output",
            "call_id": "call-1",
            "output": "sunny",
        },
    ]
    with _make_client() as client:
        response = client.post(
            "/v1/responses",
            json=_responses_payload(input=items),
        )
    assert response.status_code == 200
    rendered = client.app.state.tokenizer_batcher.last_rendered
    assert "sunny" in rendered
    assert "get_weather" in rendered


def test_responses_text_output_and_instructions():
    with _make_client(output="Hello from MegaLM") as client:
        response = client.post(
            "/v1/responses",
            json={
                "model": "test-model",
                "input": "hi",
                "instructions": "be brief",
            },
        )
    assert response.status_code == 200
    body = response.json()
    rendered = client.app.state.tokenizer_batcher.last_rendered
    assert "be brief" in rendered
    messages = [item for item in body["output"] if item["type"] == "message"]
    assert messages
    assert messages[0]["content"][0]["text"] == "Hello from MegaLM"


def test_responses_error_shape():
    with _make_client() as client:
        response = client.post(
            "/v1/responses", json=_responses_payload(model="unknown")
        )
    assert response.status_code == 404
    body = response.json()
    assert body["error"]["type"] == "invalid_request_error"


def test_anthropic_count_tokens():
    """count_tokens has no max_tokens field in the Anthropic protocol."""

    with _make_client() as client:
        response = client.post(
            "/v1/messages/count_tokens",
            json={
                "model": "test-model",
                "messages": [{"role": "user", "content": "hello"}],
            },
        )
    assert response.status_code == 200
    assert response.json()["input_tokens"] > 0


def test_anthropic_tool_result_with_image_and_reference():
    messages = [
        {"role": "user", "content": "compare these"},
        {
            "role": "assistant",
            "content": [
                {
                    "type": "tool_use",
                    "id": "toolu_2",
                    "name": "lookup",
                    "input": {},
                }
            ],
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "tool_result",
                    "tool_use_id": "toolu_2",
                    "content": [
                        {"type": "text", "text": "found it"},
                        {
                            "type": "tool_reference",
                            "tool_name": "lookup",
                        },
                    ],
                }
            ],
        },
    ]
    with _make_client() as client:
        response = client.post("/v1/messages", json=_anthropic_payload(messages=messages))
    assert response.status_code == 200
    rendered = client.app.state.tokenizer_batcher.last_rendered
    assert "found it" in rendered


def test_anthropic_tool_result_image_translation():
    from megalm.gateway.agent_protocols import (
        AnthropicMessagesRequest,
        anthropic_to_chat,
    )

    request = AnthropicMessagesRequest.model_validate(
        {
            "model": "test-model",
            "max_tokens": 16,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": "toolu_3",
                            "content": [
                                {"type": "text", "text": "image follows"},
                                {
                                    "type": "image",
                                    "source": {
                                        "type": "base64",
                                        "media_type": "image/png",
                                        "data": "YWJj",
                                    },
                                },
                            ],
                        }
                    ],
                }
            ],
        }
    )
    payload = anthropic_to_chat(request)
    messages = [message.model_dump() for message in payload.messages]
    roles = [message["role"] for message in messages]
    assert roles == ["tool", "user"]
    assert messages[1]["content"] == [
        {
            "type": "image_url",
            "image_url": {"url": "data:image/png;base64,YWJj"},
        }
    ]


def test_responses_store_previous_retrieve_and_cancel():
    with _make_client(output="Sunny in SF") as client:
        first = client.post(
            "/v1/responses",
            json={"model": "test-model", "input": "weather?"},
        )
        assert first.status_code == 200
        response_id = first.json()["id"]

        retrieved = client.get(f"/v1/responses/{response_id}")
        assert retrieved.status_code == 200
        assert retrieved.json()["id"] == response_id

        second = client.post(
            "/v1/responses",
            json={
                "model": "test-model",
                "input": "and tomorrow?",
                "previous_response_id": response_id,
            },
        )
        assert second.status_code == 200
        assert second.json()["previous_response_id"] == response_id
        rendered = client.app.state.tokenizer_batcher.last_rendered
        assert "Sunny in SF" in rendered
        assert "and tomorrow?" in rendered

        cancelled = client.post(f"/v1/responses/{response_id}/cancel")
        assert cancelled.json()["status"] == "cancelled"

        missing = client.get("/v1/responses/does-not-exist")
        assert missing.status_code == 404
        missing_previous = client.post(
            "/v1/responses",
            json={
                "model": "test-model",
                "input": "hi",
                "previous_response_id": "does-not-exist",
            },
        )
        assert missing_previous.status_code == 404


def test_multimodal_request_keeps_tools():
    """Tool calling now works together with image inputs."""

    omni = _FakeOmniBatcher(ModelTokenizer(Path("model")), _TOOL_OUTPUT)
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "weather in this city?"},
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": "image/png",
                        "data": "iVBORw0KGgo=",
                    },
                },
            ],
        }
    ]
    with _make_client(omni_batcher=omni) as client:
        response = client.post(
            "/v1/messages",
            json=_anthropic_payload(
                messages=messages,
                tools=[_ANTHROPIC_TOOL],
                tool_choice={"type": "tool", "name": "get_weather"},
            ),
        )
    assert response.status_code == 200
    body = response.json()
    assert any(block["type"] == "tool_use" for block in body["content"])
    assert omni.last_tools
    assert omni.last_tools[0]["function"]["name"] == "get_weather"


def test_responses_streaming_retrieve_replays_events():
    with _make_client() as client:
        created = client.post(
            "/v1/responses", json=_responses_payload(stream=True)
        )
        events = _parse_sse(created)
        types = [event_type for event_type, _ in events]
        completed = [data for kind, data in events if kind == "response.completed"][-1]
        response_id = completed["response"]["id"]

        replay = client.get(
            f"/v1/responses/{response_id}", params={"stream": "true"}
        )
        replay_types = [event_type for event_type, _ in _parse_sse(replay)]
        assert replay_types == types

        sequence = [data["sequence_number"] for _, data in events]
        partial = client.get(
            f"/v1/responses/{response_id}",
            params={"stream": "true", "starting_after": sequence[1]},
        )
        partial_types = [event_type for event_type, _ in _parse_sse(partial)]
        assert partial_types == types[2:]


def test_responses_builtin_tool_requires_executor():
    payload = _responses_payload(
        tools=[
            {
                "type": "web_search_preview",
            }
        ],
        tool_choice="auto",
    )
    with _make_client() as client:
        response = client.post("/v1/responses", json=payload)
    assert response.status_code == 400
    assert "require a tool server" in response.json()["error"]["message"]


def test_responses_builtin_tool_with_executor():
    search_item = {
        "type": "web_search_call",
        "id": "ws_1",
        "status": "completed",
        "action": {"type": "search", "query": "weather SF"},
    }

    def executor(tools, request):
        assert tools[0]["type"] == "web_search_preview"
        return [search_item]

    payload = _responses_payload(
        tools=[
            {
                "type": "web_search_preview",
            }
        ],
        tool_choice="auto",
    )
    with _make_client(builtin_tool_executor=executor) as client:
        response = client.post("/v1/responses", json=payload)
        streamed = client.post(
            "/v1/responses", json={**payload, "stream": True}
        )
    assert response.status_code == 200
    assert response.json()["output"][0] == search_item
    types = [event_type for event_type, _ in _parse_sse(streamed)]
    assert types[2:4] == [
        "response.output_item.added",
        "response.output_item.done",
    ]
