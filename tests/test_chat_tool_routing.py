import asyncio
from pathlib import Path
from types import SimpleNamespace

import orjson
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from megalm.contracts.request import FinishReason, OutputModality
from megalm.gateway.chat_api import install_chat_routes
from megalm.megalm_rs import reasoning_boundary
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
    def __init__(
        self,
        tokenizer,
        output=(
            '<|message_model|><|content_invoke_tool_json|>'
            '{"name":"get_weather","args":{"city":"SF"}}'
            '<|end_message|>'
        ),
        logprobs=False,
    ):
        self.tokenizer = tokenizer
        self.output = output
        self.logprobs = logprobs
        self.submitted_sampling = None

    async def submit(self, token_ids, *, max_new_tokens, sampling, **kwargs):
        self.submitted_sampling = sampling
        stream = SimpleNamespace(
            request_id="req-1",
            prompt_tokens=len(token_ids),
            events=asyncio.Queue(),
        )
        token_ids_out = tuple(self.tokenizer.encode(self.output))
        row_logprobs = (
            tuple(-0.25 * (index + 1) for index in range(len(token_ids_out)))
            if self.logprobs
            else ()
        )
        stream.events.put_nowait(
            SimpleNamespace(
                token_ids=token_ids_out,
                audio=(),
                finish_reason=FinishReason.STOP,
                error=None,
                logprobs=row_logprobs,
            )
        )
        return stream


def _make_client(output=None, logprobs=False, tokenizer=None):
    app = FastAPI()
    tokenizer = tokenizer or ModelTokenizer(Path("model"))
    tokenizer_batcher = _TokenizerBatcher(tokenizer)
    admission = (
        _AdmissionBatcher(tokenizer, output, logprobs=logprobs)
        if output is not None
        else _AdmissionBatcher(tokenizer, logprobs=logprobs)
    )
    install_chat_routes(
        app,
        replica=_FakeReplica(),
        tokenizer=tokenizer,
        tokenizer_batcher=tokenizer_batcher,
        admission_batcher=admission,
        omni_admission_batcher=SimpleNamespace(),
        served_model_name="test-model",
    )
    app.state.tokenizer_batcher = tokenizer_batcher
    return TestClient(app)


@pytest.fixture
def client():
    with _make_client() as test_client:
        yield test_client


def test_agent_style_tool_result_round_trip() -> None:
    """LangChain/OpenAI-Agent style multi-turn messages render and complete."""

    tokenizer = ModelTokenizer(Path("model"))
    tools = [
        {
            "type": "function",
            "function": {
                "name": "get_weather",
                "parameters": {
                    "type": "object",
                    "properties": {"city": {"type": "string"}},
                },
            },
        }
    ]
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "weather in SF?"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "call_0",
                    "type": "function",
                    "function": {
                        "name": "get_weather",
                        "arguments": '{"city":"SF"}',
                    },
                }
            ],
        },
        {"role": "tool", "tool_call_id": "call_0", "content": "sunny"},
        {"role": "user", "content": "and tomorrow?"},
    ]
    with _make_client(tokenizer=tokenizer) as test_client:
        response = test_client.post(
            "/v1/chat/completions",
            json=_payload(messages=messages, tools=tools),
        )
        assert response.status_code == 200
        rendered = test_client.app.state.tokenizer_batcher.last_rendered
        assert "sunny" in rendered
        assert "get_weather" in rendered
        assert response.json()["choices"][0]["message"]["tool_calls"]


def _payload(**overrides):
    payload = {
        "model": "test-model",
        "messages": [{"role": "user", "content": "weather?"}],
    }
    payload.update(overrides)
    return payload


def test_tool_call_non_streaming_via_http(client):
    response = client.post(
        "/v1/chat/completions",
        json=_payload(
            tools=[
                {
                    "type": "function",
                    "function": {
                        "name": "get_weather",
                        "parameters": {"type": "object"},
                    },
                }
            ]
        ),
    )
    assert response.status_code == 200
    body = response.json()
    choice = body["choices"][0]
    assert choice["finish_reason"] == "tool_calls"
    call = choice["message"]["tool_calls"][0]
    assert call["function"]["name"] == "get_weather"
    assert "tool_declare" in client.app.state.tokenizer_batcher.last_rendered


def test_invalid_tool_choice_rejected(client):
    response = client.post(
        "/v1/chat/completions",
        json=_payload(
            tools=[{"type": "function", "function": {"name": "get_weather"}}],
            tool_choice={"type": "function", "function": {"name": "missing"}},
        ),
    )
    assert response.status_code == 400
    assert "unknown tool function" in response.json()["detail"]


def test_tools_without_tools_list_rejected(client):
    response = client.post(
        "/v1/chat/completions",
        json=_payload(tool_choice="auto"),
    )
    assert response.status_code == 400


def test_response_format_json_object_injects_instruction_and_passes():
    with _make_client(output='{"city":"SF","temp":20}') as test_client:
        response = test_client.post(
            "/v1/chat/completions",
            json=_payload(response_format={"type": "json_object"}),
        )
    assert response.status_code == 200
    body = response.json()
    assert body["choices"][0]["message"]["content"] == '{"city":"SF","temp":20}'
    rendered = test_client.app.state.tokenizer_batcher.last_rendered
    assert "valid JSON object" in rendered
    assert "message_system" in rendered


def test_response_format_json_schema_is_forwarded():
    schema = {
        "type": "object",
        "required": ["city"],
        "properties": {"city": {"type": "string"}},
    }
    with _make_client(output='{"city":"SF"}') as test_client:
        response = test_client.post(
            "/v1/chat/completions",
            json=_payload(
                response_format={
                    "type": "json_schema",
                    "json_schema": {"name": "weather", "schema": schema},
                }
            ),
        )
    assert response.status_code == 200
    rendered = test_client.app.state.tokenizer_batcher.last_rendered
    assert "JSON Schema" in rendered
    assert "city" in rendered


@pytest.mark.parametrize(
    "response_format",
    [
        {"type": "bogus"},
        {"type": "json_schema", "json_schema": {}},
        {"type": "json_schema", "json_schema": {"schema": []}},
    ],
)
def test_invalid_response_format_rejected(response_format):
    with _make_client() as test_client:
        response = test_client.post(
            "/v1/chat/completions",
            json=_payload(response_format=response_format),
        )
    assert response.status_code == 400


def test_reasoning_separated_from_content_non_streaming():
    with _make_client(
        output=(
            "<|message_model|><|content_thinking|>hidden reasoning"
            "<|end_message|><|message_model|><|content_text|>"
            "visible answer<|end_message|>"
        )
    ) as test_client:
        response = test_client.post(
            "/v1/chat/completions",
            json=_payload(),
        )
    assert response.status_code == 200
    message = response.json()["choices"][0]["message"]
    assert message["reasoning_content"] == "hidden reasoning"
    assert message["content"] == "visible answer"


def test_reasoning_streamed_before_content():
    with _make_client(
        output=(
            "<|message_model|><|content_thinking|>step one<|end_message|>"
            "<|message_model|><|content_text|>final<|end_message|>"
        )
    ) as test_client:
        with test_client.stream(
            "POST",
            "/v1/chat/completions",
            json=_payload(stream=True),
        ) as response:
            assert response.status_code == 200
            events = [
                orjson.loads(line[len("data: "):])
                for line in response.iter_lines()
                if line.startswith("data: ") and line != "data: [DONE]"
            ]
    deltas = [event["choices"][0]["delta"] for event in events if event["choices"]]
    reasoning = [
        delta["reasoning_content"] for delta in deltas if "reasoning_content" in delta
    ]
    content = [delta["content"] for delta in deltas if "content" in delta]
    assert reasoning == ["step one"]
    assert content == ["final"]
    finish = [event for event in events if event["choices"] and event["choices"][0].get("finish_reason")]
    assert finish[-1]["choices"][0]["finish_reason"] == "stop"


def test_tool_call_arguments_are_the_args_object():
    with _make_client(
        output=(
            '<|message_model|>get_weather<|content_invoke_tool_json|>'
            '{"name":"get_weather","args":{"city":"SF","unit":"c"}}'
            "<|end_message|>"
        )
    ) as test_client:
        response = test_client.post(
            "/v1/chat/completions",
            json=_payload(
                tools=[
                    {
                        "type": "function",
                        "function": {
                            "name": "get_weather",
                            "parameters": {"type": "object"},
                        },
                    }
                ]
            ),
        )
    assert response.status_code == 200
    choice = response.json()["choices"][0]
    assert choice["finish_reason"] == "tool_calls"
    call = choice["message"]["tool_calls"][0]
    assert call["function"]["name"] == "get_weather"
    assert call["function"]["arguments"] == '{"city":"SF","unit":"c"}'
    assert choice["message"]["content"] is None


def test_plain_text_output_is_not_mistaken_for_a_tool_call():
    with _make_client(output="just {not json} text") as test_client:
        response = test_client.post(
            "/v1/chat/completions",
            json=_payload(
                tools=[
                    {
                        "type": "function",
                        "function": {
                            "name": "get_weather",
                            "parameters": {"type": "object"},
                        },
                    }
                ]
            ),
        )
    assert response.status_code == 200
    choice = response.json()["choices"][0]
    assert choice["finish_reason"] == "stop"
    assert "tool_calls" not in choice["message"]
    assert choice["message"]["content"] == "just {not json} text"


def test_reasoning_boundary_pyfunction_matches_reference_scan():
    """Token-level scan, not a text suffix, decides the initial reasoning state."""

    start, end, other = 300_001, 300_002, 300_099
    specials = [start, end, other]
    assert reasoning_boundary([start], start, end, specials) is True
    assert reasoning_boundary([end], start, end, specials) is False
    assert reasoning_boundary([start, other], start, end, specials) is None
    assert reasoning_boundary([start, 7], start, end, specials) is True


def test_text_grammar_uses_prompt_tokens_for_reasoning_boundary():
    """A trailing text `<think>` must not win over a newer chat special token."""

    class _TokenGrammarTokenizer(ModelTokenizer):
        def output_grammar(self):
            return "text"

        def token_to_id(self, token):
            if token == "<think>":
                return 300_001
            if token == "</think>":
                return 300_002
            return super().token_to_id(token)

        def special_token_kinds(self):
            kinds = super().special_token_kinds()
            kinds[300_001] = "thinking"
            kinds[300_002] = "thinking"
            kinds[300_099] = "other"
            return kinds

        def render_chat(self, messages, *, tools=None, open_thinking=False):
            return "prompt <think>"

        def encode(self, text):
            # Oldest token is <think>, but the newest is an unrelated special
            # token: the reference scan stops there and defaults to content.
            if text == "prompt <think>":
                return (300_001, 300_099)
            return super().encode(text)

    tokenizer = _TokenGrammarTokenizer(Path("model"))
    with _make_client(
        output="reason</think>answer",
        tokenizer=tokenizer,
    ) as test_client:
        response = test_client.post("/v1/chat/completions", json=_payload())
    assert response.status_code == 200
    message = response.json()["choices"][0]["message"]
    assert "reasoning_content" not in message
    assert message["content"] == "reason</think>answer"


def test_text_grammar_models_keep_the_think_splitter():
    """Regression guard for Qwen3-style models served by the same gateway."""

    class _TextGrammarTokenizer(ModelTokenizer):
        def output_grammar(self):
            return "text"

    tokenizer = _TextGrammarTokenizer(Path("model"))
    with _make_client(
        output="<think>hidden reasoning</think>visible answer",
        tokenizer=tokenizer,
    ) as test_client:
        response = test_client.post("/v1/chat/completions", json=_payload())
    assert response.status_code == 200
    message = response.json()["choices"][0]["message"]
    assert message["reasoning_content"] == "hidden reasoning"
    assert message["content"] == "visible answer"


def test_output_without_thinking_markers_has_no_reasoning_field():
    with _make_client(output="plain answer") as test_client:
        response = test_client.post(
            "/v1/chat/completions",
            json=_payload(),
        )
    assert response.status_code == 200
    message = response.json()["choices"][0]["message"]
    assert "reasoning_content" not in message
    assert message["content"] == "plain answer"


def test_logprobs_flow_to_non_streaming_response():
    with _make_client(output="hello world", logprobs=True) as test_client:
        response = test_client.post(
            "/v1/chat/completions",
            json=_payload(logprobs=True),
        )
    assert response.status_code == 200
    choice = response.json()["choices"][0]
    entries = choice["logprobs"]["content"]
    assert all(entry["logprob"] < 0 for entry in entries)
    assert all(entry["top_logprobs"] == [] for entry in entries)
    assert "".join(entry["token"] for entry in entries) == "hello world"
    assert test_client.app.state.tokenizer_batcher.last_rendered is not None


def test_logprobs_absent_when_engine_did_not_collect():
    with _make_client(output="hello world") as test_client:
        response = test_client.post(
            "/v1/chat/completions",
            json=_payload(),
        )
    assert response.status_code == 200
    assert response.json()["choices"][0]["logprobs"] is None


def test_top_logprobs_rejected_with_clear_error():
    with _make_client() as test_client:
        response = test_client.post(
            "/v1/chat/completions",
            json=_payload(logprobs=True, top_logprobs=3),
        )
    assert response.status_code == 400
    assert "top_logprobs" in response.json()["detail"]


def test_parser_diagnostics_are_logged_and_metrics_recorded(caplog):
    """A degraded Inkling block stays visible but becomes observable."""

    app = FastAPI()
    tokenizer = ModelTokenizer(Path("model"))
    tokenizer_batcher = _TokenizerBatcher(tokenizer)
    admission = _AdmissionBatcher(
        tokenizer,
        output=(
            '<|content_invoke_tool_json|>{"name":"f","args":42}'
            "<|end_message|>visible fallback"
        ),
    )
    replica = _FakeReplica()
    from megalm.gateway.server import ReplicaMetrics

    replica.metrics = ReplicaMetrics()
    install_chat_routes(
        app,
        replica=replica,
        tokenizer=tokenizer,
        tokenizer_batcher=tokenizer_batcher,
        admission_batcher=admission,
        omni_admission_batcher=SimpleNamespace(),
        served_model_name="test-model",
    )
    with TestClient(app) as test_client, caplog.at_level("WARNING"):
        response = test_client.post("/v1/chat/completions", json=_payload())

    assert response.status_code == 200
    choice = response.json()["choices"][0]
    assert choice["message"]["content"] and '"args":42' in choice["message"]["content"]
    assert replica.metrics.parser_pushes >= 1
    assert replica.metrics.parser_diagnostics >= 1
    events = [record.message for record in caplog.records]
    assert "parser_degraded" in events


def test_replica_metrics_snapshot_exposes_latency_fields():
    from megalm.gateway.server import ReplicaMetrics

    metrics = ReplicaMetrics()
    snapshot = metrics.snapshot()
    assert snapshot["ttft_samples"] == 0
    assert snapshot["ttft_ms_mean"] == 0.0
    assert snapshot["tpot_samples"] == 0
    assert snapshot["tpot_ms_mean"] == 0.0

    metrics.ttft_ms_total += 42.0
    metrics.ttft_samples += 1
    metrics.tpot_ms_total += 8.0
    metrics.tpot_samples += 2
    snapshot = metrics.snapshot()
    assert snapshot["ttft_ms_mean"] == 42.0
    assert snapshot["tpot_ms_mean"] == 4.0
