"""Rust tool-parser registry and Python gateway wiring tests.

These tests exercise the cross-language boundary without loading model weights:
the Rust extension is already built, so we can feed it decoded text directly.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from megalm.megalm_rs import PyOutputParser, PyToolParser

from megalm.gateway.chat_api import _make_output_parser
from megalm.runtime.tokenizer import FrontendError, ModelTokenizer

DSML = "\uff5cDSML\uff5c"
WEATHER_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get the weather",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {"type": "string"},
                    "days": {"type": "integer"},
                    "metric": {"type": "boolean"},
                },
            },
        },
    }
]


def _dsml_call(*, city: str = "SF", days: int = 3, metric: bool = True) -> str:
    return (
        f"<{DSML}function_calls>\n"
        f'<{DSML}invoke name="get_weather">\n'
        f'<{DSML}parameter name="city" string="true">{city}</{DSML}parameter>\n'
        f'<{DSML}parameter name="days" string="false">{days}</{DSML}parameter>\n'
        f'<{DSML}parameter name="metric" string="false">{str(metric).lower()}'
        f"</{DSML}parameter>\n"
        f"</{DSML}invoke>\n"
        f"</{DSML}function_calls>"
    )


def test_rust_deepseek_parser_coerces_schema_types() -> None:
    parser = PyToolParser(
        "deepseek_v32",
        json.dumps([tool["function"] for tool in WEATHER_TOOLS]),
    )
    text = "Let me check. " + _dsml_call()
    visible, calls = parser.push(text[:17])
    assert visible == "Let me check. "
    assert calls == []

    visible, calls = parser.push(text[17:])
    assert visible == ""
    assert len(calls) == 1
    index, name, arguments = calls[0]
    assert index == 0
    assert name == "get_weather"
    assert json.loads(arguments) == {"city": "SF", "days": 3, "metric": True}
    assert parser.preserve_special_tokens() is True


def test_rust_qwen_parser_buffers_split_markers() -> None:
    parser = PyToolParser(
        "qwen3_xml",
        json.dumps([tool["function"] for tool in WEATHER_TOOLS]),
    )
    chunks = [
        "hi <tool",
        '_call>\n{"name":"get_weather",',
        '"arguments":{"city":"SF"}}\n</tool_call>',
    ]
    visible = []
    calls = []
    for chunk in chunks:
        text, produced = parser.push(chunk)
        visible.append(text)
        calls.extend(produced)
    assert "".join(visible) == "hi "
    assert len(calls) == 1
    assert calls[0][1] == "get_weather"
    assert json.loads(calls[0][2]) == {"city": "SF"}


def test_output_parser_uses_dynamic_family_parser() -> None:
    parser = PyOutputParser(
        "req-1",
        [],
        ["get_weather"],
        None,
        None,
        False,
        "deepseek_v32",
        json.dumps([tool["function"] for tool in WEATHER_TOOLS]),
    )
    text = _dsml_call()
    content = ""
    calls = []
    for character in text:
        chunk_content, reasoning, produced, stopped = parser.push(character)
        assert reasoning == ""
        assert stopped is False
        content += chunk_content
        calls.extend(produced)
    content += parser.finish()[0]
    assert content == ""
    assert len(calls) == 1
    assert calls[0][0] == "req-1_0"
    assert json.loads(calls[0][2])["days"] == 3


def _tokenizer_with_template(
    template: str, model_type: str = "", model_name: str = "model"
) -> ModelTokenizer:
    tokenizer = object.__new__(ModelTokenizer)
    tokenizer._template_source = template
    tokenizer._model_type = model_type
    tokenizer._model_name = model_name.lower()
    return tokenizer


@pytest.mark.parametrize(
    ("template", "model_type", "model_name", "expected"),
    [
        (f"<{DSML}function_calls>", "", "model", ("deepseek_v32", True)),
        (f"<{DSML}tool_calls>", "", "model", ("deepseek_v4", True)),
        ("<tool_call>\n{{ ... }}\n</tool_call>", "qwen3", "model", ("qwen3_xml", False)),
        ("<tool_call>\n{{ ... }}\n</tool_call>", "", "model", ("hermes", False)),
        ("plain chat template", "", "model", ("generic_json", False)),
        ("[TOOL_CALLS] [{...}]", "", "model", ("mistral", False)),
        ("functools[{...}]", "", "model", ("phi4mini", False)),
        ("<|action_start|><|plugin|>{...}", "", "model", ("internlm2", True)),
        ("<seed:tool_call>...</seed:tool_call>", "", "model", ("seed_oss", False)),
        ("<minimax:tool_call>...</minimax:tool_call>", "", "model", ("minimax_m2", False)),
        ("<|tool_calls_section_begin|>", "", "model", ("kimi_k2", True)),
        ("<tool_sep>", "", "model", ("hy_v3", False)),
        ("<function=get_weather><parameter=city>SF</parameter>", "", "model", ("qwen3_coder", False)),
        ("<arg_key>city</arg_key><arg_value>SF</arg_value>", "", "model", ("glm47", False)),
        ("plain", "", "Qwen3-Coder-30B", ("qwen3_coder", False)),
        ("plain", "", "Meta-Llama-3.1-8B", ("llama3_json", False)),
        ("plain", "", "hy4-model", ("hy_v3", False)),
        ("plain", "", "Nemotron-3-Super", ("nemotron_v3", False)),
        ("plain", "", "granite-4.0", ("granite4", False)),
        ("plain", "", "glm-4.6", ("glm45", False)),
    ],
)
def test_tokenizer_detects_tool_parser(
    template: str,
    model_type: str,
    model_name: str,
    expected: tuple[str, bool],
) -> None:
    tokenizer = _tokenizer_with_template(template, model_type, model_name)
    assert tokenizer.tool_call_parser() == expected


def test_tool_parser_environment_override(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MEGALM_TOOL_PARSER", "hermes")
    tokenizer = _tokenizer_with_template("plain chat template")
    assert tokenizer.tool_call_parser() == ("hermes", False)
    monkeypatch.setenv("MEGALM_TOOL_PARSER", "deepseek_v32")
    assert tokenizer.tool_call_parser() == ("deepseek_v32", True)
    monkeypatch.setenv("MEGALM_TOOL_PARSER", "not_a_parser")
    with pytest.raises(ValueError, match="unknown MEGALM_TOOL_PARSER"):
        tokenizer.tool_call_parser()


def test_agent_arguments_json_string_is_canonicalized_for_template() -> None:
    tokenizer = ModelTokenizer(Path("model"))
    messages = [
        {"role": "user", "content": "weather?"},
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
    ]
    rendered = tokenizer.render_chat(messages)
    assert '"city":"SF"' in rendered
    assert "sunny" in rendered

    messages[1]["tool_calls"][0]["function"]["arguments"] = "not json"
    with pytest.raises(FrontendError, match="valid JSON"):
        tokenizer.render_chat(messages)


def test_forced_tool_name_is_filtered_in_rust() -> None:
    parser = PyOutputParser(
        "req-1",
        [],
        ["allowed", "other"],
        None,
        None,
        False,
        "generic_json",
        None,
        "allowed",
    )
    content, _, calls, _ = parser.push(
        '{"name":"other","args":{}}{"name":"allowed","args":{"x":1}}'
    )
    assert content == ""
    assert len(calls) == 1
    assert calls[0][1] == "allowed"


def test_gateway_selects_family_parser_for_special_token_grammar(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tokenizer = ModelTokenizer(Path("model"))
    monkeypatch.setattr(tokenizer, "output_grammar", lambda: "text")
    monkeypatch.setattr(tokenizer, "tool_call_parser", lambda: ("deepseek_v32", True))
    parser = _make_output_parser(
        SimpleNamespace(request_id="req-1"),
        tokenizer=tokenizer,
        stop_sequences=(),
        tools=tuple(WEATHER_TOOLS),
        forced_tool=None,
        preserve_special_tokens=True,
    )
    assert isinstance(parser, PyOutputParser)
    text = _dsml_call(city="Paris")
    content, _, calls, _ = parser.push(text)
    assert content == ""
    assert len(calls) == 1
    assert json.loads(calls[0][2])["city"] == "Paris"


def test_gateway_disables_tool_parsing_without_declared_tools(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tokenizer = ModelTokenizer(Path("model"))
    monkeypatch.setattr(tokenizer, "output_grammar", lambda: "text")
    monkeypatch.setattr(tokenizer, "tool_call_parser", lambda: ("deepseek_v32", True))
    parser = _make_output_parser(
        SimpleNamespace(request_id="req-1"),
        tokenizer=tokenizer,
        stop_sequences=(),
        tools=(),
        forced_tool=None,
        preserve_special_tokens=True,
    )
    content, _, calls, _ = parser.push(_dsml_call())
    assert calls == []
    assert "function_calls" in content