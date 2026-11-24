import json
import shutil
from pathlib import Path

from megalm.runtime.tokenizer import (
    ModelTokenizer,
    _special_token_text,
    _template_strftime_now,
    _template_tojson,
)


def test_special_token_text_accepts_string_and_mapping() -> None:
    assert _special_token_text({"bos_token": "<s>"}, "bos_token") == "<s>"
    assert (
        _special_token_text({"bos_token": {"content": "<s>"}}, "bos_token") == "<s>"
    )
    assert _special_token_text({}, "bos_token") == ""


def test_template_tojson_accepts_vllm_style_keywords() -> None:
    assert _template_tojson({"x": 1}, ensure_ascii=True, sort_keys=True) == '{"x": 1}'


def test_template_tojson_supports_hf_indent_and_ascii() -> None:
    assert _template_tojson(
        {"b": 1, "a": "é"}, indent=2, sort_keys=True, ensure_ascii=True
    ) == (
        '{\n  "a": "\\u00e9",\n  "b": 1\n}'
    )


def test_template_strftime_now_formats_current_year() -> None:
    assert len(_template_strftime_now("%Y")) == 4


def test_generation_tags_are_stripped_with_whitespace_control() -> None:
    assert (
        ModelTokenizer._strip_generation_tags(
            "a{%- generation -%}B{%  endgeneration  %}C"
        )
        == "aBC"
    )


def test_named_templates_select_tool_use_when_tools_are_present(tmp_path: Path) -> None:
    shutil.copy(Path("model") / "tokenizer.json", tmp_path / "tokenizer.json")
    (tmp_path / "tokenizer_config.json").write_text(
        json.dumps(
            {
                "eos_token": "<|content_model_end_sampling|>",
                "chat_template": {
                    "default": "{{ messages[0].content }}",
                    "tool_use": "TOOLS:{{ tools|length }}",
                },
            }
        ),
        encoding="utf-8",
    )
    tokenizer = ModelTokenizer(tmp_path)
    assert tokenizer.render_chat(({"role": "user", "content": "hi"},)) == "hi"
    assert (
        tokenizer.render_chat(
            ({"role": "user", "content": "hi"},),
            tools=[{"type": "function"}],
        )
        == "TOOLS:1"
    )


def test_extra_template_variables_are_available() -> None:
    tokenizer = ModelTokenizer("model")
    rendered = tokenizer.render_chat(
        ({"role": "user", "content": "hi"},),
        extras={"reasoning_effort": 0.4},
    )
    assert "Thinking effort level: 0.4" in rendered


def test_plain_assistant_history_has_template_optional_fields() -> None:
    tokenizer = ModelTokenizer("model")

    rendered = tokenizer.render_chat(
        (
            {"role": "user", "content": "first"},
            {"role": "assistant", "content": "answer"},
            {"role": "user", "content": "next"},
        ),
        open_thinking=False,
    )

    assert "answer" in rendered
    # Inkling-2's template does not append a Think block after the assistant
    # message; it closes the message and (optionally) opens the next turn.
    assert rendered.endswith("<|end_message|>") or rendered.endswith(
        "<|message_model_end_sampling|>"
    )


def test_tokenizer_collects_all_eos_ids_from_model_config(tmp_path: Path) -> None:
    source = Path("model")
    for name in ("tokenizer.json", "tokenizer_config.json"):
        shutil.copy(source / name, tmp_path / name)
    (tmp_path / "chat_template.jinja").write_text(
        "{{ messages[0].content }}", encoding="utf-8"
    )
    (tmp_path / "config.json").write_text(
        json.dumps({"model_type": "llama", "eos_token_id": [1, 130073]}),
        encoding="utf-8",
    )

    tokenizer = ModelTokenizer(tmp_path)

    assert tokenizer.eos_token_id in tokenizer.eos_token_ids
    assert 1 in tokenizer.eos_token_ids
    assert 130073 in tokenizer.eos_token_ids


def test_tokenizer_falls_back_to_generation_config_pad(tmp_path: Path) -> None:
    """MiniCPM4 omits pad_token but declares pad_token_id in generation config."""

    shutil.copy(Path("model") / "tokenizer.json", tmp_path / "tokenizer.json")
    (tmp_path / "tokenizer_config.json").write_text(
        json.dumps(
            {
                "bos_token": None,
                "eos_token": "<|content_model_end_sampling|>",
                "pad_token": None,
                "chat_template": "{{ messages[0].content }}",
            }
        ),
        encoding="utf-8",
    )
    (tmp_path / "generation_config.json").write_text(
        json.dumps({"pad_token_id": 2}), encoding="utf-8"
    )
    (tmp_path / "config.json").write_text(
        json.dumps({"model_type": "minicpm", "eos_token_id": [2, 73440]}),
        encoding="utf-8",
    )

    tokenizer = ModelTokenizer(tmp_path)

    assert tokenizer.pad_token_id == 2
    assert tokenizer.eos_token_ids == frozenset({tokenizer.eos_token_id, 2, 73440})


def test_render_chat_with_tools_uses_extended_tojson() -> None:
    tokenizer = ModelTokenizer("model")

    rendered = tokenizer.render_chat(
        ({"role": "user", "content": "weather?"},),
        tools=[
            {
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "description": "weather lookup",
                    "parameters": {"type": "object", "properties": {}},
                },
            }
        ],
    )

    assert "tool_declare" in rendered
    assert '"name":"get_weather"' in rendered
    assert '"type":"function"' in rendered


def test_audio_placeholder_expands_without_vision_boundaries() -> None:
    tokenizer = object.__new__(ModelTokenizer)
    tokenizer.image_token_id = 10
    tokenizer.video_token_id = 11
    tokenizer.audio_token_id = 12
    tokenizer.vision_start_token_id = 20
    tokenizer.vision_end_token_id = 21
    tokenizer.audio_start_token_id = 30
    tokenizer.audio_end_token_id = 31

    expanded = tokenizer._expand_media_placeholders(
        (1, 30, 12, 31, 2),
        image_feature_counts=(),
        video_frame_feature_counts=(),
        video_timestamps=(),
        audio_feature_counts=(4,),
        video_timestamp_tokens=False,
    )

    assert expanded == (1, 30, 12, 12, 12, 12, 31, 2)
