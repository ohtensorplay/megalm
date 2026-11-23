from megalm.runtime.tokenizer import ModelTokenizer


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
