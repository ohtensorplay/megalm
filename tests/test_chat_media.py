import base64

import pytest

from megalm.gateway.chat_media import MediaRequestError, parse_media_chat


def _wav_payload() -> bytes:
    return b"RIFF\x04\x00\x00\x00WAVE"


def test_audio_url_is_normalized_and_kept_as_immutable_audio() -> None:
    encoded = base64.b64encode(_wav_payload()).decode()

    parsed = parse_media_chat(
        (
            {
                "role": "user",
                "content": [
                    {
                        "type": "audio_url",
                        "audio_url": {
                            "url": f"data:audio/wav;base64,{encoded}"
                        },
                    },
                    {"type": "text", "text": "Transcribe it."},
                ],
            },
        )
    )

    assert parsed.images == ()
    assert parsed.videos == ()
    assert len(parsed.audios) == 1
    assert parsed.audios[0].payload == _wav_payload()
    assert parsed.messages[0]["content"][0] == {"type": "audio"}


def test_openai_input_audio_base64_contract_is_supported() -> None:
    parsed = parse_media_chat(
        (
            {
                "role": "user",
                "content": [
                    {
                        "type": "input_audio",
                        "input_audio": {
                            "data": base64.b64encode(
                                _wav_payload()
                            ).decode(),
                            "format": "wav",
                        },
                    }
                ],
            },
        )
    )

    assert parsed.audios[0].mime_type == "audio/wav"


def test_audio_declared_mime_must_match_bytes() -> None:
    encoded = base64.b64encode(b"not-wave").decode()

    with pytest.raises(MediaRequestError, match="do not match"):
        parse_media_chat(
            (
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "audio_url",
                            "audio_url": {
                                "url": f"data:audio/wav;base64,{encoded}"
                            },
                        }
                    ],
                },
            )
        )
