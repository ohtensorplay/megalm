import asyncio
import base64
import io
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest
from PIL import Image

from megalm.gateway.chat_media import (
    MediaRequestError,
    parse_media_chat,
)
from megalm.gateway.media_fetch import (
    MediaFetchConfig,
    MediaFetchError,
    MediaFetchStats,
    fetch_remote_media,
    resolve_remote_media,
)


def _png(size: tuple[int, int] = (64, 64)) -> bytes:
    output = io.BytesIO()
    Image.new("RGB", size, (200, 30, 40)).save(output, format="PNG")
    return output.getvalue()


def _test_config(**overrides: object) -> MediaFetchConfig:
    options: dict[str, object] = {
        "allow_http": True,
        "allow_loopback": True,
        "max_attempts": 1,
        "max_redirects": 3,
        "total_timeout_s": 5.0,
        "read_timeout_s": 1.0,
        "connect_timeout_s": 1.0,
    }
    options.update(overrides)
    return MediaFetchConfig(**options)  # type: ignore[arg-type]


class _Handler(BaseHTTPRequestHandler):
    png = _png()
    flaky_calls = 0

    def log_message(self, *_: object) -> None:  # silence test noise
        pass

    def do_GET(self) -> None:
        path = self.path.split("?", 1)[0]
        if path == "/flaky":
            type(self).flaky_calls += 1
            if type(self).flaky_calls == 1:
                self.send_response(503)
                self.send_header("Content-Length", "0")
                self.end_headers()
                return
            self.send_response(200)
            self.send_header("Content-Type", "image/png")
            self.send_header("Content-Length", str(len(self.png)))
            self.end_headers()
            self.wfile.write(self.png)
            return
        if path == "/redirect-evil":
            self.send_response(302)
            self.send_header("Location", "http://evil.example/image.png")
            self.send_header("Content-Length", "0")
            self.end_headers()
            return
        if path == "/redirect":
            self.send_response(302)
            self.send_header("Location", "/image.png")
            self.send_header("Content-Length", "0")
            self.end_headers()
            return
        if path == "/loop":
            self.send_response(302)
            self.send_header("Location", "/loop")
            self.send_header("Content-Length", "0")
            self.end_headers()
            return
        if path == "/slow":
            import time

            time.sleep(1.0)
            self.send_response(200)
            self.send_header("Content-Type", "image/png")
            self.send_header("Content-Length", str(len(self.png)))
            self.end_headers()
            self.wfile.write(self.png)
            return
        if path == "/empty":
            self.send_response(200)
            self.send_header("Content-Length", "0")
            self.end_headers()
            return
        if path == "/large":
            body = b"x" * 4096
            self.send_response(200)
            self.send_header("Content-Type", "image/png")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        if path == "/chunked":
            self.send_response(200)
            self.send_header("Content-Type", "application/octet-stream")
            self.send_header("Transfer-Encoding", "chunked")
            self.end_headers()
            middle = len(self.png) // 2
            for part in (self.png[:middle], self.png[middle:]):
                self.wfile.write(f"{len(part):X}\r\n".encode())
                self.wfile.write(part)
                self.wfile.write(b"\r\n")
            self.wfile.write(b"0\r\n\r\n")
            return
        if path == "/not-found":
            self.send_response(404)
            self.send_header("Content-Length", "0")
            self.end_headers()
            return
        content_type = {
            "/image.png": "image/png",
            "/octet": "application/octet-stream",
            "/conflict": "image/jpeg",
        }.get(path, "image/png")
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(self.png)))
        self.end_headers()
        self.wfile.write(self.png)


@pytest.fixture()
def media_server():
    server = ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}"
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


def test_remote_media_is_sniffed_when_content_type_is_generic(media_server):
    stats = MediaFetchStats()
    mime, payload = asyncio.run(
        fetch_remote_media(
            f"{media_server}/octet",
            modality="image",
            config=_test_config(),
            stats=stats,
        )
    )
    assert mime == "image/png"
    assert payload == _Handler.png
    assert stats.snapshot()["successes"] == 1
    assert stats.snapshot()["bytes_downloaded"] == len(payload)


def test_chunked_transfer_and_redirect_are_supported(media_server):
    stats = MediaFetchStats()
    mime, payload = asyncio.run(
        fetch_remote_media(
            f"{media_server}/redirect",
            modality="image",
            config=_test_config(),
            stats=stats,
        )
    )
    assert mime == "image/png"
    assert payload == _Handler.png
    assert stats.snapshot()["redirects"] == 1

    stats = MediaFetchStats()
    mime, payload = asyncio.run(
        fetch_remote_media(
            f"{media_server}/chunked",
            modality="image",
            config=_test_config(),
            stats=stats,
        )
    )
    assert mime == "image/png"
    assert payload == _Handler.png


def test_redirect_loop_is_bounded(media_server):
    with pytest.raises(MediaFetchError) as exc:
        asyncio.run(
            fetch_remote_media(
                f"{media_server}/loop",
                modality="image",
                config=_test_config(max_redirects=2),
                stats=MediaFetchStats(),
            )
        )
    assert exc.value.code == "too_many_redirects"
    assert exc.value.status_code == 400


def test_byte_limit_is_enforced_from_content_length(media_server):
    with pytest.raises(MediaFetchError) as exc:
        asyncio.run(
            fetch_remote_media(
                f"{media_server}/large",
                modality="image",
                config=_test_config(max_bytes={"image": 1024}),
                stats=MediaFetchStats(),
            )
        )
    assert exc.value.code == "media_too_large"


def test_timeout_maps_to_504(media_server):
    with pytest.raises(MediaFetchError) as exc:
        asyncio.run(
            fetch_remote_media(
                f"{media_server}/slow",
                modality="image",
                config=_test_config(
                    read_timeout_s=0.15, total_timeout_s=1.0, max_attempts=1
                ),
                stats=MediaFetchStats(),
            )
        )
    assert exc.value.code == "timeout"
    assert exc.value.status_code == 504


def test_ssrf_guard_blocks_loopback_and_http_by_default(media_server):
    with pytest.raises(MediaFetchError) as exc:
        asyncio.run(
            fetch_remote_media(
                f"{media_server}/image.png", modality="image", stats=MediaFetchStats()
            )
        )
    assert exc.value.code == "http_disabled"

    with pytest.raises(MediaFetchError) as exc:
        asyncio.run(
            fetch_remote_media(
                f"{media_server}/image.png",
                modality="image",
                config=_test_config(allow_loopback=False),
                stats=MediaFetchStats(),
            )
        )
    assert exc.value.code == "blocked_address"


def test_unsupported_scheme_and_conflicting_bytes_are_rejected(media_server):
    with pytest.raises(MediaFetchError) as exc:
        asyncio.run(
            fetch_remote_media(
                "file:///etc/passwd", modality="image", stats=MediaFetchStats()
            )
        )
    assert exc.value.code == "unsupported_scheme"

    with pytest.raises(MediaFetchError) as exc:
        asyncio.run(
            fetch_remote_media(
                f"{media_server}/conflict",
                modality="image",
                config=_test_config(),
                stats=MediaFetchStats(),
            )
        )
    assert exc.value.code == "invalid_media"


def test_retries_are_bounded(media_server):
    port = int(media_server.rsplit(":", 1)[1])
    closed_port = port + 1000
    stats = MediaFetchStats()
    with pytest.raises(MediaFetchError) as exc:
        asyncio.run(
            fetch_remote_media(
                f"http://127.0.0.1:{closed_port}/image.png",
                modality="image",
                config=_test_config(max_attempts=2, allow_loopback=True),
                stats=stats,
            )
        )
    assert exc.value.code == "upstream_unreachable"
    assert stats.snapshot()["attempts"] == 2
    assert stats.snapshot()["failures"]["upstream_unreachable"] == 1


def test_resolve_remote_media_normalizes_messages_for_inline_parser(media_server):
    messages = (
        {
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"{media_server}/image.png",
                        "detail": "high",
                    },
                },
                {"type": "text", "text": "What is this?"},
            ],
        },
    )
    normalized, report = asyncio.run(
        resolve_remote_media(messages, config=_test_config(), stats=MediaFetchStats())
    )
    assert report.fetched == 1
    assert normalized[0]["content"][0]["image_url"]["url"].startswith("data:image/png;base64,")
    assert normalized[0]["content"][0]["image_url"]["detail"] == "high"

    parsed = parse_media_chat(normalized)
    assert len(parsed.images) == 1
    assert parsed.images[0].payload == _Handler.png
    assert parsed.messages[0]["content"][1] == {"type": "text", "text": "What is this?"}


def test_detail_is_passthrough_and_never_changes_payload():
    """The gateway carries detail as a fact; the processor applies the budget."""

    large_png = _png((1024, 1024))
    for detail in ("auto", "low", "high"):
        messages = (
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{base64.b64encode(large_png).decode()}",
                            "detail": detail,
                        },
                    }
                ],
            },
        )
        parsed = parse_media_chat(messages)
        assert parsed.images[0].detail == detail
        assert parsed.images[0].mime_type == "image/png"
        assert parsed.images[0].payload == large_png

    default = parse_media_chat(
        (
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{base64.b64encode(large_png).decode()}",
                        },
                    }
                ],
            },
        )
    )
    assert default.images[0].detail == "auto"


def test_unknown_detail_is_rejected():
    messages = (
        {
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/png;base64,{base64.b64encode(_png()).decode()}",
                        "detail": "ultra",
                    },
                }
            ],
        },
    )
    with pytest.raises(MediaRequestError, match="detail"):
        parse_media_chat(messages)


def test_config_snapshot_is_reportable():
    snapshot = _test_config().snapshot()
    assert snapshot["allow_http"] is True
    assert snapshot["max_bytes"]["image"] > 0


def test_domain_allowlist_exact_suffix_and_redirect(media_server, monkeypatch):
    import ipaddress

    from megalm.gateway import media_fetch

    monkeypatch.setattr(
        media_fetch,
        "_resolve_host",
        lambda host, port, config: ipaddress.ip_address("127.0.0.1"),
    )
    port = media_server.rsplit(":", 1)[1]

    # ".example.com" matches the exact domain and subdomains.
    mime, payload = asyncio.run(
        fetch_remote_media(
            f"http://cdn.example.com:{port}/image.png",
            modality="image",
            config=_test_config(allowed_domains=(".example.com",)),
            stats=MediaFetchStats(),
        )
    )
    assert mime == "image/png"
    assert payload == _Handler.png

    with pytest.raises(MediaFetchError) as exc:
        asyncio.run(
            fetch_remote_media(
                f"http://evil.other:{port}/image.png",
                modality="image",
                config=_test_config(allowed_domains=(".example.com",)),
                stats=MediaFetchStats(),
            )
        )
    assert exc.value.code == "domain_not_allowed"

    # A redirect from an allowed host to a disallowed one must be rejected on
    # the second hop (sglang-omni re-validates every hop too).
    with pytest.raises(MediaFetchError) as exc:
        asyncio.run(
            fetch_remote_media(
                f"http://127.0.0.1:{port}/redirect-evil",
                modality="image",
                config=_test_config(allowed_domains=("127.0.0.1",)),
                stats=MediaFetchStats(),
            )
        )
    assert exc.value.code == "domain_not_allowed"


def test_backslash_userinfo_confusion_is_rejected(media_server):
    """vLLM GHSA-v359-jj2v-j536: '\\' before '@' must not change the host.

    Python's urlsplit keeps the backslash inside userinfo/netloc rather than
    treating it as a separator; our single-parse + pinned-IP design has no
    second parser that could disagree, and userinfo is rejected outright.
    """

    port = media_server.rsplit(":", 1)[1]
    for url in (
        f"http://127.0.0.1:{port}\\@evil.example/image.png",
        "https://example.com\\@safe.example.org/image.png",
    ):
        with pytest.raises(MediaFetchError) as exc:
            asyncio.run(
                fetch_remote_media(
                    url,
                    modality="image",
                    config=_test_config(allowed_domains=("safe.example.org",)),
                    stats=MediaFetchStats(),
                )
            )
        assert exc.value.status_code == 400
        assert exc.value.code in {
            "userinfo_disallowed",
            "domain_not_allowed",
            "blocked_address",
        }


def test_upstream_4xx_maps_to_422_and_5xx_is_retried(media_server):
    with pytest.raises(MediaFetchError) as exc:
        asyncio.run(
            fetch_remote_media(
                f"{media_server}/not-found",
                modality="image",
                config=_test_config(),
                stats=MediaFetchStats(),
            )
        )
    assert exc.value.code == "upstream_status"
    assert exc.value.status_code == 422

    _Handler.flaky_calls = 0
    stats = MediaFetchStats()
    mime, payload = asyncio.run(
        fetch_remote_media(
            f"{media_server}/flaky",
            modality="image",
            config=_test_config(max_attempts=2),
            stats=stats,
        )
    )
    assert mime == "image/png"
    assert payload == _Handler.png
    assert stats.snapshot()["attempts"] == 2


def test_chat_route_fetches_remote_image_and_maps_errors(media_server, monkeypatch):
    from pathlib import Path
    from types import SimpleNamespace

    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from megalm.contracts.request import FinishReason
    from megalm.gateway import chat_api
    from megalm.gateway.chat_api import install_chat_routes
    from megalm.runtime.tokenizer import ModelTokenizer

    monkeypatch.setattr(chat_api, "MEDIA_FETCH_CONFIG", _test_config())

    class _Replica:
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

    tokenizer = ModelTokenizer(Path("model"))

    class _Batcher:
        async def encode(self, text):
            return tuple(tokenizer.encode(text))

    class _OmniBatcher:
        def __init__(self):
            self.images = None

        async def submit(self, messages, images, **kwargs):
            self.images = images
            stream = SimpleNamespace(
                request_id="req-omni",
                prompt_tokens=1,
                events=asyncio.Queue(),
            )
            stream.events.put_nowait(
                SimpleNamespace(
                    token_ids=tuple(tokenizer.encode("ok")),
                    audio=(),
                    finish_reason=FinishReason.STOP,
                    error=None,
                    logprobs=(),
                )
            )
            return stream

    omni = _OmniBatcher()
    app = FastAPI()
    install_chat_routes(
        app,
        replica=_Replica(),
        tokenizer=tokenizer,
        tokenizer_batcher=_Batcher(),
        admission_batcher=SimpleNamespace(submit=None),
        omni_admission_batcher=omni,
        served_model_name="test-model",
    )
    client = TestClient(app)
    payload = {
        "model": "test-model",
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": f"{media_server}/image.png"},
                    }
                ],
            }
        ],
    }

    response = client.post("/v1/chat/completions", json=payload)
    assert response.status_code == 200
    assert len(omni.images) == 1
    assert omni.images[0].payload == _Handler.png

    monkeypatch.setattr(
        chat_api, "MEDIA_FETCH_CONFIG", _test_config(allow_loopback=False)
    )
    response = client.post("/v1/chat/completions", json=payload)
    assert response.status_code == 400
