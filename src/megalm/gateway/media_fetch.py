"""Bounded, SSRF-safe remote media fetch for the Gateway media data plane.

Remote URLs are resolved to bytes *before* admission, then handed to the
existing inline-data path and sealed memfd transport. The HTTP client here is
deliberately small: it speaks HTTP/1.1 with a pinned resolved address, so DNS
rebinding between validation and connect cannot redirect a request to an
internal service. No new runtime dependency is required.
"""

from __future__ import annotations

import asyncio
import base64
import ipaddress
import logging
import math
import os
import socket
import ssl
import time
import urllib.parse
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Any

from megalm.gateway.chat_media import (
    MAX_HTTP_AUDIO_BYTES,
    MAX_HTTP_IMAGE_BYTES,
    MAX_HTTP_VIDEO_BYTES,
    MediaRequestError,
    resolve_media_mime,
)
from megalm.observability import log_event

_LOGGER = logging.getLogger("megalm.gateway.media_fetch")

_MAX_HEADER_BYTES = 16 * 1024
_READ_CHUNK_BYTES = 64 * 1024
_REDIRECT_STATUSES = frozenset({301, 302, 303, 307, 308})
_ALLOWED_SCHEMES = frozenset({"http", "https"})

_DEFAULT_LIMITS: dict[str, int] = {
    "image": MAX_HTTP_IMAGE_BYTES,
    "video": MAX_HTTP_VIDEO_BYTES,
    "audio": MAX_HTTP_AUDIO_BYTES,
}
_SUPPORTED_TYPES: dict[str, frozenset[str]] = {
    "image": frozenset(
        {"image/gif", "image/jpeg", "image/png", "image/webp"}
    ),
    "video": frozenset(
        {
            "video/mp4",
            "video/quicktime",
            "video/webm",
            "video/x-matroska",
            "video/x-msvideo",
        }
    ),
    "audio": frozenset(
        {
            "audio/flac",
            "audio/mp4",
            "audio/mpeg",
            "audio/ogg",
            "audio/wav",
            "audio/x-wav",
        }
    ),
}
_PART_MODALITIES: dict[str, str] = {
    "image_url": "image",
    "video_url": "video",
    "audio_url": "audio",
}


class MediaFetchError(MediaRequestError):
    """A remote media failure with a stable code and HTTP status mapping."""

    def __init__(self, code: str, status_code: int, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.status_code = status_code
        self.message = message

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"MediaFetchError(code={self.code!r}, status={self.status_code})"


def _env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _env_int(name: str, default: int, minimum: int, maximum: int) -> int:
    try:
        value = int(os.environ.get(name, ""))
    except ValueError:
        return default
    return value if minimum <= value <= maximum else default


def _env_float(name: str, default: float, minimum: float, maximum: float) -> float:
    try:
        value = float(os.environ.get(name, ""))
    except ValueError:
        return default
    return value if minimum <= value <= maximum else default


@dataclass(frozen=True, slots=True)
class MediaFetchConfig:
    """Cold-path policy for remote media. All limits are hard bounds."""

    enabled: bool = True
    allow_http: bool = False
    max_redirects: int = 3
    max_attempts: int = 2
    max_concurrent: int = 4
    connect_timeout_s: float = 5.0
    read_timeout_s: float = 15.0
    total_timeout_s: float = 60.0
    # Test-only seam for a loopback fixture server. Never read from the
    # environment: production configs always reject loopback/private hosts.
    allow_loopback: bool = False
    user_agent: str = "MegaLM-MediaFetch/1.0"
    # Exact hostnames or ".suffix" entries. Empty means unrestricted, matching
    # vLLM's default; production deployments should pin this down.
    allowed_domains: tuple[str, ...] = ()
    max_bytes: Mapping[str, int] = field(
        default_factory=lambda: dict(_DEFAULT_LIMITS)
    )

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "allowed_domains",
            tuple(
                domain.strip().rstrip(".").lower()
                for domain in self.allowed_domains
                if domain.strip()
            ),
        )
        if not 0 <= self.max_redirects <= 8:
            raise ValueError("max_redirects must be in [0, 8]")
        if not 1 <= self.max_attempts <= 3:
            raise ValueError("max_attempts must be in [1, 3]")
        if not 1 <= self.max_concurrent <= 16:
            raise ValueError("max_concurrent must be in [1, 16]")
        for name in ("connect_timeout_s", "read_timeout_s", "total_timeout_s"):
            value = getattr(self, name)
            if not 0 < value <= 300 or not math.isfinite(value):
                raise ValueError(f"{name} must be positive and finite")
        if self.connect_timeout_s > self.total_timeout_s:
            raise ValueError("connect_timeout_s exceeds total_timeout_s")
        for modality, limit in self.max_bytes.items():
            if modality not in _DEFAULT_LIMITS:
                raise ValueError(f"unknown media modality {modality!r}")
            if not 0 < limit <= _DEFAULT_LIMITS[modality]:
                raise ValueError(f"invalid byte limit for {modality}")

    def limit(self, modality: str) -> int:
        return self.max_bytes.get(modality, 0)

    def allowed_types(self, modality: str) -> frozenset[str]:
        return _SUPPORTED_TYPES.get(modality, frozenset())

    def snapshot(self) -> dict[str, Any]:
        fields = (
            "enabled",
            "allow_http",
            "max_redirects",
            "max_attempts",
            "max_concurrent",
            "connect_timeout_s",
            "read_timeout_s",
            "total_timeout_s",
            "user_agent",
        )
        return {
            **{name: getattr(self, name) for name in fields},
            "allowed_domains": list(self.allowed_domains),
            "max_bytes": dict(sorted(self.max_bytes.items())),
        }

    @classmethod
    def from_env(cls) -> MediaFetchConfig:
        return cls(
            enabled=_env_bool("MEGALM_MEDIA_REMOTE_URLS", True),
            allow_http=_env_bool("MEGALM_MEDIA_ALLOW_HTTP", False),
            max_redirects=_env_int("MEGALM_MEDIA_MAX_REDIRECTS", 3, 0, 8),
            max_attempts=_env_int("MEGALM_MEDIA_MAX_ATTEMPTS", 2, 1, 3),
            max_concurrent=_env_int("MEGALM_MEDIA_MAX_CONCURRENT", 4, 1, 16),
            connect_timeout_s=_env_float(
                "MEGALM_MEDIA_CONNECT_TIMEOUT_S", 5.0, 0.1, 60.0
            ),
            read_timeout_s=_env_float(
                "MEGALM_MEDIA_READ_TIMEOUT_S", 15.0, 0.1, 120.0
            ),
            total_timeout_s=_env_float(
                "MEGALM_MEDIA_TOTAL_TIMEOUT_S", 60.0, 1.0, 300.0
            ),
            allowed_domains=tuple(
                entry
                for entry in os.environ.get(
                    "MEGALM_MEDIA_ALLOWED_DOMAINS", ""
                ).split(",")
                if entry.strip()
            ),
        )


@dataclass(slots=True)
class MediaFetchStats:
    """Process-local counters folded into the Gateway status payload."""

    attempts: int = 0
    successes: int = 0
    redirects: int = 0
    bytes_downloaded: int = 0
    failures: dict[str, int] = field(default_factory=dict)

    def record_attempt(self) -> None:
        self.attempts += 1

    def record_redirect(self) -> None:
        self.redirects += 1

    def record_bytes(self, count: int) -> None:
        self.bytes_downloaded += count

    def record_success(self) -> None:
        self.successes += 1

    def record_failure(self, code: str) -> None:
        self.failures[code] = self.failures.get(code, 0) + 1

    def snapshot(self) -> dict[str, Any]:
        return {
            "attempts": self.attempts,
            "successes": self.successes,
            "redirects": self.redirects,
            "bytes_downloaded": self.bytes_downloaded,
            "failures": dict(sorted(self.failures.items())),
        }


MEDIA_FETCH_CONFIG = MediaFetchConfig.from_env()
MEDIA_FETCH_STATS = MediaFetchStats()


@dataclass(frozen=True, slots=True)
class RemoteFetchReport:
    fetched: int
    bytes_downloaded: int
    hosts: tuple[str, ...]


@lru_cache(maxsize=1)
def _tls_context() -> ssl.SSLContext:
    context = ssl.create_default_context()
    context.check_hostname = True
    context.verify_mode = ssl.CERT_REQUIRED
    return context


def _assert_domain_allowed(host: str, config: MediaFetchConfig) -> None:
    """Enforce the optional media-domain allowlist (vLLM/sglang-omni parity).

    A leading dot matches the exact domain and all subdomains; other entries
    match exactly. The check happens before DNS resolution on every redirect
    hop, so a redirect cannot bypass it.
    """

    if not config.allowed_domains:
        return
    normalized = host.rstrip(".").lower()
    for rule in config.allowed_domains:
        if rule.startswith("."):
            if normalized == rule[1:] or normalized.endswith(rule):
                return
        elif normalized == rule:
            return
    raise MediaFetchError(
        "domain_not_allowed",
        400,
        f"media domain {host!r} is not in the allowlist",
    )


def _validate_address(
    address: ipaddress.IPv4Address | ipaddress.IPv6Address,
    config: MediaFetchConfig,
) -> None:
    if config.allow_loopback:
        return
    mapped = getattr(address, "ipv4_mapped", None)
    if mapped is not None:
        address = mapped
    if (
        not address.is_global
        or address.is_loopback
        or address.is_private
        or address.is_link_local
        or address.is_multicast
        or address.is_reserved
        or address.is_unspecified
    ):
        raise MediaFetchError(
            "blocked_address",
            400,
            "media URL resolves to a non-public address",
        )


def _resolve_host(
    host: str,
    port: int,
    config: MediaFetchConfig,
) -> ipaddress.IPv4Address | ipaddress.IPv6Address:
    """Resolve, validate and pin one address for the upcoming connect."""

    try:
        literal = ipaddress.ip_address(host)
    except ValueError:
        literal = None
    if literal is not None:
        _validate_address(literal, config)
        return literal
    try:
        infos = socket.getaddrinfo(host, port, type=socket.SOCK_STREAM)
    except (OSError, UnicodeError) as exc:
        raise MediaFetchError(
            "dns_failure", 400, "media URL host could not be resolved"
        ) from exc
    addresses: list[ipaddress.IPv4Address | ipaddress.IPv6Address] = []
    for info in infos:
        address = ipaddress.ip_address(info[4][0])
        _validate_address(address, config)
        addresses.append(address)
    if not addresses:
        raise MediaFetchError(
            "dns_failure", 400, "media URL host has no usable address"
        )
    for address in addresses:
        if address.version == 4:
            return address
    return addresses[0]


def _parse_url(
    url: str,
    config: MediaFetchConfig,
) -> tuple[str, int, str, str]:
    try:
        parts = urllib.parse.urlsplit(url)
    except ValueError as exc:
        raise MediaFetchError("invalid_url", 400, "media URL is invalid") from exc
    scheme = parts.scheme.lower()
    if scheme not in _ALLOWED_SCHEMES:
        raise MediaFetchError(
            "unsupported_scheme", 400, "media URL scheme is not supported"
        )
    if scheme == "http" and not config.allow_http:
        raise MediaFetchError(
            "http_disabled", 400, "http media URLs are disabled; use https"
        )
    host = parts.hostname
    if not host:
        raise MediaFetchError("missing_host", 400, "media URL has no host")
    if parts.username or parts.password:
        raise MediaFetchError(
            "userinfo_disallowed",
            400,
            "media URL must not contain credentials",
        )
    try:
        port = parts.port
    except ValueError as exc:
        raise MediaFetchError("invalid_port", 400, "media URL port is invalid") from exc
    if port is None:
        port = 443 if scheme == "https" else 80
    if not 1 <= port <= 65535:
        raise MediaFetchError("invalid_port", 400, "media URL port is out of range")
    path = parts.path or "/"
    if parts.query:
        path = f"{path}?{parts.query}"
    try:
        path.encode("latin-1")
    except UnicodeEncodeError as exc:
        raise MediaFetchError("invalid_url", 400, "media URL path is invalid") from exc
    return host, port, scheme, path


async def _read_line(reader: asyncio.StreamReader, timeout: float) -> bytes:
    return await asyncio.wait_for(reader.readline(), timeout)


async def _read_exact(
    reader: asyncio.StreamReader,
    size: int,
    timeout: float,
    stats: MediaFetchStats,
) -> bytes:
    parts: list[bytes] = []
    remaining = size
    while remaining:
        chunk = await asyncio.wait_for(
            reader.read(min(remaining, _READ_CHUNK_BYTES)), timeout
        )
        if not chunk:
            raise MediaFetchError(
                "upstream_truncated", 502, "media response ended early"
            )
        parts.append(chunk)
        remaining -= len(chunk)
        stats.record_bytes(len(chunk))
    return b"".join(parts)


async def _read_chunked(
    reader: asyncio.StreamReader,
    limit: int,
    timeout: float,
    stats: MediaFetchStats,
) -> bytes:
    parts: list[bytes] = []
    total = 0
    while True:
        size_line = await _read_line(reader, timeout)
        size_token = size_line.strip().split(b";", 1)[0]
        if not size_token:
            continue
        try:
            size = int(size_token, 16)
        except ValueError as exc:
            raise MediaFetchError(
                "bad_chunk", 400, "media response has an invalid chunk size"
            ) from exc
        if size == 0:
            while True:
                trailer = await _read_line(reader, timeout)
                if trailer in (b"\r\n", b"\n", b""):
                    break
            break
        total += size
        if total > limit:
            raise MediaFetchError(
                "media_too_large", 400, "media payload exceeds its byte limit"
            )
        parts.append(await _read_exact(reader, size, timeout, stats))
        delimiter = await asyncio.wait_for(reader.readexactly(2), timeout)
        if delimiter != b"\r\n":
            raise MediaFetchError(
                "bad_chunk", 400, "media response has a malformed chunk"
            )
    return b"".join(parts)


async def _read_until_eof(
    reader: asyncio.StreamReader,
    limit: int,
    timeout: float,
    stats: MediaFetchStats,
) -> bytes:
    parts: list[bytes] = []
    total = 0
    while True:
        chunk = await asyncio.wait_for(reader.read(_READ_CHUNK_BYTES), timeout)
        if not chunk:
            break
        total += len(chunk)
        if total > limit:
            raise MediaFetchError(
                "media_too_large", 400, "media payload exceeds its byte limit"
            )
        parts.append(chunk)
        stats.record_bytes(len(chunk))
    return b"".join(parts)


async def _http_get(
    *,
    host: str,
    address: ipaddress.IPv4Address | ipaddress.IPv6Address,
    port: int,
    scheme: str,
    path: str,
    modality: str,
    config: MediaFetchConfig,
    stats: MediaFetchStats,
    timeout: float,
    connect_timeout: float,
) -> tuple[int, dict[str, str], bytes]:
    """One HTTP/1.1 exchange against the pre-validated, pinned address."""

    host_header = f"[{host}]" if ":" in host and not host.startswith("[") else host
    if port not in (80, 443):
        host_header = f"{host_header}:{port}"
    request = (
        f"GET {path} HTTP/1.1\r\n"
        f"Host: {host_header}\r\n"
        f"User-Agent: {config.user_agent}\r\n"
        "Accept: */*\r\n"
        "Accept-Encoding: identity\r\n"
        "Connection: close\r\n"
        "\r\n"
    ).encode("latin-1")

    reader, writer = await asyncio.wait_for(
        asyncio.open_connection(
            str(address),
            port,
            ssl=_tls_context() if scheme == "https" else None,
            server_hostname=host if scheme == "https" else None,
        ),
        connect_timeout,
    )
    try:
        writer.write(request)
        await asyncio.wait_for(writer.drain(), timeout)
        status_line = await _read_line(reader, timeout)
        pieces = status_line.decode("latin-1").strip().split(" ", 2)
        if len(pieces) < 2 or not pieces[1].isdigit():
            raise MediaFetchError(
                "bad_response", 502, "media host returned an invalid status line"
            )
        status = int(pieces[1])

        headers: dict[str, str] = {}
        header_bytes = 0
        while True:
            line = await _read_line(reader, timeout)
            if line in (b"\r\n", b"\n", b""):
                break
            header_bytes += len(line)
            if header_bytes > _MAX_HEADER_BYTES:
                raise MediaFetchError(
                    "headers_too_large", 400, "media response headers are too large"
                )
            name, separator, value = line.partition(b":")
            if separator:
                headers[
                    name.decode("latin-1").strip().lower()
                ] = value.decode("latin-1").strip()

        limit = config.limit(modality)
        declared_length: int | None = None
        if "content-length" in headers:
            try:
                declared_length = int(headers["content-length"])
            except ValueError as exc:
                raise MediaFetchError(
                    "invalid_content_length",
                    400,
                    "media response has an invalid Content-Length",
                ) from exc
            if declared_length < 0:
                raise MediaFetchError(
                    "invalid_content_length",
                    400,
                    "media response has a negative Content-Length",
                )
            if declared_length > limit:
                raise MediaFetchError(
                    "media_too_large", 400, "media payload exceeds its byte limit"
                )

        encoding = headers.get("transfer-encoding", "").lower()
        if "chunked" in encoding:
            body = await _read_chunked(reader, limit, timeout, stats)
        elif declared_length is not None:
            body = await _read_exact(reader, declared_length, timeout, stats)
        else:
            body = await _read_until_eof(reader, limit, timeout, stats)
        if not body and 200 <= status < 300:
            raise MediaFetchError("empty_media", 400, "media response body is empty")
        return status, headers, body
    finally:
        writer.close()
        try:
            await asyncio.wait_for(writer.wait_closed(), 1.0)
        except (TimeoutError, OSError):
            pass


async def fetch_remote_media(
    url: str,
    *,
    modality: str,
    config: MediaFetchConfig | None = None,
    stats: MediaFetchStats | None = None,
) -> tuple[str, bytes]:
    """Fetch one remote media URL and return ``(mime_type, payload)``."""

    config = config or MEDIA_FETCH_CONFIG
    stats = stats or MEDIA_FETCH_STATS
    if not config.enabled:
        raise MediaFetchError(
            "remote_disabled", 400, "remote media URLs are disabled"
        )
    if modality not in _SUPPORTED_TYPES:
        raise MediaFetchError("invalid_modality", 400, "media modality is invalid")

    started = time.monotonic()
    deadline = started + config.total_timeout_s
    current = url
    redirects = 0
    hosts: list[str] = []
    try:
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise MediaFetchError("timeout", 504, "media fetch timed out")
            host, port, scheme, path = _parse_url(current, config)
            _assert_domain_allowed(host, config)
            hosts.append(host)
            address = await asyncio.wait_for(
                asyncio.to_thread(_resolve_host, host, port, config),
                min(config.connect_timeout_s, remaining),
            )

            for attempt in range(config.max_attempts):
                attempt_remaining = deadline - time.monotonic()
                if attempt_remaining <= 0:
                    raise MediaFetchError("timeout", 504, "media fetch timed out")
                stats.record_attempt()
                try:
                    status, headers, body = await _http_get(
                        host=host,
                        address=address,
                        port=port,
                        scheme=scheme,
                        path=path,
                        modality=modality,
                        config=config,
                        stats=stats,
                        timeout=min(config.read_timeout_s, attempt_remaining),
                        connect_timeout=min(
                            config.connect_timeout_s, attempt_remaining
                        ),
                    )
                    # Retry transient upstream responses the same way vLLM's
                    # _is_retryable() does; 4xx otherwise fails immediately.
                    if (
                        500 <= status < 600 or status in (408, 429)
                    ) and attempt + 1 < config.max_attempts:
                        continue
                    break
                except TimeoutError as exc:
                    if attempt + 1 < config.max_attempts:
                        continue
                    raise MediaFetchError(
                        "timeout", 504, "media fetch timed out"
                    ) from exc
                except OSError as exc:
                    if attempt + 1 < config.max_attempts:
                        continue
                    raise MediaFetchError(
                        "upstream_unreachable", 502, "media host could not be reached"
                    ) from exc

            location = headers.get("location")
            if status in _REDIRECT_STATUSES and location:
                redirects += 1
                if redirects > config.max_redirects:
                    raise MediaFetchError(
                        "too_many_redirects", 400, "media URL redirected too many times"
                    )
                stats.record_redirect()
                current = urllib.parse.urljoin(current, location)
                continue
            if not 200 <= status < 300:
                # vLLM maps a media host's 4xx (except 408/429) to 422 because
                # the client-supplied URL is unprocessable, not our own 400.
                status_code = 422 if 400 <= status < 500 else 502
                raise MediaFetchError(
                    "upstream_status",
                    status_code,
                    f"media host returned HTTP {status}",
                )
            try:
                mime_type = resolve_media_mime(
                    headers.get("content-type"),
                    body,
                    modality,
                    config.allowed_types(modality),
                )
            except MediaRequestError as exc:
                raise MediaFetchError("invalid_media", 400, str(exc)) from exc
            stats.record_success()
            log_event(
                _LOGGER,
                logging.INFO,
                "media_fetch",
                result="ok",
                modality=modality,
                host=hosts[0],
                bytes=len(body),
                redirects=redirects,
                duration_ms=int((time.monotonic() - started) * 1000),
            )
            return mime_type, body
    except MediaFetchError as exc:
        stats.record_failure(exc.code)
        log_event(
            _LOGGER,
            logging.WARNING,
            "media_fetch",
            result="error",
            code=exc.code,
            modality=modality,
            host=hosts[0] if hosts else "",
            duration_ms=int((time.monotonic() - started) * 1000),
        )
        raise
    except TimeoutError as exc:
        stats.record_failure("timeout")
        raise MediaFetchError("timeout", 504, "media fetch timed out") from exc


async def resolve_remote_media(
    messages: Sequence[Mapping[str, Any]],
    *,
    config: MediaFetchConfig | None = None,
    stats: MediaFetchStats | None = None,
) -> tuple[tuple[dict[str, Any], ...], RemoteFetchReport]:
    """Replace remote media URLs with inline data URLs before parsing.

    The existing synchronous parser and sealed memfd transport stay unchanged;
    this function only normalizes the untrusted URL forms at the boundary.
    """

    config = config or MEDIA_FETCH_CONFIG
    stats = stats or MEDIA_FETCH_STATS
    normalized = [dict(message) for message in messages]
    pending: list[tuple[int, int, str, str, str]] = []
    for message_index, message in enumerate(normalized):
        content = message.get("content")
        if not isinstance(content, list):
            continue
        for part_index, part in enumerate(content):
            if not isinstance(part, dict):
                continue
            modality = _PART_MODALITIES.get(str(part.get("type")))
            if modality is None:
                continue
            source = part.get(str(part.get("type")))
            if isinstance(source, str):
                url = source
            elif isinstance(source, dict):
                url = source.get("url")
            else:
                continue
            if not isinstance(url, str) or url.startswith("data:"):
                continue
            try:
                scheme = urllib.parse.urlsplit(url).scheme
            except ValueError:
                continue  # the synchronous parser reports the malformed URL
            if scheme:
                pending.append(
                    (message_index, part_index, str(part["type"]), url, modality)
                )

    if not pending:
        return tuple(normalized), RemoteFetchReport(0, 0, ())

    semaphore = asyncio.Semaphore(config.max_concurrent)
    hosts: set[str] = set()
    bytes_downloaded = 0

    async def fetch_one(entry: tuple[int, int, str, str, str]) -> None:
        nonlocal bytes_downloaded
        message_index, part_index, key, url, modality = entry
        async with semaphore:
            mime_type, payload = await fetch_remote_media(
                url, modality=modality, config=config, stats=stats
            )
        data_url = (
            f"data:{mime_type};base64,{base64.b64encode(payload).decode('ascii')}"
        )
        source = normalized[message_index]["content"][part_index][key]
        if isinstance(source, str):
            normalized[message_index]["content"][part_index][key] = {
                "url": data_url
            }
        else:
            updated = dict(source)
            updated["url"] = data_url
            normalized[message_index]["content"][part_index][key] = updated
        bytes_downloaded += len(payload)
        hosts.add(urllib.parse.urlsplit(url).hostname or "")

    # gather(return_exceptions=True) keeps the original exception type instead
    # of wrapping it in an ExceptionGroup, so the route can map it to a stable
    # 400/502/504. Siblings are cancelled and awaited before re-raising.
    tasks = [asyncio.create_task(fetch_one(entry)) for entry in pending]
    outcomes = await asyncio.gather(*tasks, return_exceptions=True)
    for outcome in outcomes:
        if isinstance(outcome, BaseException):
            for task in tasks:
                if not task.done():
                    task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            raise outcome
    return (
        tuple(normalized),
        RemoteFetchReport(len(pending), bytes_downloaded, tuple(sorted(hosts))),
    )


__all__ = [
    "MEDIA_FETCH_CONFIG",
    "MEDIA_FETCH_STATS",
    "MediaFetchConfig",
    "MediaFetchError",
    "MediaFetchStats",
    "RemoteFetchReport",
    "fetch_remote_media",
    "resolve_remote_media",
]
