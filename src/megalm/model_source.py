"""Resolve model references through the official Hub SDKs.

A model reference is either a local directory or a Hub repository id. Remote
references use one of these forms::

    mega://models/namespace/repository@revision
    megahub://models/namespace/repository
    hf://namespace/repository@revision
    huggingface://namespace/repository
    ms://namespace/repository@revision
    modelscope://namespace/repository

A bare ``namespace/repository`` is resolved via ``--source`` (``auto`` by
default, which tries MegaHub, Hugging Face and ModelScope in that order).
Nothing in this module is on the inference hot path: downloads happen once at
startup, and the returned path is an ordinary local checkpoint directory.

Each Hub is accessed through its own official SDK, so authentication, caching,
resumable transfers, revisions and file filters keep the upstream semantics:

* MegaHub -> ``megatensors.snapshot_download`` (``MEGA_TOKEN`` / ``MEGA_ENDPOINT``)
* Hugging Face -> ``huggingface_hub.snapshot_download`` (``HF_TOKEN`` / ``HF_ENDPOINT``)
* ModelScope -> ``modelscope.snapshot_download`` (``MODELSCOPE_API_TOKEN`` / ``MODELSCOPE_ENDPOINT``)
"""

from __future__ import annotations

import argparse
import inspect
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Sequence

_LOGGER = logging.getLogger(__name__)

SOURCE_ORDER = ("mega", "hf", "ms")
SOURCE_ALIASES = {
    "auto": "auto",
    "mega": "mega",
    "megahub": "mega",
    "hf": "hf",
    "huggingface": "hf",
    "ms": "ms",
    "modelscope": "ms",
}
SCHEME_SOURCES = {
    "megahub://": "mega",
    "mega://": "mega",
    "megahub:": "mega",
    "mega:": "mega",
    "huggingface://": "hf",
    "hf://": "hf",
    "huggingface:": "hf",
    "hf:": "hf",
    "modelscope://": "ms",
    "ms://": "ms",
    "modelscope:": "ms",
    "ms:": "ms",
}
_REPO_TYPES = {
    "models": "model",
    "datasets": "dataset",
    "spaces": "space",
    "mcps": "mcp",
}
_SDK_HINTS = {
    "mega": "pip install megatensors",
    "hf": "pip install huggingface_hub",
    "ms": "pip install modelscope",
}
_REPO_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*(?:/[A-Za-z0-9][A-Za-z0-9._-]*)?$")


class ModelSourceError(RuntimeError):
    """Raised when a model reference cannot be resolved to a local directory."""


@dataclass(frozen=True, slots=True)
class ModelRef:
    source: str
    repo_id: str
    revision: str | None = None
    repo_type: str | None = None


@dataclass(frozen=True, slots=True)
class ResolveOptions:
    revision: str | None = None
    cache_dir: str | Path | None = None
    local_dir: str | Path | None = None
    include: tuple[str, ...] = ()
    exclude: tuple[str, ...] = ()
    offline: bool = False
    endpoint: str | None = None
    token: str | None = None


def normalize_source(value: str) -> str:
    source = SOURCE_ALIASES.get(value.strip().lower())
    if source is None:
        choices = ", ".join(sorted(SOURCE_ALIASES))
        raise ModelSourceError(f"unknown model source {value!r}; choose from {choices}")
    return source


def parse_model_ref(model: str | Path, *, source: str = "auto") -> Path | ModelRef:
    """Split a model reference into a local path or a Hub repository reference."""
    text = str(model).strip()
    if not text:
        raise ModelSourceError("empty model reference")

    explicit_source: str | None = None
    rest = text
    for scheme in sorted(SCHEME_SOURCES, key=len, reverse=True):
        if text.startswith(scheme):
            explicit_source = SCHEME_SOURCES[scheme]
            rest = text[len(scheme) :]
            break

    if explicit_source is None:
        path = Path(text).expanduser()
        looks_local = (
            path.exists()
            or text.startswith(("/", ".", "~"))
            or (len(text) >= 2 and text[1] == ":")
        )
        if looks_local:
            return path

    source = normalize_source(explicit_source or source)
    segments = [part for part in rest.split("/") if part]
    repo_type: str | None = None
    if explicit_source == "mega" and segments and segments[0] in _REPO_TYPES:
        repo_type = _REPO_TYPES[segments[0]]
        segments = segments[1:]

    if len(segments) < 2 or len(segments) > 3:
        raise ModelSourceError(
            f"invalid repository reference {text!r}; expected 'namespace/name'"
        )
    revision: str | None = None
    for index in range(2):
        if "@" in segments[index]:
            repo_name, revision = segments[index].split("@", 1)
            if not repo_name or not revision:
                raise ModelSourceError(f"invalid repository reference: {text!r}")
            segments[index] = repo_name
            break

    repo_id = "/".join(segments[:2])
    if not _REPO_ID_RE.fullmatch(repo_id):
        raise ModelSourceError(
            f"invalid repository id {repo_id!r}; expected 'namespace/name'"
        )
    return ModelRef(
        source=source,
        repo_id=repo_id,
        revision=revision,
        repo_type=repo_type,
    )


def _common_kwargs(ref: ModelRef, options: ResolveOptions) -> dict[str, Any]:
    kwargs: dict[str, Any] = {"repo_id": ref.repo_id}
    revision = options.revision or ref.revision
    if revision:
        kwargs["revision"] = revision
    if options.cache_dir:
        kwargs["cache_dir"] = str(options.cache_dir)
    if options.local_dir:
        kwargs["local_dir"] = str(options.local_dir)
    if options.include:
        kwargs["allow_patterns"] = list(options.include)
    if options.exclude:
        kwargs["ignore_patterns"] = list(options.exclude)
    if options.offline:
        kwargs["local_files_only"] = True
    if options.endpoint:
        kwargs["endpoint"] = options.endpoint
    if options.token:
        kwargs["token"] = options.token
    return kwargs


def _download_mega(ref: ModelRef, options: ResolveOptions) -> str:
    from megatensors import snapshot_download

    kwargs = _common_kwargs(ref, options)
    if ref.repo_type:
        kwargs["repo_type"] = ref.repo_type
    return str(snapshot_download(**kwargs))


def _download_hf(ref: ModelRef, options: ResolveOptions) -> str:
    from huggingface_hub import snapshot_download

    return str(snapshot_download(**_common_kwargs(ref, options)))


def _download_ms(ref: ModelRef, options: ResolveOptions) -> str:
    from modelscope import snapshot_download

    kwargs = _common_kwargs(ref, options)
    # ModelScope renamed allow/ignore patterns in 1.x; keep older clients working.
    parameters = inspect.signature(snapshot_download).parameters
    if "allow_patterns" not in parameters and "allow_file_pattern" in parameters:
        if "allow_patterns" in kwargs:
            kwargs["allow_file_pattern"] = kwargs.pop("allow_patterns")
    if "ignore_patterns" not in parameters and "ignore_file_pattern" in parameters:
        if "ignore_patterns" in kwargs:
            kwargs["ignore_file_pattern"] = kwargs.pop("ignore_patterns")
    return str(snapshot_download(**kwargs))


_BACKENDS: dict[str, Callable[[ModelRef, ResolveOptions], str]] = {
    "mega": _download_mega,
    "hf": _download_hf,
    "ms": _download_ms,
}


def resolve_model_ref(
    model: str | Path,
    *,
    source: str = "auto",
    revision: str | None = None,
    cache_dir: str | Path | None = None,
    local_dir: str | Path | None = None,
    include: Sequence[str] = (),
    exclude: Sequence[str] = (),
    offline: bool = False,
    endpoint: str | None = None,
    token: str | None = None,
) -> Path:
    """Return a local checkpoint directory for ``model``, downloading if needed."""
    ref = parse_model_ref(model, source=source)
    if isinstance(ref, Path):
        if not ref.is_dir():
            raise ModelSourceError(f"model path is not a directory: {ref}")
        return ref

    options = ResolveOptions(
        revision=revision,
        cache_dir=cache_dir,
        local_dir=local_dir,
        include=tuple(include),
        exclude=tuple(exclude),
        offline=offline,
        endpoint=endpoint,
        token=token,
    )
    candidates = SOURCE_ORDER if ref.source == "auto" else (ref.source,)
    failures: list[str] = []
    for name in candidates:
        try:
            location = Path(_BACKENDS[name](ref, options)).expanduser()
        except ImportError as exc:
            failures.append(f"{name}: SDK unavailable ({_SDK_HINTS[name]}): {exc}")
            continue
        except Exception as exc:
            failures.append(f"{name}: {type(exc).__name__}: {exc}")
            continue
        if not location.is_dir():
            failures.append(f"{name}: SDK returned a non-directory: {location}")
            continue
        _LOGGER.info(
            "model_snapshot_resolved source=%s repo=%s path=%s",
            name,
            ref.repo_id,
            location,
        )
        return location

    detail = "\n  - ".join(failures)
    raise ModelSourceError(
        f"cannot resolve model {str(model)!r} (source={source}):\n  - {detail}"
    )


def add_model_source_args(parser: argparse.ArgumentParser) -> None:
    """Register the shared Hub download flags on a CLI parser."""
    group = parser.add_argument_group(
        "model source (MegaHub / Hugging Face / ModelScope)"
    )
    group.add_argument(
        "--source",
        choices=tuple(SOURCE_ALIASES),
        default="auto",
        help=(
            "Hub used for a bare repo id; 'auto' tries mega, hf, ms in order. "
            "Explicit mega://, hf:// and ms:// references override it."
        ),
    )
    group.add_argument("--revision", default=None, help="Hub revision (branch, tag or commit)")
    group.add_argument("--cache-dir", default=None, help="override the SDK cache directory")
    group.add_argument("--local-dir", default=None, help="download straight into this directory")
    group.add_argument(
        "--include",
        action="append",
        default=[],
        metavar="GLOB",
        help="only download matching files (repeatable); SDK allow_patterns",
    )
    group.add_argument(
        "--exclude",
        action="append",
        default=[],
        metavar="GLOB",
        help="skip matching files (repeatable); SDK ignore_patterns",
    )
    group.add_argument(
        "--offline",
        action="store_true",
        help="only use files already present in the SDK cache (local_files_only)",
    )
    group.add_argument(
        "--endpoint",
        default=None,
        help="override the Hub endpoint (env: MEGA_ENDPOINT / HF_ENDPOINT / MODELSCOPE_ENDPOINT)",
    )
    group.add_argument(
        "--token",
        default=None,
        help="Hub token (prefer MEGA_TOKEN / HF_TOKEN / MODELSCOPE_API_TOKEN)",
    )


def model_path_from_args(model: str | Path, args: argparse.Namespace) -> Path:
    """Resolve ``model`` using flags registered by :func:`add_model_source_args`."""
    return resolve_model_ref(
        model,
        source=args.source,
        revision=args.revision,
        cache_dir=args.cache_dir,
        local_dir=args.local_dir,
        include=args.include,
        exclude=args.exclude,
        offline=args.offline,
        endpoint=args.endpoint,
        token=args.token,
    )


__all__ = [
    "ModelRef",
    "ModelSourceError",
    "ResolveOptions",
    "SOURCE_ORDER",
    "add_model_source_args",
    "model_path_from_args",
    "normalize_source",
    "parse_model_ref",
    "resolve_model_ref",
]
