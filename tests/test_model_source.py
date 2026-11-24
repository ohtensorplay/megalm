"""Tests for Hub-backed model resolution (no network calls)."""

from __future__ import annotations

import sys
from pathlib import Path
from types import ModuleType

import pytest

from megalm import download, model_source
from megalm.model_source import (
    ModelSourceError,
    parse_model_ref,
    resolve_model_ref,
)


@pytest.fixture
def backends(tmp_path, monkeypatch):
    """Replace all SDK backends with local no-op snapshot functions."""
    calls: list[tuple[str, object, object]] = []

    def make(name: str):
        def snapshot(ref, options):
            calls.append((name, ref, options))
            location = tmp_path / name
            location.mkdir(exist_ok=True)
            return str(location)

        return snapshot

    replacement = {name: make(name) for name in model_source.SOURCE_ORDER}
    monkeypatch.setattr(model_source, "_BACKENDS", replacement)
    return calls


def test_local_path_roundtrip(tmp_path, backends) -> None:
    assert resolve_model_ref(tmp_path) == tmp_path
    assert backends == []


def test_missing_local_path_is_rejected(tmp_path, backends) -> None:
    missing = tmp_path / "not-there"
    with pytest.raises(ModelSourceError, match="not a directory"):
        resolve_model_ref(missing)
    assert backends == []


@pytest.mark.parametrize(
    ("reference", "source", "expected"),
    [
        ("hf:Qwen/Qwen3-8B", "auto", ("hf", "Qwen/Qwen3-8B", None, None)),
        (
            "huggingface://Qwen/Qwen3-8B@v1",
            "auto",
            ("hf", "Qwen/Qwen3-8B", "v1", None),
        ),
        ("ms:Qwen/Qwen3-8B", "auto", ("ms", "Qwen/Qwen3-8B", None, None)),
        (
            "modelscope://Qwen/Qwen3-8B@master",
            "auto",
            ("ms", "Qwen/Qwen3-8B", "master", None),
        ),
        (
            "mega://models/acme/llm@main/path",
            "auto",
            ("mega", "acme/llm", "main", "model"),
        ),
        (
            "megahub:/datasets/acme/data",
            "auto",
            ("mega", "acme/data", None, "dataset"),
        ),
        ("Qwen/Qwen3-8B", "modelscope", ("ms", "Qwen/Qwen3-8B", None, None)),
    ],
)
def test_scheme_and_source_parsing(reference, source, expected) -> None:
    ref = parse_model_ref(reference, source=source)
    assert (
        ref.source,
        ref.repo_id,
        ref.revision,
        ref.repo_type,
    ) == expected


@pytest.mark.parametrize("bad", ["", "hf://only-namespace", "hf://a/b/c/d", "ms:../etc"])
def test_invalid_references(bad) -> None:
    with pytest.raises(ModelSourceError):
        parse_model_ref(bad)


def test_uri_scheme_overrides_source_flag(backends) -> None:
    path = resolve_model_ref("hf:Qwen/Qwen3-8B", source="ms")
    assert path.name == "hf"
    assert [name for name, _, _ in backends] == ["hf"]
    assert backends[0][1].source == "hf"


def test_auto_falls_back_after_failure(monkeypatch, backends) -> None:
    tried: list[str] = []

    def broken(ref, options):
        tried.append("mega")
        raise RuntimeError("hub offline")

    monkeypatch.setitem(model_source._BACKENDS, "mega", broken)
    path = resolve_model_ref("acme/llm", source="auto")
    assert path.name == "hf"
    assert tried == ["mega"]
    assert [name for name, _, _ in backends] == ["hf"]


def test_auto_reports_every_failure(monkeypatch) -> None:
    def broken(name):
        def snapshot(ref, options):
            raise RuntimeError(f"{name} failed")

        return snapshot

    monkeypatch.setattr(
        model_source,
        "_BACKENDS",
        {name: broken(name) for name in model_source.SOURCE_ORDER},
    )
    with pytest.raises(ModelSourceError, match="mega failed") as exc:
        resolve_model_ref("acme/llm")
    message = str(exc.value)
    assert "hf failed" in message and "ms failed" in message


def test_options_are_forwarded_to_sdk(backends) -> None:
    resolve_model_ref(
        "hf:Qwen/Qwen3-8B",
        revision="main",
        cache_dir="/cache",
        local_dir="/local",
        include=["*.safetensors"],
        exclude=["*.bin"],
        offline=True,
        endpoint="https://hf.internal",
        token="secret",
    )
    _, ref, options = backends[0]
    assert ref.repo_id == "Qwen/Qwen3-8B"
    assert options.revision == "main"
    assert str(options.cache_dir) == "/cache"
    assert str(options.local_dir) == "/local"
    assert options.include == ("*.safetensors",)
    assert options.exclude == ("*.bin",)
    assert options.offline is True
    assert options.endpoint == "https://hf.internal"
    assert options.token == "secret"


def test_download_cli_uses_official_sdk_and_prints_path(tmp_path, monkeypatch, capsys) -> None:
    seen: dict[str, object] = {}

    def fake(ref, options):
        seen["repo_id"] = ref.repo_id
        seen["source"] = ref.source
        location = tmp_path / "snapshot"
        location.mkdir(exist_ok=True)
        return str(location)

    monkeypatch.setitem(model_source._BACKENDS, "hf", fake)
    assert download.main(["hf:Qwen/Qwen3-8B", "--quiet"]) == 0
    assert capsys.readouterr().out == ""
    assert seen == {"repo_id": "Qwen/Qwen3-8B", "source": "hf"}

    assert download.main(["hf:Qwen/Qwen3-8B"]) == 0
    assert capsys.readouterr().out.strip() == str(tmp_path / "snapshot")


def test_download_cli_default_repo() -> None:
    args = download.build_parser().parse_args([])
    assert args.repo_id == "Nanbeige/Nanbeige4.2-3B"
    assert args.source == "auto"


def test_download_cli_reports_resolution_error(monkeypatch, capsys) -> None:
    def broken(ref, options):
        raise RuntimeError("auth required")

    monkeypatch.setattr(
        model_source,
        "_BACKENDS",
        {name: broken for name in model_source.SOURCE_ORDER},
    )
    assert download.main(["acme/llm"]) == 2
    assert "auth required" in capsys.readouterr().err


@pytest.mark.parametrize(
    ("backend", "module_name"),
    [
        (model_source._download_mega, "megatensors"),
        (model_source._download_hf, "huggingface_hub"),
        (model_source._download_ms, "modelscope"),
    ],
)
def test_backends_call_official_sdk(backend, module_name, monkeypatch, tmp_path) -> None:
    recorded: dict[str, object] = {}
    sdk = ModuleType(module_name)

    def snapshot_download(**kwargs):
        recorded.update(kwargs)
        return str(tmp_path)

    sdk.snapshot_download = snapshot_download  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, module_name, sdk)

    ref = model_source.ModelRef(source="hf", repo_id="acme/llm", repo_type="model")
    options = model_source.ResolveOptions(
        revision="v1", include=["*.safetensors"], exclude=["*.bin"]
    )
    assert backend(ref, options) == str(tmp_path)
    assert recorded["repo_id"] == "acme/llm"
    assert recorded["revision"] == "v1"
    assert recorded["allow_patterns"] == ["*.safetensors"]
    assert recorded["ignore_patterns"] == ["*.bin"]
    if module_name == "megatensors":
        assert recorded["repo_type"] == "model"


def test_modelscope_legacy_pattern_names(monkeypatch, tmp_path) -> None:
    sdk = ModuleType("modelscope")
    recorded: dict[str, object] = {}

    def snapshot_download(
        repo_id,
        *,
        allow_file_pattern=None,
        ignore_file_pattern=None,
        **kwargs,
    ):
        recorded.update(kwargs)
        recorded["allow_file_pattern"] = allow_file_pattern
        recorded["ignore_file_pattern"] = ignore_file_pattern
        return str(tmp_path)

    sdk.snapshot_download = snapshot_download  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "modelscope", sdk)

    ref = model_source.ModelRef(source="ms", repo_id="acme/llm")
    options = model_source.ResolveOptions(
        include=["*.safetensors"], exclude=["*.bin"], offline=True
    )
    model_source._download_ms(ref, options)
    assert recorded["allow_file_pattern"] == ["*.safetensors"]
    assert recorded["ignore_file_pattern"] == ["*.bin"]
    assert recorded["local_files_only"] is True


def test_missing_sdk_error_mentions_install(monkeypatch) -> None:
    def unavailable(ref, options):
        raise ModuleNotFoundError("No module named 'megatensors'")

    monkeypatch.setattr(model_source, "_BACKENDS", {"mega": unavailable})
    with pytest.raises(ModelSourceError, match="pip install megatensors"):
        resolve_model_ref("mega:acme/llm")
