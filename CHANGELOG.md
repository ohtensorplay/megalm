# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- Hub-backed model loading: `mega://`, `hf://` and `ms://` references (plus bare
  repo ids with `--source`) resolved through the official `megatensors`,
  `huggingface_hub` and `modelscope` SDKs, shared by the engine, gateway and
  `megalm-download` CLIs.
- GitHub community infrastructure: issue and pull-request templates, code
  ownership, Dependabot, code of conduct, security policy and contributing
  guide.
- GitHub Actions workflows for linting, CPU unit tests and wheel/sdist
  publishing via maturin.

## [0.2.0] - 2026-11-23

### Added

- Qwen3 / Qwen3.5, Qwen2.5-Omni, Inkling-2, OXR and Nanbeige4.2 model support.
- OpenAI-compatible Chat Completions and Responses APIs with tool calling,
  structured output, multimodal input and multi-LoRA serving.
- Radix and hybrid KV caching, speculative decoding, KV-cache quantization and
  CUDA-graph execution on the TensorPlay runtime.
- Python/Rust (`PyO3` + maturin) packaging with `megalm-serve` and
  `megalm-download` entry points.

[Unreleased]: https://github.com/ohtensorplay/megalm/compare/v0.2.0...HEAD
[0.2.0]: https://github.com/ohtensorplay/megalm/releases/tag/v0.2.0
