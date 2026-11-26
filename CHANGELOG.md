# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- `megalm-bench`, a Rust serving benchmark for OpenAI-compatible endpoints:
  shared seeded workloads (synthetic / JSONL / ShareGPT), closed- and open-loop
  arrival control, TTFT/ITL/TPOT/E2EL percentiles, SLO goodput, peak gauges,
  error taxonomy and automatic provenance (commit, source digests, workload
  digest, versions, hardware). Multiple engines run sequentially with the same
  request set; `report` renders comparison tables (`docs/benchmarks/serving.md`).
  Trial seeds and queue-wait limits are recorded per trial; latency stats include
  population std; peak gauges ignore failed requests; markdown reports carry
  explicit validity warnings.
- Native Gemma 4 executor: text decoder with heterogeneous KV geometry, YOCO
  KV sharing, per-layer embeddings, dual RoPE, dense+MoE feed-forward, final
  logit soft-capping, checkpoint-native MTP assistant, vision/audio towers and
  encoder-free unified embedders. Dense and MoE logits match HuggingFace on
  synthetic checkpoints; `google/gemma-4-E2B-it` loads and runs end to end.
  Shared MTP/embedding scaffolding lives in `models/components/` and is reused
  by Qwen 3.5 (`docs/models/gemma4.md`).
- End-to-end Gemma 4 image/audio serving: Transformers-free media frontends,
  tokenizer placeholder expansion with Gemma-specific boundary tokens, 1-D
  position binding, tower weight loading and embedding scatter into prompt rows.
  Vision per-linear clipping metadata is loaded and applied, reaching cosine
  1.0 / ~0.0018 relative error vs HuggingFace on `gemma-4-E2B-it`.
- Gemma 4 fast attention: sliding-window masking added to the shared Triton
  prefill/decode/GQA-decode kernels (default-off, so other models are unchanged),
  removing the Python-loop fallback on all 35 layers. Windowed decode/split
  kernels now skip masked blocks entirely. Prefill fell 297 → 39 ms.
- Gemma 4 fused decode path: one packed QKV projection and a generalized fused
  Q/K/V-norm + RoPE + cache-write kernel cover both RoPE flavours and shared
  layers, reusing the shared Triton kernel with constexpr flags. Single-token
  decode GEMVs, a fused gate/up GEMV + GELU-tanh, and fused residual/scale
  kernels remove most per-layer launches. On `gemma-4-E2B-it` batch-1 throughput
  rose 81 → ~135 tok/s (at vLLM's TRITON_ATTN level) and batch-8 reached
  ~756 tok/s, with prefill ~1.2× faster and startup ~10× faster
  (`docs/benchmarks/gemma4-vllm.md`).
- Production embedding-model serving: `POST /v1/embeddings` with a dedicated
  non-generative worker task, native BGE/BERT encoder, CLS/MEAN/LAST pooling,
  request coalescing, OpenAI `float`/`base64` encodings, and parity tests against
  HuggingFace. On `BAAI/bge-base-en-v1.5`, 1.26×–1.61× the pinned vLLM baseline
  across four throughput scenarios on RTX 3090 (`docs/models/embedding.md`).
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
