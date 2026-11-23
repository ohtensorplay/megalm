<div align="center">
  <h1>🚀 MegaLM</h1>
  <p><strong>A high-performance LLM inference engine built on <a href="https://github.com/lexing-2026/TensorPlay">TensorPlay</a></strong></p>
  <p>
    <a href="#about-megalm">About</a> ·
    <a href="#built-on-tensorplay">Built on TensorPlay</a> ·
    <a href="#key-features">Key Features</a> ·
    <a href="#quick-start">Quick Start</a> ·
    <a href="#performance">Performance</a> ·
    <a href="#documentation">Documentation</a>
  </p>
  <p><em>Powered by <a href="https://tensorplay.cn">TensorPlay</a> — a learner-first, DIY-ready tensor &amp; kernel framework.</em></p>
  <p>
    <a href="./README.md"><strong>EN</strong></a> ·
    <a href="./README.zh.md"><strong>中文</strong></a>
  </p>
</div>

---

## About MegaLM

MegaLM is the LLM inference and serving engine of the **[TensorPlay](https://github.com/lexing-2026/TensorPlay) ecosystem**, targeting mainstream models such as Qwen3 / Qwen3.5 and designed for low-latency, high-throughput online serving.

If [TensorPlay](https://github.com/lexing-2026/TensorPlay) provides the low-level foundation for tensors, operators, and kernels, then MegaLM is the **inference and serving layer** built on top of it — applying TensorPlay's tensor runtime and kernel system to LLM inference, with a complete set of model capabilities, scheduling strategies, and serving protocols.

- ⚡ **Fast**: Custom fused operators and continuous batching, competitive for both single-request and high-concurrency workloads.
- 🪶 **Light**: Hot paths do not depend on Transformers; streaming weight loading for faster startup.
- 🧩 **Complete**: Broad open-model coverage, multimodal inputs, tool calling, structured output, and observability.
- 🔌 **Compatible**: OpenAI-compatible API that plugs into existing ecosystems and clients.

> Requirements: Linux · Python ≥ 3.12 · NVIDIA Ampere or newer GPU.

---

## 🧱 Built on TensorPlay

MegaLM and TensorPlay form a top-to-bottom technology stack:

| Layer | Project | Responsibility |
| --- | --- | --- |
| **Inference & serving** | **MegaLM** | Model execution, continuous batching, KV cache, sampling, serving gateway, and protocols |
| **Tensor & kernel foundation** | [TensorPlay](https://github.com/lexing-2026/TensorPlay) | Tensor runtime, operator dispatch, kernel system, serialization, and device abstraction |
| **Compiler backends** (optional) | [tensorplay-tensorrt](https://github.com/lexing-2026/tensorplay-tensorrt) and others | Compilation and backend acceleration for TensorPlay |

MegaLM's compute kernels and weight loading are all built on TensorPlay — from tensor operations and operator dispatch to checkpoint serialization, the entire execution stack shares one set of underlying semantics. This means:

- **One foundation, many uses**: the same TensorPlay capabilities serve both framework learning and production inference.
- **End-to-end control**: from low-level kernels to serving protocols, everything is connected within one ecosystem, reducing heterogeneous dependencies.
- **Extensible**: TensorPlay's backend and hardware ecosystem can continuously bring new acceleration possibilities to MegaLM.

---

## ✨ Key Features

### High-Performance Inference
- **Continuous batching**: decode-first, chunked prefill, and mixed prefill/decode batches, balancing time-to-first-token and overall throughput.
- **Smart caching**: real GPU paged KV and page-aligned Radix prefix cache, zero-copy reuse of shared prefixes, and automatic eviction of cold prefixes under memory pressure.
- **Hierarchical KV**: optional pinned host and filesystem tiers with content-addressed pages, async DMA, staged promotions, and scheduler-safe reuse.
- **Custom operators**: fused RMSNorm, paged attention, grouped tensor-core MoE, and persistent routing, hand-optimized on TensorPlay.
- **CUDA Graph**: bucket capture/replay during decode to eliminate kernel launch overhead.

### Broad Model Support
- Covers **Qwen3 / Qwen3.5** (Dense / MoE), **Qwen3-VL**, Llama, Mistral, Mixtral, Qwen2, Yi, DeepSeek, Phi3, InternLM, Granite, ERNIE, and more.
- Native support for **Qwen2.5-Omni** multimodal Thinker and the **Inkling-2** recurrent architecture.

### Complete Serving Protocols
- **OpenAI Chat Completions**: ready-to-use streaming inference endpoint.
- **Anthropic Messages** and **OpenAI Responses**: native protocol endpoints with response store and tool-call events.
- **Function Calling**: incremental tool-call parsing across model families with streaming argument output.
- **Decision model endpoints**: `/v1/systemone` and `/v1/decisions` — use any chat model as a decision model without dedicated weights.

### Production Ready
- **Structured output**: engine-level constrained decoding for `json_object` / `json_schema`, guaranteeing valid JSON.
- **Multimodal media**: image, video, and audio inputs with SSRF-protected remote URL fetching.
- **Observability**: Prometheus metrics endpoint, structured logs, and distributed tracing.
- **Context management**: automatic history trimming for stable long conversations.

---

## 🚀 Quick Start

> **Platform**: Linux with an NVIDIA Ampere or newer GPU, Python ≥ 3.12. Other platforms are not currently supported.

### 1. Environment Setup

```bash
python3.12 -m venv .venv
source .venv/bin/activate
```

### 2. Installation

```bash
python -m pip install tensorplay
python -m pip install -e .
```

### 3. Offline Inference

```bash
python -m megalm \
  --model <your-model> \
  --prompt "Introduce yourself in one sentence." \
  --max-new-tokens 16
```

### 4. Launch the API Server

```bash
megalm-serve <your-model> \
  --served-model-name <model-name> \
  --host 127.0.0.1 \
  --port 8000
```

### 5. Verify and Send a Request

Check that the server is ready:

```bash
curl http://127.0.0.1:8000/health
# {"status":"ok"}
```

Send a chat completion request (use the same name as `--served-model-name`):

```bash
curl http://127.0.0.1:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "<model-name>",
    "messages": [{"role": "user", "content": "Introduce yourself in one sentence."}],
    "max_tokens": 128,
    "stream": false
  }'
```

Set `"stream": true` to receive Server-Sent Events. Model and runtime information is available at `/v1/models` and `/api/inference/status`; Prometheus metrics at `/metrics`.

---

## 📊 Performance

### OXR-1.0 serving case study — reference measurement

The chart below is a **reference snapshot** from a single-case serving comparison, not a general-purpose benchmark. It uses the OXR-1.0 vision-language model on one RTX 3090, with native MegaLM measured against official vLLM 0.25.

| Test setup | Configuration |
| --- | --- |
| Model | OXR-1.0 (vision-language) |
| Hardware | 1 × RTX 3090 (BF16) |
| Input | Fixed image + prompt |
| Baseline | Official vLLM 0.25 |
| Scenarios | Decode throughput, TTFT, concurrent aggregate throughput, E2E latency |

![OXR-1.0 serving: native MegaLM vs official vLLM 0.25](docs/oxr_vllm_comparison.png)

Representative data points from the four panels:

| Metric | MegaLM (native) | vLLM 0.25 |
| --- | --- | --- |
| TTFT | **5.3 ms** | 8.8 ms |
| Decode (32-token output) | **482 tok/s** | 445 tok/s |
| Concurrent 8, aggregate | **2663 tok/s** | 2299 tok/s |
| E2E (128-token output) | **272 ms** | 294 ms |

In this configuration, MegaLM measured ahead of the baseline in all four scenarios, and greedy output matched the reference token-for-token.

> **Note**: These are single-run reference numbers for one model and one hardware setup. Results vary with model, hardware, driver, load, and server configuration, and should not be read as a universal performance claim. For the full test configuration, see [`docs/oxr_vllm_comparison.pdf`](docs/oxr_vllm_comparison.pdf).

---

## 📚 Documentation

- [TensorPlay website](https://tensorplay.cn) · [TensorPlay source](https://github.com/lexing-2026/TensorPlay)
- [`docs/observability.md`](docs/observability.md): metrics and dashboard setup.
- [`docs/oxr_vllm_comparison.pdf`](docs/oxr_vllm_comparison.pdf): full benchmark configuration.

---

<div align="center">
  <p>MegaLM is part of the TensorPlay ecosystem. If it's useful to you, feel free to Star and share feedback.</p>
</div>
