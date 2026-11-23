<div align="center">
  <img src="docs/assets/github-banner-1280x320.png" alt="MegaLM — Inference in motion. 基于 TensorPlay 的推理引擎" width="1000">

  <p><strong>基于 <a href="https://github.com/lexing-2026/TensorPlay">TensorPlay</a> 构建的高性能大模型推理引擎</strong></p>

  <p>
    <a href="#关于-megalm">关于</a> ·
    <a href="#-动态">动态</a> ·
    <a href="#-基于-tensorplay">技术基石</a> ·
    <a href="#-核心能力">核心能力</a> ·
    <a href="#-快速开始">快速开始</a> ·
    <a href="#-api-端点">API 端点</a> ·
    <a href="#-性能表现">性能表现</a> ·
    <a href="#-文档">文档</a>
  </p>
  <p>
    <a href="./README.md"><strong>EN</strong></a> ·
    <a href="./README.zh.md"><strong>中文</strong></a>
  </p>
  <p>
    <img src="https://img.shields.io/badge/python-%E2%89%A53.12-23347A?style=flat-square&logo=python&logoColor=white" alt="Python 3.12+">
    <img src="https://img.shields.io/badge/platform-Linux-23347A?style=flat-square" alt="Linux">
    <img src="https://img.shields.io/badge/CUDA-Ampere%2B-76B900?style=flat-square&logo=nvidia&logoColor=white" alt="NVIDIA Ampere+">
    <a href="https://github.com/ohtensorplay/megalm/stargazers"><img src="https://img.shields.io/github/stars/ohtensorplay/megalm?style=flat-square&label=stars" alt="stars"></a>
  </p>
</div>

---

## 📰 动态

- **[2026/11] 原生支持 Inkling-2** — MegaLM 现已端到端运行 `thinkingmachines/Inkling-2-Mini`：26.4B 总参数 / 3.4B 激活的循环架构模型，具备 24 层混合 local/global 注意力、48 次 loop 执行，以及 NVFP4/MXFP8 专家权重常驻解码；热路径不依赖 Transformers。
- **[2026/11] 原生适配 OXR-1.0 视觉语言模型** — LLM 与视觉塔均在 TensorPlay 热路径执行，支持图像占位符展开与完整的视觉缓存语义。

---

## 关于 MegaLM

MegaLM 是 **TensorPlay 生态**中的大模型推理与服务引擎，面向 Qwen3 / Qwen3.5 等主流模型，专为低延迟、高吞吐的线上服务场景设计。

如果说 [TensorPlay](https://github.com/lexing-2026/TensorPlay) 提供了张量、算子与内核的底层基石，那么 MegaLM 就是建立在这块基石之上的**推理与服务层**：将 TensorPlay 的张量运行时与内核体系用于大模型推理，提供完整的模型能力、调度策略与服务协议。

- ⚡ **快**：自研融合算子与连续批处理，单请求与高并发场景均具备竞争力。
- 🪶 **轻**：热路径不依赖 Transformers，权重流式装载，启动更轻。
- 🧩 **全**：覆盖主流开源模型、多模态输入、工具调用、结构化输出与可观测性。
- 🔌 **兼容**：OpenAI-compatible API，可直接接入现有生态与客户端。

> 运行环境：Linux · Python ≥ 3.12 · NVIDIA Ampere 及以上 GPU。

---

## 🧱 基于 TensorPlay

MegaLM 与 TensorPlay 是一套自上而下的技术栈：

| 层次 | 项目 | 职责 |
| --- | --- | --- |
| **推理与服务层** | **MegaLM** | 模型执行、连续批处理、KV 缓存、采样、服务网关与协议 |
| **张量与内核基石** | [TensorPlay](https://github.com/lexing-2026/TensorPlay) | 张量运行时、算子分发、内核体系、序列化与设备抽象 |
| **编译后端**（可选） | [tensorplay-tensorrt](https://github.com/lexing-2026/tensorplay-tensorrt) 等 | 为 TensorPlay 提供编译与后端加速能力 |

MegaLM 的计算内核与权重装载均基于 TensorPlay 构建——从张量运算、算子分发到 checkpoint 的序列化读取，整个执行栈共享同一套底层语义。这意味着：

- **一套基石，多种用途**：同一份 TensorPlay 能力既服务于框架学习，也服务于生产推理。
- **端到端可控**：从底层内核到上层服务协议由同一生态贯通，减少异构依赖。
- **可扩展**：TensorPlay 的后端与硬件生态可持续为 MegaLM 带来新的加速可能。

---

## ✨ 核心能力

### 高性能推理
- **连续批处理**：Decode 优先、Chunked Prefill、Prefill/Decode 混合批处理，兼顾首 token 延迟与整体吞吐。
- **智能缓存**：真实 GPU paged KV 与页对齐 Radix 前缀缓存，共享前缀零拷贝复用，显存压力下自动回收冷前缀。
- **分层 KV**：可选的 pinned host 与文件系统层，内容寻址页、异步 DMA、分阶段提升与调度器安全复用。
- **自研算子**：fused RMSNorm、paged attention、grouped tensor-core MoE、持久化路由等关键路径基于 TensorPlay 手写优化。
- **CUDA Graph**：Decode 阶段 bucket capture/replay，消除内核 launch 开销。

### 广泛模型支持
- 覆盖 **Qwen3 / Qwen3.5**（Dense / MoE）、**Qwen3-VL**、Llama、Mistral、Mixtral、Qwen2、Yi、DeepSeek、Phi3、InternLM、Granite、ERNIE 等主流模型。
- 原生支持 **Qwen2.5-Omni** 多模态 Thinker，以及 **Inkling-2** 循环架构模型。

### 完整服务协议
- **OpenAI Chat Completions**：开箱即用的流式推理接口。
- **Anthropic Messages** 与 **OpenAI Responses**：原生协议端点，含 response store 与工具调用事件。
- **Function Calling**：多模型族的增量工具调用解析，流式参数输出。
- **决策模型端点**：`/v1/systemone` 与 `/v1/decisions`，无需专用权重即可将 chat 模型用作决策模型。

### 生产就绪
- **结构化输出**：`json_object` / `json_schema` 引擎级约束解码，保证输出合法 JSON。
- **多模态媒体**：图片、视频、音频输入，远程 URL 抓取具备 SSRF 防护。
- **可观测性**：Prometheus 指标端点、结构化日志、分布式追踪。
- **上下文管理**：自动历史裁剪，保障长对话稳定服务。

---

## 🚀 快速开始

> **平台要求**：Linux + NVIDIA Ampere 及以上 GPU，Python ≥ 3.12。暂不支持其他平台。

### 1. 准备环境

```bash
python3.12 -m venv .venv
source .venv/bin/activate
```

### 2. 安装

```bash
python -m pip install tensorplay
python -m pip install -e .
```

### 3. 离线推理

```bash
python -m megalm \
  --model <your-model> \
  --prompt "你好，请用一句话介绍你自己。" \
  --max-new-tokens 16
```

### 4. 启动 API 服务

```bash
megalm-serve <your-model> \
  --served-model-name <model-name> \
  --host 127.0.0.1 \
  --port 8000
```

### 5. 验证并发送请求

确认服务就绪：

```bash
curl http://127.0.0.1:8000/health
# {"status":"ok"}
```

发送 Chat Completions 请求（`model` 需与 `--served-model-name` 一致）：

```bash
curl http://127.0.0.1:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "<model-name>",
    "messages": [{"role": "user", "content": "你好，请用一句话介绍你自己。"}],
    "max_tokens": 128,
    "stream": false
  }'
```

设置 `"stream": true` 即以 Server-Sent Events 流式返回。模型与运行时信息见 `/v1/models`、`/api/inference/status`，Prometheus 指标见 `/metrics`。

---

## 🔌 API 端点

| 端点 | 说明 |
| --- | --- |
| `POST /v1/chat/completions` | OpenAI 风格 Chat，支持流式、工具调用与结构化输出 |
| `POST /v1/completions` | OpenAI 风格文本补全 |
| `POST /v1/messages` | Anthropic Messages API（另含 `/v1/messages/count_tokens`） |
| `POST /v1/responses` | OpenAI Responses API，内置 response store |
| `POST /v1/systemone` | TypeSafe 决策问题 |
| `POST /v1/decisions` | SGLang 风格决策问题 |
| `GET /v1/models` | 可用模型信息 |
| `GET /health` | 服务健康检查 |
| `GET /metrics` | Prometheus 指标 |

---

## 📊 性能表现

### OXR-1.0 服务案例 — 参考测量

下图是单次服务对比的**参考快照**，并非通用基准测试。测试使用 OXR-1.0 视觉语言模型、单卡 RTX 3090，对比原生 MegaLM 与官方 vLLM 0.25。

| 测试条件 | 配置 |
| --- | --- |
| 模型 | OXR-1.0（视觉语言） |
| 硬件 | 1 × RTX 3090（BF16） |
| 输入 | 固定的图像 + prompt |
| 对比基线 | 官方 vLLM 0.25 |
| 场景 | Decode 吞吐、TTFT、并发聚合吞吐、E2E 延迟 |

![OXR-1.0 服务：原生 MegaLM 对比官方 vLLM 0.25](docs/oxr_vllm_comparison.png)

下表为四个子图中的代表性数据点：

| 指标 | MegaLM（原生） | vLLM 0.25 |
| --- | --- | --- |
| 首 token 延迟（TTFT） | **5.3 ms** | 8.8 ms |
| Decode 吞吐（32 token 输出） | **482 tok/s** | 445 tok/s |
| 并发 8 聚合吞吐 | **2663 tok/s** | 2299 tok/s |
| 端到端延迟（128 token 输出） | **272 ms** | 294 ms |

在该配置下，四个场景的参考测量均优于基线，贪心输出与参考实现逐 token 一致。

> **注意**：以上为单一模型、单一硬件下的单次参考测量。结果会随模型、硬件、驱动、负载与服务配置变化，不应视为通用性能结论。完整测试配置见 [`docs/oxr_vllm_comparison.pdf`](docs/oxr_vllm_comparison.pdf)。

---

## 📚 文档

- [TensorPlay 官方网站](https://tensorplay.cn) · [TensorPlay 源码](https://github.com/lexing-2026/TensorPlay) · [社区论坛](https://forum.tensorplay.org)
- [MEGA Hub](https://mega.tensorplay.cn) · [ohtensorplay 组织](https://github.com/ohtensorplay)
- [`docs/observability.md`](docs/observability.md)：监控指标与面板接入配置。
- [`docs/oxr_vllm_comparison.pdf`](docs/oxr_vllm_comparison.pdf)：完整测试配置。
- [`docs/assets/`](docs/assets/)：品牌资产 — Logo、Banner 与应用图标。

---

<div align="center">
  <p>MegaLM 是 TensorPlay 生态的一部分。如果对你有帮助，欢迎 Star 与反馈。</p>
</div>
