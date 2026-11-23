<div align="center">
  <h1>🚀 MegaLM</h1>
  <p><strong>基于 <a href="https://github.com/lexing-2026/TensorPlay">TensorPlay</a> 构建的高性能大模型推理引擎</strong></p>
  <p>
    <a href="#-关于-megalm">关于</a> ·
    <a href="#-基于-tensorplay">技术基石</a> ·
    <a href="#-核心能力">核心能力</a> ·
    <a href="#-快速开始">快速开始</a> ·
    <a href="#-性能表现">性能表现</a> ·
    <a href="#-文档">文档</a>
  </p>
  <p><em>Powered by <a href="https://tensorplay.cn">TensorPlay</a> — 面向学习与实验的张量与内核框架。</em></p>
  <p>
    <a href="./README.md"><strong>EN</strong></a> ·
    <a href="./README.zh.md"><strong>中文</strong></a>
  </p>
</div>

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

### 安装

```bash
python -m pip install tensorplay
python -m pip install -e .
```

### 离线推理

```bash
python -m megalm \
  --model <your-model> \
  --prompt "你好，请用一句话介绍你自己。" \
  --max-new-tokens 16
```

### 启动 API 服务

```bash
megalm-serve <your-model> \
  --served-model-name <model-name> \
  --host 127.0.0.1 \
  --port 8000
```

服务启动后即可通过 `/v1/chat/completions` 流式推理，并从 `/v1/models`、`/api/inference/status` 获取模型与运行时信息。

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

- [TensorPlay 官方网站](https://tensorplay.cn) · [TensorPlay 源码](https://github.com/lexing-2026/TensorPlay)
- [`docs/observability.md`](docs/observability.md)：监控指标与面板接入配置。
- [`docs/oxr_vllm_comparison.pdf`](docs/oxr_vllm_comparison.pdf)：完整测试配置。

---

<div align="center">
  <p>MegaLM 是 TensorPlay 生态的一部分。如果对你有帮助，欢迎 Star 与反馈。</p>
</div>
