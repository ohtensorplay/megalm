# MegaLM 文档

MegaLM 是 TensorPlay 的推理与服务层。Linux、Python ≥ 3.12、NVIDIA Ampere 及以上。

按目标选文档：

| 目标 | 文档 |
| --- | --- |
| 安装并运行离线 / 在线推理 | [快速开始](getting_started/quickstart.md) |
| MegaHub / Hugging Face / ModelScope 仓库加载 | [模型来源](getting_started/model-sources.md) |
| 用 Docker 构建 / 运行服务 | [Docker 部署](deployment/docker.md) |
| 系统分层、进程边界、模块位置 | [架构](design/architecture.md) |
| ModelSpec、冷启动状态机、ExecutionPlan | [ExecutionPlan 与冷启动](design/execution-plan.md) |
| 请求生命周期、连续批处理、抢占 | [调度器](design/scheduler.md) |
| 分页 KV、Radix、分层卸载、混合状态 | [KV Cache](design/kv-cache.md) |
| FP8 / INT4 / NVFP4 KV | [KV Cache 量化](design/kv-cache-quantization.md) |
| `response_format` 的合法输出保证 | [结构化输出](design/structured-output.md) |
| 多 LoRA | [Multi-LoRA](design/lora.md) |
| Stax 预填充下沉 | [编译后端](design/compile-backend.md) |
| 远程媒体抓取与 SSRF | [媒体数据面](design/media-plane.md) |
| Prometheus / Grafana / OTLP | [可观测性](design/observability.md) |
| 特定模型适配 | [Nanbeige4.2-3B](models/nanbeige.md) |
| 基准结果与方法 | [基准](benchmarks/README.md) |
| 统一 serving 基准工具（吞吐、TTFT、ITL、goodput、provenance） | [serving benchmark](benchmarks/serving.md) |

设计文档的格式：适用范围、机制说明、代码落点、配置、验证。

性能声明需要同时记录仓库 commit、源码摘要、驱动 / CUDA / Torch 版本、模型 artifact、dtype / 量化、请求集、并发、输出长度、特性开关与硬件。详见 [基准方法](benchmarks/README.md)。
