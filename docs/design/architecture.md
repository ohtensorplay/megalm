# 架构

适用范围：当前实现，单机、TP=1。TensorPlay 提供张量运行时、算子分派、kernel 与序列化；MegaLM 在其上实现模型执行、调度、KV 缓存和服务协议。

## 1. 分层

```text
HTTP / SSE
    │
Gateway（gateway/server.py，FastAPI）
    │   协议适配、tokenizer、admission batching、远程媒体抓取、指标
    │   进程内直调 / sealed memfd + 控制帧
Replica（GPU owner：EmbeddedReplica | ProcessReplica）
    │   Scheduler → Engine → Model → Kernels
TensorPlay
```

Gateway 不执行模型，Engine 不处理 HTTP。`--replica-mode process` 只替换 Gateway 与 Replica 之间的传输，Scheduler、Model、CacheHandle 不变。

| 形态 | 类 | 说明 |
| --- | --- | --- |
| 嵌入式 | `EmbeddedReplica` | 同进程，零 RPC |
| 进程隔离 | `ProcessReplica` | 媒体走 sealed memfd，控制走版本化帧 |
| 多 GPU | — | `TopologySpec` 预留 TP / DP 字段；运行时 TP=1 |

多机 Controller、placement lease、P/D 分离不在本版本。

## 2. 模块落点

| 模块 | 路径 | 职责 |
| --- | --- | --- |
| 契约 | `src/megalm/contracts/` | `ModelSpec`、`ExecutionPlan`、`InferenceRequest`、`CacheHandle`、LoRA / StructuredOutput |
| 规划 | `startup.py`、`planning.py` | `PlanDraft`、冷启动状态机、不可变 `ExecutionPlan` |
| 引擎 | `runtime/engine.py`、`runtime/batch.py` | `add_request` / `step`、采样、CUDA Graph、事件 |
| 调度 | `runtime/scheduler.py` | 队列、连续批处理、抢占、超时 |
| 缓存 | `runtime/cache.py`、`radix_cache.py`、`hybrid_cache.py`、`kv_offload.py` | 分页 KV、前缀复用、分层页、混合状态 |
| 模型 | `models/`、`adapters/` | 原生前向；adapter 只声明模型事实 |
| Kernel | `kernels/` | 算子族独立文件 + typed host wrapper |
| Gateway | `gateway/` | 协议、媒体、tokenizer、指标、Replica 桥接 |
| Rust core | `rust/src/` | 流式解析、radix cache、SSE 帧、指标渲染 |

## 3. 请求路径

```text
CREATED → ROUTING → ADMITTED → READY → PREFILL → DECODE → FINISHING
                        │                          │
                        └─ WAIT_MEDIA              ├─ OUTPUT_BLOCKED
                                                   └─ PREEMPTED → PREFILL
终态：FINISHED / CANCELLED / REJECTED / FAILED
```

`WAIT_MEDIA` 不占语言模型 KV；`READY` 后开始分配缓存。Prefill 可分块，Decode 每轮通常一个 token。抢占回收运行期状态并重做 Prefill；已发布的完整页由分层缓存保留。Cancel、deadline、fatal error 从任意非终态进入清理，并释放 sampler / grammar / MTP 状态。

调度代价与回收顺序见 [调度器](scheduler.md)，缓存见 [KV Cache](kv-cache.md)。

## 4. 约束

- 每个可变状态只有一个所有者。
- 功能关闭时不分配资源、不安装 hook、热路径无分支。
- Scheduler、Cache Manager、Gateway 中禁止 `if model_name == ...`。
- 模型差异用 adapter 与组合表达，不继承一棵通用 Transformer 类层级。
- HTTP 逻辑不进入 Engine；大媒体不进入控制帧。
- 请求期间不 autotune；kernel fault 后不热切换布局或后端。
- 每个 kernel 算子族独立源文件与 typed wrapper；禁止万能 namespace 与 re-export 壳。
- 每个优化有对齐 reference 的测试与同条件基准。

## 5. Rust core

`rust/src/parser/` 按职责拆为 `inkling`、`tool/`（按模型族解析工具语法，参数词法扫描保留原文）、`output`、`reasoning`、`stop`、`json_format`。`bindings/` 只做 PyO3 facade。`cache/`、`sse.rs`、`frames.rs`、`metrics.rs` 各自独立。

UTF-8 holdback 用 `char_indices()`，避免切到多字节字符中间。随机流与全切分点测试覆盖任意输入与切分点不 panic。Criterion benchmark 位于 `rust/benches/parsers.rs`。解析器计数前缀为 `megalm_replica_parser_*`。

## 6. 验证

层级输出与最终 logits 对齐 checkpoint reference；greedy 序列一致；chunked prefill 与一次性 prefill 一致；prefix hit / miss 与抢占恢复一致；图像、视频、音频、MRoPE、GDN 状态有独立金测；`ExecutionPlan` 的 artifact / variant / layout 与实际 typed binding 一致。


