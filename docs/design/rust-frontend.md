# Rust 前端（tokenizer / chat renderer / axum server）

> 本文记录已落地的能力。Python 侧只保留 GPU worker 必需的进程入口与帧协议文件；Rust 前端是唯一的 HTTP 入口。

## 1. 架构

```text
axum (megalm-server)
  ├─ tokenizer / chat renderer  (Rust，HF tokenizer + minijinja)
  ├─ parser / SSE / metrics     (megalm_rs，复用既有 Rust core)
  └─ engine client  ──MGLM v6 frame / AF_UNIX socketpair──>  python -m megalm.gateway.worker
                                                              └─ Scheduler → Engine → Model → Kernels
```

Rust 进程独占 HTTP、tokenize、chat 模板渲染、输出解析与 SSE 装配；GPU 侧仍是 Python worker，但请求热路径不进入 Python 网关、不持有 GIL、不经过 FastAPI。Worker 是已有的 `gateway/worker.py`，协议是已有的 10 字节 MGLM v6 帧；没有引入第二套引擎协议。

## 2. Crate 划分

| Crate | 路径 | 职责 |
| --- | --- | --- |
| `megalm-rs` | `rust/src/` | 既有 parser / SSE / radix / frames / metrics，已从 `mod` 改为 `pub mod` 供前端复用 |
| `megalm-renderer` | `rust/crates/renderer/` | 本地 chat 模板加载与渲染：HF 风格 `trim_blocks`/`lstrip_blocks`、Python 字符串/字典方法、扩展 `tojson`（`sort_keys`/`separators`/`indent`/`ensure_ascii`，保留对象键顺序）、`raise_exception`/`strftime_now`、strict undefined、generation 块剥离、命名模板（`default`/`tool_use`）与额外模板变量 |
| `megalm-tokenizer` | `rust/crates/tokenizer/` | `ModelTokenizer` / `ModelDecodeStream` / 多模态 placeholder 展开，逐项对齐 `runtime/tokenizer.py` |
| `megalm-protocol` | `rust/crates/protocol/` | MGLM v6 帧编解码与请求/响应类型 |
| `megalm-engine-client` | `rust/crates/engine-client/` | socketpair 启动 worker、命令关联、输出事件分发 |
| `megalm-server` | `rust/crates/server/` | axum router 与 OpenAI 兼容端点 |

## 3. 与 Python 前端的对齐验证

本地模型上 `render_chat` + `encode` 的渲染字节数、token id 序列与 Python `ModelTokenizer` 完全一致；`output_grammar()`、`tool_call_parser()` 一致。`cargo test -p megalm-tokenizer --test model` 在 `MEGALM_TEST_MODEL` 指向含 `tokenizer.json` 的模型目录时，用 `tests/fixtures/inkling_chat.json`（由 Python 前端生成的 golden 文本）逐字节比对纯聊、工具声明、thinking 等场景的渲染结果。渲染器的 Jinja 兼容性由 `cargo test -p megalm-renderer --test jinja_compat` 覆盖，对照 HF `apply_chat_template` 的 Jinja 环境设置与 vLLM 的 `tojson` 扩展。

## 4. 已接线的端点

- `GET /health`、`GET /api/health`、`GET /metrics`、`GET /profiling`、`GET /v1/models`、`GET /server_info`
- `POST /v1/chat/completions`：流式 SSE 与非流式；工具调用、reasoning、`n>1` 并行 choice、逐请求 LoRA、历史裁剪（对齐 `_fit_text_history`）、`initial_in_thinking` token 级判定；`chat_template_kwargs`（或顶层 `reasoning_effort`）透传给 Jinja 模板，保留变量不可覆盖；多模态（image/video/audio）经 `SUBMIT_MULTIMODAL_BATCH.chat_template_kwargs` 传给 worker，走 sealed memfd
- `POST /v1/completions`（流式与非流式）
- `POST /v1/embeddings`：OpenAI 嵌入；`--task embed` 的 worker 走独立池化数据面
  （`SUBMIT_EMBEDDINGS` / `EMBEDDING_RESULT`），不进入生成调度器
- `POST /v1/messages`、`POST /v1/messages/count_tokens`（Anthropic Messages，含流式事件转换）
- `POST /v1/responses`、`GET /v1/responses/{id}`、`POST /v1/responses/{id}/cancel`（OpenAI Responses，含内存 response store 与流式事件转换）
- `POST /v1/decisions`、`POST /v1/systemone`（结构化决策：label token 校验 + `score_labels` logprob 打分 + 概率归并）
- `POST /v1/load_lora_adapter`、`POST /v1/unload_lora_adapter`
- `POST /api/inference/cancel`、`/validate`、`/load`、`GET /api/inference/status`
- `GET/POST /api/models/loras`、`DELETE /api/models/loras/{reference}`、`GET /api/models/list`

## 5. 状态与剩余缺口

- 全部 25 个 Python 路由都有 Rust handler，无 501、无缺失。
- 结构化决策已接线：`decisions.rs` 移植 `decision_api.py` 的 prompt / label token 校验 / 概率归并，经 `score_labels` 走 worker 的 label logprob 路径；prompt 与 Python 逐字节对齐。`sampling_from_wire` 与 `scheduler.py` 均已支持 `score_labels`。
- 多模态媒体面已接线：`media.rs` 移植 `parse_media_chat`，`remote_media.rs` 移植 `resolve_remote_media`（reqwest + public-address DNS resolver，http 默认禁用、域名白名单、逐跳重定向校验），`engine-client/media.rs` 创建 sealed memfd，writer 用 `sendmsg` + `SCM_RIGHTS` 附带 FD，走 `SUBMIT_MULTIMODAL_BATCH`；`n>1` 通过 `dup` 共享 sealed 文件。Axum body limit 96 MiB。多模态请求不做本地历史裁剪（与 Python 一致），LoRA 与多模态互斥。
- 前端 microbatching 均已落地：admission 侧 `BatchingOptions` 在 200µs 窗口内合并并发 `submit_batch` 为单帧（上限 64，结果按 `request_id` 回分），`--admission-coalesce-us` / `--admission-max-batch` 可调；tokenizer 侧 `TokenizerBatcher` 把并发 `encode` 合并成一次 `encode_batch`（在 blocking pool 上执行），`--tokenizer-coalesce-us` / `--tokenizer-max-batch` 可调。多模态帧不进 admission batcher，因为 FD 权能不能跨调用方合并。
- 请求级 profiling 已落地：设置 `--profile-file` 或 `MEGALM_PROFILE_FILE` 后按 JSONL 写 `admitted` / `first_token` 里程碑，`GET /profiling` 返回与 Python `profiling.view` 同构的 timelines / stage_breakdowns / hops / event_count；未设置时零开销。
- P/D 分离已移植（`server/src/pd_router.rs`）、同样为 opt-in：`--pd-prefill-url` 与 `--pd-decode-url` 必须同时给出；预填强制非流式取回 `kv_transfer_params`，decode 带 handle 流式转发，过滤 hop-by-hop 头，记录 hop_sent/hop_received 里程碑。用两个本地 mock upstream 实测了握手与 SSE 转发。
- Responses/Messages 的流式事件转换覆盖文本、reasoning 与工具调用；built-in 工具在无 tool server 时与 Python 同样返回 400。
- TP>1 已接线并带 watchdog：Rust 侧分配 rendezvous 端口、设置 RANK/WORLD_SIZE/CUDA_VISIBLE_DEVICES 并启动 `megalm.gateway.tp_worker` followers；`WorkerSupervisor` 每 500ms 用 `try_wait` 检查所有 rank，任一退出则 kill + wait 整组后重建并热替换 `EngineClient`（10s 冷却防止坏配置刷屏），stream id 由 supervisor 统一分配避免跨代冲突；`/server_info.worker_generation` 可见重建代数。在途请求随旧 transport 断开而失败，由调用方重试。
- `EngineConfig` 的 78 个字段已与 Python 逐字段对齐（见 `engine-client/examples/config.rs` 的校验：`rust-only []`、`python-only []`）；CLI 暴露了其中常用开关（含 attention 系列），其余走 Python 默认值。
- 客户端 `command_id` 使用 `AtomicU64` 但 `SubmitBatch` 结构仍带占位 `command_id`，由 client 覆盖；后续可收敛为无该字段的 wire 类型。

## 6. 构建与运行

```bash
cd rust
cargo build --release -p megalm-server
./target/release/megalm-server \
  --model /path/to/model \
  --served-model-name my-model \
  --host 127.0.0.1 --port 8000
```

`megalm_rs` 的 PyO3 扩展仍由 maturin 从 `rust/Cargo.toml` 构建，workspace 化不影响原 Python 安装路径。

## 6.1 真机验证结论

在 RTX 3090 + tensorplay 环境完成真实 GPU 端到端验证：Rust server 收到 worker READY，chat（流式/非流式）、decisions、systemone、messages、responses、health、server_info 均返回真实推理结果（HTTP 200）。多模态请求返回 400 `the loaded native model does not support visual inputs`，与 Python 引擎对该 checkpoint 的行为一致（`model_type=inkling2_loop` 无视觉处理器，`input_modalities=[]`）。

## 7. 鉴权（API Key）

未配置 key 时不安装任何鉴权中间件，行为与之前完全一致；配置后受保护端点只接受 `Authorization: Bearer <key>`。

```bash
# 单个 key，命令行或环境变量二选一
megalm-server --model /path/to/model --api-key secret
MEGALM_API_KEY=secret megalm-server --model /path/to/model

# 多个 key：重复旗标，或以逗号 / 空格分隔
megalm-server --model /path/to/model --api-key a --api-key b
```

- 受保护前缀：`/v1`、`/v2`、`/inference`、`/api`（含 `/api/inference/*`、`/api/models/*` 管理端点）。
- 免鉴权：`OPTIONS` 预检，以及 `/health`、`/api/health`、`/metrics`、`/server_info`。
- key 以 SHA-256 摘要保存，比较使用 `subtle` 的常量时间比较；不匹配返回 `401 {"error": "Unauthorized"}`。
- 未认证请求在 handler 之外被拒绝，不会进入引擎；未配置 key 时热路径没有额外开销。
