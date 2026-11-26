# 可观测性

> 实现位置：`tensorplay-observability` 共享包（与本仓库的 `megalm.observability` /
> `observability_otel` / `megalm_rs::metrics` 薄壳）。megaRL 复用同一份协议与渲染器，
> 只在指标前缀（`megalm_` / `megarl_`）和 RL 语义上区分。

MegaLM 暴露四类信号：`GET /metrics`（Prometheus 0.0.4）、OTLP/HTTP（可选 extra）、`GET /health`、stderr 单行 JSON 日志。

| 信号 | 入口 | 格式 |
| --- | --- | --- |
| 指标拉取 | `GET /metrics` | Prometheus text 0.0.4，支持原生 label |
| 指标 + 链路推送 | OTLP/HTTP | OpenTelemetry，over HTTP/protobuf |
| 健康 | `GET /health` | `{"status":"ok"}`，异常 503 |
| 日志 | stderr | `JsonLogFormatter`：时间戳 / 级别 / logger / event / pid + 结构化字段 + trace / span id |

## 1. 指标格式

`/metrics` 由 Rust core 渲染：

- 每个序列有 `# HELP` 与 `# TYPE`。
- 计数 / 累计量标 `counter`；均值、比率、容量、空闲 / 在途量标 `gauge`。
- `_histogram` 结尾的字段渲染为真 histogram：`_bucket{le="..."}` 累积、`_sum`、`_count`，可直接用 `histogram_quantile()`。`_sum` 默认由桶上界近似；延迟直方图另带 `__sum__` / `__count__` 提供精确值。
- 指标名不加 `_total` 后缀；`rate()` / `increase()` 对 counter 有效。
- 原生 label 描述符 `{"__value__":3,"__type__":"counter","__labels__":{"status":"503","route":"/v1/chat"}}` 渲染为 `megalm_http_responses{route="...",status="503"} 3`；HELP / TYPE 只输出一次。
- `megalm_replica_*` 在嵌入式与子进程两种拓扑下都存在，面板无需按部署模式改写查询。Rust 前端始终在线生产 `megalm_proxy_admitted_requests`、`megalm_replica_ttft_ms_mean` / `_tpot_ms_mean` / `_e2e_ms_mean`、三组延迟 `_histogram`、`finish_reasons_*` 与 `parser_{pushes,finishes,calls,diagnostics,bytes}`；无需 opt-in。

常用指标：

| 名称 | 含义 |
| --- | --- |
| `megalm_proxy_admitted_requests` | 网关接收请求数 |
| `megalm_replica_ttft_ms_mean` · `megalm_replica_tpot_ms_mean` | TTFT / TPOT 均值 |
| `megalm_replica_parser_pushes` · `_diagnostics` · `_bytes` | Rust 解析器计数 |
| `megalm_engine_steps` · `megalm_engine_emitted_tokens` | 引擎步数 / 输出 token |
| `megalm_engine_sequence_batch_histogram` | 序列批大小分布 |
| `megalm_engine_structured_requests` · `_degraded` · `_mask_steps` · `_rejected` · `_grammar_errors` | 约束解码计数 |
| `megalm_engine_structured_compile_failures` · `_matcher_errors` · `_active_matchers` · `_bitmask_buffers` | 约束解码健康度 |
| `megalm_replica_ttft_ms_histogram` · `_tpot_ms_histogram` · `_e2e_ms_histogram` | 延迟分布 |
| `megalm_scheduler_preemptions` · `_deadline_expirations` · `_overload_rejections` · `_cache_capacity_waits` | 抢占 / 超时 / 过载 / 容量等待 |
| `megalm_frontend_tokenizer_batcher_*` · `_admission_batcher_*` · `_omni_admission_batcher_*` | 前端批处理器 |
| `megalm_worker_admitted_requests` · `_sent_bytes` | replica 工作进程吞吐 |

多副本各自暴露 `/metrics`；维度由抓取器附加（`instance`、`job`）。

## 2. Prometheus + Grafana

```yaml
scrape_configs:
  - job_name: megalm
    scrape_interval: 5s
    metrics_path: /metrics
    static_configs:
      - targets: ["megalm-host:8000"]
        labels: { model: qwen3 }
```

```promql
rate(megalm_proxy_admitted_requests[5m])

histogram_quantile(0.95, sum by (le) (rate(megalm_replica_ttft_ms_histogram_bucket[5m])))
histogram_quantile(0.99, sum by (le) (rate(megalm_replica_e2e_ms_histogram_bucket[5m])))

megalm_replica_tpot_ms_mean

rate(megalm_engine_finish_reasons_stop[5m])
rate(megalm_engine_finish_reasons_length[5m])
rate(megalm_engine_finish_reasons_cancelled[5m])

rate(megalm_scheduler_overload_rejections[5m])
rate(megalm_scheduler_preemptions[5m])
rate(megalm_scheduler_deadline_expirations[5m])

rate(megalm_engine_structured_degraded[5m])
  / clamp_min(rate(megalm_engine_structured_requests[5m]), 1)

increase(megalm_replica_parser_diagnostics[15m])
```

其他直接抓取 `/metrics` 的面板：VictoriaMetrics（同 `scrape_configs`）、Thanos / Mimir / Cortex、Grafana Cloud、阿里云 / 腾讯云 / 华为云 Prometheus、Datadog（Prometheus integration）、New Relic（nri-prometheus）。仅支持 OTLP 的 SaaS 可部署 OpenTelemetry Collector 做 `prometheus` receiver → `otlphttp` exporter 转换。

## 3. 日志

日志是单行 JSON 写到 stderr。Loki 用 Promtail / Alloy 的 `json` 解析器提取 `event` / `level` / `request_id`；ELK / OpenSearch 用 Filebeat 容器输入 + `json.message_key`；Datadog Logs 开启容器采集并按 `event` / `level` 分面。

结构化事件：`parser_degraded`、`response_format_invalid`、`constrained_decoding_skipped`、`constrained_decoding_unavailable`、`constrained_grammar_error`、`constrained_token_rejected`、`constrained_request_id_reused`。Rust 前端同样以单行 JSON 写 stderr（时间戳 / 级别 / logger / event / pid + 字段）：解析器降级逐条记录 `parser`、`kind`、`message`、`position`、`context`、`expected`、`streaming`；`response_format` 校验失败记录 `request_id` 与 `error`。两者都是观测性副作用，不改变返回内容。

## 4. 告警

```yaml
groups:
  - name: megalm
    rules:
      - alert: MegaLMHighTTFT
        expr: megalm_replica_ttft_ms_mean > 2000
        for: 5m
      - alert: MegaLMConstraintRejections
        expr: rate(megalm_engine_structured_rejected[5m]) > 0.1
        for: 5m
      - alert: MegaLMGrammarErrors
        expr: increase(megalm_engine_structured_grammar_errors[15m]) > 0
      - alert: MegaLMReplicaDown
        expr: up{job="megalm"} == 0
        for: 1m
```

## 5. OTLP

OTLP 是**库级可选能力**，不是 Rust 前端 CLI 的参数。Rust 前端（`megalm-server`）只暴露 `/metrics` / `/health` / JSONL 日志；OpenTelemetry 通过两条路径接入：

1. **Collector 转换（推荐，语言无关）**：用 OpenTelemetry Collector 的 `prometheus` receiver 抓取 `/metrics`，再由 `otlphttp` exporter 推送到任意 OTLP 后端。PromQL 与告警规则保持不变。
2. **Python 库 hook**：`pip install megalm-runtime[otel]` 后，`megalm.observability_otel` 提供 `configure_telemetry()`、`mirror_snapshot(snapshot)` 与 `log_event()`。`mirror_snapshot()` 把 `/metrics` 快照的嵌套数值展开成 `megalm.<name>` observable gauge，供 Python 侧嵌入者复用；未配置或缺少依赖时全部 no-op。

- 指标：`megalm.ttft` / `.tpot` / `.e2e`（带 `model`）、`megalm.http.responses` / `.http.request.duration`（带 `http.response.status_code` / `http.route`）、`megalm.requests`。
- 追踪：`start_server_span()` / `end_server_span()` 创建 SERVER span 并提取入站 W3C `traceparent`；子 span 有 `tokenizer.encode`、`gateway.admit`（带 input_tokens、choice 数、request_id）。5xx 与异常标记 ERROR。
- 日志关联：`log_event()` 注入 `trace_id` / `span_id`。

## 6. 请求级 JSONL Profiling（opt-in）

`megalm-server --profile-file <path>`（或 `MEGALM_PROFILE_FILE`）开启请求级里程碑记录。
Python Gateway / Worker 与 Rust 前端写**同一 schema**的逐行 JSONL：

```
{"ts_ns":…, "pid":…, "run_id":…, "request_id":…, "event":…, "stage":…, "extra":{…}}
```

- milestone：`admission`、`prefill_start`、`first_token`、`hop_sent`、`hop_received`、`terminal_response`。
  Rust 的 `admitted` 在加载时归一为 `admission`，因此两前端文件可按 request 合并。
- `run_id` 由 `MEGALM_PROFILE_RUN_ID` 或 `--profile-run-id` 指定，默认 uuid4；同一 run 的
  Gateway / Worker / TP follower 通过环境变量继承相同文件与 run id。
- active stage 用 `ContextVar`（asyncio）+ `threading.local`（replica 线程）双绑定。
- 只读 `GET /profiling[?request_id=…]` 返回 `timeline`、`stage_breakdown`、`hop_breakdown`
  及 `hops`；hop 按 `(request_id, src, dst, chunk_id)` 跨进程配对。也可离线用
  `megalm.profiling.view(glob)` 合并多进程文件。
- 元数据只接受小 JSON 标量；写失败吞掉并计数，`/metrics` 的 `profiling` 快照暴露
  `events_recorded` / `dropped_writes` / `dropped_metadata`。
- **默认关闭时零成本**：未配置文件时 `record()` 仅一次 `None` 检查；profiler 是可删除插件，
  Scheduler / Engine 热路径零改动，开启不改变输出与失败语义。

## 7. 边界

已覆盖：请求量与结束原因分类、HTTP 状态码 / 耗时（按路由与状态 label）、TTFT / TPOT / E2E 分布、队列深度与抢占 / 超时 / 过载 / 容量等待、KV 与前缀缓存命中率、GDN 混合状态槽位、缓存容量失败、CUDA Graph 捕获失败 / 可用性、多模态按类型计数、Rust 解析器计数、约束解码全链路、前端批处理器、GPU 利用率与显存 gauge、opt-in 请求级 JSONL 时间线与跨进程 hop 配对。

未覆盖：传统扁平指标仍按名称后缀展开维度，原生 label 在 OTLP 侧可用；引擎 step 级 span 尚缺（当前 `prefill_start` 记在提交时刻而非首个 prefill kernel）；进程间子 span 的完整传播需要扩展 wire 帧字段；IPC（Gateway ↔ Worker）帧级 hop 尚未打点。指标是进程级，多副本靠抓取器 `instance` 或 OTLP `service.name` / pid 区分。
