# 可观测性接入指南

MegaLM 对外暴露三类标准信号，可接入主流监控/日志面板：

| 信号 | 入口 | 格式 |
|---|---|---|
| 指标（拉取） | `GET /metrics` | Prometheus text exposition 0.0.4，支持原生 label |
| 指标 + 链路（推送） | OTLP/HTTP | OpenTelemetry 1.44，OTLP over HTTP/protobuf |
| 健康 | `GET /health` | JSON `{"status":"ok"}`，异常返回 503 |
| 日志 | stderr | 单行 JSON（`JsonLogFormatter`），含 `timestamp/level/logger/event/pid` + 结构化字段 + trace/span id |

## 指标格式约定

`/metrics` 由 Rust core 渲染，输出符合 Prometheus 0.0.4：

- 每个序列都有 `# HELP` 与 `# TYPE`。
- 计数/累计量标为 `counter`；均值、比率、容量、空闲/在途量标为 `gauge`。
- 名称以 `_histogram` 结尾的字段渲染为**真 histogram**：`_bucket{le="..."}`（累积）、`_sum`、`_count`，可直接用 `histogram_quantile()`。注意 `_sum` 由桶上界近似得出（非逐样本精确和），分位数为近似值。
- 为保持兼容，指标名**不加** `_total` 后缀；`rate()`/`increase()` 对 `counter` 类型依然有效。
- **原生 label**：形如 `{"__value__":3,"__type__":"counter","__labels__":{"status":"503","route":"/v1/chat"}}` 的描述符会渲染为 `megalm_http_responses{route="...",status="503"} 3`；同一指标的 HELP/TYPE 只输出一次。
- 直方图可携带 `__sum__` / `__count__` 提供**精确**的样本总数与总和（延迟直方图已接入），避免桶上界近似。
- `megalm_replica_*`（延迟均值、直方图、解析器计数、请求结果）在**嵌入式与子进程两种拓扑下均存在**，面板无需按部署模式改写查询。

常用指标：

| 名称 | 含义 |
|---|---|
| `megalm_proxy_admitted_requests` | 网关接收请求数 |
| `megalm_replica_ttft_ms_mean` / `megalm_replica_tpot_ms_mean` | 首 token / 每 token 延迟均值 |
| `megalm_replica_parser_pushes` / `_diagnostics` / `_bytes` | Rust 解析器计数 |
| `megalm_engine_steps` / `megalm_engine_emitted_tokens` | 引擎步数 / 输出 token |
| `megalm_engine_sequence_batch_histogram` | 序列批大小分布（真 histogram） |
| `megalm_engine_structured_requests` / `_structured_degraded` / `_structured_mask_steps` / `_structured_rejected` / `_structured_grammar_errors` | 约束解码计数 |
| `megalm_engine_structured_compile_failures` / `_matcher_errors` / `_active_matchers` / `_bitmask_buffers` | 约束解码健康度 |
| `megalm_engine_structured_degraded_by_reason_*` / `_finish_reasons_*` | 降级原因与请求结束原因分类 |
| `megalm_replica_ttft_ms_histogram` / `_tpot_ms_histogram` / `_e2e_ms_histogram` | TTFT / TPOT / 端到端延迟分布（真 histogram） |
| `megalm_scheduler_preemptions` / `_deadline_expirations` / `_overload_rejections` / `_cache_capacity_waits` | 队列抢占、超时、过载与容量等待（饱和度/错误信号） |
| `megalm_frontend_tokenizer_batcher_*` / `_admission_batcher_*` / `_omni_admission_batcher_*` | 前端批处理器吞吐与批次分布 |
| `megalm_worker_admitted_requests` / `_sent_bytes` | replica 工作进程吞吐 |

维度由抓取器自动附加（如 Prometheus 的 `instance`、`job`）；多副本各自暴露 `/metrics`。

---

## Prometheus + Grafana（通用底座）

```yaml
# prometheus.yml
scrape_configs:
  - job_name: megalm
    scrape_interval: 5s
    metrics_path: /metrics
    static_configs:
      - targets: ["megalm-host:8000"]
        labels:
          model: qwen3
```

Grafana 添加 Prometheus datasource 指向该 Prometheus，常用面板查询：

```promql
# QPS
rate(megalm_proxy_admitted_requests[5m])

# 首 token 延迟 P95（真延迟直方图，单位 ms）
histogram_quantile(0.95, sum by (le) (rate(megalm_replica_ttft_ms_histogram_bucket[5m])))

# 端到端延迟 P99
histogram_quantile(0.99, sum by (le) (rate(megalm_replica_e2e_ms_histogram_bucket[5m])))

# 平均 TPOT（两种拓扑均存在）
megalm_replica_tpot_ms_mean

# 请求结束原因分布（渲染器无 label，按名称后缀展开）
rate(megalm_engine_finish_reasons_stop[5m])
rate(megalm_engine_finish_reasons_length[5m])
rate(megalm_engine_finish_reasons_cancelled[5m])

# 过载/抢占/超时
rate(megalm_scheduler_overload_rejections[5m])
rate(megalm_scheduler_preemptions[5m])
rate(megalm_scheduler_deadline_expirations[5m])

# 约束解码降级率
rate(megalm_engine_structured_degraded[5m])
  / clamp_min(rate(megalm_engine_structured_requests[5m]), 1)

# 解析器诊断突增
increase(megalm_replica_parser_diagnostics[15m])
```

---

## 其他 Prometheus 兼容面板

以下面板都直接抓取上面的 `/metrics`，无需改代码：

- **VictoriaMetrics / VictoriaMetrics Cloud**：`scrape_configs` 与 Prometheus 相同；也可 `-promscrape.config` 复用。
- **Thanos / Grafana Mimir / Cortex**：作为 Prometheus 的远端存储或代替 Prometheus 抓取。
- **Grafana Cloud**：用 "Prometheus" datasource 配 Hosted Metrics 的 remote write 端点，或让 Grafana Agent 抓取后 remote write。
- **阿里云 Prometheus / 腾讯云 Prometheus / 华为云 CCE 监控**：创建 Prometheus 实例 → 添加"自定义抓取"任务，target 填 `/metrics`，与上面 job 等价。
- **Datadog**：启用 Datadog Agent 的 Prometheus integration，在 pod/服务上打注解：
  ```yaml
  annotations:
    ad.datadoghq.com/megalm.checks: |
      {"prometheus": {"instances": [{"prometheus_url": "http://%%host%%:8000/metrics", "namespace": "megalm", "metrics": ["megalm.*"]}]}}
  ```
- **New Relic**：用 nri-prometheus 集成配置 `prometheus_url: http://host:8000/metrics`，指标会以 `prometheus.megalm.*` 命名空间上报。
- **Honeycomb / Lightstep 等仅 OTLP 的 SaaS**：当前版本不提供 OTLP 推送；请用其 Prometheus 抓取能力（多数支持），或在中间部署 OpenTelemetry Collector 的 `prometheus` receiver + `otlphttp` exporter 做协议转换。

---

## 日志接入（Loki / ELK / Datadog Logs）

日志是单行 JSON 写到 stderr，容器平台可直接采集 stdout/stderr：

- **Grafana Loki**：用 Promtail / Grafana Alloy 采集容器日志，`json` 解析器提取 `event`、`level`、`request_id`：
  ```yaml
  - job_name: megalm-logs
    static_configs:
      - targets: [localhost]
        labels: { job: megalm }
    pipeline_stages:
      - json: {}
      - labels:
          level:
          event:
  ```
  LogQL 示例：`{job="megalm"} | json | event="constrained_token_rejected"`。
- **ELK / OpenSearch**：Filebeat 容器输入 + `json.message_key` 解析，按 `event` 建索引模板。
- **Datadog Logs**：Agent 开启容器采集并添加 JSON 解析，按 `event`、`level` 分面。

结构化事件（可直接建告警）：`parser_degraded`、`response_format_invalid`、
`constrained_decoding_skipped`、`constrained_decoding_unavailable`、
`constrained_grammar_error`、`constrained_token_rejected`、
`constrained_request_id_reused`。

---

## 告警示例（Prometheus rules）

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

## OTLP 与分布式追踪

安装可选 extra `pip install megalm-runtime[otel]` 后，用 CLI 或标准 `OTEL_*` 环境变量开启：

```bash
megalm-serve --model ./model \
  --otlp-metrics-endpoint http://otel-collector:4318/v1/metrics \
  --otlp-traces-endpoint  http://otel-collector:4318/v1/traces \
  --service-name megalm

# 或：export OTEL_EXPORTER_OTLP_ENDPOINT=http://otel-collector:4318
```

- **指标**：TTFT/TPOT/E2E 直方图（`megalm.ttft` / `.tpot` / `.e2e`，带 `model` 等属性）、
  HTTP 响应计数与耗时（`megalm.http.responses` / `.http.request.duration`，带
  `http.response.status_code` / `http.route`）、请求计数（`megalm.requests`）。
  `/metrics` 快照的每个数值还会自动镜像为 `megalm.<name>` observable gauge（首次出现时注册）。
- **追踪**：ASGI 中间件为每个 HTTP 请求创建 SERVER span，提取入站 W3C `traceparent`；
  关键子 span 包括 `tokenizer.encode` 与 `gateway.admit`（携带 input_tokens、choice 数、
  request_id 等属性）。5xx 与异常自动标记为 ERROR。
- **日志关联**：`log_event()` 自动把当前 `trace_id` / `span_id` 注入 JSON 日志，
  可在 Loki/Tempo/Grafana 中从日志跳转到 trace。
- 未配置或缺少依赖时所有信号为 no-op，不影响嵌入式使用与测试。

`GET /metrics` 与 OTLP 可同时开启：Prometheus 系面板走拉取，Datadog/New Relic/Honeycomb
等可直接接 Collector 的 OTLP 出口。

## 当前覆盖与边界

已覆盖：请求量与结束原因分类、HTTP 状态码/耗时（按路由与状态 label）、TTFT/TPOT/E2E
延迟分布（精确 sum/count）、队列深度与抢占/超时/过载/容量等待、KV 与前缀缓存命中率/利用率、
GDN 混合状态槽位利用率、缓存容量失败计数、CUDA graph 捕获失败/可用性、多模态按类型计数、
Rust 解析器计数、约束解码全链路指标、前端批处理器指标、GPU 利用率与显存 gauge。

尚未覆盖：

- 渲染器虽支持 label（描述符形式），但传统扁平指标仍按名称后缀展开维度；OTLP 侧有原生 label。
- 引擎内部 step 级细粒度 span（当前以指标/日志为主），进程间子 span 的完整传播需扩展 wire 帧字段。
- 指标为进程级；多副本由抓取器加 `instance` 标签区分，OTLP 侧由 `service.name` / pid 区分。
