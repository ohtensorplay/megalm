# 统一 serving benchmark

`megalm-bench` 是一个用 Rust 实现的 OpenAI 兼容服务基准工具。它不绑定 MegaLM：
MegaLM、vLLM、SGLang、llama.cpp server 只要暴露 `/v1/chat/completions` 或
`/v1/completions`，就能用**同一份请求集、同一套采样参数、同一个计时口径**测量。
设计参考 `refs/vllm` 的 `vllm bench serve`（`vllm/vllm/benchmarks/serve.py`）、
`refs/sglang-omni` 的 stage/spec harness 与 `refs/UPSTREAMS.md` 的记录要求。

选 Rust 的原因：基准客户端本身不能成为瓶颈，GIL 下的 Python SSE 解析在高并发
会污染 TTFT/ITL；`reqwest(stream)` + `tokio` 可复用连接池并精确按单调时钟打点，
且工具与 Rust 前端、Criterion CI 共用一套工具链。vLLM 的 bench serve 也提供
`vllm-rs` Rust fast path，口径一致。

## 构建

```bash
cd rust
cargo build --release -p megalm-bench
./target/release/megalm-bench --help

# 也可装到 ~/.cargo/bin，直接在任意目录使用
cargo install --path crates/bench --locked
```

`cargo test -p megalm-bench` 会用本地 mock SSE server 跑端到端测试，无需 GPU。
CI 里 `cargo test --locked` 与 `--dry-run` smoke gate 覆盖该工具。

## 快速开始

单引擎：

```bash
rust/target/release/megalm-bench serve \
  --engine megalm=http://127.0.0.1:8000 \
  --model "$MODEL_ID" \
  --dataset synthetic --prompt-tokens 512 --max-tokens 128 \
  --concurrency 1,8,16 --trials 3 --warmup 2 \
  --num-prompts 64 --seed 0 \
  --output /tmp/opencode/serve-megalm.json
```

多引擎顺序对比（单卡时**必须**顺序跑，避免互相抢 GPU；请求集和随机种子共享）：

```bash
rust/target/release/megalm-bench serve \
  --engine megalm=http://127.0.0.1:8000 \
  --engine vllm=http://127.0.0.1:8001 \
  --model "$MODEL_ID" \
  --dataset synthetic --prompt-tokens 512 --max-tokens 128 \
  --concurrency 1,8,16 --trials 3 --warmup 2 \
  --ignore-eos --temperature 0 \
  --output /tmp/opencode/serve-ab.json \
  --markdown /tmp/opencode/serve-ab.md

rust/target/release/megalm-bench report \
  --input /tmp/opencode/serve-ab.json --format markdown
```

已有报告不会被覆盖，需覆盖时传 `--force`。`--dry-run` 只校验参数并输出 workload
摘要与 digest，不连接服务器。

## 请求集

| `--dataset` | 输入 | 说明 |
| --- | --- | --- |
| `synthetic` | `--prompt-tokens` | 固定文本重复到目标长度；传 `--tokenizer <tokenizer.json或目录>` 可按真实词表截断并给出精确 prompt token 数 |
| `jsonl` | `--dataset-path` | 每行 `{"prompt": ..., "output_length": 128}`，也接受 `input`/`text` 和 `max_tokens`/`max_new_tokens` |
| `sharegpt` | `--dataset-path` | `conversations` 里最后一个 human 消息作为 prompt，`output_length` 或 gpt 回复的词数作为输出预算 |

`--num-prompts` 限定样本数；`--seed` 决定抽样、洗牌与到达间隔，所有引擎使用
**同一个** `Workload` 实例（不是按种子各自重建后期望相同）。报告里记录
`workload.digest`：prompt 与输出预算的 SHA-256。两个报告 digest 不同则不可比较。
不传 tokenizer 时，`prompt_tokens_exact=false`，此时 token 数只以服务端
`usage.prompt_tokens` 为准，不能拿 `--prompt-tokens` 当实测值。

## 负载模型

- **闭环**（默认 `--request-rate inf`）：所有请求在 t=0 入队，并发受
  `--concurrency`（再受 `--max-concurrency` 上限）信号量控制。适合容量扫描。
- **开环**（`--request-rate R`）：按到达过程排程，`--burstiness 1` 为 Poisson，
  `inf` 为等间隔；其他值用 Gamma(shape=burstiness) 采样。累计延迟按
  `N/R` 归一化，降低种子间的吞吐抖动（与 vLLM 一致）。
- 每个并发级别先做 `--warmup` 次不计时请求，再做 `--trials` 次计时；JSON 保存每次
   trial 的指标与 `median`（逐字段逐元素中位数），不挑最好一次。
- 到达时间、请求开始/结束、每个 SSE chunk 到达时间都基于一个 trial 统一的
  `Instant` 原点；等待信号量的时间单独记为 `queue_wait_ms`。

## 指标定义

| 指标 | 定义 |
| --- | --- |
| `ttft_ms` | 请求发出到第一个非空 content chunk |
| `itl_ms` | 相邻 content chunk 到达间隔；`itl_complete=false` 表示有 chunk 含多个 token，此时 ITL 只是 chunk 间隔 |
| `tpot_ms` | 单请求 `(e2el - ttft) / (output_tokens - 1)`，输出为 1 时记 0 |
| `e2el_ms` | 请求发出到 `[DONE]`/流结束 |
| `output_throughput` | 成功请求的输出 token 总数 / trial 墙钟时间（`duration_s`） |
| `achieved_rps` / `offered_rps` | 完成请求数 / 墙钟；发起速率（闭环时等于 achieved） |
| `goodput_rps` | 满足全部 `--slo-ttft-ms` / `--slo-tpot-ms` / `--slo-e2e-ms` 且（开 `ignore-eos` 时）输出达到预算的请求数 / 墙钟 |
| `peak_output_throughput` / `peak_concurrency` | 1 秒桶内的最大输出 token 数与最大在途请求数 |
| `token_source` | `usage`：服务端上报 token 数；`chunks`：按 content chunk 计数（报告会显式标出） |
| `load_generation_valid` | **仅开环**有意义：`queue_wait_ms.p95 <= --max-queue-wait-ms`（默认 1000 ms）；false 表示客户端本身成了瓶颈。闭环下所有请求按设计在信号量排队，该字段记 true 并标 `load_generation_check=closed_loop_not_applicable`，不触发有效性警告 |
| `status_counts` / `error_classes` | HTTP 状态码与 `http_4xx`/`http_5xx`/`timeout`/`transport`/`protocol` 分类 |

所有延迟统计都给出 `n`/`mean`/`std`/`p50`/`p90`/`p95`/`p99`（`std` 为 numpy 默认
的总体标准差 ddof=0），分位数采用**线性插值**（numpy 默认口径，记录在
`percentile_method`）。峰值吞吐与峰值并发只统计成功请求，避免失败请求产生幻影在途
连接。每个 trial 额外记录 `seed` 与 `queue_wait_limit_ms`，便于单独复现某次到达序列。
`--save-detailed` 会把每请求的 `start_ms`/`end_ms`/`token_times_ms`/`error` 一并写入
JSON，便于画 CDF 和复核。Markdown 报告会在表格上方直接列出有效性警告（客户端过载、
无服务端 token 统计、ITL 仅为 chunk 间隔、贪心输出不一致）。

## 可复现性记录

报告 `provenance` 自动写入：schema 版本、UTC 时间、脱敏后的命令行、工具与 rustc
版本、git commit/branch/dirty 条目数、OS/架构/GPU（nvidia-smi）、全部引擎 URL、
`config`（endpoint、dataset、seed 相关负载参数、temperature、ignore-eos、
stream_options、request_rate、burstiness、并发、trial 数、`--metadata` 自定义项）、
`source_digests`（默认 `rust/crates/bench/src` + `rust/Cargo.toml` + `Cargo.lock`，
可用 `--source-path` 覆盖）以及 workload digest。API key 与敏感 header 在命令行
回显中会被打码。

对照 vLLM 的 bench serve：它不会自动把 seed/版本写进结果，需要手工 `--metadata`；
本工具把这些做成默认行为。`refs/UPSTREAMS.md` 要求的 model artifact、dtype/量化、
特性开关、硬件请在启动时用 `--metadata key=value` 补齐，例如：

```bash
--metadata dtype=bfloat16 quant=none tp=1 gpu=RTX3090 driver=570.xx
```

## 对比纪律

1. 同模型 artifact、同 dtype/量化、同 tokenizer、同 `--prompt-tokens`/`--max-tokens`、
   同 `--ignore-eos`/`--temperature`，用同一份 workload digest。
2. 单卡对比必须串行跑（本工具的顺序 `--engine` 就是为此设计），预热和测量窗口分开。
3. 先看 `failed`、`load_generation_valid`、`token_source`、`itl_complete`，再看吞吐。
   `token_source=chunks` 的数字只能定位，不能作为正式结论。
4. 报告中的数字需同时引用 commit 与 source digest；CI 数字要注明软件栈版本。
5. 贪心对比（ `--temperature 0`）加 `--verify-output`：工具会把同一 workload 下每个
   引擎的拼接输出做 SHA-256 写入 `output_verification.digests`，`match=false` 说明
   输出不一致（此时性能数字不可直接对比）。再加 `--fail-on-output-mismatch` 可让
   CI 直接失败。若各引擎的 `served-model-name` 不同，只能分别运行并在命令中显式
   记录（不要悄悄改请求集）。

## JSON 结构

```text
{
  "schema_version": 1,
  "provenance": { ... },
  "results": {
    "<engine>": [
      { "concurrency": 8, "trials": [{...}, {...}], "median": {...} }
    ]
  }
}
```

`serve` 的 `median` 是逐字段逐元素中位数（嵌套对象/数组递归），因此 trial 级
的 p95 与跨 trial p95 不会混淆。`report --format json|summary|markdown` 从这份
JSON 渲染表格；`vs base` 是同并发下相对第一个引擎的输出吞吐比。
