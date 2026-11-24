# Nanbeige4.2-3B：MegaLM vs SGLang（单卡 RTX 3090）

本对比在同一张 RTX 3090 上、**不同时启动两个服务**（避免 24GB 显存争用），各跑一次完整基准后夹测。两组服务均使用本地 `model/nanbeige42/` checkpoint、bf16、greedy、streaming、并发 1/2/4、输出预算 64 token、每点 3 次试验取中位数。

## 环境

| 项 | MegaLM | SGLang |
| --- | --- | --- |
| 版本 | `megalm-runtime 0.2.0` + TensorPlay 1.3.0a0 | `sglang 0.5.23`（原生 `srt/models/nanbeige.py`） |
| 启动 | `megalm.gateway.server`，`--dtype bfloat16 --max-model-len 4096 --max-cache-tokens 8192 --gpu-memory-utilization 0.85 --cuda-graph-batches 1,2,4,8` | `sglang.launch_server --tp-size 1 --mem-fraction-static 0.8 --context-length 4096 --trust-remote-code` |
| 注意力后端 | MegaLM Triton paged attention | flashinfer |
| CUDA Graph | 捕获 3 个 bucket（1/2/4/8） | 启用（`decode.backend=full`，max_bs=48） |

> 注：以上基准采集时前端为 Python 网关；当前前端已切换为 Rust `megalm-server`（`rust/crates/server`），GPU 引擎与 worker 不变。

## 结果（长提示词，两边均输出满 64 token）

提示词：`请详细解释人工智能在医学影像诊断中的应用，包括数据准备、模型训练和临床部署三个方面，并各举一个具体例子。`

| 并发 | MegaLM TTFT (ms) | SGLang TTFT (ms) | MegaLM 解码 (tok/s) | SGLang 解码 (tok/s) | 吞吐比 | MegaLM E2E (ms) | SGLang E2E (ms) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 23.5 | 50.7 | **160.6** | 76.9 | 2.09× | 1192 | 2495 |
| 2 | 30.4 | 53.7 | **295.9** | 144.2 | 2.05× | 1281 | 2619 |
| 4 | 50.7 | 95.4 | **585.3** | 283.4 | 2.07× | 1310 | 2708 |

SGLang 在长提示词下复测两轮，结果一致（76.2 / 140.8 / 281.8 与 76.9 / 144.2 / 283.4 tok/s），非偶发。

## 短提示词参考

提示词：`你好，请用一句话介绍你自己。`

| 并发 | MegaLM (tok/s) | SGLang (tok/s) | 备注 |
| --- | --- | --- | --- |
| 1 | 155.5 | 163.5 | MegaLM 实际输出约 29 token（提前 EOS），SGLang 跑满 64，E2E 不可直接比较 |
| 2 | 237.0 | 298.4 | 同上 |
| 4 | 524.9 | 591.3 | 同上 |

## 说明与 caveat

- 两边 chat template 渲染的 prompt token 数略有差异（短提示 41 vs 38，长提示约 60 vs 57），但同一次对比中输出 token 数一致，吞吐/TTFT 可比。
- 吞吐为聚合值 `总输出 token / 该批最慢请求 E2E`，包含 prefill；输出均为 64 token/请求。
- 该结果只描述这组参数、这张 RTX 3090（sm86）与这个负载，不应外推到其他 GPU/长度/并发。SGLang 在 sm86 上使用 flashinfer 后端，长上下文下的 decode 明显慢于 MegaLM 的 Triton paged 内核。
- 测量脚本：`tests/bench_nanbeige_compare.py`，原始数据：`/tmp/opencode/nb_megalm_fair.json`、`nb_sglang_fair2.json`。
