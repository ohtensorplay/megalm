# 嵌入模型（BGE / BERT 编码器）

MegaLM 提供独立于生成调度器的嵌入任务路径：无采样、无解码、无 KV arena，
一次 padded 编码器前向得到一条向量。设计参考为仓库内锁定的 `refs/vllm`
（commit `d6dbdb9`）。最终验收以重新编译后的产物运行。

## 启动

```bash
megalm-serve \
  --model BAAI/bge-base-en-v1.5 \
  --served-model-name bge-base-en-v1.5 \
  --task embed \
  --dtype float16 \
  --port 8000
```

`--task embed` 让 worker 构造 `EmbeddingEngine` 而不是生成 `Engine`。相关参数：

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `--task` | `generate` | `embed` 开启嵌入路径 |
| `--dtype` | `bfloat16` | bge 推荐 `float16` |
| `--embedding-max-batch-size` | `64` | 单次编码器前向的输入上限（超出自动切片） |
| `--embedding-max-sequence-length` | 模型上限 | 输入 token 上限 |
| `--max-model-len` | 32768 | 前端模型长度声明；嵌入实际使用模型/配置上限 |

嵌入 worker 复用现有 Rust 前端与 Unix socket 数据面，新增 `SUBMIT_EMBEDDINGS`
/ `EMBEDDING_RESULT` 帧；请求在 worker 内按默认 1 ms 窗口合并（与生成侧
admission coalescing 一致），再按池化选项分组。

## API

`POST /v1/embeddings`，OpenAI 兼容：

```bash
curl http://127.0.0.1:8000/v1/embeddings \
  -H 'Content-Type: application/json' \
  -d '{"model":"bge-base-en-v1.5","input":["query one","query two"]}'
```

支持字段：

| 字段 | 默认值 | 说明 |
| --- | --- | --- |
| `input` | 必填 | 字符串、字符串数组，或 token-id 数组（跳过分词） |
| `model` | 空 | 非空时必须等于 `--served-model-name`，否则 404 |
| `encoding_format` | `float` | `float` 或 `base64`（小端 float32） |
| `dimensions` | 模型维度 | Matryoshka 截断，范围 `[1, embedding_size]` |
| `use_activation` | 池化配置 | 覆盖 L2 归一化开关 |
| `truncate_prompt_tokens` / `truncation_side` | 模型上限 / `right` | 超长输入截断 |

响应包含 `object=list`、`data[].embedding`、`usage.prompt_tokens` 与
`usage.total_tokens`（无 `completion_tokens`，与 vLLM 一致）。
非法 `encoding_format`、空输入、非法 token id、非零维度等返回统一
`{"error": {...}}` 400 信封。

## 池化与数值

- Sentence-Transformers `modules.json` / `1_Pooling/config.json` 优先；
  bge 解析为 CLS + Normalize。无 ST 元数据时 BERT 默认 CLS + L2。
- 支持 CLS / MEAN / LAST；注意力掩码参与 MEAN 与前向，padding 不影响结果。
- bge 权重与 HuggingFace `BertModel` 参考逐元素对齐（CPU fp32）：
  `max_abs_err ≈ 1.6e-7`，cosine = 1.0。`devdoc/check_bge_parity.py`。
- HTTP 服务使用 fp16，与 HF fp32 参考 cosine ≥ 0.999998。

## 性能验收

同一 RTX 3090、同一 `BAAI/bge-base-en-v1.5` 权重、相同请求集与并发，
对比仓库锁定的 `refs/vllm` 生产基线（`devdoc/bench_embed_vllm.py`，
3 轮取中位数；参考侧因自带 FA 扩展与本地 torch ABI 不兼容，用 harness shim
强制 Triton attention 后端）：

| 批次 × 并发 × 长度 | MegaLM texts/s | vLLM texts/s | 加速 |
| --- | ---: | ---: | ---: |
| 1 × 32 × 32 | 566.54 | 351.88 | 1.61× |
| 8 × 16 × 64 | 1146.53 | 827.14 | 1.39× |
| 16 × 8 × 128 | 916.26 | 728.30 | 1.26× |
| 32 × 4 × 128 | 921.95 | 729.96 | 1.26× |

原始输出见 `devdoc/embed_benchmark.txt`。基准环境另有常驻 OXR 服务占用约
15 GB 显存，两侧条件相同。
