# 快速开始

环境：Linux、Python ≥ 3.12、NVIDIA Ampere 及以上 GPU。

## 安装

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install tensorplay
python -m pip install -e .
python -m pip install -e '.[hub]'   # 从 Hub 仓库 ID 加载时需要
```

## 离线推理

```bash
python -m megalm --model <model> --prompt "Introduce yourself in one sentence." --max-new-tokens 16
```

## 在线服务

```bash
megalm-server --model <model> --served-model-name <name> --host 127.0.0.1 --port 8000
```

```bash
curl http://127.0.0.1:8000/health
# {"status":"ok"}

curl http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"<name>","messages":[{"role":"user","content":"Hello"}],"max_tokens":128,"stream":false}'
```

`stream: true` 返回 SSE。模型与运行时状态在 `/v1/models` 和 `/api/inference/status`，Prometheus 指标在 `/metrics`。

## 端点

| 方法 | 路径 | 用途 |
| --- | --- | --- |
| POST | `/v1/chat/completions` | OpenAI Chat；流式、工具、结构化输出 |
| POST | `/v1/completions` | OpenAI 文本补全 |
| POST | `/v1/embeddings` | OpenAI 嵌入（BGE / BERT，见 [`docs/models/embedding.md`](../models/embedding.md)） |
| POST | `/v1/messages` | Anthropic Messages（含 `count_tokens`） |
| POST | `/v1/responses` | OpenAI Responses（内置 response store） |
| POST | `/v1/systemone` · `/v1/decisions` | 决策模型读法 |
| POST | `/v1/load_lora_adapter` · `/v1/unload_lora_adapter` | 运行时 LoRA 注册 |
| GET | `/v1/models` · `/health` · `/metrics` | 模型、健康、指标 |
