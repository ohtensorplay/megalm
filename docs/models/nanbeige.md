# Nanbeige4.2-3B

22 层 dense RoPE/GQA 解码器，整栈用共享权重执行 `num_loops=2` 次，实际计算深度 44 层；每层每次执行有独立 KV 缓存。3B 非嵌入参数。

## 1. 权重下载

MegaLM 只读取本地 checkpoint；`megalm-download` 通过各 Hub 的官方 SDK
（`megatensors` / `huggingface_hub` / `modelscope`）下载快照：

```bash
python -m pip install -e '.[hub]'
python -m megalm.download                                   # 默认仓库
python -m megalm.download hf:Nanbeige/Nanbeige4.2-3B \
  --local-dir model/nanbeige42
```

生产前端只加载本地目录，先下载再启动：

```bash
megalm-download hf:Nanbeige/Nanbeige4.2-3B --local-dir model/nanbeige42
megalm-server --model model/nanbeige42 --served-model-name nanbeige4.2-3b
```

只下载服务所需文件（重复 `--include`，传给 SDK 的 `allow_patterns`）：

```bash
python -m megalm.download hf:Nanbeige/Nanbeige4.2-3B \
  --local-dir model/nanbeige42 \
  --include '*.safetensors' --include '*.safetensors.index.json' \
  --include config.json --include generation_config.json \
  --include 'tokenizer*' --include special_tokens_map.json --include added_tokens.json
```

断点续传、SHA 校验、限速与缓存布局由 SDK 负责；`HF_TOKEN` / `HF_ENDPOINT`
等认证和端点环境变量沿用各 SDK 约定。详见
[模型来源](../getting_started/model-sources.md)。

## 2. 运行

```bash
python -m megalm --model model/nanbeige42 \
  --prompt "你好，请用一句话介绍你自己。" --max-new-tokens 32

megalm-server --model model/nanbeige42 \
  --served-model-name nanbeige4.2-3b --host 127.0.0.1 --port 8000
```

## 3. 实现

| 层 | 落点 |
| --- | --- |
| 契约 | `src/megalm/adapters/nanbeige.py` 读 `config.json` 生成带 `LoopSpec(core_start=0, core_end=22, num_passes=2)` 的 `ModelSpec`。未发布的可选特性（n-gram / mHC / depth attention / LoopSplit / QK-Norm / bias / rope scaling）显式报错。 |
| 原生模型 | `src/megalm/models/nanbeige/` 复用 `GenericCausalLM` 的 TensorPlay 算子与 safetensors 流式加载器；只重写前向调度：每轮 loop 复用同一份物理层权重，重置残差流，用 `layer_index + loop_idx * 22` 写独立 KV 层，每轮结束做 RMSNorm（`skip_loop_final_norm=false`）。 |
| 调度 / 缓存 | `startup.py` 与 `runtime/engine.py` 按 `ModelSpec.loop` 分配循环 KV 层，不按模型名分支；Nanbeige 得到 44 个 KV 层，Inkling-2 行为不变。 |
| 下载 | `src/megalm/model_source.py` 解析 `mega://` / `hf://` / `ms://` 并调用官方 SDK；`src/megalm/download.py` 是 CLI 包装，入口注册在 `pyproject.toml`。 |

## 4. 验证

```bash
PYTHONPATH=src python -m pytest tests/test_nanbeige_adapter.py tests/test_model_source.py -q
```
