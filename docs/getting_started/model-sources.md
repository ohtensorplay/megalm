# 模型来源：MegaHub / Hugging Face / ModelScope

MegaLM 的前端都加载本地 checkpoint 目录。`python -m megalm` 与
`megalm-download` 可以直接接受 Hub 仓库 ID：它们先调用各 Hub 的**官方 SDK**
下载快照，之后和本地目录完全一样。生产前端 `megalm-server`（Rust）只接受
本地路径，请先用 `megalm-download` 拿到目录再传给 `--model`。`refs/` 里的
vLLM / SGLang 只读参考不参与这条路径。

## 1. 安装 Hub SDK

默认安装只包含运行时；解析远端仓库需要可选依赖：

```bash
python -m pip install -e '.[hub]'
# 或按需：pip install megatensors / huggingface_hub / modelscope
```

| 前缀 | Hub | 官方 SDK | 认证 / 端点环境变量 |
| --- | --- | --- | --- |
| `mega://`、`megahub://` | MegaHub | `megatensors.snapshot_download` | `MEGA_TOKEN`、`MEGA_ENDPOINT`、`MEGA_HUB_CACHE` |
| `hf://`、`huggingface://` | Hugging Face | `huggingface_hub.snapshot_download` | `HF_TOKEN`、`HF_ENDPOINT`、`HF_HUB_CACHE` |
| `ms://`、`modelscope://` | ModelScope | `modelscope.snapshot_download` | `MODELSCOPE_API_TOKEN`、`MODELSCOPE_ENDPOINT`、`MODELSCOPE_CACHE` |

也支持单冒号简写（`hf:Qwen/Qwen3-8B`、`ms:...`、`mega:...`）。

## 2. 运行远端模型

```bash
# Python CLI：仓库 ID 由 --source 决定用哪个 Hub（auto 依次尝试 mega、hf、ms）
python -m megalm --model Qwen/Qwen3-8B --prompt "你好" --source hf

# 生产前端：先下载到本地，再用 --model 指向该目录
megalm-download hf:Qwen/Qwen3-8B --local-dir models/qwen3
megalm-server --model models/qwen3 --served-model-name qwen3

# 显式前缀，忽略 --source
python -m megalm --model ms:Qwen/Qwen3-8B --prompt "你好"
python -m megalm --model mega://models/acme/llm@main --prompt "你好"
```

Python CLI 的解析顺序：本地存在的目录 → 显式前缀 → `--source`（`auto` 时 mega →
hf → ms）。`auto` 会跳过失败的 Hub 并在最后汇总所有错误；网络受限时建议显式指定来源。
`megalm-server` 没有 `--source`，也没有 Hub ID 解析逻辑。

## 3. 下载但不启动

```bash
python -m megalm.download Qwen/Qwen3-8B --source hf
python -m megalm.download ms:Qwen/Qwen3-8B --local-dir model/qwen3
python -m megalm.download hf:Qwen/Qwen3-8B \
  --include '*.safetensors' config.json --include 'tokenizer*'
```

命令输出本地 checkpoint 目录；作为 `--model <目录>` 传给
`megalm-server`。不带参数时沿用历史默认 `Nanbeige/Nanbeige4.2-3B`。

## 4. 共享参数

三个命令都支持同一组参数：

| 参数 | 作用 |
| --- | --- |
| `--source {auto,mega,hf,ms,...}` | 裸仓库 ID 使用哪个 Hub（别名 megahub / huggingface / modelscope） |
| `--revision REV` | 分支、tag 或 commit；覆盖 URI 里的 `@rev` |
| `--cache-dir DIR` | 覆盖 SDK 缓存目录 |
| `--local-dir DIR` | 直接下载到指定目录（SDK 的 `local_dir`） |
| `--include GLOB` | 只下载匹配文件，可重复；传给 SDK 的 `allow_patterns` |
| `--exclude GLOB` | 跳过匹配文件，可重复；传给 SDK 的 `ignore_patterns` |
| `--offline` | 只用缓存中已有文件（`local_files_only`），不联网 |
| `--endpoint URL` | 覆盖 Hub 端点；优先用对应环境变量 |
| `--token TOKEN` | Hub token；多用户机器建议改用环境变量，避免出现在进程参数里 |

传输、断点续传、完整性校验、代理与缓存布局由各 SDK 负责，MegaLM 不重复实现；
如需限速请使用 SDK 的配置或前置代理。

> 注意：旧的 `python -m megalm.download` 内置限速下载器（硬上限 5 MiB/s、
> `HF_ENDPOINT` 等参数）已被官方 SDK 取代；`--rate-limit` / `--verify-sha256`
> 不再提供，校验与传输交给 SDK。HTTP 端点偏好请设置 `HF_ENDPOINT` 等环境变量。

## 5. 代码落点

| 层 | 文件 |
| --- | --- |
| 引用解析 / SDK 适配 / CLI 参数 | `src/megalm/model_source.py` |
| 下载命令 | `src/megalm/download.py` |
| 引擎 CLI | `src/megalm/__main__.py`、`src/megalm/bootstrap.py` |
| 服务 CLI | `src/megalm/gateway/server.py` |
| 单元测试（全 mock，无网络） | `tests/test_model_source.py` |

验证：

```bash
PYTHONPATH=src python -m pytest tests/test_model_source.py -q
```
