# Multi-LoRA

适用范围：文本 generic causal LM（`GenericCausalLM` 家族：Llama、Mistral、Qwen2、Yi、DeepSeek、Phi3、InternLM2/3、Seed-OSS、Granite、ERNIE 4.5 dense）。不支持 MoE 专家投影、embedding / `lm_head`、视觉与 Omni tower、多模态请求、per-request `scale`。

## 1. 使用

```bash
megalm-server --model ./model --served-model-name megalm \
  --enable-lora --lora-path /adapters/sql --lora-path /adapters/style \
  --max-lora-rank 16 --max-loras 4
```

```bash
curl http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"megalm","messages":[{"role":"user","content":"..."}],
       "lora_request":{"lora_name":"sql","lora_int_id":1}}'
```

`{"lora_name":"sql"}` 与 `{"lora":{"id":1}}` 也由 `LoRARequest.from_payload` 归一化。不携带字段的请求用基座权重。

| 方法 | 端点 | 请求体 |
| --- | --- | --- |
| POST | `/v1/load_lora_adapter` | `{"lora_path":"...","lora_name":"sql","lora_id":1}` |
| POST | `/v1/unload_lora_adapter` | `{"lora_name":"sql"}` 或 `{"lora_id":1}` |
| GET | `/api/models/loras` | — |
| POST | `/api/models/loras` | `{"path":"...","name":"sql"}` |
| DELETE | `/api/models/loras/{name_or_id}` | — |

`--lora-allowed-root` 可重复指定，限制运行时加载目录；startup `--lora-path` 隐式允许。未设置时接受任意本地路径。重复 id / name、卸载被在途请求 pin 住的 adapter 在执行前拒绝。

客户端错误码：

| code | HTTP | 可重试 | 原因 |
| --- | --- | --- | --- |
| `lora_disabled` | 400 | 否 | 未以 `--enable-lora` 启动 |
| `lora_invalid` | 400 | 否 | scale / rank / geometry 非法或 PEFT 特性不支持 |
| `lora_adapter_not_found` | 404 | 否 | 名称或 id 未知 |
| `lora_adapter_busy` | 409 | 是 | 重复注册或被 pin 时卸载 |
| `lora_slots_exhausted` | 429 | 是 | 全部 GPU slot 被 pin |
| `lora_unavailable` | 503 | 是 | 其他运行时失败 |

错误 detail 携带 `code` / `message` / `retryable`，终态事件带同一 code；嵌入式与进程拓扑下映射一致。

## 2. adapter 格式

```text
adapter/
├── adapter_config.json       # r, lora_alpha, target_modules, use_rslora
└── adapter_model.safetensors # ...lora_A.weight / ...lora_B.weight
```

`bias`、`use_dora`、`modules_to_save` 拒绝。`r` 为 1/2/4/8/16/32/64/128/256 且不超过 `max_lora_rank`。支持的 target：`q_proj`、`k_proj`、`v_proj`、`o_proj`、`gate_proj`、`up_proj`、`down_proj`，packed target 的所有 slice 必须齐全。缩放 `alpha/r`（rsLoRA 为 `alpha/sqrt(r)`）在加载时折进 B；A 为 `[rank, in]`，B 为 `[out, rank]`。

## 3. 实现

1. `contracts/lora.py` 定义 `LoRARequest`、`LoRAConfig`、`LoRAAdapterSpec` 与 packed-target 映射；`InferenceRequest` 携带不可变 `lora_request`。
2. `loading/lora.py` 用有界内存 reader 流式加载；q/k/v 打包成一对 `qkv_proj` A/B，gate/up 打包成 `gate_up_proj`。每个逻辑 slice 独立 A；B 为块对角，便于多 adapter 共用一个 bank。
3. `runtime/lora.py` 按 target 分配固定形状 GPU bank `[max_loras, rank, in]` / `[max_loras, out, rank]`，通过 `GenericCausalLM.bind_lora` 绑定到 dense 层。
4. `kernels/lora.py` 实现 BGMV：每 token 选 slot，两阶段 Triton shrink / expand 把 `(x @ A[slot]) @ B[slot]` 加进基座投影输出。CPU、strided 张量与数值参考走 TensorPlay fallback。
5. `LoRAManager` 在 CPU 注册 adapter，激活到 GPU slot 并 LRU 复用；slot 在请求生命周期内 pin。激活先校验几何再淘汰 LRU victim，bank 写失败则恢复 victim；registry 与 slot 表由可重入锁保护。
6. scheduler 把逐序列 slot 展开成逐 token 的 `ExecutionBatch.lora_slots`；同一 forward 可混用不同 adapter。全基座 batch 的 slot 张量为 `None`，不启动 LoRA kernel。

## 4. 边界

LoRA 改变 K/V 投影，adapter 身份进入 radix namespace：adapter A 算出的 KV 页不会给基座或 adapter B 复用。LoRA batch 退出 CUDA Graph decode 与 ngram / MTP 投机路径；captured graph 保持基座语义。未启用时无 bank、无 hook、batch 无 slot 张量。

## 5. 指标

`EngineMetrics`：`lora_requests`、`lora_steps`、`lora_tokens`、`lora_eager_steps`、`lora_fallback_steps`。`LoRAManager.stats()`：`loads`、`failed_loads`、`unloads`、`activations`、`evictions`、`pins`、`rejections` 与当前 slot 使用量。日志：`engine.lora_loaded`、`engine.lora_unloaded`、`engine.lora_load_failed`、`engine.lora_unload_failed`、`engine.lora_rejected`。`lora_eager_steps` 统计本可用基座 graph 但主动跳过的 pass；`lora_fallback_steps` 统计走 TensorPlay kernel 的 pass。

## 6. 基准

`scripts/bench_lora.py` 测基座与 LoRA decode 延迟。无 `--model-path` 时构造小型合成 llama checkpoint（适合回归，不是生产吞吐数字）。4 层合成模型的记录结果在 `docs/benchmarks/lora-benchmark.json`：

| 模式 | 16 token 中位延迟 |
| --- | --- |
| LoRA disabled | 27.1 ms |
| LoRA enabled，基座请求 | 27.4 ms |
| LoRA enabled，1 个活跃 adapter | 40.9 ms |

## 7. 验证

```bash
pytest tests/test_lora_contract.py
pytest tests/test_lora_runtime_cpu.py
pytest tests/test_lora_kernels_cuda.py
pytest tests/test_lora_engine_integration.py
```
