# Qwen3-VL-8B AWQ int4：MegaLM vs SGLang（单卡 RTX 3090）

同一张 RTX 3090（sm86）、同一个 AWQ int4 checkpoint、同一个 512-token prompt、129 个贪心输出 token。两个引擎各自独占 GPU 运行，模型只加载一次，预热 1 轮后计时 3 轮，每轮在批次全部结束后同步一次。原始数字见 [qwen3vl-awq-sglang.json](qwen3vl-awq-sglang.json)，脚本见 [bench_qwen3vl_awq.py](bench_qwen3vl_awq.py) / [bench_qwen3vl_awq_sglang.py](bench_qwen3vl_awq_sglang.py)。

## 结果

吞吐 tok/s（越高越好，3 次均值）：

| batch | SGLang 0.5.23 | MegaLM（Marlin W4A16） | 差异 |
| --- | --- | --- | --- |
| 1 | 118.8 | 118.6 | −0.2% |
| 8 | 525.7 | 526.0 | +0.1% |
| 16 | 687.1 | **691.4** | **+0.6%** |

b1 的相位拆分（同一进程内 `gen=1` 与 `gen=129` 差分）：

| | prefill | decode 每步 |
| --- | --- | --- |
| MegaLM | **103.0 ms** | 7.65 ms |
| SGLang | 109.9 ms | **7.54 ms** |

MegaLM 的 prefill 快 6.3%，decode 每步慢 1.5%（每步约 0.11 ms，属 host 侧调度开销）；b8/b16 两项合计后 MegaLM 略快。

## 消融：块大小 vs 量化内核

两组改动容易混淆，分开记录。`chunk` 是 `prefill_chunk_size`，`decode-chunk` 是 `decode_chunk_steps`：

| 配置 | b1 | b8 | b16 |
| --- | --- | --- | --- |
| chunk 512，Triton int4 | 108.0 | 421.1 | 522.9 |
| chunk 512，Marlin（gate_up 仍走 Triton） | 117.4 | 475.2 | 592.0 |
| chunk 8192，Triton int4 | 103.7 | 487.7 | 660.4 |
| chunk 8192，Marlin（含 gate_up） | **118.6** | **526.0** | **691.4** |

结论：

- **b16 的多数提升来自 prefill 块大小**（522.9 → 660.4，+26%），不是内核。16 × 512 = 8192 token，512 的块会把整批 prefill 拆成 16 个低占用率的 step；8192 一次装下。8192 本来就是 `EngineConfig` 的默认值，表中 chunk 512 的两行是早期 harness 的覆盖值，不能当作引擎默认行为。
- Marlin 在固定块大小下带来 +7.8%（b8，488.0 → 526.0）和 +4.7%（b16，660.4 → 691.4）。
- b1 的 chunk 512 与 8192 两行同时受 `decode-chunk`（8 vs 1）影响，且 Triton 行三次实测为 103.9 / 101.0 / 106.2，方差大，**不应读出趋势**；b1 的块大小对单个 512-token prefill 没有影响。
- `decode-chunk` 在 1 与 8 之间对 b1 只差约 0.8%，本表最终配置用 1。

## b8 decode 逐 kernel（chunk 512，ms/步）

| MegaLM | ms | SGLang | ms |
| --- | --- | --- | --- |
| Marlin decode | 5.519 | Marlin decode | 5.276 |
| lm_head bf16 GEMM | 1.451 | lm_head GEMM | 1.431 |
| GQA decode | 1.026 | flashinfer paged prefill | 1.102 |
| GQA reduce | 0.092 | merge states | 0.118 |
| pair-SwiGLU | 0.189 | act_and_mul | 0.066 |
| add + rmsnorm | 0.153 | add + rmsnorm | 0.237 |
| QK-norm + RoPE + KV write | 0.123 | QK-norm + mrope + KV store | 0.200 |
| 其余 Marlin 变体 | 0.344 | elementwise | 0.075 |
| 合计 | 8.897 | 合计 | 8.505 |

两边的 Marlin 模板参数完全一致（decode `256,1,8,8,true,4,2`，prefill `256,4,16,4,false,4,2`）。MegaLM 在 attention（1.12 vs 1.22）与 norm（0.15 vs 0.24）上更快，SGLang 的 `act_and_mul` 比独立的 pair-SwiGLU 少一次启动。逐 kernel 合计差 0.39 ms/步，但端到端 wall time 相同，差值主要是 profiler 归因噪声。

## 正确性

- 贪心输出 `[3197, 279, 58751, 374, 3873, 11, 279, 5662]` 在 `MEGALM_MARLIN=0/1`、CUDA Graph 回放前后完全一致，并与 HuggingFace 贪心续写相同。
- Marlin vs Triton 在 M ∈ {9…512} 上最大相对误差 8.1e-3（约 1 个 bf16 ULP）。
- pair-SwiGLU 新核在 rows ∈ {1,2,7,8,17,64,129} 上最大相对误差 3.9e-3。
- 相关测试 62 项通过（`test_kernels_cuda` / `test_int4_linear` / `test_fallback_kernels` / `test_kernel_capabilities` / `test_inkling2_int4_cuda`）。

## 复现

```bash
# MegaLM（Marlin 需要 nvcc，首次运行编译并缓存）
PYTHONPATH=<tensorplay>:<megaLM/src> MEGALM_MARLIN=1 \
  python docs/benchmarks/bench_qwen3vl_awq.py --batches 1,8,16 --reps 3

# SGLang（在能 import sglang 的环境里）
python docs/benchmarks/bench_qwen3vl_awq_sglang.py --batches 1,8,16 --reps 3
```

## Caveat

- 单卡、单次会话、单组参数的结果，不外推到其他 GPU、上下文长度或并发。
- 这张卡上有另一个进程间歇占用 9–23 GB。只采纳空闲窗口内、重复稳定的数据；一次未预热的 SGLang 冒烟跑出 30.1 tok/s，确认是被抢占后丢弃，重跑为 119.2 tok/s。
- SGLang 关闭了 radix cache 以对齐 MegaLM 的 `enable_prefix_cache=False`；两边都未使用前缀复用。
- SGLang 使用 flashinfer 注意力与 CUDA Graph；MegaLM 使用 Triton paged attention 与 CUDA Graph。
- 权重保持 checkpoint 的 int4 形态与 scale，没有 bf16 常驻副本；`lm_head` 仍为 bf16，未量化。
- Marlin 路径由 `MEGALM_MARLIN=1` 开启，默认关闭；关闭时回落到 Triton int4 融合 GEMM。
