# 编译后端

`compile_backend="stax"` 是 opt-in 的冷路径 prefill lowering provider；默认 `"none"`。手写 Triton / CUDA kernel 仍是默认路径。Stax 不替代 decode CUDA Graph replay。

## 1. 接入位置

```text
model load → LoRA bind → Stax region probe → plan seal / CUDA Graph capture → serving
```

`Engine.__init__` 在权重与 LoRA 加载后、`DecodeGraphRunner` 捕获 bucket 前调用 `megalm.runtime.compile_backend.compile_model`。被拒的 region 保持 eager；日志事件 `engine.compile`。

## 2. shape 策略

每个精确 prefill token 数编译一个 artifact，不用 `dynamic=True`。当前 TensorPlay 构建上符号形状会保留见过的最大形状：小调用返回超大张量，大调用被截断；`dynamic_shapes` 在形状变化后触发非法内存访问。因此只分派编译过的精确尺寸。

- `--compile-buckets`：要编译的 token 数。默认取 `--prefill-chunk-size`。
- `--compile-min-tokens`：编译下限，默认 `1024`。
- 其他 batch size、`decode=True`、3-D 输入、admission 后带 adapter 的 module 都走 eager。decode CUDA Graph bucket 不含编译 region。
- engine 默认 `--prefill-chunk-size` 为 512，低于编译下限；仅加 `--compile-backend stax` 不会编译任何 region，原因记录为下限。编译 bucket 必须大于最大 CUDA Graph decode batch（默认 16）且不超过 `--max-num-batched-tokens`。

## 3. 准入门禁

对每个候选 module 与 bucket：

1. **native lowering**：`tensorplay.compile(fn, backend="stax", strict_native=True)`。`Stax lowering failed` 视为拒绝，不静默回退 graph executor。
2. **parity**：与 eager kernel 在 bucket 形状上比较，bf16 相对误差 `2e-2`，fp16 `5e-3`，或 `--compile-tolerance`。
3. **shape**：编译结果形状必须与 eager 完全相同。
4. **capture**：artifact 在带 graph-private pool 的临时 `tensorplay.cuda.CUDAGraph` 中捕获并 replay。
5. **speed**：七次成对 eager / compiled 计时的中位数达到 `--compile-min-speedup`（默认 `1.05`）。一个 bucket 通过即接纳该 region；低于门禁的 bucket 保持 eager。

每个 region 至少一个 bucket 通过才会被接纳；失败原因按 region 记录在 runtime snapshot。

## 4. 权重与产物共享

权重以运行时输入（`requires_grad` 的 storage-sharing alias）传入编译函数；否则 Stax 把非 grad 张量折进 artifact，既复制权重又把计算绑定到启动时的值。运行时输入带来：

- 相同 `(width, intermediate)` 与 bucket 的所有层共用一个 artifact；28 层 dense 模型只编译一个，不是 28 个。
- 原地权重更新可见；替换权重张量对象需要重新编译。
- 同形状 module 复用一次准入决定，省去逐层计时噪声。每个新增 module 仍在最小可接受 bucket 上做一次 parity 检查。

## 5. 显存与启动预算

一个 bucket 持有 0.4–0.9 GB workspace 与 pool（权重不复制）。同形状层不额外增加。

- `--compile-max-artifact-bytes`（默认 2 GiB）限制整次 pass 观测到的设备显存增长。超限后停止，后续 region 保持 eager；探测中即将超限的 region 被拒并释放 artifact。检查基于观测增长，单个 artifact 仍可能先占满再触发检查，启用时要留显存余量。
- `--compile-startup-budget`（默认 `0`，不限）在 N 秒后停止探测。
- 首个 artifact 在 RTX 3090 上约 1–5.5 s；后续 bucket 更便宜，同形状层约 1 ms。
- 首个 Stax compile 会加载约 290 MB 运行时且不归还；全部拒绝的 pass 仍付出这部分启动成本。

被拒 artifact 丢弃并在下一个 region 前 flush allocator cache。

## 6. 覆盖范围

| region | 候选 | 当前构建实测 |
| --- | --- | --- |
| `dense_mlp`（默认） | SwiGLU / SiLU MLP：`GenericCausalLM` / Qwen3 `DenseMLP`、Qwen3.5 `PackedDenseMLP` 与 `PackedSharedExpertMLP`、有偏 / 无偏 `PackedSwiGLU` | parity 通过、显存有界，但 0.99–1.00×；默认门禁拒绝 |
| `attn_proj`（显式开启） | 文本注意力的 `qkv_proj` / `o_proj` | parity 通过，~1.00×；默认门禁拒绝 |
| `vision_mlp`（显式开启） | `vision` / `tower` / `encoder` 下的 `PackedSwiGLU` MLP | 2048 bucket 上 0.83×；默认门禁拒绝 |

故意跳过：MoE routed expert（grouped INT4 / GEMM kernel）、INT4 投影、未显式选择的 vision tower、自带 `_linear` backend 的 module、带活跃 adapter 的 module。

实测在单卡 RTX 3090、bf16、`tensorplay.inference_mode()` 下取得，含首次调用与重复调用的显存；数值随 GPU 负载有几个百分点波动。

| layer shape | bucket | compiled / eager | artifact | pass growth |
| --- | --- | ---: | ---: | ---: |
| 1024→2×2816→1024 | 512, 1024, 2048, 4096 | 0.987–0.999× | 395 MB | 382 MB |
| 2048→2×5632→2048 | 1024, 2048 | 0.997–0.999× | 511 MB | 515 MB |
| 4096→2×12288→4096 | 1024 | 1.000× | 889 MB | 889 MB |
| 3584→2×4608→3584（vision） | 2048 | 0.826× | 545 MB | 724 MB |

## 7. 启用与回滚

启用后的常见结果是 `accepted: 0` 加逐 region 原因（如 `1024: speedup 0.998 below 1.050`），表示手写路径至少一样快。诊断时把门禁降到 0 看实际比例；serving 保持 1.05，因为 break-even artifact 要付 0.4–0.9 GB 显存。

```bash
megalm-server --model model --compile-backend stax --prefill-chunk-size 1024 \
  --compile-regions dense_mlp

megalm-server --model model --compile-backend stax \
  --compile-regions dense_mlp,attn_proj,vision_mlp
```

> `--compile-buckets` / `--compile-min-speedup` / `--compile-max-artifact-bytes`
> / `--compile-startup-budget` 是离线编译工具的调优参数，Rust serving CLI
> 当前只暴露 `--compile-backend` 与 `--compile-regions`；其余仍由工具的
> 配置文件或 Python 侧入口控制。

回滚用 `--compile-backend none`，不改权重、cache 布局或 plan schema。runtime snapshot 的 `compile` 报告 backend、attempted / accepted、artifact 与 resident bytes，以及每个 region 的 bucket、parity、speedup、dispatch threshold、`runtime_calls`、`runtime_fallbacks`。

## 8. 运行时失败

编译调用有 guard：同步失败或输出形状错误会退役该 bucket 并返回 eager 结果。首次失败记 `engine.compile.fallback` 并计入 snapshot。异步设备 fault（后续同步点才暴露的非法内存访问）不在 guard 范围内，这是所有 CUDA kernel 调用的固有属性；准入门禁用于降低概率，不用于恢复。

## 9. 限制

- 当前构建没有盈利 region：dense MLP 0.99–1.00×，attention 投影 ~1.00×，vision MLP 0.83×，默认全部拒绝。MoE routed expert 与 decode GEMV 保持手写 kernel。
- 静态 bucket 之外的可变 prefill batch 走 eager；稳定 chunk size 的 chunked prefill 是目标形态。
- provider 仍是 runtime capability，但其 admitted artifact 的 identity 由 `ExecutionPlan.compiled`（`CompiledPlan`）封印：digest 绑定 provider 源码与策略，regions/buckets 限定准入范围，artifact 显存计入 `memory.workspace_bytes`。`PlanDraft.verify_sealed` 与 `plan_digest` 现在覆盖 artifact 漂移；默认 `backend=none` 时该项为空，热路径零成本。
- 拆分 `gate_up` 成两个 weight view 会让 artifact 降到 0.82×；交错布局是目前最好的形式。
- TensorPlay 的 Stax 拒绝 `F.silu(..., inplace=False)`；编译变体用 `gate * gate.sigmoid() * up`。

## 10. 验证

`tests/test_compile_backend_cuda.py` 在真实 Qwen3 `DenseMLP` 上复现准入（parity、prefill 分派、decode 绕过）；后续 TensorPlay 构建开始盈利时测试仍应保持通过。
