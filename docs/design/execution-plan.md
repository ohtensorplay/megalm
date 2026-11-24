# ExecutionPlan 与冷启动

适用范围：所有模型与启动路径。请求热路径不选择 kernel、不做 autotune。

## 1. ModelSpec

adapter 只声明模型事实：

```text
ModelSpec
├─ architecture identity
├─ layer sequence（full attention / gated delta net / dense FFN / MoE）
├─ position schema（RoPE / MRoPE）
├─ state schema（paged KV / conv state / recurrent state / vision embeddings）
├─ weight mapping 与 packing 规则
└─ parallel constraints
```

Scheduler、Cache Manager、Gateway 中不按模型名分支。原生 TensorPlay 模型是唯一热路径实现；运行时不依赖 Transformers model 实现。

## 2. 冷启动状态机

```text
DISCOVER → DRAFT → LOAD/PACK → ALLOCATE/BIND → WARM/CAPTURE → SEAL → READY
              │                                      │
              └─ provider rejected ◄──────────────────┘
                         │
                  DESTROY → portable DRAFT
```

| 阶段 | 内容 | 禁止 |
| --- | --- | --- |
| `DISCOVER` | 读 `ModelSpec`、checkpoint manifest、GPU / 驱动 / 工具链能力、已有 artifact | 加载完整权重、在线 benchmark |
| `DRAFT` | 选 backend id、variant / tactic、shape guard、权重布局、workspace 公式、Graph buckets、显存上界 | 创建进程内 callable |
| `LOAD/PACK` | checkpoint 流式装入最终布局 | 为 fallback 常驻两套专家权重 |
| `ALLOCATE/BIND` | 先扣权重 / workspace / Graph / 安全余量，剩余交给 Cache Manager；算子族绑定 typed 函数 | registry 查询、字符串分派、lazy import、`**kwargs` ABI |
| `WARM/CAPTURE` | 编译已选 Triton variant，验证 eager 与每个 CUDA Graph bucket，记录静态显存 | 请求路径同步 |
| `SEAL` | 生成带 digest 的不可变 `ExecutionPlan` | —— |

`PlanDraft` 只存在于启动进程。`PlanDraft.verify_sealed()` 核对 model digest、hardware、dtype、kernel selection、binding digest 与 workspace；`SEAL` 前 Replica 不报告 ready。

## 3. ExecutionPlan

```text
ExecutionPlan
├─ model_digest / hardware / topology / dtype / workload
├─ weights（分片与打包布局）
├─ kernels（KernelSelection + kernel_binding_digest）
├─ memory（权重、workspace、Graph、安全余量）
├─ cache（分页几何、页面大小、容量）
├─ collectives / cuda_graph_buckets / vision_backend
└─ plan_digest / schema_version
```

`KernelSelection` 只含可序列化的身份、布局、guard、资源事实；进程内 callable 不进入协议。每个算子族在 `BIND` 时解析为自己的 typed binding 并反查身份，没有通用 Tensor ABI。单一有序文本 stream 可跨同几何层共享 workspace；Vision 等独立 stream 有独立 workspace。所有 rank 验证同一 digest。

## 4. 算子 provider

TensorPlay 保留 reference / fallback，Triton 是 Ampere 路径的可移植基线。CUDA、CUTLASS、CuTe、TileLang、FlashInfer、sgl_kernel 是可删除的候选 provider，不形成一层通用框架。

候选只在冷路径按 GPU 架构、dtype / 量化、权重布局、shape guard、workspace 与 Graph 能力探测。生产准入条件：

1. reference 数值或 sampling 语义对齐；
2. forward 内无动态分配、主机同步、运行时编译；
3. 静态地址下可完成 CUDA Graph capture / replay；
4. 目标模型完整层路径与固定 serving workload 上稳定胜过现有后端；
5. 不支持的硬件、dtype、shape 在执行前回退，不依赖运行时异常。

孤立 GEMM microbenchmark 只用于定位。每个算子族独立源文件与 typed wrapper；生产绑定、测试、benchmark 直接导入具体模块。共享实现下沉到明确组件，不用 re-export 壳遮盖巨文件。

## 5. 失败与回退

首选 provider 在 ready 前失败：销毁本次草案的 GPU 对象，整组 rank 最多重试一次 portable `DRAFT`，再失败则启动失败。shape guard 未命中进入计划内已绑定的 fallback。ready 后 kernel fault 不做单请求热切换；整组重建并用已隔离该 artifact 的新草案。

原始 benchmark、profile trace、编译产物留在 CI artifact；已审核胜者与证据 digest 写入 kernel evidence lock。运行时读取 lock，不生成或搜索 tactic。

## 6. 验证

`contracts/plan.py` 的 schema 与字段合法性；`startup.py` 的阶段跃迁、provider 失败回退、GPU 对象释放；`PlanDraft.verify_sealed()` 的字段漂移；typed binding 在 `BIND` 时的身份反查。


