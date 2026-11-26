# DiffusionGemma 高性能执行路径：代码逐层讲解

本文按**一次去噪 step 的调用链**，拆解 26B AWQ-INT4 checkpoint 在原生执行器中的实现。
核心思想只有一句：把“每个算子一次 kernel、每个专家一次 GEMM”改成“对同一批大张量只扫一遍、所有专家一次 launch”。

验收数据见 `devdoc/DIFFUSIONGEMMA_ACCEPTANCE.md`：同一 RTX 3090、同一 checkpoint、同样 512 token 预算下，
checkpoint 默认采样为 **596.64 tok/s vs vLLM 0.34 的 486.82 tok/s（+22.56%）**；
两边都固定 8 步/画布（各 16 次 decoder call）时为 **535.60 vs 493.74 tok/s（+8.48%）**。

## 调用链总览

```
DiffusionGemmaGenerator.generate                  generate.py
  └─ DiffusionGemmaForBlockDiffusion.denoise       model.py
       ├─ 预计算本 step 的 RoPE（30 层共用）
       └─ DiffusionLayer × 30
            ├─ DiffusionAttention
            │    ├─ finalize_qkv 合并后的单次 Marlin 投影    marlin_moe.py
            │    ├─ prepare_diffusion_qkv（RMSNorm+RoPE+V norm） diffusion_attention.py
            │    └─ diffusion_attention（在线 softmax GQA）   diffusion_attention.py
            └─ DiffusionSparseMoE
                 └─ grouped_marlin_moe（路由+up+GELU+down+加权） marlin_moe.py + csrc/marlin_moe.cu
  └─ sample_diffusion_logits（softcap/温度/采样/熵/自条件一次扫描） diffusion_sample.py
  └─ StableAndConfidentStopping / accept_and_renoise
```

---

## 1. `generate.py`：采样循环接线

CUDA 且启用 Triton 时，整个采样走融合路径：

```python
sample_cuda = device.type == "cuda" and self.model.kernel_policy.use_triton
if sample_cuda:
    from megalm.kernels.diffusion_sample import sample_diffusion_logits
    softcap = self.model.config.text.final_logit_softcapping or 0.0
```

```python
if sample_cuda:
    logits = self.model.denoise(
        canvas, caches,
        self_conditioning_probs=self_conditioning,   # 不再对上一轮 logits 重做 softmax
        return_raw_logits=True,                      # 模型吐原始 bf16 logits
    )
    uniforms = tensorplay.rand((batch, canvas_length), ...)
    denoiser, new_argmax, entropy, processed = sample_diffusion_logits(
        logits, uniforms, temperature, self.model.dtype, softcap=softcap,
    )
else:
    # 旧路径：softmax -> multinomial -> argmax -> entropy -> self-conditioning
```

另外 `stopped_rows` 改为 Python list 缓存，每个 step 只调用一次 `stopped.tolist()`；
旧代码里的 `bool(stopped.any())`、`tolist()`、`bool(stopped.all())` 是三次 host sync。

`model.py` 侧对应两个接口变化（`DiffusionGemmaModel.denoise`）：

```python
soft = self_conditioning_probs
if self_conditioning_logits is not None:
    soft = softmax(self_conditioning_logits.float(), dim=-1).to(self.dtype)
...
logits = F.linear(hidden, self.lm_head()).float()
cap = self.config.text.final_logit_softcapping
if cap is not None: logits = cap * tanh(logits / cap)
if return_raw_logits:
    return logits            # 交给融合采样器
return logits.float()
```

并预计算一次 RoPE 供所有层复用：

```python
rotations = {kind: rope(flat_positions) for kind, rope in self.rotary.items()}
for layer in self.layers:
    hidden = layer.denoise(..., rotation=rotations[layer.self_attn.kind])
```

---

## 2. `diffusion_sample.py`：三个 Triton kernel 完成一次 logits 扫描

入口：`sample_diffusion_logits(logits, uniforms, temperature, conditioning_dtype, softcap)`，
返回 `(sampled, argmax, entropy, conditioning)`。

logits 形状为 `[rows, VOCAB]`，26B 的 `VOCAB=262144`，fp32 下约 268MB。
旧实现把这样一个张量读 4~5 遍；新实现只完整读一遍。

### 2.1 `_summarize_logits[(rows, chunks)]`

每个 `(token, 词表分块)` 一个 program，`BLOCK=4096`：

```python
x = tl.load(logits + row*VOCAB + indices, valid, other=0.0)
x = tl.where(valid, _scale_logits(x, inv_temperature, SOFTCAP), -inf)
rounded = x.to(dtype).to(tl.float32)             # 显式模拟 BF16 舍入
tl.store(conditioning + row*VOCAB + indices, rounded, valid)

maximum = tl.max(x, axis=0)
shifted = tl.where(maximum == -inf, -inf, x - maximum)
weight  = tl.exp(shifted)
entropy_mass = weight * tl.where(shifted == -inf, 0.0, -shifted)
winner = tl.min(tl.where(valid & (x == maximum), indices, VOCAB))
round_mass = tl.exp(round_max - rounded)

tl.store(parts + base, maximum)
tl.store(parts + base + CHUNKS,     tl.sum(weight))        # softmax 分母
tl.store(parts + base + CHUNKS*2,   tl.sum(entropy_mass))  # 熵分子
tl.store(parts + base + CHUNKS*3,   tl.sum(round_mass))    # 舍入后分母
tl.store(winners + row*CHUNKS + chunk, winner)
```

`_scale_logits` 用 `libdevice.tanh` 精确实现 `softcap * tanh(x/softcap) / temperature`，
并把每个 token 的自条件值先存下来，后面复用。

### 2.2 `_sample_logits[(rows,)]`

每个 token 一个 program，把该 token 所有 chunk 的统计量放进一个向量：

```python
maximum = tl.max(maxima)
masses = load(parts + base + CHUNKS) * tl.exp(maxima - maximum)
total  = tl.sum(masses)
ent = log(total) + sum(terms*scale + masses*offsets) / total
argmax = tl.min(tl.where(maxima == maximum, ids, VOCAB))

# 逆 CDF：先定位 uniform 落在哪个 chunk
target = uniforms[row] * total
prefix = tl.cumsum(masses)
selected = tl.min(tl.where((prefix > target) & valid, chunk, CHUNKS))
lower = tl.max(tl.where(chunk < selected, prefix, 0.0))
local_target = max(target - lower, 0) * exp(maximum - local_max)

# 只重扫被选中的那个 BLOCK
weights = tl.where(indices < VOCAB, tl.exp(scaled - local_max), 0.0)
local_prefix = tl.cumsum(weights)
token = tl.min(tl.where((local_prefix > local_target) & (indices < VOCAB), indices, VOCAB))
```

这是**精确的全词表类别采样**：不物化 softmax，只重新读一个 4096 宽的 tile；
argmax、熵、自条件归一化分母都在这一个 kernel 里产出。

### 2.3 `_normalize_conditioning[(rows, chunks)]`

把 `_summarize_logits` 写入的舍入 logits 就地变成概率：

```python
tl.store(conditioning + offset, tl.exp(x - maximum) * inverse, indices < VOCAB)
```

下一轮 `denoise(self_conditioning_probs=...)` 直接消费它，不再重算 softmax。

数值验证：`tests/test_diffusiongemma_cuda_kernels.py::test_diffusion_sampling_matches_full_vocab_reference`
在 `width=262144` 上对采样 ID、argmax、熵、条件概率逐一和 torch double 参考对比。

---

## 3. `marlin_moe.cu` + `marlin_moe.py`：融合专家 GEMM

### 3.1 加载期：GPU 上直接 repack

`pack_experts(packed, scale_zero)` → `megalm_marlin_moe_pack`：

```cpp
pack_codes<<<...>>>(codes, packed, words, n, k);
pack_scales<<<...>>>(scale_zero, scales, scale_count);
```

`pack_codes` 每个线程按 Marlin 的 interleave `{0,2,4,6,1,3,5,7}` 把 8 个 nibble 拼成一个 `uint32`；
`pack_scales` 做 8×8 转置。对比之前在 CPU numpy 上打包（加载 45~70s），这里完全在 GPU 上完成。

### 3.2 路由对齐 `align_routes`

```cpp
const unsigned peers = __match_any_sync(active, expert);
if (lane == __ffs(peers) - 1) counts[warp][expert] += __popc(peers);
```

一个 block 内：CUB `BlockScan` 前缀和 + warp 级 `__match_any_sync`，两遍完成：

1. 统计每个专家的 route 数，为每个专家预留 `block_rows` 对齐的连续行区间（`expert_ids`）；
2. 第二遍把 route 的全局下标写进 `sorted`，尾部填 `routes` 作为 padding 标记。

替代了 Python 侧的 `argsort + bincount + .tolist()` host 同步循环。

### 3.3 内核配置：按设备和工作量选择，不硬编码

```cpp
constexpr Config kConfigs[] = {
  config<8,64>(), config<8,128>(), config<16,64>(), config<16,128>(),
  config<32,64>(), config<48,64>(), config<64,64>(),
};

// megalm_marlin_moe_prepare
cudaDeviceGetAttribute(&sms, ...MultiProcessorCount, device);
cudaDeviceGetAttribute(&shared, ...MaxSharedMemoryPerBlockOptin, device);
cudaFuncSetAttribute(...);
cudaOccupancyMaxActiveBlocksPerMultiprocessor(&resident, cfg.kernel, cfg.threads, cfg.shared);
grids[i] = sms * std::min(resident, i < 4 ? 4 : 2);
```

`config<Rows,TileK>()` 用 constexpr 计算 shared memory，设备放不下就把 `grids[i]=0`。
Python `_launch_config` 再按稀疏度选 `block_rows`：

```python
for block_rows in (8, 16, 32, 48, 64):
    if routes * 10 < experts * block_rows * 9:
        break
```

route 数不足以填满“专家 × 行块”时用更小的行块，减少 padding 浪费。

### 3.4 一次 MoE 前向的 kernel 序列

`megalm_marlin_moe`：

```cpp
cudaMemsetAsync(locks, 0, ...);
align_routes<<<1, 256>>>(...);
project(up_config,   x,          q13, s13, ..., stage, ...);   // gate/up GEMM
gelu_mul<<<...>>>(stage, activation, elements, intermediate);
project(down_config, activation, q2,   s2,  ..., stage, ...);  // down GEMM
reduce_routes<<<...>>>(stage, weights, output, tokens, top_k, hidden);
```

`project()` 调用的是从 vLLM 移植的 `marlin_moe_wna16::Marlin` 模板（`src/megalm/kernels/csrc/marlin/moe_template.h`），
并传入 `sorted / expert_ids / padded / top_k`，由内核自己按专家分块取权重。
**没有 128 次 Python 专家循环，也没有每层 256 次 GEMM launch。**

`reduce_routes` 用 FP32 累加 top-k 权重后再写回 bf16：

```cpp
float sum = 0.0f;
for (int slot = 0; slot < top_k; ++slot)
    sum += __bfloat162float(down[(route+slot)*hidden + column]) * weights[route+slot];
output[element] = __float2bfloat16(sum);
```

### 3.5 同一内核复用于所有投影

`marlin_project` 用 `align_dense` 造一个“单专家”布局，然后调同一个 `project()`：

```cpp
align_dense<<<...>>>(rows, block_rows, sorted, expert_ids, padded);
project(kConfigs[configuration], input, packed, scales, sorted, expert_ids,
        padded, output, temporary, locks, rows, n, k, 1, grid, stream);
```

注意 `cudaMemsetAsync(locks, 0, ...)`：vLLM 的 Marlin 屏障要求每次调用前清空 split-K 锁，
早期手工包装器缺少 FP32 `C_tmp` 和正确的 thread tile / grid 配置，导致 NaN 与随机大误差。

### 3.6 启用条件与回退

```python
def supports_marlin_moe(device, dtype, hidden, intermediate, experts, group_size):
    return (device.type == "cuda" and dtype == bfloat16 and group_size == 32
            and hidden % 128 == 0 and (2*intermediate) % 128 == 0
            and 0 < experts <= 256 and device.major >= 8)
```

`KernelPolicy.diffusion_marlin_moe: bool | None = None` 默认 auto：支持则尝试 `prepare_marlin_moe`，
编译失败静默回退 grouped Triton；设为 `True` 时失败直接抛错。

---

## 4. 注意力：QKV 合并 + 两个 Triton kernel

### 4.1 加载期合并 QKV

`DiffusionAttention.finalize_qkv()`：

```python
projections = [self.q_proj, self.k_proj]
if self.v_proj is not None: projections.append(self.v_proj)

self.qkv_marlin_q = tensorplay.cat(
    [p.int4.marlin_q.reshape(K // 16, -1) for p in projections], dim=-1)
self.qkv_marlin_s = tensorplay.cat(
    [p.int4.marlin_s.reshape(K // 32, -1) for p in projections], dim=-1)
```

关键性质：Marlin 的排列每 64 通道重复，所以把三个已打包权重的 packed 段直接拼接，
等价于“先拼完整权重再打包”。`load_weights` 末尾对所有层调用 `finalize_qkv()`。

前向变成一次 GEMM：

```python
qkv = marlin_project(hidden, self.qkv_marlin_q, self.qkv_marlin_s)
query = qkv[:, :query_width].view(tokens, heads, head_dim)
key   = qkv[:, query_width:key_end].view(tokens, kv_heads, head_dim)
value = qkv[:, key_end:].view(tokens, kv_heads, head_dim) if has_v else key
```

### 4.2 `prepare_diffusion_qkv`：RMSNorm + RoPE + V norm 三合一

一个 program 负责 `(token, head)`：

```python
first  = tl.load(src + offset).to(tl.float32)
second = tl.load(src + offset + HALF).to(tl.float32)
inverse = tl.rsqrt((sum(first*first + second*second)/(2*HALF)) + EPS)
first = (first * inverse * tl.load(weight + i).to(tl.float32)).to(dtype).to(tl.float32)
...
y0 = (first*c).to(dtype).to(tl.float32) - (second*s).to(dtype).to(tl.float32)
y1 = (second*c).to(dtype).to(tl.float32) + (first*s).to(dtype).to(tl.float32)
```

`enable_fp_fusion=False` 加显式 `.to(dtype)` 是为了逐位匹配 eager BF16 的舍入顺序，
因此融合后 tiny parity 仍然成立。

### 4.3 `diffusion_attention._attention`：在线 softmax GQA

```python
kv_head = head // (HEADS // KV_HEADS)          # 不 repeat_interleave
for start in range(key_start, key_end, BN):
    scores = tl.dot(query, tl.trans(keys)).to(query.dtype).to(tl.float32)
    visible = (ki < key_len) & (qi < query_len)
    if CAUSAL: visible &= ki <= q_position
    if WINDOW: visible &= ki > q_position - WINDOW
    scores = tl.where(visible, scores, -inf)
    maximum = tl.maximum(running_max, tl.max(scores, axis=1))
    alpha = tl.exp(running_max - finite_max)
    weights = tl.exp(scores - finite_max[:, None])
    acc = tl.dot(weights.to(query.dtype), values, acc * alpha[:, None])
```

没有 `[heads, seq, seq]` score 矩阵，也没有 K/V 的 `repeat_interleave` 复制。
仅在 `not self.softcap` 时走这条路径（26B 该字段为 0），否则保留原来精确的 softcap 实现。

---

## 5. 测试如何锁住数值正确性

`tests/test_diffusiongemma_cuda_kernels.py`：

| 测试 | 对照 |
|---|---|
| `test_grouped_marlin_repack_and_experts` | `pack_experts` 与 `pack_marlin_batch` 逐位相等；MoE 输出对照 torch FP32/FP64 手算，相对误差 < 0.5% |
| `test_diffusion_sampling_matches_full_vocab_reference` | 词表 262144 下采样 ID、argmax、熵、条件概率对照 torch double |
| `test_diffusion_attention_causal_and_bidirectional` | head_dim 256/512 × causal/bidirectional 对照 |
| `test_diffusion_prepares_strided_qkv` | strided 输入下 QKV 预处理对照 |

外加原有 tiny checkpoint 的 HF 逐 token parity（`test_diffusiongemma_generate_parity.py`）。

---

## 6. 为什么能从 212 tok/s 提到 596 tok/s

| 环节 | 早期实现 | 当前实现 |
|---|---|---|
| MoE | grouped Triton 2.5ms/层；per-expert Marlin 31ms/层 | vLLM 融合 Marlin：路由 + up + GELU + down + 加权一次 launch |
| QKV | 3 次 int4 GEMM + 3 次 norm/RoPE + `repeat_interleave` | 1 次 Marlin 投影 + 1 次融合 norm/RoPE + 连续 GQA |
| 采样 | softmax / multinomial / argmax / entropy / self-cond 各读一遍 268MB | 3 个 Triton kernel，对 logits 只完整读一遍，逆 CDF 只重扫一个 tile |

固定 8 步的公平对照仍有 +8.48%，说明收益不是靠少走去噪步数获得的。

**方法论教训**

1. GPU 优化要先看**大张量的读写遍数**和 **kernel launch 次数**，而不是单个算子的毫秒数；
   1ms 的 softmax 在 262144 词表上意味着 268MB 的显存往返。
2. 解决最大瓶颈后必须**重新 profile**：MoE 修好后，注意力和采样才成为可回收成本。
3. 有现成参考实现时，先读它的**调度结构**（融合内核、launch 配置、workspace），再抄数学；
   只修单 GEMM 的数值问题是拿不到参考性能的。
4. 性能结论要有公平对照：两边固定相同 decoder call 数，再报吞吐。
