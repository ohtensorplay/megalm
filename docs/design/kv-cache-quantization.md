# KV Cache 量化

适用范围：`megalm.runtime.kv_quant` 与 attention kernel 的量化路径。默认 `auto` 不量化。

## 1. scheme

| `kv_cache_dtype` | mode | scale 粒度 | 每元素字节 | 硬件 |
| --- | --- | --- | --- | --- |
| `auto` | 无 | — | 2（fp16 / bf16） | 所有 |
| `fp8_e4m3fn` | `FP8_PER_TENSOR` | 每层每 tensor 一个标量（K / V 各一） | 1 | SM89+ |
| `fp8_e5m2` | `FP8_PER_TENSOR` | 同上 | 1 | SM86+ |
| `fp8_per_head` | `FP8_PER_HEAD` | 每 KV head 一个校准 scale | 1 | SM86+ |
| `fp8_per_token_head` | `FP8_PER_TOKEN_HEAD` | 每 (token, KV head) 动态 scale | 1 + 4B/head | SM86+ |
| `int4_per_token_head` | `INT4_PER_TOKEN_HEAD` | 每 (token, KV head) 动态 scale + zero point | 0.5 + 4B/head | SM86+ |
| `nvfp4` | `NVFP4` | 每 16 元素一个 E4M3 block scale | 0.5 + 0.0625 | SM86+（软件解码） |

统一数学：写 cache 时 `x/scale → fp8` 或 4-bit packing，attention 读取时按 scale 反量化或走原生 fp8。scale 默认 `1.0`，可由校准值覆盖。TensorPlay 原生 affine fp8 op 与 Triton / 手写编解码实现同一数学。

## 2. 不变量与布局

- `KVQuantConfig` 挂在 cache 张量上（`PagedKVCache.layer()` 的每层视图，以及 Qwen3.5 / Qwen3-VL / Inkling-2 的 flat view）。kernel wrapper 从张量读元数据，不在 model forward 调用链逐层传参。
- 动态 scale（mode 3 / 4）以 fp32 内嵌在 cache 行 data 区之后：`[data | scale(4B)]`，data 区长度 `data_dim`（fp8 为 `head_dim`，int4 为 `head_dim/2`，nvfp4 为 `head_dim/2 + head_dim/16`）。scale 视图用 `as_strided` 从同一 storage 切出，shape `(pages, page_size, heads)`，页复制自动携带 scale。
- `MODE=0` 走原 fp16 代码，产出 bit-identical 的旧行为。
- 内存预算按真实 `content_dim`（含 4B scale 与 4-bit packing）。offload identity 含完整量化签名；Radix key 不含量化信息。
- `index_select` 不支持 fp8 / 打包 4-bit；Python gather 先反量化再 gather。
- 新块 zeroing 由 `tensorplay.zeros` 保证；内嵌 scale 初始化为 `1.0`，避免未写槽解成 NaN / Inf。

## 3. scale 来源

1. 静态标量（mode 1）：config 或 CLI，默认 `1.0`。
2. per-head 校准（mode 2）：checkpoint 中 `[H,1,1]` scale 由 `loading.safetensors.read_kv_scale_vectors` 读取；K / V reshape 成 `[num_kv_heads]`，Q 按 GQA 组内 amax 缩减，多层再逐元素 amax 得到引擎级单一向量。checkpoint scale 优先于 CLI 标量；权重加载用 `exclude_suffixes=KV_SCALE_SUFFIXES` 跳过这些元数据张量。
3. 动态（mode 3 / 4）：写 cache 时现算。fp8 `scale = max(absmax / QMAX, 1e-6)`；int4 非对称 `[0,15]`，`scale = max((max-min)/15, 1e-6)`，zero point 隐写在 fp32 scale 的最低 4 位尾数，读写双方都做 `& -16`。
4. NVFP4（mode 5）：每 16 元素一个 E4M3 block scale，`6.0` 为 E2M1 最大值，另有 checkpoint 全局 scale。

运行时 TP 恒为 1，没有 TP scale 切片与复制分支。

## 4. Q 量化与原生 FP8 attention

`--kv-cache-q-dtype`、`--kv-cache-q-scale`、`--kv-cache-q-group`、`--native-fp8-attn`：`q = fp8(x / q_scale)`，per-head group 按 KV head（覆盖该组全部 query head）。score 折叠 `bmm1_scale = softmax_scale * q_scale * k_scale`，输出折叠 `v_scale`。prefill kernel 的 `tl.dot` 两个 operand 保持 fp8（P 也 cast 到 fp8）；decode 单 token 无 M 维，走 fp32 累加。

硬件门禁：FA3 / FA4 原生 fp8 需 SM90 / SM100；Triton fp8 MMA 需 SM89+。SM86 只有 E5M2 存储与反量化到 fp16 的路径，`--native-fp8-attn` 明确报错并建议关闭。未量化 / 非原生路径不受影响。

## 5. kernel 路径

| 路径 | 写入量化 | 读取反量化 |
| --- | --- | --- |
| Qwen3 / Qwen3.5 / Qwen3-VL 标准注意力 | `rotary_cache.py` | `paged_decode.py`、`paged_prefill.py` |
| Qwen3.5 / Inkling-2 packed QKV | `qwen35.py::active_kv_write` | 同上 |
| Qwen2.5-Omni 文本注意力 | `rotary_cache.py` | `paged_decode.py`、`paged_prefill.py` |
| Inkling-2 相对注意力 | `qwen35.py::active_kv_write` | `inkling2_paged_decode.py` + 回退路径 |
| CPU / fallback | `fallback.py` + TensorPlay 原生 op | `fallback.py`、`paged_prefill_sdpa.py` |

## 6. 配置

```text
--kv-cache-dtype {auto,fp8_e4m3fn,fp8_e5m2,fp8_per_head,fp8_per_token_head,int4_per_token_head,nvfp4}
--kv-cache-key-scale FLOAT
--kv-cache-value-scale FLOAT
--kv-cache-q-dtype {auto,fp8_e4m3fn,fp8_e5m2}
--kv-cache-q-scale FLOAT
--kv-cache-q-group {per_tensor,per_kv_head}
--native-fp8-attn
--nvfp4-global-scale FLOAT
--int4-use-rht
```

`CacheStats.snapshot()["kv_quant"]` 暴露完整配置，由快照渲染器发布到指标。

## 7. INT4 随机 Hadamard 旋转

`--int4-use-rht`：写 cache 前对 K / V 做 `H @ (signs ⊙ x)`（`H@H = d·I`，signs 为确定性 ±1），摊平离群维度。读 attention 时把 Q 旋转到同一空间、`softmax_scale /= head_dim`，输出逆旋转 `((y@H) ⊙ signs)/head_dim`。Python fallback / gather 路径在反量化时逆旋转 K / V。`head_dim` 必须是 2 的幂；H 与 signs 按 `(head_dim, device)` 缓存。outlier 重的 KV 上误差下降约 2×。

## 8. 未做

FA3 / FA4 CUDA 原生 fp8 attention 后端（当前用 Triton），TP 分片 scale 加载。两者需要对应后端 kernel 与多卡运行时。

## 9. 验证

`tests/test_kv_quant.py` 覆盖配置校验、cache 几何 / 元数据、原生量化往返、fallback 写入 + 解码、Triton 解码 / 写入 / prefill 的 5 种 mode，并与 fp16 kernel 对照量化噪声（E5M2 典型最大误差 < 0.2；4-bit 量级 0.05–0.8）。另含 RHT 矩阵正交性、正逆变换往返、引擎门禁与 outlier KV 精度对照。
