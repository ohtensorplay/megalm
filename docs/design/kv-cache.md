# KV Cache

适用范围：分页 KV、Radix 前缀缓存、分层卸载、混合状态检查点。量化配置见 [KV Cache 量化](kv-cache-quantization.md)。

## 1. 统一内存规划

```text
固定区：weights / CUDA Graph / required workspace
弹性区：paged KV / GDN state / prefix checkpoints / vision embedding cache
```

`ALLOCATE/BIND` 先扣除固定区与安全余量，剩余交给 Cache Manager。每个请求只有一个 `CacheHandle`：

```text
CacheHandle
├─ KV page refs
├─ GDN conv/recurrent state
├─ aligned prefix checkpoint
├─ vision embedding refs
└─ request table slot
```

分配是事务化的：全部 commit 或全部 rollback。不按 `max_tokens` 一次性预留 KV；Decode 按 page 增长。

## 2. 分页与 Radix

GPU 侧是 paged arena，Radix 管理页对齐前缀的复用。页键按 token 页内容链计算；命中时逐页比对 token，摘要碰撞或文件损坏只降级为 miss。只缓存完整页，未发布尾页不进入可复用集合。

## 3. 分层页

```text
GPU paged arena ──D2H（专用 stream）──► pinned host tier ──线程异步──► filesystem tier
       ▲                                     │                         │
       └──────────── H2D（同一 stream） ◄─────┴───── 磁盘 → host 暂存 ◄──┘
```

- GPU 是唯一被 attention kernel 读取的层；host 是 GPU 唯一的 DMA 对端；disk 必须经 host 暂存。
- 页键是 `blake2b(identity || namespace || page_size || page_0 || ... || page_i)`。identity 含模型 digest、dtype、page size、KV head、head dim 与层映射。键随请求增量计算，不用 Python `hash`。
- host entry 保存完整 token page 并在命中时逐页比较。entry 用 `refs` 钉住，DMA / 磁盘读写期间不被 LRU 回收；GPU 页由 cache manager 的 `retain/release_refs` 钉住直到 D2H 完成。
- 容量不足、队列满、CRC 失败、文件消失都降级为 miss；offload 不参与请求成败。

**写入**：调度器在页边界调用 `publish_prefix`，radix 插入返回 `inserted_pages`。只有新插入的物理页提交给 `HostKVTier.submit_store`；专用 stream 逐层 `copy_(non_blocking=True)`，记录 CUDA event。`StoreTicket` 持有 GPU 共享 ref 直到 event 完成。轮询完成后 host entry 变为 ready，级联写盘（临时文件 + `os.replace`，页文件含 magic / version / identity / 摘要 / token / CRC）。disk tier 独立字节预算与 LRU；删除竞态中读方只得到 miss。

**提升**：`admit_with_prefix` 先走 radix 命中，host 只补齐命中之后的连续页后缀，host 缺页时 disk 继续补齐。状态机：

```text
lookup(host, disk) → reserve extension GPU pages
    ├─ host ready → pin host → H2D → event ready → publish radix
    └─ disk hit   → reserve host slots → disk→host → pin host → H2D → publish radix
                      └─ 失败 → 提升到失败点之前的连续前缀
```

提升期间请求留在 waiting；scheduler 每轮检查 `promotion_pending()`。H2D event ready 后 `CopyTask.attach()` 让计算 stream 等 event，再更新 `cached_tokens` / `published_tokens` 并插入 radix 且 pin。取消 / 释放要等在途 H2D 完成后才释放 GPU 页。

| 配置 | 默认 | 含义 |
| --- | --- | --- |
| `--kv-host-cache-bytes` | 0（关闭） | pinned host 页池字节数 |
| `--kv-disk-cache-dir` | 无 | 文件层级根目录 |
| `--kv-disk-cache-bytes` | 0（关闭） | 文件层字节预算；需同时配 host 层 |
| `--kv-disk-read-threads` / `--kv-disk-write-threads` | 4 / 2 | 读写线程池独立 |

`CacheStats.snapshot()["offload"]` 暴露容量、占用、pin 数、hits / misses、store / load 页数、淘汰、失败计数、pending 作业与提升等待时间。磁盘目录 `config.json` 记录 identity；配置漂移启动失败。冷启动扫描页文件重建 LRU，损坏或不匹配的页文件删除。

## 4. 混合状态

Qwen3.5 前缀命中必须同时恢复全注意力 KV、对齐边界的 GDN recurrent / conv state、MRoPE / 媒体位置状态。完整混合状态检查点只建在 KV page 对齐边界，不逐 page 无条件保存 GDN 状态。Inkling-2 的短卷积状态无法从共享前缀重建，带 `short_convolution` 的模型关闭前缀缓存；chunked prefill 仍可跨块续接。

## 5. 未做

跨节点 P2P / 对象存储 secondary tier、ARC 策略、直接 NVMe DMA、多进程共享同一 disk 目录的写者协调、未发布尾页的抢占 swap。DMA 使用单条专用 stream 逐层 `copy_`，语义与多 stream + 批量指针描述符等价。

## 6. 验证

`tests/test_kv_offload.py` 覆盖键链增量与 identity 隔离、host LRU / pin / 去重、token 精确校验、CPU 端到端 store→promote、disk 级联 / 重启恢复 / CRC 损坏降级 / 容量淘汰 / 配置漂移、manager 从 host 提升被淘汰前缀、radix 命中后的 host 后缀提升、取消等待、scheduler 等待提升完成；另有 CUDA 小 tensor 烟测验证专用 stream / event 的 D2H / H2D 语义。


