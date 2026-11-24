# 调度器

适用范围：单 Replica 内的 admission、连续批处理、抢占、终止。多机 Controller 与 placement lease 不在本版本。

## 1. 请求状态

```text
CREATED → ROUTING → LEASED → ADMITTED → READY → PREFILL → DECODE → FINISHING
                        │                          │
                        └─ WAIT_MEDIA              ├─ OUTPUT_BLOCKED
                                                   └─ PREEMPTED → PREFILL
终态：FINISHED / CANCELLED / REJECTED / FAILED
```

`WAIT_MEDIA` 不占语言模型 KV。`READY` 后才开始缓存事务。Prefill 可分块，Decode 每轮通常一个 token。输出阻塞的请求暂停 Decode，超时后取消。抢占回收运行期状态并重做 Prefill；已发布完整页由分层缓存保留。Cancel、deadline、fatal error 从任意非终态进入清理，释放 sampler / grammar / MTP 状态。

## 2. 调度代价

```text
WorkCost = attention cost + GDN cost + MoE active-token cost
         + vision patch cost + collective cost
```

每轮顺序：

1. 为活跃 Decode 保留 SLA 时间预算；
2. 用剩余预算安排 chunked prefill；
3. 视觉任务走独立队列与 CUDA stream；
4. priority 用 aging 防止饥饿；
5. 输出阻塞请求暂停 Decode，超时取消；
6. 显存压力先回收可重算缓存，再抢占请求；
7. Leader 生成一次 `ExecutionBatch`。

`ExecutionBatch` 固定、紧凑、可回放；不广播 Python 请求对象。一个 batch 可同时含 Decode、普通 Prefill、Chunked Prefill；算子内部按 phase 选 kernel。LoRA slot 张量 `-1` 表示基座；全基座 batch 的 slot 为 `None`，不启动 LoRA kernel。

## 3. 回收顺序

1. 过期视觉 embedding；
2. 冷门 prefix checkpoint；
3. 未运行请求的预留；
4. 可重算的被抢占请求；
5. 仍不满足返回 `overloaded`，不依赖 CUDA OOM 流控。

带 recurrent / conv 状态的模型只有在匹配页边界检查点存在时允许 lower-tier 命中；只恢复 KV 的伪命中属于正确性错误。

## 4. 终止语义

结构化输出 matcher 在 `complete_host()` 前按已采样 token 推进；终止态请求以 `FinishReason.STOP` 结束，不空转到 EOS 或长度上限。xgrammar 的 fill / accept / is_terminated 异常只降级该请求并记 `constrained_grammar_error`。deadline 过期走 scheduler 内部 cancel 时，`finish_step` 对 CANCELLED / REJECTED 事件执行与 `Engine.cancel()` 相同的清理。重复 request_id 不覆盖活跃 matcher，`begin()` 失败开放并记 `constrained_request_id_reused`。

## 5. 验证

greedy 序列一致；chunked prefill 与一次性 prefill 一致；prefix hit / miss 与抢占恢复一致；慢客户端不拖死其他请求；deadline 过期后无 grammar / sampler 状态泄漏；过载拒绝在执行前发生。


