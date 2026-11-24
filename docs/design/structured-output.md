# 结构化输出

适用范围：`response_format` 为 `json_object` / `json_schema`，纯文本语法、无 tools、Replica 未启用投机解码，且安装了可选依赖 `structured`。其他情况降级为尽力而为。

## 1. 契约

`contracts/request.py` 定义不可变、可哈希且 schema 规范化的 `StructuredOutput`，由 `SamplingParams.structured_output` 携带。网关 `gateway/chat_api.py` 把 `response_format` 转成该对象；tools / Inkling 语法组合自动降级。

## 2. 执行路径

```text
response_format → StructuredOutput → GrammarMatcher → CPU bitmask
        → H2D（仅 need_apply 行）→ Triton kernel 置 -inf → 采样
        → 按已发出 token 推进 matcher
```

`runtime/structured.py`：

- 懒加载 xgrammar；按 `TokenizerInfo` 构建 `GrammarCompiler`。词表类型自动探测，`vocab_size` 取模型 lm_head 维度。
- 编译结果按规范化 schema 文本做有界 LRU 缓存；每请求一个 `GrammarMatcher`。
- `_warn` 按原因去重，上限 256 条；`_warned_reasons` / `_matchers` / `_bitmask_buffers` 受 RLock 保护。
- `GrammarManager.bitmask_buffer()` 复用宿主 bitmask；`fill_next_token_bitmask` 覆写整行，不会残留旧比特。
- `begin()` 遇到活跃 request_id 返回失败开放并记 `constrained_request_id_reused`；`add_request` 只回滚本次创建的 matcher。

`runtime/engine.py`：

- `_apply_grammar_mask()` 汇总本步约束行，只对 `need_apply=True` 的行做 H2D 拷贝。
- 约束请求把 `use_greedy_graph` 置假，绕开绕过 sampler 的 CUDA Graph。
- `_advance_grammar_for_tokens()` 在 `complete_host()` 之前按采样 token 推进 matcher；进入终止态时通过 `structurally_complete` 让 scheduler 以 `FinishReason.STOP` 结束。
- 完成 / 取消时释放 matcher；`InferenceRequest` 构造或 `scheduler.submit` 失败时回滚 grammar 状态。

`kernels/grammar_mask.py` 的 Triton kernel 把禁用 token 的 logits 置 `-inf`，CPU 回退用 `masked_fill_`。logits 行跨度 `logits_vocab` 与 grammar 词表 `grammar_vocab` 解耦；padding 出的额外 logits 由 kernel 直接置 `-inf`，不需要更宽的 bitmask。

## 3. 降级

缺依赖、schema 不受支持、投机解码、tools / Inkling 组合都自动降级，不中断生成。结构化日志：`constrained_decoding_skipped`、`constrained_decoding_unavailable`、`constrained_token_rejected`、`constrained_grammar_error`、`constrained_request_id_reused`、`response_format_invalid`。日志统一走 `log_event()`，直接 `extra=` 的字段不会被 `JsonLogFormatter` 丢弃。

xgrammar 的 fill / accept / is_terminated 异常在 `RequestGrammar` 内捕获，置 `failed` 并短路，只降级该请求。

## 4. 指标

`EngineMetrics.snapshot()` 合并 `GrammarManager.stats()`：`structured_requests`、`structured_degraded`、`structured_mask_steps`、`structured_masked_rows`、`structured_rejected`、`structured_compile_failures`、`structured_matcher_errors`、`structured_active_matchers`、`structured_bitmask_buffers`、`structured_grammar_errors`，以及按降级原因和结束原因展开的分类计数。

## 5. 未做

严格 tool calling 的 structural tag；投机解码下的 draft 链 mask 与拒绝回滚。

## 6. 验证

`tests/test_structured_output.py`、`tests/test_structured_output_engine.py`：契约校验、Grammar 编译缓存、位掩码只放行合法 token、非法续接拒绝、`n` 范围 `1..128`、降级路径、bitmask 缓冲复用、按原因去重日志、并发 begin / release、词表 padding（CPU + CUDA）、校验失败回滚、多 choice matcher 隔离、logprobs 兼容、指标计数。CUDA 端到端用真实 checkpoint 验证 `json_object` / `json_schema` 输出可被 `json.loads` 解析且满足 schema。
