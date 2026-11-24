# Prefill/Decode 分离路由

MegaLM 网关支持 vLLM/SGLang 风格的 disaggregated serving 代理路由。网关只负责**请求投放**：先向 prefill 上游发送带 `kv_transfer_params.do_remote_prefill` 的请求，拿到 KV transfer handle，再把它随 `do_remote_decode` 转发给 decode 上游；KV 字节在两个引擎实例之间直传，不经过网关。

## 前置条件

两个上游引擎都必须理解 `kv_transfer_params` 握手协议。只配置一侧会被启动参数校验拒绝，避免 KV 状态被静默丢弃。

## 配置

```bash
megalm-server --model /models/Qwen3-8B \
  --host 0.0.0.0 --port 8000 \
  --pd-prefill-url http://prefill-host:8000 \
  --pd-decode-url  http://decode-host:8000 \
  --pd-timeout-s 300
```

`--pd-prefill-url` 与 `--pd-decode-url` 必须成对提供。启用后，`/v1/chat/completions` 与 `/v1/completions` 会被代理到这两个上游；其余端点仍走本地 Replica。非流式响应与 SSE 流式响应都会被透传（含上游状态码与 OpenAI 错误体）。

## 失败语义

| 场景 | 网关行为 |
| --- | --- |
| prefill 上游返回 4xx/5xx | 原样透传状态码与响应体 |
| prefill 未返回 `kv_transfer_params` | 返回 502 |
| 上游连接失败 / 超时 | 返回 502 |
| decode 流式响应 | 以 SSE 透传，客户端断连时释放上游连接 |

## 限制

网关不搬运 KV 张量；实际 transfer 由两个引擎实例通过各自的 connector 完成。因此 prefill 与 decode 上游必须部署在 KV 可直达的网络域内（通常是同一节点或 RDMA/NCCL 互联的节点）。
