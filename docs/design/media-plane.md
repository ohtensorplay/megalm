# 媒体数据面

适用范围：Gateway 把请求正文里的媒体安全转换成 `ImageInput` / `VideoInput` / `AudioInput`。视觉塔与 Replica 调度沿用 sealed memfd 传输和 `docs/design/architecture.md` 的媒体流水线。

## 1. 处理阶段

```text
HTTP JSON（data: / http(s): URL）
        │ RequestBodyLimitMiddleware（控制帧硬上限）
        ▼
resolve_remote_media()
        │ http(s) URL → 有界抓取 → data: URL；有界并发 gather
        ▼
parse_media_chat()
        │ base64 解码、magic bytes 校验、detail 处理、SHA-256 摘要
        ▼
ImageInput / VideoInput / AudioInput
        │
        ▼
seal_media_inputs()（sealed memfd + SHA-256 描述符）
        │
        ▼
Replica / Vision stream
```

远程抓取只发生在 admission 之前的冷路径。所有 URL 均为 `data:` 时该模块零开销。抓取结果不进入控制帧：Gateway 构造的 `data:` URL 是临时对象，`seal_media_inputs` 把不可变字节放进带 seal 的匿名文件，只传描述符与 fd。

## 2. 安全边界

- 协议白名单：`http` / `https`；`file:`、`gopher:` 直接 400。默认仅 https，需显式开关才允许明文 http。
- DNS rebinding：`getaddrinfo` 解析后先校验每个候选 IP，再用该 IP 直接建连；TLS SNI 与证书校验仍用原主机名。校验与连接之间没有第二次解析。
- 地址黑名单：`ipaddress.is_global` 配合 loopback / private / link-local / multicast / reserved / unspecified 显式判断，展开 IPv4-mapped IPv6；URL 携带 userinfo 一并拒绝。
- 域名白名单：逗号分隔的精确域名或 `.suffix`（如 `.hf.co`）。校验在 DNS 解析前执行，重定向每跳重做。默认空 = 不限制。
- 解析混淆：只用 `urllib.parse.urlsplit` 解析一次，按固定 IP 建连，拒绝 userinfo，不存在第二个解析器与校验结果打架的窗口。
- 重定向：`Location` 用 `urljoin` 解析，scheme / host / IP / 域名白名单每跳重来；超过 `max_redirects`（默认 3）失败。
- 响应体硬上限：先校验 `Content-Length`，超限读取前拒绝；chunked 与无长度响应流式累计字节中止。单图 20 MiB、单视频 64 MiB、单音频 32 MiB，单请求总媒体 64 MiB。
- 超时：connect / read / total 三类，`wait_for` 落实到每次 socket 读取；total 覆盖含重试的整次抓取，超时映射 504。
- 重试：连接 / 读取类 `OSError`、超时、上游 5xx / 408 / 429 最多 2 次；其余 4xx 与内容校验失败不重试。上游 4xx（除 408/429）映射 422。

测试专用配置 `MediaFetchConfig(allow_loopback=True)` 是唯一的环回放行方式，不从环境变量读取。

## 3. 配置

| 环境变量 | 默认 | 含义 |
| --- | --- | --- |
| `MEGALM_MEDIA_REMOTE_URLS` | `1` | `0` 关闭全部远程 URL 抓取（400） |
| `MEGALM_MEDIA_ALLOW_HTTP` | `0` | `1` 允许 http URL |
| `MEGALM_MEDIA_MAX_REDIRECTS` | `3` | 单请求最大重定向次数（0–8） |
| `MEGALM_MEDIA_MAX_ATTEMPTS` | `2` | 单 URL 最大尝试次数（1–3） |
| `MEGALM_MEDIA_MAX_CONCURRENT` | `4` | 单请求并发抓取数（1–16） |
| `MEGALM_MEDIA_CONNECT_TIMEOUT_S` | `5` | 建连超时 |
| `MEGALM_MEDIA_READ_TIMEOUT_S` | `15` | 单次读取超时 |
| `MEGALM_MEDIA_TOTAL_TIMEOUT_S` | `60` | 含重试的整次抓取超时 |
| `MEGALM_MEDIA_ALLOWED_DOMAINS` | 空 | 逗号分隔的域名白名单，支持 `.suffix` |

非法值回退默认。配置与运行时计数在 `GET /api/inference/status` 的 `frontend.media_fetch`。

## 4. 错误码

| code | HTTP | 触发 |
| --- | --- | --- |
| `remote_disabled` · `http_disabled` · `unsupported_scheme` · `blocked_address` · `userinfo_disallowed` · `invalid_port` | 400 | 策略拒绝 |
| `media_too_large` · `invalid_media` · `empty_media` · `bad_chunk` · `headers_too_large` | 400 | 内容或体积不合法 |
| `too_many_redirects` · `invalid_content_length` · `domain_not_allowed` | 400 | 协议或策略不合法 |
| `upstream_status`（上游 4xx，除 408/429） | 422 | 客户端给的 URL 不可处理 |
| `upstream_unreachable` · `bad_response` · `upstream_truncated` · `upstream_status`（上游 5xx） | 502 | 上游故障 |
| `timeout` | 504 | 超时 |

每次抓取输出结构化 `media_fetch` 日志（`result`、`code`、`modality`、`host`、`bytes`、`redirects`、`duration_ms`），不记录完整 URL。Gateway 替换完成时输出 `chat_media_remote_resolved`。失败不中断其他并发媒体；`gather(return_exceptions=True)` 取消并等待兄弟任务，保留原始异常类型，不包装成 `ExceptionGroup`。

## 5. detail 语义

Gateway 只校验并透传 `detail`，不改一个字节：

- `ChatCompletionRequest` → `parse_media_chat` → `ImageInput(detail=...)`。
- 跨进程 Replica 经 `media_ipc` 的 image descriptor 无损传递 `detail`。
- 视觉处理器（`Qwen35ImageProcessor`、`Qwen25OmniImageProcessor`、`OXRImageProcessor`）在 `_prepare_one` 里调用 `components.media_resize.detail_pixel_limits(detail, min_pixels, max_pixels)`：`auto` / `high` 保持原预算，`low` 把上限压到 `512×512`，再由模型 `smart_resize` 做 patch 对齐缩放。原始字节与 SHA-256 在所有档位相同，没有二次缩放或重编码。

缓存隔离：`VisionCacheKey` 含 `detail`；`InferenceRequest.media_hashes` 与 `media_admission` 的 prefix fingerprint 也含 detail。采用 `min_pixels/max_pixels` 路线兑现 `detail=low`，不在网关复制媒体处理器逻辑。

## 6. 取舍

不缓存下载字节；复用引擎侧内容摘要的 vision cache。用标准库 `asyncio` 自研 HTTP/1.1 客户端，不引入额外 HTTP 依赖。安全语义与错误分类对齐主流服务，但实现独立。

## 7. 验证

```bash
PYTHONPATH=src pytest -q tests/test_media_fetch.py tests/test_chat_media.py tests/test_detail_pipeline.py
```

覆盖 Content-Type 嗅探、chunked、重定向与重定向环、大小上限、超时映射、重试计数、SSRF / 默认策略拒绝、域名白名单（精确 / 后缀 / 重定向绕过）、反斜杠解析混淆回归、4xx→422、5xx 重试、类型冲突、消息标准化、detail 透传与 processor 预算、IPC 往返、缓存键隔离、状态快照。
