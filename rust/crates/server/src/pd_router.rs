//! Opt-in prefill/decode disaggregation routing (port of `pd_router.py`).
//!
//! The gateway owns request placement: the prefill hop is forced non-streaming
//! so the transfer handle arrives as JSON; that handle is attached to the
//! decode hop, which may stream. KV bytes move directly between the two engine
//! instances — only the small handle travels through this process.

use std::convert::Infallible;
use std::sync::Arc;
use std::time::Duration;

use axum::body::Body;
use axum::http::{HeaderMap, StatusCode};
use axum::response::{IntoResponse, Response};
use futures::StreamExt;
use reqwest::Client;
use serde_json::{json, Value};

use crate::profiling::Profiler;

const HOP_BY_HOP: &[&str] = &[
    "connection",
    "content-length",
    "transfer-encoding",
    "keep-alive",
    "proxy-authenticate",
    "proxy-authorization",
    "te",
    "trailer",
    "upgrade",
];

#[derive(Debug, Clone)]
pub struct PDConfig {
    pub prefill_url: String,
    pub decode_url: String,
    pub timeout_s: f64,
    pub max_connections: usize,
}

impl PDConfig {
    pub fn new(prefill_url: String, decode_url: String, timeout_s: f64, max_connections: usize) -> Result<Self, String> {
        for (name, value) in [("prefill_url", &prefill_url), ("decode_url", &decode_url)] {
            let parsed = reqwest::Url::parse(value)
                .map_err(|_| format!("{name} must be an absolute http(s) URL"))?;
            if !matches!(parsed.scheme(), "http" | "https") || parsed.host_str().is_none() {
                return Err(format!("{name} must be an absolute http(s) URL"));
            }
        }
        if !timeout_s.is_finite() || timeout_s <= 0.0 || timeout_s > 3600.0 {
            return Err("timeout_s must be in (0, 3600]".into());
        }
        if max_connections == 0 {
            return Err("max_connections must be positive".into());
        }
        Ok(Self {
            prefill_url,
            decode_url,
            timeout_s,
            max_connections,
        })
    }
}

#[derive(Clone)]
pub struct PDRouter {
    config: Arc<PDConfig>,
    client: Client,
}

impl PDRouter {
    pub fn new(config: PDConfig) -> Result<Self, String> {
        let client = Client::builder()
            .timeout(Duration::from_secs_f64(config.timeout_s))
            .pool_max_idle_per_host(config.max_connections)
            .build()
            .map_err(|err| format!("cannot build P/D HTTP client: {err}"))?;
        Ok(Self {
            config: Arc::new(config),
            client,
        })
    }

    fn endpoint(base: &str, path: &str) -> String {
        format!("{}{}", base.trim_end_matches('/'), path)
    }

    fn relay_headers(source: &HeaderMap) -> HeaderMap {
        let mut headers = HeaderMap::new();
        for (name, value) in source {
            if HOP_BY_HOP.contains(&name.as_str()) {
                continue;
            }
            headers.insert(name.clone(), value.clone());
        }
        headers
    }

    fn with_kv_transfer(payload: &Value, params: Value) -> Value {
        let mut body = payload.clone();
        if let Value::Object(map) = &mut body {
            map.insert("kv_transfer_params".into(), params);
        }
        body
    }

    fn hop(profiler: &Profiler, event: &str, request_id: &str, src: &str, dst: &str, chunk_id: &str) {
        profiler.record(
            event,
            request_id,
            json!({"src": src, "dst": dst, "chunk_id": chunk_id}),
        );
    }

    /// Run the prefill → decode handshake and relay the decode reply.
    pub async fn route(
        &self,
        payload: Value,
        path: &str,
        profiler: &Profiler,
    ) -> Result<Response, String> {
        let hop_request_id = payload
            .get("request_id")
            .and_then(Value::as_str)
            .filter(|id| !id.is_empty())
            .map(str::to_string)
            .unwrap_or_else(|| uuid::Uuid::new_v4().simple().to_string());
        let prefill_dst = self.config.prefill_url.clone();
        let decode_dst = self.config.decode_url.clone();

        // Prefill must stay non-streaming: the handle is returned as JSON even
        // when the client requested SSE on the decode hop.
        let mut prefill = payload.clone();
        if let Value::Object(map) = &mut prefill {
            map.insert("stream".into(), json!(false));
        }
        let prefill = Self::with_kv_transfer(&prefill, json!({"do_remote_prefill": true}));
        Self::hop(profiler, "hop_sent", &hop_request_id, "gateway", &prefill_dst, "prefill");
        let prefill_response = self
            .client
            .post(Self::endpoint(&prefill_dst, path))
            .json(&prefill)
            .send()
            .await
            .map_err(|err| format!("prefill upstream is unavailable: {err}"))?;
        Self::hop(profiler, "hop_received", &hop_request_id, "gateway", &prefill_dst, "prefill");

        let status = prefill_response.status().as_u16();
        let prefill_headers = Self::relay_headers(prefill_response.headers());
        if status >= 400 {
            let bytes = prefill_response.bytes().await.unwrap_or_default();
            let mut response = (
                StatusCode::from_u16(status).unwrap_or(StatusCode::BAD_GATEWAY),
                bytes.to_vec(),
            )
                .into_response();
            *response.headers_mut() = prefill_headers;
            return Ok(response);
        }
        let prefill_body: Value = prefill_response
            .json()
            .await
            .map_err(|_| "prefill upstream returned invalid JSON".to_string())?;
        let handle = prefill_body.get("kv_transfer_params").filter(|value| value.is_object());
        let handle = match handle {
            Some(value) if value.as_object().is_some_and(|map| !map.is_empty()) => value.clone(),
            _ => return Err("prefill upstream did not return kv_transfer_params".into()),
        };

        let mut decode_params = json!({"do_remote_decode": true});
        if let (Value::Object(target), Value::Object(source)) = (&mut decode_params, &handle) {
            for (key, value) in source {
                target.insert(key.clone(), value.clone());
            }
        }
        let decode = Self::with_kv_transfer(&payload, decode_params);
        let streaming = payload.get("stream").and_then(Value::as_bool).unwrap_or(false);

        Self::hop(profiler, "hop_sent", &hop_request_id, "gateway", &decode_dst, "decode");
        let decode_response = self
            .client
            .post(Self::endpoint(&decode_dst, path))
            .json(&decode)
            .send()
            .await
            .map_err(|err| format!("decode upstream is unavailable: {err}"))?;
        Self::hop(profiler, "hop_received", &hop_request_id, "gateway", &decode_dst, "decode");

        let status = decode_response.status().as_u16();
        let headers = Self::relay_headers(decode_response.headers());
        if !streaming {
            let bytes = decode_response.bytes().await.unwrap_or_default();
            let mut response = (
                StatusCode::from_u16(status).unwrap_or(StatusCode::BAD_GATEWAY),
                bytes.to_vec(),
            )
                .into_response();
            *response.headers_mut() = headers;
            return Ok(response);
        }

        let profiler = profiler.clone();
        let stream = async_stream::stream! {
            let mut index = 0usize;
            let mut body = decode_response.bytes_stream();
            while let Some(chunk) = body.next().await {
                let Ok(bytes) = chunk else { break };
                Self::hop(&profiler, "hop_received", &hop_request_id, &decode_dst, "gateway", &index.to_string());
                index += 1;
                yield Ok::<axum::body::Bytes, Infallible>(axum::body::Bytes::from(bytes));
            }
        };
        let mut response = Body::from_stream(stream).into_response();
        *response.status_mut() = StatusCode::from_u16(status).unwrap_or(StatusCode::BAD_GATEWAY);
        *response.headers_mut() = headers;
        Ok(response)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use axum::http::{header, HeaderValue};

    #[test]
    fn config_requires_two_absolute_http_urls() {
        assert!(PDConfig::new(
            "https://prefill.example".into(),
            "https://decode.example".into(),
            300.0,
            100,
        )
        .is_ok());
        assert!(PDConfig::new("prefill.example".into(), "https://decode.example".into(), 300.0, 100).is_err());
        assert!(PDConfig::new("ftp://prefill.example".into(), "https://decode.example".into(), 300.0, 100).is_err());
        assert!(PDConfig::new("https://a.example".into(), "https://b.example".into(), 0.0, 100).is_err());
        assert!(PDConfig::new("https://a.example".into(), "https://b.example".into(), 300.0, 0).is_err());
    }

    #[test]
    fn kv_transfer_params_are_attached_and_stream_is_forced_off() {
        let payload = json!({"model": "m", "messages": [], "stream": true});
        let mut prefill = payload.clone();
        prefill["stream"] = json!(false);
        let prefill = PDRouter::with_kv_transfer(&prefill, json!({"do_remote_prefill": true}));
        assert_eq!(prefill["stream"], false);
        assert_eq!(prefill["kv_transfer_params"]["do_remote_prefill"], true);

        let decode = PDRouter::with_kv_transfer(&payload, json!({"do_remote_decode": true, "handle": "x"}));
        assert_eq!(decode["stream"], true);
        assert_eq!(decode["kv_transfer_params"]["handle"], "x");
    }

    #[test]
    fn hop_by_hop_headers_are_dropped() {
        let mut headers = HeaderMap::new();
        headers.insert(header::CONNECTION, HeaderValue::from_static("keep-alive"));
        headers.insert(header::CONTENT_TYPE, HeaderValue::from_static("application/json"));
        let relayed = PDRouter::relay_headers(&headers);
        assert!(relayed.get(header::CONNECTION).is_none());
        assert_eq!(relayed.get(header::CONTENT_TYPE).unwrap(), "application/json");
    }

    #[tokio::test]
    async fn two_hop_handshake_carries_handle_and_relays_reply() {
        use axum::routing::post;
        use axum::{Json, Router};
        use std::sync::Mutex as StdMutex;

        let prefill_seen: Arc<StdMutex<Value>> = Arc::new(StdMutex::new(Value::Null));
        let decode_seen: Arc<StdMutex<Value>> = Arc::new(StdMutex::new(Value::Null));

        let prefill_seen_inner = prefill_seen.clone();
        let prefill_app = Router::new().route(
            "/v1/chat/completions",
            post(move |Json(body): Json<Value>| {
                let seen = prefill_seen_inner.clone();
                async move {
                    *seen.lock().unwrap() = body;
                    Json(json!({"kv_transfer_params": {"handle": "h1", "remote": true}}))
                }
            }),
        );
        let decode_seen_inner = decode_seen.clone();
        let decode_app = Router::new().route(
            "/v1/chat/completions",
            post(move |Json(body): Json<Value>| {
                let seen = decode_seen_inner.clone();
                async move {
                    *seen.lock().unwrap() = body;
                    (StatusCode::CREATED, Json(json!({"choices": []})))
                }
            }),
        );

        let prefill_listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let decode_listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let prefill_addr = prefill_listener.local_addr().unwrap();
        let decode_addr = decode_listener.local_addr().unwrap();
        tokio::spawn(async move { axum::serve(prefill_listener, prefill_app).await.unwrap() });
        tokio::spawn(async move { axum::serve(decode_listener, decode_app).await.unwrap() });

        let config = PDConfig::new(
            format!("http://{prefill_addr}"),
            format!("http://{decode_addr}"),
            5.0,
            4,
        )
        .unwrap();
        let router = PDRouter::new(config).unwrap();
        let profiler = Profiler::disabled();
        let response = router
            .route(
                json!({"model": "m", "stream": false, "request_id": "r1"}),
                "/v1/chat/completions",
                &profiler,
            )
            .await
            .unwrap();

        assert_eq!(response.status(), StatusCode::CREATED);
        let prefill = prefill_seen.lock().unwrap().clone();
        assert_eq!(prefill["stream"], false);
        assert_eq!(prefill["kv_transfer_params"]["do_remote_prefill"], true);
        assert_eq!(prefill["request_id"], "r1");
        let decode = decode_seen.lock().unwrap().clone();
        assert_eq!(decode["kv_transfer_params"]["do_remote_decode"], true);
        assert_eq!(decode["kv_transfer_params"]["handle"], "h1");
        assert_eq!(decode["kv_transfer_params"]["remote"], true);
    }

    #[tokio::test]
    async fn streaming_decode_reply_is_relayed_chunk_by_chunk() {
        use axum::body::Body;
        use axum::routing::post;
        use axum::{Json, Router};
        use std::convert::Infallible;

        let prefill_app = Router::new().route(
            "/v1/chat/completions",
            post(|| async { Json(json!({"kv_transfer_params": {"h": "x"}})) }),
        );
        let decode_app = Router::new().route(
            "/v1/chat/completions",
            post(|| async {
                let stream = async_stream::stream! {
                    yield Ok::<_, Infallible>(axum::body::Bytes::from_static(b"data: one\n\n"));
                    yield Ok::<_, Infallible>(axum::body::Bytes::from_static(b"data: two\n\n"));
                };
                Body::from_stream(stream)
            }),
        );
        let prefill_listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let decode_listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let prefill_addr = prefill_listener.local_addr().unwrap();
        let decode_addr = decode_listener.local_addr().unwrap();
        tokio::spawn(async move { axum::serve(prefill_listener, prefill_app).await.unwrap() });
        tokio::spawn(async move { axum::serve(decode_listener, decode_app).await.unwrap() });

        let router = PDRouter::new(
            PDConfig::new(
                format!("http://{prefill_addr}"),
                format!("http://{decode_addr}"),
                5.0,
                4,
            )
            .unwrap(),
        )
        .unwrap();
        let response = router
            .route(
                json!({"model": "m", "stream": true}),
                "/v1/chat/completions",
                &Profiler::disabled(),
            )
            .await
            .unwrap();
        let body = axum::body::to_bytes(response.into_body(), 1024 * 1024)
            .await
            .unwrap();
        assert_eq!(&body[..], b"data: one\n\ndata: two\n\n");
    }

    #[test]
    fn endpoint_joins_without_double_slash() {
        assert_eq!(
            PDRouter::endpoint("https://a.example/", "/v1/chat/completions"),
            "https://a.example/v1/chat/completions"
        );
    }
}
