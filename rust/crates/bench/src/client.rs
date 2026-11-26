//! OpenAI-compatible streaming client.  One pooled `reqwest::Client` is shared
//! by all in-flight requests; timing uses a monotonic `Instant` provided by the
//! caller so the whole trial has a single time origin.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

use anyhow::{anyhow, Result};
use futures::StreamExt;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};

use crate::cli::{EndpointKind, ServeArgs};
use crate::workload::RequestSpec;

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Sample {
    pub request_id: String,
    /// Milliseconds from the trial start to request admission.
    pub start_ms: f64,
    /// Milliseconds from the trial start to stream completion.
    pub end_ms: f64,
    /// Wall time spent waiting for a concurrency permit.
    pub queue_wait_ms: f64,
    pub e2e_ms: f64,
    pub ttft_ms: Option<f64>,
    pub output_tokens: u32,
    pub prompt_tokens: Option<u32>,
    /// `usage` when the server reported token counts, `chunks` otherwise.
    pub token_source: String,
    pub chunk_count: u32,
    /// Arrival time of each content chunk, used for ITL and peak throughput.
    pub token_times_ms: Vec<f64>,
    pub finish_reason: String,
    pub status: u16,
    pub error: String,
    pub error_class: String,
    /// Concatenated content deltas, used for cross-engine output verification.
    pub text: String,
}

impl Sample {
    fn failed(request_id: &str, base: Instant, error: String, status: u16) -> Self {
        let now = elapsed_ms(base);
        Self {
            request_id: request_id.to_owned(),
            start_ms: now,
            end_ms: now,
            queue_wait_ms: 0.0,
            e2e_ms: 0.0,
            ttft_ms: None,
            output_tokens: 0,
            prompt_tokens: None,
            token_source: "none".to_owned(),
            chunk_count: 0,
            token_times_ms: Vec::new(),
            finish_reason: String::new(),
            status,
            error_class: classify_error(status, &error),
            error,
            text: String::new(),
        }
    }
}

fn classify_error(status: u16, error: &str) -> String {
    if error.contains("timeout") || error.contains("timed out") {
        return "timeout".to_owned();
    }
    if status >= 500 {
        return "http_5xx".to_owned();
    }
    if status >= 400 {
        return "http_4xx".to_owned();
    }
    if error == "no content delta received" || error == "empty stream" {
        return "protocol".to_owned();
    }
    if status == 0 {
        return "transport".to_owned();
    }
    "unknown".to_owned()
}

pub fn elapsed_ms(base: Instant) -> f64 {
    base.elapsed().as_secs_f64() * 1000.0
}

pub struct Client {
    model: String,
    endpoint: EndpointKind,
    url: String,
    http: reqwest::Client,
    api_key: Option<String>,
    headers: Vec<(String, String)>,
    extra_body: serde_json::Map<String, Value>,
    include_usage: bool,
    usage_disabled: Arc<AtomicBool>,
}

impl Client {
    pub fn new(args: &ServeArgs, engine_url: &str, max_inflight: usize) -> Result<Self> {
        let http = reqwest::Client::builder()
            .timeout(Duration::from_secs(args.timeout_secs))
            .connect_timeout(Duration::from_secs(10))
            .pool_max_idle_per_host(max_inflight.max(1))
            .build()?;
        Ok(Self {
            model: args.model.clone(),
            endpoint: args.endpoint,
            url: format!(
                "{}/{}",
                engine_url.trim_end_matches('/'),
                args.endpoint.path().trim_start_matches('/')
            ),
            http,
            api_key: args.api_key.clone(),
            headers: args.headers()?,
            extra_body: args.extra_body()?,
            include_usage: !args.no_include_usage,
            usage_disabled: Arc::new(AtomicBool::new(false)),
        })
    }

    fn body(
        &self,
        spec: &RequestSpec,
        max_tokens: u32,
        temperature: f32,
        ignore_eos: bool,
    ) -> Value {
        let mut body = match self.endpoint {
            EndpointKind::Chat => {
                let mut body = json!({
                    "model": self.model,
                    "messages": [json!({"role": "user", "content": spec.prompt})],
                    "temperature": temperature,
                    "max_tokens": max_tokens,
                    "stream": true,
                });
                if ignore_eos {
                    body["ignore_eos"] = json!(true);
                }
                body
            }
            EndpointKind::Completions => {
                let mut body = json!({
                    "model": self.model,
                    "prompt": spec.prompt,
                    "temperature": temperature,
                    "max_tokens": max_tokens,
                    "stream": true,
                });
                if ignore_eos {
                    body["ignore_eos"] = json!(true);
                }
                body
            }
        };
        if let Some(object) = body.as_object_mut() {
            for (key, value) in &self.extra_body {
                if key != "messages" {
                    object.insert(key.clone(), (*value).clone());
                }
            }
        }
        body
    }

    /// Run one streaming request.  `queued_ms` is the admission time supplied
    /// by the runner; the difference to `start_ms` is client-side queue wait.
    pub async fn stream(
        &self,
        spec: &RequestSpec,
        max_tokens: u32,
        temperature: f32,
        ignore_eos: bool,
        base: Instant,
        queued_ms: f64,
    ) -> Sample {
        let mut allow_retry = true;
        loop {
            let (sample, retry) = self
                .stream_once(
                    spec,
                    max_tokens,
                    temperature,
                    ignore_eos,
                    base,
                    queued_ms,
                    allow_retry,
                )
                .await;
            if retry {
                // The server rejected `stream_options`; retry without it once.
                allow_retry = false;
                continue;
            }
            return sample;
        }
    }

    #[allow(clippy::too_many_arguments)]
    async fn stream_once(
        &self,
        spec: &RequestSpec,
        max_tokens: u32,
        temperature: f32,
        ignore_eos: bool,
        base: Instant,
        queued_ms: f64,
        allow_usage_retry: bool,
    ) -> (Sample, bool) {
        let send_usage = self.include_usage && !self.usage_disabled.load(Ordering::Relaxed);
        let mut body = self.body(spec, max_tokens, temperature, ignore_eos);
        if send_usage {
            if let Some(object) = body.as_object_mut() {
                object.insert("stream_options".to_owned(), json!({"include_usage": true}));
            }
        }
        let mut request = self.http.post(&self.url).json(&body);
        if let Some(key) = &self.api_key {
            request = request.bearer_auth(key);
        }
        for (name, value) in &self.headers {
            request = request.header(name, value);
        }

        let start_ms = elapsed_ms(base);
        let queue_wait_ms = (start_ms - queued_ms).max(0.0);
        let response = match request.send().await {
            Ok(response) => response,
            Err(err) => {
                let message = err.to_string();
                let mut sample = Sample::failed(&spec.request_id, base, message, 0);
                sample.start_ms = start_ms;
                sample.queue_wait_ms = queue_wait_ms;
                return (sample, false);
            }
        };
        let status = response.status().as_u16();
        if status >= 400 {
            let text = response.text().await.unwrap_or_default();
            let message = format!(
                "HTTP {status}: {}",
                text.chars().take(300).collect::<String>()
            );
            if send_usage && matches!(status, 400 | 422) && allow_usage_retry {
                self.usage_disabled.store(true, Ordering::Relaxed);
                return (
                    Sample::failed(&spec.request_id, base, message, status),
                    true,
                );
            }
            let mut sample = Sample::failed(&spec.request_id, base, message, status);
            sample.start_ms = start_ms;
            sample.queue_wait_ms = queue_wait_ms;
            return (sample, false);
        }

        let mut ttft: Option<f64> = None;
        let mut text = String::new();
        let mut token_times: Vec<f64> = Vec::new();
        let mut chunk_count = 0u32;
        let mut reported_output: Option<u32> = None;
        let mut reported_prompt: Option<u32> = None;
        let mut finish_reason = String::new();
        let mut stream = response.bytes_stream();
        let mut buffer: Vec<u8> = Vec::with_capacity(4096);
        let mut stream_error: Option<String> = None;

        while let Some(chunk) = stream.next().await {
            let bytes = match chunk {
                Ok(bytes) => bytes,
                Err(err) => {
                    stream_error = Some(err.to_string());
                    break;
                }
            };
            buffer.extend_from_slice(&bytes);
            while let Some(position) = buffer.iter().position(|byte| *byte == b'\n') {
                let line = String::from_utf8_lossy(&buffer[..position]).into_owned();
                buffer.drain(..=position);
                let line = line.trim_end_matches('\r').trim();
                if line.is_empty() || line.starts_with(':') || !line.starts_with("data:") {
                    continue;
                }
                let data = line[5..].trim();
                if data == "[DONE]" {
                    continue;
                }
                let value: Value = match serde_json::from_str(data) {
                    Ok(value) => value,
                    Err(err) => {
                        stream_error = Some(format!("invalid SSE payload: {err}"));
                        break;
                    }
                };
                if let Some(error) = value.get("error") {
                    stream_error = Some(format!("server error: {error}"));
                    break;
                }
                if let Some(usage) = value.get("usage").and_then(Value::as_object) {
                    if let Some(count) = usage.get("completion_tokens").and_then(Value::as_u64) {
                        reported_output = Some(count.min(u32::MAX as u64) as u32);
                    }
                    if let Some(count) = usage.get("prompt_tokens").and_then(Value::as_u64) {
                        reported_prompt = Some(count.min(u32::MAX as u64) as u32);
                    }
                }
                let choices = value.get("choices").and_then(Value::as_array);
                let choice = choices.and_then(|items| items.first());
                let content = match self.endpoint {
                    EndpointKind::Chat => choice
                        .and_then(|item| item.get("delta"))
                        .and_then(|delta| delta.get("content"))
                        .and_then(Value::as_str),
                    EndpointKind::Completions => choice
                        .and_then(|item| item.get("text"))
                        .and_then(Value::as_str),
                };
                if let Some(reason) = choice
                    .and_then(|item| item.get("finish_reason"))
                    .and_then(Value::as_str)
                {
                    if !reason.is_empty() {
                        finish_reason = reason.to_owned();
                    }
                }
                if let Some(delta) = content.filter(|value| !value.is_empty()) {
                    text.push_str(delta);
                    let now = elapsed_ms(base);
                    if ttft.is_none() {
                        ttft = Some(now - start_ms);
                    }
                    token_times.push(now);
                    chunk_count += 1;
                }
            }
        }
        let end_ms = elapsed_ms(base);
        if let Some(err) = stream_error {
            let mut sample = Sample::failed(&spec.request_id, base, err, status);
            sample.start_ms = start_ms;
            sample.end_ms = end_ms;
            sample.queue_wait_ms = queue_wait_ms;
            return (sample, false);
        }

        let output_tokens = reported_output.unwrap_or(chunk_count);
        let token_source = if reported_output.is_some() {
            "usage"
        } else {
            "chunks"
        };
        let error = if ttft.is_none() && chunk_count == 0 {
            if output_tokens > 0 {
                String::new()
            } else {
                "no content delta received".to_owned()
            }
        } else {
            String::new()
        };
        (
            Sample {
                request_id: spec.request_id.clone(),
                start_ms,
                end_ms,
                queue_wait_ms,
                e2e_ms: end_ms - start_ms,
                ttft_ms: ttft,
                output_tokens,
                prompt_tokens: reported_prompt,
                token_source: token_source.to_owned(),
                chunk_count,
                token_times_ms: token_times,
                finish_reason,
                status,
                error,
                error_class: String::new(),
                text,
            },
            false,
        )
    }

    /// Readiness probe used before warmup.  A non-2xx response is an error.
    pub async fn ready(&self, timeout: Duration) -> Result<()> {
        let url = format!(
            "{}/v1/models",
            self.url.trim_end_matches(args_endpoint_suffix(&self.url))
        );
        let response = self.http.get(&url).timeout(timeout).send().await?;
        if !response.status().is_success() {
            return Err(anyhow!("readiness probe returned {}", response.status()));
        }
        Ok(())
    }
}

/// The client URL already ends in a completion path; strip it for /v1/models.
fn args_endpoint_suffix(url: &str) -> &str {
    if url.ends_with("/v1/chat/completions") {
        "/v1/chat/completions"
    } else {
        "/v1/completions"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn error_classes_are_stable() {
        assert_eq!(classify_error(503, ""), "http_5xx");
        assert_eq!(classify_error(400, "bad request"), "http_4xx");
        assert_eq!(classify_error(0, "request timeout"), "timeout");
        assert_eq!(classify_error(0, "no content delta received"), "protocol");
    }
}
