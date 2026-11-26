//! End-to-end test: a local OpenAI-compatible SSE server, driven by the real
//! benchmark, with no GPU or external network involved.

use std::fs;
use std::time::{SystemTime, UNIX_EPOCH};

use anyhow::Result;
use axum::body::{Body, Bytes};
use axum::extract::Json;
use axum::http::{header, StatusCode};
use axum::response::IntoResponse;
use axum::routing::{get, post};
use axum::Router;
use clap::Parser;
use megalm_bench::{run_serve, Cli, Command};
use serde_json::{json, Value};
use tokio::net::TcpListener;

async fn chat(body: Bytes) -> impl IntoResponse {
    let request: Value = serde_json::from_slice(&body).unwrap_or_default();
    let tokens = request
        .get("max_tokens")
        .and_then(Value::as_u64)
        .unwrap_or(6)
        .min(32) as usize;
    let with_usage = request.get("stream_options").is_some();
    let mut events: Vec<Bytes> = Vec::new();
    for index in 0..tokens {
        let reason = if index + 1 == tokens {
            "\"length\""
        } else {
            "null"
        };
        events.push(Bytes::from(format!(
            "data: {{\"choices\":[{{\"delta\":{{\"content\":\"x\"}},\"finish_reason\":{reason}}}]}}\n\n"
        )));
    }
    if with_usage {
        events.push(Bytes::from(format!(
            "data: {{\"choices\":[],\"usage\":{{\"prompt_tokens\":16,\"completion_tokens\":{tokens}}}}}\n\n"
        )));
    }
    events.push(Bytes::from_static(b"data: [DONE]\n\n"));
    (
        StatusCode::OK,
        [(header::CONTENT_TYPE, "text/event-stream")],
        Body::from_stream(futures::stream::iter(
            events.into_iter().map(Ok::<_, std::convert::Infallible>),
        )),
    )
}

async fn models() -> impl IntoResponse {
    Json(json!({"object": "list", "data": [{"id": "mock"}]}))
}

async fn failing() -> impl IntoResponse {
    (StatusCode::INTERNAL_SERVER_ERROR, "boom")
}

fn temp_dir() -> std::path::PathBuf {
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    let dir = std::env::temp_dir().join(format!("megalm-bench-it-{nanos}"));
    fs::create_dir_all(&dir).unwrap();
    dir
}

fn args(
    base_url: &str,
    output: &std::path::Path,
    markdown: &std::path::Path,
    no_usage: bool,
) -> Command {
    let mut raw = vec![
        "megalm-bench".to_owned(),
        "serve".to_owned(),
        "--base-url".to_owned(),
        base_url.to_owned(),
        "--model".to_owned(),
        "mock".to_owned(),
        "--num-prompts".to_owned(),
        "4".to_owned(),
        "--concurrency".to_owned(),
        "2".to_owned(),
        "--trials".to_owned(),
        "1".to_owned(),
        "--warmup".to_owned(),
        "1".to_owned(),
        "--max-tokens".to_owned(),
        "6".to_owned(),
        "--prompt-tokens".to_owned(),
        "16".to_owned(),
        "--save-detailed".to_owned(),
        "--quiet".to_owned(),
        "--output".to_owned(),
        output.to_string_lossy().to_string(),
        "--markdown".to_owned(),
        markdown.to_string_lossy().to_string(),
    ];
    if no_usage {
        raw.push("--no-include-usage".to_owned());
    }
    Cli::parse_from(raw).command
}

#[tokio::test]
async fn end_to_end_streaming_report_and_fallback() -> Result<()> {
    let app = Router::new()
        .route("/v1/chat/completions", post(chat))
        .route("/v1/models", get(models));
    let listener = TcpListener::bind("127.0.0.1:0").await?;
    let base_url = format!("http://{}", listener.local_addr()?);
    let server = tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });

    let dir = temp_dir();
    let output = dir.join("report.json");
    let markdown = dir.join("report.md");
    let Command::Serve(serve) = args(&base_url, &output, &markdown, false) else {
        panic!("expected serve");
    };
    run_serve(*serve).await?;

    let text = fs::read_to_string(&output)?;
    let report: Value = serde_json::from_str(&text)?;
    assert_eq!(report["schema_version"], 1);
    assert_eq!(report["provenance"]["workload"]["details"]["seed"], 0);
    let row = &report["results"]["default"][0]["trials"][0];
    assert_eq!(row["completed"], 4);
    assert_eq!(row["failed"], 0);
    assert_eq!(row["output_tokens"], 24);
    assert_eq!(row["ttft_ms"]["n"], 4);
    assert_eq!(row["token_source"], "usage");
    assert_eq!(row["samples"][0]["prompt_tokens"], 16);
    assert!(row["output_throughput"].as_f64().unwrap_or(0.0) > 0.0);
    let rendered = fs::read_to_string(&markdown)?;
    assert!(rendered.contains("| default | 2 |"));

    // Servers that reject stream_options must still be measurable, with the
    // token-count source clearly recorded as chunk-based.
    let output2 = dir.join("fallback.json");
    let markdown2 = dir.join("fallback.md");
    let Command::Serve(serve2) = args(&base_url, &output2, &markdown2, true) else {
        panic!("expected serve");
    };
    run_serve(*serve2).await?;
    let fallback: Value = serde_json::from_str(&fs::read_to_string(&output2)?)?;
    let row2 = &fallback["results"]["default"][0]["trials"][0];
    assert_eq!(row2["token_source"], "chunks");
    assert_eq!(row2["output_tokens"], 24);

    server.abort();
    fs::remove_dir_all(dir).ok();
    Ok(())
}

#[tokio::test]
async fn greedy_output_verification_compares_engines() -> Result<()> {
    let app = Router::new().route("/v1/chat/completions", post(chat));
    let listener = TcpListener::bind("127.0.0.1:0").await?;
    let url = format!("http://{}", listener.local_addr()?);
    let server = tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });

    let dir = temp_dir();
    let output = dir.join("verify.json");
    let Command::Serve(serve) = Cli::parse_from([
        "megalm-bench",
        "serve",
        "--engine",
        &format!("a={url}"),
        "--engine",
        &format!("b={url}"),
        "--model",
        "mock",
        "--num-prompts",
        "2",
        "--concurrency",
        "1",
        "--trials",
        "1",
        "--warmup",
        "0",
        "--max-tokens",
        "4",
        "--verify-output",
        "--fail-on-output-mismatch",
        "--quiet",
        "--output",
        &output.to_string_lossy(),
    ])
    .command
    else {
        panic!("expected serve");
    };
    run_serve(*serve).await?;
    let report: Value = serde_json::from_str(&fs::read_to_string(&output)?)?;
    assert_eq!(report["output_verification"]["match"], true);
    assert_eq!(
        report["output_verification"]["digests"]["a"],
        report["output_verification"]["digests"]["b"]
    );

    server.abort();
    fs::remove_dir_all(dir).ok();
    Ok(())
}

#[tokio::test]
async fn http_errors_are_counted_not_fatal() -> Result<()> {
    let app = Router::new().route("/v1/chat/completions", post(failing));
    let listener = TcpListener::bind("127.0.0.1:0").await?;
    let base_url = format!("http://{}", listener.local_addr()?);
    let server = tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });

    let dir = temp_dir();
    let output = dir.join("errors.json");
    let markdown = dir.join("errors.md");
    let Command::Serve(serve) = args(&base_url, &output, &markdown, false) else {
        panic!("expected serve");
    };
    run_serve(*serve).await?;
    let report: Value = serde_json::from_str(&fs::read_to_string(&output)?)?;
    let row = &report["results"]["default"][0]["trials"][0];
    assert_eq!(row["failed"], 4);
    assert_eq!(row["error_classes"]["http_5xx"], 4);
    assert_eq!(row["status_counts"]["500"], 4);

    server.abort();
    fs::remove_dir_all(dir).ok();
    Ok(())
}
