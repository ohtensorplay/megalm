//! Axum router and OpenAI-compatible handlers.

use std::collections::HashMap;
use std::os::fd::OwnedFd;
use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};

use axum::body::Body;
use axum::extract::{DefaultBodyLimit, Path, Query, State};
use axum::http::{header, StatusCode};
use axum::middleware::from_fn_with_state;
use axum::response::{IntoResponse, Response};
use axum::routing::{delete, get, post};
use axum::{body::Bytes, Json, Router};
use base64::Engine as _;
use futures::StreamExt;
use megalm_engine_client::media::{duplicate_fd, seal_memfd};
use megalm_protocol::{
    EmbeddingSubmission, Inspect, MediaDescriptor, MultimodalSubmission, Submission, SubmitBatch,
    SubmitEmbeddings, SubmitMultimodalBatch,
};
use megalm_rs::parser::{JsonFormatValidator, MarkerKind};
use serde::Deserialize;
use serde_json::{json, Value};
use tokio::sync::mpsc;

use crate::media::{has_media_parts, parse_media_chat, ParsedMediaChat};
use crate::middleware::authenticate_api_key;
use crate::state::AppState;
use crate::stream::{
    chat_stream, collect_chat, completion_stream, make_parser, merge_collected, owned_decoder,
    StreamContext,
};
use crate::types::{
    normalized_response_format, normalized_tools, ChatCompletionRequest, CompletionRequest,
    EmbeddingInput, EmbeddingRequest,
};

pub type SharedState = Arc<AppState>;

pub fn build_router(state: AppState) -> Router {
    let state = Arc::new(state);
    let auth = state.auth.clone();
    let router = Router::new()
        .route("/health", get(health))
        .route("/api/health", get(api_health))
        .route("/ready", get(ready))
        .route("/metrics", get(metrics))
        .route("/profiling", get(profiling))
        .route("/v1/models", get(models))
        .route("/server_info", get(server_info))
        .route("/v1/chat/completions", post(chat_completions))
        .route("/v1/completions", post(completions))
        .route("/v1/embeddings", post(embeddings))
        .route("/v1/load_lora_adapter", post(load_lora))
        .route("/v1/unload_lora_adapter", post(unload_lora))
        .route("/api/inference/cancel", post(cancel))
        .route("/api/inference/validate", post(validate))
        .route("/api/inference/load", post(load_model))
        .route("/api/inference/status", get(inference_status))
        .route("/api/models/list", get(models_list))
        .route("/api/models/loras", get(list_loras).post(load_lora))
        .route("/api/models/loras/{reference}", delete(unload_lora_path))
        .route("/v1/messages", post(crate::agents::anthropic_messages))
        .route(
            "/v1/messages/count_tokens",
            post(crate::agents::anthropic_count_tokens),
        )
        .route("/v1/responses", post(crate::agents::openai_responses))
        .route(
            "/v1/responses/{response_id}",
            get(crate::agents::retrieve_response),
        )
        .route(
            "/v1/responses/{response_id}/cancel",
            post(crate::agents::cancel_response),
        )
        .route("/v1/decisions", post(crate::decisions::decisions))
        .route("/v1/systemone", post(crate::decisions::systemone))
        // Opt-in control plane (routes exist but report 404 while disabled).
        .route("/admin/pause", post(crate::control_plane::admin_pause))
        .route(
            "/admin/continue",
            post(crate::control_plane::admin_continue),
        )
        .route("/admin/status", get(crate::control_plane::admin_status))
        .route(
            "/admin/weights/checker",
            get(crate::control_plane::admin_weights_checker),
        )
        .route(
            "/admin/weights/checker",
            post(crate::control_plane::admin_verify_weights),
        )
        .route(
            "/admin/update_weights",
            post(crate::control_plane::weight_update),
        )
        .route(
            "/admin/weights/history",
            get(crate::control_plane::weight_update_history),
        )
        .route(
            "/controller/register",
            post(crate::control_plane::controller_register),
        )
        .route(
            "/controller/heartbeat",
            post(crate::control_plane::controller_heartbeat),
        )
        .route(
            "/controller/drain",
            post(crate::control_plane::controller_drain),
        )
        .route(
            "/controller/activate",
            post(crate::control_plane::controller_activate),
        )
        .route(
            "/controller/acquire",
            post(crate::control_plane::controller_acquire),
        )
        .route(
            "/controller/release",
            post(crate::control_plane::controller_release),
        )
        .route(
            "/controller/status",
            get(crate::control_plane::controller_status),
        )
        .route(
            "/lease/status",
            get(crate::control_plane::lease_tracker_status),
        )
        .route("/router/ready", get(crate::replica_router::router_ready))
        .route("/router/status", get(crate::replica_router::router_status))
        // In-process stage graph (opt-in via MEGALM_STAGE_TOPOLOGY).
        .route(
            "/stage/submit",
            post(crate::stage_graph::stage_graph_submit),
        )
        .route(
            "/stage/result/{request_id}",
            get(crate::stage_graph::stage_graph_result),
        )
        .route("/stage/status", get(crate::stage_graph::stage_graph_status));

    // Inline base64 media can be up to ~86 MiB; the axum default of 2 MiB
    // would reject every multimodal request before it reaches the parser.
    let router = router.layer(DefaultBodyLimit::max(96 * 1024 * 1024));
    let router = router.with_state(state);
    if auth.is_enabled() {
        // Keep auth outside the handlers: rejected requests never reach the
        // engine, and unauthenticated deployments pay no middleware cost.
        router.layer(from_fn_with_state(auth, authenticate_api_key))
    } else {
        router
    }
}

fn now() -> i64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs() as i64
}

fn error(status: StatusCode, message: impl Into<String>) -> Response {
    (
        status,
        Json(json!({"error": {"message": message.into(), "type": "invalid_request_error"}})),
    )
        .into_response()
}

/// Mirror Python's `model_dump(exclude_none=True)` for P/D forwarding.
/// The raw JSON body is preserved so `model_extra` fields are not lost.
fn check_paused(state: &SharedState) -> Option<Response> {
    let paused = state.admin.as_ref().is_some_and(|admin| admin.is_paused());
    if paused {
        Some(error(
            StatusCode::SERVICE_UNAVAILABLE,
            "Replica paused by admin control plane",
        ))
    } else {
        None
    }
}

fn strip_nulls(value: &mut Value) {
    match value {
        Value::Object(map) => {
            map.retain(|_, item| !item.is_null());
            for item in map.values_mut() {
                strip_nulls(item);
            }
        }
        Value::Array(items) => {
            for item in items {
                strip_nulls(item);
            }
        }
        _ => {}
    }
}

async fn health(State(state): State<SharedState>) -> Response {
    let descriptor = state.client.descriptor().await;
    if descriptor.is_null() {
        return error(StatusCode::SERVICE_UNAVAILABLE, "Replica is unavailable");
    }
    Json(json!({"status": "ok"})).into_response()
}

/// Mirrors `/health` for router-worker readiness probes.
async fn ready(State(state): State<SharedState>) -> Response {
    let descriptor = state.client.descriptor().await;
    if descriptor.is_null() {
        return error(StatusCode::SERVICE_UNAVAILABLE, "Replica is unavailable");
    }
    Json(json!({"ready": true})).into_response()
}

async fn api_health() -> Response {
    Json(json!({
        "status": "ok",
        "backend": "megalm",
        "device_type": "linux",
        "chat_only": true,
    }))
    .into_response()
}

async fn metrics(State(state): State<SharedState>) -> Response {
    let snapshot = match state.client.snapshot().await {
        Ok(snapshot) => snapshot,
        Err(err) => return error(StatusCode::SERVICE_UNAVAILABLE, err.to_string()),
    };
    // Fold the opt-in profiler counters into the same Prometheus payload.
    // Disabled profiling still emits zeros, so the series are always present.
    let payload = serde_json::to_string(&with_replica_metrics(
        with_profiling_metrics(snapshot, &state.profiler),
        &state.metrics,
    ))
    .unwrap_or_else(|_| "{}".into());
    let text = megalm_rs::metrics::render_prometheus_metrics(&payload).unwrap_or_default();
    (
        StatusCode::OK,
        [(
            header::CONTENT_TYPE,
            "text/plain; version=0.0.4; charset=utf-8",
        )],
        text,
    )
        .into_response()
}

#[derive(Debug, Deserialize)]
struct ProfilingQuery {
    request_id: Option<String>,
    /// Optional comma-separated JSONL files/directories/globs to merge, for
    /// cross-process views. Defaults to the profiler's own local file.
    path: Option<String>,
}

/// Split the `path` query parameter into concrete JSONL paths. Empty entries
/// and whitespace are ignored so `?path=a.jsonl,,b.jsonl` is well-defined.
fn profiling_paths(spec: Option<&str>) -> Vec<std::path::PathBuf> {
    spec.unwrap_or("")
        .split(',')
        .map(str::trim)
        .filter(|part| !part.is_empty())
        .map(std::path::PathBuf::from)
        .collect()
}

/// Add the opt-in profiler counters to the client snapshot so `/metrics`
/// always exposes the G10 counters as Prometheus series.
fn with_profiling_metrics(mut snapshot: Value, profiler: &crate::profiling::Profiler) -> Value {
    snapshot["profiling"] = profiler.snapshot();
    snapshot
}

/// Fold the always-on proxy / replica telemetry into the same snapshot so the
/// Prometheus renderer exposes the documented `megalm_proxy_*` and
/// `megalm_replica_*` series alongside the worker's runtime metrics.
fn with_replica_metrics(
    mut snapshot: Value,
    metrics: &crate::replica_metrics::ReplicaMetrics,
) -> Value {
    if let (Some(target), Value::Object(extra)) = (snapshot.as_object_mut(), metrics.snapshot()) {
        for (key, value) in extra {
            target.insert(key, value);
        }
    }
    snapshot
}

async fn profiling(
    State(state): State<SharedState>,
    Query(query): Query<ProfilingQuery>,
) -> Response {
    let paths = profiling_paths(query.path.as_deref());
    let view = if paths.is_empty() {
        state.profiler.view(query.request_id.as_deref())
    } else {
        crate::profiling::view_files(&paths, query.request_id.as_deref())
    };
    Json(view).into_response()
}

async fn models(State(state): State<SharedState>) -> Response {
    Json(json!({
        "object": "list",
        "data": [{
            "id": state.served_model_name,
            "object": "model",
            "created": 0,
            "owned_by": "megalm",
        }],
    }))
    .into_response()
}

async fn server_info(State(state): State<SharedState>) -> Response {
    let descriptor = state.client.descriptor().await;
    let runtime = state.client.snapshot().await.unwrap_or(Value::Null);
    Json(json!({
        "descriptor": descriptor,
        "served_model_name": state.served_model_name,
        "replica_transport": "unix_socket_v1",
        "frontend": {"tool_parser": state.tool_parser, "grammar": state.grammar},
        "worker_generation": state.client.generation(),
        "runtime": runtime,
    }))
    .into_response()
}

#[derive(Debug)]
struct MultimodalPlan {
    parsed: ParsedMediaChat,
    audio_output: Option<Value>,
    output_modalities: Vec<String>,
}

#[derive(Debug)]
struct Prepared {
    base_request_id: String,
    /// Best-effort post-hoc `response_format` validator. `None` when the
    /// request did not ask for a JSON response format.
    format_validator: Option<JsonFormatValidator>,
    token_ids: Vec<u32>,
    prompt_tokens: usize,
    dropped_messages: usize,
    initial_in_thinking: bool,
    marker_kinds: Vec<(u32, MarkerKind)>,
    sampling: Value,
    tools: Vec<Value>,
    template_kwargs: Option<Value>,
    forced_tool: Option<String>,
    stop_sequences: Vec<String>,
    include_usage: bool,
    created: i64,
    lora: Option<Value>,
    multimodal: Option<MultimodalPlan>,
}

fn marker_kinds(tokenizer: &megalm_tokenizer::ModelTokenizer) -> Vec<(u32, MarkerKind)> {
    tokenizer
        .special_token_kinds()
        .into_iter()
        .map(|(id, kind)| (id, MarkerKind::parse(&kind)))
        .collect()
}

fn in_thinking_from(state: &SharedState, token_ids: &[u32], rendered: &str) -> bool {
    let (start_token, end_token) = if state.grammar == "gemma4" {
        ("<|channel>", "<channel|>")
    } else {
        ("<think>", "</think>")
    };
    let think_start = state.tokenizer.token_to_id(start_token);
    let think_end = state.tokenizer.token_to_id(end_token);
    if let (Some(start_id), Some(end_id)) = (think_start, think_end) {
        let ids = state
            .tokenizer
            .special_token_kinds()
            .keys()
            .copied()
            .collect::<Vec<_>>();
        megalm_rs::parser::last_reasoning_boundary(token_ids, start_id, end_id, &ids)
            .unwrap_or(false)
    } else {
        rendered.trim_end().ends_with("<think>")
    }
}

async fn encode_turn(
    state: &SharedState,
    messages: &[Value],
    tools: &[Value],
    open_thinking: bool,
    template_kwargs: Option<&Value>,
    start: usize,
    system_count: usize,
) -> Result<(Vec<u32>, bool), Response> {
    let system = &messages[..system_count];
    let retained = system
        .iter()
        .chain(messages[start..].iter())
        .cloned()
        .collect::<Vec<_>>();
    let tools_ref = if tools.is_empty() { None } else { Some(tools) };
    let rendered = state
        .tokenizer
        .render_chat_with(&retained, tools_ref, open_thinking, template_kwargs)
        .map_err(|err| error(StatusCode::BAD_REQUEST, err.to_string()))?;
    let token_ids = state
        .tokenizer_batcher
        .encode(rendered.clone())
        .await
        .map_err(|err| error(StatusCode::BAD_REQUEST, err))?;
    let thinking = in_thinking_from(state, &token_ids, &rendered);
    Ok((token_ids, thinking))
}

/// Port of `chat_api._fit_text_history`: drop whole oldest turns only when the
/// prompt exceeds the budget, keeping every leading system message.
async fn fit_text_history(
    state: &SharedState,
    messages: &[Value],
    tools: &[Value],
    open_thinking: bool,
    template_kwargs: Option<&Value>,
    max_prompt_tokens: Option<usize>,
) -> Result<(Vec<u32>, usize, bool), Response> {
    let mut system_count = 0;
    while system_count < messages.len()
        && messages[system_count].get("role").and_then(Value::as_str) == Some("system")
    {
        system_count += 1;
    }
    let mut starts = vec![system_count];
    starts.extend(
        (system_count + 1..messages.len())
            .filter(|&index| messages[index].get("role").and_then(Value::as_str) == Some("user")),
    );

    let (ids, thinking) = encode_turn(
        state,
        messages,
        tools,
        open_thinking,
        template_kwargs,
        starts[0],
        system_count,
    )
    .await?;
    if ids.is_empty() {
        return Err(error(StatusCode::BAD_REQUEST, "chat rendered no tokens"));
    }
    let Some(budget) = max_prompt_tokens else {
        return Ok((ids, 0, thinking));
    };
    if ids.len() <= budget {
        return Ok((ids, 0, thinking));
    }

    let (low, high) = (1usize, starts.len() - 1);
    let mut best: Option<(Vec<u32>, usize, bool)> = None;
    let (mut low, mut high) = (low, high);
    while low <= high {
        let middle = (low + high) / 2;
        let start = starts[middle];
        let (candidate, candidate_thinking) = encode_turn(
            state,
            messages,
            tools,
            open_thinking,
            template_kwargs,
            start,
            system_count,
        )
        .await?;
        if candidate.len() <= budget {
            best = Some((candidate, start - system_count, candidate_thinking));
            high = middle - 1;
        } else {
            low = middle + 1;
        }
    }
    if let Some(best) = best {
        return Ok(best);
    }
    let (newest, _) = encode_turn(
        state,
        messages,
        tools,
        open_thinking,
        template_kwargs,
        starts[starts.len() - 1],
        system_count,
    )
    .await?;
    Err(error(
        StatusCode::BAD_REQUEST,
        format!(
            "system prompt and latest turn require {} prompt tokens, but only {budget} remain after reserving output tokens",
            newest.len()
        ),
    ))
}

fn request_lora(request: &ChatCompletionRequest) -> Option<Value> {
    for key in ["lora_request", "lora", "lora_name"] {
        if let Some(value) = request.extra.get(key) {
            match value {
                Value::Null => continue,
                Value::String(name) if !name.is_empty() => {
                    return Some(json!({
                        "lora_name": name,
                        "lora_int_id": 0,
                        "lora_path": "",
                        "scale": 1.0,
                    }))
                }
                Value::Object(_) => return Some(value.clone()),
                _ => continue,
            }
        }
    }
    None
}

async fn prepare_chat(
    state: &SharedState,
    request: &ChatCompletionRequest,
) -> Result<Prepared, Response> {
    if request.model != state.served_model_name {
        return Err(error(StatusCode::NOT_FOUND, "model not found"));
    }
    let (tools, forced_tool) =
        normalized_tools(request.tools.as_deref(), request.tool_choice.as_ref())
            .map_err(|message| error(StatusCode::BAD_REQUEST, message))?;
    let template_kwargs = request
        .template_extras()
        .map_err(|message| error(StatusCode::BAD_REQUEST, message))?;
    let (format_mode, format_schema) = normalized_response_format(request.response_format.as_ref())
        .map_err(|message| error(StatusCode::BAD_REQUEST, message))?;
    // Best-effort validation of the generated text. Engine-side structured
    // decoding does the real enforcement; this only feeds the
    // `response_format_invalid` diagnostic when the model still drifts.
    let format_validator = match (format_mode.as_str(), format_schema.as_deref()) {
        ("json_object", _) => Some(JsonFormatValidator::new(None)),
        ("json_schema", Some(schema)) => serde_json::from_str::<Value>(schema)
            .ok()
            .map(|value| JsonFormatValidator::new(Some(value))),
        _ => None,
    };

    // A remote URL is normalized to an inline data URL before the synchronous
    // media parser runs; `data:` URLs and plain-text chats skip this entirely.
    let mut messages = request.messages.clone();
    if has_media_parts(&messages) {
        messages = crate::remote_media::resolve_remote_media(&messages, &state.fetch_config)
            .await
            .map_err(|failure| {
                (
                    StatusCode::from_u16(failure.status).unwrap_or(StatusCode::BAD_REQUEST),
                    Json(json!({
                        "detail": failure.message,
                        "code": failure.code,
                    })),
                )
                    .into_response()
            })?;
    }

    // Multimodal requests keep their raw messages: the worker renders the chat
    // template and expands media placeholders against the sealed descriptors.
    let mut multimodal_plan = if has_media_parts(&messages) {
        let parsed = parse_media_chat(&messages)
            .map_err(|message| error(StatusCode::BAD_REQUEST, message))?;
        let output_modalities = request
            .extra
            .get("output_modalities")
            .and_then(Value::as_array)
            .map(|items| {
                items
                    .iter()
                    .filter_map(|item| item.as_str().map(str::to_string))
                    .collect::<Vec<_>>()
            })
            .filter(|items| !items.is_empty())
            .unwrap_or_else(|| vec!["text".into()]);
        let audio_output = request
            .extra
            .get("audio_output")
            .cloned()
            .filter(|value| !value.is_null());
        Some(MultimodalPlan {
            parsed,
            audio_output,
            output_modalities,
        })
    } else {
        None
    };
    let lora = request_lora(request);
    if lora.is_some() && !state.enable_lora {
        return Err(error(StatusCode::BAD_REQUEST, "LoRA is disabled"));
    }
    if lora.is_some() && multimodal_plan.is_some() {
        return Err(error(
            StatusCode::BAD_REQUEST,
            "LoRA is only supported on text-only requests",
        ));
    }

    if let Some(plan) = &multimodal_plan {
        messages = plan.parsed.messages.clone();
    }
    if !format_mode.is_empty() {
        messages.insert(
            0,
            json!({
                "role": "system",
                "content": format!(
                    "Respond with a single valid {} value and nothing else.",
                    if format_mode == "json_object" { "JSON object" } else { "JSON value matching the schema" }
                ),
            }),
        );
    }
    let max_new_tokens = request.max_new_tokens();
    let (token_ids, dropped_messages, initial_in_thinking) =
        if let Some(plan) = &mut multimodal_plan {
            plan.parsed.messages = messages.clone();
            // Best-effort local render only seeds the output parser. The worker
            // renders the authoritative prompt and reports its prompt token count.
            let seed = state
                .tokenizer
                .render_chat_with(
                    &messages,
                    if tools.is_empty() { None } else { Some(&tools) },
                    request.open_thinking,
                    template_kwargs.as_ref(),
                )
                .ok();
            match seed {
                Some(rendered) => {
                    let ids = state
                        .tokenizer_batcher
                        .encode(rendered.clone())
                        .await
                        .unwrap_or_default();
                    let thinking = in_thinking_from(state, &ids, &rendered);
                    (ids, 0, thinking)
                }
                None => (Vec::new(), 0, false),
            }
        } else {
            let budget = state.max_model_len.saturating_sub(max_new_tokens as usize);
            if budget == 0 {
                return Err(error(
                    StatusCode::BAD_REQUEST,
                    "max_tokens leaves no prompt budget",
                ));
            }
            fit_text_history(
                state,
                &messages,
                &tools,
                request.open_thinking,
                template_kwargs.as_ref(),
                Some(budget),
            )
            .await?
        };

    let structured = if format_mode == "json_schema" {
        format_schema.map(|schema| json!({"kind": "json_schema", "schema_json": schema}))
    } else if format_mode == "json_object" {
        Some(json!({"kind": "json_object"}))
    } else {
        None
    };
    let mut sampling = request.sampling(structured);
    if let Some(forced) = &forced_tool {
        if let Value::Object(map) = &mut sampling {
            map.insert("forced_tool".into(), json!(forced));
        }
    }

    Ok(Prepared {
        base_request_id: request
            .cancel_id
            .clone()
            .unwrap_or_else(|| uuid::Uuid::new_v4().simple().to_string()),
        format_validator,
        prompt_tokens: token_ids.len(),
        token_ids,
        dropped_messages,
        initial_in_thinking,
        marker_kinds: marker_kinds(state.tokenizer),
        sampling,
        tools,
        template_kwargs,
        forced_tool,
        stop_sequences: request.stop_sequences(),
        include_usage: request.include_usage(),
        created: now(),
        lora,
        multimodal: multimodal_plan,
    })
}

fn choice_request_id(base: &str, index: usize) -> String {
    if index == 0 {
        base.to_string()
    } else {
        format!("{base}-c{index}")
    }
}

fn check_results(
    result: megalm_protocol::SubmitResult,
    fallback: usize,
) -> Result<HashMap<String, usize>, Response> {
    let mut prompt_tokens = HashMap::new();
    for item in result.results {
        if !item.ok {
            let message = item
                .error
                .as_ref()
                .and_then(|err| err.get("message"))
                .and_then(Value::as_str)
                .unwrap_or("submission rejected");
            return Err(error(StatusCode::BAD_REQUEST, message));
        }
        prompt_tokens.insert(
            item.request_id,
            item.prompt_tokens.unwrap_or(fallback as u64) as usize,
        );
    }
    Ok(prompt_tokens)
}

async fn submit_choices(
    state: &SharedState,
    prepared: &Prepared,
    max_new_tokens: u64,
    choices: &[(String, u64)],
) -> Result<HashMap<String, usize>, Response> {
    let submissions = choices
        .iter()
        .map(|(request_id, stream_id)| Submission {
            request_id: request_id.clone(),
            stream_id: *stream_id,
            token_ids: prepared.token_ids.clone(),
            max_new_tokens,
            sampling: prepared.sampling.clone(),
            lora: prepared.lora.clone(),
            output_modalities: vec!["text".into()],
            audio_output: None,
        })
        .collect();
    let batch = SubmitBatch {
        command_id: 0, // overwritten by the client with the correlated id
        target_active_requests: choices.len(),
        submissions,
    };
    let result = state
        .client
        .submit_batch(&batch)
        .await
        .map_err(|err| error(StatusCode::SERVICE_UNAVAILABLE, err.to_string()))?;
    check_results(result, prepared.prompt_tokens)
}

/// Seal each distinct media payload once, then pass a duplicated descriptor to
/// every choice. The worker requires contiguous, non-overlapping ranges and
/// closes the descriptors after reading them.
async fn submit_multimodal_choices(
    state: &SharedState,
    prepared: &Prepared,
    request: &ChatCompletionRequest,
    choices: &[(String, u64)],
) -> Result<HashMap<String, usize>, Response> {
    let plan = prepared.multimodal.as_ref().ok_or_else(|| {
        error(
            StatusCode::INTERNAL_SERVER_ERROR,
            "multimodal plan is missing",
        )
    })?;

    let mut sealed: Vec<(MediaDescriptor, OwnedFd)> = Vec::with_capacity(plan.parsed.inputs.len());
    for input in &plan.parsed.inputs {
        let fd = seal_memfd(&input.payload)
            .map_err(|err| error(StatusCode::BAD_REQUEST, err.to_string()))?;
        sealed.push((
            MediaDescriptor {
                kind: input.kind.as_str().to_string(),
                mime_type: input.mime_type.clone(),
                size: input.payload.len(),
                sha256: input.sha256.clone(),
                detail: input.detail.clone(),
                sampling: input.sampling.clone(),
            },
            fd,
        ));
    }

    let mut descriptors = Vec::with_capacity(sealed.len() * choices.len());
    let mut fds: Vec<OwnedFd> = Vec::with_capacity(sealed.len() * choices.len());
    let mut submissions = Vec::with_capacity(choices.len());
    for (request_id, stream_id) in choices {
        let media_start = descriptors.len();
        for (descriptor, fd) in &sealed {
            descriptors.push(descriptor.clone());
            fds.push(
                duplicate_fd(fd)
                    .map_err(|err| error(StatusCode::SERVICE_UNAVAILABLE, err.to_string()))?,
            );
        }
        submissions.push(MultimodalSubmission {
            request_id: request_id.clone(),
            stream_id: *stream_id,
            messages: plan.parsed.messages.clone(),
            media_start,
            media_count: sealed.len(),
            max_new_tokens: request.max_new_tokens(),
            sampling: prepared.sampling.clone(),
            open_thinking: request.open_thinking,
            chat_template_kwargs: prepared.template_kwargs.clone(),
            tools: prepared.tools.clone(),
            output_modalities: plan.output_modalities.clone(),
            audio_output: plan.audio_output.clone(),
        });
    }

    let batch = SubmitMultimodalBatch {
        command_id: 0, // overwritten by the client with the correlated id
        target_active_requests: choices.len(),
        submissions,
        media: descriptors,
    };
    let result = state
        .client
        .submit_multimodal_batch(&batch, fds)
        .await
        .map_err(|err| error(StatusCode::SERVICE_UNAVAILABLE, err.to_string()))?;
    check_results(result, 0)
}

fn choice_context(
    state: &SharedState,
    prepared: &Prepared,
    request_id: String,
    index: usize,
    prompt_tokens: usize,
) -> Result<StreamContext, Response> {
    let parser = make_parser(
        state,
        &request_id,
        prepared.stop_sequences.clone(),
        &prepared.tools,
        prepared.forced_tool.as_deref(),
        prepared.initial_in_thinking,
        &prepared.token_ids,
        &prepared.marker_kinds,
    )
    .map_err(|message| error(StatusCode::BAD_REQUEST, message))?;
    Ok(StreamContext {
        request_id,
        model: state.served_model_name.clone(),
        created: prepared.created,
        index,
        prompt_tokens,
        include_usage: prepared.include_usage,
        parser,
        format_validator: prepared.format_validator.clone(),
        profiler: state.profiler.clone(),
        metrics: state.metrics.clone(),
        started: std::time::Instant::now(),
        lease_tracker: Some(state.lease_tracker.clone()),
    })
}

/// Outcome of the shared chat dispatcher: either a collected response or a
/// byte stream of OpenAI-compatible SSE frames.
pub(crate) enum ChatOutcome {
    Json(Value),
    Stream(
        std::pin::Pin<
            Box<dyn futures::Stream<Item = Result<Bytes, std::convert::Infallible>> + Send>,
        >,
    ),
}

pub(crate) async fn dispatch_chat(
    state: SharedState,
    request: ChatCompletionRequest,
) -> Result<ChatOutcome, Response> {
    let prepared = match prepare_chat(&state, &request).await {
        Ok(prepared) => prepared,
        Err(response) => return Err(response),
    };
    let count = request.n.unwrap_or(1).clamp(1, 128);
    let prefix_fingerprint = crate::control_plane::fingerprint_token_ids(&prepared.token_ids, 16);
    let choices = (0..count)
        .map(|index| {
            (
                choice_request_id(&prepared.base_request_id, index),
                state.client.next_stream_id(),
            )
        })
        .collect::<Vec<_>>();

    // Subscribe before submitting so no output frame is lost.
    let mut receivers = Vec::with_capacity(choices.len());
    for (_, stream_id) in &choices {
        receivers.push(state.client.subscribe(*stream_id).await);
    }
    let prompt_tokens = if prepared.multimodal.is_some() {
        submit_multimodal_choices(&state, &prepared, &request, &choices).await
    } else {
        submit_choices(&state, &prepared, request.max_new_tokens(), &choices).await
    };
    let prompt_tokens = match prompt_tokens {
        Ok(prompt_tokens) => prompt_tokens,
        Err(response) => {
            for (_, stream_id) in &choices {
                state.client.unsubscribe(*stream_id).await;
            }
            return Err(response);
        }
    };
    // One admitted request per accepted submission, mirrored into both the
    // `proxy` (gateway) and `replica` (engine) metric namespaces.
    state.metrics.admit_request();
    for (request_id, _) in &choices {
        let count = prompt_tokens.get(request_id).copied().unwrap_or(0);
        state
            .profiler
            .record("admitted", request_id, json!({"prompt_tokens": count}));
        state.lease_tracker.begin(request_id, &prefix_fingerprint);
    }

    let contexts = choices
        .iter()
        .enumerate()
        .map(|(index, (request_id, _))| {
            let count = prompt_tokens
                .get(request_id)
                .copied()
                .unwrap_or(prepared.prompt_tokens);
            choice_context(&state, &prepared, request_id.clone(), index, count)
        })
        .collect::<Result<Vec<_>, _>>();
    let contexts = match contexts {
        Ok(contexts) => contexts,
        Err(response) => {
            for (request_id, _) in &choices {
                state.lease_tracker.finish(request_id, true);
            }
            for (_, stream_id) in &choices {
                state.client.unsubscribe(*stream_id).await;
            }
            return Err(response);
        }
    };

    if request.stream {
        let (frame_tx, mut frame_rx) = mpsc::unbounded_channel::<Vec<u8>>();
        for (context, events) in contexts.into_iter().zip(receivers) {
            let tx = frame_tx.clone();
            let decoder = owned_decoder(state.tokenizer, !state.preserve_special_tokens);
            let tracker = state.lease_tracker.clone();
            let request_id = context.request_id.clone();
            tokio::spawn(async move {
                let stream = chat_stream(context, decoder, events);
                tokio::pin!(stream);
                while let Some(bytes) = stream.next().await {
                    let bytes = bytes.unwrap_or_default();
                    // Only the outermost stream emits the DONE terminator.
                    if bytes.as_ref() != b"data: [DONE]\n\n" {
                        let _ = tx.send(bytes.to_vec());
                    }
                }
                tracker.finish(&request_id, false);
            });
        }
        drop(frame_tx);
        let stream = async_stream::stream! {
            while let Some(bytes) = frame_rx.recv().await {
                yield Ok::<Bytes, std::convert::Infallible>(Bytes::from(bytes));
            }
            yield Ok::<Bytes, std::convert::Infallible>(Bytes::from_static(b"data: [DONE]\n\n"));
        };
        Ok(ChatOutcome::Stream(Box::pin(stream)))
    } else {
        let mut tasks = Vec::with_capacity(contexts.len());
        for (context, events) in contexts.into_iter().zip(receivers) {
            let decoder = owned_decoder(state.tokenizer, !state.preserve_special_tokens);
            let tracker = state.lease_tracker.clone();
            let request_id = context.request_id.clone();
            tasks.push(tokio::spawn(async move {
                let response = collect_chat(context, decoder, events).await;
                let failed = response["choices"][0]["finish_reason"] == "error";
                tracker.finish(&request_id, failed);
                response
            }));
        }
        let mut responses = Vec::with_capacity(tasks.len());
        for task in tasks {
            match task.await {
                Ok(response) => responses.push(response),
                Err(_) => {
                    return Err(error(
                        StatusCode::INTERNAL_SERVER_ERROR,
                        "choice task failed",
                    ))
                }
            }
        }
        for (_, stream_id) in &choices {
            state.client.unsubscribe(*stream_id).await;
        }
        let mut response = merge_collected(responses);
        response["megalm"] = json!({"dropped_messages": prepared.dropped_messages});
        Ok(ChatOutcome::Json(response))
    }
}

async fn chat_completions(
    State(state): State<SharedState>,
    Json(payload): Json<Value>,
) -> Response {
    if let Some(response) = check_paused(&state) {
        return response;
    }
    if let Some(router) = &state.replica_router {
        let mut body = payload.clone();
        strip_nulls(&mut body);
        return match router
            .route(body, "/v1/chat/completions", &state.profiler)
            .await
        {
            Ok(response) => response,
            Err(message) => error(StatusCode::BAD_GATEWAY, message),
        };
    }
    if let Some(pd) = &state.pd {
        let mut body = payload.clone();
        strip_nulls(&mut body);
        return match pd
            .route(body, "/v1/chat/completions", &state.profiler)
            .await
        {
            Ok(response) => response,
            Err(message) => error(StatusCode::BAD_GATEWAY, message),
        };
    }
    let request: ChatCompletionRequest = match serde_json::from_value(payload) {
        Ok(request) => request,
        Err(err) => return error(StatusCode::BAD_REQUEST, err.to_string()),
    };
    match dispatch_chat(state, request).await {
        Ok(ChatOutcome::Json(value)) => Json(value).into_response(),
        Ok(ChatOutcome::Stream(stream)) => Body::from_stream(stream).into_response(),
        Err(response) => response,
    }
}

async fn completions(State(state): State<SharedState>, Json(payload): Json<Value>) -> Response {
    let started = std::time::Instant::now();
    if let Some(response) = check_paused(&state) {
        return response;
    }
    if let Some(router) = &state.replica_router {
        let mut body = payload.clone();
        strip_nulls(&mut body);
        return match router.route(body, "/v1/completions", &state.profiler).await {
            Ok(response) => response,
            Err(message) => error(StatusCode::BAD_GATEWAY, message),
        };
    }
    if let Some(pd) = &state.pd {
        let mut body = payload.clone();
        strip_nulls(&mut body);
        return match pd.route(body, "/v1/completions", &state.profiler).await {
            Ok(response) => response,
            Err(message) => error(StatusCode::BAD_GATEWAY, message),
        };
    }
    let request: CompletionRequest = match serde_json::from_value(payload) {
        Ok(request) => request,
        Err(err) => return error(StatusCode::BAD_REQUEST, err.to_string()),
    };
    if request.model != state.served_model_name {
        return error(StatusCode::NOT_FOUND, "model not found");
    }
    let prompt = request.prompt.as_str().unwrap_or("");
    let token_ids = match state.tokenizer_batcher.encode(prompt.to_string()).await {
        Ok(ids) => ids,
        Err(err) => return error(StatusCode::BAD_REQUEST, err),
    };
    let request_id = request
        .cancel_id
        .clone()
        .unwrap_or_else(|| uuid::Uuid::new_v4().simple().to_string());
    let stream_id = state.client.next_stream_id();
    let sampling = ChatCompletionRequest {
        temperature: request.temperature,
        top_p: request.top_p,
        ignore_eos: request.ignore_eos,
        ..ChatCompletionRequest::default()
    }
    .sampling(None);
    let batch = SubmitBatch {
        command_id: 0,
        target_active_requests: 1,
        submissions: vec![Submission {
            request_id: request_id.clone(),
            stream_id,
            token_ids: token_ids.clone(),
            max_new_tokens: request.max_tokens.unwrap_or(256),
            sampling,
            lora: None,
            output_modalities: vec!["text".into()],
            audio_output: None,
        }],
    };
    let events = state.client.subscribe(stream_id).await;
    if let Err(response) = state
        .client
        .submit_batch(&batch)
        .await
        .map_err(|err| error(StatusCode::SERVICE_UNAVAILABLE, err.to_string()))
        .and_then(|result| check_results(result, token_ids.len()))
    {
        state.client.unsubscribe(stream_id).await;
        return response;
    }
    let fingerprint = crate::control_plane::fingerprint_token_ids(&token_ids, 16);
    state.lease_tracker.begin(&request_id, &fingerprint);
    state.metrics.admit_request();
    if request.stream {
        let usage_prompt_tokens = request
            .extra
            .get("stream_options")
            .and_then(|options| options.get("include_usage"))
            .and_then(Value::as_bool)
            .unwrap_or(false)
            .then_some(token_ids.len());
        let decoder = owned_decoder(state.tokenizer, true);
        let tracker = state.lease_tracker.clone();
        let stream_id_done = request_id.clone();
        let mut inner = Box::pin(completion_stream(
            request_id,
            state.served_model_name.clone(),
            now(),
            events,
            decoder,
            Some(state.lease_tracker.clone()),
            usage_prompt_tokens,
            state.metrics.clone(),
            started,
        ));
        let stream = async_stream::stream! {
            while let Some(bytes) = inner.next().await {
                yield bytes;
            }
            tracker.finish(&stream_id_done, false);
        };
        return (
            [(axum::http::header::CONTENT_TYPE, "text/event-stream")],
            Body::from_stream(stream),
        )
            .into_response();
    }
    let mut decoder = owned_decoder(state.tokenizer, true);
    let mut text = String::new();
    let mut completion_tokens = 0usize;
    let mut first_token_recorded = false;
    let mut ttft_ms: Option<f64> = None;
    let mut finish_reason = "stop".to_owned();
    let mut events = events;
    while let Some(event) = events.recv().await {
        let delta = decoder.push(&event.token_ids);
        if !first_token_recorded && !delta.is_empty() {
            first_token_recorded = true;
            let elapsed = started.elapsed().as_secs_f64() * 1_000.0;
            state.metrics.record_ttft_ms(elapsed);
            ttft_ms = Some(elapsed);
            state.lease_tracker.first_token(&request_id);
        }
        text.push_str(&delta);
        completion_tokens += event.token_ids.len();
        if let Some(reason) = event.finish_reason {
            finish_reason = crate::stream::map_finish_reason(&reason);
            break;
        }
        if event.error.is_some() {
            finish_reason = "error".into();
            break;
        }
    }
    let elapsed = started.elapsed().as_secs_f64() * 1_000.0;
    if let (Some(ttft), true) = (ttft_ms, completion_tokens > 1) {
        state
            .metrics
            .record_tpot_ms((elapsed - ttft) / (completion_tokens - 1) as f64);
    }
    state.metrics.record_e2e_ms(elapsed);
    state.metrics.record_finish_reason(&finish_reason);
    state.lease_tracker.finish(&request_id, false);
    state.client.unsubscribe(stream_id).await;
    let response = json!({
        "id": format!("cmpl-{request_id}"),
        "object": "text_completion",
        "created": now(),
        "model": state.served_model_name,
        "choices": [{"index": 0, "text": text, "finish_reason": "stop"}],
        "usage": {
            "prompt_tokens": token_ids.len(),
            "completion_tokens": completion_tokens,
            "total_tokens": token_ids.len() + completion_tokens,
        },
    });
    Json(response).into_response()
}

/// OpenAI-compatible dense embeddings. The embedding worker never enters the
/// generation scheduler: one padded encoder forward produces one vector per
/// input, in request order.
async fn embeddings(State(state): State<SharedState>, Json(payload): Json<Value>) -> Response {
    if let Some(response) = check_paused(&state) {
        return response;
    }
    if let Some(router) = &state.replica_router {
        let mut body = payload.clone();
        strip_nulls(&mut body);
        return match router.route(body, "/v1/embeddings", &state.profiler).await {
            Ok(response) => response,
            Err(message) => error(StatusCode::BAD_GATEWAY, message),
        };
    }
    let request: EmbeddingRequest = match serde_json::from_value(payload) {
        Ok(request) => request,
        Err(err) => return error(StatusCode::BAD_REQUEST, err.to_string()),
    };
    if let Err(message) = request.validate() {
        return error(StatusCode::BAD_REQUEST, message);
    }
    if !request.model.is_empty() && request.model != state.served_model_name {
        return error(StatusCode::NOT_FOUND, "model not found");
    }
    let (submissions, prompt_tokens): (Vec<EmbeddingSubmission>, usize) = match request.input() {
        Ok(EmbeddingInput::Texts(texts)) => {
            let encoded = match state
                .tokenizer
                .encode_batch_with_special(&texts, request.add_special_tokens)
            {
                Ok(encoded) => encoded,
                Err(err) => return error(StatusCode::BAD_REQUEST, err.to_string()),
            };
            let tokens = encoded.iter().map(Vec::len).sum();
            (
                encoded
                    .into_iter()
                    .map(|ids| EmbeddingSubmission {
                        request_id: uuid::Uuid::new_v4().simple().to_string(),
                        token_ids: ids,
                    })
                    .collect(),
                tokens,
            )
        }
        Ok(EmbeddingInput::TokenIds(rows)) => {
            let tokens = rows.iter().map(Vec::len).sum();
            (
                rows.into_iter()
                    .map(|ids| EmbeddingSubmission {
                        request_id: uuid::Uuid::new_v4().simple().to_string(),
                        token_ids: ids,
                    })
                    .collect(),
                tokens,
            )
        }
        Err(message) => return error(StatusCode::BAD_REQUEST, message),
    };
    let batch = SubmitEmbeddings {
        command_id: 0,
        submissions,
        dimensions: request.dimensions,
        normalize: request.use_activation,
        truncate_prompt_tokens: request.truncate_prompt_tokens,
        truncation_side: request.truncation_side.clone(),
    };
    let result = match state.client.embed(&batch).await {
        Ok(result) => result,
        Err(err) => return error(StatusCode::SERVICE_UNAVAILABLE, err.to_string()),
    };
    if result.results.len() != batch.submissions.len() {
        return error(
            StatusCode::INTERNAL_SERVER_ERROR,
            "embedding worker returned an incomplete batch",
        );
    }
    let mut data = Vec::with_capacity(result.results.len());
    for (index, item) in result.results.into_iter().enumerate() {
        if !item.ok {
            let message = item
                .error
                .as_ref()
                .and_then(|err| err.get("message"))
                .and_then(Value::as_str)
                .unwrap_or("embedding request failed");
            return error(StatusCode::BAD_REQUEST, message);
        }
        let embedding = if request.encoding_format == "base64" {
            let mut bytes = Vec::with_capacity(item.embedding.len() * 4);
            for value in item.embedding {
                bytes.extend_from_slice(&value.to_le_bytes());
            }
            json!(base64::engine::general_purpose::STANDARD.encode(bytes))
        } else {
            json!(item.embedding)
        };
        data.push(json!({
            "object": "embedding",
            "index": index,
            "embedding": embedding,
        }));
    }
    Json(json!({
        "object": "list",
        "data": data,
        "model": state.served_model_name,
        "usage": {
            "prompt_tokens": prompt_tokens,
            "total_tokens": prompt_tokens,
        },
    }))
    .into_response()
}

async fn load_lora(State(state): State<SharedState>, Json(payload): Json<Value>) -> Response {
    let path = payload
        .get("lora_path")
        .or_else(|| payload.get("path"))
        .and_then(Value::as_str);
    let Some(path) = path else {
        return error(StatusCode::BAD_REQUEST, "lora_path is required");
    };
    let request = Inspect {
        command_id: 0,
        op: Some("load_lora".into()),
        path: Some(path.into()),
        name: payload
            .get("lora_name")
            .or_else(|| payload.get("name"))
            .and_then(Value::as_str)
            .map(str::to_string),
        id: payload
            .get("lora_id")
            .or_else(|| payload.get("id"))
            .cloned(),
        ..Inspect::default()
    };
    match state.client.inspect(&request).await {
        Ok(value) => {
            let lora = value
                .get("snapshot")
                .and_then(|snapshot| snapshot.get("lora"))
                .or_else(|| value.get("lora"))
                .cloned();
            Json(json!({"status": "success", "lora": lora})).into_response()
        }
        Err(err) => error(StatusCode::CONFLICT, err.to_string()),
    }
}

async fn unload_lora(State(state): State<SharedState>, Json(payload): Json<Value>) -> Response {
    let request = Inspect {
        command_id: 0,
        op: Some("unload_lora".into()),
        id: payload
            .get("lora_id")
            .cloned()
            .or_else(|| payload.get("lora_name").cloned()),
        ..Inspect::default()
    };
    match state.client.inspect(&request).await {
        Ok(_) => Json(json!({"status": "success"})).into_response(),
        Err(err) => error(StatusCode::CONFLICT, err.to_string()),
    }
}

async fn unload_lora_path(
    State(state): State<SharedState>,
    Path(reference): Path<String>,
) -> Response {
    let request = Inspect {
        command_id: 0,
        op: Some("unload_lora".into()),
        name: Some(reference),
        ..Inspect::default()
    };
    match state.client.inspect(&request).await {
        Ok(_) => Json(json!({"status": "success"})).into_response(),
        Err(err) => error(StatusCode::CONFLICT, err.to_string()),
    }
}

async fn list_loras(State(state): State<SharedState>) -> Response {
    let descriptor = state.client.descriptor().await;
    Json(descriptor.get("lora").cloned().unwrap_or(json!({}))).into_response()
}

async fn cancel(State(state): State<SharedState>, Json(payload): Json<Value>) -> Response {
    let Some(id) = payload.get("cancel_id").and_then(Value::as_str) else {
        return error(StatusCode::BAD_REQUEST, "cancel_id is required");
    };
    match state.client.cancel(id).await {
        Ok(true) => Json(json!({"status": "cancelled", "cancel_id": id})).into_response(),
        Ok(false) => error(StatusCode::NOT_FOUND, "request is not active"),
        Err(err) => error(StatusCode::SERVICE_UNAVAILABLE, err.to_string()),
    }
}

async fn validate(State(state): State<SharedState>, Json(payload): Json<Value>) -> Response {
    let path = payload
        .get("model_path")
        .and_then(Value::as_str)
        .unwrap_or("");
    let valid = path == state.served_model_name;
    Json(json!({
        "valid": valid,
        "message": if valid { "ok" } else { "model is not loaded" },
        "identifier": state.served_model_name,
        "display_name": state.served_model_name,
        "is_gguf": false,
        "is_lora": false,
        "requires_trust_remote_code": false,
    }))
    .into_response()
}

async fn load_model(State(state): State<SharedState>, Json(payload): Json<Value>) -> Response {
    let path = payload
        .get("model_path")
        .and_then(Value::as_str)
        .unwrap_or("");
    if path != state.served_model_name {
        return error(StatusCode::CONFLICT, "another model is loaded");
    }
    Json(json!({"status": "ready", "model": state.served_model_name})).into_response()
}

async fn inference_status(State(state): State<SharedState>) -> Response {
    let descriptor = state.client.descriptor().await;
    Json(json!({
        "status": "ready",
        "model": state.served_model_name,
        "capabilities": descriptor.get("input_modalities").cloned().unwrap_or(json!([])),
    }))
    .into_response()
}

async fn models_list(State(state): State<SharedState>) -> Response {
    Json(json!({
        "models": [{
            "id": state.served_model_name,
            "name": state.served_model_name,
            "is_vision": false,
            "is_lora": false,
            "is_gguf": false,
            "is_audio": false,
        }],
        "default_models": [state.served_model_name],
    }))
    .into_response()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn profiling_paths_splits_and_ignores_empty_entries() {
        let paths = profiling_paths(Some(" a.jsonl , , /tmp/dir/*.jsonl "));
        assert_eq!(paths.len(), 2);
        assert_eq!(paths[0], std::path::PathBuf::from("a.jsonl"));
        assert_eq!(paths[1], std::path::PathBuf::from("/tmp/dir/*.jsonl"));
        assert!(profiling_paths(None).is_empty());
        assert!(profiling_paths(Some("   ")).is_empty());
    }

    #[test]
    fn metrics_include_profiling_counters() {
        let snapshot = with_profiling_metrics(
            json!({"proxy": {"admitted_requests": 3}}),
            &crate::profiling::Profiler::disabled(),
        );
        let text =
            megalm_rs::metrics::render_prometheus_metrics(&json!(snapshot).to_string()).unwrap();
        assert!(text.contains("megalm_profiling_events_recorded 0\n"));
        assert!(text.contains("megalm_profiling_dropped_writes 0\n"));
        assert!(text.contains("megalm_profiling_dropped_metadata 0\n"));
        // The original client metrics must survive unchanged.
        assert!(text.contains("megalm_proxy_admitted_requests 3\n"));
    }

    #[test]
    fn metrics_include_live_proxy_and_replica_counters() {
        let metrics = crate::replica_metrics::ReplicaMetrics::new();
        metrics.admit_request();
        metrics.record_ttft_ms(12.0);
        metrics.record_e2e_ms(30.0);
        metrics.record_finish_reason("stop");
        let mut stats = megalm_rs::parser::ParserStats::default();
        stats.pushes = 4;
        stats.bytes = 64;
        metrics.record_parser(&stats);

        let snapshot = with_replica_metrics(
            with_profiling_metrics(
                json!({"worker": {"admitted_requests": 1}}),
                &crate::profiling::Profiler::disabled(),
            ),
            &metrics,
        );
        let text =
            megalm_rs::metrics::render_prometheus_metrics(&json!(snapshot).to_string()).unwrap();
        // Worker metrics survive the fold.
        assert!(text.contains("megalm_worker_admitted_requests 1\n"));
        // Live producers replaced the test-only injected value.
        assert_eq!(
            text.matches("megalm_proxy_admitted_requests 1\n").count(),
            1
        );
        assert!(
            text.contains("megalm_replica_ttft_ms_mean 12.0\n"),
            "{text}"
        );
        assert!(text.contains("megalm_replica_e2e_ms_mean 30.0\n"), "{text}");
        assert!(text.contains("megalm_replica_ttft_ms_histogram_count 1\n"));
        assert!(text.contains("megalm_replica_finish_reasons_stop 1\n"));
        assert!(text.contains("megalm_replica_parser_pushes 4\n"));
        assert!(text.contains("megalm_replica_parser_bytes 64\n"));
        assert!(text.contains("# TYPE megalm_replica_parser_bytes counter\n"));
    }
}
