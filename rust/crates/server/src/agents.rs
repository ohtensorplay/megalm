//! Anthropic Messages and OpenAI Responses adapters over the chat dispatcher.

use std::convert::Infallible;

use async_stream::stream;
use axum::body::{Body, Bytes};
use axum::extract::{Path, State};
use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use axum::Json;
use futures::StreamExt;
use serde_json::{json, Value};

use crate::app::{dispatch_chat, ChatOutcome, SharedState};
use crate::types::ChatCompletionRequest;

fn bad_request(message: impl Into<String>) -> Response {
    (
        StatusCode::BAD_REQUEST,
        Json(json!({"type": "error", "error": {"type": "invalid_request_error", "message": message.into()}})),
    )
        .into_response()
}

fn openai_error(status: StatusCode, message: impl Into<String>) -> Response {
    let status_code = status.as_u16();
    (
        status,
        Json(json!({
            "error": {"message": message.into(), "type": "invalid_request_error", "code": status_code}
        })),
    )
        .into_response()
}

fn anthropic_content(value: &Value) -> Result<String, String> {
    match value {
        Value::String(text) => Ok(text.clone()),
        Value::Array(parts) => {
            let mut text = String::new();
            for part in parts {
                if part.get("type").and_then(Value::as_str) == Some("text") {
                    text.push_str(part.get("text").and_then(Value::as_str).unwrap_or(""));
                }
            }
            Ok(text)
        }
        _ => Err("content must be text or a content-part array".into()),
    }
}

fn anthropic_tools(value: Option<&Value>) -> Option<Vec<Value>> {
    let items = value?.as_array()?;
    let tools = items
        .iter()
        .filter_map(|tool| {
            let name = tool.get("name")?.as_str()?;
            Some(json!({
                "type": "function",
                "function": {
                    "name": name,
                    "description": tool.get("description").cloned().unwrap_or(Value::Null),
                    "parameters": tool.get("input_schema").cloned().unwrap_or(json!({})),
                }
            }))
        })
        .collect::<Vec<_>>();
    (!tools.is_empty()).then_some(tools)
}

fn anthropic_to_chat(payload: &Value, model: &str) -> Result<ChatCompletionRequest, String> {
    let mut messages = Vec::new();
    if let Some(system) = payload.get("system") {
        let text = anthropic_content(system)?;
        if !text.is_empty() {
            messages.push(json!({"role": "system", "content": text}));
        }
    }
    let input = payload
        .get("messages")
        .and_then(Value::as_array)
        .ok_or("messages must be an array")?;
    for message in input {
        let role = message
            .get("role")
            .and_then(Value::as_str)
            .ok_or("message has no role")?;
        messages.push(json!({"role": role, "content": anthropic_content(message.get("content").unwrap_or(&Value::Null))?}));
    }
    let mut request = json!({
        "model": model,
        "messages": messages,
        "stream": payload.get("stream").and_then(Value::as_bool).unwrap_or(false),
    });
    if let Some(Value::Number(max)) = payload.get("max_tokens") {
        request["max_tokens"] = json!(max);
    }
    if let Some(tools) = anthropic_tools(payload.get("tools")) {
        request["tools"] = json!(tools);
    }
    for (from, to) in [("temperature", "temperature"), ("top_p", "top_p")] {
        if let Some(value) = payload.get(from) {
            request[to] = value.clone();
        }
    }
    if let Some(stops) = payload.get("stop_sequences") {
        request["stop"] = stops.clone();
    }
    serde_json::from_value(request).map_err(|err| err.to_string())
}

fn chat_to_anthropic(body: &Value, model: &str) -> Value {
    let choice = body
        .get("choices")
        .and_then(Value::as_array)
        .and_then(|items| items.first());
    let message = choice
        .and_then(|item| item.get("message"))
        .cloned()
        .unwrap_or(json!({}));
    let mut content = Vec::new();
    if let Some(thinking) = message.get("reasoning_content").and_then(Value::as_str) {
        if !thinking.is_empty() {
            content.push(json!({"type": "thinking", "thinking": thinking}));
        }
    }
    if let Some(text) = message.get("content").and_then(Value::as_str) {
        if !text.is_empty() {
            content.push(json!({"type": "text", "text": text}));
        }
    }
    if let Some(calls) = message.get("tool_calls").and_then(Value::as_array) {
        for call in calls {
            let function = call.get("function").cloned().unwrap_or(json!({}));
            content.push(json!({
                "type": "tool_use",
                "id": call.get("id").cloned().unwrap_or(json!("")),
                "name": function.get("name").cloned().unwrap_or(json!("")),
                "input": parse_arguments(function.get("arguments").and_then(Value::as_str)),
            }));
        }
    }
    let finish = choice
        .and_then(|item| item.get("finish_reason"))
        .and_then(Value::as_str)
        .unwrap_or("stop");
    let stop_reason = match finish {
        "length" => "max_tokens",
        "tool_calls" => "tool_use",
        _ => "end_turn",
    };
    let usage = body.get("usage").cloned().unwrap_or(json!({}));
    json!({
        "id": format!("msg_{}", body.get("id").and_then(Value::as_str).unwrap_or("unknown")),
        "type": "message",
        "role": "assistant",
        "model": model,
        "content": content,
        "stop_reason": stop_reason,
        "stop_sequence": Value::Null,
        "usage": {
            "input_tokens": usage.get("prompt_tokens").cloned().unwrap_or(json!(0)),
            "output_tokens": usage.get("completion_tokens").cloned().unwrap_or(json!(0)),
        },
    })
}

fn parse_arguments(text: Option<&str>) -> Value {
    text.and_then(|text| serde_json::from_str::<Value>(text).ok())
        .unwrap_or(json!({}))
}

/// Stream OpenAI chat SSE frames as Anthropic Messages events.
pub fn anthropic_stream(
    mut source: impl futures::Stream<Item = Result<Bytes, Infallible>> + Send + Unpin + 'static,
    model: String,
) -> impl futures::Stream<Item = Bytes> + Send {
    stream! {
        let id = format!("msg_{}", uuid::Uuid::new_v4().simple());
        let mut started = false;
        let mut open_blocks: Vec<String> = Vec::new();
        let mut output_tokens = 0usize;
        let mut pending = Vec::new();
        while let Some(item) = source.next().await {
            let bytes = item.unwrap_or_default();
            pending.extend_from_slice(bytes.as_ref());
            while let Some(end) = pending.windows(2).position(|w| w == b"\n\n") {
                let frame: Vec<u8> = pending.drain(..end + 2).collect();
                let text = String::from_utf8_lossy(&frame);
                let Some(payload) = text.strip_prefix("data: ") else { continue };
                let payload = payload.trim();
                if payload == "[DONE]" {
                    continue;
                }
                let Ok(chunk) = serde_json::from_str::<Value>(payload) else { continue };
                let delta = chunk
                    .get("choices")
                    .and_then(Value::as_array)
                    .and_then(|items| items.first())
                    .and_then(|item| item.get("delta"))
                    .cloned()
                    .unwrap_or(json!({}));
                if !started {
                    started = true;
                    let start = json!({
                        "type": "message_start",
                        "message": {
                            "id": id, "type": "message", "role": "assistant",
                            "model": model, "content": [], "stop_reason": Value::Null,
                            "usage": {"input_tokens": 0, "output_tokens": 0},
                        }
                    });
                    yield Bytes::from(sse_frame(&start));
                }
                if let Some(thinking) = delta.get("reasoning_content").and_then(Value::as_str) {
                    if !thinking.is_empty() {
                        if !open_blocks.iter().any(|kind| kind == "thinking") {
                            open_blocks.push("thinking".into());
                            yield Bytes::from(sse_frame(&json!({
                                "type": "content_block_start",
                                "index": open_blocks.len() - 1,
                                "content_block": {"type": "thinking", "thinking": ""},
                            })));
                        }
                        let index = open_blocks.iter().position(|kind| kind == "thinking").unwrap_or(0);
                        yield Bytes::from(sse_frame(&json!({
                            "type": "content_block_delta", "index": index,
                            "delta": {"type": "thinking_delta", "thinking": thinking},
                        })));
                    }
                }
                if let Some(content) = delta.get("content").and_then(Value::as_str) {
                    if !content.is_empty() {
                        output_tokens += 1;
                        if open_blocks.first().map(String::as_str) != Some("text") {
                            open_blocks.insert(0, "text".into());
                            yield Bytes::from(sse_frame(&json!({
                                "type": "content_block_start", "index": 0,
                                "content_block": {"type": "text", "text": ""},
                            })));
                        }
                        yield Bytes::from(sse_frame(&json!({
                            "type": "content_block_delta", "index": 0,
                            "delta": {"type": "text_delta", "text": content},
                        })));
                    }
                }
                if let Some(calls) = delta.get("tool_calls").and_then(Value::as_array) {
                    for call in calls {
                        let index = call.get("index").and_then(Value::as_u64).unwrap_or(0) as usize;
                        let function = call.get("function").cloned().unwrap_or(json!({}));
                        if let Some(name) = function.get("name").and_then(Value::as_str) {
                            yield Bytes::from(sse_frame(&json!({
                                "type": "content_block_start", "index": index,
                                "content_block": {
                                    "type": "tool_use",
                                    "id": call.get("id").and_then(Value::as_str).unwrap_or(""),
                                    "name": name, "input": {},
                                },
                            })));
                        }
                        if let Some(arguments) = function.get("arguments").and_then(Value::as_str) {
                            yield Bytes::from(sse_frame(&json!({
                                "type": "content_block_delta", "index": index,
                                "delta": {"type": "input_json_delta", "partial_json": arguments},
                            })));
                        }
                    }
                }
            }
        }
        for index in 0..open_blocks.len() {
            yield Bytes::from(sse_frame(&json!({"type": "content_block_stop", "index": index})));
        }
        yield Bytes::from(sse_frame(&json!({
            "type": "message_delta",
            "delta": {"stop_reason": "end_turn", "stop_sequence": Value::Null},
            "usage": {"output_tokens": output_tokens},
        })));
        yield Bytes::from(sse_frame(&json!({"type": "message_stop"})));
    }
}

fn sse_frame(value: &Value) -> Vec<u8> {
    let mut bytes = b"event: ".to_vec();
    bytes.extend_from_slice(
        value
            .get("type")
            .and_then(Value::as_str)
            .unwrap_or("message")
            .as_bytes(),
    );
    bytes.extend_from_slice(b"\ndata: ");
    bytes.extend_from_slice(serde_json::to_string(value).unwrap_or_default().as_bytes());
    bytes.extend_from_slice(b"\n\n");
    bytes
}

pub async fn anthropic_messages(
    State(state): State<SharedState>,
    Json(payload): Json<Value>,
) -> Response {
    let model = state.served_model_name.clone();
    let request = match anthropic_to_chat(&payload, &model) {
        Ok(request) => request,
        Err(message) => return bad_request(message),
    };
    match dispatch_chat(state, request).await {
        Ok(ChatOutcome::Json(body)) => Json(chat_to_anthropic(&body, &model)).into_response(),
        Ok(ChatOutcome::Stream(source)) => {
            Body::from_stream(anthropic_stream(source, model).map(Ok::<_, Infallible>))
                .into_response()
        }
        Err(response) => response,
    }
}

pub async fn anthropic_count_tokens(
    State(state): State<SharedState>,
    Json(payload): Json<Value>,
) -> Response {
    let model = state.served_model_name.clone();
    let request = match anthropic_to_chat(&payload, &model) {
        Ok(request) => request,
        Err(message) => return bad_request(message),
    };
    let messages = request.messages.iter().cloned().collect::<Vec<_>>();
    let tools = request.tools.as_deref();
    let extras = request.template_extras().ok().flatten();
    let rendered = match state
        .tokenizer
        .render_chat_with(&messages, tools, false, extras.as_ref())
    {
        Ok(text) => text,
        Err(err) => return bad_request(err.to_string()),
    };
    let ids = match state.tokenizer_batcher.encode(rendered).await {
        Ok(ids) => ids,
        Err(err) => return bad_request(err),
    };
    Json(json!({"input_tokens": ids.len()})).into_response()
}

fn responses_input_messages(
    input: &Value,
    instructions: Option<&Value>,
    previous: Option<&Value>,
) -> Result<Vec<Value>, String> {
    let mut messages = Vec::new();
    if let Some(text) = instructions.and_then(Value::as_str) {
        if !text.is_empty() {
            messages.push(json!({"role": "system", "content": text}));
        }
    }
    if let Some(previous) = previous {
        if let Some(output) = previous.get("output").and_then(Value::as_array) {
            for item in output {
                if item.get("type").and_then(Value::as_str) != Some("message") {
                    continue;
                }
                let role = item
                    .get("role")
                    .and_then(Value::as_str)
                    .unwrap_or("assistant");
                let mut text = String::new();
                if let Some(parts) = item.get("content").and_then(Value::as_array) {
                    for part in parts {
                        text.push_str(part.get("text").and_then(Value::as_str).unwrap_or(""));
                    }
                }
                messages.push(json!({"role": role, "content": text}));
            }
        }
    }
    match input {
        Value::String(text) => messages.push(json!({"role": "user", "content": text})),
        Value::Array(items) => {
            for item in items {
                let role = item.get("role").and_then(Value::as_str).unwrap_or("user");
                let content = match item.get("content") {
                    Some(Value::String(text)) => text.clone(),
                    Some(Value::Array(parts)) => parts
                        .iter()
                        .filter_map(|part| part.get("text").and_then(Value::as_str))
                        .collect::<Vec<_>>()
                        .join(""),
                    _ => item
                        .get("text")
                        .and_then(Value::as_str)
                        .unwrap_or("")
                        .to_string(),
                };
                messages.push(json!({"role": role, "content": content}));
            }
        }
        _ => return Err("input must be text or an array".into()),
    }
    if messages.is_empty() {
        return Err("input produced no messages".into());
    }
    Ok(messages)
}

fn responses_to_chat(
    payload: &Value,
    previous: Option<&Value>,
    model: &str,
) -> Result<ChatCompletionRequest, String> {
    let messages = responses_input_messages(
        payload.get("input").unwrap_or(&Value::Null),
        payload.get("instructions"),
        previous,
    )?;
    let mut request = json!({
        "model": model,
        "messages": messages,
        "stream": payload.get("stream").and_then(Value::as_bool).unwrap_or(false),
    });
    if let Some(max) = payload.get("max_output_tokens") {
        request["max_tokens"] = max.clone();
    }
    // Only function tools are model-side. Built-in tools require a separate
    // tool server and are rejected by `openai_responses` (same as Python).
    let tools: Vec<Value> = payload
        .get("tools")
        .and_then(Value::as_array)
        .map(|items| {
            items
                .iter()
                .filter(|tool| tool.get("type").and_then(Value::as_str) == Some("function"))
                .map(|tool| {
                    let name = tool.get("name").and_then(Value::as_str).unwrap_or("");
                    let parameters = tool
                        .get("parameters")
                        .or_else(|| tool.get("input_schema"))
                        .cloned()
                        .unwrap_or(json!({"type": "object", "properties": {}}));
                    let mut function = json!({"name": name, "parameters": parameters});
                    if let Some(description) = tool.get("description") {
                        function["description"] = description.clone();
                    }
                    json!({"type": "function", "function": function})
                })
                .collect()
        })
        .unwrap_or_default();
    if !tools.is_empty() {
        request["tools"] = json!(tools);
    }
    serde_json::from_value(request).map_err(|err| err.to_string())
}

fn chat_to_responses(body: &Value, model: &str, stored: bool) -> Value {
    let choice = body
        .get("choices")
        .and_then(Value::as_array)
        .and_then(|items| items.first());
    let message = choice
        .and_then(|item| item.get("message"))
        .cloned()
        .unwrap_or(json!({}));
    let text = message
        .get("content")
        .and_then(Value::as_str)
        .unwrap_or("")
        .to_string();
    let usage = body.get("usage").cloned().unwrap_or(json!({}));
    let id = format!(
        "resp_{}",
        body.get("id").and_then(Value::as_str).unwrap_or("unknown")
    );
    let mut output = Vec::new();
    if let Some(thinking) = message.get("reasoning_content").and_then(Value::as_str) {
        if !thinking.is_empty() {
            output.push(json!({
                "type": "reasoning",
                "summary": [{"type": "summary_text", "text": thinking}],
            }));
        }
    }
    output.push(json!({
        "type": "message",
        "id": format!("msg_{id}"),
        "status": "completed",
        "role": "assistant",
        "content": [{"type": "output_text", "text": text, "annotations": []}],
    }));
    json!({
        "id": id,
        "object": "response",
        "created_at": body.get("created").cloned().unwrap_or(json!(0)),
        "status": "completed",
        "model": model,
        "output": output,
        "store": stored,
        "usage": {
            "input_tokens": usage.get("prompt_tokens").cloned().unwrap_or(json!(0)),
            "output_tokens": usage.get("completion_tokens").cloned().unwrap_or(json!(0)),
            "total_tokens": usage.get("total_tokens").cloned().unwrap_or(json!(0)),
        },
    })
}

pub fn responses_stream(
    mut source: impl futures::Stream<Item = Result<Bytes, Infallible>> + Send + Unpin + 'static,
    model: String,
    store: bool,
) -> (
    impl futures::Stream<Item = Bytes> + Send,
    tokio::sync::oneshot::Receiver<Value>,
) {
    let (tx, rx) = tokio::sync::oneshot::channel();
    let stream = stream! {
        let id = format!("resp_{}", uuid::Uuid::new_v4().simple());
        let mut text = String::new();
        let mut pending = Vec::new();
        let mut started = false;
        let mut item_added = false;
        while let Some(item) = source.next().await {
            let bytes = item.unwrap_or_default();
            pending.extend_from_slice(bytes.as_ref());
            while let Some(end) = pending.windows(2).position(|w| w == b"\n\n") {
                let frame: Vec<u8> = pending.drain(..end + 2).collect();
                let text_frame = String::from_utf8_lossy(&frame);
                let Some(payload) = text_frame.strip_prefix("data: ") else { continue };
                let payload = payload.trim();
                if payload == "[DONE]" {
                    continue;
                }
                let Ok(chunk) = serde_json::from_str::<Value>(payload) else { continue };
                if !started {
                    started = true;
                    yield Bytes::from(sse_frame(&json!({
                        "type": "response.created",
                        "response": {"id": id, "object": "response", "model": model, "status": "in_progress"},
                    })));
                }
                let delta_text = chunk
                    .get("choices")
                    .and_then(Value::as_array)
                    .and_then(|items| items.first())
                    .and_then(|item| item.get("delta"))
                    .and_then(|delta| delta.get("content"))
                    .and_then(Value::as_str)
                    .unwrap_or("");
                if !delta_text.is_empty() {
                    if !item_added {
                        item_added = true;
                        yield Bytes::from(sse_frame(&json!({
                            "type": "response.output_item.added",
                            "output_index": 0,
                            "item": {"type": "message", "role": "assistant", "content": []},
                        })));
                    }
                    text.push_str(delta_text);
                    yield Bytes::from(sse_frame(&json!({
                        "type": "response.output_text.delta",
                        "output_index": 0, "item_id": format!("msg_{id}"),
                        "delta": delta_text,
                    })));
                }
            }
        }
        if item_added {
            yield Bytes::from(sse_frame(&json!({
                "type": "response.output_text.done",
                "output_index": 0, "item_id": format!("msg_{id}"), "text": text,
            })));
        }
        let completed = json!({
            "id": id,
            "object": "response",
            "created_at": 0,
            "status": "completed",
            "model": model,
            "store": store,
            "output": [{
                "type": "message", "id": format!("msg_{id}"), "status": "completed",
                "role": "assistant",
                "content": [{"type": "output_text", "text": text, "annotations": []}],
            }],
            "usage": {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0},
        });
        let _ = tx.send(completed.clone());
        yield Bytes::from(sse_frame(&json!({"type": "response.completed", "response": completed})));
    };
    (stream, rx)
}

fn builtin_tools(payload: &Value) -> Vec<String> {
    payload
        .get("tools")
        .and_then(Value::as_array)
        .map(|items| {
            items
                .iter()
                .filter(|tool| {
                    tool.is_object()
                        && !matches!(
                            tool.get("type").and_then(Value::as_str),
                            None | Some("function")
                        )
                })
                .filter_map(|tool| tool.get("type").and_then(Value::as_str).map(str::to_string))
                .collect()
        })
        .unwrap_or_default()
}

pub async fn openai_responses(
    State(state): State<SharedState>,
    Json(payload): Json<Value>,
) -> Response {
    let model = state.served_model_name.clone();
    // Python's gateway only wires built-in tools when a tool server is
    // configured; the standalone Rust frontend has none, so reject them with
    // the same stable error instead of silently treating them as function tools.
    let builtins = builtin_tools(&payload);
    if !builtins.is_empty() {
        return openai_error(
            StatusCode::BAD_REQUEST,
            format!(
                "built-in tools ({}) require a tool server",
                builtins.join(", ")
            ),
        );
    }
    let store = payload
        .get("store")
        .and_then(Value::as_bool)
        .unwrap_or(true);
    let previous = match payload.get("previous_response_id").and_then(Value::as_str) {
        Some(id) => match state.responses.read().await.get(id).cloned() {
            Some(value) => Some(value),
            None => {
                return openai_error(StatusCode::NOT_FOUND, format!("Response '{id}' not found."))
            }
        },
        None => None,
    };
    let request = match responses_to_chat(&payload, previous.as_ref(), &model) {
        Ok(request) => request,
        Err(message) => return openai_error(StatusCode::BAD_REQUEST, message),
    };
    match dispatch_chat(state.clone(), request).await {
        Ok(ChatOutcome::Json(body)) => {
            let native = chat_to_responses(&body, &model, store);
            if store {
                state.responses.write().await.insert(
                    native
                        .get("id")
                        .and_then(Value::as_str)
                        .unwrap_or("")
                        .to_string(),
                    native.clone(),
                );
            }
            Json(native).into_response()
        }
        Ok(ChatOutcome::Stream(source)) => {
            let (stream, completed) = responses_stream(source, model, store);
            tokio::spawn(async move {
                if let Ok(body) = completed.await {
                    if store {
                        state.responses.write().await.insert(
                            body.get("id")
                                .and_then(Value::as_str)
                                .unwrap_or("")
                                .to_string(),
                            body,
                        );
                    }
                }
            });
            Body::from_stream(stream.map(Ok::<_, Infallible>)).into_response()
        }
        Err(response) => response,
    }
}

pub async fn retrieve_response(
    State(state): State<SharedState>,
    Path(id): Path<String>,
) -> Response {
    match state.responses.read().await.get(&id).cloned() {
        Some(value) => Json(value).into_response(),
        None => openai_error(StatusCode::NOT_FOUND, format!("Response '{id}' not found.")),
    }
}

pub async fn cancel_response(State(state): State<SharedState>, Path(id): Path<String>) -> Response {
    match state.responses.write().await.get_mut(&id) {
        Some(value) => {
            value["status"] = json!("cancelled");
            Json(value.clone()).into_response()
        }
        None => openai_error(StatusCode::NOT_FOUND, format!("Response '{id}' not found.")),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn responses_filters_function_tools_and_detects_builtins() {
        let payload = json!({
            "input": "Hi",
            "tools": [
                {"type": "function", "name": "lookup", "description": "Look up", "parameters": {"type": "object"}},
                {"type": "web_search_preview"},
                {"type": "code_interpreter"}
            ]
        });
        assert_eq!(
            builtin_tools(&payload),
            vec!["web_search_preview", "code_interpreter"]
        );
        let request = responses_to_chat(&payload, None, "served").unwrap();
        let tools = request.tools.unwrap();
        assert_eq!(tools.len(), 1);
        assert_eq!(tools[0]["function"]["name"], "lookup");
        assert_eq!(tools[0]["function"]["description"], "Look up");
        assert_eq!(tools[0]["function"]["parameters"]["type"], "object");
    }

    #[test]
    fn anthropic_request_converts_tools_and_system() {
        let payload = json!({
            "model": "claude",
            "system": "Be brief.",
            "max_tokens": 64,
            "messages": [{"role": "user", "content": [{"type": "text", "text": "Hi"}]}],
            "tools": [{"name": "lookup", "description": "look up", "input_schema": {"type": "object"}}],
            "stop_sequences": ["END"],
        });
        let request = anthropic_to_chat(&payload, "served").unwrap();
        assert_eq!(request.model, "served");
        assert_eq!(request.messages.len(), 2);
        assert_eq!(request.messages[0]["content"], "Be brief.");
        assert_eq!(request.max_tokens, Some(64));
        assert_eq!(request.stop_sequences(), vec!["END"]);
        let tool = &request.tools.unwrap()[0];
        assert_eq!(tool["function"]["name"], "lookup");
        assert_eq!(tool["function"]["parameters"]["type"], "object");
    }

    #[test]
    fn anthropic_response_converts_tool_calls() {
        let body = json!({
            "id": "chatcmpl-1",
            "choices": [{"message": {
                "role": "assistant",
                "content": "",
                "tool_calls": [{"id": "call_1", "function": {
                    "name": "lookup", "arguments": "{\"q\":1}"
                }}],
            }, "finish_reason": "tool_calls"}],
            "usage": {"prompt_tokens": 5, "completion_tokens": 2},
        });
        let converted = chat_to_anthropic(&body, "served");
        assert_eq!(converted["stop_reason"], "tool_use");
        let content = converted["content"].as_array().unwrap();
        assert_eq!(content.len(), 1);
        assert_eq!(content[0]["type"], "tool_use");
        assert_eq!(content[0]["input"]["q"], 1);
        assert_eq!(converted["usage"]["input_tokens"], 5);
    }

    #[test]
    fn responses_input_and_output_round_trip() {
        let payload = json!({
            "model": "gpt",
            "instructions": "Be brief.",
            "input": [{"role": "user", "content": [{"type": "input_text", "text": "Hi"}]}],
            "max_output_tokens": 32,
        });
        let request = responses_to_chat(&payload, None, "served").unwrap();
        assert_eq!(request.messages.len(), 2);
        let body = json!({
            "id": "chatcmpl-2",
            "created": 123,
            "choices": [{"message": {"role": "assistant", "content": "Hello"}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 3, "completion_tokens": 1, "total_tokens": 4},
        });
        let native = chat_to_responses(&body, "served", true);
        assert_eq!(native["object"], "response");
        assert_eq!(native["output"][0]["content"][0]["text"], "Hello");
        assert_eq!(native["usage"]["total_tokens"], 4);

        let followup = json!({"input": "Again"});
        let messages = responses_input_messages(&followup["input"], None, Some(&native)).unwrap();
        assert_eq!(messages.len(), 2);
        assert_eq!(messages[0]["role"], "assistant");
        assert_eq!(messages[1]["content"], "Again");
    }
}
