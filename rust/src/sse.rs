//! OpenAI-compatible SSE chat-chunk assembly.

use pyo3::prelude::*;
use pyo3::types::PyBytes;
use serde_json::{json, Map, Value};

fn choice(index: usize, delta: Value, finish_reason: Option<&str>) -> Value {
    json!({
        "index": index,
        "delta": delta,
        "logprobs": null,
        "finish_reason": finish_reason,
    })
}

fn choice_with_logprobs(
    index: usize,
    delta: Value,
    finish_reason: Option<&str>,
    logprobs: Value,
) -> Value {
    json!({
        "index": index,
        "delta": delta,
        "logprobs": logprobs,
        "finish_reason": finish_reason,
    })
}

fn sse_frame(payload: Value) -> Vec<u8> {
    let mut bytes = b"data: ".to_vec();
    bytes.extend_from_slice(&serde_json::to_vec(&payload).expect("json serializes"));
    bytes.extend_from_slice(b"\n\n");
    bytes
}

fn usage_value(prompt_tokens: i64, completion_tokens: i64) -> Value {
    json!({
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": prompt_tokens + completion_tokens,
    })
}

/// First chunk of a chat stream announcing the assistant role.
pub fn role_chunk(id: &str, model: &str, created: i64, index: usize) -> Vec<u8> {
    sse_frame(json!({
        "id": format!("chatcmpl-{id}"),
        "object": "chat.completion.chunk",
        "created": created,
        "model": model,
        "choices": [choice(index, json!({"role": "assistant"}), None)],
    }))
}

/// Per-token content chunk; the hottest frame on the streaming path.
pub fn content_chunk(
    id: &str,
    model: &str,
    created: i64,
    index: usize,
    content: &str,
) -> Vec<u8> {
    sse_frame(json!({
        "id": format!("chatcmpl-{id}"),
        "object": "chat.completion.chunk",
        "created": created,
        "model": model,
        "choices": [choice(index, json!({"content": content}), None)],
    }))
}

/// Content chunk carrying per-token logprobs next to the text delta.
///
/// `entries_json` is a pre-serialized OpenAI `logprobs.content` array; it is
/// embedded verbatim so the Python side avoids a second Rust round trip.
pub fn content_chunk_with_logprobs(
    id: &str,
    model: &str,
    created: i64,
    index: usize,
    content: &str,
    entries_json: &str,
) -> Vec<u8> {
    let entries: Value = serde_json::from_str(entries_json)
        .unwrap_or_else(|_| json!([]));
    sse_frame(json!({
        "id": format!("chatcmpl-{id}"),
        "object": "chat.completion.chunk",
        "created": created,
        "model": model,
        "choices": [choice_with_logprobs(
            index,
            json!({"content": content}),
            None,
            json!({"content": entries}),
        )],
    }))
}

/// Terminal content chunk carrying the finish reason and optional usage.
pub fn finish_chunk(
    id: &str,
    model: &str,
    created: i64,
    index: usize,
    finish_reason: &str,
    usage: Option<(i64, i64)>,
) -> Vec<u8> {
    let mut payload = json!({
        "id": format!("chatcmpl-{id}"),
        "object": "chat.completion.chunk",
        "created": created,
        "model": model,
        "choices": [choice(index, json!({}), Some(finish_reason))],
    });
    if let Some((prompt_tokens, completion_tokens)) = usage {
        payload["usage"] = usage_value(prompt_tokens, completion_tokens);
    }
    sse_frame(payload)
}

/// OpenAI `include_usage` trailer: no choices, usage only.
pub fn usage_chunk(
    id: &str,
    model: &str,
    created: i64,
    prompt_tokens: i64,
    completion_tokens: i64,
) -> Vec<u8> {
    sse_frame(json!({
        "id": format!("chatcmpl-{id}"),
        "object": "chat.completion.chunk",
        "created": created,
        "model": model,
        "choices": [],
        "usage": usage_value(prompt_tokens, completion_tokens),
    }))
}

/// Audio output chunk carrying one base64-encoded waveform segment.
pub fn audio_chunk(
    id: &str,
    model: &str,
    created: i64,
    index: usize,
    audio_data_b64: &str,
    format: &str,
) -> Vec<u8> {
    sse_frame(json!({
        "id": format!("chatcmpl-{id}"),
        "object": "chat.completion.chunk",
        "created": created,
        "model": model,
        "choices": [choice(
            index,
            json!({"audio": {"data": audio_data_b64, "format": format}}),
            None,
        )],
    }))
}

/// DeepSeek-style reasoning delta: `delta.reasoning_content`.
pub fn reasoning_chunk(id: &str, model: &str, created: i64, index: usize, reasoning: &str) -> Vec<u8> {
    sse_frame(json!({
        "id": format!("chatcmpl-{id}"),
        "object": "chat.completion.chunk",
        "created": created,
        "model": model,
        "choices": [choice(index, json!({"reasoning_content": reasoning}), None)],
    }))
}

/// OpenAI tool-call delta chunk carrying a full function invocation.
///
/// Streaming clients accumulate `function.arguments` across chunks; emitting
/// the complete arguments in a single delta is valid and keeps the gateway
/// parser stateless per frame.
pub fn tool_call_chunk(
    id: &str,
    model: &str,
    created: i64,
    choice_index: usize,
    call_id: &str,
    index: usize,
    name: &str,
    arguments: &str,
) -> Vec<u8> {
    sse_frame(json!({
        "id": format!("chatcmpl-{id}"),
        "object": "chat.completion.chunk",
        "created": created,
        "model": model,
        "choices": [choice(choice_index, json!({
            "tool_calls": [{
                "index": index,
                "id": call_id,
                "type": "function",
                "function": {"name": name, "arguments": arguments},
            }],
        }), None)],
    }))
}

/// SSE frame for one incremental tool-call argument fragment.
///
/// The first fragment carries the call id and function name; later fragments
/// carry only the index and argument text, matching OpenAI's streaming shape.
pub fn tool_call_delta_chunk(
    id: &str,
    model: &str,
    created: i64,
    choice_index: usize,
    index: usize,
    call_id: Option<&str>,
    name: Option<&str>,
    arguments: &str,
) -> Vec<u8> {
    let mut function = Map::new();
    if let Some(name) = name {
        function.insert("name".to_string(), Value::String(name.to_string()));
    }
    function.insert(
        "arguments".to_string(),
        Value::String(arguments.to_string()),
    );
    let mut call = Map::new();
    call.insert("index".to_string(), json!(index));
    if let Some(call_id) = call_id {
        call.insert("id".to_string(), Value::String(call_id.to_string()));
        call.insert("type".to_string(), Value::String("function".to_string()));
    }
    call.insert("function".to_string(), Value::Object(function));
    sse_frame(json!({
        "id": format!("chatcmpl-{id}"),
        "object": "chat.completion.chunk",
        "created": created,
        "model": model,
        "choices": [choice(choice_index, json!({ "tool_calls": [call] }), None)],
    }))
}

/// Error frame; the payload is a prebuilt JSON object string.
pub fn raw_json_frame(payload_json: &str) -> Vec<u8> {
    let mut bytes = b"data: ".to_vec();
    bytes.extend_from_slice(payload_json.as_bytes());
    bytes.extend_from_slice(b"\n\n");
    bytes
}

#[pyfunction]
#[pyo3(signature = (id, model, created, index=0))]
pub(crate) fn sse_role_chunk(py: Python<'_>, id: &str, model: &str, created: i64, index: usize) -> Py<PyBytes> {
    PyBytes::new_bound(py, &role_chunk(id, model, created, index)).into()
}

#[pyfunction]
#[pyo3(signature = (id, model, created, content, index=0))]
pub(crate) fn sse_content_chunk(
    py: Python<'_>,
    id: &str,
    model: &str,
    created: i64,
    content: &str,
    index: usize,
) -> Py<PyBytes> {
    PyBytes::new_bound(py, &content_chunk(id, model, created, index, content)).into()
}

#[pyfunction]
pub(crate) fn sse_content_chunk_with_logprobs(
    py: Python<'_>,
    id: &str,
    model: &str,
    created: i64,
    content: &str,
    index: usize,
    entries_json: &str,
) -> Py<PyBytes> {
    PyBytes::new_bound(
        py,
        &content_chunk_with_logprobs(
            id,
            model,
            created,
            index,
            content,
            entries_json,
        ),
    )
    .into()
}

#[pyfunction]
#[pyo3(signature = (id, model, created, index, finish_reason, usage=None))]
pub(crate) fn sse_finish_chunk(
    py: Python<'_>,
    id: &str,
    model: &str,
    created: i64,
    index: usize,
    finish_reason: &str,
    usage: Option<(i64, i64)>,
) -> Py<PyBytes> {
    PyBytes::new_bound(
        py,
        &finish_chunk(id, model, created, index, finish_reason, usage),
    )
    .into()
}

#[pyfunction]
pub(crate) fn sse_usage_chunk(
    py: Python<'_>,
    id: &str,
    model: &str,
    created: i64,
    prompt_tokens: i64,
    completion_tokens: i64,
) -> Py<PyBytes> {
    PyBytes::new_bound(
        py,
        &usage_chunk(id, model, created, prompt_tokens, completion_tokens),
    )
    .into()
}

#[pyfunction]
#[pyo3(signature = (id, model, created, audio_data_b64, format, index=0))]
pub(crate) fn sse_audio_chunk(
    py: Python<'_>,
    id: &str,
    model: &str,
    created: i64,
    audio_data_b64: &str,
    format: &str,
    index: usize,
) -> Py<PyBytes> {
    PyBytes::new_bound(
        py,
        &audio_chunk(id, model, created, index, audio_data_b64, format),
    )
    .into()
}

#[pyfunction]
pub(crate) fn sse_raw_json(py: Python<'_>, payload_json: &str) -> Py<PyBytes> {
    PyBytes::new_bound(py, &raw_json_frame(payload_json)).into()
}

#[cfg(test)]
mod sse_tests {
    use super::{content_chunk, finish_chunk, role_chunk, usage_chunk};

    #[test]
    fn role_chunk_announces_assistant() {
        let frame = String::from_utf8(role_chunk("r1", "m", 1, 0)).unwrap();
        assert!(frame.starts_with("data: "));
        assert!(frame.ends_with("\n\n"));
        assert!(frame.contains("\"role\":\"assistant\""));
        assert!(frame.contains("\"id\":\"chatcmpl-r1\""));
        assert!(frame.contains("\"finish_reason\":null"));
    }

    #[test]
    fn content_chunk_escapes_and_carries_delta() {
        let frame = String::from_utf8(content_chunk("r1", "m", 1, 0, "a\"b\n你好")).unwrap();
        assert!(frame.contains("\"content\":\"a\\\"b\\n你好\""));
    }

    #[test]
    fn finish_chunk_attaches_usage_only_when_present() {
        let frame = String::from_utf8(finish_chunk("r1", "m", 1, 0, "stop", None)).unwrap();
        assert!(frame.contains("\"finish_reason\":\"stop\""));
        assert!(!frame.contains("usage"));

        let frame = String::from_utf8(finish_chunk("r1", "m", 1, 0, "length", Some((3, 5))))
            .unwrap();
        assert!(frame.contains("\"total_tokens\":8"));
    }

    #[test]
    fn usage_chunk_has_empty_choices() {
        let frame = String::from_utf8(usage_chunk("r1", "m", 1, 3, 5)).unwrap();
        assert!(frame.contains("\"choices\":[]"));
        assert!(frame.contains("\"prompt_tokens\":3"));
    }
}

