//! Output parsing and SSE assembly for the Rust frontend.

use std::convert::Infallible;
use std::sync::Arc;
use std::time::Instant;

use async_stream::try_stream;
use axum::body::Bytes;
use megalm_engine_client::OutputEvent;
use megalm_rs::parser::{
    create_tool_parser, InklingOutput, InklingParser, JsonFormatValidator, MarkerKind, OutputParser,
    ParseError, ParserStats, StopMatcher, Tool, ToolCall,
};
use megalm_rs::sse;
use megalm_tokenizer::ModelDecodeStream;
use serde_json::{json, Map, Value};
use tokio::sync::mpsc;

/// Parser abstraction so the streaming loop is identical for text-grammar and
/// Inkling typed-block models.
pub enum UnifiedParser {
    Text(OutputParser),
    Inkling {
        parser: InklingParser,
        stop: Option<StopMatcher>,
        terminal: bool,
    },
}

impl UnifiedParser {
    pub fn push(&mut self, delta: &str) -> (String, String, Vec<ToolCall>, bool) {
        match self {
            Self::Text(parser) => parser.push(delta),
            Self::Inkling {
                parser,
                stop,
                terminal,
            } => {
                if *terminal {
                    return (String::new(), String::new(), Vec::new(), true);
                }
                let mut output = InklingOutput::default();
                parser.push(delta, &mut output);
                let InklingOutput {
                    content,
                    reasoning,
                    calls,
                } = output;
                let (emitted, stopped) = match stop {
                    Some(matcher) => matcher.push(&content),
                    None => (content, false),
                };
                if stopped {
                    *terminal = true;
                }
                (emitted, reasoning, calls, stopped)
            }
        }
    }

    /// Per-request parser counters, folded into the process replica metrics
    /// once the stream ends (mirrors Python's `parser.stats()` tuple).
    pub fn stats(&self) -> &ParserStats {
        match self {
            Self::Text(parser) => parser.stats(),
            Self::Inkling { parser, .. } => parser.stats(),
        }
    }

    /// Structured diagnostics accumulated by the underlying parser.
    pub fn take_diagnostics(&mut self) -> Vec<ParseError> {
        match self {
            Self::Text(parser) => parser.take_diagnostics(),
            Self::Inkling { parser, .. } => parser.take_diagnostics(),
        }
    }

    pub fn finish(&mut self) -> (String, String, Vec<ToolCall>) {
        match self {
            Self::Text(parser) => parser.finish(),
            Self::Inkling {
                parser,
                stop,
                terminal,
            } => {
                if *terminal {
                    return (String::new(), String::new(), Vec::new());
                }
                let mut output = InklingOutput::default();
                parser.finish(&mut output);
                let InklingOutput {
                    content,
                    reasoning,
                    calls,
                } = output;
                let emitted = match stop {
                    Some(matcher) => {
                        let (head, _) = matcher.push(&content);
                        let mut combined = head;
                        combined.push_str(&matcher.finish());
                        combined
                    }
                    None => content,
                };
                (emitted, reasoning, calls)
            }
        }
    }
}

/// Fold one request's parser counters into replica metrics and emit a
/// structured `parser_degraded` event for every block the parser had to
/// degrade to visible text. This is the Rust counterpart of the Python
/// gateway's `_record_parser_telemetry`.
fn record_parser_telemetry(
    metrics: &crate::replica_metrics::ReplicaMetrics,
    parser: &mut UnifiedParser,
    request_id: &str,
    streaming: bool,
) {
    let diagnostics = parser.take_diagnostics();
    metrics.record_parser(parser.stats());
    if !diagnostics.is_empty() {
        crate::replica_metrics::emit_parser_degraded(request_id, streaming, &diagnostics);
    }
}

/// Best-effort `response_format` check. Mirrors the Python gateway's
/// `check_format`: validation is observational only, never changes the
/// response, and failures become a single structured warning log.
fn check_format(validator: Option<&JsonFormatValidator>, request_id: &str, text: &str) {
    let Some(validator) = validator else {
        return;
    };
    if text.trim().is_empty() {
        return;
    }
    if let Err(error) = validator.validate(text) {
        let mut fields = Map::new();
        fields.insert("request_id".into(), json!(request_id));
        fields.insert("error".into(), json!(error));
        crate::replica_metrics::emit_event(
            "megalm.gateway.chat",
            "warning",
            "response_format_invalid",
            fields,
        );
    }
}

pub struct StreamContext {
    pub request_id: String,
    pub model: String,
    pub created: i64,
    pub index: usize,
    pub prompt_tokens: usize,
    pub include_usage: bool,
    pub parser: UnifiedParser,
    /// Best-effort `response_format` validator (opt-in per request).
    pub format_validator: Option<JsonFormatValidator>,
    pub profiler: Arc<crate::profiling::Profiler>,
    /// Process-level metrics producer for `megalm_replica_*`.
    pub metrics: Arc<crate::replica_metrics::ReplicaMetrics>,
    /// Admission instant used for TTFT / TPOT / E2E latency observations.
    pub started: Instant,
    /// Optional placement-lease tracker (opt-in control plane).
    pub lease_tracker: Option<crate::control_plane::LeaseTracker>,
}

/// Build the SSE byte stream for one chat completion choice.
pub fn chat_stream(
    mut context: StreamContext,
    mut decoder: ModelDecodeStream<'static>,
    mut events: mpsc::UnboundedReceiver<OutputEvent>,
) -> impl futures::Stream<Item = Result<Bytes, Infallible>> + Send {
    try_stream! {
        let id = context.request_id.clone();
        let model = context.model.clone();
        let created = context.created;
        let index = context.index;
        // The frontend has handed the request to the engine; prefill is now
        // underway. The GPU worker may record a more precise prefill_start in
        // its own JSONL file - view_files() merges both by request_id.
        context
            .profiler
            .record_prefill_start(&id, context.prompt_tokens);
        yield Bytes::from(sse::role_chunk(&id, &model, created, index));

        let mut completion_tokens = 0usize;
        let mut format_text = String::new();
        let mut terminal = false;
        let mut first_token_recorded = false;
        let mut ttft_ms: Option<f64> = None;
        let mut finish_reason = "stop".to_owned();
        while let Some(event) = events.recv().await {
            completion_tokens += event.token_ids.len();
            let delta = decoder.push(&event.token_ids);
            if !first_token_recorded && !delta.is_empty() {
                first_token_recorded = true;
                let elapsed = context.started.elapsed().as_secs_f64() * 1_000.0;
                context.metrics.record_ttft_ms(elapsed);
                ttft_ms = Some(elapsed);
                context.profiler.record_first_token(&id, index);
                if let Some(tracker) = &context.lease_tracker {
                    tracker.first_token(&id);
                }
            }
            let (text, reasoning, calls, stopped) = context.parser.push(&delta);
            if !reasoning.is_empty() {
                yield Bytes::from(sse::reasoning_chunk(&id, &model, created, index, &reasoning));
            }
            for (call_index, call) in calls.iter().enumerate() {
                yield Bytes::from(sse::tool_call_chunk(
                    &id,
                    &model,
                    created,
                    index,
                    &call.id,
                    call_index,
                    &call.name,
                    &call.arguments,
                ));
            }
            if !text.is_empty() {
                format_text.push_str(&text);
                yield Bytes::from(sse::content_chunk(&id, &model, created, index, &text));
            }
            if stopped || event.finish_reason.is_some() || event.error.is_some() {
                if let Some(reason) = event.finish_reason.as_deref() {
                    finish_reason = map_finish_reason(reason);
                }
                if event.error.is_some() {
                    finish_reason = "error".into();
                }
                terminal = true;
                break;
            }
        }
        if !terminal {
            let (text, reasoning, calls) = context.parser.finish();
            if !reasoning.is_empty() {
                yield Bytes::from(sse::reasoning_chunk(&id, &model, created, index, &reasoning));
            }
            for (call_index, call) in calls.iter().enumerate() {
                yield Bytes::from(sse::tool_call_chunk(
                    &id,
                    &model,
                    created,
                    index,
                    &call.id,
                    call_index,
                    &call.name,
                    &call.arguments,
                ));
            }
            if !text.is_empty() {
                format_text.push_str(&text);
                yield Bytes::from(sse::content_chunk(&id, &model, created, index, &text));
            }
        }
        let usage = context
            .include_usage
            .then_some((context.prompt_tokens as i64, completion_tokens as i64));
        yield Bytes::from(sse::finish_chunk(
            &id,
            &model,
            created,
            index,
            "stop",
            usage,
        ));
        yield Bytes::from_static(b"data: [DONE]\n\n");
        let elapsed = context.started.elapsed().as_secs_f64() * 1_000.0;
        if let (Some(ttft), true) = (ttft_ms, completion_tokens > 1) {
            context
                .metrics
                .record_tpot_ms((elapsed - ttft) / (completion_tokens - 1) as f64);
        }
        context.metrics.record_e2e_ms(elapsed);
        context.metrics.record_finish_reason(&finish_reason);
        record_parser_telemetry(&context.metrics, &mut context.parser, &id, true);
        check_format(context.format_validator.as_ref(), &id, &format_text);
        context
            .profiler
            .record_terminal(&id, index, completion_tokens, "stop");
    }
}

/// SSE byte stream for raw text completions.
pub fn completion_stream(
    id: String,
    model: String,
    created: i64,
    mut events: mpsc::UnboundedReceiver<OutputEvent>,
    mut decoder: ModelDecodeStream<'static>,
    lease_tracker: Option<crate::control_plane::LeaseTracker>,
    usage_prompt_tokens: Option<usize>,
    metrics: Arc<crate::replica_metrics::ReplicaMetrics>,
    started: Instant,
) -> impl futures::Stream<Item = Result<Bytes, Infallible>> + Send {
    try_stream! {
        let mut first_token_recorded = false;
        let mut completion_tokens = 0;
        let mut finish_reason = "stop".to_owned();
        let mut ttft_ms: Option<f64> = None;
        while let Some(event) = events.recv().await {
            completion_tokens += event.token_ids.len();
            let text = decoder.push(&event.token_ids);
            if !text.is_empty() {
                if !first_token_recorded {
                    first_token_recorded = true;
                    let elapsed = started.elapsed().as_secs_f64() * 1_000.0;
                    metrics.record_ttft_ms(elapsed);
                    ttft_ms = Some(elapsed);
                    if let Some(tracker) = &lease_tracker {
                        tracker.first_token(&id);
                    }
                }
                let frame = json!({
                    "id": format!("cmpl-{id}"),
                    "object": "text_completion",
                    "created": created,
                    "model": model,
                    "choices": [{"index": 0, "text": text, "finish_reason": Value::Null}],
                });
                yield Bytes::from(sse::raw_json_frame(&serde_json::to_string(&frame).unwrap_or_default()));
            }
            if let Some(error) = event.error {
                let frame = json!({"error": error});
                yield Bytes::from(sse::raw_json_frame(&frame.to_string()));
                return;
            }
            if let Some(reason) = event.finish_reason {
                finish_reason = map_finish_reason(&reason);
                break;
            }
        }
        let frame = json!({
            "id": format!("cmpl-{id}"),
            "object": "text_completion",
            "created": created,
            "model": model,
            "choices": [{"index": 0, "text": "", "finish_reason": finish_reason}],
        });
        yield Bytes::from(sse::raw_json_frame(&serde_json::to_string(&frame).unwrap_or_default()));
        if let Some(prompt_tokens) = usage_prompt_tokens {
            let frame = json!({
                "id": format!("cmpl-{id}"),
                "object": "text_completion",
                "created": created,
                "model": model,
                "choices": [],
                "usage": {
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                    "total_tokens": prompt_tokens + completion_tokens,
                },
            });
            yield Bytes::from(sse::raw_json_frame(&frame.to_string()));
        }
        yield Bytes::from_static(b"data: [DONE]\n\n");
        let elapsed = started.elapsed().as_secs_f64() * 1_000.0;
        if let (Some(ttft), true) = (ttft_ms, completion_tokens > 1) {
            metrics.record_tpot_ms((elapsed - ttft) / (completion_tokens - 1) as f64);
        }
        metrics.record_e2e_ms(elapsed);
        metrics.record_finish_reason(&finish_reason);
    }
}

/// Collect one choice into an OpenAI-shaped chat-completion object.
pub async fn collect_chat(
    mut context: StreamContext,
    mut decoder: ModelDecodeStream<'static>,
    mut events: mpsc::UnboundedReceiver<OutputEvent>,
) -> Value {
    let mut text = String::new();
    let mut reasoning = String::new();
    let mut calls = Vec::new();
    let mut completion_tokens = 0usize;
    let mut finish_reason = "stop".to_string();
    let mut first_token_recorded = false;
    let mut ttft_ms: Option<f64> = None;
    context
        .profiler
        .record_prefill_start(&context.request_id, context.prompt_tokens);
    while let Some(event) = events.recv().await {
        completion_tokens += event.token_ids.len();
        let delta = decoder.push(&event.token_ids);
        if !first_token_recorded && !delta.is_empty() {
            first_token_recorded = true;
            let elapsed = context.started.elapsed().as_secs_f64() * 1_000.0;
            context.metrics.record_ttft_ms(elapsed);
            ttft_ms = Some(elapsed);
            context
                .profiler
                .record_first_token(&context.request_id, context.index);
            if let Some(tracker) = &context.lease_tracker {
                tracker.first_token(&context.request_id);
            }
        }
        let (part_text, part_reasoning, part_calls, _stopped) = context.parser.push(&delta);
        text.push_str(&part_text);
        reasoning.push_str(&part_reasoning);
        calls.extend(part_calls);
        if let Some(reason) = event.finish_reason {
            finish_reason = map_finish_reason(&reason);
            break;
        }
        if event.error.is_some() {
            finish_reason = "error".into();
            break;
        }
    }
    let (tail_text, tail_reasoning, tail_calls) = context.parser.finish();
    text.push_str(&tail_text);
    reasoning.push_str(&tail_reasoning);
    calls.extend(tail_calls);

    let mut message = json!({"role": "assistant", "content": text.clone()});
    if !reasoning.is_empty() {
        message["reasoning_content"] = json!(reasoning);
    }
    if !calls.is_empty() {
        let tool_calls = calls
            .iter()
            .enumerate()
            .map(|(index, call)| {
                json!({
                    "index": index,
                    "id": call.id,
                    "type": "function",
                    "function": {"name": call.name, "arguments": call.arguments},
                })
            })
            .collect::<Vec<_>>();
        message["tool_calls"] = json!(tool_calls);
    }
    let response = json!({
        "id": format!("chatcmpl-{}", context.request_id),
        "object": "chat.completion",
        "created": context.created,
        "model": context.model,
        "choices": [{"index": context.index, "message": message, "finish_reason": finish_reason}],
        "usage": {
            "prompt_tokens": context.prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": context.prompt_tokens + completion_tokens,
        },
    });
    let elapsed = context.started.elapsed().as_secs_f64() * 1_000.0;
    if let (Some(ttft), true) = (ttft_ms, completion_tokens > 1) {
        context
            .metrics
            .record_tpot_ms((elapsed - ttft) / (completion_tokens - 1) as f64);
    }
    context.metrics.record_e2e_ms(elapsed);
    context.metrics.record_finish_reason(&finish_reason);
    record_parser_telemetry(&context.metrics, &mut context.parser, &context.request_id, false);
    check_format(context.format_validator.as_ref(), &context.request_id, &text);
    context.profiler.record_terminal(
        &context.request_id,
        context.index,
        completion_tokens,
        &finish_reason,
    );
    response
}

/// Merge per-choice responses into a single non-streaming response.
pub fn merge_collected(responses: Vec<Value>) -> Value {
    let Some(first) = responses.first() else {
        return json!({"choices": []});
    };
    let mut merged = first.clone();
    let mut choices = Vec::new();
    let mut prompt_tokens = 0u64;
    let mut completion_tokens = 0u64;
    for response in &responses {
        if let Some(choice) = response
            .get("choices")
            .and_then(Value::as_array)
            .and_then(|items| items.first())
        {
            choices.push(choice.clone());
        }
        if let Some(usage) = response.get("usage") {
            prompt_tokens += usage.get("prompt_tokens").and_then(Value::as_u64).unwrap_or(0);
            completion_tokens += usage
                .get("completion_tokens")
                .and_then(Value::as_u64)
                .unwrap_or(0);
        }
    }
    merged["choices"] = json!(choices);
    merged["usage"] = json!({
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": prompt_tokens + completion_tokens,
    });
    merged
}

pub(crate) fn map_finish_reason(reason: &str) -> String {
    match reason {
        "stop" | "cancelled" => "stop",
        "length" => "length",
        "rejected" | "error" => "error",
        other => other,
    }
    .to_string()
}

/// Construct the output parser for one request, honoring the model's grammar.
#[allow(clippy::too_many_arguments)]
pub fn make_parser(
    state: &crate::state::AppState,
    request_id: &str,
    stop_sequences: Vec<String>,
    tools: &[Value],
    forced_tool: Option<&str>,
    initial_in_thinking: bool,
    prompt_token_ids: &[u32],
    marker_kinds: &[(u32, MarkerKind)],
) -> Result<UnifiedParser, String> {
    if state.grammar == "inkling" && state.preserve_special_tokens {
        let mut parser = InklingParser::new(request_id);
        parser.initialize(prompt_token_ids, marker_kinds);
        let stop = if stop_sequences.is_empty() {
            None
        } else {
            Some(StopMatcher::new(stop_sequences))
        };
        return Ok(UnifiedParser::Inkling {
            parser,
            stop,
            terminal: false,
        });
    }

    let tool_names = tools
        .iter()
        .filter_map(|tool| {
            tool.get("function")
                .and_then(|function| function.get("name"))
                .and_then(Value::as_str)
                .map(str::to_string)
        })
        .filter(|name| forced_tool.map_or(true, |forced| forced == name))
        .collect::<Vec<_>>();
    if tool_names.is_empty() {
        return Ok(UnifiedParser::Text(OutputParser::new(
            request_id,
            stop_sequences,
            Vec::new(),
            Some("<think>".into()),
            Some("</think>".into()),
            initial_in_thinking,
        )));
    }
    let tool_specs = tools
        .iter()
        .filter_map(|tool| serde_json::from_value::<Tool>(tool.clone()).ok())
        .collect::<Vec<_>>();
    let parser = create_tool_parser(&state.tool_parser, &tool_specs)
        .map_err(|err| format!("cannot create tool parser: {err}"))?;
    Ok(UnifiedParser::Text(OutputParser::with_tool_parser(
        request_id,
        stop_sequences,
        parser,
        Some("<think>".into()),
        Some("</think>".into()),
        initial_in_thinking,
    )))
}

/// Owned decode stream: the tokenizer is leaked once for the process lifetime.
pub fn owned_decoder(
    tokenizer: &'static megalm_tokenizer::ModelTokenizer,
    skip_special_tokens: bool,
) -> ModelDecodeStream<'static> {
    tokenizer.decode_stream(skip_special_tokens)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn merge_collected_sums_usage_and_keeps_choices() {
        let a = json!({
            "choices": [{"index": 0, "message": {"role": "assistant", "content": "a"}}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 2, "total_tokens": 12},
        });
        let b = json!({
            "choices": [{"index": 1, "message": {"role": "assistant", "content": "b"}}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 3, "total_tokens": 13},
        });
        let merged = merge_collected(vec![a, b]);
        assert_eq!(merged["choices"].as_array().unwrap().len(), 2);
        assert_eq!(merged["usage"]["prompt_tokens"], 20);
        assert_eq!(merged["usage"]["completion_tokens"], 5);
        assert_eq!(merged["usage"]["total_tokens"], 25);
    }

    #[test]
    fn text_parser_splits_reasoning_and_honors_stop() {
        let mut parser = UnifiedParser::Text(OutputParser::new(
            "test",
            vec!["STOP".into()],
            Vec::new(),
            Some("<think>".into()),
            Some("</think>".into()),
            false,
        ));
        let (text, reasoning, calls, stopped) = parser.push("<think>why</think>hi STOP tail");
        assert_eq!(reasoning, "why");
        assert_eq!(text, "hi ");
        assert!(calls.is_empty());
        assert!(stopped);
    }
}
