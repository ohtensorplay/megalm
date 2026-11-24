//! Structured-decision endpoints (`/v1/systemone`, `/v1/decisions`).
//!
//! Port of `decision_api.py`. Each question is rendered through the chat
//! template, its answer labels are verified to be single tokens at the answer
//! position, and the worker returns their log-probabilities via
//! `SamplingParams.score_labels` on the normal output stream.

use std::collections::{HashMap, HashSet};

use axum::extract::State;
use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use axum::Json;
use futures::future::try_join_all;
use megalm_protocol::{Submission, SubmitBatch};
use serde::Deserialize;
use serde_json::{json, Map, Value};

use crate::app::SharedState;

const PROMPT_FORMAT_VERSION: i64 = 1;
const MAX_QUESTIONS: usize = 64;
const MAX_CHOICES: usize = 26;
const MAX_LEVELS: usize = 10;
const CHOICE_LABELS: &[&str] = &[
    "A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O", "P", "Q",
    "R", "S", "T", "U", "V", "W", "X", "Y", "Z",
];
const LEVEL_LABELS: &[&str] = &["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"];
const NOUL_LABELS: &[&str] = &["yes", "no"];

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct SystemOneQuestion {
    #[serde(rename = "type")]
    kind: String,
    instructions: Option<String>,
    criteria: Option<Value>,
}

#[derive(Debug, Deserialize)]
struct SystemOneRequest {
    questions: HashMap<String, SystemOneQuestion>,
    state: Option<Value>,
    chat_template_kwargs: Option<Value>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct DecisionItem {
    id: String,
    #[serde(rename = "type")]
    kind: String,
    question: String,
    options: Option<Vec<Value>>,
    choices: Option<Vec<Value>>,
    levels: Option<Vec<Value>>,
    yes: Option<String>,
    no: Option<String>,
}

fn default_temperature() -> f64 {
    1.0
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct DecisionsRequest {
    input: Option<Value>,
    state: Option<Value>,
    questions: Vec<DecisionItem>,
    #[serde(default = "default_temperature")]
    temperature: f64,
    chat_template_kwargs: Option<Value>,
    prompt_format_version: Option<i64>,
    images: Option<Value>,
}

fn bad_request(message: impl Into<String>) -> Response {
    (
        StatusCode::BAD_REQUEST,
        Json(json!({"detail": message.into()})),
    )
        .into_response()
}

fn unavailable(message: impl Into<String>) -> Response {
    (
        StatusCode::SERVICE_UNAVAILABLE,
        Json(json!({"detail": message.into()})),
    )
        .into_response()
}

/// Python `_text`: strings pass through; other values use canonical JSON.
fn text(value: &Value) -> String {
    if let Some(text) = value.as_str() {
        return text.to_string();
    }
    serde_json::to_string(&canonical(value)).unwrap_or_default()
}

/// Python `str()` for the value types the decision contract permits.
fn py_str(value: &Value) -> String {
    match value {
        Value::String(text) => text.clone(),
        Value::Bool(true) => "True".into(),
        Value::Bool(false) => "False".into(),
        Value::Null => "None".into(),
        other => text(other),
    }
}

fn canonical(value: &Value) -> Value {
    match value {
        Value::Object(map) => {
            let mut keys: Vec<&String> = map.keys().collect();
            keys.sort();
            let mut sorted = Map::new();
            for key in keys {
                sorted.insert(key.clone(), canonical(&map[key]));
            }
            Value::Object(sorted)
        }
        Value::Array(items) => Value::Array(items.iter().map(canonical).collect()),
        other => other.clone(),
    }
}

fn check_template_kwargs(kwargs: Option<&Value>) -> Result<(), String> {
    let Some(kwargs) = kwargs else {
        return Ok(());
    };
    let Some(map) = kwargs.as_object() else {
        return Ok(());
    };
    if map.contains_key("chat_template") {
        return Err("request chat templates are not accepted".into());
    }
    if let Some(value) = map.get("enable_thinking") {
        if !value.is_null() && value.as_bool() != Some(false) {
            return Err("decision reads require thinking disabled".into());
        }
    }
    Ok(())
}

fn check_name(name: &str, what: &str) -> Result<String, String> {
    let trimmed = name.trim();
    if trimmed.is_empty() {
        return Err(format!("{what} must be a non-empty string"));
    }
    if trimmed
        .chars()
        .any(|character| character.is_control() || character == '\r' || character == '\n' || character == '\t')
    {
        return Err(format!("{what} must not contain control or line-break characters"));
    }
    Ok(trimmed.to_string())
}

fn render_labels(state: &str, instructions: Option<&str>, lines: &[String], closing: &str) -> String {
    let mut parts = vec![state.to_string(), String::new()];
    if let Some(instructions) = instructions {
        if !instructions.is_empty() {
            parts.push(format!("Question: {instructions}"));
        }
    }
    parts.extend(lines.iter().cloned());
    parts.push(closing.to_string());
    parts.join("\n")
}

fn system_one_prompt(state: &str, question: &SystemOneQuestion) -> Result<(String, Vec<String>), String> {
    let instructions = question.instructions.as_deref();
    match question.kind.as_str() {
        "choice" => {
            let Some(criteria) = question.criteria.as_ref().and_then(Value::as_object) else {
                return Err("a choice question needs at least one named option".into());
            };
            if criteria.is_empty() {
                return Err("a choice question needs at least one named option".into());
            }
            if criteria.len() > MAX_CHOICES {
                return Err(format!("a choice question allows at most {MAX_CHOICES} options"));
            }
            let mut lines = Vec::new();
            let mut names = Vec::new();
            for (name, description) in criteria {
                let name = check_name(name, "choice option name")?;
                let mut line = format!("{}: {name}", CHOICE_LABELS[names.len()]);
                if !description.is_null() {
                    let description = text(description);
                    if !description.is_empty() {
                        line.push_str(&format!(" - {description}"));
                    }
                }
                lines.push(line);
                names.push(name);
            }
            let lowered: HashSet<String> = names.iter().map(|name| name.to_lowercase()).collect();
            if lowered.len() != names.len() {
                return Err("choice option names must be unique".into());
            }
            let labels = CHOICE_LABELS[..names.len()].iter().map(|label| label.to_string()).collect();
            Ok((
                render_labels(
                    state,
                    instructions,
                    &lines,
                    "Answer with the letter of one option only.",
                ),
                labels,
            ))
        }
        "score" => {
            let Some(criteria) = question.criteria.as_ref().and_then(Value::as_array) else {
                return Err("a score question needs 2 to 10 ordered levels".into());
            };
            if criteria.len() < 2 || criteria.len() > MAX_LEVELS {
                return Err("a score question needs 2 to 10 ordered levels".into());
            }
            let levels: Vec<String> = criteria.iter().map(text).collect();
            if levels.iter().any(|level| level.trim().is_empty()) {
                return Err("score levels must not be blank".into());
            }
            let lines: Vec<String> = levels
                .iter()
                .enumerate()
                .map(|(index, level)| format!("{}: {level}", LEVEL_LABELS[index]))
                .collect();
            let labels = LEVEL_LABELS[..levels.len()].iter().map(|label| label.to_string()).collect();
            Ok((
                render_labels(
                    state,
                    instructions,
                    &lines,
                    "Answer with the number of one level only.",
                ),
                labels,
            ))
        }
        "noul" => {
            if let Some(criteria) = question.criteria.as_ref() {
                if !criteria.is_object() && !criteria.is_null() {
                    return Err("noul criteria must be an object or null".into());
                }
            }
            let map = question
                .criteria
                .as_ref()
                .and_then(Value::as_object);
            if let Some(map) = map {
                for key in map.keys() {
                    if key != "true" && key != "false" {
                        return Err(format!("unknown noul criteria keys: [{key}]"));
                    }
                }
            }
            let mut lines = Vec::new();
            for (label, key) in [("yes", "true"), ("no", "false")] {
                if let Some(description) = map.and_then(|map| map.get(key)) {
                    let description = text(description);
                    if !description.is_empty() {
                        lines.push(format!("{label}: {description}"));
                    }
                }
            }
            if instructions.unwrap_or("").is_empty() && lines.is_empty() {
                return Err("a noul question needs instructions or true/false descriptions".into());
            }
            Ok((
                render_labels(state, instructions, &lines, "Answer with one label only."),
                NOUL_LABELS.iter().map(|label| label.to_string()).collect(),
            ))
        }
        other => Err(format!("unsupported systemone question type {other:?}")),
    }
}

fn decision_prompt(state: &str, item: &DecisionItem) -> Result<(String, Vec<String>), String> {
    let instructions = Some(item.question.as_str());
    if item.kind == "choice" || item.kind == "predicate" {
        let raw = item.options.as_ref().or(item.choices.as_ref());
        let Some(raw) = raw else {
            return Err(format!("a choice needs 1 to {MAX_CHOICES} options"));
        };
        if raw.is_empty() || raw.len() > MAX_CHOICES {
            return Err(format!("a choice needs 1 to {MAX_CHOICES} options"));
        }
        let mut lines = Vec::new();
        let mut names = Vec::new();
        for option in raw {
            let Some(value) = option.get("value") else {
                return Err("choice options need a string value".into());
            };
            if !value.is_string() && !value.is_boolean() {
                return Err("choice option values must be strings or booleans".into());
            }
            let name = check_name(&py_str(value), "choice value")?;
            let mut line = format!("{}: {name}", CHOICE_LABELS[names.len()]);
            if let Some(description) = option.get("description") {
                let description = text(description);
                if !description.is_empty() {
                    line.push_str(&format!(" - {description}"));
                }
            }
            lines.push(line);
            names.push(name);
        }
        let lowered: HashSet<String> = names.iter().map(|name| name.to_lowercase()).collect();
        if lowered.len() != names.len() {
            return Err("choice values must be unique".into());
        }
        let labels = CHOICE_LABELS[..names.len()].iter().map(|label| label.to_string()).collect();
        return Ok((
            render_labels(state, instructions, &lines, "Answer with the letter of one option only."),
            labels,
        ));
    }
    if item.kind == "score" {
        let levels = item.levels.as_ref().map(Vec::as_slice).unwrap_or(&[]);
        if levels.len() < 2 || levels.len() > MAX_LEVELS {
            return Err("a score question needs 2 to 10 levels".into());
        }
        let rendered: Vec<String> = levels
            .iter()
            .map(|level| match level {
                Value::Object(map) => map.get("label").map(py_str).unwrap_or_else(|| py_str(level)),
                other => text(other),
            })
            .collect();
        if rendered.iter().any(|level| level.trim().is_empty()) {
            return Err("score levels must not be blank".into());
        }
        let lines: Vec<String> = rendered
            .iter()
            .enumerate()
            .map(|(index, level)| format!("{}: {level}", LEVEL_LABELS[index]))
            .collect();
        let labels = LEVEL_LABELS[..rendered.len()].iter().map(|label| label.to_string()).collect();
        return Ok((
            render_labels(state, instructions, &lines, "Answer with the number of one level only."),
            labels,
        ));
    }
    let mut lines = Vec::new();
    if let Some(yes) = &item.yes {
        lines.push(format!("yes: {}", text(&Value::String(yes.clone()))));
    }
    if let Some(no) = &item.no {
        lines.push(format!("no: {}", text(&Value::String(no.clone()))));
    }
    Ok((
        render_labels(state, instructions, &lines, "Answer with one label only."),
        NOUL_LABELS.iter().map(|label| label.to_string()).collect(),
    ))
}

/// Verify each answer label is exactly one token appended to the rendered
/// prompt, and return those token ids plus the base prompt length.
async fn label_token_ids(
    state: &SharedState,
    rendered: &str,
    labels: &[String],
    question_id: &str,
) -> Result<(Vec<u32>, usize), String> {
    let base = state
        .tokenizer_batcher
        .encode(rendered.to_string())
        .await
        .map_err(|err| err)?;
    let mut token_ids = Vec::with_capacity(labels.len());
    for label in labels {
        let candidate = state
            .tokenizer_batcher
            .encode(format!("{rendered}{label}"))
            .await?;
        if candidate.len() != base.len() + 1 || candidate[..base.len()] != base[..] {
            return Err(format!(
                "question {question_id:?}: answer label {label:?} is not a single token at the answer position"
            ));
        }
        token_ids.push(candidate[base.len()]);
    }
    let unique: HashSet<u32> = token_ids.iter().copied().collect();
    if unique.len() != token_ids.len() {
        return Err(format!("question {question_id:?}: answer labels are not distinct tokens"));
    }
    Ok((token_ids, base.len()))
}

fn probabilities(logprobs: &[f64], temperature: f64) -> Result<(Vec<f64>, f64), String> {
    if logprobs.is_empty() {
        return Err("a decision read returned no label scores".into());
    }
    if logprobs.iter().any(|value| !value.is_finite()) {
        return Err("a decision read returned invalid scores".into());
    }
    let scaled: Vec<f64> = logprobs.iter().map(|value| value / temperature).collect();
    let maximum = scaled.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let weights: Vec<f64> = scaled.iter().map(|value| (value - maximum).exp()).collect();
    let total: f64 = weights.iter().sum();
    let probabilities: Vec<f64> = weights.iter().map(|value| value / total).collect();
    let label_mass = logprobs
        .iter()
        .fold(0.0f64, |sum, value| sum + value.exp())
        .min(1.0);
    Ok((probabilities, label_mass))
}

struct ReadResult {
    probabilities: Vec<f64>,
    label_mass: f64,
    prompt_tokens: usize,
}

async fn read_decision(
    state: &SharedState,
    prompt: &str,
    labels: &[String],
    temperature: f64,
    request_id: String,
    max_model_len: usize,
) -> Result<ReadResult, String> {
    let messages = vec![json!({"role": "user", "content": prompt})];
    let rendered = state
        .tokenizer
        .render_chat(&messages, None, false)
        .map_err(|err| err.to_string())?;
    let (token_ids, prompt_tokens) = label_token_ids(state, &rendered, labels, &request_id).await?;
    if prompt_tokens + 1 > max_model_len {
        return Err("decision prompt exceeds the model context length".into());
    }

    let stream_id = state.client.next_stream_id();
    let mut events = state.client.subscribe(stream_id).await;
    let sampling = crate::types::ChatCompletionRequest::default()
        .sampling_with_labels(None, &token_ids);
    // The engine emits score-label log-probabilities whenever `score_labels`
    // is set, independent of the `logprobs` flag (same as Python's `_read`).
    let batch = SubmitBatch {
        command_id: 0,
        target_active_requests: 1,
        submissions: vec![Submission {
            request_id: request_id.clone(),
            stream_id,
            token_ids: state
                .tokenizer_batcher
                .encode(rendered)
                .await?,
            max_new_tokens: 1,
            sampling,
            lora: None,
            output_modalities: vec!["text".into()],
            audio_output: None,
        }],
    };
    let result = state
        .client
        .submit_batch(&batch)
        .await
        .map_err(|err| err.to_string())?;
    for item in result.results {
        if !item.ok {
            let message = item
                .error
                .as_ref()
                .and_then(|error| error.get("message"))
                .and_then(Value::as_str)
                .unwrap_or("decision read was rejected");
            return Err(message.to_string());
        }
    }

    let mut label_logprobs = Vec::new();
    let mut terminated = false;
    for _ in 0..16 {
        let Some(event) = events.recv().await else {
            break;
        };
        if !event.token_ids.is_empty() {
            if event.token_ids != token_ids {
                return Err("decision read returned unexpected label ids".into());
            }
            label_logprobs.extend(event.logprobs.iter().map(|value| *value as f64));
        }
        if event.error.is_some() {
            return Err("decision read failed".into());
        }
        if event.finish_reason.is_some() {
            terminated = true;
            break;
        }
    }
    state.client.unsubscribe(stream_id).await;
    if !terminated {
        return Err("decision read did not terminate".into());
    }
    if label_logprobs.len() != labels.len() {
        return Err("decision read returned an incomplete label distribution".into());
    }
    let (probabilities, label_mass) = probabilities(&label_logprobs, temperature)?;
    Ok(ReadResult {
        probabilities,
        label_mass,
        prompt_tokens,
    })
}

fn argmax(probabilities: &[f64]) -> usize {
    probabilities
        .iter()
        .enumerate()
        .max_by(|left, right| left.1.partial_cmp(right.1).unwrap_or(std::cmp::Ordering::Equal))
        .map(|(index, _)| index)
        .unwrap_or(0)
}

fn system_one_answer(
    question: &SystemOneQuestion,
    probabilities: &[f64],
    label_mass: f64,
) -> Value {
    let best = argmax(probabilities);
    let confidence = probabilities[best] * label_mass;
    match question.kind.as_str() {
        "choice" => {
            let names: Vec<String> = question
                .criteria
                .as_ref()
                .and_then(Value::as_object)
                .map(|map| map.keys().cloned().collect())
                .unwrap_or_default();
            let mut probs = Map::new();
            for (index, name) in names.iter().enumerate() {
                probs.insert(name.clone(), json!(probabilities[index]));
            }
            json!({
                "type": "choice",
                "choice": names[best],
                "probabilities": Value::Object(probs),
                "confidence": confidence,
            })
        }
        "score" => {
            let levels: Vec<String> = question
                .criteria
                .as_ref()
                .and_then(Value::as_array)
                .map(|items| items.iter().map(text).collect())
                .unwrap_or_default();
            let score: f64 = probabilities
                .iter()
                .enumerate()
                .map(|(index, probability)| index as f64 * probability)
                .sum();
            let mut legend = Map::new();
            let mut probs = Map::new();
            for index in 0..probabilities.len() {
                legend.insert(index.to_string(), json!(levels.get(index).cloned().unwrap_or_default()));
                probs.insert(index.to_string(), json!(probabilities[index]));
            }
            json!({
                "type": "score",
                "score": score,
                "legend": Value::Object(legend),
                "probabilities": Value::Object(probs),
                "confidence": confidence,
            })
        }
        _ => {
            let yes = probabilities[0];
            json!({
                "type": "noul",
                "noul": yes,
                "probabilities": {"yes": yes, "no": probabilities[1]},
                "confidence": confidence,
            })
        }
    }
}

fn decision_answer(item: &DecisionItem, labels: &[String], probabilities: &[f64], label_mass: f64) -> Value {
    let best = argmax(probabilities);
    let mut answer = json!({"type": item.kind, "label_mass": label_mass});
    if item.kind == "choice" || item.kind == "predicate" {
        let raw = item.options.as_ref().or(item.choices.as_ref());
        let values: Vec<String> = raw
            .map(|items| items.iter().map(|option| py_str(option.get("value").unwrap_or(&Value::Null))).collect())
            .unwrap_or_default();
        let mut probs = Map::new();
        for (value, probability) in values.iter().zip(probabilities) {
            probs.insert(value.clone(), json!(probability));
        }
        answer["choice"] = json!(values.get(best).cloned().unwrap_or_default());
        answer["probabilities"] = Value::Object(probs);
    } else if item.kind == "score" {
        let mut probs = Map::new();
        for (index, probability) in probabilities.iter().enumerate() {
            probs.insert(index.to_string(), json!(probability));
        }
        answer["probabilities"] = Value::Object(probs);
    } else {
        let mut probs = Map::new();
        for (label, probability) in NOUL_LABELS.iter().zip(probabilities) {
            probs.insert(label.to_string(), json!(probability));
        }
        answer["probabilities"] = Value::Object(probs);
    }
    if item.kind == "score" {
        answer["score"] = json!(
            probabilities
                .iter()
                .enumerate()
                .map(|(index, probability)| index as f64 * probability)
                .sum::<f64>()
        );
    } else if item.kind != "choice" && item.kind != "predicate" {
        answer["choice"] = json!(labels[best]);
    }
    answer
}

pub async fn systemone(State(state): State<SharedState>, Json(payload): Json<Value>) -> Response {
    let request: SystemOneRequest = match serde_json::from_value(payload) {
        Ok(request) => request,
        Err(err) => return bad_request(err.to_string()),
    };
    if let Err(message) = check_template_kwargs(request.chat_template_kwargs.as_ref()) {
        return bad_request(message);
    }
    if request.questions.len() > MAX_QUESTIONS {
        return bad_request(format!("at most {MAX_QUESTIONS} questions per request"));
    }
    if request.questions.is_empty() {
        return bad_request("questions must not be empty");
    }
    let state_text = request.state.as_ref().map(text).unwrap_or_default();
    let root = uuid::Uuid::new_v4().simple().to_string();
    let mut prepared = Vec::new();
    for (question_id, question) in &request.questions {
        if question_id.trim().is_empty() {
            return bad_request("question ids must not be blank");
        }
        match system_one_prompt(&state_text, question) {
            Ok((prompt, labels)) => prepared.push((question_id.clone(), question, prompt, labels)),
            Err(message) => return bad_request(message),
        }
    }
    let reads = try_join_all(prepared.iter().enumerate().map(|(index, (_, _, prompt, labels))| {
        let state = state.clone();
        let prompt = prompt.clone();
        let labels = labels.clone();
        let request_id = format!("decision-{root}-{index}");
        async move {
            read_decision(&state, &prompt, &labels, 1.0, request_id, state.max_model_len).await
        }
    }))
    .await;
    let reads = match reads {
        Ok(reads) => reads,
        Err(message) => {
            if message.starts_with("decision read") || message.contains("did not terminate") {
                return unavailable(message);
            }
            return bad_request(message);
        }
    };

    let mut answers = Map::new();
    let mut diagnostics = Map::new();
    let mut input_tokens = 0usize;
    for ((question_id, question, _, _labels), read) in prepared.iter().zip(reads) {
        answers.insert(
            question_id.clone(),
            system_one_answer(question, &read.probabilities, read.label_mass),
        );
        diagnostics.insert(
            question_id.clone(),
            json!({"label_mass": read.label_mass, "argmax_is_label": true}),
        );
        input_tokens += read.prompt_tokens;
    }
    Json(json!({
        "id": format!("decision-{root}"),
        "object": "structured_decision",
        "created": std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs(),
        "model": state.served_model_name,
        "answers": Value::Object(answers),
        "usage": {"input_tokens": input_tokens, "output_tokens": 0},
        "diagnostics": Value::Object(diagnostics),
    }))
    .into_response()
}

pub async fn decisions(State(state): State<SharedState>, Json(payload): Json<Value>) -> Response {
    let request: DecisionsRequest = match serde_json::from_value(payload) {
        Ok(request) => request,
        Err(err) => return bad_request(err.to_string()),
    };
    if let Err(message) = check_template_kwargs(request.chat_template_kwargs.as_ref()) {
        return bad_request(message);
    }
    if !matches!(request.prompt_format_version, None | Some(PROMPT_FORMAT_VERSION)) {
        return bad_request(format!(
            "unsupported prompt_format_version {:?}",
            request.prompt_format_version
        ));
    }
    if request.images.as_ref().is_some_and(|value| !value.is_null() && value.as_array().is_some_and(|items| !items.is_empty())) {
        return bad_request("decision image inputs are not supported");
    }
    if request.questions.len() > MAX_QUESTIONS {
        return bad_request(format!("at most {MAX_QUESTIONS} questions per request"));
    }
    if !request.temperature.is_finite() || request.temperature <= 0.0 {
        return bad_request("temperature must be positive");
    }
    let state_text = text(request.input.as_ref().or(request.state.as_ref()).unwrap_or(&Value::Null));
    if state_text.trim().is_empty() {
        return bad_request("input must not be blank");
    }
    let root = uuid::Uuid::new_v4().simple().to_string();
    let mut seen = HashSet::new();
    let mut prepared = Vec::new();
    for item in &request.questions {
        if !seen.insert(item.id.clone()) {
            return bad_request(format!("duplicate question id {:?}", item.id));
        }
        match decision_prompt(&state_text, item) {
            Ok((prompt, labels)) => prepared.push((item, prompt, labels)),
            Err(message) => return bad_request(message),
        }
    }
    let reads = try_join_all(prepared.iter().enumerate().map(|(index, (_, prompt, labels))| {
        let state = state.clone();
        let prompt = prompt.clone();
        let labels = labels.clone();
        let temperature = request.temperature;
        let request_id = format!("decision-{root}-{index}");
        async move {
            read_decision(&state, &prompt, &labels, temperature, request_id, state.max_model_len).await
        }
    }))
    .await;
    let reads = match reads {
        Ok(reads) => reads,
        Err(message) => {
            if message.starts_with("decision read") || message.contains("did not terminate") {
                return unavailable(message);
            }
            return bad_request(message);
        }
    };
    let mut answers = Map::new();
    let mut input_tokens = 0usize;
    for ((item, _, labels), read) in prepared.iter().zip(reads) {
        answers.insert(
            item.id.clone(),
            decision_answer(item, &labels, &read.probabilities, read.label_mass),
        );
        input_tokens += read.prompt_tokens;
    }
    Json(json!({
        "object": "decisions",
        "model": state.served_model_name,
        "prompt_format_version": PROMPT_FORMAT_VERSION,
        "answers": Value::Object(answers),
        "usage": {"prompt_tokens": input_tokens, "completion_tokens": 0},
    }))
    .into_response()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn canonical_text_sorts_keys_and_keeps_unicode() {
        let value = json!({"b": 2, "a": {"d": "值", "c": 1}});
        assert_eq!(text(&value), "{\"a\":{\"c\":1,\"d\":\"值\"},\"b\":2}");
        assert_eq!(text(&Value::String("raw".into())), "raw");
    }

    #[test]
    fn systemone_request_state_is_rendered_into_prompt() {
        // Python SYSTEMONE_BODY uses a JSON object as `state`; `_text` emits
        // canonical JSON at the head of every question prompt.
        let request: SystemOneRequest = serde_json::from_value(json!({
            "state": {"ticket": "My card was charged twice for one order."},
            "questions": {
                "urgent": {
                    "type": "noul",
                    "instructions": "Reply within an hour?"
                }
            }
        }))
        .unwrap();
        let state_text = request.state.as_ref().map(text).unwrap_or_default();
        assert_eq!(
            state_text,
            "{\"ticket\":\"My card was charged twice for one order.\"}"
        );
        let question = &request.questions["urgent"];
        let (prompt, labels) = system_one_prompt(&state_text, question).unwrap();
        assert_eq!(labels, vec!["yes", "no"]);
        assert!(prompt.starts_with(
            "{\"ticket\":\"My card was charged twice for one order.\"}\n\nQuestion: Reply within an hour?\n"
        ));
    }

    #[test]
    fn systemone_choice_prompt_and_labels() {
        let question = SystemOneQuestion {
            kind: "choice".into(),
            instructions: Some("Pick".into()),
            criteria: Some(json!({"cats": "feline", "dogs": null})),
        };
        let (prompt, labels) = system_one_prompt("ctx", &question).unwrap();
        assert_eq!(labels, vec!["A", "B"]);
        assert!(prompt.contains("Question: Pick"));
        assert!(prompt.contains("A: cats - feline"));
        assert!(prompt.contains("B: dogs"));
        assert!(prompt.ends_with("Answer with the letter of one option only."));
    }

    #[test]
    fn prompts_match_python_reference_byte_for_byte() {
        let choice = SystemOneQuestion {
            kind: "choice".into(),
            instructions: Some("Pick".into()),
            criteria: Some(json!({"cats": "feline", "dogs": null})),
        };
        let (prompt, labels) = system_one_prompt("ctx", &choice).unwrap();
        assert_eq!(
            prompt,
            "ctx\n\nQuestion: Pick\nA: cats - feline\nB: dogs\nAnswer with the letter of one option only."
        );
        assert_eq!(labels, vec!["A", "B"]);

        let yes_no = DecisionItem {
            id: "q1".into(),
            kind: "yes_no".into(),
            question: "Is it?".into(),
            options: None,
            choices: None,
            levels: None,
            yes: Some("it is".into()),
            no: Some("it is not".into()),
        };
        let (prompt, labels) = decision_prompt("ctx", &yes_no).unwrap();
        assert_eq!(
            prompt,
            "ctx\n\nQuestion: Is it?\nyes: it is\nno: it is not\nAnswer with one label only."
        );
        assert_eq!(labels, vec!["yes", "no"]);

        let predicate = DecisionItem {
            id: "q2".into(),
            kind: "predicate".into(),
            question: "Pick".into(),
            options: Some(vec![json!({"value": true}), json!({"value": "y"})]),
            choices: None,
            levels: None,
            yes: None,
            no: None,
        };
        let (prompt, labels) = decision_prompt("ctx", &predicate).unwrap();
        assert_eq!(
            prompt,
            "ctx\n\nQuestion: Pick\nA: True\nB: y\nAnswer with the letter of one option only."
        );
        assert_eq!(labels, vec!["A", "B"]);
    }

    #[test]
    fn decision_yes_no_prompt_and_probabilities() {
        let item = DecisionItem {
            id: "q1".into(),
            kind: "yes_no".into(),
            question: "Is it?".into(),
            options: None,
            choices: None,
            levels: None,
            yes: Some("it is".into()),
            no: Some("it is not".into()),
        };
        let (prompt, labels) = decision_prompt("ctx", &item).unwrap();
        assert_eq!(labels, vec!["yes", "no"]);
        assert!(prompt.contains("yes: it is"));

        let (probs, mass) = probabilities(&[2.0_f64.ln(), 1.0_f64.ln()], 1.0).unwrap();
        assert!((probs[0] - 2.0 / 3.0).abs() < 1e-9);
        assert!(mass > 0.0 && mass <= 1.0);
    }

    #[test]
    fn decision_answer_reports_choice_and_score() {
        let item = DecisionItem {
            id: "q1".into(),
            kind: "predicate".into(),
            question: "Pick".into(),
            options: Some(vec![json!({"value": "x"}), json!({"value": "y"})]),
            choices: None,
            levels: None,
            yes: None,
            no: None,
        };
        let answer = decision_answer(&item, &["A".into(), "B".into()], &[0.2, 0.8], 0.9);
        assert_eq!(answer["choice"], "y");
        assert_eq!(answer["probabilities"]["x"], 0.2);
    }
}
