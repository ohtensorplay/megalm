//! OpenAI-compatible request types and sampling-params lowering.

use std::collections::HashMap;

use serde::{Deserialize, Serialize};
use serde_json::{json, Map, Value};

/// Variables the frontend owns. Requests may not override them through
/// `chat_template_kwargs` (mirrors vLLM's `resolve_chat_template_kwargs`).
const RESERVED_TEMPLATE_VARS: &[&str] = &[
    "chat_template",
    "tokenize",
    "messages",
    "tools",
    "add_generation_prompt",
    "open_thinking",
    "enable_thinking",
    "add_vision_id",
    "add_audio_id",
    "documents",
    "bos_token",
    "eos_token",
    "pad_token",
    "unk_token",
];

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(default)]
pub struct ChatCompletionRequest {
    pub model: String,
    pub messages: Vec<Value>,
    pub max_tokens: Option<u64>,
    pub max_completion_tokens: Option<u64>,
    pub stream: bool,
    pub stream_options: Option<Value>,
    pub temperature: Option<f32>,
    pub top_p: Option<f32>,
    pub top_k: Option<u32>,
    pub min_p: Option<f32>,
    pub repetition_penalty: Option<f32>,
    pub frequency_penalty: Option<f32>,
    pub presence_penalty: Option<f32>,
    pub seed: Option<i64>,
    pub stop: Option<Value>,
    pub stop_token_ids: Option<Vec<u32>>,
    pub ignore_eos: bool,
    pub n: Option<usize>,
    pub logprobs: Option<bool>,
    pub top_logprobs: Option<u32>,
    pub tools: Option<Vec<Value>>,
    pub tool_choice: Option<Value>,
    pub response_format: Option<Value>,
    pub open_thinking: bool,
    pub chat_template_kwargs: Option<Value>,
    pub cancel_id: Option<String>,
    #[serde(flatten)]
    #[allow(dead_code)]
    pub extra: HashMap<String, Value>,
}

impl Default for ChatCompletionRequest {
    fn default() -> Self {
        Self {
            model: String::new(),
            messages: Vec::new(),
            max_tokens: None,
            max_completion_tokens: None,
            stream: false,
            stream_options: None,
            temperature: None,
            top_p: None,
            top_k: None,
            min_p: None,
            repetition_penalty: None,
            frequency_penalty: None,
            presence_penalty: None,
            seed: None,
            stop: None,
            stop_token_ids: None,
            ignore_eos: false,
            n: None,
            logprobs: None,
            top_logprobs: None,
            tools: None,
            tool_choice: None,
            response_format: None,
            open_thinking: false,
            chat_template_kwargs: None,
            cancel_id: None,
            extra: HashMap::new(),
        }
    }
}

impl ChatCompletionRequest {
    pub fn max_new_tokens(&self) -> u64 {
        self.max_completion_tokens
            .or(self.max_tokens)
            .unwrap_or(256)
    }

    pub fn stop_sequences(&self) -> Vec<String> {
        match &self.stop {
            Some(Value::String(text)) => vec![text.clone()],
            Some(Value::Array(items)) => items
                .iter()
                .filter_map(|item| item.as_str().map(str::to_string))
                .collect(),
            _ => Vec::new(),
        }
    }

    /// Extra Jinja variables for the chat template, filtered to keys the
    /// template may legitimately read. `reasoning_effort` is also accepted as
    /// a top-level convenience field.
    pub fn template_extras(&self) -> std::result::Result<Option<Value>, String> {
        let mut extras = Map::new();
        if let Some(value) = &self.chat_template_kwargs {
            let Value::Object(kwargs) = value else {
                return Err("chat_template_kwargs must be an object".into());
            };
            for key in ["chat_template", "tokenize"] {
                if kwargs.contains_key(key) {
                    return Err(format!("unexpected chat template kwarg: {key}"));
                }
            }
            for (key, value) in kwargs {
                if !RESERVED_TEMPLATE_VARS.contains(&key.as_str()) {
                    extras.insert(key.clone(), value.clone());
                }
            }
        }
        if !extras.contains_key("reasoning_effort") {
            if let Some(value) = self.extra.get("reasoning_effort") {
                extras.insert("reasoning_effort".into(), value.clone());
            }
        }
        Ok((!extras.is_empty()).then(|| Value::Object(extras)))
    }

    pub fn include_usage(&self) -> bool {
        self.stream_options
            .as_ref()
            .and_then(|options| options.get("include_usage"))
            .and_then(Value::as_bool)
            .unwrap_or(false)
    }

    /// Decision reads need the engine to score a fixed label set instead of
    /// sampling. The field is omitted for ordinary requests.
    pub fn sampling_with_labels(
        &self,
        structured_output: Option<Value>,
        score_labels: &[u32],
    ) -> Value {
        let mut value = self.sampling(structured_output);
        if let Value::Object(map) = &mut value {
            map.insert("score_labels".into(), json!(score_labels));
        }
        value
    }

    pub fn sampling(&self, structured_output: Option<Value>) -> Value {
        let mut map = Map::new();
        map.insert("temperature".into(), json!(self.temperature.unwrap_or(0.0)));
        map.insert("top_p".into(), json!(self.top_p.unwrap_or(1.0)));
        map.insert("top_k".into(), json!(self.top_k.unwrap_or(0)));
        map.insert("min_p".into(), json!(self.min_p.unwrap_or(0.0)));
        map.insert(
            "repetition_penalty".into(),
            json!(self.repetition_penalty.unwrap_or(1.0)),
        );
        map.insert("seed".into(), json!(self.seed));
        map.insert(
            "stop_token_ids".into(),
            json!(self.stop_token_ids.clone().unwrap_or_default()),
        );
        map.insert("ignore_eos".into(), json!(self.ignore_eos));
        map.insert("logprobs".into(), json!(self.logprobs.unwrap_or(false)));
        map.insert("top_logprobs".into(), json!(self.top_logprobs.unwrap_or(0)));
        if let Some(structured) = structured_output {
            map.insert("structured_output".into(), structured);
        }
        Value::Object(map)
    }
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(default)]
pub struct CompletionRequest {
    pub model: String,
    pub prompt: Value,
    pub max_tokens: Option<u64>,
    pub stream: bool,
    pub temperature: Option<f32>,
    pub top_p: Option<f32>,
    #[serde(default)]
    pub ignore_eos: bool,
    pub cancel_id: Option<String>,
    #[serde(flatten)]
    #[allow(dead_code)]
    pub extra: HashMap<String, Value>,
}

impl Default for CompletionRequest {
    fn default() -> Self {
        Self {
            model: String::new(),
            prompt: Value::Null,
            max_tokens: None,
            stream: false,
            temperature: None,
            top_p: None,
            ignore_eos: false,
            cancel_id: None,
            extra: HashMap::new(),
        }
    }
}

/// Normalize `tools` / `tool_choice` into `(tools, forced_tool_name)`.
pub fn normalized_tools(
    tools: Option<&[Value]>,
    tool_choice: Option<&Value>,
) -> Result<(Vec<Value>, Option<String>), String> {
    let mut selected = Vec::new();
    for tool in tools.unwrap_or_default() {
        if tool.get("type").and_then(Value::as_str) != Some("function") {
            continue;
        }
        let function = tool.get("function").ok_or("tool has no function")?;
        let name = function
            .get("name")
            .and_then(Value::as_str)
            .ok_or("tool function has no name")?;
        if name.is_empty() {
            return Err("tool function name is empty".into());
        }
        if selected.iter().any(|item: &Value| {
            item.get("function")
                .and_then(|f| f.get("name"))
                .and_then(Value::as_str)
                == Some(name)
        }) {
            return Err("duplicate tool name".into());
        }
        let mut normalized = tool.clone();
        if let Value::Object(map) = &mut normalized {
            map.insert("type".into(), json!("function"));
        }
        selected.push(normalized);
    }

    let forced = match tool_choice {
        None | Some(Value::Null) => None,
        Some(Value::String(choice)) => match choice.as_str() {
            "none" => return Ok((Vec::new(), None)),
            "auto" | "required" => None,
            other => return Err(format!("unsupported tool_choice `{other}`")),
        },
        Some(Value::Object(map)) => {
            if map.get("type").and_then(Value::as_str) == Some("function") {
                map.get("function")
                    .and_then(|function| function.get("name"))
                    .and_then(Value::as_str)
                    .map(str::to_string)
            } else {
                return Err("unsupported tool_choice object".into());
            }
        }
        Some(_) => return Err("unsupported tool_choice".into()),
    };
    Ok((selected, forced))
}

/// Response-format mode and optional canonical JSON schema.
pub fn normalized_response_format(
    format: Option<&Value>,
) -> Result<(String, Option<String>), String> {
    let Some(format) = format else {
        return Ok((String::new(), None));
    };
    let kind = format
        .get("type")
        .and_then(Value::as_str)
        .ok_or("response_format has no type")?;
    match kind {
        "json_object" => Ok(("json_object".into(), None)),
        "json_schema" => {
            let schema = format
                .get("json_schema")
                .and_then(|schema| schema.get("schema"))
                .ok_or("response_format.json_schema has no schema")?;
            let mut canonical = serde_json::to_value(schema).map_err(|err| err.to_string())?;
            canonical = canonicalize(&canonical);
            Ok((
                "json_schema".into(),
                Some(serde_json::to_string(&canonical).map_err(|err| err.to_string())?),
            ))
        }
        other => Err(format!("unsupported response_format `{other}`")),
    }
}

fn canonicalize(value: &Value) -> Value {
    match value {
        Value::Object(map) => {
            let mut keys: Vec<&String> = map.keys().collect();
            keys.sort();
            let mut sorted = Map::new();
            for key in keys {
                sorted.insert(key.clone(), canonicalize(&map[key]));
            }
            Value::Object(sorted)
        }
        Value::Array(items) => Value::Array(items.iter().map(canonicalize).collect()),
        other => other.clone(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn template_extras_filters_reserved_variables() {
        let request = ChatCompletionRequest {
            chat_template_kwargs: Some(json!({
                "reasoning_effort": 0.4,
                "messages": [],
                "tools": [],
                "enable_thinking": true,
                "custom": "value",
            })),
            ..ChatCompletionRequest::default()
        };
        assert_eq!(
            request.template_extras().unwrap(),
            Some(json!({"reasoning_effort": 0.4, "custom": "value"}))
        );
    }

    #[test]
    fn template_extras_rejects_template_override_keys() {
        for key in ["chat_template", "tokenize"] {
            let request = ChatCompletionRequest {
                chat_template_kwargs: Some(json!({key: "x"})),
                ..ChatCompletionRequest::default()
            };
            assert!(request.template_extras().is_err());
        }
        let request = ChatCompletionRequest {
            chat_template_kwargs: Some(json!([])),
            ..ChatCompletionRequest::default()
        };
        assert!(request.template_extras().is_err());
    }

    #[test]
    fn template_extras_accepts_top_level_reasoning_effort() {
        let mut request = ChatCompletionRequest::default();
        request
            .extra
            .insert("reasoning_effort".into(), json!("low"));
        assert_eq!(
            request.template_extras().unwrap(),
            Some(json!({"reasoning_effort": "low"}))
        );
    }
}
