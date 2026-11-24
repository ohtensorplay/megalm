//! Tokenizer + trusted local template frontend (ported from `runtime/tokenizer.py`).

use std::collections::{HashMap, HashSet};
use std::path::Path;

use serde_json::Value;
use tokenizers::Tokenizer;

use crate::decode::ModelDecodeStream;
use crate::error::{message, FrontendError, Result};
use crate::multimodal::{expand_media_placeholders, MultimodalEncoding};
use megalm_renderer::{ChatRenderer, RenderContext};

const INKLING_MESSAGE_MODEL: &str = "<|message_model|>";
const INKLING_CONTENT_TEXT: &str = "<|content_text|>";
const INKLING_CONTENT_THINKING: &str = "<|content_thinking|>";
const INKLING_INVOKE_TOOL_JSON: &str = "<|content_invoke_tool_json|>";
const INKLING_END_MESSAGE: &str = "<|end_message|>";
const INKLING_MODEL_END_SAMPLING: &str = "<|content_model_end_sampling|>";

/// Owned tokenizer and chat-template frontend for one model directory.
pub struct ModelTokenizer {
    pub(crate) tokenizer: Tokenizer,
    renderer: ChatRenderer,
    pub(crate) model_type: String,
    adds_bos: bool,
    special_token_text: HashMap<String, String>,
    pub bos_token_id: Option<u32>,
    pub eos_token_id: u32,
    pub eos_token_ids: HashSet<u32>,
    pub pad_token_id: u32,
    pub image_token_id: Option<u32>,
    pub video_token_id: Option<u32>,
    pub audio_token_id: Option<u32>,
    pub vision_start_token_id: Option<u32>,
    pub vision_end_token_id: Option<u32>,
    pub audio_start_token_id: Option<u32>,
    pub audio_end_token_id: Option<u32>,
    grammar: String,
    tool_parser: String,
    tool_preserve_special: bool,
}

impl ModelTokenizer {
    /// Load `tokenizer.json`, `tokenizer_config.json`, `config.json` and the
    /// chat template from `model_path`. No Transformers dependency.
    pub fn new(model_path: impl AsRef<Path>) -> Result<Self> {
        let root = model_path.as_ref().to_path_buf();
        let tokenizer_path = root.join("tokenizer.json");
        let tokenizer = Tokenizer::from_file(&tokenizer_path).map_err(|_| FrontendError::Load {
            path: tokenizer_path.display().to_string(),
            source: std::io::Error::new(
                std::io::ErrorKind::NotFound,
                "tokenizer.json is missing or invalid",
            ),
        })?;
        let config = read_object(&root.join("tokenizer_config.json"))?;
        let renderer = ChatRenderer::new(&root)?;
        let template_source = renderer.source().to_string();

        let model_name = root
            .file_name()
            .map(|name| name.to_string_lossy().to_lowercase())
            .unwrap_or_default();
        let model_config = read_object_optional(&root.join("config.json"))?;
        let model_type = model_config
            .as_ref()
            .and_then(|config| config.get("model_type"))
            .and_then(Value::as_str)
            .unwrap_or("")
            .to_lowercase();

        let special_token_text = ["bos_token", "eos_token", "pad_token", "unk_token"]
            .into_iter()
            .map(|key| (key.to_string(), special_token_text(&config, key)))
            .collect::<HashMap<_, _>>();

        let bos_token_id = optional_special_id(&tokenizer, &config, "bos_token")?;
        let probe = tokenizer
            .encode("a", false)
            .map_err(FrontendError::Tokenizer)?;
        let adds_bos = probe.get_ids().first().copied() == bos_token_id && bos_token_id.is_some();

        let eos_token_id = special_id(&tokenizer, &config, "eos_token")?;
        let mut eos_token_ids = HashSet::from([eos_token_id]);
        if let Some(config_eos) = model_config
            .as_ref()
            .and_then(|config| config.get("eos_token_id"))
        {
            match config_eos {
                Value::Number(number) => {
                    if let Some(id) = number.as_u64() {
                        eos_token_ids.insert(id as u32);
                    }
                }
                Value::Array(items) => {
                    for item in items {
                        if let Some(id) = item.as_u64() {
                            eos_token_ids.insert(id as u32);
                        }
                    }
                }
                _ => {}
            }
        }

        let pad_token_id = match optional_special_id(&tokenizer, &config, "pad_token")? {
            Some(id) => id,
            None => {
                let generation = read_object_optional(&root.join("generation_config.json"))?;
                generation
                    .as_ref()
                    .and_then(|config| config.get("pad_token_id"))
                    .and_then(Value::as_u64)
                    .map(|id| id as u32)
                    .unwrap_or(eos_token_id)
            }
        };

        let first_token_id = |tokens: &[&str]| -> Option<u32> {
            tokens
                .iter()
                .filter_map(|token| tokenizer.token_to_id(token))
                .next()
        };
        let image_token_id = first_token_id(&["<|image_pad|>", "<|IMAGE|>"]);
        let video_token_id = first_token_id(&["<|video_pad|>", "<|VIDEO|>"]);
        let audio_token_id = first_token_id(&["<|AUDIO|>"]);
        let vision_start_token_id = first_token_id(&["<|vision_start|>", "<|vision_bos|>"]);
        let vision_end_token_id = first_token_id(&["<|vision_end|>", "<|vision_eos|>"]);
        let audio_start_token_id = first_token_id(&["<|audio_bos|>"]);
        let audio_end_token_id = first_token_id(&["<|audio_eos|>"]);

        let grammar = if INKLING_MARKERS
            .iter()
            .all(|marker| tokenizer.token_to_id(marker).is_some())
        {
            "inkling".to_string()
        } else {
            "text".to_string()
        };
        let (tool_parser, tool_preserve_special) =
            detect_tool_parser(&template_source, &model_name, &model_type)?;

        Ok(Self {
            tokenizer,
            renderer,
            model_type,
            adds_bos,
            special_token_text,
            bos_token_id,
            eos_token_id,
            eos_token_ids,
            pad_token_id,
            image_token_id,
            video_token_id,
            audio_token_id,
            vision_start_token_id,
            vision_end_token_id,
            audio_start_token_id,
            audio_end_token_id,
            grammar,
            tool_parser,
            tool_preserve_special,
        })
    }

    /// Vocabulary size including added tokens.
    pub fn vocab_size(&self) -> usize {
        self.tokenizer.get_vocab_size(true)
    }

    pub fn token_to_id(&self, token: &str) -> Option<u32> {
        self.tokenizer.token_to_id(token)
    }

    /// Map every special-token id to the coarse kind used by the parser.
    pub fn special_token_kinds(&self) -> HashMap<u32, String> {
        self.tokenizer
            .get_added_tokens_decoder()
            .into_iter()
            .filter(|(_, added)| added.special)
            .map(|(id, added)| {
                let kind = match added.content.as_str() {
                    INKLING_MESSAGE_MODEL => "message",
                    INKLING_CONTENT_TEXT => "text",
                    INKLING_CONTENT_THINKING => "thinking",
                    _ => "other",
                };
                (id, kind.to_string())
            })
            .collect()
    }

    /// `"inkling"` when the tokenizer can express typed content blocks,
    /// otherwise the generic `<think>` text grammar.
    pub fn output_grammar(&self) -> &str {
        &self.grammar
    }

    /// Rust tool-call parser name and whether special tokens must be preserved.
    pub fn tool_call_parser(&self) -> (&str, bool) {
        (&self.tool_parser, self.tool_preserve_special)
    }

    pub fn encode(&self, text: &str) -> Result<Vec<u32>> {
        if text.is_empty() {
            return message("text must be a non-empty string");
        }
        Ok(self
            .tokenizer
            .encode(text, false)
            .map_err(FrontendError::Tokenizer)?
            .get_ids()
            .to_vec())
    }

    pub fn encode_batch(&self, texts: &[String]) -> Result<Vec<Vec<u32>>> {
        if texts.is_empty() || texts.iter().any(String::is_empty) {
            return message("tokenizer batch must contain non-empty strings");
        }
        let encodings = self
            .tokenizer
            .encode_batch(texts.to_vec(), false)
            .map_err(FrontendError::Tokenizer)?;
        Ok(encodings
            .iter()
            .map(|encoding| encoding.get_ids().to_vec())
            .collect())
    }

    pub fn decode(&self, token_ids: &[u32], skip_special_tokens: bool) -> Result<String> {
        self.tokenizer
            .decode(token_ids, skip_special_tokens)
            .map_err(FrontendError::Tokenizer)
    }

    pub fn decode_stream(&self, skip_special_tokens: bool) -> ModelDecodeStream<'_> {
        ModelDecodeStream::new(&self.tokenizer, skip_special_tokens)
    }

    /// Render messages to the assistant prompt text.
    pub fn render_chat(
        &self,
        messages: &[Value],
        tools: Option<&[Value]>,
        open_thinking: bool,
    ) -> Result<String> {
        self.render_chat_with(messages, tools, open_thinking, None)
    }

    /// Render with additional chat-template variables such as
    /// `reasoning_effort`. Reserved variables (`messages`, `tools`, ...) stay
    /// under the frontend's control.
    pub fn render_chat_with(
        &self,
        messages: &[Value],
        tools: Option<&[Value]>,
        open_thinking: bool,
        extras: Option<&Value>,
    ) -> Result<String> {
        let normalized = normalize_messages(messages)?;
        let bos = if self.adds_bos {
            String::new()
        } else {
            self.special_text("bos_token")
        };
        Ok(self.renderer.render(&RenderContext {
            messages: normalized,
            tools: tools.map(<[Value]>::to_vec),
            open_thinking,
            bos_token: bos,
            eos_token: self.special_text("eos_token"),
            pad_token: self.special_text("pad_token"),
            unk_token: self.special_text("unk_token"),
            extras: extras.cloned(),
        })?)
    }

    pub fn encode_chat(
        &self,
        messages: &[Value],
        tools: Option<&[Value]>,
        open_thinking: bool,
    ) -> Result<Vec<u32>> {
        self.encode(&self.render_chat(messages, tools, open_thinking)?)
    }

    /// Encode a multimodal chat, expanding media placeholders into feature rows.
    #[allow(clippy::too_many_arguments)]
    pub fn encode_multimodal_chat(
        &self,
        messages: &[Value],
        image_feature_counts: &[usize],
        video_frame_feature_counts: &[Vec<usize>],
        video_timestamps: &[Vec<f64>],
        audio_feature_counts: &[usize],
        tools: Option<&[Value]>,
        open_thinking: bool,
        video_timestamp_tokens: bool,
    ) -> Result<MultimodalEncoding> {
        if image_feature_counts.iter().any(|count| *count == 0) {
            return message("image feature counts must be positive");
        }
        if audio_feature_counts.iter().any(|count| *count == 0) {
            return message("audio feature counts must be positive");
        }
        if video_frame_feature_counts.len() != video_timestamps.len() {
            return message("video frame counts and timestamps differ");
        }
        if video_frame_feature_counts
            .iter()
            .zip(video_timestamps)
            .any(|(frames, stamps)| {
                frames.is_empty()
                    || frames.len() != stamps.len()
                    || frames.iter().any(|count| *count == 0)
            })
        {
            return message("video frame metadata is invalid");
        }

        let mut rendered = self.render_chat(messages, tools, open_thinking)?;
        if self.model_type == "oxr" && rendered.contains("<image>") {
            rendered = rendered.replace("<image>", "<|image_pad|>");
        }
        if !image_feature_counts.is_empty() && self.image_token_id.is_none() {
            return message("tokenizer has no image placeholder token");
        }
        if !video_frame_feature_counts.is_empty() && self.video_token_id.is_none() {
            return message("tokenizer has no video placeholder token");
        }
        if (!image_feature_counts.is_empty() || !video_frame_feature_counts.is_empty())
            && (self.vision_start_token_id.is_none() || self.vision_end_token_id.is_none())
        {
            return message("tokenizer has no vision boundary tokens");
        }
        if !audio_feature_counts.is_empty()
            && (self.audio_token_id.is_none()
                || self.audio_start_token_id.is_none()
                || self.audio_end_token_id.is_none())
        {
            return message("tokenizer has no audio placeholder tokens");
        }

        let input_ids = expand_media_placeholders(
            self,
            &self.encode(&rendered)?,
            image_feature_counts,
            video_frame_feature_counts,
            video_timestamps,
            audio_feature_counts,
            video_timestamp_tokens,
        )?;
        let token_type_ids = input_ids
            .iter()
            .map(|id| {
                if Some(*id) == self.image_token_id {
                    1
                } else if Some(*id) == self.video_token_id {
                    2
                } else if Some(*id) == self.audio_token_id {
                    3
                } else {
                    0
                }
            })
            .collect::<Vec<_>>();
        if token_type_ids.iter().filter(|kind| **kind == 1).count()
            != image_feature_counts.iter().sum::<usize>()
        {
            return message("expanded image placeholders were not preserved by the tokenizer");
        }
        if token_type_ids.iter().filter(|kind| **kind == 2).count()
            != video_frame_feature_counts
                .iter()
                .flat_map(|frames| frames.iter())
                .sum::<usize>()
        {
            return message("expanded video placeholders were not preserved by the tokenizer");
        }
        if token_type_ids.iter().filter(|kind| **kind == 3).count()
            != audio_feature_counts.iter().sum::<usize>()
        {
            return message("expanded audio placeholders were not preserved by the tokenizer");
        }

        Ok(MultimodalEncoding {
            input_ids,
            token_type_ids,
            image_count: image_feature_counts.len(),
            video_count: video_frame_feature_counts.len(),
            audio_count: audio_feature_counts.len(),
        })
    }

    pub(crate) fn special_text(&self, key: &str) -> String {
        self.special_token_text
            .get(key)
            .cloned()
            .unwrap_or_default()
    }

    pub(crate) fn encode_timestamp(&self, timestamp: f64) -> Result<Vec<u32>> {
        self.tokenizer
            .encode(format!("<{timestamp:.1} seconds>"), false)
            .map(|encoding| encoding.get_ids().to_vec())
            .map_err(FrontendError::Tokenizer)
    }
}

const INKLING_MARKERS: [&str; 6] = [
    INKLING_MESSAGE_MODEL,
    INKLING_CONTENT_TEXT,
    INKLING_CONTENT_THINKING,
    INKLING_INVOKE_TOOL_JSON,
    INKLING_END_MESSAGE,
    INKLING_MODEL_END_SAMPLING,
];

fn special_token_text(config: &Value, key: &str) -> String {
    match config.get(key) {
        Some(Value::Object(map)) => map
            .get("content")
            .and_then(Value::as_str)
            .unwrap_or("")
            .to_string(),
        Some(Value::String(text)) => text.clone(),
        _ => String::new(),
    }
}

fn special_id(tokenizer: &Tokenizer, config: &Value, key: &str) -> Result<u32> {
    let token = match config.get(key) {
        Some(Value::Object(map)) => map.get("content").and_then(Value::as_str),
        Some(Value::String(text)) => Some(text.as_str()),
        _ => None,
    }
    .ok_or_else(|| FrontendError::Message(format!("tokenizer config has no string {key}")))?;
    tokenizer.token_to_id(token).ok_or_else(|| {
        FrontendError::Message(format!("special token {token:?} is absent from vocabulary"))
    })
}

fn optional_special_id(tokenizer: &Tokenizer, config: &Value, key: &str) -> Result<Option<u32>> {
    if config.get(key).is_none() || config.get(key) == Some(&Value::Null) {
        Ok(None)
    } else {
        special_id(tokenizer, config, key).map(Some)
    }
}

fn read_object(path: &Path) -> Result<Value> {
    let text = std::fs::read_to_string(path).map_err(|source| FrontendError::Load {
        path: path.display().to_string(),
        source,
    })?;
    serde_json::from_str(&text).map_err(|source| FrontendError::Json {
        path: path.display().to_string(),
        source,
    })
}

fn read_object_optional(path: &Path) -> Result<Option<Value>> {
    if !path.is_file() {
        return Ok(None);
    }
    read_object(path).map(Some)
}

/// Normalize OpenAI chat messages into the stable shape HF templates expect.
fn normalize_messages(messages: &[Value]) -> Result<Vec<Value>> {
    if messages.is_empty() {
        return message("chat requires at least one message");
    }
    let mut normalized = Vec::with_capacity(messages.len());
    for item in messages {
        let role = item.get("role").and_then(Value::as_str);
        let role = role
            .filter(|role| !role.is_empty())
            .ok_or_else(|| FrontendError::Message("every chat message requires a role".into()))?;
        let mut item = item.clone();
        let content = item.get("content").cloned().unwrap_or(Value::Null);
        let content = match content {
            Value::String(text) => Value::String(text),
            Value::Array(parts) => Value::Array(
                parts
                    .into_iter()
                    .map(|part| {
                        if part.is_object() {
                            Ok(part)
                        } else {
                            message::<Value>("chat content parts must be objects")
                        }
                    })
                    .collect::<Result<Vec<_>>>()?,
            ),
            Value::Null => Value::String(String::new()),
            _ => return message("chat content must be text or a content-part array"),
        };
        item["role"] = Value::String(role.to_string());
        item["content"] = content;
        if role == "assistant" {
            if let Some(calls) = item.get("tool_calls").cloned() {
                if let Value::Array(calls) = calls {
                    let mut normalized_calls = Vec::with_capacity(calls.len());
                    for call in calls {
                        let mut call = call.as_object().cloned().ok_or_else(|| {
                            FrontendError::Message(
                                "assistant tool_calls entries must be objects".into(),
                            )
                        })?;
                        if let Some(function) = call.get("function").cloned() {
                            if let Value::Object(function) = function {
                                let mut function = function;
                                if let Some(arguments) = function.get("arguments").cloned() {
                                    if let Value::String(text) = arguments {
                                        let parsed = if text.trim().is_empty() {
                                            serde_json::Map::new()
                                        } else {
                                            match serde_json::from_str::<Value>(&text) {
                                                Ok(Value::Object(map)) => map,
                                                Ok(_) => {
                                                    return message(
                                                        "assistant tool call arguments must be a JSON object",
                                                    )
                                                }
                                                Err(_) => {
                                                    return message(
                                                        "assistant tool call arguments must be valid JSON",
                                                    )
                                                }
                                            }
                                        };
                                        function.insert("arguments".into(), Value::Object(parsed));
                                    }
                                }
                                call.insert("function".into(), Value::Object(function));
                            }
                        }
                        normalized_calls.push(Value::Object(call));
                    }
                    item["tool_calls"] = Value::Array(normalized_calls);
                }
            }
        }
        if item.get("reasoning_content").is_none() {
            item["reasoning_content"] = Value::Null;
        }
        if item.get("tool_calls").is_none() {
            item["tool_calls"] = Value::Array(Vec::new());
        }
        normalized.push(item);
    }
    Ok(normalized)
}

const KNOWN_TOOL_PARSERS: [&str; 21] = [
    "deepseek_v32",
    "deepseek_v4",
    "hermes",
    "granite4",
    "qwen3_xml",
    "qwen3_coder",
    "seed_oss",
    "glm45",
    "glm47",
    "hy_v3",
    "hy4",
    "minimax_m2",
    "minicpm5",
    "minicpm5_xml",
    "kimi_k2",
    "nemotron_v3",
    "mistral",
    "phi4mini",
    "internlm2",
    "llama3_json",
    "llama4_json",
];

#[cfg(test)]
mod whitelist_parity_tests {
    use super::KNOWN_TOOL_PARSERS;

    // Compile-time copy of the Python whitelist. The two frontends must accept
    // the same `MEGALM_TOOL_PARSER` names or an env override would behave
    // differently depending on which process reads it.
    const PY_TOKENIZER: &str = include_str!("../../../../src/megalm/runtime/tokenizer.py");

    #[test]
    fn python_and_rust_tool_parser_whitelists_match() {
        let start = PY_TOKENIZER
            .find("            known = {")
            .expect("Python whitelist");
        let rest = &PY_TOKENIZER[start..];
        let end = rest.find("            }").expect("Python whitelist end");
        let mut python: Vec<&str> = rest[..end]
            .lines()
            .filter_map(|line| {
                let item = line.trim().trim_end_matches(',');
                (item.starts_with('"') && item.ends_with('"')).then(|| item.trim_matches('"'))
            })
            .collect();
        python.sort_unstable();

        let mut rust = KNOWN_TOOL_PARSERS.to_vec();
        rust.sort_unstable();
        assert_eq!(python, rust, "tool parser whitelists diverged");
    }
}

fn detect_tool_parser(
    template: &str,
    model_name: &str,
    model_type: &str,
) -> Result<(String, bool)> {
    if let Ok(env_override) = std::env::var("MEGALM_TOOL_PARSER") {
        let name = env_override.trim().to_lowercase();
        if matches!(
            name.as_str(),
            "" | "none" | "off" | "generic" | "generic_json" | "json"
        ) {
            return Ok(("generic_json".into(), false));
        }
        if !KNOWN_TOOL_PARSERS.contains(&name.as_str()) {
            return message(format!("unknown MEGALM_TOOL_PARSER `{name}`"));
        }
        let preserve = matches!(
            name.as_str(),
            "deepseek_v32" | "deepseek_v4" | "internlm2" | "minicpm5" | "minicpm5_xml" | "kimi_k2"
        );
        return Ok((name, preserve));
    }

    let dsml = "\u{ff5c}DSML\u{ff5c}";
    if template.contains(dsml) && template.contains("function_calls") {
        return Ok(("deepseek_v32".into(), true));
    }
    if template.contains(dsml) && template.contains("tool_calls") {
        return Ok(("deepseek_v4".into(), true));
    }
    if template.contains("[TOOL_CALLS]") {
        return Ok(("mistral".into(), false));
    }
    if template.contains("functools[") {
        return Ok(("phi4mini".into(), false));
    }
    if template.contains("<|action_start|><|plugin|>") || model_name.contains("internlm2") {
        return Ok(("internlm2".into(), true));
    }
    if template.contains("<seed:tool_call>") {
        return Ok(("seed_oss".into(), false));
    }
    if template.contains("<minimax:tool_call>") {
        return Ok(("minimax_m2".into(), false));
    }
    if template.contains("<|tool_calls_section_begin|>") || model_name.contains("kimi-k2") {
        return Ok(("kimi_k2".into(), true));
    }
    if template.contains("<function name=") && template.contains("<param name=") {
        return Ok(("minicpm5".into(), true));
    }
    if model_name.contains("minicpm5") {
        return Ok(("minicpm5".into(), true));
    }
    if template.contains("<tool_sep>") || model_name.contains("hy3") || model_name.contains("hy4") {
        return Ok(("hy_v3".into(), false));
    }
    if template.contains("<function=") && template.contains("<parameter=") {
        return Ok(("qwen3_coder".into(), false));
    }
    if template.contains("<arg_key>") {
        if model_name.contains("glm-4.5") || model_name.contains("glm-4.6") {
            return Ok(("glm45".into(), false));
        }
        return Ok(("glm47".into(), false));
    }
    if model_name.contains("qwen3-coder")
        || model_name.contains("qwen2.5-coder")
        || model_name.contains("qwen3.5")
    {
        return Ok(("qwen3_coder".into(), false));
    }
    if model_name.contains("nemotron") {
        return Ok(("nemotron_v3".into(), false));
    }
    if model_name.contains("granite-4") || model_name.contains("granite4") {
        return Ok(("granite4".into(), false));
    }
    if model_name.contains("qwen") || model_type.contains("qwen") {
        return Ok(("qwen3_xml".into(), false));
    }
    if model_name.contains("mistral") || model_name.contains("mixtral") {
        return Ok(("mistral".into(), false));
    }
    if model_name.contains("llama-4") {
        return Ok(("llama4_json".into(), false));
    }
    if model_name.contains("llama-3") {
        return Ok(("llama3_json".into(), false));
    }
    if model_name.contains("glm-5") || model_name.contains("glm-4.7") {
        return Ok(("glm47".into(), false));
    }
    if model_name.contains("glm-4.5") || model_name.contains("glm-4.6") {
        return Ok(("glm45".into(), false));
    }
    if model_name.contains("phi-4-mini") || model_name.contains("phi4mini") {
        return Ok(("phi4mini".into(), false));
    }
    if template.contains("<tool_call>") {
        let lowered = template.to_lowercase();
        if model_type.contains("qwen") || lowered.contains("qwen") {
            return Ok(("qwen3_xml".into(), false));
        }
        return Ok(("hermes".into(), false));
    }
    Ok(("generic_json".into(), false))
}
