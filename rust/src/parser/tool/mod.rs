//! Model-family tool-call parsers.
//!
//! The grammar and per-delta state live here, in Rust, so every Gateway
//! replica shares one implementation and a malformed model output can never
//! abort the Python process. Python keeps the OpenAI protocol adaptation and
//! the chat-template side; this module only answers "what text and which tool
//! calls did this decoded delta produce?".

pub mod parameters;

mod deepseek;
mod json;
mod kimi;
mod xml;

use std::collections::{btree_map, BTreeMap};
use std::fmt;

use serde_json::Value;

/// One completed tool invocation extracted from generated text.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ToolCall {
    pub id: String,
    pub name: String,
    pub arguments: String,
}

/// One function-style tool made available to the model.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct Tool {
    pub name: String,
    pub description: Option<String>,
    pub parameters: Value,
    pub strict: Option<bool>,
}

impl Tool {
    /// Build schema-less tools from the names alone.
    ///
    /// The gateway usually has the full parameter schema. This constructor is
    /// for the compatibility path and for tests that only need name gating.
    pub fn from_names(names: &[String]) -> Vec<Self> {
        names
            .iter()
            .map(|name| Self {
                name: name.clone(),
                description: None,
                parameters: Value::Object(Default::default()),
                strict: None,
            })
            .collect()
    }
}

/// One tool-call update emitted while parsing assistant text.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ToolCallDelta {
    /// Stable parser-local tool index for this call within one assistant turn.
    pub tool_index: usize,
    /// Function name, present on the first update for one tool call.
    pub name: Option<String>,
    /// Arguments text contributed by this update.
    pub arguments: String,
}

/// One ordered event emitted while parsing assistant text.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ToolParserEvent {
    /// Plain assistant text that is not part of any tool call.
    Text(String),
    /// A tool-call update extracted from assistant text.
    ToolCall(ToolCallDelta),
}

/// Result of advancing tool parsing with one assistant-text input.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct ToolParserOutput {
    /// Ordered parser events committed by this input.
    pub events: Vec<ToolParserEvent>,
}

impl ToolParserOutput {
    /// Append one visible text event if `text` is non-empty.
    pub fn push_text(&mut self, text: impl AsRef<str> + Into<String>) {
        if text.as_ref().is_empty() {
            return;
        }
        if let Some(ToolParserEvent::Text(last_text)) = self.events.last_mut() {
            last_text.push_str(text.as_ref());
            return;
        }
        self.events.push(ToolParserEvent::Text(text.into()));
    }

    /// Append one tool-call update event.
    pub fn push_call(&mut self, call: ToolCallDelta) {
        self.events.push(ToolParserEvent::ToolCall(call));
    }

    /// Return all plain assistant text committed by this output.
    pub fn normal_text(&self) -> String {
        self.events
            .iter()
            .filter_map(|event| match event {
                ToolParserEvent::Text(text) => Some(text.as_str()),
                ToolParserEvent::ToolCall(_) => None,
            })
            .collect()
    }

    /// Return all tool-call updates committed by this output.
    pub fn calls(&self) -> Vec<&ToolCallDelta> {
        self.events
            .iter()
            .filter_map(|event| match event {
                ToolParserEvent::Text(_) => None,
                ToolParserEvent::ToolCall(call) => Some(call),
            })
            .collect()
    }

    /// Append another parser output onto this one.
    pub fn append(&mut self, other: Self) {
        for event in other.events {
            match event {
                ToolParserEvent::Text(text) => self.push_text(text),
                ToolParserEvent::ToolCall(call) => self.push_call(call),
            }
        }
    }

    /// Flatten text and merge argument fragments for the same tool call.
    pub fn coalesce(self) -> Self {
        let mut merged = BTreeMap::<usize, ToolCallDelta>::new();
        let mut order = Vec::new();
        let normal_text = self.normal_text();

        for call in self.calls() {
            match merged.entry(call.tool_index) {
                btree_map::Entry::Vacant(entry) => {
                    order.push(call.tool_index);
                    entry.insert(call.clone());
                }
                btree_map::Entry::Occupied(mut entry) => {
                    let existing = entry.get_mut();
                    if existing.name.is_none() {
                        existing.name = call.name.clone();
                    }
                    existing.arguments.push_str(&call.arguments);
                }
            }
        }

        let mut output = Self::default();
        output.push_text(normal_text);
        for call in order.into_iter().filter_map(|index| merged.remove(&index)) {
            output.push_call(call);
        }
        output
    }
}

/// Error returned by a tool parser.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ToolParserError(pub String);

impl fmt::Display for ToolParserError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(&self.0)
    }
}

impl std::error::Error for ToolParserError {}

pub type Result<T> = std::result::Result<T, ToolParserError>;

/// Incremental parser that extracts tool calls from assistant output.
pub trait ToolParser: Send {
    /// Construct a parser instance for one request stream.
    fn create(tools: &[Tool]) -> Result<Box<dyn ToolParser>>
    where
        Self: Sized + 'static;

    /// Whether decoded output must preserve tokenizer special tokens.
    fn preserve_special_tokens(&self) -> bool {
        false
    }

    /// Model-emitted tool-call id for one call, when the model provides one.
    fn tool_call_id(&self, _tool_index: usize) -> Option<&str> {
        None
    }

    /// Feed one decoded delta, appending committed output into `output`.
    fn parse_into(&mut self, chunk: &str, output: &mut ToolParserOutput) -> Result<()>;

    /// Flush buffered partial state at end of stream.
    fn finish(&mut self) -> Result<ToolParserOutput>;

    /// Clear state and return currently uncommitted buffered text.
    fn reset(&mut self) -> String;
}

/// Construct a parser by registry name.
pub fn create(name: &str, tools: &[Tool]) -> Result<Box<dyn ToolParser>> {
    match name {
        "" | "generic_json" | "json" | "none" => {
            json::GenericJsonToolParser::create(tools)
        }
        "hermes" | "granite4" | "granite-4" => json::HermesToolParser::create(tools),
        "qwen3_xml" | "qwen" => json::Qwen3XmlToolParser::create(tools),
        "qwen3_coder" | "qwen-coder" | "nemotron_v3" | "nemotron-v3" => {
            xml::Qwen3CoderToolParser::create(tools)
        }
        "seed_oss" | "seed-oss" => xml::SeedOssToolParser::create(tools),
        "glm45" | "glm-4.5" | "glm-4.6" => xml::Glm45ToolParser::create(tools),
        "glm47" | "glm-4.7" | "glm-5" => xml::Glm47ToolParser::create(tools),
        "hy_v3" | "hy3" | "hy4" => xml::HyV3ToolParser::create(tools),
        "minimax_m2" | "minimax-m2" => xml::MinimaxM2ToolParser::create(tools),
        "kimi_k2" | "kimi-k2" => kimi::KimiK2ToolParser::create(tools),
        "mistral" | "mixtral" => json::MistralToolParser::create(tools),
        "phi4mini" | "phi-4-mini" => json::Phi4MiniJsonToolParser::create(tools),
        "internlm2" | "internlm" => json::Internlm2ToolParser::create(tools),
        "llama3_json" | "llama4_json" | "pythonic" => {
            json::Llama3JsonToolParser::create(tools)
        }
        "deepseek_v32" | "deepseek-v3.2" | "deepseek_v3.2" => {
            deepseek::DeepSeekV32ToolParser::create(tools)
        }
        "deepseek_v4" | "deepseek-v4" => {
            deepseek::DeepSeekV4ToolParser::create(tools)
        }
        other => Err(ToolParserError(format!(
            "unsupported tool parser `{other}`"
        ))),
    }
}

/// Names accepted by [`create`].
pub const PARSER_NAMES: &[&str] = &[
    "generic_json",
    "hermes",
    "granite4",
    "qwen3_xml",
    "qwen3_coder",
    "seed_oss",
    "glm45",
    "glm47",
    "hy_v3",
    "minimax_m2",
    "kimi_k2",
    "nemotron_v3",
    "mistral",
    "phi4mini",
    "internlm2",
    "llama3_json",
    "llama4_json",
    "deepseek_v32",
    "deepseek_v4",
];

/// Whether a registry name needs special tokens preserved by the decoder.
pub fn preserve_special_tokens(name: &str) -> bool {
    matches!(
        name,
        "deepseek_v32"
            | "deepseek-v3.2"
            | "deepseek_v3.2"
            | "deepseek_v4"
            | "deepseek-v4"
            | "internlm2"
            | "internlm"
            | "kimi_k2"
            | "kimi-k2"
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn unknown_parser_is_rejected() {
        match create("does_not_exist", &[]) {
            Err(error) => assert!(error.to_string().contains("unsupported tool parser")),
            Ok(_) => panic!("unknown parser must be rejected"),
        }
    }

    #[test]
    fn parser_names_construct() {
        for name in PARSER_NAMES {
            let parser = create(name, &Tool::from_names(&["f".to_string()])).unwrap();
            assert_eq!(
                parser.preserve_special_tokens(),
                preserve_special_tokens(name)
            );
        }
    }

    #[test]
    fn coalesce_merges_argument_fragments() {
        let mut output = ToolParserOutput::default();
        output.push_text("a");
        output.push_call(ToolCallDelta {
            tool_index: 0,
            name: Some("f".to_string()),
            arguments: "{\"x\":".to_string(),
        });
        output.push_call(ToolCallDelta {
            tool_index: 0,
            name: None,
            arguments: "1}".to_string(),
        });
        output.push_text("b");
        let merged = output.coalesce();
        assert_eq!(merged.normal_text(), "ab");
        assert_eq!(merged.calls().len(), 1);
        assert_eq!(merged.calls()[0].arguments, "{\"x\":1}");
    }
}