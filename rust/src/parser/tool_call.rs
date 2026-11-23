//! Backward-compatible wrapper over the generic JSON tool parser.
//!
//! New code should use [`crate::parser::tool`] directly. This wrapper keeps the
//! original `(visible_text, calls)` API used by `OutputParser` and the Python
//! binding while the model-family parsers live in the `tool` module.

use super::tool::{create, Tool, ToolParser, ToolParserOutput};
use super::{ParseError, ToolCall};

/// Streaming tool-call parser over decoded text.
pub struct ToolCallParser {
    id_prefix: String,
    inner: Box<dyn ToolParser>,
}

impl ToolCallParser {
    pub fn new(id_prefix: impl Into<String>, tool_names: Vec<String>) -> Self {
        let tools = Tool::from_names(&tool_names);
        let inner = create("generic_json", &tools)
            .expect("generic JSON tool parser must always construct");
        Self {
            id_prefix: id_prefix.into(),
            inner,
        }
    }

    /// Feed one decoded delta. Returns `(visible_text, completed_calls)`.
    pub fn push(&mut self, delta: &str) -> (String, Vec<ToolCall>) {
        let mut output = ToolParserOutput::default();
        self.inner
            .parse_into(delta, &mut output)
            .expect("generic JSON tool parser must not fail");
        self.collect(output)
    }

    /// Flush the buffer at end of generation. Returns any final text/calls.
    pub fn finish(&mut self) -> (String, Vec<ToolCall>) {
        let output = self
            .inner
            .finish()
            .expect("generic JSON tool parser must not fail");
        self.collect(output)
    }

    fn collect(&self, output: ToolParserOutput) -> (String, Vec<ToolCall>) {
        let text = output.normal_text();
        let calls = output
            .calls()
            .into_iter()
            .filter_map(|call| {
                call.name.as_ref().map(|name| ToolCall {
                    id: format!("{}_{}", self.id_prefix, call.tool_index),
                    name: name.clone(),
                    arguments: call.arguments.clone(),
                })
            })
            .collect();
        (text, calls)
    }

    /// The generic JSON parser is lexical and does not emit structured
    /// diagnostics; this keeps the `OutputParser` telemetry hook total.
    pub fn take_diagnostics(&mut self) -> Vec<ParseError> {
        Vec::new()
    }
}

#[cfg(test)]
mod tool_call_tests {
    use super::ToolCallParser;

    #[test]
    fn extracts_a_complete_call_and_hides_its_json() {
        let mut parser = ToolCallParser::new("call", vec![]);
        let (visible, calls) = parser.push(
            "Let me check. <tool_call>{\"name\":\"get_weather\",\"arguments\":{\"city\":\"SF\"}}</tool_call>",
        );
        assert!(visible.contains("Let me check."));
        assert!(!visible.contains("get_weather"));
        assert!(!visible.contains('{'));
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].id, "call_0");
        assert_eq!(calls[0].name, "get_weather");
        assert_eq!(calls[0].arguments, "{\"city\":\"SF\"}");
    }

    #[test]
    fn buffers_an_object_split_across_deltas() {
        let mut parser = ToolCallParser::new("call", vec![]);
        assert_eq!(parser.push("x{\"name\":\"f\",\"arg").0, "x");
        let (visible, calls) = parser.push("s\":{\"k\":1}}done");
        assert_eq!(visible, "done");
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].id, "call_0");
        assert_eq!(calls[0].arguments, "{\"k\":1}");
    }

    #[test]
    fn supports_multiple_calls_with_incrementing_ids() {
        let mut parser = ToolCallParser::new("c", vec![]);
        let (_, calls) = parser.push(
            "{\"name\":\"a\",\"args\":{}}{\"name\":\"b\",\"args\":{\"v\":2}}",
        );
        assert_eq!(calls.len(), 2);
        assert_eq!(calls[0].id, "c_0");
        assert_eq!(calls[1].id, "c_1");
        assert_eq!(calls[1].name, "b");
        assert_eq!(calls[1].arguments, "{\"v\":2}");
    }

    #[test]
    fn keeps_non_tool_json_as_visible_text() {
        let mut parser = ToolCallParser::new("call", vec![]);
        let (visible, calls) = parser.push("answer {\"value\":42} end");
        assert!(calls.is_empty());
        assert_eq!(visible, "answer {\"value\":42} end");
    }

    #[test]
    fn arguments_string_is_passed_through_verbatim() {
        let mut parser = ToolCallParser::new("call", vec![]);
        let (_, calls) = parser.push(r#"{"name":"f","arguments":"{\"x\":1}"}"#);
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].arguments, "{\"x\":1}");
    }

    #[test]
    fn object_arguments_are_not_reserialized() {
        let mut parser = ToolCallParser::new("call", vec![]);
        let (_, calls) = parser.push(r#"{"name": "f", "args": { "z" : 1, "a" : [2, 3] }}"#);
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].arguments, r#"{ "z" : 1, "a" : [2, 3] }"#);
    }

    #[test]
    fn accepts_malformed_but_balanced_arguments() {
        let mut parser = ToolCallParser::new("call", vec![]);
        let (visible, calls) = parser.push(r#"{"name":"f","args":{"x":1,}}"#);
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].arguments, r#"{"x":1,}"#);
        assert!(visible.is_empty());
    }

    #[test]
    fn field_order_and_extra_keys_are_tolerated() {
        let mut parser = ToolCallParser::new("call", vec![]);
        let (_, calls) = parser.push(
            r#"{"args":{"b":2},"description":"ignored","name":"say_\"hi\""}"#,
        );
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].name, "say_\"hi\"");
        assert_eq!(calls[0].arguments, r#"{"b":2}"#);
    }

    #[test]
    fn finish_flushes_unterminated_object_as_text() {
        let mut parser = ToolCallParser::new("call", vec![]);
        assert_eq!(parser.push("tail{\"name\":\"f\"").0, "tail");
        let (visible, calls) = parser.finish();
        assert!(calls.is_empty());
        assert_eq!(visible, "{\"name\":\"f\"");
    }

    #[test]
    fn trims_bare_function_name_header() {
        let mut parser = ToolCallParser::new(
            "call",
            vec!["get_weather".to_string(), "other".to_string()],
        );
        let (visible, calls) =
            parser.push("get_weather{\"name\":\"get_weather\",\"args\":{}}");
        assert_eq!(visible, "");
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].name, "get_weather");
    }

    #[test]
    fn keeps_name_like_text_when_not_a_tool_header() {
        let mut parser = ToolCallParser::new("call", vec!["other".to_string()]);
        let (visible, calls) =
            parser.push("use get_weather{\"name\":\"get_weather\",\"args\":{}}");
        assert!(visible.contains("get_weather"));
        assert_eq!(calls.len(), 1);
    }

    #[test]
    fn ignores_braces_inside_strings() {
        let mut parser = ToolCallParser::new("call", vec![]);
        let (visible, calls) = parser.push("say {not json} {\"name\":\"f\",\"args\":{\"s\":\"}\"}}");
        assert!(calls.len() == 1);
        assert_eq!(calls[0].name, "f");
        assert_eq!(calls[0].arguments, "{\"s\":\"}\"}");
        assert_eq!(visible, "say {not json} ");
    }
}