//! DeepSeek V3.2 DSML tool calls.
//!
//! V3.2 wraps calls in `‹dsml›function_calls` / `‹/dsml›function_calls`
//! blocks and emits one `invoke` per call. Parameter values are raw text with
//! an explicit `string="true|false"` attribute; the parser converts
//! non-string values through the request's tool schema.

use super::parameters::{arguments_object, ToolSchemas};
use super::{
    Result, Tool, ToolCallDelta, ToolParser, ToolParserError, ToolParserOutput,
};

const DSML: &str = "\u{ff5c}DSML\u{ff5c}";
const TOOL_CALLS_START_PREFIX: &str = "<";
const V32_CALLS_START_SUFFIX: &str = "function_calls>";
const V32_CALLS_END_SUFFIX: &str = "function_calls>";
const V4_CALLS_START_SUFFIX: &str = "tool_calls>";
const V4_CALLS_END_SUFFIX: &str = "tool_calls>";
const TOOL_CALLS_END_PREFIX: &str = "</";
const INVOKE_START_PREFIX: &str = "<";
const INVOKE_START_SUFFIX: &str = "invoke";
const INVOKE_END_PREFIX: &str = "</";
const INVOKE_END_SUFFIX: &str = "invoke>";
const PARAMETER_START_PREFIX: &str = "<";
const PARAMETER_START_SUFFIX: &str = "parameter";
const PARAMETER_END_PREFIX: &str = "</";
const PARAMETER_END_SUFFIX: &str = "parameter>";

fn marker(prefix: &str, suffix: &str) -> String {
    format!("{prefix}{DSML}{suffix}")
}

/// Tool parser for DeepSeek V3.2 DSML output.
pub struct DeepSeekV32ToolParser {
    buffer: String,
    schemas: ToolSchemas,
    emitted: usize,
    in_block: bool,
    calls_start: String,
    calls_end: String,
    invoke_start: String,
    invoke_end: String,
    parameter_start: String,
    parameter_end: String,
}

impl DeepSeekV32ToolParser {
    fn new(tools: &[Tool]) -> Self {
        Self::new_with_tokens(
            tools,
            V32_CALLS_START_SUFFIX,
            V32_CALLS_END_SUFFIX,
        )
    }

    fn new_with_tokens(
        tools: &[Tool],
        start_suffix: &str,
        end_suffix: &str,
    ) -> Self {
        Self {
            buffer: String::new(),
            schemas: ToolSchemas::from_tools(tools),
            emitted: 0,
            in_block: false,
            calls_start: marker(TOOL_CALLS_START_PREFIX, start_suffix),
            calls_end: marker(TOOL_CALLS_END_PREFIX, end_suffix),
            invoke_start: marker(INVOKE_START_PREFIX, INVOKE_START_SUFFIX),
            invoke_end: marker(INVOKE_END_PREFIX, INVOKE_END_SUFFIX),
            parameter_start: marker(PARAMETER_START_PREFIX, PARAMETER_START_SUFFIX),
            parameter_end: marker(PARAMETER_END_PREFIX, PARAMETER_END_SUFFIX),
        }
    }

    fn emit(&mut self, name: String, arguments: String, output: &mut ToolParserOutput) {
        output.push_call(ToolCallDelta {
            tool_index: self.emitted,
            name: Some(name),
            arguments,
        });
        self.emitted += 1;
    }

    fn drain(&mut self, output: &mut ToolParserOutput, final_flush: bool) {
        loop {
            if self.in_block {
                match self.buffer.find(&self.calls_end) {
                    Some(position) => {
                        let block = self.buffer[..position].to_string();
                        let consumed = position + self.calls_end.len();
                        self.in_block = false;
                        self.handle_calls(&block, output);
                        self.buffer.drain(..consumed);
                        continue;
                    }
                    None => {
                        if final_flush {
                            let text = format!("{}{}", self.calls_start, self.buffer);
                            output.push_text(text);
                            self.buffer.clear();
                            self.in_block = false;
                        }
                        break;
                    }
                }
            }

            match self.buffer.find(&self.calls_start) {
                Some(position) => {
                    output.push_text(&self.buffer[..position]);
                    self.buffer
                        .drain(..position + self.calls_start.len());
                    self.in_block = true;
                }
                None => {
                    if final_flush {
                        let text = std::mem::take(&mut self.buffer);
                        output.push_text(text);
                    } else {
                        let safe = safe_text_len(&self.buffer, &self.calls_start);
                        output.push_text(&self.buffer[..safe]);
                        self.buffer.drain(..safe);
                    }
                    break;
                }
            }
        }
    }

    fn handle_calls(&mut self, block: &str, output: &mut ToolParserOutput) {
        let mut cursor = 0usize;
        loop {
            let Some(relative) = block[cursor..].find(&self.invoke_start) else {
                let tail = block[cursor..].trim();
                if !tail.is_empty() {
                    output.push_text(tail);
                }
                break;
            };
            let invoke_start = cursor + relative;
            let gap = block[cursor..invoke_start].trim();
            if !gap.is_empty() {
                output.push_text(gap);
            }
            let Some(tag_end_relative) = block[invoke_start..].find('>') else {
                output.push_text(format!("{}{}", self.calls_start, block));
                return;
            };
            let tag_end = invoke_start + tag_end_relative;
            let Some(body_end_relative) = block[tag_end + 1..].find(&self.invoke_end) else {
                output.push_text(format!("{}{}", self.calls_start, block));
                return;
            };
            let body_end = tag_end + 1 + body_end_relative;
            let tag = &block[invoke_start..=tag_end];
            let body = &block[tag_end + 1..body_end];
            match self.parse_invoke(tag, body) {
                Some((name, arguments)) => self.emit(name, arguments, output),
                None => {
                    output.push_text(format!("{}{}", self.calls_start, block));
                    return;
                }
            }
            cursor = body_end + self.invoke_end.len();
        }
    }

    fn parse_invoke(&self, tag: &str, body: &str) -> Option<(String, String)> {
        let name = parse_attr(tag, "name")?;
        if name.is_empty() {
            return None;
        }
        let mut pairs = Vec::new();
        let mut cursor = 0usize;
        while let Some(relative) = body[cursor..].find(&self.parameter_start) {
            let parameter_start = cursor + relative;
            let tag_end_relative = body[parameter_start..].find('>')?;
            let tag_end = parameter_start + tag_end_relative;
            let parameter_tag = &body[parameter_start..=tag_end];
            let value_start = tag_end + 1;
            let value_end_relative = body[value_start..].find(&self.parameter_end)?;
            let value_end = value_start + value_end_relative;
            let value = &body[value_start..value_end];
            let Some(parameter_name) = parse_attr(parameter_tag, "name") else {
                return None;
            };
            let force_string = parse_attr(parameter_tag, "string")
                .map(|value| value.eq_ignore_ascii_case("true"))
                .unwrap_or(false);
            let converted =
                self.schemas
                    .convert_param(&name, &parameter_name, value, force_string);
            pairs.push((parameter_name, converted));
            cursor = value_end + self.parameter_end.len();
        }
        let arguments = arguments_object(pairs).ok()?;
        Some((name, arguments))
    }
}

impl ToolParser for DeepSeekV32ToolParser {
    fn create(tools: &[Tool]) -> Result<Box<dyn ToolParser>> {
        Ok(Box::new(Self::new(tools)))
    }

    fn preserve_special_tokens(&self) -> bool {
        true
    }

    fn parse_into(&mut self, chunk: &str, output: &mut ToolParserOutput) -> Result<()> {
        if chunk.is_empty() {
            return Ok(());
        }
        if chunk.len() > 64 * 1024 * 1024 {
            return Err(ToolParserError(
                "tool parser delta is too large".to_string(),
            ));
        }
        self.buffer.push_str(chunk);
        self.drain(output, false);
        Ok(())
    }

    fn finish(&mut self) -> Result<ToolParserOutput> {
        let mut output = ToolParserOutput::default();
        self.drain(&mut output, true);
        Ok(output)
    }

    fn reset(&mut self) -> String {
        self.in_block = false;
        self.emitted = 0;
        std::mem::take(&mut self.buffer)
    }
}

/// Tool parser for DeepSeek V4 DSML output (same grammar, `tool_calls` wrapper).
pub struct DeepSeekV4ToolParser(DeepSeekV32ToolParser);

impl ToolParser for DeepSeekV4ToolParser {
    fn create(tools: &[Tool]) -> Result<Box<dyn ToolParser>> {
        Ok(Box::new(Self(DeepSeekV32ToolParser::new_with_tokens(
            tools,
            V4_CALLS_START_SUFFIX,
            V4_CALLS_END_SUFFIX,
        ))))
    }

    fn preserve_special_tokens(&self) -> bool {
        true
    }

    fn parse_into(&mut self, chunk: &str, output: &mut ToolParserOutput) -> Result<()> {
        self.0.parse_into(chunk, output)
    }

    fn finish(&mut self) -> Result<ToolParserOutput> {
        self.0.finish()
    }

    fn reset(&mut self) -> String {
        self.0.reset()
    }
}

/// Parse one attribute from a tag, accepting single or double quotes.
fn parse_attr(tag: &str, name: &str) -> Option<String> {
    let mut from = 0usize;
    loop {
        let relative = tag[from..].find(name)?;
        let start = from + relative;
        let before_ok = start == 0
            || tag[..start]
                .chars()
                .next_back()
                .map(|ch| ch.is_whitespace() || ch == '<' || ch == '|')
                .unwrap_or(true);
        if !before_ok {
            from = start + name.len();
            continue;
        }
        let after = &tag[start + name.len()..];
        let after = after.trim_start();
        let Some(after) = after.strip_prefix('=') else {
            from = start + name.len();
            continue;
        };
        let after = after.trim_start();
        let quote = after.chars().next()?;
        if quote != '"' && quote != '\'' {
            from = start + name.len();
            continue;
        }
        let rest = &after[quote.len_utf8()..];
        let end = rest.find(quote)?;
        return Some(rest[..end].to_string());
    }
}

fn safe_text_len(buffer: &str, marker: &str) -> usize {
    let max = buffer.len().min(marker.len().saturating_sub(1));
    for keep in (1..=max).rev() {
        let split = buffer.len() - keep;
        if buffer.is_char_boundary(split) && marker.starts_with(&buffer[split..]) {
            return split;
        }
    }
    buffer.len()
}

#[cfg(test)]
mod tests {
    use serde_json::{json, Value};

    use super::{DeepSeekV32ToolParser, DeepSeekV4ToolParser};
    use crate::parser::tool::{Tool, ToolParser, ToolParserOutput};

    const D: &str = "\u{ff5c}DSML\u{ff5c}";

    fn tools() -> Vec<Tool> {
        vec![Tool {
            name: "get_weather".to_string(),
            description: None,
            parameters: json!({
                "type": "object",
                "properties": {
                    "location": {"type": "string"},
                    "days": {"type": "integer"},
                    "flag": {"type": "boolean"}
                }
            }),
            strict: None,
        }]
    }

    fn invoke(params: &[(&str, &str, bool)]) -> String {
        let params = params
            .iter()
            .map(|(name, value, is_string)| {
                format!(
                    "<{D}parameter name=\"{name}\" string=\"{is_string}\">{value}</{D}parameter>"
                )
            })
            .collect::<Vec<_>>()
            .join("\n");
        format!("<{D}invoke name=\"get_weather\">\n{params}\n</{D}invoke>")
    }

    fn call(params: &[(&str, &str, bool)]) -> String {
        format!("<{D}function_calls>\n{}\n</{D}function_calls>", invoke(params))
    }

    fn run(chunks: &[String]) -> ToolParserOutput {
        let mut parser = DeepSeekV32ToolParser::create(&tools()).unwrap();
        let mut output = ToolParserOutput::default();
        for chunk in chunks {
            parser.parse_into(chunk, &mut output).unwrap();
        }
        output.append(parser.finish().unwrap());
        output
    }

    #[test]
    fn parses_complete_call_and_converts_schema_types() {
        let text = call(&[
            ("location", "SF", true),
            ("days", "3", false),
            ("flag", "true", false),
        ]);
        let output = run(&[text]);
        assert!(output.normal_text().is_empty());
        assert_eq!(output.calls().len(), 1);
        assert_eq!(output.calls()[0].name.as_deref(), Some("get_weather"));
        let arguments: Value = serde_json::from_str(&output.calls()[0].arguments).unwrap();
        assert_eq!(arguments, json!({"location": "SF", "days": 3, "flag": true}));
    }

    #[test]
    fn string_attribute_overrides_schema() {
        let text = call(&[("days", "3", true)]);
        let output = run(&[text]);
        let arguments: Value = serde_json::from_str(&output.calls()[0].arguments).unwrap();
        assert_eq!(arguments, json!({"days": "3"}));
    }

    #[test]
    fn split_across_every_char_boundary() {
        let text = format!("Thinking... {}", call(&[("location", "NYC", true)]));
        let mut parser = DeepSeekV32ToolParser::create(&tools()).unwrap();
        let mut streamed = ToolParserOutput::default();
        let mut start = 0usize;
        for (index, _) in text.char_indices().skip(1) {
            parser
                .parse_into(&text[start..index], &mut streamed)
                .unwrap();
            start = index;
        }
        parser.parse_into(&text[start..], &mut streamed).unwrap();
        streamed.append(parser.finish().unwrap());
        assert!(streamed.normal_text().contains("Thinking..."));
        assert_eq!(streamed.calls().len(), 1);
    }

    #[test]
    fn parallel_invokes_get_incrementing_indices() {
        let merged = format!(
            "<{D}function_calls>\n{}\n{}\n</{D}function_calls>",
            invoke(&[("location", "SF", true)]),
            invoke(&[("location", "NYC", true)]),
        );
        let output = run(&[merged]);
        assert_eq!(output.calls().len(), 2);
        assert_eq!(output.calls()[0].tool_index, 0);
        assert_eq!(output.calls()[1].tool_index, 1);
    }

    #[test]
    fn deepseek_v4_uses_tool_calls_wrapper() {
        let text = call(&[("location", "SF", true)])
            .replace("function_calls>", "tool_calls>");
        let mut parser = DeepSeekV4ToolParser::create(&tools()).unwrap();
        let mut output = ToolParserOutput::default();
        parser.parse_into(&text, &mut output).unwrap();
        output.append(parser.finish().unwrap());
        assert!(output.normal_text().is_empty());
        assert_eq!(output.calls().len(), 1);
        assert_eq!(output.calls()[0].name.as_deref(), Some("get_weather"));
    }

    #[test]
    fn incomplete_block_is_flushed_as_text() {
        let text = format!("<{D}function_calls>\n<{D}invoke name=\"get_weather\">");
        let output = run(&[text]);
        assert!(output.calls().is_empty());
        assert!(output.normal_text().contains("get_weather"));
    }
}