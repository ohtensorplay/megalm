//! XML-style tool-call grammars with named parameters.
//!
//! These families look similar (`<tool_call>`-style wrappers) but disagree on
//! how the function name and parameters are encoded. One small state machine
//! handles all of them and reuses the schema-aware value coercion from the
//! DeepSeek parser.

use super::parameters::{arguments_object, ToolSchemas};
use super::{Result, Tool, ToolCallDelta, ToolParser, ToolParserError, ToolParserOutput};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum XmlStyle {
    /// `<function=name><parameter=key>value</parameter></function>`
    Qwen,
    /// `name` followed by `<arg_key>key</arg_key><arg_value>value</arg_value>`
    Glm { separator: Option<&'static str> },
    /// `<invoke name="name"><parameter name="key">value</parameter></invoke>`
    Minimax,
}

#[derive(Debug, Clone, Copy)]
pub struct XmlToolConfig {
    start_marker: &'static str,
    end_marker: &'static str,
    style: XmlStyle,
}

const QWEN_CODER_CONFIG: XmlToolConfig = XmlToolConfig {
    start_marker: "<tool_call>",
    end_marker: "</tool_call>",
    style: XmlStyle::Qwen,
};

const SEED_OSS_CONFIG: XmlToolConfig = XmlToolConfig {
    start_marker: "<seed:tool_call>",
    end_marker: "</seed:tool_call>",
    style: XmlStyle::Qwen,
};

const GLM45_CONFIG: XmlToolConfig = XmlToolConfig {
    start_marker: "<tool_call>",
    end_marker: "</tool_call>",
    style: XmlStyle::Glm { separator: None },
};

const GLM47_CONFIG: XmlToolConfig = XmlToolConfig {
    start_marker: "<tool_call>",
    end_marker: "</tool_call>",
    style: XmlStyle::Glm { separator: None },
};

const HY_V3_CONFIG: XmlToolConfig = XmlToolConfig {
    start_marker: "<tool_call>",
    end_marker: "</tool_call>",
    style: XmlStyle::Glm {
        separator: Some("<tool_sep>"),
    },
};

const MINIMAX_M2_CONFIG: XmlToolConfig = XmlToolConfig {
    start_marker: "<minimax:tool_call>",
    end_marker: "</minimax:tool_call>",
    style: XmlStyle::Minimax,
};

/// Incremental parser for XML-style named-parameter tool calls.
pub struct XmlToolParser {
    buffer: String,
    config: XmlToolConfig,
    schemas: ToolSchemas,
    emitted: usize,
    in_block: bool,
}

impl XmlToolParser {
    fn new(config: XmlToolConfig, tools: &[Tool]) -> Self {
        Self {
            buffer: String::new(),
            config,
            schemas: ToolSchemas::from_tools(tools),
            emitted: 0,
            in_block: false,
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
        let start = self.config.start_marker;
        let end = self.config.end_marker;
        loop {
            if self.in_block {
                match self.buffer.find(end) {
                    Some(position) => {
                        let block = self.buffer[..position].to_string();
                        let consumed = position + end.len();
                        self.in_block = false;
                        self.handle_block(&block, output);
                        self.buffer.drain(..consumed);
                        continue;
                    }
                    None => {
                        if final_flush {
                            let text = format!("{start}{}", self.buffer);
                            output.push_text(text);
                            self.buffer.clear();
                            self.in_block = false;
                        }
                        break;
                    }
                }
            }

            match self.buffer.find(start) {
                Some(position) => {
                    output.push_text(&self.buffer[..position]);
                    self.buffer.drain(..position + start.len());
                    self.in_block = true;
                }
                None => {
                    if final_flush {
                        let text = std::mem::take(&mut self.buffer);
                        output.push_text(text);
                    } else {
                        let safe = safe_text_len(&self.buffer, start);
                        output.push_text(&self.buffer[..safe]);
                        self.buffer.drain(..safe);
                    }
                    break;
                }
            }
        }
    }

    fn handle_block(&mut self, block: &str, output: &mut ToolParserOutput) {
        let parsed = match self.config.style {
            XmlStyle::Qwen => parse_qwen_block(block),
            XmlStyle::Glm { separator } => parse_glm_block(block, separator),
            XmlStyle::Minimax => parse_minimax_block(block),
        };
        match parsed {
            Some((name, pairs)) => {
                let values = pairs
                    .into_iter()
                    .map(|(key, value)| {
                        let converted =
                            self.schemas.convert_param(&name, &key, &value, false);
                        (key, converted)
                    })
                    .collect();
                match arguments_object(values) {
                    Ok(arguments) => self.emit(name, arguments, output),
                    Err(_) => output.push_text(self.block_as_text(block)),
                }
            }
            None => output.push_text(self.block_as_text(block)),
        }
    }

    fn block_as_text(&self, block: &str) -> String {
        format!("{}{}{}", self.config.start_marker, block, self.config.end_marker)
    }
}

impl ToolParser for XmlToolParser {
    fn create(_tools: &[Tool]) -> Result<Box<dyn ToolParser>> {
        Err(ToolParserError(
            "XmlToolParser must be constructed with a configuration".to_string(),
        ))
    }

    fn parse_into(&mut self, chunk: &str, output: &mut ToolParserOutput) -> Result<()> {
        if chunk.len() > 64 * 1024 * 1024 {
            return Err(ToolParserError("tool parser delta is too large".to_string()));
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

macro_rules! xml_parser {
    ($name:ident, $config:ident) => {
        pub struct $name(XmlToolParser);

        impl ToolParser for $name {
            fn create(tools: &[Tool]) -> Result<Box<dyn ToolParser>> {
                Ok(Box::new(Self(XmlToolParser::new($config, tools))))
            }

            fn parse_into(
                &mut self,
                chunk: &str,
                output: &mut ToolParserOutput,
            ) -> Result<()> {
                self.0.parse_into(chunk, output)
            }

            fn finish(&mut self) -> Result<ToolParserOutput> {
                self.0.finish()
            }

            fn reset(&mut self) -> String {
                self.0.reset()
            }
        }
    };
}

xml_parser!(Qwen3CoderToolParser, QWEN_CODER_CONFIG);
xml_parser!(SeedOssToolParser, SEED_OSS_CONFIG);
xml_parser!(Glm45ToolParser, GLM45_CONFIG);
xml_parser!(Glm47ToolParser, GLM47_CONFIG);
xml_parser!(HyV3ToolParser, HY_V3_CONFIG);
xml_parser!(MinimaxM2ToolParser, MINIMAX_M2_CONFIG);

/// Parse `<function=name><parameter=key>value</parameter>...</function>`.
fn parse_qwen_block(block: &str) -> Option<(String, Vec<(String, String)>)> {
    const FUNCTION_START: &str = "<function=";
    const FUNCTION_END: &str = "</function>";
    const PARAMETER_START: &str = "<parameter=";
    const PARAMETER_END: &str = "</parameter>";

    let after_start = block.find(FUNCTION_START)? + FUNCTION_START.len();
    let name_end = block[after_start..].find('>')? + after_start;
    let name = block[after_start..name_end].trim();
    let function_body_end = block[name_end..].find(FUNCTION_END)? + name_end;
    let body = &block[name_end + 1..function_body_end];
    if name.is_empty() {
        return None;
    }

    let mut pairs = Vec::new();
    let mut cursor = 0usize;
    while let Some(relative) = body[cursor..].find(PARAMETER_START) {
        let start = cursor + relative + PARAMETER_START.len();
        let key_end = body[start..].find('>')? + start;
        let key = body[start..key_end].trim();
        let value_end = body[key_end..].find(PARAMETER_END)? + key_end;
        let value = &body[key_end + 1..value_end];
        pairs.push((key.to_string(), trim_wrapping_newlines(value)));
        cursor = value_end + PARAMETER_END.len();
    }
    Some((name.to_string(), pairs))
}

/// Parse GLM/HY style `name` + `<arg_key>` / `<arg_value>` pairs.
fn parse_glm_block(
    block: &str,
    separator: Option<&str>,
) -> Option<(String, Vec<(String, String)>)> {
    const KEY_START: &str = "<arg_key>";
    const KEY_END: &str = "</arg_key>";
    const VALUE_START: &str = "<arg_value>";
    const VALUE_END: &str = "</arg_value>";

    let first_key = block.find(KEY_START)?;
    let mut name = block[..first_key].trim();
    if let Some(separator) = separator {
        name = name.strip_suffix(separator).unwrap_or(name).trim();
    }
    if name.is_empty() {
        return None;
    }

    let mut pairs = Vec::new();
    let mut cursor = first_key;
    while let Some(relative) = block[cursor..].find(KEY_START) {
        let key_start = cursor + relative + KEY_START.len();
        let key_end = block[key_start..].find(KEY_END)? + key_start;
        let key = &block[key_start..key_end];
        let value_relative = block[key_end..].find(VALUE_START)?;
        let value_start = key_end + value_relative + VALUE_START.len();
        let value_end = block[value_start..].find(VALUE_END)? + value_start;
        pairs.push((key.to_string(), block[value_start..value_end].to_string()));
        cursor = value_end + VALUE_END.len();
    }
    Some((name.to_string(), pairs))
}

/// Parse `<invoke name="..."><parameter name="...">value</parameter>`.
fn parse_minimax_block(block: &str) -> Option<(String, Vec<(String, String)>)> {
    const INVOKE_START: &str = "<invoke";
    const INVOKE_END: &str = "</invoke>";
    const PARAMETER_START: &str = "<parameter";
    const PARAMETER_END: &str = "</parameter>";

    let tag_end = block.find(INVOKE_START)? + block[block.find(INVOKE_START)?..].find('>')?;
    let name = parse_attr(&block[..=tag_end], "name")?;
    let body_start = tag_end + 1;
    let body_end = block[body_start..].find(INVOKE_END)? + body_start;
    let body = &block[body_start..body_end];

    let mut pairs = Vec::new();
    let mut cursor = 0usize;
    while let Some(relative) = body[cursor..].find(PARAMETER_START) {
        let start = cursor + relative;
        let tag_end_relative = body[start..].find('>')?;
        let tag_end = start + tag_end_relative;
        let key = parse_attr(&body[start..=tag_end], "name")?;
        let value_start = tag_end + 1;
        let value_end = body[value_start..].find(PARAMETER_END)? + value_start;
        pairs.push((key, body[value_start..value_end].to_string()));
        cursor = value_end + PARAMETER_END.len();
    }
    Some((name, pairs))
}

fn parse_attr(tag: &str, name: &str) -> Option<String> {
    let needle = format!("{name}=\"");
    let start = tag.find(&needle)? + needle.len();
    let end = tag[start..].find('"')? + start;
    Some(tag[start..end].to_string())
}

fn trim_wrapping_newlines(value: &str) -> String {
    let mut value = value;
    if let Some(rest) = value.strip_prefix("\r\n") {
        value = rest;
    } else if let Some(rest) = value.strip_prefix('\n') {
        value = rest;
    }
    if let Some(rest) = value.strip_suffix("\r\n") {
        value = rest;
    } else if let Some(rest) = value.strip_suffix('\n') {
        value = rest;
    }
    value.to_string()
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

    use super::*;
    use crate::parser::tool::ToolParserOutput;

    fn tools() -> Vec<Tool> {
        vec![Tool {
            name: "get_weather".to_string(),
            description: None,
            parameters: json!({
                "type": "object",
                "properties": {
                    "city": {"type": "string"},
                    "days": {"type": "integer"}
                }
            }),
            strict: None,
        }]
    }

    fn run(parser: &mut Box<dyn ToolParser>, chunks: &[&str]) -> ToolParserOutput {
        let mut output = ToolParserOutput::default();
        for chunk in chunks {
            parser.parse_into(chunk, &mut output).unwrap();
        }
        output.append(parser.finish().unwrap());
        output
    }

    #[test]
    fn qwen_coder_extracts_function_and_parameters() {
        let mut parser = Qwen3CoderToolParser::create(&tools()).unwrap();
        let output = run(
            &mut parser,
            &[
                "hi <tool_call>\n<function=get_weather>\n<parameter=city>SF</parameter>\n",
                "<parameter=days>3</parameter>\n</function>\n</tool_call>",
            ],
        );
        assert_eq!(output.normal_text(), "hi ");
        assert_eq!(output.calls().len(), 1);
        assert_eq!(output.calls()[0].name.as_deref(), Some("get_weather"));
        let args: Value = serde_json::from_str(&output.calls()[0].arguments).unwrap();
        assert_eq!(args, json!({"city": "SF", "days": 3}));
    }

    #[test]
    fn glm_extracts_name_and_arg_pairs() {
        let mut parser = Glm47ToolParser::create(&tools()).unwrap();
        let output = run(
            &mut parser,
            &["<tool_call>get_weather\n<arg_key>city</arg_key><arg_value>SF</arg_value>\n</tool_call>"],
        );
        assert_eq!(output.calls().len(), 1);
        assert_eq!(output.calls()[0].name.as_deref(), Some("get_weather"));
        assert_eq!(
            serde_json::from_str::<Value>(&output.calls()[0].arguments).unwrap(),
            json!({"city": "SF"})
        );
    }

    #[test]
    fn hy3_uses_tool_separator() {
        let mut parser = HyV3ToolParser::create(&tools()).unwrap();
        let output = run(
            &mut parser,
            &["<tool_call>get_weather<tool_sep>\n<arg_key>city</arg_key><arg_value>SF</arg_value>\n</tool_call>"],
        );
        assert_eq!(output.calls().len(), 1);
        assert_eq!(output.calls()[0].name.as_deref(), Some("get_weather"));
    }

    #[test]
    fn minimax_m2_extracts_invoke_parameters() {
        let mut parser = MinimaxM2ToolParser::create(&tools()).unwrap();
        let output = run(
            &mut parser,
            &["<minimax:tool_call><invoke name=\"get_weather\">\n<parameter name=\"city\">SF</parameter>\n</invoke></minimax:tool_call>"],
        );
        assert_eq!(output.calls().len(), 1);
        assert_eq!(output.calls()[0].name.as_deref(), Some("get_weather"));
        assert_eq!(
            serde_json::from_str::<Value>(&output.calls()[0].arguments).unwrap(),
            json!({"city": "SF"})
        );
    }

    #[test]
    fn malformed_block_stays_visible() {
        let mut parser = Qwen3CoderToolParser::create(&tools()).unwrap();
        let text = "<tool_call><function=get_weather></tool_call>";
        let output = run(&mut parser, &[text]);
        assert!(output.calls().is_empty());
        assert_eq!(output.normal_text(), text);
    }
}
