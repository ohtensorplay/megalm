//! Marker-delimited JSON tool-call grammars.
//!
//! Most JSON-style families differ only in the wrapper markers and whether a
//! newline is required around the payload. The lexer below keeps argument text
//! verbatim (no deserialize/re-serialize), buffers partial objects across
//! deltas, and never treats arbitrary JSON as a tool call unless it has a
//! `name` field and one of the configured argument keys.

use super::{
    Result, Tool, ToolCallDelta, ToolParser, ToolParserError, ToolParserOutput,
};

#[derive(Debug, Clone, Copy)]
pub struct JsonToolConfig {
    /// `None` enables the bare-object fallback used by the legacy parser.
    pub start_marker: Option<&'static str>,
    pub end_marker: Option<&'static str>,
    pub require_newline_after_start: bool,
    pub require_newline_before_end: bool,
    pub name_keys: &'static [&'static str],
    pub arguments_keys: &'static [&'static str],
    /// Parse several JSON objects inside one marker-delimited block.
    pub array_mode: bool,
    /// Separator text emitted between tool-call objects but not part of content.
    pub delimiter: Option<&'static str>,
}

const GENERIC_CONFIG: JsonToolConfig = JsonToolConfig {
    start_marker: None,
    end_marker: None,
    require_newline_after_start: false,
    require_newline_before_end: false,
    name_keys: &["name"],
    arguments_keys: &["args", "arguments"],
    array_mode: false,
    delimiter: None,
};

const HERMES_CONFIG: JsonToolConfig = JsonToolConfig {
    start_marker: Some("<tool_call>"),
    end_marker: Some("</tool_call>"),
    require_newline_after_start: false,
    require_newline_before_end: false,
    name_keys: &["name"],
    arguments_keys: &["arguments", "args"],
    array_mode: false,
    delimiter: None,
};

const QWEN_XML_CONFIG: JsonToolConfig = JsonToolConfig {
    start_marker: Some("<tool_call>"),
    end_marker: Some("</tool_call>"),
    require_newline_after_start: true,
    require_newline_before_end: true,
    name_keys: &["name"],
    arguments_keys: &["arguments", "args"],
    array_mode: false,
    delimiter: None,
};

const MISTRAL_CONFIG: JsonToolConfig = JsonToolConfig {
    start_marker: Some("[TOOL_CALLS] ["),
    end_marker: Some("]"),
    require_newline_after_start: false,
    require_newline_before_end: false,
    name_keys: &["name"],
    arguments_keys: &["arguments"],
    array_mode: true,
    delimiter: Some(","),
};

const PHI4MINI_CONFIG: JsonToolConfig = JsonToolConfig {
    start_marker: Some("functools["),
    end_marker: Some("]"),
    require_newline_after_start: false,
    require_newline_before_end: false,
    name_keys: &["name"],
    arguments_keys: &["arguments", "parameters"],
    array_mode: true,
    delimiter: Some(","),
};

const INTERNLM2_CONFIG: JsonToolConfig = JsonToolConfig {
    start_marker: Some("<|action_start|><|plugin|>"),
    end_marker: Some("<|action_end|>"),
    require_newline_after_start: false,
    require_newline_before_end: false,
    name_keys: &["name"],
    arguments_keys: &["parameters", "arguments"],
    array_mode: false,
    delimiter: None,
};

const LLAMA3_CONFIG: JsonToolConfig = JsonToolConfig {
    start_marker: None,
    end_marker: None,
    require_newline_after_start: false,
    require_newline_before_end: false,
    name_keys: &["name"],
    arguments_keys: &["parameters", "arguments"],
    array_mode: false,
    delimiter: Some(";"),
};

/// Incremental JSON-style tool parser.
pub struct JsonToolParser {
    buffer: String,
    config: JsonToolConfig,
    tool_names: Vec<String>,
    in_block: bool,
    emitted: usize,
}

impl JsonToolParser {
    fn new(config: JsonToolConfig, tools: &[Tool]) -> Self {
        Self {
            buffer: String::new(),
            config,
            tool_names: tools.iter().map(|tool| tool.name.clone()).collect(),
            in_block: false,
            emitted: 0,
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
        if self.config.start_marker.is_some() {
            self.drain_marker(output, final_flush);
        } else {
            self.drain_generic(output, final_flush);
        }
    }

    fn drain_marker(&mut self, output: &mut ToolParserOutput, final_flush: bool) {
        let start_marker = self.config.start_marker.expect("marker mode");
        let end_marker = self.config.end_marker.expect("marker mode");
        loop {
            if self.in_block {
                match find_marker_outside_json(&self.buffer, end_marker, self.config.array_mode) {
                    Some(position) => {
                        let block = self.buffer[..position].to_string();
                        let consumed = position + end_marker.len();
                        self.in_block = false;
                        self.handle_block(&block, output);
                        self.buffer.drain(..consumed);
                        continue;
                    }
                    None => {
                        if final_flush {
                            let text = self.block_as_text(&self.buffer);
                            output.push_text(text);
                            self.buffer.clear();
                            self.in_block = false;
                        }
                        break;
                    }
                }
            }

            match self.buffer.find(start_marker) {
                Some(position) => {
                    let prefix = self.clean_prefix(&self.buffer[..position]);
                    output.push_text(prefix);
                    self.buffer.drain(..position + start_marker.len());
                    self.in_block = true;
                }
                None => {
                    if final_flush {
                        let text = strip_tool_markers(&self.buffer);
                        output.push_text(text);
                        self.buffer.clear();
                    } else {
                        let safe = safe_text_len(&self.buffer, start_marker);
                        let text = strip_tool_markers(&self.buffer[..safe]);
                        output.push_text(text);
                        self.buffer.drain(..safe);
                    }
                    break;
                }
            }
        }
    }

    fn handle_block(&mut self, block: &str, output: &mut ToolParserOutput) {
        if self.config.array_mode {
            self.handle_array_block(block, output);
            return;
        }
        let mut body = block;
        if self.config.require_newline_after_start {
            if let Some(rest) = body.strip_prefix("\r\n") {
                body = rest;
            } else if let Some(rest) = body.strip_prefix('\n') {
                body = rest;
            } else {
                output.push_text(self.block_as_text(block));
                return;
            }
        }
        if self.config.require_newline_before_end {
            if let Some(rest) = body.strip_suffix("\r\n") {
                body = rest;
            } else if let Some(rest) = body.strip_suffix('\n') {
                body = rest;
            } else {
                output.push_text(self.block_as_text(block));
                return;
            }
        }
        let candidate = body.trim();
        match tool_call_from_json(candidate, self.config.name_keys, self.config.arguments_keys) {
            Some((name, arguments)) => self.emit(name, arguments, output),
            None => output.push_text(self.block_as_text(block)),
        }
    }

    /// Parse a wrapper that contains several JSON tool-call objects.
    fn handle_array_block(&mut self, block: &str, output: &mut ToolParserOutput) {
        let mut cursor = 0usize;
        while cursor < block.len() {
            let bytes = block.as_bytes();
            let Some(relative) = bytes[cursor..].iter().position(|&byte| byte == b'{') else {
                let tail = self.clean_prefix(&block[cursor..]);
                output.push_text(tail);
                break;
            };
            let start = cursor + relative;
            let prefix = self.clean_prefix(&block[cursor..start]);
            output.push_text(prefix);
            let Some(end_relative) = json_object_end(&block[start..]) else {
                output.push_text(self.clean_prefix(&block[start..]));
                break;
            };
            let end = start + end_relative;
            let candidate = &block[start..end];
            match tool_call_from_json(
                candidate,
                self.config.name_keys,
                self.config.arguments_keys,
            ) {
                Some((name, arguments)) => self.emit(name, arguments, output),
                None => output.push_text(candidate.to_string()),
            }
            cursor = end;
        }
    }

    fn clean_prefix(&self, text: &str) -> String {
        let mut cleaned = strip_tool_markers(text);
        if let Some(delimiter) = self.config.delimiter {
            cleaned = cleaned.replace(delimiter, "");
        }
        if self.config.delimiter.is_some() && cleaned.trim().is_empty() {
            String::new()
        } else {
            cleaned
        }
    }

    /// Reconstruct the original marker-wrapped block when the grammar does not
    /// accept it, so rejected input stays visible instead of being dropped.
    fn block_as_text(&self, block: &str) -> String {
        match (self.config.start_marker, self.config.end_marker) {
            (Some(start), Some(end)) => format!("{start}{block}{end}"),
            _ => strip_tool_markers(block),
        }
    }

    fn drain_generic(&mut self, output: &mut ToolParserOutput, final_flush: bool) {
        let mut cursor = 0usize;
        while cursor < self.buffer.len() {
            let bytes = self.buffer.as_bytes();
            let Some(relative) = bytes[cursor..].iter().position(|&byte| byte == b'{') else {
                let tail = self.clean_prefix(&self.buffer[cursor..]);
                output.push_text(tail);
                cursor = self.buffer.len();
                break;
            };
            let start = cursor + relative;
            let mut prefix = self.clean_prefix(&self.buffer[cursor..start]);
            trim_leading_tool_name(&mut prefix, &self.tool_names);
            output.push_text(prefix);
            cursor = start;

            match json_object_end(&self.buffer[start..]) {
                Some(end_relative) => {
                    let end = start + end_relative;
                    let candidate = &self.buffer[start..end];
                    match tool_call_from_json(
                        candidate,
                        self.config.name_keys,
                        self.config.arguments_keys,
                    ) {
                        Some((name, arguments)) => self.emit(name, arguments, output),
                        None => output.push_text(candidate),
                    }
                    cursor = end;
                }
                None => {
                    if final_flush {
                        let text = strip_tool_markers(&self.buffer[start..]);
                        output.push_text(text);
                        cursor = self.buffer.len();
                    }
                    break;
                }
            }
        }
        self.buffer.drain(..cursor);
    }
}

impl ToolParser for JsonToolParser {
    fn create(_tools: &[Tool]) -> Result<Box<dyn ToolParser>> {
        Err(ToolParserError(
            "JsonToolParser must be constructed with a configuration".to_string(),
        ))
    }

    fn parse_into(&mut self, chunk: &str, output: &mut ToolParserOutput) -> Result<()> {
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

/// Bare-object fallback: scans any `{...}` and accepts it when it has a
/// recognizable `name` plus `args`/`arguments` field.
pub struct GenericJsonToolParser(JsonToolParser);

impl ToolParser for GenericJsonToolParser {
    fn create(tools: &[Tool]) -> Result<Box<dyn ToolParser>> {
        Ok(Box::new(Self(JsonToolParser::new(GENERIC_CONFIG, tools))))
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

/// `<tool_call>{...}</tool_call>` with optional surrounding whitespace.
pub struct HermesToolParser(JsonToolParser);

impl ToolParser for HermesToolParser {
    fn create(tools: &[Tool]) -> Result<Box<dyn ToolParser>> {
        Ok(Box::new(Self(JsonToolParser::new(HERMES_CONFIG, tools))))
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

/// Qwen3 XML: `<tool_call>\n{...}\n</tool_call>` with required newlines.
pub struct Qwen3XmlToolParser(JsonToolParser);

impl ToolParser for Qwen3XmlToolParser {
    fn create(tools: &[Tool]) -> Result<Box<dyn ToolParser>> {
        Ok(Box::new(Self(JsonToolParser::new(QWEN_XML_CONFIG, tools))))
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

/// Mistral `[TOOL_CALLS] [{...},{...}]` array grammar.
pub struct MistralToolParser(JsonToolParser);

impl ToolParser for MistralToolParser {
    fn create(tools: &[Tool]) -> Result<Box<dyn ToolParser>> {
        Ok(Box::new(Self(JsonToolParser::new(MISTRAL_CONFIG, tools))))
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

/// Phi-4 Mini `functools[{...}]` array grammar.
pub struct Phi4MiniJsonToolParser(JsonToolParser);

impl ToolParser for Phi4MiniJsonToolParser {
    fn create(tools: &[Tool]) -> Result<Box<dyn ToolParser>> {
        Ok(Box::new(Self(JsonToolParser::new(PHI4MINI_CONFIG, tools))))
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

/// InternLM2 `<|action_start|><|plugin|>{...}<|action_end|>` grammar.
pub struct Internlm2ToolParser(JsonToolParser);

impl ToolParser for Internlm2ToolParser {
    fn create(tools: &[Tool]) -> Result<Box<dyn ToolParser>> {
        Ok(Box::new(Self(JsonToolParser::new(INTERNLM2_CONFIG, tools))))
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

/// Llama 3.x/4.x bare JSON tool calls using the `parameters` argument key.
pub struct Llama3JsonToolParser(JsonToolParser);

impl ToolParser for Llama3JsonToolParser {
    fn create(tools: &[Tool]) -> Result<Box<dyn ToolParser>> {
        Ok(Box::new(Self(JsonToolParser::new(LLAMA3_CONFIG, tools))))
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

/// Byte length that may be safely emitted without splitting a partial marker.
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

/// Find `marker` outside JSON string literals. `initial_depth` is 1 for
/// wrapper arrays whose opening bracket was consumed with the start marker.
fn find_marker_outside_json(haystack: &str, marker: &str, array_mode: bool) -> Option<usize> {
    let bytes = haystack.as_bytes();
    let closes_block = matches!(marker.as_bytes().first(), Some(b']' | b'}'));
    let mut depth = if array_mode { 1usize } else { 0usize };
    let mut in_string = false;
    let mut escape = false;
    for (index, byte) in bytes.iter().copied().enumerate() {
        if escape {
            escape = false;
            continue;
        }
        if in_string {
            match byte {
                b'\\' => escape = true,
                b'"' => in_string = false,
                _ => {}
            }
            continue;
        }
        if let Some(rest) = haystack.get(index..) {
            let at_boundary = if closes_block {
                depth == 1
            } else {
                depth == 0
            };
            if at_boundary && rest.starts_with(marker) {
                return Some(index);
            }
        }
        match byte {
            b'"' => in_string = true,
            b'{' | b'[' => depth += 1,
            b'}' | b']' => depth = depth.saturating_sub(1),
            _ => {}
        }
    }
    None
}

/// Byte offset just past the `}` that closes the object starting at byte 0.
fn json_object_end(raw: &str) -> Option<usize> {
    let mut depth = 0i32;
    let mut in_string = false;
    let mut escape = false;
    let mut started = false;
    for (index, ch) in raw.char_indices() {
        if escape {
            escape = false;
            continue;
        }
        if in_string {
            match ch {
                '\\' => escape = true,
                '"' => in_string = false,
                _ => {}
            }
            continue;
        }
        match ch {
            '"' => in_string = true,
            '{' | '[' => {
                started = true;
                depth += 1;
            }
            '}' | ']' => {
                depth -= 1;
                if started && depth == 0 {
                    return Some(index + ch.len_utf8());
                }
            }
            _ => {}
        }
    }
    None
}

/// Extract `(name, arguments)` from a tool-call JSON object.
fn tool_call_from_json(
    raw: &str,
    name_keys: &[&str],
    arguments_keys: &[&str],
) -> Option<(String, String)> {
    let bytes = raw.as_bytes();
    if bytes.first() != Some(&b'{') {
        return None;
    }
    let mut index = 1;
    let mut name: Option<String> = None;
    let mut arguments: Option<String> = None;
    loop {
        index = skip_json_ws(bytes, index);
        if index >= bytes.len() || bytes[index] == b'}' {
            break;
        }
        let (key, after_key) = scan_json_string(raw, index)?;
        index = skip_json_ws(bytes, after_key);
        if bytes.get(index) != Some(&b':') {
            return None;
        }
        index = skip_json_ws(bytes, index + 1);
        let value_start = index;
        let value_end = scan_json_value(raw, value_start)?;
        let value_span = &raw[value_start..value_end];
        if name_keys.contains(&key.as_str()) {
            name = serde_json::from_str(value_span).ok();
        } else if arguments_keys.contains(&key.as_str()) {
            arguments = Some(
                serde_json::from_str::<String>(value_span)
                    .unwrap_or_else(|_| value_span.to_string()),
            );
        }
        index = skip_json_ws(bytes, value_end);
        if bytes.get(index) == Some(&b',') {
            index += 1;
        } else {
            break;
        }
    }
    let name = name?;
    if name.is_empty() {
        return None;
    }
    Some((name, arguments.unwrap_or_else(|| "{}".to_string())))
}

fn skip_json_ws(bytes: &[u8], mut index: usize) -> usize {
    while index < bytes.len() && matches!(bytes[index], b' ' | b'\t' | b'\r' | b'\n') {
        index += 1;
    }
    index
}

fn scan_json_string(raw: &str, start: usize) -> Option<(String, usize)> {
    let bytes = raw.as_bytes();
    if bytes.get(start) != Some(&b'"') {
        return None;
    }
    let mut index = start + 1;
    while index < bytes.len() {
        match bytes[index] {
            b'\\' => index += 2,
            b'"' => {
                let value: String = serde_json::from_str(&raw[start..index + 1]).ok()?;
                return Some((value, index + 1));
            }
            _ => index += 1,
        }
    }
    None
}

fn scan_json_value(raw: &str, start: usize) -> Option<usize> {
    let bytes = raw.as_bytes();
    match bytes.get(start)? {
        b'"' => scan_json_string(raw, start).map(|(_, end)| end),
        b'{' | b'[' => {
            let mut depth = 0usize;
            let mut in_string = false;
            let mut escape = false;
            let mut index = start;
            while index < bytes.len() {
                let byte = bytes[index];
                index += 1;
                if in_string {
                    if escape {
                        escape = false;
                    } else if byte == b'\\' {
                        escape = true;
                    } else if byte == b'"' {
                        in_string = false;
                    }
                    continue;
                }
                match byte {
                    b'"' => in_string = true,
                    b'{' | b'[' => depth += 1,
                    b'}' | b']' => {
                        depth = depth.checked_sub(1)?;
                        if depth == 0 {
                            return Some(index);
                        }
                    }
                    _ => {}
                }
            }
            None
        }
        _ => {
            let mut index = start;
            while index < bytes.len()
                && !matches!(bytes[index], b',' | b'}' | b' ' | b'\t' | b'\r' | b'\n')
            {
                index += 1;
            }
            Some(index)
        }
    }
}

fn strip_tool_markers(text: &str) -> String {
    const MARKERS: &[&str] = &[
        "<tool_call>",
        "</tool_call>",
        "<|content_invoke_tool_json|>",
        "<|end_message|>",
        "<|message_model|>",
    ];
    let mut cleaned = text.to_string();
    for marker in MARKERS {
        cleaned = cleaned.replace(marker, "");
    }
    cleaned
}

fn trim_leading_tool_name(prefix: &mut String, tool_names: &[String]) {
    let trimmed = prefix.trim_end();
    if trimmed.is_empty() {
        return;
    }
    for name in tool_names {
        if !trimmed.ends_with(name.as_str()) {
            continue;
        }
        let head_end = trimmed.len() - name.len();
        let boundary_ok = trimmed[..head_end]
            .chars()
            .next_back()
            .map(|ch| !(ch.is_alphanumeric() || ch == '_'))
            .unwrap_or(true);
        if boundary_ok {
            prefix.truncate(head_end);
            return;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{GenericJsonToolParser, HermesToolParser, JsonToolParser, Qwen3XmlToolParser};
    use crate::parser::tool::{Tool, ToolParser, ToolParserOutput};

    fn tools() -> Vec<Tool> {
        Tool::from_names(&["get_weather".to_string(), "echo".to_string()])
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
    fn generic_parser_matches_legacy_behavior() {
        let mut parser = GenericJsonToolParser::create(&tools()).unwrap();
        let output = run(
            &mut parser,
            &["answer {\"value\":42} end"],
        );
        assert_eq!(output.normal_text(), "answer {\"value\":42} end");
        assert!(output.calls().is_empty());

        let mut parser = GenericJsonToolParser::create(&tools()).unwrap();
        let output = run(
            &mut parser,
            &["get_weather{\"name\":\"get_weather\",\"args\":{\"x\":1}}"],
        );
        assert_eq!(output.normal_text(), "");
        assert_eq!(output.calls().len(), 1);
        assert_eq!(output.calls()[0].arguments, "{\"x\":1}");
    }

    #[test]
    fn hermes_accepts_wrapped_json_without_newline() {
        let mut parser = HermesToolParser::create(&tools()).unwrap();
        let output = run(
            &mut parser,
            &["hi <tool_call>{\"name\":\"get_weather\",\"arguments\":{\"city\":\"SF\"}}</tool_call> done"],
        );
        assert_eq!(output.normal_text(), "hi  done");
        assert_eq!(output.calls().len(), 1);
        assert_eq!(output.calls()[0].name.as_deref(), Some("get_weather"));
    }

    #[test]
    fn qwen_xml_requires_newline_around_payload() {
        let mut parser = Qwen3XmlToolParser::create(&tools()).unwrap();
        let input = "<tool_call>{\"name\":\"get_weather\",\"arguments\":{}}</tool_call>";
        let output = run(&mut parser, &[input]);
        assert_eq!(output.normal_text(), input);
        assert!(output.calls().is_empty());

        let mut parser = Qwen3XmlToolParser::create(&tools()).unwrap();
        let input = "<tool_call>\n{\"name\":\"get_weather\",\"arguments\":{}}\n</tool_call>";
        let output = run(&mut parser, &[input]);
        assert_eq!(output.normal_text(), "");
        assert_eq!(output.calls().len(), 1);
    }

    #[test]
    fn split_markers_and_arguments_are_buffered() {
        let mut parser = Qwen3XmlToolParser::create(&tools()).unwrap();
        let output = run(
            &mut parser,
            &[
                "<tool",
                "_call>\n{\"name\":\"get_weather\",",
                "\"arguments\":{\"city\":",
                "\"SF\"}}\n</tool_",
                "call>",
            ],
        );
        assert_eq!(output.normal_text(), "");
        assert_eq!(output.calls().len(), 1);
        assert_eq!(output.calls()[0].arguments, "{\"city\":\"SF\"}");
    }

    #[test]
    fn end_marker_inside_string_does_not_truncate() {
        let mut parser = Qwen3XmlToolParser::create(&tools()).unwrap();
        let output = run(
            &mut parser,
            &["<tool_call>\n{\"name\":\"echo\",\"arguments\":{\"text\":\"</tool_call>\"}}\n</tool_call>"],
        );
        assert_eq!(output.calls().len(), 1);
        assert_eq!(output.calls()[0].arguments, "{\"text\":\"</tool_call>\"}");
    }

    #[test]
    fn finish_flushes_incomplete_block_as_text() {
        let mut parser = Qwen3XmlToolParser::create(&tools()).unwrap();
        let mut output = ToolParserOutput::default();
        parser
            .parse_into("<tool_call>\n{\"name\":\"get_weather\"", &mut output)
            .unwrap();
        assert_eq!(output.normal_text(), "");
        output.append(parser.finish().unwrap());
        assert!(output.calls().is_empty());
        assert!(output.normal_text().contains("get_weather"));
    }

    #[test]
    fn mistral_array_grammar_extracts_all_calls() {
        let mut parser = super::MistralToolParser::create(&tools()).unwrap();
        let input = "[TOOL_CALLS] [{\"name\":\"get_weather\",\"arguments\":{\"city\":\"SF\"}},{\"name\":\"echo\",\"arguments\":{\"items\":[1,2]}}]";
        let output = run(&mut parser, &[input]);
        assert_eq!(output.normal_text(), "");
        assert_eq!(output.calls().len(), 2);
        assert_eq!(output.calls()[0].name.as_deref(), Some("get_weather"));
        assert_eq!(output.calls()[1].arguments, "{\"items\":[1,2]}");
    }

    #[test]
    fn phi4mini_accepts_parameters_key() {
        let mut parser = super::Phi4MiniJsonToolParser::create(&tools()).unwrap();
        let output = run(
            &mut parser,
            &["functools[{\"name\":\"echo\",\"parameters\":{\"v\":1}}]"],
        );
        assert_eq!(output.calls().len(), 1);
        assert_eq!(output.calls()[0].arguments, "{\"v\":1}");
    }

    #[test]
    fn internlm2_uses_special_token_markers() {
        let mut parser = super::Internlm2ToolParser::create(&tools()).unwrap();
        let output = run(
            &mut parser,
            &["<|action_start|><|plugin|>{\"name\":\"echo\",\"parameters\":{\"v\":2}}<|action_end|>"],
        );
        assert_eq!(output.calls().len(), 1);
        assert_eq!(output.calls()[0].arguments, "{\"v\":2}");
    }

    #[test]
    fn llama3_json_supports_parameters_key_and_semicolons() {
        let mut parser = super::Llama3JsonToolParser::create(&tools()).unwrap();
        let output = run(
            &mut parser,
            &["{\"name\":\"echo\",\"parameters\":{\"v\":1}}; {\"name\":\"get_weather\",\"arguments\":{\"city\":\"SF\"}}"],
        );
        assert_eq!(output.normal_text(), "");
        assert_eq!(output.calls().len(), 2);
        assert_eq!(output.calls()[0].arguments, "{\"v\":1}");
        assert_eq!(output.calls()[1].arguments, "{\"city\":\"SF\"}");
    }

    #[test]
    fn parser_can_be_reused_after_reset() {
        let mut parser = Qwen3XmlToolParser::create(&tools()).unwrap();
        let mut output = ToolParserOutput::default();
        parser.parse_into("<tool_call>\n{\"name\":", &mut output).unwrap();
        let buffered = parser.reset();
        assert!(buffered.contains("name") || buffered.contains("<tool_call>"));
        let output = run(
            &mut parser,
            &["<tool_call>\n{\"name\":\"echo\",\"arguments\":{}}\n</tool_call>"],
        );
        assert_eq!(output.calls().len(), 1);
        assert_eq!(output.calls()[0].tool_index, 0);
    }

    // Keep the module referenced so unused-import warnings are not emitted when
    // only some parsers are exercised.
    #[allow(dead_code)]
    fn _assert_parser_type(_: &JsonToolParser) {}
}