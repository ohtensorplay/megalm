//! Gemma-4 native tool-call grammar.
//!
//! The reference emits
//! `<|tool_call>call:name{key:<|"|>value<|"|>,num:42}<tool_call|>`, where
//! `<|"|>` is an escaped string delimiter. This parser buffers until the
//! closing marker, converts the custom argument syntax to a JSON object, and
//! falls back to visible text for incomplete or malformed blocks.

use serde_json::Value;

use super::parameters::{arguments_object, ToolSchemas};
use super::{Result, Tool, ToolCallDelta, ToolParser, ToolParserError, ToolParserOutput};

const START: &str = "<|tool_call>";
const END: &str = "<tool_call|>";
const STRING_DELIM: &str = "<|\"|>";
const CALL_PREFIX: &str = "call:";

fn safe_text_len(buffer: &str, marker: &str) -> usize {
    let max = marker.len() - 1;
    let from = buffer.len().saturating_sub(max);
    for (pos, _) in buffer.char_indices() {
        if pos < from {
            continue;
        }
        let tail = &buffer[pos..];
        if marker.starts_with(tail) {
            return pos;
        }
    }
    buffer.len()
}

/// Split top-level `key:value` pairs, respecting string delimiters and
/// nested braces/brackets.
fn split_pairs(args: &str) -> Vec<(String, String)> {
    let bytes = args.as_bytes();
    let mut pairs = Vec::new();
    let mut start = 0usize;
    let mut i = 0usize;
    let mut depth = 0i32;
    let mut in_string = false;
    while i < bytes.len() {
        if args[i..].starts_with(STRING_DELIM) {
            in_string = !in_string;
            i += STRING_DELIM.len();
            continue;
        }
        match bytes[i] {
            b'{' | b'[' if !in_string => depth += 1,
            b'}' | b']' if !in_string => depth -= 1,
            b',' if !in_string && depth == 0 => {
                pairs.push(parse_pair(&args[start..i]));
                start = i + 1;
            }
            _ => {}
        }
        i += 1;
    }
    if start < args.len() {
        pairs.push(parse_pair(&args[start..]));
    }
    pairs.into_iter().flatten().collect()
}

fn parse_pair(piece: &str) -> Option<(String, String)> {
    let mut parts = piece.splitn(2, ':');
    let key = parts.next()?.trim();
    let value = parts.next()?.trim();
    if key.is_empty() {
        return None;
    }
    let key = key
        .strip_prefix(STRING_DELIM)
        .and_then(|rest| rest.strip_suffix(STRING_DELIM))
        .unwrap_or(key);
    Some((key.to_string(), value.to_string()))
}

fn parse_value(raw: &str, schemas: &ToolSchemas, tool: &str, key: &str) -> Value {
    if let Some(inner) = raw
        .strip_prefix(STRING_DELIM)
        .and_then(|rest| rest.strip_suffix(STRING_DELIM))
    {
        return Value::String(inner.to_string());
    }
    if let Ok(value) = serde_json::from_str::<Value>(raw) {
        return value;
    }
    schemas.convert_param(tool, key, raw, false)
}

/// Parse one complete `<|tool_call>...<tool_call|>` block.
fn parse_block(block: &str, schemas: &ToolSchemas) -> Option<(String, String)> {
    let body = block.strip_prefix(CALL_PREFIX)?;
    let open = body.find('{')?;
    let close = body.rfind('}')?;
    if close <= open {
        return None;
    }
    let name = body[..open].trim();
    if name.is_empty() {
        return None;
    }
    let args = &body[open + 1 .. close];
    let values = split_pairs(args)
        .into_iter()
        .map(|(key, value)| {
            let parsed = parse_value(&value, schemas, name, &key);
            (key, parsed)
        })
        .collect();
    Some((name.to_string(), arguments_object(values).ok()?))
}

pub struct Gemma4ToolParser {
    buffer: String,
    in_block: bool,
    emitted: usize,
    schemas: ToolSchemas,
}

impl Gemma4ToolParser {
    fn new(tools: &[Tool]) -> Self {
        Self {
            buffer: String::new(),
            in_block: false,
            emitted: 0,
            schemas: ToolSchemas::from_tools(tools),
        }
    }

    fn drain(&mut self, output: &mut ToolParserOutput, final_flush: bool) {
        loop {
            if self.in_block {
                match self.buffer.find(END) {
                    Some(position) => {
                        let block = self.buffer[..position].to_string();
                        self.buffer.drain(..position + END.len());
                        self.in_block = false;
                        match parse_block(&block, &self.schemas) {
                            Some((name, arguments)) => {
                                output.push_call(ToolCallDelta {
                                    tool_index: self.emitted,
                                    name: Some(name),
                                    arguments,
                                });
                                self.emitted += 1;
                            }
                            None => output.push_text(format!("{START}{block}{END}")),
                        }
                        continue;
                    }
                    None => {
                        if final_flush {
                            let text = format!("{START}{}", self.buffer);
                            output.push_text(text);
                            self.buffer.clear();
                            self.in_block = false;
                        }
                        break;
                    }
                }
            }
            match self.buffer.find(START) {
                Some(position) => {
                    output.push_text(&self.buffer[..position]);
                    self.buffer.drain(..position + START.len());
                    self.in_block = true;
                }
                None => {
                    let safe = if final_flush {
                        self.buffer.len()
                    } else {
                        safe_text_len(&self.buffer, START)
                    };
                    output.push_text(&self.buffer[..safe]);
                    self.buffer.drain(..safe);
                    break;
                }
            }
        }
    }

    fn block_as_text(&self) -> String {
        format!("{START}{}{END}", self.buffer)
    }
}

impl ToolParser for Gemma4ToolParser {
    fn create(tools: &[Tool]) -> Result<Box<dyn ToolParser>> {
        Ok(Box::new(Self::new(tools)))
    }

    fn preserve_special_tokens(&self) -> bool {
        true
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
        if self.in_block {
            output.push_text(self.block_as_text());
            self.buffer.clear();
            self.in_block = false;
        }
        self.drain(&mut output, true);
        Ok(output)
    }

    fn reset(&mut self) -> String {
        self.in_block = false;
        self.emitted = 0;
        std::mem::take(&mut self.buffer)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_native_tool_call() {
        let mut parser = Gemma4ToolParser::new(&[]);
        let mut output = ToolParserOutput::default();
        parser
            .parse_into(
                "<|tool_call>call:get_weather{city:<|\"|>Paris<|\"|>,days:3}<tool_call|>",
                &mut output,
            )
            .unwrap();
        let coalesced = output.coalesce();
        let calls = coalesced.calls();
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].name.as_deref(), Some("get_weather"));
        let args: Value = serde_json::from_str(&calls[0].arguments).unwrap();
        assert_eq!(args["city"], "Paris");
        assert_eq!(args["days"], 3);
    }

    #[test]
    fn keeps_incomplete_block_as_text() {
        let mut parser = Gemma4ToolParser::new(&[]);
        let mut output = ToolParserOutput::default();
        parser.parse_into("hello <|tool_call>call:x", &mut output).unwrap();
        assert_eq!(output.coalesce().normal_text(), "hello ");
    }
}
