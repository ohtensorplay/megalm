//! Kimi K2 token-delimited tool calls.
//!
//! ```text
//! <|tool_calls_section_begin|>
//! <|tool_call_begin|>functions.get_weather:0<|tool_call_argument_begin|>{"city":"NYC"}<|tool_call_end|>
//! <|tool_calls_section_end|>
//! ```
//!
//! The marker tokens are special tokens, so the decoder must preserve them.

use super::{Result, Tool, ToolCallDelta, ToolParser, ToolParserOutput};

const SECTION_START: &str = "<|tool_calls_section_begin|>";
const SECTION_END: &str = "<|tool_calls_section_end|>";
const CALL_START: &str = "<|tool_call_begin|>";
const CALL_END: &str = "<|tool_call_end|>";
const ARGUMENTS_START: &str = "<|tool_call_argument_begin|>";

/// Incremental parser for Kimi K2 tool-call sections.
pub struct KimiK2ToolParser {
    buffer: String,
    emitted: usize,
    in_section: bool,
}

impl KimiK2ToolParser {
    fn new() -> Self {
        Self {
            buffer: String::new(),
            emitted: 0,
            in_section: false,
        }
    }

    fn drain(&mut self, output: &mut ToolParserOutput, final_flush: bool) {
        loop {
            if self.in_section {
                let leading = self.buffer.len() - self.buffer.trim_start().len();
                if leading > 0 {
                    self.buffer.drain(..leading);
                }
                if self.buffer.starts_with(SECTION_END) {
                    self.buffer.drain(..SECTION_END.len());
                    self.in_section = false;
                    continue;
                }
                let Some(relative) = self.buffer.find(CALL_START) else {
                    if final_flush {
                        self.flush_section(output);
                    }
                    break;
                };
                if relative > 0 {
                    let gap = self.buffer[..relative].trim();
                    if !gap.is_empty() {
                        output.push_text(gap);
                    }
                    self.buffer.drain(..relative);
                }

                let Some(arguments_relative) = self.buffer.find(ARGUMENTS_START) else {
                    if final_flush {
                        self.flush_section(output);
                    }
                    break;
                };
                let header = &self.buffer[CALL_START.len()..arguments_relative];
                let arguments_from = arguments_relative + ARGUMENTS_START.len();
                let Some(end_relative) = self.buffer[arguments_from..].find(CALL_END) else {
                    if final_flush {
                        self.flush_section(output);
                    }
                    break;
                };
                let arguments_end = arguments_from + end_relative;
                let arguments = self.buffer[arguments_from..arguments_end].trim().to_string();
                if let Some(name) = parse_call_name(header) {
                    output.push_call(ToolCallDelta {
                        tool_index: self.emitted,
                        name: Some(name),
                        arguments,
                    });
                    self.emitted += 1;
                }
                self.buffer.drain(..arguments_end + CALL_END.len());
                continue;
            }

            match self.buffer.find(SECTION_START) {
                Some(position) => {
                    output.push_text(&self.buffer[..position]);
                    self.buffer.drain(..position + SECTION_START.len());
                    self.in_section = true;
                }
                None => {
                    if final_flush {
                        let text = std::mem::take(&mut self.buffer);
                        output.push_text(text);
                    } else {
                        let safe = safe_text_len(&self.buffer, SECTION_START);
                        output.push_text(&self.buffer[..safe]);
                        self.buffer.drain(..safe);
                    }
                    break;
                }
            }
        }
    }

    fn flush_section(&mut self, output: &mut ToolParserOutput) {
        let text = format!("{SECTION_START}{}", self.buffer);
        output.push_text(text);
        self.buffer.clear();
        self.in_section = false;
    }
}

impl ToolParser for KimiK2ToolParser {
    fn create(_tools: &[Tool]) -> Result<Box<dyn ToolParser>> {
        Ok(Box::new(Self::new()))
    }

    fn preserve_special_tokens(&self) -> bool {
        true
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
        self.in_section = false;
        self.emitted = 0;
        std::mem::take(&mut self.buffer)
    }
}

/// `functions.get_weather:0` -> `get_weather`.
fn parse_call_name(header: &str) -> Option<String> {
    let header = header.trim();
    let without_index = header.split_once(':').map(|(name, _)| name).unwrap_or(header);
    let name = without_index.rsplit('.').next().unwrap_or(without_index).trim();
    if name.is_empty() {
        None
    } else {
        Some(name.to_string())
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
    use super::{KimiK2ToolParser, ARGUMENTS_START, CALL_END, CALL_START, SECTION_END, SECTION_START};
    use crate::parser::tool::{ToolParser, ToolParserOutput};

    #[test]
    fn parses_kimi_section_and_preserves_prefix_text() {
        let text = format!(
            "Thinking {SECTION_START}\n{functions}functions.get_weather:0{args}{{\"city\":\"NYC\"}}{call_end}\n{section_end}",
            functions = CALL_START,
            args = ARGUMENTS_START,
            call_end = CALL_END,
            section_end = SECTION_END,
        );
        let mut parser = KimiK2ToolParser::create(&[]).unwrap();
        let mut output = ToolParserOutput::default();
        parser.parse_into(&text, &mut output).unwrap();
        output.append(parser.finish().unwrap());
        assert_eq!(output.normal_text(), "Thinking ");
        assert_eq!(output.calls().len(), 1);
        assert_eq!(output.calls()[0].name.as_deref(), Some("get_weather"));
        assert_eq!(output.calls()[0].arguments, "{\"city\":\"NYC\"}");
    }
}
