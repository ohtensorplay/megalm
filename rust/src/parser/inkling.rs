//! Inkling typed content-block streaming parser.
//!
//! Inkling-2 does not emit a single ``<think>`` section followed by plain
//! text.  It emits self-describing blocks, each opened by a dedicated special
//! token and closed by ``<|end_message|>`` (or the sampling-end token)::
//!
//! ```text
//! <|message_model|><|content_thinking|>...reasoning...<|end_message|>
//! <|message_model|><|content_text|>...visible text...<|end_message|>
//! <|message_model|>get_weather<|content_invoke_tool_json|>
//!     {"name":"get_weather","args":{"city":"SF"}}<|end_message|>
//! <|content_model_end_sampling|>
//! ```
//!
//! Blocks may repeat in any order.  ``<|content_invoke_tool_text|>`` and
//! ``<|content_tool_error|>`` blocks carry visible text, and the optional
//! function name between ``<|message_model|>`` and the content-kind marker is
//! metadata rather than assistant content.
//!
//! The parser is a resumable state machine over a text buffer: markers split
//! across decoded deltas are held back, and the tool-call header is parsed
//! incrementally so the function name and its ``args`` object can be recovered
//! without waiting for the whole block.  Tool arguments are the *verbatim*
//! ``args`` value span (never re-serialized), which keeps the text
//! prefix-stable for clients that accumulate argument fragments.
//!
//! Unlike the Qwen3-oriented [`crate::parser::OutputParser`], this parser needs the
//! tokenizer's special tokens to be preserved by the decoder, because the
//! block markers *are* the grammar.

use super::error::{ErrorKind, ParseError};
use super::stats::ParserStats;
use crate::parser::ToolCall;

pub const MESSAGE_MODEL: &str = "<|message_model|>";
pub const CONTENT_TEXT: &str = "<|content_text|>";
pub const CONTENT_THINKING: &str = "<|content_thinking|>";
pub const CONTENT_INVOKE_TOOL_JSON: &str = "<|content_invoke_tool_json|>";
pub const CONTENT_INVOKE_TOOL_TEXT: &str = "<|content_invoke_tool_text|>";
pub const CONTENT_TOOL_ERROR: &str = "<|content_tool_error|>";
pub const CONTENT_MODEL_END_SAMPLING: &str = "<|content_model_end_sampling|>";
pub const END_MESSAGE: &str = "<|end_message|>";

/// Markers that may appear between blocks (or inside a message header).
const IDLE_MARKERS: &[&'static str] = &[
    MESSAGE_MODEL,
    CONTENT_TEXT,
    CONTENT_THINKING,
    CONTENT_INVOKE_TOOL_JSON,
    CONTENT_INVOKE_TOOL_TEXT,
    CONTENT_TOOL_ERROR,
    CONTENT_MODEL_END_SAMPLING,
    END_MESSAGE,
];

/// Markers that close the block currently being parsed.
const BLOCK_END_MARKERS: &[&'static str] = &[END_MESSAGE, CONTENT_MODEL_END_SAMPLING];

/// Markers that may follow ``<|message_model|>``.
const HEADER_MARKERS: &[&'static str] = &[
    CONTENT_TEXT,
    CONTENT_THINKING,
    CONTENT_INVOKE_TOOL_JSON,
    CONTENT_INVOKE_TOOL_TEXT,
    CONTENT_TOOL_ERROR,
    CONTENT_MODEL_END_SAMPLING,
    END_MESSAGE,
];

/// Coarse kind of a special token, used to seed the parser from a prompt tail.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MarkerKind {
    Message,
    Text,
    Thinking,
    /// Any other special token: it terminates the backward prompt scan.
    Other,
}

impl MarkerKind {
    /// Parse the string form sent by the Python frontend.
    pub fn parse(value: &str) -> Self {
        match value {
            "message" => Self::Message,
            "text" => Self::Text,
            "thinking" => Self::Thinking,
            _ => Self::Other,
        }
    }
}

/// Events committed by one [`InklingParser::push`] or
/// [`InklingParser::finish`] call.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct InklingOutput {
    pub content: String,
    pub reasoning: String,
    pub calls: Vec<ToolCall>,
}

impl InklingOutput {
    fn push_text(&mut self, text: String) {
        if !text.is_empty() {
            self.content.push_str(&text);
        }
    }

    fn push_reasoning(&mut self, text: String) {
        if !text.is_empty() {
            self.reasoning.push_str(&text);
        }
    }

    fn push_call(&mut self, call: ToolCall) {
        self.calls.push(call);
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Mode {
    /// Between blocks.
    Idle,
    /// After ``<|message_model|>``: the function name is metadata.
    MessageHeader,
    /// Inside a ``<|content_text|>`` (or raw tool text) block.
    Text,
    /// Inside a ``<|content_thinking|>`` block.
    Reasoning,
    /// Inside a tool block, before the ``args`` value starts.
    ToolHeader,
    /// Inside a tool block, scanning the ``args`` object.
    ToolArgs,
}

/// Lexical scan state for the top-level ``args`` object.
///
/// The buffer is not drained while a tool block is open, so `pos` is a stable
/// index into it and the scan resumes where the previous delta stopped.
#[derive(Debug, Clone, Default)]
struct JsonScan {
    pos: usize,
    object_depth: usize,
    array_depth: usize,
    in_string: bool,
    escape: bool,
    started: bool,
    complete: bool,
}

/// Streaming parser for Inkling typed content blocks.
#[derive(Debug)]
pub struct InklingParser {
    id_prefix: String,
    buffer: String,
    mode: Mode,
    scan: JsonScan,
    name: String,
    raw_block: String,
    emitted_tool_count: usize,
    diagnostics: Vec<ParseError>,
    stats: ParserStats,
}

impl InklingParser {
    pub fn new(id_prefix: impl Into<String>) -> Self {
        Self {
            id_prefix: id_prefix.into(),
            buffer: String::new(),
            mode: Mode::Idle,
            scan: JsonScan::default(),
            name: String::new(),
            raw_block: String::new(),
            emitted_tool_count: 0,
            diagnostics: Vec::new(),
            stats: ParserStats::default(),
        }
    }

    /// Per-request counters for the gateway's Prometheus aggregation.
    pub fn stats(&self) -> &ParserStats {
        &self.stats
    }

    /// Take structured diagnostics accumulated since the last call.
    ///
    /// The gateway emits these as `parser_degraded` log events and increments
    /// the parser failure counter; degradation never aborts the stream.
    pub fn take_diagnostics(&mut self) -> Vec<ParseError> {
        std::mem::take(&mut self.diagnostics)
    }

    /// Seed the parser mode from the prompt tail.
    ///
    /// Generation continues inside whichever block the prompt left open, so the
    /// last relevant special token decides the starting mode.  Scanning stops
    /// at the first special token that is not a block opener, mirroring the
    /// reference parser's ``initialize()``.
    pub fn initialize(
        &mut self,
        prompt_token_ids: &[u32],
        marker_kinds: &[(u32, MarkerKind)],
    ) {
        self.mode = Mode::Idle;
        for token_id in prompt_token_ids.iter().rev().copied() {
            match marker_kinds.iter().find(|(id, _)| *id == token_id) {
                Some((_, MarkerKind::Message)) => {
                    self.mode = Mode::MessageHeader;
                    return;
                }
                Some((_, MarkerKind::Thinking)) => {
                    self.mode = Mode::Reasoning;
                    return;
                }
                Some((_, MarkerKind::Text)) => {
                    self.mode = Mode::Text;
                    return;
                }
                Some((_, MarkerKind::Other)) => return,
                None => {}
            }
        }
    }

    /// Feed one decoded delta and append committed events to `output`.
    pub fn push(&mut self, delta: &str, output: &mut InklingOutput) {
        self.stats.record_push(delta);
        self.buffer.push_str(delta);
        while self.step(output) {}
    }

    /// Flush buffered text at end of generation.
    ///
    /// An unterminated tool block is surfaced as visible text rather than
    /// dropped, so a truncated generation never loses user-visible output.
    pub fn finish(&mut self, output: &mut InklingOutput) {
        self.stats.record_finish();
        match self.mode {
            Mode::Idle | Mode::Text => {
                let text = std::mem::take(&mut self.buffer);
                output.push_text(text);
            }
            Mode::MessageHeader => {
                self.buffer.clear();
            }
            Mode::Reasoning => {
                let text = std::mem::take(&mut self.buffer);
                output.push_reasoning(text);
            }
            Mode::ToolHeader | Mode::ToolArgs => {
                self.stats.diagnostics += 1;
                self.diagnostics.push(ParseError::new(
                    "Inkling",
                    ErrorKind::IncompleteAtFinish,
                    "generation ended inside a tool block",
                    self.buffer.len(),
                    &self.buffer,
                    "<|end_message|>",
                ));
                let mut text = std::mem::take(&mut self.raw_block);
                text.push_str(&self.buffer);
                output.push_text(text);
                self.buffer.clear();
            }
        }
        self.reset();
    }

    fn reset(&mut self) {
        self.mode = Mode::Idle;
        self.scan = JsonScan::default();
        self.name.clear();
        self.raw_block.clear();
        self.emitted_tool_count = 0;
    }

    /// Advance the state machine once; returns whether progress was made.
    fn step(&mut self, output: &mut InklingOutput) -> bool {
        match self.mode {
            Mode::Idle | Mode::MessageHeader | Mode::Text | Mode::Reasoning => {
                self.step_text(output)
            }
            Mode::ToolHeader => self.step_tool_header(output),
            Mode::ToolArgs => self.step_tool_args(output),
        }
    }

    fn step_text(&mut self, output: &mut InklingOutput) -> bool {
        let markers: &[&str] = match self.mode {
            Mode::Idle => IDLE_MARKERS,
            Mode::MessageHeader => HEADER_MARKERS,
            Mode::Text | Mode::Reasoning => BLOCK_END_MARKERS,
            Mode::ToolHeader | Mode::ToolArgs => return false,
        };

        if let Some((index, marker)) = find_earliest(&self.buffer, markers) {
            if index > 0 {
                let text = self.buffer[..index].to_string();
                self.buffer.drain(..index);
                self.emit_text(text, output);
                return true;
            }
            self.buffer.drain(..marker.len());
            self.enter_marker(marker);
            return true;
        }

        // No full marker: emit everything except a suffix that could still
        // grow into one after the next decoded delta.
        let keep = markers
            .iter()
            .map(|marker| partial_prefix_len(&self.buffer, marker))
            .max()
            .unwrap_or(0);
        let emit = self.buffer.len() - keep;
        if emit == 0 {
            return false;
        }
        let text = self.buffer[..emit].to_string();
        self.buffer.drain(..emit);
        self.emit_text(text, output);
        true
    }

    fn emit_text(&mut self, text: String, output: &mut InklingOutput) {
        match self.mode {
            Mode::Reasoning => output.push_reasoning(text),
            // Function-name metadata between the model marker and the content
            // kind marker is not assistant-visible content.
            Mode::MessageHeader => {}
            Mode::Idle | Mode::Text => output.push_text(text),
            Mode::ToolHeader | Mode::ToolArgs => output.push_text(text),
        }
    }

    fn enter_marker(&mut self, marker: &str) {
        self.mode = match marker {
            MESSAGE_MODEL => Mode::MessageHeader,
            CONTENT_TEXT | CONTENT_INVOKE_TOOL_TEXT | CONTENT_TOOL_ERROR => Mode::Text,
            CONTENT_THINKING => Mode::Reasoning,
            CONTENT_INVOKE_TOOL_JSON => {
                self.scan = JsonScan::default();
                self.name.clear();
                // The block marker itself is never part of the surfaced raw
                // text: malformed output shows the JSON payload, not the
                // structural markers.
                self.raw_block.clear();
                Mode::ToolHeader
            }
            _ => Mode::Idle,
        };
    }

    fn step_tool_header(&mut self, output: &mut InklingOutput) -> bool {
        match parse_tool_header(&self.buffer) {
            Ok(Some((consumed, name))) => {
                self.raw_block.push_str(&self.buffer[..consumed]);
                self.buffer.drain(..consumed);
                self.name = name;
                self.mode = Mode::ToolArgs;
                true
            }
            Ok(None) => false,
            Err(error) => self.malformed(output, error),
        }
    }

    fn step_tool_args(&mut self, output: &mut InklingOutput) -> bool {
        if !self.scan.complete {
            match scan_json_object(&self.buffer, &mut self.scan) {
                Ok(true) => {}
                Ok(false) => return false,
                Err(error) => return self.malformed(output, error),
            }
        }

        let args_end = self.scan.pos;
        match parse_tool_close(&self.buffer[args_end..]) {
            Ok(Some(consumed)) => {
                let arguments = self.buffer[..args_end].to_string();
                let name = std::mem::take(&mut self.name);
                let index = self.emitted_tool_count;
                self.emitted_tool_count += 1;
                self.stats.calls += 1;
                output.push_call(ToolCall {
                    id: format!("{}_{}", self.id_prefix, index),
                    name,
                    arguments,
                });
                self.buffer.drain(..args_end + consumed);
                self.mode = Mode::Idle;
                true
            }
            Ok(None) => false,
            Err(error) => self.malformed(output, error),
        }
    }

    /// Recover from a structurally invalid tool block.
    ///
    /// The raw block (markers included) is surfaced as visible text up to the
    /// closing marker, which keeps malformed model output visible instead of
    /// silently dropping it.  Returns whether the parser advanced.
    fn malformed(&mut self, output: &mut InklingOutput, error: ParseError) -> bool {
        let Some((index, marker)) = find_earliest(&self.buffer, BLOCK_END_MARKERS) else {
            return false;
        };
        self.diagnostics.push(error);
        self.stats.diagnostics += 1;
        let mut text = std::mem::take(&mut self.raw_block);
        text.push_str(&self.buffer[..index]);
        output.push_text(text);
        self.buffer.drain(..index + marker.len());
        self.mode = Mode::Idle;
        true
    }
}

/// Byte index of the longest suffix of `buffer` that is a proper prefix of
/// `marker`.
///
/// The returned length is always a UTF-8 boundary in `marker`, so callers can
/// slice `&marker[..len]` for non-ASCII markers.
fn partial_prefix_len(buffer: &str, marker: &str) -> usize {
    let Some(first_byte) = marker.as_bytes().first().copied() else {
        return 0;
    };
    let max_len = buffer.len().min(marker.len().saturating_sub(1));
    let tail_start = buffer.len() - max_len;
    let buffer_bytes = buffer.as_bytes();
    let marker_bytes = marker.as_bytes();
    // Ascending index = descending length, so the longest match wins.
    for index in tail_start..buffer.len() {
        if buffer_bytes[index] != first_byte {
            continue;
        }
        let len = buffer.len() - index;
        if buffer.is_char_boundary(index)
            && marker.is_char_boundary(len)
            && marker_bytes[..len] == buffer_bytes[index..]
        {
            return len;
        }
    }
    0
}

/// Earliest marker occurrence in `buffer`, if any.
fn find_earliest(buffer: &str, markers: &[&'static str]) -> Option<(usize, &'static str)> {
    markers
        .iter()
        .filter_map(|marker| buffer.find(*marker).map(|index| (index, *marker)))
        .min_by_key(|(index, _)| *index)
}

fn is_json_ws(byte: u8) -> bool {
    matches!(byte, b' ' | b'\t' | b'\r' | b'\n')
}

fn skip_ws(buffer: &str, mut index: usize) -> usize {
    let bytes = buffer.as_bytes();
    while index < bytes.len() && is_json_ws(bytes[index]) {
        index += 1;
    }
    index
}

/// Match `literal` at `index`.
///
/// `Ok(Some(end))` on a full match, `Ok(None)` when the buffer ends inside the
/// literal, and `Err(())` on a mismatch.
fn expect_lit(buffer: &str, index: usize, literal: &str) -> Result<Option<usize>, ParseError> {
    let rest = &buffer[index..];
    let mismatch = || {
        ParseError::new(
            "Inkling",
            ErrorKind::UnexpectedToken,
            "literal mismatch",
            index,
            buffer,
            format!("`{literal}`"),
        )
    };
    if rest.len() >= literal.len() {
        if rest.as_bytes()[..literal.len()] == *literal.as_bytes() {
            Ok(Some(index + literal.len()))
        } else {
            Err(mismatch())
        }
    } else if literal.as_bytes()[..rest.len()] == *rest.as_bytes() {
        Ok(None)
    } else {
        Err(mismatch())
    }
}

/// Parse a JSON string literal starting at `start`.
fn parse_json_string(
    buffer: &str,
    start: usize,
) -> Result<Option<(String, usize)>, ParseError> {
    let bytes = buffer.as_bytes();
    if start >= bytes.len() {
        return Ok(None);
    }
    let invalid = |message, expected| {
        ParseError::new(
            "Inkling",
            ErrorKind::InvalidJson,
            message,
            start,
            buffer,
            expected,
        )
    };
    if bytes[start] != b'"' {
        return Err(invalid("function name is not a string", "`\"`"));
    }
    let mut index = start + 1;
    while index < bytes.len() {
        match bytes[index] {
            b'\\' => index += 2,
            b'"' => {
                let raw = &buffer[start..index + 1];
                let value: String = serde_json::from_str(raw).map_err(|_| {
                    invalid("invalid JSON string escape", "valid string literal")
                })?;
                return Ok(Some((value, index + 1)));
            }
            _ => index += 1,
        }
    }
    Ok(None)
}

/// Parse ``{"name": <string>, "args":`` and return the consumed length plus
/// the function name.
fn parse_tool_header(buffer: &str) -> Result<Option<(usize, String)>, ParseError> {
    let mut index = skip_ws(buffer, 0);
    index = match expect_lit(buffer, index, "{")? {
        Some(next) => next,
        None => return Ok(None),
    };
    index = skip_ws(buffer, index);
    index = match expect_lit(buffer, index, "\"name\"")? {
        Some(next) => next,
        None => return Ok(None),
    };
    index = skip_ws(buffer, index);
    index = match expect_lit(buffer, index, ":")? {
        Some(next) => next,
        None => return Ok(None),
    };
    index = skip_ws(buffer, index);
    let (name, mut next) = match parse_json_string(buffer, index)? {
        Some(parsed) => parsed,
        None => return Ok(None),
    };
    next = skip_ws(buffer, next);
    next = match expect_lit(buffer, next, ",")? {
        Some(after) => after,
        None => return Ok(None),
    };
    next = skip_ws(buffer, next);
    next = match expect_lit(buffer, next, "\"args\"")? {
        Some(after) => after,
        None => return Ok(None),
    };
    next = skip_ws(buffer, next);
    next = match expect_lit(buffer, next, ":")? {
        Some(after) => after,
        None => return Ok(None),
    };
    next = skip_ws(buffer, next);
    if name.is_empty() {
        return Err(ParseError::new(
            "Inkling",
            ErrorKind::InvalidJson,
            "empty function name",
            next,
            buffer,
            "non-empty string",
        ));
    }
    Ok(Some((next, name)))
}

/// Scan one top-level JSON object from `scan.pos`, resuming across deltas.
///
/// Returns `Ok(true)` once the matching brace is consumed; `scan.pos` is then
/// the exclusive end of the object.  `Err(())` reports a non-object value or
/// unbalanced braces/brackets.  Arrays are tracked separately from objects so
/// a nested array cannot be mistaken for the end of the top-level object.
fn scan_json_object(buffer: &str, scan: &mut JsonScan) -> Result<bool, ParseError> {
    let bytes = buffer.as_bytes();
    if !scan.started {
        let start = skip_ws(buffer, 0);
        if start >= bytes.len() {
            return Ok(false);
        }
        if bytes[start] != b'{' {
            return Err(ParseError::new(
                "Inkling",
                ErrorKind::UnexpectedToken,
                "tool arguments must be a JSON object",
                start,
                buffer,
                "`{`",
            ));
        }
        scan.started = true;
        scan.object_depth = 1;
        scan.pos = start + 1;
    }

    let mut index = scan.pos;
    while index < bytes.len() {
        let byte = bytes[index];
        index += 1;
        if scan.in_string {
            if scan.escape {
                scan.escape = false;
            } else if byte == b'\\' {
                scan.escape = true;
            } else if byte == b'"' {
                scan.in_string = false;
            }
            continue;
        }
        match byte {
            b'"' => scan.in_string = true,
            b'{' => scan.object_depth += 1,
            b'}' => {
                scan.object_depth = scan.object_depth.checked_sub(1).ok_or_else(|| {
                    ParseError::new(
                        "Inkling",
                        ErrorKind::Unbalanced,
                        "unbalanced object brace",
                        index - 1,
                        buffer,
                        "matching `{`",
                    )
                })?;
                if scan.object_depth == 0 && scan.array_depth == 0 {
                    scan.pos = index;
                    scan.complete = true;
                    return Ok(true);
                }
                if scan.object_depth == 0 {
                    return Err(ParseError::new(
                        "Inkling",
                        ErrorKind::Unbalanced,
                        "array still open when object closed",
                        index - 1,
                        buffer,
                        "matching `]`",
                    ));
                }
            }
            b'[' => scan.array_depth += 1,
            b']' => {
                scan.array_depth = scan.array_depth.checked_sub(1).ok_or_else(|| {
                    ParseError::new(
                        "Inkling",
                        ErrorKind::Unbalanced,
                        "unbalanced array bracket",
                        index - 1,
                        buffer,
                        "matching `[`",
                    )
                })?;
            }
            _ => {}
        }
    }
    scan.pos = bytes.len();
    Ok(false)
}

/// Parse the wrapper close: optional whitespace, ``}``, optional whitespace and
/// one block-end marker.  Returns the consumed length.
fn parse_tool_close(buffer: &str) -> Result<Option<usize>, ParseError> {
    let mut index = skip_ws(buffer, 0);
    index = match expect_lit(buffer, index, "}")? {
        Some(next) => next,
        None => return Ok(None),
    };
    index = skip_ws(buffer, index);
    let mut last_error = None;
    for marker in BLOCK_END_MARKERS {
        match expect_lit(buffer, index, marker) {
            Ok(Some(end)) => return Ok(Some(end)),
            Ok(None) => return Ok(None),
            Err(error) => last_error = Some(error),
        }
    }
    Err(last_error.unwrap_or_else(|| {
        ParseError::new(
            "Inkling",
            ErrorKind::UnexpectedToken,
            "missing block end marker",
            index,
            buffer,
            "`<|end_message|>`",
        )
    }))
}

#[cfg(test)]
mod tests {
    use super::{
        CONTENT_THINKING, CONTENT_TEXT, InklingOutput, InklingParser, MESSAGE_MODEL, MarkerKind,
    };
    use crate::parser::ToolCall;

    fn parser() -> InklingParser {
        InklingParser::new("call")
    }

    fn collect(chunks: &[&str]) -> InklingOutput {
        let mut parser = parser();
        let mut output = InklingOutput::default();
        for chunk in chunks {
            parser.push(chunk, &mut output);
        }
        parser.finish(&mut output);
        output
    }

    fn names(output: &InklingOutput) -> Vec<&str> {
        output.calls.iter().map(|call| call.name.as_str()).collect()
    }

    fn arguments(output: &InklingOutput) -> Vec<&str> {
        output
            .calls
            .iter()
            .map(|call| call.arguments.as_str())
            .collect()
    }

    #[test]
    fn reasoning_and_text_blocks_are_separated() {
        let output = collect(&[concat!(
            "<|content_thinking|>reason<|end_message|>",
            "<|message_model|><|content_text|>answer<|end_message|>",
            "<|content_model_end_sampling|>"
        )]);
        assert_eq!(output.reasoning, "reason");
        assert_eq!(output.content, "answer");
        assert!(output.calls.is_empty());
    }

    #[test]
    fn split_markers_are_held_back() {
        let output = collect(&[
            "<|content_thin",
            "king|>rea",
            "son<|end_mes",
            "sage|><|message_",
            "model|><|content_text|>answer",
            "<|end_message|>",
        ]);
        assert_eq!(output.reasoning, "reason");
        assert_eq!(output.content, "answer");
    }

    #[test]
    fn tool_block_streams_name_and_arguments() {
        let output = collect(&[concat!(
            "<|message_model|><|content_invoke_tool_json|>",
            "{\"name\":\"get_weather\",\"args\":{\"city\":\"SF\"}}",
            "<|end_message|>"
        )]);
        assert_eq!(names(&output), ["get_weather"]);
        assert_eq!(arguments(&output), [r#"{"city":"SF"}"#]);
        assert_eq!(output.calls[0].id, "call_0");
        assert!(output.content.is_empty());
    }

    #[test]
    fn tool_name_in_message_header_is_metadata() {
        let output = collect(&[concat!(
            "<|message_model|>get_weather<|content_invoke_tool_json|>",
            "{\"name\":\"get_weather\",\"args\":{}}<|end_message|>"
        )]);
        assert!(output.content.is_empty());
        assert_eq!(names(&output), ["get_weather"]);
        assert_eq!(arguments(&output), ["{}"]);
    }

    #[test]
    fn multiple_tool_blocks_get_incrementing_indices() {
        let output = collect(&[concat!(
            "<|content_invoke_tool_json|>{\"name\":\"a\",\"args\":{\"x\":1}}<|end_message|>",
            "<|message_model|><|content_invoke_tool_json|>{\"name\":\"b\",\"args\":{\"y\":2}}<|end_message|>"
        )]);
        assert_eq!(names(&output), ["a", "b"]);
        assert_eq!(output.calls[0].id, "call_0");
        assert_eq!(output.calls[1].id, "call_1");
    }

    #[test]
    fn text_after_a_tool_block_stays_visible() {
        let output = collect(&[concat!(
            "<|content_invoke_tool_json|>{\"name\":\"a\",\"args\":{}}<|end_message|>",
            "<|content_text|>done<|end_message|>"
        )]);
        assert_eq!(names(&output), ["a"]);
        assert_eq!(output.content, "done");
    }

    #[test]
    fn raw_tool_text_and_tool_error_blocks_are_visible() {
        let output = collect(&[concat!(
            "<|content_invoke_tool_text|>search SF<|end_message|>",
            "<|content_tool_error|>failed<|end_message|>"
        )]);
        assert_eq!(output.content, "search SFfailed");
        assert!(output.calls.is_empty());
    }

    #[test]
    fn plain_text_without_blocks_passes_through() {
        let output = collect(&["plain ", "answer"]);
        assert_eq!(output.content, "plain answer");
        assert!(output.reasoning.is_empty());
    }

    #[test]
    fn whitespace_inside_the_tool_header_is_tolerated() {
        let output = collect(&[concat!(
            "<|content_invoke_tool_json|>",
            "{ \"name\" : \"f\" , \"args\" : {\"k\": 1} }<|end_message|>"
        )]);
        assert_eq!(names(&output), ["f"]);
        assert_eq!(arguments(&output), [r#"{"k": 1}"#]);
    }

    #[test]
    fn brace_inside_an_argument_string_does_not_close_the_object() {
        let output = collect(&[concat!(
            "<|content_invoke_tool_json|>",
            r#"{"name":"f","args":{"text":"} <|end_message|>"}}"#,
            "<|end_message|>"
        )]);
        assert_eq!(names(&output), ["f"]);
        assert_eq!(
            arguments(&output),
            [r#"{"text":"} <|end_message|>"}"#]
        );
    }

    #[test]
    fn nested_arrays_do_not_close_the_arguments_object() {
        let output = collect(&[concat!(
            "<|content_invoke_tool_json|>",
            r#"{"name":"f","args":{"items":[[1,2],[3,4]],"n":2}}"#,
            "<|end_message|>"
        )]);
        assert_eq!(names(&output), ["f"]);
        assert_eq!(
            arguments(&output),
            [r#"{"items":[[1,2],[3,4]],"n":2}"#]
        );
    }

    #[test]
    fn nested_arrays_are_prefix_stable_across_deltas() {
        let mut parser = parser();
        let mut output = InklingOutput::default();
        parser.push(
            "<|content_invoke_tool_json|>{\"name\":\"f\",\"args\":{\"items\":[[1,2],",
            &mut output,
        );
        assert!(output.calls.is_empty());
        parser.push("[3,4]]}}<|end_message|>", &mut output);
        assert_eq!(names(&output), ["f"]);
        assert_eq!(arguments(&output), [r#"{"items":[[1,2],[3,4]]}"#]);
    }

    #[test]
    fn malformed_tool_block_surfaces_as_visible_text() {
        let output = collect(&[concat!(
            "<|content_invoke_tool_json|>{\"name\":\"f\",\"args\":42}",
            "<|end_message|>"
        )]);
        assert!(output.calls.is_empty());
        assert!(output.content.contains("\"args\":42"));
        assert!(!output.content.contains("<|content_invoke_tool_json|>"));
    }

    #[test]
    fn incomplete_tool_block_is_flushed_as_text() {
        let mut parser = parser();
        let mut output = InklingOutput::default();
        parser.push("<|content_invoke_tool_json|>{\"name\":\"f\",\"args\":{\"x\":1}", &mut output);
        assert!(output.calls.is_empty());
        parser.finish(&mut output);
        assert!(output.calls.is_empty());
        assert!(output.content.contains("\"name\":\"f\""));
    }

    #[test]
    fn initialize_from_prompt_tail() {
        let marker_kinds = vec![
            (200_001, MarkerKind::Message),
            (200_004, MarkerKind::Text),
            (200_008, MarkerKind::Thinking),
            (200_010, MarkerKind::Other),
        ];

        let mut thinking = parser();
        thinking.initialize(&[200_008], &marker_kinds);
        let mut output = InklingOutput::default();
        thinking.push("reason<|end_message|>", &mut output);
        assert_eq!(output.reasoning, "reason");
        assert!(output.content.is_empty());

        let mut text = parser();
        text.initialize(&[200_004], &marker_kinds);
        let mut output = InklingOutput::default();
        text.push("answer<|end_message|>", &mut output);
        assert_eq!(output.content, "answer");

        let mut header = parser();
        header.initialize(&[200_001], &marker_kinds);
        let mut output = InklingOutput::default();
        header.push(
            "get_weather<|content_invoke_tool_json|>{\"name\":\"f\",\"args\":{}}<|end_message|>",
            &mut output,
        );
        assert!(output.content.is_empty());
        assert_eq!(names(&output), ["f"]);
    }

    #[test]
    fn initialize_stops_at_an_unrelated_special_token() {
        let marker_kinds = vec![
            (200_001, MarkerKind::Message),
            (200_010, MarkerKind::Other),
        ];
        let mut seeded = parser();
        // The newest token is unrelated, so the scan must not reach the older
        // message marker.
        seeded.initialize(&[200_001, 200_010], &marker_kinds);
        let mut output = InklingOutput::default();
        seeded.push("<|content_text|>answer<|end_message|>", &mut output);
        assert_eq!(output.content, "answer");
    }

    #[test]
    fn partial_marker_at_end_of_input_is_flushed_by_finish() {
        let mut parser = parser();
        let mut output = InklingOutput::default();
        parser.push("answer<|end_", &mut output);
        assert_eq!(output.content, "answer");
        parser.finish(&mut output);
        assert_eq!(output.content, "answer<|end_");
    }

    #[test]
    fn marker_only_input_produces_no_text() {
        let output = collect(&["<|message_model|><|content_text|><|end_message|>"]);
        assert_eq!(output.content, "");
        assert_eq!(output.reasoning, "");
    }

    #[test]
    fn tool_arguments_are_prefix_stable_across_deltas() {
        let mut parser = parser();
        let mut output = InklingOutput::default();
        parser.push("<|content_invoke_tool_json|>{\"name\":\"f\",\"args\":", &mut output);
        assert!(output.calls.is_empty());
        parser.push("{\"city\":", &mut output);
        assert!(output.calls.is_empty());
        parser.push("\"SF\"}}<|end_message|>", &mut output);
        assert_eq!(names(&output), ["f"]);
        assert_eq!(arguments(&output), [r#"{"city":"SF"}"#]);
    }

    #[test]
    fn ids_are_unique_per_request() {
        let first = collect(&["<|content_invoke_tool_json|>{\"name\":\"a\",\"args\":{}}<|end_message|>"]);
        let second = collect(&["<|content_invoke_tool_json|>{\"name\":\"b\",\"args\":{}}<|end_message|>"]);
        let _: &ToolCall = &first.calls[0];
        assert_eq!(second.calls[0].id, "call_0");
    }

    #[test]
    fn malformed_tool_block_produces_structured_diagnostic() {
        let mut parser = parser();
        let mut output = InklingOutput::default();
        parser.push(
            "<|content_invoke_tool_json|>{\"name\":\"f\",\"args\":42}<|end_message|>",
            &mut output,
        );
        let diagnostics = parser.take_diagnostics();
        assert_eq!(diagnostics.len(), 1);
        assert_eq!(diagnostics[0].parser, "Inkling");
        assert!(!diagnostics[0].message.is_empty());
        assert!(!diagnostics[0].context.is_empty());
        assert!(!diagnostics[0].expected.is_empty());
        // Diagnostics are drained, not replayed.
        assert!(parser.take_diagnostics().is_empty());
    }

    #[test]
    fn incomplete_tool_block_is_diagnosed_at_finish() {
        let mut parser = parser();
        let mut output = InklingOutput::default();
        parser.push("<|content_invoke_tool_json|>{\"name\":\"f\",\"args\":{", &mut output);
        parser.finish(&mut output);
        let diagnostics = parser.take_diagnostics();
        assert_eq!(diagnostics.len(), 1);
        assert_eq!(diagnostics[0].kind.as_str(), "incomplete_at_finish");
    }

    #[test]
    fn stats_track_pushes_calls_and_diagnostics() {
        let mut parser = parser();
        let mut output = InklingOutput::default();
        parser.push(
            "<|content_invoke_tool_json|>{\"name\":\"f\",\"args\":{}}<|end_message|>",
            &mut output,
        );
        let stats = parser.stats();
        assert_eq!(stats.pushes, 1);
        assert_eq!(stats.calls, 1);
        assert_eq!(stats.diagnostics, 0);
        assert!(stats.bytes > 0);
        parser.finish(&mut output);
        assert_eq!(parser.stats().finishes, 1);
    }

    #[test]
    fn constant_markers_match_the_template() {
        assert_eq!(MESSAGE_MODEL, "<|message_model|>");
        assert_eq!(CONTENT_TEXT, "<|content_text|>");
        assert_eq!(CONTENT_THINKING, "<|content_thinking|>");
    }
}