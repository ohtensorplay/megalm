//! Unified streaming output parser for text-grammar models.

use std::collections::BTreeSet;

use super::error::ParseError;
use super::stats::ParserStats;
use super::{
    reasoning::ThinkingSplitter, StopMatcher, ToolCall, ToolCallParser, ToolParser,
    ToolParserOutput,
};

/// Tool-call backend for one request stream.
///
/// The legacy variant keeps the original generic parser API; the dynamic
/// variant routes to a model-family parser from the Rust registry.
enum ToolCallExtractor {
    Legacy {
        parser: ToolCallParser,
        forced_tool: Option<String>,
    },
    Dynamic {
        id_prefix: String,
        inner: Box<dyn ToolParser>,
        forced_tool: Option<String>,
        rejected: BTreeSet<usize>,
    },
}

impl ToolCallExtractor {
    fn set_forced_tool(&mut self, forced_tool: Option<String>) {
        match self {
            Self::Legacy { forced_tool: slot, .. } | Self::Dynamic { forced_tool: slot, .. } => {
                *slot = forced_tool;
            }
        }
    }

    fn push(&mut self, delta: &str) -> (String, Vec<ToolCall>) {
        match self {
            Self::Legacy { parser, forced_tool } => {
                let (text, calls) = parser.push(delta);
                let calls = calls
                    .into_iter()
                    .filter(|call| {
                        forced_tool
                            .as_deref()
                            .map(|forced| forced == call.name)
                            .unwrap_or(true)
                    })
                    .collect();
                (text, calls)
            }
            Self::Dynamic {
                id_prefix,
                inner,
                forced_tool,
                rejected,
            } => {
                let mut output = ToolParserOutput::default();
                match inner.parse_into(delta, &mut output) {
                    Ok(()) => {
                        dynamic_calls(id_prefix, output, forced_tool.as_deref(), rejected)
                    }
                    Err(_) => (delta.to_string(), Vec::new()),
                }
            }
        }
    }

    fn finish(&mut self) -> (String, Vec<ToolCall>) {
        match self {
            Self::Legacy { parser, forced_tool } => {
                let (text, calls) = parser.finish();
                let calls = calls
                    .into_iter()
                    .filter(|call| {
                        forced_tool
                            .as_deref()
                            .map(|forced| forced == call.name)
                            .unwrap_or(true)
                    })
                    .collect();
                (text, calls)
            }
            Self::Dynamic {
                id_prefix,
                inner,
                forced_tool,
                rejected,
            } => match inner.finish() {
                Ok(output) => {
                    dynamic_calls(id_prefix, output, forced_tool.as_deref(), rejected)
                }
                Err(_) => (inner.reset(), Vec::new()),
            },
        }
    }

    fn take_diagnostics(&mut self) -> Vec<super::ParseError> {
        match self {
            Self::Legacy { parser, .. } => parser.take_diagnostics(),
            Self::Dynamic { .. } => Vec::new(),
        }
    }
}

fn dynamic_calls(
    id_prefix: &str,
    output: ToolParserOutput,
    forced_tool: Option<&str>,
    rejected: &mut BTreeSet<usize>,
) -> (String, Vec<ToolCall>) {
    let output = output.coalesce();
    let text = output.normal_text();
    let calls = output
        .calls()
        .into_iter()
        .filter_map(|call| {
            let name = call.name.as_ref()?;
            if let Some(forced) = forced_tool {
                if forced != name {
                    rejected.insert(call.tool_index);
                    return None;
                }
            }
            if rejected.contains(&call.tool_index) {
                return None;
            }
            Some(ToolCall {
                id: format!("{id_prefix}_{}", call.tool_index),
                name: name.clone(),
                arguments: call.arguments.clone(),
            })
        })
        .collect();
    (text, calls)
}

/// Combined thinking-split, stop-matching and tool-call extraction for one
/// response stream.
///
/// Decoded deltas first flow through thinking separation, then tool-call
/// extraction (so a stop sequence never truncates a tool-call JSON object),
/// then the stop matcher. This keeps the Python streaming loop to a single
/// cross-language call per delta.
pub struct OutputParser {
    thinking: Option<ThinkingSplitter>,
    tool_parser: Option<ToolCallExtractor>,
    stop_matcher: Option<StopMatcher>,
    terminal: bool,
    stats: ParserStats,
}

impl OutputParser {
    pub fn new(
        id_prefix: impl Into<String>,
        stop_sequences: Vec<String>,
        tool_names: Vec<String>,
        thinking_start: Option<String>,
        thinking_end: Option<String>,
        initial_in_thinking: bool,
    ) -> Self {
        let id_prefix = id_prefix.into();
        let tool_parser = if tool_names.is_empty() {
            None
        } else {
            Some(ToolCallExtractor::Legacy {
                parser: ToolCallParser::new(id_prefix, tool_names),
                forced_tool: None,
            })
        };
        Self::build(
            stop_sequences,
            tool_parser,
            thinking_start,
            thinking_end,
            initial_in_thinking,
        )
    }

    /// Build an output parser around a model-family tool parser.
    pub fn with_tool_parser(
        id_prefix: impl Into<String>,
        stop_sequences: Vec<String>,
        tool_parser: Box<dyn ToolParser>,
        thinking_start: Option<String>,
        thinking_end: Option<String>,
        initial_in_thinking: bool,
    ) -> Self {
        Self::build(
            stop_sequences,
            Some(ToolCallExtractor::Dynamic {
                id_prefix: id_prefix.into(),
                inner: tool_parser,
                forced_tool: None,
                rejected: BTreeSet::new(),
            }),
            thinking_start,
            thinking_end,
            initial_in_thinking,
        )
    }

    fn build(
        stop_sequences: Vec<String>,
        tool_parser: Option<ToolCallExtractor>,
        thinking_start: Option<String>,
        thinking_end: Option<String>,
        initial_in_thinking: bool,
    ) -> Self {
        let stop_matcher = if stop_sequences.is_empty() {
            None
        } else {
            Some(StopMatcher::new(stop_sequences))
        };
        let thinking = match (thinking_start, thinking_end) {
            (Some(start), Some(end)) => {
                Some(ThinkingSplitter::new(start, end, initial_in_thinking))
            }
            _ => None,
        };
        Self {
            thinking,
            tool_parser,
            stop_matcher,
            terminal: false,
            stats: ParserStats::default(),
        }
    }

    /// Restrict emitted tool calls to one function name (OpenAI named
    /// `tool_choice`). Calls for other names are dropped by the Rust core.
    pub fn set_forced_tool(&mut self, forced_tool: Option<String>) {
        if let Some(parser) = &mut self.tool_parser {
            parser.set_forced_tool(forced_tool);
        }
    }

    /// Per-request counters for the gateway's Prometheus aggregation.
    pub fn stats(&self) -> &ParserStats {
        &self.stats
    }

    /// Take structured diagnostics from the embedded tool parser.
    pub fn take_diagnostics(&mut self) -> Vec<ParseError> {
        let errors = match &mut self.tool_parser {
            Some(parser) => parser.take_diagnostics(),
            None => Vec::new(),
        };
        self.stats.diagnostics += errors.len() as u64;
        errors
    }

    /// Feed one decoded delta. Returns `(content, reasoning, calls, stopped)`.
    ///
    /// `stopped` is latched: once a stop sequence matches, all later output is
    /// suppressed, mirroring engine-level stop semantics.
    pub fn push(&mut self, delta: &str) -> (String, String, Vec<ToolCall>, bool) {
        self.stats.record_push(delta);
        if self.terminal {
            return (String::new(), String::new(), Vec::new(), true);
        }
        let (content, reasoning) = match &mut self.thinking {
            Some(splitter) => splitter.push(delta),
            None => (delta.to_string(), String::new()),
        };
        let (visible, calls) = match &mut self.tool_parser {
            Some(parser) => parser.push(&content),
            None => (content, Vec::new()),
        };
        let (emitted, stopped) = match &mut self.stop_matcher {
            Some(matcher) => matcher.push(&visible),
            None => (visible, false),
        };
        if stopped {
            self.terminal = true;
            if let Some(splitter) = &mut self.thinking {
                splitter.finish();
            }
            if let Some(parser) = &mut self.tool_parser {
                parser.finish();
            }
        }
        self.stats.calls += calls.len() as u64;
        (emitted, reasoning, calls, stopped)
    }

    /// Flush trailing text, reasoning and tool calls at end of generation.
    /// Returns `(content, reasoning, calls)`. No output after a stop match.
    pub fn finish(&mut self) -> (String, String, Vec<ToolCall>) {
        self.stats.record_finish();
        if self.terminal {
            return (String::new(), String::new(), Vec::new());
        }
        let (tail_content, reasoning) = match &mut self.thinking {
            Some(splitter) => splitter.finish(),
            None => (String::new(), String::new()),
        };
        let (text, calls) = match &mut self.tool_parser {
            Some(parser) => {
                let (visible, first) = parser.push(&tail_content);
                let (remaining, second) = parser.finish();
                let mut calls = first;
                calls.extend(second);
                (format!("{visible}{remaining}"), calls)
            }
            None => (tail_content, Vec::new()),
        };
        let emitted = match &mut self.stop_matcher {
            Some(matcher) => {
                let (head, _) = matcher.push(&text);
                let mut combined = head;
                combined.push_str(&matcher.finish());
                combined
            }
            None => text,
        };
        self.stats.calls += calls.len() as u64;
        (emitted, reasoning, calls)
    }
}

#[cfg(test)]
mod output_parser_tests {
    use super::OutputParser;

    fn parser(stops: &[&str], tools: &[&str]) -> OutputParser {
        OutputParser::new(
            "call",
            stops.iter().map(|s| s.to_string()).collect(),
            tools.iter().map(|s| s.to_string()).collect(),
            None,
            None,
            false,
        )
    }

    #[test]
    fn tool_call_survives_stop_matcher_holdback() {
        let mut parser = parser(&["STOP"], &["f"]);
        // "{" begins a tool call that must not be held back by stop matching.
        let (visible, reasoning, calls, stopped) =
            parser.push("hi {\"name\":\"f\",\"args\":{\"x\":1}} STOP");
        assert!(stopped);
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].name, "f");
        assert!(!visible.contains("STOP"));
        assert!(reasoning.is_empty());
    }

    #[test]
    fn stop_match_suppresses_later_output() {
        let mut parser = parser(&["END"], &["f"]);
        let (visible, _, calls, stopped) = parser.push("hello END{\"name\":\"f\"}");
        assert!(stopped);
        assert!(!visible.contains("END"));
        // Tool extraction runs before stop matching; a call whose JSON is
        // already complete in this delta is still reported.
        assert_eq!(calls.len(), 1);
        let (visible, _, calls, stopped) = parser.push("trailing");
        assert!(stopped);
        assert_eq!(visible, "");
        assert!(calls.is_empty());
        assert_eq!(parser.finish(), (String::new(), String::new(), Vec::new()));
    }

    #[test]
    fn forced_tool_name_filters_legacy_calls() {
        let mut parser = parser(&[], &["allowed", "other"]);
        parser.set_forced_tool(Some("allowed".to_string()));
        let (_, _, calls, _) = parser.push(
            "{\"name\":\"other\",\"args\":{}}{\"name\":\"allowed\",\"args\":{\"x\":1}}",
        );
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].name, "allowed");
    }

    #[test]
    fn forced_tool_name_filters_family_calls() {
        use crate::parser::{create_tool_parser, Tool};

        let tools = Tool::from_names(&["allowed".to_string()]);
        let family = create_tool_parser("generic_json", &tools).unwrap();
        let mut parser = OutputParser::with_tool_parser(
            "call",
            Vec::new(),
            family,
            None,
            None,
            false,
        );
        parser.set_forced_tool(Some("allowed".to_string()));
        let (_, _, calls, _) = parser.push(
            "{\"name\":\"other\",\"args\":{}}{\"name\":\"allowed\",\"args\":{\"x\":1}}",
        );
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].name, "allowed");
    }

    #[test]
    fn no_stops_or_tools_passes_text_through() {
        let mut parser = parser(&[], &[]);
        assert_eq!(
            parser.push("hello "),
            ("hello ".to_string(), String::new(), Vec::new(), false)
        );
        let (visible, reasoning, calls, stopped) = parser.push("world");
        assert!(!stopped);
        assert!(calls.is_empty());
        assert!(reasoning.is_empty());
        assert_eq!(visible, "world");
    }

    #[test]
    fn finish_flushes_held_back_tail_and_pending_call() {
        let mut parser = parser(&["STOP"], &["f"]);
        let (_, _, _, stopped) = parser.push("abcST");
        assert!(!stopped);
        let (visible, _, calls) = parser.finish();
        assert_eq!(visible, "cST");
        assert!(calls.is_empty());
    }

    fn thinking_parser(initial: bool) -> OutputParser {
        OutputParser::new(
            "call",
            vec![],
            vec![],
            Some("<think>".to_string()),
            Some("</think>".to_string()),
            initial,
        )
    }

    #[test]
    fn thinking_section_is_separated_from_content() {
        let mut parser = thinking_parser(false);
        let (content, reasoning, _, stopped) =
            parser.push("<think>chain of thought</think>final answer");
        assert!(!stopped);
        assert_eq!(content, "final answer");
        assert_eq!(reasoning, "chain of thought");
    }

    #[test]
    fn thinking_markers_split_across_deltas() {
        let mut parser = thinking_parser(false);
        let (content, reasoning, _, stopped) = parser.push("abc<thi");
        assert!(!stopped);
        assert_eq!(content, "abc");
        assert!(reasoning.is_empty());
        let (content, reasoning, _, _) = parser.push("nk>hidden");
        assert_eq!(content, "");
        assert_eq!(reasoning, "hidden");
        let (content, reasoning, _, _) = parser.push("</th");
        assert_eq!(content, "");
        assert_eq!(reasoning, "");
        let (content, reasoning, _, _) = parser.push("ink>visible");
        assert_eq!(content, "visible");
        assert_eq!(reasoning, "");
    }

    #[test]
    fn prompt_seeded_thinking_collects_reasoning() {
        let mut parser = thinking_parser(true);
        let (content, reasoning, _, _) = parser.push("step one");
        assert_eq!(content, "");
        assert_eq!(reasoning, "step one");
        let (content, reasoning, _, _) = parser.push("</think>answer");
        assert_eq!(content, "answer");
        assert_eq!(reasoning, "");
        let (content, reasoning, _) = parser.finish();
        assert_eq!(content, "");
        assert_eq!(reasoning, "");
    }

    #[test]
    fn unclosed_thinking_flushes_as_reasoning() {
        let mut parser = thinking_parser(false);
        // The reasoning text has no marker prefix, so it is emitted as soon
        // as it arrives; finish() then has nothing left to flush.
        let (content, reasoning, _, _) = parser.push("<think>partial");
        assert_eq!(content, "");
        assert_eq!(reasoning, "partial");
        let (content, reasoning, _) = parser.finish();
        assert_eq!(content, "");
        assert_eq!(reasoning, "");
    }

    #[test]
    fn plain_output_without_markers_is_content() {
        let mut parser = thinking_parser(false);
        let (content, reasoning, _, _) = parser.push("no markers here");
        assert_eq!(content, "no markers here");
        assert!(reasoning.is_empty());
    }

    #[test]
    fn multibyte_text_before_a_partial_marker_does_not_panic() {
        let mut parser = thinking_parser(false);
        // A BMP char (3 bytes) and an emoji (4 bytes) land inside the
        // holdback window, so byte-wise slicing there would split a char.
        let (content, reasoning, _, _) = parser.push("你好😀<thi");
        assert_eq!(content, "你好😀");
        assert!(reasoning.is_empty());
        let (content, reasoning, _, _) = parser.push("nk>hidden</think>after");
        assert_eq!(content, "after");
        assert_eq!(reasoning, "hidden");
    }
}

