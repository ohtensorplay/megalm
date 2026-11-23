//! PyO3 bindings for the streaming output parsers.

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyBytes;

use crate::parser::{
    create_tool_parser, InklingOutput, InklingParser, MarkerKind, OutputParser, StopMatcher,
    Tool, ToolCall, ToolCallParser, ToolParser, ToolParserOutput,
};
use crate::sse::{reasoning_chunk, tool_call_chunk, tool_call_delta_chunk};


/// Streaming stop-sequence matcher exposed to the Python gateway.
///
/// ``push`` returns ``(emitted_text, stopped)``; after a match everything is
/// suppressed, mirroring the engine-level stop semantics.
#[pyclass]
#[derive(Default)]
pub(super) struct PyStopMatcher {
    inner: StopMatcher,
}

#[pymethods]
impl PyStopMatcher {
    #[new]
    #[pyo3(signature = (stop_sequences))]
    fn new(stop_sequences: Vec<String>) -> PyResult<Self> {
        if stop_sequences.is_empty() {
            return Err(PyValueError::new_err(
                "stop_sequences must not be empty; use the no-op path instead",
            ));
        }
        Ok(Self {
            inner: StopMatcher::new(stop_sequences),
        })
    }

    fn push(&mut self, delta: &str) -> PyResult<(String, bool)> {
        let (emitted, stopped) = self.inner.push(delta);
        Ok((emitted, stopped))
    }

    fn finish(&mut self) -> String {
        self.inner.finish()
    }
}

/// Unified streaming output parser exposed to the Python gateway.
///
/// ``push`` returns ``(content, reasoning, calls, stopped)``; ``finish``
/// returns ``(content, reasoning, calls)``. Each call is an
/// ``(id, name, arguments_json)`` tuple.
/// Split complete tool calls into OpenAI-style argument deltas.
///
/// The first fragment keeps the call id and function name; later fragments
/// carry only `(id, None, argument_text)`. The id is repeated so the gateway
/// can map every fragment back to its call without extra state.
fn fragment_calls(
    calls: Vec<ToolCall>,
    fragment_chars: usize,
) -> Vec<(String, Option<String>, String)> {
    let fragment_chars = fragment_chars.max(1);
    let mut fragments = Vec::new();
    for call in calls {
        let mut chars = call.arguments.chars().peekable();
        let mut first = true;
        loop {
            let mut piece = String::new();
            for _ in 0..fragment_chars {
                match chars.next() {
                    Some(character) => piece.push(character),
                    None => break,
                }
            }
            if piece.is_empty() && !first {
                break;
            }
            fragments.push((
                call.id.clone(),
                if first {
                    Some(call.name.clone())
                } else {
                    None
                },
                piece,
            ));
            first = false;
            if chars.peek().is_none() {
                break;
            }
        }
    }
    fragments
}

#[pyclass]
pub(super) struct PyOutputParser {
    inner: OutputParser,
}

#[pymethods]
impl PyOutputParser {
    #[new]
    #[pyo3(signature = (id_prefix, stop_sequences, tool_names, thinking_start, thinking_end, initial_in_thinking, tool_parser_name=None, tools_json=None, forced_tool=None))]
    fn new(
        id_prefix: String,
        stop_sequences: Vec<String>,
        tool_names: Vec<String>,
        thinking_start: Option<String>,
        thinking_end: Option<String>,
        initial_in_thinking: bool,
        tool_parser_name: Option<String>,
        tools_json: Option<String>,
        forced_tool: Option<String>,
    ) -> PyResult<Self> {
        let dynamic = match tool_parser_name.as_deref() {
            None | Some("") | Some("generic_json") | Some("json") | Some("none") => None,
            Some(name) => {
                let tools = match tools_json {
                    Some(raw) => serde_json::from_str::<Vec<Tool>>(&raw).map_err(|error| {
                        PyValueError::new_err(format!(
                            "invalid tool schemas passed to Rust parser: {error}"
                        ))
                    })?,
                    None => Tool::from_names(&tool_names),
                };
                let parser = create_tool_parser(name, &tools)
                    .map_err(|error| PyValueError::new_err(error.to_string()))?;
                Some(parser)
            }
        };
        let mut inner = match dynamic {
            Some(parser) => OutputParser::with_tool_parser(
                id_prefix,
                stop_sequences,
                parser,
                thinking_start,
                thinking_end,
                initial_in_thinking,
            ),
            None => OutputParser::new(
                id_prefix,
                stop_sequences,
                tool_names,
                thinking_start,
                thinking_end,
                initial_in_thinking,
            ),
        };
        inner.set_forced_tool(forced_tool);
        Ok(Self { inner })
    }

    fn push(
        &mut self,
        delta: &str,
    ) -> (String, String, Vec<(String, String, String)>, bool) {
        let (content, reasoning, calls, stopped) = self.inner.push(delta);
        (
            content,
            reasoning,
            calls
                .into_iter()
                .map(|call| (call.id, call.name, call.arguments))
                .collect(),
            stopped,
        )
    }

    fn finish(&mut self) -> (String, String, Vec<(String, String, String)>) {
        let (content, reasoning, calls) = self.inner.finish();
        (
            content,
            reasoning,
            calls
                .into_iter()
                .map(|call| (call.id, call.name, call.arguments))
                .collect(),
        )
    }

    #[pyo3(signature = (delta, fragment_chars=16))]
    fn push_fragments(
        &mut self,
        delta: &str,
        fragment_chars: usize,
    ) -> (String, String, Vec<(String, Option<String>, String)>, bool) {
        let (content, reasoning, calls, stopped) = self.inner.push(delta);
        (
            content,
            reasoning,
            fragment_calls(calls, fragment_chars),
            stopped,
        )
    }

    #[pyo3(signature = (fragment_chars=16))]
    fn finish_fragments(
        &mut self,
        fragment_chars: usize,
    ) -> (String, String, Vec<(String, Option<String>, String)>) {
        let (content, reasoning, calls) = self.inner.finish();
        (content, reasoning, fragment_calls(calls, fragment_chars))
    }

    /// Per-request counters: (pushes, finishes, calls, diagnostics, bytes).
    fn stats(&self) -> (u64, u64, u64, u64, u64) {
        self.inner.stats().as_tuple()
    }

    /// Structured diagnostics from the embedded tool parser.
    fn take_diagnostics(&mut self) -> Vec<(String, String, String, usize, String, String)> {
        self.inner
            .take_diagnostics()
            .into_iter()
            .map(|error| {
                (
                    error.parser.to_string(),
                    error.kind.as_str().to_string(),
                    error.message,
                    error.position,
                    error.context,
                    error.expected,
                )
            })
            .collect()
    }
}

/// Incremental tool-call parser exposed to the Python gateway.
///
/// ``push`` and ``finish`` each return ``(visible_text, calls)`` where every
/// call is an ``(id, name, arguments_json)`` tuple.
#[pyclass]
pub(super) struct PyToolCallParser {
    inner: ToolCallParser,
}

#[pymethods]
impl PyToolCallParser {
    #[new]
    #[pyo3(signature = (id_prefix, tool_names))]
    fn new(id_prefix: String, tool_names: Vec<String>) -> Self {
        Self {
            inner: ToolCallParser::new(id_prefix, tool_names),
        }
    }

    fn push(&mut self, delta: &str) -> (String, Vec<(String, String, String)>) {
        let (visible, calls) = self.inner.push(delta);
        (
            visible,
            calls
                .into_iter()
                .map(|call| (call.id, call.name, call.arguments))
                .collect(),
        )
    }

    fn finish(&mut self) -> (String, Vec<(String, String, String)>) {
        let (visible, calls) = self.inner.finish();
        (
            visible,
            calls
                .into_iter()
                .map(|call| (call.id, call.name, call.arguments))
                .collect(),
        )
    }
}

/// Model-family tool parser exposed to the Python gateway.
///
/// ``push`` returns ``(visible_text, calls)`` where each call is
/// ``(tool_index, name, arguments)``. A call with ``name=None`` is an
/// argument fragment that the caller should append to the same index.
#[pyclass]
pub(super) struct PyToolParser {
    inner: Box<dyn ToolParser>,
}

impl PyToolParser {
    fn into_calls(output: &ToolParserOutput) -> Vec<(usize, Option<String>, String)> {
        output
            .calls()
            .into_iter()
            .map(|call| {
                (
                    call.tool_index,
                    call.name.clone(),
                    call.arguments.clone(),
                )
            })
            .collect()
    }
}

#[pymethods]
impl PyToolParser {
    #[new]
    #[pyo3(signature = (parser_name, tools_json=None, tool_names=None))]
    fn new(
        parser_name: &str,
        tools_json: Option<String>,
        tool_names: Option<Vec<String>>,
    ) -> PyResult<Self> {
        let tools = match tools_json {
            Some(raw) => serde_json::from_str::<Vec<Tool>>(&raw).map_err(|error| {
                PyValueError::new_err(format!("invalid tool schemas: {error}"))
            })?,
            None => Tool::from_names(&tool_names.unwrap_or_default()),
        };
        let inner = create_tool_parser(parser_name, &tools)
            .map_err(|error| PyValueError::new_err(error.to_string()))?;
        Ok(Self { inner })
    }

    fn push(
        &mut self,
        delta: &str,
    ) -> PyResult<(String, Vec<(usize, Option<String>, String)>)> {
        let mut output = ToolParserOutput::default();
        self.inner
            .parse_into(delta, &mut output)
            .map_err(|error| PyValueError::new_err(error.to_string()))?;
        Ok((output.normal_text(), Self::into_calls(&output)))
    }

    fn finish(&mut self) -> PyResult<(String, Vec<(usize, Option<String>, String)>)> {
        let output = self
            .inner
            .finish()
            .map_err(|error| PyValueError::new_err(error.to_string()))?;
        Ok((output.normal_text(), Self::into_calls(&output)))
    }

    fn reset(&mut self) -> String {
        self.inner.reset()
    }

    fn preserve_special_tokens(&self) -> bool {
        self.inner.preserve_special_tokens()
    }

    fn tool_call_id(&self, tool_index: usize) -> Option<String> {
        self.inner.tool_call_id(tool_index).map(str::to_string)
    }
}

/// Inkling typed content-block parser exposed to the Python gateway.
///
/// Inkling-2 separates reasoning, visible text and tool calls with dedicated
/// special tokens, so the decoder must preserve them and this parser owns the
/// block grammar. ``push`` returns ``(content, reasoning, calls, stopped)``
/// and ``finish`` returns ``(content, reasoning, calls)``, matching
/// [`PyOutputParser`] so the streaming loop stays parser-agnostic.
#[pyclass]
pub(super) struct PyInklingParser {
    inner: InklingParser,
    stop_matcher: Option<StopMatcher>,
    terminal: bool,
}

impl PyInklingParser {
    fn into_calls(calls: Vec<ToolCall>) -> Vec<(String, String, String)> {
        calls
            .into_iter()
            .map(|call| (call.id, call.name, call.arguments))
            .collect()
    }
}

#[pymethods]
impl PyInklingParser {
    #[new]
    #[pyo3(signature = (id_prefix, stop_sequences, prompt_token_ids=Vec::new(), marker_kinds=Vec::new()))]
    fn new(
        id_prefix: String,
        stop_sequences: Vec<String>,
        prompt_token_ids: Vec<u32>,
        marker_kinds: Vec<(u32, String)>,
    ) -> Self {
        let kinds: Vec<(u32, MarkerKind)> = marker_kinds
            .iter()
            .map(|(token_id, kind)| (*token_id, MarkerKind::parse(kind)))
            .collect();
        let mut inner = InklingParser::new(id_prefix);
        inner.initialize(&prompt_token_ids, &kinds);
        let stop_matcher = if stop_sequences.is_empty() {
            None
        } else {
            Some(StopMatcher::new(stop_sequences))
        };
        Self {
            inner,
            stop_matcher,
            terminal: false,
        }
    }

    fn push(&mut self, delta: &str) -> (String, String, Vec<(String, String, String)>, bool) {
        if self.terminal {
            return (String::new(), String::new(), Vec::new(), true);
        }
        let mut output = InklingOutput::default();
        self.inner.push(delta, &mut output);
        let InklingOutput {
            content,
            reasoning,
            calls,
        } = output;
        let (emitted, stopped) = match &mut self.stop_matcher {
            Some(matcher) => matcher.push(&content),
            None => (content, false),
        };
        if stopped {
            self.terminal = true;
        }
        (emitted, reasoning, Self::into_calls(calls), stopped)
    }

    fn finish(&mut self) -> (String, String, Vec<(String, String, String)>) {
        if self.terminal {
            return (String::new(), String::new(), Vec::new());
        }
        let mut output = InklingOutput::default();
        self.inner.finish(&mut output);
        let InklingOutput {
            content,
            reasoning,
            calls,
        } = output;
        let emitted = match &mut self.stop_matcher {
            Some(matcher) => {
                let (head, _) = matcher.push(&content);
                let mut combined = head;
                combined.push_str(&matcher.finish());
                combined
            }
            None => content,
        };
        (emitted, reasoning, Self::into_calls(calls))
    }

    /// Per-request counters: (pushes, finishes, calls, diagnostics, bytes).
    fn stats(&self) -> (u64, u64, u64, u64, u64) {
        self.inner.stats().as_tuple()
    }

    /// Take structured diagnostics produced since the previous call.
    ///
    /// Each item is `(parser, kind, message, position, context, expected)`.
    fn take_diagnostics(
        &mut self,
    ) -> Vec<(String, String, String, usize, String, String)> {
        self.inner
            .take_diagnostics()
            .into_iter()
            .map(|error| {
                (
                    error.parser.to_string(),
                    error.kind.as_str().to_string(),
                    error.message,
                    error.position,
                    error.context,
                    error.expected,
                )
            })
            .collect()
    }
}

#[pyfunction]
#[pyo3(signature = (id, model, created, reasoning, index=0))]
pub(super) fn sse_reasoning_chunk(
    py: Python<'_>,
    id: &str,
    model: &str,
    created: i64,
    reasoning: &str,
    index: usize,
) -> Py<PyBytes> {
    PyBytes::new_bound(py, &reasoning_chunk(id, model, created, index, reasoning)).into()
}

#[pyfunction]
#[pyo3(signature = (id, model, created, choice_index, index, call_id=None, name=None, arguments=""))]
pub(super) fn sse_tool_call_delta_chunk(
    py: Python<'_>,
    id: &str,
    model: &str,
    created: i64,
    choice_index: usize,
    index: usize,
    call_id: Option<String>,
    name: Option<String>,
    arguments: &str,
) -> Py<PyBytes> {
    PyBytes::new_bound(
        py,
        &tool_call_delta_chunk(
            id,
            model,
            created,
            choice_index,
            index,
            call_id.as_deref(),
            name.as_deref(),
            arguments,
        ),
    )
    .into()
}
#[pyfunction]
#[pyo3(signature = (id, model, created, choice_index, call_id, index, name, arguments))]
pub(super) fn sse_tool_call_chunk(
    py: Python<'_>,
    id: &str,
    model: &str,
    created: i64,
    choice_index: usize,
    call_id: &str,
    index: usize,
    name: &str,
    arguments: &str,
) -> Py<PyBytes> {
    PyBytes::new_bound(
        py,
        &tool_call_chunk(
            id,
            model,
            created,
            choice_index,
            call_id,
            index,
            name,
            arguments,
        ),
    )
    .into()
}

#[cfg(test)]
mod tests {
    use super::fragment_calls;
    use crate::parser::ToolCall;

    #[test]
    fn fragment_calls_splits_arguments_and_keeps_name_on_first() {
        let call = ToolCall {
            id: "req_0".to_string(),
            name: "f".to_string(),
            arguments: "{\"a\":1,\"b\":2}".to_string(),
        };
        let fragments = fragment_calls(vec![call], 4);
        assert!(fragments.len() > 1);
        assert_eq!(
            fragments[0],
            (
                "req_0".to_string(),
                Some("f".to_string()),
                "{\"a\"".to_string()
            )
        );
        for (id, name, _) in &fragments[1..] {
            assert_eq!(id, "req_0");
            assert!(name.is_none());
        }
        let joined: String = fragments
            .iter()
            .map(|(_, _, arguments)| arguments.as_str())
            .collect();
        assert_eq!(joined, "{\"a\":1,\"b\":2}");
    }

    #[test]
    fn fragment_calls_emits_one_fragment_for_empty_arguments() {
        let call = ToolCall {
            id: "req_0".to_string(),
            name: "f".to_string(),
            arguments: String::new(),
        };
        let fragments = fragment_calls(vec![call], 16);
        assert_eq!(
            fragments,
            vec![("req_0".to_string(), Some("f".to_string()), String::new())]
        );
    }
}
