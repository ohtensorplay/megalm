//! Thin PyO3 facade over the Rust core.
//!
//! Parsing, SSE assembly and caching live in dedicated modules; this layer
//! only exposes them to the Python gateway and owns no logic of its own.

mod cache;
mod helpers;
mod parsers;
mod validator;

use pyo3::prelude::*;

use crate::sse::{
    sse_audio_chunk, sse_content_chunk, sse_content_chunk_with_logprobs, sse_finish_chunk,
    sse_raw_json, sse_role_chunk, sse_usage_chunk,
};

use cache::PyRadixPrefixCache;
use helpers::{
    decode_replica_frame_header, encode_replica_frame, prometheus_metrics, reasoning_boundary,
};
use parsers::{
    sse_reasoning_chunk, sse_tool_call_chunk, sse_tool_call_delta_chunk, PyInklingParser,
    PyOutputParser, PyStopMatcher, PyToolCallParser, PyToolParser,
};
use validator::PyJsonFormatValidator;

pub(super) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<PyStopMatcher>()?;
    module.add_class::<PyJsonFormatValidator>()?;
    module.add_class::<PyOutputParser>()?;
    module.add_class::<PyInklingParser>()?;
    module.add_class::<PyToolCallParser>()?;
    module.add_class::<PyToolParser>()?;
    module.add_class::<PyRadixPrefixCache>()?;
    module.add_function(pyo3::wrap_pyfunction!(prometheus_metrics, module)?)?;
    module.add_function(pyo3::wrap_pyfunction!(reasoning_boundary, module)?)?;
    module.add_function(pyo3::wrap_pyfunction!(sse_tool_call_chunk, module)?)?;
    module.add_function(pyo3::wrap_pyfunction!(sse_tool_call_delta_chunk, module)?)?;
    module.add_function(pyo3::wrap_pyfunction!(encode_replica_frame, module)?)?;
    module.add_function(pyo3::wrap_pyfunction!(decode_replica_frame_header, module)?)?;
    module.add_function(pyo3::wrap_pyfunction!(sse_role_chunk, module)?)?;
    module.add_function(pyo3::wrap_pyfunction!(sse_content_chunk, module)?)?;
    module.add_function(pyo3::wrap_pyfunction!(sse_finish_chunk, module)?)?;
    module.add_function(pyo3::wrap_pyfunction!(sse_usage_chunk, module)?)?;
    module.add_function(pyo3::wrap_pyfunction!(sse_reasoning_chunk, module)?)?;
    module.add_function(pyo3::wrap_pyfunction!(sse_content_chunk_with_logprobs, module)?)?;
    module.add_function(pyo3::wrap_pyfunction!(sse_audio_chunk, module)?)?;
    module.add_function(pyo3::wrap_pyfunction!(sse_raw_json, module)?)?;
    Ok(())
}
