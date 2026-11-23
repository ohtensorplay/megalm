//! PyO3 bindings for frames, metrics and raw JSON helpers.

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyBytes;

use crate::frames::{decode_frame_header, encode_frame};
use crate::metrics::render_prometheus_metrics;

#[pyfunction]
pub(super) fn encode_replica_frame(py: Python<'_>, frame_type: u8, payload_json: &str) -> PyResult<Py<PyBytes>> {
    let frame = encode_frame(frame_type, payload_json).map_err(PyValueError::new_err)?;
    Ok(PyBytes::new_bound(py, &frame).into())
}

#[pyfunction]
pub(super) fn decode_replica_frame_header(header: &[u8]) -> PyResult<(u8, usize)> {
    decode_frame_header(header).map_err(PyValueError::new_err)
}

#[pyfunction]
pub(super) fn prometheus_metrics(payload_json: &str) -> PyResult<String> {
    render_prometheus_metrics(payload_json)
        .map_err(|error| PyValueError::new_err(error.to_string()))
}

/// Last reasoning boundary in a prompt, as seen by the Qwen3 reference
/// parser. `None` means the scan stopped at an unrelated special token and
/// the model's default (content) state applies.
#[pyfunction]
#[pyo3(signature = (prompt_token_ids, start_token_id, end_token_id, special_token_ids))]
pub(super) fn reasoning_boundary(
    prompt_token_ids: Vec<u32>,
    start_token_id: u32,
    end_token_id: u32,
    special_token_ids: Vec<u32>,
) -> Option<bool> {
    crate::parser::last_reasoning_boundary(
        &prompt_token_ids,
        start_token_id,
        end_token_id,
        &special_token_ids,
    )
}
