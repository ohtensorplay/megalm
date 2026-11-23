//! MegaLM gateway host-path accelerators.
//!
//! The crate is split by concern: [`parser`] owns the streaming output state
//! machines, [`sse`] assembles OpenAI-compatible chunks, [`cache`] holds the
//! radix prefix cache, and [`bindings`] is the thin PyO3 facade. No parsing
//! logic lives in this crate root.

mod bindings;
mod cache;
mod frames;
mod metrics;
pub mod parser;
mod sse;

use pyo3::prelude::*;

/// Gateway host-path accelerators.
#[pymodule]
fn megalm_rs(module: &Bound<'_, PyModule>) -> PyResult<()> {
    bindings::register(module)
}
