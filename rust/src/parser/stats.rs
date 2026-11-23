//! Per-request parser counters exported to the gateway.
//!
//! One instance lives for one response stream. The gateway aggregates them
//! into replica-level Prometheus metrics at the end of the request, which
//! keeps the Rust side allocation-free and the Python side model-agnostic.

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct ParserStats {
    /// Number of `push` calls (SSE deltas, or one logical chunk).
    pub pushes: u64,
    /// Number of `finish` calls.
    pub finishes: u64,
    /// Completed tool calls emitted.
    pub calls: u64,
    /// Structured diagnostics produced (malformed/incomplete blocks).
    pub diagnostics: u64,
    /// Bytes of decoded text fed into the parser.
    pub bytes: u64,
}

impl ParserStats {
    /// Stable tuple shape used by the PyO3 binding.
    pub fn as_tuple(&self) -> (u64, u64, u64, u64, u64) {
        (
            self.pushes,
            self.finishes,
            self.calls,
            self.diagnostics,
            self.bytes,
        )
    }

    pub fn record_push(&mut self, delta: &str) {
        self.pushes += 1;
        self.bytes += delta.len() as u64;
    }

    pub fn record_finish(&mut self) {
        self.finishes += 1;
    }
}
