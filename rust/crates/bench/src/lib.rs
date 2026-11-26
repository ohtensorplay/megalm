//! Backend-agnostic serving benchmark for OpenAI-compatible inference servers.
//!
//! One process drives every engine through the same HTTP protocol and the same
//! seeded request set, so head-to-head numbers are comparable.  Usage:
//!
//! ```text
//! megalm-bench serve --engine megalm=http://127.0.0.1:8000 \
//!   --engine vllm=http://127.0.0.1:8001 --model qwen3-8b \
//!   --concurrency 1,8,16 --trials 3 --output /tmp/opencode/serve.json
//! ```

pub mod cli;
pub mod client;
pub mod metrics;
pub mod provenance;
pub mod report;
pub mod runner;
pub mod workload;

pub use cli::{parse_engines, Cli, Command, EngineSpec, ServeArgs};
pub use client::{Client, Sample};
pub use metrics::CaseMetrics;
pub use runner::run_serve;
pub use workload::{RequestSpec, Workload};

#[cfg(test)]
pub(crate) fn test_args() -> ServeArgs {
    use clap::Parser;
    Cli::parse_from([
        "megalm-bench",
        "serve",
        "--model",
        "test-model",
        "--base-url",
        "http://127.0.0.1:8000",
    ])
    .command
    .into_serve()
}
