//! Command-line surface of the benchmark.  Flags intentionally mirror
//! `vllm bench serve` where the meanings match, so existing commands translate.

use std::collections::BTreeMap;
use std::path::PathBuf;

use anyhow::{anyhow, bail, Result};
use clap::{Args, Parser, Subcommand, ValueEnum};
use serde::{Deserialize, Serialize};

#[derive(Parser, Debug)]
#[command(
    name = "megalm-bench",
    version,
    about = "Serving benchmark for OpenAI-compatible inference endpoints"
)]
pub struct Cli {
    #[command(subcommand)]
    pub command: Command,
}

#[derive(Subcommand, Debug)]
pub enum Command {
    /// Drive one or more servers and save a JSON report.
    Serve(Box<ServeArgs>),
    /// Render a comparison table from a saved report.
    Report(ReportArgs),
}

#[derive(ValueEnum, Clone, Copy, Debug, PartialEq, Eq)]
pub enum ReportFormat {
    Markdown,
    Json,
    Summary,
}

impl Command {
    /// Helper for tests that only need a serving configuration.
    pub fn into_serve(self) -> ServeArgs {
        match self {
            Self::Serve(args) => *args,
            Self::Report(_) => panic!("expected the serve subcommand"),
        }
    }
}

#[derive(Args, Debug)]
pub struct ReportArgs {
    /// JSON report written by `serve`.
    #[arg(long, value_name = "FILE")]
    pub input: PathBuf,
    #[arg(long, value_enum, default_value = "markdown")]
    pub format: ReportFormat,
    /// Write the rendered report to a file instead of stdout.
    #[arg(long, value_name = "FILE")]
    pub output: Option<PathBuf>,
}

#[derive(Args, Debug)]
pub struct ServeArgs {
    /// Server under test, as NAME=URL. Repeat for a sequential head-to-head run.
    #[arg(long, value_name = "NAME=URL")]
    pub engine: Vec<String>,
    /// Single-server shorthand when --engine is absent.
    #[arg(long, default_value = "http://127.0.0.1:8000")]
    pub base_url: String,
    /// Name used with --base-url.
    #[arg(long, default_value = "default")]
    pub name: String,
    /// Served model id sent in the request body.
    #[arg(long)]
    pub model: String,
    /// OpenAI-compatible endpoint family.
    #[arg(long, value_enum, default_value = "chat")]
    pub endpoint: EndpointKind,
    /// Bearer token for the endpoint.
    #[arg(long)]
    pub api_key: Option<String>,
    /// Extra request header as KEY=VALUE. Repeatable.
    #[arg(long, value_name = "KEY=VALUE")]
    pub header: Vec<String>,
    /// Extra JSON object merged into every request body. Repeatable.
    #[arg(long, value_name = "JSON")]
    pub extra_body: Vec<String>,
    /// Per-request timeout in seconds.
    #[arg(long, default_value_t = 300)]
    pub timeout_secs: u64,
    /// Poll /v1/models for up to this many seconds before measuring (0 skips).
    #[arg(long, default_value_t = 0)]
    pub ready_timeout_secs: u64,

    /// Request set: synthetic text, JSONL records, or ShareGPT conversations.
    #[arg(long, value_enum, default_value = "synthetic")]
    pub dataset: DatasetKind,
    /// Path for jsonl/sharegpt datasets.
    #[arg(long, value_name = "FILE")]
    pub dataset_path: Option<PathBuf>,
    /// Target prompt length for synthetic requests.
    #[arg(long, default_value_t = 512)]
    pub prompt_tokens: usize,
    /// Tokenizer file (or directory containing tokenizer.json) for exact lengths.
    #[arg(long, value_name = "FILE")]
    pub tokenizer: Option<PathBuf>,
    /// Output token budget per request.
    #[arg(long, default_value_t = 128)]
    pub max_tokens: u32,
    /// Measured requests per trial.
    #[arg(long, default_value_t = 64)]
    pub num_prompts: usize,
    /// Random seed for dataset sampling and arrival delays.
    #[arg(long, default_value_t = 0)]
    pub seed: u64,
    /// Force the full output budget by ignoring end-of-sequence tokens.
    #[arg(long)]
    pub ignore_eos: bool,
    /// Sampling temperature (use 0 for greedy comparisons).
    #[arg(long, default_value_t = 0.0)]
    pub temperature: f32,

    /// Comma-separated concurrency levels, e.g. 1,8,16.
    #[arg(long, default_value = "1,8,16")]
    pub concurrency: String,
    /// Measured trials per concurrency level; the median is reported.
    #[arg(long, default_value_t = 1)]
    pub trials: usize,
    /// Unmeasured warmup requests before the first trial.
    #[arg(long, default_value_t = 1)]
    pub warmup: usize,
    /// Requests per second; `inf` means closed-loop (all requests queued at once).
    #[arg(long, default_value = "inf")]
    pub request_rate: f64,
    /// Burstiness of the arrival process: 1 = Poisson, inf = constant spacing.
    #[arg(long, default_value_t = 1.0)]
    pub burstiness: f64,
    /// Cap on in-flight requests; 0 means the largest concurrency level.
    #[arg(long, default_value_t = 0)]
    pub max_concurrency: usize,

    /// Goodput SLO: maximum time to first token in ms (0 disables).
    #[arg(long, default_value_t = 0.0)]
    pub slo_ttft_ms: f64,
    /// Goodput SLO: maximum time per output token in ms (0 disables).
    #[arg(long, default_value_t = 0.0)]
    pub slo_tpot_ms: f64,
    /// Goodput SLO: maximum end-to-end latency in ms (0 disables).
    #[arg(long, default_value_t = 0.0)]
    pub slo_e2e_ms: f64,
    /// Client-side p95 queue wait above which `load_generation_valid` is false.
    #[arg(long, default_value_t = 1000.0)]
    pub max_queue_wait_ms: f64,

    /// Do not request server-side token usage; count content chunks instead.
    #[arg(long)]
    pub no_include_usage: bool,
    /// Keep per-request samples in the JSON report.
    #[arg(long)]
    pub save_detailed: bool,
    /// Compare greedy outputs across engines and record a digest per engine.
    #[arg(long)]
    pub verify_output: bool,
    /// Return a non-zero exit code when verified outputs differ.
    #[arg(long)]
    pub fail_on_output_mismatch: bool,

    /// Overwrite existing report files instead of refusing.
    #[arg(long)]
    pub force: bool,
    /// JSON report destination.
    #[arg(long, value_name = "FILE")]
    pub output: Option<PathBuf>,
    /// Human-readable comparison table destination.
    #[arg(long, value_name = "FILE")]
    pub markdown: Option<PathBuf>,
    /// Provenance entry as KEY=VALUE. Repeatable.
    #[arg(long, value_name = "KEY=VALUE", alias = "config")]
    pub metadata: Vec<String>,
    /// File whose SHA-256 records the exact source under test. Repeatable.
    #[arg(long, value_name = "FILE")]
    pub source_path: Vec<PathBuf>,
    /// Validate arguments and print the workload summary without contacting servers.
    #[arg(long)]
    pub dry_run: bool,
    /// Suppress progress lines.
    #[arg(long)]
    pub quiet: bool,
}

#[derive(ValueEnum, Clone, Copy, Debug, PartialEq, Eq)]
pub enum EndpointKind {
    Chat,
    Completions,
}

impl EndpointKind {
    pub fn path(self) -> &'static str {
        match self {
            Self::Chat => "/v1/chat/completions",
            Self::Completions => "/v1/completions",
        }
    }
}

#[derive(ValueEnum, Clone, Copy, Debug, PartialEq, Eq)]
pub enum DatasetKind {
    Synthetic,
    Jsonl,
    Sharegpt,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct EngineSpec {
    pub name: String,
    pub url: String,
}

pub fn parse_engines(args: &ServeArgs) -> Result<Vec<EngineSpec>> {
    let mut engines = Vec::new();
    if args.engine.is_empty() {
        engines.push(EngineSpec {
            name: args.name.clone(),
            url: args.base_url.clone(),
        });
    } else {
        for entry in &args.engine {
            let (name, url) = entry
                .split_once('=')
                .ok_or_else(|| anyhow!("--engine expects NAME=URL, got {entry:?}"))?;
            if name.is_empty() || url.is_empty() {
                bail!("--engine expects non-empty NAME and URL, got {entry:?}");
            }
            if engines.iter().any(|existing| existing.name == name) {
                bail!("duplicate engine name {name:?}");
            }
            engines.push(EngineSpec {
                name: name.to_owned(),
                url: url.to_owned(),
            });
        }
    }
    for engine in &engines {
        if !(engine.url.starts_with("http://") || engine.url.starts_with("https://")) {
            bail!(
                "engine URL must start with http:// or https://: {}",
                engine.url
            );
        }
    }
    Ok(engines)
}

pub fn parse_positive_list(value: &str, flag: &str) -> Result<Vec<usize>> {
    let items: Vec<usize> = value
        .split(',')
        .map(|part| part.trim().parse::<usize>())
        .collect::<std::result::Result<_, _>>()
        .map_err(|_| anyhow!("--{flag} must be a comma-separated list of positive integers"))?;
    if items.is_empty() || items.contains(&0) {
        bail!("--{flag} must be a comma-separated list of positive integers");
    }
    Ok(items)
}

fn parse_pairs(items: &[String], flag: &str, sep: char) -> Result<BTreeMap<String, String>> {
    let mut out = BTreeMap::new();
    for item in items {
        let (key, value) = item
            .split_once(sep)
            .ok_or_else(|| anyhow!("--{flag} expects KEY{sep}VALUE, got {item:?}"))?;
        if key.trim().is_empty() {
            bail!("--{flag} has an empty key in {item:?}");
        }
        out.insert(key.trim().to_owned(), value.trim().to_owned());
    }
    Ok(out)
}

impl ServeArgs {
    pub fn concurrency(&self) -> Result<Vec<usize>> {
        parse_positive_list(&self.concurrency, "concurrency")
    }

    pub fn max_inflight(&self) -> Result<usize> {
        if self.max_concurrency > 0 {
            return Ok(self.max_concurrency);
        }
        Ok(self.concurrency()?.into_iter().max().unwrap_or(1))
    }

    pub fn metadata(&self) -> Result<BTreeMap<String, String>> {
        parse_pairs(&self.metadata, "metadata", '=')
    }

    pub fn headers(&self) -> Result<Vec<(String, String)>> {
        let mut out = Vec::new();
        for item in &self.header {
            let (key, value) = item
                .split_once('=')
                .ok_or_else(|| anyhow!("--header expects KEY=VALUE, got {item:?}"))?;
            if key.trim().is_empty() {
                bail!("--header has an empty key in {item:?}");
            }
            out.push((key.trim().to_owned(), value.trim().to_owned()));
        }
        Ok(out)
    }

    pub fn extra_body(&self) -> Result<serde_json::Map<String, serde_json::Value>> {
        let mut merged = serde_json::Map::new();
        for raw in &self.extra_body {
            let value: serde_json::Value = serde_json::from_str(raw)
                .map_err(|err| anyhow!("--extra-body must be a JSON object: {err}"))?;
            let object = value
                .as_object()
                .ok_or_else(|| anyhow!("--extra-body must be a JSON object"))?;
            for (key, value) in object {
                merged.insert(key.clone(), value.clone());
            }
        }
        Ok(merged)
    }

    pub fn validate(&self) -> Result<()> {
        if self.model.trim().is_empty() {
            bail!("--model is required");
        }
        if self.num_prompts == 0 || self.max_tokens == 0 || self.trials == 0 {
            bail!("--num-prompts, --max-tokens and --trials must be positive");
        }
        if self.timeout_secs == 0 {
            bail!("--timeout-secs must be positive");
        }
        if self.request_rate < 0.0 || self.request_rate.is_nan() {
            bail!("--request-rate must be non-negative or inf");
        }
        if self.burstiness <= 0.0 || self.burstiness.is_nan() {
            bail!("--burstiness must be positive or inf");
        }
        if self.request_rate.is_finite() && self.request_rate == 0.0 {
            bail!("--request-rate must be positive or inf");
        }
        if matches!(self.dataset, DatasetKind::Jsonl | DatasetKind::Sharegpt)
            && self.dataset_path.is_none()
        {
            bail!("--dataset-path is required for this dataset");
        }
        let _ = self.concurrency()?;
        let _ = self.max_inflight()?;
        let _ = self.metadata()?;
        let _ = self.headers()?;
        let _ = self.extra_body()?;
        for value in [
            self.slo_ttft_ms,
            self.slo_tpot_ms,
            self.slo_e2e_ms,
            self.max_queue_wait_ms,
        ] {
            if value < 0.0 || value.is_nan() {
                bail!("goodput SLOs and --max-queue-wait-ms must be non-negative");
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn positive_lists_reject_zero_and_junk() {
        assert_eq!(
            parse_positive_list("1, 8,16", "concurrency").unwrap(),
            vec![1, 8, 16]
        );
        assert!(parse_positive_list("1,0", "concurrency").is_err());
        assert!(parse_positive_list("1,x", "concurrency").is_err());
        assert!(parse_positive_list("", "concurrency").is_err());
    }

    #[test]
    fn engine_specs_are_validated() {
        let mut args = crate::test_args();
        args.engine = vec!["a=http://127.0.0.1:1".into(), "b=http://127.0.0.1:2".into()];
        let engines = parse_engines(&args).unwrap();
        assert_eq!(engines[1].name, "b");
        args.engine = vec!["broken".into()];
        assert!(parse_engines(&args).is_err());
        args.engine = vec!["a=ftp://host".into()];
        assert!(parse_engines(&args).is_err());
    }
}
