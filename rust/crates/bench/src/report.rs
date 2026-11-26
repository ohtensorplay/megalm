//! Human-readable rendering of saved reports.  The comparison is computed
//! from the median trial at each concurrency level, never from cherry-picked
//! single runs.

use std::fs;
use std::path::Path;

use anyhow::{bail, Result};
use serde_json::{json, Value};

use crate::cli::{ReportArgs, ReportFormat};

fn at<'a>(value: &'a Value, path: &[&str]) -> Option<&'a Value> {
    path.iter()
        .try_fold(value, |current, key| current.get(*key))
}

fn number(value: &Value, path: &[&str]) -> f64 {
    at(value, path).and_then(Value::as_f64).unwrap_or(0.0)
}

fn count(value: &Value, path: &[&str]) -> u64 {
    at(value, path).and_then(Value::as_u64).unwrap_or(0)
}

fn text(value: &Value, path: &[&str]) -> String {
    at(value, path)
        .and_then(Value::as_str)
        .unwrap_or("")
        .to_owned()
}

fn rows(report: &Value) -> Vec<(String, Value)> {
    let mut out = Vec::new();
    if let Some(results) = report.get("results").and_then(Value::as_object) {
        for (engine, levels) in results {
            if let Some(levels) = levels.as_array() {
                for level in levels {
                    out.push((engine.clone(), level.clone()));
                }
            }
        }
    }
    out
}

fn summary_row(engine: &str, level: &Value) -> Value {
    let median = level.get("median").unwrap_or(&Value::Null);
    json!({
        "engine": engine,
        "concurrency": level.get("concurrency").cloned().unwrap_or(Value::Null),
        "completed": count(median, &["completed"]),
        "failed": count(median, &["failed"]),
        "duration_s": number(median, &["duration_s"]),
        "offered_rps": number(median, &["offered_rps"]),
        "achieved_rps": number(median, &["achieved_rps"]),
        "goodput_rps": number(median, &["goodput_rps"]),
        "output_throughput": number(median, &["output_throughput"]),
        "peak_output_throughput": number(median, &["peak_output_throughput"]),
        "ttft_p50_ms": number(median, &["ttft_ms", "p50"]),
        "ttft_std_ms": number(median, &["ttft_ms", "std"]),
        "ttft_p95_ms": number(median, &["ttft_ms", "p95"]),
        "e2el_p95_ms": number(median, &["e2el_ms", "p95"]),
        "tpot_mean_ms": number(median, &["tpot_ms", "mean"]),
        "itl_p95_ms": number(median, &["itl_ms", "p95"]),
        "itl_complete": at(median, &["itl_complete"]).and_then(Value::as_bool).unwrap_or(false),
        "token_source": text(median, &["token_source"]),
        "load_generation_valid": at(median, &["load_generation_valid"]).and_then(Value::as_bool).unwrap_or(false),
        "load_generation_check": text(median, &["load_generation_check"]),
    })
}

pub fn render_markdown(report: &Value) -> String {
    let summary: Vec<Value> = rows(report)
        .iter()
        .map(|(engine, level)| summary_row(engine, level))
        .collect();
    let mut out = String::new();
    out.push_str("# Serving benchmark\n\n");
    let commit = text(report, &["provenance", "git", "commit"]);
    if !commit.is_empty() {
        out.push_str(&format!(
            "- commit: `{}` (dirty entries: {})\n",
            &commit[..commit.len().min(12)],
            count(report, &["provenance", "git", "dirty_entries"])
        ));
    }
    let digest = text(report, &["provenance", "workload", "digest"]);
    if !digest.is_empty() {
        out.push_str(&format!(
            "- workload digest: `{}`\n",
            &digest[..digest.len().min(16)]
        ));
    }
    out.push_str(&format!(
        "- endpoint: {}, percentile method: {}\n\n",
        text(report, &["provenance", "config", "endpoint"]),
        crate::metrics::PERCENTILE_METHOD
    ));
    // Surface the caveats next to the numbers instead of leaving them buried
    // in JSON: a published table must be self-checking.
    let mut warnings = Vec::new();
    if summary.iter().any(|row| {
        row["load_generation_check"].as_str().unwrap_or("") == "open_loop"
            && !row["load_generation_valid"].as_bool().unwrap_or(false)
    }) {
        warnings.push("client load generation was saturated (queue wait above the limit); throughput is not a capacity result");
    }
    if summary
        .iter()
        .any(|row| !row["token_source"].as_str().unwrap_or("").contains("usage"))
    {
        warnings.push("some requests had no server-reported token usage; output tokens were counted from chunks")
    }
    if summary
        .iter()
        .any(|row| !row["itl_complete"].as_bool().unwrap_or(false))
    {
        warnings.push("ITL is chunk-interval only (multiple tokens per chunk); do not report it as per-token latency")
    }
    if let Some(verification) = report.get("output_verification") {
        if verification["match"].as_bool() == Some(false) {
            warnings.push(
                "greedy outputs differ across engines; the comparison is not apples-to-apples",
            )
        }
    }
    if !warnings.is_empty() {
        out.push_str("> **Validity warnings:** ");
        out.push_str(&warnings.join("; "));
        out.push_str("\n\n");
    }
    out.push_str(
        "| engine | conc. | out tok/s | vs base | req/s | goodput/s | TTFT p50/p95 ms | E2EL p95 ms | TPOT mean ms | ITL p95 ms | peak tok/s | errors |\n",
    );
    out.push_str("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n");
    let mut baseline: std::collections::BTreeMap<u64, f64> = std::collections::BTreeMap::new();
    for row in &summary {
        let concurrency = row["concurrency"].as_u64().unwrap_or(0);
        let throughput = row["output_throughput"].as_f64().unwrap_or(0.0);
        let base = *baseline.entry(concurrency).or_insert(throughput);
        let ratio = if base > 0.0 { throughput / base } else { 0.0 };
        out.push_str(&format!(
            "| {} | {} | {:.1} | {:.2}x | {:.2} | {:.2} | {:.1}/{:.1} | {:.1} | {:.2} | {:.2} | {:.1} | {}/{} |\n",
            row["engine"].as_str().unwrap_or(""),
            concurrency,
            throughput,
            ratio,
            row["achieved_rps"].as_f64().unwrap_or(0.0),
            row["goodput_rps"].as_f64().unwrap_or(0.0),
            row["ttft_p50_ms"].as_f64().unwrap_or(0.0),
            row["ttft_p95_ms"].as_f64().unwrap_or(0.0),
            row["e2el_p95_ms"].as_f64().unwrap_or(0.0),
            row["tpot_mean_ms"].as_f64().unwrap_or(0.0),
            row["itl_p95_ms"].as_f64().unwrap_or(0.0),
            row["peak_output_throughput"].as_f64().unwrap_or(0.0),
            row["failed"].as_u64().unwrap_or(0),
            row["completed"].as_u64().unwrap_or(0) + row["failed"].as_u64().unwrap_or(0),
        ));
    }
    out
}

pub fn summary(report: &Value) -> Value {
    json!({"rows": rows(report).iter().map(|(engine, level)| summary_row(engine, level)).collect::<Vec<_>>()})
}

pub fn render_command(args: &ReportArgs) -> Result<()> {
    let text = fs::read_to_string(&args.input)?;
    let report: Value = serde_json::from_str(&text)?;
    let output = match args.format {
        ReportFormat::Markdown => render_markdown(&report),
        ReportFormat::Json => serde_json::to_string_pretty(&summary(&report))?,
        ReportFormat::Summary => summary(&report)
            .get("rows")
            .and_then(Value::as_array)
            .map(|rows| {
                rows.iter()
                    .map(|row| {
                        format!(
                            "{} c={} out={:.1} tok/s ttft_p95={:.1} ms errors={}",
                            row["engine"].as_str().unwrap_or(""),
                            row["concurrency"].as_u64().unwrap_or(0),
                            row["output_throughput"].as_f64().unwrap_or(0.0),
                            row["ttft_p95_ms"].as_f64().unwrap_or(0.0),
                            row["failed"].as_u64().unwrap_or(0),
                        )
                    })
                    .collect::<Vec<_>>()
                    .join("\n")
            })
            .unwrap_or_default(),
    };
    let rendered = if output.ends_with('\n') {
        output
    } else {
        format!("{output}\n")
    };
    if let Some(path) = &args.output {
        if path.exists() {
            bail!("{} already exists", path.display());
        }
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent)?;
        }
        fs::write(Path::new(path), rendered)?;
    } else {
        print!("{rendered}");
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn markdown_includes_comparison_and_provenance() {
        let report = json!({
            "provenance": {"git": {"commit": "abcdef123456", "dirty_entries": 2},
                           "workload": {"digest": "0123456789abcdef"}, "config": {"endpoint": "/v1/chat/completions"}},
            "results": {
                "a": [{"concurrency": 1, "median": {"output_throughput": 100.0, "achieved_rps": 1.0,
                    "ttft_ms": {"p50": 5.0, "p95": 9.0}, "e2el_ms": {"p95": 50.0},
                    "tpot_ms": {"mean": 2.0}, "itl_ms": {"p95": 2.5},
                    "peak_output_throughput": 110.0, "failed": 0, "completed": 10, "goodput_rps": 1.0}}],
                "b": [{"concurrency": 1, "median": {"output_throughput": 50.0, "achieved_rps": 0.5,
                    "ttft_ms": {"p50": 6.0, "p95": 10.0}, "e2el_ms": {"p95": 60.0},
                    "tpot_ms": {"mean": 4.0}, "itl_ms": {"p95": 4.5},
                    "peak_output_throughput": 55.0, "failed": 1, "completed": 9, "goodput_rps": 0.5}}],
            }
        });
        let rendered = render_markdown(&report);
        assert!(rendered.contains("abcdef123456"));
        assert!(rendered.contains("0.50x"));
        assert!(rendered.contains("| b | 1 | 50.0"));
    }

    #[test]
    fn markdown_flags_unusable_numbers() {
        let report = json!({
            "results": {"a": [{"concurrency": 1, "median": {
                "output_throughput": 10.0, "completed": 1, "failed": 0,
                "load_generation_valid": false, "load_generation_check": "open_loop",
                "token_source": "chunks",
                "itl_complete": false,
                "ttft_ms": {"p50": 1.0, "p95": 2.0, "std": 0.5},
            }}]},
            "output_verification": {"match": false},
        });
        let rendered = render_markdown(&report);
        assert!(rendered.contains("client load generation was saturated"));
        assert!(rendered.contains("counted from chunks"));
        assert!(rendered.contains("greedy outputs differ"));

        // Closed-loop queue wait is caused by the concurrency semaphore, not
        // by a saturated client; it must not raise a false validity warning.
        let closed = json!({
            "results": {"a": [{"concurrency": 1, "median": {
                "load_generation_valid": false,
                "load_generation_check": "closed_loop_not_applicable",
                "token_source": "usage", "itl_complete": true,
                "ttft_ms": {"p50": 1.0, "p95": 2.0},
            }}]},
        });
        assert!(!render_markdown(&closed).contains("load generation was saturated"));
    }
}
