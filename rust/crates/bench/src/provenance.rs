//! Everything needed to reproduce or reject a published number.  Unlike the
//! upstream benchmark, seed, commit, source digests and the exact request set
//! are captured automatically rather than left to the caller's memory.

use std::collections::BTreeMap;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::{SystemTime, UNIX_EPOCH};

use serde_json::{json, Value};
use sha2::{Digest, Sha256};

use crate::cli::{EngineSpec, ServeArgs};
use crate::workload::{hex, Workload};

pub fn repo_root() -> PathBuf {
    let start = std::env::current_dir().unwrap_or_else(|_| PathBuf::from("."));
    for candidate in start.ancestors() {
        if candidate.join(".git").exists() {
            return candidate.to_path_buf();
        }
    }
    start
}

fn git_output(root: &Path, args: &[&str]) -> Option<String> {
    let output = Command::new("git")
        .args(args)
        .current_dir(root)
        .output()
        .ok()?;
    if !output.status.success() {
        return None;
    }
    Some(String::from_utf8_lossy(&output.stdout).trim().to_owned())
}

fn command_line() -> Vec<String> {
    let mut redact_next = false;
    let mut redact_header = false;
    std::env::args()
        .map(|arg| {
            if redact_next {
                redact_next = false;
                return "***".to_owned();
            }
            if redact_header {
                redact_header = false;
                let name = arg.split('=').next().unwrap_or("").to_ascii_lowercase();
                return if name.contains("token") || name.contains("key") || name.contains("secret")
                {
                    format!("{name}=***")
                } else {
                    arg
                };
            }
            if arg == "--api-key" {
                redact_next = true;
                return arg;
            }
            if arg == "--header" {
                redact_header = true;
                return arg;
            }
            if let Some(value) = arg.strip_prefix("--header=") {
                let name = value.split('=').next().unwrap_or("").to_ascii_lowercase();
                if name.contains("token") || name.contains("key") || name.contains("secret") {
                    return format!("--header={name}=***");
                }
            }
            arg
        })
        .collect()
}

fn digest_file(path: &Path) -> Option<String> {
    let bytes = fs::read(path).ok()?;
    Some(hex(&Sha256::digest(&bytes)))
}

fn collect_digests(root: &Path, paths: &[PathBuf]) -> BTreeMap<String, String> {
    let mut out = BTreeMap::new();
    let mut queue: Vec<PathBuf> = paths
        .iter()
        .map(|path| {
            if path.is_absolute() {
                path.clone()
            } else {
                root.join(path)
            }
        })
        .collect();
    while let Some(path) = queue.pop() {
        let Ok(metadata) = fs::symlink_metadata(&path) else {
            continue;
        };
        if metadata.is_dir() {
            let name = path
                .file_name()
                .and_then(|name| name.to_str())
                .unwrap_or("");
            if matches!(name, "target" | ".git" | "__pycache__" | "node_modules") {
                continue;
            }
            if let Ok(entries) = fs::read_dir(&path) {
                for entry in entries.flatten() {
                    queue.push(entry.path());
                }
            }
            continue;
        }
        if !metadata.is_file() {
            continue;
        }
        let name = path
            .strip_prefix(root)
            .unwrap_or(&path)
            .to_string_lossy()
            .replace('\\', "/");
        if let Some(digest) = digest_file(&path) {
            out.insert(name, digest);
        }
    }
    out
}

/// ISO-8601 UTC timestamp without a datetime dependency.
pub fn utc_now() -> String {
    let seconds = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs() as i64;
    let days = seconds.div_euclid(86_400);
    let within = seconds.rem_euclid(86_400);
    let (year, month, day) = civil_from_days(days);
    format!(
        "{year:04}-{month:02}-{day:02}T{:02}:{:02}:{:02}Z",
        within / 3600,
        (within % 3600) / 60,
        within % 60
    )
}

fn civil_from_days(days: i64) -> (i64, i64, i64) {
    let z = days + 719_468;
    let era = if z >= 0 { z } else { z - 146_096 } / 146_097;
    let day_of_era = z - era * 146_097;
    let year_of_era =
        (day_of_era - day_of_era / 1460 + day_of_era / 36_524 - day_of_era / 146_096) / 365;
    let year = year_of_era + era * 400;
    let day_of_year = day_of_era - (365 * year_of_era + year_of_era / 4 - year_of_era / 100);
    let month_prime = (5 * day_of_year + 2) / 153;
    let day = day_of_year - (153 * month_prime + 2) / 5 + 1;
    let month = month_prime + if month_prime < 10 { 3 } else { -9 };
    (year + if month <= 2 { 1 } else { 0 }, month, day)
}

fn gpu() -> Value {
    let output = Command::new("nvidia-smi")
        .args([
            "--query-gpu=name,driver_version,memory.total",
            "--format=csv,noheader",
        ])
        .output();
    match output {
        Ok(output) if output.status.success() => {
            let text = String::from_utf8_lossy(&output.stdout);
            let lines: Vec<&str> = text
                .lines()
                .map(str::trim)
                .filter(|line| !line.is_empty())
                .collect();
            json!(lines)
        }
        _ => Value::Null,
    }
}

fn tool_version() -> String {
    let output = Command::new("rustc").arg("--version").output();
    let rustc = match output {
        Ok(output) if output.status.success() => {
            String::from_utf8_lossy(&output.stdout).trim().to_owned()
        }
        _ => "unknown".to_owned(),
    };
    format!("megalm-bench {} / {rustc}", env!("CARGO_PKG_VERSION"))
}

pub fn collect(
    args: &ServeArgs,
    engines: &[EngineSpec],
    workload: &Workload,
) -> anyhow::Result<Value> {
    let root = repo_root();
    let sources = if args.source_path.is_empty() {
        vec![
            PathBuf::from("rust/crates/bench/src"),
            PathBuf::from("rust/Cargo.toml"),
            PathBuf::from("rust/Cargo.lock"),
        ]
    } else {
        args.source_path.clone()
    };
    let mut config = args.metadata()?;
    config.insert("endpoint".to_owned(), args.endpoint.path().to_owned());
    config.insert(
        "dataset".to_owned(),
        workload.description["dataset"]
            .as_str()
            .unwrap_or("")
            .to_owned(),
    );
    config.insert("temperature".to_owned(), args.temperature.to_string());
    config.insert("ignore_eos".to_owned(), args.ignore_eos.to_string());
    config.insert(
        "stream_options.include_usage".to_owned(),
        (!args.no_include_usage).to_string(),
    );
    config.insert(
        "request_rate".to_owned(),
        if args.request_rate.is_finite() {
            args.request_rate.to_string()
        } else {
            "inf".to_owned()
        },
    );
    config.insert(
        "burstiness".to_owned(),
        if args.burstiness.is_finite() {
            args.burstiness.to_string()
        } else {
            "inf".to_owned()
        },
    );
    config.insert(
        "max_queue_wait_ms".to_owned(),
        args.max_queue_wait_ms.to_string(),
    );
    config.insert("verify_output".to_owned(), args.verify_output.to_string());
    config.insert("num_prompts".to_owned(), args.num_prompts.to_string());
    config.insert("trials".to_owned(), args.trials.to_string());
    config.insert("warmup".to_owned(), args.warmup.to_string());
    config.insert(
        "max_concurrency".to_owned(),
        args.max_inflight()?.to_string(),
    );
    config.insert(
        "concurrency".to_owned(),
        args.concurrency()?
            .iter()
            .map(usize::to_string)
            .collect::<Vec<_>>()
            .join(","),
    );

    let dirty = git_output(&root, &["status", "--porcelain"])
        .map(|text| text.lines().count())
        .unwrap_or(0);
    Ok(json!({
        "schema_version": 1,
        "timestamp_utc": utc_now(),
        "command": command_line(),
        "tool": tool_version(),
        "git": {
            "commit": git_output(&root, &["rev-parse", "HEAD"]),
            "branch": git_output(&root, &["rev-parse", "--abbrev-ref", "HEAD"]),
            "dirty_entries": dirty,
        },
        "host": {
            "os": std::env::consts::OS,
            "arch": std::env::consts::ARCH,
            "gpu": gpu(),
        },
        "engines": engines
            .iter()
            .map(|engine| json!({"name": engine.name, "url": engine.url}))
            .collect::<Vec<_>>(),
        "config": config,
        "source_digests": collect_digests(&root, &sources),
        "workload": {
            "digest": workload.digest(),
            "details": workload.description,
        },
    }))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cli::parse_engines;

    #[test]
    fn civil_dates_round_trip_epoch() {
        assert_eq!(civil_from_days(0), (1970, 1, 1));
        assert_eq!(civil_from_days(20_454), (2026, 1, 1));
    }

    #[test]
    fn provenance_captures_seed_and_sources() {
        let args = crate::test_args();
        let engines = parse_engines(&args).unwrap();
        let workload = crate::workload::build(
            crate::cli::DatasetKind::Synthetic,
            None,
            2,
            16,
            4,
            None,
            123,
        )
        .unwrap();
        let report = collect(&args, &engines, &workload).unwrap();
        assert_eq!(report["workload"]["details"]["seed"], 123);
        assert!(!report["source_digests"].as_object().unwrap().is_empty());
        assert_eq!(report["schema_version"], 1);
    }
}
