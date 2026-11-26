//! Trial orchestration: one seeded workload, one pooled client per engine,
//! readiness + warmup, then a concurrency × trial matrix.

use std::collections::BTreeMap;
use std::fs;
use std::path::Path;
use std::sync::Arc;
use std::time::{Duration, Instant};

use anyhow::{anyhow, bail, Result};
use futures::future::join_all;
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use tokio::sync::Semaphore;

use crate::cli::{parse_engines, ServeArgs};
use crate::client::{elapsed_ms, Client, Sample};
use crate::metrics::{median_value, summarize, Slo};
use crate::provenance;
use crate::report::render_markdown;
use crate::workload::{build, hex, RequestSpec, Seeded};

pub async fn run_serve(args: ServeArgs) -> Result<()> {
    args.validate()?;
    let engines = parse_engines(&args)?;
    let workload = build(
        args.dataset,
        args.dataset_path.as_deref(),
        args.num_prompts,
        args.prompt_tokens,
        args.max_tokens,
        args.tokenizer.as_deref(),
        args.seed,
    )?;
    let concurrency = args.concurrency()?;
    let max_inflight = args.max_inflight()?;

    if args.dry_run {
        println!(
            "{}",
            serde_json::to_string_pretty(&json!({
                "engines": engines,
                "workload": workload.description,
                "workload_digest": workload.digest(),
                "concurrency": concurrency,
                "trials": args.trials,
                "max_queue_wait_ms": args.max_queue_wait_ms,
                "request_rate": if args.request_rate.is_finite() {
                    json!(args.request_rate)
                } else {
                    json!("inf")
                },
                "burstiness": if args.burstiness.is_finite() {
                    json!(args.burstiness)
                } else {
                    json!("inf")
                },
            }))?
        );
        return Ok(());
    }

    for path in [args.output.as_deref(), args.markdown.as_deref()]
        .into_iter()
        .flatten()
    {
        if path.exists() && !args.force {
            bail!(
                "{} already exists; pass --force to overwrite",
                path.display()
            );
        }
    }

    let provenance = provenance::collect(&args, &engines, &workload)?;
    let mut report = json!({
        "schema_version": 1,
        "provenance": provenance,
        "results": {},
    });

    let mut output_digests = BTreeMap::new();
    let mut verification_source: Option<Value> = None;
    for engine in &engines {
        if !args.quiet {
            eprintln!("== {} @ {} ==", engine.name, engine.url);
        }
        let client = Arc::new(Client::new(&args, &engine.url, max_inflight)?);
        if args.ready_timeout_secs > 0 {
            wait_ready(&client, Duration::from_secs(args.ready_timeout_secs)).await?;
        }
        warmup(&client, &workload.requests, &args).await;

        let mut rows = Vec::new();
        for level in &concurrency {
            let mut trials = Vec::new();
            for trial in 0..args.trials {
                // Trial index perturbs only the arrival seed, never the shared
                // workload, so outputs remain comparable across engines.
                let trial_seed = args.seed.wrapping_add(trial as u64);
                let (samples, duration_s, offered_rps) = run_case(
                    &client,
                    &workload.requests,
                    *level,
                    max_inflight,
                    &args,
                    trial_seed,
                )
                .await?;
                if args.verify_output && trial == 0 && *level == concurrency[0] {
                    let mut joined = String::new();
                    for sample in &samples {
                        joined.push_str(&sample.request_id);
                        joined.push('\u{0}');
                        joined.push_str(&sample.text);
                        joined.push('\n');
                    }
                    output_digests
                        .insert(engine.name.clone(), hex(&Sha256::digest(joined.as_bytes())));
                    verification_source = Some(json!({
                        "concurrency": level,
                        "trial": trial,
                        "seed": trial_seed,
                    }));
                }
                let metrics = summarize(
                    &samples,
                    duration_s,
                    offered_rps,
                    Slo {
                        ttft_ms: args.slo_ttft_ms,
                        tpot_ms: args.slo_tpot_ms,
                        e2el_ms: args.slo_e2e_ms,
                        max_tokens: args.max_tokens,
                        ignore_eos: args.ignore_eos,
                    },
                    if args.request_rate.is_finite() {
                        Some(args.max_queue_wait_ms)
                    } else {
                        None
                    },
                );
                let mut row = serde_json::to_value(metrics)?;
                row["seed"] = json!(trial_seed);
                row["queue_wait_limit_ms"] = if args.request_rate.is_finite() {
                    json!(args.max_queue_wait_ms)
                } else {
                    json!(null)
                };
                if args.save_detailed {
                    row["samples"] = serde_json::to_value(&samples)?;
                }
                if !args.quiet {
                    eprintln!(
                        "  c={level} trial={trial}: {:.1} out tok/s, p95 TTFT {:.1} ms, {} failed",
                        row["output_throughput"].as_f64().unwrap_or(0.0),
                        row["ttft_ms"]["p95"].as_f64().unwrap_or(0.0),
                        row["failed"].as_u64().unwrap_or(0),
                    );
                }
                trials.push(row);
            }
            let median = median_value(&trials);
            if !args.quiet {
                eprintln!(
                    "  c={level} median: {:.1} out tok/s, {} req/s",
                    median["output_throughput"].as_f64().unwrap_or(0.0),
                    median["achieved_rps"].as_f64().unwrap_or(0.0),
                );
            }
            rows.push(json!({
                "concurrency": level,
                "trials": trials,
                "median": median,
            }));
        }
        report["results"][engine.name.clone()] = json!(rows);
    }

    if args.verify_output && output_digests.len() > 1 {
        let first = output_digests.values().next().cloned().unwrap_or_default();
        let matches = output_digests.values().all(|digest| digest == &first);
        if !matches && !args.quiet {
            eprintln!("greedy outputs differ across engines: {output_digests:?}");
        }
        report["output_verification"] = json!({
            "enabled": true,
            "source": verification_source,
            "digests": output_digests,
            "match": matches,
        });
        if !matches && args.fail_on_output_mismatch {
            bail!("greedy outputs differ across engines");
        }
    }

    let text = serde_json::to_string_pretty(&report)?;
    let markdown = render_markdown(&report);
    if let Some(path) = &args.output {
        write_file(path, &format!("{text}\n"), args.force)?;
        if !args.quiet {
            eprintln!("wrote {}", path.display());
        }
    } else {
        println!("{text}");
    }
    if let Some(path) = &args.markdown {
        write_file(path, &markdown, args.force)?;
        if !args.quiet {
            eprintln!("wrote {}", path.display());
        }
    }
    Ok(())
}

fn write_file(path: &Path, text: &str, force: bool) -> Result<()> {
    if path.exists() && !force {
        bail!(
            "{} already exists; pass --force to overwrite",
            path.display()
        );
    }
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    fs::write(path, text)?;
    Ok(())
}

async fn wait_ready(client: &Client, timeout: Duration) -> Result<()> {
    let started = Instant::now();
    loop {
        if client.ready(Duration::from_millis(1000)).await.is_ok() {
            return Ok(());
        }
        if started.elapsed() >= timeout {
            return Err(anyhow!("server was not ready within {:?}", timeout));
        }
        tokio::time::sleep(Duration::from_millis(500)).await;
    }
}

async fn warmup(client: &Client, requests: &[RequestSpec], args: &ServeArgs) {
    for index in 0..args.warmup {
        let spec = &requests[index % requests.len()];
        let base = Instant::now();
        let sample = client
            .stream(
                spec,
                args.max_tokens,
                args.temperature,
                args.ignore_eos,
                base,
                0.0,
            )
            .await;
        if !sample.error.is_empty() && !args.quiet {
            eprintln!("  warmup {} failed: {}", index, sample.error);
        }
    }
}

/// Inter-arrival delays, normalized like `vllm bench serve` so a fixed rate
/// gives stable throughput across seeds.  Closed loop (`inf`) queues all
/// requests at once behind the concurrency semaphore.
fn arrivals(count: usize, rate: f64, burstiness: f64, seed: u64) -> Vec<f64> {
    if !rate.is_finite() {
        return vec![0.0; count];
    }
    let mut rng = Seeded::new(seed);
    let mut offsets = Vec::with_capacity(count);
    let mut total = 0.0;
    for _ in 0..count {
        let delay = if burstiness.is_infinite() {
            1.0 / rate
        } else {
            rng.gamma(burstiness) / (rate * burstiness)
        };
        total += delay;
        offsets.push(total);
    }
    let target = count as f64 / rate;
    if total > 0.0 {
        let scale = target / total;
        for offset in &mut offsets {
            *offset *= scale;
        }
    }
    offsets
}

async fn run_case(
    client: &Arc<Client>,
    requests: &[RequestSpec],
    concurrency: usize,
    max_inflight: usize,
    args: &ServeArgs,
    seed: u64,
) -> Result<(Vec<Sample>, f64, f64)> {
    let base = Instant::now();
    let permits = Arc::new(Semaphore::new(concurrency.min(max_inflight).max(1)));
    let offsets = arrivals(requests.len(), args.request_rate, args.burstiness, seed);
    let mut tasks = Vec::with_capacity(requests.len());

    for (index, spec) in requests.iter().enumerate() {
        let client = Arc::clone(client);
        let permits = Arc::clone(&permits);
        let spec = spec.clone();
        let delay = offsets[index];
        let max_tokens = args.max_tokens;
        let temperature = args.temperature;
        let ignore_eos = args.ignore_eos;
        tasks.push(tokio::spawn(async move {
            if delay > 0.0 {
                tokio::time::sleep(Duration::from_secs_f64(delay)).await;
            }
            let queued_ms = elapsed_ms(base);
            let _permit = permits.acquire().await.expect("semaphore is never closed");
            client
                .stream(&spec, max_tokens, temperature, ignore_eos, base, queued_ms)
                .await
        }));
    }

    let samples = join_all(tasks)
        .await
        .into_iter()
        .map(|result| result.map_err(|err| anyhow!("request task panicked: {err}")))
        .collect::<Result<Vec<_>>>()?;
    if samples.is_empty() {
        return Err(anyhow!("no samples collected"));
    }
    let first = samples
        .iter()
        .map(|sample| sample.start_ms)
        .fold(f64::INFINITY, f64::min);
    let last = samples
        .iter()
        .map(|sample| sample.end_ms)
        .fold(f64::NEG_INFINITY, f64::max);
    let duration_s = ((last - first) / 1000.0).max(1e-6);
    let offered_rps = if args.request_rate.is_finite() {
        args.request_rate
    } else {
        samples.len() as f64 / duration_s
    };
    Ok((samples, duration_s, offered_rps))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn closed_loop_arrivals_are_immediate() {
        assert_eq!(arrivals(4, f64::INFINITY, 1.0, 0), vec![0.0, 0.0, 0.0, 0.0]);
    }

    #[test]
    fn constant_rate_arrivals_are_normalized() {
        let offsets = arrivals(10, 10.0, f64::INFINITY, 0);
        assert!((offsets.last().copied().unwrap_or(0.0) - 1.0).abs() < 1e-9);
        for pair in offsets.windows(2) {
            assert!((pair[1] - pair[0] - 0.1).abs() < 1e-9);
        }
    }

    #[test]
    fn poisson_arrivals_are_reproducible() {
        assert_eq!(arrivals(8, 5.0, 1.0, 11), arrivals(8, 5.0, 1.0, 11));
    }
}
