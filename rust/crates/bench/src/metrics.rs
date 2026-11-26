//! Metric definitions follow `vllm bench serve`: TTFT, TPOT, ITL and E2EL with
//! numpy-style linear percentiles, plus SLO goodput and one-second peak gauges.
//! Every aggregate records its own sample count so partial telemetry is visible.

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};
use serde_json::{json, Value};

use crate::client::Sample;

pub const PERCENTILE_METHOD: &str = "linear_interpolation";

#[derive(Clone, Copy, Debug, Default, Serialize, Deserialize, PartialEq)]
pub struct Stats {
    pub n: usize,
    pub mean: f64,
    /// Population standard deviation (numpy `np.std`, ddof=0).
    pub std: f64,
    pub p50: f64,
    pub p90: f64,
    pub p95: f64,
    pub p99: f64,
}

pub fn round(value: f64, digits: i32) -> f64 {
    let factor = 10f64.powi(digits);
    (value * factor).round() / factor
}

/// Linear interpolation between order statistics, matching numpy.percentile.
pub fn percentile(sorted: &[f64], fraction: f64) -> f64 {
    if sorted.is_empty() {
        return 0.0;
    }
    if sorted.len() == 1 {
        return sorted[0];
    }
    let position = (sorted.len() - 1) as f64 * fraction.clamp(0.0, 1.0);
    let lower = position.floor() as usize;
    let upper = position.ceil() as usize;
    if lower == upper {
        sorted[lower]
    } else {
        sorted[lower] + (sorted[upper] - sorted[lower]) * (position - lower as f64)
    }
}

fn stats(mut values: Vec<f64>) -> Stats {
    values.retain(|value| value.is_finite());
    if values.is_empty() {
        return Stats::default();
    }
    values.sort_by(f64::total_cmp);
    let n = values.len();
    let mean = values.iter().sum::<f64>() / n as f64;
    let variance = values.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / n as f64;
    Stats {
        n,
        mean: round(mean, 3),
        std: round(variance.sqrt(), 3),
        p50: round(percentile(&values, 0.50), 3),
        p90: round(percentile(&values, 0.90), 3),
        p95: round(percentile(&values, 0.95), 3),
        p99: round(percentile(&values, 0.99), 3),
    }
}

#[derive(Clone, Copy, Debug, Default)]
pub struct Slo {
    pub ttft_ms: f64,
    pub tpot_ms: f64,
    pub e2el_ms: f64,
    pub max_tokens: u32,
    pub ignore_eos: bool,
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct CaseMetrics {
    pub duration_s: f64,
    pub completed: usize,
    pub failed: usize,
    pub offered_rps: f64,
    pub achieved_rps: f64,
    pub good_requests: usize,
    pub goodput_rps: f64,
    pub output_tokens: u64,
    pub input_tokens: u64,
    pub output_throughput: f64,
    pub input_throughput: f64,
    pub total_token_throughput: f64,
    pub peak_output_throughput: f64,
    pub peak_concurrency: usize,
    pub output_tokens_per_request: f64,
    pub ttft_ms: Stats,
    pub e2el_ms: Stats,
    pub tpot_ms: Stats,
    pub itl_ms: Stats,
    pub queue_wait_ms: Stats,
    pub token_source: String,
    pub itl_complete: bool,
    pub truncated_requests: usize,
    pub status_counts: BTreeMap<String, usize>,
    pub error_classes: BTreeMap<String, usize>,
    pub load_generation_valid: bool,
    /// `open_loop` (queue wait is a real client-bottleneck signal) or
    /// `closed_loop_not_applicable` (all requests are queued by design).
    pub load_generation_check: String,
    pub percentile_method: String,
}

pub fn summarize(
    samples: &[Sample],
    duration_s: f64,
    offered_rps: f64,
    slo: Slo,
    // `None` for closed-loop runs, where the concurrency semaphore deliberately
    // queues every request and a large queue wait says nothing about the client.
    max_queue_wait_ms: Option<f64>,
) -> CaseMetrics {
    let mut metrics = CaseMetrics {
        duration_s: round(duration_s, 4),
        offered_rps: round(offered_rps, 3),
        percentile_method: PERCENTILE_METHOD.to_owned(),
        load_generation_check: if max_queue_wait_ms.is_some() {
            "open_loop".to_owned()
        } else {
            "closed_loop_not_applicable".to_owned()
        },
        ..Default::default()
    };
    let good = samples
        .iter()
        .filter(|sample| sample.error.is_empty())
        .collect::<Vec<_>>();
    metrics.completed = good.len();
    metrics.failed = samples.len() - good.len();
    metrics.achieved_rps = round(good.len() as f64 / duration_s.max(1e-9), 3);

    let mut ttfts = Vec::new();
    let mut e2els = Vec::new();
    let mut tpots = Vec::new();
    let mut itls = Vec::new();
    let mut waits = Vec::new();
    let mut sources = BTreeMap::<String, usize>::new();
    let mut complete = true;

    for sample in samples {
        *metrics
            .status_counts
            .entry(sample.status.to_string())
            .or_insert(0) += 1;
        if !sample.error.is_empty() {
            let class = if sample.error_class.is_empty() {
                "unknown".to_owned()
            } else {
                sample.error_class.clone()
            };
            *metrics.error_classes.entry(class).or_insert(0) += 1;
        }
    }

    for sample in &good {
        metrics.output_tokens += sample.output_tokens as u64;
        metrics.input_tokens += sample.prompt_tokens.unwrap_or(0) as u64;
        *sources.entry(sample.token_source.clone()).or_insert(0) += 1;
        if let Some(ttft) = sample.ttft_ms {
            ttfts.push(ttft);
        }
        e2els.push(sample.e2e_ms);
        waits.push(sample.queue_wait_ms);
        // TPOT excludes the first token; short outputs contribute zero so the
        // metric is defined for every completed request (as in vLLM).
        let tpot = match (sample.ttft_ms, sample.output_tokens) {
            (Some(ttft), tokens) if tokens > 1 => (sample.e2e_ms - ttft) / (tokens - 1) as f64,
            _ => 0.0,
        };
        tpots.push(tpot);
        if sample.chunk_count != sample.output_tokens || sample.token_source != "usage" {
            // Chunk-counted streams cannot give per-token ITL; mark it plainly.
            complete = false;
        }
        for pair in sample.token_times_ms.windows(2) {
            itls.push((pair[1] - pair[0]).max(0.0));
        }
        if slo.ignore_eos && sample.output_tokens < slo.max_tokens {
            metrics.truncated_requests += 1;
        }
        let ttft_ok =
            slo.ttft_ms <= 0.0 || sample.ttft_ms.is_some_and(|value| value <= slo.ttft_ms);
        let tpot_ok = slo.tpot_ms <= 0.0 || tpot <= slo.tpot_ms;
        let e2e_ok = slo.e2el_ms <= 0.0 || sample.e2e_ms <= slo.e2el_ms;
        let length_ok = !slo.ignore_eos || sample.output_tokens >= slo.max_tokens;
        if ttft_ok && tpot_ok && e2e_ok && length_ok {
            metrics.good_requests += 1;
        }
    }

    let itl_count = itls.len();
    metrics.ttft_ms = stats(ttfts);
    metrics.e2el_ms = stats(e2els);
    metrics.tpot_ms = stats(tpots);
    metrics.itl_ms = stats(itls);
    metrics.queue_wait_ms = stats(waits);
    metrics.goodput_rps = round(metrics.good_requests as f64 / duration_s.max(1e-9), 3);
    metrics.output_throughput = round(metrics.output_tokens as f64 / duration_s.max(1e-9), 3);
    metrics.input_throughput = round(metrics.input_tokens as f64 / duration_s.max(1e-9), 3);
    metrics.total_token_throughput = round(
        (metrics.output_tokens + metrics.input_tokens) as f64 / duration_s.max(1e-9),
        3,
    );
    metrics.output_tokens_per_request = round(
        metrics.output_tokens as f64 / metrics.completed.max(1) as f64,
        3,
    );
    metrics.token_source = if sources.len() == 1 {
        sources.keys().next().cloned().unwrap_or_default()
    } else {
        sources
            .iter()
            .map(|(key, count)| format!("{key}:{count}"))
            .collect::<Vec<_>>()
            .join(",")
    };
    metrics.itl_complete = complete && itl_count > 0;
    metrics.load_generation_valid = match max_queue_wait_ms {
        Some(limit) => metrics.queue_wait_ms.p95 <= limit,
        None => true,
    };

    // vLLM computes the peak gauges from successful requests only.  Failed
    // samples have no token timestamps and would otherwise inflate peak
    // concurrency with phantom in-flight requests.
    let (peak_output, peak_concurrency) = peaks(&good);
    metrics.peak_output_throughput = round(peak_output, 3);
    metrics.peak_concurrency = peak_concurrency;
    metrics
}

/// Peak output tokens per second and peak in-flight requests in 1s buckets.
fn peaks(samples: &[&Sample]) -> (f64, usize) {
    let mut tokens = BTreeMap::<i64, u64>::new();
    let mut active = BTreeMap::<i64, i64>::new();
    for sample in samples {
        for time in &sample.token_times_ms {
            *tokens.entry((time / 1000.0).floor() as i64).or_insert(0) += 1;
        }
        let first = (sample.start_ms / 1000.0).floor() as i64;
        let last = (sample.end_ms / 1000.0).floor() as i64;
        for bucket in first..=last.min(first + 3600) {
            *active.entry(bucket).or_insert(0) += 1;
        }
    }
    (
        tokens.values().copied().max().unwrap_or(0) as f64,
        active.values().copied().max().unwrap_or(0).max(0) as usize,
    )
}

/// Element-wise median of trial metrics, including nested maps and arrays.
pub fn median_value(values: &[Value]) -> Value {
    if values.is_empty() {
        return Value::Null;
    }
    match &values[0] {
        Value::Number(_) => {
            let mut numbers: Vec<f64> = values.iter().filter_map(Value::as_f64).collect::<Vec<_>>();
            numbers.sort_by(f64::total_cmp);
            json!(round(percentile(&numbers, 0.5), 4))
        }
        Value::Object(_) => {
            let mut keys: Vec<String> = Vec::new();
            for value in values {
                if let Some(object) = value.as_object() {
                    for key in object.keys() {
                        if !keys.contains(key) {
                            keys.push(key.clone());
                        }
                    }
                }
            }
            let mut object = serde_json::Map::new();
            for key in keys {
                let entries: Vec<Value> = values
                    .iter()
                    .filter_map(|value| value.get(&key).cloned())
                    .collect();
                object.insert(key, median_value(&entries));
            }
            Value::Object(object)
        }
        Value::Array(_) => {
            let length = values
                .iter()
                .filter_map(Value::as_array)
                .map(Vec::len)
                .max()
                .unwrap_or(0);
            (0..length)
                .map(|index| {
                    median_value(
                        &values
                            .iter()
                            .filter_map(|value| value.as_array())
                            .filter_map(|array| array.get(index).cloned())
                            .collect::<Vec<_>>(),
                    )
                })
                .collect()
        }
        first => first.clone(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn percentile_matches_linear_interpolation() {
        let sorted = vec![1.0, 2.0, 3.0, 4.0];
        assert_eq!(percentile(&sorted, 0.0), 1.0);
        assert_eq!(percentile(&sorted, 0.5), 2.5);
        assert_eq!(percentile(&sorted, 1.0), 4.0);
        assert_eq!(percentile(&[], 0.5), 0.0);
    }

    #[test]
    fn nested_median_is_element_wise() {
        let a = json!({"ttft": {"p95": 10.0}, "n": 2});
        let b = json!({"ttft": {"p95": 20.0}, "n": 4});
        let c = json!({"ttft": {"p95": 30.0}, "n": 6});
        let median = median_value(&[a, b, c]);
        assert_eq!(median["ttft"]["p95"], 20.0);
        assert_eq!(median["n"], 4.0);
    }

    #[test]
    fn stats_records_population_std() {
        let result = stats(vec![1.0, 3.0]);
        assert_eq!(result.mean, 2.0);
        assert_eq!(result.std, 1.0);
        assert_eq!(stats(vec![5.0]).std, 0.0);
    }

    #[test]
    fn failed_requests_do_not_inflate_peak_concurrency() {
        let good = Sample {
            request_id: "ok".into(),
            start_ms: 0.0,
            end_ms: 100.0,
            queue_wait_ms: 0.0,
            e2e_ms: 100.0,
            ttft_ms: Some(50.0),
            output_tokens: 1,
            prompt_tokens: Some(1),
            token_source: "usage".into(),
            chunk_count: 1,
            token_times_ms: vec![50.0],
            finish_reason: "stop".into(),
            status: 200,
            error: String::new(),
            error_class: String::new(),
            text: "x".into(),
        };
        let mut failed = good.clone();
        failed.request_id = "bad".into();
        failed.error = "HTTP 500".into();
        failed.error_class = "http_5xx".into();
        let with_failure = summarize(&[good, failed], 1.0, 2.0, Slo::default(), Some(1000.0));
        assert_eq!(with_failure.peak_concurrency, 1);
        assert_eq!(with_failure.failed, 1);
    }

    #[test]
    fn summarize_counts_errors_and_goodput() {
        let sample = Sample {
            request_id: "r".into(),
            start_ms: 0.0,
            end_ms: 1000.0,
            queue_wait_ms: 0.0,
            e2e_ms: 1000.0,
            ttft_ms: Some(100.0),
            output_tokens: 10,
            prompt_tokens: Some(50),
            token_source: "usage".into(),
            chunk_count: 10,
            token_times_ms: (1..=10).map(|i| i as f64 * 100.0).collect(),
            finish_reason: "length".into(),
            status: 200,
            error: String::new(),
            error_class: String::new(),
            text: "x".to_owned(),
        };
        let metrics = summarize(&[sample], 1.0, 1.0, Slo::default(), Some(1000.0));
        assert_eq!(metrics.completed, 1);
        assert_eq!(metrics.output_throughput, 10.0);
        assert_eq!(metrics.achieved_rps, 1.0);
        assert_eq!(metrics.tpot_ms.n, 1);
        assert!(metrics.load_generation_valid);
    }
}
