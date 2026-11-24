//! Process-level telemetry for the Rust gateway frontend.
//!
//! This module is the Rust counterpart of the Python `ReplicaMetrics` /
//! `ProcessReplicaMetrics` snapshots that used to live in the deleted host
//! path. It reproduces the exact snapshot shape the Prometheus renderer in
//! `megalm_rs::metrics` and the dashboards in `docs/design/observability.md`
//! expect, so `megalm_proxy_*` and `megalm_replica_*` have a real producer
//! again after the frontend moved into Rust.
//!
//! Hot-path updates are lock-free atomics. The only lock protects the
//! finish-reason map and is taken once per completed request, never inside a
//! decode loop.

use std::collections::BTreeMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Mutex;
use std::time::{SystemTime, UNIX_EPOCH};

use megalm_rs::parser::{ParseError, ParserStats};
use serde_json::{json, Map, Value};

/// Latency bucket boundaries in milliseconds. Kept identical to Python's
/// `observability.LATENCY_BUCKETS_MS` so `histogram_quantile()` series and
/// cross-frontend merges stay comparable.
pub const LATENCY_BUCKETS_MS: [f64; 13] = [
    1.0, 5.0, 10.0, 25.0, 50.0, 100.0, 250.0, 500.0, 1000.0, 2500.0, 5000.0, 10000.0, 60000.0,
];

/// Fixed-boundary histogram backed by atomics. Storing per-bucket counts (not
/// cumulative ones) matches `megalm_rs::metrics`, which accumulates buckets in
/// sorted `le` order when rendering a Prometheus histogram.
struct LatencyHistogram<const N: usize> {
    bounds: [f64; N],
    buckets: [AtomicU64; N],
    /// Exact sum in microseconds: sub-millisecond precision without atomics
    /// of floating point values.
    sum_us: AtomicU64,
    count: AtomicU64,
}

impl<const N: usize> LatencyHistogram<N> {
    fn new(bounds: [f64; N]) -> Self {
        Self {
            bounds,
            buckets: std::array::from_fn(|_| AtomicU64::new(0)),
            sum_us: AtomicU64::new(0),
            count: AtomicU64::new(0),
        }
    }

    fn observe_ms(&self, milliseconds: f64) {
        if !milliseconds.is_finite() || milliseconds < 0.0 {
            return;
        }
        self.sum_us
            .fetch_add((milliseconds * 1_000.0) as u64, Ordering::Relaxed);
        self.count.fetch_add(1, Ordering::Relaxed);
        let index = self
            .bounds
            .iter()
            .position(|bound| milliseconds <= *bound)
            .unwrap_or(N - 1);
        self.buckets[index].fetch_add(1, Ordering::Relaxed);
    }

    fn samples(&self) -> u64 {
        self.count.load(Ordering::Relaxed)
    }

    fn mean_ms(&self) -> f64 {
        let count = self.samples();
        if count == 0 {
            0.0
        } else {
            self.sum_us.load(Ordering::Relaxed) as f64 / count as f64 / 1_000.0
        }
    }

    /// Same shape as Python `histogram_totals(hist, total_sum, total_count)`:
    /// bucket keys plus `__sum__` / `__count__` for exact observation totals.
    fn snapshot(&self) -> Value {
        let mut buckets = Map::new();
        for (bound, bucket) in self.bounds.iter().zip(&self.buckets) {
            let key = if bound.fract() == 0.0 {
                format!("{}", *bound as u64)
            } else {
                format!("{bound}")
            };
            buckets.insert(key, json!(bucket.load(Ordering::Relaxed)));
        }
        buckets.insert(
            "__sum__".into(),
            json!(self.sum_us.load(Ordering::Relaxed) as f64 / 1_000.0),
        );
        buckets.insert("__count__".into(), json!(self.samples()));
        Value::Object(buckets)
    }
}

/// Aggregate counters for one gateway process.
pub struct ReplicaMetrics {
    admitted_requests: AtomicU64,
    ttft: LatencyHistogram<13>,
    tpot: LatencyHistogram<13>,
    e2e: LatencyHistogram<13>,
    finish_reasons: Mutex<BTreeMap<String, u64>>,
    parser_pushes: AtomicU64,
    parser_finishes: AtomicU64,
    parser_calls: AtomicU64,
    parser_diagnostics: AtomicU64,
    parser_bytes: AtomicU64,
}

impl ReplicaMetrics {
    pub fn new() -> Self {
        Self {
            admitted_requests: AtomicU64::new(0),
            ttft: LatencyHistogram::new(LATENCY_BUCKETS_MS),
            tpot: LatencyHistogram::new(LATENCY_BUCKETS_MS),
            e2e: LatencyHistogram::new(LATENCY_BUCKETS_MS),
            finish_reasons: Mutex::new(BTreeMap::new()),
            parser_pushes: AtomicU64::new(0),
            parser_finishes: AtomicU64::new(0),
            parser_calls: AtomicU64::new(0),
            parser_diagnostics: AtomicU64::new(0),
            parser_bytes: AtomicU64::new(0),
        }
    }

    /// One HTTP request was accepted and handed to the engine.
    pub fn admit_request(&self) {
        self.admitted_requests.fetch_add(1, Ordering::Relaxed);
    }

    pub fn record_ttft_ms(&self, milliseconds: f64) {
        self.ttft.observe_ms(milliseconds);
    }

    pub fn record_tpot_ms(&self, milliseconds: f64) {
        self.tpot.observe_ms(milliseconds);
    }

    pub fn record_e2e_ms(&self, milliseconds: f64) {
        self.e2e.observe_ms(milliseconds);
    }

    pub fn record_finish_reason(&self, reason: &str) {
        let mut reasons = self
            .finish_reasons
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        *reasons.entry(reason.to_owned()).or_insert(0) += 1;
    }

    /// Fold one request's Rust parser counters into the process totals.
    /// `stats.diagnostics` already counts the degraded blocks, matching the
    /// Python oracle's `parser_diagnostics += stats[3]`.
    pub fn record_parser(&self, stats: &ParserStats) {
        self.parser_pushes
            .fetch_add(stats.pushes, Ordering::Relaxed);
        self.parser_finishes
            .fetch_add(stats.finishes, Ordering::Relaxed);
        self.parser_calls.fetch_add(stats.calls, Ordering::Relaxed);
        self.parser_diagnostics
            .fetch_add(stats.diagnostics, Ordering::Relaxed);
        self.parser_bytes.fetch_add(stats.bytes, Ordering::Relaxed);
    }

    /// Snapshot folded into `/metrics` next to the worker's runtime snapshot.
    /// The `proxy` / `replica` key layout matches the Python topology-neutral
    /// contract, so dashboards need no deployment-mode changes.
    pub fn snapshot(&self) -> Value {
        let admitted = self.admitted_requests.load(Ordering::Relaxed);
        let finish_reasons = self
            .finish_reasons
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .clone();
        json!({
            "proxy": {
                "admitted_requests": admitted,
            },
            "replica": {
                "admitted_requests": admitted,
                "ttft_ms_mean": self.ttft.mean_ms(),
                "ttft_samples": self.ttft.samples(),
                "tpot_ms_mean": self.tpot.mean_ms(),
                "tpot_samples": self.tpot.samples(),
                "e2e_ms_mean": self.e2e.mean_ms(),
                "e2e_samples": self.e2e.samples(),
                "ttft_ms_histogram": self.ttft.snapshot(),
                "tpot_ms_histogram": self.tpot.snapshot(),
                "e2e_ms_histogram": self.e2e.snapshot(),
                "finish_reasons": finish_reasons,
                "parser": {
                    "pushes": self.parser_pushes.load(Ordering::Relaxed),
                    "finishes": self.parser_finishes.load(Ordering::Relaxed),
                    "calls": self.parser_calls.load(Ordering::Relaxed),
                    "diagnostics": self.parser_diagnostics.load(Ordering::Relaxed),
                    "bytes": self.parser_bytes.load(Ordering::Relaxed),
                },
            },
        })
    }
}

impl Default for ReplicaMetrics {
    fn default() -> Self {
        Self::new()
    }
}

// ---- Structured diagnostics logging ---------------------------------------

/// Emit one `parser_degraded` single-line JSON event per diagnostic, matching
/// the Python `JsonLogFormatter` schema and field names exactly. A degraded
/// block is still surfaced as visible text; this log is what makes the
/// degradation observable in Loki / ELK / Datadog.
pub fn emit_parser_degraded(request_id: &str, streaming: bool, diagnostics: &[ParseError]) {
    for diagnostic in diagnostics {
        let mut fields = Map::new();
        fields.insert("request_id".into(), json!(request_id));
        fields.insert("parser".into(), json!(diagnostic.parser));
        fields.insert("kind".into(), json!(diagnostic.kind.as_str()));
        fields.insert("message".into(), json!(diagnostic.message));
        fields.insert("position".into(), json!(diagnostic.position));
        fields.insert("context".into(), json!(diagnostic.context));
        fields.insert("expected".into(), json!(diagnostic.expected));
        fields.insert("streaming".into(), json!(streaming));
        emit_event(
            "megalm.replica.parser",
            "warning",
            "parser_degraded",
            fields,
        );
    }
}

/// Emit one single-line JSON log with the standard `JsonLogFormatter`
/// envelope: timestamp / level / logger / event / pid, then event fields.
pub fn emit_event(logger: &str, level: &str, event: &str, fields: Map<String, Value>) {
    eprintln!("{}", event_payload(logger, level, event, fields));
}

/// Pure envelope builder so the log schema can be asserted without capturing
/// stderr.
fn event_payload(logger: &str, level: &str, event: &str, fields: Map<String, Value>) -> Value {
    let mut payload = Map::new();
    payload.insert("timestamp".into(), json!(utc_timestamp_ms(now_ns())));
    payload.insert("level".into(), json!(level));
    payload.insert("logger".into(), json!(logger));
    payload.insert("event".into(), json!(event));
    payload.insert("pid".into(), json!(std::process::id()));
    payload.extend(fields);
    Value::Object(payload)
}

fn now_ns() -> u128 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_nanos()
}

fn utc_timestamp_ms(ns: u128) -> String {
    let seconds = (ns / 1_000_000_000) as i64;
    let millis = ((ns % 1_000_000_000) / 1_000_000) as u32;
    let days = seconds.div_euclid(86_400);
    let within_day = seconds.rem_euclid(86_400);
    let (year, month, day) = civil_from_days(days);
    format!(
        "{year:04}-{month:02}-{day:02}T{:02}:{:02}:{:02}.{millis:03}Z",
        within_day / 3_600,
        (within_day % 3_600) / 60,
        within_day % 60,
    )
}

/// Howard Hinnant's `civil_from_days`: days since 1970-01-01 → (year, month,
/// day). Avoids pulling a date/time crate onto the serving hot path.
fn civil_from_days(days: i64) -> (i64, i64, i64) {
    let z = days + 719_468;
    let era = if z >= 0 { z } else { z - 146_096 } / 146_097;
    let day_of_era = z - era * 146_097;
    let year_of_era =
        (day_of_era - day_of_era / 1_460 + day_of_era / 36_524 - day_of_era / 146_096) / 365;
    let year = year_of_era + era * 400;
    let day_of_year = day_of_era - (365 * year_of_era + year_of_era / 4 - year_of_era / 100);
    let month_prime = (5 * day_of_year + 2) / 153;
    let day = day_of_year - (153 * month_prime + 2) / 5 + 1;
    let month = month_prime + if month_prime < 10 { 3 } else { -9 };
    (year + if month <= 2 { 1 } else { 0 }, month, day)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn histogram_buckets_are_per_bucket_and_snapshot_is_cumulative_ready() {
        let hist = LatencyHistogram::new(LATENCY_BUCKETS_MS);
        hist.observe_ms(0.5);
        hist.observe_ms(4.0);
        hist.observe_ms(7.0);
        hist.observe_ms(100_000.0); // beyond the largest bound folds into it
        let snapshot = hist.snapshot();
        assert_eq!(snapshot["__count__"], 4);
        assert_eq!(snapshot["1"], 1); // 0.5 ms
        assert_eq!(snapshot["5"], 1); // 4.0 ms
        assert_eq!(snapshot["10"], 1); // 7.0 ms
        assert_eq!(snapshot["60000"], 1); // beyond the largest bound
        assert_eq!(snapshot["__sum__"].as_f64().unwrap() as u64, 100_011);
    }

    #[test]
    fn means_are_zero_without_samples() {
        let metrics = ReplicaMetrics::new();
        let snapshot = metrics.snapshot();
        assert_eq!(snapshot["replica"]["ttft_ms_mean"], 0.0);
        assert_eq!(snapshot["replica"]["tpot_ms_mean"], 0.0);
        assert_eq!(snapshot["replica"]["ttft_samples"], 0);
    }

    #[test]
    fn snapshot_matches_python_replica_contract() {
        let metrics = ReplicaMetrics::new();
        metrics.admit_request();
        metrics.record_ttft_ms(12.5);
        metrics.record_tpot_ms(3.0);
        metrics.record_e2e_ms(25.0);
        metrics.record_finish_reason("stop");
        metrics.record_finish_reason("length");
        let mut stats = ParserStats::default();
        stats.pushes = 7;
        stats.finishes = 1;
        stats.calls = 2;
        stats.diagnostics = 1;
        stats.bytes = 128;
        metrics.record_parser(&stats);

        let snapshot = metrics.snapshot();
        assert_eq!(snapshot["proxy"]["admitted_requests"], 1);
        assert_eq!(snapshot["replica"]["admitted_requests"], 1);
        assert_eq!(snapshot["replica"]["ttft_ms_mean"], 12.5);
        assert_eq!(snapshot["replica"]["ttft_samples"], 1);
        assert_eq!(snapshot["replica"]["tpot_ms_mean"], 3.0);
        assert_eq!(snapshot["replica"]["finish_reasons"]["stop"], 1);
        assert_eq!(snapshot["replica"]["finish_reasons"]["length"], 1);
        assert_eq!(snapshot["replica"]["parser"]["pushes"], 7);
        assert_eq!(snapshot["replica"]["parser"]["bytes"], 128);
        assert_eq!(snapshot["replica"]["parser"]["diagnostics"], 1);
        assert!(snapshot["replica"]["ttft_ms_histogram"]["__sum__"].is_number());
    }

    #[test]
    fn civil_from_days_matches_known_dates() {
        assert_eq!(civil_from_days(0), (1970, 1, 1));
        assert_eq!(civil_from_days(1), (1970, 1, 2));
        assert_eq!(civil_from_days(20_454), (2026, 1, 1));
        assert_eq!(civil_from_days(-1), (1969, 12, 31));
    }

    #[test]
    fn timestamp_is_iso_8601_utc_with_milliseconds() {
        // 2026-01-01T00:00:00.123Z
        let ns = (20_454i128 * 86_400) as u128 * 1_000_000_000 + 123_000_000;
        assert_eq!(utc_timestamp_ms(ns), "2026-01-01T00:00:00.123Z");
    }

    #[test]
    fn structured_event_has_the_shared_json_log_envelope() {
        let mut fields = Map::new();
        fields.insert("request_id".into(), json!("req-1"));
        fields.insert("error".into(), json!("missing required property 'name'"));
        let payload = event_payload(
            "megalm.gateway.chat",
            "warning",
            "response_format_invalid",
            fields,
        );
        assert_eq!(payload["level"], "warning");
        assert_eq!(payload["logger"], "megalm.gateway.chat");
        assert_eq!(payload["event"], "response_format_invalid");
        assert_eq!(payload["pid"], std::process::id());
        assert_eq!(payload["request_id"], "req-1");
        assert!(payload["timestamp"]
            .as_str()
            .is_some_and(|value| value.ends_with('Z') && value.len() == 24));
    }
}
