//! Prometheus text exposition (format 0.0.4) for replica metric snapshots.
//!
//! The snapshot is a nested JSON object. Leaf numbers and booleans become
//! scalar metrics prefixed with `megalm_`; dictionaries whose key ends with
//! `_histogram` become real Prometheus histograms (`_bucket`, `_sum`,
//! `_count`) usable by `histogram_quantile()`.
//!
//! For topology-independent, label-rich series a leaf may instead be an
//! explicit descriptor object:
//!
//! ```json
//! {"http_requests": {"__value__": 3,
//!                    "__type__": "counter",
//!                    "__labels__": {"status": "503", "route": "/v1/chat"}}}
//! ```
//!
//! which renders as `megalm_http_requests{route="/v1/chat",status="503"} 3`.
//! HELP/TYPE are emitted once per metric name. Histograms may carry
//! `__sum__` / `__count__` to supply exact observation totals instead of the
//! bucket-upper-bound approximation.

use std::collections::HashSet;

use serde_json::Map;

fn metric_component(component: &str) -> String {
    let mut sanitized: String = component
        .chars()
        .map(|value| {
            if value.is_ascii_alphanumeric() || value == '_' {
                value.to_ascii_lowercase()
            } else {
                '_'
            }
        })
        .collect();
    if sanitized.is_empty() {
        sanitized.push_str("value");
    }
    if sanitized.as_bytes()[0].is_ascii_digit() {
        sanitized.insert(0, 'm');
    }
    sanitized
}

fn metric_name(path: &[String]) -> String {
    let mut name = String::from("megalm_");
    name.push_str(&path.join("_"));
    name
}

/// Heuristic gauge detection used when a descriptor does not declare
/// `__type__`. Counts and accumulated totals default to `counter`; values
/// representing a current level (means, rates, capacities, queue depths,
/// byte sizes, cache residency) are gauges.
fn is_gauge(path: &[String]) -> bool {
    let name = path.join("_");
    // Cumulative totals that happen to match a gauge marker.
    const COUNTER_OVERRIDES: &[&str] = &["evicted_bytes", "parser_bytes", "sent_bytes"];
    if COUNTER_OVERRIDES.iter().any(|marker| name.contains(marker)) {
        return false;
    }
    const GAUGE_MARKERS: &[&str] = &[
        "_mean",
        "_rate",
        "hit_rate",
        "max_",
        "free_",
        "pending_",
        "active_",
        "in_flight",
        "inflight",
        "waiting_",
        "running_",
        "queued_",
        "allocated_",
        "total_pages",
        "cached_pages",
        "pinned_",
        "evictable_",
        "page_refs",
        "entries",
        "nodes",
        "_occupancy",
        "_utilization",
        "util",
        "capacity",
        "_bytes",
        "_size",
        "reclaimable",
        "used_",
    ];
    GAUGE_MARKERS.iter().any(|marker| name.contains(marker))
}

fn declare_metric(
    name: &str,
    metric_type: &str,
    output: &mut String,
    declared: &mut HashSet<String>,
) {
    if !declared.insert(name.to_string()) {
        return;
    }
    output.push_str("# HELP ");
    output.push_str(name);
    output.push(' ');
    output.push_str(&name.replace('_', " "));
    output.push_str(".\n# TYPE ");
    output.push_str(name);
    output.push(' ');
    output.push_str(metric_type);
    output.push('\n');
}

fn write_metric(name: &str, value: &str, output: &mut String) {
    output.push_str(name);
    output.push(' ');
    output.push_str(value);
    output.push('\n');
}

fn scalar_type(path: &[String], declared_type: Option<&str>) -> &'static str {
    match declared_type {
        Some("gauge") => "gauge",
        Some("counter") | Some(_) => "counter",
        None => {
            if is_gauge(path) {
                "gauge"
            } else {
                "counter"
            }
        }
    }
}

fn write_scalar(
    path: &[String],
    value: &str,
    metric_type: &str,
    output: &mut String,
    declared: &mut HashSet<String>,
) {
    let name = metric_name(path);
    declare_metric(&name, metric_type, output, declared);
    write_metric(&name, value, output);
}

fn escape_label_value(value: &str) -> String {
    value
        .replace('\\', "\\\\")
        .replace('"', "\\\"")
        .replace('\n', "\\n")
}

fn write_labeled(
    path: &[String],
    value: &serde_json::Value,
    labels: &Map<String, serde_json::Value>,
    metric_type: &str,
    output: &mut String,
    declared: &mut HashSet<String>,
) {
    let name = metric_name(path);
    declare_metric(&name, metric_type, output, declared);
    output.push_str(&name);
    output.push('{');
    let mut entries: Vec<(&String, &serde_json::Value)> = labels.iter().collect();
    entries.sort_by(|a, b| a.0.cmp(b.0));
    for (index, (key, label_value)) in entries.iter().enumerate() {
        if index > 0 {
            output.push(',');
        }
        output.push_str(&metric_component(key));
        output.push_str("=\"");
        let rendered = match label_value {
            serde_json::Value::String(text) => text.clone(),
            other => other.to_string(),
        };
        output.push_str(&escape_label_value(&rendered));
        output.push('"');
    }
    output.push_str("} ");
    output.push_str(&value.to_string());
    output.push('\n');
}

fn format_le(value: f64) -> String {
    if value.fract() == 0.0 && value.is_finite() {
        format!("{}", value as i64)
    } else {
        format!("{value}")
    }
}

fn write_histogram(
    path: &[String],
    entries: &Map<String, serde_json::Value>,
    output: &mut String,
    declared: &mut HashSet<String>,
) {
    let name = metric_name(path);
    declare_metric(&name, "histogram", output, declared);

    let mut buckets: Vec<(f64, u64)> = Vec::new();
    for (key, value) in entries {
        if key.starts_with("__") {
            continue;
        }
        if let Some(bucket) = key.parse::<f64>().ok() {
            if let Some(count) = value.as_u64().or_else(|| value.as_f64().map(|v| v as u64)) {
                buckets.push((bucket, count));
            }
        }
    }
    buckets.sort_by(|a, b| a.0.total_cmp(&b.0));

    let mut cumulative = 0u64;
    let mut sum = 0.0f64;
    for (le, count) in buckets {
        cumulative = cumulative.saturating_add(count);
        sum += le * count as f64;
        output.push_str(&format!(
            "{name}_bucket{{le=\"{}\"}} {cumulative}\n",
            format_le(le)
        ));
    }
    output.push_str(&format!("{name}_bucket{{le=\"+Inf\"}} {cumulative}\n"));

    // Prefer exact observation totals when the producer supplies them.
    let exact_sum = entries.get("__sum__").and_then(|v| v.as_f64());
    let exact_count = entries.get("__count__").and_then(|v| v.as_u64());
    output.push_str(&format!("{name}_sum {}\n", exact_sum.unwrap_or(sum)));
    output.push_str(&format!(
        "{name}_count {}\n",
        exact_count.unwrap_or(cumulative)
    ));
}

fn flatten_metrics(
    value: &serde_json::Value,
    path: &mut Vec<String>,
    output: &mut String,
    declared: &mut HashSet<String>,
) {
    match value {
        serde_json::Value::Object(entries) => {
            // Explicit labeled descriptor: {"__value__":..,"__labels__":{...}}
            if let (Some(sample), Some(labels)) =
                (entries.get("__value__"), entries.get("__labels__"))
            {
                if let serde_json::Value::Object(label_map) = labels {
                    let declared_type = entries
                        .get("__type__")
                        .and_then(|v| v.as_str());
                    let metric_type = scalar_type(path, declared_type);
                    write_labeled(path, sample, label_map, metric_type, output, declared);
                    return;
                }
            }
            if path.last().is_some_and(|key| key.ends_with("_histogram")) {
                write_histogram(path, entries, output, declared);
                return;
            }
            for (key, child) in entries {
                path.push(metric_component(key));
                flatten_metrics(child, path, output, declared);
                path.pop();
            }
        }
        serde_json::Value::Array(items) => {
            // An array of labeled descriptors represents multiple samples of
            // the SAME metric, distinguished by their label sets. Render each
            // at the current path instead of flattening to item_N.
            let all_labeled = !items.is_empty()
                && items.iter().all(|item| {
                    item.as_object().is_some_and(|object| {
                        object.contains_key("__value__")
                            && object.contains_key("__labels__")
                    })
                });
            if all_labeled {
                for item in items {
                    let object = item.as_object().unwrap();
                    let sample = object.get("__value__").unwrap();
                    let labels = object.get("__labels__").unwrap().as_object().unwrap();
                    let metric_type = scalar_type(
                        path,
                        object.get("__type__").and_then(|v| v.as_str()),
                    );
                    write_labeled(path, sample, labels, metric_type, output, declared);
                }
                return;
            }
            for (index, child) in items.iter().enumerate() {
                path.push(format!("item_{index}"));
                flatten_metrics(child, path, output, declared);
                path.pop();
            }
        }
        serde_json::Value::Number(number) => {
            write_scalar(path, &number.to_string(), scalar_type(path, None), output, declared);
        }
        serde_json::Value::Bool(flag) => {
            write_scalar(path, if *flag { "1" } else { "0" }, scalar_type(path, None), output, declared);
        }
        _ => {}
    }
}

/// Render a nested snapshot as Prometheus text exposition 0.0.4.
pub fn render_prometheus_metrics(payload_json: &str) -> Result<String, serde_json::Error> {
    let payload: serde_json::Value = serde_json::from_str(payload_json)?;
    let mut output = String::new();
    let mut declared = HashSet::new();
    let mut path = Vec::new();
    flatten_metrics(&payload, &mut path, &mut output, &mut declared);
    Ok(output)
}

#[cfg(test)]
mod metrics_tests {
    use super::render_prometheus_metrics;

    #[test]
    fn renders_nested_numeric_fields_with_help_and_type() {
        let rendered = render_prometheus_metrics(
            r#"{"proxy":{"admitted_requests":3},"runtime":{"gpu":{"util":0.5}}}"#,
        )
        .unwrap();
        assert!(rendered.contains("# HELP megalm_proxy_admitted_requests megalm proxy admitted requests.\n"));
        assert!(rendered.contains("# TYPE megalm_proxy_admitted_requests counter\n"));
        assert!(rendered.contains("megalm_proxy_admitted_requests 3\n"));
        assert!(rendered.contains("# TYPE megalm_runtime_gpu_util gauge\n"));
        assert!(rendered.contains("megalm_runtime_gpu_util 0.5\n"));
    }

    #[test]
    fn cumulative_byte_totals_are_counters_not_gauges() {
        let rendered = render_prometheus_metrics(
            r#"{"replica":{"parser":{"bytes":1024}},"worker":{"sent_bytes":2048},"gpu":{"memory_bytes":4096}}"#,
        )
        .unwrap();
        assert!(
            rendered.contains("# TYPE megalm_replica_parser_bytes counter\n"),
            "{rendered}"
        );
        assert!(
            rendered.contains("# TYPE megalm_worker_sent_bytes counter\n"),
            "{rendered}"
        );
        // A live capacity gauge must keep its gauge type.
        assert!(
            rendered.contains("# TYPE megalm_gpu_memory_bytes gauge\n"),
            "{rendered}"
        );
    }

    #[test]
    fn skips_non_numeric_fields() {
        let rendered = render_prometheus_metrics(
            r#"{"status":"ok","cancelled":true,"hist":{"1":2}}"#,
        )
        .unwrap();
        assert!(!rendered.contains("status"));
        assert!(rendered.contains("megalm_cancelled 1\n"));
        assert!(rendered.contains("megalm_hist_m1 2\n"));
    }

    #[test]
    fn histogram_keys_ending_in_histogram_emit_bucket_sum_and_count() {
        let rendered = render_prometheus_metrics(
            r#"{"engine":{"sequence_batch_histogram":{"1":5,"2":3,"10":1}}}"#,
        )
        .unwrap();
        let name = "megalm_engine_sequence_batch_histogram";
        assert!(rendered.contains(&format!("# TYPE {name} histogram\n")));
        assert!(rendered.contains(&format!("{name}_bucket{{le=\"1\"}} 5\n")));
        assert!(rendered.contains(&format!("{name}_bucket{{le=\"2\"}} 8\n")));
        assert!(rendered.contains(&format!("{name}_bucket{{le=\"10\"}} 9\n")));
        assert!(rendered.contains(&format!("{name}_bucket{{le=\"+Inf\"}} 9\n")));
        assert!(rendered.contains(&format!("{name}_sum 21\n")));
        assert!(rendered.contains(&format!("{name}_count 9\n")));
    }

    #[test]
    fn empty_histogram_emits_zero_series() {
        let rendered =
            render_prometheus_metrics(r#"{"h_histogram":{}}"#).unwrap();
        assert!(rendered.contains("megalm_h_histogram_count 0\n"));
        assert!(rendered.contains("megalm_h_histogram_sum 0\n"));
        assert!(rendered.contains("megalm_h_histogram_bucket{le=\"+Inf\"} 0\n"));
    }

    #[test]
    fn metric_components_are_sanitized() {
        let rendered = render_prometheus_metrics(r#"{"1bad-key":2}"#).unwrap();
        assert!(rendered.contains("megalm_m1bad_key 2\n"));
    }

    #[test]
    fn labeled_descriptors_render_native_labels_and_declare_header_once() {
        let rendered = render_prometheus_metrics(
            r#"{"http_requests":[
                 {"__value__":3,"__type__":"counter","__labels__":{"reason":"stop","route":"/v1/chat"}},
                 {"__value__":2,"__type__":"counter","__labels__":{"reason":"error","route":"/v1/chat"}}
            ]}"#,
        )
        .unwrap();
        assert_eq!(
            rendered.matches("# TYPE megalm_http_requests counter\n").count(),
            1
        );
        assert!(rendered.contains(
            "megalm_http_requests{reason=\"error\",route=\"/v1/chat\"} 2\n"
        ));
        assert!(rendered.contains(
            "megalm_http_requests{reason=\"stop\",route=\"/v1/chat\"} 3\n"
        ));
    }

    #[test]
    fn histogram_accepts_exact_sum_and_count() {
        let rendered = render_prometheus_metrics(
            r#"{"latency_ms_histogram":{"5":1,"25":2,"__sum__":37.5,"__count__":3}}"#,
        )
        .unwrap();
        assert!(rendered.contains("megalm_latency_ms_histogram_sum 37.5\n"));
        assert!(rendered.contains("megalm_latency_ms_histogram_count 3\n"));
    }

    #[test]
    fn label_values_are_escaped() {
        let rendered = render_prometheus_metrics(
            r#"{"e":{"v":{"__value__":1,"__labels__":{"msg":"a\"b\\c"}}}}"#,
        )
        .unwrap();
        assert!(rendered.contains("msg=\"a\\\"b\\\\c\""));
    }
}
