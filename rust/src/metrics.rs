//! Prometheus text exposition (format 0.0.4) for the megaLM gateway.
//!
//! Thin wrapper around the shared `tensorplay-observability` renderer, which
//! pins the metric prefix to `megalm` so existing /metrics series are unchanged.

/// Render a nested snapshot as Prometheus text exposition 0.0.4.
pub fn render_prometheus_metrics(payload_json: &str) -> Result<String, serde_json::Error> {
    tensorplay_observability::render_prometheus_metrics(payload_json, "megalm")
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
}
