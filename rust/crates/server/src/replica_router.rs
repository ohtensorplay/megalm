//! Opt-in stateless replica router for horizontal gateway scaling
//! (port of `replica_router.py`). Each request is forwarded to exactly one
//! worker; 4xx replies are relayed verbatim and only transport failures count
//! towards worker disabling.

use std::collections::HashMap;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use axum::extract::State;
use axum::http::{HeaderMap, StatusCode};
use axum::response::{IntoResponse, Response};
use axum::Json;
use reqwest::Client;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};

use crate::profiling::{milestone, Profiler};

const HOP_BY_HOP: &[&str] = &[
    "connection",
    "content-length",
    "transfer-encoding",
    "keep-alive",
    "proxy-authenticate",
    "proxy-authorization",
    "te",
    "trailer",
    "upgrade",
];

#[derive(Debug, Default, Clone, Copy, PartialEq, Eq, Deserialize, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum Policy {
    #[default]
    RoundRobin,
    LeastRequest,
    Random,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct WorkerSpec {
    pub url: String,
    pub name: String,
}

impl WorkerSpec {
    fn validate(&self) -> Result<(), String> {
        let parsed = reqwest::Url::parse(&self.url)
            .map_err(|_| "worker url must be an absolute http(s) URL".to_string())?;
        if !matches!(parsed.scheme(), "http" | "https") || parsed.host_str().is_none() {
            return Err("worker url must be an absolute http(s) URL".into());
        }
        if self.name.trim().is_empty() {
            return Err("worker name must not be empty".into());
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Default, Serialize)]
pub struct WorkerStats {
    pub inflight: i64,
    pub requests: i64,
    pub failures: i64,
    pub consecutive_failures: i64,
    pub disabled_until_ns: Option<u128>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct RouterConfig {
    pub workers: Vec<WorkerSpec>,
    #[serde(default)]
    pub policy: Policy,
    #[serde(default = "default_timeout")]
    pub timeout_s: f64,
    #[serde(default = "default_connections")]
    pub max_connections: usize,
    #[serde(default = "default_threshold")]
    pub failure_threshold: i64,
    #[serde(default = "default_cooldown")]
    pub cooldown_s: f64,
    #[serde(default = "default_health_path")]
    pub health_path: String,
}

fn default_timeout() -> f64 {
    300.0
}
fn default_connections() -> usize {
    100
}
fn default_threshold() -> i64 {
    3
}
fn default_cooldown() -> f64 {
    30.0
}
fn default_health_path() -> String {
    "/health".into()
}

impl Default for RouterConfig {
    fn default() -> Self {
        Self {
            workers: Vec::new(),
            policy: Policy::RoundRobin,
            timeout_s: 300.0,
            max_connections: 100,
            failure_threshold: 3,
            cooldown_s: 30.0,
            health_path: "/health".into(),
        }
    }
}

impl RouterConfig {
    pub fn validate(&self) -> Result<(), String> {
        if self.workers.is_empty() {
            return Err("router requires at least one worker".into());
        }
        let mut urls = std::collections::HashSet::new();
        let mut names = std::collections::HashSet::new();
        for worker in &self.workers {
            worker.validate()?;
            if !names.insert(worker.name.clone()) {
                return Err("router worker names must be unique".into());
            }
            if !urls.insert(worker.url.clone()) {
                return Err("router worker urls must be unique".into());
            }
        }
        if self.policy != Policy::RoundRobin && self.workers.len() < 2 {
            return Err(format!("{} policy requires two or more workers", self.policy_str()));
        }
        if !self.timeout_s.is_finite() || self.timeout_s <= 0.0 || self.timeout_s > 3600.0 {
            return Err("timeout_s must be in (0, 3600]".into());
        }
        if self.max_connections == 0 {
            return Err("max_connections must be positive".into());
        }
        if self.failure_threshold <= 0 {
            return Err("failure_threshold must be positive".into());
        }
        if self.cooldown_s <= 0.0 {
            return Err("cooldown_s must be positive".into());
        }
        if !self.health_path.starts_with('/') {
            return Err("health_path must start with '/'".into());
        }
        Ok(())
    }

    fn policy_str(&self) -> &'static str {
        match self.policy {
            Policy::RoundRobin => "round_robin",
            Policy::LeastRequest => "least_request",
            Policy::Random => "random",
        }
    }
}

pub struct ReplicaRouter {
    config: Arc<RouterConfig>,
    stats: Arc<Mutex<HashMap<String, WorkerStats>>>,
    next: AtomicUsize,
    client: Client,
}

impl ReplicaRouter {
    pub fn new(config: RouterConfig) -> Result<Self, String> {
        config.validate()?;
        let mut stats = HashMap::new();
        for worker in &config.workers {
            stats.insert(worker.name.clone(), WorkerStats::default());
        }
        let client = Client::builder()
            .timeout(Duration::from_secs_f64(config.timeout_s))
            .pool_max_idle_per_host(config.max_connections)
            .build()
            .map_err(|err| format!("cannot build router HTTP client: {err}"))?;
        Ok(Self {
            config: Arc::new(config),
            stats: Arc::new(Mutex::new(stats)),
            next: AtomicUsize::new(0),
            client,
        })
    }

    fn enabled_workers(&self) -> Vec<usize> {
        let now = now_ns();
        let stats = self.stats.lock().unwrap();
        (0..self.config.workers.len())
            .filter(|index| {
                stats
                    .get(&self.config.workers[*index].name)
                    .is_none_or(|stat| stat.disabled_until_ns.is_none_or(|until| until <= now))
            })
            .collect()
    }

    fn mark_start(&self, index: usize) {
        let name = &self.config.workers[index].name;
        let mut stats = self.stats.lock().unwrap();
        if let Some(stat) = stats.get_mut(name) {
            stat.inflight += 1;
            stat.requests += 1;
        }
    }

    fn mark_failure(&self, index: usize) {
        let name = &self.config.workers[index].name;
        let mut stats = self.stats.lock().unwrap();
        if let Some(stat) = stats.get_mut(name) {
            stat.failures += 1;
            stat.consecutive_failures += 1;
            stat.inflight = (stat.inflight - 1).max(0);
            if stat.consecutive_failures >= self.config.failure_threshold {
                stat.disabled_until_ns =
                    Some(now_ns() + (self.config.cooldown_s * 1e9) as u128);
            }
        }
    }

    fn mark_success(&self, index: usize) {
        let name = &self.config.workers[index].name;
        let mut stats = self.stats.lock().unwrap();
        if let Some(stat) = stats.get_mut(name) {
            stat.consecutive_failures = 0;
            stat.inflight = (stat.inflight - 1).max(0);
        }
    }

    fn select(&self) -> Result<usize, String> {
        let enabled = self.enabled_workers();
        if enabled.is_empty() {
            return Err("no healthy replica is available".into());
        }
        match self.config.policy {
            Policy::RoundRobin => Ok(enabled[self.next.fetch_add(1, Ordering::Relaxed) % enabled.len()]),
            Policy::LeastRequest => {
                let stats = self.stats.lock().unwrap();
                Ok(*enabled
                    .iter()
                    .min_by_key(|index| {
                        stats
                            .get(&self.config.workers[**index].name)
                            .map(|stat| stat.inflight)
                            .unwrap_or(0)
                    })
                    .unwrap())
            }
            Policy::Random => Ok(enabled[rand_index() % enabled.len()]),
        }
    }

    fn relay_headers(source: &HeaderMap) -> HeaderMap {
        let mut headers = HeaderMap::new();
        for (name, value) in source {
            if HOP_BY_HOP.contains(&name.as_str()) {
                continue;
            }
            headers.insert(name.clone(), value.clone());
        }
        headers
    }

    /// Forward one request to a single worker and relay the reply.
    ///
    /// Hop milestones use the same `(request_id, src, dst, chunk_id)` key as
    /// the P/D router so `profiling::view_files` can pair them across processes.
    pub async fn route(
        &self,
        payload: Value,
        path: &str,
        profiler: &Profiler,
    ) -> Result<Response, String> {
        let index = self.select()?;
        let worker = &self.config.workers[index];
        self.mark_start(index);
        let url = format!("{}{}", worker.url.trim_end_matches('/'), path);
        let hop_id = payload
            .get("request_id")
            .and_then(Value::as_str)
            .filter(|id| !id.is_empty())
            .map(str::to_string)
            .unwrap_or_else(|| uuid::Uuid::new_v4().simple().to_string());
        tracing::info!(event = "hop_sent", stage = "router", src = "gateway", dst = %worker.url, request_id = %hop_id);
        profiler.record_hop(milestone::HOP_SENT, &hop_id, "gateway", &worker.url, "request");
        let response = self
            .client
            .post(&url)
            .json(&payload)
            .send()
            .await
            .map_err(|err| {
                // mark_failure also releases the in-flight slot; without it a
                // transport error would leak capacity on the chosen worker.
                self.mark_failure(index);
                format!("replica {} is unavailable: {err}", worker.name)
            })?;
        let status = response.status().as_u16();
        let headers = Self::relay_headers(response.headers());
        if (400..500).contains(&status) {
            // Client errors are relayed verbatim and never count against the
            // worker: one bad request must not pollute healthy replicas.
            self.mark_success(index);
        } else if status >= 500 {
            self.mark_failure(index);
        } else {
            self.mark_success(index);
        }
        let bytes = response.bytes().await.unwrap_or_default();
        profiler.record_hop(milestone::HOP_RECEIVED, &hop_id, "gateway", &worker.url, "request");
        let mut out = (StatusCode::from_u16(status).unwrap_or(StatusCode::BAD_GATEWAY), bytes.to_vec()).into_response();
        if !headers.is_empty() {
            *out.headers_mut() = headers;
        }
        Ok(out)
    }

    pub async fn ready(&self) -> Value {
        let mut healthy = 0usize;
        let total = self.config.workers.len();
        for worker in &self.config.workers {
            let url = format!("{}{}", worker.url.trim_end_matches('/'), self.config.health_path);
            if let Ok(response) = self.client.get(&url).send().await {
                if response.status().is_success() {
                    healthy += 1;
                }
            }
        }
        json!({"ready": healthy == total, "healthy": healthy, "total": total})
    }

    pub fn snapshot(&self) -> Value {
        let stats = self.stats.lock().unwrap();
        json!({
            "policy": self.config.policy_str(),
            "workers": self.config.workers.iter().map(|worker| {
                let stat = stats.get(&worker.name).cloned().unwrap_or_default();
                json!({
                    "name": worker.name,
                    "url": worker.url,
                    "inflight": stat.inflight,
                    "requests": stat.requests,
                    "failures": stat.failures,
                    "consecutive_failures": stat.consecutive_failures,
                    "disabled_until_ns": stat.disabled_until_ns,
                })
            }).collect::<Vec<_>>(),
        })
    }
}

fn now_ns() -> u128 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_nanos()
}

fn rand_index() -> usize {
    // Process-local pseudo-randomness is enough for worker selection.
    use std::sync::atomic::{AtomicU64, Ordering};
    static SEED: AtomicU64 = AtomicU64::new(0x9E37_79B9_7F4A_7C15);
    let x = SEED.fetch_add(1, Ordering::Relaxed);
    let mut x = x ^ (x << 13);
    x ^= x >> 7;
    x ^= x << 17;
    x as usize
}

pub async fn router_ready(State(state): State<crate::app::SharedState>) -> Response {
    match &state.replica_router {
        Some(router) => Json(router.ready().await).into_response(),
        None => (StatusCode::NOT_FOUND, "router is disabled").into_response(),
    }
}

pub async fn router_status(State(state): State<crate::app::SharedState>) -> Response {
    match &state.replica_router {
        Some(router) => Json(router.snapshot()).into_response(),
        None => (StatusCode::NOT_FOUND, "router is disabled").into_response(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn config() -> RouterConfig {
        RouterConfig {
            workers: vec![
                WorkerSpec { url: "http://a.example".into(), name: "a".into() },
                WorkerSpec { url: "http://b.example".into(), name: "b".into() },
            ],
            ..RouterConfig::default()
        }
    }

    #[test]
    fn config_validation_matches_python() {
        assert!(ReplicaRouter::new(config()).is_ok());
        let mut bad = config();
        bad.workers.clear();
        assert_eq!(ReplicaRouter::new(bad).err().unwrap(), "router requires at least one worker");
        let mut bad = config();
        bad.workers[1].name = "a".into();
        assert_eq!(ReplicaRouter::new(bad).err().unwrap(), "router worker names must be unique");
        let mut bad = config();
        bad.policy = Policy::Random;
        bad.workers.pop();
        assert!(ReplicaRouter::new(bad).err().unwrap().contains("requires two or more workers"));
        let mut bad = config();
        bad.health_path = "health".into();
        assert_eq!(ReplicaRouter::new(bad).err().unwrap(), "health_path must start with '/'");
    }

    #[test]
    fn round_robin_selects_all_workers() {
        let router = ReplicaRouter::new(config()).unwrap();
        let mut seen = std::collections::HashSet::new();
        for _ in 0..8 { seen.insert(router.select().unwrap()); }
        assert_eq!(seen.len(), 2);
    }

    #[test]
    fn failures_disable_worker_then_expire() {
        let router = ReplicaRouter::new(RouterConfig {
            cooldown_s: 0.001,
            ..config()
        }).unwrap();
        for _ in 0..3 { router.mark_failure(0); }
        assert!(!router.enabled_workers().contains(&0));
        std::thread::sleep(Duration::from_millis(5));
        assert!(router.enabled_workers().contains(&0));
    }

    #[tokio::test]
    async fn route_records_hop_pair_in_profiling_view() {
        use tokio::io::{AsyncReadExt, AsyncWriteExt};

        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        let mock = tokio::spawn(async move {
            let (mut socket, _) = listener.accept().await.unwrap();
            let mut buffer = vec![0u8; 8192];
            let _ = socket.read(&mut buffer).await.unwrap_or_default();
            let body = "{\"ok\":true}";
            let response = format!(
                "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
                body.len()
            );
            socket.write_all(response.as_bytes()).await.unwrap();
        });

        let path = std::env::temp_dir().join(format!(
            "megalm-router-hop-{}",
            uuid::Uuid::new_v4().simple()
        ));
        let profiler = Profiler::for_test(path.clone());
        let router = ReplicaRouter::new(RouterConfig {
            workers: vec![WorkerSpec {
                url: format!("http://{addr}"),
                name: "mock".into(),
            }],
            ..RouterConfig::default()
        })
        .unwrap();
        let response = router
            .route(json!({"request_id": "hop-req"}), "/v1/chat/completions", &profiler)
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::OK);
        mock.await.unwrap();
        drop(profiler);

        let view = crate::profiling::view_files(std::slice::from_ref(&path), Some("hop-req"));
        let hops = view["hops"].as_array().unwrap();
        assert_eq!(hops.len(), 1, "a router request must emit one hop pair");
        assert_eq!(hops[0]["request_id"], "hop-req");
        assert_eq!(hops[0]["src"], "gateway");
        assert_eq!(hops[0]["chunk_id"], "request");
        assert_eq!(hops[0]["matched"], true);
        let _ = std::fs::remove_file(path);
    }

    #[tokio::test]
    async fn transport_error_releases_inflight_slot() {
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        drop(listener); // force connection refused
        let router = ReplicaRouter::new(RouterConfig {
            workers: vec![WorkerSpec {
                url: format!("http://{addr}"),
                name: "dead".into(),
            }],
            ..RouterConfig::default()
        })
        .unwrap();
        let profiler = crate::profiling::Profiler::disabled();
        assert!(router.route(json!({}), "/v1/chat/completions", &profiler).await.is_err());
        let snapshot = router.snapshot();
        let worker = &snapshot["workers"][0];
        assert_eq!(worker["inflight"], 0, "failed relay must not leak capacity");
        assert_eq!(worker["failures"], 1);
    }
}
