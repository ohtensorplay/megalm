//! In-process declarative stage-graph relay (port of `stage_graph.py`).
//!
//! Stages are colocated worker threads. Every relay carries request id,
//! monotonic sequence number, deadline, cancel flag, chunk id, ACK and a
//! terminal error; payloads are owned byte buffers (the Python version uses
//! sealed memfds, which are unnecessary for Rust's in-process ownership).

use std::collections::{HashMap, HashSet, VecDeque};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
#[allow(unused_imports)]
use std::thread::{self, JoinHandle};
#[allow(unused_imports)]
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use axum::extract::State;
use axum::http::StatusCode;
use axum::response::IntoResponse;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};

#[derive(Debug, Clone, PartialEq, Eq, Deserialize, Serialize)]
pub enum FinishReason {
    Stop,
    Length,
    Error,
}

impl FinishReason {
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Stop => "stop",
            Self::Length => "length",
            Self::Error => "error",
        }
    }
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct StageSpec {
    pub name: String,
    #[serde(default)]
    pub next: Vec<String>,
    #[serde(default)]
    pub stream_to: Vec<String>,
    #[serde(default)]
    pub wait_for: Vec<String>,
    #[serde(default = "default_placement")]
    pub placement: String,
}

fn default_placement() -> String {
    "cpu".into()
}

impl StageSpec {
    pub fn validate(&self) -> Result<(), String> {
        if self.name.trim().is_empty() {
            return Err("stage name must not be empty".into());
        }
        if self.placement.trim().is_empty() {
            return Err(format!("stage {} placement is empty", self.name));
        }
        if duplicates(&self.next) {
            return Err(format!("stage {} has duplicate next stages", self.name));
        }
        if duplicates(&self.stream_to) {
            return Err(format!("stage {} has duplicate stream_to stages", self.name));
        }
        let next: HashSet<&str> = self.next.iter().map(String::as_str).collect();
        if self.stream_to.iter().any(|name| !next.contains(name.as_str())) {
            return Err(format!("stage {} streams to a non-successor", self.name));
        }
        if self.wait_for.iter().any(|name| name == &self.name) {
            return Err(format!("stage {} references itself", self.name));
        }
        Ok(())
    }
}

fn duplicates(items: &[String]) -> bool {
    let mut seen = HashSet::new();
    items.iter().any(|item| !seen.insert(item))
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct StageTopology {
    pub stages: Vec<StageSpec>,
    #[serde(default)]
    pub vram_budget_bytes: i64,
    #[serde(default)]
    pub stage_vram_bytes: HashMap<String, i64>,
}

impl StageTopology {
    pub fn validate(&self) -> Result<(), String> {
        if self.stages.is_empty() {
            return Err("stage names must be non-empty and unique".into());
        }
        let mut names = HashSet::new();
        for stage in &self.stages {
            stage.validate()?;
            if !names.insert(stage.name.clone()) {
                return Err("stage names must be non-empty and unique".into());
            }
        }
        for stage in &self.stages {
            for target in stage.next.iter().chain(stage.stream_to.iter()).chain(stage.wait_for.iter()) {
                if !names.contains(target) {
                    return Err(format!("stage {} references unknown stage {}", stage.name, target));
                }
            }
        }
        self.check_cycles()?;
        if self.vram_budget_bytes < 0 {
            return Err("VRAM budget cannot be negative".into());
        }
        let mut total = 0i64;
        for (name, bytes) in &self.stage_vram_bytes {
            if !names.contains(name) {
                return Err("VRAM budget names an unknown stage".into());
            }
            if *bytes < 0 || *bytes > self.vram_budget_bytes {
                return Err(format!("stage VRAM budget {name} exceeds {}", self.vram_budget_bytes));
            }
            total += bytes;
        }
        if total > self.vram_budget_bytes {
            return Err(format!("stage VRAM budget exceeds {}", self.vram_budget_bytes));
        }
        Ok(())
    }

    fn check_cycles(&self) -> Result<(), String> {
        let mut indegree: HashMap<&str, usize> = HashMap::new();
        let mut outgoing: HashMap<&str, Vec<&str>> = HashMap::new();
        for stage in &self.stages {
            indegree.entry(&stage.name).or_insert(0);
            outgoing.entry(&stage.name).or_default();
        }
        for stage in &self.stages {
            for target in &stage.next {
                *indegree.get_mut(target.as_str()).unwrap() += 1;
                outgoing.get_mut(stage.name.as_str()).unwrap().push(target);
            }
        }
        let mut queue: VecDeque<&str> = indegree
            .iter()
            .filter(|(_, degree)| **degree == 0)
            .map(|(name, _)| *name)
            .collect();
        let mut visited = 0usize;
        while let Some(name) = queue.pop_front() {
            visited += 1;
            for target in outgoing.get(name).into_iter().flatten() {
                let degree = indegree.get_mut(*target).unwrap();
                *degree -= 1;
                if *degree == 0 {
                    queue.push_back(target);
                }
            }
        }
        if visited != self.stages.len() {
            return Err("stage graph has a cycle".into());
        }
        Ok(())
    }

    pub fn roots(&self) -> Vec<String> {
        let mut referenced = HashSet::new();
        for stage in &self.stages {
            for target in &stage.next {
                referenced.insert(target.clone());
            }
        }
        self.stages
            .iter()
            .filter(|stage| !referenced.contains(&stage.name))
            .map(|stage| stage.name.clone())
            .collect()
    }

    #[allow(dead_code)]
    pub fn terminals(&self) -> Vec<String> {
        self.stages
            .iter()
            .filter(|stage| stage.next.is_empty())
            .map(|stage| stage.name.clone())
            .collect()
    }

    pub fn stage(&self, name: &str) -> Option<&StageSpec> {
        self.stages.iter().find(|stage| stage.name == name)
    }
}

#[allow(dead_code)]
#[derive(Debug, Clone)]
pub struct StageEnvelope {
    pub request_id: String,
    pub stage: String,
    pub sequence_no: u64,
    pub payload: Vec<u8>,
    pub payload_size: usize,
    pub deadline_ns: Option<u128>,
    pub cancelled: bool,
    pub chunk_id: String,
    pub ack: bool,
    pub error: Option<String>,
}

#[allow(dead_code)]
impl StageEnvelope {
    pub fn is_cancelled(&self) -> bool {
        self.cancelled
    }

    pub fn payload(&self) -> &[u8] {
        &self.payload
    }

    pub fn metadata(&self) -> Value {
        json!({
            "request_id": self.request_id,
            "stage": self.stage,
            "sequence_no": self.sequence_no,
            "payload_size": self.payload_size,
            "deadline_ns": self.deadline_ns,
            "cancelled": self.cancelled,
            "chunk_id": self.chunk_id,
            "ack": self.ack,
            "error": self.error,
        })
    }
}

#[allow(dead_code)]
#[derive(Debug, Clone)]
pub struct StageOutput {
    pub payload: Vec<u8>,
    pub finish_reason: FinishReason,
    pub error: Option<String>,
    pub stream: bool,
}

impl StageOutput {
    pub fn new(payload: Vec<u8>) -> Self {
        Self { payload, finish_reason: FinishReason::Stop, error: None, stream: false }
    }
}

#[derive(Debug, Clone)]
pub struct StageResult {
    pub request_id: String,
    pub payloads: Vec<(String, Vec<u8>)>,
    pub finish_reason: FinishReason,
    pub error: Option<String>,
    pub ack_sequence_no: u64,
}

pub type StageHandlerFn = Arc<dyn Fn(StageEnvelope) -> StageOutput + Send + Sync>;

#[derive(Clone)]
pub struct StageGraphRuntime {
    topology: Arc<StageTopology>,
    handlers: Arc<HashMap<String, StageHandlerFn>>,
    #[allow(dead_code)]
    queue_limit: usize,
    sequence: Arc<AtomicU64>,
    running: Arc<Mutex<bool>>,
    cancel: Arc<Mutex<HashSet<String>>>,
    results: Arc<Mutex<HashMap<String, StageResult>>>,
    #[allow(dead_code)]
    threads: Arc<Mutex<Vec<JoinHandle<()>>>>,
}

impl StageGraphRuntime {
    pub fn new(topology: StageTopology, handlers: HashMap<String, StageHandlerFn>) -> Result<Self, String> {
        topology.validate()?;
        for stage in &topology.stages {
            if !handlers.contains_key(&stage.name) {
                return Err(format!("missing stage handlers {}", stage.name));
            }
        }
        Ok(Self {
            topology: Arc::new(topology),
            handlers: Arc::new(handlers),
            queue_limit: 64,
            sequence: Arc::new(AtomicU64::new(1)),
            running: Arc::new(Mutex::new(false)),
            cancel: Arc::new(Mutex::new(HashSet::new())),
            results: Arc::new(Mutex::new(HashMap::new())),
            threads: Arc::new(Mutex::new(Vec::new())),
        })
    }

    pub fn started(&self) -> bool {
        *self.running.lock().unwrap()
    }

    pub fn start(&self) -> Result<(), String> {
        let mut running = self.running.lock().unwrap();
        if *running {
            return Err("stage graph is already running".into());
        }
        *running = true;
        Ok(())
    }

    #[allow(dead_code)]
    pub fn close(&self) {
        *self.running.lock().unwrap() = false;
        let handles: Vec<JoinHandle<()>> = self.threads.lock().unwrap().drain(..).collect();
        for handle in handles {
            let _ = handle.join();
        }
    }

    pub fn submit(&self, request_id: &str, payload: Vec<u8>) -> Result<u64, String> {
        if !self.started() {
            return Err("stage graph is not running".into());
        }
        if self.results.lock().unwrap().contains_key(request_id) {
            return Err(format!("duplicate stage request {request_id}"));
        }
        let sequence_no = self.sequence.fetch_add(1, Ordering::Relaxed);
        self.results.lock().unwrap().insert(
            request_id.to_string(),
            StageResult {
                request_id: request_id.to_string(),
                payloads: Vec::new(),
                finish_reason: FinishReason::Stop,
                error: None,
                ack_sequence_no: sequence_no,
            },
        );
        let payload_size = payload.len();
        let roots = self.topology.roots();
        let cancelled = self.cancel.lock().unwrap().contains(request_id);
        let results = self.results.clone();
        let topology = self.topology.clone();
        let handlers = self.handlers.clone();
        // Process inline so the result is complete the moment submit returns;
        // background threads could otherwise expose a partially-filled result
        // to an immediate /stage/result poll.
        for root in roots {
            let envelope = StageEnvelope {
                request_id: request_id.to_string(),
                stage: root,
                sequence_no: self.sequence.fetch_add(1, Ordering::Relaxed),
                payload: payload.clone(),
                payload_size,
                deadline_ns: None,
                cancelled,
                chunk_id: "0".into(),
                ack: false,
                error: None,
            };
            Self::process(&topology, &handlers, &results, request_id, envelope);
        }
        Ok(sequence_no)
    }

    fn process(
        topology: &StageTopology,
        handlers: &HashMap<String, StageHandlerFn>,
        results: &Arc<Mutex<HashMap<String, StageResult>>>,
        request_id: &str,
        envelope: StageEnvelope,
    ) {
        if envelope.is_cancelled() {
            Self::emit(results, request_id, &envelope.stage, Vec::new(), FinishReason::Stop);
            return;
        }
        let Some(handler) = handlers.get(&envelope.stage) else { return };
        let output = handler(envelope.clone());
        if let Some(error) = output.error {
            Self::emit_error(results, request_id, error);
            return;
        }
        Self::emit(results, request_id, &envelope.stage, output.payload.clone(), output.finish_reason.clone());
        if output.finish_reason != FinishReason::Stop {
            return;
        }
        let Some(stage) = topology.stage(&envelope.stage) else { return };
        for next in &stage.next {
            let streaming = stage.stream_to.contains(next);
            let child = StageEnvelope {
                request_id: request_id.to_string(),
                stage: next.clone(),
                sequence_no: envelope.sequence_no + 1,
                payload: output.payload.clone(),
                payload_size: output.payload.len(),
                deadline_ns: envelope.deadline_ns,
                cancelled: envelope.cancelled,
                chunk_id: if streaming { envelope.chunk_id.clone() } else { "0".into() },
                ack: false,
                error: None,
            };
            Self::process(topology, handlers, results, request_id, child);
        }
    }

    fn emit(
        results: &Arc<Mutex<HashMap<String, StageResult>>>,
        request_id: &str,
        stage: &str,
        payload: Vec<u8>,
        finish_reason: FinishReason,
    ) {
        if let Some(result) = results.lock().unwrap().get_mut(request_id) {
            result.payloads.push((stage.to_string(), payload));
            if finish_reason != FinishReason::Stop {
                result.finish_reason = finish_reason;
            }
        }
    }

    fn emit_error(results: &Arc<Mutex<HashMap<String, StageResult>>>, request_id: &str, error: String) {
        if let Some(result) = results.lock().unwrap().get_mut(request_id) {
            result.finish_reason = FinishReason::Error;
            result.error = Some(error);
        }
    }

    #[allow(dead_code)]
    pub fn cancel(&self, request_id: &str) -> bool {
        self.cancel.lock().unwrap().insert(request_id.to_string())
    }

    pub fn drain(&self, request_id: &str) -> Option<StageResult> {
        self.results.lock().unwrap().remove(request_id)
    }

    pub fn snapshot(&self) -> Value {
        json!({
            "running": self.started(),
            "stages": self.topology.stages.iter().map(|stage| json!({
                "name": stage.name,
                "next": stage.next,
                "stream_to": stage.stream_to,
                "wait_for": stage.wait_for,
                "placement": stage.placement,
            })).collect::<Vec<_>>(),
            "pending": self.results.lock().unwrap().len(),
        })
    }
}

pub async fn stage_graph_submit(
    State(state): State<crate::app::SharedState>,
    axum::Json(payload): axum::Json<Value>,
) -> axum::response::Response {
    let Some(graph) = &state.stage_graph else {
        return (StatusCode::NOT_FOUND, "stage graph is disabled").into_response();
    };
    let request_id = payload
        .get("request_id")
        .and_then(Value::as_str)
        .unwrap_or("")
        .to_string();
    let bytes = payload
        .get("payload")
        .and_then(Value::as_str)
        .map(|text| text.as_bytes().to_vec())
        .unwrap_or_default();
    match graph.submit(&request_id, bytes) {
        Ok(sequence) => axum::Json(json!({"request_id": request_id, "ack_sequence_no": sequence})).into_response(),
        Err(message) => (StatusCode::BAD_REQUEST, message).into_response(),
    }
}

pub async fn stage_graph_result(
    State(state): State<crate::app::SharedState>,
    axum::extract::Path(request_id): axum::extract::Path<String>,
) -> axum::response::Response {
    let Some(graph) = &state.stage_graph else {
        return (StatusCode::NOT_FOUND, "stage graph is disabled").into_response();
    };
    match graph.drain(&request_id) {
        Some(result) => axum::Json(json!({
            "request_id": result.request_id,
            "payloads": result.payloads.iter().map(|(stage, bytes)| json!({
                "stage": stage,
                "payload": String::from_utf8_lossy(bytes),
            })).collect::<Vec<_>>(),
            "finish_reason": result.finish_reason.as_str(),
            "error": result.error,
            "ack_sequence_no": result.ack_sequence_no,
        })).into_response(),
        None => (StatusCode::NOT_FOUND, "no result for request").into_response(),
    }
}

pub async fn stage_graph_status(
    State(state): State<crate::app::SharedState>,
) -> axum::response::Response {
    match &state.stage_graph {
        Some(graph) => axum::Json(graph.snapshot()).into_response(),
        None => (StatusCode::NOT_FOUND, "stage graph is disabled").into_response(),
    }
}

#[allow(dead_code)]
fn now_ns() -> u128 {
    SystemTime::now().duration_since(UNIX_EPOCH).unwrap_or_default().as_nanos()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn topology() -> StageTopology {
        StageTopology {
            stages: vec![
                StageSpec { name: "a".into(), next: vec!["b".into()], stream_to: vec![], wait_for: vec![], placement: "cpu".into() },
                StageSpec { name: "b".into(), next: vec![], stream_to: vec![], wait_for: vec![], placement: "cpu".into() },
            ],
            vram_budget_bytes: 1024,
            stage_vram_bytes: HashMap::new(),
        }
    }

    #[test]
    fn topology_validation_matches_python_rules() {
        assert!(topology().validate().is_ok());
        let mut bad = topology();
        bad.stages[0].next = vec!["b".into(), "b".into()];
        assert_eq!(topology_with(bad).err().as_deref(), Some("stage a has duplicate next stages"));
        let mut bad = topology();
        bad.stages[0].next = vec!["missing".into()];
        assert!(topology_with(bad).err().is_some_and(|e| e.contains("references unknown stage")));
        let mut bad = topology();
        bad.stages[1].next = vec!["a".into()];
        assert_eq!(topology_with(bad).err().as_deref(), Some("stage graph has a cycle"));
    }

    fn topology_with(value: StageTopology) -> Result<(), String> { value.validate() }

    #[test]
    fn runtime_fans_out_payloads_and_tracks_results() {
        let handler: StageHandlerFn = Arc::new(|envelope| {
            let mut payload = envelope.payload;
            payload.push(b'!');
            StageOutput::new(payload)
        });
        let mut handlers = HashMap::new();
        handlers.insert("a".to_string(), handler.clone());
        handlers.insert("b".to_string(), handler);
        let runtime = StageGraphRuntime::new(topology(), handlers).unwrap();
        runtime.start().unwrap();
        let ack = runtime.submit("req-1", b"hi".to_vec()).unwrap();
        let mut result = None;
        for _ in 0..100 {
            if let Some(value) = runtime.drain("req-1") { result = Some(value); break; }
            std::thread::sleep(Duration::from_millis(5));
        }
        let result = result.unwrap();
        assert_eq!(result.ack_sequence_no, ack);
        assert_eq!(result.payloads.len(), 2);
        assert_eq!(result.payloads[1].1, b"hi!!".to_vec());
        runtime.close();
    }

    #[test]
    fn cancel_prevents_stage_execution() {
        let handler: StageHandlerFn = Arc::new(|_| StageOutput::new(vec![1]));
        let mut handlers = HashMap::new();
        handlers.insert("a".to_string(), handler.clone());
        handlers.insert("b".to_string(), handler);
        let runtime = StageGraphRuntime::new(topology(), handlers).unwrap();
        runtime.start().unwrap();
        runtime.cancel("req-cancel");
        runtime.submit("req-cancel", vec![7]).unwrap();
        std::thread::sleep(Duration::from_millis(30));
        let result = runtime.drain("req-cancel").unwrap();
        assert!(result.payloads[0].1.is_empty());
        runtime.close();
    }
}
