//! Opt-in gateway control plane, ported from the Python composition-root
//! plugins (`admin.py`, `controller.py`, `weight_update.py`, `lease_tracker.py`).
//!
//! All state is process-local and no GPU worker changes are required: the
//! weight-update control message reuses the existing MGLM v6 INSPECT frame.

use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use axum::extract::State;
use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use axum::Json;
use megalm_protocol::Inspect;
use serde::{Deserialize, Serialize};
use serde_json::{json, Map, Value};

use crate::app::SharedState;

/// Wall-clock nanoseconds, used by admin `paused_at_ns` / `resumed_at_ns`
/// (Python's `time.time_ns()`).
fn now_ns() -> u128 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_nanos()
}

/// Monotonic nanoseconds, used by controller leases and replica heartbeats
/// (Python's `time.monotonic_ns()`).
fn monotonic_ns() -> u128 {
    let mut ts = libc::timespec {
        tv_sec: 0,
        tv_nsec: 0,
    };
    // SAFETY: `timespec` is a plain POD struct and CLOCK_MONOTONIC is
    // supported on every platform this crate targets.
    unsafe { libc::clock_gettime(libc::CLOCK_MONOTONIC, &mut ts) };
    ts.tv_sec as u128 * 1_000_000_000 + ts.tv_nsec as u128
}

fn err(status: StatusCode, message: impl Into<String>) -> Response {
    (status, Json(json!({"detail": message.into()}))).into_response()
}

// ---------------------------------------------------------------------------
// Admin pause / weights checker
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize)]
pub struct AdminState {
    pub paused: bool,
    pub reason: String,
    pub pause_count: u64,
    pub paused_at_ns: Option<u128>,
    pub resumed_at_ns: Option<u128>,
}

impl Default for AdminState {
    fn default() -> Self {
        Self {
            paused: false,
            reason: String::new(),
            pause_count: 0,
            paused_at_ns: None,
            resumed_at_ns: None,
        }
    }
}

#[derive(Clone)]
pub struct AdminController {
    state: Arc<Mutex<AdminState>>,
}

impl AdminController {
    pub fn new() -> Self {
        Self {
            state: Arc::new(Mutex::new(AdminState::default())),
        }
    }

    pub fn is_paused(&self) -> bool {
        self.state.lock().unwrap().paused
    }

    pub fn pause(&self, reason: &str) -> Value {
        let mut state = self.state.lock().unwrap();
        state.paused = true;
        state.reason = reason.to_string();
        state.pause_count += 1;
        state.paused_at_ns = Some(now_ns());
        tracing::warn!(reason = %reason, "admin_paused");
        serde_json::to_value(&*state).unwrap_or(Value::Null)
    }

    pub fn resume(&self) -> Value {
        let mut state = self.state.lock().unwrap();
        state.paused = false;
        state.reason.clear();
        state.resumed_at_ns = Some(now_ns());
        tracing::info!("admin_resumed");
        serde_json::to_value(&*state).unwrap_or(Value::Null)
    }

    pub fn snapshot(&self) -> Value {
        serde_json::to_value(&*self.state.lock().unwrap()).unwrap_or(Value::Null)
    }

    /// Exact projection from Python `AdminController.weights_checker()`:
    /// startup.{draft_model_digest,kernel_binding_digest,compiled.digest,
    /// tensor_parallel} + hardware.{accelerator,device_name}.
    pub fn weights_checker(descriptor: &Value) -> Value {
        let empty = json!({});
        let startup = descriptor
            .get("startup")
            .filter(|value| value.is_object())
            .unwrap_or(&empty);
        let hardware = descriptor
            .get("hardware")
            .filter(|value| value.is_object())
            .unwrap_or(&empty);
        let compiled = startup
            .get("compiled")
            .filter(|value| value.is_object())
            .unwrap_or(&empty);
        let pick = |section: &Value, key: &str| -> Value {
            section.get(key).cloned().unwrap_or(Value::Null)
        };
        json!({
            "model_digest": pick(startup, "draft_model_digest"),
            "kernel_binding_digest": pick(startup, "kernel_binding_digest"),
            "compiled_digest": pick(compiled, "digest"),
            "tensor_parallel": pick(startup, "tensor_parallel"),
            "accelerator": pick(hardware, "accelerator"),
            "device_name": pick(hardware, "device_name"),
        })
    }
}

#[derive(Debug, Deserialize)]
pub struct PauseBody {
    #[serde(default)]
    reason: String,
}

pub async fn admin_pause(
    State(state): State<SharedState>,
    Json(body): Json<PauseBody>,
) -> Response {
    if let Some(admin) = &state.admin {
        Json(admin.pause(&body.reason)).into_response()
    } else {
        err(StatusCode::NOT_FOUND, "admin control plane is disabled")
    }
}

pub async fn admin_continue(State(state): State<SharedState>) -> Response {
    if let Some(admin) = &state.admin {
        Json(admin.resume()).into_response()
    } else {
        err(StatusCode::NOT_FOUND, "admin control plane is disabled")
    }
}

pub async fn admin_status(State(state): State<SharedState>) -> Response {
    if let Some(admin) = &state.admin {
        Json(admin.snapshot()).into_response()
    } else {
        err(StatusCode::NOT_FOUND, "admin control plane is disabled")
    }
}

pub async fn admin_weights_checker(State(state): State<SharedState>) -> Response {
    if state.admin.is_none() {
        return err(StatusCode::NOT_FOUND, "admin control plane is disabled");
    }
    Json(AdminController::weights_checker(&state.client.descriptor().await)).into_response()
}

/// POST /admin/weights/checker - compare supplied expectations against the
/// live descriptor view. Shape matches Python's `verify_weights`: non-null
/// expected values that differ become `mismatches[key] = {expected, actual}`.
pub async fn admin_verify_weights(
    State(state): State<SharedState>,
    Json(body): Json<Value>,
) -> Response {
    if state.admin.is_none() {
        return err(StatusCode::NOT_FOUND, "admin control plane is disabled");
    }
    let current = AdminController::weights_checker(&state.client.descriptor().await);
    let mut mismatches = Map::new();
    if let (Some(expected), Some(current)) = (body.as_object(), current.as_object()) {
        for (key, value) in expected {
            if value.is_null() || !current.contains_key(key) {
                continue;
            }
            if value.to_string() != current[key].to_string() {
                mismatches.insert(
                    key.clone(),
                    json!({"expected": value, "actual": current[key]}),
                );
            }
        }
    }
    Json(json!({
        "match": mismatches.is_empty(),
        "mismatches": mismatches,
        "current": current,
    }))
    .into_response()
}

// ---------------------------------------------------------------------------
// Replica registry + placement leases
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum ReplicaStatus {
    Active,
    Draining,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct ReplicaRegistration {
    pub replica_id: String,
    pub url: String,
    pub node_id: String,
    pub capacity: i64,
    #[serde(default)]
    pub prefixes: Vec<String>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct IdBody {
    #[serde(default)]
    pub replica_id: String,
    pub capacity: Option<i64>,
    pub prefixes: Option<Vec<String>>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct LeaseBody {
    pub request_id: String,
    #[serde(default)]
    pub prefix_fingerprint: String,
}

#[derive(Debug, Clone, Deserialize)]
pub struct ReleaseBody {
    #[serde(default)]
    pub lease_id: String,
}

#[derive(Debug, Clone)]
struct ReplicaState {
    registration: ReplicaRegistration,
    status: ReplicaStatus,
    inflight: i64,
    failures: i64,
    last_heartbeat_ns: u128,
}

#[derive(Debug, Clone, Serialize)]
pub struct PlacementLease {
    pub lease_id: String,
    pub request_id: String,
    pub replica_id: String,
    pub prefix_fingerprint: String,
    pub expires_ns: u128,
    pub first_token: bool,
}

impl PlacementLease {
    /// Python `PlacementLease.describe()` - always includes `replayable`.
    fn describe(&self) -> Value {
        let mut value = serde_json::to_value(self).unwrap_or(Value::Null);
        if let Value::Object(map) = &mut value {
            map.insert("replayable".into(), json!(!self.first_token));
        }
        value
    }
}

#[derive(Debug, Clone)]
struct LeaseState {
    lease: PlacementLease,
}

pub struct Controller {
    replicas: Arc<Mutex<HashMap<String, ReplicaState>>>,
    leases: Arc<Mutex<HashMap<String, LeaseState>>>,
    next_lease: AtomicU64,
    lease_ttl: Duration,
    heartbeat_ttl: Duration,
}

impl Controller {
    pub fn new() -> Self {
        Self {
            replicas: Arc::new(Mutex::new(HashMap::new())),
            leases: Arc::new(Mutex::new(HashMap::new())),
            next_lease: AtomicU64::new(1),
            lease_ttl: Duration::from_secs(30),
            heartbeat_ttl: Duration::from_secs(90),
        }
    }

    /// Port of Python `_sweep_locked`: drop stale replicas, expire leases
    /// (releasing their inflight slot), then drop leases whose replica is gone.
    fn sweep(&self) {
        let now = monotonic_ns();
        let heartbeat_ttl = self.heartbeat_ttl.as_nanos();
        let mut replicas = self.replicas.lock().unwrap();
        replicas.retain(|_, state| now - state.last_heartbeat_ns <= heartbeat_ttl);
        let mut leases = self.leases.lock().unwrap();
        let expired: Vec<PlacementLease> = leases
            .values()
            .filter(|state| state.lease.expires_ns <= now)
            .map(|state| state.lease.clone())
            .collect();
        for lease in &expired {
            if let Some(state) = replicas.get_mut(&lease.replica_id) {
                state.inflight = (state.inflight - 1).max(0);
            }
        }
        leases.retain(|_, state| state.lease.expires_ns > now);
        leases.retain(|_, state| replicas.contains_key(&state.lease.replica_id));
    }

    pub fn register(&self, registration: ReplicaRegistration) -> Result<Value, String> {
        if registration.replica_id.trim().is_empty()
            || registration.url.trim().is_empty()
            || registration.node_id.trim().is_empty()
        {
            return Err("replica identity must not be empty".into());
        }
        if registration.capacity <= 0 {
            return Err("replica capacity must be positive".into());
        }
        let mut replicas = self.replicas.lock().unwrap();
        if replicas.contains_key(&registration.replica_id) {
            return Err(format!(
                "replica '{}' is already registered",
                registration.replica_id
            ));
        }
        let value = serde_json::to_value(&registration).unwrap_or(Value::Null);
        replicas.insert(
            registration.replica_id.clone(),
            ReplicaState {
                registration,
                status: ReplicaStatus::Active,
                inflight: 0,
                failures: 0,
                last_heartbeat_ns: monotonic_ns(),
            },
        );
        Ok(value)
    }

    /// Returns false for unknown replicas, matching Python's bool contract.
    pub fn heartbeat(&self, replica_id: &str, capacity: Option<i64>, prefixes: Option<Vec<String>>) -> bool {
        self.sweep();
        let mut replicas = self.replicas.lock().unwrap();
        let Some(state) = replicas.get_mut(replica_id) else { return false };
        state.last_heartbeat_ns = monotonic_ns();
        if let Some(capacity) = capacity {
            state.registration.capacity = capacity;
        }
        if let Some(prefixes) = prefixes {
            state.registration.prefixes = prefixes;
        }
        true
    }

    pub fn drain(&self, replica_id: &str) -> bool {
        self.sweep();
        let mut replicas = self.replicas.lock().unwrap();
        let Some(state) = replicas.get_mut(replica_id) else { return false };
        state.status = ReplicaStatus::Draining;
        true
    }

    pub fn activate(&self, replica_id: &str) -> bool {
        self.sweep();
        let mut replicas = self.replicas.lock().unwrap();
        let Some(state) = replicas.get_mut(replica_id) else { return false };
        state.status = ReplicaStatus::Active;
        true
    }

    /// Refuses to deregister a replica with active leases (Python returns false).
    #[allow(dead_code)]
    pub fn deregister(&self, replica_id: &str) -> bool {
        self.sweep();
        let mut replicas = self.replicas.lock().unwrap();
        let Some(state) = replicas.get(replica_id) else { return false };
        if state.inflight > 0 {
            return false;
        }
        replicas.remove(replica_id);
        true
    }

    pub fn acquire(&self, request_id: &str, prefix_fingerprint: &str) -> Result<Value, String> {
        self.sweep();
        let mut replicas = self.replicas.lock().unwrap();
        // Python minimises (not prefix_match, inflight, failures, replica_id):
        // matching replicas are preferred but non-matching ones stay eligible.
        let selected = replicas
            .iter()
            .filter(|(_, state)| {
                state.status == ReplicaStatus::Active && state.inflight < state.registration.capacity
            })
            .min_by_key(|(id, state)| {
                let matched = prefix_fingerprint.is_empty()
                    || state.registration.prefixes.iter().any(|p| p == prefix_fingerprint);
                (!matched, state.inflight, state.failures, (*id).clone())
            })
            .map(|(id, _)| id.clone())
            .ok_or_else(|| "no replica has capacity for this request".to_string())?;
        let state = replicas.get_mut(&selected).unwrap();
        state.inflight += 1;
        let lease_id = format!("lease-{}", self.next_lease.fetch_add(1, Ordering::Relaxed));
        let lease = PlacementLease {
            lease_id: lease_id.clone(),
            request_id: request_id.to_string(),
            replica_id: selected,
            prefix_fingerprint: prefix_fingerprint.to_string(),
            expires_ns: monotonic_ns() + self.lease_ttl.as_nanos(),
            first_token: false,
        };
        let value = lease.describe();
        self.leases
            .lock()
            .unwrap()
            .insert(lease_id, LeaseState { lease });
        Ok(value)
    }

    /// Returns false for unknown leases, matching Python's bool contract.
    pub fn release(&self, lease_id: &str) -> bool {
        let replica_id = {
            let mut leases = self.leases.lock().unwrap();
            let Some(state) = leases.remove(lease_id) else { return false };
            state.lease.replica_id
        };
        // Keep the same lock order as `sweep`/`acquire` (replicas -> leases):
        // release the lease lock before touching the replica map.
        let mut replicas = self.replicas.lock().unwrap();
        if let Some(state) = replicas.get_mut(&replica_id) {
            state.inflight = (state.inflight - 1).max(0);
        }
        true
    }

    pub fn mark_first_token(&self, lease_id: &str) -> bool {
        let mut leases = self.leases.lock().unwrap();
        let Some(state) = leases.get_mut(lease_id) else { return false };
        state.lease.first_token = true;
        true
    }

    /// Counts a failure on the replica, releases the lease and returns whether
    /// the request is replayable (no first token yet), matching Python.
    pub fn on_failure(&self, lease_id: &str) -> bool {
        let leases = self.leases.lock().unwrap();
        let Some(state) = leases.get(lease_id) else { return false };
        let replica_id = state.lease.replica_id.clone();
        let replayable = !state.lease.first_token;
        drop(leases);
        let mut replicas = self.replicas.lock().unwrap();
        if let Some(state) = replicas.get_mut(&replica_id) {
            state.failures += 1;
        }
        drop(replicas);
        self.release(lease_id);
        replayable
    }

    #[allow(dead_code)]
    pub fn placement(&self, request_id: &str, prefix: &str) -> Option<String> {
        let value = self.acquire(request_id, prefix).ok()?;
        let id = value.get("lease_id")?.as_str()?.to_string();
        let lease = self.leases.lock().unwrap().get(&id)?.clone().lease;
        self.replicas
            .lock()
            .unwrap()
            .get(&lease.replica_id)
            .map(|state| state.registration.url.clone())
    }

    pub fn snapshot(&self) -> Value {
        self.sweep();
        let replicas: Vec<Value> = {
            let replicas = self.replicas.lock().unwrap();
            let mut out: Vec<Value> = replicas
                .values()
                .map(|state| {
                    let mut value =
                        serde_json::to_value(&state.registration).unwrap_or(json!({}));
                    if let Value::Object(map) = &mut value {
                        map.insert("inflight".into(), json!(state.inflight));
                        map.insert("failures".into(), json!(state.failures));
                        map.insert(
                            "status".into(),
                            json!(match state.status {
                                ReplicaStatus::Active => "active",
                                ReplicaStatus::Draining => "draining",
                            }),
                        );
                    }
                    value
                })
                .collect();
            out.sort_by(|a, b| {
                a["replica_id"]
                    .as_str()
                    .unwrap_or("")
                    .cmp(b["replica_id"].as_str().unwrap_or(""))
            });
            out
        };
        let active_leases = self.leases.lock().unwrap().len();
        let healthy = replicas
            .iter()
            .filter(|value| value["status"] == "active")
            .count();
        json!({
            "replicas": replicas,
            "active_leases": active_leases,
            "healthy_replicas": healthy,
        })
    }

    #[allow(dead_code)]
    pub fn lease(&self, lease_id: &str) -> Option<Value> {
        self.leases
            .lock()
            .unwrap()
            .get(lease_id)
            .map(|state| state.lease.describe())
    }
}

pub async fn controller_register(
    State(state): State<SharedState>,
    Json(body): Json<ReplicaRegistration>,
) -> Response {
    match state.controller.register(body) {
        Ok(value) => Json(value).into_response(),
        Err(message) => err(StatusCode::BAD_REQUEST, message),
    }
}

pub async fn controller_heartbeat(
    State(state): State<SharedState>,
    Json(body): Json<IdBody>,
) -> Response {
    if state
        .controller
        .heartbeat(&body.replica_id, body.capacity, body.prefixes)
    {
        Json(json!({"ok": true})).into_response()
    } else {
        err(StatusCode::NOT_FOUND, "unknown replica")
    }
}

pub async fn controller_drain(
    State(state): State<SharedState>,
    Json(body): Json<IdBody>,
) -> Response {
    if state.controller.drain(&body.replica_id) {
        Json(json!({"ok": true})).into_response()
    } else {
        err(StatusCode::NOT_FOUND, "unknown replica")
    }
}

pub async fn controller_activate(
    State(state): State<SharedState>,
    Json(body): Json<IdBody>,
) -> Response {
    if state.controller.activate(&body.replica_id) {
        Json(json!({"ok": true})).into_response()
    } else {
        err(StatusCode::NOT_FOUND, "unknown replica")
    }
}

pub async fn controller_acquire(
    State(state): State<SharedState>,
    Json(body): Json<LeaseBody>,
) -> Response {
    match state
        .controller
        .acquire(&body.request_id, &body.prefix_fingerprint)
    {
        Ok(value) => Json(value).into_response(),
        Err(message) => err(StatusCode::SERVICE_UNAVAILABLE, message),
    }
}

pub async fn controller_release(
    State(state): State<SharedState>,
    Json(body): Json<ReleaseBody>,
) -> Response {
    Json(json!({"ok": state.controller.release(&body.lease_id)})).into_response()
}

pub async fn controller_status(State(state): State<SharedState>) -> Response {
    Json(state.controller.snapshot()).into_response()
}

// ---------------------------------------------------------------------------
// Lease tracker (milestone hooks)
// ---------------------------------------------------------------------------

/// Stable prefix fingerprint, byte-for-byte identical to the Python plugin:
/// `blake2b(",".join(str(t) for t in token_ids[:64]), digest_size=8).hexdigest()[:length]`.
pub fn fingerprint_token_ids(token_ids: &[u32], length: usize) -> String {
    let sample: Vec<String> = token_ids
        .iter()
        .take(64)
        .map(|token| token.to_string())
        .collect();
    if sample.is_empty() {
        return String::new();
    }
    let payload = sample.join(",").into_bytes();
    let digest = blake2b8(&payload);
    let hex: String = digest.iter().map(|b| format!("{b:02x}")).collect();
    hex.chars().take(length).collect()
}

/// Minimal BLAKE2b with an 8-byte digest (unkeyed), matching Python's
/// `hashlib.blake2b(data, digest_size=8)`.
fn blake2b8(data: &[u8]) -> [u8; 8] {
    const IV: [u64; 8] = [
        0x6a09_e667_f3bc_c908,
        0xbb67_ae85_84ca_a73b,
        0x3c6e_f372_fe94_f82b,
        0xa54f_f53a_5f1d_36f1,
        0x510e_527f_ade6_82d1,
        0x9b05_688c_2b3e_6c1f,
        0x1f83_d9ab_fb41_bd6b,
        0x5be0_cd19_137e_2179,
    ];
    const SIGMA: [[usize; 16]; 10] = [
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
        [14, 10, 4, 8, 9, 15, 13, 6, 1, 12, 0, 2, 11, 7, 5, 3],
        [11, 8, 12, 0, 5, 2, 15, 13, 10, 14, 3, 6, 7, 1, 9, 4],
        [7, 9, 3, 1, 13, 12, 11, 14, 2, 6, 5, 10, 4, 0, 15, 8],
        [9, 0, 5, 7, 2, 4, 10, 15, 14, 1, 11, 12, 6, 8, 3, 13],
        [2, 12, 6, 10, 0, 11, 8, 3, 4, 13, 7, 5, 15, 14, 1, 9],
        [12, 5, 1, 15, 14, 13, 4, 10, 0, 7, 6, 3, 9, 2, 8, 11],
        [13, 11, 7, 14, 12, 1, 3, 9, 5, 0, 15, 4, 8, 6, 2, 10],
        [6, 15, 14, 9, 11, 3, 0, 8, 12, 2, 13, 7, 1, 4, 10, 5],
        [10, 2, 8, 4, 7, 6, 1, 5, 15, 11, 9, 14, 3, 12, 13, 0],
    ];

    fn g(v: &mut [u64; 16], a: usize, b: usize, c: usize, d: usize, x: u64, y: u64) {
        v[a] = v[a].wrapping_add(v[b]).wrapping_add(x);
        v[d] = (v[d] ^ v[a]).rotate_right(32);
        v[c] = v[c].wrapping_add(v[d]);
        v[b] = (v[b] ^ v[c]).rotate_right(24);
        v[a] = v[a].wrapping_add(v[b]).wrapping_add(y);
        v[d] = (v[d] ^ v[a]).rotate_right(16);
        v[c] = v[c].wrapping_add(v[d]);
        v[b] = (v[b] ^ v[c]).rotate_right(63);
    }

    fn compress(h: &mut [u64; 8], block: &[u8], t: u128, last: bool) {
        let mut m = [0u64; 16];
        for (i, word) in m.iter_mut().enumerate() {
            let slice = &block[i * 8..i * 8 + 8];
            *word = u64::from_le_bytes(slice.try_into().unwrap());
        }
        let mut v = [0u64; 16];
        v[..8].copy_from_slice(h);
        v[8..].copy_from_slice(&IV);
        v[12] ^= t as u64;
        v[13] ^= (t >> 64) as u64;
        if last {
            v[14] = !v[14];
        }
        for round in 0..12 {
            let s = &SIGMA[round % 10];
            g(&mut v, 0, 4, 8, 12, m[s[0]], m[s[1]]);
            g(&mut v, 1, 5, 9, 13, m[s[2]], m[s[3]]);
            g(&mut v, 2, 6, 10, 14, m[s[4]], m[s[5]]);
            g(&mut v, 3, 7, 11, 15, m[s[6]], m[s[7]]);
            g(&mut v, 0, 5, 10, 15, m[s[8]], m[s[9]]);
            g(&mut v, 1, 6, 11, 12, m[s[10]], m[s[11]]);
            g(&mut v, 2, 7, 8, 13, m[s[12]], m[s[13]]);
            g(&mut v, 3, 4, 9, 14, m[s[14]], m[s[15]]);
        }
        for i in 0..8 {
            h[i] ^= v[i] ^ v[i + 8];
        }
    }

    let mut h = IV;
    h[0] ^= 0x0101_0008; // fanout=1, depth=1, key length=0, digest length=8

    let mut offset = 0usize;
    while data.len() - offset > 128 {
        let block = &data[offset..offset + 128];
        let t = (offset + 128) as u128;
        compress(&mut h, block, t, false);
        offset += 128;
    }

    let mut last_block = [0u8; 128];
    let remaining = &data[offset..];
    last_block[..remaining.len()].copy_from_slice(remaining);
    let t = data.len() as u128;
    compress(&mut h, &last_block, t, true);

    h[0].to_le_bytes()
}

#[derive(Clone, Default)]
pub struct LeaseTracker {
    controller: Option<Arc<Controller>>,
    by_request: Arc<Mutex<HashMap<String, String>>>,
    acquire_failures: Arc<AtomicU64>,
}

impl LeaseTracker {
    pub fn new(controller: Option<Arc<Controller>>) -> Self {
        Self {
            controller,
            by_request: Arc::new(Mutex::new(HashMap::new())),
            acquire_failures: Arc::new(AtomicU64::new(0)),
        }
    }

    pub fn is_enabled(&self) -> bool {
        self.controller.is_some()
    }

    /// Acquire a placement lease at request start. Matches Python's
    /// `LeaseTracker.begin`: acquire failures are counted, not fatal.
    pub fn begin(&self, request_id: &str, prefix_fingerprint: &str) {
        let Some(controller) = &self.controller else {
            return;
        };
        match controller.acquire(request_id, prefix_fingerprint) {
            Ok(value) => {
                if let Some(id) = value.get("lease_id").and_then(Value::as_str) {
                    self.by_request
                        .lock()
                        .unwrap()
                        .insert(request_id.to_string(), id.to_string());
                }
            }
            Err(message) => {
                self.acquire_failures.fetch_add(1, Ordering::Relaxed);
                tracing::warn!(
                    event = "lease_acquire_failed",
                    request_id = %request_id,
                    error = %message
                );
            }
        }
    }

    /// Mark the lease as having produced its first token.
    pub fn first_token(&self, request_id: &str) {
        let Some(controller) = &self.controller else {
            return;
        };
        let id = self.by_request.lock().unwrap().get(request_id).cloned();
        if let Some(id) = id {
            let _ = controller.mark_first_token(&id);
        }
    }

    /// Release or fail the lease at request end. Returns true when a failed
    /// request is replayable, exactly like Python's `finish`.
    pub fn finish(&self, request_id: &str, failed: bool) -> bool {
        let id = self.by_request.lock().unwrap().remove(request_id);
        let Some(id) = id else { return false };
        match &self.controller {
            Some(controller) if failed => controller.on_failure(&id),
            Some(controller) => {
                let _ = controller.release(&id);
                false
            }
            None => false,
        }
    }

    /// Snapshot shape matches Python's `lease_tracker_snapshot()`.
    pub fn snapshot(&self) -> Value {
        if !self.is_enabled() {
            return json!({"enabled": false});
        }
        json!({
            "enabled": true,
            "active": self.by_request.lock().unwrap().len(),
            "acquire_failures": self.acquire_failures.load(Ordering::Relaxed),
        })
    }
}

pub async fn lease_tracker_status(State(state): State<SharedState>) -> Response {
    Json(state.lease_tracker.snapshot()).into_response()
}

// ---------------------------------------------------------------------------
// Weight update coordinator
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct WeightUpdateRequest {
    pub path: String,
    pub expected_digest: Option<String>,
    #[serde(default = "default_source")]
    pub source: String,
    #[serde(default = "default_true")]
    pub abort_inflight: bool,
}

fn default_source() -> String {
    "disk".into()
}
fn default_true() -> bool {
    true
}

impl WeightUpdateRequest {
    fn validate(&self) -> Result<(), String> {
        if self.path.trim().is_empty() {
            return Err("weight path must not be empty".into());
        }
        if self.source != "disk" && self.source != "tensor" && self.source != "collective" {
            return Err(format!("unsupported weight source {}", self.source));
        }
        Ok(())
    }

    fn inspect_payload(&self, command_id: u64) -> Inspect {
        Inspect {
            command_id,
            op: Some("update_weights".into()),
            path: Some(self.path.clone()),
            expected_digest: self.expected_digest.clone(),
            source: Some(self.source.clone()),
            ..Inspect::default()
        }
    }
}

#[derive(Debug, Clone, Serialize)]
pub struct WeightUpdateResult {
    pub ok: bool,
    pub before_digest: Option<String>,
    pub after_digest: Option<String>,
    pub ranks: Vec<String>,
    pub duration_ms: f64,
    pub error: Option<String>,
}

pub fn aggregate_rank_results(results: &[(String, Value)]) -> WeightUpdateResult {
    if results.is_empty() {
        return WeightUpdateResult {
            ok: false,
            before_digest: None,
            after_digest: None,
            ranks: Vec::new(),
            duration_ms: 0.0,
            error: Some("no rank results to aggregate".into()),
        };
    }
    let ranks: Vec<String> = results.iter().map(|(rank, _)| rank.clone()).collect();
    for (rank, value) in results {
        let ok = value.get("ok").and_then(Value::as_bool).unwrap_or(false);
        if !ok {
            let message = value
                .get("error")
                .and_then(Value::as_str)
                .unwrap_or("unknown error");
            return WeightUpdateResult {
                ok: false,
                before_digest: None,
                after_digest: None,
                ranks: ranks.clone(),
                duration_ms: 0.0,
                error: Some(format!("rank {rank} failed: {message}")),
            };
        }
    }
    let digests: Vec<String> = results
        .iter()
        .filter_map(|(_, value)| value.get("digest").and_then(Value::as_str).map(str::to_string))
        .collect();
    let after = digests.first().cloned();
    if digests.iter().any(|digest| Some(digest) != after.as_ref()) {
        return WeightUpdateResult {
            ok: false,
            before_digest: None,
            after_digest: None,
            ranks,
            duration_ms: 0.0,
            error: Some("rank weight digests disagree after update".into()),
        };
    }
    WeightUpdateResult {
        ok: true,
        before_digest: None,
        after_digest: after,
        ranks,
        duration_ms: 0.0,
        error: None,
    }
}

pub struct WeightUpdateCoordinator {
    admin: Arc<AdminController>,
    history: Arc<Mutex<Vec<Value>>>,
    command_id: AtomicU64,
    timeout: Duration,
}

impl WeightUpdateCoordinator {
    pub fn new(admin: Arc<AdminController>) -> Result<Self, String> {
        Ok(Self {
            admin,
            history: Arc::new(Mutex::new(Vec::new())),
            command_id: AtomicU64::new(1),
            timeout: Duration::from_secs(300),
        })
    }

    /// Python returns `{"history": [{"path": ..., **result.describe()}, ...]}`.
    pub fn history(&self) -> Value {
        json!({"history": self.history.lock().unwrap().clone()})
    }

    pub async fn update(&self, state: &SharedState, request: WeightUpdateRequest) -> Response {
        if let Err(message) = request.validate() {
            return err(StatusCode::BAD_REQUEST, message);
        }
        let id = self.command_id.fetch_add(1, Ordering::Relaxed);
        let started = Instant::now();
        let _pause = self.admin.pause(&format!("weight update: {}", request.path));
        let result = match tokio::time::timeout(self.timeout, state.client.inspect(&request.inspect_payload(id))).await {
            Ok(Ok(value)) => {
                let ranks: Vec<(String, Value)> = value
                    .get("rank_results")
                    .and_then(Value::as_array)
                    .map(|items| {
                        items
                            .iter()
                            .enumerate()
                            .map(|(index, item)| (format!("rank {index}"), item.clone()))
                            .collect()
                    })
                    .unwrap_or_default();
                let mut result = aggregate_rank_results(&ranks);
                if let Some(expected) = &request.expected_digest {
                    if result.after_digest.as_deref() != Some(expected.as_str()) {
                        result.ok = false;
                        result.error =
                            Some("updated weight digest does not match expected".into());
                    }
                }
                result.duration_ms = started.elapsed().as_secs_f64() * 1000.0;
                result
            }
            Ok(Err(message)) => WeightUpdateResult {
                ok: false,
                before_digest: None,
                after_digest: None,
                ranks: Vec::new(),
                duration_ms: started.elapsed().as_secs_f64() * 1000.0,
                error: Some(message.to_string()),
            },
            Err(_) => WeightUpdateResult {
                ok: false,
                before_digest: None,
                after_digest: None,
                ranks: Vec::new(),
                duration_ms: started.elapsed().as_secs_f64() * 1000.0,
                error: Some(format!(
                    "another weight update is in progress (timeout {}s)",
                    self.timeout.as_secs()
                )),
            },
        };
        let _resume = self.admin.resume();
        let value = serde_json::to_value(&result).unwrap_or(Value::Null);
        let mut entry = json!({"path": request.path});
        if let (Value::Object(target), Value::Object(source)) = (&mut entry, &value) {
            for (key, item) in source {
                target.insert(key.clone(), item.clone());
            }
        }
        self.history.lock().unwrap().push(entry);
        if result.ok {
            Json(value).into_response()
        } else {
            // FastAPI wraps an unsuccessful result in HTTPException detail,
            // where `detail` is the result object itself.
            (StatusCode::BAD_GATEWAY, Json(json!({"detail": value}))).into_response()
        }
    }
}

pub async fn weight_update(
    State(state): State<SharedState>,
    Json(body): Json<WeightUpdateRequest>,
) -> Response {
    match &state.weight_update {
        Some(coordinator) => coordinator.update(&state, body).await,
        None => err(StatusCode::NOT_FOUND, "weight update control plane is disabled"),
    }
}

pub async fn weight_update_history(State(state): State<SharedState>) -> Response {
    match &state.weight_update {
        Some(coordinator) => Json(coordinator.history()).into_response(),
        None => err(StatusCode::NOT_FOUND, "weight update control plane is disabled"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn admin_pause_resume_snapshot_matches_python_shape() {
        let admin = AdminController::new();
        assert_eq!(admin.snapshot()["pause_count"], 0);
        admin.pause("maintenance");
        assert_eq!(admin.snapshot()["paused"], true);
        assert_eq!(admin.snapshot()["reason"], "maintenance");
        assert_eq!(admin.snapshot()["pause_count"], 1);
        assert!(admin.snapshot()["paused_at_ns"].is_u64());
        admin.resume();
        assert_eq!(admin.snapshot()["paused"], false);
        assert_eq!(admin.snapshot()["reason"], "");
        assert_eq!(admin.snapshot()["pause_count"], 1);
    }

    #[test]
    fn controller_lease_lifecycle_and_snapshot() {
        let controller = Controller::new();
        controller
            .register(ReplicaRegistration {
                replica_id: "r1".into(),
                url: "http://x".into(),
                node_id: "n1".into(),
                capacity: 2,
                prefixes: vec!["p".into()],
            })
            .unwrap();
        let lease = controller.acquire("req1", "p").unwrap();
        assert_eq!(lease["lease_id"], "lease-1");
        assert_eq!(lease["replica_id"], "r1");
        assert_eq!(controller.placement("req2", "p").as_deref(), Some("http://x"));
        assert_eq!(controller.mark_first_token("lease-1"), true);
        assert_eq!(controller.mark_first_token("missing"), false);
        assert!(controller.lease("lease-1").unwrap()["replayable"] == false);
        assert_eq!(controller.release("lease-1"), true);
        assert_eq!(controller.release("lease-1"), false);
        assert_eq!(controller.release("lease-2"), true);
        assert_eq!(controller.snapshot()["active_leases"], 0);
        assert_eq!(controller.snapshot()["healthy_replicas"], 1);
    }

    #[test]
    fn weight_update_frame_and_aggregation_match_python() {
        let request = WeightUpdateRequest {
            path: "/tmp/w".into(),
            expected_digest: Some("abc".into()),
            source: "disk".into(),
            abort_inflight: true,
        };
        let frame = serde_json::to_value(request.inspect_payload(7)).unwrap();
        assert_eq!(frame["op"], "update_weights");
        assert_eq!(frame["command_id"], 7);
        assert_eq!(frame["path"], "/tmp/w");
        assert_eq!(frame["expected_digest"], "abc");
        assert_eq!(frame["source"], "disk");

        let ok = aggregate_rank_results(&[
            ("r0".into(), json!({"ok": true, "digest": "abc"})),
            ("r1".into(), json!({"ok": true, "digest": "abc"})),
        ]);
        assert!(ok.ok);
        assert_eq!(ok.after_digest.as_deref(), Some("abc"));
        let bad = aggregate_rank_results(&[("r0".into(), json!({"ok": false, "error": "boom"}))]);
        assert_eq!(bad.error.as_deref(), Some("rank r0 failed: boom"));
    }

    #[test]
    fn weights_checker_projection_matches_python() {
        let descriptor = json!({
            "startup": {
                "draft_model_digest": "d1",
                "kernel_binding_digest": "k1",
                "compiled": {"digest": "c1"},
                "tensor_parallel": 2
            },
            "hardware": {"accelerator": "cuda", "device_name": "RTX 3090"}
        });
        assert_eq!(AdminController::weights_checker(&descriptor), json!({
            "model_digest": "d1",
            "kernel_binding_digest": "k1",
            "compiled_digest": "c1",
            "tensor_parallel": 2,
            "accelerator": "cuda",
            "device_name": "RTX 3090",
        }));
        // Missing sections project to null, never to guessed defaults.
        let checker = AdminController::weights_checker(&json!({}));
        assert_eq!(checker["model_digest"], Value::Null);
        assert_eq!(checker["tensor_parallel"], Value::Null);
    }

    #[test]
    fn controller_prefix_preference_and_bool_contract_match_python() {
        let controller = Controller::new();
        controller
            .register(ReplicaRegistration {
                replica_id: "a".into(),
                url: "http://a".into(),
                node_id: "n1".into(),
                capacity: 2,
                prefixes: vec![],
            })
            .unwrap();
        controller
            .register(ReplicaRegistration {
                replica_id: "b".into(),
                url: "http://b".into(),
                node_id: "n2".into(),
                capacity: 2,
                prefixes: vec!["p".into()],
            })
            .unwrap();
        // Prefix match wins even though "a" sorts first.
        let lease = controller.acquire("req", "p").unwrap();
        assert_eq!(lease["replica_id"], "b");
        assert_eq!(lease["replayable"], true);
        assert_eq!(lease["expires_ns"].as_u64().unwrap() > 1_000_000_000, true);

        assert!(controller.heartbeat("b", None, None));
        assert!(!controller.heartbeat("missing", None, None));
        assert!(controller.drain("b"));
        assert!(!controller.drain("missing"));
        assert!(controller.activate("b"));
        assert!(!controller.activate("missing"));
        // Active lease blocks deregistration, exactly like Python.
        assert!(!controller.deregister("b"));
        assert!(controller.release(lease["lease_id"].as_str().unwrap()));
        assert!(controller.deregister("b"));
        assert!(!controller.deregister("b"));
    }

    #[test]
    fn lease_fingerprint_is_stable() {
        assert_eq!(fingerprint_token_ids(&[1, 2, 3, 4], 16), "5155f51de58afefe");
        assert_eq!(fingerprint_token_ids(&[], 16), "");
    }

    #[test]
    fn lease_tracker_lifecycle_and_snapshot_match_python() {
        let controller = Arc::new(Controller::new());
        controller
            .register(ReplicaRegistration {
                replica_id: "r1".into(),
                url: "http://x".into(),
                node_id: "n1".into(),
                capacity: 2,
                prefixes: vec!["p".into()],
            })
            .unwrap();
        let tracker = LeaseTracker::new(Some(controller.clone()));
        assert_eq!(tracker.snapshot()["enabled"], true);
        assert_eq!(tracker.snapshot()["active"], 0);
        assert_eq!(tracker.snapshot()["acquire_failures"], 0);

        tracker.begin("req1", "p");
        assert_eq!(tracker.snapshot()["active"], 1);
        assert_eq!(controller.snapshot()["active_leases"], 1);
        tracker.first_token("req1");
        assert!(controller.lease("lease-1").unwrap()["first_token"] == true);
        assert_eq!(tracker.finish("req1", false), false);
        assert_eq!(tracker.snapshot()["active"], 0);
        assert_eq!(controller.snapshot()["active_leases"], 0);

        let disabled = LeaseTracker::new(None);
        assert_eq!(disabled.snapshot(), json!({"enabled": false}));
        disabled.begin("req2", "");
        assert_eq!(disabled.finish("req2", true), false);
    }
}
