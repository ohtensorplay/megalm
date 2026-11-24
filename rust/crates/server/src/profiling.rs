//! Opt-in request-level milestone recorder (JSONL).
//!
//! This is the Rust counterpart of ``src/megalm/profiling.py``. Both write the
//! same JSONL schema so a single directory (or several per-process files) can
//! be merged by request::
//!
//! ```text
//! {"ts_ns": ..., "pid": ..., "run_id": ..., "request_id": ...,
//!  "event": ..., "stage": ..., "extra": {...}}
//! ```
//!
//! Disabled by default: when `MEGALM_PROFILE_FILE` is unset, `record()` is a
//! single `Option` check with no allocation and no file I/O. Write failures are
//! swallowed and counted so profiling can never change serving semantics.

use std::cell::RefCell;
use std::fs::{File, OpenOptions};
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

use serde_json::{json, Map, Value};

/// Canonical milestones from `GAP-PLAN.md` §11. Custom event names are still
/// accepted by [`Profiler::record`], so a caller can add one without editing
/// this module.
pub mod milestone {
    pub const ADMISSION: &str = "admission";
    pub const PREFILL_START: &str = "prefill_start";
    pub const FIRST_TOKEN: &str = "first_token";
    pub const HOP_SENT: &str = "hop_sent";
    pub const HOP_RECEIVED: &str = "hop_received";
    pub const TERMINAL_RESPONSE: &str = "terminal_response";

    pub const ALL: [&str; 6] = [
        ADMISSION,
        PREFILL_START,
        FIRST_TOKEN,
        HOP_SENT,
        HOP_RECEIVED,
        TERMINAL_RESPONSE,
    ];
}

/// Active-stage binding for the current thread. This is the Rust counterpart
/// of Python's `threading.local` half of the dual `ContextVar` + thread-local
/// binding. Async tasks that migrate threads should pass an explicit stage to
/// [`Profiler::record_stage`] instead (the caller knows it best).
mod active_stage {
    use super::*;

    thread_local! {
        static STAGE: RefCell<String> = RefCell::new(String::new());
    }

    pub fn current() -> String {
        STAGE.with(|cell| cell.borrow().clone())
    }

    /// Bind `stage` for the current thread until the returned guard is dropped.
    pub fn enter(stage: &str) -> StageGuard {
        let previous = STAGE.with(|cell| std::mem::replace(&mut *cell.borrow_mut(), stage.to_owned()));
        StageGuard { previous }
    }

    pub struct StageGuard {
        previous: String,
    }

    impl Drop for StageGuard {
        fn drop(&mut self) {
            STAGE.with(|cell| {
                *cell.borrow_mut() = std::mem::take(&mut self.previous);
            });
        }
    }
}

pub use active_stage::StageGuard;
pub fn current_stage() -> String {
    active_stage::current()
}

/// Bind the active profiling stage on the current thread (RAII).
pub fn with_stage(stage: &str) -> StageGuard {
    active_stage::enter(stage)
}

/// Keep only small JSON scalars in metadata. Tensors, arrays and nested
/// objects are dropped without being inspected (no clone of large payloads).
/// Long strings are truncated so a profile line stays bounded.
fn clean_extra(extra: &Value) -> (Value, usize) {
    let Value::Object(map) = extra else {
        return (json!({}), 0);
    };
    let mut clean = Map::new();
    let mut dropped = 0usize;
    for (key, value) in map {
        match value {
            Value::Null | Value::Bool(_) | Value::Number(_) => {
                clean.insert(key.clone(), value.clone());
            }
            Value::String(text) => {
                let truncated: String = text.chars().take(MAX_STRING_CHARS).collect();
                clean.insert(key.clone(), Value::String(truncated));
            }
            _ => dropped += 1,
        }
    }
    (Value::Object(clean), dropped)
}

const MAX_STRING_CHARS: usize = 512;

/// Default stage attribution when the caller does not supply one. Finer than
/// the old "everything except first_token is admission" rule: each canonical
/// milestone maps to the stage that actually owns it.
fn stage_for_event(event: &str) -> &'static str {
    match event {
        milestone::PREFILL_START => "prefill",
        milestone::FIRST_TOKEN => "decode",
        milestone::HOP_SENT | milestone::HOP_RECEIVED => "router",
        milestone::TERMINAL_RESPONSE => "gateway",
        _ => "gateway", // admission / "admitted" alias / custom events
    }
}

#[derive(Clone)]
pub struct Profiler {
    writer: Option<Arc<Mutex<BufWriter<File>>>>,
    run_id: String,
    path: Option<PathBuf>,
    /// Default stage used when neither the call site nor the thread binding
    /// supplies one (from `MEGALM_PROFILE_STAGE`).
    stage: String,
    events_recorded: Arc<AtomicU64>,
    dropped_writes: Arc<AtomicU64>,
    dropped_metadata: Arc<AtomicU64>,
    warned: Arc<AtomicBool>,
}

impl std::fmt::Debug for Profiler {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("Profiler")
            .field("enabled", &self.writer.is_some())
            .field("run_id", &self.run_id)
            .field("path", &self.path)
            .field("stage", &self.stage)
            .field("events_recorded", &self.events_recorded.load(Ordering::Relaxed))
            .field("dropped_writes", &self.dropped_writes.load(Ordering::Relaxed))
            .field("dropped_metadata", &self.dropped_metadata.load(Ordering::Relaxed))
            .finish()
    }
}

impl Profiler {
    pub fn from_env() -> Self {
        let run_id = std::env::var("MEGALM_PROFILE_RUN_ID")
            .unwrap_or_else(|_| uuid::Uuid::new_v4().simple().to_string());
        let stage = std::env::var("MEGALM_PROFILE_STAGE").unwrap_or_default();
        let Ok(path) = std::env::var("MEGALM_PROFILE_FILE") else {
            return Self::disabled_with(run_id, stage);
        };
        let path = PathBuf::from(path);
        if let Some(parent) = path.parent() {
            if !parent.as_os_str().is_empty() && std::fs::create_dir_all(parent).is_err() {
                return Self::disabled_with(run_id, stage);
            }
        }
        let file = match OpenOptions::new().create(true).append(true).open(&path) {
            Ok(file) => file,
            Err(err) => {
                eprintln!("cannot open profile file {}: {err}", path.display());
                return Self::disabled_with(run_id, stage);
            }
        };
        Self {
            writer: Some(Arc::new(Mutex::new(BufWriter::new(file)))),
            run_id,
            path: Some(path),
            stage,
            events_recorded: Arc::new(AtomicU64::new(0)),
            dropped_writes: Arc::new(AtomicU64::new(0)),
            dropped_metadata: Arc::new(AtomicU64::new(0)),
            warned: Arc::new(AtomicBool::new(false)),
        }
    }

    /// Append one milestone. Failures are swallowed and counted, so profiling
    /// can never change serving semantics. The critical section holds no
    /// `.await`, so a plain mutex keeps the recorder callable from both HTTP
    /// tasks and the blocking P/D streaming loop.
    pub fn record(&self, event: &str, request_id: &str, extra: Value) {
        self.record_stage(event, request_id, "", extra);
    }

    /// Append one milestone with an explicit stage. An empty `stage` falls
    /// back to the thread binding, then the configured default, then the
    /// per-event default (see [`stage_for_event`]).
    pub fn record_stage(&self, event: &str, request_id: &str, stage: &str, extra: Value) {
        let Some(writer) = &self.writer else {
            return;
        };
        let (extra, dropped) = clean_extra(&extra);
        let stage = if !stage.is_empty() {
            stage.to_owned()
        } else {
            let bound = current_stage();
            if !bound.is_empty() {
                bound
            } else if !self.stage.is_empty() {
                self.stage.clone()
            } else {
                stage_for_event(event).to_owned()
            }
        };
        let timestamp = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos();
        let line = json!({
            "ts_ns": timestamp,
            "pid": std::process::id(),
            "run_id": self.run_id,
            "request_id": request_id,
            "event": event,
            "stage": stage,
            "extra": extra,
        });
        let text = match serde_json::to_string(&line) {
            Ok(text) => text,
            Err(_) => {
                self.dropped_writes.fetch_add(1, Ordering::Relaxed);
                self.warn_once("cannot serialize profile event");
                return;
            }
        };
        let mut writer = writer.lock().unwrap_or_else(|poisoned| poisoned.into_inner());
        if writeln!(writer, "{text}").is_ok() && writer.flush().is_ok() {
            self.events_recorded.fetch_add(1, Ordering::Relaxed);
            self.dropped_metadata.fetch_add(dropped as u64, Ordering::Relaxed);
        } else {
            self.dropped_writes.fetch_add(1, Ordering::Relaxed);
            self.warn_once("cannot write profile event");
        }
    }

    /// Cheap check for call sites that must not allocate a `json!` value when
    /// profiling is disabled.
    pub fn enabled(&self) -> bool {
        self.writer.is_some()
    }

    /// Record `admission` without allocating when disabled.
    pub fn record_admission(&self, request_id: &str, prompt_tokens: usize) {
        if self.enabled() {
            self.record(milestone::ADMISSION, request_id, json!({"prompt_tokens": prompt_tokens}));
        }
    }

    /// Record `prefill_start` without allocating when disabled.
    pub fn record_prefill_start(&self, request_id: &str, prompt_tokens: usize) {
        if self.enabled() {
            self.record(milestone::PREFILL_START, request_id, json!({"prompt_tokens": prompt_tokens}));
        }
    }

    /// Record `first_token` without allocating when disabled.
    pub fn record_first_token(&self, request_id: &str, choice: usize) {
        if self.enabled() {
            self.record(milestone::FIRST_TOKEN, request_id, json!({"choice": choice}));
        }
    }

    /// Record `terminal_response` without allocating when disabled.
    pub fn record_terminal(
        &self,
        request_id: &str,
        choice: usize,
        completion_tokens: usize,
        finish_reason: &str,
    ) {
        if self.enabled() {
            self.record(
                milestone::TERMINAL_RESPONSE,
                request_id,
                json!({
                    "choice": choice,
                    "completion_tokens": completion_tokens,
                    "finish_reason": finish_reason,
                }),
            );
        }
    }

    /// Record a `hop_sent` / `hop_received` pair without allocating when
    /// disabled. `event` must be one of the two hop milestone names.
    pub fn record_hop(
        &self,
        event: &str,
        request_id: &str,
        src: &str,
        dst: &str,
        chunk_id: &str,
    ) {
        if self.enabled() {
            self.record(
                event,
                request_id,
                json!({"src": src, "dst": dst, "chunk_id": chunk_id}),
            );
        }
    }

    fn warn_once(&self, message: &str) {
        if !self.warned.swap(true, Ordering::Relaxed) {
            eprintln!("profiling: {message}");
        }
    }

    /// Cold-path metrics snapshot (for `/metrics` / `/server_info`).
    pub fn snapshot(&self) -> Value {
        json!({
            "enabled": self.writer.is_some(),
            "events_recorded": self.events_recorded.load(Ordering::Relaxed),
            "dropped_writes": self.dropped_writes.load(Ordering::Relaxed),
            "dropped_metadata": self.dropped_metadata.load(Ordering::Relaxed),
            "run_id": self.run_id,
        })
    }

    /// Build the same view shape as Python's `profiling.view`: run ids,
    /// per-request timelines and stage breakdowns, hop pairs and event count.
    /// Reading the view never starts the recorder.
    pub fn view(&self, request_id: Option<&str>) -> Value {
        let Some(path) = &self.path else {
            return json!({"enabled": false});
        };
        view_files(std::slice::from_ref(path), request_id)
    }

    fn disabled_with(run_id: String, stage: String) -> Self {
        Self {
            writer: None,
            run_id,
            path: None,
            stage,
            events_recorded: Arc::new(AtomicU64::new(0)),
            dropped_writes: Arc::new(AtomicU64::new(0)),
            dropped_metadata: Arc::new(AtomicU64::new(0)),
            warned: Arc::new(AtomicBool::new(false)),
        }
    }

    /// A no-op recorder for tests that need a `Profiler` without file I/O.
    #[cfg(test)]
    pub fn disabled() -> Self {
        Self::disabled_with(String::new(), String::new())
    }

    /// An enabled recorder writing to `path`, for crate-wide tests that need
    /// to verify call sites actually emit milestones.
    #[cfg(test)]
    pub(crate) fn for_test(path: PathBuf) -> Self {
        let file = OpenOptions::new()
            .create(true)
            .append(true)
            .open(&path)
            .expect("test profile file");
        Self {
            writer: Some(Arc::new(Mutex::new(BufWriter::new(file)))),
            run_id: "test-run".into(),
            path: Some(path),
            stage: String::new(),
            events_recorded: Arc::new(AtomicU64::new(0)),
            dropped_writes: Arc::new(AtomicU64::new(0)),
            dropped_metadata: Arc::new(AtomicU64::new(0)),
            warned: Arc::new(AtomicBool::new(false)),
        }
    }
}

impl Default for Profiler {
    fn default() -> Self {
        Self::from_env()
    }
}

/// Bridge from `engine-client`'s IPC frame observer into the request profiler.
///
/// Both legs use the same `(src, dst, chunk_id)` key (the P/D router convention)
/// so `view_files` can pair them:
/// `src=gateway, dst=worker, chunk_id=submit`.
pub struct IpcFrameObserver {
    profiler: Arc<Profiler>,
}

impl IpcFrameObserver {
    pub fn new(profiler: Arc<Profiler>) -> Self {
        Self { profiler }
    }
}

impl megalm_engine_client::FrameObserver for IpcFrameObserver {
    fn frame_sent(&self, request_id: &str, kind: &str) {
        self.profiler
            .record_hop(milestone::HOP_SENT, request_id, "gateway", "worker", kind);
    }

    fn frame_received(&self, request_id: &str, kind: &str) {
        self.profiler
            .record_hop(milestone::HOP_RECEIVED, request_id, "gateway", "worker", kind);
    }
}

/// Minimal glob matcher for one path component. Supports `*`, `?` and `[...]`
/// (including `[a-z]` ranges and `[!...]` negation). Implemented locally so no
/// new Cargo dependency is pulled onto the serving hot path.
fn glob_match(pattern: &str, text: &str) -> bool {
    fn inner(pattern: &[u8], text: &[u8]) -> bool {
        if pattern.is_empty() {
            return text.is_empty();
        }
        match pattern[0] {
            b'*' => {
                let mut rest = 1;
                while rest < pattern.len() && pattern[rest] == b'*' {
                    rest += 1;
                }
                (0..=text.len()).any(|skip| inner(&pattern[rest..], &text[skip..]))
            }
            b'?' => !text.is_empty() && inner(&pattern[1..], &text[1..]),
            b'[' => {
                let Some(end) = pattern.iter().position(|&byte| byte == b']') else {
                    return !text.is_empty()
                        && text[0] == b'['
                        && inner(&pattern[1..], &text[1..]);
                };
                if text.is_empty() {
                    return false;
                }
                let class = &pattern[1..end];
                let (negated, class) = match class.first() {
                    Some(b'!') | Some(b'^') => (true, &class[1..]),
                    _ => (false, class),
                };
                let mut hit = false;
                let mut index = 0;
                while index < class.len() {
                    if index + 2 < class.len() && class[index + 1] == b'-' {
                        if class[index] <= text[0] && text[0] <= class[index + 2] {
                            hit = true;
                        }
                        index += 3;
                    } else {
                        if class[index] == text[0] {
                            hit = true;
                        }
                        index += 1;
                    }
                }
                hit != negated && inner(&pattern[end + 1..], &text[1..])
            }
            literal => {
                !text.is_empty() && text[0] == literal && inner(&pattern[1..], &text[1..])
            }
        }
    }
    inner(pattern.as_bytes(), text.as_bytes())
}

/// Expand a glob in the final path component to concrete files. Returns the
/// path unchanged when it contains no metacharacters.
fn expand_glob(path: &Path) -> Vec<PathBuf> {
    let Some(pattern) = path.to_str() else {
        return vec![path.to_path_buf()];
    };
    if !pattern.contains(['*', '?', '[']) {
        return vec![path.to_path_buf()];
    }
    let parent = match path.parent() {
        Some(parent) if !parent.as_os_str().is_empty() => parent,
        _ => Path::new("."),
    };
    let Some(name_pattern) = path.file_name().and_then(|name| name.to_str()) else {
        return Vec::new();
    };
    let mut matched = Vec::new();
    if let Ok(entries) = std::fs::read_dir(parent) {
        for entry in entries.flatten() {
            if glob_match(name_pattern, &entry.file_name().to_string_lossy()) {
                matched.push(entry.path());
            }
        }
    }
    matched.sort();
    matched
}

/// Load milestone lines from files, directories of `*.jsonl`, or glob
/// patterns; normalize the `admitted` alias to `admission`, and sort by
/// `(ts_ns, pid, event)`. Duplicate paths (e.g. a file plus its directory)
/// are deduplicated by canonical path.
fn load_events(paths: &[PathBuf]) -> Vec<Value> {
    let mut files: Vec<PathBuf> = Vec::new();
    for path in paths {
        for expanded in expand_glob(path) {
            if expanded.is_dir() {
                if let Ok(entries) = std::fs::read_dir(&expanded) {
                    for entry in entries.flatten() {
                        let candidate = entry.path();
                        if candidate.extension().and_then(|ext| ext.to_str()) == Some("jsonl") {
                            files.push(candidate);
                        }
                    }
                }
            } else {
                files.push(expanded);
            }
        }
    }
    files.sort();
    let mut seen = std::collections::HashSet::new();
    files.retain(|path| {
        let key = std::fs::canonicalize(path).unwrap_or_else(|_| path.clone());
        seen.insert(key)
    });

    let mut events = Vec::new();
    for file in files {
        let Ok(text) = std::fs::read_to_string(&file) else {
            continue;
        };
        for line in text.lines() {
            let Ok(mut event) = serde_json::from_str::<Value>(line) else {
                continue;
            };
            if event.get("ts_ns").is_none() {
                continue;
            }
            if event["event"] == "admitted" {
                event["event"] = json!(milestone::ADMISSION);
            }
            events.push(event);
        }
    }
    events.sort_by(|left, right| {
        left["ts_ns"]
            .as_u64()
            .unwrap_or(0)
            .cmp(&right["ts_ns"].as_u64().unwrap_or(0))
            .then_with(|| left["pid"].as_u64().unwrap_or(0).cmp(&right["pid"].as_u64().unwrap_or(0)))
            .then_with(|| left["event"].as_str().unwrap_or("").cmp(right["event"].as_str().unwrap_or("")))
    });
    events
}

/// Build a combined profiling view from several files/directories. This is the
/// cross-process counterpart of Python's `view(*paths, request_id=...)`: the
/// GPU worker and the Rust frontend write separate JSONL files that merge by
/// `request_id` here.
pub fn view_files(paths: &[PathBuf], request_id: Option<&str>) -> Value {
    let events = load_events(paths);
    let run_ids: Vec<String> = {
        let mut ids: Vec<String> = events
            .iter()
            .filter_map(|event| event["run_id"].as_str().map(str::to_string))
            .filter(|id| !id.is_empty())
            .collect();
        ids.sort();
        ids.dedup();
        ids
    };
    let mut ids: Vec<String> = events
        .iter()
        .filter_map(|event| event["request_id"].as_str().map(str::to_string))
        .filter(|id| request_id.is_none_or(|selected| selected == id))
        .collect();
    ids.sort();
    ids.dedup();

    let mut timelines = Map::new();
    let mut breakdowns = Map::new();
    for id in &ids {
        let selected: Vec<&Value> = events
            .iter()
            .filter(|event| event["request_id"].as_str() == Some(id.as_str()))
            .collect();
        let origin = selected
            .first()
            .and_then(|event| event["ts_ns"].as_u64())
            .unwrap_or(0);
        let timeline: Vec<Value> = selected
            .iter()
            .map(|event| {
                let ts = event["ts_ns"].as_u64().unwrap_or(0);
                json!({
                    "event": event["event"],
                    "stage": event["stage"],
                    "pid": event["pid"],
                    "run_id": event["run_id"],
                    "ts_ns": ts,
                    "elapsed_ms": (ts.saturating_sub(origin) as f64) / 1_000_000.0,
                    "extra": event["extra"],
                })
            })
            .collect();
        let mut stages: Map<String, Value> = Map::new();
        for pair in timeline.windows(2) {
            let stage = pair[0]["stage"].as_str().unwrap_or("unknown");
            let duration = (pair[1]["ts_ns"].as_u64().unwrap_or(0)
                .saturating_sub(pair[0]["ts_ns"].as_u64().unwrap_or(0)) as f64)
                / 1_000_000.0;
            let entry = stages.entry(stage.to_string()).or_insert(json!(0.0));
            *entry = json!(entry.as_f64().unwrap_or(0.0) + duration.max(0.0));
        }
        timelines.insert(id.clone(), Value::Array(timeline));
        breakdowns.insert(id.clone(), Value::Object(stages));
    }

    let mut sent: std::collections::HashMap<String, u64> = std::collections::HashMap::new();
    let mut received: std::collections::HashMap<String, u64> = std::collections::HashMap::new();
    for event in &events {
        let kind = event["event"].as_str().unwrap_or("");
        if kind != milestone::HOP_SENT && kind != milestone::HOP_RECEIVED {
            continue;
        }
        if ![ "src", "dst", "chunk_id" ]
            .iter()
            .all(|key| event["extra"][*key].is_string())
        {
            continue;
        }
        let key = format!(
            "{}|{}|{}|{}",
            event["request_id"].as_str().unwrap_or(""),
            event["extra"]["src"].as_str().unwrap_or(""),
            event["extra"]["dst"].as_str().unwrap_or(""),
            event["extra"]["chunk_id"].as_str().unwrap_or("")
        );
        let ts = event["ts_ns"].as_u64().unwrap_or(0);
        let target = if kind == milestone::HOP_SENT { &mut sent } else { &mut received };
        target
            .entry(key)
            .and_modify(|value| {
                if ts < *value {
                    *value = ts;
                }
            })
            .or_insert(ts);
    }
    let mut hops = Vec::new();
    for (key, sent_ns) in sent {
        let parts: Vec<&str> = key.split('|').collect();
        let received_ns = received.get(&key).copied();
        hops.push(json!({
            "request_id": parts.first().copied().unwrap_or(""),
            "src": parts.get(1).copied().unwrap_or(""),
            "dst": parts.get(2).copied().unwrap_or(""),
            "chunk_id": parts.get(3).copied().unwrap_or(""),
            "sent_ns": sent_ns,
            "received_ns": received_ns,
            "latency_ms": received_ns.map(|value| (value.saturating_sub(sent_ns) as f64) / 1_000_000.0),
            "matched": received_ns.is_some(),
        }));
    }
    hops.sort_by(|left, right| {
        left["request_id"]
            .as_str()
            .unwrap_or("")
            .cmp(right["request_id"].as_str().unwrap_or(""))
            .then_with(|| left["chunk_id"].as_str().unwrap_or("").cmp(right["chunk_id"].as_str().unwrap_or("")))
    });

    json!({
        "enabled": true,
        "run_ids": run_ids,
        "timelines": Value::Object(timelines),
        "stage_breakdowns": Value::Object(breakdowns),
        "hops": hops,
        "event_count": events.len(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_profiler(path: Option<PathBuf>, writer: Option<Arc<Mutex<BufWriter<File>>>>) -> Profiler {
        Profiler {
            writer,
            run_id: "test-run".into(),
            path,
            stage: String::new(),
            events_recorded: Arc::new(AtomicU64::new(0)),
            dropped_writes: Arc::new(AtomicU64::new(0)),
            dropped_metadata: Arc::new(AtomicU64::new(0)),
            warned: Arc::new(AtomicBool::new(false)),
        }
    }

    fn temp_path(name: &str) -> PathBuf {
        std::env::temp_dir().join(format!("megalm-profile-{name}-{}", uuid::Uuid::new_v4().simple()))
    }

    #[test]
    fn records_jsonl_milestones() {
        let path = temp_path("milestones");
        let file = OpenOptions::new().create(true).append(true).open(&path).unwrap();
        let profiler = test_profiler(Some(path.clone()), Some(Arc::new(Mutex::new(BufWriter::new(file)))));
        assert!(profiler.writer.is_some());
        profiler.record(milestone::FIRST_TOKEN, "req-1", json!({"choice": 0}));
        drop(profiler);
        let text = std::fs::read_to_string(&path).unwrap();
        let line: Value = serde_json::from_str(text.trim()).unwrap();
        assert_eq!(line["event"], "first_token");
        assert_eq!(line["request_id"], "req-1");
        assert_eq!(line["stage"], "decode");
        assert_eq!(line["extra"]["choice"], 0);

        let profiler = test_profiler(Some(path.clone()), None);
        let view = profiler.view(None);
        assert_eq!(view["enabled"], true);
        assert_eq!(view["event_count"], 1);
        assert_eq!(view["run_ids"][0], "test-run");
        assert_eq!(view["timelines"]["req-1"][0]["event"], "first_token");
        assert_eq!(view["timelines"]["req-1"][0]["stage"], "decode");
        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn records_all_canonical_milestones_with_fine_stages() {
        assert_eq!(milestone::ALL.len(), 6);
        let path = temp_path("all-milestones");
        let file = OpenOptions::new().create(true).append(true).open(&path).unwrap();
        let profiler = test_profiler(Some(path.clone()), Some(Arc::new(Mutex::new(BufWriter::new(file)))));
        profiler.record(milestone::ADMISSION, "req", json!({"prompt_tokens": 12}));
        profiler.record("admitted", "req", json!({"prompt_tokens": 12}));
        profiler.record(milestone::PREFILL_START, "req", json!({}));
        profiler.record(milestone::FIRST_TOKEN, "req", json!({"choice": 0}));
        profiler.record(
            milestone::HOP_SENT,
            "req",
            json!({"src": "gateway", "dst": "decode", "chunk_id": "0"}),
        );
        profiler.record(
            milestone::HOP_RECEIVED,
            "req",
            json!({"src": "decode", "dst": "gateway", "chunk_id": "0"}),
        );
        profiler.record(milestone::TERMINAL_RESPONSE, "req", json!({"finish": "stop"}));
        assert_eq!(profiler.snapshot()["events_recorded"], 7);
        drop(profiler);

        let profiler = test_profiler(Some(path.clone()), None);
        let view = profiler.view(Some("req"));
        let timeline = view["timelines"]["req"].as_array().unwrap();
        assert_eq!(timeline.len(), 7);
        let stages: Vec<&str> = timeline
            .iter()
            .map(|event| event["stage"].as_str().unwrap_or(""))
            .collect();
        assert_eq!(stages, vec!["gateway", "gateway", "prefill", "decode", "router", "router", "gateway"]);
        // The Rust "admitted" alias is normalized on load, just like Python.
        assert_eq!(timeline[1]["event"], "admission");
        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn explicit_and_bound_stages_override_event_default() {
        let path = temp_path("stage-binding");
        let file = OpenOptions::new().create(true).append(true).open(&path).unwrap();
        let profiler = test_profiler(Some(path.clone()), Some(Arc::new(Mutex::new(BufWriter::new(file)))));
        profiler.record_stage(milestone::FIRST_TOKEN, "r1", "custom", json!({}));
        {
            let _guard = with_stage("thread-stage");
            profiler.record(milestone::ADMISSION, "r1", json!({}));
        }
        profiler.record(milestone::ADMISSION, "r1", json!({}));
        drop(profiler);

        let text = std::fs::read_to_string(&path).unwrap();
        let events: Vec<Value> = text.lines().map(|line| serde_json::from_str(line).unwrap()).collect();
        assert_eq!(events[0]["stage"], "custom");
        assert_eq!(events[1]["stage"], "thread-stage");
        // Binding is RAII: after the guard drops, the default applies again.
        assert_eq!(events[2]["stage"], "gateway");
        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn metadata_is_sanitized_and_counted() {
        let path = temp_path("sanitize");
        let file = OpenOptions::new().create(true).append(true).open(&path).unwrap();
        let profiler = test_profiler(Some(path.clone()), Some(Arc::new(Mutex::new(BufWriter::new(file)))));
        profiler.record(
            milestone::ADMISSION,
            "req",
            json!({
                "small": 7,
                "flag": true,
                "nil": null,
                "text": "x".repeat(MAX_STRING_CHARS + 50),
                "tensor": [1, 2, 3],
                "nested": {"a": 1},
            }),
        );
        assert_eq!(profiler.snapshot()["dropped_metadata"], 2);
        drop(profiler);

        let text = std::fs::read_to_string(&path).unwrap();
        let line: Value = serde_json::from_str(text.trim()).unwrap();
        assert_eq!(line["extra"]["small"], 7);
        assert_eq!(line["extra"]["flag"], true);
        assert_eq!(line["extra"]["nil"], Value::Null);
        assert_eq!(line["extra"]["text"].as_str().unwrap().chars().count(), MAX_STRING_CHARS);
        assert!(line["extra"].get("tensor").is_none());
        assert!(line["extra"].get("nested").is_none());
        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn load_events_supports_file_directory_glob_and_dedup() {
        let dir = std::env::temp_dir().join(format!("megalm-profile-glob-{}", uuid::Uuid::new_v4().simple()));
        std::fs::create_dir_all(&dir).unwrap();
        let a = dir.join("a.jsonl");
        let b = dir.join("b.jsonl");
        std::fs::write(
            &a,
            format!(
                "{}\nnot-json\n",
                json!({"ts_ns": 2, "pid": 1, "run_id": "r1", "request_id": "q", "event": "first_token", "stage": "", "extra": {}}),
            ),
        )
        .unwrap();
        std::fs::write(
            &b,
            json!({"ts_ns": 1, "pid": 2, "run_id": "r2", "request_id": "q", "event": "admission", "stage": "", "extra": {}}).to_string() + "\n",
        )
        .unwrap();

        assert_eq!(load_events(std::slice::from_ref(&a)).len(), 1);
        assert_eq!(load_events(std::slice::from_ref(&dir)).len(), 2);
        let glob = dir.join("*.jsonl");
        assert_eq!(load_events(std::slice::from_ref(&glob)).len(), 2);
        // A file plus its directory must not double-count.
        assert_eq!(load_events(&[a.clone(), dir.clone()]).len(), 2);
        // Sorted by timestamp across processes.
        let events = load_events(std::slice::from_ref(&dir));
        assert_eq!(events[0]["event"], "admission");
        assert_eq!(events[1]["event"], "first_token");
        // Character-class globs work too.
        assert_eq!(load_events(std::slice::from_ref(&dir.join("[ab].jsonl"))).len(), 2);
        let _ = std::fs::remove_dir_all(dir);
    }

    #[test]
    fn view_files_merges_cross_process_hops() {
        let dir = std::env::temp_dir().join(format!("megalm-profile-dir-{}", uuid::Uuid::new_v4().simple()));
        std::fs::create_dir_all(&dir).unwrap();
        let gateway = dir.join("gateway.jsonl");
        let worker = dir.join("worker.jsonl");
        std::fs::write(
            &gateway,
            format!(
                "{}\n{}\n",
                json!({"ts_ns": 100, "pid": 1, "run_id": "r", "request_id": "q", "event": "admission", "stage": "gateway", "extra": {}}),
                json!({"ts_ns": 200, "pid": 1, "run_id": "r", "request_id": "q", "event": "hop_sent", "stage": "router", "extra": {"src": "gw", "dst": "wk", "chunk_id": "0"}}),
            ),
        )
        .unwrap();
        std::fs::write(
            &worker,
            json!({"ts_ns": 300, "pid": 2, "run_id": "r", "request_id": "q", "event": "first_token", "stage": "decode", "extra": {}}).to_string() + "\n",
        )
        .unwrap();

        // A directory is loaded just like an explicit file list.
        let view = view_files(std::slice::from_ref(&dir), Some("q"));
        assert_eq!(view["event_count"], 3);
        assert_eq!(view["run_ids"][0], "r");
        assert_eq!(view["timelines"]["q"][2]["event"], "first_token");
        let hops = view["hops"].as_array().unwrap();
        assert_eq!(hops.len(), 1);
        assert_eq!(hops[0]["matched"], false); // no hop_received for this chunk
        assert_eq!(hops[0]["latency_ms"], Value::Null);
        let _ = std::fs::remove_dir_all(dir);
    }

    #[test]
    fn milestone_helpers_record_full_lifecycle() {
        let path = temp_path("helpers");
        let file = OpenOptions::new().create(true).append(true).open(&path).unwrap();
        let profiler = test_profiler(Some(path.clone()), Some(Arc::new(Mutex::new(BufWriter::new(file)))));
        assert!(profiler.enabled());
        profiler.record_admission("req", 23);
        profiler.record_prefill_start("req", 23);
        profiler.record_first_token("req", 0);
        profiler.record_hop(milestone::HOP_SENT, "req", "gw", "wk", "0");
        profiler.record_hop(milestone::HOP_RECEIVED, "req", "wk", "gw", "0");
        profiler.record_terminal("req", 0, 32, "stop");
        drop(profiler);
        let text = std::fs::read_to_string(&path).unwrap();
        let events: Vec<Value> = text.lines().map(|line| serde_json::from_str(line).unwrap()).collect();
        let names: Vec<&str> = events.iter().map(|e| e["event"].as_str().unwrap()).collect();
        assert_eq!(
            names,
            vec![
                "admission",
                "prefill_start",
                "first_token",
                "hop_sent",
                "hop_received",
                "terminal_response",
            ]
        );
        assert_eq!(events[0]["extra"]["prompt_tokens"], 23);
        assert_eq!(events[5]["extra"]["finish_reason"], "stop");
        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn pd_router_style_hops_pair_in_the_view() {
        // Mirrors the deleted tests/test_profiling_gateway.py test
        // `test_pd_router_hops_pair_in_the_view`: a P/D handshake writes two
        // hop pairs (chunk_id "prefill" / "decode") that must pair in the view.
        let path = temp_path("pd-hops");
        let file = OpenOptions::new().create(true).append(true).open(&path).unwrap();
        let profiler = test_profiler(Some(path.clone()), Some(Arc::new(Mutex::new(BufWriter::new(file)))));
        for chunk in ["prefill", "decode"] {
            // pd_router uses the same (src, dst) for both legs of one chunk so
            // hop_breakdown can pair them by (request_id, src, dst, chunk_id).
            profiler.record_hop(milestone::HOP_SENT, "hop-req", "gateway", "http://worker", chunk);
            profiler.record_hop(milestone::HOP_RECEIVED, "hop-req", "gateway", "http://worker", chunk);
        }
        // Force deterministic pairing order while keeping the schema intact.
        drop(profiler);
        let profiler = test_profiler(Some(path.clone()), None);
        let view = profiler.view(Some("hop-req"));
        let hops = view["hops"].as_array().unwrap();
        assert_eq!(hops.len(), 2);
        for hop in hops {
            assert_eq!(hop["matched"], true);
            assert_eq!(hop["request_id"], "hop-req");
            assert_eq!(hop["src"], "gateway");
            assert!(hop["chunk_id"] == "prefill" || hop["chunk_id"] == "decode");
        }
        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn ipc_frame_observer_writes_pairable_hops() {
        let path = temp_path("ipc-hops");
        let observer = IpcFrameObserver::new(Arc::new(test_profiler(
            Some(path.clone()),
            {
                let file = OpenOptions::new().create(true).append(true).open(&path).unwrap();
                Some(Arc::new(Mutex::new(BufWriter::new(file))))
            },
        )));
        use megalm_engine_client::FrameObserver;
        observer.frame_sent("ipc-req", "submit");
        observer.frame_received("ipc-req", "submit");

        let view = view_files(std::slice::from_ref(&path), Some("ipc-req"));
        let hops = view["hops"].as_array().unwrap();
        assert_eq!(hops.len(), 1);
        assert_eq!(hops[0]["src"], "gateway");
        assert_eq!(hops[0]["dst"], "worker");
        assert_eq!(hops[0]["chunk_id"], "submit");
        assert_eq!(hops[0]["matched"], true);
        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn disabled_profile_is_a_noop() {
        let profiler = Profiler::disabled();
        assert!(profiler.writer.is_none());
        assert!(!profiler.enabled());
        assert_eq!(profiler.snapshot()["enabled"], false);
        // Recording must not allocate a line, open a file, or panic.
        profiler.record(milestone::FIRST_TOKEN, "req", json!({"choice": 0}));
        profiler.record_admission("req", 1);
        profiler.record_prefill_start("req", 1);
        profiler.record_first_token("req", 0);
        profiler.record_terminal("req", 0, 1, "stop");
        profiler.record_hop(milestone::HOP_SENT, "req", "a", "b", "0");
        assert_eq!(profiler.snapshot()["events_recorded"], 0);
        assert_eq!(profiler.view(None)["enabled"], false);
    }
}
