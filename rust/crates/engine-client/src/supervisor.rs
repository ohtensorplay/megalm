//! Worker process supervision.
//!
//! A tensor-parallel group is all-or-nothing: if any rank exits, every other
//! rank must be torn down before a fresh group can rendezvous. This module owns
//! the group, detects rank failure with `try_wait`, rebuilds the whole group,
//! and swaps the `EngineClient` used by new requests. In-flight requests on the
//! dead transport are failed by the client's reader loop.

use std::os::fd::OwnedFd;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex, RwLock};
use std::time::{Duration, Instant};

use megalm_protocol::{
    EmbeddingResult, Inspect, SubmitBatch, SubmitEmbeddings, SubmitMultimodalBatch, SubmitResult,
};
use serde_json::Value;
use tokio::sync::{mpsc, watch};

use crate::{
    BatchingOptions, EngineClient, EngineConfig, FrameObserver, OutputEvent, Result, WorkerGroup,
};

struct SupervisorState {
    client: Arc<EngineClient>,
    // `Child` is Send but not Sync; keep the process group behind its own
    // mutex so the supervisor remains shareable across HTTP worker tasks.
    group: Mutex<Option<WorkerGroup>>,
}

/// Shared handle to the current worker transport. All engine calls go through
/// this handle so a rebuild is invisible to request handlers.
pub struct WorkerSupervisor {
    inner: RwLock<SupervisorState>,
    config: EngineConfig,
    python: String,
    options: BatchingOptions,
    next_stream_id: AtomicU64,
    generation: AtomicU64,
    last_rebuild: Mutex<Option<Instant>>,
    rebuild_lock: tokio::sync::Mutex<()>,
    /// Optional IPC frame observer, re-installed on every rebuilt client so a
    /// worker failure does not silently drop profiling milestones.
    observer: RwLock<Option<Arc<dyn FrameObserver>>>,
}

impl WorkerSupervisor {
    /// Spawn the initial worker group and wrap it in a supervisor.
    pub fn spawn(
        config: &EngineConfig,
        python: &str,
        options: BatchingOptions,
    ) -> Result<Arc<Self>> {
        let (client, group) = EngineClient::spawn_with_options(config, python, options.clone())?;
        Ok(Arc::new(Self {
            inner: RwLock::new(SupervisorState {
                client: Arc::new(client),
                group: Mutex::new(Some(group)),
            }),
            config: config.clone(),
            python: python.to_string(),
            options,
            next_stream_id: AtomicU64::new(1),
            generation: AtomicU64::new(1),
            last_rebuild: Mutex::new(None),
            rebuild_lock: tokio::sync::Mutex::new(()),
            observer: RwLock::new(None),
        }))
    }

    /// Install an IPC frame observer on the current client and remember it for
    /// future rebuilt clients. Called by the server only when profiling is on.
    pub fn set_observer(&self, observer: Arc<dyn FrameObserver>) {
        *self.observer.write().unwrap() = Some(Arc::clone(&observer));
        self.current().set_observer(observer);
    }

    /// Clone the current client. Never hold this across a long operation if a
    /// rebuild may happen — methods below snapshot it for their own duration.
    pub fn current(&self) -> Arc<EngineClient> {
        self.inner.read().unwrap().client.clone()
    }

    pub fn generation(&self) -> u64 {
        self.generation.load(Ordering::Relaxed)
    }

    /// Stream IDs are owned by the supervisor so a rebuild cannot collide with
    /// subscriptions still winding down on the previous transport.
    pub fn next_stream_id(&self) -> u64 {
        self.next_stream_id.fetch_add(1, Ordering::Relaxed)
    }

    // ---- EngineClient delegation ------------------------------------------

    pub async fn descriptor(&self) -> Value {
        self.current().descriptor().await
    }

    pub async fn submit_batch(&self, request: &SubmitBatch) -> Result<SubmitResult> {
        self.current().submit_batch(request).await
    }

    pub async fn submit_multimodal_batch(
        &self,
        request: &SubmitMultimodalBatch,
        fds: Vec<OwnedFd>,
    ) -> Result<SubmitResult> {
        self.current().submit_multimodal_batch(request, fds).await
    }

    pub async fn embed(&self, request: &SubmitEmbeddings) -> Result<EmbeddingResult> {
        self.current().embed(request).await
    }

    pub async fn cancel(&self, request_id: &str) -> Result<bool> {
        self.current().cancel(request_id).await
    }

    pub async fn inspect(&self, request: &Inspect) -> Result<Value> {
        self.current().inspect(request).await
    }

    pub async fn snapshot(&self) -> Result<Value> {
        self.current().snapshot().await
    }

    pub async fn subscribe(&self, stream_id: u64) -> mpsc::UnboundedReceiver<OutputEvent> {
        self.current().subscribe(stream_id).await
    }

    pub async fn unsubscribe(&self, stream_id: u64) {
        self.current().unsubscribe(stream_id).await
    }

    pub async fn stop(&self) -> Result<()> {
        self.current().stop().await
    }

    // ---- Supervision -------------------------------------------------------

    /// Poll rank liveness until `shutdown` fires. A failed group is killed in
    /// full before any rank of the replacement group is started.
    pub async fn run(
        self: Arc<Self>,
        poll: Duration,
        mut shutdown: watch::Receiver<bool>,
    ) {
        loop {
            tokio::select! {
                _ = tokio::time::sleep(poll) => {}
                result = shutdown.changed() => {
                    if result.is_err() {
                        tracing::debug!("worker watchdog shutdown channel closed");
                    }
                    return;
                }
            }
            let failed = self.has_failed_rank();
            if !failed {
                continue;
            }
            let _guard = self.rebuild_lock.lock().await;
            if !self.has_failed_rank() {
                continue;
            }
            // A permanently broken config must not spin the respawn loop.
            let cooldown = self
                .last_rebuild
                .lock()
                .unwrap()
                .is_some_and(|at| at.elapsed() < Duration::from_secs(10));
            if cooldown {
                continue;
            }
            self.rebuild().await;
        }
    }

    fn has_failed_rank(&self) -> bool {
        let state = self.inner.read().unwrap();
        let mut group = state.group.lock().unwrap();
        // A missing group means the last rebuild failed; retry after cooldown.
        let failed = match group.as_mut() {
            None => true,
            Some(group) => group.any_exited(),
        };
        failed
    }

    async fn rebuild(self: &Arc<Self>) {
        *self.last_rebuild.lock().unwrap() = Some(Instant::now());

        // Reap the old group first: a stale rank holding the rendezvous port or
        // GPU memory would make the new group fail in a different way.
        {
            let state = self.inner.read().unwrap();
            let mut group = state.group.lock().unwrap();
            if let Some(group) = group.as_mut() {
                group.kill();
                group.wait_all();
            }
            *group = None;
        }

        match EngineClient::spawn_with_options(&self.config, &self.python, self.options.clone()) {
            Ok((client, group)) => {
                let client = Arc::new(client);
                if let Some(observer) = self.observer.read().unwrap().clone() {
                    client.set_observer(observer);
                }
                let mut state = self.inner.write().unwrap();
                state.client = client;
                *state.group.lock().unwrap() = Some(group);
                self.generation.fetch_add(1, Ordering::Relaxed);
                tracing::warn!(
                    generation = self.generation(),
                    "worker group rebuilt after rank failure"
                );
            }
            Err(err) => {
                tracing::error!(%err, "worker group rebuild failed; will retry");
            }
        }
    }

    /// Kill and reap the whole group. Safe to call after the watchdog exits.
    pub async fn shutdown(self: &Arc<Self>) {
        let _guard = self.rebuild_lock.lock().await;
        let client = {
            let state = self.inner.read().unwrap();
            let mut group = state.group.lock().unwrap();
            if let Some(group) = group.as_mut() {
                group.kill();
                group.wait_all();
            }
            *group = None;
            state.client.clone()
        };
        let _ = client.stop().await;
    }
}
