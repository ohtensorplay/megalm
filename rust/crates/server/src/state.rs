//! Shared gateway state.

use std::collections::HashMap;
use std::sync::Arc;

use megalm_engine_client::WorkerSupervisor;
use megalm_tokenizer::ModelTokenizer;
use serde_json::Value;
use sha2::{Digest, Sha256};
use tokio::sync::RwLock;

pub type ApiKeyHash = [u8; 32];

pub fn hash_api_key(api_key: &str) -> ApiKeyHash {
    Sha256::digest(api_key.as_bytes()).into()
}

/// Bearer-token credentials shared by the auth middleware.
///
/// Keys are stored as SHA-256 digests so comparisons are fixed-length and can
/// run in constant time. An empty set means authentication is disabled.
#[derive(Clone)]
pub struct AuthState {
    api_key_hashes: Vec<ApiKeyHash>,
}

impl AuthState {
    pub fn new(api_keys: Vec<String>) -> Self {
        Self {
            api_key_hashes: api_keys
                .into_iter()
                .filter(|key| !key.is_empty())
                .map(|key| hash_api_key(&key))
                .collect(),
        }
    }

    pub fn is_enabled(&self) -> bool {
        !self.api_key_hashes.is_empty()
    }

    pub fn api_key_hashes(&self) -> &[ApiKeyHash] {
        &self.api_key_hashes
    }
}

pub struct AppState {
    pub client: Arc<WorkerSupervisor>,
    pub tokenizer: &'static ModelTokenizer,
    pub served_model_name: String,
    pub grammar: String,
    pub tool_parser: String,
    pub preserve_special_tokens: bool,
    pub max_model_len: usize,
    pub enable_lora: bool,
    /// OpenAI Responses store. Frontend-local, never persisted.
    pub responses: RwLock<HashMap<String, Value>>,
    pub auth: AuthState,
    pub profiler: Arc<crate::profiling::Profiler>,
    /// Process-level `megalm_proxy_*` / `megalm_replica_*` counters. Always on
    /// (a handful of atomics) so the metric series exist without opt-in flags.
    pub metrics: Arc<crate::replica_metrics::ReplicaMetrics>,
    pub tokenizer_batcher: crate::tokenizer_batcher::TokenizerBatcher,
    pub fetch_config: crate::remote_media::FetchConfig,
    /// Optional prefill/decode disaggregation router. None keeps all requests
    /// on the local worker (the default single-node path).
    pub pd: Option<Arc<crate::pd_router::PDRouter>>,
    /// Opt-in control plane plugins, all process-local.
    pub admin: Option<Arc<crate::control_plane::AdminController>>,
    pub controller: Arc<crate::control_plane::Controller>,
    pub lease_tracker: crate::control_plane::LeaseTracker,
    pub weight_update: Option<Arc<crate::control_plane::WeightUpdateCoordinator>>,
    pub replica_router: Option<Arc<crate::replica_router::ReplicaRouter>>,
    pub stage_graph: Option<Arc<crate::stage_graph::StageGraphRuntime>>,
}

impl AppState {
    pub fn new(
        client: Arc<WorkerSupervisor>,
        tokenizer: ModelTokenizer,
        served_model_name: String,
        descriptor: Value,
        enable_lora: bool,
        tokenizer_coalesce_wait: std::time::Duration,
        tokenizer_max_batch: usize,
    ) -> Self {
        let grammar = tokenizer.output_grammar().to_string();
        let (tool_parser, tool_preserve) = {
            let (name, preserve) = tokenizer.tool_call_parser();
            (name.to_string(), preserve)
        };
        let preserve_special_tokens =
            grammar == "inkling" || grammar == "gemma4" || tool_preserve;
        // Install the IPC frame observer only when profiling is enabled. The
        // supervisor re-installs it on rebuilt worker clients as well.
        let profiler = Arc::new(crate::profiling::Profiler::from_env());
        if profiler.enabled() {
            client.set_observer(Arc::new(crate::profiling::IpcFrameObserver::new(
                Arc::clone(&profiler),
            )));
        }
        let tokenizer: &'static ModelTokenizer = Box::leak(Box::new(tokenizer));
        let tokenizer_batcher = crate::tokenizer_batcher::TokenizerBatcher::new(
            tokenizer,
            tokenizer_coalesce_wait,
            tokenizer_max_batch,
        );
        let max_model_len = descriptor
            .get("max_model_len")
            .and_then(Value::as_u64)
            .unwrap_or(32_768) as usize;
        Self {
            client,
            tokenizer,
            served_model_name,
            grammar,
            tool_parser,
            preserve_special_tokens,
            max_model_len,
            enable_lora,
            responses: RwLock::new(HashMap::new()),
            auth: AuthState::new(Vec::new()),
            profiler,
            metrics: Arc::new(crate::replica_metrics::ReplicaMetrics::new()),
            tokenizer_batcher,
            fetch_config: crate::remote_media::FetchConfig::from_env(),
            pd: None,
            admin: None,
            controller: Arc::new(crate::control_plane::Controller::new()),
            lease_tracker: crate::control_plane::LeaseTracker::default(),
            weight_update: None,
            replica_router: None,
            stage_graph: None,
        }
    }

    /// Enable opt-in prefill/decode disaggregation routing.
    pub fn with_pd(mut self, pd: Option<Arc<crate::pd_router::PDRouter>>) -> Self {
        self.pd = pd;
        self
    }

    /// Enable the opt-in admin / controller / weight-update control plane.
    pub fn with_control_plane(mut self, enabled: bool) -> anyhow::Result<Self> {
        if enabled {
            let admin = Arc::new(crate::control_plane::AdminController::new());
            let controller = Arc::new(crate::control_plane::Controller::new());
            self.weight_update = Some(Arc::new(
                crate::control_plane::WeightUpdateCoordinator::new(admin.clone())
                    .map_err(anyhow::Error::msg)?,
            ));
            self.admin = Some(admin);
            self.lease_tracker =
                crate::control_plane::LeaseTracker::new(Some(controller.clone()));
            self.controller = controller;
        }
        Ok(self)
    }

    /// Enable the opt-in horizontal replica router.
    pub fn with_replica_router(
        mut self,
        router: Option<Arc<crate::replica_router::ReplicaRouter>>,
    ) -> Self {
        self.replica_router = router;
        self
    }

    /// Attach a prebuilt in-process stage graph runtime.
    pub fn with_stage_graph(
        mut self,
        graph: Option<Arc<crate::stage_graph::StageGraphRuntime>>,
    ) -> Self {
        if let Some(graph) = &graph {
            let _ = graph.start();
        }
        self.stage_graph = graph;
        self
    }

    /// Configure the API keys accepted by guarded HTTP routes.
    pub fn with_api_keys(mut self, api_keys: Vec<String>) -> Self {
        self.auth = AuthState::new(api_keys);
        self
    }
}
