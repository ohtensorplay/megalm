//! Native client for the MGLM v6 worker protocol.
//!
//! Owns the transport, correlates command replies and fans streamed output
//! frames out to per-request subscribers. No Python interpreter is involved in
//! the request path.

pub mod config;
pub mod error;
pub mod media;
pub mod stream;
pub mod supervisor;

use std::collections::{HashMap, HashSet};
use std::io;
use std::time::Duration;
use std::os::fd::{AsRawFd, FromRawFd, OwnedFd, RawFd};
use std::process::{Child, Command, Stdio};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::sync::RwLock as StdRwLock;

use megalm_protocol::{
    decode_header, encode_json, FrameType, Header, Inspect as InspectRequest, Ready, SubmitBatch,
    SubmitMultimodalBatch, SubmitResult,
};
use serde_json::Value;
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::UnixStream;
use tokio::sync::{mpsc, oneshot, Mutex, RwLock};
use tokio::time::Instant;

pub use config::EngineConfig;
pub use error::{EngineClientError, Result};
pub use stream::OutputEvent;
pub use supervisor::WorkerSupervisor;

/// The GPU worker process group: rank0 plus optional TP followers.
pub struct WorkerGroup {
    pub leader: Child,
    pub followers: Vec<Child>,
}

impl WorkerGroup {
    pub fn kill(&mut self) {
        let _ = self.leader.kill();
        for follower in &mut self.followers {
            let _ = follower.kill();
        }
    }

    /// True when any rank has already exited. A TP group is all-or-nothing, so
    /// the supervisor rebuilds every rank when this returns true.
    pub fn any_exited(&mut self) -> bool {
        if matches!(self.leader.try_wait(), Ok(Some(_))) {
            return true;
        }
        self.followers
            .iter_mut()
            .any(|follower| matches!(follower.try_wait(), Ok(Some(_))))
    }

    /// Reap every rank after a kill so PIDs and rendezvous resources are freed
    /// before the replacement group starts.
    pub fn wait_all(&mut self) {
        let _ = self.leader.wait();
        for follower in &mut self.followers {
            let _ = follower.wait();
        }
    }
}

/// How concurrent `submit_batch` calls are coalesced onto the wire.
#[derive(Debug, Clone)]
pub struct BatchingOptions {
    /// Maximum window to wait for more callers before sending one frame.
    pub coalesce_wait: Duration,
    /// Maximum submissions packed into one wire frame.
    pub max_batch_size: usize,
}

impl Default for BatchingOptions {
    fn default() -> Self {
        Self {
            coalesce_wait: Duration::from_micros(200),
            max_batch_size: 64,
        }
    }
}

struct BatchItem {
    request: SubmitBatch,
    reply: oneshot::Sender<Result<SubmitResult>>,
}

/// A frame queued for the writer. Multimodal frames carry sealed descriptors
/// that must be transferred with SCM_RIGHTS alongside the first frame byte.
struct OutgoingFrame {
    bytes: Vec<u8>,
    fds: Vec<OwnedFd>,
}

impl OutgoingFrame {
    fn plain(bytes: Vec<u8>) -> Self {
        Self {
            bytes,
            fds: Vec::new(),
        }
    }
}

/// Optional observer for Gateway ↔ Worker IPC frame milestones. The server
/// installs one only when request profiling is enabled; with no observer the
/// I/O loops pay a single `Option` check, so the default path stays zero-cost.
pub trait FrameObserver: Send + Sync {
    /// A submit frame carrying `request_id` was queued to the worker.
    fn frame_sent(&self, request_id: &str, kind: &str);
    /// The matching submit-result frame was read back from the worker.
    fn frame_received(&self, request_id: &str, kind: &str);
}

/// Shared slot so observers can be installed after an `EngineClient` is built.
pub type SharedFrameObserver = Arc<StdRwLock<Option<Arc<dyn FrameObserver>>>>;

/// One connected worker, shared by all gateway requests.
pub struct EngineClient {
    commands: mpsc::UnboundedSender<OutgoingFrame>,
    pending: Arc<Mutex<HashMap<u64, oneshot::Sender<Value>>>>,
    outputs: Arc<Mutex<HashMap<u64, mpsc::UnboundedSender<OutputEvent>>>>,
    next_command_id: Arc<AtomicU64>,
    next_stream_id: AtomicU64,
    descriptor: Arc<RwLock<Option<Value>>>,
    batcher: mpsc::UnboundedSender<BatchItem>,
    observer: SharedFrameObserver,
}

impl EngineClient {
    /// Attach to an already-connected transport with default batching.
    pub fn connect(stream: UnixStream) -> Self {
        Self::connect_with_options(stream, BatchingOptions::default())
    }

    /// Attach to a transport with explicit admission-batching settings.
    pub fn connect_with_options(stream: UnixStream, options: BatchingOptions) -> Self {
        let (command_tx, command_rx) = mpsc::unbounded_channel::<OutgoingFrame>();
        let pending = Arc::new(Mutex::new(HashMap::new()));
        let outputs = Arc::new(Mutex::new(HashMap::new()));
        let descriptor = Arc::new(RwLock::new(None));
        let next_command_id = Arc::new(AtomicU64::new(1));
        let observer: SharedFrameObserver = Arc::new(StdRwLock::new(None));

        // Both split halves share the same underlying socket descriptor; keep
        // its number for the SCM_RIGHTS writer while the halves own the fd.
        let raw_fd = stream.as_raw_fd();
        let (reader, writer) = tokio::io::split(stream);
        tokio::spawn(writer_loop(raw_fd, writer, command_rx));
        tokio::spawn(reader_loop(
            reader,
            Arc::clone(&pending),
            Arc::clone(&outputs),
            Arc::clone(&descriptor),
        ));

        let (batcher_tx, batcher_rx) = mpsc::unbounded_channel::<BatchItem>();
        tokio::spawn(batcher_loop(
            batcher_rx,
            command_tx.clone(),
            Arc::clone(&pending),
            Arc::clone(&next_command_id),
            options,
            Arc::clone(&observer),
        ));

        Self {
            commands: command_tx,
            pending,
            outputs,
            next_command_id,
            next_stream_id: AtomicU64::new(1),
            descriptor,
            batcher: batcher_tx,
            observer,
        }
    }

    /// Install an IPC frame observer (e.g. the request profiler). A no-op when
    /// called twice: the latest observer simply replaces the previous one.
    pub fn set_observer(&self, observer: Arc<dyn FrameObserver>) {
        if let Ok(mut slot) = self.observer.write() {
            *slot = Some(observer);
        }
    }

    /// Spawn `python -m megalm.gateway.worker` on a socketpair and attach.
    ///
    /// For `tensor_parallel > 1` this also starts the `tp_worker` followers and
    /// allocates a rendezvous port, mirroring `ProcessReplica.start()`.
    pub fn spawn(config: &EngineConfig, python: &str) -> Result<(Self, WorkerGroup)> {
        Self::spawn_with_options(config, python, BatchingOptions::default())
    }

    /// Spawn the worker with explicit frontend admission-batching settings.
    pub fn spawn_with_options(
        config: &EngineConfig,
        python: &str,
        options: BatchingOptions,
    ) -> Result<(Self, WorkerGroup)> {
        let mut fds = [0i32; 2];
        // SAFETY: socketpair writes exactly two file descriptors into `fds`.
        if unsafe { libc::socketpair(libc::AF_UNIX, libc::SOCK_STREAM, 0, fds.as_mut_ptr()) } != 0 {
            return Err(EngineClientError::Spawn("socketpair failed".into()));
        }
        let parent_fd = fds[0];
        let child_fd = fds[1];

        let ranks = config.tensor_parallel.max(1);
        let rendezvous_port = if ranks > 1 {
            let probe = std::net::TcpListener::bind("127.0.0.1:0")
                .map_err(|err| EngineClientError::Spawn(err.to_string()))?;
            Some(probe.local_addr().map_err(|err| EngineClientError::Spawn(err.to_string()))?.port())
        } else {
            None
        };
        let mut config = config.clone();
        if let Some(port) = rendezvous_port {
            config.tp_init_method = format!("tcp://127.0.0.1:{port}");
        }
        let encoded = config.encode()?;

        let mut leader_command = Command::new(python);
        leader_command
            .arg("-m")
            .arg("megalm.gateway.worker")
            .arg("--fd")
            .arg(child_fd.to_string())
            .arg("--config")
            .arg(&encoded)
            .stdin(Stdio::null());
        let port_string = rendezvous_port.map(|port| port.to_string());
        let ranks_string = ranks.to_string();
        if let (Some(_), Some(port_string)) = (rendezvous_port, port_string.as_deref()) {
            leader_command.envs([
                ("MASTER_ADDR", "127.0.0.1"),
                ("MASTER_PORT", port_string),
                ("RANK", "0"),
                ("WORLD_SIZE", ranks_string.as_str()),
                ("CUDA_VISIBLE_DEVICES", "0"),
            ]);
        }
        let mut leader = leader_command
            .spawn()
            .map_err(|err| EngineClientError::Spawn(err.to_string()))?;

        let mut followers = Vec::with_capacity(ranks - 1);
        for rank in 1..ranks {
            let mut command = Command::new(python);
            command
                .arg("-m")
                .arg("megalm.gateway.tp_worker")
                .arg("--config")
                .arg(&encoded)
                .stdin(Stdio::null());
            let rank_string = rank.to_string();
            if let (Some(_), Some(port_string)) = (rendezvous_port, port_string.as_deref()) {
                command.envs([
                    ("MASTER_ADDR", "127.0.0.1"),
                    ("MASTER_PORT", port_string),
                    ("RANK", rank_string.as_str()),
                    ("WORLD_SIZE", ranks_string.as_str()),
                    ("CUDA_VISIBLE_DEVICES", rank_string.as_str()),
                ]);
            }
            match command.spawn() {
                Ok(child) => followers.push(child),
                Err(err) => {
                    let _ = leader.kill();
                    for child in &mut followers {
                        let _ = child.kill();
                    }
                    return Err(EngineClientError::Spawn(err.to_string()));
                }
            }
        }

        // SAFETY: parent_fd came from a successful socketpair call.
        unsafe { libc::close(child_fd) };
        let std_stream = unsafe { std::os::unix::net::UnixStream::from_raw_fd(parent_fd) };
        std_stream
            .set_nonblocking(true)
            .map_err(|err| EngineClientError::Spawn(err.to_string()))?;
        let stream = UnixStream::from_std(std_stream)
            .map_err(|err| EngineClientError::Spawn(err.to_string()))?;
        Ok((
            Self::connect_with_options(stream, options),
            WorkerGroup { leader, followers },
        ))
    }

    pub fn next_stream_id(&self) -> u64 {
        self.next_stream_id.fetch_add(1, Ordering::Relaxed)
    }

    /// The descriptor published by the worker's READY frame.
    pub async fn descriptor(&self) -> Value {
        self.descriptor.read().await.clone().unwrap_or(Value::Null)
    }

    async fn request(&self, kind: FrameType, body: Value) -> Result<Value> {
        let command_id = self.next_command_id.fetch_add(1, Ordering::Relaxed);
        let mut body = body;
        if let Value::Object(map) = &mut body {
            map.insert("command_id".into(), Value::from(command_id));
        }
        let (tx, rx) = oneshot::channel();
        self.pending.lock().await.insert(command_id, tx);
        let frame = encode_json(kind, &body)?;
        self.commands
            .send(OutgoingFrame::plain(frame))
            .map_err(|_| EngineClientError::TransportClosed)?;
        rx.await.map_err(|_| EngineClientError::TransportClosed)
    }

    /// Submit a batch of token-id requests. Concurrent callers are coalesced
    /// onto one wire frame by the background batcher.
    pub async fn submit_batch(&self, request: &SubmitBatch) -> Result<SubmitResult> {
        let (tx, rx) = oneshot::channel();
        self.batcher
            .send(BatchItem {
                request: request.clone(),
                reply: tx,
            })
            .map_err(|_| EngineClientError::TransportClosed)?;
        rx.await.map_err(|_| EngineClientError::TransportClosed)?
    }

    /// Submit a multimodal batch. `fds` are sealed media files in the same
    /// order as `request.media` and are transferred via SCM_RIGHTS.
    ///
    /// Multimodal frames are sent directly rather than through the coalescing
    /// batcher because their descriptor rights cannot be merged across
    /// callers.
    pub async fn submit_multimodal_batch(
        &self,
        request: &SubmitMultimodalBatch,
        fds: Vec<OwnedFd>,
    ) -> Result<SubmitResult> {
        let command_id = self.next_command_id.fetch_add(1, Ordering::Relaxed);
        let mut body = request.clone();
        body.command_id = command_id;
        let frame = encode_json(FrameType::SubmitMultimodalBatch, &body)?;

        let (tx, rx) = oneshot::channel();
        self.pending.lock().await.insert(command_id, tx);
        if self
            .commands
            .send(OutgoingFrame { bytes: frame, fds })
            .is_err()
        {
            self.pending.lock().await.remove(&command_id);
            return Err(EngineClientError::TransportClosed);
        }
        let value = rx.await.map_err(|_| EngineClientError::TransportClosed)?;
        serde_json::from_value(value).map_err(EngineClientError::from)
    }

    pub async fn cancel(&self, request_id: &str) -> Result<bool> {
        let body = serde_json::json!({ "request_id": request_id });
        let value = self.request(FrameType::Cancel, body).await?;
        Ok(value
            .get("snapshot")
            .and_then(|snapshot| snapshot.get("cancelled"))
            .or_else(|| value.get("cancelled"))
            .and_then(Value::as_bool)
            .unwrap_or(false))
    }

    /// Snapshot, LoRA load/unload and other inspect operations.
    pub async fn inspect(&self, request: &InspectRequest) -> Result<Value> {
        let body = serde_json::to_value(request)?;
        self.request(FrameType::Inspect, body).await
    }

    pub async fn snapshot(&self) -> Result<Value> {
        let body = self.request(FrameType::Inspect, serde_json::json!({})).await?;
        match body.get("snapshot") {
            Some(snapshot) => Ok(snapshot.clone()),
            None => Ok(body),
        }
    }

    /// Register a channel that receives output events for `stream_id`.
    pub async fn subscribe(&self, stream_id: u64) -> mpsc::UnboundedReceiver<OutputEvent> {
        let (tx, rx) = mpsc::unbounded_channel();
        self.outputs.lock().await.insert(stream_id, tx);
        rx
    }

    pub async fn unsubscribe(&self, stream_id: u64) {
        self.outputs.lock().await.remove(&stream_id);
    }

    pub async fn stop(&self) -> Result<()> {
        let frame = encode_json(FrameType::Stop, &serde_json::json!({}))?;
        let _ = self.commands.send(OutgoingFrame::plain(frame));
        Ok(())
    }
}

/// Merge caller batches into a single wire batch. Pure so the fan-out logic
/// can be tested without a worker.
fn merge_batches(items: &[BatchItem], command_id: u64) -> SubmitBatch {
    let mut submissions = Vec::new();
    let mut target = 0usize;
    for item in items {
        submissions.extend(item.request.submissions.iter().cloned());
        target = target.max(item.request.target_active_requests);
    }
    SubmitBatch {
        command_id,
        target_active_requests: target.max(submissions.len()),
        submissions,
    }
}

async fn batcher_loop(
    mut incoming: mpsc::UnboundedReceiver<BatchItem>,
    commands: mpsc::UnboundedSender<OutgoingFrame>,
    pending: Arc<Mutex<HashMap<u64, oneshot::Sender<Value>>>>,
    next_command_id: Arc<AtomicU64>,
    options: BatchingOptions,
    observer: SharedFrameObserver,
) {
    while let Some(first) = incoming.recv().await {
        let mut items = vec![first];
        let deadline = Instant::now() + options.coalesce_wait;
        while items.len() < options.max_batch_size {
            let remaining = deadline.saturating_duration_since(Instant::now());
            match tokio::time::timeout(remaining, incoming.recv()).await {
                Ok(Some(item)) => items.push(item),
                Ok(None) | Err(_) => break,
            }
        }

        let command_id = next_command_id.fetch_add(1, Ordering::Relaxed);
        let batch = merge_batches(&items, command_id);
        let frame = match encode_json(FrameType::SubmitBatch, &batch) {
            Ok(frame) => frame,
            Err(err) => {
                for item in items {
                    let _ = item
                        .reply
                        .send(Err(EngineClientError::Message(err.to_string())));
                }
                continue;
            }
        };
        let (tx, rx) = oneshot::channel();
        pending.lock().await.insert(command_id, tx);
        if commands.send(OutgoingFrame::plain(frame)).is_err() {
            pending.lock().await.remove(&command_id);
            for item in items {
                let _ = item.reply.send(Err(EngineClientError::TransportClosed));
            }
            return;
        }
        // Frame is on the wire. Observe once per distinct request_id in the
        // coalesced batch, so a batched frame does not inflate hop counts.
        if let Ok(slot) = observer.read() {
            if let Some(observer) = slot.as_ref() {
                let mut seen = HashSet::new();
                for item in &items {
                    for submission in &item.request.submissions {
                        if seen.insert(&submission.request_id) {
                            observer.frame_sent(&submission.request_id, "submit");
                        }
                    }
                }
            }
        }

        let value = match rx.await {
            Ok(value) => value,
            Err(_) => {
                for item in items {
                    let _ = item.reply.send(Err(EngineClientError::TransportClosed));
                }
                continue;
            }
        };
        let result: SubmitResult = match serde_json::from_value(value) {
            Ok(result) => result,
            Err(err) => {
                for item in items {
                    let _ = item
                        .reply
                        .send(Err(EngineClientError::Message(err.to_string())));
                }
                continue;
            }
        };
        // The submit-result frame is back; pair it with the frame_sent event
        // using the same (request_id, src, dst, chunk_id) key in the profiler.
        if let Ok(slot) = observer.read() {
            if let Some(observer) = slot.as_ref() {
                for entry in &result.results {
                    observer.frame_received(&entry.request_id, "submit");
                }
            }
        }
        for item in items {
            let ids: HashSet<String> = item
                .request
                .submissions
                .iter()
                .map(|submission| submission.request_id.clone())
                .collect();
            let results = result
                .results
                .iter()
                .filter(|entry| ids.contains(&entry.request_id))
                .cloned()
                .collect();
            let _ = item.reply.send(Ok(SubmitResult { command_id, results }));
        }
    }
    // The client is gone: unblock any callers still queued.
    while let Some(item) = incoming.recv().await {
        let _ = item.reply.send(Err(EngineClientError::TransportClosed));
    }
}

async fn writer_loop(
    fd: RawFd,
    mut writer: tokio::io::WriteHalf<UnixStream>,
    mut commands: mpsc::UnboundedReceiver<OutgoingFrame>,
) {
    while let Some(frame) = commands.recv().await {
        if frame.fds.is_empty() {
            if writer.write_all(&frame.bytes).await.is_err() {
                break;
            }
            continue;
        }
        // Descriptor rights travel with the first byte of the frame. On EAGAIN
        // wait for writability before retrying so no fd is transferred twice.
        let mut sent = 0usize;
        let mut failed = false;
        loop {
            match media::sendmsg_with_fds(fd, &frame.bytes[sent..], &frame.fds) {
                Ok(count) if count > 0 => {
                    sent += count;
                    break;
                }
                Ok(_) => {
                    failed = true;
                    break;
                }
                Err(err) if err.kind() == io::ErrorKind::Interrupted => continue,
                Err(err) if err.kind() == io::ErrorKind::WouldBlock => {
                    tokio::task::yield_now().await;
                }
                Err(_) => {
                    failed = true;
                    break;
                }
            }
        }
        if failed {
            break;
        }
        if sent < frame.bytes.len()
            && writer.write_all(&frame.bytes[sent..]).await.is_err()
        {
            break;
        }
    }
}

async fn reader_loop(
    mut reader: tokio::io::ReadHalf<UnixStream>,
    pending: Arc<Mutex<HashMap<u64, oneshot::Sender<Value>>>>,
    outputs: Arc<Mutex<HashMap<u64, mpsc::UnboundedSender<OutputEvent>>>>,
    descriptor: Arc<RwLock<Option<Value>>>,
) {
    loop {
        let mut header = [0u8; 10];
        if reader.read_exact(&mut header).await.is_err() {
            break;
        }
        let header = match decode_header(&header) {
            Ok(header) => header,
            Err(_) => break,
        };
        let mut body = vec![0u8; header.body_len];
        if reader.read_exact(&mut body).await.is_err() {
            break;
        }
        dispatch(header, &body, &pending, &outputs, &descriptor).await;
    }
    // Unblock every waiter when the transport dies.
    pending.lock().await.clear();
    outputs.lock().await.clear();
}

async fn dispatch(
    header: Header,
    body: &[u8],
    pending: &Arc<Mutex<HashMap<u64, oneshot::Sender<Value>>>>,
    outputs: &Arc<Mutex<HashMap<u64, mpsc::UnboundedSender<OutputEvent>>>>,
    descriptor: &Arc<RwLock<Option<Value>>>,
) {
    let value: Value = match serde_json::from_slice(body) {
        Ok(value) => value,
        Err(_) => return,
    };
    match header.kind {
        FrameType::Ready => {
            if let Ok(ready) = serde_json::from_value::<Ready>(value) {
                *descriptor.write().await = Some(ready.descriptor);
            }
        }
        FrameType::SubmitResult | FrameType::Snapshot => {
            if let Some(id) = value.get("command_id").and_then(Value::as_u64) {
                if let Some(tx) = pending.lock().await.remove(&id) {
                    let _ = tx.send(value);
                }
            }
        }
        FrameType::Outputs => {
            if let Ok(payload) = serde_json::from_value::<megalm_protocol::Outputs>(value) {
                if let Ok(slots) = payload.slots() {
                    let subscribers = outputs.lock().await;
                    for slot in slots {
                        if let Some(tx) = subscribers.get(&slot.stream_id) {
                            let _ = tx.send(OutputEvent::from(slot));
                        }
                    }
                }
            }
        }
        _ => {}
    }
}

/// Re-exported so callers can close the child cleanly.
pub use std::process::Child as WorkerChild;

/// Ensure the fd consumed by `Command` is not leaked on failure paths.
impl Drop for EngineClient {
    fn drop(&mut self) {
        let _ = self.commands.send(OutgoingFrame::plain(Vec::new()));
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use megalm_protocol::Submission;

    fn item(request_id: &str, stream_id: u64) -> BatchItem {
        let (tx, _rx) = oneshot::channel();
        BatchItem {
            request: SubmitBatch {
                command_id: 0,
                target_active_requests: 1,
                submissions: vec![Submission {
                    request_id: request_id.into(),
                    stream_id,
                    token_ids: vec![1, 2, 3],
                    max_new_tokens: 16,
                    sampling: serde_json::json!({}),
                    lora: None,
                    output_modalities: vec!["text".into()],
                    audio_output: None,
                }],
            },
            reply: tx,
        }
    }

    #[test]
    fn worker_group_detects_rank_failure_and_reaps() {
        let mut group = WorkerGroup {
            leader: Command::new("sleep").arg("60").spawn().unwrap(),
            followers: vec![Command::new("sleep").arg("60").spawn().unwrap()],
        };
        assert!(!group.any_exited());
        group.kill();
        group.wait_all();
        assert!(group.any_exited());
    }

    #[test]
    fn merge_batches_preserves_submissions_and_target() {
        let items = vec![item("a", 1), item("b", 2)];
        let merged = merge_batches(&items, 42);
        assert_eq!(merged.command_id, 42);
        assert_eq!(merged.submissions.len(), 2);
        assert_eq!(merged.submissions[0].request_id, "a");
        assert_eq!(merged.submissions[1].stream_id, 2);
        assert_eq!(merged.target_active_requests, 2);
    }

    /// Receive the fixed 10-byte header together with any SCM_RIGHTS fds.
    fn recv_header_with_fds(fd: RawFd) -> io::Result<(usize, Vec<OwnedFd>)> {
        #[repr(align(8))]
        struct Control([u8; 512]);
        let mut header = [0u8; 10];
        let iov = libc::iovec {
            iov_base: header.as_mut_ptr() as *mut std::os::raw::c_void,
            iov_len: header.len(),
        };
        let mut control = Control([0u8; 512]);
        let mut message: libc::msghdr = unsafe { std::mem::zeroed() };
        message.msg_iov = &iov as *const libc::iovec as *mut libc::iovec;
        message.msg_iovlen = 1;
        message.msg_control = control.0.as_mut_ptr() as *mut std::os::raw::c_void;
        message.msg_controllen = control.0.len();
        // SAFETY: every buffer is live and sized for one call.
        let received = unsafe { libc::recvmsg(fd, &mut message, 0) };
        if received < 0 {
            return Err(io::Error::last_os_error());
        }
        let mut fds = Vec::new();
        // SAFETY: `message` was filled by the successful recvmsg above.
        unsafe {
            let mut header_ptr = libc::CMSG_FIRSTHDR(&message);
            while !header_ptr.is_null() {
                if (*header_ptr).cmsg_level == libc::SOL_SOCKET
                    && (*header_ptr).cmsg_type == libc::SCM_RIGHTS
                {
                    let data = libc::CMSG_DATA(header_ptr) as *const libc::c_int;
                    let count = ((*header_ptr).cmsg_len as usize
                        - libc::CMSG_LEN(0) as usize)
                        / std::mem::size_of::<libc::c_int>();
                    for index in 0..count {
                        fds.push(OwnedFd::from_raw_fd(*data.add(index)));
                    }
                    break;
                }
                header_ptr = libc::CMSG_NXTHDR(&message, header_ptr);
            }
        }
        Ok((received as usize, fds))
    }

    #[tokio::test]
    async fn writer_loop_delivers_frame_with_fd_rights() {
        let (write_end, mut read_end) = UnixStream::pair().unwrap();
        let raw_fd = write_end.as_raw_fd();
        let (_read_half, writer) = tokio::io::split(write_end);
        let (tx, rx) = mpsc::unbounded_channel::<OutgoingFrame>();
        tokio::spawn(writer_loop(raw_fd, writer, rx));

        let sealed = media::seal_memfd(b"writer-loop-media").unwrap();
        let body = serde_json::json!({"probe": true});
        let frame = encode_json(FrameType::SubmitMultimodalBatch, &body).unwrap();
        tx.send(OutgoingFrame {
            bytes: frame.clone(),
            fds: vec![sealed],
        })
        .unwrap();

        let (header_len, fds) = loop {
            match recv_header_with_fds(read_end.as_raw_fd()) {
                Ok(result) => break result,
                Err(err) if err.kind() == io::ErrorKind::WouldBlock => {
                    tokio::task::yield_now().await;
                }
                Err(err) => panic!("recvmsg failed: {err}"),
            }
        };
        assert_eq!(header_len, 10);
        assert_eq!(fds.len(), 1);
        let header = decode_header(&frame[..10]).unwrap();
        assert_eq!(header.kind, FrameType::SubmitMultimodalBatch);

        let mut received = vec![0u8; header.body_len];
        read_end.read_exact(&mut received).await.unwrap();
        assert_eq!(received, frame[10..]);

        let mut payload = [0u8; 17];
        let read = unsafe {
            libc::pread(
                fds[0].as_raw_fd(),
                payload.as_mut_ptr() as *mut std::os::raw::c_void,
                payload.len(),
                0,
            )
        };
        assert_eq!(read, 17);
        assert_eq!(&payload, b"writer-loop-media");
    }

    #[tokio::test]
    async fn coalesced_results_are_fanned_out_by_request_id() {
        let (command_tx, mut command_rx) = mpsc::unbounded_channel::<OutgoingFrame>();
        let pending = Arc::new(Mutex::new(HashMap::new()));
        let next_id = Arc::new(AtomicU64::new(1));
        let (batcher_tx, batcher_rx) = mpsc::unbounded_channel::<BatchItem>();
        tokio::spawn(batcher_loop(
            batcher_rx,
            command_tx,
            Arc::clone(&pending),
            next_id,
            BatchingOptions {
                coalesce_wait: Duration::from_millis(5),
                max_batch_size: 8,
            },
            Arc::new(StdRwLock::new(None)),
        ));

        let (tx_a, rx_a) = oneshot::channel();
        let (tx_b, rx_b) = oneshot::channel();
        batcher_tx
            .send(item_with_reply("a", 1, tx_a))
            .unwrap();
        batcher_tx
            .send(item_with_reply("b", 2, tx_b))
            .unwrap();

        let frame = command_rx.recv().await.unwrap().bytes;
        let header = decode_header(&frame[..10]).unwrap();
        let body: Value = serde_json::from_slice(&frame[10..]).unwrap();
        assert_eq!(header.kind, FrameType::SubmitBatch);
        let submissions = body["submissions"].as_array().unwrap();
        assert_eq!(submissions.len(), 2);

        let command_id = body["command_id"].as_u64().unwrap();
        let reply = serde_json::json!({
            "command_id": command_id,
            "results": [
                {"request_id": "a", "ok": true},
                {"request_id": "b", "ok": true},
            ],
        });
        if let Some(waiter) = pending.lock().await.remove(&command_id) {
            let _ = waiter.send(reply);
        }
        let result_a = rx_a.await.unwrap().unwrap();
        let result_b = rx_b.await.unwrap().unwrap();
        assert_eq!(result_a.results.len(), 1);
        assert_eq!(result_a.results[0].request_id, "a");
        assert_eq!(result_b.results.len(), 1);
        assert_eq!(result_b.results[0].request_id, "b");
    }

    #[tokio::test]
    async fn frame_observer_sees_each_request_sent_and_received() {
        use std::sync::Mutex as StdMutex;
        use crate::FrameObserver;

        #[derive(Default)]
        struct Recorder {
            sent: StdMutex<Vec<String>>,
            received: StdMutex<Vec<String>>,
        }
        impl FrameObserver for Recorder {
            fn frame_sent(&self, request_id: &str, kind: &str) {
                self.sent.lock().unwrap().push(format!("{request_id}:{kind}"));
            }
            fn frame_received(&self, request_id: &str, kind: &str) {
                self.received.lock().unwrap().push(format!("{request_id}:{kind}"));
            }
        }

        let recorder = Arc::new(Recorder::default());
        let observer: crate::SharedFrameObserver =
            Arc::new(StdRwLock::new(Some(recorder.clone())));
        let (command_tx, mut command_rx) = mpsc::unbounded_channel::<OutgoingFrame>();
        let pending = Arc::new(Mutex::new(HashMap::new()));
        let next_id = Arc::new(AtomicU64::new(1));
        let (batcher_tx, batcher_rx) = mpsc::unbounded_channel::<BatchItem>();
        tokio::spawn(batcher_loop(
            batcher_rx,
            command_tx,
            Arc::clone(&pending),
            next_id,
            BatchingOptions {
                coalesce_wait: Duration::from_millis(5),
                max_batch_size: 8,
            },
            observer,
        ));

        let (tx_a, rx_a) = oneshot::channel();
        let (tx_b, rx_b) = oneshot::channel();
        batcher_tx.send(item_with_reply("a", 1, tx_a)).unwrap();
        batcher_tx.send(item_with_reply("b", 2, tx_b)).unwrap();
        let frame = command_rx.recv().await.unwrap().bytes;
        let body: Value = serde_json::from_slice(&frame[10..]).unwrap();
        let command_id = body["command_id"].as_u64().unwrap();
        let reply = serde_json::json!({
            "command_id": command_id,
            "results": [
                {"request_id": "a", "ok": true},
                {"request_id": "b", "ok": true},
            ],
        });
        if let Some(waiter) = pending.lock().await.remove(&command_id) {
            let _ = waiter.send(reply);
        }
        let _ = rx_a.await.unwrap().unwrap();
        let _ = rx_b.await.unwrap().unwrap();

        let mut sent = recorder.sent.lock().unwrap().clone();
        let mut received = recorder.received.lock().unwrap().clone();
        sent.sort();
        received.sort();
        // A coalesced frame is observed once per distinct request id.
        assert_eq!(sent, vec!["a:submit", "b:submit"]);
        assert_eq!(received, vec!["a:submit", "b:submit"]);
    }

    fn item_with_reply(
        request_id: &str,
        stream_id: u64,
        reply: oneshot::Sender<Result<SubmitResult>>,
    ) -> BatchItem {
        BatchItem {
            request: SubmitBatch {
                command_id: 0,
                target_active_requests: 1,
                submissions: vec![Submission {
                    request_id: request_id.into(),
                    stream_id,
                    token_ids: vec![1],
                    max_new_tokens: 1,
                    sampling: serde_json::json!({}),
                    lora: None,
                    output_modalities: vec!["text".into()],
                    audio_output: None,
                }],
            },
            reply,
        }
    }
}
