//! Errors from the native engine client.

#[derive(Debug, thiserror::Error)]
pub enum EngineClientError {
    #[error("failed to start worker: {0}")]
    Spawn(String),
    #[error("invalid engine configuration: {0}")]
    Config(String),
    #[error("worker transport closed")]
    TransportClosed,
    #[error("worker reported failure: {0}")]
    Worker(String),
    #[error("media transport error: {0}")]
    Media(String),
    #[error("{0}")]
    Message(String),
    #[error(transparent)]
    Protocol(#[from] megalm_protocol::ProtocolError),
    #[error(transparent)]
    Json(#[from] serde_json::Error),
}

pub type Result<T> = std::result::Result<T, EngineClientError>;
