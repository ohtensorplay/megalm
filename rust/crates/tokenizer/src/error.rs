//! Errors raised by the local tokenizer/template frontend.

#[derive(Debug, thiserror::Error)]
pub enum FrontendError {
    #[error("{0}")]
    Message(String),
    #[error("cannot load tokenizer assets from {path}: {source}")]
    Load {
        path: String,
        #[source]
        source: std::io::Error,
    },
    #[error("invalid JSON in {path}: {source}")]
    Json {
        path: String,
        #[source]
        source: serde_json::Error,
    },
    #[error(transparent)]
    Tokenizer(#[from] tokenizers::Error),
    #[error(transparent)]
    Renderer(#[from] megalm_renderer::RenderError),
}

pub type Result<T> = std::result::Result<T, FrontendError>;

pub(crate) fn message<T>(text: impl Into<String>) -> Result<T> {
    Err(FrontendError::Message(text.into()))
}
