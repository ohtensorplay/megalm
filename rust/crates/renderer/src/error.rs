//! Errors from loading or rendering a model's chat template.

#[derive(Debug, thiserror::Error)]
pub enum RenderError {
    #[error("{0}")]
    Message(String),
    #[error("cannot read {path}: {source}")]
    Io {
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
    #[error("template error: {0}")]
    Template(#[from] minijinja::Error),
}

pub type Result<T> = std::result::Result<T, RenderError>;
