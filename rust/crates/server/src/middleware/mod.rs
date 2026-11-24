//! Axum middleware for the Rust frontend.

mod auth;

pub use auth::authenticate_api_key;
