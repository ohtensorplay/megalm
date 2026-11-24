//! Bearer-token authentication for guarded HTTP routes.
//!
//! An API key is optional; when configured, the server accepts one of the keys
//! in the `Authorization: Bearer <key>` header. OPTIONS preflight requests and
//! helper endpoints such as `/health` stay open.

use axum::Json;
use axum::extract::{Request, State};
use axum::http::header::AUTHORIZATION;
use axum::http::{HeaderValue, Method, StatusCode};
use axum::middleware::Next;
use axum::response::{IntoResponse, Response};
use serde_json::json;
use subtle::ConstantTimeEq;

use crate::state::{ApiKeyHash, AuthState, hash_api_key};

/// Route prefixes that expose models or mutate server state.
const GUARDED_PREFIXES: &[&str] = &["/v1", "/v2", "/inference", "/api"];

/// Helper endpoints that must stay reachable without credentials (probes,
/// Prometheus scraping, and server introspection).
const UNGUARDED_PATHS: &[&str] = &["/health", "/api/health", "/metrics", "/server_info"];

/// Reject unauthenticated requests to guarded routes when API keys are set.
///
/// The middleware is only installed when keys exist, so unauthenticated local
/// deployments keep working with no added hot-path cost.
pub async fn authenticate_api_key(
    State(auth): State<AuthState>,
    req: Request,
    next: Next,
) -> Response {
    if req.method() == Method::OPTIONS || !requires_auth(req.uri().path()) {
        return next.run(req).await;
    }
    if verify_token(req.headers().get(AUTHORIZATION), auth.api_key_hashes()) {
        return next.run(req).await;
    }

    (
        StatusCode::UNAUTHORIZED,
        Json(json!({ "error": "Unauthorized" })),
    )
        .into_response()
}

fn requires_auth(path: &str) -> bool {
    if UNGUARDED_PATHS.contains(&path) {
        return false;
    }
    GUARDED_PREFIXES.iter().any(|prefix| path.starts_with(prefix))
}

fn verify_token(authorization: Option<&HeaderValue>, api_key_hashes: &[ApiKeyHash]) -> bool {
    let Some(authorization) = authorization else {
        return false;
    };
    let Ok(authorization) = authorization.to_str() else {
        return false;
    };
    let Some((scheme, token)) = authorization.split_once(' ') else {
        return false;
    };
    if !scheme.eq_ignore_ascii_case("bearer") || token.is_empty() {
        return false;
    }

    let token_hash = hash_api_key(token);
    let mut matched = false;
    for api_key_hash in api_key_hashes {
        matched |= constant_time_eq(&token_hash, api_key_hash);
    }
    matched
}

fn constant_time_eq(left: &ApiKeyHash, right: &ApiKeyHash) -> bool {
    bool::from(left.ct_eq(right))
}

#[cfg(test)]
mod tests {
    use axum::body::{Body, to_bytes};
    use axum::http::Request;
    use axum::middleware::from_fn_with_state;
    use axum::routing::get;
    use axum::{Json, Router};
    use serde_json::{Value, json};
    use tower::ServiceExt;

    use super::{HeaderValue, authenticate_api_key, constant_time_eq, requires_auth, verify_token};
    use crate::state::{AuthState, hash_api_key};

    fn test_router(auth: AuthState) -> Router {
        Router::new()
            .route("/v1/models", get(|| async { Json(json!({"object": "list"})) }))
            .route("/health", get(|| async { "ok" }))
            .route("/api/health", get(|| async { "ok" }))
            .layer(from_fn_with_state(auth, authenticate_api_key))
    }

    async fn call(
        app: &Router,
        method: &str,
        uri: &str,
        authorization: Option<&str>,
    ) -> (axum::http::StatusCode, Value) {
        let mut builder = Request::builder().method(method).uri(uri);
        if let Some(authorization) = authorization {
            builder = builder.header("authorization", authorization);
        }
        let response = app
            .clone()
            .oneshot(builder.body(Body::empty()).expect("build request"))
            .await
            .expect("call router");
        let status = response.status();
        let bytes = to_bytes(response.into_body(), usize::MAX)
            .await
            .expect("read body");
        let body = serde_json::from_slice(&bytes).unwrap_or(Value::Null);
        (status, body)
    }

    #[test]
    fn hash_is_stable_and_distinguishes_keys() {
        assert_eq!(hash_api_key("secret"), hash_api_key("secret"));
        assert_ne!(hash_api_key("secret"), hash_api_key("secrex"));
    }

    #[test]
    fn constant_time_eq_checks_sha256_digests() {
        assert!(constant_time_eq(&hash_api_key("secret"), &hash_api_key("secret")));
        assert!(!constant_time_eq(&hash_api_key("secret"), &hash_api_key("secrex")));
        assert!(!constant_time_eq(
            &hash_api_key("secret"),
            &hash_api_key("secret-more")
        ));
    }

    #[test]
    fn verify_token_accepts_any_configured_bearer_key() {
        let hashes = vec![hash_api_key("secret-a"), hash_api_key("secret-b")];
        assert!(verify_token(
            Some(&HeaderValue::from_static("Bearer secret-a")),
            &hashes
        ));
        assert!(verify_token(
            Some(&HeaderValue::from_static("bearer secret-b")),
            &hashes
        ));
    }

    #[test]
    fn verify_token_rejects_missing_malformed_and_wrong_tokens() {
        let hashes = vec![hash_api_key("secret")];
        assert!(!verify_token(None, &hashes));
        assert!(!verify_token(
            Some(&HeaderValue::from_static("Basic secret")),
            &hashes
        ));
        assert!(!verify_token(
            Some(&HeaderValue::from_static("Bearer")),
            &hashes
        ));
        assert!(!verify_token(
            Some(&HeaderValue::from_static("Bearer wrong")),
            &hashes
        ));
        assert!(!verify_token(Some(&HeaderValue::from_static("Bearer secret")), &[]));
    }

    #[test]
    fn requires_auth_covers_model_and_admin_routes_only() {
        for path in [
            "/v1/models",
            "/v1/chat/completions",
            "/api/inference/load",
            "/api/models/loras",
        ] {
            assert!(requires_auth(path), "{path} should be guarded");
        }
        for path in ["/health", "/api/health", "/metrics", "/server_info"] {
            assert!(!requires_auth(path), "{path} should stay open");
        }
    }

    #[tokio::test]
    async fn guarded_route_rejects_missing_or_wrong_token() {
        let app = test_router(AuthState::new(vec!["secret".into()]));
        for authorization in [None, Some("Bearer wrong"), Some("Basic secret")] {
            let (status, body) = call(&app, "GET", "/v1/models", authorization).await;
            assert_eq!(status, axum::http::StatusCode::UNAUTHORIZED);
            assert_eq!(body, json!({"error": "Unauthorized"}));
        }
    }

    #[tokio::test]
    async fn guarded_route_accepts_any_configured_key() {
        let app = test_router(AuthState::new(vec!["secret-a".into(), "secret-b".into()]));
        for token in ["secret-a", "secret-b"] {
            let (status, _) = call(&app, "GET", "/v1/models", Some(&format!("Bearer {token}"))).await;
            assert_eq!(status, axum::http::StatusCode::OK);
        }
    }

    #[tokio::test]
    async fn health_and_options_bypass_auth() {
        let app = test_router(AuthState::new(vec!["secret".into()]));
        for path in ["/health", "/api/health"] {
            let (status, _) = call(&app, "GET", path, None).await;
            assert_eq!(status, axum::http::StatusCode::OK, "{path} should stay open");
        }
        let (status, _) = call(&app, "OPTIONS", "/v1/models", None).await;
        assert_ne!(status, axum::http::StatusCode::UNAUTHORIZED);
    }
}
