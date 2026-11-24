//! Remote media resolution for chat requests (port of `media_fetch.py`).
//!
//! Only public HTTP(S) endpoints are reachable. DNS is resolved through a
//! validating resolver and every redirect hop is re-validated, so a redirect
//! cannot bypass the domain allowlist or reach a private address.

use std::net::{IpAddr, Ipv4Addr, Ipv6Addr, SocketAddr};
use std::sync::Arc;
use std::time::{Duration, Instant};

use base64::Engine as _;
use futures::StreamExt;
use reqwest::dns::{Addrs, Name, Resolve, Resolving};
use serde_json::{json, Value};
use tokio::sync::Semaphore;

use crate::media::{resolve_media_mime, MediaKind};

const MAX_IMAGE_BYTES: usize = 20 * 1024 * 1024;
const MAX_VIDEO_BYTES: usize = 64 * 1024 * 1024;
const MAX_AUDIO_BYTES: usize = 32 * 1024 * 1024;
const REDIRECT_STATUSES: &[u16] = &[301, 302, 303, 307, 308];

#[derive(Debug, Clone)]
pub struct FetchConfig {
    pub enabled: bool,
    pub allow_http: bool,
    pub max_redirects: usize,
    pub max_attempts: usize,
    pub max_concurrent: usize,
    pub connect_timeout: Duration,
    pub read_timeout: Duration,
    pub total_timeout: Duration,
    pub allowed_domains: Vec<String>,
    pub user_agent: String,
}

fn env_bool(name: &str, default: bool) -> bool {
    match std::env::var(name) {
        Ok(value) => matches!(value.trim().to_ascii_lowercase().as_str(), "1" | "true" | "yes" | "on"),
        Err(_) => default,
    }
}

fn env_usize(name: &str, default: usize, min: usize, max: usize) -> usize {
    std::env::var(name)
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .filter(|value| *value >= min && *value <= max)
        .unwrap_or(default)
}

fn env_secs(name: &str, default: f64, min: f64, max: f64) -> Duration {
    let value = std::env::var(name)
        .ok()
        .and_then(|value| value.parse::<f64>().ok())
        .filter(|value| value.is_finite() && *value >= min && *value <= max)
        .unwrap_or(default);
    Duration::from_secs_f64(value)
}

impl FetchConfig {
    pub fn from_env() -> Self {
        let connect_timeout = env_secs("MEGALM_MEDIA_CONNECT_TIMEOUT_S", 5.0, 0.1, 60.0);
        let total_timeout = env_secs("MEGALM_MEDIA_TOTAL_TIMEOUT_S", 60.0, 1.0, 300.0);
        Self {
            enabled: env_bool("MEGALM_MEDIA_REMOTE_URLS", true),
            allow_http: env_bool("MEGALM_MEDIA_ALLOW_HTTP", false),
            max_redirects: env_usize("MEGALM_MEDIA_MAX_REDIRECTS", 3, 0, 8),
            max_attempts: env_usize("MEGALM_MEDIA_MAX_ATTEMPTS", 2, 1, 3),
            max_concurrent: env_usize("MEGALM_MEDIA_MAX_CONCURRENT", 4, 1, 16),
            connect_timeout: connect_timeout.min(total_timeout),
            read_timeout: env_secs("MEGALM_MEDIA_READ_TIMEOUT_S", 15.0, 0.1, 120.0),
            total_timeout,
            allowed_domains: std::env::var("MEGALM_MEDIA_ALLOWED_DOMAINS")
                .unwrap_or_default()
                .split(',')
                .map(|entry| entry.trim().trim_end_matches('.').to_ascii_lowercase())
                .filter(|entry| !entry.is_empty())
                .collect(),
            user_agent: "MegaLM-MediaFetch/1.0".into(),
        }
    }
}

#[derive(Debug)]
pub struct FetchError {
    pub code: &'static str,
    pub status: u16,
    pub message: String,
}

impl FetchError {
    fn new(code: &'static str, status: u16, message: impl Into<String>) -> Self {
        Self {
            code,
            status,
            message: message.into(),
        }
    }
}

fn v4_is_public(address: Ipv4Addr) -> bool {
    let octets = address.octets();
    let [a, b, c, d] = octets;
    if a == 0 || a == 10 || a == 127 || a >= 224 {
        return false;
    }
    if a == 100 && (64..=127).contains(&b) {
        return false;
    }
    if a == 169 && b == 254 {
        return false;
    }
    if a == 172 && (16..=31).contains(&b) {
        return false;
    }
    if a == 192 && b == 168 {
        return false;
    }
    if a == 192 && b == 0 && c == 0 {
        return false;
    }
    if a == 192 && b == 0 && c == 2 {
        return false;
    }
    if a == 192 && b == 88 && c == 99 {
        return false;
    }
    if a == 198 && (b == 18 || b == 19) {
        return false;
    }
    if a == 198 && b == 51 && c == 100 {
        return false;
    }
    if a == 203 && b == 0 && c == 113 {
        return false;
    }
    // 255.255.255.255 is already covered by a >= 224; keep `d` used for
    // future byte-level checks without triggering dead-code lints.
    let _ = d;
    true
}

fn v6_is_public(address: Ipv6Addr) -> bool {
    if let Some(mapped) = address.to_ipv4_mapped() {
        return v4_is_public(mapped);
    }
    let o = address.octets();
    let prefix = &o[..8];
    if prefix.iter().all(|byte| *byte == 0) && (o[15] == 0 || o[15] == 1) {
        return false; // :: and ::1
    }
    if o[0] & 0xfe == 0xfc {
        return false; // fc00::/7 unique local
    }
    if o[0] == 0xfe && (o[1] & 0xc0) == 0x80 {
        return false; // fe80::/10 link local
    }
    if o[0] == 0xff {
        return false; // multicast
    }
    if o[0] == 0x20 && o[1] == 0x01 && o[2] == 0x0d && o[3] == 0xb8 {
        return false; // 2001:db8::/32 documentation
    }
    if o[0] == 0x20 && o[1] == 0x01 && (o[2] & 0xfe) == 0x00 {
        return false; // 2001::/23
    }
    if o[0] == 0x00 && o[1] == 0x64 && o[2] == 0xff && o[3] == 0x9b {
        return false; // 64:ff9b::/96
    }
    if prefix == [0x01, 0x00, 0x00, 0x00, 0, 0, 0, 0] {
        return false; // 100::/64 discard-only
    }
    if o[0] == 0x20 && o[1] == 0x02 {
        return false; // 2002::/16 6to4
    }
    true
}

fn address_is_public(address: IpAddr) -> bool {
    match address {
        IpAddr::V4(v4) => v4_is_public(v4),
        IpAddr::V6(v6) => v6_is_public(v6),
    }
}

struct PublicResolver;

impl Resolve for PublicResolver {
    fn resolve(&self, name: Name) -> Resolving {
        let host = name.as_str().to_string();
        Box::pin(async move {
            let mut addresses = Vec::new();
            if let Ok(ip) = host.parse::<IpAddr>() {
                if !address_is_public(ip) {
                    return Err("media URL resolves to a non-public address".into());
                }
                addresses.push(SocketAddr::new(ip, 0));
            } else {
                let resolved = tokio::net::lookup_host((host.as_str(), 0))
                    .await
                    .map_err(|_| "media URL host could not be resolved")?;
                for address in resolved {
                    if !address_is_public(address.ip()) {
                        return Err("media URL resolves to a non-public address".into());
                    }
                    addresses.push(SocketAddr::new(address.ip(), 0));
                }
            }
            if addresses.is_empty() {
                return Err("media URL host has no usable address".into());
            }
            Ok(Box::new(addresses.into_iter()) as Addrs)
        })
    }
}

fn assert_domain_allowed(host: &str, config: &FetchConfig) -> Result<(), FetchError> {
    if config.allowed_domains.is_empty() {
        return Ok(());
    }
    let host = host.trim_end_matches('.').to_ascii_lowercase();
    for rule in &config.allowed_domains {
        if let Some(suffix) = rule.strip_prefix('.') {
            if host == suffix || host.ends_with(rule) {
                return Ok(());
            }
        } else if host == *rule {
            return Ok(());
        }
    }
    Err(FetchError::new(
        "domain_not_allowed",
        400,
        format!("media domain {host:?} is not in the allowlist"),
    ))
}

#[derive(Debug)]
struct ParsedUrl {
    host: String,
}

fn parse_url(url: &str, config: &FetchConfig) -> Result<ParsedUrl, FetchError> {
    let parsed = reqwest::Url::parse(url)
        .map_err(|_| FetchError::new("invalid_url", 400, "media URL is invalid"))?;
    let scheme = parsed.scheme().to_ascii_lowercase();
    if scheme != "http" && scheme != "https" {
        return Err(FetchError::new(
            "unsupported_scheme",
            400,
            "media URL scheme is not supported",
        ));
    }
    if scheme == "http" && !config.allow_http {
        return Err(FetchError::new(
            "http_disabled",
            400,
            "http media URLs are disabled; use https",
        ));
    }
    if parsed.username() != "" || parsed.password().is_some() {
        return Err(FetchError::new(
            "userinfo_disallowed",
            400,
            "media URL must not contain credentials",
        ));
    }
    let host = parsed
        .host_str()
        .ok_or_else(|| FetchError::new("missing_host", 400, "media URL has no host"))?
        .to_ascii_lowercase();
    let port = parsed
        .port()
        .unwrap_or(if scheme == "https" { 443 } else { 80 });
    if !(1..=65535).contains(&port) {
        return Err(FetchError::new("invalid_port", 400, "media URL port is out of range"));
    }
    Ok(ParsedUrl { host })
}

fn modality_limit(kind: MediaKind) -> usize {
    match kind {
        MediaKind::Image => MAX_IMAGE_BYTES,
        MediaKind::Video => MAX_VIDEO_BYTES,
        MediaKind::Audio => MAX_AUDIO_BYTES,
    }
}

async fn send_with_retry(
    client: &reqwest::Client,
    url: &str,
    config: &FetchConfig,
    deadline: Instant,
) -> Result<reqwest::Response, FetchError> {
    for attempt in 0..config.max_attempts {
        if Instant::now() >= deadline {
            return Err(FetchError::new("timeout", 504, "media fetch timed out"));
        }
        match client.get(url).send().await {
            Ok(response) => {
                let status = response.status().as_u16();
                let retryable = (500..600).contains(&status) || status == 408 || status == 429;
                if retryable && attempt + 1 < config.max_attempts {
                    continue;
                }
                return Ok(response);
            }
            Err(_) if attempt + 1 < config.max_attempts => continue,
            Err(_) => {
                return Err(FetchError::new(
                    "upstream_unreachable",
                    502,
                    "media host could not be reached",
                ))
            }
        }
    }
    Err(FetchError::new("timeout", 504, "media fetch timed out"))
}

async fn fetch_remote_media(
    url: &str,
    kind: MediaKind,
    config: &FetchConfig,
) -> Result<(String, Vec<u8>), FetchError> {
    if !config.enabled {
        return Err(FetchError::new("remote_disabled", 400, "remote media URLs are disabled"));
    }
    let deadline = Instant::now() + config.total_timeout;
    let client = reqwest::Client::builder()
        .dns_resolver(Arc::new(PublicResolver))
        .redirect(reqwest::redirect::Policy::none())
        .connect_timeout(config.connect_timeout)
        .timeout(config.read_timeout)
        .user_agent(&config.user_agent)
        .build()
        .map_err(|_| FetchError::new("invalid_config", 500, "cannot build media fetch client"))?;

    let mut current = url.to_string();
    let mut redirects = 0usize;
    loop {
        if Instant::now() >= deadline {
            return Err(FetchError::new("timeout", 504, "media fetch timed out"));
        }
        let parsed = parse_url(&current, config)?;
        assert_domain_allowed(&parsed.host, config)?;
        if let Ok(ip) = parsed.host.parse::<IpAddr>() {
            if !address_is_public(ip) {
                return Err(FetchError::new(
                    "blocked_address",
                    400,
                    "media URL resolves to a non-public address",
                ));
            }
        }

        let response = send_with_retry(&client, &current, config, deadline).await?;
        let status = response.status().as_u16();
        if REDIRECT_STATUSES.contains(&status) {
            if let Some(location) = response.headers().get(reqwest::header::LOCATION) {
                if let Ok(location) = location.to_str() {
                    redirects += 1;
                    if redirects > config.max_redirects {
                        return Err(FetchError::new(
                            "too_many_redirects",
                            400,
                            "media URL redirected too many times",
                        ));
                    }
                    current = reqwest::Url::parse(&current)
                        .and_then(|base| base.join(location))
                        .map(|url| url.to_string())
                        .map_err(|_| FetchError::new("invalid_url", 400, "redirect URL is invalid"))?;
                    continue;
                }
            }
        }
        if !(200..300).contains(&status) {
            let status_code = if (400..500).contains(&status) { 422 } else { 502 };
            return Err(FetchError::new(
                "upstream_status",
                status_code,
                format!("media host returned HTTP {status}"),
            ));
        }
        let declared = response
            .headers()
            .get(reqwest::header::CONTENT_TYPE)
            .and_then(|value| value.to_str().ok())
            .map(str::to_string);
        let limit = modality_limit(kind);
        let mut payload = Vec::new();
        let mut stream = response.bytes_stream();
        while let Some(chunk) = stream.next().await {
            let chunk = chunk.map_err(|_| {
                FetchError::new("upstream_unreachable", 502, "media host could not be reached")
            })?;
            payload.extend_from_slice(&chunk);
            if payload.len() > limit {
                return Err(FetchError::new(
                    "invalid_media",
                    400,
                    format!("{} payload exceeds its byte limit", kind.as_str()),
                ));
            }
        }
        if payload.is_empty() {
            return Err(FetchError::new(
                "invalid_media",
                400,
                "remote media payload is empty",
            ));
        }
        let mime_type = resolve_media_mime(declared.as_deref(), &payload, kind)
            .map_err(|message| FetchError::new("invalid_media", 400, message))?;
        return Ok((mime_type, payload));
    }
}

/// Replace remote media URLs in chat messages with inline data URLs so the
/// existing synchronous parser and sealed memfd transport stay unchanged.
pub async fn resolve_remote_media(
    messages: &[Value],
    config: &FetchConfig,
) -> Result<Vec<Value>, FetchError> {
    let mut normalized: Vec<Value> = messages.to_vec();
    let mut pending = Vec::new();
    for (message_index, message) in normalized.iter().enumerate() {
        let Some(parts) = message.get("content").and_then(Value::as_array) else {
            continue;
        };
        for (part_index, part) in parts.iter().enumerate() {
            let Some(kind) = part.get("type").and_then(Value::as_str) else {
                continue;
            };
            let modality = match kind {
                "image_url" => MediaKind::Image,
                "video_url" => MediaKind::Video,
                "audio_url" => MediaKind::Audio,
                _ => continue,
            };
            let source = part.get(kind);
            let url = match source {
                Some(Value::String(url)) => url.clone(),
                Some(value @ Value::Object(_)) => value
                    .get("url")
                    .and_then(Value::as_str)
                    .unwrap_or("")
                    .to_string(),
                _ => continue,
            };
            if url.is_empty() || url.starts_with("data:") {
                continue;
            }
            if reqwest::Url::parse(&url)
                .map(|parsed| !parsed.scheme().is_empty())
                .unwrap_or(false)
            {
                pending.push((message_index, part_index, kind.to_string(), url, modality));
            }
        }
    }
    if pending.is_empty() {
        return Ok(normalized);
    }
    if !config.enabled {
        return Err(FetchError::new("remote_disabled", 400, "remote media URLs are disabled"));
    }

    let semaphore = Arc::new(Semaphore::new(config.max_concurrent));
    let mut tasks = Vec::with_capacity(pending.len());
    for (message_index, part_index, key, url, modality) in pending {
        let semaphore = semaphore.clone();
        let config = config.clone();
        tasks.push(async move {
            let _permit = semaphore.acquire().await.unwrap();
            let (mime_type, payload) = fetch_remote_media(&url, modality, &config).await?;
            let encoded = base64::engine::general_purpose::STANDARD.encode(&payload);
            Ok::<_, FetchError>((
                message_index,
                part_index,
                key,
                format!("data:{mime_type};base64,{encoded}"),
            ))
        });
    }
    let resolved = futures::future::try_join_all(tasks).await?;
    for (message_index, part_index, key, data_url) in resolved {
        let source = &mut normalized[message_index]["content"][part_index][key];
        if source.is_string() {
            *source = json!({"url": data_url});
        } else if let Some(map) = source.as_object_mut() {
            map.insert("url".into(), json!(data_url));
        }
    }
    Ok(normalized)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn private_addresses_are_rejected() {
        for address in [
            "127.0.0.1",
            "10.0.0.1",
            "172.16.0.1",
            "192.168.1.1",
            "169.254.1.1",
            "100.64.0.1",
            "224.0.0.1",
            "0.0.0.0",
            "::1",
            "fc00::1",
            "fe80::1",
            "::ffff:127.0.0.1",
            "2001:db8::1",
        ] {
            let ip: IpAddr = address.parse().unwrap();
            assert!(!address_is_public(ip), "{address} should be blocked");
        }
        for address in ["8.8.8.8", "1.1.1.1", "2606:4700:4700::1111"] {
            let ip: IpAddr = address.parse().unwrap();
            assert!(address_is_public(ip), "{address} should be public");
        }
    }

    #[test]
    fn domain_allowlist_matches_exact_and_subdomains() {
        let config = FetchConfig {
            allowed_domains: vec!["example.com".into(), ".trusted.org".into()],
            ..FetchConfig {
                enabled: true,
                allow_http: false,
                max_redirects: 3,
                max_attempts: 2,
                max_concurrent: 4,
                connect_timeout: Duration::from_secs(5),
                read_timeout: Duration::from_secs(15),
                total_timeout: Duration::from_secs(60),
                allowed_domains: Vec::new(),
                user_agent: String::new(),
            }
        };
        assert!(assert_domain_allowed("EXAMPLE.COM.", &config).is_ok());
        assert!(assert_domain_allowed("cdn.trusted.org", &config).is_ok());
        assert!(assert_domain_allowed("evil-example.com", &config).is_err());
    }

    #[test]
    fn http_requires_explicit_opt_in() {
        let mut config = FetchConfig::from_env();
        config.allow_http = false;
        assert_eq!(
            parse_url("http://example.com/a.png", &config)
                .unwrap_err()
                .code,
            "http_disabled"
        );
        assert!(parse_url("https://example.com/a.png", &config).is_ok());
        assert_eq!(
            parse_url("file:///etc/passwd", &config).unwrap_err().code,
            "unsupported_scheme"
        );
    }

    #[tokio::test]
    async fn data_urls_are_left_untouched() {
        let config = FetchConfig {
            enabled: false,
            ..FetchConfig::from_env()
        };
        let messages = vec![json!({
            "role": "user",
            "content": [{"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}}]
        })];
        let resolved = resolve_remote_media(&messages, &config).await.unwrap();
        assert_eq!(resolved, messages);
    }
}
