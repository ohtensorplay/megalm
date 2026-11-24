//! Parsing and validation of inline multimodal chat media.
//!
//! Port of `chat_media.parse_media_chat`: only inline base64 data URLs are
//! accepted, every payload is MIME/signature checked, and each message's media
//! parts are replaced with bare `{"type": "..."}` placeholders that the worker
//! expands against the sealed descriptors sent out-of-band.

use base64::Engine as _;
use serde_json::{json, Value};
use sha2::{Digest, Sha256};

const MAX_HTTP_IMAGES: usize = 8;
const MAX_HTTP_VIDEOS: usize = 1;
const MAX_HTTP_AUDIOS: usize = 4;
const MAX_HTTP_IMAGE_BYTES: usize = 20 * 1024 * 1024;
const MAX_HTTP_VIDEO_BYTES: usize = 64 * 1024 * 1024;
const MAX_HTTP_AUDIO_BYTES: usize = 32 * 1024 * 1024;
const MAX_HTTP_MEDIA_BYTES: usize = 64 * 1024 * 1024;

const IMAGE_TYPES: &[&str] = &["image/gif", "image/jpeg", "image/png", "image/webp"];
const VIDEO_TYPES: &[&str] = &[
    "video/mp4",
    "video/quicktime",
    "video/webm",
    "video/x-matroska",
    "video/x-msvideo",
];
const AUDIO_TYPES: &[&str] = &[
    "audio/flac",
    "audio/mp4",
    "audio/mpeg",
    "audio/ogg",
    "audio/wav",
    "audio/x-wav",
];

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MediaKind {
    Image,
    Video,
    Audio,
}

impl MediaKind {
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Image => "image",
            Self::Video => "video",
            Self::Audio => "audio",
        }
    }
}

#[derive(Debug, Clone)]
pub struct ParsedMediaInput {
    pub kind: MediaKind,
    pub mime_type: String,
    pub payload: Vec<u8>,
    pub sha256: String,
    pub detail: Option<String>,
    pub sampling: Option<Value>,
}

#[derive(Debug, Clone)]
pub struct ParsedMediaChat {
    pub messages: Vec<Value>,
    pub inputs: Vec<ParsedMediaInput>,
}

/// Fast pre-check so the text path is untouched for ordinary chats.
pub fn has_media_parts(messages: &[Value]) -> bool {
    messages.iter().any(|message| {
        message
            .get("content")
            .and_then(Value::as_array)
            .is_some_and(|parts| {
                parts.iter().any(|part| {
                    !matches!(
                        part.get("type").and_then(Value::as_str),
                        None | Some("text")
                    )
                })
            })
    })
}

pub fn parse_media_chat(messages: &[Value]) -> Result<ParsedMediaChat, String> {
    let mut normalized = Vec::with_capacity(messages.len());
    let mut inputs: Vec<ParsedMediaInput> = Vec::new();
    let mut total_bytes = 0usize;

    for message in messages {
        let Some(content) = message.get("content") else {
            return Err("message content must be text or a non-empty content-part list".into());
        };
        if content.is_string() {
            normalized.push(message.clone());
            continue;
        }
        let Some(parts) = content.as_array() else {
            return Err("message content must be text or a non-empty content-part list".into());
        };
        if parts.is_empty() {
            return Err("message content must be text or a non-empty content-part list".into());
        }
        let mut out_parts = Vec::with_capacity(parts.len());
        for part in parts {
            if !part.is_object() {
                return Err("message content part must be an object".into());
            }
            let kind = part.get("type").and_then(Value::as_str);
            match kind {
                Some("text") => {
                    let text = part
                        .get("text")
                        .and_then(Value::as_str)
                        .ok_or("text content part is invalid")?;
                    out_parts.push(json!({"type": "text", "text": text}));
                }
                Some("image_url") => {
                    if count(&inputs, MediaKind::Image) >= MAX_HTTP_IMAGES {
                        return Err(format!("at most {MAX_HTTP_IMAGES} images are allowed"));
                    }
                    let source = source_object(part.get("image_url"), "image_url")?;
                    let detail = validate_image_options(&source)?;
                    let (mime, payload) = decode_data_url(
                        source.get("url").unwrap_or(&Value::Null),
                        MediaKind::Image,
                        MAX_HTTP_IMAGE_BYTES,
                    )?;
                    total_bytes += payload.len();
                    validate_total(total_bytes)?;
                    validate_signature(MediaKind::Image, &mime, &payload)?;
                    let digest = hex_sha256(&payload);
                    inputs.push(ParsedMediaInput {
                        kind: MediaKind::Image,
                        mime_type: mime,
                        payload,
                        sha256: digest,
                        detail: Some(detail),
                        sampling: None,
                    });
                    out_parts.push(json!({"type": "image"}));
                }
                Some("video_url") => {
                    if count(&inputs, MediaKind::Video) >= MAX_HTTP_VIDEOS {
                        return Err(format!("at most {MAX_HTTP_VIDEOS} video is allowed"));
                    }
                    let source = source_object(part.get("video_url"), "video_url")?;
                    let sampling = video_sampling(&source)?;
                    let (mime, payload) = decode_data_url(
                        source.get("url").unwrap_or(&Value::Null),
                        MediaKind::Video,
                        MAX_HTTP_VIDEO_BYTES,
                    )?;
                    total_bytes += payload.len();
                    validate_total(total_bytes)?;
                    validate_signature(MediaKind::Video, &mime, &payload)?;
                    let digest = hex_sha256(&payload);
                    inputs.push(ParsedMediaInput {
                        kind: MediaKind::Video,
                        mime_type: mime,
                        payload,
                        sha256: digest,
                        detail: None,
                        sampling: Some(sampling),
                    });
                    out_parts.push(json!({"type": "video"}));
                }
                Some("audio_url") | Some("input_audio") => {
                    if count(&inputs, MediaKind::Audio) >= MAX_HTTP_AUDIOS {
                        return Err(format!("at most {MAX_HTTP_AUDIOS} audio inputs are allowed"));
                    }
                    let (mime, payload) = if kind == Some("audio_url") {
                        let source = source_object(part.get("audio_url"), "audio_url")?;
                        let keys: Vec<&String> = source
                            .as_object()
                            .map(|map| map.keys().collect())
                            .unwrap_or_default();
                        if keys != ["url"] {
                            return Err("audio_url only supports the url field".into());
                        }
                        decode_data_url(
                            source.get("url").unwrap_or(&Value::Null),
                            MediaKind::Audio,
                            MAX_HTTP_AUDIO_BYTES,
                        )?
                    } else {
                        decode_input_audio(part.get("input_audio").unwrap_or(&Value::Null))?
                    };
                    total_bytes += payload.len();
                    validate_total(total_bytes)?;
                    validate_signature(MediaKind::Audio, &mime, &payload)?;
                    let digest = hex_sha256(&payload);
                    inputs.push(ParsedMediaInput {
                        kind: MediaKind::Audio,
                        mime_type: mime,
                        payload,
                        sha256: digest,
                        detail: None,
                        sampling: None,
                    });
                    out_parts.push(json!({"type": "audio"}));
                }
                Some("audio") => {
                    return Err("audio content must use audio_url or input_audio".into())
                }
                other => return Err(format!("unsupported content part type {other:?}")),
            }
        }
        let mut message = message.clone();
        message["content"] = json!(out_parts);
        normalized.push(message);
    }

    if inputs.is_empty() {
        return Err("multimodal chat contains no media".into());
    }
    Ok(ParsedMediaChat {
        messages: normalized,
        inputs,
    })
}

fn count(inputs: &[ParsedMediaInput], kind: MediaKind) -> usize {
    inputs.iter().filter(|input| input.kind == kind).count()
}

fn source_object(value: Option<&Value>, name: &str) -> Result<Value, String> {
    match value {
        Some(Value::String(url)) => Ok(json!({"url": url})),
        Some(value @ Value::Object(_)) if value.get("url").and_then(Value::as_str).is_some() => {
            Ok(value.clone())
        }
        _ => Err(format!("{name} must contain a string URL")),
    }
}

fn validate_image_options(source: &Value) -> Result<String, String> {
    if let Some(map) = source.as_object() {
        for key in map.keys() {
            if key != "url" && key != "detail" {
                return Err(format!("unsupported image options: {key}"));
            }
        }
    }
    let detail = source
        .get("detail")
        .and_then(Value::as_str)
        .unwrap_or("auto");
    if !matches!(detail, "auto" | "low" | "high") {
        return Err("image detail must be one of 'auto', 'low' or 'high'".into());
    }
    Ok(detail.to_string())
}

fn video_sampling(source: &Value) -> Result<Value, String> {
    let allowed = [
        "url",
        "fps",
        "target_fps",
        "num_frames",
        "start_seconds",
        "end_seconds",
    ];
    if let Some(map) = source.as_object() {
        for key in map.keys() {
            if !allowed.contains(&key.as_str()) {
                return Err(format!("unsupported video options: {key}"));
            }
        }
    }
    if source.get("fps").is_some() && source.get("target_fps").is_some() {
        return Err("video fps and target_fps are aliases".into());
    }
    let num_frames = match source.get("num_frames") {
        None | Some(Value::Null) => None,
        Some(Value::Number(number)) if number.is_i64() => Some(number.as_i64().unwrap()),
        _ => return Err("video num_frames is invalid".into()),
    };
    let explicit_fps = source.get("target_fps").or_else(|| source.get("fps"));
    let target_fps = if num_frames.is_some() && explicit_fps.is_none() {
        None
    } else {
        match explicit_fps {
            None => Some(2.0),
            Some(value) => Some(
                value
                    .as_f64()
                    .ok_or("video target_fps is invalid")?,
            ),
        }
    };
    let start_seconds = match source.get("start_seconds") {
        None | Some(Value::Null) => 0.0,
        Some(value) => value.as_f64().ok_or("video start_seconds is invalid")?,
    };
    let end_seconds = match source.get("end_seconds") {
        None | Some(Value::Null) => None,
        Some(value) => Some(value.as_f64().ok_or("video end_seconds is invalid")?),
    };
    Ok(json!({
        "target_fps": target_fps,
        "num_frames": num_frames,
        "start_seconds": start_seconds,
        "end_seconds": end_seconds,
    }))
}

fn decode_data_url(value: &Value, kind: MediaKind, limit: usize) -> Result<(String, Vec<u8>), String> {
    let Some(url) = value.as_str() else {
        return Err(format!(
            "{} URL must be an inline base64 data URL; remote and filesystem URLs are disabled",
            kind.as_str()
        ));
    };
    let Some(body) = url.strip_prefix("data:") else {
        return Err(format!(
            "{} URL must be an inline base64 data URL; remote and filesystem URLs are disabled",
            kind.as_str()
        ));
    };
    let (header, encoded) = body
        .split_once(',')
        .ok_or_else(|| format!("{} data URL must use base64", kind.as_str()))?;
    let Some(mime_prefix) = header.strip_suffix(";base64") else {
        return Err(format!("{} data URL must use base64", kind.as_str()));
    };
    let mime_type = mime_prefix.to_ascii_lowercase();
    let allowed = match kind {
        MediaKind::Image => IMAGE_TYPES,
        MediaKind::Video => VIDEO_TYPES,
        MediaKind::Audio => AUDIO_TYPES,
    };
    if !allowed.contains(&mime_type.as_str()) {
        return Err(format!("unsupported {} MIME type {mime_type:?}", kind.as_str()));
    }
    if encoded.is_empty() || encoded.len() > ((limit + 2) / 3) * 4 {
        return Err(format!("{} payload exceeds its byte limit", kind.as_str()));
    }
    let payload = base64::engine::general_purpose::STANDARD
        .decode(encoded)
        .map_err(|_| format!("{} data URL contains invalid base64", kind.as_str()))?;
    if payload.is_empty() || payload.len() > limit {
        return Err(format!("{} payload exceeds its byte limit", kind.as_str()));
    }
    Ok((mime_type, payload))
}

fn decode_input_audio(value: &Value) -> Result<(String, Vec<u8>), String> {
    let map = value
        .as_object()
        .ok_or("input_audio must be an object")?;
    if map.len() != 2 || !map.contains_key("data") || !map.contains_key("format") {
        return Err("input_audio requires only data and format fields".into());
    }
    let format = map
        .get("format")
        .and_then(Value::as_str)
        .unwrap_or("")
        .to_ascii_lowercase();
    let mime_type = match format.as_str() {
        "flac" => "audio/flac",
        "mp3" | "mpeg" => "audio/mpeg",
        "ogg" => "audio/ogg",
        "wav" => "audio/wav",
        other => return Err(format!("unsupported input_audio format {other:?}")),
    };
    let encoded = map
        .get("data")
        .and_then(Value::as_str)
        .ok_or("input_audio data must be base64")?;
    if encoded.is_empty() || encoded.len() > ((MAX_HTTP_AUDIO_BYTES + 2) / 3) * 4 {
        return Err("audio payload exceeds its byte limit".into());
    }
    let payload = base64::engine::general_purpose::STANDARD
        .decode(encoded)
        .map_err(|_| "input_audio data contains invalid base64".to_string())?;
    if payload.is_empty() || payload.len() > MAX_HTTP_AUDIO_BYTES {
        return Err("audio payload exceeds its byte limit".into());
    }
    Ok((mime_type.to_string(), payload))
}

fn validate_total(total: usize) -> Result<(), String> {
    if total > MAX_HTTP_MEDIA_BYTES {
        Err("request media exceeds its total byte limit".into())
    } else {
        Ok(())
    }
}

/// Detect a supported MIME type from magic bytes only (port of
/// `chat_media.sniff_media_mime`).
pub fn sniff_media_mime(payload: &[u8], kind: MediaKind) -> Option<String> {
    let mime = match kind {
        MediaKind::Image => {
            if payload.starts_with(b"\x89PNG\r\n\x1a\n") {
                "image/png"
            } else if payload.starts_with(b"\xff\xd8\xff") {
                "image/jpeg"
            } else if payload.starts_with(b"GIF87a") || payload.starts_with(b"GIF89a") {
                "image/gif"
            } else if payload.len() >= 12 && payload.starts_with(b"RIFF") && &payload[8..12] == b"WEBP" {
                "image/webp"
            } else {
                return None;
            }
        }
        MediaKind::Video => {
            if payload.len() >= 12 && &payload[4..8] == b"ftyp" {
                "video/mp4"
            } else if payload.starts_with(b"\x1a\x45\xdf\xa3") {
                "video/webm"
            } else if payload.len() >= 12 && payload.starts_with(b"RIFF") && &payload[8..12] == b"AVI " {
                "video/x-msvideo"
            } else {
                return None;
            }
        }
        MediaKind::Audio => {
            if payload.len() >= 12 && payload.starts_with(b"RIFF") && &payload[8..12] == b"WAVE" {
                "audio/wav"
            } else if payload.starts_with(b"fLaC") {
                "audio/flac"
            } else if payload.starts_with(b"OggS") {
                "audio/ogg"
            } else if payload.starts_with(b"ID3")
                || (payload.len() >= 2 && payload[0] == 0xFF && payload[1] & 0xE0 == 0xE0)
            {
                "audio/mpeg"
            } else if payload.len() >= 12 && &payload[4..8] == b"ftyp" {
                "audio/mp4"
            } else {
                return None;
            }
        }
    };
    Some(mime.to_string())
}

/// Port of `chat_media.resolve_media_mime`: a declared type must agree with
/// the bytes; a generic or missing type is resolved purely from magic bytes.
pub fn resolve_media_mime(
    declared: Option<&str>,
    payload: &[u8],
    kind: MediaKind,
) -> Result<String, String> {
    let detected = sniff_media_mime(payload, kind)
        .ok_or_else(|| format!("{} bytes are not a recognized media format", kind.as_str()))?;
    let allowed = match kind {
        MediaKind::Image => IMAGE_TYPES,
        MediaKind::Video => VIDEO_TYPES,
        MediaKind::Audio => AUDIO_TYPES,
    };
    let normalized = declared
        .map(|value| value.split(';').next().unwrap_or("").trim().to_ascii_lowercase())
        .unwrap_or_default();
    let generic = normalized.is_empty()
        || normalized == "application/octet-stream"
        || normalized == "binary/octet-stream";
    if generic {
        if !allowed.contains(&detected.as_str()) {
            return Err(format!("unsupported {} MIME type {detected:?}", kind.as_str()));
        }
        return Ok(detected);
    }
    if !allowed.contains(&normalized.as_str()) {
        return Err(format!("unsupported {} MIME type {normalized:?}", kind.as_str()));
    }
    let aliases: &[(&str, &str)] = match kind {
        MediaKind::Video => &[("video/quicktime", "video/mp4")],
        MediaKind::Audio => &[("audio/x-wav", "audio/wav")],
        MediaKind::Image => &[],
    };
    let normalized_alias = aliases
        .iter()
        .find(|(from, _)| *from == normalized.as_str())
        .map(|(_, to)| *to)
        .unwrap_or(normalized.as_str());
    let detected_alias = aliases
        .iter()
        .find(|(from, _)| *from == detected.as_str())
        .map(|(_, to)| *to)
        .unwrap_or(detected.as_str());
    if normalized_alias != detected_alias {
        validate_signature(kind, &normalized, payload)?;
    }
    Ok(normalized)
}

fn validate_signature(kind: MediaKind, mime: &str, payload: &[u8]) -> Result<(), String> {
    let valid = match (kind, mime) {
        (MediaKind::Image, "image/png") => payload.starts_with(b"\x89PNG\r\n\x1a\n"),
        (MediaKind::Image, "image/jpeg") => payload.starts_with(b"\xff\xd8\xff"),
        (MediaKind::Image, "image/gif") => {
            payload.starts_with(b"GIF87a") || payload.starts_with(b"GIF89a")
        }
        (MediaKind::Image, "image/webp") => {
            payload.len() >= 12 && payload.starts_with(b"RIFF") && &payload[8..12] == b"WEBP"
        }
        (MediaKind::Video, "video/mp4") | (MediaKind::Video, "video/quicktime") => {
            payload.len() >= 12 && &payload[4..8] == b"ftyp"
        }
        (MediaKind::Video, "video/webm") | (MediaKind::Video, "video/x-matroska") => {
            payload.starts_with(b"\x1a\x45\xdf\xa3")
        }
        (MediaKind::Video, "video/x-msvideo") => {
            payload.len() >= 12 && payload.starts_with(b"RIFF") && &payload[8..12] == b"AVI "
        }
        (MediaKind::Audio, "audio/wav") | (MediaKind::Audio, "audio/x-wav") => {
            payload.len() >= 12 && payload.starts_with(b"RIFF") && &payload[8..12] == b"WAVE"
        }
        (MediaKind::Audio, "audio/flac") => payload.starts_with(b"fLaC"),
        (MediaKind::Audio, "audio/ogg") => payload.starts_with(b"OggS"),
        (MediaKind::Audio, "audio/mpeg") => {
            payload.starts_with(b"ID3")
                || (payload.len() >= 2 && payload[0] == 0xFF && payload[1] & 0xE0 == 0xE0)
        }
        (MediaKind::Audio, "audio/mp4") => payload.len() >= 12 && &payload[4..8] == b"ftyp",
        _ => false,
    };
    if valid {
        Ok(())
    } else {
        Err(format!("{} bytes do not match the declared MIME type", kind.as_str()))
    }
}

fn hex_sha256(payload: &[u8]) -> String {
    let digest = Sha256::digest(payload);
    let mut text = String::with_capacity(64);
    for byte in digest {
        text.push_str(&format!("{byte:02x}"));
    }
    text
}

#[cfg(test)]
mod tests {
    use super::*;

    fn png_data_url() -> String {
        let png = b"\x89PNG\r\n\x1a\npayload";
        format!(
            "data:image/png;base64,{}",
            base64::engine::general_purpose::STANDARD.encode(png)
        )
    }

    #[test]
    fn parses_image_part_and_computes_digest() {
        let messages = vec![json!({
            "role": "user",
            "content": [
                {"type": "text", "text": "what is this"},
                {"type": "image_url", "image_url": {"url": png_data_url(), "detail": "high"}}
            ]
        })];
        assert!(has_media_parts(&messages));
        let parsed = parse_media_chat(&messages).unwrap();
        assert_eq!(parsed.inputs.len(), 1);
        assert_eq!(parsed.inputs[0].kind, MediaKind::Image);
        assert_eq!(parsed.inputs[0].sha256.len(), 64);
        let parts = parsed.messages[0]["content"].as_array().unwrap();
        assert_eq!(parts[0]["type"], "text");
        assert_eq!(parts[1], json!({"type": "image"}));
    }

    #[test]
    fn rejects_non_data_urls_and_bad_signatures() {
        let messages = vec![json!({
            "role": "user",
            "content": [{"type": "image_url", "image_url": {"url": "https://example.com/x.png"}}]
        })];
        assert!(parse_media_chat(&messages).is_err());

        let messages = vec![json!({
            "role": "user",
            "content": [{"type": "image_url", "image_url": {
                "url": format!("data:image/png;base64,{}",
                    base64::engine::general_purpose::STANDARD.encode(b"not a png"))
            }}]
        })];
        assert!(parse_media_chat(&messages).is_err());
    }

    #[test]
    fn parses_input_audio_and_video_sampling() {
        let wav = b"RIFF1234WAVEfmt data";
        let messages = vec![json!({
            "role": "user",
            "content": [{"type": "input_audio", "input_audio": {
                "data": base64::engine::general_purpose::STANDARD.encode(wav),
                "format": "wav"
            }}]
        })];
        let parsed = parse_media_chat(&messages).unwrap();
        assert_eq!(parsed.inputs[0].mime_type, "audio/wav");

        let sampling = video_sampling(&json!({"url": "x", "num_frames": 8, "start_seconds": 1.5}))
            .unwrap();
        assert_eq!(sampling["target_fps"], Value::Null);
        assert_eq!(sampling["num_frames"], 8);
        assert_eq!(sampling["start_seconds"], 1.5);
    }
}
