//! MGLM v6 wire protocol used between the frontend and the GPU worker.
//!
//! Frame layout (10-byte header, all integers big-endian):
//! `b"MGLM" | version:u8 = 6 | kind:u8 | body_len:u32 | body`.

use serde::{Deserialize, Serialize};
use serde_json::Value;

pub const MAGIC: [u8; 4] = *b"MGLM";
pub const VERSION: u8 = 6;
pub const HEADER_LEN: usize = 10;
pub const MAX_BODY_BYTES: usize = 64 * 1024 * 1024;

#[repr(u8)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FrameType {
    Ready = 1,
    SubmitBatch = 2,
    SubmitResult = 3,
    Cancel = 4,
    Inspect = 5,
    Snapshot = 6,
    Outputs = 7,
    Stop = 8,
    Failure = 9,
    SubmitMultimodal = 10,
    SubmitMultimodalBatch = 11,
    SubmitEmbeddings = 12,
    EmbeddingResult = 13,
}

impl FrameType {
    pub fn from_u8(value: u8) -> Option<Self> {
        Some(match value {
            1 => Self::Ready,
            2 => Self::SubmitBatch,
            3 => Self::SubmitResult,
            4 => Self::Cancel,
            5 => Self::Inspect,
            6 => Self::Snapshot,
            7 => Self::Outputs,
            8 => Self::Stop,
            9 => Self::Failure,
            10 => Self::SubmitMultimodal,
            11 => Self::SubmitMultimodalBatch,
            12 => Self::SubmitEmbeddings,
            13 => Self::EmbeddingResult,
            _ => return None,
        })
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Header {
    pub kind: FrameType,
    pub body_len: usize,
}

#[derive(Debug, thiserror::Error)]
pub enum ProtocolError {
    #[error("short frame header: got {got}, need {HEADER_LEN}")]
    ShortHeader { got: usize },
    #[error("bad magic in frame header")]
    BadMagic,
    #[error("unsupported protocol version {0}")]
    BadVersion(u8),
    #[error("unknown frame type {0}")]
    UnknownFrameType(u8),
    #[error("frame body too large: {0} bytes")]
    BodyTooLarge(usize),
    #[error(transparent)]
    Json(#[from] serde_json::Error),
}

pub type Result<T> = std::result::Result<T, ProtocolError>;

/// Encode one frame with a JSON body.
pub fn encode_json<T: Serialize>(kind: FrameType, body: &T) -> Result<Vec<u8>> {
    let body = serde_json::to_vec(body)?;
    if body.len() > MAX_BODY_BYTES {
        return Err(ProtocolError::BodyTooLarge(body.len()));
    }
    Ok(encode(kind, &body))
}

/// Encode one frame with a pre-serialized body.
pub fn encode(kind: FrameType, body: &[u8]) -> Vec<u8> {
    let mut frame = Vec::with_capacity(HEADER_LEN + body.len());
    frame.extend_from_slice(&MAGIC);
    frame.push(VERSION);
    frame.push(kind as u8);
    frame.extend_from_slice(&(body.len() as u32).to_be_bytes());
    frame.extend_from_slice(body);
    frame
}

/// Decode and validate a frame header.
pub fn decode_header(header: &[u8]) -> Result<Header> {
    if header.len() < HEADER_LEN {
        return Err(ProtocolError::ShortHeader { got: header.len() });
    }
    if header[..4] != MAGIC {
        return Err(ProtocolError::BadMagic);
    }
    if header[4] != VERSION {
        return Err(ProtocolError::BadVersion(header[4]));
    }
    let kind = FrameType::from_u8(header[5]).ok_or(ProtocolError::UnknownFrameType(header[5]))?;
    let body_len = u32::from_be_bytes([header[6], header[7], header[8], header[9]]) as usize;
    if body_len > MAX_BODY_BYTES {
        return Err(ProtocolError::BodyTooLarge(body_len));
    }
    Ok(Header { kind, body_len })
}

/// Deserialize a frame body into a typed value.
pub fn decode_json<T: serde::de::DeserializeOwned>(header: Header, body: &[u8]) -> Result<T> {
    if body.len() != header.body_len {
        return Err(ProtocolError::BodyTooLarge(body.len()));
    }
    Ok(serde_json::from_slice(body)?)
}

// ---- Request payloads (frontend -> worker) --------------------------------

#[derive(Debug, Clone, Serialize)]
pub struct SubmitBatch {
    pub command_id: u64,
    pub target_active_requests: usize,
    pub submissions: Vec<Submission>,
}

/// A non-generative embedding request: token ids only, no sampling state.
#[derive(Debug, Clone, Serialize)]
pub struct EmbeddingSubmission {
    pub request_id: String,
    pub token_ids: Vec<u32>,
}

#[derive(Debug, Clone, Serialize)]
pub struct SubmitEmbeddings {
    pub command_id: u64,
    pub submissions: Vec<EmbeddingSubmission>,
    pub dimensions: Option<usize>,
    pub normalize: Option<bool>,
    pub truncate_prompt_tokens: Option<usize>,
    pub truncation_side: String,
}

#[derive(Debug, Clone, Deserialize)]
pub struct EmbeddingResultItem {
    pub request_id: String,
    #[serde(default)]
    pub ok: bool,
    #[serde(default)]
    pub error: Option<Value>,
    #[serde(default)]
    pub embedding: Vec<f32>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct EmbeddingResult {
    pub command_id: u64,
    pub results: Vec<EmbeddingResultItem>,
}

#[derive(Debug, Clone, Serialize)]
pub struct Submission {
    pub request_id: String,
    pub stream_id: u64,
    pub token_ids: Vec<u32>,
    pub max_new_tokens: u64,
    pub sampling: Value,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub lora: Option<Value>,
    pub output_modalities: Vec<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub audio_output: Option<Value>,
}

/// One descriptor of a sealed media file passed out-of-band via SCM_RIGHTS.
#[derive(Debug, Clone, Serialize)]
pub struct MediaDescriptor {
    pub kind: String,
    pub mime_type: String,
    pub size: usize,
    pub sha256: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub detail: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub sampling: Option<Value>,
}

/// A multimodal submission: the worker renders `messages` and expands media
/// placeholders itself, so no local token IDs are sent on this path.
#[derive(Debug, Clone, Serialize)]
pub struct MultimodalSubmission {
    pub request_id: String,
    pub stream_id: u64,
    pub messages: Vec<Value>,
    pub media_start: usize,
    pub media_count: usize,
    pub max_new_tokens: u64,
    pub sampling: Value,
    pub open_thinking: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub chat_template_kwargs: Option<Value>,
    pub tools: Vec<Value>,
    pub output_modalities: Vec<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub audio_output: Option<Value>,
}

#[derive(Debug, Clone, Serialize)]
pub struct SubmitMultimodalBatch {
    pub command_id: u64,
    pub target_active_requests: usize,
    pub submissions: Vec<MultimodalSubmission>,
    /// Descriptors in the same order as the file descriptors attached to the
    /// frame's SCM_RIGHTS ancillary data.
    pub media: Vec<MediaDescriptor>,
}

#[derive(Debug, Clone, Serialize)]
pub struct Cancel {
    pub request_id: String,
    pub command_id: u64,
}

#[derive(Debug, Clone, Serialize, Default)]
pub struct Inspect {
    pub command_id: u64,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub op: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub path: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub id: Option<Value>,
    /// Weight-update control plane fields (no-op for other inspect ops).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub expected_digest: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub source: Option<String>,
}

// ---- Response payloads (worker -> frontend) -------------------------------

#[derive(Debug, Clone, Deserialize)]
pub struct Ready {
    pub descriptor: Value,
}

#[derive(Debug, Clone, Deserialize)]
pub struct SubmitResultItem {
    pub request_id: String,
    pub ok: bool,
    #[serde(default)]
    pub error: Option<Value>,
    #[serde(default)]
    pub prompt_tokens: Option<u64>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct SubmitResult {
    pub command_id: u64,
    pub results: Vec<SubmitResultItem>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct Snapshot {
    pub command_id: u64,
    pub snapshot: Value,
}

/// The worker emits a fixed 7-slot array per output batch:
/// `[stream_id, token_ids, audio, finish_reason, error, logprobs, top_logprobs]`.
#[derive(Debug, Clone)]
pub struct OutputSlot {
    pub stream_id: u64,
    pub token_ids: Vec<u32>,
    pub audio: Value,
    pub finish_reason: Option<String>,
    pub error: Option<Value>,
    pub logprobs: Vec<f32>,
    pub top_logprobs: Vec<Vec<(u32, f32)>>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct Outputs {
    pub outputs: Vec<Vec<Value>>,
}

impl Outputs {
    /// Parse the seven fixed slots of each output row.
    pub fn slots(&self) -> Result<Vec<OutputSlot>> {
        self.outputs
            .iter()
            .map(|row| {
                if row.len() != 7 {
                    return Err(ProtocolError::Json(
                        serde_json::from_str::<Value>("null").unwrap_err(),
                    ));
                }
                let stream_id = row[0].as_u64().unwrap_or(0);
                let token_ids = row[1]
                    .as_array()
                    .map(|ids| {
                        ids.iter()
                            .filter_map(|id| id.as_u64().map(|v| v as u32))
                            .collect()
                    })
                    .unwrap_or_default();
                let finish_reason = row[3].as_str().map(str::to_string);
                let logprobs = row[5]
                    .as_array()
                    .map(|values| {
                        values
                            .iter()
                            .filter_map(|v| v.as_f64().map(|f| f as f32))
                            .collect()
                    })
                    .unwrap_or_default();
                let top_logprobs = row[6]
                    .as_array()
                    .map(|entries| {
                        entries
                            .iter()
                            .map(|entry| {
                                entry
                                    .as_array()
                                    .map(|pairs| {
                                        pairs
                                            .iter()
                                            .filter_map(|pair| {
                                                let pair = pair.as_array()?;
                                                Some((
                                                    pair.first()?.as_u64()? as u32,
                                                    pair.get(1)?.as_f64()? as f32,
                                                ))
                                            })
                                            .collect()
                                    })
                                    .unwrap_or_default()
                            })
                            .collect()
                    })
                    .unwrap_or_default();
                Ok(OutputSlot {
                    stream_id,
                    token_ids,
                    audio: row[2].clone(),
                    finish_reason,
                    error: if row[4].is_null() {
                        None
                    } else {
                        Some(row[4].clone())
                    },
                    logprobs,
                    top_logprobs,
                })
            })
            .collect()
    }
}

#[derive(Debug, Clone, Deserialize)]
pub struct Failure {
    pub error: FailureError,
}

#[derive(Debug, Clone, Deserialize)]
pub struct FailureError {
    #[serde(rename = "type")]
    pub kind: String,
    pub message: String,
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn multimodal_batch_matches_worker_wire_shape() {
        let batch = SubmitMultimodalBatch {
            command_id: 7,
            target_active_requests: 1,
            submissions: vec![MultimodalSubmission {
                request_id: "r1".into(),
                stream_id: 3,
                messages: vec![serde_json::json!({
                    "role": "user",
                    "content": [{"type": "image"}],
                })],
                media_start: 0,
                media_count: 1,
                max_new_tokens: 16,
                sampling: serde_json::json!({"temperature": 0.0}),
                open_thinking: false,
                chat_template_kwargs: None,
                tools: Vec::new(),
                output_modalities: vec!["text".into()],
                audio_output: None,
            }],
            media: vec![MediaDescriptor {
                kind: "image".into(),
                mime_type: "image/png".into(),
                size: 15,
                sha256: "abc".into(),
                detail: Some("high".into()),
                sampling: None,
            }],
        };
        let value = serde_json::to_value(&batch).unwrap();
        assert_eq!(value["command_id"], 7);
        assert_eq!(value["target_active_requests"], 1);
        let submission = &value["submissions"][0];
        assert_eq!(submission["request_id"], "r1");
        assert_eq!(submission["stream_id"], 3);
        assert_eq!(submission["media_start"], 0);
        assert_eq!(submission["media_count"], 1);
        assert_eq!(submission["output_modalities"][0], "text");
        assert!(submission.get("audio_output").is_none());
        let descriptor = &value["media"][0];
        assert_eq!(descriptor["kind"], "image");
        assert_eq!(descriptor["mime_type"], "image/png");
        assert_eq!(descriptor["sha256"], "abc");
        assert_eq!(descriptor["detail"], "high");
        assert!(descriptor.get("sampling").is_none());
    }

    #[test]
    fn multimodal_submission_carries_template_kwargs() {
        let submission = MultimodalSubmission {
            request_id: "r1".into(),
            stream_id: 3,
            messages: vec![json!({"role": "user", "content": "x"})],
            media_start: 0,
            media_count: 0,
            max_new_tokens: 16,
            sampling: json!({"temperature": 0.0}),
            open_thinking: false,
            chat_template_kwargs: Some(json!({"reasoning_effort": 0.4})),
            tools: Vec::new(),
            output_modalities: vec!["text".into()],
            audio_output: None,
        };
        let value = serde_json::to_value(&submission).unwrap();
        assert_eq!(value["chat_template_kwargs"]["reasoning_effort"], 0.4);

        let submission = MultimodalSubmission {
            chat_template_kwargs: None,
            ..submission
        };
        let value = serde_json::to_value(&submission).unwrap();
        assert!(value.get("chat_template_kwargs").is_none());
    }

    #[test]
    fn frame_header_round_trips() {
        let frame = encode_json(
            FrameType::SubmitMultimodalBatch,
            &serde_json::json!({"x": 1}),
        )
        .unwrap();
        let header = decode_header(&frame[..HEADER_LEN]).unwrap();
        assert_eq!(header.kind, FrameType::SubmitMultimodalBatch);
        assert_eq!(header.body_len, frame.len() - HEADER_LEN);
    }
}
