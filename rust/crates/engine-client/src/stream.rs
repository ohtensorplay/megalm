//! Streamed output events delivered by the worker.

use megalm_protocol::OutputSlot;
use serde_json::Value;

#[derive(Debug, Clone)]
pub struct OutputEvent {
    pub stream_id: u64,
    pub token_ids: Vec<u32>,
    pub audio: Value,
    pub finish_reason: Option<String>,
    pub error: Option<Value>,
    pub logprobs: Vec<f32>,
    pub top_logprobs: Vec<Vec<(u32, f32)>>,
}

impl From<OutputSlot> for OutputEvent {
    fn from(slot: OutputSlot) -> Self {
        Self {
            stream_id: slot.stream_id,
            token_ids: slot.token_ids,
            audio: slot.audio,
            finish_reason: slot.finish_reason,
            error: slot.error,
            logprobs: slot.logprobs,
            top_logprobs: slot.top_logprobs,
        }
    }
}
