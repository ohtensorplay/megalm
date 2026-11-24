//! Media placeholder expansion for multimodal chats.

use crate::error::{message, Result};
use crate::model::ModelTokenizer;

const OXR_IMAGE_START_ID: u32 = 151_931;
const OXR_IMAGE_END_ID: u32 = 151_932;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MultimodalEncoding {
    pub input_ids: Vec<u32>,
    pub token_type_ids: Vec<u8>,
    pub image_count: usize,
    pub video_count: usize,
    pub audio_count: usize,
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn expand_media_placeholders(
    tokenizer: &ModelTokenizer,
    input_ids: &[u32],
    image_feature_counts: &[usize],
    video_frame_feature_counts: &[Vec<usize>],
    video_timestamps: &[Vec<f64>],
    audio_feature_counts: &[usize],
    video_timestamp_tokens: bool,
) -> Result<Vec<u32>> {
    if tokenizer.model_type == "oxr" && tokenizer.image_token_id.is_some() {
        let image_token = tokenizer.image_token_id.unwrap();
        let mut expanded = Vec::with_capacity(input_ids.len());
        let mut image_index = 0usize;
        for id in input_ids {
            if *id == image_token {
                let count = image_feature_counts.get(image_index).copied().ok_or_else(|| {
                    crate::error::FrontendError::Message(
                        "chat has more image placeholders than image inputs".into(),
                    )
                })?;
                expanded.push(OXR_IMAGE_START_ID);
                expanded.extend(std::iter::repeat(image_token).take(count));
                expanded.push(OXR_IMAGE_END_ID);
                image_index += 1;
            } else {
                expanded.push(*id);
            }
        }
        if image_index != image_feature_counts.len() {
            return message("image inputs and chat image placeholders differ");
        }
        return Ok(expanded);
    }

    let start = tokenizer.vision_start_token_id;
    let end = tokenizer.vision_end_token_id;
    let audio_start = tokenizer.audio_start_token_id;
    let audio_end = tokenizer.audio_end_token_id;
    let mut image_index = 0usize;
    let mut video_index = 0usize;
    let mut audio_index = 0usize;
    let mut expanded = Vec::with_capacity(input_ids.len());
    let mut cursor = 0usize;

    while cursor < input_ids.len() {
        let triplet = &input_ids[cursor..(cursor + 3).min(input_ids.len())];
        if triplet.len() == 3
            && tokenizer.image_token_id.is_some()
            && start.is_some()
            && end.is_some()
            && triplet == [start.unwrap(), tokenizer.image_token_id.unwrap(), end.unwrap()]
        {
            let count = image_feature_counts.get(image_index).copied().ok_or_else(|| {
                crate::error::FrontendError::Message(
                    "chat has more image placeholders than image inputs".into(),
                )
            })?;
            expanded.push(start.unwrap());
            expanded.extend(std::iter::repeat(tokenizer.image_token_id.unwrap()).take(count));
            expanded.push(end.unwrap());
            image_index += 1;
            cursor += 3;
            continue;
        }
        if triplet.len() == 3
            && tokenizer.video_token_id.is_some()
            && start.is_some()
            && end.is_some()
            && triplet == [start.unwrap(), tokenizer.video_token_id.unwrap(), end.unwrap()]
        {
            let frames = video_frame_feature_counts.get(video_index).ok_or_else(|| {
                crate::error::FrontendError::Message(
                    "chat has more video placeholders than video inputs".into(),
                )
            })?;
            let stamps = video_timestamps.get(video_index).ok_or_else(|| {
                crate::error::FrontendError::Message("video frame metadata is invalid".into())
            })?;
            if video_timestamp_tokens {
                for (count, timestamp) in frames.iter().zip(stamps) {
                    expanded.extend(tokenizer.encode_timestamp(*timestamp)?);
                    expanded.push(start.unwrap());
                    expanded.extend(
                        std::iter::repeat(tokenizer.video_token_id.unwrap()).take(*count),
                    );
                    expanded.push(end.unwrap());
                }
            } else {
                expanded.push(start.unwrap());
                expanded.extend(
                    std::iter::repeat(tokenizer.video_token_id.unwrap()).take(frames.iter().sum()),
                );
                expanded.push(end.unwrap());
            }
            video_index += 1;
            cursor += 3;
            continue;
        }
        if triplet.len() == 3
            && tokenizer.audio_token_id.is_some()
            && audio_start.is_some()
            && audio_end.is_some()
            && triplet
                == [
                    audio_start.unwrap(),
                    tokenizer.audio_token_id.unwrap(),
                    audio_end.unwrap(),
                ]
        {
            let count = audio_feature_counts.get(audio_index).copied().ok_or_else(|| {
                crate::error::FrontendError::Message(
                    "chat has more audio placeholders than audio inputs".into(),
                )
            })?;
            expanded.push(audio_start.unwrap());
            expanded.extend(std::iter::repeat(tokenizer.audio_token_id.unwrap()).take(count));
            expanded.push(audio_end.unwrap());
            audio_index += 1;
            cursor += 3;
            continue;
        }
        expanded.push(input_ids[cursor]);
        cursor += 1;
    }
    if image_index != image_feature_counts.len() {
        return message("image inputs and chat image placeholders differ");
    }
    if video_index != video_frame_feature_counts.len() {
        return message("video inputs and chat video placeholders differ");
    }
    if audio_index != audio_feature_counts.len() {
        return message("audio inputs and chat audio placeholders differ");
    }
    Ok(expanded)
}
