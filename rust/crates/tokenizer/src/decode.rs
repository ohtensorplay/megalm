//! Stateful incremental detokenization.

use tokenizers::{Tokenizer, TokenizerImpl};

/// Concrete decode-stream type used by the owned [`Tokenizer`].
type Stream<'a> = tokenizers::DecodeStream<
    'a,
    tokenizers::models::ModelWrapper,
    tokenizers::normalizers::NormalizerWrapper,
    tokenizers::pre_tokenizers::PreTokenizerWrapper,
    tokenizers::processors::PostProcessorWrapper,
    tokenizers::decoders::DecoderWrapper,
>;

/// Borrowed wrapper around the native HF decode stream.
pub struct ModelDecodeStream<'a> {
    stream: Stream<'a>,
}

impl<'a> ModelDecodeStream<'a> {
    pub(crate) fn new(tokenizer: &'a Tokenizer, skip_special_tokens: bool) -> Self {
        let inner: &TokenizerImpl<_, _, _, _, _> = tokenizer;
        Self {
            stream: inner.decode_stream(skip_special_tokens),
        }
    }

    /// Advance the decoder with one batch of token ids and return the new text.
    pub fn push(&mut self, token_ids: &[u32]) -> String {
        let mut out = String::new();
        for id in token_ids {
            if let Some(chunk) = self.stream.step(*id).unwrap_or(None) {
                out.push_str(&chunk);
            }
        }
        out
    }
}
