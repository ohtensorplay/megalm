//! Reasoning-state detection and tag-delimited reasoning splitting.
//!
//! Mirrors the reference parser's `reasoning` module: the prompt boundary is
//! derived from token IDs (not a text suffix), and `*think*` delimiters are
//! stripped from decoded text before it reaches the caller.

/// Splits decoded text into reasoning vs. content by text markers.
///
/// Qwen3-style models emit ``<think>`` … ``</think>`` as plain tokens, so the
/// reasoning section survives special-token skipping and can be separated on
/// the gateway host path. A generation prompt that already ends with the start
/// marker seeds `in_thinking`, since the model then continues inside the
/// reasoning block without re-emitting the marker.
#[derive(Debug)]
pub(super) struct ThinkingSplitter {
    start: String,
    end: String,
    pending: String,
    in_thinking: bool,
    /// Optional prefix emitted by the model at the start of reasoning that
    /// must not be exposed (Gemma-4 emits `thought\n`).
    strip_prefix: Option<String>,
    prefix_buffer: String,
}

impl ThinkingSplitter {
    pub(super) fn new(start: String, end: String, initial_in_thinking: bool) -> Self {
        Self::with_prefix(start, end, initial_in_thinking, None)
    }

    pub(super) fn with_prefix(
        start: String,
        end: String,
        initial_in_thinking: bool,
        strip_prefix: Option<String>,
    ) -> Self {
        Self {
            start,
            end,
            pending: String::new(),
            in_thinking: initial_in_thinking,
            strip_prefix,
            prefix_buffer: String::new(),
        }
    }

    pub(super) fn set_prefix(&mut self, prefix: impl Into<String>) {
        self.strip_prefix = Some(prefix.into());
    }

    fn emit_reasoning(
        &mut self,
        text: &str,
        reasoning: &mut String,
        final_flush: bool,
    ) {
        let Some(prefix) = self.strip_prefix.clone() else {
            reasoning.push_str(text);
            return;
        };
        self.prefix_buffer.push_str(text);
        if self.prefix_buffer.starts_with(&prefix) {
            let rest = self.prefix_buffer[prefix.len()..].to_string();
            self.prefix_buffer.clear();
            reasoning.push_str(&rest);
        } else if prefix.starts_with(&self.prefix_buffer) {
            if final_flush {
                let rest = std::mem::take(&mut self.prefix_buffer);
                reasoning.push_str(&rest);
            }
        } else {
            let rest = std::mem::take(&mut self.prefix_buffer);
            reasoning.push_str(&rest);
        }
    }

    pub(super) fn push(&mut self, delta: &str) -> (String, String) {
        self.pending.push_str(delta);
        self.drain(false)
    }

    pub(super) fn finish(&mut self) -> (String, String) {
        self.drain(true)
    }

    fn drain(&mut self, final_flush: bool) -> (String, String) {
        let mut content = String::new();
        let mut reasoning = String::new();
        loop {
            let marker = if self.in_thinking {
                &self.end
            } else {
                &self.start
            };
            match self.pending.find(marker.as_str()) {
                Some(index) => {
                    let text = self.pending[..index].to_string();
                    self.pending.drain(..index + marker.len());
                    if self.in_thinking {
                        self.emit_reasoning(&text, &mut reasoning, final_flush);
                    } else {
                        content.push_str(&text);
                    }
                    self.in_thinking = !self.in_thinking;
                }
                None => break,
            }
        }
        if !self.pending.is_empty() {
            let split = if final_flush {
                self.pending.len()
            } else {
                self.safe_split_index()
            };
            let text = self.pending[..split].to_string();
            self.pending.drain(..split);
            if self.in_thinking {
                self.emit_reasoning(&text, &mut reasoning, final_flush);
            } else {
                content.push_str(&text);
            }
        }
        (content, reasoning)
    }

    /// Byte index up to which `pending` is definitely safe to emit: the
    /// earliest tail position where either marker could still be starting.
    /// A marker can only begin within the last `marker_len` bytes, so the
    /// scan window stays short regardless of pending size.
    ///
    /// Positions come from `char_indices()` rather than a raw byte range, so
    /// a multibyte character inside the window is never sliced in half. This
    /// mirrors the reference `partial_suffix_len()` scan.
    fn safe_split_index(&self) -> usize {
        let max_len = self.start.len().max(self.end.len());
        let window_start = self.pending.len().saturating_sub(max_len);
        for (pos, _) in self.pending.char_indices() {
            if pos < window_start {
                continue;
            }
            let suffix = &self.pending[pos..];
            if self.start.starts_with(suffix) || self.end.starts_with(suffix) {
                return pos;
            }
        }
        self.pending.len()
    }
}

/// Determine the reasoning state implied by the last reasoning boundary in a
/// prompt, if any.
///
/// Scans token IDs backwards. `<think>` starts reasoning, `</think>` ends it,
/// and any other special token stops the scan — a chat template's trailing
/// role/header token must not be mistaken for "no reasoning boundary".
/// `None` means no boundary was found and the caller's default applies.
pub fn last_reasoning_boundary(
    prompt_token_ids: &[u32],
    start_token_id: u32,
    end_token_id: u32,
    special_token_ids: &[u32],
) -> Option<bool> {
    for token_id in prompt_token_ids.iter().rev().copied() {
        if token_id == start_token_id {
            return Some(true);
        }
        if token_id == end_token_id {
            return Some(false);
        }
        if special_token_ids.contains(&token_id) {
            return None;
        }
    }
    None
}

#[cfg(test)]
mod reasoning_tests {
    use super::{last_reasoning_boundary, ThinkingSplitter};

    const START: u32 = 200_001;
    const END: u32 = 200_002;
    const OTHER_SPECIAL: u32 = 200_100;
    const SPECIALS: &[u32] = &[START, END, OTHER_SPECIAL];

    #[test]
    fn start_token_means_reasoning() {
        assert_eq!(last_reasoning_boundary(&[START], START, END, SPECIALS), Some(true));
    }

    #[test]
    fn end_token_means_content() {
        assert_eq!(last_reasoning_boundary(&[END], START, END, SPECIALS), Some(false));
    }

    #[test]
    fn unrelated_special_token_stops_the_scan() {
        // The newest token is a chat-template header: the older start token
        // must not leak through.
        assert_eq!(
            last_reasoning_boundary(&[START, OTHER_SPECIAL], START, END, SPECIALS),
            None
        );
    }

    #[test]
    fn plain_tokens_do_not_stop_the_scan() {
        assert_eq!(
            last_reasoning_boundary(&[START, 42, 7], START, END, SPECIALS),
            Some(true)
        );
    }

    #[test]
    fn empty_prompt_has_no_boundary() {
        assert_eq!(last_reasoning_boundary(&[], START, END, SPECIALS), None);
    }

    #[test]
    fn gemma4_thought_prefix_is_stripped_across_deltas() {
        let mut splitter = ThinkingSplitter::with_prefix(
            "<|channel>".into(),
            "<channel|>".into(),
            false,
            Some("thought\n".into()),
        );
        let (content, reasoning) = splitter.push("<|channel>tho");
        assert!(content.is_empty() && reasoning.is_empty());
        let (content, reasoning) = splitter.push("ught\nBecause reasons<channel|>done");
        assert_eq!(reasoning, "Because reasons");
        assert_eq!(content, "done");
    }
}
