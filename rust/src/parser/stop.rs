//! Streaming stop-sequence matcher.

/// Streaming stop-sequence matcher over decoded text.
///
/// Feed decoded deltas to [`push`][StopMatcher::push]; it returns the text
/// that is safe to emit upstream and whether a stop sequence matched. A tail
/// that could still grow into a stop sequence is held back until more text
/// arrives or [`finish`][StopMatcher::finish] flushes it.
#[derive(Debug, Default)]
pub struct StopMatcher {
    stops: Vec<String>,
    pending: String,
    terminal: bool,
    holdback: usize,
}

impl StopMatcher {
    /// Build a matcher for the given stop sequences.
    ///
    /// Empty stop sequences are ignored: an empty string would match
    /// everywhere and stall the stream.
    pub fn new<I, S>(stops: I) -> Self
    where
        I: IntoIterator<Item = S>,
        S: Into<String>,
    {
        let stops: Vec<String> = stops
            .into_iter()
            .map(Into::into)
            .filter(|stop| !stop.is_empty())
            .collect();
        let holdback = stops
            .iter()
            .map(|stop| stop.chars().count())
            .max()
            .unwrap_or(1)
            - 1;
        Self {
            stops,
            holdback,
            ..Self::default()
        }
    }

    /// Whether a stop sequence already matched.
    ///
    /// Public library API; the Python facade latches this into its return
    /// tuple instead of calling the accessor.
    #[allow(dead_code)]
    pub fn is_stopped(&self) -> bool {
        self.terminal
    }

    /// Feed one decoded delta; returns the safe-to-emit text and whether a
    /// stop sequence matched in this or any earlier delta.
    pub fn push(&mut self, delta: &str) -> (String, bool) {
        if self.terminal {
            return (String::new(), true);
        }
        self.pending.push_str(delta);
        if let Some(index) = self.earliest_match() {
            self.terminal = true;
            let emitted = self.pending[..index].to_string();
            self.pending.clear();
            return (emitted, true);
        }
        let emitted = self.drain_safe_prefix();
        (emitted, false)
    }

    /// Flush the held-back tail when generation ends without a stop match.
    pub fn finish(&mut self) -> String {
        std::mem::take(&mut self.pending)
    }

    /// Byte index of the earliest stop match inside `pending`, if any.
    fn earliest_match(&self) -> Option<usize> {
        self.stops
            .iter()
            .filter_map(|stop| self.pending.find(stop.as_str()))
            .min()
    }

    /// Emit everything except a tail that could still grow into a match.
    fn drain_safe_prefix(&mut self) -> String {
        if self.holdback == 0 {
            return std::mem::take(&mut self.pending);
        }
        let chars = self.pending.chars().count();
        let keep = self.holdback.min(chars);
        let split = chars - keep;
        if split == 0 {
            return String::new();
        }
        let index = self
            .pending
            .char_indices()
            .nth(split)
            .map(|(index, _)| index)
            .unwrap_or(self.pending.len());
        let emitted = self.pending[..index].to_string();
        self.pending.drain(..index);
        emitted
    }
}

#[cfg(test)]
mod tests {
    use super::StopMatcher;

    #[test]
    fn passes_through_without_stops() {
        let mut matcher = StopMatcher::new(Vec::<String>::new());
        assert_eq!(matcher.push("hello "), ("hello ".into(), false));
        assert_eq!(matcher.push("world"), ("world".into(), false));
        assert_eq!(matcher.finish(), "");
    }

    #[test]
    fn holds_back_partial_stop_prefix() {
        let mut matcher = StopMatcher::new(["STOP"]);
        let (delta, stopped) = matcher.push("abcST");
        assert!(!stopped);
        assert_eq!(delta, "ab");
        let (delta, stopped) = matcher.push("OP!next");
        assert!(stopped);
        assert_eq!(delta, "c");
        // Everything after the stop is suppressed.
        assert_eq!(matcher.push("more"), ("".into(), true));
    }

    #[test]
    fn matches_across_delta_boundaries() {
        let mut matcher = StopMatcher::new(["</tool>"]);
        assert_eq!(matcher.push("a</to"), ("".into(), false));
        let (delta, stopped) = matcher.push("ol>b");
        assert!(stopped);
        assert_eq!(delta, "a");
        let _ = matcher.finish();
    }

    #[test]
    fn picks_the_earliest_of_multiple_stops() {
        let mut matcher = StopMatcher::new(["zzzz", "END"]);
        let (delta, stopped) = matcher.push("abcENDzzzz");
        assert!(stopped);
        assert_eq!(delta, "abc");
    }

    #[test]
    fn finish_flushes_held_back_tail() {
        let mut matcher = StopMatcher::new(["STOP"]);
        let _ = matcher.push("abcST");
        assert_eq!(matcher.finish(), "cST");
    }

    #[test]
    fn multibyte_boundaries_are_respected() {
        let mut matcher = StopMatcher::new(["结束"]);
        let (delta, stopped) = matcher.push("你好，结");
        assert!(!stopped);
        assert_eq!(delta, "你好，");
        let (delta, stopped) = matcher.push("束了");
        assert!(stopped);
        assert_eq!(delta, "");
    }
}

