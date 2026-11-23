//! Structured parser diagnostics.
//!
//! Streaming parsers must not abort a generation when the model emits a
//! malformed block: the gateway degrades to visible text instead. But silently
//! swallowing the failure makes production incidents undiagnosable. This
//! module carries the failed position, a short context snippet and the
//! expected token out of the Rust core so the gateway can emit a structured
//! log event and a Prometheus counter.

/// Coarse category of a parser failure.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ErrorKind {
    /// A required literal or JSON token did not match.
    UnexpectedToken,
    /// The buffer ended in the middle of a marker, string or object.
    Unterminated,
    /// Braces or brackets were unbalanced.
    Unbalanced,
    /// A JSON string escape or value was invalid.
    InvalidJson,
    /// Generation ended while a tool block was still open.
    IncompleteAtFinish,
}

impl ErrorKind {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::UnexpectedToken => "unexpected_token",
            Self::Unterminated => "unterminated",
            Self::Unbalanced => "unbalanced",
            Self::InvalidJson => "invalid_json",
            Self::IncompleteAtFinish => "incomplete_at_finish",
        }
    }
}

/// One structured parser diagnostic.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ParseError {
    /// Parser that produced the diagnostic, e.g. `"Inkling"`.
    pub parser: &'static str,
    pub kind: ErrorKind,
    pub message: String,
    /// Byte offset into the parser buffer where the failure was detected.
    pub position: usize,
    /// Short raw snippet around `position` (at most 80 chars).
    pub context: String,
    /// Token that was expected at `position`, if any.
    pub expected: String,
}

impl ParseError {
    pub fn new(
        parser: &'static str,
        kind: ErrorKind,
        message: impl Into<String>,
        position: usize,
        buffer: &str,
        expected: impl Into<String>,
    ) -> Self {
        let start = buffer
            .char_indices()
            .take_while(|(index, _)| *index < position)
            .map(|(index, _)| index)
            .last()
            .unwrap_or(0);
        let end = buffer
            .char_indices()
            .map(|(index, _)| index)
            .find(|index| *index >= position + 80)
            .unwrap_or(buffer.len());
        Self {
            parser,
            kind,
            message: message.into(),
            position,
            context: buffer[start..end].to_string(),
            expected: expected.into(),
        }
    }

    /// Human-readable form used in logs and Python error messages.
    pub fn render(&self) -> String {
        if self.expected.is_empty() {
            format!(
                "{}: at byte {}: {} near {:?}",
                self.parser, self.position, self.message, self.context
            )
        } else {
            format!(
                "{}: at byte {}: {} near {:?}: expected {}",
                self.parser, self.position, self.message, self.context, self.expected
            )
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{ErrorKind, ParseError};

    #[test]
    fn render_includes_context_and_expected() {
        let error = ParseError::new(
            "Inkling",
            ErrorKind::UnexpectedToken,
            "invalid tool header",
            3,
            "abc{\"name\":1}",
            "`\"name\"`",
        );
        let text = error.render();
        assert!(text.contains("Inkling"));
        assert!(text.contains("byte 3"));
        assert!(text.contains("expected"));
        assert_eq!(error.kind.as_str(), "unexpected_token");
    }

    #[test]
    fn context_does_not_split_multibyte_chars() {
        let error = ParseError::new(
            "Qwen",
            ErrorKind::Unterminated,
            "buffer ended",
            6, // start of the emoji
            "你好😀世界",
            "",
        );
        // Start is the last boundary strictly before position, so one char of
        // lead-in is kept and no multibyte character is split.
        assert_eq!(error.context, "好😀世界");
    }
}
