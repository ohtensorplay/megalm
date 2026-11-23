//! Streaming output parsers for the gateway host path.

pub mod error;

mod inkling;
mod json_format;
mod reasoning;
mod output;
mod stats;
mod stop;
mod tool;
mod tool_call;

pub use error::{ErrorKind, ParseError};
pub use inkling::{InklingOutput, InklingParser, MarkerKind};
pub use json_format::JsonFormatValidator;
pub use output::OutputParser;
pub use reasoning::last_reasoning_boundary;
pub use stats::ParserStats;
pub use stop::StopMatcher;
pub use tool::{
    create as create_tool_parser, preserve_special_tokens as tool_parser_preserves_special_tokens,
    Tool, ToolCall, ToolCallDelta, ToolParser, ToolParserEvent, ToolParserOutput, PARSER_NAMES,
};
pub use tool_call::ToolCallParser;

#[cfg(test)]
mod robustness_tests {
    use super::{InklingOutput, InklingParser, OutputParser, StopMatcher, ToolCallParser};

    /// Tiny xorshift generator: deterministic input, no extra dependency.
    struct Rng(u64);

    impl Rng {
        fn next(&mut self) -> u64 {
            let mut x = self.0;
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            self.0 = x;
            x
        }
    }

    const ALPHABET: &[char] = &[
        '{', '}', '[', ']', '"', '\\', ':', ',', '<', '>', '|', '/', '-', '_', 'a', 'b',
        'x', ' ', '\n', '你', '😀', '0', '9',
    ];

    fn noisy(seed: u64, count: usize) -> String {
        let mut rng = Rng(seed);
        (0..count)
            .map(|_| ALPHABET[(rng.next() as usize) % ALPHABET.len()])
            .collect()
    }

    /// Every parser survives arbitrary deltas, arbitrary split points and the
    /// final flush without panicking. A panic here would abort the Python
    /// gateway process, so this is a hard production invariant.
    #[test]
    fn parsers_never_panic_on_arbitrary_input() {
        for seed in 0..64u64 {
            let text = noisy(seed * 7919 + 1, 192);
            let mut rng = Rng(seed ^ 0x5eed);
            let boundaries: Vec<usize> = text
                .char_indices()
                .map(|(index, _)| index)
                .filter(|_| rng.next() % 3 == 0)
                .collect();

            let mut stop = StopMatcher::new(["STOP", "</think>"]);
            let mut tool = ToolCallParser::new("call", vec!["f".to_string()]);
            let mut output = OutputParser::new(
                "call",
                vec!["STOP".to_string()],
                vec!["f".to_string()],
                Some("<think>".to_string()),
                Some("</think>".to_string()),
                false,
            );
            let mut inkling = InklingParser::new("call");
            let mut inkling_output = InklingOutput::default();

            let mut start = 0usize;
            for end in boundaries.iter().copied().chain(std::iter::once(text.len())) {
                let chunk = &text[start..end];
                let _ = stop.push(chunk);
                let _ = tool.push(chunk);
                let _ = output.push(chunk);
                inkling.push(chunk, &mut inkling_output);
                start = end;
            }
            let _ = stop.finish();
            let _ = tool.finish();
            let _ = output.finish();
            inkling.finish(&mut inkling_output);
        }
    }
}
