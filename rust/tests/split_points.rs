//! Integration tests: every possible delta split point must produce the same
//! parse result as feeding the whole block at once.
//!
//! The gateway has no control over where the tokenizer splits a special token
//! or a JSON object across SSE deltas. The reference parser suite calls this
//! `split_by_chars`; here we cover the full cross product for representative
//! Inkling and JSON tool-call inputs.

use megalm_rs::parser::{InklingOutput, InklingParser, ToolCallParser};

const INKLING: &str = concat!(
    "<|message_model|><|content_thinking|>reason<|end_message|>",
    "<|message_model|><|content_text|>answer<|end_message|>",
    "<|message_model|>get_weather<|content_invoke_tool_json|>",
    "{\"name\":\"get_weather\",\"args\":{\"city\":\"SF\",\"n\":2}}",
    "<|end_message|>",
);

/// The production `ToolCallParser` uses the generic JSON grammar (no wrapper
/// markers), which is how Qwen-style output reaches it with special tokens
/// skipped. Marker-wrapped grammars live in the sibling `tool` module.
const JSON_TOOL: &str = concat!(
    "Let me check. ",
    "{\"name\":\"get_weather\",\"arguments\":{\"city\":\"SF\",\"n\":2}}",
    " done",
);

/// Feed `text` as deltas split at every single byte boundary (UTF-8 safe).
fn split_everywhere_inkling(text: &str) -> Vec<InklingOutput> {
    let boundaries: Vec<usize> = text.char_indices().map(|(index, _)| index).collect();
    let mut results = Vec::new();
    for split in boundaries {
        let mut parser = InklingParser::new("call");
        let mut output = InklingOutput::default();
        parser.push(&text[..split], &mut output);
        parser.push(&text[split..], &mut output);
        parser.finish(&mut output);
        results.push(output);
    }
    results
}

#[test]
fn inkling_result_is_invariant_to_every_split_point() {
    let results = split_everywhere_inkling(INKLING);
    let first = &results[0];
    assert_eq!(first.reasoning, "reason");
    assert_eq!(first.content, "answer");
    assert_eq!(first.calls.len(), 1);
    assert_eq!(first.calls[0].name, "get_weather");
    assert_eq!(first.calls[0].arguments, r#"{"city":"SF","n":2}"#);
    for (index, output) in results.iter().enumerate().skip(1) {
        assert_eq!(output, first, "split at boundary {index} diverged");
    }
}

#[test]
fn inkling_result_is_invariant_to_character_by_character_delivery() {
    let mut parser = InklingParser::new("call");
    let mut output = InklingOutput::default();
    for character in INKLING.chars() {
        let mut buffer = [0u8; 4];
        parser.push(character.encode_utf8(&mut buffer), &mut output);
    }
    parser.finish(&mut output);
    assert_eq!(output.reasoning, "reason");
    assert_eq!(output.content, "answer");
    assert_eq!(output.calls.len(), 1);
}

#[test]
fn json_tool_result_is_invariant_to_every_split_point() {
    let boundaries: Vec<usize> = JSON_TOOL.char_indices().map(|(index, _)| index).collect();
    let mut reference: Option<(String, Vec<_>)> = None;
    for split in boundaries.iter().copied() {
        let mut parser = ToolCallParser::new("call", vec!["get_weather".into()]);
        let (mut visible, mut calls) = parser.push(&JSON_TOOL[..split]);
        let (tail, tail_calls) = parser.push(&JSON_TOOL[split..]);
        let (flushed, flushed_calls) = parser.finish();
        visible.push_str(&tail);
        visible.push_str(&flushed);
        calls.extend(tail_calls);
        calls.extend(flushed_calls);
        match &reference {
            None => reference = Some((visible, calls)),
            Some((ref expected_visible, ref expected_calls)) => {
                assert_eq!(visible, *expected_visible, "split at {split} diverged (text)");
                assert_eq!(calls, *expected_calls, "split at {split} diverged (calls)");
            }
        }
    }
    let (visible, calls) = reference.unwrap();
    assert_eq!(visible, "Let me check.  done");
    assert_eq!(calls.len(), 1);
    assert_eq!(calls[0].name, "get_weather");
    assert_eq!(calls[0].arguments, r#"{"city":"SF","n":2}"#);
}

/// Marker-wrapped grammars must hold back a partial `<tool_call>` suffix
/// across deltas. The holdback is implemented in `parser::tool`, so this is a
/// live regression test rather than an ignored probe.
#[test]
fn wrapped_tool_result_is_invariant_when_marker_is_split() {
    const WRAPPED: &str = concat!(
        "Let me check. <tool_call>\n",
        "{\"name\":\"get_weather\",\"arguments\":{\"city\":\"SF\"}}\n",
        "</tool_call> done",
    );
    let boundaries: Vec<usize> = WRAPPED.char_indices().map(|(index, _)| index).collect();
    let mut reference: Option<String> = None;
    for split in boundaries {
        let mut parser = megalm_rs::parser::create_tool_parser(
            "qwen3_xml",
            &[megalm_rs::parser::Tool::from_names(&["get_weather".into()])[0].clone()],
        )
        .unwrap();
        let mut output = megalm_rs::parser::ToolParserOutput::default();
        parser.parse_into(&WRAPPED[..split], &mut output).unwrap();
        parser.parse_into(&WRAPPED[split..], &mut output).unwrap();
        let text = output.normal_text();
        match &reference {
            None => reference = Some(text),
            Some(expected) => assert_eq!(text, *expected, "split at {split} diverged"),
        }
    }
}
