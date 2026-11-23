//! Throughput benchmarks for the gateway host-path parsers.
//!
//! Each iteration models one SSE delta: a short decoded chunk fed through the
//! parser's public `push` path. These numbers guard against regressions in the
//! hot loop (one cross-language call per token batch).

use criterion::{criterion_group, criterion_main, Criterion};
use megalm_rs::parser::{InklingOutput, InklingParser, OutputParser, ToolCallParser};
use std::hint::black_box;

const INKLING_DELTA: &str = concat!(
    "<|message_model|><|content_thinking|>planning step<|end_message|>",
    "<|message_model|><|content_text|>the answer<|end_message|>",
);

const TOOL_DELTA: &str = concat!(
    "calling <tool_call>\n",
    "{\"name\":\"get_weather\",\"arguments\":{\"city\":\"SF\",\"days\":3}}\n",
    "</tool_call>",
);

fn inkling(c: &mut Criterion) {
    c.bench_function("inkling/push_mixed_blocks", |b| {
        b.iter(|| {
            let mut parser = InklingParser::new("call");
            let mut output = InklingOutput::default();
            parser.push(black_box(INKLING_DELTA), &mut output);
            parser.finish(&mut output);
            black_box(output)
        })
    });
}

fn tool_call(c: &mut Criterion) {
    c.bench_function("tool_call/push_xml_wrapped", |b| {
        b.iter(|| {
            let mut parser = ToolCallParser::new("call", vec!["get_weather".into()]);
            let (visible, calls) = parser.push(black_box(TOOL_DELTA));
            let _ = parser.finish();
            black_box((visible, calls))
        })
    });
}

fn unified(c: &mut Criterion) {
    c.bench_function("output/push_with_thinking_and_tool", |b| {
        b.iter(|| {
            let mut parser = OutputParser::new(
                "call",
                vec![],
                vec!["get_weather".into()],
                Some("<think>".into()),
                Some("</think>".into()),
                false,
            );
            let result = parser.push(black_box(TOOL_DELTA));
            let _ = parser.finish();
            black_box(result)
        })
    });
}

criterion_group!(benches, inkling, tool_call, unified);
criterion_main!(benches);
