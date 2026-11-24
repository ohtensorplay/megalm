//! Compatibility with the Jinja dialect used by HF `apply_chat_template`.

use std::path::PathBuf;
use std::sync::atomic::{AtomicUsize, Ordering};

use megalm_renderer::{ChatRenderer, RenderContext};
use serde_json::{json, Value};

fn context(messages: Vec<Value>, tools: Option<Vec<Value>>) -> RenderContext {
    RenderContext {
        messages,
        tools,
        open_thinking: false,
        bos_token: String::new(),
        eos_token: String::new(),
        pad_token: String::new(),
        unk_token: String::new(),
        extras: None,
    }
}

fn render_source(source: &str) -> String {
    ChatRenderer::from_source(source.into())
        .unwrap()
        .render(&context(
            vec![json!({"role": "user", "content": "x"})],
            None,
        ))
        .unwrap()
}

#[test]
fn tojson_defaults_match_python_json_dumps() {
    let renderer = ChatRenderer::from_source("{{ value | tojson }}".into()).unwrap();
    let value = json!({"b": 1, "a": {"y": true, "x": null}});
    let rendered = renderer
        .render(&RenderContext {
            extras: Some(json!({"value": value})),
            ..context(vec![json!({"role": "user", "content": "x"})], None)
        })
        .unwrap();
    assert_eq!(rendered, r#"{"b": 1, "a": {"y": true, "x": null}}"#);
}

#[test]
fn tojson_supports_hf_options() {
    let cases = [
        (
            "{{ value | tojson(sort_keys=true) }}",
            r#"{"a": "é", "b": 1}"#,
        ),
        (
            r#"{{ value | tojson(separators=(",", ":")) }}"#,
            r#"{"b":1,"a":"é"}"#,
        ),
        (
            "{{ value | tojson(ensure_ascii=true) }}",
            r#"{"b": 1, "a": "\u00e9"}"#,
        ),
        (
            "{{ value | tojson(indent=2, sort_keys=true) }}",
            "{\n  \"a\": \"é\",\n  \"b\": 1\n}",
        ),
        (
            r#"{{ value | tojson(indent="..", sort_keys=true) }}"#,
            "{\n..\"a\": \"é\",\n..\"b\": 1\n}",
        ),
    ];
    let value = json!({"b": 1, "a": "é"});
    for (source, expected) in cases {
        let rendered = ChatRenderer::from_source(source.into())
            .unwrap()
            .render(&RenderContext {
                extras: Some(json!({"value": value.clone()})),
                ..context(vec![json!({"role": "user", "content": "x"})], None)
            })
            .unwrap();
        assert_eq!(rendered, expected, "template: {source}");
    }
}

#[test]
fn hf_whitespace_flags_drop_trailing_block_newline() {
    let source = "{% if add_generation_prompt %}\nhello\n{% endif %}\n";
    assert_eq!(render_source(source), "hello\n");
}

#[test]
fn python_string_methods_are_available() {
    let renderer = ChatRenderer::from_source(
        "{{ c.startswith('he') }} {{ c.endswith('lo') }} {{ c.strip('h') }} \
         {{ c.lstrip('h') }} {{ c.rstrip('o') }} {{ c.split('e') | join('-') }}"
            .into(),
    )
    .unwrap();
    let rendered = renderer
        .render(&RenderContext {
            extras: Some(json!({"c": "hello"})),
            ..context(vec![json!({"role": "user", "content": "x"})], None)
        })
        .unwrap();
    assert_eq!(rendered, "True True ello ello hell h-llo");
}

#[test]
fn generation_tags_with_whitespace_control_are_stripped() {
    assert_eq!(
        render_source("a{%- generation -%}B{%  endgeneration  %}C"),
        "aBC"
    );
}

#[test]
fn strftime_now_global_is_available() {
    let rendered = render_source("{{ strftime_now('%Y') }}");
    assert_eq!(rendered.len(), 4);
    assert!(rendered.chars().all(|c| c.is_ascii_digit()));
}

#[test]
fn documents_and_extra_template_variables_are_defined() {
    let source = "{% if documents is defined %}D{% endif %}:{{ reasoning_effort }}";
    let renderer = ChatRenderer::from_source(source.into()).unwrap();
    let rendered = renderer
        .render(&RenderContext {
            extras: Some(json!({"reasoning_effort": 0.4})),
            ..context(vec![json!({"role": "user", "content": "x"})], None)
        })
        .unwrap();
    assert_eq!(rendered, "D:0.4");
}

#[test]
fn reserved_variables_cannot_be_overridden_by_extras() {
    let renderer =
        ChatRenderer::from_source("{{ messages|length }}:{{ add_generation_prompt }}".into())
            .unwrap();
    let rendered = renderer
        .render(&RenderContext {
            extras: Some(json!({"messages": [], "add_generation_prompt": false})),
            ..context(vec![json!({"role": "user", "content": "x"})], None)
        })
        .unwrap();
    assert_eq!(rendered, "1:True");
}

struct TempDir(PathBuf);

impl TempDir {
    fn new(tag: &str) -> Self {
        static NEXT: AtomicUsize = AtomicUsize::new(0);
        let id = NEXT.fetch_add(1, Ordering::Relaxed);
        let path =
            std::env::temp_dir().join(format!("megalm-renderer-{tag}-{}-{id}", std::process::id()));
        std::fs::create_dir_all(&path).unwrap();
        Self(path)
    }

    fn write(&self, name: &str, content: &str) {
        std::fs::write(self.0.join(name), content).unwrap();
    }
}

impl Drop for TempDir {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

#[test]
fn named_templates_from_config_select_tool_use_when_tools_are_present() {
    let dir = TempDir::new("named");
    dir.write(
        "tokenizer_config.json",
        &json!({
            "chat_template": {
                "default": "DEFAULT",
                "tool_use": "TOOLS:{{ tools|length }}"
            }
        })
        .to_string(),
    );
    let renderer = ChatRenderer::new(&dir.0).unwrap();
    let message = || vec![json!({"role": "user", "content": "x"})];
    assert_eq!(
        renderer.render(&context(message(), None)).unwrap(),
        "DEFAULT"
    );
    assert_eq!(
        renderer
            .render(&context(message(), Some(vec![json!({"type": "function"})])))
            .unwrap(),
        "TOOLS:1"
    );
}

#[test]
fn chat_template_directory_is_loaded() {
    let dir = TempDir::new("dir");
    let templates = dir.0.join("chat_templates");
    std::fs::create_dir(&templates).unwrap();
    std::fs::write(templates.join("default.jinja"), "{{ messages|length }}").unwrap();
    let renderer = ChatRenderer::new(&dir.0).unwrap();
    assert_eq!(
        renderer
            .render(&context(
                vec![json!({"role": "user", "content": "x"})],
                None
            ))
            .unwrap(),
        "1"
    );
}
