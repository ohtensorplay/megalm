//! Compare the Rust frontend against the Python frontend on a local model.

use megalm_tokenizer::ModelTokenizer;
use serde_json::json;

fn model_path() -> Option<std::path::PathBuf> {
    std::env::var("MEGALM_TEST_MODEL")
        .or_else(|_| {
            std::env::var("HOME").map(|home| format!("{home}/projects/TensorPlay/megaLM/model"))
        })
        .ok()
        .map(std::path::PathBuf::from)
        .filter(|path| path.join("tokenizer.json").is_file())
}

#[test]
fn rendered_prompts_match_python_frontend() {
    let Some(path) = model_path() else {
        eprintln!("skipping: no local model at MEGALM_TEST_MODEL");
        return;
    };
    let tokenizer = ModelTokenizer::new(&path).expect("load tokenizer");
    let golden: Vec<String> =
        serde_json::from_str(include_str!("fixtures/inkling_chat.json")).expect("fixture");
    let tools = vec![json!({
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "d",
            "parameters": {"type": "object", "properties": {}}
        }
    })];
    let cases = [
        (vec![json!({"role": "user", "content": "Hello!"})], None),
        (
            vec![
                json!({"role": "system", "content": "You are helpful."}),
                json!({"role": "user", "content": "Hi"}),
            ],
            None,
        ),
        (
            vec![
                json!({"role": "user", "content": "first"}),
                json!({"role": "assistant", "content": "answer"}),
                json!({"role": "user", "content": "next"}),
            ],
            None,
        ),
        (
            vec![json!({"role": "user", "content": "Hello!"})],
            Some(tools.as_slice()),
        ),
        (vec![json!({"role": "user", "content": "Hello!"})], None),
    ];
    for (expected, (messages, tools)) in golden.iter().zip(cases) {
        let rendered = tokenizer
            .render_chat(&messages, tools, false)
            .expect("render chat");
        assert_eq!(&rendered, expected);
    }
}

#[test]
fn template_kwargs_reach_the_chat_template() {
    let Some(path) = model_path() else {
        eprintln!("skipping: no local model at MEGALM_TEST_MODEL");
        return;
    };
    let tokenizer = ModelTokenizer::new(&path).expect("load tokenizer");
    let messages = vec![json!({"role": "user", "content": "Hello!"})];
    let rendered = tokenizer
        .render_chat_with(
            &messages,
            None,
            false,
            Some(&json!({"reasoning_effort": 0.4})),
        )
        .expect("render chat");
    assert!(rendered.contains("Thinking effort level: 0.4"));
}

#[test]
fn renders_and_encodes_local_model() {
    let Some(path) = model_path() else {
        eprintln!("skipping: no local model at MEGALM_TEST_MODEL");
        return;
    };
    let tokenizer = ModelTokenizer::new(&path).expect("load tokenizer");
    let messages = vec![
        json!({"role": "system", "content": "You are helpful."}),
        json!({"role": "user", "content": "Hello!"}),
    ];
    let rendered = tokenizer
        .render_chat(&messages, None, false)
        .expect("render chat");
    assert!(!rendered.is_empty(), "rendered prompt is empty");
    let ids = tokenizer.encode(&rendered).expect("encode");
    assert!(!ids.is_empty(), "encoded prompt is empty");
    assert_eq!(
        ids.len(),
        tokenizer.encode_batch(&[rendered]).unwrap()[0].len()
    );

    let mut stream = tokenizer.decode_stream(false);
    let text = stream.push(&ids);
    assert!(!text.is_empty(), "incremental decode produced nothing");
    assert!(tokenizer.vocab_size() > 0);
}
