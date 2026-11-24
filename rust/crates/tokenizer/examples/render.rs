use megalm_tokenizer::ModelTokenizer;
use serde_json::json;

fn main() {
    let path = std::env::var("MEGALM_TEST_MODEL").unwrap_or_else(|_| {
        format!(
            "{}/projects/TensorPlay/megaLM/model",
            std::env::var("HOME").unwrap()
        )
    });
    let tokenizer = ModelTokenizer::new(&path).unwrap();
    let messages = vec![
        json!({"role": "system", "content": "You are helpful."}),
        json!({"role": "user", "content": "Hello!"}),
    ];
    let rendered = tokenizer.render_chat(&messages, None, false).unwrap();
    let ids = tokenizer.encode(&rendered).unwrap();
    println!("RENDER_BYTES {}", rendered.len());
    println!("IDS {}", ids.iter().take(24).map(|id| id.to_string()).collect::<Vec<_>>().join(","));
    println!("IDS_LEN {}", ids.len());
    println!("GRAMMAR {}", tokenizer.output_grammar());
    println!("TOOL {:?}", tokenizer.tool_call_parser());
}
