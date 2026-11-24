//! Cross-request tokenizer batching.
//!
//! Concurrent `encode` calls are coalesced into one `tokenizers::encode_batch`
//! invocation, executed on the blocking pool so a large prompt never stalls a
//! tokio worker thread.

use std::time::Duration;

use megalm_tokenizer::ModelTokenizer;
use tokio::sync::{mpsc, oneshot};
use tokio::time::Instant;

struct EncodeItem {
    text: String,
    reply: oneshot::Sender<Result<Vec<u32>, String>>,
}

#[derive(Clone)]
pub struct TokenizerBatcher {
    tx: mpsc::UnboundedSender<EncodeItem>,
}

impl TokenizerBatcher {
    pub fn new(
        tokenizer: &'static ModelTokenizer,
        coalesce_wait: Duration,
        max_batch: usize,
    ) -> Self {
        let (tx, rx) = mpsc::unbounded_channel::<EncodeItem>();
        tokio::spawn(batch_loop(
            tokenizer,
            rx,
            coalesce_wait,
            max_batch.max(1),
        ));
        Self { tx }
    }

    pub async fn encode(&self, text: String) -> Result<Vec<u32>, String> {
        let (tx, rx) = oneshot::channel();
        self.tx
            .send(EncodeItem { text, reply: tx })
            .map_err(|_| "tokenizer batcher stopped".to_string())?;
        rx.await.map_err(|_| "tokenizer batcher stopped".to_string())?
    }
}

async fn batch_loop(
    tokenizer: &'static ModelTokenizer,
    mut incoming: mpsc::UnboundedReceiver<EncodeItem>,
    coalesce_wait: Duration,
    max_batch: usize,
) {
    while let Some(first) = incoming.recv().await {
        let mut items = vec![first];
        let deadline = Instant::now() + coalesce_wait;
        while items.len() < max_batch {
            let remaining = deadline.saturating_duration_since(Instant::now());
            match tokio::time::timeout(remaining, incoming.recv()).await {
                Ok(Some(item)) => items.push(item),
                Ok(None) | Err(_) => break,
            }
        }

        let texts: Vec<String> = items.iter().map(|item| item.text.clone()).collect();
        let result = tokio::task::spawn_blocking(move || tokenizer.encode_batch(&texts)).await;
        match result {
            Ok(Ok(ids)) => {
                for (item, ids) in items.into_iter().zip(ids) {
                    let _ = item.reply.send(Ok(ids));
                }
            }
            Ok(Err(err)) => {
                let message = err.to_string();
                for item in items {
                    let _ = item.reply.send(Err(message.clone()));
                }
            }
            Err(err) => {
                let message = format!("tokenizer batch task failed: {err}");
                for item in items {
                    let _ = item.reply.send(Err(message.clone()));
                }
            }
        }
    }
    while let Some(item) = incoming.recv().await {
        let _ = item
            .reply
            .send(Err("tokenizer batcher stopped".into()));
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn coalesces_concurrent_encodes_matching_single_encode() {
        let path = std::env::var("MEGALM_TEST_MODEL")
            .or_else(|_| {
                std::env::var("HOME")
                    .map(|home| format!("{home}/projects/TensorPlay/megaLM/model"))
            })
            .map(std::path::PathBuf::from)
            .ok()
            .filter(|path| path.join("tokenizer.json").is_file());
        let Some(path) = path else {
            eprintln!("skipping: no local model at MEGALM_TEST_MODEL");
            return;
        };

        let tokenizer: &'static ModelTokenizer =
            Box::leak(Box::new(ModelTokenizer::new(&path).unwrap()));
        let batcher = TokenizerBatcher::new(tokenizer, Duration::from_millis(2), 32);
        let texts = vec![
            "Hello world".to_string(),
            "The quick brown fox".to_string(),
            "多模态 tokenizer 批处理".to_string(),
            "Another prompt for batching".to_string(),
        ];
        let mut tasks = Vec::new();
        for text in texts.clone() {
            let batcher = batcher.clone();
            tasks.push(tokio::spawn(async move { batcher.encode(text).await }));
        }
        let results = futures::future::join_all(tasks).await;
        for (index, result) in results.into_iter().enumerate() {
            let ids = result.unwrap().unwrap();
            assert_eq!(ids, tokenizer.encode(&texts[index]).unwrap());
        }
    }
}
