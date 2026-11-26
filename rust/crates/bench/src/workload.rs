//! Deterministic request sets shared by every engine in a run.

use std::fs;
use std::path::Path;

use anyhow::{anyhow, bail, Result};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use sha2::{Digest, Sha256};

/// Fixed, content-neutral prose.  No network or local files are needed to
/// build a long prompt, and every engine receives byte-identical text.
pub const PASSAGE: &str = "The history of computing is a history of abstraction. \
Each generation of engineers hides the previous generation's complexity behind \
a new interface, and each new interface makes the next layer of machinery \
possible. A compiler hides registers; an operating system hides memory; a \
virtual machine hides the instruction set. When the abstraction is good, the \
machine feels simpler than it is, and the programmer can think about the \
problem instead of the substrate. When the abstraction leaks, the programmer \
is forced back down the stack, and the illusion collapses.";

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct RequestSpec {
    pub request_id: String,
    pub prompt: String,
    pub output_tokens: u32,
    pub prompt_tokens_hint: u32,
    pub prompt_tokens_exact: bool,
}

#[derive(Clone, Debug)]
pub struct Workload {
    pub requests: Vec<RequestSpec>,
    pub description: Value,
}

impl Workload {
    /// Digest of the exact prompts and output budgets sent to each server.
    pub fn digest(&self) -> String {
        let payload: Vec<Value> = self
            .requests
            .iter()
            .map(|spec| {
                json!({
                    "id": spec.request_id,
                    "prompt": spec.prompt,
                    "output_tokens": spec.output_tokens,
                })
            })
            .collect();
        hex(&Sha256::digest(
            serde_json::to_vec(&payload).expect("request specs are serializable"),
        ))
    }
}

pub fn hex(bytes: &[u8]) -> String {
    let mut out = String::with_capacity(bytes.len() * 2);
    for byte in bytes {
        out.push_str(&format!("{byte:02x}"));
    }
    out
}

/// SplitMix64.  A tiny fixed PRNG keeps workloads reproducible across Rust
/// versions and avoids depending on a rand crate release for published numbers.
pub struct Seeded {
    state: u64,
}

impl Seeded {
    pub fn new(seed: u64) -> Self {
        Self {
            state: seed.wrapping_add(0x9E37_79B9_7F4A_7C15),
        }
    }

    pub fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    /// Uniform in (0, 1].
    pub fn next_f64(&mut self) -> f64 {
        ((self.next_u64() >> 11) as f64 + 1.0) / ((1u64 << 53) as f64 + 1.0)
    }

    pub fn shuffle<T>(&mut self, items: &mut [T]) {
        if items.len() < 2 {
            return;
        }
        for index in (1..items.len()).rev() {
            let other = (self.next_u64() % (index as u64 + 1)) as usize;
            items.swap(index, other);
        }
    }

    /// Box-Muller standard normal.
    pub fn normal(&mut self) -> f64 {
        let u = self.next_f64();
        let v = self.next_f64();
        (-2.0 * u.ln()).sqrt() * (std::f64::consts::TAU * v).cos()
    }

    /// Marsaglia-Tsang gamma variate with the given shape and scale 1.0.
    pub fn gamma(&mut self, shape: f64) -> f64 {
        if shape == 1.0 {
            return self.expovariate(1.0);
        }
        if shape >= 1.0 {
            loop {
                let normal = self.normal();
                let v = (1.0 + normal / (9.0 * shape).sqrt()).powi(3);
                if v <= 0.0 {
                    continue;
                }
                let u = self.next_f64();
                if u < 1.0 - 0.0331 * normal.powi(4)
                    || u.ln() < 0.5 * normal * normal + shape * (1.0 - v + v.ln())
                {
                    return shape * v;
                }
            }
        }
        loop {
            let u = self.next_f64();
            let v = self.next_f64();
            let x = u.powf(1.0 / shape);
            let y = v.powf(1.0 / (1.0 - shape));
            if x + y <= 1.0 && x + y > 0.0 {
                let e = -self.next_f64().ln();
                return e * x / (x + y);
            }
        }
    }

    /// Exponential delay for a Poisson process of the given rate.
    pub fn expovariate(&mut self, rate: f64) -> f64 {
        -self.next_f64().max(1e-12).ln() / rate
    }
}

fn tokenizer_from(path: Option<&Path>) -> Result<Option<tokenizers::Tokenizer>> {
    let Some(path) = path else {
        return Ok(None);
    };
    let candidate = if path.is_dir() {
        path.join("tokenizer.json")
    } else {
        path.to_path_buf()
    };
    if !candidate.is_file() {
        bail!("tokenizer file not found: {}", candidate.display());
    }
    let tokenizer = tokenizers::Tokenizer::from_file(&candidate)
        .map_err(|err| anyhow!("cannot load {}: {err}", candidate.display()))?;
    Ok(Some(tokenizer))
}

fn synthetic(
    count: usize,
    prompt_tokens: usize,
    output_tokens: u32,
    tokenizer_path: Option<&Path>,
) -> Result<(Vec<RequestSpec>, Value)> {
    let tokenizer = tokenizer_from(tokenizer_path)?;
    let (prompt, hint, exact, description) = match tokenizer {
        Some(tokenizer) => {
            let ids = tokenizer
                .encode(PASSAGE, false)
                .map_err(|e| anyhow!(e.to_string()))?;
            let ids = ids.get_ids();
            if ids.is_empty() {
                bail!("tokenizer produced no tokens for the synthetic passage");
            }
            let padded: Vec<u32> = ids.iter().copied().cycle().take(prompt_tokens).collect();
            let prompt = tokenizer
                .decode(&padded, false)
                .map_err(|e| anyhow!(e.to_string()))?;
            let actual = tokenizer
                .encode(prompt.as_str(), false)
                .map_err(|e| anyhow!(e.to_string()))?
                .get_ids()
                .len() as u32;
            (
                prompt,
                actual,
                true,
                json!({
                    "tokenizer": tokenizer_path.map(|p| p.display().to_string()),
                    "prompt_tokens": actual,
                    "prompt_tokens_exact": true,
                }),
            )
        }
        None => {
            let words: Vec<&str> = PASSAGE.split_whitespace().collect();
            let prompt = words
                .iter()
                .cycle()
                .take(prompt_tokens)
                .copied()
                .collect::<Vec<_>>()
                .join(" ");
            (
                prompt,
                prompt_tokens as u32,
                false,
                json!({
                    "tokenizer": null,
                    "prompt_tokens_target": prompt_tokens,
                    "prompt_tokens_exact": false,
                }),
            )
        }
    };

    let requests = (0..count)
        .map(|index| RequestSpec {
            request_id: format!("syn-{index:05}"),
            prompt: prompt.clone(),
            output_tokens,
            prompt_tokens_hint: hint,
            prompt_tokens_exact: exact,
        })
        .collect();
    Ok((requests, description))
}

fn record_prompt(record: &Value) -> Result<String> {
    if let Some(conversations) = record.get("conversations").and_then(Value::as_array) {
        let human: Vec<&str> = conversations
            .iter()
            .filter_map(|item| item.as_object())
            .filter(|item| {
                matches!(
                    item.get("from").and_then(Value::as_str),
                    Some("human" | "user")
                )
            })
            .filter_map(|item| item.get("value").and_then(Value::as_str))
            .collect();
        if human.is_empty() {
            bail!("ShareGPT record has no human message");
        }
        return Ok(human.join("\n"));
    }
    for key in ["prompt", "input", "text"] {
        if let Some(value) = record.get(key).and_then(Value::as_str) {
            if !value.is_empty() {
                return Ok(value.to_owned());
            }
        }
    }
    bail!("record has no prompt field (expected prompt/input/text/conversations)");
}

fn record_output(record: &Value, default: u32) -> u32 {
    for key in [
        "output_length",
        "max_tokens",
        "max_new_tokens",
        "output_tokens",
    ] {
        if let Some(value) = record.get(key).and_then(Value::as_u64) {
            if value > 0 && value <= u32::MAX as u64 {
                return value as u32;
            }
        }
    }
    if let Some(conversations) = record.get("conversations").and_then(Value::as_array) {
        for item in conversations.iter().rev().filter_map(Value::as_object) {
            if matches!(
                item.get("from").and_then(Value::as_str),
                Some("gpt" | "assistant")
            ) {
                if let Some(text) = item.get("value").and_then(Value::as_str) {
                    let words = text.split_whitespace().count();
                    if words > 0 {
                        return words.min(u32::MAX as usize) as u32;
                    }
                }
            }
        }
    }
    default
}

fn from_file(
    path: &Path,
    output_tokens: u32,
    count: usize,
    seed: u64,
) -> Result<(Vec<RequestSpec>, Value)> {
    let text =
        fs::read_to_string(path).map_err(|err| anyhow!("cannot read {}: {err}", path.display()))?;
    let mut records = Vec::new();
    for (line_number, line) in text.lines().enumerate() {
        if line.trim().is_empty() {
            continue;
        }
        let value: Value = serde_json::from_str(line).map_err(|err| {
            anyhow!(
                "{}:{}: invalid JSON: {err}",
                path.display(),
                line_number + 1
            )
        })?;
        if !value.is_object() {
            bail!(
                "{}:{}: expected a JSON object",
                path.display(),
                line_number + 1
            );
        }
        records.push(value);
    }
    if records.is_empty() {
        bail!("{} contains no records", path.display());
    }
    let mut rng = Seeded::new(seed);
    rng.shuffle(&mut records);
    records.truncate(count);
    let requests: Vec<RequestSpec> = records
        .iter()
        .enumerate()
        .map(|(index, record)| RequestSpec {
            request_id: format!("ds-{index:05}"),
            prompt: record_prompt(record).unwrap_or_default(),
            output_tokens: record_output(record, output_tokens),
            prompt_tokens_hint: 0,
            prompt_tokens_exact: false,
        })
        .filter(|spec| !spec.prompt.is_empty())
        .collect();
    if requests.is_empty() {
        bail!("{} has no usable prompts", path.display());
    }
    Ok((
        requests,
        json!({"path": path.display().to_string(), "records": records.len()}),
    ))
}

pub fn build(
    dataset: crate::cli::DatasetKind,
    dataset_path: Option<&Path>,
    count: usize,
    prompt_tokens: usize,
    output_tokens: u32,
    tokenizer: Option<&Path>,
    seed: u64,
) -> Result<Workload> {
    if count == 0 {
        bail!("--num-prompts must be positive");
    }
    if output_tokens == 0 {
        bail!("--max-tokens must be positive");
    }
    let (requests, mut description) = match dataset {
        crate::cli::DatasetKind::Synthetic => {
            if prompt_tokens == 0 {
                bail!("--prompt-tokens must be positive for synthetic data");
            }
            synthetic(count, prompt_tokens, output_tokens, tokenizer)?
        }
        crate::cli::DatasetKind::Jsonl | crate::cli::DatasetKind::Sharegpt => {
            let path = dataset_path.ok_or_else(|| anyhow!("--dataset-path is required"))?;
            from_file(path, output_tokens, count, seed)?
        }
    };
    let format = match dataset {
        crate::cli::DatasetKind::Synthetic => "synthetic",
        crate::cli::DatasetKind::Jsonl => "jsonl",
        crate::cli::DatasetKind::Sharegpt => "sharegpt",
    };
    description["dataset"] = json!(format);
    description["num_prompts"] = json!(requests.len());
    description["seed"] = json!(seed);
    description["max_tokens"] = json!(output_tokens);
    Ok(Workload {
        requests,
        description,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rng_is_reproducible() {
        let mut a = Seeded::new(7);
        let mut b = Seeded::new(7);
        let draw = |rng: &mut Seeded| (0..32).map(|_| rng.next_u64()).collect::<Vec<_>>();
        assert_eq!(draw(&mut a), draw(&mut b));
        assert_ne!(draw(&mut a), draw(&mut Seeded::new(8)));
    }

    #[test]
    fn workload_is_byte_identical_for_the_same_seed() {
        let build = || {
            build(
                crate::cli::DatasetKind::Synthetic,
                None,
                4,
                64,
                16,
                None,
                42,
            )
            .unwrap()
        };
        assert_eq!(build().digest(), build().digest());
        assert_eq!(build().requests[0].prompt, build().requests[3].prompt);
    }

    #[test]
    fn sharegpt_records_are_loaded() {
        let dir = std::env::temp_dir().join(format!("megalm-bench-test-{}", std::process::id()));
        fs::create_dir_all(&dir).unwrap();
        let path = dir.join("data.jsonl");
        fs::write(
            &path,
            "{\"conversations\":[{\"from\":\"human\",\"value\":\"hello\"},{\"from\":\"gpt\",\"value\":\"world\"}],\"output_length\":9}\n",
        )
        .unwrap();
        let workload = build(
            crate::cli::DatasetKind::Sharegpt,
            Some(&path),
            1,
            0,
            32,
            None,
            0,
        )
        .unwrap();
        assert_eq!(workload.requests[0].prompt, "hello");
        assert_eq!(workload.requests[0].output_tokens, 9);
        fs::remove_dir_all(dir).ok();
    }
}
