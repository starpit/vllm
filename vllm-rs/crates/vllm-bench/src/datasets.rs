// SPDX-License-Identifier: Apache-2.0
// Copyright contributors to the vLLM project

//! Dataset loading for `vllm bench` — supports ShareGPT and random prompt generation.
//!
//! Matches Python vLLM's `ShareGPTDataset.sample()` methodology.

use std::path::Path;

use anyhow::{Context, Result};
use tokenizers::Tokenizer;

/// A single benchmark request with pre-computed token lengths.
#[derive(Debug, Clone)]
#[allow(dead_code)]
pub struct SampleRequest {
    /// The prompt text to send.
    pub prompt: String,
    /// Number of prompt tokens (after tokenization).
    pub prompt_len: usize,
    /// Expected output length (tokens).
    pub expected_output_len: usize,
}

/// Load a ShareGPT-format dataset, matching Python's `ShareGPTDataset.sample()`.
///
/// JSON format: `[{"conversations": [{"from": "human", "value": "..."}, {"from": "gpt", "value": "..."}]}]`
///
/// For each conversation:
/// - First "human" turn → prompt
/// - First "gpt" turn → expected output (used for length estimation)
/// - Filter: `prompt_len >= 4`, `output_len >= 4`, `prompt_len + output_len <= max_model_len`
/// - Shuffle with seed, take `num_prompts`
pub fn load_sharegpt(
    path: &Path,
    tokenizer: &Tokenizer,
    num_prompts: usize,
    max_model_len: Option<usize>,
    seed: u64,
) -> Result<Vec<SampleRequest>> {
    let data = std::fs::read_to_string(path)
        .with_context(|| format!("Failed to read ShareGPT dataset from {}", path.display()))?;
    let entries: Vec<serde_json::Value> =
        serde_json::from_str(&data).context("Failed to parse ShareGPT JSON")?;

    let max_len = max_model_len.unwrap_or(usize::MAX);

    let mut samples: Vec<SampleRequest> = Vec::new();

    for entry in &entries {
        let conversations = match entry.get("conversations").and_then(|c| c.as_array()) {
            Some(c) => c,
            None => continue,
        };

        // Find first human turn and first GPT turn.
        let human_text = conversations
            .iter()
            .find(|c| c.get("from").and_then(|f| f.as_str()) == Some("human"))
            .and_then(|c| c.get("value").and_then(|v| v.as_str()));
        let gpt_text = conversations
            .iter()
            .find(|c| c.get("from").and_then(|f| f.as_str()) == Some("gpt"))
            .and_then(|c| c.get("value").and_then(|v| v.as_str()));

        let (prompt, output) = match (human_text, gpt_text) {
            (Some(h), Some(g)) => (h, g),
            _ => continue,
        };

        // Tokenize to get lengths.
        let prompt_encoding = tokenizer
            .encode(prompt, false)
            .map_err(|e| anyhow::anyhow!("Tokenization failed: {e}"))?;
        let output_encoding = tokenizer
            .encode(output, false)
            .map_err(|e| anyhow::anyhow!("Tokenization failed: {e}"))?;

        let prompt_len = prompt_encoding.get_ids().len();
        let output_len = output_encoding.get_ids().len();

        // Filter: matching Python's min thresholds.
        if prompt_len < 4 || output_len < 4 {
            continue;
        }
        if prompt_len + output_len > max_len {
            continue;
        }

        samples.push(SampleRequest {
            prompt: prompt.to_string(),
            prompt_len,
            expected_output_len: output_len,
        });
    }

    anyhow::ensure!(
        !samples.is_empty(),
        "No valid samples found in ShareGPT dataset at {}. \
         Check that the file contains conversations with 'human' and 'gpt' turns.",
        path.display()
    );

    // Shuffle deterministically with seed, matching Python's random.seed(seed) + random.shuffle().
    shuffle_with_seed(&mut samples, seed);

    // Take up to num_prompts.
    samples.truncate(num_prompts);

    Ok(samples)
}

/// Fisher-Yates shuffle with a deterministic xorshift64 PRNG.
fn shuffle_with_seed<T>(data: &mut [T], seed: u64) {
    let mut state = seed.max(1); // Avoid zero state.
    for i in (1..data.len()).rev() {
        // xorshift64
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        let j = (state as usize) % (i + 1);
        data.swap(i, j);
    }
}
