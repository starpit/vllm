// SPDX-License-Identifier: Apache-2.0
// Copyright contributors to the vLLM project

//! Dataset loading for `vllm bench` — supports ShareGPT, random prompt generation,
//! and RAG datasets (HotpotQA, 2WikiMultihopQA, MuSiQue, MS MARCO).

use std::io::BufRead;
use std::path::Path;

use anyhow::{Context, Result};
use clap::ValueEnum;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use tokenizers::Tokenizer;
use tokenizers::tokenizer::PostProcessor;

// ---------------------------------------------------------------------------
// RAG dataset abstraction
// ---------------------------------------------------------------------------

/// A single RAG sample: question + acceptable answers + document fragments.
#[derive(Debug, Clone)]
pub struct RagSample {
    /// The question to answer.
    pub question: String,
    /// All acceptable answers (for accuracy scoring).
    pub answers: Vec<String>,
    /// Document fragments as (label, text) pairs.
    pub documents: Vec<(String, String)>,
}

/// Available RAG datasets for `vllm bench ragindex`.
#[derive(Debug, Clone, Copy, ValueEnum)]
pub enum RagDataset {
    /// HotpotQA distractor validation set (10 Wikipedia paragraphs per query).
    Hotpotqa,
    /// 2WikiMultihopQA dev set (multi-hop reasoning over Wikipedia).
    Multihop,
    /// MuSiQue validation set (2-4 hop multi-hop QA, 20 paragraphs per query).
    Musique,
    /// MS MARCO v2.1 validation set (~10 Bing search passages per query).
    Msmarco,
}

impl std::fmt::Display for RagDataset {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Hotpotqa => write!(f, "hotpotqa"),
            Self::Multihop => write!(f, "multihop"),
            Self::Musique => write!(f, "musique"),
            Self::Msmarco => write!(f, "msmarco"),
        }
    }
}

/// Fetch a RAG dataset and convert to the common `RagSample` format.
pub fn fetch_rag_dataset(which: RagDataset, num_queries: usize) -> Result<Vec<RagSample>> {
    match which {
        RagDataset::Hotpotqa => fetch_hotpotqa(num_queries),
        RagDataset::Multihop => fetch_multihop(num_queries),
        RagDataset::Musique => fetch_musique(num_queries),
        RagDataset::Msmarco => fetch_msmarco(num_queries),
    }
}

// ---------------------------------------------------------------------------
// Shared helpers
// ---------------------------------------------------------------------------

fn download_parquet_as_json(
    cache_dir_name: &str,
    cache_filename: &str,
    parquet_filename: &str,
    url: &str,
    label: &str,
) -> Result<Vec<serde_json::Value>> {
    let cache_dir = dirs::cache_dir()
        .ok_or_else(|| anyhow::anyhow!("Could not determine cache directory"))?
        .join("vllm-bench")
        .join(cache_dir_name);
    std::fs::create_dir_all(&cache_dir)?;
    let cache_file = cache_dir.join(cache_filename);

    if cache_file.exists() {
        let data = std::fs::read_to_string(&cache_file)?;
        return Ok(serde_json::from_str(&data)?);
    }

    eprintln!("Downloading {label}...");
    let client = reqwest::blocking::Client::builder()
        .timeout(std::time::Duration::from_secs(300))
        .build()?;
    let parquet_path = cache_dir.join(parquet_filename);
    let response = client.get(url).header("User-Agent", "vllm-bench").send()?;
    let bytes = response.bytes()?;
    std::fs::write(&parquet_path, &bytes)?;

    let records = parquet_to_json_records(&parquet_path)?;
    eprintln!("Caching {} records as JSON...", records.len());
    let json_str = serde_json::to_string(&records)?;
    std::fs::write(&cache_file, json_str.as_bytes())?;
    Ok(records)
}

/// Read a parquet file and return rows as JSON values.
fn parquet_to_json_records(parquet_path: &std::path::Path) -> Result<Vec<serde_json::Value>> {
    use arrow::json::writer::{JsonArray, Writer};
    use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;

    eprintln!("Reading parquet...");
    let file = std::fs::File::open(parquet_path)?;
    let builder = ParquetRecordBatchReaderBuilder::try_new(file)?;
    let reader = builder.build()?;

    let batches: Vec<_> = reader.collect::<std::result::Result<Vec<_>, _>>()?;
    let batch_refs: Vec<&_> = batches.iter().collect();

    let mut buf = Vec::new();
    let mut writer = Writer::<_, JsonArray>::new(&mut buf);
    writer.write_batches(&batch_refs)?;
    writer.finish()?;
    drop(writer);

    let records: Vec<serde_json::Value> = serde_json::from_slice(&buf)?;
    eprintln!("Read {} records.", records.len());
    Ok(records)
}

// ---------------------------------------------------------------------------
// HotpotQA
// ---------------------------------------------------------------------------

fn fetch_hotpotqa(num_queries: usize) -> Result<Vec<RagSample>> {
    let raw = download_parquet_as_json(
        "hotpotqa",
        "validation.json",
        "validation.parquet",
        "https://huggingface.co/datasets/hotpotqa/hotpot_qa/resolve/main/distractor/validation-00000-of-00001.parquet",
        "HotpotQA distractor validation set",
    )?;

    let mut samples = Vec::new();
    for record in raw.iter().take(num_queries) {
        let question = record["question"].as_str().unwrap_or("").to_string();
        let answer = record["answer"].as_str().unwrap_or("").to_string();

        let context = &record["context"];
        let titles = context["title"].as_array().cloned().unwrap_or_default();
        let sentences = context["sentences"].as_array().cloned().unwrap_or_default();

        if question.is_empty() || answer.is_empty() || titles.is_empty() {
            continue;
        }

        let documents: Vec<(String, String)> = titles
            .iter()
            .zip(sentences.iter())
            .map(|(title_val, sents_val)| {
                let title = title_val.as_str().unwrap_or("").to_string();
                let text = sents_val
                    .as_array()
                    .map(|arr| {
                        arr.iter()
                            .filter_map(|s| s.as_str())
                            .collect::<Vec<_>>()
                            .join(" ")
                    })
                    .unwrap_or_default();
                (title, text)
            })
            .collect();

        samples.push(RagSample {
            question,
            answers: vec![answer],
            documents,
        });
    }
    Ok(samples)
}

// ---------------------------------------------------------------------------
// 2WikiMultihopQA
// ---------------------------------------------------------------------------

fn fetch_multihop(num_queries: usize) -> Result<Vec<RagSample>> {
    let raw = download_parquet_as_json(
        "multihop",
        "dev.json",
        "dev.parquet",
        "https://huggingface.co/datasets/xanhho/2WikiMultihopQA/resolve/main/dev.parquet",
        "2WikiMultihopQA dev set",
    )?;

    let mut samples = Vec::new();
    for record in raw.iter().take(num_queries) {
        let question = record["question"].as_str().unwrap_or("").to_string();
        let answer = record["answer"].as_str().unwrap_or("").to_string();

        let context_str = record["context"].as_str().unwrap_or("[]");
        let context: Vec<(String, Vec<String>)> =
            serde_json::from_str(context_str).unwrap_or_default();

        if question.is_empty() || answer.is_empty() || context.is_empty() {
            continue;
        }

        let documents: Vec<(String, String)> = context
            .into_iter()
            .map(|(title, sents)| (title, sents.join(" ")))
            .collect();

        samples.push(RagSample {
            question,
            answers: vec![answer],
            documents,
        });
    }
    Ok(samples)
}

// ---------------------------------------------------------------------------
// MuSiQue
// ---------------------------------------------------------------------------

fn fetch_musique(num_queries: usize) -> Result<Vec<RagSample>> {
    let cache_dir = dirs::cache_dir()
        .ok_or_else(|| anyhow::anyhow!("Could not determine cache directory"))?
        .join("vllm-bench")
        .join("musique");
    std::fs::create_dir_all(&cache_dir)?;
    let cache_file = cache_dir.join("validation.jsonl");

    if !cache_file.exists() {
        eprintln!("Downloading MuSiQue validation set...");
        let url = "https://huggingface.co/datasets/dgslibisey/MuSiQue/resolve/main/musique_ans_v1.0_dev.jsonl";
        let client = reqwest::blocking::Client::builder()
            .timeout(std::time::Duration::from_secs(300))
            .build()?;
        let response = client.get(url).header("User-Agent", "vllm-bench").send()?;
        let bytes = response.bytes()?;
        std::fs::write(&cache_file, &bytes)?;
    }

    let file = std::fs::File::open(&cache_file)?;
    let reader = std::io::BufReader::new(file);
    let mut samples = Vec::new();

    for line in reader.lines() {
        let line = line?;
        if line.trim().is_empty() {
            continue;
        }
        let record: serde_json::Value = serde_json::from_str(&line)?;

        let question = record["question"].as_str().unwrap_or("").to_string();
        let answer = record["answer"].as_str().unwrap_or("").to_string();

        let answer_aliases: Vec<String> = record["answer_aliases"]
            .as_array()
            .map(|arr| {
                arr.iter()
                    .filter_map(|v| v.as_str().map(|s| s.to_string()))
                    .collect()
            })
            .unwrap_or_default();

        let paragraphs_raw = record["paragraphs"].as_array().cloned().unwrap_or_default();
        if question.is_empty() || answer.is_empty() || paragraphs_raw.is_empty() {
            continue;
        }

        let documents: Vec<(String, String)> = paragraphs_raw
            .into_iter()
            .map(|p| {
                let title = p["title"].as_str().unwrap_or("").to_string();
                let text = p["paragraph_text"].as_str().unwrap_or("").to_string();
                (title, text)
            })
            .collect();

        let mut answers = vec![answer];
        answers.extend(answer_aliases);

        samples.push(RagSample {
            question,
            answers,
            documents,
        });

        if samples.len() >= num_queries {
            break;
        }
    }
    Ok(samples)
}

// ---------------------------------------------------------------------------
// MS MARCO
// ---------------------------------------------------------------------------

fn fetch_msmarco(num_queries: usize) -> Result<Vec<RagSample>> {
    let raw = download_parquet_as_json(
        "msmarco",
        "validation.json",
        "validation.parquet",
        "https://huggingface.co/datasets/microsoft/ms_marco/resolve/main/v2.1/validation-00000-of-00001.parquet",
        "MS MARCO v2.1 validation set",
    )?;

    let mut samples = Vec::new();
    for record in &raw {
        let question = record["query"].as_str().unwrap_or("").to_string();

        let mut answers: Vec<String> = Vec::new();
        if let Some(arr) = record["answers"].as_array() {
            for a in arr {
                if let Some(s) = a
                    .as_str()
                    .filter(|s| !s.is_empty() && *s != "No Answer Present.")
                {
                    answers.push(s.to_string());
                }
            }
        }
        if let Some(arr) = record["wellFormedAnswers"].as_array() {
            for a in arr {
                if let Some(s) = a.as_str().filter(|s| !s.is_empty() && *s != "[]") {
                    answers.push(s.to_string());
                }
            }
        }

        if question.is_empty() || answers.is_empty() {
            continue;
        }

        let passages_obj = &record["passages"];
        let texts = passages_obj["passage_text"]
            .as_array()
            .cloned()
            .unwrap_or_default();
        let urls = passages_obj["url"].as_array().cloned().unwrap_or_default();

        if texts.is_empty() {
            continue;
        }

        let documents: Vec<(String, String)> = urls
            .iter()
            .zip(texts.iter())
            .map(|(url_val, text_val)| {
                let url = url_val.as_str().unwrap_or("").to_string();
                let text = text_val.as_str().unwrap_or("").to_string();
                (url, text)
            })
            .collect();

        samples.push(RagSample {
            question,
            answers,
            documents,
        });

        if samples.len() >= num_queries {
            break;
        }
    }
    Ok(samples)
}

// ---------------------------------------------------------------------------
// Permutation generation (reusable)
// ---------------------------------------------------------------------------

/// Generate permutations of `0..n`, up to `max_perms`.
/// Uses Heap's algorithm for small n, random sampling for large.
pub fn permutations(n: usize, max_perms: usize) -> Vec<Vec<usize>> {
    if n <= 1 {
        return vec![(0..n).collect()];
    }
    let total: usize = (1..=n).product();
    if total <= max_perms {
        let mut result = Vec::with_capacity(total);
        let mut a: Vec<usize> = (0..n).collect();
        let mut c = vec![0usize; n];
        result.push(a.clone());
        let mut i = 0;
        while i < n {
            if c[i] < i {
                if i % 2 == 0 {
                    a.swap(0, i);
                } else {
                    a.swap(c[i], i);
                }
                result.push(a.clone());
                c[i] += 1;
                i = 0;
            } else {
                c[i] = 0;
                i += 1;
            }
        }
        result
    } else {
        use rand::seq::SliceRandom;
        let mut rng = rand::thread_rng();
        let mut result = Vec::with_capacity(max_perms);
        result.push((0..n).collect());
        result.push((0..n).rev().collect());
        while result.len() < max_perms {
            let mut perm: Vec<usize> = (0..n).collect();
            perm.shuffle(&mut rng);
            if !result.contains(&perm) {
                result.push(perm);
            }
        }
        result
    }
}

// ---------------------------------------------------------------------------
// Accuracy evaluation (reusable)
// ---------------------------------------------------------------------------

/// Evaluate response against any acceptable answer.
/// Returns 1.0 if any answer is a substring match or token F1 >= 0.5.
pub fn evaluate_accuracy(response: &str, answers: &[String]) -> f64 {
    let resp_lower = response.to_lowercase();
    for ans in answers {
        let ans_lower = ans.to_lowercase();
        if resp_lower.contains(&ans_lower) {
            return 1.0;
        }
        let f1 = compute_token_f1(&ans_lower, &resp_lower);
        if f1 >= 0.5 {
            return 1.0;
        }
    }
    0.0
}

/// Compute best token F1 across all acceptable answers.
pub fn best_token_f1(answers: &[String], actual: &str) -> f64 {
    answers
        .iter()
        .map(|ans| compute_token_f1(&ans.to_lowercase(), &actual.to_lowercase()))
        .fold(0.0_f64, f64::max)
}

fn normalize_tokens(text: &str) -> Vec<String> {
    text.split(|c: char| !c.is_alphanumeric())
        .filter(|s| !s.is_empty())
        .map(|s| s.to_string())
        .collect()
}

fn compute_token_f1(expected: &str, actual: &str) -> f64 {
    let et = normalize_tokens(expected);
    let at = normalize_tokens(actual);
    if et.is_empty() && at.is_empty() {
        return 1.0;
    }
    if et.is_empty() || at.is_empty() {
        return 0.0;
    }
    let common: usize = et.iter().filter(|t| at.contains(t)).count();
    if common == 0 {
        return 0.0;
    }
    let p = common as f64 / at.len() as f64;
    let r = common as f64 / et.len() as f64;
    2.0 * p * r / (p + r)
}

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

    // Shuffle deterministically with StdRng (Fisher-Yates, same algorithm as Python's
    // random.shuffle — note: PRNG differs from Python's MT19937 so order won't match
    // bit-for-bit at the same seed, but the algorithm is correct).
    shuffle_with_seed(&mut samples, seed);

    // Take up to num_prompts.
    samples.truncate(num_prompts);

    Ok(samples)
}

/// Fisher-Yates shuffle with `StdRng` (seeded), matching Python's `random.shuffle` algorithm.
fn shuffle_with_seed<T>(data: &mut [T], seed: u64) {
    let mut rng = StdRng::seed_from_u64(seed);
    for i in (1..data.len()).rev() {
        let j = rng.gen_range(0..=i);
        data.swap(i, j);
    }
}

/// Generate random prompts matching Python's `RandomDataset.sample()`.
///
/// Algorithm matches Python exactly:
/// 1. Seed `StdRng` from `seed` (Python uses `numpy.default_rng(seed)` — PCG64,
///    which differs bit-for-bit but we follow the same call ordering).
/// 2. Sample `input_lens`, `output_lens`, `offsets` in that order (consuming the RNG
///    in the same sequence as Python's `get_sampling_params()`).
/// 3. For each request `i`:
///    - `inner = allowed[(offset[i] + i + j) % len(allowed)]` for `j` in `0..input_len`
///    - Decode → re-encode with up to 10 retries (`gen_prompt_decode_to_target_len`):
///      * if len < target: pad with random tokens from full vocab `[0, vocab_size)` (same as Python)
///      * if len > target: truncate
pub fn generate_random(
    tokenizer: &Tokenizer,
    num_requests: usize,
    input_len: usize,
    output_len: usize,
    range_ratio: f64,
    prefix_len: usize,
    seed: u64,
) -> Result<Vec<SampleRequest>> {
    anyhow::ensure!(
        (0.0..1.0).contains(&range_ratio),
        "range_ratio must be in [0, 1), got {range_ratio}"
    );

    // Build allowed tokens (exclude special tokens), matching Python.
    let vocab_size = tokenizer.get_vocab_size(true);
    let special_ids: std::collections::HashSet<u32> = tokenizer
        .get_added_vocabulary()
        .get_added_tokens_decoder()
        .iter()
        .filter(|(_, t)| t.special)
        .map(|(id, _)| *id)
        .collect();
    let allowed_tokens: Vec<u32> = (0..vocab_size as u32)
        .filter(|id| !special_ids.contains(id))
        .collect();
    let num_allowed = allowed_tokens.len();

    anyhow::ensure!(
        num_allowed > 0,
        "No non-special tokens found in tokenizer vocabulary"
    );

    let mut rng = StdRng::seed_from_u64(seed);

    // Matching Python's `get_sampling_params()`:
    //   real_input_len = max(0, input_len - num_special_tokens_to_add())
    // Python's `num_special_tokens_to_add()` queries the tokenizer's post-processor
    // for how many special tokens it prepends/appends to a single (non-pair) sequence.
    let num_special = tokenizer
        .get_post_processor()
        .map(|pp| pp.added_tokens(false))
        .unwrap_or(0);
    let real_input_len = input_len.saturating_sub(num_special);

    // Matching Python's `get_sampling_params()` RNG call order:
    //   1. rng.integers(input_low, input_high+1, size=N)  → input_lens
    //   2. rng.integers(output_low, output_high+1, size=N) → output_lens
    //   3. rng.integers(0, vocab_size, size=N)            → offsets
    let input_low = (real_input_len as f64 * (1.0 - range_ratio)).floor() as usize;
    let input_high = (real_input_len as f64 * (1.0 + range_ratio)).ceil() as usize;
    let output_low = ((output_len as f64 * (1.0 - range_ratio)).floor() as usize).max(1);
    let output_high = ((output_len as f64 * (1.0 + range_ratio)).ceil() as usize).max(1);

    let input_lens: Vec<usize> = (0..num_requests)
        .map(|_| rng.gen_range(input_low..=input_high))
        .collect();
    let output_lens: Vec<usize> = (0..num_requests)
        .map(|_| rng.gen_range(output_low..=output_high))
        .collect();
    // Offsets sampled from full vocab range [0, vocab_size), matching Python.
    let offsets: Vec<usize> = (0..num_requests)
        .map(|_| rng.gen_range(0..vocab_size))
        .collect();

    // Generate prefix once (prefix_len=0 → empty), matching Python's `get_prefix()`.
    let prefix_token_ids: Vec<u32> = if prefix_len > 0 {
        let raw: Vec<u32> = (0..prefix_len)
            .map(|_| allowed_tokens[rng.gen_range(0..num_allowed)])
            .collect();
        let (_, ids) = gen_prompt_to_target_len(tokenizer, &mut rng, raw, prefix_len, vocab_size)?;
        ids
    } else {
        Vec::new()
    };

    let mut samples = Vec::with_capacity(num_requests);
    for i in 0..num_requests {
        let req_input_len = input_lens[i];
        // inner_seq = allowed_tokens[(offset + index + arange(input_len)) % len(allowed)]
        let inner_seq: Vec<u32> = (0..req_input_len)
            .map(|j| allowed_tokens[(offsets[i] + i + j) % num_allowed])
            .collect();

        // token_sequence = prefix_token_ids + inner_seq
        let token_sequence: Vec<u32> = prefix_token_ids.iter().copied().chain(inner_seq).collect();

        let total_target_len = prefix_len + req_input_len;

        let (prompt, actual_ids) = gen_prompt_to_target_len(
            tokenizer,
            &mut rng,
            token_sequence,
            total_target_len,
            vocab_size,
        )?;

        samples.push(SampleRequest {
            prompt,
            prompt_len: actual_ids.len(),
            expected_output_len: output_lens[i],
        });
    }

    Ok(samples)
}

/// Decode token ids to text, re-encode, and iteratively adjust to `target_len`.
///
/// Matches Python's `gen_prompt_decode_to_target_len` exactly:
/// - Up to 10 retries (`max_retry = 10`)
/// - When short: extend with random tokens from `[0, vocab_size)` (full vocab, NOT allowed-only)
/// - When long: truncate to `target_len`
/// - After `remain_num_try <= 0`: break unconditionally
///
/// Returns `(prompt_string, final_token_ids)`.
fn gen_prompt_to_target_len(
    tokenizer: &Tokenizer,
    rng: &mut StdRng,
    mut token_ids: Vec<u32>,
    target_len: usize,
    vocab_size: usize,
) -> Result<(String, Vec<u32>)> {
    let mut remain: i32 = 10;
    let mut prompt;
    loop {
        prompt = tokenizer
            .decode(&token_ids, true)
            .map_err(|e| anyhow::anyhow!("Decode failed: {e}"))?;
        let encoded = tokenizer
            .encode(prompt.as_str(), false)
            .map_err(|e| anyhow::anyhow!("Encode failed: {e}"))?;
        token_ids = encoded.get_ids().to_vec();

        if remain <= 0 {
            break;
        }
        if token_ids.len() == target_len {
            break;
        } else if token_ids.len() < target_len {
            // Pad with random tokens from full vocab range, matching Python:
            //   extra_tokens = rng.integers(0, vocab_size, size=needed)
            let needed = target_len - token_ids.len();
            for _ in 0..needed {
                token_ids.push(rng.gen_range(0..vocab_size as u32));
            }
        } else {
            token_ids.truncate(target_len);
        }
        remain -= 1;
    }
    Ok((prompt, token_ids))
}
