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
#[cfg_attr(not(feature = "rag"), allow(dead_code))]
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
    /// NarrativeQA validation set (long-form story summaries from Gutenberg
    /// books and movie scripts; ~660-word summary per document, multiple
    /// questions per doc — long-context territory where RAPTOR's level
    /// summaries are designed to help).
    Narrativeqa,
    /// QASPER validation set (NLP scientific papers with multi-section full
    /// text and 4–5 expert-annotated questions per paper; tests retrieval
    /// over structured long documents).
    Qasper,
}

/// Query execution mode for `vllm bench ragindex`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum)]
pub enum QueryMode {
    /// Plain chat with gold documents inlined.
    Plain,
    /// SPNL span query with LEANN retrieval.
    Spans,
}

impl std::fmt::Display for RagDataset {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Hotpotqa => write!(f, "hotpotqa"),
            Self::Multihop => write!(f, "multihop"),
            Self::Musique => write!(f, "musique"),
            Self::Msmarco => write!(f, "msmarco"),
            Self::Narrativeqa => write!(f, "narrativeqa"),
            Self::Qasper => write!(f, "qasper"),
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
        RagDataset::Narrativeqa => fetch_narrativeqa(num_queries),
        RagDataset::Qasper => fetch_qasper(num_queries),
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
// Multi-shard parquet helper
// ---------------------------------------------------------------------------

/// Like `download_parquet_as_json` but takes N URLs (parquet shards) and
/// concatenates the rows. Used by datasets whose HF parquet split is not
/// in a single file.
fn download_parquet_shards_as_json(
    cache_dir_name: &str,
    cache_filename: &str,
    urls: &[&str],
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

    eprintln!("Downloading {label} ({} shards)...", urls.len());
    let client = reqwest::blocking::Client::builder()
        .timeout(std::time::Duration::from_secs(600))
        .build()?;

    let mut all_records: Vec<serde_json::Value> = Vec::new();
    for (i, url) in urls.iter().enumerate() {
        let shard_path = cache_dir.join(format!("shard-{i}.parquet"));
        let response = client.get(*url).header("User-Agent", "vllm-bench").send()?;
        let bytes = response.bytes()?;
        std::fs::write(&shard_path, &bytes)?;
        let mut records = parquet_to_json_records(&shard_path)?;
        all_records.append(&mut records);
    }
    eprintln!("Caching {} records as JSON...", all_records.len());
    let json_str = serde_json::to_string(&all_records)?;
    std::fs::write(&cache_file, json_str.as_bytes())?;
    Ok(all_records)
}

// ---------------------------------------------------------------------------
// NarrativeQA
// ---------------------------------------------------------------------------

fn fetch_narrativeqa(num_queries: usize) -> Result<Vec<RagSample>> {
    // The HF parquet split for `deepmind/narrativeqa` validation is sharded
    // across two files. URLs from the HF datasets-server parquet API.
    let raw = download_parquet_shards_as_json(
        "narrativeqa",
        "validation.json",
        &[
            "https://huggingface.co/api/datasets/deepmind/narrativeqa/parquet/default/validation/0.parquet",
            "https://huggingface.co/api/datasets/deepmind/narrativeqa/parquet/default/validation/1.parquet",
        ],
        "NarrativeQA validation set",
    )?;

    let mut samples = Vec::new();
    for record in &raw {
        let document = &record["document"];
        let summary = &document["summary"];
        let title = summary["title"].as_str().unwrap_or("").to_string();
        let text = summary["text"].as_str().unwrap_or("").to_string();
        let question = record["question"]["text"]
            .as_str()
            .unwrap_or("")
            .to_string();

        // Two reference answers per question; collect both.
        let answers: Vec<String> = record["answers"]
            .as_array()
            .map(|arr| {
                arr.iter()
                    .filter_map(|a| a["text"].as_str().map(|s| s.to_string()))
                    .filter(|s| !s.trim().is_empty())
                    .collect()
            })
            .unwrap_or_default();

        if question.is_empty() || text.is_empty() || answers.is_empty() {
            continue;
        }

        samples.push(RagSample {
            question,
            answers,
            documents: vec![(title, text)],
        });

        if samples.len() >= num_queries {
            break;
        }
    }
    Ok(samples)
}

// ---------------------------------------------------------------------------
// QASPER
// ---------------------------------------------------------------------------

fn fetch_qasper(num_queries: usize) -> Result<Vec<RagSample>> {
    // Each QASPER row is one paper carrying 4–5 expert-annotated questions
    // and a sectioned full text. We flatten to one RagSample per question:
    // documents = paper sections, answers = union of extractive_spans +
    // free_form_answer across all annotators.
    let raw = download_parquet_shards_as_json(
        "qasper",
        "validation.json",
        &["https://huggingface.co/api/datasets/allenai/qasper/parquet/qasper/validation/0.parquet"],
        "QASPER validation set",
    )?;

    let mut samples = Vec::new();
    'papers: for record in &raw {
        let title = record["title"].as_str().unwrap_or("").to_string();
        let abstract_text = record["abstract"].as_str().unwrap_or("").to_string();

        // Build the document list once per paper: abstract + each section.
        let mut documents: Vec<(String, String)> = Vec::new();
        if !abstract_text.is_empty() {
            documents.push((format!("{title} — Abstract"), abstract_text));
        }
        let full_text = &record["full_text"];
        let section_names = full_text["section_name"]
            .as_array()
            .cloned()
            .unwrap_or_default();
        let sections = full_text["paragraphs"]
            .as_array()
            .cloned()
            .unwrap_or_default();
        for (name_val, paras_val) in section_names.iter().zip(sections.iter()) {
            let name = name_val.as_str().unwrap_or("").to_string();
            let body = paras_val
                .as_array()
                .map(|arr| {
                    arr.iter()
                        .filter_map(|p| p.as_str())
                        .collect::<Vec<_>>()
                        .join("\n")
                })
                .unwrap_or_default();
            if body.trim().is_empty() {
                continue;
            }
            documents.push((format!("{title} — {name}"), body));
        }
        if documents.is_empty() {
            continue;
        }

        let qas = &record["qas"];
        let questions = qas["question"].as_array().cloned().unwrap_or_default();
        let answers_lists = qas["answers"].as_array().cloned().unwrap_or_default();

        for (q_val, ann_list_val) in questions.iter().zip(answers_lists.iter()) {
            let question = q_val.as_str().unwrap_or("").to_string();
            if question.is_empty() {
                continue;
            }

            // Collect all distinct answer strings across annotators. Skip
            // unanswerable / yes-no markers — we evaluate by substring/F1
            // and those don't carry over.
            let mut answers: Vec<String> = Vec::new();
            if let Some(ann_arr) = ann_list_val.as_array() {
                for ann in ann_arr {
                    let inner = &ann["answer"];
                    let inner_arr = inner.as_array().cloned().unwrap_or_default();
                    for a in &inner_arr {
                        if a["unanswerable"].as_bool().unwrap_or(false) {
                            continue;
                        }
                        if let Some(spans) = a["extractive_spans"].as_array() {
                            for s in spans {
                                if let Some(t) = s.as_str().filter(|t| !t.trim().is_empty()) {
                                    answers.push(t.to_string());
                                }
                            }
                        }
                        if let Some(ff) = a["free_form_answer"]
                            .as_str()
                            .filter(|s| !s.trim().is_empty())
                        {
                            answers.push(ff.to_string());
                        }
                        if let Some(yn) = a["yes_no"].as_bool() {
                            answers.push(if yn { "yes".into() } else { "no".into() });
                        }
                    }
                }
            }
            answers.sort();
            answers.dedup();
            if answers.is_empty() {
                continue;
            }

            samples.push(RagSample {
                question,
                answers,
                documents: documents.clone(),
            });

            if samples.len() >= num_queries {
                break 'papers;
            }
        }
    }
    Ok(samples)
}

// ---------------------------------------------------------------------------
// Permutation generation (reusable)
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// Accuracy evaluation (reusable)
// ---------------------------------------------------------------------------

/// Evaluate response against any acceptable answer.
/// Returns 1.0 if any answer is a substring match or token F1 >= 0.5.
#[cfg(any(feature = "rag", test))]
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
#[cfg(any(feature = "rag", test))]
pub fn best_token_f1(answers: &[String], actual: &str) -> f64 {
    answers
        .iter()
        .map(|ans| compute_token_f1(&ans.to_lowercase(), &actual.to_lowercase()))
        .fold(0.0_f64, f64::max)
}

#[cfg(any(feature = "rag", test))]
fn normalize_tokens(text: &str) -> Vec<String> {
    text.split(|c: char| !c.is_alphanumeric())
        .filter(|s| !s.is_empty())
        .map(|s| s.to_string())
        .collect()
}

#[cfg(any(feature = "rag", test))]
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
