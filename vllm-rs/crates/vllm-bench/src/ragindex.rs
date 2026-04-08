// SPDX-License-Identifier: Apache-2.0
// Copyright contributors to the vLLM project

//! `vllm bench ragindex` — end-to-end LEANN RAG benchmark.
//!
//! Exercises the full RAG pipeline: corpus indexing via LEANN, vector
//! retrieval, and LLM generation via SPNL span queries. For each query
//! from a selectable dataset (hotpotqa, multihop, musique, msmarco), an
//! `Augment` SPNL node indexes the corpus (once), retrieves top-k fragments,
//! and generates an answer. Reports running and final accuracy, F1, and
//! latency statistics.

use std::collections::HashSet;
use std::hash::{Hash, Hasher};
use std::time::Instant;

use anyhow::Result;
use indicatif::{ProgressBar, ProgressStyle};
#[cfg(feature = "rag")]
use spnl_core::ir::{Augment, Document};
use spnl_core::ir::{Generate, GenerateMetadata, Message, Query as SpnlQuery};
use vllm_config::{CudaGraphConfig, CudaGraphMode};
use vllm_serve::llm::{LLM, LLMBuilder, SamplingParams};

use crate::args::BenchRagindexArgs;
use crate::datasets::{QueryMode, best_token_f1, evaluate_accuracy, fetch_rag_dataset};

const BOLD: &str = "\x1b[1m";
const DIM: &str = "\x1b[2m";
const RST: &str = "\x1b[0m";

// ---------------------------------------------------------------------------
// LLM construction
// ---------------------------------------------------------------------------

fn build_llm(args: &BenchRagindexArgs) -> Result<LLM> {
    let model = args.resolved_model().map_err(|e| anyhow::anyhow!(e))?;
    let mut builder = LLMBuilder::new(&model)
        .device(&args.device)
        .dtype(&args.dtype)
        .gpu_memory_utilization(args.gpu_memory_utilization)
        .max_num_seqs(args.max_num_seqs)
        .block_size(args.block_size)
        .tensor_parallel_size(args.tensor_parallel_size)
        .enforce_eager(args.enforce_eager)
        .enable_prefix_caching(true);

    builder = builder.max_num_batched_tokens(8192);

    if let Some(len) = args.max_model_len {
        builder = builder.max_model_len(len);
    }
    if let Some(ref token) = args.hf_token {
        builder = builder.hf_token(token);
    }
    if let Some(ref gguf) = args.gguf_file {
        builder = builder.gguf_file(gguf);
    }
    if !args.enforce_eager {
        let sizes = CudaGraphConfig::parse_sizes("auto");
        if !sizes.is_empty() {
            builder = builder.cuda_graph_config(CudaGraphConfig {
                enabled: true,
                mode: CudaGraphMode::default(),
                capture_sizes: sizes,
                num_warmups: 3,
            });
        }
    }
    builder.build()
}

// ---------------------------------------------------------------------------
// Running statistics
// ---------------------------------------------------------------------------

struct RunningStats {
    accs: Vec<f64>,
    f1s: Vec<f64>,
    latencies_ms: Vec<f64>,
    total_prompt_tokens: u64,
    total_cached_tokens: u64,
}

impl RunningStats {
    fn new(capacity: usize) -> Self {
        Self {
            accs: Vec::with_capacity(capacity),
            f1s: Vec::with_capacity(capacity),
            latencies_ms: Vec::with_capacity(capacity),
            total_prompt_tokens: 0,
            total_cached_tokens: 0,
        }
    }

    fn push(&mut self, acc: f64, f1: f64, ms: f64, prompt_tokens: u32, cached_tokens: u32) {
        self.accs.push(acc);
        self.f1s.push(f1);
        self.latencies_ms.push(ms);
        self.total_prompt_tokens += prompt_tokens as u64;
        self.total_cached_tokens += cached_tokens as u64;
    }

    fn cache_pct(&self) -> f64 {
        if self.total_prompt_tokens == 0 {
            0.0
        } else {
            self.total_cached_tokens as f64 / self.total_prompt_tokens as f64 * 100.0
        }
    }

    fn n(&self) -> usize {
        self.accs.len()
    }

    fn avg_acc(&self) -> f64 {
        self.accs.iter().sum::<f64>() / self.n().max(1) as f64
    }

    fn avg_f1(&self) -> f64 {
        self.f1s.iter().sum::<f64>() / self.n().max(1) as f64
    }

    fn avg_ms(&self) -> f64 {
        self.latencies_ms.iter().sum::<f64>() / self.n().max(1) as f64
    }

    fn p50_ms(&self) -> f64 {
        percentile_of(&self.latencies_ms, 50.0)
    }

    fn p99_ms(&self) -> f64 {
        percentile_of(&self.latencies_ms, 99.0)
    }
}

// ---------------------------------------------------------------------------
// Block overlap tracker — measures potential KV cache reuse
// ---------------------------------------------------------------------------

/// Tracks token-block hashes across queries to measure how many blocks
/// *could* be reused if the cache worked perfectly. Hashes each block's
/// tokens with a fixed parent (like relocatable blocks), so any two queries
/// sharing the same token block will match.
struct BlockOverlapTracker {
    block_size: usize,
    seen: HashSet<u64>,
    total_blocks: u64,
    reusable_blocks: u64,
}

impl BlockOverlapTracker {
    fn new(block_size: usize) -> Self {
        Self {
            block_size,
            seen: HashSet::new(),
            total_blocks: 0,
            reusable_blocks: 0,
        }
    }

    /// Record blocks from a query's prompt tokens. Returns the number of
    /// blocks that were seen in a previous query (potential reuse).
    fn record(&mut self, prompt_tokens: &[u32]) -> u64 {
        let mut reused = 0u64;
        for chunk in prompt_tokens.chunks(self.block_size) {
            if chunk.len() < self.block_size {
                break; // skip partial trailing block
            }
            self.total_blocks += 1;
            let hash = hash_block(chunk);
            if !self.seen.insert(hash) {
                // Already seen in a previous query
                reused += 1;
                self.reusable_blocks += 1;
            }
        }
        reused
    }

    fn potential_pct(&self) -> f64 {
        if self.total_blocks == 0 {
            0.0
        } else {
            self.reusable_blocks as f64 / self.total_blocks as f64 * 100.0
        }
    }

    fn unique_blocks(&self) -> usize {
        self.seen.len()
    }
}

/// Hash a block's tokens the same way the scheduler does for relocatable
/// blocks (parent = NONE_HASH = 0).
fn hash_block(tokens: &[u32]) -> u64 {
    let mut hasher = std::hash::DefaultHasher::new();
    0u64.hash(&mut hasher); // NONE_HASH parent
    tokens.hash(&mut hasher);
    hasher.finish()
}

fn percentile_of(values: &[f64], p: f64) -> f64 {
    if values.is_empty() {
        return 0.0;
    }
    let mut sorted = values.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let rank = (p / 100.0) * (sorted.len() - 1) as f64;
    let lo = rank.floor() as usize;
    let hi = rank.ceil() as usize;
    if lo == hi {
        sorted[lo]
    } else {
        let frac = rank - lo as f64;
        sorted[lo] * (1.0 - frac) + sorted[hi] * frac
    }
}

// ---------------------------------------------------------------------------
// Entry point
// ---------------------------------------------------------------------------

pub(crate) fn run_bench_ragindex(args: BenchRagindexArgs) -> Result<()> {
    vllm_common::telemetry::init_tracing(&args.log_level);

    let dataset_name = args.dataset.to_string();
    let samples = fetch_rag_dataset(args.dataset, args.num_queries)?;
    let n_queries = samples.len();

    if n_queries == 0 {
        eprintln!("No queries found in dataset.");
        return Ok(());
    }

    let mut llm = build_llm(&args)?;
    let model_name = llm.model_name().to_string();
    let embedding_model = args.resolved_embedding_model();

    let sampling = SamplingParams {
        max_tokens: Some(args.max_tokens as u32),
        temperature: 0.0,
        ..SamplingParams::default()
    };

    eprintln!();
    eprintln!("{BOLD}vLLM Rust \u{2014} RAG index benchmark{RST}");
    eprintln!("  Dataset:    {dataset_name}");
    eprintln!("  Model:      {model_name}");
    eprintln!("  Embedding:  {embedding_model}");
    eprintln!("  Queries:    {n_queries}");
    eprintln!("  Top-k:      {}", args.max_aug);
    eprintln!("  Mode:       {:?}", args.query_mode);

    #[cfg(not(feature = "rag"))]
    {
        eprintln!();
        eprintln!("{BOLD}Skipped{RST} (build with --features rag for LEANN indexing)");
        return Ok(());
    }

    #[cfg(feature = "rag")]
    {
        if args.force_reindex {
            let aug_defaults = vllm_serve::augment::AugmentOptions::default();
            let _ = std::fs::remove_dir_all(&aug_defaults.index_dir);
        }

        // Build corpus for LEANN indexing.
        // Include a content hash in the filename so different -n values
        // produce different indexes (avoids stale index reuse).
        let mut doc_labels = std::collections::BTreeSet::new();
        let mut corpus_parts = Vec::new();
        for sample in &samples {
            for (label, text) in &sample.documents {
                if doc_labels.insert(label.clone()) {
                    corpus_parts.push(format!("{label}: {text}"));
                }
            }
        }
        let total_docs = doc_labels.len();
        let corpus_text = corpus_parts.join("\n\n");

        let mut hasher = std::hash::DefaultHasher::new();
        total_docs.hash(&mut hasher);
        for label in &doc_labels {
            label.hash(&mut hasher);
        }
        let corpus_hash = hasher.finish();
        let corpus_filename = format!("{dataset_name}_{corpus_hash:016x}_corpus.txt");
        eprintln!("  Corpus:     {total_docs} documents");

        let corpus_doc = (corpus_filename, Document::Text(corpus_text));
        let use_spans = args.query_mode == QueryMode::Spans;

        eprintln!();
        if args.concurrency > 1 {
            eprintln!(
                "  Concurrency: {} (per-query latency is amortized)",
                args.concurrency
            );
        }
        let mut stats = RunningStats::new(n_queries);
        let mut overlap = BlockOverlapTracker::new(args.block_size);
        let pb = make_progress_bar(n_queries);

        // Build the SPNL query for one sample. Same shape regardless of
        // concurrency mode — the difference is whether we submit one or
        // K queries to the engine at once.
        let build_query = |sample: &crate::datasets::RagSample| -> SpnlQuery {
            let augment = SpnlQuery::Augment(Augment {
                embedding_model: embedding_model.clone(),
                body: Box::new(SpnlQuery::Message(Message::User(sample.question.clone()))),
                doc: corpus_doc.clone(),
            });
            let question = SpnlQuery::Message(Message::User(format!(
                "Based on the above context, answer: {}",
                sample.question
            )));
            let children = vec![augment, question];
            SpnlQuery::Generate(Generate {
                metadata: GenerateMetadata {
                    model: model_name.clone(),
                    max_tokens: Some(args.max_tokens as i32),
                    temperature: Some(0.0),
                },
                input: Box::new(if use_spans {
                    SpnlQuery::Plus(children)
                } else {
                    SpnlQuery::Seq(children)
                }),
            })
        };

        let mut qi: usize = 0;
        for chunk in samples.chunks(args.concurrency.max(1)) {
            let chunk_start = Instant::now();

            // Sequential path (concurrency=1) keeps the existing
            // execute_spnl flow so per-query latency stays meaningful.
            // Concurrent path (>1) collects chunk queries and submits
            // them in one continuous-batched generate call.
            let outputs: Vec<vllm_serve::llm::RequestOutput> = if args.concurrency <= 1 {
                let query = build_query(&chunk[0]);
                let result = llm.execute_spnl(query, Some(sampling.clone()), false, false)?;
                vec![result.output().clone()]
            } else {
                let queries: Vec<SpnlQuery> = chunk.iter().map(&build_query).collect();
                llm.execute_spnl_concurrent(queries, Some(sampling.clone()))?
            };

            let chunk_ms = chunk_start.elapsed().as_secs_f64() * 1000.0;
            let per_query_ms = chunk_ms / chunk.len() as f64;

            for (i, output) in outputs.iter().enumerate() {
                let sample = &chunk[i];
                let response = &output.outputs[0].text;
                let acc = evaluate_accuracy(response, &sample.answers);
                let f1 = best_token_f1(&sample.answers, response);
                let prompt_tokens = output.prompt_token_ids.len() as u32;
                let cached_tokens = output.num_cached_tokens;
                let reused_blocks = overlap.record(&output.prompt_token_ids);
                stats.push(acc, f1, per_query_ms, prompt_tokens, cached_tokens);

                if args.debug && qi < 3 {
                    pb.suspend(|| {
                        eprintln!("  {DIM}Q: {}{RST}", sample.question);
                        eprintln!("  {DIM}A: {response}{RST}");
                        eprintln!("  {DIM}Expected: {}{RST}", sample.answers.join(" | "));
                        eprintln!(
                            "  {DIM}acc={acc:.0} F1={f1:.3} {} reusable_blocks={reused_blocks}{RST}",
                            fmt_ms(per_query_ms)
                        );
                    });
                }

                qi += 1;
                update_progress(&pb, &stats, &overlap);
            }
        }
        pb.finish_and_clear();

        // -- Summary --
        eprintln!();
        let mode_label = if use_spans { "Spans" } else { "Plain" };
        println!("{BOLD}=== RAG Index Benchmark Results ({mode_label}) ==={RST}");
        println!("  Dataset:  {dataset_name}");
        println!("  Queries:  {n_queries}");
        println!(
            "  Blocks:   {} unique, {:.0}% potential reuse",
            overlap.unique_blocks(),
            overlap.potential_pct(),
        );
        println!();
        print_stats(mode_label, &stats, &overlap);
        println!();
    }

    Ok(())
}

fn make_progress_bar(n: usize) -> ProgressBar {
    ProgressBar::new(n as u64).with_style(
        ProgressStyle::default_bar()
            .template("  {bar:30.green/green} {pos:>4}/{len}  {msg}")
            .unwrap(),
    )
}

fn update_progress(pb: &ProgressBar, stats: &RunningStats, overlap: &BlockOverlapTracker) {
    pb.set_message(format!(
        "acc={:.1}%  F1={:.3}  cache={:.0}%  potential={:.0}%  avg={}  p50={}  p99={}",
        stats.avg_acc() * 100.0,
        stats.avg_f1(),
        stats.cache_pct(),
        overlap.potential_pct(),
        fmt_ms(stats.avg_ms()),
        fmt_ms(stats.p50_ms()),
        fmt_ms(stats.p99_ms()),
    ));
    pb.inc(1);
}

fn print_stats(label: &str, s: &RunningStats, overlap: &BlockOverlapTracker) {
    println!(
        "  {BOLD}{label}{RST}:  acc={BOLD}{:.1}%{RST}  F1={BOLD}{:.3}{RST}  cache={BOLD}{:.0}%{RST}  potential={BOLD}{:.0}%{RST}  avg={}  p50={}  p99={}",
        s.avg_acc() * 100.0,
        s.avg_f1(),
        s.cache_pct(),
        overlap.potential_pct(),
        fmt_ms(s.avg_ms()),
        fmt_ms(s.p50_ms()),
        fmt_ms(s.p99_ms()),
    );
}

/// Format milliseconds as a human-readable duration (e.g. "7.4s", "238ms").
fn fmt_ms(ms: f64) -> String {
    if ms >= 1000.0 {
        format!("{:.1}s", ms / 1000.0)
    } else {
        format!("{:.0}ms", ms)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[cfg(feature = "rag")]
    #[test]
    fn augment_query_structure() {
        use spnl_core::ir::{Augment, Document};

        let query = SpnlQuery::Generate(Generate {
            metadata: GenerateMetadata {
                model: "test-model".to_string(),
                max_tokens: Some(64),
                temperature: Some(0.0),
            },
            input: Box::new(SpnlQuery::Plus(vec![
                SpnlQuery::Augment(Augment {
                    embedding_model: "embed-model".to_string(),
                    body: Box::new(SpnlQuery::Message(Message::User("What is X?".to_string()))),
                    doc: (
                        "doc.txt".to_string(),
                        Document::Text("Some document text".to_string()),
                    ),
                }),
                SpnlQuery::Message(Message::User("Answer the question.".to_string())),
            ])),
        });

        match &query {
            SpnlQuery::Generate(g) => {
                assert_eq!(g.metadata.model, "test-model");
                match g.input.as_ref() {
                    SpnlQuery::Plus(children) => {
                        assert_eq!(children.len(), 2);
                        assert!(matches!(&children[0], SpnlQuery::Augment(_)));
                    }
                    other => panic!("expected Plus, got {other:?}"),
                }
            }
            other => panic!("expected Generate, got {other:?}"),
        }

        let json = serde_json::to_string(&query).unwrap();
        let parsed: SpnlQuery = serde_json::from_str(&json).expect("should round-trip");
        assert!(matches!(parsed, SpnlQuery::Generate(_)));
    }

    #[test]
    fn evaluate_accuracy_basic() {
        assert!((evaluate_accuracy("Paris", &["Paris".to_string()]) - 1.0).abs() < f64::EPSILON);
        assert!(
            (evaluate_accuracy("I think Paris is the answer", &["Paris".to_string()]) - 1.0).abs()
                < f64::EPSILON
        );
        assert!(evaluate_accuracy("London", &["Paris".to_string()]).abs() < f64::EPSILON);
    }

    #[test]
    fn best_token_f1_basic() {
        let f1 = best_token_f1(&["the cat sat".to_string()], "the cat sat on the mat");
        assert!(f1 > 0.5);
    }
}
