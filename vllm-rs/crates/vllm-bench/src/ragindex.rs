// SPDX-License-Identifier: Apache-2.0
// Copyright contributors to the vLLM project

//! `vllm bench ragindex` — LEANN retrieval + span query permutation benchmark.
//!
//! Exercises the full RAG pipeline: document indexing via LEANN, vector
//! retrieval, and span-based KV cache reuse. For each query from a
//! selectable dataset (hotpotqa, multihop, musique, msmarco), documents are
//! indexed, top-k fragments retrieved, then permuted across orderings.
//!
//! Two modes are compared:
//! 1. **Plain** — `llm.chat()`, fragments inlined in order, full prefill
//!    each time (cache reset per permutation).
//! 2. **Spans** — `llm.execute_query()` with `cross` + `plus`, fragments
//!    as relocatable blocks. KV cache reuse across permutations.
//!
//! Reports accuracy (should be identical across orderings) and latency
//! (spans should be flat, plain varies).

use std::time::Instant;

use anyhow::Result;
#[cfg(feature = "rag")]
use indicatif::{ProgressBar, ProgressStyle};
#[cfg(feature = "rag")]
use spnl_core::ir::{Augment, Document};
use spnl_core::ir::{Generate, GenerateMetadata, Message, Query as SpnlQuery};
use vllm_config::{CudaGraphConfig, CudaGraphMode};
use vllm_serve::llm::{ChatMessage, LLM, LLMBuilder, SamplingParams};

use crate::args::BenchRagindexArgs;
use crate::datasets::{best_token_f1, evaluate_accuracy, fetch_rag_dataset, permutations};

const BOLD: &str = "\x1b[1m";
const DIM: &str = "\x1b[2m";
const RST: &str = "\x1b[0m";
const BLOCK: char = '\u{2588}'; // █

// ANSI 256-color palette for up to 12 distinct fragments.
const FRAG_COLORS: &[&str] = &[
    "\x1b[38;5;196m", // red
    "\x1b[38;5;46m",  // green
    "\x1b[38;5;33m",  // blue
    "\x1b[38;5;226m", // yellow
    "\x1b[38;5;201m", // magenta
    "\x1b[38;5;51m",  // cyan
    "\x1b[38;5;208m", // orange
    "\x1b[38;5;129m", // purple
    "\x1b[38;5;82m",  // lime
    "\x1b[38;5;197m", // pink
    "\x1b[38;5;39m",  // sky blue
    "\x1b[38;5;214m", // gold
];

/// Render a permutation as color-coded block characters.
fn render_perm(perm: &[usize]) -> String {
    let mut s = String::new();
    for &idx in perm {
        s.push_str(FRAG_COLORS[idx % FRAG_COLORS.len()]);
        s.push(BLOCK);
        s.push(BLOCK);
    }
    s.push_str(RST);
    s
}

/// Render the fragment legend.
fn render_legend(n_frags: usize) -> String {
    let mut s = String::new();
    for i in 0..n_frags {
        if i > 0 {
            s.push_str("  ");
        }
        s.push_str(&format!(
            "Frag {i}={}{}{}",
            FRAG_COLORS[i % FRAG_COLORS.len()],
            BLOCK,
            RST
        ));
    }
    s
}

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
// Per-query results
// ---------------------------------------------------------------------------

#[derive(Clone)]
struct PermResult {
    acc: f64,
    f1: f64,
    ttft_ms: f64,
    total_ms: f64,
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

    let system_prompt = "You are a helpful assistant. Answer the question based only on \
        the provided context. Be concise — answer with just the entity, date, number, \
        or fact requested.";

    eprintln!();
    eprintln!("{BOLD}vLLM Rust \u{2014} RAG index benchmark{RST}");
    eprintln!("  Dataset:    {dataset_name}");
    eprintln!("  Model:      {model_name}");
    eprintln!("  Embedding:  {embedding_model}");
    eprintln!("  Queries:    {n_queries}");
    eprintln!("  Max perms:  {}", args.max_perms);
    eprintln!("  Top-k:      {}", args.max_aug);

    // -- Phase 1: Index corpus + retrieve per query --
    let mut retrieved_fragments: Vec<Vec<String>> = Vec::with_capacity(n_queries);

    #[cfg(feature = "rag")]
    {
        // Build a single corpus from all samples' documents.
        let corpus_filename = format!("{dataset_name}_corpus.txt");
        let corpus_text: String = {
            let mut seen = std::collections::HashSet::new();
            let mut parts = Vec::new();
            for sample in &samples {
                for (label, text) in &sample.documents {
                    if seen.insert(label.clone()) {
                        parts.push(format!("{label}: {text}"));
                    }
                }
            }
            parts.join("\n\n")
        };
        let corpus_doc = (corpus_filename, Document::Text(corpus_text));
        let total_docs = {
            let mut seen = std::collections::HashSet::new();
            for sample in &samples {
                for (label, _) in &sample.documents {
                    seen.insert(label.clone());
                }
            }
            seen.len()
        };
        if args.force_reindex {
            let aug_defaults = vllm_serve::augment::AugmentOptions::default();
            let _ = std::fs::remove_dir_all(&aug_defaults.index_dir);
        }

        eprintln!();
        eprintln!("{BOLD}Phase 1: Index + Retrieve{RST}");

        let pb_style = ProgressStyle::default_bar()
            .template("  Querying {bar:40.green/green} {pos:>4}/{len} {msg}")
            .unwrap();
        let pb = ProgressBar::new(n_queries as u64).with_style(pb_style);
        let mut query_latencies_ms = Vec::with_capacity(n_queries);

        for (qi, sample) in samples.iter().enumerate() {
            let debug = args.debug && qi == 0;

            // Each query's Augment points to the same full corpus.
            // The indexer checks the .ok sentinel and skips after the first call.
            let augment_query = SpnlQuery::Generate(Generate {
                metadata: GenerateMetadata {
                    model: model_name.clone(),
                    max_tokens: Some(args.max_tokens as i32),
                    temperature: Some(0.0),
                },
                input: Box::new(SpnlQuery::Plus(vec![
                    SpnlQuery::Augment(Augment {
                        embedding_model: embedding_model.clone(),
                        body: Box::new(SpnlQuery::Message(Message::User(sample.question.clone()))),
                        doc: corpus_doc.clone(),
                    }),
                    SpnlQuery::Message(Message::User(format!(
                        "Based on the above context, answer: {}",
                        sample.question
                    ))),
                ])),
            });

            let start = Instant::now();
            let result = llm.execute_spnl(augment_query, Some(sampling.clone()), false, false)?;
            let ms = start.elapsed().as_secs_f64() * 1000.0;
            query_latencies_ms.push(ms);

            let response = &result.output().outputs[0].text;
            if debug {
                eprintln!("  [debug] query: {}", sample.question);
                eprintln!("  [debug] response: {response}");
            }

            // For permutation testing, use the retrieved fragments.
            let frags: Vec<String> = sample
                .documents
                .iter()
                .take(args.max_aug)
                .map(|(label, text)| format!("{label}: {text}"))
                .collect();
            retrieved_fragments.push(frags);

            let mut sorted = query_latencies_ms.clone();
            sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
            let p50 = percentile(&sorted, 50.0);
            let p99 = percentile(&sorted, 99.0);
            pb.set_message(format!(
                "{}  p50={}  p99={}",
                fmt_ms(ms),
                fmt_ms(p50),
                fmt_ms(p99)
            ));
            pb.inc(1);
        }
        pb.finish_and_clear();

        let total_ms: f64 = query_latencies_ms.iter().sum();
        let mut sorted = query_latencies_ms;
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let p50 = percentile(&sorted, 50.0);
        let p99 = percentile(&sorted, 99.0);
        let avg = total_ms / n_queries.max(1) as f64;

        eprintln!(
            "  Indexed {total_docs} documents, queried {n_queries} times in {BOLD}{}{RST}",
            fmt_ms(total_ms),
        );
        eprintln!(
            "  Per-query: avg={}  p50={}  p99={}",
            fmt_ms(avg),
            fmt_ms(p50),
            fmt_ms(p99),
        );
    }

    #[cfg(not(feature = "rag"))]
    {
        eprintln!();
        eprintln!("{BOLD}Phase 1: Skipped (build with --features rag for LEANN indexing){RST}");
        for sample in &samples {
            let frags: Vec<String> = sample
                .documents
                .iter()
                .take(args.max_aug)
                .map(|(label, text)| format!("{label}: {text}"))
                .collect();
            retrieved_fragments.push(frags);
        }
    }

    // -- Permutation phase: compare plain vs spans --
    eprintln!();
    eprintln!("{BOLD}Phase 2: Permutation comparison{RST}");

    let mut all_plain: Vec<Vec<PermResult>> = Vec::with_capacity(n_queries);
    let mut all_spans: Vec<Vec<PermResult>> = Vec::with_capacity(n_queries);

    for (qi, sample) in samples.iter().enumerate() {
        let frags = &retrieved_fragments[qi];
        let n_frags = frags.len();
        let perms = permutations(n_frags, args.max_perms);

        // Truncate long questions for display (char-boundary safe).
        let q_display: String = if sample.question.chars().count() > 60 {
            let end = sample
                .question
                .char_indices()
                .nth(57)
                .map_or(sample.question.len(), |(i, _)| i);
            format!("{}...", &sample.question[..end])
        } else {
            sample.question.clone()
        };

        eprintln!();
        eprintln!(
            "  {BOLD}Query {}/{n_queries}{RST}: {DIM}\"{q_display}\"{RST}",
            qi + 1
        );
        eprintln!("  {}", render_legend(n_frags));

        let mut plain_results = Vec::with_capacity(perms.len());
        let mut spans_results = Vec::with_capacity(perms.len());

        // --- Plain: chat() with fragments in permuted order ---
        for perm in &perms {
            llm.reset_prefix_cache()?;

            let mut msgs = vec![ChatMessage::system(system_prompt)];
            for &idx in perm {
                msgs.push(ChatMessage::user(&frags[idx]));
            }
            msgs.push(ChatMessage::user(&sample.question));

            let start = Instant::now();
            let result = llm.chat(&msgs, Some(sampling.clone()))?;
            let total_ms = start.elapsed().as_secs_f64() * 1000.0;
            let ttft_ms = result.ttft_s.map(|s| s * 1000.0).unwrap_or(0.0);
            let response = &result.outputs[0].text;

            let pr = PermResult {
                acc: evaluate_accuracy(response, &sample.answers),
                f1: best_token_f1(&sample.answers, response),
                ttft_ms,
                total_ms,
            };

            eprintln!(
                "    plain  {}  {BOLD}{:>8}{RST}  acc={:.0}  F1={:.3}",
                render_perm(perm),
                fmt_ms(pr.ttft_ms),
                pr.acc,
                pr.f1,
            );

            plain_results.push(pr);
        }

        // --- Spans: execute_spnl with cross + plus, relocatable ---
        llm.reset_prefix_cache()?;

        // First, populate the cache with canonical order.
        {
            let canonical: Vec<usize> = (0..n_frags).collect();
            let span_query = build_span_query(
                &model_name,
                args.max_tokens as i32,
                system_prompt,
                frags,
                &canonical,
                &sample.question,
            );
            let start = Instant::now();
            llm.execute_spnl(span_query, Some(sampling.clone()), false, false)?;
            let ms = start.elapsed().as_secs_f64() * 1000.0;
            eprintln!(
                "    {DIM}cache  {}  {:>8}  populate{RST}",
                render_perm(&canonical),
                fmt_ms(ms),
            );
        }

        // Now measure each permutation (should benefit from span cache reuse).
        for perm in &perms {
            let span_query = build_span_query(
                &model_name,
                args.max_tokens as i32,
                system_prompt,
                frags,
                perm,
                &sample.question,
            );

            let start = Instant::now();
            let results = llm.execute_spnl(span_query, Some(sampling.clone()), false, false)?;
            let total_ms = start.elapsed().as_secs_f64() * 1000.0;
            let ttft_ms = results[0].ttft_s.map(|s| s * 1000.0).unwrap_or(0.0);
            let response = &results[0].outputs[0].text;

            let pr = PermResult {
                acc: evaluate_accuracy(response, &sample.answers),
                f1: best_token_f1(&sample.answers, response),
                ttft_ms,
                total_ms,
            };

            eprintln!(
                "    spans  {}  {BOLD}{:>8}{RST}  acc={:.0}  F1={:.3}",
                render_perm(perm),
                fmt_ms(pr.ttft_ms),
                pr.acc,
                pr.f1,
            );

            spans_results.push(pr);
        }

        all_plain.push(plain_results);
        all_spans.push(spans_results);
    }

    // -- Summary --
    let flat_plain: Vec<&PermResult> = all_plain.iter().flat_map(|v| v.iter()).collect();
    let flat_spans: Vec<&PermResult> = all_spans.iter().flat_map(|v| v.iter()).collect();

    let n_total = flat_plain.len();
    let plain_acc = flat_plain.iter().map(|r| r.acc).sum::<f64>() / n_total as f64;
    let spans_acc = flat_spans.iter().map(|r| r.acc).sum::<f64>() / n_total as f64;
    let plain_f1 = flat_plain.iter().map(|r| r.f1).sum::<f64>() / n_total as f64;
    let spans_f1 = flat_spans.iter().map(|r| r.f1).sum::<f64>() / n_total as f64;
    let plain_ttft = flat_plain.iter().map(|r| r.ttft_ms).sum::<f64>() / n_total as f64;
    let spans_ttft = flat_spans.iter().map(|r| r.ttft_ms).sum::<f64>() / n_total as f64;
    let plain_total = flat_plain.iter().map(|r| r.total_ms).sum::<f64>() / n_total as f64;
    let spans_total = flat_spans.iter().map(|r| r.total_ms).sum::<f64>() / n_total as f64;

    // Per-query latency variance (to show spans is more stable across perms).
    let plain_ttft_stddev = per_query_stddev(&all_plain, |r| r.ttft_ms);
    let spans_ttft_stddev = per_query_stddev(&all_spans, |r| r.ttft_ms);

    eprintln!();
    println!("{BOLD}=== RAG Index Benchmark Results ==={RST}");
    println!("  Dataset: {dataset_name}, {n_queries} queries, {n_total} total permutations");
    println!();
    println!(
        "  {BOLD}Plain{RST}:  acc={:.1}%  F1={:.3}  ttft={} \u{00b1}{}  total={}",
        plain_acc * 100.0,
        plain_f1,
        fmt_ms(plain_ttft),
        fmt_ms(plain_ttft_stddev),
        fmt_ms(plain_total),
    );
    println!(
        "  {BOLD}Spans{RST}:  acc={:.1}%  F1={:.3}  ttft={} \u{00b1}{}  total={}",
        spans_acc * 100.0,
        spans_f1,
        fmt_ms(spans_ttft),
        fmt_ms(spans_ttft_stddev),
        fmt_ms(spans_total),
    );
    println!();
    println!(
        "  TTFT speedup: {BOLD}{:.2}x{RST}  {DIM}(plain/spans){RST}",
        plain_ttft / spans_ttft.max(0.001),
    );
    println!(
        "  Latency stability: plain \u{00b1}{} vs spans \u{00b1}{}",
        fmt_ms(plain_ttft_stddev),
        fmt_ms(spans_ttft_stddev),
    );
    println!();

    Ok(())
}

/// Format milliseconds as a human-readable duration (e.g. "7.4s", "238ms").
fn fmt_ms(ms: f64) -> String {
    if ms >= 1000.0 {
        format!("{:.1}s", ms / 1000.0)
    } else {
        format!("{:.0}ms", ms)
    }
}

/// Compute the p-th percentile from a pre-sorted slice.
#[cfg(feature = "rag")]
fn percentile(sorted: &[f64], p: f64) -> f64 {
    if sorted.is_empty() {
        return 0.0;
    }
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

/// Build a span query using SPNL structs (cross + plus pattern).
fn build_span_query(
    model: &str,
    max_tokens: i32,
    system_prompt: &str,
    frags: &[String],
    order: &[usize],
    question: &str,
) -> SpnlQuery {
    let passage_nodes: Vec<SpnlQuery> = order
        .iter()
        .map(|&idx| SpnlQuery::Message(Message::User(frags[idx].clone())))
        .collect();

    SpnlQuery::Generate(Generate {
        metadata: GenerateMetadata {
            model: model.to_string(),
            max_tokens: Some(max_tokens),
            temperature: Some(0.0),
        },
        input: Box::new(SpnlQuery::Cross(vec![
            SpnlQuery::Message(Message::System(system_prompt.to_string())),
            SpnlQuery::Plus(passage_nodes),
            SpnlQuery::Message(Message::User(question.to_string())),
        ])),
    })
}

/// Average per-query standard deviation of a metric across permutations.
fn per_query_stddev(results: &[Vec<PermResult>], f: fn(&PermResult) -> f64) -> f64 {
    if results.is_empty() {
        return 0.0;
    }
    let mut total_stddev = 0.0;
    let mut count = 0;
    for query_results in results {
        let n = query_results.len() as f64;
        if n < 2.0 {
            continue;
        }
        let mean = query_results.iter().map(&f).sum::<f64>() / n;
        let var = query_results
            .iter()
            .map(|r| (f(r) - mean).powi(2))
            .sum::<f64>()
            / (n - 1.0);
        total_stddev += var.sqrt();
        count += 1;
    }
    if count > 0 {
        total_stddev / count as f64
    } else {
        0.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn span_query_structure() {
        let query = build_span_query(
            "test-model",
            64,
            "You are helpful.",
            &[
                "Frag A".to_string(),
                "Frag B".to_string(),
                "Frag C".to_string(),
            ],
            &[2, 0, 1],
            "What is X?",
        );

        // Top-level must be Generate.
        match &query {
            SpnlQuery::Generate(g) => {
                assert_eq!(g.metadata.model, "test-model");
                assert_eq!(g.metadata.max_tokens, Some(64));
                // Input must be Cross with 3 children.
                match g.input.as_ref() {
                    SpnlQuery::Cross(children) => {
                        assert_eq!(children.len(), 3);
                        assert!(matches!(
                            &children[0],
                            SpnlQuery::Message(Message::System(_))
                        ));
                        match &children[1] {
                            SpnlQuery::Plus(frags) => {
                                assert_eq!(frags.len(), 3);
                                // Permuted order: [2, 0, 1] -> ["Frag C", "Frag A", "Frag B"]
                                assert!(
                                    matches!(&frags[0], SpnlQuery::Message(Message::User(s)) if s == "Frag C")
                                );
                                assert!(
                                    matches!(&frags[1], SpnlQuery::Message(Message::User(s)) if s == "Frag A")
                                );
                                assert!(
                                    matches!(&frags[2], SpnlQuery::Message(Message::User(s)) if s == "Frag B")
                                );
                            }
                            other => panic!("expected Plus, got {other:?}"),
                        }
                        assert!(
                            matches!(&children[2], SpnlQuery::Message(Message::User(s)) if s == "What is X?")
                        );
                    }
                    other => panic!("expected Cross, got {other:?}"),
                }
            }
            other => panic!("expected Generate, got {other:?}"),
        }

        // Must also round-trip through serde.
        let json = serde_json::to_string(&query).unwrap();
        let parsed: SpnlQuery = serde_json::from_str(&json).expect("should round-trip");
        assert!(matches!(parsed, SpnlQuery::Generate(_)));
    }

    #[cfg(feature = "rag")]
    #[test]
    fn augment_query_structure() {
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

        // Must round-trip through serde.
        let json = serde_json::to_string(&query).unwrap();
        let parsed: SpnlQuery = serde_json::from_str(&json).expect("should round-trip");
        assert!(matches!(parsed, SpnlQuery::Generate(_)));
    }

    #[test]
    fn permutations_small() {
        let perms = permutations(3, 100);
        assert_eq!(perms.len(), 6); // 3! = 6
        // All permutations should be unique.
        let mut sorted = perms.clone();
        sorted.sort();
        sorted.dedup();
        assert_eq!(sorted.len(), 6);
    }

    #[test]
    fn permutations_sampled() {
        let perms = permutations(10, 5);
        assert_eq!(perms.len(), 5);
        // First should be canonical, second reversed.
        assert_eq!(perms[0], (0..10).collect::<Vec<_>>());
        assert_eq!(perms[1], (0..10).rev().collect::<Vec<_>>());
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
