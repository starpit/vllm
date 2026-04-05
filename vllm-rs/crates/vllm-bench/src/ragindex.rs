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

    // -- Index phase: build Augment query for each sample to trigger LEANN --
    let mut retrieved_fragments: Vec<Vec<String>> = Vec::with_capacity(n_queries);

    #[cfg(feature = "rag")]
    {
        eprintln!();
        eprintln!("{BOLD}Phase 1: Index + Retrieve{RST}");

        let pb_style = ProgressStyle::default_bar()
            .template("  index {bar:40.green/green} {pos:>4}/{len} {msg}")
            .unwrap();
        let pb = ProgressBar::new(n_queries as u64).with_style(pb_style);
        let mut index_total_ms = 0.0f64;

        for (qi, sample) in samples.iter().enumerate() {
            let debug = args.debug && qi == 0;

            // Build corpus text as a single document from all passages.
            let corpus_text: String = sample
                .documents
                .iter()
                .map(|(label, text)| format!("{label}: {text}"))
                .collect::<Vec<_>>()
                .join("\n\n");

            // Build the Augment SPNL query using proper structs.
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
                        doc: (format!("query_{qi}.txt"), Document::Text(corpus_text)),
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
            index_total_ms += ms;

            let response = &result.output().outputs[0].text;
            if debug {
                eprintln!("  [debug] query: {}", sample.question);
                eprintln!("  [debug] response: {response}");
            }

            // For permutation testing, use the original documents (up to max_aug).
            let frags: Vec<String> = sample
                .documents
                .iter()
                .take(args.max_aug)
                .map(|(label, text)| format!("{label}: {text}"))
                .collect();
            retrieved_fragments.push(frags);

            pb.set_message(format!("{ms:.0}ms"));
            pb.inc(1);
        }
        pb.finish();

        eprintln!(
            "  Total index+generate time: {BOLD}{index_total_ms:.0}ms{RST} ({:.0}ms/query)",
            index_total_ms / n_queries as f64,
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

    let query_style = ProgressStyle::default_bar()
        .template("  {msg} {bar:40} {pos:>4}/{len}")
        .unwrap();
    let pb = ProgressBar::new(n_queries as u64).with_style(query_style);

    for (qi, sample) in samples.iter().enumerate() {
        let frags = &retrieved_fragments[qi];
        let n_frags = frags.len();
        let perms = permutations(n_frags, args.max_perms);

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

            plain_results.push(PermResult {
                acc: evaluate_accuracy(response, &sample.answers),
                f1: best_token_f1(&sample.answers, response),
                ttft_ms,
                total_ms,
            });
        }

        // --- Spans: execute_query with cross + plus, relocatable ---
        llm.reset_prefix_cache()?;

        // First, populate the cache with canonical order.
        {
            let span_query = build_span_query(
                &model_name,
                args.max_tokens as i32,
                system_prompt,
                frags,
                &(0..n_frags).collect::<Vec<_>>(),
                &sample.question,
            );
            llm.execute_spnl(span_query, Some(sampling.clone()), false, false)?;
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

            spans_results.push(PermResult {
                acc: evaluate_accuracy(response, &sample.answers),
                f1: best_token_f1(&sample.answers, response),
                ttft_ms,
                total_ms,
            });
        }

        all_plain.push(plain_results);
        all_spans.push(spans_results);

        let pa = all_plain
            .iter()
            .flat_map(|v| v.iter())
            .map(|r| r.acc)
            .sum::<f64>()
            / all_plain.iter().map(|v| v.len()).sum::<usize>().max(1) as f64;
        let sa = all_spans
            .iter()
            .flat_map(|v| v.iter())
            .map(|r| r.acc)
            .sum::<f64>()
            / all_spans.iter().map(|v| v.len()).sum::<usize>().max(1) as f64;
        pb.set_message(format!(
            "plain={:.0}%  spans={:.0}%",
            pa * 100.0,
            sa * 100.0,
        ));
        pb.inc(1);
    }
    pb.finish();

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
        "  {BOLD}Plain{RST}:  acc={:.1}%  F1={:.3}  ttft={:.0}ms \u{00b1}{:.0}ms  total={:.0}ms",
        plain_acc * 100.0,
        plain_f1,
        plain_ttft,
        plain_ttft_stddev,
        plain_total,
    );
    println!(
        "  {BOLD}Spans{RST}:  acc={:.1}%  F1={:.3}  ttft={:.0}ms \u{00b1}{:.0}ms  total={:.0}ms",
        spans_acc * 100.0,
        spans_f1,
        spans_ttft,
        spans_ttft_stddev,
        spans_total,
    );
    println!();
    println!(
        "  TTFT speedup: {BOLD}{:.2}x{RST}  {DIM}(plain/spans){RST}",
        plain_ttft / spans_ttft.max(0.001),
    );
    println!(
        "  Latency stability: plain \u{00b1}{:.0}ms vs spans \u{00b1}{:.0}ms",
        plain_ttft_stddev, spans_ttft_stddev,
    );
    println!();

    // Clean up index files created during the benchmark.
    let _ = std::fs::remove_dir_all("data/spnl");

    Ok(())
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
