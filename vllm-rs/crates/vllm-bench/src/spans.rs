// SPDX-License-Identifier: Apache-2.0
// Copyright contributors to the vLLM project

//! `vllm bench spans` — benchmark relocatable KV cache blocks (spans).
//!
//! Demonstrates that span-enabled prefix caching allows KV cache reuse
//! regardless of document ordering, while standard prefix caching only
//! hits when the prefix is identical.
//!
//! **Workload:** Simulates RAG with N preloaded documents:
//!   1. Prefill `[doc_0, doc_1, ..., doc_{N-1}, query_0]` — populate cache.
//!   2. Prefill `[doc_{N-1}, ..., doc_0, query_1]` — reversed order.
//!
//! With spans enabled, step 2 gets full cache hits on all document blocks
//! (fan-in hashing makes blocks order-independent). Without spans, step 2
//! is a complete cache miss because the prefix differs.

use std::time::Instant;

use anyhow::Result;
use vllm_config::CudaGraphConfig;
use vllm_serve::llm::{LLM, LLMBuilder, Prompt, SamplingParams};

use crate::args::BenchSpansArgs;

/// Pad a token sequence to a multiple of `block_size` using `pad_token`.
fn pad_to_block(tokens: &[u32], block_size: usize, pad_token: u32) -> Vec<u32> {
    let remainder = tokens.len() % block_size;
    if remainder == 0 {
        return tokens.to_vec();
    }
    let pad_count = block_size - remainder;
    let mut padded = tokens.to_vec();
    padded.extend(std::iter::repeat_n(pad_token, pad_count));
    padded
}

/// Build a document block: [span_token, content_tokens..., pad...]
/// Padded to exactly `block_size` tokens.
fn make_document(doc_id: u32, block_size: usize, span_token: u32, pad_token: u32) -> Vec<u32> {
    // Fill the block with span_token + deterministic content.
    let mut tokens = Vec::with_capacity(block_size);
    tokens.push(span_token);
    for j in 1..block_size {
        tokens.push(1000 + doc_id * 100 + j as u32);
    }
    pad_to_block(&tokens, block_size, pad_token)
}

/// Build an LLM with prefix caching disabled (baseline: full recompute).
fn create_llm_no_prefix_cache(args: &BenchSpansArgs) -> Result<LLM> {
    let model = args.resolved_model().map_err(|e| anyhow::anyhow!(e))?;
    let mut builder = LLMBuilder::new(&model)
        .device(&args.device)
        .dtype(&args.dtype)
        .gpu_memory_utilization(args.gpu_memory_utilization)
        .max_num_seqs(args.max_num_seqs)
        .block_size(args.block_size)
        .enforce_eager(args.enforce_eager)
        .enable_prefix_caching(false); // no caching

    let max_batched = args.num_docs * args.block_size + args.query_len + 512;
    builder = builder.max_num_batched_tokens(max_batched.max(8192));

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
                capture_sizes: sizes,
                num_warmups: 3,
            });
        }
    }

    builder.build()
}

/// Build an LLM from bench args (with prefix caching enabled).
fn create_llm(args: &BenchSpansArgs) -> Result<LLM> {
    let model = args.resolved_model().map_err(|e| anyhow::anyhow!(e))?;
    let mut builder = LLMBuilder::new(&model)
        .device(&args.device)
        .dtype(&args.dtype)
        .gpu_memory_utilization(args.gpu_memory_utilization)
        .max_num_seqs(args.max_num_seqs)
        .block_size(args.block_size)
        .enforce_eager(args.enforce_eager)
        .enable_prefix_caching(true);

    let max_batched = args.num_docs * args.block_size + args.query_len + 512;
    builder = builder.max_num_batched_tokens(max_batched.max(8192));

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
                capture_sizes: sizes,
                num_warmups: 3,
            });
        }
    }

    builder.build()
}

/// Run the spans benchmark.
///
/// Measures prefill latency for a "reordered RAG" workload with and without
/// span-aware prefix caching. Runs both configurations side-by-side using
/// environment variables to toggle spans.
pub(crate) fn run_bench_spans(args: BenchSpansArgs) -> Result<()> {
    vllm_common::telemetry::init_tracing(&args.log_level);

    let block_size = args.block_size;
    let num_docs = args.num_docs;
    let query_len = args.query_len;
    let num_iters = args.num_iters;
    let span_token = args.span_token;
    let pad_token = args.pad_token;

    eprintln!("vLLM Rust — spans benchmark");
    eprintln!(
        "  num_docs={num_docs}, block_size={block_size}, query_len={query_len}, iters={num_iters}"
    );
    eprintln!("  span_token={span_token}, pad_token={pad_token}");

    // Build documents: each is exactly one block.
    let documents: Vec<Vec<u32>> = (0..num_docs as u32)
        .map(|i| make_document(i, block_size, span_token, pad_token))
        .collect();

    // Build prompts per iteration with unique queries (so cache doesn't
    // carry over between iterations).
    let make_prompts = |iter: usize| -> (Vec<u32>, Vec<u32>) {
        let base_a = 5000 + (iter as u32) * 1000;
        let base_b = 6000 + (iter as u32) * 1000;
        let query_a: Vec<u32> = (0..query_len).map(|j| base_a + j as u32).collect();
        let query_b: Vec<u32> = (0..query_len).map(|j| base_b + j as u32).collect();

        // Prompt 1: [doc_0, doc_1, ..., doc_{N-1}, query_a]
        let mut forward = Vec::new();
        for doc in &documents {
            forward.extend_from_slice(doc);
        }
        forward.extend_from_slice(&query_a);

        // Prompt 2: [doc_{N-1}, ..., doc_0, query_b] (reversed doc order)
        let mut reversed = Vec::new();
        for doc in documents.iter().rev() {
            reversed.extend_from_slice(doc);
        }
        reversed.extend_from_slice(&query_b);

        (forward, reversed)
    };

    let sampling = SamplingParams {
        max_tokens: Some(1), // We only care about prefill time
        temperature: 0.0,
        ignore_eos: true,
        detokenize: false,
        ..SamplingParams::default()
    };

    // --- Run without spans (baseline) ---
    // Baseline: prefix caching disabled entirely. The reversed prompt must
    // recompute all tokens from scratch.
    eprintln!("\n--- Baseline (no prefix caching) ---");

    // SAFETY: env vars set before LLM construction, single-threaded bench.
    unsafe {
        std::env::set_var("VLLM_V1_SPANS_ENABLED", "false");
    }

    let mut llm_baseline = create_llm_no_prefix_cache(&args)?;
    let mut baseline_latencies = Vec::with_capacity(num_iters);

    for iter in 0..num_iters {
        let (_, prompt_reversed) = make_prompts(iter);

        // No step 1 needed — there's no cache to populate.
        // Just measure full prefill of the reversed prompt.
        let start = Instant::now();
        llm_baseline.generate(&[Prompt::TokenIds(prompt_reversed)], Some(sampling.clone()))?;
        let elapsed = start.elapsed().as_secs_f64() * 1000.0; // ms
        baseline_latencies.push(elapsed);

        if iter == 0 {
            eprintln!("  iter {}: full prefill = {elapsed:.2} ms", iter + 1);
        }
    }
    drop(llm_baseline);

    // --- Run with spans ---
    eprintln!("\n--- Spans enabled ---");

    unsafe {
        std::env::set_var("VLLM_V1_SPANS_ENABLED", "true");
        std::env::set_var("VLLM_V1_SPANS_TOKEN_PLUS", span_token.to_string());
    }

    let mut llm_spans = create_llm(&args)?;
    let mut spans_latencies = Vec::with_capacity(num_iters);

    for iter in 0..num_iters {
        let (prompt_forward, prompt_reversed) = make_prompts(iter);

        // Step 1: populate cache with forward ordering.
        llm_spans.generate(&[Prompt::TokenIds(prompt_forward)], Some(sampling.clone()))?;

        // Step 2: measure prefill with reversed ordering.
        let start = Instant::now();
        llm_spans.generate(&[Prompt::TokenIds(prompt_reversed)], Some(sampling.clone()))?;
        let elapsed = start.elapsed().as_secs_f64() * 1000.0;
        spans_latencies.push(elapsed);

        if iter == 0 {
            eprintln!("  iter {}: reversed prefill = {elapsed:.2} ms", iter + 1);
        }
    }
    drop(llm_spans);

    // Restore env.
    unsafe {
        std::env::remove_var("VLLM_V1_SPANS_ENABLED");
        std::env::remove_var("VLLM_V1_SPANS_TOKEN_PLUS");
    }

    // --- Results ---
    let avg = |v: &[f64]| v.iter().sum::<f64>() / v.len() as f64;
    let baseline_avg = avg(&baseline_latencies);
    let spans_avg = avg(&spans_latencies);
    let speedup = baseline_avg / spans_avg;

    println!();
    println!("=== Spans Benchmark Results ===");
    println!("Workload: {num_docs} docs × {block_size} tokens + {query_len} query tokens");
    println!("Prefill latency for reversed doc order (ms):");
    println!("  Baseline = no caching, full recompute");
    println!("  Spans    = per-block KV cache reuse (only query recomputed)");
    println!();
    println!("  {:<20} {:>10} {:>10}", "", "Baseline", "Spans");
    println!("  {:-<20} {:-<10} {:-<10}", "", "", "");
    println!(
        "  {:<20} {:>10.2} {:>10.2}",
        "Avg (ms)", baseline_avg, spans_avg
    );
    if baseline_latencies.len() > 1 {
        let min = |v: &[f64]| v.iter().cloned().fold(f64::INFINITY, f64::min);
        let max = |v: &[f64]| v.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        println!(
            "  {:<20} {:>10.2} {:>10.2}",
            "Min (ms)",
            min(&baseline_latencies),
            min(&spans_latencies)
        );
        println!(
            "  {:<20} {:>10.2} {:>10.2}",
            "Max (ms)",
            max(&baseline_latencies),
            max(&spans_latencies)
        );
    }
    println!();
    println!("  Speedup: {speedup:.2}x");
    if speedup > 1.5 {
        println!("  Spans enabled full cache reuse for reordered documents.");
    } else {
        println!("  Note: speedup may be limited by model size, query length, or block size.");
    }
    println!();

    Ok(())
}
