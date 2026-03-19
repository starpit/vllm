// SPDX-License-Identifier: Apache-2.0
// Copyright contributors to the vLLM project

//! `vllm bench spans` — benchmark relocatable KV cache blocks (spans).
//!
//! Tests all permutations of document ordering to prove that span-enabled
//! prefix caching allows KV cache reuse regardless of order.

use std::time::Instant;

use anyhow::Result;
use indicatif::{ProgressBar, ProgressStyle};
use vllm_config::{CudaGraphConfig, CudaGraphMode};
use vllm_serve::llm::{LLM, LLMBuilder, Prompt, SamplingParams};

use crate::args::BenchSpansArgs;

// ---------------------------------------------------------------------------
// Colors — ANSI 256-color palette for up to 12 distinct documents
// ---------------------------------------------------------------------------

const DOC_COLORS: &[&str] = &[
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
const RST: &str = "\x1b[0m";
const DIM: &str = "\x1b[2m";
const BOLD: &str = "\x1b[1m";
const BLOCK_CHAR: char = '\u{2588}'; // full block: █

/// Render a permutation as color-coded block characters.
fn render_perm(perm: &[usize], _doc_blocks: usize) -> String {
    let mut s = String::new();
    for &doc_idx in perm {
        let color = DOC_COLORS[doc_idx % DOC_COLORS.len()];
        s.push_str(color);
        s.push(BLOCK_CHAR);
    }
    s.push_str(RST);
    s
}

/// Render the document legend.
fn render_legend(num_docs: usize, _doc_blocks: usize) -> String {
    let mut s = String::new();
    for i in 0..num_docs {
        if i > 0 {
            s.push_str("  ");
        }
        let color = DOC_COLORS[i % DOC_COLORS.len()];
        s.push_str(&format!("Doc {i}="));
        s.push_str(color);
        s.push(BLOCK_CHAR);
        s.push_str(RST);
    }
    s
}

// ---------------------------------------------------------------------------
// Permutation generation
// ---------------------------------------------------------------------------

fn permutations(n: usize, max_perms: usize) -> Vec<Vec<usize>> {
    if n <= 1 {
        return vec![(0..n).collect()];
    }
    let total: usize = (1..=n).product();
    if total <= max_perms {
        // Heap's algorithm — all permutations.
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
        // Sample random permutations.
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
// Prompt construction
// ---------------------------------------------------------------------------

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

fn make_document(
    doc_id: u32,
    block_size: usize,
    doc_blocks: usize,
    span_token: u32,
    pad_token: u32,
) -> Vec<u32> {
    let mut tokens = Vec::with_capacity(block_size * doc_blocks);
    for b in 0..doc_blocks {
        tokens.push(span_token);
        for j in 1..block_size {
            tokens.push(1000 + doc_id * 1000 + b as u32 * 100 + j as u32);
        }
    }
    pad_to_block(&tokens, block_size, pad_token)
}

fn build_prompt(
    documents: &[Vec<u32>],
    order: &[usize],
    query_base: u32,
    query_len: usize,
) -> Vec<u32> {
    let mut prompt = Vec::new();
    for &doc_idx in order {
        prompt.extend_from_slice(&documents[doc_idx]);
    }
    for j in 0..query_len {
        prompt.push(query_base + j as u32);
    }
    prompt
}

// ---------------------------------------------------------------------------
// LLM construction
// ---------------------------------------------------------------------------

fn build_llm(args: &BenchSpansArgs, prefix_caching: bool) -> Result<LLM> {
    let model = args.resolved_model().map_err(|e| anyhow::anyhow!(e))?;
    let mut builder = LLMBuilder::new(&model)
        .device(&args.device)
        .dtype(&args.dtype)
        .gpu_memory_utilization(args.gpu_memory_utilization)
        .max_num_seqs(args.max_num_seqs)
        .block_size(args.block_size)
        .enforce_eager(args.enforce_eager)
        .enable_prefix_caching(prefix_caching);

    let total_doc_tokens = args.num_docs * args.doc_blocks * args.block_size;
    let max_batched = total_doc_tokens + args.query_len + 512;
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
                mode: CudaGraphMode::default(),
                capture_sizes: sizes,
                num_warmups: 3,
            });
        }
    }

    builder.build()
}

// ---------------------------------------------------------------------------
// Main benchmark
// ---------------------------------------------------------------------------

pub(crate) fn run_bench_spans(args: BenchSpansArgs) -> Result<()> {
    vllm_common::telemetry::init_tracing(&args.log_level);

    let block_size = args.block_size;
    let num_docs = args.num_docs;
    let doc_blocks = args.doc_blocks;
    let query_len = args.query_len;
    let span_token = args.span_token;
    let pad_token = args.pad_token;
    let doc_tokens = doc_blocks * block_size;
    let total_cached = num_docs * doc_tokens;
    let max_perms = args.max_perms;

    // Documents WITH span tokens (order-independent caching).
    let docs_with_spans: Vec<Vec<u32>> = (0..num_docs as u32)
        .map(|i| make_document(i, block_size, doc_blocks, span_token, pad_token))
        .collect();

    // Documents WITHOUT span tokens (normal prefix caching, order-dependent).
    // Use a regular filler token instead of the span token so block hashes
    // chain normally — reordering breaks cache hits.
    let no_span_filler = span_token.wrapping_add(1); // any token that isn't span_token
    let docs_no_spans: Vec<Vec<u32>> = (0..num_docs as u32)
        .map(|i| make_document(i, block_size, doc_blocks, no_span_filler, pad_token))
        .collect();

    let perms = permutations(num_docs, max_perms);
    let num_perms = perms.len();
    let total_factorial: usize = (1..=num_docs).product();

    let sampling = SamplingParams {
        max_tokens: Some(1),
        temperature: 0.0,
        ignore_eos: true,
        detokenize: false,
        ..SamplingParams::default()
    };

    let canonical: Vec<usize> = (0..num_docs).collect();

    let spinner_style = ProgressStyle::with_template("  {spinner:.cyan} {msg}")
        .unwrap()
        .tick_strings(&[
            "\u{28fb}", "\u{28fd}", "\u{28fe}", "\u{28f7}", "\u{28ef}", "\u{28df}", "\u{287f}",
            "\u{28bf}", "\u{2847}", "\u{280b}", "\u{281b}", "\u{2839}", "\u{2838}",
        ]);

    // -- Header --
    eprintln!();
    eprintln!("{BOLD}vLLM Rust \u{2014} spans benchmark{RST}");
    eprintln!(
        "  {num_docs} docs x {doc_tokens} tok/doc ({total_cached} cached) + {query_len} query"
    );
    eprintln!("  {}", render_legend(num_docs, doc_blocks));
    if num_perms < total_factorial {
        eprintln!("  Testing {num_perms}/{total_factorial} permutations (sampled)");
    } else {
        eprintln!("  Testing all {num_perms} permutations");
    }

    // -----------------------------------------------------------------------
    // Helper: run a sequence of permutations and return per-perm latencies.
    // First request (canonical) populates cache; remaining are measured.
    // -----------------------------------------------------------------------
    let run_perms = |llm: &mut LLM,
                     docs: &[Vec<u32>],
                     perms: &[Vec<usize>],
                     query_base_start: u32,
                     label: &str|
     -> Result<(f64, Vec<f64>)> {
        // Populate cache with canonical order.
        llm.reset_prefix_cache()?;
        let perm_str = render_perm(&canonical, doc_blocks);
        let pb = ProgressBar::new_spinner()
            .with_style(spinner_style.clone())
            .with_message(format!("{perm_str}  {DIM}populate ({label})...{RST}"));
        pb.enable_steady_tick(std::time::Duration::from_millis(80));

        let populate_prompt = build_prompt(docs, &canonical, query_base_start, query_len);
        let pop_start = Instant::now();
        llm.generate(&[Prompt::TokenIds(populate_prompt)], Some(sampling.clone()))?;
        let populate_ms = pop_start.elapsed().as_secs_f64() * 1000.0;

        pb.finish_and_clear();
        eprintln!("    {perm_str}  {BOLD}{populate_ms:>8.1}ms{RST}  {DIM}populate ({label}){RST}");

        // Run each permutation and measure.
        let mut latencies = Vec::with_capacity(perms.len());
        for (pi, perm) in perms.iter().enumerate() {
            let perm_str = render_perm(perm, doc_blocks);
            let query_base = query_base_start + 1000 + (pi as u32) * 1000;

            let pb = ProgressBar::new_spinner()
                .with_style(spinner_style.clone())
                .with_message(format!(
                    "{perm_str}  {DIM}{}/{} ({label})...{RST}",
                    pi + 1,
                    perms.len()
                ));
            pb.enable_steady_tick(std::time::Duration::from_millis(80));

            let prompt = build_prompt(docs, perm, query_base, query_len);
            let start = Instant::now();
            llm.generate(&[Prompt::TokenIds(prompt)], Some(sampling.clone()))?;
            let ms = start.elapsed().as_secs_f64() * 1000.0;

            pb.finish_and_clear();
            eprintln!("    {perm_str}  {BOLD}{ms:>8.1}ms{RST}  {DIM}{label}{RST}");
            latencies.push(ms);
        }
        Ok((populate_ms, latencies))
    };

    // -----------------------------------------------------------------------
    // Without spans: documents use a regular token instead of span_token.
    // Normal prefix caching — block hashes chain by position, so reordering
    // documents breaks cache hits.
    // -----------------------------------------------------------------------
    eprintln!();
    eprintln!("{BOLD}Without spans{RST} {DIM}(prefix caching, order-dependent){RST}");

    unsafe {
        std::env::set_var("VLLM_V1_SPANS_ENABLED", "true");
        std::env::set_var("VLLM_V1_SPANS_TOKEN_PLUS", span_token.to_string());
    }

    let mut llm = build_llm(&args, true)?;
    // Skip perms[0] (canonical order) — it's the populate step and would
    // always cache-hit even without spans, biasing results.
    let test_perms: Vec<Vec<usize>> = perms.iter().filter(|p| *p != &canonical).cloned().collect();
    let (no_spans_populate, no_spans_latencies) =
        run_perms(&mut llm, &docs_no_spans, &test_perms, 50000, "no spans")?;

    // -----------------------------------------------------------------------
    // With spans: documents use span_token at block boundaries.
    // Span-aware hashing resets parent chain — blocks cache independently
    // of position, so reordering still gets full cache hits.
    // -----------------------------------------------------------------------
    eprintln!();
    eprintln!("{BOLD}With spans{RST} {DIM}(prefix caching, order-independent){RST}");

    let (spans_populate, spans_latencies) =
        run_perms(&mut llm, &docs_with_spans, &test_perms, 60000, "spans")?;

    drop(llm);

    unsafe {
        std::env::remove_var("VLLM_V1_SPANS_ENABLED");
        std::env::remove_var("VLLM_V1_SPANS_TOKEN_PLUS");
    }

    // -----------------------------------------------------------------------
    // Summary
    // -----------------------------------------------------------------------
    // Compute per-permutation speedups: no_spans[i] / spans[i]
    let mut speedups: Vec<f64> = no_spans_latencies
        .iter()
        .zip(spans_latencies.iter())
        .map(|(&ns, &sp)| ns / sp)
        .collect();
    speedups.sort_by(|a, b| a.partial_cmp(b).unwrap());

    let p50 = |v: &[f64]| {
        let mid = v.len() / 2;
        if v.len() % 2 == 0 && v.len() > 1 {
            (v[mid - 1] + v[mid]) / 2.0
        } else {
            v[mid]
        }
    };

    let no_spans_avg = no_spans_latencies.iter().sum::<f64>() / no_spans_latencies.len() as f64;
    let spans_avg = spans_latencies.iter().sum::<f64>() / spans_latencies.len() as f64;

    eprintln!();
    println!("{BOLD}=== Results ({} perms) ==={RST}", test_perms.len());
    println!();
    println!(
        "  Without spans:  avg {no_spans_avg:>7.1}ms  {DIM}({:.1}x vs populate){RST}",
        no_spans_populate / no_spans_avg
    );
    println!(
        "  With spans:     avg {spans_avg:>7.1}ms  {DIM}({:.1}x vs populate){RST}",
        spans_populate / spans_avg
    );
    println!();
    println!(
        "  {BOLD}Spans speedup{RST}  min {BOLD}{:.1}x{RST}  p50 {BOLD}{:.1}x{RST}  max {BOLD}{:.1}x{RST}",
        speedups.first().unwrap_or(&0.0),
        p50(&speedups),
        speedups.last().unwrap_or(&0.0),
    );
    println!();

    Ok(())
}
