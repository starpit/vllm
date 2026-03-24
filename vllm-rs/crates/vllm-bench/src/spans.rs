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

fn make_document(doc_id: u32, block_size: usize, doc_blocks: usize, pad_token: u32) -> Vec<u32> {
    let mut tokens = Vec::with_capacity(block_size * doc_blocks);
    for b in 0..doc_blocks {
        for j in 0..block_size {
            tokens.push(1000 + doc_id * 1000 + b as u32 * 100 + j as u32);
        }
    }
    pad_to_block(&tokens, block_size, pad_token)
}

/// Build a prompt from ordered documents + query.
/// When `with_annotations` is true, each document's blocks are annotated as
/// `Relocatable` for span-aware caching.
fn build_prompt(
    documents: &[Vec<u32>],
    order: &[usize],
    query_base: u32,
    query_len: usize,
    block_size: usize,
    with_annotations: bool,
) -> Prompt {
    let mut tokens = Vec::new();
    let mut annotations = std::collections::BTreeMap::new();

    for &doc_idx in order {
        let doc = &documents[doc_idx];
        let start_block = tokens.len() / block_size;
        tokens.extend_from_slice(doc);
        if with_annotations {
            let end_block = tokens.len() / block_size;
            for b in start_block..end_block {
                annotations.insert(b, vllm_common::BlockKind::Relocatable);
            }
        }
    }
    for j in 0..query_len {
        tokens.push(query_base + j as u32);
    }

    if with_annotations && !annotations.is_empty() {
        Prompt::TokenIdsWithAnnotations(tokens, annotations)
    } else {
        Prompt::TokenIds(tokens)
    }
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

    if let Some(n) = args.nested {
        return run_bench_nested(&args, n);
    }

    let block_size = args.block_size;
    let num_docs = args.num_docs;
    let doc_blocks = args.doc_blocks;
    let query_len = args.query_len;
    let pad_token = args.pad_token;
    let doc_tokens = doc_blocks * block_size;
    let total_cached = num_docs * doc_tokens;
    let max_perms = args.max_perms;

    // All documents use the same token data — the difference is whether
    // block annotations are provided (relocatable caching) or not.
    let docs: Vec<Vec<u32>> = (0..num_docs as u32)
        .map(|i| make_document(i, block_size, doc_blocks, pad_token))
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
                     label: &str,
                     with_annotations: bool|
     -> Result<(f64, Vec<f64>)> {
        // Populate cache with canonical order.
        llm.reset_prefix_cache()?;
        let perm_str = render_perm(&canonical, doc_blocks);
        let pb = ProgressBar::new_spinner()
            .with_style(spinner_style.clone())
            .with_message(format!("{perm_str}  {DIM}populate ({label})...{RST}"));
        pb.enable_steady_tick(std::time::Duration::from_millis(80));

        let populate_prompt = build_prompt(
            docs,
            &canonical,
            query_base_start,
            query_len,
            block_size,
            with_annotations,
        );
        let pop_start = Instant::now();
        llm.generate(&[populate_prompt], Some(sampling.clone()))?;
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

            let prompt = build_prompt(
                docs,
                perm,
                query_base,
                query_len,
                block_size,
                with_annotations,
            );
            let start = Instant::now();
            llm.generate(&[prompt], Some(sampling.clone()))?;
            let ms = start.elapsed().as_secs_f64() * 1000.0;

            pb.finish_and_clear();
            eprintln!("    {perm_str}  {BOLD}{ms:>8.1}ms{RST}  {DIM}{label}{RST}");
            latencies.push(ms);
        }
        Ok((populate_ms, latencies))
    };

    // -----------------------------------------------------------------------
    // Without spans: no block annotations.
    // Normal prefix caching — block hashes chain by position, so reordering
    // documents breaks cache hits.
    // -----------------------------------------------------------------------
    eprintln!();
    eprintln!("{BOLD}Without spans{RST} {DIM}(prefix caching, order-dependent){RST}");

    let mut llm = build_llm(&args, true)?;
    // Skip perms[0] (canonical order) — it's the populate step and would
    // always cache-hit even without spans, biasing results.
    let test_perms: Vec<Vec<usize>> = perms.iter().filter(|p| *p != &canonical).cloned().collect();
    let (no_spans_populate, no_spans_latencies) =
        run_perms(&mut llm, &docs, &test_perms, 50000, "no spans", false)?;

    // -----------------------------------------------------------------------
    // With spans: documents have Relocatable block annotations.
    // Span-aware hashing resets parent chain — blocks cache independently
    // of position, so reordering still gets full cache hits.
    // -----------------------------------------------------------------------
    eprintln!();
    eprintln!("{BOLD}With spans{RST} {DIM}(prefix caching, order-independent){RST}");

    let (spans_populate, spans_latencies) =
        run_perms(&mut llm, &docs, &test_perms, 60000, "spans", true)?;

    drop(llm);

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
        if v.len().is_multiple_of(2) && v.len() > 1 {
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

// ---------------------------------------------------------------------------
// Nested generate benchmark
// ---------------------------------------------------------------------------

fn run_bench_nested(args: &BenchSpansArgs, num_inner: usize) -> Result<()> {
    let block_size = args.block_size;
    let pad_token = args.pad_token;
    let query_len = args.query_len;
    let inner_tokens = args.inner_tokens;

    let mut llm = build_llm(args, true)?;

    let inner_sampling = SamplingParams {
        max_tokens: Some(inner_tokens),
        temperature: 0.0,
        ignore_eos: true,
        detokenize: false,
        ..SamplingParams::default()
    };

    let outer_sampling = SamplingParams {
        max_tokens: Some(1),
        temperature: 0.0,
        ignore_eos: true,
        detokenize: false,
        ..SamplingParams::default()
    };

    eprintln!();
    eprintln!("{BOLD}vLLM Rust \u{2014} nested spans benchmark{RST}");
    eprintln!("  {num_inner} inner generates x {inner_tokens} tokens each + {query_len} query");
    eprintln!();

    // -----------------------------------------------------------------------
    // Step 1: Run N inner generates with seal=true.
    // Each inner generate's output is padded and cached so the outer
    // generate can hit it.
    // -----------------------------------------------------------------------
    eprintln!("{BOLD}Step 1:{RST} Run {num_inner} inner generates (seal=true)");

    // Store full inner sequences (prompt + output) for the outer prompt.
    let mut inner_sequences: Vec<Vec<u32>> = Vec::with_capacity(num_inner);
    for i in 0..num_inner {
        // Each inner generate has a unique synthetic prompt (one block).
        let prompt_tokens: Vec<u32> = (0..block_size)
            .map(|j| 2000 + i as u32 * 1000 + j as u32)
            .collect();

        // Annotate prompt blocks as Relocatable so hashes match the outer.
        let num_prompt_blocks = prompt_tokens.len() / block_size;
        let mut inner_ann = std::collections::BTreeMap::new();
        for b in 0..num_prompt_blocks {
            inner_ann.insert(b, vllm_common::BlockKind::Relocatable);
        }
        let prompt = Prompt::TokenIdsWithAnnotations(prompt_tokens.clone(), inner_ann);

        let start = Instant::now();
        let results = llm.generate_sealed(&[prompt], Some(inner_sampling.clone()), true, true)?;
        let ms = start.elapsed().as_secs_f64() * 1000.0;

        let output_tokens = &results[0].outputs[0].token_ids;
        eprintln!(
            "    inner[{i}]  {BOLD}{ms:>8.1}ms{RST}  {DIM}{} prompt + {} output tokens{RST}",
            prompt_tokens.len(),
            output_tokens.len()
        );
        // Full sequence = prompt + output (matches what's in KV cache).
        let mut full_seq = prompt_tokens;
        full_seq.extend_from_slice(output_tokens);
        inner_sequences.push(full_seq);
    }

    // -----------------------------------------------------------------------
    // Step 2: Build outer prompt from full inner sequences + query.
    // Each inner sequence (prompt + output) is padded to a block boundary
    // and annotated as Relocatable. The token content must match what was
    // sealed in the KV cache for cache hits.
    // -----------------------------------------------------------------------
    let mut outer_tokens = Vec::new();
    let mut annotations = std::collections::BTreeMap::new();

    for seq in &inner_sequences {
        // Pad to block boundary before each inner sequence.
        let remainder = outer_tokens.len() % block_size;
        if remainder > 0 {
            let pad_count = block_size - remainder;
            outer_tokens.extend(std::iter::repeat_n(pad_token, pad_count));
        }
        let start_block = outer_tokens.len() / block_size;
        outer_tokens.extend_from_slice(seq);
        // Pad the sequence itself to block boundary.
        let remainder = outer_tokens.len() % block_size;
        if remainder > 0 {
            let pad_count = block_size - remainder;
            outer_tokens.extend(std::iter::repeat_n(pad_token, pad_count));
        }
        let end_block = outer_tokens.len() / block_size;
        for b in start_block..end_block {
            annotations.insert(b, vllm_common::BlockKind::Relocatable);
        }
    }
    let total_inner_tokens = outer_tokens.len();

    // Build two outer prompts with different query suffixes.
    let make_outer = |query_id: u32| -> Vec<u32> {
        let mut tokens = outer_tokens.clone();
        for j in 0..query_len {
            tokens.push(90000 + query_id * 1000 + j as u32);
        }
        tokens
    };

    let outer_baseline = make_outer(0);
    let outer_spans = make_outer(1);

    eprintln!();
    eprintln!(
        "  Outer prompt: {} tokens ({total_inner_tokens} from inner outputs, {query_len} query)",
        outer_baseline.len()
    );

    // -----------------------------------------------------------------------
    // Step 2: Outer WITHOUT annotations (baseline — full recompute).
    // -----------------------------------------------------------------------
    eprintln!();
    eprintln!("{BOLD}Step 2:{RST} Outer generate WITHOUT annotations (baseline)");

    let start = Instant::now();
    llm.generate(
        &[Prompt::TokenIds(outer_baseline)],
        Some(outer_sampling.clone()),
    )?;
    let baseline_ms = start.elapsed().as_secs_f64() * 1000.0;
    eprintln!("    {BOLD}{baseline_ms:>8.1}ms{RST}  {DIM}full recompute{RST}");

    // -----------------------------------------------------------------------
    // Step 3: Outer WITH Relocatable annotations.
    // The inner output blocks have matching content in the KV cache from
    // step 1 (if seal worked). The scheduler should detect cache hits for
    // those blocks and skip their recomputation.
    // -----------------------------------------------------------------------
    eprintln!();
    eprintln!("{BOLD}Step 3:{RST} Outer generate WITH Relocatable annotations");

    let start = Instant::now();
    llm.generate(
        &[Prompt::TokenIdsWithAnnotations(outer_spans, annotations)],
        Some(outer_sampling.clone()),
    )?;
    let spans_ms = start.elapsed().as_secs_f64() * 1000.0;
    eprintln!("    {BOLD}{spans_ms:>8.1}ms{RST}  {DIM}span cache hit{RST}");

    drop(llm);

    // -----------------------------------------------------------------------
    // Summary
    // -----------------------------------------------------------------------
    eprintln!();
    println!("{BOLD}=== Nested Spans Results ==={RST}");
    println!();
    println!("  Baseline (no annotations):  {BOLD}{baseline_ms:>8.1}ms{RST}");
    println!(
        "  With Relocatable:           {BOLD}{spans_ms:>8.1}ms{RST}  ({BOLD}{:.1}x{RST} speedup)",
        baseline_ms / spans_ms
    );
    println!();

    Ok(())
}
