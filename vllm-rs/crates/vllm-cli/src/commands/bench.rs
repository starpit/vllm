// SPDX-License-Identifier: Apache-2.0
// Copyright contributors to the vLLM project

//! `vllm bench` subcommand — benchmarking tools.
//!
//! `vllm bench latency` measures end-to-end latency of processing a single
//! batch of requests, matching the Python `vllm bench latency` behavior.
//!
//! Uses the full engine stack (`LLM` → `AsyncEngine` → scheduler → worker)
//! rather than driving the worker directly.

use std::time::Instant;

use anyhow::Result;
use tracing::info;
use vllm_common::telemetry;
use vllm_serve::llm::{LLM, LLMBuilder, SamplingParams};

use crate::args::{BenchCommand, BenchCommands, BenchLatencyArgs};

// ---------------------------------------------------------------------------
// LLM construction
// ---------------------------------------------------------------------------

/// Build an [`LLM`] from bench args.
fn create_llm(args: &BenchLatencyArgs, model: &str) -> Result<LLM> {
    let mut builder = LLMBuilder::new(model)
        .device(&args.device)
        .dtype(&args.dtype)
        .gpu_memory_utilization(args.gpu_memory_utilization);

    if let Some(ref token) = args.hf_token {
        builder = builder.hf_token(token);
    }
    if let Some(ref gguf) = args.gguf_file {
        builder = builder.gguf_file(gguf);
    }

    builder.build()
}

// ---------------------------------------------------------------------------
// Bench runner
// ---------------------------------------------------------------------------

/// Run the `bench latency` subcommand.
///
/// Mirrors Python `vllm bench latency`: creates an LLM, generates
/// `batch_size` dummy-token prompts of length `input_len`, and times
/// each `generate()` call.
///
/// This is deliberately **not** async — `LLM` owns its own tokio runtime,
/// so it must not be created inside an existing runtime.
fn run_bench_latency(args: BenchLatencyArgs) -> Result<()> {
    telemetry::init_tracing(&args.log_level);

    let model = args.resolved_model().map_err(|e| anyhow::anyhow!(e))?;

    info!("vLLM Rust — latency benchmark");
    info!(
        "Model: {}, device: {}, dtype: {}",
        model, args.device, args.dtype
    );
    info!(
        "Iters: {}, batch_size: {}, input_len: {}, output_len: {}, warmup: {}",
        args.num_iters, args.batch_size, args.input_len, args.output_len, args.num_iters_warmup
    );

    // Initialize the full engine stack.
    let load_start = Instant::now();
    let llm = create_llm(&args, &model)?;
    let load_elapsed = load_start.elapsed();
    info!("Model loaded in {:.2}s", load_elapsed.as_secs_f64());

    // Build dummy prompts: batch_size prompts of input_len random-ish token IDs.
    // Mirrors Python: `np.random.randint(10000, size=(batch_size, input_len))`
    let dummy_prompts: Vec<Vec<u32>> = (0..args.batch_size)
        .map(|i| {
            (0..args.input_len)
                .map(|j| ((i * 997 + j * 31 + 42) % 10000) as u32)
                .collect()
        })
        .collect();

    let sampling_params = SamplingParams {
        temperature: 1.0,
        top_p: 1.0,
        ignore_eos: true,
        max_tokens: Some(args.output_len as u32),
        ..SamplingParams::default()
    };

    // Warmup.
    if args.num_iters_warmup > 0 {
        info!("Running {} warmup iteration(s)...", args.num_iters_warmup);
        for _ in 0..args.num_iters_warmup {
            llm.generate_token_ids(&dummy_prompts, Some(sampling_params.clone()))?;
        }
        info!("Warmup complete");
    }

    // Timed runs.
    let mut latencies: Vec<f64> = Vec::with_capacity(args.num_iters);

    for _ in 0..args.num_iters {
        let start = Instant::now();
        llm.generate_token_ids(&dummy_prompts, Some(sampling_params.clone()))?;
        latencies.push(start.elapsed().as_secs_f64());
    }

    // Compute stats (matching Python output format).
    let avg_latency: f64 = latencies.iter().sum::<f64>() / latencies.len() as f64;
    let total_tokens = args.num_iters as u64 * args.batch_size as u64 * args.output_len as u64;
    let total_elapsed: f64 = latencies.iter().sum();
    let throughput = total_tokens as f64 / total_elapsed;

    let percentages = [10.0, 25.0, 50.0, 75.0, 90.0, 99.0];
    let percentiles: Vec<f64> = percentages
        .iter()
        .map(|&p| percentile(&latencies, p))
        .collect();

    println!();
    println!("=== Benchmark Results ===");
    println!("Iterations:      {}", args.num_iters);
    println!("Batch size:      {}", args.batch_size);
    println!("Input len:       {} tokens", args.input_len);
    println!("Output len:      {}", args.output_len);
    println!("Total tokens:    {total_tokens}");
    println!("Throughput:      {throughput:.1} tokens/s");
    println!();
    println!("Avg latency: {avg_latency:.4}s");
    for (&pct, &val) in percentages.iter().zip(percentiles.iter()) {
        println!("{pct:.0}% percentile latency: {val:.4}s");
    }

    // Write JSON output if requested.
    if let Some(ref path) = args.output_json {
        let pct_map: serde_json::Map<String, serde_json::Value> = percentages
            .iter()
            .zip(percentiles.iter())
            .map(|(&p, &v)| (format!("{p:.0}"), serde_json::json!(v)))
            .collect();
        let results = serde_json::json!({
            "avg_latency": avg_latency,
            "latencies": latencies,
            "percentiles": pct_map,
        });
        std::fs::write(path, serde_json::to_string_pretty(&results)?)?;
        info!("Results written to {path}");
    }

    Ok(())
}

/// Dispatch bench subcommands.
pub async fn run_bench(cmd: BenchCommand) -> Result<()> {
    match cmd.command {
        BenchCommands::Latency(args) => {
            // LLM creates its own tokio runtime, so we must exit the
            // current one before calling run_bench_latency.
            tokio::task::spawn_blocking(move || run_bench_latency(*args)).await??;
            Ok(())
        }
        BenchCommands::Serve(_) => {
            anyhow::bail!("bench serve is not yet implemented");
        }
        BenchCommands::Throughput(_) => {
            anyhow::bail!("bench throughput is not yet implemented");
        }
    }
}

/// Compute the p-th percentile of a sorted slice. Returns 0 for empty input.
fn percentile(data: &[f64], p: f64) -> f64 {
    if data.is_empty() {
        return 0.0;
    }
    let mut sorted = data.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let idx = ((p / 100.0) * (sorted.len() - 1) as f64).round() as usize;
    sorted[idx.min(sorted.len() - 1)]
}
