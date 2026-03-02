// SPDX-License-Identifier: Apache-2.0
// Copyright contributors to the vLLM project

//! `vllm bench` subcommand — throughput & latency benchmark.
//!
//! Measures:
//! - **Prefill latency** (time-to-first-token, TTFT)
//! - **Decode inter-token latency** (ITL)
//! - **Total throughput** (tokens/sec)

use std::collections::HashMap;
use std::time::Instant;

use anyhow::Result;
use tracing::info;
use vllm_common::telemetry;
use vllm_core::scheduler::output::{CachedRequestData, NewRequestData, SchedulerOutput};
use vllm_executor::worker::Worker;

use crate::args::BenchArgs;

// ---------------------------------------------------------------------------
// Backend-polymorphic worker creation
// ---------------------------------------------------------------------------

/// Initialize cache using real memory detection.
fn init_bench_cache(worker: &mut dyn Worker, gpu_memory_utilization: f64) -> Result<()> {
    let available = worker.determine_available_memory()?;
    // Simple heuristic for bench: allocate blocks based on available memory.
    // Use 1 MiB per block as a rough estimate (actual size depends on model config).
    let utilization = gpu_memory_utilization.clamp(0.0, 1.0);
    let cache_bytes = (available as f64 * utilization) as usize;
    let num_blocks = (cache_bytes / (1024 * 1024)).clamp(16, 4096);
    info!(
        "Bench cache: available={:.1} GB, utilization={}, blocks={}",
        available as f64 / (1024.0 * 1024.0 * 1024.0),
        utilization,
        num_blocks
    );
    worker.initialize_cache(num_blocks, 0)?;
    Ok(())
}

#[cfg(feature = "metal")]
fn create_bench_worker(args: &BenchArgs, model: String) -> Result<Box<dyn Worker>> {
    use vllm_mlx::worker::{MlxWorker, MlxWorkerConfig};

    info!("Bench: using MLX backend (Apple Silicon GPU)");
    let config = MlxWorkerConfig {
        model_path: model,
        dtype: args.dtype.clone(),
        hf_token: args.hf_token.clone(),
        cache_dir: None,
        block_size: 16,
        lora_adapter: None,
        pooling_strategy: "auto".to_string(),
    };
    let mut worker = MlxWorker::new(config);
    worker.init_device()?;
    worker.load_model()?;
    init_bench_cache(&mut worker, args.gpu_memory_utilization)?;
    Ok(Box::new(worker))
}

#[cfg(not(feature = "metal"))]
fn create_bench_worker(args: &BenchArgs, model: String) -> Result<Box<dyn Worker>> {
    use vllm_executor::candle_worker::{CandleWorker, CandleWorkerConfig};

    info!("Bench: using Candle backend");
    let config = CandleWorkerConfig {
        model_path: model,
        device_str: args.device.clone(),
        dtype: args.dtype.clone(),
        hf_token: args.hf_token.clone(),
        cache_dir: None,
        block_size: 16,
        gguf_file: args.gguf_file.clone(),
        lora_adapter: None,
        pooling_strategy: "auto".to_string(),
        is_pooling: false,
    };
    let mut worker = CandleWorker::new(config);
    worker.init_device()?;
    worker.load_model()?;
    init_bench_cache(&mut worker, args.gpu_memory_utilization)?;
    Ok(Box::new(worker))
}

// ---------------------------------------------------------------------------
// Scheduler output helpers
// ---------------------------------------------------------------------------

/// Build a scheduler output for a prefill (new request).
fn make_prefill_output(req_id: &str, prompt_ids: &[u32]) -> SchedulerOutput {
    let mut num_scheduled = HashMap::new();
    num_scheduled.insert(req_id.to_string(), prompt_ids.len());

    SchedulerOutput {
        scheduled_new_reqs: vec![NewRequestData {
            req_id: req_id.to_string(),
            prompt_token_ids: Some(prompt_ids.to_vec()),
            block_ids: vec![],
            num_computed_tokens: 0,
            sampling_params: None,
            mm_data: None,
        }],
        num_scheduled_tokens: num_scheduled,
        total_num_scheduled_tokens: prompt_ids.len(),
        ..SchedulerOutput::make_empty()
    }
}

/// Build a scheduler output for a decode step (cached request, 1 token).
fn make_decode_output(req_id: &str) -> SchedulerOutput {
    let mut num_scheduled = HashMap::new();
    num_scheduled.insert(req_id.to_string(), 1);

    SchedulerOutput {
        scheduled_cached_reqs: CachedRequestData {
            req_ids: vec![req_id.to_string()],
            ..CachedRequestData::make_empty()
        },
        num_scheduled_tokens: num_scheduled,
        total_num_scheduled_tokens: 1,
        ..SchedulerOutput::make_empty()
    }
}

/// Build a scheduler output that marks a request as finished.
fn make_finish_output(req_id: &str) -> SchedulerOutput {
    let mut finished = std::collections::HashSet::new();
    finished.insert(req_id.to_string());
    SchedulerOutput {
        finished_req_ids: finished,
        ..SchedulerOutput::make_empty()
    }
}

// ---------------------------------------------------------------------------
// Bench runner
// ---------------------------------------------------------------------------

/// Per-request timing results.
struct RequestTiming {
    prefill_ms: f64,
    decode_itl_ms: Vec<f64>,
}

/// Run one request through prefill + decode steps, returning timing.
fn run_one_request(
    worker: &mut dyn Worker,
    req_id: &str,
    prompt_ids: &[u32],
    max_tokens: usize,
) -> Result<RequestTiming> {
    // Prefill.
    let prefill_start = Instant::now();
    let sched = make_prefill_output(req_id, prompt_ids);
    worker.execute_model(&sched)?;
    let prefill_ms = prefill_start.elapsed().as_secs_f64() * 1000.0;

    // Decode steps.
    let mut decode_itl_ms = Vec::with_capacity(max_tokens.saturating_sub(1));
    for _ in 1..max_tokens {
        let decode_start = Instant::now();
        let sched = make_decode_output(req_id);
        worker.execute_model(&sched)?;
        decode_itl_ms.push(decode_start.elapsed().as_secs_f64() * 1000.0);
    }

    // Clean up.
    let finish = make_finish_output(req_id);
    worker.execute_model(&finish)?;

    Ok(RequestTiming {
        prefill_ms,
        decode_itl_ms,
    })
}

/// Run the bench subcommand.
pub async fn run_bench(args: BenchArgs) -> Result<()> {
    telemetry::init_tracing(&args.log_level);

    let model = args.resolved_model().map_err(|e| anyhow::anyhow!(e))?;

    info!("vLLM Rust — benchmark mode");
    info!(
        "Model: {}, device: {}, dtype: {}",
        model, args.device, args.dtype
    );
    info!(
        "Requests: {}, prompt_len: {}, max_tokens: {}, warmup: {}",
        args.num_requests, args.prompt_len, args.max_tokens, args.warmup
    );

    // Initialize worker.
    let load_start = Instant::now();
    let mut worker = create_bench_worker(&args, model)?;
    let load_elapsed = load_start.elapsed();
    info!("Model loaded in {:.2}s", load_elapsed.as_secs_f64());

    let prompt_ids: Vec<u32> = (0..args.prompt_len as u32).collect();

    // Warmup.
    if args.warmup > 0 {
        info!("Running {} warmup request(s)...", args.warmup);
        for i in 0..args.warmup {
            let req_id = format!("warmup-{i}");
            run_one_request(worker.as_mut(), &req_id, &prompt_ids, args.max_tokens)?;
        }
        info!("Warmup complete");
    }

    // Timed runs.
    let mut timings: Vec<RequestTiming> = Vec::with_capacity(args.num_requests);
    let bench_start = Instant::now();

    for i in 0..args.num_requests {
        let req_id = format!("bench-{i}");
        let timing = run_one_request(worker.as_mut(), &req_id, &prompt_ids, args.max_tokens)?;
        timings.push(timing);
    }

    let bench_elapsed = bench_start.elapsed();

    // Compute stats.
    let total_tokens = args.num_requests as u64 * args.max_tokens as u64;
    let throughput = total_tokens as f64 / bench_elapsed.as_secs_f64();

    let avg_prefill_ms: f64 =
        timings.iter().map(|t| t.prefill_ms).sum::<f64>() / timings.len() as f64;

    let all_itl: Vec<f64> = timings
        .iter()
        .flat_map(|t| t.decode_itl_ms.iter().copied())
        .collect();
    let avg_itl_ms = if all_itl.is_empty() {
        0.0
    } else {
        all_itl.iter().sum::<f64>() / all_itl.len() as f64
    };
    let p50_itl = percentile(&all_itl, 50.0);
    let p99_itl = percentile(&all_itl, 99.0);

    println!();
    println!("=== Benchmark Results ===");
    println!("Requests:        {}", args.num_requests);
    println!("Prompt len:      {} tokens", args.prompt_len);
    println!("Max tokens:      {}", args.max_tokens);
    println!("Total tokens:    {total_tokens}");
    println!("Elapsed:         {:.3}s", bench_elapsed.as_secs_f64());
    println!("Throughput:      {throughput:.1} tokens/s");
    println!();
    println!("--- Prefill (TTFT) ---");
    println!("  avg:  {avg_prefill_ms:.2}ms");
    println!();
    println!("--- Decode (ITL) ---");
    println!("  avg:  {avg_itl_ms:.2}ms");
    println!("  p50:  {p50_itl:.2}ms");
    println!("  p99:  {p99_itl:.2}ms");

    worker.shutdown();
    Ok(())
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
