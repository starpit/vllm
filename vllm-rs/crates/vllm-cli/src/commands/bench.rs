// SPDX-License-Identifier: Apache-2.0
// Copyright contributors to the vLLM project

//! `vllm bench` subcommand — simple throughput benchmark.

use std::time::Instant;

use anyhow::Result;
use tracing::info;
use vllm_common::telemetry;
use vllm_executor::candle_worker::{CandleWorker, CandleWorkerConfig};
use vllm_executor::worker::Worker;

use crate::args::BenchArgs;

/// Run the bench subcommand.
pub async fn run_bench(args: BenchArgs) -> Result<()> {
    telemetry::init_tracing(&args.log_level);

    let model = args.resolved_model().map_err(|e| anyhow::anyhow!(e))?;

    info!("vLLM Rust — benchmark mode");
    info!("Model: {}, device: {}, dtype: {}", model, args.device, args.dtype);
    info!(
        "Requests: {}, prompt_len: {}, max_tokens: {}",
        args.num_requests, args.prompt_len, args.max_tokens
    );

    // Initialize worker.
    let worker_config = CandleWorkerConfig {
        model_path: model,
        device_str: args.device,
        dtype: args.dtype,
        hf_token: args.hf_token,
        cache_dir: None,
    };

    let mut worker = CandleWorker::new(worker_config);
    worker.init_device()?;

    let load_start = Instant::now();
    worker.load_model()?;
    let load_elapsed = load_start.elapsed();
    info!("Model loaded in {:.2}s", load_elapsed.as_secs_f64());

    worker.initialize_cache(1024, 0)?;

    // Build a synthetic scheduler output and run forward passes.
    use std::collections::HashMap;
    use vllm_core::scheduler::output::{NewRequestData, SchedulerOutput};

    let prompt_ids: Vec<u32> = (0..args.prompt_len as u32).collect();

    let mut total_tokens = 0u64;
    let bench_start = Instant::now();

    for i in 0..args.num_requests {
        let req_id = format!("bench-{i}");
        let mut num_scheduled = HashMap::new();
        num_scheduled.insert(req_id.clone(), args.prompt_len);

        let sched_output = SchedulerOutput {
            scheduled_new_reqs: vec![NewRequestData {
                req_id: req_id.clone(),
                prompt_token_ids: Some(prompt_ids.clone()),
                block_ids: vec![],
                num_computed_tokens: 0,
                sampling_params: None,
            }],
            num_scheduled_tokens: num_scheduled,
            total_num_scheduled_tokens: args.prompt_len,
            ..SchedulerOutput::make_empty()
        };

        let output = worker.execute_model(&sched_output)?;
        let tokens = output.get_tokens(&req_id).map(|t| t.len()).unwrap_or(0);
        total_tokens += tokens as u64;
    }

    let bench_elapsed = bench_start.elapsed();
    let throughput = total_tokens as f64 / bench_elapsed.as_secs_f64();

    println!();
    println!("=== Benchmark Results ===");
    println!("Requests:      {}", args.num_requests);
    println!("Total tokens:  {total_tokens}");
    println!("Elapsed:       {:.3}s", bench_elapsed.as_secs_f64());
    println!("Throughput:    {throughput:.1} tokens/s");
    println!("Latency (avg): {:.3}ms/request", bench_elapsed.as_millis() as f64 / args.num_requests as f64);

    worker.shutdown();
    Ok(())
}
