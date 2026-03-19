// SPDX-License-Identifier: Apache-2.0
// Copyright contributors to the vLLM project

//! `vllm serve` subcommand — start the OpenAI-compatible API server.

use std::io::IsTerminal;
use std::sync::Arc;
use std::time::Instant;

use anyhow::Result;
use tracing::info;
use vllm_common::telemetry;
use vllm_config::{CudaGraphConfig, CudaGraphMode};
use vllm_serve::init::{VllmConfig, initialize_stack};
use vllm_serve::server::{AppState, ServerConfig};

use crate::args::ServeArgs;

/// Print the vLLM ASCII art banner with Ferris to stderr.
/// Uses ANSI colors when stderr is a terminal, monochrome otherwise.
fn print_banner(version: &str, model: &str) {
    let color = std::io::stderr().is_terminal();
    // w=white bold, o=orange, b=blue, f=rust orange, r=reset
    let (w, o, b, f, r) = if color {
        (
            "\x1b[97;1m",
            "\x1b[93m",
            "\x1b[94m",
            "\x1b[38;5;202m",
            "\x1b[0m",
        )
    } else {
        ("", "", "", "", "")
    };
    eprintln!();
    eprintln!("{f}\u{2588} \u{2588}         \u{2588} \u{2588}{r}");
    eprintln!(
        "{f}\u{2580}\u{2588}  \u{2584}\u{2588}\u{2588}\u{2588}\u{2588}\u{2588}\u{2584}  \u{2588}\u{2580}{r}          {w}\u{2588}     \u{2588}     \u{2588}\u{2584}   \u{2584}\u{2588}{r}"
    );
    eprintln!(
        "{f} \u{2580}\u{2584}\u{2588}\u{2588}\u{2588}\u{2580}\u{2588}\u{2580}\u{2588}\u{2588}\u{2588}\u{2584}\u{2580} {r}    {o}\u{2584}\u{2584}{r} {b}\u{2584}\u{2588}{r} {w}\u{2588}     \u{2588}     \u{2588} \u{2580}\u{2584}\u{2580} \u{2588}{r}  version {w}{version}{r}"
    );
    eprintln!(
        "{f} \u{2584}\u{2580}\u{2588}\u{2588}\u{2588}\u{2580}\u{2580}\u{2580}\u{2588}\u{2588}\u{2588}\u{2580}\u{2584} {r}     {o}\u{2588}{r}{b}\u{2584}\u{2588}\u{2580}{r} {w}\u{2588}     \u{2588}     \u{2588}     \u{2588}{r}  model   {w}{model}{r}"
    );
    eprintln!(
        "{f} \u{2588} \u{2584}\u{2580}\u{2580}\u{2580}\u{2580}\u{2580}\u{2580}\u{2580}\u{2584} \u{2588} {r}      {b}\u{2580}\u{2580}{r}  {w}\u{2580}\u{2580}\u{2580}\u{2580}\u{2580} \u{2580}\u{2580}\u{2580}\u{2580}\u{2580} \u{2580}     \u{2580}{r}"
    );
    eprintln!();
}

/// Run the serve subcommand.
pub async fn run_serve(args: ServeArgs) -> Result<()> {
    let startup_start = Instant::now();

    // 1. Init tracing (with optional OpenTelemetry export).
    #[cfg(feature = "otel")]
    let _otel_guard = {
        if let Some(ref endpoint) = args.otlp_traces_endpoint {
            let otel_config = telemetry::OtelConfig {
                endpoint: endpoint.clone(),
            };
            let guard = telemetry::init_tracing_with_otel(&args.log_level, &otel_config);
            if guard.is_some() {
                tracing::info!("OpenTelemetry tracing enabled → {}", endpoint);
            }
            guard
        } else {
            telemetry::init_tracing(&args.log_level);
            None
        }
    };
    #[cfg(not(feature = "otel"))]
    {
        if args.otlp_traces_endpoint.is_some() {
            eprintln!(
                "WARNING: --otlp-traces-endpoint requires building with --features otel. \
                 Ignoring."
            );
        }
        telemetry::init_tracing(&args.log_level);
    }

    let model = args.resolved_model().map_err(|e| anyhow::anyhow!(e))?;
    let host = args.host.clone();
    let port = args.port;
    let enable_metrics = args.enable_metrics;
    let tool_call_parser_name = args.tool_call_parser.clone();
    let reasoning_parser_name = args.reasoning_parser.clone();

    print_banner(env!("CARGO_PKG_VERSION"), &model);
    info!("Device: {}, dtype: {}", args.device, args.dtype);
    if let Some(ref spec_model) = args.speculative_model {
        info!(
            "Speculative decoding: {} (k={}, ngram_max={}, ngram_min={})",
            spec_model,
            args.num_speculative_tokens,
            args.ngram_prompt_lookup_max,
            args.ngram_prompt_lookup_min
        );
    }
    if let Some(ref adapter) = args.lora_adapter {
        info!("LoRA adapter: {}", adapter);
    }

    // 2. Convert CLI args to VllmConfig and initialize the full stack
    //    (on a blocking thread to avoid starving the tokio I/O driver
    //    during model download / weight loading).
    let config = VllmConfig {
        model,
        device: args.device.clone(),
        dtype: args.dtype.clone(),
        max_model_len: args.max_model_len,
        max_num_seqs: args.max_num_seqs,
        block_size: args.block_size,
        gpu_memory_utilization: args.gpu_memory_utilization,
        hf_token: args.hf_token.clone(),
        gguf_file: args.gguf_file.clone(),
        speculative_model: args.speculative_model.clone(),
        num_speculative_tokens: args.num_speculative_tokens,
        ngram_prompt_lookup_max: args.ngram_prompt_lookup_max,
        ngram_prompt_lookup_min: args.ngram_prompt_lookup_min,
        lora_adapter: args.lora_adapter.clone(),
        pooling_strategy: args.pooling_strategy.clone(),
        tensor_parallel_size: args.tensor_parallel_size,
        pipeline_parallel_size: args.pipeline_parallel_size,
        num_nodes: args.num_nodes,
        node_rank: args.node_rank,
        master_addr: args.master_addr.clone(),
        master_port: args.master_port,
        disable_async_scheduling: args.disable_async_scheduling,
        runner: args.runner.clone(),
        cuda_graph_config: if args.enforce_eager {
            None
        } else {
            let parsed_mode = CudaGraphMode::parse(&args.cuda_graph_mode)
                .unwrap_or(CudaGraphMode::FullAndPiecewise);
            let sizes = CudaGraphConfig::parse_sizes(&args.cuda_graph_sizes);
            let capture_sizes = if sizes.is_empty() {
                // "auto" → compute Python-matching sizes
                CudaGraphConfig::auto_capture_sizes(args.max_num_seqs)
            } else {
                sizes
            };
            Some(CudaGraphConfig {
                enabled: true,
                mode: parsed_mode,
                capture_sizes,
                num_warmups: 2,
            })
        },
        enable_prefix_caching: !args.no_prefix_caching,
        enforce_eager: args.enforce_eager,
        cuda_graph_mode: args.cuda_graph_mode.clone(),
        max_num_batched_tokens: args.max_num_batched_tokens,
        cublas_autotune: args.cublas_autotune,
        kv_cache_dtype: args.kv_cache_dtype.clone(),
        calculate_kv_scales: args.calculate_kv_scales,
        distributed_executor_backend: args.distributed_executor_backend.clone(),
    };

    // Multi-node follower: run headless (no engine, no HTTP server).
    // This blocks until the leader sends a Shutdown command.
    #[cfg(feature = "nccl")]
    if config.num_nodes > 1 && config.node_rank > 0 {
        info!(
            "Follower node (rank {}): entering headless mode",
            config.node_rank
        );
        return tokio::task::spawn_blocking(move || {
            vllm_serve::init::initialize_and_run_follower(&config)
        })
        .await
        .expect("initialize_and_run_follower panicked");
    }

    // Keep a clone for /server_info (before we move config into the blocking task).
    let vllm_config_snapshot = config.clone();

    let mut stack = tokio::task::spawn_blocking(move || initialize_stack(&config))
        .await
        .expect("initialize_stack panicked")?;

    // 2b. Configure tool call parser if specified.
    if let Some(ref parser_name) = tool_call_parser_name {
        let parser = vllm_serve::tool_parser::get_tool_parser(parser_name)
            .map_err(|e| anyhow::anyhow!(e))?;
        // We need mutable access before the engine is shared.
        Arc::get_mut(&mut stack.engine)
            .expect("engine should not be shared yet")
            .set_tool_parser(parser);
        info!("Tool call parser: {}", parser_name);
    }

    // 2c. Configure reasoning parser if specified.
    if let Some(ref parser_name) = reasoning_parser_name {
        let vocab = stack
            .engine
            .tokenizer()
            .ok_or_else(|| anyhow::anyhow!("reasoning parser requires a tokenizer"))?
            .get_vocab();
        let parser = vllm_serve::reasoning_parser::get_reasoning_parser(parser_name, &vocab)
            .map_err(|e| anyhow::anyhow!(e))?;
        Arc::get_mut(&mut stack.engine)
            .expect("engine should not be shared yet")
            .set_reasoning_parser(parser);
        info!("Reasoning parser: {}", parser_name);
    }

    // 3. Spawn the engine step loop.
    let _step_handle = stack.engine.spawn_step_loop();

    // 4. Build server config and serve.
    let bind_address = format!("{}:{}", host, port);
    let server_config = ServerConfig {
        bind_address: bind_address.clone(),
        version: format!("0.1.0-rust ({})", stack.model_name),
        cors_enabled: true,
        metrics_enabled: enable_metrics,
        ssl_keyfile: args.ssl_keyfile.clone(),
        ssl_certfile: args.ssl_certfile.clone(),
        ssl_ca_certs: args.ssl_ca_certs.clone(),
        startup_instant: Some(startup_start),
    };

    let is_pooling = args.runner == "pooling";
    let app_state = Arc::new(AppState {
        engine: stack.engine,
        config: server_config,
        is_pooling,
        vllm_config: Some(vllm_config_snapshot),
    });
    vllm_serve::server::serve(app_state)
        .await
        .map_err(|e| anyhow::anyhow!("{e}"))?;

    Ok(())
}
