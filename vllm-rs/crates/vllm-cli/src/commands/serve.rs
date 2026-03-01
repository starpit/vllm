// SPDX-License-Identifier: Apache-2.0
// Copyright contributors to the vLLM project

//! `vllm serve` subcommand — start the OpenAI-compatible API server.

use std::sync::Arc;
use std::time::Instant;

use anyhow::Result;
use tracing::info;
use vllm_common::telemetry;
use vllm_serve::init::{VllmConfig, initialize_stack};
use vllm_serve::server::{AppState, ServerConfig};

use crate::args::ServeArgs;

/// Run the serve subcommand.
pub async fn run_serve(args: ServeArgs) -> Result<()> {
    let startup_start = Instant::now();

    // 1. Init tracing.
    telemetry::init_tracing(&args.log_level);

    let model = args.resolved_model().map_err(|e| anyhow::anyhow!(e))?;
    let host = args.host.clone();
    let port = args.port;
    let enable_metrics = args.enable_metrics;
    let tool_call_parser_name = args.tool_call_parser.clone();

    info!("vLLM Rust — starting server");
    info!("Model: {}", model);
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
    };

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

    // 3. Spawn the engine step loop.
    let _step_handle = stack.engine.spawn_step_loop();

    // 4. Build server config and serve.
    let bind_address = format!("{}:{}", host, port);
    let is_ssl = args.ssl_certfile.is_some() && args.ssl_keyfile.is_some();
    let startup_elapsed = startup_start.elapsed();
    let scheme = if is_ssl { "https" } else { "http" };
    info!(
        "Serving on {}://{} (startup: {:.2}s)",
        scheme,
        bind_address,
        startup_elapsed.as_secs_f64()
    );
    let server_config = ServerConfig {
        bind_address: bind_address.clone(),
        version: format!("0.1.0-rust ({})", stack.model_name),
        cors_enabled: true,
        metrics_enabled: enable_metrics,
        ssl_keyfile: args.ssl_keyfile.clone(),
        ssl_certfile: args.ssl_certfile.clone(),
        ssl_ca_certs: args.ssl_ca_certs.clone(),
    };

    let app_state = Arc::new(AppState {
        engine: stack.engine,
        config: server_config,
    });
    vllm_serve::server::serve(app_state)
        .await
        .map_err(|e| anyhow::anyhow!("{e}"))?;

    Ok(())
}
