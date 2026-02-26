// SPDX-License-Identifier: Apache-2.0
// Copyright contributors to the vLLM project

//! `vllm serve` subcommand — start the OpenAI-compatible API server.

use std::sync::Arc;

use anyhow::Result;
use tracing::info;
use vllm_common::telemetry;
use vllm_serve::server::{AppState, ServerConfig};

use crate::args::ServeArgs;
use crate::init::initialize_stack;

/// Run the serve subcommand.
pub async fn run_serve(args: ServeArgs) -> Result<()> {
    // 1. Init tracing.
    telemetry::init_tracing(&args.log_level);

    let model = args.resolved_model().map_err(|e| anyhow::anyhow!(e))?;
    let host = args.host.clone();
    let port = args.port;
    let enable_metrics = args.enable_metrics;

    info!("vLLM Rust — starting server");
    info!("Model: {}", model);
    info!("Device: {}, dtype: {}", args.device, args.dtype);

    // 2. Initialize the full stack (on a blocking thread to avoid starving
    //    the tokio I/O driver during model download / weight loading).
    let stack = tokio::task::spawn_blocking(move || initialize_stack(&args))
        .await
        .expect("initialize_stack panicked")?;

    // 3. Spawn the engine step loop.
    let _step_handle = stack.engine.spawn_step_loop();

    // 4. Build server config and serve.
    let bind_address = format!("{}:{}", host, port);
    let server_config = ServerConfig {
        bind_address: bind_address.clone(),
        version: format!("0.1.0-rust ({})", stack.model_name),
        cors_enabled: true,
        metrics_enabled: enable_metrics,
    };

    let app_state = Arc::new(AppState {
        engine: stack.engine,
        config: server_config,
    });

    info!("Serving on http://{}", bind_address);
    vllm_serve::server::serve(app_state)
        .await
        .map_err(|e| anyhow::anyhow!("{e}"))?;

    Ok(())
}
