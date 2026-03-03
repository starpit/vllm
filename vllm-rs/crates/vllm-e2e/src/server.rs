// SPDX-License-Identifier: Apache-2.0
// Copyright contributors to the vLLM project

//! Test server helper — starts the vLLM server in-process on a background task.

use std::sync::Arc;
use std::time::Duration;

use anyhow::{Context, Result};
use tokio::task::JoinHandle;

/// Default timeout waiting for the server to become healthy.
const DEFAULT_STARTUP_TIMEOUT: Duration = Duration::from_secs(120);

/// Poll interval when waiting for /health.
const HEALTH_POLL_INTERVAL: Duration = Duration::from_millis(100);

/// A running vLLM server for E2E testing.
///
/// The server runs in-process on a background tokio task. On drop, the task
/// is aborted and the port is released.
pub struct TestServer {
    step_handle: JoinHandle<()>,
    server_handle: JoinHandle<()>,
    port: u16,
    base_url: String,
}

impl TestServer {
    /// Create a builder for configuring a test server with the given model.
    pub fn builder(model: &str) -> TestServerBuilder {
        TestServerBuilder {
            model: model.to_string(),
            extra_args: Vec::new(),
            startup_timeout: DEFAULT_STARTUP_TIMEOUT,
            port: None,
            tool_call_parser: None,
            lora_adapter: None,
            pooling_strategy: None,
            disable_async_scheduling: false,
            runner: "generate".to_string(),
            tensor_parallel_size: 1,
            dtype: None,
            device: None,
        }
    }

    /// The base URL (e.g. `http://127.0.0.1:12345`).
    pub fn base_url(&self) -> &str {
        &self.base_url
    }

    /// The port the server is listening on.
    pub fn port(&self) -> u16 {
        self.port
    }
}

impl Drop for TestServer {
    fn drop(&mut self) {
        self.step_handle.abort();
        self.server_handle.abort();
    }
}

/// Builder for configuring and starting a [`TestServer`].
pub struct TestServerBuilder {
    model: String,
    extra_args: Vec<String>,
    startup_timeout: Duration,
    port: Option<u16>,
    tool_call_parser: Option<String>,
    lora_adapter: Option<String>,
    pooling_strategy: Option<String>,
    disable_async_scheduling: bool,
    runner: String,
    dtype: Option<String>,
    device: Option<String>,
    tensor_parallel_size: usize,
}

impl TestServerBuilder {
    /// Add extra CLI arguments (e.g. `--tool-call-parser hermes`).
    pub fn with_args(mut self, args: &[&str]) -> Self {
        self.extra_args.extend(args.iter().map(|s| s.to_string()));
        self
    }

    /// Override the port (default: random free port).
    pub fn with_port(mut self, port: u16) -> Self {
        self.port = Some(port);
        self
    }

    /// Override the startup timeout.
    pub fn with_timeout(mut self, timeout: Duration) -> Self {
        self.startup_timeout = timeout;
        self
    }

    /// Set the tool call parser (e.g. "hermes", "llama3_json", "kimi_k2").
    pub fn with_tool_call_parser(mut self, parser: &str) -> Self {
        self.tool_call_parser = Some(parser.to_string());
        self
    }

    /// Set a LoRA adapter to load (local path or HF repo ID).
    pub fn with_lora_adapter(mut self, adapter: &str) -> Self {
        self.lora_adapter = Some(adapter.to_string());
        self
    }

    /// Set the pooling strategy for embeddings (e.g. "mean", "cls", "last").
    pub fn with_pooling_strategy(mut self, strategy: &str) -> Self {
        self.pooling_strategy = Some(strategy.to_string());
        self
    }

    /// Disable async scheduling (use synchronous step loop instead).
    pub fn with_sync_scheduling(mut self) -> Self {
        self.disable_async_scheduling = true;
        self
    }

    /// Set the runner type: "generate" (default) or "pooling".
    /// In pooling mode, embedding requests go through the scheduler and
    /// generation endpoints are rejected.
    pub fn with_runner(mut self, runner: &str) -> Self {
        self.runner = runner.to_string();
        self
    }

    /// Set the tensor parallel size (number of GPUs).
    pub fn with_tensor_parallel_size(mut self, tp: usize) -> Self {
        self.tensor_parallel_size = tp;
        self
    }

    /// Override the weight dtype (e.g. "f32", "f16", "bf16").
    pub fn with_dtype(mut self, dtype: &str) -> Self {
        self.dtype = Some(dtype.to_string());
        self
    }

    /// Override the device (e.g. "cpu", "cuda:0", "metal").
    pub fn with_device(mut self, device: &str) -> Self {
        self.device = Some(device.to_string());
        self
    }

    /// Start the server in-process and wait for it to become healthy.
    pub async fn start(self) -> Result<TestServer> {
        // Initialize tracing. Silent by default; set RUST_LOG=info to see
        // download/loading progress. Idempotent — only the first call takes effect.
        vllm_common::telemetry::init_tracing("off");

        let port = self.port.unwrap_or_else(|| {
            let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
            listener.local_addr().unwrap().port()
        });

        let base_url = format!("http://127.0.0.1:{port}");
        let bind_address = format!("127.0.0.1:{port}");
        let model = self.model.clone();

        // Build VllmConfig for the requested configuration.
        let pooling_strategy = self
            .pooling_strategy
            .clone()
            .unwrap_or_else(|| "auto".to_string());
        let runner = self.runner.clone();
        let dtype = self.dtype.clone().unwrap_or_else(|| "auto".to_string());
        let device = self.device.clone().unwrap_or_else(|| "auto".to_string());
        let config = vllm_serve::init::VllmConfig {
            model: model.clone(),
            device,
            dtype,
            max_num_seqs: 256,
            block_size: 16,
            gpu_memory_utilization: 0.9,
            lora_adapter: self.lora_adapter.clone(),
            pooling_strategy,
            tensor_parallel_size: self.tensor_parallel_size,
            disable_async_scheduling: self.disable_async_scheduling,
            runner: runner.clone(),
            ..Default::default()
        };

        // Initialize the full stack (blocking — downloads model, loads weights).
        let mut stack =
            tokio::task::spawn_blocking(move || vllm_serve::init::initialize_stack(&config))
                .await
                .context("initialize_stack panicked")?
                .context("failed to initialize stack")?;

        // Configure tool call parser if requested.
        if let Some(ref parser_name) = self.tool_call_parser {
            let parser = vllm_serve::tool_parser::get_tool_parser(parser_name)
                .map_err(|e| anyhow::anyhow!(e))?;
            Arc::get_mut(&mut stack.engine)
                .expect("engine should not be shared yet")
                .set_tool_parser(parser);
            tracing::info!("Tool call parser configured: {}", parser_name);
        }

        // Spawn the engine step loop.
        let step_handle = stack.engine.spawn_step_loop();

        // Build server config and app state.
        let server_config = vllm_serve::server::ServerConfig {
            bind_address,
            version: format!("0.1.0-rust-test ({})", stack.model_name),
            cors_enabled: true,
            metrics_enabled: false,
            ssl_keyfile: None,
            ssl_certfile: None,
            ssl_ca_certs: None,
            startup_instant: None,
        };

        let is_pooling = runner == "pooling";
        let app_state = Arc::new(vllm_serve::server::AppState {
            engine: stack.engine,
            config: server_config,
            is_pooling,
        });

        // Spawn the HTTP server on a background task.
        let server_handle = tokio::spawn(async move {
            if let Err(e) = vllm_serve::server::serve(app_state).await {
                tracing::error!("Test server error: {e}");
            }
        });

        let test_server = TestServer {
            step_handle,
            server_handle,
            port,
            base_url,
        };

        // Wait for the server to become healthy.
        wait_for_health(&test_server.base_url, self.startup_timeout).await?;

        tracing::info!("Test server healthy on port {port}");
        Ok(test_server)
    }
}

/// Poll the /health endpoint until it returns 200 or we time out.
async fn wait_for_health(base_url: &str, timeout: Duration) -> Result<()> {
    let client = reqwest::Client::new();
    let health_url = format!("{base_url}/health");
    let start = std::time::Instant::now();

    loop {
        if start.elapsed() > timeout {
            anyhow::bail!(
                "Server did not become healthy within {}s",
                timeout.as_secs()
            );
        }

        match client.get(&health_url).send().await {
            Ok(resp) if resp.status().is_success() => return Ok(()),
            _ => {}
        }

        tokio::time::sleep(HEALTH_POLL_INTERVAL).await;
    }
}
