// SPDX-License-Identifier: Apache-2.0
// Copyright contributors to the vLLM project

//! Test server helper — spawns `vllm serve` as a child process and manages its lifecycle.

use std::process::{Child, Command, Stdio};
use std::time::{Duration, Instant};

use anyhow::{Context, Result, bail};

/// Default timeout waiting for the server to become healthy.
const DEFAULT_STARTUP_TIMEOUT: Duration = Duration::from_secs(120);

/// Poll interval when waiting for /health.
const HEALTH_POLL_INTERVAL: Duration = Duration::from_millis(500);

/// A running `vllm serve` process for E2E testing.
///
/// On drop, the child process is killed. Use [`TestServer::start`] to launch.
pub struct TestServer {
    child: Child,
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
        // Best-effort kill.
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}

/// Builder for configuring and starting a [`TestServer`].
pub struct TestServerBuilder {
    model: String,
    extra_args: Vec<String>,
    startup_timeout: Duration,
    port: Option<u16>,
}

impl TestServerBuilder {
    /// Add extra CLI arguments (e.g. `--tool-call-parser hermes`).
    pub fn with_args(mut self, args: &[&str]) -> Self {
        self.extra_args
            .extend(args.iter().map(|s| s.to_string()));
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

    /// Start the server and wait for it to become healthy.
    pub async fn start(self) -> Result<TestServer> {
        let port = self.port.unwrap_or_else(|| {
            // Find a free port by binding to :0 and reading the assigned port.
            let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
            listener.local_addr().unwrap().port()
        });

        let base_url = format!("http://127.0.0.1:{port}");

        // Find the vllm binary. Prefer the pre-built binary in target/debug or target/release.
        let binary = find_vllm_binary()?;

        let mut cmd = Command::new(&binary);
        cmd.arg("serve")
            .arg(&self.model)
            .arg("--host")
            .arg("127.0.0.1")
            .arg("--port")
            .arg(port.to_string())
            .arg("--log-level")
            .arg("info");

        for arg in &self.extra_args {
            cmd.arg(arg);
        }

        // Inherit stderr so we can see server logs in test output.
        cmd.stdout(Stdio::piped()).stderr(Stdio::inherit());

        tracing::info!("Starting vllm server: {:?}", cmd);
        let child = cmd.spawn().context("failed to spawn vllm binary")?;

        let mut server = TestServer {
            child,
            port,
            base_url,
        };

        // Wait for the server to become healthy.
        if let Err(e) = wait_for_health(&server.base_url, self.startup_timeout, &mut server.child).await {
            // Kill on failure so we don't leak processes.
            let _ = server.child.kill();
            let _ = server.child.wait();
            // Prevent the Drop from trying to kill again.
            std::mem::forget(server);
            return Err(e);
        }

        tracing::info!("Server healthy on port {port}");
        Ok(server)
    }
}

/// Find the vllm binary, searching target/debug and target/release.
fn find_vllm_binary() -> Result<String> {
    // Check for VLLM_E2E_BINARY env var override.
    if let Ok(path) = std::env::var("VLLM_E2E_BINARY") {
        return Ok(path);
    }

    // Use CARGO_MANIFEST_DIR to locate the workspace root.
    // This crate lives at vllm-rs/crates/vllm-e2e, so workspace root is ../../.
    let manifest_dir = env!("CARGO_MANIFEST_DIR");
    let workspace_root = std::path::Path::new(manifest_dir)
        .parent()  // crates/
        .and_then(|p| p.parent())  // vllm-rs/
        .unwrap_or_else(|| std::path::Path::new("."));

    let candidates = [
        workspace_root.join("target/debug/vllm"),
        workspace_root.join("target/release/vllm"),
    ];

    for candidate in &candidates {
        if candidate.exists() {
            return Ok(candidate.to_string_lossy().to_string());
        }
    }

    // Fall back to PATH.
    if which_exists("vllm") {
        return Ok("vllm".to_string());
    }

    bail!(
        "Cannot find vllm binary. Build it first with `cargo build -p vllm-cli --features metal` \
         or set VLLM_E2E_BINARY to the path. (searched: {})",
        workspace_root.join("target/debug/vllm").display()
    )
}

fn which_exists(name: &str) -> bool {
    Command::new("which")
        .arg(name)
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .status()
        .map(|s| s.success())
        .unwrap_or(false)
}

/// Poll the /health endpoint until it returns 200, the child exits, or we time out.
async fn wait_for_health(base_url: &str, timeout: Duration, child: &mut Child) -> Result<()> {
    let client = reqwest::Client::new();
    let health_url = format!("{base_url}/health");
    let start = Instant::now();

    loop {
        if start.elapsed() > timeout {
            bail!(
                "Server did not become healthy within {}s",
                timeout.as_secs()
            );
        }

        // Check if the child process has exited (non-blocking).
        match child.try_wait() {
            Ok(Some(status)) => {
                bail!(
                    "Server process exited before becoming healthy (exit status: {status})"
                );
            }
            Ok(None) => {} // still running
            Err(e) => {
                bail!("Failed to check server process status: {e}");
            }
        }

        match client.get(&health_url).send().await {
            Ok(resp) if resp.status().is_success() => return Ok(()),
            Ok(resp) => {
                tracing::debug!("Health check returned {}, retrying...", resp.status());
            }
            Err(_) => {
                tracing::debug!("Health check connection refused, retrying...");
            }
        }

        tokio::time::sleep(HEALTH_POLL_INTERVAL).await;
    }
}
