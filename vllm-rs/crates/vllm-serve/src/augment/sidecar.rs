use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use anyhow::{Context, Result, bail};
use tracing::info;

const HEALTH_POLL_INTERVAL: Duration = Duration::from_millis(100);
const STARTUP_TIMEOUT: Duration = Duration::from_secs(120);

/// A child vllm-rs process serving an embedding model.
pub struct EmbeddingSidecar {
    child: Mutex<std::process::Child>,
    pub port: u16,
    pub base_url: String,
}

impl EmbeddingSidecar {
    /// Spawn a vllm-rs sidecar serving `model` as a pooling (embedding) runner.
    fn spawn(model: &str) -> Result<Self> {
        let port = {
            let listener = std::net::TcpListener::bind("127.0.0.1:0")?;
            listener.local_addr()?.port()
        };
        let base_url = format!("http://127.0.0.1:{port}/v1");

        let binary = resolve_vllm_binary().context("cannot find vllm binary for sidecar")?;

        let mut cmd = std::process::Command::new(&binary);
        cmd.arg("serve")
            .arg(model)
            .arg("--host")
            .arg("127.0.0.1")
            .arg("--port")
            .arg(port.to_string())
            .arg("--runner")
            .arg("pooling")
            .arg("--enforce-eager")
            .arg("--log-level")
            .arg(current_log_level())
            .stdout(std::process::Stdio::inherit())
            .stderr(std::process::Stdio::inherit());

        info!(model, port, "Spawning embedding sidecar");

        let child = cmd.spawn().with_context(|| {
            format!("failed to spawn embedding sidecar at {}", binary.display())
        })?;

        // Poll /health until ready, fail fast on child exit.
        // Uses raw HTTP/1.1 over TCP to avoid reqwest's internal tokio runtime,
        // which would panic if dropped inside a spawn_blocking context.
        let start = Instant::now();

        let child_mtx = Mutex::new(child);
        loop {
            {
                let mut c = child_mtx.lock().unwrap();
                if let Some(status) = c.try_wait().context("failed to check sidecar status")? {
                    bail!("Embedding sidecar exited before becoming healthy ({status})");
                }
            }
            if start.elapsed() > STARTUP_TIMEOUT {
                let mut c = child_mtx.lock().unwrap();
                let _ = c.kill();
                let _ = c.wait();
                bail!(
                    "Embedding sidecar for {model} did not become healthy within {}s",
                    STARTUP_TIMEOUT.as_secs()
                );
            }
            if health_check_raw(port) {
                break;
            }
            std::thread::sleep(HEALTH_POLL_INTERVAL);
        }

        info!(model, port, "Embedding sidecar healthy");

        Ok(Self {
            child: child_mtx,
            port,
            base_url,
        })
    }
}

impl Drop for EmbeddingSidecar {
    fn drop(&mut self) {
        let mut child = self.child.lock().unwrap();
        let _ = child.kill();
        let _ = child.wait();
    }
}

/// Caches sidecar processes by model name so each embedding model is spawned
/// at most once per engine lifetime.
#[derive(Default)]
pub struct SidecarManager {
    sidecars: Mutex<HashMap<String, Arc<EmbeddingSidecar>>>,
}

impl SidecarManager {
    pub fn new() -> Self {
        Self::default()
    }

    /// Return an existing sidecar for `model`, or spawn a new one.
    /// If an existing sidecar has exited, it is respawned.
    pub fn get_or_spawn(&self, model: &str) -> Result<Arc<EmbeddingSidecar>> {
        let mut map = self.sidecars.lock().unwrap();
        if let Some(existing) = map.get(model) {
            // Check if the sidecar is still alive.
            let mut child_guard = existing.child.lock().unwrap();
            match child_guard.try_wait() {
                Ok(Some(status)) => {
                    tracing::warn!(
                        model,
                        ?status,
                        "Embedding sidecar exited unexpectedly, respawning"
                    );
                    drop(child_guard);
                    // Fall through to respawn.
                }
                _ => return Ok(Arc::clone(existing)),
            }
        }
        let sidecar = Arc::new(EmbeddingSidecar::spawn(model)?);
        map.insert(model.to_string(), Arc::clone(&sidecar));
        Ok(sidecar)
    }
}

/// Map the current tracing max level to a CLI `--log-level` string.
fn current_log_level() -> &'static str {
    match tracing::level_filters::LevelFilter::current().into_level() {
        Some(l) if l <= tracing::Level::ERROR => "error",
        Some(l) if l <= tracing::Level::WARN => "warn",
        Some(l) if l <= tracing::Level::INFO => "info",
        Some(l) if l <= tracing::Level::DEBUG => "debug",
        Some(_) => "trace",
        None => "warn", // OFF → default to warn
    }
}

/// Find the `vllm` binary. Checks:
/// 1. `VLLM_BINARY` env var
/// 2. `vllm` in the same directory as the current executable (or its parent if in `deps/`)
/// 3. `vllm` on PATH
fn resolve_vllm_binary() -> Result<std::path::PathBuf> {
    // Explicit override
    if let Ok(p) = std::env::var("VLLM_BINARY") {
        let path = std::path::PathBuf::from(p);
        if path.exists() {
            return Ok(path);
        }
        bail!("VLLM_BINARY set to {} but file not found", path.display());
    }

    // Next to current exe (handles both direct binary and test runner in deps/)
    if let Ok(exe) = std::env::current_exe()
        && let Some(dir) = exe.parent()
    {
        let profile_dir = if dir.ends_with("deps") {
            dir.parent().unwrap_or(dir)
        } else {
            dir
        };
        let candidate = profile_dir.join("vllm");
        if candidate.exists() {
            return Ok(candidate);
        }
    }

    // Fall back to PATH
    if let Ok(output) = std::process::Command::new("which").arg("vllm").output()
        && output.status.success()
    {
        let path = String::from_utf8_lossy(&output.stdout).trim().to_string();
        if !path.is_empty() {
            return Ok(std::path::PathBuf::from(path));
        }
    }

    bail!("cannot find vllm binary; set VLLM_BINARY or ensure it is on PATH")
}

/// Raw HTTP/1.1 health check using std::net — avoids reqwest's internal tokio
/// runtime which panics when dropped inside spawn_blocking.
fn health_check_raw(port: u16) -> bool {
    use std::io::{Read, Write};
    let Ok(mut stream) = std::net::TcpStream::connect_timeout(
        &std::net::SocketAddr::from(([127, 0, 0, 1], port)),
        Duration::from_secs(1),
    ) else {
        return false;
    };
    let _ = stream.set_read_timeout(Some(Duration::from_secs(2)));
    if stream
        .write_all(b"GET /health HTTP/1.1\r\nHost: 127.0.0.1\r\nConnection: close\r\n\r\n")
        .is_err()
    {
        return false;
    }
    let mut buf = [0u8; 64];
    if let Ok(n) = stream.read(&mut buf) {
        let response = String::from_utf8_lossy(&buf[..n]);
        response.contains("200")
    } else {
        false
    }
}
