// SPDX-License-Identifier: Apache-2.0
// Copyright contributors to the vLLM project

//! `vllm bench sweep` — parameter sweep orchestrator.
//!
//! Runs a benchmark command under multiple parameter combinations (loaded from
//! JSON files) and collects results into a timestamped output directory.
//! Mirrors Python's `vllm bench sweep serve` and `vllm bench sweep startup`.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::process::Command;

use anyhow::{Context, Result};

use crate::args::{SweepCommand, SweepCommands, SweepServeArgs, SweepStartupArgs};

// ---------------------------------------------------------------------------
// Parameter sweep types
// ---------------------------------------------------------------------------

/// A single parameter combination: key → value (CLI flag name → value string).
type ParamItem = BTreeMap<String, serde_json::Value>;

/// Read parameter combinations from a JSON file.
/// Accepts either a JSON array of objects or a single object whose values are
/// objects (keys become `_benchmark_name`).
fn read_params(path: &str) -> Result<Vec<ParamItem>> {
    let text = std::fs::read_to_string(path)
        .with_context(|| format!("Failed to read params file: {path}"))?;
    let val: serde_json::Value =
        serde_json::from_str(&text).with_context(|| format!("Invalid JSON in {path}"))?;

    match val {
        serde_json::Value::Array(arr) => arr
            .into_iter()
            .map(|v| {
                v.as_object()
                    .map(|m| m.iter().map(|(k, v)| (k.clone(), v.clone())).collect())
                    .ok_or_else(|| anyhow::anyhow!("Expected JSON object in array"))
            })
            .collect(),
        serde_json::Value::Object(map) => {
            let mut out = Vec::new();
            for (name, v) in map {
                let obj = v
                    .as_object()
                    .ok_or_else(|| anyhow::anyhow!("Expected JSON object for key {name}"))?;
                let mut item: ParamItem = obj.iter().map(|(k, v)| (k.clone(), v.clone())).collect();
                item.insert("_benchmark_name".to_string(), serde_json::json!(name));
                out.push(item);
            }
            Ok(out)
        }
        _ => anyhow::bail!("Params file must be a JSON array or object"),
    }
}

/// Apply a parameter combination to a command, appending --key value pairs.
fn apply_overrides(cmd: &[String], overrides: &ParamItem) -> Vec<String> {
    let mut out = cmd.to_vec();
    for (key, val) in overrides {
        if key.starts_with('_') {
            continue; // skip metadata keys like _benchmark_name
        }
        let flag = if key.starts_with("--") {
            key.clone()
        } else {
            format!("--{}", key.replace('_', "-"))
        };
        match val {
            serde_json::Value::Bool(true) => out.push(flag),
            serde_json::Value::Bool(false) => {}
            _ => {
                out.push(flag);
                out.push(val_to_string(val));
            }
        }
    }
    out
}

fn val_to_string(v: &serde_json::Value) -> String {
    match v {
        serde_json::Value::String(s) => s.clone(),
        serde_json::Value::Number(n) => n.to_string(),
        serde_json::Value::Bool(b) => b.to_string(),
        other => other.to_string(),
    }
}

/// Produce a filesystem-safe name for a parameter combination.
fn sanitize_name(item: &ParamItem) -> String {
    let parts: Vec<String> = item
        .iter()
        .filter(|(k, _)| !k.starts_with('_'))
        .map(|(k, v)| format!("{k}={}", val_to_string(v)))
        .collect();
    if parts.is_empty() {
        "default".to_string()
    } else {
        parts
            .join("-")
            .replace(['/', '\\', ' ', ':', '"', '\''], "_")
    }
}

/// Strip an existing --output-json / --output_json flag from a command.
fn strip_output_json(cmd: &[String]) -> Vec<String> {
    let mut out = Vec::new();
    let mut skip_next = false;
    for arg in cmd {
        if skip_next {
            skip_next = false;
            continue;
        }
        if arg == "--output-json" || arg == "--output_json" {
            skip_next = true;
            continue;
        }
        if arg.starts_with("--output-json=") || arg.starts_with("--output_json=") {
            continue;
        }
        out.push(arg.clone());
    }
    out
}

// ---------------------------------------------------------------------------
// Run a single benchmark invocation
// ---------------------------------------------------------------------------

fn run_one(
    cmd: &[String],
    output_path: &Path,
    show_stdout: bool,
    dry_run: bool,
) -> Result<Option<serde_json::Value>> {
    // If results already exist, load and return them (resume support).
    if output_path.exists() {
        eprintln!("[SKIP] Results exist: {}", output_path.display());
        let text = std::fs::read_to_string(output_path)?;
        return Ok(Some(serde_json::from_str(&text)?));
    }

    eprintln!("[RUN] {}", shell_words::join(cmd));
    eprintln!("  -> {}", output_path.display());

    if dry_run {
        return Ok(None);
    }

    if let Some(parent) = output_path.parent() {
        std::fs::create_dir_all(parent)?;
    }

    let status = Command::new(&cmd[0])
        .args(&cmd[1..])
        .stdout(if show_stdout {
            std::process::Stdio::inherit()
        } else {
            std::process::Stdio::null()
        })
        .stderr(std::process::Stdio::inherit())
        .status()
        .with_context(|| format!("Failed to execute: {}", cmd[0]))?;

    if !status.success() {
        anyhow::bail!("Command exited with status {status}");
    }

    let text = std::fs::read_to_string(output_path)
        .with_context(|| format!("Expected output file: {}", output_path.display()))?;
    Ok(Some(serde_json::from_str(&text)?))
}

// ---------------------------------------------------------------------------
// Sweep: serve
// ---------------------------------------------------------------------------

/// Wait for a server at `base_url` to become ready by attempting TCP
/// connections to its host:port. We avoid pulling in reqwest blocking —
/// once the TCP socket accepts we assume the server is ready.
fn wait_for_server(base_url: &str, timeout_secs: u64) -> Result<()> {
    // Parse host:port from the URL.
    let url: url::Url = base_url
        .parse()
        .with_context(|| format!("Invalid base URL: {base_url}"))?;
    let host = url.host_str().unwrap_or("127.0.0.1");
    let port = url.port().unwrap_or(8000);
    let addr = format!("{host}:{port}");

    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(timeout_secs);
    eprintln!("Waiting for server at {addr} ...");
    loop {
        if std::time::Instant::now() > deadline {
            anyhow::bail!("Server did not become ready within {timeout_secs}s");
        }
        if std::net::TcpStream::connect_timeout(&addr.parse()?, std::time::Duration::from_secs(2))
            .is_ok()
        {
            eprintln!("Server ready.");
            return Ok(());
        }
        std::thread::sleep(std::time::Duration::from_secs(2));
    }
}

fn run_sweep_serve(args: SweepServeArgs) -> Result<()> {
    let serve_cmd: Vec<String> = shell_words::split(&args.serve_cmd)?;
    let bench_cmd: Vec<String> = shell_words::split(&args.bench_cmd)?;

    let serve_params = match &args.serve_params {
        Some(p) => read_params(p)?,
        None => vec![BTreeMap::new()],
    };
    let bench_params = match &args.bench_params {
        Some(p) => read_params(p)?,
        None => vec![BTreeMap::new()],
    };

    let timestamp = args
        .resume
        .clone()
        .unwrap_or_else(|| chrono::Local::now().format("%Y%m%d_%H%M%S").to_string());
    let output_dir = PathBuf::from(&args.output_dir).join(&timestamp);

    if let Some(ref _resume) = args.resume {
        if !output_dir.exists() {
            anyhow::bail!(
                "Cannot resume from non-existent directory: {}",
                output_dir.display()
            );
        }
        eprintln!("Resuming from {}", output_dir.display());
    }

    for serve_comb in &serve_params {
        let server_cmd = apply_overrides(&serve_cmd, serve_comb);

        // Start the server as a child process.
        eprintln!("\n[BEGIN SERVER] {}", shell_words::join(&server_cmd));
        if args.dry_run {
            for bench_comb in &bench_params {
                for run in 0..args.num_runs {
                    let dir_name = format!(
                        "SERVE-{}-BENCH-{}",
                        sanitize_name(serve_comb),
                        sanitize_name(bench_comb)
                    );
                    let out_path = output_dir.join(&dir_name).join(format!("run={run}.json"));
                    let mut full_cmd = apply_overrides(&bench_cmd, bench_comb);
                    full_cmd = strip_output_json(&full_cmd);
                    full_cmd.extend(["--output-json".to_string(), out_path.display().to_string()]);
                    eprintln!("[DRY-RUN] {}", shell_words::join(&full_cmd));
                }
            }
            eprintln!("[END SERVER]");
            continue;
        }

        let mut server_proc = Command::new(&server_cmd[0])
            .args(&server_cmd[1..])
            .stdout(if args.show_stdout {
                std::process::Stdio::inherit()
            } else {
                std::process::Stdio::null()
            })
            .stderr(std::process::Stdio::inherit())
            .spawn()
            .with_context(|| format!("Failed to start server: {}", server_cmd[0]))?;

        // Extract base URL from bench_cmd or default.
        let base_url = bench_cmd
            .windows(2)
            .find_map(|w| {
                if w[0] == "--base-url" || w[0] == "--base_url" {
                    Some(w[1].clone())
                } else {
                    None
                }
            })
            .unwrap_or_else(|| "http://127.0.0.1:8000".to_string());

        let server_result = (|| -> Result<()> {
            wait_for_server(&base_url, args.server_ready_timeout as u64)?;

            for bench_comb in &bench_params {
                let dir_name = format!(
                    "SERVE-{}-BENCH-{}",
                    sanitize_name(serve_comb),
                    sanitize_name(bench_comb)
                );
                for run in 0..args.num_runs {
                    let out_path = output_dir.join(&dir_name).join(format!("run={run}.json"));
                    let mut full_cmd = apply_overrides(&bench_cmd, bench_comb);
                    full_cmd = strip_output_json(&full_cmd);
                    full_cmd.extend(["--output-json".to_string(), out_path.display().to_string()]);
                    run_one(&full_cmd, &out_path, args.show_stdout, false)?;
                }
            }
            Ok(())
        })();

        // Always kill server.
        let _ = server_proc.kill();
        let _ = server_proc.wait();
        eprintln!("[END SERVER]");

        server_result?;
    }

    if !args.dry_run {
        eprintln!("\nSweep complete. Results in {}", output_dir.display());
    }

    Ok(())
}

// ---------------------------------------------------------------------------
// Sweep: startup
// ---------------------------------------------------------------------------

fn run_sweep_startup(args: SweepStartupArgs) -> Result<()> {
    let startup_cmd: Vec<String> = shell_words::split(&args.startup_cmd)?;

    let serve_params = match &args.serve_params {
        Some(p) => read_params(p)?,
        None => vec![BTreeMap::new()],
    };
    let startup_params = match &args.startup_params {
        Some(p) => read_params(p)?,
        None => vec![BTreeMap::new()],
    };

    let timestamp = args
        .resume
        .clone()
        .unwrap_or_else(|| chrono::Local::now().format("%Y%m%d_%H%M%S").to_string());
    let output_dir = PathBuf::from(&args.output_dir).join(&timestamp);

    if let Some(ref _resume) = args.resume {
        if !output_dir.exists() {
            anyhow::bail!(
                "Cannot resume from non-existent directory: {}",
                output_dir.display()
            );
        }
        eprintln!("Resuming from {}", output_dir.display());
    }

    for serve_comb in &serve_params {
        for startup_comb in &startup_params {
            let dir_name = format!(
                "SERVE-{}-STARTUP-{}",
                sanitize_name(serve_comb),
                sanitize_name(startup_comb)
            );

            for run in 0..args.num_runs {
                let out_path = output_dir.join(&dir_name).join(format!("run={run}.json"));

                let mut full_cmd = apply_overrides(&startup_cmd, serve_comb);
                full_cmd = apply_overrides(&full_cmd, startup_comb);
                full_cmd = strip_output_json(&full_cmd);
                full_cmd.extend(["--output-json".to_string(), out_path.display().to_string()]);

                run_one(&full_cmd, &out_path, args.show_stdout, args.dry_run)?;
            }
        }
    }

    if !args.dry_run {
        eprintln!("\nSweep complete. Results in {}", output_dir.display());
    }

    Ok(())
}

// ---------------------------------------------------------------------------
// Public entry point
// ---------------------------------------------------------------------------

pub(crate) fn run_bench_sweep(cmd: SweepCommand) -> Result<()> {
    match cmd.command {
        SweepCommands::Serve(args) => run_sweep_serve(args),
        SweepCommands::Startup(args) => run_sweep_startup(args),
    }
}
