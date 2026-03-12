// SPDX-License-Identifier: Apache-2.0
// Copyright contributors to the vLLM project

//! `vllm bench` — benchmarking tools for the vLLM Rust inference engine.
//!
//! This crate provides latency, serving, and throughput benchmarks that can be
//! driven from any CLI frontend. The canonical entry point is [`run_bench`].

mod args;
pub(crate) mod datasets;
mod latency;
mod serve;
mod startup;
mod sweep;
mod throughput;

pub use args::{
    BenchCommand, BenchCommands, BenchLatencyArgs, BenchServeArgs, BenchStartupArgs,
    BenchThroughputArgs, SweepCommand, SweepCommands, SweepServeArgs, SweepStartupArgs,
};

/// Format a rate as `it/s` (fast) or `s/it` (slow), matching Python tqdm style.
pub(crate) fn fmt_tqdm_rate(state: &indicatif::ProgressState, w: &mut dyn std::fmt::Write) {
    let per_sec = state.per_sec();
    if per_sec >= 1.0 {
        write!(w, "{per_sec:.2}it/s").unwrap();
    } else if per_sec > 0.0 {
        write!(w, "{:.2}s/it", 1.0 / per_sec).unwrap();
    } else {
        write!(w, "?it/s").unwrap();
    }
}

/// Dispatch bench subcommands.
pub async fn run_bench(cmd: BenchCommand) -> anyhow::Result<()> {
    match cmd.command {
        BenchCommands::Latency(args) => {
            // LLM creates its own tokio runtime, so we must exit the
            // current one before calling run_bench_latency.
            tokio::task::spawn_blocking(move || latency::run_bench_latency(*args)).await??;
            Ok(())
        }
        BenchCommands::Serve(args) => serve::run_bench_serve(args).await,
        BenchCommands::Startup(args) => {
            tokio::task::spawn_blocking(move || startup::run_bench_startup(args)).await??;
            Ok(())
        }
        BenchCommands::Sweep(cmd) => {
            tokio::task::spawn_blocking(move || sweep::run_bench_sweep(cmd)).await??;
            Ok(())
        }
        BenchCommands::Throughput(args) => {
            tokio::task::spawn_blocking(move || throughput::run_bench_throughput(args)).await??;
            Ok(())
        }
    }
}
