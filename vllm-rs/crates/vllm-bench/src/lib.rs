// SPDX-License-Identifier: Apache-2.0
// Copyright contributors to the vLLM project

//! `vllm bench` — benchmarking tools for the vLLM Rust inference engine.
//!
//! This crate provides latency, serving, and throughput benchmarks that can be
//! driven from any CLI frontend. The canonical entry point is [`run_bench`].

mod args;
mod latency;
mod serve;
mod throughput;

pub use args::{
    BenchCommand, BenchCommands, BenchLatencyArgs, BenchServeArgs, BenchThroughputArgs,
};

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
        BenchCommands::Throughput(args) => {
            tokio::task::spawn_blocking(move || throughput::run_bench_throughput(args)).await??;
            Ok(())
        }
    }
}
