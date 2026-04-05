// SPDX-License-Identifier: Apache-2.0
// Copyright contributors to the vLLM project

//! `vllm bench` — benchmarking tools for the vLLM Rust inference engine.
//!
//! This crate provides latency, serving, and throughput benchmarks that can be
//! driven from any CLI frontend. The canonical entry point is [`run_bench`].

mod args;
pub(crate) mod datasets;
mod hotpotqa;
mod latency;
mod longbench;
mod msmarco;
mod multihop;
mod musique;
mod niah;
mod ragcsv;
mod ragindex;
mod ruler;
mod serve;
mod spans;
mod startup;
mod sweep;
mod throughput;

pub use args::{
    BenchCommand, BenchCommands, BenchHotpotqaArgs, BenchLatencyArgs, BenchLongbenchArgs,
    BenchMsmarcoArgs, BenchMultihopArgs, BenchMusiqueArgs, BenchNiahArgs, BenchRagcsvArgs,
    BenchRagindexArgs, BenchRulerArgs, BenchServeArgs, BenchSpansArgs, BenchStartupArgs,
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
        BenchCommands::Spans(args) => {
            tokio::task::spawn_blocking(move || spans::run_bench_spans(args)).await??;
            Ok(())
        }
        BenchCommands::Niah(args) => {
            tokio::task::spawn_blocking(move || niah::run_bench_niah(args)).await??;
            Ok(())
        }
        BenchCommands::Ruler(args) => {
            tokio::task::spawn_blocking(move || ruler::run_bench_ruler(args)).await??;
            Ok(())
        }
        BenchCommands::Ragcsv(args) => {
            tokio::task::spawn_blocking(move || ragcsv::run_bench_ragcsv(args)).await??;
            Ok(())
        }
        BenchCommands::Multihop(args) => {
            tokio::task::spawn_blocking(move || multihop::run_bench_multihop(args)).await??;
            Ok(())
        }
        BenchCommands::Musique(args) => {
            tokio::task::spawn_blocking(move || musique::run_bench_musique(args)).await??;
            Ok(())
        }
        BenchCommands::Hotpotqa(args) => {
            tokio::task::spawn_blocking(move || hotpotqa::run_bench_hotpotqa(args)).await??;
            Ok(())
        }
        BenchCommands::Msmarco(args) => {
            tokio::task::spawn_blocking(move || msmarco::run_bench_msmarco(args)).await??;
            Ok(())
        }
        BenchCommands::Longbench(args) => {
            tokio::task::spawn_blocking(move || longbench::run_bench_longbench(args)).await??;
            Ok(())
        }
        BenchCommands::Ragindex(args) => {
            tokio::task::spawn_blocking(move || ragindex::run_bench_ragindex(args)).await??;
            Ok(())
        }
    }
}
