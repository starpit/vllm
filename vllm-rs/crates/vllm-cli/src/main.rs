// SPDX-License-Identifier: Apache-2.0
// Copyright contributors to the vLLM project

//! vLLM CLI binary — standalone Rust inference engine.
//!
//! Usage:
//!   vllm serve --model <path-or-hf-id>
//!   vllm bench --model <path-or-hf-id>
//!   vllm convert --input <dir> --output <dir>

mod args;
mod commands;

use clap::Parser;

use crate::args::{Cli, Commands};

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let cli = Cli::parse();

    match cli.command {
        Commands::Serve(args) => commands::serve::run_serve(*args).await,
        Commands::Bench(args) => commands::bench::run_bench(args).await,
        Commands::Batch(args) => commands::batch::run_batch(args).await,
        Commands::Convert(args) => commands::convert::run_convert(args).await,
    }
}
