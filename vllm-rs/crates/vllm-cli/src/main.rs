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
    // Load .env file if present (before clap parses, so env vars are visible).
    dotenvy::dotenv().ok();

    let cli = Cli::parse();

    match cli.command {
        Commands::Serve(args) => commands::serve::run_serve(*args).await,
        #[cfg(feature = "bench")]
        Commands::Bench(cmd) => commands::bench::run_bench(cmd).await,
        Commands::Batch(args) | Commands::RunBatch(args) => commands::batch::run_batch(args).await,
        Commands::Chat(args) => commands::chat::run_chat(args).await,
        Commands::CollectEnv(args) => commands::collect_env::run_collect_env(args).await,
        Commands::Complete(args) => commands::chat::run_complete(args).await,
        Commands::Convert(args) => commands::convert::run_convert(args).await,
        #[cfg(feature = "gce")]
        Commands::Gce(cmd) => {
            use crate::args::GceSubcommand;
            match cmd.command {
                GceSubcommand::Up(args) => commands::gce::run_up(*args).await,
                GceSubcommand::Down(args) => commands::gce::run_down(args).await,
            }
        }
        #[cfg(feature = "top")]
        Commands::Top(args) => commands::top::run_top(args).await,
    }
}
