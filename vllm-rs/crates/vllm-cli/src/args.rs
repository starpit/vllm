// SPDX-License-Identifier: Apache-2.0
// Copyright contributors to the vLLM project

//! CLI argument definitions using clap derive.

use clap::{Parser, Subcommand};

/// vLLM — High-throughput LLM serving engine (Rust)
#[derive(Parser, Debug)]
#[command(name = "vllm", version, about, long_about = None)]
pub struct Cli {
    #[command(subcommand)]
    pub command: Commands,
}

#[derive(Subcommand, Debug)]
pub enum Commands {
    /// Start the OpenAI-compatible API server.
    Serve(ServeArgs),
    /// Run a quick benchmark against a model.
    Bench(BenchArgs),
    /// Convert model weights between formats (stub).
    Convert(ConvertArgs),
}

/// Arguments for the `serve` subcommand.
#[derive(Parser, Debug)]
#[command(override_usage = "vllm serve [MODEL] [OPTIONS]")]
pub struct ServeArgs {
    /// Model to serve: local path or HuggingFace model ID
    /// (e.g. "meta-llama/Llama-3.2-1B"). Takes precedence over --model.
    pub model_tag: Option<String>,

    /// Path to a local model directory, or HuggingFace model ID.
    /// Can also be specified as a positional argument.
    #[arg(long, env = "VLLM_MODEL")]
    pub model: Option<String>,

    /// Host address to bind.
    #[arg(long, default_value = "0.0.0.0")]
    pub host: String,

    /// Port to listen on.
    #[arg(long, default_value_t = 8000)]
    pub port: u16,

    /// Device: "cpu", "cuda:N", "metal", or "auto" (auto-detect best GPU).
    #[arg(long, default_value = "auto")]
    pub device: String,

    /// Weight dtype: "auto", "float16", "bfloat16", "float32".
    /// "auto" reads torch_dtype from config.json (default).
    #[arg(long, default_value = "auto")]
    pub dtype: String,

    /// Maximum model context length (overrides config.json).
    #[arg(long)]
    pub max_model_len: Option<usize>,

    /// Maximum number of concurrent sequences.
    #[arg(long, default_value_t = 256)]
    pub max_num_seqs: usize,

    /// HuggingFace token for gated models.
    #[arg(long, env = "HF_TOKEN")]
    pub hf_token: Option<String>,

    /// Log level: "trace", "debug", "info", "warn", "error".
    #[arg(long, default_value = "info")]
    pub log_level: String,

    /// Enable /metrics Prometheus endpoint.
    #[arg(long)]
    pub enable_metrics: bool,

    /// KV cache block size in tokens.
    #[arg(long, default_value_t = 16)]
    pub block_size: usize,

    /// Fraction of GPU memory to use for KV cache (0.0–1.0).
    #[arg(long, default_value_t = 0.9, env = "VLLM_GPU_MEMORY_UTILIZATION")]
    pub gpu_memory_utilization: f64,

    /// Specific GGUF filename to download from a HuggingFace repo.
    /// Example: --gguf-file llama-2-7b-chat.Q4_K_M.gguf
    #[arg(long)]
    pub gguf_file: Option<String>,

    /// Tool call parser to use (e.g. "hermes", "llama3_json").
    /// Enables structured tool call extraction from model output.
    #[arg(long)]
    pub tool_call_parser: Option<String>,

    /// Enable automatic tool choice (model decides when to call tools).
    #[arg(long)]
    pub enable_auto_tool_choice: bool,

    /// Path to SSL/TLS private key file (PEM format).
    #[arg(long)]
    pub ssl_keyfile: Option<String>,

    /// Path to SSL/TLS certificate file (PEM format).
    #[arg(long)]
    pub ssl_certfile: Option<String>,

    /// Path to CA certificates file for client certificate verification (PEM).
    #[arg(long)]
    pub ssl_ca_certs: Option<String>,
}

impl ServeArgs {
    /// Resolve the effective model path/ID.
    ///
    /// Priority: positional `model_tag` > `--model` flag > VLLM_MODEL env var.
    /// Returns an error if none is specified.
    pub fn resolved_model(&self) -> Result<String, String> {
        if let Some(ref tag) = self.model_tag {
            Ok(tag.clone())
        } else if let Some(ref m) = self.model {
            Ok(m.clone())
        } else {
            Err("model is required: provide as positional arg or --model flag".to_string())
        }
    }
}

/// Arguments for the `bench` subcommand.
#[derive(Parser, Debug)]
#[command(override_usage = "vllm bench [MODEL] [OPTIONS]")]
pub struct BenchArgs {
    /// Model: local path or HuggingFace model ID.
    pub model_tag: Option<String>,

    /// Path to a local model directory, or HuggingFace model ID.
    #[arg(long, env = "VLLM_MODEL")]
    pub model: Option<String>,

    /// Device: "cpu", "cuda:N", "metal", or "auto" (auto-detect best GPU).
    #[arg(long, default_value = "auto")]
    pub device: String,

    /// Weight dtype: "auto", "float16", "bfloat16", "float32".
    #[arg(long, default_value = "auto")]
    pub dtype: String,

    /// Number of requests to send.
    #[arg(long, default_value_t = 10)]
    pub num_requests: usize,

    /// Prompt length in tokens.
    #[arg(long, default_value_t = 32)]
    pub prompt_len: usize,

    /// Maximum output tokens per request.
    #[arg(long, default_value_t = 64)]
    pub max_tokens: usize,

    /// Number of warmup requests before timing.
    #[arg(long, default_value_t = 2)]
    pub warmup: usize,

    /// HuggingFace token.
    #[arg(long, env = "HF_TOKEN")]
    pub hf_token: Option<String>,

    /// Log level.
    #[arg(long, default_value = "info")]
    pub log_level: String,

    /// Fraction of GPU memory to use for KV cache (0.0–1.0).
    #[arg(long, default_value_t = 0.9, env = "VLLM_GPU_MEMORY_UTILIZATION")]
    pub gpu_memory_utilization: f64,

    /// Specific GGUF filename to download from a HuggingFace repo.
    #[arg(long)]
    pub gguf_file: Option<String>,
}

impl BenchArgs {
    /// Resolve the effective model path/ID.
    pub fn resolved_model(&self) -> Result<String, String> {
        if let Some(ref tag) = self.model_tag {
            Ok(tag.clone())
        } else if let Some(ref m) = self.model {
            Ok(m.clone())
        } else {
            Err("model is required: provide as positional arg or --model flag".to_string())
        }
    }
}

/// Arguments for the `convert` subcommand (stub).
#[derive(Parser, Debug)]
pub struct ConvertArgs {
    /// Input model directory.
    #[arg(long)]
    pub input: String,

    /// Output directory.
    #[arg(long)]
    pub output: String,

    /// Target dtype.
    #[arg(long, default_value = "f16")]
    pub dtype: String,
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use clap::CommandFactory;

    #[test]
    fn test_cli_help_doesnt_panic() {
        // Verify the CLI definition is valid.
        Cli::command().debug_assert();
    }

    #[test]
    fn test_parse_serve_positional_model() {
        // Python-compatible: `vllm serve meta-llama/Llama-3.2-1B`
        let cli = Cli::parse_from(["vllm", "serve", "meta-llama/Llama-3.2-1B", "--port", "9000"]);
        match cli.command {
            Commands::Serve(args) => {
                assert_eq!(args.resolved_model().unwrap(), "meta-llama/Llama-3.2-1B");
                assert_eq!(args.port, 9000);
            }
            _ => panic!("expected Serve command"),
        }
    }

    #[test]
    fn test_parse_serve_flag_model() {
        // Also supported: `vllm serve --model meta-llama/Llama-3.2-1B`
        let cli = Cli::parse_from([
            "vllm",
            "serve",
            "--model",
            "meta-llama/Llama-3.2-1B",
            "--device",
            "cpu",
        ]);
        match cli.command {
            Commands::Serve(args) => {
                assert_eq!(args.resolved_model().unwrap(), "meta-llama/Llama-3.2-1B");
                assert_eq!(args.device, "cpu");
            }
            _ => panic!("expected Serve command"),
        }
    }

    #[test]
    fn test_serve_positional_takes_precedence() {
        let cli = Cli::parse_from(["vllm", "serve", "positional-model", "--model", "flag-model"]);
        match cli.command {
            Commands::Serve(args) => {
                // Positional wins.
                assert_eq!(args.resolved_model().unwrap(), "positional-model");
            }
            _ => panic!("expected Serve command"),
        }
    }

    #[test]
    fn test_serve_no_model_errors() {
        let cli = Cli::parse_from(["vllm", "serve"]);
        match cli.command {
            Commands::Serve(args) => {
                assert!(args.resolved_model().is_err());
            }
            _ => panic!("expected Serve command"),
        }
    }

    #[test]
    fn test_parse_bench_positional_model() {
        let cli = Cli::parse_from(["vllm", "bench", "/path/to/model", "--num-requests", "5"]);
        match cli.command {
            Commands::Bench(args) => {
                assert_eq!(args.resolved_model().unwrap(), "/path/to/model");
                assert_eq!(args.num_requests, 5);
            }
            _ => panic!("expected Bench command"),
        }
    }

    #[test]
    fn test_parse_bench_flag_model() {
        let cli = Cli::parse_from(["vllm", "bench", "--model", "/path/to/model"]);
        match cli.command {
            Commands::Bench(args) => {
                assert_eq!(args.resolved_model().unwrap(), "/path/to/model");
            }
            _ => panic!("expected Bench command"),
        }
    }

    #[test]
    fn test_parse_convert_args() {
        let cli = Cli::parse_from([
            "vllm", "convert", "--input", "/in", "--output", "/out", "--dtype", "bf16",
        ]);
        match cli.command {
            Commands::Convert(args) => {
                assert_eq!(args.input, "/in");
                assert_eq!(args.output, "/out");
                assert_eq!(args.dtype, "bf16");
            }
            _ => panic!("expected Convert command"),
        }
    }
}
