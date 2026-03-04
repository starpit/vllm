// SPDX-License-Identifier: Apache-2.0
// Copyright contributors to the vLLM project

//! CLI argument definitions for `vllm bench` subcommands.

use clap::{Parser, Subcommand};

/// Container for `bench` subcommands.
#[derive(Parser, Debug)]
pub struct BenchCommand {
    #[command(subcommand)]
    pub command: BenchCommands,
}

#[derive(Subcommand, Debug)]
pub enum BenchCommands {
    /// Measure latency (TTFT, ITL, throughput) for a model.
    Latency(Box<BenchLatencyArgs>),
    /// Benchmark online serving throughput (not yet implemented).
    Serve(BenchServeArgs),
    /// Benchmark offline throughput (not yet implemented).
    Throughput(BenchThroughputArgs),
}

/// Arguments for `vllm bench latency`.
#[derive(Parser, Debug)]
#[command(override_usage = "vllm bench latency [MODEL...] [OPTIONS]")]
pub struct BenchLatencyArgs {
    /// Model(s): local path or HuggingFace model ID (positional, repeatable).
    pub model_tags: Vec<String>,

    /// Path to a local model directory, or HuggingFace model ID (repeatable, comma-separated).
    #[arg(short = 'm', long = "model", env = "VLLM_MODEL", value_delimiter = ',')]
    pub models: Vec<String>,

    /// Device: "cpu", "cuda:N", "metal", or "auto" (auto-detect best GPU).
    #[arg(long, default_value = "auto")]
    pub device: String,

    /// Weight dtype: "auto", "float16", "bfloat16", "float32".
    #[arg(long, default_value = "auto")]
    pub dtype: String,

    /// Number of benchmark iterations.
    #[arg(long, default_value_t = 30)]
    pub num_iters: usize,

    /// Input prompt length in tokens.
    #[arg(long, default_value_t = 32)]
    pub input_len: usize,

    /// Number of output tokens per iteration.
    #[arg(long, default_value_t = 128)]
    pub output_len: usize,

    /// Number of warmup iterations before timing.
    #[arg(long, default_value_t = 10)]
    pub num_iters_warmup: usize,

    /// Number of requests per iteration (batch size). Repeatable/comma-separated.
    #[arg(
        short = 'b',
        long = "batch-size",
        value_delimiter = ',',
        default_value = "8"
    )]
    pub batch_sizes: Vec<usize>,

    /// Percentiles to display (comma-separated, e.g. 10,50,90,99).
    #[arg(long = "percentile", value_delimiter = ',', default_value = "50")]
    pub percentiles: Vec<f64>,

    /// HuggingFace token.
    #[arg(long, env = "HF_TOKEN")]
    pub hf_token: Option<String>,

    /// Log level (default "warn" to suppress per-request engine logs;
    /// use "info" or "debug" for verbose output).
    #[arg(long, default_value = "warn")]
    pub log_level: String,

    /// Fraction of GPU memory to use for KV cache (0.0–1.0).
    #[arg(long, default_value_t = 0.9, env = "VLLM_GPU_MEMORY_UTILIZATION")]
    pub gpu_memory_utilization: f64,

    /// Maximum model context length (overrides config.json).
    #[arg(long)]
    pub max_model_len: Option<usize>,

    /// Maximum number of concurrent sequences.
    #[arg(long, default_value_t = 256)]
    pub max_num_seqs: usize,

    /// KV cache block size in tokens.
    #[arg(long, default_value_t = 16)]
    pub block_size: usize,

    /// Number of GPUs for tensor parallelism (default: 1).
    #[arg(long, default_value_t = 1)]
    pub tensor_parallel_size: usize,

    /// Enable prefix caching (KV cache reuse for shared prompt prefixes).
    /// Disabled by default for benchmarking — prefix caching skews latency
    /// because repeated prompts get cache hits.
    #[arg(long)]
    pub enable_prefix_caching: bool,

    /// Specific GGUF filename to download from a HuggingFace repo.
    #[arg(long)]
    pub gguf_file: Option<String>,

    /// Path to write JSON results.
    #[arg(long)]
    pub output_json: Option<String>,

    /// Disable CUDA graphs and run all steps eagerly.
    /// Default: false (CUDA graphs enabled on CUDA devices).
    #[arg(long)]
    pub enforce_eager: bool,

    /// Comma-separated list of batch sizes to capture as CUDA graphs.
    #[arg(long, default_value = "1,2,4,8,16,32,64,128,256")]
    pub cuda_graph_sizes: String,

    /// Sampling temperature (0 = greedy, >0 = random sampling).
    #[arg(long, default_value_t = 1.0)]
    pub temperature: f64,

    /// Top-p (nucleus) sampling cutoff (1.0 = disabled).
    #[arg(long, default_value_t = 1.0)]
    pub top_p: f64,

    /// Top-k sampling cutoff (0 = disabled).
    #[arg(long, default_value_t = 0)]
    pub top_k: i32,

    /// Do not detokenize responses (excludes detokenization time from latency).
    #[arg(long)]
    pub disable_detokenize: bool,
}

impl BenchLatencyArgs {
    /// Resolve the list of models by merging positional and --model values.
    pub fn resolved_models(&self) -> Result<Vec<String>, String> {
        let mut all: Vec<String> = self.model_tags.clone();
        all.extend(self.models.clone());
        if all.is_empty() {
            Err("model is required: provide as positional arg or --model flag".to_string())
        } else {
            Ok(all)
        }
    }
}

/// Arguments for `vllm bench serve` (stub).
#[derive(Parser, Debug)]
pub struct BenchServeArgs {}

/// Arguments for `vllm bench throughput` (stub).
#[derive(Parser, Debug)]
pub struct BenchThroughputArgs {}
