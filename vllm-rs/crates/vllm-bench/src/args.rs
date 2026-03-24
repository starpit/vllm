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
    /// Benchmark online serving (send requests to a running server).
    Serve(BenchServeArgs),
    /// Benchmark the startup time of vLLM models.
    Startup(BenchStartupArgs),
    /// Parameter sweep: run benchmarks over multiple configurations.
    Sweep(SweepCommand),
    /// Benchmark offline throughput (batch generation).
    Throughput(BenchThroughputArgs),
    /// Benchmark relocatable KV cache blocks (spans) for RAG workloads.
    Spans(BenchSpansArgs),
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

    /// Maximum number of tokens processed in a single scheduler iteration.
    #[arg(long)]
    pub max_num_batched_tokens: Option<usize>,

    /// KV cache block size in tokens.
    #[arg(long, default_value_t = 16)]
    pub block_size: usize,

    /// Number of GPUs for tensor parallelism (default: 1).
    #[arg(long, default_value_t = 1)]
    pub tensor_parallel_size: usize,

    /// Disable prefix caching (KV cache reuse for shared prompt prefixes).
    /// Prefix caching is enabled by default, matching Python vLLM.
    #[arg(long)]
    pub no_prefix_caching: bool,

    /// Enable prefix caching (kept for compatibility; prefix caching is
    /// already enabled by default). Use --no-prefix-caching to disable.
    #[arg(long, hide = true)]
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
    #[arg(long, default_value = "auto")]
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

/// Arguments for `vllm bench throughput`.
///
/// Mirrors Python's `vllm bench throughput` — measures offline inference
/// throughput by generating a batch of random-length prompts through the
/// LLM engine and reporting requests/s and tokens/s.
#[derive(Parser, Debug)]
#[command(override_usage = "vllm bench throughput [MODEL] [OPTIONS]")]
pub struct BenchThroughputArgs {
    /// Model: local path or HuggingFace model ID (positional).
    pub model_tag: Option<String>,

    /// Path to a local model directory, or HuggingFace model ID.
    #[arg(short = 'm', long = "model", env = "VLLM_MODEL")]
    pub model: Option<String>,

    /// Device: "cpu", "cuda:N", "metal", or "auto" (auto-detect best GPU).
    #[arg(long, default_value = "auto")]
    pub device: String,

    /// Weight dtype: "auto", "float16", "bfloat16", "float32".
    #[arg(long, default_value = "auto")]
    pub dtype: String,

    /// Dataset name: "random" (default) or "sharegpt".
    #[arg(long, default_value = "random")]
    pub dataset_name: String,

    /// Path to dataset file (required for sharegpt).
    #[arg(long)]
    pub dataset_path: Option<String>,

    /// Number of prompts to process.
    #[arg(long, default_value_t = 1000)]
    pub num_prompts: usize,

    /// Input prompt length for each request (tokens).
    #[arg(long, default_value_t = 1024)]
    pub input_len: usize,

    /// Output length for each request (tokens).
    #[arg(long, default_value_t = 128)]
    pub output_len: usize,

    /// HuggingFace token.
    #[arg(long, env = "HF_TOKEN")]
    pub hf_token: Option<String>,

    /// Log level.
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

    /// Maximum number of tokens processed in a single scheduler iteration.
    #[arg(long)]
    pub max_num_batched_tokens: Option<usize>,

    /// KV cache block size in tokens.
    #[arg(long, default_value_t = 16)]
    pub block_size: usize,

    /// Number of GPUs for tensor parallelism.
    #[arg(long, default_value_t = 1)]
    pub tensor_parallel_size: usize,

    /// Specific GGUF filename to download from a HuggingFace repo.
    #[arg(long)]
    pub gguf_file: Option<String>,

    /// Path to write JSON results.
    #[arg(long)]
    pub output_json: Option<String>,

    /// Do not detokenize responses.
    #[arg(long)]
    pub disable_detokenize: bool,

    /// Disable CUDA graphs and run all steps eagerly.
    #[arg(long)]
    pub enforce_eager: bool,

    /// Comma-separated list of batch sizes to capture as CUDA graphs.
    #[arg(long, default_value = "auto")]
    pub cuda_graph_sizes: String,

    /// Disable prefix caching.
    #[arg(long)]
    pub no_prefix_caching: bool,

    /// Random seed for prompt generation.
    #[arg(long, default_value_t = 0)]
    pub seed: u64,

    /// Range ratio for sampling input/output lengths (random dataset only).
    /// Defines a symmetric range [len*(1-r), len*(1+r)]. Default 0.0 = fixed length.
    /// Must be in [0, 1). Matches Python's `--random-range-ratio`.
    #[arg(long, default_value_t = 0.0)]
    pub random_range_ratio: f64,

    /// Number of fixed prefix tokens prepended to each random prompt.
    /// Total input length = prefix_len + sampled input_len.
    /// Matches Python's `--random-prefix-len`.
    #[arg(long, default_value_t = 0)]
    pub random_prefix_len: usize,
}

impl BenchThroughputArgs {
    /// Resolve the model path/ID.
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

/// Arguments for `vllm bench serve`.
///
/// Mirrors Python's `vllm bench serve` — benchmarks online serving by
/// sending concurrent HTTP requests to a running vLLM server and measuring
/// TTFT, TPOT, ITL, and end-to-end latency.
#[derive(Parser, Debug)]
#[command(override_usage = "vllm bench serve [OPTIONS]")]
pub struct BenchServeArgs {
    /// Model name to use in API requests. If not specified, fetches the
    /// first model from the server's /v1/models endpoint.
    #[arg(long)]
    pub model: Option<String>,

    /// Tokenizer to use for prompt generation / length filtering.
    /// Defaults to the model name. Useful when the model name is not a
    /// HuggingFace repo (e.g. Ollama-style names like "llama3.2:3b") —
    /// set this to the corresponding HF repo ID.
    #[arg(long)]
    pub tokenizer: Option<String>,

    /// Server base URL.
    #[arg(long, default_value = "http://127.0.0.1:8000")]
    pub base_url: String,

    /// API endpoint path.
    #[arg(long, default_value = "/v1/completions")]
    pub endpoint: String,

    /// Number of prompts to send.
    #[arg(long, default_value_t = 1000)]
    pub num_prompts: usize,

    /// Input prompt length for each request (tokens).
    #[arg(long, default_value_t = 1024)]
    pub input_len: usize,

    /// Output length for each request (tokens).
    #[arg(long, default_value_t = 128)]
    pub output_len: usize,

    /// Request rate (requests/sec). Use "inf" for all-at-once.
    #[arg(long, default_value_t = f64::INFINITY)]
    pub request_rate: f64,

    /// Maximum number of concurrent requests.
    #[arg(long)]
    pub max_concurrency: Option<usize>,

    /// Random seed.
    #[arg(long, default_value_t = 0)]
    pub seed: u64,

    /// Disable tqdm-style progress bar.
    #[arg(long)]
    pub disable_tqdm: bool,

    /// Path to save JSON results.
    #[arg(long)]
    pub output_json: Option<String>,

    /// Comma-separated percentile metrics to report: ttft,tpot,itl,e2el.
    #[arg(long, default_value = "ttft,tpot,itl")]
    pub percentile_metrics: String,

    /// Comma-separated percentiles to report (e.g. 50,90,99).
    #[arg(long, default_value = "99")]
    pub metric_percentiles: String,

    /// Burstiness of request arrival pattern.
    /// 1.0 = Poisson (exponential delays), <1.0 = bursty, >1.0 = more uniform.
    /// Use "inf" for constant inter-request delay.
    #[arg(long, default_value_t = 1.0)]
    pub burstiness: f64,

    /// Number of warmup requests to send before timing.
    #[arg(long, default_value_t = 0)]
    pub num_warmups: usize,

    /// Dataset name: "random" (default) or "sharegpt".
    #[arg(long, default_value = "random")]
    pub dataset_name: String,

    /// Path to dataset file (required for sharegpt).
    #[arg(long)]
    pub dataset_path: Option<String>,

    /// Ignore EOS token (force generation to max output length).
    /// Defaults to true for random prompts, matching Python's behavior.
    #[arg(long, default_value_t = true)]
    pub ignore_eos: bool,

    /// Sampling temperature.
    #[arg(long)]
    pub temperature: Option<f64>,

    /// Top-p sampling parameter.
    #[arg(long)]
    pub top_p: Option<f64>,

    /// Top-k sampling parameter.
    #[arg(long)]
    pub top_k: Option<i32>,

    /// API key for authentication.
    #[arg(long, env = "OPENAI_API_KEY")]
    pub api_key: Option<String>,

    /// Range ratio for sampling input/output lengths (random dataset only).
    /// Defines a symmetric range [len*(1-r), len*(1+r)]. Default 0.0 = fixed length.
    /// Must be in [0, 1). Matches Python's `--random-range-ratio`.
    #[arg(long, default_value_t = 0.0)]
    pub random_range_ratio: f64,

    /// Number of fixed prefix tokens prepended to each random prompt.
    /// Total input length = prefix_len + sampled input_len.
    /// Matches Python's `--random-prefix-len`.
    #[arg(long, default_value_t = 0)]
    pub random_prefix_len: usize,

    /// Disable SSL certificate verification.
    #[arg(long)]
    pub insecure: bool,

    /// Save benchmark results to a JSON file (auto-generated filename).
    #[arg(long)]
    pub save_result: bool,

    /// Directory to save results in (used with --save-result).
    #[arg(long)]
    pub result_dir: Option<String>,

    /// Override auto-generated result filename (used with --save-result).
    #[arg(long)]
    pub result_filename: Option<String>,

    /// Label for this benchmark run (used in auto-generated filenames).
    #[arg(long)]
    pub label: Option<String>,
}

/// Arguments for `vllm bench startup`.
///
/// Mirrors Python's `vllm bench startup` — measures cold and warm startup time
/// by repeatedly constructing the LLM engine.
#[derive(Parser, Debug)]
#[command(override_usage = "vllm bench startup [MODEL] [OPTIONS]")]
pub struct BenchStartupArgs {
    /// Model: local path or HuggingFace model ID (positional).
    pub model_tag: Option<String>,

    /// Path to a local model directory, or HuggingFace model ID.
    #[arg(short = 'm', long = "model", env = "VLLM_MODEL")]
    pub model: Option<String>,

    /// Device: "cpu", "cuda:N", "metal", or "auto".
    #[arg(long, default_value = "auto")]
    pub device: String,

    /// Weight dtype: "auto", "float16", "bfloat16", "float32".
    #[arg(long, default_value = "auto")]
    pub dtype: String,

    /// Number of cold startup iterations.
    #[arg(long, default_value_t = 3)]
    pub num_iters_cold: usize,

    /// Number of warmup iterations before benchmarking warm startups.
    #[arg(long, default_value_t = 1)]
    pub num_iters_warmup: usize,

    /// Number of warm startup iterations.
    #[arg(long, default_value_t = 3)]
    pub num_iters_warm: usize,

    /// HuggingFace token.
    #[arg(long, env = "HF_TOKEN")]
    pub hf_token: Option<String>,

    /// Log level.
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

    /// Maximum number of tokens processed in a single scheduler iteration.
    #[arg(long)]
    pub max_num_batched_tokens: Option<usize>,

    /// KV cache block size in tokens.
    #[arg(long, default_value_t = 16)]
    pub block_size: usize,

    /// Number of GPUs for tensor parallelism.
    #[arg(long, default_value_t = 1)]
    pub tensor_parallel_size: usize,

    /// Disable prefix caching.
    #[arg(long)]
    pub no_prefix_caching: bool,

    /// Specific GGUF filename to download from a HuggingFace repo.
    #[arg(long)]
    pub gguf_file: Option<String>,

    /// Path to write JSON results.
    #[arg(long)]
    pub output_json: Option<String>,

    /// Disable CUDA graphs and run all steps eagerly.
    #[arg(long)]
    pub enforce_eager: bool,

    /// Comma-separated list of batch sizes to capture as CUDA graphs.
    #[arg(long, default_value = "auto")]
    pub cuda_graph_sizes: String,
}

impl BenchStartupArgs {
    /// Resolve the model path/ID.
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

// ---------------------------------------------------------------------------
// Sweep
// ---------------------------------------------------------------------------

/// Container for `sweep` subcommands.
#[derive(Parser, Debug)]
pub struct SweepCommand {
    #[command(subcommand)]
    pub command: SweepCommands,
}

#[derive(Subcommand, Debug)]
pub enum SweepCommands {
    /// Run vLLM server benchmark under multiple settings.
    Serve(SweepServeArgs),
    /// Benchmark vLLM startup time over parameter combinations.
    Startup(SweepStartupArgs),
}

/// Arguments for `vllm bench sweep serve`.
#[derive(Parser, Debug)]
pub struct SweepServeArgs {
    /// The command used to run the server (e.g. "vllm serve model --port 8000").
    #[arg(long)]
    pub serve_cmd: String,

    /// The command used to run the benchmark (e.g. "vllm bench serve ...").
    #[arg(long)]
    pub bench_cmd: String,

    /// Path to JSON file with parameter combinations for `vllm serve`.
    #[arg(long)]
    pub serve_params: Option<String>,

    /// Path to JSON file with parameter combinations for `vllm bench serve`.
    #[arg(long)]
    pub bench_params: Option<String>,

    /// Output directory for results.
    #[arg(short = 'o', long, default_value = "results")]
    pub output_dir: String,

    /// Number of runs per parameter combination.
    #[arg(long, default_value_t = 3)]
    pub num_runs: usize,

    /// Print commands without executing them.
    #[arg(long)]
    pub dry_run: bool,

    /// Resume a previous sweep from a timestamped directory.
    #[arg(long)]
    pub resume: Option<String>,

    /// Show stdout from sub-processes.
    #[arg(long)]
    pub show_stdout: bool,

    /// Timeout (seconds) to wait for the server to become ready.
    #[arg(long, default_value_t = 300)]
    pub server_ready_timeout: u32,
}

/// Arguments for `vllm bench sweep startup`.
#[derive(Parser, Debug)]
pub struct SweepStartupArgs {
    /// The command used to run the startup benchmark.
    #[arg(long, default_value = "vllm bench startup")]
    pub startup_cmd: String,

    /// Path to JSON file with parameter combinations for serve/model args.
    #[arg(long)]
    pub serve_params: Option<String>,

    /// Path to JSON file with parameter combinations for startup args.
    #[arg(long)]
    pub startup_params: Option<String>,

    /// Output directory for results.
    #[arg(short = 'o', long, default_value = "results")]
    pub output_dir: String,

    /// Number of runs per parameter combination.
    #[arg(long, default_value_t = 1)]
    pub num_runs: usize,

    /// Print commands without executing them.
    #[arg(long)]
    pub dry_run: bool,

    /// Resume a previous sweep from a timestamped directory.
    #[arg(long)]
    pub resume: Option<String>,

    /// Show stdout from sub-processes.
    #[arg(long)]
    pub show_stdout: bool,
}

// ---------------------------------------------------------------------------
// Spans benchmark args
// ---------------------------------------------------------------------------

/// Arguments for `vllm bench spans`.
#[derive(Parser, Debug)]
#[command(override_usage = "vllm bench spans [MODEL] [OPTIONS]")]
pub struct BenchSpansArgs {
    /// Model: local path or HuggingFace model ID.
    pub model_tag: Option<String>,

    /// Model (--model flag or VLLM_MODEL env).
    #[arg(short = 'm', long = "model", env = "VLLM_MODEL")]
    pub model: Option<String>,

    /// Device: "cpu", "cuda:N", "metal", or "auto".
    #[arg(long, default_value = "auto")]
    pub device: String,

    /// Weight dtype: "auto", "float16", "bfloat16", "float32".
    #[arg(long, default_value = "auto")]
    pub dtype: String,

    /// Number of documents to preload.
    #[arg(long, default_value_t = 3)]
    pub num_docs: usize,

    /// Number of cache blocks per document.
    /// Each document is `doc_blocks x block_size` tokens.
    #[arg(long, default_value_t = 4)]
    pub doc_blocks: usize,

    /// KV cache block size in tokens.
    #[arg(long, default_value_t = 16)]
    pub block_size: usize,

    /// Number of query tokens appended after documents.
    #[arg(long, default_value_t = 32)]
    pub query_len: usize,

    /// Maximum number of permutations to test (all if n! <= this).
    #[arg(long, default_value_t = 24)]
    pub max_perms: usize,

    /// Token ID used for padding blocks to block_size boundaries.
    #[arg(long, default_value_t = 0)]
    pub pad_token: u32,

    /// HuggingFace token for gated models.
    #[arg(long, env = "HF_TOKEN")]
    pub hf_token: Option<String>,

    /// Fraction of GPU memory to use for KV cache.
    #[arg(long, default_value_t = 0.9)]
    pub gpu_memory_utilization: f64,

    /// Maximum number of concurrent sequences.
    #[arg(long, default_value_t = 256)]
    pub max_num_seqs: usize,

    /// Maximum model context length override.
    #[arg(long)]
    pub max_model_len: Option<usize>,

    /// GGUF filename for quantized models.
    #[arg(long)]
    pub gguf_file: Option<String>,

    /// Disable CUDA graphs (use eager mode).
    #[arg(long)]
    pub enforce_eager: bool,

    /// Log level.
    #[arg(long, default_value = "warn")]
    pub log_level: String,

    /// Run nested generate benchmark: N inner generates followed by an outer
    /// generate that reuses their outputs via seal + volatile.
    #[arg(long)]
    pub nested: Option<usize>,

    /// Number of output tokens for each inner generate (nested mode).
    #[arg(long, default_value_t = 64)]
    pub inner_tokens: u32,
}

impl BenchSpansArgs {
    /// Resolve the model path from positional or --model args.
    pub fn resolved_model(&self) -> Result<String, String> {
        match (&self.model_tag, &self.model) {
            (Some(tag), _) => Ok(tag.clone()),
            (None, Some(m)) => Ok(m.clone()),
            (None, None) => Err("No model specified. Use positional arg or --model.".into()),
        }
    }
}
