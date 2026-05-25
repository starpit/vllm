// SPDX-License-Identifier: Apache-2.0
// Copyright contributors to the vLLM project

//! CLI argument definitions using clap derive.

use clap::{Parser, Subcommand};

#[cfg(feature = "bench")]
pub use vllm_bench::BenchCommand;
#[cfg(feature = "bench")]
#[cfg(test)]
pub use vllm_bench::BenchCommands;

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
    Serve(Box<ServeArgs>),
    /// Run benchmarks (latency, serving, throughput).
    #[cfg(feature = "bench")]
    Bench(BenchCommand),
    /// Process a batch of OpenAI-compatible requests offline.
    Batch(BatchArgs),
    /// Process a batch of OpenAI-compatible requests offline (alias for `batch`).
    #[command(name = "run-batch")]
    RunBatch(BatchArgs),
    /// Generate chat completions via the running API server.
    Chat(ChatArgs),
    /// Collect and print environment information for bug reports.
    CollectEnv(CollectEnvArgs),
    /// Manage GCE GPU VM instances.
    #[cfg(feature = "gce")]
    Gce(GceCommand),
    /// Deploy to Kubernetes with GPU support.
    #[cfg(feature = "k8s")]
    K8s(K8sCommand),
    /// Generate text completions via the running API server.
    Complete(CompleteArgs),
    /// Convert model weights between formats (stub).
    Convert(ConvertArgs),
    /// Download a model from HuggingFace Hub without starting the server.
    Pull(PullArgs),
    /// List cached models.
    Ls(ListArgs),
    /// List cached models.
    List(ListArgs),
    /// Remove a cached model.
    Rm(RmArgs),
    /// Live TUI dashboard — monitor a running vllm server.
    #[cfg(feature = "top")]
    Top(TopArgs),
    /// Ferrite tooling: inspect compiled-in model backbones, etc.
    #[cfg(any(feature = "cuda", feature = "metal"))]
    Ferrite(FerriteCommand),
}

/// `vllm ferrite <subcommand>`.
#[cfg(any(feature = "cuda", feature = "metal"))]
#[derive(Parser, Debug)]
pub struct FerriteCommand {
    #[command(subcommand)]
    pub command: FerriteSubcommand,
}

#[cfg(any(feature = "cuda", feature = "metal"))]
#[derive(Subcommand, Debug)]
pub enum FerriteSubcommand {
    /// Print the per-bucket backbone instruction list for compiled
    /// ferrite variants. Optional positional filters AND-substring
    /// match against `<arch>/<variant_stem>` — e.g.
    /// `vllm ferrite info llama 3.2 awq`.
    Info(FerriteInfoArgs),
}

#[cfg(any(feature = "cuda", feature = "metal"))]
#[derive(Parser, Debug)]
pub struct FerriteInfoArgs {
    /// Color/style output. `auto` uses ANSI when stdout is a tty
    /// and `NO_COLOR` is unset; `always` forces it on (useful when
    /// piping into `less -R`); `never` disables.
    #[arg(long, value_enum, default_value_t = ColorWhen::Auto)]
    pub color: ColorWhen,
    /// Run cuBLAS-pick analysis instead of the per-bucket backbone
    /// dump: for every `Cublas` Gemm pick, classify by margin vs.
    /// the best non-cuBLAS standalone-GEMM kernel and identify
    /// fusion-gap reasons (Gemm→Add / Norm→Gemm / Gemm→ScalarMul /
    /// lm_head). Uses the bundled cost CSV for exact-row lookups.
    /// cuda-only — keyed off `ferrite_cuda_targets` profiles.
    #[cfg(feature = "cuda")]
    #[arg(short = 'c', long = "cublas-analysis")]
    pub cublas_analysis: bool,
    /// With `-c/--cublas-analysis`, also print a per-arch breakdown.
    #[cfg(feature = "cuda")]
    #[arg(long, requires = "cublas_analysis")]
    pub per_arch: bool,
    /// Substring filters. A variant is shown when its
    /// `<arch>/<variant_stem>` contains every filter (case-
    /// insensitive). Empty = show every compiled variant.
    pub filters: Vec<String>,
}

#[cfg(any(feature = "cuda", feature = "metal"))]
#[derive(clap::ValueEnum, Clone, Copy, Debug, Default)]
pub enum ColorWhen {
    #[default]
    Auto,
    Always,
    Never,
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
    #[arg(short = 'm', long, env = "VLLM_MODEL")]
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

    /// Maximum number of tokens processed in a single scheduler iteration.
    #[arg(long)]
    pub max_num_batched_tokens: Option<usize>,

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

    /// Reasoning parser to use (e.g. "deepseek_r1", "qwen3").
    /// Extracts <think>...</think> blocks into a separate reasoning_content field.
    #[arg(long)]
    pub reasoning_parser: Option<String>,

    /// Default extra kwargs for the chat template, as JSON.
    /// Merged with per-request chat_template_kwargs (request overrides).
    /// Example: '{"enable_thinking": false}'
    #[arg(long, value_parser = parse_json_map)]
    pub default_chat_template_kwargs: Option<std::collections::HashMap<String, serde_json::Value>>,

    /// Chat template override. Accepts either an inline Jinja string
    /// or a path to a `tokenizer_config.json` / `.jinja` file. Used
    /// for GGUFs whose metadata lacks `tokenizer.chat_template`.
    #[arg(long)]
    pub chat_template: Option<String>,

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

    /// Pooling strategy for /v1/embeddings: "auto", "last", "cls", "mean".
    /// "auto" detects from 1_Pooling/config.json, defaults to "last".
    #[arg(long, default_value = "auto")]
    pub pooling_strategy: String,

    /// Speculative decoding model.
    ///
    /// `ngram` selects the n-gram proposer (matches in the request's own
    /// token history). Any other value is treated as a draft-model local
    /// path or HuggingFace repo ID; the draft-model proposer is wired in
    /// phase 4 of `vllm-rs/DRAFT_SPEC_DECODE_PLAN.md` and the engine will
    /// refuse to start until then.
    #[arg(long)]
    pub speculative_model: Option<String>,

    /// Number of speculative tokens to propose per step (default: 2).
    /// Only used when --speculative-model is set.
    ///
    /// K=2 is the empirical sweet spot for the realistic draft-model
    /// regime (target much larger than draft, e.g. Llama-3.1-8B
    /// target + Llama-3.2-1B draft on Apple Silicon). 5-run distribution
    /// at 8B+1B, M4: baseline 45 ms TPOT → K=1 34, K=2 32, K=4 39,
    /// K=6 53 (worse than baseline; chain cost overwhelms amortization
    /// and acceptance drops past K=2). Raise this only after measuring
    /// on the target+draft pair you care about; on smaller targets
    /// (e.g. 3B+1B) spec decode loses at any K and the right answer
    /// is no `--speculative-model`.
    #[arg(long, default_value_t = 2)]
    pub num_speculative_tokens: usize,

    /// Maximum n-gram size for prompt lookup (default: 4).
    /// Only used when --speculative-model ngram.
    #[arg(long, default_value_t = 4)]
    pub ngram_prompt_lookup_max: usize,

    /// Minimum n-gram size for prompt lookup (default: 1).
    /// Only used when --speculative-model ngram.
    #[arg(long, default_value_t = 1)]
    pub ngram_prompt_lookup_min: usize,

    /// Optional dtype override for the draft model's weights
    /// ("auto", "float16", "bfloat16", "float32"). Mirrors Python's
    /// `--speculative-config.draft_model_dtype`. Ignored unless
    /// --speculative-model points at a draft model.
    #[arg(long)]
    pub draft_model_dtype: Option<String>,

    /// LoRA adapter to load. Path to a local directory containing
    /// adapter_config.json and adapter_model.safetensors, or a
    /// HuggingFace repo ID.
    #[arg(long)]
    pub lora_adapter: Option<String>,

    /// Number of GPUs for tensor parallelism (default: 1).
    /// Splits model weights across N GPUs using NCCL all-reduce.
    #[arg(long, default_value_t = 1)]
    pub tensor_parallel_size: usize,

    /// Number of GPU stages for pipeline parallelism (default: 1).
    /// Splits model layers across N GPU stages using NCCL P2P send/recv.
    /// Total GPUs used = tensor_parallel_size * pipeline_parallel_size.
    #[arg(long, default_value_t = 1)]
    pub pipeline_parallel_size: usize,

    /// Number of nodes for multi-node tensor parallelism (default: 1).
    /// When > 1, GPUs are split across nodes using TCP rendezvous for NCCL.
    #[arg(long, default_value_t = 1)]
    pub num_nodes: usize,

    /// This node's rank in the multi-node setup (0 = master, default: 0).
    #[arg(long, default_value_t = 0)]
    pub node_rank: usize,

    /// Master node address for multi-node NCCL rendezvous (default: localhost).
    #[arg(long, default_value = "localhost")]
    pub master_addr: String,

    /// Master port for multi-node NCCL rendezvous (default: 29500).
    #[arg(long, default_value_t = 29500)]
    pub master_port: u16,

    /// Disable async scheduling (overlap of GPU execution and CPU scheduling).
    /// By default, async scheduling is enabled for better throughput.
    #[arg(long)]
    pub disable_async_scheduling: bool,

    /// Runner type: "generate" (default) or "pooling".
    /// In pooling mode, embedding requests go through the scheduler and
    /// generation endpoints (chat, completions) are rejected.
    #[arg(long, default_value = "generate")]
    pub runner: String,

    /// Comma-separated CUDA graph capture batch sizes.
    /// CUDA graphs accelerate decode steps by replaying a captured kernel
    /// sequence in a single driver call.
    #[arg(long, default_value = "auto")]
    pub cuda_graph_sizes: String,

    /// Disable CUDA graphs and run all steps eagerly.
    /// Equivalent to Python vLLM's --enforce-eager flag.
    /// Default: false (CUDA graphs enabled on CUDA devices).
    #[arg(long)]
    pub enforce_eager: bool,

    /// Distributed executor backend: "auto" (default) or "external_launcher".
    /// With "external_launcher", the job launcher (torchrun, mpirun, SLURM)
    /// spawns N processes. Each reads RANK, LOCAL_RANK, WORLD_SIZE,
    /// MASTER_ADDR, MASTER_PORT from env and runs its own engine instance.
    #[arg(long, default_value = "auto")]
    pub distributed_executor_backend: String,

    /// CUDA graph mode: controls piecewise vs monolithic graph capture.
    /// Options: "auto", "none", "full", "piecewise", "full-and-piecewise", "full-decode-only".
    /// - "auto": Full for SM < 90 (Ampere/Ada), FullAndPiecewise for SM >= 90 (Hopper+)
    /// - "none": No CUDA graphs (same as --enforce-eager)
    ///
    /// Default: "auto".
    #[arg(long, default_value = "auto")]
    pub cuda_graph_mode: String,

    /// Disable prefix caching (KV cache reuse for shared prompt prefixes).
    /// By default, prefix caching is enabled.
    #[arg(long)]
    pub no_prefix_caching: bool,

    /// Benchmark cublasLt algorithms during warmup to find faster GEMM kernels.
    /// Adds a few seconds to startup. Mainly benefits compute-bound prefill GEMMs.
    #[arg(long)]
    pub cublas_autotune: bool,

    /// KV cache data type: "auto" (use model dtype) or "fp8_e4m3" (FP8).
    /// FP8 halves KV cache memory, doubling capacity.
    #[arg(long, default_value = "auto")]
    pub kv_cache_dtype: String,

    /// Compute KV scales dynamically from the first forward pass.
    /// Only used with --kv-cache-dtype fp8_e4m3.
    #[arg(long)]
    pub calculate_kv_scales: bool,

    /// Target URL for OpenTelemetry traces (OTLP gRPC endpoint).
    /// Example: http://localhost:4317
    /// Requires building with --features otel.
    #[arg(long, env = "OTEL_EXPORTER_OTLP_TRACES_ENDPOINT")]
    pub otlp_traces_endpoint: Option<String>,
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

/// Arguments for the `chat` subcommand.
///
/// Supports two modes:
/// - **In-process** (default when `--model` is given): loads the model locally
///   and runs inference directly — no server needed.
/// - **Remote** (when `--url` is given without `--model`): connects to a
///   running vLLM server's OpenAI-compatible API.
#[derive(Parser, Debug)]
#[command(override_usage = "vllm chat [MODEL] [OPTIONS]")]
pub struct ChatArgs {
    /// Model to load in-process: local path or HuggingFace model ID.
    /// When set, runs inference locally without needing a running server.
    pub model_tag: Option<String>,

    /// Model: local path or HuggingFace model ID (alternative to positional arg).
    #[arg(short = 'm', long, env = "VLLM_MODEL")]
    pub model: Option<String>,

    /// URL of a running OpenAI-compatible API server (remote mode).
    /// Used only when no --model is specified.
    #[arg(long, default_value = "http://localhost:8000/v1")]
    pub url: String,

    /// Model name to request from the remote server.
    #[arg(long)]
    pub model_name: Option<String>,

    /// API key for remote server authentication.
    #[arg(long, env = "OPENAI_API_KEY")]
    pub api_key: Option<String>,

    /// System prompt to prepend to the conversation.
    #[arg(long)]
    pub system_prompt: Option<String>,

    /// Send a single message and exit (non-interactive mode).
    #[arg(short = 'q', long, value_name = "MESSAGE")]
    pub quick: Option<String>,

    /// Send prompt(s) and exit. Multiple values become separate turns in
    /// a multi-turn conversation (model responds to each in order).
    #[arg(short = 'p', long, value_name = "PROMPT", num_args = 1..)]
    pub prompt: Vec<String>,

    /// Print performance metrics after generation: startup time, TTFT,
    /// inter-token latency (ITL), and tokens/sec. Best used with --prompt.
    #[arg(long)]
    pub bench: bool,

    /// Device: "cpu", "cuda:N", "metal", or "auto" (auto-detect best GPU).
    #[arg(long, default_value = "auto")]
    pub device: String,

    /// Weight dtype: "auto", "float16", "bfloat16", "float32".
    #[arg(long, default_value = "auto")]
    pub dtype: String,

    /// HuggingFace token for gated models.
    #[arg(long, env = "HF_TOKEN")]
    pub hf_token: Option<String>,

    /// Specific GGUF filename to download from a HuggingFace repo.
    #[arg(long)]
    pub gguf_file: Option<String>,

    /// Maximum model context length (overrides config.json).
    #[arg(long)]
    pub max_model_len: Option<usize>,

    /// Maximum number of tokens to generate per response.
    #[arg(long)]
    pub max_tokens: Option<u32>,

    /// Sampling temperature (0.0 = greedy, 1.0 = default).
    #[arg(long)]
    pub temperature: Option<f64>,

    /// Number of GPUs for tensor parallelism (default: 1).
    #[arg(long, default_value_t = 1)]
    pub tensor_parallel_size: usize,

    /// Disable CUDA graphs (use eager mode).
    #[arg(long)]
    pub enforce_eager: bool,

    /// Chat template override. Accepts either an inline Jinja string
    /// or a path to a `tokenizer_config.json` / `.jinja` file. Useful
    /// for GGUFs whose metadata lacks `tokenizer.chat_template`.
    #[arg(long)]
    pub chat_template: Option<String>,
}

impl ChatArgs {
    /// Resolve the effective model path/ID (if any).
    pub fn resolved_model(&self) -> Option<String> {
        self.model_tag.clone().or_else(|| self.model.clone())
    }
}

/// Arguments for the `complete` subcommand.
#[derive(Parser, Debug)]
#[command(override_usage = "vllm complete [OPTIONS]")]
pub struct CompleteArgs {
    /// URL of the running OpenAI-compatible API server.
    #[arg(long, default_value = "http://localhost:8000/v1")]
    pub url: String,

    /// Model name for completions (default: first model from server).
    #[arg(long)]
    pub model_name: Option<String>,

    /// API key for authentication.
    #[arg(long, env = "OPENAI_API_KEY")]
    pub api_key: Option<String>,

    /// Maximum number of tokens to generate.
    #[arg(long)]
    pub max_tokens: Option<usize>,

    /// Send a single prompt and exit (non-interactive mode).
    #[arg(short = 'q', long, value_name = "PROMPT")]
    pub quick: Option<String>,
}

/// Arguments for the `collect-env` subcommand (no options).
#[derive(Parser, Debug)]
#[command(override_usage = "vllm collect-env")]
pub struct CollectEnvArgs {}

/// Arguments for the `batch` subcommand.
#[derive(Parser, Debug)]
#[command(override_usage = "vllm batch [MODEL] [OPTIONS]")]
pub struct BatchArgs {
    /// Model: local path or HuggingFace model ID.
    pub model_tag: Option<String>,

    /// Path to a local model directory, or HuggingFace model ID.
    #[arg(short = 'm', long, env = "VLLM_MODEL")]
    pub model: Option<String>,

    /// Input JSONL file containing batch requests.
    #[arg(short = 'i', long)]
    pub input: String,

    /// Output JSONL file for batch results.
    #[arg(short = 'o', long)]
    pub output: String,

    /// Device: "cpu", "cuda:N", "metal", or "auto" (auto-detect best GPU).
    #[arg(long, default_value = "auto")]
    pub device: String,

    /// Weight dtype: "auto", "float16", "bfloat16", "float32".
    #[arg(long, default_value = "auto")]
    pub dtype: String,

    /// HuggingFace token for gated models.
    #[arg(long, env = "HF_TOKEN")]
    pub hf_token: Option<String>,

    /// Log level: "trace", "debug", "info", "warn", "error".
    #[arg(long, default_value = "info")]
    pub log_level: String,

    /// Fraction of GPU memory to use for KV cache (0.0–1.0).
    #[arg(long, default_value_t = 0.9, env = "VLLM_GPU_MEMORY_UTILIZATION")]
    pub gpu_memory_utilization: f64,

    /// Tool call parser to use (e.g. "hermes", "llama3_json").
    #[arg(long)]
    pub tool_call_parser: Option<String>,

    /// Reasoning parser to use (e.g. "deepseek_r1", "qwen3").
    #[arg(long)]
    pub reasoning_parser: Option<String>,

    /// Default extra kwargs for the chat template, as JSON.
    /// Merged with per-request chat_template_kwargs (request overrides).
    /// Example: '{"enable_thinking": false}'
    #[arg(long, value_parser = parse_json_map)]
    pub default_chat_template_kwargs: Option<std::collections::HashMap<String, serde_json::Value>>,

    /// Specific GGUF filename to download from a HuggingFace repo.
    #[arg(long)]
    pub gguf_file: Option<String>,
}

impl BatchArgs {
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

/// Arguments for `vllm rm`.
#[derive(Parser, Debug)]
#[command(override_usage = "vllm rm <MODEL>")]
pub struct RmArgs {
    /// Model to remove (e.g. "google/gemma-2-2b"). Must match the model ID shown by `vllm ls`.
    pub model: String,
}

/// Arguments for `vllm ls` / `vllm list`.
#[derive(Parser, Debug)]
pub struct ListArgs {
    /// Sort order: "name" (default) or "size".
    #[arg(short = 's', long, default_value = "name")]
    pub sort: ListSort,
}

#[derive(Clone, Debug, clap::ValueEnum)]
pub enum ListSort {
    Name,
    Size,
}

/// Arguments for the `pull` subcommand.
#[derive(Parser, Debug)]
#[command(override_usage = "vllm pull <MODEL> [OPTIONS]")]
pub struct PullArgs {
    /// HuggingFace model ID or local path (e.g. "meta-llama/Llama-3.2-1B").
    pub model: String,

    /// HuggingFace API token for gated models.
    #[arg(long, env = "HF_TOKEN")]
    pub hf_token: Option<String>,

    /// Specific GGUF filename to download (for GGUF repos).
    #[arg(long)]
    pub gguf_file: Option<String>,

    /// GGUF quantization to prefer (e.g. Q4_K_M, Q8_0). Case-insensitive.
    /// When set, auto-selects the matching GGUF file from the repo.
    #[arg(short = 'q', long)]
    pub quantization: Option<String>,
}

#[cfg(feature = "gce")]
/// GCE subcommands: `vllm gce up` / `vllm gce down`.
#[derive(Parser, Debug)]
pub struct GceCommand {
    #[command(subcommand)]
    pub command: GceSubcommand,
}

#[cfg(feature = "gce")]
#[derive(Subcommand, Debug)]
pub enum GceSubcommand {
    /// Provision a GCE VM with GPUs.
    Up(Box<GceUpArgs>),
    /// Tear down a GCE VM.
    Down(GceDownArgs),
    /// Manage GCE images for vllm-rs.
    Image(GceImageCommand),
}

#[cfg(feature = "gce")]
/// Image subcommands: `vllm gce image build` / `vllm gce image list`.
#[derive(Parser, Debug)]
pub struct GceImageCommand {
    #[command(subcommand)]
    pub command: GceImageSubcommand,
}

#[cfg(feature = "gce")]
#[derive(Subcommand, Debug)]
pub enum GceImageSubcommand {
    /// Build a GCE image with dev toolchain pre-installed.
    Build(GceImageBuildArgs),
    /// List GCE images tagged for vllm-rs.
    List(GceImageListArgs),
    /// List GCE images tagged for vllm-rs (alias for `list`).
    Ls(GceImageListArgs),
}

#[cfg(feature = "gce")]
/// Arguments for `vllm gce image build`.
#[derive(Parser, Debug)]
#[command(override_usage = "vllm gce image build <SOURCE_DIR> [OPTIONS]")]
pub struct GceImageBuildArgs {
    /// Path to Rust workspace to upload and build on the VM.
    pub source_dir: String,

    /// Build a production image: binary installed to /usr/local/bin,
    /// source and dev toolchain removed.
    #[arg(short = 'p', long)]
    pub production: bool,

    /// Base GCE boot image to build from.
    #[arg(
        long,
        default_value = "projects/ubuntu-os-accelerator-images/global/images/ubuntu-accelerator-2404-amd64-with-nvidia-580-v20260225"
    )]
    pub image: String,

    /// Image tag (used as label and name prefix).
    /// Defaults to "vllm-rs-prod" with --production, "vllm-rs-dev" otherwise.
    #[arg(short = 't', long)]
    pub tag: Option<String>,

    /// Image version string.
    /// Defaults to "prod-{YYYYMMDD-HHMMSS}" with --production,
    /// "dev-{YYYYMMDD-HHMMSS}" otherwise.
    #[arg(short = 'v', long)]
    pub version: Option<String>,

    /// GCE project.
    #[arg(long, env = "GCP_PROJECT")]
    pub project: Option<String>,

    /// Path to GCP service account credentials JSON.
    #[arg(long, env = "GOOGLE_APPLICATION_CREDENTIALS")]
    pub gcp_credentials: Option<String>,

    /// GCP service account email to assign to the builder instance.
    /// Only set this if the SA key has iam.serviceAccountUser on itself.
    #[arg(long)]
    pub gcp_service_account: Option<String>,

    /// GCS bucket for sccache shared cache.
    #[arg(long, env = "SCCACHE_GCS_BUCKET")]
    pub sccache_gcs_bucket: Option<String>,

    /// GCS key prefix for sccache shared cache (default: vllm-rs-dev-$USER).
    #[arg(long, env = "SCCACHE_GCS_KEY_PREFIX")]
    pub sccache_gcs_key_prefix: Option<String>,
}

#[cfg(feature = "gce")]
/// Arguments for `vllm gce image list`.
#[derive(Parser, Debug)]
#[command(override_usage = "vllm gce image list [OPTIONS]")]
pub struct GceImageListArgs {
    /// Image tag to filter on (e.g. "vllm-rs-dev", "vllm-rs-prod").
    /// If omitted, lists all vllm-rs images (dev and prod).
    #[arg(short = 't', long)]
    pub tag: Option<String>,

    /// GCE project.
    #[arg(long, env = "GCP_PROJECT")]
    pub project: Option<String>,

    /// Path to GCP service account credentials JSON.
    #[arg(long, env = "GOOGLE_APPLICATION_CREDENTIALS")]
    pub gcp_credentials: Option<String>,

    /// GCS bucket for sccache shared cache (accepted for compatibility, ignored).
    #[arg(long, env = "SCCACHE_GCS_BUCKET", hide = true)]
    pub sccache_gcs_bucket: Option<String>,

    /// GCS key prefix for sccache shared cache (accepted for compatibility, ignored).
    #[arg(long, env = "SCCACHE_GCS_KEY_PREFIX", hide = true)]
    pub sccache_gcs_key_prefix: Option<String>,

    /// GCP service account email (accepted for compatibility, ignored).
    #[arg(long, hide = true)]
    pub gcp_service_account: Option<String>,
}

#[cfg(feature = "gce")]
/// Arguments for `vllm gce up`.
#[derive(Parser, Debug)]
#[command(override_usage = "vllm gce up <NAME> [OPTIONS]")]
pub struct GceUpArgs {
    /// Instance name (used for both creation and teardown).
    pub name: String,

    /// Number of nodes (>1 for multi-node tensor parallelism).
    #[arg(short = 'n', long, default_value_t = 1)]
    pub nodes: u32,

    /// Number of GPUs per node.
    #[arg(short = 'c', long, default_value_t = 1)]
    pub gpu_count: u32,

    /// GPU class: "l40s", "a100-40", "a100-80", "h100".
    #[arg(short = 'k', long, default_value = "l40s")]
    pub gpu_class: String,

    /// Local directory to transfer and build on the VM (dev mode).
    #[arg(long)]
    pub dev: Option<String>,

    /// Local port for SSH tunnel to remote port 8000.
    #[arg(short = 'p', long, default_value_t = 8000)]
    pub local_port: u16,

    /// GCE boot image (full self-link). If omitted, auto-selects the latest
    /// image with tag "vllm-rs-dev" (or "vllm-rs-prod" without --dev).
    #[arg(long)]
    pub image: Option<String>,

    /// GCE zone.
    #[arg(long, default_value = "us-central1-a")]
    pub zone: String,

    /// GCE project.
    #[arg(long, env = "GCP_PROJECT")]
    pub project: Option<String>,

    /// Path to GCP service account credentials JSON.
    #[arg(long, env = "GOOGLE_APPLICATION_CREDENTIALS")]
    pub gcp_credentials: Option<String>,

    /// GCP service account email to assign to the instance.
    /// Only set this if the SA key has iam.serviceAccountUser on itself.
    #[arg(long)]
    pub gcp_service_account: Option<String>,

    /// HuggingFace token (passed to the VM for model downloads).
    #[arg(long, env = "HF_TOKEN")]
    pub hf_token: Option<String>,

    /// Use preemptible/SPOT VMs (cheaper but may be preempted).
    #[arg(short = 'P', long)]
    pub preemptible: bool,

    /// GCS bucket for sccache shared cache.
    #[arg(long, env = "SCCACHE_GCS_BUCKET")]
    pub sccache_gcs_bucket: Option<String>,

    /// GCS key prefix for sccache shared cache (default: vllm-rs-dev-$USER).
    #[arg(long, env = "SCCACHE_GCS_KEY_PREFIX")]
    pub sccache_gcs_key_prefix: Option<String>,

    /// Model to serve: HuggingFace model ID or path (required).
    #[arg(short = 'm', long)]
    pub model: String,

    /// Extra arguments passed to `vllm serve` on the remote VM (after `--`).
    #[arg(last = true)]
    pub serve_args: Vec<String>,
}

#[cfg(feature = "gce")]
/// Arguments for `vllm gce down`.
#[derive(Parser, Debug)]
pub struct GceDownArgs {
    /// Instance name to delete.
    pub name: String,

    /// GCE zone.
    #[arg(long, default_value = "us-central1-a")]
    pub zone: String,

    /// GCE project.
    #[arg(long, env = "GCP_PROJECT")]
    pub project: Option<String>,

    /// Force deletion (treat not-found as success, skip confirmation).
    #[arg(short = 'f', long)]
    pub force: bool,
}

// ---------------------------------------------------------------------------
// K8s subcommands
// ---------------------------------------------------------------------------

#[cfg(feature = "k8s")]
/// Top-level `vllm k8s` command.
#[derive(Parser, Debug)]
pub struct K8sCommand {
    #[command(subcommand)]
    pub command: K8sSubcommand,
}

#[cfg(feature = "k8s")]
#[derive(Subcommand, Debug)]
pub enum K8sSubcommand {
    /// Deploy vLLM to a Kubernetes cluster.
    Up(Box<K8sUpArgs>),
    /// Tear down a vLLM Kubernetes deployment.
    Down(K8sDownArgs),
    /// Preload models into a PVC for fast startup.
    Preload(K8sPreloadArgs),
}

#[cfg(feature = "k8s")]
/// Arguments for `vllm k8s up`.
#[derive(Parser, Debug)]
#[command(override_usage = "vllm k8s up <NAME> [OPTIONS]")]
pub struct K8sUpArgs {
    /// Resource name (used for Deployment/StatefulSet and teardown).
    pub name: String,

    /// Number of nodes (>1 for multi-node tensor parallelism).
    #[arg(short = 'n', long, default_value_t = 1)]
    pub nodes: u32,

    /// Number of GPUs per node.
    #[arg(short = 'c', long, default_value_t = 1)]
    pub gpu_count: u32,

    /// Container image.
    #[arg(long, default_value = vllm_k8s::up::DEFAULT_IMAGE)]
    pub image: String,

    /// Model to serve: HuggingFace model ID or path.
    #[arg(short = 'm', long)]
    pub model: String,

    /// HuggingFace token.
    #[arg(long, env = "HF_TOKEN")]
    pub hf_token: Option<String>,

    /// Local port for port forwarding.
    #[arg(short = 'p', long, default_value_t = 8000)]
    pub local_port: u16,

    /// Kubernetes namespace.
    #[arg(long)]
    pub namespace: Option<String>,

    /// Shared memory size for /dev/shm (NCCL).
    #[arg(long, default_value = "16Gi")]
    pub shm_size: String,

    /// PVC with preloaded models (mounted read-only at ~/.cache/huggingface).
    #[arg(short = 'P', long)]
    pub preload: Option<String>,

    /// Extra arguments passed to `vllm serve` on the pod (after `--`).
    #[arg(last = true)]
    pub serve_args: Vec<String>,
}

#[cfg(feature = "k8s")]
/// Arguments for `vllm k8s down`.
#[derive(Parser, Debug)]
pub struct K8sDownArgs {
    /// Resource name to delete.
    pub name: String,

    /// Kubernetes namespace.
    #[arg(long)]
    pub namespace: Option<String>,

    /// Force deletion (treat not-found as success).
    #[arg(short = 'f', long)]
    pub force: bool,
}

#[cfg(feature = "k8s")]
/// Arguments for `vllm k8s preload`.
#[derive(Parser, Debug)]
#[command(override_usage = "vllm k8s preload <PVC_NAME> <MODEL>...")]
pub struct K8sPreloadArgs {
    /// PVC name to create/reuse for model storage.
    pub pvc_name: String,

    /// Models to download (HuggingFace model IDs).
    #[arg(required = true)]
    pub models: Vec<String>,

    /// Kubernetes namespace.
    #[arg(long)]
    pub namespace: Option<String>,

    /// PVC storage size.
    #[arg(long, default_value = "100Gi")]
    pub size: String,

    /// Storage class (uses cluster default if omitted).
    #[arg(long)]
    pub storage_class: Option<String>,

    /// Container image for the download Job.
    #[arg(long, default_value = vllm_k8s::up::DEFAULT_IMAGE)]
    pub image: String,

    /// HuggingFace token.
    #[arg(long, env = "HF_TOKEN")]
    pub hf_token: Option<String>,
}

/// Arguments for the `top` subcommand.
#[cfg(feature = "top")]
#[derive(Parser, Debug)]
#[command(override_usage = "vllm top [OPTIONS]")]
pub struct TopArgs {
    /// Server host to connect to.
    #[arg(long, default_value = "localhost")]
    pub host: String,

    /// Server port to connect to.
    #[arg(long, default_value_t = 8000)]
    pub port: u16,

    /// Poll interval in milliseconds.
    #[arg(long, default_value_t = 1000)]
    pub interval: u64,
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Parse a JSON string into a `HashMap<String, serde_json::Value>`.
/// Used as a clap `value_parser` for `--default-chat-template-kwargs`.
fn parse_json_map(s: &str) -> Result<std::collections::HashMap<String, serde_json::Value>, String> {
    serde_json::from_str(s).map_err(|e| format!("invalid JSON: {e}"))
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
    #[cfg(feature = "bench")]
    fn test_parse_bench_latency_positional_model() {
        let cli = Cli::parse_from([
            "vllm",
            "bench",
            "latency",
            "/path/to/model",
            "--num-iters",
            "5",
        ]);
        match cli.command {
            Commands::Bench(cmd) => match cmd.command {
                BenchCommands::Latency(args) => {
                    assert_eq!(
                        args.resolved_models().unwrap(),
                        vec!["/path/to/model".to_string()]
                    );
                    assert_eq!(args.num_iters, 5);
                }
                _ => panic!("expected Latency subcommand"),
            },
            _ => panic!("expected Bench command"),
        }
    }

    #[test]
    #[cfg(feature = "bench")]
    fn test_parse_bench_latency_flag_model() {
        let cli = Cli::parse_from(["vllm", "bench", "latency", "--model", "/path/to/model"]);
        match cli.command {
            Commands::Bench(cmd) => match cmd.command {
                BenchCommands::Latency(args) => {
                    assert_eq!(
                        args.resolved_models().unwrap(),
                        vec!["/path/to/model".to_string()]
                    );
                }
                _ => panic!("expected Latency subcommand"),
            },
            _ => panic!("expected Bench command"),
        }
    }

    #[test]
    #[cfg(feature = "bench")]
    fn test_parse_bench_latency_defaults() {
        let cli = Cli::parse_from(["vllm", "bench", "latency", "some-model"]);
        match cli.command {
            Commands::Bench(cmd) => match cmd.command {
                BenchCommands::Latency(args) => {
                    assert_eq!(args.num_iters, 30);
                    assert_eq!(args.input_len, 32);
                    assert_eq!(args.output_len, 128);
                    assert_eq!(args.num_iters_warmup, 10);
                    assert_eq!(args.batch_sizes, vec![8]);
                    assert!(args.output_json.is_none());
                }
                _ => panic!("expected Latency subcommand"),
            },
            _ => panic!("expected Bench command"),
        }
    }

    #[test]
    #[cfg(feature = "bench")]
    fn test_bench_requires_subcommand() {
        // `vllm bench` alone (no subcommand) should fail to parse.
        let result = Cli::try_parse_from(["vllm", "bench"]);
        assert!(result.is_err());
    }

    #[test]
    fn test_parse_batch_args() {
        let cli = Cli::parse_from([
            "vllm",
            "batch",
            "meta-llama/Llama-3.2-1B",
            "-i",
            "input.jsonl",
            "-o",
            "output.jsonl",
        ]);
        match cli.command {
            Commands::Batch(args) => {
                assert_eq!(args.resolved_model().unwrap(), "meta-llama/Llama-3.2-1B");
                assert_eq!(args.input, "input.jsonl");
                assert_eq!(args.output, "output.jsonl");
                assert_eq!(args.device, "auto");
            }
            _ => panic!("expected Batch command"),
        }
    }

    #[test]
    fn test_batch_resolved_model() {
        // Flag model.
        let cli = Cli::parse_from([
            "vllm",
            "batch",
            "--model",
            "flag-model",
            "-i",
            "in.jsonl",
            "-o",
            "out.jsonl",
        ]);
        match cli.command {
            Commands::Batch(args) => {
                assert_eq!(args.resolved_model().unwrap(), "flag-model");
            }
            _ => panic!("expected Batch command"),
        }

        // Positional takes precedence over --model.
        let cli = Cli::parse_from([
            "vllm",
            "batch",
            "positional-model",
            "--model",
            "flag-model",
            "-i",
            "in.jsonl",
            "-o",
            "out.jsonl",
        ]);
        match cli.command {
            Commands::Batch(args) => {
                assert_eq!(args.resolved_model().unwrap(), "positional-model");
            }
            _ => panic!("expected Batch command"),
        }

        // No model → error.
        let cli = Cli::parse_from(["vllm", "batch", "-i", "in.jsonl", "-o", "out.jsonl"]);
        match cli.command {
            Commands::Batch(args) => {
                assert!(args.resolved_model().is_err());
            }
            _ => panic!("expected Batch command"),
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

    // -- Runner flag tests --

    #[test]
    fn test_serve_runner_default_is_generate() {
        let cli = Cli::parse_from(["vllm", "serve", "some-model"]);
        match cli.command {
            Commands::Serve(args) => {
                assert_eq!(args.runner, "generate");
            }
            _ => panic!("expected Serve command"),
        }
    }

    #[test]
    fn test_serve_runner_pooling() {
        let cli = Cli::parse_from(["vllm", "serve", "some-model", "--runner", "pooling"]);
        match cli.command {
            Commands::Serve(args) => {
                assert_eq!(args.runner, "pooling");
            }
            _ => panic!("expected Serve command"),
        }
    }

    // -- Chat command tests --

    #[test]
    fn test_chat_defaults_remote_mode() {
        let cli = Cli::parse_from(["vllm", "chat"]);
        match cli.command {
            Commands::Chat(args) => {
                assert!(args.resolved_model().is_none());
                assert_eq!(args.url, "http://localhost:8000/v1");
                assert!(args.system_prompt.is_none());
                assert!(args.quick.is_none());
                assert!(args.prompt.is_empty());
                assert!(!args.bench);
                assert_eq!(args.device, "auto");
                assert_eq!(args.dtype, "auto");
                assert_eq!(args.max_tokens, None);
            }
            _ => panic!("expected Chat command"),
        }
    }

    #[test]
    fn test_chat_inproc_positional_model() {
        let cli = Cli::parse_from(["vllm", "chat", "Qwen/Qwen2.5-0.5B"]);
        match cli.command {
            Commands::Chat(args) => {
                assert_eq!(args.resolved_model().unwrap(), "Qwen/Qwen2.5-0.5B");
            }
            _ => panic!("expected Chat command"),
        }
    }

    #[test]
    fn test_chat_inproc_flag_model() {
        let cli = Cli::parse_from(["vllm", "chat", "--model", "my-model", "--device", "cpu"]);
        match cli.command {
            Commands::Chat(args) => {
                assert_eq!(args.resolved_model().unwrap(), "my-model");
                assert_eq!(args.device, "cpu");
            }
            _ => panic!("expected Chat command"),
        }
    }

    #[test]
    fn test_chat_positional_takes_precedence() {
        let cli = Cli::parse_from(["vllm", "chat", "positional", "--model", "flag"]);
        match cli.command {
            Commands::Chat(args) => {
                assert_eq!(args.resolved_model().unwrap(), "positional");
            }
            _ => panic!("expected Chat command"),
        }
    }

    #[test]
    fn test_chat_quick_short_flag() {
        let cli = Cli::parse_from(["vllm", "chat", "-q", "hello"]);
        match cli.command {
            Commands::Chat(args) => {
                assert_eq!(args.quick.as_deref(), Some("hello"));
            }
            _ => panic!("expected Chat command"),
        }
    }

    #[test]
    fn test_chat_prompt_and_bench() {
        let cli = Cli::parse_from([
            "vllm",
            "chat",
            "my-model",
            "--prompt",
            "Tell me a joke",
            "--bench",
        ]);
        match cli.command {
            Commands::Chat(args) => {
                assert_eq!(args.prompt, vec!["Tell me a joke"]);
                assert!(args.bench);
                assert_eq!(args.resolved_model().unwrap(), "my-model");
            }
            _ => panic!("expected Chat command"),
        }
    }

    #[test]
    fn test_chat_system_prompt() {
        let cli = Cli::parse_from([
            "vllm",
            "chat",
            "--system-prompt",
            "You are a pirate.",
            "--url",
            "http://example.com/v1",
        ]);
        match cli.command {
            Commands::Chat(args) => {
                assert_eq!(args.system_prompt.as_deref(), Some("You are a pirate."));
                assert_eq!(args.url, "http://example.com/v1");
            }
            _ => panic!("expected Chat command"),
        }
    }

    #[test]
    fn test_chat_all_inproc_options() {
        let cli = Cli::parse_from([
            "vllm",
            "chat",
            "my-model",
            "--device",
            "metal",
            "--dtype",
            "float16",
            "--max-model-len",
            "4096",
            "--gguf-file",
            "model.gguf",
            "--max-tokens",
            "256",
        ]);
        match cli.command {
            Commands::Chat(args) => {
                assert_eq!(args.device, "metal");
                assert_eq!(args.dtype, "float16");
                assert_eq!(args.max_model_len, Some(4096));
                assert_eq!(args.gguf_file.as_deref(), Some("model.gguf"));
                assert_eq!(args.max_tokens, Some(256));
            }
            _ => panic!("expected Chat command"),
        }
    }

    // -- Complete command tests --

    #[test]
    fn test_complete_defaults() {
        let cli = Cli::parse_from(["vllm", "complete"]);
        match cli.command {
            Commands::Complete(args) => {
                assert_eq!(args.url, "http://localhost:8000/v1");
                assert!(args.model_name.is_none());
                assert!(args.max_tokens.is_none());
                assert!(args.quick.is_none());
            }
            _ => panic!("expected Complete command"),
        }
    }

    #[test]
    fn test_complete_all_options() {
        let cli = Cli::parse_from([
            "vllm",
            "complete",
            "--url",
            "http://host:9000/v1",
            "--model-name",
            "gpt-4",
            "--max-tokens",
            "256",
            "-q",
            "Once upon a time",
        ]);
        match cli.command {
            Commands::Complete(args) => {
                assert_eq!(args.url, "http://host:9000/v1");
                assert_eq!(args.model_name.as_deref(), Some("gpt-4"));
                assert_eq!(args.max_tokens, Some(256));
                assert_eq!(args.quick.as_deref(), Some("Once upon a time"));
            }
            _ => panic!("expected Complete command"),
        }
    }

    // -- run-batch alias tests --

    #[test]
    fn test_run_batch_alias() {
        let cli = Cli::parse_from([
            "vllm",
            "run-batch",
            "my-model",
            "-i",
            "in.jsonl",
            "-o",
            "out.jsonl",
        ]);
        match cli.command {
            Commands::RunBatch(args) => {
                assert_eq!(args.resolved_model().unwrap(), "my-model");
                assert_eq!(args.input, "in.jsonl");
                assert_eq!(args.output, "out.jsonl");
            }
            _ => panic!("expected RunBatch command"),
        }
    }

    #[test]
    fn test_run_batch_same_args_as_batch() {
        // Verify run-batch accepts all the same flags as batch.
        let cli = Cli::parse_from([
            "vllm",
            "run-batch",
            "--model",
            "flag-model",
            "-i",
            "in.jsonl",
            "-o",
            "out.jsonl",
            "--device",
            "cuda:0",
            "--dtype",
            "bfloat16",
        ]);
        match cli.command {
            Commands::RunBatch(args) => {
                assert_eq!(args.resolved_model().unwrap(), "flag-model");
                assert_eq!(args.device, "cuda:0");
                assert_eq!(args.dtype, "bfloat16");
            }
            _ => panic!("expected RunBatch command"),
        }
    }

    // -- collect-env tests --

    #[test]
    fn test_collect_env_no_args() {
        let cli = Cli::parse_from(["vllm", "collect-env"]);
        assert!(matches!(cli.command, Commands::CollectEnv(_)));
    }

    // -- distributed-executor-backend tests --

    #[test]
    fn test_serve_distributed_backend_default() {
        let cli = Cli::parse_from(["vllm", "serve", "some-model"]);
        match cli.command {
            Commands::Serve(args) => {
                assert_eq!(args.distributed_executor_backend, "auto");
            }
            _ => panic!("expected Serve command"),
        }
    }

    #[test]
    fn test_serve_distributed_backend_external_launcher() {
        let cli = Cli::parse_from([
            "vllm",
            "serve",
            "some-model",
            "--distributed-executor-backend",
            "external_launcher",
        ]);
        match cli.command {
            Commands::Serve(args) => {
                assert_eq!(args.distributed_executor_backend, "external_launcher");
            }
            _ => panic!("expected Serve command"),
        }
    }

    #[test]
    fn test_serve_distributed_backend_with_tp() {
        let cli = Cli::parse_from([
            "vllm",
            "serve",
            "some-model",
            "--distributed-executor-backend",
            "external_launcher",
            "--tensor-parallel-size",
            "4",
        ]);
        match cli.command {
            Commands::Serve(args) => {
                assert_eq!(args.distributed_executor_backend, "external_launcher");
                assert_eq!(args.tensor_parallel_size, 4);
            }
            _ => panic!("expected Serve command"),
        }
    }
}
