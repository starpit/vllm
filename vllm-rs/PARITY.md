# vLLM Feature Parity Punchlist: Python vs Rust

> Generated 2026-02-28 | Rust port: `vllm-rs/` on branch `feat/rust` (613+38 tests, 0 clippy errors)

### Legend

| Symbol | Meaning |
|--------|---------|
| &#x1F535; | Fully implemented |
| &#x1F7E1; | Partially implemented |
| &#x2795; | Planned (in [`PORT_PLAN.md`](PORT_PLAN.md)) |
| &#x1F534; | Not implemented / not planned |

---

## Summary: Major Feature Groups

| Feature Group | Python | Rust | Parity |
|---|:---:|:---:|---|
| [Model Architectures](#model-architectures) | &#x1F535; | &#x1F7E1; | `███░░░░░░░` 9/36 |
| [Quantization](#quantization) | &#x1F535; | &#x1F7E1; | `█░░░░░░░░░` 1/11 |
| [Serving / OpenAI API](#serving--openai-api) | &#x1F535; | &#x1F7E1; | `█████░░░░░` 13/25 |
| [Sampling & Decoding](#sampling--decoding) | &#x1F535; | &#x1F7E1; | `████████░░` 17/21 |
| [KV Cache & Attention](#kv-cache--attention) | &#x1F535; | &#x1F7E1; | `█████░░░░░` 9/19 |
| [Scheduling](#scheduling) | &#x1F535; | &#x1F7E1; | `████████░░` 8/10 |
| [Hardware Backends](#hardware-backends) | &#x1F535; | &#x1F7E1; | `████░░░░░░` 3/8 |
| [Parallelism & Distribution](#parallelism--distribution) | &#x1F535; | &#x1F7E1; | `███░░░░░░░` 3/10 |
| [Performance Optimizations](#performance-optimizations) | &#x1F535; | &#x1F7E1; | `███░░░░░░░` 3/12 |
| [LoRA / Adapters](#lora--adapters) | &#x1F535; | &#x1F534; | `░░░░░░░░░░` 0/5 |
| [Speculative Decoding](#speculative-decoding) | &#x1F535; | &#x1F534; | `░░░░░░░░░░` 0/5 |
| [Multimodal / Vision-Language](#multimodal--vision-language) | &#x1F535; | &#x2795; | `░░░░░░░░░░` 0/10 |
| [Structured Output](#structured-output--guided-decoding) | &#x1F535; | &#x2795; | `░░░░░░░░░░` 0/4 |
| [Tool Calling](#tool-calling--function-calling) | &#x1F535; | &#x1F7E1; | `█████░░░░░` 5/6 |
| [Embeddings & Pooling](#embeddings--pooling) | &#x1F535; | &#x1F534; | `░░░░░░░░░░` 0/4 |
| [Observability & Operations](#observability--operations) | &#x1F535; | &#x1F7E1; | `█████████░` 6/7 |
| [CLI & Deployment](#cli--deployment) | &#x1F535; | &#x1F7E1; | `██████████` 14/15 |
| | | **Total** | `████░░░░░░` **90/198** |

---

## Model Architectures

### Candle Backend (CPU / CUDA)

| Architecture | Python | Rust |
|---|:---:|:---:|
| LLaMA / LLaMA 2 / LLaMA 3 | &#x1F535; | &#x1F535; |
| Mistral | &#x1F535; | &#x1F535; |
| Qwen2 | &#x1F535; | &#x1F535; |
| Qwen3 | &#x1F535; | &#x1F535; |
| Phi-3 | &#x1F535; | &#x1F535; |
| Gemma 2 | &#x1F535; | &#x1F535; |
| DeepSeek V2 / V3 (MLA + MoE) | &#x1F535; | &#x1F535; |
| Command R (Cohere) | &#x1F535; | &#x1F535; |
| Quantized LLaMA (GGUF) | &#x1F535; | &#x1F535; |
| Gemma 1 | &#x1F535; | &#x1F534; |
| Qwen 1 | &#x1F535; | &#x1F534; |
| Qwen2 MoE | &#x1F535; | &#x1F534; |
| Qwen3 MoE | &#x1F535; | &#x1F534; |
| Mixtral (MoE) | &#x1F535; | &#x2795; |
| GPT-NeoX | &#x1F535; | &#x2795; |
| GPT-J | &#x1F535; | &#x2795; |
| Falcon | &#x1F535; | &#x2795; |
| BLOOM | &#x1F535; | &#x2795; |
| MPT | &#x1F535; | &#x2795; |
| StarCoder / StarCoder2 | &#x1F535; | &#x2795; |
| OPT | &#x1F535; | &#x1F534; |
| Phi-1 / Phi-2 | &#x1F535; | &#x1F534; |
| Phi-4 | &#x1F535; | &#x1F534; |
| Gemma 3 / 3n | &#x1F535; | &#x1F534; |
| ChatGLM / GLM-4 | &#x1F535; | &#x1F534; |
| Baichuan | &#x1F535; | &#x1F534; |
| DBRX | &#x1F535; | &#x1F534; |
| Mamba / Bamba / Jamba | &#x1F535; | &#x1F534; |
| OLMo / OLMoE | &#x1F535; | &#x1F534; |
| Nemotron | &#x1F535; | &#x1F534; |
| Exaone | &#x1F535; | &#x1F534; |
| StableLM | &#x1F535; | &#x1F534; |
| Solar | &#x1F535; | &#x1F534; |
| Arctic (MoE) | &#x1F535; | &#x1F534; |
| PLaMo | &#x1F535; | &#x1F534; |
| Zamba 2 | &#x1F535; | &#x1F534; |
| ~200+ other architectures | &#x1F535; | &#x1F534; |

### MLX Backend (Apple Silicon)

| Architecture | Python | Rust |
|---|:---:|:---:|
| LLaMA / LLaMA 2 / LLaMA 3 | N/A | &#x1F535; |
| Mistral | N/A | &#x1F535; |
| Qwen2 (with bias) | N/A | &#x1F535; |
| Qwen3 (with QK norms) | N/A | &#x1F535; |
| Gemma 1 | N/A | &#x1F535; |
| Gemma 2 | N/A | &#x1F535; |
| Phi-3 (fused projections) | N/A | &#x1F535; |
| DeepSeek V2 / V3 | N/A | &#x1F535; |
| Command R (Cohere) | N/A | &#x1F535; |
| Quantized LLaMA (mlx-community 4-bit) | N/A | &#x1F535; |
| Quantized Mistral (4-bit) | N/A | &#x1F535; |
| Quantized Qwen2/3 (4-bit) | N/A | &#x1F535; |
| Quantized Gemma 1 (4-bit) | N/A | &#x1F535; |
| Quantized Gemma 2 (4-bit) | N/A | &#x1F535; |
| Quantized Phi-3 (4-bit) | N/A | &#x1F535; |
| Quantized Command R (4-bit) | N/A | &#x1F535; |

> Python vLLM does not have an MLX backend. The Rust MLX backend is unique to the Rust port.

---

## Quantization

| Method | Python | Rust |
|---|:---:|:---:|
| GGUF (Q4_0 / Q4_K / Q8_0 / etc.) | &#x1F535; | &#x1F535; |
| MLX native 4-bit group quantization | N/A | &#x1F535; |
| GPTQ | &#x1F535; | &#x1F534; |
| AWQ | &#x1F535; | &#x1F534; |
| Marlin (GPTQ-Marlin / AWQ-Marlin) | &#x1F535; | &#x1F534; |
| FP8 (FBGemm / ModelOpt) | &#x1F535; | &#x1F534; |
| BitsAndBytes (4-bit / 8-bit) | &#x1F535; | &#x1F534; |
| SqueezeLLM | &#x1F535; | &#x1F534; |
| Compressed Tensors | &#x1F535; | &#x1F534; |
| TorchAO | &#x1F535; | &#x1F534; |
| MXFP4 | &#x1F535; | &#x1F534; |
| CPU WNA16 | &#x1F535; | &#x1F534; |
| Experts INT8 (MoE) | &#x1F535; | &#x1F534; |

---

## Serving / OpenAI API

| Feature | Python | Rust |
|---|:---:|:---:|
| `POST /v1/chat/completions` | &#x1F535; | &#x1F535; |
| `POST /v1/completions` | &#x1F535; | &#x1F535; |
| SSE streaming (chat) | &#x1F535; | &#x1F535; |
| SSE streaming (completions) | &#x1F535; | &#x1F535; |
| `GET /v1/models` | &#x1F535; | &#x1F535; |
| `GET /health` | &#x1F535; | &#x1F535; |
| `GET /version` | &#x1F535; | &#x1F535; |
| `n` parameter (multiple completions) | &#x1F535; | &#x1F535; |
| Multi-prompt completions | &#x1F535; | &#x1F535; |
| Chat templates (Jinja2) | &#x1F535; | &#x1F535; |
| `POST /v1/embeddings` | &#x1F535; | &#x1F534; |
| `POST /v1/chat/completions` tool_calls | &#x1F535; | &#x1F535; |
| `response_format` (JSON mode/schema) | &#x1F535; | &#x2795; |
| Anthropic Messages API | &#x1F535; | &#x1F534; |
| gRPC server | &#x1F535; | &#x2795; |
| MCP tool server | &#x1F535; | &#x1F534; |
| Batch processing | &#x1F535; | &#x1F534; |
| Responses API | &#x1F535; | &#x1F534; |
| Speech-to-text | &#x1F535; | &#x1F534; |
| Realtime API | &#x1F535; | &#x1F534; |
| SageMaker integration | &#x1F535; | &#x1F534; |
| SSL / TLS | &#x1F535; | &#x1F534; |
| CORS | &#x1F535; | &#x1F535; |
| `usage` field in responses | &#x1F535; | &#x1F535; |
| `best_of` / `n` with reranking | &#x1F535; | &#x1F534; |

---

## Sampling & Decoding

| Feature | Python | Rust |
|---|:---:|:---:|
| Greedy (argmax) | &#x1F535; | &#x1F535; |
| Temperature | &#x1F535; | &#x1F535; |
| Top-k | &#x1F535; | &#x1F535; |
| Top-p (nucleus) | &#x1F535; | &#x1F535; |
| Min-p | &#x1F535; | &#x1F535; |
| Repetition penalty | &#x1F535; | &#x1F535; |
| Frequency penalty | &#x1F535; | &#x1F535; |
| Presence penalty | &#x1F535; | &#x1F535; |
| Logprobs | &#x1F535; | &#x1F535; |
| Prompt logprobs | &#x1F535; | &#x1F534; |
| Logit bias | &#x1F535; | &#x1F535; |
| Beam search | &#x1F535; | &#x1F534; |
| `best_of` | &#x1F535; | &#x1F534; |
| `max_tokens` / `max_completion_tokens` | &#x1F535; | &#x1F535; |
| Stop strings | &#x1F535; | &#x1F535; |
| Stop token IDs | &#x1F535; | &#x1F535; |
| EOS detection (multi-EOS) | &#x1F535; | &#x1F535; |
| `ignore_eos` | &#x1F535; | &#x1F535; |
| `min_tokens` | &#x1F535; | &#x1F535; |
| Seed (reproducible sampling) | &#x1F535; | &#x1F535; |
| Guided decoding (grammar/regex/JSON) | &#x1F535; | &#x2795; |

> All penalty/filter/logprobs features use a unified `Sampler::sample_one()` entry point that operates on CPU logit vectors in both CandleWorker and MlxWorker. Prompt logprobs are not yet implemented (requires running logprobs on every prefill position).

---

## KV Cache & Attention

| Feature | Python | Rust |
|---|:---:|:---:|
| Per-request KV cache | &#x1F535; | &#x1F535; |
| Block-based KV cache (PagedAttention) | &#x1F535; | &#x1F535; |
| KV block pool (pre-allocated) | &#x1F535; | &#x1F535; |
| Direct block KV reads (no gather copy) | &#x1F535; | &#x1F535; |
| Paged decode attention (per-block scoring) | &#x1F535; | &#x1F535; |
| Prefix caching (hash-based) | &#x1F535; | &#x1F535; |
| Automatic prefix caching | &#x1F535; | &#x1F535; |
| Chunked prefill | &#x1F535; | &#x1F535; |
| KV cache compression (latent caching) | &#x1F535; | &#x1F534; |
| KV cache offloading (CPU ↔ GPU) | &#x1F535; | &#x1F534; |
| KV cache transfer (distributed) | &#x1F535; | &#x1F534; |
| Multi-group KV cache (hybrid models) | &#x1F535; | &#x1F534; |
| FlashAttention v2 | &#x1F535; | &#x1F534; |
| FlashInfer | &#x1F535; | &#x1F534; |
| FlexAttention | &#x1F535; | &#x1F534; |
| xFormers | &#x1F535; | &#x1F534; |
| MLA (Multi-head Latent Attention) | &#x1F535; | &#x1F535; |
| Sliding window attention | &#x1F535; | &#x1F534; |
| Tree attention | &#x1F535; | &#x1F534; |

---

## Scheduling

| Feature | Python | Rust |
|---|:---:|:---:|
| Continuous batching | &#x1F535; | &#x1F535; |
| FCFS request queue | &#x1F535; | &#x1F535; |
| Priority request queue | &#x1F535; | &#x1F535; |
| Preemption | &#x1F535; | &#x1F535; |
| Chunked prefill scheduling | &#x1F535; | &#x1F535; |
| Block allocation / eviction | &#x1F535; | &#x1F535; |
| Prefix cache hits | &#x1F535; | &#x1F535; |
| Pause / resume | &#x1F535; | &#x1F535; |
| Multi-step scheduling | &#x1F535; | &#x1F534; |
| Async scheduler | &#x1F535; | &#x1F534; |

---

## Hardware Backends

| Backend | Python | Rust |
|---|:---:|:---:|
| CPU | &#x1F535; | &#x1F535; |
| CUDA (NVIDIA GPU) | &#x1F535; | &#x1F7E1; |
| Metal / MLX (Apple Silicon) | &#x1F534; | &#x1F535; |
| ROCm (AMD GPU) | &#x1F535; | &#x1F534; |
| TPU | &#x1F535; | &#x1F534; |
| XPU (Intel) | &#x1F535; | &#x1F534; |
| AWS Neuron | &#x1F535; | &#x1F534; |
| Device auto-detection | &#x1F535; | &#x1F535; |
| Memory profiling / `--gpu-memory-utilization` | &#x1F535; | &#x1F535; |

> Rust CUDA support uses candle-core's CUDA backend. Custom CUDA kernels (PagedAttention v1/v2, fused ops) are not yet ported.

---

## Parallelism & Distribution

| Feature | Python | Rust |
|---|:---:|:---:|
| Parallel config types (TP/PP groups) | &#x1F535; | &#x1F535; |
| UniProc executor (single-process) | &#x1F535; | &#x1F535; |
| MultiProc executor (multi-worker) | &#x1F535; | &#x1F535; |
| Tensor parallelism (actual sharding) | &#x1F535; | &#x1F7E1; |
| Pipeline parallelism | &#x1F535; | &#x1F534; |
| NCCL communication | &#x1F535; | &#x2795; |
| Ray distributed executor | &#x1F535; | &#x1F534; |
| Expert parallelism (MoE) | &#x1F535; | &#x1F534; |
| Data parallelism | &#x1F535; | &#x1F534; |
| Weight transfer / migration | &#x1F535; | &#x1F534; |

> Rust has ColumnParallelLinear / RowParallelLinear layer types and ResolvedParallelConfig, but actual multi-GPU sharded execution is not wired end-to-end.

---

## Performance Optimizations

| Feature | Python | Rust |
|---|:---:|:---:|
| CUDA graphs | &#x1F535; | &#x1F534; |
| FlashAttention v2 kernels | &#x1F535; | &#x1F534; |
| FlashInfer kernels | &#x1F535; | &#x1F534; |
| xFormers memory-efficient attention | &#x1F535; | &#x1F534; |
| Fused SiLU-and-mul kernel | &#x1F535; | &#x1F534; |
| Fused RMSNorm kernel | &#x1F535; | &#x1F534; |
| Fused RoPE kernel | &#x1F535; | &#x1F534; |
| Custom all-reduce kernel | &#x1F535; | &#x1F534; |
| MoE fused routing kernels | &#x1F535; | &#x1F534; |
| MLX lazy eval graph fusion | N/A | &#x1F535; |
| MLX single-eval sampling fusion | N/A | &#x1F535; |
| Pre-transposed weights (Metal) | N/A | &#x1F535; |
| Native dtype inference (`--dtype auto`) | &#x1F535; | &#x1F535; |
| Continuous batching | &#x1F535; | &#x1F535; |
| Paged KV (no gather copy on decode) | &#x1F535; | &#x1F535; |

---

## LoRA / Adapters

| Feature | Python | Rust |
|---|:---:|:---:|
| LoRA adapter loading | &#x1F535; | &#x1F534; |
| Multi-LoRA serving | &#x1F535; | &#x1F534; |
| LoRA weight merging | &#x1F535; | &#x1F534; |
| Punica kernels | &#x1F535; | &#x1F534; |
| Dynamic adapter switching | &#x1F535; | &#x1F534; |

---

## Speculative Decoding

| Feature | Python | Rust |
|---|:---:|:---:|
| Draft model (MLP speculator) | &#x1F535; | &#x1F534; |
| Eagle speculative decoding | &#x1F535; | &#x1F534; |
| Medusa heads | &#x1F535; | &#x1F534; |
| N-gram proposer | &#x1F535; | &#x1F534; |
| Suffix decoding | &#x1F535; | &#x1F534; |

---

## Multimodal / Vision-Language

| Feature | Python | Rust |
|---|:---:|:---:|
| Image input processing | &#x1F535; | &#x2795; |
| LLaVA | &#x1F535; | &#x2795; |
| Qwen-VL / Qwen2.5-VL | &#x1F535; | &#x2795; |
| Pixtral | &#x1F535; | &#x1F534; |
| InternVL | &#x1F535; | &#x1F534; |
| Phi-3V / Phi-4MM | &#x1F535; | &#x1F534; |
| Gemma 3 multimodal | &#x1F535; | &#x1F534; |
| Molmo | &#x1F535; | &#x1F534; |
| PaliGemma | &#x1F535; | &#x1F534; |
| Audio models (Whisper, Qwen-Audio) | &#x1F535; | &#x1F534; |

---

## Structured Output / Guided Decoding

| Feature | Python | Rust |
|---|:---:|:---:|
| `response_format: json_object` | &#x1F535; | &#x2795; |
| `response_format: json_schema` | &#x1F535; | &#x2795; |
| Grammar-guided logit masking | &#x1F535; | &#x2795; |
| Regex-constrained decoding | &#x1F535; | &#x2795; |

> Protocol types for `response_format` exist in Rust but enforcement is not implemented.

---

## Tool Calling / Function Calling

| Feature | Python | Rust |
|---|:---:|:---:|
| `tools` / `tool_choice` request fields | &#x1F535; | &#x1F535; |
| Chat template tool definitions | &#x1F535; | &#x1F535; |
| Model-emitted tool call parsing | &#x1F535; | &#x1F535; |
| `tool_calls` in response | &#x1F535; | &#x1F535; |
| Streaming tool call deltas | &#x1F535; | &#x1F535; |
| Parallel tool calls | &#x1F535; | &#x2795; |

> Tool calling is fully functional end-to-end (Phases 12a+12b). Chat templates pass tool definitions to models; HermesToolParser (`<tool_call>` tags) and LlamaJsonToolParser (raw JSON / `<|python_tag|>`) extract structured `ToolCall` objects from model output. Streaming tool call deltas supported. Use `--tool-call-parser hermes|llama3_json`. Parallel tool calls (multiple tool calls in one response) work; forced single-tool choice (`tool_choice: {function: {name}}`) validation is not yet implemented.

---

## Embeddings & Pooling

| Feature | Python | Rust |
|---|:---:|:---:|
| `/v1/embeddings` endpoint | &#x1F535; | &#x1F534; |
| Embedding model architectures (BERT, etc.) | &#x1F535; | &#x1F534; |
| Pooling strategies (CLS, mean, last) | &#x1F535; | &#x1F534; |
| Reward / reranking models | &#x1F535; | &#x1F534; |

---

## Observability & Operations

| Feature | Python | Rust |
|---|:---:|:---:|
| Prometheus `/metrics` endpoint | &#x1F535; | &#x1F535; |
| Request counters (total, active) | &#x1F535; | &#x1F535; |
| Token counters (prompt, generation) | &#x1F535; | &#x1F535; |
| Latency histograms | &#x1F535; | &#x1F535; |
| Structured logging (tracing) | &#x1F535; | &#x1F535; |
| OpenTelemetry tracing | &#x1F535; | &#x1F534; |
| ORCA metrics | &#x1F535; | &#x1F534; |
| MLX per-step timing (prefill/decode ms) | N/A | &#x1F535; |
| Benchmark CLI (`vllm bench`) | &#x1F535; | &#x1F535; |

---

## CLI & Deployment

| Feature | Python | Rust |
|---|:---:|:---:|
| `vllm serve <model>` | &#x1F535; | &#x1F535; |
| `vllm bench <model>` | &#x1F535; | &#x1F535; |
| `vllm convert` | &#x1F535; | &#x1F535; |
| HF Hub model download | &#x1F535; | &#x1F535; |
| Sharded weight loading | &#x1F535; | &#x1F535; |
| `--model` / positional model arg | &#x1F535; | &#x1F535; |
| `--dtype auto` / explicit dtype | &#x1F535; | &#x1F535; |
| `--device auto` / explicit device | &#x1F535; | &#x1F535; |
| `--gpu-memory-utilization` | &#x1F535; | &#x1F535; |
| `--gguf-file` | &#x1F535; | &#x1F535; |
| `--tool-call-parser` | &#x1F535; | &#x1F535; |
| `--features metal` (MLX backend) | N/A | &#x1F535; |
| Dockerfile.cpu | &#x1F535; | &#x1F535; |
| Dockerfile.cuda | &#x1F535; | &#x1F535; |
| `VLLM_MODEL` env var | &#x1F535; | &#x1F535; |
| PyO3 bridge (Rust scheduler in Python) | &#x1F535; | &#x1F7E1; |
| Python fallback for unported models | &#x1F535; | &#x2795; |
| Standalone binary (no Python runtime) | &#x1F534; | &#x1F535; |

---

## Stats

| Metric | Python | Rust |
|---|---|---|
| Model architectures | ~248 | 9 candle + 9 MLX (+ quantized variants) |
| Quantization methods | ~14 | 2 (GGUF + MLX native 4-bit) |
| Attention backends | ~15 | 1 (custom SDPA) |
| Hardware backends | 6 (CUDA, ROCm, CPU, TPU, XPU, Neuron) | 3 (CPU, CUDA, Metal/MLX) |
| Lines of code | ~507K Python + ~89K C++/CUDA | ~30.7K Rust |
| Test count | ~948 test files | 651 passing tests (613+38 MLX) |
| Crate count | N/A | 13 crates |
