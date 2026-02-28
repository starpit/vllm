# vLLM Feature Parity Punchlist: Python vs Rust

> Generated 2026-02-28 | Rust port: `vllm-rs/` on branch `feat/rust` (754 tests: 701 unit + 53 e2e, 0 clippy errors)

### Legend

| Symbol | Meaning |
|--------|---------|
| &#x1F535; | Fully implemented |
| &#x1F7E1; | Partially implemented |
| &#x2795; | Planned (in [`PORT_PLAN.md`](PORT_PLAN.md)) |
| &#x1F534; | Not implemented / not planned |

### Priority (for incomplete features)

| Priority | Meaning |
|----------|---------|
| **P4** | Highest — production blockers, widely needed, or near-free to implement |
| **P3** | High — meaningfully expands user base or enables key use cases |
| **P2** | Medium — useful improvement, broader coverage |
| **P1** | Lowest — niche, edge-case, or low demand |

### Test columns

| Column | Meaning |
|--------|---------|
| **Unit** | Number of unit tests (`#[test]` / `#[tokio::test]`) covering this feature |
| **E2E** | Number of end-to-end tests (in `vllm-e2e` crate) covering this feature |

> Test counts reflect tests that **directly** exercise each feature. ~168 additional unit tests cover cross-cutting infrastructure (layers, weight loading, tensor ops, protocol codec, engine core, config, errors) and are not attributed to individual rows.

---

## Summary: Major Feature Groups

| Feature Group | Python | Rust | Unit | E2E | Parity |
|---|:---:|:---:|---:|---:|---|
| [Model Architectures](#model-architectures) | &#x1F535; | &#x1F7E1; | 76 | 21 | `███░░░░░░░` 9/36 |
| [Quantization](#quantization) | &#x1F535; | &#x1F7E1; | 13 | 0 | `█░░░░░░░░░` 1/11 |
| [Serving / OpenAI API](#serving--openai-api) | &#x1F535; | &#x1F7E1; | 96 | 18 | `██████░░░░` 14/25 |
| [Sampling & Decoding](#sampling--decoding) | &#x1F535; | &#x1F7E1; | 49 | 12 | `████████░░` 18/21 |
| [KV Cache & Attention](#kv-cache--attention) | &#x1F535; | &#x1F7E1; | 96 | 0 | `█████░░░░░` 9/19 |
| [Scheduling](#scheduling) | &#x1F535; | &#x1F7E1; | 67 | 0 | `████████░░` 8/10 |
| [Hardware Backends](#hardware-backends) | &#x1F535; | &#x1F7E1; | 28 | 2 | `████░░░░░░` 3/8 |
| [Parallelism & Distribution](#parallelism--distribution) | &#x1F535; | &#x1F7E1; | 33 | 0 | `███░░░░░░░` 3/10 |
| [Performance Optimizations](#performance-optimizations) | &#x1F535; | &#x1F7E1; | 12 | 0 | `███░░░░░░░` 3/12 |
| [LoRA / Adapters](#lora--adapters) | &#x1F535; | &#x1F534; | 0 | 0 | `░░░░░░░░░░` 0/5 |
| [Speculative Decoding](#speculative-decoding) | &#x1F535; | &#x1F534; | 0 | 0 | `░░░░░░░░░░` 0/5 |
| [Multimodal / Vision-Language](#multimodal--vision-language) | &#x1F535; | &#x2795; | 0 | 0 | `░░░░░░░░░░` 0/10 |
| [Structured Output](#structured-output--guided-decoding) | &#x1F535; | &#x1F535; | 12 | 0 | `██████████` 4/4 |
| [Tool Calling](#tool-calling--function-calling) | &#x1F535; | &#x1F535; | 25 | 0 | `██████████` 7/7 |
| [Embeddings & Pooling](#embeddings--pooling) | &#x1F535; | &#x1F534; | 0 | 0 | `░░░░░░░░░░` 0/6 |
| [Observability & Operations](#observability--operations) | &#x1F535; | &#x1F535; | 15 | 0 | `██████████` 7/7 |
| [CLI & Deployment](#cli--deployment) | &#x1F535; | &#x1F7E1; | 14 | 0 | `██████████` 14/15 |
| | | **Total** | **536** | **53** | `█████░░░░░` **99/199** |

---

## Model Architectures

### Candle Backend (CPU / CUDA)

| Architecture | Python | Rust | Unit | E2E | Pri |
|---|:---:|:---:|---:|---:|:---:|
| LLaMA / LLaMA 2 / LLaMA 3 | &#x1F535; | &#x1F535; | 10 | 0 | |
| Mistral | &#x1F535; | &#x1F535; | 0 | 0 | |
| Qwen2 | &#x1F535; | &#x1F535; | 4 | 0 | |
| Qwen3 | &#x1F535; | &#x1F535; | 0 | 0 | |
| Phi-3 | &#x1F535; | &#x1F535; | 0 | 0 | |
| Gemma 2 | &#x1F535; | &#x1F535; | 7 | 0 | |
| DeepSeek V2 / V3 (MLA + MoE) | &#x1F535; | &#x1F535; | 7 | 0 | |
| Command R (Cohere) | &#x1F535; | &#x1F535; | 8 | 0 | |
| Kimi K2.5 (text-only, via DeepSeek V2) | &#x1F535; | &#x1F535; | 0 | 0 | |
| Quantized LLaMA (GGUF) | &#x1F535; | &#x1F535; | 1 | 0 | |
| Gemma 1 | &#x1F535; | &#x1F534; | — | — | P1 |
| Qwen 1 | &#x1F535; | &#x1F534; | — | — | P1 |
| Qwen2 MoE | &#x1F535; | &#x1F535; | 8 | 0 | |
| Qwen3 MoE | &#x1F535; | &#x1F535; | 8 | 0 | |
| Qwen3 Next (hybrid linear attn) | &#x1F535; | &#x1F534; | — | — | P3 |
| Mixtral (MoE) | &#x1F535; | &#x2795; | — | — | P3 |
| GPT-NeoX | &#x1F535; | &#x2795; | — | — | P1 |
| GPT-J | &#x1F535; | &#x2795; | — | — | P1 |
| Falcon | &#x1F535; | &#x2795; | — | — | P1 |
| BLOOM | &#x1F535; | &#x2795; | — | — | P1 |
| MPT | &#x1F535; | &#x2795; | — | — | P1 |
| StarCoder / StarCoder2 | &#x1F535; | &#x2795; | — | — | P2 |
| OPT | &#x1F535; | &#x1F534; | — | — | P1 |
| Phi-1 / Phi-2 | &#x1F535; | &#x1F534; | — | — | P1 |
| Phi-4 | &#x1F535; | &#x1F534; | — | — | P3 |
| Gemma 3 / 3n | &#x1F535; | &#x1F534; | — | — | P4 |
| ChatGLM / GLM-4 | &#x1F535; | &#x1F534; | — | — | P2 |
| Baichuan | &#x1F535; | &#x1F534; | — | — | P1 |
| DBRX | &#x1F535; | &#x1F534; | — | — | P1 |
| Mamba / Bamba / Jamba | &#x1F535; | &#x1F534; | — | — | P2 |
| OLMo / OLMoE | &#x1F535; | &#x1F534; | — | — | P1 |
| Nemotron | &#x1F535; | &#x1F534; | — | — | P1 |
| Exaone | &#x1F535; | &#x1F534; | — | — | P1 |
| StableLM | &#x1F535; | &#x1F534; | — | — | P1 |
| Solar | &#x1F535; | &#x1F534; | — | — | P1 |
| Arctic (MoE) | &#x1F535; | &#x1F534; | — | — | P1 |
| PLaMo | &#x1F535; | &#x1F534; | — | — | P1 |
| Kimi Linear (KDA + MoE) | &#x1F535; | &#x1F534; | — | — | P2 |
| Zamba 2 | &#x1F535; | &#x1F534; | — | — | P1 |
| ~200+ other architectures | &#x1F535; | &#x1F534; | — | — | P1 |

### MLX Backend (Apple Silicon)

| Architecture | Python | Rust | Unit | E2E | Pri |
|---|:---:|:---:|---:|---:|:---:|
| LLaMA / LLaMA 2 / LLaMA 3 | N/A | &#x1F535; | 5 | 2 | |
| Mistral | N/A | &#x1F535; | 0 | 0 | |
| Qwen2 (with bias) | N/A | &#x1F535; | 0 | 0 | |
| Qwen3 (with QK norms) | N/A | &#x1F535; | 0 | 0 | |
| Gemma 1 | N/A | &#x1F535; | 2 | 0 | |
| Gemma 2 | N/A | &#x1F535; | 7 | 0 | |
| Phi-3 (fused projections) | N/A | &#x1F535; | 4 | 0 | |
| DeepSeek V2 / V3 | N/A | &#x1F535; | 5 | 2 | |
| Command R (Cohere) | N/A | &#x1F535; | 4 | 0 | |
| Quantized LLaMA (mlx-community 4-bit) | N/A | &#x1F535; | 4 | 8 | |
| Quantized Mistral (4-bit) | N/A | &#x1F535; | 0 | 2 | |
| Quantized Qwen2/3 (4-bit) | N/A | &#x1F535; | 0 | 5 | |
| Quantized Gemma 1 (4-bit) | N/A | &#x1F535; | 0 | 0 | |
| Quantized Gemma 2 (4-bit) | N/A | &#x1F535; | 0 | 2 | |
| Quantized Phi-3 (4-bit) | N/A | &#x1F535; | 0 | 2 | |
| Quantized DeepSeek V2 (4-bit) | N/A | &#x1F535; | 2 | 2 | |
| Quantized Command R (4-bit) | N/A | &#x1F535; | 0 | 0 | |
| Qwen3 MoE (float) | N/A | &#x1F535; | 4 | 0 | |
| Quantized Qwen3 MoE (4/8-bit) | N/A | &#x1F535; | 0 | 4 | |
| Qwen2 MoE (float + quantized) | N/A | &#x1F535; | 0 | 0 | |
| Kimi K2.5 text-only (via DeepSeek V2) | N/A | &#x1F535; | 0 | 0 | |
| Quantized Kimi K2.5 text-only (4-bit) | N/A | &#x1F535; | 0 | 0 | |

> Python vLLM does not have an MLX backend. The Rust MLX backend is unique to the Rust port. All E2E tests use `mlx-community` models and exercise the MLX backend (`--features metal`). MLX unit tests require `--test-threads=1`.

---

## Quantization

| Method | Python | Rust | Unit | E2E | Pri |
|---|:---:|:---:|---:|---:|:---:|
| GGUF (Q4_0 / Q4_K / Q8_0 / etc.) | &#x1F535; | &#x1F535; | 11 | 0 | |
| MLX native 4-bit group quantization | N/A | &#x1F535; | 2 | 0 | |
| GPTQ | &#x1F535; | &#x1F534; | — | — | P3 |
| AWQ | &#x1F535; | &#x1F534; | — | — | P3 |
| Marlin (GPTQ-Marlin / AWQ-Marlin) | &#x1F535; | &#x1F534; | — | — | P2 |
| FP8 (FBGemm / ModelOpt) | &#x1F535; | &#x1F534; | — | — | P3 |
| BitsAndBytes (4-bit / 8-bit) | &#x1F535; | &#x1F534; | — | — | P2 |
| SqueezeLLM | &#x1F535; | &#x1F534; | — | — | P1 |
| Compressed Tensors | &#x1F535; | &#x1F534; | — | — | P1 |
| TorchAO | &#x1F535; | &#x1F534; | — | — | P1 |
| MXFP4 | &#x1F535; | &#x1F534; | — | — | P1 |
| CPU WNA16 | &#x1F535; | &#x1F534; | — | — | P1 |
| Experts INT8 (MoE) | &#x1F535; | &#x1F534; | — | — | P1 |

---

## Serving / OpenAI API

| Feature | Python | Rust | Unit | E2E | Pri |
|---|:---:|:---:|---:|---:|:---:|
| `POST /v1/chat/completions` | &#x1F535; | &#x1F535; | 9 | 4 | |
| `POST /v1/completions` | &#x1F535; | &#x1F535; | 11 | 0 | |
| SSE streaming (chat) | &#x1F535; | &#x1F535; | 6 | 8 | |
| SSE streaming (completions) | &#x1F535; | &#x1F535; | 0 | 0 | |
| `GET /v1/models` | &#x1F535; | &#x1F535; | 3 | 0 | |
| `GET /health` | &#x1F535; | &#x1F535; | 1 | 0 | |
| `GET /version` | &#x1F535; | &#x1F535; | 1 | 1 | |
| `n` parameter (multiple completions) | &#x1F535; | &#x1F535; | 4 | 2 | |
| Multi-prompt completions | &#x1F535; | &#x1F535; | 2 | 0 | |
| Chat templates (Jinja2) | &#x1F535; | &#x1F535; | 14 | 3 | |
| `POST /v1/embeddings` | &#x1F535; | &#x1F534; | — | — | P3 |
| `POST /v1/chat/completions` tool_calls | &#x1F535; | &#x1F535; | 3 | 0 | |
| `response_format` (JSON mode/schema) | &#x1F535; | &#x1F535; | 6 | 0 | |
| Anthropic Messages API | &#x1F535; | &#x1F534; | — | — | P2 |
| gRPC server | &#x1F535; | &#x2795; | — | — | P1 |
| MCP tool server | &#x1F535; | &#x1F534; | — | — | P2 |
| Batch processing | &#x1F535; | &#x1F534; | — | — | P3 |
| Responses API | &#x1F535; | &#x1F534; | — | — | P1 |
| Speech-to-text | &#x1F535; | &#x1F534; | — | — | P1 |
| Realtime API | &#x1F535; | &#x1F534; | — | — | P1 |
| SageMaker integration | &#x1F535; | &#x1F534; | — | — | P1 |
| SSL / TLS | &#x1F535; | &#x1F534; | — | — | P3 |
| CORS | &#x1F535; | &#x1F535; | 0 | 0 | |
| `usage` field in responses | &#x1F535; | &#x1F535; | 2 | 0 | |
| `best_of` / `n` with reranking | &#x1F535; | &#x1F534; | — | — | P2 |

> Unit counts include tests from `engine.rs`, `server.rs`, `protocol.rs`, and `chat_template.rs`. Tokenizer (6 tests) and detokenizer (18 tests) provide additional cross-cutting serving coverage not attributed to individual rows.

---

## Sampling & Decoding

| Feature | Python | Rust | Unit | E2E | Pri |
|---|:---:|:---:|---:|---:|:---:|
| Greedy (argmax) | &#x1F535; | &#x1F535; | 4 | 1 | |
| Temperature | &#x1F535; | &#x1F535; | 2 | 2 | |
| Top-k | &#x1F535; | &#x1F535; | 1 | 1 | |
| Top-p (nucleus) | &#x1F535; | &#x1F535; | 1 | 1 | |
| Min-p | &#x1F535; | &#x1F535; | 1 | 1 | |
| Repetition penalty | &#x1F535; | &#x1F535; | 2 | 1 | |
| Frequency penalty | &#x1F535; | &#x1F535; | 2 | 1 | |
| Presence penalty | &#x1F535; | &#x1F535; | 2 | 1 | |
| Logprobs | &#x1F535; | &#x1F535; | 5 | 1 | |
| Prompt logprobs | &#x1F535; | &#x1F534; | — | — | P3 |
| Logit bias | &#x1F535; | &#x1F535; | 3 | 0 | |
| Beam search | &#x1F535; | &#x1F534; | — | — | P1 |
| `best_of` | &#x1F535; | &#x1F534; | — | — | P2 |
| `max_tokens` / `max_completion_tokens` | &#x1F535; | &#x1F535; | 2 | 2 | |
| Stop strings | &#x1F535; | &#x1F535; | 3 | 0 | |
| Stop token IDs | &#x1F535; | &#x1F535; | 1 | 0 | |
| EOS detection (multi-EOS) | &#x1F535; | &#x1F535; | 1 | 0 | |
| `ignore_eos` | &#x1F535; | &#x1F535; | 1 | 0 | |
| `min_tokens` | &#x1F535; | &#x1F535; | 1 | 0 | |
| Seed (reproducible sampling) | &#x1F535; | &#x1F535; | 2 | 1 | |
| Guided decoding (grammar/regex/JSON) | &#x1F535; | &#x1F535; | 6 | 0 | P4 |

> Unit counts drawn from `sampling.rs` (27 tests — validation, serde, types) and `sampler.rs` (21 tests — greedy, temperature, top-k/p, min-p, penalties, logprobs, grammar mask). Guided decoding unit tests count API-level resolve/conflict tests in `engine.rs`; grammar engine tests are in the [Structured Output](#structured-output--guided-decoding) section.

---

## KV Cache & Attention

| Feature | Python | Rust | Unit | E2E | Pri |
|---|:---:|:---:|---:|---:|:---:|
| Per-request KV cache | &#x1F535; | &#x1F535; | 1 | 0 | |
| Block-based KV cache (PagedAttention) | &#x1F535; | &#x1F535; | 19 | 0 | |
| KV block pool (pre-allocated) | &#x1F535; | &#x1F535; | 13 | 0 | |
| Direct block KV reads (no gather copy) | &#x1F535; | &#x1F535; | 2 | 0 | |
| Paged decode attention (per-block scoring) | &#x1F535; | &#x1F535; | 6 | 0 | |
| Prefix caching (hash-based) | &#x1F535; | &#x1F535; | 5 | 0 | |
| Automatic prefix caching | &#x1F535; | &#x1F535; | 2 | 0 | |
| Chunked prefill | &#x1F535; | &#x1F535; | 1 | 0 | |
| KV cache compression (latent caching) | &#x1F535; | &#x1F534; | — | — | P2 |
| KV cache offloading (CPU ↔ GPU) | &#x1F535; | &#x1F534; | — | — | P3 |
| KV cache transfer (distributed) | &#x1F535; | &#x1F534; | — | — | P1 |
| Multi-group KV cache (hybrid models) | &#x1F535; | &#x1F534; | — | — | P1 |
| FlashAttention v2 | &#x1F535; | &#x1F534; | — | — | P3 |
| FlashInfer | &#x1F535; | &#x1F534; | — | — | P2 |
| FlexAttention | &#x1F535; | &#x1F534; | — | — | P1 |
| xFormers | &#x1F535; | &#x1F534; | — | — | P1 |
| MLA (Multi-head Latent Attention) | &#x1F535; | &#x1F535; | 4 | 0 | |
| Sliding window attention | &#x1F535; | &#x1F535; | 13 | 0 | |
| Tree attention | &#x1F535; | &#x1F534; | — | — | P1 |

> Unit counts: `block_pool.rs` (19), `free_block_queue.rs` (16), `kv_cache_manager.rs` (13), `kv_cache_block.rs` (11), `kv_block_pool.rs` (13), `attention.rs` (25), MLX `cache.rs` (1). Sliding window: 7 attention.rs + 2 gemma2.rs interleaved + 2 qwen2.rs max_window_layers + 1 MLX phi3 trim + 1 array-format parsing = 13. Per-row counts reflect the primary feature each test targets; some tests cross-cut multiple rows. Total section: 103 unit tests.

---

## Scheduling

| Feature | Python | Rust | Unit | E2E | Pri |
|---|:---:|:---:|---:|---:|:---:|
| Continuous batching | &#x1F535; | &#x1F535; | 6 | 0 | |
| FCFS request queue | &#x1F535; | &#x1F535; | 7 | 0 | |
| Priority request queue | &#x1F535; | &#x1F535; | 7 | 0 | |
| Preemption | &#x1F535; | &#x1F535; | 3 | 0 | |
| Chunked prefill scheduling | &#x1F535; | &#x1F535; | 2 | 0 | |
| Block allocation / eviction | &#x1F535; | &#x1F535; | 19 | 0 | |
| Prefix cache hits | &#x1F535; | &#x1F535; | 5 | 0 | |
| Pause / resume | &#x1F535; | &#x1F535; | 3 | 0 | |
| Multi-step scheduling | &#x1F535; | &#x1F534; | — | — | P3 |
| Async scheduler | &#x1F535; | &#x1F534; | — | — | P2 |

> Unit counts: `scheduler/core.rs` (30), `scheduler/request_queue.rs` (14), `scheduler/output.rs` (5), `scheduler/interface.rs` (2), `request.rs` (16). Block allocation count includes `block_pool.rs` and `kv_cache_manager.rs` allocate/free/eviction tests counted above in KV Cache; per-row counts here reflect scheduler-specific tests.

---

## Hardware Backends

| Backend | Python | Rust | Unit | E2E | Pri |
|---|:---:|:---:|---:|---:|:---:|
| CPU | &#x1F535; | &#x1F535; | 4 | 0 | |
| CUDA (NVIDIA GPU) | &#x1F535; | &#x1F7E1; | 2 | 0 | P3 |
| Metal / MLX (Apple Silicon) | &#x1F534; | &#x1F535; | 4 | 0 | |
| ROCm (AMD GPU) | &#x1F535; | &#x1F534; | — | — | P2 |
| TPU | &#x1F535; | &#x1F534; | — | — | P1 |
| XPU (Intel) | &#x1F535; | &#x1F534; | — | — | P1 |
| AWS Neuron | &#x1F535; | &#x1F534; | — | — | P1 |
| Device auto-detection | &#x1F535; | &#x1F535; | 4 | 0 | |
| Memory profiling / `--gpu-memory-utilization` | &#x1F535; | &#x1F535; | 6 | 2 | |

> Unit counts from `candle_worker.rs` (24 total) and `mlx_worker.rs` (4). Device auto-detection includes `parse_device` tests for cpu/cuda/metal/auto. E2E float16 tests validate dtype selection end-to-end.

---

## Parallelism & Distribution

| Feature | Python | Rust | Unit | E2E | Pri |
|---|:---:|:---:|---:|---:|:---:|
| Parallel config types (TP/PP groups) | &#x1F535; | &#x1F535; | 10 | 0 | |
| UniProc executor (single-process) | &#x1F535; | &#x1F535; | 10 | 0 | |
| MultiProc executor (multi-worker) | &#x1F535; | &#x1F535; | 6 | 0 | |
| Tensor parallelism (actual sharding) | &#x1F535; | &#x1F7E1; | 0 | 0 | P3 |
| Pipeline parallelism | &#x1F535; | &#x1F534; | — | — | P2 |
| NCCL communication | &#x1F535; | &#x2795; | — | — | P3 |
| Ray distributed executor | &#x1F535; | &#x1F534; | — | — | P1 |
| Expert parallelism (MoE) | &#x1F535; | &#x1F534; | — | — | P2 |
| Data parallelism | &#x1F535; | &#x1F534; | — | — | P2 |
| Weight transfer / migration | &#x1F535; | &#x1F534; | — | — | P1 |

> Unit counts: `parallel.rs` (10), `uniproc.rs` (10), `multiproc.rs` (6), `worker.rs` (7 — NoopWorker lifecycle).

---

## Performance Optimizations

| Feature | Python | Rust | Unit | E2E | Pri |
|---|:---:|:---:|---:|---:|:---:|
| CUDA graphs | &#x1F535; | &#x1F534; | — | — | P3 |
| FlashAttention v2 kernels | &#x1F535; | &#x1F534; | — | — | P3 |
| FlashInfer kernels | &#x1F535; | &#x1F534; | — | — | P2 |
| xFormers memory-efficient attention | &#x1F535; | &#x1F534; | — | — | P1 |
| Fused SiLU-and-mul kernel | &#x1F535; | &#x1F534; | — | — | P2 |
| Fused RMSNorm kernel | &#x1F535; | &#x1F534; | — | — | P2 |
| Fused RoPE kernel | &#x1F535; | &#x1F534; | — | — | P2 |
| Custom all-reduce kernel | &#x1F535; | &#x1F534; | — | — | P1 |
| MoE fused routing kernels | &#x1F535; | &#x1F534; | — | — | P2 |
| MLX lazy eval graph fusion | N/A | &#x1F535; | 0 | 0 | |
| MLX single-eval sampling fusion | N/A | &#x1F535; | 0 | 0 | |
| Pre-transposed weights (Metal) | N/A | &#x1F535; | 0 | 0 | |
| Native dtype inference (`--dtype auto`) | &#x1F535; | &#x1F535; | 4 | 2 | |
| Continuous batching | &#x1F535; | &#x1F535; | 6 | 0 | |
| Paged KV (no gather copy on decode) | &#x1F535; | &#x1F535; | 2 | 0 | |

> CPU kernel stubs in `vllm-kernels` (12 tests: rotary 3, activation 3, norm 2, cache 2, attention 2) provide building blocks for performance features. Native dtype unit tests count `candle_worker.rs` dtype parsing tests.

---

## LoRA / Adapters

| Feature | Python | Rust | Unit | E2E | Pri |
|---|:---:|:---:|---:|---:|:---:|
| LoRA adapter loading | &#x1F535; | &#x1F534; | — | — | P3 |
| Multi-LoRA serving | &#x1F535; | &#x1F534; | — | — | P2 |
| LoRA weight merging | &#x1F535; | &#x1F534; | — | — | P2 |
| Punica kernels | &#x1F535; | &#x1F534; | — | — | P1 |
| Dynamic adapter switching | &#x1F535; | &#x1F534; | — | — | P2 |

---

## Speculative Decoding

| Feature | Python | Rust | Unit | E2E | Pri |
|---|:---:|:---:|---:|---:|:---:|
| Draft model (MLP speculator) | &#x1F535; | &#x1F534; | — | — | P2 |
| Eagle speculative decoding | &#x1F535; | &#x1F534; | — | — | P2 |
| Medusa heads | &#x1F535; | &#x1F534; | — | — | P2 |
| N-gram proposer | &#x1F535; | &#x1F534; | — | — | P3 |
| Suffix decoding | &#x1F535; | &#x1F534; | — | — | P1 |

---

## Multimodal / Vision-Language

| Feature | Python | Rust | Unit | E2E | Pri |
|---|:---:|:---:|---:|---:|:---:|
| Image input processing | &#x1F535; | &#x2795; | — | — | P3 |
| LLaVA | &#x1F535; | &#x2795; | — | — | P2 |
| Qwen-VL / Qwen2.5-VL | &#x1F535; | &#x2795; | — | — | P3 |
| Pixtral | &#x1F535; | &#x1F534; | — | — | P2 |
| InternVL | &#x1F535; | &#x1F534; | — | — | P2 |
| Phi-3V / Phi-4MM | &#x1F535; | &#x1F534; | — | — | P2 |
| Gemma 3 multimodal | &#x1F535; | &#x1F534; | — | — | P2 |
| Molmo | &#x1F535; | &#x1F534; | — | — | P1 |
| PaliGemma | &#x1F535; | &#x1F534; | — | — | P1 |
| Audio models (Whisper, Qwen-Audio) | &#x1F535; | &#x1F534; | — | — | P1 |

---

## Structured Output / Guided Decoding

| Feature | Python | Rust | Unit | E2E | Pri |
|---|:---:|:---:|---:|---:|:---:|
| `response_format: json_object` | &#x1F535; | &#x1F535; | 2 | 0 | |
| `response_format: json_schema` | &#x1F535; | &#x1F535; | 3 | 0 | |
| Grammar-guided logit masking | &#x1F535; | &#x1F535; | 5 | 0 | |
| Regex-constrained decoding | &#x1F535; | &#x1F535; | 2 | 0 | P4 |

> Unit counts from `grammar.rs` (8 tests: mask basic/empty/all-allowed, build vocabulary, JSON schema compile, regex digit, guided_grammar regex variant, advance+finish) plus `engine.rs` resolve/conflict tests (4 tests, attributed to [Serving](#serving--openai-api) `response_format` row). Total section: 12 unit tests.

---

## Tool Calling / Function Calling

| Feature | Python | Rust | Unit | E2E | Pri |
|---|:---:|:---:|---:|---:|:---:|
| `tools` / `tool_choice` request fields | &#x1F535; | &#x1F535; | 7 | 0 | |
| Chat template tool definitions | &#x1F535; | &#x1F535; | 2 | 0 | |
| Model-emitted tool call parsing | &#x1F535; | &#x1F535; | 6 | 0 | |
| `tool_calls` in response | &#x1F535; | &#x1F535; | 2 | 0 | |
| Streaming tool call deltas | &#x1F535; | &#x1F535; | 4 | 0 | |
| Parallel tool calls | &#x1F535; | &#x1F535; | 2 | 0 | |
| `tool_choice: {function: {name}}` validation | &#x1F535; | &#x1F535; | 2 | 0 | |

> Unit counts from `tool_parser.rs` (18 tests: Hermes/LlamaJson single/multi/malformed, streaming partial JSON, registry lookup) and `engine.rs` tool_choice helpers (7 tests: is_tool_choice_none variants, get_tool_choice_function_name variants). Total section: 25 unit tests.

---

## Embeddings & Pooling

> Embedding in Python vLLM is a **server-level execution mode**, not just an API endpoint. It is selected via `--runner pooling` or auto-detected from model architecture (`is_pooling_model = True` or the presence of a sentence-transformers `pooling_config`). The execution path is fundamentally different from generation: a single forward pass returns hidden-state vectors — there is no iterative decode loop and no KV cache for generation. Decoder models (LLaMA, Mistral, etc.) can be repurposed for embeddings via the sentence-transformers pattern (`--convert embed`), while dedicated embedding architectures (BERT, ModernBERT) use encoder-only bidirectional attention.

| Feature | Python | Rust | Unit | E2E | Pri |
|---|:---:|:---:|---:|---:|:---:|
| `/v1/embeddings` endpoint | &#x1F535; | &#x1F534; | — | — | P3 |
| Pooling execution mode (`--runner pooling`) | &#x1F535; | &#x1F534; | — | — | P3 |
| Decoder-based embedding (sentence-transformers) | &#x1F535; | &#x1F534; | — | — | P3 |
| Encoder-only models (BERT, ModernBERT) | &#x1F535; | &#x1F534; | — | — | P2 |
| Pooling strategies (CLS, mean, last) | &#x1F535; | &#x1F534; | — | — | P3 |
| Reward / reranking models | &#x1F535; | &#x1F534; | — | — | P1 |

---

## Observability & Operations

| Feature | Python | Rust | Unit | E2E | Pri |
|---|:---:|:---:|---:|---:|:---:|
| Prometheus `/metrics` endpoint | &#x1F535; | &#x1F535; | 2 | 0 | |
| Request counters (total, active) | &#x1F535; | &#x1F535; | 1 | 0 | |
| Token counters (prompt, generation) | &#x1F535; | &#x1F535; | 1 | 0 | |
| Latency histograms | &#x1F535; | &#x1F535; | 0 | 0 | |
| Structured logging (tracing) | &#x1F535; | &#x1F535; | 1 | 0 | |
| OpenTelemetry tracing | &#x1F535; | &#x1F534; | — | — | P2 |
| ORCA metrics (endpoint-load-metrics header) | &#x1F535; | &#x1F535; | 8 | 0 | |
| MLX per-step timing (prefill/decode ms) | N/A | &#x1F535; | 2 | 0 | |
| Benchmark CLI (`vllm bench`) | &#x1F535; | &#x1F535; | 0 | 0 | |

> Unit counts: `metrics.rs` (4 — singleton, counter, gauge, encode), `orca.rs` (5 — text/JSON/case/unsupported/nonzero), `server.rs` ORCA tests (3 — text/JSON/absent), `telemetry.rs` (1 — init idempotent), `mlx_worker.rs` (2 — timing fields, avg latency).

---

## CLI & Deployment

| Feature | Python | Rust | Unit | E2E | Pri |
|---|:---:|:---:|---:|---:|:---:|
| `vllm serve <model>` | &#x1F535; | &#x1F535; | 2 | 0 | |
| `vllm bench <model>` | &#x1F535; | &#x1F535; | 2 | 0 | |
| `vllm convert` | &#x1F535; | &#x1F535; | 1 | 0 | |
| HF Hub model download | &#x1F535; | &#x1F535; | 0 | 0 | |
| Sharded weight loading | &#x1F535; | &#x1F535; | 0 | 0 | |
| `--model` / positional model arg | &#x1F535; | &#x1F535; | 3 | 0 | |
| `--dtype auto` / explicit dtype | &#x1F535; | &#x1F535; | 0 | 0 | |
| `--device auto` / explicit device | &#x1F535; | &#x1F535; | 0 | 0 | |
| `--gpu-memory-utilization` | &#x1F535; | &#x1F535; | 0 | 0 | |
| `--gguf-file` | &#x1F535; | &#x1F535; | 0 | 0 | |
| `--tool-call-parser` | &#x1F535; | &#x1F535; | 0 | 0 | |
| `--features metal` (MLX backend) | N/A | &#x1F535; | 0 | 0 | |
| Dockerfile.cpu | &#x1F535; | &#x1F535; | 0 | 0 | |
| Dockerfile.cuda | &#x1F535; | &#x1F535; | 0 | 0 | |
| `VLLM_MODEL` env var | &#x1F535; | &#x1F535; | 0 | 0 | |
| PyO3 bridge (Rust scheduler in Python) | &#x1F535; | &#x1F7E1; | 0 | 0 | P1 |
| Python fallback for unported models | &#x1F535; | &#x2795; | — | — | P1 |
| Standalone binary (no Python runtime) | &#x1F534; | &#x1F535; | 0 | 0 | |

> Unit counts from `args.rs` (8 — CLI parsing: help, serve/bench positional/flag model, precedence, no-model error, convert) and `init.rs` (6 — extract model name, compute num blocks variants).

---

---

## Stats

| Metric | Python | Rust |
|---|---|---|
| Model architectures | ~248 | 9 candle + 9 MLX (+ quantized variants) |
| Quantization methods | ~14 | 2 (GGUF + MLX native 4-bit) |
| Attention backends | ~15 | 1 (custom SDPA) |
| Hardware backends | 6 (CUDA, ROCm, CPU, TPU, XPU, Neuron) | 3 (CPU, CUDA, Metal/MLX) |
| Lines of code | ~507K Python + ~89K C++/CUDA | ~30.7K Rust |
| Unit tests | ~948 test files | 701 passing (660 non-MLX + 41 MLX) |
| E2E tests | — | 53 passing (24 basic serving + 21 chat/sampling + 8 streaming) |
| Crate count | N/A | 14 crates (incl. vllm-e2e) |
