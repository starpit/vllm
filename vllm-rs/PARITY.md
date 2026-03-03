# vLLM Feature Parity Punchlist: Python vs Rust

> Generated 2026-03-02 | Rust port: `vllm-rs/` on branch `feat/rust` (1018 tests: 904 unit + 114 e2e, 0 clippy errors)

### Legend

| Symbol | Meaning | Count |
|--------|---------|------:|
| &#x1F535; | Fully implemented | 175 |
| &#x1F7E1; | Partially implemented | 6 |
| &#x1F534; | Not implemented | 93 |

### Priority (for incomplete features)

| Priority | Meaning | Count |
|----------|---------|------:|
| **P4** | Highest — production blockers, widely needed, or near-free to implement | 0 |
| **P3** | High — meaningfully expands user base or enables key use cases | 12 |
| **P2** | Medium — useful improvement, broader coverage | 32 |
| **P1** | Lowest — niche, edge-case, or low demand | 49 |
| **P0** | Won't do — deprecated in Python vLLM V1+ or superseded | 6 |

### Test columns

| Column | Meaning |
|--------|---------|
| **Unit** | Number of unit tests (`#[test]` / `#[tokio::test]`) covering this feature |
| **E2E** | Number of end-to-end tests (in `vllm-e2e` crate) covering this feature |

> Test counts reflect tests that **directly** exercise each feature. ~168 additional unit tests cover cross-cutting infrastructure (layers, weight loading, tensor ops, protocol codec, engine core, config, errors) and are not attributed to individual rows.

---

## Summary: Major Feature Groups

> **Python Features** and **Rust Parity** exclude P0 items (deprecated / won't implement). **Rust Additions** counts features unique to the Rust port (e.g. MLX backend, standalone binary).

| Feature Group | Python | Rust Parity | +Rust | Unit | E2E |
|---|---:|---|---:|---:|---:|
| [Model Architectures](#model-architectures) | 36 | `████░░░░░░` 14/36 | 32 | 139 | 31 |
| [Quantization](#quantization) | 12 | `█████░░░░░` 6/12 | 1 | 84 | 18 |
| [Serving / OpenAI API](#serving--openai-api) | 24 | `███████░░░` 17/24 | 0 | 104 | 33 |
| [Sampling & Decoding](#sampling--decoding) | 19 | `██████████` 19/19 | 0 | 53 | 13 |
| [KV Cache & Attention](#kv-cache--attention) | 19 | `█████░░░░░` 10/19 | 0 | 105 | 0 |
| [Scheduling](#scheduling) | 10 | `█████████░` 9/10 | 0 | 73 | 1 |
| [Hardware Backends](#hardware-backends) | 8 | `████░░░░░░` 3/8 | 1 | 67 | 5 |
| [Parallelism & Distribution](#parallelism--distribution) | 10 | `████░░░░░░` 3+2/10 | 0 | 39 | 1 |
| [Performance Optimizations](#performance-optimizations) | 13 | `███████░░░` 9/13 | 4 | 55 | 0 |
| [GPU Compute Kernels (Triton)](#gpu-compute-kernels-triton-equivalents) | 12 | `████░░░░░░` 5/12 | 0 | 30 | 0 |
| [LoRA / Adapters](#lora--adapters) | 5 | `████░░░░░░` 2/5 | 0 | 12 | 8 |
| [Speculative Decoding](#speculative-decoding) | 5 | `██░░░░░░░░` 1/5 | 0 | 20 | 0 |
| [Multimodal / Vision-Language](#multimodal--vision-language) | 10 | `███░░░░░░░` 3/10 | 12 | 13 | 5 |
| [Structured Output](#structured-output--guided-decoding) | 4 | `██████████` 4/4 | 0 | 12 | 0 |
| [Tool Calling](#tool-calling--function-calling) | 7 | `██████████` 7/7 | 0 | 25 | 0 |
| [Embeddings & Pooling](#embeddings--pooling) | 8 | `████████░░` 6/8 | 0 | 31 | 19 |
| [Observability & Operations](#observability--operations) | 7 | `██████████` 7/7 | 1 | 15 | 0 |
| [CLI & Deployment](#cli--deployment) | 17 | `██████████` 16/17 | 2 | 19 | 6 |
| **Total** | **214** | `██████░░░░` **132/214** | **37** | **774** | **104** |

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
| Quantized Gemma 3 (GGUF, text + multimodal) | &#x1F535; | &#x1F535; | 2 | 3 | |
| Gemma 1 | &#x1F535; | &#x1F534; | — | — | P1 |
| Qwen 1 | &#x1F535; | &#x1F534; | — | — | P1 |
| Qwen2 MoE | &#x1F535; | &#x1F535; | 8 | 0 | |
| Qwen3 MoE | &#x1F535; | &#x1F535; | 8 | 0 | |
| Qwen3 Next (hybrid linear attn) | &#x1F535; | &#x1F535; | 26 | 3 | |
| Mixtral (MoE) | &#x1F535; | &#x1F535; | 7 | 0 | |
| Granite (IBM) | &#x1F535; | &#x1F535; | 3 | 4 | |
| Quantized Granite (GGUF) | &#x1F535; | &#x1F535; | 0 | 2 | |
| GPT-NeoX | &#x1F535; | &#x1F534; | — | — | P1 |
| GPT-J | &#x1F535; | &#x1F534; | — | — | P1 |
| Falcon | &#x1F535; | &#x1F534; | — | — | P1 |
| BLOOM | &#x1F535; | &#x1F534; | — | — | P1 |
| MPT | &#x1F535; | &#x1F534; | — | — | P1 |
| StarCoder / StarCoder2 | &#x1F535; | &#x1F534; | — | — | P2 |
| OPT | &#x1F535; | &#x1F534; | — | — | P1 |
| Phi-1 / Phi-2 | &#x1F535; | &#x1F534; | — | — | P1 |
| Phi-4 (via Phi3ForCausalLM + LongRoPE) | &#x1F535; | &#x1F535; | 0 | 4 | |
| Gemma 3 (text-only) | &#x1F535; | &#x1F535; | 9 | 0 | |
| Gemma 3 VLM (SigLIP + projector + LM) | &#x1F535; | &#x1F535; | 4 | 3 | |
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
| Phi-3 / Phi-4 (fused projections, LongRoPE) | N/A | &#x1F535; | 10 | 4 | |
| DeepSeek V2 / V3 | N/A | &#x1F535; | 5 | 2 | |
| Command R (Cohere) | N/A | &#x1F535; | 4 | 0 | |
| Quantized LLaMA (mlx-community 4-bit) | N/A | &#x1F535; | 4 | 8 | |
| Quantized Mistral (4-bit) | N/A | &#x1F535; | 0 | 2 | |
| Quantized Qwen2/3 (4-bit) | N/A | &#x1F535; | 0 | 5 | |
| Quantized Gemma 1 (4-bit) | N/A | &#x1F535; | 0 | 0 | |
| Quantized Gemma 2 (4-bit) | N/A | &#x1F535; | 0 | 2 | |
| Quantized Phi-3 / Phi-4 (4-bit) | N/A | &#x1F535; | 0 | 6 | |
| Quantized DeepSeek V2 (4-bit) | N/A | &#x1F535; | 2 | 2 | |
| Quantized Command R (4-bit) | N/A | &#x1F535; | 0 | 0 | |
| Qwen3 MoE (float) | N/A | &#x1F535; | 4 | 0 | |
| Quantized Qwen3 MoE (4/8-bit) | N/A | &#x1F535; | 0 | 4 | |
| Qwen2 MoE (float + quantized) | N/A | &#x1F535; | 0 | 0 | |
| Gemma 3 (text-only) | N/A | &#x1F535; | 5 | 4 | |
| Quantized Gemma 3 (4-bit) | N/A | &#x1F535; | 0 | 4 | |
| Gemma 3 VLM (float, SigLIP + projector + LM) | N/A | &#x1F535; | 0 | 5 | |
| Quantized Gemma 3 VLM (4-bit LM, float vision) | N/A | &#x1F535; | 0 | 5 | |
| Kimi K2.5 text-only (via DeepSeek V2) | N/A | &#x1F535; | 0 | 0 | |
| Quantized Kimi K2.5 text-only (4-bit) | N/A | &#x1F535; | 0 | 0 | |
| Mixtral (MoE, float) | N/A | &#x1F535; | 3 | 0 | |
| Qwen3 Next (hybrid GDN + full attn + MoE, float) | N/A | &#x1F535; | 3 | 0 | |
| Quantized Qwen3 Next (4-bit) | N/A | &#x1F535; | 0 | 0 | |
| Quantized Mixtral (MoE, 4-bit) | N/A | &#x1F535; | 0 | 0 | |
| Quantized Granite (4-bit) | N/A | &#x1F535; | 0 | 3 | |

> Python vLLM does not have an MLX backend. The Rust MLX backend is unique to the Rust port. All E2E tests use `mlx-community` models and exercise the MLX backend (`--features metal`). MLX unit tests require `--test-threads=1`.

---

## Quantization

| Method | Python | Rust | Unit | E2E | Pri |
|---|:---:|:---:|---:|---:|:---:|
| GGUF (Q4_0 / Q4_K / Q8_0 / etc.) | &#x1F535; | &#x1F535; | 19 | 3 | |
| MLX native 4-bit group quantization | N/A | &#x1F535; | 2 | 0 | |
| GPTQ (INT4, candle + MLX) | &#x1F535; | &#x1F535; | 14 | 4 | |
| AWQ (INT4, candle + MLX) | &#x1F535; | &#x1F535; | 12 | 4 | |
| Marlin (GPTQ-Marlin / AWQ-Marlin) | &#x1F535; | &#x1F534; | — | — | P2 |
| FP8 (FBGemm / ModelOpt) | &#x1F535; | &#x1F534; | — | — | P3 |
| BitsAndBytes NF4 (4-bit, candle + MLX) | &#x1F535; | &#x1F535; | 22 | 4 | |
| BitsAndBytes INT8 (8-bit, candle + MLX) | &#x1F535; | &#x1F535; | 15 | 3 | |
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
| `POST /v1/embeddings` | &#x1F535; | &#x1F535; | 19 | 19 | |
| `POST /v1/chat/completions` tool_calls | &#x1F535; | &#x1F535; | 3 | 0 | |
| `response_format` (JSON mode/schema) | &#x1F535; | &#x1F535; | 6 | 0 | |
| Anthropic Messages API | &#x1F535; | &#x1F534; | — | — | P2 |
| gRPC server | &#x1F535; | &#x1F534; | — | — | P1 |
| MCP tool server | &#x1F535; | &#x1F534; | — | — | P2 |
| Batch processing (`vllm batch`) | &#x1F535; | &#x1F535; | 3 | 6 | |
| Responses API | &#x1F535; | &#x1F534; | — | — | P1 |
| Speech-to-text | &#x1F535; | &#x1F534; | — | — | P1 |
| Realtime API | &#x1F535; | &#x1F534; | — | — | P1 |
| SageMaker integration | &#x1F535; | &#x1F534; | — | — | P1 |
| SSL / TLS | &#x1F535; | &#x1F535; | 5 | 0 | |
| CORS | &#x1F535; | &#x1F535; | 0 | 0 | |
| `usage` field in responses | &#x1F535; | &#x1F535; | 2 | 0 | |
| `best_of` / `n` with reranking | &#x1F535; | &#x1F534; | — | — | P0 |

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
| Prompt logprobs | &#x1F535; | &#x1F535; | 4 | 1 | |
| Logit bias | &#x1F535; | &#x1F535; | 3 | 0 | |
| Beam search | &#x1F535; | &#x1F534; | — | — | P0 |
| `best_of` | &#x1F535; | &#x1F534; | — | — | P0 |
| `max_tokens` / `max_completion_tokens` | &#x1F535; | &#x1F535; | 2 | 2 | |
| Stop strings | &#x1F535; | &#x1F535; | 3 | 0 | |
| Stop token IDs | &#x1F535; | &#x1F535; | 1 | 0 | |
| EOS detection (multi-EOS) | &#x1F535; | &#x1F535; | 1 | 0 | |
| `ignore_eos` | &#x1F535; | &#x1F535; | 1 | 0 | |
| `min_tokens` | &#x1F535; | &#x1F535; | 1 | 0 | |
| Seed (reproducible sampling) | &#x1F535; | &#x1F535; | 2 | 1 | |
| Guided decoding (grammar/regex/JSON) | &#x1F535; | &#x1F535; | 6 | 0 | |

> Unit counts drawn from `sampling.rs` (27 tests — validation, serde, types) and `sampler.rs` (24 tests — greedy, temperature, top-k/p, min-p, penalties, logprobs, prompt logprobs, grammar mask), plus 1 engine propagation test. Guided decoding unit tests count API-level resolve/conflict tests in `engine.rs`; grammar engine tests are in the [Structured Output](#structured-output--guided-decoding) section.

---

## KV Cache & Attention

| Feature | Python | Rust | Unit | E2E | Pri |
|---|:---:|:---:|---:|---:|:---:|
| Per-request KV cache | &#x1F535; | &#x1F535; | 1 | 0 | |
| Block-based KV cache (PagedAttention) | &#x1F535; | &#x1F535; | 19 | 0 | |
| KV block pool (pre-allocated) | &#x1F535; | &#x1F535; | 13 | 0 | |
| Direct block KV reads (no gather copy) | &#x1F535; | &#x1F535; | 2 | 0 | |
| Paged decode attention (per-block scoring) | &#x1F535; | &#x1F535; | 6 | 0 | |
| Prefix caching (hash-based, Candle + MLX) | &#x1F535; | &#x1F535; | 5 | 3 | |
| Automatic prefix caching | &#x1F535; | &#x1F535; | 2 | 3 | |
| Chunked prefill | &#x1F535; | &#x1F535; | 1 | 0 | |
| KV cache compression (latent caching) | &#x1F535; | &#x1F534; | — | — | P2 |
| KV cache offloading (CPU ↔ GPU) | &#x1F535; | &#x1F534; | — | — | P3 |
| KV cache transfer (distributed) | &#x1F535; | &#x1F534; | — | — | P1 |
| Multi-group KV cache (hybrid models) | &#x1F535; | &#x1F534; | — | — | P1 |
| FlashAttention v2 (single-seq + batched varlen + paged decode) | &#x1F535; | &#x1F535; | 15 | 0 | |
| FlashInfer | &#x1F535; | &#x1F534; | — | — | P2 |
| FlexAttention | &#x1F535; | &#x1F534; | — | — | P1 |
| xFormers | &#x1F535; | &#x1F534; | — | — | P0 |
| MLA (Multi-head Latent Attention) | &#x1F535; | &#x1F535; | 4 | 0 | |
| Sliding window attention | &#x1F535; | &#x1F535; | 13 | 0 | |
| Batched attention metadata (cu_seqlens, slot_mapping, block_table) | &#x1F535; | &#x1F535; | 9 | 0 | |
| Tree attention | &#x1F535; | &#x1F534; | — | — | P1 |

> Unit counts: `block_pool.rs` (19), `free_block_queue.rs` (16), `kv_cache_manager.rs` (13), `kv_cache_block.rs` (11), `kv_block_pool.rs` (13), `attention.rs` (34 — 25 SDPA/paged + 9 FlashAttention v2), MLX `cache.rs` (3 incl. truncate). FlashAttention v2 tests: decode BF16/F16, prefill BF16/F16, GQA BF16 decode/prefill, head_dim=128, sliding window, attention_with_cache dispatch — all compare FA2 CUDA output against CPU SDPA reference. Sliding window: 7 attention.rs + 2 gemma2.rs interleaved + 2 qwen2.rs max_window_layers + 1 MLX phi3 trim + 1 array-format parsing = 13. E2E prefix caching: `e_prefix_caching.rs` — 2 CPU (CandleWorker) + 1 MLX (Metal). CandleWorker uses paged KvBlockPool; MLX uses worker-level `HashMap<u64, MlxKvCache>` pool with COW cloning. Per-row counts reflect the primary feature each test targets; some tests cross-cut multiple rows. Total section: 114 unit tests.
>
> **Batched attention metadata note:** The Rust port has `AttentionMetadata` (with `query_start_loc`, `seq_lens`, `block_ids`, `tokens_before`, cached `block_table_gpu`, `decode_slot_mapping_gpu` per request) used by `forward_batch()`. On CUDA, batched FA2 uses `flash_attn_varlen` for prefill/mixed batches and `flash_attn_varlen_paged` for all-decode batches — the paged path reads K/V directly from the block pool via `block_table`, eliminating per-request gathers and `Tensor::cat`. This matches Python vLLM's paged FlashAttention decode path.
> - **CUDA**: FlashAttention varlen (`flash_attn_varlen`) for prefill/mixed + paged FA2 (`flash_attn_varlen_paged`) for decode — uses `cu_seqlens` and `block_table` to handle ragged sequences without padding or gather.
> - **MLX**: Two approaches: (a) **Padded-batch SDPA** — left-pad inputs to uniform KV length, use `BatchKVCache` with per-sequence padding offsets, construct padding-aware causal masks (proven by mlx-lm's `BatchGenerator`; wastes compute on pad tokens but works with stock MLX SDPA). Effort: ~1-2 weeks, low risk. (b) **Custom Metal PagedAttention kernels** — block-table-based paged attention Metal shaders; no padding, higher throughput, more implementation effort. Effort: ~4-6 weeks production-quality, medium risk.
>   - **Existing implementations**: The same core kernel (by the mistral.rs author) lives in two repos: `mistralrs-paged-attn/src/metal/` (~2,100 lines Metal shader + ~1,070 lines Rust dispatch via candle `CustomOp1`) and HF `kernels-community/paged-attention` (same shader, Obj-C++ dispatch). The kernel handles per-block QK scoring, cross-warp softmax reduction via `simd_shuffle_xor`, V accumulation, partitioned attention V2 for long sequences, GQA, FP8 cache, soft-capping, ALiBi. Templated across head sizes (64/80/96/128/192/256) and block sizes (8/16/32). HF also has a separate `kernels-community/metal-flash-sdpa` (~2,100 lines Metal) for varlen prefill — the complementary kernel.
>   - **Integration challenge**: MLX's `metal_kernel` API (bindings exist in mlx-sys but aren't wrapped in mlx-rs) auto-generates kernel signatures from inputs/outputs. The paged attention kernel uses function constants, explicit threadgroup memory allocation, and 19 buffers — features the `metal_kernel` convenience API doesn't support. Options: restructure the kernel to fit (losing some optimizations), contribute threadgroup memory support upstream to MLX, or go raw Metal via the `metal` crate (requires sharing `MTLBuffer` pointers between MLX's command queue and a separate dispatch — fragile, since mlx-c doesn't expose raw buffer pointers).
>   - **Recommendation**: Padded-batch SDPA first (low effort, proven pattern), custom Metal second only if MLX upstream doesn't add native paged attention (tracking MLX issues #2228, #2955).
> - **Impact (downgraded from P4 to P3)**: On MLX, the current batched-forward-with-per-request-attention already yields ~33% throughput gain at n=4. MLX lazy eval already implicitly batches the N separate SDPA kernel dispatches into a single `eval()` / Metal command buffer, so the kernel-launch-overhead savings are already captured. The remaining gap is GPU occupancy within the attention shader itself — a modest incremental win (est. 10-20% ITL reduction at n=4-8). The padded-batch SDPA approach also introduces padding waste when requests have different sequence lengths, partially offsetting gains. TTFT is unaffected (prefill is projection/MLP-dominated and rarely overlaps). The high-value path is CUDA FlashAttention varlen (ragged sequences, no padding), but that requires C FFI — a larger lift gated on CUDA backend maturity.

---

## Scheduling

| Feature | Python | Rust | Unit | E2E | Pri |
|---|:---:|:---:|---:|---:|:---:|
| Iteration-level scheduling (add/remove per step) | &#x1F535; | &#x1F535; | 6 | 0 | |
| Batched forward pass (cross-request token concat) | &#x1F535; | &#x1F535; | 7 | 0 | |
| FCFS request queue | &#x1F535; | &#x1F535; | 7 | 0 | |
| Priority request queue | &#x1F535; | &#x1F535; | 7 | 0 | |
| Preemption | &#x1F535; | &#x1F535; | 3 | 0 | |
| Chunked prefill scheduling | &#x1F535; | &#x1F535; | 2 | 0 | |
| Block allocation / eviction | &#x1F535; | &#x1F535; | 19 | 0 | |
| Prefix cache hits | &#x1F535; | &#x1F535; | 5 | 0 | |
| Pause / resume | &#x1F535; | &#x1F535; | 3 | 0 | |
| Multi-step scheduling (`--num-scheduler-steps`) | &#x1F535; | &#x1F534; | — | — | P0 |
| Async scheduler | &#x1F535; | &#x1F535; | 6 | 1 | |

> **Continuous batching note:** Python vLLM's "continuous batching" combines two things: (1) iteration-level scheduling — the scheduler can add/remove requests at each step, and (2) batched model execution — all scheduled requests' tokens are concatenated into a single flat 1D `input_ids` tensor and processed in one `model.forward()` call, with per-request boundaries tracked via attention metadata. The Rust port now implements both: `CandleWorker` (paged KV path) concatenates all requests' tokens into flat `[total_tokens]` tensors, builds `AttentionMetadata` with per-request slicing info, and calls `model.forward_batch()` — a single pass that batches embedding, projections, norms, and MLP across all requests while running attention per-request via `BatchedKvCacheStorage`. `MlxWorker` defers `eval()` across all per-request forward passes, enabling MLX graph fusion into a single Metal command buffer. `LlamaForCausalLM` provides a real batched implementation (covering LLaMA, Mistral, Qwen2, Qwen3, Phi-3); other architectures fall back to the default per-request loop. The remaining gap vs Python vLLM is batched attention kernels (FlashAttention varlen / FlashInfer) — the Rust port still runs attention per-request within the batched forward.
>
> Unit counts: `scheduler/core.rs` (30), `scheduler/request_queue.rs` (14), `scheduler/output.rs` (5), `scheduler/interface.rs` (2), `request.rs` (16). Batched forward: `attention_metadata.rs` (3), `llama.rs` forward_batch equivalence (2), `candle_worker.rs` paged multi-request (2). Block allocation count includes `block_pool.rs` and `kv_cache_manager.rs` allocate/free/eviction tests counted above in KV Cache; per-row counts here reflect scheduler-specific tests.
>
> **Async scheduler note:** Enabled by default (matching Python vLLM V1). The executor runs on a dedicated OS thread; the step loop schedules the next batch while the GPU executes the current one — overlapping CPU scheduling with GPU execution. Channel-based communication (`sync_channel(1)`) provides backpressure. Disable with `--disable-async-scheduling` CLI flag. Unit tests: `engine_core.rs` (6 — take_executor, step_errors_after_take, schedule_next empty/with_requests, finalize_step, shutdown_after_take). E2E: all 96 E2E tests exercise the async path; 1 explicit sync-path smoke test (`test_sync_scheduling_smollm_chat`).
>
> **Multi-step scheduling note:** Python vLLM's `--num-scheduler-steps` (default 1) was a V0 engine feature that ran N decode steps per scheduler call to amortize scheduling overhead. The V1 engine (default since vLLM 0.8.x) deprecated this flag — V1's async scheduler and persistent `InputBatch` achieve the same throughput gains without multi-step complexity. P0 — not worth implementing.

---

## Hardware Backends

| Backend | Python | Rust | Unit | E2E | Pri |
|---|:---:|:---:|---:|---:|:---:|
| CPU | &#x1F535; | &#x1F535; | 4 | 0 | |
| CUDA (NVIDIA GPU) | &#x1F535; | &#x1F7E1; | 47 | 8 | P3 |
| Metal / MLX (Apple Silicon) | &#x1F534; | &#x1F535; | 4 | 0 | |
| ROCm (AMD GPU) | &#x1F535; | &#x1F534; | — | — | P2 |
| TPU | &#x1F535; | &#x1F534; | — | — | P1 |
| XPU (Intel) | &#x1F535; | &#x1F534; | — | — | P1 |
| AWS Neuron | &#x1F535; | &#x1F534; | — | — | P1 |
| Device auto-detection | &#x1F535; | &#x1F535; | 4 | 0 | |
| Memory profiling / `--gpu-memory-utilization` | &#x1F535; | &#x1F535; | 6 | 2 | |

> Unit counts from `candle_worker.rs` (24 total) and `mlx_worker.rs` (4). CUDA: 29 GPU kernel unit tests (norm 13 incl. fused_add_rms_norm, activation 7, rotary 5, cache 4) + 5 MoE kernel tests (topk_softmax pow2/non-pow2/f32, moe_sum f32/bf16) + 9 FlashAttention v2 tests + 2 device detection tests + 2 misc = 47. E2E: 5 CUDA safetensors tests (SmolLM-135M + Qwen2.5-0.5B) + 3 CUDA GGUF tests (Gemma3-1B Q4_K_M) = 8. MoE E2E tests need ≥80GB GPU (smallest safetensors MoE models are 14B+ params). The CUDA backend supports E2E inference (verified: Qwen2.5-0.5B BF16 on L40S) with 8 fused CUDA kernels (RMSNorm, fused add+RMSNorm, SiLU+mul/GELU+mul, RoPE, reshape_and_cache, QK-norm+RoPE, MoE topk_softmax, MoE moe_sum) plus FlashAttention v2 (via `candle-flash-attn` crate). MoE models (Qwen3MoE, Qwen2MoE, DeepSeekV2) use GPU-accelerated top-k softmax gating and per-expert batched cuBLAS GEMMs instead of the per-token CPU loop. GPU KV block pool, VRAM-based block allocation, and GPU↔CPU block swapping. Remaining for full parity: CUDA graphs, tensor parallelism. Device auto-detection includes `parse_device` tests for cpu/cuda/metal/auto. E2E float16 tests validate dtype selection end-to-end.

---

## Parallelism & Distribution

| Feature | Python | Rust | Unit | E2E | Pri |
|---|:---:|:---:|---:|---:|:---:|
| Parallel config types (TP/PP groups) | &#x1F535; | &#x1F535; | 10 | 0 | |
| UniProc executor (single-process) | &#x1F535; | &#x1F535; | 10 | 0 | |
| MultiProc executor (multi-worker) | &#x1F535; | &#x1F535; | 6 | 0 | |
| Tensor parallelism (weight sharding + multi-GPU init) | &#x1F535; | &#x1F7E1; | 3 | 1 | P3 |
| Pipeline parallelism | &#x1F535; | &#x1F534; | — | — | P2 |
| NCCL communication (bindings + ProcessGroup trait) | &#x1F535; | &#x1F7E1; | 3 | 0 | P3 |
| Ray distributed executor | &#x1F535; | &#x1F534; | — | — | P1 |
| Expert parallelism (MoE) | &#x1F535; | &#x1F534; | — | — | P2 |
| Data parallelism | &#x1F535; | &#x1F534; | — | — | P2 |
| Weight transfer / migration | &#x1F535; | &#x1F534; | — | — | P1 |

> Unit counts: `parallel.rs` (10), `uniproc.rs` (10), `multiproc.rs` (6), `worker.rs` (7 — NoopWorker lifecycle), `nccl.rs` (3 — NCCL all-reduce/all-gather on 2x L40S). E2E: `test_cuda_tp2_qwen2_completion` (1 — TP=2 Qwen2.5-0.5B on 2x L40S).

---

## Performance Optimizations

| Feature | Python | Rust | Unit | E2E | Pri |
|---|:---:|:---:|---:|---:|:---:|
| CUDA graphs | &#x1F535; | &#x1F7E1; | 5 | 0 | P3 | Infrastructure done; blocked on candle default-stream capture (candle#3083) |
| FlashAttention v2 kernels (single-seq + batched varlen) | &#x1F535; | &#x1F535; | 15 | 0 | |
| FlashInfer kernels | &#x1F535; | &#x1F534; | — | — | P2 |
| xFormers memory-efficient attention | &#x1F535; | &#x1F534; | — | — | P1 |
| Fused SiLU-and-mul kernel | &#x1F535; | &#x1F535; | 7 | 0 | |
| Fused RMSNorm kernel (+ fused add+RMSNorm) | &#x1F535; | &#x1F535; | 13 | 0 | |
| Fused RoPE kernel | &#x1F535; | &#x1F535; | 5 | 0 | |
| Custom all-reduce kernel | &#x1F535; | &#x1F534; | — | — | P1 |
| MoE fused routing kernels (topk softmax + moe_sum) | &#x1F535; | &#x1F535; | 5 | 0 | |
| MLX lazy eval graph fusion | N/A | &#x1F535; | 0 | 0 | |
| MLX single-eval sampling fusion | N/A | &#x1F535; | 0 | 0 | |
| MLX cross-request deferred eval | N/A | &#x1F535; | 0 | 0 | |
| Pre-transposed weights (Metal) | N/A | &#x1F535; | 0 | 0 | |
| Native dtype inference (`--dtype auto`) | &#x1F535; | &#x1F535; | 4 | 2 | |
| Mixed prefill+decode in single forward pass | &#x1F535; | &#x1F535; | 0 | 0 | |
| Persistent InputBatch (cross-iteration reuse) | &#x1F535; | &#x1F535; | 14 | 0 | |
| Paged KV (no gather copy on decode) | &#x1F535; | &#x1F535; | 2 | 0 | |

> Fused CUDA kernels: `vllm-kernels/csrc/` contains 7 custom CUDA kernel files compiled via nvcc (SM80/86/89/90). Each kernel has CPU and CUDA implementations behind the `KernelSet` trait, with `ops.rs` auto-dispatch wiring all model architectures to use fused kernels when on CUDA. `fused_add_rms_norm` saves 1 kernel launch + 1 tensor allocation per decoder layer (called in every model architecture). MoE routing kernels (`moe_topk_kernels.cu`, `moe_align_kernels.cu`) adapted from Python vLLM's `csrc/moe/` — `topk_softmax` uses warp-level fused softmax+argmax for power-of-2 expert counts, CUB BlockReduce fallback for arbitrary counts; `moe_sum` reduces expert outputs via template-specialized topk unrolling. MoE models (Qwen3MoE, Qwen2MoE, DeepSeekV2) use GPU gating + per-expert batched cuBLAS GEMMs. 34 GPU unit tests compare CUDA output against CPU reference across f32/f16/bf16 (norm 13, activation 7, rotary 5, cache 4, MoE 5). FlashAttention v2 via `candle-flash-attn` crate (0.9.2): auto-dispatches on CUDA F16/BF16 in `attention_with_cache()`. 9 FA2 unit tests. CPU kernel stubs (15 tests: rotary 3, activation 3, norm 2, cache 2, attention 2, MoE 3) provide CPU fallback paths. Native dtype unit tests count `candle_worker.rs` dtype parsing tests. Persistent InputBatch: 14 unit tests in `input_batch.rs`.
>
> **Persistent InputBatch note:** `CandleWorker` now maintains a persistent `InputBatch` struct across engine steps, matching Python vLLM V1's `InputBatch`. Per-request state (block tables, tokens-in-pool, positions, last token ID) lives in dense slot arrays that are delta-updated each step. Finished requests are swap-removed to keep the array compact. `prepare_inputs()` builds flat token/position tensors and `AttentionMetadata` from the slot arrays without HashMap lookups. This eliminates per-step allocation overhead on the decode hot path — the steady-state case where N concurrent requests each generate 1 token per step.

---

## GPU Compute Kernels (Triton Equivalents)

> Python vLLM contains **72+ Triton kernel files** (`@triton.jit`) implementing GPU-optimized operations for NVIDIA hardware. Triton is a Python→PTX compiler — it cannot be used from Rust. The Rust port uses different strategies per backend:
> - **CUDA**: Custom CUDA kernels via `cudarc`, or FFI bindings to C++ libraries (FlashAttention, FlashInfer, CUTLASS)
> - **MLX**: Framework-provided Metal kernels + lazy eval graph fusion covers attention, norm, RoPE, and activation implicitly
> - **CPU**: Many operations (sampling, penalties, logprobs) are trivially fast on CPU for 1D per-request logit vectors
>
> Rows marked ✱ overlap with items in [Performance Optimizations](#performance-optimizations), [KV Cache & Attention](#kv-cache--attention), or [LoRA / Adapters](#lora--adapters) and are included here for a complete kernel-level view. Test counts are attributed to their primary sections to avoid double-counting.

| Kernel Category | Python | Rust | Unit | E2E | Pri |
|---|:---:|:---:|---:|---:|:---:|
| Triton attention (prefill / decode / unified) ✱ | &#x1F535; | &#x1F534; | — | — | P3 |
| Merge attention states | &#x1F535; | &#x1F534; | — | — | P2 |
| KV cache write (reshape_and_cache) | &#x1F535; | &#x1F535; | 4 | 0 | |
| GPU sampling (top-k / top-p / penalties / logprobs) | &#x1F535; | &#x1F535; | 0 | 0 | |
| Fused MoE routing + expert matmul ✱ | &#x1F535; | &#x1F7E1; | 5 | 0 | P2 |
| Fused layer ops (activation / norm / RoPE / attention) ✱ | &#x1F535; | &#x1F535; | 34 | 0 | |
| Quantization compute (FP8 / INT8 / AWQ matmul) | &#x1F535; | &#x1F534; | — | — | P2 |
| Mamba / SSM ops (selective scan, SSD, conv1d) | &#x1F535; | &#x1F534; | — | — | P2 |
| FLA ops (fused recurrent, KDA, chunk) | &#x1F535; | &#x1F534; | — | — | P3 |
| LoRA kernels (expand / shrink, fused_moe_lora) ✱ | &#x1F535; | &#x1F534; | — | — | P3 |
| Speculative decoding Triton ops ✱ | &#x1F535; | &#x1F534; | — | — | P2 |
| GPU batch utilities (block_table, buffer, input_batch) | &#x1F535; | &#x1F534; | — | — | P3 |

> **72 Triton files breakdown**: attention ops (6), sampling (8), fused MoE (6), quantization compute (5), Mamba/SSM (7), FLA/linear attention (11), LoRA (5), speculative decoding (2), model-level/misc (22+).
>
> **Rust strategies by category**:
> - *KV cache write*: CUDA path uses a fused `reshape_and_cache` kernel (`csrc/cache_kernels.cu`) that scatters all tokens in a single kernel launch via slot_mapping. CPU path uses `KvBlockPool::scatter_new_kv()` per-token loop. MLX uses `MlxKvCache`. 4 CUDA unit tests verify scatter correctness (basic, f16, padding skip, single-token).
> - *GPU sampling*: All sampling runs on CPU in both CandleWorker and MlxWorker. For per-request forward passes this is trivially fast (~µs for a 1D logits vector). GPU sampling kernels only matter for batched inference where logits are a 2D `[batch, vocab]` tensor. Tests attributed to [Sampling & Decoding](#sampling--decoding).
> - *Fused layer ops*: CUDA path has fused kernels for RMSNorm + fused add+RMSNorm (`csrc/layernorm_kernels.cu`), SiLU+mul / GELU+mul (`csrc/activation_kernels.cu`), RoPE (`csrc/pos_encoding_kernels.cu`), fused QK-norm+RoPE (`csrc/qk_norm_rope_kernels.cu`, used by Gemma3), and FlashAttention v2 (`candle-flash-attn` crate). All custom kernels use vectorized 128-bit loads via `vec_utils.cuh`. `ops.rs` auto-dispatches to CUDA when tensors are on GPU, falling back to candle ops on CPU. `attention_with_cache()` auto-dispatches to FA2 on CUDA F16/BF16. All model decoder layers use `fused_add_rms_norm` to merge the residual add + post-attention norm into a single kernel launch. MLX lazy eval fuses the same operations into single Metal command buffers. 34 CUDA unit tests (norm 13 + activation 7 + rotary 5 + FlashAttention 9) compare against CPU reference across f32/f16/bf16.
> - *Fused MoE routing*: CUDA `topk_softmax` kernel (`csrc/moe_topk_kernels.cu`, adapted from Python vLLM's `csrc/moe/topk_softmax_kernels.cu`) fuses softmax + top-k selection into a single kernel with warp-level butterfly reduction for power-of-2 expert counts (1–512) and CUB BlockReduce fallback for arbitrary counts. `moe_sum` kernel (`csrc/moe_align_kernels.cu`) reduces `[tokens, topk, hidden]` → `[tokens, hidden]` with template-specialized topk unrolling. Expert GEMMs use per-expert batched cuBLAS matmul (tokens grouped by expert assignment, one matmul per active expert) — simpler than Python vLLM's Triton fused_moe_kernel but effective for small active expert counts (2–8). MoE models (Qwen3MoE, Qwen2MoE, DeepSeekV2MoE) auto-dispatch to GPU gating when on CUDA. 5 CUDA unit tests. E2E tests require ≥80GB GPU (smallest safetensors MoE models are 14B+ total params, ~31GB BF16). Future: custom tiled GEMM or cuBLAS batched GEMM for higher throughput.
> - *Triton attention*: Python vLLM has its own Triton attention implementations (distinct from the FlashAttention C++ library). Both serve the same purpose: batched variable-length attention. The Rust port uses FlashAttention v2 via `candle-flash-attn` crate — auto-dispatches in `attention_with_cache()` on CUDA F16/BF16. The Triton attention row above tracks the batched varlen Triton kernels specifically (distinct from FA2).
> - *Model-gated kernels*: Mamba/SSM, FLA, LoRA, and speculative decoding kernels are only needed when those model types or features are implemented — they are blocked by their parent feature.

---

## LoRA / Adapters

| Feature | Python | Rust | Unit | E2E | Pri |
|---|:---:|:---:|---:|---:|:---:|
| LoRA adapter loading | &#x1F535; | &#x1F535; | 8 | 4 | P3 |
| Multi-LoRA serving | &#x1F535; | &#x1F534; | — | — | P2 |
| LoRA weight merging (single adapter) | &#x1F535; | &#x1F535; | 4 | 4 | P2 |
| Punica kernels | &#x1F535; | &#x1F534; | — | — | P0 |
| Dynamic adapter switching | &#x1F535; | &#x1F534; | — | — | P2 |

---

## Speculative Decoding

| Feature | Python | Rust | Unit | E2E | Pri |
|---|:---:|:---:|---:|---:|:---:|
| Draft model (MLP speculator) | &#x1F535; | &#x1F534; | — | — | P2 |
| Eagle speculative decoding | &#x1F535; | &#x1F534; | — | — | P2 |
| Medusa heads | &#x1F535; | &#x1F534; | — | — | P2 |
| N-gram proposer | &#x1F535; | &#x1F535; | 20 | 0 | |
| Suffix decoding | &#x1F535; | &#x1F534; | — | — | P1 |

> Unit counts from `ngram.rs` (16 — proposer algorithm: empty/single/no-match, bigram/trigram/unigram matching, n-gram size priority, most-recent-match preference, max-token limits, code-like patterns, boundary cases) and `engine_core.rs` (4 — proposer creation, disabled-by-default, propose-after-step, cleared-after-schedule). N-gram proposer scans the request's own token history for matching n-grams and proposes continuation tokens; the target model verifies drafts in a single multi-token forward pass with greedy acceptance. CLI: `--speculative-model ngram --num-speculative-tokens K --ngram-prompt-lookup-max N --ngram-prompt-lookup-min N`.

---

## Multimodal / Vision-Language

| Feature | Python | Rust | Unit | E2E | Pri |
|---|:---:|:---:|---:|---:|:---:|
| Image input processing (base64 data URI) | &#x1F535; | &#x1F535; | 0 | 5 | |
| LLaVA | &#x1F535; | &#x1F534; | — | — | P2 |
| Qwen2-VL / Qwen2.5-VL (vision encoder + Qwen2 LM) | &#x1F535; | &#x1F535; | 8 | 5 | |
| Pixtral | &#x1F535; | &#x1F534; | — | — | P2 |
| InternVL | &#x1F535; | &#x1F534; | — | — | P2 |
| Phi-3V / Phi-4MM | &#x1F535; | &#x1F534; | — | — | P2 |
| Gemma 3 multimodal (SigLIP vision + projector) | &#x1F535; | &#x1F535; | 4 | 8 | |
| Molmo | &#x1F535; | &#x1F534; | — | — | P1 |
| PaliGemma | &#x1F535; | &#x1F534; | — | — | P1 |
| Audio models (Whisper, Qwen-Audio) | &#x1F535; | &#x1F534; | — | — | P1 |

> **Gemma 3 VLM implementation**: Full `Gemma3ForConditionalGeneration` support on both Candle (CPU/CUDA) and MLX (Metal) backends, including quantized MLX models (4-bit language model with float vision tower). Architecture: SigLIP vision encoder → AvgPool2d → GemmaRMSNorm → projection → merge with text embeddings → Gemma3 language model. Image input via OpenAI-compatible base64 data URI in chat messages. Unit tests: 2 weight-name validation (against real HF checkpoints), 1 projector shape, 1 config parsing. E2E: 3 Candle (server start, text-only chat, max_tokens) + 5 MLX (server start, text-only, image chat, image stream, image max_tokens).

> **Qwen2-VL / Qwen2.5-VL implementation**: Full `Qwen2VLForConditionalGeneration` and `Qwen2_5_VLForConditionalGeneration` support on both Candle and MLX backends. Architecture: custom ViT with 3D patch embedding (Conv3d-as-Linear) + 2D RoPE → PatchMerger (2x2 spatial merge + GELU MLP) → Qwen2 LLM backbone. Qwen2.5-VL variant uses RMSNorm + SwiGLU MLP in the vision encoder instead of LayerNorm + QuickGELU. M-RoPE support added to RotaryEmbedding for 3-section position encoding. CLIP normalization for image preprocessing with smart_resize. Quantized MLX models dequantize vision encoder weights at load time. Unit tests: 5 config/preprocessing + 3 M-RoPE. E2E (MLX): server start, text-only chat, image chat, image stream, image max_tokens.

---

## Structured Output / Guided Decoding

| Feature | Python | Rust | Unit | E2E | Pri |
|---|:---:|:---:|---:|---:|:---:|
| `response_format: json_object` | &#x1F535; | &#x1F535; | 2 | 0 | |
| `response_format: json_schema` | &#x1F535; | &#x1F535; | 3 | 0 | |
| Grammar-guided logit masking | &#x1F535; | &#x1F535; | 5 | 0 | |
| Regex-constrained decoding | &#x1F535; | &#x1F535; | 2 | 0 | |

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
| `/v1/embeddings` endpoint | &#x1F535; | &#x1F535; | 19 | 10 | |
| Pooling strategies (last, CLS, mean) | &#x1F535; | &#x1F535; | 19 | 10 | |
| Auto-detect pooling from `1_Pooling/config.json` | &#x1F535; | &#x1F535; | 5 | 0 | |
| `--pooling-strategy` CLI flag (auto/last/cls/mean) | &#x1F535; | &#x1F535; | 1 | 3 | |
| Matryoshka dimension truncation | &#x1F535; | &#x1F535; | 2 | 1 | |
| Pooling execution mode (`--runner pooling`) | &#x1F535; | &#x1F535; | 12 | 9 | |
| Encoder-only models (BERT, ModernBERT) | &#x1F535; | &#x1F534; | — | — | P2 |
| Reward / reranking models | &#x1F535; | &#x1F534; | — | — | P1 |

> **Implementation notes:**
>
> - **Decoder-as-embedder works end-to-end:** `/v1/embeddings` runs a single `hidden_states()` forward pass (no KV cache, no decode loop), pools, L2-normalizes, and returns OpenAI-compatible responses. Tested with LLaMA, Qwen2, SmolLM on both Candle and MLX backends.
> - **Three pooling strategies:** `Last` (default for decoder models like e5-mistral, gte-Qwen2), `Cls` (first token, for encoder models), `Mean` (average all tokens, most common for BERT-family). Strategy is resolved at model-load time: explicit `--pooling-strategy` > auto-detect from `1_Pooling/config.json` > default to `Last`.
> - **Auto-detection:** sentence-transformers models publish `1_Pooling/config.json` with boolean fields (`pooling_mode_mean_tokens`, `pooling_mode_cls_token`, `pooling_mode_lasttoken`). Both Candle and MLX workers download this file from HF Hub and detect the strategy automatically.
> - **`--runner pooling` is fully implemented:** Embedding requests flow through the scheduler like generation requests. The worker calls `hidden_states()` + pool + normalize (no decode loop). Requests finish after one forward pass. Generation endpoints (`/v1/chat/completions`, `/v1/completions`) return 400 in pooling mode. 12 unit tests + 9 E2E tests (SmolLM-135M on MLX).
> - **Remaining gap is encoder-only models:** BERT, NomicBERT, ModernBERT need bidirectional attention (remove causal mask) — a fundamentally different attention mode from the current decoder-only pipeline.

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
| `vllm batch -i <in> -o <out>` | &#x1F535; | &#x1F535; | 5 | 6 | |
| `vllm convert` | &#x1F535; | &#x1F535; | 1 | 0 | |
| HF Hub model download | &#x1F535; | &#x1F535; | 0 | 0 | |
| Sharded weight loading | &#x1F535; | &#x1F535; | 0 | 0 | |
| `--model` / positional model arg | &#x1F535; | &#x1F535; | 3 | 0 | |
| `--dtype auto` / explicit dtype | &#x1F535; | &#x1F535; | 0 | 0 | |
| `--device auto` / explicit device | &#x1F535; | &#x1F535; | 0 | 0 | |
| `--gpu-memory-utilization` | &#x1F535; | &#x1F535; | 0 | 0 | |
| `--gguf-file` | &#x1F535; | &#x1F535; | 0 | 0 | |
| `--tool-call-parser` | &#x1F535; | &#x1F535; | 0 | 0 | |
| `--speculative-model ngram` | &#x1F535; | &#x1F535; | 0 | 0 | |
| `--runner [generate\|pooling]` | &#x1F535; | &#x1F535; | 2 | 0 | |
| `--features metal` (MLX backend) | N/A | &#x1F535; | 0 | 0 | |
| Dockerfile.cpu | &#x1F535; | &#x1F535; | 0 | 0 | |
| Dockerfile.cuda | &#x1F535; | &#x1F535; | 0 | 0 | |
| `VLLM_MODEL` env var | &#x1F535; | &#x1F535; | 0 | 0 | |
| PyO3 bridge (Rust scheduler in Python) | &#x1F535; | &#x1F7E1; | 0 | 0 | P1 |
| Python fallback for unported models | &#x1F535; | &#x1F534; | — | — | P1 |
| Standalone binary (no Python runtime) | &#x1F534; | &#x1F535; | 0 | 0 | |

> Unit counts from `args.rs` (10 — CLI parsing: help, serve/bench/batch positional/flag model, precedence, no-model error, convert) and `init.rs` (6 — extract model name, compute num blocks variants). Batch runner: `batch.rs` (3 — JSONL parse/write).

---

## Cargo Feature Flags

> The Rust port gates heavy optional dependencies behind Cargo feature flags. All flags default to **on** so `cargo build` works identically to a build with every subsystem. Use `--no-default-features` for minimal builds, then opt in to individual features as needed. Feature flags propagate through the crate dependency chain: `vllm-cli` → `vllm-serve` → leaf crates.

| Flag | Default | Dependencies gated | Crates affected |
|---|:---:|---|---|
| `guided-decoding` | yes | `outlines-core` | vllm-models, vllm-executor, vllm-mlx, vllm-serve, vllm-cli |
| `multimodal` | yes | `image` | vllm-model, vllm-serve, vllm-cli |
| `chat-template` | yes | `minijinja`, `minijinja-contrib` | vllm-serve, vllm-cli |
| `tls` | yes | `axum-server`, `rustls`, `rustls-pemfile` | vllm-serve, vllm-cli |
| `metrics` | yes | `prometheus` | vllm-serve, vllm-cli |
| `multiproc` | **no** | `zeromq` | vllm-protocol, vllm-engine, vllm-serve, vllm-cli |

### Build profiles

```sh
# Full build (all defaults on — identical to pre-feature-flags behavior)
cargo build -p vllm-cli

# Minimal build (no optional subsystems)
cargo build -p vllm-cli --no-default-features

# Minimal + single feature
cargo build -p vllm-cli --no-default-features --features guided-decoding

# CUDA without optional deps
cargo build -p vllm-cli --no-default-features --features cuda

# Metal without optional deps
cargo build -p vllm-cli --no-default-features --features metal
```

> `multiproc` is excluded from the default set because most users use `UniProcExecutor` (single-process). The `zeromq` crate adds significant compile time and a native dependency. Enable it explicitly with `--features multiproc` when using the multi-process executor.
>
> Hardware backend flags (`cuda`, `metal`, `candle-metal`) are orthogonal to the optional subsystem flags and can be combined freely.

---

---

## Stats

| Metric | Python | Rust |
|---|---|---|
| Model architectures | ~248 | 12 candle + 11 MLX (+ quantized variants) |
| Quantization methods | ~14 | 6 (GGUF + MLX native 4-bit + GPTQ INT4 + AWQ INT4 + BnB NF4 + BnB INT8) |
| Attention backends | ~15 | 3 (custom SDPA + FlashAttention v2 single-seq/varlen + paged FA2 decode) |
| Hardware backends | 6 (CUDA, ROCm, CPU, TPU, XPU, Neuron) | 3 (CPU, CUDA, Metal/MLX) |
| Lines of code | ~507K Python + ~89K C++/CUDA | ~30.7K Rust |
| Unit tests | ~948 test files | 910 passing (849 non-MLX + 61 MLX) |
| E2E tests | — | 120 passing (37 basic serving + 22 chat/sampling + 8 streaming + 5 tool parser + 10 embedding + 4 GPTQ + 4 AWQ + 7 BnB + 6 LLM API + 4 LoRA + 6 batch + 7 multimodal) |
| Crate count | N/A | 14 crates (incl. vllm-e2e) |
