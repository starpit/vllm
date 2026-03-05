# vLLM Feature Parity: Python vs Rust

> Last updated: 2026-03-05

| Symbol | Meaning | Count |
|--------|---------|------:|
| ✅ | Implemented | 127 |
| ⚠️ | Partial | 1 |
| ❌ | Not implemented | 136 |
| ➕ | Rust-only | 13 |

---

## Summary

> Counts are for Rust parity against Python features. **Rust-only** = features unique to the Rust port.

| Section | Rust ✅ | Rust ⚠️ | Rust ❌ | Rust-only |
|---|---:|---:|---:|---:|
| [Hardware Platforms](#hardware-platforms) | 3 | 0 | 6 | 1 |
| [Multi-GPU & Distribution](#multi-gpu-and-distribution) | 1 | 1 | 7 | 0 |
| [CLI Commands](#cli-commands) | 10 | 0 | 0 | 1 |
| [OpenAI-Compatible API Endpoints](#openai-compatible-api-endpoints) | 7 | 0 | 6 | 0 |
| [Other API Protocols](#other-api-protocols) | 0 | 0 | 7 | 3 |
| [Model Architectures — Decoder-Only LLMs](#model-architectures-decoder-only-llms) | 15 | 0 | 20 | 0 |
| [Model Architectures — Encoder / Embedding](#model-architectures-encoder-embedding) | 0 | 0 | 4 | 0 |
| [Model Architectures — Vision-Language / Multimodal](#model-architectures-vision-language-multimodal) | 2 | 0 | 9 | 0 |
| [Model Architectures — Audio / Speech](#model-architectures-audio-speech) | 0 | 0 | 4 | 0 |
| [Model Architectures — Speculative Decoding Draft Models](#model-architectures-speculative-decoding-draft-models) | 0 | 0 | 4 | 0 |
| [Quantization Methods](#quantization-methods) | 4 | 0 | 8 | 1 |
| [Attention Backends](#attention-backends) | 5 | 0 | 8 | 1 |
| [Sampling & Decoding](#sampling-and-decoding) | 19 | 0 | 3 | 0 |
| [Structured Output / Guided Decoding](#structured-output-guided-decoding) | 5 | 0 | 5 | 0 |
| [Tool Calling / Function Calling](#tool-calling-function-calling) | 6 | 0 | 2 | 0 |
| [Scheduling](#scheduling) | 6 | 0 | 1 | 0 |
| [KV Cache](#kv-cache) | 4 | 0 | 4 | 0 |
| [LoRA & Adapters](#lora-and-adapters) | 2 | 0 | 4 | 0 |
| [Speculative Decoding](#speculative-decoding) | 1 | 0 | 5 | 0 |
| [Multimodal Input](#multimodal-input) | 1 | 0 | 4 | 0 |
| [Embeddings & Pooling](#embeddings-and-pooling) | 4 | 0 | 3 | 1 |
| [Serving Features](#serving-features) | 6 | 0 | 7 | 1 |
| [Performance Optimizations](#performance-optimizations) | 7 | 0 | 4 | 0 |
| [CUDA Compute Kernels](#cuda-compute-kernels) | 6 | 0 | 8 | 0 |
| [Observability & Operations](#observability-and-operations) | 6 | 0 | 1 | 2 |
| [Engine & Architecture](#engine-and-architecture) | 7 | 0 | 2 | 2 |
| **Total** | **127** | **1** | **136** | **13** |

---

## Hardware Platforms

| Feature | Python | Rust | Notes |
|---|:---:|:---:|---|
| CPU inference | ✅ | ✅ | Rust uses candle; Python uses PyTorch CPU |
| NVIDIA CUDA | ✅ | ✅ | Rust verified on L40S (SM89) |
| Apple Metal (MLX) | ❌ | ✅ | Rust-only; `--features metal` via mlx-rs |
| AMD ROCm / HIP | ✅ | ❌ |  |
| Google TPU | ✅ | ❌ |  |
| Intel XPU (Arc / Data Center) | ✅ | ❌ |  |
| AWS Neuron / Inferentia | ✅ | ❌ | Via plugin |
| Intel OpenVINO | ✅ | ❌ | Via plugin |
| Habana Gaudi (HPU) | ✅ | ❌ | Via plugin |
| Device auto-detection | ✅ | ✅ | Rust: Metal > CUDA > CPU |

---

## Multi-GPU & Distribution

| Feature | Python | Rust | Notes |
|---|:---:|:---:|---|
| Tensor parallelism (TP) | ✅ | ✅ | Rust: NCCL all-reduce wired via ThreadPoolExecutor; verified Qwen2.5-14B TP=2 on 2x L40S |
| Pipeline parallelism (PP) | ✅ | ❌ |  |
| Data parallelism (DP) | ✅ | ❌ |  |
| Expert parallelism (EP) for MoE | ✅ | ❌ |  |
| Multi-node distributed inference | ✅ | ⚠️ | Rust: TCP rendezvous + NCCL ID distribution; needs control channel for headless nodes |
| NCCL custom all-reduce | ✅ | ❌ |  |
| Prefill context parallelism | ✅ | ❌ |  |
| Decode context parallelism | ✅ | ❌ |  |
| Dual batch overlap (DBO) | ✅ | ❌ |  |

---

## CLI Commands

| Feature | Python | Rust | Notes |
|---|:---:|:---:|---|
| `serve` — start HTTP server | ✅ | ✅ |  |
| `chat` — interactive REPL | ✅ | ✅ | Rust superset: in-process + remote mode; --bench --prompt |
| `complete` — interactive REPL | ✅ | ✅ | Remote mode (connects to running server) |
| `bench latency` | ✅ | ✅ |  |
| `bench throughput` | ✅ | ✅ | Offline batch throughput (requests/s and tokens/s) |
| `bench serve` | ✅ | ✅ | Online serving benchmark (TTFT/TPOT/ITL/E2EL via HTTP) |
| `bench startup` | ✅ | ✅ | Cold/warm startup time measurement |
| `bench sweep` | ✅ | ✅ | serve + startup subcommands; plot subcommands omitted (Python-only matplotlib) |
| Offline batch inference (CLI) | ✅ | ✅ | Python: `run-batch`; Rust: `batch` + `run-batch` alias |
| `collect-env` | ✅ | ✅ | Rust-tailored: reports rustc/cargo/features instead of PyTorch/pip |
| `convert` model weights | ❌ | ⚠️ | Rust: argument parsing only (stub) |
| `top` — live TUI dashboard | ❌ | ✅ | Rust-only; ratatui-based |

---

## OpenAI-Compatible API Endpoints

| Feature | Python | Rust | Notes |
|---|:---:|:---:|---|
| `POST /v1/chat/completions` | ✅ | ✅ | Streaming + non-streaming |
| `POST /v1/completions` | ✅ | ✅ | Streaming + non-streaming |
| `POST /v1/embeddings` | ✅ | ✅ |  |
| `GET /v1/models` | ✅ | ✅ |  |
| `POST /v1/score` (cross-encoder) | ✅ | ❌ |  |
| `POST /v1/rerank` | ✅ | ❌ |  |
| `POST /classify` | ✅ | ❌ |  |
| `GET /health` | ✅ | ✅ |  |
| `GET /version` | ✅ | ✅ |  |
| `GET /metrics` (Prometheus) | ✅ | ✅ |  |
| `POST /tokenize` | ✅ | ❌ |  |
| `POST /detokenize` | ✅ | ❌ |  |
| `POST /v1/chat/completions/render` | ✅ | ❌ | Render chat to string without generating |

---

## Other API Protocols

| Feature | Python | Rust | Notes |
|---|:---:|:---:|---|
| OpenAI Responses API (`/v1/responses`) | ✅ | ❌ |  |
| Anthropic Messages API (`/v1/messages`) | ✅ | ❌ |  |
| gRPC server | ✅ | ❌ |  |
| SageMaker (`/ping` / `/invocations`) | ✅ | ❌ |  |
| WebSocket realtime (`/v1/realtime`) | ✅ | ❌ |  |
| Audio transcription (`/v1/audio/transcriptions`) | ✅ | ❌ |  |
| MCP (Model Context Protocol) server | ✅ | ❌ |  |
| Offline Rust `LLM` API (no HTTP) | N/A | ✅ | Rust-only `LLM::generate()`/`.chat()`/`.chat_stream()` |
| JSON stats (`/stats` / `/stats/live` SSE) | ❌ | ✅ | Rust-only live stats stream |
| ORCA load-reporting headers | ❌ | ✅ | Rust-only |

---

## Model Architectures — Decoder-Only LLMs

| Architecture | Python | Rust (Candle) | Rust (MLX) | Notes |
|---|:---:|:---:|:---:|---|
| LLaMA / LLaMA 2 / LLaMA 3 | ✅ | ✅ | ✅ |  |
| Mistral | ✅ | ✅ | ✅ | Shares LLaMA code path |
| Qwen2 / Qwen2.5 | ✅ | ✅ | ✅ |  |
| Qwen3 | ✅ | ✅ | ✅ |  |
| Phi-3 / Phi-4 | ✅ | ✅ | ✅ | LongRoPE for Phi-4 |
| Gemma 2 | ✅ | ✅ | ✅ |  |
| Gemma 3 (text-only) | ✅ | ✅ | ✅ |  |
| DeepSeek V2 / V3 (MLA + MoE) | ✅ | ✅ | ✅ |  |
| Command R (Cohere) | ✅ | ✅ | ✅ |  |
| Qwen2 MoE | ✅ | ✅ | ✅ |  |
| Qwen3 MoE | ✅ | ✅ | ✅ |  |
| Mixtral (MoE) | ✅ | ✅ | ✅ |  |
| Granite (IBM) | ✅ | ✅ | ✅ |  |
| Kimi K2.5 | ✅ | ✅ | ✅ | Uses DeepSeek V2 backbone |
| Qwen3-Next (hybrid GDN + MoE) | ✅ | ✅ | ✅ | Linear attention + full attention |
| GPT-NeoX | ✅ | ❌ | ❌ |  |
| GPT-J | ✅ | ❌ | ❌ |  |
| GPT-BigCode / StarCoder2 | ✅ | ❌ | ❌ |  |
| Falcon / Falcon-H1 | ✅ | ❌ | ❌ |  |
| BLOOM | ✅ | ❌ | ❌ |  |
| OPT | ✅ | ❌ | ❌ |  |
| OLMo / OLMo2 / OLMoE | ✅ | ❌ | ❌ |  |
| Nemotron / Nemotron-H | ✅ | ❌ | ❌ |  |
| Exaone / Exaone4 | ✅ | ❌ | ❌ |  |
| StableLM | ✅ | ❌ | ❌ |  |
| Baichuan | ✅ | ❌ | ❌ |  |
| ChatGLM / GLM-4 | ✅ | ❌ | ❌ |  |
| DBRX (MoE) | ✅ | ❌ | ❌ |  |
| Arctic (MoE) | ✅ | ❌ | ❌ |  |
| Mamba / Bamba / Jamba | ✅ | ❌ | ❌ | SSM-based |
| Zamba 2 | ✅ | ❌ | ❌ |  |
| PLaMo 2 / PLaMo 3 | ✅ | ❌ | ❌ |  |
| ERNIE 4.5 (dense + MoE + MTP) | ✅ | ❌ | ❌ |  |
| SolarPro | ✅ | ❌ | ❌ |  |
| 100+ additional architectures | ✅ | ❌ | ❌ | Long tail of niche models |

---

## Model Architectures — Encoder / Embedding

| Architecture | Python | Rust | Notes |
|---|:---:|:---:|---|
| BERT | ✅ | ❌ |  |
| ModernBERT | ✅ | ❌ |  |
| RoBERTa | ✅ | ❌ |  |
| ColBERT / ColQwen3 | ✅ | ❌ | Late-interaction rerankers |

---

## Model Architectures — Vision-Language / Multimodal

| Architecture | Python | Rust | Notes |
|---|:---:|:---:|---|
| Gemma 3 VLM (SigLIP + projector) | ✅ | ✅ |  |
| Qwen2-VL / Qwen2.5-VL | ✅ | ✅ |  |
| LLaMA 4 (Mllama4) | ✅ | ❌ |  |
| Qwen3-VL | ✅ | ❌ |  |
| Phi-3-Vision / Phi-4-MM | ✅ | ❌ |  |
| LLaVA / Pixtral | ✅ | ❌ |  |
| InternVL2 | ✅ | ❌ |  |
| DeepSeek-VL2 | ✅ | ❌ |  |
| Molmo / Molmo2 | ✅ | ❌ |  |
| PaliGemma | ✅ | ❌ |  |
| 30+ additional VLMs | ✅ | ❌ |  |

---

## Model Architectures — Audio / Speech

| Architecture | Python | Rust | Notes |
|---|:---:|:---:|---|
| Whisper | ✅ | ❌ |  |
| Qwen2-Audio / Qwen3-ASR | ✅ | ❌ |  |
| Ultravox | ✅ | ❌ |  |
| Voxtral / Voxtral-Realtime | ✅ | ❌ |  |

---

## Model Architectures — Speculative Decoding Draft Models

| Architecture | Python | Rust | Notes |
|---|:---:|:---:|---|
| EAGLE / EAGLE3 heads | ✅ | ❌ |  |
| Medusa heads | ✅ | ❌ |  |
| MLP Speculator | ✅ | ❌ |  |
| DeepSeek-Eagle | ✅ | ❌ |  |

---

## Quantization Methods

| Method | Python | Rust | Notes |
|---|:---:|:---:|---|
| GGUF (all k-quant variants) | ✅ | ✅ |  |
| GPTQ | ✅ | ✅ | Rust: LLaMA-family only |
| AWQ | ✅ | ✅ | Rust: LLaMA-family only |
| BitsAndBytes NF4 (4-bit) | ✅ | ✅ | Rust: LLaMA-family only |
| MLX 4-bit quantized | N/A | ✅ | Rust-only; mlx-community models |
| FP8 (W8A8 / W8A16) | ✅ | ❌ |  |
| Marlin kernels (AWQ/GPTQ) | ✅ | ❌ |  |
| Compressed-tensors (Neural Magic) | ✅ | ❌ |  |
| TorchAO (int4/int8/fp8) | ✅ | ❌ |  |
| MXFP4 (microscaling) | ✅ | ❌ |  |
| ModelOpt (NVIDIA FP4/FP8) | ✅ | ❌ |  |
| Quark (AMD) | ✅ | ❌ |  |
| FP8 KV cache quantization | ✅ | ❌ |  |

---

## Attention Backends

| Backend | Python | Rust | Notes |
|---|:---:|:---:|---|
| Scaled dot-product (CPU) | ✅ | ✅ |  |
| FlashAttention-2 (single sequence) | ✅ | ✅ | Rust: CUDA only |
| FlashAttention-2 varlen (batched prefill) | ✅ | ✅ |  |
| Paged FlashAttention-2 (batched decode) | ✅ | ✅ | Rust: forked candle-flash-attn |
| FlashAttention-3 | ✅ | ❌ |  |
| FlashInfer | ✅ | ❌ |  |
| Triton attention | ✅ | ❌ |  |
| ROCm AITER attention | ✅ | ❌ |  |
| FlashInfer MLA (DeepSeek) | ✅ | ❌ |  |
| Triton MLA (DeepSeek) | ✅ | ❌ |  |
| Tree attention (speculative) | ✅ | ❌ |  |
| Mamba1 / Mamba2 SSM | ✅ | ❌ |  |
| GDN (gated diffusion network) | ✅ | ✅ | Rust: Qwen3-Next |
| MLX paged decode attention | N/A | ✅ | Rust-only |

---

## Sampling & Decoding

| Feature | Python | Rust | Notes |
|---|:---:|:---:|---|
| Greedy (argmax) | ✅ | ✅ |  |
| Temperature scaling | ✅ | ✅ |  |
| Top-k | ✅ | ✅ |  |
| Top-p (nucleus) | ✅ | ✅ |  |
| Min-p | ✅ | ✅ |  |
| Repetition penalty | ✅ | ✅ |  |
| Frequency penalty | ✅ | ✅ |  |
| Presence penalty | ✅ | ✅ |  |
| Logit bias | ✅ | ✅ |  |
| Logprobs (top-N) | ✅ | ✅ |  |
| Prompt logprobs | ✅ | ✅ |  |
| Random seed | ✅ | ✅ |  |
| Stop strings | ✅ | ✅ |  |
| Stop token IDs | ✅ | ✅ |  |
| `ignore_eos` | ✅ | ✅ |  |
| `min_tokens` | ✅ | ✅ |  |
| `n > 1` completions | ✅ | ✅ |  |
| `echo` (return prompt in output) | ✅ | ✅ |  |
| `bad_words` blocklist | ✅ | ❌ |  |
| `allowed_token_ids` whitelist | ✅ | ❌ |  |
| `truncate_prompt_tokens` | ✅ | ❌ |  |
| ~~Beam search~~ | ✅ | ❌ | ~~Deprecated in Python V1~~ |
| ~~`best_of` / `n` with rejection~~ | ✅ | ❌ | ~~Deprecated in Python V1~~ |
| GPU-side fused sampling (CUDA) | ✅ | ✅ | Rust: Gumbel-max + fused top-k/top-p/min-p |

---

## Structured Output / Guided Decoding

| Feature | Python | Rust | Notes |
|---|:---:|:---:|---|
| JSON schema | ✅ | ✅ |  |
| JSON object (freeform) | ✅ | ✅ |  |
| Regex constraint | ✅ | ✅ |  |
| Choice (enum) | ✅ | ✅ |  |
| EBNF grammar | ✅ | ❌ |  |
| Structural tags | ✅ | ❌ |  |
| Backend: outlines-core FSM | ✅ | ✅ |  |
| Backend: xgrammar | ✅ | ❌ |  |
| Backend: guidance | ✅ | ❌ |  |
| Backend: lm-format-enforcer | ✅ | ❌ |  |

---

## Tool Calling / Function Calling

| Feature | Python | Rust | Notes |
|---|:---:|:---:|---|
| `tools` / `tool_choice` in chat completions | ✅ | ✅ |  |
| `auto` / `required` / `none` tool choice | ✅ | ✅ |  |
| Named tool choice | ✅ | ✅ |  |
| Parallel tool calls | ✅ | ✅ |  |
| Hermes tool parser | ✅ | ✅ |  |
| Llama3 JSON tool parser | ✅ | ✅ |  |
| 25+ additional model-specific parsers | ✅ | ❌ | Mistral, DeepSeek, Qwen3, Pythonic, etc. |
| Reasoning parsers (strip CoT tokens) | ✅ | ❌ | DeepSeek-R1, Qwen3, etc. |

---

## Scheduling

| Feature | Python | Rust | Notes |
|---|:---:|:---:|---|
| Continuous batching | ✅ | ✅ |  |
| FCFS scheduling policy | ✅ | ✅ |  |
| Priority scheduling policy | ✅ | ✅ |  |
| Chunked prefill | ✅ | ✅ | Default on in both |
| Prefix caching (APC) | ✅ | ✅ | Hash-based block reuse; default on in both |
| Async scheduling (overlap GPU/CPU) | ✅ | ✅ |  |
| ~~Preemption (swap to CPU)~~ | ✅ | ❌ | ~~Deprecated in Python V1~~ |
| Custom scheduler class (pluggable) | ✅ | ❌ |  |

---

## KV Cache

| Feature | Python | Rust | Notes |
|---|:---:|:---:|---|
| Paged KV cache (block pool) | ✅ | ✅ |  |
| Block allocation / free / reuse | ✅ | ✅ |  |
| Prefix-cached block lookup | ✅ | ✅ |  |
| Contiguous KV buffer (CUDA decode opt) | ✅ | ✅ | Rust: avoids Tensor::cat per step |
| CPU swap space | ✅ | ❌ | Python: default 4 GB |
| KV cache offloading to CPU | ✅ | ❌ |  |
| FP8 KV cache | ✅ | ❌ |  |
| KV transfer / disaggregated prefill | ✅ | ❌ | Python: NCCL, LMCache, NIXL, Mooncake connectors |

---

## LoRA & Adapters

| Feature | Python | Rust | Notes |
|---|:---:|:---:|---|
| Single LoRA adapter at startup | ✅ | ✅ | PEFT-format safetensors |
| Multi-LoRA concurrent serving | ✅ | ❌ |  |
| Dynamic LoRA hot-load/unload (REST API) | ✅ | ❌ |  |
| Fully sharded LoRA (across TP ranks) | ✅ | ❌ |  |
| Punica batched LoRA GEMM kernels | ✅ | ❌ |  |
| rsLoRA scaling | ✅ | ✅ |  |

---

## Speculative Decoding

| Feature | Python | Rust | Notes |
|---|:---:|:---:|---|
| N-gram prompt lookup proposer | ✅ | ✅ |  |
| Draft model (separate small LM) | ✅ | ❌ |  |
| EAGLE / EAGLE3 draft heads | ✅ | ❌ |  |
| Medusa draft heads | ✅ | ❌ |  |
| Suffix decoding | ✅ | ❌ |  |
| Tree attention verification | ✅ | ❌ |  |

---

## Multimodal Input

| Feature | Python | Rust | Notes |
|---|:---:|:---:|---|
| Image input (single / batched) | ✅ | ✅ | Rust: Gemma3-MM, Qwen2-VL, Qwen2.5-VL |
| Video input | ✅ | ❌ |  |
| Audio input | ✅ | ❌ |  |
| Image embeddings (pre-encoded) | ✅ | ❌ |  |
| Mixed modalities (image + audio + text) | ✅ | ❌ |  |

---

## Embeddings & Pooling

| Feature | Python | Rust | Notes |
|---|:---:|:---:|---|
| `/v1/embeddings` endpoint | ✅ | ✅ |  |
| `--runner pooling` mode | N/A | ✅ | Rust-only CLI flag |
| Pooling: last token | ✅ | ✅ |  |
| Pooling: CLS token | ✅ | ✅ |  |
| Pooling: mean | ✅ | ✅ |  |
| Cross-encoder scoring | ✅ | ❌ |  |
| Reranking | ✅ | ❌ |  |
| Classification head | ✅ | ❌ |  |

---

## Serving Features

| Feature | Python | Rust | Notes |
|---|:---:|:---:|---|
| SSE streaming | ✅ | ✅ |  |
| TLS / HTTPS | ✅ | ✅ |  |
| Mutual TLS (client cert) | ❌ | ✅ | Rust-only; via `--ssl-ca-certs` |
| Chat templates (Jinja2) | ✅ | ✅ | Rust: minijinja |
| CORS | ✅ | ✅ |  |
| `stream_options.include_usage` | ✅ | ✅ |  |
| Incremental detokenization | ✅ | ✅ |  |
| OpenTelemetry tracing | ✅ | ❌ |  |
| Sleep / wake (GPU memory release) | ✅ | ❌ |  |
| RLHF pause / resume / weight update | ✅ | ❌ |  |
| Dynamic LoRA REST endpoints | ✅ | ❌ |  |
| Prefix cache reset endpoint | ✅ | ❌ |  |
| `/server_info` endpoint | ✅ | ❌ |  |
| Elastic DP scaling | ✅ | ❌ |  |

---

## Performance Optimizations

| Feature | Python | Rust | Notes |
|---|:---:|:---:|---|
| CUDA graphs (decode) | ✅ | ✅ | Rust: configurable batch sizes |
| Fused RMS norm + residual add | ✅ | ✅ | CUDA kernel |
| Fused rotary embeddings | ✅ | ✅ | CUDA kernel |
| Fused MoE gating (top-k) | ✅ | ✅ | CUDA kernel |
| Fused reshape-and-cache | ✅ | ✅ | CUDA kernel |
| Fused GPU sampling | ✅ | ✅ | CUDA kernel |
| Triton kernels | ✅ | ❌ | Rust has no Triton equivalent |
| Torch.compile / inductor | ✅ | ❌ |  |
| Weight-only INT8/FP8 GEMM | ✅ | ❌ |  |
| Fused cross-entropy loss | ✅ | ❌ |  |
| NVTX profiling annotations | ✅ | ✅ | Rust: `--features profiling` |

---

## CUDA Compute Kernels

| Kernel | Python | Rust | Notes |
|---|:---:|:---:|---|
| `fused_add_rms_norm` | ✅ | ✅ | Vectorized 128-bit loads |
| Rotary embedding (fused) | ✅ | ✅ |  |
| `reshape_and_cache` | ✅ | ✅ |  |
| MoE top-k gating | ✅ | ✅ |  |
| GPU sampling (Gumbel-max) | ✅ | ✅ |  |
| FlashAttention-2 (paged) | ✅ | ✅ | Forked candle-flash-attn |
| Paged attention v1/v2 (PagedAttention) | ✅ | ❌ |  |
| `silu_and_mul` fused activation | ✅ | ❌ |  |
| `gelu_and_mul` fused activation | ✅ | ❌ |  |
| Fused MoE GEMM | ✅ | ❌ |  |
| Marlin (INT4 GEMM) | ✅ | ❌ |  |
| FP8 GEMM | ✅ | ❌ |  |
| Prefix caching hash kernel | ✅ | ❌ |  |
| Custom all-reduce | ✅ | ❌ |  |

---

## Observability & Operations

| Feature | Python | Rust | Notes |
|---|:---:|:---:|---|
| Prometheus metrics endpoint | ✅ | ✅ |  |
| Request latency metrics (TTFT / ITL) | ✅ | ✅ |  |
| KV cache utilization gauge | ✅ | ✅ |  |
| Token throughput counters | ✅ | ✅ |  |
| Queue depth metrics | ✅ | ✅ |  |
| OpenTelemetry tracing | ✅ | ❌ |  |
| `collect-env` diagnostic dump | ✅ | ✅ | Rust version reports system/GPU/toolchain/features |
| Live stats SSE stream | ❌ | ✅ | Rust-only: `/stats/live` |
| TUI dashboard (`vllm top`) | ❌ | ✅ | Rust-only |

---

## Engine & Architecture

| Feature | Python | Rust | Notes |
|---|:---:|:---:|---|
| Async engine (non-blocking) | ✅ | ✅ |  |
| In-process engine client | ✅ | ✅ |  |
| Single-worker executor | ✅ | ✅ |  |
| Multi-worker executor | ✅ | ✅ | Rust: tokio channels |
| Ray executor (distributed) | ✅ | ❌ |  |
| External launcher executor | ✅ | ❌ |  |
| PyO3 scheduler bridge | N/A | ✅ | Rust scheduler usable from Python |
| Offline `LLM` API (programmatic) | ✅ | ✅ | Python: `LLM` class; Rust: `LLM` struct |
| HuggingFace tokenizers | ✅ | ✅ | Same `tokenizers` library |
| HuggingFace Hub model download | ✅ | ✅ |  |
| Standalone binary (no Python) | N/A | ✅ | Rust-only |

