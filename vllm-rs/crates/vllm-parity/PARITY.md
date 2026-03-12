# vLLM Feature Parity: Python vs Rust

> Last updated: 2026-03-12

| Symbol | Meaning | Count |
|--------|---------|------:|
| ✅ 🟦 | Implemented | 158 |
| ⚠️ 🟨 | Partial | 3 |
| ❌ 🟥 | Not implemented | 121 |
| 🚫 | Won't fix | 3 |

---

## Summary

> Counts are for Rust parity against Python features.

| Section | Parity | ✅ | ⚠️ | ❌ |
|---|---|---:|---:|---:|
| [Hardware Platforms](#hardware-platforms) | 🟦🟦🟥🟥🟥🟥🟥🟥 | 2 | 0 | 6 |
| [Multi-GPU & Distribution](#multi-gpu--distribution) | 🟦🟥🟥🟥🟥🟥🟥🟥🟥 | 1 | 0 | 8 |
| [CLI Commands](#cli-commands) | 🟦🟦🟦🟦🟦🟦🟦🟦🟦🟦 | 10 | 0 | 0 |
| [OpenAI-Compatible API Endpoints](#openai-compatible-api-endpoints) | 🟦🟦🟦🟦🟦🟦🟦🟦🟦🟦🟥🟥🟥 | 10 | 0 | 3 |
| [Other API Protocols](#other-api-protocols) | 🟦🟥🟥🟥🟥🟥🟥 | 1 | 0 | 6 |
| [Model Architectures — Decoder-Only LLMs](#model-architectures--decoder-only-llms) | 🟦🟦🟦🟦🟦🟦🟦🟦🟦🟦🟦🟦🟦🟨🟨🟥🟥🟥🟥🟥🟥🟥🟥🟥🟥🟥🟥🟥🟥🟥🟥🟥🟥🟥🟥 | 13 | 2 | 20 |
| [Model Architectures — Encoder / Embedding](#model-architectures--encoder--embedding) | 🟥🟥🟥🟥 | 0 | 0 | 4 |
| [Model Architectures — Vision-Language / Multimodal](#model-architectures--vision-language--multimodal) | 🟥🟥🟥🟥🟥🟥🟥🟥🟥🟥🟥 | 0 | 0 | 11 |
| [Model Architectures — Audio / Speech](#model-architectures--audio--speech) | 🟥🟥🟥🟥 | 0 | 0 | 4 |
| [Model Architectures — Speculative Decoding Draft Models](#model-architectures--speculative-decoding-draft-models) | 🟥🟥🟥🟥 | 0 | 0 | 4 |
| [Quantization Methods](#quantization-methods) | 🟦🟦🟦🟦🟦🟦🟥🟥🟥🟥🟥🟥🟥 | 6 | 0 | 7 |
| [Attention Backends](#attention-backends) | 🟦🟦🟦🟥🟥🟥🟥🟥🟥🟥🟥🟥 | 3 | 0 | 9 |
| [Sampling & Decoding](#sampling--decoding) | 🟦🟦🟦🟦🟦🟦🟦🟦🟦🟦🟦🟦🟦🟦🟦🟦🟦🟦🟦🟦🟦🟦 | 22 | 0 | 0 |
| [Structured Output / Guided Decoding](#structured-output--guided-decoding) | 🟦🟦🟦🟦🟦🟦🟦 | 7 | 0 | 0 |
| [Tool Calling / Function Calling](#tool-calling--function-calling) | 🟦🟦🟦🟦🟦🟦🟦🟦🟦🟦🟨🟥🟥🟥🟥🟥🟥 | 10 | 1 | 6 |
| [Scheduling](#scheduling) | 🟦🟦🟦🟦🟦🟦 | 6 | 0 | 0 |
| [KV Cache](#kv-cache) | 🟦🟦🟦🟦🟦🟥🟥🟥 | 5 | 0 | 3 |
| [LoRA & Adapters](#lora--adapters) | 🟦🟦🟥🟥🟥🟥 | 2 | 0 | 4 |
| [Speculative Decoding](#speculative-decoding) | 🟦🟥🟥🟥🟥🟥 | 1 | 0 | 5 |
| [Multimodal Input](#multimodal-input) | 🟥🟥🟥🟥🟥 | 0 | 0 | 5 |
| [Embeddings & Pooling](#embeddings--pooling) | 🟦🟦🟦🟦🟥🟥🟥 | 4 | 0 | 3 |
| [Serving Features](#serving-features) | 🟦🟦🟦🟦🟦🟦🟦🟦🟦🟦🟥🟥🟥 | 10 | 0 | 3 |
| [Performance Optimizations](#performance-optimizations) | 🟦🟦🟦🟦🟦🟦🟦🟦🟦🟥🟥🟥🟥 | 9 | 0 | 4 |
| [CUDA Compute Kernels](#cuda-compute-kernels) | 🟦🟦🟦🟦🟦🟦🟦🟦🟦🟦🟦🟦🟦🟦🟦🟦🟦🟦🟦🟦🟦🟦🟥🟥🟥🟥 | 22 | 0 | 4 |
| [Observability & Operations](#observability--operations) | 🟦🟦🟦🟦🟦🟦🟦 | 7 | 0 | 0 |
| [Engine & Architecture](#engine--architecture) | 🟦🟦🟦🟦🟦🟦🟦🟥🟥 | 7 | 0 | 2 |
| **Total** | 🟦🟦🟦🟦🟦🟦🟦🟦🟦🟦🟦🟦🟦🟦🟦🟦🟦🟦🟦🟦🟦🟦🟦🟦🟦🟦🟦🟦🟦🟦🟦🟦🟦🟦🟦🟦🟦🟦🟦🟦🟦🟦🟦🟦🟦🟦🟦🟦🟦🟦🟦🟦🟦🟦🟦🟦🟦🟦🟦🟦🟦🟦🟦🟦🟦🟦🟦🟦🟦🟦🟦🟦🟦🟦🟦🟦🟦🟦🟦🟦🟦🟦🟦🟦🟦🟦🟦🟦🟦🟦🟦🟦🟦🟦🟦🟦🟦🟦🟦🟦🟦🟦🟦🟦🟦🟦🟦🟦🟦🟦🟦🟦🟦🟦🟦🟦🟦🟦🟦🟦🟦🟦🟦🟦🟦🟦🟦🟦🟦🟦🟦🟦🟦🟦🟦🟦🟦🟦🟦🟦🟦🟦🟦🟦🟦🟦🟦🟦🟦🟦🟦🟦🟦🟦🟦🟦🟦🟦🟨🟨🟨🟥🟥🟥🟥🟥🟥🟥🟥🟥🟥🟥🟥🟥🟥🟥🟥🟥🟥🟥🟥🟥🟥🟥🟥🟥🟥🟥🟥🟥🟥🟥🟥🟥🟥🟥🟥🟥🟥🟥🟥🟥🟥🟥🟥🟥🟥🟥🟥🟥🟥🟥🟥🟥🟥🟥🟥🟥🟥🟥🟥🟥🟥🟥🟥🟥🟥🟥🟥🟥🟥🟥🟥🟥🟥🟥🟥🟥🟥🟥🟥🟥🟥🟥🟥🟥🟥🟥🟥🟥🟥🟥🟥🟥🟥🟥🟥🟥🟥🟥🟥🟥🟥🟥🟥🟥🟥🟥🟥🟥🟥🟥🟥🟥🟥🟥🟥🟥🟥🟥🟥🟥 | **158** | **3** | **121** |

---

## Hardware Platforms

| Feature | Python | Rust | Notes |
|---|:---:|:---:|---|
| ~~CPU inference~~ | ✅ | ❌ | 🚫 Won't fix — Removed with CandleWorker; requires CUDA or Metal |
| NVIDIA CUDA | ✅ | ✅ | CudaWorker verified on L40S (SM89) |
| Apple Metal (MLX) | ❌ | ✅ | Rust-only; `--features metal` via mlx-rs |
| WebGPU (wgpu) | ❌ | ✅ | Rust-only; `--features wgpu` via wgpu-rs (Metal/Vulkan/DX12); ~51 tok/s Qwen2.5-0.5B on M1 Max |
| AMD ROCm / HIP | ✅ | ❌ |  |
| Google TPU | ✅ | ❌ |  |
| Intel XPU (Arc / Data Center) | ✅ | ❌ |  |
| AWS Neuron / Inferentia | ✅ | ❌ | Via plugin |
| Intel OpenVINO | ✅ | ❌ | Via plugin |
| Habana Gaudi (HPU) | ✅ | ❌ | Via plugin |
| Device auto-detection | ✅ | ✅ | Rust: Metal > CUDA > wgpu (no CPU fallback) |

---

## Multi-GPU & Distribution

| Feature | Python | Rust | Notes |
|---|:---:|:---:|---|
| Tensor parallelism (TP) — single-node | ✅ | ✅ | CudaWorker: NCCL all-reduce via ThreadPoolExecutor; all dense + MoE archs wired; E2E verified TP=2 Qwen2.5-0.5B + Gemma3-4B on 2x L40S |
| Tensor parallelism (TP) — multi-node | ✅ | ❌ |  |
| Pipeline parallelism (PP) | ✅ | ❌ |  |
| Data parallelism (DP) | ✅ | ❌ |  |
| Expert parallelism (EP) for MoE | ✅ | ❌ | Needed for MoE models where experts are split across GPUs |
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
| `POST /tokenize` | ✅ | ✅ | Prompt mode; chat mode with chat-template feature |
| `POST /detokenize` | ✅ | ✅ |  |
| `POST /v1/chat/completions/render` | ✅ | ✅ | Renders chat template and returns prompt text + token IDs |

---

## Other API Protocols

| Feature | Python | Rust | Notes |
|---|:---:|:---:|---|
| OpenAI Responses API (`/v1/responses`) | ✅ | ❌ |  |
| Anthropic Messages API (`/v1/messages`) | ✅ | ✅ | Streaming + non-streaming; translates to internal chat completion |
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
| Mistral | ✅ | ✅ | ✅ | LLaMA alias in CudaWorker |
| Qwen2 / Qwen2.5 | ✅ | ✅ | ✅ |  |
| Qwen3 | ✅ | ✅ | ✅ | LLaMA alias in CudaWorker |
| Phi-3 / Phi-4 | ✅ | ✅ | ✅ | LLaMA alias in CudaWorker; LongRoPE for Phi-4 |
| Gemma 2 | ✅ | ✅ | ✅ |  |
| Gemma 3 (text-only) | ✅ | ✅ | ✅ | CUDA graphs supported; Gemma3ForConditionalGeneration resolved as text-only via text_config |
| DeepSeek V2 / V3 (MLA + MoE) | ✅ | ⚠️ | ✅ | V2/V2-Lite at parity (non-absorbed MLA + 6 CUDA kernels + YaRN RoPE); V3 gaps: grouped top-k routing + sigmoid scoring + e_score_correction_bias + noaux_tc routing |
| Command R (Cohere) | ✅ | ✅ | ✅ | BNB 4-bit verified on L40S |
| Qwen2 MoE | ✅ | ✅ | ✅ | MoE + shared expert (gated) |
| Qwen3 MoE | ✅ | ✅ | ✅ | MoE + shared expert + QK-norm |
| Mixtral (MoE) | ✅ | ✅ | ✅ | Pure MoE (8 experts top-2) |
| Granite (IBM) | ✅ | ✅ | ✅ | LLaMA + 4 scalar multipliers |
| Kimi K2.5 | ✅ | ✅ | ✅ | DeepSeek V2 backbone |
| Qwen3-Next (hybrid GDN + MoE) | ✅ | ⚠️ | ✅ | Gaps: chunked prefill (uses fused_recurrent not chunk_gated_delta_rule); GDN TP; spec decode token splitting; has_initial_state flag; conv1d bias; L2 norm in recurrence; MTP |
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
| Gemma 3 VLM (SigLIP + projector) | ✅ | ❌ | Removed with CandleWorker; not ported to CudaWorker |
| Qwen2-VL / Qwen2.5-VL | ✅ | ❌ | Removed with CandleWorker; not ported to CudaWorker |
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
| GGUF (all k-quant variants) | ✅ | ✅ | llama.cpp-derived dequant kernels; BS=1 fused dequant-matvec + BS>1 Q8_1 dot products; archs: LLaMA/Qwen2/Qwen3; E2E: Qwen2.5-0.5B + Qwen3-0.6B GGUF; CUDA graphs disabled (incompatible with dynamic allocs) |
| GPTQ | ✅ | ✅ | Marlin W4A16 on SM80+; symmetric + desc_act (activation ordering); fused QKV/gate_up at load; post-GEMM bias_add_inplace for linear bias; CUDA graphs work; archs: LLaMA/Qwen2/Gemma2/Granite; note: asymmetric zero-points not passed (uint4b8 bakes in zp like Python vLLM) |
| AWQ | ✅ | ✅ | Marlin W4A16 on SM80+; fused QKV/gate_up at load; CUDA graphs work; archs: LLaMA/Qwen2/Gemma2/Granite; E2E verified Qwen2.5-0.5B |
| BitsAndBytes NF4 (4-bit) | ✅ | ✅ | Dequant-then-cuBLAS GEMM; double quantization supported; per-shard matmuls for QKV and gate/up; archs: LLaMA/Qwen2/Gemma2 (+ aliases Mistral/Qwen3/Phi-3/Granite); E2E verified unsloth/Qwen3-0.6B-bnb-4bit |
| Quantized MoE (FP8/INT8/INT4 experts) | ✅ | ❌ | CudaWorker MoE kernel is BF16/F16 only; Python supports FP8 W8A8 + INT8 W8A8 + INT4 W4A16 |
| MLX 4-bit quantized | N/A | ✅ | Rust-only; mlx-community models |
| FP8 (W8A8 / W8A16) | ✅ | ❌ |  |
| Marlin kernels (AWQ/GPTQ) | ✅ | ✅ | W4A16 only; auto-converts at load on SM80+; use_fp32_reduce matches Python default |
| Compressed-tensors (Neural Magic) | ✅ | ❌ |  |
| TorchAO (int4/int8/fp8) | ✅ | ❌ |  |
| MXFP4 (microscaling) | ✅ | ❌ |  |
| ModelOpt (NVIDIA FP4/FP8) | ✅ | ❌ |  |
| Quark (AMD) | ✅ | ❌ |  |
| FP8 KV cache quantization | ✅ | ✅ | CudaWorker: FP8 E4M3 KV cache with dequant-gather + contiguous FA2 |

---

## Attention Backends

| Backend | Python | Rust | Notes |
|---|:---:|:---:|---|
| ~~Scaled dot-product (CPU)~~ | ✅ | ❌ | 🚫 Won't fix — Removed with CandleWorker |
| FlashAttention-2 (single sequence) | ✅ | ✅ | CudaWorker: direct FFI |
| FlashAttention-2 varlen (batched prefill) | ✅ | ❌ | Removed with CandleWorker |
| Paged FlashAttention-2 (all batches) | ✅ | ✅ | CudaWorker: direct FFI to vllm-flash-attn fork; prefill + decode + mixed |
| FlashAttention-3 | ✅ | ❌ |  |
| FlashInfer | ✅ | ❌ |  |
| Triton attention | ✅ | ❌ |  |
| ROCm AITER attention | ✅ | ❌ |  |
| FlashInfer MLA (DeepSeek) | ✅ | ❌ |  |
| Triton MLA (DeepSeek) | ✅ | ❌ |  |
| Tree attention (speculative) | ✅ | ❌ |  |
| Mamba1 / Mamba2 SSM | ✅ | ❌ |  |
| GDN (gated diffusion network) | ✅ | ✅ | Rust: Qwen3-Next; gaps: chunked prefill algo + GDN TP |
| MLX paged decode attention | N/A | ✅ | Rust-only |

---

## Sampling & Decoding

| Feature | Python | Rust | Notes |
|---|:---:|:---:|---|
| Greedy (argmax) | ✅ | ✅ | CudaWorker: in-graph argmax for CUDA graphs |
| Temperature scaling | ✅ | ✅ | CudaWorker: Gumbel-max (fast) or fused kernel |
| Top-k | ✅ | ✅ | CudaWorker: fused radix-select kernel |
| Top-p (nucleus) | ✅ | ✅ | CudaWorker: fused kernel |
| Min-p | ✅ | ✅ | CudaWorker: fused kernel |
| Repetition penalty | ✅ | ✅ | CudaWorker: fused GPU kernel (one block per request) |
| Frequency penalty | ✅ | ✅ | CudaWorker: fused with rep/pres in single kernel |
| Presence penalty | ✅ | ✅ | CudaWorker: fused with rep/freq in single kernel |
| Logit bias | ✅ | ✅ | CudaWorker: CSR-packed scatter-add kernel |
| Logprobs (top-N) | ✅ | ✅ | CudaWorker: fused log-softmax + top-K kernel |
| Prompt logprobs | ✅ | ✅ |  |
| Random seed | ✅ | ✅ | Per-request StdRng seeded from user seed |
| Stop strings | ✅ | ✅ |  |
| Stop token IDs | ✅ | ✅ |  |
| `ignore_eos` | ✅ | ✅ |  |
| `min_tokens` | ✅ | ✅ | CudaWorker: GPU kernel suppresses EOS/stop tokens until min_tokens reached |
| `n > 1` completions | ✅ | ✅ | Engine-level (not worker-level) |
| `echo` (return prompt in output) | ✅ | ✅ |  |
| `bad_words` blocklist | ✅ | ✅ | CudaWorker: CPU suffix matching + GPU mask kernel |
| `allowed_token_ids` whitelist | ✅ | ✅ | CudaWorker: reuses grammar mask kernel (CSR allow-list) |
| `truncate_prompt_tokens` | ✅ | ✅ | Truncates from left (keeps last N tokens) |
| ~~Beam search~~ | ✅ | ❌ | ~~Deprecated in Python V1~~ |
| ~~`best_of` / `n` with rejection~~ | ✅ | ❌ | ~~Deprecated in Python V1~~ |
| GPU-side fused sampling (CUDA) | ✅ | ✅ | CudaWorker: argmax + Gumbel-max + fused top-k/top-p/min-p; in-graph for CUDA graphs; full LogitsProcessor pipeline (no CPU fallback) |

---

## Structured Output / Guided Decoding

| Feature | Python | Rust | Notes |
|---|:---:|:---:|---|
| JSON schema | ✅ | ✅ |  |
| JSON object (freeform) | ✅ | ✅ |  |
| Regex constraint | ✅ | ✅ |  |
| Choice (enum) | ✅ | ✅ |  |
| EBNF grammar | ✅ | ✅ | via llguidance Lark parser |
| Structural tags | ✅ | ✅ | Port of Python llguidance StructTag.to_grammar(); converts to Lark grammar |
| ~~Backend: outlines-core FSM~~ | ✅ | ❌ | ~~Deprecated in Python V1~~ Replaced by llguidance |
| ~~Backend: xgrammar~~ | ✅ | ❌ | ~~Deprecated in Python V1~~ Redundant; llguidance covers all constraint types (JSON schema/regex/grammar/choice/structural tags). Python keeps xgrammar for historical compatibility. |
| Backend: guidance / llguidance | ✅ | ✅ | llguidance 1.6 (same engine as Python guidance backend) |
| ~~Backend: lm-format-enforcer~~ | ✅ | ❌ | ~~Deprecated in Python V1~~ Redundant; llguidance covers all constraint types. Python keeps lm-format-enforcer for historical compatibility. |

---

## Tool Calling / Function Calling

| Feature | Python | Rust | Notes |
|---|:---:|:---:|---|
| `tools` / `tool_choice` in chat completions | ✅ | ✅ |  |
| `auto` / `required` / `none` tool choice | ✅ | ✅ |  |
| Named tool choice | ✅ | ✅ |  |
| Parallel tool calls | ✅ | ✅ |  |
| `hermes` tool parser | ✅ | ✅ |  |
| `llama3_json` / `llama4_json` tool parser | ✅ | ✅ |  |
| `mistral` tool parser | ✅ | ✅ | v11+ and pre-v11 formats; auto-detected |
| `kimi_k2` tool parser | ✅ | ✅ |  |
| `deepseek_v3` tool parser | ✅ | ❌ | Unicode special-token delimiters + regex |
| `deepseek_v31` tool parser | ✅ | ❌ | Same as v3 with slightly simpler regex |
| `deepseek_v32` tool parser | ✅ | ❌ | DSML XML tags + per-parameter type coercion |
| `granite` tool parser | ✅ | ✅ | `<|tool_call|>` / `<tool_call>` + JSON array; streaming + non-streaming; E2E tests |
| `qwen3_coder` tool parser | ✅ | ❌ | XML-like tags (`<tool_call>` `<function=...>` `<parameter=...>`) + type coercion |
| `qwen3_xml` tool parser | ✅ | ❌ | SAX-style XML parsing via expat; 1318 lines |
| `jamba` tool parser | ✅ | ⚠️ | Parser implemented; needs E2E testing once JambaForCausalLM is supported |
| Other tool parsers (20) | ✅ | ❌ | granite-20b-fc, pythonic, llama4_pythonic, phi4_mini_json, internlm, xlam, longcat, glm45, glm47, functiongemma, hunyuan_a13b, minimax, minimax_m2, olmo3, openai, seed_oss, step3, step3p5, ernie45, gigachat3 |
| Reasoning parsers (strip CoT tokens) | ✅ | ✅ | DeepSeek-R1, Qwen3; --reasoning-parser CLI arg; streaming + non-streaming |

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
| ~~Custom scheduler class (pluggable)~~ | ✅ | ❌ | 🚫 Won't fix — Python-only: dynamic class loading by import path; even Python marks this unstable |

---

## KV Cache

| Feature | Python | Rust | Notes |
|---|:---:|:---:|---|
| Paged KV cache (block pool) | ✅ | ✅ |  |
| Block allocation / free / reuse | ✅ | ✅ |  |
| Prefix-cached block lookup | ✅ | ✅ |  |
| Contiguous KV buffer (CUDA decode opt) | ✅ | ✅ | Avoids Tensor::cat per step |
| CPU swap space | ✅ | ❌ | Python: default 4 GB |
| KV cache offloading to CPU | ✅ | ❌ |  |
| FP8 KV cache | ✅ | ✅ | --kv-cache-dtype fp8_e4m3 + --calculate-kv-scales; 2x block capacity |
| KV transfer / disaggregated prefill | ✅ | ❌ | Python: NCCL, LMCache, NIXL, Mooncake connectors |

---

## LoRA & Adapters

| Feature | Python | Rust | Notes |
|---|:---:|:---:|---|
| Single LoRA adapter at startup | ✅ | ✅ | CPU-side weight merging at load time (CudaWorker + MLX) |
| Multi-LoRA concurrent serving | ✅ | ❌ |  |
| Dynamic LoRA hot-load/unload (REST API) | ✅ | ❌ |  |
| Fully sharded LoRA (across TP ranks) | ✅ | ❌ |  |
| Punica batched LoRA GEMM kernels | ✅ | ❌ |  |
| rsLoRA scaling | ✅ | ✅ | Supported via LoraAdapterConfig::scaling() |

---

## Speculative Decoding

| Feature | Python | Rust | Notes |
|---|:---:|:---:|---|
| N-gram prompt lookup proposer | ✅ | ✅ | KMP-based proposer + greedy rejection sampling in CudaWorker; 6 E2E tests; --speculative-model ngram --num-speculative-tokens N |
| Draft model (separate small LM) | ✅ | ❌ |  |
| EAGLE / EAGLE3 draft heads | ✅ | ❌ |  |
| Medusa draft heads | ✅ | ❌ |  |
| Suffix decoding | ✅ | ❌ |  |
| Tree attention verification | ✅ | ❌ |  |

---

## Multimodal Input

| Feature | Python | Rust | Notes |
|---|:---:|:---:|---|
| Image input (single / batched) | ✅ | ❌ | Removed with CandleWorker; not ported to CudaWorker |
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
| OpenTelemetry tracing | ✅ | ✅ | `--features otel --otlp-traces-endpoint`; OTLP/gRPC export via tracing-opentelemetry |
| Sleep / wake (GPU memory release) | ✅ | ✅ | Level 1: free weights + KV cache + CUDA graphs; wake reloads from disk; POST /sleep /wake_up GET /is_sleeping /gpu_memory |
| RLHF pause / resume / weight update | ✅ | ❌ |  |
| Dynamic LoRA REST endpoints | ✅ | ❌ |  |
| Prefix cache reset endpoint | ✅ | ✅ |  |
| `/server_info` endpoint | ✅ | ✅ | Mirrors Python: vllm_config (text/json) + vllm_env + system_env; secrets filtered |
| Elastic DP scaling | ✅ | ❌ |  |

---

## Performance Optimizations

| Feature | Python | Rust | Notes |
|---|:---:|:---:|---|
| CUDA graphs (decode) | ✅ | ✅ | CudaWorker: BS=[1-32] with batch padding + in-graph argmax; MoE compatible |
| Fused RMS norm + residual add | ✅ | ✅ | CUDA kernel; vectorized 128-bit loads |
| Fused rotary embeddings | ✅ | ✅ | CUDA kernel |
| Fused MoE gating (top-k) | ✅ | ✅ | TRT-LLM topk_softmax kernel with GpuTensor FFI |
| Fused reshape-and-cache | ✅ | ✅ | CUDA kernel |
| Fused GPU sampling | ✅ | ✅ | CUDA kernel; full LogitsProcessor pipeline on GPU |
| cublasLt with plan caching | ✅ | ✅ | Plans cached by (M K N dtype has_bias); 32MB workspace (matches PyTorch default) |
| Weight dtype casting at load | ✅ | ✅ | GpuWeights casts F32→BF16/F16 via pinned host memory (matches Python torch_dtype auto-cast) |
| Triton kernels | ✅ | ❌ | Rust has no Triton equivalent |
| Torch.compile / inductor | ✅ | ❌ |  |
| Weight-only INT8/FP8 GEMM | ✅ | ❌ |  |
| Fused cross-entropy loss | ✅ | ❌ |  |
| NVTX profiling annotations | ✅ | ✅ | Rust: `--features profiling` |

---

## CUDA Compute Kernels

| Kernel | Python | Rust | Notes |
|---|:---:|:---:|---|
| `fused_add_rms_norm` | ✅ | ✅ | Vectorized 128-bit loads; wired into all model decoder layers |
| Rotary embedding (fused) | ✅ | ✅ |  |
| `reshape_and_cache` | ✅ | ✅ |  |
| MoE top-k gating | ✅ | ✅ | TRT-LLM topk_softmax kernel |
| GPU sampling (argmax + Gumbel-max + fused top-k/p/min-p) | ✅ | ✅ | In-graph for CUDA graphs; zero-copy D2D scatter |
| FlashAttention-2 (paged prefill+decode) | ✅ | ✅ | Direct FFI to vllm-flash-attn fork |
| Paged attention v1/v2 (PagedAttention) | ✅ | ❌ |  |
| `silu_and_mul` fused activation | ✅ | ✅ | Vectorized 128-bit loads; combined gate_up variant |
| `gelu_and_mul` fused activation | ✅ | ✅ | Vectorized 128-bit loads; combined gate_up variant |
| Fused MoE GEMM | ✅ | ✅ | WMMA tensor-core kernel (128/128/32); BF16/F16 only; perf gaps vs Triton: fixed tile sizes (2-3x some shapes) + WMMA vs native mma PTX (10-30%) + no GROUP_SIZE_M L2 grouping + no chunked processing (OOM risk large batches) |
| Marlin (INT4 GEMM) | ✅ | ✅ | W4A16 fused dequant+GEMM; 270 kernel instantiations (FP16/BF16 × GPTQ/AWQ); use_fp32_reduce=true |
| GGUF dequant kernels (k-quants) | ✅ | ✅ | llama.cpp-derived; BS=1 fused dequant-matvec + BS>1 Q8_1 dot products; Q4_0/Q4_1/Q5_0/Q5_1/Q8_0/Q2K-Q8K |
| BitsAndBytes NF4 dequant | ✅ | ✅ | Dequant-then-cuBLAS; double quantization supported; shared dequant scratch buffer |
| Embedding gather | ❌ | ✅ | Vectorized CUDA kernel |
| Split QKV | ❌ | ✅ | Separates fused QKV tensor on GPU |
| Fused QKV + RoPE | ❌ | ✅ | Combined split+rotary in one launch |
| Apply penalties (fused rep/freq/pres) | ✅ | ✅ | One block per request; no CPU fallback |
| Apply logit bias (CSR scatter-add) | ✅ | ✅ | Sparse CSR-packed; rebuilt only on batch change |
| Apply grammar mask (CSR allow-list) | ✅ | ✅ | Shared with allowed_token_ids |
| Log-softmax + top-K (fused logprobs) | ✅ | ✅ | Single kernel for logprob extraction from raw logits |
| Apply min_tokens (suppress EOS) | ✅ | ✅ | Scatter -inf to EOS/stop tokens |
| Cast to f32 | ✅ | ✅ |  |
| MoE align block size | ✅ | ✅ | Small + large batch paths (ported from Python vLLM); gap: no token_mask support (not used by Mixtral/Qwen MoE) |
| MoE sum (reduction) | ✅ | ✅ |  |
| Sigmoid-mul-add (shared expert gate) | ✅ | ✅ | Vectorized 128-bit loads |
| QK-norm + RoPE (fused) | ❌ | ✅ | Per-head RMS norm + NeoX RoPE; Qwen3 MoE / Gemma3 |
| MLA CUDA kernels (DeepSeek) | ✅ | ✅ | 6 fused kernels for non-absorbed MLA + YaRN RoPE |
| Fused recurrent GDN kernel | ❌ | ✅ | Qwen3-Next gated delta rule (fused_recurrent_gated_delta_rule) |
| QKVZ grouped-head split | ❌ | ✅ | Eliminates 8 CPU round-trips per GDN layer |
| Conv output split | ❌ | ✅ | Eliminates 3 CPU round-trips per GDN layer |
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
| OpenTelemetry tracing | ✅ | ✅ | `--features otel --otlp-traces-endpoint`; OTLP/gRPC export via tracing-opentelemetry |
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

