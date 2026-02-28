# vLLM Port to Rust: Full System Plan

## Implementation Progress

> **Last updated**: 2026-02-27 (Sampling gaps DONE — min_p, penalties, logprobs, logit_bias wired end-to-end)

| Phase | Status | Details |
|-------|--------|---------|
| **1a. Workspace setup** | **DONE** | 11-crate Cargo workspace, CI (GitHub Actions), build.rs stub, .gitignore |
| **1b. Core types** | **DONE** | `vllm-common` (65 tests), `vllm-config` (11 tests) |
| **1c. KV cache manager** | **DONE** | `vllm-core`: KVCacheBlock, FreeKVCacheBlockQueue, BlockPool, KVCacheManager (47 tests) |
| **1d. Scheduler** | **DONE** | `vllm-core`: SchedulerInterface trait, FCFS+Priority queues, full Scheduler (54 tests) |
| **1e. PyO3 bridge** | **DONE** | `vllm-pyo3`: RustScheduler exposed to Python via PyO3 |
| **2a. Serialization** | **DONE** | `vllm-protocol`: MsgpackEncoder/Decoder, codec module (12 tests) |
| **2b. ZMQ transport** | **DONE** | `vllm-protocol`: EngineInput/OutputSocket, FrontendInput/OutputSocket, message framing (12 tests) |
| **2c. Protocol messages** | **DONE** | `vllm-protocol`: EngineCoreRequestType, handshake types, UtilityOutput, PauseMode |
| **2d. Engine core loop** | **DONE** | `vllm-engine`: EngineCore, EngineCoreConfig, step/busy-loop, request management (13 tests) |
| **2e. Executor trait** | **DONE** | `vllm-engine`: Executor trait, NoopExecutor, ModelRunnerOutput (3 tests) |
| **2f. Engine core client** | **DONE** | `vllm-engine`: EngineCoreClient trait, InprocClient (10 tests) |
| **3a. OpenAI protocol types** | **DONE** | `vllm-serve`: ChatCompletion/Completion request/response types, streaming types, model types, error types (25 tests) |
| **3b. Axum HTTP server** | **DONE** | `vllm-serve`: axum routes for /v1/chat/completions, /v1/completions, /v1/models, /health, /version, CORS, SSE streaming (7 tests) |
| **3c. Async engine interface** | **DONE** | `vllm-serve`: AsyncEngine, request lifecycle management, streaming deltas, output routing (6 tests) |
| **3d. Tokenizer integration** | **DONE** | `vllm-serve`: Tokenizer wrapper (HF `tokenizers` crate), encode/decode, byte-level BPE (6 tests) |
| **3e. Output processing (detokenization)** | **DONE** | `vllm-serve`: IncrementalDetokenizer (sliding-window), check_stop_strings, per-request state, AsyncEngine integration (24 tests) |
| **4a. Executor + Worker abstraction** | **DONE** | `vllm-executor`: Worker trait, NoopWorker, UniProcExecutor, MultiprocExecutor, parallel state types (33 tests) |
| **4b. Expanded ModelRunnerOutput** | **DONE** | `vllm-engine`: expanded ModelRunnerOutput with logprobs, per-request indexing, from_token_map; expanded Executor trait with sleep/wake_up/check_health (5 new tests) |
| **5a. Tensor abstraction** | **DONE** | `vllm-model`: candle-core wrapper, DType/Device helpers, tensor creation, from_raw_bytes, sharding, TensorInfo (12 tests) |
| **5b. SafeTensors weight loading** | **DONE** | `vllm-model`: SafeTensorsFile, SafeTensorsIndex, ModelWeights (single + sharded), HfModelConfig (18 tests) |
| **5c. CUDA kernel FFI stubs** | **DONE** | `vllm-kernels`: AttentionKernels, CacheKernels, NormKernels, ActivationKernels, RotaryKernels traits + CPU impls (12 tests) |
| **5d. Layer abstractions** | **DONE** | `vllm-model`: Linear, ColumnParallelLinear, RowParallelLinear, RmsNorm, Embedding, VocabParallelEmbedding, RotaryEmbedding, Activation functions (26 tests) |
| **6a. Model definition framework** | **DONE** | `vllm-models`: Model trait, ModelRegistry, Sampler (greedy+temperature+top-k/p), ScaledDotProductAttention (3+7+7 tests) |
| **6b. LLaMA model** | **DONE** | `vllm-models`: LlamaConfig, LlamaMLP, LlamaAttention (GQA), LlamaDecoderLayer, LlamaModel, LlamaForCausalLM, weight loading, tied embeddings (7 tests) |
| **6b. Priority 1 models (cont.)** | **DONE** | Qwen2 (reuses LLaMA), Qwen3 (reuses LLaMA), Gemma2 (GELU, GemmaRmsNorm, 4 norms, softcap), Phi-3 (alias), DeepSeek V2/V3 (MLA+MoE+YaRN), **Command R** (CohereLayerNorm, parallel attn+MLP, logit scaling, interleaved RoPE); `vllm-model`: GemmaRmsNorm, CohereLayerNorm, YaRN RoPE (27+6 new tests) |
| 6c. Priority 2 models | Not started | GPT-NeoX, GPT-J, Falcon, BLOOM, MPT, StarCoder, multimodal |
| 6d. Priority 3 models | Not started | Long-tail architectures |
| 6e. Python fallback | Not started | PyO3 bridge for unported models |
| **7: Standalone Binary** | **DONE** | `vllm-cli` crate: `vllm serve/bench/convert`, CandleWorker, HF Hub download (sharded), Prometheus metrics, tracing, Dockerfiles |
| **8a. Working generation** | **DONE** | Worker token buffer, correct position IDs, stop criteria (max_tokens/EOS/stop_token_ids), wire SamplingParams to Sampler |
| **8b. Chat templates** | **DONE** | minijinja-based chat template rendering from tokenizer_config.json |
| **8b+. n param + multi-prompt** | **DONE** | `n` parameter for chat/completion/streaming, multi-prompt completions (P×N choices) |
| **8c. KV cache in forward pass** | **DONE** | KvCache type, Model::forward() accepts cache, per-layer KV concat, CandleWorker per-request caches, O(1) decode (5 new tests) |
| **9a. Metal Tier 1** | **DONE** | `--features metal` enables Apple Silicon GPU, `--device auto`, pre-transposed weights (6-10x speedup), RoPE F32 fix, ~47ms/tok on 1.7B model |
| **8e. Native dtype inference** | **DONE** | `--dtype auto` (default) reads torch_dtype from config.json, f16/bf16 throughout forward pass, f32 upcast for norm variance + softmax, drop accelerate (512 tests) |
| **8d. Paged KV cache** | **DONE** | KvBlockPool (block-level tensor pool), gather/scatter around forward passes, scheduler block IDs → tensors, CandleWorker dual-path (paged + legacy fallback) (525 tests) |
| **8d+. Direct block KV** | **DONE** | KvCacheStorage/LayerKvHandle abstractions, attention layers interact with block pool directly, deferred scatter via flush(), no CandleWorker gather/scatter loops (525 tests) |
| **8f. GGUF quantized models** | **DONE** | GgufFile loader, GGUF metadata→HfModelConfig, tensor name mapping, QuantizedLinear (QMatMul), QuantizedLlamaForCausalLM with interleaved RoPE (GGML convention), GgufModelFactory registry, CandleWorker GGUF auto-detection (local/HF), `--gguf-file` CLI arg, HF repo auto-select (prefers Q4_K_M), base-repo tokenizer fallback. Verified Llama-3.1-8B-Instruct-Q4_K_M producing coherent output. (546 tests) |
| **9b. Paged attention infra** | **DONE** | PagedKvBlockRefs, PendingWrite enum (FullScatter/NewToken), paged_decode_attention() reference impl, attention_with_cache() unified helper (deduplicates 3 model attention layers). Rust per-block loop is correct but slower than gather — awaits fused Metal kernel in Tier 2. (552 tests) |
| **10a. MLX crate skeleton** | **DONE** | `vllm-mlx` crate: MlxWorker (Worker trait), MlxModel trait, MlxModelRegistry, MLX KV cache (8 tests) |
| **10b. MLX LLaMA model** | **DONE** | LLaMA/Mistral/Qwen2 using mlx-rs nn primitives (Linear, RmsNorm, Rope, Embedding, SDPA), safetensors weight loading, prefill+decode with KV cache |
| **10c. MLX benchmark + tuning** | **DONE** | Timing instrumentation (per-step graph/sample/total ms), eval fusion (1 eval instead of 2), `vllm bench` decode loop with TTFT/ITL/p50/p99/throughput, MLX backend support in bench, --warmup. Baseline: 13.6ms ITL, 70 tok/s on SmolLM2-1.7B (564 tests) |
| **10d. MLX quantized models** | **DONE** | QuantizedLinear/QuantizedEmbedding with weight/scales/biases triplets, QuantConfig parsed from config.json `"quantization"` field, MlxModelRegistry quantized_models map with get_factory(arch, quantized), MlxWorker auto-detects quantization. Direct struct construction from loaded arrays (no wasted quantize step). (568 tests) |
| **10d+. Multi-EOS + logging** | **DONE** | Support all eos_token_ids from config.json (matching Python vLLM), request-finished log includes finish_reason/stop_reason, HTTP handler logs max_completion_tokens, fixed vacuous EOS tests (553 tests) |
| **10e. MLX additional models** | **DONE** | Gemma v1 (2 norms, GemmaRmsNorm +1 offset, gelu_approximate, tied embeddings), Gemma2 (4 norms, logit softcapping), Phi-3 (fused qkv_proj + gate_up_proj, split_axis), Qwen2 bias loading, MlxEmbedTokens/MlxLmHead mixed-precision enums (582 tests) |
| **11a. Memory profiling + `--gpu-memory-utilization`** | **DONE** | Real memory detection via `sysinfo` crate (CandleWorker: available RAM, MlxWorker: total unified memory), `--gpu-memory-utilization` CLI flag (default 0.9, env `VLLM_GPU_MEMORY_UTILIZATION`) on serve + bench, replaces hardcoded 4/8 GiB stubs (570 tests) |
| **8g. Sampling gaps** | **DONE** | Unified `Sampler::sample_one()`: min_p, repetition/frequency/presence penalties, logit_bias, logprobs (top-N). `LogprobsOutput`/`TokenLogprob` in `vllm-common`. CPU-side penalty application in both CandleWorker + MlxWorker. Logprobs propagated through `EngineCoreOutput` → API responses (chat + completion, streaming + non-streaming). 12 new sampler unit tests. (629 tests) |
| 12a. Tool calling protocol | Not started | `tools`/`tool_choice` fields in chat completion request, `tool_calls` in assistant response, chat template integration for tool definitions |
| 12b. Tool call response parsing | Not started | Detect model-emitted tool calls (special tokens, JSON blocks), parse into structured `ToolCall` objects, streaming tool call deltas |
| 12c. Structured output / constrained decoding | Not started | `response_format` (json_object, json_schema), grammar-guided logit masking via `llguidance` or similar, schema-to-grammar compilation |
| 9c. Metal Tier 2 (legacy candle) | Superseded | Custom MSL fused kernels approach superseded by MLX backend. Use `--features candle-metal` for legacy path |
| 9d. Metal Tier 3 | Partially superseded | UMA-aware KV cache, memory pressure handling. Zero-copy weight loading and quantization are handled natively by MLX backend (Phase 10d) |

### Phase Totals

| Phase | New Lines | Files | New Tests | Total Tests | Notes |
|-------|----------:|------:|----------:|------------:|-------|
| 1 | ~7,200 | 27 new | 177 | 177 | |
| 2 | ~2,300 | 8 new | 52 | 229 | |
| 3a-c | ~2,450 | 5 new | 32 | 261 | |
| 3d-e | ~1,150 | 2 new | 30 | 291 | |
| 4 | ~1,900 | 5 new | 38 | 329 | |
| 5 | ~3,200 | 14 new | 68 | 397 | |
| 6a-b | ~1,630 | 5 new | 24 | 421 | |
| 6c | ~1,290 | 2 new + mods | 15 | 436 | |
| 7 | ~1,400 | 12 new + 9 mod | 28 | 464 | |
| 8a | ~250 | 4 mod | 10 | 474 | |
| 8b | ~300 | 1 new + 3 mod | 10 | 484 | |
| 8b+ | ~300 | 2 mod | 15 | 499 | |
| 8c | ~150 | 5 mod | 5 | 504 | |
| 8e | ~240 | 9 mod | 8 | 512 | |
| 9a | ~80 | 7 mod | 2 | 506 | Metal Tier 1, ~47ms/tok |
| 8f | ~1,528 | 3 new + 9 mod | 21 | 546 | GGUF, interleaved RoPE |
| 9b | ~720 | 6 mod | 6 | 552 | Paged decode attention |
| 10a+b | ~1,200 | 4 new + 3 mod | 8 | 560 | MLX lazy eval backend |
| 10c | ~350 | 3 mod | 4 | 564 | Eval fusion, bench decode |
| 10d | ~500 | 1 new + 3 mod | 4 | 568 | MLX quantized models |
| 10d+ | ~125 | 6 mod | 1 | 569 | Multi-EOS, logging |
| 11a | ~95 | 8 mod | 1 | 570 | Memory detection, --gpu-memory-utilization |
| 10e | ~2,380 | 3 new + 3 mod | 13 | 582 | Gemma v1/2, Phi-3, Qwen2 bias, mixed-precision |
| 6b | ~2,530 | 2 new + 6 mod | 17 | 599 | DeepSeek V2/V3 MLA+MoE+YaRN, Qwen3 QK norms |
| 6b | ~850 | 2 new + 4 mod | 18 | 617 | Command R (candle+MLX float+quantized) |
| 8g | ~450 | 10 mod | 12 | 629 | Sampling gaps (min_p, penalties, logprobs, logit_bias) |
| **Total** | **~32,000** | **94 files** | **629** | **629** | **0 clippy errors** |

### Known limitations / follow-ups
- **MLX YaRN RoPE**: The MLX backend uses `nn::Rope` which doesn't apply YaRN frequency corrections. Models with `rope_scaling` (e.g., Qwen3 with YaRN factor=4.0, DeepSeek V2 with factor=40.0) will generate correctly within the original context window but won't have correct positional encoding beyond it. Fix: either implement a custom MLX RoPE that pre-applies YaRN corrections, or upstream YaRN support to mlx-rs `nn::Rope`.
- **DeepSeek V2 MoE**: Token-by-token expert routing is a correct reference implementation but serializes expert execution. For production, batch tokens by expert assignment to maximize GPU utilization (expert parallelism).
- **DeepSeek V2 latent KV caching**: Currently caches full expanded K/V after kv_b_proj. Could instead cache the compressed `[kv_lora_rank + qk_rope_head_dim]` latent per token — much smaller KV cache at the cost of re-expanding during decode. This is how the Python vLLM optimizes it.
- **Qwen3 QK norms**: Optional per-head q_norm/k_norm (RMSNorm on [head_dim]) auto-detected from safetensors weight presence. Applied after Q/K projection and reshape, before RoPE. Both float and quantized MLX LLaMA attention paths support this. The candle LLaMA path does not yet have QK norm support (would need similar changes if running Qwen3 on candle).

### Phase 10d plan: MLX quantized models

**Strategy**: Load mlx-community format models — pre-quantized safetensors with MLX's native 4-bit group quantization (packed u32 weights + separate f16 scales/biases arrays). No GGUF parsing, no runtime quantization. MLX's `mlx_load_safetensors()` uses mmap under the hood; on Apple Silicon UMA with `storageModeShared` Metal buffers, weights go from disk → page cache → GPU-accessible memory with zero copies. The `quantized_matmul` Metal kernel reads the packed bits and dequantizes in-register during matmul — weights are never fully materialized as fp16/fp32.

**Why not GGUF for MLX?** GGUF uses GGML's packed block formats (Q4_0, Q4_K_M, etc.) which are structurally incompatible with MLX's group quantization format. Loading GGUF into MLX would require either (a) bit-level format conversion per quant type, or (b) dequantize-then-requantize through a different scheme. The mlx-community ecosystem on HuggingFace has thousands of pre-quantized models in MLX's native format, uses HF-standard tensor names, and half-split RoPE convention — so there's no practical need for GGUF on the MLX path.

**mlx-community model format** (e.g., `mlx-community/Meta-Llama-3.1-8B-Instruct-4bit`):
- Regular safetensors files with HF-standard names
- Each linear layer has 3 tensors: `*.weight` (packed u32), `*.scales` (f16), `*.biases` (f16)
- `config.json` contains `"quantization": {"group_size": 64, "bits": 4}` field
- Norms stay full precision (f32), embedding/lm_head also quantized with same format
- Standard `tokenizer.json` / `tokenizer_config.json` (no base-repo fallback needed)

**What to build**:

1. **Quantized LLaMA model** (`vllm-mlx/src/models/quantized_llama.rs` or extend existing): Uses `nn::QuantizedLinear` for Q/K/V/O projections and MLP linear layers. Uses `nn::QuantizedEmbedding` for embed_tokens. `lm_head` uses `QuantizedEmbedding::as_linear()` if tied, else `QuantizedLinear`. Norms stay as `nn::RmsNorm` (f32). RoPE stays as `nn::Rope` (half-split HF convention, same as existing MLX LLaMA).

2. **Weight loading for quantized triplets**: Extend `load_safetensors_weights()` + `assign_weight()` pattern. For each `QuantizedLinear`, assign `.inner.weight` (packed u32), `.scales`, and `.biases` from the loaded `HashMap<String, Array>`. The naming convention matches HF standard: `model.layers.0.self_attn.q_proj.weight/scales/biases`.

3. **Quantization detection from config.json**: Parse `HfModelConfig.extra["quantization"]` — if present, extract `group_size` and `bits`, route to quantized model factory.

4. **Registry routing**: `MlxModelRegistry` needs to select quantized vs non-quantized for the same arch (`"LlamaForCausalLM"`). Options: (a) separate `quantized_models` map keyed by arch, checked first when config has `"quantization"`; (b) single factory that inspects config and returns the right variant. Approach (a) mirrors candle's `gguf_models` pattern.

5. **MlxWorker changes**: `load_model()` checks config for `"quantization"` field before looking up factory. No GGUF path resolution needed — all loading goes through existing safetensors/HF Hub download path. Dtype for KV cache: likely f16 (QuantizedLinear output may be f16 on Metal, unlike candle QMatMul which always outputs f32 on CPU).

**What's free (no work needed)**:
- Zero-copy mmap weight loading — MLX's `load_safetensors()` already does this
- HF Hub download — existing `MlxWorker` download path works for mlx-community repos
- RoPE — half-split convention, same as existing MLX LLaMA (no interleaved GGML handling)
- Tensor name convention — HF standard, same as non-quantized safetensors

### Phase 11a plan: Memory profiling + `--gpu-memory-utilization`

**Problem**: Both `CandleWorker` and `MlxWorker` return hardcoded memory constants (4 GiB and 8 GiB respectively) from `determine_available_memory()`. Python vLLM queries real device memory, runs a dummy forward pass to measure peak activation + weight memory, and gives the remainder (scaled by `gpu_memory_utilization`, default 0.9) to KV cache. Our Rust port skips all of this, which means KV cache sizing is wrong on every machine.

**Python reference** (`vllm/v1/worker/gpu_worker.py`):
1. `MemorySnapshot` calls `torch.cuda.mem_get_info()` → `(free, total)`
2. `requested_memory = ceil(total * gpu_memory_utilization)`
3. `profile_run()` runs a dummy forward at `max_num_batched_tokens`
4. `memory_profiling` context measures weights + peak activation + non-torch overhead
5. `available_kv_bytes = requested_memory - non_kv_cache_memory`

**What to build**:

1. **`--gpu-memory-utilization` CLI flag** (`vllm-cli/src/args.rs`):
   - Float between 0.0 and 1.0, default `0.9`
   - Passed through to `init_cache()` / `compute_num_blocks()`
   - Env var fallback: `VLLM_GPU_MEMORY_UTILIZATION` (Python doesn't have this but it's convenient)

2. **Real memory detection per backend**:
   - **CPU** (`CandleWorker`): Use `sysinfo` crate → `System::new_with_specifics()` → `total_memory()` and `available_memory()`. Report available (not total) since other processes share RAM.
   - **MLX/Metal** (`MlxWorker`): Use `sysctl("hw.memsize")` for total physical memory (Apple Silicon unified memory). Alternatively `mlx_rs` may expose `metal::device_info()` — check API. On UMA, GPU and CPU share the same pool so total physical memory is the right number.
   - **CUDA** (`CandleWorker` with CUDA device): `cuMemGetInfo_v2` via `cudarc` crate (already a candle dependency) → `(free, total)`. Report free memory at the time of query.

3. **Memory profiling pass** (stretch goal — can defer):
   - Python runs a dummy forward to measure peak activation memory, then subtracts it. This is important for large models where activations consume significant GPU memory.
   - Simpler Rust approach: after `load_model()`, measure memory delta (model weights size is known from safetensors metadata). Subtract weight memory from total available before computing KV blocks.
   - Full dummy-forward profiling can be added later when we have CUDA memory tracking.

4. **Updated `compute_num_blocks()`** (`vllm-cli/src/init.rs`):
   - Accept `gpu_memory_utilization: f64` parameter
   - `cache_memory = (available_bytes as f64 * gpu_memory_utilization) as usize` (currently hardcoded to 0.9)
   - Optionally subtract estimated model weight size if known

5. **Tests**:
   - Unit test: `determine_available_memory()` returns > 0 and reasonable value (< 1 TiB, > 512 MiB)
   - Unit test: `--gpu-memory-utilization 0.5` produces ~half the blocks of default 0.9
   - Unit test: `compute_num_blocks` with explicit utilization fraction

**Key files**: `vllm-cli/src/args.rs`, `vllm-cli/src/init.rs`, `vllm-executor/src/candle_worker.rs`, `vllm-mlx/src/worker.rs`, `vllm-executor/src/worker.rs` (Worker trait — no change needed, already returns `usize`)

**Dependencies**: None — independent of all other phases. New crate dep: `sysinfo` (lightweight, cross-platform).

**Effort**: ~1 session for real memory detection + CLI flag. Dummy forward profiling is a stretch goal.

### What was built in Phase 1

**`vllm-common`** (5 files, 65 tests):
- `error.rs` — `VllmError` enum (Validation, RequestNotFound, Engine, Scheduler, Serialization, Internal), `VllmResult<T>`
- `sampling.rs` — `SamplingParams` (19 fields matching Python incl. `logit_bias`), `SamplingType`, `RequestOutputKind`, validation, `TokenLogprob`, `LogprobsOutput`
- `request.rs` — `Request` struct (20 fields), `RequestStatus` enum (11 variants), `Ord` impl for priority scheduling
- `engine_io.rs` — `FinishReason`, `EngineCoreRequest`, `EngineCoreOutput`, `EngineCoreOutputs`, `EngineCoreEvent`, `StopReason`

**`vllm-config`** (5 files, 11 tests):
- `scheduler.rs` — `SchedulerConfig`, `SchedulerPolicy` (Fcfs/Priority), `RunnerType`
- `cache.rs` — `CacheConfig`, `KVCacheConfig`, `KVCacheSpec` (7 attention types), `KVCacheGroupSpec`, `KVCacheTensor`
- `parallel.rs` — `ParallelConfig` (18 fields)
- `model.rs` — `ModelConfig`, `ModelDType`, `AttnType`

**`vllm-core`** (10 files, 101 tests):
- `kv_cache_block.rs` — `KVCacheBlock` (arena-indexed linked list), `BlockHash`/`BlockHashWithGroupId` types, hash packing
- `free_block_queue.rs` — `FreeKVCacheBlockQueue` doubly-linked list with sentinel nodes, O(1) ops
- `block_pool.rs` — `BlockPool` (arena + free list + `BlockHashToBlockMap` with `SmallVec<[usize; 1]>` optimization)
- `kv_cache_manager.rs` — `KVCacheManager` (prefix cache hits, block allocation, per-request tracking)
- `scheduler/interface.rs` — `SchedulerInterface` trait, `PauseState` enum
- `scheduler/output.rs` — `SchedulerOutput`, `NewRequestData`, `CachedRequestData`
- `scheduler/request_queue.rs` — `FCFSRequestQueue` (VecDeque), `PriorityRequestQueue` (BinaryHeap)
- `scheduler/core.rs` — `Scheduler` (full 3-phase algorithm: RUNNING→WAITING→build output), `KVCacheManagerOps` trait, `SimpleBlockTracker`

**`vllm-pyo3`** (1 file):
- `RustScheduler` PyO3 class — drop-in Python replacement with `schedule()`, `add_request()`, `finish_requests()`, pause/resume, prefix cache reset

**CI** (`.github/workflows/rust.yml`):
- Jobs: check, test, clippy, fmt, pyo3-build

### What was built in Phase 2

**`vllm-protocol`** (4 files, 26 tests):
- `codec.rs` — `MsgpackEncoder`/`MsgpackDecoder` wrapping `rmp-serde`, `encode()`/`decode()` convenience functions
- `messages.rs` — `EngineCoreRequestType` (Add/Abort/StartDpWave/Utility/ExecutorFailed), `HandshakeHello`/`HandshakeReady`, `EngineZmqAddresses`, `EngineHandshakeMetadata`, `UtilityRequest`/`UtilityOutput`, `PauseMode`, `ENGINE_CORE_DEAD` sentinel
- `transport.rs` — `EngineInputSocket` (DEALER), `EngineOutputSocket` (PUSH), `FrontendInputSocket` (ROUTER), `FrontendOutputSocket` (PULL), `build_request_message()`, `parse_request_type()`

**`vllm-engine`** (5 files, 26 tests):
- `engine_core.rs` — `EngineCore` struct (scheduler + executor orchestration), `EngineCoreConfig`, `step()` (schedule→execute→update), `run_busy_loop()`, request management (add/abort/queue), pause/resume, prefix cache reset, shutdown
- `executor.rs` — `Executor` trait (execute_model, initialize_cache, determine_available_memory, shutdown), `ModelRunnerOutput`, `NoopExecutor` (dummy token generator for testing)
- `core_client.rs` — `EngineCoreClient` trait (get_output, add_request, abort_requests, pause/resume, shutdown), `InprocClient` (in-process direct-call implementation)
- `error.rs` — `EngineError` (Scheduler/Executor/RequestNotFound/Shutdown/Transport/Config)

### What was built in Phase 3a-c

**`vllm-serve`** (5 files, 32 tests):
- `protocol.rs` — OpenAI-compatible API types: `ChatCompletionRequest`/`Response`, `CompletionRequest`/`Response`, streaming variants (`ChatCompletionStreamResponse`, `CompletionStreamResponse`), `UsageInfo`, `ErrorResponse`, `ModelCard`/`ModelList`, `DeltaMessage`, `StopCondition`, `CompletionPrompt` (25 tests)
- `server.rs` — axum HTTP server with routes: `POST /v1/chat/completions`, `POST /v1/completions`, `GET /v1/models`, `GET /health`, `GET /version`; CORS middleware via tower-http; SSE streaming for chat completions with `[DONE]` sentinel (7 tests)
- `engine.rs` — `AsyncEngine` async wrapper around `EngineCoreClient`: request lifecycle (submit → poll → respond), streaming via `mpsc` channels with `StreamDelta`, output routing, request conversion (chat/completion → `EngineCoreRequest`), `SamplingParams` construction (6 tests)
- `error.rs` — `ServeError` enum with `IntoResponse` impl for axum error handling
- `lib.rs` — Module declarations

**Notable changes to existing crates:**
- `vllm-core`: Added `Send` bounds to `RequestQueue` and `KVCacheManagerOps` traits (required for async/multi-threaded serving)

### What was built in Phase 3d-e

**`vllm-serve`** (2 new files, 30 new tests):
- `tokenizer.rs` — `Tokenizer` wrapper around HuggingFace `tokenizers` crate: `from_file()`, `encode()`, `decode()`, `id_to_token()`, `vocab_size()`, `eos_token_id()`, `is_special_token()`, special-ID caching (6 tests)
- `detokenizer.rs` — `IncrementalDetokenizer` with sliding-window decode (re-decodes ~6 tokens of context to handle tokenizer-dependent boundaries), `check_stop_strings()` with windowed search, stop-buffer for streaming, delta/cumulative output modes, `min_tokens` enforcement (18 tests)

**Notable changes to existing files:**
- `engine.rs` — `AsyncEngine` now accepts optional `Arc<Tokenizer>` via `with_tokenizer()` constructor; `RequestState` holds per-request `IncrementalDetokenizer`; `process_output()` drives incremental detokenization and stop-string detection; `StreamDelta` carries `text: Option<String>` for real detokenized text; `chat_to_engine_request()`/`completion_to_engine_request()` use real tokenization when available (6 new tests)
- `server.rs` — SSE `stream_chat_response()` uses `delta.text` for real decoded content
- `vllm-common/sampling.rs` — Added `include_stop_str_in_output: bool` to `SamplingParams`
- New dependency: `tokenizers = "0.22"` (HuggingFace tokenizers crate — same Rust library that powers the Python `tokenizers` package)

### What was built in Phase 4

**`vllm-executor`** (5 new files, 33 new tests):
- `error.rs` — `ExecutorError` enum (WorkerInit, WorkerExecution, WorkerUnhealthy, WorkerDied, Communication, Shutdown, Config, Timeout), `ExecutorResult<T>`
- `worker.rs` — `Worker` trait (init_device, load_model, initialize_cache, determine_available_memory, execute_model, compile_or_warm_up_model, check_health, sleep, wake_up, shutdown), `WorkerConfig`, `NoopWorker` (7 tests)
- `uniproc.rs` — `UniProcExecutor` wrapping a single `Worker`, implements `Executor` trait, full initialization sequence (init_device → load_model → compile), sleep/wake tag management, integration test with `EngineCore` (11 tests)
- `multiproc.rs` — `MultiprocExecutor` managing multiple workers as tokio tasks, `WorkerRequest`/`WorkerResponse` RPC types, `WorkerHandle` with oneshot channels, `collective_rpc_blocking()` using `block_in_place`, output-rank selection for TP/PP (6 tests)
- `parallel.rs` — `ParallelGroup` (name, world_size, rank, ranks, next/prev rank), `ResolvedParallelConfig` (TP+PP+DP groups, single_gpu, tensor_parallel, tensor_pipeline_parallel constructors, is_driver, is_output_rank, output_rank), serde support (9 tests)

**Notable changes to existing files:**
- `vllm-engine/executor.rs` — Expanded `ModelRunnerOutput` to match Python's structure: `req_ids`, `req_id_to_index`, `sampled_token_ids` per request, `logprobs` (with `TokenLogprob` and `LogprobsOutput` types), `prompt_logprobs_dict`, `draft_token_ids`; added `from_token_map()`, `get_tokens()`, `num_requests()`, `is_empty()` helpers; expanded `Executor` trait with `check_health()`, `sleep()`, `wake_up()` methods (5 new tests)
- `vllm-engine/engine_core.rs` — Updated `update_from_output()` to use new `ModelRunnerOutput::get_tokens()` API

### What was built in Phase 5

**`vllm-model`** (8 new files, 56 new tests):
- `tensor.rs` — Tensor/device abstractions wrapping `candle-core`: DType helpers (dtype_size, dtype_from_str, dtype_to_str), tensor creation (zeros, ones, from_slice, from_raw_bytes), tensor sharding (shard_tensor for TP), TensorInfo, ModelError/ModelResult error types (12 tests)
- `weight.rs` — SafeTensors weight loading: SafeTensorsFile (open, load_tensor, load_tensor_cast, tensor_infos, load_all), SafeTensorsIndex (sharded model.safetensors.index.json), ModelWeights (from_dir, from_single_file, from_index, get, get_cast, take, total_size_bytes), HfModelConfig (config.json parser with architectures, dimensions, norm_eps, rope_theta, extra fields) (18 tests)
- `layers/linear.rs` — Linear layer (weight+bias, forward via matmul), ColumnParallelLinear (shard weight dim 0), RowParallelLinear (shard weight dim 1), weight loading from ModelWeights (8 tests)
- `layers/norm.rs` — RmsNorm (mean(x^2) -> sqrt -> recip -> scale), load from weights (6 tests)
- `layers/embedding.rs` — Embedding (token lookup), VocabParallelEmbedding (sharded vocab dim 0) (4 tests)
- `layers/activation.rs` — Activation enum (Silu, Gelu, GeluErf, Relu, QuickGelu), standalone functions, candle Module impl (6 tests)
- `layers/rotary.rs` — RotaryEmbedding (precomputed cos/sin cache, apply to q/k, NeoX-style rotation) (4 tests)

**`vllm-kernels`** (6 new files, 12 new tests):
- `error.rs` — KernelError/KernelResult types
- `attention.rs` — AttentionKernels trait (paged_attention_v1/v2), CpuAttentionKernels stub (2 tests)
- `cache.rs` — CacheKernels trait (reshape_and_cache, swap_blocks), CpuCacheKernels stub (2 tests)
- `norm.rs` — NormKernels trait (rms_norm, fused_add_rms_norm), CpuNormKernels implementation (2 tests)
- `activation.rs` — ActivationKernels trait (silu_and_mul, gelu_and_mul, gelu_new_and_mul), CpuActivationKernels implementation (3 tests)
- `rotary.rs` — RotaryKernels trait (rotary_embedding), CpuRotaryKernels implementation (3 tests)

**New dependencies:**
- `candle-core = "0.9"` — Tensor library (CPU + CUDA + Metal backends, same ecosystem as HuggingFace); `metal` + `accelerate` features behind `--features metal`
- `safetensors = "0.7"` — SafeTensors file format (same Rust library that powers the Python package)
- `half = "2.7"` — f16/bf16 types for tensor data
- `memmap2 = "0.9"` — Memory-mapped file I/O for large weight files
- `tempfile = "3"` — Temporary files for testing

### What was built in Phase 6a-b

**`vllm-models`** (5 new files, 24 new tests):
- `lib.rs` — `Model` trait (forward: input_ids + positions → logits), `ModelFactory` type alias for registry
- `registry.rs` — `ModelRegistry` mapping HuggingFace architecture names to factory functions, default registry with LLaMA + Mistral (3 tests)
- `sampler.rs` — `Sampler` with greedy (argmax), temperature, top-k, top-p, min-p sampling; unified `sample_one()` entry point with penalty application (repetition/frequency/presence), logit bias, and logprobs computation; probability-based random selection (19 tests)
- `attention.rs` — `scaled_dot_product_attention` with causal masking, GQA support (repeat_kv), softmax helper, causal mask generation (7 tests)
- `llama.rs` — Full LLaMA/Mistral model implementation: `LlamaConfig` (from HfModelConfig), `LlamaMLP` (gate+up SiLU-gated FFN), `LlamaAttention` (Q/K/V projections + RoPE + scaled dot-product attention + output projection, GQA support), `LlamaDecoderLayer` (pre-norm attention + post-norm MLP with residuals), `LlamaModel` (embedding + N layers + final RMS norm), `LlamaForCausalLM` (model + lm_head, tied embeddings support), `create_llama` factory function (7 tests)

**Key features:**
- Complete forward pass: input_ids → embedding → transformer layers → logits
- Weight loading from safetensors files with standard HF naming conventions
- GQA (grouped-query attention) support
- Tied word embeddings support
- Causal masking for prefill sequences
- RoPE integration via existing `RotaryEmbedding`
- Mistral architecture reuse (same code path as LLaMA)

### What was built in Phase 6c

**`vllm-models`** (2 new files, 11 new tests):
- `qwen2.rs` — Qwen2 model implementation: `Qwen2Config` (from HfModelConfig, rope_theta default = 1M), delegates to `LlamaForCausalLM` since architecture is identical (QKV bias handled automatically by existing layer loading), `create_qwen2` factory function (4 tests)
- `gemma2.rs` — Full Gemma 2 model implementation: `Gemma2Config` (query_pre_attn_scalar, attn_logit_softcapping, final_logit_softcapping, attention_bias from config.json extra fields), `Gemma2MLP` (GELU-tanh gated FFN), `Gemma2Attention` (custom query scaling, soft capping support), `Gemma2DecoderLayer` (4 GemmaRmsNorms per layer), `Gemma2Model` (embedding × sqrt(hidden_size) normalization), `Gemma2ForCausalLM` (tied embeddings, logit soft capping via tanh), `create_gemma2` factory function (7 tests)

**`vllm-model`** (modifications, 4 new tests):
- `layers/norm.rs` — Added `GemmaRmsNorm` (RMSNorm variant with +1 weight offset: `y = x * (1 + w) / rms(x)`), load from weights, zeros constructor (4 tests)
- `layers/mod.rs` — Re-exported `GemmaRmsNorm`

**Registry additions:**
- `Qwen2ForCausalLM` → `create_qwen2` (Qwen2 factory with 1M rope_theta default)
- `Phi3ForCausalLM` → `create_llama` (Phi-3 is architecturally identical to LLaMA)
- `Gemma2ForCausalLM` → `create_gemma2` (Gemma2 factory)

**Key features:**
- Qwen2 reuses LLaMA since architecture is identical (only QKV bias and rope_theta differ)
- Gemma2: GELU(tanh) activation, GemmaRMSNorm (+1 weight offset), 4 norms per layer (input, post-attention, pre-feedforward, post-feedforward), query_pre_attn_scalar scaling, attention/logit soft capping, embedding normalization
- Phi-3 registered as LLaMA alias (Python vLLM also inherits directly from LlamaForCausalLM)
- 6 model architectures now registered: LLaMA, Mistral, Qwen2, Phi-3, Gemma2

### What was built in Phase 8a

**Working generation loop** (4 modified files, 10 new tests):

**Worker token buffer** (`candle_worker.rs`):
- `HashMap<String, Vec<u32>>` token buffer per request — stores prompt + generated tokens
- `HashMap<String, SamplingParams>` stores per-request sampling params from scheduler output
- On new request: stores prompt tokens and sampling params in buffers
- On decode step: feeds last token from buffer (not placeholder 0) as model input
- After sampling: appends new token to buffer
- On finished requests: cleans up buffer entries from `finished_req_ids`

**Correct position IDs** (`candle_worker.rs`):
- Prefill: positions = `num_computed_tokens..num_computed_tokens + num_tokens` (accounts for prefix cache)
- Decode: position = `num_computed_tokens` from `CachedRequestData` (continues from end of cached sequence)

**Stop criteria** (`engine_core.rs`):
- `eos_token_id: Option<u32>` in `EngineCoreConfig` and `EngineCore`
- `check_stop_criteria()` method checks: max_tokens → `FinishReason::Length`, EOS token → `FinishReason::Stop`, `stop_token_ids` → `FinishReason::Stop`
- Respects `ignore_eos` flag in `SamplingParams`
- `update_from_output()` now appends tokens to request state via `scheduler.append_output_tokens()`, checks stop criteria, and calls `finish_requests()` for finished requests

**SamplingParams wired to Sampler** (`candle_worker.rs`):
- Uses `Sampler` from `vllm-models` for non-greedy requests (temperature > 0)
- Supports temperature, top-k, top-p sampling per request
- Keeps argmax fast path for greedy (temperature ≈ 0)
- `Sampler` created per `execute_model()` call to avoid `Send` issues (ThreadRng is !Send)

**Scheduler request accessors** (`scheduler/core.rs`):
- `get_request(req_id)` / `get_request_mut(req_id)` — access request state for stop criteria
- `append_output_tokens(req_id, tokens)` — updates both canonical `requests` map and `running` list

**EOS token extraction** (`init.rs`):
- Parses `eos_token_id` from HfModelConfig extra fields (handles both int and array formats)
- Passes through to `EngineCoreConfig`

### What was built in Phase 8b

**Chat template module** (`vllm-serve/src/chat_template.rs`, 1 new file, 10 new tests):
- `ChatTemplate` struct: holds Jinja2 template string, optional BOS/EOS token strings
- `from_tokenizer_config()`: parses `tokenizer_config.json` for `chat_template` field (handles both string and array-of-objects formats)
- `apply()`: renders template with `minijinja` engine, passing `messages`, `add_generation_prompt`, `bos_token`, `eos_token`
- `TemplateMessage` struct for template context
- `raise_exception` function for Jinja2 compatibility (many HF templates use it)
- Handles `bos_token`/`eos_token` as either plain string or `{content: "..."}` object
- Supports all HF template formats: ChatML, LLaMA 3, Mistral, etc.

**AsyncEngine integration** (`engine.rs`):
- `chat_template: Option<Arc<ChatTemplate>>` field on `AsyncEngine`
- `with_tokenizer_and_template()` constructor
- `chat_to_engine_request()` applies template when available, falls back to newline concatenation

**CLI init** (`init.rs`):
- `try_load_chat_template()` loads from `tokenizer_config.json` in model directory
- Passes template to `AsyncEngine::with_tokenizer_and_template()`

**New dependency**: `minijinja = "2"` (Rust Jinja2 engine, with `builtins` feature for filters like `upper`)

### What was built in Phase 8b+ (`n` parameter + multi-prompt)

**`vllm-serve/src/engine.rs`** (2 modified files, 15 new tests):

**Structural additions:**
- `StreamDelta`: added `index: u32` field — choice index for n>1 support
- `RequestState`: added `choice_index: u32` field — position in response choices
- `submit_request()`: added `choice_index` parameter; moved `requests_total`/`prompt_tokens_total` metrics out to callers (once per HTTP request vs once per engine-core request)
- `process_output()`: sets `delta.index = req_state.choice_index` when constructing `StreamDelta`

**`n > 1` for `chat_completion()`:**
- Tokenizes prompt once, submits `n` child requests with derived IDs (`{base_id}-{i}` for n>1)
- Seed derivation: `seed.wrapping_add(i as u64)` for independent sampling
- Polls all children sequentially, builds `n` choices with proper indices
- Usage: `prompt_tokens` counted once, `completion_tokens` summed across all n

**`n > 1` + multi-prompt for `completion()`:**
- New `tokenize_completion_prompts()` helper normalizes `CompletionPrompt` → `Vec<Vec<u32>>`
- New `tokenize_text()` helper encodes text with or without tokenizer
- Loops P×N (prompts × n), `choice_index = p_idx * n + n_idx`
- Usage: `prompt_tokens` = sum of unique prompt lengths (not multiplied by n)

**`n > 1` streaming for `chat_completion_stream()`:**
- All `n` children share one `(tx, rx)` channel (tx is cloned per child)
- Each child has its own `choice_index` in `RequestState`, embedded in `StreamDelta`
- Original tx dropped so rx closes when all children finish

**`vllm-serve/src/server.rs`:**
- `stream_chat_response()`: changed hardcoded `index: 0` to `delta.index`

**Tests** (15 new, all use `#[tokio::test(flavor = "multi_thread", worker_threads = 2)]` for async tests due to `block_in_place` in step loop):
- `test_chat_completion_n1_regression` / `test_chat_completion_n2`
- `test_completion_n1_regression` / `test_completion_n3`
- `test_completion_multi_prompt_n1` / `test_completion_multi_prompt_n2`
- `test_chat_completion_stream_n2`
- `test_tokenize_completion_prompts_*` (single, multiple, token_ids, none)
- `test_stream_delta_has_choice_index`

### Key design decisions
- **Arena-style indices** instead of Rc/RefCell for linked list pointers (cache-friendly, no GC overhead)
- **SmallVec<[usize; 1]>** for block hash map values (inline single-block case, heap only for duplicates)
- **KVCacheManagerOps trait** decouples scheduler from KV cache implementation (allows SimpleBlockTracker for testing)
- **Sentinel blocks** appended to arena for queue head/tail (eliminates null-check branches)
- Single KV cache group focus initially; multi-group hybrid model support deferred

### Remaining Phase 1 work
- [ ] Wire `RustScheduler` into Python `vllm/v1/engine/core.py` as drop-in replacement
- [ ] Add feature flag to switch between Python and Rust scheduler
- [ ] Integration test: run existing Python test suite against Rust scheduler
- [ ] Benchmark Rust vs Python scheduler latency
- [ ] Port `kv_cache_coordinator.py` for hybrid model multi-group support
- [ ] Port `async_scheduler.py` async variant

---

## Context

vLLM is a high-performance LLM inference and serving engine. The current codebase is:
- **~507K lines of Python** across 1,352 files
- **~89K lines of C++/CUDA** across 242 files in `csrc/`
- **~250 model architectures** in `vllm/model_executor/models/`
- **~948 test files**
- Heavy dependencies: PyTorch, Transformers, FastAPI, ZMQ, and ~60 Python packages

**Goals**: Performance (eliminate GIL/Python overhead in control plane), safety (Rust's type system and memory safety), and standalone binary (single deployable artifact without Python runtime).

**Strategy**: Incremental port via PyO3, replacing Python components one at a time while keeping the system functional at every step. CUDA kernels stay as-is, accessed via FFI.

---

## Proposed Rust Workspace Structure

```
vllm-rs/
  Cargo.toml                    # workspace root
  crates/
    vllm-core/                  # Scheduler, KV cache mgmt, block allocation
    vllm-engine/                # Engine core loop, request lifecycle
    vllm-serve/                 # HTTP (axum) + gRPC (tonic) servers
    vllm-executor/              # Worker/process management, distributed coordination
    vllm-model/                 # Model loading, weight mgmt, tensor abstractions
    vllm-models/                # Model architecture implementations
    vllm-kernels/               # FFI bindings to CUDA/C++ kernels in csrc/
    vllm-config/                # Configuration types and parsing
    vllm-protocol/              # Wire formats: msgpack, serialization, ZMQ
    vllm-common/                # Shared types, errors, logging, metrics
    vllm-pyo3/                  # PyO3 bridge module (temporary, shrinks over time)
  csrc/                         # Existing CUDA/C++ kernels (symlink or copy)
  build.rs                      # CUDA kernel compilation
```

**Key Rust dependencies**: `tokio`, `axum`, `tonic`, `pyo3`, `serde`, `rmp-serde` (msgpack), `zeromq`, `candle-core` (tensor ops), `safetensors`, `prometheus`, `tracing`

---

## Phase 1: Foundation + Scheduler (~2-3 months) — IN PROGRESS

The scheduler and KV cache manager are pure algorithmic code with minimal PyTorch dependencies - ideal first targets.

### 1a. Workspace setup ✅ DONE
- [x] Initialize Cargo workspace with 11 crate stubs
- [x] Set up CI (build, test, clippy, fmt) — `.github/workflows/rust.yml`
- [x] Set up `build.rs` for CUDA kernel compilation (stub)
- [x] Establish PyO3 bridge crate with basic Python module

### 1b. Core types (`vllm-common`, `vllm-config`) ✅ DONE
- [x] `vllm/v1/request.py` → `Request`, `RequestStatus`, `FinishReason`
- [x] `vllm/v1/engine/__init__.py` → `EngineCoreRequest`, `EngineCoreOutput`, `EngineCoreOutputs`
- [x] `vllm/sampling_params.py` → `SamplingParams`
- [x] `vllm/config/` → `ParallelConfig`, `SchedulerConfig`, `CacheConfig`, `ModelConfig`
- [x] `vllm/v1/kv_cache_interface.py` → `KVCacheConfig`, `KVCacheSpec`, `KVCacheGroupSpec`

### 1c. KV cache manager (`vllm-core`) ✅ DONE
- [x] `block_pool.py` → `BlockPool` with arena-based free-list management
- [x] `kv_cache_manager.py` → `KVCacheManager` (block allocation, eviction, prefix cache)
- [x] `kv_cache_utils.py` → `KVCacheBlock`, `FreeKVCacheBlockQueue`, block hashing
- [ ] `kv_cache_coordinator.py` → Multi-layer cache coordination (deferred — single-group works)
- [ ] `single_type_kv_cache_manager.py` → Per-type cache manager (deferred)

### 1d. Scheduler (`vllm-core`) ✅ DONE
- [x] `interface.py` → `SchedulerInterface` trait
- [x] `scheduler.py` → Main `Scheduler` implementation (3-phase algorithm, preemption, chunked prefill)
- [x] `output.py` → `SchedulerOutput`, `NewRequestData`, `CachedRequestData`
- [x] `request_queue.py` → `FCFSRequestQueue`, `PriorityRequestQueue`
- [ ] `async_scheduler.py` → Async variant (deferred)

### 1e. PyO3 bridge — PARTIAL
- [x] Expose `RustScheduler` to Python implementing `SchedulerInterface`
- [ ] Expose `RustKVCacheManager` to Python
- [ ] Wire into existing `vllm/v1/engine/core.py` as drop-in replacement
- [ ] Feature flag to switch between Python and Rust scheduler

### Milestone 1 deliverable
`pip install vllm` with `--features rust-scheduler` uses Rust scheduler. All existing tests pass. Benchmark shows measurable scheduling latency reduction.

---

## Phase 2: Engine Core + IPC (~2-3 months)

The engine core (`vllm/v1/engine/core.py`) is the central coordinator. Currently uses ZMQ + msgpack for IPC between the API process and the engine process.

### 2a. Serialization (`vllm-protocol`)
Port from `vllm/v1/serial_utils.py`:
- `MsgpackEncoder` / `MsgpackDecoder` → Rust msgpack with serde
- Ensure wire-compatible with existing Python encoder (for mixed Rust/Python operation)

### 2b. ZMQ transport (`vllm-protocol`)
Port ZMQ socket management from `vllm/utils/network_utils.py`:
- Request/reply patterns for engine commands
- Pub/sub for streaming outputs
- Use `zeromq` crate or raw `libzmq` FFI

### 2c. Engine core loop (`vllm-engine`)
Port `vllm/v1/engine/core.py` → `EngineCore`:
- Main `run_engine_loop()` — poll for requests, run scheduler, dispatch to executor, collect outputs
- Request management (add/abort/pause)
- Handshake protocol with API server
- Stats collection

### 2d. Engine core client (`vllm-engine`)
Port `vllm/v1/engine/core_client.py`:
- `EngineCoreClient` — async client that talks to engine core over ZMQ
- Both in-process and multi-process variants

### Milestone 2 deliverable
Rust engine core process communicating with Python API server over ZMQ. Scheduler + engine core are Rust. Worker/executor still Python.

---

## Phase 3: Serving Layer (~2-3 months)

Replace FastAPI with axum for HTTP and tonic for gRPC. This eliminates the largest Python runtime dependency for the user-facing surface.

### 3a. OpenAI-compatible HTTP server (`vllm-serve`)
Port from `vllm/entrypoints/openai/`:
- `/v1/completions`, `/v1/chat/completions`, `/v1/embeddings`
- `/v1/models`, `/health`, `/version`
- SSE streaming for chat completions
- Request validation (map Pydantic models to serde structs)
- `vllm/entrypoints/chat_utils.py` → chat template rendering

### 3b. gRPC server (`vllm-serve`)
Port from `vllm/entrypoints/grpc_server.py` and `vllm/grpc/`:
- Protobuf service definitions
- Streaming RPCs

### 3c. API protocol types (`vllm-serve`)
Port from `vllm/entrypoints/openai/protocol.py`:
- OpenAI request/response types
- Streaming chunk types
- Error responses

### 3d. Output processing
Port from `vllm/v1/engine/`:
- `output_processor.py` → Convert engine outputs to API responses
- `detokenizer.py` → Incremental detokenization (calls into tokenizer)
- `input_processor.py` → Prompt processing pipeline

### 3e. Tokenizer integration
- Use `tokenizers` crate (HuggingFace tokenizers in Rust) for tokenization/detokenization
- The `tokenizers` Python library is already a Rust library with Python bindings, so this is essentially using the same underlying code

### Milestone 3 deliverable
Rust HTTP/gRPC server → Rust engine core → Rust scheduler. Only the executor/worker layer remains in Python. The server binary can be started with minimal Python initialization.

---

## Phase 4: Executor + Distributed (~3-4 months)

This phase ports the process/worker management and distributed communication layers.

### 4a. Executor abstraction (`vllm-executor`)
Port from `vllm/v1/executor/`:
- `abstract.py` → `Executor` trait
- `uniproc_executor.py` → Single-process executor
- `multiproc_executor.py` → Multi-process executor (fork + manage workers)
- `ray_executor.py` → Ray integration (keep as optional Python bridge)

### 4b. Worker management (`vllm-executor`)
Port from `vllm/v1/worker/`:
- `worker_base.py` → `Worker` trait
- `gpu_worker.py` → GPU worker initialization, KV cache allocation
- Process lifecycle management

### 4c. Distributed communication (`vllm-executor`)
Port from `vllm/distributed/`:
- `parallel_state.py` → Tensor/pipeline parallel group management
- `communication_op.py` → All-reduce, all-gather, broadcast
- NCCL integration via FFI (NCCL is a C library)

### 4d. GPU memory management
Port from `vllm/device_allocator/`:
- CUDA memory allocator wrappers
- Memory profiling and budget calculation

### Milestone 4 deliverable
Rust manages all process orchestration and distributed setup. Workers initialize correctly and communicate via NCCL. Model execution still delegates to PyTorch.

---

## Phase 5: Model Infrastructure (~3-4 months)

Port the model loading and tensor manipulation infrastructure. This is the bridge between Rust and the GPU kernels.

### 5a. Tensor abstraction
- Use `candle-core` as the Rust tensor library, OR
- Create thin Rust wrappers around raw CUDA allocations
- Key decision: candle gives a lot for free but adds a dependency; raw CUDA gives maximum control

### 5b. Weight loading (`vllm-model`)
Port from `vllm/model_executor/model_loader/`:
- SafeTensors loading (use `safetensors` crate — same underlying Rust library)
- GGUF loading (use `gguf` crate)
- Weight sharding for tensor parallelism
- Quantized weight handling

### 5c. CUDA kernel bindings (`vllm-kernels`)
Create Rust FFI bindings for all kernels in `csrc/`:
- Paged attention v1/v2 (`csrc/attention/`)
- Cache operations (`csrc/cache_kernels.cu`)
- Activation kernels (`csrc/activation_kernels.cu`)
- LayerNorm (`csrc/layernorm_kernels.cu`)
- Positional encoding (`csrc/pos_encoding_kernels.cu`)
- MoE kernels (`csrc/moe/`)
- Quantization kernels (`csrc/quantization/`)
- Custom all-reduce (`csrc/custom_all_reduce.cu`)
- Use `cc` crate + `bindgen` for building and binding

### 5d. Layer abstractions (`vllm-model`)
Port from `vllm/model_executor/layers/`:
- `linear.py` → Linear layers (with quantization support)
- `attention/` → Attention layer wrappers
- `layernorm.py` → LayerNorm variants
- `rotary_embedding.py` → RoPE implementations
- `activation.py` → SiLU, GELU, etc.
- `vocab_parallel_embedding.py` → Parallel embeddings

### Milestone 5 deliverable
Rust can load model weights, allocate GPU memory, and invoke CUDA kernels. A simple model (e.g., a single transformer layer) runs entirely in Rust.

---

## Phase 6: Model Architectures (~6-12 months, parallelizable)

This is the longest phase. There are ~250 model architectures. Prioritize by popularity.

### 6a. Model definition framework (`vllm-models`)
- Define `Model` trait with `forward()`, `load_weights()`, `sample()`
- Create macros/helpers for common patterns (transformer blocks, MLP, attention)
- Port `vllm/model_executor/models/config.py` → model registry

### 6b. Priority 1 models (most used)
1. `llama.py` → LLaMA / LLaMA 2 / LLaMA 3
2. `mistral.py` → Mistral
3. `qwen2.py` → Qwen2
4. `gemma.py` / `gemma2.py` → Gemma
5. `phi3.py` → Phi-3
6. `deepseek_v2.py` → DeepSeek V2/V3
7. `commandr.py` → Command R

### 6c. Priority 2 models
- GPT-NeoX, GPT-J, Falcon, BLOOM, MPT, StarCoder
- Multimodal: LLaVA, Qwen-VL, InternVL

### 6d. Priority 3 models
- Remaining architectures, ported on-demand
- Community contributions welcome with clear porting guide

### 6e. Python fallback
- For unported models, keep a PyO3 bridge that delegates to the Python model implementation
- This allows the Rust engine to serve ANY model, falling back to Python for unsupported architectures

### Milestone 6 deliverable
Top 10 model architectures run natively in Rust. Remaining models work via Python fallback.

---

## Phase 7: Standalone Binary + Polish (~2-3 months)

### 7a. Remove Python runtime dependency
- For supported models, the binary runs without Python
- Tokenizer: native Rust via `tokenizers` crate
- Config loading: direct HuggingFace Hub API calls
- Model downloading: `hf-hub` crate

### 7b. CLI
- `vllm serve <model>` — starts the server
- `vllm bench <model>` — runs benchmarks
- `vllm convert <model>` — converts model formats
- Use `clap` for argument parsing

### 7c. Docker images
- `vllm/vllm-rust:latest` — minimal image with CUDA runtime + vllm binary
- Dramatically smaller than current Python-based images

### 7d. Observability
- Prometheus metrics (use `prometheus` crate)
- OpenTelemetry tracing (use `opentelemetry` crate)
- Structured logging (use `tracing` crate)

### Milestone 7 deliverable
`./vllm serve meta-llama/Llama-3-8B` works as a single ~50MB binary (+ CUDA libs). Full OpenAI API compatibility.

---

## Phase 8: Functional Inference (~1-2 months)

Phase 7 delivered a standalone binary that loads models and serves HTTP requests, but the generation loop has critical correctness bugs that prevent useful text output. This phase fixes them.

### Current gaps (discovered via code audit)

1. **Decode token IDs are zeros** — `CandleWorker` has no per-request token buffer. After prefill, every decode step feeds `[0, 0, ...]` instead of the last sampled token. Generation produces garbage after step 1.
2. **Position IDs reset each step** — Decode steps use positions `0..N` instead of continuing from `num_computed_tokens`. RoPE embeddings are wrong.
3. **No stop criteria** — `EngineCore::update_from_output()` never checks `max_tokens`, EOS token, or `stop_token_ids`. Requests run forever.
4. **Sampler not wired up** — `CandleWorker` always does raw argmax, ignoring per-request `SamplingParams` (temperature, top-k, top-p). *(Fixed in 8a; full penalty/logprobs support added in 8g.)*
5. **Chat template not applied** — Messages concatenated with bare newlines instead of model-specific format (`[INST]`, `<|im_start|>`, etc.).
6. **No KV cache reuse** — Full attention recompute every step (O(n²) in sequence length).

### 8a. Working generation (gaps 1-4)

**Worker token buffer** (`candle_worker.rs`, ~50 lines):
- [x] Add `HashMap<String, Vec<u32>>` token buffer in `CandleWorker`
- [x] On new request: store prompt token IDs in buffer
- [x] On decode step: feed last sampled token (not 0) as input
- [x] On request finish/abort: remove buffer entry
- [x] Clean up buffers for requests no longer in scheduler output

**Correct position IDs** (`candle_worker.rs`, ~20 lines):
- [x] For new requests: positions = `num_computed_tokens..num_computed_tokens + num_tokens`
- [x] For decode steps: position = `num_computed_tokens` from scheduler (the next position after all cached tokens)

**Stop criteria** (`engine_core.rs`, ~80 lines):
- [x] In `update_from_output()`: call `scheduler.append_output_tokens()` to advance request state
- [x] Check `output_token_ids.len() >= max_tokens` → `FinishReason::Length`
- [x] Check sampled token against `eos_token_id` → `FinishReason::Stop`
- [x] Check sampled token against `stop_token_ids` → `FinishReason::Stop`
- [x] Call `scheduler.finish_requests()` for finished requests

**Wire SamplingParams to Sampler** (`candle_worker.rs` + `engine_core.rs`, ~60 lines):
- [x] Pass `SamplingParams` per request through scheduler output or maintain a params map in worker
- [x] Use `Sampler` for non-greedy requests (temperature > 0)
- [x] Keep argmax fast path for greedy (temperature == 0)

**Key files**: `vllm-executor/src/candle_worker.rs`, `vllm-engine/src/engine_core.rs`, `vllm-common/src/request.rs`

### 8b. Chat templates (gap 5)

- [x] Parse `tokenizer_config.json` from HF Hub for `chat_template` field
- [x] Implement Jinja2 renderer using `minijinja` crate for chat template strings
- [x] Apply template in `AsyncEngine::chat_to_engine_request()` instead of bare newline concatenation
- [x] Support all HF template formats: ChatML, LLaMA 3, Mistral, etc. (via full Jinja2 engine)

**Key files**: `vllm-serve/src/engine.rs`, `vllm-cli/src/init.rs` (download tokenizer_config.json)

### 8c. KV cache in forward pass (gap 6) — DONE

Transforms O(n²) full recompute into O(1) incremental decode.

- [x] Added `KvCache` type (`Vec<Option<(Tensor, Tensor)>>`) and `LayerKvCache` type alias
- [x] Updated `Model::forward()` to accept `kv_cache: Option<&mut KvCache>`, added `num_layers()` method
- [x] Updated `scaled_dot_product_attention` to handle `q_len != kv_len` (general causal mask)
- [x] In `LlamaAttention`/`Gemma2Attention`: concatenate new K/V with cached, update cache in-place
- [x] Threaded KV cache through all layers: `{Llama,Gemma2}{Attention,DecoderLayer,Model,ForCausalLM}`
- [x] Updated `CandleWorker`: per-request `kv_caches` HashMap, prefill populates cache, decode feeds only last token
- [x] Per-request forward passes (required for per-request KV cache with different sequence lengths)

**Key files**: `vllm-models/src/lib.rs`, `vllm-models/src/attention.rs`, `vllm-models/src/llama.rs`, `vllm-models/src/gemma2.rs`, `vllm-executor/src/candle_worker.rs`

---

## Phase 9: GPU Acceleration

### 9a. Metal Tier 1 — candle-core Metal Backend — DONE

Basic Metal inference on Apple Silicon using candle's built-in Metal shaders. No custom kernels.

**Feature flags** (3 Cargo.toml files):
- `vllm-executor`: `metal = ["candle-core/metal", "candle-core/accelerate"]`
- `vllm-kernels`: `metal = ["candle-core/metal", "candle-core/accelerate"]`
- `vllm-cli`: `metal = ["vllm-executor/metal", "vllm-kernels/metal"]`
- Build with: `cargo build -p vllm-cli --features metal`

**Device auto-detection** (`vllm-executor/src/candle_worker.rs`):
- `parse_device()` now handles `"cpu"`, `"cuda:N"`, `"metal"`, `"metal:N"`, `"auto"`
- `auto_detect_device()` cascades: Metal → CUDA → CPU
- CLI `--device` default changed from `"cpu"` to `"auto"`

**Performance fixes**:
- `Linear` weights pre-transposed at construction time — `Linear::new()` stores `weight.t().contiguous()` once, forward is a single `x.matmul(&self.weight)`. Eliminated ~6.5 GB of redundant weight copies per forward pass, giving **6-10x speedup**
- Attention Q/K/V: `.contiguous()` after transpose (required by Accelerate/Metal BLAS)
- RoPE frequency precomputation changed from F64 to F32 tensors (Metal doesn't support F64 matmul)

**Benchmarks** (SmolLM2-1.7B-Instruct, f32, Apple Silicon Metal):
- "What is 2+2?" (42 prompt + 2 completion tokens): 0.26s total
- Haiku generation (38 prompt + 17 completion tokens): 0.80s total (~47ms/token decode)
- Correct, coherent output confirmed

**Key insight**: All layer ops, weight loading, KV cache, and sampling were already device-agnostic. The architecture designed in Phases 5-6 (candle tensor ops, `&Device` threading) paid off — Metal support required ~80 lines of new code across 7 files.

**Key files**: `vllm-executor/src/candle_worker.rs`, `vllm-models/src/attention.rs`, `vllm-model/src/layers/linear.rs`, `vllm-model/src/layers/rotary.rs`, `vllm-cli/src/args.rs`, 3× `Cargo.toml`

#### Metal vs CUDA Architecture

| Aspect | CUDA (datacenter) | Metal (Apple Silicon) |
|--------|-------------------|----------------------|
| Memory | Discrete VRAM, explicit host↔device transfers | Unified Memory (UMA), shared address space |
| Multi-GPU | NCCL, TP/PP across devices | Single GPU, no multi-device |
| KV cache swapping | swap_blocks between host/device RAM | No-op or trivial (same physical memory) |
| Parallelism | MultiprocExecutor, arbitrary TP/PP | UniProcExecutor, tp=1 pp=1 always |
| Max memory | 80-192GB HBM per GPU | 64-192GB unified (M4 Ultra max) |

#### Metal Tier 2 (Custom MSL Kernels) — SUPERSEDED

Custom Metal Shading Language kernels were planned but superseded by the MLX backend (Phase 10). MLX eliminates dispatch overhead at the graph level (~42ms of ~50ms ITL was Metal dispatch from ~1200 kernel dispatches) rather than reducing it kernel-by-kernel. The legacy candle Metal path remains via `--features candle-metal`.

#### Metal Tier 3 — Remaining Items

- **3a. UMA-aware KV cache**: On UMA, `swap_blocks` can be a pointer/index remap instead of a data copy. Implement `UmaKVCacheManager` or add UMA-aware paths to existing `KVCacheManager`.
- **3d. Memory pressure handling**: Monitor `os_proc_available_memory()` or `dispatch_source_create(DISPATCH_SOURCE_TYPE_MEMORYPRESSURE, ...)`. Evict KV cache blocks under pressure (recomputable). Gracefully reduce batch size when memory is tight.
- Tier 3b (zero-copy weight loading) and 3c (quantization) are free via MLX backend.

#### Lessons from Metal Tier 1

- **Pre-transpose weights at load time**: Calling `weight.t().contiguous()` per forward copies entire weight matrices. Pre-transposing once at construction gave 6-10x speedup.
- **Metal doesn't support F64 matmul**: RoPE frequency precomputation changed from F64 to F32 tensors. Scalar `powf` still computed in f64 for precision, then cast to f32.
- **Feature unification works well**: Enabling `candle-core/metal` on any one crate propagates to all via Cargo feature unification.

### 8e. Native dtype inference (f16/bf16) ✅ DONE

Preserves native f16/bf16 dtype through weight loading, layers, and forward passes.
Halves memory usage and KV cache size. On Metal GPU, matmul runs in native dtype via
candle's MLX GEMM kernels. On CPU, the gemm crate handles f16 (but not bf16); CPU
matmul is not BLAS-optimized for half types so ITL improvement is minimal there.

**What was implemented:**

- [x] `--dtype auto` (default) reads `torch_dtype` from config.json, falls back to F16
- [x] `candle_dtype()` returns `Option<DType>` — `None` for auto, resolved in `load_model()`
- [x] `CandleWorker` stores `resolved_dtype` field, exposed via accessor for block size calc
- [x] RmsNorm/GemmaRmsNorm: upcast only variance reduction to f32, cast tiny rsqrt `[batch,1]` back to native dtype — avoids 2 full-tensor copies per norm
- [x] Attention softmax: upcast scores to f32 before softmax, cast back before V matmul
- [x] `Model::forward()` casts logits to F32 for sampling (both LlamaForCausalLM, Gemma2ForCausalLM)
- [x] `compute_num_blocks()` uses `dtype_size(dtype)` instead of hardcoded `*4`
- [x] Dropped `candle-core/accelerate` from metal feature — it hard-errored on f16 CPU matmul (`"the accelerate backend does not support f16 matmul"`), and all heavy compute goes through the Metal GPU backend
- [x] 8 new tests (f16 norm dtype/values, gemma f16 norm, f16 attention prefill/decode, auto dtype parsing, f16 block calc)

**Key findings:**
- CPU gemm crate: supports F16 matmul but not BF16; neither is BLAS-optimized
- Metal GPU backend: native F16/BF16 matmul via `call_mlx_gemm` — works correctly
- ITL improvement limited by Metal kernel dispatch overhead (~350+ dispatches per decode token) and O(n) KV cache concat copies per layer, not by dtype. Pre-allocated KV cache (Phase 8d) and fused kernels (Metal Tier 2) are needed for ITL gains.
- [ ] **Sampling: f32 logits** — The final lm_head projection should produce f32
  logits for sampling (temperature/top-k/top-p need f32 precision). Cast only the
  logits tensor, not the entire forward pass.
- [ ] **Tests** — Differential testing: run same prompts in f32 and f16, verify
  outputs match within tolerance. Test on CPU, Metal, and CUDA (if available).

**Key files**: `vllm-model/src/tensor.rs`, `vllm-model/src/weight.rs`,
`vllm-model/src/layers/*.rs`, `vllm-models/src/attention.rs`,
`vllm-executor/src/candle_worker.rs`, `vllm-cli/src/args.rs`

**Dependencies**: None — can be done independently of paged KV cache (8d) or Metal
Tier 2 (9b). Should be done *before* quantization (9c) since quantized formats
dequantize to f16, not f32.

**Effort**: ~1-2 sessions. High ROI — biggest single-request latency improvement
available without custom kernels.

### 8d. Paged KV cache — cross-request prefix sharing

Phase 8c added per-request contiguous KV caches. This phase connects the tensor
storage to the existing `KVCacheManager` block-tracking infrastructure so that
requests sharing a common prefix (e.g., system prompt) reuse the same cached K/V
blocks instead of recomputing them.

**Sub-tasks:**

- [ ] **Block tensor pool** — A `KvBlockPool` that allocates fixed-size KV tensor
  blocks (shape `[block_size, num_kv_heads, head_dim]` per layer). Blocks are
  identified by index matching `BlockPool` in `vllm-core`.
- [ ] **Map scheduler block IDs → tensor slices** — When `SchedulerOutput`
  provides `block_ids` for a request, the worker resolves them to tensor
  references. Attention reads/writes K/V through these block references instead
  of a per-request contiguous tensor.
- [ ] **Paged attention kernel** — Modify `scaled_dot_product_attention` (or add
  a `paged_attention` variant) that gathers K/V from non-contiguous blocks
  before computing scores, or iterates over blocks in-place.
- [ ] **Prefill: populate blocks** — On prefill, write K/V into the blocks
  assigned by the scheduler. The `KVCacheManager` content-hashes blocks for
  prefix caching.
- [ ] **Decode: append to last block** — On decode, write the new token's K/V
  into the current block's next slot. When a block is full, the scheduler
  allocates a new one.
- [ ] **Copy-on-write** — When the scheduler indicates a block is shared
  (ref_count > 1) and needs mutation, copy the block's tensor data to a new
  block before writing.
- [ ] **Block free** — When `finished_req_ids` clears a request, its block
  ref-counts decrement. Blocks reaching zero are returned to the free pool (both
  in `BlockPool` and the tensor pool).
- [ ] **Update `CandleWorker`** — Replace `kv_caches: HashMap<String, KvCache>`
  with a global `KvBlockPool`. `execute_model` maps each request's block IDs to
  tensor slices and passes them through the forward pass.

**Key files**: `vllm-executor/src/candle_worker.rs`, `vllm-models/src/attention.rs`,
`vllm-core/src/block_pool.rs`, `vllm-core/src/kv_cache_manager.rs`

**Dependencies**: The scheduler and block pool already track block allocation,
hashing, and ref-counts. This phase adds the tensor backing store and wires it
into the forward pass.

### Milestone 8 deliverable
`./vllm serve meta-llama/Llama-3.2-1B` generates correct, coherent text with proper stop conditions. Chat endpoint applies correct prompt format. Non-greedy sampling works. KV cache makes decode steps O(1) per token. Paged KV cache enables cross-request prefix sharing — concurrent requests with the same system prompt share cached K/V blocks without recomputation.

---

## Phase 12: Tool Calling + Structured Output

OpenAI-compatible tool/function calling and structured output support. This is a serving-layer feature that sits above the model — models already generate the right tokens when prompted correctly; this phase adds the protocol plumbing, response parsing, and optionally constrained decoding to guarantee valid output.

### 12a. Tool calling protocol types + chat template integration

**Protocol types** (`vllm-serve/src/protocol.rs`):
- [ ] Add `tools: Option<Vec<Tool>>` and `tool_choice: Option<ToolChoice>` to `ChatCompletionRequest`
- [ ] `Tool` struct: `type: "function"`, `function: FunctionDef` (name, description, parameters as JSON Schema)
- [ ] `ToolChoice` enum: `"none"` | `"auto"` | `"required"` | `{ type: "function", function: { name } }`
- [ ] Add `tool_calls: Option<Vec<ToolCallDelta>>` to `ChatCompletionMessage` (response) and `DeltaMessage` (streaming)
- [ ] `ToolCall` struct: `id`, `type: "function"`, `function: { name, arguments }` (arguments is a JSON string)
- [ ] `ToolCallDelta`: same shape but all fields optional (for streaming partial tool calls)
- [ ] Add `role: "tool"` variant to message types, with `tool_call_id` field for tool results

**Chat template integration** (`vllm-serve/src/chat_template.rs`, `engine.rs`):
- [ ] Pass `tools` array into minijinja template context when rendering (most model templates already handle tool definitions — e.g., LLaMA 3.1+, Qwen2.5, Mistral v3+)
- [ ] Pass `tool_choice` into template context (some templates use it to force tool-call format)
- [ ] No model-specific code needed — the Jinja2 templates in `tokenizer_config.json` already encode tool schemas into the prompt for each model family

**Key insight**: Modern chat templates (LLaMA 3.1+, Qwen2.5, Mistral) natively handle `tools` in Jinja2. The minijinja engine already supports the required Jinja2 features. This sub-phase is primarily protocol types + passing data through to the template.

### 12b. Tool call response parsing

The model generates tool calls as text (JSON blocks, special tokens, or model-specific formats). This sub-phase parses that raw text into structured `ToolCall` objects in the API response.

**Response parser** (`vllm-serve/src/tool_parser.rs`, new file):
- [ ] `ToolCallParser` trait with `parse_tool_calls(text: &str) -> Option<Vec<ToolCall>>` and `parse_stream_delta(delta: &str) -> Option<ToolCallDelta>>`
- [ ] **Hermes/generic parser**: Detects `<tool_call>{"name": ..., "arguments": ...}</tool_call>` blocks (used by LLaMA 3.1+, Qwen2.5, many fine-tuned models)
- [ ] **Mistral parser**: Detects `[TOOL_CALLS]` token followed by JSON array
- [ ] **Streaming support**: Incremental JSON parsing — buffer partial tool call text, emit `ToolCallDelta` chunks as `function.name` and `function.arguments` fragments become available
- [ ] Parser selection: auto-detect from chat template content or model architecture, or allow `--tool-call-parser` CLI flag (matching Python vLLM's approach)

**Integration** (`vllm-serve/src/engine.rs`, `server.rs`):
- [ ] `AsyncEngine` holds optional `ToolCallParser`
- [ ] Non-streaming: after generation completes, run parser on full output text; if tool calls detected, populate `tool_calls` field and set `finish_reason: "tool_calls"`
- [ ] Streaming: run parser incrementally on each delta; emit `tool_calls` deltas alongside or instead of `content` deltas
- [ ] When `tool_choice: "none"`, skip tool parsing entirely
- [ ] When `tool_choice: { function: { name } }`, validate that the parsed tool call matches the requested function

**Key files (Python reference)**: `vllm/entrypoints/openai/tool_parsers/` — contains Hermes, Mistral, LLaMA, Jamba, and other model-specific parsers. The streaming logic in Python is complex; start with non-streaming, then add streaming.

### 12c. Structured output / constrained decoding

Constrained decoding forces the model to produce output matching a given format (JSON object, JSON schema, regex, grammar). This requires modifying logits at each decode step before sampling.

**`response_format` protocol support** (`vllm-serve/src/protocol.rs`):
- [ ] Add `response_format: Option<ResponseFormat>` to `ChatCompletionRequest`
- [ ] `ResponseFormat` enum: `{ type: "text" }` (default, no-op) | `{ type: "json_object" }` (valid JSON) | `{ type: "json_schema", json_schema: { name, schema, strict } }`

**Logit processor infrastructure** (`vllm-models/src/logit_processor.rs`, new file):
- [ ] `LogitProcessor` trait: `fn process(&mut self, token_ids: &[u32], logits: &mut Tensor) -> Result<()>`
- [ ] `LogitProcessorPipeline`: chain of processors applied in order before sampling
- [ ] Wire into `CandleWorker` and `MlxWorker`: after forward pass produces logits, apply per-request logit processors before sampler

**Grammar-guided decoding** (`vllm-models/src/grammar.rs` or new crate):
- [ ] Evaluate Rust grammar engines: `llguidance` (Microsoft, used by Python vLLM), `outlines-core` (Rust core of outlines), or custom CFG/PDA implementation
- [ ] `GrammarLogitProcessor`: implements `LogitProcessor`, maintains grammar state machine, masks logits for invalid next tokens
- [ ] JSON mode (`type: "json_object"`): compile a generic JSON grammar, apply as logit processor
- [ ] JSON Schema mode (`type: "json_schema"`): compile schema → grammar (handle object keys, enum values, string patterns, numeric ranges), apply as logit processor
- [ ] `strict: true` vs `strict: false`: strict mode enforces exact schema match via grammar; non-strict mode uses JSON grammar only

**Performance considerations**:
- Grammar state update and logit masking must be fast (runs every decode step per request)
- Pre-compile grammars for common schemas; cache compiled grammars by schema hash
- Token vocabulary → grammar transition table can be pre-computed at model load time
- Vocabulary-aware masking: map grammar state → set of valid token IDs → logit mask tensor

**Dependencies**: This is the most complex sub-phase. 12a and 12b are independent and can be done first. 12c depends on nothing in 12a/12b (it's a separate logit-level mechanism) but is complementary — structured output ensures tool call arguments are valid JSON.

### Milestone 12 deliverable
`./vllm serve meta-llama/Llama-3.1-8B-Instruct` with `tools` in the chat completion request returns structured `tool_calls` in the response. `response_format: { type: "json_schema", json_schema: { schema: ... } }` forces output to conform to the given schema. Streaming tool call deltas work correctly. Compatible with OpenAI client libraries.

---

## Cross-Cutting Concerns

### Testing strategy
- **Unit tests**: Each Rust crate has comprehensive unit tests
- **Integration tests**: Rust engine serving requests end-to-end
- **Compatibility tests**: Run existing Python test suite against Rust implementation
- **Benchmark suite**: Latency, throughput, memory at each phase vs. Python baseline
- **Conformance tests**: OpenAI API compatibility test suite

### PyO3 bridge lifecycle
The `vllm-pyo3` crate shrinks over time:
- Phase 1: Exposes scheduler to Python
- Phase 2-3: Exposes engine + server to Python
- Phase 4-5: Exposes executor + model infra to Python
- Phase 6+: Only used for fallback model execution
- Phase 7: Optional, only for models not yet ported

### Risk mitigation
- **Correctness**: Differential testing — run same inputs through Python and Rust, compare outputs
- **Performance regression**: Continuous benchmarking at each phase
- **Scope creep**: Each phase has a clear deliverable that works independently
- **Community adoption**: Maintain Python API compatibility throughout (same `pip install vllm` interface)

### Estimated timeline
| Phase | Duration | Cumulative |
|-------|----------|------------|
| 1: Foundation + Scheduler | 2-3 months | 2-3 months |
| 2: Engine Core + IPC | 2-3 months | 4-6 months |
| 3: Serving Layer | 2-3 months | 6-9 months |
| 4: Executor + Distributed | 3-4 months | 9-13 months |
| 5: Model Infrastructure | 3-4 months | 12-17 months |
| 6: Model Architectures | 6-12 months | 18-29 months |
| 7: Standalone Binary | 2-3 months | 20-32 months |
| 12: Tool Calling + Structured Output | 2-3 months | — |

Phases 5-6 can partially overlap. Phase 12 can be done in parallel with model architecture work (Phase 6) as it's a serving-layer feature. With a dedicated team of 3-5 Rust engineers, the critical path is ~18-24 months to a fully functional Rust serving engine for top models.

---

## Key Files Reference

| Component | Python Source | Rust Target Crate |
|-----------|-------------|-------------------|
| Scheduler | `vllm/v1/core/sched/scheduler.py` | `vllm-core` |
| KV Cache Mgr | `vllm/v1/core/kv_cache_manager.py` | `vllm-core` |
| Block Pool | `vllm/v1/core/block_pool.py` | `vllm-core` |
| Engine Core | `vllm/v1/engine/core.py` | `vllm-engine` |
| Async LLM | `vllm/v1/engine/async_llm.py` | `vllm-engine` |
| Core Client | `vllm/v1/engine/core_client.py` | `vllm-engine` |
| Serialization | `vllm/v1/serial_utils.py` | `vllm-protocol` |
| API Server | `vllm/entrypoints/openai/api_server.py` | `vllm-serve` |
| gRPC Server | `vllm/entrypoints/grpc_server.py` | `vllm-serve` |
| Executor | `vllm/v1/executor/abstract.py` | `vllm-executor` |
| GPU Worker | `vllm/v1/worker/gpu_worker.py` | `vllm-executor` |
| Model Runner | `vllm/v1/worker/gpu_model_runner.py` | `vllm-executor` |
| Config | `vllm/config/*.py` | `vllm-config` |
| Request | `vllm/v1/request.py` | `vllm-common` |
| SamplingParams | `vllm/sampling_params.py` | `vllm-common` |
| CUDA Kernels | `csrc/` | `vllm-kernels` (FFI only) |
| Models | `vllm/model_executor/models/` | `vllm-models` |
| Layers | `vllm/model_executor/layers/` | `vllm-model` |
