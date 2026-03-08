# Handoff: Closing the Throughput Gap with Python vLLM

## Current State

**Throughput (default, 1000 prompts)**: Rust 12.17 req/s vs Python 17.41 req/s (~30% gap)
**Throughput (200 prompts)**: Rust 15.77 req/s vs Python 18.74 req/s (~16% gap)
**Latency**: Rust ~1.306s vs Python ~1.30s (~0.5% gap)

All benchmarks: `vllm bench throughput --model Qwen/Qwen2.5-3B-Instruct` (defaults: 1000 prompts, 1024 input, 128 output)

### What's been done this session

- **Prefill CUDA graphs re-enabled** (`cuda_worker.rs`): The `use_prefill_graph = false` guard was stale — the model's forward pass already uses contiguous FA2 (not paged) for fresh prefills (`tokens_before == 0`), so the original correctness issue no longer applies. Re-enabled with conditions: single request, fresh prefill, captured graph exists for padded size. Note: the throughput benchmark doesn't hit this path (scheduler batches multiple prefills), but it helps latency-sensitive single-request scenarios like chat.

- **vllm-cuda dead code cleanup** (`kernels.rs`, `weights.rs`): Removed unused `num_splits_heuristic` function, unused `head_dim_rounded` and `data_start` variables. Fixes 4 clippy errors with `-D warnings`.

- **`CachingAllocator::trim()` implemented** (`alloc.rs`): Releases free unsplit segments back to CUDA driver via `cuMemFree`. After profiling dry-run, frees ~3 GiB of activation memory back to CUDA so it can be used for KV cache. Matches PyTorch's `release_cached_blocks()`.

- **Streaming weight loading** (`weights.rs`, `llama.rs`, `gemma2.rs`, `cuda_worker.rs`): Rewrote `GpuWeights` to match Python vLLM's approach — weights stay on CPU (mmap'd safetensors files) and are copied to GPU one at a time via `take()`. Fused weights (QKV, gate_up) use `take_into()` to copy directly from CPU → GPU offset in a pre-allocated buffer. No more shard-level GPU buffers. Saves ~3.4 GiB of GPU memory during model loading.

- **nsys steady-state analysis**: Profiled with `--delay` to isolate steady-state from startup. Found that the 61 `cuStreamSynchronize` calls are ALL from cublasLt algorithm benchmarking during startup (one-time cost). In steady state, both Rust and Python use only event-based sync — no unnecessary stream syncs to eliminate.

### Memory improvements

| Metric | Before | After | Python |
|--------|--------|-------|--------|
| weights+overhead | 9.6 GiB | **6.2 GiB** | 5.79 GiB |
| Available KV cache | 27.3 GiB | **30.6 GiB** | 33.49 GiB |
| KV cache blocks | 49,613 | **55,697** | ~60,975 |
| nvidia-smi usage | 41,080 MiB | **~43,000 MiB** | 42,742 MiB |

Remaining ~2.9 GiB gap: peak activations are 3.0 GiB vs Python's ~0.5 GiB. Python's `torch.compile` fuses operations and reduces intermediate tensor sizes.

### What was already done (prior sessions)
- Background executor thread with 2-batch pipeline — matches Python's `step_with_batch_queue`
- CUDA graph capture for decode (BS=1..512) and prefill (128..8192, but prefill graphs disabled due to FA2 correctness issue)
- cublasLt with plan caching for all GEMMs
- PyTorch-style caching allocator (replaces arena)
- Fused CUDA kernels: rms_norm, fused_add_rms_norm, silu_and_mul, rotary, embedding_gather, reshape_and_cache
- In-graph argmax for greedy decode
- FA2 num_splits=1, defaults match Python, KV cache memory formula match

### nsys profiling findings

**Startup (full run, 50 prompts):**

| Metric | Rust | Python |
|--------|------|--------|
| cuStreamSynchronize | 61 calls, 1.51s | 0 calls |
| cuEventSynchronize | 135 calls, 3.72s | — |
| cudaEventSynchronize | — | 113 calls, 7.53s |
| cudaDeviceSynchronize | — | 29 calls, 3.7ms |

The 61 Rust `cuStreamSynchronize` are from cublasLt algorithm benchmarking (3-10 algos × ~6 unique GEMM shapes = ~60 syncs). One-time startup cost, not a per-step issue.

**Steady-state (captured with `--delay` to skip startup):**

| Metric | Rust (10s capture) | Python (10s capture) |
|--------|-------------------|---------------------|
| cuStreamSynchronize | **0** | — |
| cuEventSynchronize | 138 calls | — |
| cudaEventSynchronize | — | 113 calls |
| cudaDeviceSynchronize | — | 29 calls |

Both use only event-based sync in steady state. No unnecessary stream syncs.

**GPU kernels are faster in Rust** — the bottleneck is CPU-side (eager prefill dispatch overhead):

| Metric | Rust | Python |
|--------|------|--------|
| GPU kernel time | 8.4s | 9.2s |
| Wall time | 12.8s | ~10.7s |
| **GPU idle time** | **4.4s (34%)** | **~1.5s (14%)** |

### Root causes of remaining throughput gap

1. **Batched prefill runs eager (no torch.compile equivalent)** — Prefill CUDA graphs are now enabled for single fresh-prefill requests, but the throughput benchmark batches many prefills together (`num_reqs >> 1`), so graphs don't apply there. Python uses `torch.compile` for all prefills (batched or not), fusing everything into optimized kernels with minimal CPU dispatch overhead. Our eager batched prefill launches ~365 individual kernels per step (10/layer × 36 layers). This is the primary cause of the 34% GPU idle time.

2. **Peak activation memory gap (3.0 GiB vs ~0.5 GiB)** — Without torch.compile fusing, our eager forward pass materializes more intermediate tensors, consuming more memory. This reduces available KV cache, which may cause more preemption under heavy load.

3. **Batch transition stalls** — every 256 sequences in a 1000-prompt run, ~256 new prefills must be chunked. During these transitions, the eager prefill path (cause #1) makes these slower.

## Rules of the Road

1. **Be data-driven, no guessing.** Use nsys or insert instrumentation. Don't just guess at bottlenecks.
2. **Always check what Python does.** Whenever you think "this is the simplest/most pragmatic path" — STOP and look at what the Python code actually does. If our code differs from Python, that's a red flag.
3. **Iterate until we are at least as fast.** Don't stop at "close enough."
4. **Use default settings for benchmarks.** Don't pass explicit CLI flags. The defaults should match Python.

## Benchmark Commands

**Rust** (on nick, use defaults):
```bash
cd /root/vllm/vllm-rs
./target/release/vllm bench throughput --model Qwen/Qwen2.5-3B-Instruct
```

**Python** (on nick3, use defaults):
```bash
source /root/vllm/.venv/bin/activate
vllm bench throughput --model Qwen/Qwen2.5-3B-Instruct
```

Both use defaults: 1000 prompts, 1024 input tokens, 128 output tokens, greedy decoding, BF16.

**nsys profiling** (use --num-prompts 200 to keep profiles manageable):
```bash
# Rust — full run
nsys profile -o /tmp/rust_throughput --force-overwrite=true --stats=true \
  ./target/release/vllm bench throughput --model Qwen/Qwen2.5-3B-Instruct --num-prompts 50

# Rust — steady-state only (skip startup)
nsys profile -o /tmp/rust_steady --force-overwrite=true --stats=true --delay=8 --duration=10 \
  ./target/release/vllm bench throughput --model Qwen/Qwen2.5-3B-Instruct --num-prompts 200

# Python (on nick3)
nsys profile -o /tmp/python_steady --force-overwrite=true --stats=true --delay=25 --duration=10 \
  vllm bench throughput --model Qwen/Qwen2.5-3B-Instruct --num-prompts 200
```

## Pod Details

| Pod | GPU | Purpose | Path |
|-----|-----|---------|------|
| nick | 1x L40S (48GB, Ada SM89) | Rust dev + testing | `/root/vllm/vllm-rs/` |
| nick2 | 2x L40S | Rust TP testing | `/root/vllm/vllm-rs/` |
| nick3 | 1x L40S | Python vLLM reference | `/root/vllm/` (`.venv/` for Python env) |

All pods: CUDA 12.9, always `export RUSTC_WRAPPER=/usr/bin/sccache` on nick/nick2.

## Key Files

- **Benchmark**: `crates/vllm-bench/src/throughput.rs`
- **LLM path** (what bench uses): `crates/vllm-serve/src/llm.rs`
- **Pipeline**: `crates/vllm-engine/src/core_client.rs` — `PipelineState`, `get_output_pipelined`
- **CudaWorker execute_model**: `crates/vllm-executor/src/cuda_worker.rs` — `execute_model_inner`
- **KV cache memory formula**: `crates/vllm-executor/src/cuda_worker.rs` — `compute_available_kv_bytes`
- **CUDA graphs**: `crates/vllm-cuda/src/graph.rs`
- **Caching allocator**: `crates/vllm-cuda/src/alloc.rs` — `CachingAllocator`
- **Weight loading**: `crates/vllm-cuda/src/weights.rs` — `GpuWeights` (streaming CPU→GPU)
- **CUDA kernels**: `crates/vllm-cuda/csrc/` and `crates/vllm-cuda/src/kernels.rs`
- **FlashAttention-2**: `third_party/vllm-flash-attn/`

## Test Commands

```bash
# CUDA E2E (14 tests, ~60s)
cargo test -p vllm-e2e --features e2e,cuda --release --test e1_basic_serving test_cuda -- --ignored --test-threads=1

# CUDA kernel unit tests (151 tests)
cargo test -p vllm-cuda --features cuda

# KV cache memory budget tests (3 tests)
cargo test -p vllm-executor --features cuda -- test_kv_cache

# Engine + executor unit tests (41 tests)
cargo test -p vllm-engine -p vllm-executor

# Full workspace clippy (CUDA)
cargo clippy --workspace --exclude vllm-pyo3 --exclude vllm-mlx --features cuda -- -D warnings
```
