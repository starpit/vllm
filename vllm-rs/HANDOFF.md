# Handoff: Closing the Throughput Gap with Python vLLM

## Current State

**Throughput (default, 1000 prompts)**: Rust 12.14 req/s vs Python 17.41 req/s (~30% gap)
**Throughput (200 prompts)**: Rust 15.47 req/s vs Python 18.74 req/s (~17% gap)
**Latency**: Rust ~1.306s vs Python ~1.30s (~0.5% gap)

All benchmarks: `vllm bench throughput --model Qwen/Qwen2.5-3B-Instruct` (defaults: 1000 prompts, 1024 input, 128 output)

### What's been done this session
- **FA2 num_splits=1**: Removed Rust-side split-K heuristic for paged attention. Python's upstream `vllm-flash-attn` C code always sets `params.num_splits = 1`. Our code was computing num_splits > 1 for small batches, causing the expensive split-K path unnecessarily. (Latency improved from 1.33s to 1.306s.)
- **Defaults match Python**: `max_num_seqs=256`, `max_num_batched_tokens=8192` — verified against Python `arg_utils.py` `LLM_CLASS` context.
- **KV cache memory sizing matches Python's formula**: `total * util - (weights + peak_activations + 150MB)`. Previously we reserved `6 * peak_activations` as headroom, resulting in only 365K KV tokens. Now we get 794K tokens (Python: 975K). GPU memory usage: Rust 41GB vs Python 42.7GB.
- **nsys profiling done for both Python and Rust** — see findings below.

### What was already done (prior sessions)
- Background executor thread with 2-batch pipeline — matches Python's `step_with_batch_queue`
- CUDA graph capture for decode (BS=1..512) and prefill (128..8192, but prefill graphs disabled due to FA2 correctness issue)
- cublasLt with plan caching for all GEMMs
- PyTorch-style caching allocator (replaces arena)
- Fused CUDA kernels: rms_norm, fused_add_rms_norm, silu_and_mul, rotary, embedding_gather, reshape_and_cache
- In-graph argmax for greedy decode

### nsys profiling findings (200 prompts, Qwen2.5-3B)

**GPU kernels are faster in Rust** — the bottleneck is CPU-side:

| Metric | Rust | Python |
|--------|------|--------|
| GPU kernel time | 8.4s | 9.2s |
| Wall time | 12.8s | ~10.7s |
| **GPU idle time** | **4.4s (34%)** | **~1.5s (14%)** |
| GPU memory | 41 GB | 42.7 GB |
| cuStreamSynchronize | 61 calls, 1.57s | — |
| cudaDeviceSynchronize | — | 2069 calls, 160ms |
| H2D copies | 1371, 841ms | 2543, 1027ms |
| cuMemFree | 138, 103ms | 135, 42ms |
| Graph launches | 128, 84ms | 127, 42ms |

### Memory gap: Rust 41 GB vs Python 42.7 GB (~1.7 GB short)

After fixing the KV cache formula, Rust allocates 794K KV tokens vs Python's 975K. The remaining ~1.7 GB gap is because our `CachingAllocator::trim()` is a no-op (`alloc.rs` line 538) — it never returns memory to CUDA after the profiling dry-run. Python's PyTorch allocator does `torch.cuda.empty_cache()` after profiling, freeing temp allocations back to CUDA, so that memory becomes available for KV cache. Fix: implement `trim()` to release unsplit segments back to the driver via `cuMemFree`.

### Root causes of GPU idle time (next steps)

1. **61 `cuStreamSynchronize` calls (1.57s total, avg 25ms each)** — explicit GPU sync points in `execute_model` that block the CPU from scheduling the next batch. Python keeps everything async until the final D2H copy. Find and eliminate these.

2. **Prefill path runs eager (no CUDA graphs)** — `cuda_worker.rs` line 1838: `let use_prefill_graph = false`. Comment: "Prefill graphs are disabled: they capture paged FA2 which produces incorrect results for q_len > 1." Python uses torch.compile for prefill, fusing everything into an optimized graph. Our eager prefill launches ~100+ individual kernels per step with CPU dispatch overhead between each. Fix the FA2 correctness issue and enable prefill graphs.

3. **Batch transition stalls** — every 256 sequences in a 1000-prompt run, ~256 new prefills must be chunked through the scheduler. During these transitions, there is a multi-second gap where no completions are reported. Python has these too but they are much shorter. The eager prefill path (cause #2) makes these worse.

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
# Rust
nsys profile -o /tmp/rust_throughput --force-overwrite=true \
  ./target/release/vllm bench throughput --model Qwen/Qwen2.5-3B-Instruct --num-prompts 200
nsys stats /tmp/rust_throughput.nsys-rep --report cuda_gpu_kern_sum --force-export=true
nsys stats /tmp/rust_throughput.nsys-rep --report cuda_api_sum --force-export=true

# Python (must use --trace-fork-before-exec=true to capture child engine process)
nsys profile -o /tmp/python_throughput --force-overwrite=true --trace-fork-before-exec=true \
  bash -c 'source /root/vllm/.venv/bin/activate && vllm bench throughput --model Qwen/Qwen2.5-3B-Instruct --num-prompts 200'
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
