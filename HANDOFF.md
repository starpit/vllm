# Context Handoff: vllm-cuda Optimizations

Worktree: `.claude/worktrees/cuda-models/` on branch `feat/cuda-qwen2-gemma2`
Latest commit: 360ee9053 — "perf: prefill CUDA graphs + transfer stream H2D overlap"

## Edit/build/test cycle

1. Edit files locally in worktree at `.claude/worktrees/cuda-models/vllm-rs/crates/`
2. Sync to pod: `oc rsync crates/ nick:/root/vllm/vllm-rs/crates/ --exclude=target` (repeat for `third_party/`). For single files that oc rsync misses, pipe via `cat file | oc rsh nick bash -c 'cat > /path'`
3. Build: `oc rsh nick bash -c 'cd /root/vllm/vllm-rs && export RUSTC_WRAPPER=/usr/bin/sccache && cargo build -p vllm-cli --features cuda-backend --release 2>&1'`
4. Unit tests: `oc rsh nick bash -c 'cd /root/vllm/vllm-rs && export RUSTC_WRAPPER=/usr/bin/sccache && cargo test -p vllm-cuda --features cuda -- --include-ignored 2>&1'` (140 tests)
5. Clippy: `oc rsh nick bash -c 'cd /root/vllm/vllm-rs && export RUSTC_WRAPPER=/usr/bin/sccache && cargo clippy --workspace --exclude vllm-pyo3 --exclude vllm-mlx --features cuda-backend -- -D warnings 2>&1'`
6. Bench latency: `oc rsh nick bash -c 'cd /root/vllm/vllm-rs && ./target/release/vllm bench latency --model <MODEL> [--temperature 0] [--enforce-eager] 2>&1'`
7. Bench defaults: BS=8, input_len=32, output_len=128, warmup=10, iters=30
8. IMPORTANT: Run debug build + release build in parallel (kernel compilation takes ~7 min). Never run two bench commands concurrently — GPU contention gives garbage numbers.

Pod: `oc rsh nick` — L40S GPU, CUDA 12.9, Rust 1.93, path `/root/vllm/vllm-rs/`. `nick2` has 2×L40S for TP testing.

## Gotchas

- oc rsync sometimes silently fails. Verify with `oc rsh nick bash -c 'grep -n "pattern" /path'`. Fallback: `cat local | oc rsh nick bash -c 'cat > remote'`
- Edition 2024 — Rust 1.93 enforces `unsafe {}` blocks inside `unsafe fn`. vllm-cuda uses `#![allow(unsafe_op_in_unsafe_fn)]`.
- `r#gen` not `gen` — gen is a keyword in edition 2024
- `cargo fmt` can't run `--all` from worktree (nested workspace confusion). Run `cargo fmt -p <crate>` per crate.
- vllm-cuda links against vllm-kernels for shared CUDA kernels. Only `embedding_kernels.cu` is compiled by vllm-cuda's `build.rs`. Don't add shared kernels back or you get duplicate symbol linker errors.
- Llama-3.2-3B is a gated model — 401 on the pod. Use `Qwen/Qwen2.5-0.5B` and `Qwen/Qwen2.5-3B` for benchmarking.

## What was done (cumulative)

### Fused QKV split + RoPE kernel
- New `fused_qkv_rope_kernel` in `pos_encoding_kernels.cu`
- Reads from fused QKV GEMM output `[num_tokens, q_size + 2*kv_size]`, applies vectorized RoPE (128-bit loads, NeoX-style) to Q and K, copies V
- Replaces 2 separate kernel launches (split_qkv + rotary_embedding_inplace) with 1
- Wired into LLaMA and Gemma2 attention forward paths

### In-graph argmax for greedy decode (654f9e160)
- `CudaGraphRunner::capture()` now captures `argmax_batched` + `memcpy_dtod_async` (scatter to persistent input_ids) inside the graph
- `replay()` returns `ReplayOutput { logits, token_ids }` and takes `skip_input_ids_h2d: bool`
- Greedy graph fast path in `CudaWorker::execute_model()` uses in-graph argmax directly — no separate kernel launch
- `last_graph_batch_size` tracking skips input_ids H2D when batch composition unchanged between steps
- Non-greedy graph path discards argmax, samples from logits as before
- Invalidation on batch composition change (finished/new requests) or eager path

### cublasLt GEMM with plan caching (f10900b64)
- **All GEMMs now use cublasLt** (was: `cublasGemmEx` for plain GEMM, cublasLt only for bias GEMM)
- **Plan caching**: `HashMap<(M,K,N,dtype,has_bias), GemmPlan>` stores pre-created matmul descriptors, matrix layouts, and heuristic-selected algorithm. First call creates the plan, subsequent calls with same shapes skip all descriptor API calls.
- Eliminates ~1900 descriptor create/destroy API calls per decode step after warmup
- `CublasHandle::gemm()` now takes `&mut self` (for plan cache). `Linear::forward()` takes `&mut CublasHandle`.

### Graph batch padding (f10900b64)
- `CudaGraphRunner::nearest_graph_size(batch_size)` finds smallest captured size >= batch_size
- Decode batches pad to nearest graph size (e.g. BS=3 → graph BS=4, BS=5 → BS=8)
- Padded slots get safe dummy values: `slot_mapping=-1` (no KV write), `cu_seqlens` repeated (zero-length seqs)
- Only `num_reqs` token IDs read back from D2H (padded slots ignored)
- Graph capture sizes expanded: `[1, 2, 4, 8, 16, 32]` (was `[1, 2, 4, 8]`)
- Arena pre-sizing fixed to 256 tokens (was `max_bs * 32` which OOMed for BS=32 on 3B models)

### Per-layer arena scoping + max_num_batched_tokens
- **Root cause**: Arena bump-allocates ALL intermediates for the entire forward pass (unlike PyTorch which frees per-op). A 36-layer 3B model with 2048 tokens accumulated ~4GB of scratch → GPU OOM.
- **Per-layer arena scoping**: Pre-allocate two persistent buffers (`hs_buf`, `res_buf`) for inter-layer state. After each layer, copy `hidden_states` (from scratch) to `hs_buf`, then `set_offset(layer_scratch_base)` to reclaim all intermediates. `residual` is updated in-place by `fused_add_rms_norm` so no copy needed (except first layer where it points to `hs_buf`). Peak memory now proportional to 1 layer's scratch, not N layers.
- **`--max-num-batched-tokens` CLI flag**: Added to `VllmConfig`, CLI args, bench args, `LLMBuilder`, and `CudaWorkerConfig`.
- **Arena pre-sizing**: Warmup dummy forward now uses `max_num_batched_tokens` (single sequence) instead of hardcoded 256 tokens.
- Wired into LLaMA, Qwen2 (delegates to LLaMA), and Gemma2 model forward passes.

### Prefill CUDA graphs (360ee9053 — LATEST)
- **New `PrefillGraphRunner`** in `graph.rs`: captures CUDA graphs for single-sequence prefill at power-of-2 token counts [128, 256, 512, 1024, 2048, 4096, 8192] (filtered by `max_num_batched_tokens`).
- Persistent input buffers: `input_ids`, `positions`, `slot_mapping` sized for max_tokens; `cu_seqlens_q/k` [2], `block_table` [1, MAX_BLOCKS], `last_token_indices` [1].
- `capture()` runs forward with `last_token_indices` + argmax under graph capture. Output: `[1, vocab_size]`.
- `replay()` pads real tokens to captured size, padding slots get `slot_mapping=-1`. H2D on transfer stream + event sync.
- **Wired into `CudaWorker::execute_model()`**: used when `num_reqs == 1 && q_len == seq_len` (fresh single-request prefill).
- Captured after decode graphs in `compile_or_warm_up_model()`, before cublasLt benchmarking (so prefill GEMM shapes get benchmarked too).
- **Results**: 9-10% prefill latency reduction at input_len=512-1024 on Qwen2.5-0.5B.

### Transfer stream H2D overlap (360ee9053 — LATEST)
- All three graph replay paths (`replay()`, `replay_decode_fast()`, prefill `replay()`) now use `device.transfer_stream` for H2D copies instead of `compute_stream`.
- Event sync via `device.sync_transfer_to_compute()` before graph launch.
- Frees compute stream from H2D serialization; enables PCIe controller to pipeline multiple small copies.
- No measurable throughput impact on decode (copies are tiny), but architecturally correct for future cross-step pipelining.

### Persistent GPU decode metadata (bc5ae1824)
- **New CUDA kernel** `update_decode_metadata` in `embedding_kernels.cu`: in one launch, increments `positions[i] += 1`, computes `slot_mapping[i]` from new position + block_table, increments `cu_seqlens_k[1..N+1] += 1`.
- **`replay_decode_fast()`** on `CudaGraphRunner`: uses the GPU kernel instead of building 5 CPU Vecs + 5 H2D copies. `cu_seqlens_q` skipped entirely (constant for decode). Block table only H2D-copied when blocks actually change.
- **`graph_metadata_valid` flag** on `CudaWorker`: tracks whether persistent buffers have valid state. First decode step after batch composition change uses full H2D; subsequent steps use fast path.
- Rust FFI wrapper: `kernels::update_decode_metadata_gpu()` in `kernels.rs`.

### cublasLt algorithm benchmarking (bc5ae1824 — LATEST)
- **`CublasHandle::benchmark_plans()`**: runs during warmup after plan cache is populated from both prefill warmup and graph capture (decode shapes). Gets top 8 algorithms per GEMM shape from heuristic, benchmarks each (3 warmup + 10 timed with CUDA events), keeps the fastest.
- **`event_elapsed()`** added to `driver.rs` for GPU timing.
- Benchmarking found 7-40% faster algorithms on individual GEMM shapes vs heuristic on L40S:
  - M=1 K=4864 N=896 (down_proj BS=1): algo #1 is **39.7%** faster
  - M=32 K=4864 N=896: algo #4 is **18.8%** faster
  - M=2048 K=896 N=1152 (prefill QKV): algo #1 is **21.2%** faster
  - M=2048 K=896 N=151936 (prefill lm_head): algo #1 is **14.7%** faster
- Decode throughput stable (GEMMs are memory-bound at small M); prefill benefits not captured by latency bench.

### GPU-aware max_num_batched_tokens (bc5ae1824 — LATEST)
- After `init_device()`, queries free VRAM via `determine_available_memory()`.
- >=60GB free: uses 8192. Otherwise: 2048. Matches Python vLLM defaults.
- User can still override with `--max-num-batched-tokens`.

### E2E correctness tests — DONE (prior to this session)

### Chat garbage bug — FIXED (upstream engine-level fix, prior to this session)

### CudaWorker model correctness — FIXED
- **Missing QKV bias**: Fused QKV loading (`load_fused`) dropped bias tensors — `Linear::new(qkv_w, None)`.
  Qwen2 has QKV bias; LLaMA does not. Fix: concat Q/K/V biases and pass to `Linear::new`.
- **Per-layer arena copy ordering**: Residual was copied to `res_buf` AFTER `hs_buf` was overwritten with MLP output,
  corrupting the residual on the first layer. Fix: copy residual first when it aliases `hs_buf`.
- Both LLaMA and Gemma2 models fixed.
- Verified: `Qwen/Qwen2.5-0.5B` → "Paris. It is the largest city in Europe..."
- Verified: `unsloth/Llama-3.2-3B-Instruct` → perfect Rayleigh scattering explanation

### Llama 3 RoPE scaling — ADDED
- Both candle `RotaryEmbedding::new_llama3()` and CudaWorker `RotaryCache::new()` now support
  `rope_type: "llama3"` frequency-dependent scaling (factor, low_freq_factor, high_freq_factor).
- Parsed from config.json `rope_scaling` in both candle `LlamaConfig::from_hf_config()` and
  CudaWorker `llama_config_from_hf()`.

### Configurable CUDA graph capture sizes
- `CudaWorkerConfig` now takes `cuda_graph_sizes: Vec<usize>` from `CudaGraphConfig`.
- Default throughput bench passes `1,2,4,8,16,32,64,128,256` — CudaWorker captures all of them
  instead of hardcoded `[1,2,4,8,16,32]`.

### Non-greedy GPU metadata fast path
- `replay_decode_fast()` now used for non-greedy graph decode (temp>0) when batch composition
  is unchanged, saving 5 CPU Vec builds + 5 H2D copies per decode step.
- Previously only greedy decode used the fast path.

### Pinned host staging buffers
- **New `PinnedBuf`** type in `cpu_gpu_buf.rs`: host-only pinned (page-locked) memory wrapper.
  Unlike `CpuGpuBuf`, no GPU allocation — used when GPU destination is managed elsewhere (e.g., graph persistent buffers).
- **New `HostStaging`** struct in `cuda_worker.rs`: pre-allocated pinned buffers for all graph
  replay metadata (input_ids, positions, slot_mapping, cu_seqlens_q/k, block_table, host_token_ids,
  sampling_packed). ~520KB total, allocated after graph capture.
- All graph replay H2D paths now use pinned memory for true async DMA (pageable memory forces
  synchronous staging inside the CUDA driver, defeating `_async`).
- All D2H token ID paths use pinned `host_token_ids` buffer.
- GPU sampling params packed into pinned `sampling_packed` buffer.
- Eliminates 6+ heap Vec allocations per batch composition change.
- `fill_block_table()` helper deduplicates 4× copy-pasted block table building code.
- 5 new unit tests for `PinnedBuf` (create, zero-cap, u32 slice, i64 slice, H2D roundtrip).

## Performance (L40S, default bench: BS=8, in=32, out=128)

Current numbers (after all optimizations):

| Model | Temp | Mode | tok/s |
|-------|------|------|-------|
| Qwen2.5-0.5B | 0 | Eager | 3259 |
| Qwen2.5-0.5B | 0 | Graphs | 3639 |
| Qwen2.5-0.5B BS=1 in=512 | 0 | Eager | 395.6 |
| Qwen2.5-0.5B BS=1 in=512 | 0 | Prefill+Decode graphs | 436.3 |
| Qwen2.5-0.5B BS=1 in=1024 | 0 | Eager | 354.4 |
| Qwen2.5-0.5B BS=1 in=1024 | 0 | Prefill+Decode graphs | 386.5 |
| Qwen2.5-3B | 0 | Graphs | 807 |
| Qwen2.5-3B | - | Throughput | 13580 total tok/s |

## Testing methodology

- Unit tests: `cargo test -p vllm-cuda --features cuda -- --include-ignored` — 145 tests covering tensor ops, GEMM, layers, kernels, model forward passes, plan caching, arena scoping, pinned buffers. Run on pod only (needs GPU).
- Local check: `cargo check -p vllm-cuda --tests` catches Rust compilation errors without needing CUDA.
- Benchmarking: Always run ONE bench command at a time (GPU contention). Compare against baselines in table above. Use `--enforce-eager` to isolate kernel-level changes from graph effects.
- E2E correctness tests exist and pass.

## Known issues

| Issue | Severity | Notes |
|-------|----------|-------|
| ~~Throughput bench OOM on 3B~~ | ~~High~~ | **FIXED** — per-layer arena scoping + `max_num_batched_tokens` wiring. |
| ~~No cuBLAS autotuning~~ | ~~Medium~~ | **FIXED** — `benchmark_plans()` finds 7-40% faster algorithms per shape. |
| ~~H2D copies per decode step~~ | ~~Medium~~ | **FIXED** — persistent GPU metadata with `update_decode_metadata` kernel. |
| ~~Hardcoded max_num_batched_tokens~~ | ~~Low~~ | **FIXED** — GPU-aware auto-detection from VRAM. |
| Gap to Python vLLM on 3B | Low | ~3.3% behind Python on `bench throughput` (13.58k vs 14.05k tok/s). Nearly at parity. |
| Arena pre-sizes to ~1GB for 3B | Low | Correct behavior. Could auto-size from model config. |
| ~~RoPE scaling for Llama 3.2~~ | ~~Low~~ | **FIXED** — llama3 rope_scaling implemented in both backends. |
| `--bench` warmup corrupts chat | Low | Chat `--bench` mode warmup (1-token gen) leaves dirty state, garbling subsequent output. Non-bench chat works perfectly. |

## Next steps (priority order)

1. **Throughput bench** — Nearly at parity: 13.58k vs Python 14.05k tok/s (~3.3% gap).
2. ~~**Prefill graph capture**~~ — **DONE** (360ee9053). Single-sequence prefill graphs at [128..8192] token counts. 9-10% uplift.
3. ~~**H2D/compute overlap**~~ — **DONE** (360ee9053). Transfer stream for all graph H2D. Cross-step pipelining deferred.
4. ~~**LLaMA-family aliases**~~ — **DONE**. Added Qwen3ForCausalLM and Phi3ForCausalLM to LLaMA match arm in CudaWorker. E2E verified: Qwen3-0.6B on L40S.
5. **More model architectures** — Port remaining architectures (MoE, DeepSeek, Gemma3, etc.) to vllm-cuda backend.

## Key files

- cublasLt plan cache + benchmarking: `crates/vllm-cuda/src/cublas.rs` (`CublasHandle`, `GemmPlan`, `ensure_plan`, `benchmark_plans`)
- Fused QKV+RoPE kernel: `crates/vllm-kernels/csrc/pos_encoding_kernels.cu`
- Decode metadata kernel: `crates/vllm-kernels/csrc/embedding_kernels.cu` (`update_decode_metadata`)
- Rust FFI + wrapper: `crates/vllm-cuda/src/kernels.rs` (`fused_qkv_rope()`, `update_decode_metadata_gpu()`)
- Model call sites: `crates/vllm-cuda/src/model/llama.rs`, `crates/vllm-cuda/src/model/gemma2.rs`
- CUDA graph runner: `crates/vllm-cuda/src/graph.rs` (`CapturedGraph`, `ReplayOutput`, `nearest_graph_size`, `replay_decode_fast`)
- CUDA worker: `crates/vllm-executor/src/cuda_worker.rs` (greedy graph fast path, batch padding, `graph_metadata_valid`, `HostStaging`)
- Pinned host buffers: `crates/vllm-cuda/src/cpu_gpu_buf.rs` (`PinnedBuf`, `CpuGpuBuf`)
- Arena scoping: `crates/vllm-cuda/src/arena.rs` (`set_offset`, scoping tests), `crates/vllm-cuda/src/model/llama.rs` (per-layer scoping pattern)
- Config: `crates/vllm-serve/src/init.rs` (GPU-aware `max_num_batched_tokens`), `crates/vllm-executor/src/cuda_worker.rs` (`CudaWorkerConfig`)
- Prefill graph runner: `crates/vllm-cuda/src/graph.rs` (`PrefillGraphRunner`, `PrefillInputTensors`, `PrefillReplayOutput`)
- Old split_qkv kernel: Still in `embedding_kernels.cu` and `kernels.rs` (not removed — may be useful for non-RoPE models)
