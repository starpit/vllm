# Context Handoff: vllm-cuda Optimizations

Worktree: `.claude/worktrees/cuda-models/` on branch `feat/cuda-qwen2-gemma2`

## PRIORITY 1: Paged FlashAttention-2 is broken — must fix before re-enabling CUDA graphs

### The bug

Paged FA2 (`run_mha_paged` in `flash_api.cu`) produces **incorrect results during decode** after the first request completes. Multi-turn chat produces garbled text on the second turn. Single-turn works perfectly.

**Root cause**: Unknown. The paged FA2 kernel reads K/V directly from the block cache via `block_table`. The data in the blocks is correct (verified: gather + contiguous FA2 reads the same blocks and produces correct output). Something in how `run_mha_paged` indexes or reads paged K/V is wrong for subsequent requests.

**What was ruled out**:
- Prefix caching (disabled — no effect)
- `tokens_before` / `cu_seqlens_q` / `cu_seqlens_k` (all verified correct via debug prints)
- Block table contents (verified via D2H)
- Contiguous FA2 for prefill (works correctly for all sequence lengths)
- Arena overflow (arena auto-grows, no overflow)
- cuBLAS plan caching (different M values get different plans)
- CUDA graph warmup contamination (still broken with `enforce_eager=true`, no warmup)
- Slot mapping (verified correct)

**What IS known**:
- The bug affects BOTH the CudaWorker (`vllm-cuda`) and the CandleWorker (`vllm-models` + `candle-flash-attn`)
- Both use the same `run_mha_paged` C kernel in `third_party/candle-flash-attn/kernels/flash_api.cu`
- Gathering K/V from blocks into contiguous tensors and using non-paged `run_mha` works correctly
- The first request's decode (paged FA2) works. The second request's decode (paged FA2) fails.
- Python vLLM uses paged FA2 for decode without issues, so there is likely a parameter mismatch or fork-specific bug in our `run_mha_paged` wrapper

### Current workaround

**CudaWorker** (`llama.rs`, `gemma2.rs`): Replaced paged FA2 with gather + contiguous FA2 for all non-fresh-prefill paths (decode + prefix-cached prefill). `KvCachePool::gather_kv_contiguous()` does per-block D2D memcpy to build contiguous K/V tensors, then calls `flash_attn_contiguous`.

**CandleWorker** (separate session): Similar — disabled `batched_flash_attention_with_cache` (paged FA2), falls back to per-request gather + single-seq FA2.

### Performance impact of workaround

The gather approach has two major costs:
1. **`stream_synchronize` per layer per K+V** — D2H of block_table requires sync (kills async pipelining)
2. **O(num_blocks) D2D memcpy per layer** — extra memory bandwidth
3. **CUDA graphs disabled** (`enforce_eager=true`) — the gather does host-side work that can't be captured in a graph

This negates most of the CUDA graph decode optimizations. Decode tok/s will regress significantly.

### How to fix properly

1. **Write a minimal reproducer**: Call `run_mha_paged` twice with different sequence lengths in a unit test. First call should work, second should fail. This isolates whether the bug is in the kernel or in the calling code.

2. **Compare parameter setup with Python vLLM**: Python's `flash_attn_with_kvcache` entry point sets up `Flash_fwd_params` differently than our `run_mha_paged`. Compare field by field.

3. **Check `num_splits` handling**: Our code passes `num_splits=0`. Python might pass a different value. `num_splits=0` means "auto" but the kernel code treats `0 > 1` as false, so it's equivalent to `num_splits=1`.

4. **Write a CUDA gather kernel**: Replace the CPU-driven per-block D2D memcpy with a single CUDA kernel that reads block_table and gathers K/V in one launch. This eliminates the `stream_synchronize` and enables CUDA graph capture.

5. **Re-enable CUDA graphs**: Once either paged FA2 is fixed OR the gather kernel is graph-capturable, restore `enforce_eager=false` and the decode graph path.

### Key files for investigation

- `run_mha_paged` C kernel: `third_party/candle-flash-attn/kernels/flash_api.cu:140`
- CudaWorker FFI call: `crates/vllm-cuda/src/kernels.rs:1031` (`flash_attn_paged_ext`)
- CandleWorker FFI call: `third_party/candle-flash-attn/src/lib.rs:1155` (`FlashAttnPagedVarLen`)
- Gather workaround: `crates/vllm-cuda/src/kv_cache.rs:96` (`gather_kv_contiguous`)
- Model attention: `crates/vllm-cuda/src/model/llama.rs:351`, `gemma2.rs:277`

---

## Edit/build/test cycle

1. Edit files locally in worktree at `.claude/worktrees/cuda-models/vllm-rs/crates/`
2. Sync to pod: `tar cf - --exclude target crates/ | oc rsh nick bash -c 'cd /root/vllm/vllm-rs && tar xf -'`. For single files: `cat file | oc rsh nick bash -c 'cat > /path'`
3. Build: `oc rsh nick bash -c 'cd /root/vllm/vllm-rs && export RUSTC_WRAPPER=/usr/bin/sccache && cargo build -p vllm-cli --features cuda-backend --release 2>&1'`
4. Unit tests: `oc rsh nick bash -c 'cd /root/vllm/vllm-rs && export RUSTC_WRAPPER=/usr/bin/sccache && cargo test -p vllm-cuda --features cuda -- --include-ignored 2>&1'` (140 tests)
5. Clippy: `oc rsh nick bash -c 'cd /root/vllm/vllm-rs && export RUSTC_WRAPPER=/usr/bin/sccache && cargo clippy --workspace --exclude vllm-pyo3 --exclude vllm-mlx --features cuda-backend -- -D warnings 2>&1'`
6. Multi-turn chat test: `oc rsh nick bash -c 'cd /root/vllm/vllm-rs && echo -e "why is the sky blue?\nand why red at night?" | ./target/release/vllm chat --model Qwen/Qwen2.5-0.5B --max-tokens 50 2>&1'`
7. IMPORTANT: `cargo fmt` can't run `--all` from worktree. Run `cargo fmt -p <crate>` per crate.
8. NOTE: cudaforge uses content hashing — `touch` alone won't trigger recompilation. Actual content must change.

Pod: `oc rsh nick` — L40S GPU, CUDA 12.9, Rust 1.93, path `/root/vllm/vllm-rs/`. `nick2` has 2xL40S for TP testing.

## Gotchas

- oc rsync sometimes silently fails. Verify with `oc rsh nick bash -c 'grep -n "pattern" /path'`. Fallback: `cat local | oc rsh nick bash -c 'cat > remote'`
- Edition 2024 — Rust 1.93 enforces `unsafe {}` blocks inside `unsafe fn`. vllm-cuda uses `#![allow(unsafe_op_in_unsafe_fn)]`.
- `r#gen` not `gen` — gen is a keyword in edition 2024
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

### Prefill CUDA graphs (360ee9053)
- **New `PrefillGraphRunner`** in `graph.rs`: captures CUDA graphs for single-sequence prefill at power-of-2 token counts [128, 256, 512, 1024, 2048, 4096, 8192] (filtered by `max_num_batched_tokens`).
- **CURRENTLY DISABLED** (`use_prefill_graph = false`) — prefill graphs captured paged FA2 which is broken.
- 9-10% prefill latency improvement when working.

### Paged FA2 workaround — gather + contiguous FA2 (LATEST)
- Replaced paged FA2 in LLaMA and Gemma2 attention with gather + contiguous FA2
- New `KvCachePool::gather_kv_contiguous()`: D2H block_table, per-block D2D memcpy to contiguous tensor
- `enforce_eager=true`, `enable_prefix_caching=false` in defaults (temporary)
- Fixes multi-turn chat corruption on both CudaWorker and CandleWorker paths
- **Performance regression**: No CUDA graphs, extra memory copies — needs proper fix (see Priority 1 above)

### cudaforge — incremental CUDA kernel builds (VERIFIED on pod)
- **Problem**: Every `.cu` file change triggered a FULL rebuild of ALL kernels (10 custom + 7 Marlin + 49 FA2 = 66 files). Took ~15 minutes on the pod.
- **Solution**: `cudaforge` crate (crates.io v0.1.4, https://github.com/guoqingbao/cudaforge) — drop-in replacement for `cc`/`bindgen_cuda` in build.rs. Content-hash based incremental builds.
- **Migrated**: Both `vllm-kernels/build.rs` and `vllm-cuda/build.rs` now use `cudaforge::KernelBuilder`.
- **Verified on pod**: 1 kernel change → "Compiling 1 of 10 kernels" → **5.4 seconds** (was ~15 min). 28/28 vllm-kernels CUDA unit tests pass (norm/activation/rotary/cache/quantize/moe). Chat smoke test OK. Paged FA2 bug is still open (see Priority 1).
- Auto-detects compute cap from `nvidia-smi` (single-arch). 24-thread parallel compilation. CUTLASS cached at `~/.cudaforge/git/checkouts/`.
- **Env override**: Set `CUDA_COMPUTE_CAP=89` for L40S, or let auto-detect handle it.
- **NOTE**: `touch` won't trigger recompilation — cudaforge uses content hashing, not mtime.

### E2E correctness tests — NEW
- `test_cuda_correctness_completion_semantic`: validates "paris" in "capital of France" completion
- `test_cuda_correctness_multi_turn_chat`: 2-turn chat, verifies model remembers turn 1 context
- `test_cuda_correctness_nongreedy_chat`: temperature=0.7, validates coherent text
- Added `assistant_msg()` helper, `assert_coherent_text` calls to existing CUDA tests

### Other completed work (prior sessions)
- CudaWorker model correctness fixes (QKV bias, arena copy ordering)
- Llama 3 RoPE scaling
- Configurable CUDA graph capture sizes
- Non-greedy GPU metadata fast path
- Pinned host staging buffers
- Transfer stream H2D overlap
- Persistent GPU decode metadata kernel
- cublasLt algorithm benchmarking (opt-in)
- GPU-aware max_num_batched_tokens
- Per-step allocation reduction
- Throughput bench prompt generation fix

## Performance (L40S, default bench: BS=8, in=32, out=128)

**Before paged FA2 workaround** (with CUDA graphs):

| Model | Temp | Mode | tok/s |
|-------|------|------|-------|
| Qwen2.5-0.5B | 0 | Graphs | 3639 |
| Qwen2.5-3B | 0 | Graphs | 807 |
| Qwen2.5-3B | - | Throughput | **17051 total tok/s** |

**After workaround** (enforce_eager, no graphs): Not yet benchmarked. Expected significant regression.

## Known issues

| Issue | Severity | Notes |
|-------|----------|-------|
| **Paged FA2 broken for multi-request decode** | **CRITICAL** | See Priority 1 above. Workaround in place but kills perf. |
| CUDA graphs disabled | High | Blocked by paged FA2 fix (gather does host sync). |
| Prefix caching disabled | Medium | Blocked by paged FA2 fix. |
| Prefill graphs disabled | Medium | Captured paged FA2; re-enable after fix. |
| Arena OOM at max_num_batched_tokens=8192 on 3B | Medium | Contiguous arena needs ~2.5GB. Chunked prefill would fix. |

## Key files

- cublasLt plan cache + benchmarking: `crates/vllm-cuda/src/cublas.rs`
- Fused QKV+RoPE kernel: `crates/vllm-kernels/csrc/pos_encoding_kernels.cu`
- Decode metadata kernel: `crates/vllm-kernels/csrc/embedding_kernels.cu`
- Model attention (with gather workaround): `crates/vllm-cuda/src/model/llama.rs`, `gemma2.rs`
- KV cache pool + gather: `crates/vllm-cuda/src/kv_cache.rs`
- CUDA graph runner: `crates/vllm-cuda/src/graph.rs`
- CUDA worker: `crates/vllm-executor/src/cuda_worker.rs`
- Config defaults: `crates/vllm-serve/src/init.rs`
- FA2 C kernel: `third_party/candle-flash-attn/kernels/flash_api.cu`
- FA2 Rust binding: `third_party/candle-flash-attn/src/lib.rs`
