# Handoff: vllm-cuda Backend

Branch: `feat/cuda-qwen2-gemma2`
Worktree: `.claude/worktrees/cuda-models/`

## Crate Consolidation (c9f9a1888)

vllm-cuda is now the sole GPU backend. Major refactoring:

### Removed
- **CandleWorker** (`vllm-executor/src/candle_worker.rs`, `cuda_graph.rs`) — replaced by CudaWorker
- **vllm-kernels crate** — CUDA kernel sources (`csrc/`) and `build.rs` merged into `vllm-cuda`
- **candle-flash-attn** (`third_party/candle-flash-attn/`) — vllm-cuda uses `third_party/vllm-flash-attn/` + `flash-attn-shim/`
- **Candle model implementations** — all arch files deleted from vllm-models (llama.rs, qwen2.rs, gemma2.rs, deepseek_v2.rs, etc.), plus ops.rs, registry.rs, marlin_linear.rs, GPTQ/AWQ/BnB models

### Kept in vllm-models (shared infra)
- `Sampler`, `AttentionMetadata`, `KvBlockPool`, `KvCacheStorage`, `Model` trait
- `embedding` (PoolingStrategy), `grammar` (GrammarGuide)
- Used by both CudaWorker and MlxWorker

### Feature flags
- `cuda` is the single flag everywhere (was split `cuda`/`cuda-backend`)
- `vllm-cuda` crate: `cuda` = compile CUDA kernels + cudarc; without it, only metadata types
- TP (`--tensor-parallel-size > 1`) bails at runtime — old CandleWorker TP code preserved as comments in `init.rs`

### Crate graph (GPU path)
```
vllm-cli --features cuda
  → vllm-serve (cuda)
    → vllm-executor (cuda)
      → vllm-cuda (cuda) — GpuTensor, kernels, FA2, models
  → vllm-models — Sampler, AttentionMetadata, KvBlockPool (no GPU code)
```

### Build commands
```bash
# Local (no CUDA)
cargo build -p vllm-cli
cargo clippy --workspace --exclude vllm-pyo3 --exclude vllm-mlx -- -D warnings

# CUDA (on pod)
cargo build -p vllm-cli --features cuda
cargo clippy --workspace --exclude vllm-pyo3 --exclude vllm-mlx --features cuda -- -D warnings
cargo test -p vllm-cuda --features cuda              # 145 kernel tests
cargo test -p vllm-e2e --features e2e,cuda --release --test e1_basic_serving test_cuda -- --ignored --test-threads=1  # 12 E2E
```

---

## Paged FlashAttention-2 — FIXED ✅

Paged FA2 fully working. Multi-turn chat, non-contiguous block tables, all GQA configs — verified on L40S.

- 17 CUDA unit tests passing (all `max_diff=0.000000`)
- 14 CUDA E2E tests passing (including multi-turn, 3-turn, interleaved users)
- `vllm chat` multi-turn verified manually with Qwen2.5-0.5B-Instruct

## Root Cause

The FFI shim (`ffi_shim.cu`) called `run_mha_fwd_()` — the **standard** FA2 kernel (`compute_attn_1rowblock`). This kernel has **no `block_table` support** — it treats K/V as contiguous memory and ignores the block table entirely. Only the **splitkv** kernel (`compute_attn_1rowblock_splitkv`) supports paged KV via `block_table` + `resolve_thread_kv_page_slice_offset`.

Python vLLM masked this because its `seqlenq_ngroups_swapped` optimization (applied for all GQA decode) always routes through `set_params_splitkv`, which invokes the splitkv kernel.

With contiguous block tables `[0,1,2]` the standard kernel happened to work (memory is contiguous anyway). With non-contiguous mappings `[2,0,1]` it read from wrong physical blocks → garbled output.

## Fix (commit 013cba535)

1. **`ffi_shim.cu`**: Added `force_split_kernel` parameter to `run_mha_fwd()`, matching upstream `flash_api.cpp`. When `block_table != nullptr`, force the splitkv kernel:
   ```cpp
   run_mha_fwd(params, stream, /*force_split_kernel=*/paged);
   ```

2. **`kernels.rs`**: Removed dead `run_mha` extern (candle-flash-attn symbol). Rewrote `flash_attn_contiguous()` to use `mha_varlen_fwd` with null block_table — makes vllm-cuda self-contained.

3. **Test fix**: `test_noncontiguous_blocks` used `block_size=4` which violates upstream requirement (`block_size % 16 == 0`). Fixed to `block_size=16, kv_len=32`.

## Test Coverage

### Unit tests (17, all in `crates/vllm-cuda/src/kernels.rs`)

| Test | Config | Validates |
|------|--------|-----------|
| `test_mha_varlen_fwd_contiguous_basic` | MHA, contiguous | Basic non-paged path |
| `test_mha_varlen_fwd_paged_basic` | MHA, 1 block | Single-block paged |
| `test_mha_varlen_fwd_paged_noncontiguous_blocks` | MHA, bt=[2,0], bs=16 | Non-contiguous blocks |
| `test_mha_varlen_fwd_paged_batch2` | MHA, batch=2 | Batched paged |
| `test_mha_varlen_fwd_paged_gqa` | GQA 14:2 | GQA + paged |
| `test_paged_vs_contiguous_gqa` | GQA 14:2, [0,1,2] vs [1,2,0] | Shuffle equivalence |
| `test_paged_shuffle_gqa_7to1_hdim64` | GQA 14:2, d=64 | Qwen2.5-0.5B config |
| `test_paged_shuffle_gqa_4to1_hdim128` | GQA 32:8, d=128 | Common GQA config |
| `test_paged_shuffle_mha_hdim64` | MHA 8:8, d=64 | Non-GQA + splitkv |
| `test_paged_shuffle_hdim32` | GQA 8:2, d=32 | Small head dim |
| `test_paged_shuffle_hdim128_gqa` | GQA 32:4, d=128 | Large head dim |
| `test_paged_shuffle_single_block` | 1 page | Edge case |
| `test_paged_shuffle_many_blocks` | 5 pages | Multi-block |
| `test_paged_shuffle_multi_tile` | kv=200, 13 pages | Spans 2 kBlockN tiles |
| `test_paged_shuffle_batch4` | batch=4 | Batched shuffle |
| `test_paged_shuffle_batch2_gqa_hdim128` | batch=2, GQA, d=128 | Combined stress |
| `test_contiguous_varlen_fwd` | null block_table | Contiguous via mha_varlen_fwd |

### E2E tests (in `crates/vllm-e2e/tests/e1_basic_serving.rs`)

| Test | Validates |
|------|-----------|
| `test_cuda_paged_fa2_three_turn_chat` | 3-turn conversation, recalls "42", computes 42*2=84 |
| `test_cuda_paged_fa2_interleaved_multi_turn` | 2 users, interleaved requests, both recall facts |
| `test_cuda_correctness_multi_turn_chat` | (existing) 2-turn, recalls "Claude" |

## Architecture

- `third_party/vllm-flash-attn/src/` — upstream kernel source (unchanged)
- `third_party/flash-attn-shim/ffi_shim.cu` — thin FFI shim (~200 lines), our only custom CUDA
- `third_party/flash-attn-shim/compat/` — PyTorch header stubs
- `crates/vllm-cuda/build.rs` — cudaforge incremental build, compiles upstream + shim → `libvllm_flash_attn.a`

### Key design decisions

- ffi_shim.cu includes only `flash.h` (declarations) not `flash_fwd_launch_template.h` (definitions)
- CUTLASS pinned to `62750a2b` (upstream's exact submodule commit)
- `seqused_k` (per-seq K lengths) instead of `cu_seqlens_k` (cumulative) for paged attention
- `force_split_kernel=true` when `block_table != nullptr` — mandatory, standard kernel has no paging
- `num_splits=1` always (split-K accum buffers not yet wired from Rust side)

## Build/test commands

```bash
# Unit tests (all 17, ~2 sec after first build)
cargo test -p vllm-cuda --features cuda --release -- --ignored --nocapture

# E2E tests (paged FA2 only, ~6 sec)
cargo test -p vllm-e2e --features e2e,cuda --release --test e1_basic_serving test_cuda_paged_fa2 -- --ignored --test-threads=1

# All CUDA E2E tests (~45 sec)
cargo test -p vllm-e2e --features e2e,cuda --release --test e1_basic_serving test_cuda -- --ignored --test-threads=1

# Manual multi-turn chat
echo -e "What is 2+2?\nWhat is 3+3?" | ./target/release/vllm chat --model Qwen/Qwen2.5-0.5B-Instruct --max-tokens 20
```

Pod: `oc rsh nick` — L40S, CUDA 12.9, path `/root/vllm/vllm-rs/`. Always `export RUSTC_WRAPPER=/usr/bin/sccache`.

## Upstream References

- Python vLLM flash-attn pin: `cmake/external_projects/vllm_flash_attn.cmake` → commit `5824e6e2`
- CUTLASS pin: `62750a2b`
- Key upstream file: `csrc/flash_attn/flash_api.cpp` → `mha_varlen_fwd()` (line 516)
- Standard kernel (NO paging): `flash_fwd_kernel.h` → `compute_attn_1rowblock` (line 52)
- SplitKV kernel (HAS paging): `flash_fwd_kernel.h` → `compute_attn_1rowblock_splitkv` (line 499)
- Page resolution: `utils.h` → `resolve_thread_kv_page_slice_offset` (line 300)
