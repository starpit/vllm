# Phase 4 Handoff: Fix the rms_norm intrinsic

## Status

The infrastructure works: `fuse!`, `compile!`, `ferrite.toml`, `FeriteKernel`,
`launch_fused_norm_gemm`, `forward_from_qkv`, `gemm_accumulate`, the llama.rs
fused forward path. The rms_norm intrinsic produces 0.00e0 at small dimensions
and passes compute-sanitizer with 0 OOB errors.

**But the model produces garbage.** The per-site inv_rms selection is wrong.

## The bug

Each thread in the GEMM handles 2 rows: `m_abs` and `m_abs + 8`. The prologue
computes `inv_rms_0` for row `m_abs` and `inv_rms_1` for row `m_abs + 8`.

At each A-load cp.async site, the per-site code needs to know: is this load
for row `m_abs` or row `m_abs + 8`? It selects the corresponding `inv_rms`.

**The wrong approach (current):** Compare `gmem_src < row_base_1`. This assumes
loads are ordered by row address. CUTLASS interleaves loads from both rows
within a K-tile — the ordering depends on the PitchLinearWarpRakedThreadMap
which is opaque to us.

**The right approach:** Don't guess. Use the PTX parser to trace each cp.async
site's `gmem_src` register back through the address computation chain. The
chain ends at the A-param and includes the row offset. Extract the row offset
from the traced chain. This is what ferrite's escape perimeter analysis does —
apply it here.

## What needs to happen

1. **Trace each A-load's address computation** using `PtxParser`. For each
   cp.async site that gets replaced, trace `gmem_src` back to identify which
   row (relative to the tile) it accesses. The parser already does this for
   param tracing — extend it to extract the row component.

2. **Pass the row info to the per-site code.** Either:
   - Add a `{ROW_PARITY}` placeholder (0 or 1) that `replace_a_loads_with_inline_fn`
     fills in per-site based on the traced row.
   - Or compute row from address at runtime using division (expensive but correct).

3. **Verify with compute-sanitizer** at model dimensions (M=256, N=2560, K=2048).

4. **Run the model end-to-end** and verify correct text output.

## Key files

- `intrinsic_rms_norm.rs` — the `rms_norm_computation()` factory. The `per_site`
  Vec<String> is where the inv_rms selection and weight address code lives.
- `fuse_general.rs` — `replace_a_loads_with_inline_fn()` iterates over cp.async
  sites and emits per-site code. This is where row tracing would happen.
- `fuse_cp_async.rs` — `classify_cp_async_loads()` already classifies A vs B.
  Could be extended to also extract row info.
- `parser.rs` — `PtxParser`, `trace_param_registers_pub()`. The address tracing
  infrastructure.

## Test commands

```bash
# Unit tests (fast)
cargo test -p ptx-fusion-macros --lib

# GPU correctness at all dimensions
cargo test -p ptx-fusion --features cuda --test cuda_fuse_general -- fused_norm_gemm --nocapture

# Compute sanitizer (check for OOB reads)
/usr/local/cuda-12.9/bin/compute-sanitizer --tool memcheck \
  cargo test -p ptx-fusion --features cuda --test cuda_fuse_general -- rms_norm_gemm_gpu_correctness --nocapture

# End-to-end model
cargo run --release --features ferrite --bin vllm -- serve --model Qwen/Qwen2.5-3B-Instruct
```

## The principle

**Do not guess at CUTLASS internals.** Ferrite's premise is escape perimeter
analysis — the kernel is a black box. Extract what you need from the PTX.
The swizzle bug, the load-parity bug, and the weight-address bug all came
from hardcoding assumptions about CUTLASS's thread mapping. Trace the PTX.
