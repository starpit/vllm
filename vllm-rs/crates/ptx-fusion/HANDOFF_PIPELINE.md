# Pipeline Handoff — Session 2 Honest State

## What works

**Standard norms + CUTLASS GEMMs** — the ferrite path with `fused_add_rms_norm_inplace`
(the stock vllm kernel) for normalization and flat-param CUTLASS GEMMs for all linear
layers produces correct model output on both Qwen2.5-0.5B and 3B. This is committed at
`c6ed7adc5`. The CUTLASS GEMM perimeter replacement is solid — 0.00e0 vs cuBLAS with
real model weights (test1, test3 in `ferrite_gemm_real_weights.rs`).

## What doesn't work

**Fused norm+GEMM** (`compile!` with `a = rms_norm, b = gemm_64x128x32`) produces garbage
in production model inference. It passes all GPU tests at all dimensions (including
M=1024, K=2048, N=2560 with real weights — test2, test5). But when chained across 24
transformer layers in the actual model, small precision differences compound and the
output is garbage.

## Root cause (TWO bugs)

### Bug 1: Wrong kernel variant extracted

`PipelineStage::from_ptx("rms_norm", ptx)` calls `PtxParser::parse(source)` which
finds the FIRST `.entry` in the multi-entry PTX file. For `vllm_rms_norm.ptx`, the
entries are:

1. `_Z15rms_norm_kernelIfE...` — **f32 variant** ← this is what gets extracted
2. `_Z15rms_norm_kernelI6__halfE...` — f16 variant
3. `_Z15rms_norm_kernelI13__nv_bfloat16E...` — **bf16 variant** ← this is what production uses

The decomposition analyzes the f32 kernel. Production runs bf16. The f32 kernel uses
`ld.global.nc.v4.f32` (4 f32 values per load). The bf16 kernel uses different load
patterns (bf16 vec loads + f32 conversion). Even if we used the extracted code, it
would be the wrong code.

**Fix**: `PipelineStage::from_ptx` needs to accept an entry name hint (e.g., "bfloat16")
so the bf16 variant is extracted. Or the `compile!` macro should extract the correct
entry before calling `from_ptx`.

**File**: `crates/ptx-fusion-macros/src/pipeline.rs`, lines 114-133

### Bug 2: Hand-written prologue ignores extracted code

`build_prologue_from_decomposition()` in `pipeline_compile.rs` receives a
`ReductionDecomposition` that contains the ACTUAL accumulation loop body, finalization
code (warp shuffle + SMEM reduce + rsqrt), and emit body extracted from the real
kernel's PTX. It ignores all of it and writes new PTX from scratch:

- Lines 585-604: Hand-written K-loop with scalar `ld.global.nc.b16` loads (one element
  at a time, stride = ntid.x). The real kernel uses vectorized loads.
- Lines 605-620: Hand-written 5-round warp shuffle. The real kernel has a more complex
  reduction tree with shared memory.
- Lines 621-660: Hand-written 4-warp SMEM reduce. The real kernel's reduction is different.

The hand-written code produces slightly different floating-point results because:
- Different load granularity (scalar vs vectorized)
- Different accumulation order (different elements per thread)
- Same FMA instruction but applied to different partial sums

These tiny differences (< 0.01 per layer) compound over 24 layers and produce garbage.

**Fix**: Rewrite `build_prologue_from_decomposition` to emit the ACTUAL extracted PTX
lines from the decomposition:
- `decomp.accumulate_loops[*].header_line..backedge_line` → the real K-loop body
- `source_lines[decomp.finalize_range.0..finalize_range.1]` → the real reduction code
- Register names need prefixing to avoid collisions with the GEMM body

**File**: `crates/ptx-fusion-macros/src/pipeline_compile.rs`, lines 510-710

## What the decomposition gives you

(dumped from the f32 variant — bf16 will be similar but with bf16 loads + cvt)

**Accumulate loop 0** (vectorized, 4 elements per iteration):
```ptx
ld.global.nc.v4.f32 {%f17, %f18, %f19, %f20}, [addr];
fma.rn.f32 %f25, %f17, %f17, %f83;    // sq += x[0]^2
fma.rn.f32 %f26, %f18, %f18, %f25;    // sq += x[1]^2
fma.rn.f32 %f27, %f19, %f19, %f26;    // sq += x[2]^2
fma.rn.f32 %f83, %f20, %f20, %f27;    // sq += x[3]^2
```

**Accumulate loop 1** (scalar tail):
```ptx
ld.global.nc.f32 %f28, [addr];
fma.rn.f32 %f83, %f28, %f28, %f83;
```

**Finalize** (105 lines): warp shuffle reduction → SMEM broadcast → second warp shuffle
→ div + add eps + rsqrt → store to shared `s_inv_rms`.

**Emit body**: loads input and weight, multiplies by inv_rms and weight, stores output.

All of this is available in `ReductionDecomposition`. The prologue builder just needs
to emit it with register renaming.

## Adaptation challenge

The extracted code runs with the rms_norm kernel's original thread count and block
structure. The fused kernel runs inside a CUTLASS CTA with 128 threads. The extracted
code needs adaptation:

1. **Thread count**: The rms_norm kernel launches with `min(256, hidden_size)` threads.
   The CUTLASS CTA has 128 threads. The extracted accumulate loop uses `blockDim.x` for
   stride — with 128 threads, each thread handles more elements but the accumulation
   order changes. **This is the core precision issue.**

2. **Row iteration**: The original kernel processes one row per CTA. The fused prologue
   needs to process `tile_m` rows (one per CTA tile). The accumulate loop needs to be
   wrapped in an outer row loop.

3. **Input address**: The original kernel reads from `param_1` (input ptr). The fused
   kernel needs to read from `_ferrite_rms_input`.

4. **Shared memory**: The original kernel uses `_ZZ16block_reduce_sumfE6shared` for
   reduction scratch and `s_inv_rms` for the finalized value. These names need renaming
   to avoid collision with CUTLASS's shared memory.

The RIGHT approach: extract the bf16 variant's actual instructions, rename registers
and SMEM with a `_ferrite_rms_` prefix, adapt the input address, wrap in a row loop.
Do NOT rewrite the accumulation — use the exact same instructions in the exact same
order.

For the thread count issue: the original bf16 kernel at hidden=896 runs with 256
threads. With 128 CUTLASS threads, the accumulate loop's stride changes. This WILL
produce different partial sums. To get bitwise-identical results, either:
- Use a CUTLASS config with 256 threads
- Have each of the 128 threads do 2x work in the same order as 256 threads would
- Accept the precision difference and use a tighter tolerance (may still diverge over
  24 layers)

## Files

| File | What |
|------|------|
| `pipeline.rs:114-133` | `from_ptx` — extracts wrong (f32) entry |
| `pipeline.rs:185-280` | `decompose_reduction_impl` — the analysis (correct) |
| `pipeline_compile.rs:510-710` | `build_prologue_from_decomposition` — hand-written prologue (broken) |
| `pipeline_compile.rs:223-370` | `build_reduction_computation` — calls the broken builder |
| `vllm-cuda/tests/ferrite_gemm_real_weights.rs` | Tests 1-8 |
| `vllm-cuda/src/model/llama.rs` | Currently committed with standard norms + CUTLASS GEMMs (working) |
| `vllm-cuda/src/ferrite.rs` | `launch_fused_norm_gemm` uses per-arg kernelParams (correct) |

## Test commands

```bash
# All real-weight tests (tests 1-8)
cargo test -p vllm-cuda --features ferrite --test ferrite_gemm_real_weights -- --nocapture

# Working config (standard norms + CUTLASS GEMMs)
cargo run --release --features cuda,ferrite --bin vllm -- chat --model Qwen/Qwen2.5-0.5B-Instruct --enforce-eager --prompt "why is the sky blue?"

# Non-ferrite baseline
cargo run --release --features cuda --bin vllm -- chat --model Qwen/Qwen2.5-0.5B-Instruct --enforce-eager --prompt "why is the sky blue?"
```

## What to do next

1. Fix `from_ptx` to extract the bf16 entry (not f32)
2. Rewrite `build_prologue_from_decomposition` to emit actual extracted PTX
   with register/SMEM renaming — NOT hand-written PTX
3. Handle the 128-vs-256 thread count issue (the hardest part)
4. Run test8 (24-layer chain) and verify convergence
5. Run vllm serve and verify correct output
6. THEN wire silu_mul+GEMM_down into llama.rs as the second fused kernel
