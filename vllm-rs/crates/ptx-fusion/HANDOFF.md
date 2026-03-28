# Ferrite Phase 3→4 Handoff

## The Goal

Ferrite eliminates kernel launch overhead and GMEM round-trips in the llama.rs
forward pass by fusing adjacent operations at compile time via PTX escape
perimeter analysis.

**Current state**: 11 kernel launches per layer, 3.8s on Qwen2.5-3B (vs 3.48s cuBLAS).
Slower because standalone CUTLASS GEMMs lose to cuBLAS at decode, and bias_add is a
separate launch.

**Target state**: 6 launches per layer. The norm+GEMM fusions eliminate 2 GMEM
round-trips per layer (one for each rms_norm intermediate). The GEMM+SiLU fusion
eliminates 1 more. Launch overhead savings: ~25us/layer without CUDA graphs.

```
Current (11 launches):              Target (6 launches):
1. fused_add_rms_norm               1. fused_norm_qkv (norm+GEMM prologue)
2. QKV GEMM
3. split_qkv + RoPE + KV write      2. split_qkv + RoPE + KV (unchanged)
4. flash_attention                   3. flash_attention (unchanged)
5. O proj GEMM                      4. O proj + residual (beta=1.0, already works)
6. fused_add_rms_norm
7. gate_up GEMM                     5. fused_norm_gate_up_silu (norm+GEMM+SiLU)
8. silu_and_mul
9. down GEMM                        6. down proj + residual (beta=1.0, already works)
```

## What Has Been Built

### The `fuse!` proc macro (fuse_general.rs)

A compile-time macro that takes two PTX kernels + a binding and produces a
single fused kernel. Kernel-agnostic: it only sees escape perimeters.

```rust
ptx_fusion::fuse!(
    a = "kernels/rms_norm.ptx",
    b = "kernels/silu_mul.ptx",
    bind = { a.param_0 => b.param_1 },
    name = "fused_norm_silu",
    const = FUSED_PTX,
);
```

Five handoff paths, auto-selected from perimeter analysis:

| Pattern | Mechanism | Detection |
|---------|-----------|-----------|
| elem → elem (same thread map) | Register | Both no-SMEM/no-MMA, scalar store/load |
| elem → elem (different) | SMEM | One has SMEM or barriers |
| GEMM → elem pointwise | Epilogue injection | Producer has MMA + cvt.rn.bf16x2.f32 |
| pointwise → GEMM A-input | Inline at cp.async | Consumer has MMA + cp.async |
| rms_norm → GEMM A-input | Intrinsic prologue | (currently via separate macro) |

### The intrinsics shortcut

The general `fuse!` macro works by analyzing perimeters: find the bound
store/load sites, redirect the data through SMEM (or registers), merge params.
This is fully kernel-agnostic and works for any pair of kernels.

**But**: for rms_norm → GEMM prologue fusion, the general approach breaks down.
The rms_norm kernel has a cooperative reduction (sum of squares across the row),
and the reduction must use the GEMM's thread-to-row mapping (not rms_norm's own
mapping). You can't extract the reduction from rms_norm's PTX and inject it into
an arbitrary GEMM — the thread mappings won't align.

**The solution**: rms_norm becomes an **intrinsic** — the macro generates its PTX
from scratch, tailored to the GEMM's thread layout. The formula is simple
(sum_sq → rsqrt → normalize), and the thread-to-row mapping for CUTLASS
(lane/4 + warp*16, 4 threads cooperate per row) is known. This is NOT extracting
from a .ptx file — it's emitting PTX for a known formula.

This is the same pattern as the epilogue activations: SiLU, GELU, ReLU are also
intrinsics (hardcoded PTX emitters in fuse_epilogue.rs). The general extraction
path (extract computation between ld.global and st.global) handles arbitrary
pointwise consumers. Intrinsics handle operations that need tighter coupling
with the GEMM's internals (reductions, activations on f32 accumulators).

**What this means for future ops**: any operation that is purely pointwise (each
element independent) can go through the general `fuse!` path — extract from PTX,
no intrinsic needed. Operations with reductions (layer_norm, group_norm, softmax)
would need intrinsics, but they're all simple formulas.

### The `replace_a_loads_with_inline_fn` mechanism

The foundational mechanism for GEMM prologue fusion. Works inside the GEMM's
existing tiling loop — no staging buffer, no extra SMEM, no Megakernels-style
tile-at-a-time coordination loop.

At each A-matrix cp.async site (inside the GEMM's K-loop):
1. Load 16 bytes (8 bf16) from GMEM at the cp.async's source address
2. Unpack bf16 → f32
3. Apply a parameterized pointwise function (`PointwiseComputation`)
4. Repack f32 → bf16
5. Write to SMEM at the cp.async's destination address

The cp.async pipeline (commit_group/wait_group) becomes synchronous after
replacement, but the GEMM body and MMA instructions are completely untouched.
B-matrix cp.async loads are preserved.

`PointwiseComputation` has four fields:
- `instructions`: per-element PTX (parameterized on `{INPUT}`)
- `param_loads`: ld.param for the function's scalar params
- `prologue`: pre-loop reduction code (for rms_norm: compute inv_rms)
- `extra_reg_decls`: register declarations for prologue/computation

For pointwise-only functions (scale, bias): `prologue` is empty.
For rms_norm: `prologue` contains the cooperative reduction (sum_sq → rsqrt).

### Perimeter replacement compatibility

`replace_perimeter()` was fixed to preserve extra named params. When the rms_norm
intrinsic adds `_ferrite_rms_weight`, `_ferrite_rms_epsilon`, `_ferrite_rms_hidden`
as separate `.param` declarations alongside the CUTLASS struct param, the perimeter
replacement only touches the `.b8 NAME[368]` struct param (flattening it to
`ferrite_params[88]`). The named scalar params pass through untouched.

This means the chained pipeline works:
1. `build_rms_norm_gemm()` on original CUTLASS PTX → adds rms params + prologue
2. `replace_perimeter()` on the result → flattens CUTLASS struct to 88 bytes

The final kernel params: `_ferrite_rms_weight` (u64), `_ferrite_rms_epsilon` (f32),
`_ferrite_rms_hidden` (u32), `ferrite_params[88]`.

## Build and Test Methodology

### Compile-time pipeline

All fusion happens at `cargo build` via proc macros. The fused PTX is baked into
the binary as `const &str`. No runtime JIT, no graph tracing.

```
Source PTX → [proc macro at compile time] → Fused PTX (const &str) → ptxas → cubin
```

### Testing philosophy

Every PTX transformation has a **GPU correctness test** that launches the kernel
and compares against a reference. ptxas validation is necessary but not sufficient —
ptxas-valid PTX can still produce wrong output.

Test structure:
1. **Unit tests** (ptx-fusion-macros, `--lib`): 78 tests. Parse, classify, formula
   building, register offsetting. Pure Rust, no GPU needed.
2. **GPU tests** (ptx-fusion, `--features cuda`): 97 tests across 9 test files.
   Each test allocates GPU memory, launches kernels, compares output.
3. **End-to-end**: Qwen2.5-3B-Instruct produces correct text output.

For the `fuse!` tests specifically (`cuda_fuse_general.rs`, 19 tests):
- SMEM handoff: rms_norm→silu_mul at 10 sizes (1×128 to 256×3456), all 0.00e0
- Register handoff: rms_norm→scale at 6 sizes + determinism + vs special-purpose
- GEMM prologue identity: 4 sizes, 0.00e0 vs unmodified GEMM
- GEMM prologue scale*2: 4 sizes, GEMM(A*2,B) = 2*GEMM(A,B) at 0.00e0
- rms_norm + GEMM intrinsic: 0.00e0 vs CPU rms_norm + GPU GEMM

### How to run

```bash
cd vllm-rs

# Unit tests (fast, no GPU)
cargo test -p ptx-fusion-macros --lib

# GPU tests (need CUDA GPU)
cargo test -p ptx-fusion --features cuda --test cuda_fuse_general -- --nocapture
cargo test -p ptx-fusion --features cuda --test cuda_flat_gemm -- --nocapture

# All GPU tests
cargo test -p ptx-fusion --features cuda -- --nocapture

# End-to-end model inference
cargo run --release --features ferrite -- serve --model Qwen/Qwen2.5-3B-Instruct
```

## Interplay Between `fuse!` and the Persistent Wrapper

The `persistent.rs` module wraps a fused kernel in a persistent work-queue loop:
one kernel launch, all tiles processed via atomic tile counter. This was built
for the 3-phase MLP (norm→GEMM+SiLU→GEMM) where the persistent loop eliminates
launch overhead across all three phases.

**How persistent relates to fuse!**:

The persistent wrapper is ORTHOGONAL to fuse!. `fuse!` handles the data plumbing
(redirecting stores/loads between phases). The persistent wrapper handles the
execution model (one launch, tile loop, atomic dispatch). They compose:

```
fuse!(norm, gemm, bind=...) → fused PTX (one tile, one launch)
persistent!(fused_ptx)      → persistent PTX (all tiles, one launch)
```

Currently the 3-phase MLP (`fuse_3phase_mlp!`) uses the persistent wrapper
and achieves 0.79x (slower due to atomic/barrier overhead at small M).
The new Phase 3 fusion (fuse! + GEMM prologue) does NOT use the persistent
wrapper — it's a single-tile kernel launched with CUTLASS's grid dispatch.

**For the forward pass**: the persistent wrapper isn't needed for the 11→6
reduction. Each fused kernel (norm+GEMM, norm+GEMM+SiLU) uses CUTLASS's
own grid dispatch. The persistent wrapper becomes relevant if we want to
go from 6 launches to 1 launch (one persistent kernel per layer), but
that's a future optimization.

## Remaining TODOs for Victory

### Must-have: llama.rs integration (Phase 4)

**1. fused_norm_qkv**: `fuse_rms_norm_gemm_flat!` on the QKV GEMM config.
- The macro exists and is GPU-proven for 64x128x32 config
- Need to verify it works for the QKV GEMM dimensions (M=batch, N=7680, K=2560)
- Wire into `LlamaAttention::forward()`: replace `fused_add_rms_norm` + `qkv_proj.forward_ferrite()`
  with a single fused kernel launch
- The fused kernel takes: weight_ptr, epsilon, hidden_size, then ferrite_params[88]
  where A_ptr = raw input (not normalized)

**2. fused_norm_gate_up_silu**: composition of prologue + epilogue.
- Need: rms_norm prologue (intrinsic) + SiLU epilogue injection on same kernel
- The `build_rms_norm_gemm` handles the prologue
- `inject_activation_into_epilogue` handles the epilogue
- Chain: build_rms_norm_gemm → inject_silu → replace_perimeter
- Create a `fuse_rms_norm_gemm_silu_flat!` macro that chains all three
- This is the hardest remaining fusion (3 transformations composed)
- BUT: silu_and_mul takes TWO inputs (gate, up), not one. The gate_up GEMM produces
  [M, 2*intermediate], and silu_and_mul splits it. Injecting SiLU into the epilogue
  would only apply to HALF the output columns. This may need to stay as 2 separate
  GEMMs (gate GEMM with SiLU epilogue + up GEMM) or require the split to happen
  in the epilogue. **This needs further design work.**

**3. O proj + residual**: already works via beta=1.0 in flat params. Just need
to set beta=1.0 and pass the residual as C_ptr in the forward pass.

**4. down proj + residual**: same as O proj.

**5. Benchmark**: compare fused vs unfused latency. The target is beating
cuBLAS (3.48s) by enough to justify the complexity.

### Should-have: GPU correctness for epilogue injection

The GEMM epilogue injection path (GEMM → pointwise) only has a ptxas test,
not a GPU correctness test. Need to add a test that launches the fused
GEMM+scale kernel and compares output.

### Nice-to-have: general extraction for non-GEMM pointwise consumers

The epilogue injection currently uses `extract_pointwise_computation()` to
pull the consumer's function from its PTX. This is tested for scale.ptx
(ptxas valid) but not GPU-verified for arbitrary consumers. Testing with
SiLU/GELU consumers extracted from real PTX would strengthen confidence.

### Nice-to-have: more tile configs

The rms_norm intrinsic hardcodes CUTLASS register names (%r264 for tid.x,
%r280 for absolute M). These are correct for all 4 CUTLASS bf16 tile configs
(64x64x32, 64x128x32, 128x128x32, 128x128x64) because they share the same
thread-to-row mapping (PitchLinearWarpRakedThreadMap). But verification on
all configs would be good.

### Future: general reduction extraction

The current approach uses intrinsics for reductions (rms_norm from scratch).
A fully general approach would extract the reduction from the producer's PTX:
1. Identify the reduction phase (instructions using SMEM + barriers)
2. Identify the per-element phase (after last barrier, before st.global)
3. Re-emit the reduction using the GEMM's thread mapping

This is hard because the reduction's thread mapping must match the GEMM's.
The intrinsic approach works well for known operations (rms_norm, layer_norm)
and is the pragmatic path. General extraction is a research problem.

### Future: persistent layer kernel

Wrap the entire fused layer (all 6 launches) in a single persistent kernel.
This eliminates ALL launch overhead. Requires:
- Atomic tile counter across phases
- SMEM management across phases
- Careful barrier numbering

The persistent wrapper in `persistent.rs` handles single fused kernels.
Extending it to a full layer is significant work but the machinery exists.

## Key Files

```
ptx-fusion-macros/src/
  fuse_general.rs          General fusion engine (5 handoff paths, PointwiseComputation)
  intrinsic_rms_norm.rs    rms_norm intrinsic (reduction + normalize from scratch)
  fuse_epilogue.rs         Epilogue injection (SiLU, GELU, ReLU intrinsics)
  fuse_cp_async.rs         cp.async classification, explicit load replacement
  perimeter.rs             Flat-param perimeter replacement (preserves extra named params)
  parser.rs                PTX parser, escape perimeter extraction, def-use graph
  lib.rs                   All proc macros including fuse!, fuse_rms_norm_gemm_flat!

ptx-fusion/
  tests/cuda_fuse_general.rs  19 GPU correctness tests for all fusion paths
  tests/cuda_flat_gemm.rs     10 flat-param GEMM vs cuBLAS tests
  kernels/                    PTX files + derivations JSON

vllm-cuda/src/
  ferrite.rs               FerriteCutlass dispatcher (flat-param launch)
  layers.rs                Linear::forward_ferrite()
  model/llama.rs           Forward pass (currently 11 launches, target 6)
```
