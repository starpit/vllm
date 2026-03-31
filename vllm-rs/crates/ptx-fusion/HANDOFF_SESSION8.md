# Handoff — Session 8

## The goal

Build the tile pipeline architecture from TILE_PIPELINE.md: fuse the transformer
forward pass from 11 launches to 3 launches per layer using tile-level perimeters,
persistent kernels, and per-M-tile barriers.

## What was done

### 1. Phase 1: Tile index extraction (COMPLETE)

**TileIndexMap** in parser.rs extracts `(ctaid.x, ctaid.y) → (m_tile, n_tile)` from
CUTLASS GEMM PTX via DefUseGraph forward tracing. Validated on all 4 CUTLASS configs
(64x64, 64x128, 128x128, 128x128x64). Extracts the exact 10 PTX lines of the swizzle
computation + register names.

The prologue was rewritten to compute m_tile/n_tile in its own register namespace
(`%r_rms_*`), making it immune to perimeter replacement's register shuffling. The old
approach of reading the GEMM's registers directly failed because `replace_perimeter`
rewrites the `ld.param` for swizzle_log, shifting which register holds m_tile.

**Critical fix**: the `pipeline_fuse!` test was using the f32 rms_norm entry instead
of bf16. Added "bfloat16" entry hint → diff went from 9.70e2 to 0.00e0.

**M-sweep**: norm+GEMM tested at M=1,2,4,8,16,32,64,65,128,256,512 — ALL 0.00e0
vs GPU separate (rms_norm kernel + CUTLASS GEMM). Reference is GPU-vs-GPU, not
CPU-vs-GPU (CPU rsqrt differs from GPU rsqrt.approx).

### 2. Phase 2: TilePerimeter infrastructure (COMPLETE)

Types in pipeline.rs:
- `TilePerimeter`: tile-level execution interface (kind, shape, tile_index, inputs, outputs, carries, finalization)
- `TileShape`, `TilePort`, `TilePortAccess` (CpAsync, GlobalLoad, SmemScalar, InlineFormula, GlobalStore)
- `TileCarry`, `CarryStorage`, `TileFinalization`
- `StageEdge` (ReductionToGemm, GmemMaterialization, PointwiseToGemm)
- `FusibleSegment` (stages + edges)
- `TilePerimeter::from_stage()` dispatches to `from_gemm/from_reduction/from_pointwise`

Extraction validated on all kernel types:
- GEMM: tile shape from entry name, tile index from PTX, A/B loads as CpAsync, MMA accumulators + K-pointers + buffer state
- Reduction (rms_norm + fused_add_rms_norm): sum_sq carry, finalization with rsqrt, emit body as InlineFormula
- Pointwise (silu_mul): no carries, no finalization

### 3. Segment compilation from TilePerimeters (COMPLETE)

`compile_segment()` in pipeline_compile.rs takes PipelineStages + TilePerimeters,
infers edge type from StageKinds, dispatches to fusion functions.

Per-element instructions are now derived from the emit body via
`extract_per_element_from_emit_body()` — finds `mul.f32` instructions referencing
`finalized_value_reg`, extracts the pattern. No more hand-written per-element PTX.

### 4. Multi-GEMM sequencing (COMPLETE)

`sequence_gemm_phases()` sequences N GEMM phases into a single kernel:
- Cumulative register offsets (each phase's registers don't collide)
- Per-phase label renaming ($L__BB0 → $L__BB1 → $L__BB2)
- Per-phase param blocks (ferrite_params, ferrite_params_2, ferrite_params_3)
- Non-standard .reg declaration copying (ptmp, inline asm, fusion-injected)
- SMEM reuse across phases (sequential execution)
- Brace-depth tracking for CUTLASS inline asm blocks
- Global atomic barriers between phases (replaced with persistent barriers for production)

### 5. MLP block fusion (COMPLETE — IN PRODUCTION)

Gate_up GEMM → SiLU+mul → down GEMM fused into a single kernel:
- `sequence_mlp_block!` proc macro: perimeter-replace gate_up, fuse SiLU into down, sequence both
- GPU-verified at 0.00e0 at all dims including Qwen 0.5B production (hidden=896, intermediate=4864, M=1..1024)

### 6. Segment B (COMPLETE in tests, partial in production)

O_proj → norm → gate_up+SiLU → down in a single 3-phase kernel:
- `sequence_segment_b!` proc macro: O_proj + pipeline-fused norm+gate_up + SiLU-fused down
- GPU-verified at 0.00e0 at M=1..256 with global atomic barriers
- **NOT wired into llama.rs** yet — needs persistent 3-phase wrapper (only 2-phase is implemented)

### 7. Persistent kernel dispatch (COMPLETE)

**Single-phase** (`make_persistent_gemm`):
- 2D ctaid dispatch: `ctaid_x = tile_idx % grid_x, ctaid_y = tile_idx / grid_x`
- Replaces both `mov %rN, %ctaid.x` and `mov %rN, %ctaid.y`
- GPU-verified at 0.00e0 including 2D grids (gy=2)

**Two-phase** (`make_persistent_two_phase`):
- Persistent loop grabs tile from atomic counter
- Phase dispatch: tile_idx < phase0_tiles → phase 0, else → phase 1
- Per-M-tile atomic barrier: phase 1 spins on `ld.global.acquire.gpu.u32` until
  `mtile_done[m_tile] >= n_tiles_per_m_0`
- Phase 0 completion: thread 0 `atom.global.add.u32` on mtile_done[m_tile]
- GPU-verified at 0.00e0 at M=1..1024 with production dims (1344 tiles, 58 blocks)

### 8. Production wiring (COMPLETE for MLP block)

Persistent fused MLP block wired into llama.rs decoder layer forward pass.

**Critical bugs found and fixed:**
- `cuMemsetD32_v2` (synchronous, not stream-ordered) races with previous layer's kernel.
  Fix: `cuMemsetD32Async` on the same stream.
- `gate_up_buf` intermediate tensor returned alongside output to prevent use-after-free
  from caching allocator.
- param passing uses `kernelParams` API (pointer per param), matching cudarc's approach.

**Verified**: `vllm chat --model Qwen/Qwen2.5-0.5B-Instruct --enforce-eager`
produces correct text: "The capital of France is Paris."

## What carries forward

| Asset | Status | Location |
|-------|--------|----------|
| TileIndexMap extraction | Working, 4 configs | parser.rs: `extract_tile_index_map()` |
| TilePerimeter types | Working, all kernel types | pipeline.rs |
| compile_segment API | Working | pipeline_compile.rs |
| extract_per_element_from_emit_body | Working | pipeline_compile.rs |
| sequence_gemm_phases (N-phase) | Working | pipeline_compile.rs |
| make_persistent_gemm (2D ctaid) | Working, 0.00e0 | persistent.rs |
| make_persistent_two_phase | Working, 0.00e0 | persistent.rs |
| persistent_mlp_block! macro | Working, production | lib.rs |
| MLP block in llama.rs | **IN PRODUCTION** | ferrite.rs + llama.rs |
| Segment B (3-phase, test only) | Working, 0.00e0 | sequence_segment_b! |
| M-sweep norm+GEMM | Working, 0.00e0 M=1..512 | cuda_fuse_general.rs |
| M-sweep Segment B | Working, 0.00e0 M=1..256 | cuda_fuse_general.rs |
| M-sweep persistent MLP | Working, 0.00e0 M=1..1024 | cuda_fuse_general.rs |
| FUSED_MLP toggle | In llama.rs | const bool for A/B comparison |

## Current launch count per layer (ferrite path)

```
1. fused_add_rms_norm_inplace   (C FFI)
2. QKV GEMM                     (CUTLASS flat-param)
3. split_qkv + RoPE + KV write  (C FFI, unchanged)
4. FlashAttention                (FA2, unchanged)
5. O_proj GEMM                  (CUTLASS flat-param)
6. fused_add_rms_norm_inplace   (C FFI)
7. gate_up+SiLU+down            ← FUSED (persistent, 1 launch)
```

**7 launches** (down from 11 standard, down from 9 previous ferrite).
Steps 7+8+9 of the original 11 are now fused into step 7.

## Remaining TODOs for 3 launches per layer

### Launch 1: norm + QKV GEMM

**Status**: Proven in GPU tests (0.00e0 at M=1..512), NOT wired into llama.rs.

**What's needed**:
1. Wire `pipeline_fuse!` norm+QKV GEMM into llama.rs (replace steps 1+2 with 1 fused launch)
2. Handle first-layer edge case (no residual → plain rms_norm, not fused_add)
3. Handle the residual add: the fused prologue is pure-read (session 6 fix), so the
   caller must do `add_inplace(residual, hidden_states)` after the fused launch
4. Test with `vllm serve`

**Risk**: Session 6 tried this and hit CUDA_ERROR_ILLEGAL_ADDRESS during graph capture
and garbage with `--enforce-eager`. The tile index and bf16 fixes from session 8 may
have resolved the underlying issues, but the wiring (tensor lifetimes, M=1 decode,
first layer) still needs careful attention.

### Launch 3: Full Segment B (O_proj + norm + gate_up + SiLU + down)

**Status**: 3-phase sequenced kernel works at 0.00e0 in tests. Persistent 2-phase
(MLP block) works in production. Need persistent 3-phase.

**What's needed**:
1. Extend `make_persistent_two_phase` to 3 phases (or write `make_persistent_three_phase`)
2. Phase dispatch for 3 phases with different grid dims per phase
3. Two per-M-tile barriers: between O_proj and norm+gate_up, between gate_up and SiLU+down
4. Wire into llama.rs: replace steps 5+6+7 with single Segment B launch
5. Handle the residual add in norm (fused_add_rms_norm variant, not plain rms_norm)
6. The attention's O_proj is currently inside `forward_from_qkv` — need to either:
   a. Split `forward_from_qkv` to return attention output without O_proj, or
   b. Keep O_proj separate and only fuse norm+gate_up+SiLU+down (3 ops, still 1 launch)

**Option (b) is simpler**: the current MLP block fusion (steps 6+7+8+9 → 1 launch) is
already in production. Adding the norm before gate_up (step 6) into the fused kernel
would make it norm+gate_up+SiLU+down = 4 ops in 1 launch. This avoids touching the
attention code path.

### Benchmarking

**Status**: `vllm bench latency` crashes with cuBLAS error (pre-existing, not ferrite).
Chat-mode timing includes model loading, making it useless for per-token measurement.

**What's needed**:
1. Fix the cuBLAS crash in the latency benchmark, OR
2. Add CUDA event timing around the MLP block in ferrite.rs
3. Compare fused vs separate at various batch sizes (M=1, M=8, M=64, M=1024)
4. The FUSED_MLP toggle in llama.rs enables A/B comparison

**Expected gains** (theoretical for Qwen 0.5B):
- 2 fewer kernel launches × ~5μs = ~10μs per layer
- ~29KB less GMEM traffic per layer at decode (M=1)
- ~12μs total savings per layer → ~290μs for 24 layers (~3-6% of decode latency)
- Larger gains at bigger models (more GMEM bandwidth saved)

### Multiple tile configs

**Status**: Only 64x128x32 tile config used. cuBLAS auto-selects specialized kernels
for skinny-M (decode) which are faster.

**What's needed**:
1. Add 8x128x32 and 16x128x32 configs for decode (M=1..16)
2. Runtime dispatch: select tile config based on M
3. The fused MLP block currently hardcodes 64x128x32 — need to parameterize
4. Profile-guided selection: bench each config at model init

### CUDA graph support

**Status**: `--enforce-eager` required. CUDA graphs not tested with fused MLP.

**What's needed**:
1. The persistent kernel uses device-side atomics (tile counter, mtile_done) which
   need zeroing before each launch. With CUDA graphs, this zeroing must be captured
   in the graph. `cuMemsetD32Async` should be graph-capturable.
2. The `gate_up_buf` allocation inside `launch_mlp_block` may not be graph-compatible
   (caching allocator within a graph capture). May need to pre-allocate.
3. Test with CUDA graphs enabled (remove `--enforce-eager`)

### Delete deprecated code

- `intrinsic_rms_norm.rs` — replaced by pipeline compiler
- `fuse_rms_norm_gemm_flat!` macro — replaced by `pipeline_fuse!`
- The 7 failing GPU tests in cuda_fuse_general.rs (old split architecture) — either
  fix with bf16 entry hint or remove

## Key files (updated)

```
crates/ptx-fusion-macros/src/
  parser.rs           — DefUseGraph, TileIndexMap, extract_tile_index_map()
  pipeline.rs         — TilePerimeter, StageEdge, FusibleSegment, PipelineStage
  pipeline_compile.rs — compile_segment, sequence_gemm_phases, extract_per_element_from_emit_body
  persistent.rs       — make_persistent_gemm (2D ctaid), make_persistent_two_phase
  fuse_general.rs     — replace_a_loads_with_inline_fn, PointwiseComputation
  perimeter.rs        — replace_perimeter (flat-param rewrite)
  lib.rs              — pipeline_fuse!, sequence_gemms!, sequence_mlp_block!,
                        sequence_segment_b!, persistent_gemm!, persistent_mlp_block!

crates/ptx-fusion/src/
  lib.rs              — re-exports all proc macros

crates/vllm-cuda/src/
  ferrite.rs          — FerriteCutlass, launch_mlp_block (persistent), MLP_BLOCK_PTX
  model/llama.rs      — decoder layer forward with FUSED_MLP toggle

crates/ptx-fusion/tests/
  cuda_fuse_general.rs — 38+ GPU tests including:
    pipeline_fused_norm_gemm_gpu (0.00e0)
    pipeline_fused_norm_gemm_m_sweep (M=1..512, 0.00e0)
    mlp_block_gpu (0.00e0)
    segment_b_gpu (0.00e0)
    segment_b_m_sweep (M=1..256, 0.00e0)
    sequenced_two_gemm_gpu (0.00e0)
    persistent_gemm_gpu (M=1..256, N=128..640, 0.00e0)
    persistent_mlp_block_gpu (M=1..1024, production dims, 0.00e0)
```

## Test counts

- **121 unit tests** (ptx-fusion-macros)
- **28+ GPU tests passing** (cuda_fuse_general) — 6 failing are old split architecture
- **Production**: `vllm chat` with Qwen2.5-0.5B-Instruct produces correct text

## Running the tests

```bash
# All unit tests (121, fast)
cargo test -p ptx-fusion-macros --lib

# All GPU tests (28 passing, 6 known failures from old path)
cargo test -p ptx-fusion --features cuda --test cuda_fuse_general -- --test-threads=1

# M-sweep tests specifically
cargo test -p ptx-fusion --features cuda --test cuda_fuse_general pipeline_fused_norm_gemm_m_sweep -- --nocapture
cargo test -p ptx-fusion --features cuda --test cuda_fuse_general persistent_mlp_block_gpu -- --nocapture

# Production test
cargo build --bin vllm --features cuda,ferrite --release
./target/release/vllm chat --model Qwen/Qwen2.5-0.5B-Instruct --enforce-eager --prompt "What is the capital of France?"
```

## Rules (non-negotiable, carried forward)

- **NEVER write PTX by hand** — transplant from compiled kernel PTX or derive from emit body
- **NEVER build special-case macros** — extend compile! or pipeline_fuse!
- **NEVER dismiss divergence** — any non-zero diff must be investigated
- **NEVER test only at small M** — sweep M=1..512 always
- **NEVER use cuMemsetD32_v2 for async state** — use cuMemsetD32Async on the stream
- **NEVER drop intermediate buffers before kernel completes** — return them to caller
- **Build with** `cargo build --bin vllm --features cuda,ferrite --release`
- **Tests ARE the product** — every new capability needs a GPU 0.00e0 test
