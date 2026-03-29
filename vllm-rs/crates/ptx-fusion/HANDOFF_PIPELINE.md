# Pipeline Architecture Handoff (Session 2)

## What Was Built (10 commits)

This session completed the full MLP pipeline: single-kernel launch for
`rms_norm → GEMM_gate_up → barrier → SiLU+mul → GEMM_down`.

### Phase 5G-1: tile_m extraction + pipeline compiler wiring

- `extract_gemm_shape_from_entry()`: parses GemmShapeILi{M}ELi{N}ELi{K}E from CUTLASS mangled names
- `compile!` macro routes Reduction→TiledGemm and Pointwise→TiledGemm through pipeline compiler
- Deleted `intrinsic_rms_norm.rs` (286 lines of hand-written PTX)
- Added swizzle unswizzling to prologue: reads N from `ferrite_params[68]`, computes swizzle_log

### Phase 5G-2: SiLU+mul → GEMM fusion (Pointwise→TiledGemm)

`build_silu_mul_computation()` in `pipeline_compile.rs`:
- **Trick**: set GEMM_down's lda = 2*intermediate so A-loads naturally read gate columns
- Per-site: load up values at `GMEM_SRC + intermediate_bytes`
- Instructions: SiLU(gate) * up using fast exp approximation from fuse_epilogue.rs
- GPU-verified: **0.00e0 diff**

### Phase 5G-3: Two-phase persistent kernel assembler

`assemble_two_phase_kernel()` in `pipeline_compile.rs`:
- Parses each GEMM kernel into sections (header/params/regs/shared/body)
- Phase 2's `ferrite_params` → `ferrite_params2`, labels prefixed with `$L_p2_`
- Persistent tile loops with atomic counters per phase
- 2D tile decomposition: `rem.u32 %r_ptile_x, %r_ptile, %r_pgridx` + `div.u32`
- Global barrier: atomicAdd arrival + spin-wait between phases
- ~5700 lines of fused PTX, passes ptxas on sm_89

### Phase 5G-4: Production wiring

`mlp_pipeline_fuse!` proc macro + `launch_mlp_pipeline()` in ferrite.rs:
- Allocates gate_up intermediate buffer + counter/barrier u32s
- Packs 280-byte param buffer (two GEMM param sets + extras + persistent params)
- llama.rs: single `launch_mlp_pipeline()` call replaces 3 kernel launches

## Test Results

- 103 unit tests + 34 CUDA GPU tests, all passing
- MLP pipeline: 0.00e0 diff at all tested dimensions
- Multi-tile: (64,256,256), (128,128,256), (64,256,512) — all 0.00e0
- Benchmark: **2.17x speedup** at M=64, hidden=2560, intermediate=3456

## Key Files

```
pipeline_compile.rs   fuse_reduction_into_gemm(), fuse_pointwise_into_gemm(),
                      build_silu_mul_computation(), assemble_two_phase_kernel(),
                      parse_kernel_sections(), extract_gemm_shape_from_entry()
lib.rs                compile! (Reduction→TiledGemm + Pointwise→TiledGemm routing),
                      mlp_pipeline_fuse!, compute_extra_param_bytes()
ferrite.rs            launch_mlp_pipeline(), get_or_load_mlp_pipeline()
llama.rs              MLP_PIPELINE_PTX const, single-call MLP forward path
```

## What Remains

### End-to-end model inference
Wire into `vllm serve` and validate with Qwen2.5-3B-Instruct. The pipeline is
wired into llama.rs but hasn't been tested with a real model yet.

### Residual add fusion
The down GEMM could accumulate into the residual buffer (beta=1.0, C=residual)
to eliminate the separate add_inplace call. This requires careful ownership
handling since the residual tensor is passed through to the next layer.

### Larger tile configs
Currently hardcoded to 64x128x32. For prefill (large M), 128x128x32 or
128x128x64 tiles would be better. The pipeline compiler extracts tile dims
from the entry name, so adding configs is straightforward.

### Counter-reset optimization
The persistent loop currently resets counters via host memcpy before each
launch. A device-side reset (atomic exchange at kernel start) would avoid
this overhead.

## Test Commands

```bash
# Unit tests (103 tests)
cargo test -p ptx-fusion-macros --lib

# GPU tests (34 tests)
cargo test -p ptx-fusion --features cuda --test cuda_fuse_general -- --nocapture

# MLP pipeline only
cargo test -p ptx-fusion --features cuda --test cuda_fuse_general -- mlp_pipeline --nocapture

# Build vllm-cuda with ferrite
cargo build -p vllm-cuda --features ferrite
```
