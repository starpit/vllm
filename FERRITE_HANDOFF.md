# Ferrite Context Handoff — March 21, 2026

## What Ferrite Is

Ferrite is a Rust-native megakernel compiler for LLM inference. It generates
fused GPU kernels at compile time via proc macros. The forward pass of a
transformer layer becomes one (or few) kernel launches instead of 11.

**The value**: intermediates stay in registers/shared memory between operations.
No HBM round-trips between norm, GEMM, activation, GEMM, residual.

## Where the Code Is

```
~/vllm/.claude/worktrees/claude3/vllm-rs/
├── crates/
│   ├── ferrite-ptx/src/           # Layer 0+1: PTX code generation
│   │   ├── lib.rs                 # PtxBuilder: 50+ PTX instruction emitters
│   │   ├── config.rs              # GemmConfig (64×64, 128×128 tile configs)
│   │   ├── atoms.rs               # CopyAtom, TransformAtom, MmaAtom, EpilogueAtom
│   │   ├── pipeline.rs            # MainloopPipeline: THE single K-loop
│   │   ├── gemm.rs                # GEMM builders (standalone + pipeline)
│   │   ├── fused.rs               # Fused RmsNorm→GEMM→SiLU builder
│   │   ├── silu.rs                # SiLU standalone + epilogue phase
│   │   ├── gelu.rs                # GELU standalone + epilogue phase
│   │   ├── rmsnorm.rs             # RMSNorm standalone kernel
│   │   ├── convert.rs             # f32→f16 conversion kernel
│   │   ├── smem.rs                # Shared memory layout utilities
│   │   └── tile.rs                # Norm factor computation, tile utilities
│   │
│   ├── ferrite-macros/src/        # Layer 3: Proc macro fusion engine
│   │   ├── lib.rs                 # #[fuse] proc macro entry point
│   │   ├── ops.rs                 # OpGraph DAG with edge-based composition
│   │   ├── parse.rs               # Rust AST → OpGraph
│   │   ├── strategy.rs            # Edge classification → FusionPlan (Stages)
│   │   └── codegen.rs             # FusionPlan → PTX → TokenStream
│   │
│   ├── ferrite-runtime/src/       # Runtime: JIT kernel loading
│   │   ├── lib.rs
│   │   ├── kernel.rs              # JitKernel: PTX → cuModuleLoadData
│   │   └── tensor.rs              # Tensor type (stub)
│   │
│   └── ferrite-poc/src/           # POC: benchmarks + hand-written kernels
│       ├── main.rs                # All benchmarks (GEMM, fused, MLP, FA)
│       ├── gemm_128x128.rs        # Hand-written 128×128 GEMM + fused + MLP
│       ├── flash_attn.rs          # Hand-written flash attention (71 TFLOPS)
│       ├── triton_style.rs        # Hand-written 64×64 GEMM (55 TFLOPS)
│       ├── fuse_bench.rs          # Proc macro fused kernel benchmark
│       ├── fuse_test.rs           # Proc macro integration test
│       ├── mlp_bench.rs           # Proc macro MLP block benchmark
│       └── ptx_builder/           # Legacy PtxBuilder copy (pre-extraction)
│
├── FERRITE.md                     # Full design doc with results
└── FERRITE_HANDOFF.md             # This file
```

## What Works (Performance Results)

All benchmarks on NVIDIA L4 (SM89, Ada Lovelace):

| Kernel | TFLOPS | vs Reference |
|--------|--------|-------------|
| GEMM 64×64 | 55 | matches Triton exactly |
| GEMM 128×128 | 48 | matches Triton exactly |
| Flash Attention (d=64) | **71** | **1.04× faster than FA2** |
| Fused RmsNorm→GEMM→SiLU | 49 | 1.37× faster than torch.compile |
| MLP block (norm→GEMM→SiLU→GEMM) | 51 | 1.21× faster than torch.compile |
| SiLU standalone | 237 GB/s | matches PyTorch |
| GELU standalone | 237 GB/s | matches PyTorch |

## Architecture

### The Proven Process

Every kernel that matched/exceeded references followed this process:
1. Dump reference PTX (Triton, FA2, CUTLASS)
2. Study it line by line
3. Hand-write PTX that matches exactly
4. Verify identical performance
5. Then parameterize in PtxBuilder

**Do NOT skip steps 1-4.** Designing from the algorithm description produces
kernels that are 2-3× slower. The hand-written PTX IS the reference.

### Layer 0: PtxBuilder (`ferrite-ptx/src/lib.rs`)

Emits PTX instructions as formatted strings. No LLVM. Key features:
- 50+ instruction emitters (mma, ldmatrix, cp.async, ex2, etc.)
- Register allocator with scoping (`begin_scope`/`end_scope` for PTX `{ }` blocks)
- `finalize()` wraps with header, .reg declarations, kernel entry point

**Critical**: PTX is emitted as one flat scheduling region. Individual inline
asm blocks (like LLVM generates) create scheduling barriers that cost 1.77×.

### Layer 1: Pipeline + Atoms (`pipeline.rs`, `atoms.rs`)

The MainloopPipeline is the GEMM K-loop parameterized by atoms:
- `CopyAtom`: cp.async (SM89), future: TMA (SM90+)
- `TransformAtom`: Identity, RmsNorm (2× mul.rn.f16x2)
- `MmaAtom`: mma.sync.aligned.m16n8k16
- `EpilogueAtom`: Identity, SiLU (6 ALU), GELU (7 ALU)

**Standalone GEMM** = `Pipeline<CpAsync, CpAsync, Identity, Mma16816>`
**Fused RmsNorm→GEMM→SiLU** = `Pipeline<CpAsync, CpAsync, RmsNorm, Mma16816>` + SiLuEpilogue

Pipeline generates PTX that is 88% of hand-written quality (12% ptxas
scheduling gap). Hand-written kernels are used for peak performance.

### Layer 3: Proc Macro (`ferrite-macros/`)

`#[ferrite::fuse(arch = "sm_89")]` on a function:
1. **parse.rs**: Rust AST → OpGraph (real DAG with `Edge` connections)
2. **strategy.rs**: Walk edges, classify: elementwise→GEMM = TransformAtom,
   GEMM→elementwise = EpilogueAtom, GEMM→GEMM = intermediate via global/L2
3. **codegen.rs**: Generate PTX at compile time, embed as `const PTX: &str`

Adding new ops requires ONLY adding `OpKind` + `OpClass`. Zero strategy changes.
Tested: GELU and ResidualAdd added with zero strategy modifications.

### Flash Attention (`flash_attn.rs`)

Separate kernel, not part of the GEMM pipeline. Structure:
```
Prologue: Load Q → smem (stays for entire KV-loop)
KV-loop: For each K/V block:
  cp.async K → smem (double-buffered)
  S = Q @ K^T (MMA, f32 accumulators)
  Online softmax: max, exp2, sum, rescale O
  P stays in registers (f32→f16x2 conversion, no smem round-trip)
  cp.async V → smem
  O += P @ V (MMA)
Epilogue: O /= l_i, convert f32→f16, store
```

**Config matching FA2**: BLOCK_M=128, BLOCK_N=64, 4 warps (128 threads),
B128 XOR smem swizzle, double-buffered K/V. This EXACT config is what
exceeded FA2 — changing to 8 warps was 13% slower.

## Lessons Learned (CRITICAL — read before working)

### 1. Copy before innovating
When a reference implementation (Triton, FA2, CUTLASS) exists, copy its EXACT
PTX/configuration first. Do not design from the algorithm. Five fusion attempts
failed because of "simplifications" that removed critical elements.

### 2. LLVM is the wrong path
Individual inline asm blocks create ptxas scheduling barriers → 1.77× slower.
Emit PTX strings directly. ptxas is the real backend.

### 3. The pipeline abstraction works but has a 12% quality gap
Pipeline-generated PTX is 88% of hand-written quality. Both use 255 hardware
regs, nearly identical instruction counts (207 vs 204). The gap is from PTX
instruction ORDERING affecting ptxas's SASS scheduling heuristics. Not fixable
with PTX-level changes — it's a ptxas black box issue.

### 4. Tile size matters enormously
64×64 fusion: 0.48-0.69× of unfused (FAILED for months)
128×128 fusion: 1.04-1.37× of unfused (SUCCEEDED immediately)
CUTLASS uses 128×128. Copy their tile size.

### 5. f16x2 packed ops, not f32 unpack/repack
`mul.rn.f16x2` is 2 instructions per b32 register.
Unpack→f32 mul→repack is 12 instructions. CUTLASS uses `fma.rn.f16x2`.

### 6. Warp count is not "more is better"
Flash attention: 4 warps (128 threads) EXCEEDS 8 warps (256 threads).
FA2 uses 4 warps with more registers per thread → better scheduling.

### 7. Never fall back to separate kernels
The whole point is ONE kernel. If fusion is slow, fix the fusion approach,
don't emit separate launches.

### 8. Compare against torch.compile, not eager
Our MLP is 1.37× faster than torch.compile (honest), not 1.7× (vs eager).

## Test Inventory (155 total)

| Suite | Tests | What they cover |
|-------|-------|----------------|
| ferrite-macros | 19 | DAG edges, strategy, op classification |
| ferrite-ptx atoms | 20 | Each atom individually |
| ferrite-ptx config | 22 | Derived constants for both tile configs |
| ferrite-ptx pipeline | 17 | Instruction counts, scheduling |
| ferrite-ptx fused | 26 | PTX structure, block scoping |
| ferrite-ptx kernels | 26 | SiLU, GELU, RMSNorm, Convert |
| ferrite-poc flash_attn | 25 | FA structural + algorithmic invariants |

Run: `cargo test -p ferrite-ptx -p ferrite-macros` (130 tests)
Run: `cargo test -p ferrite-poc --bin ferrite-poc -- flash_attn` (25 tests)

## What's Next (in priority order)

### 1. Attention d=128 (REQUIRED for LLaMA)
LLaMA uses head_dim=128, our FA is d=64. Need to scale up.
Process: dump FA2's d=128 PTX (already compiled at /tmp/fa2_ptx/), study it,
replicate. The hdim128 variant is a separate .cu file in flash-attention.

### 2. Causal masking in attention (REQUIRED for generation)
FA2's PTX has separate masked/unmasked loop phases.
Add `setp` + conditional `-inf` store before softmax.

### 3. Easy elementwise ops (~1 day each)
- **Rotary embeddings**: `cos(pos*freq)*x + sin(pos*freq)*rotate(x)`. New atom.
- **SiLU × up**: `silu(gate) * up`. Elementwise multiply of two tensors.
- Both are `OpClass::Elementwise`, plug into DAG automatically.

### 4. Paged KV cache (REQUIRED for serving)
FA2 uses `block_table` for indirect addressing. Our FA has contiguous K/V.
Add block_table lookup to K/V address computation.

### 5. Integration with vllm-rs
Wire Ferrite kernels into `vllm-cuda/src/model/llama.rs` behind a feature flag.
Map Ferrite's `DevicePtr` to vllm-rs's `GpuTensor`/`TensorView`.

### 6. Close the 12% pipeline gap
The gap between pipeline-generated and hand-written PTX. Affects all
pipeline-generated kernels. Root cause: ptxas scheduling heuristics.
Diminishing returns — might need newer CUDA toolkit or SASS inspection.

## How to Run

```bash
cd ~/vllm/.claude/worktrees/claude3/vllm-rs

# All tests
cargo test -p ferrite-ptx -p ferrite-macros

# Flash attention tests
cargo test -p ferrite-poc --bin ferrite-poc -- flash_attn

# Full benchmark suite
cargo run --bin ferrite-poc --release

# Proc macro fused benchmark
cargo run --bin ferrite-fuse-bench --release

# Proc macro MLP benchmark
cargo run --bin ferrite-mlp-bench --release

# Triton comparison
~/.venv/bin/python3 triton_bench.py
~/.venv/bin/python3 triton_fused_comparison.py
```

## Key Files for Reference PTX

- `/tmp/triton_128x128.ptx` — Triton's 128×128 GEMM (our GEMM matches this)
- `/tmp/triton_flash_attn.ptx` — Triton's flash attention
- `/tmp/fa2_ptx/flash_fwd_hdim64_fp16_sm80.ptx` — FA2's compiled PTX (10MB, all variants)
- `/tmp/fa2_kernel.ptx` — FA2's main forward kernel extracted (10740 lines)
- `/tmp/ferrite_flash_attn.ptx` — Our flash attention PTX
- `/tmp/ferrite_fused_128x128.ptx` — Our fused RmsNorm→GEMM→SiLU PTX

## GPU

NVIDIA L4 (SM89, Ada Lovelace). CUDA driver 580.x (supports CUDA 13.0).
CUDA toolkit 12.0 (old — system ptxas is ancient but driver JIT is modern).
`~/.venv` has torch, triton, numpy. Use `uv pip` for package management.
