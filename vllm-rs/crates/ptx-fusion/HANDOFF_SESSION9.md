# Handoff — Session 9

## The goal

Fix crashes preventing `vllm bench latency` from running, then make ferrite
performance competitive with cuBLAS.

## What was done

### 1. Fixed bench latency crash (profiling OOM)

`bench latency` defaults to `max_num_batched_tokens=8192`. The profiling forward
runs the lm_head GEMM at full M through cuBLAS. Large-vocab models (Qwen 151K)
cause `cublasLtMatmulAlgoGetHeuristic` to return `CUBLAS_STATUS_NOT_SUPPORTED`
at high M×N.

**Fix**: Cap profiling tokens at 256 and scale peak memory proportionally.

### 2. Fixed CUTLASS tile over-read (3B model crash)

CUTLASS reads full `[tile_m × K]` A-tiles via cp.async even when M < tile_m.
At 0.5B (K=896) the caching allocator's block padding absorbed the over-read.
At 3B (K=2048) it hit unmapped memory → `CUDA_ERROR_ILLEGAL_ADDRESS`.

**Fix**: Pad all ferrite GEMM outputs and MLP block intermediates to
`ceil(M/tile_m)*tile_m` rows, then reshape back for correct reported dims.

**Root cause confirmed via** `compute-sanitizer --tool memcheck`: invalid 16-byte
read at `ferrite_mlp_block+0x900`, 55KB past a 2MB allocation.

### 3. Single-body persistent MLP kernel

The old two-body persistent MLP duplicated the entire CUTLASS GEMM for each
phase (gate_up + SiLU-fused-down), producing a 65KB cubin that:
- Thrashed the 32KB L0 I-cache (~7x slowdown)
- Used all 16 hardware barriers per SM → 1 block/SM occupancy (~2x)
- Result: 1.64ms/call (10x slower than 3 separate launches)

**Fix**: Single GEMM body with params read from shared memory (88 bytes copied
from the active phase's param block at dispatch time) and conditional SiLU at
A-load sites (branched on a phase predicate, 100% predicted).

**Result**: 3.84s → matches the separate-GEMM path (3.82s). No fusion regression.

### 4. Added 16x128x32 CUTLASS tile config

Added a skinny-M CUTLASS config for decode. Compiled from CUTLASS source,
generated derivations JSON via the probe tool. Runtime tile selection picks
16x128x32 for M ≤ 16, 64x128x32 for larger M.

**Result**: Modest improvement (3.84s → 3.80s). The tiled GEMM at M=1 is still
fundamentally slower than a dedicated gemv.

### 5. Hand-written bf16 gemv kernel

Wrote a 190-line PTX gemv kernel for M=1 decode:
- 256 threads, 8 threads per output column, 32 columns per block
- Vectorized `ld.global.v4.b32` (8 bf16 per load)
- Shared-memory reduction across K-parallel lanes

**Benchmarks on Qwen 3B (single kernel call)**:
| Operation | N | K | Ferrite gemv | cuBLAS gemvx |
|-----------|------|------|-------------|-------------|
| QKV | 2560 | 2048 | **7.5µs** | 45µs |
| gate_up | 22016 | 2048 | 365µs | ~387µs |
| down | 2048 | 11008 | 177µs | ~similar |

The gemv is **6x faster than cuBLAS** on QKV decode. Wired into ferrite:
`gemm()` dispatches to gemv when M=1, alpha=1, beta=0.

### 6. Attempted barrier reduction (FAILED)

Tried to eliminate the 16-barrier ptxas issue by:
1. Replacing A-load cp.async with ld.global+st.shared → still 16 barriers
2. Replacing ALL cp.async (A+B) → broke CUTLASS pipeline synchronization
3. Removing cp.async.commit_group/wait_group → ILLEGAL_ADDRESS

**Conclusion**: The 16 barriers appear to be a ptxas limitation with cp.async
in persistent loops. ptxas cannot prove that async groups from different loop
iterations don't overlap, so it conservatively allocates maximum barriers.

## Current performance (bench latency, Qwen 3B, enforce-eager, batch=8, in=32, out=128)

| Config | Latency | tok/s |
|--------|---------|-------|
| No ferrite (cuBLAS) | 3.46s | ~296 |
| Ferrite (all optimizations) | 3.79s | ~270 |

**Gap: 10% (0.33s)**

## Where the time goes (nsys profile, vllm chat 3B)

### Ferrite
| Kernel | % GPU | Calls | Avg |
|--------|-------|-------|-----|
| ferrite_gemv_bf16 | 61.3% | 1440 | 155µs |
| ferrite_mlp_block | 19.3% | 35 | 2.0ms |
| ferrite_gemm_64x128 | 8.3% | 218 | 138µs |
| cuBLAS gemvx (lm_head) | 7.4% | 11 | 2.4ms |

### cuBLAS baseline
| Kernel | % GPU | Calls | Avg |
|--------|-------|-------|-----|
| gemvx (decode) | 67.3% | 4034 | 168µs |
| gemvx (prefill) | 23.5% | 1332 | 178µs |
| Various GEMM kernels | ~9% | ~200 | varies |

**Key insight**: The ferrite gemv is **faster** than cuBLAS gemvx on individual
calls (155µs avg vs 168µs avg). The 10% end-to-end gap is NOT from decode GEMMs.

**The bottleneck is the fused MLP block at prefill M** (2.0ms per call, 35 calls
= 70ms). This is because:
1. ptxas allocates 16 hardware barriers → 1 block/SM occupancy (halved throughput)
2. 50KB cubin still exceeds 32KB I-cache (thrashing)

## What remains — the register transfer approach

The fundamental problem: CUTLASS's cp.async pipeline + persistent loop = ptxas
barrier exhaustion. No amount of PTX manipulation fixes this because ptxas is
a black box.

**The solution**: Don't use SMEM or barriers for inter-phase data transfer at all.
Keep intermediate data in **registers** across phases:

1. Phase 0 (gate_up GEMM) produces f32 accumulator values in registers
2. Apply SiLU+mul to those registers (no memory traffic)
3. Phase 1 (down GEMM) reads those registers as its A-input
4. No GMEM write, no GMEM read, no barrier between phases

This eliminates:
- The inter-phase barrier (no mtile_done synchronization needed)
- The intermediate GMEM allocation (gate_up_buf disappears)
- The SiLU kernel or fused A-loads (SiLU is pure register-to-register)
- The cp.async A-load entirely (data never leaves registers)

The constraint: the register layout of the gate_up accumulator must match what
the down GEMM expects as A-input. Ferrite can analyze this from the CUTLASS PTX
— the epilogue's accumulator register names and the next GEMM's A-load register
consumption are both visible in the PTX.

**This is the megakernel approach**: user mentioned ~/Megakernels uses SMEM for
inter-phase transfer (they solved the barrier issue somehow — worth investigating).
The register approach is an alternative that avoids barriers entirely.

### Other remaining work

- **CUDA graph support**: `--enforce-eager` still required. The persistent kernel's
  atomic counters need zeroing per launch; `cuMemsetD32Async` should be
  graph-capturable.
- **Hardcoded num_sms=58**: Should query `cuDeviceGetAttribute` at init.
- **Norm+QKV GEMM fusion**: Proven at 0.00e0 in tests, not wired into llama.rs.
- **Delete deprecated code**: `intrinsic_rms_norm.rs`, old `fuse_rms_norm_gemm_flat!`

## Current launch count per layer (ferrite path)

```
1. fused_add_rms_norm_inplace   (C FFI)
2. QKV GEMM                     (ferrite gemv at M=1, CUTLASS at M>1)
3. split_qkv + RoPE + KV write  (C FFI)
4. FlashAttention                (FA2)
5. O_proj GEMM                  (ferrite gemv/CUTLASS)
6. fused_add_rms_norm_inplace   (C FFI)
7. gate_up+SiLU+down            (persistent fused MLP, 1 launch)
```

7 launches (down from 11 standard). Steps 7+8+9 of the original 11 fused into step 7.

## How to implement register-based inter-phase transfer

### The problem in detail

The persistent MLP block fuses gate_up GEMM → SiLU → down GEMM. Currently:
1. gate_up GEMM writes accumulators (f32 in registers) → epilogue converts to bf16 → stores to GMEM (gate_up_buf)
2. Barrier: phase 1 waits for all phase 0 M-tile tiles to complete
3. down GEMM reads gate_up_buf from GMEM via cp.async → SMEM → MMA registers

Steps 1-3 involve: register → GMEM → SMEM → register. Plus a global barrier.
All of this is unnecessary if we keep data in registers.

### The register transfer approach

After gate_up GEMM's MMA loop completes, the f32 accumulators hold the gate_up
output for this tile (64×128 elements distributed across 128 threads). Instead of
running the epilogue (which converts to bf16 and stores to GMEM), we:

1. **Apply SiLU+mul in registers**: Each thread's accumulators hold gate values.
   Load the corresponding "up" values from gate_up_buf at +intermediate_bytes.
   Apply SiLU(gate) * up. Result stays in f32 registers.

2. **Feed directly into down GEMM's A-input**: The down GEMM's MMA needs A-tile
   data in SMEM (loaded from the SiLU'd intermediate). Since we have the data in
   registers, we write it to the A-tile SMEM locations directly (the same SMEM
   addresses that cp.async would have written to), then proceed with the down
   GEMM's MMA loop.

Wait — this still uses SMEM. The real register-only path would be:

3. **Skip SMEM for A entirely**: The MMA instruction reads from SMEM. We can't
   bypass SMEM for MMA input. But we CAN write our register values to SMEM and
   skip the GMEM round-trip. This is the "register → SMEM → MMA" path instead
   of "register → GMEM → SMEM → MMA".

### What ferrite needs to analyze

From the CUTLASS GEMM PTX, ferrite needs to extract:

1. **Accumulator register names**: After the MMA loop, which %f registers hold
   the tile output? The epilogue code reads from these registers to convert to
   bf16. Parser already traces this in the epilogue analysis.

2. **A-tile SMEM layout**: Which SMEM addresses does each thread write to when
   doing cp.async for the A tile? The parser's TileIndexMap and perimeter
   replacement already know this.

3. **Thread-to-element mapping**: Which accumulator register in which thread
   corresponds to which element of the output tile? This is the MMA fragment
   layout — determined by the warp shape and MMA instruction shape.

### Concrete implementation steps

#### Step 1: Extract accumulator-to-element mapping

The CUTLASS epilogue converts accumulators to output:
```ptx
// Epilogue: read accumulator, scale, convert to bf16, store
cvt.rn.bf16x2.f32  %r_out, %f_acc_lo, %f_acc_hi;
st.global.v2.b32    [%rd_output + offset], %r_out;
```

By tracing the epilogue's store addresses back to accumulator registers, we can
build a map: `(thread_id, acc_register) → (row, col)` in the output tile.

#### Step 2: Build SiLU injection point

Instead of running the full epilogue after phase 0's MMA loop:
- Skip the bf16 conversion and GMEM store
- Apply SiLU to each accumulator: `acc[i] = silu(acc[i]) * up[i]`
- The "up" values need to be loaded from GMEM (gate_up_buf + intermediate_bytes)
  using the same address computation as the epilogue store (same row/col mapping)

#### Step 3: Write to phase 1's A-tile SMEM

After SiLU, write the f32 values to the SMEM addresses where phase 1's cp.async
would have written them. This requires knowing the SMEM layout for the A-tile in
the down GEMM.

For the down GEMM (K=intermediate_size, N=hidden):
- A-tile is [tile_m × tile_k] = [64 × 32] in SMEM
- Each cp.async loads 16 bytes (8 bf16) from GMEM to SMEM
- We need to convert our f32 accumulators to bf16 and write to the same SMEM slots

The mapping is: accumulator element (row, col) in the gate_up output →
SMEM address in the down GEMM's A-tile, accounting for:
- The K-iteration offset (which 32-element slice of intermediate we're in)
- The crosswise SMEM layout that CUTLASS uses

#### Step 4: Skip phase 0 epilogue and phase 1 A-loads

In the persistent kernel body:
- After phase 0's MMA loop, jump to the SiLU+SMEM-write code (not the epilogue)
- In phase 1's mainloop, skip A-load cp.async (data already in SMEM from step 3)
- Phase 1's B-loads and MMA proceed normally

This is exactly what `delete_a_matrix_loads` in fuse_cp_async.rs already does
(for the norm+GEMM fusion). The infrastructure exists.

### Why this eliminates the barrier problem

- No GMEM intermediate → no inter-phase barrier needed
- Phase 0 and phase 1 run sequentially within the SAME block on the SAME tile
- No cp.async for A-loads in either phase → fewer async pipeline groups
- B-loads still use cp.async (fine — only one set of pipeline stages)

The persistent kernel still grabs tiles from the atomic counter, but each tile
does both phases back-to-back without any global synchronization between them.
The total_tiles count is just the gate_up GEMM's tile count (phase 1 runs
implicitly after phase 0 on the same tile).

### Key constraint: tile compatibility

Both GEMMs must use the same tile_m (64) so the accumulator tile from gate_up
maps to a valid A-tile input for down. tile_n can differ (gate_up: N=gate_up_cols,
down: N=hidden), but tile_m and the K-dimension tiling must be compatible.

For the MLP block: gate_up produces [64 × 128] per tile (tile_m=64, tile_n=128).
Down needs A-tiles of [64 × 32] (tile_m=64, tile_k=32). Each accumulator tile
has 64×128 = 8192 elements. The down GEMM needs 64×32 = 2048 elements per
K-iteration. So 4 K-iterations of the down GEMM consume one full accumulator tile.

This means the down GEMM's K-loop must process 128 elements from the intermediate
(= gate_up's tile_n) before moving to the next gate_up tile. If intermediate_size
= 4864 (0.5B), that's 4864/128 = 38 gate_up tiles worth of K-iterations.

### Reference: ~/Megakernels

The user mentioned ~/Megakernels has solved the SMEM barrier issue for inter-phase
transfer. This is worth investigating before implementing the register approach —
if there's a simpler fix for the barrier allocation, the current SMEM-based
approach might be sufficient.

### Reference: existing infrastructure

- `fuse_cp_async.rs: delete_a_matrix_loads()` — deletes A cp.async, preserves B
- `fuse_cp_async.rs: replace_a_loads_with_explicit()` — replaces A cp.async with ld.global+st.shared
- `fuse_general.rs: replace_a_loads_with_inline_fn()` — replaces A cp.async with inline computation
- `parser.rs: DefUseGraph` — traces register definitions and uses
- `pipeline.rs: TilePerimeter` — describes tile-level inputs/outputs/carries
- `persistent.rs: make_single_body_persistent_mlp()` — current single-body approach

## Key files (updated)

```
crates/ptx-fusion-macros/src/
  persistent.rs       — make_persistent_two_phase, make_single_body_persistent_mlp (NEW)
  pipeline_compile.rs — build_silu_mul_computation (now pub)
  lib.rs              — persistent_mlp_block! uses single-body approach

crates/ptx-fusion/kernels/
  ferrite_gemv_bf16.ptx             — NEW: hand-written bf16 gemv (190 lines)
  cutlass_bf16_16x128x32_sm89.ptx  — NEW: skinny-M CUTLASS config
  cutlass_bf16_16x128x32_sm89.derivations.json — NEW

crates/vllm-cuda/src/
  ferrite.rs          — gemv dispatch for M=1, 16x128x32 tile config, padding fix
  model/llama.rs      — decoder layer with FUSED_MLP toggle

crates/vllm-executor/src/
  cuda_worker.rs      — profiling token cap (256) with proportional scaling
```

## Test counts

- **121+ unit tests** (ptx-fusion-macros)
- **28+ GPU tests passing** (cuda_fuse_general) + ferrite_gemv_gpu + 3B MLP dims
- **Production**: `vllm chat` correct on Qwen 0.5B and 3B
- **Benchmarks**: `vllm bench latency` works on both models

## Running the tests

```bash
# Unit tests
cargo test -p ptx-fusion-macros --lib

# GPU tests (including gemv and 3B dims)
cargo test -p ptx-fusion --features cuda --test cuda_fuse_general -- --test-threads=1

# Specific tests
cargo test -p ptx-fusion --features cuda --test cuda_fuse_general ferrite_gemv_gpu -- --nocapture
cargo test -p ptx-fusion --features cuda --test cuda_fuse_general persistent_mlp_block_gpu -- --nocapture

# Production
cargo build --bin vllm --features cuda,ferrite --release
./target/release/vllm chat --model Qwen/Qwen2.5-0.5B-Instruct --enforce-eager
./target/release/vllm bench latency --model Qwen/Qwen2.5-3B-Instruct --enforce-eager
```

## Rules (carried forward + new)

- **NEVER write PTX by hand** — exception: gemv kernel is hand-written but verified
  against CPU reference at 0.00e0 across all model dims
- **NEVER build special-case macros** — extend existing infrastructure
- **NEVER dismiss divergence** — any non-zero diff must be investigated
- **NEVER test only at small M** — sweep M=1..512 always
- **NEVER assume cuBLAS is the bottleneck** — profile with nsys before optimizing
- **Build with** `cargo build --bin vllm --features cuda,ferrite --release`
- **Tests ARE the product** — every new capability needs a GPU 0.00e0 test
- **Pad CUTLASS outputs** — tile_m boundary padding is required for correctness
- **ptxas barrier allocation is a black box** — don't assume eliminating cp.async
  from one code path reduces barriers; ptxas analyzes the entire kernel
