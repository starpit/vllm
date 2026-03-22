# Ferrite Metal: Megakernel Fusion on Apple Silicon

> Companion to FERRITE.md (CUDA). Parallel development track.

---

## Architecture

Same proc macro → atom DAG → lowering pipeline as CUDA Ferrite, but:
- **Layer 0**: MslBuilder (emits MSL strings) instead of PtxBuilder (emits PTX strings)
- **Stage 3**: More work needed (Metal compiler doesn't do ptxas-level scheduling)
- **Stage 4**: MSL → metal compiler → AIR → driver finalizer (vs PTX → ptxas → SASS)

```
#[ferrite::fuse]                          ← shared proc macro
fn transformer_block(x, w, ...) { ... }   ← shared fusion IR
        ↓                    ↓
   PtxBuilder            MslBuilder        ← backend-specific Layer 0
        ↓                    ↓
   PTX string            MSL string        ← one kernel, one function
        ↓                    ↓
   ptxas → SASS          metal → AIR → finalizer
```

## Reference Implementation: MFA in ccv

**Location:** `~/git/ccv/lib/nnc/mfa/kernels/`

### GEMMKernel (1011 lines)

MSL source built via `createSource()` composed of:
1. `createConstants()` — compile-time tile parameters
2. `createUtilities()` — helper functions (get_sram, multiply_accumulate)
3. `createInitializeC()` — zero accumulators
4. `createMultiplyIterations()` — the K-loop (two variants: async copy and direct load)
5. `createStoreC()` — write results to device memory

**The K-loop structure:**
```
for k = 0..K step K_group:
    // Async copy: device → threadgroup (one simdgroup does the copy)
    if (sidx == 0):
        simdgroup_event.async_copy(A_block, A_src, ...)
        simdgroup_event.async_copy(B_block, B_src, ...)
        simdgroup_event.wait(2, events)
    threadgroup_barrier()

    // Multiply-accumulate (all simdgroups)
    for k_inner = 0..K_group step 8:
        // Load A fragments: simdgroup_matrix_storage.load(A_block_src, ...)
        // Load B fragments: simdgroup_matrix_storage.load(B_block_src, ...)
        // MMA: C_sram.multiply(*A_sram, *B_sram)
    threadgroup_barrier()
```

**Key parameters from descriptor:**
- `blockDimensions` = (M_block, N_block, K_block) — e.g., (48, 48, 32) for f16
- `leadingBlockDimensions` — threadgroup memory padding for bank conflicts
- `preferAsyncLoad` — true on apple9 (M3+), false on older
- `transposeState` — per-operand layout
- `registerM/registerN` — per-simdgroup tile within the block
- `splits` — simdgroup partitioning

**Atom decomposition:**
- `multiply_accumulate()` (lines 525-555) = CopyAtom (load from smem) + MmaAtom (simdgroup multiply)
- The async copy section (lines 950-975) = CopyAtom (device → threadgroup)
- `createInitializeC()` = accumulator init
- `createStoreC()` = EpilogueAtom (identity store, or bias+store)
- **TransformAtom slot**: between `A->load()` and `C->multiply()` — currently identity

### AttentionKernel (3233 lines)

Forward-only for inference. Same `createSource()` pattern.
- Parallelization along R (row) dimension
- Online softmax (max tracking + exp + normalize)
- Q × K^T matmul → softmax → P × V matmul
- Block dimensions: (parallelization, traversal, head)
- GQA support via Hq/Hk ratio
- D-blocking for large head dimensions (register pressure management)

### Tile Sizes (from `getBlockDimensions()`)

**apple9 (M3+):** 32×32×8 with padding (32, 32, 32) for f16
**older:** 48×48×32 for f16, 48×48×24 for f32

These are small compared to CUDA (128×128). Apple GPUs have smaller SIMD width (32 vs CUDA's 32 warps × 32 lanes) and different register/threadgroup memory tradeoffs.

## Plan

### Phase 0: MslBuilder + Standalone GEMM

**Goal:** Emit MSL that matches MFA's GEMMKernel performance.

Create `ferrite-metal/`:
```
ferrite-metal/
├── Cargo.toml
├── src/
│   ├── lib.rs          — MslBuilder struct
│   ├── config.rs       — MetalGemmConfig (blockDimensions, async strategy, etc.)
│   ├── msl_builder.rs  — MSL string builder (CodeWriter equivalent)
│   ├── gemm.rs         — GEMM kernel emitter (port of GEMMKernel.cpp)
│   ├── atoms.rs        — Metal atom trait impls
│   └── bench.rs        — benchmark harness using metal-rs
```

**Step 1:** Port `CodeWriter` to Rust — template substitution engine for MSL generation
**Step 2:** Port `GEMMKernelDescriptor` — tile size selection, async strategy, precision config
**Step 3:** Port `GEMMKernel::createSource()` — full MSL emission
**Step 4:** Benchmark on M1/M2/M3 against MFA

**Success criterion:** Within 10% of MFA's GEMM throughput.

### Phase 1: Atom Decomposition

Refactor the monolithic GEMM into composable atoms:

```rust
// Metal CopyAtom: simdgroup_event async_copy (apple9) or direct load
struct MetalAsyncCopy { prefer_async: bool }
impl CopyAtom for MetalAsyncCopy { ... }

// Metal MmaAtom: simdgroup_multiply_accumulate
struct MetalSimdgroupMma { register_m: u16, register_n: u16 }
impl MmaAtom for MetalSimdgroupMma { ... }

// Transform slot: between load and multiply
struct IdentityTransform;
struct RmsNormTransform { /* norm factor regs */ }
impl TransformAtom for RmsNormTransform { ... }

// Epilogue: SiLU on accumulators
struct SiluEpilogue;
impl EpilogueAtom for SiluEpilogue { ... }
```

Verify: reassembled atoms = identical perf to monolithic GEMM.
Then: first fused kernel (norm→GEMM→SiLU) on Metal.

### Phase 2: Attention

Port MFA's `AttentionKernel` forward path into the atom framework.
This is Metal-specific — the traversal pattern, D-blocking, and online softmax
are different from CUDA FlashAttention.

### Phase 3: Full LLaMA Megakernel

One `kernel void llama(...)` dispatch. All layers. Embedding to logits.
Weight prefetching across layers via async copy.

### Phase 4: Cost Model

Replace MFA's per-head-dim parameter tables with programmable cost model.
Chip-generation aware (apple8 vs apple9 changes async strategy).

## Dependencies on CUDA Ferrite

- **Atom trait definitions** (`CopyAtom`, `TransformAtom`, `MmaAtom`, `EpilogueAtom`) — shared interface, Metal-specific implementations
- **Fusion strategy engine** (`strategy.rs`) — shared, hardware-parameterized
- **Proc macro** (`ferrite-macros/`) — shared parser + graph builder, backend-specific lowering

## Metal-Specific Challenges (vs CUDA)

1. **No ptxas equivalent** — Metal's driver finalizer is opaque. Performance debugging is benchmark-only.
2. **Smaller tiles** — 32×32 or 48×48 vs CUDA's 128×128. Less compute-to-memory amortization for transforms.
3. **simdgroup_matrix is 8×8** — finer granularity than CUDA's m16n8k16.
4. **threadgroup memory** — 32KB on M1, more on M3+. Constrains double buffering.
5. **No inline assembly** — must express everything in MSL C++. No escape hatch for special instructions.
6. **Unified memory** — device memory is also CPU-accessible. Different bandwidth characteristics than discrete GPU HBM.
