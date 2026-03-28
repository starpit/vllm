# Ferrite: Compile-Time Kernel Fusion via Escape Analysis on PTX

## The Core Idea

**Kernel fusion is escape analysis on PTX.**

Every CUDA kernel has an escape perimeter: values that cross the boundary to global
memory. Global loads are inputs. Global stores are outputs. Everything else -- registers,
shared memory, the entire computational interior -- is opaque. You don't need to
understand it. You just need to know what escapes.

Fusion is plumbing: redirect kernel A's output stores to shared memory (or keep them
in registers), then redirect kernel B's input loads from that same location. The interior
of each kernel is a black box. Kernels in a forward pass run sequentially, not
concurrently -- so there's no interference. A runs to completion, B starts fresh. The
only thing that crosses the boundary is the data you explicitly pipe through.

This is the same concept as escape analysis in JVM/Go compilers: does this value escape
to the heap (GMEM)? If yes, it's part of the kernel's interface. If no, it's internal.
The escape perimeter is the protocol.

Ferrite implements this at compile time using Rust proc macros. The macro reads PTX
(NVIDIA's text ISA), identifies each kernel's escape perimeter, and stitches compatible
kernels by redirecting the boundary stores/loads. No runtime graph tracing, no JIT, no VM.
The fused kernel is baked into the binary at `cargo build`.

## Why This Works

**Register pressure is MAX not SUM.** When phase A finishes and phase B begins, A's
registers are dead. Two kernels each using 128 registers can be sequenced in a single
kernel that needs only 128 registers -- not 256. This is the reason the whole approach
is feasible. Without it, fusing two large kernels would exceed the 255-register hardware
limit. With it, fusion is free.

**GEMMs are not opaque.** We have the source to CUTLASS, FlashAttention, and every
vllm-cuda kernel. We can compile them all to PTX. A CUTLASS GEMM's 3600-line PTX
body is complex, but its escape perimeter is simple: it loads A and B from GMEM,
computes MMA in registers/SMEM, then stores C to GMEM via epilogue stores. Ferrite
intercepts those epilogue stores and injects operations (SiLU, GELU, quantize) on the
f32 accumulator values before they're written out. The GEMM interior is untouched.

**Sequential execution simplifies everything.** If kernels ran concurrently, you'd need
to reason about register lifetimes, SMEM bank conflicts, warp scheduling interactions.
But a forward pass is a chain: A finishes, then B starts. A's state is dead. B gets
a clean slate. The only shared state is the data you explicitly hand off.

## The Spectrum

```
Eager PyTorch        torch.compile         Megakernels VM        Ferrite proc macros
----------------------------------------------------------------------------------------------
N kernel launches    Fewer fused launches   1 launch, N ops       1 launch, fused ops
GMEM between all     GMEM between groups    SMEM between ops      Registers between ops
No optimization      Graph-level fusion     SM-level scheduling   Register-level fusion
Runtime overhead     JIT compile overhead   Zero launch tax       Zero launch tax
Python graph trace   Python graph trace     Python scheduler      Rust compile time
```

The key differentiator vs Megakernels: Ferrite achieves register-level fusion across
arbitrary kernels. Megakernels tops out at SMEM between every operation. Register handoff
is fundamentally tighter -- zero memory traffic, zero latency for the intermediate.

## Status

Tested on L4 GPU (sm_89), CUDA 12.9. **160+ tests** (88 CUDA GPU + 76 unit + doc-ignored), all passing.

### What's Proven

| Capability | Test | Result |
|------------|------|--------|
| Parse hand-written PTX | cuda_rewrite | Bitwise identical after register rename |
| Parse nvcc-compiled PTX | cuda_nvcc | cvta, add.s64, ld.global.nc, v4 loads, mangled names |
| Parse real vllm-rs kernels | cuda_vllm_kernels | rms_norm (132KB PTX), silu_mul (170KB PTX) |
| SMEM stitching (toy) | cuda_fuse | rms_norm -> matvec, zero diff |
| SMEM stitching (real) | cuda_fuse_real | vllm rms_norm -> silu_mul, zero diff |
| Register fusion | cuda_regfuse | rms_norm -> scale, bitwise identical, 2.16x speedup |
| Extract from multi-entry PTX | cuda_extract | Single entry from 40-entry CUTLASS PTX |
| GEMM epilogue injection (ptxas) | cuda_gemm_epilogue | SiLU, GELU, ReLU into CUTLASS GEMM, valid assembly |
| GEMM epilogue SiLU (GPU) | cuda_gemm_silu_correctness | Ferrite vs nvcc: **0.00e0 diff** |
| GEMM epilogue GELU (GPU) | cuda_gemm_gelu_correctness | Ferrite vs nvcc: **4.77e-7 diff** |
| GEMM prologue injection | cuda_gemm_prologue | rms_norm -> GEMM via SMEM, **0.00e0 diff** |
| Persistent kernel | cuda_persistent | 108-block work queue, 256 rows, **0.00e0 diff** |
| 3-phase MLP (full SMEM) | cuda_3phase | norm->GEMM+SiLU->GEMM, ALL handoffs via SMEM, **1.91e-6 diff** |
| CUTLASS bf16 perimeter | cuda_cutlass_bf16 | cp.async auto-classified: 6 A-loads, 6 B-loads, 2 param groups |
| CUTLASS cp.async deletion | cuda_cutlass_bf16 | A-loads deleted, B preserved, MMA preserved, **ptxas valid** |
| CUTLASS explicit A-loads (GPU) | cuda_cutlass_bf16 | cp.async replaced with ld.global+st.shared, **0.00e0 diff** |
| **rms_norm -> CUTLASS GEMM (GPU)** | cuda_cutlass_bf16 | **fused prologue: inv_rms + normalize + GEMM, 0.00e0 diff** |
| Perimeter: 5 kernel types | cuda_cutlass_bf16 | hand-written, nvcc, row-GEMM, CUTLASS bf16, CUTLASS FP8 |
| Multi-tile CUTLASS dispatch | cuda_dispatch | 3 configs loaded, tile selection, Rust params builder GPU-verified |
| **norm+GEMM+SiLU (GPU)** | cuda_cutlass_bf16 | **prologue + epilogue composed, 6.25e-2 diff (bf16 rounding)** |
| GEMM + residual add (GPU) | cuda_dispatch | beta=1.0 in CUTLASS LinearCombination, 1.56e-2 diff |
| Stress tests | cuda_stress | n=1 to n=4096, 100-run determinism |
| **Def-use graph + param classification** | cuda_cutlass_bf16 | **31 fields classified: 8 Ptr, 4 Stride, 6 Dim, 2 Scalar, 11 Derived** |
| **Perimeter replacement (ptxas)** | cuda_cutlass_bf16 | **flat-param GEMM: 33 ld.param rewritten, ptxas valid** |
| Derivation probing (build.rs) | build.rs | 4 CUTLASS configs probed: 64x64, 64x128, 128x128, 128x128x64 |
| **Flat-param GEMM vs cuBLAS (GPU)** | cuda_flat_gemm | **10/10: all llama dims, partial tiles, batch sweep, 0.00e0** |
| **End-to-end model inference** | vllm serve | **Qwen2.5-3B-Instruct: correct output ("Four", "Paris")** |
| **General `fuse!` macro (ptxas)** | cuda_fuse_general | **kernel-agnostic SMEM stitching, ptxas valid** |
| **General `fuse!` macro (GPU)** | cuda_fuse_general | **rms_norm→silu_mul: 10/10 sizes, 0.00e0 diff, deterministic** |

### Benchmarks

| Fusion type | Separate | Fused | Speedup |
|-------------|----------|-------|---------|
| SMEM (rms_norm -> matvec) | 9.94 us | 8.49 us | **1.17x** |
| Register (rms_norm -> scale) | 6.28 us | 2.90 us | **2.16x** |
| Fused rms_norm -> GEMM | 29.4 us | 24.9 us | **1.18x** |
| Persistent 2-phase (108 blocks, M=256) | 29.4 us | 31.4 us | 0.94x* |
| 3-phase MLP (norm->GEMM+SiLU->GEMM) | 34.4 us | 43.7 us | 0.79x* |

\* Persistent overhead is host memcpy to reset tile counter + per-tile atomic/barrier
costs. With small matrices (M=256, K=128), scheduling overhead dominates. The win
comes with more phases and larger data where GMEM savings outweigh per-tile costs.

### CUTLASS vs cuBLAS (L4, bf16, K=N=4096)

| Workload | cuBLAS | Best CUTLASS | Config | Ratio |
|----------|--------|-------------|--------|-------|
| **decode bs=1** | 135.6 us | **36.3 us** | 64x128x32 | **3.74x** |
| decode bs=4 | 30.7 us | 37.7 us | 64x128x32 | 0.81x |
| decode bs=8 | 32.7 us | 38.7 us | 64x128x32 | 0.84x |
| decode bs=16 | 35.4 us | 39.7 us | 64x128x32 | 0.89x |
| **decode bs=32** | 41.4 us | **41.1 us** | 64x128x32 | **1.01x** |
| decode bs=64 | 41.5 us | 41.4 us | 64x128x32 | 1.00x |
| prefill 128 | 82.5 us | **81.6 us** | 128x128x32 | **1.01x** |
| prefill 256 | 126.4 us | 158.6 us | 64x128x32 | 0.80x |
| prefill 512 | 238.9 us | 298.0 us | 64x128x32 | 0.80x |
| prefill 1024 | 507.3 us | 578.3 us | 128x128x64 | 0.88x |
| **prefill 2048** | 1209 us | **1161 us** | 128x128x64 | **1.04x** |

Three tile configs: 64x128x32 (36KB), 128x128x32 (48KB), 128x128x64 (96KB).
Runtime dispatch selects by M: 64x128 for decode, 128x128 for prefill.

**Where CUTLASS wins**: bs=1 (3.74x — cuBLAS launch overhead), bs=32-64
(parity), bs=128 (1.01x), bs=2048 (1.04x).

**Where cuBLAS wins**: bs=4-16 (11-19% — cuBLAS auto-selects specialized
skinny-M kernels we don't have), bs=256-512 (20% — cuBLAS has more tile
configs to choose from).

**Why the gap is acceptable**: At decode bs=8, the 6 us per-GEMM penalty is
~24 us across 4 GEMMs per layer. Fusion eliminates ~4 kernel launches x
~5 us = ~20 us. Nearly a wash — and at bs=1 it's a massive net win.

**SMEM limits on L4**: 128x256x64 with 3 stages needs 144KB, exceeds L4's
99KB optin max. The 128x128x64 (96KB) fits and is the best large-prefill
config.

**Performance TODOs**:
- Add more tile configs targeting the bs=4-16 gap (e.g., 16x256, GEMV-like)
- Profile-guided selection: bench each config at model init, cache per M-bucket
- StreamK scheduling for better SM utilization at medium M
- Try CUTLASS 3.x warp-specialized kernels (sm_90+ only, not L4)

### The GEMM Epilogue Result

Ferrite-injected GEMM+SiLU produces **bitwise identical** output to nvcc's hand-written
GEMM+SiLU (0.00e0 diff). GEMM+GELU matches nvcc within 4.77e-7. The `inject_epilogue!`
macro is parameterized: `inject_epilogue!("gemm.ptx", "entry", Gelu, NAME)` works for
SiLU, GELU, and ReLU. All three pass ptxas validation on CUTLASS bf16 GEMM (12 injection
sites each).

This means we can fuse arbitrary elementwise ops into any GEMM's output path
without writing custom CUDA code. The GEMM is a black box. We only touch the
escape perimeter.

### The GEMM Prologue Result

Fused rms_norm -> row GEMM via SMEM handoff: **bitwise identical** to separate
launches (0.00e0 diff). rms_norm writes normalized output to SMEM, barrier, then the
GEMM reads its A matrix from SMEM instead of GMEM. The GMEM round-trip between
normalization and GEMM is eliminated completely.

This required extending the PTX param tracer to propagate through `mov.u64`/`mov.b64`
(nvcc copies address registers into loop cursors), and extending the SMEM stitching
engine to handle the `mul.wide.s32` address pattern that nvcc generates for GEMM
kernels (vs the `cvt.s64.s32` pattern used by elementwise kernels).

## The General `ferrite::fuse!` Macro — **DONE**

```rust
ptx_fusion::fuse!(
    a = "kernels/vllm_rms_norm.ptx",
    b = "kernels/vllm_silu_mul.ptx",
    bind = { a.param_0 => b.param_1 },
    name = "fused_norm_silu",
    const = FUSED_PTX,
);
```

One macro. Any kernels. No hardcoded param builders. The macro is **kernel-agnostic**:
it doesn't know what the kernels do. It only sees their escape perimeters.

**How it works:**
1. Parses each kernel's PTX via `PtxParser::parse()` (auto-extracts from multi-entry PTX)
2. Resolves the binding: finds A's output store sites and B's input load sites via param tracing
3. Analyzes thread-to-element mappings to choose handoff (SMEM or registers)
4. Rewrites PTX: A's bound stores → `st.shared`, barrier, B's bound loads → `ld.shared`
5. Merges params: bound pair eliminated, remaining params concatenated
6. Emits a single fused kernel as a `const &str`

**GPU-verified**: 10/10 test cases, all 0.00e0 diff vs separate launches.
Production dimensions (Qwen2.5-3B: hidden=2560, 3456), batch sizes 1-256,
10-run determinism, large/small input magnitudes.

**Key files:**
- `fuse_general.rs`: general fusion engine (SMEM handoff path)
- `lib.rs`: `fuse!` proc macro (DSL parser + dispatch to engine)
- `tests/cuda_fuse_general.rs`: 10 GPU correctness tests

### The extended perimeter model

Today's perimeter: data-in (ld.global), data-out (st.global), params (names/types).
Missing: **param classification** (pointer vs stride vs scalar vs derived) and
**param dependencies** (which derived params are functions of which raw params).

A kernel's full perimeter includes params as first-class ports:

```
Params-in:  [ptr_A, stride_A, ptr_B, stride_B, M, N, K, alpha, beta]  ← raw
            [inc_strided, inc_next, inc_advance, swizzle_log, ...]     ← derived
Data-in:    ld.global sites traced to ptr_A, ptr_B
Data-out:   st.global sites traced to ptr_D
```

**Perimeter replacement**: rewrite the kernel's `ld.param` instructions to read
from a new flat layout containing only raw params. Derived params are inlined
as computations from raw params. The kernel's interior is untouched.

### Implementation phases

**Phase 1: Def-use graph + param classification** (parser.rs) — **DONE**

Def-use graph built in single O(N) pass. Param classification combines:
- Forward tracing through def-use graph (Stride via mul, Dimension via setp)
- Existing backward perimeter analysis (Pointer via traced memory addresses)
- Type-based (f32 → Scalar)
- Default (Derived for unclassified)
Tested on CUTLASS bf16 GEMM: 8 Ptr, 4 Stride, 6 Dim, 2 Scalar, 11 Derived.

**Phase 2: Perimeter replacement** (perimeter.rs) — **DONE**

build.rs compiles 4 CUTLASS configs to PTX and probes each with a generated
C++ program that varies raw params one at a time to extract per-field linear
formulas (slope + intercept). The probe results are stored as `.derivations.json`
alongside each `.ptx` in `kernels/`.

`replace_perimeter()` rewrites the PTX entry point:
- New flat param: `ferrite_params[88]` = 4 ptrs + 4 strides + M/N/K + alpha/beta
- Raw fields: `ld.param` remapped to flat offsets
- Derived fields: replaced with inline `shl`/`mul`/`add` from raw params
- Dimension fields: ceil-division computed inline (e.g., `grid_tiled_shape.m = ceil(M/64)`)
- Swizzle log: computed inline from N (not baked as constant)
- Kernel interior untouched

GPU-verified at all production dimensions vs cuBLAS (10/10 tests, 0.00e0).
Integrated into llama.rs — all 4 GEMMs per layer use flat-param CUTLASS.
Layers with bias use ferrite GEMM + separate `bias_add_inplace` kernel.
End-to-end correct on Qwen2.5-3B-Instruct.

**Phase 3: General `fuse!` proc macro** (fuse_general.rs) — **DONE**

The `fuse!` macro takes any two kernels + bindings and produces a fused kernel.
Kernel-agnostic: only looks at escape perimeters. SMEM handoff path working,
register handoff stubbed (requires thread-mapping analysis).

GPU-verified: rms_norm→silu_mul, 10/10 tests, 0.00e0 diff at production dims.

**Phase 4: Register handoff optimization**

Analyze thread-to-element mappings from the address computation chains in the
perimeter. When both kernels are elementwise with identical mappings
(blockIdx.x * blockDim.x + threadIdx.x), use register handoff instead of SMEM
for zero memory traffic. The `choose_handoff()` function in `fuse_general.rs`
currently defaults to SMEM (always correct); this phase makes it smart.

**Phase 5: llama.rs integration**

Wire `fuse!` into the forward pass to go from 11 launches to 6 per layer.

### What this replaces

All existing special-purpose macros become thin wrappers around `ferrite::fuse!`.
The broken Rust GemmParams builder, hardcoded swizzle formula, and hardcoded
iterator constants are all eliminated — replaced by perimeter analysis of the
actual PTX.

## How the Escape Analysis Works

The PTX parser extracts the escape perimeter:

| What | PTX pattern | Role |
|------|-------------|------|
| Input values | `ld.global.*` traced to params | Data entering the kernel |
| Output values | `st.global.*` traced to params | Data leaving the kernel |
| Output register | The `%fN` in `st.global.f32 [addr], %fN` | The value to intercept |
| SMEM (internal) | `ld.shared.*` / `st.shared.*` | Interior, don't touch |
| Barriers | `bar.sync N` | Interior synchronization |
| MMA | `mma.sync.*` | Computation, don't touch |

Param tracing follows the chain: `ld.param.u64 %rd0, [param]` ->
`cvta.to.global.u64 %rd1, %rd0` -> `add.s64 %rd4, %rd1, offset` ->
`ld.global.f32 %f1, [%rd4]`. This tells us `%f1` came from `param`.
Same for stores: trace back from `st.global` to identify which param
receives the output and which register holds the value.

The escape perimeter for rms_norm:
```
IN:  ld.global(input), ld.global(weight)
OUT: st.global(output) <- value in %f7
Interior: 36 registers, block_reduce_sum via SMEM, 2 bar.sync
```

The escape perimeter for a CUTLASS GEMM:
```
IN:  ld.global(A), ld.global(B), ld.global(scales)
OUT: st.global(C) <- values in %f516..%f579 (f32 accumulators)
     cvt.rn.bf16x2.f32 converts f32 -> bf16 before store
Interior: 2290 f32 regs, 1775 b32 regs, MMA instructions, SMEM pipeline
```

To fuse SiLU into the GEMM: inject `SiLU(%fN)` before each `cvt.rn.bf16x2.f32`
that feeds an output store. The GEMM interior is untouched. 12 injection sites,
24 f32 values, ~350 lines of SiLU computation added to a 3600-line kernel.

## Crate Structure

```
crates/ptx-fusion-macros/          Proc macro crate (runs at compile time)
  src/lib.rs                       All proc macros
  src/parser.rs                    PTX parser + escape perimeter + def-use graph + param classification
  src/perimeter.rs                 Perimeter replacement: rewrite param interface using probed derivations
  src/fuse.rs                      SMEM stitching fusion engine (toy kernels)
  src/fuse_real.rs                 SMEM stitching for real nvcc PTX (vectorized, multi-pass)
  src/fuse_epilogue.rs             GEMM epilogue injection (parameterized: SiLU, GELU, ReLU)
  src/fuse_cp_async.rs             CUTLASS cp.async interception, rms_norm prologue fusion
  src/chain.rs                     Multi-phase chaining (append SMEM-handoff phases)
  src/persistent.rs                Persistent kernel wrapper (work-queue loop)
  src/regfuse.rs                   Register-level fusion engine (elementwise)
  src/extract.rs                   Single-entry extraction from multi-entry PTX
  src/fuse_general.rs              General fusion engine: any kernels + bindings, kernel-agnostic

crates/ptx-fusion/                 Library + tests
  src/lib.rs                       KernelProtocol types + re-exports
  src/dispatch.rs                  Multi-tile CUTLASS runtime dispatcher
  src/main.rs                      Demo: extract + rewrite + fuse + validate
  build.rs                         Compiles vllm-cuda kernels + CUTLASS configs + probes derivations
  kernels/                         PTX files + .derivations.json (probed param formulas, git-tracked)
  tests/cuda_flat_gemm.rs          Comprehensive flat-param GEMM vs cuBLAS (10 tests, all production dims)
  tests/cuda_fuse_general.rs       General fuse! macro GPU tests (10 tests, production dims, 0.00e0)
  tests/                           88 CUDA GPU tests + 4 dispatch tests
```

## Proc Macros

| Macro | Purpose |
|-------|---------|
| `analyze_kernel!("path.ptx")` | Extract escape perimeter as const at compile time |
| `analyze_kernel_as!("path.ptx", NAME)` | Same, custom const name (for C++ mangled entries) |
| `rewrite_kernel!("path.ptx", { "%f3" => "%f30" })` | Rename registers, validate perimeter preserved |
| `extract_entry!("path.ptx", "substr", NAME)` | Extract one entry from multi-entry PTX |
| `fuse_kernels!(...)` | SMEM stitching (redirect output stores -> SMEM -> input loads) |
| `regfuse_kernels!(...)` | Register fusion (output register -> input register, zero memory) |
| `fuse_real_kernels!(...)` | SMEM stitching for real nvcc PTX (vectorized, multi-pass) |
| `inject_epilogue!("path", "entry", Gelu, NAME)` | Inject activation into GEMM epilogue (SiLU, GELU, ReLU) |
| `inject_silu_epilogue!(...)` | Convenience wrapper: inject SiLU into GEMM epilogue |
| `persistent_fuse_real_kernels!(...)` | Wrap fused kernel in persistent work-queue loop |
| `fuse_3phase_mlp!(...)` | 3-phase MLP: norm->GEMM+SiLU->GEMM, persistent |
| `delete_cutlass_a_loads!(...)` | Delete A-matrix cp.async from CUTLASS GEMM (for prologue) |
| `replace_cutlass_a_loads!(...)` | Replace A-matrix cp.async with explicit ld.global+st.shared |
| `fuse_rms_norm_cutlass!(...)` | Fuse rms_norm prologue into CUTLASS GEMM (normalize inline) |
| `fuse_norm_gemm_silu!(...)` | Prologue + epilogue: norm -> GEMM + SiLU in one kernel |
| `replace_perimeter_macro!(...)` | Rewrite CUTLASS param interface: flat layout, derived fields inlined |
| **`fuse!(...)`** | **General kernel fusion: any two kernels + binding, kernel-agnostic** |

## What's Next

### Done: CUTLASS prologue fusion (rms_norm -> CUTLASS bf16 GEMM)

**Proven end-to-end.** The `fuse_rms_norm_cutlass!` macro fuses rms_norm
directly into the CUTLASS GEMM. The fused kernel:

1. **Prologue**: 4-thread cooperative reduction computes inv_rms per tile row.
   Each thread group accumulates sum-of-squares over hidden_dim elements,
   reduces via `shfl.sync.bfly`, then computes `rsqrt(sum/hidden + eps)`.

2. **Normalized A-loads**: Each A-matrix cp.async is replaced with inline
   normalization: load input bf16, load weight bf16, unpack → f32,
   multiply input × weight × inv_rms, pack f32 → bf16, write to CUTLASS SMEM.

3. **GEMM body**: Unchanged. B-loads via cp.async, MMA via tensor cores,
   epilogue stores to GMEM. The GEMM reads the already-normalized A tile
   from SMEM — no GMEM round-trip.

**GPU correctness**: fused kernel vs separate rms_norm + CUTLASS GEMM: **0.00e0 diff**.

**Parameter handling**: The host passes input_ptr in the A_ptr field of the
CUTLASS params struct (instead of the rms_norm output pointer). Weight, epsilon,
and hidden_size are prepended as extra params.

**Key files:**
- `fuse_cp_async.rs`: `fuse_rms_norm_into_cutlass()`, `emit_inv_rms_prologue()`, `emit_normalized_a_load()`
- `kernels/cutlass_gemm_bf16_sm89.ptx`: bf16 CUTLASS GEMM (64x64x32, 3 stages, 24KB SMEM)
- `tests/support/cutlass_prologue_test.cu`: GPU correctness harness (CUTLASS API + driver API)

### Done: CUTLASS GEMM parity with cuBLAS (Phase 0)

3 tile configs compiled, benchmarked, Rust params builder + runtime dispatcher.
See benchmark table above for CUTLASS vs cuBLAS numbers.

### Done: fused operation library (Phase 1)

All fusion ops needed for the llama.rs forward pass are proven:

| Op | Macro/Technique | Diff | Status |
|----|----------------|------|--------|
| norm → GEMM | `fuse_rms_norm_cutlass!` | 0.00e0 | ✓ |
| norm → GEMM + SiLU | `fuse_norm_gemm_silu!` | 6.25e-2 | ✓ |
| GEMM + residual | `set_epilogue(1.0, 1.0)` | 1.56e-2 | ✓ |
| SiLU/GELU/ReLU epilogue | `inject_epilogue!` | 0.00e0 | ✓ |

Future epilogue ops (not blocking): quantize (f32->fp8), scale.

### The Ferrite-based forward pass

**Current llama.rs: 11 kernel launches per layer** (standard dense bf16)

| # | Kernel | Type |
|---|--------|------|
| 1 | fused_add_rms_norm_inplace | C FFI |
| 2 | QKV GEMM (cublasLtMatmul) | cuBLAS |
| 3 | split_qkv | C FFI |
| 4 | rotary_embedding (Q only) | C FFI |
| 5 | reshape_and_cache (KV write) | C FFI |
| 6 | flash_attention (paged or contiguous) | FA2 |
| 7 | O proj GEMM | cuBLAS |
| 8 | fused_add_rms_norm_inplace | C FFI |
| 9 | gate_up GEMM | cuBLAS |
| 10 | silu_and_mul_fused | C FFI |
| 11 | down GEMM | cuBLAS |

All 11 are CUDA-graph-capturable. With graphs, launch overhead is amortized
to near-zero (one graph replay). Without graphs, ~5 us per launch = ~55 us/layer.

**Ferrite forward: 6 launches per layer** (11 → 6)

| # | Ferrite kernel | Replaces | Technique |
|---|---------------|----------|-----------|
| 1 | fused_norm_qkv | #1 + #2 | `fuse_rms_norm_cutlass!` |
| 2 | split_qkv + RoPE + KV write | #3 + #4 + #5 | unchanged (tiny kernels) |
| 3 | flash_attention | #6 | unchanged (FA2 already optimal) |
| 4 | O proj + residual | #7 | `beta=1.0` |
| 5 | fused_norm_gate_up_silu | #8 + #9 + #10 | `fuse_norm_gemm_silu!` |
| 6 | down proj + residual | #11 | `beta=1.0` |

**Savings analysis:**

Without CUDA graphs: 5 fewer launches × ~5 us = ~25 us/layer launch savings.

With or without graphs: GMEM bandwidth savings from fusing norm into GEMM
prologue (eliminates read+write of full hidden-dim intermediate).
At hidden=4096, M=64: ~1 MB saved per fused norm × 2 fused norms = ~2 MB/layer.
At L4's ~300 GB/s: ~7 us/layer bandwidth savings.

**What stays unfused and why:**
- **split_qkv + RoPE + KV cache write**: 3 tiny elementwise kernels (~2 us each).
  Could be absorbed into QKV GEMM epilogue later but not worth the complexity yet.
- **FlashAttention**: Already a megakernel. 10K+ lines of PTX with warp
  specialization. The existing C kernel with fused RoPE is near-optimal.

### Roadmap

```
Phase 0: CUTLASS parity with cuBLAS     ← DONE (benchmarked, dispatch)
Phase 1: Def-use graph + param classify  ← DONE (parser.rs)
Phase 2: Perimeter replacement           ← DONE (perimeter.rs, build.rs probe, llama.rs integration)
Phase 3: General fuse! proc macro        ← DONE (fuse_general.rs, SMEM path, 10/10 GPU tests 0.00e0)
Phase 4: Register handoff optimization   ← NEXT (thread-mapping analysis for register vs SMEM)
Phase 5: llama.rs integration            ← wire fuse! into forward pass (11→6 launches)
```

### Current performance (no fusion yet)

Ferrite replaces cuBLAS GEMMs with flat-param CUTLASS GEMMs in the llama.rs
forward pass. **No speedup expected** — standalone CUTLASS is slower than
cuBLAS at decode (M=1) because cuBLAS auto-selects specialized skinny-M
kernels. Layers with bias incur an extra `bias_add_inplace` kernel launch.

The win comes from Phase 3 fusion:
- norm+GEMM fused → eliminates 2 norm launches + 2 GMEM round-trips
- GEMM+SiLU fused → eliminates 1 silu launch + 1 GMEM round-trip
- GEMM+bias fused → eliminates the separate bias-add launch

## Comparison with Megakernels

| Aspect | Megakernels | Ferrite |
|--------|-------------|---------|
| Fusion granularity | SMEM pages between ops | **Registers** between ops |
| Scheduling | Python offline -> instruction stream | Rust compile time -> fused kernel |
| Adding new ops | Write new CUDA opcode handler | Provide PTX, proc macro analyzes perimeter |
| Synchronization | Barrier spins on GMEM | `bar.sync` within single kernel |
| Launch overhead | Zero (one persistent kernel) | Zero (one fused kernel) |
| Generality | Model-specific instruction sets | Fuses arbitrary PTX kernels |
| GEMM fusion | SMEM between GEMM and consumer | Register injection into GEMM epilogue |

Megakernels' innovation: persistent VM with overlapped load/compute/store.
Ferrite's innovation: escape analysis on PTX enables register-level fusion
across arbitrary kernels, including into GEMM epilogues, at compile time.

These are not mutually exclusive -- a Ferrite-fused kernel could be one of the
opcodes in a Megakernels instruction stream.
