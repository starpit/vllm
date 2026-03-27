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

Tested on L4 GPU (sm_89), CUDA 12.9. **71 tests** (59 CUDA GPU + 12 doc-ignored + 10 unit), all passing.

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
| Multi-tile CUTLASS dispatch | cuda_dispatch | 3 configs loaded, tile selection, grid dim computation |
| Stress tests | cuda_stress | n=1 to n=4096, 100-run determinism |

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
  src/parser.rs                    PTX parser + escape perimeter extraction
  src/fuse.rs                      SMEM stitching fusion engine (toy kernels)
  src/fuse_real.rs                 SMEM stitching for real nvcc PTX (vectorized, multi-pass)
  src/fuse_epilogue.rs             GEMM epilogue injection (parameterized: SiLU, GELU, ReLU)
  src/fuse_cp_async.rs             CUTLASS cp.async interception, rms_norm prologue fusion
  src/chain.rs                     Multi-phase chaining (append SMEM-handoff phases)
  src/persistent.rs                Persistent kernel wrapper (work-queue loop)
  src/regfuse.rs                   Register-level fusion engine (elementwise)
  src/extract.rs                   Single-entry extraction from multi-entry PTX

crates/ptx-fusion/                 Library + tests
  src/lib.rs                       KernelProtocol types + re-exports
  src/dispatch.rs                  Multi-tile CUTLASS runtime dispatcher
  src/main.rs                      Demo: extract + rewrite + fuse + validate
  build.rs                         Compiles vllm-cuda csrc/ kernels to PTX at build time
  kernels/                         Hand-written, nvcc-compiled, and CUTLASS PTX files
  tests/                           59 CUDA GPU tests + 4 dispatch tests
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

### In progress: CUTLASS GEMM parity with cuBLAS (Phase 0)

**Status**: 3 tile configs compiled, benchmarked, runtime dispatcher built.

**Done**:
- Compiled 64x64x32 (24KB), 128x128x32 (48KB), 128x128x64 (96KB) to PTX
- Benchmarked against cuBLAS at K=N=4096 (decode and prefill)
- Built `CutlassDispatch` Rust runtime: loads configs, selects by M
- Discovered L4 SMEM limit (99KB optin) blocks 128x256x64 with 3 stages

**Remaining**:
- Build Rust `GemmParams` constructor (currently C++ side only)
- Wire into llama.rs as feature-gated alternative to cuBLAS
- Profile-guided selection (optional: bench at model init, cache per M-bucket)

### Near-term: fused operation library (Phase 1)

**Already proven**: `fuse_rms_norm_cutlass!`, `inject_epilogue!` (SiLU/GELU/ReLU)

**Need to build**:
- `fuse_gemm_residual!` -- residual add in GEMM epilogue
- `fuse_norm_gemm_silu!` -- prologue + epilogue combined
- More epilogue ops: quantize (f32->fp8), scale

### The Ferrite-based forward pass

```
Current llama.rs (9+ kernel launches per layer):
  norm -> QKV GEMM -> split+RoPE -> FA2 -> O GEMM -> residual
  -> norm -> gate_up GEMM -> SiLU*mul -> down GEMM -> residual

Ferrite forward (5 launches per layer):
  fused_norm_qkv()              // norm→GEMM prologue (1 launch, was 2)
  split_qkv_rope() + FA2()     // stays separate (FA2 already optimal)
  fused_o_proj_residual()       // GEMM + residual epilogue (1 launch, was 2)
  fused_norm_gate_up_silu()     // norm→GEMM + SiLU epilogue (1 launch, was 3)
  fused_down_residual()         // GEMM + residual epilogue (1 launch, was 2)
```

Each `fuse_*!` macro generates: fused PTX (compile-time) + `launch()` function
(runtime). The forward reads like normal Rust with different function names.

### Roadmap

```
Phase 0: CUTLASS parity with cuBLAS     ← IN PROGRESS (benchmarked, dispatch built)
Phase 1: Fused operation library          ← mostly done, extend with residual/combined
Phase 2: Runtime integration (llama.rs)   ← wire into OwnedTensor/CachingAllocator
Phase 3: FlashAttention perimeter         ← exploratory
Phase 4: Persistent layer kernel          ← endgame
```

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
