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

Tested on L4 GPU (sm_89), CUDA 12.9. **65 tests** (53 CUDA GPU + 12 doc-ignored + 10 unit), all passing.

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
| Perimeter: 5 kernel types | cuda_cutlass_bf16 | hand-written, nvcc, row-GEMM, CUTLASS bf16, CUTLASS FP8 |
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
  src/fuse_cp_async.rs             CUTLASS cp.async A/B classification and deletion
  src/chain.rs                     Multi-phase chaining (append SMEM-handoff phases)
  src/persistent.rs                Persistent kernel wrapper (work-queue loop)
  src/regfuse.rs                   Register-level fusion engine (elementwise)
  src/extract.rs                   Single-entry extraction from multi-entry PTX

crates/ptx-fusion/                 Library + tests
  src/lib.rs                       KernelProtocol types + re-exports
  src/main.rs                      Demo: extract + rewrite + fuse + validate
  build.rs                         Compiles vllm-cuda csrc/ kernels to PTX at build time
  kernels/                         Hand-written, nvcc-compiled, and CUTLASS PTX files
  tests/                           52 CUDA GPU tests
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

## What's Next

### Immediate: CUTLASS prologue fusion (rms_norm -> CUTLASS bf16 GEMM)

**Where we are.** The GEMM appears 4 times per LLaMA layer. CUTLASS loads matrix
tiles via `cp.async.cg.shared.global` (async GMEM -> SMEM copy). The perimeter
model (`AsyncCopyPort`) now identifies these as input ports. The proc macro
auto-classifies which cp.async loads are for A vs B by tracing GMEM source
registers back to struct-param offsets -- no hardcoded assumptions.

A-matrix cp.async replacement is proven end-to-end:
- **Deletion**: 6 A-loads deleted, 6 B-loads + 16 MMA preserved, ptxas valid
- **Explicit replacement**: Each deleted cp.async replaced with
  `ld.global.v4.b32 + st.shared.v4.b32` (with mask predication for boundary tiles)
- **GPU correctness**: Explicit-load kernel vs original kernel: **0.00e0 diff**
  (bitwise identical output, verified via CUTLASS API params + driver API launch)

The SMEM destination address computation (`SharedStorageBase + f(tid)`) still
runs after cp.async replacement. The replacement reuses the same SMEM address
registers and GMEM source registers — only the transfer mechanism changes
(synchronous ld+st instead of async cp.async).

**Failed approaches (don't repeat these):**
- Separate SMEM handoff buffer alongside CUTLASS: fails because CUTLASS uses
  all available SMEM dynamically (extern shared). No room for a second buffer.
- SMEM-to-SMEM copy (ld.shared handoff -> st.shared CUTLASS tile): same problem,
  needs a separate buffer that doesn't fit.

**Correct approach:** The prologue writes directly into CUTLASS's SMEM tile
locations -- the same addresses the deleted cp.async would have written to.
No separate buffer needed.

**Remaining step:**

1. **Fuse with rms_norm.** Replace the explicit GMEM loads (which currently
   load A from GMEM just like cp.async did) with inline rms_norm computation.
   rms_norm outputs f32; the handoff converts f32->bf16 (`cvt.rn.bf16.f32`)
   before writing to CUTLASS's SMEM. The rms_norm phase runs with 128 threads
   (matching CUTLASS blockDim). Each thread must produce the correct bf16 values
   for its SMEM slot, determined by tracing the GMEM source register `[%rdN]`
   to figure out which A-matrix elements each slot corresponds to.

**Key files:**
- `fuse_cp_async.rs`: `delete_a_matrix_loads()`, `replace_a_loads_with_explicit()`
- `parser.rs`: `AsyncCopyPort` in `KernelProtocol`, struct-param tracing
- `kernels/cutlass_gemm_bf16_sm89.ptx`: bf16 CUTLASS GEMM (64x64x32, 3 stages, 24KB SMEM)
- `tests/support/cutlass_prologue_test.cu`: GPU correctness harness (CUTLASS API + driver API)

### Near-term

- **More epilogue ops**: quantize (f32->fp8), scale, residual add.
- **FlashAttention**: compile FA2/FA3 to PTX, identify escape perimeter.

### The full forward pass

Every op in a transformer layer compiles to PTX. Ferrite identifies each kernel's
escape perimeter and stitches them into one persistent kernel at `cargo build` time.

```
Current (9 kernel launches per layer):
  fused_add_rms_norm -> QKV GEMM -> RoPE -> FlashAttn -> O GEMM
  -> fused_add_rms_norm -> gate_up GEMM -> silu_mul -> down GEMM

Ferrite target (1 launch per layer):
  persistent_kernel {
    loop {
      tile = next_tile();
      phase_norm(tile);           // SMEM handoff ->
      phase_qkv_gemm(tile);      // epilogue injects RoPE ->
      phase_attention(tile);      // reads from SMEM ->
      phase_o_gemm(tile);        // epilogue injects residual+norm ->
      phase_gate_up_gemm(tile);  // epilogue injects SiLU ->
      phase_down_gemm(tile);     // epilogue injects residual
    }
  }
```

We have the source to every kernel (CUTLASS, FlashAttention, vllm-cuda csrc/).
We can compile all of them to PTX. Ferrite can identify escape perimeters and
modify them. The remaining work is the persistent kernel framework.

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
