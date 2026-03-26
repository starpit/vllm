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

Tested on L4 GPU (sm_89), CUDA 12.9. **49 tests** (39 CUDA GPU + 10 unit), all passing.

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
| Stress tests | cuda_stress | n=1 to n=4096, 100-run determinism |

### Benchmarks

| Fusion type | Separate | Fused | Speedup |
|-------------|----------|-------|---------|
| SMEM (rms_norm -> matvec) | 9.94 us | 8.49 us | **1.17x** |
| Register (rms_norm -> scale) | 6.28 us | 2.90 us | **2.16x** |

### The GEMM Epilogue Result

Ferrite-injected GEMM+SiLU produces **bitwise identical** output to nvcc's hand-written
GEMM+SiLU (0.00e0 diff). GEMM+GELU matches nvcc within 4.77e-7. The `inject_epilogue!`
macro is parameterized: `inject_epilogue!("gemm.ptx", "entry", Gelu, NAME)` works for
SiLU, GELU, and ReLU. All three pass ptxas validation on CUTLASS bf16 GEMM (12 injection
sites each).

This means we can fuse arbitrary elementwise ops into any GEMM's output path
without writing custom CUDA code. The GEMM is a black box. We only touch the
escape perimeter.

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
  src/regfuse.rs                   Register-level fusion engine (elementwise)
  src/extract.rs                   Single-entry extraction from multi-entry PTX

crates/ptx-fusion/                 Library + tests
  src/lib.rs                       KernelProtocol types + re-exports
  src/main.rs                      Demo: extract + rewrite + fuse + validate
  build.rs                         Compiles vllm-cuda csrc/ kernels to PTX at build time
  kernels/                         Hand-written, nvcc-compiled, and CUTLASS PTX files
  tests/                           39 CUDA GPU tests
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

## What's Next

### Immediate

- **GEMM prologue injection**: fuse rms_norm output into GEMM input loads. Same escape
  analysis -- intercept `ld.global` on the A matrix and redirect from SMEM.
- **More epilogue ops**: add quantize (f32->fp8), scale, residual add to `ActivationFn`.
  The framework is parameterized -- just add emission functions.

### Near-term

- **Persistent tiled kernel**: one grid (108 blocks on L4), each block grabs tiles from
  a work queue and runs the full pipeline. The proc macro generates the phase sequence
  from each kernel's PTX. SMEM is reused between non-overlapping phases.
- **FlashAttention fusion**: compile FA2/FA3 to PTX, identify escape perimeter, fuse
  RoPE into prologue and output projection into epilogue.

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
