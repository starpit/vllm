# Ferrite: Proc-Macro Kernel Fusion for CUDA

## The Idea

MLX uses lazy graph construction: operations are recorded, nothing executes until `eval()`.
At eval time, MLX sees the full graph and emits a single fused Metal shader. Intermediates
stay in registers/threadgroup memory instead of round-tripping through device memory.

CUDA can't do this naively because kernels are pre-compiled binaries -- you can't fuse two
CUBINs after the fact. Systems like `torch.compile` and XLA solve this by tracing a Python
graph at runtime and JIT-compiling fused kernels. But they only fuse elementwise/reduction
ops. They can't fuse across MMA boundaries because GEMM is an opaque cuBLAS call.

The Megakernels project (~/git/Megakernels) takes a different approach: a single persistent
GPU kernel acts as a virtual machine, executing instructions fed from the host. Four warp
groups (controller, loader, consumer, storer) pipeline operations through shared memory pages.
This eliminates kernel launch overhead but intermediates still flow through SMEM between
every operation -- not true register-level fusion.

**Ferrite's thesis**: use Rust proc macros to analyze and fuse CUDA kernels at compile time.
No runtime graph tracing, no JIT, no VM. The proc macro reads PTX (NVIDIA's text ISA),
extracts each kernel's "protocol" (registers, SMEM, params, I/O pattern), and stitches
compatible kernels into a single fused kernel where intermediates pass through SMEM or
registers instead of global memory.

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

## Status: All 5 Steps Complete + Real Kernel Fusion + GEMM Epilogue Injection

All steps from the original roadmap are implemented and tested on an L4 GPU (sm_89)
with CUDA 12.9. **36 CUDA tests**, all passing. CUTLASS GEMM epilogue modification
proven via ptxas validation.

### Step 1: CUDA Correctness (DONE)

Register renaming on hand-written PTX produces bitwise identical output on GPU.

```bash
cargo test -p ptx-fusion --features cuda --test cuda_rewrite -- --nocapture
```

### Step 2: nvcc-Compiled PTX (DONE)

Parser hardened for real compiler output:
- `cvta.to.global.u64` (address space cast) in pointer tracing
- `add.s64` (nvcc uses signed adds for pointer arithmetic)
- `ld.global.nc` (non-coherent loads), vectorized `v4` loads
- `.b32`/`.b64` register types, `$L__BB` labels, `.pragma` directives
- C++ mangled entry names, multi-entry PTX files

```bash
cargo test -p ptx-fusion --features cuda --test cuda_nvcc -- --nocapture
```

### Step 3: SMEM Stitching (DONE)

`fuse_kernels!` macro: fuses two kernels via SMEM handoff.

```
BEFORE (two launches):
  rms_norm: ld.global(input) -> compute -> st.global(output)
  matvec:   ld.global(vec_in) -> compute -> st.global(vec_out)

AFTER (one launch, SMEM handoff):
  fused: ld.global(input) -> rms_norm_body -> st.shared(smem_buf)
         bar.sync
         ld.shared(smem_buf) -> matvec_body -> st.global(vec_out)
```

```bash
cargo test -p ptx-fusion --features cuda --test cuda_fuse -- --nocapture
```

### Step 4: Benchmarks (DONE)

| Fusion type | Separate | Fused | Speedup |
|-------------|----------|-------|---------|
| SMEM (rms_norm -> matvec) | 9.94 us | 8.49 us | **1.17x** |
| Register (rms_norm -> scale) | 6.28 us | 2.90 us | **2.16x** |

SMEM fusion eliminates one kernel launch + GMEM round-trip.
Register fusion eliminates the launch + all intermediate memory traffic.

```bash
cargo test -p ptx-fusion --features cuda --test cuda_bench -- --nocapture
cargo test -p ptx-fusion --features cuda --test cuda_regfuse -- regfused_benchmark --nocapture
```

### Step 5: Register-Level Fusion (DONE)

`regfuse_kernels!` macro: fuses elementwise kernels with zero overhead.
The intermediate value stays in a register -- no SMEM, no GMEM, no barrier.

```
BEFORE: st.global.f32 [addr], %f7;   ld.global.f32 %f1, [addr];
AFTER:  mov.f32 %f17, %f7;           (no memory traffic at all)
```

Only works when thread i's output feeds thread i's input (elementwise chains).

```bash
cargo test -p ptx-fusion --features cuda --test cuda_regfuse -- --nocapture
```

### Real Kernel Fusion (DONE)

**The milestone**: fuse actual vllm-rs production kernels compiled from `csrc/`.

`fuse_real_kernels!` macro handles the full complexity of nvcc output:
- Vectorized `v4` loads/stores (`ld.global.nc.v4.f32`, `st.global.v4.f32`)
- Multi-pass kernels (rms_norm's 2-pass block-reduce + apply)
- Warp shuffles (`shfl.sync.down.b32`)
- SMEM block reductions (`block_reduce_sum`)
- Multiple store/load sites (vectorized loop + scalar tail)
- Multi-entry PTX extraction (`extract_entry!`)

**Fused: `rms_norm_kernel<float>` + `act_and_mul_kernel<silu, float>`**

```
Separate: [0.16198641, 0.17073932, 0.17935595, 0.18782166]...
Fused:    [0.16198641, 0.17073932, 0.17935595, 0.18782166]...
PASS: max_diff=0.00e0
```

Address rewriting approach: compute `row_global_base = global_base + row_offset * 4`
once, then for each st/ld.global: `smem_addr = smem_base + (global_addr - row_base)`.
Works for any "one block per row" kernel without dissecting the address chain.

```bash
cargo test -p ptx-fusion --features cuda --test cuda_fuse_real -- --nocapture
```

### GEMM Epilogue Fusion (DONE)

**The critical step toward full forward pass fusion**: inject an elementwise operation
directly into a CUTLASS GEMM's epilogue, at the PTX level.

`inject_silu_epilogue!` finds every `cvt.rn.bf16x2.f32` in the GEMM epilogue (where
f32 accumulator values are converted to bf16 output) and injects SiLU computation on
each f32 value before conversion. The activation runs entirely in registers -- zero
extra memory traffic, zero extra kernel launches.

Tested on a real CUTLASS 2.x FP8 E4M3 GEMM (sm_89):
- 3600-line kernel with MMA instructions, tiled epilogue, inline asm
- 12 SiLU injection sites (24 f32 accumulator values)
- Modified PTX passes ptxas validation

This proves Ferrite can modify the output path of production GEMMs. The same
approach works for GELU, quantization, scaling, or any elementwise epilogue op.

```bash
cargo test -p ptx-fusion --features cuda --test cuda_gemm_epilogue -- --nocapture
```

## Crate Structure

```
crates/ptx-fusion-macros/          Proc macro crate (runs at compile time)
  src/lib.rs                       All proc macros
  src/parser.rs                    PTX parser + protocol extraction + param tracing
  src/fuse.rs                      SMEM stitching fusion engine (toy kernels)
  src/fuse_real.rs                 SMEM stitching for real nvcc PTX (vectorized, multi-pass)
  src/fuse_epilogue.rs             GEMM epilogue injection (SiLU into CUTLASS)
  src/regfuse.rs                   Register-level fusion engine (elementwise)
  src/extract.rs                   Single-entry extraction from multi-entry PTX

crates/ptx-fusion/                 Library + tests
  src/lib.rs                       KernelProtocol types + re-exports
  src/main.rs                      Demo: extract + rewrite + fuse + validate
  build.rs                         Compiles vllm-cuda csrc/ kernels to PTX at build time
  kernels/rms_norm.ptx             Hand-written: elementwise normalize
  kernels/matvec.ptx               Hand-written: matrix-vector with SMEM
  kernels/scale.ptx                Hand-written: elementwise multiply
  kernels/rms_norm_real.ptx        nvcc-compiled: simple rms_norm
  kernels/matvec_real.ptx          nvcc-compiled: matvec with dynamic SMEM
  kernels/vllm_rms_norm.ptx        nvcc-compiled: real vllm-rs rms_norm (all specializations)
  kernels/vllm_silu_mul.ptx        nvcc-compiled: real vllm-rs silu_mul (all specializations)
  kernels/cutlass_gemm_sm89.ptx    nvcc-compiled: CUTLASS FP8 GEMM (40 tile configs)
  tests/cuda_rewrite.rs            Register renaming correctness (2 tests)
  tests/cuda_fuse.rs               SMEM fusion correctness (1 test)
  tests/cuda_bench.rs              SMEM fusion benchmark (1 test)
  tests/cuda_regfuse.rs            Register fusion correctness + benchmark (2 tests)
  tests/cuda_stress.rs             Edge cases + determinism (13 tests)
  tests/cuda_nvcc.rs               nvcc PTX parsing + correctness (5 tests)
  tests/cuda_extract.rs            Multi-entry extraction + GPU run (4 tests)
  tests/cuda_vllm_kernels.rs       Real vllm kernel protocol + correctness (4 tests)
  tests/cuda_fuse_real.rs          Real kernel fusion correctness (1 test)
  tests/cuda_gemm_epilogue.rs      CUTLASS GEMM epilogue SiLU injection (2 tests)
```

## Proc Macros

| Macro | Purpose |
|-------|---------|
| `analyze_kernel!("path.ptx")` | Extract protocol as const at compile time |
| `analyze_kernel_as!("path.ptx", NAME)` | Same, but specify the const name (for mangled entries) |
| `rewrite_kernel!("path.ptx", { "%f3" => "%f30" })` | Rename registers, emit rewritten PTX + protocol |
| `extract_entry!("path.ptx", "substr", NAME)` | Extract one entry from multi-entry PTX |
| `fuse_kernels!("a.ptx", "b.ptx", "name", "a_out", "b_in")` | SMEM fusion (toy PTX) |
| `regfuse_kernels!("a.ptx", "b.ptx", "name", "a_out", "b_in")` | Register fusion (elementwise) |
| `fuse_real_kernels!(...)` | SMEM fusion for real nvcc PTX (vectorized, multi-pass) |
| `inject_silu_epilogue!("path.ptx", "entry", NAME)` | Inject SiLU into CUTLASS GEMM epilogue |

## Key Design Decisions

**Why PTX, not CUDA source?**
PTX is a well-specified text ISA with a formal grammar. It's what the CUDA driver actually
compiles (to SASS). Parsing C++ with templates, macros, and `__device__` functions is
orders of magnitude harder. PTX tells you ground truth: actual register counts, actual
memory operations, actual instruction sequences.

**Why proc macros, not runtime JIT?**
The model architecture is known at compile time. There's no reason to pay JIT overhead at
runtime. Proc macros run once at `cargo build`, the fused kernel is baked into the binary.
Compile-time errors are better than runtime errors.

**Why not rewrite kernels from scratch (original Ferrite approach)?**
Writing correct, high-performance GEMM/attention/norm kernels from scratch is years of
work. The existing kernels in vLLM, CUTLASS, FlashAttention already work. This approach
treats them as black boxes with analyzable interfaces -- fuse what exists instead of
rewriting everything.

**Register pressure is MAX not SUM for sequential phases.**
When phase 1 finishes and phase 2 begins, phase 1's registers are dead (unless doing
register handoff). Two kernels each using 128 registers can be sequenced in a single
kernel that only needs 128 registers -- not 256. This makes stitching far more feasible
than people assume.

**GEMM epilogue injection via bf16 conversion interception.**
CUTLASS GEMMs compute in f32 accumulators, then convert to bf16 via `cvt.rn.bf16x2.f32`
before writing to GMEM. By injecting elementwise ops (SiLU, GELU, quantize) on the f32
values just before this conversion, we fuse post-GEMM ops into the GEMM itself. The
activation runs in registers on already-computed values -- zero extra memory traffic,
zero extra launches. This is the key to fusing across GEMM boundaries.

**Address rewriting via row_global_base subtraction.**
Rather than dissecting each kernel's address chain, compute `row_global_base` once (the
global address of the first element of this row), then for each store/load:
`smem_addr = smem_base + (global_addr - row_global_base)`. This works for any
"one block per row" kernel regardless of how nvcc compiled the address arithmetic.

## What's Next

The path to a single-kernel forward pass:

### Immediate (machinery proven, needs integration)

- **GEMM epilogue correctness test**: run the SiLU-injected CUTLASS GEMM on actual
  data and verify output matches GEMM + separate SiLU. The PTX passes ptxas; next
  step is GPU execution with real FP8 inputs.
- **Generalize epilogue injection**: parameterize by activation function (GELU, quantize,
  scale, residual add) instead of hardcoding SiLU. The injection framework is generic;
  just need to emit different instruction sequences.
- **GEMM prologue injection**: fuse rms_norm output into GEMM input loads. Same approach
  as epilogue but targeting `ld.global` sites that read the A matrix.

### Near-term (architecture work)

- **Persistent tiled kernel**: one grid (108 blocks on L4), each block grabs tiles from
  a work queue and runs the full pipeline: norm -> GEMM -> attn -> GEMM -> norm -> GEMM
  -> act -> GEMM. Intermediates stay in SMEM between phases. The proc macro generates
  the phase sequence from each kernel's PTX.
- **Multi-phase SMEM manager**: different phases need different SMEM layouts (GEMM uses
  SMEM for A/B tiles, norm uses SMEM for reduction). Allocate per-phase, reuse across
  non-overlapping phases.
- **FlashAttention fusion**: compile FA2/FA3 to PTX, extract entry, fuse RoPE into
  the attention prologue and output projection into the epilogue.

### The full forward pass vision

Every op in a transformer layer is compiled to PTX. Ferrite stitches them into one
persistent kernel at `cargo build` time. One launch per layer (or per model), all
intermediates in SMEM/registers, zero GMEM round-trips between ops.

```
Current (9 kernel launches per layer):
  fused_add_rms_norm -> QKV GEMM -> RoPE -> FlashAttn -> O GEMM
  -> fused_add_rms_norm -> gate_up GEMM -> silu_mul -> down GEMM

Ferrite target (1 launch per layer):
  persistent_kernel {
    loop {
      tile = next_tile();
      phase_norm(tile);           // SMEM handoff
      phase_qkv_gemm(tile);      // epilogue injects RoPE
      phase_attention(tile);      // reads from SMEM
      phase_o_gemm(tile);        // epilogue injects residual+norm
      phase_gate_up_gemm(tile);  // epilogue injects SiLU
      phase_down_gemm(tile);     // epilogue injects residual
    }
  }
```

We have the source to every kernel (CUTLASS, FlashAttention, vllm-cuda csrc/).
We can compile all of them to PTX. Ferrite can now modify GEMM epilogues.
The remaining work is the persistent kernel framework and multi-phase SMEM management.

## Comparison with Megakernels

| Aspect | Megakernels | Ferrite |
|--------|-------------|---------|
| Fusion granularity | SMEM pages between ops | SMEM or registers between ops |
| Scheduling | Python offline -> instruction stream | Rust compile time -> fused kernel |
| Adding new ops | Write new CUDA opcode handler | Provide PTX, proc macro analyzes it |
| Synchronization | Barrier spins on GMEM | `bar.sync` within single kernel |
| Launch overhead | Zero (one persistent kernel) | Zero (one fused kernel) |
| Generality | Model-specific instruction sets | Fuses arbitrary PTX kernels |
| Warp utilization | 4 warp groups, some idle during phases | All warps active on current phase |

Megakernels' key innovation is the persistent VM model with overlapped load/compute/store.
Ferrite's key innovation is treating kernel fusion as a compile-time transformation on PTX.
These are not mutually exclusive -- a Ferrite-fused kernel could be one of the opcodes in
a Megakernels instruction stream.
