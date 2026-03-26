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

## Status: All 5 Steps Complete + Real Kernel Fusion

All steps from the original roadmap are implemented and tested on an L4 GPU (sm_89)
with CUDA 12.9. **34 CUDA tests**, all passing.

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

## Crate Structure

```
crates/ptx-fusion-macros/          Proc macro crate (runs at compile time)
  src/lib.rs                       All proc macros: analyze, rewrite, fuse, regfuse, extract
  src/parser.rs                    PTX parser + protocol extraction + param tracing
  src/fuse.rs                      SMEM stitching fusion engine (toy kernels)
  src/fuse_real.rs                 SMEM stitching for real nvcc PTX (vectorized, multi-pass)
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
  tests/cuda_rewrite.rs            Register renaming correctness (2 tests)
  tests/cuda_fuse.rs               SMEM fusion correctness (1 test)
  tests/cuda_bench.rs              SMEM fusion benchmark (1 test)
  tests/cuda_regfuse.rs            Register fusion correctness + benchmark (2 tests)
  tests/cuda_stress.rs             Edge cases + determinism (13 tests)
  tests/cuda_nvcc.rs               nvcc PTX parsing + correctness (5 tests)
  tests/cuda_extract.rs            Multi-entry extraction + GPU run (4 tests)
  tests/cuda_vllm_kernels.rs       Real vllm kernel protocol + correctness (4 tests)
  tests/cuda_fuse_real.rs          Real kernel fusion correctness (1 test)
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

**Address rewriting via row_global_base subtraction.**
Rather than dissecting each kernel's address chain, compute `row_global_base` once (the
global address of the first element of this row), then for each store/load:
`smem_addr = smem_base + (global_addr - row_global_base)`. This works for any
"one block per row" kernel regardless of how nvcc compiled the address arithmetic.

## What's Next

- **Half/bf16 fusion**: the kernels have half and bf16 specializations in the same PTX.
  The extraction and fusion machinery works; just need to test with fp16 data paths.
- **Three-way chains**: fuse A -> B -> C in one pass (e.g., rms_norm -> linear_proj -> silu_mul).
  Requires chain-aware SMEM allocation.
- **Benchmark on real inference**: wire fused kernels into vllm-rs model execution,
  measure end-to-end latency reduction on actual token generation.
- **Cross-GEMM fusion**: the holy grail. Fuse normalization into GEMM's epilogue or
  activation into GEMM's prologue. Requires understanding CUTLASS/cuBLAS PTX structure.

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
