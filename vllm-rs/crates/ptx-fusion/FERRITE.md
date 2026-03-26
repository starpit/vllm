# Ferrite: Proc-Macro Kernel Fusion for CUDA

## The Idea

MLX uses lazy graph construction: operations are recorded, nothing executes until `eval()`.
At eval time, MLX sees the full graph and emits a single fused Metal shader. Intermediates
stay in registers/threadgroup memory instead of round-tripping through device memory.

CUDA can't do this naively because kernels are pre-compiled binaries — you can't fuse two
CUBINs after the fact. Systems like `torch.compile` and XLA solve this by tracing a Python
graph at runtime and JIT-compiling fused kernels. But they only fuse elementwise/reduction
ops. They can't fuse across MMA boundaries because GEMM is an opaque cuBLAS call.

The Megakernels project (~/git/Megakernels) takes a different approach: a single persistent
GPU kernel acts as a virtual machine, executing instructions fed from the host. Four warp
groups (controller, loader, consumer, storer) pipeline operations through shared memory pages.
This eliminates kernel launch overhead but intermediates still flow through SMEM between
every operation — not true register-level fusion.

**Ferrite's thesis**: use Rust proc macros to analyze and fuse CUDA kernels at compile time.
No runtime graph tracing, no JIT, no VM. The proc macro reads PTX (NVIDIA's text ISA),
extracts each kernel's "protocol" (registers, SMEM, params, I/O pattern), and stitches
compatible kernels into a single fused kernel where intermediates pass through SMEM or
registers instead of global memory.

## The Spectrum

```
Eager PyTorch        torch.compile         Megakernels VM        Ferrite proc macros
─────────────────────────────────────────────────────────────────────────────────────
N kernel launches    Fewer fused launches   1 launch, N ops       1 launch, fused ops
GMEM between all     GMEM between groups    SMEM between ops      Registers between ops
No optimization      Graph-level fusion     SM-level scheduling   Register-level fusion
Runtime overhead     JIT compile overhead   Zero launch tax       Zero launch tax
Python graph trace   Python graph trace     Python scheduler      Rust compile time
```

## What Exists Today (POC)

### Crate Structure

```
crates/ptx-fusion-macros/     Proc macro crate (runs at compile time)
  src/lib.rs                  analyze_kernel! and rewrite_kernel! macros
  src/parser.rs               PTX parser + protocol extraction

crates/ptx-fusion/            Library + test binary
  src/lib.rs                  KernelProtocol types (used by both macro and runtime)
  src/main.rs                 Demo: extract + rewrite + validate protocols
  tests/cuda_rewrite.rs       CUDA test: verify rewritten PTX produces same output
  kernels/rms_norm.ptx        Easy kernel: per-element normalize, no SMEM
  kernels/matvec.ptx          Hard kernel: matrix-vector with SMEM + barriers
```

### What the Macros Do

**`analyze_kernel!("kernels/rms_norm.ptx")`** — reads PTX at compile time, emits a const:

```rust
const RMS_NORM: KernelProtocol = KernelProtocol {
    name: "rms_norm",
    registers: &[(".f32", 16), (".pred", 4), (".u32", 8), (".u64", 8)],  // 36 total
    smem_regions: &[],
    total_smem_bytes: 0,
    params: &[
        KernelParam { name: "input",   ptx_type: ".u64", is_pointer: true,  index: 0 },
        KernelParam { name: "output",  ptx_type: ".u64", is_pointer: true,  index: 1 },
        KernelParam { name: "weight",  ptx_type: ".u64", is_pointer: true,  index: 2 },
        KernelParam { name: "n",       ptx_type: ".u32", is_pointer: false, index: 3 },
        KernelParam { name: "epsilon", ptx_type: ".f32", is_pointer: false, index: 4 },
    ],
    global_loads: &[
        DataPort { param_name: "input",  data_type: "f32", line: 38 },
        DataPort { param_name: "weight", data_type: "f32", line: 47 },
    ],
    global_stores: &[
        DataPort { param_name: "output", data_type: "f32", line: 51 },
    ],
    smem_loads: 0, smem_stores: 0,
    barriers: &[],
    has_mma: false,
};
```

**`rewrite_kernel!("kernels/rms_norm.ptx", { "%f3" => "%f30", "%r3" => "%r30" })`** —
renames registers in the PTX, updates `.reg` declarations to fit the new indices, and
re-extracts the protocol. The rewritten protocol must match the original (same I/O, same
SMEM, same barriers) — only register names change.

### What the Parser Extracts

The PTX parser (`parser.rs`) extracts from compiled PTX:

| What | How |
|------|-----|
| Register budget | `.reg .f32 %f<16>` directives |
| Shared memory | `.shared .align 16 .f32 smem_vec[4096]` directives |
| Kernel params | `.param` directives in the entry signature |
| Global loads | `ld.global.*` instructions, traced back to params via `ld.param` + `add.u64` chains |
| Global stores | `st.global.*` instructions, same tracing |
| SMEM access count | `ld.shared.*` and `st.shared.*` counts |
| Barriers | `bar.sync N` instructions |
| MMA usage | `wmma.*` or `mma.sync.*` instructions |

The param tracing is key: the parser follows `ld.param.u64 %rd0, [input]` then tracks
`add.u64 %rd4, %rd0, %rd3` to know that `ld.global.f32 %f1, [%rd4]` reads from the
`input` parameter. This gives us the actual I/O protocol, not just "reads from GMEM."

### Validation (runs on macOS, no CUDA needed)

```bash
cargo run -p ptx-fusion
```

Outputs the full protocol for both kernels and validates that rewritten protocols match
originals. Both pass.

## How to Pick Up on a CUDA Machine

### Step 1: Run the CUDA correctness test

```bash
cargo test -p ptx-fusion --features cuda -- --nocapture
```

This loads original and rewritten PTX via the CUDA driver (cudarc calls `cuModuleLoadData`),
runs both on identical inputs, and asserts bitwise identical output. If this passes, register
renaming is proven correct on real hardware.

The test file is `tests/cuda_rewrite.rs`. It tests:
- **rms_norm**: 256-element input, compares original vs rewritten output
- **matvec**: 64x128 matrix, compares original vs rewritten output

Both tests also sanity-check that the output isn't all zeros.

### Step 2: Test with real nvcc-compiled PTX

The sample PTX files are hand-written. The next validation step is to compile real CUDA
kernels and feed the output to the parser:

```bash
# Compile a real kernel to PTX
nvcc -ptx -arch=sm_80 some_real_kernel.cu -o real_kernel.ptx

# Add to the crate and analyze
analyze_kernel!("kernels/real_kernel.ptx");
```

nvcc-generated PTX will be more complex (predicated instructions, vectorized loads,
compiler-generated register names). The parser may need hardening for:
- Predicated instructions (`@%p0 ld.global.v4.f16 ...`)
- Vectorized loads/stores (`ld.global.v2.f32`, `ld.global.v4.f16`)
- Indirect addressing patterns
- Multiple `.entry` points in one PTX module

### Step 3: Prove the SMEM stitching step

This is the first real fusion milestone. Take two kernels where kernel A writes to GMEM
and kernel B reads from GMEM at the same location. Rewrite:
- Kernel A's `st.global` → `st.shared` (output goes to SMEM instead of GMEM)
- Kernel B's `ld.global` for that param → `ld.shared` (input comes from SMEM instead of GMEM)
- Insert `bar.sync` between them
- Wrap both in a single `__global__` entry point

Concretely, with our two sample kernels:

```
BEFORE (two launches):
  rms_norm: ld.global(input) → compute → st.global(output)   # writes to GMEM
  matvec:   ld.global(vec_in) → compute → st.global(vec_out)  # reads from GMEM

AFTER (one launch, SMEM handoff):
  fused: ld.global(input) → rms_norm_body → st.shared(smem_buf)
         bar.sync
         ld.shared(smem_buf) → matvec_body → st.global(vec_out)
```

The proc macro for this (`fuse_kernels!`) needs to:
1. Analyze both protocols
2. Match A's output port to B's input port (by param name or explicit annotation)
3. Allocate an SMEM region for the handoff
4. Rewrite A's `st.global` on the matched output → `st.shared` to the new SMEM region
5. Rewrite B's `ld.global` on the matched input → `ld.shared` from the new SMEM region
6. Merge `.reg` declarations (renaming B's registers to avoid collisions)
7. Merge `.shared` declarations
8. Insert `bar.sync` at the seam
9. Emit a single `.entry` wrapping both phases

### Step 4: Benchmark the fused kernel

Compare:
```python
# Baseline: two launches
rms_norm_kernel<<<grid, block>>>(input, tmp, weight, n, eps);
matvec_kernel<<<grid, block>>>(matrix, tmp, output, M, K);

# Fused: one launch
fused_rms_norm_matvec<<<grid, block>>>(input, weight, matrix, output, n, eps, M, K);
```

Expected wins:
- One fewer kernel launch (~5-10us)
- Eliminated GMEM write + read for the intermediate (`tmp`): saves `n * sizeof(f32)` bytes
  of GMEM bandwidth in each direction

### Step 5: Register-level fusion (the hard goal)

SMEM stitching eliminates GMEM round-trips but still costs SMEM bandwidth at each seam.
The ultimate goal is register-level handoff: kernel A's output stays in registers and
kernel B reads directly from those registers.

This requires:
- Knowing the exact register layout of A's output (which MMA fragment type, which thread
  owns which elements)
- Ensuring B's input layout matches (or inserting a register shuffle)
- Combined register pressure must fit within 255 registers (but it's MAX not SUM when
  phases are sequential — only additive for the handoff registers)

This is architecture-specific (sm_80 MMA fragments differ from sm_90) and is where the
proc macro becomes a real compiler. But the protocol extraction from Steps 1-4 provides
the foundation — you can't do register fusion without first knowing the register layout.

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
treats them as black boxes with analyzable interfaces — fuse what exists instead of
rewriting everything.

**Register pressure is MAX not SUM for sequential phases.**
When phase 1 finishes and phase 2 begins, phase 1's registers are dead (unless doing
register handoff). Two kernels each using 128 registers can be sequenced in a single
kernel that only needs 128 registers — not 256. This makes stitching far more feasible
than people assume.

## Comparison with Megakernels

| Aspect | Megakernels | Ferrite |
|--------|-------------|---------|
| Fusion granularity | SMEM pages between ops | SMEM or registers between ops |
| Scheduling | Python offline → instruction stream | Rust compile time → fused kernel |
| Adding new ops | Write new CUDA opcode handler | Provide PTX, proc macro analyzes it |
| Synchronization | Barrier spins on GMEM | `bar.sync` within single kernel |
| Launch overhead | Zero (one persistent kernel) | Zero (one fused kernel) |
| Generality | Model-specific instruction sets | Fuses arbitrary PTX kernels |
| Warp utilization | 4 warp groups, some idle during phases | All warps active on current phase |

Megakernels' key innovation is the persistent VM model with overlapped load/compute/store.
Ferrite's key innovation is treating kernel fusion as a compile-time transformation on PTX.
These are not mutually exclusive — a Ferrite-fused kernel could be one of the opcodes in
a Megakernels instruction stream.
