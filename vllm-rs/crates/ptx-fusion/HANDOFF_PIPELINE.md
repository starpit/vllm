# Pipeline Architecture Handoff

## The Goal

Ferrite fuses CUDA kernels at compile time by analyzing their PTX. The original
approach fused kernels pairwise (A→B) with special-case code for each pattern.
This session replaced that with a **tiled pipeline model** that composes N stages
into a single kernel — designed from the start for the full MLP block:

```
rms_norm → GEMM_gate_up → SiLU → GEMM_down → residual_add
```

The pipeline model treats each kernel as a stage with a typed pattern (Pointwise,
Reduction, TiledGemm), decomposes reductions into streamable phases, and generates
a fused kernel where data flows through SMEM/registers without touching GMEM
between stages.

## What Was Built (9 commits)

### Phase 5A: Parser extensions (parser.rs)

Three new capabilities added to the PTX parser:

**`detect_loops(lines) → Vec<LoopDescriptor>`**
- Label-branch pairing in O(N): collects labels, finds back-edges, computes nesting
- Validated: rms_norm has 4 loops, CUTLASS GEMM has 1 K-loop

**`analyze_carries(lines, loop) → Vec<CarryRegister>`**
- Identifies registers that carry state across loop iterations
- Uses use-before-def ordering (not just same-instruction self-modification)
- Classifies: Accumulator (fma.f32), InductionVar (add.s32), BufferState (selp),
  TilePointer (add.s64), MmaAccumulator (mma.sync dest)
- Key fix: distinguishes true carries from loop-local variables by checking if a
  register's first use precedes its first def within the loop body. This correctly
  classifies silu_mul as Pointwise (its fma instructions are SiLU computation,
  not cross-iteration accumulation)

**`DefUseGraph::trace_backward(reg, depth) → Vec<(depth, node_idx)>`**
- Mirror of existing trace_forward: BFS through defs map
- Used by thread-to-row extraction to trace address chains

### Phase 5B: Pipeline stage descriptors (pipeline.rs)

**`PipelineStage::from_ptx(name, source) → Result<PipelineStage>`**

Combines parser + loop detection + carry analysis + pattern classification:

- `StagePattern::Pointwise` — no reduction, no MMA (silu_mul, scale)
- `StagePattern::Reduction { accumulators, reduce_method }` — loops with
  accumulators + shfl/SMEM reduce (rms_norm → WarpShuffleAndSmem)
- `StagePattern::TiledGemm { a_loads, b_loads, mma_accumulators, pipeline_depth }`
  — cp.async + MMA + K-loop (CUTLASS)

For multi-entry PTX (vllm kernels with f32/f16/bf16 variants), automatically
extracts the first entry before analysis. `source_lines` stores the extracted
entry so line numbers match between loops and source.

### Phase 5C: Reduction decomposition (pipeline.rs)

**`PipelineStage::decompose_reduction() → Option<ReductionDecomposition>`**

Splits a Reduction stage into three phases by classifying its loops:

- **Accumulate loops**: contain accumulator carries (e.g., sum-of-squares via fma)
- **Emit loops**: contain st.global (output stores)
- **Finalize range**: code between last accumulate back-edge and first emit header

For rms_norm:
- Accumulate: 2 loops ($L__BB0_2 vectorized, $L__BB0_5 scalar tail)
- Finalize: warp shuffle + SMEM reduce + rsqrt → inv_rms
- Emit: 2 loops ($L__BB0_16 vectorized, $L__BB0_19 scalar tail)

Detects the finalized value register (`%f12` = inv_rms, loaded from SMEM after
the reduction barrier). Extracts the emit loop body as raw PTX lines.

### Phase 5D: Pipeline compiler (pipeline_compile.rs)

**`fuse_reduction_into_gemm(reduction, gemm, name) → Result<String>`**

Produces a single fused PTX kernel from a Reduction + TiledGemm stage pair.
Builds a `PointwiseComputation` and feeds it into the existing
`replace_a_loads_with_inline_fn` infrastructure.

The `PointwiseComputation` has four parts:

**1. Prologue** (emitted once before first A-load):
- Row loop: for each of tile_m rows, all 128 threads cooperate on reduction
- K-loop: each thread loads bf16 from GMEM, converts to f32, accumulates sum_sq
  via `fma.rn.f32` (pattern extracted from rms_norm decomposition)
- Warp shuffle: 5 rounds of `shfl.sync.down.b32 + add.f32`
- SMEM reduce: lane 0 of each warp writes to `_ferrite_warp_scratch[warp_id]`,
  barrier, thread 0 sums 4 warp contributions
- Finalize: `div / add eps / rsqrt` → inv_rms stored in `_ferrite_inv_rms[row]`
- Per-thread setup: compute m_rel from tid.x using extracted thread map,
  load inv_rms for both rows (m_rel and m_rel + row_stride),
  compute GMEM row base addresses rb0 and rb1

**2. Per-site code** (emitted at each A-matrix cp.async replacement):
- Row parity: `setp.ge.u64 {GMEM_SRC}, rb1` → selects inv_rms0 or inv_rms1
- Weight loading: `weight_ptr + (gmem_src - row_base[parity])` → 16 bytes (8 bf16)
- Unpack 8 bf16 weights to 8 named f32 registers

**3. Per-element instructions** (applied to each f32 after bf16→f32 conversion):
```ptx
mul.f32  {INPUT}, {INPUT}, %f_rms_inv;      // × inv_rms
mul.f32  {INPUT}, {INPUT}, %f_rms_wt{ELEM_IDX};  // × weight
```

**4. Extra params** prepended to kernel entry:
- `_ferrite_rms_input` (u64), `_ferrite_rms_weight` (u64),
  `_ferrite_rms_epsilon` (f32), `_ferrite_rms_hidden` (u32),
  `_ferrite_rms_a_stride` (u64)

### Phase 5F: Thread-to-row extraction (pipeline_compile.rs)

**`extract_thread_row_map(lines) → Option<ThreadRowMap>`**

Extracts `lanes_per_row_log2`, `rows_per_warp_log2`, `row_stride` from the
GEMM PTX by tracing the first A-load's address chain backward:

1. Find first cp.async → extract gmem_src register
2. Trace backward through def-use graph (depth 10)
3. Find `mul.lo.s64 %rd, stride_reg, row_reg` — the stride multiply
4. Identify which operand is the stride (traces to ld.param) vs the row
5. Trace row_reg backward:
   - `shr.s32` with shift 1-4 (not 5) → `lanes_per_row_log2`
   - `shl.b32` with shift 3-5 → `rows_per_warp_log2`

Verified on 64x128x32: extracts (2, 4, 8) = lane/4, warp*16, stride 8.

### Proc macro: pipeline_fuse!

```rust
const FUSED: &str = ptx_fusion_macros::pipeline_fuse!(
    "kernels/vllm_rms_norm.ptx",
    "kernels/cutlass_bf16_64x128x32_sm89.ptx",
    "fused_norm_gemm"
);
```

Reads both PTX files, runs the full pipeline (analyze → extract stages →
decompose reduction → build computation → replace A-loads → perimeter replace),
emits fused PTX as a const `&str` at compile time.

Includes `dedup_reg_declarations()` to handle the case where both
`replace_a_loads_with_inline_fn` and `replace_perimeter` insert temp register
declarations after the `.reg .b64` line. Only deduplicates in the kernel's
declaration block (not inside inline asm `{ .reg .pred p; }` blocks).

## Key Bugs Found and Fixed

**SMEM clobbering** (prologue_isolation test caught this):
The warp reduction scratch and the inv_rms result array shared the same SMEM
(`_ferrite_inv_rms`). Warp lane 0 writes to `[warp_id * 4]` overlapped with
inv_rms values for rows 0-3. Fix: separate `_ferrite_warp_scratch[4]` SMEM.

**Register declaration dedup removing inline asm predicates**:
`dedup_reg_declarations` initially removed ALL duplicate `.reg` lines. But
CUTLASS epilogue uses inline asm blocks `{ .reg .pred p; setp p; @p st; }` that
repeat `.reg .pred p;` per block. Fix: only dedup in the kernel's declaration
block (before the first instruction), not inside inline asm.

## Methodology

Every decision follows from **escape perimeter analysis on PTX**:

1. **Never guess at CUTLASS internals.** The thread-to-row mapping, tile size,
   buffer rotation, and pipeline depth are all extracted from the PTX. The only
   runtime computation is one `setp.ge.u64` per cp.async site for row parity.

2. **The pipeline compiler is an orchestrator, not a new PTX emitter.** It builds
   a `PointwiseComputation` and feeds it into the existing, proven
   `replace_a_loads_with_inline_fn`. The cp.async replacement, bf16 unpack/repack,
   register allocation, and entry point modification are all handled by the
   existing infrastructure.

3. **Test-driven debugging.** The prologue_isolation test (standalone kernel that
   just runs the prologue and writes inv_rms to GMEM) was the key diagnostic
   that caught the SMEM clobbering bug. When the full fused kernel produced wrong
   results, isolating the prologue immediately pinpointed the issue.

4. **Reduction decomposition from PTX, not from mathematical knowledge.** The
   sum-of-squares formula, warp shuffle pattern, and rsqrt finalization are all
   detected from the actual rms_norm kernel PTX structure (loops, carries,
   shfl instructions). The pipeline compiler doesn't "know" what rms_norm is —
   it sees a Reduction pattern with accumulate/finalize/emit phases.

## What Remains

### tile_m hardcoded to 64
The prologue `mov.u32 %r_rms_nrows, 64` should be derived from the GEMM PTX.
Options:
- Count the number of distinct row addresses across all A-loads (tile_m = max_row + row_stride)
- Extract from the loop bound or grid dimension computation
- Parse from the CUTLASS template parameters in the mangled entry name
  (e.g., `GemmShapeILi64ELi128ELi32E` → tile_m=64)

The last option is simplest and most reliable. The entry name contains the
full CUTLASS configuration.

### Delete intrinsic_rms_norm.rs
Once `pipeline_fuse!` is wired into llama.rs (replacing `fuse_rms_norm_gemm_flat!`),
the 286-line hand-written intrinsic can be deleted. Both paths currently coexist.

### Wire into llama.rs (Phase 5E)
Replace the existing `fuse_rms_norm_gemm_flat!` calls in the llama.rs forward pass
with `pipeline_fuse!`. The param layout is slightly different (5 extra params instead
of 6 — the pipeline version doesn't need the `_ferrite_rms_n` swizzle param since
perimeter replacement handles swizzle inline).

### Multi-GEMM pipeline + driver loop (Phase 5G)
The big one. Extend the pipeline compiler for:
```
rms_norm → GEMM_gate_up → SiLU → GEMM_down → residual_add
```

This requires:
- GEMM→GEMM handoff: GEMM1 output tile stays in SMEM, GEMM2 reads it as A-input
- Generated driver loop: iterates over tiles, sequences stages
- Software pipelining: overlap GEMM1's next tile load with GEMM2's current compute

The existing infrastructure handles:
- SiLU → GEMM epilogue injection (proven, fuse_epilogue.rs)
- GEMM + residual → beta=1.0 (proven)
- rms_norm → GEMM prologue (just proven via pipeline compiler)

The new piece is the GEMM-to-GEMM tile handoff with the generated driver loop.

## Test Commands

```bash
# Unit tests (fast, no GPU) — 99 tests
cargo test -p ptx-fusion-macros --lib

# Pipeline-specific unit tests
cargo test -p ptx-fusion-macros --lib -- pipeline

# GPU correctness — 27/28 pass (1 pre-existing failure in compile_macro_metadata)
cargo test -p ptx-fusion --features cuda --test cuda_fuse_general -- --nocapture

# Pipeline GPU tests only
cargo test -p ptx-fusion --features cuda --test cuda_fuse_general -- pipeline_fused --nocapture

# Prologue isolation test
cargo test -p ptx-fusion --features cuda --test cuda_fuse_general -- prologue_isolation --nocapture

# Compute sanitizer
/usr/local/cuda-12.9/bin/compute-sanitizer --tool memcheck \
  cargo test -p ptx-fusion --features cuda --test cuda_fuse_general -- pipeline_fused_norm_gemm_gpu --nocapture
```

## Key Files

```
pipeline.rs          PipelineStage, StagePattern, ReductionDecomposition, from_ptx()
pipeline_compile.rs  fuse_reduction_into_gemm(), build_prologue, extract_thread_row_map()
parser.rs            detect_loops(), analyze_carries(), trace_backward() [new additions]
fuse_general.rs      replace_a_loads_with_inline_fn(), PointwiseComputation [existing, used as-is]
lib.rs               pipeline_fuse! proc macro, dedup_reg_declarations()
```
