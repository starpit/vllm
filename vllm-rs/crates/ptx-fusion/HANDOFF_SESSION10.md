# Handoff — Session 10

**READ FIRST**: `crates/ptx-fusion/FERRITE.md` — the master plan for the entire
ferrite project. It defines the architecture (escape analysis on PTX), the target
forward pass (6 launches/layer), the roadmap (Phases 0-5G), and the proven
capabilities. Everything in this session implements Phase 5G.

## The goal

Implement register transfer for the persistent MLP kernel to eliminate the
16-barrier ptxas allocation that forces 1 block/SM occupancy. This is Phase 5G
of the ferrite roadmap: "Multi-GEMM pipeline + driver loop."

## What was done

### 1. Verified non-persistent eliminates barrier exhaustion

The persistent kernel's 16 barriers come from cp.async inside an unbounded
persistent loop — ptxas can't prove async groups don't overlap. A non-persistent
kernel with the same cp.async code gets **1 barrier**.

**Test**: `sequenced_mlp_barrier_count` in `pipeline_compile.rs`
**Result**: sequenced (non-persistent) = 1 barrier, persistent = 16 barriers.
This confirmed the entire approach.

### 2. Built GemmBodyDescriptor — CUTLASS GEMM decomposition via PTX analysis

`extract_gemm_body()` parses a perimeter-replaced CUTLASS GEMM and returns:
- Preamble (tile index computation, param loads, pipeline prologue)
- K-loop body (MMA + cp.async loads + pipeline sync)
- Epilogue (accumulator → SMEM rearrange → scale → bf16 → st.global)
- MMA accumulator registers (via `analyze_carries` with `CarryRole::MmaAccumulator`)
- Tile pointers, induction variables, buffer state registers

**Key discovery**: 64×128×32 has **128** MMA accumulators (not 64). 64×64×32 has 64.
This drove the decision to use 64×64×32 for both producer and consumer GEMMs.

**Tests**: `extract_gemm_body_64x128x32`, `extract_gemm_body_64x64x32`

### 3. Built accumulator spill/reload

`build_accum_spill_reload()` generates PTX to save/restore MMA accumulators to SMEM.
Each thread spills its accumulators to a unique SMEM slot: `base + tid * num_accum * 4`.

**Test**: `accum_spill_reload_64x64x32`

**NOTE**: This was later **abandoned** — see section 6 below. The spill approach
doesn't fit in the SMEM budget for 2-block occupancy. Both accumulator sets
(producer + consumer) live in registers simultaneously instead.

### 4. Built epilogue store analysis

`analyze_epilogue_stores()` traces all st.global instructions in the CUTLASS epilogue
backward through the def-use graph to decompose their addresses as:
`base_addr + sum_of(stride_regs)`.

Identified:
- Base address register (`%rd22`)
- Row register (`%r174`) — found via `cvt.s64.s32` in the address chain
- Column register (`%r824`) — found via `mul.wide.s32` in the address chain
- Stride registers (`%rd20` = 2-row stride, `%rd21` = 8-row stride)
- 16 stores with correct stride chain decomposition

This is fully general — works on any CUTLASS epilogue that computes addresses as
linear combinations of param-derived strides. No hardcoded register names.

**Test**: `analyze_epilogue_stores_64x128x32`

### 5. Built epilogue redirect to SMEM

`redirect_epilogue_to_smem()` rewrites the CUTLASS epilogue to write output to
SMEM scratch instead of GMEM. The MMA fragment rearrangement, alpha scaling, and
bf16 conversion are preserved unchanged — only the final st.global stores are
replaced with st.shared stores.

SMEM addresses computed from the row/col registers identified by the store analysis:
`smem_base + row * tile_n * 2 + col * 2 + stride_row_offset`.

Handles both `.v4.u32` (128-wide tiles) and `.v2.u32` (64-wide tiles) store formats
automatically from the analysis.

**Test**: `redirect_epilogue_to_smem_64x128x32`

### 6. SMEM budget discovery — no spill, keep both accum sets in registers

**L4 (sm_89): 100KB SMEM per SM. For 2 blocks/SM: 50KB per block max.**

Original plan: 24KB CUTLASS + 32KB down accum spill = 56KB > 50KB. Doesn't fit.

**Solution**: Don't spill. Keep both gate_up and down MMA accumulators in registers.
- gate_up: 64 f32 accum regs
- down: 64 f32 accum regs (idle during gate_up, accumulating during down K-iters)
- CUTLASS scratch: ~40 regs
- Total: ~168 regs/thread

Register file: 65536 regs/SM. At 2 blocks × 128 threads = 256 threads:
65536/256 = 256 regs/thread max. 168 < 256. **Fits.**

SMEM layout (no spill):
- CUTLASS mainloop: 24KB (offset 0)
- Gate scratch: 8KB (offset 24KB)
- Up scratch: 8KB (offset 32KB)
- **Total: 40KB < 50KB** ✓

### 7. Extended PointwiseComputation with ALoadSource

Added `ALoadSource` enum to `PointwiseComputation`:
- `ALoadSource::Gmem` — standard ld.global (default, backward compatible)
- `ALoadSource::Smem { smem_base_reg }` — ld.shared from SMEM scratch

Modified `replace_a_loads_with_inline_fn()` to dispatch between GMEM and SMEM loads
based on the source mode. Added `{LOAD_INDEX_X16}` placeholder for byte-offset computation.

All 131 existing tests pass with the new field (default = Gmem).

### 8. Built SiLU+mul from SMEM scratch

`build_silu_mul_computation_smem()` — same SiLU fast-exp approximation as the GMEM
version, but loads gate values from SMEM via `ALoadSource::Smem` and up values from
a second SMEM scratch region via `ld.shared` in the per_site code.

**Test**: `silu_from_smem_ptxas_valid` — 95 registers, 1 barrier, passes ptxas.

### 9. Built K-loop de-pipelining

`depipeline_k_loop()` removes cp.async software pipeline from a CUTLASS K-loop:
- Replaces A cp.async with ld.shared (from SMEM scratch) or ld.global
- Replaces B cp.async with explicit ld.global + st.shared
- Strips cp.async.commit_group and cp.async.wait_group

**Test**: `depipeline_k_loop_64x64x32` — 2 A-loads + 2 B-loads replaced.

**NOTE**: This was built but may not be needed. The final approach uses
`replace_a_loads_with_inline_fn` with `ALoadSource::Smem` instead, which
preserves the B-load cp.async pipeline (keeping the 1-barrier efficiency).
The depipelined approach was a fallback.

### 10. Built fuse_gemm_pointwise_gemm — the main assembly function

`fuse_gemm_pointwise_gemm()` takes:
- Producer GEMM (flat-param PTX) — e.g., gate_up
- Consumer GEMM (flat-param PTX) — e.g., down
- PointwiseComputation — e.g., SiLU+mul from SMEM
- ProducerOutput mode — Single or Paired (for gate+up)
- Kernel name

Produces a single non-persistent kernel with the driver loop:

```
entry fused_mlp(ferrite_params[88], ferrite_params_2[88],
                _num_producer_n_tiles, _intermediate_n_offset) {
    // Consumer preamble: tile index from grid (defines output tile)
    // Zero consumer accumulators
    // Driver loop:
    $L_driver_loop:
        // Gate GEMM: preamble + K-loop + redirected epilogue → SMEM scratch A
        // Up GEMM: preamble + K-loop + redirected epilogue → SMEM scratch B
        // Consumer K-loop: A-loads read SiLU'd data from scratch, B-loads from GMEM
        // Loop back
    // Consumer epilogue: accumulators → bf16 → st.global
}
```

Register namespace merging: consumer registers offset by producer's register count
(same approach as `sequence_gemm_phases`). Labels renamed: producer uses `$L__BB0`,
up copy uses `$L__BB_up`, consumer uses `$L__BB_cons`.

**Test**: `fuse_gemm_pointwise_gemm_ptxas` — **151 registers, 1 barrier, 0 spills**.

### 11. Built register_transfer_mlp! proc macro

Proc macro that calls `fuse_gemm_pointwise_gemm` at compile time. Takes the same
6 args as `persistent_mlp_block!`. Re-exported from `ptx-fusion` crate.

### 12. Wired into ferrite.rs

- `MLP_REGTRANSFER_PTX` constant: fused kernel compiled at build time
- `mlp_regtransfer: Option<CudaFunc>` field on `FerriteCutlass`
- Kernel loaded in constructor with 40KB SMEM

### 13. GPU test — first run

`register_transfer_mlp_gpu()` in `cuda_fuse_general.rs`. The kernel loads and
launches on GPU but produces **wrong results** (max_diff ~67-290) and crashes
with ILLEGAL_ADDRESS at M=128.

**Root cause**: The producer GEMM's preamble reads `%ctaid.x` and `%ctaid.y` to
compute tile coordinates. In the fused kernel, these grid coordinates belong to
the consumer (down) GEMM, not the producer. The producer's n_tile should come
from the driver loop counter, and its m_tile from the consumer's m_tile.

This is the same problem `make_persistent_two_phase` solves by replacing
`mov.u32 %rN, %ctaid.x` with computed tile coordinates.

## Current state

**What passes ptxas**:
- Fused kernel: 5396 lines, 151 registers, 1 barrier, 0 spills, 0 gmem
- All 131 unit tests pass
- `vllm-cuda` builds with the kernel loaded

**What fails on GPU**:
- Correctness: wrong results because producer preamble tile index is wrong
- ILLEGAL_ADDRESS at larger M values (same root cause — reads wrong memory)

## What remains — immediate next step

### Fix producer tile index override

**The bug**: The producer GEMM's preamble reads `%ctaid.x` and `%ctaid.y` (hardware
special registers) to compute its tile coordinates. In the fused kernel, the grid
is sized for the CONSUMER's tile decomposition, so `%ctaid.x`/`%ctaid.y` give
the consumer's tile, not the producer's. The producer produces garbage output.

**The kernel param layout** (as emitted by `fuse_gemm_pointwise_gemm`):
```
.entry fused_mlp(
    .param .align 8 .b8 ferrite_params[88],      // producer (gate_up) GEMM flat params
    .param .align 8 .b8 ferrite_params_2[88],     // consumer (down) GEMM flat params
    .param .u32 _num_producer_n_tiles,             // outer loop count = ceil(intermediate/64)
    .param .u32 _intermediate_n_offset             // N-tile offset for "up" half
)
```

**The driver loop registers**:
- `%r_driver_iter` — current outer loop iteration (0, 1, 2, ...)
- `%r_driver_total` — loaded from `_num_producer_n_tiles`
- `%r_driver_n_off` — loaded from `_intermediate_n_offset`
- `%p_driver_loop` — loop predicate
- `%r_gate_scratch` — constant 24576 (gate scratch SMEM offset)
- `%r_up_scratch` — constant 32768 (up scratch SMEM offset)

**The consumer preamble** runs first (with `$L__BB_cons` label prefix, registers
offset by producer's register count). It computes the output tile coordinates from
`%ctaid.x` and `%ctaid.y` using the CUTLASS swizzle. After it runs, certain
consumer registers hold m_tile and n_tile. These can be identified by running
`extract_tile_index_map()` on the consumer PTX.

`extract_tile_index_map()` (parser.rs) returns a `TileIndexMap` with:
- `ctaid_x_reg` — register holding `%ctaid.x` (e.g., `%r177` → after offset: `%r941`)
- `m_tile_reg` — register holding the M-tile index (e.g., `%r2` → `%r766`)
- `n_tile_reg` — register holding the N-tile index (e.g., `%r3` → `%r767`)
- `swizzle_log_reg` — register holding the swizzle parameter
- `tile_index_lines` — the PTX lines that compute the tile index

**The fix** — two approaches:

**Approach A (simpler)**: Replace `mov.u32 %rN, %ctaid.x` in the producer preamble
with `mov.u32 %rN, <computed_value>`. The CUTLASS preamble starts with
`mov.u32 %r<ctaid_x_reg>, %ctaid.x` and `mov.u32 %r<ctaid_y_reg>, %ctaid.y`.
For the producer:
- Compute the swizzled ctaid_x from (m_tile, driver_iter) using the same swizzle
  formula CUTLASS uses: `ctaid_x = m_tile * swizzle_tile + (n_tile & (swizzle_tile - 1))`
- Compute ctaid_y: `ctaid_y = n_tile >> swizzle_log`
- Replace the `mov.u32 %rN, %ctaid.x` with `mov.u32 %rN, %r_computed_ctaid_x`
- Same for ctaid_y.
- The m_tile comes from the consumer's tile index (same M rows).
- The n_tile comes from `%r_driver_iter` (gate) or `%r_driver_iter + %r_driver_n_off` (up).

**Approach B (cleaner)**: Skip the producer preamble entirely. Emit custom tile
index computation using `%r_driver_iter` and the consumer's m_tile, then jump
directly to the producer's K-loop. The preamble's main job is tile index + param
loads + pipeline prologue. The param loads happen via `_active_params` or
`ferrite_params` which is already correct. The pipeline prologue (initial cp.async
fills) is essential and must be preserved.

Approach A is simpler because it preserves the entire preamble structure. The
only change is two `mov` instructions. Use `extract_tile_index_map()` to find
which register receives `%ctaid.x` and `%ctaid.y`, then replace those lines.

**For the up GEMM copy**: Same fix but n_tile = `%r_driver_iter + %r_driver_n_off`.
The up copy already has labels renamed to `$L__BB_up`.

**Where to make the change**: In `fuse_gemm_pointwise_gemm()` in `pipeline_compile.rs`,
around lines 1132-1149 (gate GEMM emission) and 1152-1166 (up GEMM emission).
Currently these just emit the producer preamble verbatim. Need to scan for
`%ctaid.x` and `%ctaid.y` references and replace them.

### The gate/up scratch SMEM layout

The redirected epilogue writes bf16 output to SMEM scratch in **row-major linear layout**:
`scratch_base + row * tile_n * 2 + col * 2`, where row and col are the same
registers the CUTLASS epilogue computed for its GMEM output addresses.

The SiLU computation at the consumer's A-load sites reads from this scratch via
`ld.shared.v4.b32 [%r_gate_scratch + LOAD_INDEX * 16]`. Each A-load site reads
16 bytes (8 bf16 = 4 b32). The LOAD_INDEX is 0, 1, 2, ... for successive A-load
sites within the K-loop.

**Potential issue**: The A-load sites' sequential indexing (0, 1, 2...) may not
map to the correct positions in the linear scratch layout. The cp.async A-loads
read from specific (row, K-column) positions that may not be sequential. This
mapping needs verification — if wrong, the SiLU gets the wrong gate/up values
for each element, producing numerically wrong (but structurally valid) output.

If the sequential mapping is wrong, the fix is to compute the SMEM scratch address
from the same row/K-column expression that the cp.async GMEM source uses, but
with the SMEM stride instead of the GMEM stride. The `EpilogueStoreMap.row_reg`
and `EpilogueStoreMap.col_reg` provide the row/col registers; the scratch stride
is `tile_n * 2` (compile-time constant).

### After tile index fix

1. Run GPU correctness test — target 0.00e0 (or at least bf16 precision)
2. Sweep M=1,8,16,64,128,256,512 × production dims (0.5B, 3B)
3. Wire `launch_mlp_register_transfer()` method in ferrite.rs
4. Benchmark with `vllm bench latency`

### Longer term

1. **Generalize to 3-GEMM chain**: The full post-attention segment is
   O_proj → norm → gate_up → SiLU → down → residual (3 GEMMs). The current
   `fuse_gemm_pointwise_gemm` handles 2 GEMMs. Extending to accept a
   `Vec<Stage>` pipeline chain is the next architecture step.

2. **Norm prologue**: Fuse rms_norm into the gate_up GEMM prologue using the
   existing `fuse_reduction_into_gemm` infrastructure. This eliminates the
   norm → gate_up GMEM round-trip.

3. **CUDA graph support**: The non-persistent kernel should be graph-capturable
   (no atomic counters to zero). Verify and enable.

4. **Residual add**: Set beta=1.0 in the down GEMM's epilogue params to fuse
   the residual addition.

## Key files (updated)

```
crates/ptx-fusion-macros/src/
  pipeline_compile.rs    — GemmBodyDescriptor, epilogue analysis, epilogue redirect,
                           accum spill/reload, depipeline, fuse_gemm_pointwise_gemm,
                           build_silu_mul_computation_smem (+1885 lines)
  fuse_general.rs        — ALoadSource enum, SMEM load support in replace_a_loads (+54 lines)
  parser.rs              — parse_instruction made pub(crate), carry analysis test (+54 lines)
  lib.rs                 — register_transfer_mlp! proc macro (+97 lines)

crates/ptx-fusion/
  src/lib.rs             — re-export register_transfer_mlp
  tests/cuda_fuse_general.rs — register_transfer_mlp_gpu test (+194 lines)

crates/vllm-cuda/src/
  ferrite.rs             — MLP_REGTRANSFER_PTX constant, mlp_regtransfer field,
                           kernel loading in constructor (+23 lines)
```

## SMEM budget summary

```
L4 (sm_89): 100KB SMEM/SM, 50KB/block at 2 blocks/SM
Tile config: 64×64×32 for both producer and consumer

Layout:
  [0, 24KB)    CUTLASS mainloop (A+B tiles, 3-stage pipeline)
  [24KB, 32KB) Gate scratch (64×64 bf16 = 8KB)
  [32KB, 40KB) Up scratch (64×64 bf16 = 8KB)
  Total: 40KB < 50KB ✓

Register budget:
  Producer accumulators: 64 f32 (from 64×64×32 MMA)
  Consumer accumulators: 64 f32 (persist across outer loop)
  CUTLASS scratch: ~40 mixed
  Total: ~168 regs < 256 max (at 2 blocks/SM)
```

## Test counts

- **131 unit tests** (ptx-fusion-macros) — all pass
- **1 new GPU test** (register_transfer_mlp_gpu) — launches but wrong results (tile index bug)
- **All existing GPU tests** — not re-run this session (no changes to existing GPU test code)

## Running the tests

```bash
# Unit tests (all 131)
cargo test -p ptx-fusion-macros --lib

# Specific new tests
cargo test -p ptx-fusion-macros --lib fuse_gemm_pointwise_gemm_ptxas -- --nocapture
cargo test -p ptx-fusion-macros --lib sequenced_mlp_barrier_count -- --nocapture
cargo test -p ptx-fusion-macros --lib extract_gemm_body -- --nocapture
cargo test -p ptx-fusion-macros --lib analyze_epilogue_stores -- --nocapture
cargo test -p ptx-fusion-macros --lib silu_from_smem_ptxas_valid -- --nocapture

# GPU test (currently fails — tile index bug)
cargo test -p ptx-fusion --features cuda --test cuda_fuse_general register_transfer_mlp_gpu -- --nocapture --test-threads=1

# Build vllm-cuda with ferrite
cargo build -p vllm-cuda --features cuda,ferrite
```

## Rules (carried forward + new)

- **NEVER write PTX by hand** — all transforms via PTX analysis
- **NEVER build special-case macros** — extend existing infrastructure
- **NEVER dismiss divergence** — any non-zero diff must be investigated
- **NEVER test only at small M** — sweep M=1..512
- **NEVER assume what's slow** — profile with nsys first
- **NEVER hardcode CUTLASS assumptions** — extract from PTX using the parser
- **Build with** `cargo build -p vllm-cuda --features cuda,ferrite`
- **NEW: No one-off extractions** — every analysis must work on arbitrary CUTLASS PTX
- **NEW: SMEM budget is 50KB/block** at 2 blocks/SM on L4 (not 99KB)
- **NEW: Register pressure is SUM when both phases are live** — not MAX
  (FERRITE.md line 29 applies to sequential phases, not overlapping accumulators)
