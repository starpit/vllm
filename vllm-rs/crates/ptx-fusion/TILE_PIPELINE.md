# Ferrite Tile-Level Pipeline Architecture

## The Goal

One kernel launch per fusible segment of the transformer forward pass. Every
intermediate value stays in registers, SMEM, or streams through GMEM with
per-tile barriers. No unnecessary kernel launches, no unnecessary GMEM
round-trips.

The ultimate target per layer: **3 launches** (down from 11):

```
Launch 1: norm + QKV GEMM                           (fused)
Launch 2: FlashAttention                             (black box)
Launch 3: O_proj + norm + gate_up + SiLU + down      (persistent, 3 GEMMs + norm + SiLU)
```

## What We Proved in Session 7

- **test14j**: The fused prologue (128 threads) computes inv_rms that differs from
  the standard C kernel (896 threads) by 5.45e-6 at layer 9. This is FP non-
  associativity from different accumulation order, not a bug. Accepted.
- **test15b**: M=1..128 all 0.00e0. M=256 produces garbage (diff=15.9). The bug is
  in the prologue's tile index computation: m_tiles 0-2 correct, m_tile 3+ broken.
- **Production wiring**: llama.rs with fused norm+GEMM produces correct chat output
  (decode, small M). Throughput bench crashes at M>128 (the same test15b bug).
  The wiring was reverted after testing.
- **Megakernels study** (`~/Megakernels`): Their decode path uses matvec with explicit
  warp specialization (16 consumer warps, loader/storer/controller warps). Norm result
  stays in registers, fed directly to matvec. No GMEM round-trip. Key insight: they
  avoid the norm-split problem by not using tiled GEMM for decode. For prefill with
  tiled GEMM, the split (reduction prologue + per-element at A-loads) is unavoidable.

## What NOT To Do

These are hard-won lessons from 6+ sessions of debugging.

1. **NEVER reimplement the GEMM's tile index computation.** The current prologue
   hand-writes a swizzle calculation that disagrees with the GEMM body at m_tile≥3.
   EXTRACT the tile index FROM the GEMM's PTX and REUSE it. One computation, shared.

2. **NEVER transplant PTX between incompatible execution contexts.** The rms_norm
   kernel runs with 896 threads; the GEMM prologue runs with 128. Transplanting
   the reduction code produces different FP accumulation order → different inv_rms.
   For a self-consistent pipeline, GENERATE the accumulation code for the target
   thread count, or accept the FP difference (it's mathematically valid).

3. **NEVER test only at small M.** The M=256 bug went undetected because all tests
   used M=8. Every test must sweep M=1,2,4,8,16,32,64,65,128,256,512.

4. **NEVER chase bit-exactness with the old C kernels.** The fused pipeline is its
   own implementation. Different accumulation order → different FP results. Both are
   correct. The only test that matters: does `vllm serve` produce coherent text?

5. **NEVER write PTX by hand for compute logic.** The per-element normalize
   (`mul inv_rms, mul weight`) is the same whether hand-written or transplanted —
   we proved this with test14j. But the PLUMBING (tile index, row bases, K-offsets)
   is where bugs hide. Extract plumbing from the GEMM's own PTX.

## The Forward Pass (One Layer)

```
Op 1: fused_add_rms_norm    [M,H] → [M,H]           Reduction along H
Op 2: QKV GEMM              [M,H] × [N_qkv,H]^T     TiledGemm, K=H
      ─── attention (FlashAttention, black box) ───
Op 3: O_proj GEMM           [M,Q] × [H,Q]^T          TiledGemm, K=Q
Op 4: fused_add_rms_norm    [M,H] → [M,H]            Reduction along H
Op 5: Gate+Up GEMM          [M,H] × [2I,H]^T         TiledGemm, K=H
Op 6: SiLU+mul              [M,2I] → [M,I]            Pointwise
Op 7: Down GEMM             [M,I] × [H,I]^T           TiledGemm, K=I
```

Fusible segments:
- **Launch 1**: Op1 → Op2 (norm + QKV GEMM)
- **Launch 2**: FlashAttention (black box)
- **Launch 3**: Op3 → Op4 → Op5 → Op6 → Op7 (O_proj + norm + gate_up + SiLU + down)

Launch 3 has two different K dimensions (H for gate_up, I for down), so it requires
GMEM materialization between the gate_up and down phases. But all phases run in ONE
persistent kernel with per-tile barriers — no extra launches.

## Core Concept: TilePerimeter

The current perimeter model is kernel-level: "this kernel loads from GMEM and stores
to GMEM." The tile-level perimeter describes what flows in and out of each TILE
ITERATION within a kernel's mainloop.

### Why This Matters

The CUTLASS GEMM's K-loop iterates over K-tiles. At each iteration, it loads an
A-tile and B-tile via cp.async, does MMA, accumulates. The BOUNDARY of each
iteration is well-defined: the cp.async sites (inputs) and the MMA accumulators
(carries). This boundary is the tile perimeter.

To fuse an upstream operation (norm) into the GEMM, we replace what happens at the
A-tile boundary: instead of cp.async from GMEM, we load-normalize-store-to-SMEM.
The GEMM interior (MMA, B-loads, epilogue) is untouched.

The key insight: the tile perimeter is the minimal interface needed for fusion.
Everything inside is a black box.

### The Struct

```rust
struct TilePerimeter {
    name: String,
    kind: StageKind,              // Reduction, TiledGemm, Pointwise
    tile_shape: TileShape,        // (tile_m, tile_n, tile_k)

    // HOW ctaid.x,y MAP TO (m_tile, n_tile)
    // Extracted FROM the GEMM's own PTX, not reimplemented.
    tile_index: TileIndexMap,

    // What flows in/out per K-tile iteration
    inputs: Vec<TilePort>,        // A-tile, B-tile, scalars
    outputs: Vec<TilePort>,       // accumulated result, per-element formulas

    // State carried across K-tile iterations
    carries: Vec<TileCarry>,      // MMA accumulators, reduction partial sums

    // What runs after all K-tiles complete
    finalization: Option<Finalization>,
}
```

### TileIndexMap — The Critical Missing Piece

The CUTLASS GemmIdentityThreadblockSwizzle<4> maps ctaid to tiles:
```ptx
mov.u32   %r, %ctaid.x;
ld.param  %swizzle_log, [params+24];       // loaded from param struct
shr.s32   %m_tile, %r, %swizzle_log;      // m_tile = ctaid.x >> swizzle_log
and.b32   %n_group, %r, (1<<swizzle_log)-1;
add.s32   %n_tile, %n_group, %ctaid.y << swizzle_log;
```

After perimeter replacement, `swizzle_log` is computed inline from N. The prologue
MUST use the SAME computation — extracted from the GEMM's PTX via DefUseGraph, not
reimplemented.

```rust
struct TileIndexMap {
    /// The PTX lines that compute m_tile and n_tile from ctaid.x/y.
    /// Extracted from the GEMM body, renamed, reused verbatim in the prologue.
    tile_index_lines: Vec<String>,
    /// Register holding m_tile after the computation.
    m_tile_reg: String,
    /// Register holding n_tile after the computation.
    n_tile_reg: String,
}
```

## TilePerimeter for Each Stage Type

### Reduction (fused_add_rms_norm)

The norm reduces along H (hidden dimension), producing one scalar (inv_rms) per row.
It decomposes into three phases:

1. **Accumulate**: iterate over all H elements, compute sum_sq per row
2. **Finalize**: warp shuffle → SMEM broadcast → rsqrt → inv_rms
3. **Emit**: for each element, `output = input * inv_rms * weight`

In the fused pipeline:
- Phases 1+2 become the GEMM **prologue** (runs once per M-tile, before K-loop)
- Phase 3 becomes the **per-A-load transform** (runs at each cp.async site in K-loop)

```
tile_shape: (tile_m, 1, H)    // reduces full hidden dim, no N tiling
tile_index: inherited from downstream GEMM

inputs:
  - residual[M,H]      from GMEM
  - hs_input[M,H]      from GMEM
  - weight[H]           from GMEM

carries:
  - sum_sq: f32 per row  (ReductionAccumulator, 64 scalars = 256 bytes)

finalization:
  - inv_rms = rsqrt(sum_sq / H + eps)
  - stored in SMEM array: _ferrite_inv_rms[tile_m]

outputs:
  - inv_rms: scalar per row, broadcast to all K-tiles (in SMEM)
  - normalized_element: per-element formula, NOT materialized
    (applied inline at downstream GEMM's A-load sites)
```

The accumulation state is **256 bytes** for tile_m=64. Trivially fits in SMEM.

### TiledGemm (CUTLASS)

```
tile_shape: (64, 128, 32)     // from CUTLASS template config
tile_index: extracted from PTX (the swizzle computation)

inputs per K-iteration:
  - A_tile[tile_m, tile_k]   via cp.async (or from upstream stage's emit formula)
  - B_tile[tile_n, tile_k]   via cp.async (always weight matrix, from GMEM)

carries:
  - MMA accumulators: f32 fragments in REGISTERS (not SMEM)
  - buffer state: SMEM ping-pong for 3-stage software pipeline
  - K-pointer: induction variable advancing by tile_k * sizeof(bf16)

finalization:
  - epilogue: scale f32 accumulators, convert to bf16, store to GMEM
  - (or inject downstream pointwise before bf16 conversion)

outputs:
  - D_tile[tile_m, tile_n]   written to GMEM after all K-tiles
```

### Pointwise (SiLU+mul)

```
tile_shape: inherited from downstream GEMM's A-tile

inputs per K-iteration:
  - gate[tile_m, tile_k]    from GMEM (first half of gate_up output)
  - up[tile_m, tile_k]      from GMEM (second half, at +intermediate_bytes offset)

carries: none
finalization: none

outputs:
  - activated = silu(gate) * up, per-element
    NOT materialized — applied inline at downstream GEMM's A-load sites
```

## The DAG Structure

Within each fusible segment, the topology is a **chain with scalar side-channels**.
No general DAG scheduler is needed.

```
Segment A (Launch 1):
  norm ──→ [inv_rms: scalar/row] ──→ QKV GEMM (broadcast to all K-tiles)
  norm ──→ [normalized: formula] ──→ QKV GEMM (inline at each A-load)

Segment B (Launch 3):
  O_proj GEMM ──→ [GMEM] ──→ norm ──→ gate_up GEMM ──→ [GMEM] ──→ SiLU ──→ down GEMM
                                  └→ [inv_rms: scalar] ──→ gate_up GEMM
```

The "fan-out" from norm is two edges to the SAME consumer (scalar + per-element).
The SiLU "fan-in" (gate + up) is two halves of one tensor (offset addressing).

Representation:
```rust
struct FusibleSegment {
    stages: Vec<TilePerimeter>,
    edges: Vec<StageEdge>,
}

enum StageEdge {
    /// Reduction output consumed at next GEMM's A-loads (prologue + per-element)
    ReductionToGemm,
    /// GEMM output materialized to GMEM, next stage reads from GMEM
    GmemMaterialization,
    /// Pointwise formula injected at next GEMM's A-loads
    PointwiseToGemm,
}
```

## Accumulation and Streaming

Every stage has an **accumulation pattern**:

| Stage | Accumulates | State Size | When Finalized |
|-------|------------|-----------|---------------|
| rms_norm | sum_sq per row | 64 f32 = 256 bytes | After all H elements |
| GEMM | MMA partial products | Registers (~64 f32/thread) | After all K-tiles → epilogue |
| SiLU+mul | nothing | 0 | Immediate (pointwise) |

The accumulation state is always **tiny** compared to the data. Reductions produce
scalars. MMA accumulators live in registers. The data itself streams through GMEM.
SMEM holds tile buffers (transient) and accumulation state (persistent within M-tile).

### Streaming Between Phases

Within Launch 3 (O_proj + norm + gate_up + SiLU + down), the phases are:

```
Phase 1: O_proj GEMM
  - Writes [M, H] to GMEM (needed by attention residual AND norm)
  - Per-tile: each block writes its [tile_m, tile_n] output tile

  ── per-M-tile barrier (atomic counter) ──

Phase 2: norm accumulation
  - Reads O_proj output + residual from GMEM
  - Accumulates sum_sq per row (256 bytes in SMEM)
  - Finalizes inv_rms → SMEM array

Phase 3: gate_up GEMM (K=H)
  - At each A-load: load from GMEM, normalize (mul inv_rms, mul weight), store to SMEM
  - B-loads: gate_up weights from GMEM
  - MMA + epilogue → writes [M, 2I] to GMEM

  ── per-M-tile barrier ──

Phase 4: SiLU + down GEMM (K=I)
  - At each A-load: load gate+up from GMEM, compute silu(gate)*up, store to SMEM
  - B-loads: down_proj weights from GMEM
  - MMA + epilogue → writes [M, H] to GMEM
```

The barriers between phases are **per-M-tile** (atomic counter), not global. As soon
as all N-tiles for a given M-tile of O_proj are done, the norm for that M-tile can
start — even if other M-tiles of O_proj are still running.

### SMEM Budget

Peak SMEM = one GEMM's tile buffers + norm state:
- GEMM A+B tiles (3-stage pipeline): ~36KB for 64×128×32
- Norm inv_rms array: 256 bytes
- Norm warp scratch: 32 bytes
- Total: ~37KB — well within L4's 99KB optin max

The GEMMs run sequentially (different phases), so SMEM is reused between them.

### Why GMEM Between gate_up and down

The gate_up output is [M, 2*intermediate]. For Qwen 0.5B: [M, 2*4864] = [M, 9728]
elements in bf16 = ~19KB per row. For tile_m=64 rows: ~1.2MB. This exceeds SMEM.

The down GEMM's K-loop iterates over K=intermediate, reading [tile_m, tile_k] tiles
from the gate_up output. It needs access to the full intermediate dimension, which
requires GMEM.

However, this can be **streamed**: as gate_up writes each [tile_m, tile_n=128] output
tile, the down GEMM can start consuming the corresponding K-tiles (tile_n=128 contains
4 K-tiles of tile_k=32). Per-tile barriers enable this overlap.

## The FP Accumulation Non-Issue

The fused prologue runs the norm reduction with 128 threads (GEMM block size).
The standalone C kernel uses min(hidden_size, 1024) threads. Different thread counts
→ different FP accumulation order → different sum_sq → different inv_rms.

**test14j proved**: at layer 9, inv_rms differs by 5.45e-6 for 1 row, causing 1
element out of 896 to differ by 1 bf16 ULP.

**This is not a bug.** FP addition is non-associative. Different accumulation orders
produce different but equally valid results. The model tolerates this — it was trained
with mixed precision across varying hardware.

Once the fused pipeline IS the implementation (no comparison against the old C kernel),
the results are self-consistent and deterministic. The only validation: `vllm serve`
produces correct text.

## FlashAttention as a Black Box (and Future Work)

FA is Launch 2 — a black box between the two fusible segments. But it DOES have a
tile-level perimeter:

```
tile input:  Q_block [block_size, head_dim]    — one chunk of query rows
tile output: O_block [block_size, head_dim]    — corresponding output rows
```

FA's interior is complex in ways that break TilePerimeter assumptions:

1. **Data-dependent carries**: online softmax rescales the ENTIRE previous accumulator
   when a new max is found. Our model assumes carry updates are data-independent
   (additive accumulation, MMA). FA's carry update: `acc = acc * exp(old_max - new_max) + new`.

2. **Indirect addressing**: K/V loaded from paged cache via block table indirection.
   Not base+stride.

3. **Two-GEMM iteration**: each iteration does `scores = Q @ K^T` then
   `output += softmax(scores) @ V`. Two matrix multiplies interleaved with softmax.

4. **Variable trip count**: different sequences have different KV lengths.

These make FA unfusible with the current TilePerimeter model. But FA's BOUNDARY is
clean (rows in, rows out). Future work could fuse at FA's boundary:
- Replace Q-load sites with an inline transform (skip the split_qkv + RoPE launches)
- Inject into FA's epilogue (before O_block write)

This requires identifying Q-load and O-store sites in FA's PTX — the same perimeter
analysis we do for CUTLASS, just applied to FA's (much more complex) PTX.

## PTX Analysis Needed

### E.1: Tile Index Extraction (eliminates m_tile bug)

Extract the `(ctaid.x, ctaid.y) → (m_tile, n_tile)` computation FROM the GEMM's PTX.

The CUTLASS PTX at the kernel entry:
```ptx
mov.u32   %r115, %ctaid.x;
ld.param  %r116, [params+24];              // swizzle_log
shr.s32   %r1, %r115, %r116;              // m_tile
// ... mask and add for n_tile ...
```

After perimeter replacement, `swizzle_log` is computed inline from N. The analysis:

1. Find all uses of `%ctaid.x` via DefUseGraph
2. Trace forward to find the `shr` instruction → identifies m_tile register
3. Trace forward to find the `and`/`add` instructions → identifies n_tile register
4. Extract the LINE RANGE of the computation
5. The prologue REUSES these exact lines (register-renamed)

No reimplementation. One computation. Shared between prologue and GEMM body.

### E.2: K-Loop Identification

Among all detected loops (from `detect_loops()`), identify THE K-loop:
- Contains cp.async loads classified as A-matrix and B-matrix
- Has `CarryRole::TilePointer` carries (K-pointer advancing by tile_k)
- Has `CarryRole::MmaAccumulator` carries

Already nearly complete — combine existing `LoopDescriptor` + `CarryRegister` + `AsyncCopyPort` classification.

### E.3: A-Load Address Decomposition

For each A-load cp.async, decompose the GMEM address:
```
gmem_addr = base_ptr + m_offset + k_offset + lane_offset
```

Using `DefUseGraph::trace_backward` from `gmem_src`:
- Branch reaching K-loop induction var → k_offset
- Branch reaching m_tile register → m_offset
- Branch reaching %tid.x → lane_offset

This tells the prologue which parts of the address to preserve (lane) and which
come from the tile index computation (m_offset).

### What We Already Have

| Analysis | Status | Location |
|----------|--------|----------|
| Loop detection | Done | parser.rs: `detect_loops()` |
| Carry analysis | Done | parser.rs: `analyze_carries()` |
| A/B load classification | Done | fuse_cp_async.rs: `classify_cp_async_loads()` |
| Thread-to-row mapping | Done | pipeline_compile.rs: `extract_thread_row_map()` |
| Reduction decomposition | Done | pipeline.rs: `ReductionDecomposition` |
| Stage classification | Done | pipeline.rs: `PipelineStage::from_ptx()` |
| DefUseGraph | Done | parser.rs: `DefUseGraph::build()`, `trace_forward/backward` |
| Perimeter replacement | Done | perimeter.rs: `replace_perimeter()` |
| A-load replacement | Done | fuse_general.rs: `replace_a_loads_with_inline_fn()` |
| Persistent kernel wrapper | Done | persistent.rs: `make_persistent()` |
| Tile index extraction | **NEW** | parser.rs (proposed) |
| K-loop identification | **NEW** (mostly done) | pipeline.rs (proposed) |
| A-load address decomposition | **NEW** | fuse_cp_async.rs (proposed) |

## Implementation Plan

### Phase 1: Fix the Foundation

1. **Extract tile index from GEMM PTX** — the m_tile/n_tile computation, reusable
2. **Rewrite the prologue to use extracted tile index** — eliminate the reimplementation
3. **Test at ALL M values** (1-512) — the M-sweep must be 0.00e0

This immediately fixes the M>128 bug and makes the existing norm→GEMM fusion reliable.

### Phase 2: TilePerimeter Infrastructure

4. **Define TilePerimeter struct** in pipeline.rs
5. **Extract TilePerimeter from CUTLASS GEMM PTX** — tile_shape, tile_index, carries, A/B loads
6. **Extract TilePerimeter from rms_norm PTX** — accumulation pattern, finalization
7. **Generate norm prologue from TilePerimeter** — using extracted tile index, generated accumulation

### Phase 3: Segment Compilation

8. **Compile Segment A** (norm → QKV GEMM) from TilePerimeters
9. **Test Segment A** at all M values, compare against separate launches
10. **Compile Segment B** (O_proj + norm + gate_up + SiLU + down) as persistent kernel
11. **Test Segment B** at all M values

### Phase 4: Production

12. **Wire into llama.rs** — replace the 11-launch path with 3 launches
13. **Test with `vllm serve`** — correct text generation
14. **Benchmark** — latency and throughput vs cuBLAS baseline
15. **Add skinny-M CUTLASS configs** (8x128, 16x128) to close the decode gap vs cuBLAS

## Verification at Every Step

Every test must sweep: M=1,2,4,8,16,32,64,65,128,256,512

- **Unit tests**: TilePerimeter extraction matches known CUTLASS configs
- **Correctness tests**: fused vs separate launches, 0.00e0 at all M
- **24-layer test**: test14e equivalent, all tokens match (self-consistent, not vs C kernel)
- **Production test**: `vllm serve` with Qwen2.5-0.5B, coherent chat output
- **Benchmark**: `vllm bench latency` and `vllm bench throughput`

## Key Files

```
crates/ptx-fusion-macros/src/
  parser.rs           — DefUseGraph, detect_loops, analyze_carries + NEW tile index extraction
  pipeline.rs         — PipelineStage, StagePattern, ReductionDecomposition + NEW TilePerimeter
  pipeline_compile.rs — fuse_reduction_into_gemm, prologue generation (REWRITE to use TilePerimeter)
  fuse_cp_async.rs    — cp.async classification + NEW A-load address decomposition
  fuse_general.rs     — replace_a_loads_with_inline_fn (existing, reuse)
  persistent.rs       — make_persistent (existing, reuse)

crates/vllm-cuda/src/
  ferrite.rs          — launch_fused_add_norm_gemm, build_flat_params
  model/llama.rs      — forward pass wiring (3 launches)

crates/vllm-cuda/tests/
  test_transformer_block.rs — M-sweep tests, segment tests, 24-layer test
```
