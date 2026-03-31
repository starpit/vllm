# Handoff — Session 11

**READ FIRST**: `crates/ptx-fusion/FERRITE.md` — the master plan for the entire
ferrite project. Then `HANDOFF_SESSION10.md` — the starting point for this session.

## The goal

Fix the GPU correctness bug in `fuse_gemm_pointwise_gemm` so the register transfer
MLP kernel produces correct results. Session 10 left the kernel passing ptxas but
producing wrong results on GPU due to the producer preamble reading hardware
`%ctaid.x`/`%ctaid.y` (which belong to the consumer's grid, not the producer's).

## What was done this session

### 1. Producer tile index override (CORRECT — keep this)

The producer GEMM preamble reads `%ctaid.x`/`%ctaid.y` to compute its tile
coordinates. In the fused kernel, the grid belongs to the consumer. The producer's
tile coordinates come from the driver loop counter.

**Solution**: Inverse swizzle. Before each producer preamble, compute fake
ctaid.x/y from the consumer's m_tile and the driver iteration:

```
// CUTLASS swizzle: m_tile = ctaid_x >> swiz, n_tile = (ctaid_y << swiz) | (ctaid_x & mask)
// Inverse:
ctaid_x = (m_tile << swiz) | (n_tile & mask)
ctaid_y = n_tile >> swiz
```

Then replace `mov.u32 %rN, %ctaid.x` → `mov.u32 %rN, %r_prod_ctaid_x` in the
producer preamble (same pattern as `make_persistent_two_phase` in persistent.rs).

**Swizzle_log computation**: The perimeter replacement replaces the raw
`ld.param.u32 [param+24]` (swizzle_log) with a `selp` chain computed from N.
So we can't load swizzle_log from params directly. Instead, compute it from
the producer's N param at runtime using the CUTLASS formula:
```
grid_n = ceil(N / tile_n)
swizzle_log = (grid_n >= 3) ? 2 : (grid_n >= 2) ? 1 : 0
```
**NOTE**: CUTLASS SwizzleN=4 uses `grid_n >= 3` for log=2, NOT `grid_n >= 4`.
The persistent kernel code at persistent.rs:502 uses `>= 4` which is wrong for
SwizzleN=4 but happens to work for 128-wide tiles where grid_n is always large.

**extract_tile_index_map search limit**: Increased from 50 to 100 lines because
perimeter replacement inserts extra lines that push `%ctaid.y` past line 50.

**Where**: `fuse_gemm_pointwise_gemm()` in pipeline_compile.rs, around the gate
and up GEMM emission.

**Test**: `fuse_gemm_pointwise_gemm_ptxas` passes (was already passing before
this fix — ptxas doesn't catch runtime tile index errors).

### 2. Scratch layout fix — gmem_src-based addressing (CORRECT — keep this)

**The problem**: The redirect epilogue writes to scratch in row-major order:
`scratch_addr = scratch_base + row * tile_n * 2 + col * 2`. The SiLU A-load
replacement was reading with sequential `a_load_index * 16` offsets. These are
different thread→element mappings (epilogue uses OutputTileOptimalThreadMap,
A-loads use PitchLinearWarpRakedThreadMap).

**The solution**: Use the cp.async's GMEM source register to compute the scratch
address. With consumer `A_ptr = 0` and `lda = tile_n`, the GMEM source address
`gmem_src = row * lda * 2 + col * 2` is exactly the row-major byte offset within
the tile. Masking with `tile_mask = tile_m * tile_n * 2 - 1 = 8191` extracts the
within-tile offset (handles m_tile > 0 by wrapping out the tile-base offset).

```ptx
cvt.u32.u64  %r_fn_smoff, <gmem_src_rd64>;
and.b32      %r_fn_smoff, %r_fn_smoff, 8191;
add.u32      %r_fn_smoff, <scratch_base>, %r_fn_smoff;
ld.shared.v4.b32 {data}, [%r_fn_smoff];
```

**Changes**:
- `ALoadSource::Smem` now has a `tile_mask: u32` field (fuse_general.rs)
- The Smem handler computes scratch address from gmem_src instead of a_load_index
- `build_silu_mul_computation_smem` takes `tile_mask` parameter
- The per_site up-load also uses gmem_src-based addressing via `{GMEM_SRC}` placeholder
- New register `%r_fn_smoff` declared when source is Smem

**Host-side requirement**: Consumer params must have `A_ptr = 0` and `lda = tile_n`
(not intermediate_dim). This makes gmem_src encode the row-major tile offset.

### 3. Consumer param rename (CORRECT — keep this)

The consumer body (preamble, K-loop, epilogue) references `ferrite_params` which
is the PRODUCER's param block in the fused kernel. Must be renamed to
`ferrite_params_2`. Applied via `.replace("ferrite_params", "ferrite_params_2")`
on all consumer lines.

**Caution**: This is a substring replace. If applied to a line that already has
`ferrite_params_2`, it would produce `ferrite_params_2_2`. In practice this
doesn't happen because the consumer's raw PTX always uses `ferrite_params`.

### 4. Consumer preamble placement (BROKEN — this is the unsolved bug)

**The problem**: The consumer preamble includes a pipeline prologue that pre-fills
the 3-stage cp.async pipeline. In the fused kernel, cp.async A-loads are replaced
with ld.shared from SMEM scratch. If the consumer preamble runs before the producer
fills scratch, the pipeline prologue reads zeros from empty scratch.

**What was tried**:
1. Move entire consumer preamble inside the driver loop (after producer bar.sync)
   → This re-runs the CUTLASS accumulator zeroing code every iteration, destroying
   accumulated values from previous iterations.
2. Skip accumulator zeroing in the in-loop preamble by detecting `mov.f32 <accum>, 0f00000000`
   patterns using `cons_fused_desc.mma_accumulators` → Still produces zeros for
   blocks > 0. The single-tile case (1 block) works with 0.00e0 precision, but
   multi-block cases produce all-zero output from non-zero blocks.

**Root cause (not fully diagnosed)**: The consumer preamble is being treated as an
opaque blob. Moving it around and skipping lines doesn't work because it has complex
internal dependencies (pipeline state, buffer rotation, address registers, bounds
check predicates) that interact in non-obvious ways. The approach of "emit the
preamble and hope" violates the ferrite principle of analysis-driven code emission.

**What the analysis test revealed** (test: `analyze_fused_kernel_consumer_liveness`):
- 73 registers are live from consumer preamble into K-loop
- 35 of those are ctaid.x-derived (differ between blocks)
- Most are address registers (%rd308-311) and scratch temps
- The A-load addresses use the gmem_src masking correctly (verified by tracing)
- The B-load cp.async addresses depend on n_tile (ctaid-derived, expected)
- The B-load predicates depend on thread index, not ctaid (ruled out)
- No obvious single register explains the multi-block failure

### 5. GPU test results

**Single tile (hidden=64, intermediate=64, M=64, grid=1x1)**: **0.00e0 PASS**

This proves:
- Producer tile index override works
- Scratch addressing works (gmem_src masking)
- SiLU computation works
- Consumer K-loop + epilogue works
- param rename works

**Multi-block**: All zero output from blocks > 0. The MMA accumulators stay at
their initial zero value. Either the pipeline prologue doesn't fill mainloop SMEM
correctly, or there's a subtle interaction in the preamble that breaks for
ctaid.x > 0.

## The correct approach (not yet implemented)

The consumer preamble must be DECOMPOSED using the existing analysis infrastructure,
not treated as an opaque blob. The existing tools can do this:

### Decompose the consumer preamble into logical parts

Use `extract_tile_index_map()` → identifies tile index lines
Use `analyze_carries()` → identifies MMA accumulator registers
Use cp.async classification from `classify_cp_async_loads()` → identifies pipeline prologue
Use `DefUseGraph` → identifies register dependencies between parts

The preamble has these logical sections:
1. **Tile index**: ctaid.x/y → m_tile, n_tile via swizzle
2. **Bounds check**: early exit if tile out of range
3. **Address setup**: compute A_base, B_base from params + tile index
4. **Pipeline prologue**: fill 3 pipeline stages (cp.async or ld.shared from scratch)
5. **Accumulator init**: zero MMA accumulators
6. **K-loop entry**: branch to K-loop body

In the fused kernel:
- Parts 1-3 run ONCE before the driver loop (or once per driver iteration — they're idempotent)
- Part 4 runs INSIDE the driver loop AFTER the producer fills scratch
- Part 5 runs ONCE before the driver loop (accumulators persist across iterations)
- Part 6 runs after part 4

The analysis should identify the LINE BOUNDARIES between these parts by:
- Finding the first cp.async (or its replacement) → start of part 4
- Finding MMA accumulator defs (from `analyze_carries`) → part 5
- Using DefUseGraph to verify no data flows backward between parts

### Emit each part where it belongs

```
// BEFORE driver loop:
emit(tile_index_lines)         // part 1
emit(bounds_check_lines)       // part 2
emit(address_setup_lines)      // part 3
emit(accumulator_zeroing)      // part 5 (from analyze_carries)

// INSIDE driver loop:
emit(producer gate + up)
emit(bar.sync)
emit(pipeline_prologue_lines)  // part 4 (re-run each iteration)
emit(consumer K-loop)

// AFTER driver loop:
emit(consumer epilogue)
```

### What the CUTLASS 64x64x32 preamble actually looks like

From examining the consumer preamble in the fused PTX (lines 2980-4069):

```
Lines 2983-2999:  Tile index (ctaid.x/y → m_tile, n_tile via perimeter-replaced swizzle)
Lines 3000-3006:  Bounds check (grid_m <= m_tile || grid_n <= n_tile → early exit via bra $L__BB_cons_19 → ret)
Lines 3007-3110:  Address setup (A_base, B_base, strides, thread offsets, predicate masks)
Lines 3111-3200:  Pipeline stage 0 fill (first cp.async/ld.shared + SiLU for A, cp.async for B)
Lines 3200-3400:  Pipeline stage 1 fill
Lines 3400-3600:  Pipeline stage 2 fill
Lines 3600-3970:  More pipeline fills + buffer rotation
Lines 3996-4027:  Accumulator zeroing:
                    mov.f32 %f1204, 0f00000000;     ← seed zero
                    mov.f32 %f1205, %f1204;          ← spread via chain
                    mov.f32 %f1206, %f1204;
                    ... (32 total registers, %f1204-%f1235)
Lines 4028:       @%p118 bra $L__BB_cons_4;          ← enter K-loop body
```

Key observation: the accumulator zeroing at lines 3996-4027 uses a CHAIN pattern.
First `%f1204` is set to zero, then all other accumulators are set to `%f1204`.
This is NOT just a list of `mov.f32 %f, 0` — the registers `%f1204-%f1235` are
both accumulators AND the source for spreading zero. Skipping just the zero-init
doesn't work because `%f1204` is also used as a "zero constant" by later code.

The 32 registers zeroed here (%f1204-%f1235) are one of the two ping-pong accumulator
sets. The other set (%f788-%f851, 64 registers) is zeroed separately (at line 69,
before the driver loop). The MMA instructions alternate between these sets:
```
mma.sync {%f788,...}, {A}, {B}, {%f1232,...};   // D = A*B + C_set2, write set1
mma.sync {%f1232,...}, {A}, {B}, {%f788,...};   // D = A*B + C_set1, write set2
```

Both sets must be zero at the start. The pre-loop zeroing handles set1 (%f788-851).
The preamble zeroing handles set2 (%f1204-1235). Moving the preamble inside the
loop means set2 gets re-zeroed every iteration.

### Why this is different from what was tried

The failed approach: take the preamble as a string, emit it, and try to skip
specific patterns. This breaks because:
- The accumulator zeroing isn't just `mov.f32 %f, 0` — it uses a chain
  (`mov.f32 %f1204, 0; mov.f32 %f1205, %f1204; ...`) that also initializes
  non-accumulator scratch registers
- The pipeline prologue interleaves with address computation in ways that
  depend on the CUTLASS pipeline stage count
- Register liveness between sections isn't obvious from line content

The correct approach: use DefUseGraph to identify which lines belong to which
section based on their DATA DEPENDENCIES, then emit each section independently.

## Important implementation details

### How cp.async A-load replacement works (end-to-end)

`replace_a_loads_with_inline_fn` in fuse_general.rs replaces each A-matrix cp.async
with this sequence:

1. `setp.ne.b32 %p, <mask>, 0` — check the cp.async mask (bounds predicate)
2. `ld.shared.v4.b32 {temps}, [scratch_addr]` — load 16 bytes (8 bf16) from scratch
3. Per-site instructions: load "up" values from up_scratch at the SAME offset
4. Per-element: unpack bf16→f32, apply SiLU(gate)*up, repack f32→bf16
5. `st.shared.v4.b32 [smem_dst], {temps}` — store result to **mainloop SMEM**

Step 5 is critical: `smem_dst` is the ORIGINAL cp.async destination (the mainloop
SMEM slot in the TensorOp crosswise layout). The subsequent `ld.shared` + `ldmatrix`
in the K-loop body reads from this same SMEM slot for MMA. So the data flows:
scratch → SiLU → mainloop SMEM → ldmatrix → MMA.

### The redirect epilogue and dead ctaid reads

The redirect epilogue (`redirect_epilogue_to_smem`) replaces `st.global` with
`st.shared` but preserves all address computation code. This means the CUTLASS
epilogue's ctaid.x/y reads (for GMEM output address computation) are still present
as dead code. They compute GMEM addresses that are never dereferenced.

The epilogue also has `ld.global` instructions (for beta accumulation). These are
PREDICATED on beta != 0. With beta=0 in the producer params, they don't execute.
Unconditional `ld.global` instructions load epilogue params (alpha, beta, ptrs) from
the param block — these are always valid.

### Consumer GEMM processing in fuse_gemm_pointwise_gemm

The consumer GEMM goes through TWO processing steps:

1. `replace_a_loads_with_inline_fn(consumer_ptx, "", pointwise)` → produces `cons_fused`
   This replaces A-matrix cp.async with SiLU-from-scratch. The resulting PTX is a
   complete standalone GEMM kernel with modified A-loads.

2. `extract_gemm_body(&cons_fused)` → produces `cons_fused_desc`
   This decomposes the fused consumer into preamble/K-loop/epilogue.

The `cons_fused_desc` is what gets emitted in the fused kernel. Its preamble includes
the SiLU A-load replacements (in the pipeline prologue section). Its K-loop includes
SiLU A-load replacements (in the K-loop body).

### Register namespace merging

Producer registers use their original namespace (%r0, %r1, ..., %r739).
Consumer registers are offset by the producer's register count:
`offsets = compute_register_offsets(&prod_proto.registers, &cons_fused_proto.registers)`

`offset_all_registers(line, &offsets)` shifts consumer register numbers.
Named registers (containing `_`) are NOT offset — they use names like `%r_fn_smoff`.

### Grid and kernel params

The fused kernel entry:
```
.entry fused_mlp(
    .param .align 8 .b8 ferrite_params[88],      // producer (gate_up) flat params
    .param .align 8 .b8 ferrite_params_2[88],     // consumer (down) flat params
    .param .u32 _num_producer_n_tiles,             // driver loop count
    .param .u32 _intermediate_n_offset             // N-tile offset for "up" half
)
```

Grid: sized for the CONSUMER's output tile decomposition.
```
grid_m = ceil(M / tile_m)
grid_n = ceil(hidden / tile_n)
// Linearized with CUTLASS swizzle:
grid_dim = (grid_m * swizzle_tile, grid_n / swizzle_tile, 1)
```

### Flat param layout (88 bytes)

```
Offset  Size  Field
0       8     A_ptr (u64)
8       8     B_ptr (u64)
16      8     C_ptr (u64)
24      8     D_ptr (u64)
32      8     lda (u64)
40      8     ldb (u64)
48      8     ldc (u64)
56      8     ldd (u64)
64      4     M (i32)
68      4     N (i32)
72      4     K (i32)
76      4     alpha (f32)
80      4     beta (f32)
84      4     (padding)
```

For the consumer: A_ptr=0, lda=tile_n=64, B_ptr=down_weight, ldb=intermediate,
M=batch, N=hidden, K=intermediate (or tile_n for single-iteration test).

### Fused kernel section structure (from /tmp/regtransfer_mlp.ptx)

```
Line    Section
36      Consumer tile index (pre-loop) — manual swizzle computation
51      Driver loop initialization
136     $L_driver_loop:
137     Gate GEMM (producer, N-tile = iter)
725     Gate epilogue → SMEM scratch A
1556    bar.sync 0
1558    Up GEMM (producer, N-tile = iter + offset)
2147    Up epilogue → SMEM scratch B
2978    bar.sync 0
2980    Consumer preamble (in-loop)
4069    Down K-loop (consumer, A from SiLU scratch)
4598    Loop control → $L_driver_loop
4603    Down epilogue (output to GMEM)
5469    ret
```

## What to keep from commit eb2258965

1. **Tile index inverse swizzle** — the approach and math are correct
2. **ALoadSource::Smem with tile_mask** — gmem_src-based scratch addressing
3. **build_silu_mul_computation_smem with tile_mask** — per_site up-load via gmem_src
4. **Consumer param rename** — ferrite_params → ferrite_params_2
5. **extract_tile_index_map search limit = 100** — needed for perimeter-replaced PTX
6. **GPU test structure** — test configs, reference computation, comparison

## What to rebuild

`fuse_gemm_pointwise_gemm()` — specifically the consumer preamble handling.
Replace the opaque-blob approach with analysis-driven decomposition.

## Current state

**What passes ptxas**:
- Fused kernel: 140 registers, 1 barrier, 0 spills, 0 gmem
- All 131 unit tests pass
- `vllm-cuda` builds with the kernel loaded

**What passes on GPU**:
- Single-tile (1 block): 0.00e0

**What fails on GPU**:
- Multi-block: blocks > 0 produce zero MMA accumulators

## Key files

```
crates/ptx-fusion-macros/src/
  pipeline_compile.rs    — fuse_gemm_pointwise_gemm (the main assembly function),
                           GemmBodyDescriptor, epilogue analysis, epilogue redirect,
                           build_silu_mul_computation_smem (with tile_mask),
                           analyze_fused_kernel_consumer_liveness (debug analysis test)
  fuse_general.rs        — ALoadSource::Smem with tile_mask, gmem_src-based addressing
                           in replace_a_loads_with_inline_fn
  parser.rs              — extract_tile_index_map (search limit 100), DefUseGraph,
                           analyze_carries, detect_loops, CarryRole
  persistent.rs          — make_persistent_two_phase (reference for ctaid replacement)
  lib.rs                 — register_transfer_mlp! proc macro

crates/ptx-fusion/
  src/lib.rs             — re-export register_transfer_mlp
  tests/cuda_fuse_general.rs — register_transfer_mlp_gpu test

crates/vllm-cuda/src/
  ferrite.rs             — MLP_REGTRANSFER_PTX constant, kernel loading
```

## SMEM budget (unchanged from session 10)

```
L4 (sm_89): 100KB SMEM/SM, 50KB/block at 2 blocks/SM
Tile config: 64x64x32 for both producer and consumer

Layout:
  [0, 24KB)    CUTLASS mainloop (A+B tiles, 3-stage pipeline)
  [24KB, 32KB) Gate scratch (64x64 bf16 = 8KB)
  [32KB, 40KB) Up scratch (64x64 bf16 = 8KB)
  Total: 40KB < 50KB

Register budget:
  Producer accumulators: 64 f32 (from 64x64x32 MMA)
  Consumer accumulators: 64 f32 (persist across outer loop)
  CUTLASS scratch: ~40 mixed
  Total: ~168 regs < 256 max (at 2 blocks/SM)
```

## Multi-iteration consumer K-loop (not yet addressed)

When `intermediate > prod_tile_n` (64), the driver loop runs multiple iterations.
Each iteration produces one 64-wide slice of intermediate activations. The consumer
must process only that 64-wide K-slice per iteration, accumulating across iterations.

This requires:
1. Consumer K param = prod_tile_n (64), not intermediate_dim
2. Consumer B-load base address must advance by `prod_tile_n * ldb * sizeof(bf16)`
   across driver iterations (the weight matrix columns shift each iteration)
3. Consumer preamble's pipeline prologue must re-init each iteration (new scratch data)

This is SEPARATE from the multi-block bug and only matters when intermediate > 64.
The current test uses intermediate = 64 (1 driver iteration) to isolate it.

## Running the tests

```bash
# Unit tests (all 131)
cargo test -p ptx-fusion-macros --lib

# Specific tests
cargo test -p ptx-fusion-macros --lib fuse_gemm_pointwise_gemm_ptxas -- --nocapture
cargo test -p ptx-fusion-macros --lib analyze_fused_kernel_consumer_liveness -- --nocapture

# GPU test (single-tile passes, multi-block fails)
cargo test -p ptx-fusion --features cuda --test cuda_fuse_general register_transfer_mlp_gpu -- --nocapture --test-threads=1

# Build vllm-cuda with ferrite
cargo build -p vllm-cuda --features cuda,ferrite
```

## Rules (carried forward)

- **NEVER write PTX by hand** — all transforms via PTX analysis
- **NEVER build special-case macros** — extend existing infrastructure
- **NEVER dismiss divergence** — any non-zero diff must be investigated
- **NEVER test only at small M** — sweep M=1..512
- **NEVER assume what's slow** — profile with nsys first
- **NEVER hardcode CUTLASS assumptions** — extract from PTX using the parser
- **Build with** `cargo build -p vllm-cuda --features cuda,ferrite`
- **No one-off extractions** — every analysis must work on arbitrary CUTLASS PTX
- **SMEM budget is 50KB/block** at 2 blocks/SM on L4 (not 99KB)
- **Register pressure is SUM when both phases are live** — not MAX
- **NEW: Don't treat preambles as opaque blobs** — decompose using DefUseGraph,
  analyze_carries, and cp.async classification. Emit each logical section where
  it belongs based on analyzed data dependencies.
- **NEW: Consumer A_ptr must be 0, lda must be tile_n** — required for gmem_src
  scratch addressing to work. The host must set these in the consumer flat params.
- **NEW: CUTLASS SwizzleN=4 uses grid_n >= 3 for swizzle_log=2** — not >= 4.
  Check the CUTLASS source: `get_log_tile()` in threadblock_swizzle.h.
