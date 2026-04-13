# Scheduled megakernel — context handoff

> Read this end to end before touching anything in
> `ferrite-solver/src/{kernel_library,schedule,scheduled_codegen}.rs` or
> `templates/scheduled/megakernel.cu`.

## Quick start (15 minutes to orient)

You are working in **a git worktree**, not the primary repo. The path is
`/home/moosevan/vllm/.claude/worktrees/claude4` and the branch is
`worktree-claude4`. The user has multiple worktrees in parallel — do **not**
`cd` to the original repo at `~/vllm/vllm-rs`. Stay in this worktree.

```bash
cd /home/moosevan/vllm/.claude/worktrees/claude4/vllm-rs
git log --oneline -5    # confirm you're on worktree-claude4 at c399f7638 or later
```

**The 15-minute orientation read:**

1. This file (you're reading it)
2. `crates/ferrite-solver/src/kernel_library.rs` — start at line 50
   (`enum BoundKernel`), then read down through the cost / coalesce passes
3. `crates/ferrite-solver/src/schedule.rs` lines 1–270 — `CostModel`,
   `BARRIER_COST_MMA_UNITS`, `score_dag`, `partition_into_waves`
4. `crates/ferrite-solver/templates/scheduled/megakernel.cu` line ranges in
   the **megakernel.cu line map** below — don't read the whole file
5. `tools/cublas_bench/README.md` — the cuBLAS L4 ceiling numbers

After that you should be able to find any spot you need.

**The user's communication style** (this matters): terse, impatient, will push
hard for performance results. Has zero tolerance for "hacks" — every change
must fit the DAG/cost-model framework (see "Methodology rule" below). Has
called out hand-edits to `megakernel.cu` outside the framework as "the wrong
fix" repeatedly. Will *not* be impressed by 1 ms wins; the goal is structural
movement toward `<24 ms`. Don't pad your responses; give the answer and the
bench number.

## TL;DR — current state

- **Branch**: `worktree-claude4`
- **Last commit**: `27ddec346` "feat(kernel_library): polyalgorithmic cutlass tile selection"
- **Bench at seq=1024 (Llama-1B prefill, L4 sm_89)**: **~55 ms** avg over 50 iters
- **Golden**: passes; `max_abs_err=8192.00, max_rel_err=0.9434%` (unchanged baseline)
- **DAG state at production dims**: 80 nodes, 80 waves, 58 CTAs

The scheduled megakernel runs the entire LLaMA-1B prefill forward pass as a
single cooperative-launch CUDA kernel. The architecture is BSP wave-front: each
wave is a set of dependency-independent ops, separated by grid barriers.

## Performance goals

Hardware ceiling on L4 (`tools/cublas_bench/README.md` measured at seq=1024):

```
qkv         M=1024 K=2048 N=2304   1.571 ms (16 layers, cuBLAS bf16)
o_proj      M=1024 K=2048 N=2048   1.563 ms
gate        M=1024 K=2048 N=8192   6.536 ms
up          M=1024 K=2048 N=8192   6.760 ms
down        M=1024 K=8192 N=2048   7.060 ms
─────────────────────────────────
cuBLAS GEMM-only total: 23.49 ms (68.3% of L4 peak)
```

| target | meaning |
|---|---|
| **42 ms** | prior project best (the `fused/` megakernel — also a BSP design) |
| **~28-30 ms** | cuBLAS GEMM ceiling 23.5 + attention/norms/rope overhead, the realistic floor |
| **<24 ms** | requires *hiding* non-GEMM work behind GEMMs via fusion; cuBLAS-only can't do this |

**The user explicitly wants <24 ms.** That requires real fusion work that
eliminates gmem round-trips between phases. The DAG framework is the substrate
for this; the wins themselves are custom kernel work.

## Methodology rule (NON-NEGOTIABLE)

> Every optimization must be expressed as a DAG/cost-model transformation, not
> a hand-edit in megakernel.cu.

The user has called this out repeatedly. The acceptable shapes are:

- New `BoundKernel` variant in `kernel_library.rs`
- New `coalesce_*` pass that pattern-matches a DAG subgraph and rewrites it
- New `CutlassTile` (or similar) variant + a corresponding kernel namespace
- Cost-model improvements in `schedule.rs::CostModel` / `BoundKernel::cost`
- Cost-gated dispatch via `try_coalesce` in `coalesce_with_target_profile`

The unacceptable shapes are:

- Hardcoded "if this layer/phase use this kernel" branches in megakernel.cu
- Hand-edits to dispatch arms outside what a coalesce pass would emit
- Skipping cost gating ("but I know this is faster")

The rule exists because earlier in the session the user watched ~10 hours of
point-fix hacking that produced ~1 ms per intervention and missed the actual
structural wins. The DAG abstraction was built specifically to make
optimization principled; if we don't use it that way, it's just bookkeeping.

## What works (committed wins this session)

- `7c0a10239` **150 → 55 ms**: full cutlass-small dispatch for all 4 GEMM phases.
  The bug that blocked this for hours was an OOB stack write — see "Critical
  lessons" below.
- `ffbda6411` Fan-in fusion pass (`coalesce_consumer_fanin`): pattern-matches
  per-row producer rows feeding a single wave-coop consumer, rewrites as one
  fused wave-coop node. Wave count 128 → 80.
- `e77869455` Cost-gated coalesce framework (`try_coalesce` + `score_dag`):
  every fusion pass is decided by `score_dag(before)` vs `score_dag(after)`.
  Failed experiments auto-revert.
- `27ddec346` Polyalgorithmic cutlass tile selection: `CutlassGemmLayer` carries
  a `CutlassTile` choice; `coalesce_with_target_profile` searches over
  candidates per phase.

## What was tried and didn't work — DO NOT REPEAT

| experiment | result | why it failed |
|---|---|---|
| `pfl_cutlass` big tile (256×128×32) for gate_up/down | regressed (gate_up 19.3 → 21.3, down 10.2 → 13.7) | Adding the namespace bloats the megakernel binary; register pressure crosses dispatch arms in the same TU. The fused kernel uses big tile fine because it's a single-purpose kernel. |
| `cooperative_groups::this_grid().sync()` instead of gmem-flag spin barrier | identical (~115 µs/barrier either way) | Grid barrier cost is fundamental on L4 sm_89; mechanism doesn't matter |
| Monomorphic wave splitting (split topo level by `kernel_tag`) | null | DAG levels are *already* monomorphic per phase per layer; same-tag cross-level merge cancels out the split |
| 5-stage `pfl_cutlass_small` (vs 4) | regressed | Extra shmem hurts cumulative megakernel performance; 4 is the local optimum for this TU context |
| `pfl_cutlass_narrow` (128×64×32) as polyalgo candidate | ~1 ms regression | Cost gate picks Narrow for some phases per the bin-pack rounding model, but the bench reality is binary bloat from the second namespace cancels the win. The cost model doesn't yet have a `binary_footprint` term. |

**Lesson**: any new cutlass namespace added to `megakernel.cu` costs ~1 ms in
the rest of the binary via register pressure. The cost model needs to predict
this before we add more polyalgo candidates.

## Critical lessons

### The OOB stack write saga (READ THIS)

For ~6 hours we chased a "compute-sanitizer reports misaligned `__shared__`
write inside `flashinfer::write_o_reg_gmem` at seq=1024 only when case 12 is
present in the dispatch switch". I tried:

- Disabling `case 12` body to bare `break;` → still crashed
- Removing `cooperative_groups::this_grid().sync()` from `case 12` → still crashed
- Replacing pfl_cutlass big with pfl_cutlass small in `case 12` → still crashed
- Adding `__syncthreads()` at fi_attn dispatch entry → still crashed
- Marking `CutlassGemmLayer{GateUp}` as non-wave-coop → deadlocked
- Sanitizer pointed at `flashinfer/persistent.cuh:268`, an `ldsm` from a
  permuted smem offset that *should* be 16-aligned

**The actual bug**: `phase_clock[NUM_CLOCK_SLOTS=10]` was a stack array, indexed
by `op.kernel_tag`, and `kernel_tag` values for the cutlass arms were 10..13 —
**OOB stack write trampling adjacent locals, including FlashInfer's smem state
on the next dispatch**. Bumping `NUM_CLOCK_SLOTS` to 14 fixed everything in one
line and unlocked the 150 → 55 ms drop.

The sanitizer-reported address was the *downstream symptom*, not the cause.

**Generalized lesson**: any "weird smem misalignment" or "illegal address" that
doesn't make algorithmic sense in this codebase — **first check for sizing
constants in `megakernel.cu`**: `NUM_NODES`, `NUM_OPS`, `NUM_WAVES`,
`NUM_CLOCK_SLOTS`, any stack-allocated array indexed by `kernel_tag`. They get
out of sync with whatever new tag you just added. Both megakernel.cu and the
test harness reader (`scheduled_megakernel_test.rs`) need updating in lockstep.

### "Removing barriers" via fan-in fusion is a lie

I expected `coalesce_consumer_fanin` to save 48 grid barriers (16 layers × 3
patterns) and improve idle/sync by ~5 ms. The bench actually moved by ~1 ms.

**Why**: the fan-in dispatch arm contains an *intra-arm*
`cooperative_groups::this_grid().sync()` because the producer's outputs are
gmem writes that the consumer's gmem reads need globally visible. Net sync
count is unchanged — we just moved the sync from "wave-end barrier" to
"intra-arm sync". `BoundKernel::FusedFaninLayer::cost()` now accounts for this
by adding `BARRIER_COST_MMA_UNITS` to the cost, so the gate is honest.

**To actually save barriers**, the fused dispatch arm needs to use `__syncthreads`
(block-local) instead of `cg::this_grid().sync()`. That requires the consumer
to read only what its own CTA wrote — i.e., CTA-local dataflow. For the
gate+up case 12 this *was* true and dropping the inner sync saved ~0.4 ms
(commit `ea7ea6611`). For attn_norm→qkv it isn't true (cutlass GEMM reads all
of M; per-row producers wrote subsets across CTAs).

The **only path** to actually saving these syncs is **custom fused kernels**
where the producer output stays in shmem/registers and the consumer reads from
the same shmem. This is option 1 in the plan below.

### Bench is noisy

avg-over-50-iters bench has a ~0.5–1 ms variance band on this hardware. Don't
trust differences smaller than ~1 ms. Re-run twice if a "win" is in that band.

### Don't run multiple GPU tests in one cargo invocation

GPU tests share CUDA context. If one crashes, all subsequent ones fail with
sticky errors (cudaFuncSetAttribute → "illegal memory access" etc.). Always run
ONE GPU test at a time:

```
RUN_GPU_TESTS=1 cargo test -p ferrite-test-harness --release --features cuda <one_test_name> -- --nocapture --ignored
```

## The reified DAG and per-layer dataflow

Before reading any DAG code, internalize what work the model actually does.
LLaMA-1B prefill at seq=1024 has **16 layers**, each running **8 phases**:

```
Phase            reads                              writes              shape
─────────────────────────────────────────────────────────────────────────────
attn_norm        hidden_states[seq, hd]             rms_rope[seq, hd]   per row
qkv (GEMM)       rms_rope, qkv_w[3*hd, hd]          qkv[seq, 3*hd]      M=seq, K=hd, N=qkv_dim
rope             qkv (Q/K halves), KV cache         qkv_rotated, KV     per row
attention        Q, K, V (KV cache)                 attn_out[seq, hd]   per row
o_proj (GEMM)    attn_out, o_w[hd, hd]              hidden_states+=     M=seq, K=hd, N=hd
mlp_norm         hidden_states                      rms_gate[seq, hd]   per row
gate_up (2GEMMs) rms_gate, gate_w, up_w             silu_out[seq, id]   M=seq, K=hd, N=intermediate
down (GEMM)      silu_out, down_w[hd, id]           hidden_states+=     M=seq, K=id, N=hd
```

For Llama-1B: `hd=2048, id=8192, num_attn_heads=32, num_kv_heads=8, head_dim=64,
qkv_dim=(32+2*8)*64=3072, num_layers=16, seq_len=1024`.

**The DAG is a chain.** No intra-layer parallelism. Each phase strictly depends
on the previous. The only parallelism is *within* a phase, which we get via
wave-cooperative dispatch (every CTA participates in one cutlass GEMM call) and
LPT bin-packing (per-row tile phases distribute their tiles across CTAs).

**The reified DAG** (`reified_dag.rs`) produces nodes at the finest possible
granularity: one per `(layer, phase, row_tile, col_tile)` work unit. At seq=1024
with row_tile=16 and Llama-1B dims, that's ~85k nodes per forward. **Coalesce
passes** (`kernel_library.rs`) collapse subsets of these into fewer wave-cooperative
nodes — currently:
- `coalesce_with_flashinfer_attention`: all attention rows of a layer → 1 node
- `coalesce_gemm_phase`: all (row, col) GEMM tiles of a layer → 1 node
- `coalesce_consumer_fanin`: norm/rope rows + their wave-coop consumer → 1 node

After all passes the DAG has **80 nodes per forward** (down from 85k).

## How to read megakernel.cu (line range map)

`templates/scheduled/megakernel.cu` is ~3000 lines. Don't scroll. Use these
ranges (line numbers approximate, look for the labelled comments):

```
   1–30   File header + includes (cutlass, flashinfer, cooperative_groups)
  32–139  namespace pfl_cutlass            (256x128x32 big tile — UNUSED, see history)
 143–205  namespace pfl_cutlass_small      (128x128x32, 4 stages — current default)
 207–280  namespace pfl_cutlass_narrow     (128x64x32, 4 stages — polyalgo alt)
 282–370  TileOp struct, WAVE_OPS / NODE_ID_FOR_OP / WAVE_CTA_OFFSETS const arrays
 350–370  FlashInferKTraits + FlashInferRunner typedefs
 380–430  SchedRuntime struct (per-launch ptrs: flashinfer plan, flags, barrier, phase_clocks)
 440–500  tile_attn_norm + tile_mlp_norm   (warp-0 only, called from fan-in arms)
 503–560  Static shmem layout for tile bodies (TILE_SHMEM_BYTES, kSmemBytes calc)
 660–720  tile_rope                         (called from fan-in 15)
 720–820  tile_attention                    (DEAD — replaced by FlashInfer)
 820–960  tile_qkv / tile_o_proj            (DEAD — replaced by cutlass)
 960–1090 tile_gate_up                      (DEAD — replaced by cutlass)
1090–1265 tile_down                         (DEAD — replaced by cutlass)
1267–1310 tile_mlp_norm body
1440–1565 TILE_CUTLASS_GEMM_BODY macro      ← The shared cutlass dispatch body
1568–1581 grid_barrier                      ← gmem-flag spin barrier between waves
1582+    __global__ scheduled_megakernel    ← THE megakernel function
1614–1625 phase_clock declaration           ← !!! NUM_CLOCK_SLOTS gotcha lives here
1627+    main wave loop:
   1635   per-op phase_clock idle accounting
   1639   switch (op.kernel_tag)
   1742   case PHASE_ATTN_NORM..PHASE_DOWN  ← dead arms but referenced
   1750   case 10..13: cutlass dispatch arms (qkv, oproj, gate_up, down)
   1839   case PHASE_FLASHINFER_ATTN
   1890   case 14: fan-in attn_norm + cutlass qkv
   1911   case 15: fan-in rope + flashinfer attention
   1923   case 16: fan-in mlp_norm + cutlass gate+up
   1980   wave_coop tag check + tick stamper
   1987   end of per-op loop
   1995   grid_barrier end-of-wave
2070+    extern "C" launcher                ← cudaLaunchCooperativeKernel call site
   2090   kSmemBytes calculation (max of tile arena + cutlass + FI)
   2110   cudaLaunchCooperativeKernel
```

## TileOp / WAVE_OPS schema

```c++
struct TileOp { uint32_t kernel_tag; uint32_t layer; uint32_t row; uint32_t col; };
__device__ const TileOp WAVE_OPS[NUM_OPS] = { ... };
__device__ const uint32_t WAVE_CTA_OFFSETS[NUM_WAVES * (NUM_CTAS + 1)] = { ... };
__device__ const uint32_t NODE_ID_FOR_OP[NUM_OPS] = { ... };
```

- `kernel_tag` is the dispatch switch key. See "kernel_tag map" section above.
- `layer` is the layer index for layer-fused ops.
- `row` and `col` are the tile coordinates for `HandWrittenRowTile` ops, OR
  re-purposed slots for cutlass / fan-in ops:
  - For `CutlassGemmLayer`: `row = phase.tag()` (cross-check), `col = tile_tag` (0=Small, 1=Narrow)
  - For `FusedFaninLayer`: `row` unused (0), `col = tile_tag` for cutlass consumers
- `WAVE_CTA_OFFSETS` is a flat prefix-sum table: `WAVE_OPS[ wave_cta_offsets[w*(N+1)+c] .. wave_cta_offsets[w*(N+1)+c+1] ]`
  is CTA `c`'s op stream in wave `w`.
- The dispatch loop runs `for (i = off; i < end; ++i) { dispatch WAVE_OPS[i]; }` per CTA.
- The wave-end `grid_barrier` ensures all CTAs finish before the next wave's
  WAVE_OPS reads.

**The encoding lives in `scheduled_codegen/mod.rs::emit_scheduled_megakernel_cu`
lines ~80–125.** When you add a new BoundKernel variant, this is the file that
serializes it into WAVE_OPS.

## The fused/ kernel — prior art, NOT the same thing

`templates/fused/` contains the OTHER megakernel in this project — a hand-tuned,
non-scheduled, single-purpose Llama-1B prefill kernel that hits **~42 ms** at
seq=1024. It's the prior project best.

**Differences vs the scheduled kernel:**
- Fused has no scheduler — it's a hardcoded sequence of phases
- Fused uses TK kittens (`warp::load_async`, `warp::mma_ABt_base`) — a different
  template library than CUTLASS multistage
- Fused already has fused norm+gemm patterns in some paths — **read these for
  prior art** when implementing Step B
- Fused uses a single set of tile shapes hand-picked for Llama-1B — no polyalgo
- Fused does its own grid sync via `cooperative_groups::this_grid().sync()` and
  ad-hoc patterns

**Important**: the fused kernel uses `pfl_cutlass` (the big 256×128×32 tile) for
`down` successfully. We tried this in the scheduled kernel and regressed (see
"What was tried and didn't work"). The lesson is that *what works in
single-purpose fused doesn't necessarily work in scheduled* because of the
cross-arm register pressure — the megakernel's register footprint is bound by
the worst dispatch arm in the TU. This is why **Step A (resource-aware cost
model with `binary_footprint`) is the prerequisite for ever using big tile.**

The fused kernel's source layout: `templates/fused/{kernel.cu,
preamble_header.cu, gemm_cutlass_mcta.cu, gemm_gate_up_mcta.cu, attention.cu, rmsnorm.cu, ...}`.
For Step B you'll specifically want to read `gemm_cutlass_mcta.cu` (the only
file that uses pfl_cutlass and shows how `MmaMultistage` is invoked from inside
a megakernel) and any of the `rmsnorm*` files for the norm body.

## Variant generation pipeline

The scheduled megakernel is generated as **multiple variants** (one per shape
config) in the same TU, each with its own namespace:

```
namespace pfl_sched_tiny             { ... }
namespace pfl_sched_medium           { ... }
namespace pfl_sched_llama_3_2_1b_seq64   { ... }
namespace pfl_sched_llama_3_2_1b_seq1024 { ... }
```

Each gets its own:
- `WAVE_OPS`, `NODE_ID_FOR_OP`, `WAVE_CTA_OFFSETS` const arrays
- `__global__ scheduled_megakernel_<name>` function
- `extern "C" void launch_scheduled_megakernel_<name>(...)`

Generation entry points in `ferrite-solver/src/lib.rs`:
- `scheduled_prefill_tiny` (line ~654) — smallest test variant
- `scheduled_prefill_medium` (line ~700) — medium test variant
- `scheduled_prefill_variants_for_dsl` (line ~760) — production variants from
  the build script

The build script (`ferrite-test-harness/build.rs`) calls these and writes the
result to `crates/ferrite-test-harness/target/release/build/.../scheduled_prefill_*.cu`.
NVCC compiles all variants into one shared library that the harness loads at
test time.

**Practical implication**: when you change `megakernel.cu` (the askama template),
all variants regenerate on next `cargo build`. NVCC compilation can take
minutes per variant — be patient. If you see a stale cache, `rm` the
`~/.cudaforge/git/checkouts/...` `.a` files.



### Crates / files

```
vllm-rs/
  crates/
    ferrite-solver/
      src/
        kernel_library.rs       ← BoundKernel enum, coalesce passes, try_coalesce framework
        schedule.rs             ← CostModel, partition_into_waves, BARRIER_COST_MMA_UNITS, score_dag
        target_profile.rs       ← TargetProfile { num_sm, kernel choices, ... }
        reified_dag.rs          ← Per-phase per-row reified DAG (the input)
        scheduled_codegen/mod.rs ← Renders WAVE_OPS table from CoalescedDag + WaveSchedule
      templates/
        scheduled/megakernel.cu ← The rendered megakernel template (3000+ lines)
    ferrite-test-harness/
      tests/
        scheduled_megakernel_test.rs  ← Golden + smoke + bench tests
  tools/
    cublas_bench/               ← cuBLAS reference numbers per shape
```

### Key types

```rust
// kernel_library.rs
enum BoundKernel {
    HandWrittenRowTile { phase, layer, row, col },
    FlashInferAttentionLayer { layer },
    CutlassGemmLayer { layer, phase: GemmPhase, tile: CutlassTile },
    FusedFaninLayer { layer, producer_phase: Phase, consumer: FaninConsumer },
}

enum CutlassTile { Small /* 128x128x32 */, Narrow /* 128x64x32 */ }

enum FaninConsumer {
    CutlassGemm(GemmPhase, CutlassTile),  // tile preserved through fusion
    FlashInferAttention,
}

// schedule.rs
struct CostModel { dims, tiles, num_ctas }
fn score_dag(dag, num_ctas) -> u64    // = partition_into_waves(...).predicted_cost
const BARRIER_COST_MMA_UNITS: u64 = 100;  // 100 mma units ≈ 1 µs at 1.5 GHz
```

### kernel_tag map (megakernel.cu dispatch switch)

```
0..7   PHASE_*               hand-written per-row tile bodies (mostly dead now)
8      PHASE_FLASHINFER_ATTN flashinfer attention persistent runner
9      IDLE_SLOT             phase_clock idle/sync slot (not a tag)
10     CutlassGemmLayer Qkv      ← op.col carries CutlassTile (0=Small, 1=Narrow)
11     CutlassGemmLayer OProj    ← same
12     CutlassGemmLayer GateUp   ← same
13     CutlassGemmLayer Down     ← same
14     FusedFaninLayer AttnNorm → CutlassQkv
15     FusedFaninLayer Rope     → FlashInferAttention
16     FusedFaninLayer MlpNorm  → CutlassGateUp
```

`NUM_CLOCK_SLOTS = 17` in both `megakernel.cu` and `scheduled_megakernel_test.rs`.
**Keep these in sync when adding tags.**

### Cost-gated coalesce flow (`coalesce_with_target_profile`)

```
reified DAG
   │
   ▼ trivial 1:1 coalesce
   │
   ▼ try_coalesce: coalesce_with_flashinfer_attention
   │
   ▼ try_coalesce: cutlass polyalgo per phase × per tile candidate
   │       (CUTLASS_TILE_CANDIDATES = [Small, Narrow])
   │
   ▼ try_coalesce: coalesce_consumer_fanin (3 patterns)
   │
final CoalescedDag → partition_into_waves → render megakernel.cu
```

Every `try_coalesce` reverts the candidate if `score_dag(after) >= score_dag(before)`.

## Current bench breakdown

```
attn_norm          0.00 ms  ← absorbed into fanin an+qkv
qkv (hw)           0.00 ms
rope               0.00 ms  ← absorbed into fanin rope+at
attention(hw)      0.00 ms
o_proj (hw)        0.00 ms
mlp_norm           0.00 ms  ← absorbed into fanin mn+gtup
gate_up (hw)       0.00 ms
down (hw)          0.00 ms
fi_attn            0.00 ms
idle/sync          5.90 ms  ← grid barrier wait, 80 waves × ~75 µs
qkv (cls)          0.00 ms  ← absorbed into fanin an+qkv
o_proj (cls)       2.46 ms  ← standalone (no fan-in pattern)
gate_up (cls)      0.00 ms  ← absorbed into fanin mn+gtup
down (cls)         9.77 ms  ← standalone (down's only consumer is next layer's attn_norm, which is absorbed; the down itself is not)
fanin an+qkv       7.50 ms  ← attn_norm rows + cutlass qkv
fanin rope+at     10.60 ms  ← rope rows + flashinfer attention
fanin mn+gtup     22.54 ms  ← mlp_norm rows + cutlass gate+up
─────────────────────────────────
sum-of-maxes:     58.67 ms
avg over 50:      55.28 ms
```

**Where the gaps are**:
- `fanin mn+gtup` 22.5 ms — gate_up alone is 19 ms cuBLAS-equivalent should be 13.3
- `down (cls)` 9.77 ms — cuBLAS 7.06
- `fanin an+qkv` 7.50 ms — attn_norm should ideally cost ~0 (hidden in qkv prologue)
- `fanin rope+at` 10.60 ms — rope ~7 ms could be hidden in attention

The structural ceiling for the current BSP-with-grid-barriers architecture is
**~36 ms** (sum of optimistic phase costs + minimum barrier overhead). To go
below that requires either:
- Custom fused kernels eliminating gmem round-trips (option 1 below)
- Cross-layer pipelining with double-buffered hidden_states (option 4 below)

## The plan (in priority order)

### Step A — Resource-aware cost model (PURE RUST, ONE SESSION) — START HERE

**Concrete TODO checklist** (execute in order, each step is committable):

- [ ] **A.1** Add `Resources` struct to `kernel_library.rs`. Fields:
  `shmem_bytes: u32, registers_per_thread: u32, threads_per_cta: u32,
  binary_footprint_bytes: u32`. Add `BoundKernel::resources(&self) -> Resources`
  method. For each existing variant, return reasonable estimates:
  - `HandWrittenRowTile{Norm}`: shmem=0, regs=24, threads=32 (warp-0), footprint=2KB
  - `HandWrittenRowTile{Rope}`: shmem=0, regs=32, threads=32, footprint=3KB
  - `FlashInferAttentionLayer`: shmem=~70KB (FI_SHARED_STORAGE_BYTES at compile time
    — pull the constant in via the cost model or hardcode the measured value),
    regs=128, threads=256, footprint=20KB (FI is heavy)
  - `CutlassGemmLayer{tile=Small}`: shmem=sizeof(pfl_cutlass_small::SharedStorage)≈49KB,
    regs=128, threads=256, footprint=15KB
  - `CutlassGemmLayer{tile=Narrow}`: shmem=sizeof(pfl_cutlass_narrow::SharedStorage)≈25KB,
    regs=96, threads=256, footprint=13KB
  - `FusedFaninLayer{...CutlassGemm(...,tile)}`: same as the inner cutlass + ~1KB
    fan-in glue
  - These numbers don't have to be exact — they have to be RELATIVELY correct
    so the gate's inequalities work.
- [ ] **A.2** Extend `CostModel` with a per-binding penalty:
  `binary_footprint_penalty_per_use: u64`. Calibration: pick a value such that
  adding a polyalgo candidate with `binary_footprint_bytes=15000` makes the
  Narrow tile reject at seq=1024. The current data points: Narrow regressed by
  ~1 ms at production dims; the cost gate at production dims sees ~5000 mma
  units of work per phase; so the penalty should be ~6000 mma units per
  binary KB if we want 15KB to outweigh the savings.
- [ ] **A.3** Update `BoundKernel::cost(model)` to add the binary footprint
  penalty: `cost += model.binary_footprint_penalty(self.resources())`. The
  penalty is paid PER USE — so a polyalgo candidate that's used in many waves
  pays it many times. This naturally penalizes adding rarely-winning kernels.
- [ ] **A.4** Add a per-CTA shmem budget check in `partition_into_waves`. After
  bin-packing each wave, compute `max_shmem_per_cta = max over CTAs of sum of
  shmem of all bindings on that CTA`. If `max_shmem_per_cta > TARGET_MAX_SHMEM`,
  return a wave schedule with `predicted_cost = u64::MAX` so the cost gate
  rejects it. Today this is checked at compile time via static_assert but not
  during cost-model search.
- [ ] **A.5** Add unit test `cost_model_rejects_binary_bloat`:
  - Build a tiny DAG
  - Apply `try_coalesce` with a transform that adds a CutlassGemmLayer{Narrow}
    binding for one phase
  - Verify the transform reverts (cost gate sees the binary penalty)
  - This locks in the behavior so future changes don't silently re-enable bloat
- [ ] **A.6** Re-enable `CutlassTile::Narrow` in `CUTLASS_TILE_CANDIDATES` (it's
  already there). Run the bench. Expected outcomes:
  - Either the gate now correctly REJECTS Narrow for every phase (back to
    Small everywhere) and the bench returns to ~53 ms
  - OR the gate accepts Narrow for some phase where it actually wins (genuine
    polyalgo win, > 1 ms improvement)
- [ ] **A.7** Update `try_coalesce_rejects_a_regression` test to also assert
  the binary-bloat reject case (or make it a separate test)
- [ ] **A.8** `cargo fmt -p ferrite-solver && cargo clippy -p ferrite-solver -- -D warnings`
- [ ] **A.9** Run all 3 GPU tests one at a time. Golden must still pass with
  unchanged err. Bench should be 53–55 ms (in the noise band).
- [ ] **A.10** Commit. Suggested commit message:
  `feat(kernel_library): resource-aware cost model with binary_footprint penalty`

**Bench impact**: ~0 ms (the gate now correctly auto-rejects regressions). The
*win* is that the framework is now safe for adding more polyalgo candidates,
fused norm+gemm kernels, etc. Without this, every Step B/C experiment risks
the same kind of binary-bloat regression we just hit with Narrow.

**File:line pointers** for the work:
- `crates/ferrite-solver/src/kernel_library.rs:50` — `enum BoundKernel`
- `crates/ferrite-solver/src/kernel_library.rs:185` — `impl BoundKernel { fn kind() }`
- `crates/ferrite-solver/src/kernel_library.rs:248` — `impl BoundKernel { fn cost() }`
- `crates/ferrite-solver/src/schedule.rs:34` — `struct CostModel`
- `crates/ferrite-solver/src/schedule.rs:96` — `fn cutlass_gemm_total`
- `crates/ferrite-solver/src/schedule.rs:158` — `pub fn partition_into_waves`
- `crates/ferrite-solver/src/kernel_library.rs:993` — `fn try_coalesce`
- `crates/ferrite-solver/src/kernel_library.rs:1010` — `fn coalesce_with_target_profile`

(Line numbers as of commit `c399f7638`. Use `grep -n` if drift.)

---

**Original Step A scope (kept for context):**


**Motivation**: today's cost gate doesn't predict binary bloat. Adding
`pfl_cutlass_narrow` regressed by ~1 ms even though the cost model said it
would help. Without this, we can't safely add more polyalgo candidates or
fused kernels.

**Scope**:

1. Add `Resources` struct on `BoundKernel`:
   ```rust
   pub struct Resources {
       pub shmem_bytes: u32,
       pub registers_per_thread: u32,
       pub threads_per_cta: u32,
       /// Estimated bytes the dispatch arm contributes to the megakernel
       /// binary. Polyalgo searches that add this candidate pay this in
       /// every wave that runs.
       pub binary_footprint_bytes: u32,
   }

   impl BoundKernel {
       pub fn resources(&self) -> Resources { ... }
   }
   ```

2. Extend `CostModel::cutlass_gemm_total` (or add a new cost dimension) to
   include a small per-binding penalty proportional to `binary_footprint_bytes`.
   Calibrate the penalty so adding a polyalgo candidate that doesn't actually
   speed up its phase gets rejected (the Narrow regression case).

3. Add a per-CTA shmem budget check in `partition_into_waves`. Today we just
   `static_assert` at compile time on `kSmemBytes ≤ TARGET_MAX_DYNAMIC_SHMEM_BYTES`,
   which is coarse. The scheduler should reject bin-packs that would exceed
   the SM budget at runtime.

4. New unit test: `cost_model_rejects_binary_bloat` — verify the gate would
   have rejected the Narrow tile addition.

**Bench impact**: probably 0 ms (Narrow may auto-revert via the new gate). The
*win* is making the framework correct.

**Estimated time**: 1 session, ~300 lines of Rust.

### Step B — Custom fused norm+gemm kernel (CUDA WORK, MULTI-SESSION)

**Motivation**: this is where the actual bench wins live. The fused kernel reads
`hidden_states` once, computes the norm scale in shmem, then runs the cutlass
mainloop reading from the (already-normalized) shmem. The `rms_rope` /
`rms_gate` gmem buffer becomes unnecessary. **Saves the entire norm wave +
the gmem round-trip**.

**Estimated savings**: 5–8 ms across attn_norm→qkv and mlp_norm→gate_up.

**Scope**:

1. New `BoundKernel` variant:
   ```rust
   FusedNormGemmLayer {
       layer: u16,
       norm_phase: Phase,        // AttnNorm or MlpNorm
       gemm_phase: GemmPhase,    // Qkv or GateUp
       tile: CutlassTile,
   }
   ```

2. New coalesce pass `coalesce_norm_into_gemm`. Pattern: per-row
   `HandWrittenRowTile { norm_phase }` whose **only** consumer is a single
   `CutlassGemmLayer { gemm_phase }` for the same layer. Same shape as the
   existing `coalesce_consumer_fanin`. ~100 lines.

3. New cost model entry: cost should be roughly `cutlass_gemm_total - one_norm_wave_cost`,
   so the gate accepts when the norm is genuinely absorbed.

4. **The hard part — new dispatch arm in `megakernel.cu`**: a custom kernel that:
   - cp.async loads `hidden_states[m_tile, k]` into shmem
   - Computes per-row sum-of-squares + scale (warp reduce)
   - Applies scale + the rms norm weight in-place in shmem (or in registers)
   - Hands off to the cutlass mainloop, which reads from this shmem instead of
     the original `IteratorA::Params` gmem layout
   - cutlass mainloop then runs as normal, producing the GEMM output to gmem

5. The cutlass `IteratorA` needs to be retargeted at shmem instead of gmem, OR
   we write a custom mainloop that loads from shmem. CUTLASS supports the
   former via `cutlass::transform::threadblock::RegularTileAccessIterator` —
   look at how `MmaMultistage` constructs its smem iterators internally.

6. **Alternative approach**: instead of routing cutlass through shmem (which is
   invasive), write a hand-rolled wmma fused kernel similar to what's in
   `templates/fused/gemm_cutlass_mcta.cu` and `templates/fused/preamble.cu`.
   The fused/ kernel does fused norm+gemm in some paths — vendor that pattern.

7. Once the kernel exists, the polyalgo gate will accept it automatically when
   it beats the unfused baseline.

**Estimated time**: 2-3 sessions. The DAG pass is ~half a session; the kernel
itself is the bulk.

**Risks**:
- Shmem budget: norm needs activation rows in shmem; cutlass mainloop also
  needs A/B in shmem. Fitting both in 99 KB on L4 is tight. May need to use
  smaller M_tile.
- Numerical precision: the fused norm+gemm needs to keep enough precision
  through the in-shmem normalize step to match the unfused golden.
- This breaks the polyalgo abstraction slightly: the fused kernel is no
  longer just "a cutlass GEMM" — it's a custom kernel that *contains* a
  cutlass mainloop. The DAG pass logic is the same, the dispatch arm is
  different.

### Step C — Custom fused down+residual+attn_norm (CUDA WORK, MULTI-SESSION)

**Motivation**: layer N's `down` writes `hidden_states` (with residual). Layer
N+1's `attn_norm` reads `hidden_states`. If `down`'s epilogue computes the
residual + the next layer's norm scale + applies it, the attn_norm wave is
eliminated entirely.

**Estimated savings**: 3-5 ms (eliminates 1 wave per layer × 16 layers + the
gmem round-trip for `hidden_states`).

**Scope**:
1. New variant `FusedGemmEpilogueNorm { gemm: CutlassGemmLayer, next_norm_phase: Phase }`
2. New coalesce pass `coalesce_gemm_epilogue_norm`: pattern is "the only
   consumer of this CutlassGemmLayer's output is the next layer's per-row
   norm rows"
3. **The hard part**: a custom CUTLASS epilogue template that does the residual
   add, computes the per-row reduce in the epilogue's smem, applies the scale,
   then writes the normalized result.

**Estimated time**: 2 sessions. CUTLASS epilogue templates are complex.

### Step D — Cross-layer pipelining (MAJOR REWRITE)

**Motivation**: the only path to <24 ms is overlapping work across the layer
boundary. Today every wave uses all 58 CTAs and we wait for each layer to
complete before starting the next.

**Scope**:
1. **Wave abstraction rewrite**: a wave can use a SUBSET of CTAs. Other CTAs
   in the same wave run a different op concurrently.
2. **Schedule rewrite**: `partition_into_waves` allows two ops in the same
   wave to run on disjoint CTA subsets if they're mutually independent in the
   DAG.
3. **Double-buffered `hidden_states`**: layer N writes to `hidden_states_a`,
   layer N+1 reads from `hidden_states_a` and writes to `hidden_states_b`.
   `down` and `attn_norm` of consecutive layers no longer have a true data
   dep on the same buffer.
4. **Per-CTA op stream rebuild**: today's `WAVE_CTA_OFFSETS` table assumes all
   CTAs in a wave have the same op stream layout. This needs to support
   per-CTA-subset op streams.

**Estimated time**: 4-6 sessions. This is the biggest rewrite but the only
path to actually beating cuBLAS-only.

**Risks**: doubles the `hidden_states` memory footprint. The fused kernel
shows this is acceptable on L4 but it changes the runtime memory plan.

### Steps that explicitly should NOT be done (covered above)

- More cutlass tile shapes without resource-aware cost model (Step A blocks this)
- Wave merging on adjacent waves (no dep-free pairs exist)
- Switching grid_barrier mechanism (CG vs gmem-flag is identical cost)
- Bumping `MODEL_ROW_TILE` (requires deleting dead handwritten bodies; per-tile
  dispatch overhead is sub-millisecond, not the bottleneck)

## Testing methodology

### Three GPU tests, run individually

```bash
cd vllm-rs

# Golden: validates correctness against committed CPU reference at seq=64
RUN_GPU_TESTS=1 cargo test -p ferrite-test-harness --release --features cuda \
  llama_1b_seq64_h_final_matches_committed_golden -- --nocapture --ignored

# Smoke: single-launch sanity check at seq=1024
RUN_GPU_TESTS=1 cargo test -p ferrite-test-harness --release --features cuda \
  llama_1b_seq1024_smoke -- --nocapture --ignored

# Bench: 50-iter timed run with per-phase max-clocks rollup
RUN_GPU_TESTS=1 cargo test -p ferrite-test-harness --release --features cuda \
  llama_1b_seq1024_bench -- --nocapture --ignored
```

**Run them ONE AT A TIME** (sticky CUDA context errors otherwise).

### Acceptance criteria for any change

- **Golden invariant**: `max_abs_err <= 8192.00, max_rel_err <= 0.95%`. These
  are baked into `regen_llama_1b_seq64_golden` and represent the bf16
  accumulator-precision floor. Any change that increases either is broken.

- **Bench**: report avg-over-50-iters. Differences < 1 ms are noise. Any change
  that regresses by > 1 ms needs justification or revert.

- **Unit tests**: `cargo test -p ferrite-solver --lib`. The
  `try_coalesce_rejects_a_regression` test must still pass (proves the gate
  works). The `coalesce_with_target_profile_dispatches_correctly` test must
  still pass (proves the polyalgo + fan-in passes compose correctly).

- **fmt + clippy**: `cargo fmt -p ferrite-solver && cargo clippy -p ferrite-solver -- -D warnings`
  before every commit. Same for `ferrite-test-harness` if you touched it.

### Sanitizer for debugging

```bash
RUN_GPU_TESTS=1 /usr/local/cuda-12.9/compute-sanitizer/compute-sanitizer \
  --print-limit 5 \
  cargo test -p ferrite-test-harness --release --features cuda \
  <test> -- --nocapture --ignored 2>&1 | grep -E "Invalid|Error|====" | head -40
```

**Reminder**: a sanitizer-reported smem misalignment is *probably* a sizing
constant OOB, not a real alignment bug. Check `NUM_CLOCK_SLOTS`, `NUM_NODES`,
`NUM_OPS`, `NUM_WAVES`, anything indexed by `kernel_tag` first.

### Build flags

- `--features cuda` (no `ferrite` for `ferrite-solver` — that's a different crate)
- The cudaforge cache is shared across worktrees. If a build picks up stale
  `.a` files, `rm` them under `~/.cudaforge/git/checkouts/`.

### CUDA 12.9 toolchain

```bash
PATH=/usr/local/cuda/bin:$PATH  # 12.9.86 — already in nvcc default
```

The fused/ baseline kernel takes minutes to compile via `cicc` — don't kill it.

## How to commit

```bash
cargo fmt -p ferrite-solver
cargo clippy -p ferrite-solver -- -D warnings
git add -u crates
git commit -m "$(cat <<'EOF'
<type>(<scope>): <one-line summary>

<paragraph: what changed and why>

<paragraph: bench numbers before/after when applicable>

<paragraph: known issues, follow-ups, or design decisions>

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>
EOF
)"
```

Always run fmt + clippy + tests before commit. Never amend; always create new
commits. Don't push.

## Recent commit log (most-recent first, this session's work)

```
27ddec346 feat(kernel_library): polyalgorithmic cutlass tile selection
97f8f0ea1 refactor(schedule): thread num_ctas through CostModel + add cutlass_gemm_per_cta
320cf9a07 perf(scheduled_megakernel): pfl_cutlass_small 3 → 4 cp.async stages
ea7ea6611 perf(scheduled_megakernel): drop CTA-local grid sync between gate and up cutlass passes
ffbda6411 feat(kernel_library): coalesce_consumer_fanin pass + fan-in dispatch arms
e77869455 feat(kernel_library): cost-gated coalesce framework
79d3ced42 docs(scheduled_megakernel): record null-result CG grid sync experiment
a5e40d7be docs(scheduled_megakernel): record failed pfl_cutlass big-tile experiment
7c0a10239 perf(scheduled_megakernel): full cutlass-small dispatch — 150 → 55 ms
73c62ee4d fix(scheduled_megakernel): tag CutlassGemmLayer phases as wave-cooperative for tick stamping
187c77b07 perf(scheduled_megakernel): cutlass-small for qkv/oproj/down — 196 → 149 ms
```

## Final note for the next session

The user is impatient and will push hard for results. You will be tempted to
hand-edit `megakernel.cu` for "just one quick fix". **Don't.** Every time we
did this in the previous session, we either regressed by 1 ms or got nothing
and burned an hour. The DAG/cost-model framework is the only thing that has
delivered repeatable wins (150 → 55 ms via 4 cutlass arms + 1 line of OOB
fix, all driven by passes).

If a fix doesn't fit a DAG pass, it's almost certainly the wrong fix.

The user explicitly stated the goal is `<24 ms`. That's below the cuBLAS-only
ceiling and requires custom fused kernels (Step B and/or D). It is achievable
but it takes real CUDA engineering, not micro-tuning. Plan multi-session and
don't try to land it in one shot.
