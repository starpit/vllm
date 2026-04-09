# Scheduled megakernel — context handoff

> Read this end to end before touching anything in
> `vllm-tk-macros-core/src/{kernel_library,schedule,scheduled_codegen}.rs` or
> `templates/scheduled/megakernel.cu`.

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
RUN_GPU_TESTS=1 cargo test -p vllm-tk-test-harness --release --features cuda <one_test_name> -- --nocapture --ignored
```

## Architecture map

### Crates / files

```
vllm-rs/
  crates/
    vllm-tk-macros-core/
      src/
        kernel_library.rs       ← BoundKernel enum, coalesce passes, try_coalesce framework
        schedule.rs             ← CostModel, partition_into_waves, BARRIER_COST_MMA_UNITS, score_dag
        target_profile.rs       ← TargetProfile { num_sm, kernel choices, ... }
        reified_dag.rs          ← Per-phase per-row reified DAG (the input)
        scheduled_codegen/mod.rs ← Renders WAVE_OPS table from CoalescedDag + WaveSchedule
      templates/
        scheduled/megakernel.cu ← The rendered megakernel template (3000+ lines)
    vllm-tk-test-harness/
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

### Step A — Resource-aware cost model (PURE RUST, ONE SESSION)

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
RUN_GPU_TESTS=1 cargo test -p vllm-tk-test-harness --release --features cuda \
  llama_1b_seq64_h_final_matches_committed_golden -- --nocapture --ignored

# Smoke: single-launch sanity check at seq=1024
RUN_GPU_TESTS=1 cargo test -p vllm-tk-test-harness --release --features cuda \
  llama_1b_seq1024_smoke -- --nocapture --ignored

# Bench: 50-iter timed run with per-phase max-clocks rollup
RUN_GPU_TESTS=1 cargo test -p vllm-tk-test-harness --release --features cuda \
  llama_1b_seq1024_bench -- --nocapture --ignored
```

**Run them ONE AT A TIME** (sticky CUDA context errors otherwise).

### Acceptance criteria for any change

- **Golden invariant**: `max_abs_err <= 8192.00, max_rel_err <= 0.95%`. These
  are baked into `regen_llama_1b_seq64_golden` and represent the bf16
  accumulator-precision floor. Any change that increases either is broken.

- **Bench**: report avg-over-50-iters. Differences < 1 ms are noise. Any change
  that regresses by > 1 ms needs justification or revert.

- **Unit tests**: `cargo test -p vllm-tk-macros-core --lib`. The
  `try_coalesce_rejects_a_regression` test must still pass (proves the gate
  works). The `coalesce_with_target_profile_dispatches_correctly` test must
  still pass (proves the polyalgo + fan-in passes compose correctly).

- **fmt + clippy**: `cargo fmt -p vllm-tk-macros-core && cargo clippy -p vllm-tk-macros-core -- -D warnings`
  before every commit. Same for `vllm-tk-test-harness` if you touched it.

### Sanitizer for debugging

```bash
RUN_GPU_TESTS=1 /usr/local/cuda-12.9/compute-sanitizer/compute-sanitizer \
  --print-limit 5 \
  cargo test -p vllm-tk-test-harness --release --features cuda \
  <test> -- --nocapture --ignored 2>&1 | grep -E "Invalid|Error|====" | head -40
```

**Reminder**: a sanitizer-reported smem misalignment is *probably* a sizing
constant OOB, not a real alignment bug. Check `NUM_CLOCK_SLOTS`, `NUM_NODES`,
`NUM_OPS`, `NUM_WAVES`, anything indexed by `kernel_tag` first.

### Build flags

- `--features cuda` (no `ferrite` for `vllm-tk-macros-core` — that's a different crate)
- The cudaforge cache is shared across worktrees. If a build picks up stale
  `.a` files, `rm` them under `~/.cudaforge/git/checkouts/`.

### CUDA 12.9 toolchain

```bash
PATH=/usr/local/cuda/bin:$PATH  # 12.9.86 — already in nvcc default
```

The fused/ baseline kernel takes minutes to compile via `cicc` — don't kill it.

## How to commit

```bash
cargo fmt -p vllm-tk-macros-core
cargo clippy -p vllm-tk-macros-core -- -D warnings
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
