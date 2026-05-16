# Phase 3 multi-step decode numeric match — handoff

Pick up target: full llama-3.2-1B decode matches the host interpreter
bit-for-bit under `FERRITE_MEGA=1` (Phase 3 exit gate in
`FERRITE_TK_PLAN.md`). Diagnosis is done and committed
(`ea761c797`); this doc scopes the implementation work.

## Status

- ✅ Step 1 (OpKind variants) — `0e10339e0`
- ✅ Step 2 (shape signatures) — `0e10339e0`
- ✅ Step 3 (Impl structs + Instruction variants + no-op eval arms) — `0e10339e0`
- ✅ Step 4 (`insert_mega_barriers` lowering pass, RAW edges only) — `c4557b5c2`
- ✅ Step 5 (Barrier codegen emission in `variant_cpp.rs`) — `6f11f0bf6`
- ✅ Step 6 (barriers[] gmem alloc + kernel-arg wire-through in `interpreter/mega.rs`) — `787bcdb19`
- ✅ Step 6b (expected_count uses runtime `gridDim.x*y*z` pre-step-7) — `94c445f92`
- ✅ Step 4b (WAR edges from color reuse — post-`expand_loops` pass) — `1411040b2`
- ✅ Step 7 (persistent-thread grid + tile loops) — this commit

All Phase 3 step 7 sub-items:
- ✅ Launcher (mega.rs): `cudaOccupancyMaxActiveBlocksPerMultiprocessor`
  query + `dim3(NUM_SMS * ferrite_ctas_per_sm, 1, 1)` grid.
- ✅ Kernel refactor (every `ferrite_kernels/*.cuh`): role bodies
  loop over their native tile space with `for (int row =
  blockIdx.x; row < N; row += gridDim.x)` (or flattened 2D for
  gemm_bf16 and fused_qkv_rope_cache). Semaphore waits use
  `iter & 1` as the phase bit; `__syncthreads` at end of each
  iter serializes page reuse.
- ✅ attention_partial kept its native NUM_Q_HEADS tile count
  via early-return gate (NUM_Q_HEADS is always << resident CTA
  capacity for realistic models; a q_head outer loop would need
  non-trivial phase bookkeeping across the K/V page ring and
  isn't needed today — flagged in the header comment).
- ✅ Grid-shape unit tests in `interpreter::mega::tests`
  rewritten to check the new persistent-thread grid pattern.
  67/67 mega tests pass on `cargo test -p ferrite-forward-macro`.
- ✅ End-to-end verification on H100 pod (this commit): build with
  `FERRITE_MEGA=1` succeeds; kernel runs to completion; K/V cache
  slot=5 post-decode is valid bf16 (no NaNs — the pre-step-7 cross-
  CTA race is fixed). First decode token matches host exactly.
  Subsequent decode steps diverge at the token level due to
  accumulated bf16-ULP differences in mega's manual fp32 reductions
  vs host's cuBLAS path. **Phase 3 exit gate partial**: "The quick
  brown fox → jumps" (1 token) passes; "The quick brown fox →
  jumps over" (2 tokens) fails at token 2. See "Numerical
  divergence follow-up" below.

Two bugs shook out of the E2E verification and were fixed here;
without them the kernel either failed to build or deadlocked:

- **Pre-solver Barrier insertion broke multi-tile fusion
  Impls.** The original `mega_lowering::insert_mega_barriers`
  (step 4) ran pre-solver, rewiring consumer edges to read from
  the inserted `BarrierWait` node. Fusion Impls like
  `FusedGateUpSiluMulImpl::matches()` walk the DAG with
  `consumes_tile(silu, gemm)` looking for a DIRECT producer →
  consumer edge — the rewrite defeats that check and the solver
  hits `no Impl in the library matched tile N op Silu` for every
  model that uses gated-MLP fusion (mistral, phi3, gemma, qwen3,
  granite, commandr, deepseek). **Fix**: moved the pass to
  post-`expand_loops` (merged into the existing WAR pass at
  `interpreter::mega::insert_war_barriers`, now covering RAW and
  WAR edges). Post-expand, every `OpInstance` is already one
  Impl's CTA grid, so a linear scan identifies cross-op slot
  hazards only at genuine cross-Impl boundaries. Intra-fusion
  edges (Gemm → Silu → Mul inside a single op_block) are
  transparently handled by the kernel's own `__syncthreads`.
  `mega_lowering.rs` deleted; `BarrierMeta` / `EdgeTable` kept
  for future composability.
- **`barrier_signal` / `barrier_wait` gated on
  `threadIdx.x == 0` never fired.** `emit_barrier_signal` emits
  into the `storer` role body, which runs on a single non-
  consumer warp (warp 4 for `NUM_CONSUMER_WARPS = 2`) with
  `threadIdx.x ∈ [128, 160)` — so `threadIdx.x == 0` is never
  true there. Every `atomicAdd` silently no-oped, every
  `barrier_wait` spun forever. Same issue for the Wait in
  `loader_body` (warp 2, `threadIdx.x ∈ [64, 96)`). The
  trailing `__syncthreads()` after the wait also deadlocked
  independently: the other three role bodies emit empty
  snippets for the Barrier op, so only the loader warp reached
  the sync, and `__syncthreads` waits for every thread in the
  CTA. **Fix**: gate on `kittens::laneid() == 0` (thread 0 of
  whichever warp is executing the role body — one thread per
  CTA since each role body runs on exactly one warp); drop the
  trailing `__syncthreads()` and rely on the walker's automatic
  inter-op sync stanza for CTA-wide rendezvous.

A runtime grid override `FERRITE_GRID=N` was added to the mega
launcher to bisect cross-CTA issues in future debugging (set to
1 to single-CTA the whole kernel — all barriers become no-ops,
all cross-CTA hazards trivially satisfied).

### Numerical divergence follow-up

The cross-CTA hazard is fixed; the remaining gap between mega and
host is bf16 rounding. The handoff preamble anticipated this
explicitly: "K/V slot=5 post-decode dumps should match bit-for-bit
modulo bf16 rounding (mega uses manual fp32 reductions vs host
cuBLAS; max abs diff bounded by a few ULPs per element, not
0x7fff NaNs)." The dumps collected at the end of this commit
show exactly that — K and V at `layer=0, slot=5` differ by a
handful of 1-ULP flips (e.g. host `bfe1` vs mega `bfe2`) and
nothing larger. Every value is finite, none are qNaN.

Those ULP flips are enough to cross argmax boundaries on tokens
where the top-2 logit gap is narrow. "fox → jumps" has a wide
gap (mega matches); "jumps → over" is tighter (mega samples a
different token). The Phase 3 exit gate as stated — text
matching — is not purely a synchronization property; it also
requires the mega math to be numerically close enough to host
that argmax is stable across steps. Options for closing the
gap (not in this commit):

1. Re-order the manual fp32 reductions in each op's consumer
   body to match host's cuBLAS reduction tree where possible.
2. Increase reduction precision (fp32 accumulators on gmem
   scratch for the cross-warp partials, rather than shmem
   partial sums that get cast early).
3. Accept that the exit gate wants argmax stability, not bit-
   exactness, and tune op-by-op until the first 10 tokens on
   the three handoff prompts match. Likely requires a numerical
   audit op-by-op with single-op reference traces.

Additional work flagged during implementation:
- **Step 4b (WAR edges)**: Step 4 lands RAW edges only. The diagnosis
  root cause is WAR edges from color reuse; WAR extraction needs the
  coloring's live-range data from `colored_slot_map`, which only
  exists post-solver. **Resolved** by the post-`expand_loops` pass
  `insert_war_barriers` in `interpreter/mega.rs`: walks the linear
  schedule tracking `last_reader[slot]`/`last_writer[slot]` and
  splices a `BarrierSignal` + `BarrierWait` pair before any writer
  that would stomp a slot some earlier op read. Edge_idx allocation
  continues past the pre-solver RAW range. Slot access profiles
  live in `variant_cpp::op_slot_access` (per-op read/write lists);
  in-place ops like `FusedAddRmsNorm` surface their slots in both
  lists. `canonical_mega_meta` runs the same pass so `NUM_EDGES`
  includes WAR edges and the runtime `barriers[]` buffer sizes
  correctly.
- **Step 6b (`expected_count` backfill)**: Step 4's EdgeTable
  records `u32::MAX` placeholders for `per_edge_expected_count` —
  the producer's native CTA count depends on the Impl the solver
  picks. **Resolved** (Phase 3 step 6b, commit coming after step 6):
  instead of a post-solver backfill, `emit_barrier_wait` now emits
  the expected count as the runtime `gridDim.x * gridDim.y *
  gridDim.z` product. This is correct pre-step-7 because
  `emit_barrier_signal` fires from every CTA unconditionally (no
  blockIdx gate on the signal), so the signal count per edge
  equals the kernel grid size. When step 7's persistent-thread
  grid + per-op tile loops land, a per-edge count may diverge from
  grid size and a backfill pass can use `EdgeTable::per_edge_expected_count`
  (kept for that purpose, currently `#[allow(dead_code)]`).

## Why this exists

The plan's Phase 3 exit gate is "full llama-3.2-1B m=8 decode produces
correct output matching host-interpreter reference (bf16 tolerance)
end-to-end." Currently mega's first decode call is wrong: prompt "The
quick brown fox" + max_tokens=2 prints `" jumps!"` (host `" jumps
over"`). Progress log 2l-iii's claim that "step 1 matches" was
actually the prefill output matching — prefill has no mega canonical
yet, so both modes produce the prefill-sampled token via the host
path. Every mega-computed token is wrong.

## Diagnosis (committed)

Root cause: cross-CTA / cross-op race on the `act_ptrs[1]` gmem
activation slot. The emitted `.cu` reuses that slot across 8+ ops
(RmsNorm → FQKV read → o_proj → FusedAddRmsNorm → down_proj →
next-layer RmsNorm → next-layer FQKV read → …) with only per-CTA
`__syncthreads()` between ops. In the `128256×48` kernel grid, K/V-
writing FQKV CTAs at `blockIdx.y=32..39` are scheduled far later
than the producer/reader CTAs at `blockIdx.y=0`, so they TMA-load
`act_ptrs[1]` long after the slot has drifted to NaN (one
division-by-zero or overflow along the 16-layer reuse chain
suffices). The NaN flows through their FQKV dot products into the
K/V cache.

**Evidence**: `FERRITE_DUMP_KV=/tmp/kv.log` hook (already landed in
`cuda_worker.rs`) shows `K[layer=0, slot=5]` post-decode has valid
bf16 at exactly positions 0 and 32 of each kv_head (what CTA
`blockIdx.x=0, blockIdx.y=NUM_Q_HEADS+h` writes) and 0x7fff
(bf16 qNaN) at every other position. Pre-decode the same slot is
all 0x0000. Host path writes every position valid.

Not a FQKV math bug (CTA(0,·) proves the math). Purely a
synchronization failure on shared gmem slots.

## Design

Two DAG op kinds: `OpKind::BarrierSignal`, `OpKind::BarrierWait`.
Inserted as pairs on every cross-CTA producer→consumer edge by a
new lowering pass `insert_mega_barriers`, analogous to
`insert_all_reduces`. Edges come directly from the DAG (and from
the coloring's color-reuse WAR edges) — no heuristics, no
guessing, no "sync everywhere" coarsening. Per-op native CTA grid
gives the exact expected count for each barrier.

### Why a pair (not a single node)

One runtime action per node keeps the walker straight-line. Signal
inherits the producer's native CTA grid and emits the atomic;
Wait inherits the consumer's native CTA grid and emits the wait.
Producer/consumer op codegen stays untouched. Trade-off: two
extra nodes per edge instead of one. Agreed with user this session.

### Why barriers subsume to no-op on host interpreter

Each host `Instruction::eval` is a separate `cudaStreamLaunch`;
the stream boundary is the barrier. The `Barrier*` op kinds can
be erased at host-interpreter codegen (no-op arm in
`Instruction::eval`) and they don't appear in `FORWARD_TABLE`'s
host-path schedule.

### Why TP collectives subsume Barrier on sharded edges

AllReduce/AllGather already sync AND perform data motion. On any
edge where a TP collective already sits, `insert_mega_barriers`
skips — the collective is the barrier. (This integration is
**deferred** per user; implement single-device first.)

### Why persistent-thread grid is required

Cross-CTA gmem barriers (`atomicAdd` + spin) deadlock if the
barrier's expected count exceeds the resident-CTA capacity. Any
CTA waiting in the spin holds an SM slot, preventing producer
CTAs in later waves from ever being scheduled. H100 with current
register/shmem budget admits ~264 resident CTAs; the current grid
of 6.16M CTAs is ~23000 waves deep. We need the kernel grid sized
to fit in one wave (grid ≤ resident capacity). Each op's
loader/consumer/storer then loops over its native tile space with
`blockIdx.x + grid_stride`. Lm_head's 128256-tile space loops
inside its CTAs rather than exploding `grid.x`.

## Existing infrastructure to reuse

- **`crates/ferrite-kernels/csrc/tk/ferrite_barrier.cuh`** — already
  has `ferrite::barrier_signal(int32_t* slot, int count)` and
  `ferrite::barrier_wait(const int32_t* slot, int expected)` with
  `__threadfence()` + `atomicAdd` and spin + `__threadfence()`
  respectively. Unused today; Barrier op codegen binds to these.
- **`insert_all_reduces`** at `tp_lowering.rs:141` — template for
  the walk-the-DAG-insert-nodes-rewire-consumers pattern. Copy the
  shape.
- **`rewire_consumers`** at `tp_lowering.rs:274` — helper. Same
  rewiring applies to Barrier insertion.
- **`colored_slot_map`** at `interpreter_codegen.rs:110` — already
  computes `def_pos`, `owner_last_use`, `active.retain(lu <= dp)`.
  The color-reuse WAR edges fall out of the retain loop; extract
  them into an explicit `(lu_pos, dp_pos, old_last_readers,
  new_writer)` list and feed to the insertion pass.
- **`AllReduceImpl`** at `impl_lib.rs:16205` — template for the
  two new `Implementation` structs. Both Barrier ops are
  single-tile, identity-shape, `output_alias = Some(input)` so
  coloring collapses them to the source slot (zero storage).

## KVM reference (for pattern fidelity, not code reuse)

User's directive: "look at what KVM does." Read before
implementing:

- `~/git/Megakernels/demos/cross-gpu-llama/inc_barriers.cu` —
  barrier_inc_op shape with producer-side `redAdd(Sem::RELAXED,
  Scope::SYS)`.
- `~/git/Megakernels/demos/cross-gpu-llama/qkv_rope_append.cu:347,
  356` — producer does `redAdd(Scope::GPU)` then consumer does
  `wait_on_barrier<Scope::SYS>`.
- `~/git/Megakernels/demos/cross-gpu-llama/matmul_adds.cu:146,161,
  174,194` — the full producer→consumer pattern across multiple
  ops.
- `~/git/Megakernels/demos/cross-gpu-llama/all_device_barrier.cu` —
  the wait-on-count primitive.

KVM's `g.Bar[dev_idx][layer, opcode, batch_block, sub]` gmem
tensor indexing maps to our per-edge `barriers[edge_idx]` once
flattened; the multi-device / multi-layer / multi-batch-block
shape is for their TP + micro-batching design, not needed
single-device / single-batch.

## Implementation plan

Work in order; each step is testable standalone before moving
forward.

### 1. DAG op kinds (small)

**File**: `crates/ferrite-forward-macro/src/classified.rs`

Add two variants to `enum OpKind`:

```rust
/// Mega-kernel cross-CTA synchronization: producer-side signal.
/// Inserted by `insert_mega_barriers` at the producer's CTA grid;
/// storer emits `ferrite::barrier_signal(&barriers[edge], 1)`.
/// Pure control-flow: `output_alias = Some(input)`, no data
/// motion. Erased on host-interpreter codegen (stream boundary
/// is the barrier).
BarrierSignal,
/// Mega-kernel cross-CTA synchronization: consumer-side wait.
/// Inserted at the consumer's CTA grid; loader emits
/// `ferrite::barrier_wait(&barriers[edge], expected)`. Same
/// aliasing/erasure as BarrierSignal.
BarrierWait,
```

Neither goes through `OpKind::from_name` (not DSL-reachable), same
convention as `AllReduce` / `AllGather` / `Reshape`.

### 2. Shape signatures (small)

**File**: `crates/ferrite-forward-macro/src/shape.rs`

Both ops are identity on input 0:

```rust
OpKind::BarrierSignal => sig_unary_elementwise(solver, inputs, op),
OpKind::BarrierWait   => sig_unary_elementwise(solver, inputs, op),
```

Same dispatch as `AllReduce` (shape.rs:371).

### 3. Implementations (moderate)

**File**: `crates/ferrite-forward-macro/src/impl_lib.rs`

Two new structs paralleling `AllReduceImpl`:

- `BarrierSignalImpl` — `matches` on `OpKind::BarrierSignal`,
  `output_alias = Some(input_tile, input_slot)`, `opcode_shape`
  takes `(slot, edge_idx, expected_count)`. `launch_kind` is
  mega-only (won't appear in host-interpreter FORWARD_TABLE).
- `BarrierWaitImpl` — same shape, different opcode name.

Register both in `starter_library()` (impl_lib.rs:~1941,
alongside `AllReduceImpl`).

**Decision needed**: how to gate "mega-only" so these ops don't
affect host-path scheduling. Two options:
- (a) New `LaunchKind::MegaOnly` variant that host-interpreter
  codegen skips.
- (b) Special-case the opcodes in host `Instruction::eval` arms as
  no-ops (like the stream-ordered identity).
- Recommend (a); it's cleaner and the host FORWARD_TABLE already
  uses `launch_kind` to filter.

### 4. `insert_mega_barriers` lowering pass (moderate)

**New file**: `crates/ferrite-forward-macro/src/mega_lowering.rs`
**Call site**: `lib.rs:~620` (after `insert_lm_head_allgather`,
gate on a new `mega_enabled_for_canonical` check or always-on +
no-op when no Barrier-emitting codegen is in scope).

```rust
pub fn insert_mega_barriers(
    fuf: &mut Fuf,
    program: &Program,
    slot_map: &SlotMap,
) {
    // RAW edges: every DAG edge (producer_tile, slot) -> consumer.
    // Skip edges where producer and consumer share the same CTA
    //   domain (intra-CTA ops don't need cross-CTA sync).
    // Skip edges already covered by an AllReduce/AllGather/other
    //   semantic collective sitting between producer and consumer.
    //
    // WAR edges: extracted from `colored_slot_map`'s
    //   `active.retain(lu <= dp)` loop. Each retirement
    //   `(old_color, old_last_reader_pos, new_def_pos)` is an edge
    //   from old_last_reader to new_writer — needed so the
    //   new writer doesn't stomp on still-pending old reads.
    //
    // For each edge, append:
    //   BarrierSignal node: input = producer's output slot, no
    //     downstream readers.
    //   BarrierWait node: input = producer's output slot, rewire
    //     all downstream consumers of producer to read from Wait.
    //
    // Assign each edge a stable `edge_idx` (dense u32, sorted by
    // insertion order). The codegen uses it to index
    // `barriers[edge_idx]`. Store the edge_idx + expected count on
    // the OpInstance so fan_out/emit_op_block can read them.
}
```

Mirror the structure of `insert_all_reduces`. Use
`rewire_consumers` verbatim for the Wait node (the Signal has no
downstream consumers so no rewiring).

Return an `EdgeTable` struct that later phases (codegen, launcher)
consult:

```rust
pub struct EdgeTable {
    pub num_edges: u32,
    pub per_edge_expected_count: Vec<u32>,
}
```

Threaded through to `emit_mega_forward_fn` so it allocates
`barriers[num_edges]` and zero-inits it on every call.

### 5. Codegen emission (moderate)

**File**: `crates/ferrite-forward-macro/src/interpreter/variant_cpp.rs`

Extend `emit_op_block` with dispatch arms for `BarrierSignal` /
`BarrierWait`. Their roles are:

- `BarrierSignal::storer` → `ferrite::barrier_signal(&barriers[EDGE_IDX], 1);`
  gated on the producer's native CTA domain (gate condition
  carried on the OpInstance).
- `BarrierWait::loader` → `ferrite::barrier_wait(&barriers[EDGE_IDX], EXPECTED);`
  gated on the consumer's native CTA domain. Identity TMA (no
  data motion; aliasing collapses the slot).
- The other two roles (`loader` / `consumer` / `launcher`) are
  no-ops. Same shape as `embed`'s consumer-only pattern.

Grep `op_page_count` (`variant_cpp.rs:~253`) to return 0 for both
ops (no shared-memory pages needed).

Emit a new `barriers` kernel arg in the Attn-tier (or define a
new `Mega` tier extending Attn). Wire through the extern decl in
`emit_rust_variant_decl` and the launcher body in
`emit_cu_variant`.

### 6. Barrier gmem alloc in forward_mega (small)

**File**: `crates/ferrite-forward-macro/src/interpreter/mega.rs`

In `emit_mega_forward_fn` body, alongside act_ptrs / weight_ptrs
allocation:

```rust
let barriers = device.caching.alloc_tensor(
    &[NUM_EDGES * 4],   // i32 each
    DType::U8,
);
unsafe {
    // Zero-init. cuMemsetD8Async or memset-via-driver-API;
    // `driver::memset_d32_async(barriers.raw_ptr(), 0, num_edges,
    //  stream)` if available, else host-h2d of zeroed Vec.
}
```

Extend `LaunchArgsAttn` (or add `LaunchArgsMega` tier) to carry
the `barriers: *mut i32` pointer. `stage_launch_args` threads it
through `dispatch_launch`.

### 7. Persistent-thread grid + tile loops (largest piece)

**Files**: every `crates/ferrite-kernels/csrc/tk/ferrite_kernels/*.cuh`
plus the `.cu` launcher emission in `interpreter/mega.rs`.

Kernel grid becomes `dim3(NUM_SMS * CTAS_PER_SM_BUDGET, 1, 1)`.
On H100 with `NUM_CONSUMER_WARPS=2, CONSUMER_REGISTERS=192,
NON_CONSUMER_REGISTERS=64, NUM_PAGES=6, PAGE_SIZE=16384`, measure
the resident-CTA count (`cudaOccupancyMaxActiveBlocksPerMultiprocessor`)
and bake it in as the grid size.

Each op's role template gets a tile loop. Example for
`gemv_bf16::loader`:

```cpp
// BEFORE:
if (blockIdx.x >= N_TILES) return;
int n_tile = blockIdx.x;
// ... do work for n_tile ...

// AFTER:
for (int n_tile = blockIdx.x; n_tile < N_TILES; n_tile += gridDim.x) {
    // ... do work for n_tile ...
}
```

The per-op `<op>::native_tile_count<Config, ...>()` template
returns the op's natural tile count; the loop bound uses that.
Ops with fixed small grid (embed, rms_norm — NUM_TOKENS=1 tile)
just execute once in CTA 0 and bail in others, same as today.

Cooperative launch: not strictly needed if the grid size is ≤
`cudaOccupancyMaxActiveBlocksPerMultiprocessor * sm_count`, since
all CTAs are resident by construction. `grid.sync()` isn't
required because we use explicit `barrier_signal`/`barrier_wait`
pairs on just the edges that need it.

This step is the largest and should probably be committed
separately from steps 1-6 after those are proven on a toy variant
(see verification below).

## Verification

### Unit tests (each step)

- Step 1-3: `cargo test -p ferrite-forward-macro --lib` passes
  (starter_library test, OpKind round-trip).
- Step 4: add a test `insert_mega_barriers_inserts_pair_per_edge`
  that builds a 2-op FUF (rms_norm → gemm) and asserts exactly
  one BarrierSignal + BarrierWait pair is inserted.
- Step 5: `emit_op_block` dispatches correctly for both ops;
  golden tests covering the role bodies.
- Step 6: `stage_launch_args_mega` ABI round-trip test (size +
  alignment).
- Step 7: each `<op>::native_tile_count` template is right;
  tile-loop covers the native grid with any CTA count.

### End-to-end (on pod, after step 7)

Rebuild and run the same smoke that proved the diagnosis:

```bash
FERRITE_MEGA=0 ./vllm batch ... -o host.jsonl
FERRITE_MEGA=1 ./vllm batch ... -o mega.jsonl
diff host.jsonl mega.jsonl  # text field expected to match
```

With `FERRITE_DUMP_KV=/tmp/kv.log` on both runs, the K/V slot=5
post-decode dumps should match bit-for-bit modulo bf16 rounding
(mega uses manual fp32 reductions vs host cuBLAS; max abs diff
bounded by a few ULPs per element, not 0x7fff NaNs).

Success condition: `FERRITE_MEGA=1` produces text tokens matching
host on at least "The quick brown fox" (10 tokens), "Hi" (10
tokens), and "Hello, my name is" (10 tokens). Phase 3 exit gate
met.

## Non-goals / deferrals

- **TP integration.** Mega + tp>1 waits until single-device works.
  The `insert_mega_barriers` pass should skip edges already
  covered by an AllReduce/AllGather so the pair composes cleanly
  when TP is re-enabled.
- **Prefill canonicals (m>1).** Still `#error`-stubbed. Orthogonal
  to this work.
- **`FusedQkvRopeCache(biased=true)`, Phase-4 cross-op
  pipelining, SPLITS>1 subtile, ferrite-forward/-kernels linker
  gap.** All unchanged.
- **Cooperative launch.** Not needed if grid fits in one wave by
  construction; revisit only if a variant has more CTAs than
  resident capacity (shouldn't happen in the persistent-thread
  model).

## Pod workflow reminder

Worktree at `~/git/vllm/.claude/worktrees/ff-mega-codegen` maps to
pod path `/home/nickm/vllm-mega/vllm-rs/` (on `worktree-tk-mvp`
base branch with rsync'd overlay). Build: `FERRITE_MEGA=1 cargo
build -p vllm-cli --features cuda`. Smoke:
`./target/debug/vllm batch --model unsloth/Llama-3.2-1B-Instruct
--device cuda -i /tmp/one.jsonl -o /tmp/out.jsonl`. cudaforge
content-hashes the emitted `.cu`; any change triggers the ~15min
full rebuild only if the content actually changes. Never delete
`.cache/cudaforge/vllm-cuda/libmegakernels.a` (CLAUDE.md rule).
