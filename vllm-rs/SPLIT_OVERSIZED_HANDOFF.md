# Handoff — ff-subtile worktree, post-phase-2 K-tile + N-tile gap

## Where to work

- **Worktree:** `/Users/nickm/git/vllm/.claude/worktrees/ff-subtile`
- **Branch:** `worktree-ff-subtile`
- **Crate:** `vllm-rs/crates/ferrite-wavefront/`
- **HEAD:** `36024bc11f` — split_oversized_loads_pass phase 2 (K-loop rewrite)
- **Pod:** `nick`, path `/home/nickm/vllm-ff-subtile/vllm-rs/`

`pwd && git branch --show-current` first thing — per
`memory/feedback_handoff_worktree_match.md`.

## What this session did

Phase 2 of `split_oversized_loads_pass` landed (`36024bc11f`,
~900 LOC + 6 tests). The pass takes the conservative lowering's
oversized External LoadAsyncs and rewrites them, paired with their
consumer `WgmmaMmaAB_RegSmem` matmul-emit-sequence, into a K-loop
over 128×128 chunks. Mirrors AttnDecode_Qkt's K-loop pattern.
Postcondition (defensive walk): every `Instr::LoadAsync.tile.byte_size()`
is `≤ PAGE_SIZE` after the pass.

110 lib tests pass on the H100 pod (104→110: +6 phase-2 tests).

## What this session uncovered

The prior handoff (`35c0cb7231`) said:

> N-tiling beyond a single 128-col page is a separate transform
> (not in this pass; M=N=128 is the only case the conservative
> lowering produces today).

That assumption was wrong about the production state. Driving
ferrite-model-llama's proc-macro on the pod (`FERRITE_WAVEFRONT=1
FERRITE_MODELS=llama-3.2-1b cargo build`) hits the pass's N>128
compile-time guard before any K-tile rewrite can fire:

```
[wavefront] llama-3.2-1b: 244 fuf tiles, 115 subgraphs → 181 sources
  (146 weights, 32 prefix-kv), 258 ops; ops [..., ("Gemm", 113), ...]
error: custom attribute panicked
  --> crates/ferrite-model-llama/src/lib.rs:14:1
  = help: message: split_oversized_loads_pass: B operand tile.cols = 2048,
                   expected 128. N-tiling beyond a single 128-col page is
                   a separate transform (not in this pass).
```

The conservative lowering for Llama-3.2-1B produces ONE `SubOp::MatmulTile`
per logical Linear (113 Gemms total = 7×16 + lm_head), with **full**
`TensorRegion`s on both sides:

| Operand | shape | bytes | status |
|---|---|---|---|
| A (act): | M × K_full = 128 × 2048 | 512 KB | oversized along K |
| B (weight, q_proj): | K_full × N = 2048 × 2048 | 8 MB | oversized along **both** K and N |
| Output: | M × N = 128 × 2048 | 512 KB | oversized along N |

`partition.rs::lower_partitioned` is NOT in the path that produces
the SubtileIR for Llama (the `[wavefront]` log shows 113 Gemm nodes,
not 113×n_blocks). One Gemm SubOp per Linear, full output.

**The K-tile machinery in phase 2 is correct and tested.** But
launching the kernel needs the M/N-tiling sibling pass to land
first, otherwise the N>128 guard panics before K-tile ever fires.

## Phase 3: N-tile (and M-tile) sibling pass

This is the next blocker for the kernel launch.

### What needs to happen architecturally

Each oversized Gemm needs to become an OUTER N-loop wrapping the
existing inner K-loop (which `split_oversized_loads_pass` already
emits). The output store needs to be folded into the N-loop body.

Sketch:

```text
BarrierInit a_page Ready One
BarrierInit b_page Ready One
ForLoopOpenConst { n_var, n: N_blocks }    ← N_full / 128
  init_rt_zero(rt_d)                       ← re-init per N tile
  ForLoopOpenConst { k_var, n: K_blocks }
    TmaExpect A_page                       ← A independent of n_var
    LoadAsync A chunk LinearLoop(k_var, stride_A_k)
    TmaExpect B_page
    LoadAsync B chunk LinearLoop(k_var, stride_B_k)  + LinearLoop(n_var, stride_B_n)
    PageBarrierWaitLoopStart0 A k_var
    PageBarrierWaitLoopStart0 B k_var
    load_shmem_to_reg(A_page, rt_a)
    wgmma_fence_acc(rt_d)
    wgmma_mma_ab_reg_smem(...AccAccumulate, FenceExternal)
    wgmma_async_wait(0)
  ForLoopClose { k_var }
  store_reg_tile_to_shmem_warpgroup(rt_d, dst_page)
  StoreAsync dst_page → output, byte_off = LinearLoop(n_var, stride_out_n)
ForLoopClose { n_var }
CommitGroupBulk
ThreadfenceDevice
PageBarrierArrive Done dst_page             ← arrive ONCE, not per N tile
```

### Issues that need design

1. **Two-variable byte offsets.** B's per-iter byte offset depends on
   BOTH `k_var` (stride = `K_BLOCK × N_full × 2` along rows of K×N
   row-major) AND `n_var` (stride = `N_BLOCK × 2` along cols).
   `ByteOffsetExpr::LinearLoop` carries one `var`/`stride_bytes`
   pair. **A new `ByteOffsetExpr::Affine2D { var_a, stride_a,
   var_b, stride_b, base }` variant** (or similar) is required, with
   matching player emit. Same applies to A (which only depends on
   `k_var`, but the variant takes both for symmetry).

2. **Output StoreAsync needs N-loop indexing.** Currently
   `emit_store_and_arrive` emits one `StoreAsync(dst → output)` with
   a `ByteOffsetExpr::Const`. Inside the N-loop, the StoreAsync's
   `byte_off` becomes `LinearLoop(n_var, stride_out_n)` and its
   tile shape becomes `128×128` (not `128×N_full`). Use
   `StoreSpec::new::<128, 128, Bf16>` on the typed-witness path
   instead of `new_runtime_shape`.

3. **CommitGroupBulk / ThreadfenceDevice / Arrive-Done placement.**
   These currently live AFTER the matmul-emit-sequence. With N
   stores per output, you want:
   - Inside the N-loop: nothing (the StoreAsyncs accumulate).
   - After the N-loop close: ONE CommitGroupBulk + ThreadfenceDevice
     + PageBarrierArrive Done.

4. **Init_rt_zero placement.** The K-tile pass currently emits
   `init_rt_zero` ONCE outside the loop. With N-tiling, it needs
   to be re-issued per N iter (each output tile gets a fresh
   accumulator).

5. **M-tile.** For workloads `[1024, 2048, 4096]` in the macro spec,
   M = workload exceeds PAGE_ROWS = 128. M-tiling adds a third
   outermost loop. Defer to a follow-up if not blocking — the smaller
   workloads (which compile to separate kernels per the
   `workloads = [1, 2, 4, 8, 64, 512, 1024, 2048, 4096]` spec) hit
   N-tile first.

### Where to land it

**Recommendation:** New sibling pass file
`crates/ferrite-wavefront/src/passes/n_tile_loads.rs` that runs
**before** `split_oversized_loads_pass` in the §6.5 pipeline (line
570 of `lower_subtile_tape_to_tk_tape.rs`):

```rust
crate::passes::n_tile_loads_pass(&mut out);     // NEW — runs first
validate_tk_tape(&out).expect("n_tile_loads_pass: invalid TkTape");

crate::passes::split_oversized_loads_pass(&mut out);  // existing K-tile
validate_tk_tape(&out).expect("split_oversized_loads_pass: invalid TkTape");
```

The N-tile pass produces `N_blocks` MatmulTile-emit-sequences each
with B region `K_full × 128` (still oversized along K → 512 KB), and
the K-tile pass then chunks each of those into 128×128.

Pass shape mirrors `split_oversized_loads.rs`:

- Walk the tape, find Gemm-emit-sequences whose B operand has
  `tile.cols > PAGE_COLS`.
- For each, splice the (LoadAsyncs + matmul-emit-sequence + StoreAsync
  + commit + fence + arrive) span with an N-loop wrapping a copy of
  the inner sequence per N iter.
- Same compile-time invariants (typed const-generic witnesses, sealed
  enums, where-clauses) per `feedback_ff_subtile_compile_time_inviolable`.

### Estimated scope

~600 LOC + tests. Complexity comes from:

- The `Affine2D` byte-offset variant and its player emit.
- The output StoreAsync rewrite (needed once for the byte_off, once
  for the tile shape, once for the typed `new` constructor).
- The CommitGroupBulk / Arrive-Done batching across N iters.

The K-tile sibling already proves the pattern (~900 LOC + 6 tests
on the same kind of synthetic tape harness). N-tile is structurally
similar but has the extra wrinkle of the output store.

## Pod paths + commands

Path: `/home/nickm/vllm-ff-subtile/vllm-rs/` on `nick`. Context:
`nickm/api-fmaas-vllm-d-fmaas-res-ibm-com:6443/nickm@us.ibm.com`.

Sync:

```
oc --context nickm/... rsync \
  /Users/nickm/git/vllm/.claude/worktrees/ff-subtile/vllm-rs/crates/ferrite-wavefront/src/ \
  nick:/home/nickm/vllm-ff-subtile/vllm-rs/crates/ferrite-wavefront/src/
```

Pod tests:

```
oc --context nickm/... rsh nick bash -c 'cd /home/nickm/vllm-ff-subtile/vllm-rs && \
  FERRITE_MODELS=llama-3.2-1b cargo test -p ferrite-wavefront --lib'
```

Force regen the .cu (when N-tile lands and the build no longer
panics):

```
oc --context nickm/... rsh nick bash -c 'cd /home/nickm/vllm-ff-subtile/vllm-rs && \
  cargo clean -p ferrite-forward-macro -p ferrite-model-llama && \
  FERRITE_WAVEFRONT=1 FERRITE_MODELS=llama-3.2-1b \
    cargo build -p ferrite-model-llama --features cuda'
```

Inspect emitted .cu:
`~/.cache/cudaforge/megakernels/tk_decode_full_llama_3_2_1b.cu`.

nvcc compile (on pod):

```
nvcc -gencode=arch=compute_90a,code=sm_90a -std=c++20 -O3 \
  --use_fast_math --expt-extended-lambda --expt-relaxed-constexpr \
  -DNDEBUG -DKITTENS_HOPPER -Xcompiler=-fPIC -Xcompiler=-fno-strict-aliasing \
  -I third_party/thunderkittens/include \
  -c ~/.cache/cudaforge/megakernels/tk_decode_full_llama_3_2_1b.cu \
  -o /tmp/mk.o
```

## Open tasks

- **Phase 3 — n_tile_loads_pass (the actual rewrite).** Blocker for
  the kernel launch.
- #71 split_oversized_loads_pass phase 2 — **DONE** at `36024bc11f`.
- #53 RopeRotateInterleaved (Instr step, deferred).
- 2 deferred audit findings: `subtile-cols-idx-flat-u16`, S4 PageId
  pub(crate) inner.

## What "done" looks like for the next session

1. `n_tile_loads_pass` lands (~600 LOC + tests). Same compile-time-
   safety bar (typed witnesses, sealed enums) as phase 2.
2. `cargo test -p ferrite-wavefront --lib` green.
3. `ByteOffsetExpr::Affine2D` (or chosen variant for two-var
   indexing) lands with the player emit.
4. `StoreSpec::new_loop_indexed_n` (or similar) emits the per-N-tile
   output store with typed `SmemTileSpec<128, 128, Bf16>` witness.
5. Pod regen produces a .cu with the K-then-N nested loops.
6. nvcc + ptxas compile cleanly.
7. cudaFuncSetAttribute accepts the DYN_SMEM (substrate is
   224 KB ≤ Hopper's 228 KB — verified by const_assert).
8. The kernel actually runs (`vllm bench latency` or similar) and
   produces coherent decode output for a Llama-3.2-1B prompt.

Steps 1-2 alone get the next session past the panic. Steps 3-8
unblock the original goal.
