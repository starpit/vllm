# Handoff — ff-subtile worktree, post-iter-3 audit + split_oversized_loads_pass phase 1

## Where to work

- **Worktree:** `/Users/nickm/git/vllm/.claude/worktrees/ff-subtile`
- **Branch:** `worktree-ff-subtile`
- **Crate:** `vllm-rs/crates/ferrite-wavefront/`
- **HEAD:** `157a3ea461` — split_oversized_loads_pass phase 1 (detect + diagnostic)
- **Pod path:** `/home/nickm/vllm-ff-subtile/vllm-rs/` on `nick`

`pwd && git branch --show-current` first thing — per `memory/feedback_handoff_worktree_match.md`. The worktree branched from `feat/rust`; commits stay on `worktree-ff-subtile`.

## What this session did

The user asked for "sync to pod and try the launch." That surfaced a real lowering correctness bug: `emit_external_load` was attempting to TMA-load full-K weight regions (8 MB for Llama-3.2-1B's 2048×2048 projection weights, 525 MB for lm_head) into single 32 KB pages. Pre-audit, the kernel emitted these huge loads silently; runtime would have OOB'd into adjacent pages.

Before that surfaced, the audit-fix-audit loop the user kicked off ran for three full iterations + a fourth in flight. **46 of 48 confirmed compile-time-safety findings closed.**

| Commits this session (newest → oldest) | Cluster |
|---|---|
| `157a3ea461` | split_oversized_loads_pass phase 1 — detect + diagnostic |
| `98dab8db7d` | validate_tk_tape pair-check for TmaExpect/LoadAsync tile drift |
| `d69bdbea5f` | iter-3 batch (exhaustive page-instr match, runtime-shape PAGE_SIZE asserts (later removed by 157a3ea461), sv-arena cap, external-load triple — the latter reverted post-plan-re-read) |
| `676932d739` | TmaExpect.kind sealed (REAL BUG: was arming page_done while load arrived on page_ready) + tile_type_spec swizzle parity |
| `b43a80d5b1` | SubstratePool enum + SUBSTRATE_SWIZZLE_BYTES + static-smem accounting |
| `7bcfb334f2` | seal LoadSpec/StoreSpec fields, FenceTag/AccTag enums, TmaExpect.tile |
| `97da162805` | dim-role typed witnesses (M64/M128/K128/N128/MPerWarp32 + bridge MDimFor/KDimFor/NDimFor) for WgmmaShape |
| `13c5f1ea42` | sealed WgmmaShape witness |
| `0d0c0f0dcd` | sealed ArrivalCount enum + PageBarrier-driven semaphore emit |
| `40a92f70c3` | seal SmemTileId::from_page (PageSubTileShape/ActSubTileShape sealed markers) + TileShape boundary harden |
| `63722522cb` | substrate / DYN_SMEM compile-time witness chain (PageTileSpec, ActTileSpec, byte_size, st_type_literal helper, HOPPER_MAX_DYN_SMEM_BYTES const_assert; NUM_PAGES=5/NUM_ACT_PAGES=4 to fit Hopper) |
| `1f7771146a` | page_coalesce_pass — second §6.5 optimizer pass |
| `70e4bdda7b` | phase A step 8 — pod nvcc compile passes (sm_90a) |

102→104 lib tests pass at every commit boundary.

## State right now

The conservative lowering produces a `Instr::LoadAsync` with `tile.byte_size() = 8388608` for Llama-3.2-1B's q_proj/k_proj/v_proj weights. The new `split_oversized_loads_pass` (phase 1) detects this at codegen time and panics with full diagnostic:

```
split_oversized_loads_pass [phase 1 detect-only]:
Instr::LoadAsync at instr-index 16 has tile 2048×2048×2B = 8388608 bytes,
exceeds PAGE_SIZE = 32768 bytes. dst_page=1, barrier_page=1, src_arg=3.
```

The .cu cannot be regenerated until phase 2 (the actual rewrite) lands. **Phase 2 is the blocker for "actually launch the kernel on Hopper."**

The substrate itself is sound:
- 5 page_buf + 4 act_buf + sealed semaphore arrays = 224 KB ≤ Hopper's 228 KB (verified by const_assert at compile time)
- 102→104 lib tests + 22 doctests pass
- 46/48 audit findings closed; 2 deferred per `feedback_no_speculative_witnesses`
- The previous "phase A step 8 — pod nvcc compile passes" milestone (`70e4bdda7b`) is invalidated by the iter-3 PAGE_SIZE / TmaExpect-barrier-kind / tile_type_spec-swizzle fixes — the .cu it produced was syntactically clean but would have OOB'd at runtime. The fixes since then make the .cu actually correct *once phase 2 lands*.

## Phase 2: the actual rewrite

This is the work to unblock the launch. ~400 LOC + tests. Mirror `passes/page_coalesce.rs` shape.

### Algorithm

For each `Instr::LoadAsync(spec)` where `spec.tile.byte_size() > PAGE_SIZE`:

1. **Walk forward to identify the consuming MatmulTile-emit-sequence.** The lowering's MatmulTile arm emits this fixed shape:
   ```
   init_rt_zero(rt_d)
   load_shmem_to_reg_warpgroup(a_page → rt_a)
   wgmma_fence_acc(rt_d)
   wgmma_mma_ab_reg_smem(rt_d, rt_a, b_page, FenceExternal, AccReset, W4)
   wgmma_async_wait(0)
   store_reg_tile_to_shmem_warpgroup(rt_d → dst_page)
   ```
   The big LoadAsync's `dst_page` is either `a_page` or `b_page` for the next MatmulTile-sequence in tape order. Walk forward to find that sequence.

2. **Find the matching second LoadAsync.** Both A and B operands are likely big; both need K-fragmenting in lockstep. The other operand's LoadAsync is the next LoadAsync in tape order before the matmul sequence (the lowering emits `resolve_input_page` for A then B before the matmul).

3. **Compute K-block count.** A is `M × K_full`; B is `K_full × N`. K_block = TK 2.0 WGMMA bf16 K (= 128). K_blocks = `K_full / 128`. Both A.cols and B.rows agree on K_full.

4. **Compute per-K-block byte strides.** Both LoadSpecs have `byte_off: ByteOffsetExpr`; for the rewrite, replace each with `ByteOffsetExpr::LinearLoop { var, stride_bytes, base }` where:
   - A's K-stride: K-chunk along cols of an M×K_full row-major tile = `K_block × elem_bytes` (= `128 × 2 = 256` for bf16).
   - B's K-stride: K-chunk along rows of a K_full×N row-major tile = `K_block × N × elem_bytes` (= `128 × N × 2`).
   - `base` is the LoadSpec's existing `byte_off` (preserved as the iter-0 base).
   - `var` is a fresh `LoopVarId` minted at the rewrite site (use `tape.alloc_loop_var()` or equivalent).

5. **Splice in the rewrite.** Replace the `[LoadAsync_A, LoadAsync_B, ...MatmulTile-emit-sequence]` span with:
   ```
   init_rt_zero(rt_d, ...)                                    // unchanged
   BarrierInit{page_id: A_page, kind: Ready, count: One}      // NEW
   BarrierInit{page_id: B_page, kind: Ready, count: One}      // NEW
   ForLoopOpenConst { var, n: K_blocks }                      // NEW
     TmaExpect{barrier_page: A_page, kind: Ready, tile: A_chunk}  // NEW per iter
     LoadAsync{spec: A_chunk_spec with LinearLoop byte_off}   // REPLACES big LoadAsync_A
     TmaExpect{barrier_page: B_page, kind: Ready, tile: B_chunk}  // NEW per iter
     LoadAsync{spec: B_chunk_spec with LinearLoop byte_off}   // REPLACES big LoadAsync_B
     PageBarrierWaitLoopStart0{page_id: A_page, kind: Ready, role}  // NEW
     PageBarrierWaitLoopStart0{page_id: B_page, kind: Ready, role}  // NEW
     load_shmem_to_reg_warpgroup(a_page → rt_a)              // unchanged
     wgmma_fence_acc(rt_d)                                   // unchanged
     wgmma_mma_ab_reg_smem(rt_d, rt_a, b_page, FenceExternal, AccAccumulate, W4)  // FROM AccReset
     PageBarrierArrive{page_id: A_page, kind: Done, role}    // NEW: signal A consumed
     PageBarrierArrive{page_id: B_page, kind: Done, role}    // NEW: signal B consumed
   ForLoopClose { var }                                      // NEW
   wgmma_async_wait(0, W4)                                   // unchanged
   store_reg_tile_to_shmem_warpgroup(rt_d, dst_page, ...)    // unchanged
   ```

   `AccAccumulate` always (init_rt_zero outside the loop makes iter 0 equivalent to `AccReset + accumulate-into-zero`).

### Subtleties to watch

- **Page reuse with parity.** Each iteration overwrites A_page and B_page. The conservative pattern uses `BarrierWaitLoopStart0/Start1` for parity-alternating waits. Mirror what the AttnDecode K-loop does (see `lower_subtile_tape_to_tk_tape::SubOp::AttnDecode` arm around line 1394–1500) — it has the same producer/consumer cycle on K and V tile pages.

- **TmaExpect needs to be re-armed per iteration**, not just once before the loop. expect_bytes is consumed by the load. The pattern: TmaExpect → LoadAsync (drains transaction-bytes), then on the next iter the barrier is re-init'd / the next TmaExpect arms again.

- **The lowering of MatmulTile already does this for AttnDecode K-loop.** Look at how `lower_subtile_tape_to_tk_tape.rs:~1394–1500` does the K-loop body for AttnDecode — that's the closest existing pattern for a per-iter (TmaExpect, LoadAsync, Wait, Compute, Arrive) cycle. The K-tile pass for Gemm should produce structurally the same shape.

- **`ByteOffsetExpr::LinearLoop` field-construction.** The const-generic `linear_loop::<STRIDE>(var, ByteStride::NEW, base)` constructor requires STRIDE as a const. In the pass, stride is computed at tape-runtime (depends on `spec.tile.cols`). Use field-literal construction: `ByteOffsetExpr::LinearLoop { var, stride_bytes, base }`. Fields are `pub(crate)` so the pass module can construct directly.

- **The pass postcondition** added to `validate_tk_tape`: `for every Instr::LoadAsync, spec.tile.byte_size() <= PAGE_SIZE`. Add the new validation arm and a matching `TkValidationError` variant.

- **Test coverage.** Mirror `passes/page_coalesce.rs` test shape — synthetic tape with one big LoadAsync + matmul-sequence; assert post-pass shape includes `ForLoopOpenConst`, K_blocks-many small LoadAsyncs, etc.

### What NOT to do

- **Don't push K-tiling into SubtileIR.** The user explicitly rejected that direction (3× restated "isn't this a TkTape→TkTape transformation?"). The plan §0 says SubtileIR is target-agnostic; K=128 is a TK 2.0 / Hopper specific number.

- **Don't pattern-match too aggressively.** The pass works on a CLEAR pattern: big LoadAsync followed (in tape order, after possibly some intervening Instrs from the second External resolve) by a MatmulTile-emit-sequence. If the pattern doesn't match, leave the LoadAsync alone (the validator postcondition will fail and codegen-time-error). Don't try to handle non-matmul big loads in this pass.

- **Don't add a `debug_assert!` PAGE_SIZE check anywhere.** Per `feedback_compile_time_or_garbage`, the pass IS the gate. The construction-time asserts in `LoadSpec::new_runtime_shape`/`StoreSpec::new_runtime_shape` were removed in `157a3ea461` for exactly this reason — they pre-empted the pass.

## Pod paths + commands

- Sync: `oc --context nickm/api-fmaas-vllm-d-fmaas-res-ibm-com:6443/nickm@us.ibm.com rsync /Users/nickm/git/vllm/.claude/worktrees/ff-subtile/vllm-rs/crates/ferrite-wavefront/src/ nick:/home/nickm/vllm-ff-subtile/vllm-rs/crates/ferrite-wavefront/src/`
- Force regen on pod: `oc --context ... rsh nick bash -c 'cd /home/nickm/vllm-ff-subtile/vllm-rs && cargo clean -p ferrite-forward-macro -p ferrite-model-llama && FERRITE_WAVEFRONT=1 FERRITE_MODELS=llama-3.2-1b cargo build -p ferrite-model-llama --features cuda'`
- Inspect emitted .cu: `~/.cache/cudaforge/megakernels/tk_decode_full_llama_3_2_1b.cu`
- nvcc compile: `nvcc -gencode=arch=compute_90a,code=sm_90a -std=c++20 -O3 --use_fast_math --expt-extended-lambda --expt-relaxed-constexpr -DNDEBUG -DKITTENS_HOPPER -Xcompiler=-fPIC -Xcompiler=-fno-strict-aliasing -I third_party/thunderkittens/include -c ~/.cache/cudaforge/megakernels/tk_decode_full_llama_3_2_1b.cu -o /tmp/mk.o`

## Iter-4 audit (still in flight when this handoff was written)

The 4th iteration of the audit-fix workflow returned 9 confirmed findings. Most are minor witness-coverage gaps in the same vein as iter-2/3 cleanups (TileTypeSpec.{rows,cols,dtype} fields are `pub` should be `pub(crate)`; `wgmma_async_wait(n: u32)` should be sealed to {Zero, One, Two}; etc.). None are blocking the launch path. Triage those AFTER phase 2 lands and the kernel actually runs.

The full iter-4 output is at `/private/tmp/claude-502/-Users-nickm-git-vllm--claude-worktrees-ff-subtile/779e6b2f-0507-4b8c-90c3-5424d5968068/tasks/w9oj6z4qv.output` if you want to review.

## Open tasks

- #71 split_oversized_loads_pass phase 2 (the actual rewrite) — THIS IS THE BLOCKER
- #53 RopeRotateInterleaved (Instr step, deferred)
- 2 deferred audit findings: subtile-cols-idx-flat-u16, S4 PageId pub(crate) inner

## What "done" looks like for the next session

1. Phase 2 lands (~400 LOC + tests).
2. `cargo test -p ferrite-wavefront --lib` green.
3. Pod regen produces a .cu.
4. nvcc + ptxas compile the .cu cleanly.
5. cudaFuncSetAttribute accepts the DYN_SMEM (it should: 224 KB ≤ 228 KB).
6. The kernel actually runs (vllm bench latency or similar) and produces coherent decode output for a Llama-3.2-1B prompt.

Step 6 is the original goal the audit-fix loop was supposed to unblock. After 14+ commits worth of typed-witness work, phase 2 is the last piece between the substrate and the launch.
