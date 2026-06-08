# Handoff — ff-subtile, post-audit + Patches 3+4 + revert of nb=128 wedge

## Where to work

- **Worktree:** `/Users/nickm/git/vllm/.claude/worktrees/ff-subtile`
- **Branch:** `worktree-ff-subtile`
- **Crate:** `vllm-rs/crates/ferrite-wavefront/`
- **HEAD:** `b92b291bf0` — Patches 3+4 (validator arena-membership + rt_alias exhaustive coverage)
- **Pod:** `nick`, path `/home/nickm/vllm-ff-subtile/vllm-rs/`

`pwd && git branch --show-current` first thing.

## Session arc

| Commit | Summary |
|---|---|
| `36024bc11f` | Phase 2 — K-loop rewrite for oversized LoadAsyncs |
| `3edbefc62e` | Handoff: phase 2 done; production hits N>128 first |
| `f0fc5eae1b` | Phase 3 NK extension + `ByteOffsetExpr::Affine2D` |
| `ef640385d2` | Handoff: conservative lowering broken at every op boundary |
| `b92b291bf0` | **Audit Patches 3+4** — validator arena-membership + rt_alias exhaustive |

114 → 116 lib tests on the H100 pod.

## What the audit found (2026-06-08)

A multi-agent contract audit (115 agents, 21 confirmed findings out
of 34 candidates after 3-lens adversarial verification) identified
**three independent root causes**, not just the page-shape mismatch
the user spotted earlier:

### Group A — page-shape contract end-to-end (9 findings)

The lowering's compute arms in `lower_subtile_tape_to_tk_tape.rs`
hardcode `SmemTileId::<128, 128, Bf16>::from_page(...)` reads
regardless of what the `SubtileIR::TensorRegion` actually says. With
`nb = u32::MAX` at `codegen.rs:6093`, `lower_region` emits ONE node
per Linear with the FULL output region — `{1×2048}` for RmsNorm,
`{2048×2048}` for q_proj weight, etc. Concrete consequences:

- **#1 RmsNorm**: input region `{1×2048}` (4 KiB) loaded into a 32 KiB
  page; compute reads all 128×128 = 32 KiB; row_sum averages 1 valid
  + 127 garbage rows; `inv_cols` hardcoded to `1/128` (#19) instead
  of `1/hidden`. Variance 16× too large on Llama-3.2-1B.
- **#2 RopeRotate**: rotates 1 of 32 q-heads; remaining 31 heads
  carry stale page bytes verbatim through the chain.
- **#3 RopeAppend**: stores 32 KiB to a 1 KiB cache slot — 31 KiB OOB
  cache writes, cross-position KV corruption.
- **#5 MatmulTile**: oversized External LoadAsyncs (525 MiB for
  lm_head, 32 MiB for gate/up/down) — `cp.async.bulk` writes past
  the 32 KiB page slot into adjacent page_buf entries / mbarrier
  semaphores.
- **#8 StoreAsync smem-side OOB**: TK 2.0's
  `cp.async.bulk.global.shared::cta.bulk_group` performs no
  smem-side bounds check (PTX ISA 9.7.8.24); the `bytes` operand is
  whatever the producer puts on the IR. No `cudaErrorIllegalAddress`,
  no ptxas refusal — silent wrong gmem output (first 32 KB correct,
  remainder is whatever live shmem the read sweeps through, including
  mbarrier phase bytes).

### Group C — false "validator-checked" claims (4 findings)

Three passes (`rt_alias`, `page_coalesce`, `split_oversized_loads`)
declared "validator-checked" postconditions in module docs and the
orchestrator `.expect()`-ed them. **`validate_tk_tape` checked none.**
It only enforced fence-before-arrive and the *first* TmaExpect/LoadAsync
agreement.

**Closed by Patch 3 + Patch 4 in `b92b291bf0`:**

- **#9 / #14**: `RegTileSlotNotInArena` variant added to
  `TkValidationError`; `check_reg_tile_arena_membership` now walks
  every Instr exhaustively and asserts every `RegTileSlot` is a key
  in `tape.reg_tile_arena`. Closes the rt_alias arena-membership gap
  that previously had ZERO defensive walk.
- **#12**: rt_alias's two `_ => {}` catch-alls (in `instr_rt_accesses`
  and `rewrite_instr`) replaced with exhaustive Instr variant lists.
  New rt-bearing Instr variants are now rustc E0004 at three sites
  (validator + both rt_alias matches).
- **#13**: docstrings on `page_coalesce` and `split_oversized_loads`
  updated to say "in-pass-checked, NOT validator-checked" (the
  truth — their postconditions can't be validator-stage-blind because
  pre-pass tapes legitimately violate them).
- **rt_alias fail-soft removed**: `slot_remap.insert(slot, slot);
  continue;` at coalesce.rs:335-340 → panic with full diagnostic.

### Group E — AttnDecode KV cache layout (4 findings)

Independent of Group A. Even with page shape fixed, AttnDecode's
KV-cache loads are structurally wrong:

- **#16 chunk-stride**: loop stride = `CHUNK_ROWS * ROW_BYTES`
  = 128 × 1024 = 128 KiB; per-iter TMA fills 32 KiB. **75% of every
  chunk is skipped.** Each iteration covers cache positions
  `[i*128 .. i*128+32)` only.
- **#17 K-tile col-axis crosses heads**: smem tile is 128 cols;
  cache row layout is `num_kv_heads × head_dim` = 8 × 64 → 128 cols
  span 2 KV heads (head h cols 0..64 + head h+1 cols 0..64).
  Q@K^T sums across two distinct heads.
- **#18 RopeAppend cache write OOB**: see Group A finding #3.
- **#20 seq_len convention**: kernel arg name says "seq_len" but is
  interpreted as "chunk count = positions/128"; no IR-level
  enforcement, host-side launcher mistake silently truncates the
  prefix or reads off-end.

### Group F — substrate sizing

- **#21 NUM_PAGES = 5 < observed peak 6-7** for the 16-layer
  Llama-3.2-1B decode tape. `page_coalesce_pass` panics at
  proc-macro time (good — no silent runtime OOB). Bumping
  `NUM_PAGES` violates the Hopper 228 KiB cap. Real fix is a
  shmem→gmem spill pass (`page_coalesce.rs:393` documents the
  escape hatch).

## What's left — the actual unblock

Three patches close the remaining 17 of 21 findings. The five-patch
plan was itemized in the audit synthesis (full at
`/private/tmp/claude-502/.../w1hux08zp.output`). Patches 3+4 are
landed. Patches 1, 2, 5 are sized below.

### Patch 1 — seal page-shape contract end-to-end (largest, multi-day)

The wedge tried this session: change `nb` at `codegen.rs:6093` from
`u32::MAX` to `128`. **Surfaces a deeper gap**: `lower_dag_to_tape`'s
`input_producer` loop (`subtile_tape.rs:1003-1010`) picks ONE writer
per consumer's input region (`break;` on first overlap). With an
N-tiled producer (16 writers each writing a disjoint col-block) and
a downstream Cat::Whole consumer reading the whole tensor, the SSA
edge validator emits `EdgeMismatch` because actual_read_writers = [1]
but expected_preds = [1..16].

**Two changes are needed in lockstep:**

1. **Producer side — `subtile_ir::lower_region`**: extend N-tiling
   beyond Gemm. Element-wise ops (Mul, Add, Silu, SiluMul) tile
   trivially. RmsNorm needs cross-chunk reduction (sum-of-squares
   accumulated across `hidden / 128` chunks, then divide and apply
   gamma in a finalize pass). RopeRotate / RopeAppend need
   head-aligned tiling. AttnDecode is already loop-shaped per
   AttnDecode_Qkt's K-loop; tiling its input q region by N gives one
   AttnDecode per q-head block.

2. **Consumer side — `subtile_tape::lower_dag_to_tape`**: drop the
   `break;` in the `input_producer` loop. Collect ALL overlapping
   writers per consumer-input. The `ComputeInput::Computed` slot
   handle becomes `Vec<Computed>` or similar, and the lowering
   arms in `lower_subtile_tape_to_tk_tape` consume from the
   per-block predecessor pages rather than one merged page.

3. **Hardware seal — `LoadSpec`/`StoreSpec`**: delete
   `new_runtime_shape` constructors; require a `where
   PageSubTileShape::byte_size() <= PAGE_SIZE` const-generic bound.
   A SubtileIR region wider than the substrate page becomes a Rust
   compile error, NOT a runtime OOB.

Estimated scope: 1500-2500 LOC across 3+ files, plus tests and the
RmsNorm chunked-reduction rewrite. Multi-day effort.

### Patch 2 — AttnDecode KvCachePageShape (medium)

Decouple K/V smem tile from the page tile shape. K/V tile becomes
`SmemTileId<CHUNK_ROWS, HEAD_DIM, Bf16>` per kv-head; iterate
kv-heads explicitly. Chunk stride/fill ratio enforced as
`where ChunkBytes == ChunkPositions * RowBytes` const identity.
RopeAppend StoreSpec becomes `<1, ROW_BYTES, Bf16>` (one cache row).
`seq_len` kernel arg becomes typed `ChunkCount` newtype on the
launcher side. ~600-1000 LOC.

### Patch 5 — shmem→gmem spill pass (small-medium)

Walk the tape, identify PageIds with the longest non-recently-used
gap, spill to a per-warp gmem scratch region across that gap. New
`Instr::SpillToGmem` / `Instr::ReloadFromGmem`. Lets `NUM_PAGES`
stay at 5 (Hopper 228 KiB cap holds). ~400 LOC + tests.

## Pod paths + commands

Path: `/home/nickm/vllm-ff-subtile/vllm-rs/` on `nick`. Context:
`nickm/api-fmaas-vllm-d-fmaas-res-ibm-com:6443/nickm@us.ibm.com`.

Sync src/:

```
oc --context nickm/... rsync \
  /Users/nickm/git/vllm/.claude/worktrees/ff-subtile/vllm-rs/crates/ferrite-wavefront/src/ \
  nick:/home/nickm/vllm-ff-subtile/vllm-rs/crates/ferrite-wavefront/src/
```

Pod tests (116 should pass after `b92b291bf0`):

```
oc --context nickm/... rsh nick bash -c \
  'cd /home/nickm/vllm-ff-subtile/vllm-rs && \
   FERRITE_MODELS=llama-3.2-1b cargo test -p ferrite-wavefront --lib'
```

Force regen the .cu (today this hits the Group A page-shape gap;
will work once Patch 1 lands):

```
oc --context nickm/... rsh nick bash -c \
  'cd /home/nickm/vllm-ff-subtile/vllm-rs && \
   cargo clean -p ferrite-forward-macro -p ferrite-model-llama && \
   FERRITE_WAVEFRONT=1 FERRITE_MODELS=llama-3.2-1b \
   cargo build -p ferrite-model-llama --features cuda'
```

Inspect emitted .cu (when it builds):
`~/.cache/cudaforge/megakernels/tk_decode_full_llama_3_2_1b.cu`.

## Open tasks

- **Patch 1** — full page-shape contract seal. Largest single piece
  of work remaining; closes 9 findings.
- **Patch 2** — AttnDecode KvCachePageShape. Closes 4 findings.
- **Patch 5** — shmem→gmem spill pass. Closes 1 finding.

## What "done" looks like for the next session

1. Patch 1 lands as a sequence of 4-6 commits:
   (a) `lower_dag_to_tape` collects all overlapping writers (drop
       the `break;`).
   (b) `lower_region` N-tiles element-wise ops (Mul, Add, Silu,
       SiluMul) — trivial since per-chunk math is local.
   (c) `lower_region` N-tiles RmsNorm with cross-chunk reduction;
       RmsNorm arm in `lower_subtile_tape_to_tk_tape` rewritten to
       chunked sum-of-squares + finalize.
   (d) `lower_region` head-tiles RopeRotate / RopeAppend / AttnDecode.
   (e) `LoadSpec::new_runtime_shape` / `StoreSpec::new_runtime_shape`
       deleted; const-generic where-clause enforces byte_size <=
       PAGE_SIZE.
   (f) `nb = 128` (or whatever produces page-shaped nodes) at
       codegen.rs:6093.
2. Patch 2 lands.
3. Pod regen produces a .cu with all chained ops correctly tiled.
4. nvcc + ptxas compile cleanly.
5. Kernel runs (`vllm bench latency` or similar) and produces
   coherent decode output for a Llama-3.2-1B prompt.

The K-tile + N-tile pass machinery (Phase 2 + Phase 3) and the
validator hardening (Patches 3+4) are in place and tested. The
remaining work is the producer/consumer co-evolution at the
SubtileIR/lowering boundary.
