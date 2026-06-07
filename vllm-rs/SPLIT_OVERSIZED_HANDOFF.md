# Handoff — ff-subtile worktree, post-phase-3 NK extension; conservative-lowering broken at every op boundary

## Where to work

- **Worktree:** `/Users/nickm/git/vllm/.claude/worktrees/ff-subtile`
- **Branch:** `worktree-ff-subtile`
- **Crate:** `vllm-rs/crates/ferrite-wavefront/`
- **HEAD:** `f0fc5eae1b` — split_oversized_loads_pass NK extension + Affine2D byte_off
- **Pod:** `nick`, path `/home/nickm/vllm-ff-subtile/vllm-rs/`

`pwd && git branch --show-current` first thing — per
`memory/feedback_handoff_worktree_match.md`.

## Session arc (3 commits since 35c0cb7231)

| Commit | Summary |
|---|---|
| `36024bc11f` | **Phase 2** — K-loop rewrite. Each oversized LoadAsync paired with its `WgmmaMmaAB_RegSmem` consumer is replaced by a K-loop over 128×128 chunks. Mirrors AttnDecode_Qkt. |
| `3edbefc62e` | Handoff doc update: phase 2 done; production hits N>128 first. |
| `f0fc5eae1b` | **Phase 3** — NK extension. Adds `ByteOffsetExpr::Affine2D` variant + player emit; dispatch in `plan_rewrite` (`n_blocks == 1` → K-only; `>1` → NK with outer N-loop wrapping inner K-loop and per-N output StoreAsync). 114 lib tests pass on the H100 pod. |

## State right now

The K-tile + N-tile machinery is correct and tested for the
oversized LoadAsync side. Driving the proc-macro on the pod (after
`f0fc5eae1b`) advances PAST the N>128 guard but hits a deeper gap:

```
error: custom attribute panicked
  --> crates/ferrite-model-llama/src/lib.rs:14:1
  = help: message: split_oversized_loads_pass: A operand of Wgmma at idx 20
                   has no LoadAsync (a_page = PageId(0)). The pass cannot
                   K-fragment a prior-op page — re-staging from gmem
                   requires routing changes upstream.
```

For Llama-3.2-1B's `q = gemm(normed, q_proj_weight)`:

- `B` (q_proj weight) is `External` → `emit_external_load` emits a
  fresh oversized LoadAsync. The pass K/N-tiles it. ✅
- `A` (normed = RmsNorm output) is `Computed` →
  `resolve_input_page` returns the existing page WITHOUT emitting a
  LoadAsync. The pass's K-tile path needs a gmem source to re-load
  per-K chunks; there isn't one in the tape. ❌

The deeper issue: the conservative lowering is **broken at every op
boundary** for `hidden > PAGE_COLS` (= 128), not just at the Gemm
LoadAsync side. Concrete trace:

1. **RmsNorm forward**: arm reads input page as
   `SmemTileId::<128, 128, Bf16>` and writes output as
   `SmemTileId::<128, 128, Bf16>` (single 32 KB page). The
   `SubtileIR::TensorRegion` for the output says `128 × hidden =
   128 × 2048` (= 512 KB). Only the first 128 cols of K are computed.
2. **`emit_store_and_arrive`**: takes `tile = region_tile_shape(out)`
   = `128 × 2048` and emits `StoreAsync(dst_page, tile=128×2048)`.
   The TMA store reads 512 KB from a 32 KB smem page → OOB on the
   smem read side. The gmem write region is `128 × 2048` but only
   the first 32 KB is meaningful data.
3. **Gemm reads `A=normed`**: SubtileIR shows `Computed(rmsnorm_slot)`,
   `resolve_input_page` returns the same `dst_page`. The Gemm reads
   it as `SmemTileId::<128, 128, Bf16>` — gets the (correct, since
   that's all RmsNorm wrote) first 128 cols. K=2048 portion is lost.
4. **Gemm writes output**: same OOB pattern (`emit_store_and_arrive`
   stores `128 × N_full` from a `128 × 128` page).

Every chained op carries this corruption. The test fixtures pass
because they're synthetic single-op tapes — the chain isn't there.

## What needs to happen (the actual unblock)

The K-tile + N-tile machinery this session built is **necessary
but not sufficient**. The conservative lowering needs M/N/K tiling
**at the op level**, coordinated across the chain. Three options
in increasing scope:

### Option A: Op-level tiling in the SubtileIR / lowering (RIGHT but BIG)

Restructure the SubtileIR so each `SubtileNode` is page-shaped:
RmsNorm becomes `M_blocks × N_blocks` nodes each producing a 128×128
sub-tile of the logical RmsNorm output. The Gemm chain consumes
those page-shaped outputs naturally.

This is what the `partition.rs::lower_partitioned` path produces
(N=64 blocks for q_proj at TP=1) but it's not in the wavefront
ferrite-model-llama lowering path — that path produces ONE node per
Linear with full TensorRegions. Routing wavefront through
`lower_partitioned` (or porting the same fragmentation logic) is
the architectural fix.

Scope: substantial. Touches the SubtileIR construction, every
SubOp arm in `lower_subtile_tape_to_tk_tape`, and the page
allocation strategy. Probably 2-3 weeks of work to do right.

### Option B: Tape-level rewrite that re-stages from gmem (TRACTABLE)

Extend `split_oversized_loads_pass` so when A is `Computed` (no
LoadAsync), the pass walks BACK through the tape to find the prior
op's `StoreAsync` whose `src_page == a_page`. That `StoreAsync`'s
`dst_arg` is the gmem tensor that holds the prior op's full output.
The pass then emits per-K-chunk `LoadAsync`s using that tensor as
the source for `A`.

This works ONLY IF the prior op's gmem output is correct. Today it
isn't (per #2 above — RmsNorm OOBs the smem read side). So Option B
requires Option B' first:

### Option B': Fix the smem-side OOB at every op (also TRACTABLE)

A sibling pass that walks `StoreAsync` Instrs and detects the case
where `tile.byte_size() > PAGE_SIZE` (smem side over-read). For
each, rewrite into an N-loop that issues per-N-chunk StoreAsyncs
from the same dst_page (whose 128×128 contents are NOT the right
data for cols ≥ 128 anyway — so this needs the prior op's compute
arm to have ALREADY been M/N-tiled, see Option A).

Recursive: B' depends on the upstream RmsNorm/etc. arms producing
correct multi-chunk output, which means the lowering's RmsNorm arm
itself needs M/N-tiling.

### The honest assessment

The scope of work to actually launch the kernel is **larger than the
prior handoffs anticipated**. Phase 2 + Phase 3 are correct and
well-scoped. The remaining work is:

1. **Lowering-level M/N tiling** for non-Gemm ops (RmsNorm,
   Add, Mul, SiluMul, RopeRotate, RopeAppend) so each page-shaped
   write-side maps to a 128×128 chunk. This belongs in the SubOp
   arms of `lower_subtile_tape_to_tk_tape`, NOT in a TkTape→TkTape
   pass.

2. **Computed-input re-staging** in `split_oversized_loads_pass`
   so when A is Computed, the pass walks back to the prior op's
   StoreAsync to recover the source gmem tensor and emits fresh
   LoadAsyncs.

OR, alternative:

3. **Route ferrite-model-llama through `lower_partitioned`** so each
   SubtileNode is page-shaped at the SubtileIR level. Then the pass
   pipeline as-is (rt_alias, page_coalesce, split_oversized_loads
   with K-only) is sufficient.

Option 3 is probably the cleanest path to a working kernel, IF
`lower_partitioned`'s output is well-formed for wavefront's
codegen (it was designed for the partition.rs / TP path).

## Pod paths + commands

Path: `/home/nickm/vllm-ff-subtile/vllm-rs/` on `nick`. Context:
`nickm/api-fmaas-vllm-d-fmaas-res-ibm-com:6443/nickm@us.ibm.com`.

Sync src/:

```
oc --context nickm/... rsync \
  /Users/nickm/git/vllm/.claude/worktrees/ff-subtile/vllm-rs/crates/ferrite-wavefront/src/ \
  nick:/home/nickm/vllm-ff-subtile/vllm-rs/crates/ferrite-wavefront/src/
```

Pod tests (114 should pass after `f0fc5eae1b`):

```
oc --context nickm/... rsh nick bash -c \
  'cd /home/nickm/vllm-ff-subtile/vllm-rs && \
   FERRITE_MODELS=llama-3.2-1b cargo test -p ferrite-wavefront --lib'
```

Force regen the .cu (today this hits the "A has no LoadAsync"
panic; will work once Option 3 lands or Option 2 covers Computed
inputs):

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

- **The actual unblock** — pick Option 1, 2+B', or 3 from §"What
  needs to happen". Option 3 is the recommended path.
- #71 split_oversized_loads_pass phase 2 — **DONE** at `36024bc11f`.
- Phase 3 NK extension — **DONE** at `f0fc5eae1b`.
- #53 RopeRotateInterleaved (Instr step, deferred).
- 2 deferred audit findings: `subtile-cols-idx-flat-u16`, S4 PageId
  pub(crate) inner.

## What "done" looks like for the next session

1. Pick a path (Option 1, 2+B', or 3). Option 3 recommended.
2. If Option 3: route ferrite-model-llama through
   `lower_partitioned`, validate the resulting SubtileIR has
   page-shaped nodes (244 fuf tiles → many more nodes per Gemm),
   verify the existing pass pipeline handles it.
3. Pod regen produces a .cu with all chained ops correctly tiled.
4. nvcc + ptxas compile cleanly.
5. cudaFuncSetAttribute accepts the DYN_SMEM (substrate is
   224 KB ≤ Hopper's 228 KB — verified by const_assert).
6. Kernel runs (`vllm bench latency` or similar) and produces
   coherent decode output for a Llama-3.2-1B prompt.

The K-tile + N-tile pass machinery is in place; it just needs the
upstream lowering to produce well-formed page-shaped nodes (or the
Computed-input re-staging extension) for it to actually fire on
production tapes.
