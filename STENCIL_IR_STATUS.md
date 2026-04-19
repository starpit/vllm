# Stencil IR — status & handoff

Dated 2026-04-19. Companion to `STENCIL_IR_DESIGN.md` (vocabulary freeze) and `STENCIL_IR_SKETCH.md` (struct shapes). This doc is the "what landed, what's next, where to look" layer; the other two stay untouched.

## Read order for a fresh session

1. `STENCIL_IR_DESIGN.md` — vocabulary (3 roles, 5 dep kinds, 3 clarifications). Frozen.
2. `STENCIL_IR_SKETCH.md` — §11 struct sketch that preceded the crate. Largely realised; see divergences at the bottom of this doc.
3. This doc — current state + next steps.

## What's landed

Nine commits on `worktree-ff2` on top of `500a4ca4c`:

| Commit | Layer | Tests |
|---|---|---|
| `6e0435c44` | `ferrite-stencil` crate: IR types + FA2 prefill template + SM90/SM89 ArchMap + print round-trip | 5 |
| `685d24950` | Paged-KV decode instantiation of same Attn template (gather addr, per-sequence bound) | 6 |
| `11054159e` | Arch-neutral primitives: `classify_axes` / `region_pipeline_depth` / `topo_order_within_iter` | 5 |
| `d77532c4b` | FUF→Stencil lowering v1 in `ferrite-forward-macro::lower_to_stencil` | 4 |
| `216d03354` | Wavefront scheduler: preamble / body / epilogue, Pipeline iter_offset, arch-specific barriers | 7 |
| `cbd0e88dd` | Capstone end-to-end: FUF + Assignment → Megakernel → Schedule | 1 |
| `8f10848db` | Parallel wire-up into `forward!` drive: tolerant `lower_assignment_partial` + per-variant telemetry line; runs on every real-model expansion | 6 |
| `e31388fe7` | `LowerHints::from_model_bounds` — derive `head_dim` + `num_head_groups` from config, bake into telemetry (`h=… g=…`) | 7 |
| _pending_   | `emit::emit_kernel_sketch` — (Region, Schedule, ArchMap) → CUDA-shaped source; structure real, bodies stub | 3 |

28 tests green through `cargo clippy -p ferrite-stencil --tests -- -D warnings` and same for `ferrite-forward-macro`. Run:

```
cd vllm-rs && cargo test -p ferrite-stencil && cargo test -p ferrite-forward-macro --lib lower_to_stencil
```

## Pipeline that now exists

```
Fuf + solver::Assignment + impl_lib::ImplementationLibrary
    → ferrite_forward_macro::lower_to_stencil::lower_assignment(..., &LowerHints)
    → ferrite_stencil::Megakernel { regions, control }
    → for region in regions: ferrite_stencil::schedule_wavefront(region, &ArchMap)
    → ferrite_stencil::Schedule { preamble, body, epilogue, pipeline_depth, serial_axis, parallel_axes }
```

Each `Step` in the schedule carries `node: NodeId`, `iter_offset: i32` (+P for pipeline-source loads), and `barriers_before: SmallVec<[BarrierPrim; 2]>` already lowered through the chosen `ArchMap`. SM90 and SM89 regions are identical; only the arch table differs.

## Key file pointers

- Crate root: `vllm-rs/crates/ferrite-stencil/`
  - `src/ir.rs` — core types + `validate()`
  - `src/template.rs` — `attn_region` (prefill + finite-window) + `attn_region_paged_decode`
  - `src/arch.rs` — `ArchMap`, `HardwareUnit`, `BarrierPrim`, `sm90_fa2`, `sm89_fa2`
  - `src/schedule.rs` — arch-neutral primitives
  - `src/wavefront.rs` — preamble/body/epilogue scheduler
  - `src/print.rs` — round-trip printer (used by tests, not by emitter)
- Lowering: `vllm-rs/crates/ferrite-forward-macro/src/lower_to_stencil.rs`
- FUF producer pointers (read-only from stencil's POV):
  - `ferrite-forward-macro/src/fuf.rs` — `Fuf`, `FufNode`, `TileId`, `FufInput`
  - `ferrite-forward-macro/src/classified.rs` — `OpKind` (attention is one variant)
  - `ferrite-forward-macro/src/solver.rs` — `Assignment`, `SubgraphId`, `WorkloadAssignments`
  - `ferrite-forward-macro/src/impl_lib.rs` — `Implementation` trait; attention impls at ~5288, 4615

## Finish-line target

*Efficient megakernel execution from the FUF* — i.e. comm/compute overlap, cross-subtile parallelism, good SM utilization. The plumbing above (IR, templates, scheduler, lowering, parallel wire-up) is *pre-emitter*; none of it produces running code. The critical path from here is the emitter.

Emitter plan (sketched, not committed):
1. ✅ **sketch-level emission** (`emit::emit_kernel_sketch`): round-trippable CUDA-shaped text with real signature / grid / role dispatch / pipeline loop / barrier prims. Not compilable.
2. ✅ **intrinsic expansion** (`emit_ops::expand`): all 7 FA2 ops (`load_q/k/v_tile`, `qk_matmul`, `softmax_update`, `pv_matmul`, `store_o_tile`) render concrete pseudocode per (tag, arch). Helpers (`tma_load_2d`, `wgmma_mma_async`, `cp_async_128`, `stg_128`, …) still symbolic.
3. **build integration & first running kernel** — detailed plan below.
4. **multi-region composition**: populate `Megakernel.control` in the lowering (QKV+RoPE → FA2 → O_proj), emit as a persistent megakernel with inter-region barriers.
5. **region templates for non-attention ops** to actually compose. Only needed once step 4 is wired; until then they're dead weight.

## Step 3 plan: build integration & first running kernel

**Goal**: get a generated stencil kernel to compile, launch, and produce numerically correct output on real GPU tensors. Correctness before performance — the first landed kernel is allowed to be slower than the hand-tuned `AttentionPrefillContiguousImpl` it shadows. Performance parity is step 3f.

**Scope narrowing**:
- SM89 only to start. No TMA, no wgmma — just `cp.async.ca` + `mma.sync`. Half the surface area, and it runs on the L4 dev box.
- Single shape profile: `head_dim=128`, `tile_q=64`, `tile_k=32`, `pipe=2`. Picked to match what the existing FA2 kernel handles.
- Prefill attention only. Paged decode → after prefill works end-to-end.

**Helper-resolution strategy**: a hand-written CUDA prelude header (`stencil_prelude_sm89.cuh`) containing inline `__device__` functions / `#define` macros for every helper the emitter references — `cp_async_128`, `cp_async_wait_group`, `mma_sync_accumulate`, `row_max`, `exp2f_frag`, `stg_128`, etc. Prelude is reviewed once by humans; emitter stays simple (each op keeps emitting the same call it does today, nothing changes above the prelude line). Rejecting alternative (direct PTX asm in bodies) — too hard to review per-op, too easy to break silently.

**Build layout**: new crate `ferrite-stencil-kernels` (sibling of `ferrite-kernels`). Its `build.rs`:
1. Pulls `ferrite-stencil` as a build-dep.
2. Enumerates the kernel profiles it wants to build (v1: one profile — `fa2_prefill_sm89_h128_tq64_tk32`).
3. For each profile, calls `emit_kernel_sketch` → writes `<OUT_DIR>/<profile>.cu` with the prelude `#include`d at the top.
4. Shells out to nvcc (`/usr/local/cuda-12.9/bin/nvcc` per `feedback_cuda_toolchain_path`) to compile each .cu into a .cubin / PTX / .o, linked into a single `libstencil_kernels.a`.
5. Emits `cargo:rustc-link-lib=static=stencil_kernels` + `cargo:rerun-if-changed=` for the prelude header and the stencil crate source.

**Launch glue**: hand-written first, auto-generated later. A `ferrite-stencil-kernels::launch::fa2_prefill_sm89` function taking `CUstream` + device pointers + `(num_q_tiles, num_kv_tiles)` and calling `cuLaunchKernel` with the symbol loaded from the linked cubin. Auto-generation from the emitter (launch dims derived from schedule's parallel_axes) becomes step 3c once the hand-written path proves the plumbing.

**Sub-commits**:

- **3a — prelude + first compiling generated .cu**
  Write `stencil_prelude_sm89.cuh` (vanilla impl — no cp.async yet, just straight gmem→smem copies and `__syncthreads`). Add `ferrite-stencil-kernels` crate with `build.rs` that writes a *hand-picked* kernel source (borrowed from an existing FA2 reference) with the prelude included, compiles it. Unit test: link against the crate, verify the symbol resolves. No end-to-end correctness yet. **Deliverable**: nvcc compiles something the emitter will eventually produce.

- **3b — hand-written launch wrapper + CPU-reference integration test**
  Add a Rust `launch::fa2_prefill_sm89(stream, q, k, v, o, m, n, d, …)` that calls the 3a kernel. Integration test: generate tiny random Q/K/V in pinned host memory, copy to device, launch, copy O back, compare against a CPU scalar FA2 reference implemented in the test. Tight rtol/atol for bf16. **Deliverable**: proof that launching a stencil-layout kernel through our glue is correct.

- **3c — cut over to generated source**
  Replace the hand-picked kernel source in 3a with `emit_kernel_sketch(attn_region(...), schedule_wavefront(...), sm89_fa2())`. Prelude stays identical. 3b's test must still pass — any delta surfaces in the diff of generated vs hand-picked source, reviewed at commit time. **Deliverable**: the emitter's output runs correctly end-to-end on GPU.

- **3d — add cp.async for overlap**
  Upgrade the prelude to use real `cp.async.ca` intrinsics (behind a feature flag or a compile-time constant in the prelude header). Scheduler already emits `cp_async_wait_group(depth)` — should Just Work once the prelude reads cp.async-commit/wait macros. Verify 3b's correctness test still passes; add a microbench comparing to the non-pipelined path. **Deliverable**: first measurable perf datapoint from generated kernels.

- **3e — SM90 prelude + TMA/wgmma**
  Add `stencil_prelude_sm90.cuh` with TMA descriptor setup and wgmma macros. Parameterize `ferrite-stencil-kernels/build.rs` to compile one SM89 + one SM90 profile. Integration test on H100 is gated (no H100 in the L4 CI); run it ad-hoc. **Deliverable**: SM90 compiles and the H100 path has at least one green run.

- **3f — wire generated kernel into the Impl dispatch**
  Swap `AttentionPrefillContiguousImpl::emit_call` to emit a call into `ferrite-stencil-kernels::launch::…` for the shape profiles that have generated kernels. Keep the old launch path as fallback for shapes outside the generated set. Gate behind a feature flag (`ferrite-stencil-kernels`) so we can A/B correctness via `vllm chat` (per `feedback_no_run_chat`). **Deliverable**: a coherent `vllm chat` run that routes attention through a generated kernel, end-to-end.

**Known hazards**:
- nvcc build time is long (minutes, per `feedback_cuda_compile`); keep the profile set minimal until 3f.
- Caching-allocator / worktree lib collisions (`feedback_cudaforge_cache`) — 3a's build.rs output lands in `OUT_DIR` not a shared path, but watch for `.a` clobbering across worktrees once 3f flips the default.
- Cost-table keys (`fa2_attn_bf16_h128`) live in `cost_l4_sm89.csv`; once 3f is default the generated kernel's cost row should shadow the hand-rolled one. Validate predicted vs actual once 3d gives us numbers.
- Integration tests must run through `vllm chat` at 3f, not just unit tests (`feedback_verify_foundations` / `feedback_trace_real_input`).

## What's deferred (priority order)

### 1. Wire lowering into the macro drive — **parallel pass landed; emitter still TODO**
`lower_assignment_partial` is now called inside the per-model loop in `lib.rs` right after the existing `ferrite · …` telemetry line. It lowers the first workload point's `Assignment`, picks `sm90_fa2` or `sm89_fa2` from `target_profile.compute_capability`, schedules every region, and prints a `ferrite stencil · <variant> · X/Y regions scheduled on <arch> · N subgraphs skipped` line. Verified on llama-{2,3}-{7,13,70}b × every quant variant — 100% region scheduling success, 0 errors. **Does not yet feed codegen**; the skipped subgraphs are all non-attention impls (gemm, rmsnorm, silu, rope, …) with no region template.

Remaining work on this axis:
- **(b) replace codegen path**: migrate `emit_model` to read `Megakernel` + `Schedule`. Requires every Impl the library ships to have a region template — i.e. blocked on deferred item #5 and friends. Pre-req before this is worth attempting: bridge back from `FufOpRef.tag: &'static str` to a concrete `TileId` so the emitter knows which kernel launch each step corresponds to (see hazards below).

### 2. Real shape inference to replace `LowerHints` — **partial: bounds-derived fields landed**
`LowerHints::from_model_bounds(&model.bounds)` now fills `head_dim` (from `bounds["head_dim"]`) and `num_head_groups` (= `num_attention_heads / num_key_value_heads`, clamped ≥ 1). Wired into the macro drive; telemetry shows `h=<dim> g=<groups>` per variant. Verified against llama-{2,3,3.2}-* — MHA → g=1, GQA matches expected ratios.

Still pass-through:
- `tile_q`, `tile_k`, `pipe` — from per-impl cost-model tuning; lives on the Impl itself, not the FUF. Add accessors on `AttentionPrefillContiguousImpl` etc. that return the tile dims the kernel was calibrated for.
- `tokens_per_page` — from `KvCachePool` config, an extern reachable via `FufInput::Extern { kind: ExternKind::KvCache, .. }`.

Observation surfaced by the telemetry: variants with `head_dim=64` (smollm2, llama-3.2-1b) show `0/0` regions scheduled — their attention tiles are claimed by an Impl whose name doesn't match the three we template (`attention_prefill_contiguous`, `attention_via_cache`, `sliding_attention_prefill_contiguous`). Worth digging into before declaring deferred item #1's parallel path "covers the set"; `sliding_attention_via_cache` (impl_lib.rs:5433) is a likely culprit.

### 3. Explicit prologue steps in `Schedule`
Currently `Schedule` declares `pipeline_depth: P` but the prologue Vec is empty. The emitter needs either the empty vec + P (it unrolls the prologue itself) or the expanded `P * |body-loads|` steps here. Pick one; sketch already says inline expansion is fine.

### 4. Megakernel CFG (inter-region `ControlEdge`s)
Currently `Megakernel.control: Vec::new()`. First use: compose `QKV+RoPE → FA2 → O_proj` as three regions linked by `DepKind::Barrier`. Lowering would need to map non-attention impls too, which loops back to #1 (codegen replacement).

### 5. MoE region template + `DataDependent` dep kind
Design §7.2. Requires the static-per-invocation CTA assignment protocol from `bincount[E]`. Only bite once attention emission is working end-to-end.

## Integration hazards

- **Proc-macro crate runs at compile time.** `ferrite-stencil` is a library dep of `ferrite-forward-macro`; `Megakernel` values exist only during macro expansion. Emitter turns them into `TokenStream`; no `Megakernel` at runtime.
- **`FufOpRef.tag: &'static str` is a leaky abstraction.** Real lowering-to-emission needs a way to resolve the tag back to the FUF tile so the emitter can emit the right kernel launch. Either (a) add a `TileId` field to `FufOpRef` during lowering, or (b) keep a side-table in `Megakernel`. Prefer (a).
- **Axis IDs are region-local.** `Q_TILE=0` in prefill and `B_AXIS=0` in decode are both id 0 — that's fine because they're in different `Region`s, but the scheduler must never mix them across regions. Current code doesn't, but any future cross-region analysis has to respect this.
- **`debug_assert!` in `wavefront::build_step`** asserts `iter_offset <= pipeline_depth`. Release builds silently accept mismatches. If templates ever diverge dep vectors from `p.pipe`, switch to a hard error.

## Sketch → reality divergences worth knowing

- `AxisId` is `u16` (sketch said `SmallVec<(AxisId, i32)>`, left unspecified). Works.
- `RegionTemplate` wasn't implemented as a type; templates are free functions (`attn_region`, `attn_region_paged_decode`). Collapsed into plain functions because the `build: fn(&TemplateArgs)` indirection added no information when there's one call site per template.
- `HardwareUnit::Warpgroup` uses `role_name` + `num_wg` rather than `first_warp` + `count`. Matches the design doc's "reclaim controller wg, 5 wg = 20 warps" comment rather than pinning exact warp indices — those are codegen-time decisions, not mapping-time.
- `ArchMap.role: fn(Role, &Region) -> HardwareUnit` is a plain `fn`, no trait object. Two arches, both written by hand, no dynamic dispatch needed.

## Memory pointer

`~/.claude/projects/-home-moosevan-vllm/memory/project_ferrite_stencil.md` tracks this status for future sessions via auto-memory; if you update the status here, skim that file too.
