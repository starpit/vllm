# Stencil IR — status & handoff

Dated 2026-04-19. Companion to `STENCIL_IR_DESIGN.md` (vocabulary freeze) and `STENCIL_IR_SKETCH.md` (struct shapes). This doc is the "what landed, what's next, where to look" layer; the other two stay untouched.

## Read order for a fresh session

1. `STENCIL_IR_DESIGN.md` — vocabulary (3 roles, 5 dep kinds, 3 clarifications). Frozen.
2. `STENCIL_IR_SKETCH.md` — §11 struct sketch that preceded the crate. Largely realised; see divergences at the bottom of this doc.
3. This doc — current state + next steps.

## What's landed

Seven commits on `worktree-ff2` on top of `500a4ca4c`:

| Commit | Layer | Tests |
|---|---|---|
| `6e0435c44` | `ferrite-stencil` crate: IR types + FA2 prefill template + SM90/SM89 ArchMap + print round-trip | 5 |
| `685d24950` | Paged-KV decode instantiation of same Attn template (gather addr, per-sequence bound) | 6 |
| `11054159e` | Arch-neutral primitives: `classify_axes` / `region_pipeline_depth` / `topo_order_within_iter` | 5 |
| `d77532c4b` | FUF→Stencil lowering v1 in `ferrite-forward-macro::lower_to_stencil` | 4 |
| `216d03354` | Wavefront scheduler: preamble / body / epilogue, Pipeline iter_offset, arch-specific barriers | 7 |
| `cbd0e88dd` | Capstone end-to-end: FUF + Assignment → Megakernel → Schedule | 1 |
| _pending_   | Parallel wire-up into `forward!` drive: tolerant `lower_assignment_partial` + per-variant telemetry line; runs on every real-model expansion | 6 |

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

## What's deferred (priority order)

### 1. Wire lowering into the macro drive — **parallel pass landed; emitter still TODO**
`lower_assignment_partial` is now called inside the per-model loop in `lib.rs` right after the existing `ferrite · …` telemetry line. It lowers the first workload point's `Assignment`, picks `sm90_fa2` or `sm89_fa2` from `target_profile.compute_capability`, schedules every region, and prints a `ferrite stencil · <variant> · X/Y regions scheduled on <arch> · N subgraphs skipped` line. Verified on llama-{2,3}-{7,13,70}b × every quant variant — 100% region scheduling success, 0 errors. **Does not yet feed codegen**; the skipped subgraphs are all non-attention impls (gemm, rmsnorm, silu, rope, …) with no region template.

Remaining work on this axis:
- **(b) replace codegen path**: migrate `emit_model` to read `Megakernel` + `Schedule`. Requires every Impl the library ships to have a region template — i.e. blocked on deferred item #5 and friends. Pre-req before this is worth attempting: bridge back from `FufOpRef.tag: &'static str` to a concrete `TileId` so the emitter knows which kernel launch each step corresponds to (see hazards below).

### 2. Real shape inference to replace `LowerHints`
`LowerHints { head_dim, num_head_groups, tile_q, tile_k, pipe, tokens_per_page }` is pass-through today. The right source:
- `head_dim`, `num_head_groups` — from the FUF Attention tile's input (slot 0 = Q) output shape via upstream tile lookup (Q tile has shape `[..., num_heads * head_dim]` after `FusedQkvRopePrefillImpl`).
- `tile_q`, `tile_k`, `pipe` — from per-impl cost-model tuning; lives on the Impl itself, not the FUF. Add accessors on `AttentionPrefillContiguousImpl` etc. that return the tile dims the kernel was calibrated for.
- `tokens_per_page` — from `KvCachePool` config, an extern reachable via `FufInput::Extern { kind: ExternKind::KvCache, .. }`.

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
