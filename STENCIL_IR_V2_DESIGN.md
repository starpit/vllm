# Stencil IR v2 — design

Status: design-before-code. Dated 2026-04-20. Supersedes the ff2 sketch in spirit; reuses the vocabulary (roles, dep kinds) but fixes the granularity and scope issues.

Sits between FUF (op-level math DAG, fully unrolled, const-propped) and codegen (Rust launch dispatch today; optional megakernel CUDA later). Purpose: make the solver, scheduler, and codegen operate on a **re-rolled, tile-parametric** graph so compile time scales with *distinct patterns*, not with *unrolled occurrences*.

## 1. Why this exists

**Root cause of slow compilation**: FUF is fully unrolled across layers. Every downstream pass — solver, scheduler, codegen — fights the unrolling:

- Solver re-solves identical subgraph structure per layer (32× for llama).
- Codegen emits ~487 fragments / 9630 call sites / 1920 inline fallbacks **per model variant**, most of which are identical-modulo-layer-index.
- Incidental fragment dedup defeated by per-layer literals baked into abstract bodies (`ctx.kv_cache.k_cache(#layer)` etc.).

**The fix is structural, not polish**: introduce an IR level where repetition is represented as *iteration domain* rather than *unrolled tiles*. Downstream passes then see a graph whose node count is O(distinct patterns), not O(distinct patterns × num_layers × …).

Non-goal polish (fragment layer-parameterization, proc-macro crate split, kernel api/impl split) gives ≤2× wins and leaves the fundamental quadratic-ish growth in place. This doc is for the 10–30× win.

Secondary goal: the same IR is a better substrate for megakernel codegen (SM90+ warp specialization, TMA, WGMMA) than ff2's IR, because shared-axis semantics across Regions tell the wavefront scheduler where cross-Region pipelining is legal.

## 2. Vocabulary

Three IR levels. Arch-specific concerns live in level 3.

1. **FUF** — existing. Op-level math DAG, fully unrolled, const-propped.
2. **Stencil IR (this doc)** — arch-neutral. N-D iteration domain, role-annotated nodes, typed dep edges, Region CFG with shared-axis semantics across Region boundaries.
3. **Lowering** — per target:
   - **Rust launch dispatch** (today's runtime on SM89 and below): each Region lowers to one kernel launch; the Region's axes become `for` loops in emitted Rust; roles collapse into a single launch that does Load+Compute+Store internally.
   - **Megakernel CUDA** (deferred, SM90+): each Region lowers to a chunk of a `__global__`; shared axes across Regions enable cross-Region pipelining (TMA load of tile t+1 overlaps WGMMA compute of t); roles lower to warp-group specialization.

Same IR, two lowerings. Neither is on critical path of the other.

### 2.1. Roles (frozen, 3)

Inherited from ff2 design §2 verbatim:

| Role | What |
|---|---|
| `Load` | global → shared tile move |
| `Compute` | tensor-core math on tiles |
| `Store` | shared → global incl. atomic variants |

Roles are *informational* — they tell lowerings what kind of hardware resource the node wants. SM89 lowering collapses all three into one kernel; SM90+ specializes.

### 2.2. Dep edge kinds (frozen, 5)

Inherited from ff2 design §3:

| Kind | Meaning |
|---|---|
| `Raw` | same-CTA true data dep |
| `Pipeline` | intra-region producer/consumer, staged |
| `AtomicReduce` | cross-CTA accumulation |
| `Barrier` | region boundary, all CTAs pass |
| `DataDependent` | runtime value decides dispatch (MoE, spec accept) |

Same semantics as ff2. Edge vectors describe offset on the iteration domain.

## 3. What's new vs ff2

Two structural changes relative to ff2's design:

### 3.1. N-D spatial domains + shared-axis semantics across Regions

ff2's Region is a self-contained sub-tile stencil with its own axes. Axes don't *name-match* across Regions; the region-graph CFG only carries control edges.

**We add**: axes are **symbolic names** (`tile_t`, `tile_d_inter`, `tile_d_head`, …). Two Regions that share an axis name share deps across the Region boundary — wavefront pipelining is legal along the shared axis. Regions with disjoint axes implicitly barrier at the boundary.

This is what makes Impl fusion choices *visible in the IR*:

- Fused MLP Impl (`gate_proj + up_proj + silu·mul + down_proj`) → one Region with domain `(tile_t, tile_d_inter)`. Internal edges carry dep vectors on both axes; pipelined fusion is intra-Region.
- Unfused MLP Impls → four Regions each with domain `(tile_t,)`. `tile_d_inter` doesn't cross the Region boundaries → intermediates materialize. Structurally reflects why fusion wins.

Axis set on a Region is therefore an **emergent property of the Impl picks**, not pre-committed.

### 3.2. Periodicity re-rolling

ff2 lowers `for sg in assignment.subgraphs()` → one Region per subgraph, leaving the FUF's per-layer unroll intact. This has to go.

**We add**: a detection pass that finds maximal DAG-isomorphic subsets of Regions under affine substitution on an axis. The repeating unit collapses to one Region with an **outer iteration axis** (let's call it `repeat` — the IR doesn't know or care that a transformer's repeat-bound equals `num_hidden_layers`). Cross-iteration deps become dep vectors with `Δrepeat = ±1` on the outer axis.

This is pure DAG algebra. Applies to layer repetition, head repetition, token repetition equally — the IR is domain-agnostic.

## 4. IR shape

```rust
pub type AxisId = u16;
pub type NodeId = u16;
pub type RegionId = u16;
pub type ScalarId = u16;

/// Symbolic axis name + bound. Name-equality across Regions means
/// deps flow; name-disequality means barrier.
pub struct Axis {
    pub id: AxisId,
    pub name: &'static str,    // e.g. "tile_t", "tile_d_inter", "repeat"
    pub bound: Bound,
}

pub enum Bound {
    Const(u32),
    RegionEntryScalar(ScalarId),   // resolved at launch; e.g. T / tile_t
    IndexedScalar(ScalarId, AxisId),
    Unbounded,
}

pub struct Domain {
    pub axes: Vec<Axis>,
    pub predicates: Vec<Predicate>,   // affine constraints (causal mask etc.)
}

pub struct Node {
    pub id: NodeId,
    pub role: Role,                   // Load | Compute | Store
    pub op: FufOpRef,                 // backlink to the FUF op being realized
    pub addr: Option<LoadAddr>,       // affine in domain coords + region-entry scalars
}

pub struct Edge {
    pub src: NodeId,
    pub dst: NodeId,
    pub kind: DepKind,
    pub vector: DepVector,            // offset in axis coords (Δ per axis)
}

pub struct Region {
    pub id: RegionId,
    pub name: &'static str,
    pub domain: Domain,
    pub entry_scalars: Vec<ScalarBinding>,
    pub nodes: Vec<Node>,
    pub edges: Vec<Edge>,
    pub impl_id: ImplId,              // the solver's Impl pick that owns this Region
}

pub struct RegionGraph {
    pub regions: Vec<Region>,
    pub control: Vec<ControlEdge>,    // Barrier / DataDependent between Regions
}
```

**Axis identity across Regions**: `axis.name` is the shared symbol. A name that appears in two Regions' domains is the same axis — wavefront scheduler can pipeline. Different names = implicit barrier.

**Stability**: axis names are drawn from a small frozen vocabulary (below). New axes get added to the vocabulary, not invented ad-hoc, so cross-Region sharing is always intentional.

### 4.1. Axis vocabulary (v1)

Frozen set. Adding an axis requires a justification similar to adding a role.

- `tile_t` — token-tile. The primary spatial axis for every per-token op + activation-side of every GEMM. Shared across essentially every Region in a transformer forward.
- `tile_d_inter` — MLP intermediate dim tile. Present only in MLP Regions; enables gate/up/silu·mul/down fusion.
- `tile_d_head` — head-dim tile. Present in attention Regions (pre-softmax / post-softmax matmuls).
- `head_group` — head-group axis in attention. Shared within attention sub-regions, not with MLP.
- `q_tile` / `kv_tile` — attention-internal sub-tile axes. Intra-region only.
- `repeat` — outer iteration bound introduced by periodicity detection. Generally shared across the Regions inside the detected period.

Not axes (intentionally): `layer`, `block`, `transformer_block`. The IR doesn't know these terms. `repeat` is the generic form.

## 5. Sub-tiling pass (step 1, load-bearing)

Input: FUF. Output: FUF annotated with natural tile axes per op.

Rewrite every FUF op into tile-parametric form. For each op, declare:
- Which axes it naturally lives over (subset of vocabulary).
- Dep vectors to upstream ops in the FUF.
- Role assignment for each piece.

### 5.1. Per-op axis assignment

| FUF op family | Axes | Notes |
|---|---|---|
| `rmsnorm`, `silu`, `mul` (elementwise), `add` | `(tile_t,)` | Per-token; weights (norm gain) are region-entry scalars. |
| `embed` | `(tile_t,)` | Gather; address is `token_ids[tile_t]`. |
| `gemm` (activation-side input) | `(tile_t, tile_d_out)` | `tile_d_out` may be `tile_d_inter` (mlp), `tile_d_head × head_group` (attn projections), or the hidden dim for the output projection. Weights are region-entry (no tile axis). |
| `rope_append` | `(tile_t, head_group)` | Per-token per-head; q/k/v outputs carry the same axes. |
| `attention` | `(head_group, q_tile, kv_tile)` | Cross-token reduction → stencil boundary. Not shareable with MLP's axes. |
| `lm_head` (final gemm) | `(tile_t, tile_d_vocab)` | `tile_d_vocab` is a new axis — vocab-dim tile. Intra-region only for now. |

This list is the operational output of the sub-tiling pass: every FUF op gets a row.

### 5.2. Tile size — symbolic

Tile sizes are **symbolic parameters** (`TILE_T`, `TILE_D_INTER`, …) resolved at lowering time. The IR is tile-size-invariant. One Stencil graph serves every tile size; the solver or arch-lowering pass instantiates.

Consequence: no combinatorial expansion in tile-size space at IR construction time. Specialization happens inside one codegen pass that picks tile sizes per arch.

### 5.3. Stencil boundaries

Ops whose dep pattern can't be expressed with affine dep vectors on spatial axes are **boundaries**:

- `attention` — cross-tile reduction (softmax normalizes across all K tiles).
- `embed` — boundary only because it's the DAG source (no upstream).
- `lm_head` — DAG sink.
- `rmsnorm` — reduces across D (if the hidden dim is tiled); if we don't tile hidden dim in the outer stencil (and we don't — we tile only `tile_d_inter` in MLPs), rmsnorm is token-wise. *Not* a boundary.

Boundaries produce `Barrier` control edges between adjacent Regions, and their axes don't share with their neighbors.

## 6. Region formation + periodicity (step 2, load-bearing)

### 6.1. Region formation

Runs after the solver picks Impls. Group tile-parametric FUF ops into Regions:

- Each Impl's claimed tiles + their sub-tiled forms → one Region.
- Region's axes = union of axes declared by its constituent ops.
- Axes that exit the Region (shared with downstream Region via name-equality) become inter-Region dep edges.
- Axes that die inside the Region are Region-internal only.

Output: a `RegionGraph` with one Region per Impl instance. Still fully unrolled along the layer dim at this point — one Region per (Impl, layer).

### 6.2. Periodicity detection

Find maximal subsets of Regions that are isomorphic under affine substitution on some axis.

**Algorithm sketch**:

1. Canonicalize each Region to a hash over `(nodes, edges, axes)` — ignoring node IDs (positional) and replacing any `repeat`-like index-into-state with a symbolic marker.
2. Group Regions by canonical hash.
3. Within each hash group, verify that consecutive members differ only by an affine shift on their region-entry scalars (the layer index, the kv_cache layer, etc.).
4. Collapse the group to one Region + an outer `repeat` axis with bound = group size. Per-iteration scalars (layer index, kv_cache layer) become `IndexedScalar(.., repeat_axis)` — resolved at launch from a per-iteration table.
5. Rewrite inter-Region edges: edges between members of the collapsed group become intra-Region dep vectors with `Δrepeat ≠ 0`; edges crossing into/out of the group become control edges.

**What can defeat this**: if the FUF has per-layer literals baked into node addresses or scalar expressions in non-affine ways (e.g., `k_cache(layer) where layer is a hardcoded match)`, the canonical hash differs across layers and detection fails. The sub-tiling pass has to lift such literals to region-entry scalars, i.e., replace `k_cache(3)` with `k_cache(scalar[repeat])` where the scalar resolves to the layer index at iteration `repeat = 3`.

This lifting is the main correctness work for periodicity detection. It's a FUF-level pattern: any per-op-per-layer-varying literal that today is baked into a tile's state must become a scalar indexed by the outer iteration.

### 6.3. What periodicity detection does NOT do

- Does not invent fusion. Fusion is the solver's job (via Impl picks).
- Does not invent axes. The sub-tiling pass assigns axes; detection only collapses along a discovered period.
- Does not reorder dependencies. Only rewrites edges in place.

Keeps the pass simple and auditable. If it fails on a model, it falls back to emitting the unrolled graph — correctness is never compromised by failed detection.

## 7. Solver pivot (step 3, correctness risk)

Solver today: picks `Impl` per `Subgraph` (unrolled FUF subgraph) per `WorkloadPoint`. Search space ≈ subgraph_count × impl_options × workload_count.

Solver after pivot: picks `Impl` per *pattern* — per (canonical subgraph, workload_point). Search space ≈ pattern_count × impl_options × workload_count. `pattern_count ≈ subgraph_count / num_layers` for a transformer forward.

**Correctness risk**: the solver today could in principle pick Marlin at layer 0 and Fp8 at layer 1. After the pivot, it must pick one Impl for the entire repeat group. Per existing invariants (weights have uniform quant format per model; workload point is layer-invariant), this has always been true in practice. But it becomes *enforced* rather than happens-to-hold.

Mitigation: the pivot lands behind a feature flag; compare emit volume + solver picks side-by-side with the pre-pivot solver on a variety of models; green-lit only after A/B agrees.

## 8. Codegen (step 4, mechanical once IR stable)

### 8.1. Rust launch dispatch (critical path — today's runtime)

One fragment body per Region × canonical Impl pick. The fragment takes:
- Tile-indexed activations (as today's `TensorView<'_>` / `OwnedTensor`)
- Region-entry scalars (replaced today's baked-in literals)
- Iteration-indexed scalars (the `scalar[repeat]` refs — as `&[T]`)

Call site wraps the fragment call in `for repeat in 0..REPEAT_BOUND { ... }` loops. Sub-tile axes (`tile_t`, `tile_d_inter`) are iterated inside the kernel (fragment implements its own inner loops — not the codegen's concern at this level).

Fragment library dedup becomes **structural**: two Regions with the same (canonical hash, Impl pick) are the same fragment by construction. No more string-matching against abstract bodies. The N_layers factor disappears from fragment count — that's the compile-time win.

### 8.2. Megakernel CUDA (deferred)

Same IR, different emitter. Each Region's sub-tile structure + role annotations + dep vectors drive warp-specialized `__global__` emission. Shared-axis semantics across Regions enable cross-Region wavefronting (next Region's Load overlaps this Region's Compute+Store).

ff2's `emit_kittens.rs` / `emit_mega.rs` get ported to consume v2 Regions. Most of that work is mechanical — the emitters already produce CUDA per Region; they just need to thread the outer `repeat` iteration through. Defer until after Rust-side lowering is green.

## 9. Staging

Each step leaves the codebase correct and buildable. Parallel old/new paths during migration.

1. ✅ **Extract `ferrite-stencil-ir` crate** — IR types only (`Axis`, `Domain`, `Node`, `Edge`, `Region`, `RegionGraph`). Reuse ff2 shapes where they apply, drop megakernel-emitter code. Zero behavior change. *Landed 2026-04-20 (commit `aef01d8ca`).*

2. ✅ **Sub-tiling pass** — walks FUF, annotates each op with its axis set + dep vectors per §5. Prints the sub-tiled graph for inspection. No consumers yet — pure augmentation of FUF. *Landed 2026-04-20 (commit `aef01d8ca`). Module: `ferrite-forward-macro/src/subtile.rs`. 8 unit tests.*

3. ✅ **Region formation** — after the existing solver picks Impls, group into Regions per §6.1. Emit `RegionGraph` struct. Again: no consumers yet; diff the Region count vs per-layer-subgraph count to verify Region formation is correct. *Landed 2026-04-20 (commit `aef01d8ca`). Module: `ferrite-forward-macro/src/region_formation.rs`. 4 unit tests.*

4. ✅ **Periodicity detection — measurement.** Canonical-hash + group regions; build a CollapsePlan describing what a post-collapse graph would look like. *Landed 2026-04-20 (commit `aef01d8ca`). Modules: `periodicity.rs` + `periodicity_plan.rs`. 8 unit tests. **Literal lifting not yet needed** — today's Region nodes carry only op tags + roles; no per-layer literals bake in until codegen hangs addresses or entry scalars off the IR.*

   Measured on real llama FUFs:

   | Model | Layers | Regions | Classes | Collapse | Control edges |
   |---|---|---|---|---|---|
   | llama-2-7b  | 32 | 227 | 8 | **28.4×** | 290 → 10 |
   | llama-2-13b | 40 | 283 | 8 | **35.4×** | 362 → 10 |
   | llama-2-70b | 80 | 563 | 8 | **70.4×** | 722 → 10 |

5. ✅ **Class → Impl consistency checker.** Confirms the load-bearing invariant for codegen: every Region in a periodic class picks the same solver Impl. *Landed 2026-04-20 (commit `420a71690`). Module: `class_impl.rs`. 3 unit tests. `form_regions` now returns `FormedRegions { graph, region_subgraphs }`.*

   **Real-FUF result**: all llama variants clean. **23 of 218 non-llama variants flag heterogeneous impls** (1–2 classes each, patterns like `[ImplId(3), ImplId(37)]` repeating). See §13 for follow-up.

   Note: the "solver pivot" in §7 turned out to be a verification problem, not a search re-architecture. Today's solver already picks Impls per-subgraph and the question is whether those picks are class-uniform. They mostly are; a few edge cases (§13) need either solver constraint or codegen tolerance.

6. ⏳ **Rust codegen pivot** — replace today's `codegen::emit_model` with a Region-driven emitter. One fragment body per (Region, Impl); call sites wrap in outer iteration loops. Verify with `vllm chat` on one model per arch family. **This is where the compile-time win lands on-disk.** Depends on step 5's invariant — either restrict codegen to the consistent variants or tolerate heterogeneity (see §13).

7. ⏳ **Delete the old path** — once every architecture runs through the new emitter and passes golden tests.

8. ⏳ **(Deferred)** Port ff2's megakernel emitter to consume v2 Regions. Not on critical path of the compile-time fix.

Critical-path remaining: step 6 (~1 week) + step 7 (~2 days) = ~1.5 weeks for the ~10–30× compile-time speedup. Step 8 is pure upside.

## 10. Open questions

- **How much per-layer state does the current FUF bake?** Whenever a node references a specific layer index (kv_cache slot, weight field name) as a concrete literal, periodicity detection requires it to be lifted to an indexed scalar. Survey the `impl_lib` impls; expect 3-5 of them (rope impls, attention impls). Bounded scope, but needs enumerating before committing to the periodicity design.

- **Does fragment dedup on the Rust side still need a string key?** After the pivot, Regions are structurally de-duped by (canonical hash, Impl pick) at Region-formation time. The codegen-time dedup library might become vestigial. Confirm post-pivot; simplify if so.

- **Multi-output Regions** (e.g. `qkv_rope` producing Q/K/V). Today's fragment infrastructure bailed out because `output_alias` + multi-output was tractable-but-deferred. Post-pivot, Regions have explicit axes on their outputs; multi-output is expressible as multiple Store nodes sharing a domain. Confirm the codegen path handles this cleanly before committing.

- **Non-transformer architectures** (MoE, Mamba, diffusion models). Periodicity detection is domain-agnostic, so in principle these Just Work. Validation is needed on at least one non-vanilla-transformer to prove the abstraction. MoE's expert selection is a natural `DataDependent` region boundary; Mamba's SSM loop is another periodicity instance.

- **Workload-point dimension interaction with `repeat`**. Today the solver picks per-workload (num_tokens bucket × sk_bucket). After the pivot, Impl picks are per-pattern-per-workload, and the `repeat` axis is orthogonal. Verify no cross-workload complication sneaks in.

## 11. Related work / prior art

- ff2's `STENCIL_IR_DESIGN.md` — vocabulary (roles, dep kinds) and sub-tile stencil structure. This doc keeps those verbatim and fixes the scope issues (no periodicity, 1-Region-per-subgraph, no shared-axis semantics).
- Polyhedral model — iteration-domain-based IR with affine deps, inspiration for the axis vocabulary and dep vectors.
- TVM / Halide — schedule-separate-from-algorithm, scheduling primitives (tile, fuse, vectorize) — we don't adopt the language but the mindset of "one algorithm, many schedules" matches.
- HazyResearch Megakernels — the sub-tile-level wavefront target for SM90+ emission (reuse ff2's port work when we get there).

## 12. What would make this wrong

Red flags that would tell us this refactor is the wrong move:

- Periodicity detection fails to collapse on >50% of models (means the FUF bakes too much per-layer state to be worth lifting). *Mitigation possible*: do the lifting; likely 3-5 impl fixes.
- Solver pivot produces measurably worse Impl picks (means per-layer pick heterogeneity was actually useful). *Unlikely but measurable*.
- Rust codegen with outer loops produces runtime regressions vs unrolled bodies (means LLVM can't vectorize / const-prop through the loop as well as it did through unrolling). *Measurable via nsys on one model*.

None of these are show-stoppers without investigation, but each is worth measuring at the relevant staging step rather than discovering at integration.

## 13. Status & handoff (2026-04-20)

**Branch**: `worktree-ff3`. Two commits on top of `0ab1aad98`:
- `aef01d8ca` — IR + subtile + region_formation + periodicity + CollapsePlan
- `420a71690` — class→impl consistency checker + FormedRegions refactor

**What builds**: everything. `cargo build -p ferrite-models --release` exercises all 218 model variants through the full pipeline. Stencil diagnostic prints alongside each variant's ferrite line (see "stencil · N regions → N classes" entries).

**Code map**:
- `vllm-rs/crates/ferrite-stencil-ir/` — IR types. 249 lines. Lowering-agnostic (the name).
- `vllm-rs/crates/ferrite-forward-macro/src/subtile.rs` — FUF → tile-parametric annotation. Axis vocabulary + per-op rules + Gemm role inference.
- `.../src/region_formation.rs` — subgraphs → Regions + control edges. Emits `FormedRegions { graph, region_subgraphs }`.
- `.../src/periodicity.rs` — canonical hashing + class grouping + summary stats.
- `.../src/periodicity_plan.rs` — measured post-collapse shape (`CollapsePlan`).
- `.../src/class_impl.rs` — class→Impl consistency check, strict + tolerant resolvers.
- `.../src/lib.rs` — pipeline wired into the macro drive (search for `stencil ·`). Diagnostic only; does not affect emitted code.

**Test coverage**: 162 unit tests total across the crate; 23 new ones cover the stencil pipeline. All pass. `cargo clippy --all-targets -- -D warnings` clean.

### Open items for the next session

**(A) Investigate the 23 heterogeneous-impl variants.** Likely one structural cause. Suspects:
- Gemma3 alternates local (sliding) / global attention per layer — but subtile.rs gives `SlidingAttention` a different OpKind tag than `Attention`, so they *should* land in different classes. Worth confirming the canonical hash actually distinguishes them on real Gemma3 FUFs.
- Qwen3's first-layer QK-norm inserts reshape tiles that might make layer-0's attention Region differ structurally from layer-1+. If so, layer-0 is its own class (period 1) and layer-1+ is another class (period N-1) — *already consistent*, not a real problem.
- Some prefill vs decode split that I'm only seeing because I picked `sfufs.per_workload.iter().next()` nondeterministically.

Quick path: print a heterogeneous variant's per-class op-tag sequence + member RegionIds + per-member ImplIds. One variant should expose the pattern.

**(B) Step 6 — Rust codegen pivot.** Biggest work item. Structure:
1. Move fragment emission to be class-indexed rather than subgraph-indexed. Today `codegen.rs::emit_model`'s `FragmentLibrary` dedups by stringified abstract body; post-pivot, dedup happens structurally at Region-formation time via `canonical_hash`. The library keying changes from string → `class_idx`.
2. Emit one fragment body per (class, workload_point, Impl). Call site wraps in `for _repeat in 0..PERIOD { ... }` over the class's period.
3. Per-iteration state (layer index, kv_cache slot) needs to flow as a loop variable into the fragment. **This is where §6.2 "literal lifting" becomes real** — the Impl's `emit_call` must replace baked layer literals with the loop variable. Bounded to 3–5 impls (`FusedQkvRope*`, `AttentionViaCache*`, `RopeAppendRef*`, `SlidingAttention*`).

Suggested approach: keep today's unrolled emitter as the default; gate the pivot behind `--cfg stencil_codegen` or an env var so A/B comparison is easy. Land per-family: do llama first (all variants clean per step 5), validate with `vllm chat`, then extend to archs that pass consistency, tackle heterogeneous ones last.

**(C) Delete the per-subgraph fragment dedup string machinery** once step 6 is confirmed. `FragmentLibrary::by_sig` keyed on `"ti=...|ci=...|wt=...|body=..."` becomes redundant — the new `class_idx` is the dedup key.

### Red flags to watch for

- If the class-consistent subset (currently 195 variants) ships before all 218, the emitter needs to fall back to the old per-subgraph path for the stragglers. Don't let the code duplicate emit logic across two passes — parameterize one emitter over "per-class vs per-subgraph."
- Runtime perf — if codegen can no longer bake the layer index as a literal, `ctx.kv_cache.k_cache(layer)` becomes an indirect indexed fetch. Trivial cost next to kernel launch latency, but measure once on llama-2-7b via `vllm bench latency` before calling it done.
- `vllm chat` (with `timeout`, per your memory) is the correctness gate. Type checks and unit tests don't prove inference correctness.

### If the scope feels wrong

The measurements in step 4's table are the anchor. Any time compile time feels intractable again, check whether the regression is from reintroducing per-layer work (unrolling, per-layer literals in abstract bodies, per-layer fragment keys) or from somewhere else. The refactor is only valuable if the N_layers factor stays collapsed.
