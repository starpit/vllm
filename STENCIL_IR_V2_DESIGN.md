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

   Initial result flagged **23 of 218 variants heterogeneous** (`[ImplId(3), ImplId(37)]` and similar GemmRef-vs-tuned pairs repeating). Root cause turned out to be the canonical hash being too coarse: two bare `Gemm` Regions per transformer layer (attention-output O and MLP-down) both serialize to the same intra-Region structure (one Gemm node, one axis set, zero edges) and so collapse into one class despite having different shapes and different optimal kernels. **Fix**: extend the canonical hash with a 1-hop neighbor signature (sorted op-tags of upstream + downstream Regions via control edges) so attn-O (upstream = `Attention`) splits from MLP-down (upstream = `Mul`). Diagnostic + fix landed in a follow-up commit; see §13.

   Note: the "solver pivot" in §7 turned out to be a verification problem, not a search re-architecture. Today's solver already picks Impls per-subgraph and the question is whether those picks are class-uniform. With the neighbor-aware hash they are — except for a small structural edge case in Qwen3 (§13 item A.2) that legitimately cannot be split by op-tag topology alone.

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

**Branch**: `worktree-ff3`. Six commits on top of `0ab1aad98`:
- `aef01d8ca` — IR + subtile + region_formation + periodicity + CollapsePlan
- `420a71690` — class→impl consistency checker + FormedRegions refactor
- `5f3e96cd5` — neighbor-aware canonical hash; 23 → 2 heterogeneous variants
- `db9c13847` — `StencilBundle` plumbed through emit_wave_walk / emit_subgraph. Scaffolding only; no emission change.
- `6323d514e` — **6.1 landed**: per-layer arrays on Weights + `WeightLayout` read rewrite. `cargo expand` on llama-2-13b shows `input_layernorm: [RmsNorm; 40]`, `mlp_down_proj: [LinearLayer; 40]`, etc. collapse to array fields. Fused qkv/gate_up accessors stay flat (stems embed sibling layer idxs) — deferred follow-up.
- `61dd42ed9` — **6.2.a landed**: `repeat_var: Option<TokenStream>` + `layer_expr(u64)` helper on EmitCtx. Both construction sites set `None`, so emitted code is byte-identical; the field is dormant until 6.2.b wires loop emission.

**What builds**: everything. `cargo build -p ferrite-models --release` exercises all 218 model variants through the full pipeline. Stencil diagnostic prints alongside each variant's ferrite line (see "stencil · N regions → N classes" entries).

**Code map**:
- `vllm-rs/crates/ferrite-stencil-ir/` — IR types. 249 lines. Lowering-agnostic (the name).
- `vllm-rs/crates/ferrite-forward-macro/src/subtile.rs` — FUF → tile-parametric annotation. Axis vocabulary + per-op rules + Gemm role inference.
- `.../src/region_formation.rs` — subgraphs → Regions + control edges. Emits `FormedRegions { graph, region_subgraphs }`.
- `.../src/periodicity.rs` — canonical hashing + class grouping + summary stats.
- `.../src/periodicity_plan.rs` — measured post-collapse shape (`CollapsePlan`).
- `.../src/class_impl.rs` — class→Impl consistency check, strict + tolerant resolvers.
- `.../src/lib.rs` — pipeline wired into the macro drive (search for `stencil ·`). Diagnostic only; does not affect emitted code.

**Test coverage**: 161 unit tests total across the crate (added `boundary_positions_split_from_interior` + `four_layer_interior_collapses`; rewrote four tests whose synthetic linear chains no longer collapse under the stricter hash). All pass. `cargo clippy --all-targets -- -D warnings` clean.

**Measurement after the hash fix**: distribution of class counts shifted from a tight 8/10 to 12/15/17/18/22 across 218 variants. Collapse factor for llama / mistral drops from ~28× to ~19×; still well-collapsed and every class is now impl-consistent. Pre-fix 23 heterogeneous variants → 2 remaining (both Qwen3; see A.2).

### Quick-start for next session

Entry point: **6.2.b class-loop emission** (§13 item B below). 6.1 + 6.2.a have landed; the IR + plumbing is in place. The remaining work is the actual loop-emission rewrite — this is the critical-path win.

Before writing code, validate two empirical questions:

1. **Schedule contiguity**: is today's wave schedule layer-serial (all of layer 0's waves, then layer 1's, …) or does it interleave across layers? If serial, option (a) from §B.6.2 works directly: emit a class loop at the first wave mentioning any member, skip members in later waves. If interleaved, plan on re-planning the schedule over collapsed Regions (bigger refactor of `schedule.rs`).

   Quick check: pick llama-2-7b, print the subgraph-ids visited per wave in `emit_wave_walk`, see whether they cluster by layer. Expected layer-serial per the "283 waves across 40 layers, avg 7 waves/layer" stencil output, but confirm before committing to option (a).

2. **`layer_expr` suffix parity**: today's emit bakes `#layer` with `layer: usize` → `quote!` produces a `5usize`-suffixed literal via `ToTokens for usize`. 6.2.a's `layer_expr` produces `u64_unsuffixed` tokens. Before lifting in impls, make `layer_expr` match `usize::to_tokens` byte-for-byte (return `quote! { #concrete_usize }`) so the pre-pivot unrolled path stays byte-identical. Two impl sites use `layer as u64` in non-quote positions (`impl_lib.rs:4760`, `:5548`); those need a separate concrete-u64 binding preserved alongside the TokenStream.

**2026-04-20 empirical check result (llama-2-13b `cargo expand` inspection)**: the wave schedule IS layer-serial — each layer's subgraphs occupy a contiguous run of waves, and no wave mixes subgraphs from different layers. Option (a) is viable on schedule grounds.

**But** the inspection surfaced a second, deeper issue the sub-step plan hadn't flagged: the residual stream threads **across iterations** via borrowed tile views, e.g. layer 0 emits `let t_8_0 = (*t_0_0).as_view();` and layer 1 emits `let t_15_0 = (*t_8_0).as_view();` — iteration `N+1`'s first view is a borrow of iteration `N`'s last owner. Collapsing this into a single `for __repeat in 0..N { … }` body requires turning that inter-iteration chain into a loop-carried mutable variable (`let mut residual = embed_out; for __repeat in 0..N { residual = body(residual, __repeat); }`), which is a data-flow rewrite, not a textual loop wrap.

Same problem with tile-local idents: `t_0_0`, `t_5_0`, `t_20_0`, … are globally unique today (allocated by TileId). Inside one loop body the emitter would need single-iteration idents — which implies **emitting from the collapsed FUF, not the unrolled one** (re-schedule over the collapsed Region, use its single-iteration tile ids inside the body, hoist cross-iteration state out). That is a real restructuring of `emit_wave_walk` / `build_local_map`, not a wrap in a `for`.

Practical consequence: 6.2.b is bigger than the sub-step list suggested. Before continuing, open question for the next session: do we (i) build the collapsed-FUF codegen path from scratch (cleanest but significant), (ii) analyse the unrolled emission post-hoc to detect loop-liftable regions and residual streams (fragile, but reuses today's emitter), or (iii) stop short of full loop emission for this phase — ship the per-layer weight-array win (already in 6.1) plus the fragment dedup that's already structural, and defer true loop emission until the IR pipeline surfaces the collapsed FUF to the emitter. Option (iii) is the honest incremental move.

Sub-steps inside 6.2.b (land each on its own commit):

1. ✅ **`layer_expr` suffix fix** (commit `b1b4df294`) — emits a `usize`-suffixed literal when `repeat_var` is None so the dormant helper is a drop-in rewrite at every `let layer = … as usize; quote!{..#layer..}` site. No behavior change; unlocks sub-step 2.
2. **Literal lifting in 3–5 impls** (`FusedQkvRopeCacheImpl`, `AttentionViaCacheImpl`, `RopeAppendRef`, `SlidingAttentionImpl`, plus fp8/interleaved variants). Route every site that baked `layer` into quote through `ctx.layer_expr(layer_u64)`. `repeat_var` stays `None` so `cargo expand` on llama-2-13b is byte-identical pre/post lift (that's the gate).
3. **Class-loop emit in `emit_wave_walk`** — when a bucket's schedule exposes a contiguous run of periodic class members, emit one `for __repeat in 0..#N { … }` loop wrapping fragments indexed by `__repeat`. Gate behind `FERRITE_STENCIL_CODEGEN=1` so the default build remains today's unrolled path until llama-2-7b passes `vllm chat` under the new path.
4. **Heterogeneous tolerance** for the 2 Qwen3 variants (A.2). Emit two fragments keyed `(class_idx, impl_id)`, dispatch via a baked `const TABLE: [u8; N]`.
5. **Validation + flip default** — `vllm chat` (with `timeout`) on llama-2-7b (correctness gate), `vllm bench latency` delta ≤ ~1% (runtime gate), `cargo build -p vllm-cli --features cuda --release` wall-time delta (compile-time gate). Flip `FERRITE_STENCIL_CODEGEN=1` to the default only after all arches green.

### Open items for the next session

**(A) Heterogeneous-impl variants — finished except for two edge cases.**

*A.1 Resolved (21 of 23)*: the common pattern was two bare-Gemm Regions per layer (attention-O, MLP-down) sharing a class. 1-hop neighbor-signature hash splits them. Landed.

*A.2 Still flagged: qwen3-1.7b, qwen3-8b.* Same heterogeneity pair `[ImplId(3), ImplId(37)]` = GemmRef vs CutlassGemv. Dump: two per-layer Regions `rep_ops=["Gemm"] · period=56` (or 72 for 8b), both with `up=[RmsNorm] down=[Reshape]`. These are Qwen3's Q and K projections — both produced by a per-head RmsNorm (the QK-norm) and both consumed by a Reshape before the RoPE fusion. The entire op-tag chain `Gemm → Reshape → RmsNorm → RopeAppend → Attention` is identical. **They are genuinely indistinguishable by op-tag topology alone.** Distinguishing them would require shape information in the IR, which §5.2 explicitly forbids to keep the IR tile-size-invariant.

   Resolution: this is the case the step-6 codegen pivot needs to handle via *heterogeneity tolerance* (not solver constraint). A class that picks two impls emits two fragment bodies keyed by `(class_idx, impl_id)`; the call site dispatches per-iteration using a small indirection table the solver already has (`region_subgraphs[rid] → impl_id`). Cost: two fragments instead of one for that class only. Every other class in the variant keeps its single-fragment collapse.

**(B) Step 6 — Rust codegen pivot.** Biggest work item. Sub-step status:

- **6.0 ✅ Plumbing (commit `db9c13847`)**: `StencilBundle` computed per workload bucket inside `emit_forward_for_bucket` / `emit_forward_backbone_for_bucket`; carries `class_of: HashMap<SubgraphId, usize>`, `class_impl_id: Vec<Option<ImplId>>` (None = heterogeneous), `class_members: Vec<Vec<SubgraphId>>`. Threaded through `emit_wave_walk` → `emit_subgraph` via an ignored `_stencil` param. Not yet consumed — next sub-steps read it.

- **6.1 ⏳ Weights-struct per-layer arrays.** The big load-bearing piece. Today `emit_weights_struct` (codegen.rs line ~1050) emits one `pub attn_norm_0: RmsNorm, pub attn_norm_1: RmsNorm, …` field per layer and a matching per-field `let` in `load_with`. Call sites emit `&wm.attn_norm_3`. For a `for repeat in 0..N { … }` loop, we need `&wm.attn_norm[repeat]`. Three options considered (2026-04-20 discussion):
  - (1) **True arrays** in struct + loader builds `[RmsNorm; N]`. Cleanest; changes three build-up vecs (`fields`, `lets`, `field_shorthand`) + the constructor shape. Loader body stays per-layer — array assembly is `let attn_norm = [attn_norm_0, attn_norm_1, …];` after the existing per-layer lets, with the flat lets consumed into it.
  - (2) **Accessor methods** — keep flat fields, add `fn attn_norms(&self) -> [&RmsNorm; N]` per family. Minimal loader change; LLVM should inline. Less clean but bounded.
  - (3) **Inline `match repeat { … }`** at call site. No struct change. Biggest LLVM-visible cost — loses most of the compile-time win and risks runtime regression per §12.
  User preference expressed: "cleanest" → (1). Implementation sketch:
  1. After `collect_accessors`, run a family-detection pass: group accessors whose `name` matches `<stem>_<N>` for consecutive `N = 0..num_hidden_layers`, identical `rust_type`, and identical `FieldLoad` plan shape (same variant + identical structure modulo prefix index).
  2. For each family emit `pub #stem: [#ty; N]` in the struct; skip the per-layer flat fields.
  3. In the loader body, keep the per-layer `let #name = …` lines (unchanged from today — they're what actually load each layer's weight), and at the Self{} constructor append array assembly: `let #stem = [#stem_0, #stem_1, …, #stem_{N-1}];` then `#stem,` in the constructor (replacing the N flat field shorthands). For ungrouped accessors, today's path unchanged.
  4. Accessor data model: add `family: Option<(Ident /* stem */, u64 /* index */)>` to `WeightAccessor`, populated at `default_required_weights` time when the source weight has `Some(index)`. Call-site weight-arg emission in `emit_subgraph` reads this: when `family.is_some()` and we're emitting a class loop, `&wm.#stem[#loop_var]`; else today's `&wm.#name`.

- **6.2.a ✅ `repeat_var` plumbing (commit `61dd42ed9`).** `EmitCtx` gains `repeat_var: Option<TokenStream>` + `layer_expr(concrete: u64) -> TokenStream`. Both today's ctx constructions (Concrete inline-fallback + Abstract fragment body) set `None`; `layer_expr(c)` returns `c`'s literal tokens in that case, matching pre-pivot emission. Dormant until 6.2.b flips `repeat_var` to the loop ident.

- **6.2.b ⏳ Class-loop call sites + literal lifting.** Once 6.1 + 6.2.a land:
  - Rewrite `emit_wave_walk` to walk classes instead of subgraphs where possible: for each homogeneous class that intersects this bucket's wave schedule, emit one fragment body (abstract-emit via the class representative) and a call site `for __repeat in 0..#period { __frag_N(args_indexed_by___repeat); }`. Heterogeneous classes (Qwen3 A.2) fall back to per-subgraph emission — the bundle's `class_impl_id[c] == None` flag gates this.
  - Tension with today's wave scheduler: waves interleave subgraphs across layers for parallelism (wave W may have layer 0's attn + layer 1's mlp). A class's members are therefore not contiguous in the emission order. Solutions:
    (a) Emit the class loop at the first wave that mentions any class member; skip class members in later waves. Loses wave-parallelism across classes but preserves wave-parallelism within. Simplest.
    (b) Re-plan waves to group-by-class — bigger refactor of schedule.rs.
    (a) first; reassess if runtime regresses.
  - Literal lifting: `#layer` inside `impl_lib.rs` is baked from `node.inputs` via `rope_kv_cache_layer(…)`. To accept a loop variable, the impl's `emit_call` needs a way to receive an override. Add a new field on `EmitCtx`: `repeat_var: Option<TokenStream>`. When `Some(tokens)`, emit_call substitutes `#tokens` for the concrete layer literal wherever it's derived from a `KvCache(layer)` input. Affected impls (per grep for `k_cache(#layer)` / `k_cache(#layer)` in impl_lib.rs): `FusedQkvRopeCacheImpl` (incl. `_fp8` + `_interleaved`), `AttentionViaCacheImpl`, `RopeAppendRef`, `SlidingAttentionImpl`. 3-5 impls, bounded.

- **6.3 ⏳ Validation.** `vllm chat` on llama-2-7b (correctness gate, `timeout` wrapper per memory), `vllm bench latency` delta ≤ ~1% (runtime gate). Compare `cargo build -p vllm-cli --features cuda --release` wall time pre/post pivot.

- **6.4 ⏳ Heterogeneous tolerance (Qwen3 A.2).** Emit N fragments keyed `(class_idx, impl_id)`; call site dispatches `match __repeat_impl_table[__repeat] { ImplId(3) => __frag_A(...), ImplId(37) => __frag_B(...) }`. Table filled from `stencil.class_members` + `assignment.impl_of(sg)` at macro time — baked into a `const TABLE: [u8; N]`.

Suggested gating: `--cfg stencil_codegen` or `FERRITE_STENCIL_CODEGEN=1` env var. Keep today's unrolled emitter as default until 6.3 on llama is green; flip the default when all arches green.

**(C) Delete the per-subgraph fragment dedup string machinery** once step 6 is confirmed. `FragmentLibrary::by_sig` keyed on `"ti=...|ci=...|wt=...|body=..."` becomes redundant — the new `class_idx` is the dedup key.

### Red flags to watch for

- If the class-consistent subset (currently 216 of 218 variants) ships before the two Qwen3 stragglers land via A.2's `(class_idx, impl_id)` tolerance, the emitter needs to fall back to the old per-subgraph path for them. Don't let the code duplicate emit logic across two passes — parameterize one emitter over "per-class vs per-subgraph."
- Runtime perf — if codegen can no longer bake the layer index as a literal, `ctx.kv_cache.k_cache(layer)` becomes an indirect indexed fetch. Trivial cost next to kernel launch latency, but measure once on llama-2-7b via `vllm bench latency` before calling it done.
- `vllm chat` (with `timeout`, per your memory) is the correctness gate. Type checks and unit tests don't prove inference correctness.

### If the scope feels wrong

The measurements in step 4's table are the anchor. Any time compile time feels intractable again, check whether the regression is from reintroducing per-layer work (unrolling, per-layer literals in abstract bodies, per-layer fragment keys) or from somewhere else. The refactor is only valuable if the N_layers factor stays collapsed.
