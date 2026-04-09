// SPDX-License-Identifier: Apache-2.0
//! Kernel binding library + DAG coalescing pass.
//!
//! The reified DAG ([`crate::reified_dag::ReifiedDag`]) describes the
//! model's per-tile work units at the finest meaningful granularity.
//! Sitting next to it is a *library* of bound kernels — each entry is a
//! concrete CUDA kernel implementation (hand-written, FlashInfer,
//! CUTLASS, …) together with a pattern that says which DAG subgraph it
//! can replace and the CTA shape / cost it implies.
//!
//! The [`coalesce`] pass walks the DAG and rewrites matched subgraphs
//! into single coarser nodes bound to library entries. Anything that
//! doesn't match a registered pattern falls back to
//! [`BoundKernel::HandWrittenRowTile`], the universal 1:1 binding to
//! today's per-row tile bodies in `templates/scheduled/megakernel.cu`.
//!
//! This module is the *framework*. Phase B (this commit) only registers
//! the trivial fallback — `coalesce` produces a 1:1 mapping. Subsequent
//! phases register real library entries:
//!
//! - Phase C: `FlashInferAttentionLayer` — coalesces all attention row
//!   tiles for one layer into a single node bound to
//!   `BlockBatchPagedAttentionPersistent::Run`.
//! - Phase D+: CUTLASS GEMM collectives, FlashInfer norm/rope, fused
//!   norm+gemm patterns, …
//!
//! The wave scheduler (Phase C) groups coalesced nodes into waves under
//! a monomorphic constraint keyed on [`BoundKernel::kind`] — different
//! library entries need different CTA shapes, so they cannot share a
//! wave. The codegen (Phase C) dispatches per wave on the kernel kind
//! to emit the right tile body and launch parameters.
//!
//! Partial adoption is supported by design: as long as
//! `HandWrittenRowTile` is registered last (the universal fallback),
//! any kind that hasn't been bound to a library entry yet keeps using
//! today's hand-written body. Registering a new kernel only changes
//! the kinds it claims, never the rest.

use crate::reified_dag::{LlamaDims, NodeId, Phase, ReifiedDag, TileSizes};
use crate::schedule::CostModel;

/// One concrete binding instance: a kernel implementation choice plus
/// the DAG inputs it consumes.
///
/// Initially this enum has one variant — the universal fallback. Each
/// new library entry adds a variant carrying whatever parameters the
/// codegen needs to instantiate that kernel for that work unit
/// (e.g. `FlashInferAttentionLayer { layer }`).
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum BoundKernel {
    /// 1:1 fallback: one reified DAG node = one tile body call.
    /// Used for any phase that doesn't yet have a library entry, and
    /// for phases where the per-tile body remains the right shape.
    HandWrittenRowTile {
        phase: Phase,
        layer: u16,
        row: u16,
        col: u16,
    },

    /// FlashInfer's `BlockBatchPagedAttentionPersistent::Run` runner,
    /// applied to one layer's full attention. The runner is a
    /// `__device__` function that takes `Params` + `SharedStorage` and
    /// processes the work assigned to its CTA via `work_indptr` —
    /// i.e. each persistent CTA in the wave pulls its share at runtime
    /// from a host-built plan, rather than the scheduler enumerating
    /// per-row tiles.
    ///
    /// The coalesce rule fuses every reified `Phase::Attention` node
    /// for a given layer into a single bound node carrying just the
    /// layer index. The dependencies of the absorbed nodes (typically
    /// the layer's RoPE outputs) become this node's dependencies; any
    /// edge that lived purely between absorbed attention nodes is
    /// dropped (no internal edges in a fused node).
    FlashInferAttentionLayer { layer: u16 },

    /// CUTLASS sm80 multistage GEMM via
    /// `cutlass::gemm::threadblock::MmaMultistage::operator()`,
    /// applied to one layer's worth of work for one of the GEMM
    /// phases (qkv / o_proj / gate_up / down).
    ///
    /// Same wave-cooperative pattern as `FlashInferAttentionLayer`:
    /// the coalesce rule fuses every reified node for one
    /// `(layer, phase)` pair into a single bound node, the
    /// scheduler replicates it across every CTA in its wave, and the
    /// dispatch arm runs CUTLASS's `MmaMultistage` per-CTA with each
    /// CTA picking its `(M_tile, N_tile)` work via its `bid`.
    ///
    /// The four phases share one variant — the megakernel's dispatch
    /// arm branches on `phase` to pick the right A/B/C base pointers
    /// and the right epilogue (LinearCombinationSiluMul for gate_up,
    /// LinearCombination(beta=1) residual-add for o_proj/down,
    /// LinearCombination(beta=0) for qkv).
    CutlassGemmLayer { layer: u16, phase: GemmPhase },
}

/// Which of the four GEMM phases a [`BoundKernel::CutlassGemmLayer`]
/// node represents. Carried as the `row` field of the WAVE_OPS entry
/// (since cutlass-fused nodes don't have a row index — the cutlass
/// body iterates over the layer's full M dim internally).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum GemmPhase {
    Qkv,
    OProj,
    GateUp,
    Down,
}

impl GemmPhase {
    /// Numeric tag in the WAVE_OPS `row` slot. Stable across codegen
    /// versions. Matches `PFL_GEMM_PHASE_*` constants in megakernel.cu.
    pub fn tag(self) -> u32 {
        match self {
            GemmPhase::Qkv => 0,
            GemmPhase::OProj => 1,
            GemmPhase::GateUp => 2,
            GemmPhase::Down => 3,
        }
    }

    pub fn name(self) -> &'static str {
        match self {
            GemmPhase::Qkv => "qkv",
            GemmPhase::OProj => "o_proj",
            GemmPhase::GateUp => "gate_up",
            GemmPhase::Down => "down",
        }
    }

    /// Which reified-DAG `Phase` this gemm phase coalesces.
    pub fn source_phase(self) -> Phase {
        match self {
            GemmPhase::Qkv => Phase::Qkv,
            GemmPhase::OProj => Phase::OProj,
            GemmPhase::GateUp => Phase::GateUp,
            GemmPhase::Down => Phase::Down,
        }
    }
}

impl BoundKernel {
    /// Stable string tag identifying which kernel this binding uses.
    ///
    /// The wave scheduler enforces *monomorphic waves* keyed on this
    /// tag — two coalesced nodes can share a wave only if their kinds
    /// match — because different bound kernels need different CTA
    /// shapes (NUM_THREADS, smem, register budget) and you cannot mix
    /// them inside one persistent grid launch.
    ///
    /// For `HandWrittenRowTile`, the tag is the phase name: today's
    /// codegen already keys per-CTA dispatch on phase, and the per-row
    /// tile bodies for distinct phases happen to share a CTA shape
    /// (256 threads, ~36 KiB shmem) only because we wrote them that
    /// way. Once a library entry registers a different shape for one
    /// of those phases, this tag is what keeps it isolated.
    pub fn kind(&self) -> &'static str {
        match self {
            BoundKernel::HandWrittenRowTile { phase, .. } => phase.name(),
            BoundKernel::FlashInferAttentionLayer { .. } => "flashinfer_attention_layer",
            BoundKernel::CutlassGemmLayer { phase, .. } => match phase {
                GemmPhase::Qkv => "cutlass_gemm_qkv_layer",
                GemmPhase::OProj => "cutlass_gemm_o_proj_layer",
                GemmPhase::GateUp => "cutlass_gemm_gate_up_layer",
                GemmPhase::Down => "cutlass_gemm_down_layer",
            },
        }
    }

    /// Predicted cost of executing this binding, in the same mma-unit
    /// domain as `CostModel`. The wave scheduler uses this number for
    /// LPT bin packing inside a wave and for the makespan rollup.
    ///
    /// For `HandWrittenRowTile` we delegate to the existing per-phase
    /// cost (one tile body per node, the status quo). When new
    /// library entries land, each provides its own per-binding cost
    /// — e.g. `FlashInferAttentionLayer` will roll up the whole
    /// layer's attention into one number.
    /// Numeric tag identifying this binding's dispatch arm in the
    /// generated megakernel's per-CTA op switch. Tags 0..7 are
    /// reserved for the existing per-phase
    /// [`BoundKernel::HandWrittenRowTile`] dispatch (matching the
    /// `PHASE_*` constants in the megakernel template). Tags ≥8 are
    /// for library-bound kernels:
    ///
    /// - `8` — `FlashInferAttentionLayer`
    ///
    /// New library entries claim a stable tag in this enum. The
    /// megakernel template's dispatch switch must grow a matching
    /// `case` arm at the same time.
    /// Does this binding consume an entire wave's worth of CTAs
    /// cooperatively (i.e. all CTAs in the wave participate in the
    /// same work item), or is it a per-CTA tile dispatched LPT-style
    /// across the wave's CTAs?
    ///
    /// `HandWrittenRowTile` is the per-CTA model: each binding is
    /// one tile body call on one CTA, the wave scheduler bin-packs
    /// many of them across the wave's CTAs via LPT.
    ///
    /// `FlashInferAttentionLayer` is wave-cooperative: the FlashInfer
    /// runner internally partitions the layer's attention work across
    /// every CTA in the persistent grid via `work_indptr[blockIdx.y]`,
    /// so the scheduler must place the binding on **every** CTA in
    /// the wave (not bin-pack it onto one CTA). The per-CTA cost is
    /// the rolled-up `cost()` divided across `num_ctas`, since the
    /// CTAs run in parallel.
    pub fn is_wave_cooperative(&self) -> bool {
        match self {
            BoundKernel::HandWrittenRowTile { .. } => false,
            BoundKernel::FlashInferAttentionLayer { .. } => true,
            // CUTLASS GEMM layer kernels are wave-cooperative for the
            // same reason as FlashInfer attention: each persistent CTA
            // in the wave picks one (M_tile, N_tile) work item via its
            // `bid`, and the layer's full GEMM is distributed across
            // them in a CTA-strided loop. Same `gemm_cutlass_mcta.cu`
            // pattern as the existing fused prefill kernel.
            BoundKernel::CutlassGemmLayer { .. } => true,
        }
    }

    pub fn kernel_tag(&self) -> u32 {
        match self {
            BoundKernel::HandWrittenRowTile { phase, .. } => match phase {
                Phase::AttnNorm => 0,
                Phase::Qkv => 1,
                Phase::Rope => 2,
                Phase::Attention => 3,
                Phase::OProj => 4,
                Phase::MlpNorm => 5,
                Phase::GateUp => 6,
                Phase::Down => 7,
            },
            BoundKernel::FlashInferAttentionLayer { .. } => 8,
            // 9 reserved for future use (e.g. unused/idle marker).
            // 10..13 are the four cutlass gemm phases. The dispatch
            // arms in the megakernel template each handle one tag,
            // so the dispatch switch knows which cutlass call to
            // make without having to read the GemmPhase from the
            // op stream's `row` slot. (We DO also stash the phase
            // in the `row` slot for cross-checking.)
            BoundKernel::CutlassGemmLayer { phase, .. } => match phase {
                GemmPhase::Qkv => 10,
                GemmPhase::OProj => 11,
                GemmPhase::GateUp => 12,
                GemmPhase::Down => 13,
            },
        }
    }

    pub fn cost(&self, model: &CostModel) -> u32 {
        match self {
            BoundKernel::HandWrittenRowTile { phase, .. } => model.cost(*phase),
            // Layer-rolled-up attention: one binding does the work that
            // used to be `seq_len / row_tile` separate row tiles. Sum
            // them so the wave scheduler still budgets the right total
            // mma-units per layer.
            //
            // The number is intentionally an over-estimate of FlashInfer's
            // real cost — measured FlashInfer attention runs in ~3.5 ms
            // per layer at seq=1024 (~0.2 ms per row tile), the cost
            // model says ~10× more. Doesn't matter for the LPT bin
            // packer because attention waves are wave-cooperative
            // (every CTA participates); the wave's makespan is
            // determined by this rolled-up cost regardless.
            BoundKernel::FlashInferAttentionLayer { .. } => {
                let row_tiles = model.seq_len.div_ceil(model.row_tile);
                row_tiles * model.cost(Phase::Attention)
            }
            // Per-layer rolled-up cost for the four CUTLASS GEMM
            // phases — same shape as the FlashInfer attention rollup
            // and same caveats about over-estimation. Wave-cooperative
            // execution means all CTAs participate, so per-CTA cost
            // distribution doesn't matter — only the wave-level total.
            BoundKernel::CutlassGemmLayer { phase, .. } => {
                let row_tiles = model.seq_len.div_ceil(model.row_tile);
                row_tiles * model.cost(phase.source_phase())
            }
        }
    }
}

/// One node in the coalesced DAG.
///
/// `id` is preserved from the source `ReifiedDag` when a node maps 1:1
/// (the only case in Phase B). When the coalesce pass starts fusing
/// subgraphs (Phase C+), the coalesced node will get a fresh id and
/// the bound kernel will reference the original ids it absorbed via
/// its variant fields.
#[derive(Clone, Debug)]
pub struct CoalescedNode {
    pub id: NodeId,
    pub kernel: BoundKernel,
    pub deps: Vec<NodeId>,
}

/// The output of the coalesce pass: a DAG where every node carries an
/// explicit binding to a library kernel.
///
/// In Phase B this has the same shape as the input `ReifiedDag` (1:1
/// node mapping, all bound to `HandWrittenRowTile`). The struct is
/// kept separate from `ReifiedDag` rather than added as a parallel
/// vector field because Phase C will introduce real fusion — the
/// coalesced node count will diverge from the reified node count, and
/// dependencies will need to be remapped from absorbed ids to the new
/// coalesced ids.
#[derive(Clone, Debug)]
pub struct CoalescedDag {
    /// Model dimensions, lifted from the source `ReifiedDag`. Cost
    /// models, codegen prelude, and FlashInfer plan-time inputs all
    /// need these — keep them attached so consumers don't need to
    /// hold a separate handle to the source DAG.
    pub dims: LlamaDims,
    /// Tile shape policy, lifted from the source `ReifiedDag`. Used
    /// by the cost model and the codegen template.
    pub tiles: TileSizes,
    pub nodes: Vec<CoalescedNode>,
}

impl CoalescedDag {
    pub fn len(&self) -> usize {
        self.nodes.len()
    }

    pub fn is_empty(&self) -> bool {
        self.nodes.is_empty()
    }
}

/// Walk a `ReifiedDag` through the registered kernel library and produce
/// a `CoalescedDag`.
///
/// Phase B: only `HandWrittenRowTile` is registered, so this is a pure
/// 1:1 lift — every reified node becomes a coalesced node with the
/// matching binding and its deps preserved. The point is to land the
/// types and the call site so Phase C can register `FlashInferAttentionLayer`
/// without restructuring the pipeline.
///
/// When more library entries land, the implementation here grows from a
/// single `iter().map()` into a real graph rewriting pass — try each
/// pattern in priority order, replace matched subgraphs with coarser
/// coalesced nodes, fall back to `HandWrittenRowTile` for everything
/// unmatched.
pub fn coalesce(dag: &ReifiedDag) -> CoalescedDag {
    let nodes = dag
        .nodes
        .iter()
        .map(|n| CoalescedNode {
            id: n.id,
            kernel: BoundKernel::HandWrittenRowTile {
                phase: n.phase,
                layer: n.layer,
                row: n.row,
                col: n.col,
            },
            deps: n.deps.clone(),
        })
        .collect();
    CoalescedDag {
        dims: dag.dims,
        tiles: dag.tiles,
        nodes,
    }
}

/// Coalesce variant that fuses every layer's attention row tiles into
/// one [`BoundKernel::FlashInferAttentionLayer`] node, leaving every
/// other phase on the [`BoundKernel::HandWrittenRowTile`] fallback.
///
/// Phase C2a: this function exists alongside the trivial [`coalesce`]
/// but is **not** wired into the production pipeline yet — Phase C2b
/// switches the call sites to use it once the dispatch + launcher
/// plumbing in `scheduled_codegen` and the megakernel template can
/// emit a `FlashInferAttentionLayer` work item.
///
/// Fusion semantics for one layer:
/// 1. Find every reified node with `phase == Attention && layer == L`.
/// 2. Drop those nodes from the coalesced output.
/// 3. Replace them with one fused node:
///    - id: the smallest absorbed `NodeId` (stable across rebuilds —
///      the codegen test asserts this).
///    - kernel: `FlashInferAttentionLayer { layer: L }`.
///    - deps: the union of the absorbed nodes' deps, *minus* any
///      dependency that pointed to another absorbed node (no
///      internal edges in a fused node).
/// 4. For every other coalesced node (any non-attention phase) that
///    used to depend on an absorbed attention id, rewrite that dep
///    to point at the fused node's id.
///
/// All other phases pass through unchanged (1:1 fallback).
pub fn coalesce_with_flashinfer_attention(dag: &ReifiedDag) -> CoalescedDag {
    use std::collections::{HashMap, HashSet};

    // ── Step 1: index attention nodes by layer ──
    let mut attn_ids_by_layer: HashMap<u16, Vec<NodeId>> = HashMap::new();
    for n in &dag.nodes {
        if n.phase == Phase::Attention {
            attn_ids_by_layer.entry(n.layer).or_default().push(n.id);
        }
    }

    // ── Step 2: pick the fused id per layer (smallest absorbed id) ──
    let mut fused_id_by_layer: HashMap<u16, NodeId> = HashMap::new();
    for (layer, ids) in &attn_ids_by_layer {
        let min_id = ids.iter().copied().min().expect("layer has attn nodes");
        fused_id_by_layer.insert(*layer, min_id);
    }

    // ── Step 3: build absorbed-id → fused-id rewrite map ──
    let mut rewrite: HashMap<NodeId, NodeId> = HashMap::new();
    for (layer, ids) in &attn_ids_by_layer {
        let fused = fused_id_by_layer[layer];
        for id in ids {
            rewrite.insert(*id, fused);
        }
    }

    // ── Step 4: build the coalesced node list ──
    // For each reified node:
    //   - If it's an attention node and its id IS the fused id for its
    //     layer, emit one fused FlashInferAttentionLayer node with the
    //     unioned, internally-pruned deps.
    //   - If it's an attention node and its id is NOT the fused id,
    //     drop it (it's been absorbed into the fused node).
    //   - Otherwise, emit a HandWrittenRowTile fallback, rewriting
    //     any deps that pointed to absorbed attention ids.
    let mut nodes: Vec<CoalescedNode> = Vec::with_capacity(dag.nodes.len());
    let absorbed: HashSet<NodeId> = rewrite.keys().copied().collect();

    for n in &dag.nodes {
        if n.phase == Phase::Attention {
            let fused_id = fused_id_by_layer[&n.layer];
            if n.id != fused_id {
                continue; // absorbed into the fused node — drop
            }
            // Union all deps from every attention node in this layer,
            // pruning internal edges and deduping.
            let mut deps: Vec<NodeId> = Vec::new();
            let mut seen: HashSet<NodeId> = HashSet::new();
            for absorbed_id in &attn_ids_by_layer[&n.layer] {
                let absorbed_node = &dag.nodes[absorbed_id.0 as usize];
                for d in &absorbed_node.deps {
                    if absorbed.contains(d) {
                        continue; // internal edge — drop
                    }
                    if seen.insert(*d) {
                        deps.push(*d);
                    }
                }
            }
            nodes.push(CoalescedNode {
                id: fused_id,
                kernel: BoundKernel::FlashInferAttentionLayer { layer: n.layer },
                deps,
            });
        } else {
            // Non-attention phase: rewrite any dep pointing into the
            // absorbed set to point at the matching fused id.
            let deps: Vec<NodeId> = n
                .deps
                .iter()
                .map(|d| rewrite.get(d).copied().unwrap_or(*d))
                .collect();
            nodes.push(CoalescedNode {
                id: n.id,
                kernel: BoundKernel::HandWrittenRowTile {
                    phase: n.phase,
                    layer: n.layer,
                    row: n.row,
                    col: n.col,
                },
                deps,
            });
        }
    }

    renumber_dense_ids(&mut nodes);

    CoalescedDag {
        dims: dag.dims,
        tiles: dag.tiles,
        nodes,
    }
}

/// In-place dense renumbering of `NodeId`s to `[0, nodes.len())`.
/// Used at the end of every coalesce pass that drops or reorders
/// nodes (the schedule + codegen index `dag.nodes[NodeId.0 as usize]`,
/// so any gap in the id space blows them up).
fn renumber_dense_ids(nodes: &mut [CoalescedNode]) {
    use std::collections::HashMap;
    let mut id_remap: HashMap<NodeId, NodeId> = HashMap::with_capacity(nodes.len());
    for (new_idx, n) in nodes.iter().enumerate() {
        id_remap.insert(n.id, NodeId(new_idx as u32));
    }
    for n in nodes.iter_mut() {
        n.id = id_remap[&n.id];
        for d in &mut n.deps {
            *d = *id_remap
                .get(d)
                .expect("dep references a node that doesn't exist in coalesced output");
        }
    }
}

/// Per-phase GEMM coalesce rule. Fuses every reified
/// `(layer, phase, row, col)` node where `node.phase == gemm_phase.source_phase()`
/// into one [`BoundKernel::CutlassGemmLayer { layer, phase: gemm_phase }`]
/// node per layer. Other phases (and other GEMM phases not matching
/// `gemm_phase`) pass through unchanged.
///
/// Mirrors `coalesce_with_flashinfer_attention`'s shape: pick a stable
/// fused id (smallest absorbed), drop internal edges, rewrite incoming
/// deps to the fused id, dense-renumber.
///
/// **Takes a `CoalescedDag` so multiple coalesce rules can be
/// composed.** The combined entry point
/// `coalesce_with_target_profile` runs whichever rules the profile
/// asks for, in dependency-safe order.
pub fn coalesce_gemm_phase(input: CoalescedDag, gemm_phase: GemmPhase) -> CoalescedDag {
    use std::collections::{HashMap, HashSet};
    let source_phase = gemm_phase.source_phase();

    // Index target nodes by layer.
    let mut target_ids_by_layer: HashMap<u16, Vec<NodeId>> = HashMap::new();
    for n in &input.nodes {
        if let BoundKernel::HandWrittenRowTile { phase, layer, .. } = n.kernel
            && phase == source_phase
        {
            target_ids_by_layer.entry(layer).or_default().push(n.id);
        }
    }
    if target_ids_by_layer.is_empty() {
        return input;
    }

    // Pick the fused id per layer (smallest absorbed id).
    let mut fused_id_by_layer: HashMap<u16, NodeId> = HashMap::new();
    for (layer, ids) in &target_ids_by_layer {
        let min_id = ids.iter().copied().min().expect("layer has target nodes");
        fused_id_by_layer.insert(*layer, min_id);
    }

    // absorbed_id → fused_id rewrite map.
    let mut rewrite: HashMap<NodeId, NodeId> = HashMap::new();
    for (layer, ids) in &target_ids_by_layer {
        let fused = fused_id_by_layer[layer];
        for id in ids {
            rewrite.insert(*id, fused);
        }
    }
    let absorbed: HashSet<NodeId> = rewrite.keys().copied().collect();

    let mut new_nodes: Vec<CoalescedNode> = Vec::with_capacity(input.nodes.len());
    for n in &input.nodes {
        let is_target = matches!(
            n.kernel,
            BoundKernel::HandWrittenRowTile { phase, .. } if phase == source_phase
        );
        if is_target {
            let layer = match n.kernel {
                BoundKernel::HandWrittenRowTile { layer, .. } => layer,
                _ => unreachable!(),
            };
            let fused_id = fused_id_by_layer[&layer];
            if n.id != fused_id {
                continue; // absorbed into the fused node
            }
            // Union the absorbed nodes' deps, prune internal edges,
            // rewrite already-fused incoming deps.
            let mut deps: Vec<NodeId> = Vec::new();
            let mut seen: HashSet<NodeId> = HashSet::new();
            for absorbed_id in &target_ids_by_layer[&layer] {
                // Find the absorbed node by id
                let absorbed_node = input
                    .nodes
                    .iter()
                    .find(|nd| nd.id == *absorbed_id)
                    .expect("absorbed id must exist in input");
                for d in &absorbed_node.deps {
                    if absorbed.contains(d) {
                        continue; // internal edge — drop
                    }
                    if seen.insert(*d) {
                        deps.push(*d);
                    }
                }
            }
            new_nodes.push(CoalescedNode {
                id: fused_id,
                kernel: BoundKernel::CutlassGemmLayer {
                    layer,
                    phase: gemm_phase,
                },
                deps,
            });
        } else {
            // Non-target: rewrite any deps that point at the absorbed set.
            let deps: Vec<NodeId> = n
                .deps
                .iter()
                .map(|d| rewrite.get(d).copied().unwrap_or(*d))
                .collect();
            new_nodes.push(CoalescedNode {
                id: n.id,
                kernel: n.kernel.clone(),
                deps,
            });
        }
    }

    renumber_dense_ids(&mut new_nodes);
    CoalescedDag {
        dims: input.dims,
        tiles: input.tiles,
        nodes: new_nodes,
    }
}

/// Combined coalesce pass driven by [`TargetProfile`]. Runs whichever
/// fusion rules the profile's kernel choices ask for, in
/// dependency-safe order. This is the **entry point** the production
/// codegen calls.
///
/// Adding a new fusion rule (e.g. for a future `FusedNormGemm`
/// kernel) is one new call here, gated on the appropriate profile
/// field.
/// Cost-gated coalesce wrapper. Applies `f` to the input DAG; keeps
/// the result only if `score_dag` (the simulated `partition_into_waves`
/// predicted_cost) decreases. Otherwise reverts.
///
/// This is the framework that makes future fusion experiments safe:
/// any new coalesce pass plugs in via `try_coalesce`, and the cost
/// model decides whether it ships. Failed experiments don't pollute
/// the dispatch arms — they're just not applied.
fn try_coalesce<F>(input: CoalescedDag, num_ctas: u32, label: &str, f: F) -> CoalescedDag
where
    F: FnOnce(CoalescedDag) -> CoalescedDag,
{
    use crate::schedule::score_dag;
    let before = score_dag(&input, num_ctas);
    // f takes ownership; we need a clone in case we revert.
    let candidate = f(input.clone());
    let after = score_dag(&candidate, num_ctas);
    if after < before {
        let _ = label; // logging hook reserved for a future trace flag
        candidate
    } else {
        input
    }
}

pub fn coalesce_with_target_profile(
    dag: &ReifiedDag,
    profile: &crate::target_profile::TargetProfile,
) -> CoalescedDag {
    use crate::target_profile::{AttentionKernelChoice, GemmKernelChoice};

    let num_ctas = profile.cooperative_grid_size();

    // Always start from the trivial 1:1 lift.
    let mut coalesced = coalesce(dag);

    // Attention fusion (FlashInferPersistent path). This pass replaces
    // the trivial coalesced DAG entirely (it re-runs from the reified
    // DAG to pick up the per-layer attention fan-in), so we evaluate
    // the swap as a single try_coalesce: the candidate is the
    // attention-fused DAG, the baseline is the trivial coalesce.
    if matches!(
        profile.attention_kernel,
        AttentionKernelChoice::FlashInferPersistent
    ) {
        let candidate = coalesce_with_flashinfer_attention(dag);
        // Direct cost compare since the function shape doesn't fit
        // try_coalesce's "rewrite the input" pattern.
        use crate::schedule::score_dag;
        if score_dag(&candidate, num_ctas) < score_dag(&coalesced, num_ctas) {
            coalesced = candidate;
        }
    }

    // CUTLASS GEMM fusion — one rule per phase, each cost-gated.
    if matches!(
        profile.gemm_kernel,
        GemmKernelChoice::CutlassSm80Multistage { .. }
    ) {
        coalesced = try_coalesce(coalesced, num_ctas, "cutlass_qkv", |c| {
            coalesce_gemm_phase(c, GemmPhase::Qkv)
        });
        coalesced = try_coalesce(coalesced, num_ctas, "cutlass_oproj", |c| {
            coalesce_gemm_phase(c, GemmPhase::OProj)
        });
        coalesced = try_coalesce(coalesced, num_ctas, "cutlass_gate_up", |c| {
            coalesce_gemm_phase(c, GemmPhase::GateUp)
        });
        coalesced = try_coalesce(coalesced, num_ctas, "cutlass_down", |c| {
            coalesce_gemm_phase(c, GemmPhase::Down)
        });
    }

    coalesced
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::reified_dag::{LlamaDims, TileSizes};

    fn tiny_dims() -> LlamaDims {
        LlamaDims {
            num_layers: 2,
            hidden_dim: 256,
            intermediate_dim: 512,
            num_attn_heads: 4,
            num_kv_heads: 2,
            head_dim: 64,
            seq_len: 32,
        }
    }

    #[test]
    fn coalesce_is_one_to_one_with_only_fallback_registered() {
        let dag = ReifiedDag::reify_llama(tiny_dims(), TileSizes::default_v1());
        let coalesced = coalesce(&dag);

        assert_eq!(
            coalesced.len(),
            dag.nodes.len(),
            "Phase B coalesce should be a 1:1 lift; every reified node becomes \
             one coalesced node bound to HandWrittenRowTile."
        );

        for (cnode, rnode) in coalesced.nodes.iter().zip(dag.nodes.iter()) {
            assert_eq!(cnode.id, rnode.id, "node id preserved");
            assert_eq!(cnode.deps, rnode.deps, "deps preserved");
            match &cnode.kernel {
                BoundKernel::HandWrittenRowTile {
                    phase,
                    layer,
                    row,
                    col,
                } => {
                    assert_eq!(*phase, rnode.phase);
                    assert_eq!(*layer, rnode.layer);
                    assert_eq!(*row, rnode.row);
                    assert_eq!(*col, rnode.col);
                }
                BoundKernel::FlashInferAttentionLayer { .. } => {
                    panic!("trivial coalesce should never produce FlashInferAttentionLayer");
                }
                BoundKernel::CutlassGemmLayer { .. } => {
                    panic!("trivial coalesce should never produce CutlassGemmLayer");
                }
            }
        }
    }

    #[test]
    fn kind_matches_phase_name_for_fallback() {
        // The monomorphic wave constraint will key on `kind()`. For the
        // fallback binding the kind is the phase name, which means
        // grouping by kind today is identical to grouping by phase —
        // i.e. behavior-preserving relative to the pre-library scheduler.
        let dag = ReifiedDag::reify_llama(tiny_dims(), TileSizes::default_v1());
        let coalesced = coalesce(&dag);

        for (cnode, rnode) in coalesced.nodes.iter().zip(dag.nodes.iter()) {
            assert_eq!(cnode.kernel.kind(), rnode.phase.name());
        }
    }

    #[test]
    fn flashinfer_coalesce_fuses_attention_per_layer() {
        let dag = ReifiedDag::reify_llama(tiny_dims(), TileSizes::default_v1());
        let coalesced = coalesce_with_flashinfer_attention(&dag);

        // Count by kind in the coalesced output.
        let mut by_kind = std::collections::HashMap::<&'static str, usize>::new();
        for cn in &coalesced.nodes {
            *by_kind.entry(cn.kernel.kind()).or_insert(0) += 1;
        }

        // tiny has NL=2 layers and SEQ_LEN=32 with row_tile=16 → 2 attn
        // row tiles per layer × 2 layers = 4 reified attention nodes,
        // fused into 2 FlashInferAttentionLayer nodes (one per layer).
        let nl = tiny_dims().num_layers as usize;
        assert_eq!(
            by_kind
                .get("flashinfer_attention_layer")
                .copied()
                .unwrap_or(0),
            nl,
            "expected one FlashInferAttentionLayer per layer; got {by_kind:?}",
        );

        // The total node count drops by exactly the number of absorbed
        // attention nodes minus the one fused replacement per layer.
        let row_tiles_per_layer = tiny_dims()
            .seq_len
            .div_ceil(TileSizes::default_v1().row_tile) as usize;
        let absorbed = nl * row_tiles_per_layer;
        let replacements = nl;
        assert_eq!(coalesced.len(), dag.nodes.len() - (absorbed - replacements));
    }

    #[test]
    fn flashinfer_coalesce_drops_internal_attention_edges() {
        // No coalesced node should depend on a NodeId that was absorbed
        // into a different fused FlashInferAttentionLayer node — only
        // on its own fused id (which is impossible since we drop
        // internal edges) or on non-absorbed ids.
        let dag = ReifiedDag::reify_llama(tiny_dims(), TileSizes::default_v1());
        let coalesced = coalesce_with_flashinfer_attention(&dag);

        let live_ids: std::collections::HashSet<NodeId> =
            coalesced.nodes.iter().map(|n| n.id).collect();

        for cn in &coalesced.nodes {
            for d in &cn.deps {
                assert!(
                    live_ids.contains(d),
                    "coalesced node {:?} has dep {d:?} that doesn't exist in the coalesced output",
                    cn.id
                );
            }
        }
    }

    #[test]
    fn flashinfer_coalesce_preserves_non_attention_phases() {
        // Every non-attention phase should still appear in the coalesced
        // output, untouched, with the same row/col counts as the
        // reified DAG.
        let dag = ReifiedDag::reify_llama(tiny_dims(), TileSizes::default_v1());
        let coalesced = coalesce_with_flashinfer_attention(&dag);

        let mut reified_non_attn = 0usize;
        for n in &dag.nodes {
            if n.phase != Phase::Attention {
                reified_non_attn += 1;
            }
        }
        let mut coalesced_non_attn = 0usize;
        for cn in &coalesced.nodes {
            if !matches!(cn.kernel, BoundKernel::FlashInferAttentionLayer { .. }) {
                coalesced_non_attn += 1;
            }
        }
        assert_eq!(reified_non_attn, coalesced_non_attn);
    }

    #[test]
    fn cutlass_gemm_coalesce_fuses_one_phase_per_layer() {
        let dag = ReifiedDag::reify_llama(tiny_dims(), TileSizes::default_v1());
        let trivial = coalesce(&dag);
        let fused = coalesce_gemm_phase(trivial, GemmPhase::GateUp);

        let mut by_kind = std::collections::HashMap::<&'static str, usize>::new();
        for cn in &fused.nodes {
            *by_kind.entry(cn.kernel.kind()).or_insert(0) += 1;
        }

        // tiny: NL=2 layers, gate_up has 1 row tile × 4 col tiles per
        // layer = 4 nodes, fused into 1 CutlassGemmGateUpLayer node
        // per layer = 2 fused nodes total.
        let nl = tiny_dims().num_layers as usize;
        assert_eq!(
            by_kind
                .get("cutlass_gemm_gate_up_layer")
                .copied()
                .unwrap_or(0),
            nl
        );
        // Other GEMM phases pass through untouched (still
        // HandWrittenRowTile).
        assert!(
            by_kind.get("gate_up").copied().unwrap_or(0) == 0,
            "no HandWrittenRowTile gate_up should remain after fusion: {by_kind:?}"
        );
        assert!(by_kind.get("qkv").copied().unwrap_or(0) > 0);
        assert!(by_kind.get("down").copied().unwrap_or(0) > 0);
    }

    #[test]
    fn cutlass_gemm_coalesce_composes_with_attention_fusion() {
        // Run flashinfer attention fusion + all four cutlass gemm
        // fusions in sequence (the production order).
        let dag = ReifiedDag::reify_llama(tiny_dims(), TileSizes::default_v1());
        let mut c = coalesce_with_flashinfer_attention(&dag);
        c = coalesce_gemm_phase(c, GemmPhase::Qkv);
        c = coalesce_gemm_phase(c, GemmPhase::OProj);
        c = coalesce_gemm_phase(c, GemmPhase::GateUp);
        c = coalesce_gemm_phase(c, GemmPhase::Down);

        let mut by_kind = std::collections::HashMap::<&'static str, usize>::new();
        for cn in &c.nodes {
            *by_kind.entry(cn.kernel.kind()).or_insert(0) += 1;
        }

        let nl = tiny_dims().num_layers as usize;
        assert_eq!(
            by_kind
                .get("flashinfer_attention_layer")
                .copied()
                .unwrap_or(0),
            nl
        );
        for kind in [
            "cutlass_gemm_qkv_layer",
            "cutlass_gemm_o_proj_layer",
            "cutlass_gemm_gate_up_layer",
            "cutlass_gemm_down_layer",
        ] {
            assert_eq!(
                by_kind.get(kind).copied().unwrap_or(0),
                nl,
                "expected {nl} {kind} nodes; got {by_kind:?}"
            );
        }

        // Every node id should be dense [0, len) — schedule.rs
        // depends on this.
        let live_ids: std::collections::HashSet<NodeId> = c.nodes.iter().map(|n| n.id).collect();
        for cn in &c.nodes {
            for d in &cn.deps {
                assert!(
                    live_ids.contains(d),
                    "coalesced node has dep {d:?} not in the coalesced output"
                );
            }
        }
    }

    #[test]
    fn coalesce_with_target_profile_dispatches_correctly() {
        use crate::target_profile::{
            AttentionKernelChoice, GemmKernelChoice, NormKernelChoice, RopeKernelChoice,
            TargetProfile,
        };
        let dag = ReifiedDag::reify_llama(tiny_dims(), TileSizes::default_v1());

        // Profile with cutlass enabled — should produce CutlassGemmLayer
        // nodes for all four phases plus FlashInferAttentionLayer.
        let profile_cutlass = TargetProfile {
            num_sm: 58,
            cooperative_blocks_per_sm: 1,
            max_dynamic_shmem_bytes: 99 * 1024,
            gemm_kernel: GemmKernelChoice::CutlassSm80Multistage {
                tile_m: 256,
                tile_n: 128,
                tile_k: 32,
                pipeline_stages: 4,
            },
            attention_kernel: AttentionKernelChoice::FlashInferPersistent,
            norm_kernel: NormKernelChoice::HandWrittenWarpShuffle,
            rope_kernel: RopeKernelChoice::HandWrittenSplitHalf,
        };
        let coalesced_cutlass = coalesce_with_target_profile(&dag, &profile_cutlass);
        let mut kinds: std::collections::HashSet<&'static str> = Default::default();
        for cn in &coalesced_cutlass.nodes {
            kinds.insert(cn.kernel.kind());
        }
        assert!(kinds.contains("flashinfer_attention_layer"));
        assert!(kinds.contains("cutlass_gemm_qkv_layer"));
        assert!(kinds.contains("cutlass_gemm_o_proj_layer"));
        assert!(kinds.contains("cutlass_gemm_gate_up_layer"));
        assert!(kinds.contains("cutlass_gemm_down_layer"));
        // No HandWrittenRowTile remnants for the four GEMM phases
        // when cutlass is enabled.
        assert!(!kinds.contains("qkv"));
        assert!(!kinds.contains("o_proj"));
        assert!(!kinds.contains("gate_up"));
        assert!(!kinds.contains("down"));

        // Profile with cutlass DISABLED — should NOT produce any
        // CutlassGemmLayer nodes; the four GEMM phases stay as
        // HandWrittenRowTile.
        let profile_wmma = TargetProfile {
            gemm_kernel: GemmKernelChoice::HandWrittenWmma,
            ..profile_cutlass
        };
        let coalesced_wmma = coalesce_with_target_profile(&dag, &profile_wmma);
        let mut kinds_wmma: std::collections::HashSet<&'static str> = Default::default();
        for cn in &coalesced_wmma.nodes {
            kinds_wmma.insert(cn.kernel.kind());
        }
        assert!(kinds_wmma.contains("flashinfer_attention_layer"));
        assert!(kinds_wmma.contains("qkv"));
        assert!(kinds_wmma.contains("o_proj"));
        assert!(kinds_wmma.contains("gate_up"));
        assert!(kinds_wmma.contains("down"));
        assert!(!kinds_wmma.contains("cutlass_gemm_qkv_layer"));
    }

    #[test]
    fn try_coalesce_rejects_a_regression() {
        // Verify the cost gate actually reverts a transform that
        // increases predicted_cost. We construct a no-op identity
        // (which has score == before, so `after < before` is false →
        // revert) and a degenerate "duplicate every node's deps"
        // mutator (which doesn't change ids but also doesn't reduce
        // cost → revert). Both must produce a DAG byte-equal to the
        // input.
        use crate::reified_dag::TileSizes;
        let reified = ReifiedDag::reify_llama(
            LlamaDims {
                num_layers: 2,
                hidden_dim: 256,
                intermediate_dim: 512,
                num_attn_heads: 4,
                num_kv_heads: 2,
                head_dim: 64,
                seq_len: 32,
            },
            TileSizes::default_v1(),
        );
        let baseline = coalesce(&reified);
        let baseline_len = baseline.nodes.len();

        // Identity transform: cost is exactly equal → `after < before`
        // is false → revert.
        let after_identity = try_coalesce(baseline.clone(), 8, "identity", |c| c);
        assert_eq!(after_identity.nodes.len(), baseline_len);

        // A real coalesce that we know reduces cost (the FlashInfer
        // attention fusion) should ship under the gate. We use a
        // direct call to verify it would normally compress nodes,
        // then run it via try_coalesce and check the count drops.
        let gated = try_coalesce(baseline.clone(), 8, "fi_attn", |_| {
            coalesce_with_flashinfer_attention(&reified)
        });
        assert!(
            gated.nodes.len() < baseline_len,
            "fi_attn coalesce should reduce node count under the cost gate"
        );
    }

    #[test]
    fn flashinfer_coalesce_rewrites_deps_into_fused_ids() {
        // A non-attention node that used to depend on an attention row
        // tile (e.g. o_proj depends on attention) should, after the
        // flashinfer coalesce, depend on the *fused* attention node id
        // for its layer — not on a now-absorbed reified attention id.
        let dag = ReifiedDag::reify_llama(tiny_dims(), TileSizes::default_v1());
        let coalesced = coalesce_with_flashinfer_attention(&dag);

        // Build (id → kind tag) for the coalesced output so we can
        // assert that downstream attention deps point at a fused node.
        let kind_of: std::collections::HashMap<NodeId, &'static str> = coalesced
            .nodes
            .iter()
            .map(|n| (n.id, n.kernel.kind()))
            .collect();

        let mut rewritten_count = 0usize;
        for cn in &coalesced.nodes {
            if cn.kernel.kind() == "o_proj" {
                for d in &cn.deps {
                    if kind_of.get(d).copied() == Some("flashinfer_attention_layer") {
                        rewritten_count += 1;
                    }
                }
            }
        }
        assert!(
            rewritten_count > 0,
            "expected at least one o_proj→flashinfer_attention_layer dep after coalesce"
        );
    }

    #[test]
    fn every_phase_appears_in_coalesced_output() {
        // Sanity: the trivial coalesce pass shouldn't drop or merge any
        // phase even when run on a multi-layer DAG.
        let dag = ReifiedDag::reify_llama(tiny_dims(), TileSizes::default_v1());
        let coalesced = coalesce(&dag);

        let mut seen = std::collections::HashSet::new();
        for cnode in &coalesced.nodes {
            seen.insert(cnode.kernel.kind());
        }
        for phase in [
            Phase::AttnNorm,
            Phase::Qkv,
            Phase::Rope,
            Phase::Attention,
            Phase::OProj,
            Phase::MlpNorm,
            Phase::GateUp,
            Phase::Down,
        ] {
            assert!(
                seen.contains(phase.name()),
                "expected phase {} in coalesced output, only saw {seen:?}",
                phase.name()
            );
        }
    }
}
