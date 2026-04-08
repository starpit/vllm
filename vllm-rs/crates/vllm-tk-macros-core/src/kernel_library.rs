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
            // real cost — Phase C2b will pivot this to a real measured
            // number once we have a bench. Treating each layer as a
            // single big work unit is the right thing for the LPT bin
            // packer either way: a flashinfer-attention wave consumes
            // all the CTAs at once, not by per-CTA partitioning, so the
            // wave's makespan is determined by this rolled-up cost
            // regardless of the per-CTA breakdown.
            BoundKernel::FlashInferAttentionLayer { .. } => {
                let row_tiles = model.seq_len.div_ceil(model.row_tile);
                row_tiles * model.cost(Phase::Attention)
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

    // ── Step 5: renumber NodeIds to be dense [0, nodes.len()). ──
    // The schedule + codegen passes index into `dag.nodes[NodeId.0 as
    // usize]` and assume `NodeId == array index`. Fusion creates gaps
    // (absorbed ids 1..N-1 disappear, leaving a hole) and the array
    // shrinks, so without renumbering the surviving ids point past
    // the end of the new array. Build an old_id → new_id remap from
    // the post-fusion order, then rewrite every node's id and every
    // dep through it.
    let mut id_remap: HashMap<NodeId, NodeId> = HashMap::with_capacity(nodes.len());
    for (new_idx, n) in nodes.iter().enumerate() {
        id_remap.insert(n.id, NodeId(new_idx as u32));
    }
    for n in &mut nodes {
        n.id = id_remap[&n.id];
        for d in &mut n.deps {
            *d = *id_remap
                .get(d)
                .expect("dep references a node that doesn't exist in coalesced output");
        }
    }

    CoalescedDag {
        dims: dag.dims,
        tiles: dag.tiles,
        nodes,
    }
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
