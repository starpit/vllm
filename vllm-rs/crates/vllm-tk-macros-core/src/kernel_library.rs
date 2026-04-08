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

use crate::reified_dag::{NodeId, Phase, ReifiedDag};

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
    CoalescedDag { nodes }
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
