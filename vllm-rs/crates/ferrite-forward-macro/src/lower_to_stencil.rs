// SPDX-License-Identifier: Apache-2.0
//! FUF → Stencil IR lowering, v1.
//!
//! Consumes the solver's `Assignment` (tile → subgraph → Impl) and
//! produces a `Megakernel` of stencil `Region`s. For v1 we handle
//! the two attention impls — `AttentionPrefillContiguousImpl` and
//! `AttentionViaCacheImpl` — which both claim a singleton
//! `OpKind::Attention` tile. Everything else is out of scope and
//! returns an error: the point of this commit is to prove the
//! pipeline connects end-to-end on the target stress case, not to
//! replace codegen.
//!
//! The lowering is pattern-match on `Implementation::name()`. Param
//! inference (head_dim, tile_q, tile_k, pipe) is passed in by the
//! caller for now; shape-driven inference from the FUF arrives when
//! we extend beyond singleton attention.
//!
//! Integration: slot is between `solver::solve()` and `codegen::emit_model()`
//! in `lib.rs`. This commit adds the pass itself + unit tests; the
//! wire-up into the macro drive is deferred to the next commit,
//! because changing codegen to consume `Megakernel` is a bigger
//! edit than the lowering itself.

#![allow(dead_code)]

use ferrite_stencil::{
    AttnParams, Megakernel, PagedDecodeParams, Region, Window, attn_region,
    attn_region_paged_decode,
};

use crate::fuf::{Fuf, TileId};
use crate::impl_lib::ImplementationLibrary;
use crate::solver::{Assignment, SubgraphId};

/// Outcome of a tolerant pass over an `Assignment`: every subgraph
/// whose Impl has a region template is lowered; every subgraph whose
/// Impl does not is skipped (not an error). Used by the parallel
/// wire-up in `lib.rs` so the new pipeline runs on real models
/// without blocking on templates for every non-attention Impl.
#[derive(Debug)]
pub struct LowerReport {
    pub mk: Megakernel,
    pub supported_subgraphs: usize,
    /// Subgraphs that had no region template, with the offending
    /// impl name for telemetry.
    pub skipped: Vec<(SubgraphId, &'static str)>,
}

/// Hardware-facing tile/pipe parameters the lowering can't yet
/// infer from the FUF. Pass-through for v1; later commits will
/// derive these from FUF shapes + solver cost hints.
#[derive(Clone, Copy, Debug)]
pub struct LowerHints {
    pub head_dim: u32,
    pub num_head_groups: u32,
    pub tile_q: u32,
    pub tile_k: u32,
    pub pipe: u32,
    pub tokens_per_page: u32,
}

impl Default for LowerHints {
    fn default() -> Self {
        Self {
            head_dim: 128,
            num_head_groups: 8,
            tile_q: 128,
            tile_k: 64,
            pipe: 3,
            tokens_per_page: 256,
        }
    }
}

impl LowerHints {
    /// Pull the two bounds-derivable fields (`head_dim`,
    /// `num_head_groups` = num_q_heads / num_kv_heads) off the
    /// model's bounds table. The remaining fields — `tile_q`,
    /// `tile_k`, `pipe`, `tokens_per_page` — still come from
    /// `Default`; they belong on the Impl (tile tuning) and on the
    /// KvCachePool config respectively. See `STENCIL_IR_STATUS.md`
    /// item #2 for the remaining work.
    pub fn from_model_bounds(bounds: &std::collections::BTreeMap<String, u64>) -> Self {
        let default = Self::default();
        let head_dim = bounds
            .get("head_dim")
            .copied()
            .map(|v| v as u32)
            .unwrap_or(default.head_dim);
        let num_q = bounds.get("num_attention_heads").copied();
        let num_kv = bounds.get("num_key_value_heads").copied();
        let num_head_groups = match (num_q, num_kv) {
            (Some(q), Some(kv)) if kv > 0 => ((q / kv).max(1)) as u32,
            _ => default.num_head_groups,
        };
        Self {
            head_dim,
            num_head_groups,
            ..default
        }
    }
}

#[derive(Debug)]
pub enum LowerError {
    UnsupportedImpl { name: &'static str },
    EmptyAssignment,
    SubgraphWithoutTiles { sg: SubgraphId },
}

impl std::fmt::Display for LowerError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            LowerError::UnsupportedImpl { name } => {
                write!(f, "lowering has no region template for impl {:?}", name)
            }
            LowerError::EmptyAssignment => write!(f, "assignment has no subgraphs"),
            LowerError::SubgraphWithoutTiles { sg } => {
                write!(f, "subgraph {:?} claims no tiles", sg)
            }
        }
    }
}

impl std::error::Error for LowerError {}

pub fn lower_assignment(
    fuf: &Fuf,
    assignment: &Assignment,
    library: &ImplementationLibrary,
    hints: &LowerHints,
) -> Result<Megakernel, LowerError> {
    if assignment.num_subgraphs() == 0 {
        return Err(LowerError::EmptyAssignment);
    }

    let mut subgraphs: Vec<SubgraphId> = assignment.subgraphs().collect();
    subgraphs.sort();

    let mut regions: Vec<Region> = Vec::new();

    for sg in subgraphs {
        let tiles = assignment.tiles_in_subgraph(sg);
        if tiles.is_empty() {
            return Err(LowerError::SubgraphWithoutTiles { sg });
        }
        let impl_id = assignment
            .impl_of(sg)
            .expect("subgraph with tiles has impl");
        let imp = library.get(impl_id);
        let region = lower_impl(imp.name(), &tiles, fuf, hints)?;
        regions.push(region);
    }

    Ok(Megakernel {
        regions,
        control: Vec::new(),
    })
}

/// Tolerant variant: lowers every supported subgraph, records the
/// rest as skipped. Used by the macro drive while non-attention
/// region templates are still being written.
pub fn lower_assignment_partial(
    fuf: &Fuf,
    assignment: &Assignment,
    library: &ImplementationLibrary,
    hints: &LowerHints,
) -> LowerReport {
    let mut regions: Vec<Region> = Vec::new();
    let mut skipped: Vec<(SubgraphId, &'static str)> = Vec::new();
    let mut supported = 0usize;

    let mut subgraphs: Vec<SubgraphId> = assignment.subgraphs().collect();
    subgraphs.sort();

    for sg in subgraphs {
        let tiles = assignment.tiles_in_subgraph(sg);
        if tiles.is_empty() {
            continue;
        }
        let Some(impl_id) = assignment.impl_of(sg) else {
            continue;
        };
        let imp = library.get(impl_id);
        match lower_impl(imp.name(), &tiles, fuf, hints) {
            Ok(region) => {
                regions.push(region);
                supported += 1;
            }
            Err(LowerError::UnsupportedImpl { name }) => {
                skipped.push((sg, name));
            }
            Err(_) => {}
        }
    }

    LowerReport {
        mk: Megakernel {
            regions,
            control: Vec::new(),
        },
        supported_subgraphs: supported,
        skipped,
    }
}

fn lower_impl(
    impl_name: &'static str,
    _tiles: &[TileId],
    _fuf: &Fuf,
    hints: &LowerHints,
) -> Result<Region, LowerError> {
    match impl_name {
        "attention_prefill_contiguous" => Ok(attn_region(&AttnParams {
            window: Window::Infinite,
            head_dim: hints.head_dim,
            tile_q: hints.tile_q,
            tile_k: hints.tile_k,
            num_head_groups: hints.num_head_groups,
            pipe: hints.pipe,
        })),
        "attention_via_cache" => Ok(attn_region_paged_decode(&PagedDecodeParams {
            head_dim: hints.head_dim,
            tile_k: hints.tile_k,
            num_head_groups: hints.num_head_groups,
            pipe: hints.pipe,
            tokens_per_page: hints.tokens_per_page,
        })),
        // Sliding-window prefill is the same Attn(W=finite) shape.
        "sliding_attention_prefill_contiguous" => Ok(attn_region(&AttnParams {
            window: Window::Finite(4096),
            head_dim: hints.head_dim,
            tile_q: hints.tile_q,
            tile_k: hints.tile_k,
            num_head_groups: hints.num_head_groups,
            pipe: hints.pipe,
        })),
        other => Err(LowerError::UnsupportedImpl { name: other }),
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use ferrite_stencil::ir;

    use super::*;
    use crate::classified::OpKind;
    use crate::fuf::{FufNode, TileId};
    use crate::impl_lib::{
        AttentionPrefillContiguousImpl, AttentionViaCacheImpl, ImplementationLibrary,
    };
    use crate::solver::{Assignment, SubgraphId};

    fn singleton_attention_fuf() -> Fuf {
        Fuf {
            nodes: vec![FufNode {
                id: TileId(0),
                op: OpKind::Attention,
                inputs: vec![],
                outputs: vec![],
            }],
        }
    }

    fn singleton_assignment(impl_id: crate::impl_lib::ImplId) -> Assignment {
        let sg = SubgraphId(0);
        let mut cover = HashMap::new();
        cover.insert(TileId(0), sg);
        let mut impls = HashMap::new();
        impls.insert(sg, impl_id);
        Assignment {
            cover,
            impls,
            predicted_us: 0.0,
        }
    }

    #[test]
    fn lowers_attention_prefill_to_fa2_region() {
        let fuf = singleton_attention_fuf();
        let mut lib = ImplementationLibrary::new();
        let impl_id = lib.push(Box::new(AttentionPrefillContiguousImpl));
        let assignment = singleton_assignment(impl_id);

        let mk = lower_assignment(&fuf, &assignment, &lib, &LowerHints::default())
            .expect("lowering succeeds");

        assert_eq!(mk.regions.len(), 1);
        let r = &mk.regions[0];
        assert_eq!(r.name, "fa2_prefill");
        assert_eq!(r.nodes.len(), 7);
        ir::validate(r).expect("lowered region validates");

        // Matches the hand-authored template byte-for-byte on structure.
        let hand = attn_region(&AttnParams {
            window: Window::Infinite,
            head_dim: 128,
            tile_q: 128,
            tile_k: 64,
            num_head_groups: 8,
            pipe: 3,
        });
        assert_eq!(r.nodes.len(), hand.nodes.len());
        assert_eq!(r.edges.len(), hand.edges.len());
        assert_eq!(r.domain.axes.len(), hand.domain.axes.len());
        assert_eq!(r.domain.predicates.len(), hand.domain.predicates.len());
    }

    #[test]
    fn lowers_attention_via_cache_to_paged_decode() {
        let fuf = singleton_attention_fuf();
        let mut lib = ImplementationLibrary::new();
        let impl_id = lib.push(Box::new(AttentionViaCacheImpl));
        let assignment = singleton_assignment(impl_id);

        let mk = lower_assignment(&fuf, &assignment, &lib, &LowerHints::default())
            .expect("lowering succeeds");
        assert_eq!(mk.regions.len(), 1);
        assert_eq!(mk.regions[0].name, "paged_decode");
        ir::validate(&mk.regions[0]).expect("decode region validates");
    }

    #[test]
    fn unsupported_impl_returns_error() {
        // Use the name of an impl we haven't templated yet to force
        // the error path. The impl itself doesn't need to exist in
        // the library for this test — we construct a tiny stub.
        use crate::impl_lib::Implementation;

        #[derive(Debug, Default)]
        struct FakeGemm;
        impl Implementation for FakeGemm {
            fn name(&self) -> &'static str {
                "gemm_rowmajor"
            }
            fn target_compatible(&self, _: &crate::target::TargetProfile) -> bool {
                true
            }
            fn workload_constraint(&self) -> crate::impl_lib::WorkloadConstraint {
                crate::impl_lib::WorkloadConstraint::Any
            }
            fn matches(
                &self,
                _: &Fuf,
                _: TileId,
                _: &crate::target::TargetProfile,
            ) -> Option<crate::impl_lib::MatchInfo> {
                None
            }
            fn cost_us(&self, _: &crate::impl_lib::MatchInfo, _: &crate::impl_lib::CostCtx) -> f64 {
                0.0
            }
            fn resources(&self, _: &crate::impl_lib::MatchInfo) -> crate::impl_lib::Resources {
                crate::impl_lib::Resources::ZERO
            }
            fn launch_kind(&self) -> crate::impl_lib::LaunchKind {
                crate::impl_lib::LaunchKind::HostCallback
            }
            fn supported_input_handoffs(&self) -> &[crate::impl_lib::Handoff] {
                &[]
            }
            fn supported_output_handoffs(&self) -> &[crate::impl_lib::Handoff] {
                &[]
            }
            fn input_layouts(
                &self,
                _: &crate::impl_lib::MatchInfo,
            ) -> Vec<crate::impl_lib::Layout> {
                vec![]
            }
            fn output_layouts(
                &self,
                _: &crate::impl_lib::MatchInfo,
            ) -> Vec<crate::impl_lib::Layout> {
                vec![]
            }
            fn is_compute_bound(&self) -> bool {
                false
            }
            fn emit_call(&self, _: &crate::emit::EmitCtx) -> proc_macro2::TokenStream {
                proc_macro2::TokenStream::new()
            }
        }

        let fuf = singleton_attention_fuf();
        let mut lib = ImplementationLibrary::new();
        let impl_id = lib.push(Box::new(FakeGemm));
        let assignment = singleton_assignment(impl_id);

        let err = lower_assignment(&fuf, &assignment, &lib, &LowerHints::default())
            .expect_err("unsupported impl must fail");
        match err {
            LowerError::UnsupportedImpl { name } => {
                assert_eq!(name, "gemm_rowmajor");
            }
            other => panic!("wrong error variant: {:?}", other),
        }
    }

    #[test]
    fn lowered_region_has_valid_structure_for_scheduling() {
        let fuf = singleton_attention_fuf();
        let mut lib = ImplementationLibrary::new();
        let impl_id = lib.push(Box::new(AttentionPrefillContiguousImpl));
        let assignment = singleton_assignment(impl_id);
        let mk = lower_assignment(&fuf, &assignment, &lib, &LowerHints::default()).unwrap();
        let r = &mk.regions[0];

        // The axis classifier + topo-sort primitives run cleanly on
        // a lowered region — proves the output feeds the scheduler
        // substrate we already have.
        let classes = ferrite_stencil::classify_axes(r);
        assert_eq!(classes.len(), 3);
        assert_eq!(ferrite_stencil::region_pipeline_depth(r), 3);
        let order = ferrite_stencil::topo_order_within_iter(r);
        assert_eq!(order.len(), r.nodes.len());
    }

    #[test]
    fn hints_from_bounds_derives_head_dim_and_gqa_groups() {
        use std::collections::BTreeMap;

        // Llama-2-7B: MHA, 32 Q heads = 32 KV heads → groups = 1.
        let mut mha = BTreeMap::new();
        mha.insert("head_dim".into(), 128);
        mha.insert("num_attention_heads".into(), 32);
        mha.insert("num_key_value_heads".into(), 32);
        let h = LowerHints::from_model_bounds(&mha);
        assert_eq!(h.head_dim, 128);
        assert_eq!(h.num_head_groups, 1);

        // Llama-3-70B: GQA 64 Q / 8 KV → groups = 8.
        let mut gqa = BTreeMap::new();
        gqa.insert("head_dim".into(), 128);
        gqa.insert("num_attention_heads".into(), 64);
        gqa.insert("num_key_value_heads".into(), 8);
        let h = LowerHints::from_model_bounds(&gqa);
        assert_eq!(h.num_head_groups, 8);

        // Missing bounds fall back to Default without panicking.
        let empty = BTreeMap::new();
        let h = LowerHints::from_model_bounds(&empty);
        assert_eq!(h.head_dim, LowerHints::default().head_dim);
        assert_eq!(h.num_head_groups, LowerHints::default().num_head_groups);
    }

    #[test]
    fn partial_lowering_skips_unsupported_subgraphs() {
        // Assignment mixes one supported (attention) and one unsupported
        // (fake gemm) subgraph. Partial lowering should lower the
        // attention one and record the other as skipped.
        use crate::impl_lib::Implementation;

        #[derive(Debug, Default)]
        struct FakeGemm;
        impl Implementation for FakeGemm {
            fn name(&self) -> &'static str {
                "gemm_rowmajor"
            }
            fn target_compatible(&self, _: &crate::target::TargetProfile) -> bool {
                true
            }
            fn workload_constraint(&self) -> crate::impl_lib::WorkloadConstraint {
                crate::impl_lib::WorkloadConstraint::Any
            }
            fn matches(
                &self,
                _: &Fuf,
                _: TileId,
                _: &crate::target::TargetProfile,
            ) -> Option<crate::impl_lib::MatchInfo> {
                None
            }
            fn cost_us(&self, _: &crate::impl_lib::MatchInfo, _: &crate::impl_lib::CostCtx) -> f64 {
                0.0
            }
            fn resources(&self, _: &crate::impl_lib::MatchInfo) -> crate::impl_lib::Resources {
                crate::impl_lib::Resources::ZERO
            }
            fn launch_kind(&self) -> crate::impl_lib::LaunchKind {
                crate::impl_lib::LaunchKind::HostCallback
            }
            fn supported_input_handoffs(&self) -> &[crate::impl_lib::Handoff] {
                &[]
            }
            fn supported_output_handoffs(&self) -> &[crate::impl_lib::Handoff] {
                &[]
            }
            fn input_layouts(
                &self,
                _: &crate::impl_lib::MatchInfo,
            ) -> Vec<crate::impl_lib::Layout> {
                vec![]
            }
            fn output_layouts(
                &self,
                _: &crate::impl_lib::MatchInfo,
            ) -> Vec<crate::impl_lib::Layout> {
                vec![]
            }
            fn is_compute_bound(&self) -> bool {
                false
            }
            fn emit_call(&self, _: &crate::emit::EmitCtx) -> proc_macro2::TokenStream {
                proc_macro2::TokenStream::new()
            }
        }

        let fuf = Fuf {
            nodes: vec![
                FufNode {
                    id: TileId(0),
                    op: OpKind::Attention,
                    inputs: vec![],
                    outputs: vec![],
                },
                FufNode {
                    id: TileId(1),
                    op: OpKind::Attention,
                    inputs: vec![],
                    outputs: vec![],
                },
            ],
        };
        let mut lib = ImplementationLibrary::new();
        let attn_id = lib.push(Box::new(AttentionPrefillContiguousImpl));
        let gemm_id = lib.push(Box::new(FakeGemm));
        let sg_attn = SubgraphId(0);
        let sg_gemm = SubgraphId(1);
        let mut cover = HashMap::new();
        cover.insert(TileId(0), sg_attn);
        cover.insert(TileId(1), sg_gemm);
        let mut impls = HashMap::new();
        impls.insert(sg_attn, attn_id);
        impls.insert(sg_gemm, gemm_id);
        let assignment = Assignment {
            cover,
            impls,
            predicted_us: 0.0,
        };

        let report = lower_assignment_partial(&fuf, &assignment, &lib, &LowerHints::default());
        assert_eq!(report.supported_subgraphs, 1);
        assert_eq!(report.mk.regions.len(), 1);
        assert_eq!(report.mk.regions[0].name, "fa2_prefill");
        assert_eq!(report.skipped.len(), 1);
        assert_eq!(report.skipped[0].1, "gemm_rowmajor");
    }

    #[test]
    fn end_to_end_lower_and_schedule_on_sm90() {
        // Capstone: prove lowering's output feeds the wavefront
        // scheduler, so the whole pipeline from solver Assignment to
        // per-CTA preamble/body/epilogue works without a hand-authored
        // Region in the middle.
        let fuf = singleton_attention_fuf();
        let mut lib = ImplementationLibrary::new();
        let impl_id = lib.push(Box::new(AttentionPrefillContiguousImpl));
        let assignment = singleton_assignment(impl_id);
        let mk = lower_assignment(&fuf, &assignment, &lib, &LowerHints::default()).unwrap();

        let region = &mk.regions[0];
        let sched = ferrite_stencil::schedule_wavefront(region, &ferrite_stencil::sm90_fa2())
            .expect("schedule succeeds on lowered region");

        assert_eq!(sched.pipeline_depth, 3);
        assert_eq!(sched.preamble.len(), 1, "preamble = load_q");
        assert_eq!(sched.body.len(), 5, "body = load_k, load_v, qk, sm, pv");
        assert_eq!(sched.epilogue.len(), 1, "epilogue = store_o");

        let tag = |n: u16| region.nodes[n as usize].op.tag;
        assert_eq!(tag(sched.preamble[0].node), "load_q_tile");
        assert_eq!(tag(sched.epilogue[0].node), "store_o_tile");

        // Pipeline loads in body carry iter_offset = P.
        for step in &sched.body {
            let t = tag(step.node);
            let expected = if t == "load_k_tile" || t == "load_v_tile" {
                3
            } else {
                0
            };
            assert_eq!(
                step.iter_offset, expected,
                "step {} iter_offset mismatch",
                t
            );
        }
    }
}
