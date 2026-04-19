// SPDX-License-Identifier: Apache-2.0
//! Paged-KV decode instantiation of the Attn template. Proves the
//! template is genuinely parametric: same node/edge structure as
//! FA2 prefill, but with a gather-capable K/V load address and a
//! per-sequence KV bound. Exercises `AxisDivGather`, `AxisModStride`,
//! and `Bound::IndexedScalar` — all vocabulary that prefill didn't
//! touch.

use ferrite_stencil::ir::{self, AddrTerm, Bound, DepKind, Role};
use ferrite_stencil::{PagedDecodeParams, Window, attn_region_paged_decode, print, sm90_fa2};

fn params() -> PagedDecodeParams {
    PagedDecodeParams {
        head_dim: 128,
        tile_k: 64,
        num_head_groups: 8,
        pipe: 3,
        tokens_per_page: 256,
        window: Window::Infinite,
    }
}

#[test]
fn region_validates() {
    let r = attn_region_paged_decode(&params());
    ir::validate(&r).expect("decode region validates");
}

#[test]
fn structure_matches_prefill_skeleton() {
    let r = attn_region_paged_decode(&params());

    // Same 7 nodes: 3 Load + 3 Compute + 1 Store.
    let (n_load, n_comp, n_store) = (
        r.nodes.iter().filter(|n| n.role == Role::Load).count(),
        r.nodes.iter().filter(|n| n.role == Role::Compute).count(),
        r.nodes.iter().filter(|n| n.role == Role::Store).count(),
    );
    assert_eq!((n_load, n_comp, n_store), (3, 3, 1));

    // Same edges: 2 Pipeline + 4 Raw + 1 softmax self-loop = 7.
    assert_eq!(r.edges.len(), 7);

    // No causal predicate (M=1 makes it trivial).
    assert_eq!(r.domain.predicates.len(), 0);
}

#[test]
fn k_load_uses_gather() {
    let r = attn_region_paged_decode(&params());
    let k = r.nodes.iter().find(|n| n.op.tag == "load_k_tile").unwrap();
    let addr = k.addr.as_ref().expect("K load has address");

    let has_gather = addr
        .terms
        .iter()
        .any(|t| matches!(t, AddrTerm::AxisDivGather { .. }));
    let has_mod = addr
        .terms
        .iter()
        .any(|t| matches!(t, AddrTerm::AxisModStride { .. }));
    assert!(has_gather, "K load must use AxisDivGather for paged KV");
    assert!(
        has_mod,
        "K load must resolve within-page offset via AxisModStride"
    );
}

#[test]
fn kv_tile_is_indexed_per_batch() {
    let r = attn_region_paged_decode(&params());
    let kv = r.domain.axes.iter().find(|a| a.name == "kv_tile").unwrap();
    assert!(
        matches!(kv.bound, Bound::IndexedScalar(_, _)),
        "kv_tile bound must be IndexedScalar for per-sequence KV length, got {:?}",
        kv.bound
    );
}

#[test]
fn sm90_prints_same_warpgroup_layout() {
    // Proves arch mapping is genuinely region-independent: the
    // SM90 table written for FA2 prefill drops onto the decode
    // region unchanged.
    let r = attn_region_paged_decode(&params());
    let s = print::print_region(&r, &sm90_fa2());
    assert!(s.contains("wg[loader×1]"));
    assert!(s.contains("wg[consumer×3]"));
    assert!(s.contains("wg[storer×1]"));
    // Pipeline edge still lowers to NamedSem — same as prefill.
    assert!(s.contains("Pipeline NamedSem"));
}

#[test]
fn internal_edges_are_raw_or_pipeline() {
    let r = attn_region_paged_decode(&params());
    for e in &r.edges {
        assert!(
            matches!(e.kind, DepKind::Raw | DepKind::Pipeline),
            "edge {}→{} leaked region-boundary dep kind: {:?}",
            e.src,
            e.dst,
            e.kind
        );
    }
}
