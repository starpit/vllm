// SPDX-License-Identifier: Apache-2.0
//! Round-trip test: build Attn(W=∞) for FA2 prefill, validate,
//! map on SM90 and SM89, snapshot the printed output. Exercises
//! the full vocabulary (3 roles, 2 region-internal dep kinds) on
//! a realistic region shape.

use ferrite_stencil::ir::{self, DepKind, Role};
use ferrite_stencil::{AttnParams, Window, attn_region, print, sm89_fa2, sm90_fa2};

fn params() -> AttnParams {
    AttnParams {
        window: Window::Infinite,
        head_dim: 128,
        tile_q: 128,
        tile_k: 64,
        num_head_groups: 8,
        pipe: 3,
    }
}

#[test]
fn region_validates() {
    let r = attn_region(&params());
    ir::validate(&r).expect("region validates");
}

#[test]
fn structure_matches_sketch() {
    let r = attn_region(&params());

    // 3 axes, only causal predicate for W=∞.
    assert_eq!(r.domain.axes.len(), 3);
    assert_eq!(r.domain.predicates.len(), 1);

    // 7 nodes: 3 Load + 3 Compute + 1 Store.
    assert_eq!(r.nodes.len(), 7);
    let n_load = r.nodes.iter().filter(|n| n.role == Role::Load).count();
    let n_comp = r.nodes.iter().filter(|n| n.role == Role::Compute).count();
    let n_store = r.nodes.iter().filter(|n| n.role == Role::Store).count();
    assert_eq!((n_load, n_comp, n_store), (3, 3, 1));

    // Every internal edge is Raw or Pipeline.
    for e in &r.edges {
        assert!(matches!(e.kind, DepKind::Raw | DepKind::Pipeline));
    }

    // Exactly 2 Pipeline edges (K→QK, V→PV).
    let n_pipe = r
        .edges
        .iter()
        .filter(|e| e.kind == DepKind::Pipeline)
        .count();
    assert_eq!(n_pipe, 2);

    // Exactly 1 self-loop on softmax_update (Raw, kv_tile=-1).
    let self_edges = r.edges.iter().filter(|e| e.src == e.dst).count();
    assert_eq!(self_edges, 1);
}

#[test]
fn window_predicate_emitted_only_when_finite() {
    let r_inf = attn_region(&params());
    let mut p = params();
    p.window = Window::Finite(4096);
    let r_fin = attn_region(&p);

    assert_eq!(r_inf.domain.predicates.len(), 1);
    assert_eq!(r_fin.domain.predicates.len(), 2);
    assert_eq!(r_inf.entry_scalars.len(), 2);
    assert_eq!(r_fin.entry_scalars.len(), 3);
}

#[test]
fn sm90_vs_sm89_differ_only_in_mapping() {
    let r = attn_region(&params());
    let s90 = print::print_region(&r, &sm90_fa2());
    let s89 = print::print_region(&r, &sm89_fa2());

    assert!(s90.contains("wg[loader×1]"));
    assert!(s90.contains("wg[consumer×3]"));
    assert!(s90.contains("wg[storer×1]"));
    assert!(s90.contains("pipe_depth: 3"));

    assert!(s89.contains("all_warps"));
    assert!(s89.contains("pipe_depth: 2"));
    assert!(!s89.contains("wg["));
}

#[test]
fn snapshot_sm90_print() {
    let r = attn_region(&params());
    let s = print::print_region(&r, &sm90_fa2());
    // Not a strict golden — just asserts key lines so the printed
    // shape is visible in test output.
    let expected_substrings = [
        "region attn_prefill (arch=sm90_fa2)",
        "q_tile : 0..num_q_tiles",
        "kv_tile : 0..num_kv_tiles",
        "head_group : 0..8",
        "1·kv_tile + -1·q_tile ≤ 0",
        "Load(load_q_tile) -> wg[loader×1]",
        "Compute(qk_matmul) -> wg[consumer×3]",
        "Store(store_o_tile) -> wg[storer×1]",
        "n1 -> n3 [Pipeline NamedSem",
        "n4 -> n4 [Raw Mbarrier vec=kv_tile-1]",
        "pipe_depth: 3",
    ];
    for s_expect in expected_substrings {
        assert!(s.contains(s_expect), "missing {:?} in:\n{}", s_expect, s);
    }
}
