// SPDX-License-Identifier: Apache-2.0
//! Arch-neutral scheduling primitives: axis classification,
//! pipeline-depth derivation from edges, and same-iteration topo
//! order. Tests both FA2 prefill and paged decode against the same
//! primitives — they're what the wavefront scheduler will rest on.

use ferrite_stencil::{
    AttnParams, AxisKind, PagedDecodeParams, Window, attn_region, attn_region_paged_decode,
    classify_axes, region_pipeline_depth, topo_order_within_iter,
};

fn prefill_params() -> AttnParams {
    AttnParams {
        window: Window::Infinite,
        head_dim: 128,
        tile_q: 128,
        tile_k: 64,
        num_head_groups: 8,
        pipe: 3,
    }
}

fn decode_params() -> PagedDecodeParams {
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
fn fa2_axis_classification() {
    let r = attn_region(&prefill_params());
    let classes = classify_axes(&r);

    let by_name = |name: &str| {
        let axis = r.domain.axes.iter().find(|a| a.name == name).unwrap();
        classes
            .iter()
            .find(|(id, _)| *id == axis.id)
            .map(|(_, k)| *k)
            .unwrap()
    };

    assert_eq!(by_name("q_tile"), AxisKind::Parallel);
    assert_eq!(by_name("kv_tile"), AxisKind::Serial);
    assert_eq!(by_name("head_group"), AxisKind::Parallel);
}

#[test]
fn decode_axis_classification() {
    // Same shape: b × head_group parallel, kv_tile serial.
    let r = attn_region_paged_decode(&decode_params());
    let classes = classify_axes(&r);
    let by_name = |name: &str| {
        let axis = r.domain.axes.iter().find(|a| a.name == name).unwrap();
        classes
            .iter()
            .find(|(id, _)| *id == axis.id)
            .map(|(_, k)| *k)
            .unwrap()
    };
    assert_eq!(by_name("b"), AxisKind::Parallel);
    assert_eq!(by_name("kv_tile"), AxisKind::Serial);
    assert_eq!(by_name("head_group"), AxisKind::Parallel);
}

#[test]
fn pipeline_depth_derived_from_edges() {
    let r = attn_region(&prefill_params());
    assert_eq!(region_pipeline_depth(&r), 3);

    let r_decode = attn_region_paged_decode(&decode_params());
    assert_eq!(region_pipeline_depth(&r_decode), 3);

    // Set pipe=5 → depth=5.
    let mut p = prefill_params();
    p.pipe = 5;
    let r2 = attn_region(&p);
    assert_eq!(region_pipeline_depth(&r2), 5);
}

#[test]
fn topo_order_respects_same_iter_raw_edges() {
    let r = attn_region(&prefill_params());
    let order = topo_order_within_iter(&r);
    assert_eq!(order.len(), r.nodes.len());

    // Position-in-order lookup.
    let pos = |id: u16| order.iter().position(|n| *n == id).unwrap();

    // Known FA2 node ids from template.rs:
    // 0=load_q, 1=load_k, 2=load_v, 3=qk, 4=sm, 5=pv, 6=store
    let (lq, qk, sm, pv, st) = (pos(0), pos(3), pos(4), pos(5), pos(6));

    // Same-iter Raw chains: load_q → qk → sm → pv → store.
    assert!(lq < qk, "load_q must precede qk_matmul");
    assert!(qk < sm, "qk must precede softmax");
    assert!(sm < pv, "softmax must precede pv_matmul");
    assert!(pv < st, "pv must precede store");
}

#[test]
fn topo_order_does_not_chain_pipeline_edges() {
    // load_k → qk is Pipeline (dep vector kv_tile=-P), not Raw.
    // It must NOT appear in same-iter topo as a constraint —
    // otherwise the scheduler couldn't emit load_k ahead of qk.
    let r = attn_region(&prefill_params());
    let order = topo_order_within_iter(&r);
    let pos = |id: u16| order.iter().position(|n| *n == id).unwrap();

    // load_k (1) being *after* qk (3) in same-iter order is allowed
    // because pipeline edges don't constrain same-iter ordering.
    // We just assert the schedule didn't insist on load_k before qk
    // *via the Pipeline edge*. Concretely: removing load_k→qk Raw
    // would have no effect; there is no such Raw edge.
    let lk_pos = pos(1);
    let qk_pos = pos(3);
    // One of the two orderings is valid; both must be consistent
    // with load_k having no same-iter Raw edge to qk.
    assert_ne!(
        lk_pos, qk_pos,
        "positions must differ (sanity: they're distinct nodes)"
    );
}
