// SPDX-License-Identifier: Apache-2.0
//! Wavefront scheduler tests. Assert the preamble/body/epilogue
//! partition matches the design intent on both FA2 prefill and
//! paged decode, under both SM90 and SM89 mappings.

use ferrite_stencil::arch::BarrierPrim;
use ferrite_stencil::ir::Role;
use ferrite_stencil::{
    AttnParams, PagedDecodeParams, Window, attn_region, attn_region_paged_decode,
    schedule_wavefront, sm89_fa2, sm90_fa2,
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
    }
}

#[test]
fn fa2_prefill_partition_on_sm90() {
    let r = attn_region(&prefill_params());
    let sched = schedule_wavefront(&r, &sm90_fa2()).unwrap();

    // serial axis = kv_tile; parallel = q_tile, head_group
    assert!(sched.serial_axis.is_some());
    assert_eq!(sched.parallel_axes.len(), 2);
    assert_eq!(sched.pipeline_depth, 3);

    let tag = |node_id: u16| r.nodes[node_id as usize].op.tag;

    // Preamble is exactly load_q (no kv_tile in its address).
    assert_eq!(sched.preamble.len(), 1);
    assert_eq!(tag(sched.preamble[0].node), "load_q_tile");

    // Epilogue is exactly store_o (kv_tile reduced out of its address).
    assert_eq!(sched.epilogue.len(), 1);
    assert_eq!(tag(sched.epilogue[0].node), "store_o_tile");

    // Body = {load_k, load_v, qk, sm, pv}, loads before computes.
    let body_tags: Vec<&str> = sched.body.iter().map(|s| tag(s.node)).collect();
    assert_eq!(body_tags.len(), 5);
    let load_positions: Vec<usize> = body_tags
        .iter()
        .enumerate()
        .filter(|(_, t)| t.starts_with("load_"))
        .map(|(i, _)| i)
        .collect();
    let compute_positions: Vec<usize> = body_tags
        .iter()
        .enumerate()
        .filter(|(_, t)| !t.starts_with("load_") && !t.starts_with("store_"))
        .map(|(i, _)| i)
        .collect();
    assert!(
        load_positions.iter().max().unwrap() < compute_positions.iter().min().unwrap(),
        "body loads must precede body computes; got {:?}",
        body_tags
    );
}

#[test]
fn pipeline_producers_carry_iter_offset() {
    let r = attn_region(&prefill_params());
    let sched = schedule_wavefront(&r, &sm90_fa2()).unwrap();
    let tag = |node_id: u16| r.nodes[node_id as usize].op.tag;

    for step in &sched.body {
        let t = tag(step.node);
        if t == "load_k_tile" || t == "load_v_tile" {
            assert_eq!(step.iter_offset, 3, "{} must lead the consumer by P=3", t);
        } else {
            assert_eq!(step.iter_offset, 0, "{} must not lead", t);
        }
    }
}

#[test]
fn sm90_barriers_are_named_sem_and_mbarrier() {
    let r = attn_region(&prefill_params());
    let sched = schedule_wavefront(&r, &sm90_fa2()).unwrap();

    // qk has one incoming Pipeline edge (from load_k) and one Raw
    // edge (from load_q). Both barriers should appear.
    let tag = |node_id: u16| r.nodes[node_id as usize].op.tag;
    let qk_step = sched
        .body
        .iter()
        .find(|s| tag(s.node) == "qk_matmul")
        .unwrap();
    let has_named_sem = qk_step
        .barriers_before
        .iter()
        .any(|b| matches!(b, BarrierPrim::NamedSem { .. }));
    let has_mbarrier = qk_step
        .barriers_before
        .iter()
        .any(|b| matches!(b, BarrierPrim::Mbarrier));
    assert!(has_named_sem, "qk must wait on a Pipeline NamedSem");
    assert!(has_mbarrier, "qk must wait on a Raw mbarrier (from load_q)");
}

#[test]
fn sm89_uses_cp_async_group_for_pipeline() {
    let r = attn_region(&prefill_params());
    let sched = schedule_wavefront(&r, &sm89_fa2()).unwrap();
    let tag = |node_id: u16| r.nodes[node_id as usize].op.tag;

    let qk_step = sched
        .body
        .iter()
        .find(|s| tag(s.node) == "qk_matmul")
        .unwrap();
    let has_cp_async = qk_step
        .barriers_before
        .iter()
        .any(|b| matches!(b, BarrierPrim::CpAsyncGroup { .. }));
    assert!(has_cp_async, "SM89 must lower Pipeline to cp.async group");
}

#[test]
fn decode_partition_matches_prefill_shape() {
    let r = attn_region_paged_decode(&decode_params());
    let sched = schedule_wavefront(&r, &sm90_fa2()).unwrap();
    let tag = |node_id: u16| r.nodes[node_id as usize].op.tag;

    assert_eq!(sched.pipeline_depth, 3);
    assert_eq!(sched.preamble.len(), 1);
    assert_eq!(tag(sched.preamble[0].node), "load_q_tile");
    assert_eq!(sched.epilogue.len(), 1);
    assert_eq!(tag(sched.epilogue[0].node), "store_o_tile");
    assert_eq!(sched.body.len(), 5);
}

#[test]
fn softmax_self_loop_registers_as_incoming_raw() {
    let r = attn_region(&prefill_params());
    let sched = schedule_wavefront(&r, &sm90_fa2()).unwrap();
    let tag = |node_id: u16| r.nodes[node_id as usize].op.tag;

    let sm = sched
        .body
        .iter()
        .find(|s| tag(s.node) == "softmax_update")
        .unwrap();
    // Softmax has a Raw self-loop (kv_tile=-1) plus a same-iter Raw
    // from qk. Both lower to Mbarrier on SM90.
    let n_mbar = sm
        .barriers_before
        .iter()
        .filter(|b| matches!(b, BarrierPrim::Mbarrier))
        .count();
    assert!(
        n_mbar >= 2,
        "softmax must fence on both its self-loop and qk→sm edge; got {} mbarriers",
        n_mbar
    );
}

#[test]
fn every_body_node_is_accounted_for() {
    let r = attn_region(&prefill_params());
    let sched = schedule_wavefront(&r, &sm90_fa2()).unwrap();

    let total = sched.preamble.len() + sched.body.len() + sched.epilogue.len();
    assert_eq!(
        total,
        r.nodes.len(),
        "every node in the region must appear exactly once in the schedule"
    );

    // And no duplicates.
    let mut seen = Vec::new();
    for s in sched
        .preamble
        .iter()
        .chain(sched.body.iter())
        .chain(sched.epilogue.iter())
    {
        assert!(!seen.contains(&s.node), "node {} scheduled twice", s.node);
        seen.push(s.node);
    }

    // Sanity: every Role::Store goes to epilogue (not body), given
    // the FA2 shape where store_o reduces kv_tile.
    for step in &sched.body {
        let role = r.nodes[step.node as usize].role;
        assert_ne!(
            role,
            Role::Store,
            "Store must not appear in body when serial axis is reduced"
        );
    }
}
