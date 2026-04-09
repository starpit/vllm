// SPDX-License-Identifier: Apache-2.0
//! CP5-A validation tests.
//!
//! These tests validate that the **model** is right, NOT that the
//! solver works (there's no solver yet — that's CP5-B). They build
//! a tile graph, populate a library, hand-construct an
//! `Assignment` corresponding to the natural-sm89 lowering, and
//! verify:
//!
//!   1. Every constraint reports `Satisfied` on the assignment.
//!   2. The cost function returns a number close to the CP4
//!      microbench measurement (~38 ms without attention, ~48 ms
//!      with).
//!
//! If both pass, the model types and the cost calibration are
//! self-consistent and CP5-B can plug a solver on top with
//! confidence.

use crate::lowering::assignment::{Assignment, CompilationUnitId, ScheduleSlot, SubgraphId};
use crate::lowering::constraint::ConstraintStatus;
use crate::lowering::cost::cost_us;
use crate::lowering::implementation::{Handoff, ImplId};
use crate::lowering::library::ImplementationLibrary;
use crate::lowering::problem::Problem;
use crate::lowering::tile_graph::{TileGraph, TileId, TileKind};
use crate::target_profile::TargetProfile;

/// Look up an `ImplId` by name in the library. Tests use this to
/// build the assignment without depending on the library's
/// internal index ordering.
fn impl_by_name(library: &ImplementationLibrary, name: &str) -> ImplId {
    library
        .iter()
        .find(|(_, imp)| imp.name() == name)
        .map(|(id, _)| id)
        .unwrap_or_else(|| panic!("library has no implementation named {name:?}"))
}

/// For a given tile kind, return the impl in the library that
/// claims a single-tile subgraph of that kind in the natural
/// sm_89 lowering. Used to make the test assignment construction
/// readable.
fn natural_sm89_impl_for(library: &ImplementationLibrary, kind: TileKind) -> &'static str {
    let _ = library;
    match kind {
        TileKind::RmsNorm => "vllm_rs_rms_norm",
        TileKind::GemmQkv => "cublas_gemm_ex_qkv",
        TileKind::GemmOProj => "cublas_gemm_ex_oproj",
        TileKind::GemmGate => "cublas_gemm_ex_gate",
        TileKind::GemmUp => "cublas_gemm_ex_up",
        TileKind::GemmDown => "cublas_gemm_ex_down",
        TileKind::QkvSplit => "qkv_split_free",
        TileKind::Rope => "vllm_rs_rotary_embedding",
        TileKind::KvCacheWrite => "kv_cache_write",
        TileKind::Attention => "flashinfer_standalone_fa2",
        TileKind::ResidualAdd => "residual_add",
        // GateUpConcat + SiluMul are claimed *together* by the
        // VllmRsSiluAndMulFusedImpl as one two-tile subgraph; the
        // test below handles them specially.
        TileKind::GateUpConcat | TileKind::SiluMul => {
            unreachable!("GateUpConcat / SiluMul handled by the silu_and_mul claim, not 1:1")
        }
    }
}

/// Hand-build the natural-sm89 [`Assignment`] for the given tile
/// graph. Each tile gets its own subgraph + impl + schedule slot
/// EXCEPT GateUpConcat + SiluMul which share a two-tile subgraph
/// claimed by `vllm_rs_silu_and_mul_fused`.
///
/// All subgraphs land in their own compilation unit (each
/// HostCallback is its own kernel boundary). Steps are assigned
/// in topological order.
fn build_natural_sm89_assignment(
    tile_graph: &TileGraph,
    library: &ImplementationLibrary,
) -> Assignment {
    let mut a = Assignment::default();
    let mut next_subgraph: u32 = 0;
    let mut next_unit: u32 = 0;
    let mut next_step: u32 = 0;

    // First pass: assign subgraphs + impls per tile, with the
    // silu+mul fusion handled inline.
    let mut tile_to_subgraph: std::collections::HashMap<TileId, SubgraphId> =
        std::collections::HashMap::new();
    let mut subgraph_impl: std::collections::HashMap<SubgraphId, ImplId> =
        std::collections::HashMap::new();

    for node in tile_graph.iter_topo() {
        if tile_to_subgraph.contains_key(&node.id) {
            continue; // already claimed (e.g. SiluMul claimed with GateUpConcat)
        }

        let sg = SubgraphId(next_subgraph);
        next_subgraph += 1;

        match node.kind {
            TileKind::GateUpConcat => {
                // Claim this concat AND its consumer SiluMul under
                // one subgraph realized by vllm_rs_silu_and_mul_fused.
                let silu = tile_graph
                    .nodes
                    .iter()
                    .find(|n| n.kind == TileKind::SiluMul && n.deps.contains(&node.id))
                    .expect("every GateUpConcat has a SiluMul consumer");
                tile_to_subgraph.insert(node.id, sg);
                tile_to_subgraph.insert(silu.id, sg);
                let imp_id = impl_by_name(library, "vllm_rs_silu_and_mul_fused");
                subgraph_impl.insert(sg, imp_id);
            }
            _ => {
                tile_to_subgraph.insert(node.id, sg);
                let name = natural_sm89_impl_for(library, node.kind);
                let imp_id = impl_by_name(library, name);
                subgraph_impl.insert(sg, imp_id);
            }
        }
    }

    a.cover.extend(tile_to_subgraph);
    a.impls.extend(subgraph_impl);

    // Second pass: assign each subgraph its own compilation unit
    // and a unique step (sequential — no concurrency in the
    // baseline natural lowering).
    let mut subgraph_seen: std::collections::BTreeSet<SubgraphId> = Default::default();
    for node in tile_graph.iter_topo() {
        let sg = a.cover[&node.id];
        if subgraph_seen.insert(sg) {
            let unit = CompilationUnitId(next_unit);
            next_unit += 1;
            let slot = ScheduleSlot {
                step: next_step,
                unit,
            };
            next_step += 1;
            a.schedule.insert(sg, slot);
        }
    }

    // Third pass: assign default RowMajorBf16 layouts to every tile
    // (the natural-sm89 lowering doesn't introduce conversions).
    for node in tile_graph.iter_topo() {
        a.layouts.insert(
            node.id,
            crate::lowering::implementation::Layout::RowMajorBf16,
        );
    }

    // Fourth pass: insert StreamOrder handoffs between consecutive
    // subgraphs along every dep edge that crosses a subgraph
    // boundary. (StreamOrder = same stream, implicit ordering.)
    for node in tile_graph.iter_topo() {
        let consumer_sg = a.cover[&node.id];
        for dep in &node.deps {
            let producer_sg = a.cover[dep];
            if producer_sg != consumer_sg {
                a.handoffs
                    .entry((producer_sg, consumer_sg))
                    .or_insert(Handoff::StreamOrder);
            }
        }
    }

    a
}

#[test]
fn natural_sm89_assignment_satisfies_all_static_constraints() {
    let tile_graph = TileGraph::build_llama_forward(16);
    let library = ImplementationLibrary::l4_sm89_starter();
    let profile = TargetProfile::l4_sm89();
    let problem = Problem::build(&tile_graph, &library, &profile);
    let assignment = build_natural_sm89_assignment(&tile_graph, &library);

    // Every static constraint must report Satisfied (or Unknown
    // for ones that need extra info we haven't added — none of
    // the constraints in this set should be Unknown for a fully
    // populated assignment).
    let mut violated = Vec::new();
    let mut unknown = Vec::new();
    for c in &problem.static_constraints {
        match c.check(&assignment, &tile_graph, &library, &profile) {
            ConstraintStatus::Satisfied => {}
            ConstraintStatus::Violated => violated.push(format!("{c:?}")),
            ConstraintStatus::Unknown => unknown.push(format!("{c:?}")),
        }
    }

    assert!(
        violated.is_empty(),
        "constraints violated by the natural-sm89 assignment: {violated:#?}",
    );
    assert!(
        unknown.is_empty(),
        "constraints in unknown state for a complete assignment: {unknown:#?}",
    );
}

#[test]
fn natural_sm89_cost_matches_cp4_microbench_estimate() {
    // Build the same assignment, compute the cost, assert it's
    // within the right ballpark of the CP4 measurement.
    //
    // CP4 measurements (50-iter avg, L4 sm_89, Llama-1B seq=1024):
    //   cuBLAS-only GEMMs:                  36.6 ms
    //   Natural forward (no attention):     37.8 ms
    //   Natural forward + estimated attn:  ~48.0 ms
    //
    // The cost model includes attention (the FlashInfer standalone
    // entry contributes ~625 µs/layer × 16 = 10 ms), plus the
    // per-launch overhead for every HostCallback (80 launches ×
    // 80 µs = 6.4 ms — but the cost model uses the launch_cost_us
    // calibrated in CP2 which is 80 µs).
    //
    // Expected total: ~36 ms (GEMMs) + ~10 ms (attention) +
    //                 ~1 ms (norms/rope/silu) + ~6 ms (launch
    //                 overhead × ~80 calls) ≈ 53 ms.
    //
    // The CP4 microbench measured 37.8 ms WITHOUT attention and
    // WITHOUT explicit launch-overhead accounting (cuBLAS+vllm-rs
    // FFI calls in a tight Rust loop, where the launch overhead
    // is hidden inside cuBLAS's internal cudaLaunchKernel cost
    // already baked into the per-call wall-clock).
    //
    // So this assertion checks that the model produces a number
    // in the same ORDER OF MAGNITUDE as the measurement, not exact
    // agreement. CP5-B will refine the cost model from real per-impl
    // microbench data.

    let tile_graph = TileGraph::build_llama_forward(16);
    let library = ImplementationLibrary::l4_sm89_starter();
    let profile = TargetProfile::l4_sm89();
    let assignment = build_natural_sm89_assignment(&tile_graph, &library);

    let predicted_us = cost_us(&assignment, &tile_graph, &library, &profile);
    let predicted_ms = predicted_us / 1000.0;

    eprintln!(
        "natural sm_89 predicted: {:.2} ms (CP4 measured: 37.8 ms no-attn, ~48 ms with-attn)",
        predicted_ms
    );

    // Plausible range: between 30 ms and 100 ms. This is a
    // sanity bound — the model returns SOMETHING reasonable, not
    // wildly off (e.g. 0 or 1000 ms). CP5-B will tighten this.
    assert!(
        predicted_ms > 30.0 && predicted_ms < 100.0,
        "predicted cost {predicted_ms} ms is outside the plausible 30-100 ms range",
    );
}

#[test]
fn dependency_order_constraint_catches_swapped_subgraphs() {
    // Negative test: if we maliciously schedule a producer AFTER
    // its consumer, the DependencyOrder constraint must catch it.
    let tile_graph = TileGraph::build_llama_forward(2);
    let library = ImplementationLibrary::l4_sm89_starter();
    let profile = TargetProfile::l4_sm89();
    let problem = Problem::build(&tile_graph, &library, &profile);
    let mut assignment = build_natural_sm89_assignment(&tile_graph, &library);

    // Find layer 0's qkv subgraph and its consumer (qkv_split).
    // Swap their step ordering.
    let qkv_tile = tile_graph
        .iter_topo()
        .find(|n| n.kind == TileKind::GemmQkv && n.layer == 0)
        .unwrap();
    let split_tile = tile_graph
        .iter_topo()
        .find(|n| n.kind == TileKind::QkvSplit && n.layer == 0)
        .unwrap();
    let qkv_sg = assignment.cover[&qkv_tile.id];
    let split_sg = assignment.cover[&split_tile.id];
    let qkv_step = assignment.schedule[&qkv_sg].step;
    let split_step = assignment.schedule[&split_sg].step;
    // Sanity: qkv was scheduled before split before we swap.
    assert!(qkv_step < split_step);
    // Swap.
    let qkv_slot = assignment.schedule[&qkv_sg];
    let split_slot = assignment.schedule[&split_sg];
    assignment.schedule.insert(
        qkv_sg,
        ScheduleSlot {
            step: split_slot.step,
            unit: qkv_slot.unit,
        },
    );
    assignment.schedule.insert(
        split_sg,
        ScheduleSlot {
            step: qkv_slot.step,
            unit: split_slot.unit,
        },
    );

    // Now at least one DependencyOrder constraint must report
    // Violated (the qkv → qkv_split edge specifically).
    let mut found_violation = false;
    for c in &problem.static_constraints {
        if matches!(
            c.check(&assignment, &tile_graph, &library, &profile),
            ConstraintStatus::Violated
        ) {
            found_violation = true;
            break;
        }
    }
    assert!(
        found_violation,
        "DependencyOrder constraint did not catch a swapped producer/consumer schedule"
    );
}

#[test]
fn cooperative_exclusive_constraint_holds_for_all_host_callback_lowering() {
    // The natural-sm89 lowering uses zero CooperativeLaunch impls
    // (everything is HostCallback), so CooperativeExclusive
    // trivially holds. Sanity test that the constraint doesn't
    // false-positive on the all-HostCallback case.
    let tile_graph = TileGraph::build_llama_forward(4);
    let library = ImplementationLibrary::l4_sm89_starter();
    let profile = TargetProfile::l4_sm89();
    let assignment = build_natural_sm89_assignment(&tile_graph, &library);
    let coop = crate::lowering::constraint::Constraint::CooperativeExclusive;
    assert_eq!(
        coop.check(&assignment, &tile_graph, &library, &profile),
        ConstraintStatus::Satisfied
    );
}

#[test]
fn library_starter_has_all_kinds_covered() {
    // Sanity: for every TileKind that appears in the natural-sm89
    // lowering, the library has at least one matching impl.
    let library = ImplementationLibrary::l4_sm89_starter();
    let tile_graph = TileGraph::build_llama_forward(1);
    let profile = TargetProfile::l4_sm89();
    for node in tile_graph.iter_topo() {
        // SiluMul + GateUpConcat are claimed together; we only
        // need one matcher to fire on at least one of them.
        let mut any_match = false;
        for (_, imp) in library.iter() {
            if imp.matches(&tile_graph, node.id, &profile).is_some() {
                any_match = true;
                break;
            }
        }
        assert!(
            any_match,
            "no library impl matches tile {:?} ({})",
            node.id,
            node.kind.name()
        );
    }
}
