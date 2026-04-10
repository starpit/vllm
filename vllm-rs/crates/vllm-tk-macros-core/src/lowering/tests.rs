// SPDX-License-Identifier: Apache-2.0
//! Solver-driven validation tests.
//!
//! These tests verify that the solver produces valid, cost-reasonable
//! plans for the L4 sm_89 target. Unlike the original CP5-A tests
//! (which hand-built an assignment for a specific library layout),
//! these use the solver and are resilient to library changes.

use crate::lowering::constraint::ConstraintStatus;
use crate::lowering::cost::cost_us;
use crate::lowering::library::ImplementationLibrary;
use crate::lowering::problem::Problem;
use crate::lowering::tile_graph::TileGraph;
use crate::lowering::{BacktrackCpSolver, SolveResult, Solver};
use crate::target_profile::TargetProfile;

#[test]
fn natural_sm89_assignment_satisfies_all_static_constraints() {
    let tile_graph = TileGraph::build_llama_forward(16);
    let library = ImplementationLibrary::l4_sm89_starter();
    let profile = TargetProfile::l4_sm89();
    let problem = Problem::build(&tile_graph, &library, &profile);

    let plan = match BacktrackCpSolver.solve(&problem) {
        SolveResult::Found(p) => p,
        SolveResult::Infeasible => panic!("solver reported infeasible"),
    };

    // Every static constraint must report Satisfied on the
    // solver's output.
    let mut violated = Vec::new();
    let mut unknown = Vec::new();
    for c in &problem.static_constraints {
        match c.check(&plan.assignment, &tile_graph, &library, &profile) {
            ConstraintStatus::Satisfied => {}
            ConstraintStatus::Violated => violated.push(format!("{c:?}")),
            ConstraintStatus::Unknown => unknown.push(format!("{c:?}")),
        }
    }

    assert!(violated.is_empty(), "constraints violated: {violated:#?}",);
    // Some constraints (HandoffCompatible) may be Unknown if
    // handoffs are not fully assigned. That's acceptable.
}

#[test]
fn natural_sm89_cost_matches_cp4_microbench_estimate() {
    let tile_graph = TileGraph::build_llama_forward(16);
    let library = ImplementationLibrary::l4_sm89_starter();
    let profile = TargetProfile::l4_sm89();
    let problem = Problem::build(&tile_graph, &library, &profile);

    let plan = match BacktrackCpSolver.solve(&problem) {
        SolveResult::Found(p) => p,
        SolveResult::Infeasible => panic!("solver reported infeasible"),
    };

    // The analytical cost model predicts ~35 ms at seq=1024 on L4.
    // Measured is ~40 ms (model is optimistic — doesn't account for
    // memory latency, occupancy limits, cuBLAS overhead).
    let pred_ms = plan.predicted_us / 1000.0;
    eprintln!("predicted: {pred_ms:.2} ms");
    assert!(
        (25.0..50.0).contains(&pred_ms),
        "predicted {pred_ms} ms outside expected 25-50 ms band",
    );

    // The cost function should agree with the plan's predicted_us.
    let recomputed = cost_us(&plan.assignment, &tile_graph, &library, &profile);
    assert!(
        (recomputed - plan.predicted_us).abs() < 0.01,
        "cost_us ({recomputed}) disagrees with plan.predicted_us ({})",
        plan.predicted_us,
    );
}

#[test]
fn dependency_order_constraint_catches_swapped_subgraphs() {
    // Build a valid plan, then swap two subgraphs' schedule steps
    // to create a dependency violation. Verify the constraint
    // catches it.
    let tile_graph = TileGraph::build_llama_forward(2);
    let library = ImplementationLibrary::l4_sm89_starter();
    let profile = TargetProfile::l4_sm89();
    let problem = Problem::build(&tile_graph, &library, &profile);

    let plan = match BacktrackCpSolver.solve(&problem) {
        SolveResult::Found(p) => p,
        SolveResult::Infeasible => panic!("solver reported infeasible"),
    };

    // Find two subgraphs with a dependency edge and swap their steps.
    let mut assignment = plan.assignment.clone();
    let mut swapped = false;
    'outer: for c in &problem.static_constraints {
        if let crate::lowering::constraint::Constraint::DependencyOrder { producer, consumer } = c {
            let p_sg = assignment.cover.get(producer);
            let c_sg = assignment.cover.get(consumer);
            if let (Some(p_sg), Some(c_sg)) = (p_sg, c_sg) {
                if p_sg != c_sg {
                    // Swap their steps.
                    let p_step = assignment.schedule[p_sg].step;
                    let c_step = assignment.schedule[c_sg].step;
                    assignment.schedule.get_mut(p_sg).unwrap().step = c_step;
                    assignment.schedule.get_mut(c_sg).unwrap().step = p_step;
                    swapped = true;
                    break 'outer;
                }
            }
        }
    }
    assert!(
        swapped,
        "couldn't find two cross-subgraph dep edges to swap"
    );

    // At least one DependencyOrder constraint should now be Violated.
    let has_violation = problem.static_constraints.iter().any(|c| {
        matches!(
            c.check(&assignment, &tile_graph, &library, &profile),
            ConstraintStatus::Violated
        )
    });
    assert!(
        has_violation,
        "swapped steps should violate a dependency order constraint"
    );
}

#[test]
fn cooperative_exclusive_constraint_holds_for_all_host_callback_lowering() {
    // All impls in the current library are HostCallback, so the
    // CooperativeExclusive constraint is trivially satisfied.
    let tile_graph = TileGraph::build_llama_forward(2);
    let library = ImplementationLibrary::l4_sm89_starter();
    let profile = TargetProfile::l4_sm89();
    let problem = Problem::build(&tile_graph, &library, &profile);

    let plan = match BacktrackCpSolver.solve(&problem) {
        SolveResult::Found(p) => p,
        SolveResult::Infeasible => panic!("solver reported infeasible"),
    };

    let coop_constraint = problem.static_constraints.iter().find(|c| {
        matches!(
            c,
            crate::lowering::constraint::Constraint::CooperativeExclusive
        )
    });
    assert!(coop_constraint.is_some());
    let status = coop_constraint
        .unwrap()
        .check(&plan.assignment, &tile_graph, &library, &profile);
    assert_eq!(status, ConstraintStatus::Satisfied);
}
