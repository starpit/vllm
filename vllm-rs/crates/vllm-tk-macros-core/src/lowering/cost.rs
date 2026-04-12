// SPDX-License-Identifier: Apache-2.0
//! Wall-clock cost objective over a complete [`Assignment`].
//!
//! ## What this computes
//!
//! Total predicted wall-clock for one forward pass under the
//! given assignment, in microseconds. The solver's branch-and-bound
//! pruning uses this as the upper bound it tries to minimize.
//!
//! ```text
//! cost_us(assignment) =
//!     sum over scheduled subgraphs of:
//!         impl.cost_us(match, profile)
//!         * concurrency_model.contention_factor(impl, concurrent_impls)
//!     + sum over scheduled handoffs of:
//!         handoff.cost_us(profile)
//!     + num_coop_units × launch_cost_us
//!     + num_regular_units × per_launch_overhead_us
//! ```
//!
//! The CSV costs are **compute-only** (launch overhead subtracted),
//! so each distinct `CompilationUnitId` adds the per-launch overhead
//! back. `DeviceCallable` impls don't add launch cost — they run
//! inside an enclosing kernel.
//!
//! ## What it does NOT model (yet)
//!
//! - Layout-conversion costs from inserted conversion impls (the
//!   conversion impls themselves contribute their own cost when
//!   added; what's missing is the *L2 thrash* from a conversion
//!   that touches a large buffer).
//! - L2-cache reuse across consecutive same-shape calls. cuBLAS's
//!   per-call cost is ~constant in our calibration; it would be
//!   smaller for back-to-back identical calls. Marginal effect.
//! - GPU-side scheduler overhead per launch beyond the
//!   `launch_cost_us` constant.
//!
//! These can be added later as the cost model gets more accurate.

use std::collections::HashMap;

use crate::lowering::assignment::{Assignment, SubgraphId};
use crate::lowering::concurrency::ConcurrencyModel;
use crate::lowering::implementation::{Implementation, LaunchKind, MatchInfo};
use crate::lowering::library::ImplementationLibrary;
use crate::lowering::tile_graph::TileGraph;
use crate::target_profile::TargetProfile;

/// Compute the predicted wall-clock for one full forward pass
/// under the given assignment. Returns `f64::INFINITY` if any
/// subgraph references an unbound impl (the assignment is
/// incomplete).
pub fn cost_us(
    assignment: &Assignment,
    tile_graph: &TileGraph,
    library: &ImplementationLibrary,
    profile: &TargetProfile,
) -> f64 {
    let concurrency = ConcurrencyModel::new(profile);

    // Group subgraphs by step so we can apply per-step contention.
    let mut by_step: HashMap<u32, Vec<SubgraphId>> = HashMap::new();
    for (sg, slot) in &assignment.schedule {
        by_step.entry(slot.step).or_default().push(*sg);
    }

    let mut total_us: f64 = 0.0;

    // ── Per-subgraph compute cost (with contention factor) ──
    for (step, subgraphs) in &by_step {
        // Resolve each subgraph to its impl + match.
        let mut step_entries: Vec<(SubgraphId, &dyn Implementation, MatchInfo)> =
            Vec::with_capacity(subgraphs.len());
        for &sg in subgraphs {
            let Some(impl_id) = assignment.impls.get(&sg) else {
                return f64::INFINITY;
            };
            let imp = library.get(*impl_id);
            let claimed = assignment.tiles_in_subgraph(sg);
            let layer = claimed
                .first()
                .map(|t| tile_graph.nodes[t.0 as usize].layer)
                .unwrap_or(0);
            // Build a fresh MatchInfo for cost lookup. The matcher's
            // boundary_inputs/outputs aren't needed for cost, only
            // claimed_tiles + layer.
            let m = MatchInfo {
                claimed_tiles: claimed,
                boundary_inputs: vec![],
                boundary_outputs: vec![],
                layer,
            };
            step_entries.push((sg, imp, m));
        }

        // For each entry, compute its cost with the contention
        // factor accounting for the OTHER entries in this step.
        for i in 0..step_entries.len() {
            let (_sg, imp, m) = &step_entries[i];
            let others: Vec<&dyn Implementation> = step_entries
                .iter()
                .enumerate()
                .filter_map(|(j, (_, oi, _))| if j == i { None } else { Some(*oi) })
                .collect();
            let factor = concurrency.contention_factor(*imp, &others);
            if !factor.is_finite() {
                return f64::INFINITY;
            }
            total_us += imp.cost_us(m, profile) * factor;
        }
        let _ = step;
    }

    // ── Handoff costs ──
    for ((_, _), handoff) in &assignment.handoffs {
        total_us += handoff.cost_us(profile);
    }

    // ── Per-launch overhead ──
    //
    // Each distinct CompilationUnitId represents one __global__
    // launch. The CSV costs have per-launch overhead subtracted
    // (compute-only), so we add it back here.
    //
    // - CooperativeLaunch: pays the higher cooperative launch cost.
    // - HostCallback / RegularLaunch: pays per_launch_overhead_us.
    // - DeviceCallable: no launch cost — runs inside an enclosing
    //   kernel. If all impls in a unit are DeviceCallable, that
    //   unit's single launch cost is already counted by the
    //   enclosing unit.
    {
        use std::collections::HashSet;
        let mut regular_units: HashSet<u32> = HashSet::new();
        let mut coop_units: HashSet<u32> = HashSet::new();

        for sg in assignment.subgraphs() {
            let Some(slot) = assignment.schedule.get(&sg) else {
                continue;
            };
            let Some(impl_id) = assignment.impls.get(&sg) else {
                continue;
            };
            let imp = library.get(*impl_id);
            let uid = slot.unit.0;
            match imp.launch_kind() {
                LaunchKind::CooperativeLaunch => {
                    coop_units.insert(uid);
                }
                LaunchKind::HostCallback | LaunchKind::RegularLaunch => {
                    regular_units.insert(uid);
                }
                LaunchKind::DeviceCallable => {}
            }
        }

        total_us += coop_units.len() as f64 * profile.lowering.launch_cost_us as f64;
        total_us += regular_units.len() as f64 * profile.lowering.per_launch_overhead_us as f64;
    }

    total_us
}
