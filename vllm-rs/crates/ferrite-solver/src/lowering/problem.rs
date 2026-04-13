// SPDX-License-Identifier: Apache-2.0
//! [`Problem`] — bundles tile graph + library + profile +
//! constraint set into one input the solver consumes.
//!
//! The point of this type is that the solver doesn't know
//! anything about how the tile graph was built or where the
//! library entries came from — it only sees a `Problem` with
//! variables to bind and a constraint set to satisfy.
//!
//! Building a `Problem`:
//!
//! ```ignore
//! let tile_graph = TileGraph::build_llama_forward(16);
//! let library = ImplementationLibrary::l4_sm89_starter();
//! let profile = TargetProfile::l4_sm89();
//! let problem = Problem::build(&tile_graph, &library, &profile);
//! ```
//!
//! `Problem::build` populates the constraint set automatically
//! from the tile graph's structure (one [`Constraint::DependencyOrder`]
//! per dep edge, one [`Constraint::CoverComplete`] for the whole
//! graph, etc.) plus the target's hard limits (one
//! [`Constraint::CompilationUnitRegBudget`] / `ShmemBudget` per
//! compilation unit, allocated lazily as the solver creates units).

use crate::lowering::constraint::Constraint;
use crate::lowering::library::ImplementationLibrary;
use crate::lowering::tile_graph::TileGraph;
use crate::target_profile::TargetProfile;

/// One lowering problem instance: the tile graph to cover, the
/// library of implementations to choose from, the target's
/// constraints, and the precomputed constraint set the solver
/// must satisfy.
/// Controls whether the solver is allowed to fuse across precision
/// truncation points. In `Serving` mode, fusion is unrestricted
/// (slightly different but more precise output). In `BitExact` mode,
/// every dependency edge requires a GMEM truncation — no fusion can
/// skip a dtype boundary — producing bit-identical output to the
/// unfused reference.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum PrecisionMode {
    /// Allow fusion across truncation points. Output may differ
    /// from the unfused reference by the accumulated precision of
    /// skipped truncations. Default for serving.
    #[default]
    Serving,
    /// Require all handoffs to truncate to storage dtype. No fusion
    /// can skip a GMEM write. Output is bit-identical to the
    /// unfused reference. Useful for golden validation / debugging.
    BitExact,
}

pub struct Problem<'a> {
    pub tile_graph: &'a TileGraph,
    pub library: &'a ImplementationLibrary,
    pub profile: &'a TargetProfile,
    pub static_constraints: Vec<Constraint>,
}

impl<'a> Problem<'a> {
    /// Build with default precision mode (Serving — allow fusion).
    pub fn build(
        tile_graph: &'a TileGraph,
        library: &'a ImplementationLibrary,
        profile: &'a TargetProfile,
    ) -> Self {
        Self::build_with_precision(tile_graph, library, profile, PrecisionMode::Serving)
    }

    /// Build with explicit precision mode.
    pub fn build_with_precision(
        tile_graph: &'a TileGraph,
        library: &'a ImplementationLibrary,
        profile: &'a TargetProfile,
        precision: PrecisionMode,
    ) -> Self {
        let mut static_constraints: Vec<Constraint> = Vec::new();

        // 1. CoverComplete: every tile must be claimed.
        static_constraints.push(Constraint::CoverComplete {
            num_tiles: tile_graph.nodes.len() as u32,
        });

        // 2. CooperativeExclusive: at most one cooperative grid
        //    per step on this target. Always required.
        static_constraints.push(Constraint::CooperativeExclusive);

        // 3. DependencyOrder + IntermediateMaterialized + PrecisionBounded:
        //    one of each per dep edge in the tile graph.
        let require_truncation = precision == PrecisionMode::BitExact;
        for node in &tile_graph.nodes {
            for dep in &node.deps {
                static_constraints.push(Constraint::DependencyOrder {
                    producer: *dep,
                    consumer: node.id,
                });
                static_constraints.push(Constraint::IntermediateMaterialized {
                    producer: *dep,
                    consumer: node.id,
                });
                static_constraints.push(Constraint::PrecisionBounded {
                    producer: *dep,
                    consumer: node.id,
                    require_truncation,
                });
            }
        }

        Problem {
            tile_graph,
            library,
            profile,
            static_constraints,
        }
    }
}
