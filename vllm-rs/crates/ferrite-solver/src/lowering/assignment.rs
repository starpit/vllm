// SPDX-License-Identifier: Apache-2.0
//! [`Assignment`] — the joint decision-variable tuple the solver
//! builds incrementally.
//!
//! The solver searches over partial assignments. Each variable in
//! the joint tuple has a domain and current binding (or `None`
//! when unassigned). Constraints are checked over partial
//! assignments — when a constraint can't determine satisfaction
//! yet (because some referenced variables are still unbound), it
//! returns [`ConstraintStatus::Unknown`] and the search proceeds.
//!
//! The variables:
//!
//! - **`cover`**: per [`TileId`], which `SubgraphId` claims it.
//!   `None` = unclaimed (the search will assign it later).
//!
//! - **`impls`**: per `SubgraphId`, which [`ImplId`] from the
//!   library realizes it. Determined together with the cover —
//!   the cover decision is "this subgraph exists and is claimed
//!   by this impl."
//!
//! - **`schedule`**: per `SubgraphId`, when (`step`) and where
//!   (`compilation_unit`) it runs. Two subgraphs in the same step
//!   run "concurrently" subject to the [`ConcurrencyModel`]. Two
//!   subgraphs in the same compilation unit share NVCC's reg/shmem
//!   budget.
//!
//! - **`handoffs`**: per dependency edge between two distinct
//!   subgraphs, which [`Handoff`] mechanism conveys the dep.
//!
//! - **`layouts`**: per [`TileId`] that's a boundary output, which
//!   [`Layout`] it lives in after the producer writes it.
//!
//! ## Why partial state matters for the CP solver
//!
//! Constraint propagation works by repeatedly asking "given what's
//! already bound, can this remaining variable still take this
//! value, or is it pruned?" Constraints that touch only bound
//! variables return Satisfied/Violated; constraints that touch
//! unbound variables return Unknown and propagate domain
//! restrictions back to the search. The CP backtracking solver
//! commits one variable at a time and re-propagates.
//!
//! ## ILP encoding (later)
//!
//! When we add the ILP backend, the same `Assignment` shape is
//! encoded as binary indicator variables (`x_{tile, impl} = 1` if
//! tile is claimed by impl), and the constraints become linear
//! inequalities over those variables. The cost function is the
//! same. CP and ILP backends consume the same `Assignment`
//! definition; they just search it differently.

use std::collections::BTreeMap;

use crate::lowering::implementation::{Handoff, ImplId, Layout};
use crate::lowering::tile_graph::TileId;

/// Stable identifier for one claimed subgraph in an [`Assignment`].
/// Subgraphs are created on demand by the solver — each time it
/// commits "these tiles are claimed by this impl as one unit," a
/// fresh `SubgraphId` is allocated.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct SubgraphId(pub u32);

/// Stable identifier for one compilation unit (one `__global__`
/// function emitted by the backend). Two subgraphs in the same
/// `CompilationUnitId` share NVCC's reg/shmem budget; their
/// resource demands union via [`crate::lowering::Resources::union_max`].
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct CompilationUnitId(pub u32);

/// Per-subgraph schedule slot.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct ScheduleSlot {
    /// Logical step in the execution order. Step ordering must
    /// respect data deps. Two subgraphs in the same step run
    /// "concurrently" — the [`crate::lowering::ConcurrencyModel`]
    /// decides what the actual wall-clock contention is.
    pub step: u32,
    /// Compilation unit this subgraph compiles into. For
    /// `HostCallback` impls each subgraph gets its own unit
    /// (cuBLAS calls don't share `__global__`s with anything).
    /// For `DeviceCallable` impls multiple subgraphs can share a
    /// unit subject to the resource-union constraint.
    pub unit: CompilationUnitId,
}

/// The joint decision-variable tuple. All fields are partial maps
/// — `None` / missing key means "unassigned." A complete
/// `Assignment` has every [`TileId`] in `cover`, every
/// `SubgraphId` in `impls` and `schedule`, every dep edge in
/// `handoffs`, every boundary output in `layouts`.
#[derive(Clone, Debug, Default)]
pub struct Assignment {
    /// Tile → which subgraph claims it.
    pub cover: BTreeMap<TileId, SubgraphId>,
    /// Subgraph → which implementation realizes it.
    pub impls: BTreeMap<SubgraphId, ImplId>,
    /// Subgraph → schedule slot (step + compilation unit).
    pub schedule: BTreeMap<SubgraphId, ScheduleSlot>,
    /// Dep edge `(producer_subgraph, consumer_subgraph)` →
    /// handoff mechanism.
    pub handoffs: BTreeMap<(SubgraphId, SubgraphId), Handoff>,
    /// Tile → its current layout (the layout the producer wrote
    /// it in, modulo any inserted conversions).
    pub layouts: BTreeMap<TileId, Layout>,
}

impl Assignment {
    /// Whether every [`TileId`] in the tile graph has been
    /// covered. The CP solver uses this as a termination
    /// condition.
    pub fn is_cover_complete(&self, num_tiles: usize) -> bool {
        self.cover.len() == num_tiles
    }

    /// Tiles claimed by a particular subgraph, in arbitrary order.
    pub fn tiles_in_subgraph(&self, subgraph: SubgraphId) -> Vec<TileId> {
        self.cover
            .iter()
            .filter_map(|(t, s)| if *s == subgraph { Some(*t) } else { None })
            .collect()
    }

    /// All distinct subgraph ids currently in the assignment.
    pub fn subgraphs(&self) -> impl Iterator<Item = SubgraphId> + '_ {
        self.impls.keys().copied()
    }

    /// All subgraphs in a given compilation unit.
    pub fn subgraphs_in_unit(&self, unit: CompilationUnitId) -> Vec<SubgraphId> {
        self.schedule
            .iter()
            .filter_map(|(s, slot)| if slot.unit == unit { Some(*s) } else { None })
            .collect()
    }

    /// Compilation unit a given subgraph belongs to (if scheduled).
    pub fn unit_of(&self, subgraph: SubgraphId) -> Option<CompilationUnitId> {
        self.schedule.get(&subgraph).map(|s| s.unit)
    }
}
