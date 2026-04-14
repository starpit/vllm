// SPDX-License-Identifier: Apache-2.0
//! Explicit constraint types over an Assignment.
//!
//! Ported from old ferrite-solver (`lowering/constraint.rs`). The
//! DP solver we ship today enforces most of these implicitly by
//! construction; listing them as data makes the ILP / CP backends
//! possible later without rewriting the solver API, and makes it
//! obvious when a scheduler / codegen pass violates one.
//!
//! No transformer vocabulary lives here. Constraints talk about
//! tiles, subgraphs, impls, resources, layouts, handoffs — pure
//! structural things the compiler knows about.

#![allow(dead_code)]

use crate::fuf::TileId;
use crate::impl_lib::{Handoff, ImplId, Layout, Resources};
use crate::solver::SubgraphId;

/// A constraint the solver + scheduler + codegen must keep true
/// for any valid Assignment. The DP solver satisfies these by
/// construction; the CP / ILP backends (if we add them) linearize
/// them as explicit clauses.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Constraint {
    /// Every tile in the FUF is covered by exactly one subgraph.
    /// Failure mode: a tile slipped through the solver without
    /// getting claimed.
    CoverComplete,

    /// `subgraph` is claimed by `impl_id`, and `impl_id`'s
    /// matcher actually accepts the tiles covered by `subgraph`.
    SubgraphMatches {
        subgraph: SubgraphId,
        impl_id: ImplId,
    },

    /// `impl_id` can run on the current target profile.
    TargetCompatible { impl_id: ImplId },

    /// `impl_id` admits the current workload point (num_tokens).
    WorkloadCompatible { impl_id: ImplId },

    /// A producer subgraph's step index precedes its consumer's.
    /// Dep edges between subgraphs must flow forward through the
    /// schedule.
    DependencyOrder {
        producer: SubgraphId,
        consumer: SubgraphId,
    },

    /// At most one cooperative-exclusive Impl resident per step.
    /// (Only relevant once the library has cooperative-grid kernels.)
    CooperativeExclusive,

    /// Union of resources across subgraphs sharing compilation
    /// unit `unit` does not exceed `cap` on the target. Compilation
    /// units are scheduler-allocated; `unit` indexes them.
    ResourceBudget { unit: u32, cap: Resources },

    /// The handoff mechanism conveying data between two subgraphs
    /// is in both producer's and consumer's supported sets, and is
    /// available on the target.
    HandoffCompatible {
        producer: SubgraphId,
        consumer: SubgraphId,
        handoff: Handoff,
    },

    /// Producer's output layout matches consumer's input layout for
    /// the tile conveyed between them, or a conversion subgraph
    /// sits between them.
    LayoutCompatible {
        producer: SubgraphId,
        consumer: SubgraphId,
        layout: Layout,
    },

    /// A tile whose output is shared by multiple consumers is
    /// materialized to storage dtype before any consumer reads it.
    /// Relevant only when a handoff between producer and consumers
    /// truncates precision.
    IntermediateMaterialized { tile: TileId },
}

/// Partial evaluation outcome. `Unknown` arises when the constraint
/// references a variable the current Assignment hasn't bound yet —
/// relevant for backtracking CP. The DP solver only ever evaluates
/// complete assignments, so it only sees Satisfied / Violated.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ConstraintStatus {
    Satisfied,
    Violated,
    Unknown,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_variant_is_constructible_and_distinguishable() {
        // Doubles as documentation of the full variant set. If a
        // variant goes away or its fields change, this test breaks
        // and forces an audit.
        let cs: Vec<Constraint> = vec![
            Constraint::CoverComplete,
            Constraint::SubgraphMatches {
                subgraph: SubgraphId(0),
                impl_id: ImplId(0),
            },
            Constraint::TargetCompatible { impl_id: ImplId(0) },
            Constraint::WorkloadCompatible { impl_id: ImplId(0) },
            Constraint::DependencyOrder {
                producer: SubgraphId(0),
                consumer: SubgraphId(1),
            },
            Constraint::CooperativeExclusive,
            Constraint::ResourceBudget {
                unit: 0,
                cap: Resources::ZERO,
            },
            Constraint::HandoffCompatible {
                producer: SubgraphId(0),
                consumer: SubgraphId(1),
                handoff: Handoff::StreamEvent,
            },
            Constraint::LayoutCompatible {
                producer: SubgraphId(0),
                consumer: SubgraphId(1),
                layout: Layout::Plain,
            },
            Constraint::IntermediateMaterialized { tile: TileId(0) },
        ];
        // No two consecutive entries compare equal; tests the Eq
        // impl treats fields as part of identity.
        for w in cs.windows(2) {
            assert_ne!(w[0], w[1]);
        }
    }

    #[test]
    fn constraint_status_has_three_distinct_outcomes() {
        let s = [
            ConstraintStatus::Satisfied,
            ConstraintStatus::Violated,
            ConstraintStatus::Unknown,
        ];
        for (i, a) in s.iter().enumerate() {
            for (j, b) in s.iter().enumerate() {
                if i == j {
                    assert_eq!(a, b);
                } else {
                    assert_ne!(a, b);
                }
            }
        }
    }
}
