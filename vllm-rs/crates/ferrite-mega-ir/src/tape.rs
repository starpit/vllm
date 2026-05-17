// SPDX-License-Identifier: Apache-2.0
//! `MegaTape` container — the typed proof-carrying value.
//!
//! In the const-generic edition, the substrate budget is encoded
//! via `MegaTapeBuilder<NUM_PAGES, ...>`'s const generics; the
//! resulting `MegaTape` is just a `Vec<MegaNode>`. Each node's
//! load-bearing fields were validated at MONOMORPHIZATION TIME by
//! the variant's `new::<...>` constructor's `const {}` block, so
//! the tape is structurally substrate-correct: "if it compiles, it
//! runs coherently."

#![allow(dead_code)]

use std::fmt;

use crate::nodes::MegaNode;

/// Runtime snapshot of the substrate budget the tape was built
/// against. Stamped by `MegaTapeBuilder::finish()` from its const
/// generics so `cuda_emit` can read the values without needing the
/// builder's const-generic parameters at the emit call site.
///
/// Every field here is a value the emitted `.cu`'s `Config` struct
/// or kernel weight-pointer arithmetic must carry (see
/// `ferrite_substrate.cuh`).
#[derive(Clone, Copy, Debug)]
pub struct TapeBudget {
    pub num_pages: u32,
    pub num_consumer_warps: u32,
    pub page_size: u32,
    pub scratch_bytes: u32,
    pub num_edges: u32,
    /// `num_hidden_layers` for the canonical's model. Carried on
    /// the tape (not on per-op nodes) because every op in a tape
    /// shares the same value, and the emit step needs it for
    /// `weight_ptrs[acc * NUM_LAYERS + layer]` arithmetic.
    pub num_layers: u32,
}

pub struct MegaTape {
    nodes: Vec<MegaNode>,
    budget: TapeBudget,
}

impl fmt::Debug for MegaTape {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("MegaTape")
            .field("nodes", &self.nodes.len())
            .field("budget", &self.budget)
            .finish()
    }
}

impl MegaTape {
    /// Read-only access for the syntactic emit step.
    pub fn nodes(&self) -> &[MegaNode] {
        &self.nodes
    }

    /// Substrate budget snapshot the tape was built against.
    pub fn budget(&self) -> TapeBudget {
        self.budget
    }

    /// Crate-private constructor.
    #[doc(hidden)]
    pub(crate) fn __build_from_nodes(nodes: Vec<MegaNode>, budget: TapeBudget) -> Self {
        Self { nodes, budget }
    }
}
