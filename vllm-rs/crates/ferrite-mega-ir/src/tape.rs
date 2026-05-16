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

pub struct MegaTape {
    nodes: Vec<MegaNode>,
}

impl fmt::Debug for MegaTape {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("MegaTape")
            .field("nodes", &self.nodes.len())
            .finish()
    }
}

impl MegaTape {
    /// Read-only access for the syntactic emit step.
    pub fn nodes(&self) -> &[MegaNode] {
        &self.nodes
    }

    /// Crate-private constructor.
    #[doc(hidden)]
    pub(crate) fn __build_from_nodes(nodes: Vec<MegaNode>) -> Self {
        Self { nodes }
    }
}
