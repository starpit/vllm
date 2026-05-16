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

    /// PROC-MACRO USE ONLY. Build a `MegaTape` from a `Vec<MegaNode>`
    /// where each node was constructed via that variant's
    /// `__new_for_emit` runtime ctor. The proc-macro emits a parallel
    /// literal-const-arg `MegaTapeBuilder::push_*::<...>(...)` call
    /// per node into the user's `build_mega_tape_<canonical>()` fn,
    /// which fires the substrate-proof const-asserts at user-build
    /// time (Phase C step 1's contract). This direct ctor lets the
    /// proc-macro assemble a `MegaTape` value at proc-macro time so
    /// `cuda_emit::lower_to_cuda` can pattern-match it for syntactic
    /// `.cu` emission (Phase C step 2). Both paths share plain-u32
    /// fields so the emit reads the same numbers either way.
    #[doc(hidden)]
    pub fn __build_from_nodes(nodes: Vec<MegaNode>) -> Self {
        Self { nodes }
    }
}
