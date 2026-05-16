// SPDX-License-Identifier: Apache-2.0
//! `MegaTape` container — the typed proof-carrying value the plan's
//! invariant (§1) is stated about.
//!
//! Construction is sealed inside this crate. The only public path
//! to obtain a `MegaTape` is [`crate::lower::lower`], which
//! consumes a `Vec<Instruction<W>>` and produces the tape only when
//! every variant's substrate proofs are satisfiable. If the input
//! semantic Tape can't be substrate-lifted (e.g. cumulative arrive
//! count and requested phase parity disagree), `lower` panics at
//! proc-macro construction time, surfacing as a compile error.
//!
//! Reading a `MegaTape` is a syntactic walk: pattern-match each
//! [`crate::nodes::MegaNode`], extract the typed proofs, splice
//! into the emit step's CUDA strings.

#![allow(dead_code)]

use std::fmt;

use crate::nodes::MegaNode;
use crate::substrate::SubstrateBudget;

pub struct MegaTape {
    nodes: Vec<MegaNode>,
    substrate: SubstrateBudget,
}

impl fmt::Debug for MegaTape {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("MegaTape")
            .field("nodes", &self.nodes.len())
            .field("substrate", &self.substrate)
            .finish()
    }
}

impl MegaTape {
    /// Read-only access for the syntactic emit step.
    pub fn nodes(&self) -> &[MegaNode] {
        &self.nodes
    }

    /// Substrate budget the tape was lowered against.
    pub fn substrate(&self) -> &SubstrateBudget {
        &self.substrate
    }

    /// Crate-private constructor used by [`crate::lower::lower`].
    /// All correctness invariants must be discharged before nodes
    /// are pushed; this function does NO validation.
    #[doc(hidden)]
    pub(crate) fn __build_from_nodes(nodes: Vec<MegaNode>, substrate: SubstrateBudget) -> Self {
        Self { nodes, substrate }
    }
}
