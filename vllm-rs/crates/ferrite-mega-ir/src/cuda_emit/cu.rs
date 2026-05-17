// SPDX-License-Identifier: Apache-2.0
//! Emitted-CUDA AST: thin string-fragment wrappers carrying enough
//! type structure that a misuse of the [`tk`](super::tk) API surface
//! is a Rust compile error rather than a runtime miswire.
//!
//! Each `Cu*` type is a `String` with a typed wrapper. The Rust type
//! system enforces "this fragment is a statement" vs "this fragment
//! is an expression" vs "this fragment is a sequence of statements
//! grouped into a block" — passing a [`CuExpr`] where a [`CuStmt`] is
//! expected is a compile error, even though both are `String` under
//! the covers.
//!
//! See `MEGA_IR_PLAN.md` §0 / §8.0 for the contract: emit is pure
//! literal transcription of typed IR getters into typed [`tk`] calls.
//! No inference; if a value isn't on a node's typed getter, the IR
//! is incomplete and emit code does not exist for that variant yet.
//!
//! [`tk`]: super::tk

use std::fmt::{self, Write};

/// A single CUDA expression fragment (no trailing semicolon).
///
/// Examples: `ss.page_ready[3]`, `kittens::warpid()`, `EPS`.
#[derive(Clone, Debug)]
pub struct CuExpr(pub(crate) String);

impl CuExpr {
    pub fn new(s: impl Into<String>) -> Self {
        Self(s.into())
    }
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl fmt::Display for CuExpr {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

/// A single CUDA statement fragment (terminated by `;` or a `{...}`
/// block; the [`tk`](super::tk) wrappers are responsible for ending
/// each statement appropriately).
#[derive(Clone, Debug)]
pub struct CuStmt(pub(crate) String);

impl CuStmt {
    pub fn new(s: impl Into<String>) -> Self {
        Self(s.into())
    }
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl fmt::Display for CuStmt {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

/// An ordered sequence of statements forming the body of a role
/// (loader / launcher / consumer / storer) accumulated across every
/// `MegaNode` in the tape.
#[derive(Clone, Debug, Default)]
pub struct CuBlock {
    stmts: Vec<CuStmt>,
}

impl CuBlock {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn push(&mut self, stmt: CuStmt) -> &mut Self {
        self.stmts.push(stmt);
        self
    }

    pub fn extend(&mut self, other: CuBlock) -> &mut Self {
        self.stmts.extend(other.stmts);
        self
    }

    pub fn is_empty(&self) -> bool {
        self.stmts.is_empty()
    }

    pub fn len(&self) -> usize {
        self.stmts.len()
    }

    /// Render the block with each statement on its own line, indented
    /// by `indent` spaces.
    pub fn render(&self, indent: usize) -> String {
        let pad = " ".repeat(indent);
        let mut out = String::new();
        for s in &self.stmts {
            // Allow multi-line CuStmt (e.g. `{ ... }` blocks): split
            // and indent each line so the output is uniformly padded.
            for line in s.as_str().lines() {
                writeln!(out, "{pad}{line}").unwrap();
            }
        }
        out
    }
}

/// A complete emitted `.cu` translation unit for one canonical: an
/// includes preamble, the per-canonical `Config` struct, the four
/// role bodies inlined in tape order, and the kernel entry point.
///
/// Returned by [`lower_to_cuda`](super::lower_to_cuda); written to
/// disk by the proc-macro side at expansion time and picked up by
/// `cudaforge` for compilation.
#[derive(Clone, Debug)]
pub struct CuVariant {
    /// Canonical name (e.g. `llama_3_2_1b_m_1_sk_128`). Used as the
    /// suffix on the kernel's extern-C symbol name and the `.cu`
    /// filename.
    pub canonical: String,
    /// Full `.cu` source text.
    pub source: String,
    /// Diagnostics from the emit step. Populated when the tape
    /// contains variants for which the per-variant role-body emit
    /// has not been written yet (Sprint 1 ships RmsNorm only). The
    /// `source` field still holds the kernel scaffold; the unhandled
    /// variant is rendered as a `// SKIPPED: <variant>` comment so
    /// readers can see what's missing.
    pub skipped_variants: Vec<String>,
}

impl CuVariant {
    pub fn new(canonical: String, source: String) -> Self {
        Self {
            canonical,
            source,
            skipped_variants: Vec::new(),
        }
    }
}
