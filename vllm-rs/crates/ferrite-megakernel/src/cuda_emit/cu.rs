// SPDX-License-Identifier: Apache-2.0
//! Emitted-CUDA AST: thin string-fragment wrappers carrying enough
//! type structure that misuse of the [`tk20`](super::tk20) Rust API
//! is a Rust compile error rather than a runtime miswire.
//!
//! See `MEGA_IR_PLAN.md` §0 / §8.0 / §8.0a + `CUDA_EMIT_TK20_AUDIT.md`
//! at the worktree root for the contract: emit is pure literal
//! transcription of typed IR getters into TK 2.0 primitive calls
//! from `third_party/thunderkittens/include/`.

use std::fmt::{self, Write};

/// A single CUDA expression fragment (no trailing semicolon).
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
/// block; the [`tk20`](super::tk20) wrappers are responsible for
/// ending each statement appropriately).
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
            for line in s.as_str().lines() {
                writeln!(out, "{pad}{line}").unwrap();
            }
        }
        out
    }
}

/// A complete emitted `.cu` translation unit for one canonical.
#[derive(Clone, Debug)]
pub struct CuVariant {
    pub canonical: String,
    pub source: String,
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
