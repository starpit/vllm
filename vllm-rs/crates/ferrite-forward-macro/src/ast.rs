// SPDX-License-Identifier: Apache-2.0
//! Abstract syntax tree for the `#[forward]` DSL.
//!
//! The AST mirrors the source body verbatim — no classification,
//! no shape inference, no substitution. It preserves the user's
//! identifiers and loop bounds as symbolic names so later passes
//! can resolve them per-model.

use syn::Ident;

/// Top-level: the body of the `#[forward]` carrier fn.
#[derive(Clone, Debug)]
pub struct Ast {
    pub statements: Vec<Stmt>,
}

/// A statement in the DSL body.
#[derive(Clone, Debug)]
pub enum Stmt {
    /// `name = expr;`
    Assign { target: Ident, value: Expr },
    /// `(a, b, c) = expr;` — tuple destructuring assignment.
    AssignTuple { targets: Vec<Ident>, value: Expr },
    /// `for ivar in 0..<bound> { body }` — symbolic trip count.
    For {
        ivar: Ident,
        start: BoundExpr,
        end: BoundExpr,
        body: Vec<Stmt>,
    },
}

/// A loop bound expression. Always either an integer literal
/// written in the DSL or an identifier that names a symbolic
/// bound supplied per-model (e.g. `num_hidden_layers`).
#[derive(Clone, Debug)]
pub enum BoundExpr {
    Lit(u64),
    Ident(Ident),
}

/// A value-producing expression inside the DSL body.
#[derive(Clone, Debug)]
pub enum Expr {
    /// A bare identifier read, e.g. `hidden_states` or `input_ids`.
    Var(Ident),
    /// A dotted path, e.g. `self_attn.q_proj`. Stored as the
    /// sequence of segments from left to right.
    Path(Vec<Ident>),
    /// `target[index]` — indexing. `target` may be a `Var` or a
    /// `Path`. `index` is restricted to a single identifier (the
    /// enclosing loop variable) at the parse level; validation of
    /// that restriction happens in a later pass.
    Index { target: Box<Expr>, index: Ident },
    /// A named op call, e.g. `gemm(x, w)`.
    Call { op: Ident, args: Vec<Expr> },
    /// `lhs * rhs` — the only binary operator the DSL currently
    /// admits (used e.g. by `gate * up` feeding the `mlp.down_proj`
    /// gemm).
    Mul { lhs: Box<Expr>, rhs: Box<Expr> },
}
