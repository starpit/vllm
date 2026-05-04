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
    /// `if <cond> { then_body } else { else_body }` — compile-time
    /// conditional. The condition must be one of a small closed set
    /// of predicates over a loop-induction variable (see
    /// [`BoolExpr`]). Evaluated at unroll time.
    If {
        cond: BoolExpr,
        then_body: Vec<Stmt>,
        else_body: Vec<Stmt>,
    },
}

/// Boolean predicate shape for `if` conditions. Deliberately narrow
/// — just enough to express layer-indexed dispatch patterns without
/// extending the expression IR with booleans or binary arithmetic.
#[derive(Clone, Debug)]
pub enum BoolExpr {
    /// `ivar % divisor == remainder`.
    Modulo {
        ivar: Ident,
        divisor: BoundExpr,
        remainder: BoundExpr,
    },
    /// `ivar % divisor != remainder`.
    NotModulo {
        ivar: Ident,
        divisor: BoundExpr,
        remainder: BoundExpr,
    },
    /// `ivar < bound`.
    Less { ivar: Ident, bound: BoundExpr },
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
    /// `lhs * rhs` (SwiGLU `gate * up`). Tensor × tensor.
    Mul { lhs: Box<Expr>, rhs: Box<Expr> },
    /// `lhs + rhs`. Admitted when the RHS (or LHS) is a scalar
    /// literal — e.g. Gemma's `w + 1.0` on rmsnorm weights. Tensor
    /// + tensor goes through the named `add(a, b)` op instead.
    Add { lhs: Box<Expr>, rhs: Box<Expr> },
    /// A numeric scalar literal (f64). Only valid as an operand to
    /// `Add` / `Mul`; not a standalone assignment.
    ScalarLit(f64),
    /// `sqrt(<bound_name>)` — a compile-time scalar whose value is
    /// `(bounds[name] as f64).sqrt()`. Resolved at CFG-build time
    /// (where per-model bounds are known). Used e.g. by Gemma's
    /// embedding scale `embed(ids, w) * sqrt(hidden_size)`.
    SqrtBound(Ident),
    /// `scalar(<name>)` / `recip_scalar(<name>)` — a compile-time
    /// scalar read from `ModelParams.scalars` (non-integer top-level
    /// fields of the model's `config.json`). Resolved at CFG-build
    /// time. `recip == true` folds to `1.0 / scalars[name]`, used
    /// for divisors like Granite's `logits_scaling`.
    ConfigScalar { name: Ident, recip: bool },
    /// `reshape(source, [d0, d1, ...])` — DSL-authored reshape with
    /// an explicit target shape. The target shape is preserved here
    /// for `classify` to thread into [`Program::reshape_targets`]
    /// keyed by the producing `LocalId`; the classified form drops
    /// the target list and becomes
    /// `Expr::Call { op: OpKind::Reshape, args: [source] }`,
    /// matching the synthesized-reshape shape from
    /// `shape::apply_reshape_hints`.
    ///
    /// G.5.c surface is "literal-or-bound only" — each `DimSpec`
    /// is an integer literal or a bare bound name. Bound-arithmetic
    /// (`num_tokens / vision_merge_factor`) is deliberately deferred
    /// until the merger reshape needs it.
    Reshape {
        source: Box<Expr>,
        target_shape: Vec<DimSpec>,
    },
}

/// One dim of a [`Expr::Reshape`] target shape. Restricted to the
/// minimal G.5.c surface: integer literal or a config-bound name.
#[derive(Clone, Debug)]
pub enum DimSpec {
    /// Concrete integer dim, e.g. `4`.
    Lit(u64),
    /// Symbolic bound name, e.g. `vision_embed_dim`.
    Bound(Ident),
}
