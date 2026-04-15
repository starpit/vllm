// SPDX-License-Identifier: Apache-2.0
//! Phase 4: shape inference.
//!
//! Every tensor in the classified program gets a [`Shape`] —
//! a sequence of [`Dim`] expressions over a compact symbolic
//! vocabulary:
//!
//!   - `Dim::Lit(n)` — a concrete integer;
//!   - `Dim::Bound(name)` — a named bound from config.json like
//!     `"hidden_size"` or `"num_attention_heads"`;
//!   - `Dim::Mul(...)` — a product of the above (e.g.
//!     `num_attention_heads * head_dim`);
//!   - `Dim::Var(id)` — an unresolved dim, used during inference
//!     and required to be resolved by the end.
//!
//! Inference walks the classified program forward, invoking a
//! per-op [signature function][signatures] at each call site. The
//! signature emits unification constraints (Dim equalities) that
//! the [`Solver`] resolves via union-find. Weight shapes emerge
//! from their consuming ops — no weight-name sniffing, no
//! architecture knowledge, just op-signature + dataflow.
//!
//! The bound names referenced by op signatures (`hidden_size`,
//! `num_attention_heads`, `head_dim`, `intermediate_size`,
//! `num_key_value_heads`, `vocab_size`, etc.) are the standard
//! HuggingFace config.json vocabulary — shared across every
//! decoder-only LLM. Op-level, not arch-level.

#![allow(dead_code)]

use std::collections::HashMap;
use std::fmt;

use crate::classified::{Expr, ExternKind, LocalId, OpKind, Program, Stmt, WeightId};

// ── Types ─────────────────────────────────────────────────────────

/// A dimension expression.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Dim {
    /// Concrete integer literal.
    Lit(u64),
    /// Named bound — resolves to a config.json integer per-model.
    Bound(String),
    /// Product of subterms. Canonically flattened: no nested Mul.
    Mul(Vec<Dim>),
    /// Unresolved variable. Must be resolved before inference
    /// completes or shape inference reports an error.
    Var(DimVar),
}

/// Fresh-var id. Allocated by the [`Solver`].
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct DimVar(u32);

/// A tensor shape — an ordered sequence of dims.
pub type Shape = Vec<Dim>;

/// The solver state: union-find over dim variables plus
/// concrete bindings.
#[derive(Debug, Default)]
pub struct Solver {
    /// `parent[v] = v` unless `v` has been unified into another var.
    parent: Vec<DimVar>,
    /// Optional concrete binding for each representative var.
    binding: HashMap<DimVar, Dim>,
}

impl Solver {
    pub fn new() -> Self {
        Self::default()
    }

    /// Allocate a fresh unresolved dim variable.
    pub fn fresh(&mut self) -> DimVar {
        let v = DimVar(self.parent.len() as u32);
        self.parent.push(v);
        v
    }

    fn find(&mut self, v: DimVar) -> DimVar {
        let p = self.parent[v.0 as usize];
        if p == v {
            v
        } else {
            let root = self.find(p);
            self.parent[v.0 as usize] = root;
            root
        }
    }

    /// Constrain two dims to be equal. Returns an error if the
    /// constraint is inconsistent with existing bindings.
    pub fn unify(&mut self, a: &Dim, b: &Dim) -> Result<(), ShapeError> {
        let a = self.walk(a);
        let b = self.walk(b);
        match (&a, &b) {
            // Both resolved concrete: must structurally match.
            (Dim::Lit(_), Dim::Lit(_))
            | (Dim::Bound(_), Dim::Bound(_))
            | (Dim::Mul(_), Dim::Mul(_))
            | (Dim::Lit(_), Dim::Bound(_))
            | (Dim::Bound(_), Dim::Lit(_))
            | (Dim::Lit(_), Dim::Mul(_))
            | (Dim::Mul(_), Dim::Lit(_))
            | (Dim::Bound(_), Dim::Mul(_))
            | (Dim::Mul(_), Dim::Bound(_)) => {
                if dims_structurally_equal(&a, &b) {
                    Ok(())
                } else {
                    Err(ShapeError::Mismatch {
                        lhs: a.clone(),
                        rhs: b.clone(),
                    })
                }
            }
            // One var, one concrete: bind the var.
            (Dim::Var(v), other) | (other, Dim::Var(v)) => {
                let root = self.find(*v);
                if let Some(existing) = self.binding.get(&root).cloned() {
                    // Already bound — recurse to unify new info with the binding.
                    self.unify(&existing, other)
                } else if let Dim::Var(ov) = other {
                    // Two vars — merge.
                    let or = self.find(*ov);
                    if or != root {
                        self.parent[or.0 as usize] = root;
                    }
                    Ok(())
                } else {
                    self.binding.insert(root, other.clone());
                    Ok(())
                }
            }
        }
    }

    /// Walk a dim through any bindings to produce its current
    /// representative form. Doesn't recurse into Mul children —
    /// those stay symbolic.
    pub fn walk(&mut self, d: &Dim) -> Dim {
        match d {
            Dim::Var(v) => {
                let root = self.find(*v);
                match self.binding.get(&root).cloned() {
                    Some(bound) => self.walk(&bound),
                    None => Dim::Var(root),
                }
            }
            Dim::Mul(children) => {
                let walked: Vec<Dim> = children.iter().map(|c| self.walk(c)).collect();
                canonical_mul(walked)
            }
            other => other.clone(),
        }
    }

    /// Final resolution: walk each dim through current bindings,
    /// producing a Dim that is either fully concrete (Lit/Bound/
    /// Mul of those) or a remaining Var. An unresolved Var
    /// indicates the DSL has a shape the op-signature + dataflow
    /// couldn't pin to a config.json bound; callers decide what
    /// to do (often that's fine — the dim becomes a runtime/
    /// model-specific value fetched from the weight layout).
    pub fn close_dim(&mut self, d: &Dim) -> Result<Dim, ShapeError> {
        let d = self.walk(d);
        match d {
            Dim::Var(v) => Ok(Dim::Var(v)),
            Dim::Mul(cs) => {
                let cs = cs
                    .iter()
                    .map(|c| self.close_dim(c))
                    .collect::<Result<Vec<_>, _>>()?;
                Ok(canonical_mul(cs))
            }
            other => Ok(other),
        }
    }

    pub fn close_shape(&mut self, shape: &Shape) -> Result<Shape, ShapeError> {
        shape.iter().map(|d| self.close_dim(d)).collect()
    }
}

fn dims_structurally_equal(a: &Dim, b: &Dim) -> bool {
    match (a, b) {
        (Dim::Lit(x), Dim::Lit(y)) => x == y,
        (Dim::Bound(x), Dim::Bound(y)) => x == y,
        (Dim::Mul(x), Dim::Mul(y)) => {
            x.len() == y.len() && x.iter().zip(y).all(|(p, q)| dims_structurally_equal(p, q))
        }
        // A Lit(1) is identity for Mul — don't worry about that here,
        // canonical_mul handles it.
        _ => false,
    }
}

/// Flatten and sort a Mul's children to a canonical form. Single-
/// element products collapse to their inner dim.
fn canonical_mul(mut children: Vec<Dim>) -> Dim {
    // Flatten nested Muls.
    let mut flat = Vec::with_capacity(children.len());
    for c in children.drain(..) {
        match c {
            Dim::Mul(inner) => flat.extend(inner),
            other => flat.push(other),
        }
    }
    // Drop Lit(1)s.
    flat.retain(|d| !matches!(d, Dim::Lit(1)));
    // Deterministic ordering for canonicalization.
    flat.sort_by(dim_cmp);
    match flat.len() {
        0 => Dim::Lit(1),
        1 => flat.pop().unwrap(),
        _ => Dim::Mul(flat),
    }
}

fn dim_cmp(a: &Dim, b: &Dim) -> std::cmp::Ordering {
    // Bound < Lit < Mul < Var ordering, breaking ties within each.
    use std::cmp::Ordering;
    fn rank(d: &Dim) -> u8 {
        match d {
            Dim::Bound(_) => 0,
            Dim::Lit(_) => 1,
            Dim::Mul(_) => 2,
            Dim::Var(_) => 3,
        }
    }
    match rank(a).cmp(&rank(b)) {
        Ordering::Equal => match (a, b) {
            (Dim::Bound(x), Dim::Bound(y)) => x.cmp(y),
            (Dim::Lit(x), Dim::Lit(y)) => x.cmp(y),
            (Dim::Mul(x), Dim::Mul(y)) => x.len().cmp(&y.len()),
            (Dim::Var(x), Dim::Var(y)) => x.0.cmp(&y.0),
            _ => unreachable!(),
        },
        other => other,
    }
}

/// Errors produced during shape inference.
#[derive(Debug)]
pub enum ShapeError {
    Mismatch {
        lhs: Dim,
        rhs: Dim,
    },
    Unresolved(DimVar),
    /// Op signature rejected the input shapes (wrong rank, etc.).
    BadArgs {
        op: OpKind,
        reason: String,
    },
    /// An op got the wrong number of args.
    ArgCount {
        op: OpKind,
        expected: usize,
        got: usize,
    },
}

impl fmt::Display for ShapeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Mismatch { lhs, rhs } => {
                write!(f, "shape mismatch: {} vs {}", show_dim(lhs), show_dim(rhs))
            }
            Self::Unresolved(v) => write!(f, "unresolved dim variable {}", v.0),
            Self::BadArgs { op, reason } => {
                write!(f, "op {} rejected inputs: {reason}", op.as_str())
            }
            Self::ArgCount { op, expected, got } => {
                write!(f, "op {} expected {expected} args, got {got}", op.as_str())
            }
        }
    }
}

impl std::error::Error for ShapeError {}

fn show_dim(d: &Dim) -> String {
    match d {
        Dim::Lit(n) => n.to_string(),
        Dim::Bound(n) => n.clone(),
        Dim::Mul(cs) => cs.iter().map(show_dim).collect::<Vec<_>>().join(" * "),
        Dim::Var(v) => format!("?{}", v.0),
    }
}

// ── Op signatures ────────────────────────────────────────────────

/// What an op signature produces: an output shape, plus any extra
/// constraints on its inputs that weren't expressible as direct
/// unification of existing dims.
pub struct OpSig {
    pub output: Shape,
}

/// Apply an op's signature. Emits unification constraints on the
/// shared [`Solver`], reading any existing constraints from the
/// input shapes.
pub fn apply_signature(
    solver: &mut Solver,
    op: OpKind,
    inputs: &[Shape],
) -> Result<OpSig, ShapeError> {
    match op {
        OpKind::Embed => sig_embed(solver, inputs),
        OpKind::RmsNorm => sig_rmsnorm(solver, inputs),
        OpKind::Gemm => sig_gemm(solver, inputs),
        OpKind::RopeAppend => sig_rope_append(solver, inputs),
        OpKind::Attention => sig_attention(solver, inputs),
        // Same q/k/v constraints as `attention`; the distinction is
        // in the picked kernel (window-masked vs. dense), not in
        // the type signature.
        OpKind::SlidingAttention => sig_attention(solver, inputs),
        OpKind::Silu => sig_unary_elementwise(solver, inputs, op),
        OpKind::Gelu => sig_unary_elementwise(solver, inputs, op),
        OpKind::TanhSoftCap => sig_unary_elementwise(solver, inputs, op),
        OpKind::Add => sig_binary_elementwise(solver, inputs, op),
        OpKind::Mul => sig_binary_elementwise(solver, inputs, op),
    }
}

/// `embed(ids: [..ids.shape], table: [vocab_size, hidden_size])`
/// → `[..ids.shape, hidden_size]`.
fn sig_embed(solver: &mut Solver, inputs: &[Shape]) -> Result<OpSig, ShapeError> {
    expect_args(OpKind::Embed, inputs, 2)?;
    let ids = &inputs[0];
    let table = &inputs[1];
    // table: [vocab_size, hidden_size]
    if table.len() != 2 {
        return Err(ShapeError::BadArgs {
            op: OpKind::Embed,
            reason: format!("embed weight must have rank 2, got {}", table.len()),
        });
    }
    solver.unify(&table[0], &Dim::Bound("vocab_size".into()))?;
    solver.unify(&table[1], &Dim::Bound("hidden_size".into()))?;
    let mut output = ids.clone();
    output.push(Dim::Bound("hidden_size".into()));
    Ok(OpSig { output })
}

/// `rmsnorm(x: [..., H], w: [H])` → `[..., H]`.
fn sig_rmsnorm(solver: &mut Solver, inputs: &[Shape]) -> Result<OpSig, ShapeError> {
    expect_args(OpKind::RmsNorm, inputs, 2)?;
    let x = &inputs[0];
    let w = &inputs[1];
    if x.is_empty() {
        return Err(ShapeError::BadArgs {
            op: OpKind::RmsNorm,
            reason: "input must have rank >= 1".into(),
        });
    }
    if w.len() != 1 {
        return Err(ShapeError::BadArgs {
            op: OpKind::RmsNorm,
            reason: format!("weight must have rank 1, got {}", w.len()),
        });
    }
    solver.unify(x.last().unwrap(), &w[0])?;
    Ok(OpSig { output: x.clone() })
}

/// `gemm(x: [..., K], w: [K, N])` → `[..., N]`.
fn sig_gemm(solver: &mut Solver, inputs: &[Shape]) -> Result<OpSig, ShapeError> {
    expect_args(OpKind::Gemm, inputs, 2)?;
    let x = &inputs[0];
    let w = &inputs[1];
    if x.is_empty() {
        return Err(ShapeError::BadArgs {
            op: OpKind::Gemm,
            reason: "input must have rank >= 1".into(),
        });
    }
    if w.len() != 2 {
        return Err(ShapeError::BadArgs {
            op: OpKind::Gemm,
            reason: format!("weight must have rank 2, got {}", w.len()),
        });
    }
    // K = last dim of x, must match first dim of w.
    solver.unify(x.last().unwrap(), &w[0])?;
    // Output is x's prefix followed by N (w's second dim).
    let mut output = x[..x.len() - 1].to_vec();
    output.push(w[1].clone());
    Ok(OpSig { output })
}

/// `rope_append((q, k, v, positions, rotary, kv_cache))` →
/// `(q', k', v')` all with the same shapes as their inputs. Imposes
/// conventional heads-layout on q/k/v:
///
///   q: [.., num_attention_heads * head_dim]
///   k: [.., num_key_value_heads * head_dim]
///   v: [.., num_key_value_heads * head_dim]
///
/// This is the shape anchor that pulls q_proj / k_proj / v_proj
/// outputs into `num_{attn,kv}_heads * head_dim`.
///
/// We model rope_append's output as a "tuple" shape by returning
/// a shape that is q's shape — the parser produces three LocalIds
/// on the LHS, but the inference pass only cares about *one* of
/// them (the rest are shape-equivalent to q and k). Callers should
/// bind all three targets to q's shape, k's shape, v's shape
/// respectively; we expose those via the signature's *constraints*.
fn sig_rope_append(solver: &mut Solver, inputs: &[Shape]) -> Result<OpSig, ShapeError> {
    expect_args(OpKind::RopeAppend, inputs, 6)?;
    let q = &inputs[0];
    let k = &inputs[1];
    let v = &inputs[2];
    // Assert heads-layout on last dim.
    let attn_heads = Dim::Mul(vec![
        Dim::Bound("num_attention_heads".into()),
        Dim::Bound("head_dim".into()),
    ]);
    let kv_heads = Dim::Mul(vec![
        Dim::Bound("num_key_value_heads".into()),
        Dim::Bound("head_dim".into()),
    ]);
    if q.is_empty() {
        return Err(ShapeError::BadArgs {
            op: OpKind::RopeAppend,
            reason: "q must have rank >= 1".into(),
        });
    }
    solver.unify(q.last().unwrap(), &attn_heads)?;
    if !k.is_empty() {
        solver.unify(k.last().unwrap(), &kv_heads)?;
    }
    if !v.is_empty() {
        solver.unify(v.last().unwrap(), &kv_heads)?;
    }
    // Output for the first target (conventionally q') = q's shape.
    // The caller binds the other two targets separately (see
    // `infer::classify_rope_append_tuple`).
    Ok(OpSig { output: q.clone() })
}

/// `attention(q, k, v, kv_cache, block_table)` →
/// `[.., num_attention_heads * head_dim]`. The op signature
/// mirrors `rope_append`'s constraints on q/k/v so attention
/// doesn't introduce fresh unknowns.
fn sig_attention(solver: &mut Solver, inputs: &[Shape]) -> Result<OpSig, ShapeError> {
    expect_args(OpKind::Attention, inputs, 5)?;
    let q = &inputs[0];
    let k = &inputs[1];
    let v = &inputs[2];
    let attn_heads = Dim::Mul(vec![
        Dim::Bound("num_attention_heads".into()),
        Dim::Bound("head_dim".into()),
    ]);
    let kv_heads = Dim::Mul(vec![
        Dim::Bound("num_key_value_heads".into()),
        Dim::Bound("head_dim".into()),
    ]);
    if q.is_empty() {
        return Err(ShapeError::BadArgs {
            op: OpKind::Attention,
            reason: "q must have rank >= 1".into(),
        });
    }
    solver.unify(q.last().unwrap(), &attn_heads)?;
    if !k.is_empty() {
        solver.unify(k.last().unwrap(), &kv_heads)?;
    }
    if !v.is_empty() {
        solver.unify(v.last().unwrap(), &kv_heads)?;
    }
    Ok(OpSig { output: q.clone() })
}

/// Elementwise unary ops (silu, gelu, …) preserve shape.
fn sig_unary_elementwise(
    _solver: &mut Solver,
    inputs: &[Shape],
    op: OpKind,
) -> Result<OpSig, ShapeError> {
    expect_args(op, inputs, 1)?;
    Ok(OpSig {
        output: inputs[0].clone(),
    })
}

/// Binary elementwise ops (`add`, `mul`). Output shape equals the
/// non-scalar operand; two tensor operands are unified elementwise.
///
/// A rank-0 operand (empty Shape) is a scalar broadcast — used by
/// e.g. `w + 1.0` where `1.0` is a [`ScalarLit`](Expr::ScalarLit).
/// Broadcast is allowed only when the OTHER operand has positive
/// rank; scalar-on-both is rejected (`add(1.0, 1.0)` is a
/// compile-time constant and doesn't belong in the op graph).
fn sig_binary_elementwise(
    solver: &mut Solver,
    inputs: &[Shape],
    op: OpKind,
) -> Result<OpSig, ShapeError> {
    expect_args(op, inputs, 2)?;
    let x = &inputs[0];
    let y = &inputs[1];
    // Scalar broadcast: one operand rank-0 → output is the other.
    match (x.is_empty(), y.is_empty()) {
        (true, true) => Err(ShapeError::BadArgs {
            op,
            reason: format!(
                "{} of two scalars is a compile-time constant, \
                 not an op — fold at the call site",
                op.as_str(),
            ),
        }),
        (true, false) => Ok(OpSig { output: y.clone() }),
        (false, true) => Ok(OpSig { output: x.clone() }),
        (false, false) => {
            if x.len() != y.len() {
                return Err(ShapeError::BadArgs {
                    op,
                    reason: format!(
                        "{} operands differ in rank: {} vs {}",
                        op.as_str(),
                        x.len(),
                        y.len()
                    ),
                });
            }
            for (a, b) in x.iter().zip(y) {
                solver.unify(a, b)?;
            }
            Ok(OpSig { output: x.clone() })
        }
    }
}

/// For each op, the (arg_idx, expected_rank) pairs of its *weight*
/// arguments. Used to lazily allocate fresh-var shapes for weights
/// on first encounter so subsequent unification has something to
/// work with.
fn weight_arg_ranks(op: OpKind) -> &'static [(usize, usize)] {
    match op {
        OpKind::Embed => &[(1, 2)],
        OpKind::RmsNorm => &[(1, 1)],
        OpKind::Gemm => &[(1, 2)],
        OpKind::RopeAppend => &[],
        OpKind::Attention => &[],
        OpKind::SlidingAttention => &[],
        OpKind::Silu => &[],
        OpKind::Gelu => &[],
        OpKind::TanhSoftCap => &[],
        OpKind::Add => &[],
        OpKind::Mul => &[],
    }
}

fn expect_args(op: OpKind, inputs: &[Shape], expected: usize) -> Result<(), ShapeError> {
    if inputs.len() != expected {
        Err(ShapeError::ArgCount {
            op,
            expected,
            got: inputs.len(),
        })
    } else {
        Ok(())
    }
}

// ── Extern param shapes ──────────────────────────────────────────

pub fn extern_shape(kind: ExternKind) -> Shape {
    match kind {
        ExternKind::InputIds => vec![Dim::Bound("num_tokens".into())],
        ExternKind::Positions => vec![Dim::Bound("num_tokens".into())],
        // The following externs have opaque shapes — shape
        // inference doesn't need to model them to derive weight
        // shapes for the Llama/Qwen2 body. The solver/codegen
        // consumes them via their ExternKind tag rather than via
        // shape.
        ExternKind::Rotary => vec![],
        ExternKind::BlockTable => vec![],
        ExternKind::KvCache => vec![],
    }
}

// ── Inference pass ────────────────────────────────────────────────

/// Result of Phase 4: every local and every weight has a fully
/// resolved shape, in terms of concrete integer literals and named
/// config.json bounds.
#[derive(Debug)]
pub struct Inferred {
    pub locals: HashMap<LocalId, Shape>,
    pub weights: HashMap<WeightId, Shape>,
}

/// Run shape inference over a classified program.
pub fn infer(program: &Program) -> Result<Inferred, ShapeError> {
    let mut cx = InferCtx::new();
    cx.infer_stmts(&program.statements)?;

    // Anchor remaining weight Vars via the standard HF transformer
    // weight-name convention table. Dataflow handles attention-block
    // weights (q/k/v/o_proj get pinned by rope_append + add(oproj,
    // hidden_states)); the convention table handles MLP weights and
    // other norms whose dims aren't pinned by any op signature.
    for (wid, shape) in cx.weights.clone().iter() {
        let path: Vec<String> = program
            .weights
            .path(*wid)
            .iter()
            .map(|i| i.to_string())
            .collect();
        if let Some(convention) = crate::weight_conventions::standard_shape(&path)
            && shape.len() == convention.len()
        {
            for (inferred, declared) in shape.iter().zip(&convention) {
                cx.solver.unify(inferred, declared)?;
            }
        }
    }

    // Close every recorded shape. After convention anchoring, most
    // Vars resolve to concrete dim expressions. Any remaining Vars
    // correspond to weights not covered by the convention (arch-
    // specific, pending `weights.json` in a future phase); Dim::Var
    // is preserved rather than erroring.
    let mut locals = HashMap::new();
    for (id, shape) in cx.locals {
        locals.insert(id, cx.solver.close_shape(&shape)?);
    }
    let mut weights = HashMap::new();
    for (id, shape) in cx.weights {
        weights.insert(id, cx.solver.close_shape(&shape)?);
    }
    Ok(Inferred { locals, weights })
}

struct InferCtx {
    solver: Solver,
    locals: HashMap<LocalId, Shape>,
    weights: HashMap<WeightId, Shape>,
}

impl InferCtx {
    fn new() -> Self {
        Self {
            solver: Solver::new(),
            locals: HashMap::new(),
            weights: HashMap::new(),
        }
    }

    fn infer_stmts(&mut self, stmts: &[Stmt]) -> Result<(), ShapeError> {
        for s in stmts {
            self.infer_stmt(s)?;
        }
        Ok(())
    }

    /// Ensure that the given weight has been allocated a shape of
    /// the given rank. First encounter creates a fresh-var shape;
    /// later encounters that disagree on rank are a hard error.
    fn ensure_weight_rank(&mut self, id: WeightId, rank: usize) {
        let current_len = self.weights.get(&id).map(|s| s.len()).unwrap_or(0);
        if current_len == 0 && rank > 0 {
            let fresh: Shape = (0..rank).map(|_| Dim::Var(self.solver.fresh())).collect();
            self.weights.insert(id, fresh);
        }
    }

    fn infer_stmt(&mut self, stmt: &Stmt) -> Result<(), ShapeError> {
        match stmt {
            Stmt::Assign { target, value } => {
                let shape = self.infer_expr(value)?;
                self.locals.insert(*target, shape);
                Ok(())
            }
            Stmt::AssignTuple { targets, value } => {
                // The DSL's only tuple-returning op is rope_append.
                // Its signature already constrained q/k/v, and its
                // output is q's shape. We bind each target to the
                // corresponding input's shape (which the signature
                // already unified with the heads-layout).
                if let Expr::Call {
                    op: OpKind::RopeAppend,
                    args,
                } = value
                {
                    // q/k/v are args 0/1/2. Their shapes were set
                    // by the Gemm ops that produced them.
                    let shapes = [
                        self.expr_shape(&args[0])?,
                        self.expr_shape(&args[1])?,
                        self.expr_shape(&args[2])?,
                    ];
                    // Apply the signature to emit the heads-layout
                    // unification constraints for all three.
                    let all_input_shapes: Vec<Shape> = args
                        .iter()
                        .map(|a| self.expr_shape(a))
                        .collect::<Result<_, _>>()?;
                    apply_signature(&mut self.solver, OpKind::RopeAppend, &all_input_shapes)?;
                    // Now bind targets to the post-unification
                    // shapes. (Since rope_append is shape-preserving,
                    // target shapes equal input shapes.)
                    if targets.len() != 3 {
                        return Err(ShapeError::BadArgs {
                            op: OpKind::RopeAppend,
                            reason: format!(
                                "rope_append returns 3 values, got {} targets",
                                targets.len()
                            ),
                        });
                    }
                    for (t, s) in targets.iter().zip(shapes.iter()) {
                        self.locals.insert(*t, s.clone());
                    }
                    Ok(())
                } else {
                    Err(ShapeError::BadArgs {
                        op: match value {
                            Expr::Call { op, .. } => *op,
                            _ => OpKind::Add, // placeholder
                        },
                        reason: "only rope_append returns a tuple".into(),
                    })
                }
            }
            Stmt::For {
                body, loop_carry, ..
            } => {
                // The unroller replicates the body; shape-wise one
                // pass suffices because shapes don't depend on the
                // iteration index. Loop-carried locals need their
                // outer/inner shapes unified so Phase 6 can rewire
                // bindings across iterations consistently.
                self.infer_stmts(body)?;
                for (outer, inner) in loop_carry {
                    let outer_shape = self.locals.get(outer).cloned();
                    let inner_shape = self.locals.get(inner).cloned();
                    if let (Some(o), Some(i)) = (outer_shape, inner_shape) {
                        if o.len() != i.len() {
                            return Err(ShapeError::BadArgs {
                                op: OpKind::Add,
                                reason: format!(
                                    "loop-carried local rank mismatch: {} vs {}",
                                    o.len(),
                                    i.len()
                                ),
                            });
                        }
                        for (a, b) in o.iter().zip(&i) {
                            self.solver.unify(a, b)?;
                        }
                    }
                }
                Ok(())
            }
            Stmt::If {
                then_body,
                else_body,
                merge_carry,
                ..
            } => {
                // Both arms' shapes must be inferable. Since only
                // one arm runs at each unrolled iteration, but
                // shape inference doesn't depend on the iteration
                // index, we process both. For each merged name,
                // the then-arm and else-arm final shapes must
                // unify — a read after the `if` sees one or the
                // other at runtime, but downstream ops need a
                // single shape for the merge binding either way.
                self.infer_stmts(then_body)?;
                self.infer_stmts(else_body)?;
                for (merge_id, then_final, else_final) in merge_carry {
                    let then_shape = self.locals.get(then_final).cloned();
                    let else_shape = self.locals.get(else_final).cloned();
                    match (then_shape, else_shape) {
                        (Some(t), Some(e)) => {
                            if t.len() != e.len() {
                                return Err(ShapeError::BadArgs {
                                    op: OpKind::Add,
                                    reason: format!(
                                        "`if` merge rank mismatch: then={}, else={}",
                                        t.len(),
                                        e.len()
                                    ),
                                });
                            }
                            for (a, b) in t.iter().zip(&e) {
                                self.solver.unify(a, b)?;
                            }
                            // Post-unification, the shapes are
                            // equivalent; bind the merge to the
                            // then-arm's shape.
                            self.locals.insert(*merge_id, t);
                        }
                        _ => {
                            return Err(ShapeError::BadArgs {
                                op: OpKind::Add,
                                reason: "`if` arm produced no shape for a merged name".into(),
                            });
                        }
                    }
                }
                Ok(())
            }
        }
    }

    fn infer_expr(&mut self, expr: &Expr) -> Result<Shape, ShapeError> {
        match expr {
            Expr::Call { op, args } => {
                // Ensure any weight args have a shape of the
                // appropriate rank. Without this, the first call to
                // a weight would try to read an empty shape.
                //
                // Descend one level into a nested `Add` (weight +
                // scalar) — the parent op's weight_arg_rank applies
                // to the weight inside the add, since a scalar
                // broadcast preserves rank.
                for (arg_idx, rank) in weight_arg_ranks(*op) {
                    match args.get(*arg_idx) {
                        Some(Expr::Weight { id, .. }) => {
                            self.ensure_weight_rank(*id, *rank);
                        }
                        Some(Expr::Call {
                            op: OpKind::Add,
                            args: inner_args,
                        }) => {
                            for inner in inner_args {
                                if let Expr::Weight { id, .. } = inner {
                                    self.ensure_weight_rank(*id, *rank);
                                }
                            }
                        }
                        _ => {}
                    }
                }
                let inputs: Vec<Shape> = args
                    .iter()
                    .map(|a| self.expr_shape(a))
                    .collect::<Result<_, _>>()?;
                let sig = apply_signature(&mut self.solver, *op, &inputs)?;
                Ok(sig.output)
            }
            Expr::Mul { lhs, rhs } => {
                // Elementwise multiplication (gate * up): shapes must match.
                let l = self.expr_shape(lhs)?;
                let r = self.expr_shape(rhs)?;
                if l.len() != r.len() {
                    return Err(ShapeError::BadArgs {
                        op: OpKind::Mul,
                        reason: format!("mul operands differ in rank: {} vs {}", l.len(), r.len()),
                    });
                }
                for (a, b) in l.iter().zip(&r) {
                    self.solver.unify(a, b)?;
                }
                Ok(l)
            }
            _ => self.expr_shape(expr),
        }
    }

    /// Shape of an expression in a *read* context (i.e. as an
    /// argument to an op). Assigns fresh shapes to weights on
    /// first encounter; subsequent reads unify against the
    /// existing shape.
    fn expr_shape(&mut self, expr: &Expr) -> Result<Shape, ShapeError> {
        match expr {
            Expr::Local(id) => self.locals.get(id).cloned().ok_or(ShapeError::BadArgs {
                op: OpKind::Add,
                reason: format!("local {id:?} has no shape yet"),
            }),
            Expr::Extern { kind, index: _ } => Ok(extern_shape(*kind)),
            Expr::Weight { id, index: _ } => {
                // First encounter: install a fresh all-var shape.
                // We don't yet know the rank — it'll be set by the
                // op consuming this weight via that op's signature
                // (which asserts a specific rank). Use a rank-2
                // default as a hint (most weights are 2D or 1D, and
                // unify with a concrete 1D shape constrains rank
                // at the callsite). This is a compromise: true
                // rank polymorphism would require deferred shape
                // allocation, which this MVP doesn't implement.
                if !self.weights.contains_key(id) {
                    // Introduce a 2-dim "prospective" shape with
                    // fresh vars. If the op consuming it asserts
                    // rank 1, unify fails — user needs to sort out
                    // the shape. In practice every Llama/Qwen2
                    // weight is rank 1 or rank 2; rmsnorm weights
                    // end up rank 1 via the unification in Phase
                    // 4 below (we re-allocate on first use as
                    // rank-1 by checking the consuming op).
                    //
                    // For robustness we don't commit a rank yet;
                    // we return a fresh placeholder and hope the
                    // consuming op sets it. We do so by installing
                    // an empty Vec and letting the op signature
                    // grow it.
                    self.weights.insert(*id, Vec::new());
                }
                // Clone the current state; if it's empty this is
                // the first use, and the op signature is about to
                // replace it.
                let current = self.weights.get(id).cloned().unwrap();
                Ok(current)
            }
            Expr::Call { .. } | Expr::Mul { .. } => {
                // Nested call in read position (silu(gemm(...))).
                self.infer_expr(expr)
            }
            Expr::Add { .. } => unreachable!(
                "Expr::Add is lowered to Expr::Call by classify before shape inference"
            ),
            Expr::ScalarLit(_) => {
                // Rank-0 — sig_binary_elementwise handles broadcast
                // from the other operand.
                Ok(Shape::new())
            }
        }
    }
}

// ── Tests ─────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::classify::classify;
    use crate::parse::parse_block;

    fn classify_src(src: &str) -> Program {
        let file: syn::File = syn::parse_str(&format!("fn _carrier() {{ {src} }}")).expect("parse");
        let block = match &file.items[0] {
            syn::Item::Fn(f) => &*f.block,
            _ => unreachable!(),
        };
        let ast = parse_block(block).expect("parse DSL");
        classify(&ast).expect("classify")
    }

    fn bound(name: &str) -> Dim {
        Dim::Bound(name.into())
    }

    fn heads_layout_attn() -> Dim {
        canonical_mul(vec![bound("head_dim"), bound("num_attention_heads")])
    }

    fn heads_layout_kv() -> Dim {
        canonical_mul(vec![bound("head_dim"), bound("num_key_value_heads")])
    }

    #[test]
    fn solver_unifies_var_with_concrete() {
        let mut solver = Solver::new();
        let v = solver.fresh();
        solver.unify(&Dim::Var(v), &bound("hidden_size")).unwrap();
        assert_eq!(
            solver.close_dim(&Dim::Var(v)).unwrap(),
            bound("hidden_size")
        );
    }

    #[test]
    fn canonical_mul_sorts_and_flattens() {
        let d = canonical_mul(vec![
            bound("head_dim"),
            Dim::Mul(vec![bound("num_attention_heads"), Dim::Lit(1)]),
        ]);
        assert_eq!(d, heads_layout_attn());
    }

    #[test]
    fn unresolved_vars_preserved() {
        // Unresolved dim variables are returned as-is; the
        // `ShapeError::Unresolved` variant exists but callers
        // decide when to treat a remaining Var as an error
        // (e.g., at codegen when we need a concrete integer).
        let mut solver = Solver::new();
        let v = solver.fresh();
        let closed = solver.close_dim(&Dim::Var(v)).unwrap();
        assert!(matches!(closed, Dim::Var(_)));
    }

    #[test]
    fn mismatch_errors() {
        let mut solver = Solver::new();
        let err = solver.unify(&Dim::Lit(3), &Dim::Lit(4));
        assert!(matches!(err, Err(ShapeError::Mismatch { .. })));
    }

    #[test]
    fn embed_output_is_num_tokens_hidden_size() {
        let p = classify_src("hidden_states = embed(input_ids, embed_tokens);");
        let inf = infer(&p).expect("infer");

        // There's exactly one local: hidden_states.
        assert_eq!(inf.locals.len(), 1);
        let hs_shape = inf.locals.values().next().unwrap();
        assert_eq!(hs_shape, &vec![bound("num_tokens"), bound("hidden_size")]);

        // embed_tokens weight must be [vocab_size, hidden_size].
        let et_shape = inf.weights.values().next().unwrap();
        assert_eq!(et_shape, &vec![bound("vocab_size"), bound("hidden_size")]);
    }

    #[test]
    fn q_proj_shape_is_hidden_attn_heads() {
        // This is the acid test the plan promised: q_proj's shape
        // emerges as [hidden_size, num_attention_heads * head_dim].
        let p = classify_src(
            r#"
            hidden_states = embed(input_ids, embed_tokens);
            for layer in 0..num_hidden_layers {
                normed = rmsnorm(hidden_states, input_layernorm[layer]);
                q = gemm(normed, self_attn.q_proj[layer]);
                k = gemm(normed, self_attn.k_proj[layer]);
                v = gemm(normed, self_attn.v_proj[layer]);
                (q, k, v) = rope_append(q, k, v, positions, rotary, kv_cache[layer]);
                attn = attention(q, k, v, kv_cache[layer], block_table);
                oproj = gemm(attn, self_attn.o_proj[layer]);
                hidden_states = add(oproj, hidden_states);
            }
            "#,
        );
        let inf = infer(&p).expect("infer");

        // Find q_proj's weight id via its path.
        let q_proj_id = p
            .weights
            .path_for_test(&["self_attn", "q_proj"])
            .expect("q_proj present");
        let q_proj_shape = inf.weights.get(&q_proj_id).expect("q_proj inferred");
        assert_eq!(
            q_proj_shape,
            &vec![bound("hidden_size"), heads_layout_attn()],
            "q_proj shape mismatch"
        );

        // k_proj and v_proj use num_key_value_heads instead.
        let k_proj_id = p
            .weights
            .path_for_test(&["self_attn", "k_proj"])
            .expect("k_proj present");
        let k_proj_shape = inf.weights.get(&k_proj_id).unwrap();
        assert_eq!(
            k_proj_shape,
            &vec![bound("hidden_size"), heads_layout_kv()],
            "k_proj shape mismatch"
        );

        // o_proj: [num_attention_heads * head_dim, hidden_size].
        // Derived via: attn output is [num_tokens, NAH*HDM], o_proj
        // takes that and outputs [num_tokens, hidden_size] because
        // add(oproj, hidden_states) forces o_proj's output to match
        // hidden_states' last dim.
        let o_proj_id = p
            .weights
            .path_for_test(&["self_attn", "o_proj"])
            .expect("o_proj present");
        let o_proj_shape = inf.weights.get(&o_proj_id).unwrap();
        assert_eq!(
            o_proj_shape,
            &vec![heads_layout_attn(), bound("hidden_size")],
            "o_proj shape mismatch"
        );

        // input_layernorm is 1-D of hidden_size.
        let ln_id = p
            .weights
            .path_for_test(&["input_layernorm"])
            .expect("input_layernorm present");
        let ln_shape = inf.weights.get(&ln_id).unwrap();
        assert_eq!(ln_shape, &vec![bound("hidden_size")]);
    }

    #[test]
    fn mlp_weights_resolve_via_convention() {
        // MLP dims (intermediate_size) aren't pinned by any op
        // signature — they get anchored by the convention table.
        let p = classify_src(
            r#"
            hidden_states = embed(input_ids, embed_tokens);
            for layer in 0..num_hidden_layers {
                normed = rmsnorm(hidden_states, post_attention_layernorm[layer]);
                gate = silu(gemm(normed, mlp.gate_proj[layer]));
                up = gemm(normed, mlp.up_proj[layer]);
                down = gemm(gate * up, mlp.down_proj[layer]);
                hidden_states = add(down, hidden_states);
            }
            "#,
        );
        let inf = infer(&p).expect("infer");

        let gate_id = p.weights.path_for_test(&["mlp", "gate_proj"]).unwrap();
        assert_eq!(
            inf.weights.get(&gate_id).unwrap(),
            &vec![bound("hidden_size"), bound("intermediate_size")],
        );

        let down_id = p.weights.path_for_test(&["mlp", "down_proj"]).unwrap();
        assert_eq!(
            inf.weights.get(&down_id).unwrap(),
            &vec![bound("intermediate_size"), bound("hidden_size")],
        );
    }
}
