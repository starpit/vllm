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
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
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
pub fn canonical_mul(mut children: Vec<Dim>) -> Dim {
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
    /// A mismatch that the caller can resolve by inserting `Reshape`
    /// statements into the program. Typical case: per-head operations
    /// like Qwen3's QK-norm where the DSL writes `rmsnorm(q, q_norm)`
    /// with `q: [..., heads * head_dim]` and `q_norm: [head_dim]`.
    /// `apply_reshape_hints` mutates a program clone to insert view
    /// reshapes, after which a second `infer` pass succeeds cleanly.
    ReshapeRecovery {
        hints: Vec<ReshapeHint>,
    },
}

/// A single reshape-recoverable mismatch recorded during anchoring.
/// Produced by [`infer`] when a weight's `standard_shape` declaration
/// implies the activation reaching its consumer op needs a view
/// change. Consumed by [`apply_reshape_hints`] to rewrite the program.
#[derive(Debug, Clone)]
pub struct ReshapeHint {
    /// The producer local whose output shape mismatches the consumer's
    /// expectation. A new local carrying the reshaped view will be
    /// introduced; the consumer stmt's reference to `producer_local`
    /// is rewritten to point at it.
    pub producer_local: LocalId,
    /// Full target shape for the reshaped view — this is what the
    /// consumer's op signature expects the producer to look like.
    /// Element count must equal the producer's current element count.
    pub target_shape: Shape,
    /// The weight whose declared shape triggered the recovery — kept
    /// for diagnostics.
    pub weight_id: WeightId,
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
            Self::ReshapeRecovery { hints } => {
                write!(
                    f,
                    "shape mismatch recoverable via {} reshape insertion(s)",
                    hints.len()
                )
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
        // LayerNorm has identical shape constraints to RmsNorm —
        // `(x: [..., H], w: [H]) -> [..., H]`. Same signature reused.
        OpKind::LayerNorm => sig_rmsnorm(solver, inputs),
        OpKind::Gemm => sig_gemm(solver, inputs),
        OpKind::RopeAppend => sig_rope_append(solver, inputs),
        // Same q/k/v constraints as `rope_append`; the distinction is
        // in the picked kernel (interleaved pair vs. NeoX), not in
        // the type signature.
        OpKind::RopeAppendInterleaved => sig_rope_append(solver, inputs),
        OpKind::Attention => sig_attention(solver, inputs),
        // Same q/k/v constraints as `attention`; the distinction is
        // in the picked kernel (window-masked vs. dense), not in
        // the type signature.
        OpKind::SlidingAttention => sig_attention(solver, inputs),
        // Vision varlen attention: q/k/v + cu_seqlens + max_seqlen.
        // No heads-layout anchoring — vision shapes are pinned at
        // the qkv-gemm weight, not at attention. See `sig_varlen_attention`.
        OpKind::VarlenAttention => sig_varlen_attention(solver, inputs),
        OpKind::Silu => sig_unary_elementwise(solver, inputs, op),
        OpKind::Gelu => sig_unary_elementwise(solver, inputs, op),
        // QuickGelu / GeluErf are unary elementwise like Gelu; the
        // numerical distinction lives in the picked kernel.
        OpKind::QuickGelu => sig_unary_elementwise(solver, inputs, op),
        OpKind::GeluErf => sig_unary_elementwise(solver, inputs, op),
        OpKind::TanhSoftCap => sig_unary_elementwise(solver, inputs, op),
        OpKind::Add => sig_binary_elementwise(solver, inputs, op),
        // AllReduce is identity-shape one-input — same constraint
        // as Silu / Gelu, just with the all-reduce-sum semantics
        // tracked at the OpKind level. Never inserted by any DSL;
        // the lowering pass produces it post-FUF-build, so this arm
        // exists only to keep `apply_signature` total.
        OpKind::AllReduce => sig_unary_elementwise(solver, inputs, op),
        // AllGather's output shape is the input shape with the last
        // dim multiplied by `tp_world_size` — not expressible as a
        // pure function of the input Shape alone (depends on the
        // current build's `tp_world_size` literal). Like `Reshape`,
        // the lowering pass writes the concrete output shape onto
        // the `FufNode` directly when constructing it; the FUF-level
        // shape unifier never re-derives it. If something ever
        // reaches this arm via `apply_signature`, it's a compiler
        // bug — return an explicit error instead of a silent lie.
        OpKind::AllGather => Err(ShapeError::BadArgs {
            op: OpKind::AllGather,
            reason: "apply_signature should not be called on AllGather; \
                     the lowering pass sets FufNode.outputs[0] to \
                     `[..in_shape with last_dim * tp_world_size]` \
                     directly when inserting the node post-FUF-build"
                .into(),
        }),
        OpKind::BiasAdd => sig_bias_add(solver, inputs),
        OpKind::Mul => sig_binary_elementwise(solver, inputs, op),
        // Reshape's output shape is stored on `Program::reshape_targets`
        // keyed by the stmt's target LocalId, so `apply_signature` isn't
        // a useful entry point for it — `infer_stmt` routes around this
        // arm. If something reaches here it's a compiler bug.
        OpKind::Reshape => Err(ShapeError::BadArgs {
            op: OpKind::Reshape,
            reason: "apply_signature should not be called on Reshape; \
                     infer_stmt looks up the target shape from \
                     Program::reshape_targets directly"
                .into(),
        }),
        // VisionRope is tuple-returning `(q', k') = vision_rope(q, k, cos, sin)`;
        // `Stmt::AssignTuple` handles the 2-target binding directly. Reaching
        // this arm via `apply_signature` (single-target) is fine — returns q's
        // shape, same shape-preserving semantics.
        OpKind::VisionRope => sig_vision_rope(solver, inputs),
        OpKind::MlaSplit => sig_mla_split(solver, inputs),
        OpKind::MlaAttention => sig_mla_attention(solver, inputs),
        OpKind::Moe => sig_moe(solver, inputs),
        // MmEmbedSplice: identity-shape one-input in-place. Same
        // signature as AllReduce — the splice mutates the embed
        // output buffer, shape unchanged. Never inserted by the DSL;
        // the lowering pass `insert_mm_splices` writes the node
        // post-FUF-build with outputs[0] == inputs[0], so reaching
        // this arm via `apply_signature` is fine: sig_unary_elementwise
        // re-unifies and agrees.
        OpKind::MmEmbedSplice => sig_unary_elementwise(solver, inputs, op),
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

/// `bias_add(x: [..., D], b: [D])` → `[..., D]`. Same shape math as
/// `rmsnorm` — both broadcast a rank-1 parameter over the last axis
/// of the activation. Kept distinct so error messages name the right
/// op and so `OpKind::BiasAdd` participates in its own fusion patterns.
fn sig_bias_add(solver: &mut Solver, inputs: &[Shape]) -> Result<OpSig, ShapeError> {
    expect_args(OpKind::BiasAdd, inputs, 2)?;
    let x = &inputs[0];
    let b = &inputs[1];
    if x.is_empty() {
        return Err(ShapeError::BadArgs {
            op: OpKind::BiasAdd,
            reason: "input must have rank >= 1".into(),
        });
    }
    if b.len() != 1 {
        return Err(ShapeError::BadArgs {
            op: OpKind::BiasAdd,
            reason: format!("bias must have rank 1, got {}", b.len()),
        });
    }
    solver.unify(x.last().unwrap(), &b[0])?;
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

/// `varlen_attention(q, k, v, cu_seqlens, max_seqlen)` →
/// `q.shape`. Vision-encoder attention. Inputs: q/k/v all rank-2
/// `[total_L, num_heads * head_dim]` (vision-side; the heads-layout
/// anchoring lives at the qkv-producing gemm, not here). cu_seqlens
/// and max_seqlen are opaque externs with empty shapes.
fn sig_varlen_attention(_solver: &mut Solver, inputs: &[Shape]) -> Result<OpSig, ShapeError> {
    expect_args(OpKind::VarlenAttention, inputs, 5)?;
    let q = &inputs[0];
    if q.is_empty() {
        return Err(ShapeError::BadArgs {
            op: OpKind::VarlenAttention,
            reason: "q must have rank >= 1".into(),
        });
    }
    Ok(OpSig { output: q.clone() })
}

/// `vision_rope(q, k, cos, sin)` → `(q', k')`. Single-output sig
/// returns q's shape; `Stmt::AssignTuple` binds the second target
/// to k's shape. Shape-preserving on both q and k. cos/sin are
/// opaque externs (built host-side from grid_thw); empty shapes.
fn sig_vision_rope(_solver: &mut Solver, inputs: &[Shape]) -> Result<OpSig, ShapeError> {
    expect_args(OpKind::VisionRope, inputs, 4)?;
    let q = &inputs[0];
    if q.is_empty() {
        return Err(ShapeError::BadArgs {
            op: OpKind::VisionRope,
            reason: "q must have rank >= 1".into(),
        });
    }
    Ok(OpSig { output: q.clone() })
}

/// `mla_split(kv_a: [T, kv_lora_rank + qk_rope_head_dim])` → (used as
/// single-output sig by `apply_signature`; actual 2-tuple binding happens
/// in `Stmt::AssignTuple` which calls this path first for validation).
/// Returns `kv_latent` shape `[T, kv_lora_rank]`.
fn sig_mla_split(_solver: &mut Solver, inputs: &[Shape]) -> Result<OpSig, ShapeError> {
    expect_args(OpKind::MlaSplit, inputs, 1)?;
    let kv_a = &inputs[0];
    if kv_a.is_empty() {
        return Err(ShapeError::BadArgs {
            op: OpKind::MlaSplit,
            reason: "kv_a must have rank >= 1".into(),
        });
    }
    let t_dim = kv_a[0].clone();
    Ok(OpSig {
        output: vec![t_dim, Dim::Bound("kv_lora_rank".into())],
    })
}

/// `mla_attention(q, kv_b, k_pe, positions, rotary, kv_cache, block_table)`
/// → `[T, num_attention_heads * v_head_dim]`.
/// The last 4 args are opaque externs (positions, rotary, kv_cache,
/// block_table) with empty shapes; we only look at `q`'s token dim.
fn sig_mla_attention(_solver: &mut Solver, inputs: &[Shape]) -> Result<OpSig, ShapeError> {
    expect_args(OpKind::MlaAttention, inputs, 7)?;
    let q = &inputs[0];
    if q.is_empty() {
        return Err(ShapeError::BadArgs {
            op: OpKind::MlaAttention,
            reason: "q must have rank >= 1".into(),
        });
    }
    let out_dim = Dim::Mul(vec![
        Dim::Bound("num_attention_heads".into()),
        Dim::Bound("v_head_dim".into()),
    ]);
    Ok(OpSig {
        output: vec![q[0].clone(), out_dim],
    })
}

/// `moe_block(x: [T, H], moe_weight)` → `[T, H]`. Shape-preserving;
/// the second arg is a MoE layer struct (`FusedMoELayer` /
/// `SharedFusedMoELayer` / `DeepSeekV2MoELayer` or their quant
/// flavors) — not a tensor — so it contributes an empty shape.
/// Output = hidden_states shape.
fn sig_moe(_solver: &mut Solver, inputs: &[Shape]) -> Result<OpSig, ShapeError> {
    expect_args(OpKind::Moe, inputs, 2)?;
    Ok(OpSig {
        output: inputs[0].clone(),
    })
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
        OpKind::LayerNorm => &[(1, 1)],
        OpKind::Gemm => &[(1, 2)],
        OpKind::RopeAppend => &[],
        OpKind::RopeAppendInterleaved => &[],
        OpKind::Attention => &[],
        OpKind::SlidingAttention => &[],
        OpKind::VarlenAttention => &[],
        OpKind::Silu => &[],
        OpKind::Gelu => &[],
        OpKind::QuickGelu => &[],
        OpKind::GeluErf => &[],
        OpKind::VisionRope => &[],
        OpKind::TanhSoftCap => &[],
        OpKind::Add => &[],
        // AllReduce takes one activation input, no tensor weight.
        OpKind::AllReduce => &[],
        // AllGather takes one activation input, no tensor weight.
        OpKind::AllGather => &[],
        OpKind::BiasAdd => &[(1, 1)],
        OpKind::Mul => &[],
        OpKind::Reshape => &[],
        // MlaSplit takes 1 activation input; no tensor weight args.
        OpKind::MlaSplit => &[],
        // MlaAttention takes activation inputs + opaque externs; no tensor weight args.
        OpKind::MlaAttention => &[],
        // Moe's moe[layer] is a struct (not a tensor) at arg 1;
        // weight_arg_ranks governs shape-rank assertion only, so we skip it.
        OpKind::Moe => &[],
        // MmEmbedSplice takes one activation input, no tensor weight.
        OpKind::MmEmbedSplice => &[],
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
        ExternKind::RotaryLocal => vec![],
        ExternKind::BlockTable => vec![],
        ExternKind::KvCache => vec![],
        // Vision externs. `Pixels` has the per-row patch shape that
        // shape inference needs to anchor the encoder's first GEMM
        // (`patch_embed_proj`); `Cos`/`Sin` carry the per-row half-
        // dim RoPE tables. The remaining vision externs are opaque
        // (varlen index / per-image grid / scalar) — same role as
        // the decoder's `BlockTable`/`KvCache`. Bound names anchor
        // to vision-config fields populated by per-arch crates in
        // G.5; until then they're symbolic placeholders that resolve
        // only when a real vision config is loaded.
        ExternKind::Pixels => vec![
            Dim::Bound("num_tokens".into()),
            Dim::Bound("vision_in_features".into()),
        ],
        ExternKind::Cos => vec![
            Dim::Bound("num_tokens".into()),
            Dim::Bound("vision_rope_half_dim".into()),
        ],
        ExternKind::Sin => vec![
            Dim::Bound("num_tokens".into()),
            Dim::Bound("vision_rope_half_dim".into()),
        ],
        ExternKind::CuSeqlens => vec![],
        ExternKind::GridThw => vec![],
        ExternKind::MaxSeqlen => vec![],
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
///
/// Anchors weight shapes against the per-arch `weights.json`
/// manifest (loaded by the macro and passed in via `manifest`).
/// `bounds` come from one of the arch's configs — the prober's
/// cross-size validation guarantees any config's values resolve the
/// manifest's formulas consistently, so bounds from any one size
/// suffice.
///
/// Anchoring uses **numerical equivalence** rather than structural
/// unification: if the dataflow-inferred dim and the manifest's
/// declared dim evaluate to the same integer under `bounds`, they
/// match even when their symbolic forms differ (e.g. Qwen2.5 where
/// `hidden_size == num_attention_heads * head_dim` is always true).
///
/// On a genuine mismatch — inferred and declared don't even
/// resolve to the same integer — the recovery machinery kicks in
/// (per-head norms like Qwen3's `q_norm` where dataflow produces
/// `[heads * head_dim]` but the manifest declares `[head_dim]`).
/// `ShapeError::ReshapeRecovery` then carries hints the caller
/// passes to [`apply_reshape_hints`] to synthesize `Reshape` tiles.
pub fn infer(
    program: &Program,
    manifest: &crate::weights_manifest::WeightsManifest,
    bounds: &std::collections::BTreeMap<String, u64>,
) -> Result<Inferred, ShapeError> {
    let mut cx = InferCtx::from_program(program);
    cx.infer_stmts(&program.statements, program)?;

    // Anchor weights against the manifest. Dataflow has pinned every
    // weight dim it can by now (q_proj[1] via rope_append,
    // input_layernorm via rmsnorm-of-hidden_states, etc.); the
    // manifest adds arch-specific declarations dataflow can't derive
    // (the q_norm / k_norm case).
    //
    // For numerical-equivalence anchoring: if the inferred dim and
    // the declared dim evaluate to the same integer under `bounds`,
    // no action needed — they agree. If the integers differ,
    // `try_anchor_with_recovery` checks whether the discrepancy is
    // a reshape-recoverable factor relationship and, if so, records
    // a hint.
    let mut reshape_hints: Vec<ReshapeHint> = Vec::new();
    // Sort by WeightId so projections (encountered earlier in the DSL)
    // are anchored before norms. Without this, HashMap iteration order
    // can anchor a per-head norm weight (q_norm: [head_dim]) before
    // its associated projection (q_proj: [hidden, heads*head_dim]),
    // setting the norm dim to head_dim via unification — then the
    // projection anchoring hits a Mismatch(head_dim, heads*head_dim)
    // that detect_reshape_hint can't recover (wrong direction).
    let mut weight_ids: Vec<_> = cx.weights.keys().copied().collect();
    weight_ids.sort_by_key(|w| w.0);
    for wid in &weight_ids {
        let shape = cx.weights.get(wid).cloned().unwrap();
        let shape = &shape;
        let path: Vec<String> = program
            .weights
            .path(*wid)
            .iter()
            .map(|i| i.to_string())
            .collect();
        if let Some(declared) = manifest.lookup(&path)
            && shape.len() == declared.len()
        {
            try_anchor_with_recovery(
                &mut cx,
                program,
                *wid,
                shape,
                declared,
                bounds,
                &mut reshape_hints,
            )?;
        }
    }
    if !reshape_hints.is_empty() {
        return Err(ShapeError::ReshapeRecovery {
            hints: reshape_hints,
        });
    }

    // Close every recorded shape. Any remaining Vars correspond to
    // weights the manifest didn't cover AND dataflow couldn't pin —
    // left as `Dim::Var` for downstream consumers to handle.
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

/// Evaluate a `Dim` to a concrete `u64` using `bounds`, walking
/// through the solver first to resolve any `Var`s. Returns `None`
/// if the dim doesn't close to a known bound / literal / product of
/// those (e.g. an unresolved Var) — the caller treats that as
/// "can't compare numerically."
fn eval_dim_to_u64(
    cx: &mut InferCtx,
    d: &Dim,
    bounds: &std::collections::BTreeMap<String, u64>,
) -> Option<u64> {
    let walked = cx.solver.walk(d);
    eval_closed_dim(&walked, bounds)
}

pub(crate) fn eval_closed_dim(
    d: &Dim,
    bounds: &std::collections::BTreeMap<String, u64>,
) -> Option<u64> {
    match d {
        Dim::Lit(n) => Some(*n),
        Dim::Bound(name) => bounds.get(name).copied(),
        Dim::Mul(factors) => factors
            .iter()
            .try_fold(1u64, |acc, f| eval_closed_dim(f, bounds).map(|v| acc * v)),
        Dim::Var(_) => None,
    }
}

/// Anchor a weight's shape to its manifest declaration. Uses
/// numerical equivalence: a dim pair agrees when `eval(inferred,
/// bounds) == eval(declared, bounds)`, regardless of symbolic form.
/// This handles the Qwen2.5-style "coincidence" where `hidden_size`
/// and `num_attention_heads * head_dim` are always numerically
/// equal — both forms validate.
///
/// On a genuine numeric mismatch, attempts reshape recovery: if the
/// inferred dim is a `Mul` containing the declared dim as a factor
/// (Qwen3/Gemma3 per-head norm pattern), records a `ReshapeHint`.
/// Otherwise, returns `ShapeError::Mismatch`.
fn try_anchor_with_recovery(
    cx: &mut InferCtx,
    program: &Program,
    wid: WeightId,
    inferred_shape: &Shape,
    declared_shape: &Shape,
    bounds: &std::collections::BTreeMap<String, u64>,
    hints: &mut Vec<ReshapeHint>,
) -> Result<(), ShapeError> {
    for (inferred, declared) in inferred_shape.iter().zip(declared_shape.iter()) {
        // Try structural unification first. This (a) short-circuits
        // when inferred and declared have the same symbolic form,
        // and (b) binds any unresolved `Var` on one side to the
        // concrete `Bound`/`Mul` on the other — the whole reason
        // the manifest exists for weights dataflow can't pin.
        if cx.solver.unify(inferred, declared).is_ok() {
            continue;
        }

        // Structural unify rejected because both sides are concrete
        // and symbolically different. Check numerical equivalence
        // under this model's bounds: `hidden_size` and
        // `num_attention_heads * head_dim` may always evaluate to
        // the same integer (Qwen2.5). If yes, accept — the shapes
        // agree at runtime regardless of symbolic form.
        let inferred_walked = cx.solver.walk(inferred);
        let declared_walked = cx.solver.walk(declared);
        let inf_n = eval_dim_to_u64(cx, &inferred_walked, bounds);
        let dec_n = eval_dim_to_u64(cx, &declared_walked, bounds);
        if let (Some(a), Some(b)) = (inf_n, dec_n)
            && a == b
        {
            continue;
        }

        // Genuine mismatch. Try reshape recovery for the
        // per-head-norm pattern. `detect_reshape_hint` returns the
        // split (before-consumer) + flatten-back (after-consumer)
        // hint pair — or empty if the pattern doesn't fit.
        let recovered = detect_reshape_hint(program, cx, wid, &inferred_walked, &declared_walked);
        if !recovered.is_empty() {
            hints.extend(recovered);
            return Ok(());
        }
        return Err(ShapeError::Mismatch {
            lhs: inferred_walked,
            rhs: declared_walked,
        });
    }
    Ok(())
}

/// If `inferred = Mul([.., Declared, ..])` and `declared` is a single
/// `Bound`/`Lit`, produce the hint pair describing the reshape bridge
/// that makes the consumer's activation tile line up:
///
/// - **Split hint**: reshape the activation feeding the consumer from
///   `[.., heads*head_dim]` to `[.., heads, head_dim]` so the
///   consumer's signature (e.g. `rmsnorm(x, w)` with `w: [head_dim]`)
///   unifies cleanly on the split last axis.
/// - **Flatten hint**: reshape the consumer's OUTPUT back from
///   `[.., heads, head_dim]` to the original `[.., heads*head_dim]`
///   layout so any downstream op (e.g. `rope_append`) that expects
///   the flat `heads*head_dim` layout sees the correct shape. This
///   is a no-op at the `apply_reshape_hints` pass when the consumer
///   has no downstream reader (its `rewrite_and_insert` call finds
///   nothing to rewrite and inserts nothing).
///
/// Returns an empty vec if the mismatch doesn't fit the "axis-factor"
/// pattern — at which point the caller treats it as a hard Mismatch.
fn detect_reshape_hint(
    program: &Program,
    cx: &InferCtx,
    wid: WeightId,
    inferred: &Dim,
    declared: &Dim,
) -> Vec<ReshapeHint> {
    // Only handle the "inferred is Mul containing declared as a
    // factor" shape. Bail on any other combination for now.
    let Dim::Mul(factors) = inferred else {
        return Vec::new();
    };
    // The factors OTHER than `declared` form the prefix that gets
    // split out into a new axis.
    let mut split_factors: Vec<Dim> = Vec::with_capacity(factors.len());
    let mut found_declared = false;
    for f in factors {
        if !found_declared && dims_structurally_equal(f, declared) {
            found_declared = true;
            continue;
        }
        split_factors.push(f.clone());
    }
    if !found_declared || split_factors.is_empty() {
        return Vec::new();
    }
    // Find the consumer stmt that reads this weight: its activation
    // input (the upstream producer) and its output target.
    let Some((consumer_producer_local, consumer_shape, consumer_target_local)) =
        find_consumer_activation(program, &cx.locals, wid)
    else {
        return Vec::new();
    };
    // Split target: flatten the leading axes with the split factors
    // into a single leading dim, keeping `declared` as the trailing
    // axis. E.g. `[T, heads*head_dim]` with `declared = head_dim`
    // → `[T*heads, head_dim]`, NOT `[T, heads, head_dim]`.
    //
    // The 2D target matches the `rms_norm` kernel's shape
    // expectations (`input.dim(0) = rows`, `input.dim(1) =
    // hidden_size`), so each row gets normalized over `head_dim` —
    // exactly per-head RMS norm. A 3D target would mis-dispatch the
    // kernel (it reads `dim(1)` as hidden_size → `heads`, allocates
    // a too-small output, and illegal-memory-access on the first
    // write past row 0).
    if consumer_shape.is_empty() {
        return Vec::new();
    }
    let mut leading: Vec<Dim> = consumer_shape[..consumer_shape.len() - 1].to_vec();
    leading.extend(split_factors);
    let merged_leading = match leading.len() {
        0 => return Vec::new(),
        1 => leading.into_iter().next().unwrap(),
        _ => canonical_mul(leading),
    };
    let split_shape: Shape = vec![merged_leading, declared.clone()];

    // Flatten-back target: the consumer's output under first-pass
    // inference inherits the producer's original (flat) shape for
    // shape-preserving ops like `rmsnorm`, so `consumer_shape` is
    // exactly the layout we need the rmsnorm output to be viewed as
    // before `rope_append` reads it. If the consumer's output local
    // is not known (e.g. AssignTuple destructure), skip the flatten
    // hint — the caller only synthesizes for scalar-target consumers.
    let mut hints = vec![ReshapeHint {
        producer_local: consumer_producer_local,
        target_shape: split_shape,
        weight_id: wid,
    }];
    if let Some(target_local) = consumer_target_local {
        hints.push(ReshapeHint {
            producer_local: target_local,
            target_shape: consumer_shape,
            weight_id: wid,
        });
    }
    hints
}

/// Walk the program's statements looking for a call whose args include
/// `Expr::Weight { id: wid, .. }`. Return `(activation_local,
/// activation_shape, target_local)` where the activation is the first
/// `Expr::Local` arg of that call and `target_local` is the stmt's
/// `Assign` target (or `None` for `AssignTuple` destructure — the
/// current per-head-norm recovery only uses scalar-target consumers).
fn find_consumer_activation(
    program: &Program,
    locals: &HashMap<LocalId, Shape>,
    wid: WeightId,
) -> Option<(LocalId, Shape, Option<LocalId>)> {
    fn expr_has_weight(e: &Expr, wid: WeightId) -> bool {
        match e {
            Expr::Weight { id, .. } => *id == wid,
            Expr::Add { lhs, rhs, .. } | Expr::Mul { lhs, rhs, .. } => {
                expr_has_weight(lhs, wid) || expr_has_weight(rhs, wid)
            }
            Expr::Call { args, .. } => args.iter().any(|a| expr_has_weight(a, wid)),
            _ => false,
        }
    }

    /// If the weight is inside an `Add(Weight, Scalar)` or similar
    /// intermediary (Gemma's `w + 1.0`), find the producing stmt's
    /// target local and then look for the Call that consumes it.
    fn find_through_intermediary(
        stmts: &[Stmt],
        locals: &HashMap<LocalId, Shape>,
        intermediary_local: LocalId,
    ) -> Option<(LocalId, Shape, Option<LocalId>)> {
        for s in stmts {
            match s {
                Stmt::Assign { value, target } => {
                    if let Expr::Call { args, .. } = value
                        && args
                            .iter()
                            .any(|a| matches!(a, Expr::Local(id) if *id == intermediary_local))
                    {
                        let activation = args.iter().find_map(|a| match a {
                            Expr::Local(id) if *id != intermediary_local => Some(*id),
                            _ => None,
                        })?;
                        let shape = locals.get(&activation)?.clone();
                        return Some((activation, shape, Some(*target)));
                    }
                }
                Stmt::For { body, .. } => {
                    if let Some(hit) = find_through_intermediary(body, locals, intermediary_local) {
                        return Some(hit);
                    }
                }
                Stmt::If {
                    then_body,
                    else_body,
                    ..
                } => {
                    if let Some(hit) =
                        find_through_intermediary(then_body, locals, intermediary_local)
                    {
                        return Some(hit);
                    }
                    if let Some(hit) =
                        find_through_intermediary(else_body, locals, intermediary_local)
                    {
                        return Some(hit);
                    }
                }
                _ => {}
            }
        }
        None
    }

    fn walk(
        stmts: &[Stmt],
        locals: &HashMap<LocalId, Shape>,
        wid: WeightId,
    ) -> Option<(LocalId, Shape, Option<LocalId>)> {
        for s in stmts {
            match s {
                Stmt::Assign { value, target } => {
                    if let Expr::Call { args, .. } = value
                        && args.iter().any(|a| expr_has_weight(a, wid))
                    {
                        if let Some((activation, shape)) = args
                            .iter()
                            .find_map(|a| match a {
                                Expr::Local(id) => Some(*id),
                                _ => None,
                            })
                            .and_then(|a| locals.get(&a).map(|s| (a, s.clone())))
                        {
                            return Some((activation, shape, Some(*target)));
                        }
                        if let Some(hit) = find_through_intermediary(stmts, locals, *target) {
                            return Some(hit);
                        }
                    }
                    if expr_has_weight(value, wid)
                        && !matches!(value, Expr::Call { .. })
                        && let Some(hit) = find_through_intermediary(stmts, locals, *target)
                    {
                        return Some(hit);
                    }
                }
                Stmt::AssignTuple { value, .. } => {
                    if let Expr::Call { args, .. } = value
                        && args.iter().any(|a| expr_has_weight(a, wid))
                    {
                        let activation = args.iter().find_map(|a| match a {
                            Expr::Local(id) => Some(*id),
                            _ => None,
                        })?;
                        let shape = locals.get(&activation)?.clone();
                        return Some((activation, shape, None));
                    }
                }
                Stmt::For { body, .. } => {
                    if let Some(hit) = walk(body, locals, wid) {
                        return Some(hit);
                    }
                }
                Stmt::If {
                    then_body,
                    else_body,
                    ..
                } => {
                    if let Some(hit) = walk(then_body, locals, wid) {
                        return Some(hit);
                    }
                    if let Some(hit) = walk(else_body, locals, wid) {
                        return Some(hit);
                    }
                }
            }
        }
        None
    }
    walk(&program.statements, locals, wid)
}

/// Rewrite `program` in place to materialize every `ReshapeHint`:
/// - Allocate a fresh `LocalId` for the reshaped view.
/// - Record the hint's `target_shape` under that id in
///   `program.reshape_targets`.
/// - Walk the program; in every statement that reads `producer_local`
///   (directly, via `Expr::Local`), rewrite that read to point at the
///   new local.
/// - Insert the synthesized `Reshape` statement right before the
///   first rewritten stmt in each scope.
///
/// Intended to be called in the recovery loop: `infer` returns
/// `ReshapeRecovery { hints }`, caller applies via this fn, calls
/// `infer` again. A well-formed hint set resolves on the second pass.
pub fn apply_reshape_hints(program: &mut Program, hints: &[ReshapeHint]) {
    for hint in hints {
        let producer = hint.producer_local;
        let debug_name = program.locals.name(producer).clone();
        // Fresh local for the reshaped view. Name embeds the producer's
        // name for traceability in emitted source.
        let reshaped_name = syn::Ident::new(
            &format!("{}_reshaped", debug_name),
            proc_macro2::Span::call_site(),
        );
        let new_local = program.locals.push(reshaped_name);
        program
            .reshape_targets
            .insert(new_local, hint.target_shape.clone());

        // Rewrite every downstream read of `producer` to read the new
        // local instead, and insert a `Reshape` stmt at the first
        // rewrite site within each block. The walker tracks a flag
        // per block so the insertion happens exactly once (right
        // before the first consumer stmt) and subsequent consumers
        // just use the already-introduced reshape binding.
        rewrite_and_insert(&mut program.statements, producer, new_local, &mut false);
    }
}

/// Recurse through stmts. Whenever a stmt reads `producer`, rewrite
/// it to read `replacement` instead; before the first such stmt in
/// each block, inject the synthesized Reshape stmt.
fn rewrite_and_insert(
    stmts: &mut Vec<Stmt>,
    producer: LocalId,
    replacement: LocalId,
    inserted_in_this_block: &mut bool,
) {
    let mut i = 0;
    while i < stmts.len() {
        // Recurse first so nested rewrites happen before we touch this
        // stmt (for If/For arms).
        match &mut stmts[i] {
            Stmt::For { body, .. } => {
                let mut child_flag = false;
                rewrite_and_insert(body, producer, replacement, &mut child_flag);
            }
            Stmt::If {
                then_body,
                else_body,
                ..
            } => {
                let mut t_flag = false;
                rewrite_and_insert(then_body, producer, replacement, &mut t_flag);
                let mut e_flag = false;
                rewrite_and_insert(else_body, producer, replacement, &mut e_flag);
            }
            _ => {}
        }

        // Check if this stmt reads `producer` directly in its value
        // expression. If yes, rewrite the read AND (on first match in
        // this block) inject a Reshape stmt right before it.
        let reads_producer = stmt_reads_local(&stmts[i], producer);
        if reads_producer {
            if !*inserted_in_this_block {
                let reshape_stmt = Stmt::Assign {
                    target: replacement,
                    value: Expr::Call {
                        op: OpKind::Reshape,
                        args: vec![Expr::Local(producer)],
                    },
                };
                stmts.insert(i, reshape_stmt);
                *inserted_in_this_block = true;
                i += 1; // skip past the just-inserted reshape
            }
            rewrite_local_reads(&mut stmts[i], producer, replacement);
        }
        i += 1;
    }
}

/// True if `stmt`'s top-level `value` expression reads `producer`
/// directly (as `Expr::Local(producer)`) at any depth of nested
/// `Expr::Call` / `Expr::Mul` / `Expr::Add`. Doesn't recurse into
/// child blocks (`For::body`, `If::then_body`, `If::else_body`) —
/// that's handled by the outer walker.
fn stmt_reads_local(stmt: &Stmt, producer: LocalId) -> bool {
    let value = match stmt {
        Stmt::Assign { value, .. } => value,
        Stmt::AssignTuple { value, .. } => value,
        Stmt::For { .. } | Stmt::If { .. } => return false,
    };
    expr_reads_local(value, producer)
}

fn expr_reads_local(expr: &Expr, producer: LocalId) -> bool {
    match expr {
        Expr::Local(id) => *id == producer,
        Expr::Call { args, .. } => args.iter().any(|a| expr_reads_local(a, producer)),
        Expr::Mul { lhs, rhs } | Expr::Add { lhs, rhs } => {
            expr_reads_local(lhs, producer) || expr_reads_local(rhs, producer)
        }
        _ => false,
    }
}

/// Rewrite every `Expr::Local(producer)` read in `stmt`'s top-level
/// value expression to `Expr::Local(replacement)`. Doesn't recurse
/// into child blocks.
fn rewrite_local_reads(stmt: &mut Stmt, producer: LocalId, replacement: LocalId) {
    let value = match stmt {
        Stmt::Assign { value, .. } => value,
        Stmt::AssignTuple { value, .. } => value,
        Stmt::For { .. } | Stmt::If { .. } => return,
    };
    rewrite_expr_reads(value, producer, replacement);
}

fn rewrite_expr_reads(expr: &mut Expr, producer: LocalId, replacement: LocalId) {
    match expr {
        Expr::Local(id) if *id == producer => *id = replacement,
        Expr::Call { args, .. } => {
            for a in args {
                rewrite_expr_reads(a, producer, replacement);
            }
        }
        Expr::Mul { lhs, rhs } | Expr::Add { lhs, rhs } => {
            rewrite_expr_reads(lhs, producer, replacement);
            rewrite_expr_reads(rhs, producer, replacement);
        }
        _ => {}
    }
}

struct InferCtx {
    solver: Solver,
    locals: HashMap<LocalId, Shape>,
    weights: HashMap<WeightId, Shape>,
    /// Copy of `Program::reshape_targets` — on the second inference
    /// pass (after `apply_reshape_hints` has rewritten the program),
    /// synthesized `OpKind::Reshape` statements look up their output
    /// shape here instead of going through `apply_signature`.
    reshape_targets: HashMap<LocalId, Shape>,
}

impl InferCtx {
    fn from_program(program: &Program) -> Self {
        Self {
            solver: Solver::new(),
            locals: HashMap::new(),
            weights: HashMap::new(),
            reshape_targets: program.reshape_targets.clone(),
        }
    }

    fn infer_stmts(&mut self, stmts: &[Stmt], program: &Program) -> Result<(), ShapeError> {
        for s in stmts {
            self.infer_stmt(s, program)?;
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

    fn infer_stmt(&mut self, stmt: &Stmt, program: &Program) -> Result<(), ShapeError> {
        match stmt {
            Stmt::Assign { target, value } => {
                // Synthesized `Reshape` stmts bypass the signature
                // machinery — their output shape comes from
                // `Program::reshape_targets`. Validate the input is
                // consumable (to catch malformed synthesized programs)
                // but ignore the computed input shape for typing.
                if let Expr::Call {
                    op: OpKind::Reshape,
                    args,
                } = value
                {
                    for a in args {
                        self.expr_shape(a)?;
                    }
                    let shape = self.reshape_targets.get(target).cloned().ok_or_else(|| {
                        ShapeError::BadArgs {
                            op: OpKind::Reshape,
                            reason: format!(
                                "synthesized Reshape target {target:?} missing from \
                                 Program::reshape_targets"
                            ),
                        }
                    })?;
                    self.locals.insert(*target, shape);
                    return Ok(());
                }
                let shape = self.infer_expr(value)?;
                self.locals.insert(*target, shape);
                Ok(())
            }
            Stmt::AssignTuple { targets, value } => {
                // The DSL's only tuple-returning ops are the rope-append
                // family (`rope_append` and `rope_append_interleaved`).
                // Their signatures already constrained q/k/v, and their
                // output is q's shape. We bind each target to the
                // corresponding input's shape (which the signature
                // already unified with the heads-layout).
                if let Expr::Call { op, args } = value
                    && matches!(op, OpKind::RopeAppend | OpKind::RopeAppendInterleaved)
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
                    apply_signature(&mut self.solver, *op, &all_input_shapes)?;
                    // Now bind targets to the post-unification
                    // shapes. (Since rope_append is shape-preserving,
                    // target shapes equal input shapes.)
                    if targets.len() != 3 {
                        return Err(ShapeError::BadArgs {
                            op: *op,
                            reason: format!(
                                "{} returns 3 values, got {} targets",
                                op.as_str(),
                                targets.len()
                            ),
                        });
                    }
                    for (t, s) in targets.iter().zip(shapes.iter()) {
                        self.locals.insert(*t, s.clone());
                    }
                    Ok(())
                } else if let Expr::Call { op, args } = value
                    && matches!(op, OpKind::VisionRope)
                {
                    // `(q, k) = vision_rope(q, k, cos, sin)`
                    // Both outputs are shape-preserving on their
                    // respective inputs. cos/sin are opaque externs.
                    if targets.len() != 2 {
                        return Err(ShapeError::BadArgs {
                            op: *op,
                            reason: format!(
                                "vision_rope returns 2 values, got {} targets",
                                targets.len()
                            ),
                        });
                    }
                    let q_shape = self.expr_shape(&args[0])?;
                    let k_shape = self.expr_shape(&args[1])?;
                    let all_input_shapes: Vec<Shape> = args
                        .iter()
                        .map(|a| self.expr_shape(a))
                        .collect::<Result<_, _>>()?;
                    apply_signature(&mut self.solver, *op, &all_input_shapes)?;
                    self.locals.insert(targets[0], q_shape);
                    self.locals.insert(targets[1], k_shape);
                    Ok(())
                } else if let Expr::Call { op, args } = value
                    && matches!(op, OpKind::MlaSplit)
                {
                    // `(kv_latent, k_pe) = mla_split(kv_a)`
                    // kv_a: [T, kv_lora_rank + qk_rope_head_dim]
                    // kv_latent: [T, kv_lora_rank]
                    // k_pe: [T, qk_rope_head_dim]
                    let kv_a_shape = self.expr_shape(&args[0])?;
                    if targets.len() != 2 {
                        return Err(ShapeError::BadArgs {
                            op: *op,
                            reason: format!(
                                "mla_split returns 2 values, got {} targets",
                                targets.len()
                            ),
                        });
                    }
                    let t_dim = kv_a_shape
                        .first()
                        .cloned()
                        .unwrap_or(Dim::Bound("num_tokens".into()));
                    self.locals.insert(
                        targets[0],
                        vec![t_dim.clone(), Dim::Bound("kv_lora_rank".into())],
                    );
                    self.locals.insert(
                        targets[1],
                        vec![t_dim, Dim::Bound("qk_rope_head_dim".into())],
                    );
                    Ok(())
                } else {
                    Err(ShapeError::BadArgs {
                        op: match value {
                            Expr::Call { op, .. } => *op,
                            _ => OpKind::Add, // placeholder
                        },
                        reason: "only rope_append, rope_append_interleaved, vision_rope, and mla_split return a tuple".into(),
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
                self.infer_stmts(body, program)?;
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
                self.infer_stmts(then_body, program)?;
                self.infer_stmts(else_body, program)?;
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
                // Elementwise multiplication: tensor×tensor unifies
                // shapes; scalar×tensor broadcasts the tensor shape
                // (matches `sig_binary_elementwise`).
                let l = self.expr_shape(lhs)?;
                let r = self.expr_shape(rhs)?;
                match (l.is_empty(), r.is_empty()) {
                    (true, true) => {
                        return Err(ShapeError::BadArgs {
                            op: OpKind::Mul,
                            reason: "mul of two scalars is a compile-time constant".into(),
                        });
                    }
                    (true, false) => return Ok(r),
                    (false, true) => return Ok(l),
                    (false, false) => {}
                }
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
            Expr::SqrtBound(_) => {
                // Same shape treatment as ScalarLit; the CFG builder
                // folds SqrtBound → ScalarLit once ModelParams is
                // available, but shape inference runs before that.
                Ok(Shape::new())
            }
            Expr::ConfigScalar { .. } => Ok(Shape::new()),
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
        let inf = infer(
            &p,
            &crate::weights_manifest::WeightsManifest::llama_test_conventions(),
            &std::collections::BTreeMap::new(),
        )
        .expect("infer");

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
        let inf = infer(
            &p,
            &crate::weights_manifest::WeightsManifest::llama_test_conventions(),
            &std::collections::BTreeMap::new(),
        )
        .expect("infer");

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
        let inf = infer(
            &p,
            &crate::weights_manifest::WeightsManifest::llama_test_conventions(),
            &std::collections::BTreeMap::new(),
        )
        .expect("infer");

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

    // ── Phase G.2 — vision OpKind shape signatures ──────────────
    //
    // These tests exercise `apply_signature` directly: the new
    // OpKinds aren't reachable from `from_name` yet (per the
    // parse-then-reject rule, the DSL surface lands with G.4 Impls
    // and G.5 vision-forward bodies in lockstep). Tests pin the
    // input arity and shape-preserving output.

    #[test]
    fn varlen_attention_arity_and_output_shape() {
        let mut solver = Solver::new();
        // q/k/v all rank-2 [total_L, vision_heads*head_dim]; cu_seqlens
        // and max_seqlen are externs (empty shape).
        let qkv_last = canonical_mul(vec![bound("vision_num_heads"), bound("vision_head_dim")]);
        let q = vec![bound("total_L"), qkv_last.clone()];
        let k = q.clone();
        let v = q.clone();
        let cu_seqlens = vec![];
        let max_seqlen = vec![];
        let sig = apply_signature(
            &mut solver,
            OpKind::VarlenAttention,
            &[q.clone(), k, v, cu_seqlens, max_seqlen],
        )
        .expect("varlen_attention sig");
        assert_eq!(sig.output, q, "output preserves q's shape");

        // Wrong arity — must fail with ArgCount, not silently succeed.
        let mut solver2 = Solver::new();
        let err = apply_signature(
            &mut solver2,
            OpKind::VarlenAttention,
            std::slice::from_ref(&q),
        );
        assert!(matches!(err, Err(ShapeError::ArgCount { .. })));
    }

    #[test]
    fn vision_rope_arity_and_output_shape() {
        let mut solver = Solver::new();
        let q_last = canonical_mul(vec![bound("vision_num_heads"), bound("vision_head_dim")]);
        let q = vec![bound("total_L"), q_last.clone()];
        let k = q.clone();
        let cos = vec![];
        let sin = vec![];
        let sig = apply_signature(&mut solver, OpKind::VisionRope, &[q.clone(), k, cos, sin])
            .expect("vision_rope sig");
        // apply_signature returns q's shape as the primary output;
        // AssignTuple binds the second target separately.
        assert_eq!(sig.output, q);

        let mut solver2 = Solver::new();
        let err = apply_signature(&mut solver2, OpKind::VisionRope, &[q]);
        assert!(matches!(err, Err(ShapeError::ArgCount { .. })));
    }

    #[test]
    fn quick_gelu_is_shape_preserving_unary() {
        let mut solver = Solver::new();
        let x = vec![bound("total_L"), bound("vision_intermediate_size")];
        let sig = apply_signature(&mut solver, OpKind::QuickGelu, std::slice::from_ref(&x))
            .expect("quick_gelu sig");
        assert_eq!(sig.output, x);
    }

    #[test]
    fn gelu_erf_is_shape_preserving_unary() {
        let mut solver = Solver::new();
        let x = vec![bound("total_L"), bound("vision_intermediate_size")];
        let sig = apply_signature(&mut solver, OpKind::GeluErf, std::slice::from_ref(&x))
            .expect("gelu_erf sig");
        assert_eq!(sig.output, x);
    }

    #[test]
    fn vision_op_names_round_trip_through_as_str() {
        // OpKind::as_str must be total over every variant; these
        // are the new names landed in G.2.
        assert_eq!(OpKind::VarlenAttention.as_str(), "varlen_attention");
        assert_eq!(OpKind::VisionRope.as_str(), "vision_rope");
        assert_eq!(OpKind::QuickGelu.as_str(), "quick_gelu");
        assert_eq!(OpKind::GeluErf.as_str(), "gelu_erf");
    }

    #[test]
    fn quick_gelu_and_gelu_erf_are_distinct_from_tanh_gelu() {
        // Numerics-bearing distinction: the three GELU OpKinds must
        // be PartialEq-distinct so consumer Impls can match exactly
        // one of them. (Eq is derived; this test is a structural
        // promise that future refactors can't accidentally collapse
        // them onto a single variant.)
        assert_ne!(OpKind::Gelu, OpKind::QuickGelu);
        assert_ne!(OpKind::Gelu, OpKind::GeluErf);
        assert_ne!(OpKind::QuickGelu, OpKind::GeluErf);
    }

    #[test]
    fn vision_op_names_resolve_via_from_name() {
        // G.4 gate: every name `OpKind::as_str` returns must round-
        // trip through `from_name`, otherwise classify rejects DSL
        // bodies that wrote the op (parse-then-reject violation).
        // Locks in the four arms added in G.4 lockstep with the
        // existing matchers in `impl_lib::starter_library`.
        assert_eq!(
            OpKind::from_name("varlen_attention"),
            Some(OpKind::VarlenAttention)
        );
        assert_eq!(OpKind::from_name("vision_rope"), Some(OpKind::VisionRope));
        assert_eq!(OpKind::from_name("quick_gelu"), Some(OpKind::QuickGelu));
        assert_eq!(OpKind::from_name("gelu_erf"), Some(OpKind::GeluErf));
    }

    #[test]
    fn dsl_authored_reshape_threads_through_shape_infer() {
        // End-to-end gate for G.5.c: a DSL body that writes
        // `out = reshape(x, [a, b])` must classify and shape-infer
        // without recovery, with the output local picking up the
        // declared dims from `Program::reshape_targets` exactly the
        // way a synthesized reshape would.
        //
        // The body is intentionally minimal — just enough to prove
        // the wiring chain:
        //   parse → classify → shape::infer → Inferred.locals[out]
        // is the dims we wrote in the DSL.
        let p = {
            let file: syn::File = syn::parse_str(
                "fn _carrier() { y = reshape(input_layernorm, [num_tokens, hidden_size]); }",
            )
            .expect("syn parse");
            let block = match &file.items[0] {
                syn::Item::Fn(f) => &*f.block,
                _ => unreachable!(),
            };
            let ast = crate::parse::parse_block(block).expect("DSL parse");
            crate::classify::classify(&ast).expect("classify")
        };

        // Sanity: classify recorded the target shape.
        let target = match &p.statements[0] {
            Stmt::Assign { target, .. } => *target,
            _ => panic!(),
        };
        let dims = p
            .reshape_targets
            .get(&target)
            .expect("reshape_targets entry");
        assert_eq!(dims, &vec![bound("num_tokens"), bound("hidden_size")]);

        // shape::infer must pick those dims up directly (no
        // ReshapeRecovery, no Mismatch). The empty manifest path is
        // the simplest cell since this body never references a
        // weight that needs anchoring.
        let manifest = crate::weights_manifest::WeightsManifest::empty();
        let bounds = std::collections::BTreeMap::from([
            ("num_tokens".to_string(), 256_u64),
            ("hidden_size".to_string(), 1280_u64),
        ]);
        let inferred = infer(&p, &manifest, &bounds).expect("shape::infer must succeed");

        // The reshape's output local carries the exact dims we wrote.
        let out_shape = inferred
            .locals
            .get(&target)
            .expect("inferred shape for `y`");
        assert_eq!(
            out_shape,
            &vec![bound("num_tokens"), bound("hidden_size")],
            "DSL-authored reshape output must equal the declared target shape",
        );
    }
}
