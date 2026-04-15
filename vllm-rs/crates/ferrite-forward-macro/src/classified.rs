// SPDX-License-Identifier: Apache-2.0
//! Classified AST: every free variable reference is tagged as
//! [`ExternKind`], [`WeightId`], or [`LocalId`]. String identifiers
//! live only in the side tables ([`LocalTable`], [`WeightTable`]);
//! the program itself refers to everything by numeric ID.
//!
//! This is the last stage where the caller can still recover the
//! user's original identifiers (via the tables). Passes below here
//! work with IDs only.

#![allow(dead_code)]

use syn::Ident;

/// A local binding. Each assignment statement introduces a fresh
/// `LocalId` even if the target name was previously bound; reads
/// at a site resolve to the *most recent* LocalId for that name at
/// that site (straight-line SSA).
///
/// For-loop induction variables are LocalIds whose scope is the
/// loop body.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct LocalId(pub u32);

/// A weight reference, identified by its dotted path. Two DSL
/// reads of `self_attn.q_proj[layer]` resolve to the same `WeightId`
/// (indexing is stored on the expression, not on the id).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct WeightId(pub u32);

/// The fixed enum of non-weight parameters. Every model uses the
/// same names and shapes for these; the `#[forward]` macro knows
/// about them by name.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum ExternKind {
    InputIds,
    Positions,
    Rotary,
    BlockTable,
    KvCache,
}

impl ExternKind {
    /// Map a DSL identifier to its `ExternKind`, if any.
    pub fn from_name(name: &str) -> Option<Self> {
        match name {
            "input_ids" => Some(Self::InputIds),
            "positions" => Some(Self::Positions),
            "rotary" => Some(Self::Rotary),
            "block_table" => Some(Self::BlockTable),
            "kv_cache" => Some(Self::KvCache),
            _ => None,
        }
    }
}

/// The fixed enum of DSL op kinds. One variant per op, no
/// tile-kind sub-variants. Extending the DSL with a new op means
/// adding one variant here, one shape signature in Phase 4, and
/// one kernel implementation — nothing else.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum OpKind {
    Embed,
    RmsNorm,
    Gemm,
    RopeAppend,
    Attention,
    /// Same signature as `Attention`; picked by the DSL body at
    /// sliding-window attention layers. The distinction is carried
    /// through the FUF so the solver can match distinct Impls
    /// (dense flash-attn vs. window-masked flash-attn).
    SlidingAttention,
    Silu,
    /// Gaussian-Error Linear Unit. Unary elementwise. Shape-preserving
    /// like `Silu`; paired with `Mul` in the gate/up fusion of any
    /// architecture whose MLP is `down(gelu(gate) * up)`.
    Gelu,
    /// Tanh-based soft-cap: `y = cap * tanh(x / cap)`. Unary
    /// elementwise with an additional scalar argument; shape-
    /// preserving. Used at logit exit for architectures that cap
    /// large pre-softmax magnitudes.
    TanhSoftCap,
    Add,
    /// Elementwise multiplication. Produced by the DSL's `*`
    /// operator (e.g. `gate * up` in the SwiGLU MLP). Not reachable
    /// from `from_name` because `*` is a binary operator at the
    /// parse level rather than a named call.
    Mul,
}

impl OpKind {
    /// Map a DSL op-call ident to its `OpKind`. Binary operators
    /// (currently just `*`) do not flow through this path.
    pub fn from_name(name: &str) -> Option<Self> {
        match name {
            "embed" => Some(Self::Embed),
            "rmsnorm" => Some(Self::RmsNorm),
            "gemm" => Some(Self::Gemm),
            "rope_append" => Some(Self::RopeAppend),
            "attention" => Some(Self::Attention),
            "sliding_attention" => Some(Self::SlidingAttention),
            "silu" => Some(Self::Silu),
            "gelu" => Some(Self::Gelu),
            "tanh_softcap" => Some(Self::TanhSoftCap),
            "add" => Some(Self::Add),
            _ => None,
        }
    }

    pub fn as_str(self) -> &'static str {
        match self {
            Self::Embed => "embed",
            Self::RmsNorm => "rmsnorm",
            Self::Gemm => "gemm",
            Self::RopeAppend => "rope_append",
            Self::Attention => "attention",
            Self::SlidingAttention => "sliding_attention",
            Self::Silu => "silu",
            Self::Gelu => "gelu",
            Self::TanhSoftCap => "tanh_softcap",
            Self::Add => "add",
            Self::Mul => "mul",
        }
    }
}

/// A classified DSL program.
#[derive(Clone, Debug)]
pub struct Program {
    pub statements: Vec<Stmt>,
    /// Ident for each LocalId (for diagnostics and codegen only).
    pub locals: LocalTable,
    /// Path segments for each WeightId (for diagnostics and
    /// runtime weight lookup).
    pub weights: WeightTable,
}

/// Side table: `LocalId` → debug ident.
#[derive(Clone, Debug, Default)]
pub struct LocalTable {
    entries: Vec<Ident>,
}

impl LocalTable {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn push(&mut self, name: Ident) -> LocalId {
        let id = LocalId(self.entries.len() as u32);
        self.entries.push(name);
        id
    }

    pub fn name(&self, id: LocalId) -> &Ident {
        &self.entries[id.0 as usize]
    }

    pub fn len(&self) -> usize {
        self.entries.len()
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }
}

/// Side table: `WeightId` → dotted path segments.
#[derive(Clone, Debug, Default)]
pub struct WeightTable {
    /// Invariant: paths are unique (interning).
    entries: Vec<Vec<Ident>>,
}

impl WeightTable {
    pub fn new() -> Self {
        Self::default()
    }

    /// Insert a path, returning the assigned `WeightId`. Idempotent:
    /// two calls with paths of equal string segments return the same id.
    pub fn intern(&mut self, path: Vec<Ident>) -> WeightId {
        for (i, existing) in self.entries.iter().enumerate() {
            if idents_eq(existing, &path) {
                return WeightId(i as u32);
            }
        }
        let id = WeightId(self.entries.len() as u32);
        self.entries.push(path);
        id
    }

    pub fn path(&self, id: WeightId) -> &[Ident] {
        &self.entries[id.0 as usize]
    }

    pub fn len(&self) -> usize {
        self.entries.len()
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Test helper: find a weight id by its path segments as strings.
    #[cfg(test)]
    pub fn path_for_test(&self, segments: &[&str]) -> Option<WeightId> {
        self.entries.iter().enumerate().find_map(|(i, p)| {
            if p.len() == segments.len() && p.iter().zip(segments).all(|(id, s)| id == s) {
                Some(WeightId(i as u32))
            } else {
                None
            }
        })
    }
}

fn idents_eq(a: &[Ident], b: &[Ident]) -> bool {
    a.len() == b.len() && a.iter().zip(b).all(|(l, r)| l == r)
}

/// A statement in the classified program.
#[derive(Clone, Debug)]
pub enum Stmt {
    /// `target = value` where `target` is a fresh LocalId.
    Assign { target: LocalId, value: Expr },
    /// `(t0, t1, ...) = value` where each `t_i` is a fresh LocalId.
    AssignTuple { targets: Vec<LocalId>, value: Expr },
    /// `for ivar in 0..<bound> { body }`. `ivar` is a fresh LocalId
    /// scoped to the body.
    ///
    /// `loop_carry` enumerates the names that are bound both
    /// *before* the loop and *inside* the body. Each entry is
    /// `(outer, inner)` where `outer` is the LocalId of the outer
    /// binding that body reads see initially, and `inner` is the
    /// LocalId of the body's *last* write to that name. After each
    /// iteration, the unroller re-binds `outer`'s tile to `inner`'s
    /// tile so the next iteration's reads see the iteration's
    /// previous output.
    For {
        ivar: LocalId,
        start: Bound,
        end: Bound,
        body: Vec<Stmt>,
        loop_carry: Vec<(LocalId, LocalId)>,
    },
    /// `if <predicate> { then_body } else { else_body }`. The
    /// predicate is a compile-time-evaluable function of a loop
    /// induction variable and config constants — evaluated at
    /// unroll time, each unrolled iteration descends into exactly
    /// one arm.
    ///
    /// Both arms must bind the same set of names. For each name
    /// bound in either arm, `merge_carry` has one entry
    /// `(merge_id, then_final, else_final)`: reads after the If
    /// resolve to `merge_id`; at unroll time the unroller sets
    /// `local_to_tile[merge_id]` to whichever arm ran. An arm that
    /// doesn't bind the name reuses its pre-If binding's LocalId
    /// as the arm's "final" id.
    If {
        cond: BoolPred,
        then_body: Vec<Stmt>,
        else_body: Vec<Stmt>,
        merge_carry: Vec<(LocalId, LocalId, LocalId)>,
    },
}

/// Loop bound: either a literal integer or a symbolic identifier
/// that names a per-model bound (e.g. `num_hidden_layers`). The
/// Ident is preserved here because bound resolution happens later,
/// in Phase 3 when config.json values are loaded.
#[derive(Clone, Debug)]
pub enum Bound {
    Lit(u64),
    Sym(Ident),
}

/// Boolean predicate used as an `if` condition. The predicate
/// enum is deliberately closed and narrow — it exists to express
/// layer-indexed dispatch patterns (Gemma2 alternating
/// sliding/full attention, DeepSeek-V3 "first N layers dense")
/// without extending the expression IR with booleans or general
/// binary arithmetic. Evaluated only at unroll time against
/// concrete loop-var values.
#[derive(Clone, Debug)]
pub enum BoolPred {
    /// `ivar % divisor == remainder`.
    Modulo {
        ivar: LocalId,
        divisor: Bound,
        remainder: Bound,
    },
    /// `ivar < bound`.
    Less { ivar: LocalId, bound: Bound },
}

/// A value-producing expression.
#[derive(Clone, Debug)]
pub enum Expr {
    /// Read of a local binding.
    Local(LocalId),
    /// Read of a non-weight parameter, optionally indexed by a
    /// local (the loop variable).
    Extern {
        kind: ExternKind,
        index: Option<LocalId>,
    },
    /// Read of a weight, optionally indexed by a local.
    Weight {
        id: WeightId,
        index: Option<LocalId>,
    },
    /// Op call.
    Call { op: OpKind, args: Vec<Expr> },
    /// Multiplication (`gate * up`). Tensor × tensor.
    Mul { lhs: Box<Expr>, rhs: Box<Expr> },
    /// Addition (`w + 1.0`) — the classifier resolves this to a
    /// tile-level `OpKind::Add` call with the scalar captured as
    /// `ScalarLit` inside `args`. See [`classify_expr`].
    ///
    /// A purely structural variant; classify reshapes it before
    /// downstream passes see it, so nothing below the parser needs
    /// a dedicated `Add` binop variant.
    Add { lhs: Box<Expr>, rhs: Box<Expr> },
    /// A numeric scalar literal. Used as an operand to elementwise
    /// ops that admit a scalar broadcast (e.g. Gemma's `w + 1.0`).
    ScalarLit(f64),
    /// `sqrt(<bound_name>)` — unresolved compile-time scalar. The
    /// CFG builder resolves this to `ScalarLit(bounds[name].sqrt())`
    /// per-model (so the value becomes concrete before the FUF).
    SqrtBound(Ident),
}
