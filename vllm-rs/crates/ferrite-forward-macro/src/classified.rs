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
    /// Alternate rotary cache for architectures with dual RoPE bases
    /// (e.g. Gemma3's `rope_local_base_freq` for sliding-attention
    /// layers). Lives on the Weights struct, not ForwardCtx.
    RotaryLocal,
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
            "rotary_local" => Some(Self::RotaryLocal),
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
    /// Full LayerNorm with mean subtraction (and weight; bias-free in
    /// the variants seen so far): `y = w * (x - mean(x)) / sqrt(var(x) + eps)`.
    /// Distinct math from `RmsNorm` (which omits the mean subtraction).
    /// Used by Cohere's CommandR family.
    LayerNorm,
    Gemm,
    RopeAppend,
    /// RoPE that pairs adjacent elements `(2i, 2i+1)` for rotation
    /// instead of NeoX's `(i, i + half_dim)`. Different element
    /// pairing → genuinely different math, hence its own variant
    /// rather than a flag on `RopeAppend`. Used by Cohere's CommandR
    /// family.
    RopeAppendInterleaved,
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
    /// Tensor-parallel all-reduce-sum across `tp_world_size` ranks,
    /// in place. Identity-shape: `(x: [...]) -> [...]`. Never appears
    /// in any per-arch DSL — produced exclusively by the lowering
    /// pass that inserts one of these after every `Gemm` whose weight
    /// has shard-kind `ShardDim1` (row-parallel: `o_proj`,
    /// `down_proj`). At `tp_world_size = 1` the pass is a no-op so no
    /// FUF carries this op kind. Lowers to `Instruction::AllReduce`
    /// (gated under the `nccl` feature on `ferrite-forward`).
    AllReduce,
    /// Tensor-parallel all-gather along the LAST dim of one input
    /// tile across `tp_world_size` ranks. Output shape is the input
    /// shape with the last dim multiplied by `tp_world_size`. Like
    /// `AllReduce`, never appears in any DSL — produced exclusively
    /// by the lowering pass at tp>1. Inserted after the `Gemm` whose
    /// weight is `lm_head` (vocab-parallel `ShardDim0`): the per-rank
    /// matmul produces partial logits `[N, vocab/tp]` and the
    /// AllGather reassembles `[N, vocab]` for the sampler. Mirrors
    /// Python vLLM's `tensor_model_parallel_all_gather` on the
    /// `LogitsProcessor` path. Lowers to `Instruction::AllGather`.
    AllGather,
    /// Broadcast-add of a learned per-feature bias vector across the
    /// batch/token dimensions: `bias_add(x: [..., D], b: [D]) -> [..., D]`.
    /// Semantically distinct from `Add` — `Add` is same-shape
    /// elementwise (residual stream), `BiasAdd` is a vector
    /// broadcast-add (affine-transform completion). Kept as its own
    /// op so fusion patterns that fold bias into a GEMM epilog
    /// (`(Gemm, BiasAdd)` → cuBLAS `gemm_bias`) can match distinctly
    /// from patterns that consume residual `Add`.
    BiasAdd,
    /// Elementwise multiplication. Produced by the DSL's `*`
    /// operator (e.g. `gate * up` in the SwiGLU MLP). Not reachable
    /// from `from_name` because `*` is a binary operator at the
    /// parse level rather than a named call.
    Mul,
    /// Shape view — reinterprets a tensor under a different rank
    /// without moving or copying data. Element count is preserved;
    /// the codegen maps to `OwnedTensor::reshape` / `TensorView::reshape`
    /// (metadata-only). Today Reshape tiles are **synthesized by
    /// shape inference** when an op expects a factor of the
    /// producer's last dim (e.g. per-head rmsnorm expects `[D]` but
    /// upstream gemm produces `[..., heads * D]`). The target shape
    /// is stored in [`Program::reshape_targets`] keyed by the new
    /// local's id; inference re-reads it when typechecking the
    /// synthesized statement. Not yet exposed to DSL authors via
    /// `from_name` — add an entry there if a future pattern needs
    /// explicit user-written reshape.
    Reshape,
    /// MLA kv_a split: decomposes `[T, kv_lora_rank + qk_rope_head_dim]`
    /// into `(kv_latent: [T, kv_lora_rank], k_pe: [T, qk_rope_head_dim])`.
    /// DSL form: `(kv_latent, k_pe) = mla_split(kv_a)`. Tuple-returning;
    /// the 2-target binding is handled in `Stmt::AssignTuple`. Used
    /// exclusively by DeepSeek V2/V3.
    MlaSplit,
    /// MLA full attention: applies interleaved RoPE to q_pe / k_pe, writes
    /// compressed KV to paged cache, assembles full K and V from cached
    /// kv_b + k_pe, runs flash attention, then slices the output to
    /// `v_head_dim`. Consumes `q, kv_b, k_pe` plus externs
    /// `(positions, rotary, kv_cache[layer], block_table)`. Output:
    /// `[T, num_attention_heads * v_head_dim]`. Used by DeepSeek V2/V3.
    MlaAttention,
    /// MoE block — gate routing + top-K fused GEMM for routed experts
    /// plus an optional shared expert. The DSL form is the same across
    /// every MoE arch: `moe_out = moe_block(x, moe[layer])`. Mirrors
    /// HF Python naming (`MixtralSparseMoeBlock`,
    /// `Qwen2MoeSparseMoeBlock`, `DeepseekV2MoE`). The forward math
    /// (no shared / sigmoid-gated shared / scaled-plain-add shared)
    /// is encoded in the loaded layer struct (`FusedMoELayer` /
    /// `SharedFusedMoELayer` / `DeepSeekV2MoELayer` + their FP8/Ggml
    /// flavors), and each is claimed by a distinct `Implementation`
    /// keyed on the rust_type fingerprint of the `moe[layer]` weight
    /// accessor. Shape-preserving.
    Moe,
}

impl OpKind {
    /// Map a DSL op-call ident to its `OpKind`. Binary operators
    /// (currently just `*`) do not flow through this path.
    pub fn from_name(name: &str) -> Option<Self> {
        match name {
            "embed" => Some(Self::Embed),
            "rmsnorm" => Some(Self::RmsNorm),
            "layer_norm" => Some(Self::LayerNorm),
            "gemm" => Some(Self::Gemm),
            "rope_append" => Some(Self::RopeAppend),
            "rope_append_interleaved" => Some(Self::RopeAppendInterleaved),
            "attention" => Some(Self::Attention),
            "sliding_attention" => Some(Self::SlidingAttention),
            "silu" => Some(Self::Silu),
            "gelu" => Some(Self::Gelu),
            "tanh_softcap" => Some(Self::TanhSoftCap),
            "add" => Some(Self::Add),
            "bias_add" => Some(Self::BiasAdd),
            // `Reshape` is synthesized by shape inference, not DSL-
            // writable today. Intentionally not listed in `from_name`;
            // add the arm if a future pattern needs explicit reshape.
            "mla_split" => Some(Self::MlaSplit),
            "mla_attention" => Some(Self::MlaAttention),
            "moe_block" => Some(Self::Moe),
            _ => None,
        }
    }

    pub fn as_str(self) -> &'static str {
        match self {
            Self::Embed => "embed",
            Self::RmsNorm => "rmsnorm",
            Self::LayerNorm => "layer_norm",
            Self::Gemm => "gemm",
            Self::RopeAppend => "rope_append",
            Self::RopeAppendInterleaved => "rope_append_interleaved",
            Self::Attention => "attention",
            Self::SlidingAttention => "sliding_attention",
            Self::Silu => "silu",
            Self::Gelu => "gelu",
            Self::TanhSoftCap => "tanh_softcap",
            Self::Add => "add",
            Self::BiasAdd => "bias_add",
            Self::Mul => "mul",
            Self::Reshape => "reshape",
            Self::MlaSplit => "mla_split",
            Self::MlaAttention => "mla_attention",
            Self::Moe => "moe_block",
            // No DSL surface — produced only by the post-FUF lowering
            // pass at tp>1. `from_name` deliberately omits it so a
            // user can't write `all_reduce(...)` in a `#[forward]`
            // body; the canonical path is the lowering pass.
            Self::AllReduce => "all_reduce",
            // Same DSL-omission story as AllReduce — only the
            // lowering pass produces this op kind.
            Self::AllGather => "all_gather",
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
    /// Target shapes for synthesized `Reshape` statements. Keyed by
    /// the LocalId of the reshape's output (the newly-introduced
    /// fresh local). Empty for DSL programs with no shape-mismatch
    /// recoveries.
    ///
    /// Populated by `shape::infer` when a per-axis-factor mismatch
    /// between a weight's declared shape and its consumer's inferred
    /// shape is detected (e.g. Qwen3's per-head `q_norm` of shape
    /// `[head_dim]` applied to a gemm output of shape `[..., heads *
    /// head_dim]`). The synthesizer inserts a `Reshape` Stmt and
    /// records the target shape here; the second inference pass
    /// reads this map to typecheck the synthesized statement.
    pub reshape_targets: std::collections::HashMap<LocalId, Vec<crate::shape::Dim>>,
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

/// Side table: `WeightId` → dotted path segments (plain strings).
///
/// Path segments are stringified at intern time — `proc_macro2::Ident`
/// wraps rustc's thread-local symbol bridge, so reading an Ident off
/// the main thread (e.g. via `Ident::to_string`) panics. Storing
/// `Vec<String>` keeps the classified program Send-safe for the
/// macro's per-model rayon loop.
#[derive(Clone, Debug, Default)]
pub struct WeightTable {
    /// Invariant: paths are unique (interning).
    entries: Vec<Vec<String>>,
}

impl WeightTable {
    pub fn new() -> Self {
        Self::default()
    }

    /// Insert a path, returning the assigned `WeightId`. Idempotent:
    /// two calls with paths of equal string segments return the same id.
    /// Accepts `Vec<Ident>` and stringifies — call this from the main
    /// thread during classify, while the proc-macro bridge is live.
    pub fn intern(&mut self, path: Vec<Ident>) -> WeightId {
        let path: Vec<String> = path.iter().map(|i| i.to_string()).collect();
        self.intern_str(path)
    }

    pub fn intern_str(&mut self, path: Vec<String>) -> WeightId {
        for (i, existing) in self.entries.iter().enumerate() {
            if existing == &path {
                return WeightId(i as u32);
            }
        }
        let id = WeightId(self.entries.len() as u32);
        self.entries.push(path);
        id
    }

    pub fn path(&self, id: WeightId) -> &[String] {
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
            if p.len() == segments.len() && p.iter().zip(segments).all(|(s, t)| s == t) {
                Some(WeightId(i as u32))
            } else {
                None
            }
        })
    }
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
    /// `ivar % divisor != remainder`.
    NotModulo {
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
    /// `scalar(<name>)` / `recip_scalar(<name>)` — unresolved
    /// compile-time scalar read from `ModelParams.scalars`. CFG
    /// builder folds to `ScalarLit(scalars[name])` (or its
    /// reciprocal) per-model.
    ConfigScalar { name: Ident, recip: bool },
}
