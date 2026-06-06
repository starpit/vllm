// SPDX-License-Identifier: Apache-2.0
//! The canonical wavefront **SubtileIR** — region-granular SSA over
//! tensors, the fold of `region.rs` (v2) + `subtile.rs` (v1) into one
//! module.
//!
//! ## What this is
//!
//! Every value lives in a **tensor** (a logical row-major buffer): leaf
//! `sources` (weights / activations / prefix-KV / embed / cos-sin) and one
//! **op-output tensor** per op. A [`SubtileNode`] *writes a region* of
//! one output tensor and *reads regions* of input tensors. Dependencies
//! are derived from **region overlap**: a node that reads region `R` of
//! an op-output tensor `T` depends on every earlier node whose
//! write-region on `T` overlaps `R`. Reads of a leaf source have no
//! dependency (sources are bound at eval time).
//!
//! That is the SSA model that supports **true subtile granularity** on
//! the GPU: `q_proj` split into N-blocks where each block writes a slice
//! of Q, then rope → attention reads the *whole* Q assembled from those
//! slices. The v1 whole-output `Operand::Sub` model couldn't express
//! that — and is gone (it lives only in [`legacy`] for the few v1-only
//! carcasses that have not been deleted yet: [`crate::tape`],
//! [`crate::scheduler`], [`crate::lower::lower`] which are slated for
//! deletion in later staged commits).
//!
//! ## Two-tier validation
//!
//! - **Tier A′ (decomposition equivalence):** [`eval_dag`] equals
//!   `cpu_golden` whole-op output. Bit-exact at `nb >= n` and at every
//!   N-block width that doesn't reorder a per-output reduction; only
//!   token-exact once a reduction is reassociated (split-K).
//! - **Tier A (self-consistency):** scheduled tape replay equals
//!   [`eval_dag`] (lands with the wavefront scheduler).
//!
//! `cpu_golden` (in `ferrite-forward`) is the deterministic f32 substrate
//! the player computes with — not the correctness *oracle* (that is
//! ferrite-metal non-mega at temp=0).

#![allow(dead_code)]

use std::marker::PhantomData;

// ── Identifiers & geometry ─────────────────────────────────────────

/// Dense index into a [`SubtileIR::nodes`] vector. The DAG is
/// topologically ordered: every overlap-predecessor of a node has a
/// smaller id (so a single pass over `nodes` is a valid evaluation
/// order).
/// Sealed per §2: the inner field is `pub(crate)`, so external code
/// cannot construct or read this id outside the crate. Internal
/// construction stays the cheap `SubtileId(N)` tuple form.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct SubtileId(pub(crate) u32);

/// Half-open range `[start, start + len)` along one axis.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Range {
    pub start: u32,
    pub len: u32,
}

impl Range {
    pub fn new(start: u32, len: u32) -> Self {
        Self { start, len }
    }
    pub fn end(&self) -> u32 {
        self.start + self.len
    }
}

/// A rectangular slice of a logically row-major `[rows, cols]` buffer.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Region {
    pub rows: Range,
    pub cols: Range,
}

/// Logical shape of a leaf source buffer (used by the v1
/// `LoweringInput`-side bridge in [`crate::lower`]).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SourceShape {
    pub rows: u32,
    pub cols: u32,
}

/// Elementwise op kind. Numerics mirror `cpu_golden` exactly.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum EwKind {
    /// `x / (1 + e^-x)`.
    Silu,
    /// `a * b`.
    Mul,
    /// `a + b`.
    Add,
}

// ── Tensors & regions ──────────────────────────────────────────────

/// Dense index into [`SubtileIR::tensors`]. Tensors `[0, num_sources)`
/// are leaf sources bound at eval time; the rest are op outputs written
/// by subtile nodes.
/// Sealed per §2: inner field is `pub(crate)`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct TensorId(pub(crate) u32);

/// Logical shape of a tensor (row-major `[rows, cols]`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TensorShape {
    pub rows: u32,
    pub cols: u32,
}

impl TensorShape {
    /// The region covering the whole tensor.
    pub fn whole(&self) -> Region {
        Region {
            rows: Range::new(0, self.rows),
            cols: Range::new(0, self.cols),
        }
    }
}

/// A rectangular slice of a tensor — used for both a node's input reads
/// and its single output write.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TensorRegion {
    pub tensor: TensorId,
    pub region: Region,
}

// ── Typed witnesses on SubtileIR DAG nodes ─────────────────────────
//
// Each witness encodes one IR-level invariant. Producer / consumer
// ops carry the SAME witness *value* by construction (single source of
// truth), or — in the case of [`RopeForm`] — share the SAME
// `F: RopeForm` const-generic *type* on the entire IR (so mixing
// NeoX/Interleaved in one forward is a compile error, not a runtime
// surprise).

#[doc(hidden)]
pub mod sealed {
    /// Sealing token — inner `()` is `pub(super)` so external code
    /// cannot construct a `Seal` value. Carried by every type that
    /// must be constructable only inside `subtile_ir`.
    #[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
    pub struct Seal(pub(super) ());
}

/// **`RopeForm`** — sealed marker trait selecting the rotary pairing
/// form. Llama-3.2 uses [`NeoX`]; some other architectures use
/// [`Interleaved`]. Encoded as a const-generic phantom on
/// [`SubtileIR<F>`] / [`SubtileNode<F>`] / [`SubOp<F>`] so a single
/// forward's rope nodes ALL share the same form by construction —
/// mixing NeoX and Interleaved in one IR is a compile error.
///
/// ```compile_fail
/// // Mixing NeoX and Interleaved in one IR is rejected at the type
/// // level: SubtileIR<NeoX>::nodes is Vec<SubtileNode<NeoX>>, so a
/// // SubtileNode<Interleaved> won't fit in it. No need for a runtime
/// // check; the const generic enforces it.
/// use ferrite_wavefront::subtile_ir::{
///     Interleaved, NeoX, SubOp, SubtileId, SubtileIR, SubtileNode, TensorId, TensorRegion,
///     Region, Range,
/// };
/// use std::marker::PhantomData;
/// let interleaved_node = SubtileNode::<Interleaved> {
///     id: SubtileId(0),
///     op: SubOp::<Interleaved>::RopeRotate {
///         head_dim: 4,
///         _form: PhantomData,
///     },
///     inputs: vec![],
///     output: TensorRegion {
///         tensor: TensorId(0),
///         region: Region { rows: Range::new(0, 1), cols: Range::new(0, 4) },
///     },
/// };
/// let _ir: SubtileIR<NeoX> = SubtileIR {
///     tensors: vec![],
///     num_sources: 0,
///     nodes: vec![interleaved_node], // type mismatch
///     result: TensorId(0),
/// };
/// ```
pub trait RopeForm:
    rope_form_seal::Sealed + Copy + std::fmt::Debug + PartialEq + Eq + std::hash::Hash + 'static
{
    /// Erased tag for runtime introspection (printing, unit tests).
    const TAG: RopeFormTag;
}

/// NeoX rope: pairs `(d, d + half)` per head. Llama-3.2 invariant.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum NeoX {}

/// Interleaved rope: pairs `(2k, 2k + 1)` per head.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Interleaved {}

#[doc(hidden)]
pub mod rope_form_seal {
    pub trait Sealed {}
    impl Sealed for super::NeoX {}
    impl Sealed for super::Interleaved {}
}

impl RopeForm for NeoX {
    const TAG: RopeFormTag = RopeFormTag::NeoX;
}
impl RopeForm for Interleaved {
    const TAG: RopeFormTag = RopeFormTag::Interleaved;
}

/// Erased rope-form tag for runtime use.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum RopeFormTag {
    NeoX,
    Interleaved,
}

// ── KvCacheShape sealed trait ────────────────────────────────────────
//
// Per plan §5 K7 + memory/feedback_end_to_end_compile_time_proofs (both
// INVIOLABLE): KvCacheLayout's discriminating dimensions
// (`num_kv_heads`, `head_dim`) MUST propagate as Rust const generics
// with `where`-clauses end-to-end so a producer/consumer layout drift
// (e.g. RopeAppend writes `num_kv_heads=8` while AttnDecode reads
// `num_kv_heads=4` on the same `cache_tensor`) is a `mismatched types`
// rustc error, not a runtime divergence.
//
// `KvCacheShape` is a sealed marker trait whose impls carry the const
// numeric values. `KvCacheLayout<K>` is generic over a shape; SubOp /
// SubtileNode / SubtileIR thread `K: KvCacheShape` so producer-side
// `RopeAppend.layout: KvCacheLayout<K>` and consumer-side
// `AttnDecode.layout: KvCacheLayout<K>` must literally unify.

pub mod kv_shape_seal {
    pub trait Sealed {}
}

/// Sealed marker trait carrying the K/V cache layout's numeric proof
/// (per-token `num_kv_heads * head_dim` row width) at the type level.
///
/// Two `KvCacheLayout<K1>` and `KvCacheLayout<K2>` with `K1 != K2` are
/// distinct Rust types — passing one where the other is expected is a
/// rustc error. This is the K7 compile-time witness.
pub trait KvCacheShape:
    kv_shape_seal::Sealed + Copy + std::fmt::Debug + PartialEq + Eq + std::hash::Hash + 'static
{
    const NUM_KV_HEADS: u32;
    const HEAD_DIM: u32;
    /// Maximum cache position (context length). Lowering computes
    /// per-layer base byte offsets from this constant. Different
    /// context lengths require different `KvCacheShape` impls.
    const MAX_POSITION: u32;
    /// Bytes per cache element. bf16 = 2 (the default; override for
    /// fp16/fp32 caches when those land).
    const ELEM_BYTES: u32 = 2;
    /// Per-token K (or V) row width in elements.
    const ROW_ELEMENTS: u32 = Self::NUM_KV_HEADS * Self::HEAD_DIM;
    /// Bytes per row (one cache slot at one layer): `ROW_ELEMENTS *
    /// ELEM_BYTES`. Used as the per-position stride for KV cache
    /// writes / reads.
    const ROW_BYTES: u64 = (Self::ROW_ELEMENTS as u64) * (Self::ELEM_BYTES as u64);
    /// Bytes per layer: `MAX_POSITION * ROW_BYTES`. Used to compute
    /// layer-base byte offsets at lowering.
    const LAYER_BYTES: u64 = (Self::MAX_POSITION as u64) * Self::ROW_BYTES;
}

/// Llama-3.2-1B's K/V cache shape: 8 KV heads × 64 head_dim,
/// MAX_POSITION=4096 (a representative default; multi-context
/// deployments mint a separate KvCacheShape impl per context length).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum LlamaShape8x64 {}
impl kv_shape_seal::Sealed for LlamaShape8x64 {}
impl KvCacheShape for LlamaShape8x64 {
    const NUM_KV_HEADS: u32 = 8;
    const HEAD_DIM: u32 = 64;
    const MAX_POSITION: u32 = 4096;
}

/// Test-only K/V cache shape: 1 KV head × 4 head_dim, max_pos=16.
/// Used by unit tests that need a small synthetic graph without
/// dragging in a real Llama-sized tensor footprint.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum TestShape1x4 {}
impl kv_shape_seal::Sealed for TestShape1x4 {}
impl KvCacheShape for TestShape1x4 {
    const NUM_KV_HEADS: u32 = 1;
    const HEAD_DIM: u32 = 4;
    const MAX_POSITION: u32 = 16;
}

/// Test-only K/V cache shape: 2 KV heads × 4 head_dim, max_pos=16.
/// Used by the partition / mega tests' `decode_layer` fixture
/// (hkv=2, hd=4).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum TestShape2x4 {}
impl kv_shape_seal::Sealed for TestShape2x4 {}
impl KvCacheShape for TestShape2x4 {
    const NUM_KV_HEADS: u32 = 2;
    const HEAD_DIM: u32 = 4;
    const MAX_POSITION: u32 = 16;
}

/// **`KvCacheLayout`** — sealed witness naming the K-cache (or V-cache)
/// tensor a single forward reads from / writes to. The orchestrator
/// builds ONE per cache tensor; `RopeAppend`'s write and `AttnDecode`'s
/// read both reach for the SAME instance, making layout drift between
/// producer and consumer structurally impossible.
///
/// Constructable only via [`KvCacheLayout::for_cache_tensor`] — sealed.
///
/// ```compile_fail
/// // Sealed: external code cannot construct a KvCacheLayout via the
/// // struct literal because the `_seal: sealed::Seal` and `_shape`
/// // fields are private. The only path is
/// // `KvCacheLayout::<K>::for_cache_tensor(...)`, which makes layout
/// // drift (a divergent producer / consumer fabrication) structurally
/// // impossible — and producer/consumer numeric drift between
/// // different K's is itself a `mismatched types` rustc error.
/// use ferrite_wavefront::subtile_ir::{KvCacheLayout, TensorId, LlamaShape8x64};
/// let _ = KvCacheLayout::<LlamaShape8x64> {
///     cache_tensor: TensorId(0),
///     _shape: std::marker::PhantomData,
/// };
/// ```
/// `KvCacheLayout<K>` — sealed value-typed witness whose numeric
/// dimensions live at the TYPE level via `K: KvCacheShape`. Two
/// `KvCacheLayout<K1>` and `KvCacheLayout<K2>` with `K1 != K2` are
/// distinct Rust types — producer-side `RopeAppend.layout:
/// KvCacheLayout<K>` and consumer-side `AttnDecode.layout:
/// KvCacheLayout<K>` must literally unify (per K7 / plan §5 line 394).
///
/// ```compile_fail
/// // K7 type-level guard: a KvCacheLayout<TestShape1x4> cannot be
/// // passed where a KvCacheLayout<LlamaShape8x64> is expected. This
/// // is the Paris-decode bug surfacing as a Rust type error.
/// use ferrite_wavefront::subtile_ir::{
///     KvCacheLayout, LlamaShape8x64, TensorId, TestShape1x4,
/// };
/// fn want_llama(_: KvCacheLayout<LlamaShape8x64>) {}
/// let drift = KvCacheLayout::<TestShape1x4>::for_cache_tensor(TensorId(0));
/// want_llama(drift); // expected `KvCacheLayout<LlamaShape8x64>`,
///                    // found `KvCacheLayout<TestShape1x4>`
/// ```
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct KvCacheLayout<K: KvCacheShape = LlamaShape8x64> {
    /// K-side cache tensor (rotated K is written here).
    cache_tensor: TensorId,
    /// V-side cache tensor (un-rotated V is written here). Often a
    /// separate tensor; some layouts use the same tensor as K with
    /// different layer-base offsets.
    v_cache_tensor: TensorId,
    _shape: std::marker::PhantomData<K>,
    _seal: sealed::Seal,
}

impl<K: KvCacheShape> KvCacheLayout<K> {
    /// Sealed constructor binding both K and V cache tensors into
    /// the witness; per-axis numerics come from `K`'s associated consts.
    pub const fn for_cache_tensors(
        k_cache_tensor: TensorId,
        v_cache_tensor: TensorId,
    ) -> Self {
        Self {
            cache_tensor: k_cache_tensor,
            v_cache_tensor,
            _shape: std::marker::PhantomData,
            _seal: sealed::Seal(()),
        }
    }

    /// Convenience: K and V on the SAME tensor (legacy / unified
    /// cache). Equivalent to `for_cache_tensors(t, t)`. Production
    /// code should prefer `for_cache_tensors` with distinct K/V
    /// TensorIds.
    pub const fn for_cache_tensor(cache_tensor: TensorId) -> Self {
        Self::for_cache_tensors(cache_tensor, cache_tensor)
    }

    /// K-side cache tensor.
    pub const fn cache_tensor(&self) -> TensorId {
        self.cache_tensor
    }
    /// V-side cache tensor.
    pub const fn v_cache_tensor(&self) -> TensorId {
        self.v_cache_tensor
    }
    #[inline]
    pub const fn num_kv_heads(&self) -> u32 {
        K::NUM_KV_HEADS
    }
    #[inline]
    pub const fn head_dim(&self) -> u32 {
        K::HEAD_DIM
    }
    /// Per-token K (or V) row width in elements.
    #[inline]
    pub const fn row_elements(&self) -> u32 {
        K::ROW_ELEMENTS
    }
    /// Maximum cache position (context length).
    #[inline]
    pub const fn max_position(&self) -> u32 {
        K::MAX_POSITION
    }
    /// Per-position stride in bytes (= ROW_ELEMENTS × ELEM_BYTES).
    /// Used as the const-generic stride for
    /// [`crate::tk_tape::ByteOffsetExpr::RuntimePosition`] writes
    /// to the KV cache.
    #[inline]
    pub const fn row_bytes(&self) -> u64 {
        K::ROW_BYTES
    }
    /// Per-layer byte offset: `layer × MAX_POSITION × ROW_BYTES`.
    /// Used as the `base` of [`crate::tk_tape::ByteOffsetExpr::RuntimePosition`]
    /// when writing layer L's KV cache slot at runtime position p.
    #[inline]
    pub const fn layer_base_bytes(&self, layer: u32) -> u64 {
        (layer as u64) * K::LAYER_BYTES
    }
}

/// **`KvCacheProducer`** — sealed enum naming HOW the K (or V) cache
/// that an [`SubOp::AttnDecode`] reads got populated. The variants are
/// sealed (constructable only via [`KvCacheProducer::from_rope_append`]
/// / [`KvCacheProducer::pre_populated_ext`]) and the enum is
/// `#[non_exhaustive]` so external `match`es must include a wildcard
/// arm — preventing the silent `_ =>` regression on a future variant.
///
/// ```compile_fail
/// // External match without a wildcard is rejected: the enum is
/// // #[non_exhaustive], so the compiler forces a `_ =>` arm. That
/// // makes adding a new variant a soft-fail (existing matchers route
/// // it to the wildcard) rather than a silent miscompile of the kind
/// // a non-exhaustive enum without `non_exhaustive` would suffer.
/// use ferrite_wavefront::subtile_ir::KvCacheProducer;
/// fn name(p: KvCacheProducer) -> &'static str {
///     match p {
///         KvCacheProducer::SameForwardRopeAppend { .. } => "rope_append",
///         KvCacheProducer::PrePopulatedExt { .. } => "ext",
///     }
/// }
/// ```
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum KvCacheProducer {
    /// The cache was written by a `SubOp::RopeAppend` earlier in this
    /// same forward (the producer is `nodes[producer_node_idx]`).
    SameForwardRopeAppend {
        producer_node_idx: u32,
        #[doc(hidden)]
        _seal: sealed::Seal,
    },
    /// The cache is pre-populated by an out-of-band per-op forward and
    /// is read-only inside this megakernel.
    PrePopulatedExt {
        #[doc(hidden)]
        _seal: sealed::Seal,
    },
}

impl KvCacheProducer {
    pub const fn from_rope_append(producer_node_idx: u32) -> Self {
        Self::SameForwardRopeAppend {
            producer_node_idx,
            _seal: sealed::Seal(()),
        }
    }

    pub const fn pre_populated_ext() -> Self {
        Self::PrePopulatedExt {
            _seal: sealed::Seal(()),
        }
    }
}

/// **`SoftmaxStateId`** — opaque identifier for the per-AttnDecode
/// online-softmax accumulator (`m`, `l`, `o`). The lowering walker
/// allocates one per AttnDecode; the TkTape lowering binds it to the
/// concrete register set at emit time. Constructable only inside this
/// crate.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct SoftmaxStateId {
    id: u32,
    _seal: sealed::Seal,
}

impl SoftmaxStateId {
    pub(crate) const fn new(id: u32) -> Self {
        Self {
            id,
            _seal: sealed::Seal(()),
        }
    }
    pub const fn index(&self) -> u32 {
        self.id
    }
}

// ── Sub-operations ─────────────────────────────────────────────────

/// The sub-operation a node performs. Generic over `F: RopeForm`
/// (default [`NeoX`]) so a SubtileIR's rope nodes share the same
/// pairing form by construction.
///
/// Every variant has a `cpu_golden`-backed host evaluation in
/// [`eval_node`].
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum SubOp<F: RopeForm = NeoX, K: KvCacheShape = LlamaShape8x64> {
    /// Matmul output tile over one K-chunk:
    /// `out[i, j] = Σ_l A[i, l] · W[l, j]`.
    /// `inputs[0]` = A slice `[mr, kr]`; `inputs[1]` = W slice `[kr, nr]`
    /// (W is row-major `[K, N]` per the FUF convention `gemm(x:
    /// [..., K], w: [K, N])`). Output is the dense partial
    /// `[mr.len, nr.len]` contributed by this K-chunk.
    MatmulTile,
    /// Elementwise sum of equal-shaped inputs — the split-K combine. All
    /// inputs and the output are `[out_rows, out_cols]`.
    SumReduce,
    /// Shape-preserving elementwise op over a tile. Unary (`Silu`) reads
    /// `inputs[0]`; binary (`Mul`, `Add`) read `inputs[0]` and
    /// `inputs[1]`, both matching the output shape. Col-tiling never
    /// reorders a computation, so always bit-exact vs the whole op.
    Elementwise(EwKind),
    /// Fused SwiGLU activation: `out[j] = silu(gate[j]) * up[j]`.
    /// `inputs[0]` = gate, `inputs[1]` = up, both `[out_rows, out_cols]`.
    /// Matches `cpu_golden::fused_gate_up_silu_mul`. The GPU has only a
    /// *fused* `silu_mul` arm (no standalone silu), so the MLP's separate
    /// `Silu` + `Mul` are fused into this one node *before scheduling*
    /// (so the pair is one node in the dataflow graph and downstream
    /// schedulers treat it atomically); see `crate::lower::fuse_silu_mul`.
    SiluMul,
    /// RMS-norm over each row: `out[i] = x[i] / rms(x[i,:]) * weight`,
    /// `rms = sqrt(mean(x²) + eps)`. `inputs[0]` = x `[rows, cols]`,
    /// `inputs[1]` = weight `[1, cols]`.
    RmsNorm { eps: f32 },
    /// Rotary embedding over `[rows, heads * head_dim]` in the
    /// `F: RopeForm` pairing. `inputs[0]` = x, `inputs[1]` = cos row,
    /// `inputs[2]` = sin row.
    RopeRotate {
        head_dim: u32,
        #[doc(hidden)]
        _form: PhantomData<F>,
    },
    /// The K-side `rope_append`: rotate K in the `F: RopeForm` pairing
    /// **and** write the rotated K + un-rotated V into the paged KV
    /// cache (the cache identity is bound by `layout`). The host eval is
    /// **rotation only** (identical to [`SubOp::RopeRotate`]); V and the
    /// cache write are GPU-only.
    RopeAppend {
        head_dim: u32,
        layer: u32,
        layout: KvCacheLayout<K>,
        #[doc(hidden)]
        _form: PhantomData<F>,
    },
    /// Decode attention. `inputs[0]` = Q `[Mq, num_q_heads * head_dim]`;
    /// the remaining inputs are alternating `(K_seg, V_seg)` pairs.
    /// `layout` names the K-cache (single source of truth shared with
    /// the producing `RopeAppend`); `producer` names how that cache got
    /// populated; `softmax_state` is the per-AttnDecode online-softmax
    /// accumulator id.
    AttnDecode {
        num_q_heads: u32,
        num_kv_heads: u32,
        head_dim: u32,
        scale: f32,
        layout: KvCacheLayout<K>,
        producer: KvCacheProducer,
        softmax_state: SoftmaxStateId,
    },
}

// ── Nodes & graph ──────────────────────────────────────────────────

/// One unit of work: reads `inputs` (regions of tensors), computes its
/// `op`, and writes the result to `output` (a region of one op-output
/// tensor). Produces a dense
/// `[output.region.rows.len, output.region.cols.len]` buffer that is
/// scattered into the output tensor.
#[derive(Clone, Debug)]
pub struct SubtileNode<F: RopeForm = NeoX, K: KvCacheShape = LlamaShape8x64> {
    pub id: SubtileId,
    pub op: SubOp<F, K>,
    pub inputs: Vec<TensorRegion>,
    pub output: TensorRegion,
}

/// The canonical wavefront SubtileIR — a tensor-region SSA dataflow
/// graph in the `F: RopeForm` pairing. Nodes are topologically ordered:
/// every node that reads an op-output region is preceded by the nodes
/// that write the overlapping region (so a single pass over `nodes` is
/// a valid evaluation order).
#[derive(Clone, Debug)]
pub struct SubtileIR<F: RopeForm = NeoX, K: KvCacheShape = LlamaShape8x64> {
    pub tensors: Vec<TensorShape>,
    /// `tensors[0..num_sources]` are leaf sources.
    pub num_sources: u32,
    pub nodes: Vec<SubtileNode<F, K>>,
    /// The tensor whose buffer is the forward result (logits).
    pub result: TensorId,
}

impl<F: RopeForm, K: KvCacheShape> SubtileIR<F, K> {
    pub fn shape(&self, t: TensorId) -> TensorShape {
        self.tensors[t.0 as usize]
    }
    pub fn is_source(&self, t: TensorId) -> bool {
        t.0 < self.num_sources
    }
    /// The rope form of this IR. All rope nodes use this pairing by
    /// construction (the const generic guarantees it).
    pub const fn rope_form(&self) -> RopeFormTag {
        F::TAG
    }
}

// ── Host evaluation ────────────────────────────────────────────────

/// Gather a tensor region into a dense row-major `(buf, rows, cols)`.
fn gather<F: RopeForm, K: KvCacheShape>(
    tr: &TensorRegion,
    graph: &SubtileIR<F, K>,
    bufs: &[Vec<f32>],
) -> (Vec<f32>, u32, u32) {
    let shape = graph.shape(tr.tensor);
    let src = &bufs[tr.tensor.0 as usize];
    let (r, c) = (tr.region.rows.len, tr.region.cols.len);
    let mut out = Vec::with_capacity((r * c) as usize);
    for i in 0..r {
        let row = tr.region.rows.start + i;
        let base = ((row * shape.cols) + tr.region.cols.start) as usize;
        out.extend_from_slice(&src[base..base + c as usize]);
    }
    (out, r, c)
}

/// Scatter a dense `[rows, cols]` buffer into `bufs[tensor]` at `region`.
pub fn scatter(
    bufs: &mut [Vec<f32>],
    tensor: TensorId,
    region: Region,
    data: &[f32],
    shape: TensorShape,
) {
    let (r, c) = (region.rows.len as usize, region.cols.len as usize);
    // Host-evaluator helper; r * c == data.len() is structurally
    // entailed by callers (eval_node always builds out from the
    // region geometry it then scatters with). Defensive in debug only.
    debug_assert_eq!(data.len(), r * c, "scatter data size vs region");
    let dst = &mut bufs[tensor.0 as usize];
    for i in 0..r {
        let row = region.rows.start as usize + i;
        let base = row * shape.cols as usize + region.cols.start as usize;
        dst[base..base + c].copy_from_slice(&data[i * c..(i + 1) * c]);
    }
}

/// Evaluate the SubtileIR on the host. `sources[s]` is the row-major
/// buffer for source tensor `s` (`s < num_sources`), matching
/// `graph.tensors[s]`. Returns the backing buffer of every tensor
/// (indexed by [`TensorId`]); the logits are `bufs[graph.result]`.
pub fn eval_dag<F: RopeForm, K: KvCacheShape>(graph: &SubtileIR<F, K>, sources: &[&[f32]]) -> Vec<Vec<f32>> {
    // Host-evaluator source-count check; debug-only since the
    // codegen-side validate() catches structural mismatches.
    debug_assert_eq!(
        sources.len(),
        graph.num_sources as usize,
        "source count mismatch"
    );
    let mut bufs: Vec<Vec<f32>> = graph
        .tensors
        .iter()
        .map(|t| vec![0f32; (t.rows * t.cols) as usize])
        .collect();
    for (s, src) in sources.iter().enumerate() {
        debug_assert_eq!(src.len(), bufs[s].len(), "source {s} buffer size mismatch");
        bufs[s].copy_from_slice(src);
    }
    for node in &graph.nodes {
        let out = eval_node(node, graph, &bufs);
        let shape = graph.shape(node.output.tensor);
        scatter(
            &mut bufs,
            node.output.tensor,
            node.output.region,
            &out,
            shape,
        );
    }
    bufs
}

/// Compute one node's dense `[out_rows, out_cols]` output. Per-op
/// arithmetic mirrors `cpu_golden`.
pub fn eval_node<F: RopeForm, K: KvCacheShape>(
    node: &SubtileNode<F, K>,
    graph: &SubtileIR<F, K>,
    bufs: &[Vec<f32>],
) -> Vec<f32> {
    let out_rows = node.output.region.rows.len;
    let out_cols = node.output.region.cols.len;
    match node.op {
        SubOp::MatmulTile => {
            let (a, ar, ac) = gather(&node.inputs[0], graph, bufs);
            let (w, wr, wc) = gather(&node.inputs[1], graph, bufs);
            // Host-evaluator shape checks — debug-only. The codegen
            // pipeline is the source of truth (`validate()` + the
            // typed witnesses on SubOp); these are defensive on the
            // f32 reference path only. W is `[K, N]` per the FUF
            // convention (see `SubOp::MatmulTile` doc).
            debug_assert_eq!(ac, wr, "matmul K mismatch");
            debug_assert_eq!(ar, out_rows, "matmul A rows vs out_rows");
            debug_assert_eq!(wc, out_cols, "matmul W cols vs out_cols");
            let (m, n, k) = (ar as usize, wc as usize, ac as usize);
            let mut out = vec![0f32; m * n];
            for i in 0..m {
                for j in 0..n {
                    let mut sum = 0f32;
                    for l in 0..k {
                        // W is [K, N] row-major: w[l * N + j].
                        sum += a[i * k + l] * w[l * n + j];
                    }
                    out[i * n + j] = sum;
                }
            }
            out
        }
        SubOp::SumReduce => {
            let len = (out_rows * out_cols) as usize;
            let mut out = vec![0f32; len];
            for inp in &node.inputs {
                let (b, _, _) = gather(inp, graph, bufs);
                debug_assert_eq!(b.len(), len, "reduce operand size mismatch");
                for (o, v) in out.iter_mut().zip(&b) {
                    *o += *v;
                }
            }
            out
        }
        SubOp::Elementwise(kind) => {
            let (a, ar, ac) = gather(&node.inputs[0], graph, bufs);
            debug_assert_eq!((ar, ac), (out_rows, out_cols), "elementwise shape");
            match kind {
                EwKind::Silu => a.iter().map(|&x| x / (1.0 + (-x).exp())).collect(),
                EwKind::Mul | EwKind::Add => {
                    let (b, br, bc) = gather(&node.inputs[1], graph, bufs);
                    debug_assert_eq!((br, bc), (ar, ac), "elementwise binary shape");
                    a.iter()
                        .zip(&b)
                        .map(|(&x, &y)| {
                            if matches!(kind, EwKind::Mul) {
                                x * y
                            } else {
                                x + y
                            }
                        })
                        .collect()
                }
            }
        }
        SubOp::SiluMul => {
            let (a, ar, ac) = gather(&node.inputs[0], graph, bufs);
            let (b, br, bc) = gather(&node.inputs[1], graph, bufs);
            debug_assert_eq!((ar, ac), (out_rows, out_cols), "silu_mul gate shape");
            debug_assert_eq!((br, bc), (ar, ac), "silu_mul up shape");
            a.iter()
                .zip(&b)
                .map(|(&g, &u)| (g / (1.0 + (-g).exp())) * u)
                .collect()
        }
        SubOp::RmsNorm { eps } => {
            let (x, xr, xc) = gather(&node.inputs[0], graph, bufs);
            let (wt, _wr, wc) = gather(&node.inputs[1], graph, bufs);
            debug_assert_eq!((xr, xc), (out_rows, out_cols), "rmsnorm shape");
            debug_assert_eq!(wc, out_cols, "rmsnorm weight width");
            let (m, d) = (xr as usize, xc as usize);
            let mut out = vec![0f32; m * d];
            for i in 0..m {
                let row = &x[i * d..(i + 1) * d];
                let sum_sq: f32 = row.iter().map(|&v| v * v).sum();
                let inv_rms = 1.0 / (sum_sq / d as f32 + eps).sqrt();
                for j in 0..d {
                    out[i * d + j] = row[j] * inv_rms * wt[j];
                }
            }
            out
        }
        // RopeAppend's host eval is rotation only (identical to RopeRotate);
        // its V input + the paged-cache write are GPU-only.
        SubOp::RopeRotate { head_dim, _form: _ }
        | SubOp::RopeAppend {
            head_dim, _form: _, ..
        } => {
            let (x, xr, xc) = gather(&node.inputs[0], graph, bufs);
            let (cos, _, cc) = gather(&node.inputs[1], graph, bufs);
            let (sin, _, sc) = gather(&node.inputs[2], graph, bufs);
            debug_assert_eq!((xr, xc), (out_rows, out_cols), "rope shape");
            let hd = head_dim as usize;
            let half = hd / 2;
            let (rows, cols) = (xr as usize, xc as usize);
            debug_assert_eq!(cols % hd, 0, "rope cols not a multiple of head_dim");
            debug_assert!(cc as usize >= hd && sc as usize >= hd, "rope cos/sin width");
            let heads = cols / hd;
            let mut out = x.clone();
            for r in 0..rows {
                for h in 0..heads {
                    let base = r * cols + h * hd;
                    for d in 0..half {
                        let x0 = x[base + d];
                        let x1 = x[base + half + d];
                        let (c, s) = (cos[d], sin[d]);
                        out[base + d] = x0 * c - x1 * s;
                        out[base + half + d] = x0 * s + x1 * c;
                    }
                }
            }
            out
        }
        SubOp::AttnDecode {
            num_q_heads,
            num_kv_heads,
            head_dim,
            scale,
            ..
        } => {
            // Head-block aware: this node computes a contiguous q-head range,
            // derived from the OUTPUT region (its column slice), and reads the
            // matching kv-head range — derived from the K input region. The
            // SubOp keeps the GLOBAL head counts so the GQA ratio is exact; the
            // block's own counts come from the slice widths. Each q-head's
            // attention is independent ⇒ bit-exact vs the whole op.
            let hd = head_dim as usize;
            let gqa = (num_q_heads / num_kv_heads.max(1)) as usize;
            let (q, qr, qc) = gather(&node.inputs[0], graph, bufs);
            let mq = qr as usize;
            let qh_count = qc as usize / hd; // q-heads in this block
            let qh_start = node.output.region.cols.start as usize / hd; // global first q-head
            debug_assert_eq!(
                qc as usize,
                qh_count * hd,
                "attn Q width is a head multiple"
            );
            debug_assert!(node.inputs.len() >= 3, "attn needs Q + >=1 (K,V) segment");
            debug_assert_eq!(node.inputs.len() % 2, 1, "attn inputs = Q + (K,V) pairs");
            // kv-head offset of this block (from the first K segment's column slice).
            let kvh_start = node.inputs[1].region.cols.start as usize / hd;
            // Concatenate K/V segments along sequence; each segment spans this
            // block's kv-heads (kv_count * hd wide).
            let mut k_all: Vec<f32> = Vec::new();
            let mut v_all: Vec<f32> = Vec::new();
            let mut kv_count = 0usize;
            let mut i = 1;
            while i < node.inputs.len() {
                let (k, kr, kc) = gather(&node.inputs[i], graph, bufs);
                let (v, vr, vc) = gather(&node.inputs[i + 1], graph, bufs);
                debug_assert_eq!((vr, vc), (kr, kc), "attn V seg shape");
                kv_count = kc as usize / hd;
                k_all.extend_from_slice(&k);
                v_all.extend_from_slice(&v);
                i += 2;
            }
            debug_assert!(kv_count >= 1, "attn K seg has at least one kv-head");
            let seg_w = kv_count * hd;
            let seq_len = k_all.len() / seg_w;
            let mut out = vec![0f32; mq * qh_count * hd];
            for qi in 0..mq {
                for hl in 0..qh_count {
                    // local q-head hl → global q-head → global kv-head → local kv.
                    let global_kv = (qh_start + hl) / gqa;
                    let local_kv = global_kv - kvh_start;
                    let q_off = qi * qh_count * hd + hl * hd;
                    let mut scores = vec![0f32; seq_len];
                    for (s, score) in scores.iter_mut().enumerate() {
                        let k_off = s * seg_w + local_kv * hd;
                        let mut dot = 0f32;
                        for d in 0..hd {
                            dot += q[q_off + d] * k_all[k_off + d];
                        }
                        *score = dot * scale;
                    }
                    let maxs = scores.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                    let mut sum = 0f32;
                    for sc in scores.iter_mut() {
                        *sc = (*sc - maxs).exp();
                        sum += *sc;
                    }
                    for sc in scores.iter_mut() {
                        *sc /= sum;
                    }
                    for d in 0..hd {
                        let mut val = 0f32;
                        for (s, &wgt) in scores.iter().enumerate() {
                            val += wgt * v_all[s * seg_w + local_kv * hd + d];
                        }
                        out[q_off + d] = val;
                    }
                }
            }
            out
        }
    }
}

/// The forward result buffer (logits) — `bufs[graph.result]`.
pub fn result_buffer<'a, F: RopeForm, K: KvCacheShape>(graph: &SubtileIR<F, K>, bufs: &'a [Vec<f32>]) -> &'a [f32] {
    &bufs[graph.result.0 as usize]
}

// ── Dependencies (region overlap) ──────────────────────────────────

fn ranges_overlap(a: Range, b: Range) -> bool {
    a.start < b.end() && b.start < a.end()
}

fn regions_overlap(a: Region, b: Region) -> bool {
    ranges_overlap(a.rows, b.rows) && ranges_overlap(a.cols, b.cols)
}

/// Predecessor node ids for each node: the earlier nodes whose write
/// overlaps one of this node's input reads on the same op-output tensor.
/// Reads of leaf sources contribute no dependency. This is the edge set
/// the per-target lowering turns into cross-execution-unit
/// synchronization (whatever primitive the target prefers).
pub fn predecessors<F: RopeForm, K: KvCacheShape>(graph: &SubtileIR<F, K>) -> Vec<Vec<SubtileId>> {
    // writers[t] = (node_id, out_region) for each op-output tensor, in id order.
    let mut writers: Vec<Vec<(u32, Region)>> = vec![Vec::new(); graph.tensors.len()];
    let mut preds: Vec<Vec<SubtileId>> = Vec::with_capacity(graph.nodes.len());
    for node in &graph.nodes {
        let mut p: Vec<SubtileId> = Vec::new();
        for inp in &node.inputs {
            if graph.is_source(inp.tensor) {
                continue;
            }
            for (wid, wreg) in &writers[inp.tensor.0 as usize] {
                if regions_overlap(*wreg, inp.region) && !p.contains(&SubtileId(*wid)) {
                    p.push(SubtileId(*wid));
                }
            }
        }
        p.sort();
        preds.push(p);
        writers[node.output.tensor.0 as usize].push((node.id.0, node.output.region));
    }
    preds
}

// ── Structural validation + ValidatedGraph<F> witness ──────────────

/// Sealed proof that a [`SubtileIR<F>`] passed [`validate`]. The only
/// way to obtain one is [`ValidatedGraph::new`] — internally calls
/// [`validate`] and on success wraps the borrow with a sealed marker.
/// Downstream lowerings (`lower_dag_to_tape`) take
/// `&ValidatedGraph<F>` and can elide their own runtime
/// validate-and-expect, so structural-precondition violations become
/// "no value to consume" type errors rather than runtime panics
/// (per `feedback_compile_time_or_garbage` and §5 K5).
pub struct ValidatedGraph<'g, F: RopeForm, K: KvCacheShape = LlamaShape8x64> {
    inner: &'g SubtileIR<F, K>,
    _seal: sealed::Seal,
}

impl<'g, F: RopeForm, K: KvCacheShape> ValidatedGraph<'g, F, K> {
    /// Validate `graph` and produce the sealed witness. Returns the
    /// validation error string verbatim on failure.
    pub fn new(graph: &'g SubtileIR<F, K>) -> Result<Self, String> {
        validate(graph)?;
        Ok(Self {
            inner: graph,
            _seal: sealed::Seal(()),
        })
    }

    /// Borrow the underlying graph. Consumers cannot fabricate a
    /// `ValidatedGraph` without going through [`Self::new`], so this
    /// borrow is proof-carrying.
    pub fn graph(&self) -> &'g SubtileIR<F, K> {
        self.inner
    }
}

/// Check the graph's invariants without evaluating: dense ids, in-range
/// tensors, in-bounds regions, op-output (not source) write targets,
/// op arity, and that every op-output read is covered by writers with a
/// strictly smaller id (acyclic + assembled-before-read). Returns the
/// node count on success. Prefer [`ValidatedGraph::new`] in the
/// wavefront lowerings (the typed witness elides downstream runtime
/// gates).
pub fn validate<F: RopeForm, K: KvCacheShape>(graph: &SubtileIR<F, K>) -> Result<usize, String> {
    let n_tensors = graph.tensors.len() as u32;
    if graph.num_sources > n_tensors {
        return Err(format!(
            "num_sources {} exceeds tensor count {n_tensors}",
            graph.num_sources
        ));
    }
    // Track written sub-regions per op-output tensor to check coverage.
    let mut writes: Vec<Vec<(u32, Region)>> = vec![Vec::new(); graph.tensors.len()];
    for (i, node) in graph.nodes.iter().enumerate() {
        if node.id.0 as usize != i {
            return Err(format!("node {i} has non-dense id {}", node.id.0));
        }
        let arity = node.inputs.len();
        let arity_ok = match node.op {
            SubOp::MatmulTile => arity == 2,
            SubOp::SumReduce => arity >= 1,
            SubOp::Elementwise(EwKind::Silu) => arity == 1,
            SubOp::Elementwise(EwKind::Mul | EwKind::Add) => arity == 2,
            SubOp::SiluMul => arity == 2,
            SubOp::RmsNorm { .. } => arity == 2,
            SubOp::RopeRotate { .. } => arity == 3,
            // E.12: RopeAppend takes [K, cos, sin, V, K_cache, V_cache]
            // — the K_cache / V_cache are per-layer PrefixK / PrefixV
            // sources used as TMA-store destinations for the new
            // decode token's K/V.
            SubOp::RopeAppend { .. } => arity == 6,
            SubOp::AttnDecode { .. } => arity >= 3 && arity % 2 == 1,
        };
        if !arity_ok {
            return Err(format!("node {i} op {:?} bad arity {arity}", node.op));
        }
        // Output must target an op-output tensor, in bounds.
        let ot = node.output.tensor;
        if ot.0 >= n_tensors {
            return Err(format!("node {i} writes tensor {} out of range", ot.0));
        }
        if graph.is_source(ot) {
            return Err(format!("node {i} writes leaf source tensor {}", ot.0));
        }
        let oshape = graph.shape(ot);
        if node.output.region.rows.end() > oshape.rows
            || node.output.region.cols.end() > oshape.cols
        {
            return Err(format!("node {i} output region exceeds tensor {}", ot.0));
        }
        // Inputs in bounds; op-output reads must be covered by prior writers.
        for (a, inp) in node.inputs.iter().enumerate() {
            if inp.tensor.0 >= n_tensors {
                return Err(format!(
                    "node {i} input {a} tensor {} out of range",
                    inp.tensor.0
                ));
            }
            let ishape = graph.shape(inp.tensor);
            if inp.region.rows.end() > ishape.rows || inp.region.cols.end() > ishape.cols {
                return Err(format!(
                    "node {i} input {a} region exceeds tensor {}",
                    inp.tensor.0
                ));
            }
            if !graph.is_source(inp.tensor) {
                // Some earlier write must overlap (cheap acyclicity / use-before-def check).
                let has = writes[inp.tensor.0 as usize]
                    .iter()
                    .any(|(wid, wreg)| *wid < node.id.0 && regions_overlap(*wreg, inp.region));
                if !has {
                    return Err(format!(
                        "node {i} input {a} reads tensor {} region with no prior writer",
                        inp.tensor.0
                    ));
                }
            }
        }
        writes[ot.0 as usize].push((node.id.0, node.output.region));
    }
    if graph.result.0 >= n_tensors {
        return Err(format!("result tensor {} out of range", graph.result.0));
    }
    if graph.is_source(graph.result) {
        return Err("result is a leaf source, not an op output".into());
    }
    Ok(graph.nodes.len())
}

// ── LoweringInput → SubtileIR ──────────────────────────────────────

/// Tile `[0, total)` into contiguous blocks of width `block` (last block
/// may be shorter). `block.get() >= total` yields a single whole block.
///
/// `block: NonZeroU32` discharges the load-bearing termination invariant
/// at the type level — `block == 0` would loop forever (per
/// `feedback_compile_time_or_garbage` this must be a compile error, not
/// a runtime assert).
pub fn n_blocks(total: u32, block: std::num::NonZeroU32) -> Vec<Range> {
    let block = block.get();
    let mut out = Vec::new();
    let mut start = 0;
    while start < total {
        let len = block.min(total - start);
        out.push(Range::new(start, len));
        start += len;
    }
    if out.is_empty() {
        out.push(Range::new(0, 0)); // total==0 (degenerate); keep one empty block
    }
    out
}

/// Tile `[0, total)` into **head-aligned** blocks of width ~`nb`, snapped
/// down to a whole number of `head_dim`-wide heads (at least one head). Used
/// for rope/attention so every block is a clean set of heads — the q→rope→
/// attn chain partitions on head boundaries (and o_proj split-Ks on them).
///
/// `nb: NonZeroU32` propagates the same termination witness as
/// [`n_blocks`]; `head_dim: NonZeroU32` makes the K5 termination
/// invariant structural — `block: NonZeroU32` is constructed without
/// any `.expect()`, since `NonZeroU32::saturating_mul` preserves
/// nonzero by type.
pub fn head_blocks(total: u32, nb: std::num::NonZeroU32, head_dim: std::num::NonZeroU32) -> Vec<Range> {
    use std::num::NonZeroU32;
    // `nb.get() / hd` may be zero (when nb < hd); fall back to one
    // head — `NonZeroU32::new(...).unwrap_or(NonZeroU32::MIN)` is the
    // canonical "floor at 1" idiom on a NonZeroU32-output path.
    let heads_per_block = NonZeroU32::new(nb.get() / head_dim.get()).unwrap_or(NonZeroU32::MIN);
    let block = heads_per_block.saturating_mul(head_dim);
    n_blocks(total, block)
}

/// Out-columns of an op (mirrors `crate::lower`): GEMM → n, attention →
/// `num_q_heads * head_dim`, everything else preserves input-0 width.
pub(crate) fn op_out_cols(op: crate::lower::LoweredOp, in0_cols: u32) -> u32 {
    use crate::lower::LoweredOp;
    match op {
        LoweredOp::Gemm { n } => n,
        LoweredOp::AttnDecode {
            num_q_heads,
            head_dim,
            ..
        } => num_q_heads * head_dim,
        LoweredOp::RmsNorm { .. }
        | LoweredOp::Silu
        | LoweredOp::Mul
        | LoweredOp::SiluMul
        | LoweredOp::Add
        | LoweredOp::RopeRotate { .. }
        | LoweredOp::RopeAppend { .. } => in0_cols,
    }
}

/// Lower a flat [`crate::lower::LoweringInput`] to a SubtileIR,
/// **N-block tiling every GEMM** by `nb` (output columns split into
/// `ceil(n/nb)` MatmulTile subtiles, each writing a disjoint column
/// slice of the op's output tensor — no reduce, bit-exact). All other
/// ops stay whole (one subtile writing the whole output tensor).
/// `nb >= n` ⇒ a single block (coarse, equivalent to v1). Source tensors
/// mirror `input.sources`; op-output tensor `i` is
/// `TensorId(num_sources + i)`.
/// Lower a flat [`crate::lower::LoweringInput`] to a `SubtileIR<NeoX>`.
/// Llama-3.2 uses NeoX rotary; Interleaved-form lowerings (other
/// architectures) construct `SubtileIR<Interleaved>` directly. Mixing
/// forms in one IR is impossible by construction (the IR's rope nodes
/// carry `PhantomData<F>`, so a SubtileIR<NeoX> cannot hold an
/// Interleaved-form rope node).
pub fn lower_region<K: KvCacheShape>(input: &crate::lower::LoweringInput, nb: std::num::NonZeroU32) -> SubtileIR<NeoX, K> {
    use crate::lower::{InputRef, LoweredOp};
    let num_sources = input.sources.len() as u32;
    let mut tensors: Vec<TensorShape> = input
        .sources
        .iter()
        .map(|s| TensorShape {
            rows: s.rows,
            cols: s.cols,
        })
        .collect();
    let mut op_tensor: Vec<TensorId> = Vec::with_capacity(input.ops.len());
    let mut op_cols: Vec<u32> = Vec::with_capacity(input.ops.len());
    let mut nodes: Vec<SubtileNode<NeoX, K>> = Vec::new();
    // RopeAppend node id keyed by the K-cache TensorId it writes — used
    // to compute `KvCacheProducer` for any AttnDecode that reads the
    // same cache later in the forward.
    let mut k_cache_producer_node: std::collections::HashMap<TensorId, u32> =
        std::collections::HashMap::new();
    let mut next_softmax_state: u32 = 0;

    // Resolve an InputRef to (tensor id, shape).
    let resolve = |r: InputRef,
                   op_tensor: &[TensorId],
                   op_cols: &[u32],
                   tensors: &[TensorShape]|
     -> (TensorId, u32, u32) {
        match r {
            InputRef::Op(j) => {
                let t = op_tensor[j];
                (t, tensors[t.0 as usize].rows, op_cols[j])
            }
            InputRef::Ext(e) => {
                let s = tensors[e];
                (TensorId(e as u32), s.rows, s.cols)
            }
        }
    };
    let whole = |t: TensorId, tensors: &[TensorShape]| -> TensorRegion {
        TensorRegion {
            tensor: t,
            region: tensors[t.0 as usize].whole(),
        }
    };

    for desc in &input.ops {
        let m = desc.m;
        let (in0_t, _in0_rows, in0_cols) = resolve(desc.inputs[0], &op_tensor, &op_cols, &tensors);
        let out_cols = op_out_cols(desc.op, in0_cols);
        let out_t = TensorId(tensors.len() as u32);
        tensors.push(TensorShape {
            rows: m,
            cols: out_cols,
        });

        match desc.op {
            LoweredOp::Gemm { n } => {
                // k is structurally derived from the activation's
                // column count — it is not a separate field. See
                // LoweredOp::Gemm doc.
                let k = in0_cols;
                let (w_t, _wr, _wc) = resolve(desc.inputs[1], &op_tensor, &op_cols, &tensors);
                let act = TensorRegion {
                    tensor: in0_t,
                    region: Region {
                        rows: Range::new(0, m),
                        cols: Range::new(0, k),
                    },
                };
                for blk in n_blocks(n, nb) {
                    let id = SubtileId(nodes.len() as u32);
                    nodes.push(SubtileNode {
                        id,
                        op: SubOp::MatmulTile,
                        inputs: vec![
                            act,
                            // W is row-major [K, N] per FUF
                            // `gemm(x: [..., K], w: [K, N])`. The
                            // n-block selects an N-slice of W; we
                            // read all K rows of that N-slice.
                            TensorRegion {
                                tensor: w_t,
                                region: Region {
                                    rows: Range::new(0, k),
                                    cols: blk,
                                },
                            },
                        ],
                        output: TensorRegion {
                            tensor: out_t,
                            region: Region {
                                rows: Range::new(0, m),
                                cols: blk,
                            },
                        },
                    });
                }
            }
            other => {
                // Populate per-variant typed witnesses. RopeAppend's
                // KvCacheLayout is keyed on its `K_cache` input
                // (desc.inputs[4] per the validate-arity contract);
                // AttnDecode's KvCacheLayout/KvCacheProducer are keyed
                // on its prefix-K input (desc.inputs[1] in the fused
                // decode shape).
                let subop: SubOp<NeoX, K> = match other {
                    LoweredOp::RmsNorm { eps } => SubOp::RmsNorm { eps },
                    LoweredOp::Silu => SubOp::Elementwise(EwKind::Silu),
                    LoweredOp::Mul => SubOp::Elementwise(EwKind::Mul),
                    LoweredOp::SiluMul => SubOp::SiluMul,
                    LoweredOp::Add => SubOp::Elementwise(EwKind::Add),
                    LoweredOp::RopeRotate { head_dim } => SubOp::RopeRotate {
                        head_dim,
                        _form: PhantomData,
                    },
                    LoweredOp::RopeAppend { head_dim, layer } => {
                        // K-cache TensorId comes from desc.inputs[4]
                        // (the validate arity-6 contract); for legacy
                        // 4-input fixtures (K, cos, sin, V only) fall
                        // back to the K input's own tensor — the layout
                        // witness is consulted only when the producer/
                        // consumer pair is end-to-end (RopeAppend +
                        // AttnDecode), so the sentinel never escapes.
                        let k_cache_t = if desc.inputs.len() > 4 {
                            resolve(desc.inputs[4], &op_tensor, &op_cols, &tensors).0
                        } else {
                            in0_t
                        };
                        let num_kv_heads = (in0_cols / head_dim.max(1)).max(1);
                        // K7 runtime gate: orchestrator's runtime
                        // num_kv_heads/head_dim must match
                        // `LlamaShape8x64`'s associated consts. Past
                        // this gate the witness type carries the proof.
                        assert_eq!(num_kv_heads, K::NUM_KV_HEADS);
                        assert_eq!(head_dim, K::HEAD_DIM);
                        let layout =
                            KvCacheLayout::<K>::for_cache_tensor(k_cache_t);
                        k_cache_producer_node.insert(k_cache_t, nodes.len() as u32);
                        SubOp::RopeAppend {
                            head_dim,
                            layer,
                            layout,
                            _form: PhantomData,
                        }
                    }
                    LoweredOp::AttnDecode {
                        num_q_heads,
                        num_kv_heads,
                        head_dim,
                        scale,
                    } => {
                        let (prefix_k_t, _, _) =
                            resolve(desc.inputs[1], &op_tensor, &op_cols, &tensors);
                        assert_eq!(num_kv_heads, K::NUM_KV_HEADS);
                        assert_eq!(head_dim, K::HEAD_DIM);
                        let layout =
                            KvCacheLayout::<K>::for_cache_tensor(prefix_k_t);
                        let producer = match k_cache_producer_node.get(&prefix_k_t) {
                            Some(&node_idx) => KvCacheProducer::from_rope_append(node_idx),
                            None => KvCacheProducer::pre_populated_ext(),
                        };
                        let softmax_state = SoftmaxStateId::new(next_softmax_state);
                        next_softmax_state += 1;
                        SubOp::AttnDecode {
                            num_q_heads,
                            num_kv_heads,
                            head_dim,
                            scale,
                            layout,
                            producer,
                            softmax_state,
                        }
                    }
                    LoweredOp::Gemm { .. } => unreachable!("gemm handled above"),
                };
                // A pure elementwise op (silu/mul/add/silu·mul) is tiled by the
                // output column slice like the GEMM N-blocks, so a downstream
                // per-target lowering can dispatch the tiles independently.
                // rope / attn (head structure) and
                // rmsnorm (RMS reduction) stay WHOLE here — this is the bit-exact,
                // GPU-correct reference lowering + the live per-op-schedule path.
                // The head-tiling, split-K all-reduce and replication of the
                // tensor-parallel partition live in
                // [`crate::partition::lower_partitioned`], kept separate because
                // split-K reassociates (breaks this fn's bit-exact contract) and
                // because the partition needs the new GPU emit/player arms.
                let elementwise = matches!(
                    other,
                    LoweredOp::Silu | LoweredOp::Mul | LoweredOp::Add | LoweredOp::SiluMul
                );
                let blocks = if elementwise {
                    n_blocks(out_cols, nb)
                } else {
                    vec![Range::new(0, out_cols)]
                };
                for blk in blocks {
                    let inputs: Vec<TensorRegion> = desc
                        .inputs
                        .iter()
                        .map(|r| {
                            let (t, _, _) = resolve(*r, &op_tensor, &op_cols, &tensors);
                            if elementwise {
                                TensorRegion {
                                    tensor: t,
                                    region: Region {
                                        rows: Range::new(0, m),
                                        cols: blk,
                                    },
                                }
                            } else {
                                whole(t, &tensors)
                            }
                        })
                        .collect();
                    let id = SubtileId(nodes.len() as u32);
                    nodes.push(SubtileNode {
                        id,
                        op: subop,
                        inputs,
                        output: TensorRegion {
                            tensor: out_t,
                            region: Region {
                                rows: Range::new(0, m),
                                cols: blk,
                            },
                        },
                    });
                }
            }
        }
        op_tensor.push(out_t);
        op_cols.push(out_cols);
    }

    SubtileIR {
        tensors,
        num_sources,
        nodes,
        result: op_tensor[input.result],
    }
}

// ── Tests (canonical region IR) ────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::lower::{InputRef, LoweredOp, OpDesc};
    use ferrite_forward::cpu_golden;

    fn rng_fill(n: usize, seed: u64) -> Vec<f32> {
        let mut s = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15).wrapping_add(1);
        (0..n)
            .map(|_| {
                s = s
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                let bits = (s >> 33) as u32;
                (bits as f32 / 2147483648.0) * 2.0 - 1.0
            })
            .collect()
    }

    /// N-block-tiled GEMM is bit-exact vs `cpu_golden::gemm` (disjoint
    /// output columns, same per-output reduction order). Built by hand at
    /// the IR level: source 0 = act [m,k], source 1 = W [k,n] (FUF
    /// convention; `cpu_golden::gemm` uses [n,k], so we transpose
    /// before calling it). Output tensor [m,n] is written by
    /// `ceil(n/nb)` MatmulTile blocks.
    #[test]
    fn gemm_nblock_bit_exact_vs_cpu_golden() {
        let (m, n, k) = (1u32, 130, 257);
        let act = rng_fill((m * k) as usize, 1);
        // W bytes laid out as [K, N] row-major (FUF convention).
        let w_kn = rng_fill((k * n) as usize, 2);
        // cpu_golden expects [N, K] row-major; transpose for the
        // reference call.
        let mut w_nk = vec![0f32; (n * k) as usize];
        for i in 0..k as usize {
            for j in 0..n as usize {
                w_nk[j * k as usize + i] = w_kn[i * n as usize + j];
            }
        }
        let mut want = vec![0f32; (m * n) as usize];
        cpu_golden::gemm(&act, &w_nk, &mut want, m as usize, k as usize, n as usize);

        for nb in [16u32, 48, 64, 130, 256] {
            let tensors = vec![
                TensorShape { rows: m, cols: k }, // 0: act [m,k]
                TensorShape { rows: k, cols: n }, // 1: W [k,n] (FUF)
                TensorShape { rows: m, cols: n }, // 2: out [m,n]
            ];
            let out_t = TensorId(2);
            let mut nodes: Vec<SubtileNode> = Vec::new();
            let mut start = 0u32;
            while start < n {
                let len = nb.min(n - start);
                let id = SubtileId(nodes.len() as u32);
                nodes.push(SubtileNode {
                    id,
                    op: SubOp::MatmulTile,
                    inputs: vec![
                        TensorRegion {
                            tensor: TensorId(0),
                            region: Region {
                                rows: Range::new(0, m),
                                cols: Range::new(0, k),
                            },
                        },
                        // W is [K, N]; n-block selects N-cols.
                        TensorRegion {
                            tensor: TensorId(1),
                            region: Region {
                                rows: Range::new(0, k),
                                cols: Range::new(start, len),
                            },
                        },
                    ],
                    output: TensorRegion {
                        tensor: out_t,
                        region: Region {
                            rows: Range::new(0, m),
                            cols: Range::new(start, len),
                        },
                    },
                });
                start += len;
            }
            let g = SubtileIR {
                tensors,
                num_sources: 2,
                nodes,
                result: out_t,
            };
            assert!(validate(&g).is_ok(), "valid nb={nb}");
            let bufs = eval_dag(&g, &[&act, &w_kn]);
            assert_eq!(result_buffer(&g, &bufs), &want[..], "nb={nb} bit-exact");
        }
    }

    /// Build a whole Llama-style decode layer as a `LoweringInput` and
    /// lower it with [`lower_region`] at several N-block widths; each must
    /// be bit-exact vs the `cpu_golden` composition.
    #[test]
    fn full_decode_layer_nblock_bit_exact() {
        let (h, hd, hq, hkv, i, l) = (16u32, 4u32, 4u32, 2u32, 32u32, 3u32);
        let (qdim, kvdim) = (hq * hd, hkv * hd);
        let eps = 1e-5f32;
        let scale = 1.0 / (hd as f32).sqrt();

        let res_in = rng_fill(h as usize, 101);
        let in_ln = rng_fill(h as usize, 102);
        let wq = rng_fill((qdim * h) as usize, 103);
        let wk = rng_fill((kvdim * h) as usize, 104);
        let wv = rng_fill((kvdim * h) as usize, 105);
        let cos = rng_fill(hd as usize, 106);
        let sin = rng_fill(hd as usize, 107);
        let prefix_k = rng_fill((l * kvdim) as usize, 108);
        let prefix_v = rng_fill((l * kvdim) as usize, 109);
        let wo = rng_fill((h * qdim) as usize, 110);
        let post_ln = rng_fill(h as usize, 111);
        let wgate = rng_fill((i * h) as usize, 112);
        let wup = rng_fill((i * h) as usize, 113);
        let wdown = rng_fill((h * i) as usize, 114);

        // Reference via cpu_golden.
        let (hs, hds, hqs, hkvs, is, qd, kvd) = (
            h as usize,
            hd as usize,
            hq as usize,
            hkv as usize,
            i as usize,
            qdim as usize,
            kvdim as usize,
        );
        // Weight bytes generated as [K=h, N=*] (FUF convention). Reference
        // `cpu_golden::gemm` expects [N, K]; transpose before each call.
        fn transpose_kn(w_kn: &[f32], k: usize, n: usize) -> Vec<f32> {
            let mut out = vec![0f32; n * k];
            for i in 0..k {
                for j in 0..n {
                    out[j * k + i] = w_kn[i * n + j];
                }
            }
            out
        }
        let wq_nk = transpose_kn(&wq, hs, qd);
        let wk_nk = transpose_kn(&wk, hs, kvd);
        let wv_nk = transpose_kn(&wv, hs, kvd);
        let wo_nk = transpose_kn(&wo, qd, hs);
        let wgate_nk = transpose_kn(&wgate, hs, is);
        let wup_nk = transpose_kn(&wup, hs, is);
        let wdown_nk = transpose_kn(&wdown, is, hs);
        let mut xn = vec![0f32; hs];
        cpu_golden::rmsnorm(&res_in, &in_ln, &mut xn, eps);
        let mut q = vec![0f32; qd];
        cpu_golden::gemm(&xn, &wq_nk, &mut q, 1, hs, qd);
        let mut k = vec![0f32; kvd];
        cpu_golden::gemm(&xn, &wk_nk, &mut k, 1, hs, kvd);
        let mut v = vec![0f32; kvd];
        cpu_golden::gemm(&xn, &wv_nk, &mut v, 1, hs, kvd);
        let mut q_rot = vec![0f32; qd];
        cpu_golden::rope(&q, &cos, &sin, &[0i32], 1, hqs, hds, &mut q_rot);
        let mut k_rot = vec![0f32; kvd];
        cpu_golden::rope(&k, &cos, &sin, &[0i32], 1, hkvs, hds, &mut k_rot);
        let mut k_all = prefix_k.clone();
        k_all.extend_from_slice(&k_rot);
        let mut v_all = prefix_v.clone();
        v_all.extend_from_slice(&v);
        let mut attn = vec![0f32; qd];
        cpu_golden::attention_decode(
            &q_rot,
            &k_all,
            &v_all,
            &mut attn,
            (l + 1) as usize,
            hqs,
            hkvs,
            hds,
            scale,
        );
        let mut o = vec![0f32; hs];
        cpu_golden::gemm(&attn, &wo_nk, &mut o, 1, qd, hs);
        let mut res_mid = vec![0f32; hs];
        cpu_golden::add(&o, &res_in, &mut res_mid);
        let mut xn2 = vec![0f32; hs];
        cpu_golden::rmsnorm(&res_mid, &post_ln, &mut xn2, eps);
        let mut gate = vec![0f32; is];
        cpu_golden::gemm(&xn2, &wgate_nk, &mut gate, 1, hs, is);
        let mut up = vec![0f32; is];
        cpu_golden::gemm(&xn2, &wup_nk, &mut up, 1, hs, is);
        let mut act = vec![0f32; is];
        cpu_golden::fused_gate_up_silu_mul(&gate, &up, &mut act);
        let mut down = vec![0f32; hs];
        cpu_golden::gemm(&act, &wdown_nk, &mut down, 1, is, hs);
        let mut want = vec![0f32; hs];
        cpu_golden::add(&down, &res_mid, &mut want);

        // Same layer as a LoweringInput (sources 0..=13).
        let ss = |rows: u32, cols: u32| SourceShape { rows, cols };
        // Weights stored as [K, N] per FUF; q-proj is [h, qdim],
        // k/v-proj [h, kvdim], o-proj [qdim, h], MLP gate/up [h, i],
        // down [i, h]. The weight bytes (`wq`, `wk`, etc.) are
        // generated at length `K*N` either way; the eval just reads
        // them with stride N over K rows now.
        let input = crate::lower::LoweringInput {
            sources: vec![
                ss(1, h),
                ss(1, h),
                ss(h, qdim),
                ss(h, kvdim),
                ss(h, kvdim),
                ss(1, hd),
                ss(1, hd),
                ss(l, kvdim),
                ss(l, kvdim),
                ss(qdim, h),
                ss(1, h),
                ss(h, i),
                ss(h, i),
                ss(i, h),
            ],
            ops: vec![
                OpDesc {
                    op: LoweredOp::RmsNorm { eps },
                    m: 1,
                    inputs: vec![InputRef::Ext(0), InputRef::Ext(1)],
                },
                OpDesc {
                    op: LoweredOp::Gemm { n: qdim },
                    m: 1,
                    inputs: vec![InputRef::Op(0), InputRef::Ext(2)],
                },
                OpDesc {
                    op: LoweredOp::Gemm { n: kvdim },
                    m: 1,
                    inputs: vec![InputRef::Op(0), InputRef::Ext(3)],
                },
                OpDesc {
                    op: LoweredOp::Gemm { n: kvdim },
                    m: 1,
                    inputs: vec![InputRef::Op(0), InputRef::Ext(4)],
                },
                OpDesc {
                    op: LoweredOp::RopeRotate { head_dim: hd },
                    m: 1,
                    inputs: vec![InputRef::Op(1), InputRef::Ext(5), InputRef::Ext(6)],
                },
                OpDesc {
                    op: LoweredOp::RopeRotate { head_dim: hd },
                    m: 1,
                    inputs: vec![InputRef::Op(2), InputRef::Ext(5), InputRef::Ext(6)],
                },
                OpDesc {
                    op: LoweredOp::AttnDecode {
                        num_q_heads: hq,
                        num_kv_heads: hkv,
                        head_dim: hd,
                        scale,
                    },
                    m: 1,
                    inputs: vec![
                        InputRef::Op(4),
                        InputRef::Ext(7),
                        InputRef::Ext(8),
                        InputRef::Op(5),
                        InputRef::Op(3),
                    ],
                },
                OpDesc {
                    op: LoweredOp::Gemm { n: h },
                    m: 1,
                    inputs: vec![InputRef::Op(6), InputRef::Ext(9)],
                },
                OpDesc {
                    op: LoweredOp::Add,
                    m: 1,
                    inputs: vec![InputRef::Op(7), InputRef::Ext(0)],
                },
                OpDesc {
                    op: LoweredOp::RmsNorm { eps },
                    m: 1,
                    inputs: vec![InputRef::Op(8), InputRef::Ext(10)],
                },
                OpDesc {
                    op: LoweredOp::Gemm { n: i },
                    m: 1,
                    inputs: vec![InputRef::Op(9), InputRef::Ext(11)],
                },
                OpDesc {
                    op: LoweredOp::Silu,
                    m: 1,
                    inputs: vec![InputRef::Op(10)],
                },
                OpDesc {
                    op: LoweredOp::Gemm { n: i },
                    m: 1,
                    inputs: vec![InputRef::Op(9), InputRef::Ext(12)],
                },
                OpDesc {
                    op: LoweredOp::Mul,
                    m: 1,
                    inputs: vec![InputRef::Op(11), InputRef::Op(12)],
                },
                OpDesc {
                    op: LoweredOp::Gemm { n: h },
                    m: 1,
                    inputs: vec![InputRef::Op(13), InputRef::Ext(13)],
                },
                OpDesc {
                    op: LoweredOp::Add,
                    m: 1,
                    inputs: vec![InputRef::Op(14), InputRef::Op(8)],
                },
            ],
            result: 15,
        };
        let srcs: Vec<&[f32]> = vec![
            &res_in, &in_ln, &wq, &wk, &wv, &cos, &sin, &prefix_k, &prefix_v, &wo, &post_ln,
            &wgate, &wup, &wdown,
        ];

        for nb in [4u32, 8, 1000] {
            let g = lower_region::<TestShape2x4>(&input, std::num::NonZeroU32::new(nb).unwrap());
            assert!(validate(&g).is_ok(), "valid layer nb={nb}");
            let bufs = eval_dag(&g, &srcs);
            assert_eq!(
                result_buffer(&g, &bufs),
                &want[..],
                "decode layer nb={nb} bit-exact"
            );
        }
    }

    /// A consumer reading a whole N-block-tiled output depends on EVERY
    /// block; a reader of a leaf source has no dependency.
    #[test]
    fn tiled_elementwise_depends_on_matching_block() {
        let (m, n, k) = (1u32, 6u32, 8u32);
        let input = crate::lower::LoweringInput {
            sources: vec![
                SourceShape { rows: m, cols: k },
                SourceShape { rows: n, cols: k },
            ],
            ops: vec![
                OpDesc {
                    op: LoweredOp::Gemm { n },
                    m,
                    inputs: vec![InputRef::Ext(0), InputRef::Ext(1)],
                },
                OpDesc {
                    op: LoweredOp::Silu,
                    m,
                    inputs: vec![InputRef::Op(0)],
                },
            ],
            result: 1,
        };
        let g = lower_region::<TestShape2x4>(&input, std::num::NonZeroU32::new(2).unwrap());
        let preds = predecessors(&g);
        // 3 matmul blocks (0,1,2) + 3 silu tiles (3,4,5).
        assert_eq!(g.nodes.len(), 6);
        assert!(
            preds[0].is_empty() && preds[1].is_empty() && preds[2].is_empty(),
            "matmul blocks read only sources"
        );
        assert_eq!(preds[3], vec![SubtileId(0)], "silu tile 0 ← matmul block 0");
        assert_eq!(preds[4], vec![SubtileId(1)], "silu tile 1 ← matmul block 1");
        assert_eq!(preds[5], vec![SubtileId(2)], "silu tile 2 ← matmul block 2");
    }

    #[test]
    fn validate_rejects_uncovered_read() {
        let tensors = vec![
            TensorShape { rows: 1, cols: 4 }, // 0 source
            TensorShape { rows: 1, cols: 4 }, // 1 op output (never written)
        ];
        let bad: SubtileIR = SubtileIR {
            tensors,
            num_sources: 1,
            nodes: vec![SubtileNode {
                id: SubtileId(0),
                op: SubOp::Elementwise(EwKind::Silu),
                inputs: vec![TensorRegion {
                    tensor: TensorId(1), // reads op-output 1, which no prior node wrote
                    region: Region {
                        rows: Range::new(0, 1),
                        cols: Range::new(0, 4),
                    },
                }],
                output: TensorRegion {
                    tensor: TensorId(1),
                    region: Region {
                        rows: Range::new(0, 1),
                        cols: Range::new(0, 4),
                    },
                },
            }],
            result: TensorId(1),
        };
        assert!(validate(&bad).is_err(), "use-before-def must be rejected");
    }
}
