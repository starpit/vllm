// SPDX-License-Identifier: Apache-2.0
//! [`Implementation`] trait + supporting types.
//!
//! Ported from old ferrite-solver/src/lowering/implementation.rs.
//! Detoxifications:
//! - `TileGraph` → our [`Fuf`]; `TileId`/`TileKind` → our [`TileId`]
//!   and [`OpKind`]. No `weight_name: String` or `.layer` fields
//!   anywhere — tiles are numeric and structural.
//! - `Layout::PagedKvBf16` variant dropped (Llama-KV-specific name).
//!   `RowMajorBf16, ColMajorBf16, Any` kept; generic "paged-like"
//!   layouts can be added when a real non-llama impl needs them.
//! - `tiles_of_kind(&TileGraph, claimed, TileKind)` helper replaced
//!   by a version over our Fuf that filters by `OpKind`.

#![allow(dead_code)]

use std::collections::{BTreeMap, HashSet};
use std::fmt;

use proc_macro2::TokenStream;
use quote::quote;

use crate::classified::{ExternKind, OpKind, Program, WeightId};
use crate::emit::{EmitCtx, weight_field_name};
use crate::fuf::{Fuf, FufInput, FufNode, TileId};
use crate::quantization::StorageFormat;
use crate::shape::{Dim, Shape};
use crate::target::TargetProfile;

/// Ambient context passed to [`Implementation::cost_us`]. Carries
/// everything a cost function might need to turn a MatchInfo into
/// a wall-clock estimate: the FUF for shapes/deps, the target
/// profile for hardware characteristics, and the current
/// `bounds` (all config.json numeric fields plus `num_tokens` for
/// the current workload point).
pub struct CostCtx<'a> {
    pub fuf: &'a Fuf,
    pub profile: &'a TargetProfile,
    pub bounds: &'a BTreeMap<String, u64>,
}

impl CostCtx<'_> {
    /// Evaluate a `Dim` to a concrete integer using `bounds`.
    /// Returns `None` if the dim contains a Var (should not happen
    /// for the standard body after shape inference closes).
    pub fn eval_dim(&self, dim: &Dim) -> Option<u64> {
        match dim {
            Dim::Lit(n) => Some(*n),
            Dim::Bound(name) => self.bounds.get(name).copied(),
            Dim::Mul(cs) => cs
                .iter()
                .map(|c| self.eval_dim(c))
                .try_fold(1u64, |acc, v| v.map(|x| acc.saturating_mul(x))),
            Dim::Var(_) => None,
        }
    }

    pub fn eval_shape(&self, shape: &Shape) -> Option<Vec<u64>> {
        shape.iter().map(|d| self.eval_dim(d)).collect()
    }

    /// Convenience: read `num_tokens` from bounds. Panics if not
    /// set — the solver sets it per workload point before calling.
    pub fn num_tokens(&self) -> u64 {
        self.bounds
            .get("num_tokens")
            .copied()
            .expect("solver must set num_tokens in bounds before costing")
    }

    /// Convenience: read `sk_bucket` (the discretized KV-cache length
    /// axis) from bounds. Returns 0 when the caller has not swept an
    /// sk axis — legacy 1-D workload points. Impls that don't care
    /// about `sk` ignore this; impls whose cost depends on `sk`
    /// (e.g. FlashInfer attention) read it here and the solver's 2-D
    /// sweep produces one Assignment per (num_tokens, sk_bucket).
    pub fn sk_bucket(&self) -> u64 {
        self.bounds.get("sk_bucket").copied().unwrap_or(0)
    }
}

/// Stable identifier for one [`Implementation`] in the
/// [`ImplementationLibrary`]. Indices are dense in the library's
/// `entries` vector.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct ImplId(pub u32);

/// Per-implementation hardware resource demand.
///
/// **Per-CTA** demands (regs, shmem) determine occupancy and the
/// hard shmem-budget constraint. Two implementations grouped into
/// the same compilation unit (the same `__global__` on sm_89, the
/// same warpgroup on sm_90+ with `setmaxnreg`) **union** their
/// resource demands at the unit level.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Resources {
    /// Per-CTA dynamic shmem demand, in bytes.
    pub shmem_bytes: u32,
    /// Per-thread register footprint estimate.
    pub regs_per_thread: u32,
    /// CTA size this implementation runs with.
    pub threads_per_cta: u32,
}

impl Resources {
    pub const ZERO: Self = Self {
        shmem_bytes: 0,
        regs_per_thread: 0,
        threads_per_cta: 0,
    };

    /// Element-wise max — the resource budget two impls would need
    /// if they shared a compilation unit.
    pub fn union_max(&self, other: &Resources) -> Resources {
        Resources {
            shmem_bytes: self.shmem_bytes.max(other.shmem_bytes),
            regs_per_thread: self.regs_per_thread.max(other.regs_per_thread),
            threads_per_cta: self.threads_per_cta.max(other.threads_per_cta),
        }
    }
}

/// How an implementation is invoked / scheduled.
///
/// This is a hard constraint on what the implementation can share
/// a schedule slot with: a `HostCallback` cannot be co-resident
/// with other host calls in the same step; a `CooperativeLaunch`
/// is mutually exclusive with every other `CooperativeLaunch` on
/// the device; a `DeviceCallable` runs inside an enclosing
/// `__global__` and is constrained by that kernel's resource
/// budget.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum LaunchKind {
    /// Host calls a vendor library function (cuBLAS, cublasLt,
    /// FlashInfer standalone, vllm-rs fused-op kernels) which
    /// internally launches its own grid. The host launch boundary
    /// is the synchronization point.
    HostCallback,
    /// Host launches a normal `__global__` (non-cooperative) on a
    /// stream. Multiple of these can be in flight on different
    /// streams.
    RegularLaunch,
    /// Host launches a cooperative `__global__` that grabs all SMs
    /// and excludes other cooperative grids until it finishes.
    CooperativeLaunch,
    /// A `__device__` callable invoked from inside another
    /// `__global__`'s code. Contributes to the host kernel's
    /// resource budget; cannot be launched on its own.
    DeviceCallable,
}

/// Synchronization mechanism that conveys data between two
/// implementations on a producer→consumer dependency edge.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Handoff {
    /// Two host calls on the same stream — implicit ordering, no
    /// explicit sync.
    StreamOrder,
    /// `cudaEvent` recorded after the producer + waited on before
    /// the consumer.
    StreamEvent,
    /// Kernel boundary — the next `cudaLaunchKernel` waits for
    /// the previous to complete.
    KernelBoundary,
    /// In-kernel `cooperative_groups::this_grid().sync()` or
    /// gmem-flag spin barrier.
    InKernelGridSync,
    /// shmem `mbarrier` — sm_90+ only. Used between warpgroups
    /// inside one persistent kernel.
    Mbarrier,
    /// Distributed-shmem cluster read — sm_90+ with thread-block
    /// clusters.
    DsmemRead,
    /// gmem flag spin (per-tile counter).
    GmemFlag,
    /// Intra-CTA `__syncthreads()` between DeviceCallable ops in
    /// the same persistent kernel.
    SyncThreads,
    /// No handoff — both impls are claimed by the same subgraph
    /// (the implementation handles the dep internally).
    Internal,
}

impl Handoff {
    /// Wall-clock cost in microseconds for this handoff on the
    /// given target.
    pub fn cost_us(&self, profile: &TargetProfile) -> f64 {
        let _ = profile; // target-specific cost tables land later;
        // for now fall back to empirical-or-universal constants.
        match self {
            Handoff::StreamOrder => 0.0,
            Handoff::StreamEvent => 5.0,
            Handoff::KernelBoundary => 5.0,
            Handoff::InKernelGridSync => 100.0,
            Handoff::Mbarrier => 0.1,
            Handoff::DsmemRead => 1.0,
            Handoff::GmemFlag => 0.5,
            Handoff::SyncThreads => 0.5,
            Handoff::Internal => 0.0,
        }
    }

    /// Whether this handoff truncates the intermediate to storage
    /// dtype. GMEM-transiting handoffs truncate (value is written
    /// then re-read in storage dtype). In-kernel handoffs
    /// (Internal, Mbarrier, shmem reads) preserve accumulator
    /// precision.
    pub fn truncates_to_storage_dtype(&self) -> bool {
        match self {
            Handoff::StreamOrder
            | Handoff::StreamEvent
            | Handoff::KernelBoundary
            | Handoff::InKernelGridSync
            | Handoff::GmemFlag
            | Handoff::SyncThreads => true,
            Handoff::Internal | Handoff::Mbarrier | Handoff::DsmemRead => false,
        }
    }
}

/// Layout of a tile in memory. Used by the layout-compatibility
/// constraint: a producer's output layout must equal the
/// consumer's input layout, OR a layout-conversion implementation
/// must be inserted between them.
///
/// Starts small; extended as new impls enter the library. Paged
/// / swizzled / tiled variants are added structurally (parameters
/// carrying block sizes etc.) not by naming the transformer role
/// they serve.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Layout {
    /// `[outer, inner]` row-major contiguous bf16.
    RowMajorBf16,
    /// `[inner, outer]` column-major bf16.
    ColMajorBf16,
    /// "Don't care" — the producer and consumer don't constrain
    /// the layout.
    Any,
}

impl Layout {
    /// Whether two layouts are interchangeable without an explicit
    /// conversion. `Any` matches anything.
    pub fn is_compatible_with(self, other: Layout) -> bool {
        match (self, other) {
            (Layout::Any, _) | (_, Layout::Any) => true,
            (a, b) => a == b,
        }
    }
}

/// What an [`Implementation::matches`] call returns when an impl
/// can claim a particular subgraph. Carries the per-claim metadata
/// the solver / cost model / backend need.
#[derive(Clone, Debug)]
pub struct MatchInfo {
    /// The tile ids this implementation would claim if chosen.
    /// Must be a connected subgraph of the [`Fuf`].
    pub claimed_tiles: Vec<TileId>,
    /// The boundary-input tiles this claim reads from (i.e. tiles
    /// outside `claimed_tiles` whose outputs flow in). Used by
    /// the solver to wire up handoffs.
    pub boundary_inputs: Vec<TileId>,
    /// The boundary-output tiles this claim writes to (i.e. tiles
    /// inside `claimed_tiles` whose outputs are read by tiles
    /// outside the claim).
    pub boundary_outputs: Vec<TileId>,
}

impl MatchInfo {
    pub fn size(&self) -> usize {
        self.claimed_tiles.len()
    }
    pub fn is_singleton(&self) -> bool {
        self.claimed_tiles.len() == 1
    }
}

/// Declarative workload eligibility for an implementation.
///
/// Distinct from `target_compatible` (about GPU capability). A
/// `WorkloadConstraint` expresses correctness requirements on the
/// workload itself — e.g. "this GEMV kernel only handles M=1", or
/// "this FlashInfer decode kernel wins only when the KV-cache span
/// (`sk`) is large".
///
/// Correctness, not cost: if `accepts(num_tokens, sk_bucket)` returns
/// false, the impl must NOT be picked at that workload regardless of
/// its cost. Data, not closures — so the ILP backend can linearize
/// each variant.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum WorkloadConstraint {
    /// Valid for any `(num_tokens, sk_bucket)` value.
    Any,
    /// Valid only when `num_tokens` falls in this inclusive range;
    /// unconstrained on `sk_bucket`.
    NumTokensRange { min: u32, max: u32 },
    /// Valid only when BOTH `num_tokens` and `sk_bucket` fall in
    /// their respective inclusive ranges.
    NumTokensAndSkRange {
        num_tokens: (u32, u32),
        sk_bucket: (u64, u64),
    },
}

impl WorkloadConstraint {
    pub fn accepts(&self, num_tokens: u32, sk_bucket: u64) -> bool {
        match self {
            Self::Any => true,
            Self::NumTokensRange { min, max } => num_tokens >= *min && num_tokens <= *max,
            Self::NumTokensAndSkRange {
                num_tokens: (m_min, m_max),
                sk_bucket: (sk_min, sk_max),
            } => {
                num_tokens >= *m_min
                    && num_tokens <= *m_max
                    && sk_bucket >= *sk_min
                    && sk_bucket <= *sk_max
            }
        }
    }
}

/// A method the emitted `WeightBundle` trait must expose, as
/// declared by an [`Implementation`]. Codegen aggregates these
/// declarations across every subgraph of an SFUF and emits one
/// trait method per unique accessor.
///
/// Why this lives on the Impl: a fusion impl that claims multiple
/// Gemm tiles may want the user to pre-concatenate the weights
/// into one packed buffer, so its `emit_call` can issue a single
/// cuBLAS call. Declaring a fused accessor (e.g. `mlp_gate_up_0`
/// returning one `LinearLayer`) is how the impl expresses that
/// contract to the user, without the compiler knowing anything
/// about "gate" or "up" specifically.
#[derive(Clone)]
pub struct WeightAccessor {
    /// Trait method name. Two impls that declare the same name
    /// must agree on `rust_type`; a mismatch is a hard error at
    /// trait-emission time.
    pub name: syn::Ident,
    /// Return type, as emitted Rust tokens (e.g.
    /// `::ferrite_kernels::layers::LinearLayer`).
    pub rust_type: TokenStream,
    /// DSL weights that feed this accessor. One pair for a simple
    /// accessor, multiple for a fused one. Used for dedup and
    /// documentation; the user is responsible for providing a
    /// weight object with the declared `rust_type` that stands in
    /// for the listed sources.
    pub source_weights: Vec<(WeightId, Option<u64>)>,
}

impl fmt::Debug for WeightAccessor {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("WeightAccessor")
            .field("name", &self.name.to_string())
            .field("rust_type", &self.rust_type.to_string())
            .field("source_weights", &self.source_weights)
            .finish()
    }
}

/// Default [`Implementation::required_weights`] body: one accessor
/// per unique `(WeightId, index)` referenced by a weight input of
/// any claimed tile. Name via [`weight_field_name`]; type inferred
/// from the consuming op (Embed → `Embedding`, RmsNorm → `RmsNorm`,
/// Gemm → `LinearLayer`). Preserves the single-tile-per-subgraph
/// accessor shape the FUF-walking WeightBundle emission used
/// before the trait moved to SFUF-walking.
pub fn default_required_weights(
    claimed_tiles: &[TileId],
    fuf: &Fuf,
    program: &Program,
) -> Vec<WeightAccessor> {
    let mut out: Vec<WeightAccessor> = Vec::new();
    let mut seen: HashSet<String> = HashSet::new();
    for &tid in claimed_tiles {
        let node = fuf.get(tid);
        for input in &node.inputs {
            if let FufInput::Weight { id, index, .. } = input {
                let name = weight_field_name(program, *id, *index);
                if !seen.insert(name.to_string()) {
                    continue;
                }
                out.push(WeightAccessor {
                    name,
                    rust_type: rust_type_for_weight_consumed_by(node.op),
                    source_weights: vec![(*id, *index)],
                });
            }
        }
    }
    out
}

/// The ferrite-kernels layer wrapper corresponding to an op that
/// consumes a weight. Drives the default accessor-type decision.
/// Fusion impls that pack multiple weights declare their own type
/// explicitly and bypass this.
pub fn rust_type_for_weight_consumed_by(op: OpKind) -> TokenStream {
    match op {
        OpKind::Embed => quote! { ::ferrite_kernels::layers::Embedding },
        OpKind::RmsNorm => quote! { ::ferrite_kernels::layers::RmsNorm },
        // Cohere-flavored full LayerNorm (weight only, no bias). The
        // `eps` rides on the wrapper struct, same shape as RmsNorm.
        OpKind::LayerNorm => quote! { ::ferrite_kernels::layers::CohereLayerNorm },
        OpKind::Gemm => quote! { ::ferrite_kernels::layers::LinearLayer },
        // Ops that don't consume weights in the DSL's typical
        // patterns. If the DSL routes a weight into one of these
        // unexpectedly, the user must provide a raw `GpuTensor`.
        _ => quote! { ::ferrite_cuda_core::tensor::GpuTensor },
    }
}

/// One curated implementation in the library.
///
/// Implementations are **not** generic — each entry corresponds
/// to a specific kernel from a specific source (cuBLAS, CUTLASS
/// sm_80 multistage, ThunderKittens fused gate-up, FlashInfer
/// standalone FA-2, vllm-rs `fused_add_rms_norm_inplace`, …).
///
/// Each impl declares the subgraph patterns it can match, its
/// resource demands, its launch kind, the handoff mechanisms it
/// supports, the layouts it requires, and its target
/// compatibility. Cost is calibrated from microbench data per
/// shape.
pub trait Implementation: fmt::Debug + Send + Sync {
    /// Stable name for debug / display / cost-table keys.
    fn name(&self) -> &'static str;

    /// Whether this implementation can run on the given target.
    fn target_compatible(&self, profile: &TargetProfile) -> bool;

    /// Workload eligibility. Default accepts any `num_tokens`.
    fn workload_constraint(&self) -> WorkloadConstraint {
        WorkloadConstraint::Any
    }

    /// Try to match a subgraph rooted at the given seed tile.
    /// Returns `Some(MatchInfo)` if this implementation can claim
    /// a subgraph that includes `seed`, `None` otherwise.
    ///
    /// The matcher is procedural Rust: it inspects `fuf`, walks
    /// neighbors of `seed`, and decides whether the local structure
    /// matches the impl's expected pattern. Quant-aware impls gate
    /// on weight storage via [`Fuf::storage_format_of`].
    fn matches(&self, fuf: &Fuf, seed: TileId, profile: &TargetProfile) -> Option<MatchInfo>;

    /// Predicted wall-clock cost in microseconds for one
    /// invocation of this impl on the given match.
    fn cost_us(&self, m: &MatchInfo, ctx: &CostCtx) -> f64;

    /// Per-CTA resource demand of this implementation.
    fn resources(&self, m: &MatchInfo) -> Resources;

    /// How this implementation is launched / scheduled.
    fn launch_kind(&self) -> LaunchKind;

    /// The handoff mechanisms this impl can use to **receive** its
    /// boundary inputs from upstream.
    fn supported_input_handoffs(&self) -> &[Handoff];

    /// The handoff mechanisms this impl can use to **convey** its
    /// boundary outputs to downstream.
    fn supported_output_handoffs(&self) -> &[Handoff];

    /// Required input layouts, parallel to `match.boundary_inputs`.
    fn input_layouts(&self, m: &MatchInfo) -> Vec<Layout>;

    /// Output layouts, parallel to `match.boundary_outputs`.
    fn output_layouts(&self, m: &MatchInfo) -> Vec<Layout>;

    /// Whether this impl is compute-bound (for concurrency
    /// modeling). Two compute-bound impls on the same target's
    /// tensor cores cannot productively overlap; memory-bound
    /// impls can.
    fn is_compute_bound(&self) -> bool {
        false
    }

    /// Whether this impl can be claimed by the same compilation
    /// unit / kernel as another. Default: never share
    /// (HostCallback / RegularLaunch / CooperativeLaunch). Device
    /// callables override to opt in.
    fn can_share_kernel_with(&self, _other: &dyn Implementation) -> bool {
        false
    }

    /// Emit the Rust token stream that invokes this impl's kernel
    /// for the given subgraph. Called by codegen once per subgraph
    /// after the solver has bound the subgraph to this impl.
    ///
    /// The emitted tokens must:
    /// - Read inputs via [`EmitCtx::input_expr`] or
    ///   [`EmitCtx::all_input_exprs`] for the relevant claimed tile.
    /// - Produce a `let` binding for every output slot of every
    ///   claimed tile, using [`EmitCtx::output_ident`] as the ident.
    /// - Assume ambient `wm: &impl WeightBundle`, `ctx: &ForwardCtx`,
    ///   and `device: &mut GpuDevice` bindings are in scope.
    ///
    /// Default implementation is a `compile_error!` so forgetting
    /// to implement it fails loudly at macro expansion.
    fn emit_call(&self, _ctx: &EmitCtx) -> TokenStream {
        let name = self.name();
        let msg = format!("Implementation `{name}` has no emit_call body");
        quote! { compile_error!(#msg); }
    }

    /// Declare the `WeightBundle` trait methods this impl's
    /// `emit_call` will invoke for the given claim. Codegen
    /// aggregates declarations across all picked impls in an SFUF
    /// and emits one trait method per unique accessor.
    ///
    /// Default: one accessor per weight input of each claimed
    /// tile (see [`default_required_weights`]). Fusion impls that
    /// want a packed weight (e.g. concatenated `gate_proj|up_proj`)
    /// override this to declare a single accessor whose
    /// `source_weights` lists every DSL weight the user must pack
    /// into the object they return.
    fn required_weights(
        &self,
        claimed_tiles: &[TileId],
        fuf: &Fuf,
        program: &Program,
    ) -> Vec<WeightAccessor> {
        default_required_weights(claimed_tiles, fuf, program)
    }

    /// Per-output alias declaration. For each tile-output this impl
    /// binds, return one entry:
    /// - `((tile, slot), None)` — output is its own freshly-allocated
    ///   `OwnedTensor` (the codegen drop pass will free it after its
    ///   last cross-subgraph consumer).
    /// - `((tile, slot), Some((src_tile, src_slot)))` — output is a
    ///   `TensorView` aliasing the upstream `OwnedTensor` at
    ///   `(src_tile, src_slot)`. Consumers of this output count as
    ///   uses of the underlying source.
    ///
    /// Outputs absent from the returned vec are "untracked" — e.g.
    /// paged-cache views like `kv_cache.k_cache(layer)` whose memory
    /// is owned by the cache pool, not the caching allocator.
    ///
    /// Default: every output of every claimed tile is `None`
    /// (its own `OwnedTensor`).
    #[allow(clippy::type_complexity)]
    fn output_alias(
        &self,
        claimed_tiles: &[TileId],
        fuf: &Fuf,
    ) -> Vec<((TileId, u8), Option<(TileId, u8)>)> {
        claimed_tiles
            .iter()
            .flat_map(|&t| {
                let n = fuf.get(t).outputs.len().max(1);
                (0..n as u8).map(move |s| ((t, s), None))
            })
            .collect()
    }

    /// Upstream tile-output `(tile, slot)`s that this impl's
    /// `emit_call` *moves* (consumes) into one of its own output
    /// bindings. After such a subgraph, the upstream local is no
    /// longer accessible — the codegen drop pass must not schedule
    /// a `drop(...)` for it.
    ///
    /// Distinct from `output_alias`: an alias keeps the upstream
    /// alive and shares its memory; a consume transfers ownership.
    /// In-place kernels like `scale_inplace` (ScalarMul) and
    /// `tanh_softcap_inplace` (TanhSoftCap) follow the consume
    /// pattern — the kernel mutates the buffer, and the impl
    /// rebinds the moved `OwnedTensor` as its output.
    ///
    /// Default: empty (no upstream is consumed).
    fn consumes_input_tiles(&self, _claimed_tiles: &[TileId], _fuf: &Fuf) -> Vec<(TileId, u8)> {
        Vec::new()
    }
}

/// The library: all available implementations for some target.
/// Implementations are boxed trait objects; the solver iterates
/// them, calling `matches` at each FUF tile.
#[derive(Default)]
pub struct ImplementationLibrary {
    entries: Vec<Box<dyn Implementation>>,
}

impl fmt::Debug for ImplementationLibrary {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ImplementationLibrary")
            .field("len", &self.entries.len())
            .finish()
    }
}

impl ImplementationLibrary {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn push(&mut self, imp: Box<dyn Implementation>) -> ImplId {
        let id = ImplId(self.entries.len() as u32);
        self.entries.push(imp);
        id
    }

    pub fn get(&self, id: ImplId) -> &dyn Implementation {
        &*self.entries[id.0 as usize]
    }

    pub fn len(&self) -> usize {
        self.entries.len()
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Iterate all impls with their ids. The solver uses this to
    /// enumerate candidates at each seed tile.
    pub fn iter_enumerated(&self) -> impl Iterator<Item = (ImplId, &dyn Implementation)> + '_ {
        self.entries
            .iter()
            .enumerate()
            .map(|(i, imp)| (ImplId(i as u32), imp.as_ref()))
    }
}

/// Helper used by the solver to seed-enumerate matches across the
/// whole library at a given tile. Returns every (impl_id,
/// MatchInfo) pair that can claim something including `seed`.
pub fn enumerate_matches_at_seed(
    library: &ImplementationLibrary,
    fuf: &Fuf,
    seed: TileId,
    profile: &TargetProfile,
) -> Vec<(ImplId, MatchInfo)> {
    let mut out = Vec::new();
    for (id, imp) in library.iter_enumerated() {
        if !imp.target_compatible(profile) {
            continue;
        }
        if let Some(m) = imp.matches(fuf, seed, profile) {
            out.push((id, m));
        }
    }
    out
}

/// Convenience: filter `claimed_tiles` to those of a given op
/// kind. Used by cost / layout helpers that need to count how
/// many of a claim's tiles are (for instance) gemms.
pub fn tiles_of_op<'a>(
    fuf: &'a Fuf,
    claimed: &'a [TileId],
    op: OpKind,
) -> impl Iterator<Item = TileId> + 'a {
    claimed
        .iter()
        .copied()
        .filter(move |&tid| fuf.get(tid).op == op)
}

// ── Starter library ──────────────────────────────────────────────
//
// Baseline HostCallback impl per OpKind. One trait-object per op,
// each matching that op structurally and emitting an analytical
// cost estimate. Real target-specific / quantized / fused impls
// land as additional entries that the solver picks over these
// when cheaper.

const BYTES_PER_ELEM: f64 = 2.0; // fp16/bf16 storage

fn shape_elems(ctx: &CostCtx, shape: &Shape) -> u64 {
    ctx.eval_shape(shape)
        .map(|v| v.iter().product::<u64>())
        .unwrap_or(0)
}

fn single_tile_match(fuf: &Fuf, seed: TileId, op: OpKind) -> Option<MatchInfo> {
    if fuf.get(seed).op != op {
        return None;
    }
    let node = fuf.get(seed);
    // Boundary-input tiles: upstream tiles whose outputs this tile reads.
    let boundary_inputs: Vec<TileId> = node
        .inputs
        .iter()
        .filter_map(|inp| {
            if let crate::fuf::FufInput::Tile { id, .. } = inp {
                Some(*id)
            } else {
                None
            }
        })
        .collect();
    Some(MatchInfo {
        claimed_tiles: vec![seed],
        boundary_inputs,
        boundary_outputs: vec![seed],
    })
}

fn elementwise_cost(m: &MatchInfo, ctx: &CostCtx) -> f64 {
    let node = ctx.fuf.get(m.claimed_tiles[0]);
    let mut bytes = 0u64;
    for inp in &node.inputs {
        if let crate::fuf::FufInput::Tile { id, slot } = inp {
            let upstream = ctx.fuf.get(*id);
            if let Some(shape) = upstream.outputs.get(*slot as usize) {
                bytes = bytes.saturating_add(shape_elems(ctx, shape));
            }
        }
    }
    for out in &node.outputs {
        bytes = bytes.saturating_add(shape_elems(ctx, out));
    }
    let bytes = bytes.saturating_mul(BYTES_PER_ELEM as u64);
    let gb_per_sec = ctx.profile.memory_bandwidth_gbps;
    (bytes as f64 / (gb_per_sec * 1e9)) * 1e6
}

macro_rules! trivial_impl {
    ($name:ident, $op:expr, $kernel_name:literal, $cost:expr, $emit:expr, $compute_bound:expr) => {
        #[derive(Debug, Default)]
        pub struct $name;

        impl Implementation for $name {
            fn name(&self) -> &'static str {
                $kernel_name
            }
            fn target_compatible(&self, _profile: &TargetProfile) -> bool {
                true
            }
            fn matches(
                &self,
                fuf: &Fuf,
                seed: TileId,
                _profile: &TargetProfile,
            ) -> Option<MatchInfo> {
                let info = single_tile_match(fuf, seed, $op)?;
                // Reference impls consume the weight as a dense
                // `LinearLayer` / `Embedding` / `RmsNorm`. A non-Dense
                // weight input means a quant-aware impl must cover
                // this tile — bail so the solver doesn't pick a dense
                // kernel on AWQ bits.
                if let Some(s) = weight_storage_of(fuf.get(seed))
                    && !matches!(s, StorageFormat::Dense)
                {
                    return None;
                }
                Some(info)
            }
            fn cost_us(&self, m: &MatchInfo, ctx: &CostCtx) -> f64 {
                ($cost)(m, ctx)
            }
            fn resources(&self, _m: &MatchInfo) -> Resources {
                // HostCallback kernels don't contribute to any
                // compilation unit's budget.
                Resources::ZERO
            }
            fn launch_kind(&self) -> LaunchKind {
                LaunchKind::HostCallback
            }
            fn supported_input_handoffs(&self) -> &[Handoff] {
                const H: &[Handoff] = &[Handoff::StreamOrder, Handoff::StreamEvent];
                H
            }
            fn supported_output_handoffs(&self) -> &[Handoff] {
                const H: &[Handoff] = &[Handoff::StreamOrder, Handoff::StreamEvent];
                H
            }
            fn input_layouts(&self, m: &MatchInfo) -> Vec<Layout> {
                vec![Layout::RowMajorBf16; m.boundary_inputs.len()]
            }
            fn output_layouts(&self, m: &MatchInfo) -> Vec<Layout> {
                vec![Layout::RowMajorBf16; m.boundary_outputs.len()]
            }
            fn is_compute_bound(&self) -> bool {
                $compute_bound
            }
            fn emit_call(&self, ctx: &EmitCtx) -> TokenStream {
                ($emit)(ctx)
            }
        }
    };
}

// ── Per-op emission bodies ───────────────────────────────────────
//
// These preserve the shape of the previous hardcoded OpKind match
// in codegen.rs. The kernel symbols they reference are approximate
// and cuda-gated; real ferrite-kernels bindings land as calibrated
// impls replace these reference entries.

fn emit_embed(ctx: &EmitCtx) -> TokenStream {
    // kernels::embedding_gather(weight: GpuTensor, input_ids:
    // GpuTensor, alloc, stream) -> OwnedTensor. Weight comes from
    // the bundle as `&Embedding` (struct with .weight: GpuTensor);
    // input_ids comes from ctx as TensorView, deref'd to GpuTensor
    // via the Copy impl on GpuTensor.
    let tile = ctx.primary();
    let out = ctx.output_ident(tile, 0);
    let ids = ctx.input_expr(tile, 0);
    let weight = ctx.input_expr(tile, 1);
    quote! {
        let #out = unsafe {
            ::ferrite_kernels::kernels::embedding_gather(
                (#weight).weight,
                *(#ids),
                &mut device.caching,
                device.compute_stream,
            )
        };
    }
}

fn emit_rmsnorm(ctx: &EmitCtx) -> TokenStream {
    // kernels::rms_norm(input, weight, eps, weight_offset, alloc, stream) -> OwnedTensor.
    // Singleton RmsNorm tiles pass weight_offset=0.0 (Llama/Qwen2 math).
    // Scalar-offset rmsnorms like Gemma's `(1+w)` flow through a
    // distinct Impl that consumes the upstream Add(weight, scalar)
    // tile and passes the scalar as weight_offset.
    //
    // Input is a TensorView borrowed via input_expr; the codegen-level
    // drop pass frees the upstream OwnedTensor after this subgraph
    // (or after a later subgraph if the upstream has further consumers).
    let tile = ctx.primary();
    let out = ctx.output_ident(tile, 0);
    let x = ctx.input_expr(tile, 0);
    let w = ctx.input_expr(tile, 1);
    quote! {
        let #out = unsafe {
            ::ferrite_kernels::kernels::rms_norm(
                *(#x),
                (#w).weight,
                (#w).eps,
                &mut device.caching,
                device.compute_stream,
            )
        };
    }
}

fn emit_layernorm(ctx: &EmitCtx) -> TokenStream {
    // kernels::cohere_layer_norm(input, weight, eps, alloc, stream) -> OwnedTensor.
    // Cohere's LayerNorm subtracts the mean (unlike RmsNorm) and has
    // weight only (no bias). Same call shape as `emit_rmsnorm`,
    // different kernel.
    let tile = ctx.primary();
    let out = ctx.output_ident(tile, 0);
    let x = ctx.input_expr(tile, 0);
    let w = ctx.input_expr(tile, 1);
    quote! {
        let #out = unsafe {
            ::ferrite_kernels::kernels::cohere_layer_norm(
                *(#x),
                (#w).weight,
                (#w).eps,
                &mut device.caching,
                device.compute_stream,
            )
        };
    }
}

fn emit_gemm(ctx: &EmitCtx) -> TokenStream {
    // `gemm()` in the DSL is strict matmul. Bias is a separate
    // `bias_add` tile and is claimed by its own Impl (e.g.
    // `FusedGemmBiasImpl` fuses an adjacent `(Gemm, BiasAdd)` into
    // cuBLAS's gemm_bias epilog). If the user passes a `LinearLayer`
    // that carries a bias but the DSL doesn't say `bias_add`, the
    // bias is silently ignored here — the stray bias_add, if it
    // exists, surfaces elsewhere as `UnclaimedTile`. This mirrors
    // cutlass's `.dense_weight()` emit path.
    let tile = ctx.primary();
    let out = ctx.output_ident(tile, 0);
    let x = ctx.input_expr(tile, 0);
    let w = ctx.input_expr(tile, 1);
    quote! {
        let #out = unsafe {
            device.cublas.gemm(*(#x), (#w).dense_weight(), &mut device.caching)
        };
    }
}

// No standalone `emit_rope_append`. The only ferrite-kernels path for
// "apply rotary + write KV to paged cache" is `fused_qkv_rope_cache`,
// which expects the QKV projections already fused into one packed
// tensor. `FusedQkvRopeCacheImpl` below claims the whole 4-tile
// pattern `(Gemm, Gemm, Gemm, RopeAppend)` structurally — so every
// DSL `rope_append` in Llama/Qwen2 is covered.

// No standalone `emit_attention`. Paged-cache attention
// (`flash_attn_paged`) takes Q plus cache metadata and reads K/V
// directly from the KV cache populated by `FusedQkvRopeCacheImpl`.
// `AttentionViaCacheImpl` below claims the singleton `OpKind::Attention`
// tile and ignores its DSL-visible K/V inputs entirely.

// No standalone `emit_add`: every `Add` in Llama/Qwen2 is immediately
// consumed by an `RmsNorm`, so `FusedAddRmsNormImpl` below claims
// the pair and maps to `fused_add_rms_norm_inplace`. A lone `Add`
// with no downstream `RmsNorm` surfaces as `SolveError::UnclaimedTile`,
// not as a silent fallback.

fn cost_embed(m: &MatchInfo, ctx: &CostCtx) -> f64 {
    // One gather per output element, bandwidth-bound.
    elementwise_cost(m, ctx)
}

fn cost_gemm(m: &MatchInfo, ctx: &CostCtx) -> f64 {
    // 2*M*N*K FLOPs, compute-bound on tensor cores.
    let node = ctx.fuf.get(m.claimed_tiles[0]);
    let inputs: Vec<_> = node
        .inputs
        .iter()
        .filter_map(|i| match i {
            crate::fuf::FufInput::Tile { id, slot } => ctx
                .fuf
                .get(*id)
                .outputs
                .get(*slot as usize)
                .and_then(|s| ctx.eval_shape(s)),
            crate::fuf::FufInput::Weight { id, .. } => {
                // Weight shape not threaded through FUF yet; skip —
                // callers that need accuracy should use a real
                // calibrated impl.
                let _ = id;
                None
            }
            _ => None,
        })
        .collect();
    // Input[0] is the activation: [.., K].
    let (m_dim, k_dim) = match inputs.first() {
        Some(dims) if dims.len() >= 2 => {
            let k = *dims.last().unwrap();
            let m = dims[..dims.len() - 1].iter().product::<u64>();
            (m, k)
        }
        _ => (ctx.num_tokens(), 0),
    };
    // Output last dim is N.
    let n_dim = node
        .outputs
        .first()
        .and_then(|s| ctx.eval_shape(s))
        .and_then(|v| v.last().copied())
        .unwrap_or(0);
    // Prefer the calibrated CSV baseline when the target has one —
    // keeps GemmRefImpl (cuBLAS) on the same measurement scale as
    // the CutlassGemmImpl variants so the DP's cost comparison is
    // apples-to-apples.
    if let Some(cost) = ctx
        .profile
        .cost_us_for("cublas", m_dim as u32, n_dim as u32, k_dim as u32)
    {
        return cost;
    }
    let flops = 2.0 * m_dim as f64 * n_dim as f64 * k_dim as f64;
    let peak = ctx.profile.peak_tflops_fp16 * 1e12;
    if peak == 0.0 || flops == 0.0 {
        0.0
    } else {
        (flops / peak) * 1e6
    }
}

/// Softmax scale for one attention call. Priority order:
///   1. Granite's `attention_multiplier` — a direct override (no
///      transform); the HF config already stores the final scale.
///   2. Gemma2's `query_pre_attn_scalar` — convention is
///      `scale = query_pre_attn_scalar.powf(-0.5)`.
///   3. Fallback `1 / sqrt(head_dim)` for Llama / Qwen2 / Qwen3.
pub(crate) fn attention_scale_for(model: &crate::config::ModelParams) -> f32 {
    if let Some(s) = model.scalars.get("attention_multiplier") {
        return *s as f32;
    }
    match model.scalars.get("query_pre_attn_scalar") {
        Some(q) => (*q as f32).powf(-0.5),
        None => {
            let head_dim_f = *model
                .bounds
                .get("head_dim")
                .expect("attention emit: head_dim missing from model config")
                as f32;
            1.0 / head_dim_f.sqrt()
        }
    }
}

/// Attention logit soft-cap. Reads `attn_logit_softcapping` from
/// the model config when present (Gemma2), else `0.0` meaning the
/// kernel does no soft-capping. The flash-attn kernel treats
/// `softcap <= 0` as "disabled."
pub(crate) fn attention_softcap_for(model: &crate::config::ModelParams) -> f32 {
    model
        .scalars
        .get("attn_logit_softcapping")
        .copied()
        .unwrap_or(0.0) as f32
}

fn attention_scale_tokens(ctx: &EmitCtx) -> TokenStream {
    let scale = attention_scale_for(ctx.model);
    quote! { #scale }
}

fn attention_softcap_tokens(ctx: &EmitCtx) -> TokenStream {
    let cap = attention_softcap_for(ctx.model);
    quote! { #cap }
}

/// `window_size_left` argument for sliding-window attention. Reads
/// `sliding_window` from the model config (HF convention).
/// The flash-attn kernel takes `-1` to mean "disabled" and a
/// non-negative `w` to mean "attend to the last `w` tokens"; every
/// sliding architecture we know of carries `sliding_window` in
/// tokens, so the value flows straight through.
///
/// A SlidingAttention tile that reaches this helper on a model with
/// no `sliding_window` in config is a config error — the DSL body
/// asked for sliding attention but the architecture didn't supply
/// the window size. Panic with the model name so the failure is
/// observable at macro expansion.
pub(crate) fn sliding_window_left_for(model: &crate::config::ModelParams) -> i32 {
    match model.bounds.get("sliding_window") {
        Some(w) => (*w) as i32,
        None => panic!(
            "sliding_attention tile emitted, but model `{}` has no \
             `sliding_window` in its config.json",
            model.source_stem,
        ),
    }
}

fn sliding_window_left_tokens(ctx: &EmitCtx) -> TokenStream {
    let w = sliding_window_left_for(ctx.model);
    quote! { #w }
}

/// Build the `fa2_attn_bf16_h{h}` kernel name for calibrated FA2 cost
/// lookup. FA2 handles any `(num_qo_heads, num_kv_heads)` at runtime,
/// so the cost table is keyed only on the compile-time specialization
/// dim (`head_dim`). Keep in sync with the row names emitted by
/// `ferrite-cost-sweep/src/attention_sweep.rs`.
fn fa2_attn_csv_name(head_dim: u32) -> String {
    format!("fa2_attn_bf16_h{head_dim}")
}

/// Calibrated FA2 attention cost (via CSV), falling back to the
/// analytic `cost_attention` when no row matches. Used by
/// `AttentionViaCacheImpl` / `AttentionPrefillContiguousImpl` (and
/// their sliding variants) so the solver's FA2 vs FI tiebreak runs on
/// real measured timings instead of the ~0 µs the analytic formula
/// returns at M=1 (where `flops = 4*1*1*d` underflows the TFLOPS
/// budget). Without this, FI can never beat FA2 at decode even when
/// the calibrated CSV shows it's 4.6× faster.
fn cost_attention_calibrated(m: &MatchInfo, ctx: &CostCtx) -> f64 {
    let Some(head_dim) = ctx.bounds.get("head_dim").copied() else {
        return cost_attention(m, ctx);
    };
    let name = fa2_attn_csv_name(head_dim as u32);
    let nt = ctx.num_tokens() as u32;
    let sk = ctx.sk_bucket() as u32;
    match ctx.profile.cost_us_for(&name, nt, sk, head_dim as u32) {
        Some(cost) => cost,
        None => cost_attention(m, ctx),
    }
}

fn cost_attention(m: &MatchInfo, ctx: &CostCtx) -> f64 {
    // Rough: 4 * T^2 * D.
    let node = ctx.fuf.get(m.claimed_tiles[0]);
    let q_shape = node.inputs.first().and_then(|i| match i {
        crate::fuf::FufInput::Tile { id, slot } => ctx
            .fuf
            .get(*id)
            .outputs
            .get(*slot as usize)
            .and_then(|s| ctx.eval_shape(s)),
        _ => None,
    });
    let (t, d) = match q_shape {
        Some(dims) if dims.len() >= 2 => {
            (*dims.first().unwrap(), dims[1..].iter().product::<u64>())
        }
        _ => (ctx.num_tokens(), 0),
    };
    let flops = 4.0 * t as f64 * t as f64 * d as f64;
    let peak = ctx.profile.peak_tflops_fp16 * 1e12;
    if peak == 0.0 || flops == 0.0 {
        0.0
    } else {
        (flops / peak) * 1e6
    }
}

trivial_impl!(
    EmbedRefImpl,
    OpKind::Embed,
    "embed_ref",
    cost_embed,
    emit_embed,
    false
);
trivial_impl!(
    RmsNormRefImpl,
    OpKind::RmsNorm,
    "rmsnorm_ref",
    elementwise_cost,
    emit_rmsnorm,
    false
);
trivial_impl!(
    LayerNormRefImpl,
    OpKind::LayerNorm,
    "layer_norm_ref",
    elementwise_cost,
    emit_layernorm,
    false
);
trivial_impl!(
    GemmRefImpl,
    OpKind::Gemm,
    "gemm_ref",
    cost_gemm,
    emit_gemm,
    true
);

// ── ReshapeRefImpl ───────────────────────────────────────────────
//
// Claims `OpKind::Reshape` tiles — metadata-only view changes
// synthesized by shape inference to bridge axis-factor mismatches
// (e.g. per-head QK-norm: rmsnorm on `[T, heads, head_dim]` where
// upstream produced `[T, heads * head_dim]`). Emits a single
// `TensorView::reshape(&[d0, d1, ...])` with concrete dims evaluated
// from the tile's output shape via the model's bound table. No
// allocation, no kernel launch.
//
// Declares an `output_alias` pointing at the upstream tile so the
// codegen drop pass keeps the underlying `OwnedTensor` alive until
// every consumer of the reshaped view is done.

#[derive(Debug, Default)]
pub struct ReshapeRefImpl;

impl Implementation for ReshapeRefImpl {
    fn name(&self) -> &'static str {
        "reshape_ref"
    }

    fn target_compatible(&self, _profile: &TargetProfile) -> bool {
        true
    }

    fn matches(&self, fuf: &Fuf, seed: TileId, _profile: &TargetProfile) -> Option<MatchInfo> {
        single_tile_match(fuf, seed, OpKind::Reshape)
    }

    fn cost_us(&self, _m: &MatchInfo, _ctx: &CostCtx) -> f64 {
        // Pure metadata change — a handful of nanoseconds host-side,
        // zero on the GPU. Model as 0.0 so the solver never spends
        // effort choosing between reshape variants.
        0.0
    }

    fn resources(&self, _m: &MatchInfo) -> Resources {
        Resources::ZERO
    }

    fn launch_kind(&self) -> LaunchKind {
        LaunchKind::HostCallback
    }

    fn supported_input_handoffs(&self) -> &[Handoff] {
        const H: &[Handoff] = &[Handoff::StreamOrder, Handoff::StreamEvent];
        H
    }

    fn supported_output_handoffs(&self) -> &[Handoff] {
        const H: &[Handoff] = &[Handoff::StreamOrder, Handoff::StreamEvent];
        H
    }

    fn input_layouts(&self, m: &MatchInfo) -> Vec<Layout> {
        vec![Layout::RowMajorBf16; m.boundary_inputs.len()]
    }

    fn output_layouts(&self, m: &MatchInfo) -> Vec<Layout> {
        vec![Layout::RowMajorBf16; m.boundary_outputs.len()]
    }

    fn is_compute_bound(&self) -> bool {
        false
    }

    #[allow(clippy::type_complexity)]
    fn output_alias(
        &self,
        claimed_tiles: &[TileId],
        fuf: &Fuf,
    ) -> Vec<((TileId, u8), Option<(TileId, u8)>)> {
        // The reshaped view aliases the upstream tile's OwnedTensor.
        // The drop pass must keep the upstream alive until every
        // consumer of this reshape is done.
        let reshape_id = claimed_tiles[0];
        let upstream = match fuf.get(reshape_id).inputs.first() {
            Some(FufInput::Tile { id, slot }) => Some((*id, *slot)),
            _ => None,
        };
        vec![((reshape_id, 0), upstream)]
    }

    fn emit_call(&self, ctx: &EmitCtx) -> TokenStream {
        let tile = ctx.primary();
        let out = ctx.output_ident(tile, 0);
        let input = ctx.input_expr(tile, 0);
        // Build per-dim expressions for the target shape. Each Dim is
        // one of Lit / Bound / Mul. Config.json bounds fold to integer
        // literals at emit time; the lone runtime bound `num_tokens`
        // emits as `(input).dim(0)` since the reshape preserves the
        // input's leading axis. Vars are a compiler bug at this point
        // (shape inference closed every dim).
        let shape = &ctx.fuf.get(tile).outputs[0];
        let dim_tokens: Vec<proc_macro2::TokenStream> = shape
            .iter()
            .map(|d| reshape_dim_token(d, ctx, &input))
            .collect();
        quote! {
            let #out = unsafe { (#input).reshape(&[ #( #dim_tokens ),* ]) };
        }
    }
}

/// Emit a `usize`-typed Rust expression for one Dim of a reshape
/// target shape. Config bounds fold to literals; `num_tokens` reads
/// off `ctx.input_ids.dim(0)` (the authoritative runtime source —
/// the immediate reshape input's `dim(0)` isn't stable once a prior
/// reshape has merged axes, e.g. `[T, heads*head_dim]` → `[T*heads,
/// head_dim]`); Mul recurses with `*`.
fn reshape_dim_token(
    d: &crate::shape::Dim,
    ctx: &EmitCtx,
    input: &TokenStream,
) -> proc_macro2::TokenStream {
    use crate::shape::Dim;
    let _ = input;
    match d {
        Dim::Lit(n) => {
            let v = *n as usize;
            quote! { #v }
        }
        Dim::Bound(name) if name == "num_tokens" => {
            quote! { (*ctx.input_ids).dim(0) }
        }
        Dim::Bound(name) => {
            let v = ctx.bound(name) as usize;
            quote! { #v }
        }
        Dim::Mul(factors) => {
            let mut parts = factors.iter().map(|f| reshape_dim_token(f, ctx, input));
            let first = parts.next().unwrap_or_else(|| quote! { 1usize });
            let folded = parts.fold(first, |acc, p| quote! { (#acc) * (#p) });
            quote! { (#folded) }
        }
        Dim::Var(_) => {
            panic!("reshape target dim must be closed (no Var) — compiler bug")
        }
    }
}

/// Evaluate a symbolic `Dim` to a concrete `usize` via the bound
/// table embedded in the emit context. Panics if a `Var` is reached
/// (shape inference should have closed every dim) or a bound is
/// missing (the model config is incomplete).
fn eval_dim_usize(d: &crate::shape::Dim, ctx: &EmitCtx) -> Option<usize> {
    use crate::shape::Dim;
    match d {
        Dim::Lit(n) => Some(*n as usize),
        Dim::Bound(name) => Some(ctx.bound(name) as usize),
        Dim::Mul(factors) => factors
            .iter()
            .try_fold(1usize, |acc, f| eval_dim_usize(f, ctx).map(|v| acc * v)),
        Dim::Var(_) => None,
    }
}

// Silu and Mul have no singleton impls. The only kernel in
// `ferrite-kernels` that implements them is `silu_and_mul_fused`,
// which operates on a packed `[num_tokens, 2*intermediate]` buffer
// produced by a single fused gate+up GEMM. Any DSL occurrence of
// `Silu`/`Mul` outside the `(gemm, gemm, silu, mul)` MLP pattern
// would need its own dedicated kernel + Impl — until such an op
// exists there is no structural-fallback worth providing, and an
// unmatched Silu/Mul is a library bug the solver reports via
// `SolveError::UnclaimedTile`.
//
// See `FusedGateUpSiluMulImpl` below for the matcher that claims
// the MLP pattern.

/// Baseline library: HostCallback impl per OpKind plus the
/// multi-tile fusions the `ferrite-kernels` shape requires.
/// Replaced / augmented by calibrated target-specific impls as
/// they're ported.
pub fn starter_library() -> ImplementationLibrary {
    let mut lib = ImplementationLibrary::new();
    lib.push(Box::new(EmbedRefImpl));
    lib.push(Box::new(RmsNormRefImpl));
    lib.push(Box::new(LayerNormRefImpl));
    lib.push(Box::new(GemmRefImpl));
    lib.push(Box::new(AttentionViaCacheImpl));
    // Reshape is a metadata-only view op synthesized by shape
    // inference to bridge axis-factor mismatches (e.g. per-head QK-
    // norm in Qwen3/Gemma3). Zero-cost, zero-launch; the emitted
    // code is a single `TensorView::reshape(&[..])` call.
    lib.push(Box::new(ReshapeRefImpl));
    // Multi-tile fusions. The solver's claim-size-DESC sort picks
    // these over singleton coverage when both apply; the singletons
    // stay as fallbacks for tile positions the fusion doesn't match
    // (e.g. the first layer's input_layernorm, whose upstream is
    // `embed` not `Add`, stays a singleton RmsNorm claim).
    //
    // `(Gemm, BiasAdd)` pairs not absorbed by a larger fusion (e.g.
    // a lone affine-transform gemm that's not a QKV-pre-rope or
    // gate/up-pre-MLP). Emits cuBLAS gemm_bias via `LinearLayer::forward`.
    lib.push(Box::new(FusedGemmBiasImpl));
    lib.push(Box::new(FusedGateUpSiluMulImpl));
    lib.push(Box::new(FusedGateUpGeluMulImpl));
    lib.push(Box::new(FusedAddRmsNormImpl));
    // Singleton fallback for residual `Add`s whose downstream is not
    // a RmsNorm — Cohere's parallel attn+MLP residual pair, layer-end
    // adds before any LayerNorm. Emits `add_inplace`.
    lib.push(Box::new(AddRefImpl));
    // Gemma-style 3-tile fusion: residual-Add + scalar-offset-Add
    // + RmsNorm. Claimed by the DP in preference to the 2-tile
    // FusedAddRmsNorm + standalone ScalarOffset because it's a
    // larger claim.
    lib.push(Box::new(FusedAddRmsNormWithOffsetImpl));
    // Gemma-style `rmsnorm(x, w + scalar)` for the standalone case
    // (no upstream residual-Add): scalar rides as the rms_norm
    // kernel's `weight_offset` param.
    lib.push(Box::new(ScalarOffsetRmsNormImpl));
    // Tile × scalar in-place multiply (e.g. Gemma embed scale).
    // Only claims Mul tiles whose inputs are (Tile, Scalar); the
    // tensor×tensor SwiGLU / GELU fusions claim the disjoint
    // (Tile, Tile) pattern.
    lib.push(Box::new(ScalarMulImpl));
    // Decode / prefill QKV+rope variants — the solver picks via
    // WorkloadConstraint (M=1 → Cache, M≥2 → Prefill).
    lib.push(Box::new(FusedQkvRopeCacheImpl));
    lib.push(Box::new(FusedQkvRopePrefillImpl));
    // FusedQkvQkNormRopeCacheImpl (three-gemm + fused qk_norm_rope +
    // cache) is staged in this crate but intentionally NOT registered.
    // The 3-gemm emit_call produces incorrect numerics on Qwen3/Gemma3
    // (attention output diverges from golden at prompt 0 position 0).
    // Root cause is unresolved: possibly Q/K/V ordering, reshape
    // layout, or the interaction between the per-arch rotary cache
    // and the kernel's expected layout. Until the correctness issue
    // is tracked down, Qwen3/Gemma3 go through singletons
    // (GemmRef + ReshapeRef + RmsNormRef + RopeAppendRef) which IS
    // correctness-green.
    //
    // The singleton path is correctness-equivalent to what Qwen3 shipped
    // with originally; the fused impl is a perf optimization that
    // stays scoped to a future session with a proper numerics bringup.
    //
    // Singleton fallback claims standalone RopeAppend tiles. Emits
    // `rotary_embedding_inplace` + `reshape_and_cache`.
    lib.push(Box::new(RopeAppendRefImpl));
    // Matching attention pair — decode reads from cache, prefill
    // reads the contiguous K/V produced by the prefill QKV impl.
    lib.push(Box::new(AttentionPrefillContiguousImpl));
    // Sliding-window variants of the attention pair. Claim
    // `OpKind::SlidingAttention` so the DSL author opts into window
    // masking per-tile (e.g. alternating layers via `if` in the DSL
    // body). `window_size_left` reads from `sliding_window` config.
    lib.push(Box::new(SlidingAttentionViaCacheImpl));
    lib.push(Box::new(SlidingAttentionPrefillContiguousImpl));
    // Standalone TanhSoftCap — reads `final_logit_softcapping` from
    // config. The DSL body emits `tanh_softcap(...)` only for
    // architectures that cap logits.
    lib.push(Box::new(TanhSoftCapImpl));
    // Cutlass standalone GEMM tile zoo — one Impl per tile variant
    // in `target_profiles/cost_*.csv`. `target_compatible` gates each
    // by "does this target have a calibrated cost row for this
    // variant?", so targets without CSV data silently fall back to
    // `GemmRefImpl` (cuBLAS). Matching is rejected for Gemms whose
    // output flows into a fusion (RopeAppend / Silu / Mul) so the
    // solver can never pick cutlass for a QKV or gate/up gemm that
    // would otherwise break its fusion chain.
    for tile in CUTLASS_TILE_ZOO {
        lib.push(Box::new(CutlassGemmImpl {
            tile_m: tile.0,
            tile_n: tile.1,
            stages: tile.2,
        }));
    }
    lib.push(Box::new(CutlassGemvImpl));

    // ── Marlin (AWQ / GPTQ) impls ───────────────────────────────
    //
    // Active only on models whose `quantization_config` resolves at
    // least one weight to a Marlin-consumable storage format
    // (`StorageFormat::Awq { .. }` or `StorageFormat::Gptq { .. }`).
    // Each matcher gates on quant storage, so they're no-ops on
    // dense models — the dense fused impls claim the same patterns
    // for `Dense` weights. Registered after the Cutlass zoo so they
    // land with the rest of the matmul kernels.
    lib.push(Box::new(MarlinGemmImpl));
    lib.push(Box::new(MarlinFusedGateUpSiluMulImpl));
    lib.push(Box::new(MarlinFusedGateUpGeluMulImpl));
    lib.push(Box::new(MarlinFusedQkvRopeCacheImpl));
    lib.push(Box::new(MarlinFusedQkvRopePrefillImpl));

    // ── BitsAndBytes 4-bit (NF4 / FP4) impls ────────────────────
    // Gated per-matches on `StorageFormat::Bnb4 { .. }`; stay
    // dormant on dense / Marlin-consumable models.
    lib.push(Box::new(Bnb4GemmImpl));
    lib.push(Box::new(Bnb4FusedGateUpSiluMulImpl));
    lib.push(Box::new(Bnb4FusedGateUpGeluMulImpl));
    lib.push(Box::new(Bnb4FusedQkvRopeCacheImpl));
    lib.push(Box::new(Bnb4FusedQkvRopePrefillImpl));

    // ── FP8 (E4M3) impls ────────────────────────────────────────
    // Gated per-matches on `StorageFormat::Fp8 { .. }`; stay dormant
    // on dense / Marlin / BNB4 models. Singleton only today; fused
    // QKV / gate-up peers are perf follow-ups.
    lib.push(Box::new(Fp8GemmImpl));
    lib.push(Box::new(Fp8FusedGemmBiasImpl));
    lib.push(Box::new(Fp8FusedGateUpSiluMulImpl));
    lib.push(Box::new(Fp8FusedGateUpGeluMulImpl));
    lib.push(Box::new(Fp8FusedQkvRopeCacheImpl));
    lib.push(Box::new(Fp8FusedQkvRopePrefillImpl));

    // FlashInfer paged attention — one Decode + one Prefill Impl per
    // tuple in FLASHINFER_CONFIG_SET (see `ferrite-cuda-builder`). Each
    // variant's `target_compatible` gates on whether the calibrated CSV
    // has a row for its (head_dim, softcap) family, so targets without
    // the FlashInfer tuple (either uncompiled or uncalibrated) silently
    // fall back to the FA2 attention Impls above. Keep in sync with
    // `FLASHINFER_CONFIG_SET` — adding a tuple there without mirroring
    // it here leaves the Impl unreachable; removing a tuple without
    // mirroring leaves the Impl with no matching extern symbols.
    for &head_dim in &[64u32, 128, 256] {
        for &use_softcap in &[false, true] {
            lib.push(Box::new(FlashInferAttentionDecodeImpl {
                head_dim,
                use_logits_soft_cap: use_softcap,
            }));
            lib.push(Box::new(FlashInferAttentionPrefillImpl {
                head_dim,
                use_logits_soft_cap: use_softcap,
            }));
        }
    }
    lib
}

// ── FusedGemmBiasImpl ────────────────────────────────────────────
//
// 2-tile fusion: `(Gemm, BiasAdd)` where BiasAdd consumes the Gemm's
// output at its first input slot. Emits `LinearLayer::forward` on a
// packed accessor covering (weight, bias) — the dense variant
// dispatches to cuBLAS's `gemm_bias` epilog, producing the biased
// result in one launch.
//
// This is the fallback path for affine-transform gemms not absorbed
// by a larger fusion (e.g. `FusedQkvRopeCacheImpl` absorbs Qwen2's
// QKV pre-rope `(Gemm, BiasAdd) × 3` triples, so this impl only
// claims stray pairs elsewhere in the body).
//
// Required weights: one fused accessor carrying both the Gemm's
// weight ref and the BiasAdd's bias ref. The user returns a
// `LinearLayer::Dense` with its `bias: Some(...)` populated —
// `LinearLayer::load_dense` already auto-detects bias from the
// safetensors path, so most callers get this for free.

#[derive(Debug, Default)]
pub struct FusedGemmBiasImpl;

impl Implementation for FusedGemmBiasImpl {
    fn name(&self) -> &'static str {
        "fused_gemm_bias"
    }

    fn target_compatible(&self, _profile: &TargetProfile) -> bool {
        true
    }

    fn matches(&self, fuf: &Fuf, seed: TileId, _profile: &TargetProfile) -> Option<MatchInfo> {
        let seed_node = fuf.get(seed);
        if seed_node.op != OpKind::Gemm {
            return None;
        }
        // cuBLAS `gemm_bias` epilog on dense BF16/F16 weights. AWQ /
        // GPTQ Gemms route through `MarlinFusedQkvRope*Impl` (which
        // walks the same `(Gemm, BiasAdd)` chain and dispatches to
        // `MarlinLinear::forward` — which internally adds the packed
        // `.bias` after `marlin_gemm`). Reject here so the DP doesn't
        // pick this dense-typed accessor for a quantized weight.
        if !matches!(weight_storage_of(seed_node), Some(StorageFormat::Dense)) {
            return None;
        }
        // Find a BiasAdd tile whose first tile input is this Gemm.
        // BiasAdd's shape sig is `(x, b) -> x` with `x` at slot 0 and
        // `b` at slot 1 (the bias weight, not a tile).
        let bias_node = fuf.nodes.iter().find(|n| {
            n.op == OpKind::BiasAdd
                && matches!(
                    n.inputs.first(),
                    Some(FufInput::Tile { id, .. }) if *id == seed
                )
        })?;
        let bias_id = bias_node.id;

        let mut claimed = vec![seed, bias_id];
        claimed.sort();

        // Boundary inputs: the gemm's upstream activation tile(s).
        let activation_inputs: Vec<TileId> = seed_node
            .inputs
            .iter()
            .filter_map(|i| match i {
                FufInput::Tile { id, .. } => Some(*id),
                _ => None,
            })
            .collect();

        Some(MatchInfo {
            claimed_tiles: claimed,
            boundary_inputs: activation_inputs,
            // Only BiasAdd's output is live downstream — the Gemm's
            // intermediate output is consumed inside the fused kernel.
            boundary_outputs: vec![bias_id],
        })
    }

    fn cost_us(&self, m: &MatchInfo, ctx: &CostCtx) -> f64 {
        // cuBLAS `gemm_bias` has a negligible epilog cost on top of
        // the underlying GEMM — delegate to `cost_gemm` on the Gemm
        // tile so this impl stays on the same measurement scale as
        // singleton GemmRefImpl / CutlassGemmImpl. The DP's cost
        // comparison between (strict gemm + orphaned bias_add) and
        // (fused gemm_bias) will always prefer the fusion because
        // the alternative is `f64::INFINITY` (BiasAdd has no
        // singleton impl) — cost parity is not a correctness
        // concern here, only a calibration one.
        let gemm_id = *m
            .claimed_tiles
            .iter()
            .find(|t| ctx.fuf.get(**t).op == OpKind::Gemm)
            .expect("claim contains Gemm");
        let gemm_info = MatchInfo {
            claimed_tiles: vec![gemm_id],
            boundary_inputs: vec![],
            boundary_outputs: vec![gemm_id],
        };
        cost_gemm(&gemm_info, ctx)
    }

    fn resources(&self, _m: &MatchInfo) -> Resources {
        Resources::ZERO
    }

    fn launch_kind(&self) -> LaunchKind {
        LaunchKind::HostCallback
    }

    fn supported_input_handoffs(&self) -> &[Handoff] {
        const H: &[Handoff] = &[Handoff::StreamOrder, Handoff::StreamEvent];
        H
    }

    fn supported_output_handoffs(&self) -> &[Handoff] {
        const H: &[Handoff] = &[Handoff::StreamOrder, Handoff::StreamEvent];
        H
    }

    fn input_layouts(&self, m: &MatchInfo) -> Vec<Layout> {
        vec![Layout::RowMajorBf16; m.boundary_inputs.len()]
    }

    fn output_layouts(&self, m: &MatchInfo) -> Vec<Layout> {
        vec![Layout::RowMajorBf16; m.boundary_outputs.len()]
    }

    fn is_compute_bound(&self) -> bool {
        true
    }

    fn required_weights(
        &self,
        claimed_tiles: &[TileId],
        fuf: &Fuf,
        program: &Program,
    ) -> Vec<WeightAccessor> {
        // Single accessor typed `LinearLayer`, source = the Gemm's
        // weight. The bias rides through the accessor's
        // `.dense_bias()` field; the user's loader (typically
        // `LinearLayer::load_dense`) auto-detects it from the
        // safetensors path `<prefix>.bias`. `emit_call` asserts
        // `dense_bias().is_some()` at runtime so a data/DSL mismatch
        // surfaces loudly instead of silently dropping the bias.
        let gemm_id = *claimed_tiles
            .iter()
            .find(|t| fuf.get(**t).op == OpKind::Gemm)
            .expect("claim contains Gemm");
        let gemm_weight = first_weight_ref(fuf.get(gemm_id)).expect("gemm has a weight");
        let sources = vec![gemm_weight];
        let name = fused_accessor_name(program, &sources);
        vec![WeightAccessor {
            name,
            rust_type: quote! { ::ferrite_kernels::layers::LinearLayer },
            source_weights: sources,
        }]
    }

    fn emit_call(&self, ctx: &EmitCtx) -> TokenStream {
        let gemm_id = *ctx
            .claimed_tiles
            .iter()
            .find(|t| ctx.fuf.get(**t).op == OpKind::Gemm)
            .expect("claim contains Gemm");
        let bias_id = *ctx
            .claimed_tiles
            .iter()
            .find(|t| ctx.fuf.get(**t).op == OpKind::BiasAdd)
            .expect("claim contains BiasAdd");

        let activation = ctx.input_expr(gemm_id, 0);
        let gemm_weight = first_weight_ref(ctx.fuf.get(gemm_id)).expect("gemm weight ref");
        let fused_name = fused_accessor_name(ctx.program, &[gemm_weight]);
        let weight_expr = ctx.weight_accessor(&fused_name);

        let out = ctx.output_ident(bias_id, 0);
        quote! {
            let #out = unsafe {
                debug_assert!(
                    (#weight_expr).dense_bias().is_some(),
                    "FusedGemmBiasImpl: DSL `bias_add` claimed but \
                     LinearLayer has no bias — check safetensors path"
                );
                (#weight_expr).forward(
                    #activation,
                    &mut device.cublas,
                    &mut device.caching,
                    device.compute_stream,
                )
            };
        }
    }
}

// ── FusedGateUpSiluMulImpl ───────────────────────────────────────
//
// Multi-tile impl claiming the `(Gemm, Gemm, Silu, Mul)` MLP
// pattern and mapping it to the `silu_and_mul_fused` kernel (which
// expects a packed `[num_tokens, 2*intermediate]` buffer).
//
// Match is purely structural — no phase tags, no weight-name
// checks. Pattern: two Gemm tiles sharing the same activation
// Tile-input feed a Silu and a Mul, with the Mul consuming the
// Silu output and the second Gemm output. Survives renaming
// gate_proj/up_proj → anything and survives adding new
// architectures that reuse the same topology.
//
// The impl declares one fused WeightBundle accessor (see
// `required_weights`) whose `rust_type` is `LinearLayer`. The user
// populates it with a `LinearLayer` carrying the vertically-
// concatenated `[gate_proj | up_proj]` weight; `LinearLayer::forward`
// on that produces the packed `gate_up` buffer `silu_and_mul_fused`
// wants. No intermediate D2D copies; no cross-impl contract.

#[derive(Debug, Default)]
pub struct FusedGateUpSiluMulImpl;

/// The [`StorageFormat`] of the first weight input of `node`, or
/// `None` if the node has no weight inputs. Every Gemm tile in the
/// FUF carries exactly one `FufInput::Weight`; non-Gemm tiles return
/// `None`. Dense and quant-aware impls alike read this to decide
/// whether the kernel they emit (cuBLAS / cutlass vs. Marlin) can
/// legally consume the weight's storage.
fn weight_storage_of(node: &FufNode) -> Option<&StorageFormat> {
    node.inputs.iter().find_map(|i| match i {
        FufInput::Weight { storage, .. } => Some(storage),
        _ => None,
    })
}

/// Return the `(TileId, slot)` of a node's first `FufInput::Tile`
/// input. For Gemm this identifies the activation (the weight input
/// is a `FufInput::Weight`).
fn first_tile_input(node: &crate::fuf::FufNode) -> Option<(TileId, u8)> {
    node.inputs.iter().find_map(|i| match i {
        FufInput::Tile { id, slot } => Some((*id, *slot)),
        _ => None,
    })
}

/// The `WeightId` + concrete index read by the first weight-typed
/// input of a tile. For Gemm the weight is in slot 1 of the DSL's
/// call; we don't care about position, just "which weight flows in".
fn first_weight_ref(node: &crate::fuf::FufNode) -> Option<(WeightId, Option<u64>)> {
    node.inputs.iter().find_map(|i| match i {
        FufInput::Weight { id, index, .. } => Some((*id, *index)),
        _ => None,
    })
}

/// Build a stable, structural accessor name covering multiple
/// source weights. Joins each source weight's
/// [`weight_field_name`] with `"__fused__"`, sorted for
/// order-independence. Two impls that declare the same source set
/// produce the same name.
pub fn fused_accessor_name(program: &Program, sources: &[(WeightId, Option<u64>)]) -> syn::Ident {
    let mut parts: Vec<String> = sources
        .iter()
        .map(|(id, idx)| weight_field_name(program, *id, *idx).to_string())
        .collect();
    parts.sort();
    let joined = parts.join("__fused__");
    syn::Ident::new(&joined, proc_macro2::Span::call_site())
}

impl Implementation for FusedGateUpSiluMulImpl {
    fn name(&self) -> &'static str {
        "fused_gate_up_silu_mul"
    }

    fn target_compatible(&self, _profile: &TargetProfile) -> bool {
        true
    }

    fn matches(&self, fuf: &Fuf, seed: TileId, _profile: &TargetProfile) -> Option<MatchInfo> {
        // Seed on the **gate** Gemm — the Gemm whose output feeds a
        // Silu. The FUF's topological order processes Gemms before
        // their downstream Silu/Mul consumers, so seeding on the
        // upstream Gemm commits the 4-tile claim before the DP
        // reaches (and would singleton-claim) the Silu. Seeding on
        // Silu instead would leave the solver's per-tile greedy
        // picking GemmRefImpl at the gate_gemm's earlier position
        // and orphaning the downstream Silu/Mul — see the
        // larger-claim-preferred sort in solver.rs:solve_one.
        let gate_gemm = fuf.get(seed);
        if gate_gemm.op != OpKind::Gemm {
            return None;
        }
        // Dense kernel: reject AWQ storage.
        if !matches!(weight_storage_of(gate_gemm), Some(StorageFormat::Dense)) {
            return None;
        }

        // Find a downstream Silu consuming this Gemm.
        let silu_node = fuf
            .nodes
            .iter()
            .find(|n| n.op == OpKind::Silu && consumes_tile(n, seed))?;
        let silu_id = silu_node.id;

        // Find the Mul consuming that Silu.
        let mul_node = fuf
            .nodes
            .iter()
            .find(|n| n.op == OpKind::Mul && consumes_tile(n, silu_id))?;
        let mul_id = mul_node.id;

        // Mul's other Tile input is the up Gemm.
        let up_gemm_id = mul_node.inputs.iter().find_map(|i| match i {
            FufInput::Tile { id, .. } if *id != silu_id => Some(*id),
            _ => None,
        })?;
        let up_gemm = fuf.get(up_gemm_id);
        if up_gemm.op != OpKind::Gemm {
            return None;
        }

        // Both Gemms must read the same activation tile + slot.
        if first_tile_input(gate_gemm)? != first_tile_input(up_gemm)? {
            return None;
        }

        // Claimed tiles sorted by TileId for determinism.
        let mut claimed = [seed, up_gemm_id, silu_id, mul_id];
        claimed.sort();
        let claimed = claimed.to_vec();

        let activation_tile = first_tile_input(gate_gemm)?.0;
        Some(MatchInfo {
            claimed_tiles: claimed,
            boundary_inputs: vec![activation_tile],
            boundary_outputs: vec![mul_id],
        })
    }

    fn cost_us(&self, _m: &MatchInfo, ctx: &CostCtx) -> f64 {
        // Cost = one fused GEMM (M × 2I × H) + one elementwise pass
        // over the packed [M, 2I] buffer producing [M, I].
        let num_tokens = ctx.num_tokens() as f64;
        let hidden = ctx.bounds.get("hidden_size").copied().unwrap_or(0) as f64;
        let intermediate = ctx.bounds.get("intermediate_size").copied().unwrap_or(0) as f64;

        // Compute-bound GEMM.
        let flops = 2.0 * num_tokens * (2.0 * intermediate) * hidden;
        let peak = ctx.profile.peak_tflops_fp16 * 1e12;
        let gemm_us = if peak > 0.0 && flops > 0.0 {
            (flops / peak) * 1e6
        } else {
            0.0
        };

        // Bandwidth-bound silu*mul: read 2*M*I, write M*I, bf16 = 2 B.
        let bw_gb = ctx.profile.memory_bandwidth_gbps;
        let bytes = 3.0 * num_tokens * intermediate * BYTES_PER_ELEM;
        let silu_mul_us = if bw_gb > 0.0 {
            (bytes / (bw_gb * 1e9)) * 1e6
        } else {
            0.0
        };

        gemm_us + silu_mul_us
    }

    fn resources(&self, _m: &MatchInfo) -> Resources {
        Resources::ZERO
    }

    fn launch_kind(&self) -> LaunchKind {
        LaunchKind::HostCallback
    }

    fn supported_input_handoffs(&self) -> &[Handoff] {
        const H: &[Handoff] = &[Handoff::StreamOrder, Handoff::StreamEvent];
        H
    }

    fn supported_output_handoffs(&self) -> &[Handoff] {
        const H: &[Handoff] = &[Handoff::StreamOrder, Handoff::StreamEvent];
        H
    }

    fn input_layouts(&self, m: &MatchInfo) -> Vec<Layout> {
        vec![Layout::RowMajorBf16; m.boundary_inputs.len()]
    }

    fn output_layouts(&self, m: &MatchInfo) -> Vec<Layout> {
        vec![Layout::RowMajorBf16; m.boundary_outputs.len()]
    }

    fn is_compute_bound(&self) -> bool {
        true
    }

    fn required_weights(
        &self,
        claimed_tiles: &[TileId],
        fuf: &Fuf,
        program: &Program,
    ) -> Vec<WeightAccessor> {
        // Collect the two Gemm tiles' weight refs, merge into one
        // fused accessor. The user provides a `LinearLayer` whose
        // weight is the vertically-concatenated `[gate_proj | up_proj]`.
        let sources: Vec<(WeightId, Option<u64>)> = claimed_tiles
            .iter()
            .filter_map(|t| {
                let n = fuf.get(*t);
                if n.op == OpKind::Gemm {
                    first_weight_ref(n)
                } else {
                    None
                }
            })
            .collect();
        let name = fused_accessor_name(program, &sources);
        vec![WeightAccessor {
            name,
            rust_type: quote! { ::ferrite_kernels::layers::LinearLayer },
            source_weights: sources,
        }]
    }

    fn emit_call(&self, ctx: &EmitCtx) -> TokenStream {
        // Identify the four tiles by op kind. The claim is sorted by
        // TileId (see `matches`); emission is independent of order.
        let silu_id = *ctx
            .claimed_tiles
            .iter()
            .find(|t| ctx.fuf.get(**t).op == OpKind::Silu)
            .expect("fused gate/up/silu/mul claim must contain Silu");
        let mul_id = *ctx
            .claimed_tiles
            .iter()
            .find(|t| ctx.fuf.get(**t).op == OpKind::Mul)
            .expect("fused gate/up/silu/mul claim must contain Mul");
        // Gate gemm = Silu's Tile input; up gemm = the other claimed Gemm.
        let (gate_id, _) =
            first_tile_input(ctx.fuf.get(silu_id)).expect("silu has a tile input — the gate gemm");
        let up_id = ctx
            .claimed_tiles
            .iter()
            .copied()
            .find(|t| ctx.fuf.get(*t).op == OpKind::Gemm && *t != gate_id)
            .expect("claim contains a second Gemm — the up gemm");

        // The shared activation — same Tile-input on both Gemms.
        let activation = ctx.input_expr(gate_id, 0);

        // Fused-weight accessor. Recompute the name from the claim
        // (matches `required_weights`), so user-provided trait impl
        // and emit are kept in lockstep.
        let gate_w = first_weight_ref(ctx.fuf.get(gate_id)).expect("gate gemm has a weight");
        let up_w = first_weight_ref(ctx.fuf.get(up_id)).expect("up gemm has a weight");
        let fused_name = fused_accessor_name(ctx.program, &[gate_w, up_w]);
        let weight_expr = ctx.weight_accessor(&fused_name);

        let mul_out = ctx.output_ident(mul_id, 0);
        let intermediate = ctx.bound("intermediate_size") as usize;

        quote! {
            // Fused gate+up GEMM produces a [num_tokens, 2*intermediate]
            // packed buffer that's only consumed by `silu_and_mul_fused`
            // on the next line. Bind it in an inner scope so its
            // OwnedTensor drops as soon as the kernel returns.
            let #mul_out = unsafe {
                let gate_up = (#weight_expr).forward(
                    #activation,
                    &mut device.cublas,
                    &mut device.caching,
                    device.compute_stream,
                );
                ::ferrite_kernels::kernels::silu_and_mul_fused(
                    *gate_up,
                    #intermediate,
                    &mut device.caching,
                    device.compute_stream,
                )
            };
        }
    }
}

/// Whether `node` consumes the output of `producer` via any Tile input.
fn consumes_tile(node: &crate::fuf::FufNode, producer: TileId) -> bool {
    node.inputs
        .iter()
        .any(|i| matches!(i, FufInput::Tile { id, .. } if *id == producer))
}

/// Return the `cos_sin_cache` TokenStream for a rope-related tile.
/// Checks whether the tile (or any tile in `claimed`) carries an
/// `ExternKind::RotaryLocal` input; if so emits `wm.rotary_local`,
/// otherwise `wm.rotary`. Both are macro-emitted fields on the
/// per-arch `Weights` struct — ferrite owns rotary end-to-end and
/// `ForwardCtx` carries no rotary.
fn rotary_cos_sin_tokens(fuf: &Fuf, claimed: &[TileId]) -> TokenStream {
    let uses_local = claimed.iter().any(|&tid| {
        fuf.get(tid).inputs.iter().any(|i| {
            matches!(
                i,
                FufInput::Extern {
                    kind: ExternKind::RotaryLocal,
                    ..
                }
            )
        })
    });
    if uses_local {
        quote! { wm.rotary_local.cos_sin_cache }
    } else {
        quote! { wm.rotary.cos_sin_cache }
    }
}

// ── FusedGateUpGeluMulImpl ───���──────────────────────────────���────
//
// Structural mirror of [`FusedGateUpSiluMulImpl`] for the GELU
// variant of the gate/up MLP fusion. Claims the
// `(Gemm, Gemm, Gelu, Mul)` pattern — the MLP topology of any
// architecture whose activation is GELU rather than SwiGLU.
//
// Pattern-match logic is identical to the Silu variant, seeded on
// the gate Gemm; the only structural differences in emission are:
//   - the dispatched kernel is `gelu_and_mul_fused`;
//   - the upstream activation tile is the `Gelu` node (not `Silu`).
//
// Registered alongside the Silu variant in `starter_library`; the
// two are mutually exclusive on any given FUF since a single gate
// Gemm cannot simultaneously feed a Silu and a Gelu.

#[derive(Debug, Default)]
pub struct FusedGateUpGeluMulImpl;

impl Implementation for FusedGateUpGeluMulImpl {
    fn name(&self) -> &'static str {
        "fused_gate_up_gelu_mul"
    }

    fn target_compatible(&self, _profile: &TargetProfile) -> bool {
        true
    }

    fn matches(&self, fuf: &Fuf, seed: TileId, _profile: &TargetProfile) -> Option<MatchInfo> {
        // Seed on the gate Gemm (whose output feeds the Gelu). The
        // FUF's topological order puts Gemms before their downstream
        // consumers, so seeding on the upstream Gemm commits the
        // 4-tile claim before the DP greedy-claims the Gelu.
        let gate_gemm = fuf.get(seed);
        if gate_gemm.op != OpKind::Gemm {
            return None;
        }
        // Dense kernel: reject AWQ storage.
        if !matches!(weight_storage_of(gate_gemm), Some(StorageFormat::Dense)) {
            return None;
        }
        let gelu_node = fuf
            .nodes
            .iter()
            .find(|n| n.op == OpKind::Gelu && consumes_tile(n, seed))?;
        let gelu_id = gelu_node.id;
        let mul_node = fuf
            .nodes
            .iter()
            .find(|n| n.op == OpKind::Mul && consumes_tile(n, gelu_id))?;
        let mul_id = mul_node.id;
        let up_gemm_id = mul_node.inputs.iter().find_map(|i| match i {
            FufInput::Tile { id, .. } if *id != gelu_id => Some(*id),
            _ => None,
        })?;
        let up_gemm = fuf.get(up_gemm_id);
        if up_gemm.op != OpKind::Gemm {
            return None;
        }
        if first_tile_input(gate_gemm)? != first_tile_input(up_gemm)? {
            return None;
        }
        let mut claimed = [seed, up_gemm_id, gelu_id, mul_id];
        claimed.sort();
        let claimed = claimed.to_vec();
        let activation_tile = first_tile_input(gate_gemm)?.0;
        Some(MatchInfo {
            claimed_tiles: claimed,
            boundary_inputs: vec![activation_tile],
            boundary_outputs: vec![mul_id],
        })
    }

    fn cost_us(&self, _m: &MatchInfo, ctx: &CostCtx) -> f64 {
        // Same cost shape as SwiGLU: fused [M × 2I × H] GEMM + a
        // bandwidth-bound [M, 2I] → [M, I] elementwise pass.
        let num_tokens = ctx.num_tokens() as f64;
        let hidden = ctx.bounds.get("hidden_size").copied().unwrap_or(0) as f64;
        let intermediate = ctx.bounds.get("intermediate_size").copied().unwrap_or(0) as f64;
        let flops = 2.0 * num_tokens * (2.0 * intermediate) * hidden;
        let peak = ctx.profile.peak_tflops_fp16 * 1e12;
        let gemm_us = if peak > 0.0 && flops > 0.0 {
            (flops / peak) * 1e6
        } else {
            0.0
        };
        let bw_gb = ctx.profile.memory_bandwidth_gbps;
        let bytes = 3.0 * num_tokens * intermediate * BYTES_PER_ELEM;
        let act_mul_us = if bw_gb > 0.0 {
            (bytes / (bw_gb * 1e9)) * 1e6
        } else {
            0.0
        };
        gemm_us + act_mul_us
    }

    fn resources(&self, _m: &MatchInfo) -> Resources {
        Resources::ZERO
    }

    fn launch_kind(&self) -> LaunchKind {
        LaunchKind::HostCallback
    }

    fn supported_input_handoffs(&self) -> &[Handoff] {
        const H: &[Handoff] = &[Handoff::StreamOrder, Handoff::StreamEvent];
        H
    }

    fn supported_output_handoffs(&self) -> &[Handoff] {
        const H: &[Handoff] = &[Handoff::StreamOrder, Handoff::StreamEvent];
        H
    }

    fn input_layouts(&self, m: &MatchInfo) -> Vec<Layout> {
        vec![Layout::RowMajorBf16; m.boundary_inputs.len()]
    }

    fn output_layouts(&self, m: &MatchInfo) -> Vec<Layout> {
        vec![Layout::RowMajorBf16; m.boundary_outputs.len()]
    }

    fn is_compute_bound(&self) -> bool {
        true
    }

    fn required_weights(
        &self,
        claimed_tiles: &[TileId],
        fuf: &Fuf,
        program: &Program,
    ) -> Vec<WeightAccessor> {
        let sources: Vec<(WeightId, Option<u64>)> = claimed_tiles
            .iter()
            .filter_map(|t| {
                let n = fuf.get(*t);
                if n.op == OpKind::Gemm {
                    first_weight_ref(n)
                } else {
                    None
                }
            })
            .collect();
        let name = fused_accessor_name(program, &sources);
        vec![WeightAccessor {
            name,
            rust_type: quote! { ::ferrite_kernels::layers::LinearLayer },
            source_weights: sources,
        }]
    }

    fn emit_call(&self, ctx: &EmitCtx) -> TokenStream {
        let gelu_id = *ctx
            .claimed_tiles
            .iter()
            .find(|t| ctx.fuf.get(**t).op == OpKind::Gelu)
            .expect("fused gate/up/gelu/mul claim must contain Gelu");
        let mul_id = *ctx
            .claimed_tiles
            .iter()
            .find(|t| ctx.fuf.get(**t).op == OpKind::Mul)
            .expect("fused gate/up/gelu/mul claim must contain Mul");
        let (gate_id, _) =
            first_tile_input(ctx.fuf.get(gelu_id)).expect("gelu has a tile input — the gate gemm");
        let up_id = ctx
            .claimed_tiles
            .iter()
            .copied()
            .find(|t| ctx.fuf.get(*t).op == OpKind::Gemm && *t != gate_id)
            .expect("claim contains a second Gemm — the up gemm");
        let activation = ctx.input_expr(gate_id, 0);
        let gate_w = first_weight_ref(ctx.fuf.get(gate_id)).expect("gate gemm has a weight");
        let up_w = first_weight_ref(ctx.fuf.get(up_id)).expect("up gemm has a weight");
        let fused_name = fused_accessor_name(ctx.program, &[gate_w, up_w]);
        let weight_expr = ctx.weight_accessor(&fused_name);
        let mul_out = ctx.output_ident(mul_id, 0);
        let gate_up_ident = quote::format_ident!("__fused_gate_up_{}", mul_id.0);
        let intermediate = ctx.bound("intermediate_size") as usize;
        quote! {
            let #gate_up_ident = unsafe {
                (#weight_expr).forward(
                    #activation,
                    &mut device.cublas,
                    &mut device.caching,
                    device.compute_stream,
                )
            };
            let #mul_out = unsafe {
                ::ferrite_kernels::kernels::gelu_and_mul_fused(
                    *#gate_up_ident,
                    #intermediate,
                    &mut device.caching,
                    device.compute_stream,
                )
            };
        }
    }
}

// ── ScalarMulImpl ────────────────────────────────────────────────
//
// Claims a singleton `OpKind::Mul` whose inputs are exactly one
// Tile and one Scalar — the `x * s` pattern used e.g. by Gemma's
// embedding scale (`embed(ids, w) * sqrt(hidden_size)`). Emits an
// in-place `scale_inplace` kernel call (cublas S-axpy / scalEx)
// and move-consumes the upstream OwnedTensor as the output.
//
// Tensor × tensor Muls (SwiGLU / GELU MLP fusions) have Tile+Tile
// inputs and are claimed by `FusedGateUp{Silu,Gelu}MulImpl`; they
// reject any Mul with non-Tile inputs, so the two Impl families
// pick disjoint patterns.

#[derive(Debug, Default)]
pub struct ScalarMulImpl;

impl Implementation for ScalarMulImpl {
    fn name(&self) -> &'static str {
        "scalar_mul_inplace"
    }

    fn target_compatible(&self, _profile: &TargetProfile) -> bool {
        true
    }

    fn matches(&self, fuf: &Fuf, seed: TileId, _profile: &TargetProfile) -> Option<MatchInfo> {
        let node = fuf.get(seed);
        if node.op != OpKind::Mul || node.inputs.len() != 2 {
            return None;
        }
        // Need exactly one Tile and one Scalar input (in either order).
        let mut has_tile = false;
        let mut has_scalar = false;
        for inp in &node.inputs {
            match inp {
                FufInput::Tile { .. } => has_tile = true,
                FufInput::Scalar(_) => has_scalar = true,
                _ => return None,
            }
        }
        if !(has_tile && has_scalar) {
            return None;
        }
        let tile_input: Vec<TileId> = node
            .inputs
            .iter()
            .filter_map(|i| match i {
                FufInput::Tile { id, .. } => Some(*id),
                _ => None,
            })
            .collect();
        Some(MatchInfo {
            claimed_tiles: vec![seed],
            boundary_inputs: tile_input,
            boundary_outputs: vec![seed],
        })
    }

    fn cost_us(&self, _m: &MatchInfo, ctx: &CostCtx) -> f64 {
        // One read + one write over the tensor.
        let numel = ctx.num_tokens() * ctx.bounds.get("hidden_size").copied().unwrap_or(0);
        let bw_gb = ctx.profile.memory_bandwidth_gbps;
        let bytes = 2.0 * numel as f64 * BYTES_PER_ELEM;
        if bw_gb > 0.0 {
            (bytes / (bw_gb * 1e9)) * 1e6
        } else {
            0.0
        }
    }

    fn resources(&self, _m: &MatchInfo) -> Resources {
        Resources::ZERO
    }

    fn launch_kind(&self) -> LaunchKind {
        LaunchKind::HostCallback
    }

    fn supported_input_handoffs(&self) -> &[Handoff] {
        const H: &[Handoff] = &[Handoff::StreamOrder, Handoff::StreamEvent];
        H
    }

    fn supported_output_handoffs(&self) -> &[Handoff] {
        const H: &[Handoff] = &[Handoff::StreamOrder, Handoff::StreamEvent];
        H
    }

    fn input_layouts(&self, m: &MatchInfo) -> Vec<Layout> {
        vec![Layout::RowMajorBf16; m.boundary_inputs.len()]
    }

    fn output_layouts(&self, m: &MatchInfo) -> Vec<Layout> {
        vec![Layout::RowMajorBf16; m.boundary_outputs.len()]
    }

    fn is_compute_bound(&self) -> bool {
        false
    }

    fn consumes_input_tiles(&self, claimed_tiles: &[TileId], fuf: &Fuf) -> Vec<(TileId, u8)> {
        // The kernel mutates the upstream buffer in place and the
        // emit moves it into the output binding — codegen must not
        // schedule a drop for the upstream local.
        let tile = claimed_tiles[0];
        let node = fuf.get(tile);
        let src = node
            .inputs
            .iter()
            .find_map(|i| match i {
                FufInput::Tile { id, slot } => Some((*id, *slot)),
                _ => None,
            })
            .expect("ScalarMul has a Tile input");
        vec![src]
    }

    fn emit_call(&self, ctx: &EmitCtx) -> TokenStream {
        let tile = ctx.primary();
        let out = ctx.output_ident(tile, 0);
        let node = ctx.fuf.get(tile);
        // The Tile input (the tensor to scale) and the Scalar.
        let upstream_slot = node
            .inputs
            .iter()
            .position(|i| matches!(i, FufInput::Tile { .. }))
            .expect("ScalarMul claim has a Tile input");
        let upstream = ctx
            .input_tile_ident(tile, upstream_slot)
            .expect("ScalarMul's Tile input has an ident");
        let scale: f32 = node
            .inputs
            .iter()
            .find_map(|i| match i {
                FufInput::Scalar(v) => Some(*v as f32),
                _ => None,
            })
            .expect("ScalarMul claim has a Scalar input");

        // `scale_inplace` uses cuBLAS S-axpy-like scalEx — mutates
        // the upstream buffer. Move-consume the OwnedTensor so the
        // output binding owns the mutated buffer.
        quote! {
            let #out = unsafe {
                ::ferrite_kernels::kernels::scale_inplace(
                    *#upstream,
                    #scale,
                    &device.cublas,
                );
                #upstream
            };
        }
    }
}

// ── TanhSoftCapImpl ──────────────────────────────────────────────
//
// Singleton Impl claiming [`OpKind::TanhSoftCap`]. Maps to the
// in-place `tanh_softcap_inplace` kernel: `x[i] = cap * tanh(x[i] / cap)`.
//
// The cap scalar is read from the model config under the convention
// key `final_logit_softcapping` (HF naming). Architectures that
// don't cap logits don't emit a `TanhSoftCap` tile in their DSL
// body, so they never reach this Impl. If a DSL body emits the tile
// on a model whose config has no cap, the solver reports it — a
// missing convention field is a hard error at emit time, not a
// silent no-op. (Same posture as the attention softcap: the convention
// field exists or the tile shouldn't be in the FUF.)

#[derive(Debug, Default)]
pub struct TanhSoftCapImpl;

impl Implementation for TanhSoftCapImpl {
    fn name(&self) -> &'static str {
        "tanh_softcap_inplace"
    }

    fn target_compatible(&self, _profile: &TargetProfile) -> bool {
        true
    }

    fn matches(&self, fuf: &Fuf, seed: TileId, _profile: &TargetProfile) -> Option<MatchInfo> {
        single_tile_match(fuf, seed, OpKind::TanhSoftCap)
    }

    fn cost_us(&self, _m: &MatchInfo, ctx: &CostCtx) -> f64 {
        // One bf16 read + one bf16 write over the input tensor.
        let numel = ctx.num_tokens() * ctx.bounds.get("vocab_size").copied().unwrap_or(0);
        let bw_gb = ctx.profile.memory_bandwidth_gbps;
        let bytes = 2.0 * numel as f64 * BYTES_PER_ELEM;
        if bw_gb > 0.0 {
            (bytes / (bw_gb * 1e9)) * 1e6
        } else {
            0.0
        }
    }

    fn resources(&self, _m: &MatchInfo) -> Resources {
        Resources::ZERO
    }

    fn launch_kind(&self) -> LaunchKind {
        LaunchKind::HostCallback
    }

    fn supported_input_handoffs(&self) -> &[Handoff] {
        const H: &[Handoff] = &[Handoff::StreamOrder, Handoff::StreamEvent];
        H
    }

    fn supported_output_handoffs(&self) -> &[Handoff] {
        const H: &[Handoff] = &[Handoff::StreamOrder, Handoff::StreamEvent];
        H
    }

    fn input_layouts(&self, m: &MatchInfo) -> Vec<Layout> {
        vec![Layout::RowMajorBf16; m.boundary_inputs.len()]
    }

    fn output_layouts(&self, m: &MatchInfo) -> Vec<Layout> {
        vec![Layout::RowMajorBf16; m.boundary_outputs.len()]
    }

    fn is_compute_bound(&self) -> bool {
        false
    }

    fn consumes_input_tiles(&self, claimed_tiles: &[TileId], fuf: &Fuf) -> Vec<(TileId, u8)> {
        // The kernel mutates the upstream buffer in place; the emit
        // moves the upstream OwnedTensor into the output binding so
        // the function can return it. Codegen must not schedule a
        // drop for the upstream local.
        let tile = claimed_tiles[0];
        let node = fuf.get(tile);
        let src = node
            .inputs
            .iter()
            .find_map(|i| match i {
                FufInput::Tile { id, slot } => Some((*id, *slot)),
                _ => None,
            })
            .expect("tanh_softcap input is a tile");
        vec![src]
    }

    fn emit_call(&self, ctx: &EmitCtx) -> TokenStream {
        let tile = ctx.primary();
        let out = ctx.output_ident(tile, 0);
        let upstream = ctx
            .input_tile_ident(tile, 0)
            .expect("tanh_softcap input must be a tile-sourced tensor");

        // The cap is a config convention. Missing → the DSL body
        // shouldn't have emitted this tile on this model.
        let cap: f32 = ctx.scalar("final_logit_softcapping").unwrap_or_else(|| {
            panic!(
                "tanh_softcap tile emitted, but model `{}` has no \
                     `final_logit_softcapping` in its config.json",
                ctx.model.source_stem,
            )
        }) as f32;

        // Mutate the upstream buffer in place, then move it into
        // the binding as an OwnedTensor so the emitted forward fn
        // can return it. `tanh_softcap` is terminal (post-lm_head,
        // final logits) — no later tile reads the upstream ident.
        quote! {
            let #out = unsafe {
                ::ferrite_kernels::kernels::tanh_softcap_inplace(
                    *#upstream,
                    #cap,
                    device.compute_stream,
                );
                #upstream
            };
        }
    }
}

// ── AddRefImpl ───────────────────────────────────────────────────
//
// Singleton fallback for `OpKind::Add` tiles whose downstream is
// neither `RmsNorm` (claimed by `FusedAddRmsNormImpl`) nor
// `RmsNorm`-with-scalar-offset (claimed by
// `FusedAddRmsNormWithOffsetImpl`). Necessary for arches whose
// residual stream isn't immediately followed by the next layer's
// pre-norm — Cohere's parallel attn+MLP topology adds the attn
// output and the MLP output into the residual as two separate
// `add(...)` statements with no intervening norm.
//
// Both Add inputs must be tiles (not weights/scalars/externs); the
// scalar-offset case is left to `ScalarOffsetRmsNormImpl`. Emits
// `add_inplace(residual, delta)` mutating the slot-1 (residual)
// buffer; the Add's output is bound as a TensorView alias of that
// same buffer. Convention matches `FusedAddRmsNormImpl`: slot 0 is
// the new contribution, slot 1 is the residual.

#[derive(Debug, Default)]
pub struct AddRefImpl;

impl Implementation for AddRefImpl {
    fn name(&self) -> &'static str {
        "add_ref"
    }

    fn target_compatible(&self, _profile: &TargetProfile) -> bool {
        true
    }

    fn matches(&self, fuf: &Fuf, seed: TileId, _profile: &TargetProfile) -> Option<MatchInfo> {
        let node = fuf.get(seed);
        if node.op != OpKind::Add || node.inputs.len() != 2 {
            return None;
        }
        // Both inputs must be Tiles. ScalarOffset Adds (Weight + Scalar)
        // are claimed by `ScalarOffsetRmsNormImpl`; mixed Tile+Weight
        // shouldn't occur for residual adds.
        let slot_a = match &node.inputs[0] {
            FufInput::Tile { id, .. } => *id,
            _ => return None,
        };
        let slot_b = match &node.inputs[1] {
            FufInput::Tile { id, .. } => *id,
            _ => return None,
        };
        // Defer to the (Add, RmsNorm) fusion when the downstream is a
        // RmsNorm consuming this Add's output. Cohere's LayerNorm has
        // no fusion impl yet, so we DO claim Adds whose downstream is
        // a LayerNorm (the layer-end residuals before the next layer's
        // pre-norm). When a `FusedAddCohereLayerNormImpl` lands, mirror
        // this gate.
        let downstream_is_rmsnorm = fuf
            .nodes
            .iter()
            .any(|n| n.op == OpKind::RmsNorm && consumes_tile(n, seed));
        if downstream_is_rmsnorm {
            return None;
        }
        Some(MatchInfo {
            claimed_tiles: vec![seed],
            boundary_inputs: vec![slot_a, slot_b],
            boundary_outputs: vec![seed],
        })
    }

    fn cost_us(&self, _m: &MatchInfo, ctx: &CostCtx) -> f64 {
        // Read residual + delta, write residual: 3 reads/writes per
        // element, bandwidth-bound.
        let m = ctx.num_tokens() as f64;
        let hidden = ctx.bounds.get("hidden_size").copied().unwrap_or(0) as f64;
        let bw_gb = ctx.profile.memory_bandwidth_gbps;
        let bytes = 3.0 * m * hidden * BYTES_PER_ELEM;
        if bw_gb > 0.0 {
            (bytes / (bw_gb * 1e9)) * 1e6
        } else {
            0.0
        }
    }

    fn resources(&self, _m: &MatchInfo) -> Resources {
        Resources::ZERO
    }

    fn launch_kind(&self) -> LaunchKind {
        LaunchKind::HostCallback
    }

    fn supported_input_handoffs(&self) -> &[Handoff] {
        const H: &[Handoff] = &[Handoff::StreamOrder, Handoff::StreamEvent];
        H
    }

    fn supported_output_handoffs(&self) -> &[Handoff] {
        const H: &[Handoff] = &[Handoff::StreamOrder, Handoff::StreamEvent];
        H
    }

    fn input_layouts(&self, m: &MatchInfo) -> Vec<Layout> {
        vec![Layout::RowMajorBf16; m.boundary_inputs.len()]
    }

    fn output_layouts(&self, m: &MatchInfo) -> Vec<Layout> {
        vec![Layout::RowMajorBf16; m.boundary_outputs.len()]
    }

    #[allow(clippy::type_complexity)]
    fn output_alias(
        &self,
        claimed_tiles: &[TileId],
        fuf: &Fuf,
    ) -> Vec<((TileId, u8), Option<(TileId, u8)>)> {
        // Output aliases the slot-1 input (the residual) since
        // `add_inplace` mutates that buffer directly.
        let add_id = claimed_tiles[0];
        let node = fuf.get(add_id);
        let residual_src = match node.inputs.get(1) {
            Some(FufInput::Tile { id, slot }) => Some((*id, *slot)),
            _ => None,
        };
        vec![((add_id, 0), residual_src)]
    }

    fn emit_call(&self, ctx: &EmitCtx) -> TokenStream {
        let add_id = ctx.primary();
        let out = ctx.output_ident(add_id, 0);
        // slot 0 = delta (new contribution), slot 1 = residual (mutated).
        let delta_upstream = ctx
            .input_tile_ident(add_id, 0)
            .expect("Add input 0 (delta) is a Tile");
        let residual_upstream = ctx
            .input_tile_ident(add_id, 1)
            .expect("Add input 1 (residual) is a Tile");
        quote! {
            unsafe {
                ::ferrite_kernels::kernels::add_inplace(
                    *#residual_upstream,
                    *#delta_upstream,
                    device.compute_stream,
                );
            }
            // The Add's logical output is the post-mutation residual
            // buffer — bind as a TensorView alias so downstream tiles
            // read it without an extra copy.
            let #out = unsafe { (*#residual_upstream).as_view() };
        }
    }
}

// ── FusedAddRmsNormImpl ──────────────────────────────────────────
//
// Multi-tile impl claiming `(Add, RmsNorm)` where the RmsNorm's
// input is the Add's output. Maps to
// `ferrite_kernels::fused_add_rms_norm_inplace`, which:
//   residual += delta          (in-place on residual)
//   input_buf = norm(residual) (in-place on delta's buffer)
// returning (normed_ptr_aliasing_delta, residual_ptr).
//
// Emit binds the `RmsNorm` output to the post-mutation delta buffer
// (as a `TensorView` alias) and the `Add` output to the post-mutation
// residual buffer (also as a `TensorView` alias). Both aliases borrow
// off the ambient OwnedTensor bindings that held `delta` and
// `residual` before the call — those OwnedTensors stay alive in the
// forward fn scope for the kernel's async lifetime.
//
// Seeded on the `Add`: topologically upstream, guaranteed to be
// processed before the solver reaches the `RmsNorm`. A lone `Add`
// with no downstream `RmsNorm` (does not occur in Llama/Qwen2)
// returns `None` → `UnclaimedTile` library gap.

#[derive(Debug, Default)]
pub struct FusedAddRmsNormImpl;

impl Implementation for FusedAddRmsNormImpl {
    fn name(&self) -> &'static str {
        "fused_add_rms_norm"
    }

    fn target_compatible(&self, _profile: &TargetProfile) -> bool {
        true
    }

    fn matches(&self, fuf: &Fuf, seed: TileId, _profile: &TargetProfile) -> Option<MatchInfo> {
        let add_node = fuf.get(seed);
        if add_node.op != OpKind::Add {
            return None;
        }
        // Residual-stream pattern only: both Add operands must be
        // tiles (delta + residual, both tensor-valued). An Add with
        // a `Scalar` or `Weight` input is a different shape (e.g.
        // Gemma's `w + 1.0` feeding rmsnorm) — hand it to
        // [`ScalarOffsetRmsNormImpl`] instead of mis-fusing here.
        if !add_node
            .inputs
            .iter()
            .all(|i| matches!(i, FufInput::Tile { .. }))
        {
            return None;
        }
        // Find an immediate RmsNorm consumer. "Immediate" in the
        // topological sense: any RmsNorm whose first input is this
        // Add's output. More than one such consumer is possible in
        // theory (not in Llama/Qwen2); we claim the first we find.
        let rmsnorm_node = fuf
            .nodes
            .iter()
            .find(|n| n.op == OpKind::RmsNorm && consumes_tile(n, seed))?;

        let mut claimed = [seed, rmsnorm_node.id];
        claimed.sort();
        let claimed = claimed.to_vec();

        // Boundary inputs: the Add's two upstream tiles (delta and
        // residual) — RmsNorm's weight input is a Weight, not a Tile.
        let boundary_inputs: Vec<TileId> = add_node
            .inputs
            .iter()
            .filter_map(|i| match i {
                FufInput::Tile { id, .. } => Some(*id),
                _ => None,
            })
            .collect();

        Some(MatchInfo {
            claimed_tiles: claimed,
            boundary_inputs,
            // Both outputs are live downstream: RmsNorm's normed
            // feeds the post-norm compute; Add's updated-residual
            // feeds the next residual stream.
            boundary_outputs: vec![seed, rmsnorm_node.id],
        })
    }

    fn cost_us(&self, _m: &MatchInfo, ctx: &CostCtx) -> f64 {
        // Bandwidth-bound: one read of delta + residual + weight,
        // one write of residual + normed. bf16 everywhere.
        let num_tokens = ctx.num_tokens() as f64;
        let hidden = ctx.bounds.get("hidden_size").copied().unwrap_or(0) as f64;
        let bw_gb = ctx.profile.memory_bandwidth_gbps;
        // 2 reads (delta, residual) + 2 writes (residual, normed) of
        // [num_tokens, hidden] bf16, plus weight [hidden] bf16 read.
        let bytes = (4.0 * num_tokens * hidden + hidden) * BYTES_PER_ELEM;
        if bw_gb > 0.0 {
            (bytes / (bw_gb * 1e9)) * 1e6
        } else {
            0.0
        }
    }

    fn resources(&self, _m: &MatchInfo) -> Resources {
        Resources::ZERO
    }

    fn launch_kind(&self) -> LaunchKind {
        LaunchKind::HostCallback
    }

    fn supported_input_handoffs(&self) -> &[Handoff] {
        const H: &[Handoff] = &[Handoff::StreamOrder, Handoff::StreamEvent];
        H
    }

    fn supported_output_handoffs(&self) -> &[Handoff] {
        const H: &[Handoff] = &[Handoff::StreamOrder, Handoff::StreamEvent];
        H
    }

    fn input_layouts(&self, m: &MatchInfo) -> Vec<Layout> {
        vec![Layout::RowMajorBf16; m.boundary_inputs.len()]
    }

    fn output_layouts(&self, m: &MatchInfo) -> Vec<Layout> {
        vec![Layout::RowMajorBf16; m.boundary_outputs.len()]
    }

    // Default `required_weights` suffices: the only weight input
    // across claimed tiles belongs to the RmsNorm, and it's consumed
    // by an `OpKind::RmsNorm` tile → typed as `RmsNorm` by
    // `rust_type_for_weight_consumed_by`.

    #[allow(clippy::type_complexity)]
    fn output_alias(
        &self,
        claimed_tiles: &[TileId],
        fuf: &Fuf,
    ) -> Vec<((TileId, u8), Option<(TileId, u8)>)> {
        // Both outputs are TensorView aliases of the Add tile's
        // upstream Tile inputs. The kernel mutates delta's buffer
        // in place to produce the rmsnorm output, and residual's
        // buffer to produce the updated residual; the bindings
        // simply alias those upstream OwnedTensors.
        let add_id = *claimed_tiles
            .iter()
            .find(|t| fuf.get(**t).op == OpKind::Add)
            .expect("claim contains Add");
        let rmsnorm_id = *claimed_tiles
            .iter()
            .find(|t| fuf.get(**t).op == OpKind::RmsNorm)
            .expect("claim contains RmsNorm");
        let add_node = fuf.get(add_id);
        let delta_src = match add_node.inputs.first() {
            Some(FufInput::Tile { id, slot }) => (*id, *slot),
            _ => panic!("Add input 0 (delta) must be a tile"),
        };
        let residual_src = match add_node.inputs.get(1) {
            Some(FufInput::Tile { id, slot }) => (*id, *slot),
            _ => panic!("Add input 1 (residual) must be a tile"),
        };
        vec![
            ((rmsnorm_id, 0), Some(delta_src)),
            ((add_id, 0), Some(residual_src)),
        ]
    }

    fn emit_call(&self, ctx: &EmitCtx) -> TokenStream {
        // Identify Add + RmsNorm tiles from the claim.
        let add_id = *ctx
            .claimed_tiles
            .iter()
            .find(|t| ctx.fuf.get(**t).op == OpKind::Add)
            .expect("fused add/rmsnorm claim must contain Add");
        let rmsnorm_id = *ctx
            .claimed_tiles
            .iter()
            .find(|t| ctx.fuf.get(**t).op == OpKind::RmsNorm)
            .expect("fused add/rmsnorm claim must contain RmsNorm");

        // Add's inputs: delta (slot 0), residual (slot 1). Reach past
        // `input_expr`'s `(*_).as_view()` wrapper to get the raw
        // upstream idents so we can (a) feed GpuTensor to the kernel
        // and (b) bind TensorView aliases to the same storage.
        //
        // Aliasing (rather than moving) is necessary because the
        // residual stream's first iteration sees an upstream tile
        // (the embed output) with multiple consumers — the codegen
        // drop pass handles cleanup of those upstream OwnedTensors
        // after their last cross-subgraph use.
        let delta_upstream = ctx
            .input_tile_ident(add_id, 0)
            .expect("Add input 0 (delta) is a Tile");
        let residual_upstream = ctx
            .input_tile_ident(add_id, 1)
            .expect("Add input 1 (residual) is a Tile");

        // RmsNorm's weight accessor — uses the default
        // `required_weights` declaration name.
        let node = ctx.fuf.get(rmsnorm_id);
        let (weight_id, weight_idx) = node
            .inputs
            .iter()
            .find_map(|i| match i {
                FufInput::Weight { id, index, .. } => Some((*id, *index)),
                _ => None,
            })
            .expect("RmsNorm has a weight input");
        let weight_name = weight_field_name(ctx.program, weight_id, weight_idx);
        let weight_expr = ctx.weight_accessor(&weight_name);

        let add_out = ctx.output_ident(add_id, 0);
        let rmsnorm_out = ctx.output_ident(rmsnorm_id, 0);

        quote! {
            // Fused `residual += delta; normed = norm(residual) * w`.
            // Both buffers are mutated in place; the upstream
            // OwnedTensors stay the owners and the alias bindings
            // below let downstream tiles read them as views.
            unsafe {
                let _ = ::ferrite_kernels::kernels::fused_add_rms_norm_inplace(
                    *#delta_upstream,
                    *#residual_upstream,
                    (#weight_expr).weight,
                    (#weight_expr).eps,
                    device.compute_stream,
                );
            }
            let #rmsnorm_out = unsafe { (*#delta_upstream).as_view() };
            let #add_out = unsafe { (*#residual_upstream).as_view() };
        }
    }
}

// ── FusedAddRmsNormWithOffsetImpl ────────────────────────────────
//
// Claims three tiles: a residual-stream Add (two Tile inputs), a
// scalar-offset Add (Weight + Scalar inputs), and a RmsNorm that
// consumes the residual-Add at slot 0 and the scalar-Add at slot 1.
// Emits a single `fused_add_rms_norm_inplace` call with the
// scalar as the kernel's `weight_offset` param.
//
// This is the Gemma2 mid-layer pattern:
//   hidden_states = add(delta, hidden_states);                 // residual
//   pre_ffwd     = rmsnorm(hidden_states, weight[i] + 1.0);    // norm
// and the cross-iteration pattern where the end-of-layer residual
// Add feeds the next iteration's pre-norm (or the final norm
// after the last iteration).
//
// Without this Impl, `FusedAddRmsNormImpl` absorbs the (Add,
// RmsNorm) pair, orphaning the scalar-offset Add with no other
// claim possible — the DP reports UnclaimedTile.

#[derive(Debug, Default)]
pub struct FusedAddRmsNormWithOffsetImpl;

impl Implementation for FusedAddRmsNormWithOffsetImpl {
    fn name(&self) -> &'static str {
        "fused_add_rms_norm_with_offset"
    }

    fn target_compatible(&self, _profile: &TargetProfile) -> bool {
        true
    }

    fn matches(&self, fuf: &Fuf, seed: TileId, _profile: &TargetProfile) -> Option<MatchInfo> {
        let residual_add = fuf.get(seed);
        if residual_add.op != OpKind::Add {
            return None;
        }
        // Residual-stream Add: both inputs must be tiles.
        if !residual_add
            .inputs
            .iter()
            .all(|i| matches!(i, FufInput::Tile { .. }))
        {
            return None;
        }

        // Find a RmsNorm whose input (slot 0) consumes this Add and
        // whose weight (slot 1) consumes a DIFFERENT Add that is a
        // scalar-offset Add (Weight + Scalar inputs).
        for rms in &fuf.nodes {
            if rms.op != OpKind::RmsNorm || rms.inputs.len() < 2 {
                continue;
            }
            let reads_residual_at_0 =
                matches!(rms.inputs[0], FufInput::Tile { id, .. } if id == seed);
            if !reads_residual_at_0 {
                continue;
            }
            let scalar_add_id = match rms.inputs[1] {
                FufInput::Tile { id, .. } => id,
                _ => continue,
            };
            let scalar_add = fuf.get(scalar_add_id);
            if scalar_add.op != OpKind::Add {
                continue;
            }
            // Scalar-offset Add: exactly one Weight and one Scalar.
            let mut has_weight = false;
            let mut has_scalar = false;
            let mut ok = true;
            for inp in &scalar_add.inputs {
                match inp {
                    FufInput::Weight { .. } => has_weight = true,
                    FufInput::Scalar(_) => has_scalar = true,
                    _ => {
                        ok = false;
                        break;
                    }
                }
            }
            if !(ok && has_weight && has_scalar) {
                continue;
            }

            let mut claimed = [seed, scalar_add_id, rms.id];
            claimed.sort();
            let claimed = claimed.to_vec();

            // Boundary inputs: the residual-Add's two upstream tiles
            // (delta and residual). Scalar-Add's Weight/Scalar are
            // not tile inputs, so nothing else crosses the boundary.
            let boundary_inputs: Vec<TileId> = residual_add
                .inputs
                .iter()
                .filter_map(|i| match i {
                    FufInput::Tile { id, .. } => Some(*id),
                    _ => None,
                })
                .collect();

            return Some(MatchInfo {
                claimed_tiles: claimed,
                boundary_inputs,
                // Both outputs are live downstream: RmsNorm's normed
                // feeds post-norm compute; residual-Add's updated
                // residual feeds the next residual stream.
                boundary_outputs: vec![seed, rms.id],
            });
        }
        None
    }

    fn cost_us(&self, _m: &MatchInfo, ctx: &CostCtx) -> f64 {
        // Same bandwidth cost as the non-offset variant; the `+1`
        // is an extra fp32 fadd per weight element — free on a
        // memory-bound kernel.
        let num_tokens = ctx.num_tokens() as f64;
        let hidden = ctx.bounds.get("hidden_size").copied().unwrap_or(0) as f64;
        let bw_gb = ctx.profile.memory_bandwidth_gbps;
        let bytes = (4.0 * num_tokens * hidden + hidden) * BYTES_PER_ELEM;
        if bw_gb > 0.0 {
            (bytes / (bw_gb * 1e9)) * 1e6
        } else {
            0.0
        }
    }

    fn resources(&self, _m: &MatchInfo) -> Resources {
        Resources::ZERO
    }

    fn launch_kind(&self) -> LaunchKind {
        LaunchKind::HostCallback
    }

    fn supported_input_handoffs(&self) -> &[Handoff] {
        const H: &[Handoff] = &[Handoff::StreamOrder, Handoff::StreamEvent];
        H
    }

    fn supported_output_handoffs(&self) -> &[Handoff] {
        const H: &[Handoff] = &[Handoff::StreamOrder, Handoff::StreamEvent];
        H
    }

    fn input_layouts(&self, m: &MatchInfo) -> Vec<Layout> {
        vec![Layout::RowMajorBf16; m.boundary_inputs.len()]
    }

    fn output_layouts(&self, m: &MatchInfo) -> Vec<Layout> {
        vec![Layout::RowMajorBf16; m.boundary_outputs.len()]
    }

    fn is_compute_bound(&self) -> bool {
        false
    }

    fn required_weights(
        &self,
        claimed_tiles: &[TileId],
        fuf: &Fuf,
        program: &Program,
    ) -> Vec<WeightAccessor> {
        // The scalar-offset Add's Weight input is the actual rmsnorm
        // weight. Match the default rmsnorm naming scheme so the
        // codegen's RmsNorm::load path fires.
        let scalar_add_id = *claimed_tiles
            .iter()
            .find(|t| {
                let n = fuf.get(**t);
                n.op == OpKind::Add && n.inputs.iter().any(|i| matches!(i, FufInput::Scalar(_)))
            })
            .expect("claim contains a scalar-offset Add");
        let scalar_add = fuf.get(scalar_add_id);
        let (weight_id, weight_idx) = scalar_add
            .inputs
            .iter()
            .find_map(|i| match i {
                FufInput::Weight { id, index, .. } => Some((*id, *index)),
                _ => None,
            })
            .expect("scalar-offset Add has a Weight input");
        let name = weight_field_name(program, weight_id, weight_idx);
        vec![WeightAccessor {
            name,
            rust_type: quote! { ::ferrite_kernels::layers::RmsNorm },
            source_weights: vec![(weight_id, weight_idx)],
        }]
    }

    #[allow(clippy::type_complexity)]
    fn output_alias(
        &self,
        claimed_tiles: &[TileId],
        fuf: &Fuf,
    ) -> Vec<((TileId, u8), Option<(TileId, u8)>)> {
        // Same alias relationships as `FusedAddRmsNormImpl`: the
        // residual_out aliases the residual upstream's buffer (post
        // in-place add) and the rmsnorm_out aliases the delta
        // upstream's buffer (post in-place norm).
        let residual_add_id = *claimed_tiles
            .iter()
            .find(|t| {
                let n = fuf.get(**t);
                n.op == OpKind::Add && n.inputs.iter().all(|i| matches!(i, FufInput::Tile { .. }))
            })
            .expect("claim contains a residual-stream Add");
        let rmsnorm_id = *claimed_tiles
            .iter()
            .find(|t| fuf.get(**t).op == OpKind::RmsNorm)
            .expect("claim contains a RmsNorm");
        let add_node = fuf.get(residual_add_id);
        let delta_src = match add_node.inputs.first() {
            Some(FufInput::Tile { id, slot }) => (*id, *slot),
            _ => panic!("residual Add input 0 (delta) must be a Tile"),
        };
        let residual_src = match add_node.inputs.get(1) {
            Some(FufInput::Tile { id, slot }) => (*id, *slot),
            _ => panic!("residual Add input 1 (residual) must be a Tile"),
        };
        vec![
            ((rmsnorm_id, 0), Some(delta_src)),
            ((residual_add_id, 0), Some(residual_src)),
        ]
    }

    fn emit_call(&self, ctx: &EmitCtx) -> TokenStream {
        // Identify tiles by op kind + inputs.
        let residual_add_id = *ctx
            .claimed_tiles
            .iter()
            .find(|t| {
                let n = ctx.fuf.get(**t);
                n.op == OpKind::Add && n.inputs.iter().all(|i| matches!(i, FufInput::Tile { .. }))
            })
            .expect("claim contains a residual-stream Add");
        let scalar_add_id = *ctx
            .claimed_tiles
            .iter()
            .find(|t| {
                let n = ctx.fuf.get(**t);
                n.op == OpKind::Add && n.inputs.iter().any(|i| matches!(i, FufInput::Scalar(_)))
            })
            .expect("claim contains a scalar-offset Add");
        let rmsnorm_id = *ctx
            .claimed_tiles
            .iter()
            .find(|t| ctx.fuf.get(**t).op == OpKind::RmsNorm)
            .expect("claim contains a RmsNorm");

        // Residual-Add inputs: delta (slot 0), residual (slot 1).
        let delta_upstream = ctx
            .input_tile_ident(residual_add_id, 0)
            .expect("residual Add input 0 (delta) is a Tile");
        let residual_upstream = ctx
            .input_tile_ident(residual_add_id, 1)
            .expect("residual Add input 1 (residual) is a Tile");

        // Scalar-Add: extract the Weight and the Scalar.
        let scalar_add_node = ctx.fuf.get(scalar_add_id);
        let offset: f32 = scalar_add_node
            .inputs
            .iter()
            .find_map(|i| match i {
                FufInput::Scalar(v) => Some(*v as f32),
                _ => None,
            })
            .expect("scalar-offset Add has a Scalar input");
        let (weight_id, weight_idx) = scalar_add_node
            .inputs
            .iter()
            .find_map(|i| match i {
                FufInput::Weight { id, index, .. } => Some((*id, *index)),
                _ => None,
            })
            .expect("scalar-offset Add has a Weight input");
        let weight_name = weight_field_name(ctx.program, weight_id, weight_idx);
        let weight_expr = ctx.weight_accessor(&weight_name);

        // Output aliases for downstream tiles.
        let residual_out = ctx.output_ident(residual_add_id, 0);
        let rmsnorm_out = ctx.output_ident(rmsnorm_id, 0);

        quote! {
            unsafe {
                let _ = ::ferrite_kernels::kernels::fused_add_rms_norm_inplace_with_offset(
                    *#delta_upstream,
                    *#residual_upstream,
                    (#weight_expr).weight,
                    (#weight_expr).eps,
                    #offset,
                    device.compute_stream,
                );
            }
            let #rmsnorm_out = unsafe { (*#delta_upstream).as_view() };
            let #residual_out = unsafe { (*#residual_upstream).as_view() };
        }
    }
}

// ── ScalarOffsetRmsNormImpl ──────────────────────────────────────
//
// Claims `(Add, RmsNorm)` where the Add has one [`FufInput::Weight`]
// and one [`FufInput::Scalar`] input, and the RmsNorm consumes the
// Add's output. Models like Gemma2 store their rmsnorm weights
// zero-init and treat the forward as `y = x * (1 + w) / rms(x)`;
// the DSL expresses that as `rmsnorm(x, weight[layer] + 1.0)`,
// which lowers to this Add + RmsNorm pair.
//
// Emit: a single `rms_norm` kernel call where the scalar rides as
// the kernel's `weight_offset` param (one fp32 fadd per weight
// element, memory-bound — effectively free vs. the Llama-style
// `y = x * w / rms(x)`).
//
// Residual-stream `(Add, RmsNorm)` continues to route through
// [`FusedAddRmsNormImpl`], which rejects Adds with non-Tile inputs.

#[derive(Debug, Default)]
pub struct ScalarOffsetRmsNormImpl;

impl Implementation for ScalarOffsetRmsNormImpl {
    fn name(&self) -> &'static str {
        "scalar_offset_rms_norm"
    }

    fn target_compatible(&self, _profile: &TargetProfile) -> bool {
        true
    }

    fn matches(&self, fuf: &Fuf, seed: TileId, _profile: &TargetProfile) -> Option<MatchInfo> {
        let add_node = fuf.get(seed);
        if add_node.op != OpKind::Add {
            return None;
        }
        // Need exactly one Weight input and one Scalar input.
        let mut has_weight = false;
        let mut has_scalar = false;
        for inp in &add_node.inputs {
            match inp {
                FufInput::Weight { .. } => has_weight = true,
                FufInput::Scalar(_) => has_scalar = true,
                _ => return None,
            }
        }
        if !(has_weight && has_scalar) {
            return None;
        }
        let rmsnorm_node = fuf
            .nodes
            .iter()
            .find(|n| n.op == OpKind::RmsNorm && consumes_tile(n, seed))?;
        // The RmsNorm must consume the Add at the weight position
        // (slot 1 of rmsnorm(x, w)). Slot 0 is the input tensor.
        let rmsnorm_weight_is_add = matches!(
            rmsnorm_node.inputs.get(1),
            Some(FufInput::Tile { id, .. }) if *id == seed
        );
        if !rmsnorm_weight_is_add {
            return None;
        }

        let mut claimed = [seed, rmsnorm_node.id];
        claimed.sort();
        let claimed = claimed.to_vec();

        // Boundary inputs: the RmsNorm's input tensor (slot 0).
        let boundary_inputs: Vec<TileId> = rmsnorm_node
            .inputs
            .iter()
            .take(1)
            .filter_map(|i| match i {
                FufInput::Tile { id, .. } => Some(*id),
                _ => None,
            })
            .collect();

        Some(MatchInfo {
            claimed_tiles: claimed,
            boundary_inputs,
            boundary_outputs: vec![rmsnorm_node.id],
        })
    }

    fn cost_us(&self, _m: &MatchInfo, ctx: &CostCtx) -> f64 {
        // Same cost shape as a plain rmsnorm: one read of x +
        // weight, one write of y. The +offset is free (fp32 fadd
        // per element, memory-bound).
        let num_tokens = ctx.num_tokens() as f64;
        let hidden = ctx.bounds.get("hidden_size").copied().unwrap_or(0) as f64;
        let bw_gb = ctx.profile.memory_bandwidth_gbps;
        let bytes = (2.0 * num_tokens * hidden + hidden) * BYTES_PER_ELEM;
        if bw_gb > 0.0 {
            (bytes / (bw_gb * 1e9)) * 1e6
        } else {
            0.0
        }
    }

    fn resources(&self, _m: &MatchInfo) -> Resources {
        Resources::ZERO
    }

    fn launch_kind(&self) -> LaunchKind {
        LaunchKind::HostCallback
    }

    fn supported_input_handoffs(&self) -> &[Handoff] {
        const H: &[Handoff] = &[Handoff::StreamOrder, Handoff::StreamEvent];
        H
    }

    fn supported_output_handoffs(&self) -> &[Handoff] {
        const H: &[Handoff] = &[Handoff::StreamOrder, Handoff::StreamEvent];
        H
    }

    fn input_layouts(&self, m: &MatchInfo) -> Vec<Layout> {
        vec![Layout::RowMajorBf16; m.boundary_inputs.len()]
    }

    fn output_layouts(&self, m: &MatchInfo) -> Vec<Layout> {
        vec![Layout::RowMajorBf16; m.boundary_outputs.len()]
    }

    fn is_compute_bound(&self) -> bool {
        false
    }

    fn required_weights(
        &self,
        claimed_tiles: &[TileId],
        fuf: &Fuf,
        program: &Program,
    ) -> Vec<WeightAccessor> {
        // The Add's Weight input is the actual rmsnorm weight; the
        // downstream consumer (RmsNorm tile) names its type. Declare
        // one accessor matching the default rmsnorm naming scheme
        // so the codegen's `RmsNorm::load` path fires.
        let add_id = *claimed_tiles
            .iter()
            .find(|t| fuf.get(**t).op == OpKind::Add)
            .expect("claim contains an Add");
        let add_node = fuf.get(add_id);
        let (weight_id, weight_idx) = add_node
            .inputs
            .iter()
            .find_map(|i| match i {
                FufInput::Weight { id, index, .. } => Some((*id, *index)),
                _ => None,
            })
            .expect("ScalarOffsetRmsNorm's Add has a Weight input");
        let name = weight_field_name(program, weight_id, weight_idx);
        vec![WeightAccessor {
            name,
            rust_type: quote! { ::ferrite_kernels::layers::RmsNorm },
            source_weights: vec![(weight_id, weight_idx)],
        }]
    }

    fn emit_call(&self, ctx: &EmitCtx) -> TokenStream {
        let add_id = *ctx
            .claimed_tiles
            .iter()
            .find(|t| ctx.fuf.get(**t).op == OpKind::Add)
            .expect("claim contains an Add");
        let rmsnorm_id = *ctx
            .claimed_tiles
            .iter()
            .find(|t| ctx.fuf.get(**t).op == OpKind::RmsNorm)
            .expect("claim contains a RmsNorm");

        let add_node = ctx.fuf.get(add_id);
        // The scalar literal rides on the Add's inputs.
        let offset: f32 = add_node
            .inputs
            .iter()
            .find_map(|i| match i {
                FufInput::Scalar(v) => Some(*v as f32),
                _ => None,
            })
            .expect("ScalarOffsetRmsNorm's Add has a Scalar input");

        // The RmsNorm consumes the Add at slot 1 (the weight position).
        // Slot 0 is the input tensor — we pass that straight through.
        let x = ctx.input_expr(rmsnorm_id, 0);
        // Build a weight accessor expression using the same name the
        // `required_weights` declaration emitted.
        let (weight_id, weight_idx) = add_node
            .inputs
            .iter()
            .find_map(|i| match i {
                FufInput::Weight { id, index, .. } => Some((*id, *index)),
                _ => None,
            })
            .expect("ScalarOffsetRmsNorm's Add has a Weight input");
        let name = weight_field_name(ctx.program, weight_id, weight_idx);
        let weight_expr = ctx.weight_accessor(&name);

        let out = ctx.output_ident(rmsnorm_id, 0);

        quote! {
            let #out = unsafe {
                ::ferrite_kernels::kernels::rms_norm_with_offset(
                    *(#x),
                    (#weight_expr).weight,
                    (#weight_expr).eps,
                    #offset,
                    &mut device.caching,
                    device.compute_stream,
                )
            };
        }
    }
}

// ── FusedQkvRopeCacheImpl ────────────────────────────────────────
//
// Multi-tile impl claiming `(Gemm, Gemm, Gemm, RopeAppend)` where:
//   - the three Gemms share the same activation Tile-input, and
//   - each Gemm's output feeds one of the RopeAppend's first three
//     Tile-input slots (slot 0 = q, 1 = k, 2 = v).
//
// Maps to a fused cuBLAS QKV GEMM producing a packed
// `[num_tokens, q_size + 2*kv_size]` buffer, followed by
// `ferrite_kernels::fused_qkv_rope_cache` which applies RoPE to Q and
// K, writes K/V to the paged cache at the matched layer, and returns
// Q as an OwnedTensor. Structural match, no "GemmQ/K/V" phase tags.
//
// Declared WeightAccessor: one packed `LinearLayer` covering all
// three Q/K/V projections. Users concatenate the three weights at
// load time. Eliminates the old cross-impl contract between separate
// Gemm and QkvSplit impls.
//
// Seed on Gemm (upstream in topo order). Matcher fires at any of the
// three Q/K/V gemms; whichever seeds first wins (topo order), and the
// other two are swallowed by the same claim.

#[derive(Debug, Default)]
pub struct FusedQkvRopeCacheImpl;

/// Read the layer index captured by the DSL's `kv_cache[layer]`
/// reference on a RopeAppend tile. Returns `None` if the tile has
/// no KvCache extern input (shouldn't happen for real RopeAppends).
fn rope_kv_cache_layer(node: &crate::fuf::FufNode) -> Option<u64> {
    node.inputs.iter().find_map(|i| match i {
        FufInput::Extern {
            kind: crate::classified::ExternKind::KvCache,
            index: Some(layer),
        } => Some(*layer),
        _ => None,
    })
}

/// True for either rope-append flavor — the NeoX-style `RopeAppend`
/// or the Cohere-style `RopeAppendInterleaved`. Used by the QKV+rope
/// fusion matchers and the singleton fallback so a single impl claims
/// both flavors and dispatches to the right kernel at emit time.
fn is_rope_append_op(op: OpKind) -> bool {
    matches!(op, OpKind::RopeAppend | OpKind::RopeAppendInterleaved)
}

/// True iff the layer at `layer` uses the interleaved rope flavor,
/// determined by finding the layer's rope-append tile in the FUF and
/// inspecting its `OpKind`. Used by the singleton attention impls so
/// the FA-2 call sees `is_rotary_interleaved=true` whenever the layer
/// upstream rope wrote interleaved-pattern Q/K to the cache.
fn layer_rope_is_interleaved(fuf: &Fuf, layer: u64) -> bool {
    fuf.nodes
        .iter()
        .any(|n| n.op == OpKind::RopeAppendInterleaved && rope_kv_cache_layer(n) == Some(layer))
}

/// Resolve a tile to `(gemm_tile, maybe_bias_add_tile)`:
/// - `(g, None)`    if `tile` is directly a `Gemm`.
/// - `(g, Some(b))` if `tile` is a `BiasAdd` whose first tile input
///   is a `Gemm`.
/// - `None`         otherwise.
///
/// Used by the QKV-fused impls (`FusedQkvRopeCacheImpl`,
/// `FusedQkvRopePrefillImpl`) so that the DSL can write
/// `gemm → bias_add → rope_append` (Qwen2/Qwen3 style with explicit
/// QKV bias) or `gemm → rope_append` (Llama style, no bias) and have
/// the same fusion claim both shapes.
fn unwrap_gemm_through_bias(fuf: &Fuf, tile: TileId) -> Option<(TileId, Option<TileId>)> {
    let node = fuf.get(tile);
    match node.op {
        OpKind::Gemm => Some((tile, None)),
        OpKind::BiasAdd => {
            let upstream = node.inputs.iter().find_map(|i| match i {
                FufInput::Tile { id, .. } => Some(*id),
                _ => None,
            })?;
            if fuf.get(upstream).op == OpKind::Gemm {
                Some((upstream, Some(tile)))
            } else {
                None
            }
        }
        _ => None,
    }
}

/// True if this RopeAppend tile's first three tile-inputs all resolve
/// (directly or through `BiasAdd`) to three Gemms sharing the same
/// activation with uniform bias presence — the exact pattern
/// `FusedQkvRope{Cache,Prefill}Impl::matches` requires. Used by
/// `RopeAppendRefImpl` to defer to the fused impl whenever it would
/// match (Llama, Qwen2) and only claim standalone when the pattern
/// is broken by intervening ops (Qwen3's per-head Q/K rmsnorms).
fn rope_append_has_fused_qkv_upstream(fuf: &Fuf, rope_tile: TileId) -> bool {
    let node = fuf.get(rope_tile);
    if !is_rope_append_op(node.op) || node.inputs.len() < 3 {
        return false;
    }
    let qkv_raw: Vec<TileId> = node
        .inputs
        .iter()
        .take(3)
        .filter_map(|i| match i {
            FufInput::Tile { id, .. } => Some(*id),
            _ => None,
        })
        .collect();
    if qkv_raw.len() != 3 {
        return false;
    }
    let Some(resolved): Option<Vec<(TileId, Option<TileId>)>> = qkv_raw
        .iter()
        .map(|t| unwrap_gemm_through_bias(fuf, *t))
        .collect()
    else {
        return false;
    };
    let biased = resolved[0].1.is_some();
    if resolved.iter().any(|r| r.1.is_some() != biased) {
        return false;
    }
    let gemms: Vec<TileId> = resolved.iter().map(|r| r.0).collect();
    let Some(act) = first_tile_input(fuf.get(gemms[0])) else {
        return false;
    };
    gemms
        .iter()
        .all(|t| first_tile_input(fuf.get(*t)) == Some(act))
}

impl Implementation for FusedQkvRopeCacheImpl {
    fn name(&self) -> &'static str {
        "fused_qkv_rope_cache"
    }

    fn target_compatible(&self, _profile: &TargetProfile) -> bool {
        true
    }

    fn workload_constraint(&self) -> WorkloadConstraint {
        // `fused_qkv_rope_cache` is the decode-fused kernel (writes
        // K/V to the paged cache in one shot, returns only Q — the
        // contiguous K/V are discarded). Restricts paired with the
        // decode attention impl that reads K/V from the cache.
        // Prefill is handled by `FusedQkvRopePrefillImpl`.
        WorkloadConstraint::NumTokensRange { min: 1, max: 1 }
    }

    fn matches(&self, fuf: &Fuf, seed: TileId, _profile: &TargetProfile) -> Option<MatchInfo> {
        let seed_node = fuf.get(seed);
        if seed_node.op != OpKind::Gemm {
            return None;
        }
        // Dense kernel: reject AWQ storage on the seed.
        if !matches!(weight_storage_of(seed_node), Some(StorageFormat::Dense)) {
            return None;
        }

        // Find a RopeAppend whose first three Tile inputs all resolve
        // (directly or through a `BiasAdd` wrapper) to Gemm tiles
        // sharing a common activation. The seed must be one of those
        // three Gemms.
        //
        // Bias presence must be uniform across Q/K/V — either all
        // three are Gemm→BiasAdd→RopeAppend (Qwen2/Qwen3 with QKV
        // bias) or all three are Gemm→RopeAppend (Llama, bias-less).
        // Mixed would be a DSL-level inconsistency; we reject to
        // surface it as UnclaimedTile rather than silently do the
        // wrong thing.
        let rope_node = fuf.nodes.iter().find(|n| {
            if !is_rope_append_op(n.op) || n.inputs.len() < 3 {
                return false;
            }
            let qkv_raw: Vec<TileId> = n
                .inputs
                .iter()
                .take(3)
                .filter_map(|i| match i {
                    FufInput::Tile { id, .. } => Some(*id),
                    _ => None,
                })
                .collect();
            if qkv_raw.len() != 3 {
                return false;
            }
            // Resolve each of the three to `(gemm, maybe_bias)`.
            let resolved: Option<Vec<(TileId, Option<TileId>)>> = qkv_raw
                .iter()
                .map(|t| unwrap_gemm_through_bias(fuf, *t))
                .collect();
            let Some(resolved) = resolved else {
                return false;
            };
            // Uniform bias presence.
            let biased = resolved[0].1.is_some();
            if resolved.iter().any(|r| r.1.is_some() != biased) {
                return false;
            }
            let gemms: Vec<TileId> = resolved.iter().map(|r| r.0).collect();
            if !gemms.contains(&seed) {
                return false;
            }
            // All three gemms must share the same activation.
            let act = first_tile_input(fuf.get(gemms[0]));
            act.is_some() && gemms.iter().all(|t| first_tile_input(fuf.get(*t)) == act)
        })?;
        let rope_id = rope_node.id;

        // Resolve the three rope tile-inputs to (gemm, maybe_bias).
        let qkv_raw: Vec<TileId> = rope_node
            .inputs
            .iter()
            .take(3)
            .filter_map(|i| match i {
                FufInput::Tile { id, .. } => Some(*id),
                _ => None,
            })
            .collect();
        let resolved: Vec<(TileId, Option<TileId>)> = qkv_raw
            .iter()
            .map(|t| unwrap_gemm_through_bias(fuf, *t).expect("already validated in find"))
            .collect();

        let mut claimed: Vec<TileId> = Vec::with_capacity(7);
        for (g, b) in &resolved {
            claimed.push(*g);
            if let Some(b) = b {
                claimed.push(*b);
            }
        }
        claimed.push(rope_id);
        claimed.sort();

        let activation = first_tile_input(fuf.get(resolved[0].0))?.0;
        Some(MatchInfo {
            claimed_tiles: claimed,
            boundary_inputs: vec![activation],
            // Only slot 0 (Q) is read by tiles outside the claim —
            // K/V slots flow to the paged cache and the downstream
            // Attention kernel reads them from there (C4). Declaring
            // all three for now preserves dep tracking; C4 elides.
            boundary_outputs: vec![rope_id],
        })
    }

    fn cost_us(&self, _m: &MatchInfo, ctx: &CostCtx) -> f64 {
        let m = ctx.num_tokens() as f64;
        let hidden = ctx.bounds.get("hidden_size").copied().unwrap_or(0) as f64;
        let num_q_heads = ctx.bounds.get("num_attention_heads").copied().unwrap_or(0) as f64;
        let num_kv_heads = ctx.bounds.get("num_key_value_heads").copied().unwrap_or(0) as f64;
        let head_dim = ctx.bounds.get("head_dim").copied().unwrap_or(0) as f64;
        // q_size = num_q_heads * head_dim; kv_size = num_kv_heads * head_dim.
        let n = (num_q_heads + 2.0 * num_kv_heads) * head_dim;

        let flops = 2.0 * m * n * hidden;
        let peak = ctx.profile.peak_tflops_fp16 * 1e12;
        let gemm_us = if peak > 0.0 && flops > 0.0 {
            (flops / peak) * 1e6
        } else {
            0.0
        };

        // Rope + cache write: bandwidth-bound, read packed QKV +
        // write rotated Q + K/V to cache.
        let bw_gb = ctx.profile.memory_bandwidth_gbps;
        let bytes = 3.0 * m * n * BYTES_PER_ELEM;
        let rope_cache_us = if bw_gb > 0.0 {
            (bytes / (bw_gb * 1e9)) * 1e6
        } else {
            0.0
        };

        gemm_us + rope_cache_us
    }

    fn resources(&self, _m: &MatchInfo) -> Resources {
        Resources::ZERO
    }

    fn launch_kind(&self) -> LaunchKind {
        LaunchKind::HostCallback
    }

    fn supported_input_handoffs(&self) -> &[Handoff] {
        const H: &[Handoff] = &[Handoff::StreamOrder, Handoff::StreamEvent];
        H
    }

    fn supported_output_handoffs(&self) -> &[Handoff] {
        const H: &[Handoff] = &[Handoff::StreamOrder, Handoff::StreamEvent];
        H
    }

    fn input_layouts(&self, m: &MatchInfo) -> Vec<Layout> {
        vec![Layout::RowMajorBf16; m.boundary_inputs.len()]
    }

    fn output_layouts(&self, m: &MatchInfo) -> Vec<Layout> {
        vec![Layout::RowMajorBf16; m.boundary_outputs.len()]
    }

    fn is_compute_bound(&self) -> bool {
        true
    }

    #[allow(clippy::type_complexity)]
    fn output_alias(
        &self,
        claimed_tiles: &[TileId],
        fuf: &Fuf,
    ) -> Vec<((TileId, u8), Option<(TileId, u8)>)> {
        // Slot 0 (rotated Q) is a fresh OwnedTensor returned by the
        // rope-cache kernel. Slots 1 and 2 (K, V) are paged-cache
        // views — memory owned by the kv_cache pool, not the caching
        // allocator — so they're absent from the map (untracked).
        let rope_id = *claimed_tiles
            .iter()
            .find(|t| is_rope_append_op(fuf.get(**t).op))
            .expect("claim contains RopeAppend");
        vec![((rope_id, 0), None)]
    }

    fn required_weights(
        &self,
        claimed_tiles: &[TileId],
        fuf: &Fuf,
        program: &Program,
    ) -> Vec<WeightAccessor> {
        // One fused accessor covering the three Gemm weights — the
        // bias vectors (if present) ride through the accessor's
        // `LinearLayer::dense_bias()` output and `emit_call` asserts
        // their presence whenever the DSL claimed BiasAdd tiles. The
        // user populates the accessor with a `LinearLayer::Dense`
        // whose `weight` is the concatenated `[q | k | v]`; if the
        // underlying model carries biases, `bias` is the concatenated
        // `[q_b | k_b | v_b]`. `LinearLayer::load_dense_concat(gw,
        // prefixes, stream)` does exactly that: auto-detects bias on
        // the source prefixes and streams into one packed tensor.
        let sources: Vec<(WeightId, Option<u64>)> = claimed_tiles
            .iter()
            .filter_map(|t| {
                let n = fuf.get(*t);
                if n.op == OpKind::Gemm {
                    first_weight_ref(n)
                } else {
                    None
                }
            })
            .collect();
        let name = fused_accessor_name(program, &sources);
        vec![WeightAccessor {
            name,
            rust_type: quote! { ::ferrite_kernels::layers::LinearLayer },
            source_weights: sources,
        }]
    }

    fn emit_call(&self, ctx: &EmitCtx) -> TokenStream {
        // Identify the rope tile (NeoX or interleaved). The three Gemm
        // tiles come via `unwrap_gemm_through_bias` on the rope's first
        // three tile inputs — which strips an optional BiasAdd wrapper.
        let rope_id = *ctx
            .claimed_tiles
            .iter()
            .find(|t| is_rope_append_op(ctx.fuf.get(**t).op))
            .expect("claim must contain a RopeAppend");
        let rope_node = ctx.fuf.get(rope_id);
        let interleaved = rope_node.op == OpKind::RopeAppendInterleaved;

        let qkv_raw: Vec<TileId> = rope_node
            .inputs
            .iter()
            .take(3)
            .filter_map(|i| match i {
                FufInput::Tile { id, .. } => Some(*id),
                _ => None,
            })
            .collect();
        assert_eq!(qkv_raw.len(), 3, "rope has three tile inputs (q, k, v)");
        let resolved: Vec<(TileId, Option<TileId>)> = qkv_raw
            .iter()
            .map(|t| {
                unwrap_gemm_through_bias(ctx.fuf, *t)
                    .expect("claim-time check guarantees Gemm-or-BiasAdd(Gemm)")
            })
            .collect();
        let biased = resolved[0].1.is_some();
        let qkv_ids: Vec<TileId> = resolved.iter().map(|r| r.0).collect();
        // Use first Gemm (q_gemm) as the "representative" for activation.
        let q_gemm_id = qkv_ids[0];
        let activation = ctx.input_expr(q_gemm_id, 0);

        // Fused weight accessor — source_weights is the three Gemm
        // weights. The accessor's `LinearLayer` carries an optional
        // bias auto-populated by `load_dense_concat`; we assert its
        // presence below when the DSL claimed BiasAdd tiles.
        let qkv_weights: Vec<(WeightId, Option<u64>)> = qkv_ids
            .iter()
            .map(|t| first_weight_ref(ctx.fuf.get(*t)).expect("gemm weight"))
            .collect();
        let fused_name = fused_accessor_name(ctx.program, &qkv_weights);
        let weight_expr = ctx.weight_accessor(&fused_name);

        // Config-derived kernel args.
        let hidden = ctx.bound("hidden_size") as usize;
        let num_q_heads = ctx.bound("num_attention_heads") as usize;
        let num_kv_heads = ctx.bound("num_key_value_heads") as usize;
        let head_dim = ctx.bound("head_dim") as usize;
        let q_size = num_q_heads * head_dim;
        let kv_size = num_kv_heads * head_dim;
        let _ = hidden; // reserved for future checks

        // Layer index — captured by the DSL's `kv_cache[layer]`.
        let layer = rope_kv_cache_layer(rope_node)
            .expect("RopeAppend has a KvCache extern input with a concrete layer index")
            as usize;

        // Output idents:
        //   q_out = rope.slot 0 (rotated Q, OwnedTensor returned by kernel)
        //   k_out = rope.slot 1 (written to cache)
        //   v_out = rope.slot 2 (written to cache)
        // Downstream Attention reads K/V from the paged cache directly,
        // so the k_out/v_out bindings are paged-cache `TensorView`
        // aliases — no allocation.
        let q_out = ctx.output_ident(rope_id, 0);
        let k_out = ctx.output_ident(rope_id, 1);
        let v_out = ctx.output_ident(rope_id, 2);

        // If the DSL's rope chain includes BiasAdd tiles, require the
        // packed LinearLayer to actually carry a bias — otherwise
        // `.forward()` would silently call `cublas.gemm` (no bias)
        // and the bias_add math in the DSL would be dropped.
        let bias_assert = if biased {
            quote! {
                debug_assert!(
                    (#weight_expr).dense_bias().is_some(),
                    "FusedQkvRopeCacheImpl: DSL `bias_add` on QKV claimed but \
                     packed LinearLayer has no bias — check safetensors path"
                );
            }
        } else {
            quote! {}
        };

        let rotary_cos_sin = rotary_cos_sin_tokens(ctx.fuf, ctx.claimed_tiles);

        // Pick the rope-flavor kernel at emit time. Cohere's interleaved
        // pairing has its own fused-cache kernel; the FP8 path doesn't
        // currently have an interleaved variant, so an interleaved arch
        // with FP8 KV would need that kernel added (no live arch hits
        // this combo today).
        let cache_call = if interleaved {
            quote! {
                ::ferrite_kernels::kernels::fused_qkv_interleaved_rope_cache(
                    *qkv_packed,
                    *ctx.positions,
                    #rotary_cos_sin,
                    *ctx.slot_mapping,
                    *ctx.kv_cache.k_cache(#layer),
                    *ctx.kv_cache.v_cache(#layer),
                    #q_size,
                    #kv_size,
                    #num_q_heads,
                    #head_dim,
                    &mut device.caching,
                    device.compute_stream,
                )
            }
        } else {
            quote! {
                if ctx.kv_cache.is_fp8() {
                    ::ferrite_kernels::kernels::fused_qkv_rope_cache_fp8(
                        *qkv_packed,
                        *ctx.positions,
                        #rotary_cos_sin,
                        *ctx.slot_mapping,
                        *ctx.kv_cache.k_cache(#layer),
                        *ctx.kv_cache.v_cache(#layer),
                        ctx.kv_cache.k_scale_ptr(#layer),
                        ctx.kv_cache.v_scale_ptr(#layer),
                        #q_size,
                        #kv_size,
                        #num_q_heads,
                        #head_dim,
                        &mut device.caching,
                        device.compute_stream,
                    )
                } else {
                    ::ferrite_kernels::kernels::fused_qkv_rope_cache(
                        *qkv_packed,
                        *ctx.positions,
                        #rotary_cos_sin,
                        *ctx.slot_mapping,
                        *ctx.kv_cache.k_cache(#layer),
                        *ctx.kv_cache.v_cache(#layer),
                        #q_size,
                        #kv_size,
                        #num_q_heads,
                        #head_dim,
                        &mut device.caching,
                        device.compute_stream,
                    )
                }
            }
        };

        quote! {
            // Fused QKV GEMM → packed [num_tokens, q + 2*kv] tensor,
            // then fused RoPE + paged-cache write. The packed QKV
            // buffer drops as soon as the rope-cache kernel returns.
            //
            // FP8 KV-cache path branches at runtime: the cache dtype
            // is a per-model property known only at load time.
            let #q_out = unsafe {
                #bias_assert
                let qkv_packed = (#weight_expr).forward(
                    #activation,
                    &mut device.cublas,
                    &mut device.caching,
                    device.compute_stream,
                );
                #cache_call
            };
            // K/V aliases: the paged-cache layer slices. Bound for
            // symmetry with the FUF's tuple output shape; not read by
            // any downstream tile once `AttentionViaCacheImpl` takes
            // its input from the cache directly.
            let #k_out = ctx.kv_cache.k_cache(#layer);
            let #v_out = ctx.kv_cache.v_cache(#layer);
        }
    }
}

// ── FusedQkvQkNormRopeCacheImpl ─────────────────────────────────
//
// Multi-tile impl claiming the entire QKV + per-head QK-norm + RoPE
// + paged-cache-write chain produced by Qwen3 and Gemma3:
//
//   Gemm(q) ─→ [Add(w,1.0) →] [Reshape →] RmsNorm → [Reshape →] ─┐
//   Gemm(k) ─→ [Add(w,1.0) →] [Reshape →] RmsNorm → [Reshape →] ─┤
//   Gemm(v) ─────────────────────────────────────────────────────────┤
//                                                      RopeAppend ←─┘
//
// The three Gemms share the same activation. The Q and K paths flow
// through optional Add (Gemma3's `weight + 1.0`), optional Reshape
// (split [T, heads*head_dim] → [T*heads, head_dim]), RmsNorm
// (per-head norm), optional Reshape (flatten back), then into the
// RopeAppend. The V path feeds RopeAppend directly.
//
// Emits:
//   1. Packed cuBLAS QKV GEMM (same as FusedQkvRopeCacheImpl)
//   2. `qk_norm_rope_inplace` — fused QK-norm + RoPE in one launch
//   3. `reshape_and_cache` — write K/V to the paged cache
//
// Seeds on Gemm (upstream in topo order). Whichever of the three
// Q/K/V Gemms seeds first wins; the other two are swallowed by the
// same claim.

#[derive(Debug, Default)]
pub struct FusedQkvQkNormRopeCacheImpl;

/// Walk from a Gemm tile through the optional `Add → Reshape →
/// RmsNorm → Reshape` chain and return the chain's endpoint tile
/// (the last tile before RopeAppend) plus all intermediate tiles
/// encountered. Returns `None` if the chain doesn't match.
///
/// Chain shapes (all valid):
///   Gemm → RmsNorm                         (Qwen3, no scalar offset, no reshape)
///   Gemm → Add → RmsNorm                   (Gemma3, scalar offset, no reshape)
///   Gemm → Reshape → RmsNorm → Reshape     (with reshape, no scalar offset)
///   Gemm → Add → Reshape → RmsNorm → Reshape (with reshape + scalar offset)
fn walk_qk_norm_chain(fuf: &Fuf, gemm_id: TileId) -> Option<(TileId, Vec<TileId>, Option<TileId>)> {
    // Returns (endpoint_tile, intermediates_excluding_gemm, add_tile_if_any)
    //
    // The data chain is: Gemm → [Reshape(split)] → RmsNorm → [Reshape(flatten)]
    // The Add(Weight, Scalar) for Gemma's `w + 1.0` feeds RmsNorm's
    // weight slot (slot 1), NOT the activation path. It's a side input
    // that we claim but don't walk through.
    let mut cursor = gemm_id;
    let mut intermediates: Vec<TileId> = Vec::new();
    let mut add_tile: Option<TileId> = None;

    // Step 1: optional Reshape (split to per-head)
    if let Some(reshape_node) = fuf
        .nodes
        .iter()
        .find(|n| n.op == OpKind::Reshape && consumes_tile(n, cursor))
    {
        let has_rmsnorm_downstream = fuf.nodes.iter().any(|n| {
            n.op == OpKind::RmsNorm
                && matches!(n.inputs.first(), Some(FufInput::Tile { id, .. }) if *id == reshape_node.id)
        });
        if has_rmsnorm_downstream {
            intermediates.push(reshape_node.id);
            cursor = reshape_node.id;
        }
    }

    // Step 2: required RmsNorm consuming cursor at slot 0
    let rmsnorm_node = fuf.nodes.iter().find(|n| {
        n.op == OpKind::RmsNorm
            && matches!(n.inputs.first(), Some(FufInput::Tile { id, .. }) if *id == cursor)
    })?;
    intermediates.push(rmsnorm_node.id);

    // Check if RmsNorm's weight slot (slot 1) is an Add(Weight, Scalar)
    // tile — that's the Gemma `(1+w)` offset. If so, claim the Add.
    if let Some(FufInput::Tile {
        id: weight_tile, ..
    }) = rmsnorm_node.inputs.get(1)
    {
        let wt_node = fuf.get(*weight_tile);
        if wt_node.op == OpKind::Add
            && wt_node
                .inputs
                .iter()
                .any(|i| matches!(i, FufInput::Scalar(_)))
            && wt_node
                .inputs
                .iter()
                .any(|i| matches!(i, FufInput::Weight { .. }))
        {
            add_tile = Some(*weight_tile);
            intermediates.push(*weight_tile);
        }
    }
    if rmsnorm_node.inputs.len() < 2 {
        return None;
    }

    cursor = rmsnorm_node.id;
    match &rmsnorm_node.inputs[1] {
        FufInput::Tile { id, .. } => {
            // Must be our Add tile (scalar-offset path).
            if add_tile != Some(*id) {
                return None;
            }
        }
        FufInput::Weight { .. } => {
            // Direct weight — fine (Qwen3 path, no scalar offset).
        }
        _ => return None,
    }

    // Step 4: optional Reshape (flatten back to [T, heads*head_dim])
    if let Some(reshape_node) = fuf
        .nodes
        .iter()
        .find(|n| n.op == OpKind::Reshape && consumes_tile(n, cursor))
    {
        intermediates.push(reshape_node.id);
        cursor = reshape_node.id;
    }

    Some((cursor, intermediates, add_tile))
}

/// True if this RopeAppend tile's Q and K inputs flow through a
/// `Gemm → [Add →] [Reshape →] RmsNorm → [Reshape →]` chain from
/// three shared-activation Gemms — the pattern
/// `FusedQkvQkNormRopeCacheImpl` claims. Used by `RopeAppendRefImpl`
/// to defer when this larger fusion would match.
fn rope_append_has_qk_norm_upstream(fuf: &Fuf, rope_tile: TileId) -> bool {
    let node = fuf.get(rope_tile);
    if node.op != OpKind::RopeAppend || node.inputs.len() < 3 {
        return false;
    }
    let qkv_raw: Vec<TileId> = node
        .inputs
        .iter()
        .take(3)
        .filter_map(|i| match i {
            FufInput::Tile { id, .. } => Some(*id),
            _ => None,
        })
        .collect();
    if qkv_raw.len() != 3 {
        return false;
    }

    // Q and K must come through QK-norm chains; V must be a Gemm.
    let q_input = qkv_raw[0];
    let k_input = qkv_raw[1];
    let v_id = qkv_raw[2];

    // V must be a Gemm directly.
    if fuf.get(v_id).op != OpKind::Gemm {
        return false;
    }

    // Find the Gemm at the root of Q and K chains by walking backwards.
    // The chain endpoint feeds into RopeAppend; we need the Gemm root.
    // Walk backwards from q_input: could be Reshape → RmsNorm → [Reshape →] [Add →] Gemm
    let q_gemm = find_gemm_root_of_norm_chain(fuf, q_input);
    let k_gemm = find_gemm_root_of_norm_chain(fuf, k_input);

    let (Some(q_gemm), Some(k_gemm)) = (q_gemm, k_gemm) else {
        return false;
    };

    // All three Gemms must share the same activation.
    let q_act = first_tile_input(fuf.get(q_gemm));
    let k_act = first_tile_input(fuf.get(k_gemm));
    let v_act = first_tile_input(fuf.get(v_id));
    q_act.is_some() && q_act == k_act && q_act == v_act
}

/// Walk backwards from a tile that is the endpoint of a QK-norm chain
/// (Reshape or RmsNorm) back through the chain to find the root Gemm.
fn find_gemm_root_of_norm_chain(fuf: &Fuf, endpoint: TileId) -> Option<TileId> {
    let mut cursor = endpoint;

    // If cursor is a Reshape, step back
    if fuf.get(cursor).op == OpKind::Reshape {
        cursor = first_tile_input(fuf.get(cursor))?.0;
    }
    // Now should be RmsNorm
    if fuf.get(cursor).op != OpKind::RmsNorm {
        return None;
    }
    // RmsNorm slot 0 is the tensor input
    cursor = first_tile_input(fuf.get(cursor))?.0;

    // Optional Reshape (split)
    if fuf.get(cursor).op == OpKind::Reshape {
        cursor = first_tile_input(fuf.get(cursor))?.0;
    }
    // Optional Add(Weight, Scalar)
    if fuf.get(cursor).op == OpKind::Add {
        // The Add should have a Tile input from a Gemm
        cursor = first_tile_input(fuf.get(cursor))?.0;
    }

    if fuf.get(cursor).op == OpKind::Gemm {
        Some(cursor)
    } else {
        None
    }
}

impl Implementation for FusedQkvQkNormRopeCacheImpl {
    fn name(&self) -> &'static str {
        "fused_qkv_qk_norm_rope_cache"
    }

    fn target_compatible(&self, _profile: &TargetProfile) -> bool {
        true
    }

    fn workload_constraint(&self) -> WorkloadConstraint {
        // Handles all workload points; downstream `AttentionViaCacheImpl`
        // (M=1) and `AttentionPrefillContiguousImpl` (M>=2) pick up via
        // their own WorkloadConstraint gating.
        WorkloadConstraint::Any
    }

    fn matches(&self, fuf: &Fuf, seed: TileId, _profile: &TargetProfile) -> Option<MatchInfo> {
        let seed_node = fuf.get(seed);
        if seed_node.op != OpKind::Gemm {
            return None;
        }
        if !matches!(weight_storage_of(seed_node), Some(StorageFormat::Dense)) {
            return None;
        }

        // Find a RopeAppend whose Q and K inputs flow through QK-norm
        // chains from Gemms sharing the same activation as the seed,
        // and whose V input is a Gemm also sharing that activation.
        let rope_node = fuf.nodes.iter().find(|n| {
            if n.op != OpKind::RopeAppend || n.inputs.len() < 3 {
                return false;
            }
            let qkv_raw: Vec<TileId> = n
                .inputs
                .iter()
                .take(3)
                .filter_map(|i| match i {
                    FufInput::Tile { id, .. } => Some(*id),
                    _ => None,
                })
                .collect();
            if qkv_raw.len() != 3 {
                return false;
            }

            let v_id = qkv_raw[2];
            if fuf.get(v_id).op != OpKind::Gemm {
                return false;
            }

            // Walk forward from each Gemm to verify the QK-norm chain
            // exists. We need to find which Gemms root the Q and K
            // chains.
            let q_gemm = find_gemm_root_of_norm_chain(fuf, qkv_raw[0]);
            let k_gemm = find_gemm_root_of_norm_chain(fuf, qkv_raw[1]);

            let (Some(q_gemm), Some(k_gemm)) = (q_gemm, k_gemm) else {
                return false;
            };

            let gemms = [q_gemm, k_gemm, v_id];
            if !gemms.contains(&seed) {
                return false;
            }

            let act = first_tile_input(fuf.get(gemms[0]));
            act.is_some() && gemms.iter().all(|t| first_tile_input(fuf.get(*t)) == act)
        })?;

        let rope_id = rope_node.id;
        let qkv_raw: Vec<TileId> = rope_node
            .inputs
            .iter()
            .take(3)
            .filter_map(|i| match i {
                FufInput::Tile { id, .. } => Some(*id),
                _ => None,
            })
            .collect();

        let v_id = qkv_raw[2];
        let q_gemm = find_gemm_root_of_norm_chain(fuf, qkv_raw[0]).expect("validated in find");
        let k_gemm = find_gemm_root_of_norm_chain(fuf, qkv_raw[1]).expect("validated in find");

        // Walk the Q and K chains forward to collect all intermediate tiles.
        let (_, q_chain, _) = walk_qk_norm_chain(fuf, q_gemm).expect("Q chain must be valid");
        let (_, k_chain, _) = walk_qk_norm_chain(fuf, k_gemm).expect("K chain must be valid");

        let mut claimed: Vec<TileId> = Vec::with_capacity(16);
        claimed.push(q_gemm);
        claimed.push(k_gemm);
        claimed.push(v_id);
        claimed.extend_from_slice(&q_chain);
        claimed.extend_from_slice(&k_chain);
        claimed.push(rope_id);
        claimed.sort();
        claimed.dedup();

        let activation = first_tile_input(fuf.get(q_gemm))?.0;
        Some(MatchInfo {
            claimed_tiles: claimed,
            boundary_inputs: vec![activation],
            boundary_outputs: vec![rope_id],
        })
    }

    fn cost_us(&self, _m: &MatchInfo, ctx: &CostCtx) -> f64 {
        let m = ctx.num_tokens() as f64;
        let hidden = ctx.bounds.get("hidden_size").copied().unwrap_or(0) as f64;
        let num_q_heads = ctx.bounds.get("num_attention_heads").copied().unwrap_or(0) as f64;
        let num_kv_heads = ctx.bounds.get("num_key_value_heads").copied().unwrap_or(0) as f64;
        let head_dim = ctx.bounds.get("head_dim").copied().unwrap_or(0) as f64;
        let n = (num_q_heads + 2.0 * num_kv_heads) * head_dim;

        let flops = 2.0 * m * n * hidden;
        let peak = ctx.profile.peak_tflops_fp16 * 1e12;
        let gemm_us = if peak > 0.0 && flops > 0.0 {
            (flops / peak) * 1e6
        } else {
            0.0
        };

        // QK-norm + RoPE + cache write: bandwidth-bound.
        // Read packed Q+K for norm, write normalized Q+K, then RoPE +
        // cache write (same as FusedQkvRopeCacheImpl plus the norm r/w).
        let bw_gb = ctx.profile.memory_bandwidth_gbps;
        let q_kv_bytes = m * (num_q_heads + num_kv_heads) * head_dim * BYTES_PER_ELEM;
        // norm reads + writes Q and K (2x), plus rope + cache write
        let rope_bytes = 3.0 * m * n * BYTES_PER_ELEM;
        let norm_rope_cache_us = if bw_gb > 0.0 {
            ((2.0 * q_kv_bytes + rope_bytes) / (bw_gb * 1e9)) * 1e6
        } else {
            0.0
        };

        gemm_us + norm_rope_cache_us
    }

    fn resources(&self, _m: &MatchInfo) -> Resources {
        Resources::ZERO
    }

    fn launch_kind(&self) -> LaunchKind {
        LaunchKind::HostCallback
    }

    fn supported_input_handoffs(&self) -> &[Handoff] {
        const H: &[Handoff] = &[Handoff::StreamOrder, Handoff::StreamEvent];
        H
    }

    fn supported_output_handoffs(&self) -> &[Handoff] {
        const H: &[Handoff] = &[Handoff::StreamOrder, Handoff::StreamEvent];
        H
    }

    fn input_layouts(&self, m: &MatchInfo) -> Vec<Layout> {
        vec![Layout::RowMajorBf16; m.boundary_inputs.len()]
    }

    fn output_layouts(&self, m: &MatchInfo) -> Vec<Layout> {
        vec![Layout::RowMajorBf16; m.boundary_outputs.len()]
    }

    fn is_compute_bound(&self) -> bool {
        true
    }

    #[allow(clippy::type_complexity)]
    fn output_alias(
        &self,
        claimed_tiles: &[TileId],
        fuf: &Fuf,
    ) -> Vec<((TileId, u8), Option<(TileId, u8)>)> {
        let rope_id = *claimed_tiles
            .iter()
            .find(|t| fuf.get(**t).op == OpKind::RopeAppend)
            .expect("claim contains RopeAppend");
        vec![((rope_id, 0), None)]
    }

    fn required_weights(
        &self,
        claimed_tiles: &[TileId],
        fuf: &Fuf,
        program: &Program,
    ) -> Vec<WeightAccessor> {
        // The `qk_norm_rope_inplace` kernel requires Q/K as
        // contiguous `[T, num_heads, head_dim]` tensors. A packed
        // cuBLAS QKV output `[T, q_size + 2*kv_size]` has strided
        // Q/K across tokens, so slicing won't give usable inputs.
        // Instead we declare three SEPARATE `LinearLayer`
        // accessors (one per Gemm) and let `emit_call` run three
        // GEMMs — each producing a contiguous output.
        //
        // Identify the RopeAppend tile so we can figure out the
        // Q/K/V role of each Gemm. Rope's tile inputs [0,1,2] are
        // qkv_raw; Q and K flow through the norm chain, V is a
        // Gemm directly.
        let rope_id = *claimed_tiles
            .iter()
            .find(|t| fuf.get(**t).op == OpKind::RopeAppend)
            .expect("claim must contain a RopeAppend");
        let rope_node = fuf.get(rope_id);
        let qkv_raw: Vec<TileId> = rope_node
            .inputs
            .iter()
            .take(3)
            .filter_map(|i| match i {
                FufInput::Tile { id, .. } => Some(*id),
                _ => None,
            })
            .collect();
        assert_eq!(qkv_raw.len(), 3, "rope has three tile inputs (q, k, v)");

        let v_gemm = qkv_raw[2];
        let q_gemm = find_gemm_root_of_norm_chain(fuf, qkv_raw[0])
            .expect("Q norm chain validated at match time");
        let k_gemm = find_gemm_root_of_norm_chain(fuf, qkv_raw[1])
            .expect("K norm chain validated at match time");

        // 1. Three separate LinearLayer accessors (Q, K, V).
        let mut accessors: Vec<WeightAccessor> = Vec::with_capacity(5);
        for &gemm_id in &[q_gemm, k_gemm, v_gemm] {
            let (wid, widx) = first_weight_ref(fuf.get(gemm_id)).expect("gemm has a weight input");
            let name = weight_field_name(program, wid, widx);
            accessors.push(WeightAccessor {
                name,
                rust_type: quote! { ::ferrite_kernels::layers::LinearLayer },
                source_weights: vec![(wid, widx)],
            });
        }

        // 2. Two RmsNorm weights for q_norm and k_norm.
        //
        // The weight source depends on whether the chain has a
        // scalar-offset Add: if so, the RmsNorm tile's weight slot
        // (slot 1) is a Tile input pointing at the Add, and the
        // actual Weight ref lives on the Add's inputs. Otherwise the
        // RmsNorm's slot 1 is a direct Weight input.
        let rmsnorm_ids: Vec<TileId> = claimed_tiles
            .iter()
            .filter(|t| fuf.get(**t).op == OpKind::RmsNorm)
            .copied()
            .collect();
        assert_eq!(
            rmsnorm_ids.len(),
            2,
            "FusedQkvQkNormRopeCacheImpl claim must have exactly 2 RmsNorm tiles"
        );

        for &rmsnorm_id in &rmsnorm_ids {
            let rmsnorm_node = fuf.get(rmsnorm_id);
            let (weight_id, weight_idx) = match &rmsnorm_node.inputs[1] {
                FufInput::Tile { id, .. } => {
                    // Scalar-offset path: the Add tile carries the weight.
                    let add_node = fuf.get(*id);
                    add_node
                        .inputs
                        .iter()
                        .find_map(|i| match i {
                            FufInput::Weight { id, index, .. } => Some((*id, *index)),
                            _ => None,
                        })
                        .expect("scalar-offset Add has a Weight input")
                }
                FufInput::Weight { id, index, .. } => (*id, *index),
                _ => panic!("RmsNorm slot 1 must be Tile (Add) or Weight"),
            };
            let name = weight_field_name(program, weight_id, weight_idx);
            accessors.push(WeightAccessor {
                name,
                rust_type: quote! { ::ferrite_kernels::layers::RmsNorm },
                source_weights: vec![(weight_id, weight_idx)],
            });
        }

        accessors
    }

    fn emit_call(&self, ctx: &EmitCtx) -> TokenStream {
        // Identify the RopeAppend tile.
        let rope_id = *ctx
            .claimed_tiles
            .iter()
            .find(|t| ctx.fuf.get(**t).op == OpKind::RopeAppend)
            .expect("claim must contain a RopeAppend");
        let rope_node = ctx.fuf.get(rope_id);

        // Find the three Gemms and identify Q/K/V roles.
        let qkv_raw: Vec<TileId> = rope_node
            .inputs
            .iter()
            .take(3)
            .filter_map(|i| match i {
                FufInput::Tile { id, .. } => Some(*id),
                _ => None,
            })
            .collect();
        assert_eq!(qkv_raw.len(), 3, "rope has three tile inputs (q, k, v)");

        let v_id = qkv_raw[2];
        let q_gemm = find_gemm_root_of_norm_chain(ctx.fuf, qkv_raw[0])
            .expect("Q norm chain validated at match time");
        let k_gemm = find_gemm_root_of_norm_chain(ctx.fuf, qkv_raw[1])
            .expect("K norm chain validated at match time");

        // Walk Q and K chains to get Add tiles (for weight offsets).
        let (_, _, q_add) = walk_qk_norm_chain(ctx.fuf, q_gemm).expect("Q chain valid");
        let (_, _, k_add) = walk_qk_norm_chain(ctx.fuf, k_gemm).expect("K chain valid");

        // Determine weight offsets from Add tiles.
        let q_weight_offset: f32 = q_add
            .map(|add_id| {
                ctx.fuf
                    .get(add_id)
                    .inputs
                    .iter()
                    .find_map(|i| match i {
                        FufInput::Scalar(v) => Some(*v as f32),
                        _ => None,
                    })
                    .unwrap_or(0.0)
            })
            .unwrap_or(0.0);
        let k_weight_offset: f32 = k_add
            .map(|add_id| {
                ctx.fuf
                    .get(add_id)
                    .inputs
                    .iter()
                    .find_map(|i| match i {
                        FufInput::Scalar(v) => Some(*v as f32),
                        _ => None,
                    })
                    .unwrap_or(0.0)
            })
            .unwrap_or(0.0);

        let activation = ctx.input_expr(q_gemm, 0);

        // Three separate LinearLayer accessors (Q, K, V) — matching
        // the accessors declared in `required_weights`. A packed
        // QKV gemm produces strided Q/K/V views that the
        // `qk_norm_rope_inplace` kernel can't consume; three
        // separate gemms each produce contiguous outputs.
        let (q_wid, q_widx) = first_weight_ref(ctx.fuf.get(q_gemm)).expect("Q gemm has a weight");
        let (k_wid, k_widx) = first_weight_ref(ctx.fuf.get(k_gemm)).expect("K gemm has a weight");
        let (v_wid, v_widx) = first_weight_ref(ctx.fuf.get(v_id)).expect("V gemm has a weight");
        let q_weight_name = weight_field_name(ctx.program, q_wid, q_widx);
        let k_weight_name = weight_field_name(ctx.program, k_wid, k_widx);
        let v_weight_name = weight_field_name(ctx.program, v_wid, v_widx);
        let q_weight_expr = ctx.weight_accessor(&q_weight_name);
        let k_weight_expr = ctx.weight_accessor(&k_weight_name);
        let v_weight_expr = ctx.weight_accessor(&v_weight_name);

        // RmsNorm weight accessors — ordered Q then K by topo order
        // of their RmsNorm tiles within the claim.
        let rmsnorm_ids: Vec<TileId> = ctx
            .claimed_tiles
            .iter()
            .filter(|t| ctx.fuf.get(**t).op == OpKind::RmsNorm)
            .copied()
            .collect();
        assert_eq!(rmsnorm_ids.len(), 2);

        // Determine which RmsNorm is Q's and which is K's by checking
        // which chain each belongs to.
        let q_rmsnorm = rmsnorm_ids
            .iter()
            .find(|&&id| {
                let (_, q_chain, _) = walk_qk_norm_chain(ctx.fuf, q_gemm).unwrap();
                q_chain.contains(&id)
            })
            .expect("Q rmsnorm in claim");
        let k_rmsnorm = rmsnorm_ids
            .iter()
            .find(|&&id| {
                let (_, k_chain, _) = walk_qk_norm_chain(ctx.fuf, k_gemm).unwrap();
                k_chain.contains(&id)
            })
            .expect("K rmsnorm in claim");

        // Build accessor expressions for the two norm weights.
        let q_norm_weight_ref = {
            let rmsnorm_node = ctx.fuf.get(*q_rmsnorm);
            match &rmsnorm_node.inputs[1] {
                FufInput::Tile { id, .. } => {
                    let add_node = ctx.fuf.get(*id);
                    let (wid, widx) = add_node
                        .inputs
                        .iter()
                        .find_map(|i| match i {
                            FufInput::Weight { id, index, .. } => Some((*id, *index)),
                            _ => None,
                        })
                        .expect("Add has Weight input");
                    weight_field_name(ctx.program, wid, widx)
                }
                FufInput::Weight { id, index, .. } => weight_field_name(ctx.program, *id, *index),
                _ => panic!("RmsNorm slot 1 must be Tile or Weight"),
            }
        };
        let k_norm_weight_ref = {
            let rmsnorm_node = ctx.fuf.get(*k_rmsnorm);
            match &rmsnorm_node.inputs[1] {
                FufInput::Tile { id, .. } => {
                    let add_node = ctx.fuf.get(*id);
                    let (wid, widx) = add_node
                        .inputs
                        .iter()
                        .find_map(|i| match i {
                            FufInput::Weight { id, index, .. } => Some((*id, *index)),
                            _ => None,
                        })
                        .expect("Add has Weight input");
                    weight_field_name(ctx.program, wid, widx)
                }
                FufInput::Weight { id, index, .. } => weight_field_name(ctx.program, *id, *index),
                _ => panic!("RmsNorm slot 1 must be Tile or Weight"),
            }
        };
        let q_norm_expr = ctx.weight_accessor(&q_norm_weight_ref);
        let k_norm_expr = ctx.weight_accessor(&k_norm_weight_ref);

        // Config-derived kernel args.
        let num_q_heads = ctx.bound("num_attention_heads") as usize;
        let num_kv_heads = ctx.bound("num_key_value_heads") as usize;
        let head_dim = ctx.bound("head_dim") as usize;

        let layer = rope_kv_cache_layer(rope_node)
            .expect("RopeAppend has a KvCache extern input with a concrete layer index")
            as usize;

        let q_out = ctx.output_ident(rope_id, 0);
        let k_out = ctx.output_ident(rope_id, 1);
        let v_out = ctx.output_ident(rope_id, 2);

        let rotary_cos_sin = rotary_cos_sin_tokens(ctx.fuf, ctx.claimed_tiles);

        quote! {
            // Three separate cuBLAS GEMMs producing contiguous Q, K, V
            // OwnedTensors (shape [num_tokens, *_size]). The
            // `qk_norm_rope_inplace` kernel requires contiguous Q/K
            // at [T, num_heads, head_dim]; slicing a packed QKV
            // output would leave Q/K strided across tokens.
            let mut #q_out = unsafe {
                let nt = (*ctx.input_ids).dim(0) as usize;
                let q = (#q_weight_expr).forward(
                    #activation,
                    &mut device.cublas,
                    &mut device.caching,
                    device.compute_stream,
                );
                let k = (#k_weight_expr).forward(
                    #activation,
                    &mut device.cublas,
                    &mut device.caching,
                    device.compute_stream,
                );
                let v = (#v_weight_expr).forward(
                    #activation,
                    &mut device.cublas,
                    &mut device.caching,
                    device.compute_stream,
                );

                // 3D views on the contiguous OwnedTensors: the
                // `qk_norm_rope_inplace` and `reshape_and_cache`
                // kernels read dim(0)/dim(1)/dim(2) as
                // `[num_tokens, num_heads, head_dim]`.
                let q_view = (*q).reshape(&[nt, #num_q_heads, #head_dim]);
                let k_view = (*k).reshape(&[nt, #num_kv_heads, #head_dim]);
                let v_view = (*v).reshape(&[nt, #num_kv_heads, #head_dim]);

                // Fused QK-norm + RoPE in one kernel launch —
                // operates in-place on the Q and K storage.
                ::ferrite_kernels::kernels::qk_norm_rope_inplace(
                    q_view,
                    k_view,
                    (#q_norm_expr).weight,
                    (#k_norm_expr).weight,
                    #rotary_cos_sin,
                    *ctx.positions,
                    #num_q_heads,
                    #num_kv_heads,
                    #head_dim,
                    (#q_norm_expr).eps,
                    #q_weight_offset,
                    #k_weight_offset,
                    device.compute_stream,
                );

                // Write K/V to the paged cache. K and V storage is
                // freed when the OwnedTensors drop at end of scope.
                ::ferrite_kernels::kernels::reshape_and_cache(
                    k_view,
                    v_view,
                    *ctx.kv_cache.k_cache(#layer),
                    *ctx.kv_cache.v_cache(#layer),
                    *ctx.slot_mapping,
                    ctx.kv_cache.block_size,
                    device.compute_stream,
                );

                q
            };
            // Reshape Q's OwnedTensor in-place to [T, num_q_heads,
            // head_dim] — the layout AttentionViaCacheImpl /
            // AttentionPrefillContiguousImpl expect downstream.
            unsafe {
                let nt = (*#q_out).dim(0);
                let dt = (*#q_out).dtype();
                #q_out.reshape(&[nt, #num_q_heads, #head_dim], dt);
            }
            // K/V aliases: paged-cache layer slices. The contiguous
            // K/V OwnedTensors produced above have already been
            // committed to the cache and dropped at end of the unsafe
            // block; downstream attention reads from the paged cache.
            let #k_out = ctx.kv_cache.k_cache(#layer);
            let #v_out = ctx.kv_cache.v_cache(#layer);
        }
    }
}

// ── AttentionViaCacheImpl ────────────────────────────────────────
//
// Singleton matcher on `OpKind::Attention`. Maps the DSL's
// `attention(q, k, v, kv_cache[layer], block_table)` to
// `ferrite_kernels::kernels::flash_attn_paged`, which reads Q + the
// paged KV cache directly — after `FusedQkvRopeCacheImpl` has
// written K/V to the layer's cache slice — and ignores the DSL
// tile's K/V Tile-inputs entirely.
//
// The layer index is captured on the Attention tile's own
// `FufInput::Extern { kind: KvCache, index: Some(L) }` (the DSL's
// `kv_cache[layer]` argument), same as for RopeAppend.

#[derive(Debug, Default)]
pub struct AttentionViaCacheImpl;

impl Implementation for AttentionViaCacheImpl {
    fn name(&self) -> &'static str {
        "attention_via_cache"
    }

    fn target_compatible(&self, _profile: &TargetProfile) -> bool {
        true
    }

    fn workload_constraint(&self) -> WorkloadConstraint {
        // Paged-cache-reading attention is the decode path. K/V
        // were written to the cache by the upstream decode QKV
        // impl; this impl ignores the DSL Attention tile's slot-1
        // and slot-2 inputs (which are cache aliases in the decode
        // flow). Prefill uses `AttentionPrefillContiguousImpl`.
        WorkloadConstraint::NumTokensRange { min: 1, max: 1 }
    }

    fn matches(&self, fuf: &Fuf, seed: TileId, _profile: &TargetProfile) -> Option<MatchInfo> {
        single_tile_match(fuf, seed, OpKind::Attention)
    }

    fn cost_us(&self, m: &MatchInfo, ctx: &CostCtx) -> f64 {
        // Use the calibrated `fa2_attn_...` CSV row when present so
        // the FI-vs-FA2 tiebreak runs on real timings; falls back to
        // the analytic `cost_attention` formula when no row matches.
        cost_attention_calibrated(m, ctx)
    }

    fn resources(&self, _m: &MatchInfo) -> Resources {
        Resources::ZERO
    }

    fn launch_kind(&self) -> LaunchKind {
        LaunchKind::HostCallback
    }

    fn supported_input_handoffs(&self) -> &[Handoff] {
        const H: &[Handoff] = &[Handoff::StreamOrder, Handoff::StreamEvent];
        H
    }

    fn supported_output_handoffs(&self) -> &[Handoff] {
        const H: &[Handoff] = &[Handoff::StreamOrder, Handoff::StreamEvent];
        H
    }

    fn input_layouts(&self, m: &MatchInfo) -> Vec<Layout> {
        vec![Layout::RowMajorBf16; m.boundary_inputs.len()]
    }

    fn output_layouts(&self, m: &MatchInfo) -> Vec<Layout> {
        vec![Layout::RowMajorBf16; m.boundary_outputs.len()]
    }

    fn is_compute_bound(&self) -> bool {
        true
    }

    // Default `required_weights` yields zero accessors — Attention
    // consumes no weights. The paged cache and per-call metadata live
    // on `ForwardCtx`, not `WeightBundle`.

    fn emit_call(&self, ctx: &EmitCtx) -> TokenStream {
        let tile = ctx.primary();
        let out = ctx.output_ident(tile, 0);
        let node = ctx.fuf.get(tile);

        // Q input: slot 0 of the DSL's attention call. K and V (slots
        // 1, 2) are ignored — the kernel reads them from the paged
        // cache written by the upstream FusedQkvRopeCache.
        let q_expr = ctx.input_expr(tile, 0);

        // Layer index — from the `kv_cache[layer]` extern input.
        let layer = node
            .inputs
            .iter()
            .find_map(|i| match i {
                FufInput::Extern {
                    kind: crate::classified::ExternKind::KvCache,
                    index: Some(layer),
                } => Some(*layer),
                _ => None,
            })
            .expect("Attention has a kv_cache extern with a concrete layer index")
            as usize;

        // Softmax scale: config-driven. Gemma2 sets
        // `query_pre_attn_scalar` (softmax scale = that^-0.5); Llama/
        // Qwen2 have no such field, so we fall back to the standard
        // `1/sqrt(head_dim)`. Both computed at emit time; the emitted
        // tokens are a plain `f32` literal either way.
        let scale_tokens = attention_scale_tokens(ctx);
        let softcap_tokens = attention_softcap_tokens(ctx);
        // q_size for the [num_tokens, q_size] reshape o_proj needs.
        let q_size = (ctx.bound("num_attention_heads") * ctx.bound("head_dim")) as usize;

        let rotary_cos_sin = rotary_cos_sin_tokens(ctx.fuf, ctx.claimed_tiles);
        // Discover the rope flavor used at this layer so FA-2's
        // span-rotation path uses the right element pairing on
        // unrotated KV blocks. Determined statically from the layer's
        // rope tile in the FUF.
        let interleaved_lit = layer_rope_is_interleaved(ctx.fuf, layer as u64);

        quote! {
            // Decode paged attention. Delegates to the
            // `attention_decode_from_cache` helper which:
            //  - routes to `fp8_decode_attention` when kv_cache is fp8,
            //  - else calls `flash_attn_paged_ext` with span-rotation
            //    args derived from `kv_cache.block_unrotated_gpu()` so
            //    relocatable/unrotated KV blocks get their RoPE applied
            //    by FA2 in shared memory.
            let mut #out = unsafe {
                let has_spans = !ctx.kv_cache.block_unrotated_gpu().is_null();
                let (cos_sin_ptr, rotary_dim) = if has_spans {
                    (
                        #rotary_cos_sin.raw_ptr() as *const u8,
                        #rotary_cos_sin.dim(1),
                    )
                } else {
                    (::std::ptr::null::<u8>(), 0)
                };
                ::ferrite_kernels::attention_helpers::attention_decode_from_cache(
                    #q_expr,
                    ctx.cu_seqlens_q,
                    ctx.seqused_k,
                    ctx.block_table,
                    ctx.max_seqlen_q,
                    ctx.max_seqlen_k,
                    #scale_tokens,
                    #softcap_tokens,
                    -1,   // window_size_left (-1 = disabled; sliding variant covers non-(-1))
                    ctx.kv_cache,
                    #layer,
                    device.num_sm,
                    &mut device.caching,
                    device.compute_stream,
                    cos_sin_ptr,
                    rotary_dim,
                    #interleaved_lit, // is_rotary_interleaved (rope flavor at this layer)
                )
            };
            // Flatten [num_tokens, num_q_heads, head_dim] → [num_tokens, q_size]
            // so the downstream o_proj gemm sees a 2D [M, K] input with the
            // right K dimension.
            unsafe {
                let nt = (*#out).dim(0);
                let dt = (*#out).dtype();
                #out.reshape(&[nt, #q_size], dt);
            }
        }
    }
}

// ── RopeAppendRefImpl ────────────────────────────────────────────
//
// Singleton fallback for `OpKind::RopeAppend` tiles that the QKV
// fusions (`FusedQkvRopeCacheImpl` / `FusedQkvRopePrefillImpl`)
// don't claim — typically Qwen3 / Gemma3, where per-head Q/K
// rmsnorm tiles sit between the QKV gemms and rope_append, breaking
// the gemm→rope adjacency the fused matchers require.
//
// Emits a two-kernel sequence:
//   1. `rotary_embedding_inplace(q, k, positions, cos_sin, head_dim)`
//      — applies RoPE to Q and K in-place on the 2D `[T, heads*head_dim]`
//      / `[T, kv_heads*head_dim]` upstream tensors.
//   2. `reshape_and_cache(k_3d, v_3d, k_cache, v_cache, slot_mapping,
//      block_size)` — writes K/V to the paged cache. K and V are
//      reshaped to 3D `[T, kv_heads, head_dim]` views (metadata only)
//      to match the kernel's expected layout.
//
// Output bindings: all three are 3D TensorView reshapes aliasing the
// upstream storage — `[T, q_heads, head_dim]` for Q, `[T, kv_heads,
// head_dim]` for K and V. This matches what the fused
// `FusedQkvRope{Cache,Prefill}Impl` impls produce, so the downstream
// `AttentionViaCacheImpl` / `AttentionPrefillContiguousImpl` (which
// read `q.dim(0,1,2)` as `[total_q, num_heads, head_dim]`) see the
// same layout whichever upstream fired. No move / no consume — the
// drop pass resolves aliases to the ultimate owner and keeps it
// alive until every downstream use is done.
//
// All M (decode + prefill) — kernel signatures don't depend on token
// count.

#[derive(Debug, Default)]
pub struct RopeAppendRefImpl;

impl Implementation for RopeAppendRefImpl {
    fn name(&self) -> &'static str {
        "rope_append_ref"
    }

    fn target_compatible(&self, _profile: &TargetProfile) -> bool {
        true
    }

    fn matches(&self, fuf: &Fuf, seed: TileId, _profile: &TargetProfile) -> Option<MatchInfo> {
        let node = fuf.get(seed);
        if !is_rope_append_op(node.op) {
            return None;
        }
        // Boundary inputs: q, k, v upstream tiles (slots 0, 1, 2 of
        // the rope_append). Positions / rotary / kv_cache are externs,
        // sourced from `ForwardCtx`.
        let qkv: Vec<TileId> = node
            .inputs
            .iter()
            .take(3)
            .filter_map(|i| match i {
                FufInput::Tile { id, .. } => Some(*id),
                _ => None,
            })
            .collect();
        if qkv.len() != 3 {
            return None;
        }
        // Defer to `FusedQkvRope{Cache,Prefill}Impl` when the
        // upstream pattern matches: three Gemms (optionally through
        // a BiasAdd) sharing one activation, uniform bias presence.
        // Rejecting here mirrors `gemm_is_fusion_partner` on
        // `CutlassGemmImpl` / `CutlassGemvImpl` — singletons must
        // never steal a claim the fused impl owns, because their
        // output layouts don't match what the downstream
        // `AttentionPrefillContiguousImpl` / `AttentionViaCacheImpl`
        // expects (fused impls produce contiguous `[T, heads,
        // head_dim]` K/V; the singleton rotates in-place and leaves
        // K/V at `[T, heads*head_dim]`).
        if rope_append_has_fused_qkv_upstream(fuf, seed) {
            return None;
        }
        // Note: deferral to FusedQkvQkNormRopeCacheImpl is OFF because
        // that impl is not registered in starter_library (correctness
        // bug, see comment there). Singletons must claim the Qwen3/
        // Gemma3 RopeAppend tiles unconditionally.
        Some(MatchInfo {
            claimed_tiles: vec![seed],
            boundary_inputs: qkv,
            boundary_outputs: vec![seed],
        })
    }

    fn cost_us(&self, _m: &MatchInfo, ctx: &CostCtx) -> f64 {
        // RoPE + cache write are bandwidth-bound: read packed Q+K +
        // write rotated Q+K to same buffers + write K/V to cache.
        let m = ctx.num_tokens() as f64;
        let q_size = ctx.bounds.get("num_attention_heads").copied().unwrap_or(0)
            * ctx.bounds.get("head_dim").copied().unwrap_or(0);
        let kv_size = ctx.bounds.get("num_key_value_heads").copied().unwrap_or(0)
            * ctx.bounds.get("head_dim").copied().unwrap_or(0);
        let bytes = m * (2.0 * q_size as f64 + 4.0 * kv_size as f64) * BYTES_PER_ELEM;
        let bw_gb = ctx.profile.memory_bandwidth_gbps;
        if bw_gb > 0.0 {
            (bytes / (bw_gb * 1e9)) * 1e6
        } else {
            0.0
        }
    }

    fn resources(&self, _m: &MatchInfo) -> Resources {
        Resources::ZERO
    }

    fn launch_kind(&self) -> LaunchKind {
        LaunchKind::HostCallback
    }

    fn supported_input_handoffs(&self) -> &[Handoff] {
        const H: &[Handoff] = &[Handoff::StreamOrder, Handoff::StreamEvent];
        H
    }

    fn supported_output_handoffs(&self) -> &[Handoff] {
        const H: &[Handoff] = &[Handoff::StreamOrder, Handoff::StreamEvent];
        H
    }

    fn input_layouts(&self, m: &MatchInfo) -> Vec<Layout> {
        vec![Layout::RowMajorBf16; m.boundary_inputs.len()]
    }

    fn output_layouts(&self, m: &MatchInfo) -> Vec<Layout> {
        vec![Layout::RowMajorBf16; m.boundary_outputs.len()]
    }

    fn is_compute_bound(&self) -> bool {
        false
    }

    #[allow(clippy::type_complexity)]
    fn output_alias(
        &self,
        claimed_tiles: &[TileId],
        fuf: &Fuf,
    ) -> Vec<((TileId, u8), Option<(TileId, u8)>)> {
        // All three outputs (q', k', v') are 3D TensorView reshapes
        // over the upstream tile storage — no new allocation, no
        // move. Upstream Q may itself be a TensorView (Qwen3's
        // flatten-back Reshape between QK-norm and rope_append) or
        // an OwnedTensor (architectures without QK-norm that land
        // here for other reasons), so we never consume — the drop
        // pass resolves aliases through to the ultimate owner and
        // keeps it alive until every downstream use is done.
        let rope_id = claimed_tiles[0];
        let node = fuf.get(rope_id);
        let q_src = node.inputs.first().and_then(|i| match i {
            FufInput::Tile { id, slot } => Some((*id, *slot)),
            _ => None,
        });
        let k_src = node.inputs.get(1).and_then(|i| match i {
            FufInput::Tile { id, slot } => Some((*id, *slot)),
            _ => None,
        });
        let v_src = node.inputs.get(2).and_then(|i| match i {
            FufInput::Tile { id, slot } => Some((*id, *slot)),
            _ => None,
        });
        vec![
            ((rope_id, 0), q_src),
            ((rope_id, 1), k_src),
            ((rope_id, 2), v_src),
        ]
    }

    fn emit_call(&self, ctx: &EmitCtx) -> TokenStream {
        let rope_id = ctx.primary();
        let node = ctx.fuf.get(rope_id);
        let interleaved = node.op == OpKind::RopeAppendInterleaved;

        let q_upstream = ctx
            .input_tile_ident(rope_id, 0)
            .expect("RopeAppend input 0 (q) must be a Tile");
        let k_upstream = ctx
            .input_tile_ident(rope_id, 1)
            .expect("RopeAppend input 1 (k) must be a Tile");
        let v_upstream = ctx
            .input_tile_ident(rope_id, 2)
            .expect("RopeAppend input 2 (v) must be a Tile");

        let layer = node
            .inputs
            .iter()
            .find_map(|i| match i {
                FufInput::Extern {
                    kind: crate::classified::ExternKind::KvCache,
                    index: Some(layer),
                } => Some(*layer),
                _ => None,
            })
            .expect("RopeAppend has a kv_cache extern with a concrete layer index")
            as usize;

        let head_dim = ctx.bound("head_dim") as usize;
        let num_q_heads = ctx.bound("num_attention_heads") as usize;
        let num_kv_heads = ctx.bound("num_key_value_heads") as usize;

        let q_out = ctx.output_ident(rope_id, 0);
        let k_out = ctx.output_ident(rope_id, 1);
        let v_out = ctx.output_ident(rope_id, 2);

        let rotary_cos_sin = rotary_cos_sin_tokens(ctx.fuf, ctx.claimed_tiles);

        // Interleaved arches (Cohere) pair adjacent (2i, 2i+1) elements
        // for rotation; NeoX arches pair (i, i + half_dim). Same kernel
        // signature, different math — branch at emit time.
        let rope_call = if interleaved {
            quote! {
                ::ferrite_kernels::kernels::rotary_embedding_interleaved_inplace(
                    *#q_upstream,
                    *#k_upstream,
                    *ctx.positions,
                    #rotary_cos_sin,
                    #head_dim,
                    device.compute_stream,
                );
            }
        } else {
            quote! {
                ::ferrite_kernels::kernels::rotary_embedding_inplace(
                    *#q_upstream,
                    *#k_upstream,
                    *ctx.positions,
                    #rotary_cos_sin,
                    #head_dim,
                    device.compute_stream,
                );
            }
        };

        quote! {
            unsafe {
                // RoPE runs against the flat 2D `[T, heads*head_dim]`
                // layout — rotary_embedding_*_inplace reads dim(0)/dim(1)
                // to derive tokens × total dim.
                #rope_call
                let nt = (*#k_upstream).dim(0);
                let k_3d = (*#k_upstream)
                    .as_view()
                    .reshape(&[nt, #num_kv_heads, #head_dim]);
                let v_3d = (*#v_upstream)
                    .as_view()
                    .reshape(&[nt, #num_kv_heads, #head_dim]);
                ::ferrite_kernels::kernels::reshape_and_cache(
                    *k_3d,
                    *v_3d,
                    *ctx.kv_cache.k_cache(#layer),
                    *ctx.kv_cache.v_cache(#layer),
                    *ctx.slot_mapping,
                    ctx.kv_cache.block_size,
                    device.compute_stream,
                );
            }
            // Expose Q/K/V as 3D TensorViews so the downstream
            // attention impls (which read `q.dim(0,1,2)` as
            // `[total_q, num_heads, head_dim]`) match the layout
            // produced by the fused `FusedQkvRope*` impls.
            let #q_out = unsafe {
                let nt = (*#q_upstream).dim(0);
                (*#q_upstream)
                    .as_view()
                    .reshape(&[nt, #num_q_heads, #head_dim])
            };
            let #k_out = unsafe {
                let nt = (*#k_upstream).dim(0);
                (*#k_upstream)
                    .as_view()
                    .reshape(&[nt, #num_kv_heads, #head_dim])
            };
            let #v_out = unsafe {
                let nt = (*#v_upstream).dim(0);
                (*#v_upstream)
                    .as_view()
                    .reshape(&[nt, #num_kv_heads, #head_dim])
            };
        }
    }
}

// ── FusedQkvRopePrefillImpl ──────────────────────────────────────
//
// Same 4-tile pattern as `FusedQkvRopeCacheImpl` `(Gemm, Gemm, Gemm,
// RopeAppend)`, but for the prefill path: emits the split `fused_qkv_rope`
// kernel (returns Q, K, V as contiguous OwnedTensors; does NOT write
// to the paged cache) followed by an explicit `write_kv_cache` so
// the cache is populated for future decode steps. The contiguous
// K/V tensors remain available as the tile's slot-1 and slot-2
// outputs so the downstream prefill-attention impl reads them
// directly instead of going through the cache.
//
// `WorkloadConstraint::NumTokensRange { min: 2, max: u32::MAX }` —
// paired with `AttentionPrefillContiguousImpl` for the prefill
// bucket. Decode goes through `FusedQkvRopeCacheImpl`.

#[derive(Debug, Default)]
pub struct FusedQkvRopePrefillImpl;

impl Implementation for FusedQkvRopePrefillImpl {
    fn name(&self) -> &'static str {
        "fused_qkv_rope_prefill"
    }

    fn target_compatible(&self, _profile: &TargetProfile) -> bool {
        true
    }

    fn workload_constraint(&self) -> WorkloadConstraint {
        WorkloadConstraint::NumTokensRange {
            min: 2,
            max: u32::MAX,
        }
    }

    fn matches(&self, fuf: &Fuf, seed: TileId, profile: &TargetProfile) -> Option<MatchInfo> {
        // Same pattern as the cache variant — delegate to it so the
        // matcher stays in one place.
        FusedQkvRopeCacheImpl.matches(fuf, seed, profile)
    }

    fn cost_us(&self, m: &MatchInfo, ctx: &CostCtx) -> f64 {
        // Same flops + bandwidth model as the decode-fused variant;
        // the kernel split is different but the work is the same.
        FusedQkvRopeCacheImpl.cost_us(m, ctx)
    }

    fn resources(&self, _m: &MatchInfo) -> Resources {
        Resources::ZERO
    }

    fn launch_kind(&self) -> LaunchKind {
        LaunchKind::HostCallback
    }

    fn supported_input_handoffs(&self) -> &[Handoff] {
        const H: &[Handoff] = &[Handoff::StreamOrder, Handoff::StreamEvent];
        H
    }

    fn supported_output_handoffs(&self) -> &[Handoff] {
        const H: &[Handoff] = &[Handoff::StreamOrder, Handoff::StreamEvent];
        H
    }

    fn input_layouts(&self, m: &MatchInfo) -> Vec<Layout> {
        vec![Layout::RowMajorBf16; m.boundary_inputs.len()]
    }

    fn output_layouts(&self, m: &MatchInfo) -> Vec<Layout> {
        vec![Layout::RowMajorBf16; m.boundary_outputs.len()]
    }

    fn is_compute_bound(&self) -> bool {
        true
    }

    fn required_weights(
        &self,
        claimed_tiles: &[TileId],
        fuf: &Fuf,
        program: &Program,
    ) -> Vec<WeightAccessor> {
        // Same packed QKV weight as the decode variant.
        FusedQkvRopeCacheImpl.required_weights(claimed_tiles, fuf, program)
    }

    fn emit_call(&self, ctx: &EmitCtx) -> TokenStream {
        let rope_id = *ctx
            .claimed_tiles
            .iter()
            .find(|t| is_rope_append_op(ctx.fuf.get(**t).op))
            .expect("claim must contain a RopeAppend");
        let rope_node = ctx.fuf.get(rope_id);
        let interleaved = rope_node.op == OpKind::RopeAppendInterleaved;

        // Resolve rope's three tile inputs to underlying Gemms, stripping
        // optional BiasAdd wrappers (matches `FusedQkvRopeCacheImpl`).
        let qkv_raw: Vec<TileId> = rope_node
            .inputs
            .iter()
            .take(3)
            .filter_map(|i| match i {
                FufInput::Tile { id, .. } => Some(*id),
                _ => None,
            })
            .collect();
        assert_eq!(qkv_raw.len(), 3, "rope has three tile inputs (q, k, v)");
        let resolved: Vec<(TileId, Option<TileId>)> = qkv_raw
            .iter()
            .map(|t| {
                unwrap_gemm_through_bias(ctx.fuf, *t)
                    .expect("claim-time check guarantees Gemm-or-BiasAdd(Gemm)")
            })
            .collect();
        let biased = resolved[0].1.is_some();
        let q_gemm_id = resolved[0].0;
        let activation = ctx.input_expr(q_gemm_id, 0);

        // Source weights: 3 Gemm weights (biases auto-ride through
        // the packed LinearLayer). Must match `required_weights` so
        // `fused_accessor_name` resolves identically.
        let qkv_weights: Vec<(WeightId, Option<u64>)> = resolved
            .iter()
            .map(|(g, _)| first_weight_ref(ctx.fuf.get(*g)).expect("gemm weight"))
            .collect();
        let fused_name = fused_accessor_name(ctx.program, &qkv_weights);
        let weight_expr = ctx.weight_accessor(&fused_name);

        let num_q_heads = ctx.bound("num_attention_heads") as usize;
        let num_kv_heads = ctx.bound("num_key_value_heads") as usize;
        let head_dim = ctx.bound("head_dim") as usize;
        let q_size = num_q_heads * head_dim;
        let kv_size = num_kv_heads * head_dim;

        let layer = rope_kv_cache_layer(rope_node)
            .expect("RopeAppend has a KvCache extern input with a concrete layer index")
            as usize;

        let q_out = ctx.output_ident(rope_id, 0);
        let k_out = ctx.output_ident(rope_id, 1);
        let v_out = ctx.output_ident(rope_id, 2);

        let bias_assert = if biased {
            quote! {
                debug_assert!(
                    (#weight_expr).dense_bias().is_some(),
                    "FusedQkvRopePrefillImpl: DSL `bias_add` on QKV claimed but \
                     packed LinearLayer has no bias — check safetensors path"
                );
            }
        } else {
            quote! {}
        };

        let rotary_cos_sin = rotary_cos_sin_tokens(ctx.fuf, ctx.claimed_tiles);

        // Cohere-style interleaved pairing has its own split-rope kernel.
        let split_call = if interleaved {
            quote! {
                ::ferrite_kernels::kernels::fused_qkv_interleaved_rope(
                    *qkv_packed,
                    *ctx.positions,
                    #rotary_cos_sin,
                    #q_size,
                    #kv_size,
                    #num_q_heads,
                    #num_kv_heads,
                    #head_dim,
                    &mut device.caching,
                    device.compute_stream,
                )
            }
        } else {
            quote! {
                ::ferrite_kernels::kernels::fused_qkv_rope(
                    *qkv_packed,
                    *ctx.positions,
                    #rotary_cos_sin,
                    #q_size,
                    #kv_size,
                    #num_q_heads,
                    #num_kv_heads,
                    #head_dim,
                    &mut device.caching,
                    device.compute_stream,
                )
            }
        };

        quote! {
            // Fused QKV GEMM → packed [num_tokens, q + 2*kv] tensor,
            // then RoPE-split into contiguous Q/K/V. The packed QKV
            // buffer drops as soon as the rope-split kernel returns.
            //
            // Unlike `fused_qkv_rope_cache` this does NOT write to the
            // paged cache — the cache write is a separate step below
            // so the K/V OwnedTensors remain available for contiguous
            // prefill attention.
            let (#q_out, #k_out, #v_out) = unsafe {
                #bias_assert
                let qkv_packed = (#weight_expr).forward(
                    #activation,
                    &mut device.cublas,
                    &mut device.caching,
                    device.compute_stream,
                );
                #split_call
            };
            // Commit rotated K/V into the paged cache at slot_mapping
            // so subsequent decode steps read the correct values.
            unsafe {
                ::ferrite_kernels::attention_helpers::write_kv_cache(
                    (#k_out).view(),
                    (#v_out).view(),
                    ctx.slot_mapping,
                    ctx.kv_cache,
                    #layer,
                    device.compute_stream,
                );
            }
        }
    }
}

// ── AttentionPrefillContiguousImpl ───────────────────────────────
//
// Singleton matcher on `OpKind::Attention` for the prefill bucket.
// Reads Q, K, V from the DSL tile's slots 0/1/2 (the outputs of
// the upstream `FusedQkvRopePrefillImpl`, which are real contiguous
// OwnedTensors) and calls `flash_attn_contiguous`. This matches
// what vllm-cuda's hand-written prefill path does.
//
// `WorkloadConstraint::NumTokensRange { min: 2, max: u32::MAX }`.

#[derive(Debug, Default)]
pub struct AttentionPrefillContiguousImpl;

impl Implementation for AttentionPrefillContiguousImpl {
    fn name(&self) -> &'static str {
        "attention_prefill_contiguous"
    }

    fn target_compatible(&self, _profile: &TargetProfile) -> bool {
        true
    }

    fn workload_constraint(&self) -> WorkloadConstraint {
        WorkloadConstraint::NumTokensRange {
            min: 2,
            max: u32::MAX,
        }
    }

    fn matches(&self, fuf: &Fuf, seed: TileId, _profile: &TargetProfile) -> Option<MatchInfo> {
        single_tile_match(fuf, seed, OpKind::Attention)
    }

    fn cost_us(&self, m: &MatchInfo, ctx: &CostCtx) -> f64 {
        // Same calibrated FA2 CSV row as the paged decode variant —
        // the FA2 sweep emits one cost per (h, q, k, M, sk) cell and
        // doesn't distinguish decode/prefill. Falls back to the
        // analytic formula when no row matches.
        cost_attention_calibrated(m, ctx)
    }

    fn resources(&self, _m: &MatchInfo) -> Resources {
        Resources::ZERO
    }

    fn launch_kind(&self) -> LaunchKind {
        LaunchKind::HostCallback
    }

    fn supported_input_handoffs(&self) -> &[Handoff] {
        const H: &[Handoff] = &[Handoff::StreamOrder, Handoff::StreamEvent];
        H
    }

    fn supported_output_handoffs(&self) -> &[Handoff] {
        const H: &[Handoff] = &[Handoff::StreamOrder, Handoff::StreamEvent];
        H
    }

    fn input_layouts(&self, m: &MatchInfo) -> Vec<Layout> {
        vec![Layout::RowMajorBf16; m.boundary_inputs.len()]
    }

    fn output_layouts(&self, m: &MatchInfo) -> Vec<Layout> {
        vec![Layout::RowMajorBf16; m.boundary_outputs.len()]
    }

    fn is_compute_bound(&self) -> bool {
        true
    }

    fn emit_call(&self, ctx: &EmitCtx) -> TokenStream {
        let tile = ctx.primary();
        let out = ctx.output_ident(tile, 0);
        let node = ctx.fuf.get(tile);

        // Q, K, V from the DSL's `attention(q, k, v, ...)` — slots
        // 0/1/2 are the upstream rope-append tile's three outputs.
        let q_expr = ctx.input_expr(tile, 0);
        let k_expr = ctx.input_expr(tile, 1);
        let v_expr = ctx.input_expr(tile, 2);

        // Config-driven attention scale + softcap (see
        // `AttentionViaCacheImpl::emit_call` for the rationale).
        let scale_tokens = attention_scale_tokens(ctx);
        let softcap_tokens = attention_softcap_tokens(ctx);
        let q_size = (ctx.bound("num_attention_heads") * ctx.bound("head_dim")) as usize;
        // Flag is harmless when rotary_dim=0 (no fused RoPE in FA2 on
        // the prefill path), but pass the right value for consistency
        // and for any future prefill path that does fused RoPE.
        let layer = node.inputs.iter().find_map(|i| match i {
            FufInput::Extern {
                kind: crate::classified::ExternKind::KvCache,
                index: Some(layer),
            } => Some(*layer),
            _ => None,
        });
        let interleaved_lit = layer
            .map(|l| layer_rope_is_interleaved(ctx.fuf, l))
            .unwrap_or(false);

        quote! {
            let mut #out = unsafe {
                // Fresh-prefill flash attention reads K/V directly
                // from the contiguous tensors produced by
                // `fused_qkv_rope`. K is already rotated, so the
                // cos_sin_cache pointer is null (no fused RoPE
                // inside FA2). `cu_seqlens_k = cu_seqlens_q` for
                // fresh prefill — each sequence's K length equals
                // its Q length.
                ::ferrite_kernels::kernels::flash_attn_contiguous(
                    *#q_expr,
                    *#k_expr,
                    *#v_expr,
                    *ctx.cu_seqlens_q,
                    *ctx.cu_seqlens_q,
                    ctx.max_seqlen_q,
                    ctx.max_seqlen_k,
                    #scale_tokens,
                    true, // is_causal
                    #softcap_tokens,
                    -1,   // window_size_left (-1 = disabled; sliding variant covers non-(-1))
                    &mut device.caching,
                    device.compute_stream,
                    ::std::ptr::null::<u8>(),
                    0,     // rotary_dim
                    #interleaved_lit, // is_rotary_interleaved (rope flavor at this layer)
                )
            };
            // Flatten [num_tokens, num_q_heads, head_dim] → [num_tokens, q_size]
            // so the downstream o_proj gemm sees a 2D [M, K] input with the
            // right K dimension.
            unsafe {
                let nt = (*#out).dim(0);
                let dt = (*#out).dtype();
                #out.reshape(&[nt, #q_size], dt);
            }
        }
    }
}

// ── SlidingAttentionViaCacheImpl ─────────────────────────────────
//
// Mirror of [`AttentionViaCacheImpl`] for [`OpKind::SlidingAttention`].
// Identical kernel path (paged decode through
// `attention_decode_from_cache`) but passes `window_size_left` from
// the model config's `sliding_window` field, so the flash-attn
// kernel masks out positions beyond the window. Decode-only
// (`NumTokensRange { 1, 1 }`); the prefill counterpart is
// [`SlidingAttentionPrefillContiguousImpl`].

#[derive(Debug, Default)]
pub struct SlidingAttentionViaCacheImpl;

impl Implementation for SlidingAttentionViaCacheImpl {
    fn name(&self) -> &'static str {
        "sliding_attention_via_cache"
    }

    fn target_compatible(&self, _profile: &TargetProfile) -> bool {
        true
    }

    fn workload_constraint(&self) -> WorkloadConstraint {
        WorkloadConstraint::NumTokensRange { min: 1, max: 1 }
    }

    fn matches(&self, fuf: &Fuf, seed: TileId, _profile: &TargetProfile) -> Option<MatchInfo> {
        single_tile_match(fuf, seed, OpKind::SlidingAttention)
    }

    fn cost_us(&self, m: &MatchInfo, ctx: &CostCtx) -> f64 {
        cost_attention(m, ctx)
    }

    fn resources(&self, _m: &MatchInfo) -> Resources {
        Resources::ZERO
    }

    fn launch_kind(&self) -> LaunchKind {
        LaunchKind::HostCallback
    }

    fn supported_input_handoffs(&self) -> &[Handoff] {
        const H: &[Handoff] = &[Handoff::StreamOrder, Handoff::StreamEvent];
        H
    }

    fn supported_output_handoffs(&self) -> &[Handoff] {
        const H: &[Handoff] = &[Handoff::StreamOrder, Handoff::StreamEvent];
        H
    }

    fn input_layouts(&self, m: &MatchInfo) -> Vec<Layout> {
        vec![Layout::RowMajorBf16; m.boundary_inputs.len()]
    }

    fn output_layouts(&self, m: &MatchInfo) -> Vec<Layout> {
        vec![Layout::RowMajorBf16; m.boundary_outputs.len()]
    }

    fn is_compute_bound(&self) -> bool {
        true
    }

    fn emit_call(&self, ctx: &EmitCtx) -> TokenStream {
        let tile = ctx.primary();
        let out = ctx.output_ident(tile, 0);
        let node = ctx.fuf.get(tile);
        let q_expr = ctx.input_expr(tile, 0);
        let layer = node
            .inputs
            .iter()
            .find_map(|i| match i {
                FufInput::Extern {
                    kind: crate::classified::ExternKind::KvCache,
                    index: Some(layer),
                } => Some(*layer),
                _ => None,
            })
            .expect("SlidingAttention has a kv_cache extern with a concrete layer index")
            as usize;

        let scale_tokens = attention_scale_tokens(ctx);
        let softcap_tokens = attention_softcap_tokens(ctx);
        let window_tokens = sliding_window_left_tokens(ctx);
        let q_size = (ctx.bound("num_attention_heads") * ctx.bound("head_dim")) as usize;

        let rotary_cos_sin = rotary_cos_sin_tokens(ctx.fuf, ctx.claimed_tiles);
        let interleaved_lit = layer_rope_is_interleaved(ctx.fuf, layer as u64);

        quote! {
            let mut #out = unsafe {
                let has_spans = !ctx.kv_cache.block_unrotated_gpu().is_null();
                let (cos_sin_ptr, rotary_dim) = if has_spans {
                    (
                        #rotary_cos_sin.raw_ptr() as *const u8,
                        #rotary_cos_sin.dim(1),
                    )
                } else {
                    (::std::ptr::null::<u8>(), 0)
                };
                ::ferrite_kernels::attention_helpers::attention_decode_from_cache(
                    #q_expr,
                    ctx.cu_seqlens_q,
                    ctx.seqused_k,
                    ctx.block_table,
                    ctx.max_seqlen_q,
                    ctx.max_seqlen_k,
                    #scale_tokens,
                    #softcap_tokens,
                    #window_tokens,
                    ctx.kv_cache,
                    #layer,
                    device.num_sm,
                    &mut device.caching,
                    device.compute_stream,
                    cos_sin_ptr,
                    rotary_dim,
                    #interleaved_lit, // is_rotary_interleaved (rope flavor at this layer)
                )
            };
            unsafe {
                let nt = (*#out).dim(0);
                let dt = (*#out).dtype();
                #out.reshape(&[nt, #q_size], dt);
            }
        }
    }
}

// ── SlidingAttentionPrefillContiguousImpl ────────────────────────
//
// Prefill counterpart of [`SlidingAttentionViaCacheImpl`]. Reads Q,
// K, V from slots 0/1/2 (populated by an upstream prefill QKV impl)
// and calls `flash_attn_contiguous` with the config-driven window.

#[derive(Debug, Default)]
pub struct SlidingAttentionPrefillContiguousImpl;

impl Implementation for SlidingAttentionPrefillContiguousImpl {
    fn name(&self) -> &'static str {
        "sliding_attention_prefill_contiguous"
    }

    fn target_compatible(&self, _profile: &TargetProfile) -> bool {
        true
    }

    fn workload_constraint(&self) -> WorkloadConstraint {
        WorkloadConstraint::NumTokensRange {
            min: 2,
            max: u32::MAX,
        }
    }

    fn matches(&self, fuf: &Fuf, seed: TileId, _profile: &TargetProfile) -> Option<MatchInfo> {
        single_tile_match(fuf, seed, OpKind::SlidingAttention)
    }

    fn cost_us(&self, m: &MatchInfo, ctx: &CostCtx) -> f64 {
        cost_attention(m, ctx)
    }

    fn resources(&self, _m: &MatchInfo) -> Resources {
        Resources::ZERO
    }

    fn launch_kind(&self) -> LaunchKind {
        LaunchKind::HostCallback
    }

    fn supported_input_handoffs(&self) -> &[Handoff] {
        const H: &[Handoff] = &[Handoff::StreamOrder, Handoff::StreamEvent];
        H
    }

    fn supported_output_handoffs(&self) -> &[Handoff] {
        const H: &[Handoff] = &[Handoff::StreamOrder, Handoff::StreamEvent];
        H
    }

    fn input_layouts(&self, m: &MatchInfo) -> Vec<Layout> {
        vec![Layout::RowMajorBf16; m.boundary_inputs.len()]
    }

    fn output_layouts(&self, m: &MatchInfo) -> Vec<Layout> {
        vec![Layout::RowMajorBf16; m.boundary_outputs.len()]
    }

    fn is_compute_bound(&self) -> bool {
        true
    }

    fn emit_call(&self, ctx: &EmitCtx) -> TokenStream {
        let tile = ctx.primary();
        let out = ctx.output_ident(tile, 0);
        let node = ctx.fuf.get(tile);

        let q_expr = ctx.input_expr(tile, 0);
        let k_expr = ctx.input_expr(tile, 1);
        let v_expr = ctx.input_expr(tile, 2);

        let scale_tokens = attention_scale_tokens(ctx);
        let softcap_tokens = attention_softcap_tokens(ctx);
        let window_tokens = sliding_window_left_tokens(ctx);
        let q_size = (ctx.bound("num_attention_heads") * ctx.bound("head_dim")) as usize;
        // Flag is moot here (rotary_dim=0); pass for consistency.
        let layer = node.inputs.iter().find_map(|i| match i {
            FufInput::Extern {
                kind: crate::classified::ExternKind::KvCache,
                index: Some(layer),
            } => Some(*layer),
            _ => None,
        });
        let interleaved_lit = layer
            .map(|l| layer_rope_is_interleaved(ctx.fuf, l))
            .unwrap_or(false);

        quote! {
            let mut #out = unsafe {
                ::ferrite_kernels::kernels::flash_attn_contiguous(
                    *#q_expr,
                    *#k_expr,
                    *#v_expr,
                    *ctx.cu_seqlens_q,
                    *ctx.cu_seqlens_q,
                    ctx.max_seqlen_q,
                    ctx.max_seqlen_k,
                    #scale_tokens,
                    true, // is_causal
                    #softcap_tokens,
                    #window_tokens,
                    &mut device.caching,
                    device.compute_stream,
                    ::std::ptr::null::<u8>(),
                    0,     // rotary_dim
                    #interleaved_lit, // is_rotary_interleaved (rope flavor at this layer)
                )
            };
            unsafe {
                let nt = (*#out).dim(0);
                let dt = (*#out).dtype();
                #out.reshape(&[nt, #q_size], dt);
            }
        }
    }
}

// ── CutlassGemmImpl / CutlassGemvImpl ────────────────────────────
//
// Singleton matchers on `OpKind::Gemm` parameterised by the CUTLASS
// tile shape `(tile_m, tile_n, stages)`. Backed by
// `ferrite_kernels::cutlass::cutlass_gemm` / `cutlass_gemv` — the
// prior ferrite's hand-picked tile zoo, preserved byte-for-byte in
// the launch fns.
//
// Cost for `(workload M, weight [N, K])` is read directly from
// `target.cost_table` (`target_profiles/cost_<name>.csv`); variants
// missing a CSV row for a given shape report `f64::INFINITY` so the
// DP never picks a data-less kernel. Variants whose target has no
// CSV data at all are filtered out at `target_compatible` time.
//
// `matches` rejects Gemms whose downstream consumer is a fusion
// partner (RopeAppend / Silu / Mul), so the solver can never steer
// a QKV or gate/up gemm away from its fused impl — picking cutlass
// for the gate gemm would orphan the silu tile, which has no
// singleton kernel.

/// Every cutlass tile variant exported from
/// `vllm-cuda/csrc/cutlass_standalone_gemm.cu` with a matching
/// FFI declaration in `ferrite-kernels::cutlass`. Must stay in sync
/// with that file — add / remove tiles here and in the extern block
/// together.
/// Finite "don't pick me" cost for calibrated impls at shapes not
/// in their CSV. Large enough that any calibrated alternative wins,
/// small enough to keep `is_finite()` true (the DP rejects
/// non-finite costs with `SolveError::UnreachableCost`).
const UNCALIBRATED_COST_US: f64 = 1.0e9;

const CUTLASS_TILE_ZOO: &[(u32, u32, u32)] = &[
    (32, 64, 3),
    (32, 64, 4),
    (32, 128, 3),
    (32, 128, 4),
    (32, 256, 3),
    (64, 64, 3),
    (64, 64, 4),
    (64, 128, 3),
    (64, 128, 4),
    (128, 64, 3),
    (128, 64, 4),
    (128, 128, 3),
    (128, 128, 4),
    (128, 256, 3),
    (256, 64, 3),
    (256, 64, 4),
];

#[derive(Debug, Clone)]
pub struct CutlassGemmImpl {
    pub tile_m: u32,
    pub tile_n: u32,
    pub stages: u32,
}

impl CutlassGemmImpl {
    /// Same string as `static_name`, returned as `&'static str` so the
    /// hot `target_compatible` path doesn't allocate. The CSV column
    /// matches `static_name` by construction.
    fn csv_name(&self) -> &'static str {
        self.static_name()
    }

    fn static_name(&self) -> &'static str {
        // Names are compile-time-known per CUTLASS_TILE_ZOO entry.
        match (self.tile_m, self.tile_n, self.stages) {
            (32, 64, 3) => "cutlass_32x64_s3",
            (32, 64, 4) => "cutlass_32x64_s4",
            (32, 128, 3) => "cutlass_32x128_s3",
            (32, 128, 4) => "cutlass_32x128_s4",
            (32, 256, 3) => "cutlass_32x256_s3",
            (64, 64, 3) => "cutlass_64x64_s3",
            (64, 64, 4) => "cutlass_64x64_s4",
            (64, 128, 3) => "cutlass_64x128_s3",
            (64, 128, 4) => "cutlass_64x128_s4",
            (128, 64, 3) => "cutlass_128x64_s3",
            (128, 64, 4) => "cutlass_128x64_s4",
            (128, 128, 3) => "cutlass_128x128_s3",
            (128, 128, 4) => "cutlass_128x128_s4",
            (128, 256, 3) => "cutlass_128x256_s3",
            (256, 64, 3) => "cutlass_256x64_s3",
            (256, 64, 4) => "cutlass_256x64_s4",
            _ => "cutlass_unknown",
        }
    }
}

/// True if `node`'s output is consumed by any tile of the given op
/// kind. Used to reject cutlass matches on Gemms that feed a
/// fusion partner (RopeAppend / Silu / Mul).
fn output_feeds_op(fuf: &Fuf, tile: TileId, op: OpKind) -> bool {
    fuf.nodes
        .iter()
        .any(|n| n.op == op && consumes_tile(n, tile))
}

/// True if this Gemm tile is a fusion partner (Q/K/V of a
/// RopeAppend, or gate/up of the `silu(gate) * up` pattern, or a
/// bias-carrying gemm whose output feeds a `bias_add`). The cutlass
/// singletons must never claim these — their fused impls own them
/// and the downstream chain has no singleton kernel for the
/// fusion-partner op (Silu/Mul/RopeAppend/BiasAdd).
fn gemm_is_fusion_partner(fuf: &Fuf, seed: TileId) -> bool {
    output_feeds_op(fuf, seed, OpKind::RopeAppend)
        || output_feeds_op(fuf, seed, OpKind::RopeAppendInterleaved)
        || output_feeds_op(fuf, seed, OpKind::Silu)
        || output_feeds_op(fuf, seed, OpKind::Mul)
        || output_feeds_op(fuf, seed, OpKind::BiasAdd)
}

/// Evaluate the `(M, N, K)` of a Gemm tile for CSV cost lookup.
/// M comes from the current workload (bounds[`num_tokens`]), N from
/// the output's last dim, K from the activation-input tile's last
/// dim.
fn gemm_mnk(ctx: &CostCtx, node: &crate::fuf::FufNode) -> Option<(u32, u32, u32)> {
    let m = ctx.num_tokens() as u32;
    let out_shape = node.outputs.first().and_then(|s| ctx.eval_shape(s))?;
    if out_shape.len() != 2 {
        return None;
    }
    let n = *out_shape.last()? as u32;
    let k = node.inputs.iter().find_map(|inp| match inp {
        FufInput::Tile { id, slot } => {
            let up = ctx.fuf.get(*id);
            up.outputs
                .get(*slot as usize)
                .and_then(|s| ctx.eval_shape(s))
                .and_then(|v| v.last().copied())
        }
        _ => None,
    })? as u32;
    Some((m, n, k))
}

impl Implementation for CutlassGemmImpl {
    fn name(&self) -> &'static str {
        self.static_name()
    }

    fn target_compatible(&self, profile: &TargetProfile) -> bool {
        // Hot path — called per tile × per impl × per workload.
        // `has_kernel` is O(1); the previous `kernel_names()` scan
        // allocated a Vec<String> by cloning every cost-table entry's
        // key, which produced multi-second compile-time stalls on
        // large FUFs (1000+ tiles × 16 cutlass variants).
        profile.cost_table.has_kernel(self.csv_name())
    }

    fn workload_constraint(&self) -> WorkloadConstraint {
        // M=1 is GEMV territory — `CutlassGemvImpl` owns it.
        WorkloadConstraint::NumTokensRange {
            min: 2,
            max: u32::MAX,
        }
    }

    fn matches(&self, fuf: &Fuf, seed: TileId, _profile: &TargetProfile) -> Option<MatchInfo> {
        let info = single_tile_match(fuf, seed, OpKind::Gemm)?;
        // Dense kernel: reject AWQ storage.
        if !matches!(weight_storage_of(fuf.get(seed)), Some(StorageFormat::Dense)) {
            return None;
        }
        if gemm_is_fusion_partner(fuf, seed) {
            return None;
        }
        Some(info)
    }

    fn cost_us(&self, m: &MatchInfo, ctx: &CostCtx) -> f64 {
        let node = ctx.fuf.get(m.claimed_tiles[0]);
        let Some((mm, nn, kk)) = gemm_mnk(ctx, node) else {
            return f64::INFINITY;
        };
        ctx.profile
            .cost_us_for(self.csv_name(), mm, nn, kk)
            // No CSV row for this shape → emit a finite "too
            // expensive" sentinel so the DP skips this variant
            // without tripping the `!cost.is_finite()` guard.
            .unwrap_or(UNCALIBRATED_COST_US)
    }

    fn resources(&self, _m: &MatchInfo) -> Resources {
        Resources::ZERO
    }

    fn launch_kind(&self) -> LaunchKind {
        LaunchKind::RegularLaunch
    }

    fn supported_input_handoffs(&self) -> &[Handoff] {
        const H: &[Handoff] = &[Handoff::StreamOrder, Handoff::StreamEvent];
        H
    }

    fn supported_output_handoffs(&self) -> &[Handoff] {
        const H: &[Handoff] = &[Handoff::StreamOrder, Handoff::StreamEvent];
        H
    }

    fn input_layouts(&self, m: &MatchInfo) -> Vec<Layout> {
        vec![Layout::RowMajorBf16; m.boundary_inputs.len()]
    }

    fn output_layouts(&self, m: &MatchInfo) -> Vec<Layout> {
        vec![Layout::RowMajorBf16; m.boundary_outputs.len()]
    }

    fn is_compute_bound(&self) -> bool {
        true
    }

    fn emit_call(&self, ctx: &EmitCtx) -> TokenStream {
        let tile = ctx.primary();
        let out = ctx.output_ident(tile, 0);
        let x = ctx.input_expr(tile, 0);
        let w = ctx.input_expr(tile, 1);
        let tile_m = self.tile_m;
        let tile_n = self.tile_n;
        let stages = self.stages;
        quote! {
            let #out = unsafe {
                ::ferrite_kernels::cutlass::cutlass_gemm(
                    *(#x),
                    (#w).dense_weight(),
                    ::ferrite_kernels::cutlass::CutlassTile::new(#tile_m, #tile_n, #stages),
                    &mut device.caching,
                    device.compute_stream,
                )
            };
        }
    }
}

/// M=1 SIMT GEMV specialisation — the decode path's lm_head /
/// o_proj / down_proj.
#[derive(Debug, Default)]
pub struct CutlassGemvImpl;

impl Implementation for CutlassGemvImpl {
    fn name(&self) -> &'static str {
        "cutlass_gemv"
    }

    fn target_compatible(&self, profile: &TargetProfile) -> bool {
        profile
            .cost_table
            .kernel_names()
            .iter()
            .any(|k| k == "cutlass_gemv")
    }

    fn workload_constraint(&self) -> WorkloadConstraint {
        WorkloadConstraint::NumTokensRange { min: 1, max: 1 }
    }

    fn matches(&self, fuf: &Fuf, seed: TileId, _profile: &TargetProfile) -> Option<MatchInfo> {
        let info = single_tile_match(fuf, seed, OpKind::Gemm)?;
        // Dense kernel: reject AWQ storage.
        if !matches!(weight_storage_of(fuf.get(seed)), Some(StorageFormat::Dense)) {
            return None;
        }
        if gemm_is_fusion_partner(fuf, seed) {
            return None;
        }
        Some(info)
    }

    fn cost_us(&self, m: &MatchInfo, ctx: &CostCtx) -> f64 {
        let node = ctx.fuf.get(m.claimed_tiles[0]);
        let Some((mm, nn, kk)) = gemm_mnk(ctx, node) else {
            return f64::INFINITY;
        };
        ctx.profile
            .cost_us_for("cutlass_gemv", mm, nn, kk)
            .unwrap_or(UNCALIBRATED_COST_US)
    }

    fn resources(&self, _m: &MatchInfo) -> Resources {
        Resources::ZERO
    }

    fn launch_kind(&self) -> LaunchKind {
        LaunchKind::RegularLaunch
    }

    fn supported_input_handoffs(&self) -> &[Handoff] {
        const H: &[Handoff] = &[Handoff::StreamOrder, Handoff::StreamEvent];
        H
    }

    fn supported_output_handoffs(&self) -> &[Handoff] {
        const H: &[Handoff] = &[Handoff::StreamOrder, Handoff::StreamEvent];
        H
    }

    fn input_layouts(&self, m: &MatchInfo) -> Vec<Layout> {
        vec![Layout::RowMajorBf16; m.boundary_inputs.len()]
    }

    fn output_layouts(&self, m: &MatchInfo) -> Vec<Layout> {
        vec![Layout::RowMajorBf16; m.boundary_outputs.len()]
    }

    fn is_compute_bound(&self) -> bool {
        false
    }

    fn emit_call(&self, ctx: &EmitCtx) -> TokenStream {
        let tile = ctx.primary();
        let out = ctx.output_ident(tile, 0);
        let x = ctx.input_expr(tile, 0);
        let w = ctx.input_expr(tile, 1);
        quote! {
            let #out = unsafe {
                ::ferrite_kernels::cutlass::cutlass_gemv(
                    *(#x),
                    (#w).dense_weight(),
                    &mut device.caching,
                    device.compute_stream,
                )
            };
        }
    }
}

// ── Marlin* impls ────────────────────────────────────────────────
//
// Quant-aware mirrors of `GemmRefImpl`, `FusedGateUpSiluMulImpl`,
// `FusedQkvRopeCacheImpl`, and `FusedQkvRopePrefillImpl`. Each one:
//   - Structural match identical to its dense sibling (walks the
//     same tile pattern off the seed).
//   - Weight storage gate **inverted**: requires every participating
//     Gemm's weight input to be Marlin-consumable (AWQ or GPTQ),
//     not `Dense`. The dense variants already reject quant storage,
//     so the solver picks exactly one family per model.
//   - `required_weights` declares the packed accessor with
//     `rust_type = MarlinLinear`. AWQ and GPTQ both concat their
//     qweight/scales/(qzeros|g_idx) along the N axis, so the fused
//     QKV / fused gate-up case collapses to **one** `MarlinLinear`
//     — same shape the dense fused accessor uses (one `LinearLayer`).
//   - `emit_call` invokes `MarlinLinear::forward(x, alloc, stream)`.
//     No cublas handle — marlin's kernel owns the matmul.

/// Does `tile` have a Gemm with a Marlin-consumable storage (AWQ or
/// GPTQ)? Quant-aware impls gate on this at the seed and at every
/// fused Gemm they claim. The Marlin kernel is agnostic to the
/// source format after repack — AWQ's uint4 and GPTQ's uint4b8 are
/// both first-class `b_type_id`s in the kernel — so any impl that
/// emits `MarlinLinear::forward` accepts either storage.
fn is_marlin_gemm(fuf: &Fuf, tile: TileId) -> bool {
    let node = fuf.get(tile);
    node.op == OpKind::Gemm
        && matches!(
            weight_storage_of(node),
            Some(StorageFormat::Awq { .. } | StorageFormat::Gptq { .. })
        )
}

/// BitsAndBytes 4-bit counterpart of [`is_marlin_gemm`]. A Gemm
/// whose weight resolves to `StorageFormat::Bnb4 { .. }` — the
/// Bnb4bit impl family claims these; Marlin + dense impls reject
/// them via their own storage gates.
fn is_bnb4_gemm(fuf: &Fuf, tile: TileId) -> bool {
    let node = fuf.get(tile);
    node.op == OpKind::Gemm && matches!(weight_storage_of(node), Some(StorageFormat::Bnb4 { .. }))
}

/// FP8 counterpart of [`is_marlin_gemm`]. A Gemm whose weight
/// resolves to `StorageFormat::Fp8 { .. }` — the Fp8 impl family
/// claims these; dense / Marlin / Bnb4 impls reject them via their
/// own storage gates.
fn is_fp8_gemm(fuf: &Fuf, tile: TileId) -> bool {
    let node = fuf.get(tile);
    node.op == OpKind::Gemm && matches!(weight_storage_of(node), Some(StorageFormat::Fp8 { .. }))
}

/// Pick the accessor Rust type for an FP8 GEMM — `Fp8BlockLinear`
/// for blockwise (DeepSeek-style 128×128), `Fp8Linear` for per-tensor
/// / per-channel. Both expose the same `forward(x, cublas, alloc,
/// stream) -> OwnedTensor` signature, so every `emit_call` body above
/// stays identical; only the `required_weights` rust_type and the
/// generated `Weights::load` call differ between the two storages.
fn fp8_accessor_type_for(fuf: &Fuf, gemm_tile: TileId) -> TokenStream {
    match weight_storage_of(fuf.get(gemm_tile)) {
        Some(StorageFormat::Fp8 {
            block_size: Some(_),
            ..
        }) => quote! { ::ferrite_kernels::layers::Fp8BlockLinear },
        _ => quote! { ::ferrite_kernels::layers::Fp8Linear },
    }
}

/// True if a `MarlinFusedQkvRope*Impl` would accept `seed` as one of
/// its Q/K/V Gemms — i.e., there exists a `RopeAppend` whose first
/// three tile inputs unwrap (optionally through `BiasAdd`) to three
/// Marlin Gemms sharing one activation, and `seed` is one of them.
/// Used by `MarlinGemmImpl` to defer without over-claiming.
fn marlin_fused_qkv_rope_would_match(fuf: &Fuf, seed: TileId) -> bool {
    fuf.nodes.iter().any(|n| {
        if !is_rope_append_op(n.op) || n.inputs.len() < 3 {
            return false;
        }
        let qkv_raw: Vec<TileId> = n
            .inputs
            .iter()
            .take(3)
            .filter_map(|i| match i {
                FufInput::Tile { id, .. } => Some(*id),
                _ => None,
            })
            .collect();
        if qkv_raw.len() != 3 {
            return false;
        }
        let Some(resolved): Option<Vec<(TileId, Option<TileId>)>> = qkv_raw
            .iter()
            .map(|t| unwrap_gemm_through_bias(fuf, *t))
            .collect()
        else {
            return false;
        };
        let biased = resolved[0].1.is_some();
        if resolved.iter().any(|r| r.1.is_some() != biased) {
            return false;
        }
        let gemms: Vec<TileId> = resolved.iter().map(|r| r.0).collect();
        if !gemms.contains(&seed) {
            return false;
        }
        if gemms.iter().any(|t| !is_marlin_gemm(fuf, *t)) {
            return false;
        }
        let act = first_tile_input(fuf.get(gemms[0]));
        act.is_some() && gemms.iter().all(|t| first_tile_input(fuf.get(*t)) == act)
    })
}

/// True if `MarlinFusedGateUpSiluMulImpl` (or the Gelu variant) would
/// accept `seed` as gate or up gemm — `seed` is a Marlin Gemm whose
/// output feeds `Silu` or `Gelu`, or whose output is the "up" half of
/// a `(silu_or_gelu(gate) * up)` pair. Keeps the deference tight so
/// non-SwiGLU/GELU `Mul` patterns (e.g. Granite's scalar multiplier)
/// don't trigger it.
fn marlin_fused_gate_up_would_match(fuf: &Fuf, seed: TileId) -> bool {
    // gate gemm: output feeds Silu or Gelu which feeds Mul with a
    // sibling gemm as the other input.
    for n in &fuf.nodes {
        if !matches!(n.op, OpKind::Silu | OpKind::Gelu) {
            continue;
        }
        if !consumes_tile(n, seed) {
            // seed isn't the gate; check if seed is the up gemm
            // that feeds the Mul alongside this Silu/Gelu.
            let act_id = n.id;
            for m in &fuf.nodes {
                if m.op != OpKind::Mul || !consumes_tile(m, act_id) {
                    continue;
                }
                let up_is_seed = m.inputs.iter().any(|i| match i {
                    FufInput::Tile { id, .. } => *id == seed && *id != act_id,
                    _ => false,
                });
                if up_is_seed && is_marlin_gemm(fuf, seed) {
                    return true;
                }
            }
            continue;
        }
        // seed feeds the Silu/Gelu — check that Silu/Gelu's output
        // feeds a Mul paired with another Marlin Gemm.
        let act_id = n.id;
        let has_mul_pair = fuf.nodes.iter().any(|m| {
            if m.op != OpKind::Mul || !consumes_tile(m, act_id) {
                return false;
            }
            m.inputs.iter().any(|i| match i {
                FufInput::Tile { id, .. } => *id != act_id && is_marlin_gemm(fuf, *id),
                _ => false,
            })
        });
        if has_mul_pair {
            return true;
        }
    }
    false
}

/// Singleton Marlin GEMM — the AWQ counterpart of `GemmRefImpl`.
#[derive(Debug, Default)]
pub struct MarlinGemmImpl;

impl Implementation for MarlinGemmImpl {
    fn name(&self) -> &'static str {
        "marlin_gemm"
    }

    fn target_compatible(&self, _profile: &TargetProfile) -> bool {
        true
    }

    fn matches(&self, fuf: &Fuf, seed: TileId, _profile: &TargetProfile) -> Option<MatchInfo> {
        let info = single_tile_match(fuf, seed, OpKind::Gemm)?;
        if !is_marlin_gemm(fuf, seed) {
            return None;
        }
        // Precision-gate the fusion-partner deference. The old
        // `gemm_is_fusion_partner` check deferred to any Gemm
        // feeding RopeAppend/Silu/Mul/BiasAdd. That was too eager:
        //   - Granite's `o_proj * scalar(residual_multiplier)` is
        //     a `ScalarMul`, not the `silu(gate)*up` shape —
        //     nothing would claim the pair, `UnclaimedTile`.
        //   - Qwen3's per-head QK-norm interrupts the `(Gemm,
        //     Gemm, Gemm, RopeAppend)` adjacency that
        //     `MarlinFusedQkvRope*Impl` looks for — same orphan
        //     outcome.
        // Defer only when a fused sibling would actually match at
        // this seed. `marlin_fused_qkv_rope_would_match` checks
        // the exact `(Gemm, Gemm, Gemm, RopeAppend)` adjacency
        // (optionally through `BiasAdd`); the gate-up check
        // pattern-matches the `(Gemm, Gemm, Silu, Mul)` shape.
        if marlin_fused_qkv_rope_would_match(fuf, seed)
            || marlin_fused_gate_up_would_match(fuf, seed)
        {
            return None;
        }
        Some(info)
    }

    fn cost_us(&self, m: &MatchInfo, ctx: &CostCtx) -> f64 {
        // No CSV baseline for marlin today — reuse the cuBLAS
        // analytical estimate. Parity with the future
        // `cost_<target>.csv` lookup will land alongside the marlin
        // calibration sweep.
        cost_gemm(m, ctx)
    }

    fn resources(&self, _m: &MatchInfo) -> Resources {
        Resources::ZERO
    }

    fn launch_kind(&self) -> LaunchKind {
        LaunchKind::HostCallback
    }

    fn supported_input_handoffs(&self) -> &[Handoff] {
        const H: &[Handoff] = &[Handoff::StreamOrder, Handoff::StreamEvent];
        H
    }

    fn supported_output_handoffs(&self) -> &[Handoff] {
        const H: &[Handoff] = &[Handoff::StreamOrder, Handoff::StreamEvent];
        H
    }

    fn input_layouts(&self, m: &MatchInfo) -> Vec<Layout> {
        vec![Layout::RowMajorBf16; m.boundary_inputs.len()]
    }

    fn output_layouts(&self, m: &MatchInfo) -> Vec<Layout> {
        vec![Layout::RowMajorBf16; m.boundary_outputs.len()]
    }

    fn is_compute_bound(&self) -> bool {
        true
    }

    fn required_weights(
        &self,
        claimed_tiles: &[TileId],
        fuf: &Fuf,
        program: &Program,
    ) -> Vec<WeightAccessor> {
        // Same single-weight accessor as the dense default, but
        // typed `MarlinLinear` so codegen routes the FieldLoad
        // through `MarlinLinear::load_awq` or `load_gptq` per the
        // source weight's storage format.
        let tile = claimed_tiles[0];
        let (wid, index) = first_weight_ref(fuf.get(tile)).expect("Gemm has a weight input");
        let name = weight_field_name(program, wid, index);
        vec![WeightAccessor {
            name,
            rust_type: quote! { ::ferrite_kernels::layers::MarlinLinear },
            source_weights: vec![(wid, index)],
        }]
    }

    fn emit_call(&self, ctx: &EmitCtx) -> TokenStream {
        let tile = ctx.primary();
        let out = ctx.output_ident(tile, 0);
        let x = ctx.input_expr(tile, 0);
        let w = ctx.input_expr(tile, 1);
        quote! {
            let #out = unsafe {
                (#w).forward(
                    #x,
                    &mut device.caching,
                    device.compute_stream,
                )
            };
        }
    }
}

/// Marlin fused gate/up + SwiGLU — the AWQ counterpart of
/// `FusedGateUpSiluMulImpl`. AWQ metadata concats along dim N, so
/// the fused accessor is a single `MarlinLinear` (not two).
#[derive(Debug, Default)]
pub struct MarlinFusedGateUpSiluMulImpl;

impl Implementation for MarlinFusedGateUpSiluMulImpl {
    fn name(&self) -> &'static str {
        "marlin_fused_gate_up_silu_mul"
    }

    fn target_compatible(&self, _profile: &TargetProfile) -> bool {
        true
    }

    fn matches(&self, fuf: &Fuf, seed: TileId, _profile: &TargetProfile) -> Option<MatchInfo> {
        // Mirror of `FusedGateUpSiluMulImpl.matches`; see that impl
        // for the pattern-walking rationale. The only structural
        // change is the storage gate: BOTH Gemms must be AWQ.
        let gate_gemm = fuf.get(seed);
        if !is_marlin_gemm(fuf, seed) {
            return None;
        }
        let silu_node = fuf
            .nodes
            .iter()
            .find(|n| n.op == OpKind::Silu && consumes_tile(n, seed))?;
        let silu_id = silu_node.id;
        let mul_node = fuf
            .nodes
            .iter()
            .find(|n| n.op == OpKind::Mul && consumes_tile(n, silu_id))?;
        let mul_id = mul_node.id;
        let up_gemm_id = mul_node.inputs.iter().find_map(|i| match i {
            FufInput::Tile { id, .. } if *id != silu_id => Some(*id),
            _ => None,
        })?;
        if !is_marlin_gemm(fuf, up_gemm_id) {
            return None;
        }
        if first_tile_input(gate_gemm)? != first_tile_input(fuf.get(up_gemm_id))? {
            return None;
        }
        let mut claimed = [seed, up_gemm_id, silu_id, mul_id];
        claimed.sort();
        let claimed = claimed.to_vec();
        let activation_tile = first_tile_input(gate_gemm)?.0;
        Some(MatchInfo {
            claimed_tiles: claimed,
            boundary_inputs: vec![activation_tile],
            boundary_outputs: vec![mul_id],
        })
    }

    fn cost_us(&self, m: &MatchInfo, ctx: &CostCtx) -> f64 {
        FusedGateUpSiluMulImpl.cost_us(m, ctx)
    }

    fn resources(&self, _m: &MatchInfo) -> Resources {
        Resources::ZERO
    }

    fn launch_kind(&self) -> LaunchKind {
        LaunchKind::HostCallback
    }

    fn supported_input_handoffs(&self) -> &[Handoff] {
        const H: &[Handoff] = &[Handoff::StreamOrder, Handoff::StreamEvent];
        H
    }

    fn supported_output_handoffs(&self) -> &[Handoff] {
        const H: &[Handoff] = &[Handoff::StreamOrder, Handoff::StreamEvent];
        H
    }

    fn input_layouts(&self, m: &MatchInfo) -> Vec<Layout> {
        vec![Layout::RowMajorBf16; m.boundary_inputs.len()]
    }

    fn output_layouts(&self, m: &MatchInfo) -> Vec<Layout> {
        vec![Layout::RowMajorBf16; m.boundary_outputs.len()]
    }

    fn is_compute_bound(&self) -> bool {
        true
    }

    fn required_weights(
        &self,
        claimed_tiles: &[TileId],
        fuf: &Fuf,
        program: &Program,
    ) -> Vec<WeightAccessor> {
        // Fused accessor covers gate + up; declared `MarlinLinear`
        // so codegen emits `MarlinLinear::load_awq_concat` or
        // `load_gptq_concat` depending on storage format.
        let sources: Vec<(WeightId, Option<u64>)> = claimed_tiles
            .iter()
            .filter_map(|t| {
                let n = fuf.get(*t);
                if n.op == OpKind::Gemm {
                    first_weight_ref(n)
                } else {
                    None
                }
            })
            .collect();
        let name = fused_accessor_name(program, &sources);
        vec![WeightAccessor {
            name,
            rust_type: quote! { ::ferrite_kernels::layers::MarlinLinear },
            source_weights: sources,
        }]
    }

    fn emit_call(&self, ctx: &EmitCtx) -> TokenStream {
        // Structurally identical to `FusedGateUpSiluMulImpl::emit_call`
        // but the fused MarlinLinear's `.forward` takes `(x, alloc,
        // stream)` — no cublas handle.
        let silu_id = *ctx
            .claimed_tiles
            .iter()
            .find(|t| ctx.fuf.get(**t).op == OpKind::Silu)
            .expect("fused gate/up/silu/mul claim must contain Silu");
        let mul_id = *ctx
            .claimed_tiles
            .iter()
            .find(|t| ctx.fuf.get(**t).op == OpKind::Mul)
            .expect("fused gate/up/silu/mul claim must contain Mul");
        let (gate_id, _) =
            first_tile_input(ctx.fuf.get(silu_id)).expect("silu has a tile input — the gate gemm");
        let up_id = ctx
            .claimed_tiles
            .iter()
            .copied()
            .find(|t| ctx.fuf.get(*t).op == OpKind::Gemm && *t != gate_id)
            .expect("claim contains a second Gemm — the up gemm");

        let activation = ctx.input_expr(gate_id, 0);

        let gate_w = first_weight_ref(ctx.fuf.get(gate_id)).expect("gate gemm has a weight");
        let up_w = first_weight_ref(ctx.fuf.get(up_id)).expect("up gemm has a weight");
        let fused_name = fused_accessor_name(ctx.program, &[gate_w, up_w]);
        let weight_expr = ctx.weight_accessor(&fused_name);

        let mul_out = ctx.output_ident(mul_id, 0);
        let intermediate = ctx.bound("intermediate_size") as usize;

        quote! {
            let #mul_out = unsafe {
                let gate_up = (#weight_expr).forward(
                    #activation,
                    &mut device.caching,
                    device.compute_stream,
                );
                ::ferrite_kernels::kernels::silu_and_mul_fused(
                    *gate_up,
                    #intermediate,
                    &mut device.caching,
                    device.compute_stream,
                )
            };
        }
    }
}

/// Marlin fused gate/up GELU mul — the Marlin counterpart of
/// [`FusedGateUpGeluMulImpl`] for Gemma2-style MLPs. Same 4-tile
/// (gate_gemm, up_gemm, Gelu, Mul) pattern as the Silu variant;
/// the diff is the activation op. Emits one `MarlinLinear::forward`
/// on a fused gate+up accessor, followed by
/// `gelu_and_mul_fused` — exactly what the dense GELU path does,
/// but reading from a MarlinLinear instead of a dense LinearLayer.
#[derive(Debug, Default)]
pub struct MarlinFusedGateUpGeluMulImpl;

impl Implementation for MarlinFusedGateUpGeluMulImpl {
    fn name(&self) -> &'static str {
        "marlin_fused_gate_up_gelu_mul"
    }

    fn target_compatible(&self, _profile: &TargetProfile) -> bool {
        true
    }

    fn matches(&self, fuf: &Fuf, seed: TileId, _profile: &TargetProfile) -> Option<MatchInfo> {
        // Mirror of `FusedGateUpGeluMulImpl.matches`; the only
        // structural change is the storage gate: BOTH Gemms must be
        // Marlin-consumable (AWQ or GPTQ).
        let gate_gemm = fuf.get(seed);
        if !is_marlin_gemm(fuf, seed) {
            return None;
        }
        let gelu_node = fuf
            .nodes
            .iter()
            .find(|n| n.op == OpKind::Gelu && consumes_tile(n, seed))?;
        let gelu_id = gelu_node.id;
        let mul_node = fuf
            .nodes
            .iter()
            .find(|n| n.op == OpKind::Mul && consumes_tile(n, gelu_id))?;
        let mul_id = mul_node.id;
        let up_gemm_id = mul_node.inputs.iter().find_map(|i| match i {
            FufInput::Tile { id, .. } if *id != gelu_id => Some(*id),
            _ => None,
        })?;
        if !is_marlin_gemm(fuf, up_gemm_id) {
            return None;
        }
        if first_tile_input(gate_gemm)? != first_tile_input(fuf.get(up_gemm_id))? {
            return None;
        }
        let mut claimed = [seed, up_gemm_id, gelu_id, mul_id];
        claimed.sort();
        let claimed = claimed.to_vec();
        let activation_tile = first_tile_input(gate_gemm)?.0;
        Some(MatchInfo {
            claimed_tiles: claimed,
            boundary_inputs: vec![activation_tile],
            boundary_outputs: vec![mul_id],
        })
    }

    fn cost_us(&self, m: &MatchInfo, ctx: &CostCtx) -> f64 {
        FusedGateUpGeluMulImpl.cost_us(m, ctx)
    }

    fn resources(&self, _m: &MatchInfo) -> Resources {
        Resources::ZERO
    }

    fn launch_kind(&self) -> LaunchKind {
        LaunchKind::HostCallback
    }

    fn supported_input_handoffs(&self) -> &[Handoff] {
        const H: &[Handoff] = &[Handoff::StreamOrder, Handoff::StreamEvent];
        H
    }

    fn supported_output_handoffs(&self) -> &[Handoff] {
        const H: &[Handoff] = &[Handoff::StreamOrder, Handoff::StreamEvent];
        H
    }

    fn input_layouts(&self, m: &MatchInfo) -> Vec<Layout> {
        vec![Layout::RowMajorBf16; m.boundary_inputs.len()]
    }

    fn output_layouts(&self, m: &MatchInfo) -> Vec<Layout> {
        vec![Layout::RowMajorBf16; m.boundary_outputs.len()]
    }

    fn is_compute_bound(&self) -> bool {
        true
    }

    fn required_weights(
        &self,
        claimed_tiles: &[TileId],
        fuf: &Fuf,
        program: &Program,
    ) -> Vec<WeightAccessor> {
        let sources: Vec<(WeightId, Option<u64>)> = claimed_tiles
            .iter()
            .filter_map(|t| {
                let n = fuf.get(*t);
                if n.op == OpKind::Gemm {
                    first_weight_ref(n)
                } else {
                    None
                }
            })
            .collect();
        let name = fused_accessor_name(program, &sources);
        vec![WeightAccessor {
            name,
            rust_type: quote! { ::ferrite_kernels::layers::MarlinLinear },
            source_weights: sources,
        }]
    }

    fn emit_call(&self, ctx: &EmitCtx) -> TokenStream {
        // Structurally identical to `FusedGateUpGeluMulImpl::emit_call`
        // and to the Silu Marlin variant; only the activation kernel
        // name changes.
        let gelu_id = *ctx
            .claimed_tiles
            .iter()
            .find(|t| ctx.fuf.get(**t).op == OpKind::Gelu)
            .expect("fused gate/up/gelu/mul claim must contain Gelu");
        let mul_id = *ctx
            .claimed_tiles
            .iter()
            .find(|t| ctx.fuf.get(**t).op == OpKind::Mul)
            .expect("fused gate/up/gelu/mul claim must contain Mul");
        let (gate_id, _) =
            first_tile_input(ctx.fuf.get(gelu_id)).expect("gelu has a tile input — the gate gemm");
        let up_id = ctx
            .claimed_tiles
            .iter()
            .copied()
            .find(|t| ctx.fuf.get(*t).op == OpKind::Gemm && *t != gate_id)
            .expect("claim contains a second Gemm — the up gemm");

        let activation = ctx.input_expr(gate_id, 0);

        let gate_w = first_weight_ref(ctx.fuf.get(gate_id)).expect("gate gemm has a weight");
        let up_w = first_weight_ref(ctx.fuf.get(up_id)).expect("up gemm has a weight");
        let fused_name = fused_accessor_name(ctx.program, &[gate_w, up_w]);
        let weight_expr = ctx.weight_accessor(&fused_name);

        let mul_out = ctx.output_ident(mul_id, 0);
        let intermediate = ctx.bound("intermediate_size") as usize;

        quote! {
            let #mul_out = unsafe {
                let gate_up = (#weight_expr).forward(
                    #activation,
                    &mut device.caching,
                    device.compute_stream,
                );
                ::ferrite_kernels::kernels::gelu_and_mul_fused(
                    *gate_up,
                    #intermediate,
                    &mut device.caching,
                    device.compute_stream,
                )
            };
        }
    }
}

/// Marlin fused QKV + RoPE + cache-write (decode) — the AWQ
/// counterpart of `FusedQkvRopeCacheImpl`.
#[derive(Debug, Default)]
pub struct MarlinFusedQkvRopeCacheImpl;

impl Implementation for MarlinFusedQkvRopeCacheImpl {
    fn name(&self) -> &'static str {
        "marlin_fused_qkv_rope_cache"
    }

    fn target_compatible(&self, _profile: &TargetProfile) -> bool {
        true
    }

    fn workload_constraint(&self) -> WorkloadConstraint {
        WorkloadConstraint::NumTokensRange { min: 1, max: 1 }
    }

    fn matches(&self, fuf: &Fuf, seed: TileId, _profile: &TargetProfile) -> Option<MatchInfo> {
        // Mirror of `FusedQkvRopeCacheImpl.matches`; diffs are the
        // storage gate (Q/K/V must all be Marlin-consumable — AWQ
        // or GPTQ) on top of the BiasAdd-aware unwrap. Bias is
        // handled at the kernel boundary: `MarlinLinear::forward`
        // applies `bias_add_inplace` after `marlin_gemm` when
        // `self.bias.is_some()`, and the `load_{awq,gptq}_concat`
        // loaders pack per-prefix `.bias` tensors into the fused
        // LinearLayer automatically.
        let seed_node = fuf.get(seed);
        if !is_marlin_gemm(fuf, seed) {
            return None;
        }

        let rope_node = fuf.nodes.iter().find(|n| {
            if n.op != OpKind::RopeAppend || n.inputs.len() < 3 {
                return false;
            }
            let qkv_raw: Vec<TileId> = n
                .inputs
                .iter()
                .take(3)
                .filter_map(|i| match i {
                    FufInput::Tile { id, .. } => Some(*id),
                    _ => None,
                })
                .collect();
            if qkv_raw.len() != 3 {
                return false;
            }
            let resolved: Option<Vec<(TileId, Option<TileId>)>> = qkv_raw
                .iter()
                .map(|t| unwrap_gemm_through_bias(fuf, *t))
                .collect();
            let Some(resolved) = resolved else {
                return false;
            };
            // Uniform bias presence across Q/K/V — mixed is a DSL
            // inconsistency, reject rather than silently partial-fuse.
            let biased = resolved[0].1.is_some();
            if resolved.iter().any(|r| r.1.is_some() != biased) {
                return false;
            }
            let gemms: Vec<TileId> = resolved.iter().map(|r| r.0).collect();
            if !gemms.contains(&seed) {
                return false;
            }
            if gemms.iter().any(|t| !is_marlin_gemm(fuf, *t)) {
                return false;
            }
            let act = first_tile_input(fuf.get(gemms[0]));
            act.is_some() && gemms.iter().all(|t| first_tile_input(fuf.get(*t)) == act)
        })?;
        let rope_id = rope_node.id;

        let qkv_raw: Vec<TileId> = rope_node
            .inputs
            .iter()
            .take(3)
            .filter_map(|i| match i {
                FufInput::Tile { id, .. } => Some(*id),
                _ => None,
            })
            .collect();
        let resolved: Vec<(TileId, Option<TileId>)> = qkv_raw
            .iter()
            .map(|t| unwrap_gemm_through_bias(fuf, *t).expect("already validated in find"))
            .collect();

        let mut claimed: Vec<TileId> = Vec::with_capacity(7);
        for (g, b) in &resolved {
            claimed.push(*g);
            if let Some(b) = b {
                claimed.push(*b);
            }
        }
        claimed.push(rope_id);
        claimed.sort();

        let _ = seed_node;
        let activation = first_tile_input(fuf.get(resolved[0].0))?.0;
        Some(MatchInfo {
            claimed_tiles: claimed,
            boundary_inputs: vec![activation],
            boundary_outputs: vec![rope_id],
        })
    }

    fn cost_us(&self, m: &MatchInfo, ctx: &CostCtx) -> f64 {
        FusedQkvRopeCacheImpl.cost_us(m, ctx)
    }

    fn resources(&self, _m: &MatchInfo) -> Resources {
        Resources::ZERO
    }

    fn launch_kind(&self) -> LaunchKind {
        LaunchKind::HostCallback
    }

    fn supported_input_handoffs(&self) -> &[Handoff] {
        const H: &[Handoff] = &[Handoff::StreamOrder, Handoff::StreamEvent];
        H
    }

    fn supported_output_handoffs(&self) -> &[Handoff] {
        const H: &[Handoff] = &[Handoff::StreamOrder, Handoff::StreamEvent];
        H
    }

    fn input_layouts(&self, m: &MatchInfo) -> Vec<Layout> {
        vec![Layout::RowMajorBf16; m.boundary_inputs.len()]
    }

    fn output_layouts(&self, m: &MatchInfo) -> Vec<Layout> {
        vec![Layout::RowMajorBf16; m.boundary_outputs.len()]
    }

    fn is_compute_bound(&self) -> bool {
        true
    }

    #[allow(clippy::type_complexity)]
    fn output_alias(
        &self,
        claimed_tiles: &[TileId],
        fuf: &Fuf,
    ) -> Vec<((TileId, u8), Option<(TileId, u8)>)> {
        // Same alias declaration as the dense variant: rotated Q is
        // the only OwnedTensor; K/V are paged-cache views.
        let rope_id = *claimed_tiles
            .iter()
            .find(|t| fuf.get(**t).op == OpKind::RopeAppend)
            .expect("claim contains RopeAppend");
        vec![((rope_id, 0), None)]
    }

    fn required_weights(
        &self,
        claimed_tiles: &[TileId],
        fuf: &Fuf,
        program: &Program,
    ) -> Vec<WeightAccessor> {
        let sources: Vec<(WeightId, Option<u64>)> = claimed_tiles
            .iter()
            .filter_map(|t| {
                let n = fuf.get(*t);
                if n.op == OpKind::Gemm {
                    first_weight_ref(n)
                } else {
                    None
                }
            })
            .collect();
        let name = fused_accessor_name(program, &sources);
        vec![WeightAccessor {
            name,
            rust_type: quote! { ::ferrite_kernels::layers::MarlinLinear },
            source_weights: sources,
        }]
    }

    fn emit_call(&self, ctx: &EmitCtx) -> TokenStream {
        // Mirror of `FusedQkvRopeCacheImpl::emit_call` with the
        // cuBLAS-shaped `(#w).forward(#x, &mut device.cublas, ...)`
        // replaced by `(#w).forward(#x, &mut device.caching, stream)`
        // (MarlinLinear owns the matmul internally). The FP8 KV
        // branch stays — it's orthogonal to the weight format.
        //
        // BiasAdd-aware: rope's three tile inputs are unwrapped
        // through an optional BiasAdd wrapper to find the underlying
        // Gemms. Bias is still applied — `MarlinLinear::forward`
        // does `bias_add_inplace` after `marlin_gemm` when its
        // packed bias is present (auto-detected by
        // `MarlinLinear::load_awq_concat`).
        let rope_id = *ctx
            .claimed_tiles
            .iter()
            .find(|t| ctx.fuf.get(**t).op == OpKind::RopeAppend)
            .expect("claim must contain a RopeAppend");
        let rope_node = ctx.fuf.get(rope_id);

        let qkv_raw: Vec<TileId> = rope_node
            .inputs
            .iter()
            .take(3)
            .filter_map(|i| match i {
                FufInput::Tile { id, .. } => Some(*id),
                _ => None,
            })
            .collect();
        assert_eq!(qkv_raw.len(), 3, "rope has three tile inputs (q, k, v)");
        let resolved: Vec<(TileId, Option<TileId>)> = qkv_raw
            .iter()
            .map(|t| {
                unwrap_gemm_through_bias(ctx.fuf, *t)
                    .expect("claim-time check guarantees Gemm-or-BiasAdd(Gemm)")
            })
            .collect();
        let q_gemm_id = resolved[0].0;
        let activation = ctx.input_expr(q_gemm_id, 0);

        let qkv_weights: Vec<(WeightId, Option<u64>)> = resolved
            .iter()
            .map(|(g, _)| first_weight_ref(ctx.fuf.get(*g)).expect("gemm has a weight"))
            .collect();
        let fused_name = fused_accessor_name(ctx.program, &qkv_weights);
        let weight_expr = ctx.weight_accessor(&fused_name);

        let num_q_heads = ctx.bound("num_attention_heads") as usize;
        let num_kv_heads = ctx.bound("num_key_value_heads") as usize;
        let head_dim = ctx.bound("head_dim") as usize;
        let q_size = num_q_heads * head_dim;
        let kv_size = num_kv_heads * head_dim;

        let layer = rope_kv_cache_layer(rope_node)
            .expect("RopeAppend has a KvCache extern input with a concrete layer index")
            as usize;

        let q_out = ctx.output_ident(rope_id, 0);
        let k_out = ctx.output_ident(rope_id, 1);
        let v_out = ctx.output_ident(rope_id, 2);

        let rotary_cos_sin = rotary_cos_sin_tokens(ctx.fuf, ctx.claimed_tiles);

        quote! {
            let #q_out = unsafe {
                let qkv_packed = (#weight_expr).forward(
                    #activation,
                    &mut device.caching,
                    device.compute_stream,
                );
                if ctx.kv_cache.is_fp8() {
                    ::ferrite_kernels::kernels::fused_qkv_rope_cache_fp8(
                        *qkv_packed,
                        *ctx.positions,
                        #rotary_cos_sin,
                        *ctx.slot_mapping,
                        *ctx.kv_cache.k_cache(#layer),
                        *ctx.kv_cache.v_cache(#layer),
                        ctx.kv_cache.k_scale_ptr(#layer),
                        ctx.kv_cache.v_scale_ptr(#layer),
                        #q_size,
                        #kv_size,
                        #num_q_heads,
                        #head_dim,
                        &mut device.caching,
                        device.compute_stream,
                    )
                } else {
                    ::ferrite_kernels::kernels::fused_qkv_rope_cache(
                        *qkv_packed,
                        *ctx.positions,
                        #rotary_cos_sin,
                        *ctx.slot_mapping,
                        *ctx.kv_cache.k_cache(#layer),
                        *ctx.kv_cache.v_cache(#layer),
                        #q_size,
                        #kv_size,
                        #num_q_heads,
                        #head_dim,
                        &mut device.caching,
                        device.compute_stream,
                    )
                }
            };
            let #k_out = ctx.kv_cache.k_cache(#layer);
            let #v_out = ctx.kv_cache.v_cache(#layer);
        }
    }
}

/// Marlin fused QKV + RoPE (prefill, contiguous K/V) — the AWQ
/// counterpart of `FusedQkvRopePrefillImpl`.
#[derive(Debug, Default)]
pub struct MarlinFusedQkvRopePrefillImpl;

impl Implementation for MarlinFusedQkvRopePrefillImpl {
    fn name(&self) -> &'static str {
        "marlin_fused_qkv_rope_prefill"
    }

    fn target_compatible(&self, _profile: &TargetProfile) -> bool {
        true
    }

    fn workload_constraint(&self) -> WorkloadConstraint {
        WorkloadConstraint::NumTokensRange {
            min: 2,
            max: u32::MAX,
        }
    }

    fn matches(&self, fuf: &Fuf, seed: TileId, profile: &TargetProfile) -> Option<MatchInfo> {
        // Same pattern as the cache variant — delegate so the
        // matcher lives in one place.
        MarlinFusedQkvRopeCacheImpl.matches(fuf, seed, profile)
    }

    fn cost_us(&self, m: &MatchInfo, ctx: &CostCtx) -> f64 {
        MarlinFusedQkvRopeCacheImpl.cost_us(m, ctx)
    }

    fn resources(&self, _m: &MatchInfo) -> Resources {
        Resources::ZERO
    }

    fn launch_kind(&self) -> LaunchKind {
        LaunchKind::HostCallback
    }

    fn supported_input_handoffs(&self) -> &[Handoff] {
        const H: &[Handoff] = &[Handoff::StreamOrder, Handoff::StreamEvent];
        H
    }

    fn supported_output_handoffs(&self) -> &[Handoff] {
        const H: &[Handoff] = &[Handoff::StreamOrder, Handoff::StreamEvent];
        H
    }

    fn input_layouts(&self, m: &MatchInfo) -> Vec<Layout> {
        vec![Layout::RowMajorBf16; m.boundary_inputs.len()]
    }

    fn output_layouts(&self, m: &MatchInfo) -> Vec<Layout> {
        vec![Layout::RowMajorBf16; m.boundary_outputs.len()]
    }

    fn is_compute_bound(&self) -> bool {
        true
    }

    fn required_weights(
        &self,
        claimed_tiles: &[TileId],
        fuf: &Fuf,
        program: &Program,
    ) -> Vec<WeightAccessor> {
        MarlinFusedQkvRopeCacheImpl.required_weights(claimed_tiles, fuf, program)
    }

    fn emit_call(&self, ctx: &EmitCtx) -> TokenStream {
        // Mirror of `FusedQkvRopePrefillImpl::emit_call` with
        // `MarlinLinear::forward(x, alloc, stream)` replacing the
        // cuBLAS call. BiasAdd-aware: rope's three tile inputs are
        // unwrapped through an optional BiasAdd wrapper so the
        // packed `MarlinLinear` (carrying the concat of the three
        // source .bias tensors) is still the authoritative bias
        // applicator — `MarlinLinear::forward` runs
        // `bias_add_inplace` after `marlin_gemm`.
        let rope_id = *ctx
            .claimed_tiles
            .iter()
            .find(|t| ctx.fuf.get(**t).op == OpKind::RopeAppend)
            .expect("claim must contain a RopeAppend");
        let rope_node = ctx.fuf.get(rope_id);

        let qkv_raw: Vec<TileId> = rope_node
            .inputs
            .iter()
            .take(3)
            .filter_map(|i| match i {
                FufInput::Tile { id, .. } => Some(*id),
                _ => None,
            })
            .collect();
        assert_eq!(qkv_raw.len(), 3, "rope has three tile inputs (q, k, v)");
        let resolved: Vec<(TileId, Option<TileId>)> = qkv_raw
            .iter()
            .map(|t| {
                unwrap_gemm_through_bias(ctx.fuf, *t)
                    .expect("claim-time check guarantees Gemm-or-BiasAdd(Gemm)")
            })
            .collect();
        let q_gemm_id = resolved[0].0;
        let activation = ctx.input_expr(q_gemm_id, 0);

        let qkv_weights: Vec<(WeightId, Option<u64>)> = resolved
            .iter()
            .map(|(g, _)| first_weight_ref(ctx.fuf.get(*g)).expect("gemm has a weight"))
            .collect();
        let fused_name = fused_accessor_name(ctx.program, &qkv_weights);
        let weight_expr = ctx.weight_accessor(&fused_name);

        let num_q_heads = ctx.bound("num_attention_heads") as usize;
        let num_kv_heads = ctx.bound("num_key_value_heads") as usize;
        let head_dim = ctx.bound("head_dim") as usize;
        let q_size = num_q_heads * head_dim;
        let kv_size = num_kv_heads * head_dim;

        let layer = rope_kv_cache_layer(rope_node)
            .expect("RopeAppend has a KvCache extern input with a concrete layer index")
            as usize;

        let q_out = ctx.output_ident(rope_id, 0);
        let k_out = ctx.output_ident(rope_id, 1);
        let v_out = ctx.output_ident(rope_id, 2);

        let rotary_cos_sin = rotary_cos_sin_tokens(ctx.fuf, ctx.claimed_tiles);

        quote! {
            let (#q_out, #k_out, #v_out) = unsafe {
                let qkv_packed = (#weight_expr).forward(
                    #activation,
                    &mut device.caching,
                    device.compute_stream,
                );
                ::ferrite_kernels::kernels::fused_qkv_rope(
                    *qkv_packed,
                    *ctx.positions,
                    #rotary_cos_sin,
                    #q_size,
                    #kv_size,
                    #num_q_heads,
                    #num_kv_heads,
                    #head_dim,
                    &mut device.caching,
                    device.compute_stream,
                )
            };
            unsafe {
                ::ferrite_kernels::attention_helpers::write_kv_cache(
                    (#k_out).view(),
                    (#v_out).view(),
                    ctx.slot_mapping,
                    ctx.kv_cache,
                    #layer,
                    device.compute_stream,
                );
            }
        }
    }
}

// ── BitsAndBytes 4-bit (NF4 / FP4) impl family ────────────────────
//
// Mirror of the Marlin* family but for `StorageFormat::Bnb4`. The
// structural matchers are byte-identical to the Marlin variants
// (same `(Gemm, Gemm, Silu, Mul)` / `(Gemm, Gemm, Gelu, Mul)` /
// `(Gemm, Gemm, Gemm, RopeAppend)` claim patterns); the only diffs
// are (a) the storage gate (`is_bnb4_gemm` instead of
// `is_marlin_gemm`), (b) the declared accessor `rust_type`
// (`Bnb4bitLinear`), and (c) the `emit_call` binding
// (`(#w).forward(x, &mut device.cublas, &mut device.caching,
// stream)` — BNB4's forward takes cuBLAS because the kernel does
// a dequant-then-cuBLAS-matmul, unlike Marlin's fused-matmul path).
//
// The kernel-side loader (`Bnb4bitLinear::load{,_concat}`) handles
// byte-concat of packed nibbles + absmax across fused shards, so a
// fused QKV / gate-up lands as a single `Bnb4bitLinear` just like
// the Marlin fused accessors do.

/// Singleton Bnb4 GEMM — BNB4 counterpart of `MarlinGemmImpl`.
#[derive(Debug, Default)]
pub struct Bnb4GemmImpl;

impl Implementation for Bnb4GemmImpl {
    fn name(&self) -> &'static str {
        "bnb4_gemm"
    }

    fn target_compatible(&self, _profile: &TargetProfile) -> bool {
        true
    }

    fn matches(&self, fuf: &Fuf, seed: TileId, _profile: &TargetProfile) -> Option<MatchInfo> {
        let info = single_tile_match(fuf, seed, OpKind::Gemm)?;
        if !is_bnb4_gemm(fuf, seed) {
            return None;
        }
        // Unlike Marlin + dense paths, BNB4 has no cutlass-shaped
        // singleton impl that would compete with the fused variants
        // at the same seed — so we skip `gemm_is_fusion_partner`
        // deference. If the fused BNB4 Impl (QKV-rope, gate-up-silu,
        // gate-up-gelu) matches, the DP picks it over this singleton
        // by claim-size. If it doesn't (e.g. Qwen3's per-head QK-norm
        // breaks the `(Gemm, Gemm, Gemm, RopeAppend)` adjacency for
        // q/k but not v), this singleton claims the leftover Gemms
        // rather than leaving them `UnclaimedTile`.
        Some(info)
    }

    fn cost_us(&self, m: &MatchInfo, ctx: &CostCtx) -> f64 {
        cost_gemm(m, ctx)
    }

    fn resources(&self, _m: &MatchInfo) -> Resources {
        Resources::ZERO
    }

    fn launch_kind(&self) -> LaunchKind {
        LaunchKind::HostCallback
    }

    fn supported_input_handoffs(&self) -> &[Handoff] {
        const H: &[Handoff] = &[Handoff::StreamOrder, Handoff::StreamEvent];
        H
    }

    fn supported_output_handoffs(&self) -> &[Handoff] {
        const H: &[Handoff] = &[Handoff::StreamOrder, Handoff::StreamEvent];
        H
    }

    fn input_layouts(&self, m: &MatchInfo) -> Vec<Layout> {
        vec![Layout::RowMajorBf16; m.boundary_inputs.len()]
    }

    fn output_layouts(&self, m: &MatchInfo) -> Vec<Layout> {
        vec![Layout::RowMajorBf16; m.boundary_outputs.len()]
    }

    fn is_compute_bound(&self) -> bool {
        true
    }

    fn required_weights(
        &self,
        claimed_tiles: &[TileId],
        fuf: &Fuf,
        program: &Program,
    ) -> Vec<WeightAccessor> {
        let tile = claimed_tiles[0];
        let (wid, index) = first_weight_ref(fuf.get(tile)).expect("Gemm has a weight input");
        let name = weight_field_name(program, wid, index);
        vec![WeightAccessor {
            name,
            rust_type: quote! { ::ferrite_kernels::layers::Bnb4bitLinear },
            source_weights: vec![(wid, index)],
        }]
    }

    fn emit_call(&self, ctx: &EmitCtx) -> TokenStream {
        let tile = ctx.primary();
        let out = ctx.output_ident(tile, 0);
        let x = ctx.input_expr(tile, 0);
        let w = ctx.input_expr(tile, 1);
        quote! {
            let #out = unsafe {
                (#w).forward(
                    #x,
                    &mut device.cublas,
                    &mut device.caching,
                    device.compute_stream,
                )
            };
        }
    }
}

// ── FP8 (E4M3) impl family ────────────────────────────────────────
//
// Singleton-only today. Covers per-tensor dynamic, per-tensor static,
// and per-channel FP8 via the same `Fp8Linear::forward` entry point;
// the loader sniffs scale layout at load time. Fused QKV / gate-up
// peers (`Fp8FusedQkvRope*`, `Fp8FusedGateUpSiluMulImpl`) are perf
// follow-ups — the singleton still produces correct results by
// emitting three separate FP8 GEMMs where the fused dense path
// would emit one.

/// Singleton FP8 GEMM — FP8 counterpart of `Bnb4GemmImpl`.
#[derive(Debug, Default)]
pub struct Fp8GemmImpl;

impl Implementation for Fp8GemmImpl {
    fn name(&self) -> &'static str {
        "fp8_gemm"
    }

    fn target_compatible(&self, _profile: &TargetProfile) -> bool {
        true
    }

    fn matches(&self, fuf: &Fuf, seed: TileId, _profile: &TargetProfile) -> Option<MatchInfo> {
        let info = single_tile_match(fuf, seed, OpKind::Gemm)?;
        if !is_fp8_gemm(fuf, seed) {
            return None;
        }
        // Defer to `Fp8FusedGemmBiasImpl` when the output feeds a
        // `BiasAdd` — same pattern as dense `GemmRefImpl` deferring
        // to `FusedGemmBiasImpl`. Without this, the singleton claims
        // the Gemm and leaves the downstream BiasAdd tile unclaimed.
        if output_feeds_op(fuf, seed, OpKind::BiasAdd) {
            return None;
        }
        Some(info)
    }

    fn cost_us(&self, m: &MatchInfo, ctx: &CostCtx) -> f64 {
        cost_gemm(m, ctx)
    }

    fn resources(&self, _m: &MatchInfo) -> Resources {
        Resources::ZERO
    }

    fn launch_kind(&self) -> LaunchKind {
        LaunchKind::HostCallback
    }

    fn supported_input_handoffs(&self) -> &[Handoff] {
        const H: &[Handoff] = &[Handoff::StreamOrder, Handoff::StreamEvent];
        H
    }

    fn supported_output_handoffs(&self) -> &[Handoff] {
        const H: &[Handoff] = &[Handoff::StreamOrder, Handoff::StreamEvent];
        H
    }

    fn input_layouts(&self, m: &MatchInfo) -> Vec<Layout> {
        vec![Layout::RowMajorBf16; m.boundary_inputs.len()]
    }

    fn output_layouts(&self, m: &MatchInfo) -> Vec<Layout> {
        vec![Layout::RowMajorBf16; m.boundary_outputs.len()]
    }

    fn is_compute_bound(&self) -> bool {
        true
    }

    fn required_weights(
        &self,
        claimed_tiles: &[TileId],
        fuf: &Fuf,
        program: &Program,
    ) -> Vec<WeightAccessor> {
        let tile = claimed_tiles[0];
        let (wid, index) = first_weight_ref(fuf.get(tile)).expect("Gemm has a weight input");
        let name = weight_field_name(program, wid, index);
        let rust_type = fp8_accessor_type_for(fuf, tile);
        vec![WeightAccessor {
            name,
            rust_type,
            source_weights: vec![(wid, index)],
        }]
    }

    fn emit_call(&self, ctx: &EmitCtx) -> TokenStream {
        let tile = ctx.primary();
        let out = ctx.output_ident(tile, 0);
        let x = ctx.input_expr(tile, 0);
        let w = ctx.input_expr(tile, 1);
        quote! {
            let #out = unsafe {
                (#w).forward(
                    #x,
                    &mut device.cublas,
                    &mut device.caching,
                    device.compute_stream,
                )
            };
        }
    }
}

/// Fused `(Gemm, BiasAdd)` for FP8. FP8 counterpart of
/// `FusedGemmBiasImpl`. `Fp8Linear` stores the loaded `.bias` on the
/// struct and applies it inside `forward` (cutlass
/// `cutlass_scaled_mm_with_bias` epilog), so `emit_call` is the
/// same as `Fp8GemmImpl`'s — the BiasAdd tile is absorbed into the
/// accessor rather than a separate kernel launch.
#[derive(Debug, Default)]
pub struct Fp8FusedGemmBiasImpl;

impl Implementation for Fp8FusedGemmBiasImpl {
    fn name(&self) -> &'static str {
        "fp8_fused_gemm_bias"
    }

    fn target_compatible(&self, _profile: &TargetProfile) -> bool {
        true
    }

    fn matches(&self, fuf: &Fuf, seed: TileId, _profile: &TargetProfile) -> Option<MatchInfo> {
        let seed_node = fuf.get(seed);
        if seed_node.op != OpKind::Gemm {
            return None;
        }
        if !matches!(
            weight_storage_of(seed_node),
            Some(StorageFormat::Fp8 { .. })
        ) {
            return None;
        }
        let bias_node = fuf.nodes.iter().find(|n| {
            n.op == OpKind::BiasAdd
                && matches!(
                    n.inputs.first(),
                    Some(FufInput::Tile { id, .. }) if *id == seed
                )
        })?;
        let bias_id = bias_node.id;

        let mut claimed = vec![seed, bias_id];
        claimed.sort();

        let activation_inputs: Vec<TileId> = seed_node
            .inputs
            .iter()
            .filter_map(|i| match i {
                FufInput::Tile { id, .. } => Some(*id),
                _ => None,
            })
            .collect();

        Some(MatchInfo {
            claimed_tiles: claimed,
            boundary_inputs: activation_inputs,
            boundary_outputs: vec![bias_id],
        })
    }

    fn cost_us(&self, m: &MatchInfo, ctx: &CostCtx) -> f64 {
        let gemm_id = *m
            .claimed_tiles
            .iter()
            .find(|t| ctx.fuf.get(**t).op == OpKind::Gemm)
            .expect("claim contains Gemm");
        let gemm_info = MatchInfo {
            claimed_tiles: vec![gemm_id],
            boundary_inputs: vec![],
            boundary_outputs: vec![gemm_id],
        };
        cost_gemm(&gemm_info, ctx)
    }

    fn resources(&self, _m: &MatchInfo) -> Resources {
        Resources::ZERO
    }

    fn launch_kind(&self) -> LaunchKind {
        LaunchKind::HostCallback
    }

    fn supported_input_handoffs(&self) -> &[Handoff] {
        const H: &[Handoff] = &[Handoff::StreamOrder, Handoff::StreamEvent];
        H
    }

    fn supported_output_handoffs(&self) -> &[Handoff] {
        const H: &[Handoff] = &[Handoff::StreamOrder, Handoff::StreamEvent];
        H
    }

    fn input_layouts(&self, m: &MatchInfo) -> Vec<Layout> {
        vec![Layout::RowMajorBf16; m.boundary_inputs.len()]
    }

    fn output_layouts(&self, m: &MatchInfo) -> Vec<Layout> {
        vec![Layout::RowMajorBf16; m.boundary_outputs.len()]
    }

    fn is_compute_bound(&self) -> bool {
        true
    }

    fn required_weights(
        &self,
        claimed_tiles: &[TileId],
        fuf: &Fuf,
        program: &Program,
    ) -> Vec<WeightAccessor> {
        let gemm_id = *claimed_tiles
            .iter()
            .find(|t| fuf.get(**t).op == OpKind::Gemm)
            .expect("claim contains Gemm");
        let (wid, index) = first_weight_ref(fuf.get(gemm_id)).expect("Gemm has a weight input");
        let name = weight_field_name(program, wid, index);
        let rust_type = fp8_accessor_type_for(fuf, gemm_id);
        vec![WeightAccessor {
            name,
            rust_type,
            source_weights: vec![(wid, index)],
        }]
    }

    fn emit_call(&self, ctx: &EmitCtx) -> TokenStream {
        let gemm_id = *ctx
            .claimed_tiles
            .iter()
            .find(|t| ctx.fuf.get(**t).op == OpKind::Gemm)
            .expect("claim contains Gemm");
        let bias_id = *ctx
            .claimed_tiles
            .iter()
            .find(|t| ctx.fuf.get(**t).op == OpKind::BiasAdd)
            .expect("claim contains BiasAdd");
        let activation = ctx.input_expr(gemm_id, 0);
        let (wid, index) = first_weight_ref(ctx.fuf.get(gemm_id)).expect("gemm weight ref");
        let name = weight_field_name(ctx.program, wid, index);
        let weight_expr = ctx.weight_accessor(&name);
        let out = ctx.output_ident(bias_id, 0);
        quote! {
            let #out = unsafe {
                (#weight_expr).forward(
                    #activation,
                    &mut device.cublas,
                    &mut device.caching,
                    device.compute_stream,
                )
            };
        }
    }
}

/// Fused gate/up + SwiGLU for FP8. `Fp8Linear::load_concat`
/// max-scale merges gate + up into one fused FP8 matmul, so this
/// emits one `Fp8Linear::forward` on the fused weight followed by
/// `silu_and_mul_fused`. FP8 counterpart of `Bnb4FusedGateUpSiluMulImpl`.
#[derive(Debug, Default)]
pub struct Fp8FusedGateUpSiluMulImpl;

impl Implementation for Fp8FusedGateUpSiluMulImpl {
    fn name(&self) -> &'static str {
        "fp8_fused_gate_up_silu_mul"
    }

    fn target_compatible(&self, _profile: &TargetProfile) -> bool {
        true
    }

    fn matches(&self, fuf: &Fuf, seed: TileId, _profile: &TargetProfile) -> Option<MatchInfo> {
        let gate_gemm = fuf.get(seed);
        if !is_fp8_gemm(fuf, seed) {
            return None;
        }
        let silu_node = fuf
            .nodes
            .iter()
            .find(|n| n.op == OpKind::Silu && consumes_tile(n, seed))?;
        let silu_id = silu_node.id;
        let mul_node = fuf
            .nodes
            .iter()
            .find(|n| n.op == OpKind::Mul && consumes_tile(n, silu_id))?;
        let mul_id = mul_node.id;
        let up_gemm_id = mul_node.inputs.iter().find_map(|i| match i {
            FufInput::Tile { id, .. } if *id != silu_id => Some(*id),
            _ => None,
        })?;
        if !is_fp8_gemm(fuf, up_gemm_id) {
            return None;
        }
        if first_tile_input(gate_gemm)? != first_tile_input(fuf.get(up_gemm_id))? {
            return None;
        }
        let mut claimed = [seed, up_gemm_id, silu_id, mul_id];
        claimed.sort();
        let claimed = claimed.to_vec();
        let activation_tile = first_tile_input(gate_gemm)?.0;
        Some(MatchInfo {
            claimed_tiles: claimed,
            boundary_inputs: vec![activation_tile],
            boundary_outputs: vec![mul_id],
        })
    }

    fn cost_us(&self, m: &MatchInfo, ctx: &CostCtx) -> f64 {
        FusedGateUpSiluMulImpl.cost_us(m, ctx)
    }

    fn resources(&self, _m: &MatchInfo) -> Resources {
        Resources::ZERO
    }

    fn launch_kind(&self) -> LaunchKind {
        LaunchKind::HostCallback
    }

    fn supported_input_handoffs(&self) -> &[Handoff] {
        const H: &[Handoff] = &[Handoff::StreamOrder, Handoff::StreamEvent];
        H
    }

    fn supported_output_handoffs(&self) -> &[Handoff] {
        const H: &[Handoff] = &[Handoff::StreamOrder, Handoff::StreamEvent];
        H
    }

    fn input_layouts(&self, m: &MatchInfo) -> Vec<Layout> {
        vec![Layout::RowMajorBf16; m.boundary_inputs.len()]
    }

    fn output_layouts(&self, m: &MatchInfo) -> Vec<Layout> {
        vec![Layout::RowMajorBf16; m.boundary_outputs.len()]
    }

    fn is_compute_bound(&self) -> bool {
        true
    }

    fn required_weights(
        &self,
        claimed_tiles: &[TileId],
        fuf: &Fuf,
        program: &Program,
    ) -> Vec<WeightAccessor> {
        let sources: Vec<(WeightId, Option<u64>)> = claimed_tiles
            .iter()
            .filter_map(|t| {
                let n = fuf.get(*t);
                if n.op == OpKind::Gemm {
                    first_weight_ref(n)
                } else {
                    None
                }
            })
            .collect();
        let name = fused_accessor_name(program, &sources);
        let gemm_tile = *claimed_tiles
            .iter()
            .find(|t| fuf.get(**t).op == OpKind::Gemm)
            .expect("claim contains at least one Gemm");
        let rust_type = fp8_accessor_type_for(fuf, gemm_tile);
        vec![WeightAccessor {
            name,
            rust_type,
            source_weights: sources,
        }]
    }

    fn emit_call(&self, ctx: &EmitCtx) -> TokenStream {
        let silu_id = *ctx
            .claimed_tiles
            .iter()
            .find(|t| ctx.fuf.get(**t).op == OpKind::Silu)
            .expect("fused gate/up/silu/mul claim must contain Silu");
        let mul_id = *ctx
            .claimed_tiles
            .iter()
            .find(|t| ctx.fuf.get(**t).op == OpKind::Mul)
            .expect("fused gate/up/silu/mul claim must contain Mul");
        let (gate_id, _) =
            first_tile_input(ctx.fuf.get(silu_id)).expect("silu has a tile input — the gate gemm");
        let up_id = ctx
            .claimed_tiles
            .iter()
            .copied()
            .find(|t| ctx.fuf.get(*t).op == OpKind::Gemm && *t != gate_id)
            .expect("claim contains a second Gemm — the up gemm");

        let activation = ctx.input_expr(gate_id, 0);

        let gate_w = first_weight_ref(ctx.fuf.get(gate_id)).expect("gate gemm has a weight");
        let up_w = first_weight_ref(ctx.fuf.get(up_id)).expect("up gemm has a weight");
        let fused_name = fused_accessor_name(ctx.program, &[gate_w, up_w]);
        let weight_expr = ctx.weight_accessor(&fused_name);

        let mul_out = ctx.output_ident(mul_id, 0);
        let intermediate = ctx.bound("intermediate_size") as usize;

        quote! {
            let #mul_out = unsafe {
                let gate_up = (#weight_expr).forward(
                    #activation,
                    &mut device.cublas,
                    &mut device.caching,
                    device.compute_stream,
                );
                ::ferrite_kernels::kernels::silu_and_mul_fused(
                    *gate_up,
                    #intermediate,
                    &mut device.caching,
                    device.compute_stream,
                )
            };
        }
    }
}

/// Fused gate/up + SwiGLU for BNB4. BNB4 byte-concats nibbles +
/// absmax across gate + up on load (`load_concat`), so this emits
/// one `Bnb4bitLinear::forward` on the fused weight followed by
/// `silu_and_mul_fused`.
#[derive(Debug, Default)]
pub struct Bnb4FusedGateUpSiluMulImpl;

impl Implementation for Bnb4FusedGateUpSiluMulImpl {
    fn name(&self) -> &'static str {
        "bnb4_fused_gate_up_silu_mul"
    }

    fn target_compatible(&self, _profile: &TargetProfile) -> bool {
        true
    }

    fn matches(&self, fuf: &Fuf, seed: TileId, _profile: &TargetProfile) -> Option<MatchInfo> {
        let gate_gemm = fuf.get(seed);
        if !is_bnb4_gemm(fuf, seed) {
            return None;
        }
        let silu_node = fuf
            .nodes
            .iter()
            .find(|n| n.op == OpKind::Silu && consumes_tile(n, seed))?;
        let silu_id = silu_node.id;
        let mul_node = fuf
            .nodes
            .iter()
            .find(|n| n.op == OpKind::Mul && consumes_tile(n, silu_id))?;
        let mul_id = mul_node.id;
        let up_gemm_id = mul_node.inputs.iter().find_map(|i| match i {
            FufInput::Tile { id, .. } if *id != silu_id => Some(*id),
            _ => None,
        })?;
        if !is_bnb4_gemm(fuf, up_gemm_id) {
            return None;
        }
        if first_tile_input(gate_gemm)? != first_tile_input(fuf.get(up_gemm_id))? {
            return None;
        }
        let mut claimed = [seed, up_gemm_id, silu_id, mul_id];
        claimed.sort();
        let claimed = claimed.to_vec();
        let activation_tile = first_tile_input(gate_gemm)?.0;
        Some(MatchInfo {
            claimed_tiles: claimed,
            boundary_inputs: vec![activation_tile],
            boundary_outputs: vec![mul_id],
        })
    }

    fn cost_us(&self, m: &MatchInfo, ctx: &CostCtx) -> f64 {
        FusedGateUpSiluMulImpl.cost_us(m, ctx)
    }

    fn resources(&self, _m: &MatchInfo) -> Resources {
        Resources::ZERO
    }

    fn launch_kind(&self) -> LaunchKind {
        LaunchKind::HostCallback
    }

    fn supported_input_handoffs(&self) -> &[Handoff] {
        const H: &[Handoff] = &[Handoff::StreamOrder, Handoff::StreamEvent];
        H
    }

    fn supported_output_handoffs(&self) -> &[Handoff] {
        const H: &[Handoff] = &[Handoff::StreamOrder, Handoff::StreamEvent];
        H
    }

    fn input_layouts(&self, m: &MatchInfo) -> Vec<Layout> {
        vec![Layout::RowMajorBf16; m.boundary_inputs.len()]
    }

    fn output_layouts(&self, m: &MatchInfo) -> Vec<Layout> {
        vec![Layout::RowMajorBf16; m.boundary_outputs.len()]
    }

    fn is_compute_bound(&self) -> bool {
        true
    }

    fn required_weights(
        &self,
        claimed_tiles: &[TileId],
        fuf: &Fuf,
        program: &Program,
    ) -> Vec<WeightAccessor> {
        let sources: Vec<(WeightId, Option<u64>)> = claimed_tiles
            .iter()
            .filter_map(|t| {
                let n = fuf.get(*t);
                if n.op == OpKind::Gemm {
                    first_weight_ref(n)
                } else {
                    None
                }
            })
            .collect();
        let name = fused_accessor_name(program, &sources);
        vec![WeightAccessor {
            name,
            rust_type: quote! { ::ferrite_kernels::layers::Bnb4bitLinear },
            source_weights: sources,
        }]
    }

    fn emit_call(&self, ctx: &EmitCtx) -> TokenStream {
        let silu_id = *ctx
            .claimed_tiles
            .iter()
            .find(|t| ctx.fuf.get(**t).op == OpKind::Silu)
            .expect("fused gate/up/silu/mul claim must contain Silu");
        let mul_id = *ctx
            .claimed_tiles
            .iter()
            .find(|t| ctx.fuf.get(**t).op == OpKind::Mul)
            .expect("fused gate/up/silu/mul claim must contain Mul");
        let (gate_id, _) =
            first_tile_input(ctx.fuf.get(silu_id)).expect("silu has a tile input — the gate gemm");
        let up_id = ctx
            .claimed_tiles
            .iter()
            .copied()
            .find(|t| ctx.fuf.get(*t).op == OpKind::Gemm && *t != gate_id)
            .expect("claim contains a second Gemm — the up gemm");

        let activation = ctx.input_expr(gate_id, 0);

        let gate_w = first_weight_ref(ctx.fuf.get(gate_id)).expect("gate gemm has a weight");
        let up_w = first_weight_ref(ctx.fuf.get(up_id)).expect("up gemm has a weight");
        let fused_name = fused_accessor_name(ctx.program, &[gate_w, up_w]);
        let weight_expr = ctx.weight_accessor(&fused_name);

        let mul_out = ctx.output_ident(mul_id, 0);
        let intermediate = ctx.bound("intermediate_size") as usize;

        quote! {
            let #mul_out = unsafe {
                let gate_up = (#weight_expr).forward(
                    #activation,
                    &mut device.cublas,
                    &mut device.caching,
                    device.compute_stream,
                );
                ::ferrite_kernels::kernels::silu_and_mul_fused(
                    *gate_up,
                    #intermediate,
                    &mut device.caching,
                    device.compute_stream,
                )
            };
        }
    }
}

/// Fused gate/up + GELU for BNB4 — Gemma2-style MLP.
#[derive(Debug, Default)]
pub struct Bnb4FusedGateUpGeluMulImpl;

impl Implementation for Bnb4FusedGateUpGeluMulImpl {
    fn name(&self) -> &'static str {
        "bnb4_fused_gate_up_gelu_mul"
    }

    fn target_compatible(&self, _profile: &TargetProfile) -> bool {
        true
    }

    fn matches(&self, fuf: &Fuf, seed: TileId, _profile: &TargetProfile) -> Option<MatchInfo> {
        let gate_gemm = fuf.get(seed);
        if !is_bnb4_gemm(fuf, seed) {
            return None;
        }
        let gelu_node = fuf
            .nodes
            .iter()
            .find(|n| n.op == OpKind::Gelu && consumes_tile(n, seed))?;
        let gelu_id = gelu_node.id;
        let mul_node = fuf
            .nodes
            .iter()
            .find(|n| n.op == OpKind::Mul && consumes_tile(n, gelu_id))?;
        let mul_id = mul_node.id;
        let up_gemm_id = mul_node.inputs.iter().find_map(|i| match i {
            FufInput::Tile { id, .. } if *id != gelu_id => Some(*id),
            _ => None,
        })?;
        if !is_bnb4_gemm(fuf, up_gemm_id) {
            return None;
        }
        if first_tile_input(gate_gemm)? != first_tile_input(fuf.get(up_gemm_id))? {
            return None;
        }
        let mut claimed = [seed, up_gemm_id, gelu_id, mul_id];
        claimed.sort();
        let claimed = claimed.to_vec();
        let activation_tile = first_tile_input(gate_gemm)?.0;
        Some(MatchInfo {
            claimed_tiles: claimed,
            boundary_inputs: vec![activation_tile],
            boundary_outputs: vec![mul_id],
        })
    }

    fn cost_us(&self, m: &MatchInfo, ctx: &CostCtx) -> f64 {
        FusedGateUpGeluMulImpl.cost_us(m, ctx)
    }

    fn resources(&self, _m: &MatchInfo) -> Resources {
        Resources::ZERO
    }

    fn launch_kind(&self) -> LaunchKind {
        LaunchKind::HostCallback
    }

    fn supported_input_handoffs(&self) -> &[Handoff] {
        const H: &[Handoff] = &[Handoff::StreamOrder, Handoff::StreamEvent];
        H
    }

    fn supported_output_handoffs(&self) -> &[Handoff] {
        const H: &[Handoff] = &[Handoff::StreamOrder, Handoff::StreamEvent];
        H
    }

    fn input_layouts(&self, m: &MatchInfo) -> Vec<Layout> {
        vec![Layout::RowMajorBf16; m.boundary_inputs.len()]
    }

    fn output_layouts(&self, m: &MatchInfo) -> Vec<Layout> {
        vec![Layout::RowMajorBf16; m.boundary_outputs.len()]
    }

    fn is_compute_bound(&self) -> bool {
        true
    }

    fn required_weights(
        &self,
        claimed_tiles: &[TileId],
        fuf: &Fuf,
        program: &Program,
    ) -> Vec<WeightAccessor> {
        let sources: Vec<(WeightId, Option<u64>)> = claimed_tiles
            .iter()
            .filter_map(|t| {
                let n = fuf.get(*t);
                if n.op == OpKind::Gemm {
                    first_weight_ref(n)
                } else {
                    None
                }
            })
            .collect();
        let name = fused_accessor_name(program, &sources);
        vec![WeightAccessor {
            name,
            rust_type: quote! { ::ferrite_kernels::layers::Bnb4bitLinear },
            source_weights: sources,
        }]
    }

    fn emit_call(&self, ctx: &EmitCtx) -> TokenStream {
        let gelu_id = *ctx
            .claimed_tiles
            .iter()
            .find(|t| ctx.fuf.get(**t).op == OpKind::Gelu)
            .expect("fused gate/up/gelu/mul claim must contain Gelu");
        let mul_id = *ctx
            .claimed_tiles
            .iter()
            .find(|t| ctx.fuf.get(**t).op == OpKind::Mul)
            .expect("fused gate/up/gelu/mul claim must contain Mul");
        let (gate_id, _) =
            first_tile_input(ctx.fuf.get(gelu_id)).expect("gelu has a tile input — the gate gemm");
        let up_id = ctx
            .claimed_tiles
            .iter()
            .copied()
            .find(|t| ctx.fuf.get(*t).op == OpKind::Gemm && *t != gate_id)
            .expect("claim contains a second Gemm — the up gemm");

        let activation = ctx.input_expr(gate_id, 0);

        let gate_w = first_weight_ref(ctx.fuf.get(gate_id)).expect("gate gemm has a weight");
        let up_w = first_weight_ref(ctx.fuf.get(up_id)).expect("up gemm has a weight");
        let fused_name = fused_accessor_name(ctx.program, &[gate_w, up_w]);
        let weight_expr = ctx.weight_accessor(&fused_name);

        let mul_out = ctx.output_ident(mul_id, 0);
        let intermediate = ctx.bound("intermediate_size") as usize;

        quote! {
            let #mul_out = unsafe {
                let gate_up = (#weight_expr).forward(
                    #activation,
                    &mut device.cublas,
                    &mut device.caching,
                    device.compute_stream,
                );
                ::ferrite_kernels::kernels::gelu_and_mul_fused(
                    *gate_up,
                    #intermediate,
                    &mut device.caching,
                    device.compute_stream,
                )
            };
        }
    }
}

/// Fused gate/up + GELU for FP8 — Gemma2/Gemma3/Granite-style MLP.
/// FP8 counterpart of `Bnb4FusedGateUpGeluMulImpl`.
#[derive(Debug, Default)]
pub struct Fp8FusedGateUpGeluMulImpl;

impl Implementation for Fp8FusedGateUpGeluMulImpl {
    fn name(&self) -> &'static str {
        "fp8_fused_gate_up_gelu_mul"
    }

    fn target_compatible(&self, _profile: &TargetProfile) -> bool {
        true
    }

    fn matches(&self, fuf: &Fuf, seed: TileId, _profile: &TargetProfile) -> Option<MatchInfo> {
        let gate_gemm = fuf.get(seed);
        if !is_fp8_gemm(fuf, seed) {
            return None;
        }
        let gelu_node = fuf
            .nodes
            .iter()
            .find(|n| n.op == OpKind::Gelu && consumes_tile(n, seed))?;
        let gelu_id = gelu_node.id;
        let mul_node = fuf
            .nodes
            .iter()
            .find(|n| n.op == OpKind::Mul && consumes_tile(n, gelu_id))?;
        let mul_id = mul_node.id;
        let up_gemm_id = mul_node.inputs.iter().find_map(|i| match i {
            FufInput::Tile { id, .. } if *id != gelu_id => Some(*id),
            _ => None,
        })?;
        if !is_fp8_gemm(fuf, up_gemm_id) {
            return None;
        }
        if first_tile_input(gate_gemm)? != first_tile_input(fuf.get(up_gemm_id))? {
            return None;
        }
        let mut claimed = [seed, up_gemm_id, gelu_id, mul_id];
        claimed.sort();
        let claimed = claimed.to_vec();
        let activation_tile = first_tile_input(gate_gemm)?.0;
        Some(MatchInfo {
            claimed_tiles: claimed,
            boundary_inputs: vec![activation_tile],
            boundary_outputs: vec![mul_id],
        })
    }

    fn cost_us(&self, m: &MatchInfo, ctx: &CostCtx) -> f64 {
        FusedGateUpGeluMulImpl.cost_us(m, ctx)
    }

    fn resources(&self, _m: &MatchInfo) -> Resources {
        Resources::ZERO
    }

    fn launch_kind(&self) -> LaunchKind {
        LaunchKind::HostCallback
    }

    fn supported_input_handoffs(&self) -> &[Handoff] {
        const H: &[Handoff] = &[Handoff::StreamOrder, Handoff::StreamEvent];
        H
    }

    fn supported_output_handoffs(&self) -> &[Handoff] {
        const H: &[Handoff] = &[Handoff::StreamOrder, Handoff::StreamEvent];
        H
    }

    fn input_layouts(&self, m: &MatchInfo) -> Vec<Layout> {
        vec![Layout::RowMajorBf16; m.boundary_inputs.len()]
    }

    fn output_layouts(&self, m: &MatchInfo) -> Vec<Layout> {
        vec![Layout::RowMajorBf16; m.boundary_outputs.len()]
    }

    fn is_compute_bound(&self) -> bool {
        true
    }

    fn required_weights(
        &self,
        claimed_tiles: &[TileId],
        fuf: &Fuf,
        program: &Program,
    ) -> Vec<WeightAccessor> {
        let sources: Vec<(WeightId, Option<u64>)> = claimed_tiles
            .iter()
            .filter_map(|t| {
                let n = fuf.get(*t);
                if n.op == OpKind::Gemm {
                    first_weight_ref(n)
                } else {
                    None
                }
            })
            .collect();
        let name = fused_accessor_name(program, &sources);
        let gemm_tile = *claimed_tiles
            .iter()
            .find(|t| fuf.get(**t).op == OpKind::Gemm)
            .expect("claim contains at least one Gemm");
        let rust_type = fp8_accessor_type_for(fuf, gemm_tile);
        vec![WeightAccessor {
            name,
            rust_type,
            source_weights: sources,
        }]
    }

    fn emit_call(&self, ctx: &EmitCtx) -> TokenStream {
        let gelu_id = *ctx
            .claimed_tiles
            .iter()
            .find(|t| ctx.fuf.get(**t).op == OpKind::Gelu)
            .expect("fused gate/up/gelu/mul claim must contain Gelu");
        let mul_id = *ctx
            .claimed_tiles
            .iter()
            .find(|t| ctx.fuf.get(**t).op == OpKind::Mul)
            .expect("fused gate/up/gelu/mul claim must contain Mul");
        let (gate_id, _) =
            first_tile_input(ctx.fuf.get(gelu_id)).expect("gelu has a tile input — the gate gemm");
        let up_id = ctx
            .claimed_tiles
            .iter()
            .copied()
            .find(|t| ctx.fuf.get(*t).op == OpKind::Gemm && *t != gate_id)
            .expect("claim contains a second Gemm — the up gemm");

        let activation = ctx.input_expr(gate_id, 0);

        let gate_w = first_weight_ref(ctx.fuf.get(gate_id)).expect("gate gemm has a weight");
        let up_w = first_weight_ref(ctx.fuf.get(up_id)).expect("up gemm has a weight");
        let fused_name = fused_accessor_name(ctx.program, &[gate_w, up_w]);
        let weight_expr = ctx.weight_accessor(&fused_name);

        let mul_out = ctx.output_ident(mul_id, 0);
        let intermediate = ctx.bound("intermediate_size") as usize;

        quote! {
            let #mul_out = unsafe {
                let gate_up = (#weight_expr).forward(
                    #activation,
                    &mut device.cublas,
                    &mut device.caching,
                    device.compute_stream,
                );
                ::ferrite_kernels::kernels::gelu_and_mul_fused(
                    *gate_up,
                    #intermediate,
                    &mut device.caching,
                    device.compute_stream,
                )
            };
        }
    }
}

/// Fused QKV + RoPE (decode, M=1) for FP8. Mirrors
/// `Bnb4FusedQkvRopeCacheImpl` but emits the FP8 matmul path.
///
/// Importantly, this is the impl that makes ferrite's FP8 QKV path
/// numerically equivalent to Python vLLM's `QKVParallelLinear` for
/// per-tensor weight scales: `Fp8Linear::load_concat` max-scale-
/// merges Q/K/V weight_scales via the same `requantize_with_max_scale`
/// Python does before the fused matmul. Without this impl, the
/// singleton `Fp8FusedGemmBiasImpl × 3` fallback keeps the original
/// per-proj scales — arithmetically identical to Python only when
/// weight_scales are per-channel `[N, 1]`, and drifting when they
/// collapse to per-tensor scalars.
#[derive(Debug, Default)]
pub struct Fp8FusedQkvRopeCacheImpl;

impl Implementation for Fp8FusedQkvRopeCacheImpl {
    fn name(&self) -> &'static str {
        "fp8_fused_qkv_rope_cache"
    }

    fn target_compatible(&self, _profile: &TargetProfile) -> bool {
        true
    }

    fn workload_constraint(&self) -> WorkloadConstraint {
        WorkloadConstraint::NumTokensRange { min: 1, max: 1 }
    }

    fn matches(&self, fuf: &Fuf, seed: TileId, _profile: &TargetProfile) -> Option<MatchInfo> {
        let seed_node = fuf.get(seed);
        if !is_fp8_gemm(fuf, seed) {
            return None;
        }

        let rope_node = fuf.nodes.iter().find(|n| {
            if n.op != OpKind::RopeAppend || n.inputs.len() < 3 {
                return false;
            }
            let qkv_raw: Vec<TileId> = n
                .inputs
                .iter()
                .take(3)
                .filter_map(|i| match i {
                    FufInput::Tile { id, .. } => Some(*id),
                    _ => None,
                })
                .collect();
            if qkv_raw.len() != 3 {
                return false;
            }
            let resolved: Option<Vec<(TileId, Option<TileId>)>> = qkv_raw
                .iter()
                .map(|t| unwrap_gemm_through_bias(fuf, *t))
                .collect();
            let Some(resolved) = resolved else {
                return false;
            };
            let biased = resolved[0].1.is_some();
            if resolved.iter().any(|r| r.1.is_some() != biased) {
                return false;
            }
            let gemms: Vec<TileId> = resolved.iter().map(|r| r.0).collect();
            if !gemms.contains(&seed) {
                return false;
            }
            if gemms.iter().any(|t| !is_fp8_gemm(fuf, *t)) {
                return false;
            }
            let act = first_tile_input(fuf.get(gemms[0]));
            act.is_some() && gemms.iter().all(|t| first_tile_input(fuf.get(*t)) == act)
        })?;
        let rope_id = rope_node.id;

        let qkv_raw: Vec<TileId> = rope_node
            .inputs
            .iter()
            .take(3)
            .filter_map(|i| match i {
                FufInput::Tile { id, .. } => Some(*id),
                _ => None,
            })
            .collect();
        let resolved: Vec<(TileId, Option<TileId>)> = qkv_raw
            .iter()
            .map(|t| unwrap_gemm_through_bias(fuf, *t).expect("validated in find"))
            .collect();

        let mut claimed: Vec<TileId> = Vec::with_capacity(7);
        for (g, b) in &resolved {
            claimed.push(*g);
            if let Some(b) = b {
                claimed.push(*b);
            }
        }
        claimed.push(rope_id);
        claimed.sort();

        let _ = seed_node;
        let activation = first_tile_input(fuf.get(resolved[0].0))?.0;
        Some(MatchInfo {
            claimed_tiles: claimed,
            boundary_inputs: vec![activation],
            boundary_outputs: vec![rope_id],
        })
    }

    fn cost_us(&self, m: &MatchInfo, ctx: &CostCtx) -> f64 {
        FusedQkvRopeCacheImpl.cost_us(m, ctx)
    }

    fn resources(&self, _m: &MatchInfo) -> Resources {
        Resources::ZERO
    }

    fn launch_kind(&self) -> LaunchKind {
        LaunchKind::HostCallback
    }

    fn supported_input_handoffs(&self) -> &[Handoff] {
        const H: &[Handoff] = &[Handoff::StreamOrder, Handoff::StreamEvent];
        H
    }

    fn supported_output_handoffs(&self) -> &[Handoff] {
        const H: &[Handoff] = &[Handoff::StreamOrder, Handoff::StreamEvent];
        H
    }

    fn input_layouts(&self, m: &MatchInfo) -> Vec<Layout> {
        vec![Layout::RowMajorBf16; m.boundary_inputs.len()]
    }

    fn output_layouts(&self, m: &MatchInfo) -> Vec<Layout> {
        vec![Layout::RowMajorBf16; m.boundary_outputs.len()]
    }

    fn is_compute_bound(&self) -> bool {
        true
    }

    #[allow(clippy::type_complexity)]
    fn output_alias(
        &self,
        claimed_tiles: &[TileId],
        fuf: &Fuf,
    ) -> Vec<((TileId, u8), Option<(TileId, u8)>)> {
        let rope_id = *claimed_tiles
            .iter()
            .find(|t| fuf.get(**t).op == OpKind::RopeAppend)
            .expect("claim contains RopeAppend");
        vec![((rope_id, 0), None)]
    }

    fn required_weights(
        &self,
        claimed_tiles: &[TileId],
        fuf: &Fuf,
        program: &Program,
    ) -> Vec<WeightAccessor> {
        let sources: Vec<(WeightId, Option<u64>)> = claimed_tiles
            .iter()
            .filter_map(|t| {
                let n = fuf.get(*t);
                if n.op == OpKind::Gemm {
                    first_weight_ref(n)
                } else {
                    None
                }
            })
            .collect();
        let name = fused_accessor_name(program, &sources);
        let gemm_tile = *claimed_tiles
            .iter()
            .find(|t| fuf.get(**t).op == OpKind::Gemm)
            .expect("claim contains at least one Gemm");
        let rust_type = fp8_accessor_type_for(fuf, gemm_tile);
        vec![WeightAccessor {
            name,
            rust_type,
            source_weights: sources,
        }]
    }

    fn emit_call(&self, ctx: &EmitCtx) -> TokenStream {
        let rope_id = *ctx
            .claimed_tiles
            .iter()
            .find(|t| ctx.fuf.get(**t).op == OpKind::RopeAppend)
            .expect("claim must contain a RopeAppend");
        let rope_node = ctx.fuf.get(rope_id);

        let qkv_raw: Vec<TileId> = rope_node
            .inputs
            .iter()
            .take(3)
            .filter_map(|i| match i {
                FufInput::Tile { id, .. } => Some(*id),
                _ => None,
            })
            .collect();
        assert_eq!(qkv_raw.len(), 3, "rope has three tile inputs (q, k, v)");
        let resolved: Vec<(TileId, Option<TileId>)> = qkv_raw
            .iter()
            .map(|t| {
                unwrap_gemm_through_bias(ctx.fuf, *t)
                    .expect("claim-time check guarantees Gemm-or-BiasAdd(Gemm)")
            })
            .collect();
        let q_gemm_id = resolved[0].0;
        let activation = ctx.input_expr(q_gemm_id, 0);

        let qkv_weights: Vec<(WeightId, Option<u64>)> = resolved
            .iter()
            .map(|(g, _)| first_weight_ref(ctx.fuf.get(*g)).expect("gemm has a weight"))
            .collect();
        let fused_name = fused_accessor_name(ctx.program, &qkv_weights);
        let weight_expr = ctx.weight_accessor(&fused_name);

        let num_q_heads = ctx.bound("num_attention_heads") as usize;
        let num_kv_heads = ctx.bound("num_key_value_heads") as usize;
        let head_dim = ctx.bound("head_dim") as usize;
        let q_size = num_q_heads * head_dim;
        let kv_size = num_kv_heads * head_dim;

        let layer = rope_kv_cache_layer(rope_node)
            .expect("RopeAppend has a KvCache extern input with a concrete layer index")
            as usize;

        let q_out = ctx.output_ident(rope_id, 0);
        let k_out = ctx.output_ident(rope_id, 1);
        let v_out = ctx.output_ident(rope_id, 2);

        quote! {
            let #q_out = unsafe {
                let qkv_packed = (#weight_expr).forward(
                    #activation,
                    &mut device.cublas,
                    &mut device.caching,
                    device.compute_stream,
                );
                if ctx.kv_cache.is_fp8() {
                    ::ferrite_kernels::kernels::fused_qkv_rope_cache_fp8(
                        *qkv_packed,
                        *ctx.positions,
                        wm.rotary.cos_sin_cache,
                        *ctx.slot_mapping,
                        *ctx.kv_cache.k_cache(#layer),
                        *ctx.kv_cache.v_cache(#layer),
                        ctx.kv_cache.k_scale_ptr(#layer),
                        ctx.kv_cache.v_scale_ptr(#layer),
                        #q_size,
                        #kv_size,
                        #num_q_heads,
                        #head_dim,
                        &mut device.caching,
                        device.compute_stream,
                    )
                } else {
                    ::ferrite_kernels::kernels::fused_qkv_rope_cache(
                        *qkv_packed,
                        *ctx.positions,
                        wm.rotary.cos_sin_cache,
                        *ctx.slot_mapping,
                        *ctx.kv_cache.k_cache(#layer),
                        *ctx.kv_cache.v_cache(#layer),
                        #q_size,
                        #kv_size,
                        #num_q_heads,
                        #head_dim,
                        &mut device.caching,
                        device.compute_stream,
                    )
                }
            };
            let #k_out = ctx.kv_cache.k_cache(#layer);
            let #v_out = ctx.kv_cache.v_cache(#layer);
        }
    }
}

/// Fused QKV + RoPE (prefill, contiguous K/V) for FP8.
#[derive(Debug, Default)]
pub struct Fp8FusedQkvRopePrefillImpl;

impl Implementation for Fp8FusedQkvRopePrefillImpl {
    fn name(&self) -> &'static str {
        "fp8_fused_qkv_rope_prefill"
    }

    fn target_compatible(&self, _profile: &TargetProfile) -> bool {
        true
    }

    fn workload_constraint(&self) -> WorkloadConstraint {
        WorkloadConstraint::NumTokensRange {
            min: 2,
            max: u32::MAX,
        }
    }

    fn matches(&self, fuf: &Fuf, seed: TileId, profile: &TargetProfile) -> Option<MatchInfo> {
        Fp8FusedQkvRopeCacheImpl.matches(fuf, seed, profile)
    }

    fn cost_us(&self, m: &MatchInfo, ctx: &CostCtx) -> f64 {
        Fp8FusedQkvRopeCacheImpl.cost_us(m, ctx)
    }

    fn resources(&self, _m: &MatchInfo) -> Resources {
        Resources::ZERO
    }

    fn launch_kind(&self) -> LaunchKind {
        LaunchKind::HostCallback
    }

    fn supported_input_handoffs(&self) -> &[Handoff] {
        const H: &[Handoff] = &[Handoff::StreamOrder, Handoff::StreamEvent];
        H
    }

    fn supported_output_handoffs(&self) -> &[Handoff] {
        const H: &[Handoff] = &[Handoff::StreamOrder, Handoff::StreamEvent];
        H
    }

    fn input_layouts(&self, m: &MatchInfo) -> Vec<Layout> {
        vec![Layout::RowMajorBf16; m.boundary_inputs.len()]
    }

    fn output_layouts(&self, m: &MatchInfo) -> Vec<Layout> {
        vec![Layout::RowMajorBf16; m.boundary_outputs.len()]
    }

    fn is_compute_bound(&self) -> bool {
        true
    }

    fn required_weights(
        &self,
        claimed_tiles: &[TileId],
        fuf: &Fuf,
        program: &Program,
    ) -> Vec<WeightAccessor> {
        Fp8FusedQkvRopeCacheImpl.required_weights(claimed_tiles, fuf, program)
    }

    fn emit_call(&self, ctx: &EmitCtx) -> TokenStream {
        let rope_id = *ctx
            .claimed_tiles
            .iter()
            .find(|t| ctx.fuf.get(**t).op == OpKind::RopeAppend)
            .expect("claim must contain a RopeAppend");
        let rope_node = ctx.fuf.get(rope_id);

        let qkv_raw: Vec<TileId> = rope_node
            .inputs
            .iter()
            .take(3)
            .filter_map(|i| match i {
                FufInput::Tile { id, .. } => Some(*id),
                _ => None,
            })
            .collect();
        assert_eq!(qkv_raw.len(), 3, "rope has three tile inputs (q, k, v)");
        let resolved: Vec<(TileId, Option<TileId>)> = qkv_raw
            .iter()
            .map(|t| unwrap_gemm_through_bias(ctx.fuf, *t).expect("claim-time validated"))
            .collect();
        let q_gemm_id = resolved[0].0;
        let activation = ctx.input_expr(q_gemm_id, 0);

        let qkv_weights: Vec<(WeightId, Option<u64>)> = resolved
            .iter()
            .map(|(g, _)| first_weight_ref(ctx.fuf.get(*g)).expect("gemm has a weight"))
            .collect();
        let fused_name = fused_accessor_name(ctx.program, &qkv_weights);
        let weight_expr = ctx.weight_accessor(&fused_name);

        let num_q_heads = ctx.bound("num_attention_heads") as usize;
        let num_kv_heads = ctx.bound("num_key_value_heads") as usize;
        let head_dim = ctx.bound("head_dim") as usize;
        let q_size = num_q_heads * head_dim;
        let kv_size = num_kv_heads * head_dim;

        let layer = rope_kv_cache_layer(rope_node)
            .expect("RopeAppend has a KvCache extern input with a concrete layer index")
            as usize;

        let q_out = ctx.output_ident(rope_id, 0);
        let k_out = ctx.output_ident(rope_id, 1);
        let v_out = ctx.output_ident(rope_id, 2);

        quote! {
            let (#q_out, #k_out, #v_out) = unsafe {
                let qkv_packed = (#weight_expr).forward(
                    #activation,
                    &mut device.cublas,
                    &mut device.caching,
                    device.compute_stream,
                );
                ::ferrite_kernels::kernels::fused_qkv_rope(
                    *qkv_packed,
                    *ctx.positions,
                    wm.rotary.cos_sin_cache,
                    #q_size,
                    #kv_size,
                    #num_q_heads,
                    #num_kv_heads,
                    #head_dim,
                    &mut device.caching,
                    device.compute_stream,
                )
            };
            unsafe {
                ::ferrite_kernels::attention_helpers::write_kv_cache(
                    (#k_out).view(),
                    (#v_out).view(),
                    ctx.slot_mapping,
                    ctx.kv_cache,
                    #layer,
                    device.compute_stream,
                );
            }
        }
    }
}

/// Fused QKV + RoPE (decode, M=1) for BNB4. Mirrors
/// `MarlinFusedQkvRopeCacheImpl` but emits the BNB4 matmul path.
#[derive(Debug, Default)]
pub struct Bnb4FusedQkvRopeCacheImpl;

impl Implementation for Bnb4FusedQkvRopeCacheImpl {
    fn name(&self) -> &'static str {
        "bnb4_fused_qkv_rope_cache"
    }

    fn target_compatible(&self, _profile: &TargetProfile) -> bool {
        true
    }

    fn workload_constraint(&self) -> WorkloadConstraint {
        WorkloadConstraint::NumTokensRange { min: 1, max: 1 }
    }

    fn matches(&self, fuf: &Fuf, seed: TileId, _profile: &TargetProfile) -> Option<MatchInfo> {
        let seed_node = fuf.get(seed);
        if !is_bnb4_gemm(fuf, seed) {
            return None;
        }

        let rope_node = fuf.nodes.iter().find(|n| {
            if n.op != OpKind::RopeAppend || n.inputs.len() < 3 {
                return false;
            }
            let qkv_raw: Vec<TileId> = n
                .inputs
                .iter()
                .take(3)
                .filter_map(|i| match i {
                    FufInput::Tile { id, .. } => Some(*id),
                    _ => None,
                })
                .collect();
            if qkv_raw.len() != 3 {
                return false;
            }
            let resolved: Option<Vec<(TileId, Option<TileId>)>> = qkv_raw
                .iter()
                .map(|t| unwrap_gemm_through_bias(fuf, *t))
                .collect();
            let Some(resolved) = resolved else {
                return false;
            };
            let biased = resolved[0].1.is_some();
            if resolved.iter().any(|r| r.1.is_some() != biased) {
                return false;
            }
            let gemms: Vec<TileId> = resolved.iter().map(|r| r.0).collect();
            if !gemms.contains(&seed) {
                return false;
            }
            if gemms.iter().any(|t| !is_bnb4_gemm(fuf, *t)) {
                return false;
            }
            let act = first_tile_input(fuf.get(gemms[0]));
            act.is_some() && gemms.iter().all(|t| first_tile_input(fuf.get(*t)) == act)
        })?;
        let rope_id = rope_node.id;

        let qkv_raw: Vec<TileId> = rope_node
            .inputs
            .iter()
            .take(3)
            .filter_map(|i| match i {
                FufInput::Tile { id, .. } => Some(*id),
                _ => None,
            })
            .collect();
        let resolved: Vec<(TileId, Option<TileId>)> = qkv_raw
            .iter()
            .map(|t| unwrap_gemm_through_bias(fuf, *t).expect("validated in find"))
            .collect();

        let mut claimed: Vec<TileId> = Vec::with_capacity(7);
        for (g, b) in &resolved {
            claimed.push(*g);
            if let Some(b) = b {
                claimed.push(*b);
            }
        }
        claimed.push(rope_id);
        claimed.sort();

        let _ = seed_node;
        let activation = first_tile_input(fuf.get(resolved[0].0))?.0;
        Some(MatchInfo {
            claimed_tiles: claimed,
            boundary_inputs: vec![activation],
            boundary_outputs: vec![rope_id],
        })
    }

    fn cost_us(&self, m: &MatchInfo, ctx: &CostCtx) -> f64 {
        FusedQkvRopeCacheImpl.cost_us(m, ctx)
    }

    fn resources(&self, _m: &MatchInfo) -> Resources {
        Resources::ZERO
    }

    fn launch_kind(&self) -> LaunchKind {
        LaunchKind::HostCallback
    }

    fn supported_input_handoffs(&self) -> &[Handoff] {
        const H: &[Handoff] = &[Handoff::StreamOrder, Handoff::StreamEvent];
        H
    }

    fn supported_output_handoffs(&self) -> &[Handoff] {
        const H: &[Handoff] = &[Handoff::StreamOrder, Handoff::StreamEvent];
        H
    }

    fn input_layouts(&self, m: &MatchInfo) -> Vec<Layout> {
        vec![Layout::RowMajorBf16; m.boundary_inputs.len()]
    }

    fn output_layouts(&self, m: &MatchInfo) -> Vec<Layout> {
        vec![Layout::RowMajorBf16; m.boundary_outputs.len()]
    }

    fn is_compute_bound(&self) -> bool {
        true
    }

    #[allow(clippy::type_complexity)]
    fn output_alias(
        &self,
        claimed_tiles: &[TileId],
        fuf: &Fuf,
    ) -> Vec<((TileId, u8), Option<(TileId, u8)>)> {
        let rope_id = *claimed_tiles
            .iter()
            .find(|t| fuf.get(**t).op == OpKind::RopeAppend)
            .expect("claim contains RopeAppend");
        vec![((rope_id, 0), None)]
    }

    fn required_weights(
        &self,
        claimed_tiles: &[TileId],
        fuf: &Fuf,
        program: &Program,
    ) -> Vec<WeightAccessor> {
        let sources: Vec<(WeightId, Option<u64>)> = claimed_tiles
            .iter()
            .filter_map(|t| {
                let n = fuf.get(*t);
                if n.op == OpKind::Gemm {
                    first_weight_ref(n)
                } else {
                    None
                }
            })
            .collect();
        let name = fused_accessor_name(program, &sources);
        vec![WeightAccessor {
            name,
            rust_type: quote! { ::ferrite_kernels::layers::Bnb4bitLinear },
            source_weights: sources,
        }]
    }

    fn emit_call(&self, ctx: &EmitCtx) -> TokenStream {
        let rope_id = *ctx
            .claimed_tiles
            .iter()
            .find(|t| ctx.fuf.get(**t).op == OpKind::RopeAppend)
            .expect("claim must contain a RopeAppend");
        let rope_node = ctx.fuf.get(rope_id);

        let qkv_raw: Vec<TileId> = rope_node
            .inputs
            .iter()
            .take(3)
            .filter_map(|i| match i {
                FufInput::Tile { id, .. } => Some(*id),
                _ => None,
            })
            .collect();
        assert_eq!(qkv_raw.len(), 3, "rope has three tile inputs (q, k, v)");
        let resolved: Vec<(TileId, Option<TileId>)> = qkv_raw
            .iter()
            .map(|t| {
                unwrap_gemm_through_bias(ctx.fuf, *t)
                    .expect("claim-time check guarantees Gemm-or-BiasAdd(Gemm)")
            })
            .collect();
        let q_gemm_id = resolved[0].0;
        let activation = ctx.input_expr(q_gemm_id, 0);

        let qkv_weights: Vec<(WeightId, Option<u64>)> = resolved
            .iter()
            .map(|(g, _)| first_weight_ref(ctx.fuf.get(*g)).expect("gemm has a weight"))
            .collect();
        let fused_name = fused_accessor_name(ctx.program, &qkv_weights);
        let weight_expr = ctx.weight_accessor(&fused_name);

        let num_q_heads = ctx.bound("num_attention_heads") as usize;
        let num_kv_heads = ctx.bound("num_key_value_heads") as usize;
        let head_dim = ctx.bound("head_dim") as usize;
        let q_size = num_q_heads * head_dim;
        let kv_size = num_kv_heads * head_dim;

        let layer = rope_kv_cache_layer(rope_node)
            .expect("RopeAppend has a KvCache extern input with a concrete layer index")
            as usize;

        let q_out = ctx.output_ident(rope_id, 0);
        let k_out = ctx.output_ident(rope_id, 1);
        let v_out = ctx.output_ident(rope_id, 2);

        quote! {
            let #q_out = unsafe {
                let qkv_packed = (#weight_expr).forward(
                    #activation,
                    &mut device.cublas,
                    &mut device.caching,
                    device.compute_stream,
                );
                if ctx.kv_cache.is_fp8() {
                    ::ferrite_kernels::kernels::fused_qkv_rope_cache_fp8(
                        *qkv_packed,
                        *ctx.positions,
                        wm.rotary.cos_sin_cache,
                        *ctx.slot_mapping,
                        *ctx.kv_cache.k_cache(#layer),
                        *ctx.kv_cache.v_cache(#layer),
                        ctx.kv_cache.k_scale_ptr(#layer),
                        ctx.kv_cache.v_scale_ptr(#layer),
                        #q_size,
                        #kv_size,
                        #num_q_heads,
                        #head_dim,
                        &mut device.caching,
                        device.compute_stream,
                    )
                } else {
                    ::ferrite_kernels::kernels::fused_qkv_rope_cache(
                        *qkv_packed,
                        *ctx.positions,
                        wm.rotary.cos_sin_cache,
                        *ctx.slot_mapping,
                        *ctx.kv_cache.k_cache(#layer),
                        *ctx.kv_cache.v_cache(#layer),
                        #q_size,
                        #kv_size,
                        #num_q_heads,
                        #head_dim,
                        &mut device.caching,
                        device.compute_stream,
                    )
                }
            };
            let #k_out = ctx.kv_cache.k_cache(#layer);
            let #v_out = ctx.kv_cache.v_cache(#layer);
        }
    }
}

/// Fused QKV + RoPE (prefill, contiguous K/V) for BNB4.
#[derive(Debug, Default)]
pub struct Bnb4FusedQkvRopePrefillImpl;

impl Implementation for Bnb4FusedQkvRopePrefillImpl {
    fn name(&self) -> &'static str {
        "bnb4_fused_qkv_rope_prefill"
    }

    fn target_compatible(&self, _profile: &TargetProfile) -> bool {
        true
    }

    fn workload_constraint(&self) -> WorkloadConstraint {
        WorkloadConstraint::NumTokensRange {
            min: 2,
            max: u32::MAX,
        }
    }

    fn matches(&self, fuf: &Fuf, seed: TileId, profile: &TargetProfile) -> Option<MatchInfo> {
        Bnb4FusedQkvRopeCacheImpl.matches(fuf, seed, profile)
    }

    fn cost_us(&self, m: &MatchInfo, ctx: &CostCtx) -> f64 {
        Bnb4FusedQkvRopeCacheImpl.cost_us(m, ctx)
    }

    fn resources(&self, _m: &MatchInfo) -> Resources {
        Resources::ZERO
    }

    fn launch_kind(&self) -> LaunchKind {
        LaunchKind::HostCallback
    }

    fn supported_input_handoffs(&self) -> &[Handoff] {
        const H: &[Handoff] = &[Handoff::StreamOrder, Handoff::StreamEvent];
        H
    }

    fn supported_output_handoffs(&self) -> &[Handoff] {
        const H: &[Handoff] = &[Handoff::StreamOrder, Handoff::StreamEvent];
        H
    }

    fn input_layouts(&self, m: &MatchInfo) -> Vec<Layout> {
        vec![Layout::RowMajorBf16; m.boundary_inputs.len()]
    }

    fn output_layouts(&self, m: &MatchInfo) -> Vec<Layout> {
        vec![Layout::RowMajorBf16; m.boundary_outputs.len()]
    }

    fn is_compute_bound(&self) -> bool {
        true
    }

    fn required_weights(
        &self,
        claimed_tiles: &[TileId],
        fuf: &Fuf,
        program: &Program,
    ) -> Vec<WeightAccessor> {
        Bnb4FusedQkvRopeCacheImpl.required_weights(claimed_tiles, fuf, program)
    }

    fn emit_call(&self, ctx: &EmitCtx) -> TokenStream {
        let rope_id = *ctx
            .claimed_tiles
            .iter()
            .find(|t| ctx.fuf.get(**t).op == OpKind::RopeAppend)
            .expect("claim must contain a RopeAppend");
        let rope_node = ctx.fuf.get(rope_id);

        let qkv_raw: Vec<TileId> = rope_node
            .inputs
            .iter()
            .take(3)
            .filter_map(|i| match i {
                FufInput::Tile { id, .. } => Some(*id),
                _ => None,
            })
            .collect();
        assert_eq!(qkv_raw.len(), 3, "rope has three tile inputs (q, k, v)");
        let resolved: Vec<(TileId, Option<TileId>)> = qkv_raw
            .iter()
            .map(|t| unwrap_gemm_through_bias(ctx.fuf, *t).expect("claim-time validated"))
            .collect();
        let q_gemm_id = resolved[0].0;
        let activation = ctx.input_expr(q_gemm_id, 0);

        let qkv_weights: Vec<(WeightId, Option<u64>)> = resolved
            .iter()
            .map(|(g, _)| first_weight_ref(ctx.fuf.get(*g)).expect("gemm has a weight"))
            .collect();
        let fused_name = fused_accessor_name(ctx.program, &qkv_weights);
        let weight_expr = ctx.weight_accessor(&fused_name);

        let num_q_heads = ctx.bound("num_attention_heads") as usize;
        let num_kv_heads = ctx.bound("num_key_value_heads") as usize;
        let head_dim = ctx.bound("head_dim") as usize;
        let q_size = num_q_heads * head_dim;
        let kv_size = num_kv_heads * head_dim;

        let layer = rope_kv_cache_layer(rope_node)
            .expect("RopeAppend has a KvCache extern input with a concrete layer index")
            as usize;

        let q_out = ctx.output_ident(rope_id, 0);
        let k_out = ctx.output_ident(rope_id, 1);
        let v_out = ctx.output_ident(rope_id, 2);

        quote! {
            let (#q_out, #k_out, #v_out) = unsafe {
                let qkv_packed = (#weight_expr).forward(
                    #activation,
                    &mut device.cublas,
                    &mut device.caching,
                    device.compute_stream,
                );
                ::ferrite_kernels::kernels::fused_qkv_rope(
                    *qkv_packed,
                    *ctx.positions,
                    wm.rotary.cos_sin_cache,
                    #q_size,
                    #kv_size,
                    #num_q_heads,
                    #num_kv_heads,
                    #head_dim,
                    &mut device.caching,
                    device.compute_stream,
                )
            };
            unsafe {
                ::ferrite_kernels::attention_helpers::write_kv_cache(
                    (#k_out).view(),
                    (#v_out).view(),
                    ctx.slot_mapping,
                    ctx.kv_cache,
                    #layer,
                    device.compute_stream,
                );
            }
        }
    }
}

// ── FlashInfer attention Impls ───────────────────────────────────
//
// Alternative to `AttentionViaCacheImpl` (decode) and
// `AttentionPrefillContiguousImpl` (prefill). Reads Q from the DSL
// tile's slot 0 and K/V from the paged cache written by the upstream
// fused QKV-rope tile, same as the FA2 variants, but dispatches
// through FlashInfer's persistent batch-attention kernel when the
// target has a calibrated CSV row for this `(head_dim, softcap,
// num_q_heads, num_kv_heads)` tuple. When the FFI tuple isn't
// compiled in, the emitted code falls back to the matching FA2 path.
//
// One Impl pair (Decode + Prefill) is registered per tuple in
// `FLASHINFER_CONFIG_SET`; the solver's 2-D `(num_tokens, sk_bucket)`
// sweep picks whichever wins per cell based on the calibrated costs.
// Without CSV data the prefix-gated `target_compatible` rejects the
// FI Impls and the non-FI attention Impls cover the workload.

/// Finite sk_bucket range the attention sweep calibrates. Matches
/// `ATTN_SK_BUCKETS` in `ferrite-kernels::attention_helpers` and the
/// `SK_VALUES` grid in `ferrite-cost-sweep::attention_sweep`. Changing
/// either end requires re-sweeping the target CSV.
const FI_SK_BUCKET_MIN: u64 = 128;
const FI_SK_BUCKET_MAX: u64 = 8192;

/// Build the CSV kernel name the FI Impls look up for a given
/// `(head_dim, softcap)` tuple. `(num_q_heads, num_kv_heads)` is a
/// runtime parameter of the compiled FI kernel — the same shim
/// handles any GQA ratio — so the cost table is keyed only on the
/// compile-time specialization dims. Must match the row names the
/// attention sweep emits — keep in sync with
/// `ferrite-cost-sweep/src/attention_sweep.rs`.
fn fi_csv_name(head_dim: u32, use_logits_soft_cap: bool) -> String {
    let softcap_tok = if use_logits_soft_cap {
        "softcap"
    } else {
        "nosoftcap"
    };
    format!("flashinfer_attn_bf16_h{head_dim}_{softcap_tok}")
}

/// Shared cost lookup — identical between Decode and Prefill Impls:
/// the FI plan builds the same work for both, and both variants live
/// in the same CSV row family (disambiguated at solver time by the
/// workload constraint on `num_tokens`).
fn fi_cost_us(head_dim: u32, use_logits_soft_cap: bool, ctx: &CostCtx) -> f64 {
    // Fast reject when the model's head_dim doesn't match the baked
    // tuple — every FI variant walks every Attention tile; only one's
    // head_dim matches any given model. `UNCALIBRATED_COST_US` (finite
    // sentinel) rather than `INFINITY` so the DP's non-finite-cost
    // guard (`SolveError::UnreachableCost`) doesn't fire.
    let Some(model_head_dim) = ctx.bounds.get("head_dim").copied() else {
        return UNCALIBRATED_COST_US;
    };
    if model_head_dim as u32 != head_dim {
        return UNCALIBRATED_COST_US;
    }
    let name = fi_csv_name(head_dim, use_logits_soft_cap);
    let nt = ctx.num_tokens() as u32;
    let sk = ctx.sk_bucket() as u32;
    ctx.profile
        .cost_us_for(&name, nt, sk, head_dim)
        .unwrap_or(UNCALIBRATED_COST_US)
}

/// Emit the FlashInfer call + its FA2 fallback for the decode path.
/// Factored so both the Decode Impl's `emit_call` and any future
/// sliding-window FI variant can share the body.
fn emit_fi_decode_body(ctx: &EmitCtx, head_dim: u32, use_logits_soft_cap: bool) -> TokenStream {
    let tile = ctx.primary();
    let out = ctx.output_ident(tile, 0);
    let node = ctx.fuf.get(tile);

    let q_expr = ctx.input_expr(tile, 0);
    let layer =
        node.inputs
            .iter()
            .find_map(|i| match i {
                FufInput::Extern {
                    kind: crate::classified::ExternKind::KvCache,
                    index: Some(layer),
                } => Some(*layer),
                _ => None,
            })
            .expect("Attention has a kv_cache extern with a concrete layer index") as usize;

    let scale_tokens = attention_scale_tokens(ctx);
    let softcap_tokens = attention_softcap_tokens(ctx);
    let q_size = (ctx.bound("num_attention_heads") * ctx.bound("head_dim")) as usize;

    // FlashInferConfig literal baked from the Impl's state. The
    // emitted `dtype` is always `Bf16` — the only dtype currently in
    // FLASHINFER_CONFIG_SET (Fp16 is a placeholder; see
    // `ferrite-cuda-builder::flashinfer_config`).
    quote! {
        let mut #out = unsafe {
            let fi_cfg = ::ferrite_kernels::flashinfer::FlashInferConfig {
                dtype: ::ferrite_kernels::flashinfer::FiDType::Bf16,
                head_dim: #head_dim,
                use_logits_soft_cap: #use_logits_soft_cap,
            };
            let sk_bucket = ::ferrite_kernels::attention_helpers::sk_bucket_for(
                ctx.max_seqlen_k,
            );
            let fi = ::ferrite_kernels::attention_helpers::flashinfer_attention(
                #q_expr,
                ctx.cu_seqlens_q,
                ctx.seqused_k,
                ctx.block_table,
                ctx.max_seqlen_q,
                ctx.max_seqlen_k,
                #scale_tokens,
                #softcap_tokens,
                ctx.kv_cache,
                #layer,
                device.num_sm,
                fi_cfg,
                sk_bucket,
                &mut device.caching,
                device.compute_stream,
            );
            match fi {
                Some(t) => t,
                None => {
                    // FA2 fallback — mirrors `AttentionViaCacheImpl`.
                    let has_spans = !ctx.kv_cache.block_unrotated_gpu().is_null();
                    let (cos_sin_ptr, rotary_dim) = if has_spans {
                        (
                            wm.rotary.cos_sin_cache.raw_ptr() as *const u8,
                            wm.rotary.cos_sin_cache.dim(1),
                        )
                    } else {
                        (::std::ptr::null::<u8>(), 0)
                    };
                    ::ferrite_kernels::attention_helpers::attention_decode_from_cache(
                        #q_expr,
                        ctx.cu_seqlens_q,
                        ctx.seqused_k,
                        ctx.block_table,
                        ctx.max_seqlen_q,
                        ctx.max_seqlen_k,
                        #scale_tokens,
                        #softcap_tokens,
                        -1,
                        ctx.kv_cache,
                        #layer,
                        device.num_sm,
                        &mut device.caching,
                        device.compute_stream,
                        cos_sin_ptr,
                        rotary_dim,
                        false,
                    )
                }
            }
        };
        unsafe {
            let nt = (*#out).dim(0);
            let dt = (*#out).dtype();
            #out.reshape(&[nt, #q_size], dt);
        }
    }
}

/// Prefill counterpart. Differs from the decode body in the FA2
/// fallback path: reads K/V from the DSL tile's slots 1/2 (the
/// contiguous outputs of `FusedQkvRopePrefillImpl`) and calls
/// `flash_attn_contiguous` instead of the paged decode helper. The
/// FlashInfer call is identical — the shim is paged-only and reads
/// the same cache `FusedQkvRopePrefillImpl` writes.
fn emit_fi_prefill_body(ctx: &EmitCtx, head_dim: u32, use_logits_soft_cap: bool) -> TokenStream {
    let tile = ctx.primary();
    let out = ctx.output_ident(tile, 0);
    let node = ctx.fuf.get(tile);

    let q_expr = ctx.input_expr(tile, 0);
    let k_expr = ctx.input_expr(tile, 1);
    let v_expr = ctx.input_expr(tile, 2);
    let layer =
        node.inputs
            .iter()
            .find_map(|i| match i {
                FufInput::Extern {
                    kind: crate::classified::ExternKind::KvCache,
                    index: Some(layer),
                } => Some(*layer),
                _ => None,
            })
            .expect("Attention has a kv_cache extern with a concrete layer index") as usize;

    let scale_tokens = attention_scale_tokens(ctx);
    let softcap_tokens = attention_softcap_tokens(ctx);
    let q_size = (ctx.bound("num_attention_heads") * ctx.bound("head_dim")) as usize;

    quote! {
        let mut #out = unsafe {
            let fi_cfg = ::ferrite_kernels::flashinfer::FlashInferConfig {
                dtype: ::ferrite_kernels::flashinfer::FiDType::Bf16,
                head_dim: #head_dim,
                use_logits_soft_cap: #use_logits_soft_cap,
            };
            let sk_bucket = ::ferrite_kernels::attention_helpers::sk_bucket_for(
                ctx.max_seqlen_k,
            );
            let fi = ::ferrite_kernels::attention_helpers::flashinfer_attention(
                #q_expr,
                ctx.cu_seqlens_q,
                ctx.seqused_k,
                ctx.block_table,
                ctx.max_seqlen_q,
                ctx.max_seqlen_k,
                #scale_tokens,
                #softcap_tokens,
                ctx.kv_cache,
                #layer,
                device.num_sm,
                fi_cfg,
                sk_bucket,
                &mut device.caching,
                device.compute_stream,
            );
            match fi {
                Some(t) => t,
                None => {
                    // FA2 fallback — mirrors `AttentionPrefillContiguousImpl`.
                    ::ferrite_kernels::kernels::flash_attn_contiguous(
                        *#q_expr,
                        *#k_expr,
                        *#v_expr,
                        *ctx.cu_seqlens_q,
                        *ctx.cu_seqlens_q,
                        ctx.max_seqlen_q,
                        ctx.max_seqlen_k,
                        #scale_tokens,
                        true,
                        #softcap_tokens,
                        -1,
                        &mut device.caching,
                        device.compute_stream,
                        ::std::ptr::null::<u8>(),
                        0,
                        false,
                    )
                }
            }
        };
        unsafe {
            let nt = (*#out).dim(0);
            let dt = (*#out).dtype();
            #out.reshape(&[nt, #q_size], dt);
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct FlashInferAttentionDecodeImpl {
    pub head_dim: u32,
    pub use_logits_soft_cap: bool,
}

impl Implementation for FlashInferAttentionDecodeImpl {
    fn name(&self) -> &'static str {
        // Variant-level identification happens through the CSV row
        // name; a single name is enough for the solver's diagnostics.
        "flashinfer_attention_decode"
    }

    fn target_compatible(&self, profile: &TargetProfile) -> bool {
        // O(1) membership check on the fully-qualified FI kernel name.
        // A row-less target (no sweep data for this head_dim × softcap)
        // silently falls back to FA2 via the `None` emit-call branch.
        profile
            .cost_table
            .has_kernel(&fi_csv_name(self.head_dim, self.use_logits_soft_cap))
    }

    fn workload_constraint(&self) -> WorkloadConstraint {
        WorkloadConstraint::NumTokensAndSkRange {
            num_tokens: (1, 1),
            sk_bucket: (FI_SK_BUCKET_MIN, FI_SK_BUCKET_MAX),
        }
    }

    fn matches(&self, fuf: &Fuf, seed: TileId, _profile: &TargetProfile) -> Option<MatchInfo> {
        single_tile_match(fuf, seed, OpKind::Attention)
    }

    fn cost_us(&self, _m: &MatchInfo, ctx: &CostCtx) -> f64 {
        fi_cost_us(self.head_dim, self.use_logits_soft_cap, ctx)
    }

    fn resources(&self, _m: &MatchInfo) -> Resources {
        Resources::ZERO
    }

    fn launch_kind(&self) -> LaunchKind {
        LaunchKind::HostCallback
    }

    fn supported_input_handoffs(&self) -> &[Handoff] {
        const H: &[Handoff] = &[Handoff::StreamOrder, Handoff::StreamEvent];
        H
    }

    fn supported_output_handoffs(&self) -> &[Handoff] {
        const H: &[Handoff] = &[Handoff::StreamOrder, Handoff::StreamEvent];
        H
    }

    fn input_layouts(&self, m: &MatchInfo) -> Vec<Layout> {
        vec![Layout::RowMajorBf16; m.boundary_inputs.len()]
    }

    fn output_layouts(&self, m: &MatchInfo) -> Vec<Layout> {
        vec![Layout::RowMajorBf16; m.boundary_outputs.len()]
    }

    fn is_compute_bound(&self) -> bool {
        true
    }

    fn emit_call(&self, ctx: &EmitCtx) -> TokenStream {
        emit_fi_decode_body(ctx, self.head_dim, self.use_logits_soft_cap)
    }
}

#[derive(Debug, Clone, Copy)]
pub struct FlashInferAttentionPrefillImpl {
    pub head_dim: u32,
    pub use_logits_soft_cap: bool,
}

impl Implementation for FlashInferAttentionPrefillImpl {
    fn name(&self) -> &'static str {
        "flashinfer_attention_prefill"
    }

    fn target_compatible(&self, profile: &TargetProfile) -> bool {
        profile
            .cost_table
            .has_kernel(&fi_csv_name(self.head_dim, self.use_logits_soft_cap))
    }

    fn workload_constraint(&self) -> WorkloadConstraint {
        WorkloadConstraint::NumTokensAndSkRange {
            num_tokens: (2, u32::MAX),
            sk_bucket: (FI_SK_BUCKET_MIN, FI_SK_BUCKET_MAX),
        }
    }

    fn matches(&self, fuf: &Fuf, seed: TileId, _profile: &TargetProfile) -> Option<MatchInfo> {
        single_tile_match(fuf, seed, OpKind::Attention)
    }

    fn cost_us(&self, _m: &MatchInfo, ctx: &CostCtx) -> f64 {
        fi_cost_us(self.head_dim, self.use_logits_soft_cap, ctx)
    }

    fn resources(&self, _m: &MatchInfo) -> Resources {
        Resources::ZERO
    }

    fn launch_kind(&self) -> LaunchKind {
        LaunchKind::HostCallback
    }

    fn supported_input_handoffs(&self) -> &[Handoff] {
        const H: &[Handoff] = &[Handoff::StreamOrder, Handoff::StreamEvent];
        H
    }

    fn supported_output_handoffs(&self) -> &[Handoff] {
        const H: &[Handoff] = &[Handoff::StreamOrder, Handoff::StreamEvent];
        H
    }

    fn input_layouts(&self, m: &MatchInfo) -> Vec<Layout> {
        vec![Layout::RowMajorBf16; m.boundary_inputs.len()]
    }

    fn output_layouts(&self, m: &MatchInfo) -> Vec<Layout> {
        vec![Layout::RowMajorBf16; m.boundary_outputs.len()]
    }

    fn is_compute_bound(&self) -> bool {
        true
    }

    fn emit_call(&self, ctx: &EmitCtx) -> TokenStream {
        emit_fi_prefill_body(ctx, self.head_dim, self.use_logits_soft_cap)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn attention_scalars_default_to_llama_values_when_config_silent() {
        // Llama-3.2-1B has no `query_pre_attn_scalar` or
        // `attn_logit_softcapping` — the helpers must fall back to
        // `1/sqrt(head_dim)` and `0.0` respectively, matching the
        // previously hardcoded emission byte-for-byte.
        let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("..")
            .join("..")
            .join("..")
            .join("model_architectures")
            .join("llama")
            .join("llama-3.2-1b.json");
        let model = crate::config::load_file(&path).expect("load llama-3.2-1b");

        let expected_scale = 1.0_f32 / (*model.bounds.get("head_dim").unwrap() as f32).sqrt();
        assert_eq!(attention_scale_for(&model), expected_scale);
        assert_eq!(attention_softcap_for(&model), 0.0);
    }

    #[test]
    fn attention_scalars_honor_gemma_style_config() {
        // Synthetic ModelParams mirroring Gemma2-2B's attention
        // config. The scale must flow from `query_pre_attn_scalar`,
        // not from `head_dim`; softcap must match the capping value.
        let mut bounds = std::collections::BTreeMap::new();
        bounds.insert("head_dim".to_string(), 256);
        let mut scalars = std::collections::BTreeMap::new();
        // Gemma2 ships `query_pre_attn_scalar = 256.0` on the 2B and 9B.
        scalars.insert("query_pre_attn_scalar".to_string(), 256.0);
        scalars.insert("attn_logit_softcapping".to_string(), 50.0);
        let model = crate::config::ModelParams {
            name: "gemma2_test".to_string(),
            source_stem: "gemma2_test".into(),
            source_path: std::path::PathBuf::new(),
            bounds,
            scalars,
            quantization: None,
            tie_word_embeddings: false,
            architectures: Vec::new(),
            extra_tracked_paths: Vec::new(),
            rope_scaling: None,
            rope_scaling_hash: None,
        };

        let scale = attention_scale_for(&model);
        // scale = 256^-0.5 = 1/16 = 0.0625. Exact in f32.
        assert_eq!(scale, 0.0625_f32);
        assert_eq!(attention_softcap_for(&model), 50.0_f32);
    }

    #[test]
    fn resources_union_max_takes_elementwise_max() {
        let a = Resources {
            shmem_bytes: 16 * 1024,
            regs_per_thread: 48,
            threads_per_cta: 128,
        };
        let b = Resources {
            shmem_bytes: 32 * 1024,
            regs_per_thread: 40,
            threads_per_cta: 256,
        };
        let u = a.union_max(&b);
        assert_eq!(u.shmem_bytes, 32 * 1024);
        assert_eq!(u.regs_per_thread, 48);
        assert_eq!(u.threads_per_cta, 256);
        assert_eq!(a.union_max(&a), a);
        assert_eq!(a.union_max(&Resources::ZERO), a);
    }

    #[test]
    fn handoff_variants_have_finite_nonneg_cost_on_any_profile() {
        let profile = crate::target::load_file(
            std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
                .join("..")
                .join("..")
                .join("..")
                .join("target_profiles")
                .join("l4_sm89.json")
                .as_path(),
        )
        .unwrap();
        for h in [
            Handoff::StreamOrder,
            Handoff::StreamEvent,
            Handoff::KernelBoundary,
            Handoff::InKernelGridSync,
            Handoff::Mbarrier,
            Handoff::DsmemRead,
            Handoff::GmemFlag,
            Handoff::SyncThreads,
            Handoff::Internal,
        ] {
            let c = h.cost_us(&profile);
            assert!(c.is_finite(), "{h:?} → {c}");
            assert!(c >= 0.0, "{h:?} → {c}");
        }
    }

    #[test]
    fn handoff_truncation_classification() {
        // Gmem-transiting handoffs truncate; register/shmem
        // handoffs don't.
        let truncating = [
            Handoff::StreamOrder,
            Handoff::StreamEvent,
            Handoff::KernelBoundary,
            Handoff::InKernelGridSync,
            Handoff::GmemFlag,
            Handoff::SyncThreads,
        ];
        let preserving = [Handoff::Internal, Handoff::Mbarrier, Handoff::DsmemRead];
        for h in truncating {
            assert!(h.truncates_to_storage_dtype(), "{h:?}");
        }
        for h in preserving {
            assert!(!h.truncates_to_storage_dtype(), "{h:?}");
        }
    }

    #[test]
    fn layout_any_matches_everything() {
        assert!(Layout::Any.is_compatible_with(Layout::RowMajorBf16));
        assert!(Layout::Any.is_compatible_with(Layout::ColMajorBf16));
        assert!(Layout::RowMajorBf16.is_compatible_with(Layout::Any));
        assert!(Layout::ColMajorBf16.is_compatible_with(Layout::Any));
        assert!(Layout::RowMajorBf16.is_compatible_with(Layout::RowMajorBf16));
        assert!(!Layout::RowMajorBf16.is_compatible_with(Layout::ColMajorBf16));
    }

    #[test]
    fn workload_constraint_accepts_inside_range() {
        // NumTokensRange ignores sk_bucket.
        let c = WorkloadConstraint::NumTokensRange { min: 1, max: 8 };
        assert!(c.accepts(1, 0));
        assert!(c.accepts(8, 4096));
        assert!(!c.accepts(9, 0));
        assert!(!c.accepts(0, 0));
        // Any accepts all.
        assert!(WorkloadConstraint::Any.accepts(4096, 8192));
        // NumTokensAndSkRange gates on both axes.
        let s = WorkloadConstraint::NumTokensAndSkRange {
            num_tokens: (1, 1),
            sk_bucket: (1024, 8192),
        };
        assert!(s.accepts(1, 1024));
        assert!(s.accepts(1, 8192));
        assert!(!s.accepts(1, 512)); // below sk range
        assert!(!s.accepts(1, 16384)); // above sk range
        assert!(!s.accepts(2, 4096)); // num_tokens outside
    }

    #[test]
    fn cutlass_tile_zoo_matches_csv_kernel_names() {
        // The library registers one CutlassGemmImpl per CUTLASS_TILE_ZOO
        // entry plus CutlassGemvImpl. Each entry's `csv_name()` must
        // correspond to a real kernel string in the L4 cost table
        // (byte-for-byte), otherwise `target_compatible` rejects every
        // tile and the solver silently falls back to cuBLAS.
        let profile = crate::target::load_file(
            std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
                .join("..")
                .join("..")
                .join("..")
                .join("target_profiles")
                .join("l4_sm89.json")
                .as_path(),
        )
        .expect("l4_sm89 loads");
        let csv_kernels = profile.cost_table.kernel_names();
        for (tm, tn, st) in CUTLASS_TILE_ZOO {
            let name = format!("cutlass_{tm}x{tn}_s{st}");
            assert!(
                csv_kernels.iter().any(|k| k == &name),
                "CUTLASS_TILE_ZOO has ({tm}, {tn}, {st}) but no `{name}` row in l4 CSV",
            );
        }
        assert!(csv_kernels.iter().any(|k| k == "cutlass_gemv"));
    }

    #[test]
    fn cutlass_gemm_impl_rejects_fusion_partners() {
        // CutlassGemmImpl.matches must return None for a Gemm whose
        // output feeds a RopeAppend / Silu / Mul — picking cutlass
        // there would break the fused impl's claim on the downstream
        // kernel chain (those consumers have no singleton kernel).
        use crate::classified::{ExternKind, WeightId};
        use crate::fuf::{Fuf, FufInput, FufNode, TileId};
        use crate::shape::Dim;

        let t0 = TileId(0);
        let t1 = TileId(1);
        let t2 = TileId(2);
        let fuf = Fuf {
            nodes: vec![
                FufNode {
                    id: t0,
                    op: OpKind::Embed,
                    inputs: vec![FufInput::Extern {
                        kind: ExternKind::InputIds,
                        index: None,
                    }],
                    outputs: vec![vec![Dim::Lit(1), Dim::Lit(16)]],
                },
                FufNode {
                    id: t1,
                    op: OpKind::Gemm,
                    inputs: vec![
                        FufInput::Tile { id: t0, slot: 0 },
                        FufInput::Weight {
                            id: WeightId(0),
                            index: None,
                            storage: crate::quantization::StorageFormat::Dense,
                        },
                    ],
                    outputs: vec![vec![Dim::Lit(1), Dim::Lit(16)]],
                },
                FufNode {
                    id: t2,
                    op: OpKind::RopeAppend,
                    inputs: vec![FufInput::Tile { id: t1, slot: 0 }],
                    outputs: vec![vec![Dim::Lit(1), Dim::Lit(16)]],
                },
            ],
        };
        let imp = CutlassGemmImpl {
            tile_m: 128,
            tile_n: 128,
            stages: 4,
        };
        let profile = crate::target::load_file(
            std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
                .join("..")
                .join("..")
                .join("..")
                .join("target_profiles")
                .join("l4_sm89.json")
                .as_path(),
        )
        .expect("l4_sm89 loads");
        assert!(
            imp.matches(&fuf, t1, &profile).is_none(),
            "cutlass must not match a Gemm whose output feeds RopeAppend",
        );
    }
    // ── FlashInfer attention Impls ───────────────────────────────

    /// Build a minimal `TargetProfile` with only the cost rows we
    /// want, for testing Impl gating without committing a CSV.
    fn synthetic_profile(rows: &[(&str, u32, u32, u32, f64)]) -> crate::target::TargetProfile {
        let mut cost_table = crate::target::CostTable::new();
        for (k, m, n, k_, cost) in rows {
            cost_table.insert(*k, *m, *n, *k_, *cost);
        }
        crate::target::TargetProfile {
            name: "synthetic".to_string(),
            source_path: std::path::PathBuf::from("synthetic"),
            compute_capability: 89,
            num_sms: 58,
            peak_tflops_fp16: 121.0,
            memory_bandwidth_gbps: 300.0,
            shared_memory_per_sm_kb: 100,
            cost_table,
        }
    }

    #[test]
    fn starter_library_registers_twelve_flashinfer_variants() {
        // Six tuples in `FLASHINFER_CONFIG_SET` × {Decode, Prefill} Impls.
        // If this count drifts, either the config set or the registration
        // loop changed without the mirror being updated — a silent way to
        // leave the Impl chain broken.
        let lib = starter_library();
        let fi_decode = lib
            .iter_enumerated()
            .filter(|(_, i)| i.name() == "flashinfer_attention_decode")
            .count();
        let fi_prefill = lib
            .iter_enumerated()
            .filter(|(_, i)| i.name() == "flashinfer_attention_prefill")
            .count();
        assert_eq!(fi_decode, 6, "expected 6 decode variants");
        assert_eq!(fi_prefill, 6, "expected 6 prefill variants");
    }

    #[test]
    fn flashinfer_decode_workload_constraint_is_m_eq_1_and_calibrated_sk_range() {
        let imp = FlashInferAttentionDecodeImpl {
            head_dim: 128,
            use_logits_soft_cap: false,
        };
        let c = imp.workload_constraint();
        assert!(c.accepts(1, 128)); // decode at lowest bucket
        assert!(c.accepts(1, 8192)); // decode at highest bucket
        assert!(!c.accepts(1, 64)); // below calibrated range
        assert!(!c.accepts(1, 16384)); // above calibrated range
        assert!(!c.accepts(2, 2048)); // prefill M outside decode range
    }

    #[test]
    fn flashinfer_prefill_workload_constraint_is_m_ge_2_and_calibrated_sk_range() {
        let imp = FlashInferAttentionPrefillImpl {
            head_dim: 64,
            use_logits_soft_cap: true,
        };
        let c = imp.workload_constraint();
        assert!(c.accepts(2, 128));
        assert!(c.accepts(4096, 8192));
        assert!(!c.accepts(1, 2048)); // decode M not in prefill range
        assert!(!c.accepts(2, 64)); // below calibrated sk
    }

    #[test]
    fn flashinfer_target_compatible_gated_by_csv_prefix() {
        // Empty profile — no rows of any kind.
        let empty = synthetic_profile(&[]);
        let imp = FlashInferAttentionDecodeImpl {
            head_dim: 128,
            use_logits_soft_cap: false,
        };
        assert!(
            !imp.target_compatible(&empty),
            "FI decode must be rejected when CSV has no matching rows"
        );

        // Profile with a row for a DIFFERENT head_dim — still must reject.
        let other = synthetic_profile(&[("flashinfer_attn_bf16_h64_nosoftcap", 1, 2048, 64, 5.0)]);
        assert!(
            !imp.target_compatible(&other),
            "FI decode (h=128) must be rejected when only h=64 rows exist"
        );

        // Profile with a matching row — accept. The kernel name is
        // keyed only on (head_dim, softcap) — `(q, k)` is runtime.
        let matching =
            synthetic_profile(&[("flashinfer_attn_bf16_h128_nosoftcap", 1, 2048, 128, 3.0)]);
        assert!(
            imp.target_compatible(&matching),
            "FI decode must accept when a matching-head_dim row exists"
        );

        // Softcap-matching requires exact softcap token.
        let imp_cap = FlashInferAttentionDecodeImpl {
            head_dim: 128,
            use_logits_soft_cap: true,
        };
        assert!(
            !imp_cap.target_compatible(&matching),
            "FI decode(softcap=true) must not accept a nosoftcap-only CSV",
        );
    }

    #[test]
    fn flashinfer_cost_us_finite_on_hit_and_uncalibrated_on_mismatch() {
        use crate::fuf::{Fuf, FufNode};
        use crate::shape::Dim;

        // Synthetic profile with one FlashInfer decode row at (m=1,
        // sk=2048, head_dim=128). Name is head_dim/softcap-only — the
        // kernel handles any (q, k) at runtime.
        let profile =
            synthetic_profile(&[("flashinfer_attn_bf16_h128_nosoftcap", 1, 2048, 128, 6.5)]);

        // Minimal FUF: one Attention tile. `fi_cost_us` ignores the
        // tile's shape entirely (the cost comes from the CSV lookup),
        // so we don't need realistic QKV tile plumbing.
        let t0 = TileId(0);
        let fuf = Fuf {
            nodes: vec![FufNode {
                id: t0,
                op: OpKind::Attention,
                inputs: vec![],
                outputs: vec![vec![Dim::Lit(1), Dim::Lit(128)]],
            }],
        };
        let mi = MatchInfo {
            claimed_tiles: vec![t0],
            boundary_inputs: vec![],
            boundary_outputs: vec![t0],
        };

        // Matching model bounds (h=128, q=32, k=8) → hit.
        let mut bounds: BTreeMap<String, u64> = BTreeMap::new();
        bounds.insert("head_dim".into(), 128);
        bounds.insert("num_attention_heads".into(), 32);
        bounds.insert("num_key_value_heads".into(), 8);
        bounds.insert("num_tokens".into(), 1);
        bounds.insert("sk_bucket".into(), 2048);
        let ctx = CostCtx {
            fuf: &fuf,
            profile: &profile,
            bounds: &bounds,
        };
        let imp = FlashInferAttentionDecodeImpl {
            head_dim: 128,
            use_logits_soft_cap: false,
        };
        let cost = imp.cost_us(&mi, &ctx);
        assert_eq!(cost, 6.5);

        // Mismatching head_dim — Impl baked for h=128, model has h=64.
        // All reject paths must stay finite (UNCALIBRATED_COST_US) so
        // the DP's `!cost.is_finite()` guard doesn't trip; FA2 with a
        // real calibrated cost still wins the tiebreak.
        bounds.insert("head_dim".into(), 64);
        let ctx_bad = CostCtx {
            fuf: &fuf,
            profile: &profile,
            bounds: &bounds,
        };
        let cost_bad = imp.cost_us(&mi, &ctx_bad);
        assert!(cost_bad.is_finite());
        assert_eq!(cost_bad, UNCALIBRATED_COST_US);

        // Different (q, k) combo at the SAME head_dim — the kernel
        // handles GQA ratio at runtime, so the same calibrated row
        // serves every model at h=128. This is what makes FI general
        // across models (Llama q32/k8, Qwen q28/k4, etc.).
        bounds.insert("head_dim".into(), 128);
        bounds.insert("num_attention_heads".into(), 28);
        bounds.insert("num_key_value_heads".into(), 4);
        let ctx_other_qk = CostCtx {
            fuf: &fuf,
            profile: &profile,
            bounds: &bounds,
        };
        let cost_other_qk = imp.cost_us(&mi, &ctx_other_qk);
        assert_eq!(cost_other_qk, 6.5);
    }
}
