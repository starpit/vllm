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

use crate::classified::{OpKind, Program, WeightId};
use crate::emit::{EmitCtx, weight_field_name};
use crate::fuf::{Fuf, FufInput, TileId};
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
/// workload itself — e.g. "this GEMV kernel only handles M=1".
///
/// Correctness, not cost: if `accepts(num_tokens)` returns false,
/// the impl must NOT be picked at that workload regardless of its
/// cost. Data, not closures — so the ILP backend can linearize
/// each variant.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum WorkloadConstraint {
    /// Valid for any `num_tokens` value.
    Any,
    /// Valid only when `num_tokens` falls in this inclusive range.
    NumTokensRange { min: u32, max: u32 },
}

impl WorkloadConstraint {
    pub fn accepts(&self, num_tokens: u32) -> bool {
        match self {
            Self::Any => true,
            Self::NumTokensRange { min, max } => num_tokens >= *min && num_tokens <= *max,
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
            if let FufInput::Weight { id, index } = input {
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
    /// neighbors of `seed`, and decides whether the local
    /// structure matches the impl's expected pattern.
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
                single_tile_match(fuf, seed, $op)
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
    // kernels::rms_norm(input: GpuTensor, weight: GpuTensor,
    // eps: f32, alloc, stream) -> OwnedTensor. Input is a
    // TensorView we deref to GpuTensor; weight is `&RmsNorm`
    // (fields .weight: GpuTensor, .eps: f32).
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

fn emit_gemm(ctx: &EmitCtx) -> TokenStream {
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
    let flops = 2.0 * m_dim as f64 * n_dim as f64 * k_dim as f64;
    let peak = ctx.profile.peak_tflops_fp16 * 1e12;
    if peak == 0.0 || flops == 0.0 {
        0.0
    } else {
        (flops / peak) * 1e6
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
    GemmRefImpl,
    OpKind::Gemm,
    "gemm_ref",
    cost_gemm,
    emit_gemm,
    true
);
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
    lib.push(Box::new(GemmRefImpl));
    lib.push(Box::new(AttentionViaCacheImpl));
    // Multi-tile fusions. The solver's claim-size-DESC sort picks
    // these over singleton coverage when both apply; the singletons
    // stay as fallbacks for tile positions the fusion doesn't match
    // (e.g. the first layer's input_layernorm, whose upstream is
    // `embed` not `Add`, stays a singleton RmsNorm claim).
    lib.push(Box::new(FusedGateUpSiluMulImpl));
    lib.push(Box::new(FusedAddRmsNormImpl));
    // Decode / prefill QKV+rope variants — the solver picks via
    // WorkloadConstraint (M=1 → Cache, M≥2 → Prefill).
    lib.push(Box::new(FusedQkvRopeCacheImpl));
    lib.push(Box::new(FusedQkvRopePrefillImpl));
    // Matching attention pair — decode reads from cache, prefill
    // reads the contiguous K/V produced by the prefill QKV impl.
    lib.push(Box::new(AttentionPrefillContiguousImpl));
    lib
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
        FufInput::Weight { id, index } => Some((*id, *index)),
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
        let gate_up_ident = quote::format_ident!("__fused_gate_up_{}", mul_id.0);
        let intermediate = ctx.bound("intermediate_size") as usize;

        quote! {
            // Fused gate+up GEMM producing a [num_tokens, 2*intermediate]
            // buffer. The user's WeightBundle returns a LinearLayer
            // whose weight is the vertically-concatenated
            // [gate_proj | up_proj].
            let #gate_up_ident = unsafe {
                (#weight_expr).forward(
                    #activation,
                    &mut device.cublas,
                    &mut device.caching,
                    device.compute_stream,
                )
            };
            // SiLU(gate) * up, reading the packed buffer and writing
            // a [num_tokens, intermediate] output.
            let #mul_out = unsafe {
                ::ferrite_kernels::kernels::silu_and_mul_fused(
                    *#gate_up_ident,
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
                FufInput::Weight { id, index } => Some((*id, *index)),
                _ => None,
            })
            .expect("RmsNorm has a weight input");
        let weight_name = weight_field_name(ctx.program, weight_id, weight_idx);
        let weight_expr = ctx.weight_accessor(&weight_name);

        // Output bindings. The kernel mutates delta_upstream's buffer
        // into the normed output and residual_upstream's buffer into
        // the updated residual. Downstream tiles read these via
        // `(*ident).as_view()`; `TensorView::from_raw` (unsafe) gives
        // us the Deref-to-GpuTensor shape the emit convention wants.
        let add_out = ctx.output_ident(add_id, 0);
        let rmsnorm_out = ctx.output_ident(rmsnorm_id, 0);

        quote! {
            // Fused `residual += delta; normed = norm(residual) * w`.
            // Both buffers are mutated in place; the upstream
            // OwnedTensors (#delta_upstream / #residual_upstream)
            // remain the owners — we only alias them as TensorViews
            // for downstream reads.
            unsafe {
                let _ = ::ferrite_kernels::kernels::fused_add_rms_norm_inplace(
                    *#delta_upstream,
                    *#residual_upstream,
                    (#weight_expr).weight,
                    (#weight_expr).eps,
                    device.compute_stream,
                );
            }
            // Aliases: #rmsnorm_out points at the (now-normed)
            // delta buffer; #add_out points at the (now-updated)
            // residual buffer. `.as_view()` on a dereffed upstream
            // works for both OwnedTensor and TensorView bindings
            // (both Deref to GpuTensor, and `GpuTensor::as_view` is
            // the uniform conversion).
            let #rmsnorm_out = unsafe { (*#delta_upstream).as_view() };
            let #add_out = unsafe { (*#residual_upstream).as_view() };
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

        // Find a RopeAppend whose first three Tile inputs all point
        // at Gemm tiles sharing a common activation. The seed must
        // be one of those three Gemms.
        let rope_node = fuf.nodes.iter().find(|n| {
            if n.op != OpKind::RopeAppend || n.inputs.len() < 3 {
                return false;
            }
            // First three inputs must all be Tile references.
            let qkv: Vec<TileId> = n
                .inputs
                .iter()
                .take(3)
                .filter_map(|i| match i {
                    FufInput::Tile { id, .. } => Some(*id),
                    _ => None,
                })
                .collect();
            if qkv.len() != 3 {
                return false;
            }
            if !qkv.contains(&seed) {
                return false;
            }
            // All three must be Gemms.
            if qkv.iter().any(|t| fuf.get(*t).op != OpKind::Gemm) {
                return false;
            }
            // All three must share the same activation (first Tile input).
            let act = first_tile_input(fuf.get(qkv[0]));
            act.is_some() && qkv.iter().all(|t| first_tile_input(fuf.get(*t)) == act)
        })?;
        let rope_id = rope_node.id;

        // Extract the three Gemm tile ids.
        let qkv_ids: Vec<TileId> = rope_node
            .inputs
            .iter()
            .take(3)
            .filter_map(|i| match i {
                FufInput::Tile { id, .. } => Some(*id),
                _ => None,
            })
            .collect();

        let mut claimed: Vec<TileId> = qkv_ids
            .iter()
            .copied()
            .chain(std::iter::once(rope_id))
            .collect();
        claimed.sort();

        let activation = first_tile_input(fuf.get(qkv_ids[0]))?.0;
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

    fn required_weights(
        &self,
        claimed_tiles: &[TileId],
        fuf: &Fuf,
        program: &Program,
    ) -> Vec<WeightAccessor> {
        // Collect the three Gemms' weight refs → one fused accessor.
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
        // Identify the RopeAppend tile and the three Gemms by op kind.
        let rope_id = *ctx
            .claimed_tiles
            .iter()
            .find(|t| ctx.fuf.get(**t).op == OpKind::RopeAppend)
            .expect("claim must contain a RopeAppend");
        let rope_node = ctx.fuf.get(rope_id);

        // The three Gemms are rope_node.inputs[0..3]'s tile ids.
        let qkv_ids: Vec<TileId> = rope_node
            .inputs
            .iter()
            .take(3)
            .filter_map(|i| match i {
                FufInput::Tile { id, .. } => Some(*id),
                _ => None,
            })
            .collect();
        assert_eq!(qkv_ids.len(), 3, "rope has three tile inputs (q, k, v)");
        // Use first Gemm (q_gemm) as the "representative" for activation.
        let gate_gemm_id = qkv_ids[0];
        let activation = ctx.input_expr(gate_gemm_id, 0);

        // Fused weight accessor — name reconstructed from claim.
        let qkv_weights: Vec<(WeightId, Option<u64>)> = qkv_ids
            .iter()
            .map(|t| first_weight_ref(ctx.fuf.get(*t)).expect("gemm has a weight"))
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
        // Downstream Attention (currently still a fake singleton — C4
        // territory) will read k_out/v_out via `input_expr`, so bind
        // them to layer-specific `TensorView` aliases on the paged
        // cache. After C4 lands, Attention will ignore its K/V slots
        // and those bindings become unused.
        let q_out = ctx.output_ident(rope_id, 0);
        let k_out = ctx.output_ident(rope_id, 1);
        let v_out = ctx.output_ident(rope_id, 2);
        let qkv_packed_ident = quote::format_ident!("__fused_qkv_{}", rope_id.0);

        quote! {
            // Fused QKV GEMM → packed [num_tokens, q + 2*kv] tensor.
            let #qkv_packed_ident = unsafe {
                (#weight_expr).forward(
                    #activation,
                    &mut device.cublas,
                    &mut device.caching,
                    device.compute_stream,
                )
            };
            // Fused RoPE + paged-cache write. Reads packed QKV, writes
            // K and V into the layer's slices of the paged cache, and
            // returns Q (rotated) as an OwnedTensor.
            let #q_out = unsafe {
                ::ferrite_kernels::kernels::fused_qkv_rope_cache(
                    *#qkv_packed_ident,
                    *ctx.positions,
                    ctx.rotary.cos_sin_cache,
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

        // Softmax scale: 1.0 / sqrt(head_dim). Emit at compile time.
        let head_dim = ctx.bound("head_dim") as f32;
        let scale: f32 = 1.0 / head_dim.sqrt();
        let scale_tokens = quote! { #scale };

        quote! {
            let #out = unsafe {
                ::ferrite_kernels::kernels::flash_attn_paged(
                    *#q_expr,
                    *ctx.kv_cache.k_cache(#layer),
                    *ctx.kv_cache.v_cache(#layer),
                    *ctx.cu_seqlens_q,
                    *ctx.seqused_k,
                    *ctx.block_table,
                    ctx.max_seqlen_q,
                    ctx.max_seqlen_k,
                    #scale_tokens,
                    true, // is_causal — always true for decoder-only LMs
                    ctx.kv_cache.block_size,
                    device.num_sm,
                    &mut device.caching,
                    device.compute_stream,
                )
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
            .find(|t| ctx.fuf.get(**t).op == OpKind::RopeAppend)
            .expect("claim must contain a RopeAppend");
        let rope_node = ctx.fuf.get(rope_id);

        let qkv_ids: Vec<TileId> = rope_node
            .inputs
            .iter()
            .take(3)
            .filter_map(|i| match i {
                FufInput::Tile { id, .. } => Some(*id),
                _ => None,
            })
            .collect();
        assert_eq!(qkv_ids.len(), 3, "rope has three tile inputs (q, k, v)");
        let activation = ctx.input_expr(qkv_ids[0], 0);

        let qkv_weights: Vec<(WeightId, Option<u64>)> = qkv_ids
            .iter()
            .map(|t| first_weight_ref(ctx.fuf.get(*t)).expect("gemm has a weight"))
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
        let qkv_packed_ident = quote::format_ident!("__fused_qkv_{}", rope_id.0);

        quote! {
            // Fused QKV GEMM → packed [num_tokens, q + 2*kv] tensor.
            let #qkv_packed_ident = unsafe {
                (#weight_expr).forward(
                    #activation,
                    &mut device.cublas,
                    &mut device.caching,
                    device.compute_stream,
                )
            };
            // RoPE-split: rotate Q and K, return all three as
            // contiguous OwnedTensors. Unlike `fused_qkv_rope_cache`
            // this does NOT write to the paged cache — the cache
            // write is a separate step below so the K/V OwnedTensors
            // remain available for contiguous prefill attention.
            let (#q_out, #k_out, #v_out) = unsafe {
                ::ferrite_kernels::kernels::fused_qkv_rope(
                    *#qkv_packed_ident,
                    *ctx.positions,
                    ctx.rotary.cos_sin_cache,
                    #q_size,
                    #kv_size,
                    #num_q_heads,
                    #num_kv_heads,
                    #head_dim,
                    &mut device.caching,
                    device.compute_stream,
                )
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
        // Same quadratic-in-seqlen cost as the paged decode variant
        // for now — both call FA2 underneath.
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

        // Q, K, V from the DSL's `attention(q, k, v, ...)` — slots
        // 0/1/2 are the upstream rope-append tile's three outputs.
        // The prefill QKV impl binds these to real contiguous
        // OwnedTensors (unlike the decode variant which aliases
        // K/V to the paged cache).
        let q_expr = ctx.input_expr(tile, 0);
        let k_expr = ctx.input_expr(tile, 1);
        let v_expr = ctx.input_expr(tile, 2);

        // Softmax scale: 1/sqrt(head_dim), baked in.
        let head_dim = ctx.bound("head_dim") as f32;
        let scale: f32 = 1.0 / head_dim.sqrt();
        let scale_tokens = quote! { #scale };

        quote! {
            let #out = unsafe {
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
                    0.0,  // softcap
                    -1,   // window_size_left (-1 = disabled)
                    &mut device.caching,
                    device.compute_stream,
                    ::std::ptr::null::<u8>(),
                    0,     // rotary_dim
                    false, // is_rotary_interleaved
                )
            };
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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
        let c = WorkloadConstraint::NumTokensRange { min: 1, max: 8 };
        assert!(c.accepts(1));
        assert!(c.accepts(8));
        assert!(!c.accepts(9));
        assert!(!c.accepts(0));
        assert!(WorkloadConstraint::Any.accepts(4096));
    }
}
