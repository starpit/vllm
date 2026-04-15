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

use std::collections::BTreeMap;
use std::fmt;

use proc_macro2::TokenStream;
use quote::quote;

use crate::classified::OpKind;
use crate::emit::EmitCtx;
use crate::fuf::{Fuf, TileId};
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

fn emit_rope_append(ctx: &EmitCtx) -> TokenStream {
    let tile = ctx.primary();
    let q_out = ctx.output_ident(tile, 0);
    let k_out = ctx.output_ident(tile, 1);
    let v_out = ctx.output_ident(tile, 2);
    let ins = ctx.all_input_exprs(tile);
    let (q, k, v, pos, rotary, kv) = (&ins[0], &ins[1], &ins[2], &ins[3], &ins[4], &ins[5]);
    quote! {
        let (#q_out, #k_out, #v_out) = unsafe {
            ::ferrite_kernels::kernels::rope_append_kv(
                #q, #k, #v, #pos, #rotary, #kv,
                ctx.slot_mapping,
                &mut device.caching,
                device.compute_stream,
            )
        };
    }
}

fn emit_attention(ctx: &EmitCtx) -> TokenStream {
    let tile = ctx.primary();
    let out = ctx.output_ident(tile, 0);
    let ins = ctx.all_input_exprs(tile);
    let (q, k, v, kv, bt) = (&ins[0], &ins[1], &ins[2], &ins[3], &ins[4]);
    quote! {
        let #out = unsafe {
            ::ferrite_kernels::kernels::flash_attention(
                #q, #k, #v, #kv, #bt,
                ctx.cu_seqlens_q,
                ctx.seqused_k,
                ctx.max_seqlen_q,
                ctx.max_seqlen_k,
                &mut device.caching,
                device.compute_stream,
            )
        };
    }
}

fn emit_silu(ctx: &EmitCtx) -> TokenStream {
    let tile = ctx.primary();
    let out = ctx.output_ident(tile, 0);
    let x = ctx.input_expr(tile, 0);
    quote! {
        let #out = unsafe {
            ::ferrite_kernels::kernels::silu_owned(
                #x,
                &mut device.caching,
                device.compute_stream,
            )
        };
    }
}

fn emit_add(ctx: &EmitCtx) -> TokenStream {
    let tile = ctx.primary();
    let out = ctx.output_ident(tile, 0);
    let a = ctx.input_expr(tile, 0);
    let b = ctx.input_expr(tile, 1);
    quote! {
        let #out = unsafe {
            ::ferrite_kernels::kernels::add_owned(
                #a, #b,
                &mut device.caching,
                device.compute_stream,
            )
        };
    }
}

fn emit_mul(ctx: &EmitCtx) -> TokenStream {
    let tile = ctx.primary();
    let out = ctx.output_ident(tile, 0);
    let a = ctx.input_expr(tile, 0);
    let b = ctx.input_expr(tile, 1);
    quote! {
        let #out = unsafe {
            ::ferrite_kernels::kernels::mul_owned(
                #a, #b,
                &mut device.caching,
                device.compute_stream,
            )
        };
    }
}

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
trivial_impl!(
    RopeAppendRefImpl,
    OpKind::RopeAppend,
    "rope_append_ref",
    elementwise_cost,
    emit_rope_append,
    false
);
trivial_impl!(
    AttentionRefImpl,
    OpKind::Attention,
    "attention_ref",
    cost_attention,
    emit_attention,
    true
);
trivial_impl!(
    SiluRefImpl,
    OpKind::Silu,
    "silu_ref",
    elementwise_cost,
    emit_silu,
    false
);
trivial_impl!(
    AddRefImpl,
    OpKind::Add,
    "add_ref",
    elementwise_cost,
    emit_add,
    false
);
trivial_impl!(
    MulRefImpl,
    OpKind::Mul,
    "mul_ref",
    elementwise_cost,
    emit_mul,
    false
);

/// Baseline library: one HostCallback impl per OpKind, analytical
/// cost estimates. Replaced by calibrated target-specific impls
/// as they're ported.
pub fn starter_library() -> ImplementationLibrary {
    let mut lib = ImplementationLibrary::new();
    lib.push(Box::new(EmbedRefImpl));
    lib.push(Box::new(RmsNormRefImpl));
    lib.push(Box::new(GemmRefImpl));
    lib.push(Box::new(RopeAppendRefImpl));
    lib.push(Box::new(AttentionRefImpl));
    lib.push(Box::new(SiluRefImpl));
    lib.push(Box::new(AddRefImpl));
    lib.push(Box::new(MulRefImpl));
    lib
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
