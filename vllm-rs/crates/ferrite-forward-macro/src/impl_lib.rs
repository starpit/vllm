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

/// One phase inside a megakernel `.cu` file. Each DeviceCallable
/// impl that participates in a megakernel provides this via
/// [`Implementation::device_phase`].
///
/// The generated `.cu` has three sections per phase:
/// 1. `flat_params` — the `extern "C"` launch wrapper's signature
/// 2. `kernel_body` — CUDA lines inside the `__global__` kernel
/// 3. `params_build` — launch wrapper lines copying flat→internal
/// 4. `internal_fields` — fields in the internal params struct
///
/// Phase index `idx` is assigned by codegen (0, 1, 2, …). Param
/// names use the `p{idx}_` prefix so they don't collide across phases.
#[derive(Clone, Debug)]
pub struct DevicePhase {
    /// `(c_type, name)` pairs for the `extern "C"` flat param list.
    pub flat_params: Vec<(String, String)>,
    /// Lines inside the `__global__` kernel body for this phase.
    pub kernel_body: Vec<String>,
    /// Lines in the launch wrapper that copy flat args into the
    /// internal params struct.
    pub params_build: Vec<String>,
    /// Field declarations inside the internal params struct.
    pub internal_fields: Vec<String>,
    /// Lines emitted before the params struct (includes, typedefs).
    /// Used by CUTLASS GEMM to emit `using DeviceGemm_p{idx} = ...`.
    pub preamble: Vec<String>,
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

    /// Device-phase descriptor for megakernel `.cu` generation.
    ///
    /// Returns `Some((phase, preamble_stmts, param_values))` for
    /// DeviceCallable impls that can participate in a megakernel.
    ///
    /// - `preamble_stmts`: `let` bindings emitted in the shared scope
    ///   BEFORE param assignments (output idents, shared temporaries).
    /// - `param_values`: expressions parallel to `DevicePhase::flat_params`
    ///   that evaluate to each param's value.
    ///
    /// Returns `None` for HostCallable impls (the default).
    fn device_phase(
        &self,
        _idx: usize,
        _ctx: &EmitCtx,
    ) -> Option<(DevicePhase, Vec<TokenStream>, Vec<TokenStream>)> {
        None
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
    lib.push(Box::new(FusedGateUpGeluMulImpl));
    lib.push(Box::new(FusedAddRmsNormImpl));
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

    // ── TK (ThunderKittens) natively DeviceCallable ops (sm90+) ──
    // Each TK impl runs as a phase inside the megakernel cooperative
    // kernel using kittens warp::mma / register tile types. Cost is
    // cuBLAS-calibrated minus launch_overhead → solver always prefers
    // TK DC over standalone cuBLAS, enabling 100% megakernel.
    lib.push(Box::new(TkGemmImpl));
    lib.push(Box::new(TkGemvImpl));
    lib.push(Box::new(TkAttentionDecodeImpl));
    lib.push(Box::new(TkAttentionPrefillImpl));

    // ── DeviceCallable wrappers for elementwise ops (sm89+) ──
    // Each DC wrapper saves one kernel launch by running inside an
    // enclosing megakernel. The DP sees this as a cost discount
    // (launch_overhead_us subtracted from per-call cost).
    lib.push(Box::new(DeviceCallableWrapper::new(Box::new(
        RmsNormRefImpl,
    ))));
    lib.push(Box::new(DeviceCallableWrapper::new(Box::new(
        FusedAddRmsNormImpl,
    ))));
    lib.push(Box::new(DeviceCallableWrapper::new(Box::new(
        FusedAddRmsNormWithOffsetImpl,
    ))));
    lib.push(Box::new(DeviceCallableWrapper::new(Box::new(
        ScalarOffsetRmsNormImpl,
    ))));
    lib.push(Box::new(DeviceCallableWrapper::new(Box::new(
        ScalarMulImpl,
    ))));
    lib.push(Box::new(DeviceCallableWrapper::new(Box::new(
        TanhSoftCapImpl,
    ))));
    // DC fused ops — host impls use cuBLAS, but the DC version's
    // device_phase() emits CUTLASS device-side GEMMs for megakernel
    // embedding. The DeviceCallableWrapper just changes launch_kind
    // and adjusts cost; the .cu codegen reads device_phase().
    lib.push(Box::new(DeviceCallableWrapper::new(Box::new(
        FusedGateUpSiluMulImpl,
    ))));
    lib.push(Box::new(DeviceCallableWrapper::new(Box::new(
        FusedGateUpGeluMulImpl,
    ))));
    lib.push(Box::new(DeviceCallableWrapper::new(Box::new(
        FusedQkvRopeCacheImpl,
    ))));
    lib.push(Box::new(DeviceCallableWrapper::new(Box::new(
        FusedQkvRopePrefillImpl,
    ))));
    // DC CUTLASS GEMM — every tile variant, embeddable in megakernels.
    // CUTLASS GemmUniversal::invoke() is a __device__ function.
    for tile in CUTLASS_TILE_ZOO {
        lib.push(Box::new(DeviceCallableWrapper::new(Box::new(
            CutlassGemmImpl {
                tile_m: tile.0,
                tile_n: tile.1,
                stages: tile.2,
            },
        ))));
    }
    // DC CUTLASS GEMV for BS=1 decode.
    lib.push(Box::new(DeviceCallableWrapper::new(Box::new(
        CutlassGemvImpl,
    ))));

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

// ── FusedGateUpGeluMulImpl ───────────────────────────────────────
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
                FufInput::Weight { id, index } => Some((*id, *index)),
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
                FufInput::Weight { id, index } => Some((*id, *index)),
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
                FufInput::Weight { id, index } => Some((*id, *index)),
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
                FufInput::Weight { id, index } => Some((*id, *index)),
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
                FufInput::Weight { id, index } => Some((*id, *index)),
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
        // Downstream Attention reads K/V from the paged cache directly,
        // so the k_out/v_out bindings are paged-cache `TensorView`
        // aliases — no allocation.
        let q_out = ctx.output_ident(rope_id, 0);
        let k_out = ctx.output_ident(rope_id, 1);
        let v_out = ctx.output_ident(rope_id, 2);

        quote! {
            // Fused QKV GEMM → packed [num_tokens, q + 2*kv] tensor,
            // then fused RoPE + paged-cache write. The packed QKV
            // buffer drops as soon as the rope-cache kernel returns.
            //
            // FP8 KV-cache path branches at runtime: the cache dtype
            // is a per-model property known only at load time.
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
                        ctx.rotary.cos_sin_cache,
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
                }
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

        // Softmax scale: config-driven. Gemma2 sets
        // `query_pre_attn_scalar` (softmax scale = that^-0.5); Llama/
        // Qwen2 have no such field, so we fall back to the standard
        // `1/sqrt(head_dim)`. Both computed at emit time; the emitted
        // tokens are a plain `f32` literal either way.
        let scale_tokens = attention_scale_tokens(ctx);
        let softcap_tokens = attention_softcap_tokens(ctx);
        // q_size for the [num_tokens, q_size] reshape o_proj needs.
        let q_size = (ctx.bound("num_attention_heads") * ctx.bound("head_dim")) as usize;

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
                        ctx.rotary.cos_sin_cache.raw_ptr() as *const u8,
                        ctx.rotary.cos_sin_cache.dim(1),
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
                    false, // is_rotary_interleaved
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
                let qkv_packed = (#weight_expr).forward(
                    #activation,
                    &mut device.cublas,
                    &mut device.caching,
                    device.compute_stream,
                );
                ::ferrite_kernels::kernels::fused_qkv_rope(
                    *qkv_packed,
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
        let q_expr = ctx.input_expr(tile, 0);
        let k_expr = ctx.input_expr(tile, 1);
        let v_expr = ctx.input_expr(tile, 2);

        // Config-driven attention scale + softcap (see
        // `AttentionViaCacheImpl::emit_call` for the rationale).
        let scale_tokens = attention_scale_tokens(ctx);
        let softcap_tokens = attention_softcap_tokens(ctx);
        let q_size = (ctx.bound("num_attention_heads") * ctx.bound("head_dim")) as usize;

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
                    false, // is_rotary_interleaved
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

        quote! {
            let mut #out = unsafe {
                let has_spans = !ctx.kv_cache.block_unrotated_gpu().is_null();
                let (cos_sin_ptr, rotary_dim) = if has_spans {
                    (
                        ctx.rotary.cos_sin_cache.raw_ptr() as *const u8,
                        ctx.rotary.cos_sin_cache.dim(1),
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
                    false, // is_rotary_interleaved
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

        let q_expr = ctx.input_expr(tile, 0);
        let k_expr = ctx.input_expr(tile, 1);
        let v_expr = ctx.input_expr(tile, 2);

        let scale_tokens = attention_scale_tokens(ctx);
        let softcap_tokens = attention_softcap_tokens(ctx);
        let window_tokens = sliding_window_left_tokens(ctx);
        let q_size = (ctx.bound("num_attention_heads") * ctx.bound("head_dim")) as usize;

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
                    false, // is_rotary_interleaved
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

// ── TK attention (sm90+ DeviceCallable) ─────────────────────────
//
// ThunderKittens-native attention for Hopper. Runs as a
// DeviceCallable inside a megakernel using wgmma for Q×K^T and
// attn×V, with online softmax. Same matching and emission as the
// existing FlashAttn-based impls — the compiler picks TK when its
// cost is lower (CSV `tk_attention_*` rows); the runtime kernel
// dispatch is unchanged.

#[derive(Debug)]
pub struct TkAttentionDecodeImpl;

impl Implementation for TkAttentionDecodeImpl {
    fn name(&self) -> &'static str {
        "tk_attention_decode"
    }

    fn target_compatible(&self, profile: &TargetProfile) -> bool {
        profile.compute_capability >= 90
    }

    fn workload_constraint(&self) -> WorkloadConstraint {
        WorkloadConstraint::NumTokensRange { min: 1, max: 1 }
    }

    fn matches(&self, fuf: &Fuf, seed: TileId, _profile: &TargetProfile) -> Option<MatchInfo> {
        single_tile_match(fuf, seed, OpKind::Attention)
    }

    fn cost_us(&self, m: &MatchInfo, ctx: &CostCtx) -> f64 {
        // wgmma-based attention is faster than FA2 on Hopper.
        // Apply a 10% discount on the analytical estimate; real
        // measured CSV data will override when available.
        cost_attention(m, ctx) * 0.9
    }

    fn resources(&self, _m: &MatchInfo) -> Resources {
        Resources {
            shmem_bytes: 228 * 1024,
            regs_per_thread: 232,
            threads_per_cta: 128,
        }
    }

    fn launch_kind(&self) -> LaunchKind {
        LaunchKind::DeviceCallable
    }

    fn supported_input_handoffs(&self) -> &[Handoff] {
        const H: &[Handoff] = &[Handoff::Mbarrier, Handoff::Internal, Handoff::StreamOrder];
        H
    }

    fn supported_output_handoffs(&self) -> &[Handoff] {
        const H: &[Handoff] = &[Handoff::Mbarrier, Handoff::Internal, Handoff::StreamOrder];
        H
    }

    fn input_layouts(&self, m: &MatchInfo) -> Vec<Layout> {
        vec![Layout::Any; m.boundary_inputs.len()]
    }

    fn output_layouts(&self, _m: &MatchInfo) -> Vec<Layout> {
        vec![Layout::RowMajorBf16]
    }

    fn is_compute_bound(&self) -> bool {
        true
    }

    fn can_share_kernel_with(&self, _other: &dyn Implementation) -> bool {
        true
    }

    fn emit_call(&self, ctx: &EmitCtx) -> TokenStream {
        // Reuse the same emission as AttentionViaCacheImpl — the
        // runtime kernel dispatch handles TK vs FA2 selection.
        AttentionViaCacheImpl.emit_call(ctx)
    }
}

#[derive(Debug)]
pub struct TkAttentionPrefillImpl;

impl Implementation for TkAttentionPrefillImpl {
    fn name(&self) -> &'static str {
        "tk_attention_prefill"
    }

    fn target_compatible(&self, profile: &TargetProfile) -> bool {
        profile.compute_capability >= 90
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
        cost_attention(m, ctx) * 0.9
    }

    fn resources(&self, _m: &MatchInfo) -> Resources {
        Resources {
            shmem_bytes: 228 * 1024,
            regs_per_thread: 232,
            threads_per_cta: 128,
        }
    }

    fn launch_kind(&self) -> LaunchKind {
        LaunchKind::DeviceCallable
    }

    fn supported_input_handoffs(&self) -> &[Handoff] {
        const H: &[Handoff] = &[Handoff::Mbarrier, Handoff::Internal, Handoff::StreamOrder];
        H
    }

    fn supported_output_handoffs(&self) -> &[Handoff] {
        const H: &[Handoff] = &[Handoff::Mbarrier, Handoff::Internal, Handoff::StreamOrder];
        H
    }

    fn input_layouts(&self, m: &MatchInfo) -> Vec<Layout> {
        vec![Layout::Any; m.boundary_inputs.len()]
    }

    fn output_layouts(&self, _m: &MatchInfo) -> Vec<Layout> {
        vec![Layout::RowMajorBf16]
    }

    fn is_compute_bound(&self) -> bool {
        true
    }

    fn can_share_kernel_with(&self, _other: &dyn Implementation) -> bool {
        true
    }

    fn emit_call(&self, ctx: &EmitCtx) -> TokenStream {
        AttentionPrefillContiguousImpl.emit_call(ctx)
    }
}

// ── DeviceCallableWrapper ────────────────────────────────────────
//
// Generic wrapper that makes any existing Implementation run as a
// DeviceCallable inside a megakernel. Delegates everything to the
// inner impl except launch_kind (DeviceCallable), handoffs
// (Mbarrier/SyncThreads), can_share_kernel_with (true), and
// cost_us (discounted by launch_overhead_us).

#[derive(Debug)]
pub struct DeviceCallableWrapper {
    inner: Box<dyn Implementation>,
    name: &'static str,
}

impl DeviceCallableWrapper {
    pub fn new(inner: Box<dyn Implementation>) -> Self {
        let name: &'static str = Box::leak(format!("dc_{}", inner.name()).into_boxed_str());
        Self { inner, name }
    }
}

impl Implementation for DeviceCallableWrapper {
    fn name(&self) -> &'static str {
        self.name
    }

    fn target_compatible(&self, profile: &TargetProfile) -> bool {
        profile.compute_capability >= 89 && self.inner.target_compatible(profile)
    }

    fn workload_constraint(&self) -> WorkloadConstraint {
        self.inner.workload_constraint()
    }

    fn matches(&self, fuf: &Fuf, seed: TileId, profile: &TargetProfile) -> Option<MatchInfo> {
        self.inner.matches(fuf, seed, profile)
    }

    fn cost_us(&self, m: &MatchInfo, ctx: &CostCtx) -> f64 {
        // DC impls save one kernel launch by running inside an
        // enclosing megakernel. Subtract the launch overhead so
        // the greedy DP prefers DC over standalone for cheap ops.
        (self.inner.cost_us(m, ctx) - ctx.profile.launch_overhead_us).max(0.0)
    }

    fn resources(&self, m: &MatchInfo) -> Resources {
        self.inner.resources(m)
    }

    fn launch_kind(&self) -> LaunchKind {
        LaunchKind::DeviceCallable
    }

    fn supported_input_handoffs(&self) -> &[Handoff] {
        const H: &[Handoff] = &[
            Handoff::Mbarrier,
            Handoff::SyncThreads,
            Handoff::Internal,
            Handoff::StreamOrder,
        ];
        H
    }

    fn supported_output_handoffs(&self) -> &[Handoff] {
        const H: &[Handoff] = &[
            Handoff::Mbarrier,
            Handoff::SyncThreads,
            Handoff::Internal,
            Handoff::StreamOrder,
        ];
        H
    }

    fn input_layouts(&self, m: &MatchInfo) -> Vec<Layout> {
        self.inner.input_layouts(m)
    }

    fn output_layouts(&self, m: &MatchInfo) -> Vec<Layout> {
        self.inner.output_layouts(m)
    }

    fn is_compute_bound(&self) -> bool {
        self.inner.is_compute_bound()
    }

    fn can_share_kernel_with(&self, _other: &dyn Implementation) -> bool {
        true
    }

    fn required_weights(
        &self,
        claimed_tiles: &[TileId],
        fuf: &Fuf,
        program: &Program,
    ) -> Vec<WeightAccessor> {
        self.inner.required_weights(claimed_tiles, fuf, program)
    }

    fn output_alias(
        &self,
        claimed_tiles: &[TileId],
        fuf: &Fuf,
    ) -> Vec<((TileId, u8), Option<(TileId, u8)>)> {
        self.inner.output_alias(claimed_tiles, fuf)
    }

    fn emit_call(&self, ctx: &EmitCtx) -> TokenStream {
        self.inner.emit_call(ctx)
    }

    fn consumes_input_tiles(&self, claimed_tiles: &[TileId], fuf: &Fuf) -> Vec<(TileId, u8)> {
        self.inner.consumes_input_tiles(claimed_tiles, fuf)
    }

    fn device_phase(
        &self,
        idx: usize,
        ctx: &EmitCtx,
    ) -> Option<(DevicePhase, Vec<TokenStream>, Vec<TokenStream>)> {
        dc_device_phase(self.inner.name(), idx, ctx)
    }
}

/// Generate the CUDA-side [`DevicePhase`] and Rust-side param
/// expressions for a DC-wrapped impl, keyed by the inner impl's
/// name. The set of wrappable ops is bounded by what
/// `megakernel_ops.cuh` provides `__device__` functions for.
fn dc_device_phase(
    inner_name: &str,
    idx: usize,
    ctx: &EmitCtx,
) -> Option<(DevicePhase, Vec<TokenStream>, Vec<TokenStream>)> {
    let p = format!("p{idx}");
    // Compile-time constant: num_tokens for this workload bucket.
    // Interpolated into quote!{} as a literal, NOT as `ctx.num_tokens`
    // (ForwardCtx doesn't have num_tokens — it's a compile-time value).
    let num_tokens_val = ctx.num_tokens.expect("dc_device_phase requires num_tokens") as i32;
    match inner_name {
        "rmsnorm_ref" => {
            let tile = ctx.primary();
            let x = ctx.input_expr(tile, 0);
            let w = ctx.input_expr(tile, 1);
            let out = ctx.output_ident(tile, 0);
            let hidden_size = ctx.bound("hidden_size") as i32;
            let hs = hidden_size;
            Some((
                DevicePhase {
                    flat_params: vec![
                        ("void*".into(), format!("{p}_out")),
                        ("const void*".into(), format!("{p}_input")),
                        ("const void*".into(), format!("{p}_weight")),
                        ("float".into(), format!("{p}_eps")),
                        ("int".into(), format!("{p}_hidden_size")),
                        ("int".into(), format!("{p}_num_tokens")),
                    ],
                    kernel_body: vec![format!(
                        "dc_rms_norm<__nv_bfloat16>({p}_out, {p}_input, {p}_weight, {p}_eps, {p}_hidden_size, {p}_num_tokens, smem);"
                    )],
                    params_build: vec![
                        format!("params.{p}_out = (__nv_bfloat16*){p}_out;"),
                        format!("params.{p}_input = (const __nv_bfloat16*){p}_input;"),
                        format!("params.{p}_weight = (const __nv_bfloat16*){p}_weight;"),
                        format!("params.{p}_eps = {p}_eps;"),
                        format!("params.{p}_hidden_size = {p}_hidden_size;"),
                        format!("params.{p}_num_tokens = {p}_num_tokens;"),
                    ],
                    internal_fields: vec![
                        format!("__nv_bfloat16* {p}_out"),
                        format!("const __nv_bfloat16* {p}_input"),
                        format!("const __nv_bfloat16* {p}_weight"),
                        format!("float {p}_eps"),
                        format!("int {p}_hidden_size"),
                        format!("int {p}_num_tokens"),
                    ],
                    preamble: vec![],
                },
                // Preamble: output ident binding
                vec![quote! {
                    let #out = device.caching.alloc_tensor(
                        &[#num_tokens_val as usize, #hs as usize],
                        ::ferrite_cuda_core::DType::BF16,
                    );
                }],
                // Param values parallel to flat_params
                vec![
                    quote! { #out.raw_ptr() as *mut ::core::ffi::c_void },
                    quote! { (#x).raw_ptr() as *const ::core::ffi::c_void },
                    quote! { (#w).weight.raw_ptr() as *const ::core::ffi::c_void },
                    quote! { (#w).eps },
                    quote! { #hidden_size },
                    quote! { #num_tokens_val },
                ],
            ))
        }
        "fused_add_rms_norm" => {
            // dc_fused_add_rms_norm<T>(input, residual, weight, eps, hidden_size, num_rows, smem)
            // In-place: residual += input; input = norm(residual) * w
            let add_id = *ctx
                .claimed_tiles
                .iter()
                .find(|t| ctx.fuf.get(**t).op == OpKind::Add)
                .expect("fused_add_rms_norm: claim contains Add");
            let rmsnorm_id = *ctx
                .claimed_tiles
                .iter()
                .find(|t| ctx.fuf.get(**t).op == OpKind::RmsNorm)
                .expect("fused_add_rms_norm: claim contains RmsNorm");
            let delta = ctx
                .input_tile_ident(add_id, 0)
                .expect("Add input 0 is a tile");
            let residual = ctx
                .input_tile_ident(add_id, 1)
                .expect("Add input 1 is a tile");
            let rmsnorm_node = ctx.fuf.get(rmsnorm_id);
            let (wid, widx) = rmsnorm_node
                .inputs
                .iter()
                .find_map(|i| match i {
                    FufInput::Weight { id, index } => Some((*id, *index)),
                    _ => None,
                })
                .expect("RmsNorm has weight");
            let wname = weight_field_name(ctx.program, wid, widx);
            let w = ctx.weight_accessor(&wname);
            let hidden_size = ctx.bound("hidden_size") as i32;
            let rmsnorm_out = ctx.output_ident(rmsnorm_id, 0);
            let add_out = ctx.output_ident(add_id, 0);
            Some((
                DevicePhase {
                    flat_params: vec![
                        ("void*".into(), format!("{p}_input")),
                        ("void*".into(), format!("{p}_residual")),
                        ("const void*".into(), format!("{p}_weight")),
                        ("float".into(), format!("{p}_eps")),
                        ("int".into(), format!("{p}_hidden_size")),
                        ("int".into(), format!("{p}_num_tokens")),
                    ],
                    kernel_body: vec![format!(
                        "dc_fused_add_rms_norm<__nv_bfloat16>({p}_input, {p}_residual, {p}_weight, {p}_eps, {p}_hidden_size, {p}_num_tokens, smem);"
                    )],
                    params_build: vec![
                        format!("params.{p}_input = (__nv_bfloat16*){p}_input;"),
                        format!("params.{p}_residual = (__nv_bfloat16*){p}_residual;"),
                        format!("params.{p}_weight = (const __nv_bfloat16*){p}_weight;"),
                        format!("params.{p}_eps = {p}_eps;"),
                        format!("params.{p}_hidden_size = {p}_hidden_size;"),
                        format!("params.{p}_num_tokens = {p}_num_tokens;"),
                    ],
                    internal_fields: vec![
                        format!("__nv_bfloat16* {p}_input"),
                        format!("__nv_bfloat16* {p}_residual"),
                        format!("const __nv_bfloat16* {p}_weight"),
                        format!("float {p}_eps"),
                        format!("int {p}_hidden_size"),
                        format!("int {p}_num_tokens"),
                    ],
                    preamble: vec![],
                },
                // Preamble: output alias bindings (in-place mutation)
                vec![
                    quote! { let #rmsnorm_out = unsafe { (*#delta).as_view() }; },
                    quote! { let #add_out = unsafe { (*#residual).as_view() }; },
                ],
                // Param values
                vec![
                    quote! { (*#delta).raw_ptr() as *mut ::core::ffi::c_void },
                    quote! { (*#residual).raw_ptr() as *mut ::core::ffi::c_void },
                    quote! { (#w).weight.raw_ptr() as *const ::core::ffi::c_void },
                    quote! { (#w).eps },
                    quote! { #hidden_size },
                    quote! { #num_tokens_val },
                ],
            ))
        }
        "fused_add_rms_norm_with_offset" => {
            let residual_add_id = *ctx
                .claimed_tiles
                .iter()
                .find(|t| {
                    let n = ctx.fuf.get(**t);
                    n.op == OpKind::Add
                        && n.inputs.iter().all(|i| matches!(i, FufInput::Tile { .. }))
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
            let delta = ctx
                .input_tile_ident(residual_add_id, 0)
                .expect("residual Add input 0 (delta)");
            let residual = ctx
                .input_tile_ident(residual_add_id, 1)
                .expect("residual Add input 1 (residual)");
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
                    FufInput::Weight { id, index } => Some((*id, *index)),
                    _ => None,
                })
                .expect("scalar-offset Add has a Weight input");
            let wname = weight_field_name(ctx.program, weight_id, weight_idx);
            let w = ctx.weight_accessor(&wname);
            let hidden_size = ctx.bound("hidden_size") as i32;
            let rmsnorm_out = ctx.output_ident(rmsnorm_id, 0);
            let add_out = ctx.output_ident(residual_add_id, 0);
            Some((
                DevicePhase {
                    flat_params: vec![
                        ("void*".into(), format!("{p}_input")),
                        ("void*".into(), format!("{p}_residual")),
                        ("const void*".into(), format!("{p}_weight")),
                        ("float".into(), format!("{p}_eps")),
                        ("float".into(), format!("{p}_weight_offset")),
                        ("int".into(), format!("{p}_hidden_size")),
                        ("int".into(), format!("{p}_num_tokens")),
                    ],
                    kernel_body: vec![format!(
                        "dc_fused_add_rms_norm_with_offset<__nv_bfloat16>({p}_input, {p}_residual, {p}_weight, {p}_eps, {p}_weight_offset, {p}_hidden_size, {p}_num_tokens, smem);"
                    )],
                    params_build: vec![
                        format!("params.{p}_input = (__nv_bfloat16*){p}_input;"),
                        format!("params.{p}_residual = (__nv_bfloat16*){p}_residual;"),
                        format!("params.{p}_weight = (const __nv_bfloat16*){p}_weight;"),
                        format!("params.{p}_eps = {p}_eps;"),
                        format!("params.{p}_weight_offset = {p}_weight_offset;"),
                        format!("params.{p}_hidden_size = {p}_hidden_size;"),
                        format!("params.{p}_num_tokens = {p}_num_tokens;"),
                    ],
                    internal_fields: vec![
                        format!("__nv_bfloat16* {p}_input"),
                        format!("__nv_bfloat16* {p}_residual"),
                        format!("const __nv_bfloat16* {p}_weight"),
                        format!("float {p}_eps"),
                        format!("float {p}_weight_offset"),
                        format!("int {p}_hidden_size"),
                        format!("int {p}_num_tokens"),
                    ],
                    preamble: vec![],
                },
                vec![
                    quote! { let #rmsnorm_out = unsafe { (*#delta).as_view() }; },
                    quote! { let #add_out = unsafe { (*#residual).as_view() }; },
                ],
                vec![
                    quote! { (*#delta).raw_ptr() as *mut ::core::ffi::c_void },
                    quote! { (*#residual).raw_ptr() as *mut ::core::ffi::c_void },
                    quote! { (#w).weight.raw_ptr() as *const ::core::ffi::c_void },
                    quote! { (#w).eps },
                    quote! { #offset },
                    quote! { #hidden_size },
                    quote! { #num_tokens_val },
                ],
            ))
        }
        "scalar_offset_rms_norm" => {
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
            let offset: f32 = add_node
                .inputs
                .iter()
                .find_map(|i| match i {
                    FufInput::Scalar(v) => Some(*v as f32),
                    _ => None,
                })
                .expect("scalar-offset Add has a Scalar input");
            let x = ctx.input_expr(rmsnorm_id, 0);
            let (weight_id, weight_idx) = add_node
                .inputs
                .iter()
                .find_map(|i| match i {
                    FufInput::Weight { id, index } => Some((*id, *index)),
                    _ => None,
                })
                .expect("scalar-offset Add has a Weight input");
            let wname = weight_field_name(ctx.program, weight_id, weight_idx);
            let w = ctx.weight_accessor(&wname);
            let hidden_size = ctx.bound("hidden_size") as i32;
            let hs = hidden_size;
            let out = ctx.output_ident(rmsnorm_id, 0);
            Some((
                DevicePhase {
                    flat_params: vec![
                        ("void*".into(), format!("{p}_out")),
                        ("const void*".into(), format!("{p}_input")),
                        ("const void*".into(), format!("{p}_weight")),
                        ("float".into(), format!("{p}_eps")),
                        ("float".into(), format!("{p}_weight_offset")),
                        ("int".into(), format!("{p}_hidden_size")),
                        ("int".into(), format!("{p}_num_tokens")),
                    ],
                    kernel_body: vec![format!(
                        "dc_rms_norm_with_offset<__nv_bfloat16>({p}_out, {p}_input, {p}_weight, {p}_eps, {p}_weight_offset, {p}_hidden_size, {p}_num_tokens, smem);"
                    )],
                    params_build: vec![
                        format!("params.{p}_out = (__nv_bfloat16*){p}_out;"),
                        format!("params.{p}_input = (const __nv_bfloat16*){p}_input;"),
                        format!("params.{p}_weight = (const __nv_bfloat16*){p}_weight;"),
                        format!("params.{p}_eps = {p}_eps;"),
                        format!("params.{p}_weight_offset = {p}_weight_offset;"),
                        format!("params.{p}_hidden_size = {p}_hidden_size;"),
                        format!("params.{p}_num_tokens = {p}_num_tokens;"),
                    ],
                    internal_fields: vec![
                        format!("__nv_bfloat16* {p}_out"),
                        format!("const __nv_bfloat16* {p}_input"),
                        format!("const __nv_bfloat16* {p}_weight"),
                        format!("float {p}_eps"),
                        format!("float {p}_weight_offset"),
                        format!("int {p}_hidden_size"),
                        format!("int {p}_num_tokens"),
                    ],
                    preamble: vec![],
                },
                vec![quote! {
                    let #out = device.caching.alloc_tensor(
                        &[#num_tokens_val as usize, #hs as usize],
                        ::ferrite_cuda_core::DType::BF16,
                    );
                }],
                vec![
                    quote! { #out.raw_ptr() as *mut ::core::ffi::c_void },
                    quote! { (#x).raw_ptr() as *const ::core::ffi::c_void },
                    quote! { (#w).weight.raw_ptr() as *const ::core::ffi::c_void },
                    quote! { (#w).eps },
                    quote! { #offset },
                    quote! { #hidden_size },
                    quote! { #num_tokens_val },
                ],
            ))
        }
        "scalar_mul_inplace" => {
            let tile = ctx.primary();
            let out = ctx.output_ident(tile, 0);
            let node = ctx.fuf.get(tile);
            let upstream_slot = node
                .inputs
                .iter()
                .position(|i| matches!(i, FufInput::Tile { .. }))
                .expect("ScalarMul has a Tile input");
            let upstream = ctx
                .input_tile_ident(tile, upstream_slot)
                .expect("ScalarMul Tile input ident");
            let scale: f32 = node
                .inputs
                .iter()
                .find_map(|i| match i {
                    FufInput::Scalar(v) => Some(*v as f32),
                    _ => None,
                })
                .expect("ScalarMul has a Scalar input");
            let hidden_size = ctx.bound("hidden_size") as i32;
            Some((
                DevicePhase {
                    flat_params: vec![
                        ("void*".into(), format!("{p}_x")),
                        ("float".into(), format!("{p}_scalar")),
                        ("int".into(), format!("{p}_n")),
                        ("int".into(), format!("{p}_num_rows")),
                    ],
                    kernel_body: vec![format!(
                        "dc_scalar_mul_inplace<__nv_bfloat16>({p}_x, {p}_scalar, {p}_n, {p}_num_rows);"
                    )],
                    params_build: vec![
                        format!("params.{p}_x = (__nv_bfloat16*){p}_x;"),
                        format!("params.{p}_scalar = {p}_scalar;"),
                        format!("params.{p}_n = {p}_n;"),
                        format!("params.{p}_num_rows = {p}_num_rows;"),
                    ],
                    internal_fields: vec![
                        format!("__nv_bfloat16* {p}_x"),
                        format!("float {p}_scalar"),
                        format!("int {p}_n"),
                        format!("int {p}_num_rows"),
                    ],
                    preamble: vec![],
                },
                vec![quote! { let #out = unsafe { #upstream }; }],
                vec![
                    quote! { (*#out).raw_ptr() as *mut ::core::ffi::c_void },
                    quote! { #scale },
                    quote! { #hidden_size },
                    quote! { #num_tokens_val },
                ],
            ))
        }
        "tanh_softcap_inplace" => {
            let tile = ctx.primary();
            let out = ctx.output_ident(tile, 0);
            let upstream = ctx
                .input_tile_ident(tile, 0)
                .expect("tanh_softcap input is a tile");
            let cap: f32 = ctx.scalar("final_logit_softcapping").unwrap_or_else(|| {
                panic!("tanh_softcap tile emitted but no final_logit_softcapping in config")
            }) as f32;
            let inv_cap: f32 = 1.0 / cap;
            Some((
                DevicePhase {
                    flat_params: vec![
                        ("void*".into(), format!("{p}_x")),
                        ("float".into(), format!("{p}_inv_cap")),
                        ("float".into(), format!("{p}_cap")),
                        ("int".into(), format!("{p}_n")),
                    ],
                    kernel_body: vec![format!(
                        "dc_tanh_softcap_inplace<__nv_bfloat16>({p}_x, {p}_inv_cap, {p}_cap, {p}_n);"
                    )],
                    params_build: vec![
                        format!("params.{p}_x = (__nv_bfloat16*){p}_x;"),
                        format!("params.{p}_inv_cap = {p}_inv_cap;"),
                        format!("params.{p}_cap = {p}_cap;"),
                        format!("params.{p}_n = {p}_n;"),
                    ],
                    internal_fields: vec![
                        format!("__nv_bfloat16* {p}_x"),
                        format!("float {p}_inv_cap"),
                        format!("float {p}_cap"),
                        format!("int {p}_n"),
                    ],
                    preamble: vec![],
                },
                vec![quote! { let #out = unsafe { #upstream }; }],
                vec![
                    quote! { (*#out).raw_ptr() as *mut ::core::ffi::c_void },
                    quote! { #inv_cap },
                    quote! { #cap },
                    quote! { ((*#out).numel()) as i32 },
                ],
            ))
        }
        name if name.starts_with("cutlass_") && !name.contains("gemv") => {
            // Parse cutlass_{M}x{N}_s{S}
            let (tile_m, tile_n, stages) = parse_cutlass_tile_name(name)?;
            let (warp_m, warp_n) = warp_shape_for_tb(tile_m, tile_n);
            let tb_k = 32u32;
            Some((
                DevicePhase {
                    flat_params: vec![
                        ("void*".into(), format!("{p}_C")),
                        ("const void*".into(), format!("{p}_A")),
                        ("const void*".into(), format!("{p}_B")),
                        ("int".into(), format!("{p}_M")),
                        ("int".into(), format!("{p}_N")),
                        ("int".into(), format!("{p}_K")),
                        ("float".into(), format!("{p}_alpha")),
                        ("float".into(), format!("{p}_beta")),
                    ],
                    kernel_body: vec![
                        format!("// CUTLASS GEMM kernel-level invocation"),
                        format!("GemmKernel_p{idx}()({p}_gemm_params,"),
                        format!(
                            "    *reinterpret_cast<typename GemmKernel_p{idx}::SharedStorage*>(smem));"
                        ),
                    ],
                    params_build: vec![
                        format!("// Build CUTLASS kernel params from flat args via device::Gemm"),
                        format!("{{"),
                        format!("    typename DeviceGemm_p{idx}::Arguments args("),
                        format!("        {{{p}_M, {p}_N, {p}_K}},"),
                        format!("        {{(cutlass::bfloat16_t const*){p}_A, {p}_K}},"),
                        format!("        {{(cutlass::bfloat16_t const*){p}_B, {p}_K}},"),
                        format!("        {{(cutlass::bfloat16_t*){p}_C, {p}_N}},"),
                        format!("        {{(cutlass::bfloat16_t*){p}_C, {p}_N}},"),
                        format!("        {{{p}_alpha, {p}_beta}}"),
                        format!("    );"),
                        format!("    DeviceGemm_p{idx} gemm_op;"),
                        format!("    gemm_op.initialize(args, nullptr);"),
                        format!("    static_assert("),
                        format!(
                            "        sizeof(DeviceGemm_p{idx}) >= sizeof(typename GemmKernel_p{idx}::Params),"
                        ),
                        format!("        \"DeviceGemm layout assumption violated\");"),
                        format!("    params.{p}_gemm_params = *reinterpret_cast<"),
                        format!("        typename GemmKernel_p{idx}::Params const*>(&gemm_op);"),
                        format!("}}"),
                    ],
                    internal_fields: vec![format!(
                        "typename GemmKernel_p{idx}::Params {p}_gemm_params"
                    )],
                    preamble: vec![
                        format!("#include <cutlass/cutlass.h>"),
                        format!("#include <cutlass/gemm/device/gemm.h>"),
                        format!("#include <cutlass/epilogue/thread/linear_combination.h>"),
                        format!("// Phase {idx}: CUTLASS GEMM {tile_m}x{tile_n} s{stages}"),
                        format!("using DeviceGemm_p{idx} = cutlass::gemm::device::Gemm<"),
                        format!("    cutlass::bfloat16_t, cutlass::layout::RowMajor,"),
                        format!("    cutlass::bfloat16_t, cutlass::layout::ColumnMajor,"),
                        format!("    cutlass::bfloat16_t, cutlass::layout::RowMajor,"),
                        format!("    float,"),
                        format!("    cutlass::arch::OpClassTensorOp,"),
                        format!("    cutlass::arch::Sm80,"),
                        format!("    cutlass::gemm::GemmShape<{tile_m}, {tile_n}, {tb_k}>,"),
                        format!("    cutlass::gemm::GemmShape<{warp_m}, {warp_n}, {tb_k}>,"),
                        format!("    cutlass::gemm::GemmShape<16, 8, 16>,"),
                        format!("    cutlass::epilogue::thread::LinearCombination<"),
                        format!("        cutlass::bfloat16_t, 8, float, float>,"),
                        format!(
                            "    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>,"
                        ),
                        format!("    {stages}"),
                        format!(">;"),
                        format!(
                            "using GemmKernel_p{idx} = typename DeviceGemm_p{idx}::GemmKernel;"
                        ),
                    ],
                },
                {
                    let tile = ctx.primary();
                    let w = ctx.input_expr(tile, 1);
                    let out = ctx.output_ident(tile, 0);
                    vec![
                        quote! { let __w_dense = (#w).dense_weight(); },
                        quote! { let __n = __w_dense.shape()[0] as usize; },
                        quote! { let __k = __w_dense.shape()[1] as usize; },
                        quote! {
                            let #out = device.caching.alloc_tensor(
                                &[#num_tokens_val as usize, __n],
                                ::ferrite_cuda_core::DType::BF16,
                            );
                        },
                    ]
                },
                {
                    let tile = ctx.primary();
                    let x = ctx.input_expr(tile, 0);
                    let out = ctx.output_ident(tile, 0);
                    vec![
                        quote! { #out.raw_ptr() as *mut ::core::ffi::c_void },
                        quote! { (#x).raw_ptr() as *const ::core::ffi::c_void },
                        quote! { __w_dense.raw_ptr() as *const ::core::ffi::c_void },
                        quote! { #num_tokens_val },
                        quote! { __n as i32 },
                        quote! { __k as i32 },
                        quote! { 1.0f32 },
                        quote! { 0.0f32 },
                    ]
                },
            ))
        }
        "cutlass_gemv" => {
            let tile = ctx.primary();
            let x = ctx.input_expr(tile, 0);
            let w = ctx.input_expr(tile, 1);
            let out = ctx.output_ident(tile, 0);
            Some((
                DevicePhase {
                    flat_params: vec![
                        ("void*".into(), format!("{p}_out")),
                        ("const void*".into(), format!("{p}_x")),
                        ("const void*".into(), format!("{p}_W")),
                        ("int".into(), format!("{p}_N")),
                        ("int".into(), format!("{p}_K")),
                        ("float".into(), format!("{p}_alpha")),
                        ("float".into(), format!("{p}_beta")),
                    ],
                    kernel_body: vec![format!(
                        "dc_gemv({p}_out, {p}_x, {p}_W, {p}_N, {p}_K, {p}_alpha, {p}_beta, {p}_N);"
                    )],
                    params_build: vec![
                        format!("params.{p}_out = (__nv_bfloat16*){p}_out;"),
                        format!("params.{p}_x = (const __nv_bfloat16*){p}_x;"),
                        format!("params.{p}_W = (const __nv_bfloat16*){p}_W;"),
                        format!("params.{p}_N = {p}_N;"),
                        format!("params.{p}_K = {p}_K;"),
                        format!("params.{p}_alpha = {p}_alpha;"),
                        format!("params.{p}_beta = {p}_beta;"),
                    ],
                    internal_fields: vec![
                        format!("__nv_bfloat16* {p}_out"),
                        format!("const __nv_bfloat16* {p}_x"),
                        format!("const __nv_bfloat16* {p}_W"),
                        format!("int {p}_N"),
                        format!("int {p}_K"),
                        format!("float {p}_alpha"),
                        format!("float {p}_beta"),
                    ],
                    preamble: vec![],
                },
                vec![
                    quote! { let __w_dense = (#w).dense_weight(); },
                    quote! { let __n = __w_dense.shape()[0] as usize; },
                    quote! { let __k = __w_dense.shape()[1] as usize; },
                    quote! {
                        let #out = device.caching.alloc_tensor(
                            &[1, __n],
                            ::ferrite_cuda_core::DType::BF16,
                        );
                    },
                ],
                vec![
                    quote! { #out.raw_ptr() as *mut ::core::ffi::c_void },
                    quote! { (#x).raw_ptr() as *const ::core::ffi::c_void },
                    quote! { __w_dense.raw_ptr() as *const ::core::ffi::c_void },
                    quote! { __n as i32 },
                    quote! { __k as i32 },
                    quote! { 1.0f32 },
                    quote! { 0.0f32 },
                ],
            ))
        }
        "fused_qkv_rope_cache" => {
            let rope_id = *ctx
                .claimed_tiles
                .iter()
                .find(|t| ctx.fuf.get(**t).op == OpKind::RopeAppend)
                .expect("claim contains a RopeAppend");
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
            let q_size = (num_q_heads * head_dim) as i32;
            let kv_size = (num_kv_heads * head_dim) as i32;
            let head_dim_i32 = head_dim as i32;
            let layer = rope_kv_cache_layer(rope_node).expect("RopeAppend has KvCache layer index")
                as usize;
            let q_out = ctx.output_ident(rope_id, 0);
            let k_out = ctx.output_ident(rope_id, 1);
            let v_out = ctx.output_ident(rope_id, 2);
            let qs = q_size as usize;
            Some((
                DevicePhase {
                    flat_params: vec![
                        ("void*".into(), format!("{p}_q_out")),
                        ("void*".into(), format!("{p}_key_cache")),
                        ("void*".into(), format!("{p}_value_cache")),
                        ("const void*".into(), format!("{p}_qkv")),
                        ("const void*".into(), format!("{p}_positions")),
                        ("const void*".into(), format!("{p}_cos_sin_cache")),
                        ("const void*".into(), format!("{p}_slot_mapping")),
                        ("int".into(), format!("{p}_q_size")),
                        ("int".into(), format!("{p}_kv_size")),
                        ("int".into(), format!("{p}_head_dim")),
                        ("int".into(), format!("{p}_num_tokens")),
                    ],
                    kernel_body: vec![
                        format!(
                            "dc_fused_qkv_rope_cache({p}_q_out, {p}_key_cache, {p}_value_cache,"
                        ),
                        format!("    {p}_qkv, {p}_positions, {p}_cos_sin_cache, {p}_slot_mapping,"),
                        format!(
                            "    {p}_q_size, {p}_kv_size, {p}_head_dim, {p}_num_tokens, smem);"
                        ),
                    ],
                    params_build: vec![
                        format!("params.{p}_q_out = (__nv_bfloat16*){p}_q_out;"),
                        format!("params.{p}_key_cache = (__nv_bfloat16*){p}_key_cache;"),
                        format!("params.{p}_value_cache = (__nv_bfloat16*){p}_value_cache;"),
                        format!("params.{p}_qkv = (const __nv_bfloat16*){p}_qkv;"),
                        format!("params.{p}_positions = (const uint32_t*){p}_positions;"),
                        format!(
                            "params.{p}_cos_sin_cache = (const __nv_bfloat16*){p}_cos_sin_cache;"
                        ),
                        format!("params.{p}_slot_mapping = (const int64_t*){p}_slot_mapping;"),
                        format!("params.{p}_q_size = {p}_q_size;"),
                        format!("params.{p}_kv_size = {p}_kv_size;"),
                        format!("params.{p}_head_dim = {p}_head_dim;"),
                        format!("params.{p}_num_tokens = {p}_num_tokens;"),
                    ],
                    internal_fields: vec![
                        format!("__nv_bfloat16* {p}_q_out"),
                        format!("__nv_bfloat16* {p}_key_cache"),
                        format!("__nv_bfloat16* {p}_value_cache"),
                        format!("const __nv_bfloat16* {p}_qkv"),
                        format!("const uint32_t* {p}_positions"),
                        format!("const __nv_bfloat16* {p}_cos_sin_cache"),
                        format!("const int64_t* {p}_slot_mapping"),
                        format!("int {p}_q_size"),
                        format!("int {p}_kv_size"),
                        format!("int {p}_head_dim"),
                        format!("int {p}_num_tokens"),
                    ],
                    preamble: vec![],
                },
                // Preamble: allocate output, bind kv cache, run fused GEMM
                vec![
                    quote! {
                        let #q_out = device.caching.alloc_tensor(
                            &[#num_tokens_val as usize, #qs],
                            ::ferrite_cuda_core::DType::BF16,
                        );
                    },
                    quote! { let #k_out = ctx.kv_cache.k_cache(#layer); },
                    quote! { let #v_out = ctx.kv_cache.v_cache(#layer); },
                    quote! {
                        let __qkv_tmp = unsafe {
                            (#weight_expr).forward(
                                #activation,
                                &mut device.cublas,
                                &mut device.caching,
                                device.compute_stream,
                            )
                        };
                    },
                ],
                // Param values
                vec![
                    quote! { #q_out.raw_ptr() as *mut ::core::ffi::c_void },
                    quote! { (*ctx.kv_cache.k_cache(#layer)).raw_ptr() as *mut ::core::ffi::c_void },
                    quote! { (*ctx.kv_cache.v_cache(#layer)).raw_ptr() as *mut ::core::ffi::c_void },
                    quote! { (*__qkv_tmp).raw_ptr() as *const ::core::ffi::c_void },
                    quote! { (*ctx.positions).raw_ptr() as *const ::core::ffi::c_void },
                    quote! { ctx.rotary.cos_sin_cache.raw_ptr() as *const ::core::ffi::c_void },
                    quote! { (*ctx.slot_mapping).raw_ptr() as *const ::core::ffi::c_void },
                    quote! { #q_size },
                    quote! { #kv_size },
                    quote! { #head_dim_i32 },
                    quote! { #num_tokens_val },
                ],
            ))
        }
        "fused_qkv_rope_prefill" => {
            let rope_id = *ctx
                .claimed_tiles
                .iter()
                .find(|t| ctx.fuf.get(**t).op == OpKind::RopeAppend)
                .expect("claim contains a RopeAppend");
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
            let q_size = (num_q_heads * head_dim) as i32;
            let kv_size = (num_kv_heads * head_dim) as i32;
            let head_dim_i32 = head_dim as i32;
            let qs = q_size as usize;
            let kvs = kv_size as usize;
            let q_out = ctx.output_ident(rope_id, 0);
            let k_out = ctx.output_ident(rope_id, 1);
            let v_out = ctx.output_ident(rope_id, 2);
            Some((
                DevicePhase {
                    flat_params: vec![
                        ("void*".into(), format!("{p}_q_out")),
                        ("void*".into(), format!("{p}_k_out")),
                        ("void*".into(), format!("{p}_v_out")),
                        ("const void*".into(), format!("{p}_qkv")),
                        ("const void*".into(), format!("{p}_positions")),
                        ("const void*".into(), format!("{p}_cos_sin_cache")),
                        ("int".into(), format!("{p}_q_size")),
                        ("int".into(), format!("{p}_kv_size")),
                        ("int".into(), format!("{p}_head_dim")),
                        ("int".into(), format!("{p}_num_tokens")),
                    ],
                    kernel_body: vec![
                        format!("dc_fused_qkv_rope_prefill({p}_q_out, {p}_k_out, {p}_v_out,"),
                        format!("    {p}_qkv, {p}_positions, {p}_cos_sin_cache,"),
                        format!(
                            "    {p}_q_size, {p}_kv_size, {p}_head_dim, {p}_num_tokens, smem);"
                        ),
                    ],
                    params_build: vec![
                        format!("params.{p}_q_out = (__nv_bfloat16*){p}_q_out;"),
                        format!("params.{p}_k_out = (__nv_bfloat16*){p}_k_out;"),
                        format!("params.{p}_v_out = (__nv_bfloat16*){p}_v_out;"),
                        format!("params.{p}_qkv = (const __nv_bfloat16*){p}_qkv;"),
                        format!("params.{p}_positions = (const uint32_t*){p}_positions;"),
                        format!(
                            "params.{p}_cos_sin_cache = (const __nv_bfloat16*){p}_cos_sin_cache;"
                        ),
                        format!("params.{p}_q_size = {p}_q_size;"),
                        format!("params.{p}_kv_size = {p}_kv_size;"),
                        format!("params.{p}_head_dim = {p}_head_dim;"),
                        format!("params.{p}_num_tokens = {p}_num_tokens;"),
                    ],
                    internal_fields: vec![
                        format!("__nv_bfloat16* {p}_q_out"),
                        format!("__nv_bfloat16* {p}_k_out"),
                        format!("__nv_bfloat16* {p}_v_out"),
                        format!("const __nv_bfloat16* {p}_qkv"),
                        format!("const uint32_t* {p}_positions"),
                        format!("const __nv_bfloat16* {p}_cos_sin_cache"),
                        format!("int {p}_q_size"),
                        format!("int {p}_kv_size"),
                        format!("int {p}_head_dim"),
                        format!("int {p}_num_tokens"),
                    ],
                    preamble: vec![],
                },
                vec![
                    quote! {
                        let #q_out = device.caching.alloc_tensor(
                            &[#num_tokens_val as usize, #qs as usize],
                            ::ferrite_cuda_core::DType::BF16,
                        );
                    },
                    quote! {
                        let #k_out = device.caching.alloc_tensor(
                            &[#num_tokens_val as usize, #kvs as usize],
                            ::ferrite_cuda_core::DType::BF16,
                        );
                    },
                    quote! {
                        let #v_out = device.caching.alloc_tensor(
                            &[#num_tokens_val as usize, #kvs as usize],
                            ::ferrite_cuda_core::DType::BF16,
                        );
                    },
                    quote! {
                        let __qkv_tmp = unsafe {
                            (#weight_expr).forward(
                                #activation,
                                &mut device.cublas,
                                &mut device.caching,
                                device.compute_stream,
                            )
                        };
                    },
                ],
                vec![
                    quote! { #q_out.raw_ptr() as *mut ::core::ffi::c_void },
                    quote! { #k_out.raw_ptr() as *mut ::core::ffi::c_void },
                    quote! { #v_out.raw_ptr() as *mut ::core::ffi::c_void },
                    quote! { (*__qkv_tmp).raw_ptr() as *const ::core::ffi::c_void },
                    quote! { (*ctx.positions).raw_ptr() as *const ::core::ffi::c_void },
                    quote! { ctx.rotary.cos_sin_cache.raw_ptr() as *const ::core::ffi::c_void },
                    quote! { #q_size },
                    quote! { #kv_size },
                    quote! { #head_dim_i32 },
                    quote! { #num_tokens_val },
                ],
            ))
        }
        "fused_gate_up_silu_mul" | "fused_gate_up_gelu_mul" => {
            let fn_name = if inner_name == "fused_gate_up_silu_mul" {
                "dc_fused_gate_up_silu_mul"
            } else {
                "dc_fused_gate_up_gelu_mul"
            };
            let silu_or_gelu_id = *ctx
                .claimed_tiles
                .iter()
                .find(|t| {
                    let op = ctx.fuf.get(**t).op;
                    op == OpKind::Silu || op == OpKind::Gelu
                })
                .expect("claim contains Silu or Gelu");
            let mul_id = *ctx
                .claimed_tiles
                .iter()
                .find(|t| ctx.fuf.get(**t).op == OpKind::Mul)
                .expect("claim contains Mul");
            let (gate_id, _) = first_tile_input(ctx.fuf.get(silu_or_gelu_id))
                .expect("activation has a tile input — the gate gemm");
            let up_id = ctx
                .claimed_tiles
                .iter()
                .copied()
                .find(|t| ctx.fuf.get(*t).op == OpKind::Gemm && *t != gate_id)
                .expect("claim has a second Gemm — the up gemm");
            let activation = ctx.input_expr(gate_id, 0);
            let gate_w = first_weight_ref(ctx.fuf.get(gate_id)).expect("gate gemm has a weight");
            let up_w = first_weight_ref(ctx.fuf.get(up_id)).expect("up gemm has a weight");
            let fused_name = fused_accessor_name(ctx.program, &[gate_w, up_w]);
            let weight_expr = ctx.weight_accessor(&fused_name);
            let intermediate = ctx.bound("intermediate_size") as i32;
            let hidden_size = ctx.bound("hidden_size") as i32;
            let mul_out = ctx.output_ident(mul_id, 0);
            let inter = intermediate as usize;
            Some((
                DevicePhase {
                    flat_params: vec![
                        ("void*".into(), format!("{p}_out")),
                        ("const void*".into(), format!("{p}_input")),
                        ("const void*".into(), format!("{p}_gate_weight")),
                        ("const void*".into(), format!("{p}_up_weight")),
                        ("int".into(), format!("{p}_M")),
                        ("int".into(), format!("{p}_N")),
                        ("int".into(), format!("{p}_K")),
                    ],
                    kernel_body: vec![
                        format!("{fn_name}({p}_out, {p}_input, {p}_gate_weight, {p}_up_weight,"),
                        format!("    {p}_M, {p}_N, {p}_K, smem);"),
                    ],
                    params_build: vec![
                        format!("params.{p}_out = (__nv_bfloat16*){p}_out;"),
                        format!("params.{p}_input = (const __nv_bfloat16*){p}_input;"),
                        format!("params.{p}_gate_weight = (const __nv_bfloat16*){p}_gate_weight;"),
                        format!("params.{p}_up_weight = (const __nv_bfloat16*){p}_up_weight;"),
                        format!("params.{p}_M = {p}_M;"),
                        format!("params.{p}_N = {p}_N;"),
                        format!("params.{p}_K = {p}_K;"),
                    ],
                    internal_fields: vec![
                        format!("__nv_bfloat16* {p}_out"),
                        format!("const __nv_bfloat16* {p}_input"),
                        format!("const __nv_bfloat16* {p}_gate_weight"),
                        format!("const __nv_bfloat16* {p}_up_weight"),
                        format!("int {p}_M"),
                        format!("int {p}_N"),
                        format!("int {p}_K"),
                    ],
                    preamble: vec![],
                },
                // Preamble: alloc output, get fused weight
                vec![
                    quote! {
                        let #mul_out = device.caching.alloc_tensor(
                            &[#num_tokens_val as usize, #inter],
                            ::ferrite_cuda_core::DType::BF16,
                        );
                    },
                    quote! { let __fused_w = (#weight_expr).dense_weight(); },
                    quote! { let __up_offset = #intermediate as usize * #hidden_size as usize; },
                ],
                // Param values
                vec![
                    quote! { #mul_out.raw_ptr() as *mut ::core::ffi::c_void },
                    quote! { (#activation).raw_ptr() as *const ::core::ffi::c_void },
                    quote! { __fused_w.raw_ptr() as *const ::core::ffi::c_void },
                    quote! {
                        unsafe {
                            (__fused_w.raw_ptr() as *const u8)
                                .add(__up_offset * 2) // 2 bytes per bf16
                                as *const ::core::ffi::c_void
                        }
                    },
                    quote! { #num_tokens_val },
                    quote! { #intermediate },
                    quote! { #hidden_size },
                ],
            ))
        }
        _ => None,
    }
}

/// Parse a CUTLASS tile name like "cutlass_64x128_s3" into (tile_m, tile_n, stages).
fn parse_cutlass_tile_name(name: &str) -> Option<(u32, u32, u32)> {
    let rest = name.strip_prefix("cutlass_")?;
    let (dims, stages_part) = rest.rsplit_once("_s")?;
    let (m_str, n_str) = dims.split_once('x')?;
    Some((
        m_str.parse().ok()?,
        n_str.parse().ok()?,
        stages_part.parse().ok()?,
    ))
}

/// Map threadblock shape to warp shape (matching cutlass_standalone_gemm.cu).
fn warp_shape_for_tb(tile_m: u32, tile_n: u32) -> (u32, u32) {
    match (tile_m, tile_n) {
        (64, 64) => (32, 32),
        (64, 128) => (32, 64),
        (128, 64) => (64, 32),
        (128, 128) => (64, 32),
        (128, 256) => (64, 64),
        (256, 64) => (64, 32),
        (256, 128) => (64, 32),
        _ => (tile_m / 2, tile_n / 2),
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
/// RopeAppend, or gate/up of the `silu(gate) * up` pattern). The
/// cutlass singletons must never claim these — their fused impls
/// own them and the downstream chain has no singleton kernel.
fn gemm_is_fusion_partner(fuf: &Fuf, seed: TileId) -> bool {
    output_feeds_op(fuf, seed, OpKind::RopeAppend)
        || output_feeds_op(fuf, seed, OpKind::Silu)
        || output_feeds_op(fuf, seed, OpKind::Mul)
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

// ── TK GEMM / GEMV (natively DeviceCallable, sm90+) ─────────────
//
// ThunderKittens-based GEMM and GEMV using kittens `warp::mma` /
// `warpgroup::mma_AB` for tensor-core compute. These are natively
// DeviceCallable — they run as phases inside the megakernel's
// `__global__` cooperative kernel, NOT as standalone launches.
//
// Cost model: cuBLAS calibrated cost minus launch_overhead.
// On H100 (sm90), TK wgmma GEMM matches cuBLAS throughput. The
// net savings come from eliminating one kernel launch per GEMM.
// This guarantees the solver prefers TK over standalone cuBLAS
// for every GEMM/GEMV, enabling a 100% DeviceCallable megakernel.
//
// The device_phase() emits calls to `dc_tk_gemm` / `dc_tk_gemv`
// functions in `megakernel_tk_ops.cuh`, which use kittens
// register/shared types and warp::mma internally.

#[derive(Debug)]
pub struct TkGemmImpl;

impl Implementation for TkGemmImpl {
    fn name(&self) -> &'static str {
        "tk_gemm"
    }

    fn target_compatible(&self, profile: &TargetProfile) -> bool {
        profile.compute_capability >= 90 && profile.cost_table.has_kernel("cublas")
    }

    fn workload_constraint(&self) -> WorkloadConstraint {
        // M >= 2 (prefill). M=1 is GEMV territory → TkGemvImpl.
        WorkloadConstraint::NumTokensRange {
            min: 2,
            max: u32::MAX,
        }
    }

    fn matches(&self, fuf: &Fuf, seed: TileId, _profile: &TargetProfile) -> Option<MatchInfo> {
        let info = single_tile_match(fuf, seed, OpKind::Gemm)?;
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
        // TK wgmma GEMM matches cuBLAS throughput on sm90.
        // Subtract launch_overhead because this runs inside a megakernel.
        let base = ctx
            .profile
            .cost_us_for("cublas", mm, nn, kk)
            .unwrap_or_else(|| {
                // No CSV row for this shape — fall back to analytical model.
                let flops = 2.0 * mm as f64 * nn as f64 * kk as f64;
                let peak = ctx.profile.peak_tflops_fp16 * 1e12;
                if peak == 0.0 || flops == 0.0 {
                    0.0
                } else {
                    (flops / peak) * 1e6
                }
            });
        (base - ctx.profile.launch_overhead_us).max(0.0)
    }

    fn resources(&self, _m: &MatchInfo) -> Resources {
        Resources::ZERO
    }

    fn launch_kind(&self) -> LaunchKind {
        LaunchKind::DeviceCallable
    }

    fn supported_input_handoffs(&self) -> &[Handoff] {
        const H: &[Handoff] = &[
            Handoff::Mbarrier,
            Handoff::SyncThreads,
            Handoff::Internal,
            Handoff::StreamOrder,
        ];
        H
    }

    fn supported_output_handoffs(&self) -> &[Handoff] {
        const H: &[Handoff] = &[
            Handoff::Mbarrier,
            Handoff::SyncThreads,
            Handoff::Internal,
            Handoff::StreamOrder,
        ];
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
        // Standalone fallback: use cuBLAS (same as GemmRefImpl).
        emit_gemm(ctx)
    }

    fn device_phase(
        &self,
        idx: usize,
        ctx: &EmitCtx,
    ) -> Option<(DevicePhase, Vec<TokenStream>, Vec<TokenStream>)> {
        let p = format!("p{idx}");
        let num_tokens_val =
            ctx.num_tokens
                .expect("TkGemmImpl::device_phase requires num_tokens") as i32;
        let tile = ctx.primary();
        let x = ctx.input_expr(tile, 0);
        let w = ctx.input_expr(tile, 1);
        let out = ctx.output_ident(tile, 0);
        Some((
            DevicePhase {
                flat_params: vec![
                    ("void*".into(), format!("{p}_out")),
                    ("const void*".into(), format!("{p}_A")),
                    ("const void*".into(), format!("{p}_B")),
                    ("int".into(), format!("{p}_M")),
                    ("int".into(), format!("{p}_N")),
                    ("int".into(), format!("{p}_K")),
                ],
                kernel_body: vec![format!(
                    "dc_tk_gemm({p}_out, {p}_A, {p}_B, {p}_M, {p}_N, {p}_K, smem);"
                )],
                params_build: vec![
                    format!("params.{p}_out = (__nv_bfloat16*){p}_out;"),
                    format!("params.{p}_A = (const __nv_bfloat16*){p}_A;"),
                    format!("params.{p}_B = (const __nv_bfloat16*){p}_B;"),
                    format!("params.{p}_M = {p}_M;"),
                    format!("params.{p}_N = {p}_N;"),
                    format!("params.{p}_K = {p}_K;"),
                ],
                internal_fields: vec![
                    format!("__nv_bfloat16* {p}_out"),
                    format!("const __nv_bfloat16* {p}_A"),
                    format!("const __nv_bfloat16* {p}_B"),
                    format!("int {p}_M"),
                    format!("int {p}_N"),
                    format!("int {p}_K"),
                ],
                preamble: vec![],
            },
            vec![
                quote! { let __w_dense = (#w).dense_weight(); },
                quote! { let __n = __w_dense.shape()[0] as usize; },
                quote! { let __k = __w_dense.shape()[1] as usize; },
                quote! {
                    let #out = device.caching.alloc_tensor(
                        &[#num_tokens_val as usize, __n],
                        ::ferrite_cuda_core::DType::BF16,
                    );
                },
            ],
            vec![
                quote! { #out.raw_ptr() as *mut ::core::ffi::c_void },
                quote! { (#x).raw_ptr() as *const ::core::ffi::c_void },
                quote! { __w_dense.raw_ptr() as *const ::core::ffi::c_void },
                quote! { #num_tokens_val },
                quote! { __n as i32 },
                quote! { __k as i32 },
            ],
        ))
    }
}

/// TK GEMV — natively DeviceCallable for M=1 decode.
///
/// Uses a cooperative grid where each CTA computes a slice of the
/// output vector. Kittens warp::mma with M padded to 16 gives
/// tensor-core throughput even at M=1.
#[derive(Debug)]
pub struct TkGemvImpl;

impl Implementation for TkGemvImpl {
    fn name(&self) -> &'static str {
        "tk_gemv"
    }

    fn target_compatible(&self, profile: &TargetProfile) -> bool {
        profile.compute_capability >= 90 && profile.cost_table.has_kernel("cublas")
    }

    fn workload_constraint(&self) -> WorkloadConstraint {
        WorkloadConstraint::NumTokensRange { min: 1, max: 1 }
    }

    fn matches(&self, fuf: &Fuf, seed: TileId, _profile: &TargetProfile) -> Option<MatchInfo> {
        let info = single_tile_match(fuf, seed, OpKind::Gemm)?;
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
        // TK cooperative GEMV: all CTAs collaborate on one M=1 GEMV.
        // Cost model: cuBLAS minus launch_overhead.
        let base = ctx
            .profile
            .cost_us_for("cublas", mm, nn, kk)
            .unwrap_or_else(|| {
                let flops = 2.0 * mm as f64 * nn as f64 * kk as f64;
                let peak = ctx.profile.peak_tflops_fp16 * 1e12;
                if peak == 0.0 || flops == 0.0 {
                    0.0
                } else {
                    (flops / peak) * 1e6
                }
            });
        (base - ctx.profile.launch_overhead_us).max(0.0)
    }

    fn resources(&self, _m: &MatchInfo) -> Resources {
        Resources::ZERO
    }

    fn launch_kind(&self) -> LaunchKind {
        LaunchKind::DeviceCallable
    }

    fn supported_input_handoffs(&self) -> &[Handoff] {
        const H: &[Handoff] = &[
            Handoff::Mbarrier,
            Handoff::SyncThreads,
            Handoff::Internal,
            Handoff::StreamOrder,
        ];
        H
    }

    fn supported_output_handoffs(&self) -> &[Handoff] {
        const H: &[Handoff] = &[
            Handoff::Mbarrier,
            Handoff::SyncThreads,
            Handoff::Internal,
            Handoff::StreamOrder,
        ];
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
        // Standalone fallback: use cuBLAS (same as GemmRefImpl).
        emit_gemm(ctx)
    }

    fn device_phase(
        &self,
        idx: usize,
        ctx: &EmitCtx,
    ) -> Option<(DevicePhase, Vec<TokenStream>, Vec<TokenStream>)> {
        let p = format!("p{idx}");
        let tile = ctx.primary();
        let x = ctx.input_expr(tile, 0);
        let w = ctx.input_expr(tile, 1);
        let out = ctx.output_ident(tile, 0);
        Some((
            DevicePhase {
                flat_params: vec![
                    ("void*".into(), format!("{p}_out")),
                    ("const void*".into(), format!("{p}_x")),
                    ("const void*".into(), format!("{p}_W")),
                    ("int".into(), format!("{p}_N")),
                    ("int".into(), format!("{p}_K")),
                ],
                kernel_body: vec![format!(
                    "dc_tk_gemv({p}_out, {p}_x, {p}_W, {p}_N, {p}_K, smem);"
                )],
                params_build: vec![
                    format!("params.{p}_out = (__nv_bfloat16*){p}_out;"),
                    format!("params.{p}_x = (const __nv_bfloat16*){p}_x;"),
                    format!("params.{p}_W = (const __nv_bfloat16*){p}_W;"),
                    format!("params.{p}_N = {p}_N;"),
                    format!("params.{p}_K = {p}_K;"),
                ],
                internal_fields: vec![
                    format!("__nv_bfloat16* {p}_out"),
                    format!("const __nv_bfloat16* {p}_x"),
                    format!("const __nv_bfloat16* {p}_W"),
                    format!("int {p}_N"),
                    format!("int {p}_K"),
                ],
                preamble: vec![],
            },
            vec![
                quote! { let __w_dense = (#w).dense_weight(); },
                quote! { let __n = __w_dense.shape()[0] as usize; },
                quote! { let __k = __w_dense.shape()[1] as usize; },
                quote! {
                    let #out = device.caching.alloc_tensor(
                        &[1, __n],
                        ::ferrite_cuda_core::DType::BF16,
                    );
                },
            ],
            vec![
                quote! { #out.raw_ptr() as *mut ::core::ffi::c_void },
                quote! { (#x).raw_ptr() as *const ::core::ffi::c_void },
                quote! { __w_dense.raw_ptr() as *const ::core::ffi::c_void },
                quote! { __n as i32 },
                quote! { __k as i32 },
            ],
        ))
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
            name: syn::Ident::new("gemma2_test", proc_macro2::Span::call_site()),
            source_stem: "gemma2_test".into(),
            source_path: std::path::PathBuf::new(),
            bounds,
            scalars,
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
        let c = WorkloadConstraint::NumTokensRange { min: 1, max: 8 };
        assert!(c.accepts(1));
        assert!(c.accepts(8));
        assert!(!c.accepts(9));
        assert!(!c.accepts(0));
        assert!(WorkloadConstraint::Any.accepts(4096));
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
}
