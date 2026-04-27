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
use crate::codegen::split_base_layer;
use crate::emit::weight_field_name;
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

/// Evaluate a `Dim` to a concrete integer using `bounds`. Free
/// function so non-`CostCtx` callers (e.g. `fan_out`, which only has
/// `fuf + bounds`) can reuse it without round-tripping through a
/// CostCtx instance.
pub fn eval_dim_with(dim: &Dim, bounds: &BTreeMap<String, u64>) -> Option<u64> {
    match dim {
        Dim::Lit(n) => Some(*n),
        Dim::Bound(name) => bounds.get(name).copied(),
        Dim::Mul(cs) => cs
            .iter()
            .map(|c| eval_dim_with(c, bounds))
            .try_fold(1u64, |acc, v| v.map(|x| acc.saturating_mul(x))),
        Dim::Var(_) => None,
    }
}

pub fn eval_shape_with(shape: &Shape, bounds: &BTreeMap<String, u64>) -> Option<Vec<u64>> {
    shape.iter().map(|d| eval_dim_with(d, bounds)).collect()
}

impl CostCtx<'_> {
    /// Evaluate a `Dim` to a concrete integer using `bounds`.
    /// Returns `None` if the dim contains a Var (should not happen
    /// for the standard body after shape inference closes).
    pub fn eval_dim(&self, dim: &Dim) -> Option<u64> {
        eval_dim_with(dim, self.bounds)
    }

    pub fn eval_shape(&self, shape: &Shape) -> Option<Vec<u64>> {
        eval_shape_with(shape, self.bounds)
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

/// Whether an implementation can run inside one of the megakernel
/// interpreters (`PrimMega` / `KvmMega`). See `MEGA_HANDOFF.md`.
///
/// Ordering is meaningful: `Kvm > Primitive > None`. A `Kvm`-fit
/// impl is implicitly `Primitive`-fit. The post-solve interpreter
/// selector takes the `min` over picked impls; if every pick is
/// `>= Primitive` (and the target supports it) the run uses
/// `PrimMega`, and likewise for `Kvm`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum MegakernelFit {
    /// Cannot run inside any megakernel interpreter — must use the
    /// host interpreter. Default for every existing impl.
    None,
    /// Runs inside `PrimMega`: a persistent `__global__` whose body
    /// is a switch over `[i32; 32]` opcodes calling `__device__`
    /// wrappers around vendor kernels (cutlass / flashinfer / TK).
    /// Per-op handoff is `__syncthreads()` or grid sync.
    Primitive,
    /// Runs inside `KvmMega`: vendored `~/Megakernels` template
    /// instantiated per arch with warp specialization, page virtual
    /// memory, per-SM instruction tape. Output-tile-granular
    /// `fan_out`.
    Kvm,
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
    ///
    /// Most entries are target-agnostic (a `cudaLaunchKernel`
    /// boundary costs ~5us regardless of which sm_XX runs it).
    /// `InKernelGridSync` is the exception: `cg::this_grid().sync()`
    /// rides on hardware that differs sharply across capabilities.
    /// On Hopper (sm_90+) the cooperative-groups grid barrier is
    /// implemented on top of the L2 synchronizer + TMA fences and
    /// runs in ~15us per call. On Ada / Ampere (sm_89, sm_80) there's
    /// no async-barrier hardware — falls back to atomic spin counters
    /// in gmem at ~100us. This gap is what made the older
    /// `worktree-ferrite-mega` branch's "100% megakernel" path
    /// (`f5918269f`: a single cooperative kernel launch per decoder
    /// layer with grid sync between phases) "decent" on H100 but
    /// pessimal on L4: 320 syncs × 15us ≈ 4.8ms is small enough to
    /// be amortized; 320 × 100us ≈ 32ms isn't.
    ///
    /// Without this split, `launch_overhead_us` reports the same
    /// +100us DC-vs-host delta on H100 as on L4, the solver picks
    /// host everywhere, and PrimMega is dead on every target. With
    /// it, on H100 the delta shrinks to ~10us — small enough that
    /// any op whose own cost is dominated by launch overhead (small-M
    /// decode, fused norms) prefers DC, matching the older branch's
    /// shipped behavior.
    pub fn cost_us(&self, profile: &TargetProfile) -> f64 {
        match self {
            Handoff::StreamOrder => 0.0,
            Handoff::StreamEvent => 5.0,
            Handoff::KernelBoundary => 5.0,
            Handoff::InKernelGridSync => {
                if profile.compute_capability >= 90 {
                    // Hopper: hw-accelerated cooperative-groups grid
                    // barrier via L2 synchronizer + TMA fences. The
                    // older `worktree-ferrite-mega` branch's H100
                    // 100% megakernel path runs 9 syncs/layer × 32
                    // layers in the few-millisecond range, consistent
                    // with this number.
                    15.0
                } else {
                    // Ada / Ampere: atomic spin on gmem counter,
                    // L4 sm_89 measurement. Same number as the
                    // pre-split constant.
                    100.0
                }
            }
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

/// Per-launch sync overhead an implementation pays just to enter
/// its execution context, charged once per pick — not per input
/// edge.
///
/// **Host launches** (`HostCallback` / `RegularLaunch` /
/// `CooperativeLaunch`) pay one [`Handoff::KernelBoundary`]
/// regardless of how many input tensors they gather (one
/// `cudaLaunchKernel` call serves them all). Charging per input
/// edge would over-penalize wide-fan-in kernels (e.g. attention's
/// q/k/v/positions/cache gather) by 5N us instead of 5us, which
/// is not how the hardware works.
///
/// **`DeviceCallable`** pays **zero** at the per-pick level. A
/// chain of N DC siblings sharing one cooperative kernel launch
/// pays one launch boundary at the wave level, amortized over all
/// N picks — i.e., `launch_overhead / N` per pick. With N
/// typically 5–10 inside a mega and `KernelBoundary = 5us`, that's
/// well under 1us per pick — round to zero. The wave-level launch
/// counter (TODO, matching `worktree-ferrite-mega/cost.rs:90–112`'s
/// `launch_count` term) charges the actual cooperative-launch cost
/// once per mega wave; doing it both per-pick and per-wave would
/// double-count.
///
/// **No [`Handoff::InKernelGridSync`] charge per pick.** The
/// 100us-on-Ada / 15us-on-Hopper grid.sync IS a real cost paid
/// between adjacent DC ops inside a mega, but it's a wave-internal
/// term, not a per-pick entry overhead. Charging it per-pick
/// double-counts vs the eventual wave-level model and (more
/// pragmatically) made every DC sibling lose every per-seed cost
/// comparison so the solver never picked any of them. The older
/// `worktree-ferrite-mega` branch never charged grid.sync per pick
/// anywhere in its cost summation either — it relied on the
/// per-target `launch_overhead_us` and the wave-level `launch_count`
/// term to get the ranking right.
///
/// **Solver effect.** Host pays +5us per pick; DC pays +0us. DC
/// strictly wins at every seed where it's solver-feasible. Today
/// DC siblings gate on
/// [`TargetProfile::prim_mega_compatible`] (`compute_capability >=
/// 90`), so feasibility is restricted to Hopper+ — exactly the
/// regime where prim-mega is competitive with host. On Ada the
/// gate keeps DC out of the candidate set entirely, so this
/// per-pick zero never actually mis-ranks anything.
///
/// **Phase 2** refines `DeviceCallable` per
/// [`MegakernelFit`]: Kvm-fit impls inside a KVM megakernel use
/// [`Handoff::Mbarrier`] (~0.1us) for intra-kernel handoffs and
/// produce a different launch-shape entirely. Tracked in
/// `MEGA_HANDOFF.md` §"Cost-metric refinements".
pub fn launch_overhead_us(kind: LaunchKind, profile: &TargetProfile) -> f64 {
    match kind {
        LaunchKind::HostCallback | LaunchKind::RegularLaunch | LaunchKind::CooperativeLaunch => {
            Handoff::KernelBoundary.cost_us(profile)
        }
        LaunchKind::DeviceCallable => 0.0,
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

    /// Whether this impl can run inside one of the megakernel
    /// interpreters. Default `None` means host-only — the impl
    /// participates in solver scoring exactly as before, but it
    /// disqualifies the canonical from `PrimMega` / `KvmMega`
    /// post-solve. New `DeviceCallable` wrappings should return
    /// `Primitive`; new KVM-template Impls should return `Kvm`.
    /// See `MEGA_HANDOFF.md` for the locked design.
    fn megakernel_fit(&self) -> MegakernelFit {
        MegakernelFit::None
    }

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

    // ── Host-interpreter codegen ────────────────────────────────────
    //
    // Each Impl owns the shape of its own opcode — variant ident
    // and typed payload fields. The proc-macro, processing one
    // arch's solved FUF, collects shapes from the picked Impls and
    // uses them to type-check static-slice emission. The eval
    // bodies live in `ferrite_forward::Instruction::eval`; an Impl
    // adds a kernel by overriding two methods: `opcode_shape` (the
    // variant declaration matching `Instruction<W>`'s) and
    // `fan_out` (the per-claim instances). A new Impl that forgets
    // to override these fails at codegen time — `fan_out` returns
    // `None` and `lower_bucket` panics with the Impl's name.

    /// Variant declaration this Impl contributes to the per-arch
    /// enum. Variant ident + ordered `(field_ident, field_type)`
    /// pairs. Two Impls returning the same variant ident must
    /// declare structurally identical fields — codegen verifies and
    /// panics on mismatch.
    fn opcode_shape(&self) -> OpcodeShape {
        OpcodeShape::unmigrated(self.name())
    }

    /// One [`OpInstance`] per kernel call this Impl makes at this
    /// (variant × workload-point). Field-value tokens are positional,
    /// matching the order of fields in `opcode_shape().fields`.
    /// `slots` resolves boundary-tile `(TileId, output_slot)` pairs
    /// to `u32` slot indices in the runtime tile table.
    ///
    /// Default `None` flags the Impl as not yet migrated. The
    /// codegen seam panics with the Impl's `name()` when it sees
    /// `None`.
    fn fan_out(
        &self,
        _m: &MatchInfo,
        _fuf: &Fuf,
        _program: &Program,
        _bounds: &BTreeMap<String, u64>,
        _slots: &SlotMap,
    ) -> Option<Vec<OpInstance>> {
        None
    }
}

/// Variant declaration an Impl contributes to its arch's
/// macro-emitted opcode enum.
///
/// The codegen, processing one arch's solved FUF, collects
/// `OpcodeShape`s from the Impls the solver picked for that arch.
/// From the union of shapes it emits one Rust enum per arch:
///
/// ```ignore
/// enum LlamaOp {
///     AttnNorm { layer: u32, in_slot: u32, out_slot: u32 },
///     QkvRopeAppend { layer: u32, in_slot: u32, out_q_slot: u32 },
///     // … only the variants Llama's picked Impls declared
///     Free { slot: u32 },  // injected by codegen, not by any Impl
/// }
/// ```
///
/// `Free` is added unconditionally by codegen for the drop pass.
/// Every Impl-driven variant is named here, by exactly one Impl.
/// Two Impls declaring the same variant ident must agree on field
/// shape — codegen panics on mismatch.
#[derive(Clone, Debug)]
pub struct OpcodeShape {
    /// PascalCase ident the per-arch enum uses for this variant.
    pub name: syn::Ident,
    /// Ordered field declarations. The codegen renders them as
    /// `name: type` inside the variant's struct-style payload.
    pub fields: Vec<(syn::Ident, syn::Type)>,
}

impl OpcodeShape {
    /// Build from a variant name and a list of (field_ident, type)
    /// pairs. Used inside Impls' `opcode_shape()` overrides.
    pub fn new(name: &str, fields: Vec<(&str, syn::Type)>) -> Self {
        let name_ident = syn::Ident::new(name, proc_macro2::Span::call_site());
        let fields = fields
            .into_iter()
            .map(|(f, ty)| {
                let f_ident = syn::Ident::new(f, proc_macro2::Span::call_site());
                (f_ident, ty)
            })
            .collect();
        Self {
            name: name_ident,
            fields,
        }
    }

    /// Sentinel shape returned by the trait default. The codegen
    /// checks for this and panics with the Impl's `name()` so a
    /// newly-added Impl can't silently bypass migration.
    pub(crate) fn unmigrated(impl_name: &str) -> Self {
        // The variant ident here is never actually used — codegen
        // detects the unmigrated state via `fan_out → None` long
        // before it would consume the shape. Pick a placeholder that
        // wouldn't collide with a real variant name by accident.
        let _ = impl_name;
        Self {
            name: syn::Ident::new("__Unmigrated", proc_macro2::Span::call_site()),
            fields: Vec::new(),
        }
    }
}

/// One concrete kernel-call instance an Impl emits at a given
/// (variant × workload-point). The codegen lowers this to
/// `<Arch>Op::<name> { #(field_n: <field_value>),* }` inside the
/// per-bucket static slice.
///
/// `field_values` is positional in the same order as
/// [`OpcodeShape::fields`]. Each entry is a `TokenStream` the
/// codegen drops verbatim into the constructor expression — useful
/// when a value is e.g. `slots.of(tile, slot) as u32` or a
/// pre-resolved literal.
#[derive(Clone, Debug)]
pub struct OpInstance {
    pub name: syn::Ident,
    pub field_values: Vec<TokenStream>,
}

impl OpInstance {
    /// Build with the variant ident matching `opcode_shape().name`
    /// and a vector of field-value token streams in declaration
    /// order.
    pub fn new(name: syn::Ident, field_values: Vec<TokenStream>) -> Self {
        Self { name, field_values }
    }
}

/// Compile-time mapping from `(TileId, output_slot)` → flat slot
/// index in the runtime tile table. Built once per (variant ×
/// workload-point) FUF by codegen and consumed by `fan_out` so it
/// can render `i32` slot ids into `Instruction` fields.
///
/// Indices are dense in `0..total()`. The runtime tile table is
/// allocated as `Vec<Option<TileEntry>>` of size `total()` and
/// indexed directly by these values.
///
/// Lives in the macro crate (not in `ferrite-forward`) because
/// it's a *compile-time* artifact: by the time a forward fn runs,
/// every i32 in the const `Instruction` array is already a flat
/// slot index. The runtime never sees a `(TileId, slot)` pair.
#[derive(Clone, Debug, Default)]
pub struct SlotMap {
    map: BTreeMap<(TileId, u8), u32>,
    total: u32,
}

impl SlotMap {
    pub fn new() -> Self {
        Self::default()
    }

    /// Insert `(tile, output_slot)` and assign the next dense
    /// index. Idempotent on repeats — same `(tile, output_slot)`
    /// keeps its first-assigned index.
    pub fn insert(&mut self, tile: TileId, output_slot: u8) -> u32 {
        let key = (tile, output_slot);
        if let Some(&existing) = self.map.get(&key) {
            return existing;
        }
        let idx = self.total;
        self.map.insert(key, idx);
        self.total += 1;
        idx
    }

    /// Insert `(tile, output_slot)` at a specific color. The colored
    /// slot map (linear-scan register allocation) walks tiles in
    /// def order and picks a color from the free pool, so two
    /// non-overlapping tiles can share an index. Use this instead of
    /// `insert` when the caller has already decided the color.
    /// `total()` ends up = 1 + max color seen.
    pub fn insert_at(&mut self, tile: TileId, output_slot: u8, color: u32) {
        self.map.insert((tile, output_slot), color);
        if color + 1 > self.total {
            self.total = color + 1;
        }
    }

    /// Resolve `(tile, output_slot)` → flat slot index. Panics
    /// when the pair was never inserted — codegen invariant.
    #[inline]
    pub fn of(&self, tile: TileId, output_slot: u8) -> u32 {
        match self.map.get(&(tile, output_slot)) {
            Some(&v) => v,
            None => panic!(
                "SlotMap::of: tile {tile:?} slot {output_slot} not registered \
                 — codegen forgot to insert this tile output before fan_out"
            ),
        }
    }

    /// Total number of slots allocated. Size of the runtime tile table.
    #[inline]
    pub fn total(&self) -> u32 {
        self.total
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

// ── Per-op emission bodies ───────────────────────────────────────
//
// These preserve the shape of the previous hardcoded OpKind match
// in codegen.rs. The kernel symbols they reference are approximate
// and cuda-gated; real ferrite-kernels bindings land as calibrated
// impls replace these reference entries.

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

/// Reference HostCallback impl for `OpKind::Embed`. Hand-written
/// because the host-interpreter `opcode_shape` / `fan_out`
/// overrides each reference an Impl-specific kernel symbol +
/// weight type that a shared macro can't express generically.
#[derive(Debug, Default)]
pub struct EmbedRefImpl;

impl Implementation for EmbedRefImpl {
    fn name(&self) -> &'static str {
        "embed_ref"
    }
    fn target_compatible(&self, _profile: &TargetProfile) -> bool {
        true
    }
    fn matches(&self, fuf: &Fuf, seed: TileId, _profile: &TargetProfile) -> Option<MatchInfo> {
        let info = single_tile_match(fuf, seed, OpKind::Embed)?;
        if let Some(s) = weight_storage_of(fuf.get(seed))
            && !matches!(s, StorageFormat::Dense)
        {
            return None;
        }
        Some(info)
    }
    fn cost_us(&self, m: &MatchInfo, ctx: &CostCtx) -> f64 {
        cost_embed(m, ctx)
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

    // ── Host-interpreter codegen ────────────────────────────────
    //
    // Variant `Embed { out_slot, weight_fn }`. Embed has no tile
    // inputs (`input_ids` is an Extern reachable via `ctx.input_ids`
    // ambient binding); its lone weight input is the embed_tokens
    // table. `weight_fn` is `Weights::embed_tokens` (un-layered,
    // ignores the layer arg); `out_slot` is the destination tile
    // table index.

    fn opcode_shape(&self) -> OpcodeShape {
        OpcodeShape::new(
            "Embed",
            vec![
                ("out_slot", syn::parse_quote!(u32)),
                (
                    "weight_fn",
                    syn::parse_quote!(
                        for<'a> fn(&'a Weights, u32) -> &'a ::ferrite_kernels::layers::Embedding
                    ),
                ),
            ],
        )
    }

    fn fan_out(
        &self,
        m: &MatchInfo,
        fuf: &Fuf,
        program: &Program,
        _bounds: &BTreeMap<String, u64>,
        slots: &SlotMap,
    ) -> Option<Vec<OpInstance>> {
        let tile = m.claimed_tiles[0];
        let out_slot = slots.of(tile, 0);
        let accessors = self.required_weights(&m.claimed_tiles, fuf, program);
        let acc = accessors
            .first()
            .expect("Embed: required_weights returned empty");
        let (base, _layer) = split_base_layer(&acc.name.to_string());
        let base_ident = syn::Ident::new(&base, proc_macro2::Span::call_site());
        Some(vec![OpInstance::new(
            syn::Ident::new("Embed", proc_macro2::Span::call_site()),
            vec![quote! { #out_slot }, quote! { Weights::#base_ident }],
        )])
    }
}
/// Reference HostCallback impl for `OpKind::RmsNorm`. Hand-written
/// (parallel to [`EmbedRefImpl`]) to expose `opcode_shape` /
/// `fan_out` overrides. Per-claim weight
/// selection — `input_layernorm` vs `post_attention_layernorm`,
/// each at any layer — rides on the `weight_fn` fn-pointer +
/// `layer` u32 fields.
#[derive(Debug, Default)]
pub struct RmsNormRefImpl;

impl Implementation for RmsNormRefImpl {
    fn name(&self) -> &'static str {
        "rmsnorm_ref"
    }
    fn target_compatible(&self, _profile: &TargetProfile) -> bool {
        true
    }
    fn matches(&self, fuf: &Fuf, seed: TileId, _profile: &TargetProfile) -> Option<MatchInfo> {
        let info = single_tile_match(fuf, seed, OpKind::RmsNorm)?;
        if let Some(s) = weight_storage_of(fuf.get(seed))
            && !matches!(s, StorageFormat::Dense)
        {
            return None;
        }
        Some(info)
    }
    fn cost_us(&self, m: &MatchInfo, ctx: &CostCtx) -> f64 {
        elementwise_cost(m, ctx)
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

    // ── Host-interpreter codegen ────────────────────────────────
    //
    // Variant `RmsNorm { in_slot, out_slot, layer, weight_fn }`.
    // Two RmsNorm tiles in the same arch share this Impl but can
    // bind different weight accessors (input_layernorm vs
    // post_attention_layernorm) and different layers — both ride
    // in OpInstance fields, leaving the variant declaration
    // structurally identical across all RmsNorm claims.

    fn opcode_shape(&self) -> OpcodeShape {
        OpcodeShape::new(
            "RmsNorm",
            vec![
                ("in_slot", syn::parse_quote!(u32)),
                ("out_slot", syn::parse_quote!(u32)),
                ("layer", syn::parse_quote!(u32)),
                (
                    "weight_fn",
                    syn::parse_quote!(
                        for<'a> fn(&'a Weights, u32) -> &'a ::ferrite_kernels::layers::RmsNorm
                    ),
                ),
            ],
        )
    }

    fn fan_out(
        &self,
        m: &MatchInfo,
        fuf: &Fuf,
        program: &Program,
        _bounds: &BTreeMap<String, u64>,
        slots: &SlotMap,
    ) -> Option<Vec<OpInstance>> {
        let tile = m.claimed_tiles[0];
        let node = fuf.get(tile);
        let (in_id, in_slot) = match node.inputs.first() {
            Some(FufInput::Tile { id, slot }) => (*id, *slot),
            other => panic!(
                "RmsNorm: first input must be a Tile (got {other:?}); \
                 the FUF tile shape doesn't match what fan_out expects"
            ),
        };
        let in_slot_idx = slots.of(in_id, in_slot);
        let out_slot_idx = slots.of(tile, 0);
        let accessors = self.required_weights(&m.claimed_tiles, fuf, program);
        let acc = accessors
            .first()
            .expect("RmsNorm: required_weights returned empty");
        let (base, layer) = split_base_layer(&acc.name.to_string());
        let layer = layer.unwrap_or(0) as u32;
        let base_ident = syn::Ident::new(&base, proc_macro2::Span::call_site());
        Some(vec![OpInstance::new(
            syn::Ident::new("RmsNorm", proc_macro2::Span::call_site()),
            vec![
                quote! { #in_slot_idx },
                quote! { #out_slot_idx },
                quote! { #layer },
                quote! { Weights::#base_ident },
            ],
        )])
    }
}

// ── DC sibling Impls ─────────────────────────────────────────────
//
// Each DC sibling pairs 1:1 with a host Impl that already has a
// device-callable kernel sibling in `vllm-cuda/csrc/megakernel/`
// (`dc_rms_norm`, `dc_fused_add_rms_norm`, `dc_fused_qkv_rope_cache`,
// `dc_cutlass::dc_gemm`). Same FUF claim pattern, same
// `opcode_shape`, same `fan_out` — codegen merges the host + DC
// siblings into one per-arch enum variant, and the host interpreter
// + (future) PrimMega encoder both consume the variant through
// their respective dispatch. Differences are surface-only:
//
//   - `launch_kind`            HostCallback → DeviceCallable
//   - `megakernel_fit`         None → Primitive
//   - `target_compatible`      gated on `prim_mega_compatible()`
//   - `supported_*_handoffs`   mega-internal only
//
// All other behavior (`matches`, `cost_us`, `resources`,
// `input_layouts`, `output_layouts`, `output_alias`,
// `consumes_input_tiles`, `required_weights`, `opcode_shape`,
// `fan_out`) delegates to the host counterpart so a regression in
// the host pattern propagates automatically. The
// `dc_*_share_opcode_shape` tests pin the variant-merge contract
// codegen depends on.
//
// Why a sibling rather than flipping `RmsNormRefImpl`'s
// `launch_kind`: an Impl has exactly one `launch_kind`, and the
// host kernel's `KernelBoundary` / `StreamOrder` handoffs are
// different physical machinery from the megakernel's
// `InKernelGridSync` / `SyncThreads`. Two siblings let the solver
// score each on its own merits — Phase 1 hardware naturally keeps
// host cheaper because `InKernelGridSync = 100us` outweighs
// `KernelBoundary = 5us` per edge; Phase 2 hardware
// (mbarrier-capable) flips the math.

/// Mega-internal handoffs supported by every DC sibling. Module
/// constant so all DC Impls reference the same slice.
const DC_HANDOFFS: &[Handoff] = &[Handoff::InKernelGridSync, Handoff::SyncThreads];

#[derive(Debug, Default)]
pub struct DcRmsNormImpl;

impl Implementation for DcRmsNormImpl {
    fn name(&self) -> &'static str {
        "dc_rmsnorm"
    }
    fn target_compatible(&self, profile: &TargetProfile) -> bool {
        profile.prim_mega_compatible() && RmsNormRefImpl.target_compatible(profile)
    }
    fn matches(&self, fuf: &Fuf, seed: TileId, profile: &TargetProfile) -> Option<MatchInfo> {
        RmsNormRefImpl.matches(fuf, seed, profile)
    }
    fn cost_us(&self, m: &MatchInfo, ctx: &CostCtx) -> f64 {
        RmsNormRefImpl.cost_us(m, ctx)
    }
    fn resources(&self, m: &MatchInfo) -> Resources {
        RmsNormRefImpl.resources(m)
    }
    fn launch_kind(&self) -> LaunchKind {
        LaunchKind::DeviceCallable
    }
    fn megakernel_fit(&self) -> MegakernelFit {
        MegakernelFit::Primitive
    }
    fn supported_input_handoffs(&self) -> &[Handoff] {
        DC_HANDOFFS
    }
    fn supported_output_handoffs(&self) -> &[Handoff] {
        DC_HANDOFFS
    }
    fn input_layouts(&self, m: &MatchInfo) -> Vec<Layout> {
        RmsNormRefImpl.input_layouts(m)
    }
    fn output_layouts(&self, m: &MatchInfo) -> Vec<Layout> {
        RmsNormRefImpl.output_layouts(m)
    }
    fn is_compute_bound(&self) -> bool {
        RmsNormRefImpl.is_compute_bound()
    }
    fn can_share_kernel_with(&self, _other: &dyn Implementation) -> bool {
        // Two DeviceCallable impls in the same persistent kernel
        // share a CTA; they tolerate co-residency by definition.
        true
    }
    fn output_alias(
        &self,
        claimed_tiles: &[TileId],
        fuf: &Fuf,
    ) -> Vec<((TileId, u8), Option<(TileId, u8)>)> {
        RmsNormRefImpl.output_alias(claimed_tiles, fuf)
    }
    fn consumes_input_tiles(&self, claimed_tiles: &[TileId], fuf: &Fuf) -> Vec<(TileId, u8)> {
        RmsNormRefImpl.consumes_input_tiles(claimed_tiles, fuf)
    }
    fn required_weights(
        &self,
        claimed_tiles: &[TileId],
        fuf: &Fuf,
        program: &Program,
    ) -> Vec<WeightAccessor> {
        RmsNormRefImpl.required_weights(claimed_tiles, fuf, program)
    }
    fn opcode_shape(&self) -> OpcodeShape {
        RmsNormRefImpl.opcode_shape()
    }
    fn fan_out(
        &self,
        m: &MatchInfo,
        fuf: &Fuf,
        program: &Program,
        bounds: &BTreeMap<String, u64>,
        slots: &SlotMap,
    ) -> Option<Vec<OpInstance>> {
        RmsNormRefImpl.fan_out(m, fuf, program, bounds, slots)
    }
}

#[derive(Debug, Default)]
pub struct DcFusedAddRmsNormImpl;

impl Implementation for DcFusedAddRmsNormImpl {
    fn name(&self) -> &'static str {
        "dc_fused_add_rms_norm"
    }
    fn target_compatible(&self, profile: &TargetProfile) -> bool {
        profile.prim_mega_compatible() && FusedAddRmsNormImpl.target_compatible(profile)
    }
    fn matches(&self, fuf: &Fuf, seed: TileId, profile: &TargetProfile) -> Option<MatchInfo> {
        FusedAddRmsNormImpl.matches(fuf, seed, profile)
    }
    fn cost_us(&self, m: &MatchInfo, ctx: &CostCtx) -> f64 {
        FusedAddRmsNormImpl.cost_us(m, ctx)
    }
    fn resources(&self, m: &MatchInfo) -> Resources {
        FusedAddRmsNormImpl.resources(m)
    }
    fn launch_kind(&self) -> LaunchKind {
        LaunchKind::DeviceCallable
    }
    fn megakernel_fit(&self) -> MegakernelFit {
        MegakernelFit::Primitive
    }
    fn supported_input_handoffs(&self) -> &[Handoff] {
        DC_HANDOFFS
    }
    fn supported_output_handoffs(&self) -> &[Handoff] {
        DC_HANDOFFS
    }
    fn input_layouts(&self, m: &MatchInfo) -> Vec<Layout> {
        FusedAddRmsNormImpl.input_layouts(m)
    }
    fn output_layouts(&self, m: &MatchInfo) -> Vec<Layout> {
        FusedAddRmsNormImpl.output_layouts(m)
    }
    fn is_compute_bound(&self) -> bool {
        FusedAddRmsNormImpl.is_compute_bound()
    }
    fn can_share_kernel_with(&self, _other: &dyn Implementation) -> bool {
        true
    }
    fn output_alias(
        &self,
        claimed_tiles: &[TileId],
        fuf: &Fuf,
    ) -> Vec<((TileId, u8), Option<(TileId, u8)>)> {
        FusedAddRmsNormImpl.output_alias(claimed_tiles, fuf)
    }
    fn consumes_input_tiles(&self, claimed_tiles: &[TileId], fuf: &Fuf) -> Vec<(TileId, u8)> {
        FusedAddRmsNormImpl.consumes_input_tiles(claimed_tiles, fuf)
    }
    fn required_weights(
        &self,
        claimed_tiles: &[TileId],
        fuf: &Fuf,
        program: &Program,
    ) -> Vec<WeightAccessor> {
        FusedAddRmsNormImpl.required_weights(claimed_tiles, fuf, program)
    }
    fn opcode_shape(&self) -> OpcodeShape {
        FusedAddRmsNormImpl.opcode_shape()
    }
    fn fan_out(
        &self,
        m: &MatchInfo,
        fuf: &Fuf,
        program: &Program,
        bounds: &BTreeMap<String, u64>,
        slots: &SlotMap,
    ) -> Option<Vec<OpInstance>> {
        FusedAddRmsNormImpl.fan_out(m, fuf, program, bounds, slots)
    }
}

#[derive(Debug, Default)]
pub struct DcFusedQkvRopeCacheImpl;

impl Implementation for DcFusedQkvRopeCacheImpl {
    fn name(&self) -> &'static str {
        "dc_fused_qkv_rope_cache"
    }
    fn target_compatible(&self, profile: &TargetProfile) -> bool {
        profile.prim_mega_compatible() && FusedQkvRopeCacheImpl.target_compatible(profile)
    }
    fn workload_constraint(&self) -> WorkloadConstraint {
        FusedQkvRopeCacheImpl.workload_constraint()
    }
    fn matches(&self, fuf: &Fuf, seed: TileId, profile: &TargetProfile) -> Option<MatchInfo> {
        FusedQkvRopeCacheImpl.matches(fuf, seed, profile)
    }
    fn cost_us(&self, m: &MatchInfo, ctx: &CostCtx) -> f64 {
        FusedQkvRopeCacheImpl.cost_us(m, ctx)
    }
    fn resources(&self, m: &MatchInfo) -> Resources {
        FusedQkvRopeCacheImpl.resources(m)
    }
    fn launch_kind(&self) -> LaunchKind {
        LaunchKind::DeviceCallable
    }
    fn megakernel_fit(&self) -> MegakernelFit {
        MegakernelFit::Primitive
    }
    fn supported_input_handoffs(&self) -> &[Handoff] {
        DC_HANDOFFS
    }
    fn supported_output_handoffs(&self) -> &[Handoff] {
        DC_HANDOFFS
    }
    fn input_layouts(&self, m: &MatchInfo) -> Vec<Layout> {
        FusedQkvRopeCacheImpl.input_layouts(m)
    }
    fn output_layouts(&self, m: &MatchInfo) -> Vec<Layout> {
        FusedQkvRopeCacheImpl.output_layouts(m)
    }
    fn is_compute_bound(&self) -> bool {
        FusedQkvRopeCacheImpl.is_compute_bound()
    }
    fn can_share_kernel_with(&self, _other: &dyn Implementation) -> bool {
        true
    }
    fn output_alias(
        &self,
        claimed_tiles: &[TileId],
        fuf: &Fuf,
    ) -> Vec<((TileId, u8), Option<(TileId, u8)>)> {
        FusedQkvRopeCacheImpl.output_alias(claimed_tiles, fuf)
    }
    fn consumes_input_tiles(&self, claimed_tiles: &[TileId], fuf: &Fuf) -> Vec<(TileId, u8)> {
        FusedQkvRopeCacheImpl.consumes_input_tiles(claimed_tiles, fuf)
    }
    fn required_weights(
        &self,
        claimed_tiles: &[TileId],
        fuf: &Fuf,
        program: &Program,
    ) -> Vec<WeightAccessor> {
        FusedQkvRopeCacheImpl.required_weights(claimed_tiles, fuf, program)
    }
    fn opcode_shape(&self) -> OpcodeShape {
        FusedQkvRopeCacheImpl.opcode_shape()
    }
    fn fan_out(
        &self,
        m: &MatchInfo,
        fuf: &Fuf,
        program: &Program,
        bounds: &BTreeMap<String, u64>,
        slots: &SlotMap,
    ) -> Option<Vec<OpInstance>> {
        FusedQkvRopeCacheImpl.fan_out(m, fuf, program, bounds, slots)
    }
}

/// Reference HostCallback impl for `OpKind::LayerNorm`. Hand-written
/// (parallel to [`RmsNormRefImpl`]) to expose the host-interpreter
/// `opcode_shape` / `fan_out` overrides. Same
/// per-claim weight selector pattern: `weight_fn` + `layer` fields
/// pick the right `CohereLayerNorm` accessor at runtime; the variant
/// declaration is structurally identical for every LayerNorm tile in
/// the arch (CommandR has one — the singular `layer_norm` per block —
/// but the shape stays uniform with RmsNorm so the Impl reads the
/// same way to a future maintainer).
#[derive(Debug, Default)]
pub struct LayerNormRefImpl;

impl Implementation for LayerNormRefImpl {
    fn name(&self) -> &'static str {
        "layer_norm_ref"
    }
    fn target_compatible(&self, _profile: &TargetProfile) -> bool {
        true
    }
    fn matches(&self, fuf: &Fuf, seed: TileId, _profile: &TargetProfile) -> Option<MatchInfo> {
        let info = single_tile_match(fuf, seed, OpKind::LayerNorm)?;
        if let Some(s) = weight_storage_of(fuf.get(seed))
            && !matches!(s, StorageFormat::Dense)
        {
            return None;
        }
        Some(info)
    }
    fn cost_us(&self, m: &MatchInfo, ctx: &CostCtx) -> f64 {
        elementwise_cost(m, ctx)
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

    // ── Host-interpreter codegen ────────────────────────────────
    //
    // Variant `LayerNorm { in_slot, out_slot, layer, weight_fn }`.
    // Identical shape to RmsNorm — different kernel symbol
    // (`cohere_layer_norm`) and different weight wrapper type
    // (`CohereLayerNorm`).

    fn opcode_shape(&self) -> OpcodeShape {
        OpcodeShape::new(
            "LayerNorm",
            vec![
                ("in_slot", syn::parse_quote!(u32)),
                ("out_slot", syn::parse_quote!(u32)),
                ("layer", syn::parse_quote!(u32)),
                (
                    "weight_fn",
                    syn::parse_quote!(
                        for<'a> fn(
                            &'a Weights,
                            u32,
                        )
                            -> &'a ::ferrite_kernels::layers::CohereLayerNorm
                    ),
                ),
            ],
        )
    }

    fn fan_out(
        &self,
        m: &MatchInfo,
        fuf: &Fuf,
        program: &Program,
        _bounds: &BTreeMap<String, u64>,
        slots: &SlotMap,
    ) -> Option<Vec<OpInstance>> {
        let tile = m.claimed_tiles[0];
        let node = fuf.get(tile);
        let (in_id, in_slot) = match node.inputs.first() {
            Some(FufInput::Tile { id, slot }) => (*id, *slot),
            other => panic!(
                "LayerNorm: first input must be a Tile (got {other:?}); \
                 the FUF tile shape doesn't match what fan_out expects"
            ),
        };
        let in_slot_idx = slots.of(in_id, in_slot);
        let out_slot_idx = slots.of(tile, 0);
        let accessors = self.required_weights(&m.claimed_tiles, fuf, program);
        let acc = accessors
            .first()
            .expect("LayerNorm: required_weights returned empty");
        let (base, layer) = split_base_layer(&acc.name.to_string());
        let layer = layer.unwrap_or(0) as u32;
        let base_ident = syn::Ident::new(&base, proc_macro2::Span::call_site());
        Some(vec![OpInstance::new(
            syn::Ident::new("LayerNorm", proc_macro2::Span::call_site()),
            vec![
                quote! { #in_slot_idx },
                quote! { #out_slot_idx },
                quote! { #layer },
                quote! { Weights::#base_ident },
            ],
        )])
    }
}

/// Reference HostCallback impl for `OpKind::Gemm`. Hand-written
/// (parallel to [`RmsNormRefImpl`]) to expose the host-interpreter
/// `opcode_shape` / `fan_out` overrides. Bias
/// is handled by [`FusedGemmBiasImpl`]; this Impl claims only the
/// strict-matmul DSL `gemm()`.
///
/// One Gemm tile may bind any layer of any per-block accessor (`q_proj`
/// / `k_proj` / `v_proj` / `o_proj` / `lm_head` / …) — same
/// `weight_fn` + `layer` pattern as RmsNorm.
#[derive(Debug, Default)]
pub struct GemmRefImpl;

impl Implementation for GemmRefImpl {
    fn name(&self) -> &'static str {
        "gemm_ref"
    }
    fn target_compatible(&self, _profile: &TargetProfile) -> bool {
        true
    }
    fn matches(&self, fuf: &Fuf, seed: TileId, _profile: &TargetProfile) -> Option<MatchInfo> {
        let info = single_tile_match(fuf, seed, OpKind::Gemm)?;
        if let Some(s) = weight_storage_of(fuf.get(seed))
            && !matches!(s, StorageFormat::Dense)
        {
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

    // ── Host-interpreter codegen ────────────────────────────────
    //
    // Variant `Gemm { in_slot, out_slot, layer, weight_fn }`.
    // `weight_fn` resolves to `Weights::<base>` where `<base>` is
    // the projection name (`q_proj`, `lm_head`, …); `layer` is
    // ignored for un-layered accessors per
    // `emit_weights_accessor_methods`'s contract.

    fn opcode_shape(&self) -> OpcodeShape {
        OpcodeShape::new(
            "Gemm",
            vec![
                ("in_slot", syn::parse_quote!(u32)),
                ("out_slot", syn::parse_quote!(u32)),
                ("layer", syn::parse_quote!(u32)),
                (
                    "weight_fn",
                    syn::parse_quote!(
                        for<'a> fn(&'a Weights, u32) -> &'a ::ferrite_kernels::layers::LinearLayer
                    ),
                ),
                ("n", syn::parse_quote!(u32)),
                ("k", syn::parse_quote!(u32)),
            ],
        )
    }

    fn fan_out(
        &self,
        m: &MatchInfo,
        fuf: &Fuf,
        program: &Program,
        bounds: &BTreeMap<String, u64>,
        slots: &SlotMap,
    ) -> Option<Vec<OpInstance>> {
        let tile = m.claimed_tiles[0];
        let node = fuf.get(tile);
        let (in_id, in_slot) = match node.inputs.first() {
            Some(FufInput::Tile { id, slot }) => (*id, *slot),
            other => panic!(
                "Gemm: first input must be a Tile (got {other:?}); \
                 the FUF tile shape doesn't match what fan_out expects"
            ),
        };
        let in_slot_idx = slots.of(in_id, in_slot);
        let out_slot_idx = slots.of(tile, 0);
        let accessors = self.required_weights(&m.claimed_tiles, fuf, program);
        let acc = accessors
            .first()
            .expect("Gemm: required_weights returned empty");
        let (base, layer) = split_base_layer(&acc.name.to_string());
        let layer = layer.unwrap_or(0) as u32;
        let base_ident = syn::Ident::new(&base, proc_macro2::Span::call_site());
        let (n, k) = gemm_nk_from_fuf(fuf, node, bounds)
            .expect("Gemm: weight (N, K) must resolve from FUF + bounds at fan_out time");
        Some(vec![OpInstance::new(
            syn::Ident::new("Gemm", proc_macro2::Span::call_site()),
            vec![
                quote! { #in_slot_idx },
                quote! { #out_slot_idx },
                quote! { #layer },
                quote! { Weights::#base_ident },
                quote! { #n },
                quote! { #k },
            ],
        )])
    }
}

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

    // ── Host-interpreter codegen ────────────────────────────────
    //
    // Variant `Reshape { in_slot, out_slot, dims_lit, dims_nt_pow,
    // ndim }`. Each output axis is `dims_lit[i] * num_tokens^dims_nt_pow[i]`,
    // matching today's `reshape_dim_token` lowering: every config
    // bound + literal folds into `dims_lit[i]` at codegen time, and
    // each occurrence of the runtime bound `num_tokens` increments
    // `dims_nt_pow[i]` (in practice 0 or 1 per axis). The arm body
    // computes the final shape at runtime and writes a
    // `TileEntry::Reshaped { ref_slot: in_slot, tensor }` so the
    // drop-pass keeps `in_slot`'s `OwnedTensor` alive while any
    // downstream consumer holds the reshape.

    fn opcode_shape(&self) -> OpcodeShape {
        OpcodeShape::new(
            "Reshape",
            vec![
                ("in_slot", syn::parse_quote!(u32)),
                ("out_slot", syn::parse_quote!(u32)),
                (
                    "dims_lit",
                    syn::parse_quote!([u32; ::ferrite_cuda_core::tensor::MAX_DIMS]),
                ),
                (
                    "dims_nt_pow",
                    syn::parse_quote!([u8; ::ferrite_cuda_core::tensor::MAX_DIMS]),
                ),
                ("ndim", syn::parse_quote!(u8)),
            ],
        )
    }

    fn fan_out(
        &self,
        m: &MatchInfo,
        fuf: &Fuf,
        _program: &Program,
        bounds: &BTreeMap<String, u64>,
        slots: &SlotMap,
    ) -> Option<Vec<OpInstance>> {
        let tile = m.claimed_tiles[0];
        let node = fuf.get(tile);
        let (in_id, in_slot) = match node.inputs.first() {
            Some(FufInput::Tile { id, slot }) => (*id, *slot),
            other => panic!(
                "Reshape: first input must be a Tile (got {other:?}); \
                 the FUF tile shape doesn't match what fan_out expects"
            ),
        };
        let in_slot_idx = slots.of(in_id, in_slot);
        let out_slot_idx = slots.of(tile, 0);
        let shape = &node.outputs[0];
        // Hardcoded 4 — mirrors `ferrite_cuda_core::tensor::MAX_DIMS`.
        // The proc-macro crate doesn't depend on `ferrite-cuda-core`
        // (only emits its tokens into consumer code), so the const
        // isn't reachable from here. `Instruction::Reshape`'s
        // payload (in ferrite-forward) references `MAX_DIMS`
        // symbolically; if it ever changes from 4, both this `4` and
        // that variant must be updated together.
        const RESHAPE_MAX_DIMS: usize = 4;
        assert!(
            shape.len() <= RESHAPE_MAX_DIMS,
            "Reshape: target shape has {} dims; tile table's GpuTensor \
             carries at most {RESHAPE_MAX_DIMS} (mirrors \
             ferrite_cuda_core::tensor::MAX_DIMS). Tile id {:?}",
            shape.len(),
            tile
        );
        let mut dims_lit = [1u32; RESHAPE_MAX_DIMS];
        let mut dims_nt_pow = [0u8; RESHAPE_MAX_DIMS];
        for (i, d) in shape.iter().enumerate() {
            let (lit, nt_pow) = decompose_reshape_dim(d, bounds);
            dims_lit[i] = lit;
            dims_nt_pow[i] = nt_pow;
        }
        let ndim = shape.len() as u8;
        let dims_lit_toks = dims_lit.iter().map(|v| quote! { #v });
        let dims_nt_pow_toks = dims_nt_pow.iter().map(|v| quote! { #v });
        Some(vec![OpInstance::new(
            syn::Ident::new("Reshape", proc_macro2::Span::call_site()),
            vec![
                quote! { #in_slot_idx },
                quote! { #out_slot_idx },
                quote! { [ #( #dims_lit_toks ),* ] },
                quote! { [ #( #dims_nt_pow_toks ),* ] },
                quote! { #ndim },
            ],
        )])
    }
}

/// Decompose a `Dim` into `(literal_factor, num_tokens_power)` for
/// the host-interpreter Reshape opcode. Mirrors `reshape_dim_token`'s
/// lowering: `Lit` and non-`num_tokens` `Bound` fold into the literal
/// factor at codegen time (via `bounds`); each occurrence of
/// `Bound("num_tokens")` increments the `num_tokens` power. `Mul`
/// recurses with multiplicative composition. `Var` is a compiler bug
/// — shape inference should have closed every dim.
fn decompose_reshape_dim(d: &crate::shape::Dim, bounds: &BTreeMap<String, u64>) -> (u32, u8) {
    use crate::shape::Dim;
    match d {
        Dim::Lit(n) => (*n as u32, 0),
        Dim::Bound(name) if name == "num_tokens" => (1, 1),
        Dim::Bound(name) => {
            let v = *bounds.get(name).unwrap_or_else(|| {
                panic!(
                    "Reshape: bound `{name}` not found in bounds table — \
                     the model config is incomplete"
                )
            });
            (v as u32, 0)
        }
        Dim::Mul(factors) => {
            let mut lit: u64 = 1;
            let mut nt_pow: u8 = 0;
            for f in factors {
                let (l, p) = decompose_reshape_dim(f, bounds);
                lit *= l as u64;
                nt_pow += p;
            }
            (lit as u32, nt_pow)
        }
        Dim::Var(_) => {
            panic!("Reshape target dim must be closed (no Var) — compiler bug")
        }
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
    // PrimMega sibling to RmsNormRefImpl. `target_compatible` gates
    // on `prim_mega_compatible()`, so this entry stays dormant on
    // pre-Ampere targets and is solver-feasible only when the
    // megakernel `.cu` is linkable. See `MEGA_HANDOFF.md`.
    lib.push(Box::new(DcRmsNormImpl));
    lib.push(Box::new(LayerNormRefImpl));
    // A/B hook: setting `FERRITE_DISABLE_CUBLAS_GEMM=1` at proc-macro
    // expansion time (i.e. when `forward!` runs during a build) drops
    // `GemmRefImpl` from the library, forcing every singleton Gemm
    // tile onto CUTLASS — tile zoo, SplitK, or GEMV. Fused impls that
    // call cuBLAS internally are unaffected. Use this to benchmark
    // "CUTLASS-only dispatch" vs the DP's default cost-driven mix.
    if std::env::var_os("FERRITE_DISABLE_CUBLAS_GEMM").is_none() {
        lib.push(Box::new(GemmRefImpl));
    }
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
    // CUTLASS bias-add peer to FusedGemmBiasImpl — plain
    // `cutlass::gemm::device::Gemm` with `LinearCombination` epilogue
    // and bias passed as the C operand at ldc=0 (row broadcast). One
    // Impl per CUTLASS_TILE_ZOO entry, gated per-tile on calibrated
    // CSV row presence; the DP picks the best tile per workload.
    for tile in CUTLASS_TILE_ZOO {
        lib.push(Box::new(CutlassFusedGemmBiasImpl {
            tile_m: tile.0,
            tile_n: tile.1,
            stages: tile.2,
        }));
    }
    lib.push(Box::new(FusedGateUpSiluMulImpl));
    // CUTLASS EVT peer to FusedGateUpSiluMulImpl — same claim, different
    // kernel shape. Solver's DP picks whichever has lower calibrated
    // cost per bucket; `target_compatible` gates on CSV row presence.
    lib.push(Box::new(CutlassFusedGateUpSiluMulImpl));
    lib.push(Box::new(FusedGateUpGeluMulImpl));
    // CUTLASS peer to FusedGateUpGeluMulImpl. Mirrors the cuBLAS path
    // structurally — packed CUTLASS GEMM at (M, 2I, K) followed by
    // BW-bound `gelu_and_mul_fused` — picking the GEMM tile per (M,
    // 2I, K) bucket. One Impl per CUTLASS_TILE_ZOO entry.
    for tile in CUTLASS_TILE_ZOO {
        lib.push(Box::new(CutlassFusedGateUpGeluMulImpl {
            tile_m: tile.0,
            tile_n: tile.1,
            stages: tile.2,
        }));
    }
    lib.push(Box::new(FusedAddRmsNormImpl));
    // PrimMega sibling — same claim, identical fan_out, mega-internal
    // handoffs only. Solver-feasible only on prim_mega-compatible
    // targets.
    lib.push(Box::new(DcFusedAddRmsNormImpl));
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
    // PrimMega sibling — `dc_fused_qkv_rope_cache` is wired in
    // prim_mega.cu. The prefill variant has no DC sibling yet
    // because no `dc_fused_qkv_rope_prefill` exists in
    // megakernel_ops.cuh.
    lib.push(Box::new(DcFusedQkvRopeCacheImpl));
    lib.push(Box::new(FusedQkvRopePrefillImpl));
    // CUTLASS peers — packed CUTLASS GEMM at (M, q+2*kv, hidden)
    // followed by the same fused_qkv_rope_cache / fused_qkv_rope
    // post-pass. One Impl per CUTLASS_TILE_ZOO entry per variant;
    // target_compatible gates each on calibrated CSV row presence.
    // matches() rejects biased claims (qwen2 keeps cuBLAS until the
    // bias-zoo CSV is shape-swept).
    for tile in CUTLASS_TILE_ZOO {
        lib.push(Box::new(CutlassFusedQkvRopeCacheImpl {
            tile_m: tile.0,
            tile_n: tile.1,
            stages: tile.2,
        }));
        lib.push(Box::new(CutlassFusedQkvRopePrefillImpl {
            tile_m: tile.0,
            tile_n: tile.1,
            stages: tile.2,
        }));
    }
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
    // PrimMega siblings — one per CUTLASS_TILE_ZOO entry, identical
    // tile set to the host registration above. Each delegates
    // `cost_us` to the host CutlassGemm sibling, so the SAME CSV row
    // drives both — solver scoring is apples-to-apples and the only
    // divergence is launch_kind + handoff cost (which lives at the
    // edge level). Drift between host and DC tile sets is impossible
    // because both walk the same list; the C++ X-macro is pinned to
    // it by the `dc_zoo_equals_host_zoo` test in this module.
    for tile in CUTLASS_TILE_ZOO {
        lib.push(Box::new(DcCutlassGemmImpl {
            tile_m: tile.0,
            tile_n: tile.1,
            stages: tile.2,
        }));
    }
    // CUTLASS GEMM + residual-add peer: beta=1.0 epilogue, in-place
    // on residual. 2-tile claim over `(Gemm, Add)`; DP picks over
    // FusedAddRmsNormImpl per layer-residual chain by cost.
    for tile in CUTLASS_TILE_ZOO {
        lib.push(Box::new(CutlassGemmAddImpl {
            tile_m: tile.0,
            tile_n: tile.1,
            stages: tile.2,
        }));
    }
    // PrimMega sibling for CutlassGemmAdd. Iterates the same
    // CUTLASS_TILE_ZOO so drift between host + DC tile sets is
    // structurally impossible. Same `dc_gemm` C++ template the bare
    // GEMM uses — the residual-add is a runtime `beta=1.0` only.
    for tile in CUTLASS_TILE_ZOO {
        lib.push(Box::new(DcCutlassGemmAddImpl {
            tile_m: tile.0,
            tile_n: tile.1,
            stages: tile.2,
        }));
    }
    // CUTLASS SplitK parallel — one Impl per (tile, split_k) tuple.
    // Closes the small-N/large-K tall-skinny shape class where the
    // standard tile zoo leaves cuBLAS winning. target_compatible
    // gates on CSV row presence, so uncalibrated targets skip
    // these variants silently.
    for &(tm, tn, st, sk) in CUTLASS_SPLITK_ZOO {
        lib.push(Box::new(CutlassGemmSplitKImpl {
            tile_m: tm,
            tile_n: tn,
            stages: st,
            split_k: sk,
        }));
    }

    lib.push(Box::new(CutlassGemvImpl));
    lib.push(Box::new(DcCutlassGemvImpl));

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

    // ── DeepSeek MLA + MoE ops ───────────────────────────────────
    // MLA split (kv_a → kv_latent + k_pe), full MLA attention
    // sequence, and DeepSeekV2MoE (routed + shared expert).
    lib.push(Box::new(MlaSplitRefImpl));
    lib.push(Box::new(MlaAttentionImpl));
    lib.push(Box::new(DeepSeekMoeRefImpl));

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

    // ── Host-interpreter codegen ────────────────────────────────
    //
    // Variant `FusedGemmBias { in_slot, out_slot, layer, weight_fn }`.
    // The `(Gemm, BiasAdd)` claim collapses to one kernel call against
    // a single `LinearLayer` accessor (the loader detects `<prefix>.bias`
    // and stashes it on the dense layer). `LinearLayer::forward`
    // dispatches to `gemm_bias` for dense bf16 weights.

    fn opcode_shape(&self) -> OpcodeShape {
        OpcodeShape::new(
            "FusedGemmBias",
            vec![
                ("in_slot", syn::parse_quote!(u32)),
                ("out_slot", syn::parse_quote!(u32)),
                ("layer", syn::parse_quote!(u32)),
                (
                    "weight_fn",
                    syn::parse_quote!(
                        for<'a> fn(&'a Weights, u32) -> &'a ::ferrite_kernels::layers::LinearLayer
                    ),
                ),
            ],
        )
    }

    fn fan_out(
        &self,
        m: &MatchInfo,
        fuf: &Fuf,
        program: &Program,
        _bounds: &BTreeMap<String, u64>,
        slots: &SlotMap,
    ) -> Option<Vec<OpInstance>> {
        let gemm_id = *m
            .claimed_tiles
            .iter()
            .find(|t| fuf.get(**t).op == OpKind::Gemm)
            .expect("FusedGemmBias: claim contains Gemm");
        let bias_id = *m
            .claimed_tiles
            .iter()
            .find(|t| fuf.get(**t).op == OpKind::BiasAdd)
            .expect("FusedGemmBias: claim contains BiasAdd");
        let gemm_node = fuf.get(gemm_id);
        let (in_id, in_slot) = match gemm_node.inputs.first() {
            Some(FufInput::Tile { id, slot }) => (*id, *slot),
            other => panic!("FusedGemmBias: gemm's first input must be a Tile (got {other:?})"),
        };
        let in_slot_idx = slots.of(in_id, in_slot);
        let out_slot_idx = slots.of(bias_id, 0);
        let accessors = self.required_weights(&m.claimed_tiles, fuf, program);
        let acc = accessors
            .first()
            .expect("FusedGemmBias: required_weights returned empty");
        let (base, layer) = split_base_layer(&acc.name.to_string());
        let layer = layer.unwrap_or(0) as u32;
        let base_ident = syn::Ident::new(&base, proc_macro2::Span::call_site());
        Some(vec![OpInstance::new(
            syn::Ident::new("FusedGemmBias", proc_macro2::Span::call_site()),
            vec![
                quote! { #in_slot_idx },
                quote! { #out_slot_idx },
                quote! { #layer },
                quote! { Weights::#base_ident },
            ],
        )])
    }
}

// ── CutlassFusedGemmBiasImpl ─────────────────────────────────────
//
// CUTLASS peer to `FusedGemmBiasImpl`. Same 2-tile `(Gemm, BiasAdd)`
// claim; emits `cutlass_gemm_bias` (plain `cutlass::gemm::device::Gemm`
// with `LinearCombination` epilogue, bias passed as C at ldc=0 for row
// broadcast). One Impl per CUTLASS_TILE_ZOO entry, parameterised by
// `(tile_m, tile_n, stages)`; per-tile `target_compatible` gates on
// calibrated CSV row presence so the DP picks the best tile per
// (workload M, weight N, K) bucket.

#[derive(Debug, Clone)]
pub struct CutlassFusedGemmBiasImpl {
    pub tile_m: u32,
    pub tile_n: u32,
    pub stages: u32,
}

impl CutlassFusedGemmBiasImpl {
    fn csv_name(&self) -> &'static str {
        self.static_name()
    }

    fn static_name(&self) -> &'static str {
        match (self.tile_m, self.tile_n, self.stages) {
            (16, 64, 3) => "cutlass_gemm_bias_16x64_s3",
            (16, 64, 4) => "cutlass_gemm_bias_16x64_s4",
            (16, 128, 3) => "cutlass_gemm_bias_16x128_s3",
            (16, 128, 4) => "cutlass_gemm_bias_16x128_s4",
            (32, 64, 3) => "cutlass_gemm_bias_32x64_s3",
            (32, 64, 4) => "cutlass_gemm_bias_32x64_s4",
            (32, 128, 3) => "cutlass_gemm_bias_32x128_s3",
            (32, 128, 4) => "cutlass_gemm_bias_32x128_s4",
            (32, 256, 3) => "cutlass_gemm_bias_32x256_s3",
            (64, 64, 3) => "cutlass_gemm_bias_64x64_s3",
            (64, 64, 4) => "cutlass_gemm_bias_64x64_s4",
            (64, 128, 3) => "cutlass_gemm_bias_64x128_s3",
            (64, 128, 4) => "cutlass_gemm_bias_64x128_s4",
            (128, 64, 3) => "cutlass_gemm_bias_128x64_s3",
            (128, 64, 4) => "cutlass_gemm_bias_128x64_s4",
            (128, 128, 3) => "cutlass_gemm_bias_128x128_s3",
            (128, 128, 4) => "cutlass_gemm_bias_128x128_s4",
            (128, 256, 3) => "cutlass_gemm_bias_128x256_s3",
            (256, 64, 3) => "cutlass_gemm_bias_256x64_s3",
            (256, 64, 4) => "cutlass_gemm_bias_256x64_s4",
            _ => "cutlass_gemm_bias_unknown",
        }
    }
}

impl Implementation for CutlassFusedGemmBiasImpl {
    fn name(&self) -> &'static str {
        self.static_name()
    }

    fn target_compatible(&self, profile: &TargetProfile) -> bool {
        profile.cost_table.has_kernel(self.csv_name())
    }

    fn matches(&self, fuf: &Fuf, seed: TileId, profile: &TargetProfile) -> Option<MatchInfo> {
        // Structural pattern identical to FusedGemmBiasImpl — DP
        // picks whichever has the lower calibrated cost per bucket.
        FusedGemmBiasImpl.matches(fuf, seed, profile)
    }

    fn cost_us(&self, m: &MatchInfo, ctx: &CostCtx) -> f64 {
        let gemm_tile = m
            .claimed_tiles
            .iter()
            .copied()
            .find(|t| ctx.fuf.get(*t).op == OpKind::Gemm)
            .expect("claim contains Gemm");
        let Some((mm, nn, kk)) = gemm_mnk(ctx, ctx.fuf.get(gemm_tile)) else {
            return f64::INFINITY;
        };
        ctx.profile
            .cost_us_for(self.csv_name(), mm, nn, kk)
            .unwrap_or(UNCALIBRATED_COST_US)
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
        // Share the LinearLayer accessor with FusedGemmBiasImpl so
        // codegen dedups both Impls' accessor declarations.
        FusedGemmBiasImpl.required_weights(claimed_tiles, fuf, program)
    }

    // ── Host-interpreter codegen ────────────────────────────────
    //
    // Variant `CutlassFusedGemmBias { in_slot, out_slot, layer,
    // weight_fn, tile_m, tile_n, stages, n, k }`. All zoo entries
    // share this variant — tile config rides as runtime fields. The
    // arm body calls `cutlass_gemm_bias(..., CutlassTile::new(...))`.

    fn opcode_shape(&self) -> OpcodeShape {
        OpcodeShape::new(
            "CutlassFusedGemmBias",
            vec![
                ("in_slot", syn::parse_quote!(u32)),
                ("out_slot", syn::parse_quote!(u32)),
                ("layer", syn::parse_quote!(u32)),
                (
                    "weight_fn",
                    syn::parse_quote!(
                        for<'a> fn(&'a Weights, u32) -> &'a ::ferrite_kernels::layers::LinearLayer
                    ),
                ),
                ("tile_m", syn::parse_quote!(u32)),
                ("tile_n", syn::parse_quote!(u32)),
                ("stages", syn::parse_quote!(u32)),
                ("n", syn::parse_quote!(u32)),
                ("k", syn::parse_quote!(u32)),
            ],
        )
    }

    fn fan_out(
        &self,
        m: &MatchInfo,
        fuf: &Fuf,
        program: &Program,
        bounds: &BTreeMap<String, u64>,
        slots: &SlotMap,
    ) -> Option<Vec<OpInstance>> {
        let gemm_id = *m
            .claimed_tiles
            .iter()
            .find(|t| fuf.get(**t).op == OpKind::Gemm)
            .expect("CutlassFusedGemmBias: claim contains Gemm");
        let bias_id = *m
            .claimed_tiles
            .iter()
            .find(|t| fuf.get(**t).op == OpKind::BiasAdd)
            .expect("CutlassFusedGemmBias: claim contains BiasAdd");
        let gemm_node = fuf.get(gemm_id);
        let (in_id, in_slot) = match gemm_node.inputs.first() {
            Some(FufInput::Tile { id, slot }) => (*id, *slot),
            other => {
                panic!("CutlassFusedGemmBias: gemm's first input must be a Tile (got {other:?})")
            }
        };
        let in_slot_idx = slots.of(in_id, in_slot);
        let out_slot_idx = slots.of(bias_id, 0);
        let accessors = self.required_weights(&m.claimed_tiles, fuf, program);
        let acc = accessors
            .first()
            .expect("CutlassFusedGemmBias: required_weights returned empty");
        let (base, layer) = split_base_layer(&acc.name.to_string());
        let layer = layer.unwrap_or(0) as u32;
        let base_ident = syn::Ident::new(&base, proc_macro2::Span::call_site());
        let tile_m = self.tile_m;
        let tile_n = self.tile_n;
        let stages = self.stages;
        let (n, k) = gemm_nk_from_fuf(fuf, gemm_node, bounds)
            .expect("CutlassFusedGemmBias: weight (N, K) must resolve from FUF + bounds");
        Some(vec![OpInstance::new(
            syn::Ident::new("CutlassFusedGemmBias", proc_macro2::Span::call_site()),
            vec![
                quote! { #in_slot_idx },
                quote! { #out_slot_idx },
                quote! { #layer },
                quote! { Weights::#base_ident },
                quote! { #tile_m },
                quote! { #tile_n },
                quote! { #stages },
                quote! { #n },
                quote! { #k },
            ],
        )])
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
///
/// All sources of one fused accessor share the same layer index
/// (or all are unindexed) — this is structural: a layer-N gate_proj
/// can't fuse with a layer-M up_proj. So we strip the per-component
/// `_<layer>` suffix before joining and append it once at the end.
/// Result: `mlp_gate_proj__fused__mlp_up_proj_28` rather than
/// `mlp_gate_proj_28__fused__mlp_up_proj_28`. The accessor-method
/// codegen sees a single base across all layers and collapses 40
/// per-layer methods into one method with 40 match arms.
pub fn fused_accessor_name(program: &Program, sources: &[(WeightId, Option<u64>)]) -> syn::Ident {
    debug_assert!(!sources.is_empty(), "fused_accessor_name: empty sources");
    let layer = sources[0].1;
    debug_assert!(
        sources.iter().all(|(_, idx)| *idx == layer),
        "fused_accessor_name: sources span multiple layer indices ({sources:?})"
    );
    let mut parts: Vec<String> = sources
        .iter()
        .map(|(id, _idx)| weight_field_name(program, *id, None).to_string())
        .collect();
    parts.sort();
    let mut joined = parts.join("__fused__");
    if let Some(l) = layer {
        joined = format!("{joined}_{l}");
    }
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
        // Cost = one cuBLAS GEMM at packed (M, 2I, K) + one elementwise
        // silu_mul pass over [M, 2I] producing [M, I].
        //
        // Read measured cublas cost from the calibration CSV (the
        // predictor linreg-extrapolates when the exact (M, 2I, K) row
        // isn't sampled). Fall back to peak-FLOPS roofline only when
        // the predictor has no signal at all — keeps this Impl on the
        // same measurement scale as `CutlassFusedGateUpSiluMulImpl`,
        // which sums measured cutlass CSV rows. Without this parity
        // the DP saw a wildly optimistic cuBLAS path (8× faster than
        // reality at llama-7b prefill) and never picked the CUTLASS
        // sibling.
        let num_tokens = ctx.num_tokens() as u32;
        let hidden = ctx.bounds.get("hidden_size").copied().unwrap_or(0) as u32;
        let intermediate = ctx.bounds.get("intermediate_size").copied().unwrap_or(0) as u32;
        let packed_n = 2u32.saturating_mul(intermediate);

        let gemm_us = ctx
            .profile
            .cost_us_for("cublas", num_tokens, packed_n, hidden)
            .unwrap_or_else(|| {
                let flops = 2.0 * (num_tokens as f64) * (packed_n as f64) * (hidden as f64);
                let peak = ctx.profile.peak_tflops_fp16 * 1e12;
                if peak > 0.0 && flops > 0.0 {
                    (flops / peak) * 1e6
                } else {
                    0.0
                }
            });

        // Bandwidth-bound silu*mul: read 2*M*I, write M*I, bf16 = 2 B.
        let bw_gb = ctx.profile.memory_bandwidth_gbps;
        let bytes = 3.0 * (num_tokens as f64) * (intermediate as f64) * BYTES_PER_ELEM;
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

    // ── Host-interpreter codegen ────────────────────────────────
    //
    // Variant `FusedGateUpSiluMul { in_slot, out_slot, layer, weight_fn }`.
    // The 4-tile `(Gemm, Gemm, Silu, Mul)` claim collapses to one
    // `LinearLayer::forward` (against the packed `[gate|up]` weight)
    // followed by `silu_and_mul_fused`. `intermediate_size` is baked
    // from `model.bounds` at codegen time — same as today's `emit_call`
    // baking via `ctx.bound`.

    fn opcode_shape(&self) -> OpcodeShape {
        OpcodeShape::new(
            "FusedGateUpSiluMul",
            vec![
                ("in_slot", syn::parse_quote!(u32)),
                ("out_slot", syn::parse_quote!(u32)),
                ("layer", syn::parse_quote!(u32)),
                (
                    "weight_fn",
                    syn::parse_quote!(
                        for<'a> fn(&'a Weights, u32) -> &'a ::ferrite_kernels::layers::LinearLayer
                    ),
                ),
            ],
        )
    }

    fn fan_out(
        &self,
        m: &MatchInfo,
        fuf: &Fuf,
        program: &Program,
        _bounds: &BTreeMap<String, u64>,
        slots: &SlotMap,
    ) -> Option<Vec<OpInstance>> {
        let silu_id = *m
            .claimed_tiles
            .iter()
            .find(|t| fuf.get(**t).op == OpKind::Silu)
            .expect("FusedGateUpSiluMul: claim contains Silu");
        let mul_id = *m
            .claimed_tiles
            .iter()
            .find(|t| fuf.get(**t).op == OpKind::Mul)
            .expect("FusedGateUpSiluMul: claim contains Mul");
        // Gate gemm is the Silu's tile producer; up gemm is the other
        // claimed Gemm. Both Gemms read the same activation tile/slot
        // (matches() already enforced this), so we resolve in_slot off
        // the gate gemm's first tile input.
        let gate_id = match fuf.get(silu_id).inputs.first() {
            Some(FufInput::Tile { id, .. }) => *id,
            other => {
                panic!("FusedGateUpSiluMul: Silu's first input must be a Tile (got {other:?})")
            }
        };
        let gate_node = fuf.get(gate_id);
        let (in_id, in_slot) = match gate_node.inputs.first() {
            Some(FufInput::Tile { id, slot }) => (*id, *slot),
            other => {
                panic!("FusedGateUpSiluMul: gate gemm's first input must be a Tile (got {other:?})")
            }
        };
        let in_slot_idx = slots.of(in_id, in_slot);
        let out_slot_idx = slots.of(mul_id, 0);
        let accessors = self.required_weights(&m.claimed_tiles, fuf, program);
        let acc = accessors
            .first()
            .expect("FusedGateUpSiluMul: required_weights returned empty");
        let (base, layer) = split_base_layer(&acc.name.to_string());
        let layer = layer.unwrap_or(0) as u32;
        let base_ident = syn::Ident::new(&base, proc_macro2::Span::call_site());
        Some(vec![OpInstance::new(
            syn::Ident::new("FusedGateUpSiluMul", proc_macro2::Span::call_site()),
            vec![
                quote! { #in_slot_idx },
                quote! { #out_slot_idx },
                quote! { #layer },
                quote! { Weights::#base_ident },
            ],
        )])
    }
}

/// Whether `node` consumes the output of `producer` via any Tile input.
fn consumes_tile(node: &crate::fuf::FufNode, producer: TileId) -> bool {
    node.inputs
        .iter()
        .any(|i| matches!(i, FufInput::Tile { id, .. } if *id == producer))
}

// ── CutlassFusedGateUpSiluMulImpl ────────────────────────────────
//
// CUTLASS EVT peer to [`FusedGateUpSiluMulImpl`]. Claims the exact
// same `(Gemm, Gemm, Silu, Mul)` tile pattern and declares the same
// packed `[gate|up]` LinearLayer accessor — so accessor emission is
// unchanged regardless of which Impl the DP picks per bucket.
//
// Kernel shape differs: two GEMM launches, the second with a fused
// SiLU+Mul epilogue that aux-loads the up-projection output from
// GMEM. Saves one BW-bound elementwise kernel launch + one full
// `[M, I]` GMEM round-trip of the gate output.
//
// At emit time the packed `[2I, K]` weight is sliced via
// `narrow_dim0` into `gate_w` and `up_w` views; both GEMMs read the
// same contiguous backing buffer, so there is no duplicate weight
// memory relative to the cuBLAS-packed peer.

#[derive(Debug, Default)]
pub struct CutlassFusedGateUpSiluMulImpl;

impl Implementation for CutlassFusedGateUpSiluMulImpl {
    fn name(&self) -> &'static str {
        "cutlass_fused_gate_up_silu_mul"
    }

    fn target_compatible(&self, profile: &TargetProfile) -> bool {
        profile
            .cost_table
            .has_kernel("cutlass_fused_gate_up_silu_mul")
    }

    fn matches(&self, fuf: &Fuf, seed: TileId, profile: &TargetProfile) -> Option<MatchInfo> {
        // Structural pattern identical to FusedGateUpSiluMulImpl — the
        // two impls claim the same 4 tiles; the DP picks whichever has
        // the lower calibrated cost per bucket.
        FusedGateUpSiluMulImpl.matches(fuf, seed, profile)
    }

    fn cost_us(&self, m: &MatchInfo, ctx: &CostCtx) -> f64 {
        // Two launches:
        //   (a) up GEMM — a standalone cutlass 128×128×s3 GEMM of
        //       shape `(M, I, H)`; captured in the `cutlass_128x128_s3`
        //       CSV row (the tile hard-coded in `emit_call`).
        //   (b) gate GEMM + SiLU + Mul EVT — shape `(M, I, H)`;
        //       captured in the `cutlass_fused_gate_up_silu_mul` row,
        //       which includes the epilogue's aux-load of `[M, I]`
        //       up output.
        let (_m_val, nn, kk) = {
            let gemm_tile = m
                .claimed_tiles
                .iter()
                .copied()
                .find(|t| ctx.fuf.get(*t).op == OpKind::Gemm)
                .expect("fused gate/up/silu/mul claim contains a Gemm");
            match gemm_mnk(ctx, ctx.fuf.get(gemm_tile)) {
                Some(v) => v,
                None => return f64::INFINITY,
            }
        };
        let mm = ctx.num_tokens() as u32;
        let up_us = ctx
            .profile
            .cost_us_for("cutlass_128x128_s3", mm, nn, kk)
            .unwrap_or(UNCALIBRATED_COST_US);
        let fused_us = ctx
            .profile
            .cost_us_for("cutlass_fused_gate_up_silu_mul", mm, nn, kk)
            .unwrap_or(UNCALIBRATED_COST_US);
        up_us + fused_us
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
        // Share the packed `[gate|up]` accessor with the cuBLAS peer
        // so accessor emission is deduplicated by `collect_accessors`.
        FusedGateUpSiluMulImpl.required_weights(claimed_tiles, fuf, program)
    }

    // ── Host-interpreter codegen ────────────────────────────────
    //
    // Variant `CutlassFusedGateUpSiluMul { in_slot, out_slot, layer, weight_fn }`.
    // Same shape as `FusedGateUpSiluMul` but its own variant ident —
    // the body slices the packed weight into gate/up halves and runs
    // a CUTLASS GEMM + EVT SiLU/Mul pair instead of cuBLAS+silu_and_mul.

    fn opcode_shape(&self) -> OpcodeShape {
        OpcodeShape::new(
            "CutlassFusedGateUpSiluMul",
            vec![
                ("in_slot", syn::parse_quote!(u32)),
                ("out_slot", syn::parse_quote!(u32)),
                ("layer", syn::parse_quote!(u32)),
                (
                    "weight_fn",
                    syn::parse_quote!(
                        for<'a> fn(&'a Weights, u32) -> &'a ::ferrite_kernels::layers::LinearLayer
                    ),
                ),
            ],
        )
    }

    fn fan_out(
        &self,
        m: &MatchInfo,
        fuf: &Fuf,
        program: &Program,
        _bounds: &BTreeMap<String, u64>,
        slots: &SlotMap,
    ) -> Option<Vec<OpInstance>> {
        let silu_id = *m
            .claimed_tiles
            .iter()
            .find(|t| fuf.get(**t).op == OpKind::Silu)
            .expect("CutlassFusedGateUpSiluMul: claim contains Silu");
        let mul_id = *m
            .claimed_tiles
            .iter()
            .find(|t| fuf.get(**t).op == OpKind::Mul)
            .expect("CutlassFusedGateUpSiluMul: claim contains Mul");
        let gate_id = match fuf.get(silu_id).inputs.first() {
            Some(FufInput::Tile { id, .. }) => *id,
            other => panic!(
                "CutlassFusedGateUpSiluMul: Silu's first input must be a Tile (got {other:?})"
            ),
        };
        let gate_node = fuf.get(gate_id);
        let (in_id, in_slot) = match gate_node.inputs.first() {
            Some(FufInput::Tile { id, slot }) => (*id, *slot),
            other => panic!(
                "CutlassFusedGateUpSiluMul: gate gemm's first input must be a Tile (got {other:?})"
            ),
        };
        let in_slot_idx = slots.of(in_id, in_slot);
        let out_slot_idx = slots.of(mul_id, 0);
        let accessors = self.required_weights(&m.claimed_tiles, fuf, program);
        let acc = accessors
            .first()
            .expect("CutlassFusedGateUpSiluMul: required_weights returned empty");
        let (base, layer) = split_base_layer(&acc.name.to_string());
        let layer = layer.unwrap_or(0) as u32;
        let base_ident = syn::Ident::new(&base, proc_macro2::Span::call_site());
        Some(vec![OpInstance::new(
            syn::Ident::new("CutlassFusedGateUpSiluMul", proc_macro2::Span::call_site()),
            vec![
                quote! { #in_slot_idx },
                quote! { #out_slot_idx },
                quote! { #layer },
                quote! { Weights::#base_ident },
            ],
        )])
    }
}

// ── CutlassFusedGateUpGeluMulImpl ────────────────────────────────
//
// CUTLASS peer to [`FusedGateUpGeluMulImpl`]. Mirrors the cuBLAS
// peer's structure exactly: ONE packed CUTLASS GEMM at (M, 2I, K)
// producing a `[M, 2I]` intermediate buffer, then a BW-bound
// `gelu_and_mul_fused` elementwise pass producing `[M, I]`. Tile is
// parameterised so the DP picks the best CUTLASS tile per (M, 2I, K)
// — one Impl per CUTLASS_TILE_ZOO entry.
//
// This is structurally NOT the EVT 2-GEMM design used by
// `CutlassFusedGateUpSiluMulImpl`; that design wins for BW-bound
// shapes (small M, modest I/H — granite, llama-1b) but loses for
// compute-bound shapes (gemma2/gemma3 MLPs). The packed approach is
// faithful to the cuBLAS peer and wins iff
// `cutlass_<tile>(M, 2I, K) < cublas(M, 2I, K)`.

#[derive(Debug, Clone)]
pub struct CutlassFusedGateUpGeluMulImpl {
    pub tile_m: u32,
    pub tile_n: u32,
    pub stages: u32,
}

impl CutlassFusedGateUpGeluMulImpl {
    fn csv_name(&self) -> &'static str {
        // Reuses the standalone CutlassGemm tile names — the GEMM
        // step is an ordinary cutlass_<TM>x<TN>_s<S> at packed N=2I.
        match (self.tile_m, self.tile_n, self.stages) {
            (16, 64, 3) => "cutlass_16x64_s3",
            (16, 64, 4) => "cutlass_16x64_s4",
            (16, 128, 3) => "cutlass_16x128_s3",
            (16, 128, 4) => "cutlass_16x128_s4",
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

impl Implementation for CutlassFusedGateUpGeluMulImpl {
    fn name(&self) -> &'static str {
        self.csv_name()
    }

    fn target_compatible(&self, profile: &TargetProfile) -> bool {
        profile.cost_table.has_kernel(self.csv_name())
    }

    fn matches(&self, fuf: &Fuf, seed: TileId, profile: &TargetProfile) -> Option<MatchInfo> {
        // Same 4-tile (Gemm, Gemm, Gelu, Mul) claim as cuBLAS peer.
        FusedGateUpGeluMulImpl.matches(fuf, seed, profile)
    }

    fn cost_us(&self, m: &MatchInfo, ctx: &CostCtx) -> f64 {
        // Same cost shape as FusedGateUpGeluMulImpl (cuBLAS peer):
        //   - one GEMM at packed (M, 2I, K)
        //   - one BW-bound gelu_and_mul_fused over [M, 2I] → [M, I]
        // Difference: the GEMM is a calibrated cutlass_<tile> row,
        // not cublas. The BW-bound piece is identical → comparison
        // reduces to cutlass_<tile>(M, 2I, K) vs cublas(M, 2I, K).
        let num_tokens = ctx.num_tokens() as u32;
        let hidden = ctx.bounds.get("hidden_size").copied().unwrap_or(0) as u32;
        let intermediate = ctx.bounds.get("intermediate_size").copied().unwrap_or(0) as u32;
        let packed_n = 2u32.saturating_mul(intermediate);

        // Reject shapes that this tile can't profitably handle:
        // packed_n must be a multiple of tile_n (alignment), and the
        // CSV must have a calibrated row at (M, 2I, K).
        let gemm_us = ctx
            .profile
            .cost_us_for(self.csv_name(), num_tokens, packed_n, hidden)
            .unwrap_or(UNCALIBRATED_COST_US);

        let bw_gb = ctx.profile.memory_bandwidth_gbps;
        let bytes = 3.0 * (num_tokens as f64) * (intermediate as f64) * BYTES_PER_ELEM;
        let elem_us = if bw_gb > 0.0 {
            (bytes / (bw_gb * 1e9)) * 1e6
        } else {
            0.0
        };

        let _ = m; // claim shape is constant; cost is workload-driven
        gemm_us + elem_us
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
        FusedGateUpGeluMulImpl.required_weights(claimed_tiles, fuf, program)
    }

    fn opcode_shape(&self) -> OpcodeShape {
        OpcodeShape::new(
            "CutlassFusedGateUpGeluMul",
            vec![
                ("in_slot", syn::parse_quote!(u32)),
                ("out_slot", syn::parse_quote!(u32)),
                ("layer", syn::parse_quote!(u32)),
                (
                    "weight_fn",
                    syn::parse_quote!(
                        for<'a> fn(&'a Weights, u32) -> &'a ::ferrite_kernels::layers::LinearLayer
                    ),
                ),
                ("tile_m", syn::parse_quote!(u32)),
                ("tile_n", syn::parse_quote!(u32)),
                ("stages", syn::parse_quote!(u32)),
                ("packed_n", syn::parse_quote!(u32)),
                ("k", syn::parse_quote!(u32)),
            ],
        )
    }

    fn fan_out(
        &self,
        m: &MatchInfo,
        fuf: &Fuf,
        program: &Program,
        bounds: &BTreeMap<String, u64>,
        slots: &SlotMap,
    ) -> Option<Vec<OpInstance>> {
        let gelu_id = *m
            .claimed_tiles
            .iter()
            .find(|t| fuf.get(**t).op == OpKind::Gelu)
            .expect("CutlassFusedGateUpGeluMul: claim contains Gelu");
        let mul_id = *m
            .claimed_tiles
            .iter()
            .find(|t| fuf.get(**t).op == OpKind::Mul)
            .expect("CutlassFusedGateUpGeluMul: claim contains Mul");
        let gate_id = match fuf.get(gelu_id).inputs.first() {
            Some(FufInput::Tile { id, .. }) => *id,
            other => panic!(
                "CutlassFusedGateUpGeluMul: Gelu's first input must be a Tile (got {other:?})"
            ),
        };
        let gate_node = fuf.get(gate_id);
        let (in_id, in_slot) = match gate_node.inputs.first() {
            Some(FufInput::Tile { id, slot }) => (*id, *slot),
            other => panic!(
                "CutlassFusedGateUpGeluMul: gate gemm's first input must be a Tile (got {other:?})"
            ),
        };
        let in_slot_idx = slots.of(in_id, in_slot);
        let out_slot_idx = slots.of(mul_id, 0);
        let accessors = self.required_weights(&m.claimed_tiles, fuf, program);
        let acc = accessors
            .first()
            .expect("CutlassFusedGateUpGeluMul: required_weights returned empty");
        let (base, layer) = split_base_layer(&acc.name.to_string());
        let layer = layer.unwrap_or(0) as u32;
        let base_ident = syn::Ident::new(&base, proc_macro2::Span::call_site());

        // Bake (packed_n=2I, k=H) into the Instruction for the runtime
        // assert_weight_shape check. Use the gate Gemm's K and 2× the
        // gate's N; both are static per FUF + bounds.
        let (gate_n, k) = gemm_nk_from_fuf(fuf, gate_node, bounds)
            .expect("CutlassFusedGateUpGeluMul: gate (N, K) must resolve from FUF + bounds");
        let packed_n = gate_n.saturating_mul(2);
        let tile_m = self.tile_m;
        let tile_n = self.tile_n;
        let stages = self.stages;
        Some(vec![OpInstance::new(
            syn::Ident::new("CutlassFusedGateUpGeluMul", proc_macro2::Span::call_site()),
            vec![
                quote! { #in_slot_idx },
                quote! { #out_slot_idx },
                quote! { #layer },
                quote! { Weights::#base_ident },
                quote! { #tile_m },
                quote! { #tile_n },
                quote! { #stages },
                quote! { #packed_n },
                quote! { #k },
            ],
        )])
    }
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
        // Cost = one cuBLAS GEMM at packed (M, 2I, K) + one elementwise
        // gelu_and_mul pass over [M, 2I] → [M, I].
        //
        // Read measured cublas cost from the calibration CSV (predictor
        // linreg-extrapolates when the exact (M, 2I, K) row isn't
        // sampled). Fall back to peak-FLOPS roofline only when the
        // predictor has no signal at all — keeps this Impl on the same
        // measurement scale as `CutlassFusedGateUpGeluMulImpl`, which
        // sums measured cutlass CSV rows. This is the GELU twin of the
        // silu fix in commit 4a67304ec; without it the DP saw a wildly
        // optimistic roofline cuBLAS path (~8× faster than reality)
        // and never picked the CUTLASS sibling.
        let num_tokens = ctx.num_tokens() as u32;
        let hidden = ctx.bounds.get("hidden_size").copied().unwrap_or(0) as u32;
        let intermediate = ctx.bounds.get("intermediate_size").copied().unwrap_or(0) as u32;
        let packed_n = 2u32.saturating_mul(intermediate);

        let gemm_us = ctx
            .profile
            .cost_us_for("cublas", num_tokens, packed_n, hidden)
            .unwrap_or_else(|| {
                let flops = 2.0 * (num_tokens as f64) * (packed_n as f64) * (hidden as f64);
                let peak = ctx.profile.peak_tflops_fp16 * 1e12;
                if peak > 0.0 && flops > 0.0 {
                    (flops / peak) * 1e6
                } else {
                    0.0
                }
            });

        // Bandwidth-bound gelu*mul: read 2*M*I, write M*I, bf16 = 2 B.
        let bw_gb = ctx.profile.memory_bandwidth_gbps;
        let bytes = 3.0 * (num_tokens as f64) * (intermediate as f64) * BYTES_PER_ELEM;
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

    // ── Host-interpreter codegen ────────────────────────────────
    //
    // Variant `FusedGateUpGeluMul { in_slot, out_slot, layer, weight_fn }`.
    // GELU twin of `FusedGateUpSiluMul`: same 4-tile claim shape, same
    // packed `[gate|up]` LinearLayer, only the elementwise kernel
    // differs (`gelu_and_mul_fused` vs `silu_and_mul_fused`).

    fn opcode_shape(&self) -> OpcodeShape {
        OpcodeShape::new(
            "FusedGateUpGeluMul",
            vec![
                ("in_slot", syn::parse_quote!(u32)),
                ("out_slot", syn::parse_quote!(u32)),
                ("layer", syn::parse_quote!(u32)),
                (
                    "weight_fn",
                    syn::parse_quote!(
                        for<'a> fn(&'a Weights, u32) -> &'a ::ferrite_kernels::layers::LinearLayer
                    ),
                ),
            ],
        )
    }

    fn fan_out(
        &self,
        m: &MatchInfo,
        fuf: &Fuf,
        program: &Program,
        _bounds: &BTreeMap<String, u64>,
        slots: &SlotMap,
    ) -> Option<Vec<OpInstance>> {
        let gelu_id = *m
            .claimed_tiles
            .iter()
            .find(|t| fuf.get(**t).op == OpKind::Gelu)
            .expect("FusedGateUpGeluMul: claim contains Gelu");
        let mul_id = *m
            .claimed_tiles
            .iter()
            .find(|t| fuf.get(**t).op == OpKind::Mul)
            .expect("FusedGateUpGeluMul: claim contains Mul");
        let gate_id = match fuf.get(gelu_id).inputs.first() {
            Some(FufInput::Tile { id, .. }) => *id,
            other => {
                panic!("FusedGateUpGeluMul: Gelu's first input must be a Tile (got {other:?})")
            }
        };
        let gate_node = fuf.get(gate_id);
        let (in_id, in_slot) = match gate_node.inputs.first() {
            Some(FufInput::Tile { id, slot }) => (*id, *slot),
            other => {
                panic!("FusedGateUpGeluMul: gate gemm's first input must be a Tile (got {other:?})")
            }
        };
        let in_slot_idx = slots.of(in_id, in_slot);
        let out_slot_idx = slots.of(mul_id, 0);
        let accessors = self.required_weights(&m.claimed_tiles, fuf, program);
        let acc = accessors
            .first()
            .expect("FusedGateUpGeluMul: required_weights returned empty");
        let (base, layer) = split_base_layer(&acc.name.to_string());
        let layer = layer.unwrap_or(0) as u32;
        let base_ident = syn::Ident::new(&base, proc_macro2::Span::call_site());
        Some(vec![OpInstance::new(
            syn::Ident::new("FusedGateUpGeluMul", proc_macro2::Span::call_site()),
            vec![
                quote! { #in_slot_idx },
                quote! { #out_slot_idx },
                quote! { #layer },
                quote! { Weights::#base_ident },
            ],
        )])
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

    // ── Host-interpreter codegen ────────────────────────────────
    //
    // Variant `ScalarMul { in_slot, out_slot, scale }`. In-place
    // mutator: `take_owned` lifts the upstream OwnedTensor out of
    // `in_slot`, the kernel mutates it, then we reinsert at
    // `out_slot`. The codegen drop-pass already excludes the
    // upstream from `Free` (see `consumes_input_tiles`), so the
    // take→reinsert pattern is balanced.

    fn opcode_shape(&self) -> OpcodeShape {
        OpcodeShape::new(
            "ScalarMul",
            vec![
                ("in_slot", syn::parse_quote!(u32)),
                ("out_slot", syn::parse_quote!(u32)),
                ("scale", syn::parse_quote!(f32)),
            ],
        )
    }

    fn fan_out(
        &self,
        m: &MatchInfo,
        fuf: &Fuf,
        _program: &Program,
        _bounds: &BTreeMap<String, u64>,
        slots: &SlotMap,
    ) -> Option<Vec<OpInstance>> {
        let tile = m.claimed_tiles[0];
        let node = fuf.get(tile);
        let (in_id, in_slot) = node
            .inputs
            .iter()
            .find_map(|i| match i {
                FufInput::Tile { id, slot } => Some((*id, *slot)),
                _ => None,
            })
            .expect("ScalarMul: claim has a Tile input");
        let scale: f32 = node
            .inputs
            .iter()
            .find_map(|i| match i {
                FufInput::Scalar(v) => Some(*v as f32),
                _ => None,
            })
            .expect("ScalarMul: claim has a Scalar input");
        let in_slot_idx = slots.of(in_id, in_slot);
        let out_slot_idx = slots.of(tile, 0);
        Some(vec![OpInstance::new(
            syn::Ident::new("ScalarMul", proc_macro2::Span::call_site()),
            vec![
                quote! { #in_slot_idx },
                quote! { #out_slot_idx },
                quote! { #scale },
            ],
        )])
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

    // ── Host-interpreter codegen ────────────────────────────────
    //
    // Variant `TanhSoftCap { in_slot, out_slot }`. The `cap` scalar
    // is arch-wide (`final_logit_softcapping` from config.json), so
    // it's baked into the arm body at codegen time — not a payload
    // field. In-place consume: take_owned → mutate → reinsert.

    fn opcode_shape(&self) -> OpcodeShape {
        OpcodeShape::new(
            "TanhSoftCap",
            vec![
                ("in_slot", syn::parse_quote!(u32)),
                ("out_slot", syn::parse_quote!(u32)),
            ],
        )
    }

    fn fan_out(
        &self,
        m: &MatchInfo,
        fuf: &Fuf,
        _program: &Program,
        _bounds: &BTreeMap<String, u64>,
        slots: &SlotMap,
    ) -> Option<Vec<OpInstance>> {
        let tile = m.claimed_tiles[0];
        let node = fuf.get(tile);
        let (in_id, in_slot) = match node.inputs.first() {
            Some(FufInput::Tile { id, slot }) => (*id, *slot),
            other => panic!("TanhSoftCap: first input must be a Tile (got {other:?})"),
        };
        let in_slot_idx = slots.of(in_id, in_slot);
        let out_slot_idx = slots.of(tile, 0);
        Some(vec![OpInstance::new(
            syn::Ident::new("TanhSoftCap", proc_macro2::Span::call_site()),
            vec![quote! { #in_slot_idx }, quote! { #out_slot_idx }],
        )])
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

    // ── Host-interpreter codegen ────────────────────────────────
    //
    // Variant `Add { delta_slot, residual_slot }`. No `out_slot`
    // payload — `output_alias` declares the Add's output as a View
    // alias of the residual input, and the codegen-emitted prelude
    // populates the alias slot before the interpreter loop runs.
    // The arm only performs the in-place mutation.

    fn opcode_shape(&self) -> OpcodeShape {
        OpcodeShape::new(
            "Add",
            vec![
                ("delta_slot", syn::parse_quote!(u32)),
                ("residual_slot", syn::parse_quote!(u32)),
            ],
        )
    }

    fn fan_out(
        &self,
        m: &MatchInfo,
        fuf: &Fuf,
        _program: &Program,
        _bounds: &BTreeMap<String, u64>,
        slots: &SlotMap,
    ) -> Option<Vec<OpInstance>> {
        let add_id = m.claimed_tiles[0];
        let node = fuf.get(add_id);
        let (delta_id, delta_in_slot) = match node.inputs.first() {
            Some(FufInput::Tile { id, slot }) => (*id, *slot),
            other => panic!("Add: input 0 (delta) must be a Tile (got {other:?})"),
        };
        let (residual_id, residual_in_slot) = match node.inputs.get(1) {
            Some(FufInput::Tile { id, slot }) => (*id, *slot),
            other => panic!("Add: input 1 (residual) must be a Tile (got {other:?})"),
        };
        let delta_idx = slots.of(delta_id, delta_in_slot);
        let residual_idx = slots.of(residual_id, residual_in_slot);
        Some(vec![OpInstance::new(
            syn::Ident::new("Add", proc_macro2::Span::call_site()),
            vec![quote! { #delta_idx }, quote! { #residual_idx }],
        )])
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

    // ── Host-interpreter codegen ────────────────────────────────
    //
    // Variant `FusedAddRmsNorm { delta_slot, residual_slot, layer, weight_fn }`.
    // No `out_slot` payload — `output_alias` declares both outputs
    // (rmsnorm_id slot 0 → delta upstream, add_id slot 0 → residual
    // upstream) and the alias prelude populates them as `View`
    // entries before the loop. The arm just runs the in-place
    // mutation kernel.

    fn opcode_shape(&self) -> OpcodeShape {
        OpcodeShape::new(
            "FusedAddRmsNorm",
            vec![
                ("delta_slot", syn::parse_quote!(u32)),
                ("residual_slot", syn::parse_quote!(u32)),
                ("layer", syn::parse_quote!(u32)),
                (
                    "weight_fn",
                    syn::parse_quote!(
                        for<'a> fn(&'a Weights, u32) -> &'a ::ferrite_kernels::layers::RmsNorm
                    ),
                ),
            ],
        )
    }

    fn fan_out(
        &self,
        m: &MatchInfo,
        fuf: &Fuf,
        program: &Program,
        _bounds: &BTreeMap<String, u64>,
        slots: &SlotMap,
    ) -> Option<Vec<OpInstance>> {
        let add_id = *m
            .claimed_tiles
            .iter()
            .find(|t| fuf.get(**t).op == OpKind::Add)
            .expect("FusedAddRmsNorm: claim contains Add");
        let add_node = fuf.get(add_id);
        let (delta_id, delta_in_slot) = match add_node.inputs.first() {
            Some(FufInput::Tile { id, slot }) => (*id, *slot),
            other => panic!("FusedAddRmsNorm: Add input 0 (delta) must be a Tile (got {other:?})"),
        };
        let (residual_id, residual_in_slot) = match add_node.inputs.get(1) {
            Some(FufInput::Tile { id, slot }) => (*id, *slot),
            other => {
                panic!("FusedAddRmsNorm: Add input 1 (residual) must be a Tile (got {other:?})")
            }
        };
        let delta_idx = slots.of(delta_id, delta_in_slot);
        let residual_idx = slots.of(residual_id, residual_in_slot);
        let accessors = self.required_weights(&m.claimed_tiles, fuf, program);
        let acc = accessors
            .first()
            .expect("FusedAddRmsNorm: required_weights returned empty");
        let (base, layer) = split_base_layer(&acc.name.to_string());
        let layer = layer.unwrap_or(0) as u32;
        let base_ident = syn::Ident::new(&base, proc_macro2::Span::call_site());
        Some(vec![OpInstance::new(
            syn::Ident::new("FusedAddRmsNorm", proc_macro2::Span::call_site()),
            vec![
                quote! { #delta_idx },
                quote! { #residual_idx },
                quote! { #layer },
                quote! { Weights::#base_ident },
            ],
        )])
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

    // ── Host-interpreter codegen ────────────────────────────────
    //
    // Variant `FusedAddRmsNormWithOffset { delta_slot, residual_slot,
    // layer, offset, weight_fn }`. Same alias-prelude shape as
    // `FusedAddRmsNorm`; the Gemma2 `weight + 1.0` lowering rides on
    // the per-instance `offset` field.

    fn opcode_shape(&self) -> OpcodeShape {
        OpcodeShape::new(
            "FusedAddRmsNormWithOffset",
            vec![
                ("delta_slot", syn::parse_quote!(u32)),
                ("residual_slot", syn::parse_quote!(u32)),
                ("layer", syn::parse_quote!(u32)),
                ("offset", syn::parse_quote!(f32)),
                (
                    "weight_fn",
                    syn::parse_quote!(
                        for<'a> fn(&'a Weights, u32) -> &'a ::ferrite_kernels::layers::RmsNorm
                    ),
                ),
            ],
        )
    }

    fn fan_out(
        &self,
        m: &MatchInfo,
        fuf: &Fuf,
        program: &Program,
        _bounds: &BTreeMap<String, u64>,
        slots: &SlotMap,
    ) -> Option<Vec<OpInstance>> {
        // Identify the residual-stream Add (two Tile inputs) — the
        // scalar-offset Add has a Weight + Scalar.
        let residual_add_id = *m
            .claimed_tiles
            .iter()
            .find(|t| {
                let n = fuf.get(**t);
                n.op == OpKind::Add && n.inputs.iter().all(|i| matches!(i, FufInput::Tile { .. }))
            })
            .expect("FusedAddRmsNormWithOffset: claim contains a residual-stream Add");
        let scalar_add_id = *m
            .claimed_tiles
            .iter()
            .find(|t| {
                let n = fuf.get(**t);
                n.op == OpKind::Add && n.inputs.iter().any(|i| matches!(i, FufInput::Scalar(_)))
            })
            .expect("FusedAddRmsNormWithOffset: claim contains a scalar-offset Add");

        let add_node = fuf.get(residual_add_id);
        let (delta_id, delta_in_slot) = match add_node.inputs.first() {
            Some(FufInput::Tile { id, slot }) => (*id, *slot),
            other => panic!(
                "FusedAddRmsNormWithOffset: residual Add input 0 must be a Tile (got {other:?})"
            ),
        };
        let (residual_id, residual_in_slot) = match add_node.inputs.get(1) {
            Some(FufInput::Tile { id, slot }) => (*id, *slot),
            other => panic!(
                "FusedAddRmsNormWithOffset: residual Add input 1 must be a Tile (got {other:?})"
            ),
        };
        let delta_idx = slots.of(delta_id, delta_in_slot);
        let residual_idx = slots.of(residual_id, residual_in_slot);

        let scalar_add = fuf.get(scalar_add_id);
        let offset: f32 = scalar_add
            .inputs
            .iter()
            .find_map(|i| match i {
                FufInput::Scalar(v) => Some(*v as f32),
                _ => None,
            })
            .expect("FusedAddRmsNormWithOffset: scalar-offset Add has a Scalar input");

        let accessors = self.required_weights(&m.claimed_tiles, fuf, program);
        let acc = accessors
            .first()
            .expect("FusedAddRmsNormWithOffset: required_weights returned empty");
        let (base, layer) = split_base_layer(&acc.name.to_string());
        let layer = layer.unwrap_or(0) as u32;
        let base_ident = syn::Ident::new(&base, proc_macro2::Span::call_site());
        Some(vec![OpInstance::new(
            syn::Ident::new("FusedAddRmsNormWithOffset", proc_macro2::Span::call_site()),
            vec![
                quote! { #delta_idx },
                quote! { #residual_idx },
                quote! { #layer },
                quote! { #offset },
                quote! { Weights::#base_ident },
            ],
        )])
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

    // ── Host-interpreter codegen ────────────────────────────────
    //
    // Variant `ScalarOffsetRmsNorm { in_slot, out_slot, layer,
    // offset, weight_fn }`. Singleton-output rmsnorm with a per-
    // instance `+offset` baked from the claim's Add scalar.

    fn opcode_shape(&self) -> OpcodeShape {
        OpcodeShape::new(
            "ScalarOffsetRmsNorm",
            vec![
                ("in_slot", syn::parse_quote!(u32)),
                ("out_slot", syn::parse_quote!(u32)),
                ("layer", syn::parse_quote!(u32)),
                ("offset", syn::parse_quote!(f32)),
                (
                    "weight_fn",
                    syn::parse_quote!(
                        for<'a> fn(&'a Weights, u32) -> &'a ::ferrite_kernels::layers::RmsNorm
                    ),
                ),
            ],
        )
    }

    fn fan_out(
        &self,
        m: &MatchInfo,
        fuf: &Fuf,
        program: &Program,
        _bounds: &BTreeMap<String, u64>,
        slots: &SlotMap,
    ) -> Option<Vec<OpInstance>> {
        let add_id = *m
            .claimed_tiles
            .iter()
            .find(|t| fuf.get(**t).op == OpKind::Add)
            .expect("ScalarOffsetRmsNorm: claim contains Add");
        let rmsnorm_id = *m
            .claimed_tiles
            .iter()
            .find(|t| fuf.get(**t).op == OpKind::RmsNorm)
            .expect("ScalarOffsetRmsNorm: claim contains RmsNorm");
        let rmsnorm_node = fuf.get(rmsnorm_id);
        let (in_id, in_slot) = match rmsnorm_node.inputs.first() {
            Some(FufInput::Tile { id, slot }) => (*id, *slot),
            other => panic!("ScalarOffsetRmsNorm: RmsNorm input 0 must be a Tile (got {other:?})"),
        };
        let in_slot_idx = slots.of(in_id, in_slot);
        let out_slot_idx = slots.of(rmsnorm_id, 0);

        let add_node = fuf.get(add_id);
        let offset: f32 = add_node
            .inputs
            .iter()
            .find_map(|i| match i {
                FufInput::Scalar(v) => Some(*v as f32),
                _ => None,
            })
            .expect("ScalarOffsetRmsNorm: Add has a Scalar input");

        let accessors = self.required_weights(&m.claimed_tiles, fuf, program);
        let acc = accessors
            .first()
            .expect("ScalarOffsetRmsNorm: required_weights returned empty");
        let (base, layer) = split_base_layer(&acc.name.to_string());
        let layer = layer.unwrap_or(0) as u32;
        let base_ident = syn::Ident::new(&base, proc_macro2::Span::call_site());
        Some(vec![OpInstance::new(
            syn::Ident::new("ScalarOffsetRmsNorm", proc_macro2::Span::call_site()),
            vec![
                quote! { #in_slot_idx },
                quote! { #out_slot_idx },
                quote! { #layer },
                quote! { #offset },
                quote! { Weights::#base_ident },
            ],
        )])
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
        // Cost = one cuBLAS GEMM at packed (M, q_size+2*kv_size, hidden)
        // + one bandwidth-bound rope+cache pass.
        //
        // Read measured cublas cost from the calibration CSV (the
        // predictor linreg-extrapolates when the exact (M, packed_n, K)
        // row isn't sampled). Fall back to peak-FLOPS roofline only
        // when the predictor has no signal — same shape as the
        // GateUp{Silu,Gelu}Mul peers. The roofline is ~8× optimistic
        // vs measured cuBLAS, so without this parity any future
        // `CutlassFusedQkvRope*` peer (whose GEMM cost reads measured
        // CUTLASS rows) would lose every DP race despite being faster.
        let num_tokens = ctx.num_tokens() as u32;
        let hidden = ctx.bounds.get("hidden_size").copied().unwrap_or(0) as u32;
        let num_q_heads = ctx.bounds.get("num_attention_heads").copied().unwrap_or(0) as u32;
        let num_kv_heads = ctx.bounds.get("num_key_value_heads").copied().unwrap_or(0) as u32;
        let head_dim = ctx.bounds.get("head_dim").copied().unwrap_or(0) as u32;
        // packed_n = (q_size + 2*kv_size) = (num_q + 2*num_kv) * head_dim.
        let packed_n = num_q_heads
            .saturating_add(2u32.saturating_mul(num_kv_heads))
            .saturating_mul(head_dim);

        let gemm_us = ctx
            .profile
            .cost_us_for("cublas", num_tokens, packed_n, hidden)
            .unwrap_or_else(|| {
                let flops = 2.0 * (num_tokens as f64) * (packed_n as f64) * (hidden as f64);
                let peak = ctx.profile.peak_tflops_fp16 * 1e12;
                if peak > 0.0 && flops > 0.0 {
                    (flops / peak) * 1e6
                } else {
                    0.0
                }
            });

        // Rope + cache write: bandwidth-bound, read packed QKV +
        // write rotated Q + K/V to cache.
        let bw_gb = ctx.profile.memory_bandwidth_gbps;
        let bytes = 3.0 * (num_tokens as f64) * (packed_n as f64) * BYTES_PER_ELEM;
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

    // ── Host-interpreter codegen ────────────────────────────────
    //
    // Variant `FusedQkvRopeCache { in_slot, out_slot, layer,
    // weight_fn, cos_sin_fn, biased, interleaved }`. Flags ride on
    // `OpInstance` rather than baking into the body so one arm
    // body works for every claim of this Impl in the arch — Llama,
    // Qwen2 (biased), Cohere (interleaved), and any future combo.
    //
    // Slots 1 and 2 (K/V) are paged-cache slots, written by the
    // kernel directly into `ctx.kv_cache`; downstream attention
    // reads them from the cache, not from `__tiles`. The tile
    // table entries for those slots stay `None` for this Impl's
    // lifetime — that's why `output_alias` only declares slot 0.

    fn opcode_shape(&self) -> OpcodeShape {
        OpcodeShape::new(
            "FusedQkvRopeCache",
            vec![
                ("in_slot", syn::parse_quote!(u32)),
                ("out_slot", syn::parse_quote!(u32)),
                ("layer", syn::parse_quote!(u32)),
                (
                    "weight_fn",
                    syn::parse_quote!(
                        for<'a> fn(&'a Weights, u32) -> &'a ::ferrite_kernels::layers::LinearLayer
                    ),
                ),
                (
                    "cos_sin_fn",
                    syn::parse_quote!(
                        for<'a> fn(&'a Weights, u32) -> ::ferrite_cuda_core::tensor::GpuTensor
                    ),
                ),
                ("biased", syn::parse_quote!(bool)),
                ("interleaved", syn::parse_quote!(bool)),
            ],
        )
    }

    fn fan_out(
        &self,
        m: &MatchInfo,
        fuf: &Fuf,
        program: &Program,
        _bounds: &BTreeMap<String, u64>,
        slots: &SlotMap,
    ) -> Option<Vec<OpInstance>> {
        let rope_id = *m
            .claimed_tiles
            .iter()
            .find(|t| is_rope_append_op(fuf.get(**t).op))
            .expect("FusedQkvRopeCache: claim contains a RopeAppend");
        let rope_node = fuf.get(rope_id);
        let interleaved = rope_node.op == OpKind::RopeAppendInterleaved;
        let layer = rope_kv_cache_layer(rope_node)
            .expect("FusedQkvRopeCache: RopeAppend has a KvCache extern with concrete layer index")
            as u32;

        // Resolve QKV gemms via the optional BiasAdd wrapper.
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
            .map(|t| {
                unwrap_gemm_through_bias(fuf, *t)
                    .expect("FusedQkvRopeCache: claim-time check guarantees Gemm-or-BiasAdd(Gemm)")
            })
            .collect();
        let biased = resolved[0].1.is_some();

        // Activation slot — first tile input of the q gemm.
        let q_gemm_id = resolved[0].0;
        let q_gemm_node = fuf.get(q_gemm_id);
        let (in_id, in_slot) = match q_gemm_node.inputs.first() {
            Some(FufInput::Tile { id, slot }) => (*id, *slot),
            other => {
                panic!("FusedQkvRopeCache: q_gemm's first input must be a Tile (got {other:?})")
            }
        };
        let in_slot_idx = slots.of(in_id, in_slot);
        let out_slot_idx = slots.of(rope_id, 0);

        // Packed [q | k | v] LinearLayer accessor.
        let accessors = self.required_weights(&m.claimed_tiles, fuf, program);
        let acc = accessors
            .first()
            .expect("FusedQkvRopeCache: required_weights returned empty");
        let (base, weight_layer) = split_base_layer(&acc.name.to_string());
        let weight_layer = weight_layer.unwrap_or(layer as u64) as u32;
        let base_ident = syn::Ident::new(&base, proc_macro2::Span::call_site());

        // Pick rotary cos_sin source per-claim. Mirrors
        // `rotary_cos_sin_tokens`: any RotaryLocal extern in any
        // claimed tile → use the local accessor. Llama claims never
        // produce `rotary_local_cos_sin` so its absence on Llama's
        // Weights doesn't matter.
        let uses_local = m.claimed_tiles.iter().any(|&tid| {
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
        let cos_sin_ident = if uses_local {
            syn::Ident::new("rotary_local_cos_sin", proc_macro2::Span::call_site())
        } else {
            syn::Ident::new("rotary_cos_sin", proc_macro2::Span::call_site())
        };

        // The variant carries `layer` once; both the kv_cache index and
        // the weight accessor read it. Sanity-check at codegen that the
        // weight accessor's layer suffix agrees with the rope's KvCache
        // layer — if these ever disagree, the DSL has a bug.
        assert_eq!(
            weight_layer, layer,
            "FusedQkvRopeCache: weight_layer ({weight_layer}) and \
             kv_cache layer ({layer}) disagree — DSL bug"
        );
        Some(vec![OpInstance::new(
            syn::Ident::new("FusedQkvRopeCache", proc_macro2::Span::call_site()),
            vec![
                quote! { #in_slot_idx },
                quote! { #out_slot_idx },
                quote! { #layer },
                quote! { Weights::#base_ident },
                quote! { Weights::#cos_sin_ident },
                quote! { #biased },
                quote! { #interleaved },
            ],
        )])
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

    // ── Host-interpreter codegen ────────────────────────────────
    //
    // Variant `FusedQkvQkNormRopeCache { in_slot, out_slot, layer,
    // q_weight_fn, k_weight_fn, v_weight_fn, q_norm_fn, k_norm_fn,
    // cos_sin_fn, q_offset, k_offset }`. Three separate cuBLAS GEMMs
    // (Q/K/V) followed by the fused `qk_norm_rope_inplace` kernel
    // and `reshape_and_cache`. Q's OwnedTensor is reshaped to
    // `[T, num_q_heads, head_dim]` before being stored at out_slot.

    fn opcode_shape(&self) -> OpcodeShape {
        OpcodeShape::new(
            "FusedQkvQkNormRopeCache",
            vec![
                ("in_slot", syn::parse_quote!(u32)),
                ("out_slot", syn::parse_quote!(u32)),
                ("layer", syn::parse_quote!(u32)),
                (
                    "q_weight_fn",
                    syn::parse_quote!(
                        for<'a> fn(&'a Weights, u32) -> &'a ::ferrite_kernels::layers::LinearLayer
                    ),
                ),
                (
                    "k_weight_fn",
                    syn::parse_quote!(
                        for<'a> fn(&'a Weights, u32) -> &'a ::ferrite_kernels::layers::LinearLayer
                    ),
                ),
                (
                    "v_weight_fn",
                    syn::parse_quote!(
                        for<'a> fn(&'a Weights, u32) -> &'a ::ferrite_kernels::layers::LinearLayer
                    ),
                ),
                (
                    "q_norm_fn",
                    syn::parse_quote!(
                        for<'a> fn(&'a Weights, u32) -> &'a ::ferrite_kernels::layers::RmsNorm
                    ),
                ),
                (
                    "k_norm_fn",
                    syn::parse_quote!(
                        for<'a> fn(&'a Weights, u32) -> &'a ::ferrite_kernels::layers::RmsNorm
                    ),
                ),
                (
                    "cos_sin_fn",
                    syn::parse_quote!(
                        for<'a> fn(&'a Weights, u32) -> ::ferrite_cuda_core::tensor::GpuTensor
                    ),
                ),
                ("q_offset", syn::parse_quote!(f32)),
                ("k_offset", syn::parse_quote!(f32)),
            ],
        )
    }

    fn fan_out(
        &self,
        m: &MatchInfo,
        fuf: &Fuf,
        program: &Program,
        _bounds: &BTreeMap<String, u64>,
        slots: &SlotMap,
    ) -> Option<Vec<OpInstance>> {
        let rope_id = *m
            .claimed_tiles
            .iter()
            .find(|t| fuf.get(**t).op == OpKind::RopeAppend)
            .expect("FusedQkvQkNormRopeCache: claim contains a RopeAppend");
        let rope_node = fuf.get(rope_id);
        let layer = rope_kv_cache_layer(rope_node).expect(
            "FusedQkvQkNormRopeCache: RopeAppend has a KvCache extern with concrete layer index",
        ) as u32;

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
        let q_gemm = find_gemm_root_of_norm_chain(fuf, qkv_raw[0])
            .expect("FusedQkvQkNormRopeCache: Q norm chain validated at match time");
        let k_gemm = find_gemm_root_of_norm_chain(fuf, qkv_raw[1])
            .expect("FusedQkvQkNormRopeCache: K norm chain validated at match time");

        // Activation slot — first tile input of any of the three Gemms
        // (matches() guarantees they all share it).
        let q_node = fuf.get(q_gemm);
        let (in_id, in_slot) = match q_node.inputs.first() {
            Some(FufInput::Tile { id, slot }) => (*id, *slot),
            other => panic!(
                "FusedQkvQkNormRopeCache: q_gemm's first input must be a Tile (got {other:?})"
            ),
        };
        let in_slot_idx = slots.of(in_id, in_slot);
        let out_slot_idx = slots.of(rope_id, 0);

        // Per-claim scalar offsets — Gemma3 carries `weight + 1.0`
        // before each per-head norm.
        let (_, _, q_add) =
            walk_qk_norm_chain(fuf, q_gemm).expect("FusedQkvQkNormRopeCache: Q chain valid");
        let (_, _, k_add) =
            walk_qk_norm_chain(fuf, k_gemm).expect("FusedQkvQkNormRopeCache: K chain valid");
        let q_offset: f32 = q_add
            .map(|add_id| {
                fuf.get(add_id)
                    .inputs
                    .iter()
                    .find_map(|i| match i {
                        FufInput::Scalar(v) => Some(*v as f32),
                        _ => None,
                    })
                    .unwrap_or(0.0)
            })
            .unwrap_or(0.0);
        let k_offset: f32 = k_add
            .map(|add_id| {
                fuf.get(add_id)
                    .inputs
                    .iter()
                    .find_map(|i| match i {
                        FufInput::Scalar(v) => Some(*v as f32),
                        _ => None,
                    })
                    .unwrap_or(0.0)
            })
            .unwrap_or(0.0);

        // Resolve the five accessors via required_weights — three
        // LinearLayers (Q/K/V) then two RmsNorms (q_norm/k_norm). Order
        // is fixed by required_weights's push order.
        let accessors = self.required_weights(&m.claimed_tiles, fuf, program);
        assert_eq!(
            accessors.len(),
            5,
            "FusedQkvQkNormRopeCache: expected 5 accessors (q,k,v + q_norm,k_norm)"
        );
        let resolve_acc = |acc: &WeightAccessor| -> (syn::Ident, u32) {
            let (base, l) = split_base_layer(&acc.name.to_string());
            let l = l.unwrap_or(layer as u64) as u32;
            let ident = syn::Ident::new(&base, proc_macro2::Span::call_site());
            (ident, l)
        };
        let (q_w_ident, q_w_layer) = resolve_acc(&accessors[0]);
        let (k_w_ident, k_w_layer) = resolve_acc(&accessors[1]);
        let (v_w_ident, v_w_layer) = resolve_acc(&accessors[2]);
        let (q_n_ident, q_n_layer) = resolve_acc(&accessors[3]);
        let (k_n_ident, k_n_layer) = resolve_acc(&accessors[4]);
        for (l, name) in [
            (q_w_layer, "q_weight"),
            (k_w_layer, "k_weight"),
            (v_w_layer, "v_weight"),
            (q_n_layer, "q_norm"),
            (k_n_layer, "k_norm"),
        ] {
            assert_eq!(
                l, layer,
                "FusedQkvQkNormRopeCache: {name} layer ({l}) disagrees with rope layer ({layer})"
            );
        }
        let _ = (v_id, q_gemm, k_gemm); // captured into accessors, no further use here.

        // Pick rotary cos_sin source per claim.
        let uses_local = m.claimed_tiles.iter().any(|&tid| {
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
        let cos_sin_ident = if uses_local {
            syn::Ident::new("rotary_local_cos_sin", proc_macro2::Span::call_site())
        } else {
            syn::Ident::new("rotary_cos_sin", proc_macro2::Span::call_site())
        };

        Some(vec![OpInstance::new(
            syn::Ident::new("FusedQkvQkNormRopeCache", proc_macro2::Span::call_site()),
            vec![
                quote! { #in_slot_idx },
                quote! { #out_slot_idx },
                quote! { #layer },
                quote! { Weights::#q_w_ident },
                quote! { Weights::#k_w_ident },
                quote! { Weights::#v_w_ident },
                quote! { Weights::#q_n_ident },
                quote! { Weights::#k_n_ident },
                quote! { Weights::#cos_sin_ident },
                quote! { #q_offset },
                quote! { #k_offset },
            ],
        )])
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

    // ── Host-interpreter codegen ────────────────────────────────
    //
    // Variant `AttentionViaCache { in_slot, out_slot, layer,
    // cos_sin_fn, interleaved }`. Decode-only (M=1) paged attention.
    // Reads Q from the upstream rope-cache fusion's tile output;
    // K/V live on `ctx.kv_cache` and are read by the kernel via
    // `block_table`. Scale + softcap are arch-wide (baked at codegen);
    // interleaved is per-instance because Gemma3 layers can differ.

    fn opcode_shape(&self) -> OpcodeShape {
        OpcodeShape::new(
            "AttentionViaCache",
            vec![
                ("in_slot", syn::parse_quote!(u32)),
                ("out_slot", syn::parse_quote!(u32)),
                ("layer", syn::parse_quote!(u32)),
                (
                    "cos_sin_fn",
                    syn::parse_quote!(
                        for<'a> fn(&'a Weights, u32) -> ::ferrite_cuda_core::tensor::GpuTensor
                    ),
                ),
                ("interleaved", syn::parse_quote!(bool)),
            ],
        )
    }

    fn fan_out(
        &self,
        m: &MatchInfo,
        fuf: &Fuf,
        _program: &Program,
        _bounds: &BTreeMap<String, u64>,
        slots: &SlotMap,
    ) -> Option<Vec<OpInstance>> {
        let tile = m.claimed_tiles[0];
        let node = fuf.get(tile);
        let (in_id, in_slot) = match node.inputs.first() {
            Some(FufInput::Tile { id, slot }) => (*id, *slot),
            other => panic!("AttentionViaCache: input 0 (Q) must be a Tile (got {other:?})"),
        };
        let in_slot_idx = slots.of(in_id, in_slot);
        let out_slot_idx = slots.of(tile, 0);
        let layer = node
            .inputs
            .iter()
            .find_map(|i| match i {
                FufInput::Extern {
                    kind: ExternKind::KvCache,
                    index: Some(layer),
                } => Some(*layer),
                _ => None,
            })
            .expect("AttentionViaCache: kv_cache extern with concrete layer index")
            as u32;
        let interleaved = layer_rope_is_interleaved(fuf, layer as u64);

        let uses_local = m.claimed_tiles.iter().any(|&tid| {
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
        let cos_sin_ident = if uses_local {
            syn::Ident::new("rotary_local_cos_sin", proc_macro2::Span::call_site())
        } else {
            syn::Ident::new("rotary_cos_sin", proc_macro2::Span::call_site())
        };

        Some(vec![OpInstance::new(
            syn::Ident::new("AttentionViaCache", proc_macro2::Span::call_site()),
            vec![
                quote! { #in_slot_idx },
                quote! { #out_slot_idx },
                quote! { #layer },
                quote! { Weights::#cos_sin_ident },
                quote! { #interleaved },
            ],
        )])
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
        // upstream pattern matches (three Gemms sharing one activation,
        // optionally through a BiasAdd, uniform bias presence). This
        // gate is the inverse of `output_layouts` — singletons leave
        // K/V at `[T, heads*head_dim]` while fused impls produce
        // `[T, heads, head_dim]`, and the downstream Attention impls
        // expect the latter when a fused upstream is available.
        if rope_append_has_fused_qkv_upstream(fuf, seed) {
            return None;
        }
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

    // ── Host-interpreter codegen ────────────────────────────────
    //
    // Variant `RopeAppend { q_slot, k_slot, v_slot, q_out_slot,
    // k_out_slot, v_out_slot, layer, cos_sin_fn, interleaved }`.
    // Singleton fallback for Qwen3/Gemma3-style rope_append tiles.
    // The arm runs `rotary_embedding[_interleaved]_inplace` against
    // the upstream Q/K/V buffers and `reshape_and_cache` to write
    // K/V to the paged cache. Slot 0/1/2 of the rope tile are
    // 3D `Reshaped` views aliasing the same upstream storage, so
    // downstream attention reads the (T, heads, head_dim) layout.

    fn opcode_shape(&self) -> OpcodeShape {
        OpcodeShape::new(
            "RopeAppend",
            vec![
                ("q_slot", syn::parse_quote!(u32)),
                ("k_slot", syn::parse_quote!(u32)),
                ("v_slot", syn::parse_quote!(u32)),
                ("q_out_slot", syn::parse_quote!(u32)),
                ("k_out_slot", syn::parse_quote!(u32)),
                ("v_out_slot", syn::parse_quote!(u32)),
                ("layer", syn::parse_quote!(u32)),
                (
                    "cos_sin_fn",
                    syn::parse_quote!(
                        for<'a> fn(&'a Weights, u32) -> ::ferrite_cuda_core::tensor::GpuTensor
                    ),
                ),
                ("interleaved", syn::parse_quote!(bool)),
            ],
        )
    }

    fn fan_out(
        &self,
        m: &MatchInfo,
        fuf: &Fuf,
        _program: &Program,
        _bounds: &BTreeMap<String, u64>,
        slots: &SlotMap,
    ) -> Option<Vec<OpInstance>> {
        let rope_id = m.claimed_tiles[0];
        let node = fuf.get(rope_id);
        let interleaved = node.op == OpKind::RopeAppendInterleaved;
        let resolve_tile = |idx: usize| -> (TileId, u8) {
            match node.inputs.get(idx) {
                Some(FufInput::Tile { id, slot }) => (*id, *slot),
                other => panic!("RopeAppend: input {idx} must be a Tile (got {other:?})"),
            }
        };
        let (q_id, q_in) = resolve_tile(0);
        let (k_id, k_in) = resolve_tile(1);
        let (v_id, v_in) = resolve_tile(2);
        let q_slot = slots.of(q_id, q_in);
        let k_slot = slots.of(k_id, k_in);
        let v_slot = slots.of(v_id, v_in);
        let q_out_slot = slots.of(rope_id, 0);
        let k_out_slot = slots.of(rope_id, 1);
        let v_out_slot = slots.of(rope_id, 2);
        let layer =
            node.inputs
                .iter()
                .find_map(|i| match i {
                    FufInput::Extern {
                        kind: ExternKind::KvCache,
                        index: Some(layer),
                    } => Some(*layer),
                    _ => None,
                })
                .expect("RopeAppend: kv_cache extern with concrete layer index") as u32;

        let uses_local = m.claimed_tiles.iter().any(|&tid| {
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
        let cos_sin_ident = if uses_local {
            syn::Ident::new("rotary_local_cos_sin", proc_macro2::Span::call_site())
        } else {
            syn::Ident::new("rotary_cos_sin", proc_macro2::Span::call_site())
        };

        Some(vec![OpInstance::new(
            syn::Ident::new("RopeAppend", proc_macro2::Span::call_site()),
            vec![
                quote! { #q_slot },
                quote! { #k_slot },
                quote! { #v_slot },
                quote! { #q_out_slot },
                quote! { #k_out_slot },
                quote! { #v_out_slot },
                quote! { #layer },
                quote! { Weights::#cos_sin_ident },
                quote! { #interleaved },
            ],
        )])
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

    // ── Host-interpreter codegen ────────────────────────────────
    //
    // Variant `FusedQkvRopePrefill { in_slot, q_out_slot, k_out_slot,
    // v_out_slot, layer, weight_fn, cos_sin_fn, biased, interleaved }`.
    // Three Owned outputs (Q/K/V) since the prefill kernel returns
    // contiguous OwnedTensors instead of writing to the paged cache.
    // After the kernel returns, K/V are committed to the cache via
    // `write_kv_cache` for downstream decode iterations.

    fn opcode_shape(&self) -> OpcodeShape {
        OpcodeShape::new(
            "FusedQkvRopePrefill",
            vec![
                ("in_slot", syn::parse_quote!(u32)),
                ("q_out_slot", syn::parse_quote!(u32)),
                ("k_out_slot", syn::parse_quote!(u32)),
                ("v_out_slot", syn::parse_quote!(u32)),
                ("layer", syn::parse_quote!(u32)),
                (
                    "weight_fn",
                    syn::parse_quote!(
                        for<'a> fn(&'a Weights, u32) -> &'a ::ferrite_kernels::layers::LinearLayer
                    ),
                ),
                (
                    "cos_sin_fn",
                    syn::parse_quote!(
                        for<'a> fn(&'a Weights, u32) -> ::ferrite_cuda_core::tensor::GpuTensor
                    ),
                ),
                ("biased", syn::parse_quote!(bool)),
                ("interleaved", syn::parse_quote!(bool)),
            ],
        )
    }

    fn fan_out(
        &self,
        m: &MatchInfo,
        fuf: &Fuf,
        program: &Program,
        _bounds: &BTreeMap<String, u64>,
        slots: &SlotMap,
    ) -> Option<Vec<OpInstance>> {
        let rope_id = *m
            .claimed_tiles
            .iter()
            .find(|t| is_rope_append_op(fuf.get(**t).op))
            .expect("FusedQkvRopePrefill: claim contains a RopeAppend");
        let rope_node = fuf.get(rope_id);
        let interleaved = rope_node.op == OpKind::RopeAppendInterleaved;
        let layer = rope_kv_cache_layer(rope_node)
            .expect("FusedQkvRopePrefill: RopeAppend has a KvCache extern with layer index")
            as u32;

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
            .map(|t| {
                unwrap_gemm_through_bias(fuf, *t)
                    .expect("FusedQkvRopePrefill: claim guarantees Gemm-or-BiasAdd(Gemm)")
            })
            .collect();
        let biased = resolved[0].1.is_some();

        let q_gemm_id = resolved[0].0;
        let q_gemm_node = fuf.get(q_gemm_id);
        let (in_id, in_slot) = match q_gemm_node.inputs.first() {
            Some(FufInput::Tile { id, slot }) => (*id, *slot),
            other => panic!("FusedQkvRopePrefill: q_gemm input 0 must be a Tile (got {other:?})"),
        };
        let in_slot_idx = slots.of(in_id, in_slot);
        let q_out = slots.of(rope_id, 0);
        let k_out = slots.of(rope_id, 1);
        let v_out = slots.of(rope_id, 2);

        let accessors = self.required_weights(&m.claimed_tiles, fuf, program);
        let acc = accessors
            .first()
            .expect("FusedQkvRopePrefill: required_weights returned empty");
        let (base, weight_layer) = split_base_layer(&acc.name.to_string());
        let weight_layer = weight_layer.unwrap_or(layer as u64) as u32;
        assert_eq!(
            weight_layer, layer,
            "FusedQkvRopePrefill: weight_layer disagrees with kv_cache layer"
        );
        let base_ident = syn::Ident::new(&base, proc_macro2::Span::call_site());

        let uses_local = m.claimed_tiles.iter().any(|&tid| {
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
        let cos_sin_ident = if uses_local {
            syn::Ident::new("rotary_local_cos_sin", proc_macro2::Span::call_site())
        } else {
            syn::Ident::new("rotary_cos_sin", proc_macro2::Span::call_site())
        };

        Some(vec![OpInstance::new(
            syn::Ident::new("FusedQkvRopePrefill", proc_macro2::Span::call_site()),
            vec![
                quote! { #in_slot_idx },
                quote! { #q_out },
                quote! { #k_out },
                quote! { #v_out },
                quote! { #layer },
                quote! { Weights::#base_ident },
                quote! { Weights::#cos_sin_ident },
                quote! { #biased },
                quote! { #interleaved },
            ],
        )])
    }
}

// ── CutlassFusedQkvRope{Cache,Prefill}Impl ──────────────────────
//
// CUTLASS peers to the cuBLAS `FusedQkvRope{Cache,Prefill}Impl` —
// same 4-or-7-tile claim, same boundary inputs/outputs, same
// runtime post-GEMM step (rope+paged-cache write for Cache,
// rope+contiguous-Q/K/V for Prefill). The only structural change
// vs the cuBLAS peer is the GEMM kernel: a calibrated
// `cutlass_<TM>x<TN>_s<S>` standalone-zoo tile in place of cuBLAS.
//
// Tile-zoo-pickable: one Impl per `CUTLASS_TILE_ZOO` entry,
// `target_compatible` gated on calibrated CSV row presence —
// uncalibrated targets silently fall back to the cuBLAS peer.
//
// Bias gating: the v1 Impl rejects biased claims in `matches`. The
// bias-zoo CSV (`cutlass_gemm_bias_<TM>x<TN>_s<S>`) is not yet
// shape-swept across the QKV regime, so a biased pick would race
// against `UNCALIBRATED_COST_US` and silently lose to cuBLAS.
// Qwen2's biased QKV consequently keeps the cuBLAS path until the
// bias-zoo sweep lands.

/// True iff the claim's three QKV gemms are wrapped in `BiasAdd`
/// (Qwen2/Qwen3-style explicit QKV bias).
fn fused_qkv_claim_is_biased(fuf: &Fuf, info: &MatchInfo) -> bool {
    let Some(rope_id) = info
        .claimed_tiles
        .iter()
        .find(|t| is_rope_append_op(fuf.get(**t).op))
        .copied()
    else {
        return false;
    };
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
    qkv_raw
        .iter()
        .filter_map(|t| unwrap_gemm_through_bias(fuf, *t))
        .next()
        .is_some_and(|(_, bias)| bias.is_some())
}

#[derive(Debug, Clone)]
pub struct CutlassFusedQkvRopeCacheImpl {
    pub tile_m: u32,
    pub tile_n: u32,
    pub stages: u32,
}

impl CutlassFusedQkvRopeCacheImpl {
    fn csv_name(&self) -> &'static str {
        // Reuses the standalone CutlassGemm tile names — the GEMM
        // step is an ordinary cutlass_<TM>x<TN>_s<S> at packed
        // N=q+2*kv.
        match (self.tile_m, self.tile_n, self.stages) {
            (16, 64, 3) => "cutlass_16x64_s3",
            (16, 64, 4) => "cutlass_16x64_s4",
            (16, 128, 3) => "cutlass_16x128_s3",
            (16, 128, 4) => "cutlass_16x128_s4",
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

impl Implementation for CutlassFusedQkvRopeCacheImpl {
    fn name(&self) -> &'static str {
        self.csv_name()
    }

    fn target_compatible(&self, profile: &TargetProfile) -> bool {
        profile.cost_table.has_kernel(self.csv_name())
    }

    fn workload_constraint(&self) -> WorkloadConstraint {
        FusedQkvRopeCacheImpl.workload_constraint()
    }

    fn matches(&self, fuf: &Fuf, seed: TileId, profile: &TargetProfile) -> Option<MatchInfo> {
        let info = FusedQkvRopeCacheImpl.matches(fuf, seed, profile)?;
        if fused_qkv_claim_is_biased(fuf, &info) {
            return None;
        }
        Some(info)
    }

    fn cost_us(&self, m: &MatchInfo, ctx: &CostCtx) -> f64 {
        // Same shape as FusedQkvRopeCacheImpl::cost_us (cuBLAS peer):
        //   - one GEMM at packed (M, q+2*kv, hidden)
        //   - one BW-bound rope+cache pass over the same buffer
        // Difference: the GEMM is a calibrated cutlass_<tile> row,
        // not cublas. Reduces to cutlass_<tile>(M, packed_n, K) vs
        // cublas(M, packed_n, K).
        let num_tokens = ctx.num_tokens() as u32;
        let hidden = ctx.bounds.get("hidden_size").copied().unwrap_or(0) as u32;
        let num_q_heads = ctx.bounds.get("num_attention_heads").copied().unwrap_or(0) as u32;
        let num_kv_heads = ctx.bounds.get("num_key_value_heads").copied().unwrap_or(0) as u32;
        let head_dim = ctx.bounds.get("head_dim").copied().unwrap_or(0) as u32;
        let packed_n = num_q_heads
            .saturating_add(2u32.saturating_mul(num_kv_heads))
            .saturating_mul(head_dim);

        let gemm_us = ctx
            .profile
            .cost_us_for(self.csv_name(), num_tokens, packed_n, hidden)
            .unwrap_or(UNCALIBRATED_COST_US);

        let bw_gb = ctx.profile.memory_bandwidth_gbps;
        let bytes = 3.0 * (num_tokens as f64) * (packed_n as f64) * BYTES_PER_ELEM;
        let rope_cache_us = if bw_gb > 0.0 {
            (bytes / (bw_gb * 1e9)) * 1e6
        } else {
            0.0
        };

        let _ = m;
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
        FusedQkvRopeCacheImpl.output_alias(claimed_tiles, fuf)
    }

    fn required_weights(
        &self,
        claimed_tiles: &[TileId],
        fuf: &Fuf,
        program: &Program,
    ) -> Vec<WeightAccessor> {
        FusedQkvRopeCacheImpl.required_weights(claimed_tiles, fuf, program)
    }

    fn opcode_shape(&self) -> OpcodeShape {
        OpcodeShape::new(
            "CutlassFusedQkvRopeCache",
            vec![
                ("in_slot", syn::parse_quote!(u32)),
                ("out_slot", syn::parse_quote!(u32)),
                ("layer", syn::parse_quote!(u32)),
                (
                    "weight_fn",
                    syn::parse_quote!(
                        for<'a> fn(&'a Weights, u32) -> &'a ::ferrite_kernels::layers::LinearLayer
                    ),
                ),
                (
                    "cos_sin_fn",
                    syn::parse_quote!(
                        for<'a> fn(&'a Weights, u32) -> ::ferrite_cuda_core::tensor::GpuTensor
                    ),
                ),
                ("interleaved", syn::parse_quote!(bool)),
                ("tile_m", syn::parse_quote!(u32)),
                ("tile_n", syn::parse_quote!(u32)),
                ("stages", syn::parse_quote!(u32)),
                ("packed_n", syn::parse_quote!(u32)),
                ("k", syn::parse_quote!(u32)),
            ],
        )
    }

    fn fan_out(
        &self,
        m: &MatchInfo,
        fuf: &Fuf,
        program: &Program,
        bounds: &BTreeMap<String, u64>,
        slots: &SlotMap,
    ) -> Option<Vec<OpInstance>> {
        let rope_id = *m
            .claimed_tiles
            .iter()
            .find(|t| is_rope_append_op(fuf.get(**t).op))
            .expect("CutlassFusedQkvRopeCache: claim contains a RopeAppend");
        let rope_node = fuf.get(rope_id);
        let interleaved = rope_node.op == OpKind::RopeAppendInterleaved;
        let layer = rope_kv_cache_layer(rope_node).expect(
            "CutlassFusedQkvRopeCache: RopeAppend has a KvCache extern with concrete layer index",
        ) as u32;

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
            .map(|t| {
                unwrap_gemm_through_bias(fuf, *t)
                    .expect("CutlassFusedQkvRopeCache: claim guarantees Gemm-or-BiasAdd(Gemm)")
            })
            .collect();
        // matches() rejected biased claims; runtime does not handle bias.
        debug_assert!(
            resolved[0].1.is_none(),
            "CutlassFusedQkvRopeCache: matches() should have rejected biased claim",
        );

        let q_gemm_id = resolved[0].0;
        let q_gemm_node = fuf.get(q_gemm_id);
        let (in_id, in_slot) = match q_gemm_node.inputs.first() {
            Some(FufInput::Tile { id, slot }) => (*id, *slot),
            other => panic!(
                "CutlassFusedQkvRopeCache: q_gemm's first input must be a Tile (got {other:?})"
            ),
        };
        let in_slot_idx = slots.of(in_id, in_slot);
        let out_slot_idx = slots.of(rope_id, 0);

        let accessors = self.required_weights(&m.claimed_tiles, fuf, program);
        let acc = accessors
            .first()
            .expect("CutlassFusedQkvRopeCache: required_weights returned empty");
        let (base, weight_layer) = split_base_layer(&acc.name.to_string());
        let weight_layer = weight_layer.unwrap_or(layer as u64) as u32;
        assert_eq!(
            weight_layer, layer,
            "CutlassFusedQkvRopeCache: weight_layer ({weight_layer}) and \
             kv_cache layer ({layer}) disagree — DSL bug"
        );
        let base_ident = syn::Ident::new(&base, proc_macro2::Span::call_site());

        let uses_local = m.claimed_tiles.iter().any(|&tid| {
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
        let cos_sin_ident = if uses_local {
            syn::Ident::new("rotary_local_cos_sin", proc_macro2::Span::call_site())
        } else {
            syn::Ident::new("rotary_cos_sin", proc_macro2::Span::call_site())
        };

        // Bake (packed_n=q+2*kv, k=hidden) for runtime weight-shape assert.
        // packed_n derives from arch-level `Weights::*_SIZE` at runtime,
        // but we re-derive it here from FUF + bounds for the claim's q
        // gemm so the assert can fire at codegen time if loader output
        // disagrees with FUF expectations.
        let (q_n, k) = gemm_nk_from_fuf(fuf, q_gemm_node, bounds)
            .expect("CutlassFusedQkvRopeCache: q gemm (N, K) must resolve from FUF + bounds");
        let num_kv_heads = bounds.get("num_key_value_heads").copied().unwrap_or(0) as u32;
        let head_dim = bounds.get("head_dim").copied().unwrap_or(0) as u32;
        let kv_size = num_kv_heads.saturating_mul(head_dim);
        let packed_n = q_n.saturating_add(kv_size.saturating_mul(2));
        let tile_m = self.tile_m;
        let tile_n = self.tile_n;
        let stages = self.stages;
        Some(vec![OpInstance::new(
            syn::Ident::new("CutlassFusedQkvRopeCache", proc_macro2::Span::call_site()),
            vec![
                quote! { #in_slot_idx },
                quote! { #out_slot_idx },
                quote! { #layer },
                quote! { Weights::#base_ident },
                quote! { Weights::#cos_sin_ident },
                quote! { #interleaved },
                quote! { #tile_m },
                quote! { #tile_n },
                quote! { #stages },
                quote! { #packed_n },
                quote! { #k },
            ],
        )])
    }
}

#[derive(Debug, Clone)]
pub struct CutlassFusedQkvRopePrefillImpl {
    pub tile_m: u32,
    pub tile_n: u32,
    pub stages: u32,
}

impl CutlassFusedQkvRopePrefillImpl {
    fn csv_name(&self) -> &'static str {
        CutlassFusedQkvRopeCacheImpl {
            tile_m: self.tile_m,
            tile_n: self.tile_n,
            stages: self.stages,
        }
        .csv_name()
    }
}

impl Implementation for CutlassFusedQkvRopePrefillImpl {
    fn name(&self) -> &'static str {
        self.csv_name()
    }

    fn target_compatible(&self, profile: &TargetProfile) -> bool {
        profile.cost_table.has_kernel(self.csv_name())
    }

    fn workload_constraint(&self) -> WorkloadConstraint {
        FusedQkvRopePrefillImpl.workload_constraint()
    }

    fn matches(&self, fuf: &Fuf, seed: TileId, profile: &TargetProfile) -> Option<MatchInfo> {
        let info = FusedQkvRopePrefillImpl.matches(fuf, seed, profile)?;
        if fused_qkv_claim_is_biased(fuf, &info) {
            return None;
        }
        Some(info)
    }

    fn cost_us(&self, m: &MatchInfo, ctx: &CostCtx) -> f64 {
        // Same flops + bandwidth model as the decode-fused variant;
        // the kernel split is different but the work is the same.
        CutlassFusedQkvRopeCacheImpl {
            tile_m: self.tile_m,
            tile_n: self.tile_n,
            stages: self.stages,
        }
        .cost_us(m, ctx)
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
        FusedQkvRopePrefillImpl.required_weights(claimed_tiles, fuf, program)
    }

    fn opcode_shape(&self) -> OpcodeShape {
        OpcodeShape::new(
            "CutlassFusedQkvRopePrefill",
            vec![
                ("in_slot", syn::parse_quote!(u32)),
                ("q_out_slot", syn::parse_quote!(u32)),
                ("k_out_slot", syn::parse_quote!(u32)),
                ("v_out_slot", syn::parse_quote!(u32)),
                ("layer", syn::parse_quote!(u32)),
                (
                    "weight_fn",
                    syn::parse_quote!(
                        for<'a> fn(&'a Weights, u32) -> &'a ::ferrite_kernels::layers::LinearLayer
                    ),
                ),
                (
                    "cos_sin_fn",
                    syn::parse_quote!(
                        for<'a> fn(&'a Weights, u32) -> ::ferrite_cuda_core::tensor::GpuTensor
                    ),
                ),
                ("interleaved", syn::parse_quote!(bool)),
                ("tile_m", syn::parse_quote!(u32)),
                ("tile_n", syn::parse_quote!(u32)),
                ("stages", syn::parse_quote!(u32)),
                ("packed_n", syn::parse_quote!(u32)),
                ("k", syn::parse_quote!(u32)),
            ],
        )
    }

    fn fan_out(
        &self,
        m: &MatchInfo,
        fuf: &Fuf,
        program: &Program,
        bounds: &BTreeMap<String, u64>,
        slots: &SlotMap,
    ) -> Option<Vec<OpInstance>> {
        let rope_id = *m
            .claimed_tiles
            .iter()
            .find(|t| is_rope_append_op(fuf.get(**t).op))
            .expect("CutlassFusedQkvRopePrefill: claim contains a RopeAppend");
        let rope_node = fuf.get(rope_id);
        let interleaved = rope_node.op == OpKind::RopeAppendInterleaved;
        let layer = rope_kv_cache_layer(rope_node)
            .expect("CutlassFusedQkvRopePrefill: RopeAppend has a KvCache extern with layer index")
            as u32;

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
            .map(|t| {
                unwrap_gemm_through_bias(fuf, *t)
                    .expect("CutlassFusedQkvRopePrefill: claim guarantees Gemm-or-BiasAdd(Gemm)")
            })
            .collect();
        debug_assert!(
            resolved[0].1.is_none(),
            "CutlassFusedQkvRopePrefill: matches() should have rejected biased claim",
        );

        let q_gemm_id = resolved[0].0;
        let q_gemm_node = fuf.get(q_gemm_id);
        let (in_id, in_slot) = match q_gemm_node.inputs.first() {
            Some(FufInput::Tile { id, slot }) => (*id, *slot),
            other => {
                panic!("CutlassFusedQkvRopePrefill: q_gemm input 0 must be a Tile (got {other:?})")
            }
        };
        let in_slot_idx = slots.of(in_id, in_slot);
        let q_out = slots.of(rope_id, 0);
        let k_out = slots.of(rope_id, 1);
        let v_out = slots.of(rope_id, 2);

        let accessors = self.required_weights(&m.claimed_tiles, fuf, program);
        let acc = accessors
            .first()
            .expect("CutlassFusedQkvRopePrefill: required_weights returned empty");
        let (base, weight_layer) = split_base_layer(&acc.name.to_string());
        let weight_layer = weight_layer.unwrap_or(layer as u64) as u32;
        assert_eq!(
            weight_layer, layer,
            "CutlassFusedQkvRopePrefill: weight_layer disagrees with kv_cache layer"
        );
        let base_ident = syn::Ident::new(&base, proc_macro2::Span::call_site());

        let uses_local = m.claimed_tiles.iter().any(|&tid| {
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
        let cos_sin_ident = if uses_local {
            syn::Ident::new("rotary_local_cos_sin", proc_macro2::Span::call_site())
        } else {
            syn::Ident::new("rotary_cos_sin", proc_macro2::Span::call_site())
        };

        let (q_n, k) = gemm_nk_from_fuf(fuf, q_gemm_node, bounds)
            .expect("CutlassFusedQkvRopePrefill: q gemm (N, K) must resolve from FUF + bounds");
        let num_kv_heads = bounds.get("num_key_value_heads").copied().unwrap_or(0) as u32;
        let head_dim = bounds.get("head_dim").copied().unwrap_or(0) as u32;
        let kv_size = num_kv_heads.saturating_mul(head_dim);
        let packed_n = q_n.saturating_add(kv_size.saturating_mul(2));
        let tile_m = self.tile_m;
        let tile_n = self.tile_n;
        let stages = self.stages;
        Some(vec![OpInstance::new(
            syn::Ident::new("CutlassFusedQkvRopePrefill", proc_macro2::Span::call_site()),
            vec![
                quote! { #in_slot_idx },
                quote! { #q_out },
                quote! { #k_out },
                quote! { #v_out },
                quote! { #layer },
                quote! { Weights::#base_ident },
                quote! { Weights::#cos_sin_ident },
                quote! { #interleaved },
                quote! { #tile_m },
                quote! { #tile_n },
                quote! { #stages },
                quote! { #packed_n },
                quote! { #k },
            ],
        )])
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

    // ── Host-interpreter codegen ────────────────────────────────
    //
    // Variant `AttentionPrefillContiguous { q_slot, k_slot, v_slot,
    // out_slot }`. Reads contiguous Q/K/V from upstream
    // FusedQkvRopePrefill outputs; calls flash_attn_contiguous with
    // baked scale/softcap. No layer / cos_sin needed (K is already
    // rotated upstream; rotary_dim=0 so flash-attn skips fused RoPE).

    fn opcode_shape(&self) -> OpcodeShape {
        OpcodeShape::new(
            "AttentionPrefillContiguous",
            vec![
                ("q_slot", syn::parse_quote!(u32)),
                ("k_slot", syn::parse_quote!(u32)),
                ("v_slot", syn::parse_quote!(u32)),
                ("out_slot", syn::parse_quote!(u32)),
                ("interleaved", syn::parse_quote!(bool)),
            ],
        )
    }

    fn fan_out(
        &self,
        m: &MatchInfo,
        fuf: &Fuf,
        _program: &Program,
        _bounds: &BTreeMap<String, u64>,
        slots: &SlotMap,
    ) -> Option<Vec<OpInstance>> {
        let tile = m.claimed_tiles[0];
        let node = fuf.get(tile);
        let resolve = |idx: usize| -> (TileId, u8) {
            match node.inputs.get(idx) {
                Some(FufInput::Tile { id, slot }) => (*id, *slot),
                other => {
                    panic!("AttentionPrefillContiguous: input {idx} must be a Tile (got {other:?})")
                }
            }
        };
        let (q_id, q_in) = resolve(0);
        let (k_id, k_in) = resolve(1);
        let (v_id, v_in) = resolve(2);
        let q_slot = slots.of(q_id, q_in);
        let k_slot = slots.of(k_id, k_in);
        let v_slot = slots.of(v_id, v_in);
        let out_slot = slots.of(tile, 0);

        let interleaved = node
            .inputs
            .iter()
            .find_map(|i| match i {
                FufInput::Extern {
                    kind: ExternKind::KvCache,
                    index: Some(layer),
                } => Some(*layer),
                _ => None,
            })
            .map(|l| layer_rope_is_interleaved(fuf, l))
            .unwrap_or(false);

        Some(vec![OpInstance::new(
            syn::Ident::new("AttentionPrefillContiguous", proc_macro2::Span::call_site()),
            vec![
                quote! { #q_slot },
                quote! { #k_slot },
                quote! { #v_slot },
                quote! { #out_slot },
                quote! { #interleaved },
            ],
        )])
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

    // ── Host-interpreter codegen ────────────────────────────────
    //
    // Variant `SlidingAttentionViaCache { in_slot, out_slot, layer,
    // cos_sin_fn, interleaved }`. Same shape as `AttentionViaCache`;
    // the only difference is the body bakes the sliding-window left
    // bound from `model.bounds["sliding_window"]` into the kernel
    // call instead of passing `-1`.

    fn opcode_shape(&self) -> OpcodeShape {
        OpcodeShape::new(
            "SlidingAttentionViaCache",
            vec![
                ("in_slot", syn::parse_quote!(u32)),
                ("out_slot", syn::parse_quote!(u32)),
                ("layer", syn::parse_quote!(u32)),
                (
                    "cos_sin_fn",
                    syn::parse_quote!(
                        for<'a> fn(&'a Weights, u32) -> ::ferrite_cuda_core::tensor::GpuTensor
                    ),
                ),
                ("interleaved", syn::parse_quote!(bool)),
            ],
        )
    }

    fn fan_out(
        &self,
        m: &MatchInfo,
        fuf: &Fuf,
        _program: &Program,
        _bounds: &BTreeMap<String, u64>,
        slots: &SlotMap,
    ) -> Option<Vec<OpInstance>> {
        let tile = m.claimed_tiles[0];
        let node = fuf.get(tile);
        let (in_id, in_slot) = match node.inputs.first() {
            Some(FufInput::Tile { id, slot }) => (*id, *slot),
            other => panic!("SlidingAttentionViaCache: input 0 (Q) must be a Tile (got {other:?})"),
        };
        let in_slot_idx = slots.of(in_id, in_slot);
        let out_slot_idx = slots.of(tile, 0);
        let layer = node
            .inputs
            .iter()
            .find_map(|i| match i {
                FufInput::Extern {
                    kind: ExternKind::KvCache,
                    index: Some(layer),
                } => Some(*layer),
                _ => None,
            })
            .expect("SlidingAttentionViaCache: kv_cache extern with layer index")
            as u32;
        let interleaved = layer_rope_is_interleaved(fuf, layer as u64);
        let uses_local = m.claimed_tiles.iter().any(|&tid| {
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
        let cos_sin_ident = if uses_local {
            syn::Ident::new("rotary_local_cos_sin", proc_macro2::Span::call_site())
        } else {
            syn::Ident::new("rotary_cos_sin", proc_macro2::Span::call_site())
        };
        Some(vec![OpInstance::new(
            syn::Ident::new("SlidingAttentionViaCache", proc_macro2::Span::call_site()),
            vec![
                quote! { #in_slot_idx },
                quote! { #out_slot_idx },
                quote! { #layer },
                quote! { Weights::#cos_sin_ident },
                quote! { #interleaved },
            ],
        )])
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

    // ── Host-interpreter codegen ────────────────────────────────
    //
    // Variant `SlidingAttentionPrefillContiguous { q_slot, k_slot,
    // v_slot, out_slot, interleaved }`. Same shape as
    // `AttentionPrefillContiguous` plus the sliding window value
    // baked from `model.bounds["sliding_window"]`.

    fn opcode_shape(&self) -> OpcodeShape {
        OpcodeShape::new(
            "SlidingAttentionPrefillContiguous",
            vec![
                ("q_slot", syn::parse_quote!(u32)),
                ("k_slot", syn::parse_quote!(u32)),
                ("v_slot", syn::parse_quote!(u32)),
                ("out_slot", syn::parse_quote!(u32)),
                ("interleaved", syn::parse_quote!(bool)),
            ],
        )
    }

    fn fan_out(
        &self,
        m: &MatchInfo,
        fuf: &Fuf,
        _program: &Program,
        _bounds: &BTreeMap<String, u64>,
        slots: &SlotMap,
    ) -> Option<Vec<OpInstance>> {
        let tile = m.claimed_tiles[0];
        let node = fuf.get(tile);
        let resolve = |idx: usize| -> (TileId, u8) {
            match node.inputs.get(idx) {
                Some(FufInput::Tile { id, slot }) => (*id, *slot),
                other => panic!(
                    "SlidingAttentionPrefillContiguous: input {idx} must be a Tile (got {other:?})"
                ),
            }
        };
        let (q_id, q_in) = resolve(0);
        let (k_id, k_in) = resolve(1);
        let (v_id, v_in) = resolve(2);
        let q_slot = slots.of(q_id, q_in);
        let k_slot = slots.of(k_id, k_in);
        let v_slot = slots.of(v_id, v_in);
        let out_slot = slots.of(tile, 0);
        let interleaved = node
            .inputs
            .iter()
            .find_map(|i| match i {
                FufInput::Extern {
                    kind: ExternKind::KvCache,
                    index: Some(layer),
                } => Some(*layer),
                _ => None,
            })
            .map(|l| layer_rope_is_interleaved(fuf, l))
            .unwrap_or(false);
        Some(vec![OpInstance::new(
            syn::Ident::new(
                "SlidingAttentionPrefillContiguous",
                proc_macro2::Span::call_site(),
            ),
            vec![
                quote! { #q_slot },
                quote! { #k_slot },
                quote! { #v_slot },
                quote! { #out_slot },
                quote! { #interleaved },
            ],
        )])
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
    // tile_m=16 — minimum threadblock M with sm80 tensor cores. Targets
    // small-M tall-skinny lm_head + QKV regimes where M ∈ [8, 16] and
    // larger tile_m wastes most of the threadblock's M dimension.
    // (16, 256) excluded — CUTLASS thread-map div-by-zero at that aspect.
    (16, 64, 3),
    (16, 64, 4),
    (16, 128, 3),
    (16, 128, 4),
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
            (16, 64, 3) => "cutlass_16x64_s3",
            (16, 64, 4) => "cutlass_16x64_s4",
            (16, 128, 3) => "cutlass_16x128_s3",
            (16, 128, 4) => "cutlass_16x128_s4",
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

/// Evaluate the `(M, N, K)` of a Gemm tile for CSV cost lookup.
/// M comes from the current workload (bounds[`num_tokens`]), N from
/// the output's last dim, K from the activation-input tile's last
/// dim.
fn gemm_mnk(ctx: &CostCtx, node: &crate::fuf::FufNode) -> Option<(u32, u32, u32)> {
    let m = ctx.num_tokens() as u32;
    let (n, k) = gemm_nk_from_fuf(ctx.fuf, node, ctx.bounds)?;
    Some((m, n, k))
}

/// Static (N, K) of a Gemm tile from its FUF node + variable bounds.
/// `N` is the tile's output last dim (out_features); `K` is the
/// input's last dim (in_features). Available to `fan_out` impls so
/// they can bake weight shape into the emitted `Instruction` for the
/// runtime shape-assert that guards against loader/codegen drift.
fn gemm_nk_from_fuf(
    fuf: &Fuf,
    node: &crate::fuf::FufNode,
    bounds: &BTreeMap<String, u64>,
) -> Option<(u32, u32)> {
    let out_shape = node
        .outputs
        .first()
        .and_then(|s| eval_shape_with(s, bounds))?;
    if out_shape.len() != 2 {
        return None;
    }
    let n = *out_shape.last()? as u32;
    let k = node.inputs.iter().find_map(|inp| match inp {
        FufInput::Tile { id, slot } => {
            let up = fuf.get(*id);
            up.outputs
                .get(*slot as usize)
                .and_then(|s| eval_shape_with(s, bounds))
                .and_then(|v| v.last().copied())
        }
        _ => None,
    })? as u32;
    Some((n, k))
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

    // ── Host-interpreter codegen ────────────────────────────────
    //
    // Variant `CutlassGemm { in_slot, out_slot, layer, weight_fn,
    // tile_m, tile_n, stages }`. All `CutlassGemmImpl{tile_m=…,
    // tile_n=…, stages=…}` instances share this variant — the tile
    // config rides as runtime fields. The body always calls
    // `cutlass_gemm(...)` with `CutlassTile::new(tile_m, tile_n, stages)`.

    fn opcode_shape(&self) -> OpcodeShape {
        OpcodeShape::new(
            "CutlassGemm",
            vec![
                ("in_slot", syn::parse_quote!(u32)),
                ("out_slot", syn::parse_quote!(u32)),
                ("layer", syn::parse_quote!(u32)),
                (
                    "weight_fn",
                    syn::parse_quote!(
                        for<'a> fn(&'a Weights, u32) -> &'a ::ferrite_kernels::layers::LinearLayer
                    ),
                ),
                ("tile_m", syn::parse_quote!(u32)),
                ("tile_n", syn::parse_quote!(u32)),
                ("stages", syn::parse_quote!(u32)),
                ("n", syn::parse_quote!(u32)),
                ("k", syn::parse_quote!(u32)),
            ],
        )
    }

    fn fan_out(
        &self,
        m: &MatchInfo,
        fuf: &Fuf,
        program: &Program,
        bounds: &BTreeMap<String, u64>,
        slots: &SlotMap,
    ) -> Option<Vec<OpInstance>> {
        let tile = m.claimed_tiles[0];
        let node = fuf.get(tile);
        let (in_id, in_slot) = match node.inputs.first() {
            Some(FufInput::Tile { id, slot }) => (*id, *slot),
            other => panic!("CutlassGemm: input 0 must be a Tile (got {other:?})"),
        };
        let in_slot_idx = slots.of(in_id, in_slot);
        let out_slot_idx = slots.of(tile, 0);
        let accessors = self.required_weights(&m.claimed_tiles, fuf, program);
        let acc = accessors
            .first()
            .expect("CutlassGemm: required_weights returned empty");
        let (base, layer) = split_base_layer(&acc.name.to_string());
        let layer = layer.unwrap_or(0) as u32;
        let base_ident = syn::Ident::new(&base, proc_macro2::Span::call_site());
        let tile_m = self.tile_m;
        let tile_n = self.tile_n;
        let stages = self.stages;
        let (n, k) = gemm_nk_from_fuf(fuf, node, bounds)
            .expect("CutlassGemm: weight (N, K) must resolve from FUF + bounds");
        Some(vec![OpInstance::new(
            syn::Ident::new("CutlassGemm", proc_macro2::Span::call_site()),
            vec![
                quote! { #in_slot_idx },
                quote! { #out_slot_idx },
                quote! { #layer },
                quote! { Weights::#base_ident },
                quote! { #tile_m },
                quote! { #tile_n },
                quote! { #stages },
                quote! { #n },
                quote! { #k },
            ],
        )])
    }
}

// ── DcCutlassGemmImpl ────────────────────────────────────────────
//
// PrimMega sibling to [`CutlassGemmImpl`]. Iterates the SAME
// `CUTLASS_TILE_ZOO` the host registration walks — drift between
// the DC and host tile sets is structurally impossible because
// they share the source list. Each (tile_m, tile_n, stages) tuple
// corresponds to a typedef + switch-arm pair in
// `vllm-cuda/csrc/megakernel/prim_mega.cu`'s `CUTLASS_DC_GEMM_LIST`
// expansion (`dc_cutlass.cuh` :: `dc_gemm<DcGemm_*>`); the C++
// X-macro list is pinned to `CUTLASS_TILE_ZOO` by the
// `dc_zoo_equals_host_zoo` test below.
//
// Cost delegates to CutlassGemmImpl which reads the same CSV row
// (`cutlass_<TBM>x<TBN>_s<S>`). The DC sibling carries no extra
// per-tile state; the trait-object wrapper holds (tile_m, tile_n,
// stages) and constructs a fresh CutlassGemmImpl on every call.

#[derive(Debug, Clone)]
pub struct DcCutlassGemmImpl {
    pub tile_m: u32,
    pub tile_n: u32,
    pub stages: u32,
}

impl DcCutlassGemmImpl {
    fn host(&self) -> CutlassGemmImpl {
        CutlassGemmImpl {
            tile_m: self.tile_m,
            tile_n: self.tile_n,
            stages: self.stages,
        }
    }
}

impl Implementation for DcCutlassGemmImpl {
    fn name(&self) -> &'static str {
        // The host counterpart's `name()` is `static_name()` — a
        // tile-shape-keyed `&'static str`. We mirror it under a
        // `dc_` prefix so cost-table / debug logs disambiguate the
        // two siblings while keeping the same per-tile string set.
        // One arm per CUTLASS_TILE_ZOO entry. Drift caught by
        // `dc_cutlass_name_covers_full_zoo` test below.
        match (self.tile_m, self.tile_n, self.stages) {
            (32, 64, 3) => "dc_cutlass_32x64_s3",
            (32, 64, 4) => "dc_cutlass_32x64_s4",
            (32, 128, 3) => "dc_cutlass_32x128_s3",
            (32, 128, 4) => "dc_cutlass_32x128_s4",
            (32, 256, 3) => "dc_cutlass_32x256_s3",
            (64, 64, 3) => "dc_cutlass_64x64_s3",
            (64, 64, 4) => "dc_cutlass_64x64_s4",
            (64, 128, 3) => "dc_cutlass_64x128_s3",
            (64, 128, 4) => "dc_cutlass_64x128_s4",
            (128, 64, 3) => "dc_cutlass_128x64_s3",
            (128, 64, 4) => "dc_cutlass_128x64_s4",
            (128, 128, 3) => "dc_cutlass_128x128_s3",
            (128, 128, 4) => "dc_cutlass_128x128_s4",
            (128, 256, 3) => "dc_cutlass_128x256_s3",
            (256, 64, 3) => "dc_cutlass_256x64_s3",
            (256, 64, 4) => "dc_cutlass_256x64_s4",
            _ => "dc_cutlass_unknown",
        }
    }
    fn target_compatible(&self, profile: &TargetProfile) -> bool {
        profile.prim_mega_compatible() && self.host().target_compatible(profile)
    }
    fn workload_constraint(&self) -> WorkloadConstraint {
        self.host().workload_constraint()
    }
    fn matches(&self, fuf: &Fuf, seed: TileId, profile: &TargetProfile) -> Option<MatchInfo> {
        self.host().matches(fuf, seed, profile)
    }
    fn cost_us(&self, m: &MatchInfo, ctx: &CostCtx) -> f64 {
        self.host().cost_us(m, ctx)
    }
    fn resources(&self, m: &MatchInfo) -> Resources {
        self.host().resources(m)
    }
    fn launch_kind(&self) -> LaunchKind {
        LaunchKind::DeviceCallable
    }
    fn megakernel_fit(&self) -> MegakernelFit {
        MegakernelFit::Primitive
    }
    fn supported_input_handoffs(&self) -> &[Handoff] {
        DC_HANDOFFS
    }
    fn supported_output_handoffs(&self) -> &[Handoff] {
        DC_HANDOFFS
    }
    fn input_layouts(&self, m: &MatchInfo) -> Vec<Layout> {
        self.host().input_layouts(m)
    }
    fn output_layouts(&self, m: &MatchInfo) -> Vec<Layout> {
        self.host().output_layouts(m)
    }
    fn is_compute_bound(&self) -> bool {
        self.host().is_compute_bound()
    }
    fn can_share_kernel_with(&self, _other: &dyn Implementation) -> bool {
        true
    }
    fn output_alias(
        &self,
        claimed_tiles: &[TileId],
        fuf: &Fuf,
    ) -> Vec<((TileId, u8), Option<(TileId, u8)>)> {
        self.host().output_alias(claimed_tiles, fuf)
    }
    fn consumes_input_tiles(&self, claimed_tiles: &[TileId], fuf: &Fuf) -> Vec<(TileId, u8)> {
        self.host().consumes_input_tiles(claimed_tiles, fuf)
    }
    fn required_weights(
        &self,
        claimed_tiles: &[TileId],
        fuf: &Fuf,
        program: &Program,
    ) -> Vec<WeightAccessor> {
        self.host().required_weights(claimed_tiles, fuf, program)
    }
    fn opcode_shape(&self) -> OpcodeShape {
        self.host().opcode_shape()
    }
    fn fan_out(
        &self,
        m: &MatchInfo,
        fuf: &Fuf,
        program: &Program,
        bounds: &BTreeMap<String, u64>,
        slots: &SlotMap,
    ) -> Option<Vec<OpInstance>> {
        self.host().fan_out(m, fuf, program, bounds, slots)
    }
}

// ── DcCutlassGemmAddImpl ─────────────────────────────────────────
//
// PrimMega sibling to [`CutlassGemmAddImpl`]. The host counterpart's
// kernel is the SAME `device::Gemm` instantiation as the bare GEMM —
// the residual-add is just a runtime `beta=1.0` epilogue. So no new
// C++ template is needed: the existing `dc_cutlass.cuh::dc_gemm`
// template already accepts `beta` as a parameter; the encoder just
// emits `beta=1.0` at the per-op call site instead of `beta=0.0`.
//
// Cost delegates to CutlassGemmAddImpl which reads the dedicated
// `cutlass_<TBM>x<TBN>_s<S>_add` CSV row (measured at beta=1.0).
// `target_compatible` mirrors the host's CSV-row gate so a tile
// without `_add` calibration silently filters out on Ada (and on
// any target whose CSV is missing the row).

#[derive(Debug, Clone)]
pub struct DcCutlassGemmAddImpl {
    pub tile_m: u32,
    pub tile_n: u32,
    pub stages: u32,
}

impl DcCutlassGemmAddImpl {
    fn host(&self) -> CutlassGemmAddImpl {
        CutlassGemmAddImpl {
            tile_m: self.tile_m,
            tile_n: self.tile_n,
            stages: self.stages,
        }
    }
}

impl Implementation for DcCutlassGemmAddImpl {
    fn name(&self) -> &'static str {
        // One arm per CUTLASS_TILE_ZOO entry. Drift caught by
        // `dc_cutlass_gemm_add_name_covers_full_zoo` test below.
        match (self.tile_m, self.tile_n, self.stages) {
            (32, 64, 3) => "dc_cutlass_32x64_s3_add",
            (32, 64, 4) => "dc_cutlass_32x64_s4_add",
            (32, 128, 3) => "dc_cutlass_32x128_s3_add",
            (32, 128, 4) => "dc_cutlass_32x128_s4_add",
            (32, 256, 3) => "dc_cutlass_32x256_s3_add",
            (64, 64, 3) => "dc_cutlass_64x64_s3_add",
            (64, 64, 4) => "dc_cutlass_64x64_s4_add",
            (64, 128, 3) => "dc_cutlass_64x128_s3_add",
            (64, 128, 4) => "dc_cutlass_64x128_s4_add",
            (128, 64, 3) => "dc_cutlass_128x64_s3_add",
            (128, 64, 4) => "dc_cutlass_128x64_s4_add",
            (128, 128, 3) => "dc_cutlass_128x128_s3_add",
            (128, 128, 4) => "dc_cutlass_128x128_s4_add",
            (128, 256, 3) => "dc_cutlass_128x256_s3_add",
            (256, 64, 3) => "dc_cutlass_256x64_s3_add",
            (256, 64, 4) => "dc_cutlass_256x64_s4_add",
            _ => "dc_cutlass_unknown_add",
        }
    }
    fn target_compatible(&self, profile: &TargetProfile) -> bool {
        profile.prim_mega_compatible() && self.host().target_compatible(profile)
    }
    fn workload_constraint(&self) -> WorkloadConstraint {
        self.host().workload_constraint()
    }
    fn matches(&self, fuf: &Fuf, seed: TileId, profile: &TargetProfile) -> Option<MatchInfo> {
        self.host().matches(fuf, seed, profile)
    }
    fn cost_us(&self, m: &MatchInfo, ctx: &CostCtx) -> f64 {
        self.host().cost_us(m, ctx)
    }
    fn resources(&self, m: &MatchInfo) -> Resources {
        self.host().resources(m)
    }
    fn launch_kind(&self) -> LaunchKind {
        LaunchKind::DeviceCallable
    }
    fn megakernel_fit(&self) -> MegakernelFit {
        MegakernelFit::Primitive
    }
    fn supported_input_handoffs(&self) -> &[Handoff] {
        DC_HANDOFFS
    }
    fn supported_output_handoffs(&self) -> &[Handoff] {
        DC_HANDOFFS
    }
    fn input_layouts(&self, m: &MatchInfo) -> Vec<Layout> {
        self.host().input_layouts(m)
    }
    fn output_layouts(&self, m: &MatchInfo) -> Vec<Layout> {
        self.host().output_layouts(m)
    }
    fn is_compute_bound(&self) -> bool {
        self.host().is_compute_bound()
    }
    fn can_share_kernel_with(&self, _other: &dyn Implementation) -> bool {
        true
    }
    fn output_alias(
        &self,
        claimed_tiles: &[TileId],
        fuf: &Fuf,
    ) -> Vec<((TileId, u8), Option<(TileId, u8)>)> {
        self.host().output_alias(claimed_tiles, fuf)
    }
    fn consumes_input_tiles(&self, claimed_tiles: &[TileId], fuf: &Fuf) -> Vec<(TileId, u8)> {
        self.host().consumes_input_tiles(claimed_tiles, fuf)
    }
    fn required_weights(
        &self,
        claimed_tiles: &[TileId],
        fuf: &Fuf,
        program: &Program,
    ) -> Vec<WeightAccessor> {
        self.host().required_weights(claimed_tiles, fuf, program)
    }
    fn opcode_shape(&self) -> OpcodeShape {
        self.host().opcode_shape()
    }
    fn fan_out(
        &self,
        m: &MatchInfo,
        fuf: &Fuf,
        program: &Program,
        bounds: &BTreeMap<String, u64>,
        slots: &SlotMap,
    ) -> Option<Vec<OpInstance>> {
        self.host().fan_out(m, fuf, program, bounds, slots)
    }
}

// ── CutlassGemmSplitKImpl ────────────────────────────────────────
//
// Singleton Gemm impl backed by CUTLASS `GemmSplitKParallel` — the K
// dim is split across `split_k` CTAs and reduced in a second kernel.
// Closes the "tall-skinny large-K" shape class where the standard
// tile zoo loses to cuBLAS (e.g. Qwen2-0.5B down_proj at prefill:
// M≈4096, N=896, K=4864).
//
// Same `matches` predicates as `CutlassGemmImpl` (dense bf16 only,
// not a fusion partner). Workload constraint excludes M=1 — GEMV
// owns decode. Cost is a direct CSV lookup on
// `cutlass_WxH_sS_splitN`; rows missing for a shape fall back to
// `UNCALIBRATED_COST_US` so the DP ignores the variant there.
//
// Registered alongside `CutlassGemmImpl` in `starter_library`.

/// The pure-SplitK tile variants exported by
/// `vllm-cuda/csrc/cutlass_standalone_gemm.cu` and declared as
/// externs in `ferrite-kernels::cutlass`. `(tile_m, tile_n, stages,
/// split_k)`. Keep in sync with that FFI block.
const CUTLASS_SPLITK_ZOO: &[(u32, u32, u32, u32)] = &[
    (64, 64, 4, 2),
    (64, 64, 4, 4),
    (64, 64, 4, 8),
    (64, 128, 4, 2),
    (64, 128, 4, 4),
    (64, 128, 4, 8),
    (128, 64, 4, 2),
    (128, 64, 4, 4),
    (128, 64, 4, 8),
    (128, 128, 4, 2),
    (128, 128, 4, 4),
    (128, 128, 4, 8),
];

#[derive(Debug, Clone)]
pub struct CutlassGemmSplitKImpl {
    pub tile_m: u32,
    pub tile_n: u32,
    pub stages: u32,
    pub split_k: u32,
}

impl CutlassGemmSplitKImpl {
    fn csv_name(&self) -> &'static str {
        self.static_name()
    }

    fn static_name(&self) -> &'static str {
        match (self.tile_m, self.tile_n, self.stages, self.split_k) {
            (64, 64, 4, 2) => "cutlass_64x64_s4_split2",
            (64, 64, 4, 4) => "cutlass_64x64_s4_split4",
            (64, 64, 4, 8) => "cutlass_64x64_s4_split8",
            (64, 128, 4, 2) => "cutlass_64x128_s4_split2",
            (64, 128, 4, 4) => "cutlass_64x128_s4_split4",
            (64, 128, 4, 8) => "cutlass_64x128_s4_split8",
            (128, 64, 4, 2) => "cutlass_128x64_s4_split2",
            (128, 64, 4, 4) => "cutlass_128x64_s4_split4",
            (128, 64, 4, 8) => "cutlass_128x64_s4_split8",
            (128, 128, 4, 2) => "cutlass_128x128_s4_split2",
            (128, 128, 4, 4) => "cutlass_128x128_s4_split4",
            (128, 128, 4, 8) => "cutlass_128x128_s4_split8",
            _ => "cutlass_splitk_unknown",
        }
    }
}

impl Implementation for CutlassGemmSplitKImpl {
    fn name(&self) -> &'static str {
        self.static_name()
    }

    fn target_compatible(&self, profile: &TargetProfile) -> bool {
        profile.cost_table.has_kernel(self.csv_name())
    }

    fn workload_constraint(&self) -> WorkloadConstraint {
        WorkloadConstraint::NumTokensRange {
            min: 2,
            max: u32::MAX,
        }
    }

    fn matches(&self, fuf: &Fuf, seed: TileId, _profile: &TargetProfile) -> Option<MatchInfo> {
        let info = single_tile_match(fuf, seed, OpKind::Gemm)?;
        if !matches!(weight_storage_of(fuf.get(seed)), Some(StorageFormat::Dense)) {
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

    // ── Host-interpreter codegen ────────────────────────────────

    fn opcode_shape(&self) -> OpcodeShape {
        OpcodeShape::new(
            "CutlassGemmSplitK",
            vec![
                ("in_slot", syn::parse_quote!(u32)),
                ("out_slot", syn::parse_quote!(u32)),
                ("layer", syn::parse_quote!(u32)),
                (
                    "weight_fn",
                    syn::parse_quote!(
                        for<'a> fn(&'a Weights, u32) -> &'a ::ferrite_kernels::layers::LinearLayer
                    ),
                ),
                ("tile_m", syn::parse_quote!(u32)),
                ("tile_n", syn::parse_quote!(u32)),
                ("stages", syn::parse_quote!(u32)),
                ("split_k", syn::parse_quote!(u32)),
                ("n", syn::parse_quote!(u32)),
                ("k", syn::parse_quote!(u32)),
            ],
        )
    }

    fn fan_out(
        &self,
        m: &MatchInfo,
        fuf: &Fuf,
        program: &Program,
        bounds: &BTreeMap<String, u64>,
        slots: &SlotMap,
    ) -> Option<Vec<OpInstance>> {
        let tile = m.claimed_tiles[0];
        let node = fuf.get(tile);
        let (in_id, in_slot) = match node.inputs.first() {
            Some(FufInput::Tile { id, slot }) => (*id, *slot),
            other => panic!("CutlassGemmSplitK: input 0 must be a Tile (got {other:?})"),
        };
        let in_slot_idx = slots.of(in_id, in_slot);
        let out_slot_idx = slots.of(tile, 0);
        let accessors = self.required_weights(&m.claimed_tiles, fuf, program);
        let acc = accessors
            .first()
            .expect("CutlassGemmSplitK: required_weights returned empty");
        let (base, layer) = split_base_layer(&acc.name.to_string());
        let layer = layer.unwrap_or(0) as u32;
        let base_ident = syn::Ident::new(&base, proc_macro2::Span::call_site());
        let tile_m = self.tile_m;
        let tile_n = self.tile_n;
        let stages = self.stages;
        let split_k = self.split_k;
        let (n, k) = gemm_nk_from_fuf(fuf, node, bounds)
            .expect("CutlassGemmSplitK: weight (N, K) must resolve from FUF + bounds");
        Some(vec![OpInstance::new(
            syn::Ident::new("CutlassGemmSplitK", proc_macro2::Span::call_site()),
            vec![
                quote! { #in_slot_idx },
                quote! { #out_slot_idx },
                quote! { #layer },
                quote! { Weights::#base_ident },
                quote! { #tile_m },
                quote! { #tile_n },
                quote! { #stages },
                quote! { #split_k },
                quote! { #n },
                quote! { #k },
            ],
        )])
    }
}

// ── CutlassGemmAddImpl ───────────────────────────────────────────
//
// Two-tile fusion: `(Gemm, Add)` where the Add is a residual-stream
// tile×tile add consuming the Gemm's output. Emits `cutlass_gemm_add`
// with `beta=1.0` — the CUTLASS LinearCombination epilogue computes
// `A @ B^T + residual` in-place on the residual buffer, one launch,
// no intermediate `[M, N]` delta in GMEM.
//
// One Impl per `CUTLASS_TILE_ZOO` entry (16 variants). Shares CSV
// cost rows with the singleton `CutlassGemmImpl` — the beta=1.0
// epilogue's extra aux-read of `[M, N]` bf16 is absorbed in the
// compute-bound GEMM's latency. If future calibration shows material
// drift, add dedicated `cutlass_WxH_sS_add` rows and switch
// `csv_name`.
//
// Dense-bf16 only; quantized storage Gemms (Marlin, Bnb4, Fp8, etc.)
// don't route through the cutlass tile zoo so this Impl rejects them
// the same way `CutlassGemmImpl` does.

#[derive(Debug, Clone)]
pub struct CutlassGemmAddImpl {
    pub tile_m: u32,
    pub tile_n: u32,
    pub stages: u32,
}

impl CutlassGemmAddImpl {
    fn csv_name(&self) -> &'static str {
        // Dedicated `_add` CSV row measured at beta=1.0 — captures
        // the epilogue's extra `[M, N]` aux-read cost that the plain
        // tile row (beta=0.0) doesn't see. Necessary for fair DP
        // comparison against `(singleton gemm + fused_add_rms_norm)`
        // on residual-stream chains.
        self.impl_name()
    }

    fn impl_name(&self) -> &'static str {
        match (self.tile_m, self.tile_n, self.stages) {
            (16, 64, 3) => "cutlass_16x64_s3_add",
            (16, 64, 4) => "cutlass_16x64_s4_add",
            (16, 128, 3) => "cutlass_16x128_s3_add",
            (16, 128, 4) => "cutlass_16x128_s4_add",
            (32, 64, 3) => "cutlass_32x64_s3_add",
            (32, 64, 4) => "cutlass_32x64_s4_add",
            (32, 128, 3) => "cutlass_32x128_s3_add",
            (32, 128, 4) => "cutlass_32x128_s4_add",
            (32, 256, 3) => "cutlass_32x256_s3_add",
            (64, 64, 3) => "cutlass_64x64_s3_add",
            (64, 64, 4) => "cutlass_64x64_s4_add",
            (64, 128, 3) => "cutlass_64x128_s3_add",
            (64, 128, 4) => "cutlass_64x128_s4_add",
            (128, 64, 3) => "cutlass_128x64_s3_add",
            (128, 64, 4) => "cutlass_128x64_s4_add",
            (128, 128, 3) => "cutlass_128x128_s3_add",
            (128, 128, 4) => "cutlass_128x128_s4_add",
            (128, 256, 3) => "cutlass_128x256_s3_add",
            (256, 64, 3) => "cutlass_256x64_s3_add",
            (256, 64, 4) => "cutlass_256x64_s4_add",
            _ => "cutlass_unknown_add",
        }
    }
}

impl Implementation for CutlassGemmAddImpl {
    fn name(&self) -> &'static str {
        self.impl_name()
    }

    fn target_compatible(&self, profile: &TargetProfile) -> bool {
        profile.cost_table.has_kernel(self.csv_name())
    }

    fn workload_constraint(&self) -> WorkloadConstraint {
        // Match CutlassGemmImpl's constraint — M=1 is GEMV territory
        // and has no gemm+add variant today.
        WorkloadConstraint::NumTokensRange {
            min: 2,
            max: u32::MAX,
        }
    }

    fn matches(&self, fuf: &Fuf, seed: TileId, _profile: &TargetProfile) -> Option<MatchInfo> {
        // Seed is the Gemm. Must be dense bf16, not a fusion partner
        // for an upstream chain (Silu/Mul/RopeAppend/BiasAdd); those
        // are owned by their dedicated fused Impls.
        let node = fuf.get(seed);
        if node.op != OpKind::Gemm {
            return None;
        }
        if !matches!(weight_storage_of(node), Some(StorageFormat::Dense)) {
            return None;
        }
        // Find a downstream residual-stream Add: both inputs are
        // Tiles, one of them is `seed`, the Add isn't already owned
        // by a scalar-offset path.
        let add_node = fuf.nodes.iter().find(|n| {
            if n.op != OpKind::Add || n.inputs.len() != 2 {
                return false;
            }
            let all_tiles = n.inputs.iter().all(|i| matches!(i, FufInput::Tile { .. }));
            if !all_tiles {
                return false;
            }
            consumes_tile(n, seed)
        })?;
        let add_id = add_node.id;

        // Identify the residual-side input (the Tile that isn't
        // `seed`). Convention in this codebase: slot 0 is the
        // delta (Gemm output), slot 1 is the residual. We don't
        // enforce slot order here — `cutlass_gemm_add` commutes
        // over alpha*delta + beta*residual anyway.
        let residual_src = add_node.inputs.iter().find_map(|i| match i {
            FufInput::Tile { id, slot } if *id != seed => Some((*id, *slot)),
            _ => None,
        })?;

        let activation_src = first_tile_input(node)?;

        Some(MatchInfo {
            claimed_tiles: vec![seed, add_id],
            boundary_inputs: vec![activation_src.0, residual_src.0],
            boundary_outputs: vec![add_id],
        })
    }

    fn cost_us(&self, m: &MatchInfo, ctx: &CostCtx) -> f64 {
        let gemm_tile = m
            .claimed_tiles
            .iter()
            .copied()
            .find(|t| ctx.fuf.get(*t).op == OpKind::Gemm)
            .expect("cutlass gemm+add claim contains a Gemm");
        let Some((mm, nn, kk)) = gemm_mnk(ctx, ctx.fuf.get(gemm_tile)) else {
            return f64::INFINITY;
        };
        ctx.profile
            .cost_us_for(self.csv_name(), mm, nn, kk)
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

    #[allow(clippy::type_complexity)]
    fn output_alias(
        &self,
        claimed_tiles: &[TileId],
        fuf: &Fuf,
    ) -> Vec<((TileId, u8), Option<(TileId, u8)>)> {
        // Output = the Add's logical output, which is the mutated
        // residual buffer. Alias to the residual-side input so
        // downstream consumers read the same buffer without a copy.
        let add_id = claimed_tiles
            .iter()
            .copied()
            .find(|t| fuf.get(*t).op == OpKind::Add)
            .expect("cutlass gemm+add claim contains an Add");
        let gemm_id = claimed_tiles
            .iter()
            .copied()
            .find(|t| fuf.get(*t).op == OpKind::Gemm)
            .expect("cutlass gemm+add claim contains a Gemm");
        let residual_src = fuf.get(add_id).inputs.iter().find_map(|i| match i {
            FufInput::Tile { id, slot } if *id != gemm_id => Some((*id, *slot)),
            _ => None,
        });
        vec![((add_id, 0), residual_src)]
    }

    // ── Host-interpreter codegen ────────────────────────────────
    //
    // Variant `CutlassGemmAdd { in_slot, residual_slot, layer,
    // weight_fn, tile_m, tile_n, stages }`. No out_slot — the Add's
    // output is a View alias of the residual upstream (declared via
    // output_alias and populated by the alias prelude). Body runs
    // the cutlass_gemm_add kernel which mutates the residual buffer
    // in place.

    fn opcode_shape(&self) -> OpcodeShape {
        OpcodeShape::new(
            "CutlassGemmAdd",
            vec![
                ("in_slot", syn::parse_quote!(u32)),
                ("residual_slot", syn::parse_quote!(u32)),
                ("layer", syn::parse_quote!(u32)),
                (
                    "weight_fn",
                    syn::parse_quote!(
                        for<'a> fn(&'a Weights, u32) -> &'a ::ferrite_kernels::layers::LinearLayer
                    ),
                ),
                ("tile_m", syn::parse_quote!(u32)),
                ("tile_n", syn::parse_quote!(u32)),
                ("stages", syn::parse_quote!(u32)),
                ("n", syn::parse_quote!(u32)),
                ("k", syn::parse_quote!(u32)),
            ],
        )
    }

    fn fan_out(
        &self,
        m: &MatchInfo,
        fuf: &Fuf,
        program: &Program,
        bounds: &BTreeMap<String, u64>,
        slots: &SlotMap,
    ) -> Option<Vec<OpInstance>> {
        let gemm_id = *m
            .claimed_tiles
            .iter()
            .find(|t| fuf.get(**t).op == OpKind::Gemm)
            .expect("CutlassGemmAdd: claim contains a Gemm");
        let add_id = *m
            .claimed_tiles
            .iter()
            .find(|t| fuf.get(**t).op == OpKind::Add)
            .expect("CutlassGemmAdd: claim contains an Add");
        let gemm_node = fuf.get(gemm_id);
        let (in_id, in_slot) = match gemm_node.inputs.first() {
            Some(FufInput::Tile { id, slot }) => (*id, *slot),
            other => panic!("CutlassGemmAdd: gemm input 0 must be a Tile (got {other:?})"),
        };
        let in_slot_idx = slots.of(in_id, in_slot);
        let (residual_id, residual_in) = fuf
            .get(add_id)
            .inputs
            .iter()
            .find_map(|i| match i {
                FufInput::Tile { id, slot } if *id != gemm_id => Some((*id, *slot)),
                _ => None,
            })
            .expect("CutlassGemmAdd: Add has a non-gemm Tile input (residual)");
        let residual_idx = slots.of(residual_id, residual_in);
        let accessors = self.required_weights(&m.claimed_tiles, fuf, program);
        let acc = accessors
            .first()
            .expect("CutlassGemmAdd: required_weights returned empty");
        let (base, layer) = split_base_layer(&acc.name.to_string());
        let layer = layer.unwrap_or(0) as u32;
        let base_ident = syn::Ident::new(&base, proc_macro2::Span::call_site());
        let tile_m = self.tile_m;
        let tile_n = self.tile_n;
        let stages = self.stages;
        let (n, k) = gemm_nk_from_fuf(fuf, gemm_node, bounds)
            .expect("CutlassGemmAdd: gemm (N, K) must resolve from FUF + bounds");
        Some(vec![OpInstance::new(
            syn::Ident::new("CutlassGemmAdd", proc_macro2::Span::call_site()),
            vec![
                quote! { #in_slot_idx },
                quote! { #residual_idx },
                quote! { #layer },
                quote! { Weights::#base_ident },
                quote! { #tile_m },
                quote! { #tile_n },
                quote! { #stages },
                quote! { #n },
                quote! { #k },
            ],
        )])
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

    // ── Host-interpreter codegen ────────────────────────────────

    fn opcode_shape(&self) -> OpcodeShape {
        OpcodeShape::new(
            "CutlassGemv",
            vec![
                ("in_slot", syn::parse_quote!(u32)),
                ("out_slot", syn::parse_quote!(u32)),
                ("layer", syn::parse_quote!(u32)),
                (
                    "weight_fn",
                    syn::parse_quote!(
                        for<'a> fn(&'a Weights, u32) -> &'a ::ferrite_kernels::layers::LinearLayer
                    ),
                ),
                ("n", syn::parse_quote!(u32)),
                ("k", syn::parse_quote!(u32)),
            ],
        )
    }

    fn fan_out(
        &self,
        m: &MatchInfo,
        fuf: &Fuf,
        program: &Program,
        bounds: &BTreeMap<String, u64>,
        slots: &SlotMap,
    ) -> Option<Vec<OpInstance>> {
        let tile = m.claimed_tiles[0];
        let node = fuf.get(tile);
        let (in_id, in_slot) = match node.inputs.first() {
            Some(FufInput::Tile { id, slot }) => (*id, *slot),
            other => panic!("CutlassGemv: input 0 must be a Tile (got {other:?})"),
        };
        let in_slot_idx = slots.of(in_id, in_slot);
        let out_slot_idx = slots.of(tile, 0);
        let accessors = self.required_weights(&m.claimed_tiles, fuf, program);
        let acc = accessors
            .first()
            .expect("CutlassGemv: required_weights returned empty");
        let (base, layer) = split_base_layer(&acc.name.to_string());
        let layer = layer.unwrap_or(0) as u32;
        let base_ident = syn::Ident::new(&base, proc_macro2::Span::call_site());
        let (n, k) = gemm_nk_from_fuf(fuf, node, bounds)
            .expect("CutlassGemv: weight (N, K) must resolve from FUF + bounds");
        Some(vec![OpInstance::new(
            syn::Ident::new("CutlassGemv", proc_macro2::Span::call_site()),
            vec![
                quote! { #in_slot_idx },
                quote! { #out_slot_idx },
                quote! { #layer },
                quote! { Weights::#base_ident },
                quote! { #n },
                quote! { #k },
            ],
        )])
    }
}

// ── DcCutlassGemvImpl ────────────────────────────────────────────
//
// PrimMega sibling to [`CutlassGemvImpl`]. Singleton (M=1 only).
// Delegates everything to the host counterpart: cost reads the same
// `cutlass_gemv` CSV row, matches/fan_out structurally identical.
// Override only launch_kind (DeviceCallable), megakernel_fit
// (Primitive), target_compatible (adds prim_mega_compatible >= 90),
// and the in/out handoff lists (DC_HANDOFFS). The C++ side is
// `dc_cutlass::dc_gemv<DcGemvKernel_bf16_8>` in
// `vllm-cuda/csrc/megakernel/prim_mega.cu`, parameterized on the
// same kernel typedef the host's `cutlass_gemv_launch` uses — so
// the megakernel-runtime cost matches the calibrated CSV row.

#[derive(Debug, Default)]
pub struct DcCutlassGemvImpl;

impl Implementation for DcCutlassGemvImpl {
    fn name(&self) -> &'static str {
        "dc_cutlass_gemv"
    }
    fn target_compatible(&self, profile: &TargetProfile) -> bool {
        profile.prim_mega_compatible() && CutlassGemvImpl.target_compatible(profile)
    }
    fn workload_constraint(&self) -> WorkloadConstraint {
        CutlassGemvImpl.workload_constraint()
    }
    fn matches(&self, fuf: &Fuf, seed: TileId, profile: &TargetProfile) -> Option<MatchInfo> {
        CutlassGemvImpl.matches(fuf, seed, profile)
    }
    fn cost_us(&self, m: &MatchInfo, ctx: &CostCtx) -> f64 {
        CutlassGemvImpl.cost_us(m, ctx)
    }
    fn resources(&self, m: &MatchInfo) -> Resources {
        CutlassGemvImpl.resources(m)
    }
    fn launch_kind(&self) -> LaunchKind {
        LaunchKind::DeviceCallable
    }
    fn megakernel_fit(&self) -> MegakernelFit {
        MegakernelFit::Primitive
    }
    fn supported_input_handoffs(&self) -> &[Handoff] {
        DC_HANDOFFS
    }
    fn supported_output_handoffs(&self) -> &[Handoff] {
        DC_HANDOFFS
    }
    fn input_layouts(&self, m: &MatchInfo) -> Vec<Layout> {
        CutlassGemvImpl.input_layouts(m)
    }
    fn output_layouts(&self, m: &MatchInfo) -> Vec<Layout> {
        CutlassGemvImpl.output_layouts(m)
    }
    fn is_compute_bound(&self) -> bool {
        CutlassGemvImpl.is_compute_bound()
    }
    fn can_share_kernel_with(&self, _other: &dyn Implementation) -> bool {
        true
    }
    fn output_alias(
        &self,
        claimed_tiles: &[TileId],
        fuf: &Fuf,
    ) -> Vec<((TileId, u8), Option<(TileId, u8)>)> {
        CutlassGemvImpl.output_alias(claimed_tiles, fuf)
    }
    fn consumes_input_tiles(&self, claimed_tiles: &[TileId], fuf: &Fuf) -> Vec<(TileId, u8)> {
        CutlassGemvImpl.consumes_input_tiles(claimed_tiles, fuf)
    }
    fn required_weights(
        &self,
        claimed_tiles: &[TileId],
        fuf: &Fuf,
        program: &Program,
    ) -> Vec<WeightAccessor> {
        CutlassGemvImpl.required_weights(claimed_tiles, fuf, program)
    }
    fn opcode_shape(&self) -> OpcodeShape {
        CutlassGemvImpl.opcode_shape()
    }
    fn fan_out(
        &self,
        m: &MatchInfo,
        fuf: &Fuf,
        program: &Program,
        bounds: &BTreeMap<String, u64>,
        slots: &SlotMap,
    ) -> Option<Vec<OpInstance>> {
        CutlassGemvImpl.fan_out(m, fuf, program, bounds, slots)
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

/// Accessor Rust type for an FP8 GEMM. Always `Fp8AnyLinear` — the
/// loader picks `Std` vs `Block` per claim based on the weight's
/// on-disk storage, but the per-arch Weights field type is uniform so
/// the host-interpreter `weight_fn` can have one concrete signature.
fn fp8_accessor_type_for(_fuf: &Fuf, _gemm_tile: TileId) -> TokenStream {
    quote! { ::ferrite_kernels::layers::Fp8AnyLinear }
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

    // ── Host-interpreter codegen ────────────────────────────────

    fn opcode_shape(&self) -> OpcodeShape {
        OpcodeShape::new(
            "MarlinGemm",
            vec![
                ("in_slot", syn::parse_quote!(u32)),
                ("out_slot", syn::parse_quote!(u32)),
                ("layer", syn::parse_quote!(u32)),
                (
                    "weight_fn",
                    syn::parse_quote!(
                        for<'a> fn(&'a Weights, u32) -> &'a ::ferrite_kernels::layers::MarlinLinear
                    ),
                ),
            ],
        )
    }

    fn fan_out(
        &self,
        m: &MatchInfo,
        fuf: &Fuf,
        program: &Program,
        _bounds: &BTreeMap<String, u64>,
        slots: &SlotMap,
    ) -> Option<Vec<OpInstance>> {
        let tile = m.claimed_tiles[0];
        let node = fuf.get(tile);
        let (in_id, in_slot) = match node.inputs.first() {
            Some(FufInput::Tile { id, slot }) => (*id, *slot),
            other => panic!("MarlinGemm: input 0 must be a Tile (got {other:?})"),
        };
        let in_slot_idx = slots.of(in_id, in_slot);
        let out_slot_idx = slots.of(tile, 0);
        let accessors = self.required_weights(&m.claimed_tiles, fuf, program);
        let acc = accessors
            .first()
            .expect("MarlinGemm: required_weights returned empty");
        let (base, layer) = split_base_layer(&acc.name.to_string());
        let layer = layer.unwrap_or(0) as u32;
        let base_ident = syn::Ident::new(&base, proc_macro2::Span::call_site());
        Some(vec![OpInstance::new(
            syn::Ident::new("MarlinGemm", proc_macro2::Span::call_site()),
            vec![
                quote! { #in_slot_idx },
                quote! { #out_slot_idx },
                quote! { #layer },
                quote! { Weights::#base_ident },
            ],
        )])
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

    // ── Host-interpreter codegen ────────────────────────────────

    fn opcode_shape(&self) -> OpcodeShape {
        OpcodeShape::new(
            "MarlinFusedGateUpSiluMul",
            vec![
                ("in_slot", syn::parse_quote!(u32)),
                ("out_slot", syn::parse_quote!(u32)),
                ("layer", syn::parse_quote!(u32)),
                (
                    "weight_fn",
                    syn::parse_quote!(
                        for<'a> fn(&'a Weights, u32) -> &'a ::ferrite_kernels::layers::MarlinLinear
                    ),
                ),
            ],
        )
    }

    fn fan_out(
        &self,
        m: &MatchInfo,
        fuf: &Fuf,
        program: &Program,
        _bounds: &BTreeMap<String, u64>,
        slots: &SlotMap,
    ) -> Option<Vec<OpInstance>> {
        let silu_id = *m
            .claimed_tiles
            .iter()
            .find(|t| fuf.get(**t).op == OpKind::Silu)
            .expect("MarlinFusedGateUpSiluMul: claim contains Silu");
        let mul_id = *m
            .claimed_tiles
            .iter()
            .find(|t| fuf.get(**t).op == OpKind::Mul)
            .expect("MarlinFusedGateUpSiluMul: claim contains Mul");
        let gate_id = match fuf.get(silu_id).inputs.first() {
            Some(FufInput::Tile { id, .. }) => *id,
            other => {
                panic!("MarlinFusedGateUpSiluMul: Silu input 0 must be a Tile (got {other:?})")
            }
        };
        let gate_node = fuf.get(gate_id);
        let (in_id, in_slot) = match gate_node.inputs.first() {
            Some(FufInput::Tile { id, slot }) => (*id, *slot),
            other => {
                panic!("MarlinFusedGateUpSiluMul: gate gemm input 0 must be a Tile (got {other:?})")
            }
        };
        let in_slot_idx = slots.of(in_id, in_slot);
        let out_slot_idx = slots.of(mul_id, 0);
        let accessors = self.required_weights(&m.claimed_tiles, fuf, program);
        let acc = accessors
            .first()
            .expect("MarlinFusedGateUpSiluMul: required_weights returned empty");
        let (base, layer) = split_base_layer(&acc.name.to_string());
        let layer = layer.unwrap_or(0) as u32;
        let base_ident = syn::Ident::new(&base, proc_macro2::Span::call_site());
        Some(vec![OpInstance::new(
            syn::Ident::new("MarlinFusedGateUpSiluMul", proc_macro2::Span::call_site()),
            vec![
                quote! { #in_slot_idx },
                quote! { #out_slot_idx },
                quote! { #layer },
                quote! { Weights::#base_ident },
            ],
        )])
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

    // ── Host-interpreter codegen ────────────────────────────────

    fn opcode_shape(&self) -> OpcodeShape {
        OpcodeShape::new(
            "MarlinFusedGateUpGeluMul",
            vec![
                ("in_slot", syn::parse_quote!(u32)),
                ("out_slot", syn::parse_quote!(u32)),
                ("layer", syn::parse_quote!(u32)),
                (
                    "weight_fn",
                    syn::parse_quote!(
                        for<'a> fn(&'a Weights, u32) -> &'a ::ferrite_kernels::layers::MarlinLinear
                    ),
                ),
            ],
        )
    }

    fn fan_out(
        &self,
        m: &MatchInfo,
        fuf: &Fuf,
        program: &Program,
        _bounds: &BTreeMap<String, u64>,
        slots: &SlotMap,
    ) -> Option<Vec<OpInstance>> {
        let gelu_id = *m
            .claimed_tiles
            .iter()
            .find(|t| fuf.get(**t).op == OpKind::Gelu)
            .expect("MarlinFusedGateUpGeluMul: claim contains Gelu");
        let mul_id = *m
            .claimed_tiles
            .iter()
            .find(|t| fuf.get(**t).op == OpKind::Mul)
            .expect("MarlinFusedGateUpGeluMul: claim contains Mul");
        let gate_id = match fuf.get(gelu_id).inputs.first() {
            Some(FufInput::Tile { id, .. }) => *id,
            other => {
                panic!("MarlinFusedGateUpGeluMul: Gelu input 0 must be a Tile (got {other:?})")
            }
        };
        let gate_node = fuf.get(gate_id);
        let (in_id, in_slot) = match gate_node.inputs.first() {
            Some(FufInput::Tile { id, slot }) => (*id, *slot),
            other => {
                panic!("MarlinFusedGateUpGeluMul: gate gemm input 0 must be a Tile (got {other:?})")
            }
        };
        let in_slot_idx = slots.of(in_id, in_slot);
        let out_slot_idx = slots.of(mul_id, 0);
        let accessors = self.required_weights(&m.claimed_tiles, fuf, program);
        let acc = accessors
            .first()
            .expect("MarlinFusedGateUpGeluMul: required_weights returned empty");
        let (base, layer) = split_base_layer(&acc.name.to_string());
        let layer = layer.unwrap_or(0) as u32;
        let base_ident = syn::Ident::new(&base, proc_macro2::Span::call_site());
        Some(vec![OpInstance::new(
            syn::Ident::new("MarlinFusedGateUpGeluMul", proc_macro2::Span::call_site()),
            vec![
                quote! { #in_slot_idx },
                quote! { #out_slot_idx },
                quote! { #layer },
                quote! { Weights::#base_ident },
            ],
        )])
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

    // ── Host-interpreter codegen ────────────────────────────────
    //
    // Variant `MarlinFusedQkvRopeCache { in_slot, out_slot, layer,
    // weight_fn(MarlinLinear), cos_sin_fn }`. No interleaved field
    // (Marlin gates AWQ/GPTQ; Cohere is dense-only). FP8 KV branches
    // at runtime via `ctx.kv_cache.is_fp8()`.

    fn opcode_shape(&self) -> OpcodeShape {
        OpcodeShape::new(
            "MarlinFusedQkvRopeCache",
            vec![
                ("in_slot", syn::parse_quote!(u32)),
                ("out_slot", syn::parse_quote!(u32)),
                ("layer", syn::parse_quote!(u32)),
                (
                    "weight_fn",
                    syn::parse_quote!(
                        for<'a> fn(&'a Weights, u32) -> &'a ::ferrite_kernels::layers::MarlinLinear
                    ),
                ),
                (
                    "cos_sin_fn",
                    syn::parse_quote!(
                        for<'a> fn(&'a Weights, u32) -> ::ferrite_cuda_core::tensor::GpuTensor
                    ),
                ),
            ],
        )
    }

    fn fan_out(
        &self,
        m: &MatchInfo,
        fuf: &Fuf,
        program: &Program,
        _bounds: &BTreeMap<String, u64>,
        slots: &SlotMap,
    ) -> Option<Vec<OpInstance>> {
        let rope_id = *m
            .claimed_tiles
            .iter()
            .find(|t| fuf.get(**t).op == OpKind::RopeAppend)
            .expect("MarlinFusedQkvRopeCache: claim contains a RopeAppend");
        let rope_node = fuf.get(rope_id);
        let layer = rope_kv_cache_layer(rope_node)
            .expect("MarlinFusedQkvRopeCache: RopeAppend has KvCache extern with layer index")
            as u32;
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
            .map(|t| {
                unwrap_gemm_through_bias(fuf, *t)
                    .expect("MarlinFusedQkvRopeCache: claim guarantees Gemm-or-BiasAdd(Gemm)")
            })
            .collect();
        let q_gemm_id = resolved[0].0;
        let q_node = fuf.get(q_gemm_id);
        let (in_id, in_slot) = match q_node.inputs.first() {
            Some(FufInput::Tile { id, slot }) => (*id, *slot),
            other => {
                panic!("MarlinFusedQkvRopeCache: q_gemm input 0 must be a Tile (got {other:?})")
            }
        };
        let in_slot_idx = slots.of(in_id, in_slot);
        let out_slot_idx = slots.of(rope_id, 0);
        let accessors = self.required_weights(&m.claimed_tiles, fuf, program);
        let acc = accessors
            .first()
            .expect("MarlinFusedQkvRopeCache: required_weights returned empty");
        let (base, weight_layer) = split_base_layer(&acc.name.to_string());
        let weight_layer = weight_layer.unwrap_or(layer as u64) as u32;
        assert_eq!(
            weight_layer, layer,
            "MarlinFusedQkvRopeCache: weight_layer disagrees with kv_cache layer"
        );
        let base_ident = syn::Ident::new(&base, proc_macro2::Span::call_site());
        let uses_local = m.claimed_tiles.iter().any(|&tid| {
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
        let cos_sin_ident = if uses_local {
            syn::Ident::new("rotary_local_cos_sin", proc_macro2::Span::call_site())
        } else {
            syn::Ident::new("rotary_cos_sin", proc_macro2::Span::call_site())
        };
        Some(vec![OpInstance::new(
            syn::Ident::new("MarlinFusedQkvRopeCache", proc_macro2::Span::call_site()),
            vec![
                quote! { #in_slot_idx },
                quote! { #out_slot_idx },
                quote! { #layer },
                quote! { Weights::#base_ident },
                quote! { Weights::#cos_sin_ident },
            ],
        )])
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

    // ── Host-interpreter codegen ────────────────────────────────

    fn opcode_shape(&self) -> OpcodeShape {
        OpcodeShape::new(
            "MarlinFusedQkvRopePrefill",
            vec![
                ("in_slot", syn::parse_quote!(u32)),
                ("q_out_slot", syn::parse_quote!(u32)),
                ("k_out_slot", syn::parse_quote!(u32)),
                ("v_out_slot", syn::parse_quote!(u32)),
                ("layer", syn::parse_quote!(u32)),
                (
                    "weight_fn",
                    syn::parse_quote!(
                        for<'a> fn(&'a Weights, u32) -> &'a ::ferrite_kernels::layers::MarlinLinear
                    ),
                ),
                (
                    "cos_sin_fn",
                    syn::parse_quote!(
                        for<'a> fn(&'a Weights, u32) -> ::ferrite_cuda_core::tensor::GpuTensor
                    ),
                ),
            ],
        )
    }

    fn fan_out(
        &self,
        m: &MatchInfo,
        fuf: &Fuf,
        program: &Program,
        _bounds: &BTreeMap<String, u64>,
        slots: &SlotMap,
    ) -> Option<Vec<OpInstance>> {
        let rope_id = *m
            .claimed_tiles
            .iter()
            .find(|t| fuf.get(**t).op == OpKind::RopeAppend)
            .expect("MarlinFusedQkvRopePrefill: claim contains a RopeAppend");
        let rope_node = fuf.get(rope_id);
        let layer = rope_kv_cache_layer(rope_node)
            .expect("MarlinFusedQkvRopePrefill: RopeAppend has KvCache extern with layer index")
            as u32;
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
            .map(|t| {
                unwrap_gemm_through_bias(fuf, *t)
                    .expect("MarlinFusedQkvRopePrefill: Gemm-or-BiasAdd(Gemm)")
            })
            .collect();
        let q_gemm_id = resolved[0].0;
        let q_node = fuf.get(q_gemm_id);
        let (in_id, in_slot) = match q_node.inputs.first() {
            Some(FufInput::Tile { id, slot }) => (*id, *slot),
            other => {
                panic!("MarlinFusedQkvRopePrefill: q_gemm input 0 must be a Tile (got {other:?})")
            }
        };
        let in_slot_idx = slots.of(in_id, in_slot);
        let q_out = slots.of(rope_id, 0);
        let k_out = slots.of(rope_id, 1);
        let v_out = slots.of(rope_id, 2);
        let accessors = self.required_weights(&m.claimed_tiles, fuf, program);
        let acc = accessors
            .first()
            .expect("MarlinFusedQkvRopePrefill: required_weights returned empty");
        let (base, weight_layer) = split_base_layer(&acc.name.to_string());
        let weight_layer = weight_layer.unwrap_or(layer as u64) as u32;
        assert_eq!(
            weight_layer, layer,
            "MarlinFusedQkvRopePrefill: weight_layer disagrees with kv_cache layer"
        );
        let base_ident = syn::Ident::new(&base, proc_macro2::Span::call_site());
        let uses_local = m.claimed_tiles.iter().any(|&tid| {
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
        let cos_sin_ident = if uses_local {
            syn::Ident::new("rotary_local_cos_sin", proc_macro2::Span::call_site())
        } else {
            syn::Ident::new("rotary_cos_sin", proc_macro2::Span::call_site())
        };
        Some(vec![OpInstance::new(
            syn::Ident::new("MarlinFusedQkvRopePrefill", proc_macro2::Span::call_site()),
            vec![
                quote! { #in_slot_idx },
                quote! { #q_out },
                quote! { #k_out },
                quote! { #v_out },
                quote! { #layer },
                quote! { Weights::#base_ident },
                quote! { Weights::#cos_sin_ident },
            ],
        )])
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

    // ── Host-interpreter codegen ────────────────────────────────

    fn opcode_shape(&self) -> OpcodeShape {
        OpcodeShape::new(
            "Bnb4Gemm",
            vec![
                ("in_slot", syn::parse_quote!(u32)),
                ("out_slot", syn::parse_quote!(u32)),
                ("layer", syn::parse_quote!(u32)),
                (
                    "weight_fn",
                    syn::parse_quote!(
                        for<'a> fn(
                            &'a Weights,
                            u32,
                        )
                            -> &'a ::ferrite_kernels::layers::Bnb4bitLinear
                    ),
                ),
            ],
        )
    }

    fn fan_out(
        &self,
        m: &MatchInfo,
        fuf: &Fuf,
        program: &Program,
        _bounds: &BTreeMap<String, u64>,
        slots: &SlotMap,
    ) -> Option<Vec<OpInstance>> {
        let tile = m.claimed_tiles[0];
        let node = fuf.get(tile);
        let (in_id, in_slot) = match node.inputs.first() {
            Some(FufInput::Tile { id, slot }) => (*id, *slot),
            other => panic!("Bnb4Gemm: input 0 must be a Tile (got {other:?})"),
        };
        let in_slot_idx = slots.of(in_id, in_slot);
        let out_slot_idx = slots.of(tile, 0);
        let accessors = self.required_weights(&m.claimed_tiles, fuf, program);
        let acc = accessors
            .first()
            .expect("Bnb4Gemm: required_weights returned empty");
        let (base, layer) = split_base_layer(&acc.name.to_string());
        let layer = layer.unwrap_or(0) as u32;
        let base_ident = syn::Ident::new(&base, proc_macro2::Span::call_site());
        Some(vec![OpInstance::new(
            syn::Ident::new("Bnb4Gemm", proc_macro2::Span::call_site()),
            vec![
                quote! { #in_slot_idx },
                quote! { #out_slot_idx },
                quote! { #layer },
                quote! { Weights::#base_ident },
            ],
        )])
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

    // ── Host-interpreter codegen ────────────────────────────────

    fn opcode_shape(&self) -> OpcodeShape {
        OpcodeShape::new(
            "Fp8Gemm",
            vec![
                ("in_slot", syn::parse_quote!(u32)),
                ("out_slot", syn::parse_quote!(u32)),
                ("layer", syn::parse_quote!(u32)),
                (
                    "weight_fn",
                    syn::parse_quote!(
                        for<'a> fn(&'a Weights, u32) -> &'a ::ferrite_kernels::layers::Fp8AnyLinear
                    ),
                ),
            ],
        )
    }

    fn fan_out(
        &self,
        m: &MatchInfo,
        fuf: &Fuf,
        program: &Program,
        _bounds: &BTreeMap<String, u64>,
        slots: &SlotMap,
    ) -> Option<Vec<OpInstance>> {
        let tile = m.claimed_tiles[0];
        let node = fuf.get(tile);
        let (in_id, in_slot) = match node.inputs.first() {
            Some(FufInput::Tile { id, slot }) => (*id, *slot),
            other => panic!("Fp8Gemm: input 0 must be a Tile (got {other:?})"),
        };
        let in_slot_idx = slots.of(in_id, in_slot);
        let out_slot_idx = slots.of(tile, 0);
        let accessors = self.required_weights(&m.claimed_tiles, fuf, program);
        let acc = accessors
            .first()
            .expect("Fp8Gemm: required_weights returned empty");
        let (base, layer) = split_base_layer(&acc.name.to_string());
        let layer = layer.unwrap_or(0) as u32;
        let base_ident = syn::Ident::new(&base, proc_macro2::Span::call_site());
        Some(vec![OpInstance::new(
            syn::Ident::new("Fp8Gemm", proc_macro2::Span::call_site()),
            vec![
                quote! { #in_slot_idx },
                quote! { #out_slot_idx },
                quote! { #layer },
                quote! { Weights::#base_ident },
            ],
        )])
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

    // ── Host-interpreter codegen ────────────────────────────────

    fn opcode_shape(&self) -> OpcodeShape {
        OpcodeShape::new(
            "Fp8GemmBias",
            vec![
                ("in_slot", syn::parse_quote!(u32)),
                ("out_slot", syn::parse_quote!(u32)),
                ("layer", syn::parse_quote!(u32)),
                (
                    "weight_fn",
                    syn::parse_quote!(
                        for<'a> fn(&'a Weights, u32) -> &'a ::ferrite_kernels::layers::Fp8AnyLinear
                    ),
                ),
            ],
        )
    }

    fn fan_out(
        &self,
        m: &MatchInfo,
        fuf: &Fuf,
        program: &Program,
        _bounds: &BTreeMap<String, u64>,
        slots: &SlotMap,
    ) -> Option<Vec<OpInstance>> {
        let gemm_id = *m
            .claimed_tiles
            .iter()
            .find(|t| fuf.get(**t).op == OpKind::Gemm)
            .expect("Fp8FusedGemmBias: claim contains Gemm");
        let bias_id = *m
            .claimed_tiles
            .iter()
            .find(|t| fuf.get(**t).op == OpKind::BiasAdd)
            .expect("Fp8FusedGemmBias: claim contains BiasAdd");
        let gemm_node = fuf.get(gemm_id);
        let (in_id, in_slot) = match gemm_node.inputs.first() {
            Some(FufInput::Tile { id, slot }) => (*id, *slot),
            other => panic!("Fp8FusedGemmBias: gemm's first input must be a Tile (got {other:?})"),
        };
        let in_slot_idx = slots.of(in_id, in_slot);
        let out_slot_idx = slots.of(bias_id, 0);
        let accessors = self.required_weights(&m.claimed_tiles, fuf, program);
        let acc = accessors
            .first()
            .expect("Fp8FusedGemmBias: required_weights returned empty");
        let (base, layer) = split_base_layer(&acc.name.to_string());
        let layer = layer.unwrap_or(0) as u32;
        let base_ident = syn::Ident::new(&base, proc_macro2::Span::call_site());
        Some(vec![OpInstance::new(
            syn::Ident::new("Fp8GemmBias", proc_macro2::Span::call_site()),
            vec![
                quote! { #in_slot_idx },
                quote! { #out_slot_idx },
                quote! { #layer },
                quote! { Weights::#base_ident },
            ],
        )])
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

    // ── Host-interpreter codegen ────────────────────────────────

    fn opcode_shape(&self) -> OpcodeShape {
        OpcodeShape::new(
            "Fp8FusedGateUpSiluMul",
            vec![
                ("in_slot", syn::parse_quote!(u32)),
                ("out_slot", syn::parse_quote!(u32)),
                ("layer", syn::parse_quote!(u32)),
                (
                    "weight_fn",
                    syn::parse_quote!(
                        for<'a> fn(&'a Weights, u32) -> &'a ::ferrite_kernels::layers::Fp8AnyLinear
                    ),
                ),
            ],
        )
    }

    fn fan_out(
        &self,
        m: &MatchInfo,
        fuf: &Fuf,
        program: &Program,
        _bounds: &BTreeMap<String, u64>,
        slots: &SlotMap,
    ) -> Option<Vec<OpInstance>> {
        let silu_id = *m
            .claimed_tiles
            .iter()
            .find(|t| fuf.get(**t).op == OpKind::Silu)
            .expect("Fp8FusedGateUpSiluMul: claim contains Silu");
        let mul_id = *m
            .claimed_tiles
            .iter()
            .find(|t| fuf.get(**t).op == OpKind::Mul)
            .expect("Fp8FusedGateUpSiluMul: claim contains Mul");
        let gate_id = match fuf.get(silu_id).inputs.first() {
            Some(FufInput::Tile { id, .. }) => *id,
            other => panic!("Fp8FusedGateUpSiluMul: Silu input 0 must be a Tile (got {other:?})"),
        };
        let gate_node = fuf.get(gate_id);
        let (in_id, in_slot) = match gate_node.inputs.first() {
            Some(FufInput::Tile { id, slot }) => (*id, *slot),
            other => {
                panic!("Fp8FusedGateUpSiluMul: gate gemm input 0 must be a Tile (got {other:?})")
            }
        };
        let in_slot_idx = slots.of(in_id, in_slot);
        let out_slot_idx = slots.of(mul_id, 0);
        let accessors = self.required_weights(&m.claimed_tiles, fuf, program);
        let acc = accessors
            .first()
            .expect("Fp8FusedGateUpSiluMul: required_weights returned empty");
        let (base, layer) = split_base_layer(&acc.name.to_string());
        let layer = layer.unwrap_or(0) as u32;
        let base_ident = syn::Ident::new(&base, proc_macro2::Span::call_site());
        Some(vec![OpInstance::new(
            syn::Ident::new("Fp8FusedGateUpSiluMul", proc_macro2::Span::call_site()),
            vec![
                quote! { #in_slot_idx },
                quote! { #out_slot_idx },
                quote! { #layer },
                quote! { Weights::#base_ident },
            ],
        )])
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

    // ── Host-interpreter codegen ────────────────────────────────

    fn opcode_shape(&self) -> OpcodeShape {
        OpcodeShape::new(
            "Bnb4FusedGateUpSiluMul",
            vec![
                ("in_slot", syn::parse_quote!(u32)),
                ("out_slot", syn::parse_quote!(u32)),
                ("layer", syn::parse_quote!(u32)),
                (
                    "weight_fn",
                    syn::parse_quote!(
                        for<'a> fn(
                            &'a Weights,
                            u32,
                        )
                            -> &'a ::ferrite_kernels::layers::Bnb4bitLinear
                    ),
                ),
            ],
        )
    }

    fn fan_out(
        &self,
        m: &MatchInfo,
        fuf: &Fuf,
        program: &Program,
        _bounds: &BTreeMap<String, u64>,
        slots: &SlotMap,
    ) -> Option<Vec<OpInstance>> {
        let silu_id = *m
            .claimed_tiles
            .iter()
            .find(|t| fuf.get(**t).op == OpKind::Silu)
            .expect("Bnb4FusedGateUpSiluMul: claim contains Silu");
        let mul_id = *m
            .claimed_tiles
            .iter()
            .find(|t| fuf.get(**t).op == OpKind::Mul)
            .expect("Bnb4FusedGateUpSiluMul: claim contains Mul");
        let gate_id = match fuf.get(silu_id).inputs.first() {
            Some(FufInput::Tile { id, .. }) => *id,
            other => panic!("Bnb4FusedGateUpSiluMul: Silu input 0 must be a Tile (got {other:?})"),
        };
        let gate_node = fuf.get(gate_id);
        let (in_id, in_slot) = match gate_node.inputs.first() {
            Some(FufInput::Tile { id, slot }) => (*id, *slot),
            other => {
                panic!("Bnb4FusedGateUpSiluMul: gate gemm input 0 must be a Tile (got {other:?})")
            }
        };
        let in_slot_idx = slots.of(in_id, in_slot);
        let out_slot_idx = slots.of(mul_id, 0);
        let accessors = self.required_weights(&m.claimed_tiles, fuf, program);
        let acc = accessors
            .first()
            .expect("Bnb4FusedGateUpSiluMul: required_weights returned empty");
        let (base, layer) = split_base_layer(&acc.name.to_string());
        let layer = layer.unwrap_or(0) as u32;
        let base_ident = syn::Ident::new(&base, proc_macro2::Span::call_site());
        Some(vec![OpInstance::new(
            syn::Ident::new("Bnb4FusedGateUpSiluMul", proc_macro2::Span::call_site()),
            vec![
                quote! { #in_slot_idx },
                quote! { #out_slot_idx },
                quote! { #layer },
                quote! { Weights::#base_ident },
            ],
        )])
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

    // ── Host-interpreter codegen ────────────────────────────────

    fn opcode_shape(&self) -> OpcodeShape {
        OpcodeShape::new(
            "Bnb4FusedGateUpGeluMul",
            vec![
                ("in_slot", syn::parse_quote!(u32)),
                ("out_slot", syn::parse_quote!(u32)),
                ("layer", syn::parse_quote!(u32)),
                (
                    "weight_fn",
                    syn::parse_quote!(
                        for<'a> fn(
                            &'a Weights,
                            u32,
                        )
                            -> &'a ::ferrite_kernels::layers::Bnb4bitLinear
                    ),
                ),
            ],
        )
    }

    fn fan_out(
        &self,
        m: &MatchInfo,
        fuf: &Fuf,
        program: &Program,
        _bounds: &BTreeMap<String, u64>,
        slots: &SlotMap,
    ) -> Option<Vec<OpInstance>> {
        let gelu_id = *m
            .claimed_tiles
            .iter()
            .find(|t| fuf.get(**t).op == OpKind::Gelu)
            .expect("Bnb4FusedGateUpGeluMul: claim contains Gelu");
        let mul_id = *m
            .claimed_tiles
            .iter()
            .find(|t| fuf.get(**t).op == OpKind::Mul)
            .expect("Bnb4FusedGateUpGeluMul: claim contains Mul");
        let gate_id = match fuf.get(gelu_id).inputs.first() {
            Some(FufInput::Tile { id, .. }) => *id,
            other => panic!("Bnb4FusedGateUpGeluMul: Gelu input 0 must be a Tile (got {other:?})"),
        };
        let gate_node = fuf.get(gate_id);
        let (in_id, in_slot) = match gate_node.inputs.first() {
            Some(FufInput::Tile { id, slot }) => (*id, *slot),
            other => {
                panic!("Bnb4FusedGateUpGeluMul: gate gemm input 0 must be a Tile (got {other:?})")
            }
        };
        let in_slot_idx = slots.of(in_id, in_slot);
        let out_slot_idx = slots.of(mul_id, 0);
        let accessors = self.required_weights(&m.claimed_tiles, fuf, program);
        let acc = accessors
            .first()
            .expect("Bnb4FusedGateUpGeluMul: required_weights returned empty");
        let (base, layer) = split_base_layer(&acc.name.to_string());
        let layer = layer.unwrap_or(0) as u32;
        let base_ident = syn::Ident::new(&base, proc_macro2::Span::call_site());
        Some(vec![OpInstance::new(
            syn::Ident::new("Bnb4FusedGateUpGeluMul", proc_macro2::Span::call_site()),
            vec![
                quote! { #in_slot_idx },
                quote! { #out_slot_idx },
                quote! { #layer },
                quote! { Weights::#base_ident },
            ],
        )])
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

    // ── Host-interpreter codegen ────────────────────────────────

    fn opcode_shape(&self) -> OpcodeShape {
        OpcodeShape::new(
            "Fp8FusedGateUpGeluMul",
            vec![
                ("in_slot", syn::parse_quote!(u32)),
                ("out_slot", syn::parse_quote!(u32)),
                ("layer", syn::parse_quote!(u32)),
                (
                    "weight_fn",
                    syn::parse_quote!(
                        for<'a> fn(&'a Weights, u32) -> &'a ::ferrite_kernels::layers::Fp8AnyLinear
                    ),
                ),
            ],
        )
    }

    fn fan_out(
        &self,
        m: &MatchInfo,
        fuf: &Fuf,
        program: &Program,
        _bounds: &BTreeMap<String, u64>,
        slots: &SlotMap,
    ) -> Option<Vec<OpInstance>> {
        let gelu_id = *m
            .claimed_tiles
            .iter()
            .find(|t| fuf.get(**t).op == OpKind::Gelu)
            .expect("Fp8FusedGateUpGeluMul: claim contains Gelu");
        let mul_id = *m
            .claimed_tiles
            .iter()
            .find(|t| fuf.get(**t).op == OpKind::Mul)
            .expect("Fp8FusedGateUpGeluMul: claim contains Mul");
        let gate_id = match fuf.get(gelu_id).inputs.first() {
            Some(FufInput::Tile { id, .. }) => *id,
            other => panic!("Fp8FusedGateUpGeluMul: Gelu input 0 must be a Tile (got {other:?})"),
        };
        let gate_node = fuf.get(gate_id);
        let (in_id, in_slot) = match gate_node.inputs.first() {
            Some(FufInput::Tile { id, slot }) => (*id, *slot),
            other => {
                panic!("Fp8FusedGateUpGeluMul: gate gemm input 0 must be a Tile (got {other:?})")
            }
        };
        let in_slot_idx = slots.of(in_id, in_slot);
        let out_slot_idx = slots.of(mul_id, 0);
        let accessors = self.required_weights(&m.claimed_tiles, fuf, program);
        let acc = accessors
            .first()
            .expect("Fp8FusedGateUpGeluMul: required_weights returned empty");
        let (base, layer) = split_base_layer(&acc.name.to_string());
        let layer = layer.unwrap_or(0) as u32;
        let base_ident = syn::Ident::new(&base, proc_macro2::Span::call_site());
        Some(vec![OpInstance::new(
            syn::Ident::new("Fp8FusedGateUpGeluMul", proc_macro2::Span::call_site()),
            vec![
                quote! { #in_slot_idx },
                quote! { #out_slot_idx },
                quote! { #layer },
                quote! { Weights::#base_ident },
            ],
        )])
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

    // ── Host-interpreter codegen ────────────────────────────────

    fn opcode_shape(&self) -> OpcodeShape {
        OpcodeShape::new(
            "Fp8FusedQkvRopeCache",
            vec![
                ("in_slot", syn::parse_quote!(u32)),
                ("out_slot", syn::parse_quote!(u32)),
                ("layer", syn::parse_quote!(u32)),
                (
                    "weight_fn",
                    syn::parse_quote!(
                        for<'a> fn(&'a Weights, u32) -> &'a ::ferrite_kernels::layers::Fp8AnyLinear
                    ),
                ),
                (
                    "cos_sin_fn",
                    syn::parse_quote!(
                        for<'a> fn(&'a Weights, u32) -> ::ferrite_cuda_core::tensor::GpuTensor
                    ),
                ),
            ],
        )
    }

    fn fan_out(
        &self,
        m: &MatchInfo,
        fuf: &Fuf,
        program: &Program,
        _bounds: &BTreeMap<String, u64>,
        slots: &SlotMap,
    ) -> Option<Vec<OpInstance>> {
        let rope_id = *m
            .claimed_tiles
            .iter()
            .find(|t| fuf.get(**t).op == OpKind::RopeAppend)
            .expect("Fp8FusedQkvRopeCache: claim contains a RopeAppend");
        let rope_node = fuf.get(rope_id);
        let layer = rope_kv_cache_layer(rope_node)
            .expect("Fp8FusedQkvRopeCache: RopeAppend has KvCache extern with layer index")
            as u32;
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
            .map(|t| {
                unwrap_gemm_through_bias(fuf, *t)
                    .expect("Fp8FusedQkvRopeCache: Gemm-or-BiasAdd(Gemm)")
            })
            .collect();
        let q_gemm_id = resolved[0].0;
        let q_node = fuf.get(q_gemm_id);
        let (in_id, in_slot) = match q_node.inputs.first() {
            Some(FufInput::Tile { id, slot }) => (*id, *slot),
            other => panic!("Fp8FusedQkvRopeCache: q_gemm input 0 must be a Tile (got {other:?})"),
        };
        let in_slot_idx = slots.of(in_id, in_slot);
        let out_slot_idx = slots.of(rope_id, 0);
        let accessors = self.required_weights(&m.claimed_tiles, fuf, program);
        let acc = accessors
            .first()
            .expect("Fp8FusedQkvRopeCache: required_weights returned empty");
        let (base, weight_layer) = split_base_layer(&acc.name.to_string());
        let weight_layer = weight_layer.unwrap_or(layer as u64) as u32;
        assert_eq!(
            weight_layer, layer,
            "Fp8FusedQkvRopeCache: weight_layer disagrees with kv_cache layer"
        );
        let base_ident = syn::Ident::new(&base, proc_macro2::Span::call_site());
        let uses_local = m.claimed_tiles.iter().any(|&tid| {
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
        let cos_sin_ident = if uses_local {
            syn::Ident::new("rotary_local_cos_sin", proc_macro2::Span::call_site())
        } else {
            syn::Ident::new("rotary_cos_sin", proc_macro2::Span::call_site())
        };
        Some(vec![OpInstance::new(
            syn::Ident::new("Fp8FusedQkvRopeCache", proc_macro2::Span::call_site()),
            vec![
                quote! { #in_slot_idx },
                quote! { #out_slot_idx },
                quote! { #layer },
                quote! { Weights::#base_ident },
                quote! { Weights::#cos_sin_ident },
            ],
        )])
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

    // ── Host-interpreter codegen ────────────────────────────────

    fn opcode_shape(&self) -> OpcodeShape {
        OpcodeShape::new(
            "Fp8FusedQkvRopePrefill",
            vec![
                ("in_slot", syn::parse_quote!(u32)),
                ("q_out_slot", syn::parse_quote!(u32)),
                ("k_out_slot", syn::parse_quote!(u32)),
                ("v_out_slot", syn::parse_quote!(u32)),
                ("layer", syn::parse_quote!(u32)),
                (
                    "weight_fn",
                    syn::parse_quote!(
                        for<'a> fn(&'a Weights, u32) -> &'a ::ferrite_kernels::layers::Fp8AnyLinear
                    ),
                ),
                (
                    "cos_sin_fn",
                    syn::parse_quote!(
                        for<'a> fn(&'a Weights, u32) -> ::ferrite_cuda_core::tensor::GpuTensor
                    ),
                ),
            ],
        )
    }

    fn fan_out(
        &self,
        m: &MatchInfo,
        fuf: &Fuf,
        program: &Program,
        _bounds: &BTreeMap<String, u64>,
        slots: &SlotMap,
    ) -> Option<Vec<OpInstance>> {
        let rope_id = *m
            .claimed_tiles
            .iter()
            .find(|t| fuf.get(**t).op == OpKind::RopeAppend)
            .expect("Fp8FusedQkvRopePrefill: claim contains a RopeAppend");
        let rope_node = fuf.get(rope_id);
        let layer = rope_kv_cache_layer(rope_node)
            .expect("Fp8FusedQkvRopePrefill: RopeAppend has KvCache extern with layer index")
            as u32;
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
            .map(|t| {
                unwrap_gemm_through_bias(fuf, *t)
                    .expect("Fp8FusedQkvRopePrefill: Gemm-or-BiasAdd(Gemm)")
            })
            .collect();
        let q_gemm_id = resolved[0].0;
        let q_node = fuf.get(q_gemm_id);
        let (in_id, in_slot) = match q_node.inputs.first() {
            Some(FufInput::Tile { id, slot }) => (*id, *slot),
            other => {
                panic!("Fp8FusedQkvRopePrefill: q_gemm input 0 must be a Tile (got {other:?})")
            }
        };
        let in_slot_idx = slots.of(in_id, in_slot);
        let q_out = slots.of(rope_id, 0);
        let k_out = slots.of(rope_id, 1);
        let v_out = slots.of(rope_id, 2);
        let accessors = self.required_weights(&m.claimed_tiles, fuf, program);
        let acc = accessors
            .first()
            .expect("Fp8FusedQkvRopePrefill: required_weights returned empty");
        let (base, weight_layer) = split_base_layer(&acc.name.to_string());
        let weight_layer = weight_layer.unwrap_or(layer as u64) as u32;
        assert_eq!(
            weight_layer, layer,
            "Fp8FusedQkvRopePrefill: weight_layer disagrees with kv_cache layer"
        );
        let base_ident = syn::Ident::new(&base, proc_macro2::Span::call_site());
        let uses_local = m.claimed_tiles.iter().any(|&tid| {
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
        let cos_sin_ident = if uses_local {
            syn::Ident::new("rotary_local_cos_sin", proc_macro2::Span::call_site())
        } else {
            syn::Ident::new("rotary_cos_sin", proc_macro2::Span::call_site())
        };
        Some(vec![OpInstance::new(
            syn::Ident::new("Fp8FusedQkvRopePrefill", proc_macro2::Span::call_site()),
            vec![
                quote! { #in_slot_idx },
                quote! { #q_out },
                quote! { #k_out },
                quote! { #v_out },
                quote! { #layer },
                quote! { Weights::#base_ident },
                quote! { Weights::#cos_sin_ident },
            ],
        )])
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

    // ── Host-interpreter codegen ────────────────────────────────

    fn opcode_shape(&self) -> OpcodeShape {
        OpcodeShape::new(
            "Bnb4FusedQkvRopeCache",
            vec![
                ("in_slot", syn::parse_quote!(u32)),
                ("out_slot", syn::parse_quote!(u32)),
                ("layer", syn::parse_quote!(u32)),
                (
                    "weight_fn",
                    syn::parse_quote!(
                        for<'a> fn(
                            &'a Weights,
                            u32,
                        )
                            -> &'a ::ferrite_kernels::layers::Bnb4bitLinear
                    ),
                ),
                (
                    "cos_sin_fn",
                    syn::parse_quote!(
                        for<'a> fn(&'a Weights, u32) -> ::ferrite_cuda_core::tensor::GpuTensor
                    ),
                ),
            ],
        )
    }

    fn fan_out(
        &self,
        m: &MatchInfo,
        fuf: &Fuf,
        program: &Program,
        _bounds: &BTreeMap<String, u64>,
        slots: &SlotMap,
    ) -> Option<Vec<OpInstance>> {
        let rope_id = *m
            .claimed_tiles
            .iter()
            .find(|t| fuf.get(**t).op == OpKind::RopeAppend)
            .expect("Bnb4FusedQkvRopeCache: claim contains a RopeAppend");
        let rope_node = fuf.get(rope_id);
        let layer = rope_kv_cache_layer(rope_node)
            .expect("Bnb4FusedQkvRopeCache: RopeAppend has KvCache extern with layer index")
            as u32;
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
            .map(|t| {
                unwrap_gemm_through_bias(fuf, *t)
                    .expect("Bnb4FusedQkvRopeCache: Gemm-or-BiasAdd(Gemm)")
            })
            .collect();
        let q_gemm_id = resolved[0].0;
        let q_node = fuf.get(q_gemm_id);
        let (in_id, in_slot) = match q_node.inputs.first() {
            Some(FufInput::Tile { id, slot }) => (*id, *slot),
            other => panic!("Bnb4FusedQkvRopeCache: q_gemm input 0 must be a Tile (got {other:?})"),
        };
        let in_slot_idx = slots.of(in_id, in_slot);
        let out_slot_idx = slots.of(rope_id, 0);
        let accessors = self.required_weights(&m.claimed_tiles, fuf, program);
        let acc = accessors
            .first()
            .expect("Bnb4FusedQkvRopeCache: required_weights returned empty");
        let (base, weight_layer) = split_base_layer(&acc.name.to_string());
        let weight_layer = weight_layer.unwrap_or(layer as u64) as u32;
        assert_eq!(
            weight_layer, layer,
            "Bnb4FusedQkvRopeCache: weight_layer disagrees with kv_cache layer"
        );
        let base_ident = syn::Ident::new(&base, proc_macro2::Span::call_site());
        let uses_local = m.claimed_tiles.iter().any(|&tid| {
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
        let cos_sin_ident = if uses_local {
            syn::Ident::new("rotary_local_cos_sin", proc_macro2::Span::call_site())
        } else {
            syn::Ident::new("rotary_cos_sin", proc_macro2::Span::call_site())
        };
        Some(vec![OpInstance::new(
            syn::Ident::new("Bnb4FusedQkvRopeCache", proc_macro2::Span::call_site()),
            vec![
                quote! { #in_slot_idx },
                quote! { #out_slot_idx },
                quote! { #layer },
                quote! { Weights::#base_ident },
                quote! { Weights::#cos_sin_ident },
            ],
        )])
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

    // ── Host-interpreter codegen ────────────────────────────────

    fn opcode_shape(&self) -> OpcodeShape {
        OpcodeShape::new(
            "Bnb4FusedQkvRopePrefill",
            vec![
                ("in_slot", syn::parse_quote!(u32)),
                ("q_out_slot", syn::parse_quote!(u32)),
                ("k_out_slot", syn::parse_quote!(u32)),
                ("v_out_slot", syn::parse_quote!(u32)),
                ("layer", syn::parse_quote!(u32)),
                (
                    "weight_fn",
                    syn::parse_quote!(
                        for<'a> fn(
                            &'a Weights,
                            u32,
                        )
                            -> &'a ::ferrite_kernels::layers::Bnb4bitLinear
                    ),
                ),
                (
                    "cos_sin_fn",
                    syn::parse_quote!(
                        for<'a> fn(&'a Weights, u32) -> ::ferrite_cuda_core::tensor::GpuTensor
                    ),
                ),
            ],
        )
    }

    fn fan_out(
        &self,
        m: &MatchInfo,
        fuf: &Fuf,
        program: &Program,
        _bounds: &BTreeMap<String, u64>,
        slots: &SlotMap,
    ) -> Option<Vec<OpInstance>> {
        let rope_id = *m
            .claimed_tiles
            .iter()
            .find(|t| fuf.get(**t).op == OpKind::RopeAppend)
            .expect("Bnb4FusedQkvRopePrefill: claim contains a RopeAppend");
        let rope_node = fuf.get(rope_id);
        let layer = rope_kv_cache_layer(rope_node)
            .expect("Bnb4FusedQkvRopePrefill: RopeAppend has KvCache extern with layer index")
            as u32;
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
            .map(|t| {
                unwrap_gemm_through_bias(fuf, *t)
                    .expect("Bnb4FusedQkvRopePrefill: Gemm-or-BiasAdd(Gemm)")
            })
            .collect();
        let q_gemm_id = resolved[0].0;
        let q_node = fuf.get(q_gemm_id);
        let (in_id, in_slot) = match q_node.inputs.first() {
            Some(FufInput::Tile { id, slot }) => (*id, *slot),
            other => {
                panic!("Bnb4FusedQkvRopePrefill: q_gemm input 0 must be a Tile (got {other:?})")
            }
        };
        let in_slot_idx = slots.of(in_id, in_slot);
        let q_out = slots.of(rope_id, 0);
        let k_out = slots.of(rope_id, 1);
        let v_out = slots.of(rope_id, 2);
        let accessors = self.required_weights(&m.claimed_tiles, fuf, program);
        let acc = accessors
            .first()
            .expect("Bnb4FusedQkvRopePrefill: required_weights returned empty");
        let (base, weight_layer) = split_base_layer(&acc.name.to_string());
        let weight_layer = weight_layer.unwrap_or(layer as u64) as u32;
        assert_eq!(
            weight_layer, layer,
            "Bnb4FusedQkvRopePrefill: weight_layer disagrees with kv_cache layer"
        );
        let base_ident = syn::Ident::new(&base, proc_macro2::Span::call_site());
        let uses_local = m.claimed_tiles.iter().any(|&tid| {
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
        let cos_sin_ident = if uses_local {
            syn::Ident::new("rotary_local_cos_sin", proc_macro2::Span::call_site())
        } else {
            syn::Ident::new("rotary_cos_sin", proc_macro2::Span::call_site())
        };
        Some(vec![OpInstance::new(
            syn::Ident::new("Bnb4FusedQkvRopePrefill", proc_macro2::Span::call_site()),
            vec![
                quote! { #in_slot_idx },
                quote! { #q_out },
                quote! { #k_out },
                quote! { #v_out },
                quote! { #layer },
                quote! { Weights::#base_ident },
                quote! { Weights::#cos_sin_ident },
            ],
        )])
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

    // ── Host-interpreter codegen ────────────────────────────────
    //
    // Variant `FlashInferAttentionDecode { in_slot, out_slot, layer,
    // cos_sin_fn, head_dim, use_logits_soft_cap }`. All instances of
    // `FlashInferAttentionDecodeImpl` (one per (head_dim, softcap)
    // tuple in `FLASHINFER_CONFIG_SET`) share this variant; the
    // tuple values ride as runtime fields. The body tries
    // `flashinfer_attention` and falls back to FA2's
    // `attention_decode_from_cache` when FI returns `None`.

    fn opcode_shape(&self) -> OpcodeShape {
        OpcodeShape::new(
            "FlashInferAttentionDecode",
            vec![
                ("in_slot", syn::parse_quote!(u32)),
                ("out_slot", syn::parse_quote!(u32)),
                ("layer", syn::parse_quote!(u32)),
                (
                    "cos_sin_fn",
                    syn::parse_quote!(
                        for<'a> fn(&'a Weights, u32) -> ::ferrite_cuda_core::tensor::GpuTensor
                    ),
                ),
                ("head_dim", syn::parse_quote!(u32)),
                ("use_logits_soft_cap", syn::parse_quote!(bool)),
            ],
        )
    }

    fn fan_out(
        &self,
        m: &MatchInfo,
        fuf: &Fuf,
        _program: &Program,
        _bounds: &BTreeMap<String, u64>,
        slots: &SlotMap,
    ) -> Option<Vec<OpInstance>> {
        let tile = m.claimed_tiles[0];
        let node = fuf.get(tile);
        let (in_id, in_slot) = match node.inputs.first() {
            Some(FufInput::Tile { id, slot }) => (*id, *slot),
            other => {
                panic!("FlashInferAttentionDecode: input 0 (Q) must be a Tile (got {other:?})")
            }
        };
        let in_slot_idx = slots.of(in_id, in_slot);
        let out_slot_idx = slots.of(tile, 0);
        let layer = node
            .inputs
            .iter()
            .find_map(|i| match i {
                FufInput::Extern {
                    kind: ExternKind::KvCache,
                    index: Some(layer),
                } => Some(*layer),
                _ => None,
            })
            .expect("FlashInferAttentionDecode: kv_cache extern with layer index")
            as u32;
        let uses_local = m.claimed_tiles.iter().any(|&tid| {
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
        let cos_sin_ident = if uses_local {
            syn::Ident::new("rotary_local_cos_sin", proc_macro2::Span::call_site())
        } else {
            syn::Ident::new("rotary_cos_sin", proc_macro2::Span::call_site())
        };
        let head_dim = self.head_dim;
        let softcap = self.use_logits_soft_cap;
        Some(vec![OpInstance::new(
            syn::Ident::new("FlashInferAttentionDecode", proc_macro2::Span::call_site()),
            vec![
                quote! { #in_slot_idx },
                quote! { #out_slot_idx },
                quote! { #layer },
                quote! { Weights::#cos_sin_ident },
                quote! { #head_dim },
                quote! { #softcap },
            ],
        )])
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

    // ── Host-interpreter codegen ────────────────────────────────

    fn opcode_shape(&self) -> OpcodeShape {
        OpcodeShape::new(
            "FlashInferAttentionPrefill",
            vec![
                ("q_slot", syn::parse_quote!(u32)),
                ("k_slot", syn::parse_quote!(u32)),
                ("v_slot", syn::parse_quote!(u32)),
                ("out_slot", syn::parse_quote!(u32)),
                ("layer", syn::parse_quote!(u32)),
                ("head_dim", syn::parse_quote!(u32)),
                ("use_logits_soft_cap", syn::parse_quote!(bool)),
            ],
        )
    }

    fn fan_out(
        &self,
        m: &MatchInfo,
        fuf: &Fuf,
        _program: &Program,
        _bounds: &BTreeMap<String, u64>,
        slots: &SlotMap,
    ) -> Option<Vec<OpInstance>> {
        let tile = m.claimed_tiles[0];
        let node = fuf.get(tile);
        let resolve = |idx: usize| -> (TileId, u8) {
            match node.inputs.get(idx) {
                Some(FufInput::Tile { id, slot }) => (*id, *slot),
                other => {
                    panic!("FlashInferAttentionPrefill: input {idx} must be a Tile (got {other:?})")
                }
            }
        };
        let (q_id, q_in) = resolve(0);
        let (k_id, k_in) = resolve(1);
        let (v_id, v_in) = resolve(2);
        let q_slot = slots.of(q_id, q_in);
        let k_slot = slots.of(k_id, k_in);
        let v_slot = slots.of(v_id, v_in);
        let out_slot = slots.of(tile, 0);
        let layer = node
            .inputs
            .iter()
            .find_map(|i| match i {
                FufInput::Extern {
                    kind: ExternKind::KvCache,
                    index: Some(layer),
                } => Some(*layer),
                _ => None,
            })
            .expect("FlashInferAttentionPrefill: kv_cache extern with layer index")
            as u32;
        let head_dim = self.head_dim;
        let softcap = self.use_logits_soft_cap;
        Some(vec![OpInstance::new(
            syn::Ident::new("FlashInferAttentionPrefill", proc_macro2::Span::call_site()),
            vec![
                quote! { #q_slot },
                quote! { #k_slot },
                quote! { #v_slot },
                quote! { #out_slot },
                quote! { #layer },
                quote! { #head_dim },
                quote! { #softcap },
            ],
        )])
    }
}

// ── MlaSplitRefImpl ──────────────────────────────────────────────────────────
//
// Singleton for `OpKind::MlaSplit` — DeepSeek MLA's KV-A split.
//
// Input:  kv_a  [T, kv_lora_rank + qk_rope_head_dim]
// Output 0: kv_latent  [T, kv_lora_rank]
// Output 1: k_pe       [T, qk_rope_head_dim]
//
// Emits two alloc_tensor calls (one per output) then mla_split_kv_a.
// Both outputs are OwnedTensors — no aliasing with upstream.

#[derive(Debug, Default)]
pub struct MlaSplitRefImpl;

impl Implementation for MlaSplitRefImpl {
    fn name(&self) -> &'static str {
        "mla_split_ref"
    }

    fn target_compatible(&self, _profile: &TargetProfile) -> bool {
        true
    }

    fn matches(&self, fuf: &Fuf, seed: TileId, _profile: &TargetProfile) -> Option<MatchInfo> {
        single_tile_match(fuf, seed, OpKind::MlaSplit)
    }

    fn cost_us(&self, _m: &MatchInfo, ctx: &CostCtx) -> f64 {
        let m = ctx.num_tokens() as f64;
        let kv_lora_rank = ctx.bounds.get("kv_lora_rank").copied().unwrap_or(0) as f64;
        let rope = ctx.bounds.get("qk_rope_head_dim").copied().unwrap_or(0) as f64;
        let bytes = m * (kv_lora_rank + rope) * 2.0 * BYTES_PER_ELEM; // read + write
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

    // ── Host-interpreter codegen ────────────────────────────────
    //
    // Variant `MlaSplit { in_slot, kv_latent_slot, k_pe_slot }`.
    // Two Owned outputs — both freshly allocated by the kernel.

    fn opcode_shape(&self) -> OpcodeShape {
        OpcodeShape::new(
            "MlaSplit",
            vec![
                ("in_slot", syn::parse_quote!(u32)),
                ("kv_latent_slot", syn::parse_quote!(u32)),
                ("k_pe_slot", syn::parse_quote!(u32)),
            ],
        )
    }

    fn fan_out(
        &self,
        m: &MatchInfo,
        fuf: &Fuf,
        _program: &Program,
        _bounds: &BTreeMap<String, u64>,
        slots: &SlotMap,
    ) -> Option<Vec<OpInstance>> {
        let tile = m.claimed_tiles[0];
        let node = fuf.get(tile);
        let (in_id, in_slot) = match node.inputs.first() {
            Some(FufInput::Tile { id, slot }) => (*id, *slot),
            other => panic!("MlaSplit: input 0 must be a Tile (got {other:?})"),
        };
        let in_slot_idx = slots.of(in_id, in_slot);
        let kv_latent_slot = slots.of(tile, 0);
        let k_pe_slot = slots.of(tile, 1);
        Some(vec![OpInstance::new(
            syn::Ident::new("MlaSplit", proc_macro2::Span::call_site()),
            vec![
                quote! { #in_slot_idx },
                quote! { #kv_latent_slot },
                quote! { #k_pe_slot },
            ],
        )])
    }
}

// ── MlaAttentionImpl ─────────────────────────────────────────────────────────
//
// Singleton for `OpKind::MlaAttention` — DeepSeek MLA full attention.
//
// DSL args: mla_attention(q, kv_b, k_pe, positions, rotary, kv_cache[layer], block_table)
//   q    [T, num_heads * qk_head_dim]  (q_nope + q_pe interleaved per head)
//   kv_b [T, num_heads * (nope_dim + v_head_dim)]
//   k_pe [T, rope_dim]  (single head)
//
// Emits the full MLA sequence:
//   1. mla_extract_q_pe (q → q_pe scratch)
//   2. rotary_embedding_interleaved_inplace (q_pe + k_pe)
//   3. mla_write_q_pe (rotated q_pe → back into q)
//   4. mla_assemble_k (kv_b + k_pe → K)
//   5. mla_assemble_v (kv_b → V zero-padded)
//   6. write_kv_cache  (K, V → paged pool)
//   7. attention_standard
//   8. mla_slice_attn_output (qk_head_dim → v_head_dim)
//
// Output: [T, num_heads * v_head_dim]

#[derive(Debug, Default)]
pub struct MlaAttentionImpl;

impl Implementation for MlaAttentionImpl {
    fn name(&self) -> &'static str {
        "mla_attention_ref"
    }

    fn target_compatible(&self, _profile: &TargetProfile) -> bool {
        true
    }

    fn matches(&self, fuf: &Fuf, seed: TileId, _profile: &TargetProfile) -> Option<MatchInfo> {
        single_tile_match(fuf, seed, OpKind::MlaAttention)
    }

    fn cost_us(&self, m: &MatchInfo, ctx: &CostCtx) -> f64 {
        // Use same analytic estimate as standard attention.
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

    // ── Host-interpreter codegen ────────────────────────────────
    //
    // Variant `MlaAttention { q_slot, kv_b_slot, k_pe_slot, out_slot,
    // layer, cos_sin_fn }`. The scale (with optional YaRN mscale) and
    // all head-dim bounds are baked from `model` at codegen time.

    fn opcode_shape(&self) -> OpcodeShape {
        OpcodeShape::new(
            "MlaAttention",
            vec![
                ("q_slot", syn::parse_quote!(u32)),
                ("kv_b_slot", syn::parse_quote!(u32)),
                ("k_pe_slot", syn::parse_quote!(u32)),
                ("out_slot", syn::parse_quote!(u32)),
                ("layer", syn::parse_quote!(u32)),
                (
                    "cos_sin_fn",
                    syn::parse_quote!(
                        for<'a> fn(&'a Weights, u32) -> ::ferrite_cuda_core::tensor::GpuTensor
                    ),
                ),
            ],
        )
    }

    fn fan_out(
        &self,
        m: &MatchInfo,
        fuf: &Fuf,
        _program: &Program,
        _bounds: &BTreeMap<String, u64>,
        slots: &SlotMap,
    ) -> Option<Vec<OpInstance>> {
        let tile = m.claimed_tiles[0];
        let node = fuf.get(tile);
        let resolve = |idx: usize| -> (TileId, u8) {
            match node.inputs.get(idx) {
                Some(FufInput::Tile { id, slot }) => (*id, *slot),
                other => panic!("MlaAttention: input {idx} must be a Tile (got {other:?})"),
            }
        };
        let (q_id, q_in) = resolve(0);
        let (kv_b_id, kv_b_in) = resolve(1);
        let (k_pe_id, k_pe_in) = resolve(2);
        let q_slot = slots.of(q_id, q_in);
        let kv_b_slot = slots.of(kv_b_id, kv_b_in);
        let k_pe_slot = slots.of(k_pe_id, k_pe_in);
        let out_slot = slots.of(tile, 0);
        let layer = node
            .inputs
            .iter()
            .find_map(|i| match i {
                FufInput::Extern {
                    kind: ExternKind::KvCache,
                    index: Some(layer),
                } => Some(*layer),
                _ => None,
            })
            .expect("MlaAttention: kv_cache extern with concrete layer index")
            as u32;
        let uses_local = m.claimed_tiles.iter().any(|&tid| {
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
        let cos_sin_ident = if uses_local {
            syn::Ident::new("rotary_local_cos_sin", proc_macro2::Span::call_site())
        } else {
            syn::Ident::new("rotary_cos_sin", proc_macro2::Span::call_site())
        };
        Some(vec![OpInstance::new(
            syn::Ident::new("MlaAttention", proc_macro2::Span::call_site()),
            vec![
                quote! { #q_slot },
                quote! { #kv_b_slot },
                quote! { #k_pe_slot },
                quote! { #out_slot },
                quote! { #layer },
                quote! { Weights::#cos_sin_ident },
            ],
        )])
    }
}

// ── DeepSeekMoeRefImpl ───────────────────────────────────────────────────────
//
// Singleton for `OpKind::DeepSeekMoe` — DeepSeek MoE layer.
//
// DSL: moe_out = deepseek_moe(hidden_states, moe[layer])
//
// The `moe[layer]` weight resolves to a `DeepSeekV2MoELayer` struct.
// Emits a direct call to `DeepSeekV2MoELayer::forward`.

#[derive(Debug, Default)]
pub struct DeepSeekMoeRefImpl;

impl Implementation for DeepSeekMoeRefImpl {
    fn name(&self) -> &'static str {
        "deepseek_moe_ref"
    }

    fn target_compatible(&self, _profile: &TargetProfile) -> bool {
        true
    }

    fn matches(&self, fuf: &Fuf, seed: TileId, _profile: &TargetProfile) -> Option<MatchInfo> {
        single_tile_match(fuf, seed, OpKind::DeepSeekMoe)
    }

    fn cost_us(&self, _m: &MatchInfo, _ctx: &CostCtx) -> f64 {
        UNCALIBRATED_COST_US
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
        // The `moe[layer]` DSL weight maps to a DeepSeekV2MoELayer.
        // Bypass `default_required_weights` which would assign LinearLayer type.
        let tile = claimed_tiles[0];
        let node = fuf.get(tile);
        let mut out = Vec::new();
        let mut seen = std::collections::HashSet::new();
        for input in &node.inputs {
            if let FufInput::Weight { id, index, .. } = input {
                let name = weight_field_name(program, *id, *index);
                if !seen.insert(name.to_string()) {
                    continue;
                }
                out.push(WeightAccessor {
                    name,
                    rust_type: quote! {
                        ::ferrite_kernels::layers_moe::DeepSeekV2MoELayer
                    },
                    source_weights: vec![(*id, *index)],
                });
            }
        }
        out
    }

    // ── Host-interpreter codegen ────────────────────────────────
    //
    // Variant `DeepSeekMoe { in_slot, out_slot, layer, weight_fn }`
    // where `weight_fn` returns `&DeepSeekV2MoELayer`. The body is a
    // straight `weight.forward(view, device)` call — the MoE layer
    // owns expert dispatch + topk routing internally.

    fn opcode_shape(&self) -> OpcodeShape {
        OpcodeShape::new(
            "DeepSeekMoe",
            vec![
                ("in_slot", syn::parse_quote!(u32)),
                ("out_slot", syn::parse_quote!(u32)),
                ("layer", syn::parse_quote!(u32)),
                (
                    "weight_fn",
                    syn::parse_quote!(
                        for<'a> fn(
                            &'a Weights,
                            u32,
                        )
                            -> &'a ::ferrite_kernels::layers_moe::DeepSeekV2MoELayer
                    ),
                ),
            ],
        )
    }

    fn fan_out(
        &self,
        m: &MatchInfo,
        fuf: &Fuf,
        program: &Program,
        _bounds: &BTreeMap<String, u64>,
        slots: &SlotMap,
    ) -> Option<Vec<OpInstance>> {
        let tile = m.claimed_tiles[0];
        let node = fuf.get(tile);
        let (in_id, in_slot) = match node.inputs.first() {
            Some(FufInput::Tile { id, slot }) => (*id, *slot),
            other => panic!("DeepSeekMoe: input 0 must be a Tile (got {other:?})"),
        };
        let in_slot_idx = slots.of(in_id, in_slot);
        let out_slot_idx = slots.of(tile, 0);
        let accessors = self.required_weights(&m.claimed_tiles, fuf, program);
        let acc = accessors
            .first()
            .expect("DeepSeekMoe: required_weights returned empty");
        let (base, layer) = split_base_layer(&acc.name.to_string());
        let layer = layer.unwrap_or(0) as u32;
        let base_ident = syn::Ident::new(&base, proc_macro2::Span::call_site());
        Some(vec![OpInstance::new(
            syn::Ident::new("DeepSeekMoe", proc_macro2::Span::call_site()),
            vec![
                quote! { #in_slot_idx },
                quote! { #out_slot_idx },
                quote! { #layer },
                quote! { Weights::#base_ident },
            ],
        )])
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn unmigrated_shape_carries_distinctive_placeholder_name() {
        // A shape returned from the trait default must be marked
        // distinctively enough that codegen can refuse it without a
        // false positive against a real Impl. We don't rely on
        // matching the placeholder by string in production codegen
        // — the `fan_out` → `None` check is authoritative — but if
        // the placeholder ever leaks into emitted code, it should
        // fail compilation loudly with a recognisable identifier.
        let s = OpcodeShape::unmigrated("AddRefImpl");
        assert_eq!(s.name.to_string(), "__Unmigrated");
        assert!(s.fields.is_empty());
    }

    #[test]
    fn op_instance_field_values_match_shape_field_count() {
        // Soft contract: `OpInstance::field_values.len()` should
        // equal `OpcodeShape::fields.len()` for the same variant.
        // The codegen will assert this at lower-time; the type
        // doesn't enforce it, so this test pins the convention
        // alongside a reference Impl-style call.
        let shape = OpcodeShape::new("Free", vec![("slot", syn::parse_quote!(u32))]);
        let inst = OpInstance::new(shape.name.clone(), vec![quote! { 7u32 }]);
        assert_eq!(inst.field_values.len(), shape.fields.len());
    }

    #[test]
    fn slot_map_assigns_dense_indices_in_insert_order() {
        let mut sm = SlotMap::new();
        assert_eq!(sm.insert(TileId(7), 0), 0);
        assert_eq!(sm.insert(TileId(7), 1), 1);
        assert_eq!(sm.insert(TileId(3), 0), 2);
        // Repeat returns the original index.
        assert_eq!(sm.insert(TileId(7), 0), 0);
        assert_eq!(sm.total(), 3);
        assert_eq!(sm.of(TileId(3), 0), 2);
    }

    #[test]
    #[should_panic(expected = "not registered")]
    fn slot_map_of_unregistered_panics() {
        let sm = SlotMap::new();
        let _ = sm.of(TileId(1), 0);
    }

    #[test]
    fn attention_scalars_default_to_llama_values_when_config_silent() {
        // Llama-3.2-1B has no `query_pre_attn_scalar` or
        // `attn_logit_softcapping` — the helpers must fall back to
        // `1/sqrt(head_dim)` and `0.0` respectively, matching the
        // previously hardcoded emission byte-for-byte.
        let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("..")
            .join("ferrite-model-llama")
            .join("configs")
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
        let profile = crate::target::from_profile_def(&ferrite_cuda_targets::L4_SM89);
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
        let profile = crate::target::from_profile_def(&ferrite_cuda_targets::L4_SM89);
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
    fn cutlass_gemm_impl_matches_when_output_feeds_rope_append() {
        // Architectural invariant: singletons (CutlassGemm{,SplitK,
        // Add,v}, GemmRefImpl, Fp8Gemm) MUST match unconditionally on
        // structural fit — they don't gate on what the downstream
        // consumer is. The DP solver decides between a multi-tile
        // fused claim and a chain of singletons by cost + feasibility:
        //   - When `FusedQkvRopeImpl` matches, it claims [Q,K,V,rope]
        //     in one multi-tile mask; DP picks it when cheaper.
        //   - When fused doesn't match (qwen3/gemma3 per-head qk-norm),
        //     singletons are the only feasible plan and win.
        // Singleton-side gating ("don't match if output feeds X") was
        // a hand-coded preference that the DP already encodes via cost
        // and feasibility — and it was wrong for qwen3 V-proj, where
        // it forced cuBLAS over a measurably faster CUTLASS variant.
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
        let profile = crate::target::from_profile_def(&ferrite_cuda_targets::L4_SM89);
        assert!(
            imp.matches(&fuf, t1, &profile).is_some(),
            "CutlassGemmImpl must match a Gemm structurally regardless of \
             downstream consumer; fused-vs-singleton is the DP's job",
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
            kernel_fits: std::collections::HashMap::new(),
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
    /// `decompose_reshape_dim` mirrors today's `reshape_dim_token`'s
    /// folding: every config bound + literal collapses into the
    /// `lit_part`; `num_tokens` is the only runtime factor and gets
    /// counted in `nt_pow`. Mul recurses with multiplicative
    /// composition; Var panics.
    #[test]
    fn decompose_reshape_dim_folds_config_bounds_and_counts_num_tokens() {
        use crate::shape::Dim;
        let mut bounds = std::collections::BTreeMap::new();
        bounds.insert("hidden_size".to_string(), 4096u64);
        bounds.insert("head_dim".to_string(), 64u64);
        bounds.insert("num_attention_heads".to_string(), 32u64);

        // Pure literal.
        assert_eq!(decompose_reshape_dim(&Dim::Lit(64), &bounds), (64, 0));

        // Config bound folds at codegen time.
        assert_eq!(
            decompose_reshape_dim(&Dim::Bound("hidden_size".into()), &bounds),
            (4096, 0)
        );

        // num_tokens is the only runtime factor.
        assert_eq!(
            decompose_reshape_dim(&Dim::Bound("num_tokens".into()), &bounds),
            (1, 1)
        );

        // [num_tokens * num_attention_heads, head_dim]-style
        // Mul folds bounds + counts num_tokens occurrences.
        let dim = Dim::Mul(vec![
            Dim::Bound("num_tokens".into()),
            Dim::Bound("num_attention_heads".into()),
        ]);
        assert_eq!(decompose_reshape_dim(&dim, &bounds), (32, 1));

        // Pure literal product.
        let dim = Dim::Mul(vec![Dim::Lit(2), Dim::Bound("head_dim".into())]);
        assert_eq!(decompose_reshape_dim(&dim, &bounds), (128, 0));
    }
    fn empty_model(name: &str) -> crate::config::ModelParams {
        crate::config::ModelParams {
            name: name.to_string(),
            source_stem: name.into(),
            source_path: std::path::PathBuf::new(),
            bounds: std::collections::BTreeMap::new(),
            scalars: std::collections::BTreeMap::new(),
            quantization: None,
            tie_word_embeddings: false,
            architectures: Vec::new(),
            extra_tracked_paths: Vec::new(),
            rope_scaling: None,
            rope_scaling_hash: None,
        }
    }
    fn attention_model(name: &str) -> crate::config::ModelParams {
        let mut model = empty_model(name);
        model.bounds.insert("num_attention_heads".to_string(), 32);
        model.bounds.insert("num_key_value_heads".to_string(), 8);
        model.bounds.insert("head_dim".to_string(), 128);
        model
    }

    // ── DC sibling Impl flag tests ───────────────────────────────
    //
    // Pin the load-bearing flag set on the new mega-tier siblings:
    // `launch_kind = DeviceCallable`, `megakernel_fit = Primitive`,
    // `target_compatible` gated on `prim_mega_compatible()`. The
    // selector + solver hook off these — a regression here silently
    // routes mega-eligible canonicals through the host interpreter.
    #[test]
    fn dc_rms_norm_advertises_primitive_megakernel_fit() {
        let imp = DcRmsNormImpl;
        assert_eq!(imp.launch_kind(), LaunchKind::DeviceCallable);
        assert_eq!(imp.megakernel_fit(), MegakernelFit::Primitive);
        assert!(MegakernelFit::Primitive >= MegakernelFit::None);
        assert!(MegakernelFit::Primitive < MegakernelFit::Kvm);
    }

    #[test]
    fn dc_rms_norm_target_gate_follows_prim_mega_compatible() {
        // sm_90 (H100) is the prim_mega floor — Hopper's hw-
        // accelerated grid barrier (~15us) is what makes
        // prim-mega competitive with host. Ada / Ampere fall back
        // to gmem-atomic spin counters (~100us per `grid.sync()`)
        // that nuke any mega win, so the gate excludes them.
        let h100 = crate::target::from_profile_def(&ferrite_cuda_targets::H100_SM90);
        assert!(h100.prim_mega_compatible());
        assert!(DcRmsNormImpl.target_compatible(&h100));

        // L4 (sm_89) is below the floor → DC sibling rejected.
        let l4 = crate::target::from_profile_def(&ferrite_cuda_targets::L4_SM89);
        assert!(!l4.prim_mega_compatible());
        assert!(!DcRmsNormImpl.target_compatible(&l4));
    }

    #[test]
    fn dc_rms_norm_handoffs_are_mega_internal_only() {
        // Host-stream handoffs (StreamOrder, StreamEvent,
        // KernelBoundary) are illegal for a DeviceCallable: the
        // PrimMega persistent kernel never crosses a launch
        // boundary mid-tape. Only intra-grid sync mechanisms apply.
        let imp = DcRmsNormImpl;
        let allowed: std::collections::HashSet<_> =
            [Handoff::InKernelGridSync, Handoff::SyncThreads]
                .iter()
                .copied()
                .collect();
        for &h in imp.supported_input_handoffs() {
            assert!(allowed.contains(&h), "input handoff leaked: {h:?}");
        }
        for &h in imp.supported_output_handoffs() {
            assert!(allowed.contains(&h), "output handoff leaked: {h:?}");
        }
    }

    #[test]
    fn dc_rms_norm_and_host_sibling_share_opcode_shape() {
        // Codegen merges variants by ident across Impls; the merge
        // panics on field-shape mismatch. DcRmsNormImpl deliberately
        // contributes the same "RmsNorm" variant + identical fields
        // as RmsNormRefImpl so the host interpreter's eval and the
        // (future) PrimMega encoder both consume the same wire row.
        assert_share_opcode_shape(&RmsNormRefImpl, &DcRmsNormImpl);
    }

    #[test]
    fn dc_fused_add_rms_norm_share_opcode_shape() {
        assert_share_opcode_shape(&FusedAddRmsNormImpl, &DcFusedAddRmsNormImpl);
    }

    #[test]
    fn dc_fused_qkv_rope_cache_share_opcode_shape() {
        assert_share_opcode_shape(&FusedQkvRopeCacheImpl, &DcFusedQkvRopeCacheImpl);
    }

    #[test]
    fn dc_cutlass_gemm_share_opcode_shape() {
        // Tile-shape tuples must agree per-tile; iterate the
        // canonical zoo (DC and host walk the same list) and check
        // each pairing.
        for &(m, n, s) in CUTLASS_TILE_ZOO {
            let host = CutlassGemmImpl {
                tile_m: m,
                tile_n: n,
                stages: s,
            };
            let dc = DcCutlassGemmImpl {
                tile_m: m,
                tile_n: n,
                stages: s,
            };
            assert_share_opcode_shape(&host, &dc);
        }
    }

    /// Helper: pin the variant-merge contract codegen depends on —
    /// host + DC siblings declare structurally identical opcode
    /// shapes. A drift here surfaces as a codegen panic at build
    /// time; the test catches it earlier.
    fn assert_share_opcode_shape(host: &dyn Implementation, dc: &dyn Implementation) {
        let h = host.opcode_shape();
        let d = dc.opcode_shape();
        assert_eq!(
            h.name.to_string(),
            d.name.to_string(),
            "{} vs {} variant ident drift",
            host.name(),
            dc.name()
        );
        assert_eq!(
            h.fields.len(),
            d.fields.len(),
            "{} vs {} field count drift",
            host.name(),
            dc.name()
        );
        for ((a_name, a_ty), (b_name, b_ty)) in h.fields.iter().zip(&d.fields) {
            assert_eq!(
                a_name.to_string(),
                b_name.to_string(),
                "{} vs {} field-name drift",
                host.name(),
                dc.name()
            );
            assert_eq!(
                quote!(#a_ty).to_string(),
                quote!(#b_ty).to_string(),
                "{} vs {} field-type drift on {}",
                host.name(),
                dc.name(),
                a_name
            );
        }
    }

    /// Every CUTLASS_TILE_ZOO entry resolves to a non-`unknown`
    /// `name()` arm on `DcCutlassGemmImpl`. Catches the case where a
    /// tile is added to the zoo but its name-match arm is forgotten;
    /// "dc_cutlass_unknown" would silently trash log/cost-table keys.
    #[test]
    fn dc_cutlass_name_covers_full_zoo() {
        for &(m, n, s) in CUTLASS_TILE_ZOO {
            let dc = DcCutlassGemmImpl {
                tile_m: m,
                tile_n: n,
                stages: s,
            };
            let name = dc.name();
            assert_ne!(
                name, "dc_cutlass_unknown",
                "CUTLASS_TILE_ZOO entry ({m}, {n}, {s}) has no name() arm \
                 in DcCutlassGemmImpl"
            );
            let expected = format!("dc_cutlass_{m}x{n}_s{s}");
            assert_eq!(name, expected);
        }
    }

    /// Sibling pin for `DcCutlassGemmAddImpl`. Same intent as
    /// `dc_cutlass_name_covers_full_zoo` but for the residual-add
    /// variant.
    #[test]
    fn dc_cutlass_gemm_add_name_covers_full_zoo() {
        for &(m, n, s) in CUTLASS_TILE_ZOO {
            let dc = DcCutlassGemmAddImpl {
                tile_m: m,
                tile_n: n,
                stages: s,
            };
            let name = dc.name();
            assert_ne!(
                name, "dc_cutlass_unknown_add",
                "CUTLASS_TILE_ZOO entry ({m}, {n}, {s}) has no name() arm \
                 in DcCutlassGemmAddImpl"
            );
            let expected = format!("dc_cutlass_{m}x{n}_s{s}_add");
            assert_eq!(name, expected);
        }
    }

    #[test]
    fn dc_cutlass_gemm_add_share_opcode_shape() {
        for &(m, n, s) in CUTLASS_TILE_ZOO {
            let host = CutlassGemmAddImpl {
                tile_m: m,
                tile_n: n,
                stages: s,
            };
            let dc = DcCutlassGemmAddImpl {
                tile_m: m,
                tile_n: n,
                stages: s,
            };
            assert_share_opcode_shape(&host, &dc);
        }
    }

    #[test]
    fn dc_cutlass_gemv_share_opcode_shape() {
        assert_share_opcode_shape(&CutlassGemvImpl, &DcCutlassGemvImpl);
    }

    /// Pins `vllm-cuda/csrc/cutlass_gemm_configs.cuh`'s
    /// `CUTLASS_DC_GEMM_LIST` to exactly `CUTLASS_TILE_ZOO`. Drift on
    /// either side surfaces as a test failure here rather than as a
    /// missing/extra registration the solver silently never picks.
    /// The C++ X-macro can carry warp-shape columns the Rust list
    /// omits — we compare on the (TB_M, TB_N, STAGES) projection,
    /// which is what `CutlassGemmImpl` and `DcCutlassGemmImpl` index
    /// by.
    #[test]
    fn dc_zoo_equals_host_zoo() {
        const CUH: &str = include_str!("../../vllm-cuda/csrc/cutlass_gemm_configs.cuh");
        // Parse rows of the form `X( TBM, TBN, TBK, STAGES, …)`.
        // Take only entries inside the `CUTLASS_DC_GEMM_LIST` macro
        // body (everything after `#define CUTLASS_DC_GEMM_LIST(X)`
        // up to the trailing entry without a backslash). The cuh
        // file currently has only one X-macro list, so a permissive
        // `X(…)` match suffices.
        let mut cpp: std::collections::BTreeSet<(u32, u32, u32)> =
            std::collections::BTreeSet::new();
        for line in CUH.lines() {
            let line = line.trim();
            if !line.starts_with("X(") {
                continue;
            }
            let inner = line
                .trim_start_matches("X(")
                .trim_end_matches('\\')
                .trim()
                .trim_end_matches(')')
                .trim();
            let cols: Vec<&str> = inner.split(',').map(|s| s.trim()).collect();
            // Need at least TBM, TBN, TBK, STAGES.
            if cols.len() < 4 {
                continue;
            }
            let tbm: u32 = cols[0]
                .parse()
                .unwrap_or_else(|e| panic!("CUTLASS_DC_GEMM_LIST: bad TB_M `{}`: {e}", cols[0]));
            let tbn: u32 = cols[1]
                .parse()
                .unwrap_or_else(|e| panic!("CUTLASS_DC_GEMM_LIST: bad TB_N `{}`: {e}", cols[1]));
            let stages: u32 = cols[3]
                .parse()
                .unwrap_or_else(|e| panic!("CUTLASS_DC_GEMM_LIST: bad STAGES `{}`: {e}", cols[3]));
            cpp.insert((tbm, tbn, stages));
        }
        let rust: std::collections::BTreeSet<(u32, u32, u32)> =
            CUTLASS_TILE_ZOO.iter().copied().collect();
        let cpp_only: Vec<_> = cpp.difference(&rust).copied().collect();
        let rust_only: Vec<_> = rust.difference(&cpp).copied().collect();
        assert!(
            cpp_only.is_empty() && rust_only.is_empty(),
            "DC tile-zoo drift between Rust CUTLASS_TILE_ZOO and C++ \
             CUTLASS_DC_GEMM_LIST.\n  C++ has but Rust doesn't: \
             {cpp_only:?}\n  Rust has but C++ doesn't: {rust_only:?}\n\
             Both sides must list the same (TB_M, TB_N, STAGES) tuples."
        );
    }
}
