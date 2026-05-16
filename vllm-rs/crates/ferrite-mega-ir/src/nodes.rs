// SPDX-License-Identifier: Apache-2.0
//! Typed lowered-form variants.
//!
//! One variant per `Instruction<W>` kind. Each variant's
//! load-bearing fields are substrate proofs from
//! [`crate::substrate`] — `PageId` (validated against
//! `num_pages`), `MbarrierPhase` (parity-checked against cumulative
//! arrive count), `ScratchRegion<Scope>` (within-budget +
//! disjoint), `WarpRoleTag<R>` (sealed valid-role) — chosen so that
//! constructing a variant value requires the substrate-layout
//! invariants the corresponding bug class targets (see
//! `MEGA_IR_PLAN.md` §1, §4).
//!
//! Helper newtypes (e.g. [`LayerIndex`], [`WeightRef`]) MAY ride
//! alongside for runtime weight-pointer resolution; they are NOT
//! load-bearing on their own (per `MEGA_IR_PLAN.md` §3, §4).

#![allow(dead_code)]

use crate::substrate::{
    IterCount, MbarrierPhase, PageId, ROLE_CONSUMER, ROLE_LAUNCHER, ROLE_LOADER, ROLE_STORER,
    RmsNormScope, RopeScope, ScratchRegion, WarpRoleTag,
};

/// Helper newtype: layer index for runtime weight-pointer
/// resolution. Variant fields use this ALONGSIDE substrate-proof
/// fields; never as the load-bearing field.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct LayerIndex(u32);

impl LayerIndex {
    pub fn new(idx: u32, num_layers: u32) -> Self {
        assert!(
            idx < num_layers,
            "LayerIndex out of range: {idx} >= {num_layers}",
        );
        Self(idx)
    }

    pub fn raw(self) -> u32 {
        self.0
    }
}

/// Helper newtype: per-layer weight accessor path string.
/// Construction enforces non-empty / non-whitespace.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct WeightRef(String);

impl WeightRef {
    pub fn new(path: String) -> Self {
        assert!(
            !path.trim().is_empty(),
            "WeightRef must be a non-empty path string",
        );
        Self(path)
    }

    pub fn path(&self) -> &str {
        &self.0
    }
}

/// Per-iter mbarrier wait/arrive sequence for one warp role.
/// Sprint A: RmsNorm has two consumer waits (input page,
/// weight page) per iter at the same phase, one consumer arrive
/// (output ready), and one storer wait (output) per iter.
///
/// `WarpRoleTag<R>` pins the role at the type level (bug class
/// #6). `MbarrierPhase` carries the parity validated against the
/// cumulative arrive count up to this op (bug class #3).
pub struct PerIterPhases<const R: u8>
where
    WarpRoleTag<R>: crate::substrate::IsValidWarpRole,
{
    pub wait_phase: MbarrierPhase,
    pub _role: WarpRoleTag<R>,
}

/// The typed lowered RmsNorm variant.
///
/// Substrate proofs (load-bearing):
/// - `in_page`: the input row's page slot id (validated within
///   `num_pages`). Lifecycle: was Empty at op start, was filled by
///   the loader, was read by the consumer who then arrived, was
///   read by the storer, is back to Empty at op end. The lowering
///   walks Page<State> typestate transitions to prove this; the
///   stored PageId records the slot id only.
/// - `weight_page`: the weight row's page slot id. Lifecycle: was
///   Empty, filled by loader, read by consumer (no storer reads
///   weight). Returns to Empty at op end.
/// - `partial_sums`: per-iter scratch region for the cross-warp
///   sum-of-squares reduction. ScratchRegion<RmsNormScope> proves
///   `[offset, offset+bytes) ⊆ [0, num_consumer_warps * 4)` AND
///   within `substrate.scratch_bytes` (bug classes #4, #5).
/// - `consumer_phase`, `storer_phase`: phase parity for each
///   wait site, validated against the cumulative arrive count
///   (bug class #3).
/// - `_loader_role`, `_launcher_role`, `_consumer_role`,
///   `_storer_role`: type-level role tags (bug class #6 — the
///   role pairing is fixed at variant construction; cross-role
///   action calls in the emit step have no way to refer to a
///   role this variant doesn't claim).
///
/// Helper fields (alongside, not load-bearing):
/// - `layer`: for emit-time weight-pointer resolution.
/// - `weight`: weight accessor path.
pub struct RmsNorm {
    pub in_page: PageId,
    pub weight_page: PageId,
    pub partial_sums: ScratchRegion<RmsNormScope>,
    pub consumer_phase: MbarrierPhase,
    pub storer_phase: MbarrierPhase,
    pub _loader_role: WarpRoleTag<ROLE_LOADER>,
    pub _launcher_role: WarpRoleTag<ROLE_LAUNCHER>,
    pub _consumer_role: WarpRoleTag<ROLE_CONSUMER>,
    pub _storer_role: WarpRoleTag<ROLE_STORER>,
    pub layer: LayerIndex,
    pub weight: WeightRef,
}

/// Helper newtype: per-arch rotary cache accessor — distinguishes
/// the global rotary (`wm.rotary.cos_sin_cache`) from per-layer
/// local rotary (`wm.rotary_local.cos_sin_cache`) on archs that
/// carry both (Gemma3). Construction enforces non-empty path.
/// Not load-bearing on its own (per `MEGA_IR_PLAN.md` §3, §4).
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct RotaryRef(String);

impl RotaryRef {
    pub fn new(path: String) -> Self {
        assert!(
            !path.trim().is_empty(),
            "RotaryRef must be a non-empty path string",
        );
        Self(path)
    }

    pub fn path(&self) -> &str {
        &self.0
    }
}

/// The typed lowered FusedQkvRopeCache variant.
///
/// Substrate proofs (load-bearing):
///
/// **Six page slots, each walked Empty → Filled → Produced → Empty**
/// at proc-macro construction time, so the recorded `PageId`s are
/// witness to a valid lifecycle (bug class #2):
///
/// - `in_page`: post-norm activation row feeding the QKV gemm.
/// - `qkv_weight_page`: packed Q/K/V projection weights (single
///   accessor; the LinearLayer wrapper packs the three projections
///   into one buffer at load time).
/// - `cos_sin_page`: rotary cos/sin cache page.
/// - `q_out_page`, `k_out_page`, `v_out_page`: per-token outputs.
///   Q goes to the next op (attention); K/V additionally write to
///   the global paged KV cache (the `kv_cache_layer` extern; not
///   substrate-managed because it lives in HBM, not shmem-pages).
///
/// `PageId::raw()` is bounds-validated against
/// `SubstrateBudget::num_pages` (bug class #1).
///
/// **Two scratch regions, both in `RopeScope`, validated disjoint**
/// at proc-macro time (bug class #4):
///
/// - `q_rope_buf`: per-token Q rotation tile in shmem.
/// - `k_rope_buf`: per-token K rotation tile in shmem.
///
/// Each is independently validated within `SubstrateBudget::scratch_bytes`
/// (bug class #5); their disjointness is checked via
/// [`ScratchRegion::disjoint_with`].
///
/// **Per-iter mbarrier phases** (bug class #3):
///
/// - `consumer_phase`: parity at iter 0 of the consumer wait. The
///   lowering proves this matches the cumulative arrive count at
///   op start.
/// - `storer_phase`: parity at iter 0 of the storer wait. Lowered
///   to `(arrives + 1) & 1` — the storer waits AFTER the consumer
///   arrives.
/// - `iters`: per-token iteration count (one consumer arrive per
///   token-iter, so the cumulative arrive count advances by
///   `iters.raw() * 1` after this op completes).
///
/// The kernel-internal `phase ^= 1` advances per iter; given the
/// initial phase parity matches the start-of-op cumulative count
/// and `arrives_per_iter == 1`, the kernel-internal phase at iter
/// `t` matches `(C_start + t) & 1` — the parity the hardware
/// barrier expects (proof: `(a+b) & 1 == (a&1) ^ (b&1)`).
///
/// **Warp role tags** (bug class #6): all four roles claimed; mis-
/// pairing across roles in the variant declaration is rejected by
/// the sealed-trait bound on `WarpRoleTag<R>` at the field level.
///
/// Helper fields (alongside, not load-bearing):
/// - `layer`: emit-time layer index for weight-pointer resolution.
/// - `qkv_weight`: weight accessor path (LinearLayer holding packed Q/K/V).
/// - `rotary`: rotary accessor path (global or local cos/sin cache).
/// - `biased`: dense bias attached (Qwen2-family) vs no bias (Llama).
/// - `interleaved`: interleaved-rope half-rotation pattern.
pub struct FusedQkvRopeCache {
    pub in_page: PageId,
    pub qkv_weight_page: PageId,
    pub cos_sin_page: PageId,
    pub q_out_page: PageId,
    pub k_out_page: PageId,
    pub v_out_page: PageId,
    pub q_rope_buf: ScratchRegion<RopeScope>,
    pub k_rope_buf: ScratchRegion<RopeScope>,
    pub consumer_phase: MbarrierPhase,
    pub storer_phase: MbarrierPhase,
    pub iters: IterCount,
    pub _loader_role: WarpRoleTag<ROLE_LOADER>,
    pub _launcher_role: WarpRoleTag<ROLE_LAUNCHER>,
    pub _consumer_role: WarpRoleTag<ROLE_CONSUMER>,
    pub _storer_role: WarpRoleTag<ROLE_STORER>,
    pub layer: LayerIndex,
    pub qkv_weight: WeightRef,
    pub rotary: RotaryRef,
    pub biased: bool,
    pub interleaved: bool,
}

/// The typed lowered MegaNode enum. One variant per migrated op.
///
/// Sprint A populates `RmsNorm`. Sprint B adds `FusedQkvRopeCache`.
/// Subsequent sprints add the rest per the plan §10 table.
pub enum MegaNode {
    RmsNorm(RmsNorm),
    FusedQkvRopeCache(FusedQkvRopeCache),
}
