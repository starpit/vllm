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
    MbarrierPhase, PageId, RmsNormScope, ScratchRegion, WarpRoleTag, ROLE_CONSUMER, ROLE_LAUNCHER,
    ROLE_LOADER, ROLE_STORER,
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

/// The typed lowered MegaNode enum. One variant per migrated op.
///
/// Sprint A populates `RmsNorm`. Subsequent sprints add variants
/// per the plan §10 table. The `_Reserved` variant is gone — the
/// enum is non-empty without it.
pub enum MegaNode {
    RmsNorm(RmsNorm),
}
