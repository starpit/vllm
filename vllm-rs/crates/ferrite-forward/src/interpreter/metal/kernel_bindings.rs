// SPDX-License-Identifier: Apache-2.0
//! Typed per-kernel binding-set structs (Phase 3 of
//! `FERRITE_METAL_TYPE_SAFETY_PLAN.md`).
//!
//! Each `BindingSet` struct captures the per-call-variable bindings
//! (`ArenaSlotIdx`, `LayerId`, weight thunks) for one kernel family.
//! The runtime-fixed bindings (`CuSeqlensQ`, `SeqUsedK`, `BlockTable`,
//! `NumTokensU32`) are baked into the `From<Self> for Vec<Binding<W>>`
//! conversion — they live at the same per-kernel binding index every
//! time so the lowering arm doesn't restate them.
//!
//! This catches the buffer-slot analogue of bug class #1 (the
//! `ATTN_PAGED_DEBUG_MODE` slot-99 omission): emitting a Vec<Binding>
//! by hand makes it possible to skip a runtime binding silently;
//! constructing the struct + converting can't.
//!
//! Phase 3 lands the high-binding-count attention / RoPE / QKV
//! kernels. The simpler 2-3 binding kernels (RmsNorm, SiluMul,
//! ScalarMul, Add) stay on hand-rolled `vec![Binding::…]` for now —
//! the per-kernel struct boilerplate isn't pulling its weight at that
//! size, and the visual inspection is trivial.

use super::ids::{ArenaSlotIdx, LayerId};
use super::lowered::{
    Binding, RuntimeBindingKind, WeightBundleKind, WeightLocator, WeightTensor,
};

// ── AttentionPrefillSdpaPaged (both sdpa_vector and steel variants) ─

/// Bindings for `KernelId::AttentionPrefillSdpaPaged`. Seven slots:
/// 0 = output (arena), 1 = Q (arena), 2 = CuSeqlensQ (runtime),
/// 3 = SeqUsedK (runtime), 4 = BlockTable (runtime),
/// 5 = KvCacheK\[layer\] (runtime), 6 = KvCacheV\[layer\] (runtime).
pub struct AttentionPrefillPagedBindingSet {
    pub output: ArenaSlotIdx,
    pub q: ArenaSlotIdx,
    pub kv_layer: LayerId,
}

impl From<AttentionPrefillPagedBindingSet> for Vec<Binding> {
    fn from(s: AttentionPrefillPagedBindingSet) -> Vec<Binding> {
        vec![
            Binding::ArenaSlot {
                slot: s.output.get(),
                binding_index: 0,
            },
            Binding::ArenaSlot {
                slot: s.q.get(),
                binding_index: 1,
            },
            Binding::Runtime {
                kind: RuntimeBindingKind::CuSeqlensQ,
                binding_index: 2,
            },
            Binding::Runtime {
                kind: RuntimeBindingKind::SeqUsedK,
                binding_index: 3,
            },
            Binding::Runtime {
                kind: RuntimeBindingKind::BlockTable,
                binding_index: 4,
            },
            Binding::Runtime {
                kind: RuntimeBindingKind::KvCacheK { layer: s.kv_layer },
                binding_index: 5,
            },
            Binding::Runtime {
                kind: RuntimeBindingKind::KvCacheV { layer: s.kv_layer },
                binding_index: 6,
            },
        ]
    }
}

// ── AttentionViaCache (decode) ─────────────────────────────────────

/// Bindings for `KernelId::AttentionViaCache`. Six slots:
/// 0 = output (arena), 1 = Q (arena), 2 = SeqUsedK (runtime),
/// 3 = BlockTable (runtime), 4 = KvCacheK\[layer\] (runtime),
/// 5 = KvCacheV\[layer\] (runtime).
///
/// Same `kv_layer` payload appears at slots 4 + 5 — the layer index
/// is single-field on the BindingSet so they can't drift.
pub struct AttentionViaCacheBindingSet {
    pub output: ArenaSlotIdx,
    pub q: ArenaSlotIdx,
    pub kv_layer: LayerId,
}

impl From<AttentionViaCacheBindingSet> for Vec<Binding> {
    fn from(s: AttentionViaCacheBindingSet) -> Vec<Binding> {
        vec![
            Binding::ArenaSlot {
                slot: s.output.get(),
                binding_index: 0,
            },
            Binding::ArenaSlot {
                slot: s.q.get(),
                binding_index: 1,
            },
            Binding::Runtime {
                kind: RuntimeBindingKind::SeqUsedK,
                binding_index: 2,
            },
            Binding::Runtime {
                kind: RuntimeBindingKind::BlockTable,
                binding_index: 3,
            },
            Binding::Runtime {
                kind: RuntimeBindingKind::KvCacheK { layer: s.kv_layer },
                binding_index: 4,
            },
            Binding::Runtime {
                kind: RuntimeBindingKind::KvCacheV { layer: s.kv_layer },
                binding_index: 5,
            },
        ]
    }
}

// ── RopeAppend ─────────────────────────────────────────────────────

/// Bindings for `KernelId::RopeAppend`. Eight slots:
/// 0 = Q-rotated out (arena), 1 = K out (arena), 2 = V out (arena),
/// 3 = CosSin\[layer\] (weight), 4 = Positions (runtime),
/// 5 = SlotMapping (runtime), 6 = KvCacheK\[layer\] (runtime),
/// 7 = KvCacheV\[layer\] (runtime).
///
/// The `layer` on the cos/sin table MUST match the layer on the
/// KvCache* slots — single-field `layer` on the BindingSet enforces
/// it.
pub struct RopeAppendBindingSet {
    pub q_out: ArenaSlotIdx,
    pub k_out: ArenaSlotIdx,
    pub v_out: ArenaSlotIdx,
    pub cos_sin_locator: WeightLocator,
    pub layer: LayerId,
}

impl From<RopeAppendBindingSet> for Vec<Binding> {
    fn from(s: RopeAppendBindingSet) -> Vec<Binding> {
        vec![
            Binding::ArenaSlot {
                slot: s.q_out.get(),
                binding_index: 0,
            },
            Binding::ArenaSlot {
                slot: s.k_out.get(),
                binding_index: 1,
            },
            Binding::ArenaSlot {
                slot: s.v_out.get(),
                binding_index: 2,
            },
            Binding::Weight {
                kind: WeightBundleKind::CosSin,
                which: WeightTensor::Weight,
                layer: s.layer,
                locator: s.cos_sin_locator,
                binding_index: 3,
            },
            Binding::Runtime {
                kind: RuntimeBindingKind::Positions,
                binding_index: 4,
            },
            Binding::Runtime {
                kind: RuntimeBindingKind::SlotMapping,
                binding_index: 5,
            },
            Binding::Runtime {
                kind: RuntimeBindingKind::KvCacheK { layer: s.layer },
                binding_index: 6,
            },
            Binding::Runtime {
                kind: RuntimeBindingKind::KvCacheV { layer: s.layer },
                binding_index: 7,
            },
        ]
    }
}

// ── FusedQkvRopeCache (dense BF16/F16) ─────────────────────────────

/// Bindings for `KernelId::FusedQkvRopeCache`. Eight slots:
/// 0 = Q out (arena), 1 = input (arena),
/// 2 = packed \[Q|K|V\] weight (LinearLayer, weight),
/// 3 = CosSin\[layer\] (weight), 4 = Positions (runtime),
/// 5 = SlotMapping (runtime), 6 = KvCacheK\[layer\] (runtime),
/// 7 = KvCacheV\[layer\] (runtime).
pub struct FusedQkvRopeCacheBindingSet {
    pub q_out: ArenaSlotIdx,
    pub input: ArenaSlotIdx,
    pub qkv_locator: WeightLocator,
    pub cos_sin_locator: WeightLocator,
    pub layer: LayerId,
}

impl From<FusedQkvRopeCacheBindingSet> for Vec<Binding> {
    fn from(s: FusedQkvRopeCacheBindingSet) -> Vec<Binding> {
        vec![
            Binding::ArenaSlot {
                slot: s.q_out.get(),
                binding_index: 0,
            },
            Binding::ArenaSlot {
                slot: s.input.get(),
                binding_index: 1,
            },
            Binding::Weight {
                kind: WeightBundleKind::LinearLayer,
                which: WeightTensor::Weight,
                layer: s.layer,
                locator: s.qkv_locator,
                binding_index: 2,
            },
            Binding::Weight {
                kind: WeightBundleKind::CosSin,
                which: WeightTensor::Weight,
                layer: s.layer,
                locator: s.cos_sin_locator,
                binding_index: 3,
            },
            Binding::Runtime {
                kind: RuntimeBindingKind::Positions,
                binding_index: 4,
            },
            Binding::Runtime {
                kind: RuntimeBindingKind::SlotMapping,
                binding_index: 5,
            },
            Binding::Runtime {
                kind: RuntimeBindingKind::KvCacheK { layer: s.layer },
                binding_index: 6,
            },
            Binding::Runtime {
                kind: RuntimeBindingKind::KvCacheV { layer: s.layer },
                binding_index: 7,
            },
        ]
    }
}
