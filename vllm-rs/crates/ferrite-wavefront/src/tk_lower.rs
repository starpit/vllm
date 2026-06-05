// SPDX-License-Identifier: Apache-2.0
//! `tk_lower` — TK-tape-side reflections of the SubtileIR typed
//! witnesses, holding the BufId binding the SubtileIR layer doesn't
//! have.
//!
//! **Canonical witnesses live in [`crate::subtile_ir`]** (per plan §4
//! commit 4):
//!   - [`crate::subtile_ir::KvCacheLayout`] — TensorId-keyed,
//!     target-agnostic.
//!   - [`crate::subtile_ir::KvCacheProducer`] — sealed,
//!     `#[non_exhaustive]`, target-agnostic.
//!
//! The two TK-flavored copies in this file ([`KvCacheLayout`] +
//! [`KvCacheProducer`]) carry the BufId binding the TkTape needs and
//! are slated for replacement by a single `lower_tape_to_tk` mapping
//! in plan §4 commit 6 (TensorId → BufId at lowering time). Until
//! then, this file stays for tk_tape.rs's existing imports.

use crate::metal_tape::BufId;

// ── KvCacheLayout — single source of K/V offset math ────────────────

/// Sealed witness for the K (or V) cache layout of one forward pass.
/// The orchestrator builds ONE per K-cache `BufId`; RopeAppend's write
/// and AttnDecode's read both reach for the SAME instance, making
/// drift between producer and consumer structurally impossible.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct KvCacheLayout {
    cache_buf_id: BufId,
    num_kv_heads: u32,
    head_dim: u32,
    act_elem: u32,
}

impl KvCacheLayout {
    /// Sealed constructor binding `cache_buf_id` into the witness.
    pub const fn for_buf_id(
        cache_buf_id: BufId,
        num_kv_heads: u32,
        head_dim: u32,
        act_elem: u32,
    ) -> Self {
        Self {
            cache_buf_id,
            num_kv_heads,
            head_dim,
            act_elem,
        }
    }

    pub const fn cache_buf_id(&self) -> BufId {
        self.cache_buf_id
    }

    pub const fn num_kv_heads(&self) -> u32 {
        self.num_kv_heads
    }

    pub const fn head_dim(&self) -> u32 {
        self.head_dim
    }

    pub const fn act_elem(&self) -> u32 {
        self.act_elem
    }

    /// Per-token K (or V) row stride in bytes.
    pub const fn row_bytes(&self) -> u64 {
        (self.num_kv_heads as u64) * (self.head_dim as u64) * (self.act_elem as u64)
    }

    /// Per-position cos/sin row stride in bytes.
    pub const fn cos_sin_row_bytes(&self) -> u64 {
        (self.head_dim as u64) * (self.act_elem as u64)
    }
}

// ── KvCacheProducer — typed dataflow edge ───────────────────────────

/// Sealed enum naming HOW the K (or V) cache that AttnDecode reads got
/// populated. Replaces the orchestrator's prior silent
/// `unwrap_or_else(|| GmemHandle::new_initial(...))` fallback.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum KvCacheProducer {
    /// The cache was written by a RopeAppend earlier in this same
    /// forward (LoweringInput::ops[producer_op_idx]).
    SameForwardRopeAppend {
        producer_op_idx: u32,
        #[doc(hidden)]
        _seal: sealed_kv_producer::Seal,
    },
    /// The cache is pre-populated by an out-of-band per-op forward
    /// and is read-only inside this megakernel.
    PrePopulatedExt {
        #[doc(hidden)]
        _seal: sealed_kv_producer::Seal,
    },
}

#[doc(hidden)]
pub mod sealed_kv_producer {
    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    pub struct Seal(pub(super) ());
}

impl KvCacheProducer {
    pub const fn from_rope_append(producer_op_idx: u32) -> Self {
        Self::SameForwardRopeAppend {
            producer_op_idx,
            _seal: sealed_kv_producer::Seal(()),
        }
    }

    pub const fn pre_populated_ext() -> Self {
        Self::PrePopulatedExt {
            _seal: sealed_kv_producer::Seal(()),
        }
    }
}
