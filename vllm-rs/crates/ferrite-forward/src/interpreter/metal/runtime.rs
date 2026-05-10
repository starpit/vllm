// SPDX-License-Identifier: Apache-2.0
//! Per-forward runtime buffers that the worker re-binds across forwards.
//!
//! Each [`Binding::Runtime`](super::lowered::Binding::Runtime) on the
//! lowered tape names one of these slots. The worker bakes the
//! buffer's pointer into the ICB at recording time (Option B per
//! `FERRITE_METAL_ARCHITECTURE.md` §1) — so the buffers themselves
//! must outlive every call. The engine writes new content into each
//! buffer's `contents()` on every forward and re-runs
//! `executeCommandsInBuffer`; no ICB re-recording happens on the
//! per-forward path.
//!
//! [`RuntimeBindings`] is the Metal analogue of CUDA's `ForwardCtx`:
//! one struct per worker, per-shape buffers sized for the worker's
//! largest bucket. KV cache buffers are split per layer because the
//! lowered tape carries the layer index on `KvCacheK`/`KvCacheV`
//! variants (each layer gets its own `Buffer`); other runtime tensors
//! are global per worker.

#![cfg(feature = "metal")]

use crate::interpreter::metal::__re::{Buffer, MTLBuffer};

use super::lowered::RuntimeBindingKind;

/// Per-worker runtime buffers. The worker reads `&Buffer` for each
/// runtime binding while baking its bucket ICBs; the engine refills
/// `contents()` on every forward.
///
/// All `Vec<Buffer>` fields are indexed by layer id (the `layer`
/// payload on the matching [`RuntimeBindingKind`] variant). The
/// constructor walks model meta + bucket layout to size every field
/// for the worker's largest bucket; the worker itself only borrows.
pub struct RuntimeBindings {
    pub input_ids: Buffer,
    pub positions: Buffer,
    pub slot_mapping: Buffer,
    pub cu_seqlens_q: Buffer,
    pub seq_used_k: Buffer,
    pub block_table: Buffer,
    /// Per-layer paged K cache buffers. Shape: `[num_layers]`.
    pub kv_cache_k: Vec<Buffer>,
    /// Per-layer paged V cache buffers. Shape: `[num_layers]`.
    pub kv_cache_v: Vec<Buffer>,
}

impl RuntimeBindings {
    /// Resolve a [`RuntimeBindingKind`] to the buffer the worker
    /// should bake into the ICB. Panics if a `KvCache*` layer index
    /// exceeds the held `Vec` length — that's a model-meta bug, not a
    /// data-driven failure mode.
    pub fn buffer_for(&self, kind: RuntimeBindingKind) -> &Buffer {
        match kind {
            RuntimeBindingKind::InputIds => &self.input_ids,
            RuntimeBindingKind::Positions => &self.positions,
            RuntimeBindingKind::SlotMapping => &self.slot_mapping,
            RuntimeBindingKind::CuSeqlensQ => &self.cu_seqlens_q,
            RuntimeBindingKind::SeqUsedK => &self.seq_used_k,
            RuntimeBindingKind::BlockTable => &self.block_table,
            RuntimeBindingKind::KvCacheK { layer } => &self.kv_cache_k[layer as usize],
            RuntimeBindingKind::KvCacheV { layer } => &self.kv_cache_v[layer as usize],
        }
    }
}
