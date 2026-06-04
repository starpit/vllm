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

use crate::interpreter::metal::__re::Buffer;

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
    /// `[1]` u32 — actual `num_tokens` of the in-flight forward.
    /// The pool writes the current call's `num_tokens` into this
    /// 4-byte buffer at the start of every `forward()` so kernels
    /// that need M at runtime can read it without a function
    /// constant (M varies per call).
    pub num_tokens_u32: Buffer,
    /// `[1]` u32 — number of sample rows the lm_head slice trio
    /// gathers / GEMMs / scatters this forward. Equals the length
    /// of the in-flight `last_token_indices` slice (= `num_seqs` for
    /// non-spec prefill/decode, `K+1` for spec-decode verify, `0`
    /// when the caller doesn't pass indices). Read by the
    /// `gather_last_token`/`scatter_first_to_last_row` kernels.
    pub num_sample_rows_u32: Buffer,
    /// `[num_sample_rows]` u32 — per-sample-row source/destination
    /// index into the lm_head input/output tensor. Mirrors Python
    /// vLLM's `logits_indices` and the CUDA path's
    /// `ForwardCtx.last_token_indices`. Sized at construction for
    /// the worker's largest bucket's `bucket_m`; the pool writes
    /// the in-flight slice's content at the start of every forward.
    pub sample_indices: Buffer,
    /// Per-layer GDN conv-state ring buffers (persistent f32 pool),
    /// GLOBAL-layer-indexed like `gdn_state_ssm`. Non-linear (full-attn)
    /// layers hold a dummy buffer that is never bound (the
    /// `GatedDeltaNet` lowering only emits `GdnConvState` on linear
    /// layers). Empty for non-hybrid arches.
    pub gdn_state_conv: Vec<Buffer>,
    /// Per-layer GDN recurrent (ssm) state buffers. Same indexing /
    /// dummy / emptiness contract as [`Self::gdn_state_conv`].
    pub gdn_state_ssm: Vec<Buffer>,
    /// `[num_seqs]` i32 — GDN state-pool slot id per batched sequence.
    /// Shared storage; the pool overwrites `contents()` each forward.
    pub gdn_state_indices: Buffer,
    /// `[num_seqs]` u32 — per-sequence fresh flag. Shared storage.
    pub gdn_is_fresh: Buffer,
    /// `[total_L, vision_head_dim/2]` f32 — vision 2D-RoPE angle table
    /// (`freqs`). Shared storage; the pool overwrites `contents()` each
    /// forward. 16-byte placeholder on non-vision arches.
    pub vision_rope_freqs: Buffer,
    /// `[num_tokens, vision_in_features]` model-dtype — vision patch
    /// pixel rows. Shared storage; overwritten per forward. 16-byte
    /// placeholder on non-vision arches.
    pub pixels: Buffer,
    /// `[num_tokens, vision_embed_dim]` model-dtype — Qwen3.5-VL
    /// host-interpolated learned positional embedding. Shared storage;
    /// overwritten per forward. 16-byte placeholder on non-vision arches
    /// and on towers without a learned positional embedding.
    pub vision_pos_embeds: Buffer,
    /// `[max_m, hidden]` model-dtype — projected vision embeddings for
    /// the multimodal splice. Shared storage; overwritten per forward
    /// (text-only batches leave it untouched). 16-byte placeholder on
    /// arches without the splice.
    pub mm_embeds: Buffer,
    /// `[max_m]` u32 — per-`mm_embeds`-row destination text-embedding row
    /// (`u32::MAX` = skip). Shared storage; overwritten per forward.
    pub mm_dst_rows: Buffer,
    /// `[max_m, ROT_DIM]` model-dtype — per-token MRoPE cos/sin override
    /// table (Qwen3.5-VL text decoder). Shared storage; the pool
    /// overwrites `contents()` each forward with the band-split rows the
    /// macro forward builds. Bound at the rope kernel's cos/sin slot in
    /// place of the static `WeightBundleKind::CosSin` cache when
    /// `W::MROPE_SECTION.is_some()`. 16-byte placeholder on 1D-rope arches.
    pub mrope_cos_sin: Buffer,
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
            RuntimeBindingKind::KvCacheK { layer } => &self.kv_cache_k[layer.get() as usize],
            RuntimeBindingKind::KvCacheV { layer } => &self.kv_cache_v[layer.get() as usize],
            RuntimeBindingKind::NumTokensU32 => &self.num_tokens_u32,
            RuntimeBindingKind::NumSeqsU32 => &self.num_sample_rows_u32,
            RuntimeBindingKind::SampleIndices => &self.sample_indices,
            RuntimeBindingKind::GdnConvState { layer } => {
                &self.gdn_state_conv[layer.get() as usize]
            }
            RuntimeBindingKind::GdnSsmState { layer } => &self.gdn_state_ssm[layer.get() as usize],
            RuntimeBindingKind::GdnStateIndices => &self.gdn_state_indices,
            RuntimeBindingKind::GdnIsFresh => &self.gdn_is_fresh,
            RuntimeBindingKind::VisionRopeFreqs => &self.vision_rope_freqs,
            RuntimeBindingKind::Pixels => &self.pixels,
            RuntimeBindingKind::VisionPosEmbeds => &self.vision_pos_embeds,
            RuntimeBindingKind::MmEmbeds => &self.mm_embeds,
            RuntimeBindingKind::MmDstRows => &self.mm_dst_rows,
            RuntimeBindingKind::MropeCosSin => &self.mrope_cos_sin,
        }
    }
}
