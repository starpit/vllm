// SPDX-License-Identifier: Apache-2.0
//! `MetalWorkerPool::forward()` types — Phase 5.E.
//!
//! `forward()` itself lives on `MetalWorkerPool` (see [`super::pool`])
//! because it owns checkout/checkin around the encoder lifecycle. The
//! types in this module — [`ForwardInputs`] and [`ForwardError`] — are
//! the user-facing surface of that method, kept separate so `pool.rs`
//! stays focused on the growable/capped checkout machinery.
//!
//! Per `FERRITE_METAL_ARCHITECTURE.md` §1, the per-forward path on the
//! Metal side is the hot path: bucket pick → checkout → bind inputs →
//! `executeCommandsInBuffer` → checkin. Bindings hit the GPU via the
//! buffer pointers baked into the ICB at recording time (Option B);
//! the engine refills the runtime buffers' `contents()` each call.
//! [`ForwardInputs`] is the "data the engine wants written into those
//! buffers" — the forward path validates each present slice against
//! the worker's per-buffer capacity and copies the bytes through.

#![cfg(feature = "metal")]

use crate::interpreter::metal::__re::MTLCommandBufferStatus;

use super::worker::WorkerError;

/// One forward step's runtime inputs.
///
/// The bucket selected by [`super::pool::MetalWorkerPool::pick_bucket`]
/// determines which slices the active tape's commands consume. Buckets
/// that don't reference a particular runtime buffer (e.g. a decode
/// bucket has no `cu_seqlens_q`) accept `None` for that field — the
/// forward path only copies fields the caller provides. Validation
/// against the worker's runtime buffer sizes happens before any GPU
/// work is submitted, so a malformed input never partially executes.
///
/// `num_tokens` is the *real* token count for this step. It selects
/// the bucket (smallest `bucket_m >= num_tokens`) and bounds how many
/// elements of each per-token array (`input_ids`, `positions`,
/// `slot_mapping`) are meaningful. The shader still processes
/// `bucket_m` work units — anything past `num_tokens` is padding the
/// worker fills with whatever was last in the runtime buffer; the
/// model owns interpreting the result for `[0, num_tokens)` only.
pub struct ForwardInputs<'a> {
    /// Number of real tokens this step processes.
    pub num_tokens: u32,
    /// `[num_tokens]` u32 — token ids to embed.
    pub input_ids: &'a [u32],
    /// `[num_tokens]` u32 — RoPE position per token.
    pub positions: &'a [u32],
    /// `[num_tokens]` u32 — paged-cache slot per token. Required for
    /// any bucket that runs `RopeAppend` (writes K/V into the cache);
    /// `None` for buckets that don't append.
    pub slot_mapping: Option<&'a [u32]>,
    /// `[batch+1]` u32 — prefill-only cumulative sequence boundaries.
    /// `None` for decode buckets.
    pub cu_seqlens_q: Option<&'a [u32]>,
    /// `[batch]` u32 — current K-axis used length per sequence.
    /// Required for decode buckets (read by `AttentionViaCache`);
    /// `None` for first-token-only prefill.
    pub seq_used_k: Option<&'a [u32]>,
    /// `[batch, max_blocks]` u32 — per-sequence block table for the
    /// paged KV cache. Required wherever `AttentionViaCache` or
    /// `RopeAppend` references the paged pool.
    pub block_table: Option<&'a [u32]>,
    /// `true` when this forward is a spec-decode verify batch (one
    /// or more reqs carries `spec_token_ids`). Threaded into the
    /// dispatch loop's `gate_matches` so the lm_head slice trio (gated
    /// `OnlyIfSingleSeqNoSpec`) skips and the full-`M=bucket_m`
    /// fallback (gated `OnlyIfMultiSeqOrSpec`) fires instead — the
    /// slice writes only the LAST row of logits, which is wrong when
    /// rejection sampling needs every row.
    pub has_spec_tokens: bool,
}

/// Errors produced by [`super::pool::MetalWorkerPool::forward`] before
/// or during a forward step.
#[derive(Debug)]
pub enum ForwardError {
    /// `num_tokens == 0`. Selecting a bucket for a no-op step makes
    /// no sense — the engine must guard before calling forward.
    ZeroTokens,
    /// `num_tokens` exceeded every bucket's `bucket_m`. The engine
    /// either needs to chunk the step or the model loader needs to
    /// add a wider bucket.
    NoBucketFits { num_tokens: u32, max_bucket: u32 },
    /// One of the input slices was bigger than its runtime buffer.
    /// The runtime buffer was sized at `RuntimeFactory` invocation —
    /// either the factory under-sized it for this bucket or the
    /// caller is staging more elements than the bucket admits.
    BufferTooSmall {
        kind: &'static str,
        bytes_needed: usize,
        bytes_available: usize,
    },
    /// Worker encoding / lookup failed (pipeline lookup, GEMM encode,
    /// arena slot out of range, …).
    Worker(WorkerError),
    /// `commit` + `wait_until_completed` finished with a non-Completed
    /// status. This is the GPU-side failure mode — Metal exposes only
    /// the enum, not the underlying NSError.
    ExecutionFailed(MTLCommandBufferStatus),
    /// A caller-supplied followup hook (e.g. argmax encode + wait
    /// chained on the forward CB's shared event) failed.
    Followup(String),
}

impl std::fmt::Display for ForwardError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::ZeroTokens => write!(f, "MetalWorkerPool::forward: num_tokens == 0"),
            Self::NoBucketFits {
                num_tokens,
                max_bucket,
            } => write!(
                f,
                "MetalWorkerPool::forward: no bucket fits num_tokens={num_tokens} \
                 (max bucket_m = {max_bucket})"
            ),
            Self::BufferTooSmall {
                kind,
                bytes_needed,
                bytes_available,
            } => write!(
                f,
                "MetalWorkerPool::forward: runtime buffer `{kind}` too small \
                 (needs {bytes_needed} bytes, have {bytes_available})"
            ),
            Self::Worker(e) => write!(f, "MetalWorkerPool::forward: {e}"),
            Self::ExecutionFailed(status) => write!(
                f,
                "MetalWorkerPool::forward: command buffer status = {status:?} (expected Completed)"
            ),
            Self::Followup(msg) => write!(f, "MetalWorkerPool::forward: followup hook failed: {msg}"),
        }
    }
}

impl std::error::Error for ForwardError {}

impl From<WorkerError> for ForwardError {
    fn from(e: WorkerError) -> Self {
        Self::Worker(e)
    }
}
