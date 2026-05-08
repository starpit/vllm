// SPDX-License-Identifier: Apache-2.0
//! `MetalWorker`: arena + per-bucket ICB recording.
//!
//! One worker holds:
//!  - a private arena of `metal::Buffer`s, sized by the model's
//!    post-coloring slot count (one buffer per slot id);
//!  - one [`BucketBaking`] per bucket, where each baking is a fully
//!    recorded [`IndirectCommandBuffer`] plus an *execution plan*.
//!
//! The execution plan is the missing piece the architecture doc
//! glosses over. On Apple Silicon, ICBs *must* run with
//! `inheritPipelineState=true` (per
//! `PHASE4_ICB_BREAKTHROUGH.md`) — the encoder sets the pipeline,
//! and every command in the ICB inherits it. So a bucket's ICB
//! cannot mix kernels in a single `executeCommandsInBuffer` call.
//!
//! The worker handles this by partitioning the bucket's command
//! stream into *segments*: contiguous runs of commands that share a
//! pipeline. Recording still goes into one ICB per bucket (commands
//! are stored linearly at indices `[0, num_commands)`), but the
//! per-forward path walks the bucket's [`Vec<ExecSegment>`] and does
//! one `set_compute_pipeline_state(...)` + one
//! `executeCommandsInBuffer(range)` per segment. Adjacent commands
//! with byte-identical pipelines coalesce; commands that need
//! different specialized pipelines (e.g. per-layer `RmsNorm.eps`
//! varying) become separate segments.
//!
//! `KernelId::Gemm` is routed differently. MPS' `matmul2d` is opaque
//! to the function-constant cache — and it doesn't fit into an ICB
//! either, since `encodeToCommandBuffer` opens its own internal
//! compute encoder(s). 5.C.5 lifts the per-bucket plan to a
//! [`Vec<BucketStep>`] where each step is either an ICB run (under
//! one `MTLComputePipelineState`) or an MPS GEMM dispatch. The
//! per-forward path walks the steps; ICB steps reuse the encoder
//! they share, and a `Gemm` step ends the current encoder, encodes
//! the GEMM directly into the command buffer, and the next ICB step
//! opens a fresh encoder.

#![cfg(feature = "metal")]

use std::sync::Arc;

use ferrite_metal_kernels::gemm::{GemmError, encode_gemm_into_command_buffer};
use ferrite_metal_kernels::instruction_executor::RecordingContext;
use ferrite_metal_kernels::metal::foreign_types::ForeignType;
use ferrite_metal_kernels::metal::{
    Buffer, CommandBufferRef, ComputeCommandEncoderRef, ComputePipelineState, Device,
    MTLResourceOptions, MTLResourceUsage, MTLSize, ResourceRef,
};

use super::lowered::{
    Binding, KernelId, LoweredCommand, LoweredMetalTape, WeightBundleKind, WeightTensor,
};
use super::pipelines::{PipelineLookupError, SpecializedPipelines};
use super::runtime::RuntimeBindings;
use crate::CanonicalParams;
use ferrite_cuda_core::MetalAllocator;

/// Byte size of arena slot `i`. The macro's `colored_slot_map()`
/// computes this from the FUF's per-slot shape × dtype × max bucket;
/// for the Phase 5.C smoke test the test sets it explicitly.
pub type ArenaLayout = Vec<u64>;

/// One unit of execution in a bucket's plan.
///
/// `Icb` is a contiguous run of ICB commands sharing a single
/// pipeline state — fired with one
/// `set_compute_pipeline_state` + `executeCommandsInBuffer` pair.
/// `Gemm` is an MPS dense `y = x @ W^T` dispatch encoded directly
/// into the command buffer between ICB encoder boundaries.
pub enum BucketStep {
    Icb {
        /// Pipeline state bound to the encoder before the range
        /// fires. Held for lifetime so the ICB's
        /// `inheritPipelineState=true` inherit picks up a live
        /// pipeline.
        pipeline: ComputePipelineState,
        /// Commands `[start, end)` in the bucket's ICB.
        range: std::ops::Range<usize>,
        /// `KernelId` of the first command in `range`. Diagnostic-only
        /// (used by `run_bucket_per_step_debug` to label which kernel
        /// is firing); coalesced ICB ranges share one pipeline so this
        /// kernel id is constant across the range.
        kernel: KernelId,
        /// Resources THIS step's kernel(s) actually bind. Subset of
        /// the bucket's `baked_resources`. Used for per-encoder
        /// `useResources` so we don't pay O(N) for buffers this
        /// dispatch doesn't read.
        ///
        /// Empirically: passing all 200+ bucket-wide resources to
        /// `useResources` on each per-step encoder cost ~3s/dispatch
        /// for `AttentionViaCache` on Apple Silicon (debug binary).
        /// Restricting to just the kernel's bound buffers drops it to
        /// ~200us.
        step_resources: Vec<Buffer>,
        /// Per-command bindings for direct (non-ICB) dispatch.
        /// Outer Vec: one entry per command in `range`. Inner Vec:
        /// (buffer, offset, binding_index) tuples for that command.
        ///
        /// Lets us bypass the ICB execution path (which we suspect of
        /// adding ~3s overhead per AttentionViaCache call) by
        /// dispatching the kernel directly via `setBuffer` +
        /// `dispatchThreadgroups`.
        direct_bindings: Vec<Vec<(Buffer, u64, u64)>>,
        /// Per-command dispatch shape: `(threadgroups, threads_per_threadgroup)`.
        direct_dispatch: Vec<(MTLSize, MTLSize)>,
    },
    Gemm {
        /// Activation buffer bound to MPS' `leftMatrix` (shape
        /// `[m, k]`).
        a: BoundBuffer,
        /// Weight buffer bound to MPS' `rightMatrix`. Linear-layer
        /// convention is `[n, k]` with `transposeRight=true`.
        b: BoundBuffer,
        /// Output buffer bound to MPS' `resultMatrix` (shape
        /// `[m, n]`).
        c: BoundBuffer,
        m: u32,
        n: u32,
        k: u32,
    },
}

/// A `(buffer, offset)` pair held by a `BucketStep::Gemm`. The
/// arena/weight buffers themselves outlive the worker (the arena
/// lives on the worker; weight buffers live on the model meta which
/// the pool keeps alive), so a non-owning `Buffer` clone is
/// equivalent to an `Arc` clone — `metal::Buffer` is itself a
/// reference-counted handle.
pub struct BoundBuffer {
    pub buffer: Buffer,
    pub offset: u64,
}

/// One bucket's baked artifacts: the ICB (commands recorded linearly
/// at indices `[0, num_commands)`) and the execution plan walking it.
///
/// The ICB stores ICB commands only — GEMM steps are *not* recorded
/// into the ICB (MPS doesn't fit ICBs). The `steps` vector is the
/// authoritative ordering; ICB ranges in `BucketStep::Icb` index into
/// `icb`, while `BucketStep::Gemm` stands alone.
///
/// `baked_resources` is the set of unique buffers referenced by ICB
/// commands. The descriptor is built with `inheritBuffers=false` so
/// the GPU needs every ICB-referenced buffer marked resident on the
/// firing encoder via `useResources:count:usage:` — otherwise the
/// command buffer fails with status `Error`. Collected once at bake
/// time so the per-forward residency call is a single Objective-C
/// message with the cached slice.
pub struct BucketBaking {
    pub bucket_m: u32,
    pub icb: RecordingContext,
    pub steps: Vec<BucketStep>,
    pub baked_resources: Vec<Buffer>,
}

#[derive(Debug)]
pub enum WorkerError {
    /// Lookup against [`SpecializedPipelines`] failed.
    PipelineLookup(PipelineLookupError),
    /// `RecordingContext::new` returned an error (ICB descriptor
    /// rejected by the device, max_count too small, …).
    Recording(String),
    /// `arena_layout.len()` did not match the lowered tape's
    /// `num_arena_slots`. Indicates a mismatched lowering and arena
    /// computation upstream — the macro should keep these in sync.
    ArenaShapeMismatch { expected: u32, actual: usize },
    /// A `Binding::ArenaSlot { slot, .. }` referenced a slot id
    /// outside `[0, arena_layout.len())`.
    ArenaSlotOutOfRange {
        bucket_index: usize,
        command_index: usize,
        slot: u32,
        arena_len: usize,
    },
    /// `KernelId::Gemm` reached the bake step but its
    /// `LoweredCommand::gemm_dims` was `None`. Indicates a lowering
    /// bug — the lowering pass owns populating those for `Gemm`
    /// commands.
    MissingGemmDims {
        bucket_index: usize,
        command_index: usize,
    },
    /// `KernelId::Gemm` had unexpected bindings. The worker expects
    /// (output, input, weight) at indices 0/1/2 — anything else
    /// is a lowering / model-meta contract violation.
    GemmBindingsMalformed {
        bucket_index: usize,
        command_index: usize,
        reason: &'static str,
    },
    /// `encode_gemm_into_command_buffer` rejected the dispatch.
    GemmEncode(GemmError),
    /// Resolving a `Binding::Weight` via the WtFn thunk + allocator
    /// failed. Either the layer struct didn't carry the requested
    /// tensor (e.g. bias absent on a no-bias linear) or the tensor's
    /// raw pointer didn't fall inside any of the allocator's arenas.
    WeightLookupFailed { reason: &'static str },
}

impl std::fmt::Display for WorkerError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::PipelineLookup(e) => write!(f, "MetalWorker: pipeline lookup: {e}"),
            Self::Recording(e) => write!(f, "MetalWorker: ICB recording: {e}"),
            Self::ArenaShapeMismatch { expected, actual } => write!(
                f,
                "MetalWorker: arena shape mismatch: tape expects {expected} slots, layout has {actual}"
            ),
            Self::ArenaSlotOutOfRange {
                bucket_index,
                command_index,
                slot,
                arena_len,
            } => write!(
                f,
                "MetalWorker: bucket {bucket_index} command {command_index} \
                 references arena slot {slot} but arena has only {arena_len} slots"
            ),
            Self::MissingGemmDims {
                bucket_index,
                command_index,
            } => write!(
                f,
                "MetalWorker: bucket {bucket_index} command {command_index}: \
                 KernelId::Gemm has no gemm_dims (lowering bug)"
            ),
            Self::GemmBindingsMalformed {
                bucket_index,
                command_index,
                reason,
            } => write!(
                f,
                "MetalWorker: bucket {bucket_index} command {command_index}: \
                 GEMM bindings malformed ({reason})"
            ),
            Self::GemmEncode(e) => write!(f, "MetalWorker: MPS GEMM encode: {e}"),
            Self::WeightLookupFailed { reason } => {
                write!(f, "MetalWorker: weight lookup: {reason}")
            }
        }
    }
}

impl std::error::Error for WorkerError {}

/// One per concurrent forward. Owns its arena + per-bucket bakings;
/// borrows the model meta + runtime bindings + pipeline cache via
/// references threaded through `new`.
pub struct MetalWorker<W: CanonicalParams> {
    pub arena: Vec<Buffer>,
    pub bucket_bakings: Vec<BucketBaking>,
    _marker: std::marker::PhantomData<fn() -> W>,
}

impl<W: CanonicalParams> MetalWorker<W> {
    /// Build a worker.
    ///
    /// `arena_layout[i]` = byte size of arena slot `i`. The arena is
    /// sized for the worker's largest bucket; downstream forwards
    /// re-use the same arena across buckets.
    ///
    /// `bucket_tapes` is the per-bucket lowered tape set, in the
    /// caller's bucket order. The worker bakes one ICB + execution
    /// plan per tape.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        device: Arc<Device>,
        arena_layout: &ArenaLayout,
        bucket_tapes: &[LoweredMetalTape<W>],
        pipelines: &SpecializedPipelines,
        weights: &W,
        allocator: &MetalAllocator,
        runtime: &RuntimeBindings,
    ) -> Result<Self, WorkerError> {
        // Arena slot count comes from the lowered tape (post-FUF
        // coloring). Every bucket of a given model shares the same
        // colored slot map, so checking the first bucket is enough.
        if let Some(first) = bucket_tapes.first()
            && first.num_arena_slots as usize != arena_layout.len()
        {
            return Err(WorkerError::ArenaShapeMismatch {
                expected: first.num_arena_slots,
                actual: arena_layout.len(),
            });
        }

        let arena: Vec<Buffer> = arena_layout
            .iter()
            .map(|&size| {
                // Shared storage so test code can seed/inspect arena
                // contents without staging copies. ICB-bound buffers
                // are fine in shared on Apple silicon — the existing
                // `MetalAllocator` uses the same mode.
                device.new_buffer(size, MTLResourceOptions::StorageModeShared)
            })
            .collect();

        let mut bucket_bakings = Vec::with_capacity(bucket_tapes.len());
        for (bucket_idx, tape) in bucket_tapes.iter().enumerate() {
            let baking = bake_bucket(
                bucket_idx,
                tape,
                &arena,
                pipelines,
                weights,
                allocator,
                runtime,
                device.clone(),
            )?;
            bucket_bakings.push(baking);
        }

        Ok(Self {
            arena,
            bucket_bakings,
            _marker: std::marker::PhantomData,
        })
    }

    /// Walk a bucket's plan against `cmdbuf`. For ICB steps the
    /// worker opens a compute encoder, sets the segment's pipeline,
    /// and `executeCommandsInBuffer`'s the segment's ICB range; for
    /// GEMM steps it ends the encoder and encodes the MPS GEMM
    /// directly into the command buffer. Adjacent ICB steps reuse
    /// the same encoder; an intervening GEMM forces an encoder
    /// boundary on each side.
    ///
    /// The caller is responsible for staging all `RuntimeBindings`
    /// buffer contents *before* this call, and for committing /
    /// awaiting `cmdbuf` afterwards.
    ///
    /// Per-forward this is the entire hot path on the Metal side —
    /// no allocator interaction, no argument-buffer mutation, no
    /// re-recording.
    pub fn run_bucket(
        &self,
        bucket: usize,
        device: &Device,
        cmdbuf: &CommandBufferRef,
    ) -> Result<(), WorkerError> {
        let baking = &self.bucket_bakings[bucket];
        // ICB descriptor uses `inheritBuffers=false`; the firing
        // encoder must declare every ICB-referenced buffer resident
        // before the dispatch. Pre-build the `&[&ResourceRef]` slice
        // once per call (no allocation amortized — pointer-sized refs
        // into an existing `Vec<Buffer>`).
        let resource_refs: Vec<&ResourceRef> = baking
            .baked_resources
            .iter()
            .map(|b| {
                // `&Buffer → &BufferRef → &ResourceRef` via the
                // foreign_types deref chain. Spelled out here because
                // the closure return type isn't pinned by `Vec::iter()`
                // alone.
                let r: &ResourceRef = b;
                r
            })
            .collect();
        // We open compute encoders lazily so a leading-GEMM bucket
        // doesn't open an empty one. `current` is `Some` only while
        // an encoder is live — every Gemm step ends it, every Icb
        // step opens it on demand.
        let mut current: Option<&ComputeCommandEncoderRef> = None;
        // Storage for the live encoder. `metal-rs` returns a
        // `&ComputeCommandEncoderRef` borrowed from the cmd buffer;
        // there's no owned wrapper, so we keep the active one in a
        // local `Option` and re-fetch via `cmdbuf.new_compute_command_encoder()`
        // when we need a new one.
        // Track the previous ICB step's resources so we can insert a
        // memory barrier between dependent ICB commands inside the
        // same compute encoder. ICB's `concurrentDispatchThreadgroups`
        // (the only ICB dispatch primitive) makes consecutive
        // `executeCommandsInBuffer` calls concurrent — without this
        // barrier a read-after-write hazard between e.g. RopeAppend
        // (writes KV cache) and AttentionViaCache (reads it) races and
        // produces 100x slowdown / wrong data.
        //
        // The FUF/lowered tape carries the exact slot + runtime
        // bindings each command touches, so we insert the barrier
        // unconditionally between consecutive ICB steps with the
        // PREVIOUS step's resources as the barrier's resource set.
        // (The next step's reads will be ordered after those writes.)
        // Per `memoryBarrierWithResources:` semantics, only the listed
        // resources are synchronized — much cheaper than ending the
        // encoder.
        let mut prev_step_resources: Option<Vec<&Buffer>> = None;
        for step in &baking.steps {
            match step {
                BucketStep::Icb { pipeline, range, step_resources, .. } => {
                    // End the encoder between every ICB step. Within
                    // one compute encoder Apple's `concurrentDispatchThreadgroups`
                    // (the only ICB dispatch primitive) makes
                    // consecutive `executeCommandsInBuffer` calls
                    // concurrent — and `memoryBarrierWithResources:`
                    // empirically does NOT suffice to serialize
                    // consecutive ICB execs. Cross-encoder ordering
                    // is enforced by Apple's command queue, so each
                    // ICB step becoming its own encoder gives us
                    // correct RAW dependencies. Cost is ~us per
                    // encoder boundary.
                    if let Some(enc) = current.take() {
                        enc.end_encoding();
                    }
                    let _ = step_resources; // reserved for finer-grained barrier later
                    let _ = &prev_step_resources;
                    let enc = cmdbuf.new_compute_command_encoder();
                    if std::env::var("FERRITE_METAL_FORCE_USE_RESOURCES").is_ok()
                        && !resource_refs.is_empty()
                    {
                        enc.use_resources(
                            &resource_refs,
                            MTLResourceUsage::Read | MTLResourceUsage::Write,
                        );
                    }
                    enc.set_compute_pipeline_state(pipeline);
                    baking.icb.execute_on_encoder(enc, range.clone());
                    current = Some(enc);
                }
                BucketStep::Gemm { a, b, c, m, n, k } => {
                    if let Some(enc) = current.take() {
                        enc.end_encoding();
                    }
                    // Encoder boundary clears the prev-resources tracking;
                    // ordering across encoders is provided by Apple's
                    // command queue.
                    prev_step_resources = None;
                    encode_gemm_into_command_buffer(
                        device, cmdbuf, &a.buffer, &b.buffer, &c.buffer, *m, *n, *k, 1.0, 0.0,
                        false, true, true,
                    )
                    .map_err(WorkerError::GemmEncode)?;
                }
            }
        }
        if let Some(enc) = current.take() {
            enc.end_encoding();
        }
        Ok(())
    }

    /// Debug-only mirror of [`Self::run_bucket`] that commits + waits
    /// after EACH step, with eprintln tracing. Used to bisect a GPU
    /// hang to a specific kernel dispatch. Gated behind
    /// `FERRITE_METAL_STEP_DEBUG=1`. Each step gets its own command
    /// buffer; a hang on step N blocks `wait_until_completed` for that
    /// cmdbuf and the trace pinpoints the failing kernel by index.
    pub fn run_bucket_per_step_debug(
        &self,
        bucket: usize,
        device: &Device,
        queue: &ferrite_metal_kernels::metal::CommandQueue,
    ) -> Result<(), WorkerError> {
        use ferrite_metal_kernels::metal::MTLCommandBufferStatus;
        let baking = &self.bucket_bakings[bucket];
        // Bucket-wide resource pool (used as fallback / for diagnostic
        // only). The per-step path now uses each step's own
        // `step_resources` subset for its `useResources` call.
        let _bucket_resource_refs: Vec<&ResourceRef> = baking
            .baked_resources
            .iter()
            .map(|b| {
                let r: &ResourceRef = b;
                r
            })
            .collect();
        // DIAGNOSTIC: stamp marker bytes into every arena slot at the
        // start of the bucket. If the kernels write to these slots,
        // the marker is overwritten. If we still see the marker after
        // step N, kernel N didn't write to that slot (or wrote to a
        // different buffer).
        if std::env::var_os("VLLM_STAMP_ARENA").is_some() {
            for (i, buf) in self.arena.iter().enumerate() {
                let len_bytes = buf.length() as usize;
                unsafe {
                    let p = buf.contents() as *mut u8;
                    for off in 0..len_bytes {
                        *p.add(off) = 0xAA; // marker
                    }
                }
                let _ = i;
            }
            eprintln!("[stamp] stamped 0xAA across {} arena slots", self.arena.len());
        }

        let bucket_start = std::time::Instant::now();
        for (idx, step) in baking.steps.iter().enumerate() {
            let step_start = std::time::Instant::now();
            let cb = queue.new_command_buffer();
            let kind: String;
            match step {
                BucketStep::Icb { pipeline, range, kernel, step_resources, .. } => {
                    kind = format!("Icb range={:?} kernel={:?}", range, kernel);
                    let _ = pipeline; // pipeline.label() can't be safely formatted (NSString may be nil)
                    let _ = step_resources; // see note below
                    let enc = cb.new_compute_command_encoder();
                    // SKIP useResources by default in the per-step
                    // path. Empirically on Apple Silicon (M-series, 16GB
                    // unified memory), `useResources` does eager
                    // first-touch / residency commit work that costs
                    // ~500ms per fresh buffer. With 6 distinct KV-cache
                    // / runtime buffers per AttentionViaCache call,
                    // that's ~3s/dispatch. Skipping the call relies on
                    // Apple's automatic on-demand paging through
                    // setBuffer (the buffers were already touched
                    // earlier in the forward by rope_append / write_runtime_inputs
                    // / Q-projection Gemm), which costs ~200us total.
                    //
                    // Set FERRITE_METAL_FORCE_USE_RESOURCES=1 to
                    // re-enable the original behavior for debugging.
                    if std::env::var("FERRITE_METAL_FORCE_USE_RESOURCES").is_ok() {
                        let step_refs: Vec<&ResourceRef> =
                            step_resources.iter().map(|b| b as &ResourceRef).collect();
                        if !step_refs.is_empty() {
                            enc.use_resources(
                                &step_refs,
                                MTLResourceUsage::Read | MTLResourceUsage::Write,
                            );
                        }
                    }
                    enc.set_compute_pipeline_state(pipeline);
                    baking.icb.execute_on_encoder(enc, range.clone());
                    enc.end_encoding();
                }
                BucketStep::Gemm { a, b, c, m, n, k } => {
                    kind = format!("Gemm m={m} n={n} k={k}");
                    encode_gemm_into_command_buffer(
                        device, cb, &a.buffer, &b.buffer, &c.buffer, *m, *n, *k, 1.0, 0.0, false,
                        true, true,
                    )
                    .map_err(WorkerError::GemmEncode)?;
                }
            }
            let encoded_at = step_start.elapsed();
            cb.commit();
            cb.wait_until_completed();
            let status = cb.status();
            let total = step_start.elapsed();
            let gpu_us = total.as_micros().saturating_sub(encoded_at.as_micros());
            // DIAGNOSTIC: dump non-zero / marker counts + hex bytes
            // of slot start. Find which step actually writes, and
            // what values it writes.
            let arena_summary = if std::env::var_os("VLLM_DUMP_ARENA_PER_STEP").is_some() {
                let mut s = String::new();
                for (i, buf) in self.arena.iter().enumerate() {
                    let len_bytes = buf.length() as usize;
                    let row0 = unsafe {
                        std::slice::from_raw_parts(
                            buf.contents() as *const u8,
                            len_bytes,
                        )
                    };
                    let nz = row0.iter().filter(|&&v| v != 0).count();
                    let marker = row0.iter().filter(|&&v| v == 0xAA).count();
                    // First 8 bytes hex.
                    let head: Vec<String> =
                        row0.iter().take(8).map(|b| format!("{:02x}", b)).collect();
                    s.push_str(&format!(
                        " s{i}=nz{nz}/marker{marker}/{len_bytes}/[{}]",
                        head.join(""),
                    ));
                }
                s
            } else {
                String::new()
            };
            eprintln!(
                "[step {idx}] {kind} encode={}us gpu={}us total={}us status={:?}{arena_summary}",
                encoded_at.as_micros(),
                gpu_us,
                total.as_micros(),
                status,
            );
            if status != MTLCommandBufferStatus::Completed {
                return Err(WorkerError::WeightLookupFailed {
                    reason: "per-step commit failed (see eprintln above)",
                });
            }
        }
        eprintln!(
            "[bucket {} steps={}] total={}ms",
            baking.bucket_m,
            baking.steps.len(),
            bucket_start.elapsed().as_millis(),
        );
        Ok(())
    }

    /// Production fast variant of [`Self::run_bucket_per_step_debug`]:
    /// commits + waits per step (no eprintln), no diagnostic state.
    /// Each `BucketStep` runs in its own command buffer, which on
    /// Apple Silicon empirically gives ~10x faster wall-clock
    /// throughput than the batched single-cmdbuf `run_bucket` path.
    /// Gated behind `FERRITE_METAL_PER_STEP_CMDBUF=1`.
    ///
    /// When `FERRITE_METAL_DIRECT_DISPATCH=1` is also set, ICB
    /// steps bypass the ICB entirely and dispatch via setBuffer +
    /// dispatchThreadgroups directly (matching the golden test's
    /// pattern). Diagnostic for whether ICB execution is the
    /// AttentionViaCache bottleneck.
    pub fn run_bucket_per_step_silent(
        &self,
        bucket: usize,
        device: &Device,
        queue: &ferrite_metal_kernels::metal::CommandQueue,
    ) -> Result<(), WorkerError> {
        use ferrite_metal_kernels::metal::MTLCommandBufferStatus;
        let baking = &self.bucket_bakings[bucket];
        let direct = std::env::var_os("FERRITE_METAL_DIRECT_DISPATCH").is_some();

        // Stamp marker into arena (same as per-step debug) for the
        // diagnostic dump.
        if std::env::var_os("VLLM_STAMP_ARENA").is_some() {
            for buf in self.arena.iter() {
                let len_bytes = buf.length() as usize;
                unsafe {
                    let p = buf.contents() as *mut u8;
                    for off in 0..len_bytes {
                        *p.add(off) = 0xAA;
                    }
                }
            }
        }

        let dump_per_step = std::env::var_os("VLLM_DUMP_ARENA_PER_STEP").is_some();
        for (idx, step) in baking.steps.iter().enumerate() {
            let cb = queue.new_command_buffer();
            match step {
                BucketStep::Icb {
                    pipeline,
                    range,
                    direct_bindings,
                    direct_dispatch,
                    ..
                } => {
                    let enc = cb.new_compute_command_encoder();
                    enc.set_compute_pipeline_state(pipeline);
                    if direct {
                        // Direct dispatch path — bypasses the ICB.
                        // For each command in the range, bind buffers
                        // explicitly and dispatch.
                        for (bindings, (tg, tpt)) in
                            direct_bindings.iter().zip(direct_dispatch.iter())
                        {
                            for (buffer, offset, index) in bindings {
                                enc.set_buffer(*index, Some(buffer), *offset);
                            }
                            enc.dispatch_thread_groups(*tg, *tpt);
                        }
                    } else {
                        baking.icb.execute_on_encoder(enc, range.clone());
                    }
                    enc.end_encoding();
                }
                BucketStep::Gemm { a, b, c, m, n, k } => {
                    encode_gemm_into_command_buffer(
                        device, cb, &a.buffer, &b.buffer, &c.buffer, *m, *n, *k, 1.0, 0.0, false,
                        true, true,
                    )
                    .map_err(WorkerError::GemmEncode)?;
                }
            }
            cb.commit();
            cb.wait_until_completed();
            if cb.status() != MTLCommandBufferStatus::Completed {
                return Err(WorkerError::WeightLookupFailed {
                    reason: "per-step commit failed",
                });
            }
            // DIAGNOSTIC: per-step arena dump (works in silent/direct
            // path too). Dumps non-zero / marker counts plus the first
            // 4 fp16 values of every slot, so a bisection across layers
            // can identify which step first produces all-zero output in
            // a previously-non-zero slot.
            if dump_per_step {
                let kind = match step {
                    BucketStep::Icb { kernel, range, .. } => {
                        format!("Icb {:?} {:?}", kernel, range)
                    }
                    BucketStep::Gemm { m, n, k, .. } => format!("Gemm m={m} n={n} k={k}"),
                };
                // Per-slot row stride in bytes (TinyLlama specific —
                // good enough for the bisection harness; mismatches
                // just print a different "sample" rather than crashing).
                // s0/s1/s2: hidden_size=2048 fp16 → 4096 B/row.
                // s3/s4: head_dim*num_kv_heads=256 fp16 → 512 B/row.
                // s5: intermediate_size=5632 fp16 → 11264 B/row.
                // s6: vocab=32000 fp16 → 64000 B/row.
                let row_strides: [usize; 7] = [4096, 4096, 4096, 512, 512, 11264, 64000];
                let mut s = String::new();
                for (i, buf) in self.arena.iter().enumerate() {
                    let len_bytes = buf.length() as usize;
                    let bytes = unsafe {
                        std::slice::from_raw_parts(
                            buf.contents() as *const u8,
                            len_bytes,
                        )
                    };
                    let nz = bytes.iter().filter(|&&v| v != 0).count();
                    let marker = bytes.iter().filter(|&&v| v == 0xAA).count();
                    let stride = row_strides.get(i).copied().unwrap_or(4096);
                    // 4 fp16 values at row 0 and row 22, side by side.
                    // Letting the reader see if rows 0 and 22 differ
                    // exposes "all rows identical" bugs that a single-
                    // row dump misses.
                    let head_at = |row: usize| -> String {
                        let base = row * stride;
                        let parts: Vec<String> = (0..4)
                            .map(|j| {
                                let off = base + j * 2;
                                if off + 1 >= len_bytes {
                                    String::from("nan")
                                } else {
                                    let bits =
                                        u16::from_le_bytes([bytes[off], bytes[off + 1]]);
                                    format!("{:.3}", f16_bits_to_f32(bits))
                                }
                            })
                            .collect();
                        parts.join(",")
                    };
                    s.push_str(&format!(
                        " s{i}=nz{nz}/m{marker}/r0[{}]/r22[{}]",
                        head_at(0),
                        head_at(22),
                    ));
                }
                eprintln!("[silent step={idx}] {kind}{s}");
            }
        }
        Ok(())
    }
}

/// IEEE-754 binary16 → binary32 decoder for the per-step diagnostic
/// dump. `half` lives in `[dev-dependencies]` only; pulling it into
/// the regular dep set just to print three decimal digits per slot
/// is overkill, so this open-codes the conversion.
fn f16_bits_to_f32(bits: u16) -> f32 {
    let sign = ((bits >> 15) & 1) as u32;
    let exp = ((bits >> 10) & 0x1f) as u32;
    let mant = (bits & 0x3ff) as u32;
    let f32_bits = if exp == 0 {
        if mant == 0 {
            sign << 31
        } else {
            let mut e: i32 = -14;
            let mut m = mant;
            while (m & 0x400) == 0 {
                m <<= 1;
                e -= 1;
            }
            m &= 0x3ff;
            (sign << 31) | (((e + 127) as u32) << 23) | (m << 13)
        }
    } else if exp == 0x1f {
        (sign << 31) | (0xff << 23) | (mant << 13)
    } else {
        (sign << 31) | (((exp as i32 - 15 + 127) as u32) << 23) | (mant << 13)
    };
    f32::from_bits(f32_bits)
}

/// Bake one bucket's ICB + execution plan.
#[allow(clippy::too_many_arguments)]
fn bake_bucket<W: CanonicalParams>(
    bucket_index: usize,
    tape: &LoweredMetalTape<W>,
    arena: &[Buffer],
    pipelines: &SpecializedPipelines,
    weights: &W,
    allocator: &MetalAllocator,
    runtime: &RuntimeBindings,
    device: Arc<Device>,
) -> Result<BucketBaking, WorkerError> {
    if tape.num_arena_slots as usize != arena.len() {
        return Err(WorkerError::ArenaShapeMismatch {
            expected: tape.num_arena_slots,
            actual: arena.len(),
        });
    }

    // ICB capacity sized for the worst case (one ICB slot per
    // command); GEMM steps don't consume slots but the slack is
    // cheap (a few bytes per slot) and keeps the index = command
    // index invariant.
    let mut ctx = RecordingContext::new(device, tape.commands.len().max(1))
        .map_err(WorkerError::Recording)?;
    let mut steps: Vec<BucketStep> = Vec::new();
    // Unique buffers referenced by ICB commands. Used at firing time
    // to satisfy the `inheritBuffers=false` residency contract via
    // `useResources`. Identity is by raw `metal::Buffer` pointer —
    // the ICB doesn't care about Rust ownership, only the GPU handle.
    let mut baked_resources: Vec<Buffer> = Vec::new();
    let mut baked_seen: Vec<*const _> = Vec::new();
    let record_resource = |buf: &Buffer, seen: &mut Vec<*const _>, out: &mut Vec<Buffer>| {
        let ptr = buf.as_ptr() as *const _;
        if !seen.contains(&ptr) {
            seen.push(ptr);
            out.push(buf.clone());
        }
    };

    for (cmd_idx, cmd) in tape.commands.iter().enumerate() {
        if matches!(cmd.kernel, KernelId::Gemm) {
            let dims = cmd.gemm_dims.ok_or(WorkerError::MissingGemmDims {
                bucket_index,
                command_index: cmd_idx,
            })?;
            let (a, b, c) = resolve_gemm_buffers(
                bucket_index,
                cmd_idx,
                cmd,
                arena,
                weights,
                allocator,
                runtime,
            )?;
            // Mark every GEMM-touched buffer resident on subsequent
            // ICB encoders even though MPS' own encoder doesn't fire
            // through our ICB. The surrounding `inheritBuffers=false`
            // ICB encoders re-issue `useResources(baked_resources, …)`
            // each time they open; a buffer that is read by a post-
            // GEMM ICB segment but never bound into an ICB segment's
            // own bindings (e.g. the lm_head's output that the host
            // reads after commit) would otherwise miss the residency
            // contract on every encoder it touches. Cheap dedupe via
            // raw pointer identity.
            record_resource(&a.buffer, &mut baked_seen, &mut baked_resources);
            record_resource(&b.buffer, &mut baked_seen, &mut baked_resources);
            record_resource(&c.buffer, &mut baked_seen, &mut baked_resources);
            steps.push(BucketStep::Gemm {
                a,
                b,
                c,
                m: dims.m,
                n: dims.n,
                k: dims.k,
            });
            continue;
        }

        // Every per-layer scalar (eps, attn_scale, paging strides)
        // is a `CanonicalParams` constant the macro emitted from the
        // model config — no runtime extras to thread.
        let pipeline = pipelines
            .pipeline_for::<W>(cmd.kernel, tape.bucket_m)
            .map_err(WorkerError::PipelineLookup)?;

        let bound = resolve_bindings(
            bucket_index,
            cmd_idx,
            cmd,
            arena,
            weights,
            allocator,
            runtime,
        )?;
        let bound_refs: Vec<(&Buffer, u64, u64)> =
            bound.iter().map(|(b, off, idx)| (b, *off, *idx)).collect();
        for (b, _, _) in &bound_refs {
            record_resource(*b, &mut baked_seen, &mut baked_resources);
        }
        let (tg, tpt) = mtl_size_pair(cmd);
        ctx.record_compute_dispatch(&pipeline, &bound_refs, tg, tpt);

        // Coalesce with the previous step iff (a) it's an ICB step
        // (a Gemm step forces an encoder boundary) and (b) its
        // pipeline shares the underlying ObjC pointer (specialized
        // pipelines are refcounted — same key returns same handle
        // from the cache).
        let recorded_at = cmd_idx;
        let step_resources_for_cmd: Vec<Buffer> =
            bound_refs.iter().map(|(b, _, _)| (*b).clone()).collect();
        let bindings_for_cmd: Vec<(Buffer, u64, u64)> = bound_refs
            .iter()
            .map(|(b, off, idx)| ((*b).clone(), *off, *idx))
            .collect();
        let dispatch_for_cmd = (
            MTLSize::new(tg.width, tg.height, tg.depth),
            MTLSize::new(tpt.width, tpt.height, tpt.depth),
        );
        if std::env::var_os("FERRITE_METAL_BAKE_DEBUG").is_some() {
            let bind_summary: Vec<String> = bindings_for_cmd
                .iter()
                .map(|(b, off, idx)| {
                    format!(
                        "(buf=0x{:x},len={},off={},idx={})",
                        b.as_ptr() as usize,
                        b.length(),
                        off,
                        idx,
                    )
                })
                .collect();
            eprintln!(
                "[bake bucket={} cmd={}] kernel={:?} tg=({},{},{}) tpt=({},{},{}) bindings=[{}]",
                bucket_index,
                cmd_idx,
                cmd.kernel,
                tg.width,
                tg.height,
                tg.depth,
                tpt.width,
                tpt.height,
                tpt.depth,
                bind_summary.join(","),
            );
        }
        match steps.last_mut() {
            Some(BucketStep::Icb {
                pipeline: prev,
                range,
                step_resources,
                direct_bindings,
                direct_dispatch,
                ..
            }) if same_pipeline(prev, &pipeline) => {
                range.end = recorded_at + 1;
                // Coalesced ICB range — extend its resource set + per-command bindings.
                let mut seen: Vec<*const _> =
                    step_resources.iter().map(|b| b.as_ptr() as *const _).collect();
                for buf in &step_resources_for_cmd {
                    let p = buf.as_ptr() as *const _;
                    if !seen.contains(&p) {
                        seen.push(p);
                        step_resources.push(buf.clone());
                    }
                }
                direct_bindings.push(bindings_for_cmd);
                direct_dispatch.push(dispatch_for_cmd);
            }
            _ => {
                steps.push(BucketStep::Icb {
                    pipeline,
                    range: recorded_at..(recorded_at + 1),
                    kernel: cmd.kernel,
                    step_resources: step_resources_for_cmd,
                    direct_bindings: vec![bindings_for_cmd],
                    direct_dispatch: vec![dispatch_for_cmd],
                });
            }
        }
    }

    Ok(BucketBaking {
        bucket_m: tape.bucket_m,
        icb: ctx,
        steps,
        baked_resources,
    })
}

/// Resolve `(out, in, weight)` buffers for a `KernelId::Gemm` command.
///
/// The lowering pass guarantees the binding order: index 0 → output
/// arena slot, index 1 → input arena slot, index 2 → LinearLayer
/// weight thunk. Anything else is a contract violation surfaced as
/// [`WorkerError::GemmBindingsMalformed`].
fn resolve_gemm_buffers<W: CanonicalParams>(
    bucket_index: usize,
    command_index: usize,
    cmd: &LoweredCommand<W>,
    arena: &[Buffer],
    weights: &W,
    allocator: &MetalAllocator,
    runtime: &RuntimeBindings,
) -> Result<(BoundBuffer, BoundBuffer, BoundBuffer), WorkerError> {
    let bound = resolve_bindings(
        bucket_index,
        command_index,
        cmd,
        arena,
        weights,
        allocator,
        runtime,
    )?;
    if bound.len() != 3 {
        return Err(WorkerError::GemmBindingsMalformed {
            bucket_index,
            command_index,
            reason: "expected exactly 3 bindings (out, in, weight)",
        });
    }
    // Bindings are produced in the order the lowering pass listed
    // them; their `binding_index` field carries the encoder slot but
    // we only care about positional ordering. The lowering pass uses
    // 0 = out, 1 = in, 2 = weight.
    let mut iter = bound.into_iter();
    let out = iter.next().expect("bound[0]");
    let inp = iter.next().expect("bound[1]");
    let wt = iter.next().expect("bound[2]");
    Ok((
        BoundBuffer {
            buffer: inp.0,
            offset: inp.1,
        },
        BoundBuffer {
            buffer: wt.0,
            offset: wt.1,
        },
        BoundBuffer {
            buffer: out.0,
            offset: out.1,
        },
    ))
}

/// Resolve a `Binding::Weight` against the loaded model `weights`
/// and the allocator that owns the underlying `MTLBuffer` arenas.
///
/// Calls the typed `WtFn` thunk in `kind` to get a reference to the
/// layer struct (`&RmsNorm`, `&LinearLayer`, `&Embedding`), pulls
/// out the raw GpuTensor pointer matching `which`, and asks the
/// allocator which buffer + offset that pointer belongs to.
///
/// Same shape CUDA's interpreter uses: WtFn → layer struct →
/// `GpuTensor`. The Metal-side delta is just the final pointer →
/// `(&Buffer, offset)` reverse lookup against the arena allocator.
fn resolve_weight<W: CanonicalParams>(
    weights: &W,
    allocator: &MetalAllocator,
    kind: &WeightBundleKind<W>,
    layer: u32,
    which: WeightTensor,
) -> Result<(Buffer, u64), WorkerError> {
    let tensor = match kind {
        WeightBundleKind::RmsNorm(wtfn) => (wtfn)(weights, layer).weight,
        WeightBundleKind::Embedding(wtfn) => (wtfn)(weights, layer).weight,
        WeightBundleKind::LinearLayer(wtfn) => {
            let l = (wtfn)(weights, layer);
            match which {
                WeightTensor::Weight => l.dense_weight(),
                WeightTensor::Bias => l.dense_bias().ok_or(WorkerError::WeightLookupFailed {
                    reason: "LinearLayer bias requested but not present",
                })?,
            }
        }
        WeightBundleKind::CosSin(cosfn) => (cosfn)(weights, layer),
    };
    allocator
        .buffer_for(tensor.raw_ptr())
        .ok_or(WorkerError::WeightLookupFailed {
            reason: "weight pointer not in any MetalAllocator arena \
                     — was it loaded through this allocator?",
        })
}

/// Resolve every binding on `cmd` to (buffer, offset, binding-index).
///
/// Returns owned `Buffer` clones (cheap ObjC refcount) so callers
/// don't have to thread the [`MetalAllocator`]'s arenas-`Mutex` lock
/// guard through to the encoder.
fn resolve_bindings<W: CanonicalParams>(
    bucket_index: usize,
    command_index: usize,
    cmd: &LoweredCommand<W>,
    arena: &[Buffer],
    weights: &W,
    allocator: &MetalAllocator,
    runtime: &RuntimeBindings,
) -> Result<Vec<(Buffer, u64, u64)>, WorkerError> {
    let mut out: Vec<(Buffer, u64, u64)> = Vec::with_capacity(cmd.bindings.len());
    for binding in &cmd.bindings {
        let (buf, off, idx) = match binding {
            Binding::ArenaSlot {
                slot,
                binding_index,
            } => {
                let s = *slot as usize;
                if s >= arena.len() {
                    return Err(WorkerError::ArenaSlotOutOfRange {
                        bucket_index,
                        command_index,
                        slot: *slot,
                        arena_len: arena.len(),
                    });
                }
                (arena[s].clone(), 0u64, *binding_index as u64)
            }
            Binding::Weight {
                kind,
                which,
                layer,
                binding_index,
            } => {
                let (b, off) = resolve_weight(weights, allocator, kind, *layer, *which)?;
                (b, off, *binding_index as u64)
            }
            Binding::Runtime {
                kind,
                binding_index,
            } => (
                runtime.buffer_for(*kind).clone(),
                0u64,
                *binding_index as u64,
            ),
        };
        out.push((buf, off, idx));
    }
    Ok(out)
}

fn mtl_size_pair<W: CanonicalParams>(cmd: &LoweredCommand<W>) -> (MTLSize, MTLSize) {
    let tg = MTLSize {
        width: cmd.dispatch.threadgroups.0 as u64,
        height: cmd.dispatch.threadgroups.1 as u64,
        depth: cmd.dispatch.threadgroups.2 as u64,
    };
    let tpt = MTLSize {
        width: cmd.dispatch.threads_per_threadgroup.0 as u64,
        height: cmd.dispatch.threads_per_threadgroup.1 as u64,
        depth: cmd.dispatch.threads_per_threadgroup.2 as u64,
    };
    (tg, tpt)
}

/// Compare two `ComputePipelineState`s by ObjC handle. Cached
/// pipelines for the same `(kernel, bucket, extras)` tuple are
/// pointer-equal, so this is the right test for segment coalescing.
fn same_pipeline(a: &ComputePipelineState, b: &ComputePipelineState) -> bool {
    std::ptr::eq(a.as_ptr() as *const _, b.as_ptr() as *const _)
}

#[cfg(all(test, target_os = "macos"))]
mod tests {
    use super::*;
    use crate::CanonicalParams;
    use crate::interpreter::metal::lowered::{
        Binding, DispatchShape, LoweredCommand, RuntimeBindingKind, WeightBundleKind, WeightTensor,
    };
    use ferrite_cuda_core::{DType, DeviceAllocator, GpuTensor};
    use ferrite_kernels::layers::{Embedding, Linear, LinearLayer, RmsNorm};
    use ferrite_metal_kernels::specialized_pipeline_cache::SpecializedPipelineCache;
    use std::sync::Arc;

    /// Test fixture: holds `CanonicalParams` constants AND the layer
    /// instances the WtFn thunks below dereference. Plays the role
    /// of the per-canonical `Weights` struct the macro will emit.
    struct TestWeights {
        rmsnorm_layer: RmsNorm,
        linear_layer: LinearLayer,
        embedding_layer: Embedding,
    }
    impl CanonicalParams for TestWeights {
        const HEAD_DIM: u32 = 64;
        const NUM_Q_HEADS: u32 = 32;
        const NUM_KV_HEADS: u32 = 4;
        const Q_SIZE: usize = 2048;
        const KV_SIZE: usize = 256;
        const INTERMEDIATE_SIZE: usize = 5632;
        const ATTN_SCALE: f32 = 0.125;
        const ATTN_SOFTCAP: f32 = 0.0;
        const SLIDING_WINDOW: i32 = -1;
        const KV_LORA_RANK: usize = 0;
        const QK_NOPE_HEAD_DIM: usize = 0;
        const QK_ROPE_HEAD_DIM: usize = 0;
        const V_HEAD_DIM: usize = 0;
        const FINAL_LOGIT_SOFTCAPPING: f32 = 0.0;
        const QK_HEAD_DIM: usize = 0;
        const MLA_ATTN_SCALE: f32 = 0.0;
    }

    // WtFn thunks for the three layer kinds the smoke tests use.
    // They look layers up by name on `TestWeights` directly — same
    // pattern the macro will emit for real canonical Weights.
    fn rmsnorm_thunk(w: &TestWeights, _layer: u32) -> &RmsNorm {
        &w.rmsnorm_layer
    }
    fn linear_thunk(w: &TestWeights, _layer: u32) -> &LinearLayer {
        &w.linear_layer
    }
    fn _embedding_thunk(w: &TestWeights, _layer: u32) -> &Embedding {
        &w.embedding_layer
    }

    /// Build a `TestWeights` + the `MetalAllocator` that owns its
    /// MTLBuffer arenas. The allocator is also used by the pool to
    /// resolve `Binding::Weight` lookups; tests that build a worker
    /// directly thread the same allocator into `MetalWorker::new`.
    ///
    /// Allocations are zero-filled — sufficient for verifying the
    /// recording flow. Numerical correctness lives in `pipelines.rs`'s
    /// `*_matches_cpu_golden` tests.
    fn build_test_weights() -> (Arc<TestWeights>, Arc<MetalAllocator>) {
        let device = ferrite_metal_kernels::detect_device()
            .expect("test fixture: a Metal device")
            .device
            .clone();
        let mut allocator = MetalAllocator::new(device);

        // RmsNorm weight: [Q_SIZE] f16 = 4 KB.
        let rmsnorm_bytes = vec![0u8; TestWeights::Q_SIZE * 2];
        let rmsnorm_ptr = unsafe {
            allocator
                .alloc_and_copy_host(rmsnorm_bytes.as_ptr(), rmsnorm_bytes.len())
                .expect("rmsnorm tensor")
        };
        let rmsnorm_tensor =
            unsafe { GpuTensor::new(rmsnorm_ptr, &[TestWeights::Q_SIZE], DType::F16) };

        // LinearLayer weight: [Q_SIZE, Q_SIZE] f16 = ~8 MB. Sized to
        // the largest TinyLlama-class projection so the GEMM dim
        // checks inside `encode_gemm_into_command_buffer` pass.
        let linear_bytes = vec![0u8; TestWeights::Q_SIZE * TestWeights::Q_SIZE * 2];
        let linear_ptr = unsafe {
            allocator
                .alloc_and_copy_host(linear_bytes.as_ptr(), linear_bytes.len())
                .expect("linear tensor")
        };
        let linear_tensor = unsafe {
            GpuTensor::new(
                linear_ptr,
                &[TestWeights::Q_SIZE, TestWeights::Q_SIZE],
                DType::F16,
            )
        };

        // Embedding weight: [vocab=128, hidden=Q_SIZE] f16. Tiny vocab
        // — embedding tests don't exercise vocab-size correctness.
        let embedding_bytes = vec![0u8; 128 * TestWeights::Q_SIZE * 2];
        let embedding_ptr = unsafe {
            allocator
                .alloc_and_copy_host(embedding_bytes.as_ptr(), embedding_bytes.len())
                .expect("embedding tensor")
        };
        let embedding_tensor =
            unsafe { GpuTensor::new(embedding_ptr, &[128, TestWeights::Q_SIZE], DType::F16) };

        let weights = Arc::new(TestWeights {
            rmsnorm_layer: RmsNorm::new(rmsnorm_tensor, 1e-5),
            linear_layer: LinearLayer::Dense(Linear::new(linear_tensor, None)),
            embedding_layer: Embedding::new(embedding_tensor),
        });
        (weights, Arc::new(allocator))
    }

    fn alloc_buffer(device: &Device, bytes: u64) -> Buffer {
        device.new_buffer(bytes.max(1), MTLResourceOptions::StorageModeShared)
    }

    fn empty_runtime(device: &Device, num_layers: usize) -> RuntimeBindings {
        RuntimeBindings {
            input_ids: alloc_buffer(device, 16),
            positions: alloc_buffer(device, 16),
            slot_mapping: alloc_buffer(device, 16),
            cu_seqlens_q: alloc_buffer(device, 16),
            seq_used_k: alloc_buffer(device, 16),
            block_table: alloc_buffer(device, 16),
            kv_cache_k: (0..num_layers).map(|_| alloc_buffer(device, 16)).collect(),
            kv_cache_v: (0..num_layers).map(|_| alloc_buffer(device, 16)).collect(),
        }
    }

    /// Build a synthetic 2-bucket lowered tape: each bucket runs the
    /// same `[RmsNorm, FusedAddRmsNorm]` shape twice. Verifies:
    /// arena allocation, ICB recording per command, segment
    /// coalescing across same-pipeline neighbours.
    fn build_synthetic_tape(bucket_m: u32) -> LoweredMetalTape<TestWeights> {
        let rmsnorm = LoweredCommand {
            kernel: KernelId::RmsNorm,
            dispatch: DispatchShape {
                threadgroups: (bucket_m, 1, 1),
                threads_per_threadgroup: (256, 1, 1),
            },
            bindings: vec![
                Binding::ArenaSlot {
                    slot: 0,
                    binding_index: 0,
                },
                Binding::ArenaSlot {
                    slot: 1,
                    binding_index: 1,
                },
                Binding::Weight {
                    kind: WeightBundleKind::RmsNorm(rmsnorm_thunk),
                    which: WeightTensor::Weight,
                    layer: 0,
                    binding_index: 2,
                },
            ],
            gemm_dims: None,
        };
        let fused_add_rmsnorm = LoweredCommand {
            kernel: KernelId::FusedAddRmsNorm,
            dispatch: DispatchShape {
                threadgroups: (bucket_m, 1, 1),
                threads_per_threadgroup: (256, 1, 1),
            },
            bindings: vec![
                Binding::ArenaSlot {
                    slot: 0,
                    binding_index: 0,
                },
                Binding::ArenaSlot {
                    slot: 1,
                    binding_index: 1,
                },
                Binding::Weight {
                    kind: WeightBundleKind::RmsNorm(rmsnorm_thunk),
                    which: WeightTensor::Weight,
                    layer: 0,
                    binding_index: 2,
                },
            ],
            gemm_dims: None,
        };
        // Two RmsNorm commands then two FusedAddRmsNorm commands —
        // exercises both kernels and the coalescer's same-pipeline
        // neighbour case (two RmsNorm with identical extras hit the
        // same cached pipeline; same for the FAR pair).
        LoweredMetalTape {
            bucket_m,
            num_arena_slots: 2,
            commands: vec![
                rmsnorm.clone_for_test(),
                rmsnorm,
                fused_add_rmsnorm.clone_for_test(),
                fused_add_rmsnorm,
            ],
        }
    }

    impl LoweredCommand<TestWeights> {
        // Helper for the test: hand-clone (the public LoweredCommand
        // intentionally does NOT derive Clone so the live tape stays
        // single-owner).
        fn clone_for_test(&self) -> LoweredCommand<TestWeights> {
            LoweredCommand {
                kernel: self.kernel,
                dispatch: self.dispatch,
                bindings: self
                    .bindings
                    .iter()
                    .map(|b| match b {
                        Binding::ArenaSlot {
                            slot,
                            binding_index,
                        } => Binding::ArenaSlot {
                            slot: *slot,
                            binding_index: *binding_index,
                        },
                        Binding::Weight {
                            kind,
                            which,
                            layer,
                            binding_index,
                        } => Binding::Weight {
                            kind: match kind {
                                WeightBundleKind::RmsNorm(f) => WeightBundleKind::RmsNorm(*f),
                                _ => unreachable!("smoke test only uses RmsNorm bundles"),
                            },
                            which: *which,
                            layer: *layer,
                            binding_index: *binding_index,
                        },
                        Binding::Runtime {
                            kind,
                            binding_index,
                        } => Binding::Runtime {
                            kind: *kind,
                            binding_index: *binding_index,
                        },
                    })
                    .collect(),
                gemm_dims: self.gemm_dims,
            }
        }
    }

    /// Helper: extract the `range` from a `BucketStep::Icb`, panic
    /// otherwise. Used by the smoke tests that assert on the
    /// per-bucket plan.
    fn icb_range(step: &BucketStep) -> std::ops::Range<usize> {
        match step {
            BucketStep::Icb { range, .. } => range.clone(),
            BucketStep::Gemm { .. } => panic!("expected ICB step, got Gemm"),
        }
    }

    #[test]
    fn worker_builds_and_segments_coalesce() {
        let Some(device) = ferrite_metal_kernels::detect_device() else {
            eprintln!("skipping: no Metal device");
            return;
        };
        let device = Arc::new(device.device.clone());

        let cache = Arc::new(
            SpecializedPipelineCache::with_standard_shaders((*device).clone())
                .expect("compile standard shaders"),
        );
        let pipelines = SpecializedPipelines::new(cache);

        let (weights, allocator) = build_test_weights();
        let runtime = empty_runtime(&device, 1);

        // Two buckets: M=1 (decode) and M=8 (small prefill).
        let tapes = vec![build_synthetic_tape(1), build_synthetic_tape(8)];
        let arena_layout: ArenaLayout = vec![4 * 1024, 4 * 1024];

        let worker = MetalWorker::<TestWeights>::new(
            device,
            &arena_layout,
            &tapes,
            &pipelines,
            &weights,
            &allocator,
            &runtime,
        )
        .expect("worker builds");

        assert_eq!(worker.arena.len(), 2);
        assert_eq!(worker.bucket_bakings.len(), 2);

        // Each bucket has 4 commands: 2 RmsNorm then 2 FusedAddRmsNorm.
        // Adjacent same-kernel-with-same-extras commands must coalesce
        // into one segment (pipeline-pointer identity); cross-kernel
        // boundary forces a new segment. So: 2 ICB steps per bucket,
        // no GEMM steps.
        for baking in &worker.bucket_bakings {
            assert_eq!(
                baking.steps.len(),
                2,
                "expected RmsNorm + FusedAddRmsNorm coalesced"
            );
            assert_eq!(icb_range(&baking.steps[0]), 0..2);
            assert_eq!(icb_range(&baking.steps[1]), 2..4);
        }
    }

    #[test]
    fn arena_shape_mismatch_errors() {
        let Some(device) = ferrite_metal_kernels::detect_device() else {
            eprintln!("skipping: no Metal device");
            return;
        };
        let device = Arc::new(device.device.clone());

        let cache = Arc::new(
            SpecializedPipelineCache::with_standard_shaders((*device).clone())
                .expect("compile standard shaders"),
        );
        let pipelines = SpecializedPipelines::new(cache);
        let (weights, allocator) = build_test_weights();
        let runtime = empty_runtime(&device, 1);

        let tapes = vec![build_synthetic_tape(1)]; // num_arena_slots = 2

        // Layout has only 1 slot — should error.
        let bad_layout: ArenaLayout = vec![4 * 1024];
        let err = MetalWorker::<TestWeights>::new(
            device,
            &bad_layout,
            &tapes,
            &pipelines,
            &weights,
            &allocator,
            &runtime,
        )
        .err()
        .expect("expected arena shape mismatch error");
        assert!(matches!(
            err,
            WorkerError::ArenaShapeMismatch {
                expected: 2,
                actual: 1
            }
        ));
    }

    /// Phase 5.C.4 smoke test: the worker bakes an AttentionViaCache
    /// command into a one-segment ICB. Verifies that the new
    /// `attention_via_cache_f16_specialized` kernel resolves through
    /// the specialized-pipeline cache and that the worker accepts the
    /// runtime bindings the lowering pass produces (Q + seq_used_k +
    /// block_table + per-layer kv_cache_k/v).
    #[test]
    fn worker_records_attention_via_cache() {
        let Some(device) = ferrite_metal_kernels::detect_device() else {
            eprintln!("skipping: no Metal device");
            return;
        };
        let device = Arc::new(device.device.clone());

        let cache = Arc::new(
            SpecializedPipelineCache::with_standard_shaders((*device).clone())
                .expect("compile standard shaders"),
        );
        let pipelines = SpecializedPipelines::new(cache);

        let (weights, allocator) = build_test_weights();
        let runtime = empty_runtime(&device, 1);

        // Single AttentionViaCache command at decode bucket=1
        // (batch=1, num_q_heads heads).
        let attn = LoweredCommand {
            kernel: KernelId::AttentionViaCache,
            dispatch: DispatchShape {
                threadgroups: (1, TestWeights::NUM_Q_HEADS, 1),
                threads_per_threadgroup: (TestWeights::HEAD_DIM, 1, 1),
            },
            bindings: vec![
                Binding::ArenaSlot {
                    slot: 0,
                    binding_index: 0,
                },
                Binding::ArenaSlot {
                    slot: 1,
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
                    kind: RuntimeBindingKind::KvCacheK { layer: 0 },
                    binding_index: 4,
                },
                Binding::Runtime {
                    kind: RuntimeBindingKind::KvCacheV { layer: 0 },
                    binding_index: 5,
                },
            ],
            gemm_dims: None,
        };
        let tape = LoweredMetalTape {
            bucket_m: 1,
            num_arena_slots: 2,
            commands: vec![attn],
        };

        let worker = MetalWorker::<TestWeights>::new(
            device,
            &vec![1024, 1024],
            &[tape],
            &pipelines,
            &weights,
            &allocator,
            &runtime,
        )
        .expect("worker bakes attention command");

        assert_eq!(worker.bucket_bakings.len(), 1);
        let baking = &worker.bucket_bakings[0];
        assert_eq!(baking.steps.len(), 1);
        assert_eq!(icb_range(&baking.steps[0]), 0..1);
    }

    /// Build a `KernelId::Gemm` lowered command at bucket=`bucket_m`
    /// projecting `[bucket_m, k]` × `[n, k]^T` → `[bucket_m, n]`.
    /// Bindings match the lowering pass: arena slots `(0, 1)` for
    /// (out, in) and a `LinearLayer` weight thunk.
    fn build_gemm_command(bucket_m: u32, n: u32, k: u32) -> LoweredCommand<TestWeights> {
        LoweredCommand {
            kernel: KernelId::Gemm,
            dispatch: DispatchShape {
                threadgroups: (bucket_m.div_ceil(16), n.div_ceil(16), 1),
                threads_per_threadgroup: (16, 16, 1),
            },
            bindings: vec![
                Binding::ArenaSlot {
                    slot: 0,
                    binding_index: 0,
                },
                Binding::ArenaSlot {
                    slot: 1,
                    binding_index: 1,
                },
                Binding::Weight {
                    kind: WeightBundleKind::LinearLayer(linear_thunk),
                    which: WeightTensor::Weight,
                    layer: 0,
                    binding_index: 2,
                },
            ],
            gemm_dims: Some(crate::interpreter::metal::lowered::GemmDims { m: bucket_m, n, k }),
        }
    }

    /// Phase 5.C.5: a tape carrying just one `KernelId::Gemm` command
    /// produces a single `BucketStep::Gemm` with the M/N/K the
    /// lowering pass populated. No ICB step is emitted.
    #[test]
    fn worker_routes_gemm_step() {
        let Some(device) = ferrite_metal_kernels::detect_device() else {
            eprintln!("skipping: no Metal device");
            return;
        };
        let device = Arc::new(device.device.clone());

        let cache = Arc::new(
            SpecializedPipelineCache::with_standard_shaders((*device).clone())
                .expect("compile standard shaders"),
        );
        let pipelines = SpecializedPipelines::new(cache);
        let (weights, allocator) = build_test_weights();
        let runtime = empty_runtime(&device, 1);

        let tape = LoweredMetalTape {
            bucket_m: 1,
            num_arena_slots: 2,
            commands: vec![build_gemm_command(1, 2048, 2048)],
        };

        let worker = MetalWorker::<TestWeights>::new(
            device,
            // Q-size buffers (2048 f16 = 4096 bytes; pad up).
            &vec![64 * 1024, 64 * 1024],
            &[tape],
            &pipelines,
            &weights,
            &allocator,
            &runtime,
        )
        .expect("worker bakes GEMM command");

        assert_eq!(worker.bucket_bakings.len(), 1);
        let baking = &worker.bucket_bakings[0];
        assert_eq!(baking.steps.len(), 1, "one Gemm step, no ICB step");
        match &baking.steps[0] {
            BucketStep::Gemm { m, n, k, .. } => {
                assert_eq!(*m, 1);
                assert_eq!(*n, 2048);
                assert_eq!(*k, 2048);
            }
            BucketStep::Icb { .. } => panic!("expected Gemm step, got Icb"),
        }
    }

    /// Phase 5.C.5: an ICB→Gemm→ICB tape produces three steps. The
    /// Gemm forces an encoder boundary, so the post-Gemm RmsNorm
    /// cannot coalesce with the pre-Gemm RmsNorm even though both
    /// share the same specialized pipeline.
    #[test]
    fn worker_interleaves_gemm_with_icb() {
        let Some(device) = ferrite_metal_kernels::detect_device() else {
            eprintln!("skipping: no Metal device");
            return;
        };
        let device = Arc::new(device.device.clone());

        let cache = Arc::new(
            SpecializedPipelineCache::with_standard_shaders((*device).clone())
                .expect("compile standard shaders"),
        );
        let pipelines = SpecializedPipelines::new(cache);
        let (weights, allocator) = build_test_weights();
        let runtime = empty_runtime(&device, 1);

        let rmsnorm_pre = LoweredCommand {
            kernel: KernelId::RmsNorm,
            dispatch: DispatchShape {
                threadgroups: (1, 1, 1),
                threads_per_threadgroup: (256, 1, 1),
            },
            bindings: vec![
                Binding::ArenaSlot {
                    slot: 0,
                    binding_index: 0,
                },
                Binding::ArenaSlot {
                    slot: 1,
                    binding_index: 1,
                },
                Binding::Weight {
                    kind: WeightBundleKind::RmsNorm(rmsnorm_thunk),
                    which: WeightTensor::Weight,
                    layer: 0,
                    binding_index: 2,
                },
            ],
            gemm_dims: None,
        };
        let rmsnorm_post = LoweredCommand {
            kernel: KernelId::RmsNorm,
            dispatch: rmsnorm_pre.dispatch,
            bindings: rmsnorm_pre
                .bindings
                .iter()
                .map(|b| match b {
                    Binding::ArenaSlot {
                        slot,
                        binding_index,
                    } => Binding::ArenaSlot {
                        slot: *slot,
                        binding_index: *binding_index,
                    },
                    Binding::Weight {
                        kind: WeightBundleKind::RmsNorm(f),
                        which,
                        layer,
                        binding_index,
                    } => Binding::Weight {
                        kind: WeightBundleKind::RmsNorm(*f),
                        which: *which,
                        layer: *layer,
                        binding_index: *binding_index,
                    },
                    _ => unreachable!("rmsnorm_pre uses only ArenaSlot + RmsNorm Weight"),
                })
                .collect(),
            gemm_dims: None,
        };

        let tape = LoweredMetalTape {
            bucket_m: 1,
            num_arena_slots: 2,
            commands: vec![rmsnorm_pre, build_gemm_command(1, 2048, 2048), rmsnorm_post],
        };

        let worker = MetalWorker::<TestWeights>::new(
            device,
            &vec![64 * 1024, 64 * 1024],
            &[tape],
            &pipelines,
            &weights,
            &allocator,
            &runtime,
        )
        .expect("worker bakes mixed tape");

        let baking = &worker.bucket_bakings[0];
        assert_eq!(
            baking.steps.len(),
            3,
            "Icb (rmsnorm_pre) | Gemm | Icb (rmsnorm_post)"
        );
        assert!(matches!(&baking.steps[0], BucketStep::Icb { range, .. } if *range == (0..1)));
        assert!(matches!(
            &baking.steps[1],
            BucketStep::Gemm {
                m: 1,
                n: 2048,
                k: 2048,
                ..
            }
        ));
        // Post-GEMM RmsNorm is at command index 2, not coalesced with
        // the pre-GEMM RmsNorm despite the identical pipeline.
        assert!(matches!(&baking.steps[2], BucketStep::Icb { range, .. } if *range == (2..3)));
    }
}
