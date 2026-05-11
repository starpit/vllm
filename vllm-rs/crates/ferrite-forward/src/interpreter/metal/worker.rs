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

use crate::interpreter::metal::__re::{
    Buffer, CommandBufferRef, ComputePipelineState, Device, MTLBuffer, MTLCommandBuffer,
    MTLCommandEncoder, MTLCommandQueue, MTLComputeCommandEncoder, MTLDevice, MTLResourceOptions,
    MTLSize,
};
use ::objc2::rc::Retained;
use ::objc2::runtime::ProtocolObject;
use ::objc2_metal::{MTLResource, MTLResourceUsage};
use ferrite_metal_kernels::gemm::{GemmDtype, GemmError, encode_gemm_into_command_buffer};
use ferrite_metal_kernels::instruction_executor::RecordingContext;
type ResourceRef = ProtocolObject<dyn MTLResource>;

use super::lowered::{
    Binding, KernelId, LoweredCommand, LoweredMetalTape, MetalDtype, WeightBundleKind, WeightTensor,
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
    /// A command referenced `Binding::Scratch` but the worker has no
    /// SplitK scratch buffer allocated. Indicates a lowering /
    /// `LoweredMetalTape::splitk_scratch_bytes` accounting bug —
    /// either lowering emitted Scratch without registering the
    /// scratch byte count, or the worker discarded the buffer.
    ScratchBufferMissing {
        bucket_index: usize,
        command_index: usize,
    },
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
            Self::ScratchBufferMissing {
                bucket_index,
                command_index,
            } => write!(
                f,
                "MetalWorker: bucket {bucket_index} command {command_index}: \
                 Binding::Scratch with no splitk scratch buffer allocated \
                 (lowering / tape accounting bug)"
            ),
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
    /// Shared SplitK scratch buffer. `Some` when any bucket tape
    /// requested a non-zero `splitk_scratch_bytes` (i.e. at least one
    /// `Instruction::AffineQmm` in the tape picked
    /// `QmmTKernel::SplitK`); `None` otherwise. Sized to the max
    /// `splitk_scratch_bytes` across all bucket tapes, since
    /// successive `affine_qmm_t_splitk` calls inside a single ICB
    /// run sequentially and can reuse the same buffer.
    pub splitk_scratch: Option<Buffer>,
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
        Self::new_with_residency(
            device,
            arena_layout,
            bucket_tapes,
            pipelines,
            weights,
            allocator,
            runtime,
            None,
        )
    }

    /// Same as [`Self::new`] but accepts an optional `MetalResidencySet`
    /// that worker-local arena slots get inserted into. Used by the
    /// pool to pin every per-worker arena into the wired set so cmdbuf
    /// dispatches don't race against Apple's lazy paging on
    /// Llama-3.2-class working sets.
    #[allow(clippy::too_many_arguments)]
    pub fn new_with_residency(
        device: Arc<Device>,
        arena_layout: &ArenaLayout,
        bucket_tapes: &[LoweredMetalTape<W>],
        pipelines: &SpecializedPipelines,
        weights: &W,
        allocator: &MetalAllocator,
        runtime: &RuntimeBindings,
        residency: Option<&ferrite_metal_kernels::residency::MetalResidencySet>,
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
                let buf = device
                    .newBufferWithLength_options(
                        size as usize,
                        MTLResourceOptions::StorageModeShared,
                    )
                    .expect("newBufferWithLength_options returned nil");
                if let Some(r) = residency {
                    r.insert(&buf);
                }
                buf
            })
            .collect();

        // Shared SplitK scratch buffer sized to the max across all
        // bucket tapes — one buffer suffices because successive
        // `affine_qmm_t_splitk` / `splitk_reduce_sum` pairs run
        // serially inside a single encoder. Allocated only when at
        // least one tape requested non-zero scratch; bakings that
        // don't reference `Binding::Scratch` pay nothing.
        let max_splitk_scratch_bytes: u32 = bucket_tapes
            .iter()
            .map(|t| t.splitk_scratch_bytes)
            .max()
            .unwrap_or(0);
        let splitk_scratch: Option<Buffer> = if max_splitk_scratch_bytes > 0 {
            let buf = device
                .newBufferWithLength_options(
                    max_splitk_scratch_bytes as usize,
                    MTLResourceOptions::StorageModePrivate,
                )
                .expect("newBufferWithLength_options returned nil (splitk scratch)");
            if let Some(r) = residency {
                r.insert(&buf);
            }
            Some(buf)
        } else {
            None
        };

        // Runtime metadata buffers (`input_ids`, `positions`,
        // `slot_mapping`, `cu_seqlens_q`, `seq_used_k`, `block_table`)
        // are allocated by the per-canonical `RuntimeFactory` closure
        // and never inserted into the residency set there. ICB-recorded
        // commands bind them via `set_kernel_buffer` at bake time, so
        // the encoder firing `executeCommandsInBuffer` never sees a
        // `setBuffer` for them — without an explicit residency entry,
        // Apple's lazy paging can hand back stale pages and the ICB
        // path produces garbage output (the dormant comment at
        // `pool.rs` flagging "wrong outputs for decode buckets" was
        // exactly this). The KV cache buffers in `kv_cache_k/v` are
        // already inserted by the executor that constructs the pool;
        // inserting them again would be a no-op but we skip to keep
        // the loop tight.
        if let Some(r) = residency {
            r.insert(&runtime.input_ids);
            r.insert(&runtime.positions);
            r.insert(&runtime.slot_mapping);
            r.insert(&runtime.cu_seqlens_q);
            r.insert(&runtime.seq_used_k);
            r.insert(&runtime.block_table);
            r.commit();
        }

        let mut bucket_bakings = Vec::with_capacity(bucket_tapes.len());
        for (bucket_idx, tape) in bucket_tapes.iter().enumerate() {
            let baking = bake_bucket(
                bucket_idx,
                tape,
                &arena,
                splitk_scratch.as_ref(),
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
            splitk_scratch,
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
                let r: &ResourceRef = unsafe { &*(Retained::as_ptr(b) as *const ResourceRef) };
                r
            })
            .collect();
        // One compute encoder per ICB run. A run starts on the first
        // `BucketStep::Icb`, grows across every subsequent ICB step,
        // and only ends when a `BucketStep::Gemm` forces an encoder
        // boundary (production canonicals don't emit MPS GEMMs, so
        // every q4 / bf16 Llama bucket runs under one encoder
        // end-to-end). The encoder's default `dispatchType` is
        // `Serial`, which serializes consecutive
        // `executeCommandsInBuffer` calls on the same encoder — the
        // same RAW-correctness guarantee we used to get from
        // ending+reopening the encoder per step, at a fraction of the
        // cost.
        let mut current: Option<Retained<ProtocolObject<dyn MTLComputeCommandEncoder>>> = None;
        for step in &baking.steps {
            match step {
                BucketStep::Icb {
                    pipeline,
                    range,
                    step_resources,
                    ..
                } => {
                    let _ = step_resources;
                    let enc = match current.as_ref() {
                        Some(e) => e.clone(),
                        None => {
                            let e = cmdbuf
                                .computeCommandEncoder()
                                .expect("computeCommandEncoder returned nil");
                            current = Some(e.clone());
                            e
                        }
                    };
                    if std::env::var("FERRITE_METAL_FORCE_USE_RESOURCES").is_ok()
                        && !resource_refs.is_empty()
                    {
                        // TODO(objc2-migration): port useResources optimization
                        // path to `useResources_count_usage` once we wire the
                        // raw NonNull array build-up. Diagnostic path only —
                        // runtime correctness comes from the residency set.
                        let _ = &resource_refs;
                        let _ = MTLResourceUsage::Read | MTLResourceUsage::Write;
                    }
                    enc.setComputePipelineState(pipeline);
                    baking.icb.execute_on_encoder(&enc, range.clone());
                }
                BucketStep::Gemm { a, b, c, m, n, k } => {
                    if let Some(enc) = current.take() {
                        enc.endEncoding();
                    }
                    encode_gemm_into_command_buffer(
                        device,
                        cmdbuf,
                        &a.buffer,
                        a.offset,
                        &b.buffer,
                        b.offset,
                        &c.buffer,
                        c.offset,
                        *m,
                        *n,
                        *k,
                        1.0,
                        0.0,
                        false,
                        true,
                        gemm_dtype_for::<W>(),
                    )
                    .map_err(WorkerError::GemmEncode)?;
                }
            }
        }
        if let Some(enc) = current.take() {
            enc.endEncoding();
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
        queue: &crate::interpreter::metal::__re::CommandQueue,
    ) -> Result<(), WorkerError> {
        use crate::interpreter::metal::__re::MTLCommandBufferStatus;
        let baking = &self.bucket_bakings[bucket];
        // Bucket-wide resource pool (used as fallback / for diagnostic
        // only). The per-step path now uses each step's own
        // `step_resources` subset for its `useResources` call.
        let _bucket_resource_refs: Vec<&ResourceRef> = baking
            .baked_resources
            .iter()
            .map(|b| {
                let r: &ResourceRef = unsafe { &*(Retained::as_ptr(b) as *const ResourceRef) };
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
                let len_bytes = buf.length();
                unsafe {
                    let p = buf.contents().as_ptr() as *mut u8;
                    for off in 0..len_bytes {
                        *p.add(off) = 0xAA; // marker
                    }
                }
                let _ = i;
            }
            eprintln!(
                "[stamp] stamped 0xAA across {} arena slots",
                self.arena.len()
            );
        }

        let bucket_start = std::time::Instant::now();
        for (idx, step) in baking.steps.iter().enumerate() {
            let step_start = std::time::Instant::now();
            let cb = queue.commandBuffer().expect("commandBuffer returned nil");
            let kind: String;
            match step {
                BucketStep::Icb {
                    pipeline,
                    range,
                    kernel,
                    step_resources,
                    ..
                } => {
                    kind = format!("Icb range={:?} kernel={:?}", range, kernel);
                    let _ = pipeline; // pipeline.label() can't be safely formatted (NSString may be nil)
                    let _ = step_resources; // see note below
                    let enc = cb
                        .computeCommandEncoder()
                        .expect("computeCommandEncoder returned nil");
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
                        // TODO(objc2-migration): port useResources optimization
                        // path. Diagnostic path only — runtime correctness
                        // comes from the residency set wired on the queue.
                        let _ = &step_resources;
                        let _ = MTLResourceUsage::Read | MTLResourceUsage::Write;
                    }
                    enc.setComputePipelineState(pipeline);
                    baking.icb.execute_on_encoder(&enc, range.clone());
                    enc.endEncoding();
                }
                BucketStep::Gemm { a, b, c, m, n, k } => {
                    kind = format!("Gemm m={m} n={n} k={k}");
                    encode_gemm_into_command_buffer(
                        device,
                        &cb,
                        &a.buffer,
                        a.offset,
                        &b.buffer,
                        b.offset,
                        &c.buffer,
                        c.offset,
                        *m,
                        *n,
                        *k,
                        1.0,
                        0.0,
                        false,
                        true,
                        gemm_dtype_for::<W>(),
                    )
                    .map_err(WorkerError::GemmEncode)?;
                }
            }
            let encoded_at = step_start.elapsed();
            cb.commit();
            cb.waitUntilCompleted();
            let status = cb.status();
            let total = step_start.elapsed();
            let gpu_us = total.as_micros().saturating_sub(encoded_at.as_micros());
            // DIAGNOSTIC: dump non-zero / marker counts + hex bytes
            // of slot start. Find which step actually writes, and
            // what values it writes.
            let arena_summary = if std::env::var_os("VLLM_DUMP_ARENA_PER_STEP").is_some() {
                let mut s = String::new();
                for (i, buf) in self.arena.iter().enumerate() {
                    let len_bytes = buf.length();
                    let row0 = unsafe {
                        std::slice::from_raw_parts(buf.contents().as_ptr() as *const u8, len_bytes)
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
        queue: &crate::interpreter::metal::__re::CommandQueue,
    ) -> Result<(), WorkerError> {
        // Default-on direct dispatch; the env var stays as an
        // off-switch (`=0`) for diagnosing whether the broken ICB path
        // is at fault. The historical mode (env-var must be set to "1"
        // to enable) caused divergent output on Llama-3.2 because the
        // pool's set_var setup happened too late for some forwards.
        let direct = !matches!(
            std::env::var("FERRITE_METAL_DIRECT_DISPATCH").as_deref(),
            Ok("0" | "false" | "no")
        );
        self.run_bucket_per_step_silent_inner(bucket, device, queue, direct)
    }

    fn run_bucket_per_step_silent_inner(
        &self,
        bucket: usize,
        device: &Device,
        queue: &crate::interpreter::metal::__re::CommandQueue,
        direct: bool,
    ) -> Result<(), WorkerError> {
        use crate::interpreter::metal::__re::MTLCommandBufferStatus;
        let baking = &self.bucket_bakings[bucket];

        // Stamp marker into arena (same as per-step debug) for the
        // diagnostic dump.
        if std::env::var_os("VLLM_STAMP_ARENA").is_some() {
            for buf in self.arena.iter() {
                let len_bytes = buf.length();
                unsafe {
                    let p = buf.contents().as_ptr() as *mut u8;
                    for off in 0..len_bytes {
                        *p.add(off) = 0xAA;
                    }
                }
            }
        }

        let dump_per_step = std::env::var_os("VLLM_DUMP_ARENA_PER_STEP").is_some();
        for (idx, step) in baking.steps.iter().enumerate() {
            let cb = queue.commandBuffer().expect("commandBuffer returned nil");
            match step {
                BucketStep::Icb {
                    pipeline,
                    range,
                    direct_bindings,
                    direct_dispatch,
                    ..
                } => {
                    let enc = cb
                        .computeCommandEncoder()
                        .expect("computeCommandEncoder returned nil");
                    enc.setComputePipelineState(pipeline);
                    if direct {
                        // Direct dispatch path — bypasses the ICB.
                        // For each command in the range, bind buffers
                        // explicitly and dispatch.
                        //
                        // Even though `set_buffer` implicitly tracks
                        // residency for the bound buffer, explicit
                        // `use_resource` here makes the WRITE intent
                        // visible to Apple's tracking. On Llama-3.2
                        // shapes the pageable BFloat16 cache hit a
                        // pattern where the kernel ran on a stale or
                        // partially-paged buffer, producing
                        // non-deterministic decode output. Explicit
                        // residency closes that window.
                        for (bindings, (tg, tpt)) in
                            direct_bindings.iter().zip(direct_dispatch.iter())
                        {
                            for (buffer, offset, index) in bindings {
                                // TODO(objc2-migration): re-enable
                                // useResource_usage explicit-residency path.
                                // Currently relying on the queue-attached
                                // residency set for residency.
                                unsafe {
                                    enc.setBuffer_offset_atIndex(
                                        Some(buffer),
                                        *offset as usize,
                                        *index as usize,
                                    );
                                }
                            }
                            enc.dispatchThreadgroups_threadsPerThreadgroup(*tg, *tpt);
                        }
                    } else {
                        baking.icb.execute_on_encoder(&enc, range.clone());
                    }
                    enc.endEncoding();
                }
                BucketStep::Gemm { a, b, c, m, n, k } => {
                    encode_gemm_into_command_buffer(
                        device,
                        &cb,
                        &a.buffer,
                        a.offset,
                        &b.buffer,
                        b.offset,
                        &c.buffer,
                        c.offset,
                        *m,
                        *n,
                        *k,
                        1.0,
                        0.0,
                        false,
                        true,
                        gemm_dtype_for::<W>(),
                    )
                    .map_err(WorkerError::GemmEncode)?;
                }
            }
            cb.commit();
            cb.waitUntilCompleted();
            if cb.status() != MTLCommandBufferStatus::Completed {
                // DIAGNOSTIC (transient): surface step idx + kernel +
                // bucket so the panic root cause is identifiable from
                // chat logs. Drop once root cause is fixed.
                let kind = match step {
                    BucketStep::Icb { kernel, range, .. } => {
                        format!("Icb(kernel={kernel:?}, range={range:?})")
                    }
                    BucketStep::Gemm { m, n, k, .. } => {
                        format!("Gemm(m={m}, n={n}, k={k})")
                    }
                };
                eprintln!(
                    "[ferrite-metal] per-step commit failed: bucket_m={} step_idx={}/{} step={} status={:?}",
                    baking.bucket_m,
                    idx,
                    baking.steps.len(),
                    kind,
                    cb.status(),
                );
                return Err(WorkerError::WeightLookupFailed {
                    reason: "per-step commit failed",
                });
            }
            // VLLM_DUMP_RESIDUAL_PER_LAYER: cheap layer-wise dump.
            // Fires after Embed (initial residual = input embedding)
            // AND after FusedAddRmsNorm steps (one per half-layer in
            // Llama; the after-MLP one closes a layer). Reads the
            // FIRST 8 bf16/f16 values of slot 0 (the residual
            // stream). Cross-backend bisect tool: compare against
            // MLX `[mlx-fb-layer={N}]` lines for the same forward.
            if std::env::var_os("VLLM_DUMP_RESIDUAL_PER_LAYER").is_some()
                && matches!(
                    step,
                    BucketStep::Icb {
                        kernel: KernelId::FusedAddRmsNorm,
                        ..
                    } | BucketStep::Icb {
                        kernel: KernelId::Embed,
                        ..
                    } | BucketStep::Icb {
                        kernel: KernelId::RmsNorm,
                        ..
                    } | BucketStep::Icb {
                        kernel: KernelId::AttentionViaCache,
                        ..
                    } | BucketStep::Icb {
                        kernel: KernelId::RopeAppend,
                        ..
                    } | BucketStep::Icb {
                        kernel: KernelId::Gemm,
                        ..
                    } | BucketStep::Gemm { .. }
                )
                && self.arena.len() >= 3
            {
                let kind = match step {
                    BucketStep::Icb { kernel, .. } => format!("{:?}", kernel),
                    _ => "?".to_string(),
                };
                let dump_one = |slot: usize| -> String {
                    let buf = &self.arena[slot];
                    let len_bytes = buf.length();
                    if len_bytes < 16 {
                        return String::new();
                    }
                    let bytes = unsafe {
                        std::slice::from_raw_parts(buf.contents().as_ptr() as *const u8, len_bytes)
                    };
                    let head: Vec<String> = (0..8)
                        .map(|j| {
                            let off = j * 2;
                            let bits = u16::from_le_bytes([bytes[off], bytes[off + 1]]);
                            let v = match W::METAL_DTYPE {
                                crate::interpreter::metal::MetalDtype::Bf16 => {
                                    bf16_bits_to_f32(bits)
                                }
                                _ => f16_bits_to_f32(bits),
                            };
                            format!("{:.4}", v)
                        })
                        .collect();
                    head.join(",")
                };
                eprintln!(
                    "[ferrite-residual step={idx} {kind}] s0=[{}] s1=[{}] s2=[{}]",
                    dump_one(0),
                    dump_one(1),
                    dump_one(2),
                );
            }
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
                    let len_bytes = buf.length();
                    let bytes = unsafe {
                        std::slice::from_raw_parts(buf.contents().as_ptr() as *const u8, len_bytes)
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
                                    let bits = u16::from_le_bytes([bytes[off], bytes[off + 1]]);
                                    let v = match W::METAL_DTYPE {
                                        crate::interpreter::metal::MetalDtype::Bf16 => {
                                            bf16_bits_to_f32(bits)
                                        }
                                        _ => f16_bits_to_f32(bits),
                                    };
                                    format!("{:.3}", v)
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

fn bf16_bits_to_f32(bits: u16) -> f32 {
    f32::from_bits((bits as u32) << 16)
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
    splitk_scratch: Option<&Buffer>,
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
    //
    // `FERRITE_METAL_NO_ICB_BAKE=1` collapses the ICB to a single
    // empty slot and skips the per-command `record_compute_dispatch`
    // calls below. The direct-dispatch firing path doesn't touch the
    // ICB (it reads `direct_bindings` / `direct_dispatch` off each
    // `BucketStep::Icb`), so when direct dispatch is the production
    // path the ICB allocation + per-command pointer-copy is just dead
    // weight. Diagnostic knob for the Llama-3.2 decode-corruption bug
    // — if the bug only reproduces with ICB recording enabled, that
    // points at ICB-internal state interfering with direct dispatch.
    let no_icb_bake = std::env::var_os("FERRITE_METAL_NO_ICB_BAKE").is_some();
    let icb_capacity = if no_icb_bake {
        1
    } else {
        tape.commands.len().max(1)
    };
    let mut ctx = RecordingContext::new(device, icb_capacity).map_err(WorkerError::Recording)?;
    let mut steps: Vec<BucketStep> = Vec::new();
    // Unique buffers referenced by ICB commands. Used at firing time
    // to satisfy the `inheritBuffers=false` residency contract via
    // `useResources`. Identity is by raw `metal::Buffer` pointer —
    // the ICB doesn't care about Rust ownership, only the GPU handle.
    let mut baked_resources: Vec<Buffer> = Vec::new();
    let mut baked_seen: Vec<*const _> = Vec::new();
    let record_resource = |buf: &Buffer, seen: &mut Vec<*const _>, out: &mut Vec<Buffer>| {
        let ptr = Retained::as_ptr(buf) as *const _;
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
            if std::env::var_os("FERRITE_METAL_BAKE_DEBUG").is_some() {
                eprintln!(
                    "[bake bucket={} cmd={}] Gemm m={} n={} k={} a=({:p},+{}) b=({:p},+{}) c=({:p},+{})",
                    bucket_index,
                    cmd_idx,
                    dims.m,
                    dims.n,
                    dims.k,
                    Retained::as_ptr(&a.buffer) as *mut std::ffi::c_void,
                    a.offset,
                    Retained::as_ptr(&b.buffer) as *mut std::ffi::c_void,
                    b.offset,
                    Retained::as_ptr(&c.buffer) as *mut std::ffi::c_void,
                    c.offset,
                );
            }
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
            if std::env::var_os("FERRITE_METAL_BAKE_DEBUG").is_some() {
                eprintln!(
                    "[bake bucket={} cmd={}] kernel=Gemm m={} n={} k={} \
                     a=(buf=0x{:x},off={}) b=(buf=0x{:x},off={}) c=(buf=0x{:x},off={})",
                    bucket_index,
                    cmd_idx,
                    dims.m,
                    dims.n,
                    dims.k,
                    Retained::as_ptr(&a.buffer) as *const _ as usize,
                    a.offset,
                    Retained::as_ptr(&b.buffer) as *const _ as usize,
                    b.offset,
                    Retained::as_ptr(&c.buffer) as *const _ as usize,
                    c.offset,
                );
            }
            // f16 → MPS' `MPSMatrixMultiplication` (BucketStep::Gemm).
            // bf16 → custom `gemm_bf16_specialized` kernel routed
            // through the same per-step ICB plumbing as the other
            // compute kernels. MPS doesn't accept BFloat16 (asserted
            // at runtime), so we have to drive the hardware bf16 MMA
            // ourselves via `simdgroup_bfloat8x8`.
            match W::METAL_DTYPE {
                MetalDtype::F16 => {
                    steps.push(BucketStep::Gemm {
                        a,
                        b,
                        c,
                        m: dims.m,
                        n: dims.n,
                        k: dims.k,
                    });
                }
                MetalDtype::Bf16 => {
                    let pipeline = pipelines
                        .pipeline_for_gemm_bf16(dims.m, dims.n, dims.k)
                        .map_err(WorkerError::PipelineLookup)?;
                    // gemm_bf16_specialized binding contract:
                    //   buffer(0) = output, buffer(1) = input, buffer(2) = weight
                    let bindings_for_cmd: Vec<(Buffer, u64, u64)> = vec![
                        (c.buffer.clone(), c.offset, 0u64),
                        (a.buffer.clone(), a.offset, 1u64),
                        (b.buffer.clone(), b.offset, 2u64),
                    ];
                    // Dispatch: (ceil(N/8), ceil(M/8), 1) threadgroups,
                    // 32 threads (one simdgroup) per threadgroup.
                    let dispatch_for_cmd = (
                        MTLSize {
                            width: (dims.n as u64).div_ceil(8) as usize,
                            height: (dims.m as u64).div_ceil(8) as usize,
                            depth: 1_usize,
                        },
                        MTLSize {
                            width: 32_usize,
                            height: 1_usize,
                            depth: 1_usize,
                        },
                    );
                    let step_resources_for_cmd: Vec<Buffer> =
                        vec![c.buffer.clone(), a.buffer.clone(), b.buffer.clone()];
                    // Record into the ICB at this slot too, so the
                    // (currently-broken) ICB execution path could in
                    // principle drive bf16 GEMMs once the ICB-write
                    // bug is fixed. Direct dispatch is the production
                    // path; this just keeps both surfaces in sync.
                    let bound_refs: Vec<(&Buffer, u64, u64)> = bindings_for_cmd
                        .iter()
                        .map(|(buf, off, idx)| (buf, *off, *idx))
                        .collect();
                    let (tg, tpt) = (dispatch_for_cmd.0, dispatch_for_cmd.1);
                    if !no_icb_bake {
                        ctx.record_compute_dispatch(&pipeline, &bound_refs, tg, tpt);
                    }
                    let recorded_at = cmd_idx;
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
                            let mut seen: Vec<*const _> = step_resources
                                .iter()
                                .map(|b| Retained::as_ptr(b) as *const _)
                                .collect();
                            for buf in &step_resources_for_cmd {
                                let p = Retained::as_ptr(buf) as *const _;
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
                                kernel: KernelId::Gemm,
                                step_resources: step_resources_for_cmd,
                                direct_bindings: vec![bindings_for_cmd],
                                direct_dispatch: vec![dispatch_for_cmd],
                            });
                        }
                    }
                }
                MetalDtype::Int4 => {
                    return Err(WorkerError::PipelineLookup(
                        super::pipelines::PipelineLookupError::DtypeNotYetWired(
                            KernelId::Gemm,
                            MetalDtype::Int4,
                        ),
                    ));
                }
            }
            continue;
        }

        // The lowering pass baked `library` / `function` / `constants`
        // into the command directly — every per-layer scalar (eps,
        // attn_scale, paging strides) and the `W::METAL_DTYPE`-driven
        // symbol picks happen at lowering time, so this layer is a
        // thin cache lookup.
        let pipeline = pipelines
            .pipeline_for_command(cmd)
            .map_err(WorkerError::PipelineLookup)?;

        let bound = resolve_bindings(
            bucket_index,
            cmd_idx,
            cmd,
            arena,
            splitk_scratch,
            weights,
            allocator,
            runtime,
        )?;
        let bound_refs: Vec<(&Buffer, u64, u64)> =
            bound.iter().map(|(b, off, idx)| (b, *off, *idx)).collect();
        for (b, _, _) in &bound_refs {
            record_resource(b, &mut baked_seen, &mut baked_resources);
        }
        let (tg, tpt) = mtl_size_pair(cmd);
        if !no_icb_bake {
            ctx.record_compute_dispatch(&pipeline, &bound_refs, tg, tpt);
        }

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
            MTLSize {
                width: (tg.width),
                height: (tg.height),
                depth: (tg.depth),
            },
            MTLSize {
                width: (tpt.width),
                height: (tpt.height),
                depth: (tpt.depth),
            },
        );
        if std::env::var_os("FERRITE_METAL_BAKE_DEBUG").is_some() {
            let bind_summary: Vec<String> = bindings_for_cmd
                .iter()
                .map(|(b, off, idx)| {
                    format!(
                        "(buf=0x{:x},len={},off={},idx={})",
                        Retained::as_ptr(b) as *const _ as usize,
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
                let mut seen: Vec<*const _> = step_resources
                    .iter()
                    .map(|b| Retained::as_ptr(b) as *const _)
                    .collect();
                for buf in &step_resources_for_cmd {
                    let p = Retained::as_ptr(buf) as *const _;
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
    // KernelId::Gemm never references the SplitK scratch buffer
    // (Dense GEMM has its own dispatch path via MPS), so pass None.
    let bound = resolve_bindings(
        bucket_index,
        command_index,
        cmd,
        arena,
        /*splitk_scratch=*/ None,
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
            match (which, l) {
                // Dense path
                (WeightTensor::Weight, ferrite_kernels::layers::LinearLayer::Dense(_)) => {
                    l.dense_weight()
                }
                (WeightTensor::Bias, ferrite_kernels::layers::LinearLayer::Dense(_)) => {
                    l.dense_bias().ok_or(WorkerError::WeightLookupFailed {
                        reason: "LinearLayer bias requested but not present",
                    })?
                }
                // MLX-affine path: distinct accessors per tensor role.
                #[cfg(feature = "metal")]
                (WeightTensor::Weight, ferrite_kernels::layers::LinearLayer::AffineQuant(_)) => {
                    l.affine_weight()
                }
                #[cfg(feature = "metal")]
                (
                    WeightTensor::AffineScales,
                    ferrite_kernels::layers::LinearLayer::AffineQuant(_),
                ) => l.affine_scales(),
                #[cfg(feature = "metal")]
                (
                    WeightTensor::AffineBiases,
                    ferrite_kernels::layers::LinearLayer::AffineQuant(_),
                ) => l.affine_biases(),
                #[cfg(feature = "metal")]
                (
                    WeightTensor::AffineLinearBias,
                    ferrite_kernels::layers::LinearLayer::AffineQuant(_),
                ) => l
                    .affine_linear_bias()
                    .ok_or(WorkerError::WeightLookupFailed {
                        reason: "AffineQuant linear_bias requested but not present",
                    })?,
                // Mismatch: affine WeightTensor on a Dense layer (or vice versa),
                // or any quant arm we don't expect to reach the Metal worker.
                (WeightTensor::AffineScales, _)
                | (WeightTensor::AffineBiases, _)
                | (WeightTensor::AffineLinearBias, _) => {
                    return Err(WorkerError::WeightLookupFailed {
                        reason: "AffineScales/AffineBiases/AffineLinearBias requested \
                                 but LinearLayer is not AffineQuant",
                    });
                }
                (WeightTensor::Weight | WeightTensor::Bias, _) => {
                    return Err(WorkerError::WeightLookupFailed {
                        reason: "LinearLayer arm not reachable on the Metal worker — \
                                 macro should only emit Dense or AffineQuant on metal",
                    });
                }
            }
        }
        WeightBundleKind::CosSin(cosfn) => (cosfn)(weights, layer),
        // MLX-affine int4 quantized embedding (P6). The lowering's
        // `AffineEmbed` arm always uses `layer = 0` (embed_tokens is
        // not a layered weight) and the kernel expects three buffer
        // bindings: packed weight, scales, biases.
        #[cfg(feature = "metal")]
        WeightBundleKind::AffineQuantEmbedding(wtfn) => {
            let e = (wtfn)(weights, layer);
            match which {
                WeightTensor::Weight => e.weight,
                WeightTensor::AffineScales => e.scales,
                WeightTensor::AffineBiases => e.affine_biases,
                WeightTensor::Bias
                | WeightTensor::AffineLinearBias => {
                    return Err(WorkerError::WeightLookupFailed {
                        reason: "AffineQuantEmbedding has no linear-layer bias \
                                 — embeddings only carry (weight, scales, biases)",
                    });
                }
            }
        }
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
#[allow(clippy::too_many_arguments)]
fn resolve_bindings<W: CanonicalParams>(
    bucket_index: usize,
    command_index: usize,
    cmd: &LoweredCommand<W>,
    arena: &[Buffer],
    splitk_scratch: Option<&Buffer>,
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
            Binding::Scratch { binding_index } => {
                let scratch = splitk_scratch.ok_or(WorkerError::ScratchBufferMissing {
                    bucket_index,
                    command_index,
                })?;
                (scratch.clone(), 0u64, *binding_index as u64)
            }
        };
        out.push((buf, off, idx));
    }
    Ok(out)
}

/// Map `W::METAL_DTYPE` (the dtype the rest of the metal stack speaks
/// in) to the `GemmDtype` MPS expects. `Int4` doesn't have a direct
/// MPS GEMM mapping — int4 weights need a separate dequantize-then-
/// matmul shape, not a flat MPSMatrix dtype — so we panic here until
/// the int4 routing lands.
fn gemm_dtype_for<W: CanonicalParams>() -> GemmDtype {
    match W::METAL_DTYPE {
        super::lowered::MetalDtype::F16 => GemmDtype::F16,
        super::lowered::MetalDtype::Bf16 => GemmDtype::Bf16,
        super::lowered::MetalDtype::Int4 => panic!(
            "MetalDtype::Int4 has no direct MPS GEMM dtype — int4 weights \
             must route through the AWQ / GPTQ dequant kernel, not this Gemm step",
        ),
    }
}

fn mtl_size_pair<W: CanonicalParams>(cmd: &LoweredCommand<W>) -> (MTLSize, MTLSize) {
    let tg = MTLSize {
        width: cmd.dispatch.threadgroups.0 as usize,
        height: cmd.dispatch.threadgroups.1 as usize,
        depth: cmd.dispatch.threadgroups.2 as usize,
    };
    let tpt = MTLSize {
        width: cmd.dispatch.threads_per_threadgroup.0 as usize,
        height: cmd.dispatch.threads_per_threadgroup.1 as usize,
        depth: cmd.dispatch.threads_per_threadgroup.2 as usize,
    };
    (tg, tpt)
}

/// Compare two `ComputePipelineState`s by ObjC handle. Cached
/// pipelines for the same `(kernel, bucket, extras)` tuple are
/// pointer-equal, so this is the right test for segment coalescing.
fn same_pipeline(a: &ComputePipelineState, b: &ComputePipelineState) -> bool {
    std::ptr::eq(
        Retained::as_ptr(a) as *const _,
        Retained::as_ptr(b) as *const _,
    )
}

#[cfg(all(test, target_os = "macos"))]
mod tests {
    use super::*;
    use crate::CanonicalParams;
    use crate::interpreter::metal::lowered::{
        Binding, DispatchShape, LoweredCommand, RuntimeBindingKind, WeightBundleKind, WeightTensor,
    };
    use ferrite_cuda_core::{DType, DeviceAllocator, GpuTensor};
    use ferrite_kernels::layers::{Linear, LinearLayer, RmsNorm};
    use ferrite_metal_kernels::specialized_pipeline_cache::{
        ConstantValue, SpecializedPipelineCache,
    };
    use std::sync::Arc;

    /// Test fixture: holds `CanonicalParams` constants AND the layer
    /// instances the WtFn thunks below dereference. Plays the role
    /// of the per-canonical `Weights` struct the macro will emit.
    struct TestWeights {
        rmsnorm_layer: RmsNorm,
        linear_layer: LinearLayer,
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
        // Pin the smoke tests to the f16 / MPS GEMM path. The
        // `worker_routes_gemm_step` / `worker_interleaves_gemm_with_icb`
        // tests below assert specifically on `BucketStep::Gemm`
        // (the MPS branch). The bf16 / `BucketStep::Icb` GEMM
        // routing has its own coverage in the e2e tests.
        const METAL_DTYPE: MetalDtype = MetalDtype::F16;
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

        let weights = Arc::new(TestWeights {
            rmsnorm_layer: RmsNorm::new(rmsnorm_tensor, 1e-5),
            linear_layer: LinearLayer::Dense(Linear::new(linear_tensor, None)),
        });
        (weights, Arc::new(allocator))
    }

    fn alloc_buffer(device: &Device, bytes: u64) -> Buffer {
        device
            .newBufferWithLength_options(
                bytes.max(1) as usize,
                MTLResourceOptions::StorageModeShared,
            )
            .expect("newBufferWithLength_options returned nil")
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
            library: "rmsnorm",
            function: "rmsnorm_f16_s_f16_specialized",
            constants: vec![
                ConstantValue::uint(0, bucket_m),
                ConstantValue::uint(1, <TestWeights as CanonicalParams>::Q_SIZE as u32),
                ConstantValue::float(2, <TestWeights as CanonicalParams>::RMS_NORM_EPS),
            ],
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
            library: "fused_add_rmsnorm",
            function: "fused_add_rmsnorm_f16_s_f16_specialized",
            constants: vec![
                ConstantValue::uint(0, bucket_m),
                ConstantValue::uint(1, <TestWeights as CanonicalParams>::Q_SIZE as u32),
                ConstantValue::float(2, <TestWeights as CanonicalParams>::RMS_NORM_EPS),
            ],
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
            splitk_scratch_bytes: 0,
        }
    }

    impl LoweredCommand<TestWeights> {
        // Helper for the test: hand-clone (the public LoweredCommand
        // intentionally does NOT derive Clone so the live tape stays
        // single-owner).
        fn clone_for_test(&self) -> LoweredCommand<TestWeights> {
            LoweredCommand {
                kernel: self.kernel,
                library: self.library,
                function: self.function,
                constants: self.constants.clone(),
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
                        Binding::Scratch { binding_index } => Binding::Scratch {
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
    /// `attention_via_cache_v2_f16_specialized` kernel resolves through
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
            library: "attention",
            function: "attention_via_cache_v2_f16_specialized",
            constants: vec![
                ConstantValue::uint(0, TestWeights::HEAD_DIM),
                ConstantValue::uint(1, TestWeights::NUM_Q_HEADS),
                ConstantValue::uint(2, TestWeights::NUM_KV_HEADS),
                ConstantValue::float(3, TestWeights::ATTN_SCALE),
                ConstantValue::uint(4, TestWeights::BLOCK_SIZE),
                ConstantValue::uint(5, TestWeights::MAX_BLOCKS_PER_SEQ),
            ],
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
            splitk_scratch_bytes: 0,
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
            // GEMM is opaque to the unified pipeline picker — see the
            // matching note in `lowering.rs` for the production GEMM arm.
            library: "",
            function: "",
            constants: Vec::new(),
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
            splitk_scratch_bytes: 0,
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
            library: "rmsnorm",
            function: "rmsnorm_f16_s_f16_specialized",
            constants: vec![
                ConstantValue::uint(0, 1),
                ConstantValue::uint(1, <TestWeights as CanonicalParams>::Q_SIZE as u32),
                ConstantValue::float(2, <TestWeights as CanonicalParams>::RMS_NORM_EPS),
            ],
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
            library: rmsnorm_pre.library,
            function: rmsnorm_pre.function,
            constants: rmsnorm_pre.constants.clone(),
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
            splitk_scratch_bytes: 0,
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
