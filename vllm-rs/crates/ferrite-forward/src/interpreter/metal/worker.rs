// SPDX-License-Identifier: Apache-2.0
//! `MetalWorker`: arena + per-bucket MTL4 execution plan.
//!
//! One worker holds:
//!  - a private arena of `metal::Buffer`s, sized by the model's
//!    post-coloring slot count (one buffer per slot id);
//!  - one [`BucketBaking`] per bucket, holding a `Vec<BucketStep>`
//!    (the execution plan) and the pre-built MTL4 `Mtl4Step` list.
//!
//! The execution plan partitions the bucket's command stream into
//! [`BucketStep`]s: contiguous runs of commands sharing a pipeline
//! become a single `BucketStep::Icb`, while f16 MPS GEMM steps
//! become `BucketStep::Gemm`. MTL4 execution reads the pre-baked
//! `mtl4_steps` from each baking; buckets containing a `Gemm` step
//! are ineligible for MTL4 and return an error at forward time.

#![cfg(feature = "metal")]

use std::sync::Arc;

use crate::interpreter::metal::__re::{
    Buffer, ComputePipelineState, Device, MTLBuffer, MTLDevice, MTLResourceOptions, MTLSize,
};
use ::objc2::rc::Retained;
use ::objc2::runtime::ProtocolObject;
use ::objc2_metal::MTL4ComputeCommandEncoder;

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
/// `Icb` is a contiguous run of commands sharing a single pipeline.
/// `Gemm` is an MPS dense `y = x @ W^T` step (f16 only); buckets
/// containing a `Gemm` are ineligible for MTL4 (`bake_mtl4_steps`
/// returns `None`).
pub enum BucketStep {
    Icb {
        /// Kernel id of every dispatch in this step. Coalescing
        /// requires same pipeline (same kernel), so one id is
        /// authoritative for the whole step.
        kernel: super::lowered::KernelId,
        /// Pipeline state for this step's kernel(s).
        pipeline: ComputePipelineState,
        /// Per-command explicit bindings: (buffer, offset, index).
        direct_bindings: Vec<Vec<(Buffer, u64, u64)>>,
        /// Per-command dispatch shape: (threadgroups, threads_per_tg).
        direct_dispatch: Vec<(MTLSize, MTLSize)>,
        /// Per-sub-dispatch m-axis scaling hint, parallel to
        /// `direct_dispatch`. When `Some`, the runtime rewrites
        /// `threadgroups.{axis}` proportionally with actual
        /// `num_tokens` (see
        /// [`crate::interpreter::metal::lowered::MScaling`]),
        /// shrinking the grid to the actual M instead of paying
        /// the `bucket_m`-shaped over-dispatch cost.
        direct_m_scaling: Vec<Option<super::lowered::MScaling>>,
        /// Per-sub-dispatch barrier-before flag, sourced from the
        /// compile-time DAG hazard analysis in `LoweredMetalTape`.
        /// Consumed by the MTL4 path via `Mtl4Step.barrier_before`.
        barrier_before: Vec<bool>,
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

/// One bucket's baked artifacts: the execution plan and MTL4 steps.
pub struct BucketBaking {
    pub bucket_m: u32,
    pub steps: Vec<BucketStep>,
    /// MTL4 steps built from `steps`. `None` iff any step is a
    /// `Gemm` (MPS f16) or a kernel exceeds the 31-binding cap.
    pub mtl4_steps: Option<Vec<super::mtl4::Mtl4Step>>,
}

#[derive(Debug)]
pub enum WorkerError {
    /// Lookup against [`SpecializedPipelines`] failed.
    PipelineLookup(PipelineLookupError),
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
        bucket_tapes: &[LoweredMetalTape],
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
        bucket_tapes: &[LoweredMetalTape],
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

        // Each ICB built during bake_bucket needs to be resident
        // before `executeCommandsInBuffer` reads its commands. The
        // ICB itself is an MTLAllocation but not an MTLBuffer, so we
        // go through `insert_raw` (added to MetalResidencySet for
        // exactly this case).
        if let Some(r) = residency {
            for baking in &bucket_bakings {
                if let Some(steps) = baking.mtl4_steps.as_ref() {
                    for step in steps {
                        if let Some(ctx) = step.icb.as_ref() {
                            unsafe { r.insert_raw(ctx.icb().as_ptr()); }
                        }
                    }
                }
            }
        }

        Ok(Self {
            arena,
            bucket_bakings,
            splitk_scratch,
            _marker: std::marker::PhantomData,
        })
    }

    /// MTL3 direct-dispatch fallback path. Uses MTL3 compute encoder
    /// with `setBuffer/dispatchThreadgroups` (NOT ICB / NOT MTL4) —
    /// the default Serial encoder auto-serializes dependent dispatches
    /// without per-dispatch barrier overhead that MTL4 imposes on M4.
    /// Same `BucketStep::Icb` execution plan; just a different
    /// dispatch protocol. Returns an error on Gemm (MPS) buckets
    /// since those have their own dispatch.
    pub fn run_bucket_mtl3(
        &self,
        bucket: usize,
        num_tokens: u32,
        enc: &::objc2::runtime::ProtocolObject<dyn ::objc2_metal::MTLComputeCommandEncoder>,
    ) -> Result<(), WorkerError> {
        use ::objc2_metal::{MTLCommandEncoder, MTLComputeCommandEncoder};
        let baking = &self.bucket_bakings[bucket];
        for step in &baking.steps {
            match step {
                BucketStep::Icb {
                    pipeline,
                    direct_bindings,
                    direct_dispatch,
                    direct_m_scaling,
                    ..
                } => {
                    enc.setComputePipelineState(pipeline);
                    for ((bindings, (tg, tpt)), scaling) in direct_bindings
                        .iter()
                        .zip(direct_dispatch.iter())
                        .zip(direct_m_scaling.iter())
                    {
                        for (buf, off, idx) in bindings {
                            unsafe {
                                enc.setBuffer_offset_atIndex(Some(buf), *off as usize, *idx as usize);
                            }
                        }
                        let tg_scaled = scale_tg_for_num_tokens(
                            *tg,
                            *scaling,
                            super::ids::NumTokens(num_tokens),
                        );
                        enc.dispatchThreadgroups_threadsPerThreadgroup(tg_scaled, *tpt);
                    }
                }
                BucketStep::Gemm { .. } => {
                    return Err(WorkerError::WeightLookupFailed {
                        reason: "MTL3 path cannot handle Gemm step (MPS dispatch). Use MTL4 path for Gemm buckets.",
                    });
                }
            }
        }
        Ok(())
    }

    /// Count total dispatches across all MTL4 steps in this bucket.
    /// Used by `DispatchTimingState` to size the counter heap to
    /// (dispatch_count + 1) timestamps.
    pub fn count_dispatches(&self, bucket: usize) -> usize {
        self.bucket_bakings
            .get(bucket)
            .and_then(|b| b.mtl4_steps.as_ref())
            .map(|steps| steps.iter().map(|s| s.dispatches.len()).sum())
            .unwrap_or(0)
    }

    /// Phase A.3 MTL4 execution path. Encodes the
    /// bucket's `mtl4_steps` onto a caller-provided MTL4 compute
    /// encoder using pre-baked `MTL4ArgumentTable`s. The caller owns
    /// command-buffer lifecycle (`begin`/`endCommandBuffer`),
    /// residency wiring (`useResidencySet`), commit, and event-based
    /// wait. Returns an error if MTL4 was not baked for this bucket
    /// (e.g. an MPS GEMM step disqualifies it).
    pub fn run_bucket_mtl4(
        &self,
        bucket: usize,
        num_tokens: u32,
        enc: &ProtocolObject<dyn ::objc2_metal::MTL4ComputeCommandEncoder>,
    ) -> Result<(), WorkerError> {
        self.run_bucket_mtl4_inner(bucket, num_tokens, enc, None)
    }

    /// Variant with optional GPU-timestamp instrumentation.
    ///
    /// When `timing` is `Some`, this writes a timestamp into the
    /// caller-provided counter heap immediately before each dispatch
    /// AND once at the very end. After GPU completion the caller
    /// resolves the heap to get N+1 GPU-timeline timestamps for N
    /// dispatches; the i-th dispatch ran from `ts[i]` to `ts[i+1]`.
    ///
    /// Triggered via `MetalWorkerPool::forward` when
    /// `FERRITE_METAL_DISPATCH_TIMING=1` is set.
    pub fn run_bucket_mtl4_with_timing(
        &self,
        bucket: usize,
        num_tokens: u32,
        enc: &ProtocolObject<dyn ::objc2_metal::MTL4ComputeCommandEncoder>,
        timing: &super::pool::DispatchTimingState,
    ) -> Result<(), WorkerError> {
        self.run_bucket_mtl4_inner(bucket, num_tokens, enc, Some(timing))
    }

    fn run_bucket_mtl4_inner(
        &self,
        bucket: usize,
        num_tokens: u32,
        enc: &ProtocolObject<dyn ::objc2_metal::MTL4ComputeCommandEncoder>,
        mut timing: Option<&super::pool::DispatchTimingState>,
    ) -> Result<(), WorkerError> {
        use ::objc2_metal::{
            MTL4CommandEncoder, MTL4ComputeCommandEncoder as _,
            MTL4TimestampGranularity, MTL4VisibilityOptions, MTLStages,
        };
        let baking = &self.bucket_bakings[bucket];
        let mtl4_steps =
            baking
                .mtl4_steps
                .as_ref()
                .ok_or(WorkerError::WeightLookupFailed {
                    reason: "MTL4 path requested but bucket has no mtl4_steps (Gemm or too-many-bindings fallback)",
                })?;
        // MTL4 compute encoders do NOT auto-serialize successive
        // dispatches the way MTL3's default-Serial encoder does;
        // the per-sub-dispatch `barrier_before` flag was computed
        // at macro time by `ferrite-forward-macro::interpreter_codegen
        // ::lower_bucket` from the FUF dataflow + `Implementation::
        // kv_layer_io`. Runtime does zero analysis — just emits a
        // `Dispatch→Dispatch` barrier wherever the flag fires.
        let mut ts_idx: usize = 0;
        let count_barriers = std::env::var_os("FERRITE_METAL_COUNT_BARRIERS").is_some();
        let mut total_dispatches: usize = 0;
        let mut total_barriers: usize = 0;
        for step in mtl4_steps {
            enc.setComputePipelineState(&step.pipeline);
            // ICB pre-bound fast path: when a step's dispatches were
            // pre-recorded into an MTLIndirectCommandBuffer at bake
            // time (FERRITE_METAL_ICB=1, no per-dispatch m_scaling),
            // play the whole step back as ONE
            // `executeCommandsInBuffer` driver call instead of N
            // setArgumentTable+dispatchThreadgroups round-trips.
            // Pipelines were created via the MTL4 compiler with
            // `MTL4IndirectCommandBufferSupportState::Enabled`, so
            // ICB execution under MTL4 produces the same kernel
            // state direct dispatch does.
            if let Some(ctx) = step.icb.as_ref() {
                if timing.is_none() {
                    use ::objc2::msg_send;
                    use ::objc2::runtime::AnyObject;
                    use ::objc2_foundation::NSRange;
                    let n = ctx.command_count();
                    let enc_ptr: *mut AnyObject =
                        enc as *const _ as *const AnyObject as *mut _;
                    unsafe {
                        let _: () = msg_send![
                            enc_ptr,
                            executeCommandsInBuffer: ctx.icb().as_ptr(),
                            withRange: NSRange { location: 0, length: n }
                        ];
                    }
                    if count_barriers {
                        total_dispatches += n;
                    }
                    continue;
                }
            }
            for (((table, (tg, tpt)), need_barrier), scaling) in step
                .tables
                .iter()
                .zip(step.dispatches.iter())
                .zip(step.barrier_before.iter())
                .zip(step.m_scaling.iter())
            {
                if count_barriers {
                    total_dispatches += 1;
                    if *need_barrier {
                        total_barriers += 1;
                    }
                }
                if *need_barrier {
                    // Default to `None` visibility — measured -30 ms
                    // TTFT @ 1024-tok / -89 ms @ 2048-tok on M4
                    // Llama-3.2-3B-4bit, coherent on the standard probes
                    // (short prompts, 80-tok Apollo recall, haiku
                    // composition, Llama-3.2-1B math). Within a single
                    // MTL4 compute encoder, dispatch-to-dispatch
                    // sync is sufficient for correctness — full
                    // device-coherent visibility is over-conservative
                    // for back-to-back dispatches that aren't writing
                    // to memory other dispatches in the SAME encoder
                    // need cache-coherent reads of. The final cmdbuf
                    // commit point flushes everything before the next
                    // encoder runs.
                    //
                    // `FERRITE_METAL_BARRIER_DEVICE=1` re-enables the
                    // old `Device` visibility for diagnosis if any
                    // model surfaces coherence weirdness.
                    let vis = if std::env::var_os("FERRITE_METAL_BARRIER_DEVICE").is_some() {
                        MTL4VisibilityOptions::Device
                    } else {
                        MTL4VisibilityOptions::None
                    };
                    if std::env::var_os("FERRITE_METAL_NO_BARRIERS").is_none() {
                        enc.barrierAfterEncoderStages_beforeEncoderStages_visibilityOptions(
                            MTLStages::Dispatch,
                            MTLStages::Dispatch,
                            vis,
                        );
                    }
                }
                let tg_scaled = scale_tg_for_num_tokens(
                    *tg,
                    *scaling,
                    super::ids::NumTokens(num_tokens),
                );
                if let Some(t) = timing.as_mut() {
                    if ts_idx < t.heap_capacity {
                        unsafe {
                            enc.writeTimestampWithGranularity_intoHeap_atIndex(
                                MTL4TimestampGranularity::Precise,
                                &t.heap,
                                ts_idx,
                            );
                        }
                        t.record_label(
                            ts_idx,
                            &step.pipeline,
                            step.kernel,
                            (tg_scaled.width as u32, tg_scaled.height as u32, tg_scaled.depth as u32),
                        );
                        ts_idx += 1;
                    }
                }
                enc.setArgumentTable(Some(table));
                enc.dispatchThreadgroups_threadsPerThreadgroup(tg_scaled, *tpt);
            }
        }
        // Final closing timestamp so the last dispatch's GPU time =
        // ts[N] - ts[N-1].
        if let Some(t) = timing.as_mut() {
            if ts_idx < t.heap_capacity {
                unsafe {
                    enc.writeTimestampWithGranularity_intoHeap_atIndex(
                        MTL4TimestampGranularity::Precise,
                        &t.heap,
                        ts_idx,
                    );
                }
                t.set_dispatch_count(ts_idx);
            }
        }
        if count_barriers {
            eprintln!(
                "[barrier-count] dispatches={total_dispatches} barriers={total_barriers} \
                 ({:.1}%)",
                100.0 * total_barriers as f64 / total_dispatches.max(1) as f64,
            );
        }
        Ok(())
    }
}

/// Bake one bucket's execution plan.
#[allow(clippy::too_many_arguments)]
fn bake_bucket<W: CanonicalParams>(
    bucket_index: usize,
    tape: &LoweredMetalTape,
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

    let mut steps: Vec<BucketStep> = Vec::new();

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
                    let cmd_barrier = tape
                        .barrier_before
                        .get(cmd_idx)
                        .copied()
                        .unwrap_or(true);
                    // MPS-shaped bf16 Gemm: M is the height axis but
                    // the bake here is for a dense linear that always
                    // dispatches at the actual M (no bucket_m baking),
                    // so leave m_scaling as None.
                    match steps.last_mut() {
                        Some(BucketStep::Icb {
                            pipeline: prev,
                            direct_bindings,
                            direct_dispatch,
                            direct_m_scaling,
                            barrier_before,
                            ..
                        }) if same_pipeline(prev, &pipeline) => {
                            direct_bindings.push(bindings_for_cmd);
                            direct_dispatch.push(dispatch_for_cmd);
                            direct_m_scaling.push(None);
                            barrier_before.push(cmd_barrier);
                        }
                        _ => {
                            steps.push(BucketStep::Icb {
                                kernel: KernelId::Gemm,
                                pipeline,
                                direct_bindings: vec![bindings_for_cmd],
                                direct_dispatch: vec![dispatch_for_cmd],
                                direct_m_scaling: vec![None],
                                barrier_before: vec![cmd_barrier],
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
            .pipeline_for_command::<W>(cmd)
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
        let (tg, tpt) = mtl_size_pair::<W>(cmd);

        // Coalesce with the previous step iff (a) it's an ICB step
        // (a Gemm step forces an encoder boundary) and (b) its
        // pipeline shares the underlying ObjC pointer (specialized
        // pipelines are refcounted — same key returns same handle
        // from the cache).
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
        let cmd_barrier = tape
            .barrier_before
            .get(cmd_idx)
            .copied()
            .unwrap_or(true);
        let cmd_m_scaling = cmd.dispatch.m_scaling;
        match steps.last_mut() {
            Some(BucketStep::Icb {
                pipeline: prev,
                direct_bindings,
                direct_dispatch,
                direct_m_scaling,
                barrier_before,
                ..
            }) if same_pipeline(prev, &pipeline) => {
                direct_bindings.push(bindings_for_cmd);
                direct_dispatch.push(dispatch_for_cmd);
                direct_m_scaling.push(cmd_m_scaling);
                barrier_before.push(cmd_barrier);
            }
            _ => {
                steps.push(BucketStep::Icb {
                    kernel: cmd.kernel,
                    pipeline,
                    direct_bindings: vec![bindings_for_cmd],
                    direct_dispatch: vec![dispatch_for_cmd],
                    direct_m_scaling: vec![cmd_m_scaling],
                    barrier_before: vec![cmd_barrier],
                });
            }
        }
    }

    let mtl4_steps = super::mtl4::bake_mtl4_steps(&steps, &device, tape.bucket_m);
    Ok(BucketBaking {
        bucket_m: tape.bucket_m,
        steps,
        mtl4_steps,
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
    cmd: &LoweredCommand,
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
/// Calls into the per-arch [`crate::WeightAccessors`] impl (emitted by
/// `ferrite-forward-macro::codegen::emit_weight_accessors_impl`) using
/// the binding's `(bucket, op_idx, slot, layer)` locator to recover the
/// `&Layer` struct (`&RmsNorm`, `&LinearLayer`, `&Embedding`), pulls
/// out the raw GpuTensor pointer matching `which`, and asks the
/// allocator which buffer + offset that pointer belongs to.
///
/// Same shape CUDA's interpreter uses: `WeightAccessors` → layer
/// struct → `GpuTensor`. The Metal-side delta is just the final
/// pointer → `(&Buffer, offset)` reverse lookup against the arena
/// allocator.
fn resolve_weight<W: crate::CanonicalParams + crate::WeightAccessors>(
    weights: &W,
    allocator: &MetalAllocator,
    kind: &WeightBundleKind,
    layer: u32,
    which: WeightTensor,
    locator: super::lowered::WeightLocator,
) -> Result<(Buffer, u64), WorkerError> {
    let bucket = locator.bucket;
    let op_idx = locator.op_idx;
    let slot = locator.slot;
    let tensor = match kind {
        WeightBundleKind::RmsNorm => weights.rms_norm_at(bucket, op_idx, slot, layer).weight,
        WeightBundleKind::Embedding => weights.embedding_at(bucket, op_idx, slot, layer).weight,
        WeightBundleKind::LinearLayer => {
            let l = weights.linear_at(bucket, op_idx, slot, layer);
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
        WeightBundleKind::CosSin => weights.cos_sin_at(bucket, op_idx, slot, layer),
        // MLX-affine int4 quantized embedding (P6). The lowering's
        // `AffineEmbed` arm always uses `layer = 0` (embed_tokens is
        // not a layered weight) and the kernel expects three buffer
        // bindings: packed weight, scales, biases.
        #[cfg(feature = "metal")]
        WeightBundleKind::AffineQuantEmbedding => {
            let e = weights.affine_quant_embedding_at(bucket, op_idx, slot, layer);
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
    cmd: &LoweredCommand,
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
                locator,
                binding_index,
            } => {
                let (b, off) = resolve_weight(
                    weights,
                    allocator,
                    kind,
                    layer.get(),
                    *which,
                    *locator,
                )?;
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


/// Rewrite the m-axis of a baked threadgroup grid to match the
/// actual `num_tokens` of this forward, instead of the bucket_m the
/// grid was baked against.
///
/// `scaling = None` means the kernel's grid doesn't scale with M
/// (or the lowering pass hasn't yet been taught to emit a scaling
/// hint for it) — return the baked grid unchanged. When `scaling =
/// Some(MScaling { axis, tile })`, replace `tg.{axis}` with
/// `num_tokens.div_ceil(tile)`, clamped so we never grow above the
/// baked value (guards against `num_tokens > bucket_m`, which the
/// bucket picker already rules out but defense-in-depth).
fn scale_tg_for_num_tokens(
    mut tg: MTLSize,
    scaling: Option<super::lowered::MScaling>,
    num_tokens: super::ids::NumTokens,
) -> MTLSize {
    let Some(s) = scaling else {
        return tg;
    };
    let bm = s.bucket_m.get().max(1) as u64;
    let n = (num_tokens.get().max(1) as u64).min(bm);
    let slot = match s.axis {
        super::lowered::MScaleAxis::X => &mut tg.width,
        super::lowered::MScaleAxis::Y => &mut tg.height,
        super::lowered::MScaleAxis::Z => &mut tg.depth,
    };
    // new = ceil(baseline * n / bucket_m). Clamped above to the
    // baseline so accidental num_tokens > bucket_m can't grow the
    // grid past what was baked.
    let baseline = *slot as u64;
    let scaled = (baseline.saturating_mul(n) + bm - 1) / bm;
    *slot = scaled as usize;
    tg
}

fn mtl_size_pair<W: CanonicalParams>(cmd: &LoweredCommand) -> (MTLSize, MTLSize) {
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
            num_tokens_u32: alloc_buffer(device, 4),
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
                m_scaling: None,
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
                    layer: crate::interpreter::metal::ids::LayerId(0),
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
                m_scaling: None,
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
                    layer: crate::interpreter::metal::ids::LayerId(0),
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
            barrier_before: Vec::new(),
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
                m_scaling: None,
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
                    kind: RuntimeBindingKind::KvCacheK { layer: crate::interpreter::metal::ids::LayerId(0) },
                    binding_index: 4,
                },
                Binding::Runtime {
                    kind: RuntimeBindingKind::KvCacheV { layer: crate::interpreter::metal::ids::LayerId(0) },
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
            barrier_before: Vec::new(),
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
                m_scaling: None,
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
                    layer: crate::interpreter::metal::ids::LayerId(0),
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
            barrier_before: Vec::new(),
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
                m_scaling: None,
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
                    layer: crate::interpreter::metal::ids::LayerId(0),
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
            barrier_before: Vec::new(),
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
