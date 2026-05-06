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

use ferrite_metal_kernels::gemm::{encode_gemm_into_command_buffer, GemmError};
use ferrite_metal_kernels::instruction_executor::RecordingContext;
use ferrite_metal_kernels::metal::foreign_types::ForeignType;
use ferrite_metal_kernels::metal::{
    Buffer, CommandBufferRef, ComputeCommandEncoderRef, ComputePipelineState, Device,
    MTLResourceOptions, MTLSize,
};

use super::lowered::{Binding, KernelId, LoweredCommand, LoweredMetalTape};
use super::model_meta::MetalModelMeta;
use super::pipelines::{PipelineLookupError, SpecializedPipelines};
use super::runtime::RuntimeBindings;
use crate::CanonicalParams;

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
pub struct BucketBaking {
    pub bucket_m: u32,
    pub icb: RecordingContext,
    pub steps: Vec<BucketStep>,
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
    ArenaShapeMismatch {
        expected: u32,
        actual: usize,
    },
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
    pub fn new(
        device: Arc<Device>,
        arena_layout: &ArenaLayout,
        bucket_tapes: &[LoweredMetalTape<W>],
        pipelines: &SpecializedPipelines,
        model_meta: &dyn MetalModelMeta<W>,
        runtime: &RuntimeBindings,
    ) -> Result<Self, WorkerError> {
        // Arena slot count comes from the lowered tape (post-FUF
        // coloring). Every bucket of a given model shares the same
        // colored slot map, so checking the first bucket is enough.
        if let Some(first) = bucket_tapes.first() {
            if first.num_arena_slots as usize != arena_layout.len() {
                return Err(WorkerError::ArenaShapeMismatch {
                    expected: first.num_arena_slots,
                    actual: arena_layout.len(),
                });
            }
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
                model_meta,
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
        for step in &baking.steps {
            match step {
                BucketStep::Icb { pipeline, range } => {
                    let enc = current.unwrap_or_else(|| cmdbuf.new_compute_command_encoder());
                    enc.set_compute_pipeline_state(pipeline);
                    baking.icb.execute_on_encoder(enc, range.clone());
                    current = Some(enc);
                }
                BucketStep::Gemm { a, b, c, m, n, k } => {
                    if let Some(enc) = current.take() {
                        enc.end_encoding();
                    }
                    encode_gemm_into_command_buffer(
                        device,
                        cmdbuf,
                        &a.buffer,
                        &b.buffer,
                        &c.buffer,
                        *m,
                        *n,
                        *k,
                        1.0,
                        0.0,
                        false,
                        true,
                        true,
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
}

/// Bake one bucket's ICB + execution plan.
fn bake_bucket<W: CanonicalParams>(
    bucket_index: usize,
    tape: &LoweredMetalTape<W>,
    arena: &[Buffer],
    pipelines: &SpecializedPipelines,
    model_meta: &dyn MetalModelMeta<W>,
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

    for (cmd_idx, cmd) in tape.commands.iter().enumerate() {
        if matches!(cmd.kernel, KernelId::Gemm) {
            let dims = cmd
                .gemm_dims
                .ok_or(WorkerError::MissingGemmDims {
                    bucket_index,
                    command_index: cmd_idx,
                })?;
            let (a, b, c) = resolve_gemm_buffers(
                bucket_index,
                cmd_idx,
                cmd,
                arena,
                model_meta,
                runtime,
            )?;
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

        let extras = model_meta.kernel_extras_for(cmd);
        let pipeline = pipelines
            .pipeline_for::<W>(cmd.kernel, tape.bucket_m, extras)
            .map_err(WorkerError::PipelineLookup)?;

        let bound = resolve_bindings(bucket_index, cmd_idx, cmd, arena, model_meta, runtime)?;
        let bound_refs: Vec<(&Buffer, u64, u64)> =
            bound.iter().map(|&(b, off, idx)| (b, off, idx)).collect();
        let (tg, tpt) = mtl_size_pair(cmd);
        ctx.record_compute_dispatch(&pipeline, &bound_refs, tg, tpt);

        // Coalesce with the previous step iff (a) it's an ICB step
        // (a Gemm step forces an encoder boundary) and (b) its
        // pipeline shares the underlying ObjC pointer (specialized
        // pipelines are refcounted — same key returns same handle
        // from the cache).
        let recorded_at = cmd_idx;
        match steps.last_mut() {
            Some(BucketStep::Icb { pipeline: prev, range })
                if same_pipeline(prev, &pipeline) =>
            {
                range.end = recorded_at + 1;
            }
            _ => {
                steps.push(BucketStep::Icb {
                    pipeline,
                    range: recorded_at..(recorded_at + 1),
                });
            }
        }
    }

    Ok(BucketBaking {
        bucket_m: tape.bucket_m,
        icb: ctx,
        steps,
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
    model_meta: &dyn MetalModelMeta<W>,
    runtime: &RuntimeBindings,
) -> Result<(BoundBuffer, BoundBuffer, BoundBuffer), WorkerError> {
    let bound = resolve_bindings(bucket_index, command_index, cmd, arena, model_meta, runtime)?;
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
    let out = bound[0];
    let inp = bound[1];
    let wt = bound[2];
    Ok((
        BoundBuffer {
            buffer: inp.0.clone(),
            offset: inp.1,
        },
        BoundBuffer {
            buffer: wt.0.clone(),
            offset: wt.1,
        },
        BoundBuffer {
            buffer: out.0.clone(),
            offset: out.1,
        },
    ))
}

/// Resolve every binding on `cmd` to (buffer, offset, binding-index).
fn resolve_bindings<'a, W: CanonicalParams>(
    bucket_index: usize,
    command_index: usize,
    cmd: &LoweredCommand<W>,
    arena: &'a [Buffer],
    model_meta: &'a dyn MetalModelMeta<W>,
    runtime: &'a RuntimeBindings,
) -> Result<Vec<(&'a Buffer, u64, u64)>, WorkerError> {
    let mut out: Vec<(&'a Buffer, u64, u64)> = Vec::with_capacity(cmd.bindings.len());
    for binding in &cmd.bindings {
        let (buf, off, idx) = match binding {
            Binding::ArenaSlot { slot, binding_index } => {
                let s = *slot as usize;
                if s >= arena.len() {
                    return Err(WorkerError::ArenaSlotOutOfRange {
                        bucket_index,
                        command_index,
                        slot: *slot,
                        arena_len: arena.len(),
                    });
                }
                (&arena[s], 0u64, *binding_index as u64)
            }
            Binding::Weight {
                kind,
                which,
                layer,
                binding_index,
            } => {
                let r = model_meta.weight_buffer(kind, *layer, *which);
                (r.buffer, r.offset, *binding_index as u64)
            }
            Binding::Runtime {
                kind,
                binding_index,
            } => (runtime.buffer_for(*kind), 0u64, *binding_index as u64),
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
    use crate::interpreter::metal::lowered::{
        Binding, DispatchShape, LoweredCommand, RuntimeBindingKind, WeightBundleKind, WeightTensor,
    };
    use crate::interpreter::metal::model_meta::BufferRef;
    use crate::interpreter::metal::pipelines::KernelExtras;
    use crate::CanonicalParams;
    use ferrite_kernels::layers::{Embedding, LinearLayer, RmsNorm};
    use ferrite_metal_kernels::specialized_pipeline_cache::SpecializedPipelineCache;
    use std::sync::Arc;

    /// CanonicalParams stub modelled on TinyLlama-1.1B; matches the
    /// probe in `pipelines.rs` so the same pipelines compile.
    struct TinyLlamaProbe;
    impl CanonicalParams for TinyLlamaProbe {
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

    // Stubs for typed `WtFn`. Never actually invoked — the worker
    // identifies a binding by `WeightBundleKind` discriminant alone
    // (the function-pointer identity is the model-meta's job to
    // resolve, and the test's meta uses neither).
    fn stub_rmsnorm(_w: &TinyLlamaProbe, _layer: u32) -> &'static RmsNorm {
        unreachable!("test meta resolves by discriminant, not by invoking the WtFn")
    }
    fn _stub_linear(_w: &TinyLlamaProbe, _layer: u32) -> &'static LinearLayer {
        unreachable!("test meta resolves by discriminant, not by invoking the WtFn")
    }
    fn _stub_embedding(_w: &TinyLlamaProbe, _layer: u32) -> &'static Embedding {
        unreachable!("test meta resolves by discriminant, not by invoking the WtFn")
    }

    /// Trivial model meta: every weight ask returns the same backing
    /// buffer. Sufficient for verifying the worker's recording flow
    /// — actual numerical correctness lives behind the cpu_golden
    /// hookup (Phase 5.G).
    struct StubMeta {
        rmsnorm_weight: Buffer,
        /// Backing for `LinearLayer` weight bundles (GEMM routing).
        /// Sized to the largest TinyLlama-class projection
        /// (`Q_SIZE × Q_SIZE` f16 = ~8 MB) so the buffer-length checks
        /// inside `encode_gemm_into_command_buffer` are satisfied for
        /// the smoke-test dimensions.
        linear_weight: Buffer,
    }

    impl MetalModelMeta<TinyLlamaProbe> for StubMeta {
        fn weight_buffer(
            &self,
            kind: &WeightBundleKind<TinyLlamaProbe>,
            _layer: u32,
            _which: WeightTensor,
        ) -> BufferRef<'_> {
            match kind {
                WeightBundleKind::RmsNorm(_) => BufferRef {
                    buffer: &self.rmsnorm_weight,
                    offset: 0,
                },
                WeightBundleKind::LinearLayer(_) => BufferRef {
                    buffer: &self.linear_weight,
                    offset: 0,
                },
                _ => unreachable!("smoke tests only exercise RmsNorm + LinearLayer bindings"),
            }
        }

        fn kernel_extras_for(&self, cmd: &LoweredCommand<TinyLlamaProbe>) -> KernelExtras {
            // Per-`KernelId` defaults so the smoke tests can mix
            // RmsNorm (needs `eps`) and AttentionViaCache (needs
            // `block_size` + `max_blocks_per_seq`) in the same
            // synthetic tape.
            match cmd.kernel {
                KernelId::AttentionViaCache => KernelExtras {
                    block_size: 16,
                    max_blocks_per_seq: 128,
                    ..KernelExtras::NONE
                },
                _ => KernelExtras {
                    eps: 1e-5,
                    ..KernelExtras::NONE
                },
            }
        }
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
    fn build_synthetic_tape(bucket_m: u32) -> LoweredMetalTape<TinyLlamaProbe> {
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
                    kind: WeightBundleKind::RmsNorm(stub_rmsnorm),
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
                    kind: WeightBundleKind::RmsNorm(stub_rmsnorm),
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

    impl LoweredCommand<TinyLlamaProbe> {
        // Helper for the test: hand-clone (the public LoweredCommand
        // intentionally does NOT derive Clone so the live tape stays
        // single-owner).
        fn clone_for_test(&self) -> LoweredCommand<TinyLlamaProbe> {
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

        let meta = StubMeta {
            rmsnorm_weight: alloc_buffer(&device, 4096),
            linear_weight: alloc_buffer(&device, 16 * 1024 * 1024),
        };
        let runtime = empty_runtime(&device, 1);

        // Two buckets: M=1 (decode) and M=8 (small prefill).
        let tapes = vec![build_synthetic_tape(1), build_synthetic_tape(8)];
        let arena_layout: ArenaLayout = vec![4 * 1024, 4 * 1024];

        let worker = MetalWorker::<TinyLlamaProbe>::new(
            device,
            &arena_layout,
            &tapes,
            &pipelines,
            &meta,
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
            assert_eq!(baking.steps.len(), 2, "expected RmsNorm + FusedAddRmsNorm coalesced");
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
        let meta = StubMeta {
            rmsnorm_weight: alloc_buffer(&device, 4096),
            linear_weight: alloc_buffer(&device, 16 * 1024 * 1024),
        };
        let runtime = empty_runtime(&device, 1);

        let tapes = vec![build_synthetic_tape(1)]; // num_arena_slots = 2

        // Layout has only 1 slot — should error.
        let bad_layout: ArenaLayout = vec![4 * 1024];
        let err = MetalWorker::<TinyLlamaProbe>::new(
            device, &bad_layout, &tapes, &pipelines, &meta, &runtime,
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

        let meta = StubMeta {
            rmsnorm_weight: alloc_buffer(&device, 4096),
            linear_weight: alloc_buffer(&device, 16 * 1024 * 1024),
        };
        let runtime = empty_runtime(&device, 1);

        // Single AttentionViaCache command at decode bucket=1
        // (batch=1, num_q_heads heads).
        let attn = LoweredCommand {
            kernel: KernelId::AttentionViaCache,
            dispatch: DispatchShape {
                threadgroups: (1, TinyLlamaProbe::NUM_Q_HEADS, 1),
                threads_per_threadgroup: (TinyLlamaProbe::HEAD_DIM, 1, 1),
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

        let worker = MetalWorker::<TinyLlamaProbe>::new(
            device,
            &vec![1024, 1024],
            &[tape],
            &pipelines,
            &meta,
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
    fn build_gemm_command(bucket_m: u32, n: u32, k: u32) -> LoweredCommand<TinyLlamaProbe> {
        // Stub linear thunk — never invoked (StubMeta resolves by
        // discriminant). Cast to a `WtFn` so the variant type-checks.
        fn linear_stub(_w: &TinyLlamaProbe, _layer: u32) -> &'static LinearLayer {
            unreachable!("StubMeta resolves LinearLayer by discriminant only")
        }
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
                    kind: WeightBundleKind::LinearLayer(linear_stub),
                    which: WeightTensor::Weight,
                    layer: 0,
                    binding_index: 2,
                },
            ],
            gemm_dims: Some(crate::interpreter::metal::lowered::GemmDims {
                m: bucket_m,
                n,
                k,
            }),
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
        let meta = StubMeta {
            rmsnorm_weight: alloc_buffer(&device, 4096),
            linear_weight: alloc_buffer(&device, 16 * 1024 * 1024),
        };
        let runtime = empty_runtime(&device, 1);

        let tape = LoweredMetalTape {
            bucket_m: 1,
            num_arena_slots: 2,
            commands: vec![build_gemm_command(1, 2048, 2048)],
        };

        let worker = MetalWorker::<TinyLlamaProbe>::new(
            device,
            // Q-size buffers (2048 f16 = 4096 bytes; pad up).
            &vec![64 * 1024, 64 * 1024],
            &[tape],
            &pipelines,
            &meta,
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
        let meta = StubMeta {
            rmsnorm_weight: alloc_buffer(&device, 4096),
            linear_weight: alloc_buffer(&device, 16 * 1024 * 1024),
        };
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
                    kind: WeightBundleKind::RmsNorm(stub_rmsnorm),
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
            commands: vec![
                rmsnorm_pre,
                build_gemm_command(1, 2048, 2048),
                rmsnorm_post,
            ],
        };

        let worker = MetalWorker::<TinyLlamaProbe>::new(
            device,
            &vec![64 * 1024, 64 * 1024],
            &[tape],
            &pipelines,
            &meta,
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
        assert!(matches!(&baking.steps[1], BucketStep::Gemm { m: 1, n: 2048, k: 2048, .. }));
        // Post-GEMM RmsNorm is at command index 2, not coalesced with
        // the pre-GEMM RmsNorm despite the identical pipeline.
        assert!(matches!(&baking.steps[2], BucketStep::Icb { range, .. } if *range == (2..3)));
    }
}
