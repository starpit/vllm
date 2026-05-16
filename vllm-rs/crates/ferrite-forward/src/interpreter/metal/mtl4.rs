// SPDX-License-Identifier: Apache-2.0
//! MTL4 bake artifacts and step type.
//!
//! One `Mtl4Step` is produced per `BucketStep::Icb` from a bucket's
//! plan, holding the pipeline state and a pre-built `MTL4ArgumentTable`
//! per coalesced sub-command (one per entry in `direct_bindings`).
//!
//! `BucketStep::Gemm` (MPS f16 path) is not representable on the MTL4
//! encoder surface; `bake_mtl4_steps` returns `None` for any bucket
//! containing one, and the pool asserts eligibility at forward time.
//! Production int4 / bf16 paths emit only `Icb` steps.

#![cfg(feature = "metal")]

use ::objc2::rc::Retained;
use ::objc2::runtime::ProtocolObject;
use ::objc2_metal::{MTL4ArgumentTable, MTLBuffer};

use ferrite_metal_kernels::instruction_executor::RecordingContext;

use super::__re::{
    ComputePipelineState, Device, MTL4ArgumentTableDescriptor, MTLDevice, MTLSize,
};
use super::worker::BucketStep;

/// MTL4 argument-table buffer-binding slot cap. The Metal runtime
/// enforces 31; we exit baking with `None` if any kernel asks for
/// more (would force MTL3 fallback for that bucket).
pub const MTL4_MAX_BUFFER_BINDS: usize = 31;

/// One MTL4 dispatch segment, mirroring a `BucketStep::Icb`.
///
/// `pipeline` is the same `dyn MTLComputePipelineState` the MTL3
/// path uses; MTL4's `setComputePipelineState` accepts it
/// unchanged. `tables[i]` + `dispatches[i]` together encode one
/// dispatch — `tables.len() == dispatches.len()` and matches the
/// inner length of the corresponding `direct_bindings` /
/// `direct_dispatch` in the source `BucketStep::Icb`.
pub struct Mtl4Step {
    /// Kernel id of this step's dispatches (one kernel per Icb step;
    /// coalescing requires same pipeline = same kernel). Used for
    /// the per-dispatch timing print label.
    pub kernel: super::lowered::KernelId,
    pub pipeline: ComputePipelineState,
    pub tables: Vec<Retained<ProtocolObject<dyn MTL4ArgumentTable>>>,
    pub dispatches: Vec<(MTLSize, MTLSize)>,
    /// Parallel to `dispatches`: optional per-dispatch m-axis scaling
    /// hint. When `Some`, the runtime patches the named axis of
    /// `dispatches[i].0` proportionally with actual `num_tokens`
    /// (see [`super::lowered::MScaling`]), shrinking the grid from
    /// the `bucket_m`-baked baseline down to the actual M of this
    /// forward.
    pub m_scaling: Vec<Option<super::lowered::MScaling>>,
    /// One barrier-before flag per sub-dispatch (parallel to
    /// `tables` / `dispatches`). Sourced from the macro-emitted
    /// `LoweredMetalTape::barrier_before` — no runtime analysis.
    /// `true` means the runtime must emit a `Dispatch→Dispatch`
    /// MTL4 encoder barrier before this sub-dispatch.
    pub barrier_before: Vec<bool>,
    /// Pre-recorded indirect command buffer covering all dispatches
    /// in this step. Built only when `FERRITE_METAL_ICB=1` was set
    /// at bake time AND no sub-dispatch in the step uses
    /// `m_scaling` (ICB grids are baked, can't be shrunk per
    /// forward — sub-dispatches with scaling fall back to the per-
    /// dispatch path within the same encoder). `executeCommandsInBuffer`
    /// on the MTL4 encoder plays the whole step back as one driver
    /// call.
    pub icb: Option<RecordingContext>,
}

// Mtl4Step holds a `RecordingContext` with a raw `*mut AnyObject`
// inside (the underlying MTLIndirectCommandBuffer); Apple ARC keeps
// it alive for the worker's lifetime, and the pool checks each
// worker out under a mutex (see `MetalWorkerPool::checkout`).
unsafe impl Send for Mtl4Step {}
unsafe impl Sync for Mtl4Step {}

/// Build one `Mtl4Step` per `BucketStep::Icb`. Returns `None` if any
/// step is a `Gemm` (MPS) or any kernel exceeds the argument-table
/// binding cap — the caller falls back to the MTL3 path for that
/// bucket.
pub fn bake_mtl4_steps(steps: &[BucketStep], device: &Device) -> Option<Vec<Mtl4Step>> {
    let mut out = Vec::with_capacity(steps.len());
    for step in steps {
        match step {
            BucketStep::Gemm { .. } => return None,
            BucketStep::Icb {
                kernel,
                pipeline,
                direct_bindings,
                direct_dispatch,
                direct_m_scaling,
                barrier_before,
                ..
            } => {
                debug_assert_eq!(direct_bindings.len(), direct_dispatch.len());
                debug_assert_eq!(direct_bindings.len(), direct_m_scaling.len());
                debug_assert_eq!(direct_bindings.len(), barrier_before.len());
                let mut tables = Vec::with_capacity(direct_bindings.len());
                for cmd_bindings in direct_bindings {
                    let max_idx = cmd_bindings
                        .iter()
                        .map(|(_, _, idx)| *idx as usize)
                        .max()
                        .unwrap_or(0);
                    if max_idx + 1 > MTL4_MAX_BUFFER_BINDS {
                        return None;
                    }
                    let desc = MTL4ArgumentTableDescriptor::new();
                    desc.setMaxBufferBindCount(max_idx + 1);
                    let table = device
                        .newArgumentTableWithDescriptor_error(&desc)
                        .ok()?;
                    for (buf, off, idx) in cmd_bindings {
                        // GPU virtual address + caller-supplied byte
                        // offset; the MTL3 ICB path's
                        // `set_buffer_offset_atIndex` does the
                        // equivalent under the hood.
                        let addr = buf.gpuAddress() + *off;
                        unsafe {
                            table.setAddress_atIndex(addr, *idx as usize);
                        }
                    }
                    tables.push(table);
                }
                let m_scaling_safe = direct_m_scaling.iter().all(|s| s.is_none());
                let icb = if std::env::var_os("FERRITE_METAL_ICB").is_some()
                    && m_scaling_safe
                {
                    build_icb_for_step(device, pipeline, direct_bindings, direct_dispatch)
                } else {
                    None
                };
                out.push(Mtl4Step {
                    kernel: *kernel,
                    pipeline: pipeline.clone(),
                    tables,
                    dispatches: direct_dispatch.clone(),
                    m_scaling: direct_m_scaling.clone(),
                    barrier_before: barrier_before.clone(),
                    icb,
                });
            }
        }
    }
    Some(out)
}

/// Pre-record N concurrent-dispatch commands into one ICB, one per
/// `direct_dispatch[i]`. `direct_bindings[i]` becomes the i-th
/// command's per-buffer set-kernel-buffer calls.
///
/// Pipelines must have been created with the MTL4 compiler flow +
/// `MTL4IndirectCommandBufferSupportState::Enabled` (see
/// `specialized_pipeline_cache.rs`); the legacy MTL3 bool flag
/// produces ICB-incompatible state under MTL4 encoders.
fn build_icb_for_step(
    device: &Device,
    pipeline: &ComputePipelineState,
    direct_bindings: &[Vec<(super::__re::Buffer, u64, u64)>],
    direct_dispatch: &[(MTLSize, MTLSize)],
) -> Option<RecordingContext> {
    use ferrite_metal_kernels::instruction_executor::icb_ffi::{
        IndirectCommandBuffer, IndirectCommandBufferDescriptor, MTLIndirectCommandType,
    };
    use std::sync::Arc;

    let n = direct_dispatch.len();
    if n == 0 {
        return None;
    }
    let descriptor = IndirectCommandBufferDescriptor::new();
    descriptor.set_command_types(MTLIndirectCommandType::ConcurrentDispatch as u64);
    descriptor.set_max_kernel_buffer_bind_count(MTL4_MAX_BUFFER_BINDS as u64);
    descriptor.set_inherit_pipeline_state(true);
    descriptor.set_inherit_buffers(false);
    let icb = IndirectCommandBuffer::new(device, &descriptor, n as u64, 0).ok()?;

    let mut ctx = RecordingContext {
        device: Arc::new(device.clone()),
        icb,
        command_index: 0,
    };
    let _ = pipeline; // pipeline is set on the encoder before executeCommandsInBuffer.
    for (cmd_bindings, (tg, tpt)) in direct_bindings.iter().zip(direct_dispatch.iter()) {
        let cmd = ctx.icb.indirect_compute_command_at(ctx.command_index as u64);
        for (buf, off, idx) in cmd_bindings {
            let buf_ptr: *mut ::objc2::runtime::AnyObject =
                Retained::as_ptr(buf) as *const ::objc2::runtime::AnyObject as *mut _;
            cmd.set_kernel_buffer(buf_ptr, *off, *idx);
        }
        cmd.concurrent_dispatch_threadgroups(*tg, *tpt);
        ctx.command_index += 1;
    }
    Some(ctx)
}
