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
}

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
                out.push(Mtl4Step {
                    pipeline: pipeline.clone(),
                    tables,
                    dispatches: direct_dispatch.clone(),
                    m_scaling: direct_m_scaling.clone(),
                    barrier_before: barrier_before.clone(),
                });
            }
        }
    }
    Some(out)
}
