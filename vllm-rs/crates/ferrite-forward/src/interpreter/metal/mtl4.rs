// SPDX-License-Identifier: Apache-2.0
//! Phase A.2/A.3 MTL4 bake artifacts + step type.
//!
//! See `FERRITE_METAL_MTL4_MIGRATION.md`. One `Mtl4Step` is produced
//! per `BucketStep::Icb` from a bucket's plan, holding the same
//! pipeline state the MTL3 ICB path uses (MTL4's
//! `setComputePipelineState` takes `dyn MTLComputePipelineState`),
//! plus a pre-built `MTL4ArgumentTable` per coalesced sub-command
//! (one per entry in `direct_bindings`).
//!
//! `BucketStep::Gemm` (MPS f16 fast-path) is not representable on
//! the MTL4 encoder surface yet, so any bucket containing a `Gemm`
//! step falls back to MTL3 — `bake_mtl4_steps` returns `None` in
//! that case and the pool's `FERRITE_METAL_MTL4` arm declines to
//! take the bucket. Production int4 / bf16 paths emit only `Icb`
//! steps, so the fallback is a paper safety net for the f16 path.

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
///
/// `barrier_before[i]` is the Phase-C hazard flag for sub-dispatch
/// `i`: `true` means the runtime must emit an encoder-local
/// `Dispatch→Dispatch` barrier *before* this dispatch fires
/// (MTL4 does NOT inherit MTL3's default-Serial encoder
/// auto-serialization). Computed at bake time from the
/// compile-time DAG (`LoweredCommand.{output_arena_slots,
/// input_arena_slots, writes_kv_layer, reads_kv_layer}`).
pub struct Mtl4Step {
    pub pipeline: ComputePipelineState,
    pub tables: Vec<Retained<ProtocolObject<dyn MTL4ArgumentTable>>>,
    pub dispatches: Vec<(MTLSize, MTLSize)>,
    pub barrier_before: Vec<bool>,
}

/// One sub-command's dataflow signature, derived at lowering time
/// from the `Instruction<W>` variant and threaded through bake.
///
/// Arena slots are the unit of arena-side hazard tracking (one
/// slot ↔ one arena buffer). KV layers are tracked separately
/// because the kv cache buffers are runtime bindings (not arena
/// slots) yet still need cross-dispatch ordering.
#[derive(Clone, Default)]
pub struct Dataflow {
    pub writes_arena: Vec<u32>,
    pub reads_arena: Vec<u32>,
    pub writes_kv_layer: Option<u32>,
    pub reads_kv_layer: Option<u32>,
}

/// Build one `Mtl4Step` per `BucketStep::Icb`. Returns `None` if any
/// step is a `Gemm` (MPS) or any kernel exceeds the argument-table
/// binding cap — the caller falls back to the MTL3 path for that
/// bucket.
pub fn bake_mtl4_steps(
    steps: &[BucketStep],
    step_dataflows: &[Vec<Dataflow>],
    device: &Device,
) -> Option<Vec<Mtl4Step>> {
    debug_assert_eq!(steps.len(), step_dataflows.len());
    // Up-front Gemm-step bail: MPS-routed GEMMs aren't representable
    // on the MTL4 encoder surface yet.
    for step in steps {
        if matches!(step, BucketStep::Gemm { .. }) {
            return None;
        }
    }
    let mut out = Vec::with_capacity(steps.len());
    // Phase-C hazard tracking carried across step boundaries (same
    // encoder for the whole bucket).
    let mut pending_writes_arena: std::collections::HashSet<u32> =
        std::collections::HashSet::new();
    let mut pending_reads_arena: std::collections::HashSet<u32> =
        std::collections::HashSet::new();
    let mut pending_writes_kv: std::collections::HashSet<u32> =
        std::collections::HashSet::new();
    let mut pending_reads_kv: std::collections::HashSet<u32> =
        std::collections::HashSet::new();
    let mut first = true;
    for (step, dataflows) in steps.iter().zip(step_dataflows.iter()) {
        let BucketStep::Icb {
            pipeline,
            direct_bindings,
            direct_dispatch,
            ..
        } = step
        else {
            unreachable!("Gemm filtered above");
        };
        debug_assert_eq!(direct_bindings.len(), direct_dispatch.len());
        debug_assert_eq!(direct_bindings.len(), dataflows.len());
        let mut tables = Vec::with_capacity(direct_bindings.len());
        let mut barrier_before = Vec::with_capacity(direct_bindings.len());
        for (cmd_bindings, df) in direct_bindings.iter().zip(dataflows.iter()) {
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
            let table = device.newArgumentTableWithDescriptor_error(&desc).ok()?;
            for (buf, off, idx) in cmd_bindings {
                let addr = buf.gpuAddress() + *off;
                unsafe {
                    table.setAddress_atIndex(addr, *idx as usize);
                }
            }
            tables.push(table);
            // Hazard check. RAW: my reads vs prior writes. WAW: my
            // writes vs prior writes. WAR: my writes vs prior reads.
            // KV: separate per-layer set.
            let arena_conflict = df.reads_arena.iter().any(|s| pending_writes_arena.contains(s))
                || df.writes_arena.iter().any(|s| pending_writes_arena.contains(s))
                || df.writes_arena.iter().any(|s| pending_reads_arena.contains(s));
            let kv_conflict = df
                .reads_kv_layer
                .map(|l| pending_writes_kv.contains(&l))
                .unwrap_or(false)
                || df
                    .writes_kv_layer
                    .map(|l| pending_writes_kv.contains(&l) || pending_reads_kv.contains(&l))
                    .unwrap_or(false);
            // Phase C is opt-in via FERRITE_METAL_MTL4_SELECTIVE_BARRIERS=1
            // until the compile-time DAG → barrier_before mapping is
            // verified to match MTL3 output at greedy temp across all
            // models. Default: conservative — barrier between every
            // pair of dispatches (matches the Phase A behavior).
            let selective = std::env::var_os("FERRITE_METAL_MTL4_SELECTIVE_BARRIERS").is_some();
            let need = if selective {
                !first && (arena_conflict || kv_conflict)
            } else {
                !first
            };
            if need {
                pending_writes_arena.clear();
                pending_reads_arena.clear();
                pending_writes_kv.clear();
                pending_reads_kv.clear();
            }
            for s in &df.writes_arena {
                pending_writes_arena.insert(*s);
            }
            for s in &df.reads_arena {
                pending_reads_arena.insert(*s);
            }
            if let Some(l) = df.writes_kv_layer {
                pending_writes_kv.insert(l);
            }
            if let Some(l) = df.reads_kv_layer {
                pending_reads_kv.insert(l);
            }
            barrier_before.push(need);
            first = false;
        }
        out.push(Mtl4Step {
            pipeline: pipeline.clone(),
            tables,
            dispatches: direct_dispatch.clone(),
            barrier_before,
        });
    }
    Some(out)
}
