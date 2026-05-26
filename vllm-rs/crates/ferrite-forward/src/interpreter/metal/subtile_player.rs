// SPDX-License-Identifier: Apache-2.0
//! GPU subtile tape player — the TRIVIAL [`Executor`] backend for a
//! [`ferrite_wavefront::subtile_ir::SubtileIr`].
//!
//! All intelligence (kernel pick, N-block offsets, grid, deps) is baked
//! into the IR by the compiler. This module only:
//!   1. resolves the IR's pipeline table to compute pipeline states
//!      ([`resolve_pipelines`], called once per IR), and
//!   2. binds + dispatches each `Run` into a serial compute encoder
//!      ([`MetalExecutor`]).
//!
//! `Wait`/`Signal` are no-ops here: a serial `MTLComputeCommandEncoder`
//! auto-serializes dependent dispatches in submission order, so the
//! single-stream tape's data hazards are covered by ordering. The p2p
//! flags carry real waits only when the device megakernel runs workers
//! concurrently.
//!
//! The on-device proof (N-blocked qmv == whole qmv, bit-exact) lives in
//! `tests/subtile_player_gpu.rs` as an integration test — it must NOT be
//! a lib `#[cfg(test)]` module, because the lib's test build currently
//! has unrelated bit-rot (`pipelines.rs` test-only `constants_for`
//! non-exhaustive matches) that an integration test sidesteps.
#![cfg(feature = "metal")]

use ::objc2::runtime::ProtocolObject;
use ::objc2_metal::{MTLComputeCommandEncoder, MTLSize};

use ferrite_cuda_core::MetalAllocator;
use ferrite_metal_kernels::specialized_pipeline_cache::{
    ConstantValue, PipelineKey, SpecializedPipelineCache,
};
use ferrite_metal_kernels::stream::MetalStreamError;
use ferrite_wavefront::subtile_ir::{
    Binding, BufferRef, ConstValue, Executor, FlagId, FnConst, Grid, InputKind, OpKind, PipeId,
    SubtileIr, WeightBundle, WeightRole,
};

use super::__re::{Buffer, ComputePipelineState};
use super::ids::LayerId;
use super::lowered::{RuntimeBindingKind, WeightBundleKind, WeightLocator, WeightTensor};
use super::runtime::RuntimeBindings;
use super::worker::{WorkerError, resolve_weight};
use crate::{CanonicalParams, WeightAccessors};

/// The resolved form of a [`ferrite_wavefront::subtile_ir::BufferRef`]:
/// the concrete GPU buffer plus the base byte-offset of the whole
/// tensor. Indexed by `BufId`. A binding's carried `offset` is added on
/// top at dispatch time.
pub type ResolvedBuffer = (Buffer, u64);

fn const_to_metal(c: &FnConst) -> ConstantValue {
    // `ConstSlot: From<u16>` — function-constant indices are small.
    let i = c.index as u16;
    match c.value {
        ConstValue::U32(v) => ConstantValue::uint(i, v),
        ConstValue::I32(v) => ConstantValue::int(i, v),
        ConstValue::F32(v) => ConstantValue::float(i, v),
        ConstValue::Bool(v) => ConstantValue::boolean(i, v),
    }
}

/// Resolve every `PipelineSpec` in `ir` to a compute pipeline state via
/// the specialized-pipeline cache, in `PipeId` order. Call ONCE per IR:
/// the symbols are leaked to `'static` for the cache key, and a decode
/// IR has only a handful of distinct pipelines (one per kernel × block
/// width), all process-lifetime.
pub fn resolve_pipelines(
    ir: &SubtileIr,
    cache: &SpecializedPipelineCache,
) -> Result<Vec<ComputePipelineState>, MetalStreamError> {
    ir.pipelines
        .iter()
        .map(|spec| {
            let symbol: &'static str = Box::leak(spec.symbol.clone().into_boxed_str());
            let constants: Vec<ConstantValue> = spec.constants.iter().map(const_to_metal).collect();
            let key = PipelineKey::new(spec.library, symbol, constants);
            cache.get_or_build(&key)
        })
        .collect()
}

fn mtlsize(d: [u32; 3]) -> MTLSize {
    MTLSize {
        width: d[0] as usize,
        height: d[1] as usize,
        depth: d[2] as usize,
    }
}

/// The trivial GPU player backend. Holds pre-resolved buffer + pipeline
/// tables and borrows one serial compute encoder. `run` is exactly
/// {set pipeline, bind each buffer at base+offset, dispatch grid} — no
/// decisions, no per-instruction resolution.
pub struct MetalExecutor<'a> {
    enc: &'a ProtocolObject<dyn MTLComputeCommandEncoder>,
    buffers: &'a [ResolvedBuffer],
    pipelines: &'a [ComputePipelineState],
}

impl<'a> MetalExecutor<'a> {
    pub fn new(
        enc: &'a ProtocolObject<dyn MTLComputeCommandEncoder>,
        buffers: &'a [ResolvedBuffer],
        pipelines: &'a [ComputePipelineState],
    ) -> Self {
        Self {
            enc,
            buffers,
            pipelines,
        }
    }
}

impl Executor for MetalExecutor<'_> {
    fn run(&mut self, _op: OpKind, pipeline: PipeId, bindings: &[Binding], grid: Grid) {
        self.enc
            .setComputePipelineState(&self.pipelines[pipeline.0 as usize]);
        for b in bindings {
            let (buf, base) = &self.buffers[b.buffer.0 as usize];
            unsafe {
                self.enc.setBuffer_offset_atIndex(
                    Some(buf),
                    (*base + b.offset) as usize,
                    b.index as usize,
                );
            }
        }
        self.enc
            .dispatchThreadgroups_threadsPerThreadgroup(mtlsize(grid.tg), mtlsize(grid.tpt));
    }

    fn wait(&mut self, _flag: FlagId) {}
    fn signal(&mut self, _flag: FlagId) {}
}

// ── Buffer-table resolution (BufferRef → (Buffer, base)) ────────────
//
// The inverse of the compiler's mappers. Mirrors the worker's
// `resolve_bindings`: weights via `resolve_weight`, arena slots from the
// worker arena, runtime inputs via `RuntimeBindings::buffer_for` (which
// resolves KV-cache halves etc. itself), scratch from the split-K buffer.

fn bundle_to_metal(b: WeightBundle) -> WeightBundleKind {
    match b {
        WeightBundle::RmsNorm => WeightBundleKind::RmsNorm,
        WeightBundle::Embedding => WeightBundleKind::Embedding,
        WeightBundle::LinearLayer => WeightBundleKind::LinearLayer,
        WeightBundle::CosSin => WeightBundleKind::CosSin,
        WeightBundle::AffineQuantEmbedding => WeightBundleKind::AffineQuantEmbedding,
    }
}

fn role_to_metal(r: WeightRole) -> WeightTensor {
    match r {
        WeightRole::Weight => WeightTensor::Weight,
        WeightRole::Bias => WeightTensor::Bias,
        WeightRole::AffineScales => WeightTensor::AffineScales,
        WeightRole::AffineBiases => WeightTensor::AffineBiases,
        WeightRole::AffineLinearBias => WeightTensor::AffineLinearBias,
    }
}

fn input_to_metal(k: InputKind) -> RuntimeBindingKind {
    match k {
        InputKind::InputIds => RuntimeBindingKind::InputIds,
        InputKind::Positions => RuntimeBindingKind::Positions,
        InputKind::SlotMapping => RuntimeBindingKind::SlotMapping,
        InputKind::CuSeqlensQ => RuntimeBindingKind::CuSeqlensQ,
        InputKind::SeqUsedK => RuntimeBindingKind::SeqUsedK,
        InputKind::BlockTable => RuntimeBindingKind::BlockTable,
        InputKind::KvCacheK { layer } => RuntimeBindingKind::KvCacheK {
            layer: LayerId(layer),
        },
        InputKind::KvCacheV { layer } => RuntimeBindingKind::KvCacheV {
            layer: LayerId(layer),
        },
        InputKind::NumTokens => RuntimeBindingKind::NumTokensU32,
    }
}

/// Resolve a [`SubtileIr`]'s logical buffer table to concrete
/// `(Buffer, base_offset)` in `BufId` order — the inputs the
/// [`MetalExecutor`] borrows. Call once per forward; the weight/arena
/// handles are stable and the runtime handles are the same ones the
/// normal path rebinds.
pub fn resolve_buffers<W: CanonicalParams + WeightAccessors>(
    ir: &SubtileIr,
    arena: &[Buffer],
    splitk_scratch: Option<&Buffer>,
    weights: &W,
    allocator: &MetalAllocator,
    runtime: &RuntimeBindings,
) -> Result<Vec<ResolvedBuffer>, WorkerError> {
    ir.buffers
        .iter()
        .map(|b| match b {
            BufferRef::ArenaSlot(slot) => arena
                .get(*slot as usize)
                .cloned()
                .map(|buf| (buf, 0u64))
                .ok_or(WorkerError::WeightLookupFailed {
                    reason: "wavefront: arena slot out of range",
                }),
            BufferRef::Scratch(_) => splitk_scratch.cloned().map(|buf| (buf, 0u64)).ok_or(
                WorkerError::WeightLookupFailed {
                    reason: "wavefront: split-K scratch missing",
                },
            ),
            BufferRef::Input(kind) => Ok((runtime.buffer_for(input_to_metal(*kind)).clone(), 0u64)),
            BufferRef::Weight { bundle, role, loc } => resolve_weight(
                weights,
                allocator,
                &bundle_to_metal(*bundle),
                loc.layer,
                role_to_metal(*role),
                WeightLocator {
                    bucket: loc.bucket,
                    op_idx: loc.op_idx,
                    slot: loc.slot,
                },
            ),
        })
        .collect()
}
