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

use ferrite_metal_kernels::specialized_pipeline_cache::{
    ConstantValue, PipelineKey, SpecializedPipelineCache,
};
use ferrite_metal_kernels::stream::MetalStreamError;
use ferrite_wavefront::subtile_ir::{
    Binding, ConstValue, Executor, FlagId, FnConst, Grid, OpKind, PipeId, SubtileIr,
};

use super::__re::{Buffer, ComputePipelineState};

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
