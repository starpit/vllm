// Copyright © 2024 Apple Inc.
// SPDX-License-Identifier: Apache-2.0

//! MLX-affine int4 dequantization dispatcher.
//!
//! Wraps the `affine_dequantize_*_gs_*_b_4` kernels in
//! `shaders/quantized_dequantize.metal` (faithful port of
//! `mlx/backend/metal/kernels/quantized.h:2536`).
//!
//! Mirrors `mlx/backend/metal/quantized.cpp:1657
//! fast::Quantize::eval_gpu`'s dequantize path:
//!
//! ```text
//!   constexpr int simd_size = 32;                       // unused for dequant
//!   int packs_per_int = 8 / bits;                       // 2 for bits=4
//!   size_t nthreads = out.size() / packs_per_int;       // = n_bytes
//!   auto grid_shape = w.shape();
//!   grid_shape.back() *= uint8_per_uint32;              // u32 → bytes
//!   compute_encoder.dispatch_threads(grid_dims, group_dims);
//! ```
//!
//! Bits = 4 only (the P2 mandate per `INT4_PARITY_PLAN.md`).
//! Other bits land alongside the qmv/qmm kernels in P3+.

use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_metal::{
    MTLBuffer, MTLComputeCommandEncoder, MTLComputePipelineState, MTLDevice, MTLSize,
};
use std::sync::Arc;

use crate::shader_cache::ShaderCache;
use crate::stream::MetalStreamError;

pub type Buffer = Retained<ProtocolObject<dyn MTLBuffer>>;
pub type Device = Retained<ProtocolObject<dyn MTLDevice>>;
pub type ComputeCommandEncoderRef = ProtocolObject<dyn MTLComputeCommandEncoder>;

/// Output dtype the dequant kernel produces. Picks between the
/// `affine_dequantize_f16_*` and `affine_dequantize_bf16_*` symbol
/// families.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DequantDtype {
    F16,
    Bf16,
}

impl DequantDtype {
    fn symbol_infix(self) -> &'static str {
        match self {
            Self::F16 => "f16",
            Self::Bf16 => "bf16",
        }
    }

    pub fn elem_size(self) -> usize {
        2
    }
}

/// MLX-affine int4 dequantizer. Holds a `ShaderCache` that lazily
/// builds one pipeline per `(dtype, group_size)` instantiation.
pub struct MetalAffineDequantize {
    shader_cache: Arc<ShaderCache>,
}

impl MetalAffineDequantize {
    pub fn new(device: Device) -> Result<Self, MetalStreamError> {
        Ok(Self {
            shader_cache: Arc::new(ShaderCache::new(device)?),
        })
    }

    pub fn with_shader_cache(shader_cache: Arc<ShaderCache>) -> Self {
        Self { shader_cache }
    }

    /// Dispatch `affine_dequantize_<dtype>_gs_<gs>_b_<bits>` against an
    /// open `ComputeCommandEncoder`. Caller owns the encoder + commit
    /// lifecycle; this function only emits the kernel bindings + grid.
    ///
    /// - `packed_weight`: `[N, K / 8]` U32, treated as `[N * K / 2]`
    ///   bytes by the kernel (`buffer(0)`).
    /// - `scales`, `biases`: `[N * K / group_size]` half-precision
    ///   per-group affine parameters (`buffer(1)` / `buffer(2)`).
    /// - `output`: `[N, K]` half-precision dequantized tile
    ///   (`buffer(3)`).
    /// - `out_n_elements` = `N * K`. Caller is responsible for
    ///   `output.length() >= out_n_elements * dtype.elem_size()`.
    /// - `group_size`: must be one of {32, 64, 128} (the instantiations
    ///   in `quantized_dequantize.metal`).
    /// - `bits`: must equal 4 in P2.
    #[allow(clippy::too_many_arguments)]
    pub fn execute(
        &self,
        packed_weight: &Buffer,
        scales: &Buffer,
        biases: &Buffer,
        output: &Buffer,
        out_n_elements: u64,
        group_size: u32,
        bits: u32,
        dtype: DequantDtype,
        encoder: &ComputeCommandEncoderRef,
    ) -> Result<(), MetalStreamError> {
        if bits != 4 {
            return Err(MetalStreamError::ShaderCompilationFailed(format!(
                "affine_dequantize: only bits=4 is wired in P2, got bits={bits}"
            )));
        }
        if !matches!(group_size, 32 | 64 | 128) {
            return Err(MetalStreamError::ShaderCompilationFailed(format!(
                "affine_dequantize: only group_size in {{32, 64, 128}} is wired in P2, got {group_size}"
            )));
        }
        // packs_per_int = 8 / bits = 2 for bits=4. nthreads = total output
        // element count / packs_per_int = output bytes / 2.
        let packs_per_int: u64 = 2;
        if out_n_elements % packs_per_int != 0 {
            return Err(MetalStreamError::ShaderCompilationFailed(format!(
                "affine_dequantize: out_n_elements={out_n_elements} not divisible by \
                 packs_per_int={packs_per_int} (bits={bits})"
            )));
        }
        let nthreads = out_n_elements / packs_per_int;

        let kernel_name = format!(
            "affine_dequantize_{}_gs_{}_b_{}",
            dtype.symbol_infix(),
            group_size,
            bits
        );
        let pipeline = self.shader_cache.get_pipeline(&kernel_name)?;
        encoder.setComputePipelineState(&pipeline);

        // SAFETY: the buffer pointers are non-null `Retained` objects;
        // setBuffer:offset:atIndex: only borrows them through the call.
        unsafe {
            encoder.setBuffer_offset_atIndex(Some(packed_weight), 0, 0);
            encoder.setBuffer_offset_atIndex(Some(scales), 0, 1);
            encoder.setBuffer_offset_atIndex(Some(biases), 0, 2);
            encoder.setBuffer_offset_atIndex(Some(output), 0, 3);
        }

        // Match MLX's 2D grid (`get_2d_grid_dims` over `w.shape()` with
        // `back() *= uint8_per_uint32`). For our shapes nthreads always
        // fits in u32, so a 1D grid is correct; the kernel only reads
        // `index.x + grid_dim.x * index.y`, which is just `index.x`
        // when `grid_dim.y == 1`.
        if nthreads > u32::MAX as u64 {
            return Err(MetalStreamError::ShaderCompilationFailed(format!(
                "affine_dequantize: nthreads={nthreads} exceeds u32; \
                 2D grid wrapping not yet implemented (P2 mandate covers shapes ≤ 4G threads)"
            )));
        }
        let threads_per_threadgroup_x: u64 = nthreads.min(
            pipeline
                .maxTotalThreadsPerThreadgroup()
                .min(u32::MAX as usize) as u64,
        );
        let threads_per_threadgroup = MTLSize {
            width: threads_per_threadgroup_x as usize,
            height: 1,
            depth: 1,
        };
        let threadgroups = MTLSize {
            width: (nthreads as usize).div_ceil(threads_per_threadgroup_x as usize),
            height: 1,
            depth: 1,
        };

        encoder.dispatchThreadgroups_threadsPerThreadgroup(threadgroups, threads_per_threadgroup);
        Ok(())
    }
}
