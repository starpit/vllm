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
use std::ffi::c_void;
use std::ptr::NonNull;
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

// ─────────────────────────────────────────────────────────────────
// Decode-bucket qmv dispatcher — port of `quantized.cpp:1365
// dispatch_qmv` + `:177 qmv_quad` + `:235 qmv` (which itself picks
// qmv_fast vs qmv).
// ─────────────────────────────────────────────────────────────────

/// Picked qmv variant for a given `(M, N, K, bits)` shape.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum QmvKernel {
    /// `affine_qmv_quad_*_d_<d>_*` — K must equal 64 or 128, bits must
    /// be a power of two. Most efficient on tiny K (e.g. head_dim
    /// projections). MLX `qmv_quad`.
    Quad { d: u32 },
    /// `affine_qmv_fast_*` — `N % 8 == 0 && K % 512 == 0`. The decode
    /// hot path for Llama / Qwen / Gemma. MLX `qmv_fast`.
    Fast,
    /// `affine_qmv_*` — generic fallback with bounds-checked tail.
    Generic,
}

/// Pick the right qmv variant per MLX `dispatch_qmv` (`quantized.cpp:1365`)
/// followed by the inner `qmv_fast` vs `qmv` choice (`:259`).
///
/// Mirrors the C++ exactly:
///
/// ```text
/// // dispatch_qmv:
/// if ((K == 128 || K == 64) && is_power_of_2(bits)) → qmv_quad(d=K)
/// else                                              → qmv(...)
/// // qmv():
/// bool fast = N % bn == 0 && K % 512 == 0;  // bn = 8
/// kernel = fast ? "qmv_fast" : "qmv";
/// ```
pub fn pick_qmv_kernel(n: u32, k: u32, bits: u32) -> QmvKernel {
    let pow2_bits = bits != 0 && (bits & (bits - 1)) == 0;
    if (k == 64 || k == 128) && pow2_bits {
        QmvKernel::Quad { d: k }
    } else if n % 8 == 0 && k % 512 == 0 {
        QmvKernel::Fast
    } else {
        QmvKernel::Generic
    }
}

/// Threadgroup grid + threads-per-group for a picked qmv variant.
///
/// `qmv_quad`: `bn = quads_per_simd * results_per_quadgroup = 8 * 8 = 64`
/// (`quantized.cpp:193-198`); group is one simdgroup
/// (`(simdgroup_size=32, 1, 1)`).
///
/// `qmv` / `qmv_fast`: `bn = 8`, `bk = 32`, group `(bk=32, 2, 1)` —
/// 2 simdgroups (`quantized.cpp:251-254`).
pub fn qmv_dispatch_shape(kernel: QmvKernel, m: u32, n: u32, b: u32) -> ((u32, u32, u32), (u32, u32, u32)) {
    match kernel {
        QmvKernel::Quad { .. } => {
            let bn: u32 = 64;
            ((m, n.div_ceil(bn), b), (32, 1, 1))
        }
        QmvKernel::Fast | QmvKernel::Generic => {
            let bn: u32 = 8;
            ((m, n.div_ceil(bn), b), (32, 2, 1))
        }
    }
}

/// Format the kernel symbol name for a picked qmv variant. Matches the
/// `INST_QMV_*` macros in `shaders/quantized_qmv.metal`.
pub fn qmv_kernel_name(
    kernel: QmvKernel,
    dtype: DequantDtype,
    group_size: u32,
    bits: u32,
    batched: bool,
) -> String {
    let dtype = dtype.symbol_infix();
    let batch = if batched { 1 } else { 0 };
    match kernel {
        QmvKernel::Quad { d } => format!(
            "affine_qmv_quad_{dtype}_gs_{group_size}_b_{bits}_d_{d}_batch_{batch}",
        ),
        QmvKernel::Fast => format!(
            "affine_qmv_fast_{dtype}_gs_{group_size}_b_{bits}_batch_{batch}",
        ),
        QmvKernel::Generic => {
            format!("affine_qmv_{dtype}_gs_{group_size}_b_{bits}_batch_{batch}",)
        }
    }
}

/// MLX-affine int4 decode-matvec dispatcher. Wraps the
/// `quantized_qmv` metallib's `affine_qmv_quad / fast / generic`
/// kernels. One pipeline per `(kernel_name)` is built lazily inside the
/// `ShaderCache`.
///
/// Caller invariants:
/// - Activations `x`: `[B, M, K]` row-contiguous in `dtype`.
/// - Packed weight `w`: `[N, K / pack_factor]` U32 (`pack_factor = 32 / bits = 8`
///   for bits=4).
/// - `scales`, `biases`: `[N, K / group_size]` in `dtype`.
/// - Output `y`: `[B, M, N]` in `dtype`.
/// - `bits` ∈ {4} (P3 mandate; other bits land alongside future models).
/// - `group_size` ∈ {32, 64, 128}.
/// - `b` is the batch product `out.size() / M / N`. For non-batched
///   matmul `(B == 1)`, the kernel uses the `batched=0` instantiation
///   and the buffer-7..14 batch metadata bindings are skipped (matching
///   MLX's `add_strides_and_shapes(skip = B <= 1, ...)` at
///   `quantized.cpp:128`).
pub struct MetalAffineQmv {
    shader_cache: Arc<ShaderCache>,
}

impl MetalAffineQmv {
    pub fn new(device: Device) -> Result<Self, MetalStreamError> {
        Ok(Self {
            shader_cache: Arc::new(ShaderCache::new(device)?),
        })
    }

    pub fn with_shader_cache(shader_cache: Arc<ShaderCache>) -> Self {
        Self { shader_cache }
    }

    #[allow(clippy::too_many_arguments)]
    pub fn execute(
        &self,
        x: &Buffer,
        packed_w: &Buffer,
        scales: &Buffer,
        biases: &Buffer,
        y: &Buffer,
        m: u32,
        n: u32,
        k: u32,
        b: u32,
        group_size: u32,
        bits: u32,
        dtype: DequantDtype,
        encoder: &ComputeCommandEncoderRef,
    ) -> Result<(), MetalStreamError> {
        if bits != 4 {
            return Err(MetalStreamError::ShaderCompilationFailed(format!(
                "affine_qmv: only bits=4 is wired in P3, got bits={bits}"
            )));
        }
        if !matches!(group_size, 32 | 64 | 128) {
            return Err(MetalStreamError::ShaderCompilationFailed(format!(
                "affine_qmv: only group_size in {{32, 64, 128}} is wired, got {group_size}"
            )));
        }
        if b > 1 {
            // batched=1 instantiation requires the additional buffer-7..14
            // batch-metadata bindings ported from MLX `add_strides_and_shapes`
            // (`quantized.cpp:128`); ferrite's Linear path is non-batched
            // (B=1) on every model in the current matrix, so this is a
            // forward-compat path tracked under P13 (gather variants).
            return Err(MetalStreamError::ShaderCompilationFailed(format!(
                "affine_qmv: batched=1 (B={b}) not yet wired (decode-only B=1 in P3)"
            )));
        }

        let kernel = pick_qmv_kernel(n, k, bits);
        if let QmvKernel::Quad { d } = kernel {
            // qmv_quad is templated on D = K; only K∈{64,128} are
            // instantiated. The picker already enforces this — so a
            // mismatch here is a programmer error, not a user error.
            debug_assert!(d == 64 || d == 128);
        }
        let kernel_name = qmv_kernel_name(kernel, dtype, group_size, bits, false);
        let pipeline = self.shader_cache.get_pipeline(&kernel_name)?;
        encoder.setComputePipelineState(&pipeline);

        // Buffer bindings 0-4 — packed weight, scales, biases, x, y.
        // SAFETY: all buffer pointers are valid `Retained` objects;
        // `setBuffer:offset:atIndex:` only borrows them through the call.
        unsafe {
            encoder.setBuffer_offset_atIndex(Some(packed_w), 0, 0);
            encoder.setBuffer_offset_atIndex(Some(scales), 0, 1);
            encoder.setBuffer_offset_atIndex(Some(biases), 0, 2);
            encoder.setBuffer_offset_atIndex(Some(x), 0, 3);
            encoder.setBuffer_offset_atIndex(Some(y), 0, 4);
        }

        // Buffer 5 = in_vec_size (K), buffer 6 = out_vec_size (N).
        // Both are `int` (32-bit signed) per the kernel signature; we
        // pass i32 to avoid silent reinterpretation if N or K ever
        // exceeds 2^31 (which they won't for any realistic LLM, but
        // matching MLX's signed encoding keeps the bit-pattern stable).
        let in_vec_size = k as i32;
        let out_vec_size = n as i32;
        unsafe {
            encoder.setBytes_length_atIndex(
                NonNull::new(&in_vec_size as *const i32 as *mut c_void).unwrap(),
                std::mem::size_of::<i32>(),
                5,
            );
            encoder.setBytes_length_atIndex(
                NonNull::new(&out_vec_size as *const i32 as *mut c_void).unwrap(),
                std::mem::size_of::<i32>(),
                6,
            );
        }

        // Buffers 7..14 are the batch-metadata block; skipped for B==1
        // because the kernel's `if (batched) { adjust_matrix_offsets... }`
        // dead-code-eliminates with `batched=0`, leaving these bindings
        // unread (matching MLX `add_strides_and_shapes` early-out).

        let (tg, tpg) = qmv_dispatch_shape(kernel, m, n, b);
        let threadgroups = MTLSize {
            width: tg.0 as usize,
            height: tg.1 as usize,
            depth: tg.2 as usize,
        };
        let threads_per_threadgroup = MTLSize {
            width: tpg.0 as usize,
            height: tpg.1 as usize,
            depth: tpg.2 as usize,
        };
        encoder.dispatchThreadgroups_threadsPerThreadgroup(threadgroups, threads_per_threadgroup);
        Ok(())
    }
}

// ─────────────────────────────────────────────────────────────────
// `get_qmv_batch_limit` — port of `quantized.cpp:84`. Decides where
// the matvec / matmul boundary sits per arch generation. Used by
// `lower_one` to choose between `Instruction::AffineQmm` (matvec
// branch) and `Instruction::Gemm`-equivalent (matmul branch).
// ─────────────────────────────────────────────────────────────────

/// Vector-vs-matrix limit for a given `(K, N, arch_gen)`. M < limit
/// routes to `qmv*`; M >= limit routes to `qmm*` (P4).
///
/// MLX models the `arch_size` ('d' for desktop variants like M3 Ultra
/// vs. anything else) — ferrite's `MetalTargetProfile` doesn't track
/// that today, so we conservatively use the non-'d' branch (smaller
/// limits, more aggressive matvec routing). This matches MacBook Pro
/// M3/M4 Pro/Max behavior; M3/M4 Ultra would over-route to matvec
/// versus MLX, which is correct (qmv kernels handle small M fine —
/// just slightly less efficient than qmm at the high-M boundary).
pub fn get_qmv_batch_limit(
    k: u32,
    n: u32,
    arch_gen: ferrite_metal_targets::AppleSiliconGen,
) -> u32 {
    use ferrite_metal_targets::AppleSiliconGen as G;
    match arch_gen {
        G::M1 | G::M2 => {
            if k <= 2048 && n <= 2048 {
                14
            } else if k <= 4096 && n <= 4096 {
                10
            } else {
                6
            }
        }
        G::M3 | G::M4 => {
            if k <= 2048 && n <= 2048 {
                18
            } else if k <= 4096 && n <= 4096 {
                12
            } else {
                10
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ferrite_metal_targets::AppleSiliconGen;

    #[test]
    fn qmv_kernel_pick_matches_mlx_dispatch_qmv() {
        // K==64 + pow2 bits → quad
        assert_eq!(
            pick_qmv_kernel(2048, 64, 4),
            QmvKernel::Quad { d: 64 }
        );
        assert_eq!(
            pick_qmv_kernel(2048, 128, 4),
            QmvKernel::Quad { d: 128 }
        );
        // K==96 → fast/generic, not quad
        assert_eq!(pick_qmv_kernel(2048, 96, 4), QmvKernel::Generic);
        // bits=3 (not power of 2) at K=64 → not quad
        assert_eq!(pick_qmv_kernel(2048, 64, 3), QmvKernel::Generic);

        // N%8==0 && K%512==0 → fast (Llama-1B q_proj: K=2048, N=2048)
        assert_eq!(pick_qmv_kernel(2048, 2048, 4), QmvKernel::Fast);
        // Llama-1B kv_proj: K=2048, N=512 → fast
        assert_eq!(pick_qmv_kernel(512, 2048, 4), QmvKernel::Fast);
        // Llama-1B gate/up_proj: K=2048, N=8192 → fast
        assert_eq!(pick_qmv_kernel(8192, 2048, 4), QmvKernel::Fast);
        // Llama-1B down_proj: K=8192, N=2048 → fast
        assert_eq!(pick_qmv_kernel(2048, 8192, 4), QmvKernel::Fast);

        // K%512!=0 → generic
        assert_eq!(pick_qmv_kernel(2048, 1024, 4), QmvKernel::Fast);
        assert_eq!(pick_qmv_kernel(2048, 1023, 4), QmvKernel::Generic);
        // N%8!=0 → generic
        assert_eq!(pick_qmv_kernel(2049, 2048, 4), QmvKernel::Generic);
    }

    #[test]
    fn qmv_dispatch_shape_matches_mlx_grid_dims() {
        // qmv_quad: bn = 64
        let ((tx, ty, tz), (gx, gy, gz)) =
            qmv_dispatch_shape(QmvKernel::Quad { d: 64 }, 1, 2048, 1);
        assert_eq!((tx, ty, tz), (1, 2048u32.div_ceil(64), 1));
        assert_eq!((gx, gy, gz), (32, 1, 1));

        // qmv_fast: bn = 8
        let ((tx, ty, tz), (gx, gy, gz)) = qmv_dispatch_shape(QmvKernel::Fast, 1, 2048, 1);
        assert_eq!((tx, ty, tz), (1, 2048u32.div_ceil(8), 1));
        assert_eq!((gx, gy, gz), (32, 2, 1));

        // qmv_generic shares qmv_fast's grid
        let ((tx, ty, tz), _) = qmv_dispatch_shape(QmvKernel::Generic, 1, 2049, 1);
        assert_eq!((tx, ty, tz), (1, 2049u32.div_ceil(8), 1));
    }

    #[test]
    fn qmv_kernel_name_matches_metallib_symbols() {
        // Decoded against the actual exported symbols in
        // shaders/quantized_qmv.metal's INST_QMV_* macros.
        assert_eq!(
            qmv_kernel_name(QmvKernel::Fast, DequantDtype::Bf16, 64, 4, false),
            "affine_qmv_fast_bf16_gs_64_b_4_batch_0"
        );
        assert_eq!(
            qmv_kernel_name(QmvKernel::Generic, DequantDtype::F16, 32, 4, true),
            "affine_qmv_f16_gs_32_b_4_batch_1"
        );
        assert_eq!(
            qmv_kernel_name(QmvKernel::Quad { d: 128 }, DequantDtype::Bf16, 128, 4, false),
            "affine_qmv_quad_bf16_gs_128_b_4_d_128_batch_0"
        );
    }

    #[test]
    fn qmv_batch_limit_matches_mlx_table() {
        // M3+ default branch (non-'d'): 18, 12, 10 per (K, N) buckets
        assert_eq!(get_qmv_batch_limit(2048, 2048, AppleSiliconGen::M3), 18);
        assert_eq!(get_qmv_batch_limit(4096, 4096, AppleSiliconGen::M3), 12);
        assert_eq!(get_qmv_batch_limit(8192, 8192, AppleSiliconGen::M3), 10);
        assert_eq!(get_qmv_batch_limit(2048, 2048, AppleSiliconGen::M4), 18);

        // M1/M2 branch: 14, 10, 6
        assert_eq!(get_qmv_batch_limit(2048, 2048, AppleSiliconGen::M1), 14);
        assert_eq!(get_qmv_batch_limit(4096, 4096, AppleSiliconGen::M2), 10);
        assert_eq!(get_qmv_batch_limit(8192, 8192, AppleSiliconGen::M1), 6);
    }
}
