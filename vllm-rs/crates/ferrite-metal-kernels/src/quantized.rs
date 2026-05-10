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
use crate::specialized_pipeline_cache::ConstantValue;
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

/// `&'static str` view of [`qmv_kernel_name`] for the lowering pass.
/// `LoweredCommand::function` is `&'static str`, so the macro-emit
/// path can't allocate a `String` here. Covers the non-batched (B=1)
/// instantiations only (`batch_0`); MoE batched=1 lands with P13.
///
/// Big match arm rather than `format!`+`Box::leak` so every name is
/// visible in code review and matches a real entry in the qmv
/// `INST_QMV_BATCHED` / `INST_QMV_QUAD` instantiation table at
/// `shaders/quantized_qmv.metal`.
pub fn qmv_kernel_static_name(
    kernel: QmvKernel,
    dtype: DequantDtype,
    bits: u32,
    group_size: u32,
) -> &'static str {
    debug_assert_eq!(bits, 4, "qmv_kernel_static_name: only bits=4 is wired");
    use DequantDtype::*;
    match (kernel, dtype, group_size) {
        // ── qmv_quad: D∈{64,128} × dtype × gs ─────────────────────
        (QmvKernel::Quad { d: 64 }, F16, 32) => "affine_qmv_quad_f16_gs_32_b_4_d_64_batch_0",
        (QmvKernel::Quad { d: 64 }, F16, 64) => "affine_qmv_quad_f16_gs_64_b_4_d_64_batch_0",
        (QmvKernel::Quad { d: 64 }, F16, 128) => "affine_qmv_quad_f16_gs_128_b_4_d_64_batch_0",
        (QmvKernel::Quad { d: 64 }, Bf16, 32) => "affine_qmv_quad_bf16_gs_32_b_4_d_64_batch_0",
        (QmvKernel::Quad { d: 64 }, Bf16, 64) => "affine_qmv_quad_bf16_gs_64_b_4_d_64_batch_0",
        (QmvKernel::Quad { d: 64 }, Bf16, 128) => "affine_qmv_quad_bf16_gs_128_b_4_d_64_batch_0",
        (QmvKernel::Quad { d: 128 }, F16, 32) => "affine_qmv_quad_f16_gs_32_b_4_d_128_batch_0",
        (QmvKernel::Quad { d: 128 }, F16, 64) => "affine_qmv_quad_f16_gs_64_b_4_d_128_batch_0",
        (QmvKernel::Quad { d: 128 }, F16, 128) => "affine_qmv_quad_f16_gs_128_b_4_d_128_batch_0",
        (QmvKernel::Quad { d: 128 }, Bf16, 32) => "affine_qmv_quad_bf16_gs_32_b_4_d_128_batch_0",
        (QmvKernel::Quad { d: 128 }, Bf16, 64) => "affine_qmv_quad_bf16_gs_64_b_4_d_128_batch_0",
        (QmvKernel::Quad { d: 128 }, Bf16, 128) => "affine_qmv_quad_bf16_gs_128_b_4_d_128_batch_0",
        // ── qmv_fast ──────────────────────────────────────────────
        (QmvKernel::Fast, F16, 32) => "affine_qmv_fast_f16_gs_32_b_4_batch_0",
        (QmvKernel::Fast, F16, 64) => "affine_qmv_fast_f16_gs_64_b_4_batch_0",
        (QmvKernel::Fast, F16, 128) => "affine_qmv_fast_f16_gs_128_b_4_batch_0",
        (QmvKernel::Fast, Bf16, 32) => "affine_qmv_fast_bf16_gs_32_b_4_batch_0",
        (QmvKernel::Fast, Bf16, 64) => "affine_qmv_fast_bf16_gs_64_b_4_batch_0",
        (QmvKernel::Fast, Bf16, 128) => "affine_qmv_fast_bf16_gs_128_b_4_batch_0",
        // ── qmv generic fallback ──────────────────────────────────
        (QmvKernel::Generic, F16, 32) => "affine_qmv_f16_gs_32_b_4_batch_0",
        (QmvKernel::Generic, F16, 64) => "affine_qmv_f16_gs_64_b_4_batch_0",
        (QmvKernel::Generic, F16, 128) => "affine_qmv_f16_gs_128_b_4_batch_0",
        (QmvKernel::Generic, Bf16, 32) => "affine_qmv_bf16_gs_32_b_4_batch_0",
        (QmvKernel::Generic, Bf16, 64) => "affine_qmv_bf16_gs_64_b_4_batch_0",
        (QmvKernel::Generic, Bf16, 128) => "affine_qmv_bf16_gs_128_b_4_batch_0",
        // ── Quad with non-{64,128} D — picker should never produce ─
        (QmvKernel::Quad { d }, _, _) => panic!(
            "qmv_kernel_static_name: QmvKernel::Quad with unsupported D={d} \
             — only 64 and 128 instantiated (see INST_QMV_QUAD in quantized_qmv.metal)"
        ),
        (_, _, gs) => panic!(
            "qmv_kernel_static_name: unsupported group_size={gs} \
             — only 32, 64, 128 instantiated"
        ),
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
        // K (in_vec_size) and N (out_vec_size) ride as function
        // constants 0/1 in `quantized_qmv.metal` — see the `IN_VEC_SIZE`
        // / `OUT_VEC_SIZE` declarations at the top of that shader.
        // Both are declared `constant int` to match the MLX C++ source
        // (`int` runtime args at `quantized.h:519-520`); Metal type-
        // checks the constant payload byte-for-byte against the slot
        // declaration, so we send signed.
        let constants = [
            ConstantValue::int(0, k as i32),
            ConstantValue::int(1, n as i32),
        ];
        let pipeline = self
            .shader_cache
            .get_pipeline_specialized(&kernel_name, &constants)?;
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

// ─────────────────────────────────────────────────────────────────
// Prefill-bucket qmm_t dispatcher — port of `quantized.cpp:680 qmm()`
// (transpose=true branch) + `:774 qmm_splitk()`. Used when M >=
// vector_limit (matmul branch in `dispatch_qmv`'s outer
// `QuantizedMatmul::eval_gpu` rule).
// ─────────────────────────────────────────────────────────────────

/// Picked qmm_t variant for a given `(M, N, K, B)` shape.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum QmmTKernel {
    /// `affine_qmm_t_*_alN_<bool>_batch_0` — standard prefill matmul,
    /// transpose=true. Always fires for `B > 1` or when split_k
    /// reduces to 1. MLX `qmm` at `quantized.cpp:680`.
    Standard,
    /// `affine_qmm_t_splitk_*_alN_<bool>` — split-K variant. Fires only
    /// when `B == 1` and a non-trivial split_k is feasible (the
    /// `qmm_splitk` heuristic at `quantized.cpp:788-805` targets
    /// ~512 threadgroups; falls back to `Standard` if split_k ≤ 1).
    SplitK {
        split_k: u32,
        k_partition_size: u32,
    },
}

/// Compute the splitk plan per MLX `qmm_splitk` (`quantized.cpp:788-805`):
///   bm=bn=32, target ~512 active threadgroups → split_k = max(1, 512 / (n_tiles*m_tiles))
///   cap by K/group_size, ensure K % (split_k * group_size) == 0
///   if split_k <= 1: fall back to standard qmm_t
pub fn pick_qmm_t_split_k(m: u32, n: u32, k: u32, group_size: u32) -> u32 {
    const BM: u32 = 32;
    const BN: u32 = 32;
    let n_tiles = n.div_ceil(BN);
    let m_tiles = m.div_ceil(BM);
    let current_tgs = n_tiles * m_tiles;
    if current_tgs == 0 {
        return 1;
    }
    let mut split_k = (512u32 / current_tgs).max(1);
    let group_cap = k / group_size;
    if group_cap == 0 {
        return 1;
    }
    split_k = split_k.min(group_cap);
    while split_k > 1 && (k % (split_k * group_size) != 0) {
        split_k -= 1;
    }
    split_k
}

/// Pick the right qmm_t variant per MLX's matmul-branch routing.
/// Mirrors `quantized.cpp:1411-1424`:
///
/// ```text
/// if transpose && B == 1:                           qmm_splitk
/// else if transpose:                                qmm (transpose=true)
/// ```
pub fn pick_qmm_t_kernel(m: u32, n: u32, k: u32, b: u32, group_size: u32) -> QmmTKernel {
    if b == 1 {
        let split_k = pick_qmm_t_split_k(m, n, k, group_size);
        if split_k > 1 {
            return QmmTKernel::SplitK {
                split_k,
                k_partition_size: k / split_k,
            };
        }
    }
    QmmTKernel::Standard
}

/// Threadgroup grid + threads-per-group for a picked qmm_t variant.
///
/// `qmm_t`: bm=bn=32, wm=wn=2 → group (32, 2, 2); grid
/// `(ceil(N/32), ceil(M/32), B)` (`quantized.cpp:720-721`).
///
/// `qmm_t_splitk`: same group; grid `(n_tiles, m_tiles, split_k)`
/// (`quantized.cpp:822-824`).
pub fn qmm_t_dispatch_shape(
    kernel: QmmTKernel,
    m: u32,
    n: u32,
    b: u32,
) -> ((u32, u32, u32), (u32, u32, u32)) {
    let n_tiles = n.div_ceil(32);
    let m_tiles = m.div_ceil(32);
    match kernel {
        QmmTKernel::Standard => ((n_tiles, m_tiles, b), (32, 2, 2)),
        QmmTKernel::SplitK { split_k, .. } => ((n_tiles, m_tiles, split_k), (32, 2, 2)),
    }
}

/// Format the kernel symbol name. Matches the `INST_QMM_*` macros in
/// `shaders/quantized_qmm.metal`.
pub fn qmm_t_kernel_name(
    kernel: QmmTKernel,
    dtype: DequantDtype,
    group_size: u32,
    bits: u32,
    aligned_n: bool,
) -> String {
    let dtype = dtype.symbol_infix();
    let aln = if aligned_n { "true" } else { "false" };
    match kernel {
        QmmTKernel::Standard => format!(
            "affine_qmm_t_{dtype}_gs_{group_size}_b_{bits}_alN_{aln}_batch_0",
        ),
        QmmTKernel::SplitK { .. } => format!(
            "affine_qmm_t_splitk_{dtype}_gs_{group_size}_b_{bits}_alN_{aln}",
        ),
    }
}

/// `&'static str` view of [`qmm_t_kernel_name`] for the lowering pass
/// — see [`qmv_kernel_static_name`] for the analogous discussion.
/// `LoweredCommand::function` is `&'static str`; the table below
/// enumerates every entry the `INST_QMM_ALL` macro produces in
/// `shaders/quantized_qmm.metal`.
pub fn qmm_t_kernel_static_name(
    kernel: QmmTKernel,
    dtype: DequantDtype,
    bits: u32,
    group_size: u32,
    aligned_n: bool,
) -> &'static str {
    debug_assert_eq!(bits, 4, "qmm_t_kernel_static_name: only bits=4 is wired");
    use DequantDtype::*;
    match (kernel, dtype, group_size, aligned_n) {
        // ── qmm_t Standard ────────────────────────────────────────
        (QmmTKernel::Standard, F16, 32, true) => "affine_qmm_t_f16_gs_32_b_4_alN_true_batch_0",
        (QmmTKernel::Standard, F16, 32, false) => "affine_qmm_t_f16_gs_32_b_4_alN_false_batch_0",
        (QmmTKernel::Standard, F16, 64, true) => "affine_qmm_t_f16_gs_64_b_4_alN_true_batch_0",
        (QmmTKernel::Standard, F16, 64, false) => "affine_qmm_t_f16_gs_64_b_4_alN_false_batch_0",
        (QmmTKernel::Standard, F16, 128, true) => "affine_qmm_t_f16_gs_128_b_4_alN_true_batch_0",
        (QmmTKernel::Standard, F16, 128, false) => "affine_qmm_t_f16_gs_128_b_4_alN_false_batch_0",
        (QmmTKernel::Standard, Bf16, 32, true) => "affine_qmm_t_bf16_gs_32_b_4_alN_true_batch_0",
        (QmmTKernel::Standard, Bf16, 32, false) => "affine_qmm_t_bf16_gs_32_b_4_alN_false_batch_0",
        (QmmTKernel::Standard, Bf16, 64, true) => "affine_qmm_t_bf16_gs_64_b_4_alN_true_batch_0",
        (QmmTKernel::Standard, Bf16, 64, false) => "affine_qmm_t_bf16_gs_64_b_4_alN_false_batch_0",
        (QmmTKernel::Standard, Bf16, 128, true) => "affine_qmm_t_bf16_gs_128_b_4_alN_true_batch_0",
        (QmmTKernel::Standard, Bf16, 128, false) => "affine_qmm_t_bf16_gs_128_b_4_alN_false_batch_0",
        // ── qmm_t SplitK ──────────────────────────────────────────
        (QmmTKernel::SplitK { .. }, F16, 32, true) => "affine_qmm_t_splitk_f16_gs_32_b_4_alN_true",
        (QmmTKernel::SplitK { .. }, F16, 32, false) => "affine_qmm_t_splitk_f16_gs_32_b_4_alN_false",
        (QmmTKernel::SplitK { .. }, F16, 64, true) => "affine_qmm_t_splitk_f16_gs_64_b_4_alN_true",
        (QmmTKernel::SplitK { .. }, F16, 64, false) => "affine_qmm_t_splitk_f16_gs_64_b_4_alN_false",
        (QmmTKernel::SplitK { .. }, F16, 128, true) => "affine_qmm_t_splitk_f16_gs_128_b_4_alN_true",
        (QmmTKernel::SplitK { .. }, F16, 128, false) => "affine_qmm_t_splitk_f16_gs_128_b_4_alN_false",
        (QmmTKernel::SplitK { .. }, Bf16, 32, true) => "affine_qmm_t_splitk_bf16_gs_32_b_4_alN_true",
        (QmmTKernel::SplitK { .. }, Bf16, 32, false) => "affine_qmm_t_splitk_bf16_gs_32_b_4_alN_false",
        (QmmTKernel::SplitK { .. }, Bf16, 64, true) => "affine_qmm_t_splitk_bf16_gs_64_b_4_alN_true",
        (QmmTKernel::SplitK { .. }, Bf16, 64, false) => "affine_qmm_t_splitk_bf16_gs_64_b_4_alN_false",
        (QmmTKernel::SplitK { .. }, Bf16, 128, true) => "affine_qmm_t_splitk_bf16_gs_128_b_4_alN_true",
        (QmmTKernel::SplitK { .. }, Bf16, 128, false) => "affine_qmm_t_splitk_bf16_gs_128_b_4_alN_false",
        (_, _, gs, _) => panic!(
            "qmm_t_kernel_static_name: unsupported group_size={gs} \
             — only 32, 64, 128 instantiated"
        ),
    }
}

/// MLX-affine int4 prefill-matmul (transpose=true) dispatcher. Wraps
/// the `quantized_qmm` metallib's `affine_qmm_t` and
/// `affine_qmm_t_splitk` kernels.
///
/// Caller invariants:
/// - Activations `x`: `[M, K]` row-contiguous in `dtype`. (B=1
///   currently; batched=1 is unwired in P4 — gates on the MoE
///   `adjust_matrix_offsets` port in P13.)
/// - Packed weight `w`: `[N, K / pack_factor]` U32 (`pack_factor = 8 /
///   bits = 2` for bits=4).
/// - `scales`, `biases`: `[N, K / group_size]` in `dtype`.
/// - Output `y`: `[M, N]` in `dtype` for the `Standard` kernel; for
///   `SplitK`, caller passes an intermediate scratch of shape
///   `[split_k, M, N]` in `dtype` and is responsible for the
///   downstream sum-reduce along axis 0 (mirrors
///   `quantized.cpp:861 strided_reduce_general_dispatch`).
/// - `bits` ∈ {4} (P4 mandate).
/// - `group_size` ∈ {32, 64, 128}.
pub struct MetalAffineQmmT {
    shader_cache: Arc<ShaderCache>,
}

impl MetalAffineQmmT {
    pub fn new(device: Device) -> Result<Self, MetalStreamError> {
        Ok(Self {
            shader_cache: Arc::new(ShaderCache::new(device)?),
        })
    }

    pub fn with_shader_cache(shader_cache: Arc<ShaderCache>) -> Self {
        Self { shader_cache }
    }

    /// Returns the picked kernel variant for the given shape (so the
    /// caller knows whether they need to allocate a splitk scratch
    /// `[split_k, M, N]` and emit the downstream sum-reduce).
    pub fn plan(&self, m: u32, n: u32, k: u32, b: u32, group_size: u32) -> QmmTKernel {
        pick_qmm_t_kernel(m, n, k, b, group_size)
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
    ) -> Result<QmmTKernel, MetalStreamError> {
        if bits != 4 {
            return Err(MetalStreamError::ShaderCompilationFailed(format!(
                "affine_qmm_t: only bits=4 is wired in P4, got bits={bits}"
            )));
        }
        if !matches!(group_size, 32 | 64 | 128) {
            return Err(MetalStreamError::ShaderCompilationFailed(format!(
                "affine_qmm_t: only group_size in {{32, 64, 128}} is wired, got {group_size}"
            )));
        }
        if b > 1 {
            return Err(MetalStreamError::ShaderCompilationFailed(format!(
                "affine_qmm_t: batched=1 (B={b}) not yet wired (B=1 in P4)"
            )));
        }
        if k % group_size != 0 {
            return Err(MetalStreamError::ShaderCompilationFailed(format!(
                "affine_qmm_t: K={k} not divisible by group_size={group_size}"
            )));
        }

        let aligned_n = n % 32 == 0;
        let kernel = pick_qmm_t_kernel(m, n, k, b, group_size);
        let kernel_name = qmm_t_kernel_name(kernel, dtype, group_size, bits, aligned_n);
        // K / N / M (and `k_partition_size` for splitk) ride as
        // function constants 0/1/2 (and 3) in `quantized_qmm.metal`
        // — see the `QMM_K` / `QMM_N` / `QMM_M` /
        // `QMM_K_PARTITION_SIZE` declarations at the top of that
        // shader. All declared `constant int` to match the MLX C++
        // source (`int` runtime args at `quantized.h:1719-1721`);
        // Metal type-checks the constant payload byte-for-byte
        // against the slot declaration. The split_k_partition_stride
        // that MLX passes at buffer(9) is computed inline in the
        // splitk kernel as `QMM_M * QMM_N`, so it doesn't need its
        // own constant slot.
        let mut constants = vec![
            ConstantValue::int(0, k as i32),
            ConstantValue::int(1, n as i32),
            ConstantValue::int(2, m as i32),
        ];
        if let QmmTKernel::SplitK {
            k_partition_size, ..
        } = kernel
        {
            constants.push(ConstantValue::int(3, k_partition_size as i32));
        }
        let pipeline = self
            .shader_cache
            .get_pipeline_specialized(&kernel_name, &constants)?;
        encoder.setComputePipelineState(&pipeline);

        // Buffer bindings 0..4 — packed weight, scales, biases, x, y.
        // Same layout as MLX `affine_qmm_t` kernel signature
        // (`quantized.h:1716-1721`).
        // SAFETY: all buffer pointers are valid `Retained` objects;
        // `setBuffer:offset:atIndex:` only borrows them through the call.
        unsafe {
            encoder.setBuffer_offset_atIndex(Some(packed_w), 0, 0);
            encoder.setBuffer_offset_atIndex(Some(scales), 0, 1);
            encoder.setBuffer_offset_atIndex(Some(biases), 0, 2);
            encoder.setBuffer_offset_atIndex(Some(x), 0, 3);
            encoder.setBuffer_offset_atIndex(Some(y), 0, 4);
        }

        let (tg, tpg) = qmm_t_dispatch_shape(kernel, m, n, b);
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
        Ok(kernel)
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
    fn qmm_t_split_k_matches_mlx_heuristic() {
        // Llama-1B prefill q_proj: M=64, N=2048, K=2048, gs=64, B=1.
        // n_tiles = 64, m_tiles = 2 → current_tgs = 128 → target_split_k =
        // 512/128 = 4. K/group_size = 32 → cap 32. K % (4*64) = 0 → 4 stands.
        assert_eq!(pick_qmm_t_split_k(64, 2048, 2048, 64), 4);

        // Llama-1B prefill o_proj: same shape, B=1.
        // Same as above.
        assert_eq!(pick_qmm_t_split_k(64, 2048, 2048, 64), 4);

        // Long-prompt prefill: M=512, N=2048, K=2048, gs=64.
        // n_tiles = 64, m_tiles = 16 → current_tgs = 1024 → target_split_k
        // = 512/1024 = 0, clamped to 1.
        assert_eq!(pick_qmm_t_split_k(512, 2048, 2048, 64), 1);

        // Decode-shape qmm border: M=18, N=8192, K=2048, gs=64.
        // n_tiles = 256, m_tiles = 1 → tgs = 256 → 512/256 = 2. cap = 32.
        // K % (2*64) = 0 → 2 stands.
        assert_eq!(pick_qmm_t_split_k(18, 8192, 2048, 64), 2);
    }

    #[test]
    fn qmm_t_kernel_pick_routes_splitk_only_for_b_eq_1() {
        // B == 1, decent splitk → SplitK
        let k = pick_qmm_t_kernel(64, 2048, 2048, 1, 64);
        assert!(matches!(k, QmmTKernel::SplitK { split_k: 4, .. }));

        // B > 1 → always Standard
        let k = pick_qmm_t_kernel(64, 2048, 2048, 2, 64);
        assert_eq!(k, QmmTKernel::Standard);

        // B == 1, splitk collapses to 1 → Standard
        let k = pick_qmm_t_kernel(512, 2048, 2048, 1, 64);
        assert_eq!(k, QmmTKernel::Standard);
    }

    #[test]
    fn qmm_t_kernel_name_matches_metallib_symbols() {
        assert_eq!(
            qmm_t_kernel_name(QmmTKernel::Standard, DequantDtype::Bf16, 64, 4, true),
            "affine_qmm_t_bf16_gs_64_b_4_alN_true_batch_0"
        );
        assert_eq!(
            qmm_t_kernel_name(QmmTKernel::Standard, DequantDtype::F16, 32, 4, false),
            "affine_qmm_t_f16_gs_32_b_4_alN_false_batch_0"
        );
        assert_eq!(
            qmm_t_kernel_name(
                QmmTKernel::SplitK {
                    split_k: 4,
                    k_partition_size: 512
                },
                DequantDtype::Bf16,
                128,
                4,
                true
            ),
            "affine_qmm_t_splitk_bf16_gs_128_b_4_alN_true"
        );
    }

    #[test]
    fn qmm_t_dispatch_shape_matches_mlx_grid_dims() {
        // Standard: grid (ceil(N/32), ceil(M/32), B); group (32, 2, 2).
        let ((tx, ty, tz), (gx, gy, gz)) =
            qmm_t_dispatch_shape(QmmTKernel::Standard, 64, 2048, 1);
        assert_eq!((tx, ty, tz), (64, 2, 1));
        assert_eq!((gx, gy, gz), (32, 2, 2));

        // SplitK: grid (n_tiles, m_tiles, split_k); same group.
        let ((tx, ty, tz), (gx, gy, gz)) = qmm_t_dispatch_shape(
            QmmTKernel::SplitK {
                split_k: 4,
                k_partition_size: 512,
            },
            64,
            2048,
            1,
        );
        assert_eq!((tx, ty, tz), (64, 2, 4));
        assert_eq!((gx, gy, gz), (32, 2, 2));
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
