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

/// Activation / output dtype the affine quant kernels read and write —
/// the kernel template parameter `T_act` per `INT4_PARITY_PROBES.md` §7.
/// Picks between the `affine_*_f16_s_*_*` and `affine_*_bf16_s_*_*`
/// symbol families. Historical name retained: this was `DequantDtype`
/// pre-P10b, when the `<T>` template covered both activation and scale.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum DequantDtype {
    F16,
    Bf16,
}

impl DequantDtype {
    pub fn symbol_infix(self) -> &'static str {
        match self {
            Self::F16 => "f16",
            Self::Bf16 => "bf16",
        }
    }

    pub fn elem_size(self) -> usize {
        2
    }
}

/// Storage dtype the kernel reads `scales` / `biases` device pointers
/// as — the kernel template parameter `T_scale` per
/// `INT4_PARITY_PROBES.md` §7 `Decision: in-register cast`. Llama-3.x
/// mlx-community 4bit ships F16 scales; Qwen3-MoE (and other
/// `torch_dtype: bfloat16` exports) ships BF16 scales. MLX templates
/// both natively (`mlx/.../quantized.h:INSTANTIATE_QUANTIZED_FUNCTIONS`
/// instantiates `T_scale ∈ {half, bfloat16_t}`); ferrite-metal mirrors
/// that surface.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum ScaleDtype {
    F16,
    Bf16,
}

impl ScaleDtype {
    pub fn symbol_infix(self) -> &'static str {
        match self {
            Self::F16 => "f16",
            Self::Bf16 => "bf16",
        }
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
        scale_dtype: ScaleDtype,
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
        if !out_n_elements.is_multiple_of(packs_per_int) {
            return Err(MetalStreamError::ShaderCompilationFailed(format!(
                "affine_dequantize: out_n_elements={out_n_elements} not divisible by \
                 packs_per_int={packs_per_int} (bits={bits})"
            )));
        }
        let nthreads = out_n_elements / packs_per_int;

        let kernel_name = format!(
            "affine_dequantize_{}_s_{}_gs_{}_b_{}",
            dtype.symbol_infix(),
            scale_dtype.symbol_infix(),
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
// MLX-affine int4 quantized embedding dispatcher — port of
// `nn.QuantizedEmbedding.__call__` (`python/mlx/nn/layers/quantized.py:144`).
// MLX dispatches as 3 gathers + 1 dequantize; ferrite-metal fuses them
// into one kernel (`affine_embed_<dtype>_gs_<gs>_b_4` in
// `shaders/quantized_dequantize.metal`) because the gather-row
// indirection per output is constant — not a cross-op fusion.
// ─────────────────────────────────────────────────────────────────

/// MLX-affine int4 quantized embedding lookup. Holds a `ShaderCache`
/// that lazily builds one pipeline per `(dtype, group_size, hidden_size)`
/// instantiation (`hidden_size` rides as a function constant).
pub struct MetalAffineEmbed {
    shader_cache: Arc<ShaderCache>,
}

impl MetalAffineEmbed {
    pub fn new(device: Device) -> Result<Self, MetalStreamError> {
        Ok(Self {
            shader_cache: Arc::new(ShaderCache::new(device)?),
        })
    }

    pub fn with_shader_cache(shader_cache: Arc<ShaderCache>) -> Self {
        Self { shader_cache }
    }

    /// Dispatch `affine_embed_<dtype>_gs_<gs>_b_<bits>` against an open
    /// encoder. Gathers + dequants `num_tokens` rows from a quantized
    /// embedding table in one pass.
    ///
    /// - `packed_weight`: `[vocab_size, hidden_size / 8]` U32, treated
    ///   as `[vocab_size * hidden_size / 2]` bytes by the kernel
    ///   (`buffer(0)`).
    /// - `scales`, `biases`: `[vocab_size * hidden_size / group_size]`
    ///   half-precision per-group affine parameters
    ///   (`buffer(1)` / `buffer(2)`).
    /// - `indices`: `[num_tokens]` U32 token IDs (`buffer(3)`).
    /// - `output`: `[num_tokens, hidden_size]` half-precision result
    ///   (`buffer(4)`).
    /// - `hidden_size`: ridden as `[[function_constant(0)]]` so the
    ///   pipeline bakes in the bucket's `Q_SIZE`.
    /// - `group_size`: must be one of {32, 64, 128}.
    /// - `bits`: must equal 4.
    #[allow(clippy::too_many_arguments)]
    pub fn execute(
        &self,
        packed_weight: &Buffer,
        scales: &Buffer,
        biases: &Buffer,
        indices: &Buffer,
        output: &Buffer,
        num_tokens: u32,
        hidden_size: u32,
        group_size: u32,
        bits: u32,
        dtype: DequantDtype,
        scale_dtype: ScaleDtype,
        encoder: &ComputeCommandEncoderRef,
    ) -> Result<(), MetalStreamError> {
        if bits != 4 {
            return Err(MetalStreamError::ShaderCompilationFailed(format!(
                "affine_embed: only bits=4 is wired, got bits={bits}"
            )));
        }
        if !matches!(group_size, 32 | 64 | 128) {
            return Err(MetalStreamError::ShaderCompilationFailed(format!(
                "affine_embed: only group_size in {{32, 64, 128}} is wired, got {group_size}"
            )));
        }
        // packs_per_int = 8 / bits = 2 for bits=4. Each thread emits 2
        // output elements (one packed byte). hidden_size must be a
        // multiple of pack_factor; mlx-community 4bit ships hidden_size
        // divisible by 8 (= 32-bit pack width), so /2 is always clean.
        let packs_per_int: u32 = 2;
        if !hidden_size.is_multiple_of(packs_per_int) {
            return Err(MetalStreamError::ShaderCompilationFailed(format!(
                "affine_embed: hidden_size={hidden_size} not divisible by \
                 pack_factor={packs_per_int} (bits={bits})"
            )));
        }

        let kernel_name = format!(
            "affine_embed_{}_s_{}_gs_{}_b_{}",
            dtype.symbol_infix(),
            scale_dtype.symbol_infix(),
            group_size,
            bits
        );
        // `hidden_size` rides as function_constant(0) — see
        // `AFFINE_EMBED_HIDDEN_SIZE` in `quantized_dequantize.metal`.
        let constants = [ConstantValue::uint(0, hidden_size)];
        let pipeline = self
            .shader_cache
            .get_pipeline_specialized(&kernel_name, &constants)?;
        encoder.setComputePipelineState(&pipeline);

        // SAFETY: all buffer pointers are valid `Retained` objects;
        // `setBuffer:offset:atIndex:` only borrows them through the call.
        unsafe {
            encoder.setBuffer_offset_atIndex(Some(packed_weight), 0, 0);
            encoder.setBuffer_offset_atIndex(Some(scales), 0, 1);
            encoder.setBuffer_offset_atIndex(Some(biases), 0, 2);
            encoder.setBuffer_offset_atIndex(Some(indices), 0, 3);
            encoder.setBuffer_offset_atIndex(Some(output), 0, 4);
        }

        // 2D dispatch: x = byte-offset within a token row, y = token
        // index. Each thread emits 2 output elements. The kernel itself
        // bounds-checks `index.x * 2 >= hidden_size` to catch the
        // partial trailing threadgroup when hidden_size/2 exceeds the
        // pipeline's max threadgroup width (Llama-3B: hidden=3072 → K/2
        // =1536, max tpg width 1024 forces 2 threadgroups along x).
        let bytes_per_row = hidden_size / packs_per_int;
        let max_tpg = pipeline
            .maxTotalThreadsPerThreadgroup()
            .min(u32::MAX as usize) as u32;
        let tpg_x = bytes_per_row.min(max_tpg).max(1);
        let groups_x = bytes_per_row.div_ceil(tpg_x);
        let threads_per_threadgroup = MTLSize {
            width: tpg_x as usize,
            height: 1,
            depth: 1,
        };
        let threadgroups = MTLSize {
            width: groups_x as usize,
            height: num_tokens as usize,
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
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
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
    } else if n.is_multiple_of(8) && k.is_multiple_of(512) {
        QmvKernel::Fast
    } else {
        QmvKernel::Generic
    }
}

/// Every `QmvKernel` variant that is *valid* for shape `(n, k, bits)`
/// — i.e. produces correct output. The cost sweep benches each one
/// and the solver picks the min-cost variant from the resulting CSV,
/// replacing [`pick_qmv_kernel`]'s heuristic with empirical data.
///
/// `Generic` is always valid (it has the bounds-checked tail).
/// `Fast` requires `N % 8 == 0 && K % 512 == 0`.
/// `Quad` requires `K ∈ {64, 128}` with pow2 bits.
pub fn valid_qmv_kernels(n: u32, k: u32, bits: u32) -> Vec<QmvKernel> {
    let mut out = Vec::with_capacity(3);
    let pow2_bits = bits != 0 && (bits & (bits - 1)) == 0;
    if (k == 64 || k == 128) && pow2_bits {
        out.push(QmvKernel::Quad { d: k });
    }
    if n.is_multiple_of(8) && k.is_multiple_of(512) {
        out.push(QmvKernel::Fast);
    }
    out.push(QmvKernel::Generic);
    out
}

/// CSV row name for a picked `QmvKernel`, matching what the
/// `ferrite-metal-cost-sweep::affine_qmv_sweep` binary emits.
pub fn qmv_csv_kernel_name(kernel: QmvKernel, dtype: DequantDtype, gs: u32) -> String {
    let dt = dtype.symbol_infix();
    match kernel {
        QmvKernel::Quad { d } => format!("affine_qmv_quad_{dt}_gs{gs}_d{d}"),
        QmvKernel::Fast => format!("affine_qmv_fast_{dt}_gs{gs}"),
        QmvKernel::Generic => format!("affine_qmv_{dt}_gs{gs}"),
    }
}

/// Cost-driven `QmvKernel` selection. Walks [`valid_qmv_kernels`],
/// looks each up via `cost_lookup`, and returns the variant with
/// minimum `cost_us`. When the lookup has no row for any valid
/// variant (uncalibrated chip / sweep gap), falls back to the
/// [`pick_qmv_kernel`] heuristic.
///
/// This is the **solver-side** picker. Both the codegen-time cost
/// estimator (`MetalAffineQmmImpl::cost_us`) and the runtime
/// lowering pass call it so the cost decision and the dispatch
/// decision can't disagree.
///
/// `cost_lookup` is a closure rather than a concrete profile type
/// so the macro's wrapped `TargetProfile` and the runtime's
/// `MetalTargetProfile` can both feed in without this crate having
/// to know about the macro side.
pub fn pick_qmv_kernel_by_cost(
    cost_lookup: impl Fn(&str, u32, u32, u32) -> Option<f64>,
    n: u32,
    k: u32,
    bits: u32,
    group_size: u32,
    dtype: DequantDtype,
) -> QmvKernel {
    let mut best: Option<(QmvKernel, f64)> = None;
    for kernel in valid_qmv_kernels(n, k, bits) {
        let name = qmv_csv_kernel_name(kernel, dtype, group_size);
        if let Some(cost) = cost_lookup(&name, 1, n, k) {
            match best {
                None => best = Some((kernel, cost)),
                Some((_, c)) if cost < c => best = Some((kernel, cost)),
                _ => {}
            }
        }
    }
    best.map(|(k_pick, _)| k_pick)
        .unwrap_or_else(|| pick_qmv_kernel(n, k, bits))
}

/// Threadgroup grid + threads-per-group for a picked qmv variant.
///
/// `qmv_quad`: `bn = quads_per_simd * results_per_quadgroup = 8 * 8 = 64`
/// (`quantized.cpp:193-198`); group is one simdgroup
/// (`(simdgroup_size=32, 1, 1)`).
///
/// `qmv` / `qmv_fast`: `bn = 8`, `bk = 32`, group `(bk=32, 2, 1)` —
/// 2 simdgroups (`quantized.cpp:251-254`).
pub fn qmv_dispatch_shape(
    kernel: QmvKernel,
    m: u32,
    n: u32,
    b: u32,
) -> ((u32, u32, u32), (u32, u32, u32)) {
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
    scale_dtype: ScaleDtype,
    group_size: u32,
    bits: u32,
    batched: bool,
) -> String {
    let dtype = dtype.symbol_infix();
    let sdt = scale_dtype.symbol_infix();
    let batch = if batched { 1 } else { 0 };
    match kernel {
        QmvKernel::Quad { d } => {
            format!("affine_qmv_quad_{dtype}_s_{sdt}_gs_{group_size}_b_{bits}_d_{d}_batch_{batch}",)
        }
        QmvKernel::Fast => {
            format!("affine_qmv_fast_{dtype}_s_{sdt}_gs_{group_size}_b_{bits}_batch_{batch}",)
        }
        QmvKernel::Generic => {
            format!("affine_qmv_{dtype}_s_{sdt}_gs_{group_size}_b_{bits}_batch_{batch}",)
        }
    }
}

/// `&'static str` view of [`qmv_kernel_name`] for the lowering pass.
/// `LoweredCommand::function` is `&'static str`, so the macro-emit
/// path can't allocate a `String` here. Covers the non-batched (B=1)
/// instantiations only (`batch_0`); MoE batched=1 lands with P13.
///
/// Composes the `affine_qmv_{quad,fast,generic}_<dtype>_s_<scale>_gs_<gs>_b_4[_d_<D>]_batch_0`
/// symbol from the kernel-variant axes. Returns `&'static str` via a
/// process-lifetime `LazyLock` cache so the same `(kernel, dtype,
/// scale, gs)` key always returns the same pointer (the worker's
/// `SpecializedPipelineCache` uses it as a HashMap key). The cache is
/// the seam where `(F16 × BF16 scale-dtype × {32,64,128} group-size ×
/// 3-kernel-variant)` instantiations get materialized as leaked
/// `&'static str` — no per-call format! and no 144-arm match table.
pub fn qmv_kernel_static_name(
    kernel: QmvKernel,
    dtype: DequantDtype,
    scale_dtype: ScaleDtype,
    bits: u32,
    group_size: u32,
) -> &'static str {
    debug_assert!(
        matches!(bits, 4 | 8),
        "qmv_kernel_static_name: only bits 4 and 8 are wired (got {bits})"
    );
    let key = (kernel, dtype, scale_dtype, group_size, bits);
    use std::collections::HashMap;
    use std::sync::OnceLock;
    static CACHE: OnceLock<
        std::sync::Mutex<HashMap<(QmvKernel, DequantDtype, ScaleDtype, u32, u32), &'static str>>,
    > = OnceLock::new();
    let cache = CACHE.get_or_init(|| std::sync::Mutex::new(HashMap::new()));
    let mut guard = cache.lock().expect("qmv_kernel_static_name cache poisoned");
    if let Some(&v) = guard.get(&key) {
        return v;
    }
    let supported_gs = matches!(group_size, 32 | 64 | 128);
    if !supported_gs {
        panic!(
            "qmv_kernel_static_name: unsupported group_size={group_size} \
             — only 32, 64, 128 instantiated"
        );
    }
    let dtype_s = dtype.symbol_infix();
    let scale_s = scale_dtype.symbol_infix();
    let owned = match kernel {
        QmvKernel::Quad { d } => {
            if !matches!(d, 64 | 128) {
                panic!(
                    "qmv_kernel_static_name: QmvKernel::Quad with unsupported D={d} \
                     — only 64 and 128 instantiated"
                );
            }
            format!("affine_qmv_quad_{dtype_s}_s_{scale_s}_gs_{group_size}_b_{bits}_d_{d}_batch_0")
        }
        QmvKernel::Fast => {
            format!("affine_qmv_fast_{dtype_s}_s_{scale_s}_gs_{group_size}_b_{bits}_batch_0")
        }
        QmvKernel::Generic => {
            format!("affine_qmv_{dtype_s}_s_{scale_s}_gs_{group_size}_b_{bits}_batch_0")
        }
    };
    let leaked: &'static str = Box::leak(owned.into_boxed_str());
    guard.insert(key, leaked);
    leaked
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
        scale_dtype: ScaleDtype,
        encoder: &ComputeCommandEncoderRef,
    ) -> Result<(), MetalStreamError> {
        let kernel = pick_qmv_kernel(n, k, bits);
        self.execute_with_kernel(
            kernel,
            x,
            packed_w,
            scales,
            biases,
            y,
            m,
            n,
            k,
            b,
            group_size,
            bits,
            dtype,
            scale_dtype,
            encoder,
        )
    }

    /// Variant-explicit form of [`execute`]: caller picks the
    /// `QmvKernel` rather than going through [`pick_qmv_kernel`].
    /// Used by the cost sweep to bench each valid variant per shape
    /// (the heuristic picker emits only one row per shape, which
    /// prevents the solver from being cost-driven). Production code
    /// should call [`execute`] unless it's doing its own cost
    /// lookup.
    #[allow(clippy::too_many_arguments)]
    pub fn execute_with_kernel(
        &self,
        kernel: QmvKernel,
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
        scale_dtype: ScaleDtype,
        encoder: &ComputeCommandEncoderRef,
    ) -> Result<(), MetalStreamError> {
        if !matches!(bits, 4 | 8) {
            return Err(MetalStreamError::ShaderCompilationFailed(format!(
                "affine_qmv: only bits 4 and 8 are wired, got bits={bits}"
            )));
        }
        if !matches!(group_size, 32 | 64 | 128) {
            return Err(MetalStreamError::ShaderCompilationFailed(format!(
                "affine_qmv: only group_size in {{32, 64, 128}} is wired, got {group_size}"
            )));
        }
        if b > 1 {
            return Err(MetalStreamError::ShaderCompilationFailed(format!(
                "affine_qmv: batched=1 (B={b}) not yet wired (decode-only B=1 in P3)"
            )));
        }
        if let QmvKernel::Quad { d } = kernel {
            debug_assert!(d == 64 || d == 128);
        }
        let kernel_name = qmv_kernel_name(kernel, dtype, scale_dtype, group_size, bits, false);
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
// MoE per-expert gather-matvec: faithful port of MLX
// `affine_gather_qmv_fast` / `affine_gather_qmv` from
// `mlx/backend/metal/kernels/quantized.h:1899-2021`. Reuses the
// existing `qmv_*_impl` compute kernels under per-expert weight
// slab offsets driven by an indices buffer.
//
// Used by Mixtral / Qwen2-MoE / Qwen3-MoE SwitchGLU at decode
// (M=1). Prefill (M>1) uses `affine_gather_qmm_rhs_nt` (not yet
// ported — Phase C2 of [[project-metal-moe-switchglu]]); the
// gather_qmv kernel functions for prefill too (just slower at
// the high-tokens × top_k count) by treating each (token, slot)
// as its own matvec.
// ─────────────────────────────────────────────────────────────────

/// `MetalAffineGatherQmv` — chooses between the `fast` and `generic`
/// gather kernels per the same `pick_qmv_kernel` heuristic the
/// non-gather qmv uses. Quad variant is skipped: gather kernels
/// never see `K ∈ {64, 128}` in practice (router projects from
/// `hidden_size = 2048+`).
pub struct MetalAffineGatherQmv {
    shader_cache: Arc<ShaderCache>,
}

impl MetalAffineGatherQmv {
    pub fn new(device: Device) -> Result<Self, MetalStreamError> {
        Ok(Self {
            shader_cache: Arc::new(ShaderCache::new(device)?),
        })
    }

    pub fn with_shader_cache(shader_cache: Arc<ShaderCache>) -> Self {
        Self { shader_cache }
    }

    /// `x`: `[N, K]` activations row-contiguous.
    /// `w`: `[num_experts, N_out, K / 8]` packed int4.
    /// `scales`, `biases`: `[num_experts, N_out, K / group_size]`.
    /// `rhs_indices`: `[num_tokens, top_k]` u32 — flattens into
    /// `tid.z` indexing in the kernel.
    /// `y`: `[num_tokens, top_k, N_out]`.
    /// `num_tokens`: outer activation rows (matches `N` above).
    /// `top_k`: number of experts per token (Mixtral 2, Qwen3 8…).
    /// `n_out`, `k`: per-expert weight slab dims.
    /// `group_size`, `bits`: quant params (4 / {32,64,128} only).
    #[allow(clippy::too_many_arguments)]
    pub fn execute(
        &self,
        x: &Buffer,
        packed_w: &Buffer,
        scales: &Buffer,
        biases: &Buffer,
        rhs_indices: &Buffer,
        y: &Buffer,
        num_tokens: u32,
        top_k: u32,
        n_out: u32,
        k: u32,
        group_size: u32,
        bits: u32,
        dtype: DequantDtype,
        scale_dtype: ScaleDtype,
        encoder: &ComputeCommandEncoderRef,
    ) -> Result<(), MetalStreamError> {
        if bits != 4 {
            return Err(MetalStreamError::ShaderCompilationFailed(format!(
                "affine_gather_qmv: only bits=4 wired, got bits={bits}"
            )));
        }
        if !matches!(group_size, 32 | 64 | 128) {
            return Err(MetalStreamError::ShaderCompilationFailed(format!(
                "affine_gather_qmv: only group_size in {{32,64,128}} wired, got {group_size}"
            )));
        }
        if num_tokens == 0 || top_k == 0 {
            return Err(MetalStreamError::ShaderCompilationFailed(format!(
                "affine_gather_qmv: num_tokens={num_tokens} top_k={top_k}; both > 0"
            )));
        }

        // Same Fast-vs-Generic heuristic as non-gather qmv: Fast
        // requires N%8==0 && K%512==0.
        let kernel = if n_out.is_multiple_of(8) && k.is_multiple_of(512) {
            "affine_gather_qmv_fast"
        } else {
            "affine_gather_qmv"
        };
        let dt = dtype.symbol_infix();
        let sdt = scale_dtype.symbol_infix();
        let kernel_name = format!("{kernel}_{dt}_s_{sdt}_gs_{group_size}_b_{bits}");

        let constants = [
            ConstantValue::int(0, k as i32),
            ConstantValue::int(1, n_out as i32),
        ];
        let pipeline = self
            .shader_cache
            .get_pipeline_specialized(&kernel_name, &constants)?;
        encoder.setComputePipelineState(&pipeline);

        let top_k_i = top_k as i32;
        unsafe {
            encoder.setBuffer_offset_atIndex(Some(packed_w), 0, 0);
            encoder.setBuffer_offset_atIndex(Some(scales), 0, 1);
            encoder.setBuffer_offset_atIndex(Some(biases), 0, 2);
            encoder.setBuffer_offset_atIndex(Some(x), 0, 3);
            encoder.setBuffer_offset_atIndex(Some(rhs_indices), 0, 4);
            encoder.setBuffer_offset_atIndex(Some(y), 0, 5);
            encoder.setBytes_length_atIndex(
                std::ptr::NonNull::new(&top_k_i as *const i32 as *mut std::ffi::c_void).unwrap(),
                std::mem::size_of::<i32>(),
                6,
            );
        }

        // grid = (1, n_out/8, num_tokens*top_k); threads_per_tg =
        // (32, 2, 1) for the qmv_fast/generic family.
        let bn: u32 = 8;
        let threadgroups = MTLSize {
            width: 1,
            height: n_out.div_ceil(bn) as usize,
            depth: (num_tokens as usize) * (top_k as usize),
        };
        let threads_per_threadgroup = MTLSize {
            width: 32,
            height: 2,
            depth: 1,
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
        G::M3 | G::M4 | G::M5 => {
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
    SplitK { split_k: u32, k_partition_size: u32 },
    /// `affine_qmm_t_nax_*_alN_<bool>_batch_0` — NAX (Apple9 / M4+)
    /// MMA path using `MetalPerformancePrimitives matmul2d`. Only
    /// selected when `is_nax == true` AND `K % 64 == 0`.
    /// 64×64×64 tile, no split-K (MLX NAX path at `quantized.cpp:695`).
    Nax,
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
    while split_k > 1 && !k.is_multiple_of(split_k * group_size) {
        split_k -= 1;
    }
    split_k
}

/// Pick the right qmm_t variant per MLX's matmul-branch routing.
/// Mirrors `quantized.cpp:1411-1424` plus the NAX gate at `:695`:
///
/// ```text
/// if is_nax && M >= 32 && K % 64 == 0 && group_size != 32: qmm_t_nax (M5+)
/// else if transpose && B == 1:                      qmm_splitk
/// else if transpose:                                qmm (transpose=true)
/// ```
///
/// `is_nax` should be `ferrite_metal_targets::is_nax_capable(profile.generation)`.
///
/// gs=32 is excluded from NAX dispatch because the BK=64 NAX shader
/// violates `BCOLS <= group_size` for gs=32; MLX handles that with a
/// specialized QuantizedBlockLoader path (different scale-indexing
/// semantics) which we haven't ported. gs=32 quants fall through to
/// the Standard qmm_t kernel instead.
pub fn pick_qmm_t_kernel(
    m: u32,
    n: u32,
    k: u32,
    b: u32,
    group_size: u32,
    is_nax: bool,
) -> QmmTKernel {
    // NAX's 64×64 tiling wastes work and yields too few threadgroups at
    // tiny M: it regresses to ~0.8× vs SplitK at M=16 but wins (≥1.7×)
    // from M=32 up (see `nax_vs_standard_qmm_t_bench`). Gate it to
    // M ≥ 32 so very short prefills keep the SplitK/Standard path.
    // (Belt-and-suspenders: the production qmm_t buckets start at 64,
    // so this threshold isn't normally reached.)
    const NAX_MIN_M: u32 = 32;
    if is_nax && m >= NAX_MIN_M && k.is_multiple_of(64) && group_size != 32 {
        return QmmTKernel::Nax;
    }
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
    match kernel {
        QmmTKernel::Nax => {
            // BM=BN=64 tile, TGP=128 = 4 simdgroups × 32 threads.
            let n_tiles = n.div_ceil(64);
            let m_tiles = m.div_ceil(64);
            ((n_tiles, m_tiles, b), (128, 1, 1))
        }
        _ => {
            let n_tiles = n.div_ceil(32);
            let m_tiles = m.div_ceil(32);
            match kernel {
                QmmTKernel::Standard => ((n_tiles, m_tiles, b), (32, 2, 2)),
                QmmTKernel::SplitK { split_k, .. } => ((n_tiles, m_tiles, split_k), (32, 2, 2)),
                QmmTKernel::Nax => unreachable!(),
            }
        }
    }
}

/// Format the kernel symbol name. Matches the `INST_QMM_*` macros in
/// `shaders/quantized_qmm.metal`.
pub fn qmm_t_kernel_name(
    kernel: QmmTKernel,
    dtype: DequantDtype,
    scale_dtype: ScaleDtype,
    group_size: u32,
    bits: u32,
    aligned_n: bool,
) -> String {
    let dtype = dtype.symbol_infix();
    let sdt = scale_dtype.symbol_infix();
    let aln = if aligned_n { "true" } else { "false" };
    match kernel {
        QmmTKernel::Standard => {
            format!("affine_qmm_t_{dtype}_s_{sdt}_gs_{group_size}_b_{bits}_alN_{aln}_batch_0",)
        }
        QmmTKernel::SplitK { .. } => {
            format!("affine_qmm_t_splitk_{dtype}_s_{sdt}_gs_{group_size}_b_{bits}_alN_{aln}",)
        }
        QmmTKernel::Nax => {
            format!("affine_qmm_t_nax_{dtype}_s_{sdt}_gs_{group_size}_b_{bits}_alN_{aln}_batch_0",)
        }
    }
}

/// `&'static str` view of [`qmm_t_kernel_name`] for the lowering pass
/// — see [`qmv_kernel_static_name`] for the analogous discussion.
/// `LoweredCommand::function` is `&'static str`; the table below
/// enumerates every entry the `INST_QMM_ALL` macro produces in
/// `shaders/quantized_qmm.metal`. `scale_dtype` is plumbed for
/// forward-compat with P11 — only `ScaleDtype::F16` is instantiated.
pub fn qmm_t_kernel_static_name(
    kernel: QmmTKernel,
    dtype: DequantDtype,
    scale_dtype: ScaleDtype,
    bits: u32,
    group_size: u32,
    aligned_n: bool,
) -> &'static str {
    debug_assert!(
        matches!(bits, 4 | 8),
        "qmm_t_kernel_static_name: only bits 4 and 8 are wired (got {bits})"
    );
    debug_assert!(
        !(bits == 8 && matches!(kernel, QmmTKernel::SplitK { .. })),
        "qmm_t_kernel_static_name: SplitK only instantiates bits=4 — \
         the lowering arm must route 8-bit weights to Standard or Nax"
    );
    let key = (
        std::mem::discriminant(&kernel),
        dtype,
        scale_dtype,
        group_size,
        aligned_n,
        bits,
    );
    use std::collections::HashMap;
    use std::sync::OnceLock;
    type Key = (
        std::mem::Discriminant<QmmTKernel>,
        DequantDtype,
        ScaleDtype,
        u32,
        bool,
        u32,
    );
    static CACHE: OnceLock<std::sync::Mutex<HashMap<Key, &'static str>>> = OnceLock::new();
    let cache = CACHE.get_or_init(|| std::sync::Mutex::new(HashMap::new()));
    let mut guard = cache
        .lock()
        .expect("qmm_t_kernel_static_name cache poisoned");
    if let Some(&v) = guard.get(&key) {
        return v;
    }
    if !matches!(group_size, 32 | 64 | 128) {
        panic!(
            "qmm_t_kernel_static_name: unsupported group_size={group_size} \
             — only 32, 64, 128 instantiated"
        );
    }
    if matches!(kernel, QmmTKernel::Nax) && group_size == 32 {
        panic!(
            "qmm_t_kernel_static_name: NAX dispatched with gs=32 — \
             `pick_qmm_t_kernel` should have routed to Standard"
        );
    }
    let owned = qmm_t_kernel_name(kernel, dtype, scale_dtype, group_size, bits, aligned_n);
    // SplitK kernel names lack the trailing `_batch_0` suffix —
    // `qmm_t_kernel_name` already handles that distinction.
    let leaked: &'static str = Box::leak(owned.into_boxed_str());
    guard.insert(key, leaked);
    leaked
}

/// Compute-aware variant. When `compute_dtype != dtype`, picks the
/// extended `affine_qmm_t_<act>_c_<compute>_s_<scale>_*` symbol. Only
/// the (Bf16-act, F16-compute) combo is currently instantiated — used
/// on Apple7 (M1) where bf16 simdgroup MMAs are slow-path emulation.
/// Falls back to [`qmm_t_kernel_static_name`] when compute == act.
pub fn qmm_t_kernel_static_name_with_compute(
    kernel: QmmTKernel,
    act_dtype: DequantDtype,
    compute_dtype: DequantDtype,
    scale_dtype: ScaleDtype,
    bits: u32,
    group_size: u32,
    aligned_n: bool,
) -> &'static str {
    if compute_dtype == act_dtype {
        return qmm_t_kernel_static_name(
            kernel,
            act_dtype,
            scale_dtype,
            bits,
            group_size,
            aligned_n,
        );
    }
    debug_assert_eq!(
        bits, 4,
        "qmm_t_kernel_static_name_with_compute: only bits=4"
    );
    use DequantDtype::*;
    use ScaleDtype as S;
    match (
        kernel,
        act_dtype,
        compute_dtype,
        scale_dtype,
        group_size,
        aligned_n,
    ) {
        // ── qmm_t Standard, bf16-act + f16-compute ───────────────
        (QmmTKernel::Standard, Bf16, F16, S::F16, 32, true) => {
            "affine_qmm_t_bf16_c_f16_s_f16_gs_32_b_4_alN_true_batch_0"
        }
        (QmmTKernel::Standard, Bf16, F16, S::F16, 32, false) => {
            "affine_qmm_t_bf16_c_f16_s_f16_gs_32_b_4_alN_false_batch_0"
        }
        (QmmTKernel::Standard, Bf16, F16, S::F16, 64, true) => {
            "affine_qmm_t_bf16_c_f16_s_f16_gs_64_b_4_alN_true_batch_0"
        }
        (QmmTKernel::Standard, Bf16, F16, S::F16, 64, false) => {
            "affine_qmm_t_bf16_c_f16_s_f16_gs_64_b_4_alN_false_batch_0"
        }
        (QmmTKernel::Standard, Bf16, F16, S::F16, 128, true) => {
            "affine_qmm_t_bf16_c_f16_s_f16_gs_128_b_4_alN_true_batch_0"
        }
        (QmmTKernel::Standard, Bf16, F16, S::F16, 128, false) => {
            "affine_qmm_t_bf16_c_f16_s_f16_gs_128_b_4_alN_false_batch_0"
        }
        // ── qmm_t SplitK, bf16-act + f16-compute ─────────────────
        (QmmTKernel::SplitK { .. }, Bf16, F16, S::F16, 32, true) => {
            "affine_qmm_t_splitk_bf16_c_f16_s_f16_gs_32_b_4_alN_true"
        }
        (QmmTKernel::SplitK { .. }, Bf16, F16, S::F16, 32, false) => {
            "affine_qmm_t_splitk_bf16_c_f16_s_f16_gs_32_b_4_alN_false"
        }
        (QmmTKernel::SplitK { .. }, Bf16, F16, S::F16, 64, true) => {
            "affine_qmm_t_splitk_bf16_c_f16_s_f16_gs_64_b_4_alN_true"
        }
        (QmmTKernel::SplitK { .. }, Bf16, F16, S::F16, 64, false) => {
            "affine_qmm_t_splitk_bf16_c_f16_s_f16_gs_64_b_4_alN_false"
        }
        (QmmTKernel::SplitK { .. }, Bf16, F16, S::F16, 128, true) => {
            "affine_qmm_t_splitk_bf16_c_f16_s_f16_gs_128_b_4_alN_true"
        }
        (QmmTKernel::SplitK { .. }, Bf16, F16, S::F16, 128, false) => {
            "affine_qmm_t_splitk_bf16_c_f16_s_f16_gs_128_b_4_alN_false"
        }
        // NAX path: don't override compute dtype — Apple9 has hardware
        // bf16 acceleration via the matrix unit, no fast-path needed.
        (QmmTKernel::Nax, _, _, _, _, _) => panic!(
            "qmm_t_kernel_static_name_with_compute: NAX path doesn't need a compute override"
        ),
        _ => panic!(
            "qmm_t_kernel_static_name_with_compute: unsupported (act={act_dtype:?}, \
             compute={compute_dtype:?}, scale={scale_dtype:?}, gs={group_size}) — only \
             (Bf16, F16, F16) is instantiated for the M1 fast-path"
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
        pick_qmm_t_kernel(m, n, k, b, group_size, /*is_nax=*/ false)
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
        scale_dtype: ScaleDtype,
        encoder: &ComputeCommandEncoderRef,
    ) -> Result<QmmTKernel, MetalStreamError> {
        if !matches!(bits, 4 | 8) {
            return Err(MetalStreamError::ShaderCompilationFailed(format!(
                "affine_qmm_t: only bits 4 and 8 are wired, got bits={bits}"
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
        if !k.is_multiple_of(group_size) {
            return Err(MetalStreamError::ShaderCompilationFailed(format!(
                "affine_qmm_t: K={k} not divisible by group_size={group_size}"
            )));
        }

        // 8-bit weights: Standard or Nax (`_b_8_` instantiations exist
        // for both); SplitK is b4-only. This dispatcher passes
        // is_nax=false (production NAX routing lives in the lowering
        // arm, which checks `is_nax_capable` on the live profile), so
        // the b8 guard only needs to block SplitK.
        let kernel = match pick_qmm_t_kernel(m, n, k, b, group_size, /*is_nax=*/ false) {
            QmmTKernel::SplitK { .. } if bits == 8 => QmmTKernel::Standard,
            k => k,
        };
        self.execute_with_kernel(
            x,
            packed_w,
            scales,
            biases,
            y,
            m,
            n,
            k,
            b,
            group_size,
            bits,
            dtype,
            scale_dtype,
            kernel,
            encoder,
        )
    }

    /// Like `execute`, but uses the caller-specified `kernel`
    /// variant instead of consulting `pick_qmm_t_kernel`. Test/debug
    /// path — production dispatch goes through the lowering pass.
    #[allow(clippy::too_many_arguments)]
    pub fn execute_with_kernel(
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
        scale_dtype: ScaleDtype,
        kernel: QmmTKernel,
        encoder: &ComputeCommandEncoderRef,
    ) -> Result<QmmTKernel, MetalStreamError> {
        let aligned_n = match kernel {
            QmmTKernel::Nax => n.is_multiple_of(64),
            _ => n.is_multiple_of(32),
        };
        let kernel_name =
            qmm_t_kernel_name(kernel, dtype, scale_dtype, group_size, bits, aligned_n);
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

// ─────────────────────────────────────────────────────────────────
// SplitK reduce dispatcher — port of MLX's
// `strided_reduce_general_dispatch` invocation at
// `quantized.cpp:861`. Reduces the [split_k, M, N] intermediate
// `affine_qmm_t_splitk` produces down to the final [M, N] output by
// summing along axis 0.
//
// Backed by `splitk_reduce_sum_<dtype>` in
// `shaders/quantized_splitk_reduce.metal`. Function constants 0/1/2
// hold (M, N, split_k) — same `[[function_constant(N)]]` pattern
// used by qmv/qmm_t for ICB readiness.
// ─────────────────────────────────────────────────────────────────

/// Format the splitk-reduce kernel symbol. Matches the
/// `INST_REDUCE` macros in `shaders/quantized_splitk_reduce.metal`.
pub fn splitk_reduce_kernel_static_name(dtype: DequantDtype) -> &'static str {
    match dtype {
        DequantDtype::F16 => "splitk_reduce_sum_f16",
        DequantDtype::Bf16 => "splitk_reduce_sum_bf16",
    }
}

/// Sum-along-axis-0 reduce for the qmm_t_splitk intermediate.
/// Caller supplies a `[split_k, M, N]` input and a `[M, N]` output;
/// the kernel walks one thread per output element and sums the
/// `split_k` partitions in float, matching the float accumulator
/// inside `qmm_t_impl_inline` so the SplitK + reduce composition
/// agrees with the equivalent `qmm_t` Standard run within the
/// summation-order noise floor.
pub struct MetalSplitKReduce {
    shader_cache: Arc<ShaderCache>,
}

impl MetalSplitKReduce {
    pub fn new(device: Device) -> Result<Self, MetalStreamError> {
        Ok(Self {
            shader_cache: Arc::new(ShaderCache::new(device)?),
        })
    }

    pub fn with_shader_cache(shader_cache: Arc<ShaderCache>) -> Self {
        Self { shader_cache }
    }

    /// Dispatch `splitk_reduce_sum_<dtype>` against an open
    /// ComputeCommandEncoder. Caller owns the encoder + commit.
    ///
    /// - `intermediate`: `[split_k * M * N]` flat buffer in `dtype`
    ///   (the qmm_t_splitk output). Layout `[split_k, M, N]`
    ///   row-contiguous.
    /// - `output`: `[M * N]` flat buffer in `dtype`. Caller is
    ///   responsible for sizing both buffers correctly.
    /// - `m`, `n`, `split_k`: must be > 0.
    #[allow(clippy::too_many_arguments)]
    pub fn execute(
        &self,
        intermediate: &Buffer,
        output: &Buffer,
        m: u32,
        n: u32,
        split_k: u32,
        dtype: DequantDtype,
        encoder: &ComputeCommandEncoderRef,
    ) -> Result<(), MetalStreamError> {
        if m == 0 || n == 0 || split_k == 0 {
            return Err(MetalStreamError::ShaderCompilationFailed(format!(
                "splitk_reduce_sum: M, N, split_k all must be > 0 (got M={m}, N={n}, split_k={split_k})"
            )));
        }
        let constants = [
            ConstantValue::uint(0, m),
            ConstantValue::uint(1, n),
            ConstantValue::uint(2, split_k),
        ];
        let kernel_name = splitk_reduce_kernel_static_name(dtype);
        let pipeline = self
            .shader_cache
            .get_pipeline_specialized(kernel_name, &constants)?;
        encoder.setComputePipelineState(&pipeline);

        // SAFETY: buffer pointers are valid `Retained` objects;
        // `setBuffer:offset:atIndex:` only borrows them.
        unsafe {
            encoder.setBuffer_offset_atIndex(Some(output), 0, 0);
            encoder.setBuffer_offset_atIndex(Some(intermediate), 0, 1);
        }

        // 1D dispatch over M*N output elements.
        let nthreads = m as u64 * n as u64;
        if nthreads > u32::MAX as u64 {
            return Err(MetalStreamError::ShaderCompilationFailed(format!(
                "splitk_reduce_sum: nthreads={nthreads} exceeds u32 (M={m}, N={n})"
            )));
        }
        let max_tpg = pipeline.maxTotalThreadsPerThreadgroup() as u64;
        let threads_per_threadgroup_x = nthreads.min(max_tpg);
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
// Prefill-bucket qmm_n dispatcher — port of `quantized.cpp:680
// qmm()` (transpose=false branch). Used when `M >= vector_limit`
// AND transpose=false (which for MLX means `vector_limit = 4`
// flat per `quantized.cpp:1409`; ferrite mirrors that for the
// transpose=false matmul branch). No SplitK variant for qmm_n in
// MLX (only qmm_t has splitk per `quantized.cpp:1413`).
// ─────────────────────────────────────────────────────────────────

/// Format the qmm_n kernel symbol. Matches the `INST_QMM_N` macro in
/// `shaders/quantized_qmm.metal`.
pub fn qmm_n_kernel_name(
    dtype: DequantDtype,
    scale_dtype: ScaleDtype,
    group_size: u32,
    bits: u32,
) -> String {
    let dtype = dtype.symbol_infix();
    let sdt = scale_dtype.symbol_infix();
    format!("affine_qmm_n_{dtype}_s_{sdt}_gs_{group_size}_b_{bits}_batch_0")
}

/// `&'static str` view of [`qmm_n_kernel_name`] for the lowering pass.
pub fn qmm_n_kernel_static_name(
    dtype: DequantDtype,
    scale_dtype: ScaleDtype,
    bits: u32,
    group_size: u32,
) -> &'static str {
    debug_assert_eq!(bits, 4, "qmm_n_kernel_static_name: only bits=4 is wired");
    if !matches!(group_size, 32 | 64 | 128) {
        panic!(
            "qmm_n_kernel_static_name: unsupported group_size={group_size} \
             — only 32, 64, 128 instantiated"
        );
    }
    use std::collections::HashMap;
    use std::sync::OnceLock;
    type Key = (DequantDtype, ScaleDtype, u32);
    static CACHE: OnceLock<std::sync::Mutex<HashMap<Key, &'static str>>> = OnceLock::new();
    let cache = CACHE.get_or_init(|| std::sync::Mutex::new(HashMap::new()));
    let mut guard = cache
        .lock()
        .expect("qmm_n_kernel_static_name cache poisoned");
    let key = (dtype, scale_dtype, group_size);
    if let Some(&v) = guard.get(&key) {
        return v;
    }
    let owned = qmm_n_kernel_name(dtype, scale_dtype, group_size, 4);
    let leaked: &'static str = Box::leak(owned.into_boxed_str());
    guard.insert(key, leaked);
    leaked
}

/// Threadgroup grid + threads-per-group for qmm_n. Matches MLX
/// `quantized.cpp:720-721`: bm=bn=32, wm=wn=2 → group (32, 2, 2);
/// grid `(ceil(N/32), ceil(M/32), B)`.
pub fn qmm_n_dispatch_shape(m: u32, n: u32, b: u32) -> ((u32, u32, u32), (u32, u32, u32)) {
    let n_tiles = n.div_ceil(32);
    let m_tiles = m.div_ceil(32);
    ((n_tiles, m_tiles, b), (32, 2, 2))
}

/// MLX-affine int4 prefill-matmul (transpose=false) dispatcher.
/// Wraps the `quantized_qmm` metallib's `affine_qmm_n` kernel.
///
/// Caller invariants (mirroring `MetalAffineQmmT` for the symmetric
/// transpose=true case):
/// - Activations `x`: `[M, K]` row-contiguous in `dtype`.
/// - Packed weight `w`: `[K, N / pack_factor]` U32 (`pack_factor = 8`
///   for bits=4). Note the layout differs from `qmm_t` (which is
///   `[N, K / pack_factor]`); MLX's quantize-along-last-axis convention
///   keeps the last axis of W as the quantized one — that's K for
///   transpose=true (W shape `[N, K]`) and N for transpose=false
///   (W shape `[K, N]`).
/// - `scales`, `biases`: `[K, N / group_size]` in `dtype`. Same
///   last-axis convention.
/// - Output `y`: `[M, N]` in `dtype`.
/// - `bits` ∈ {4} (P5 mandate).
/// - `group_size` ∈ {32, 64, 128}.
/// - `N % 32 == 0` (MLX `qmm_n` assumes this; the dispatcher in
///   `quantized.cpp` doesn't gate on it but kernel store_result
///   writes unconditional BN×BM tiles for full-M tiles).
pub struct MetalAffineQmmN {
    shader_cache: Arc<ShaderCache>,
}

impl MetalAffineQmmN {
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
        scale_dtype: ScaleDtype,
        encoder: &ComputeCommandEncoderRef,
    ) -> Result<(), MetalStreamError> {
        if bits != 4 {
            return Err(MetalStreamError::ShaderCompilationFailed(format!(
                "affine_qmm_n: only bits=4 is wired in P5, got bits={bits}"
            )));
        }
        if !matches!(group_size, 32 | 64 | 128) {
            return Err(MetalStreamError::ShaderCompilationFailed(format!(
                "affine_qmm_n: only group_size in {{32, 64, 128}} is wired, got {group_size}"
            )));
        }
        if b > 1 {
            return Err(MetalStreamError::ShaderCompilationFailed(format!(
                "affine_qmm_n: batched=1 (B={b}) not yet wired (B=1 in P5)"
            )));
        }
        if !n.is_multiple_of(group_size) {
            return Err(MetalStreamError::ShaderCompilationFailed(format!(
                "affine_qmm_n: N={n} not divisible by group_size={group_size} \
                 (transpose=false stores scales/biases per (K, N/gs))"
            )));
        }

        let kernel_name = qmm_n_kernel_name(dtype, scale_dtype, group_size, bits);
        // K / N / M ride as function constants 0/1/2 (matching the
        // qmm_t / qmm_t_splitk pattern in this same metallib).
        let constants = [
            ConstantValue::int(0, k as i32),
            ConstantValue::int(1, n as i32),
            ConstantValue::int(2, m as i32),
        ];
        let pipeline = self
            .shader_cache
            .get_pipeline_specialized(&kernel_name, &constants)?;
        encoder.setComputePipelineState(&pipeline);

        // SAFETY: all buffer pointers are valid `Retained` objects;
        // `setBuffer:offset:atIndex:` only borrows them through the call.
        unsafe {
            encoder.setBuffer_offset_atIndex(Some(packed_w), 0, 0);
            encoder.setBuffer_offset_atIndex(Some(scales), 0, 1);
            encoder.setBuffer_offset_atIndex(Some(biases), 0, 2);
            encoder.setBuffer_offset_atIndex(Some(x), 0, 3);
            encoder.setBuffer_offset_atIndex(Some(y), 0, 4);
        }

        let (tg, tpg) = qmm_n_dispatch_shape(m, n, b);
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
// Decode-bucket qvm / qvm_split_k dispatcher — port of
// `quantized.cpp:419 qvm()` + `:298 qvm_split_k()`. Fires when
// M < vector_limit AND transpose=false (the matvec-transpose=false
// branch at `:1444-1453`):
//   K <  1024  →  qvm
//   K >= 1024  →  qvm_split_k (split_k = K > 8192 ? 32 : 8)
// ─────────────────────────────────────────────────────────────────

/// Picked qvm variant for a given `(M, N, K)` shape (matvec
/// transpose=false branch). Mirrors `QuantizedMatmul::eval_gpu`
/// at `quantized.cpp:1445-1452`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum QvmKernel {
    /// `affine_qvm_*_batch_0` — K < 1024. MLX `qvm` at
    /// `quantized.cpp:419`.
    Standard,
    /// `affine_qvm_split_k_*` — K >= 1024. MLX `qvm_split_k` at
    /// `quantized.cpp:298`; `split_k = K > 8192 ? 32 : 8` and
    /// `split_D = ceil(K / split_k)`.
    SplitK {
        split_k: u32,
        k_partition_size: u32,
        final_block_size: u32,
    },
}

/// Pick the right qvm variant for `(M, N, K)`. Mirrors MLX's
/// transpose=false matvec routing at `quantized.cpp:1444-1453`:
///
/// ```text
/// if (K < 1024)  qvm(...)
/// else           qvm_split_k(...)
///   split_k = K > 8192 ? 32 : 8
///   split_D = ceil(K / split_k)
///   final_block_size = K - (split_k - 1) * split_D
/// ```
pub fn pick_qvm_kernel(k: u32) -> QvmKernel {
    if k < 1024 {
        QvmKernel::Standard
    } else {
        let split_k: u32 = if k > 8192 { 32 } else { 8 };
        let split_d = k.div_ceil(split_k);
        let final_block_size = k - (split_k - 1) * split_d;
        QvmKernel::SplitK {
            split_k,
            k_partition_size: split_d,
            final_block_size,
        }
    }
}

/// Threadgroup grid + threads-per-group for a picked qvm variant.
/// Matches MLX's dispatch:
///
/// qvm (`quantized.cpp:438-439`):
///   group (bk=32, num_simdgroups=2, 1); grid (M, (N+bn-1)/bn, B)
///   where `bn = min(group_size, 32) * 2`.
///
/// qvm_split_k (`quantized.cpp:322-323`):
///   group (bk=32, num_simdgroups=2, 1); grid (M, N/bn, B*split_k).
pub fn qvm_dispatch_shape(
    kernel: QvmKernel,
    m: u32,
    n: u32,
    b: u32,
    group_size: u32,
) -> ((u32, u32, u32), (u32, u32, u32)) {
    let bn: u32 = group_size.min(32) * 2; // 64 for gs ∈ {32, 64, 128}
    match kernel {
        QvmKernel::Standard => ((m, n.div_ceil(bn), b), (32, 2, 1)),
        QvmKernel::SplitK { split_k, .. } => ((m, n / bn, b * split_k), (32, 2, 1)),
    }
}

/// Format the kernel symbol name. Matches the `INST_QVM_*` macros
/// in `shaders/quantized_qvm.metal`.
pub fn qvm_kernel_name(
    kernel: QvmKernel,
    dtype: DequantDtype,
    scale_dtype: ScaleDtype,
    group_size: u32,
    bits: u32,
) -> String {
    let dtype = dtype.symbol_infix();
    let sdt = scale_dtype.symbol_infix();
    match kernel {
        QvmKernel::Standard => {
            format!("affine_qvm_{dtype}_s_{sdt}_gs_{group_size}_b_{bits}_batch_0")
        }
        QvmKernel::SplitK { .. } => {
            format!("affine_qvm_split_k_{dtype}_s_{sdt}_gs_{group_size}_b_{bits}")
        }
    }
}

/// `&'static str` view of [`qvm_kernel_name`] for the lowering pass.
pub fn qvm_kernel_static_name(
    kernel: QvmKernel,
    dtype: DequantDtype,
    scale_dtype: ScaleDtype,
    bits: u32,
    group_size: u32,
) -> &'static str {
    debug_assert_eq!(bits, 4, "qvm_kernel_static_name: only bits=4 is wired");
    if !matches!(group_size, 32 | 64 | 128) {
        panic!(
            "qvm_kernel_static_name: unsupported group_size={group_size} \
             — only 32, 64, 128 instantiated"
        );
    }
    use std::collections::HashMap;
    use std::sync::OnceLock;
    type Key = (
        std::mem::Discriminant<QvmKernel>,
        DequantDtype,
        ScaleDtype,
        u32,
    );
    static CACHE: OnceLock<std::sync::Mutex<HashMap<Key, &'static str>>> = OnceLock::new();
    let cache = CACHE.get_or_init(|| std::sync::Mutex::new(HashMap::new()));
    let mut guard = cache.lock().expect("qvm_kernel_static_name cache poisoned");
    let key = (
        std::mem::discriminant(&kernel),
        dtype,
        scale_dtype,
        group_size,
    );
    if let Some(&v) = guard.get(&key) {
        return v;
    }
    let owned = qvm_kernel_name(kernel, dtype, scale_dtype, group_size, 4);
    let leaked: &'static str = Box::leak(owned.into_boxed_str());
    guard.insert(key, leaked);
    leaked
}

/// MLX-affine int4 decode-matvec transpose=false dispatcher. Wraps
/// the `quantized_qvm` metallib's `affine_qvm` + `affine_qvm_split_k`
/// kernels.
///
/// Caller invariants (same as `MetalAffineQmmN`):
/// - `x`: `[M, K]` row-contiguous in `dtype`.
/// - `w`: `[K, N / pack_factor]` U32.
/// - `scales`, `biases`: `[K, N / group_size]` in `dtype`.
/// - `y`: `[M, N]` in `dtype` for the `Standard` kernel; for
///   `SplitK`, caller passes an intermediate scratch of shape
///   `[split_k, M, N]` and is responsible for the downstream
///   sum-reduce along axis 0 (mirrors
///   `strided_reduce_general_dispatch` at `quantized.cpp:415`).
/// - `bits` ∈ {4}.
/// - `group_size` ∈ {32, 64, 128}.
/// - `N % bn == 0` where `bn = min(group_size, 32) * 2 = 64` (the
///   `qvm` kernel ceils via `(N + bn - 1) / bn` so unaligned N is
///   safe at the grid level, but `qvm_split_k` uses `N / bn` with
///   no ceil — so it requires `N % 64 == 0`).
pub struct MetalAffineQvm {
    shader_cache: Arc<ShaderCache>,
}

impl MetalAffineQvm {
    pub fn new(device: Device) -> Result<Self, MetalStreamError> {
        Ok(Self {
            shader_cache: Arc::new(ShaderCache::new(device)?),
        })
    }

    pub fn with_shader_cache(shader_cache: Arc<ShaderCache>) -> Self {
        Self { shader_cache }
    }

    /// Returns the picked variant for the given K (so the caller
    /// knows whether to allocate a `[split_k, M, N]` scratch + emit
    /// a downstream sum-reduce).
    pub fn plan(&self, k: u32) -> QvmKernel {
        pick_qvm_kernel(k)
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
        scale_dtype: ScaleDtype,
        encoder: &ComputeCommandEncoderRef,
    ) -> Result<QvmKernel, MetalStreamError> {
        if bits != 4 {
            return Err(MetalStreamError::ShaderCompilationFailed(format!(
                "affine_qvm: only bits=4 is wired in P5, got bits={bits}"
            )));
        }
        if !matches!(group_size, 32 | 64 | 128) {
            return Err(MetalStreamError::ShaderCompilationFailed(format!(
                "affine_qvm: only group_size in {{32, 64, 128}} is wired, got {group_size}"
            )));
        }
        if b > 1 {
            return Err(MetalStreamError::ShaderCompilationFailed(format!(
                "affine_qvm: batched=1 (B={b}) not yet wired (B=1 in P5)"
            )));
        }
        if !n.is_multiple_of(group_size) {
            return Err(MetalStreamError::ShaderCompilationFailed(format!(
                "affine_qvm: N={n} not divisible by group_size={group_size} \
                 (transpose=false stores scales/biases per (K, N/gs))"
            )));
        }

        let kernel = pick_qvm_kernel(k);
        let kernel_name = qvm_kernel_name(kernel, dtype, scale_dtype, group_size, bits);
        // Function constants 0..5 (only the ones the picked variant
        // reads are needed, but the Metal compiler dead-codes unused
        // constant reads; we send all five for the splitk variant
        // and only K/N/M for the standard variant).
        let constants: Vec<ConstantValue> = match kernel {
            QvmKernel::Standard => vec![
                ConstantValue::int(0, k as i32),
                ConstantValue::int(1, n as i32),
                ConstantValue::int(2, m as i32),
            ],
            QvmKernel::SplitK {
                split_k,
                k_partition_size,
                final_block_size,
            } => vec![
                ConstantValue::int(0, k as i32),
                ConstantValue::int(1, n as i32),
                ConstantValue::int(2, m as i32),
                ConstantValue::int(3, k_partition_size as i32),
                ConstantValue::int(4, final_block_size as i32),
                ConstantValue::int(5, split_k as i32),
            ],
        };
        let pipeline = self
            .shader_cache
            .get_pipeline_specialized(&kernel_name, &constants)?;
        encoder.setComputePipelineState(&pipeline);

        // Buffer bindings 0..4 — same layout as MLX
        // `affine_qvm` kernel signature (`quantized.h:1601-1605`).
        // SAFETY: all buffer pointers are valid `Retained` objects;
        // `setBuffer:offset:atIndex:` only borrows them through the call.
        unsafe {
            encoder.setBuffer_offset_atIndex(Some(packed_w), 0, 0);
            encoder.setBuffer_offset_atIndex(Some(scales), 0, 1);
            encoder.setBuffer_offset_atIndex(Some(biases), 0, 2);
            encoder.setBuffer_offset_atIndex(Some(x), 0, 3);
            encoder.setBuffer_offset_atIndex(Some(y), 0, 4);
        }

        let (tg, tpg) = qvm_dispatch_shape(kernel, m, n, b, group_size);
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
        assert_eq!(pick_qmv_kernel(2048, 64, 4), QmvKernel::Quad { d: 64 });
        assert_eq!(pick_qmv_kernel(2048, 128, 4), QmvKernel::Quad { d: 128 });
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
            qmv_kernel_name(
                QmvKernel::Fast,
                DequantDtype::Bf16,
                ScaleDtype::F16,
                64,
                4,
                false
            ),
            "affine_qmv_fast_bf16_s_f16_gs_64_b_4_batch_0"
        );
        assert_eq!(
            qmv_kernel_name(
                QmvKernel::Generic,
                DequantDtype::F16,
                ScaleDtype::F16,
                32,
                4,
                true
            ),
            "affine_qmv_f16_s_f16_gs_32_b_4_batch_1"
        );
        assert_eq!(
            qmv_kernel_name(
                QmvKernel::Quad { d: 128 },
                DequantDtype::Bf16,
                ScaleDtype::F16,
                128,
                4,
                false
            ),
            "affine_qmv_quad_bf16_s_f16_gs_128_b_4_d_128_batch_0"
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
            qmm_t_kernel_name(
                QmmTKernel::Standard,
                DequantDtype::Bf16,
                ScaleDtype::F16,
                64,
                4,
                true
            ),
            "affine_qmm_t_bf16_s_f16_gs_64_b_4_alN_true_batch_0"
        );
        assert_eq!(
            qmm_t_kernel_name(
                QmmTKernel::Standard,
                DequantDtype::F16,
                ScaleDtype::F16,
                32,
                4,
                false
            ),
            "affine_qmm_t_f16_s_f16_gs_32_b_4_alN_false_batch_0"
        );
        assert_eq!(
            qmm_t_kernel_name(
                QmmTKernel::SplitK {
                    split_k: 4,
                    k_partition_size: 512
                },
                DequantDtype::Bf16,
                ScaleDtype::F16,
                128,
                4,
                true
            ),
            "affine_qmm_t_splitk_bf16_s_f16_gs_128_b_4_alN_true"
        );
    }

    #[test]
    fn qmm_t_dispatch_shape_matches_mlx_grid_dims() {
        // Standard: grid (ceil(N/32), ceil(M/32), B); group (32, 2, 2).
        let ((tx, ty, tz), (gx, gy, gz)) = qmm_t_dispatch_shape(QmmTKernel::Standard, 64, 2048, 1);
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
    fn qvm_kernel_pick_matches_mlx_rule() {
        // K < 1024 → Standard
        assert_eq!(pick_qvm_kernel(64), QvmKernel::Standard);
        assert_eq!(pick_qvm_kernel(1023), QvmKernel::Standard);
        // K == 1024 → SplitK with split_k=8
        let k = pick_qvm_kernel(1024);
        assert!(matches!(
            k,
            QvmKernel::SplitK {
                split_k: 8,
                k_partition_size: 128,
                final_block_size: 128,
            }
        ));
        // K = 4096 → SplitK with split_k=8, split_D=512
        let k = pick_qvm_kernel(4096);
        assert!(matches!(
            k,
            QvmKernel::SplitK {
                split_k: 8,
                k_partition_size: 512,
                final_block_size: 512,
            }
        ));
        // K = 8193 → split_k=32 (K > 8192)
        let k = pick_qvm_kernel(8193);
        assert!(matches!(k, QvmKernel::SplitK { split_k: 32, .. }));
        // K = 8200 → split_k=32, split_D=257, final_block_size=200
        // (K - 31*257 = 8200 - 7967 = 233, but 8200/32 ceil = 257)
        let k = pick_qvm_kernel(8200);
        if let QvmKernel::SplitK {
            split_k,
            k_partition_size,
            final_block_size,
        } = k
        {
            assert_eq!(split_k, 32);
            assert_eq!(k_partition_size, 257); // ceil(8200/32)
            assert_eq!(final_block_size, 8200 - 31 * 257);
        } else {
            panic!("expected SplitK, got {k:?}");
        }
    }

    #[test]
    fn qvm_dispatch_shape_matches_mlx() {
        // qvm: grid (M, ceil(N / bn), B); bn = min(gs, 32)*2 = 64
        let ((tx, ty, tz), (gx, gy, gz)) = qvm_dispatch_shape(QvmKernel::Standard, 1, 4096, 1, 64);
        assert_eq!((tx, ty, tz), (1, 4096 / 64, 1));
        assert_eq!((gx, gy, gz), (32, 2, 1));

        // qvm_split_k: grid (M, N/bn, B * split_k)
        let kernel = QvmKernel::SplitK {
            split_k: 8,
            k_partition_size: 512,
            final_block_size: 512,
        };
        let ((tx, ty, tz), _) = qvm_dispatch_shape(kernel, 1, 4096, 1, 64);
        assert_eq!((tx, ty, tz), (1, 4096 / 64, 8));
    }

    #[test]
    fn qvm_kernel_name_matches_metallib_symbols() {
        assert_eq!(
            qvm_kernel_name(
                QvmKernel::Standard,
                DequantDtype::Bf16,
                ScaleDtype::F16,
                64,
                4
            ),
            "affine_qvm_bf16_s_f16_gs_64_b_4_batch_0"
        );
        assert_eq!(
            qvm_kernel_name(
                QvmKernel::SplitK {
                    split_k: 8,
                    k_partition_size: 512,
                    final_block_size: 512
                },
                DequantDtype::F16,
                ScaleDtype::F16,
                128,
                4
            ),
            "affine_qvm_split_k_f16_s_f16_gs_128_b_4"
        );
    }

    #[test]
    fn qmm_n_kernel_name_matches_metallib_symbols() {
        assert_eq!(
            qmm_n_kernel_name(DequantDtype::Bf16, ScaleDtype::F16, 64, 4),
            "affine_qmm_n_bf16_s_f16_gs_64_b_4_batch_0"
        );
        assert_eq!(
            qmm_n_kernel_name(DequantDtype::F16, ScaleDtype::F16, 32, 4),
            "affine_qmm_n_f16_s_f16_gs_32_b_4_batch_0"
        );
    }

    #[test]
    fn qmm_n_dispatch_shape_matches_mlx() {
        // qmm_n: grid (ceil(N/32), ceil(M/32), B); group (32, 2, 2)
        let ((tx, ty, tz), (gx, gy, gz)) = qmm_n_dispatch_shape(64, 2048, 1);
        assert_eq!((tx, ty, tz), (64, 2, 1));
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
