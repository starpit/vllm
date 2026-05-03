// SPDX-License-Identifier: Apache-2.0
//! Model layers using `GpuTensor`.
//!
//! These are minimal, inference-only layer types. Weights are stored as
//! `GpuTensor` (raw GPU pointers). Forward passes use cuBLAS GEMM from
//! the `GpuDevice` and fused CUDA kernels.

use anyhow::Result;

use ferrite_cuda_core::alloc::{CachingAllocator, OwnedTensor};
use ferrite_cuda_core::cublas::CublasHandle;
use ferrite_cuda_core::tensor::{GpuTensor, TensorView};
use ferrite_cuda_core::weights::GpuWeights;

#[cfg(feature = "nccl")]
use ferrite_cuda_core::nccl::NcclGroup;
#[cfg(feature = "nccl")]
use std::sync::Arc;

// ---------------------------------------------------------------------------
// Packed-source fallback (Phi-3 family): on a missing `.weight`, detect
// the conventional packed parent (`qkv_proj` for q/k/v; `gate_up_proj` for
// gate/up) and ask `GpuWeights` to synthesize the per-slice virtual
// entries before the take. Idempotent — if the sibling has already been
// synthesized (by a prior call for q_proj, say), the packed parent is
// already gone and this is a no-op.
// ---------------------------------------------------------------------------

/// Concat-load when every source weight is in the GGUF dense map
/// (FP16/F32 GGUFs land all weights here). `take()` already checks
/// `gguf_dense` first and returns the GpuTensor; we then D2D-copy
/// each [rows, cols] slab into a packed [sum(rows), cols] buffer.
/// Bias follows the same path.
fn load_gguf_dense_concat(
    weights: &mut GpuWeights,
    prefixes: &[&str],
    stream: cudarc::driver::sys::CUstream,
) -> Result<LinearLayer> {
    debug_assert!(!prefixes.is_empty());
    let mut parts: Vec<GpuTensor> = Vec::with_capacity(prefixes.len());
    for p in prefixes {
        let name = format!("{p}.weight");
        let t = weights.take(&name)?;
        if t.ndim() != 2 {
            anyhow::bail!(
                "load_gguf_dense_concat: `{name}` has rank {}, expected 2",
                t.ndim()
            );
        }
        parts.push(t);
    }
    let cols = parts[0].dim(1);
    let dtype = parts[0].dtype();
    for (i, t) in parts.iter().enumerate() {
        if t.dim(1) != cols {
            anyhow::bail!(
                "load_gguf_dense_concat: `{}` in_features {} != {cols}",
                prefixes[i],
                t.dim(1)
            );
        }
        if t.dtype() != dtype {
            anyhow::bail!(
                "load_gguf_dense_concat: `{}` dtype {:?} != {dtype:?}",
                prefixes[i],
                t.dtype()
            );
        }
    }
    let total_rows: usize = parts.iter().map(|t| t.dim(0)).sum();
    let elem = dtype.size_bytes();
    let total_bytes = total_rows * cols * elem;
    let dst = unsafe { ferrite_cuda_core::driver::mem_alloc(total_bytes)? };
    weights.record_alloc(dst, total_bytes);
    let mut row_off: usize = 0;
    for t in &parts {
        let part_rows = t.dim(0);
        let part_bytes = part_rows * cols * elem;
        unsafe {
            ferrite_cuda_core::driver::memcpy_dtod_async(
                (dst as *mut u8).add(row_off * cols * elem),
                t.raw_ptr(),
                part_bytes,
                stream,
            )?;
        }
        row_off += part_rows;
    }
    let weight = unsafe { GpuTensor::new(dst, &[total_rows, cols], dtype) };

    // Bias: either all branches ship one or none.
    let bias_names: Vec<String> = prefixes.iter().map(|p| format!("{p}.bias")).collect();
    let any_bias = weights.contains(&bias_names[0]);
    let bias = if any_bias {
        let mut bias_parts: Vec<GpuTensor> = Vec::with_capacity(prefixes.len());
        for n in &bias_names {
            if !weights.contains(n) {
                anyhow::bail!(
                    "load_gguf_dense_concat: inconsistent bias — `{}` exists but `{n}` is missing",
                    bias_names[0]
                );
            }
            bias_parts.push(weights.take(n)?);
        }
        let bias_dtype = bias_parts[0].dtype();
        let bias_total_rows: usize = bias_parts.iter().map(|t| t.dim(0)).sum();
        let bias_elem = bias_dtype.size_bytes();
        let bias_bytes = bias_total_rows * bias_elem;
        let bdst = unsafe { ferrite_cuda_core::driver::mem_alloc(bias_bytes)? };
        weights.record_alloc(bdst, bias_bytes);
        let mut bias_off: usize = 0;
        for bt in &bias_parts {
            let n = bt.dim(0);
            unsafe {
                ferrite_cuda_core::driver::memcpy_dtod_async(
                    (bdst as *mut u8).add(bias_off * bias_elem),
                    bt.raw_ptr(),
                    n * bias_elem,
                    stream,
                )?;
            }
            bias_off += n;
        }
        Some(unsafe { GpuTensor::new(bdst, &[bias_total_rows], bias_dtype) })
    } else {
        None
    };

    Ok(LinearLayer::Dense(Linear { weight, bias }))
}

/// Narrow a replicated full-size 1D bias down to the rank's shard.
///
/// GGUF biases ship 1D and land in `gguf_dense` full-sized on every
/// rank (the shard-kind rule `Replicate`s ndim<2 tensors). For a
/// column-parallel Linear at tp>1, each rank's weight produces
/// `[tokens, out_full/world]` and the bias must match that second dim
/// — otherwise `bias_add_inplace` reads the wrong slice. This returns
/// a view into the same GPU allocation offset by `rank * (out_full/world)`
/// elements.
///
/// `per_rank_out` is the weight's per-rank out-feature count (storage
/// row count for quantized, or `dim(0)` for dense).
fn per_rank_bias_slice(
    full: ferrite_cuda_core::tensor::GpuTensor,
    rank: usize,
    world: usize,
    per_rank_out: usize,
) -> ferrite_cuda_core::tensor::GpuTensor {
    debug_assert_eq!(full.ndim(), 1, "GGUF bias must be 1D");
    debug_assert_eq!(
        full.dim(0),
        per_rank_out * world,
        "bias length must equal per-rank out × world"
    );
    let start = rank * per_rank_out;
    full.narrow_dim0(start, per_rank_out)
}

fn try_synthesize_packed_slice(weights: &mut GpuWeights, prefix: &str) -> Result<()> {
    let (parent, suffix) = match prefix.rsplit_once('.') {
        Some(split) => split,
        None => return Ok(()),
    };
    let (packed_suffix, targets): (&str, &[&str]) = match suffix {
        "q_proj" | "k_proj" | "v_proj" => ("qkv_proj", &["q_proj", "k_proj", "v_proj"]),
        "gate_proj" | "up_proj" => ("gate_up_proj", &["gate_proj", "up_proj"]),
        _ => return Ok(()),
    };
    let packed_prefix = format!("{parent}.{packed_suffix}");
    weights.synthesize_packed_row_split(&packed_prefix, targets)?;
    Ok(())
}

// ---------------------------------------------------------------------------
// Linear
// ---------------------------------------------------------------------------

/// Dense linear layer: y = x @ W^T + b
///
/// Weight is stored in `[out_features, in_features]` layout (NOT pre-transposed).
/// cuBLAS GEMM handles the transpose internally via `CUBLAS_OP_T`, which is
/// more efficient than a separate transpose copy.
pub struct Linear {
    pub weight: GpuTensor,       // [out_features, in_features]
    pub bias: Option<GpuTensor>, // [out_features]
}

impl Linear {
    /// Create from explicit weight and bias tensors.
    pub fn new(weight: GpuTensor, bias: Option<GpuTensor>) -> Self {
        debug_assert_eq!(weight.ndim(), 2);
        if let Some(ref b) = bias {
            debug_assert_eq!(b.ndim(), 1);
            debug_assert_eq!(b.dim(0), weight.dim(0));
        }
        Self { weight, bias }
    }

    /// Load from `GpuWeights` by prefix (e.g. "model.layers.0.self_attn.q_proj").
    ///
    /// Packed-source fallback: some checkpoints (Phi-3 family) ship
    /// `self_attn.qkv_proj.weight` and `mlp.gate_up_proj.weight` in place
    /// of the per-slice `q_proj`/`k_proj`/`v_proj` and `gate_proj`/`up_proj`
    /// tensors the DSL body references. When `{prefix}.weight` is missing,
    /// this function detects the packed-source convention by `prefix`'s
    /// last path segment and asks `GpuWeights` to synthesize virtual
    /// per-slice entries (even row-wise split) before the take. The
    /// split is recoverable — later sibling calls reuse the synthesized
    /// entries, so the packed tensor is walked once.
    pub fn load(weights: &mut GpuWeights, prefix: &str) -> Result<Self> {
        let weight_name = format!("{prefix}.weight");
        let bias_name = format!("{prefix}.bias");

        if !weights.contains(&weight_name) {
            try_synthesize_packed_slice(weights, prefix)?;
        }

        let weight = weights.take(&weight_name)?;
        let bias = if weights.contains(&bias_name) {
            Some(weights.take(&bias_name)?)
        } else {
            None
        };
        Ok(Self::new(weight, bias))
    }

    /// Load a tensor-parallel slice of `{prefix}.weight` (and bias)
    /// per Python vLLM's `ColumnParallelLinear` (`dim = 0`,
    /// column-parallel) and `RowParallelLinear` (`dim = 1`,
    /// row-parallel) conventions.
    ///
    /// - **`dim == 0` (column-parallel: `q_proj`, `k_proj`, `v_proj`,
    ///   `gate_proj`, `up_proj`, `lm_head`, `embed_tokens`).** The
    ///   weight is sliced along its output dim (dim 0 of the
    ///   `[out, in]` matrix). The bias, if present, is also sliced
    ///   along dim 0 — each rank holds its own contiguous chunk.
    ///   Mirrors Python `ColumnParallelLinear.weight_loader` →
    ///   `loaded_weight.narrow(output_dim=0, ...)`.
    ///
    /// - **`dim == 1` (row-parallel: `o_proj`, `down_proj`).** The
    ///   weight is sliced along its input dim (dim 1 of the
    ///   `[out, in]` matrix). The bias is **replicated full-size on
    ///   rank 0 only**, `None` on other ranks. The forward path adds
    ///   bias before the cross-rank `AllReduce`-sum, which produces
    ///   exactly one bias contribution to the residual stream —
    ///   mirrors Python `RowParallelLinear.forward` line 1543:
    ///   `bias_ = None if (self.tp_rank > 0 ...) else self.bias`.
    ///
    /// `world == 1` is supported and degrades to the unsharded
    /// `Self::load` semantics (bias is loaded on rank 0, which is
    /// the only rank). `dim` must be 0 or 1 — anything else panics.
    pub fn load_sharded(
        weights: &mut GpuWeights,
        prefix: &str,
        dim: usize,
        rank: usize,
        world: usize,
    ) -> Result<Self> {
        assert!(
            dim < 2,
            "Linear::load_sharded: dim must be 0 or 1, got {dim}"
        );
        let weight_name = format!("{prefix}.weight");
        let bias_name = format!("{prefix}.bias");

        if !weights.contains(&weight_name) {
            try_synthesize_packed_slice(weights, prefix)?;
        }

        let weight = weights.take_shard(&weight_name, dim, rank, world)?;
        let bias = if weights.contains(&bias_name) {
            if dim == 0 {
                // Column-parallel: bias shards along dim 0 too.
                Some(weights.take_shard(&bias_name, 0, rank, world)?)
            } else {
                // Row-parallel: bias is replicated full-size on rank 0,
                // None on other ranks. The bias entry stays unconsumed
                // in `weights` on rank > 0 — its mmap page is freed
                // when `GpuWeights` drops.
                if rank == 0 {
                    Some(weights.take(&bias_name)?)
                } else {
                    None
                }
            }
        } else {
            None
        };
        Ok(Self::new(weight, bias))
    }

    /// Forward: y = x @ W^T (+ bias)
    ///
    /// `x`: `[num_tokens, in_features]`
    /// Returns: `[num_tokens, out_features]` as OwnedTensor from caching allocator.
    ///
    /// # Safety
    /// All tensors must be valid GPU memory. cuBLAS handle must be on the correct stream.
    pub unsafe fn forward(
        &self,
        x: TensorView<'_>,
        cublas: &mut CublasHandle,
        alloc: &mut CachingAllocator,
    ) -> OwnedTensor {
        debug_assert_eq!(x.ndim(), 2);
        debug_assert_eq!(x.dim(1), self.weight.dim(1), "Linear: input dim mismatch");

        if let Some(bias) = self.bias {
            cublas.gemm_bias(*x, self.weight, bias, alloc)
        } else {
            cublas.gemm(*x, self.weight, alloc)
        }
    }

    pub fn out_features(&self) -> usize {
        self.weight.dim(0)
    }

    pub fn in_features(&self) -> usize {
        self.weight.dim(1)
    }
}

// ---------------------------------------------------------------------------
// MarlinLinear (INT4 quantized via Marlin GEMM)
// ---------------------------------------------------------------------------

/// Quantized linear layer using Marlin INT4×FP16→FP16 GEMM.
///
/// Weights are repacked to Marlin tiled format at load time.
/// Supports both AWQ (has zero points) and GPTQ (symmetric or with zeros).
pub struct MarlinLinear {
    /// Marlin-tiled packed INT4 weights.
    pub qweight: GpuTensor,
    /// Per-group scales `[num_groups, size_n]`, permuted for Marlin.
    pub scales: GpuTensor,
    /// Packed zero points (AWQ) or None (GPTQ symmetric).
    pub zeros: Option<GpuTensor>,
    /// Group index for act_order (desc_act) or None.
    pub g_idx: Option<GpuTensor>,
    /// Sort indices for act_order or None.
    pub g_idx_sort_indices: Option<GpuTensor>,
    /// `[num_sms]` i32 workspace for Marlin barrier sync.
    pub workspace: GpuTensor,
    /// Input features (unquantized K dimension).
    pub size_k: usize,
    /// Output features (N dimension).
    pub size_n: usize,
    /// Quantization group size.
    pub group_size: usize,
    /// Number of groups.
    pub num_groups: usize,
    /// Whether this layer has zero points.
    pub has_zp: bool,
    /// Whether act_order (desc_act) is enabled.
    pub has_act_order: bool,
    /// Marlin b_type_id: 0 = GPTQ (uint4b8), 1 = AWQ (uint4).
    pub b_type_id: i32,
    /// Device ID for the Marlin kernel.
    pub device_id: i32,
    /// Optional bias `[size_n]` — added after Marlin GEMM.
    pub bias: Option<GpuTensor>,
}

impl MarlinLinear {
    /// Forward: y = marlin_gemm(x, qweight, scales, zeros)
    ///
    /// `x`: `[num_tokens, size_k]` (F16 or BF16)
    /// Returns: `[num_tokens, size_n]`
    pub unsafe fn forward(
        &self,
        x: TensorView<'_>,
        alloc: &mut CachingAllocator,
        stream: cudarc::driver::sys::CUstream,
    ) -> OwnedTensor {
        let size_m = x.dim(0);
        debug_assert_eq!(
            x.dim(1),
            self.size_k,
            "MarlinLinear: input dim {} != size_k {}",
            x.dim(1),
            self.size_k
        );
        // Don't pass bias to Marlin kernel (would need permutation).
        // Instead, add bias after the GEMM with a simple broadcast add.
        let out = crate::kernels::marlin_gemm(
            *x,
            self.qweight,
            self.scales,
            self.zeros,
            self.g_idx,
            self.g_idx_sort_indices,
            None,
            self.workspace,
            size_m,
            self.size_n,
            self.size_k,
            self.num_groups,
            self.group_size,
            self.has_act_order,
            self.has_zp,
            self.b_type_id,
            self.device_id,
            alloc,
            stream,
        );

        if let Some(bias) = self.bias {
            crate::kernels::bias_add_inplace(out.as_gpu_tensor(), bias, stream);
        }

        out
    }

    pub fn out_features(&self) -> usize {
        self.size_n
    }

    pub fn in_features(&self) -> usize {
        self.size_k
    }
}

// ---------------------------------------------------------------------------
// Bnb4bitLinear (BitsAndBytes NF4/FP4 4-bit quantized)
// ---------------------------------------------------------------------------

/// BitsAndBytes 4-bit quantized linear layer.
///
/// Forward: dequantize packed NF4/FP4 weights to BF16 scratch buffer, then cuBLAS GEMM.
/// Matches Python `bitsandbytes.matmul_4bit` behavior.
pub struct Bnb4bitLinear {
    /// Packed NF4/FP4 nibbles `[num_packed_bytes]` U8 (2 values per byte).
    pub packed_weight: GpuTensor,
    /// Per-block scale factors `[num_blocks]` F32.
    pub absmax: GpuTensor,
    /// NF4 or FP4 lookup table `[16]` F32 on GPU.
    pub code: GpuTensor,
    /// Dequantization scratch buffer `[out_features, in_features]` BF16 on GPU.
    /// Shared across layers — the caller allocates once and passes to all layers.
    pub dequant_scratch: GpuTensor,
    /// Original output features (rows).
    pub out_features: usize,
    /// Original input features (cols).
    pub in_features: usize,
    /// Block size for quantization (typically 64).
    pub blocksize: usize,
    /// Optional bias `[out_features]`.
    pub bias: Option<GpuTensor>,
}

impl Bnb4bitLinear {
    /// Forward: dequantize → cuBLAS GEMM.
    ///
    /// `x`: `[num_tokens, in_features]`
    /// Returns: `[num_tokens, out_features]`
    pub unsafe fn forward(
        &self,
        x: TensorView<'_>,
        cublas: &mut CublasHandle,
        alloc: &mut CachingAllocator,
        stream: cudarc::driver::sys::CUstream,
    ) -> OwnedTensor {
        // BNB stores weights in [out_features, in_features] order (same as original W).
        // Dequant produces the flat W data, reshape as [out, in], then cuBLAS does x @ W^T.
        let weight_view = self
            .dequant_scratch
            .reshape(&[self.out_features, self.in_features]);

        // Dequantize into scratch buffer (reused across layers).
        crate::kernels::dequantize_bnb4bit(
            self.packed_weight,
            self.absmax,
            self.code,
            weight_view,
            self.blocksize,
            stream,
        );

        // cuBLAS GEMM: x @ weight_view^T
        if let Some(bias) = self.bias {
            cublas.gemm_bias(*x, weight_view, bias, alloc)
        } else {
            cublas.gemm(*x, weight_view, alloc)
        }
    }

    pub fn out_features(&self) -> usize {
        self.out_features
    }

    pub fn in_features(&self) -> usize {
        self.in_features
    }
}

// ---------------------------------------------------------------------------
// GgmlLinear (GGML quantized via llama.cpp kernels)
// ---------------------------------------------------------------------------

/// Quantized linear layer holding raw GGML-quantized bytes on GPU.
///
/// Forward dispatches to the appropriate dequant-matvec kernel based on GGML
/// dtype and batch size:
/// - BS=1: `dequantize_mul_mat_vec` (fused dequant + dot product)
/// - BS>1: quantize activations to Q8_1, then integer dot products
pub struct GgmlLinear {
    pub storage: crate::ggml::GgmlStorage,
    pub bias: Option<GpuTensor>,
}

impl GgmlLinear {
    /// Forward: y = ggml_matmul(weight, x) + bias
    ///
    /// Activations must be f32 (GGML kernels operate on f32).
    /// Output is f32 `[num_tokens, out_features]`.
    pub unsafe fn forward(
        &self,
        x: TensorView<'_>,
        alloc: &mut CachingAllocator,
        stream: cudarc::driver::sys::CUstream,
    ) -> OwnedTensor {
        let input_dtype = x.dtype();

        // GGML kernels require f32 activations — cast if needed.
        let cast_buf = if input_dtype != ferrite_cuda_core::dtype::DType::F32 {
            Some(crate::kernels::cast_logits_to_f32(*x, alloc, stream))
        } else {
            None
        };
        let x_f32 = if let Some(ref cast) = cast_buf {
            cast.as_gpu_tensor()
        } else {
            *x
        };

        let out_f32 = crate::ggml::ggml_matmul(&self.storage, x_f32, alloc, stream);
        drop(cast_buf);

        // ggml_matmul output is F32; bias was cast to model_dtype
        // (typically BF16) at GGUF load time so the dense path can
        // consume it directly. Cast back to F32 here when dtypes
        // disagree — `bias_add_inplace` reinterprets the bias
        // pointer as `out.dtype()` so a BF16 source against F32 out
        // would be silent garbage. The temp F32 buffer must
        // outlive the (async) `bias_add_inplace` kernel: hoist its
        // binding to function scope so the caching allocator can't
        // hand the same pointer to a downstream `cast_from_f32`
        // launch on the same stream — see
        // `feedback_tensorview_for_async_gpu`.
        let _bias_keepalive = if let Some(bias) = self.bias {
            if bias.dtype() == ferrite_cuda_core::dtype::DType::F32 {
                crate::kernels::bias_add_inplace(out_f32.as_gpu_tensor(), bias, stream);
                None
            } else {
                let bias_f32 = crate::kernels::cast_logits_to_f32(bias, alloc, stream);
                crate::kernels::bias_add_inplace(
                    out_f32.as_gpu_tensor(),
                    bias_f32.as_gpu_tensor(),
                    stream,
                );
                Some(bias_f32)
            }
        } else {
            None
        };

        // Cast back to original dtype if we converted to f32.
        if input_dtype != ferrite_cuda_core::dtype::DType::F32 {
            let out_f32_gpu = out_f32.as_gpu_tensor();
            let result = crate::kernels::cast_from_f32(out_f32_gpu, input_dtype, alloc, stream);
            drop(out_f32);
            drop(_bias_keepalive);
            result
        } else {
            drop(_bias_keepalive);
            out_f32
        }
    }

    pub fn out_features(&self) -> usize {
        self.storage.nrows
    }

    pub fn in_features(&self) -> usize {
        self.storage.ncols
    }
}

// ---------------------------------------------------------------------------
// Fp8Linear (FP8 E4M3 quantized via cublasLt FP8 GEMM)
// ---------------------------------------------------------------------------

/// FP8 (E4M3) quantized linear layer.
///
/// Weights stored as FP8 `[out_features, in_features]` with a per-tensor
/// f32 weight scale on GPU. Activations are dynamically quantized to FP8
/// per-token at runtime (or statically if `input_scale` is provided).
///
/// Forward: quantize(x) → FP8 GEMM → BF16 output.
/// Matches Python vLLM's `Fp8LinearMethod`.
pub struct Fp8Linear {
    /// FP8 E4M3 weights `[out_features, in_features]`.
    pub weight: GpuTensor,
    /// Per-tensor weight scale: single f32 scalar on GPU.
    pub weight_scale: GpuTensor,
    /// Pre-calibrated input scale (static activation quantization).
    /// If `None`, uses dynamic per-token quantization.
    pub input_scale: Option<GpuTensor>,
    /// Optional bias `[out_features]` in output dtype (BF16/F16).
    pub bias: Option<GpuTensor>,
    /// Output dtype (BF16 or F16) — determines GEMM output and bias dtype.
    pub output_dtype: ferrite_cuda_core::dtype::DType,
}

impl Fp8Linear {
    /// Forward: quantize activations → CUTLASS FP8 GEMM → output in output_dtype.
    ///
    /// Uses fused CUTLASS `cutlass_scaled_mm` (single kernel launch) with per-row
    /// activation scales and per-tensor weight scale in the epilogue.
    /// Matches Python vLLM's `cutlass_scaled_mm` exactly.
    pub unsafe fn forward(
        &self,
        x: TensorView<'_>,
        _cublas: &mut CublasHandle,
        alloc: &mut CachingAllocator,
        stream: cudarc::driver::sys::CUstream,
    ) -> OwnedTensor {
        debug_assert_eq!(x.ndim(), 2);
        debug_assert_eq!(
            x.dim(1),
            self.weight.dim(1),
            "Fp8Linear: input dim mismatch"
        );

        if let Some(ref input_scale) = self.input_scale {
            // Static activation quantization: scalar input_scale + scalar weight_scale.
            // Quantize with pre-calibrated scale, then fused CUTLASS GEMM.
            let a_scale = input_scale.as_ptr::<f32>();
            let x_fp8 = crate::kernels::scaled_fp8_quant_static(*x, a_scale, alloc, stream);
            // input_scale is [1] (scalar) — CUTLASS handles scalar a_scale correctly.
            let result = if let Some(bias) = self.bias {
                crate::kernels::cutlass_scaled_mm_with_bias(
                    x_fp8.as_gpu_tensor(),
                    self.weight,
                    *input_scale,
                    self.weight_scale,
                    bias,
                    self.output_dtype,
                    alloc,
                    stream,
                )
            } else {
                crate::kernels::cutlass_scaled_mm(
                    x_fp8.as_gpu_tensor(),
                    self.weight,
                    *input_scale,
                    self.weight_scale,
                    self.output_dtype,
                    alloc,
                    stream,
                )
            };
            drop(x_fp8);
            result
        } else {
            // Dynamic per-token activation quantization.
            // Uses CUTLASS cutlass_scaled_mm with fused per-row scale_a epilogue —
            // single kernel launch, exactly matching Python vLLM.
            //
            // 1. Quantize activations: BF16 → FP8 + per-token scales [M]
            // 2. CUTLASS FP8 GEMM with per-token a_scales + per-tensor b_scale
            //    fused into the epilogue. ONE kernel launch.
            let (x_fp8, x_scales) = crate::kernels::scaled_fp8_quant_dynamic(*x, alloc, stream);
            let result = if let Some(bias) = self.bias {
                crate::kernels::cutlass_scaled_mm_with_bias(
                    x_fp8.as_gpu_tensor(),
                    self.weight,
                    x_scales.as_gpu_tensor(),
                    self.weight_scale,
                    bias,
                    self.output_dtype,
                    alloc,
                    stream,
                )
            } else {
                crate::kernels::cutlass_scaled_mm(
                    x_fp8.as_gpu_tensor(),
                    self.weight,
                    x_scales.as_gpu_tensor(),
                    self.weight_scale,
                    self.output_dtype,
                    alloc,
                    stream,
                )
            };
            drop(x_fp8);
            drop(x_scales);
            result
        }
    }

    pub fn out_features(&self) -> usize {
        self.weight.dim(0)
    }

    pub fn in_features(&self) -> usize {
        self.weight.dim(1)
    }
}

// ---------------------------------------------------------------------------
// LinearLayer (enum dispatch: Dense, Marlin, Ggml, Bnb4bit, or Fp8)
// ---------------------------------------------------------------------------

/// Unified linear layer — dense (cuBLAS), Marlin INT4, GGML quantized, BNB 4-bit, or FP8.
///
/// Models use this everywhere they currently use `Linear`. The factory decides
/// at load time which variant to create based on weight format.
pub enum LinearLayer {
    Dense(Linear),
    Marlin(Box<MarlinLinear>),
    Ggml(Box<GgmlLinear>),
    /// Heterogeneous-dtype concat of GGUF linears. Used by
    /// `load_dense_concat_or_ggml` when the source tensors don't
    /// share a `GgmlDType` (e.g. a Q4_K_M Llama-3.2-1B has gate_proj
    /// at Q4_K but up_proj at Q6_K). Forward computes each branch
    /// separately and concatenates outputs along dim 1, producing
    /// the same `[tokens, sum(out_features)]` layout the
    /// homogeneous packed `GgmlLinear` produces — so downstream
    /// `silu_and_mul_fused` (which splits at `intermediate_size`)
    /// works without per-variant kernel changes.
    GgmlConcat(Vec<GgmlLinear>),
    Bnb4bit(Box<Bnb4bitLinear>),
    Fp8(Box<Fp8Linear>),
    Fp8Block(Box<Fp8BlockLinear>),
}

impl LinearLayer {
    /// Forward: y = x @ W^T (dense) or quantized GEMM variant.
    pub unsafe fn forward(
        &self,
        x: TensorView<'_>,
        cublas: &mut CublasHandle,
        alloc: &mut CachingAllocator,
        stream: cudarc::driver::sys::CUstream,
    ) -> OwnedTensor {
        match self {
            Self::Dense(l) => l.forward(x, cublas, alloc),
            Self::Marlin(l) => l.forward(x, alloc, stream),
            Self::Ggml(l) => {
                if ggml_probe::use_reference() {
                    // FERRITE_USE_REFERENCE=1: bypass ggml_matmul entirely, use
                    // dequant-to-f32 + cuBLAS F32 GEMM + optional bias. If this
                    // produces coherent inference, the MMVQ/DMMV kernels are the
                    // bug; if garbage, the problem is upstream (weight bytes).
                    return ggml_probe::reference_forward(
                        &l.storage, x, l.bias, cublas, alloc, stream,
                    );
                }
                if ggml_probe::is_enabled() {
                    ggml_probe::compare(&l.storage, x, cublas, alloc, stream);
                }
                l.forward(x, alloc, stream)
            }
            Self::GgmlConcat(branches) => {
                // Heterogeneous-dtype gate/up packing for Q4_K_M-style
                // mixed quants. Compute each branch separately, then
                // concat along dim 1 into the same `[tokens, sum(out)]`
                // layout the homogeneous packed `Ggml` arm produces —
                // so downstream `silu_and_mul_fused(out, intermediate)`
                // sees identical bytes regardless of which arm
                // produced them.
                debug_assert!(!branches.is_empty(), "GgmlConcat with no branches");
                let num_tokens = x.dim(0);
                let total_out: usize = branches.iter().map(|b| b.out_features()).sum();
                // Output dtype follows the input (the per-branch
                // forwards cast back to the input dtype if needed).
                let out_dtype = x.dtype();
                let packed = alloc.alloc_tensor(&[num_tokens, total_out], out_dtype);
                let row_stride_bytes = total_out * out_dtype.size_bytes();
                let elem = out_dtype.size_bytes();
                let mut col_offset_elems: usize = 0;
                // Hold every per-branch `part` alive until the loop
                // exits so the CachingAllocator can't recycle a
                // dropped branch's GPU buffer for the NEXT branch's
                // matmul allocation while D2D copies from the dropped
                // branch are still queued on the stream — same
                // use-after-free pattern documented in
                // `feedback_tensorview_for_async_gpu`.
                let mut keepalive: Vec<OwnedTensor> = Vec::with_capacity(branches.len());
                for branch in branches {
                    let part = if ggml_probe::use_reference() {
                        ggml_probe::reference_forward(
                            &branch.storage,
                            x,
                            branch.bias,
                            cublas,
                            alloc,
                            stream,
                        )
                    } else {
                        branch.forward(x, alloc, stream)
                    };
                    let part_view = part.as_gpu_tensor();
                    let part_out = branch.out_features();
                    let part_row_bytes = part_out * elem;
                    // Strided per-row D2D copy of `part` columns into
                    // `packed[:, col_offset:col_offset + part_out]`.
                    for row in 0..num_tokens {
                        let src = (part_view.raw_ptr() as *const u8).add(row * part_row_bytes);
                        let dst = (packed.as_mut_ptr() as *mut u8)
                            .add(row * row_stride_bytes + col_offset_elems * elem);
                        ferrite_cuda_core::driver::memcpy_dtod_async(
                            dst,
                            src,
                            part_row_bytes,
                            stream,
                        )
                        .expect("dtod copy in GgmlConcat::forward");
                    }
                    col_offset_elems += part_out;
                    keepalive.push(part);
                }
                drop(keepalive);
                packed
            }
            Self::Bnb4bit(l) => l.forward(x, cublas, alloc, stream),
            Self::Fp8(l) => l.forward(x, cublas, alloc, stream),
            Self::Fp8Block(l) => l.forward(x, cublas, alloc, stream),
        }
    }

    /// Access the raw dense weight tensor. Panics if quantized —
    /// CUTLASS standalone GEMM only works with dense bf16 weights.
    pub fn dense_weight(&self) -> ferrite_cuda_core::tensor::GpuTensor {
        match self {
            Self::Dense(l) => l.weight,
            Self::Marlin(_) => panic!("dense_weight() called on Marlin LinearLayer"),
            Self::Ggml(s) => {
                eprintln!(
                    "[dense_weight] called on Ggml LinearLayer (storage dtype={:?} \
                     shape=[{}, {}]). Call site backtrace:\n{}",
                    s.storage.dtype,
                    s.storage.nrows,
                    s.storage.ncols,
                    std::backtrace::Backtrace::force_capture()
                );
                panic!("dense_weight() called on Ggml LinearLayer");
            }
            Self::GgmlConcat(branches) => {
                eprintln!(
                    "[dense_weight] called on GgmlConcat LinearLayer ({} branches: {:?}). \
                     Call site backtrace:\n{}",
                    branches.len(),
                    branches
                        .iter()
                        .map(|b| (b.storage.dtype, b.storage.nrows, b.storage.ncols))
                        .collect::<Vec<_>>(),
                    std::backtrace::Backtrace::force_capture()
                );
                panic!("dense_weight() called on GgmlConcat LinearLayer");
            }
            Self::Bnb4bit(_) => panic!("dense_weight() called on Bnb4bit LinearLayer"),
            Self::Fp8(_) => panic!("dense_weight() called on Fp8 LinearLayer"),
            Self::Fp8Block(_) => panic!("dense_weight() called on Fp8Block LinearLayer"),
        }
    }

    /// Access the bias tensor from a dense layer. Returns the bias
    /// `GpuTensor` or `None` if the layer has no bias. Panics on
    /// quantized variants — solver only supports dense bf16.
    pub fn dense_bias(&self) -> Option<ferrite_cuda_core::tensor::GpuTensor> {
        match self {
            Self::Dense(l) => l.bias,
            _ => panic!("dense_bias() called on quantized LinearLayer"),
        }
    }

    /// Cheap copy for dense layers (GpuTensor metadata only, no weight copy).
    /// Panics on quantized variants — solver only supports dense bf16.
    pub fn shallow_clone(&self) -> Self {
        match self {
            Self::Dense(l) => Self::Dense(Linear::new(l.weight, l.bias)),
            _ => panic!("shallow_clone() called on quantized LinearLayer"),
        }
    }

    pub fn out_features(&self) -> usize {
        match self {
            Self::Dense(l) => l.out_features(),
            Self::Marlin(l) => l.out_features(),
            Self::Ggml(l) => l.out_features(),
            Self::GgmlConcat(branches) => branches.iter().map(|b| b.out_features()).sum(),
            Self::Bnb4bit(l) => l.out_features(),
            Self::Fp8(l) => l.out_features(),
            Self::Fp8Block(l) => l.out_features(),
        }
    }

    pub fn in_features(&self) -> usize {
        match self {
            Self::Dense(l) => l.in_features(),
            Self::Marlin(l) => l.in_features(),
            Self::Ggml(l) => l.in_features(),
            Self::GgmlConcat(branches) => branches[0].in_features(),
            Self::Bnb4bit(l) => l.in_features(),
            Self::Fp8(l) => l.in_features(),
            Self::Fp8Block(l) => l.in_features(),
        }
    }

    /// Load a single dense bf16/fp16 linear layer by safetensors prefix
    /// (e.g. `"model.lm_head"` → reads `"model.lm_head.weight"` and an
    /// optional `".bias"`).
    pub fn load_dense(weights: &mut GpuWeights, prefix: &str) -> Result<Self> {
        Ok(Self::Dense(Linear::load(weights, prefix)?))
    }

    /// Load a linear layer that may be either dense (safetensors) or
    /// GGUF-quantized. Tries `take_quantized_linear` first; falls back
    /// to `load_dense` when the prefix isn't quantized in the
    /// backing `GpuWeights`. Used by codegen for any
    /// `StorageFormat::Ggml` weight — and harmlessly equivalent to
    /// `load_dense` on safetensors models (where
    /// `take_quantized_linear` always returns `None`).
    pub fn load_dense_or_ggml(weights: &mut GpuWeights, prefix: &str) -> Result<Self> {
        let weight_name = format!("{prefix}.weight");
        if std::env::var("FERRITE_GGUF_TRACE").is_ok() {
            eprintln!(
                "[ggml] load_dense_or_ggml: prefix={prefix} weight_name={weight_name} \
                 quantized.contains={} gguf_dense.contains={} tensors.contains={}",
                weights.contains_quantized_linear(&weight_name),
                weights.gguf_dense_contains(&weight_name),
                weights.safetensor_contains(&weight_name),
            );
        }
        if let Some(storage) = weights.take_quantized_linear(&weight_name) {
            // Optional bias — GGUF rarely ships bias on linears, but
            // qwen-style checkpoints can. `take` falls back to
            // gguf_dense automatically.
            let bias_name = format!("{prefix}.bias");
            let bias = weights.take(&bias_name).ok();
            return Ok(Self::Ggml(Box::new(GgmlLinear { storage, bias })));
        }
        Self::load_dense(weights, prefix)
    }

    /// Concat-load variant of `load_dense_or_ggml`. If every prefix
    /// is GGUF-quantized in the backing `GpuWeights`, byte-stacks
    /// the per-prefix `GgmlStorage` blobs into one packed
    /// `[sum(out_features), in_features]` quantized tensor and
    /// returns `Self::Ggml`. Otherwise falls back to
    /// `load_dense_concat`.
    ///
    /// The byte-stack works because each row of a row-major
    /// quantized matrix is a whole number of GGML blocks (block
    /// boundaries never straddle rows), so `cat[gate.bytes,
    /// up.bytes]` produces a valid `[gate.nrows + up.nrows, ncols]`
    /// blob with the same dtype. Refuses if dtype or `ncols` differ
    /// across prefixes — neither is expected in practice (GGUF
    /// quantizers ship gate/up with identical layouts) but the
    /// guard prevents silent corruption if an arch ever pairs
    /// different quants.
    pub fn load_dense_concat_or_ggml(
        weights: &mut GpuWeights,
        prefixes: &[&str],
        stream: cudarc::driver::sys::CUstream,
    ) -> Result<Self> {
        if prefixes.is_empty() {
            anyhow::bail!("load_dense_concat_or_ggml: empty prefix list");
        }
        if std::env::var("FERRITE_GGUF_TRACE").is_ok() {
            eprintln!(
                "[ggml] load_dense_concat_or_ggml: prefixes={prefixes:?} \
                 (call site = either load_layered_linear_dense_concat or codegen unindexed concat)"
            );
        }

        // Probe: are ALL prefixes GGUF-quantized? (Either every
        // prefix or none — mixed would mean the GGUF loader took
        // some weights as quantized and others as dense, which it
        // doesn't do today.)
        let weight_names: Vec<String> = prefixes.iter().map(|p| format!("{p}.weight")).collect();
        let all_gguf = weight_names
            .iter()
            .all(|n| weights.contains_quantized_linear(n));

        if !all_gguf {
            // FP16/F32 GGUFs land all weights in `gguf_dense` (none
            // in `quantized`); `load_dense_concat` reads via
            // `tensor_info` + `take_into` which only see the
            // safetensors mmap map. Detect that case and assemble
            // the packed Dense linear via `take()` (which DOES check
            // gguf_dense first) + per-row D2D copy. Bias comes from
            // the same path.
            let all_gguf_dense = weight_names.iter().all(|n| weights.gguf_dense_contains(n));
            if all_gguf_dense {
                return load_gguf_dense_concat(weights, prefixes, stream);
            }
            return Self::load_dense_concat(weights, prefixes, stream);
        }

        // Take all storages.
        let mut storages: Vec<crate::ggml::GgmlStorage> = Vec::with_capacity(prefixes.len());
        for name in &weight_names {
            let s = weights
                .take_quantized_linear(name)
                .ok_or_else(|| anyhow::anyhow!("load_dense_concat_or_ggml: missing {name}"))?;
            storages.push(s);
        }

        // Validate ncols (in_features) — must be uniform across all
        // branches; mismatched dtype is OK and triggers the
        // `GgmlConcat` runtime-concat path below.
        let first = storages[0];
        for s in &storages[1..] {
            if s.ncols != first.ncols {
                anyhow::bail!(
                    "load_dense_concat_or_ggml: ncols mismatch ({} vs {})",
                    first.ncols,
                    s.ncols
                );
            }
        }
        // Bias collection: GGUF biases are dequantized into
        // `gguf_dense` at load time. If every prefix has a sibling
        // `.bias`, collect them; if any has and others don't, hard-
        // error (Qwen2's biased q/k/v is all-or-none). Bias-less
        // GGUFs (Llama gate/up, Llama q/k/v) keep `bias = None` per
        // branch and the fast-path stays the same.
        let bias_names: Vec<String> = prefixes.iter().map(|p| format!("{p}.bias")).collect();
        let any_bias = bias_names.iter().any(|n| weights.contains(n));
        let all_bias = bias_names.iter().all(|n| weights.contains(n));
        if any_bias && !all_bias {
            anyhow::bail!(
                "load_dense_concat_or_ggml: inconsistent biases across prefixes {:?} \
                 — biased Qwen-style q/k/v requires every prefix to ship a `.bias`",
                prefixes
            );
        }
        let mut biases: Vec<Option<GpuTensor>> = vec![None; prefixes.len()];
        if all_bias {
            for (i, n) in bias_names.iter().enumerate() {
                biases[i] = Some(weights.take(n)?);
            }
        }

        // Always use the per-branch `GgmlConcat` path — never allocate
        // a byte-packed copy of the source weights.
        //
        // The per-prefix `GgmlStorage` values share the underlying
        // GPU buffers in `GpuWeights.quantized` (`take_quantized_linear`
        // is non-destructive). Routing the fused load through a
        // byte-packed copy would duplicate those bytes — fatal at
        // commandR-35B scale and fundamentally wrong as a "fix" for
        // any sharing problem. `GgmlConcat::forward` produces the
        // same packed `[num_tokens, sum(out)]` activation the
        // downstream `fused_qkv_rope_cache` / `silu_and_mul` kernels
        // consume; only the activation is materialized, the weights
        // stay in their original quantized buffers.
        //
        // Each branch keeps its own bias (applied inside
        // `GgmlLinear::forward` before `GgmlConcat::forward` packs the
        // outputs), matching the heterogeneous-dtype path's contract.
        let branches: Vec<GgmlLinear> = storages
            .into_iter()
            .zip(biases)
            .map(|(s, bias)| GgmlLinear { storage: s, bias })
            .collect();
        Ok(Self::GgmlConcat(branches))
    }

    /// Tensor-parallel sharded dense linear load. See
    /// [`Linear::load_sharded`] for the full bias/dim semantics —
    /// this is the `LinearLayer` wrapper. Used by codegen at tp>1
    /// when [`crate::ferrite_forward_macro::tp_lowering::shard_kind_for_weight_path`]
    /// reports the prefix's last segment is column-parallel
    /// (`dim = 0`) or row-parallel (`dim = 1`). Pass `(rank, world)`
    /// from the runtime; `world == 1` short-circuits to the same
    /// behavior as `load_dense` plus shard-kind-aware bias rules.
    pub fn load_dense_sharded(
        weights: &mut GpuWeights,
        prefix: &str,
        dim: usize,
        rank: usize,
        world: usize,
    ) -> Result<Self> {
        // GGUF fast-path. The GGUF loader pre-shards quantized linears
        // at file-read time per `gguf_shard_kind_for_hf_name`:
        //   ShardDim0 → rows split, bias is 1D so always `Replicate`.
        //   ShardDim1 → per-row column slice, bias is 1D `Replicate`.
        // Both cases land the quantized storage in `weights.quantized`
        // with per-rank shape and the optional bias in `gguf_dense`
        // with full shape. The safetensors `Linear::load_sharded` path
        // would error with "weight not found" on either — those maps
        // are only consulted by `take_quantized_linear` / `take`.
        let weight_name = format!("{prefix}.weight");
        if let Some(storage) = weights.take_quantized_linear(&weight_name) {
            let bias_name = format!("{prefix}.bias");
            // Bias rules per-dim at tp > 1:
            //   dim=0 (column-parallel): bias shards along dim 0 too.
            //     The GGUF loader replicates 1D tensors (`gguf_shard_kind_for_hf_name`
            //     returns `Replicate` for ndim<2), so each rank currently holds
            //     the full `[out_full]` bias. Narrow it to the per-rank slice
            //     here — matches Python vLLM's `ColumnParallelLinear` weight
            //     loader + `Qwen2`'s biased q/k/v/o at tp>1.
            //   dim=1 (row-parallel): bias is added once per output element,
            //     after the NCCL all-reduce. Python vLLM's `RowParallelLinear`
            //     adds it only on rank 0; adding on every rank would sum
            //     `world * bias`. Keep on rank 0, drop on others.
            let bias = if weights.contains(&bias_name) {
                match dim {
                    1 if rank != 0 => None,
                    0 if world > 1 => weights
                        .take(&bias_name)
                        .ok()
                        .map(|b| per_rank_bias_slice(b, rank, world, storage.nrows)),
                    _ => weights.take(&bias_name).ok(),
                }
            } else {
                None
            };
            return Ok(Self::Ggml(Box::new(GgmlLinear { storage, bias })));
        }
        if let Some(w) = weights.take_gguf_dense(&weight_name) {
            // F16/F32 GGUF: linear weight lives in `gguf_dense`, already
            // per-rank sharded. Same bias rule as the quantized branch.
            let bias_name = format!("{prefix}.bias");
            let bias = if weights.contains(&bias_name) {
                match dim {
                    1 if rank != 0 => None,
                    0 if world > 1 => weights
                        .take(&bias_name)
                        .ok()
                        .map(|b| per_rank_bias_slice(b, rank, world, w.dim(0))),
                    _ => weights.take(&bias_name).ok(),
                }
            } else {
                None
            };
            return Ok(Self::Dense(Linear::new(w, bias)));
        }
        Ok(Self::Dense(Linear::load_sharded(
            weights, prefix, dim, rank, world,
        )?))
    }

    /// Load several dense linear layers and concatenate along the
    /// out-feature dim (dim 0 of the weight matrix), returning one
    /// packed `LinearLayer::Dense`.
    ///
    /// Streams each source weight directly from CPU-safetensors into
    /// the packed GPU buffer (no D2D copy, no intermediate allocation).
    /// Used by ferrite-forward's fused accessors — e.g. `FusedQkvRopeCacheImpl`
    /// expects one packed `[q_size + 2*kv_size, hidden]` weight covering
    /// the three source q/k/v projections.
    ///
    /// If any source weight has a bias, all of them must — the biases
    /// are concatenated in the same order as the weights. Otherwise
    /// the returned layer has no bias.
    pub fn load_dense_concat(
        weights: &mut GpuWeights,
        prefixes: &[&str],
        stream: cudarc::driver::sys::CUstream,
    ) -> Result<Self> {
        if prefixes.is_empty() {
            anyhow::bail!("load_dense_concat: empty prefix list");
        }

        // Packed-source fallback: if any source .weight is missing, it
        // may live under a packed parent (qkv_proj / gate_up_proj).
        // Mirrors the fallback in `Linear::load`.
        for p in prefixes {
            if !weights.contains(&format!("{p}.weight")) {
                try_synthesize_packed_slice(weights, p)?;
            }
        }

        // First pass: resolve shapes/dtype from CPU-side metadata.
        let mut shapes_dtypes: Vec<(Vec<usize>, ferrite_cuda_core::dtype::DType)> =
            Vec::with_capacity(prefixes.len());
        for p in prefixes {
            let weight_name = format!("{p}.weight");
            let (shape, dtype) = weights
                .tensor_info(&weight_name)
                .ok_or_else(|| anyhow::anyhow!("weight not found: {weight_name}"))?;
            if shape.len() != 2 {
                anyhow::bail!(
                    "load_dense_concat: `{weight_name}` has rank {}, expected 2",
                    shape.len(),
                );
            }
            shapes_dtypes.push((shape.to_vec(), dtype));
        }

        // All sources must share the same in-features (dim 1) and dtype.
        let hidden = shapes_dtypes[0].0[1];
        let dtype = shapes_dtypes[0].1;
        for (i, (shape, dt)) in shapes_dtypes.iter().enumerate() {
            if shape[1] != hidden {
                anyhow::bail!(
                    "load_dense_concat: `{}` has in_features {}, expected {}",
                    prefixes[i],
                    shape[1],
                    hidden,
                );
            }
            if *dt != dtype {
                anyhow::bail!(
                    "load_dense_concat: `{}` has dtype {:?}, expected {:?}",
                    prefixes[i],
                    dt,
                    dtype,
                );
            }
        }

        let total_out: usize = shapes_dtypes.iter().map(|(s, _)| s[0]).sum();
        let elem = dtype.size_bytes();
        let total_bytes = total_out * hidden * elem;

        // One contiguous GPU buffer; stream each source into its offset.
        let ptr = unsafe { ferrite_cuda_core::driver::mem_alloc(total_bytes)? };
        weights.record_alloc(ptr, total_bytes);
        let mut offset_bytes: usize = 0;
        for (i, p) in prefixes.iter().enumerate() {
            let weight_name = format!("{p}.weight");
            let bytes = shapes_dtypes[i].0[0] * hidden * elem;
            unsafe {
                weights.take_into(&weight_name, ptr.add(offset_bytes), stream)?;
            }
            offset_bytes += bytes;
        }
        let packed_weight = unsafe { GpuTensor::new(ptr, &[total_out, hidden], dtype) };

        // Biases: either all-or-none across the source set.
        let bias_names: Vec<String> = prefixes.iter().map(|p| format!("{p}.bias")).collect();
        let packed_bias = if weights.contains(&bias_names[0]) {
            for bn in &bias_names {
                if !weights.contains(bn) {
                    anyhow::bail!(
                        "load_dense_concat: inconsistent bias — `{}` exists but `{bn}` is missing",
                        bias_names[0],
                    );
                }
            }
            let mut per_bias_bytes: Vec<usize> = Vec::with_capacity(bias_names.len());
            let mut total_bias_bytes = 0usize;
            for bn in &bias_names {
                let (bshape, bdt) = weights
                    .tensor_info(bn)
                    .ok_or_else(|| anyhow::anyhow!("bias metadata missing: {bn}"))?;
                if bdt != dtype {
                    anyhow::bail!(
                        "load_dense_concat: bias `{bn}` dtype {:?} != weight dtype {:?}",
                        bdt,
                        dtype,
                    );
                }
                let b = bshape.iter().product::<usize>() * elem;
                per_bias_bytes.push(b);
                total_bias_bytes += b;
            }
            let bptr = unsafe { ferrite_cuda_core::driver::mem_alloc(total_bias_bytes)? };
            weights.record_alloc(bptr, total_bias_bytes);
            let mut boff = 0usize;
            for (i, bn) in bias_names.iter().enumerate() {
                unsafe {
                    weights.take_into(bn, bptr.add(boff), stream)?;
                }
                boff += per_bias_bytes[i];
            }
            Some(unsafe { GpuTensor::new(bptr, &[total_bias_bytes / elem], dtype) })
        } else {
            None
        };

        Ok(Self::Dense(Linear::new(packed_weight, packed_bias)))
    }

    /// Tensor-parallel sharded variant of [`Self::load_dense_concat`].
    /// Used by codegen at tp>1 for the fused QKV / gate_up
    /// projections, which are always column-parallel (`ShardDim0`)
    /// — no row-parallel concat exists in any current arch.
    ///
    /// Each source weight is sliced along its output dim (`dim 0` of
    /// the `[out, in]` matrix) to `[out / world, in]`, then concatenated
    /// along that same dim 0 into one packed `[(sum out_i) / world, in]`
    /// GPU buffer. Biases follow the column-parallel rule (sliced
    /// along dim 0 too) — matches Python `MergedColumnParallelLinear`
    /// / `QKVParallelLinear` weight loaders.
    ///
    /// `world == 1` short-circuits to byte-equivalent behavior with
    /// `Self::load_dense_concat`. Each source's `out` dim must be
    /// divisible by `world` — the macro's outer-loop fanout already
    /// `skip`s indivisible (variant, tp) tuples (per the activation
    /// commit `889c44b2f`), so this is a runtime invariant the
    /// compile-time set guarantees, not a per-call assertion.
    pub fn load_dense_concat_sharded(
        weights: &mut GpuWeights,
        prefixes: &[&str],
        stream: cudarc::driver::sys::CUstream,
        rank: usize,
        world: usize,
    ) -> Result<Self> {
        if prefixes.is_empty() {
            anyhow::bail!("load_dense_concat_sharded: empty prefix list");
        }

        // GGUF fast-path. The GGUF loader pre-shards every source at
        // file-read time (fused QKV and gate_up sources are all
        // ShardDim0 → per-rank rows). Each per-prefix tensor is already
        // `[out_i / world, in]`; we just collect them without touching
        // the bytes. Mirrors the tp=1 `load_dense_concat_or_ggml` path.
        let weight_names: Vec<String> = prefixes.iter().map(|p| format!("{p}.weight")).collect();
        let all_gguf_quantized = weight_names
            .iter()
            .all(|n| weights.contains_quantized_linear(n));
        if all_gguf_quantized {
            // Prefer the zero-copy `GgmlConcat` path — the storages
            // share the underlying GPU buffers in `GpuWeights.quantized`
            // (take is non-destructive). No byte-pack, no extra H2D,
            // no dequant. Concat is always column-parallel (ShardDim0),
            // so each branch's bias (if present) must be narrowed to
            // the per-rank slice matching `storage.nrows` — Qwen2's
            // biased q/k/v at tp>1 is the canonical example. GGUF 1D
            // biases are `Replicate` on-disk (full length on every rank),
            // so we compute the rank slice here against the replicated
            // allocation.
            let mut storages: Vec<crate::ggml::GgmlStorage> = Vec::with_capacity(prefixes.len());
            for name in &weight_names {
                let s = weights
                    .take_quantized_linear(name)
                    .ok_or_else(|| anyhow::anyhow!("load_dense_concat_sharded: missing {name}"))?;
                storages.push(s);
            }
            let first_ncols = storages[0].ncols;
            for s in &storages[1..] {
                if s.ncols != first_ncols {
                    anyhow::bail!(
                        "load_dense_concat_sharded: ncols mismatch ({} vs {})",
                        first_ncols,
                        s.ncols
                    );
                }
            }
            let bias_names: Vec<String> = prefixes.iter().map(|p| format!("{p}.bias")).collect();
            let any_bias = bias_names.iter().any(|n| weights.contains(n));
            let all_bias = bias_names.iter().all(|n| weights.contains(n));
            if any_bias && !all_bias {
                anyhow::bail!(
                    "load_dense_concat_sharded: inconsistent biases across prefixes {:?}",
                    prefixes
                );
            }
            let mut biases: Vec<Option<GpuTensor>> = vec![None; prefixes.len()];
            if all_bias {
                for (i, n) in bias_names.iter().enumerate() {
                    let full = weights.take(n)?;
                    biases[i] = Some(if world > 1 {
                        per_rank_bias_slice(full, rank, world, storages[i].nrows)
                    } else {
                        full
                    });
                }
            }
            let _ = stream;
            let branches: Vec<GgmlLinear> = storages
                .into_iter()
                .zip(biases)
                .map(|(s, bias)| GgmlLinear { storage: s, bias })
                .collect();
            return Ok(Self::GgmlConcat(branches));
        }
        let all_gguf_dense = weight_names.iter().all(|n| weights.gguf_dense_contains(n));
        if all_gguf_dense {
            // F16/F32 GGUF: tensors are in `gguf_dense` already per-rank
            // sharded (ShardDim0). `load_gguf_dense_concat` packs them
            // row-wise into one `[sum(rows_per_rank), cols]` buffer —
            // the contract it already has, just with per-rank inputs
            // instead of unsharded ones.
            let _ = (rank, world);
            return load_gguf_dense_concat(weights, prefixes, stream);
        }

        // Same packed-source fallback as the unsharded path.
        for p in prefixes {
            if !weights.contains(&format!("{p}.weight")) {
                try_synthesize_packed_slice(weights, p)?;
            }
        }

        // Resolve unsharded shapes/dtype from CPU-side metadata,
        // then divide each source's out dim by `world` to get the
        // per-rank packed shape.
        let mut shapes_dtypes: Vec<(Vec<usize>, ferrite_cuda_core::dtype::DType)> =
            Vec::with_capacity(prefixes.len());
        for p in prefixes {
            let weight_name = format!("{p}.weight");
            let (shape, dtype) = weights
                .tensor_info(&weight_name)
                .ok_or_else(|| anyhow::anyhow!("weight not found: {weight_name}"))?;
            if shape.len() != 2 {
                anyhow::bail!(
                    "load_dense_concat_sharded: `{weight_name}` has rank {}, expected 2",
                    shape.len(),
                );
            }
            shapes_dtypes.push((shape.to_vec(), dtype));
        }

        let hidden = shapes_dtypes[0].0[1];
        let dtype = shapes_dtypes[0].1;
        for (i, (shape, dt)) in shapes_dtypes.iter().enumerate() {
            if shape[1] != hidden {
                anyhow::bail!(
                    "load_dense_concat_sharded: `{}` has in_features {}, expected {}",
                    prefixes[i],
                    shape[1],
                    hidden,
                );
            }
            if *dt != dtype {
                anyhow::bail!(
                    "load_dense_concat_sharded: `{}` has dtype {:?}, expected {:?}",
                    prefixes[i],
                    dt,
                    dtype,
                );
            }
        }

        let elem = dtype.size_bytes();
        // Per-rank packed shape: each source's out dim / world.
        let per_rank_outs: Vec<usize> = shapes_dtypes.iter().map(|(s, _)| s[0] / world).collect();
        let total_out: usize = per_rank_outs.iter().sum();
        let total_bytes = total_out * hidden * elem;

        let ptr = unsafe { ferrite_cuda_core::driver::mem_alloc(total_bytes)? };
        weights.record_alloc(ptr, total_bytes);
        let mut offset_bytes: usize = 0;
        for (i, p) in prefixes.iter().enumerate() {
            let weight_name = format!("{p}.weight");
            let bytes = per_rank_outs[i] * hidden * elem;
            unsafe {
                weights.take_shard_into(
                    &weight_name,
                    0,
                    rank,
                    world,
                    ptr.add(offset_bytes),
                    stream,
                )?;
            }
            offset_bytes += bytes;
        }
        let packed_weight = unsafe { GpuTensor::new(ptr, &[total_out, hidden], dtype) };

        // Biases: column-parallel concat shards each bias along dim 0
        // — same rule as the weight. Either all sources have a bias
        // or none of them do (matches the unsharded contract).
        let bias_names: Vec<String> = prefixes.iter().map(|p| format!("{p}.bias")).collect();
        let packed_bias = if weights.contains(&bias_names[0]) {
            for bn in &bias_names {
                if !weights.contains(bn) {
                    anyhow::bail!(
                        "load_dense_concat_sharded: inconsistent bias — `{}` exists but `{bn}` is missing",
                        bias_names[0],
                    );
                }
            }
            // Per-rank bias bytes mirror per-rank weight outs (bias
            // shape is `[out_i]` unsharded → `[out_i / world]` per rank).
            let mut per_bias_bytes: Vec<usize> = Vec::with_capacity(bias_names.len());
            let mut total_bias_bytes = 0usize;
            for (i, bn) in bias_names.iter().enumerate() {
                let (bshape, bdt) = weights
                    .tensor_info(bn)
                    .ok_or_else(|| anyhow::anyhow!("bias metadata missing: {bn}"))?;
                if bdt != dtype {
                    anyhow::bail!(
                        "load_dense_concat_sharded: bias `{bn}` dtype {:?} != weight dtype {:?}",
                        bdt,
                        dtype,
                    );
                }
                // Ignore unsharded `bshape` here — per-rank bytes are
                // derived from the matched weight's per-rank out dim
                // (already validated as `out_i / world` above).
                let _ = bshape;
                let b = per_rank_outs[i] * elem;
                per_bias_bytes.push(b);
                total_bias_bytes += b;
            }
            let bptr = unsafe { ferrite_cuda_core::driver::mem_alloc(total_bias_bytes)? };
            weights.record_alloc(bptr, total_bias_bytes);
            let mut boff = 0usize;
            for (i, bn) in bias_names.iter().enumerate() {
                unsafe {
                    weights.take_shard_into(bn, 0, rank, world, bptr.add(boff), stream)?;
                }
                boff += per_bias_bytes[i];
            }
            Some(unsafe { GpuTensor::new(bptr, &[total_bias_bytes / elem], dtype) })
        } else {
            None
        };

        Ok(Self::Dense(Linear::new(packed_weight, packed_bias)))
    }
}

impl From<Linear> for LinearLayer {
    fn from(l: Linear) -> Self {
        Self::Dense(l)
    }
}

impl From<MarlinLinear> for LinearLayer {
    fn from(l: MarlinLinear) -> Self {
        Self::Marlin(Box::new(l))
    }
}

impl From<GgmlLinear> for LinearLayer {
    fn from(l: GgmlLinear) -> Self {
        Self::Ggml(Box::new(l))
    }
}

impl From<Fp8Linear> for LinearLayer {
    fn from(l: Fp8Linear) -> Self {
        Self::Fp8(Box::new(l))
    }
}

impl From<Fp8BlockLinear> for LinearLayer {
    fn from(l: Fp8BlockLinear) -> Self {
        Self::Fp8Block(Box::new(l))
    }
}

// ---------------------------------------------------------------------------
// Fp8BlockLinear (FP8 E4M3 with per-block scales, e.g. DeepSeek-V3)
// ---------------------------------------------------------------------------

/// FP8 block-quantized linear layer with per-block weight scales.
///
/// Weights are stored as FP8 `[out_features, in_features]` with per-block
/// scales `[ceil(N/block_n), ceil(K/block_k)]`. Forward dequantizes to BF16
/// then uses standard cuBLAS GEMM.
///
/// PERF GAP: Python uses CUTLASS block-scaled FP8 GEMM (one fused kernel)
/// or deep_gemm (Hopper). We dequant + cuBLAS which adds an extra memory
/// round-trip. Functionally correct, but slower for block-quantized models
/// like DeepSeek-V3. To close: port CUTLASS block-scaled kernel from
/// `vllm/csrc/quantization/cutlass_w8a8/`.
///
/// Matches Python vLLM's `Fp8LinearMethod` with `weight_block_size`.
pub struct Fp8BlockLinear {
    /// FP8 E4M3 weights `[out_features, in_features]`.
    pub weight: GpuTensor,
    /// Per-block weight scale inverse: `[ceil(N/block_n), ceil(K/block_k)]` f32.
    pub weight_scale_inv: GpuTensor,
    /// Block quantization block size `[block_n, block_k]`.
    pub block_size: [usize; 2],
    /// Optional bias `[out_features]`.
    pub bias: Option<GpuTensor>,
    /// Output dtype (BF16 or F16).
    pub output_dtype: ferrite_cuda_core::dtype::DType,
}

impl Fp8BlockLinear {
    /// Forward: dequant FP8 → BF16 per block, then cuBLAS GEMM.
    ///
    /// Current implementation: CPU-side dequant-then-GEMM for correctness.
    /// TODO: CUTLASS block-scaled FP8 GEMM for perf parity.
    pub unsafe fn forward(
        &self,
        x: TensorView<'_>,
        cublas: &mut CublasHandle,
        alloc: &mut CachingAllocator,
        stream: cudarc::driver::sys::CUstream,
    ) -> OwnedTensor {
        debug_assert_eq!(x.ndim(), 2);
        let k = self.weight.dim(1);
        debug_assert_eq!(x.dim(1), k, "Fp8BlockLinear: input dim mismatch");

        // Dequantize FP8 weight to BF16/F16 using per-block scales.
        let dequant_weight = crate::kernels::fp8_block_dequant(
            self.weight,
            self.weight_scale_inv,
            self.block_size,
            self.output_dtype,
            alloc,
            stream,
        );

        // Standard GEMM: x @ dequant_weight^T
        // Use as_gpu_tensor() to borrow — dequant_weight drops after GEMM,
        // returning the buffer to the caching allocator.
        let out = cublas.gemm(*x, dequant_weight.as_gpu_tensor(), alloc);
        drop(dequant_weight);

        if let Some(bias) = self.bias {
            crate::kernels::bias_add_inplace(out.as_gpu_tensor(), bias, stream);
        }

        out
    }

    pub fn out_features(&self) -> usize {
        self.weight.dim(0)
    }

    pub fn in_features(&self) -> usize {
        self.weight.dim(1)
    }
}

// ---------------------------------------------------------------------------
// Fp8AnyLinear (per-tensor / per-channel `Fp8Linear` ⨁ blockwise `Fp8BlockLinear`)
// ---------------------------------------------------------------------------

/// Storage-uniform wrapper around the two FP8 linear variants. Lets
/// the codegen pick a single `weight_fn` Rust type for FP8 GEMM Impls
/// — every claim of an FP8 Impl in a model's Weights struct holds an
/// `Fp8AnyLinear`, regardless of whether that specific tile's
/// `quantization_config` declared per-tensor / per-channel scales
/// (`Std`) or per-block scales (`Block`).
///
/// `forward` matches both inner `forward`s' signature, so the
/// per-arch interpreter arm calls `(weight_fn)(wm, layer).forward(x,
/// cublas, alloc, stream)` without caring which variant is inside.
pub enum Fp8AnyLinear {
    Std(Fp8Linear),
    Block(Fp8BlockLinear),
}

impl Fp8AnyLinear {
    /// Forward: dispatches to whichever inner FP8 layer is wrapped.
    /// Both variants take the same arguments and return `OwnedTensor`.
    ///
    /// # Safety
    /// All tensors must be valid GPU memory; `cublas` / `alloc` /
    /// `stream` must be live. Inner `forward`s carry the same
    /// invariants.
    pub unsafe fn forward(
        &self,
        x: TensorView<'_>,
        cublas: &mut CublasHandle,
        alloc: &mut CachingAllocator,
        stream: cudarc::driver::sys::CUstream,
    ) -> OwnedTensor {
        match self {
            Self::Std(l) => unsafe { l.forward(x, cublas, alloc, stream) },
            Self::Block(l) => unsafe { l.forward(x, cublas, alloc, stream) },
        }
    }

    pub fn out_features(&self) -> usize {
        match self {
            Self::Std(l) => l.out_features(),
            Self::Block(l) => l.out_features(),
        }
    }

    pub fn in_features(&self) -> usize {
        match self {
            Self::Std(l) => l.in_features(),
            Self::Block(l) => l.in_features(),
        }
    }
}

// ---------------------------------------------------------------------------
// Embedding
// ---------------------------------------------------------------------------

/// Token embedding lookup table.
///
/// Weight shape: `[vocab_size, hidden_size]`.
/// Forward gathers rows by token IDs.
pub struct Embedding {
    pub weight: GpuTensor, // [vocab_size, hidden_size]
}

impl Embedding {
    pub fn new(weight: GpuTensor) -> Self {
        debug_assert_eq!(weight.ndim(), 2);
        Self { weight }
    }

    /// Load from `GpuWeights` by prefix.
    pub fn load(weights: &mut GpuWeights, prefix: &str) -> Result<Self> {
        let weight_name = format!("{prefix}.weight");
        let weight = weights.take(&weight_name)?;
        Ok(Self::new(weight))
    }

    /// Vocab-parallel sharded load — slices the embedding table
    /// along dim 0 (`vocab_size`) so each rank holds
    /// `[vocab_size / world, hidden_size]`. Mirrors Python vLLM's
    /// `VocabParallelEmbedding` (the same scheme `ParallelLMHead`
    /// inherits, which is what makes `tie_weights` self-consistent
    /// at tp>1 — both sharded slices come from the same dim-0 cut).
    /// `world == 1` degrades to the same shape `Self::load` produces.
    pub fn load_sharded(
        weights: &mut GpuWeights,
        prefix: &str,
        rank: usize,
        world: usize,
    ) -> Result<Self> {
        let weight_name = format!("{prefix}.weight");
        // GGUF fast-path: `GgufGpuWeights::load` pre-shards `embed_tokens`
        // as ShardDim0 at file-read time (see `gguf_shard_kind_for_hf_name`
        // in `ferrite-kernels/src/ggml.rs`), so the per-rank tensor is
        // already in `gguf_dense` with shape `[vocab/world, hidden]`.
        // Take it as-is; the safetensors `take_shard` path below would
        // miss because the tensor was never inserted into the CPU
        // `tensors` map.
        if let Some(t) = weights.take_gguf_dense(&weight_name) {
            return Ok(Self::new(t));
        }
        let weight = weights.take_shard(&weight_name, 0, rank, world)?;
        Ok(Self::new(weight))
    }

    pub fn vocab_size(&self) -> usize {
        self.weight.dim(0)
    }

    pub fn hidden_size(&self) -> usize {
        self.weight.dim(1)
    }

    /// Forward: gather embedding rows by token IDs.
    ///
    /// `input_ids`: `[num_tokens]` (U32 on GPU)
    /// Returns: `[num_tokens, hidden_size]` allocated from arena.
    ///
    /// # Safety
    /// All tensors must be valid GPU memory. Stream must be valid.
    /// This currently uses a simple gather kernel (TODO: implement via CUDA kernel).
    pub unsafe fn forward(
        &self,
        _input_ids: TensorView<'_>,
        _alloc: &mut CachingAllocator,
        _stream: cudarc::driver::sys::CUstream,
    ) -> GpuTensor {
        // TODO: implement embedding gather kernel.
        // For now, return a placeholder. The kernel is trivial:
        // one thread per (token, dim) reads weight[input_ids[token], dim].
        todo!("embedding gather kernel not yet implemented")
    }
}

// ---------------------------------------------------------------------------
// RmsNorm
// ---------------------------------------------------------------------------

/// Root Mean Square Layer Normalization.
///
/// `y = x / sqrt(mean(x^2) + eps) * weight`
///
/// Weight shape: `[hidden_size]`.
pub struct RmsNorm {
    pub weight: GpuTensor, // [hidden_size]
    pub eps: f32,
}

impl RmsNorm {
    pub fn new(weight: GpuTensor, eps: f32) -> Self {
        debug_assert_eq!(weight.ndim(), 1);
        Self { weight, eps }
    }

    /// Load from `GpuWeights` by prefix.
    pub fn load(weights: &mut GpuWeights, prefix: &str, eps: f32) -> Result<Self> {
        let weight_name = format!("{prefix}.weight");
        let weight = weights.take(&weight_name)?;
        Ok(Self::new(weight, eps))
    }

    pub fn hidden_size(&self) -> usize {
        self.weight.dim(0)
    }

    // forward() will be implemented when we wire the existing CUDA kernels
    // from vllm-kernels to accept GpuTensor raw pointers.
}

/// Standard LayerNorm with weight and optional bias.
///
/// Computes: `(x - mean) / sqrt(var + eps) * weight + bias`
pub struct LayerNorm {
    pub weight: GpuTensor,       // [hidden_size]
    pub bias: Option<GpuTensor>, // [hidden_size] or None
    pub eps: f32,
}

impl LayerNorm {
    pub fn new(weight: GpuTensor, bias: Option<GpuTensor>, eps: f32) -> Self {
        debug_assert_eq!(weight.ndim(), 1);
        if let Some(ref b) = bias {
            debug_assert_eq!(b.ndim(), 1);
            debug_assert_eq!(b.dim(0), weight.dim(0));
        }
        Self { weight, bias, eps }
    }

    /// Load from `GpuWeights` by prefix. Loads bias if present.
    pub fn load(weights: &mut GpuWeights, prefix: &str, eps: f32) -> Result<Self> {
        let weight = weights.take(&format!("{prefix}.weight"))?;
        let bias_name = format!("{prefix}.bias");
        let bias = if weights.contains(&bias_name) {
            Some(weights.take(&bias_name)?)
        } else {
            None
        };
        Ok(Self::new(weight, bias, eps))
    }

    pub fn hidden_size(&self) -> usize {
        self.weight.dim(0)
    }
}

// ---------------------------------------------------------------------------
// Tensor-parallel layer wrappers
// ---------------------------------------------------------------------------

/// Column-parallel linear: shards output dim (dim=0 of weight).
///
/// After forward, optionally all-gathers output across ranks to reconstruct
/// the full output (used for lm_head). For most uses (QKV, gate_up), no
/// all-gather is needed because the downstream layer consumes the shard.
pub struct ColumnParallelLinear {
    pub inner: LinearLayer,
    pub gather_output: bool,
    #[cfg(feature = "nccl")]
    pub tp_group: Option<Arc<NcclGroup>>,
}

impl ColumnParallelLinear {
    pub fn new(inner: LinearLayer, gather_output: bool) -> Self {
        Self {
            inner,
            gather_output,
            #[cfg(feature = "nccl")]
            tp_group: None,
        }
    }

    /// Forward: y = x @ W_shard^T, optionally all-gather.
    pub unsafe fn forward(
        &self,
        x: TensorView<'_>,
        cublas: &mut CublasHandle,
        alloc: &mut CachingAllocator,
        stream: cudarc::driver::sys::CUstream,
    ) -> OwnedTensor {
        let out = self.inner.forward(x, cublas, alloc, stream);

        #[cfg(feature = "nccl")]
        if self.gather_output
            && let Some(ref group) = self.tp_group
        {
            let gathered = group.all_gather(out.as_gpu_tensor(), alloc);
            drop(out);
            return gathered;
        }

        out
    }

    pub fn out_features(&self) -> usize {
        self.inner.out_features()
    }

    pub fn in_features(&self) -> usize {
        self.inner.in_features()
    }

    #[cfg(feature = "nccl")]
    pub fn set_tp_group(&mut self, group: Arc<NcclGroup>) {
        self.tp_group = Some(group);
    }
}

/// Row-parallel linear: shards input dim (dim=1 of weight).
///
/// After forward, all-reduces output across ranks (each rank computed a
/// partial sum). Bias is added AFTER the all-reduce.
pub struct RowParallelLinear {
    pub inner: LinearLayer,
    /// Bias added after all-reduce (not inside the GEMM).
    pub bias: Option<GpuTensor>,
    #[cfg(feature = "nccl")]
    pub tp_group: Option<Arc<NcclGroup>>,
}

impl RowParallelLinear {
    pub fn new(inner: LinearLayer, bias: Option<GpuTensor>) -> Self {
        Self {
            inner,
            bias,
            #[cfg(feature = "nccl")]
            tp_group: None,
        }
    }

    /// Forward: y = all_reduce(x @ W_shard^T) + bias.
    pub unsafe fn forward(
        &self,
        x: TensorView<'_>,
        cublas: &mut CublasHandle,
        alloc: &mut CachingAllocator,
        stream: cudarc::driver::sys::CUstream,
    ) -> OwnedTensor {
        let out = self.inner.forward(x, cublas, alloc, stream);

        #[cfg(feature = "nccl")]
        if let Some(ref group) = self.tp_group {
            group
                .all_reduce_inplace(out.as_gpu_tensor())
                .expect("all_reduce failed");
        }

        if let Some(bias) = self.bias {
            crate::kernels::bias_add_inplace(out.as_gpu_tensor(), bias, stream);
        }

        out
    }

    pub fn out_features(&self) -> usize {
        self.inner.out_features()
    }

    #[cfg(feature = "nccl")]
    pub fn set_tp_group(&mut self, group: Arc<NcclGroup>) {
        self.tp_group = Some(group);
    }
}

/// Vocab-parallel embedding: shards vocab rows across ranks.
///
/// Each rank holds rows `[rank * shard_size .. (rank+1) * shard_size]`.
/// Tokens outside the local range produce zeros. All-reduce sums partial
/// results to reconstruct the full embedding.
pub struct VocabParallelEmbedding {
    pub inner: Embedding,
    /// Global vocab start offset for this rank's shard.
    pub vocab_start: usize,
    /// Global vocab end offset (exclusive) for this rank's shard.
    pub vocab_end: usize,
    #[cfg(feature = "nccl")]
    pub tp_group: Option<Arc<NcclGroup>>,
}

impl VocabParallelEmbedding {
    pub fn new(inner: Embedding, vocab_start: usize, vocab_end: usize) -> Self {
        Self {
            inner,
            vocab_start,
            vocab_end,
            #[cfg(feature = "nccl")]
            tp_group: None,
        }
    }

    pub fn hidden_size(&self) -> usize {
        self.inner.hidden_size()
    }

    #[cfg(feature = "nccl")]
    pub fn set_tp_group(&mut self, group: Arc<NcclGroup>) {
        self.tp_group = Some(group);
    }
}

// ---------------------------------------------------------------------------
// CohereLayerNorm
// ---------------------------------------------------------------------------

/// Cohere LayerNorm: full LayerNorm with mean subtraction, weight only (no bias).
///
/// `y = weight * (x - mean(x)) / sqrt(var(x) + eps)`
///
/// Used by Command R (CohereForCausalLM).
/// Weight shape: `[hidden_size]`.
pub struct CohereLayerNorm {
    pub weight: GpuTensor, // [hidden_size]
    pub eps: f32,
}

impl CohereLayerNorm {
    pub fn new(weight: GpuTensor, eps: f32) -> Self {
        debug_assert_eq!(weight.ndim(), 1);
        Self { weight, eps }
    }

    /// Load from `GpuWeights` by prefix.
    pub fn load(weights: &mut GpuWeights, prefix: &str, eps: f32) -> Result<Self> {
        let weight_name = format!("{prefix}.weight");
        let weight = weights.take(&weight_name)?;
        Ok(Self::new(weight, eps))
    }

    pub fn hidden_size(&self) -> usize {
        self.weight.dim(0)
    }
}

// ---------------------------------------------------------------------------
// GGML quantized-storage dual-path probe (FERRITE_PROBE_MMVQ=1)
//
// Runs the broken MMVQ/DMMV path AND a cuBLAS-on-dequant reference on every
// GgmlLinear call, prints max_abs_diff to stderr. Use to locate the first
// layer where the quantized path diverges from dense. Off by default.
// ---------------------------------------------------------------------------

mod ggml_probe {
    use super::*;
    use ferrite_cuda_core::dtype::DType;
    use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

    static ENABLED: AtomicBool = AtomicBool::new(false);
    static USE_REF: AtomicBool = AtomicBool::new(false);
    static INIT: std::sync::Once = std::sync::Once::new();
    static CALL_COUNT: AtomicUsize = AtomicUsize::new(0);

    fn init() {
        INIT.call_once(|| {
            let on = std::env::var("FERRITE_PROBE_MMVQ")
                .map(|v| !v.is_empty() && v != "0")
                .unwrap_or(false);
            ENABLED.store(on, Ordering::Relaxed);
            let use_ref = std::env::var("FERRITE_USE_REFERENCE")
                .map(|v| !v.is_empty() && v != "0")
                .unwrap_or(false);
            USE_REF.store(use_ref, Ordering::Relaxed);
            if on {
                eprintln!("[probe] FERRITE_PROBE_MMVQ=1 — dual-path GgmlLinear probe active");
            }
            if use_ref {
                eprintln!(
                    "[probe] FERRITE_USE_REFERENCE=1 — GgmlLinear routes through dequant+cublas"
                );
            }
        });
    }

    pub fn is_enabled() -> bool {
        init();
        ENABLED.load(Ordering::Relaxed)
    }

    pub fn use_reference() -> bool {
        init();
        USE_REF.load(Ordering::Relaxed)
    }

    /// Pure dequant-on-the-fly + cuBLAS F32 GEMM inference (no MMVQ/DMMV).
    /// Output is returned in x's original dtype to match GgmlLinear::forward.
    pub unsafe fn reference_forward(
        storage: &crate::ggml::GgmlStorage,
        x: TensorView<'_>,
        bias: Option<GpuTensor>,
        cublas: &mut CublasHandle,
        alloc: &mut CachingAllocator,
        stream: cudarc::driver::sys::CUstream,
    ) -> OwnedTensor {
        let input_dtype = x.dtype();
        let x_cast = if input_dtype != DType::F32 {
            Some(crate::kernels::cast_logits_to_f32(*x, alloc, stream))
        } else {
            None
        };
        let x_f32 = x_cast.as_ref().map(|o| o.as_gpu_tensor()).unwrap_or(*x);
        let n = storage.nrows;
        let k = storage.ncols;

        let w_f32 =
            crate::ggml::ggml_dequantize_to_tensor(storage, DType::F32, &[n, k], alloc, stream);
        let out_f32 = cublas.gemm(x_f32, w_f32.as_gpu_tensor(), alloc);
        drop(w_f32);
        drop(x_cast);

        // Same `_bias_keepalive` pattern as `GgmlLinear::forward`:
        // a temp F32 cast of the bias must outlive the async
        // `bias_add_inplace` kernel because the caching allocator
        // can hand its pointer to the downstream `cast_from_f32`
        // call on the same stream. See
        // `feedback_tensorview_for_async_gpu`.
        let _bias_keepalive = if let Some(bias) = bias {
            if bias.dtype() == DType::F32 {
                crate::kernels::bias_add_inplace(out_f32.as_gpu_tensor(), bias, stream);
                None
            } else {
                let bias_f32 = crate::kernels::cast_logits_to_f32(bias, alloc, stream);
                crate::kernels::bias_add_inplace(
                    out_f32.as_gpu_tensor(),
                    bias_f32.as_gpu_tensor(),
                    stream,
                );
                Some(bias_f32)
            }
        } else {
            None
        };

        if input_dtype != DType::F32 {
            let out_f32_gpu = out_f32.as_gpu_tensor();
            let result = crate::kernels::cast_from_f32(out_f32_gpu, input_dtype, alloc, stream);
            drop(out_f32);
            drop(_bias_keepalive);
            result
        } else {
            drop(_bias_keepalive);
            out_f32
        }
    }

    /// Run ggml_matmul (quantized path) and a cuBLAS dequant reference on the
    /// same (storage, x), log per-call max_abs_diff to stderr. Allocations are
    /// dropped before return. Result is discarded — the real forward still runs.
    pub unsafe fn compare(
        storage: &crate::ggml::GgmlStorage,
        x: TensorView<'_>,
        cublas: &mut CublasHandle,
        alloc: &mut CachingAllocator,
        stream: cudarc::driver::sys::CUstream,
    ) {
        // 1. Cast activation to F32 (ggml_matmul requires f32).
        let input_dtype = x.dtype();
        let x_cast = if input_dtype != DType::F32 {
            Some(crate::kernels::cast_logits_to_f32(*x, alloc, stream))
        } else {
            None
        };
        let x_f32 = x_cast.as_ref().map(|o| o.as_gpu_tensor()).unwrap_or(*x);
        let num_tokens = x_f32.dim(0);
        let k = x_f32.dim(1);
        let n = storage.nrows;
        debug_assert_eq!(k, storage.ncols);

        // 1b. Measure input magnitude (over first min(N_tokens, 4) * min(K, 256) elements
        // of x_f32) so we can tell if divergence is in input or output.
        let x_peek = (num_tokens * k).min(2048);
        let mut host_x = vec![0f32; x_peek];
        ferrite_cuda_core::driver::memcpy_dtoh_async(
            host_x.as_mut_ptr() as *mut u8,
            x_f32.raw_ptr(),
            x_peek * 4,
            stream,
        )
        .expect("probe: dtoh x");
        ferrite_cuda_core::driver::stream_synchronize(stream).expect("probe: sync x");
        let x_max = host_x
            .iter()
            .fold(0.0f32, |m, &v| if v.is_nan() { m } else { m.max(v.abs()) });
        let x_nan = host_x.iter().filter(|v| v.is_nan()).count();

        // 2. Path A: current quantized kernel (MMVQ or DMMV).
        let out_a = crate::ggml::ggml_matmul(storage, x_f32, alloc, stream);

        // 3. Path B: dequant weight to F32, cuBLAS F32 GEMM.
        let w_f32 =
            crate::ggml::ggml_dequantize_to_tensor(storage, DType::F32, &[n, k], alloc, stream);
        let out_b = cublas.gemm(x_f32, w_f32.as_gpu_tensor(), alloc);

        // 4. Copy both outputs to host, compute diff.
        let nelem = num_tokens * n;
        let mut host_a = vec![0f32; nelem];
        let mut host_b = vec![0f32; nelem];
        ferrite_cuda_core::driver::memcpy_dtoh_async(
            host_a.as_mut_ptr() as *mut u8,
            out_a.as_gpu_tensor().raw_ptr(),
            nelem * 4,
            stream,
        )
        .expect("probe: dtoh A");
        ferrite_cuda_core::driver::memcpy_dtoh_async(
            host_b.as_mut_ptr() as *mut u8,
            out_b.as_gpu_tensor().raw_ptr(),
            nelem * 4,
            stream,
        )
        .expect("probe: dtoh B");
        ferrite_cuda_core::driver::stream_synchronize(stream).expect("probe: sync");

        let mut max_diff = 0.0f32;
        let mut first_diff_idx = usize::MAX;
        let mut first_diff_a = 0.0f32;
        let mut first_diff_b = 0.0f32;
        let mut nan_count_a = 0usize;
        let mut nan_count_b = 0usize;
        for i in 0..nelem {
            let a = host_a[i];
            let b = host_b[i];
            if a.is_nan() {
                nan_count_a += 1;
            }
            if b.is_nan() {
                nan_count_b += 1;
            }
            let d = (a - b).abs();
            if d.is_finite() && d > max_diff {
                max_diff = d;
                if first_diff_idx == usize::MAX && d > 1e-2 {
                    first_diff_idx = i;
                    first_diff_a = a;
                    first_diff_b = b;
                }
            }
        }

        let b_norm = host_b.iter().fold(0.0f32, |m, &v| m.max(v.abs()));
        let call = CALL_COUNT.fetch_add(1, Ordering::Relaxed);
        let rel = if b_norm > 0.0 { max_diff / b_norm } else { 0.0 };

        eprintln!(
            "[probe #{call:04}] dtype={:?} M={num_tokens} N={n} K={k} \
             x_max={x_max:.4e} x_nan={x_nan} \
             max_abs={max_diff:.4e} ref_max={b_norm:.4e} rel={rel:.4e} \
             nan_a={nan_count_a} nan_b={nan_count_b} \
             first_div_idx={first_diff_idx} a={first_diff_a:.4e} b={first_diff_b:.4e}",
            storage.dtype,
        );

        drop(out_b);
        drop(w_f32);
        drop(out_a);
        drop(x_cast);
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------
