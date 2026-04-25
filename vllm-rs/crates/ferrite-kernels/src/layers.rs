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

        if let Some(bias) = self.bias {
            crate::kernels::bias_add_inplace(out_f32.as_gpu_tensor(), bias, stream);
        }

        // Cast back to original dtype if we converted to f32.
        if input_dtype != ferrite_cuda_core::dtype::DType::F32 {
            let out_f32_gpu = out_f32.as_gpu_tensor();
            let result = crate::kernels::cast_from_f32(out_f32_gpu, input_dtype, alloc, stream);
            drop(out_f32);
            result
        } else {
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
            _ => panic!("dense_weight() called on quantized LinearLayer"),
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

        if let Some(bias) = bias {
            // bias is in model dtype; for F32 out, cast the bias to F32 once per call.
            // Keep simple: use bias_add if dtypes match, else convert on the fly.
            if bias.dtype() == DType::F32 {
                crate::kernels::bias_add_inplace(out_f32.as_gpu_tensor(), bias, stream);
            } else {
                let bias_f32 = crate::kernels::cast_logits_to_f32(bias, alloc, stream);
                crate::kernels::bias_add_inplace(
                    out_f32.as_gpu_tensor(),
                    bias_f32.as_gpu_tensor(),
                    stream,
                );
                drop(bias_f32);
            }
        }

        if input_dtype != DType::F32 {
            let out_f32_gpu = out_f32.as_gpu_tensor();
            let result = crate::kernels::cast_from_f32(out_f32_gpu, input_dtype, alloc, stream);
            drop(out_f32);
            result
        } else {
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
