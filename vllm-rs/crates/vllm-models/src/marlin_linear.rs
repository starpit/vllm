// SPDX-License-Identifier: Apache-2.0
//! Marlin fused GEMM linear layer for INT4 quantized models (CUDA SM80+).
//!
//! `MarlinLinear` holds Marlin-tiled repacked INT4 weights and a pre-allocated
//! workspace. Its `forward()` calls the Marlin fused dequant+GEMM kernel
//! directly — a single kernel launch with no intermediate weight allocation.
//!
//! On model load, `GptqOrMarlin::load()` / `AwqOrMarlin::load()` automatically
//! converts to Marlin on CUDA, falling back to standard quantized linear on CPU.

use candle_core::{Device, Tensor};

#[cfg(feature = "cuda")]
use candle_core::DType;
#[cfg(feature = "cuda")]
use vllm_model::error::ModelError;
use vllm_model::error::ModelResult;
use vllm_model::layers::{AwqConfig, AwqLinear, GptqConfig, GptqLinear};
use vllm_model::weight::ModelWeights;

// ---------------------------------------------------------------------------
// MarlinLinear
// ---------------------------------------------------------------------------

/// A linear layer using Marlin fused INT4 GEMM (CUDA SM80+ only).
///
/// Created by converting a loaded `GptqLinear` or `AwqLinear` via the
/// one-time GPU repack kernels.
#[cfg(feature = "cuda")]
pub struct MarlinLinear {
    /// Marlin-tiled packed INT4 weights (repacked from GPTQ/AWQ format).
    b_q_weight: Tensor,
    /// Scale tensor `[num_groups, N]` (FP16 or BF16).
    b_scales: Tensor,
    /// Optional packed zero-point tensor.
    b_zeros: Option<Tensor>,
    /// Optional group index for act_order (desc_act).
    g_idx: Option<Tensor>,
    /// Optional permutation for act_order.
    perm: Option<Tensor>,
    /// Workspace tensor `[num_sms]` i32 for barrier sync.
    workspace: Tensor,
    /// Optional bias.
    bias: Option<Tensor>,
    /// Unquantized input features.
    size_k: usize,
    /// Output features.
    size_n: usize,
    /// Number of quantization groups.
    num_groups: usize,
    /// Group size (-1 for single group).
    group_size: i32,
    /// Weight type (GPTQ or AWQ).
    weight_type: vllm_kernels::marlin::MarlinWeightType,
}

#[cfg(feature = "cuda")]
impl MarlinLinear {
    /// Convert a GPTQ linear layer to Marlin format.
    ///
    /// Runs the GPTQ repack kernel on GPU (one-time cost at model load).
    pub fn from_gptq(
        gptq: &GptqLinear,
        _config: &GptqConfig,
        device: &Device,
    ) -> ModelResult<Self> {
        let size_k = gptq.in_features();
        let size_n = gptq.out_features();

        // Ensure weights are on the target CUDA device and contiguous
        let qweight = gptq
            .qweight()
            .to_device(device)
            .map_err(ModelError::Candle)?;
        let scales = gptq
            .scales()
            .to_device(device)
            .map_err(ModelError::Candle)?;

        // Repack to Marlin tiled layout
        let perm = gptq
            .g_idx()
            .map(|t| t.to_device(device))
            .transpose()
            .map_err(ModelError::Candle)?;
        let b_q_weight = vllm_kernels::marlin::gptq_repack(&qweight, perm.as_ref(), size_k, size_n)
            .map_err(|e| ModelError::Other(format!("GPTQ Marlin repack: {e}")))?;

        // Compute group info
        let qz_2d = gptq.qzeros().dims2().map_err(ModelError::Candle)?;
        let num_groups = qz_2d.0;
        let group_size = if num_groups > 1 {
            (size_k / num_groups) as i32
        } else {
            -1
        };

        // Allocate workspace
        let workspace = Self::alloc_workspace(device)?;

        Ok(Self {
            b_q_weight,
            b_scales: scales,
            b_zeros: None, // GPTQ uses bias-shifted (kU4B8), no explicit zeros
            g_idx: gptq
                .g_idx()
                .map(|t| t.to_device(device))
                .transpose()
                .map_err(ModelError::Candle)?,
            perm,
            workspace,
            bias: gptq
                .bias()
                .map(|t| t.to_device(device))
                .transpose()
                .map_err(ModelError::Candle)?,
            size_k,
            size_n,
            num_groups,
            group_size,
            weight_type: vllm_kernels::marlin::MarlinWeightType::GptqInt4,
        })
    }

    /// Convert an AWQ linear layer to Marlin format.
    pub fn from_awq(awq: &AwqLinear, _config: &AwqConfig, device: &Device) -> ModelResult<Self> {
        let size_k = awq.in_features();
        let size_n = awq.out_features();

        let qweight = awq
            .qweight()
            .to_device(device)
            .map_err(ModelError::Candle)?;
        let scales = awq.scales().to_device(device).map_err(ModelError::Candle)?;
        let qzeros = awq.qzeros().to_device(device).map_err(ModelError::Candle)?;

        // Repack to Marlin tiled layout
        let b_q_weight = vllm_kernels::marlin::awq_repack(&qweight, size_k, size_n)
            .map_err(|e| ModelError::Other(format!("AWQ Marlin repack: {e}")))?;

        // Group info
        let qz_2d = qzeros.dims2().map_err(ModelError::Candle)?;
        let num_groups = qz_2d.0;
        let group_size = if num_groups > 1 {
            (size_k / num_groups) as i32
        } else {
            -1
        };

        let workspace = Self::alloc_workspace(device)?;

        Ok(Self {
            b_q_weight,
            b_scales: scales,
            b_zeros: Some(qzeros), // AWQ uses kU4 with zero-points
            g_idx: None,
            perm: None,
            workspace,
            bias: awq
                .bias()
                .map(|t| t.to_device(device))
                .transpose()
                .map_err(ModelError::Candle)?,
            size_k,
            size_n,
            num_groups,
            group_size,
            weight_type: vllm_kernels::marlin::MarlinWeightType::AwqInt4,
        })
    }

    /// Allocate the workspace tensor (one i32 per SM for barrier sync).
    ///
    /// Conservative: allocate 256 slots (max SMs on any current GPU is ~144).
    fn alloc_workspace(device: &Device) -> ModelResult<Tensor> {
        // Use a conservative upper bound for SM count.
        // The Marlin kernel only uses `sms` slots for barrier locks.
        let max_sms = 256;
        Tensor::zeros(max_sms, DType::I32, device).map_err(ModelError::Candle)
    }

    /// Forward pass: fused dequant + GEMM.
    pub fn forward(&self, x: &Tensor) -> candle_core::Result<Tensor> {
        let output = vllm_kernels::marlin::marlin_gemm(
            x,
            &self.b_q_weight,
            &self.b_scales,
            self.b_zeros.as_ref(),
            self.g_idx.as_ref(),
            self.perm.as_ref(),
            &self.workspace,
            self.size_n,
            self.num_groups,
            self.group_size,
            self.weight_type,
        )
        .map_err(|e| candle_core::Error::Msg(format!("Marlin GEMM: {e}")))?;

        match &self.bias {
            Some(b) => output.broadcast_add(b),
            None => Ok(output),
        }
    }
}

// ---------------------------------------------------------------------------
// GptqOrMarlin: transparent dispatch enum
// ---------------------------------------------------------------------------

/// Quantized linear that automatically uses Marlin on CUDA SM80+.
pub enum GptqOrMarlin {
    /// CPU or SM<80 fallback: standard dequantize + cuBLAS.
    Gptq(GptqLinear),
    /// CUDA SM80+: Marlin fused GEMM.
    #[cfg(feature = "cuda")]
    Marlin(MarlinLinear),
}

impl GptqOrMarlin {
    /// Load a GPTQ linear layer, converting to Marlin on CUDA SM80+.
    pub fn load(
        weights: &mut ModelWeights,
        prefix: &str,
        config: &GptqConfig,
        device: &Device,
    ) -> ModelResult<Self> {
        let gptq = GptqLinear::from_weights(weights, prefix, config, device)?;

        #[cfg(feature = "cuda")]
        if device.is_cuda() {
            // Check alignment: Marlin requires K and N divisible by tile_size (16)
            // and N divisible by tile_n_size (64)
            let k = gptq.in_features();
            let n = gptq.out_features();
            if k % 16 == 0 && n % 64 == 0 {
                match MarlinLinear::from_gptq(&gptq, config, device) {
                    Ok(marlin) => {
                        eprintln!("[marlin] {prefix}: converted GPTQ → Marlin (K={k}, N={n})");
                        return Ok(Self::Marlin(marlin));
                    }
                    Err(e) => {
                        eprintln!("[marlin] {prefix}: conversion failed, using GPTQ fallback: {e}");
                    }
                }
            } else {
                eprintln!(
                    "[marlin] {prefix}: skipped (K={k} %16={}, N={n} %64={})",
                    k % 16,
                    n % 64
                );
            }
        }

        Ok(Self::Gptq(gptq))
    }

    /// Forward pass — dispatches to Marlin or GPTQ.
    pub fn forward(&self, x: &Tensor) -> candle_core::Result<Tensor> {
        match self {
            Self::Gptq(linear) => crate::ops::gptq_forward(linear, x),
            #[cfg(feature = "cuda")]
            Self::Marlin(linear) => linear.forward(x),
        }
    }

    pub fn in_features(&self) -> usize {
        match self {
            Self::Gptq(l) => l.in_features(),
            #[cfg(feature = "cuda")]
            Self::Marlin(l) => l.size_k,
        }
    }

    pub fn out_features(&self) -> usize {
        match self {
            Self::Gptq(l) => l.out_features(),
            #[cfg(feature = "cuda")]
            Self::Marlin(l) => l.size_n,
        }
    }
}

// ---------------------------------------------------------------------------
// AwqOrMarlin: transparent dispatch enum
// ---------------------------------------------------------------------------

/// AWQ linear that automatically uses Marlin on CUDA SM80+.
pub enum AwqOrMarlin {
    /// CPU or SM<80 fallback.
    Awq(AwqLinear),
    /// CUDA SM80+: Marlin fused GEMM.
    #[cfg(feature = "cuda")]
    Marlin(MarlinLinear),
}

impl AwqOrMarlin {
    /// Load an AWQ linear layer, converting to Marlin on CUDA SM80+.
    pub fn load(
        weights: &mut ModelWeights,
        prefix: &str,
        config: &AwqConfig,
        device: &Device,
    ) -> ModelResult<Self> {
        let awq = AwqLinear::from_weights(weights, prefix, config, device)?;

        #[cfg(feature = "cuda")]
        if device.is_cuda() {
            let k = awq.in_features();
            let n = awq.out_features();
            if k % 16 == 0 && n % 64 == 0 {
                match MarlinLinear::from_awq(&awq, config, device) {
                    Ok(marlin) => {
                        eprintln!("[marlin] {prefix}: converted AWQ → Marlin (K={k}, N={n})");
                        return Ok(Self::Marlin(marlin));
                    }
                    Err(e) => {
                        eprintln!("[marlin] {prefix}: conversion failed, using AWQ fallback: {e}");
                    }
                }
            } else {
                eprintln!(
                    "[marlin] {prefix}: skipped (K={k} %16={}, N={n} %64={})",
                    k % 16,
                    n % 64
                );
            }
        }

        Ok(Self::Awq(awq))
    }

    /// Forward pass — dispatches to Marlin or AWQ.
    pub fn forward(&self, x: &Tensor) -> candle_core::Result<Tensor> {
        match self {
            Self::Awq(linear) => crate::ops::awq_forward(linear, x),
            #[cfg(feature = "cuda")]
            Self::Marlin(linear) => linear.forward(x),
        }
    }
}
