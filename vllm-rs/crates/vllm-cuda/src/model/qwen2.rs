// SPDX-License-Identifier: Apache-2.0
//! Qwen2 model using `GpuTensor` — zero-allocation forward pass.
//!
//! Qwen2 is architecturally identical to LLaMA. The only differences are:
//! - QKV projections include bias terms (handled by `Linear::load` automatically)
//! - Default `rope_theta` is 1,000,000 (specified in config.json)
//!
//! This module re-exports LLaMA types with a Qwen2 config wrapper.

use anyhow::Result;

use crate::device::GpuDevice;
use crate::dtype::DType;
use crate::kv_cache::KvCachePool;
use crate::model::llama::{LlamaConfig, LlamaForCausalLM, TpConfig};
use crate::quant::QuantConfig;
use crate::tensor::GpuTensor;
use crate::weights::GpuWeights;

// ---------------------------------------------------------------------------
// Config
// ---------------------------------------------------------------------------

/// Qwen2 config — wraps `LlamaConfig` with Qwen2-specific defaults.
#[derive(Debug, Clone)]
pub struct Qwen2Config(pub LlamaConfig);

impl Qwen2Config {
    /// Create from a LlamaConfig, applying Qwen2 defaults.
    pub fn from_llama_config(mut config: LlamaConfig) -> Self {
        // Qwen2 defaults to rope_theta = 1M if not explicitly set.
        // The caller should parse this from config.json; this is a fallback.
        if config.rope_theta == 10000.0 {
            config.rope_theta = 1_000_000.0;
        }
        Self(config)
    }
}

// ---------------------------------------------------------------------------
// Model (delegates to LLaMA)
// ---------------------------------------------------------------------------

/// Qwen2 causal LM — structurally identical to LLaMA.
pub struct Qwen2ForCausalLM(pub LlamaForCausalLM);

impl Qwen2ForCausalLM {
    /// Load the model (dense).
    pub fn load(
        weights: &mut GpuWeights,
        config: &Qwen2Config,
        dtype: DType,
        device: &GpuDevice,
    ) -> Result<Self> {
        let model = LlamaForCausalLM::load(weights, &config.0, dtype, device)?;
        Ok(Self(model))
    }

    /// Load the model (quantized AWQ/GPTQ → Marlin).
    pub fn load_quantized(
        weights: &mut GpuWeights,
        config: &Qwen2Config,
        dtype: DType,
        qconfig: &QuantConfig,
        device: &GpuDevice,
    ) -> Result<Self> {
        let model = LlamaForCausalLM::load_quantized(weights, &config.0, dtype, qconfig, device)?;
        Ok(Self(model))
    }

    /// Load the model with TP sharding (dense).
    pub fn load_tp(
        weights: &mut GpuWeights,
        config: &Qwen2Config,
        dtype: DType,
        tp: TpConfig,
        device: &GpuDevice,
    ) -> Result<Self> {
        let model = LlamaForCausalLM::load_tp(weights, &config.0, dtype, tp, device)?;
        Ok(Self(model))
    }

    /// Load the model (BNB 4-bit).
    pub fn load_bnb4bit(
        weights: &mut GpuWeights,
        config: &Qwen2Config,
        dtype: DType,
        qconfig: &crate::quant::Bnb4bitConfig,
        device: &GpuDevice,
    ) -> Result<Self> {
        let model = LlamaForCausalLM::load_bnb4bit(weights, &config.0, dtype, qconfig, device)?;
        Ok(Self(model))
    }

    /// Forward pass: input_ids → logits.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn forward(
        &self,
        input_ids: GpuTensor,
        positions: GpuTensor,
        slot_mapping: GpuTensor,
        cu_seqlens_q: GpuTensor,
        seqused_k: GpuTensor,
        block_table: GpuTensor,
        max_seqlen_q: usize,
        max_seqlen_k: usize,
        kv_cache: &KvCachePool,
        device: &mut GpuDevice,
        last_token_indices: Option<GpuTensor>,
    ) -> GpuTensor {
        self.0.forward(
            input_ids,
            positions,
            slot_mapping,
            cu_seqlens_q,
            seqused_k,
            block_table,
            max_seqlen_q,
            max_seqlen_k,
            kv_cache,
            device,
            last_token_indices,
        )
    }

    /// Forward using caching allocator (zero D2D copies between layers).
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn forward_owned(
        &self,
        input_ids: GpuTensor,
        positions: GpuTensor,
        slot_mapping: GpuTensor,
        cu_seqlens_q: GpuTensor,
        seqused_k: GpuTensor,
        block_table: GpuTensor,
        max_seqlen_q: usize,
        max_seqlen_k: usize,
        kv_cache: &KvCachePool,
        device: &mut GpuDevice,
        last_token_indices: Option<GpuTensor>,
    ) -> GpuTensor {
        self.0.forward_owned(
            input_ids,
            positions,
            slot_mapping,
            cu_seqlens_q,
            seqused_k,
            block_table,
            max_seqlen_q,
            max_seqlen_k,
            kv_cache,
            device,
            last_token_indices,
        )
    }
}
