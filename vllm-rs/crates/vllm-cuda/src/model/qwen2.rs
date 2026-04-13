// SPDX-License-Identifier: Apache-2.0
//! Qwen2 model — zero-allocation forward pass.
//!
//! Qwen2 is architecturally identical to LLaMA except:
//! - QKV projections include bias terms
//! - Default `rope_theta` is 1,000,000 (specified in config.json)
//!
//! The dense path uses a `forward!()` invocation that generates
//! `Model`, `Model::load()`, and `Model::forward()`. The DSL body
//! is the same as Llama's except for explicit `bias_add` ops after
//! Q/K/V GEMMs — this is the math, not an optimization flag.
//!
//! The legacy `Qwen2ForCausalLM` wrapper remains for quant/TP/PP
//! until those are ported to the solver.

use anyhow::Result;

use crate::alloc::OwnedTensor;
use crate::device::GpuDevice;
use crate::dtype::DType;
use crate::kernels;
use crate::kv_cache::KvCachePool;
use crate::layers::{Embedding, Linear, LinearLayer, RmsNorm};
use crate::model::llama::{ForwardOutput, LlamaConfig, LlamaForCausalLM, RotaryCache, TpConfig};
use crate::pp::PpConfig;
use crate::quant::QuantConfig;
use crate::tensor::{GpuTensor, TensorView};
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
// Legacy model (delegates to LLaMA — quant/TP/PP)
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

    /// Load the model (FP8 quantized).
    pub fn load_fp8(
        weights: &mut GpuWeights,
        config: &Qwen2Config,
        dtype: DType,
        device: &GpuDevice,
    ) -> Result<Self> {
        let model = LlamaForCausalLM::load_fp8(weights, &config.0, dtype, 0.0, device)?;
        Ok(Self(model))
    }

    /// Load the model (per-tensor FP8 + TP).
    pub fn load_fp8_tp(
        weights: &mut GpuWeights,
        config: &Qwen2Config,
        dtype: DType,
        tp: TpConfig,
        device: &GpuDevice,
    ) -> Result<Self> {
        let model = LlamaForCausalLM::load_fp8_tp(weights, &config.0, dtype, 0.0, tp, device)?;
        Ok(Self(model))
    }

    /// Load the model (block FP8 + TP).
    pub fn load_fp8_block_tp(
        weights: &mut GpuWeights,
        config: &Qwen2Config,
        dtype: DType,
        tp: TpConfig,
        device: &GpuDevice,
    ) -> Result<Self> {
        let model =
            LlamaForCausalLM::load_fp8_block_tp(weights, &config.0, dtype, 0.0, tp, device)?;
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
        input_ids: TensorView<'_>,
        positions: TensorView<'_>,
        slot_mapping: TensorView<'_>,
        cu_seqlens_q: TensorView<'_>,
        seqused_k: TensorView<'_>,
        block_table: TensorView<'_>,
        max_seqlen_q: usize,
        max_seqlen_k: usize,
        kv_cache: &KvCachePool,
        device: &mut GpuDevice,
        last_token_indices: Option<TensorView<'_>>,
    ) -> OwnedTensor {
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

    /// Load with PP (no TP).
    pub fn load_pp(
        weights: &mut GpuWeights,
        config: &Qwen2Config,
        dtype: DType,
        pp: PpConfig,
        device: &GpuDevice,
    ) -> Result<Self> {
        let model = LlamaForCausalLM::load_pp(weights, &config.0, dtype, pp, device)?;
        Ok(Self(model))
    }

    /// Load with TP + PP.
    pub fn load_tp_pp(
        weights: &mut GpuWeights,
        config: &Qwen2Config,
        dtype: DType,
        tp: TpConfig,
        pp: PpConfig,
        device: &GpuDevice,
    ) -> Result<Self> {
        let model = LlamaForCausalLM::load_tp_pp(weights, &config.0, dtype, tp, pp, device)?;
        Ok(Self(model))
    }

    /// PP-aware forward pass.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn forward_pp(
        &self,
        input_ids: Option<TensorView<'_>>,
        intermediate: Option<(OwnedTensor, OwnedTensor)>,
        positions: TensorView<'_>,
        slot_mapping: TensorView<'_>,
        cu_seqlens_q: TensorView<'_>,
        seqused_k: TensorView<'_>,
        block_table: TensorView<'_>,
        max_seqlen_q: usize,
        max_seqlen_k: usize,
        kv_cache: &KvCachePool,
        device: &mut GpuDevice,
        last_token_indices: Option<TensorView<'_>>,
    ) -> ForwardOutput {
        self.0.forward_pp(
            input_ids,
            intermediate,
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

// ---------------------------------------------------------------------------
// Ferrite solver dispatch — generated Qwen2 forward
// ---------------------------------------------------------------------------

// Re-use the same CUTLASS GEMM FFI as Llama.
macro_rules! cutlass_gemm_ffi {
    ($($name:ident),* $(,)?) => {
        #[cfg(feature = "cuda")]
        unsafe extern "C" {
            $(
                pub fn $name(
                    c: *mut u16, a: *const u16, b: *const u16,
                    m: i32, n: i32, k: i32,
                    alpha: f32, beta: f32, stream: u64,
                ) -> i32;
            )*
        }
    };
}

cutlass_gemm_ffi!(
    cutlass_gemm_32x64_s4_launch,
    cutlass_gemm_32x64_s3_launch,
    cutlass_gemm_32x128_s4_launch,
    cutlass_gemm_32x128_s3_launch,
    cutlass_gemm_32x256_s3_launch,
    cutlass_gemm_64x64_s4_launch,
    cutlass_gemm_64x64_s3_launch,
    cutlass_gemm_64x128_s4_launch,
    cutlass_gemm_64x128_s3_launch,
    cutlass_gemm_128x64_s4_launch,
    cutlass_gemm_128x64_s3_launch,
    cutlass_gemm_128x128_s4_launch,
    cutlass_gemm_128x128_s3_launch,
    cutlass_gemm_128x256_s3_launch,
    cutlass_gemm_256x64_s4_launch,
    cutlass_gemm_256x64_s3_launch,
    cutlass_gemv_launch,
    cutlass_gemm_128x128_launch,
    cutlass_gemm_64x64_launch,
);

vllm_tk_macros::forward! {
    hidden_states = embed(input_ids, embed_tokens);
    for layer in 0..NL {
        let normed = rmsnorm(hidden_states, input_layernorm[layer]);
        let q = gemm(normed, self_attn.q_proj[layer]);
        let q = bias_add(q, self_attn.q_proj.bias[layer]);
        let k = gemm(normed, self_attn.k_proj[layer]);
        let k = bias_add(k, self_attn.k_proj.bias[layer]);
        let v = gemm(normed, self_attn.v_proj[layer]);
        let v = bias_add(v, self_attn.v_proj.bias[layer]);
        let (q, k, v) = rope_append(q, k, v, positions, rotary, kv_cache[layer]);
        let attn = attention_decode(q, k, v, kv_cache[layer], block_table);
        hidden_states = gemm_add(attn, self_attn.o_proj[layer], hidden_states);

        let normed2 = rmsnorm(hidden_states, post_attention_layernorm[layer]);
        let gate = silu(gemm(normed2, mlp.gate_proj[layer]));
        let up = gemm(normed2, mlp.up_proj[layer]);
        hidden_states = gemm_add(gate * up, mlp.down_proj[layer], hidden_states);
    }
    hidden_states = rmsnorm(hidden_states, norm);
    logits = gemm(hidden_states, lm_head);

    models: [
        // Qwen2.5 0.5B
        { layers: 24, hidden: 896, intermediate: 4864, heads: 14, kv_heads: 2, head_dim: 64, vocab: 151936 },
        // Qwen2.5 1.5B
        { layers: 28, hidden: 1536, intermediate: 8960, heads: 12, kv_heads: 2, head_dim: 128, vocab: 151936 },
        // Qwen2.5 3B
        { layers: 36, hidden: 2048, intermediate: 11008, heads: 16, kv_heads: 2, head_dim: 128, vocab: 151936 },
        // Qwen2.5 7B
        { layers: 28, hidden: 3584, intermediate: 18944, heads: 28, kv_heads: 4, head_dim: 128, vocab: 151936 },
        // Qwen2.5 14B
        { layers: 48, hidden: 5120, intermediate: 13824, heads: 40, kv_heads: 8, head_dim: 128, vocab: 152064 },
        // Qwen2.5 32B
        { layers: 64, hidden: 5120, intermediate: 27648, heads: 40, kv_heads: 8, head_dim: 128, vocab: 152064 },
        // Qwen2.5 72B
        { layers: 80, hidden: 8192, intermediate: 29568, heads: 64, kv_heads: 8, head_dim: 128, vocab: 152064 },
    ],
    target: l4_sm89,
    workloads: [1..4096],
}
