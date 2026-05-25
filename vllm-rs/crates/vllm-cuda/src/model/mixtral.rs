// SPDX-License-Identifier: Apache-2.0
//! Mixtral model using `GpuTensor` — MoE architecture.
//!
//! Mixtral is structurally identical to LLaMA except:
//! - Every MLP is replaced by a Sparse MoE (8 experts, top-2 routing)
//! - Gate is a linear projection from hidden_size to num_experts
//!
//! Weight names: `model.layers.{l}.block_sparse_moe.{gate,experts.{e}.w1/w2/w3}`

#[cfg(feature = "nccl")]
use std::sync::Arc;

use anyhow::Result;

use crate::alloc::OwnedTensor;
use crate::device::GpuDevice;
use crate::dtype::DType;
use crate::kernels;
use crate::kv_cache::KvCachePool;
use crate::layers::{Linear, RmsNorm};
use crate::layers_moe::{DenseFusedMoELayer, Fp8FusedMoELayer, FusedMoELayer, MarlinFusedMoELayer};
use crate::model::llama::{LlamaAttention, LlamaConfig, RotaryCache, TpConfig};
#[cfg(feature = "nccl")]
use crate::nccl::NcclGroup;
use crate::quant::QuantConfig;
use crate::tensor::{GpuTensor, TensorView};
use crate::weights::{self as gpu_weights, GpuWeights};

// ---------------------------------------------------------------------------
// Config
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
pub struct MixtralConfig {
    /// Base LLaMA config fields.
    pub hidden_size: usize,
    pub num_attention_heads: usize,
    pub num_kv_heads: usize,
    pub num_hidden_layers: usize,
    pub intermediate_size: usize,
    pub vocab_size: usize,
    pub max_position_embeddings: usize,
    pub rms_norm_eps: f32,
    pub rope_theta: f64,
    pub head_dim: usize,
    pub tie_word_embeddings: bool,
    /// MoE fields.
    pub num_local_experts: usize,
    pub num_experts_per_tok: usize,
}

impl MixtralConfig {
    /// Convert to LlamaConfig for attention layer reuse.
    pub fn as_llama_config(&self) -> LlamaConfig {
        LlamaConfig {
            hidden_size: self.hidden_size,
            num_attention_heads: self.num_attention_heads,
            num_kv_heads: self.num_kv_heads,
            num_hidden_layers: self.num_hidden_layers,
            intermediate_size: self.intermediate_size,
            vocab_size: self.vocab_size,
            max_position_embeddings: self.max_position_embeddings,
            rms_norm_eps: self.rms_norm_eps,
            rope_theta: self.rope_theta,
            head_dim: self.head_dim,
            tie_word_embeddings: self.tie_word_embeddings,
            llama3_rope_scaling: None,
        }
    }
}

// ---------------------------------------------------------------------------
// MixtralDecoderLayer
// ---------------------------------------------------------------------------

#[allow(clippy::large_enum_variant)]
pub enum MixtralMoE {
    Dense(FusedMoELayer),
    Quantized(MarlinFusedMoELayer),
    Fp8(Fp8FusedMoELayer),
}

impl MixtralMoE {
    pub unsafe fn forward(
        &self,
        hidden_states: TensorView<'_>,
        device: &mut GpuDevice,
    ) -> OwnedTensor {
        match self {
            Self::Dense(moe) => moe.forward(hidden_states, device),
            Self::Quantized(moe) => moe.forward(hidden_states, device),
            Self::Fp8(moe) => moe.forward(hidden_states, device),
        }
    }
}

pub struct MixtralDecoderLayer {
    pub self_attn: LlamaAttention,
    pub block_sparse_moe: MixtralMoE,
    pub input_layernorm: RmsNorm,
    pub post_attention_layernorm: RmsNorm,
}

impl MixtralDecoderLayer {
    pub fn load(
        weights: &mut GpuWeights,
        prefix: &str,
        config: &MixtralConfig,
        layer_idx: usize,
        stream: cudarc::driver::sys::CUstream,
    ) -> Result<Self> {
        let llama_cfg = config.as_llama_config();
        let self_attn = LlamaAttention::load_fused(
            weights,
            &format!("{prefix}.self_attn"),
            &llama_cfg,
            layer_idx,
            stream,
        )?;

        let moe = Self::load_moe(
            weights,
            &format!("{prefix}.block_sparse_moe"),
            config,
            None,
            stream,
        )?;

        let input_layernorm = RmsNorm::load(
            weights,
            &format!("{prefix}.input_layernorm"),
            config.rms_norm_eps,
        )?;
        let post_attention_layernorm = RmsNorm::load(
            weights,
            &format!("{prefix}.post_attention_layernorm"),
            config.rms_norm_eps,
        )?;

        Ok(Self {
            self_attn,
            block_sparse_moe: MixtralMoE::Dense(moe),
            input_layernorm,
            post_attention_layernorm,
        })
    }

    /// Load MoE block: gate + stacked expert weights.
    ///
    /// Mixtral weight layout:
    /// - `gate.weight`: `[num_experts, hidden_size]`
    /// - `experts.{e}.w1.weight`: `[intermediate_size, hidden_size]` (gate_proj)
    /// - `experts.{e}.w3.weight`: `[intermediate_size, hidden_size]` (up_proj)
    /// - `experts.{e}.w2.weight`: `[hidden_size, intermediate_size]` (down_proj)
    ///
    /// We stack into:
    /// - `w1`: `[num_experts, 2*intermediate_size, hidden_size]` (gate+up fused)
    /// - `w2`: `[num_experts, hidden_size, intermediate_size]`
    fn load_moe(
        weights: &mut GpuWeights,
        prefix: &str,
        config: &MixtralConfig,
        tp: Option<TpConfig>,
        stream: cudarc::driver::sys::CUstream,
    ) -> Result<FusedMoELayer> {
        let num_experts = config.num_local_experts;
        let inter = config.intermediate_size;
        let hidden = config.hidden_size;
        let (rank, world_size) = tp.map_or((0, 1), |t| (t.rank, t.world_size));
        let ipp = inter / world_size; // intermediate_per_partition

        // Gate weight — NOT sharded (full on every rank).
        let gate = Linear::load(weights, &format!("{prefix}.gate"))?;

        // Get dtype from first expert weight.
        let first_w1_name = format!("{prefix}.experts.0.w1.weight");
        let (_, dtype) = weights
            .tensor_info(&first_w1_name)
            .ok_or_else(|| anyhow::anyhow!("weight not found: {first_w1_name}"))?;
        let elem = dtype.size_bytes();

        // Pre-allocate stacked tensors on GPU (using sharded intermediate size).
        let w1_bytes = num_experts * 2 * ipp * hidden * elem;
        let w2_bytes = num_experts * hidden * ipp * elem;
        let w1_ptr = unsafe { crate::driver::mem_alloc(w1_bytes)? };
        let w2_ptr = unsafe { crate::driver::mem_alloc(w2_bytes)? };

        // Copy per-expert weights into stacked tensors.
        for e in 0..num_experts {
            let w1_name = format!("{prefix}.experts.{e}.w1.weight"); // gate_proj [inter, hidden]
            let w3_name = format!("{prefix}.experts.{e}.w3.weight"); // up_proj [inter, hidden]
            let w2_name = format!("{prefix}.experts.{e}.w2.weight"); // down_proj [hidden, inter]

            let expert_w1_offset = e * 2 * ipp * hidden * elem;
            let gate_proj_bytes = ipp * hidden * elem;

            unsafe {
                if world_size > 1 {
                    // w1 (gate_proj): shard dim=0 → [ipp, hidden]
                    weights.take_shard_into(
                        &w1_name,
                        0,
                        rank,
                        world_size,
                        w1_ptr.add(expert_w1_offset),
                        stream,
                    )?;
                    // w3 (up_proj): shard dim=0 → [ipp, hidden]
                    weights.take_shard_into(
                        &w3_name,
                        0,
                        rank,
                        world_size,
                        w1_ptr.add(expert_w1_offset + gate_proj_bytes),
                        stream,
                    )?;
                    // w2 (down_proj): shard dim=1 → [hidden, ipp]
                    let expert_w2_offset = e * hidden * ipp * elem;
                    weights.take_shard_into(
                        &w2_name,
                        1,
                        rank,
                        world_size,
                        w2_ptr.add(expert_w2_offset),
                        stream,
                    )?;
                } else {
                    weights.take_into(&w1_name, w1_ptr.add(expert_w1_offset), stream)?;
                    weights.take_into(
                        &w3_name,
                        w1_ptr.add(expert_w1_offset + gate_proj_bytes),
                        stream,
                    )?;
                    let expert_w2_offset = e * hidden * inter * elem;
                    weights.take_into(&w2_name, w2_ptr.add(expert_w2_offset), stream)?;
                }
            }
        }

        let w1 = unsafe { GpuTensor::new(w1_ptr, &[num_experts, 2 * ipp, hidden], dtype) };
        let w2 = unsafe { GpuTensor::new(w2_ptr, &[num_experts, hidden, ipp], dtype) };

        Ok(FusedMoELayer::Dense(Box::new(DenseFusedMoELayer {
            gate,
            w1,
            w2,
            num_experts,
            top_k: config.num_experts_per_tok,
            intermediate_size: ipp,
            hidden_size: hidden,
            renormalize: false, // Mixtral does NOT renormalize
            e_score_correction_bias: None,
            n_expert_group: 0,
            topk_group: 0,
            routed_scaling_factor: 1.0,
            #[cfg(feature = "nccl")]
            tp_group: None,
        })))
    }

    /// Load an FP8 decoder layer.
    /// Detects FP8 expert weights and constructs Fp8FusedMoELayer if present,
    /// otherwise falls back to dense BF16 MoE.
    pub fn load_fp8(
        weights: &mut GpuWeights,
        prefix: &str,
        config: &MixtralConfig,
        layer_idx: usize,
        output_dtype: DType,
        stream: cudarc::driver::sys::CUstream,
    ) -> Result<Self> {
        let llama_cfg = config.as_llama_config();
        let self_attn = LlamaAttention::load_fp8(
            weights,
            &format!("{prefix}.self_attn"),
            &llama_cfg,
            layer_idx,
            output_dtype,
            0.0, // Mixtral does not use QK-norm
            stream,
        )?;

        let moe_prefix = format!("{prefix}.block_sparse_moe");

        // Detect FP8 expert weights.
        let first_w1_name = format!("{moe_prefix}.experts.0.w1.weight");
        let is_fp8 = weights
            .tensor_info(&first_w1_name)
            .map(|(_, dt)| dt == DType::Fp8E4m3)
            .unwrap_or(false);

        let block_sparse_moe = if is_fp8 {
            let fp8_moe = gpu_weights::load_fp8_moe_experts(
                weights,
                &moe_prefix,
                config.num_local_experts,
                config.intermediate_size,
                config.hidden_size,
                config.num_experts_per_tok,
                false, // Mixtral does NOT renormalize
                "w1",
                "w3",
                "w2",
            )?;
            MixtralMoE::Fp8(fp8_moe)
        } else {
            let moe = Self::load_moe(weights, &moe_prefix, config, None, stream)?;
            MixtralMoE::Dense(moe)
        };

        let input_layernorm = RmsNorm::load(
            weights,
            &format!("{prefix}.input_layernorm"),
            config.rms_norm_eps,
        )?;
        let post_attention_layernorm = RmsNorm::load(
            weights,
            &format!("{prefix}.post_attention_layernorm"),
            config.rms_norm_eps,
        )?;

        Ok(Self {
            self_attn,
            block_sparse_moe,
            input_layernorm,
            post_attention_layernorm,
        })
    }

    /// Load a quantized (AWQ/GPTQ → Marlin) decoder layer.
    #[allow(clippy::too_many_arguments)]
    pub fn load_quantized(
        weights: &mut GpuWeights,
        prefix: &str,
        config: &MixtralConfig,
        layer_idx: usize,
        qconfig: &QuantConfig,
        workspace: GpuTensor,
        device: &GpuDevice,
    ) -> Result<Self> {
        let llama_cfg = config.as_llama_config();
        let self_attn = LlamaAttention::load_quantized(
            weights,
            &format!("{prefix}.self_attn"),
            &llama_cfg,
            layer_idx,
            qconfig,
            workspace,
            device,
        )?;

        // Mixtral uses w1/w3/w2 naming (not gate_proj/up_proj/down_proj).
        // load_marlin_moe_layer expects gate_proj/up_proj/down_proj naming.
        // We need to handle the naming difference. Check which naming is present.
        let moe_prefix = format!("{prefix}.block_sparse_moe");
        let uses_w1w3 = weights.contains(&format!("{moe_prefix}.experts.0.w1.qweight"));

        let moe = if uses_w1w3 {
            // Mixtral naming: w1=gate_proj, w3=up_proj, w2=down_proj.
            // We can't directly use load_marlin_moe_layer since it expects gate_proj/up_proj/down_proj.
            // Instead, build it manually using the same approach.
            load_marlin_moe_layer_mixtral(
                weights,
                &moe_prefix,
                config.num_local_experts,
                config.intermediate_size,
                config.hidden_size,
                config.num_experts_per_tok,
                false, // Mixtral does NOT renormalize
                qconfig,
                device.device_id as i32,
            )?
        } else {
            // Some quantized Mixtral models use standard naming.
            gpu_weights::load_marlin_moe_layer(
                weights,
                &moe_prefix,
                config.num_local_experts,
                config.intermediate_size,
                config.hidden_size,
                config.num_experts_per_tok,
                false, // Mixtral does NOT renormalize
                qconfig,
                device.device_id as i32,
            )?
        };

        let input_layernorm = RmsNorm::load(
            weights,
            &format!("{prefix}.input_layernorm"),
            config.rms_norm_eps,
        )?;
        let post_attention_layernorm = RmsNorm::load(
            weights,
            &format!("{prefix}.post_attention_layernorm"),
            config.rms_norm_eps,
        )?;

        Ok(Self {
            self_attn,
            block_sparse_moe: MixtralMoE::Quantized(moe),
            input_layernorm,
            post_attention_layernorm,
        })
    }

    /// Forward pass — identical to LlamaDecoderLayer but with MoE MLP.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn forward(
        &self,
        hidden_states: OwnedTensor,
        residual: Option<OwnedTensor>,
        positions: TensorView<'_>,
        slot_mapping: TensorView<'_>,
        cu_seqlens_q: TensorView<'_>,
        seqused_k: TensorView<'_>,
        block_table: TensorView<'_>,
        max_seqlen_q: usize,
        max_seqlen_k: usize,
        kv_cache: &KvCachePool,
        rotary: &RotaryCache,
        device: &mut GpuDevice,
    ) -> (OwnedTensor, OwnedTensor) {
        // Pre-attention norm.
        let (normed, residual) = if let Some(residual) = residual {
            let hs_gpu = *hidden_states;
            let res_gpu = *residual;
            kernels::fused_add_rms_norm_inplace(
                hs_gpu,
                res_gpu,
                self.input_layernorm.weight,
                self.input_layernorm.eps,
                device.compute_stream,
            );
            (hidden_states, residual)
        } else {
            let normed = kernels::rms_norm(
                *hidden_states,
                self.input_layernorm.weight,
                self.input_layernorm.eps,
                &mut device.caching,
                device.compute_stream,
            );
            (normed, hidden_states)
        };

        // Attention.
        let attn_output = self.self_attn.forward(
            normed.view(),
            positions,
            slot_mapping,
            cu_seqlens_q,
            seqused_k,
            block_table,
            max_seqlen_q,
            max_seqlen_k,
            kv_cache,
            rotary,
            device,
        );
        drop(normed);

        // Post-attention norm.
        let res_gpu = *residual;
        kernels::fused_add_rms_norm_inplace(
            *attn_output,
            res_gpu,
            self.post_attention_layernorm.weight,
            self.post_attention_layernorm.eps,
            device.compute_stream,
        );

        // MoE MLP.
        let mlp_output = self.block_sparse_moe.forward(attn_output.view(), device);
        drop(attn_output);

        (mlp_output, residual)
    }
}

// ---------------------------------------------------------------------------
// MixtralModel
// ---------------------------------------------------------------------------

pub struct MixtralModel {
    pub embed_tokens: crate::layers::Embedding,
    pub layers: Vec<MixtralDecoderLayer>,
    pub norm: RmsNorm,
    pub rotary: RotaryCache,
}

impl MixtralModel {
    pub fn load(
        weights: &mut GpuWeights,
        config: &MixtralConfig,
        dtype: DType,
        device: &GpuDevice,
    ) -> Result<Self> {
        let embed_tokens = crate::layers::Embedding::load(weights, "model.embed_tokens")?;

        let mut layers = Vec::with_capacity(config.num_hidden_layers);
        for i in 0..config.num_hidden_layers {
            let layer = MixtralDecoderLayer::load(
                weights,
                &format!("model.layers.{i}"),
                config,
                i,
                device.compute_stream,
            )?;
            layers.push(layer);
        }

        let norm = RmsNorm::load(weights, "model.norm", config.rms_norm_eps)?;

        let rotary = unsafe {
            RotaryCache::new(
                config.head_dim,
                config.max_position_embeddings,
                config.rope_theta,
                None,
                dtype,
                device,
            )?
        };

        Ok(Self {
            embed_tokens,
            layers,
            norm,
            rotary,
        })
    }

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
    ) -> OwnedTensor {
        let hidden_states = kernels::embedding_gather(
            self.embed_tokens.weight,
            *input_ids,
            &mut device.caching,
            device.compute_stream,
        );

        let mut hidden_states: OwnedTensor = hidden_states;
        let mut residual: Option<OwnedTensor> = None;

        for layer in &self.layers {
            let (hs, res) = layer.forward(
                hidden_states,
                residual,
                positions,
                slot_mapping,
                cu_seqlens_q,
                seqused_k,
                block_table,
                max_seqlen_q,
                max_seqlen_k,
                kv_cache,
                &self.rotary,
                device,
            );
            hidden_states = hs;
            residual = Some(res);
        }

        let hs_gpu = *hidden_states;
        let res_gpu = *residual.as_ref().unwrap().view();
        kernels::fused_add_rms_norm_inplace(
            hs_gpu,
            res_gpu,
            self.norm.weight,
            self.norm.eps,
            device.compute_stream,
        );
        drop(residual);
        hidden_states
    }
}

// ---------------------------------------------------------------------------
// MixtralForCausalLM
// ---------------------------------------------------------------------------

pub struct MixtralForCausalLM {
    pub model: MixtralModel,
    pub lm_head: Linear,
}

impl MixtralForCausalLM {
    pub fn load(
        weights: &mut GpuWeights,
        config: &MixtralConfig,
        dtype: DType,
        device: &GpuDevice,
    ) -> Result<Self> {
        let model = MixtralModel::load(weights, config, dtype, device)?;

        let lm_head = if config.tie_word_embeddings {
            Linear::new(model.embed_tokens.weight, None)
        } else {
            Linear::load(weights, "lm_head")?
        };

        Ok(Self { model, lm_head })
    }

    /// Load an FP8 quantized Mixtral model.
    /// Note: Only attention layers are FP8. MoE expert weights remain dense.
    pub fn load_fp8(
        weights: &mut GpuWeights,
        config: &MixtralConfig,
        dtype: DType,
        device: &GpuDevice,
    ) -> Result<Self> {
        let embed_tokens = crate::layers::Embedding::load(weights, "model.embed_tokens")?;

        let mut layers = Vec::with_capacity(config.num_hidden_layers);
        for i in 0..config.num_hidden_layers {
            let layer = MixtralDecoderLayer::load_fp8(
                weights,
                &format!("model.layers.{i}"),
                config,
                i,
                dtype,
                device.compute_stream,
            )?;
            layers.push(layer);
        }

        let norm = RmsNorm::load(weights, "model.norm", config.rms_norm_eps)?;
        let rotary = unsafe {
            RotaryCache::new(
                config.head_dim,
                config.max_position_embeddings,
                config.rope_theta,
                None,
                dtype,
                device,
            )?
        };

        let model = MixtralModel {
            embed_tokens,
            layers,
            norm,
            rotary,
        };

        let lm_head = if config.tie_word_embeddings {
            Linear::new(model.embed_tokens.weight, None)
        } else {
            Linear::load(weights, "lm_head")?
        };

        Ok(Self { model, lm_head })
    }

    /// Load a quantized (AWQ/GPTQ → Marlin) Mixtral model.
    pub fn load_quantized(
        weights: &mut GpuWeights,
        config: &MixtralConfig,
        dtype: DType,
        qconfig: &QuantConfig,
        device: &GpuDevice,
    ) -> Result<Self> {
        let workspace = gpu_weights::alloc_marlin_workspace(device.num_sm, device.compute_stream)?;

        let embed_tokens = crate::layers::Embedding::load(weights, "model.embed_tokens")?;

        let mut layers = Vec::with_capacity(config.num_hidden_layers);
        for i in 0..config.num_hidden_layers {
            layers.push(MixtralDecoderLayer::load_quantized(
                weights,
                &format!("model.layers.{i}"),
                config,
                i,
                qconfig,
                workspace,
                device,
            )?);
        }

        let norm = RmsNorm::load(weights, "model.norm", config.rms_norm_eps)?;
        let rotary = unsafe {
            RotaryCache::new(
                config.head_dim,
                config.max_position_embeddings,
                config.rope_theta,
                None,
                dtype,
                device,
            )?
        };

        let model = MixtralModel {
            embed_tokens,
            layers,
            norm,
            rotary,
        };

        let lm_head = if config.tie_word_embeddings {
            Linear::new(model.embed_tokens.weight, None)
        } else {
            Linear::load(weights, "lm_head")?
        };

        Ok(Self { model, lm_head })
    }

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
        let hidden_states = self.model.forward(
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
        );

        let hidden_states = if let Some(indices) = last_token_indices {
            kernels::embedding_gather(
                *hidden_states,
                *indices,
                &mut device.caching,
                device.compute_stream,
            )
        } else {
            hidden_states
        };

        self.lm_head.forward(
            hidden_states.view(),
            &mut device.cublas,
            &mut device.caching,
        )
    }
}

// ---------------------------------------------------------------------------
// Tensor Parallelism
// ---------------------------------------------------------------------------

impl MixtralDecoderLayer {
    pub fn load_tp(
        weights: &mut GpuWeights,
        prefix: &str,
        config: &MixtralConfig,
        layer_idx: usize,
        tp: TpConfig,
        stream: cudarc::driver::sys::CUstream,
    ) -> Result<Self> {
        let llama_cfg = config.as_llama_config();
        let self_attn = LlamaAttention::load_fused_tp(
            weights,
            &format!("{prefix}.self_attn"),
            &llama_cfg,
            layer_idx,
            tp,
            stream,
        )?;

        let moe = Self::load_moe(
            weights,
            &format!("{prefix}.block_sparse_moe"),
            config,
            Some(tp),
            stream,
        )?;

        let input_layernorm = RmsNorm::load(
            weights,
            &format!("{prefix}.input_layernorm"),
            config.rms_norm_eps,
        )?;
        let post_attention_layernorm = RmsNorm::load(
            weights,
            &format!("{prefix}.post_attention_layernorm"),
            config.rms_norm_eps,
        )?;

        Ok(Self {
            self_attn,
            block_sparse_moe: MixtralMoE::Dense(moe),
            input_layernorm,
            post_attention_layernorm,
        })
    }
}

impl MixtralModel {
    pub fn load_tp(
        weights: &mut GpuWeights,
        config: &MixtralConfig,
        dtype: DType,
        tp: TpConfig,
        device: &GpuDevice,
    ) -> Result<Self> {
        let embed_tokens = crate::layers::Embedding::load(weights, "model.embed_tokens")?;

        let mut layers = Vec::with_capacity(config.num_hidden_layers);
        for i in 0..config.num_hidden_layers {
            layers.push(MixtralDecoderLayer::load_tp(
                weights,
                &format!("model.layers.{i}"),
                config,
                i,
                tp,
                device.compute_stream,
            )?);
        }

        let norm = RmsNorm::load(weights, "model.norm", config.rms_norm_eps)?;
        let rotary = unsafe {
            RotaryCache::new(
                config.head_dim,
                config.max_position_embeddings,
                config.rope_theta,
                None,
                dtype,
                device,
            )?
        };

        Ok(Self {
            embed_tokens,
            layers,
            norm,
            rotary,
        })
    }
}

impl MixtralForCausalLM {
    pub fn load_tp(
        weights: &mut GpuWeights,
        config: &MixtralConfig,
        dtype: DType,
        tp: TpConfig,
        device: &GpuDevice,
    ) -> Result<Self> {
        let model = MixtralModel::load_tp(weights, config, dtype, tp, device)?;
        let lm_head = if config.tie_word_embeddings {
            Linear::new(model.embed_tokens.weight, None)
        } else {
            Linear::load(weights, "lm_head")?
        };
        Ok(Self { model, lm_head })
    }

    #[cfg(feature = "nccl")]
    pub fn set_tp_group(&mut self, group: Arc<NcclGroup>) {
        for layer in &mut self.model.layers {
            layer.self_attn.tp_group = Some(Arc::clone(&group));
            match &mut layer.block_sparse_moe {
                MixtralMoE::Dense(moe) => {
                    let FusedMoELayer::Dense(inner) = moe;
                    inner.tp_group = Some(Arc::clone(&group));
                }
                MixtralMoE::Quantized(moe) => {
                    moe.tp_group = Some(Arc::clone(&group));
                }
                MixtralMoE::Fp8(moe) => {
                    moe.tp_group = Some(Arc::clone(&group));
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Mixtral-specific Marlin MoE weight loading
// ---------------------------------------------------------------------------

/// Load Marlin MoE weights using Mixtral's `w1`/`w3`/`w2` naming convention.
///
/// Mixtral uses: `experts.{e}.w1` (gate), `experts.{e}.w3` (up), `experts.{e}.w2` (down).
/// This is equivalent to `gate_proj`/`up_proj`/`down_proj` in other MoE models.
#[allow(clippy::too_many_arguments)]
fn load_marlin_moe_layer_mixtral(
    weights: &mut GpuWeights,
    prefix: &str,
    num_experts: usize,
    intermediate_size: usize,
    hidden_size: usize,
    top_k: usize,
    renormalize: bool,
    qconfig: &QuantConfig,
    device_id: i32,
) -> Result<MarlinFusedMoELayer> {
    use crate::driver;
    use crate::weights::{
        bytes_to_u16, bytes_to_u32, concat_bytes_dim1, concat_u16_dim1, concat_u32_dim1,
        load_expert_awq_cpu, load_expert_gptq_cpu,
    };

    let stream = weights.stream();

    let (group_size, has_zp, b_type_id) = match qconfig {
        QuantConfig::Awq(cfg) => (cfg.group_size, true, 1i32),
        QuantConfig::Gptq(cfg) => (cfg.group_size, false, 0i32),
        _ => anyhow::bail!(
            "load_marlin_moe_layer_mixtral: unsupported quant config {:?}",
            qconfig
        ),
    };

    let num_groups_w1 = hidden_size.checked_div(group_size).unwrap_or(1);
    let num_groups_w2 = intermediate_size.checked_div(group_size).unwrap_or(1);

    let w1_n = 2 * intermediate_size;
    let w1_packed_per_expert = hidden_size * w1_n / 8;
    let w2_packed_per_expert = intermediate_size * hidden_size / 8;

    let w1_total_bytes = num_experts * w1_packed_per_expert * 4;
    let w2_total_bytes = num_experts * w2_packed_per_expert * 4;
    let w1_ptr = unsafe { driver::mem_alloc(w1_total_bytes)? };
    let w2_ptr = unsafe { driver::mem_alloc(w2_total_bytes)? };
    weights.record_alloc(w1_ptr, w1_total_bytes);
    weights.record_alloc(w2_ptr, w2_total_bytes);

    // Determine scale dtype from first expert's scales.
    // Try standard naming first, then compressed-tensors naming.
    let first_scales_name = format!("{prefix}.experts.0.w1.scales");
    let first_ct_scales_name = format!("{prefix}.experts.0.w1.weight_scale");
    let scales_dtype = weights
        .tensor_info(&first_scales_name)
        .or_else(|| weights.tensor_info(&first_ct_scales_name))
        .map(|(_, dt)| dt)
        .unwrap_or(DType::BF16);
    let scale_elem = scales_dtype.size_bytes();

    let w1_scales_bytes = num_experts * num_groups_w1 * w1_n * scale_elem;
    let w2_scales_bytes = num_experts * num_groups_w2 * hidden_size * scale_elem;
    let w1_scales_ptr = unsafe { driver::mem_alloc(w1_scales_bytes)? };
    let w2_scales_ptr = unsafe { driver::mem_alloc(w2_scales_bytes)? };
    weights.record_alloc(w1_scales_ptr, w1_scales_bytes);
    weights.record_alloc(w2_scales_ptr, w2_scales_bytes);

    let (w1_zeros_ptr, w2_zeros_ptr) = if has_zp {
        let w1_zp_bytes = num_experts * num_groups_w1 * (w1_n / 8) * 4;
        let w2_zp_bytes = num_experts * num_groups_w2 * (hidden_size / 8) * 4;
        let p1 = unsafe { driver::mem_alloc(w1_zp_bytes)? };
        let p2 = unsafe { driver::mem_alloc(w2_zp_bytes)? };
        weights.record_alloc(p1, w1_zp_bytes);
        weights.record_alloc(p2, w2_zp_bytes);
        (Some(p1), Some(p2))
    } else {
        (None, None)
    };

    let is_gptq = b_type_id == 0;

    for e in 0..num_experts {
        // Mixtral naming: w1=gate_proj, w3=up_proj, w2=down_proj
        let gate_prefix = format!("{prefix}.experts.{e}.w1");
        let up_prefix = format!("{prefix}.experts.{e}.w3");
        let down_prefix = format!("{prefix}.experts.{e}.w2");

        if is_gptq {
            // GPTQ / compressed-tensors path
            let (gate_qw, gate_sc) = load_expert_gptq_cpu(weights, &gate_prefix)?;
            let (up_qw, up_sc) = load_expert_gptq_cpu(weights, &up_prefix)?;

            // GPTQ qweight is [K/8, N] — concat along dim1 gives [K/8, N1+N2]
            let k_packed = gate_qw.1[0];
            let fused_k = k_packed * 8;
            let fused_n = intermediate_size * 2;

            let fused_qw = {
                let gate_n = gate_qw.1[1];
                let up_n = up_qw.1[1];
                let elem = 4usize;
                let row_gate = gate_n * elem;
                let row_up = up_n * elem;
                let row_out = (gate_n + up_n) * elem;
                let mut out = vec![0u8; k_packed * row_out];
                for r in 0..k_packed {
                    out[r * row_out..r * row_out + row_gate]
                        .copy_from_slice(&gate_qw.0[r * row_gate..(r + 1) * row_gate]);
                    out[r * row_out + row_gate..r * row_out + row_out]
                        .copy_from_slice(&up_qw.0[r * row_up..(r + 1) * row_up]);
                }
                out
            };

            let qw_nbytes = fused_qw.len();
            let qw_gpu_ptr = unsafe { driver::mem_alloc(qw_nbytes)? };
            unsafe {
                driver::memcpy_htod_async(qw_gpu_ptr, fused_qw.as_ptr(), qw_nbytes, stream)?;
            }
            let qw_gpu = unsafe { GpuTensor::new(qw_gpu_ptr, &[k_packed, fused_n], DType::I32) };

            let expert_w1_offset = e * w1_packed_per_expert * 4;
            unsafe {
                crate::kernels::gptq_repack_into(
                    qw_gpu,
                    None,
                    w1_ptr.add(expert_w1_offset),
                    fused_k,
                    fused_n,
                    device_id,
                    stream,
                );
                driver::stream_synchronize(stream)?;
                driver::mem_free(qw_gpu_ptr)?;
            }

            let fused_sc =
                concat_u16_dim1(&gate_sc.0, &up_sc.0, gate_sc.1[0], gate_sc.1[1], up_sc.1[1]);
            let mut scales_u16 = fused_sc;
            crate::quant::marlin_permute_scales(&mut scales_u16, fused_k, fused_n, group_size);

            let expert_scales_offset = e * num_groups_w1 * fused_n * scale_elem;
            let scales_bytes: Vec<u8> = scales_u16.iter().flat_map(|&v| v.to_le_bytes()).collect();
            unsafe {
                driver::memcpy_htod_async(
                    w1_scales_ptr.add(expert_scales_offset),
                    scales_bytes.as_ptr(),
                    scales_bytes.len(),
                    stream,
                )?;
            }

            // --- w2: down proj ---
            let (down_qw, down_sc) = load_expert_gptq_cpu(weights, &down_prefix)?;
            let down_k_packed = down_qw.1[0];
            let down_k = down_k_packed * 8;
            let down_n = down_qw.1[1];

            let down_qw_nbytes = down_qw.0.len();
            let down_qw_gpu_ptr = unsafe { driver::mem_alloc(down_qw_nbytes)? };
            unsafe {
                driver::memcpy_htod_async(
                    down_qw_gpu_ptr,
                    down_qw.0.as_ptr(),
                    down_qw_nbytes,
                    stream,
                )?;
            }
            let down_qw_gpu =
                unsafe { GpuTensor::new(down_qw_gpu_ptr, &[down_k_packed, down_n], DType::I32) };

            let expert_w2_offset = e * w2_packed_per_expert * 4;
            unsafe {
                crate::kernels::gptq_repack_into(
                    down_qw_gpu,
                    None,
                    w2_ptr.add(expert_w2_offset),
                    down_k,
                    down_n,
                    device_id,
                    stream,
                );
                driver::stream_synchronize(stream)?;
                driver::mem_free(down_qw_gpu_ptr)?;
            }

            let mut down_scales_u16 = bytes_to_u16(&down_sc.0);
            crate::quant::marlin_permute_scales(&mut down_scales_u16, down_k, down_n, group_size);
            let expert_w2_scales_offset = e * num_groups_w2 * hidden_size * scale_elem;
            let down_scales_bytes: Vec<u8> = down_scales_u16
                .iter()
                .flat_map(|&v| v.to_le_bytes())
                .collect();
            unsafe {
                driver::memcpy_htod_async(
                    w2_scales_ptr.add(expert_w2_scales_offset),
                    down_scales_bytes.as_ptr(),
                    down_scales_bytes.len(),
                    stream,
                )?;
            }
        } else {
            // AWQ path
            let (gate_qw, gate_sc, gate_zp) = load_expert_awq_cpu(weights, &gate_prefix)?;
            let (up_qw, up_sc, up_zp) = load_expert_awq_cpu(weights, &up_prefix)?;

            let fused_qw =
                concat_bytes_dim1(&gate_qw.0, &up_qw.0, gate_qw.1[0], gate_qw.1[1], up_qw.1[1]);

            let fused_k = gate_qw.1[0];
            let fused_n = intermediate_size * 2;
            let qw_nbytes = fused_qw.len();
            let qw_gpu_ptr = unsafe { driver::mem_alloc(qw_nbytes)? };
            unsafe {
                driver::memcpy_htod_async(qw_gpu_ptr, fused_qw.as_ptr(), qw_nbytes, stream)?;
            }
            let qw_gpu = unsafe { GpuTensor::new(qw_gpu_ptr, &[fused_k, fused_n / 8], DType::U32) };

            let expert_w1_offset = e * w1_packed_per_expert * 4;
            unsafe {
                crate::kernels::awq_repack_into(
                    qw_gpu,
                    w1_ptr.add(expert_w1_offset),
                    fused_k,
                    fused_n,
                    device_id,
                    stream,
                );
                driver::stream_synchronize(stream)?;
                driver::mem_free(qw_gpu_ptr)?;
            }

            let fused_sc =
                concat_u16_dim1(&gate_sc.0, &up_sc.0, gate_sc.1[0], gate_sc.1[1], up_sc.1[1]);
            let mut scales_u16 = fused_sc;
            crate::quant::marlin_permute_scales(&mut scales_u16, fused_k, fused_n, group_size);

            let expert_scales_offset = e * num_groups_w1 * fused_n * scale_elem;
            let scales_bytes: Vec<u8> = scales_u16.iter().flat_map(|&v| v.to_le_bytes()).collect();
            unsafe {
                driver::memcpy_htod_async(
                    w1_scales_ptr.add(expert_scales_offset),
                    scales_bytes.as_ptr(),
                    scales_bytes.len(),
                    stream,
                )?;
            }

            if has_zp && let (Some(gate_zp), Some(up_zp)) = (gate_zp, up_zp) {
                let fused_zp_u32 =
                    concat_u32_dim1(&gate_zp.0, &up_zp.0, gate_zp.1[0], gate_zp.1[1], up_zp.1[1]);
                let marlin_zp =
                    crate::quant::awq_to_marlin_zero_points(&fused_zp_u32, num_groups_w1, fused_n);
                let zp_bytes: Vec<u8> = marlin_zp.iter().flat_map(|&v| v.to_le_bytes()).collect();
                let expert_zp_offset = e * num_groups_w1 * (fused_n / 8) * 4;
                unsafe {
                    driver::memcpy_htod_async(
                        w1_zeros_ptr.unwrap().add(expert_zp_offset),
                        zp_bytes.as_ptr(),
                        zp_bytes.len(),
                        stream,
                    )?;
                }
            }

            // --- w2: down proj ---
            let (down_qw, down_sc, down_zp) = load_expert_awq_cpu(weights, &down_prefix)?;
            let down_k = down_qw.1[0];
            let down_n = down_qw.1[1] * 8;

            let down_qw_nbytes = down_qw.0.len();
            let down_qw_gpu_ptr = unsafe { driver::mem_alloc(down_qw_nbytes)? };
            unsafe {
                driver::memcpy_htod_async(
                    down_qw_gpu_ptr,
                    down_qw.0.as_ptr(),
                    down_qw_nbytes,
                    stream,
                )?;
            }
            let down_qw_gpu =
                unsafe { GpuTensor::new(down_qw_gpu_ptr, &[down_k, down_n / 8], DType::U32) };

            let expert_w2_offset = e * w2_packed_per_expert * 4;
            unsafe {
                crate::kernels::awq_repack_into(
                    down_qw_gpu,
                    w2_ptr.add(expert_w2_offset),
                    down_k,
                    down_n,
                    device_id,
                    stream,
                );
                driver::stream_synchronize(stream)?;
                driver::mem_free(down_qw_gpu_ptr)?;
            }

            let mut down_scales_u16 = bytes_to_u16(&down_sc.0);
            crate::quant::marlin_permute_scales(&mut down_scales_u16, down_k, down_n, group_size);
            let expert_w2_scales_offset = e * num_groups_w2 * hidden_size * scale_elem;
            let down_scales_bytes: Vec<u8> = down_scales_u16
                .iter()
                .flat_map(|&v| v.to_le_bytes())
                .collect();
            unsafe {
                driver::memcpy_htod_async(
                    w2_scales_ptr.add(expert_w2_scales_offset),
                    down_scales_bytes.as_ptr(),
                    down_scales_bytes.len(),
                    stream,
                )?;
            }

            if has_zp && let Some(down_zp) = down_zp {
                let down_zp_u32 = bytes_to_u32(&down_zp.0);
                let marlin_zp =
                    crate::quant::awq_to_marlin_zero_points(&down_zp_u32, num_groups_w2, down_n);
                let zp_bytes: Vec<u8> = marlin_zp.iter().flat_map(|&v| v.to_le_bytes()).collect();
                let expert_w2_zp_offset = e * num_groups_w2 * (hidden_size / 8) * 4;
                unsafe {
                    driver::memcpy_htod_async(
                        w2_zeros_ptr.unwrap().add(expert_w2_zp_offset),
                        zp_bytes.as_ptr(),
                        zp_bytes.len(),
                        stream,
                    )?;
                }
            }
        }
    }

    unsafe { driver::stream_synchronize(stream)? };

    let w1_gpu =
        unsafe { GpuTensor::new(w1_ptr, &[num_experts, w1_packed_per_expert], DType::U32) };
    let w2_gpu =
        unsafe { GpuTensor::new(w2_ptr, &[num_experts, w2_packed_per_expert], DType::U32) };
    let w1_scales_gpu = unsafe {
        GpuTensor::new(
            w1_scales_ptr,
            &[num_experts, num_groups_w1, w1_n],
            scales_dtype,
        )
    };
    let w2_scales_gpu = unsafe {
        GpuTensor::new(
            w2_scales_ptr,
            &[num_experts, num_groups_w2, hidden_size],
            scales_dtype,
        )
    };

    let w1_zeros_gpu = w1_zeros_ptr
        .map(|p| unsafe { GpuTensor::new(p, &[num_experts, num_groups_w1, w1_n / 8], DType::U32) });
    let w2_zeros_gpu = w2_zeros_ptr.map(|p| unsafe {
        GpuTensor::new(
            p,
            &[num_experts, num_groups_w2, hidden_size / 8],
            DType::U32,
        )
    });

    let workspace_bytes = 256 * 4 * std::mem::size_of::<i32>();
    let workspace_ptr = unsafe { driver::mem_alloc(workspace_bytes)? };
    weights.record_alloc(workspace_ptr, workspace_bytes);
    let workspace_gpu =
        unsafe { GpuTensor::new(workspace_ptr, &[workspace_bytes / 4], DType::I32) };

    let gate = Linear::load(weights, &format!("{prefix}.gate"))?;

    Ok(MarlinFusedMoELayer {
        gate,
        w1: w1_gpu,
        w2: w2_gpu,
        w1_scales: w1_scales_gpu,
        w2_scales: w2_scales_gpu,
        w1_zeros: w1_zeros_gpu,
        w2_zeros: w2_zeros_gpu,
        workspace: workspace_gpu,
        num_experts,
        top_k,
        intermediate_size,
        hidden_size,
        group_size,
        has_zp,
        b_type_id,
        renormalize,
        e_score_correction_bias: None,
        n_expert_group: 0,
        topk_group: 0,
        routed_scaling_factor: 1.0,
        #[cfg(feature = "nccl")]
        tp_group: None,
    })
}
