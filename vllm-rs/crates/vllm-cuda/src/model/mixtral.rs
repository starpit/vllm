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
use crate::layers_moe::FusedMoELayer;
use crate::model::llama::{LlamaAttention, LlamaConfig, RotaryCache, TpConfig};
#[cfg(feature = "nccl")]
use crate::nccl::NcclGroup;
use crate::tensor::GpuTensor;
use crate::weights::GpuWeights;

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

pub struct MixtralDecoderLayer {
    pub self_attn: LlamaAttention,
    block_sparse_moe: FusedMoELayer,
    input_layernorm: RmsNorm,
    post_attention_layernorm: RmsNorm,
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
            block_sparse_moe: moe,
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

        Ok(FusedMoELayer {
            gate,
            w1,
            w2,
            num_experts,
            top_k: config.num_experts_per_tok,
            intermediate_size: ipp,
            hidden_size: hidden,
            renormalize: false, // Mixtral does NOT renormalize
            #[cfg(feature = "nccl")]
            tp_group: None,
        })
    }

    /// Load an FP8 decoder layer.
    /// Note: Only attention layers are FP8-quantized. MoE expert weights stay dense.
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
            block_sparse_moe: moe,
            input_layernorm,
            post_attention_layernorm,
        })
    }

    /// Forward pass — identical to LlamaDecoderLayer but with MoE MLP.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn forward_owned(
        &self,
        hidden_states: OwnedTensor,
        residual: Option<OwnedTensor>,
        positions: GpuTensor,
        slot_mapping: GpuTensor,
        cu_seqlens_q: GpuTensor,
        seqused_k: GpuTensor,
        block_table: GpuTensor,
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
        let attn_output = self.self_attn.forward_owned(
            *normed,
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
        let mlp_output = self.block_sparse_moe.forward_owned(*attn_output, device);
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
    ) -> GpuTensor {
        let hidden_states = kernels::embedding_gather(
            self.embed_tokens.weight,
            input_ids,
            &mut device.caching,
            device.compute_stream,
        );

        let mut hidden_states: OwnedTensor = hidden_states;
        let mut residual: Option<OwnedTensor> = None;

        for layer in &self.layers {
            let (hs, res) = layer.forward_owned(
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
        let res_gpu = residual.as_ref().unwrap().as_gpu_tensor();
        kernels::fused_add_rms_norm_inplace(
            hs_gpu,
            res_gpu,
            self.norm.weight,
            self.norm.eps,
            device.compute_stream,
        );
        drop(residual);
        hidden_states.into_gpu_tensor()
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
        let hidden_states = self.model.forward_owned(
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
                hidden_states,
                indices,
                &mut device.caching,
                device.compute_stream,
            )
            .into_gpu_tensor()
        } else {
            hidden_states
        };

        self.lm_head
            .forward(hidden_states, &mut device.cublas, &mut device.caching)
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
            block_sparse_moe: moe,
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
            layer.block_sparse_moe.tp_group = Some(Arc::clone(&group));
        }
    }
}
