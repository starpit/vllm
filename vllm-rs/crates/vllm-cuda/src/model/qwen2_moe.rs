// SPDX-License-Identifier: Apache-2.0
//! Qwen2 MoE model using `GpuTensor`.
//!
//! Differences from Mixtral:
//! - Some layers are dense (`mlp_only_layers`), others are MoE
//! - MoE layers have a shared expert (gated by sigmoid)
//! - Routing weights are renormalized (softmax top-k then renorm)
//! - Weight names: `mlp.gate`, `mlp.experts.{e}.gate_proj/up_proj/down_proj`
//! - Shared expert: `mlp.shared_expert.gate_proj/up_proj/down_proj`
//! - Shared expert gate: `mlp.shared_expert_gate.weight`
//! - QKV has bias

#[cfg(feature = "nccl")]
use std::sync::Arc;

use anyhow::Result;

use crate::alloc::OwnedTensor;
use crate::device::GpuDevice;
use crate::dtype::DType;
use crate::kernels;
use crate::kv_cache::KvCachePool;
use crate::layers::{Linear, LinearLayer, RmsNorm};
use crate::layers_moe::{
    DenseFusedMoELayer, Fp8FusedMoELayer, FusedMoELayer, MarlinSharedFusedMoELayer,
};
use crate::model::llama::{LlamaAttention, LlamaConfig, LlamaMLP, RotaryCache, TpConfig};
#[cfg(feature = "nccl")]
use crate::nccl::NcclGroup;
use crate::quant::QuantConfig;
use crate::tensor::{GpuTensor, TensorView};
use crate::weights::{self as gpu_weights, GpuWeights};

// ---------------------------------------------------------------------------
// Config
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
pub struct Qwen2MoeConfig {
    pub hidden_size: usize,
    pub num_attention_heads: usize,
    pub num_kv_heads: usize,
    pub num_hidden_layers: usize,
    pub intermediate_size: usize,     // dense MLP intermediate
    pub moe_intermediate_size: usize, // per-expert intermediate
    pub shared_expert_intermediate_size: usize,
    pub vocab_size: usize,
    pub max_position_embeddings: usize,
    pub rms_norm_eps: f32,
    pub rope_theta: f64,
    pub head_dim: usize,
    pub tie_word_embeddings: bool,
    pub num_experts: usize,
    pub num_experts_per_tok: usize,
    /// Layer indices that use dense MLP instead of MoE.
    pub mlp_only_layers: Vec<usize>,
}

impl Qwen2MoeConfig {
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
// MLP variants
// ---------------------------------------------------------------------------

#[allow(clippy::large_enum_variant)]
pub enum Qwen2MoeMlp {
    Dense(LlamaMLP),
    MoE {
        moe: FusedMoELayer,
        shared_gate_up: Linear,
        shared_down: Linear,
        shared_expert_gate: Linear,
        shared_intermediate_size: usize,
    },
    QuantizedMoE(MarlinSharedFusedMoELayer),
    Fp8MoE {
        moe: Fp8FusedMoELayer,
        shared_gate_up: Linear,
        shared_down: Linear,
        shared_expert_gate: Linear,
        shared_intermediate_size: usize,
    },
}

impl Qwen2MoeMlp {
    pub unsafe fn forward(
        &self,
        hidden_states: TensorView<'_>,
        device: &mut GpuDevice,
    ) -> OwnedTensor {
        match self {
            Self::Dense(mlp) => mlp.forward(hidden_states, device),
            Self::MoE {
                moe,
                shared_gate_up,
                shared_down,
                shared_expert_gate,
                shared_intermediate_size,
            } => {
                let stream = device.compute_stream;

                // MoE path.
                let moe_out = moe.forward(hidden_states, device);

                // Shared expert path.
                let shared_gu =
                    shared_gate_up.forward(hidden_states, &mut device.cublas, &mut device.caching);
                let shared_activated = kernels::silu_and_mul_fused(
                    *shared_gu.view(),
                    *shared_intermediate_size,
                    &mut device.caching,
                    stream,
                );
                drop(shared_gu);

                let shared_out = shared_down.forward(
                    shared_activated.view(),
                    &mut device.cublas,
                    &mut device.caching,
                );
                drop(shared_activated);

                // Shared expert gate: sigmoid(gate(hidden_states)) * shared_out + moe_out
                let gate_logits = shared_expert_gate.forward(
                    hidden_states,
                    &mut device.cublas,
                    &mut device.caching,
                );

                // out = moe_out + sigmoid(gate_logits) * shared_out
                let result = kernels::sigmoid_mul_add(
                    *moe_out.view(),
                    *shared_out.view(),
                    *gate_logits.view(),
                    &mut device.caching,
                    stream,
                );
                drop(moe_out);
                drop(shared_out);
                drop(gate_logits);

                result
            }
            Self::QuantizedMoE(layer) => layer.forward(hidden_states, device),
            Self::Fp8MoE {
                moe,
                shared_gate_up,
                shared_down,
                shared_expert_gate,
                shared_intermediate_size,
            } => {
                let stream = device.compute_stream;
                let moe_out = moe.forward(hidden_states, device);

                let shared_gu =
                    shared_gate_up.forward(hidden_states, &mut device.cublas, &mut device.caching);
                let shared_activated = kernels::silu_and_mul_fused(
                    *shared_gu.view(),
                    *shared_intermediate_size,
                    &mut device.caching,
                    stream,
                );
                drop(shared_gu);

                let shared_out = shared_down.forward(
                    shared_activated.view(),
                    &mut device.cublas,
                    &mut device.caching,
                );
                drop(shared_activated);

                let gate_logits = shared_expert_gate.forward(
                    hidden_states,
                    &mut device.cublas,
                    &mut device.caching,
                );

                let result = kernels::sigmoid_mul_add(
                    *moe_out.view(),
                    *shared_out.view(),
                    *gate_logits.view(),
                    &mut device.caching,
                    stream,
                );
                drop(moe_out);
                drop(shared_out);
                drop(gate_logits);

                result
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Decoder layer
// ---------------------------------------------------------------------------

pub struct Qwen2MoeDecoderLayer {
    pub self_attn: LlamaAttention,
    pub mlp: Qwen2MoeMlp,
    pub input_layernorm: RmsNorm,
    pub post_attention_layernorm: RmsNorm,
}

impl Qwen2MoeDecoderLayer {
    pub fn load(
        weights: &mut GpuWeights,
        prefix: &str,
        config: &Qwen2MoeConfig,
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

        let is_dense = config.mlp_only_layers.contains(&layer_idx);
        let mlp = if is_dense {
            let dense = LlamaMLP::load_fused(
                weights,
                &format!("{prefix}.mlp"),
                config.intermediate_size,
                stream,
            )?;
            Qwen2MoeMlp::Dense(dense)
        } else {
            Self::load_moe(weights, &format!("{prefix}.mlp"), config, None, stream)?
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
            mlp,
            input_layernorm,
            post_attention_layernorm,
        })
    }

    fn load_moe(
        weights: &mut GpuWeights,
        prefix: &str,
        config: &Qwen2MoeConfig,
        tp: Option<TpConfig>,
        stream: cudarc::driver::sys::CUstream,
    ) -> Result<Qwen2MoeMlp> {
        let num_experts = config.num_experts;
        let inter = config.moe_intermediate_size;
        let hidden = config.hidden_size;
        let (rank, world_size) = tp.map_or((0, 1), |t| (t.rank, t.world_size));
        let ipp = inter / world_size; // intermediate_per_partition

        let gate = Linear::load(weights, &format!("{prefix}.gate"))?;

        let first_gate = format!("{prefix}.experts.0.gate_proj.weight");
        let (_, dtype) = weights
            .tensor_info(&first_gate)
            .ok_or_else(|| anyhow::anyhow!("weight not found: {first_gate}"))?;
        let elem = dtype.size_bytes();

        // Stack expert weights (using sharded intermediate size).
        let w1_bytes = num_experts * 2 * ipp * hidden * elem;
        let w2_bytes = num_experts * hidden * ipp * elem;
        let w1_ptr = unsafe { crate::driver::mem_alloc(w1_bytes)? };
        let w2_ptr = unsafe { crate::driver::mem_alloc(w2_bytes)? };

        for e in 0..num_experts {
            let gate_name = format!("{prefix}.experts.{e}.gate_proj.weight");
            let up_name = format!("{prefix}.experts.{e}.up_proj.weight");
            let down_name = format!("{prefix}.experts.{e}.down_proj.weight");

            let expert_w1_offset = e * 2 * ipp * hidden * elem;
            let gate_proj_bytes = ipp * hidden * elem;

            unsafe {
                if world_size > 1 {
                    weights.take_shard_into(
                        &gate_name,
                        0,
                        rank,
                        world_size,
                        w1_ptr.add(expert_w1_offset),
                        stream,
                    )?;
                    weights.take_shard_into(
                        &up_name,
                        0,
                        rank,
                        world_size,
                        w1_ptr.add(expert_w1_offset + gate_proj_bytes),
                        stream,
                    )?;
                    let expert_w2_offset = e * hidden * ipp * elem;
                    weights.take_shard_into(
                        &down_name,
                        1,
                        rank,
                        world_size,
                        w2_ptr.add(expert_w2_offset),
                        stream,
                    )?;
                } else {
                    weights.take_into(&gate_name, w1_ptr.add(expert_w1_offset), stream)?;
                    weights.take_into(
                        &up_name,
                        w1_ptr.add(expert_w1_offset + gate_proj_bytes),
                        stream,
                    )?;
                    let expert_w2_offset = e * hidden * inter * elem;
                    weights.take_into(&down_name, w2_ptr.add(expert_w2_offset), stream)?;
                }
            }
        }

        let w1 = unsafe { GpuTensor::new(w1_ptr, &[num_experts, 2 * ipp, hidden], dtype) };
        let w2 = unsafe { GpuTensor::new(w2_ptr, &[num_experts, hidden, ipp], dtype) };

        let moe = FusedMoELayer::Dense(Box::new(DenseFusedMoELayer {
            gate,
            w1,
            w2,
            num_experts,
            top_k: config.num_experts_per_tok,
            intermediate_size: ipp,
            hidden_size: hidden,
            renormalize: true, // Qwen2 MoE renormalizes
            e_score_correction_bias: None,
            n_expert_group: 0,
            topk_group: 0,
            routed_scaling_factor: 1.0,
            #[cfg(feature = "nccl")]
            tp_group: None,
        }));

        // Shared expert.
        let shared_inter = config.shared_expert_intermediate_size;
        let sipp = shared_inter / world_size;

        let shared_gate_up = if world_size > 1 {
            let gate_name = format!("{prefix}.shared_expert.gate_proj.weight");
            let up_name = format!("{prefix}.shared_expert.up_proj.weight");
            let gate_proj_bytes = sipp * hidden * elem;
            let total = 2 * gate_proj_bytes;
            let ptr = unsafe { crate::driver::mem_alloc(total)? };
            unsafe {
                weights.take_shard_into(&gate_name, 0, rank, world_size, ptr, stream)?;
                weights.take_shard_into(
                    &up_name,
                    0,
                    rank,
                    world_size,
                    ptr.add(gate_proj_bytes),
                    stream,
                )?;
            }
            let w = unsafe { GpuTensor::new(ptr, &[2 * sipp, hidden], dtype) };
            Linear::new(w, None)
        } else {
            let gate_name = format!("{prefix}.shared_expert.gate_proj.weight");
            let up_name = format!("{prefix}.shared_expert.up_proj.weight");
            let gate_proj_bytes = shared_inter * hidden * elem;
            let total = 2 * gate_proj_bytes;
            let ptr = unsafe { crate::driver::mem_alloc(total)? };
            unsafe {
                weights.take_into(&gate_name, ptr, stream)?;
                weights.take_into(&up_name, ptr.add(gate_proj_bytes), stream)?;
            }
            let w = unsafe { GpuTensor::new(ptr, &[2 * shared_inter, hidden], dtype) };
            Linear::new(w, None)
        };

        let shared_down = if world_size > 1 {
            let name = format!("{prefix}.shared_expert.down_proj.weight");
            let w = weights.take_shard(&name, 1, rank, world_size)?;
            Linear::new(w, None)
        } else {
            Linear::load(weights, &format!("{prefix}.shared_expert.down_proj"))?
        };

        // Shared expert gate: NOT sharded (tiny [1, hidden]).
        let shared_expert_gate = Linear::load(weights, &format!("{prefix}.shared_expert_gate"))?;

        Ok(Qwen2MoeMlp::MoE {
            moe,
            shared_gate_up,
            shared_down,
            shared_expert_gate,
            shared_intermediate_size: sipp,
        })
    }

    /// Load an FP8 quantized decoder layer.
    /// Only attention and dense MLP layers are FP8. MoE expert weights remain dense.
    pub fn load_fp8(
        weights: &mut GpuWeights,
        prefix: &str,
        config: &Qwen2MoeConfig,
        layer_idx: usize,
        dtype: DType,
        stream: cudarc::driver::sys::CUstream,
    ) -> Result<Self> {
        let llama_cfg = config.as_llama_config();
        let self_attn = LlamaAttention::load_fp8(
            weights,
            &format!("{prefix}.self_attn"),
            &llama_cfg,
            layer_idx,
            dtype,
            0.0, // Qwen2 MoE does not use QK-norm
            stream,
        )?;

        let is_dense = config.mlp_only_layers.contains(&layer_idx);
        let mlp = if is_dense {
            let dense = LlamaMLP::load_fp8(
                weights,
                &format!("{prefix}.mlp"),
                config.intermediate_size,
                dtype,
                stream,
            )?;
            Qwen2MoeMlp::Dense(dense)
        } else {
            // Check if expert weights are FP8.
            let moe_prefix = format!("{prefix}.mlp");
            let first_gate_name = format!("{moe_prefix}.experts.0.gate_proj.weight");
            let is_fp8 = weights
                .tensor_info(&first_gate_name)
                .map(|(_, dt)| dt == DType::Fp8E4m3)
                .unwrap_or(false);

            if is_fp8 {
                let fp8_moe = gpu_weights::load_fp8_moe_experts(
                    weights,
                    &moe_prefix,
                    config.num_experts,
                    config.moe_intermediate_size,
                    config.hidden_size,
                    config.num_experts_per_tok,
                    true, // Qwen2 MoE renormalizes
                    "gate_proj",
                    "up_proj",
                    "down_proj",
                )?;

                // Load shared expert (dense BF16) — fuse gate+up manually.
                let shared_gate_up = {
                    let gate_name = format!("{moe_prefix}.shared_expert.gate_proj.weight");
                    let up_name = format!("{moe_prefix}.shared_expert.up_proj.weight");
                    let (_, se_dtype) = weights
                        .tensor_info(&gate_name)
                        .ok_or_else(|| anyhow::anyhow!("weight not found: {gate_name}"))?;
                    let se_elem = se_dtype.size_bytes();
                    let se_inter = config.shared_expert_intermediate_size;
                    let hidden = config.hidden_size;
                    let gate_proj_bytes = se_inter * hidden * se_elem;
                    let total = 2 * gate_proj_bytes;
                    let ptr = unsafe { crate::driver::mem_alloc(total)? };
                    unsafe {
                        weights.take_into(&gate_name, ptr, stream)?;
                        weights.take_into(&up_name, ptr.add(gate_proj_bytes), stream)?;
                    }
                    let w = unsafe { GpuTensor::new(ptr, &[2 * se_inter, hidden], se_dtype) };
                    Linear::new(w, None)
                };
                let shared_down =
                    Linear::load(weights, &format!("{moe_prefix}.shared_expert.down_proj"))?;
                let shared_expert_gate =
                    Linear::load(weights, &format!("{moe_prefix}.shared_expert_gate"))?;

                Qwen2MoeMlp::Fp8MoE {
                    moe: fp8_moe,
                    shared_gate_up,
                    shared_down,
                    shared_expert_gate,
                    shared_intermediate_size: config.shared_expert_intermediate_size,
                }
            } else {
                Self::load_moe(weights, &moe_prefix, config, None, stream)?
            }
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
            mlp,
            input_layernorm,
            post_attention_layernorm,
        })
    }

    /// Load a quantized (AWQ/GPTQ → Marlin) decoder layer.
    #[allow(clippy::too_many_arguments)]
    pub fn load_quantized(
        weights: &mut GpuWeights,
        prefix: &str,
        config: &Qwen2MoeConfig,
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

        let is_dense = config.mlp_only_layers.contains(&layer_idx);
        let mlp = if is_dense {
            let dense = LlamaMLP::load_quantized(
                weights,
                &format!("{prefix}.mlp"),
                config.intermediate_size,
                qconfig,
                workspace,
                device,
            )?;
            Qwen2MoeMlp::Dense(dense)
        } else {
            Self::load_quantized_moe(
                weights,
                &format!("{prefix}.mlp"),
                config,
                qconfig,
                workspace,
                device,
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
            mlp,
            input_layernorm,
            post_attention_layernorm,
        })
    }

    #[allow(clippy::too_many_arguments)]
    fn load_quantized_moe(
        weights: &mut GpuWeights,
        prefix: &str,
        config: &Qwen2MoeConfig,
        qconfig: &QuantConfig,
        workspace: GpuTensor,
        device: &GpuDevice,
    ) -> Result<Qwen2MoeMlp> {
        let moe = gpu_weights::load_marlin_moe_layer(
            weights,
            prefix,
            config.num_experts,
            config.moe_intermediate_size,
            config.hidden_size,
            config.num_experts_per_tok,
            true, // renormalize
            qconfig,
            device.device_id as i32,
        )?;

        let shared_inter = config.shared_expert_intermediate_size;

        let mut alloc = crate::alloc::CachingAllocator::new();
        let gate_up = gpu_weights::load_fused_marlin_linear(
            weights,
            &[
                format!("{prefix}.shared_expert.gate_proj"),
                format!("{prefix}.shared_expert.up_proj"),
            ],
            qconfig,
            workspace,
            device.device_id,
            &mut alloc,
        )?;
        let down = gpu_weights::load_marlin_linear(
            weights,
            &format!("{prefix}.shared_expert.down_proj"),
            qconfig,
            workspace,
            device.device_id,
            &mut alloc,
        )?;

        let gate = Linear::load(weights, &format!("{prefix}.shared_expert_gate"))?;

        Ok(Qwen2MoeMlp::QuantizedMoE(MarlinSharedFusedMoELayer {
            moe,
            shared_gate_up: Some(LinearLayer::Marlin(Box::new(gate_up))),
            shared_down: Some(LinearLayer::Marlin(Box::new(down))),
            shared_expert_gate: Some(gate),
            intermediate_size: shared_inter,
        }))
    }

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

        let res_gpu = *residual;
        kernels::fused_add_rms_norm_inplace(
            *attn_output,
            res_gpu,
            self.post_attention_layernorm.weight,
            self.post_attention_layernorm.eps,
            device.compute_stream,
        );

        let mlp_output = self.mlp.forward(attn_output.view(), device);
        drop(attn_output);

        (mlp_output, residual)
    }
}

// ---------------------------------------------------------------------------
// Qwen2MoeForCausalLM
// ---------------------------------------------------------------------------

pub struct Qwen2MoeModel {
    pub embed_tokens: crate::layers::Embedding,
    pub layers: Vec<Qwen2MoeDecoderLayer>,
    pub norm: RmsNorm,
    pub rotary: RotaryCache,
}

impl Qwen2MoeModel {
    pub fn load(
        weights: &mut GpuWeights,
        config: &Qwen2MoeConfig,
        dtype: DType,
        device: &GpuDevice,
    ) -> Result<Self> {
        let embed_tokens = crate::layers::Embedding::load(weights, "model.embed_tokens")?;

        let mut layers = Vec::with_capacity(config.num_hidden_layers);
        for i in 0..config.num_hidden_layers {
            layers.push(Qwen2MoeDecoderLayer::load(
                weights,
                &format!("model.layers.{i}"),
                config,
                i,
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

    /// Load an FP8 quantized Qwen2 MoE model.
    /// Only attention and dense MLP layers are FP8. MoE expert weights remain dense.
    pub fn load_fp8(
        weights: &mut GpuWeights,
        config: &Qwen2MoeConfig,
        dtype: DType,
        device: &GpuDevice,
    ) -> Result<Self> {
        let embed_tokens = crate::layers::Embedding::load(weights, "model.embed_tokens")?;

        let mut layers = Vec::with_capacity(config.num_hidden_layers);
        for i in 0..config.num_hidden_layers {
            layers.push(Qwen2MoeDecoderLayer::load_fp8(
                weights,
                &format!("model.layers.{i}"),
                config,
                i,
                dtype,
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

    /// Load a quantized (AWQ/GPTQ → Marlin) Qwen2 MoE model.
    pub fn load_quantized(
        weights: &mut GpuWeights,
        config: &Qwen2MoeConfig,
        dtype: DType,
        qconfig: &QuantConfig,
        device: &GpuDevice,
    ) -> Result<Self> {
        let workspace = gpu_weights::alloc_marlin_workspace(device.num_sm, device.compute_stream)?;

        let embed_tokens = crate::layers::Embedding::load(weights, "model.embed_tokens")?;

        let mut layers = Vec::with_capacity(config.num_hidden_layers);
        for i in 0..config.num_hidden_layers {
            layers.push(Qwen2MoeDecoderLayer::load_quantized(
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

pub struct Qwen2MoeForCausalLM {
    pub model: Qwen2MoeModel,
    pub lm_head: Linear,
}

impl Qwen2MoeForCausalLM {
    pub fn load(
        weights: &mut GpuWeights,
        config: &Qwen2MoeConfig,
        dtype: DType,
        device: &GpuDevice,
    ) -> Result<Self> {
        let model = Qwen2MoeModel::load(weights, config, dtype, device)?;
        let lm_head = if config.tie_word_embeddings {
            Linear::new(model.embed_tokens.weight, None)
        } else {
            Linear::load(weights, "lm_head")?
        };
        Ok(Self { model, lm_head })
    }

    /// Load an FP8 quantized Qwen2 MoE model.
    /// Only attention and dense MLP layers are FP8. MoE expert weights remain dense.
    pub fn load_fp8(
        weights: &mut GpuWeights,
        config: &Qwen2MoeConfig,
        dtype: DType,
        device: &GpuDevice,
    ) -> Result<Self> {
        let model = Qwen2MoeModel::load_fp8(weights, config, dtype, device)?;
        let lm_head = if config.tie_word_embeddings {
            Linear::new(model.embed_tokens.weight, None)
        } else {
            Linear::load(weights, "lm_head")?
        };
        Ok(Self { model, lm_head })
    }

    /// Load a quantized (AWQ/GPTQ → Marlin) Qwen2 MoE model.
    pub fn load_quantized(
        weights: &mut GpuWeights,
        config: &Qwen2MoeConfig,
        dtype: DType,
        qconfig: &QuantConfig,
        device: &GpuDevice,
    ) -> Result<Self> {
        let model = Qwen2MoeModel::load_quantized(weights, config, dtype, qconfig, device)?;
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

impl Qwen2MoeDecoderLayer {
    pub fn load_tp(
        weights: &mut GpuWeights,
        prefix: &str,
        config: &Qwen2MoeConfig,
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

        let is_dense = config.mlp_only_layers.contains(&layer_idx);
        let mlp = if is_dense {
            let dense = LlamaMLP::load_fused_tp(
                weights,
                &format!("{prefix}.mlp"),
                config.intermediate_size,
                tp,
                stream,
            )?;
            Qwen2MoeMlp::Dense(dense)
        } else {
            Self::load_moe(weights, &format!("{prefix}.mlp"), config, Some(tp), stream)?
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
            mlp,
            input_layernorm,
            post_attention_layernorm,
        })
    }
}

impl Qwen2MoeModel {
    pub fn load_tp(
        weights: &mut GpuWeights,
        config: &Qwen2MoeConfig,
        dtype: DType,
        tp: TpConfig,
        device: &GpuDevice,
    ) -> Result<Self> {
        let embed_tokens = crate::layers::Embedding::load(weights, "model.embed_tokens")?;

        let mut layers = Vec::with_capacity(config.num_hidden_layers);
        for i in 0..config.num_hidden_layers {
            layers.push(Qwen2MoeDecoderLayer::load_tp(
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

impl Qwen2MoeForCausalLM {
    pub fn load_tp(
        weights: &mut GpuWeights,
        config: &Qwen2MoeConfig,
        dtype: DType,
        tp: TpConfig,
        device: &GpuDevice,
    ) -> Result<Self> {
        let model = Qwen2MoeModel::load_tp(weights, config, dtype, tp, device)?;
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
            match &mut layer.mlp {
                Qwen2MoeMlp::Dense(mlp) => {
                    mlp.tp_group = Some(Arc::clone(&group));
                }
                Qwen2MoeMlp::MoE { moe, .. } => {
                    let FusedMoELayer::Dense(inner) = moe;
                    inner.tp_group = Some(Arc::clone(&group));
                }
                Qwen2MoeMlp::QuantizedMoE(layer) => {
                    layer.moe.tp_group = Some(Arc::clone(&group));
                }
                Qwen2MoeMlp::Fp8MoE { moe, .. } => {
                    moe.tp_group = Some(Arc::clone(&group));
                }
            }
        }
    }
}
