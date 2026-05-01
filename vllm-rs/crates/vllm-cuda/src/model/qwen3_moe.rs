// SPDX-License-Identifier: Apache-2.0
//! Qwen3 MoE model using `GpuTensor`.
//!
//! Differences from Qwen2 MoE:
//! - QKV bias (already handled by LlamaAttention::load_fused)
//! - QK-norm: per-head RMS norm on Q and K before RoPE
//!
//! The MoE layer is identical to Qwen2 MoE. Only the attention layer differs
//! (QK-norm weights loaded via `load_fused_with_qk_norm`).

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
    Fp8BlockFusedMoELayer, Fp8FusedMoELayer, FusedMoELayer, MarlinSharedFusedMoELayer,
};
use crate::model::llama::{LlamaAttention, LlamaMLP, RotaryCache, TpConfig};
#[cfg(feature = "nccl")]
use crate::nccl::NcclGroup;
use crate::quant::QuantConfig;
use crate::tensor::{GpuTensor, TensorView};
use crate::weights::{self as gpu_weights, GpuWeights};

pub use crate::model::qwen2_moe::Qwen2MoeConfig;

/// Qwen3 MoE config — same fields as Qwen2 MoE.
pub type Qwen3MoeConfig = Qwen2MoeConfig;

// ---------------------------------------------------------------------------
// MLP variants (reuse Qwen2 MoE logic)
// ---------------------------------------------------------------------------

#[allow(clippy::large_enum_variant)]
pub enum Qwen3MoeMlp {
    Dense(LlamaMLP),
    MoE {
        moe: FusedMoELayer,
        shared_gate_up: Option<Linear>,
        shared_down: Option<Linear>,
        shared_expert_gate: Option<Linear>,
        shared_intermediate_size: usize,
    },
    QuantizedMoE(MarlinSharedFusedMoELayer),
    Fp8MoE {
        moe: Fp8FusedMoELayer,
        shared_gate_up: Option<Linear>,
        shared_down: Option<Linear>,
        shared_expert_gate: Option<Linear>,
        shared_intermediate_size: usize,
    },
    Fp8BlockMoE {
        moe: Fp8BlockFusedMoELayer,
        shared_gate_up: Option<Linear>,
        shared_down: Option<Linear>,
        shared_expert_gate: Option<Linear>,
        shared_intermediate_size: usize,
    },
}

impl Qwen3MoeMlp {
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

                let moe_out = moe.forward(hidden_states, device);

                if let (Some(shared_gu_w), Some(shared_down_w), Some(shared_gate_w)) =
                    (shared_gate_up, shared_down, shared_expert_gate)
                {
                    let shared_gu = shared_gu_w.forward(
                        hidden_states,
                        &mut device.caching,
                        device.compute_stream,
                    );
                    let shared_activated = kernels::silu_and_mul_fused(
                        *shared_gu.view(),
                        *shared_intermediate_size,
                        &mut device.caching,
                        stream,
                    );
                    drop(shared_gu);

                    let shared_out = shared_down_w.forward(
                        shared_activated.view(),
                        &mut device.caching,
                        device.compute_stream,
                    );
                    drop(shared_activated);

                    let gate_logits = shared_gate_w.forward(
                        hidden_states,
                        &mut device.caching,
                        device.compute_stream,
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
                } else {
                    moe_out
                }
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

                if let (Some(shared_gu_w), Some(shared_down_w), Some(shared_gate_w)) =
                    (shared_gate_up, shared_down, shared_expert_gate)
                {
                    let shared_gu = shared_gu_w.forward(
                        hidden_states,
                        &mut device.caching,
                        device.compute_stream,
                    );
                    let shared_activated = kernels::silu_and_mul_fused(
                        *shared_gu.view(),
                        *shared_intermediate_size,
                        &mut device.caching,
                        stream,
                    );
                    drop(shared_gu);

                    let shared_out = shared_down_w.forward(
                        shared_activated.view(),
                        &mut device.caching,
                        device.compute_stream,
                    );
                    drop(shared_activated);

                    let gate_logits = shared_gate_w.forward(
                        hidden_states,
                        &mut device.caching,
                        device.compute_stream,
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
                } else {
                    moe_out
                }
            }
            Self::Fp8BlockMoE {
                moe,
                shared_gate_up,
                shared_down,
                shared_expert_gate,
                shared_intermediate_size,
            } => {
                let stream = device.compute_stream;
                let moe_out = moe.forward(hidden_states, device);

                if let (Some(shared_gu_w), Some(shared_down_w), Some(shared_gate_w)) =
                    (shared_gate_up, shared_down, shared_expert_gate)
                {
                    let shared_gu = shared_gu_w.forward(
                        hidden_states,
                        &mut device.caching,
                        device.compute_stream,
                    );
                    let shared_activated = kernels::silu_and_mul_fused(
                        *shared_gu.view(),
                        *shared_intermediate_size,
                        &mut device.caching,
                        stream,
                    );
                    drop(shared_gu);

                    let shared_out = shared_down_w.forward(
                        shared_activated.view(),
                        &mut device.caching,
                        device.compute_stream,
                    );
                    drop(shared_activated);

                    let gate_logits = shared_gate_w.forward(
                        hidden_states,
                        &mut device.caching,
                        device.compute_stream,
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
                } else {
                    moe_out
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Decoder layer
// ---------------------------------------------------------------------------

pub struct Qwen3MoeDecoderLayer {
    pub self_attn: LlamaAttention,
    pub mlp: Qwen3MoeMlp,
    pub input_layernorm: RmsNorm,
    pub post_attention_layernorm: RmsNorm,
}

impl Qwen3MoeDecoderLayer {
    pub fn load(
        weights: &mut GpuWeights,
        prefix: &str,
        config: &Qwen3MoeConfig,
        layer_idx: usize,
        stream: cudarc::driver::sys::CUstream,
    ) -> Result<Self> {
        let llama_cfg = config.as_llama_config();

        // Qwen3 MoE uses QK-norm on all attention layers.
        let self_attn = LlamaAttention::load_fused_with_qk_norm(
            weights,
            &format!("{prefix}.self_attn"),
            &llama_cfg,
            layer_idx,
            config.rms_norm_eps,
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
            Qwen3MoeMlp::Dense(dense)
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

    pub fn load_moe(
        weights: &mut GpuWeights,
        prefix: &str,
        config: &Qwen3MoeConfig,
        tp: Option<TpConfig>,
        stream: cudarc::driver::sys::CUstream,
    ) -> Result<Qwen3MoeMlp> {
        let num_experts = config.num_experts;
        let inter = config.moe_intermediate_size;
        let hidden = config.hidden_size;
        let (rank, world_size) = tp.map_or((0, 1), |t| (t.rank, t.world_size));
        let ipp = inter / world_size;

        let gate = Linear::load(weights, &format!("{prefix}.gate"))?;

        let first_gate = format!("{prefix}.experts.0.gate_proj.weight");
        let (_, dtype) = weights
            .tensor_info(&first_gate)
            .ok_or_else(|| anyhow::anyhow!("weight not found: {first_gate}"))?;
        let elem = dtype.size_bytes();

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

        let moe = FusedMoELayer {
            gate,
            w1,
            w2,
            num_experts,
            top_k: config.num_experts_per_tok,
            intermediate_size: ipp,
            hidden_size: hidden,
            renormalize: true,
            e_score_correction_bias: None,
            n_expert_group: 0,
            topk_group: 0,
            routed_scaling_factor: 1.0,
            #[cfg(feature = "nccl")]
            tp_group: None,
        };

        let shared_inter = config.shared_expert_intermediate_size;
        let sipp = shared_inter / world_size;

        // Shared expert is optional (shared_expert_intermediate_size == 0 means
        // no shared expert, matching Python's Qwen3NextSparseMoeBlock).
        let (shared_gate_up, shared_down, shared_expert_gate) = if shared_inter > 0 {
            let gate_up = if world_size > 1 {
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

            let down = if world_size > 1 {
                let name = format!("{prefix}.shared_expert.down_proj.weight");
                let w = weights.take_shard(&name, 1, rank, world_size)?;
                Linear::new(w, None)
            } else {
                Linear::load(weights, &format!("{prefix}.shared_expert.down_proj"))?
            };

            let gate = Linear::load(weights, &format!("{prefix}.shared_expert_gate"))?;

            (Some(gate_up), Some(down), Some(gate))
        } else {
            (None, None, None)
        };

        Ok(Qwen3MoeMlp::MoE {
            moe,
            shared_gate_up,
            shared_down,
            shared_expert_gate,
            shared_intermediate_size: sipp,
        })
    }

    /// Load an FP8 quantized decoder layer.
    /// Only attention and dense MLP layers are FP8. MoE expert weights remain dense.
    /// QK-norm weights are loaded separately (they are tiny and stay dense).
    pub fn load_fp8(
        weights: &mut GpuWeights,
        prefix: &str,
        config: &Qwen3MoeConfig,
        layer_idx: usize,
        dtype: DType,
        stream: cudarc::driver::sys::CUstream,
    ) -> Result<Self> {
        let llama_cfg = config.as_llama_config();

        // Load FP8 attention with QK-norm weights.
        let self_attn = LlamaAttention::load_fp8(
            weights,
            &format!("{prefix}.self_attn"),
            &llama_cfg,
            layer_idx,
            dtype,
            config.rms_norm_eps,
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
            Qwen3MoeMlp::Dense(dense)
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
                    true, // Qwen3 MoE renormalizes
                    "gate_proj",
                    "up_proj",
                    "down_proj",
                )?;

                // Load shared expert (dense BF16).
                let shared_inter = config.shared_expert_intermediate_size;
                let (shared_gate_up, shared_down, shared_expert_gate) = if shared_inter > 0 {
                    let gate_up = {
                        let gate_name = format!("{moe_prefix}.shared_expert.gate_proj.weight");
                        let up_name = format!("{moe_prefix}.shared_expert.up_proj.weight");
                        let (_, se_dtype) = weights
                            .tensor_info(&gate_name)
                            .ok_or_else(|| anyhow::anyhow!("weight not found: {gate_name}"))?;
                        let se_elem = se_dtype.size_bytes();
                        let hidden = config.hidden_size;
                        let gate_proj_bytes = shared_inter * hidden * se_elem;
                        let total = 2 * gate_proj_bytes;
                        let ptr = unsafe { crate::driver::mem_alloc(total)? };
                        unsafe {
                            weights.take_into(&gate_name, ptr, stream)?;
                            weights.take_into(&up_name, ptr.add(gate_proj_bytes), stream)?;
                        }
                        let w =
                            unsafe { GpuTensor::new(ptr, &[2 * shared_inter, hidden], se_dtype) };
                        Linear::new(w, None)
                    };
                    let down =
                        Linear::load(weights, &format!("{moe_prefix}.shared_expert.down_proj"))?;
                    let gate = Linear::load(weights, &format!("{moe_prefix}.shared_expert_gate"))?;
                    (Some(gate_up), Some(down), Some(gate))
                } else {
                    (None, None, None)
                };

                Qwen3MoeMlp::Fp8MoE {
                    moe: fp8_moe,
                    shared_gate_up,
                    shared_down,
                    shared_expert_gate,
                    shared_intermediate_size: shared_inter,
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

    /// Load an FP8 block-quantized decoder layer.
    /// Attention and dense MLP use block FP8 (weight_scale_inv with 2D scales).
    /// MoE experts use block-scale-aware FP8 MoE kernel.
    /// With TP, expert intermediate dims are sharded across ranks.
    pub fn load_fp8_block(
        weights: &mut GpuWeights,
        prefix: &str,
        config: &Qwen3MoeConfig,
        layer_idx: usize,
        dtype: DType,
        tp: Option<TpConfig>,
        stream: cudarc::driver::sys::CUstream,
    ) -> Result<Self> {
        let llama_cfg = config.as_llama_config();

        // Load block-FP8 attention with QK-norm weights.
        let self_attn = if let Some(tp_cfg) = tp {
            LlamaAttention::load_fp8_block_tp(
                weights,
                &format!("{prefix}.self_attn"),
                &llama_cfg,
                layer_idx,
                dtype,
                config.rms_norm_eps,
                tp_cfg,
                stream,
            )?
        } else {
            LlamaAttention::load_fp8_block(
                weights,
                &format!("{prefix}.self_attn"),
                &llama_cfg,
                layer_idx,
                dtype,
                config.rms_norm_eps,
                stream,
            )?
        };

        let is_dense = config.mlp_only_layers.contains(&layer_idx);
        let mlp = if is_dense {
            let dense = if let Some(tp_cfg) = tp {
                LlamaMLP::load_fp8_block_tp(
                    weights,
                    &format!("{prefix}.mlp"),
                    config.intermediate_size,
                    dtype,
                    tp_cfg,
                    stream,
                )?
            } else {
                LlamaMLP::load_fp8_block(
                    weights,
                    &format!("{prefix}.mlp"),
                    config.intermediate_size,
                    dtype,
                    stream,
                )?
            };
            Qwen3MoeMlp::Dense(dense)
        } else {
            let moe_prefix = format!("{prefix}.mlp");
            let first_gate_name = format!("{moe_prefix}.experts.0.gate_proj.weight");
            let is_fp8 = weights
                .tensor_info(&first_gate_name)
                .map(|(_, dt)| dt == DType::Fp8E4m3)
                .unwrap_or(false);

            if is_fp8 {
                // Check if experts have block scales (weight_scale_inv) or per-tensor (weight_scale).
                let first_block_scale =
                    format!("{moe_prefix}.experts.0.gate_proj.weight_scale_inv");
                let is_block = weights.contains(&first_block_scale);

                if is_block {
                    // Block-quantized experts: keep FP8 with 3D block scales.
                    let moe = gpu_weights::load_fp8_block_moe_experts(
                        weights,
                        &moe_prefix,
                        config.num_experts,
                        config.moe_intermediate_size,
                        config.hidden_size,
                        config.num_experts_per_tok,
                        true, // Qwen3 MoE renormalizes
                        "gate_proj",
                        "up_proj",
                        "down_proj",
                        tp,
                    )?;

                    // Load shared expert (dense BF16).
                    let shared_inter = config.shared_expert_intermediate_size;
                    let (shared_gate_up, shared_down, shared_expert_gate) = if shared_inter > 0 {
                        let gate_up = {
                            let gate_name = format!("{moe_prefix}.shared_expert.gate_proj.weight");
                            let up_name = format!("{moe_prefix}.shared_expert.up_proj.weight");
                            let (_, se_dtype) = weights
                                .tensor_info(&gate_name)
                                .ok_or_else(|| anyhow::anyhow!("weight not found: {gate_name}"))?;
                            let se_elem = se_dtype.size_bytes();
                            let hidden = config.hidden_size;
                            let gate_proj_bytes = shared_inter * hidden * se_elem;
                            let total = 2 * gate_proj_bytes;
                            let ptr = unsafe { crate::driver::mem_alloc(total)? };
                            unsafe {
                                weights.take_into(&gate_name, ptr, stream)?;
                                weights.take_into(&up_name, ptr.add(gate_proj_bytes), stream)?;
                            }
                            let w = unsafe {
                                GpuTensor::new(ptr, &[2 * shared_inter, hidden], se_dtype)
                            };
                            Linear::new(w, None)
                        };
                        let down = Linear::load(
                            weights,
                            &format!("{moe_prefix}.shared_expert.down_proj"),
                        )?;
                        let gate =
                            Linear::load(weights, &format!("{moe_prefix}.shared_expert_gate"))?;
                        (Some(gate_up), Some(down), Some(gate))
                    } else {
                        (None, None, None)
                    };

                    Qwen3MoeMlp::Fp8BlockMoE {
                        moe,
                        shared_gate_up,
                        shared_down,
                        shared_expert_gate,
                        shared_intermediate_size: shared_inter,
                    }
                } else {
                    // Per-tensor FP8 experts: use existing FP8 MoE path.
                    let fp8_moe = gpu_weights::load_fp8_moe_experts(
                        weights,
                        &moe_prefix,
                        config.num_experts,
                        config.moe_intermediate_size,
                        config.hidden_size,
                        config.num_experts_per_tok,
                        true,
                        "gate_proj",
                        "up_proj",
                        "down_proj",
                    )?;

                    // Load shared expert (dense BF16).
                    let shared_inter = config.shared_expert_intermediate_size;
                    let (shared_gate_up, shared_down, shared_expert_gate) = if shared_inter > 0 {
                        let gate_up = {
                            let gate_name = format!("{moe_prefix}.shared_expert.gate_proj.weight");
                            let up_name = format!("{moe_prefix}.shared_expert.up_proj.weight");
                            let (_, se_dtype) = weights
                                .tensor_info(&gate_name)
                                .ok_or_else(|| anyhow::anyhow!("weight not found: {gate_name}"))?;
                            let se_elem = se_dtype.size_bytes();
                            let hidden = config.hidden_size;
                            let gate_proj_bytes = shared_inter * hidden * se_elem;
                            let total = 2 * gate_proj_bytes;
                            let ptr = unsafe { crate::driver::mem_alloc(total)? };
                            unsafe {
                                weights.take_into(&gate_name, ptr, stream)?;
                                weights.take_into(&up_name, ptr.add(gate_proj_bytes), stream)?;
                            }
                            let w = unsafe {
                                GpuTensor::new(ptr, &[2 * shared_inter, hidden], se_dtype)
                            };
                            Linear::new(w, None)
                        };
                        let down = Linear::load(
                            weights,
                            &format!("{moe_prefix}.shared_expert.down_proj"),
                        )?;
                        let gate =
                            Linear::load(weights, &format!("{moe_prefix}.shared_expert_gate"))?;
                        (Some(gate_up), Some(down), Some(gate))
                    } else {
                        (None, None, None)
                    };

                    Qwen3MoeMlp::Fp8MoE {
                        moe: fp8_moe,
                        shared_gate_up,
                        shared_down,
                        shared_expert_gate,
                        shared_intermediate_size: shared_inter,
                    }
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
    /// Attention uses Marlin linear layers. MoE uses MarlinFusedMoELayer.
    /// Shared experts (if any) use Marlin linear layers.
    #[allow(clippy::too_many_arguments)]
    pub fn load_quantized(
        weights: &mut GpuWeights,
        prefix: &str,
        config: &Qwen3MoeConfig,
        layer_idx: usize,
        qconfig: &QuantConfig,
        workspace: GpuTensor,
        device: &GpuDevice,
    ) -> Result<Self> {
        let llama_cfg = config.as_llama_config();

        // Load quantized attention, then add QK-norm weights on top.
        let mut self_attn = LlamaAttention::load_quantized(
            weights,
            &format!("{prefix}.self_attn"),
            &llama_cfg,
            layer_idx,
            qconfig,
            workspace,
            device,
        )?;

        let q_norm_name = format!("{prefix}.self_attn.q_norm.weight");
        let k_norm_name = format!("{prefix}.self_attn.k_norm.weight");
        if weights.contains(&q_norm_name) {
            self_attn.q_norm_weight = Some(weights.take(&q_norm_name)?);
        }
        if weights.contains(&k_norm_name) {
            self_attn.k_norm_weight = Some(weights.take(&k_norm_name)?);
        }
        self_attn.qk_norm_eps = config.rms_norm_eps;

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
            Qwen3MoeMlp::Dense(dense)
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

    /// Load quantized MoE layer (experts + optional shared expert).
    #[allow(clippy::too_many_arguments)]
    fn load_quantized_moe(
        weights: &mut GpuWeights,
        prefix: &str,
        config: &Qwen3MoeConfig,
        qconfig: &QuantConfig,
        workspace: GpuTensor,
        device: &GpuDevice,
    ) -> Result<Qwen3MoeMlp> {
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

        // Load shared expert as Marlin linear layers (if present).
        let (shared_gate_up, shared_down, shared_expert_gate) = if shared_inter > 0 {
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

            (
                Some(LinearLayer::Marlin(Box::new(gate_up))),
                Some(LinearLayer::Marlin(Box::new(down))),
                Some(gate),
            )
        } else {
            (None, None, None)
        };

        Ok(Qwen3MoeMlp::QuantizedMoE(MarlinSharedFusedMoELayer {
            moe,
            shared_gate_up,
            shared_down,
            shared_expert_gate,
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
// Qwen3MoeForCausalLM
// ---------------------------------------------------------------------------

pub struct Qwen3MoeModel {
    pub embed_tokens: crate::layers::Embedding,
    pub layers: Vec<Qwen3MoeDecoderLayer>,
    pub norm: RmsNorm,
    pub rotary: RotaryCache,
}

impl Qwen3MoeModel {
    pub fn load(
        weights: &mut GpuWeights,
        config: &Qwen3MoeConfig,
        dtype: DType,
        device: &GpuDevice,
    ) -> Result<Self> {
        let embed_tokens = crate::layers::Embedding::load(weights, "model.embed_tokens")?;

        let mut layers = Vec::with_capacity(config.num_hidden_layers);
        for i in 0..config.num_hidden_layers {
            layers.push(Qwen3MoeDecoderLayer::load(
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

    /// Load an FP8 quantized Qwen3 MoE model.
    /// Only attention and dense MLP layers are FP8. MoE expert weights remain dense.
    pub fn load_fp8(
        weights: &mut GpuWeights,
        config: &Qwen3MoeConfig,
        dtype: DType,
        device: &GpuDevice,
    ) -> Result<Self> {
        let embed_tokens = crate::layers::Embedding::load(weights, "model.embed_tokens")?;

        let mut layers = Vec::with_capacity(config.num_hidden_layers);
        for i in 0..config.num_hidden_layers {
            layers.push(Qwen3MoeDecoderLayer::load_fp8(
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

    /// Load an FP8 block-quantized Qwen3 MoE model.
    pub fn load_fp8_block(
        weights: &mut GpuWeights,
        config: &Qwen3MoeConfig,
        dtype: DType,
        tp: Option<TpConfig>,
        device: &GpuDevice,
    ) -> Result<Self> {
        let embed_tokens = crate::layers::Embedding::load(weights, "model.embed_tokens")?;

        let mut layers = Vec::with_capacity(config.num_hidden_layers);
        for i in 0..config.num_hidden_layers {
            layers.push(Qwen3MoeDecoderLayer::load_fp8_block(
                weights,
                &format!("model.layers.{i}"),
                config,
                i,
                dtype,
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

    /// Load a quantized (AWQ/GPTQ → Marlin) Qwen3 MoE model.
    pub fn load_quantized(
        weights: &mut GpuWeights,
        config: &Qwen3MoeConfig,
        dtype: DType,
        qconfig: &QuantConfig,
        device: &GpuDevice,
    ) -> Result<Self> {
        let workspace = gpu_weights::alloc_marlin_workspace(device.num_sm, device.compute_stream)?;

        let embed_tokens = crate::layers::Embedding::load(weights, "model.embed_tokens")?;

        let mut layers = Vec::with_capacity(config.num_hidden_layers);
        for i in 0..config.num_hidden_layers {
            layers.push(Qwen3MoeDecoderLayer::load_quantized(
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

pub struct Qwen3MoeForCausalLM {
    pub model: Qwen3MoeModel,
    pub lm_head: Linear,
}

impl Qwen3MoeForCausalLM {
    pub fn load(
        weights: &mut GpuWeights,
        config: &Qwen3MoeConfig,
        dtype: DType,
        device: &GpuDevice,
    ) -> Result<Self> {
        let model = Qwen3MoeModel::load(weights, config, dtype, device)?;
        let lm_head = if config.tie_word_embeddings {
            Linear::new(model.embed_tokens.weight, None)
        } else {
            Linear::load(weights, "lm_head")?
        };
        Ok(Self { model, lm_head })
    }

    /// Load an FP8 quantized Qwen3 MoE model.
    /// Only attention and dense MLP layers are FP8. MoE expert weights remain dense.
    pub fn load_fp8(
        weights: &mut GpuWeights,
        config: &Qwen3MoeConfig,
        dtype: DType,
        device: &GpuDevice,
    ) -> Result<Self> {
        let model = Qwen3MoeModel::load_fp8(weights, config, dtype, device)?;
        let lm_head = if config.tie_word_embeddings {
            Linear::new(model.embed_tokens.weight, None)
        } else {
            Linear::load(weights, "lm_head")?
        };
        Ok(Self { model, lm_head })
    }

    /// Load an FP8 block-quantized Qwen3 MoE model.
    pub fn load_fp8_block(
        weights: &mut GpuWeights,
        config: &Qwen3MoeConfig,
        dtype: DType,
        tp: Option<TpConfig>,
        device: &GpuDevice,
    ) -> Result<Self> {
        let model = Qwen3MoeModel::load_fp8_block(weights, config, dtype, tp, device)?;
        let lm_head = if config.tie_word_embeddings {
            Linear::new(model.embed_tokens.weight, None)
        } else {
            Linear::load(weights, "lm_head")?
        };
        Ok(Self { model, lm_head })
    }

    /// Load a quantized (AWQ/GPTQ → Marlin) Qwen3 MoE model.
    pub fn load_quantized(
        weights: &mut GpuWeights,
        config: &Qwen3MoeConfig,
        dtype: DType,
        qconfig: &QuantConfig,
        device: &GpuDevice,
    ) -> Result<Self> {
        let model = Qwen3MoeModel::load_quantized(weights, config, dtype, qconfig, device)?;
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
            &mut device.caching,
            device.compute_stream,
        )
    }
}

// ---------------------------------------------------------------------------
// Tensor Parallelism
// ---------------------------------------------------------------------------

impl Qwen3MoeDecoderLayer {
    pub fn load_tp(
        weights: &mut GpuWeights,
        prefix: &str,
        config: &Qwen3MoeConfig,
        layer_idx: usize,
        tp: TpConfig,
        stream: cudarc::driver::sys::CUstream,
    ) -> Result<Self> {
        let llama_cfg = config.as_llama_config();

        let self_attn = LlamaAttention::load_fused_with_qk_norm_tp(
            weights,
            &format!("{prefix}.self_attn"),
            &llama_cfg,
            layer_idx,
            config.rms_norm_eps,
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
            Qwen3MoeMlp::Dense(dense)
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

impl Qwen3MoeModel {
    pub fn load_tp(
        weights: &mut GpuWeights,
        config: &Qwen3MoeConfig,
        dtype: DType,
        tp: TpConfig,
        device: &GpuDevice,
    ) -> Result<Self> {
        let embed_tokens = crate::layers::Embedding::load(weights, "model.embed_tokens")?;

        let mut layers = Vec::with_capacity(config.num_hidden_layers);
        for i in 0..config.num_hidden_layers {
            layers.push(Qwen3MoeDecoderLayer::load_tp(
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

impl Qwen3MoeForCausalLM {
    pub fn load_tp(
        weights: &mut GpuWeights,
        config: &Qwen3MoeConfig,
        dtype: DType,
        tp: TpConfig,
        device: &GpuDevice,
    ) -> Result<Self> {
        let model = Qwen3MoeModel::load_tp(weights, config, dtype, tp, device)?;
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
                Qwen3MoeMlp::Dense(mlp) => {
                    mlp.tp_group = Some(Arc::clone(&group));
                }
                Qwen3MoeMlp::MoE { moe, .. } => {
                    moe.tp_group = Some(Arc::clone(&group));
                }
                Qwen3MoeMlp::QuantizedMoE(layer) => {
                    layer.moe.tp_group = Some(Arc::clone(&group));
                }
                Qwen3MoeMlp::Fp8MoE { moe, .. } => {
                    moe.tp_group = Some(Arc::clone(&group));
                }
                Qwen3MoeMlp::Fp8BlockMoE { moe, .. } => {
                    moe.tp_group = Some(Arc::clone(&group));
                }
            }
        }
    }
}
