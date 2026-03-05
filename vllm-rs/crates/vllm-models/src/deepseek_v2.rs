// SPDX-License-Identifier: Apache-2.0
//! DeepSeek V2/V3 model architecture.
//!
//! Implements:
//! - `DeepSeekV2ForCausalLM` — top-level model with lm_head
//! - `DeepSeekV2Model` — transformer backbone
//! - `DeepSeekV2DecoderLayer` — single decoder layer (MLA attention + MLP/MoE)
//! - `DeepSeekV2Attention` — Multi-head Latent Attention (MLA)
//! - `DeepSeekV2MoE` — Mixture of Experts with shared experts
//!
//! Novel components:
//! - **MLA**: compresses KV into a low-rank latent space via two-stage projections
//! - **MoE**: routes tokens to top-k experts via a gating network + shared experts
//! - **YaRN RoPE**: frequency-corrected rotary embeddings for long contexts
//!
//! Port of: `vllm/model_executor/models/deepseek_v2.py`

use candle_core::{DType, Device, Module, Tensor};

use vllm_model::error::{ModelError, ModelResult};
use vllm_model::layers::{
    ColumnParallelLinear, Embedding, Linear, RmsNorm, RotaryEmbedding, RowParallelLinear,
};
use vllm_model::lora::LoraAdapter;
use vllm_model::weight::{HfModelConfig, ModelWeights};

use crate::attention::attention_with_cache;
use crate::llama::LlamaMLP;

// ---------------------------------------------------------------------------
// DeepSeekV2Config
// ---------------------------------------------------------------------------

/// Parsed configuration for a DeepSeek V2/V3 model.
#[derive(Debug, Clone)]
pub struct DeepSeekV2Config {
    pub hidden_size: usize,
    pub num_attention_heads: usize,
    pub num_kv_heads: usize,
    pub num_hidden_layers: usize,
    pub intermediate_size: usize,
    pub vocab_size: usize,
    pub max_position_embeddings: usize,
    pub rms_norm_eps: f64,
    pub rope_theta: f64,
    pub tie_word_embeddings: bool,

    // MLA-specific.
    pub qk_nope_head_dim: usize,
    pub qk_rope_head_dim: usize,
    pub v_head_dim: usize,
    pub q_lora_rank: Option<usize>,
    pub kv_lora_rank: usize,

    // MoE-specific.
    pub n_routed_experts: usize,
    pub n_shared_experts: usize,
    pub num_experts_per_tok: usize,
    pub first_k_dense_replace: usize,
    pub moe_intermediate_size: usize,
    pub norm_topk_prob: bool,
    pub routed_scaling_factor: f64,

    // YaRN RoPE.
    pub rope_scaling: Option<YarnConfig>,
}

/// YaRN rope_scaling parameters.
#[derive(Debug, Clone)]
pub struct YarnConfig {
    pub factor: f64,
    pub beta_fast: f64,
    pub beta_slow: f64,
    pub mscale_all_dim: f64,
    pub original_max_position_embeddings: usize,
}

impl DeepSeekV2Config {
    /// Parse from a HuggingFace config.json.
    pub fn from_hf_config(config: &HfModelConfig) -> ModelResult<Self> {
        let hidden_size = config
            .hidden_size
            .ok_or_else(|| ModelError::Other("missing hidden_size".into()))?;
        let num_attention_heads = config
            .num_attention_heads
            .ok_or_else(|| ModelError::Other("missing num_attention_heads".into()))?;

        let extra = &config.extra;

        let get_usize = |key: &str| -> Option<usize> {
            extra.get(key).and_then(|v| v.as_u64()).map(|v| v as usize)
        };
        let get_f64 = |key: &str| -> Option<f64> { extra.get(key).and_then(|v| v.as_f64()) };
        let get_bool = |key: &str| -> Option<bool> { extra.get(key).and_then(|v| v.as_bool()) };

        // MLA dimensions.
        let qk_nope_head_dim = get_usize("qk_nope_head_dim").unwrap_or(128);
        let qk_rope_head_dim = get_usize("qk_rope_head_dim").unwrap_or(64);
        let v_head_dim = get_usize("v_head_dim").unwrap_or(128);
        let q_lora_rank = get_usize("q_lora_rank");
        let kv_lora_rank = get_usize("kv_lora_rank").unwrap_or(512);

        // MoE.
        let n_routed_experts = get_usize("n_routed_experts").unwrap_or(0);
        let n_shared_experts = get_usize("n_shared_experts").unwrap_or(0);
        let num_experts_per_tok = get_usize("num_experts_per_tok").unwrap_or(6);
        let first_k_dense_replace = get_usize("first_k_dense_replace").unwrap_or(1);
        let moe_intermediate_size =
            get_usize("moe_intermediate_size").unwrap_or(config.intermediate_size.unwrap_or(11008));
        let norm_topk_prob = get_bool("norm_topk_prob").unwrap_or(true);
        let routed_scaling_factor = get_f64("routed_scaling_factor").unwrap_or(1.0);

        // YaRN rope_scaling.
        let rope_scaling = extra.get("rope_scaling").and_then(|rs| {
            let obj = rs.as_object()?;
            Some(YarnConfig {
                factor: obj.get("factor")?.as_f64()?,
                beta_fast: obj.get("beta_fast")?.as_f64().unwrap_or(32.0),
                beta_slow: obj.get("beta_slow")?.as_f64().unwrap_or(1.0),
                mscale_all_dim: obj
                    .get("mscale_all_dim")
                    .and_then(|v| v.as_f64())
                    .unwrap_or(0.0),
                original_max_position_embeddings: obj
                    .get("original_max_position_embeddings")
                    .and_then(|v| v.as_u64())
                    .unwrap_or(4096) as usize,
            })
        });

        Ok(Self {
            hidden_size,
            num_attention_heads,
            num_kv_heads: config.num_kv_heads().unwrap_or(num_attention_heads),
            num_hidden_layers: config
                .num_hidden_layers
                .ok_or_else(|| ModelError::Other("missing num_hidden_layers".into()))?,
            intermediate_size: config
                .intermediate_size
                .ok_or_else(|| ModelError::Other("missing intermediate_size".into()))?,
            vocab_size: config
                .vocab_size
                .ok_or_else(|| ModelError::Other("missing vocab_size".into()))?,
            max_position_embeddings: config.max_position_embeddings.unwrap_or(4096),
            rms_norm_eps: config.norm_eps(),
            rope_theta: config.rope_theta.unwrap_or(10000.0),
            tie_word_embeddings: config.tie_word_embeddings.unwrap_or(false),
            qk_nope_head_dim,
            qk_rope_head_dim,
            v_head_dim,
            q_lora_rank,
            kv_lora_rank,
            n_routed_experts,
            n_shared_experts,
            num_experts_per_tok,
            first_k_dense_replace,
            moe_intermediate_size,
            norm_topk_prob,
            routed_scaling_factor,
            rope_scaling,
        })
    }

    /// Total Q/K head dimension (nope + rope parts).
    pub fn qk_head_dim(&self) -> usize {
        self.qk_nope_head_dim + self.qk_rope_head_dim
    }

    /// Whether this model uses MoE layers.
    pub fn has_moe(&self) -> bool {
        self.n_routed_experts > 0
    }
}

// ---------------------------------------------------------------------------
// DeepSeekV2MoE
// ---------------------------------------------------------------------------

/// Mixture of Experts layer with optional shared experts.
///
/// Routes each token to the top-k experts via a gating network, computes
/// each expert's MLP, then combines via weighted sum. Shared experts
/// contribute to all tokens unconditionally.
pub struct DeepSeekV2MoE {
    gate: Linear,
    experts: Vec<LlamaMLP>,
    shared_experts: Option<LlamaMLP>,
    top_k: usize,
    norm_topk_prob: bool,
    routed_scaling_factor: f64,
}

impl DeepSeekV2MoE {
    /// Load MoE weights.
    pub fn load(
        weights: &mut ModelWeights,
        prefix: &str,
        config: &DeepSeekV2Config,
        dtype: DType,
    ) -> ModelResult<Self> {
        let n = config.n_routed_experts;

        // Gate: hidden_size → n_routed_experts (no bias).
        let gate = Linear::load(weights, &format!("{prefix}.gate"), dtype)?;

        // Individual experts.
        let mut experts = Vec::with_capacity(n);
        for i in 0..n {
            let expert = LlamaMLP::load(weights, &format!("{prefix}.experts.{i}"), dtype, 0, 1)?;
            experts.push(expert);
        }

        // Shared experts.
        let shared_experts = if config.n_shared_experts > 0 {
            Some(LlamaMLP::load(
                weights,
                &format!("{prefix}.shared_experts"),
                dtype,
                0,
                1,
            )?)
        } else {
            None
        };

        Ok(Self {
            gate,
            experts,
            shared_experts,
            top_k: config.num_experts_per_tok,
            norm_topk_prob: config.norm_topk_prob,
            routed_scaling_factor: config.routed_scaling_factor,
        })
    }

    /// Create with zero weights (for testing).
    pub fn zeros(config: &DeepSeekV2Config, dtype: DType, device: &Device) -> ModelResult<Self> {
        let n = config.n_routed_experts;
        let gate = Linear::zeros(config.hidden_size, n, dtype, device)?;

        let mut experts = Vec::with_capacity(n);
        for _ in 0..n {
            experts.push(LlamaMLP::zeros(
                config.hidden_size,
                config.moe_intermediate_size,
                dtype,
                device,
            )?);
        }

        let shared_experts = if config.n_shared_experts > 0 {
            Some(LlamaMLP::zeros(
                config.hidden_size,
                config.moe_intermediate_size * config.n_shared_experts,
                dtype,
                device,
            )?)
        } else {
            None
        };

        Ok(Self {
            gate,
            experts,
            shared_experts,
            top_k: config.num_experts_per_tok,
            norm_topk_prob: config.norm_topk_prob,
            routed_scaling_factor: config.routed_scaling_factor,
        })
    }
}

impl Module for DeepSeekV2MoE {
    fn forward(&self, x: &Tensor) -> candle_core::Result<Tensor> {
        let (num_tokens, hidden_size) = x.dims2()?;

        // Compute router logits.
        let router_logits = self.gate.forward(x)?; // [num_tokens, n_experts]

        // GPU-accelerated top-k softmax gating (CUDA kernel on GPU, CPU fallback).
        let (topk_weights, topk_ids) =
            crate::ops::topk_softmax(&router_logits, self.top_k, self.norm_topk_prob)?;

        // Read routing decisions back to CPU (small: [num_tokens, top_k]).
        let ids = topk_ids.to_vec2::<u32>()?;
        let weights = topk_weights.to_vec2::<f32>()?;

        // Group tokens by expert for batched forward.
        let num_experts = self.experts.len();
        let mut expert_token_map: Vec<Vec<(usize, usize)>> = vec![vec![]; num_experts];
        for (tok, tok_ids) in ids.iter().enumerate() {
            for (k_idx, &eid) in tok_ids.iter().enumerate() {
                let eid = eid as usize;
                if eid < num_experts {
                    expert_token_map[eid].push((tok, k_idx));
                }
            }
        }

        // Build output by accumulating weighted expert contributions.
        let device = x.device();
        let dtype = x.dtype();
        let mut token_accum = vec![Tensor::zeros((1, hidden_size), dtype, device)?; num_tokens];

        for (eid, token_k_pairs) in expert_token_map.iter().enumerate() {
            if token_k_pairs.is_empty() {
                continue;
            }

            // Gather input tokens for this expert.
            let token_rows: Vec<Tensor> = token_k_pairs
                .iter()
                .map(|&(tok, _)| x.narrow(0, tok, 1))
                .collect::<candle_core::Result<Vec<_>>>()?;
            let x_batch = Tensor::cat(&token_rows, 0)?; // [batch, hidden]

            // Batched expert MLP forward (one cuBLAS matmul per projection).
            let expert_out = self.experts[eid].forward(&x_batch)?; // [batch, hidden]

            // Scale each output by routing weight * routed_scaling_factor and accumulate.
            for (batch_idx, &(tok, k_idx)) in token_k_pairs.iter().enumerate() {
                let row = expert_out.narrow(0, batch_idx, 1)?; // [1, hidden]
                let w = weights[tok][k_idx] as f64 * self.routed_scaling_factor;
                let weighted = (row * w)?;
                token_accum[tok] = (&token_accum[tok] + &weighted)?;
            }
        }

        let token_refs: Vec<&Tensor> = token_accum.iter().collect();
        let mut output = Tensor::cat(&token_refs, 0)?; // [num_tokens, hidden]

        // Add shared expert contribution.
        if let Some(ref shared) = self.shared_experts {
            let shared_out = shared.forward(x)?;
            output = (output + shared_out)?;
        }

        Ok(output)
    }
}

// ---------------------------------------------------------------------------
// DeepSeekV2Attention (MLA)
// ---------------------------------------------------------------------------

/// DeepSeek V2 Multi-head Latent Attention.
///
/// MLA compresses K/V into a low-rank latent space:
/// 1. Q path: hidden → q_a_proj → RMSNorm → q_b_proj (splits into nope + rope)
/// 2. KV path: hidden → kv_a_proj_with_mqa → split → [latent | k_pe]
///    latent → RMSNorm → kv_b_proj → split → [k_nope | v]
/// 3. Apply RoPE to q_pe and k_pe, assemble full Q and K
/// 4. Run standard attention, slice V back from padded dimension
pub struct DeepSeekV2Attention {
    // Q path.
    q_a_proj: Option<Linear>,
    q_a_layernorm: Option<RmsNorm>,
    q_b_proj: Option<ColumnParallelLinear>,
    q_proj: Option<ColumnParallelLinear>,

    // KV path.
    kv_a_proj_with_mqa: Linear,
    kv_a_layernorm: RmsNorm,
    kv_b_proj: ColumnParallelLinear,

    // Output.
    o_proj: RowParallelLinear,

    // RoPE for the rope dimensions only.
    rotary_emb: RotaryEmbedding,

    // Dimensions.
    num_heads: usize,
    qk_nope_head_dim: usize,
    qk_rope_head_dim: usize,
    qk_head_dim: usize,
    v_head_dim: usize,
    kv_lora_rank: usize,
    scale: f64,
}

impl DeepSeekV2Attention {
    /// Load attention weights.
    pub fn load(
        weights: &mut ModelWeights,
        prefix: &str,
        config: &DeepSeekV2Config,
        dtype: DType,
        device: &Device,
        rank: usize,
        world_size: usize,
    ) -> ModelResult<Self> {
        let num_heads = config.num_attention_heads / world_size;
        let qk_nope_head_dim = config.qk_nope_head_dim;
        let qk_rope_head_dim = config.qk_rope_head_dim;
        let v_head_dim = config.v_head_dim;
        let kv_lora_rank = config.kv_lora_rank;
        let qk_head_dim = qk_nope_head_dim + qk_rope_head_dim;

        // Q path: q_a_proj is replicated (low-rank bottleneck), q_b_proj is column-parallel.
        let (q_a_proj, q_a_layernorm, q_b_proj, q_proj) =
            if let Some(q_lora_rank) = config.q_lora_rank {
                let q_a = Linear::load(weights, &format!("{prefix}.q_a_proj"), dtype)?;
                let q_a_ln = RmsNorm::load(
                    weights,
                    &format!("{prefix}.q_a_layernorm"),
                    config.rms_norm_eps,
                    dtype,
                )?;
                let q_b = ColumnParallelLinear::load(
                    weights,
                    &format!("{prefix}.q_b_proj"),
                    dtype,
                    rank,
                    world_size,
                    false,
                )?;
                let _ = q_lora_rank;
                (Some(q_a), Some(q_a_ln), Some(q_b), None)
            } else {
                let q = ColumnParallelLinear::load(
                    weights,
                    &format!("{prefix}.q_proj"),
                    dtype,
                    rank,
                    world_size,
                    false,
                )?;
                (None, None, None, Some(q))
            };

        // KV path: kv_a_proj_with_mqa is replicated, kv_b_proj is column-parallel.
        let kv_a_proj_with_mqa =
            Linear::load(weights, &format!("{prefix}.kv_a_proj_with_mqa"), dtype)?;
        let kv_a_layernorm = RmsNorm::load(
            weights,
            &format!("{prefix}.kv_a_layernorm"),
            config.rms_norm_eps,
            dtype,
        )?;
        let kv_b_proj = ColumnParallelLinear::load(
            weights,
            &format!("{prefix}.kv_b_proj"),
            dtype,
            rank,
            world_size,
            false,
        )?;

        // Output projection: row-parallel (all-reduce after matmul).
        let o_proj = RowParallelLinear::load(
            weights,
            &format!("{prefix}.o_proj"),
            dtype,
            rank,
            world_size,
            true,
        )?;

        // RoPE (over rope_head_dim only).
        let rotary_emb = if let Some(ref yarn) = config.rope_scaling {
            RotaryEmbedding::new_yarn(
                qk_rope_head_dim,
                config.max_position_embeddings,
                config.rope_theta,
                yarn.factor,
                yarn.beta_fast,
                yarn.beta_slow,
                yarn.mscale_all_dim,
                yarn.original_max_position_embeddings,
                dtype,
                device,
            )?
        } else {
            RotaryEmbedding::new(
                qk_rope_head_dim,
                config.max_position_embeddings,
                config.rope_theta,
                dtype,
                device,
            )?
        };

        let scale = 1.0 / (qk_head_dim as f64).sqrt();

        Ok(Self {
            q_a_proj,
            q_a_layernorm,
            q_b_proj,
            q_proj,
            kv_a_proj_with_mqa,
            kv_a_layernorm,
            kv_b_proj,
            o_proj,
            rotary_emb,
            num_heads,
            qk_nope_head_dim,
            qk_rope_head_dim,
            qk_head_dim,
            v_head_dim,
            kv_lora_rank,
            scale,
        })
    }

    /// Create with zero weights (for testing).
    pub fn zeros(config: &DeepSeekV2Config, dtype: DType, device: &Device) -> ModelResult<Self> {
        let num_heads = config.num_attention_heads;
        let qk_nope_head_dim = config.qk_nope_head_dim;
        let qk_rope_head_dim = config.qk_rope_head_dim;
        let v_head_dim = config.v_head_dim;
        let kv_lora_rank = config.kv_lora_rank;
        let qk_head_dim = qk_nope_head_dim + qk_rope_head_dim;

        let (q_a_proj, q_a_layernorm, q_b_proj, q_proj) =
            if let Some(q_lora_rank) = config.q_lora_rank {
                let q_a = Linear::zeros(config.hidden_size, q_lora_rank, dtype, device)?;
                let q_a_ln = RmsNorm::ones(q_lora_rank, config.rms_norm_eps, dtype, device)?;
                let q_b = ColumnParallelLinear::new(
                    Linear::zeros(q_lora_rank, num_heads * qk_head_dim, dtype, device)?,
                    false,
                );
                (Some(q_a), Some(q_a_ln), Some(q_b), None)
            } else {
                let q = ColumnParallelLinear::new(
                    Linear::zeros(config.hidden_size, num_heads * qk_head_dim, dtype, device)?,
                    false,
                );
                (None, None, None, Some(q))
            };

        let kv_a_proj_with_mqa = Linear::zeros(
            config.hidden_size,
            kv_lora_rank + qk_rope_head_dim,
            dtype,
            device,
        )?;
        let kv_a_layernorm = RmsNorm::ones(kv_lora_rank, config.rms_norm_eps, dtype, device)?;
        let kv_b_proj = ColumnParallelLinear::new(
            Linear::zeros(
                kv_lora_rank,
                num_heads * (qk_nope_head_dim + v_head_dim),
                dtype,
                device,
            )?,
            false,
        );

        let o_proj = RowParallelLinear::new(
            Linear::zeros(num_heads * v_head_dim, config.hidden_size, dtype, device)?,
            true,
        );

        let rotary_emb = RotaryEmbedding::new(
            qk_rope_head_dim,
            config.max_position_embeddings,
            config.rope_theta,
            dtype,
            device,
        )?;

        let scale = 1.0 / (qk_head_dim as f64).sqrt();

        Ok(Self {
            q_a_proj,
            q_a_layernorm,
            q_b_proj,
            q_proj,
            kv_a_proj_with_mqa,
            kv_a_layernorm,
            kv_b_proj,
            o_proj,
            rotary_emb,
            num_heads,
            qk_nope_head_dim,
            qk_rope_head_dim,
            qk_head_dim,
            v_head_dim,
            kv_lora_rank,
            scale,
        })
    }

    /// Forward pass.
    ///
    /// * `hidden_states` — shape `[num_tokens, hidden_size]`
    /// * `positions` — shape `[num_tokens]`
    /// * `kv_cache` — optional per-layer KV handle
    ///
    /// Returns shape `[num_tokens, hidden_size]`.
    pub fn forward(
        &self,
        hidden_states: &Tensor,
        positions: &Tensor,
        kv_cache: Option<crate::LayerKvHandle<'_>>,
    ) -> ModelResult<Tensor> {
        let num_tokens = hidden_states.dim(0).map_err(ModelError::Candle)?;

        // --- Q path ---
        let q_full = if let (Some(q_a), Some(q_a_ln), Some(q_b)) =
            (&self.q_a_proj, &self.q_a_layernorm, &self.q_b_proj)
        {
            let q_latent = q_a.forward(hidden_states).map_err(ModelError::Candle)?;
            let q_latent = crate::ops::rms_norm(&q_latent, q_a_ln).map_err(ModelError::Candle)?;
            q_b.forward(&q_latent).map_err(ModelError::Candle)?
        } else {
            self.q_proj
                .as_ref()
                .unwrap()
                .forward(hidden_states)
                .map_err(ModelError::Candle)?
        };

        // Reshape: [num_tokens, num_heads * qk_head_dim] -> [num_tokens, num_heads, qk_head_dim]
        let q = q_full
            .reshape((num_tokens, self.num_heads, self.qk_head_dim))
            .map_err(ModelError::Candle)?;

        // Split Q into nope and rope parts.
        let q_nope = q
            .narrow(2, 0, self.qk_nope_head_dim)
            .map_err(ModelError::Candle)?;
        let q_pe = q
            .narrow(2, self.qk_nope_head_dim, self.qk_rope_head_dim)
            .map_err(ModelError::Candle)?;

        // --- KV path ---
        let kv_a = self
            .kv_a_proj_with_mqa
            .forward(hidden_states)
            .map_err(ModelError::Candle)?;
        // [num_tokens, kv_lora_rank + qk_rope_head_dim]

        // Split into latent and k_pe.
        let kv_latent = kv_a
            .narrow(1, 0, self.kv_lora_rank)
            .map_err(ModelError::Candle)?;
        let k_pe = kv_a
            .narrow(1, self.kv_lora_rank, self.qk_rope_head_dim)
            .map_err(ModelError::Candle)?;

        // kv_latent → RMSNorm → kv_b_proj.
        let kv_latent =
            crate::ops::rms_norm(&kv_latent, &self.kv_a_layernorm).map_err(ModelError::Candle)?;
        let kv_b = self
            .kv_b_proj
            .forward(&kv_latent)
            .map_err(ModelError::Candle)?;
        // [num_tokens, num_heads * (qk_nope_head_dim + v_head_dim)]

        let kv_b = kv_b
            .reshape((
                num_tokens,
                self.num_heads,
                self.qk_nope_head_dim + self.v_head_dim,
            ))
            .map_err(ModelError::Candle)?;

        // Split into k_nope and v.
        let k_nope = kv_b
            .narrow(2, 0, self.qk_nope_head_dim)
            .map_err(ModelError::Candle)?;
        let v = kv_b
            .narrow(2, self.qk_nope_head_dim, self.v_head_dim)
            .map_err(ModelError::Candle)?;

        // --- Apply RoPE to q_pe and k_pe ---
        // k_pe: [num_tokens, qk_rope_head_dim] → [num_tokens, 1, qk_rope_head_dim] (single KV head for MQA)
        let k_pe = k_pe
            .reshape((num_tokens, 1, self.qk_rope_head_dim))
            .map_err(ModelError::Candle)?;

        let (q_pe, k_pe) = self.rotary_emb.apply(&q_pe, &k_pe, positions)?;

        // --- Assemble full Q and K ---
        // Q: [nope, pe] → [num_tokens, num_heads, qk_head_dim]
        let q = Tensor::cat(&[&q_nope, &q_pe], 2).map_err(ModelError::Candle)?;

        // K: expand k_pe to all heads, then concat with k_nope.
        // k_pe: [num_tokens, 1, rope_dim] → [num_tokens, num_heads, rope_dim]
        let k_pe = k_pe
            .expand((num_tokens, self.num_heads, self.qk_rope_head_dim))
            .map_err(ModelError::Candle)?
            .contiguous()
            .map_err(ModelError::Candle)?;
        let k = Tensor::cat(&[&k_nope, &k_pe], 2).map_err(ModelError::Candle)?;

        // --- Pad V to qk_head_dim for attention alignment ---
        // V: [num_tokens, num_heads, v_head_dim] → [num_tokens, num_heads, qk_head_dim]
        let v_padded = if self.v_head_dim < self.qk_head_dim {
            let pad_size = self.qk_head_dim - self.v_head_dim;
            let padding = Tensor::zeros(
                (num_tokens, self.num_heads, pad_size),
                v.dtype(),
                v.device(),
            )
            .map_err(ModelError::Candle)?;
            Tensor::cat(&[&v, &padding], 2).map_err(ModelError::Candle)?
        } else {
            v.clone()
        };

        // --- Cache + Attention ---
        let attn_output = attention_with_cache(&q, &k, &v_padded, self.scale, kv_cache, None)?;

        // --- Slice V back to v_head_dim ---
        let attn_output = attn_output
            .narrow(2, 0, self.v_head_dim)
            .map_err(ModelError::Candle)?;

        // Reshape: [num_tokens, num_heads, v_head_dim] → [num_tokens, num_heads * v_head_dim]
        let attn_output = attn_output
            .reshape((num_tokens, self.num_heads * self.v_head_dim))
            .map_err(ModelError::Candle)?
            .contiguous()
            .map_err(ModelError::Candle)?;

        // Output projection.
        self.o_proj
            .forward(&attn_output)
            .map_err(ModelError::Candle)
    }
}

// ---------------------------------------------------------------------------
// DeepSeekV2DecoderLayer
// ---------------------------------------------------------------------------

/// A single DeepSeek V2 decoder layer.
///
/// Dense layers (index < first_k_dense_replace) use standard MLP.
/// MoE layers use the mixture-of-experts + shared experts.
pub struct DeepSeekV2DecoderLayer {
    self_attn: DeepSeekV2Attention,
    mlp: DeepSeekV2Mlp,
    input_layernorm: RmsNorm,
    post_attention_layernorm: RmsNorm,
}

/// Either a dense MLP or a MoE layer.
enum DeepSeekV2Mlp {
    Dense(LlamaMLP),
    MoE(DeepSeekV2MoE),
}

impl Module for DeepSeekV2Mlp {
    fn forward(&self, x: &Tensor) -> candle_core::Result<Tensor> {
        match self {
            DeepSeekV2Mlp::Dense(mlp) => mlp.forward(x),
            DeepSeekV2Mlp::MoE(moe) => moe.forward(x),
        }
    }
}

impl DeepSeekV2DecoderLayer {
    /// Load a decoder layer.
    #[allow(clippy::too_many_arguments)]
    pub fn load(
        weights: &mut ModelWeights,
        prefix: &str,
        config: &DeepSeekV2Config,
        layer_idx: usize,
        dtype: DType,
        device: &Device,
        rank: usize,
        world_size: usize,
    ) -> ModelResult<Self> {
        let self_attn = DeepSeekV2Attention::load(
            weights,
            &format!("{prefix}.self_attn"),
            config,
            dtype,
            device,
            rank,
            world_size,
        )?;

        let mlp = if config.has_moe() && layer_idx >= config.first_k_dense_replace {
            // MoE layers: experts are replicated (not sharded).
            DeepSeekV2Mlp::MoE(DeepSeekV2MoE::load(
                weights,
                &format!("{prefix}.mlp"),
                config,
                dtype,
            )?)
        } else {
            // Dense MLP layers: gate_up column-parallel, down_proj row-parallel.
            DeepSeekV2Mlp::Dense(LlamaMLP::load(
                weights,
                &format!("{prefix}.mlp"),
                dtype,
                rank,
                world_size,
            )?)
        };

        let input_layernorm = RmsNorm::load(
            weights,
            &format!("{prefix}.input_layernorm"),
            config.rms_norm_eps,
            dtype,
        )?;
        let post_attention_layernorm = RmsNorm::load(
            weights,
            &format!("{prefix}.post_attention_layernorm"),
            config.rms_norm_eps,
            dtype,
        )?;

        Ok(Self {
            self_attn,
            mlp,
            input_layernorm,
            post_attention_layernorm,
        })
    }

    /// Create with zero weights (for testing).
    pub fn zeros(
        config: &DeepSeekV2Config,
        layer_idx: usize,
        dtype: DType,
        device: &Device,
    ) -> ModelResult<Self> {
        let self_attn = DeepSeekV2Attention::zeros(config, dtype, device)?;

        let mlp = if config.has_moe() && layer_idx >= config.first_k_dense_replace {
            DeepSeekV2Mlp::MoE(DeepSeekV2MoE::zeros(config, dtype, device)?)
        } else {
            DeepSeekV2Mlp::Dense(LlamaMLP::zeros(
                config.hidden_size,
                config.intermediate_size,
                dtype,
                device,
            )?)
        };

        let input_layernorm =
            RmsNorm::ones(config.hidden_size, config.rms_norm_eps, dtype, device)?;
        let post_attention_layernorm =
            RmsNorm::ones(config.hidden_size, config.rms_norm_eps, dtype, device)?;

        Ok(Self {
            self_attn,
            mlp,
            input_layernorm,
            post_attention_layernorm,
        })
    }

    /// Forward pass with residual threading.
    pub fn forward(
        &self,
        hidden_states: &Tensor,
        residual: Option<&Tensor>,
        positions: &Tensor,
        kv_cache: Option<crate::LayerKvHandle<'_>>,
    ) -> ModelResult<(Tensor, Tensor)> {
        // Pre-attention layernorm: fuse previous MLP residual add when available.
        let (normed, residual) = if let Some(residual) = residual {
            crate::ops::fused_add_rms_norm(hidden_states, residual, &self.input_layernorm)
                .map_err(ModelError::Candle)?
        } else {
            let normed = crate::ops::rms_norm(hidden_states, &self.input_layernorm)
                .map_err(ModelError::Candle)?;
            (normed, hidden_states.clone())
        };

        let attn_output = self.self_attn.forward(&normed, positions, kv_cache)?;

        // Fused residual add + post-attention layernorm.
        let (normed, residual) =
            crate::ops::fused_add_rms_norm(&attn_output, &residual, &self.post_attention_layernorm)
                .map_err(ModelError::Candle)?;

        // MLP/MoE (residual add deferred to next layer or final norm).
        let mlp_output = self.mlp.forward(&normed).map_err(ModelError::Candle)?;

        Ok((mlp_output, residual))
    }
}

// ---------------------------------------------------------------------------
// DeepSeekV2Model
// ---------------------------------------------------------------------------

/// DeepSeek V2 transformer backbone.
struct DeepSeekV2Model {
    embed_tokens: Embedding,
    layers: Vec<DeepSeekV2DecoderLayer>,
    norm: RmsNorm,
}

impl DeepSeekV2Model {
    fn load(
        weights: &mut ModelWeights,
        prefix: &str,
        config: &DeepSeekV2Config,
        dtype: DType,
        device: &Device,
        rank: usize,
        world_size: usize,
    ) -> ModelResult<Self> {
        let embed_tokens = Embedding::load(weights, &format!("{prefix}.embed_tokens"), dtype)?;

        let mut layers = Vec::with_capacity(config.num_hidden_layers);
        for i in 0..config.num_hidden_layers {
            let layer = DeepSeekV2DecoderLayer::load(
                weights,
                &format!("{prefix}.layers.{i}"),
                config,
                i,
                dtype,
                device,
                rank,
                world_size,
            )?;
            layers.push(layer);
        }

        let norm = RmsNorm::load(
            weights,
            &format!("{prefix}.norm"),
            config.rms_norm_eps,
            dtype,
        )?;

        Ok(Self {
            embed_tokens,
            layers,
            norm,
        })
    }

    fn forward(
        &self,
        input_ids: &Tensor,
        positions: &Tensor,
        mut kv_cache: Option<&mut crate::KvCacheStorage<'_>>,
    ) -> ModelResult<Tensor> {
        let mut hidden_states = self
            .embed_tokens
            .forward(input_ids)
            .map_err(ModelError::Candle)?;

        let mut residual: Option<Tensor> = None;
        for (i, layer) in self.layers.iter().enumerate() {
            let layer_handle = kv_cache.as_mut().map(|s| s.layer_handle(i));
            let (hs, res) =
                layer.forward(&hidden_states, residual.as_ref(), positions, layer_handle)?;
            hidden_states = hs;
            residual = Some(res);
        }

        // Final norm: fuse last MLP's residual add into the norm.
        let (normed, _) =
            crate::ops::fused_add_rms_norm(&hidden_states, residual.as_ref().unwrap(), &self.norm)
                .map_err(ModelError::Candle)?;
        Ok(normed)
    }

    fn num_layers(&self) -> usize {
        self.layers.len()
    }
}

// ---------------------------------------------------------------------------
// DeepSeekV2ForCausalLM
// ---------------------------------------------------------------------------

/// DeepSeek V2 for causal language modeling.
pub struct DeepSeekV2ForCausalLM {
    model: DeepSeekV2Model,
    lm_head: Linear,
}

impl DeepSeekV2ForCausalLM {
    /// Load the full model from weights.
    pub fn load(
        weights: &mut ModelWeights,
        config: &DeepSeekV2Config,
        dtype: DType,
        device: &Device,
        rank: usize,
        world_size: usize,
    ) -> ModelResult<Self> {
        let model =
            DeepSeekV2Model::load(weights, "model", config, dtype, device, rank, world_size)?;

        let lm_head = if config.tie_word_embeddings {
            Linear::new(model.embed_tokens.weight().clone(), None)
        } else {
            Linear::load(weights, "lm_head", dtype)?
        };

        Ok(Self { model, lm_head })
    }
}

impl crate::Model for DeepSeekV2ForCausalLM {
    fn inject_tp_group(
        &mut self,
        group: std::sync::Arc<dyn vllm_model::process_group::ProcessGroup>,
    ) -> ModelResult<()> {
        for layer in &mut self.model.layers {
            // Attention o_proj is RowParallelLinear — needs all-reduce.
            layer.self_attn.o_proj.set_tp_group(group.clone());
            // Dense MLP layers: down_proj is RowParallelLinear — needs all-reduce.
            // MoE layers: experts are replicated, no TP group needed.
            if let DeepSeekV2Mlp::Dense(ref mut mlp) = layer.mlp {
                mlp.down_proj.set_tp_group(group.clone());
            }
        }
        Ok(())
    }

    fn inject_lora(&mut self, adapter: &LoraAdapter) -> ModelResult<()> {
        let targets = &adapter.config.target_modules;
        for (i, layer) in self.model.layers.iter_mut().enumerate() {
            let attn_prefix = format!("model.layers.{}.self_attn", i);
            // MLA has non-standard projection names — inject via direct field access.
            for name in targets {
                let key = format!("{}.{}", attn_prefix, name);
                if let Some((a, b)) = adapter.weights.get(&key) {
                    match name.as_str() {
                        "q_a_proj" => {
                            if let Some(ref mut p) = layer.self_attn.q_a_proj {
                                p.attach_lora(a.clone(), b.clone(), adapter.scaling)?;
                            }
                        }
                        "q_b_proj" => {
                            if let Some(ref mut p) = layer.self_attn.q_b_proj {
                                p.inner_mut()
                                    .attach_lora(a.clone(), b.clone(), adapter.scaling)?;
                            }
                        }
                        "q_proj" => {
                            if let Some(ref mut p) = layer.self_attn.q_proj {
                                p.inner_mut()
                                    .attach_lora(a.clone(), b.clone(), adapter.scaling)?;
                            }
                        }
                        "kv_a_proj_with_mqa" => {
                            layer.self_attn.kv_a_proj_with_mqa.attach_lora(
                                a.clone(),
                                b.clone(),
                                adapter.scaling,
                            )?;
                        }
                        "kv_b_proj" => {
                            layer.self_attn.kv_b_proj.inner_mut().attach_lora(
                                a.clone(),
                                b.clone(),
                                adapter.scaling,
                            )?;
                        }
                        "o_proj" => {
                            layer.self_attn.o_proj.inner_mut().attach_lora(
                                a.clone(),
                                b.clone(),
                                adapter.scaling,
                            )?;
                        }
                        _ => {}
                    }
                }
            }
            // MLP: dense layers use LlamaMLP, MoE layers have expert MLPs.
            // Only inject LoRA into dense MLP layers (MoE expert LoRA is rare).
            if let DeepSeekV2Mlp::Dense(ref mut mlp) = layer.mlp {
                let mlp_prefix = format!("model.layers.{}.mlp", i);
                mlp.inject_lora(&mlp_prefix, adapter)?;
            }
        }
        Ok(())
    }

    fn forward(
        &self,
        input_ids: &Tensor,
        positions: &Tensor,
        kv_cache: Option<&mut crate::KvCacheStorage<'_>>,
    ) -> ModelResult<Tensor> {
        let hidden_states = self.model.forward(input_ids, positions, kv_cache)?;
        let logits = self
            .lm_head
            .forward(&hidden_states)
            .map_err(ModelError::Candle)?;
        logits.to_dtype(DType::F32).map_err(ModelError::Candle)
    }

    fn num_layers(&self) -> usize {
        self.model.num_layers()
    }

    fn hidden_states(&self, input_ids: &Tensor, positions: &Tensor) -> ModelResult<Tensor> {
        self.model.forward(input_ids, positions, None)
    }
}

/// Factory function for the model registry.
pub fn create_deepseek_v2(
    weights: &mut ModelWeights,
    config: &HfModelConfig,
    dtype: DType,
    device: &Device,
    rank: usize,
    world_size: usize,
) -> ModelResult<Box<dyn crate::Model>> {
    let ds_config = DeepSeekV2Config::from_hf_config(config)?;
    let model = DeepSeekV2ForCausalLM::load(weights, &ds_config, dtype, device, rank, world_size)?;
    Ok(Box::new(model))
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use safetensors::tensor::TensorView;

    fn test_config() -> DeepSeekV2Config {
        DeepSeekV2Config {
            hidden_size: 32,
            num_attention_heads: 4,
            num_kv_heads: 4,
            num_hidden_layers: 2,
            intermediate_size: 64,
            vocab_size: 100,
            max_position_embeddings: 128,
            rms_norm_eps: 1e-5,
            rope_theta: 10000.0,
            tie_word_embeddings: false,
            qk_nope_head_dim: 4,
            qk_rope_head_dim: 4,
            v_head_dim: 4,
            q_lora_rank: Some(16),
            kv_lora_rank: 8,
            n_routed_experts: 4,
            n_shared_experts: 1,
            num_experts_per_tok: 2,
            first_k_dense_replace: 1,
            moe_intermediate_size: 32,
            norm_topk_prob: true,
            routed_scaling_factor: 1.0,
            rope_scaling: None,
        }
    }

    fn test_config_dense() -> DeepSeekV2Config {
        DeepSeekV2Config {
            n_routed_experts: 0,
            n_shared_experts: 0,
            ..test_config()
        }
    }

    #[test]
    fn test_deepseek_v2_config_from_hf() {
        let hf_config: HfModelConfig = serde_json::from_str(
            r#"{
                "architectures": ["DeepseekV2ForCausalLM"],
                "model_type": "deepseek_v2",
                "hidden_size": 5120,
                "num_attention_heads": 128,
                "num_key_value_heads": 128,
                "num_hidden_layers": 60,
                "intermediate_size": 12288,
                "vocab_size": 102400,
                "max_position_embeddings": 163840,
                "rms_norm_eps": 1e-6,
                "rope_theta": 10000.0,
                "qk_nope_head_dim": 128,
                "qk_rope_head_dim": 64,
                "v_head_dim": 128,
                "q_lora_rank": 1536,
                "kv_lora_rank": 512,
                "n_routed_experts": 160,
                "n_shared_experts": 2,
                "num_experts_per_tok": 6,
                "first_k_dense_replace": 1,
                "moe_intermediate_size": 1536,
                "norm_topk_prob": true,
                "routed_scaling_factor": 1.0,
                "rope_scaling": {
                    "type": "yarn",
                    "factor": 40.0,
                    "beta_fast": 32.0,
                    "beta_slow": 1.0,
                    "mscale_all_dim": 0.1,
                    "original_max_position_embeddings": 4096
                },
                "tie_word_embeddings": false
            }"#,
        )
        .unwrap();

        let config = DeepSeekV2Config::from_hf_config(&hf_config).unwrap();
        assert_eq!(config.hidden_size, 5120);
        assert_eq!(config.num_attention_heads, 128);
        assert_eq!(config.qk_nope_head_dim, 128);
        assert_eq!(config.qk_rope_head_dim, 64);
        assert_eq!(config.v_head_dim, 128);
        assert_eq!(config.q_lora_rank, Some(1536));
        assert_eq!(config.kv_lora_rank, 512);
        assert_eq!(config.n_routed_experts, 160);
        assert_eq!(config.n_shared_experts, 2);
        assert_eq!(config.num_experts_per_tok, 6);
        assert_eq!(config.first_k_dense_replace, 1);
        assert_eq!(config.moe_intermediate_size, 1536);
        assert!(config.norm_topk_prob);
        assert_eq!(config.qk_head_dim(), 192);
        assert!(config.has_moe());
        assert!(config.rope_scaling.is_some());
        let yarn = config.rope_scaling.unwrap();
        assert_eq!(yarn.factor, 40.0);
        assert_eq!(yarn.beta_fast, 32.0);
        assert_eq!(yarn.beta_slow, 1.0);
    }

    #[test]
    fn test_deepseek_v2_moe_routing() {
        let config = test_config();
        let moe = DeepSeekV2MoE::zeros(&config, DType::F32, &Device::Cpu).unwrap();

        let x = Tensor::ones(&[3, config.hidden_size], DType::F32, &Device::Cpu).unwrap();
        let out = moe.forward(&x).unwrap();
        assert_eq!(out.dims(), &[3, config.hidden_size]);
    }

    #[test]
    fn test_deepseek_v2_mla_attention_forward() {
        let config = test_config();
        let attn = DeepSeekV2Attention::zeros(&config, DType::F32, &Device::Cpu).unwrap();

        let num_tokens = 4;
        let x = Tensor::ones(&[num_tokens, config.hidden_size], DType::F32, &Device::Cpu).unwrap();
        let positions = Tensor::new(&[0u32, 1, 2, 3], &Device::Cpu).unwrap();

        let out = attn.forward(&x, &positions, None).unwrap();
        assert_eq!(out.dims(), &[num_tokens, config.hidden_size]);
    }

    #[test]
    fn test_deepseek_v2_mla_single_token() {
        let config = test_config();
        let attn = DeepSeekV2Attention::zeros(&config, DType::F32, &Device::Cpu).unwrap();

        let x = Tensor::ones(&[1, config.hidden_size], DType::F32, &Device::Cpu).unwrap();
        let positions = Tensor::new(&[0u32], &Device::Cpu).unwrap();

        let out = attn.forward(&x, &positions, None).unwrap();
        assert_eq!(out.dims(), &[1, config.hidden_size]);
    }

    #[test]
    fn test_deepseek_v2_kv_cache_e2e() {
        // Prefill + decode with KV cache on a dense (no MoE) config.
        let config = test_config_dense();

        let mut layers: Vec<DeepSeekV2DecoderLayer> = Vec::new();
        for i in 0..config.num_hidden_layers {
            layers
                .push(DeepSeekV2DecoderLayer::zeros(&config, i, DType::F32, &Device::Cpu).unwrap());
        }

        // Prefill: 3 tokens.
        let x = Tensor::ones(&[3, config.hidden_size], DType::F32, &Device::Cpu).unwrap();
        let positions = Tensor::new(&[0u32, 1, 2], &Device::Cpu).unwrap();
        let mut kv_cache: crate::KvCache = vec![None; config.num_hidden_layers];
        let mut storage = crate::KvCacheStorage::Contiguous(&mut kv_cache);

        let mut hidden = x.clone();
        let mut residual: Option<Tensor> = None;
        for (i, layer) in layers.iter().enumerate() {
            let handle = storage.layer_handle(i);
            let (hs, res) = layer
                .forward(&hidden, residual.as_ref(), &positions, Some(handle))
                .unwrap();
            hidden = hs;
            residual = Some(res);
        }
        drop(storage);
        assert_eq!(hidden.dims(), &[3, config.hidden_size]);

        // KV cache should be populated.
        for entry in &kv_cache {
            assert!(entry.is_some());
        }

        // Decode: 1 token at position 3.
        let x2 = Tensor::ones(&[1, config.hidden_size], DType::F32, &Device::Cpu).unwrap();
        let pos2 = Tensor::new(&[3u32], &Device::Cpu).unwrap();
        let mut storage = crate::KvCacheStorage::Contiguous(&mut kv_cache);

        let mut hidden2 = x2;
        let mut residual2: Option<Tensor> = None;
        for (i, layer) in layers.iter().enumerate() {
            let handle = storage.layer_handle(i);
            let (hs, res) = layer
                .forward(&hidden2, residual2.as_ref(), &pos2, Some(handle))
                .unwrap();
            hidden2 = hs;
            residual2 = Some(res);
        }
        drop(storage);
        assert_eq!(hidden2.dims(), &[1, config.hidden_size]);
    }

    #[test]
    fn test_deepseek_v2_model_from_weights() {
        // Build a tiny DeepSeek V2 model from synthetic weights (dense only).
        let config = test_config_dense();
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("model.safetensors");
        let dtype = DType::F32;
        let device = Device::Cpu;

        let mut specs: Vec<(String, Vec<usize>)> = Vec::new();

        // Embeddings.
        specs.push((
            "model.embed_tokens.weight".to_string(),
            vec![config.vocab_size, config.hidden_size],
        ));

        let num_heads = config.num_attention_heads;
        let qk_head_dim = config.qk_head_dim();
        let nope = config.qk_nope_head_dim;
        let rope_dim = config.qk_rope_head_dim;
        let v_dim = config.v_head_dim;
        let kv_lora = config.kv_lora_rank;

        for i in 0..config.num_hidden_layers {
            let prefix = format!("model.layers.{i}");

            // MLA attention weights.
            if let Some(q_lora_rank) = config.q_lora_rank {
                specs.push((
                    format!("{prefix}.self_attn.q_a_proj.weight"),
                    vec![q_lora_rank, config.hidden_size],
                ));
                specs.push((
                    format!("{prefix}.self_attn.q_a_layernorm.weight"),
                    vec![q_lora_rank],
                ));
                specs.push((
                    format!("{prefix}.self_attn.q_b_proj.weight"),
                    vec![num_heads * qk_head_dim, q_lora_rank],
                ));
            } else {
                specs.push((
                    format!("{prefix}.self_attn.q_proj.weight"),
                    vec![num_heads * qk_head_dim, config.hidden_size],
                ));
            }

            specs.push((
                format!("{prefix}.self_attn.kv_a_proj_with_mqa.weight"),
                vec![kv_lora + rope_dim, config.hidden_size],
            ));
            specs.push((
                format!("{prefix}.self_attn.kv_a_layernorm.weight"),
                vec![kv_lora],
            ));
            specs.push((
                format!("{prefix}.self_attn.kv_b_proj.weight"),
                vec![num_heads * (nope + v_dim), kv_lora],
            ));
            specs.push((
                format!("{prefix}.self_attn.o_proj.weight"),
                vec![config.hidden_size, num_heads * v_dim],
            ));

            // Dense MLP.
            specs.push((
                format!("{prefix}.mlp.gate_proj.weight"),
                vec![config.intermediate_size, config.hidden_size],
            ));
            specs.push((
                format!("{prefix}.mlp.up_proj.weight"),
                vec![config.intermediate_size, config.hidden_size],
            ));
            specs.push((
                format!("{prefix}.mlp.down_proj.weight"),
                vec![config.hidden_size, config.intermediate_size],
            ));

            specs.push((
                format!("{prefix}.input_layernorm.weight"),
                vec![config.hidden_size],
            ));
            specs.push((
                format!("{prefix}.post_attention_layernorm.weight"),
                vec![config.hidden_size],
            ));
        }

        specs.push(("model.norm.weight".to_string(), vec![config.hidden_size]));
        specs.push((
            "lm_head.weight".to_string(),
            vec![config.vocab_size, config.hidden_size],
        ));

        create_test_weights(&path, &specs);

        let mut weights = ModelWeights::from_single_file(&path, &device).unwrap();
        let model =
            DeepSeekV2ForCausalLM::load(&mut weights, &config, dtype, &device, 0, 1).unwrap();

        let input_ids = Tensor::new(&[1u32, 5, 10], &device).unwrap();
        let positions = Tensor::new(&[0u32, 1, 2], &device).unwrap();

        let logits = crate::Model::forward(&model, &input_ids, &positions, None).unwrap();
        assert_eq!(logits.dims(), &[3, config.vocab_size]);
    }

    #[test]
    fn test_deepseek_v2_registry() {
        let registry = crate::ModelRegistry::default_registry();
        assert!(registry.contains("DeepseekV2ForCausalLM"));
    }

    // -----------------------------------------------------------------------
    // Test helper
    // -----------------------------------------------------------------------

    fn create_test_weights(path: &std::path::Path, specs: &[(String, Vec<usize>)]) {
        let mut all_data: Vec<Vec<u8>> = Vec::new();
        for (name, shape) in specs {
            let num_elements: usize = shape.iter().product();
            let val = if name.contains("layernorm") || name == "model.norm.weight" {
                1.0f32
            } else {
                0.01f32
            };
            let data: Vec<u8> = (0..num_elements).flat_map(|_| val.to_le_bytes()).collect();
            all_data.push(data);
        }

        let views: Vec<(&str, TensorView<'_>)> = specs
            .iter()
            .zip(all_data.iter())
            .map(|((name, shape), data)| {
                (
                    name.as_str(),
                    TensorView::new(safetensors::Dtype::F32, shape.clone(), data).unwrap(),
                )
            })
            .collect();

        let bytes = safetensors::tensor::serialize(views, None).unwrap();
        std::fs::write(path, bytes).unwrap();
    }
}
