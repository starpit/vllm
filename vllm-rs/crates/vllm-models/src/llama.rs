// SPDX-License-Identifier: Apache-2.0
//! LLaMA model architecture.
//!
//! Implements:
//! - `LlamaForCausalLM` — top-level model with lm_head and logits
//! - `LlamaModel` — transformer backbone (embed + layers + final norm)
//! - `LlamaDecoderLayer` — single transformer layer (attention + MLP + norms)
//! - `LlamaAttention` — multi-head attention with RoPE and GQA
//! - `LlamaMLP` — SiLU-gated feed-forward network
//!
//! Also covers Mistral, which uses the same architecture.
//!
//! Port of: `vllm/model_executor/models/llama.py`

use candle_core::{DType, Device, Module, Tensor};

use vllm_model::error::{ModelError, ModelResult};
use vllm_model::layers::{
    Embedding, Linear, RmsNorm, RotaryEmbedding, RowParallelLinear, load_fused_gate_up,
    load_fused_qkv,
};
use vllm_model::lora::LoraAdapter;
use vllm_model::weight::{HfModelConfig, ModelWeights};

use crate::attention::attention_with_cache;

// ---------------------------------------------------------------------------
// LlamaConfig
// ---------------------------------------------------------------------------

/// LongRoPE scaling configuration (used by Phi-3/Phi-4 family).
///
/// Per-dimension frequency rescale factors for short and long contexts.
#[derive(Debug, Clone)]
pub struct LongRopeScaling {
    /// Rescale factors for short contexts (len = rotary_dim / 2).
    pub short_factor: Vec<f64>,
    /// Rescale factors for long contexts (len = rotary_dim / 2).
    pub long_factor: Vec<f64>,
    /// Original max position embeddings before LongRoPE extension.
    pub original_max_position_embeddings: usize,
}

/// Parsed configuration for a LLaMA model.
#[derive(Debug, Clone)]
pub struct LlamaConfig {
    pub hidden_size: usize,
    pub num_attention_heads: usize,
    pub num_kv_heads: usize,
    pub num_hidden_layers: usize,
    pub intermediate_size: usize,
    pub vocab_size: usize,
    pub max_position_embeddings: usize,
    pub rms_norm_eps: f64,
    pub rope_theta: f64,
    pub head_dim: usize,
    pub tie_word_embeddings: bool,
    /// Sliding window size for attention. When `Some(w)`, each token only
    /// attends to the most recent `w` positions. Used by Mistral, Qwen2, etc.
    pub sliding_window: Option<usize>,
    /// Fraction of head dimensions that get RoPE (default 1.0). Used by Phi-3/4.
    /// `rotary_dim = (head_dim as f64 * partial_rotary_factor) as usize`
    pub partial_rotary_factor: f64,
    /// LongRoPE scaling parameters. `None` means standard RoPE.
    pub long_rope_scaling: Option<LongRopeScaling>,
}

impl LlamaConfig {
    /// Parse from a HuggingFace config.json.
    pub fn from_hf_config(config: &HfModelConfig) -> ModelResult<Self> {
        let hidden_size = config
            .hidden_size
            .ok_or_else(|| ModelError::Other("missing hidden_size".into()))?;
        let num_attention_heads = config
            .num_attention_heads
            .ok_or_else(|| ModelError::Other("missing num_attention_heads".into()))?;

        // Parse sliding_window from config.json extras (used by Mistral, Qwen2, Phi-3, etc.).
        // Handles both scalar (4096) and array ([null, 4096, null, 4096, ...]) formats.
        // The array format is used by newer Mistral models (3.x) — we extract the first
        // non-null value as the scalar sliding window size, matching Python vLLM's
        // _remap_mistral_sliding_window() behavior.
        let sliding_window = config.extra.get("sliding_window").and_then(|v| {
            if let Some(n) = v.as_u64() {
                Some(n as usize)
            } else if let Some(arr) = v.as_array() {
                arr.iter()
                    .find_map(|item| item.as_u64())
                    .map(|n| n as usize)
            } else {
                None
            }
        });

        // Parse partial_rotary_factor (Phi-3/4 family).
        let partial_rotary_factor = config
            .extra
            .get("partial_rotary_factor")
            .and_then(|v| v.as_f64())
            .unwrap_or(1.0);

        let head_dim = config
            .head_dim()
            .unwrap_or(hidden_size / num_attention_heads);

        // Parse LongRoPE scaling (rope_scaling.type == "longrope").
        let long_rope_scaling = config.extra.get("rope_scaling").and_then(|rs| {
            let scaling_type = rs.get("type")?.as_str()?;
            if scaling_type != "longrope" {
                return None;
            }
            let short_factor: Vec<f64> = rs
                .get("short_factor")?
                .as_array()?
                .iter()
                .filter_map(|v| v.as_f64())
                .collect();
            let long_factor: Vec<f64> = rs
                .get("long_factor")?
                .as_array()?
                .iter()
                .filter_map(|v| v.as_f64())
                .collect();
            let original_max = config
                .extra
                .get("original_max_position_embeddings")
                .and_then(|v| v.as_u64())
                .unwrap_or(4096) as usize;

            if short_factor.is_empty() || long_factor.is_empty() {
                return None;
            }

            Some(LongRopeScaling {
                short_factor,
                long_factor,
                original_max_position_embeddings: original_max,
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
            head_dim,
            tie_word_embeddings: config.tie_word_embeddings.unwrap_or(false),
            sliding_window,
            partial_rotary_factor,
            long_rope_scaling,
        })
    }
}

// ---------------------------------------------------------------------------
// LlamaMLP
// ---------------------------------------------------------------------------

/// LLaMA MLP (SiLU-gated feed-forward network) with fused gate+up projection.
///
/// Forward: gate_up_proj(x) → split → SiLU(gate) * up → down_proj
///
/// The gate and up weights are concatenated into a single linear layer,
/// eliminating one cuBLAS kernel launch per layer per decode step.
///
/// Port of: `vllm/model_executor/models/llama.py::LlamaMLP`
pub struct LlamaMLP {
    gate_up_proj: Linear,
    down_proj: RowParallelLinear,
    /// Post-shard intermediate size (for splitting the fused output).
    intermediate_size: usize,
}

impl LlamaMLP {
    /// Load MLP weights from a model with fused gate+up projection.
    ///
    /// Weight names: `{prefix}.gate_proj`, `{prefix}.up_proj`, `{prefix}.down_proj`
    pub fn load(
        weights: &ModelWeights,
        prefix: &str,
        dtype: DType,
        rank: usize,
        world_size: usize,
    ) -> ModelResult<Self> {
        let (gate_up_proj, intermediate_size) = load_fused_gate_up(
            weights,
            &format!("{prefix}.gate_proj"),
            &format!("{prefix}.up_proj"),
            dtype,
            rank,
            world_size,
        )?;
        let down_proj = RowParallelLinear::load(
            weights,
            &format!("{}.down_proj", prefix),
            dtype,
            rank,
            world_size,
            true,
        )?;
        Ok(Self {
            gate_up_proj,
            down_proj,
            intermediate_size,
        })
    }

    /// Create with zero weights (for testing).
    pub fn zeros(
        hidden_size: usize,
        intermediate_size: usize,
        dtype: DType,
        device: &Device,
    ) -> ModelResult<Self> {
        let gate_up = Linear::zeros(hidden_size, 2 * intermediate_size, dtype, device)?;
        let down = Linear::zeros(intermediate_size, hidden_size, dtype, device)?;
        Ok(Self {
            gate_up_proj: gate_up,
            down_proj: RowParallelLinear::new(down, true),
            intermediate_size,
        })
    }
}

impl LlamaMLP {
    /// Inject LoRA weights into MLP projections.
    pub fn inject_lora(&mut self, prefix: &str, adapter: &LoraAdapter) -> ModelResult<()> {
        let targets = &adapter.config.target_modules;
        if targets.iter().any(|t| t == "gate_proj" || t == "up_proj") {
            eprintln!(
                "WARN: LoRA on fused gate_up projection not yet supported; \
                 gate_proj/up_proj LoRA will be ignored"
            );
        }
        // down_proj is RowParallelLinear.
        if targets.iter().any(|t| t == "down_proj") {
            let key = format!("{}.down_proj", prefix);
            if let Some((a, b)) = adapter.weights.get(&key) {
                self.down_proj
                    .inner_mut()
                    .attach_lora(a.clone(), b.clone(), adapter.scaling)?;
            }
        }
        Ok(())
    }
}

impl Module for LlamaMLP {
    fn forward(&self, x: &Tensor) -> candle_core::Result<Tensor> {
        let gate_up = self.gate_up_proj.forward(x)?;
        let gate = gate_up.narrow(1, 0, self.intermediate_size)?.contiguous()?;
        let up = gate_up
            .narrow(1, self.intermediate_size, self.intermediate_size)?
            .contiguous()?;
        let activated = crate::ops::silu_and_mul(&gate, &up)?;
        self.down_proj.forward(&activated)
    }
}

// ---------------------------------------------------------------------------
// LlamaAttention
// ---------------------------------------------------------------------------

/// LLaMA multi-head attention with RoPE, GQA, and fused QKV projection.
///
/// Forward: qkv_proj(x) → split Q/K/V → RoPE → scaled dot-product attention → o_proj
///
/// Q, K, V weights are fused into a single linear layer `[q_dim + 2*kv_dim, hidden]`,
/// eliminating two cuBLAS kernel launches per layer per decode step.
///
/// Port of: `vllm/model_executor/models/llama.py::LlamaAttention`
pub struct LlamaAttention {
    qkv_proj: Linear,
    o_proj: RowParallelLinear,
    rotary_emb: RotaryEmbedding,
    /// Post-shard Q output dimension (num_q_heads * head_dim).
    q_size: usize,
    /// Post-shard KV output dimension (num_kv_heads * head_dim).
    kv_size: usize,
    num_q_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    pub(crate) scale: f64,
    sliding_window: Option<usize>,
    /// Layer index for extracting the correct per-layer KV handle in batched forward.
    layer_idx: usize,
}

impl LlamaAttention {
    /// Load attention weights with fused QKV projection.
    ///
    /// Weight names: `{prefix}.q_proj`, `{prefix}.k_proj`, `{prefix}.v_proj`, `{prefix}.o_proj`
    #[allow(clippy::too_many_arguments)]
    pub fn load(
        weights: &ModelWeights,
        prefix: &str,
        config: &LlamaConfig,
        dtype: DType,
        device: &Device,
        rank: usize,
        world_size: usize,
        layer_idx: usize,
    ) -> ModelResult<Self> {
        let (qkv_proj, q_size, kv_size) = load_fused_qkv(weights, prefix, dtype, rank, world_size)?;
        let o_proj = RowParallelLinear::load(
            weights,
            &format!("{}.o_proj", prefix),
            dtype,
            rank,
            world_size,
            true,
        )?;

        let num_q_heads = config.num_attention_heads / world_size;
        let num_kv_heads = config.num_kv_heads / world_size;
        let head_dim = config.head_dim;

        let rotary_emb = RotaryEmbedding::new(
            head_dim,
            config.max_position_embeddings,
            config.rope_theta,
            dtype,
            device,
        )?;

        Ok(Self {
            qkv_proj,
            o_proj,
            rotary_emb,
            q_size,
            kv_size,
            num_q_heads,
            num_kv_heads,
            head_dim,
            scale: 1.0 / (head_dim as f64).sqrt(),
            sliding_window: config.sliding_window,
            layer_idx,
        })
    }

    /// Create with zero weights (for testing).
    pub fn zeros(
        config: &LlamaConfig,
        dtype: DType,
        device: &Device,
        layer_idx: usize,
    ) -> ModelResult<Self> {
        let hidden = config.hidden_size;
        let q_size = config.num_attention_heads * config.head_dim;
        let kv_size = config.num_kv_heads * config.head_dim;
        let qkv_size = q_size + 2 * kv_size;

        let qkv_proj = Linear::zeros(hidden, qkv_size, dtype, device)?;
        let o_proj = RowParallelLinear::new(Linear::zeros(q_size, hidden, dtype, device)?, true);

        let rotary_emb = RotaryEmbedding::new(
            config.head_dim,
            config.max_position_embeddings,
            config.rope_theta,
            dtype,
            device,
        )?;

        Ok(Self {
            qkv_proj,
            o_proj,
            rotary_emb,
            q_size,
            kv_size,
            num_q_heads: config.num_attention_heads,
            num_kv_heads: config.num_kv_heads,
            head_dim: config.head_dim,
            scale: 1.0 / (config.head_dim as f64).sqrt(),
            sliding_window: config.sliding_window,
            layer_idx,
        })
    }

    /// Split a fused QKV output into Q, K, V tensors reshaped to `[N, heads, head_dim]`.
    fn split_qkv(&self, qkv: &Tensor, num_tokens: usize) -> ModelResult<(Tensor, Tensor, Tensor)> {
        let q = qkv
            .narrow(1, 0, self.q_size)
            .map_err(ModelError::Candle)?
            .reshape((num_tokens, self.num_q_heads, self.head_dim))
            .map_err(ModelError::Candle)?;
        let k = qkv
            .narrow(1, self.q_size, self.kv_size)
            .map_err(ModelError::Candle)?
            .reshape((num_tokens, self.num_kv_heads, self.head_dim))
            .map_err(ModelError::Candle)?;
        let v = qkv
            .narrow(1, self.q_size + self.kv_size, self.kv_size)
            .map_err(ModelError::Candle)?
            .reshape((num_tokens, self.num_kv_heads, self.head_dim))
            .map_err(ModelError::Candle)?;
        Ok((q, k, v))
    }

    /// Forward pass.
    ///
    /// * `hidden_states` — shape `[num_tokens, hidden_size]`
    /// * `positions` — shape `[num_tokens]`
    /// * `kv_cache` — optional per-layer KV handle. When `Some`, new K/V
    ///   are concatenated with the cache and stored back. Supports both
    ///   contiguous and paged block pool storage.
    ///
    /// Returns shape `[num_tokens, hidden_size]`.
    pub fn forward(
        &self,
        hidden_states: &Tensor,
        positions: &Tensor,
        kv_cache: Option<crate::LayerKvHandle<'_>>,
    ) -> ModelResult<Tensor> {
        let num_tokens = hidden_states.dim(0).map_err(ModelError::Candle)?;

        // Fused QKV projection (single matmul).
        let qkv = self
            .qkv_proj
            .forward(hidden_states)
            .map_err(ModelError::Candle)?;
        let (q, k, v) = self.split_qkv(&qkv, num_tokens)?;

        // Apply RoPE.
        let (q, k) = self.rotary_emb.apply(&q, &k, positions)?;

        // Cache-merge + attention (paged decode reads blocks directly).
        let attn_output =
            attention_with_cache(&q, &k, &v, self.scale, kv_cache, self.sliding_window)?;

        // Reshape back to [num_tokens, num_q_heads * head_dim].
        let attn_output = attn_output
            .reshape((num_tokens, self.q_size))
            .map_err(ModelError::Candle)?;

        // Output projection.
        self.o_proj
            .forward(&attn_output)
            .map_err(ModelError::Candle)
    }

    /// Batched forward: fused QKV + RoPE on all tokens, per-request attention loop.
    pub fn forward_batch(
        &self,
        hidden_states: &Tensor,
        positions: &Tensor,
        attn_meta: &crate::AttentionMetadata,
        storage: &mut crate::BatchedKvCacheStorage<'_>,
    ) -> ModelResult<Tensor> {
        let total_tokens = hidden_states.dim(0).map_err(ModelError::Candle)?;

        // Fused QKV projection on ALL tokens at once (single matmul).
        let (q, k, v) = {
            let _p = vllm_kernels::profiling::range("qkv_proj");
            let qkv = self
                .qkv_proj
                .forward(hidden_states)
                .map_err(ModelError::Candle)?;
            self.split_qkv(&qkv, total_tokens)?
        };

        // Batched RoPE on all tokens.
        // On CUDA: fused kernel (1 launch per tensor vs 5-7 from candle decomposition).
        let (q, k) = {
            let _p = vllm_kernels::profiling::range("rope");
            #[cfg(feature = "cuda")]
            if q.device().is_cuda() {
                crate::ops::rotary_embedding(
                    &q,
                    &k,
                    positions,
                    self.rotary_emb.cos_sin_cache(),
                    self.rotary_emb.head_dim(),
                )
                .map_err(ModelError::Candle)?
            } else {
                self.rotary_emb.apply(&q, &k, positions)?
            }
            #[cfg(not(feature = "cuda"))]
            self.rotary_emb.apply(&q, &k, positions)?
        };

        // Batched FA2 on CUDA F16/BF16: single flash_attn_varlen call across all requests.
        // Falls back to per-request attention_with_cache loop on CPU/F32.
        #[cfg(feature = "cuda")]
        let use_batched_flash =
            q.device().is_cuda() && matches!(q.dtype(), DType::F16 | DType::BF16);
        #[cfg(not(feature = "cuda"))]
        let use_batched_flash = false;

        let attn_output = {
            let _p = vllm_kernels::profiling::range("flash_attn");
            if use_batched_flash {
                #[cfg(feature = "cuda")]
                {
                    let config = crate::attention::BatchedAttnConfig {
                        scale: self.scale,
                        layer_idx: self.layer_idx,
                        sliding_window: self.sliding_window,
                    };
                    crate::attention::batched_flash_attention_with_cache(
                        &q, &k, &v, &config, attn_meta, storage,
                    )?
                }
                #[cfg(not(feature = "cuda"))]
                {
                    unreachable!()
                }
            } else {
                // Per-request attention loop (CPU / F32 path).
                let mut output_parts = Vec::with_capacity(attn_meta.num_reqs);
                for req_idx in 0..attn_meta.num_reqs {
                    let (start, q_len) = attn_meta.request_slice(req_idx);
                    let q_req = q.narrow(0, start, q_len).map_err(ModelError::Candle)?;
                    let k_req = k.narrow(0, start, q_len).map_err(ModelError::Candle)?;
                    let v_req = v.narrow(0, start, q_len).map_err(ModelError::Candle)?;
                    let handle = storage.request_layer_handle(req_idx, self.layer_idx);
                    let out = attention_with_cache(
                        &q_req,
                        &k_req,
                        &v_req,
                        self.scale,
                        Some(handle),
                        self.sliding_window,
                    )?;
                    output_parts.push(out);
                }
                Tensor::cat(&output_parts, 0).map_err(ModelError::Candle)?
            }
        };

        // Reshape back to [total_tokens, num_q_heads * head_dim].
        let attn_output = attn_output
            .reshape((total_tokens, self.q_size))
            .map_err(ModelError::Candle)?;

        // Batched output projection.
        let _p = vllm_kernels::profiling::range("o_proj");
        self.o_proj
            .forward(&attn_output)
            .map_err(ModelError::Candle)
    }

    /// Inject LoRA weights into attention projections.
    pub fn inject_lora(&mut self, prefix: &str, adapter: &LoraAdapter) -> ModelResult<()> {
        let targets = &adapter.config.target_modules;
        if targets
            .iter()
            .any(|t| t == "q_proj" || t == "k_proj" || t == "v_proj")
        {
            eprintln!(
                "WARN: LoRA on fused QKV projection not yet supported; \
                 q_proj/k_proj/v_proj LoRA will be ignored"
            );
        }
        // o_proj is RowParallelLinear.
        if targets.iter().any(|t| t == "o_proj") {
            let key = format!("{}.o_proj", prefix);
            if let Some((a, b)) = adapter.weights.get(&key) {
                self.o_proj
                    .inner_mut()
                    .attach_lora(a.clone(), b.clone(), adapter.scaling)?;
            }
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// LlamaDecoderLayer
// ---------------------------------------------------------------------------

/// A single LLaMA decoder layer.
///
/// Applies: input_layernorm → attention → residual → post_attention_layernorm → MLP → residual
///
/// Port of: `vllm/model_executor/models/llama.py::LlamaDecoderLayer`
pub struct LlamaDecoderLayer {
    self_attn: LlamaAttention,
    mlp: LlamaMLP,
    input_layernorm: RmsNorm,
    post_attention_layernorm: RmsNorm,
}

impl LlamaDecoderLayer {
    /// Load a decoder layer.
    ///
    /// Weight names under `{prefix}`:
    /// - `self_attn.{q,k,v,o}_proj`
    /// - `mlp.{gate,up,down}_proj`
    /// - `input_layernorm`
    /// - `post_attention_layernorm`
    #[allow(clippy::too_many_arguments)]
    pub fn load(
        weights: &ModelWeights,
        prefix: &str,
        config: &LlamaConfig,
        dtype: DType,
        device: &Device,
        rank: usize,
        world_size: usize,
        layer_idx: usize,
    ) -> ModelResult<Self> {
        let self_attn = LlamaAttention::load(
            weights,
            &format!("{}.self_attn", prefix),
            config,
            dtype,
            device,
            rank,
            world_size,
            layer_idx,
        )?;
        let mlp = LlamaMLP::load(weights, &format!("{}.mlp", prefix), dtype, rank, world_size)?;
        let input_layernorm = RmsNorm::load(
            weights,
            &format!("{}.input_layernorm", prefix),
            config.rms_norm_eps,
            dtype,
        )?;
        let post_attention_layernorm = RmsNorm::load(
            weights,
            &format!("{}.post_attention_layernorm", prefix),
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

    /// Forward pass.
    ///
    /// * `hidden_states` — shape `[num_tokens, hidden_size]`
    /// * `positions` — shape `[num_tokens]`
    /// * `kv_cache` — optional per-layer KV handle for this layer's attention
    ///
    /// Returns hidden_states of same shape.
    pub fn forward(
        &self,
        hidden_states: &Tensor,
        positions: &Tensor,
        kv_cache: Option<crate::LayerKvHandle<'_>>,
    ) -> ModelResult<Tensor> {
        // Pre-attention layernorm + attention.
        let normed = crate::ops::rms_norm(hidden_states, &self.input_layernorm)
            .map_err(ModelError::Candle)?;
        let attn_output = self.self_attn.forward(&normed, positions, kv_cache)?;

        // Fused residual add + post-attention layernorm.
        let (normed, hidden_states) = crate::ops::fused_add_rms_norm(
            &attn_output,
            hidden_states,
            &self.post_attention_layernorm,
        )
        .map_err(ModelError::Candle)?;

        // MLP + residual.
        let mlp_output = self.mlp.forward(&normed).map_err(ModelError::Candle)?;
        let hidden_states = (hidden_states + mlp_output).map_err(ModelError::Candle)?;

        Ok(hidden_states)
    }

    /// Batched forward: norms and MLP on all tokens, attention per-request.
    pub fn forward_batch(
        &self,
        hidden_states: &Tensor,
        positions: &Tensor,
        attn_meta: &crate::AttentionMetadata,
        storage: &mut crate::BatchedKvCacheStorage<'_>,
    ) -> ModelResult<Tensor> {
        // Batched pre-attention layernorm.
        let normed = {
            let _p = vllm_kernels::profiling::range("pre_norm");
            crate::ops::rms_norm(hidden_states, &self.input_layernorm)
                .map_err(ModelError::Candle)?
        };
        // Batched Q/K/V + RoPE, per-request attention, batched o_proj.
        let attn_output = {
            let _p = vllm_kernels::profiling::range("attn");
            self.self_attn
                .forward_batch(&normed, positions, attn_meta, storage)?
        };

        // Fused residual add + post-attention layernorm.
        let (normed, hidden_states) = {
            let _p = vllm_kernels::profiling::range("post_norm");
            crate::ops::fused_add_rms_norm(
                &attn_output,
                hidden_states,
                &self.post_attention_layernorm,
            )
            .map_err(ModelError::Candle)?
        };

        // Batched MLP + residual.
        let mlp_output = {
            let _p = vllm_kernels::profiling::range("mlp");
            self.mlp.forward(&normed).map_err(ModelError::Candle)?
        };
        let hidden_states = (hidden_states + mlp_output).map_err(ModelError::Candle)?;

        Ok(hidden_states)
    }
}

// ---------------------------------------------------------------------------
// LlamaModel
// ---------------------------------------------------------------------------

/// LLaMA transformer backbone.
///
/// Embedding → N decoder layers → final RMS norm.
///
/// Port of: `vllm/model_executor/models/llama.py::LlamaModel`
pub struct LlamaModel {
    pub(crate) embed_tokens: Embedding,
    layers: Vec<LlamaDecoderLayer>,
    norm: RmsNorm,
}

impl LlamaModel {
    /// Load the model backbone.
    pub fn load(
        weights: &ModelWeights,
        prefix: &str,
        config: &LlamaConfig,
        dtype: DType,
        device: &Device,
        rank: usize,
        world_size: usize,
    ) -> ModelResult<Self> {
        let embed_tokens = Embedding::load(weights, &format!("{}.embed_tokens", prefix), dtype)?;

        let mut layers = Vec::with_capacity(config.num_hidden_layers);
        for i in 0..config.num_hidden_layers {
            let layer = LlamaDecoderLayer::load(
                weights,
                &format!("{}.layers.{}", prefix, i),
                config,
                dtype,
                device,
                rank,
                world_size,
                i,
            )?;
            layers.push(layer);
        }

        let norm = RmsNorm::load(
            weights,
            &format!("{}.norm", prefix),
            config.rms_norm_eps,
            dtype,
        )?;

        Ok(Self {
            embed_tokens,
            layers,
            norm,
        })
    }

    /// Embed token IDs into hidden states.
    pub fn embed(&self, input_ids: &Tensor) -> ModelResult<Tensor> {
        self.embed_tokens
            .forward(input_ids)
            .map_err(ModelError::Candle)
    }

    /// Run the transformer backbone on pre-computed embeddings.
    pub fn backbone(
        &self,
        mut hidden_states: Tensor,
        positions: &Tensor,
        mut kv_cache: Option<&mut crate::KvCacheStorage<'_>>,
    ) -> ModelResult<Tensor> {
        for (i, layer) in self.layers.iter().enumerate() {
            let layer_handle = kv_cache.as_mut().map(|s| s.layer_handle(i));
            hidden_states = layer.forward(&hidden_states, positions, layer_handle)?;
        }
        crate::ops::rms_norm(&hidden_states, &self.norm).map_err(ModelError::Candle)
    }

    /// Forward pass.
    ///
    /// * `input_ids` — shape `[num_tokens]`
    /// * `positions` — shape `[num_tokens]`
    /// * `kv_cache` — optional KV cache storage (contiguous or paged)
    ///
    /// Returns hidden states of shape `[num_tokens, hidden_size]`.
    pub fn forward(
        &self,
        input_ids: &Tensor,
        positions: &Tensor,
        mut kv_cache: Option<&mut crate::KvCacheStorage<'_>>,
    ) -> ModelResult<Tensor> {
        let mut hidden_states = self
            .embed_tokens
            .forward(input_ids)
            .map_err(ModelError::Candle)?;

        for (i, layer) in self.layers.iter().enumerate() {
            let layer_handle = kv_cache.as_mut().map(|s| s.layer_handle(i));
            hidden_states = layer.forward(&hidden_states, positions, layer_handle)?;
        }

        crate::ops::rms_norm(&hidden_states, &self.norm).map_err(ModelError::Candle)
    }

    /// Batched forward pass.
    pub fn forward_batch(
        &self,
        input_ids: &Tensor,
        positions: &Tensor,
        attn_meta: &crate::AttentionMetadata,
        kv_storage: &mut crate::BatchedKvCacheStorage<'_>,
    ) -> ModelResult<Tensor> {
        // Batched embedding.
        let mut hidden_states = {
            let _p = vllm_kernels::profiling::range("embed");
            self.embed_tokens
                .forward(input_ids)
                .map_err(ModelError::Candle)?
        };

        // Batched layer forward.
        for (i, layer) in self.layers.iter().enumerate() {
            let _p = vllm_kernels::profiling::range_fmt(format_args!("layer_{i}"));
            hidden_states =
                layer.forward_batch(&hidden_states, positions, attn_meta, kv_storage)?;
        }

        // Batched final norm.
        let _p = vllm_kernels::profiling::range("final_norm");
        crate::ops::rms_norm(&hidden_states, &self.norm).map_err(ModelError::Candle)
    }

    /// Number of decoder layers.
    pub fn num_layers(&self) -> usize {
        self.layers.len()
    }
}

// ---------------------------------------------------------------------------
// LlamaForCausalLM
// ---------------------------------------------------------------------------

/// LLaMA for causal language modeling.
///
/// Wraps `LlamaModel` with a language model head that projects hidden states
/// to vocabulary logits.
///
/// Port of: `vllm/model_executor/models/llama.py::LlamaForCausalLM`
pub struct LlamaForCausalLM {
    pub(crate) model: LlamaModel,
    pub(crate) lm_head: Linear,
}

impl LlamaForCausalLM {
    /// Load the full model from weights.
    pub fn load(
        weights: &ModelWeights,
        config: &LlamaConfig,
        dtype: DType,
        device: &Device,
        rank: usize,
        world_size: usize,
    ) -> ModelResult<Self> {
        let model = LlamaModel::load(weights, "model", config, dtype, device, rank, world_size)?;

        let lm_head = if config.tie_word_embeddings {
            // Reuse the embedding weight as the lm_head weight.
            Linear::new(model.embed_tokens.weight().clone(), None)
        } else {
            Linear::load(weights, "lm_head", dtype)?
        };

        Ok(Self { model, lm_head })
    }

    /// Compute logits from hidden states.
    pub fn compute_logits(&self, hidden_states: &Tensor) -> ModelResult<Tensor> {
        self.lm_head
            .forward(hidden_states)
            .map_err(ModelError::Candle)
    }

    /// Access the underlying model backbone.
    pub fn model(&self) -> &LlamaModel {
        &self.model
    }

    /// Access the config that was used to build this model.
    pub fn lm_head(&self) -> &Linear {
        &self.lm_head
    }
}

impl crate::Model for LlamaForCausalLM {
    fn inject_lora(&mut self, adapter: &LoraAdapter) -> ModelResult<()> {
        for (i, layer) in self.model.layers.iter_mut().enumerate() {
            let attn_prefix = format!("model.layers.{}.self_attn", i);
            layer.self_attn.inject_lora(&attn_prefix, adapter)?;

            let mlp_prefix = format!("model.layers.{}.mlp", i);
            layer.mlp.inject_lora(&mlp_prefix, adapter)?;
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
        let logits = self.compute_logits(&hidden_states)?;
        // Cast logits to f32 for sampling (sampler expects f32).
        logits.to_dtype(DType::F32).map_err(ModelError::Candle)
    }

    fn forward_embeds(
        &self,
        inputs_embeds: &Tensor,
        positions: &Tensor,
        kv_cache: Option<&mut crate::KvCacheStorage<'_>>,
    ) -> ModelResult<Tensor> {
        let hidden_states = self
            .model
            .backbone(inputs_embeds.clone(), positions, kv_cache)?;
        let logits = self.compute_logits(&hidden_states)?;
        logits.to_dtype(DType::F32).map_err(ModelError::Candle)
    }

    fn num_layers(&self) -> usize {
        self.model.num_layers()
    }

    fn hidden_states(&self, input_ids: &Tensor, positions: &Tensor) -> ModelResult<Tensor> {
        self.model.forward(input_ids, positions, None)
    }

    fn forward_batch(
        &self,
        input_ids: &Tensor,
        positions: &Tensor,
        attn_meta: &crate::AttentionMetadata,
        kv_storage: &mut crate::BatchedKvCacheStorage<'_>,
    ) -> ModelResult<Tensor> {
        let hidden_states = self
            .model
            .forward_batch(input_ids, positions, attn_meta, kv_storage)?;
        let logits = {
            let _p = vllm_kernels::profiling::range("lm_head");
            self.compute_logits(&hidden_states)?
        };
        logits.to_dtype(DType::F32).map_err(ModelError::Candle)
    }
}

/// Factory function for the model registry.
pub fn create_llama(
    weights: &ModelWeights,
    config: &HfModelConfig,
    dtype: DType,
    device: &Device,
    rank: usize,
    world_size: usize,
) -> ModelResult<Box<dyn crate::Model>> {
    let llama_config = LlamaConfig::from_hf_config(config)?;
    let model = LlamaForCausalLM::load(weights, &llama_config, dtype, device, rank, world_size)?;
    Ok(Box::new(model))
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn test_config() -> LlamaConfig {
        LlamaConfig {
            hidden_size: 32,
            num_attention_heads: 4,
            num_kv_heads: 4,
            num_hidden_layers: 2,
            intermediate_size: 64,
            vocab_size: 100,
            max_position_embeddings: 128,
            rms_norm_eps: 1e-5,
            rope_theta: 10000.0,
            head_dim: 8,
            tie_word_embeddings: false,
            sliding_window: None,
            partial_rotary_factor: 1.0,
            long_rope_scaling: None,
        }
    }

    fn test_config_gqa() -> LlamaConfig {
        LlamaConfig {
            num_kv_heads: 2, // GQA: 4 q heads, 2 kv heads
            ..test_config()
        }
    }

    #[test]
    fn test_llama_config_from_hf() {
        let hf_config: HfModelConfig = serde_json::from_str(
            r#"{
                "architectures": ["LlamaForCausalLM"],
                "model_type": "llama",
                "hidden_size": 4096,
                "num_attention_heads": 32,
                "num_key_value_heads": 8,
                "num_hidden_layers": 32,
                "intermediate_size": 11008,
                "vocab_size": 32000,
                "max_position_embeddings": 4096,
                "rms_norm_eps": 1e-5,
                "rope_theta": 10000.0,
                "tie_word_embeddings": false
            }"#,
        )
        .unwrap();

        let config = LlamaConfig::from_hf_config(&hf_config).unwrap();
        assert_eq!(config.hidden_size, 4096);
        assert_eq!(config.num_attention_heads, 32);
        assert_eq!(config.num_kv_heads, 8);
        assert_eq!(config.num_hidden_layers, 32);
        assert_eq!(config.intermediate_size, 11008);
        assert_eq!(config.vocab_size, 32000);
        assert_eq!(config.head_dim, 128);
        assert!(!config.tie_word_embeddings);
    }

    #[test]
    fn test_phi4_mini_config_from_hf() {
        let hf_config: HfModelConfig = serde_json::from_str(
            r#"{
                "architectures": ["Phi3ForCausalLM"],
                "model_type": "phi3",
                "hidden_size": 3072,
                "num_attention_heads": 24,
                "num_key_value_heads": 8,
                "num_hidden_layers": 32,
                "intermediate_size": 8192,
                "vocab_size": 200064,
                "max_position_embeddings": 131072,
                "rms_norm_eps": 1e-5,
                "rope_theta": 10000.0,
                "tie_word_embeddings": true,
                "partial_rotary_factor": 0.75,
                "original_max_position_embeddings": 4096,
                "rope_scaling": {
                    "type": "longrope",
                    "short_factor": [1.0, 1.0, 1.0],
                    "long_factor": [2.0, 4.0, 8.0]
                }
            }"#,
        )
        .unwrap();

        let config = LlamaConfig::from_hf_config(&hf_config).unwrap();
        assert_eq!(config.hidden_size, 3072);
        assert_eq!(config.head_dim, 128); // 3072 / 24
        assert!((config.partial_rotary_factor - 0.75).abs() < 1e-6);
        assert!(config.long_rope_scaling.is_some());

        let lr = config.long_rope_scaling.unwrap();
        assert_eq!(lr.original_max_position_embeddings, 4096);
        assert_eq!(lr.short_factor.len(), 3);
        assert_eq!(lr.long_factor.len(), 3);
        assert!((lr.long_factor[0] - 2.0).abs() < 1e-6);
        assert!(config.tie_word_embeddings);
    }

    #[test]
    fn test_llama_mlp_forward_zeros() {
        let config = test_config();
        let mlp = LlamaMLP::zeros(
            config.hidden_size,
            config.intermediate_size,
            DType::F32,
            &Device::Cpu,
        )
        .unwrap();

        let x = Tensor::ones(&[3, config.hidden_size], DType::F32, &Device::Cpu).unwrap();
        let out = mlp.forward(&x).unwrap();
        assert_eq!(out.dims(), &[3, config.hidden_size]);
    }

    #[test]
    fn test_llama_attention_forward() {
        let config = test_config();
        let attn = LlamaAttention::zeros(&config, DType::F32, &Device::Cpu, 0).unwrap();

        let num_tokens = 4;
        let x = Tensor::ones(&[num_tokens, config.hidden_size], DType::F32, &Device::Cpu).unwrap();
        let positions = Tensor::new(&[0u32, 1, 2, 3], &Device::Cpu).unwrap();

        let out = attn.forward(&x, &positions, None).unwrap();
        assert_eq!(out.dims(), &[num_tokens, config.hidden_size]);
    }

    #[test]
    fn test_llama_attention_gqa() {
        let config = test_config_gqa();
        let attn = LlamaAttention::zeros(&config, DType::F32, &Device::Cpu, 0).unwrap();

        let num_tokens = 3;
        let x = Tensor::ones(&[num_tokens, config.hidden_size], DType::F32, &Device::Cpu).unwrap();
        let positions = Tensor::new(&[0u32, 1, 2], &Device::Cpu).unwrap();

        let out = attn.forward(&x, &positions, None).unwrap();
        assert_eq!(out.dims(), &[num_tokens, config.hidden_size]);
    }

    #[test]
    fn test_llama_attention_single_token() {
        let config = test_config();
        let attn = LlamaAttention::zeros(&config, DType::F32, &Device::Cpu, 0).unwrap();

        let x = Tensor::ones(&[1, config.hidden_size], DType::F32, &Device::Cpu).unwrap();
        let positions = Tensor::new(&[0u32], &Device::Cpu).unwrap();

        let out = attn.forward(&x, &positions, None).unwrap();
        assert_eq!(out.dims(), &[1, config.hidden_size]);
    }

    #[test]
    fn test_llama_model_from_weights() {
        // Build a tiny LLaMA model from synthetic weights.
        let config = test_config();
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("model.safetensors");

        let device = Device::Cpu;
        let dtype = DType::F32;

        // Generate all the weight tensors needed.
        let mut tensor_specs: Vec<(String, Vec<usize>)> = Vec::new();

        // Embeddings.
        tensor_specs.push((
            "model.embed_tokens.weight".to_string(),
            vec![config.vocab_size, config.hidden_size],
        ));

        // Layers.
        for i in 0..config.num_hidden_layers {
            let prefix = format!("model.layers.{}", i);
            let q_size = config.num_attention_heads * config.head_dim;
            let kv_size = config.num_kv_heads * config.head_dim;

            tensor_specs.push((
                format!("{}.self_attn.q_proj.weight", prefix),
                vec![q_size, config.hidden_size],
            ));
            tensor_specs.push((
                format!("{}.self_attn.k_proj.weight", prefix),
                vec![kv_size, config.hidden_size],
            ));
            tensor_specs.push((
                format!("{}.self_attn.v_proj.weight", prefix),
                vec![kv_size, config.hidden_size],
            ));
            tensor_specs.push((
                format!("{}.self_attn.o_proj.weight", prefix),
                vec![config.hidden_size, q_size],
            ));

            tensor_specs.push((
                format!("{}.mlp.gate_proj.weight", prefix),
                vec![config.intermediate_size, config.hidden_size],
            ));
            tensor_specs.push((
                format!("{}.mlp.up_proj.weight", prefix),
                vec![config.intermediate_size, config.hidden_size],
            ));
            tensor_specs.push((
                format!("{}.mlp.down_proj.weight", prefix),
                vec![config.hidden_size, config.intermediate_size],
            ));

            tensor_specs.push((
                format!("{}.input_layernorm.weight", prefix),
                vec![config.hidden_size],
            ));
            tensor_specs.push((
                format!("{}.post_attention_layernorm.weight", prefix),
                vec![config.hidden_size],
            ));
        }

        // Final norm.
        tensor_specs.push(("model.norm.weight".to_string(), vec![config.hidden_size]));

        // LM head.
        tensor_specs.push((
            "lm_head.weight".to_string(),
            vec![config.vocab_size, config.hidden_size],
        ));

        // Create safetensors file with small random-ish weights.
        create_test_weights(&path, &tensor_specs);

        let weights = ModelWeights::from_single_file(&path, &device).unwrap();
        let model = LlamaForCausalLM::load(&weights, &config, dtype, &device, 0, 1).unwrap();

        // Forward pass.
        let input_ids = Tensor::new(&[1u32, 5, 10], &device).unwrap();
        let positions = Tensor::new(&[0u32, 1, 2], &device).unwrap();

        let logits = crate::Model::forward(&model, &input_ids, &positions, None).unwrap();
        assert_eq!(logits.dims(), &[3, config.vocab_size]);
    }

    #[test]
    fn test_llama_model_tied_embeddings() {
        let config = LlamaConfig {
            tie_word_embeddings: true,
            ..test_config()
        };

        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("model.safetensors");
        let device = Device::Cpu;
        let dtype = DType::F32;

        let mut tensor_specs: Vec<(String, Vec<usize>)> = Vec::new();
        tensor_specs.push((
            "model.embed_tokens.weight".to_string(),
            vec![config.vocab_size, config.hidden_size],
        ));

        for i in 0..config.num_hidden_layers {
            let prefix = format!("model.layers.{}", i);
            let q_size = config.num_attention_heads * config.head_dim;
            let kv_size = config.num_kv_heads * config.head_dim;

            tensor_specs.push((
                format!("{}.self_attn.q_proj.weight", prefix),
                vec![q_size, config.hidden_size],
            ));
            tensor_specs.push((
                format!("{}.self_attn.k_proj.weight", prefix),
                vec![kv_size, config.hidden_size],
            ));
            tensor_specs.push((
                format!("{}.self_attn.v_proj.weight", prefix),
                vec![kv_size, config.hidden_size],
            ));
            tensor_specs.push((
                format!("{}.self_attn.o_proj.weight", prefix),
                vec![config.hidden_size, q_size],
            ));

            tensor_specs.push((
                format!("{}.mlp.gate_proj.weight", prefix),
                vec![config.intermediate_size, config.hidden_size],
            ));
            tensor_specs.push((
                format!("{}.mlp.up_proj.weight", prefix),
                vec![config.intermediate_size, config.hidden_size],
            ));
            tensor_specs.push((
                format!("{}.mlp.down_proj.weight", prefix),
                vec![config.hidden_size, config.intermediate_size],
            ));

            tensor_specs.push((
                format!("{}.input_layernorm.weight", prefix),
                vec![config.hidden_size],
            ));
            tensor_specs.push((
                format!("{}.post_attention_layernorm.weight", prefix),
                vec![config.hidden_size],
            ));
        }

        tensor_specs.push(("model.norm.weight".to_string(), vec![config.hidden_size]));
        // No lm_head.weight — tied embeddings!

        create_test_weights(&path, &tensor_specs);

        let weights = ModelWeights::from_single_file(&path, &device).unwrap();
        let model = LlamaForCausalLM::load(&weights, &config, dtype, &device, 0, 1).unwrap();

        let input_ids = Tensor::new(&[1u32, 5], &device).unwrap();
        let positions = Tensor::new(&[0u32, 1], &device).unwrap();
        let logits = crate::Model::forward(&model, &input_ids, &positions, None).unwrap();
        assert_eq!(logits.dims(), &[2, config.vocab_size]);
    }

    #[test]
    fn test_llama_registry() {
        let registry = crate::ModelRegistry::default_registry();
        assert!(registry.contains("LlamaForCausalLM"));

        let factory = registry.get("LlamaForCausalLM").unwrap();

        // Create a tiny model via the factory.
        let config = test_config();
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("model.safetensors");
        let device = Device::Cpu;

        let mut tensor_specs: Vec<(String, Vec<usize>)> = Vec::new();
        tensor_specs.push((
            "model.embed_tokens.weight".to_string(),
            vec![config.vocab_size, config.hidden_size],
        ));
        for i in 0..config.num_hidden_layers {
            let prefix = format!("model.layers.{}", i);
            let q_size = config.num_attention_heads * config.head_dim;
            let kv_size = config.num_kv_heads * config.head_dim;
            tensor_specs.push((
                format!("{}.self_attn.q_proj.weight", prefix),
                vec![q_size, config.hidden_size],
            ));
            tensor_specs.push((
                format!("{}.self_attn.k_proj.weight", prefix),
                vec![kv_size, config.hidden_size],
            ));
            tensor_specs.push((
                format!("{}.self_attn.v_proj.weight", prefix),
                vec![kv_size, config.hidden_size],
            ));
            tensor_specs.push((
                format!("{}.self_attn.o_proj.weight", prefix),
                vec![config.hidden_size, q_size],
            ));
            tensor_specs.push((
                format!("{}.mlp.gate_proj.weight", prefix),
                vec![config.intermediate_size, config.hidden_size],
            ));
            tensor_specs.push((
                format!("{}.mlp.up_proj.weight", prefix),
                vec![config.intermediate_size, config.hidden_size],
            ));
            tensor_specs.push((
                format!("{}.mlp.down_proj.weight", prefix),
                vec![config.hidden_size, config.intermediate_size],
            ));
            tensor_specs.push((
                format!("{}.input_layernorm.weight", prefix),
                vec![config.hidden_size],
            ));
            tensor_specs.push((
                format!("{}.post_attention_layernorm.weight", prefix),
                vec![config.hidden_size],
            ));
        }
        tensor_specs.push(("model.norm.weight".to_string(), vec![config.hidden_size]));
        tensor_specs.push((
            "lm_head.weight".to_string(),
            vec![config.vocab_size, config.hidden_size],
        ));
        create_test_weights(&path, &tensor_specs);

        let hf_config: HfModelConfig = serde_json::from_str(&format!(
            r#"{{
                "architectures": ["LlamaForCausalLM"],
                "hidden_size": {},
                "num_attention_heads": {},
                "num_key_value_heads": {},
                "num_hidden_layers": {},
                "intermediate_size": {},
                "vocab_size": {},
                "max_position_embeddings": {},
                "rms_norm_eps": 1e-5,
                "rope_theta": 10000.0,
                "tie_word_embeddings": false
            }}"#,
            config.hidden_size,
            config.num_attention_heads,
            config.num_kv_heads,
            config.num_hidden_layers,
            config.intermediate_size,
            config.vocab_size,
            config.max_position_embeddings,
        ))
        .unwrap();

        let weights = ModelWeights::from_single_file(&path, &device).unwrap();
        let model = factory(&weights, &hf_config, DType::F32, &device, 0, 1).unwrap();

        let input_ids = Tensor::new(&[1u32, 2, 3], &device).unwrap();
        let positions = Tensor::new(&[0u32, 1, 2], &device).unwrap();
        let logits = model.forward(&input_ids, &positions, None).unwrap();
        assert_eq!(logits.dims(), &[3, config.vocab_size]);
    }

    #[test]
    fn test_llama_kv_cache_prefill_and_decode() {
        // Verify that KV cache is populated on prefill and used on decode,
        // and that the decode output shape is correct.
        let config = test_config();
        let attn = LlamaAttention::zeros(&config, DType::F32, &Device::Cpu, 0).unwrap();

        // Prefill: 4 tokens.
        let x = Tensor::ones(&[4, config.hidden_size], DType::F32, &Device::Cpu).unwrap();
        let positions = Tensor::new(&[0u32, 1, 2, 3], &Device::Cpu).unwrap();
        let mut cache: Option<(Tensor, Tensor)> = None;

        let handle = crate::LayerKvHandle::Contiguous(&mut cache);
        let out = attn.forward(&x, &positions, Some(handle)).unwrap();
        assert_eq!(out.dims(), &[4, config.hidden_size]);

        // Cache should now hold K/V of length 4.
        let (cached_k, cached_v) = cache.as_ref().unwrap();
        assert_eq!(cached_k.dim(0).unwrap(), 4);
        assert_eq!(cached_v.dim(0).unwrap(), 4);

        // Decode: 1 token at position 4.
        let x_decode = Tensor::ones(&[1, config.hidden_size], DType::F32, &Device::Cpu).unwrap();
        let pos_decode = Tensor::new(&[4u32], &Device::Cpu).unwrap();

        let handle = crate::LayerKvHandle::Contiguous(&mut cache);
        let out_decode = attn.forward(&x_decode, &pos_decode, Some(handle)).unwrap();
        assert_eq!(out_decode.dims(), &[1, config.hidden_size]);

        // Cache should now hold K/V of length 5.
        let (cached_k, cached_v) = cache.as_ref().unwrap();
        assert_eq!(cached_k.dim(0).unwrap(), 5);
        assert_eq!(cached_v.dim(0).unwrap(), 5);
    }

    #[test]
    fn test_llama_model_kv_cache_e2e() {
        // End-to-end: run a tiny model with KV cache through prefill + decode.
        let config = test_config();
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("model.safetensors");
        let device = Device::Cpu;
        let dtype = DType::F32;

        let mut tensor_specs: Vec<(String, Vec<usize>)> = Vec::new();
        tensor_specs.push((
            "model.embed_tokens.weight".to_string(),
            vec![config.vocab_size, config.hidden_size],
        ));
        for i in 0..config.num_hidden_layers {
            let prefix = format!("model.layers.{}", i);
            let q_size = config.num_attention_heads * config.head_dim;
            let kv_size = config.num_kv_heads * config.head_dim;
            tensor_specs.push((
                format!("{}.self_attn.q_proj.weight", prefix),
                vec![q_size, config.hidden_size],
            ));
            tensor_specs.push((
                format!("{}.self_attn.k_proj.weight", prefix),
                vec![kv_size, config.hidden_size],
            ));
            tensor_specs.push((
                format!("{}.self_attn.v_proj.weight", prefix),
                vec![kv_size, config.hidden_size],
            ));
            tensor_specs.push((
                format!("{}.self_attn.o_proj.weight", prefix),
                vec![config.hidden_size, q_size],
            ));
            tensor_specs.push((
                format!("{}.mlp.gate_proj.weight", prefix),
                vec![config.intermediate_size, config.hidden_size],
            ));
            tensor_specs.push((
                format!("{}.mlp.up_proj.weight", prefix),
                vec![config.intermediate_size, config.hidden_size],
            ));
            tensor_specs.push((
                format!("{}.mlp.down_proj.weight", prefix),
                vec![config.hidden_size, config.intermediate_size],
            ));
            tensor_specs.push((
                format!("{}.input_layernorm.weight", prefix),
                vec![config.hidden_size],
            ));
            tensor_specs.push((
                format!("{}.post_attention_layernorm.weight", prefix),
                vec![config.hidden_size],
            ));
        }
        tensor_specs.push(("model.norm.weight".to_string(), vec![config.hidden_size]));
        tensor_specs.push((
            "lm_head.weight".to_string(),
            vec![config.vocab_size, config.hidden_size],
        ));
        create_test_weights(&path, &tensor_specs);

        let weights = ModelWeights::from_single_file(&path, &device).unwrap();
        let model = LlamaForCausalLM::load(&weights, &config, dtype, &device, 0, 1).unwrap();

        // Prefill with 3 tokens and KV cache.
        let mut kv_cache: crate::KvCache = vec![None; model.model().num_layers()];
        let input_ids = Tensor::new(&[1u32, 5, 10], &device).unwrap();
        let positions = Tensor::new(&[0u32, 1, 2], &device).unwrap();
        let mut storage = crate::KvCacheStorage::Contiguous(&mut kv_cache);
        let logits =
            crate::Model::forward(&model, &input_ids, &positions, Some(&mut storage)).unwrap();
        drop(storage);
        assert_eq!(logits.dims(), &[3, config.vocab_size]);

        // All layers should have caches populated.
        for (i, entry) in kv_cache.iter().enumerate() {
            let (k, v) = entry
                .as_ref()
                .unwrap_or_else(|| panic!("layer {i} cache empty"));
            assert_eq!(k.dim(0).unwrap(), 3, "layer {i} K length");
            assert_eq!(v.dim(0).unwrap(), 3, "layer {i} V length");
        }

        // Decode: 1 token at position 3.
        let decode_ids = Tensor::new(&[15u32], &device).unwrap();
        let decode_pos = Tensor::new(&[3u32], &device).unwrap();
        let mut storage = crate::KvCacheStorage::Contiguous(&mut kv_cache);
        let logits2 =
            crate::Model::forward(&model, &decode_ids, &decode_pos, Some(&mut storage)).unwrap();
        drop(storage);
        assert_eq!(logits2.dims(), &[1, config.vocab_size]);

        // Caches should now have length 4.
        for (i, entry) in kv_cache.iter().enumerate() {
            let (k, v) = entry.as_ref().unwrap();
            assert_eq!(k.dim(0).unwrap(), 4, "layer {i} K length after decode");
            assert_eq!(v.dim(0).unwrap(), 4, "layer {i} V length after decode");
        }
    }

    // -----------------------------------------------------------------------
    // Test helper: create a safetensors file with small constant weights.
    // -----------------------------------------------------------------------

    fn create_test_weights(path: &std::path::Path, specs: &[(String, Vec<usize>)]) {
        use safetensors::tensor::TensorView;

        // Generate weight data (small values to avoid numerical issues).
        let mut all_data: Vec<Vec<u8>> = Vec::new();
        for (_, shape) in specs {
            let num_elements: usize = shape.iter().product();
            // Use small deterministic values: 0.01 for all elements.
            // Norm weights use 1.0 so that RmsNorm doesn't distort values.
            let data: Vec<u8> = (0..num_elements)
                .flat_map(|_| 0.01f32.to_le_bytes())
                .collect();
            all_data.push(data);
        }

        // Override norm weights with 1.0.
        for (i, (name, shape)) in specs.iter().enumerate() {
            if name.contains("layernorm") || (name.as_str() == "model.norm.weight") {
                let num_elements: usize = shape.iter().product();
                all_data[i] = (0..num_elements)
                    .flat_map(|_| 1.0f32.to_le_bytes())
                    .collect();
            }
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

    /// Helper: build a full LlamaForCausalLM from the test_config with small weights.
    fn make_test_model() -> (LlamaConfig, LlamaForCausalLM) {
        let config = test_config();
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("model.safetensors");
        let device = Device::Cpu;
        let dtype = DType::F32;

        let mut tensor_specs: Vec<(String, Vec<usize>)> = Vec::new();
        tensor_specs.push((
            "model.embed_tokens.weight".to_string(),
            vec![config.vocab_size, config.hidden_size],
        ));
        for i in 0..config.num_hidden_layers {
            let prefix = format!("model.layers.{}", i);
            let q_size = config.num_attention_heads * config.head_dim;
            let kv_size = config.num_kv_heads * config.head_dim;
            for (name, shape) in [
                (
                    format!("{prefix}.self_attn.q_proj.weight"),
                    vec![q_size, config.hidden_size],
                ),
                (
                    format!("{prefix}.self_attn.k_proj.weight"),
                    vec![kv_size, config.hidden_size],
                ),
                (
                    format!("{prefix}.self_attn.v_proj.weight"),
                    vec![kv_size, config.hidden_size],
                ),
                (
                    format!("{prefix}.self_attn.o_proj.weight"),
                    vec![config.hidden_size, q_size],
                ),
                (
                    format!("{prefix}.mlp.gate_proj.weight"),
                    vec![config.intermediate_size, config.hidden_size],
                ),
                (
                    format!("{prefix}.mlp.up_proj.weight"),
                    vec![config.intermediate_size, config.hidden_size],
                ),
                (
                    format!("{prefix}.mlp.down_proj.weight"),
                    vec![config.hidden_size, config.intermediate_size],
                ),
                (
                    format!("{prefix}.input_layernorm.weight"),
                    vec![config.hidden_size],
                ),
                (
                    format!("{prefix}.post_attention_layernorm.weight"),
                    vec![config.hidden_size],
                ),
            ] {
                tensor_specs.push((name, shape));
            }
        }
        tensor_specs.push(("model.norm.weight".to_string(), vec![config.hidden_size]));
        tensor_specs.push((
            "lm_head.weight".to_string(),
            vec![config.vocab_size, config.hidden_size],
        ));
        create_test_weights(&path, &tensor_specs);

        let weights = ModelWeights::from_single_file(&path, &device).unwrap();
        // Keep the dir alive by leaking it (tests are short-lived).
        std::mem::forget(dir);
        let model = LlamaForCausalLM::load(&weights, &config, dtype, &device, 0, 1).unwrap();
        (config, model)
    }

    #[test]
    fn test_forward_batch_matches_sequential_prefill() {
        // Verify that forward_batch produces the same logits as sequential
        // forward() calls for two prefill requests.
        let (config, model) = make_test_model();
        let device = Device::Cpu;

        // Two requests: 3 tokens and 2 tokens.
        let ids_a = Tensor::new(&[1u32, 2, 3], &device).unwrap();
        let pos_a = Tensor::new(&[0u32, 1, 2], &device).unwrap();
        let ids_b = Tensor::new(&[4u32, 5], &device).unwrap();
        let pos_b = Tensor::new(&[0u32, 1], &device).unwrap();

        // Sequential: two separate forward() calls (no KV cache).
        let logits_a = crate::Model::forward(&model, &ids_a, &pos_a, None).unwrap();
        let logits_b = crate::Model::forward(&model, &ids_b, &pos_b, None).unwrap();

        // Batched: single forward_batch call.
        let flat_ids = Tensor::cat(&[&ids_a, &ids_b], 0).unwrap();
        let flat_pos = Tensor::cat(&[&pos_a, &pos_b], 0).unwrap();

        let mut pool = crate::KvBlockPool::new(
            4,
            config.num_hidden_layers,
            config.num_kv_heads,
            config.head_dim,
            16,
            DType::F32,
            &device,
        )
        .unwrap();
        let attn_meta = crate::AttentionMetadata::new(
            2,
            5,
            vec![0, 3, 5],
            vec![3, 2],
            vec![3, 2],
            vec![vec![0], vec![1]],
            vec![0, 0],
            vec![true, true],
            vec!["a".into(), "b".into()],
        );
        let mut batched =
            crate::BatchedKvCacheStorage::new(&mut pool, vec![vec![0], vec![1]], vec![0, 0]);
        let logits_batched =
            crate::Model::forward_batch(&model, &flat_ids, &flat_pos, &attn_meta, &mut batched)
                .unwrap();
        batched.flush_all().unwrap();

        // Split and compare.
        let batch_a = logits_batched.narrow(0, 0, 3).unwrap();
        let batch_b = logits_batched.narrow(0, 3, 2).unwrap();

        assert_eq!(batch_a.dims(), logits_a.dims());
        assert_eq!(batch_b.dims(), logits_b.dims());

        // Logits should be identical (deterministic ops, same weights).
        let diff_a = (batch_a - logits_a).unwrap().abs().unwrap().max(0).unwrap();
        let max_diff_a: f32 = diff_a.max(0).unwrap().to_scalar().unwrap();
        assert!(
            max_diff_a < 1e-4,
            "Request A logits differ: max_diff={max_diff_a}"
        );

        let diff_b = (batch_b - logits_b).unwrap().abs().unwrap().max(0).unwrap();
        let max_diff_b: f32 = diff_b.max(0).unwrap().to_scalar().unwrap();
        assert!(
            max_diff_b < 1e-4,
            "Request B logits differ: max_diff={max_diff_b}"
        );
    }

    #[test]
    fn test_forward_batch_single_request_matches_forward() {
        // Batch of 1 should produce identical results to forward().
        let (config, model) = make_test_model();
        let device = Device::Cpu;

        let ids = Tensor::new(&[1u32, 2, 3, 4], &device).unwrap();
        let pos = Tensor::new(&[0u32, 1, 2, 3], &device).unwrap();

        let logits_seq = crate::Model::forward(&model, &ids, &pos, None).unwrap();

        let mut pool = crate::KvBlockPool::new(
            2,
            config.num_hidden_layers,
            config.num_kv_heads,
            config.head_dim,
            16,
            DType::F32,
            &device,
        )
        .unwrap();
        let attn_meta = crate::AttentionMetadata::new(
            1,
            4,
            vec![0, 4],
            vec![4],
            vec![4],
            vec![vec![0]],
            vec![0],
            vec![true],
            vec!["r".into()],
        );
        let mut batched = crate::BatchedKvCacheStorage::new(&mut pool, vec![vec![0]], vec![0]);
        let logits_batch =
            crate::Model::forward_batch(&model, &ids, &pos, &attn_meta, &mut batched).unwrap();
        batched.flush_all().unwrap();

        assert_eq!(logits_batch.dims(), logits_seq.dims());
        let diff = (logits_batch - logits_seq)
            .unwrap()
            .abs()
            .unwrap()
            .max(0)
            .unwrap();
        let max_diff: f32 = diff.max(0).unwrap().to_scalar().unwrap();
        assert!(max_diff < 1e-4, "Logits differ: max_diff={max_diff}");
    }
}
