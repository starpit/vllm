// SPDX-License-Identifier: Apache-2.0
// Copyright contributors to the vLLM project

//! LLaMA model architecture for MLX.
//!
//! Uses mlx-rs nn primitives: `Linear`, `RmsNorm`, `Rope`, `Embedding`, and
//! `fast::scaled_dot_product_attention`. All operations are lazy — the entire
//! forward pass builds a compute graph that materializes with a single `eval()`.
//!
//! Also covers Mistral and Qwen2 (same architecture with minor config diffs).

use std::collections::HashMap;
use std::path::Path;

use mlx_rs::builder::Builder;
use mlx_rs::error::Exception;
use mlx_rs::module::{Module, Param};
use mlx_rs::nn;
use mlx_rs::ops::concatenate_axis;
use mlx_rs::ops::indexing::TryIndexOp;
use mlx_rs::{Array, Dtype};

use crate::cache::MlxKvCache;
use vllm_model::weight::HfModelConfig;

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
    pub rms_norm_eps: f32,
    pub rope_theta: f32,
    pub head_dim: usize,
    pub tie_word_embeddings: bool,
    /// Sliding window size for attention. When `Some(w)`, each token only
    /// attends to the most recent `w` positions. Used by Mistral, Qwen2, etc.
    pub sliding_window: Option<usize>,
    /// Fraction of head dimensions that get RoPE (default 1.0). Used by Phi-3/4.
    pub partial_rotary_factor: f64,
    /// LongRoPE scaling parameters. `None` means standard RoPE.
    pub long_rope_scaling: Option<LongRopeScaling>,
}

impl LlamaConfig {
    /// Parse from a HuggingFace config.json.
    pub fn from_hf_config(config: &HfModelConfig) -> Result<Self, String> {
        let hidden_size = config
            .hidden_size
            .ok_or_else(|| "missing hidden_size".to_string())?;
        let num_attention_heads = config
            .num_attention_heads
            .ok_or_else(|| "missing num_attention_heads".to_string())?;

        // Parse sliding_window from config.json extras (used by Mistral, Qwen2, Phi-3, etc.).
        // Handles both scalar (4096) and array ([null, 4096, null, 4096, ...]) formats.
        // The array format is used by newer Mistral models (3.x) — we extract the first
        // non-null value as the scalar sliding window size.
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

        let num_hidden_layers = config
            .num_hidden_layers
            .ok_or_else(|| "missing num_hidden_layers".to_string())?;

        // Validate Qwen2 max_window_layers: if set and < num_hidden_layers, the model
        // wants partial-layer sliding window (only top layers). We don't support
        // that — disable sliding window and warn.
        let sliding_window = if let Some(max_window_layers) = config
            .extra
            .get("max_window_layers")
            .and_then(|v| v.as_u64())
            .map(|v| v as usize)
        {
            if max_window_layers < num_hidden_layers {
                tracing::warn!(
                    "config has max_window_layers={} < num_hidden_layers={}: \
                     partial-layer sliding window not supported, disabling sliding window",
                    max_window_layers,
                    num_hidden_layers
                );
                None
            } else {
                sliding_window
            }
        } else {
            sliding_window
        };

        // Parse partial_rotary_factor (Phi-3/4 family).
        let partial_rotary_factor = config
            .extra
            .get("partial_rotary_factor")
            .and_then(|v| v.as_f64())
            .unwrap_or(1.0);

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
            num_hidden_layers,
            intermediate_size: config
                .intermediate_size
                .ok_or_else(|| "missing intermediate_size".to_string())?,
            vocab_size: config
                .vocab_size
                .ok_or_else(|| "missing vocab_size".to_string())?,
            max_position_embeddings: config.max_position_embeddings.unwrap_or(4096),
            rms_norm_eps: config.norm_eps() as f32,
            rope_theta: config.rope_theta.unwrap_or(10000.0) as f32,
            head_dim: config
                .head_dim()
                .unwrap_or(hidden_size / num_attention_heads),
            tie_word_embeddings: config.tie_word_embeddings.unwrap_or(false),
            sliding_window,
            partial_rotary_factor,
            long_rope_scaling,
        })
    }
}

// ---------------------------------------------------------------------------
// Helper: assign weight from loaded HashMap
// ---------------------------------------------------------------------------

/// Look up a weight by name and assign it. Logs a warning if not found.
pub(crate) fn assign_weight(
    target: &mut Param<Array>,
    weights: &HashMap<String, Array>,
    name: &str,
) {
    if let Some(w) = weights.get(name) {
        target.value = w.clone();
    } else {
        tracing::warn!("Weight not found: {name}");
    }
}

// ---------------------------------------------------------------------------
// MlxLlamaMLP
// ---------------------------------------------------------------------------

/// LLaMA MLP (SiLU-gated feed-forward network) using MLX.
pub struct MlxLlamaMLP {
    pub(crate) gate_proj: nn::Linear,
    pub(crate) up_proj: nn::Linear,
    pub(crate) down_proj: nn::Linear,
}

impl MlxLlamaMLP {
    /// Create with random initialization.
    pub fn new(hidden_size: i32, intermediate_size: i32) -> Result<Self, Exception> {
        Ok(Self {
            gate_proj: nn::LinearBuilder::new(hidden_size, intermediate_size)
                .bias(false)
                .build()?,
            up_proj: nn::LinearBuilder::new(hidden_size, intermediate_size)
                .bias(false)
                .build()?,
            down_proj: nn::LinearBuilder::new(intermediate_size, hidden_size)
                .bias(false)
                .build()?,
        })
    }

    /// Load weights from a flat HashMap.
    pub fn load_weights(&mut self, weights: &HashMap<String, Array>, prefix: &str) {
        assign_weight(
            &mut self.gate_proj.weight,
            weights,
            &format!("{prefix}.gate_proj.weight"),
        );
        assign_weight(
            &mut self.up_proj.weight,
            weights,
            &format!("{prefix}.up_proj.weight"),
        );
        assign_weight(
            &mut self.down_proj.weight,
            weights,
            &format!("{prefix}.down_proj.weight"),
        );
    }

    /// Forward pass: gate_proj(x) → SiLU → * up_proj(x) → down_proj
    pub fn forward(&mut self, x: &Array) -> Result<Array, Exception> {
        let gate = self.gate_proj.forward(x)?;
        let gate = nn::silu(&gate)?;
        let up = self.up_proj.forward(x)?;
        let hidden = gate.multiply(&up)?;
        self.down_proj.forward(&hidden)
    }
}

// ---------------------------------------------------------------------------
// MlxLlamaAttention
// ---------------------------------------------------------------------------

/// LLaMA multi-head attention with RoPE and optional GQA, using MLX.
pub struct MlxLlamaAttention {
    pub(crate) q_proj: nn::Linear,
    pub(crate) k_proj: nn::Linear,
    pub(crate) v_proj: nn::Linear,
    pub(crate) o_proj: nn::Linear,
    /// Optional per-head Q norm (Qwen3 uses this).
    q_norm: Option<nn::RmsNorm>,
    /// Optional per-head K norm (Qwen3 uses this).
    k_norm: Option<nn::RmsNorm>,
    rope: nn::Rope,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    scale: f32,
    sliding_window: Option<usize>,
}

impl MlxLlamaAttention {
    /// Create a new attention layer.
    pub fn new(config: &LlamaConfig) -> Result<Self, Exception> {
        let hidden = config.hidden_size as i32;
        let q_size = (config.num_attention_heads * config.head_dim) as i32;
        let kv_size = (config.num_kv_heads * config.head_dim) as i32;

        Ok(Self {
            q_proj: nn::LinearBuilder::new(hidden, q_size).bias(false).build()?,
            k_proj: nn::LinearBuilder::new(hidden, kv_size)
                .bias(false)
                .build()?,
            v_proj: nn::LinearBuilder::new(hidden, kv_size)
                .bias(false)
                .build()?,
            o_proj: nn::LinearBuilder::new(q_size, hidden).bias(false).build()?,
            q_norm: None,
            k_norm: None,
            rope: {
                let mut r = nn::Rope::new(config.head_dim as i32);
                r.base = config.rope_theta;
                r
            },
            num_heads: config.num_attention_heads,
            num_kv_heads: config.num_kv_heads,
            head_dim: config.head_dim,
            scale: 1.0 / (config.head_dim as f32).sqrt(),
            sliding_window: config.sliding_window,
        })
    }

    /// Load weights from a flat HashMap.
    pub fn load_weights(&mut self, weights: &HashMap<String, Array>, prefix: &str) {
        assign_weight(
            &mut self.q_proj.weight,
            weights,
            &format!("{prefix}.q_proj.weight"),
        );
        assign_weight(
            &mut self.k_proj.weight,
            weights,
            &format!("{prefix}.k_proj.weight"),
        );
        assign_weight(
            &mut self.v_proj.weight,
            weights,
            &format!("{prefix}.v_proj.weight"),
        );
        assign_weight(
            &mut self.o_proj.weight,
            weights,
            &format!("{prefix}.o_proj.weight"),
        );
        // Optional bias (Qwen2 has attention bias, LLaMA/Mistral do not).
        for (proj, name) in [
            (&mut self.q_proj, "q_proj"),
            (&mut self.k_proj, "k_proj"),
            (&mut self.v_proj, "v_proj"),
            (&mut self.o_proj, "o_proj"),
        ] {
            if let Some(b) = weights.get(&format!("{prefix}.{name}.bias")) {
                proj.bias.value = Some(b.clone());
            }
        }

        // Optional QK norms (Qwen3 has per-head q_norm and k_norm).
        if let Some(w) = weights.get(&format!("{prefix}.q_norm.weight")) {
            let head_dim = w.dim(0);
            if let Ok(mut norm) = nn::RmsNormBuilder::new(head_dim).eps(1e-6).build() {
                norm.weight.value = w.clone();
                self.q_norm = Some(norm);
            }
        }
        if let Some(w) = weights.get(&format!("{prefix}.k_norm.weight")) {
            let head_dim = w.dim(0);
            if let Ok(mut norm) = nn::RmsNormBuilder::new(head_dim).eps(1e-6).build() {
                norm.weight.value = w.clone();
                self.k_norm = Some(norm);
            }
        }
    }

    /// Forward pass.
    ///
    /// * `hidden_states` — shape `[seq_len, hidden_size]`
    /// * `positions` — shape `[seq_len]` (min value used as RoPE offset)
    /// * `cache` — optional per-layer KV cache entry
    pub fn forward(
        &mut self,
        hidden_states: &Array,
        positions: &Array,
        cache: &mut Option<(Array, Array)>,
    ) -> Result<Array, Exception> {
        let seq_len = hidden_states.dim(0);

        // Q/K/V projections.
        let q = self.q_proj.forward(hidden_states)?;
        let k = self.k_proj.forward(hidden_states)?;
        let v = self.v_proj.forward(hidden_states)?;

        // Reshape: [seq, hidden] -> [seq, heads, head_dim]
        let q = q.reshape(&[seq_len, self.num_heads as i32, self.head_dim as i32])?;
        let k = k.reshape(&[seq_len, self.num_kv_heads as i32, self.head_dim as i32])?;

        // Apply optional per-head QK norms (Qwen3).
        // RmsNorm normalizes the last dimension, so [seq, heads, head_dim] → per-head norm.
        let q = if let Some(ref mut norm) = self.q_norm {
            norm.forward(&q)?
        } else {
            q
        };
        let k = if let Some(ref mut norm) = self.k_norm {
            norm.forward(&k)?
        } else {
            k
        };

        // [seq, heads, head_dim] -> [1, heads, seq, head_dim]
        let q = q.transpose_axes(&[1, 0, 2])?.expand_dims(0)?;
        let mut k = k.transpose_axes(&[1, 0, 2])?.expand_dims(0)?;
        let mut v = v
            .reshape(&[seq_len, self.num_kv_heads as i32, self.head_dim as i32])?
            .transpose_axes(&[1, 0, 2])?
            .expand_dims(0)?;

        // RoPE: uses positions[0] as offset for proper positional encoding.
        let offset = if positions.size() > 0 {
            positions.reshape(&[-1])?.min(None)?.item::<i32>()
        } else {
            0
        };
        let q = self.rope.forward((&q, offset))?;
        k = self.rope.forward((&k, offset))?;

        // KV cache update (lazy — no actual concat until eval).
        if let Some((ck, cv)) = cache.take() {
            k = concatenate_axis(&[ck, k], 2)?; // concat along seq dim
            v = concatenate_axis(&[cv, v], 2)?;
        }
        // Store the full cache (for future steps).
        *cache = Some((k.clone(), v.clone()));

        // Sliding window: trim K/V to only the last `w` positions.
        // The stored cache remains full (future tokens may still be in window),
        // but we only attend to the windowed subset.
        if let Some(w) = self.sliding_window {
            let kv_len = k.dim(2) as usize;
            if kv_len > w {
                let start = (kv_len - w) as i32;
                let end = kv_len as i32;
                // k, v shape: [1, heads, kv_len, head_dim]
                k = k.try_index((.., .., start..end, ..))?;
                v = v.try_index((.., .., start..end, ..))?;
            }
        }

        // Fused SDPA (single Metal kernel for decode when q_len=1).
        let mask = if seq_len > 1 {
            Some(mlx_rs::fast::ScaledDotProductAttentionMask::Causal)
        } else {
            None
        };
        let out = mlx_rs::fast::scaled_dot_product_attention(&q, &k, &v, self.scale, mask)?;

        // out: [1, heads, seq, head_dim] -> [seq, hidden]
        let hidden = (self.num_heads * self.head_dim) as i32;
        let out = out
            .squeeze_axes(&[0])?
            .transpose_axes(&[1, 0, 2])?
            .reshape(&[seq_len, hidden])?;

        self.o_proj.forward(&out)
    }
}

// ---------------------------------------------------------------------------
// MlxLlamaDecoderLayer
// ---------------------------------------------------------------------------

/// A single LLaMA decoder layer using MLX.
pub struct MlxLlamaDecoderLayer {
    self_attn: MlxLlamaAttention,
    mlp: MlxLlamaMLP,
    input_layernorm: nn::RmsNorm,
    post_attention_layernorm: nn::RmsNorm,
}

impl MlxLlamaDecoderLayer {
    /// Create a new decoder layer.
    pub fn new(config: &LlamaConfig) -> Result<Self, Exception> {
        Ok(Self {
            self_attn: MlxLlamaAttention::new(config)?,
            mlp: MlxLlamaMLP::new(config.hidden_size as i32, config.intermediate_size as i32)?,
            input_layernorm: nn::RmsNormBuilder::new(config.hidden_size as i32)
                .eps(config.rms_norm_eps)
                .build()?,
            post_attention_layernorm: nn::RmsNormBuilder::new(config.hidden_size as i32)
                .eps(config.rms_norm_eps)
                .build()?,
        })
    }

    /// Load weights from a flat HashMap.
    pub fn load_weights(&mut self, weights: &HashMap<String, Array>, prefix: &str) {
        self.self_attn
            .load_weights(weights, &format!("{prefix}.self_attn"));
        self.mlp.load_weights(weights, &format!("{prefix}.mlp"));
        assign_weight(
            &mut self.input_layernorm.weight,
            weights,
            &format!("{prefix}.input_layernorm.weight"),
        );
        assign_weight(
            &mut self.post_attention_layernorm.weight,
            weights,
            &format!("{prefix}.post_attention_layernorm.weight"),
        );
    }

    /// Forward pass.
    pub fn forward(
        &mut self,
        hidden_states: &Array,
        positions: &Array,
        cache: &mut Option<(Array, Array)>,
    ) -> Result<Array, Exception> {
        // Pre-attention layernorm + attention + residual.
        let normed = self.input_layernorm.forward(hidden_states)?;
        let attn_output = self.self_attn.forward(&normed, positions, cache)?;
        let hidden_states = hidden_states.add(&attn_output)?;

        // Post-attention layernorm + MLP + residual.
        let normed = self.post_attention_layernorm.forward(&hidden_states)?;
        let mlp_output = self.mlp.forward(&normed)?;
        hidden_states.add(&mlp_output)
    }
}

// ---------------------------------------------------------------------------
// MlxLlamaForCausalLM
// ---------------------------------------------------------------------------

/// LLaMA for causal language modeling using MLX.
pub struct MlxLlamaForCausalLM {
    embed_tokens: nn::Embedding,
    layers: Vec<MlxLlamaDecoderLayer>,
    norm: nn::RmsNorm,
    pub(crate) lm_head: Option<nn::Linear>,
    tie_word_embeddings: bool,
    #[allow(dead_code)]
    config: LlamaConfig,
}

impl MlxLlamaForCausalLM {
    /// Create a new model with random initialization.
    pub fn new(config: &LlamaConfig) -> Result<Self, Exception> {
        let mut layers = Vec::with_capacity(config.num_hidden_layers);
        for _ in 0..config.num_hidden_layers {
            layers.push(MlxLlamaDecoderLayer::new(config)?);
        }

        let lm_head = if config.tie_word_embeddings {
            None
        } else {
            Some(
                nn::LinearBuilder::new(config.hidden_size as i32, config.vocab_size as i32)
                    .bias(false)
                    .build()?,
            )
        };

        Ok(Self {
            embed_tokens: nn::Embedding::new(config.vocab_size as i32, config.hidden_size as i32)?,
            layers,
            norm: nn::RmsNormBuilder::new(config.hidden_size as i32)
                .eps(config.rms_norm_eps)
                .build()?,
            lm_head,
            tie_word_embeddings: config.tie_word_embeddings,
            config: config.clone(),
        })
    }

    /// Load weights from a flat HashMap of name → Array.
    pub fn load_weights(&mut self, weights: &HashMap<String, Array>) {
        assign_weight(
            &mut self.embed_tokens.weight,
            weights,
            "model.embed_tokens.weight",
        );
        for (i, layer) in self.layers.iter_mut().enumerate() {
            layer.load_weights(weights, &format!("model.layers.{i}"));
        }
        assign_weight(&mut self.norm.weight, weights, "model.norm.weight");
        if let Some(ref mut lm_head) = self.lm_head {
            assign_weight(&mut lm_head.weight, weights, "lm_head.weight");
        }
    }

    /// Load weights with a configurable prefix (e.g., "model" or "language_model.model").
    pub fn load_weights_with_prefix(&mut self, weights: &HashMap<String, Array>, prefix: &str) {
        assign_weight(
            &mut self.embed_tokens.weight,
            weights,
            &format!("{prefix}.embed_tokens.weight"),
        );
        for (i, layer) in self.layers.iter_mut().enumerate() {
            layer.load_weights(weights, &format!("{prefix}.layers.{i}"));
        }
        assign_weight(
            &mut self.norm.weight,
            weights,
            &format!("{prefix}.norm.weight"),
        );
        if let Some(ref mut lm_head) = self.lm_head {
            assign_weight(&mut lm_head.weight, weights, "lm_head.weight");
        }
    }

    /// Embed token IDs → hidden states.
    pub fn embed(&mut self, input_ids: &Array) -> Result<Array, Exception> {
        self.embed_tokens.forward(input_ids)
    }

    /// Run backbone on embeddings → logits.
    pub fn forward_embeds(
        &mut self,
        inputs_embeds: &Array,
        positions: &Array,
        kv_cache: &mut MlxKvCache,
    ) -> Result<Array, Exception> {
        let mut hidden_states = inputs_embeds.clone();
        for (i, layer) in self.layers.iter_mut().enumerate() {
            hidden_states = layer.forward(&hidden_states, positions, &mut kv_cache[i])?;
        }
        hidden_states = self.norm.forward(&hidden_states)?;

        let logits = if self.tie_word_embeddings {
            self.embed_tokens.as_linear(&hidden_states)?
        } else {
            self.lm_head.as_mut().unwrap().forward(&hidden_states)?
        };
        logits.as_dtype(Dtype::Float32)
    }

    /// Load model weights from safetensors files in a directory.
    pub fn load(
        model_dir: &Path,
        config: &LlamaConfig,
        _dtype: Dtype,
    ) -> Result<Self, Box<dyn std::error::Error + Send + Sync>> {
        let mut model = Self::new(config)?;

        // Load all safetensors weights into a flat HashMap.
        let weights = load_safetensors_weights(model_dir)?;

        // Assign weights to model parameters.
        model.load_weights(&weights);

        // Eval to materialize the loaded weights.
        mlx_rs::transforms::eval(weights.values())?;

        Ok(model)
    }
}

impl super::MlxModel for MlxLlamaForCausalLM {
    fn inject_lora(
        &mut self,
        adapter: &crate::lora::MlxLoraAdapter,
    ) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        use crate::lora::merge_lora_into_weight;
        let targets = &adapter.config.target_modules;
        for (i, layer) in self.layers.iter_mut().enumerate() {
            // Attention projections.
            let attn_prefix = format!("model.layers.{}.self_attn", i);
            for name in ["q_proj", "k_proj", "v_proj", "o_proj"] {
                if targets.iter().any(|t| t == name) {
                    let key = format!("{}.{}", attn_prefix, name);
                    if let Some((a, b)) = adapter.weights.get(&key) {
                        let proj = match name {
                            "q_proj" => &mut layer.self_attn.q_proj,
                            "k_proj" => &mut layer.self_attn.k_proj,
                            "v_proj" => &mut layer.self_attn.v_proj,
                            "o_proj" => &mut layer.self_attn.o_proj,
                            _ => unreachable!(),
                        };
                        merge_lora_into_weight(&mut proj.weight, a, b, adapter.scaling)?;
                    }
                }
            }
            // MLP projections.
            let mlp_prefix = format!("model.layers.{}.mlp", i);
            for name in ["gate_proj", "up_proj", "down_proj"] {
                if targets.iter().any(|t| t == name) {
                    let key = format!("{}.{}", mlp_prefix, name);
                    if let Some((a, b)) = adapter.weights.get(&key) {
                        let proj = match name {
                            "gate_proj" => &mut layer.mlp.gate_proj,
                            "up_proj" => &mut layer.mlp.up_proj,
                            "down_proj" => &mut layer.mlp.down_proj,
                            _ => unreachable!(),
                        };
                        merge_lora_into_weight(&mut proj.weight, a, b, adapter.scaling)?;
                    }
                }
            }
        }
        Ok(())
    }

    fn forward(
        &mut self,
        input_ids: &Array,
        positions: &Array,
        kv_cache: &mut MlxKvCache,
    ) -> mlx_rs::error::Result<Array> {
        let mut hidden_states = self.embed_tokens.forward(input_ids)?;

        for (i, layer) in self.layers.iter_mut().enumerate() {
            hidden_states = layer.forward(&hidden_states, positions, &mut kv_cache[i])?;
        }

        hidden_states = self.norm.forward(&hidden_states)?;

        // Compute logits.
        let logits = if self.tie_word_embeddings {
            self.embed_tokens.as_linear(&hidden_states)?
        } else {
            self.lm_head.as_mut().unwrap().forward(&hidden_states)?
        };

        // Cast logits to f32 for sampling.
        logits.as_dtype(Dtype::Float32)
    }

    fn num_layers(&self) -> usize {
        self.layers.len()
    }

    fn hidden_states(
        &mut self,
        input_ids: &Array,
        positions: &Array,
    ) -> mlx_rs::error::Result<Array> {
        let mut kv_cache: MlxKvCache = (0..self.layers.len()).map(|_| None).collect();
        let mut hidden_states = self.embed_tokens.forward(input_ids)?;
        for (i, layer) in self.layers.iter_mut().enumerate() {
            hidden_states = layer.forward(&hidden_states, positions, &mut kv_cache[i])?;
        }
        self.norm.forward(&hidden_states)
    }
}

// ---------------------------------------------------------------------------
// Weight loading helper
// ---------------------------------------------------------------------------

/// Load all safetensors weights from a model directory into a flat HashMap.
pub fn load_safetensors_weights(
    model_dir: &Path,
) -> Result<HashMap<String, Array>, Box<dyn std::error::Error + Send + Sync>> {
    let single_path = model_dir.join("model.safetensors");
    if single_path.exists() {
        let weights = Array::load_safetensors(&single_path)?;
        return Ok(weights);
    }

    // Sharded model.
    let index_path = model_dir.join("model.safetensors.index.json");
    if index_path.exists() {
        let index = vllm_model::weight::SafeTensorsIndex::from_file(&index_path)?;
        let shard_files = index.shard_files();

        let mut all_weights = HashMap::new();
        for shard_name in &shard_files {
            let shard_path = model_dir.join(shard_name);
            let shard_data = Array::load_safetensors(&shard_path)?;
            all_weights.extend(shard_data);
        }
        return Ok(all_weights);
    }

    Err(format!(
        "no model.safetensors or model.safetensors.index.json in {}",
        model_dir.display()
    )
    .into())
}

/// Factory function for creating an MLX LLaMA model.
pub fn create_mlx_llama(
    model_dir: &Path,
    config: &HfModelConfig,
    dtype: Dtype,
) -> Result<Box<dyn super::MlxModel>, Box<dyn std::error::Error + Send + Sync>> {
    let llama_config = LlamaConfig::from_hf_config(config)
        .map_err(|e| -> Box<dyn std::error::Error + Send + Sync> { e.into() })?;
    let model = MlxLlamaForCausalLM::load(model_dir, &llama_config, dtype)?;
    Ok(Box::new(model))
}

/// Factory function for creating an MLX LLaMA model from GPTQ weights.
///
/// Dequantizes GPTQ INT4 weights at load time, then uses the standard
/// (non-quantized) model architecture.
pub fn create_mlx_gptq_llama(
    model_dir: &Path,
    config: &HfModelConfig,
    _dtype: Dtype,
) -> Result<Box<dyn super::MlxModel>, Box<dyn std::error::Error + Send + Sync>> {
    let llama_config = LlamaConfig::from_hf_config(config)
        .map_err(|e| -> Box<dyn std::error::Error + Send + Sync> { e.into() })?;

    // Parse GPTQ config.
    let gptq_cfg = vllm_model::gptq_config::GptqQuantizeConfig::from_dir(model_dir)
        .or_else(|_| {
            config
                .extra
                .get("quantization_config")
                .ok_or_else(|| vllm_model::error::ModelError::Other("no GPTQ config found".into()))
                .and_then(vllm_model::gptq_config::GptqQuantizeConfig::from_json_value)
        })
        .map_err(|e| -> Box<dyn std::error::Error + Send + Sync> { e.into() })?;

    let gptq = gptq_cfg.to_gptq_config();

    // Load raw safetensors weights (includes qweight/qzeros/scales/g_idx).
    let raw_weights = load_safetensors_weights(model_dir)?;

    // Dequantize GPTQ weights into standard .weight tensors.
    let weights = crate::gptq::dequantize_gptq_weights(raw_weights, &gptq)?;

    // Build standard model and load dequantized weights.
    let mut model = MlxLlamaForCausalLM::new(&llama_config)?;
    model.load_weights(&weights);
    mlx_rs::transforms::eval(weights.values())?;

    Ok(Box::new(model))
}

/// Factory function for creating an MLX Qwen2 model from GPTQ weights.
pub fn create_mlx_gptq_qwen2(
    model_dir: &Path,
    config: &HfModelConfig,
    dtype: Dtype,
) -> Result<Box<dyn super::MlxModel>, Box<dyn std::error::Error + Send + Sync>> {
    // Qwen2 is architecturally identical to LLaMA, reuse the same factory.
    create_mlx_gptq_llama(model_dir, config, dtype)
}

/// Factory function for creating an MLX LLaMA model from AWQ weights.
///
/// Dequantizes AWQ INT4 weights at load time, then uses the standard
/// (non-quantized) model architecture.
pub fn create_mlx_awq_llama(
    model_dir: &Path,
    config: &HfModelConfig,
    _dtype: Dtype,
) -> Result<Box<dyn super::MlxModel>, Box<dyn std::error::Error + Send + Sync>> {
    let llama_config = LlamaConfig::from_hf_config(config)
        .map_err(|e| -> Box<dyn std::error::Error + Send + Sync> { e.into() })?;

    // Parse AWQ config.
    let awq_cfg = vllm_model::awq_config::AwqQuantizeConfig::from_dir(model_dir)
        .or_else(|_| {
            config
                .extra
                .get("quantization_config")
                .ok_or_else(|| vllm_model::error::ModelError::Other("no AWQ config found".into()))
                .and_then(vllm_model::awq_config::AwqQuantizeConfig::from_json_value)
        })
        .map_err(|e| -> Box<dyn std::error::Error + Send + Sync> { e.into() })?;

    let awq = awq_cfg.to_awq_config();

    // Load raw safetensors weights (includes qweight/qzeros/scales).
    let raw_weights = load_safetensors_weights(model_dir)?;

    // Dequantize AWQ weights into standard .weight tensors.
    let weights = crate::awq::dequantize_awq_weights(raw_weights, &awq)?;

    // Build standard model and load dequantized weights.
    let mut model = MlxLlamaForCausalLM::new(&llama_config)?;
    model.load_weights(&weights);
    mlx_rs::transforms::eval(weights.values())?;

    Ok(Box::new(model))
}

/// Factory function for creating an MLX Qwen2 model from AWQ weights.
pub fn create_mlx_awq_qwen2(
    model_dir: &Path,
    config: &HfModelConfig,
    dtype: Dtype,
) -> Result<Box<dyn super::MlxModel>, Box<dyn std::error::Error + Send + Sync>> {
    // Qwen2 is architecturally identical to LLaMA, reuse the same factory.
    create_mlx_awq_llama(model_dir, config, dtype)
}

/// Factory function for creating an MLX LLaMA model from BitsAndBytes NF4 weights.
///
/// Dequantizes BnB NF4 weights at load time, then uses the standard
/// (non-quantized) model architecture.
pub fn create_mlx_bnb_llama(
    model_dir: &Path,
    config: &HfModelConfig,
    _dtype: Dtype,
) -> Result<Box<dyn super::MlxModel>, Box<dyn std::error::Error + Send + Sync>> {
    let llama_config = LlamaConfig::from_hf_config(config)
        .map_err(|e| -> Box<dyn std::error::Error + Send + Sync> { e.into() })?;

    // Parse BnB config.
    let bnb_cfg = config
        .extra
        .get("quantization_config")
        .ok_or_else(|| vllm_model::error::ModelError::Other("no BnB config found".into()))
        .and_then(vllm_model::bnb_config::BnbQuantizeConfig::from_json_value)
        .map_err(|e| -> Box<dyn std::error::Error + Send + Sync> { e.into() })?;

    let bnb = bnb_cfg.to_bnb_config();

    // Load raw safetensors weights (includes packed uint8 + absmax).
    let raw_weights = load_safetensors_weights(model_dir)?;

    // Dequantize BnB weights into standard .weight tensors.
    let weights = crate::bnb::dequantize_bnb_weights(raw_weights, &bnb, config)?;

    // Build standard model and load dequantized weights.
    let mut model = MlxLlamaForCausalLM::new(&llama_config)?;
    model.load_weights(&weights);
    mlx_rs::transforms::eval(weights.values())?;

    Ok(Box::new(model))
}

/// Factory function for creating an MLX Qwen2 model from BitsAndBytes NF4 weights.
pub fn create_mlx_bnb_qwen2(
    model_dir: &Path,
    config: &HfModelConfig,
    dtype: Dtype,
) -> Result<Box<dyn super::MlxModel>, Box<dyn std::error::Error + Send + Sync>> {
    // Qwen2 is architecturally identical to LLaMA, reuse the same factory.
    create_mlx_bnb_llama(model_dir, config, dtype)
}

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

    #[test]
    fn test_llama_config_from_hf() {
        let hf_config: HfModelConfig = serde_json::from_str(
            r#"{
                "architectures": ["LlamaForCausalLM"],
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
        assert_eq!(config.head_dim, 128);
    }

    #[test]
    fn test_mlx_llama_mlp_forward() {
        let mut mlp = MlxLlamaMLP::new(32, 64).unwrap();
        let x = mlx_rs::ops::ones::<f32>(&[3, 32]).unwrap();
        let out = mlp.forward(&x).unwrap();
        out.eval().unwrap();
        assert_eq!(out.shape(), &[3, 32]);
    }

    #[test]
    fn test_mlx_llama_attention_forward() {
        let config = test_config();
        let mut attn = MlxLlamaAttention::new(&config).unwrap();

        let x = mlx_rs::ops::ones::<f32>(&[4, 32]).unwrap();
        let positions = Array::from_iter(0..4i32, &[4]);
        let mut cache = None;

        let out = attn.forward(&x, &positions, &mut cache).unwrap();
        out.eval().unwrap();
        assert_eq!(out.shape(), &[4, 32]);
        assert!(cache.is_some());
    }

    #[test]
    fn test_mlx_llama_model_forward() {
        let config = test_config();
        let mut model = MlxLlamaForCausalLM::new(&config).unwrap();

        let input_ids = Array::from_iter(vec![1i32, 5, 10], &[3]);
        let positions = Array::from_iter(0..3i32, &[3]);
        let mut kv_cache = crate::cache::empty_kv_cache(config.num_hidden_layers);

        let logits = <MlxLlamaForCausalLM as crate::models::MlxModel>::forward(
            &mut model,
            &input_ids,
            &positions,
            &mut kv_cache,
        )
        .unwrap();
        logits.eval().unwrap();
        assert_eq!(logits.shape(), &[3, config.vocab_size as i32]);
    }

    #[test]
    fn test_mlx_llama_kv_cache_prefill_and_decode() {
        let config = test_config();
        let mut model = MlxLlamaForCausalLM::new(&config).unwrap();
        let mut kv_cache = crate::cache::empty_kv_cache(config.num_hidden_layers);

        // Prefill: 3 tokens
        let input_ids = Array::from_iter(vec![1i32, 5, 10], &[3]);
        let positions = Array::from_iter(0..3i32, &[3]);
        let logits = <MlxLlamaForCausalLM as crate::models::MlxModel>::forward(
            &mut model,
            &input_ids,
            &positions,
            &mut kv_cache,
        )
        .unwrap();
        logits.eval().unwrap();
        assert_eq!(logits.shape(), &[3, config.vocab_size as i32]);

        // Verify KV cache is populated.
        for entry in &kv_cache {
            assert!(entry.is_some());
        }

        // Decode: 1 token at position 3
        let decode_ids = Array::from_iter(vec![15i32], &[1]);
        let decode_pos = Array::from_iter(vec![3i32], &[1]);
        let logits2 = <MlxLlamaForCausalLM as crate::models::MlxModel>::forward(
            &mut model,
            &decode_ids,
            &decode_pos,
            &mut kv_cache,
        )
        .unwrap();
        logits2.eval().unwrap();
        assert_eq!(logits2.shape(), &[1, config.vocab_size as i32]);
    }
}
