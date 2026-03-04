// SPDX-License-Identifier: Apache-2.0
//! LLaMA/Qwen2-family model implementation on WebGPU tensors.
//!
//! Provides `WgpuWorker` — a self-contained inference engine that loads
//! safetensors weights onto the GPU and runs autoregressive generation
//! using WGSL compute shaders. Supports LLaMA, Qwen2, SmolLM, and any
//! model that uses the same architecture (RMSNorm + SiLU-gated MLP +
//! RoPE + GQA, with optional QKV biases).

use std::collections::HashMap;
use std::path::Path;

use serde::Deserialize;

use crate::WgpuDevice;
use crate::ops;
use crate::tensor::WgpuTensor;

// ---------------------------------------------------------------------------
// ModelConfig
// ---------------------------------------------------------------------------

/// Model configuration parsed from HuggingFace config.json.
#[derive(Debug, Clone, Deserialize)]
pub struct ModelConfig {
    pub hidden_size: usize,
    pub num_attention_heads: usize,
    #[serde(default)]
    pub num_key_value_heads: usize,
    pub num_hidden_layers: usize,
    pub intermediate_size: usize,
    pub vocab_size: usize,
    #[serde(default = "default_max_position_embeddings")]
    pub max_position_embeddings: usize,
    #[serde(default = "default_rms_norm_eps")]
    pub rms_norm_eps: f64,
    #[serde(default = "default_rope_theta")]
    pub rope_theta: f64,
}

fn default_max_position_embeddings() -> usize {
    2048
}
fn default_rms_norm_eps() -> f64 {
    1e-6
}
fn default_rope_theta() -> f64 {
    10000.0
}

impl ModelConfig {
    pub fn num_kv_heads(&self) -> usize {
        if self.num_key_value_heads == 0 {
            self.num_attention_heads
        } else {
            self.num_key_value_heads
        }
    }

    pub fn head_dim(&self) -> usize {
        self.hidden_size / self.num_attention_heads
    }
}

// ---------------------------------------------------------------------------
// WgpuLinear
// ---------------------------------------------------------------------------

/// A linear layer: weight [out, in], optional bias [out].
/// Computes y = x @ W^T + bias, matching candle/PyTorch convention.
pub struct WgpuLinear {
    pub weight: WgpuTensor,
    pub bias: Option<WgpuTensor>,
}

impl WgpuLinear {
    pub fn forward(&self, x: &WgpuTensor) -> Result<WgpuTensor, crate::WgpuError> {
        ops::linear(x, &self.weight, self.bias.as_ref())
    }
}

// ---------------------------------------------------------------------------
// Model weights
// ---------------------------------------------------------------------------

/// A single transformer layer's weights.
pub struct LayerWeights {
    pub input_layernorm: WgpuTensor,
    pub post_attention_layernorm: WgpuTensor,
    pub qkv_proj: WgpuLinear,
    pub o_proj: WgpuLinear,
    pub gate_up_proj: WgpuLinear,
    pub down_proj: WgpuLinear,
}

/// All model weights for a LLaMA-family model.
pub struct ModelWeights {
    pub embed_tokens: WgpuTensor,
    pub layers: Vec<LayerWeights>,
    pub norm: WgpuTensor,
    pub lm_head: WgpuTensor,
}

// ---------------------------------------------------------------------------
// KV cache
// ---------------------------------------------------------------------------

/// KV cache for a single layer (contiguous, not paged).
struct LayerKvCache {
    k: Vec<f32>,
    v: Vec<f32>,
    len: usize,
}

// ---------------------------------------------------------------------------
// WgpuWorker
// ---------------------------------------------------------------------------

/// WebGPU inference worker — loads and runs LLaMA-family models.
pub struct WgpuWorker {
    pub device: WgpuDevice,
    pub config: ModelConfig,
    pub weights: Option<ModelWeights>,
    kv_caches: Vec<LayerKvCache>,
    cos_cache: Option<WgpuTensor>,
    sin_cache: Option<WgpuTensor>,
}

impl WgpuWorker {
    /// Create a new worker (weights not yet loaded).
    pub fn new(device: WgpuDevice, config: ModelConfig) -> Self {
        let num_layers = config.num_hidden_layers;
        let max_seq = config.max_position_embeddings;
        let num_kv_heads = config.num_kv_heads();
        let head_dim = config.head_dim();

        let kv_caches = (0..num_layers)
            .map(|_| LayerKvCache {
                k: vec![0.0; max_seq * num_kv_heads * head_dim],
                v: vec![0.0; max_seq * num_kv_heads * head_dim],
                len: 0,
            })
            .collect();

        Self {
            device,
            config,
            weights: None,
            kv_caches,
            cos_cache: None,
            sin_cache: None,
        }
    }

    /// Reset KV caches (for new conversation).
    pub fn reset_kv(&mut self) {
        for cache in &mut self.kv_caches {
            cache.k.fill(0.0);
            cache.v.fill(0.0);
            cache.len = 0;
        }
    }

    /// Precompute RoPE cos/sin caches.
    pub fn init_rope_cache(&mut self) -> Result<(), crate::WgpuError> {
        let max_seq = self.config.max_position_embeddings;
        let head_dim = self.config.head_dim();
        let half_dim = head_dim / 2;
        let theta = self.config.rope_theta;

        let mut cos_data = vec![0.0f32; max_seq * half_dim];
        let mut sin_data = vec![0.0f32; max_seq * half_dim];

        for pos in 0..max_seq {
            for i in 0..half_dim {
                let freq = 1.0 / (theta as f32).powf(2.0 * i as f32 / head_dim as f32);
                let angle = pos as f32 * freq;
                cos_data[pos * half_dim + i] = angle.cos();
                sin_data[pos * half_dim + i] = angle.sin();
            }
        }

        self.cos_cache = Some(WgpuTensor::from_f32(
            &self.device,
            &[max_seq, half_dim],
            &cos_data,
        )?);
        self.sin_cache = Some(WgpuTensor::from_f32(
            &self.device,
            &[max_seq, half_dim],
            &sin_data,
        )?);
        Ok(())
    }

    /// Download a model from HuggingFace and load weights onto the GPU.
    #[cfg(not(target_arch = "wasm32"))]
    pub fn from_pretrained(
        device: WgpuDevice,
        model_id: &str,
    ) -> Result<(Self, ModelConfig, tokenizers::Tokenizer), String> {
        let api = hf_hub::api::sync::Api::new().map_err(|e| format!("HF API init: {e}"))?;
        let repo = api.model(model_id.to_string());

        let config_path = repo
            .get("config.json")
            .map_err(|e| format!("download config.json: {e}"))?;
        let config_str =
            std::fs::read_to_string(&config_path).map_err(|e| format!("read config.json: {e}"))?;
        let config: ModelConfig =
            serde_json::from_str(&config_str).map_err(|e| format!("parse config.json: {e}"))?;

        let model_dir = config_path
            .parent()
            .ok_or_else(|| "can't find model dir".to_string())?;

        // Download weights (single file or sharded)
        if repo.get("model.safetensors").is_err() {
            let index_path = repo
                .get("model.safetensors.index.json")
                .map_err(|e| format!("download weights: {e}"))?;
            let index_str =
                std::fs::read_to_string(&index_path).map_err(|e| format!("read index: {e}"))?;
            let index: serde_json::Value =
                serde_json::from_str(&index_str).map_err(|e| format!("parse index: {e}"))?;
            if let Some(map) = index.get("weight_map").and_then(|v| v.as_object()) {
                let mut shards: Vec<String> = map
                    .values()
                    .filter_map(|v| v.as_str().map(|s| s.to_string()))
                    .collect();
                shards.sort();
                shards.dedup();
                for shard in &shards {
                    repo.get(shard)
                        .map_err(|e| format!("download {shard}: {e}"))?;
                }
            }
        }

        let tokenizer_path = repo
            .get("tokenizer.json")
            .map_err(|e| format!("download tokenizer.json: {e}"))?;
        let tokenizer = tokenizers::Tokenizer::from_file(&tokenizer_path)
            .map_err(|e| format!("load tokenizer: {e}"))?;

        let mut worker = Self::new(device, config.clone());
        worker
            .init_rope_cache()
            .map_err(|e| format!("RoPE init: {e}"))?;
        worker.load_weights(model_dir)?;

        Ok((worker, config, tokenizer))
    }

    /// Load model weights from a directory containing safetensors files.
    pub fn load_weights(&mut self, model_dir: &Path) -> Result<(), String> {
        let mut st_files: Vec<_> = std::fs::read_dir(model_dir)
            .map_err(|e| format!("can't read {}: {e}", model_dir.display()))?
            .filter_map(|e| e.ok())
            .filter(|e| e.path().extension().is_some_and(|ext| ext == "safetensors"))
            .map(|e| e.path())
            .collect();
        st_files.sort();

        if st_files.is_empty() {
            return Err(format!("no .safetensors files in {}", model_dir.display()));
        }

        let mut tensors: HashMap<String, (Vec<usize>, Vec<f32>)> = HashMap::new();
        for path in &st_files {
            let data = std::fs::read(path).map_err(|e| format!("read {}: {e}", path.display()))?;
            let st = safetensors::SafeTensors::deserialize(&data)
                .map_err(|e| format!("parse {}: {e}", path.display()))?;
            for name in st.names() {
                let view = st.tensor(name).map_err(|e| format!("{name}: {e}"))?;
                let shape: Vec<usize> = view.shape().to_vec();
                let f32_data = safetensors_to_f32(view.dtype(), view.data(), &shape)?;
                tensors.insert(name.to_string(), (shape, f32_data));
            }
        }

        let hidden = self.config.hidden_size;
        let num_q_heads = self.config.num_attention_heads;
        let num_kv_heads = self.config.num_kv_heads();
        let head_dim = self.config.head_dim();
        let intermediate = self.config.intermediate_size;
        let q_size = num_q_heads * head_dim;
        let kv_size = num_kv_heads * head_dim;

        let load = |name: &str| -> Result<WgpuTensor, String> {
            let (shape, data) = tensors
                .get(name)
                .ok_or_else(|| format!("missing weight: {name}"))?;
            WgpuTensor::from_f32(&self.device, shape, data).map_err(|e| format!("{name}: {e}"))
        };

        let load_bias = |name: &str| -> Result<Option<WgpuTensor>, String> {
            match tensors.get(name) {
                Some((shape, data)) => Ok(Some(
                    WgpuTensor::from_f32(&self.device, shape, data)
                        .map_err(|e| format!("{name}: {e}"))?,
                )),
                None => Ok(None),
            }
        };

        let load_linear = |weight_name: &str, bias_name: &str| -> Result<WgpuLinear, String> {
            Ok(WgpuLinear {
                weight: load(weight_name)?,
                bias: load_bias(bias_name)?,
            })
        };

        let load_fused_qkv = |prefix: &str| -> Result<WgpuLinear, String> {
            let q_name = format!("{prefix}.q_proj.weight");
            let k_name = format!("{prefix}.k_proj.weight");
            let v_name = format!("{prefix}.v_proj.weight");

            let (_, q_data) = tensors
                .get(&q_name)
                .ok_or_else(|| format!("missing {q_name}"))?;
            let (_, k_data) = tensors
                .get(&k_name)
                .ok_or_else(|| format!("missing {k_name}"))?;
            let (_, v_data) = tensors
                .get(&v_name)
                .ok_or_else(|| format!("missing {v_name}"))?;

            let total_out = q_size + 2 * kv_size;
            let mut fused = vec![0.0f32; total_out * hidden];
            fused[..q_size * hidden].copy_from_slice(&q_data[..q_size * hidden]);
            fused[q_size * hidden..(q_size + kv_size) * hidden]
                .copy_from_slice(&k_data[..kv_size * hidden]);
            fused[(q_size + kv_size) * hidden..total_out * hidden]
                .copy_from_slice(&v_data[..kv_size * hidden]);

            let weight = WgpuTensor::from_f32(&self.device, &[total_out, hidden], &fused)
                .map_err(|e| format!("fused_qkv weight: {e}"))?;

            let q_bias_name = format!("{prefix}.q_proj.bias");
            let bias = if tensors.contains_key(&q_bias_name) {
                let k_bias_name = format!("{prefix}.k_proj.bias");
                let v_bias_name = format!("{prefix}.v_proj.bias");
                let (_, qb) = tensors.get(&q_bias_name).unwrap();
                let (_, kb) = tensors
                    .get(&k_bias_name)
                    .ok_or_else(|| format!("missing {k_bias_name}"))?;
                let (_, vb) = tensors
                    .get(&v_bias_name)
                    .ok_or_else(|| format!("missing {v_bias_name}"))?;
                let mut fused_bias = vec![0.0f32; total_out];
                fused_bias[..q_size].copy_from_slice(&qb[..q_size]);
                fused_bias[q_size..q_size + kv_size].copy_from_slice(&kb[..kv_size]);
                fused_bias[q_size + kv_size..total_out].copy_from_slice(&vb[..kv_size]);
                Some(
                    WgpuTensor::from_f32(&self.device, &[total_out], &fused_bias)
                        .map_err(|e| format!("fused_qkv bias: {e}"))?,
                )
            } else {
                None
            };

            Ok(WgpuLinear { weight, bias })
        };

        let load_fused_gate_up = |prefix: &str| -> Result<WgpuLinear, String> {
            let gate_name = format!("{prefix}.gate_proj.weight");
            let up_name = format!("{prefix}.up_proj.weight");

            let (_, gate_data) = tensors
                .get(&gate_name)
                .ok_or_else(|| format!("missing {gate_name}"))?;
            let (_, up_data) = tensors
                .get(&up_name)
                .ok_or_else(|| format!("missing {up_name}"))?;

            let total_out = 2 * intermediate;
            let mut fused = vec![0.0f32; total_out * hidden];
            fused[..intermediate * hidden].copy_from_slice(&gate_data[..intermediate * hidden]);
            fused[intermediate * hidden..total_out * hidden]
                .copy_from_slice(&up_data[..intermediate * hidden]);

            let weight = WgpuTensor::from_f32(&self.device, &[total_out, hidden], &fused)
                .map_err(|e| format!("fused_gate_up: {e}"))?;

            Ok(WgpuLinear { weight, bias: None })
        };

        let embed_tokens = load("model.embed_tokens.weight")?;

        let mut layers = Vec::new();
        for i in 0..self.config.num_hidden_layers {
            let prefix = format!("model.layers.{i}");
            layers.push(LayerWeights {
                input_layernorm: load(&format!("{prefix}.input_layernorm.weight"))?,
                post_attention_layernorm: load(&format!(
                    "{prefix}.post_attention_layernorm.weight"
                ))?,
                qkv_proj: load_fused_qkv(&format!("{prefix}.self_attn"))?,
                o_proj: load_linear(
                    &format!("{prefix}.self_attn.o_proj.weight"),
                    &format!("{prefix}.self_attn.o_proj.bias"),
                )?,
                gate_up_proj: load_fused_gate_up(&format!("{prefix}.mlp"))?,
                down_proj: load_linear(
                    &format!("{prefix}.mlp.down_proj.weight"),
                    &format!("{prefix}.mlp.down_proj.bias"),
                )?,
            });
        }

        let norm = load("model.norm.weight")?;
        let lm_head = if tensors.contains_key("lm_head.weight") {
            load("lm_head.weight")?
        } else {
            load("model.embed_tokens.weight")?
        };

        self.weights = Some(ModelWeights {
            embed_tokens,
            layers,
            norm,
            lm_head,
        });
        Ok(())
    }

    /// Run a single-token forward pass and return the next token ID.
    pub async fn forward_one(
        &mut self,
        token_id: u32,
        position: usize,
    ) -> Result<u32, crate::WgpuError> {
        let weights = self
            .weights
            .as_ref()
            .ok_or_else(|| crate::WgpuError::InvalidShape("model weights not loaded".into()))?;

        let hidden_size = self.config.hidden_size;
        let num_q_heads = self.config.num_attention_heads;
        let num_kv_heads = self.config.num_kv_heads();
        let head_dim = self.config.head_dim();
        let intermediate_size = self.config.intermediate_size;
        let eps = self.config.rms_norm_eps as f32;

        // 1. Embedding lookup
        let input_ids = WgpuTensor::from_u32(&self.device, &[1], &[token_id])?;
        let mut hidden = ops::embedding(&weights.embed_tokens, &input_ids)?;
        hidden = hidden.reshape(&[1, hidden_size])?;

        let positions = WgpuTensor::from_u32(&self.device, &[1], &[position as u32])?;

        // 2. Transformer layers
        for layer_idx in 0..self.config.num_hidden_layers {
            let layer = &weights.layers[layer_idx];

            let normed = ops::rms_norm(&hidden, &layer.input_layernorm, eps)?;
            let qkv = layer.qkv_proj.forward(&normed)?;

            let q_size = num_q_heads * head_dim;
            let kv_size = num_kv_heads * head_dim;

            let qkv_data = qkv.to_f32().await?;
            let q_data: Vec<f32> = qkv_data[..q_size].to_vec();
            let k_data: Vec<f32> = qkv_data[q_size..q_size + kv_size].to_vec();
            let v_data: Vec<f32> = qkv_data[q_size + kv_size..q_size + 2 * kv_size].to_vec();

            let q_tensor =
                WgpuTensor::from_f32(&self.device, &[1, num_q_heads, head_dim], &q_data)?;
            let k_tensor =
                WgpuTensor::from_f32(&self.device, &[1, num_kv_heads, head_dim], &k_data)?;

            let cos = self.cos_cache.as_ref().unwrap();
            let sin = self.sin_cache.as_ref().unwrap();

            let q_rope = ops::rope(
                &q_tensor,
                cos,
                sin,
                &positions,
                num_q_heads as u32,
                head_dim as u32,
                self.config.max_position_embeddings as u32,
            )?;
            let k_rope = ops::rope(
                &k_tensor,
                cos,
                sin,
                &positions,
                num_kv_heads as u32,
                head_dim as u32,
                self.config.max_position_embeddings as u32,
            )?;

            // Update KV cache (CPU-side)
            let k_rope_data = k_rope.to_f32().await?;
            let cache = &mut self.kv_caches[layer_idx];
            let offset = position * num_kv_heads * head_dim;
            cache.k[offset..offset + kv_size].copy_from_slice(&k_rope_data);
            cache.v[offset..offset + kv_size].copy_from_slice(&v_data);
            cache.len = position + 1;

            // Scaled dot-product attention (CPU, GQA-aware)
            let q_rope_data = q_rope.to_f32().await?;
            let scale = 1.0 / (head_dim as f32).sqrt();
            let gqa_ratio = num_q_heads / num_kv_heads;

            let mut attn_output = vec![0.0f32; num_q_heads * head_dim];
            for h in 0..num_q_heads {
                let kv_h = h / gqa_ratio;
                let scores: Vec<f32> = (0..cache.len)
                    .map(|t| {
                        let mut dot = 0.0f32;
                        for d in 0..head_dim {
                            dot += q_rope_data[h * head_dim + d]
                                * cache.k[t * num_kv_heads * head_dim + kv_h * head_dim + d];
                        }
                        dot * scale
                    })
                    .collect();
                let mut scores = scores;
                let max_score = scores.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                let mut sum_exp = 0.0f32;
                for s in &mut scores {
                    *s = (*s - max_score).exp();
                    sum_exp += *s;
                }
                for s in &mut scores {
                    *s /= sum_exp;
                }
                for (t, &score) in scores.iter().enumerate() {
                    for d in 0..head_dim {
                        attn_output[h * head_dim + d] +=
                            score * cache.v[t * num_kv_heads * head_dim + kv_h * head_dim + d];
                    }
                }
            }

            let attn_tensor = WgpuTensor::from_f32(&self.device, &[1, q_size], &attn_output)?;
            let o_out = layer.o_proj.forward(&attn_tensor)?;

            hidden = ops::add(&hidden, &o_out)?;

            let normed2 = ops::rms_norm(&hidden, &layer.post_attention_layernorm, eps)?;

            let gate_up = layer.gate_up_proj.forward(&normed2)?;
            let gate_up_data = gate_up.to_f32().await?;
            let gate_data: Vec<f32> = gate_up_data[..intermediate_size].to_vec();
            let up_data: Vec<f32> = gate_up_data[intermediate_size..2 * intermediate_size].to_vec();
            let gate_t = WgpuTensor::from_f32(&self.device, &[1, intermediate_size], &gate_data)?;
            let up_t = WgpuTensor::from_f32(&self.device, &[1, intermediate_size], &up_data)?;
            let activated = ops::silu_mul(&gate_t, &up_t)?;
            let mlp_out = layer.down_proj.forward(&activated)?;

            hidden = ops::add(&hidden, &mlp_out)?;
        }

        // 3. Final norm + lm_head
        hidden = ops::rms_norm(&hidden, &weights.norm, eps)?;
        let logits = ops::matmul_t(&hidden, &weights.lm_head)?;
        let logits_data = logits.to_f32().await?;

        let mut max_idx = 0u32;
        let mut max_val = f32::NEG_INFINITY;
        for (i, &v) in logits_data.iter().enumerate() {
            if v > max_val {
                max_val = v;
                max_idx = i as u32;
            }
        }

        Ok(max_idx)
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Convert raw safetensors bytes to f32.
fn safetensors_to_f32(
    dtype: safetensors::Dtype,
    data: &[u8],
    shape: &[usize],
) -> Result<Vec<f32>, String> {
    let numel: usize = shape.iter().product();
    match dtype {
        safetensors::Dtype::F32 => {
            if data.len() != numel * 4 {
                return Err(format!(
                    "F32 size mismatch: {} vs {}",
                    data.len(),
                    numel * 4
                ));
            }
            Ok(bytemuck::cast_slice::<u8, f32>(data).to_vec())
        }
        safetensors::Dtype::BF16 => {
            if data.len() != numel * 2 {
                return Err(format!(
                    "BF16 size mismatch: {} vs {}",
                    data.len(),
                    numel * 2
                ));
            }
            Ok(data
                .chunks_exact(2)
                .map(|b| half::bf16::from_le_bytes([b[0], b[1]]).to_f32())
                .collect())
        }
        safetensors::Dtype::F16 => {
            if data.len() != numel * 2 {
                return Err(format!(
                    "F16 size mismatch: {} vs {}",
                    data.len(),
                    numel * 2
                ));
            }
            Ok(data
                .chunks_exact(2)
                .map(|b| half::f16::from_le_bytes([b[0], b[1]]).to_f32())
                .collect())
        }
        other => Err(format!("unsupported dtype: {other:?}")),
    }
}
