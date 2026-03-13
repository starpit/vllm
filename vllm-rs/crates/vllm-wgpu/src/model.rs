// SPDX-License-Identifier: Apache-2.0
//! LLaMA/Qwen2-family model implementation on WebGPU tensors.
//!
//! Provides `WgpuWorker` — a self-contained inference engine that loads
//! safetensors weights onto the GPU and runs autoregressive generation
//! using WGSL compute shaders. Supports LLaMA, Qwen2, SmolLM, and any
//! model that uses the same architecture (RMSNorm + SiLU-gated MLP +
//! RoPE + GQA, with optional QKV biases).

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use serde::Deserialize;

use crate::WgpuDevice;
use crate::gguf::{self, GgufDType, GgufReader};
use crate::ops;
use crate::tensor::{WgpuDType, WgpuTensor};

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
/// Computes y = x @ W^T + bias, matching PyTorch convention.
/// Stores a pre-transposed copy of the weight for coalesced matvec access.
pub struct WgpuLinear {
    pub weight: WgpuTensor,
    /// Weight transposed to [in, out] for coalesced M=1 matvec reads.
    pub weight_t: WgpuTensor,
    pub bias: Option<WgpuTensor>,
}

impl WgpuLinear {
    pub fn forward(&self, x: &WgpuTensor) -> Result<WgpuTensor, crate::WgpuError> {
        let mut out = ops::matmul_t_with_transposed(x, &self.weight, &self.weight_t)?;
        if let Some(b) = &self.bias {
            if out.numel() == b.numel() {
                // M=1: same number of elements, simple add
                out = ops::add(&out, &b.reshape(&out.shape)?)?;
            } else {
                // M>1: broadcast bias [D] across rows of [M, D]
                out = ops::add_row_broadcast(&out, b)?;
            }
        }
        Ok(out)
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
    /// lm_head transposed to [hidden, vocab] for coalesced matvec.
    pub lm_head_t: WgpuTensor,
}

// ---------------------------------------------------------------------------
// GPU KV cache
// ---------------------------------------------------------------------------

/// KV cache for a single layer stored on GPU.
struct LayerKvCache {
    /// [max_seq, num_kv_heads * head_dim]
    k: WgpuTensor,
    /// [max_seq, num_kv_heads * head_dim]
    v: WgpuTensor,
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
    /// Pre-allocated single-element u32 buffer for input token ID.
    input_id_buf: WgpuTensor,
    /// Pre-allocated single-element u32 buffer for position.
    position_buf: WgpuTensor,
    /// Resolved local model directory (set by `from_pretrained`).
    model_dir: Option<PathBuf>,
}

impl WgpuWorker {
    /// Maximum KV cache sequence length — caps models with huge max_position_embeddings.
    const MAX_KV_SEQ: usize = 4096;

    /// Create a new worker (weights not yet loaded).
    pub fn new(device: WgpuDevice, config: ModelConfig) -> Self {
        let num_layers = config.num_hidden_layers;
        let max_seq = config.max_position_embeddings.min(Self::MAX_KV_SEQ);
        let num_kv_heads = config.num_kv_heads();
        let head_dim = config.head_dim();
        let kv_dim = num_kv_heads * head_dim;

        let kv_caches = (0..num_layers)
            .map(|_| LayerKvCache {
                k: WgpuTensor::zeros(&device, &[max_seq, kv_dim], WgpuDType::F32),
                v: WgpuTensor::zeros(&device, &[max_seq, kv_dim], WgpuDType::F32),
                len: 0,
            })
            .collect();

        let input_id_buf = WgpuTensor::from_u32(&device, &[1], &[0]).unwrap();
        let position_buf = WgpuTensor::from_u32(&device, &[1], &[0]).unwrap();

        Self {
            device,
            config,
            weights: None,
            kv_caches,
            cos_cache: None,
            sin_cache: None,
            input_id_buf,
            position_buf,
            model_dir: None,
        }
    }

    /// Reset KV caches (for new conversation).
    /// Get the resolved local model directory (set by `from_pretrained`).
    pub fn model_dir(&self) -> Option<&Path> {
        self.model_dir.as_deref()
    }

    pub fn reset_kv(&mut self) {
        let max_seq = self.config.max_position_embeddings.min(Self::MAX_KV_SEQ);
        let num_kv_heads = self.config.num_kv_heads();
        let head_dim = self.config.head_dim();
        let kv_dim = num_kv_heads * head_dim;

        for cache in &mut self.kv_caches {
            cache.k = WgpuTensor::zeros(&self.device, &[max_seq, kv_dim], WgpuDType::F32);
            cache.v = WgpuTensor::zeros(&self.device, &[max_seq, kv_dim], WgpuDType::F32);
            cache.len = 0;
        }
    }

    /// Precompute RoPE cos/sin caches.
    pub fn init_rope_cache(&mut self) -> Result<(), crate::WgpuError> {
        let max_seq = self.config.max_position_embeddings.min(Self::MAX_KV_SEQ);
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

    /// Try to find and download a Q4_0 GGUF file from an HF repo.
    /// Scans the repo's sibling files for .gguf files, preferring Q4_0.
    #[cfg(not(target_arch = "wasm32"))]
    fn find_gguf_in_repo(
        repo: &hf_hub::api::sync::ApiRepo,
        model_id: &str,
    ) -> Option<std::path::PathBuf> {
        // Use the HF API to list repo files and find .gguf files
        let url = format!("https://huggingface.co/api/models/{}", model_id);
        let resp = ureq::get(&url).call().ok()?;
        let body_str = resp.into_string().ok()?;
        let body: serde_json::Value = serde_json::from_str(&body_str).ok()?;
        let siblings = body.get("siblings")?.as_array()?;
        let mut gguf_files: Vec<String> = siblings
            .iter()
            .filter_map(|s: &serde_json::Value| s.get("rfilename")?.as_str().map(|s| s.to_string()))
            .filter(|name: &String| name.ends_with(".gguf"))
            .collect();

        if gguf_files.is_empty() {
            return None;
        }

        // Prefer Q4_0 for bandwidth efficiency, then other quants
        let preferred = ["Q4_0", "Q4_K_M", "Q4_K_S", "Q8_0", "F16"];
        let chosen = preferred
            .iter()
            .find_map(|pref| {
                gguf_files
                    .iter()
                    .find(|f: &&String| f.contains(pref))
                    .cloned()
            })
            .unwrap_or_else(|| {
                gguf_files.sort();
                gguf_files.first().unwrap().clone()
            });

        eprintln!("  Found GGUF: {chosen}");
        repo.get(&chosen).ok()
    }

    /// Download tokenizer.json — try the model repo first, then infer the base model.
    #[cfg(not(target_arch = "wasm32"))]
    fn download_tokenizer(
        api: &hf_hub::api::sync::Api,
        repo: &hf_hub::api::sync::ApiRepo,
        model_id: &str,
    ) -> Result<tokenizers::Tokenizer, String> {
        // Try the model's own repo first
        if let Ok(path) = repo.get("tokenizer.json") {
            return tokenizers::Tokenizer::from_file(&path)
                .map_err(|e| format!("load tokenizer: {e}"));
        }

        // For GGUF repos, try the base model repo (e.g. unsloth/X-GGUF → meta-llama/X)
        // Common patterns: "org/Model-GGUF" → "org/Model" or known base models
        let base_candidates = infer_base_model(model_id);
        for base in &base_candidates {
            eprintln!("  Trying tokenizer from {base}...");
            let base_repo = api.model(base.to_string());
            if let Ok(path) = base_repo.get("tokenizer.json") {
                return tokenizers::Tokenizer::from_file(&path)
                    .map_err(|e| format!("load tokenizer from {base}: {e}"));
            }
        }

        Err(format!(
            "no tokenizer.json found in {model_id} or base model candidates: {base_candidates:?}. \
             Use --tokenizer to specify a tokenizer.json path."
        ))
    }

    /// Download a model from HuggingFace and load weights onto the GPU.
    /// Supports:
    /// - Direct `.gguf` file path
    /// - HuggingFace repos with `.gguf` files (auto-downloads preferred quant)
    /// - Standard HuggingFace repos with safetensors
    #[cfg(not(target_arch = "wasm32"))]
    pub fn from_pretrained(
        device: WgpuDevice,
        model_id: &str,
    ) -> Result<(Self, ModelConfig, tokenizers::Tokenizer), String> {
        // Check if model_id is a local .gguf file path
        if model_id.ends_with(".gguf") {
            let path = Path::new(model_id);
            if path.exists() {
                let (worker, config) = Self::from_gguf_path(device, path)?;
                // Try to find tokenizer.json next to the GGUF file
                let tokenizer_path = path
                    .parent()
                    .unwrap_or(Path::new("."))
                    .join("tokenizer.json");
                let tokenizer = if tokenizer_path.exists() {
                    tokenizers::Tokenizer::from_file(&tokenizer_path)
                        .map_err(|e| format!("load tokenizer: {e}"))?
                } else {
                    return Err(format!(
                        "no tokenizer.json found next to {}. Place tokenizer.json in the same directory.",
                        path.display()
                    ));
                };
                return Ok((worker, config, tokenizer));
            }
        }

        let api = hf_hub::api::sync::Api::new().map_err(|e| format!("HF API init: {e}"))?;
        let repo = api.model(model_id.to_string());

        // Try to find GGUF files in the repo.
        // First, try to list repo contents via the HF API to find .gguf files.
        let gguf_path = Self::find_gguf_in_repo(&repo, model_id);

        if let Some(gguf_file) = gguf_path {
            eprintln!("  Loading GGUF: {}", gguf_file.display());
            // Download tokenizer — try this repo first, then the base model repo
            let tokenizer = Self::download_tokenizer(&api, &repo, model_id)?;
            let _ = repo.get("tokenizer_config.json"); // best-effort for chat templates
            let (mut worker, config) = Self::from_gguf_path(device, &gguf_file)?;
            // Try to resolve model_dir from config.json cache path.
            if let Ok(cfg_path) = repo.get("config.json") {
                worker.model_dir = cfg_path.parent().map(|p| p.to_path_buf());
            }
            return Ok((worker, config, tokenizer));
        }

        // Fall back to safetensors path
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

        // Download tokenizer + tokenizer_config.json (needed for chat templates).
        let tokenizer = Self::download_tokenizer(&api, &repo, model_id)?;
        let _ = repo.get("tokenizer_config.json"); // best-effort

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

        let mut worker = Self::new(device, config.clone());
        worker
            .init_rope_cache()
            .map_err(|e| format!("RoPE init: {e}"))?;
        worker.load_weights(model_dir)?;
        worker.model_dir = Some(model_dir.to_path_buf());

        Ok((worker, config, tokenizer))
    }

    /// Load model weights from raw safetensors byte slices (one per shard).
    /// This is the WASM-friendly entry point — no filesystem access needed.
    pub fn load_weights_from_bytes(&mut self, shard_bytes: &[&[u8]]) -> Result<(), String> {
        let mut tensors: HashMap<String, (Vec<usize>, Vec<f32>)> = HashMap::new();
        for (i, data) in shard_bytes.iter().enumerate() {
            let st = safetensors::SafeTensors::deserialize(data)
                .map_err(|e| format!("parse shard {i}: {e}"))?;
            for name in st.names() {
                let view = st.tensor(name).map_err(|e| format!("{name}: {e}"))?;
                let shape: Vec<usize> = view.shape().to_vec();
                let f32_data = safetensors_to_f32(view.dtype(), view.data(), &shape)?;
                tensors.insert(name.to_string(), (shape, f32_data));
            }
        }
        self.load_weights_from_tensors(tensors)
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
        self.load_weights_from_tensors(tensors)
    }

    /// Core weight loading from a pre-parsed tensor map.
    fn load_weights_from_tensors(
        &mut self,
        tensors: HashMap<String, (Vec<usize>, Vec<f32>)>,
    ) -> Result<(), String> {
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
            let (shape, data) = tensors
                .get(weight_name)
                .ok_or_else(|| format!("missing weight: {weight_name}"))?;
            let weight = WgpuTensor::from_f32_as_f16_packed(&self.device, shape, data)
                .map_err(|e| format!("{weight_name}: {e}"))?;
            let weight_t = weight
                .transpose_2d_cpu_f16(data)
                .map_err(|e| format!("{weight_name} transpose: {e}"))?;
            Ok(WgpuLinear {
                weight,
                weight_t,
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

            let weight =
                WgpuTensor::from_f32_as_f16_packed(&self.device, &[total_out, hidden], &fused)
                    .map_err(|e| format!("fused_qkv weight: {e}"))?;
            let weight_t = weight
                .transpose_2d_cpu_f16(&fused)
                .map_err(|e| format!("fused_qkv transpose: {e}"))?;

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

            Ok(WgpuLinear {
                weight,
                weight_t,
                bias,
            })
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

            let weight =
                WgpuTensor::from_f32_as_f16_packed(&self.device, &[total_out, hidden], &fused)
                    .map_err(|e| format!("fused_gate_up: {e}"))?;
            let weight_t = weight
                .transpose_2d_cpu_f16(&fused)
                .map_err(|e| format!("fused_gate_up transpose: {e}"))?;

            Ok(WgpuLinear {
                weight,
                weight_t,
                bias: None,
            })
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
        let lm_head_name = if tensors.contains_key("lm_head.weight") {
            "lm_head.weight"
        } else {
            "model.embed_tokens.weight"
        };
        let (lm_shape, lm_data) = tensors
            .get(lm_head_name)
            .ok_or_else(|| format!("missing weight: {lm_head_name}"))?;
        let lm_head = WgpuTensor::from_f32_as_f16_packed(&self.device, lm_shape, lm_data)
            .map_err(|e| format!("{lm_head_name}: {e}"))?;
        let lm_head_t = lm_head
            .transpose_2d_cpu_f16(lm_data)
            .map_err(|e| format!("{lm_head_name} transpose: {e}"))?;

        self.weights = Some(ModelWeights {
            embed_tokens,
            layers,
            norm,
            lm_head,
            lm_head_t,
        });
        Ok(())
    }

    /// Load model weights from a GGUF file.
    ///
    /// Q4_0 linear weights are loaded directly as quantized blocks (no dequantization).
    /// Norms and embeddings are dequantized to f32 (small tensors).
    /// Q/K/V are fused into a single QKV projection; gate/up are fused likewise.
    pub fn load_weights_gguf(&mut self, gguf: &GgufReader) -> Result<(), String> {
        let hidden = self.config.hidden_size;
        let num_q_heads = self.config.num_attention_heads;
        let num_kv_heads = self.config.num_kv_heads();
        let head_dim = self.config.head_dim();
        let intermediate = self.config.intermediate_size;
        let q_size = num_q_heads * head_dim;
        let kv_size = num_kv_heads * head_dim;

        // Helper: dequantize any tensor to f32
        let deq = |name: &str| -> Result<(Vec<usize>, Vec<f32>), String> {
            let info = gguf
                .tensor_info(name)
                .ok_or_else(|| format!("missing tensor: {name}"))?;
            let data = gguf.tensor_data(name)?;
            let numel = info.numel();
            let f32_data = dequantize_gguf_to_f32(info.dtype, data, numel, name)?;
            Ok((info.shape.clone(), f32_data))
        };

        // Load a 1D tensor (norm weights, biases)
        let load_1d = |name: &str| -> Result<WgpuTensor, String> {
            let (shape, data) = deq(name)?;
            WgpuTensor::from_f32(&self.device, &shape, &data).map_err(|e| format!("{name}: {e}"))
        };

        // Load a linear layer weight as Q4_0 if available, else dequant to f16
        let load_linear_q4 = |device: &WgpuDevice,
                              name: &str,
                              expected_n: usize,
                              expected_k: usize|
         -> Result<WgpuLinear, String> {
            let info = gguf
                .tensor_info(name)
                .ok_or_else(|| format!("missing tensor: {name}"))?;
            let data = gguf.tensor_data(name)?;

            if info.dtype == GgufDType::Q4_0 {
                // Load directly as Q4_0 packed
                let weight_t =
                    WgpuTensor::from_q4_0_transposed(device, expected_n, expected_k, data)
                        .map_err(|e| format!("{name} q4_0 transpose: {e}"))?;
                // For M>1 (prefill), dequantize to f16 row-major
                let f32_data = gguf::dequantize_q4_0_to_f32(data, info.numel());
                let weight = WgpuTensor::from_f32_as_f16_packed(
                    device,
                    &[expected_n, expected_k],
                    &f32_data,
                )
                .map_err(|e| format!("{name} f16: {e}"))?;
                Ok(WgpuLinear {
                    weight,
                    weight_t,
                    bias: None,
                })
            } else {
                // Dequant to f32, then store as f16
                let numel = info.numel();
                let f32_data = dequantize_gguf_to_f32(info.dtype, data, numel, name)?;
                let weight = WgpuTensor::from_f32_as_f16_packed(
                    device,
                    &[expected_n, expected_k],
                    &f32_data,
                )
                .map_err(|e| format!("{name}: {e}"))?;
                let weight_t = weight
                    .transpose_2d_cpu_f16(&f32_data)
                    .map_err(|e| format!("{name} transpose: {e}"))?;
                Ok(WgpuLinear {
                    weight,
                    weight_t,
                    bias: None,
                })
            }
        };

        // Fuse Q/K/V raw Q4_0 blocks into a single QKV tensor
        let load_fused_qkv_q4 = |device: &WgpuDevice,
                                 layer_idx: usize|
         -> Result<WgpuLinear, String> {
            let q_name = format!("blk.{layer_idx}.attn_q.weight");
            let k_name = format!("blk.{layer_idx}.attn_k.weight");
            let v_name = format!("blk.{layer_idx}.attn_v.weight");

            let q_info = gguf
                .tensor_info(&q_name)
                .ok_or_else(|| format!("missing {q_name}"))?;
            let k_info = gguf
                .tensor_info(&k_name)
                .ok_or_else(|| format!("missing {k_name}"))?;
            let v_info = gguf
                .tensor_info(&v_name)
                .ok_or_else(|| format!("missing {v_name}"))?;

            let q_data = gguf.tensor_data(&q_name)?;
            let k_data = gguf.tensor_data(&k_name)?;
            let v_data = gguf.tensor_data(&v_name)?;

            let total_out = q_size + 2 * kv_size;

            if q_info.dtype == GgufDType::Q4_0
                && k_info.dtype == GgufDType::Q4_0
                && v_info.dtype == GgufDType::Q4_0
            {
                // Concatenate raw Q4_0 blocks: each row has hidden/32 blocks of 18 bytes
                let blocks_per_row = hidden / 32;
                let row_bytes = blocks_per_row * 18;
                let mut fused_raw = vec![0u8; total_out * row_bytes];
                fused_raw[..q_size * row_bytes].copy_from_slice(&q_data[..q_size * row_bytes]);
                fused_raw[q_size * row_bytes..(q_size + kv_size) * row_bytes]
                    .copy_from_slice(&k_data[..kv_size * row_bytes]);
                fused_raw[(q_size + kv_size) * row_bytes..total_out * row_bytes]
                    .copy_from_slice(&v_data[..kv_size * row_bytes]);

                let weight_t =
                    WgpuTensor::from_q4_0_transposed(device, total_out, hidden, &fused_raw)
                        .map_err(|e| format!("fused qkv q4_0: {e}"))?;
                // Dequant for prefill (M>1)
                let f32_data = gguf::dequantize_q4_0_to_f32(&fused_raw, total_out * hidden);
                let weight =
                    WgpuTensor::from_f32_as_f16_packed(device, &[total_out, hidden], &f32_data)
                        .map_err(|e| format!("fused qkv f16: {e}"))?;
                Ok(WgpuLinear {
                    weight,
                    weight_t,
                    bias: None,
                })
            } else {
                // Dequant each to f32, fuse, store as f16
                let q_f32 = dequantize_gguf_to_f32(q_info.dtype, q_data, q_info.numel(), &q_name)?;
                let k_f32 = dequantize_gguf_to_f32(k_info.dtype, k_data, k_info.numel(), &k_name)?;
                let v_f32 = dequantize_gguf_to_f32(v_info.dtype, v_data, v_info.numel(), &v_name)?;
                let mut fused = vec![0.0f32; total_out * hidden];
                fused[..q_size * hidden].copy_from_slice(&q_f32[..q_size * hidden]);
                fused[q_size * hidden..(q_size + kv_size) * hidden]
                    .copy_from_slice(&k_f32[..kv_size * hidden]);
                fused[(q_size + kv_size) * hidden..total_out * hidden]
                    .copy_from_slice(&v_f32[..kv_size * hidden]);
                let weight =
                    WgpuTensor::from_f32_as_f16_packed(device, &[total_out, hidden], &fused)
                        .map_err(|e| format!("fused qkv: {e}"))?;
                let weight_t = weight
                    .transpose_2d_cpu_f16(&fused)
                    .map_err(|e| format!("fused qkv transpose: {e}"))?;
                Ok(WgpuLinear {
                    weight,
                    weight_t,
                    bias: None,
                })
            }
        };

        // Fuse gate/up raw Q4_0 blocks
        let load_fused_gate_up_q4 = |device: &WgpuDevice,
                                     layer_idx: usize|
         -> Result<WgpuLinear, String> {
            let gate_name = format!("blk.{layer_idx}.ffn_gate.weight");
            let up_name = format!("blk.{layer_idx}.ffn_up.weight");

            let gate_info = gguf
                .tensor_info(&gate_name)
                .ok_or_else(|| format!("missing {gate_name}"))?;
            let up_info = gguf
                .tensor_info(&up_name)
                .ok_or_else(|| format!("missing {up_name}"))?;

            let gate_data = gguf.tensor_data(&gate_name)?;
            let up_data = gguf.tensor_data(&up_name)?;

            let total_out = 2 * intermediate;

            if gate_info.dtype == GgufDType::Q4_0 && up_info.dtype == GgufDType::Q4_0 {
                let blocks_per_row = hidden / 32;
                let row_bytes = blocks_per_row * 18;
                let mut fused_raw = vec![0u8; total_out * row_bytes];
                fused_raw[..intermediate * row_bytes]
                    .copy_from_slice(&gate_data[..intermediate * row_bytes]);
                fused_raw[intermediate * row_bytes..total_out * row_bytes]
                    .copy_from_slice(&up_data[..intermediate * row_bytes]);

                let weight_t =
                    WgpuTensor::from_q4_0_transposed(device, total_out, hidden, &fused_raw)
                        .map_err(|e| format!("fused gate_up q4_0: {e}"))?;
                let f32_data = gguf::dequantize_q4_0_to_f32(&fused_raw, total_out * hidden);
                let weight =
                    WgpuTensor::from_f32_as_f16_packed(device, &[total_out, hidden], &f32_data)
                        .map_err(|e| format!("fused gate_up f16: {e}"))?;
                Ok(WgpuLinear {
                    weight,
                    weight_t,
                    bias: None,
                })
            } else {
                let gate_f32 = dequantize_gguf_to_f32(
                    gate_info.dtype,
                    gate_data,
                    gate_info.numel(),
                    &gate_name,
                )?;
                let up_f32 =
                    dequantize_gguf_to_f32(up_info.dtype, up_data, up_info.numel(), &up_name)?;
                let mut fused = vec![0.0f32; total_out * hidden];
                fused[..intermediate * hidden].copy_from_slice(&gate_f32[..intermediate * hidden]);
                fused[intermediate * hidden..total_out * hidden]
                    .copy_from_slice(&up_f32[..intermediate * hidden]);
                let weight =
                    WgpuTensor::from_f32_as_f16_packed(device, &[total_out, hidden], &fused)
                        .map_err(|e| format!("fused gate_up: {e}"))?;
                let weight_t = weight
                    .transpose_2d_cpu_f16(&fused)
                    .map_err(|e| format!("fused gate_up transpose: {e}"))?;
                Ok(WgpuLinear {
                    weight,
                    weight_t,
                    bias: None,
                })
            }
        };

        // Embedding
        let (emb_shape, emb_data) = deq("token_embd.weight")?;
        let embed_tokens = WgpuTensor::from_f32(&self.device, &emb_shape, &emb_data)
            .map_err(|e| format!("embed_tokens: {e}"))?;

        // Layers
        let mut layers = Vec::new();
        for i in 0..self.config.num_hidden_layers {
            let input_layernorm = load_1d(&format!("blk.{i}.attn_norm.weight"))?;
            let post_attention_layernorm = load_1d(&format!("blk.{i}.ffn_norm.weight"))?;

            let qkv_proj = load_fused_qkv_q4(&self.device, i)?;
            let o_proj = load_linear_q4(
                &self.device,
                &format!("blk.{i}.attn_output.weight"),
                hidden,
                hidden,
            )?;
            let gate_up_proj = load_fused_gate_up_q4(&self.device, i)?;
            let down_proj = load_linear_q4(
                &self.device,
                &format!("blk.{i}.ffn_down.weight"),
                hidden,
                intermediate,
            )?;

            layers.push(LayerWeights {
                input_layernorm,
                post_attention_layernorm,
                qkv_proj,
                o_proj,
                gate_up_proj,
                down_proj,
            });
        }

        // Final norm
        let norm = load_1d("output_norm.weight")?;

        // LM head — use output.weight if present, else tie with embed_tokens
        let lm_head_name = if gguf.tensor_info("output.weight").is_some() {
            "output.weight"
        } else {
            "token_embd.weight"
        };
        let lm_linear = load_linear_q4(&self.device, lm_head_name, self.config.vocab_size, hidden)?;

        self.weights = Some(ModelWeights {
            embed_tokens,
            layers,
            norm,
            lm_head: lm_linear.weight,
            lm_head_t: lm_linear.weight_t,
        });
        Ok(())
    }

    /// Load a GGUF file from disk and construct a worker.
    #[cfg(not(target_arch = "wasm32"))]
    pub fn from_gguf_path(device: WgpuDevice, path: &Path) -> Result<(Self, ModelConfig), String> {
        let data = std::fs::read(path).map_err(|e| format!("read GGUF: {e}"))?;
        let gguf = GgufReader::parse(&data).map_err(|e| format!("parse GGUF: {e}"))?;
        let config = gguf.model_config()?;

        eprintln!(
            "  GGUF: {} layers, hidden={}, vocab={}, {} tensors",
            config.num_hidden_layers,
            config.hidden_size,
            config.vocab_size,
            gguf.tensor_names().len(),
        );

        let mut worker = Self::new(device, config.clone());
        worker
            .init_rope_cache()
            .map_err(|e| format!("RoPE init: {e}"))?;
        worker.load_weights_gguf(&gguf)?;

        Ok((worker, config))
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
        let q_size = num_q_heads * head_dim;
        let kv_size = num_kv_heads * head_dim;

        // 1. Embedding lookup (reuse pre-allocated buffers)
        self.device.queue.write_buffer(
            &self.input_id_buf.buffer,
            0,
            bytemuck::cast_slice(&[token_id]),
        );
        self.device.queue.write_buffer(
            &self.position_buf.buffer,
            0,
            bytemuck::cast_slice(&[position as u32]),
        );
        let mut hidden = ops::embedding(&weights.embed_tokens, &self.input_id_buf)?;
        hidden = hidden.reshape(&[1, hidden_size])?;

        // 2. Transformer layers
        // Pre-compute first layer's QKV (subsequent layers fuse add_rms_norm + qkv_matvec)
        let normed = ops::rms_norm(&hidden, &weights.layers[0].input_layernorm, eps)?;
        let mut qkv = weights.layers[0].qkv_proj.forward(&normed)?;

        for layer_idx in 0..self.config.num_hidden_layers {
            let layer = &weights.layers[layer_idx];
            // qkv shape: [1, q_size + 2*kv_size]

            // Fused QKV slice + RoPE + KV cache write (1 dispatch replaces 7 ops)
            let cos = self.cos_cache.as_ref().unwrap();
            let sin = self.sin_cache.as_ref().unwrap();
            let cache = &mut self.kv_caches[layer_idx];

            let q_rope_flat = ops::rope_slice_cache(
                &qkv,
                cos,
                sin,
                &cache.k,
                &cache.v,
                q_size,
                kv_size,
                head_dim,
                num_q_heads,
                num_kv_heads,
                position,
                self.config.max_position_embeddings.min(Self::MAX_KV_SEQ),
            )?;
            cache.len = position + 1;
            let attn_output = ops::attention(
                &q_rope_flat,
                &cache.k,
                &cache.v,
                num_q_heads as u32,
                num_kv_heads as u32,
                head_dim as u32,
                cache.len as u32,
            )?;

            let o_out = layer.o_proj.forward(&attn_output)?;

            // Fused: add_rms_norm + gate_up matvec (2 dispatches → 1)
            let (mut gate_up, hidden_new) = ops::fused_add_rms_norm_matvec(
                &hidden,
                &o_out,
                &layer.post_attention_layernorm,
                &layer.gate_up_proj.weight_t,
                (2 * intermediate_size) as u32,
                eps,
            )?;
            hidden = hidden_new;
            if let Some(b) = &layer.gate_up_proj.bias {
                gate_up = ops::add(&gate_up, &b.reshape(&gate_up.shape)?)?;
            }

            let activated = ops::silu_mul_split(&gate_up, intermediate_size)?;
            let mlp_out = layer.down_proj.forward(&activated)?;

            // Fuse post-MLP residual add with next layer's input norm + qkv matvec
            if layer_idx + 1 < self.config.num_hidden_layers {
                let next_layer = &weights.layers[layer_idx + 1];
                let total_qkv_out = (q_size + 2 * kv_size) as u32;
                let (next_qkv, hidden_new) = ops::fused_add_rms_norm_matvec(
                    &hidden,
                    &mlp_out,
                    &next_layer.input_layernorm,
                    &next_layer.qkv_proj.weight_t,
                    total_qkv_out,
                    eps,
                )?;
                hidden = hidden_new;
                // Apply qkv bias if present
                if let Some(b) = &next_layer.qkv_proj.bias {
                    qkv = ops::add(&next_qkv, &b.reshape(&next_qkv.shape)?)?;
                } else {
                    qkv = next_qkv;
                }
            } else {
                hidden = ops::add(&hidden, &mlp_out)?;
            }
            // Flush every 8 layers to balance GPU pipelining vs batch overhead
            if (layer_idx + 1) % 8 == 0 || layer_idx + 1 == self.config.num_hidden_layers {
                self.device.flush();
            }
        }

        // 3. Final norm + fused lm_head + argmax
        hidden = ops::rms_norm(&hidden, &weights.norm, eps)?;
        let max_idx = ops::matvec_argmax_transposed(
            &hidden,
            &weights.lm_head_t,
            hidden_size as u32,
            self.config.vocab_size as u32,
        )
        .await?;

        Ok(max_idx)
    }

    /// Batched forward pass for prefill: processes M tokens in one pass.
    /// Returns the next token ID (argmax of last position's logits).
    /// Uses separate rms_norm + matmul_t (not fused matvec) since M>1 is compute-bound.
    pub async fn forward_batch(
        &mut self,
        token_ids: &[u32],
        start_position: usize,
    ) -> Result<u32, crate::WgpuError> {
        let m = token_ids.len();
        if m == 0 {
            return Err(crate::WgpuError::InvalidShape(
                "forward_batch: empty token_ids".into(),
            ));
        }
        if m == 1 {
            return self.forward_one(token_ids[0], start_position).await;
        }

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
        let q_size = num_q_heads * head_dim;
        let kv_size = num_kv_heads * head_dim;

        // 1. Embedding lookup for M tokens
        let indices = WgpuTensor::from_u32(&self.device, &[m], token_ids)?;
        let mut hidden = ops::embedding(&weights.embed_tokens, &indices)?;
        hidden = hidden.reshape(&[m, hidden_size])?;

        // Build positions array [start_position, start_position+1, ..., start_position+M-1]
        let positions_data: Vec<u32> = (0..m).map(|i| (start_position + i) as u32).collect();
        let positions = WgpuTensor::from_u32(&self.device, &[m], &positions_data)?;

        let cos = self.cos_cache.as_ref().unwrap();
        let sin = self.sin_cache.as_ref().unwrap();
        let max_seq = self.config.max_position_embeddings.min(Self::MAX_KV_SEQ);

        // 2. Transformer layers
        for layer_idx in 0..self.config.num_hidden_layers {
            let layer = &weights.layers[layer_idx];

            // Input layernorm: [M, hidden] → [M, hidden]
            let normed = ops::rms_norm(&hidden, &layer.input_layernorm, eps)?;

            // QKV projection: [M, hidden] × [qkv_size, hidden]^T → [M, qkv_size]
            // (WgpuLinear::forward handles bias internally)
            let qkv = layer.qkv_proj.forward(&normed)?;

            // Slice Q, K, V from QKV output
            let q_flat = ops::slice_last_dim(&qkv, 0, q_size)?;
            let k_flat = ops::slice_last_dim(&qkv, q_size, kv_size)?;
            let v_flat = ops::slice_last_dim(&qkv, q_size + kv_size, kv_size)?;

            // Reshape for RoPE: Q [M, num_q_heads, head_dim], K [M, num_kv_heads, head_dim]
            let q_rope = q_flat.reshape(&[m, num_q_heads * head_dim])?;
            let k_for_rope = k_flat.reshape(&[m, num_kv_heads * head_dim])?;

            // Apply RoPE to Q and K
            let q_roped = ops::rope(
                &q_rope.reshape(&[m, num_q_heads, head_dim])?,
                cos,
                sin,
                &positions,
                num_q_heads as u32,
                head_dim as u32,
                max_seq as u32,
            )?;
            let k_roped = ops::rope(
                &k_for_rope.reshape(&[m, num_kv_heads, head_dim])?,
                cos,
                sin,
                &positions,
                num_kv_heads as u32,
                head_dim as u32,
                max_seq as u32,
            )?;

            // Reshape back to flat
            let q_roped_flat = q_roped.reshape(&[m, q_size])?;
            let k_roped_flat = k_roped.reshape(&[m, kv_size])?;
            let v_flat_2d = v_flat.reshape(&[m, kv_size])?;

            // Causal attention with KV cache write
            let cache = &mut self.kv_caches[layer_idx];
            let attn_output = ops::attention_prefill(
                &q_roped_flat,
                &k_roped_flat,
                &v_flat_2d,
                &cache.k,
                &cache.v,
                num_q_heads as u32,
                num_kv_heads as u32,
                head_dim as u32,
                m as u32,
                start_position as u32,
            )?;
            cache.len = start_position + m;

            // O projection: [M, q_size] → [M, hidden]
            let o_out = layer.o_proj.forward(&attn_output)?;

            // Residual add
            let hidden_post_attn = ops::add(&hidden, &o_out)?;

            // Post-attention layernorm
            let normed2 = ops::rms_norm(&hidden_post_attn, &layer.post_attention_layernorm, eps)?;

            // Gate+up projection: [M, hidden] → [M, 2*intermediate]
            // (WgpuLinear::forward handles bias internally)
            let gate_up = layer.gate_up_proj.forward(&normed2)?;

            // SiLU activation + element-wise multiply
            let activated = ops::silu_mul_split(&gate_up, intermediate_size)?;

            // Down projection: [M, intermediate] → [M, hidden]
            let mlp_out = layer.down_proj.forward(&activated)?;

            // Residual add
            hidden = ops::add(&hidden_post_attn, &mlp_out)?;

            // Flush every 4 layers to balance GPU pipelining vs batch overhead
            if (layer_idx + 1) % 4 == 0 || layer_idx + 1 == self.config.num_hidden_layers {
                self.device.flush();
            }
        }

        // 3. Final norm on [M, hidden] → take last row → [1, hidden]
        hidden = ops::rms_norm(&hidden, &weights.norm, eps)?;

        // Extract last row (last token position) via buffer copy
        let last_row = WgpuTensor::zeros(&self.device, &[1, hidden_size], WgpuDType::F32);
        let row_bytes = (hidden_size * 4) as u64; // f32 = 4 bytes
        self.device.batcher.lock().unwrap().push_copy(
            hidden.buffer.clone(),
            (m - 1) as u64 * row_bytes,
            last_row.buffer.clone(),
            0,
            row_bytes,
        );

        // Fused lm_head + argmax on last hidden state
        let max_idx = ops::matvec_argmax_transposed(
            &last_row,
            &weights.lm_head_t,
            hidden_size as u32,
            self.config.vocab_size as u32,
        )
        .await?;

        Ok(max_idx)
    }

    /// Profiled forward pass — syncs GPU at strategic points to measure actual GPU time.
    /// Returns (next_token_id, profile_report_string).
    /// Uses 3 sync points: after attention half, after MLP half, and at layer boundaries.
    /// Estimate total GPU buffer memory used by weights + KV caches (bytes).
    pub fn gpu_buffer_bytes(&self) -> usize {
        let mut total = 0usize;
        if let Some(w) = &self.weights {
            total += w.embed_tokens.size_bytes();
            total += w.norm.size_bytes();
            total += w.lm_head.size_bytes();
            total += w.lm_head_t.size_bytes();
            for layer in &w.layers {
                total += layer.input_layernorm.size_bytes();
                total += layer.post_attention_layernorm.size_bytes();
                total += layer.qkv_proj.weight.size_bytes();
                total += layer.qkv_proj.weight_t.size_bytes();
                if let Some(b) = &layer.qkv_proj.bias {
                    total += b.size_bytes();
                }
                total += layer.o_proj.weight.size_bytes();
                total += layer.o_proj.weight_t.size_bytes();
                if let Some(b) = &layer.o_proj.bias {
                    total += b.size_bytes();
                }
                total += layer.gate_up_proj.weight.size_bytes();
                total += layer.gate_up_proj.weight_t.size_bytes();
                if let Some(b) = &layer.gate_up_proj.bias {
                    total += b.size_bytes();
                }
                total += layer.down_proj.weight.size_bytes();
                total += layer.down_proj.weight_t.size_bytes();
                if let Some(b) = &layer.down_proj.bias {
                    total += b.size_bytes();
                }
            }
        }
        for cache in &self.kv_caches {
            total += cache.k.size_bytes();
            total += cache.v.size_bytes();
        }
        if let Some(c) = &self.cos_cache {
            total += c.size_bytes();
        }
        if let Some(s) = &self.sin_cache {
            total += s.size_bytes();
        }
        total
    }

    pub async fn forward_one_profiled(
        &mut self,
        token_id: u32,
        position: usize,
    ) -> Result<(u32, String, Vec<f64>), crate::WgpuError> {
        use web_time::Instant;

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
        let q_size = num_q_heads * head_dim;
        let kv_size = num_kv_heads * head_dim;
        let num_layers = self.config.num_hidden_layers;

        // Per-layer phase timings
        let mut t_attn_half = 0.0f64; // rope + attention + o_proj
        let mut t_mlp_half = 0.0f64; // fused_gate_up + silu + down_proj + fused_next_qkv
        let mut per_layer_ms: Vec<f64> = Vec::with_capacity(num_layers);

        let sync = || {
            self.device.flush();
            self.device.device.poll(wgpu::Maintain::Wait);
        };

        // 1. Embedding
        let t0 = Instant::now();
        self.device.queue.write_buffer(
            &self.input_id_buf.buffer,
            0,
            bytemuck::cast_slice(&[token_id]),
        );
        let mut hidden = ops::embedding(&weights.embed_tokens, &self.input_id_buf)?;
        hidden = hidden.reshape(&[1, hidden_size])?;
        let normed = ops::rms_norm(&hidden, &weights.layers[0].input_layernorm, eps)?;
        let mut qkv = weights.layers[0].qkv_proj.forward(&normed)?;
        sync();
        let t_embed = t0.elapsed().as_secs_f64() * 1000.0;

        // 2. Layer loop — sync twice per layer to split attention vs MLP
        for layer_idx in 0..num_layers {
            let layer = &weights.layers[layer_idx];
            let cos = self.cos_cache.as_ref().unwrap();
            let sin = self.sin_cache.as_ref().unwrap();
            let cache = &mut self.kv_caches[layer_idx];

            // --- Attention half: rope + attention + o_proj ---
            let t_layer_start = Instant::now();
            let t0 = Instant::now();
            let q_rope_flat = ops::rope_slice_cache(
                &qkv,
                cos,
                sin,
                &cache.k,
                &cache.v,
                q_size,
                kv_size,
                head_dim,
                num_q_heads,
                num_kv_heads,
                position,
                self.config.max_position_embeddings.min(Self::MAX_KV_SEQ),
            )?;
            cache.len = position + 1;
            let attn_output = ops::attention(
                &q_rope_flat,
                &cache.k,
                &cache.v,
                num_q_heads as u32,
                num_kv_heads as u32,
                head_dim as u32,
                cache.len as u32,
            )?;
            let o_out = layer.o_proj.forward(&attn_output)?;
            sync();
            t_attn_half += t0.elapsed().as_secs_f64() * 1000.0;

            // --- MLP half: fused gate_up + silu + down_proj + fused next_qkv ---
            let t0 = Instant::now();
            let (mut gate_up, hidden_new) = ops::fused_add_rms_norm_matvec(
                &hidden,
                &o_out,
                &layer.post_attention_layernorm,
                &layer.gate_up_proj.weight_t,
                (2 * intermediate_size) as u32,
                eps,
            )?;
            hidden = hidden_new;
            if let Some(b) = &layer.gate_up_proj.bias {
                gate_up = ops::add(&gate_up, &b.reshape(&gate_up.shape)?)?;
            }
            let activated = ops::silu_mul_split(&gate_up, intermediate_size)?;
            let mlp_out = layer.down_proj.forward(&activated)?;

            if layer_idx + 1 < num_layers {
                let next_layer = &weights.layers[layer_idx + 1];
                let total_qkv_out = (q_size + 2 * kv_size) as u32;
                let (next_qkv, hidden_new) = ops::fused_add_rms_norm_matvec(
                    &hidden,
                    &mlp_out,
                    &next_layer.input_layernorm,
                    &next_layer.qkv_proj.weight_t,
                    total_qkv_out,
                    eps,
                )?;
                hidden = hidden_new;
                if let Some(b) = &next_layer.qkv_proj.bias {
                    qkv = ops::add(&next_qkv, &b.reshape(&next_qkv.shape)?)?;
                } else {
                    qkv = next_qkv;
                }
            } else {
                hidden = ops::add(&hidden, &mlp_out)?;
            }
            sync();
            t_mlp_half += t0.elapsed().as_secs_f64() * 1000.0;
            per_layer_ms.push(t_layer_start.elapsed().as_secs_f64() * 1000.0);
        }

        // 3. Final norm + lm_head
        let t0 = Instant::now();
        hidden = ops::rms_norm(&hidden, &weights.norm, eps)?;
        sync();
        let t_final_norm = t0.elapsed().as_secs_f64() * 1000.0;

        let t0 = Instant::now();
        let max_idx = ops::matvec_argmax_transposed(
            &hidden,
            &weights.lm_head_t,
            hidden_size as u32,
            self.config.vocab_size as u32,
        )
        .await?;
        let t_lm_head = t0.elapsed().as_secs_f64() * 1000.0;

        let t_layers = t_attn_half + t_mlp_half;
        let t_total = t_embed + t_layers + t_final_norm + t_lm_head;
        let nl = num_layers as f64;

        // Also compute bandwidth utilization
        // Per layer weights: qkv(1152×896) + o(896×896) + gate_up(9728×896) + down(896×4864) = ~17.5M params × 2 bytes
        let params_per_layer = (q_size + 2 * kv_size) * hidden_size  // qkv
            + hidden_size * hidden_size                                // o_proj
            + 2 * intermediate_size * hidden_size                      // gate_up
            + hidden_size * intermediate_size; // down_proj
        let weight_bytes_per_layer = params_per_layer * 2; // f16
        let total_weight_bytes =
            weight_bytes_per_layer * num_layers + self.config.vocab_size * hidden_size * 2; // lm_head
        let layer_time_s = t_layers / 1000.0;
        let total_time_s = t_total / 1000.0;
        let layer_bw = (weight_bytes_per_layer * num_layers) as f64 / layer_time_s / 1e9;
        let total_bw = total_weight_bytes as f64 / total_time_s / 1e9;

        let report = format!(
            "GPU profile (pos={position}, sync per half-layer):\n\
             ┌───────────────────────────────────┬──────────┬──────────┬────────┐\n\
             │ Phase                             │ Total ms │ Per-layer│ % total│\n\
             ├───────────────────────────────────┼──────────┼──────────┼────────┤\n\
             │ embed + first rms_norm + qkv      │ {:7.2} │    —     │ {:5.1}% │\n\
             │ attn half (rope+attn+o_proj) ×{:<2} │ {:7.2} │ {:7.3}  │ {:5.1}% │\n\
             │ MLP half (gate_up→next_qkv)  ×{:<2} │ {:7.2} │ {:7.3}  │ {:5.1}% │\n\
             │ final rms_norm                    │ {:7.2} │    —     │ {:5.1}% │\n\
             │ lm_head + argmax readback         │ {:7.2} │    —     │ {:5.1}% │\n\
             ├───────────────────────────────────┼──────────┼──────────┼────────┤\n\
             │ TOTAL                             │ {:7.2} │          │        │\n\
             │  → {:.1} tok/s equivalent (with sync overhead)\n\
             │  → layers: {:.1} GB/s  total: {:.1} GB/s  (M1 Max peak: 400 GB/s)\n\
             └───────────────────────────────────┴──────────┴──────────┴────────┘",
            t_embed,
            t_embed / t_total * 100.0,
            num_layers,
            t_attn_half,
            t_attn_half / nl,
            t_attn_half / t_total * 100.0,
            num_layers,
            t_mlp_half,
            t_mlp_half / nl,
            t_mlp_half / t_total * 100.0,
            t_final_norm,
            t_final_norm / t_total * 100.0,
            t_lm_head,
            t_lm_head / t_total * 100.0,
            t_total,
            1000.0 / t_total,
            layer_bw,
            total_bw,
        );

        Ok((max_idx, report, per_layer_ms))
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Dequantize any GGUF dtype to f32.
fn dequantize_gguf_to_f32(
    dtype: GgufDType,
    data: &[u8],
    numel: usize,
    name: &str,
) -> Result<Vec<f32>, String> {
    match dtype {
        GgufDType::F32 => Ok(gguf::dequantize_f32_to_f32(data, numel)),
        GgufDType::F16 => Ok(gguf::dequantize_f16_to_f32(data, numel)),
        GgufDType::BF16 => Ok(gguf::dequantize_bf16_to_f32(data, numel)),
        GgufDType::Q4_0 => Ok(gguf::dequantize_q4_0_to_f32(data, numel)),
        GgufDType::Q4_1 => Ok(gguf::dequantize_q4_1_to_f32(data, numel)),
        GgufDType::Q6K => Ok(gguf::dequantize_q6k_to_f32(data, numel)),
        GgufDType::Q8_0 => Ok(gguf::dequantize_q8_0_to_f32(data, numel)),
        other => Err(format!("unsupported dtype {other:?} for {name}")),
    }
}

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

/// Infer base model repo IDs from a GGUF repo name.
/// E.g. "unsloth/Llama-3.2-1B-Instruct-GGUF" → ["meta-llama/Llama-3.2-1B-Instruct", "unsloth/Llama-3.2-1B-Instruct"]
#[cfg(not(target_arch = "wasm32"))]
fn infer_base_model(model_id: &str) -> Vec<String> {
    let mut candidates = Vec::new();
    let parts: Vec<&str> = model_id.splitn(2, '/').collect();
    if parts.len() != 2 {
        return candidates;
    }
    let org = parts[0];
    let name = parts[1];

    // Strip -GGUF suffix
    let base_name = name
        .strip_suffix("-GGUF")
        .or_else(|| name.strip_suffix("-gguf"))
        .unwrap_or(name);

    if base_name == name {
        return candidates; // no GGUF suffix, can't infer
    }

    // Try well-known base orgs for common model families
    let lower = base_name.to_lowercase();
    if lower.starts_with("llama") {
        candidates.push(format!("meta-llama/{base_name}"));
    } else if lower.starts_with("qwen") {
        candidates.push(format!("Qwen/{base_name}"));
    } else if lower.starts_with("mistral") {
        candidates.push(format!("mistralai/{base_name}"));
    } else if lower.starts_with("gemma") {
        candidates.push(format!("google/{base_name}"));
    } else if lower.starts_with("phi") {
        candidates.push(format!("microsoft/{base_name}"));
    }

    // Also try same org with stripped suffix
    candidates.push(format!("{org}/{base_name}"));

    candidates
}
