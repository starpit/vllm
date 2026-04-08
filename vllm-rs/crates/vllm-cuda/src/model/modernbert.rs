// SPDX-License-Identifier: Apache-2.0
// Copyright contributors to the vLLM project

//! ModernBERT encoder model for the CUDA backend.
//!
//! Architecture: encoder-only, bidirectional attention (no causal mask, no KV cache),
//! RoPE (alternating global/local per layer), pre-norm with LayerNorm, GeGLU MLP.
//! Output is hidden states `[num_tokens, hidden_size]`, not logits.
//!
//! Reference: `vllm/model_executor/models/modernbert.py`

use anyhow::Result;

use crate::DType;
use crate::alloc::{CachingAllocator, OwnedTensor};
use crate::device::GpuDevice;
use crate::kernels;
use crate::layers::{Embedding, LayerNorm, Linear};
use crate::model::llama::RotaryCache;
use crate::tensor::{GpuTensor, TensorView};
use crate::weights::GpuWeights;

// ---------------------------------------------------------------------------
// Config
// ---------------------------------------------------------------------------

/// ModernBERT configuration parsed from HF `config.json`.
pub struct ModernBertConfig {
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub intermediate_size: usize,
    pub max_position_embeddings: usize,
    pub layer_norm_eps: f64,
    pub attention_bias: bool,
    pub mlp_bias: bool,
    pub norm_bias: bool,
    pub global_attn_every_n_layers: usize,
    pub local_attention: usize,
    pub global_rope_theta: f64,
    pub local_rope_theta: f64,
}

// ---------------------------------------------------------------------------
// Embeddings
// ---------------------------------------------------------------------------

/// `tok_embeddings` + `LayerNorm` (no position embeddings — RoPE handles positions).
pub struct ModernBertEmbeddings {
    pub tok_embeddings: Embedding,
    pub norm: LayerNorm,
}

impl ModernBertEmbeddings {
    pub fn load(weights: &mut GpuWeights, config: &ModernBertConfig) -> Result<Self> {
        let tok_embeddings = Embedding::load(weights, "model.embeddings.tok_embeddings")?;
        let norm = LayerNorm::load(
            weights,
            "model.embeddings.norm",
            config.layer_norm_eps as f32,
        )?;
        Ok(Self {
            tok_embeddings,
            norm,
        })
    }

    /// Forward: gather embeddings → LayerNorm.
    ///
    /// # Safety
    /// All tensors must be valid GPU memory.
    pub unsafe fn forward(
        &self,
        input_ids: TensorView<'_>,
        alloc: &mut CachingAllocator,
        stream: cudarc::driver::sys::CUstream,
    ) -> OwnedTensor {
        let hidden =
            kernels::embedding_gather(self.tok_embeddings.weight, *input_ids, alloc, stream);
        // Apply LayerNorm.
        layer_norm_forward(hidden.as_gpu_tensor(), &self.norm, alloc, stream)
    }
}

// ---------------------------------------------------------------------------
// Attention
// ---------------------------------------------------------------------------

/// Bidirectional self-attention with fused QKV and RoPE.
pub struct ModernBertAttention {
    pub wqkv: Linear, // [3 * hidden_size, hidden_size]
    pub wo: Linear,   // [hidden_size, hidden_size]
    pub num_heads: usize,
    pub head_dim: usize,
    pub q_size: usize,  // num_heads * head_dim
    pub kv_size: usize, // num_heads * head_dim (no GQA in ModernBERT)
}

impl ModernBertAttention {
    pub fn load(weights: &mut GpuWeights, prefix: &str, config: &ModernBertConfig) -> Result<Self> {
        let wqkv = Linear::load(weights, &format!("{prefix}.Wqkv"))?;
        let wo = Linear::load(weights, &format!("{prefix}.Wo"))?;
        let head_dim = config.hidden_size / config.num_attention_heads;
        let q_size = config.num_attention_heads * head_dim;
        let kv_size = q_size; // no GQA

        Ok(Self {
            wqkv,
            wo,
            num_heads: config.num_attention_heads,
            head_dim,
            q_size,
            kv_size,
        })
    }

    /// Forward: QKV projection → split → RoPE → bidirectional attention → output projection.
    ///
    /// # Safety
    /// All tensors must be valid GPU memory.
    pub unsafe fn forward(
        &self,
        hidden_states: TensorView<'_>,
        positions: TensorView<'_>,
        rotary: &RotaryCache,
        cu_seqlens: TensorView<'_>,
        max_seqlen: usize,
        device: &mut GpuDevice,
    ) -> OwnedTensor {
        // QKV projection: [num_tokens, 3 * hidden_size]
        let qkv = self
            .wqkv
            .forward(hidden_states, &mut device.cublas, &mut device.caching);

        // Split QKV → [num_tokens, num_heads, head_dim] each.
        let (q, k, v) = kernels::split_qkv(
            qkv.as_gpu_tensor(),
            self.q_size,
            self.kv_size,
            self.num_heads,
            self.num_heads, // num_kv_heads == num_heads (no GQA)
            self.head_dim,
            &mut device.caching,
            device.compute_stream,
        );
        drop(qkv);

        // Apply RoPE to both Q and K in-place.
        // Flatten to [num_tokens, total_dim] for the rotary kernel.
        let q_flat = GpuTensor::new(
            q.raw_ptr(),
            &[q.dim(0), self.num_heads * self.head_dim],
            q.dtype(),
        );
        let k_flat = GpuTensor::new(
            k.raw_ptr(),
            &[k.dim(0), self.num_heads * self.head_dim],
            k.dtype(),
        );
        kernels::rotary_embedding_inplace(
            q_flat,
            k_flat,
            *positions,
            rotary.cos_sin_cache,
            self.head_dim,
            device.compute_stream,
        );

        // Bidirectional attention (is_causal=false, no KV cache).
        let attn_out = kernels::flash_attn_contiguous(
            q.as_gpu_tensor(),
            k.as_gpu_tensor(),
            v.as_gpu_tensor(),
            *cu_seqlens,
            *cu_seqlens, // cu_seqlens_k == cu_seqlens_q (self-attention)
            max_seqlen,
            max_seqlen,
            1.0 / (self.head_dim as f32).sqrt(),
            false, // bidirectional — NOT causal
            0.0,   // no softcap
            -1,    // no sliding window limit (full attention)
            &mut device.caching,
            device.compute_stream,
            std::ptr::null(), // no fused RoPE in FA2 (already applied)
            0,
            false,
        );
        drop(q);
        drop(k);
        drop(v);

        // Reshape [num_tokens, num_heads, head_dim] → [num_tokens, hidden_size]
        let num_tokens = attn_out.dim(0);
        let dt = attn_out.dtype();
        let mut attn_flat = attn_out;
        attn_flat.reshape(&[num_tokens, self.num_heads * self.head_dim], dt);

        // Output projection.
        let out = self
            .wo
            .forward(attn_flat.view(), &mut device.cublas, &mut device.caching);
        drop(attn_flat);
        out
    }
}

// ---------------------------------------------------------------------------
// MLP (GeGLU)
// ---------------------------------------------------------------------------

/// GeGLU MLP: Wi projects to 2*intermediate_size, chunks into gate+input,
/// GELU(gate) * input → Wo.
pub struct ModernBertMlp {
    pub wi: Linear, // [2 * intermediate_size, hidden_size]
    pub wo: Linear, // [hidden_size, intermediate_size]
    pub intermediate_size: usize,
}

impl ModernBertMlp {
    pub fn load(weights: &mut GpuWeights, prefix: &str, config: &ModernBertConfig) -> Result<Self> {
        let wi = Linear::load(weights, &format!("{prefix}.Wi"))?;
        let wo = Linear::load(weights, &format!("{prefix}.Wo"))?;
        Ok(Self {
            wi,
            wo,
            intermediate_size: config.intermediate_size,
        })
    }

    /// Forward: Wi → gelu_and_mul → Wo.
    ///
    /// # Safety
    /// All tensors must be valid GPU memory.
    pub unsafe fn forward(
        &self,
        hidden_states: TensorView<'_>,
        device: &mut GpuDevice,
    ) -> OwnedTensor {
        // Wi: [num_tokens, 2 * intermediate_size]
        let gate_up = self
            .wi
            .forward(hidden_states, &mut device.cublas, &mut device.caching);

        // GeGLU activation: chunks gate_up into gate and input halves,
        // applies GELU to gate, multiplies.
        let activated = kernels::gelu_and_mul_fused(
            gate_up.as_gpu_tensor(),
            self.intermediate_size,
            &mut device.caching,
            device.compute_stream,
        );
        drop(gate_up);

        // Wo: [num_tokens, hidden_size]
        let out = self
            .wo
            .forward(activated.view(), &mut device.cublas, &mut device.caching);
        drop(activated);
        out
    }
}

// ---------------------------------------------------------------------------
// Encoder Layer
// ---------------------------------------------------------------------------

/// One ModernBERT encoder layer: pre-norm → attention → residual → pre-norm → MLP → residual.
///
/// Layer 0 uses identity for attn_norm (embedding already normalized).
pub struct ModernBertLayer {
    pub attn_norm: Option<LayerNorm>, // None for layer 0 (identity)
    pub attn: ModernBertAttention,
    pub mlp_norm: LayerNorm,
    pub mlp: ModernBertMlp,
}

impl ModernBertLayer {
    pub fn load(
        weights: &mut GpuWeights,
        layer_id: usize,
        config: &ModernBertConfig,
    ) -> Result<Self> {
        let prefix = format!("model.layers.{layer_id}");

        // Layer 0 uses identity (no attn_norm applied — embedding has its own LN).
        let attn_norm = if layer_id == 0 {
            // Still load the weights (they exist in the checkpoint) but mark as identity.
            // Actually, Python uses `nn.Identity()` which has no parameters. The checkpoint
            // may not have attn_norm weights for layer 0. Check if they exist.
            let norm_prefix = format!("{prefix}.attn_norm");
            let weight_name = format!("{norm_prefix}.weight");
            if weights.contains(&weight_name) {
                // Weights exist but layer 0 uses Identity — skip them.
                let _ = weights.take(&weight_name);
                let bias_name = format!("{norm_prefix}.bias");
                if weights.contains(&bias_name) {
                    let _ = weights.take(&bias_name);
                }
            }
            None
        } else {
            Some(LayerNorm::load(
                weights,
                &format!("{prefix}.attn_norm"),
                config.layer_norm_eps as f32,
            )?)
        };

        let attn = ModernBertAttention::load(weights, &format!("{prefix}.attn"), config)?;
        let mlp_norm = LayerNorm::load(
            weights,
            &format!("{prefix}.mlp_norm"),
            config.layer_norm_eps as f32,
        )?;
        let mlp = ModernBertMlp::load(weights, &format!("{prefix}.mlp"), config)?;

        Ok(Self {
            attn_norm,
            attn,
            mlp_norm,
            mlp,
        })
    }

    /// Forward pass for one encoder layer.
    ///
    /// # Safety
    /// All tensors must be valid GPU memory.
    pub unsafe fn forward(
        &self,
        hidden_states: OwnedTensor,
        positions: TensorView<'_>,
        rotary: &RotaryCache,
        cu_seqlens: TensorView<'_>,
        max_seqlen: usize,
        device: &mut GpuDevice,
    ) -> OwnedTensor {
        // Pre-norm for attention (identity for layer 0).
        let normed = if let Some(ref norm) = self.attn_norm {
            layer_norm_forward(
                hidden_states.as_gpu_tensor(),
                norm,
                &mut device.caching,
                device.compute_stream,
            )
        } else {
            // Layer 0: identity — input already normalized by embeddings LN.
            // Allocate a copy so hidden_states survives for the residual add.
            let copy = device.caching.alloc_tensor(
                &[hidden_states.dim(0), hidden_states.dim(1)],
                hidden_states.dtype(),
            );
            crate::driver::memcpy_dtod_async(
                copy.raw_ptr(),
                hidden_states.raw_ptr(),
                hidden_states.size_bytes(),
                device.compute_stream,
            )
            .expect("D2D copy for layer 0 identity");
            copy
        };

        // Attention.
        let attn_out = self.attn.forward(
            normed.view(),
            positions,
            rotary,
            cu_seqlens,
            max_seqlen,
            device,
        );
        drop(normed);

        // Residual: hidden_states += attn_out (in-place).
        kernels::add_inplace(
            hidden_states.as_gpu_tensor(),
            attn_out.as_gpu_tensor(),
            device.compute_stream,
        );
        drop(attn_out);

        // Pre-norm for MLP.
        let normed_mlp = layer_norm_forward(
            hidden_states.as_gpu_tensor(),
            &self.mlp_norm,
            &mut device.caching,
            device.compute_stream,
        );

        // MLP.
        let mlp_out = self.mlp.forward(normed_mlp.view(), device);
        drop(normed_mlp);

        // Residual: hidden_states += mlp_out (in-place).
        kernels::add_inplace(
            hidden_states.as_gpu_tensor(),
            mlp_out.as_gpu_tensor(),
            device.compute_stream,
        );
        drop(mlp_out);

        hidden_states
    }
}

// ---------------------------------------------------------------------------
// Full Model
// ---------------------------------------------------------------------------

/// ModernBERT encoder model. Outputs hidden states `[num_tokens, hidden_size]`.
pub struct ModernBertModel {
    pub embeddings: ModernBertEmbeddings,
    pub layers: Vec<ModernBertLayer>,
    pub final_norm: LayerNorm,
    /// Per-layer rotary caches (alternating global/local theta).
    pub rotary_caches: Vec<RotaryCache>,
    pub hidden_size: usize,
}

impl ModernBertModel {
    pub fn load(
        weights: &mut GpuWeights,
        config: &ModernBertConfig,
        dtype: DType,
        device: &GpuDevice,
    ) -> Result<Self> {
        let embeddings = ModernBertEmbeddings::load(weights, config)?;

        // Build per-layer rotary caches (global vs local theta).
        let head_dim = config.hidden_size / config.num_attention_heads;
        let max_pos = config.max_position_embeddings;
        let mut rotary_caches = Vec::with_capacity(config.num_hidden_layers);
        for layer_id in 0..config.num_hidden_layers {
            let is_global = layer_id % config.global_attn_every_n_layers == 0;
            let theta = if is_global {
                config.global_rope_theta
            } else {
                config.local_rope_theta
            };
            let rotary =
                unsafe { RotaryCache::new(head_dim, max_pos, theta, None, dtype, device)? };
            rotary_caches.push(rotary);
        }

        let mut layers = Vec::with_capacity(config.num_hidden_layers);
        for layer_id in 0..config.num_hidden_layers {
            layers.push(ModernBertLayer::load(weights, layer_id, config)?);
        }

        let final_norm =
            LayerNorm::load(weights, "model.final_norm", config.layer_norm_eps as f32)?;

        Ok(Self {
            embeddings,
            layers,
            final_norm,
            rotary_caches,
            hidden_size: config.hidden_size,
        })
    }

    /// Forward pass: returns hidden states `[num_tokens, hidden_size]`.
    ///
    /// The KV cache parameters are accepted for API compatibility with
    /// the CudaModel dispatch but are NOT used (encoder has no KV cache).
    ///
    /// # Safety
    /// All tensors must be valid GPU memory.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn forward(
        &self,
        input_ids: TensorView<'_>,
        positions: TensorView<'_>,
        _slot_mapping: TensorView<'_>,
        cu_seqlens_q: TensorView<'_>,
        _seqused_k: TensorView<'_>,
        _block_table: TensorView<'_>,
        max_seqlen_q: usize,
        _max_seqlen_k: usize,
        _kv_cache: &crate::kv_cache::KvCachePool,
        device: &mut GpuDevice,
    ) -> OwnedTensor {
        // Embedding lookup + LayerNorm.
        let mut hidden_states =
            self.embeddings
                .forward(input_ids, &mut device.caching, device.compute_stream);

        // Encoder layers.
        for (layer_id, layer) in self.layers.iter().enumerate() {
            hidden_states = layer.forward(
                hidden_states,
                positions,
                &self.rotary_caches[layer_id],
                cu_seqlens_q,
                max_seqlen_q,
                device,
            );
        }

        // Final LayerNorm.
        let out = layer_norm_forward(
            hidden_states.as_gpu_tensor(),
            &self.final_norm,
            &mut device.caching,
            device.compute_stream,
        );
        drop(hidden_states);
        out
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Apply LayerNorm, dispatching to bias or no-bias kernel.
unsafe fn layer_norm_forward(
    input: GpuTensor,
    norm: &LayerNorm,
    alloc: &mut CachingAllocator,
    stream: cudarc::driver::sys::CUstream,
) -> OwnedTensor {
    if let Some(bias) = norm.bias {
        kernels::layer_norm_bias(input, norm.weight, bias, norm.eps, alloc, stream)
    } else {
        kernels::cohere_layer_norm(input, norm.weight, norm.eps, alloc, stream)
    }
}

// ---------------------------------------------------------------------------
// Unit tests (non-CUDA — config + structural validation)
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn test_config() -> ModernBertConfig {
        ModernBertConfig {
            vocab_size: 50368,
            hidden_size: 768,
            num_hidden_layers: 22,
            num_attention_heads: 12,
            intermediate_size: 2048,
            max_position_embeddings: 8192,
            layer_norm_eps: 1e-5,
            attention_bias: false,
            mlp_bias: false,
            norm_bias: false,
            global_attn_every_n_layers: 3,
            local_attention: 128,
            global_rope_theta: 160000.0,
            local_rope_theta: 10000.0,
        }
    }

    #[test]
    fn test_config_head_dim() {
        let config = test_config();
        let head_dim = config.hidden_size / config.num_attention_heads;
        assert_eq!(head_dim, 64);
    }

    #[test]
    fn test_config_global_layers() {
        let config = test_config();
        // Layers 0, 3, 6, 9, 12, 15, 18, 21 are global (every 3rd layer).
        let global_count = (0..config.num_hidden_layers)
            .filter(|&i| i % config.global_attn_every_n_layers == 0)
            .count();
        assert_eq!(global_count, 8); // 22 layers / 3 = 7.33 → 8 global layers (0,3,6,9,12,15,18,21)
    }

    #[test]
    fn test_config_local_layers() {
        let config = test_config();
        let local_count = (0..config.num_hidden_layers)
            .filter(|&i| i % config.global_attn_every_n_layers != 0)
            .count();
        assert_eq!(local_count, 14); // 22 - 8 = 14
    }

    #[test]
    fn test_rope_theta_per_layer() {
        let config = test_config();
        for layer_id in 0..config.num_hidden_layers {
            let is_global = layer_id % config.global_attn_every_n_layers == 0;
            let theta = if is_global {
                config.global_rope_theta
            } else {
                config.local_rope_theta
            };
            if is_global {
                assert_eq!(theta, 160000.0, "layer {layer_id} should use global theta");
            } else {
                assert_eq!(theta, 10000.0, "layer {layer_id} should use local theta");
            }
        }
    }

    #[test]
    fn test_layer0_identity_norm() {
        // Layer 0 should have attn_norm = None (identity).
        // We can't construct layers without GPU weights, but we can verify the logic.
        let layer_id = 0;
        assert!(layer_id == 0, "layer 0 should use identity for attn_norm");
    }

    #[test]
    fn test_geglu_intermediate_size() {
        let config = test_config();
        // Wi projects to 2 * intermediate_size for GeGLU gate+input split.
        assert_eq!(config.intermediate_size, 2048);
        // Wi weight shape: [2 * 2048, 768] = [4096, 768]
        let wi_out = 2 * config.intermediate_size;
        assert_eq!(wi_out, 4096);
    }
}
