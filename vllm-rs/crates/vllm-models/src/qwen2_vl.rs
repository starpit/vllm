// SPDX-License-Identifier: Apache-2.0
// Copyright contributors to the vLLM project

//! Qwen2-VL and Qwen2.5-VL multimodal (vision-language) model for Candle.
//!
//! Implements `Qwen2VLForConditionalGeneration` and `Qwen2_5_VLForConditionalGeneration`
//! which wrap:
//! - `Qwen2VisionTransformer` — custom ViT with 3D patch embedding + 2D RoPE + PatchMerger
//! - `LlamaForCausalLM` — Qwen2 text backbone (architecturally identical to LLaMA)
//!
//! Key differences from Gemma3 VLM:
//! - Vision encoder: Custom ViT with 3D patch embed (Conv3d-as-Linear) + 2D RoPE
//! - Projector: PatchMerger (2x2 spatial merge + 2-layer MLP with GELU)
//! - LLM RoPE: M-RoPE (3-section position encoding: time, height, width)
//!
//! Weight prefix mapping (HF checkpoint → code):
//! - `visual.*` → vision encoder (blocks, merger, patch_embed)
//! - `model.*` → text backbone (Qwen2/LLaMA)
//! - `lm_head.*` → language model head

use candle_core::{DType, Device, Module, Tensor};

use vllm_common::multimodal::MultimodalData;
use vllm_model::error::{ModelError, ModelResult};
use vllm_model::layers::{LayerNorm, Linear, RmsNorm};
use vllm_model::weight::{HfModelConfig, ModelWeights};

use crate::llama::{LlamaConfig, LlamaForCausalLM, LlamaModel};
use crate::qwen2::Qwen2Config;

// ---------------------------------------------------------------------------
// Config
// ---------------------------------------------------------------------------

/// Parsed Qwen2-VL vision encoder configuration.
#[derive(Debug, Clone)]
struct Qwen2VisionConfig {
    hidden_size: usize, // "embed_dim" in HF config
    num_heads: usize,
    depth: usize, // number of vision transformer blocks
    patch_size: usize,
    temporal_patch_size: usize,
    in_channels: usize,
    #[allow(dead_code)]
    mlp_ratio: f64, // intermediate_size multiplier
    /// Merge factor: spatial_merge_size (2 for 2x2)
    spatial_merge_size: usize,
    /// Whether to use Qwen2.5-VL encoder variant (RMSNorm + SwiGLU + windowed attn).
    is_qwen25: bool,
    /// Full-attention block indices for Qwen2.5-VL (empty for Qwen2-VL).
    #[allow(dead_code)]
    fullatt_block_indexes: Vec<usize>,
}

impl Qwen2VisionConfig {
    fn from_json(value: &serde_json::Value, is_qwen25: bool) -> ModelResult<Self> {
        let get_usize = |key: &str, default: usize| -> usize {
            value
                .get(key)
                .and_then(|v| v.as_u64())
                .map(|v| v as usize)
                .unwrap_or(default)
        };
        let get_f64 = |key: &str, default: f64| -> f64 {
            value.get(key).and_then(|v| v.as_f64()).unwrap_or(default)
        };

        let fullatt_block_indexes = if is_qwen25 {
            value
                .get("fullatt_block_indexes")
                .and_then(|v| v.as_array())
                .map(|arr| {
                    arr.iter()
                        .filter_map(|v| v.as_u64().map(|x| x as usize))
                        .collect()
                })
                .unwrap_or_default()
        } else {
            vec![]
        };

        Ok(Self {
            hidden_size: get_usize("embed_dim", 1280),
            num_heads: get_usize("num_heads", 16),
            depth: get_usize("depth", 32),
            patch_size: get_usize("patch_size", 14),
            temporal_patch_size: get_usize("temporal_patch_size", 2),
            in_channels: get_usize("in_channels", 3),
            mlp_ratio: get_f64("mlp_ratio", 4.0),
            spatial_merge_size: get_usize("spatial_merge_size", 2),
            is_qwen25,
            fullatt_block_indexes,
        })
    }

    #[allow(dead_code)]
    fn intermediate_size(&self) -> usize {
        (self.hidden_size as f64 * self.mlp_ratio) as usize
    }

    fn head_dim(&self) -> usize {
        self.hidden_size / self.num_heads
    }
}

/// Full Qwen2-VL multimodal config.
struct Qwen2VLConfig {
    vision_config: Qwen2VisionConfig,
    text_config: LlamaConfig,
    /// M-RoPE section sizes [s0, s1, s2] where s0+s1+s2 = head_dim/2.
    mrope_sections: [usize; 3],
    /// Merge factor for spatial patches.
    spatial_merge_size: usize,
}

impl Qwen2VLConfig {
    fn from_hf_config(config: &HfModelConfig, is_qwen25: bool) -> ModelResult<Self> {
        let vision_json = config
            .extra
            .get("vision_config")
            .ok_or_else(|| ModelError::Other("missing vision_config".into()))?;
        let vision_config = Qwen2VisionConfig::from_json(vision_json, is_qwen25)?;

        // Parse text config from the top-level config (Qwen2 style).
        let text_config = Qwen2Config::from_hf_config(config)?.0;

        // M-RoPE sections from config.json's "rope_scaling.mrope_section".
        let mrope_sections = config
            .extra
            .get("rope_scaling")
            .and_then(|rs| rs.get("mrope_section"))
            .and_then(|ms| ms.as_array())
            .map(|arr| {
                let vals: Vec<usize> = arr
                    .iter()
                    .filter_map(|v| v.as_u64().map(|x| x as usize))
                    .collect();
                if vals.len() >= 3 {
                    [vals[0], vals[1], vals[2]]
                } else {
                    // Default sections for head_dim=128: [16, 24, 24]
                    [16, 24, 24]
                }
            })
            .unwrap_or([16, 24, 24]);

        let spatial_merge_size = vision_config.spatial_merge_size;

        Ok(Self {
            vision_config,
            text_config,
            mrope_sections,
            spatial_merge_size,
        })
    }
}

// ---------------------------------------------------------------------------
// Softmax helper
// ---------------------------------------------------------------------------

fn softmax_last_dim(x: &Tensor) -> ModelResult<Tensor> {
    let max = x
        .max_keepdim(candle_core::D::Minus1)
        .map_err(ModelError::Candle)?;
    let shifted = x.broadcast_sub(&max).map_err(ModelError::Candle)?;
    let exp = shifted.exp().map_err(ModelError::Candle)?;
    let sum = exp
        .sum_keepdim(candle_core::D::Minus1)
        .map_err(ModelError::Candle)?;
    exp.broadcast_div(&sum).map_err(ModelError::Candle)
}

/// QuickGELU activation: x * sigmoid(1.702 * x)
fn quick_gelu(x: &Tensor) -> ModelResult<Tensor> {
    let scaled = (x * 1.702).map_err(ModelError::Candle)?;
    // sigmoid(y) = 1 / (1 + exp(-y))
    let sigmoid = scaled
        .neg()
        .map_err(ModelError::Candle)?
        .exp()
        .map_err(ModelError::Candle)?;
    let sigmoid = (sigmoid + 1.0)
        .map_err(ModelError::Candle)?
        .recip()
        .map_err(ModelError::Candle)?;
    x.mul(&sigmoid).map_err(ModelError::Candle)
}

// ---------------------------------------------------------------------------
// Qwen2VisionPatchEmbed
// ---------------------------------------------------------------------------

/// "Conv3d" patch embedding implemented as reshape + Linear.
///
/// Input: flattened patches `[L, C * temporal_patch_size * patch_size * patch_size]`
/// Output: `[L, embed_dim]`
struct Qwen2VisionPatchEmbed {
    proj: Linear,
}

impl Qwen2VisionPatchEmbed {
    fn load(weights: &ModelWeights, prefix: &str, dtype: DType) -> ModelResult<Self> {
        let proj = Linear::load(weights, &format!("{prefix}.proj"), dtype)?;
        Ok(Self { proj })
    }

    fn forward(&self, x: &Tensor) -> ModelResult<Tensor> {
        // x: [L, patch_dim] → [L, embed_dim]
        self.proj.forward(x).map_err(ModelError::Candle)
    }
}

// ---------------------------------------------------------------------------
// Qwen2VisionAttention
// ---------------------------------------------------------------------------

/// Multi-head attention for the vision encoder with packed QKV.
struct Qwen2VisionAttention {
    qkv: Linear,
    proj: Linear,
    num_heads: usize,
    head_dim: usize,
}

impl Qwen2VisionAttention {
    fn load(
        weights: &ModelWeights,
        prefix: &str,
        config: &Qwen2VisionConfig,
        dtype: DType,
    ) -> ModelResult<Self> {
        let qkv = Linear::load(weights, &format!("{prefix}.qkv"), dtype)?;
        let proj = Linear::load(weights, &format!("{prefix}.proj"), dtype)?;
        Ok(Self {
            qkv,
            proj,
            num_heads: config.num_heads,
            head_dim: config.head_dim(),
        })
    }

    /// Bidirectional multi-head attention with optional 2D RoPE.
    ///
    /// * `x` — `[L, hidden_size]`
    /// * `rotary_pos_emb` — optional `(cos, sin)` each `[L, head_dim]` for partial RoPE
    fn forward(
        &self,
        x: &Tensor,
        rotary_pos_emb: Option<(&Tensor, &Tensor)>,
    ) -> ModelResult<Tensor> {
        let seq_len = x.dim(0).map_err(ModelError::Candle)?;
        let hidden = self.num_heads * self.head_dim;

        // Packed QKV: [L, 3*hidden] → split into Q, K, V each [L, hidden]
        let qkv = self.qkv.forward(x).map_err(ModelError::Candle)?;
        let q = qkv.narrow(1, 0, hidden).map_err(ModelError::Candle)?;
        let k = qkv.narrow(1, hidden, hidden).map_err(ModelError::Candle)?;
        let v = qkv
            .narrow(1, 2 * hidden, hidden)
            .map_err(ModelError::Candle)?;

        // Reshape to [L, num_heads, head_dim]
        let q = q
            .reshape((seq_len, self.num_heads, self.head_dim))
            .map_err(ModelError::Candle)?;
        let k = k
            .reshape((seq_len, self.num_heads, self.head_dim))
            .map_err(ModelError::Candle)?;
        let v = v
            .reshape((seq_len, self.num_heads, self.head_dim))
            .map_err(ModelError::Candle)?;

        // Apply partial rotary embedding (first half of head_dim only).
        let (q, k) = if let Some((cos, sin)) = rotary_pos_emb {
            let rot_dim = cos.dim(1).map_err(ModelError::Candle)?;
            let q_rot = q.narrow(2, 0, rot_dim).map_err(ModelError::Candle)?;
            let q_pass = q
                .narrow(2, rot_dim, self.head_dim - rot_dim)
                .map_err(ModelError::Candle)?;
            let k_rot = k.narrow(2, 0, rot_dim).map_err(ModelError::Candle)?;
            let k_pass = k
                .narrow(2, rot_dim, self.head_dim - rot_dim)
                .map_err(ModelError::Candle)?;

            let q_rot = vllm_model::layers::rotary::apply_rotary_to_tensor(&q_rot, cos, sin)?;
            let k_rot = vllm_model::layers::rotary::apply_rotary_to_tensor(&k_rot, cos, sin)?;

            let q = Tensor::cat(&[&q_rot, &q_pass], 2).map_err(ModelError::Candle)?;
            let k = Tensor::cat(&[&k_rot, &k_pass], 2).map_err(ModelError::Candle)?;
            (q, k)
        } else {
            (q, k)
        };

        // Transpose to [num_heads, L, head_dim] for batched matmul.
        let q = q
            .transpose(0, 1)
            .map_err(ModelError::Candle)?
            .contiguous()
            .map_err(ModelError::Candle)?;
        let k = k
            .transpose(0, 1)
            .map_err(ModelError::Candle)?
            .contiguous()
            .map_err(ModelError::Candle)?;
        let v = v
            .transpose(0, 1)
            .map_err(ModelError::Candle)?
            .contiguous()
            .map_err(ModelError::Candle)?;

        // Scaled dot-product attention (bidirectional).
        let scale = (self.head_dim as f64).powf(-0.5);
        let k_tr = k
            .transpose(1, 2)
            .map_err(ModelError::Candle)?
            .contiguous()
            .map_err(ModelError::Candle)?;
        let scores = q.matmul(&k_tr).map_err(ModelError::Candle)?;
        let scores = (scores * scale).map_err(ModelError::Candle)?;

        let scores = scores.to_dtype(DType::F32).map_err(ModelError::Candle)?;
        let attn_weights = softmax_last_dim(&scores)?;
        let attn_weights = attn_weights
            .to_dtype(v.dtype())
            .map_err(ModelError::Candle)?;

        let attn_output = attn_weights.matmul(&v).map_err(ModelError::Candle)?;

        // Transpose back: [num_heads, L, head_dim] → [L, num_heads * head_dim]
        let attn_output = attn_output
            .transpose(0, 1)
            .map_err(ModelError::Candle)?
            .reshape((seq_len, hidden))
            .map_err(ModelError::Candle)?;

        self.proj.forward(&attn_output).map_err(ModelError::Candle)
    }
}

// ---------------------------------------------------------------------------
// Qwen2VisionMLP (QuickGELU variant for Qwen2-VL)
// ---------------------------------------------------------------------------

struct Qwen2VisionMLP {
    fc1: Linear,
    fc2: Linear,
}

impl Qwen2VisionMLP {
    fn load(weights: &ModelWeights, prefix: &str, dtype: DType) -> ModelResult<Self> {
        let fc1 = Linear::load(weights, &format!("{prefix}.fc1"), dtype)?;
        let fc2 = Linear::load(weights, &format!("{prefix}.fc2"), dtype)?;
        Ok(Self { fc1, fc2 })
    }

    fn forward(&self, x: &Tensor) -> ModelResult<Tensor> {
        let x = self.fc1.forward(x).map_err(ModelError::Candle)?;
        let x = quick_gelu(&x)?;
        self.fc2.forward(&x).map_err(ModelError::Candle)
    }
}

// ---------------------------------------------------------------------------
// Qwen2VisionBlock
// ---------------------------------------------------------------------------

/// Pre-norm transformer block for the vision encoder.
///
/// Qwen2-VL: LayerNorm + Attention + LayerNorm + QuickGELU MLP
struct Qwen2VisionBlock {
    norm1: LayerNorm,
    attn: Qwen2VisionAttention,
    norm2: LayerNorm,
    mlp: Qwen2VisionMLP,
}

impl Qwen2VisionBlock {
    fn load(
        weights: &ModelWeights,
        prefix: &str,
        config: &Qwen2VisionConfig,
        dtype: DType,
    ) -> ModelResult<Self> {
        let eps = 1e-6;
        let norm1 = LayerNorm::load(weights, &format!("{prefix}.norm1"), eps, dtype)?;
        let attn = Qwen2VisionAttention::load(weights, &format!("{prefix}.attn"), config, dtype)?;
        let norm2 = LayerNorm::load(weights, &format!("{prefix}.norm2"), eps, dtype)?;
        let mlp = Qwen2VisionMLP::load(weights, &format!("{prefix}.mlp"), dtype)?;

        Ok(Self {
            norm1,
            attn,
            norm2,
            mlp,
        })
    }

    fn forward(
        &self,
        hidden_states: &Tensor,
        rotary_pos_emb: Option<(&Tensor, &Tensor)>,
    ) -> ModelResult<Tensor> {
        let residual = hidden_states.clone();
        let x = self
            .norm1
            .forward(hidden_states)
            .map_err(ModelError::Candle)?;
        let x = self.attn.forward(&x, rotary_pos_emb)?;
        let hidden_states = (residual + x).map_err(ModelError::Candle)?;

        let residual = hidden_states.clone();
        let x = self
            .norm2
            .forward(&hidden_states)
            .map_err(ModelError::Candle)?;
        let x = self.mlp.forward(&x)?;
        (residual + x).map_err(ModelError::Candle)
    }
}

// ---------------------------------------------------------------------------
// Qwen2_5VisionBlock (RMSNorm + SwiGLU MLP variant)
// ---------------------------------------------------------------------------

/// Qwen2.5-VL vision block: RMSNorm + SwiGLU MLP (instead of LayerNorm + QuickGELU).
struct Qwen25VisionBlock {
    norm1: RmsNorm,
    attn: Qwen2VisionAttention,
    norm2: RmsNorm,
    /// SwiGLU: gate_proj + up_proj → SiLU * gate → down_proj
    gate_proj: Linear,
    up_proj: Linear,
    down_proj: Linear,
}

impl Qwen25VisionBlock {
    fn load(
        weights: &ModelWeights,
        prefix: &str,
        config: &Qwen2VisionConfig,
        dtype: DType,
    ) -> ModelResult<Self> {
        let eps = 1e-6;
        let norm1 = RmsNorm::load(weights, &format!("{prefix}.norm1"), eps, dtype)?;
        let attn = Qwen2VisionAttention::load(weights, &format!("{prefix}.attn"), config, dtype)?;
        let norm2 = RmsNorm::load(weights, &format!("{prefix}.norm2"), eps, dtype)?;

        let gate_proj = Linear::load(weights, &format!("{prefix}.mlp.gate_proj"), dtype)?;
        let up_proj = Linear::load(weights, &format!("{prefix}.mlp.up_proj"), dtype)?;
        let down_proj = Linear::load(weights, &format!("{prefix}.mlp.down_proj"), dtype)?;

        Ok(Self {
            norm1,
            attn,
            norm2,
            gate_proj,
            up_proj,
            down_proj,
        })
    }

    fn forward(
        &self,
        hidden_states: &Tensor,
        rotary_pos_emb: Option<(&Tensor, &Tensor)>,
    ) -> ModelResult<Tensor> {
        let residual = hidden_states.clone();
        let x = crate::ops::rms_norm(hidden_states, &self.norm1).map_err(ModelError::Candle)?;
        let x = self.attn.forward(&x, rotary_pos_emb)?;
        let hidden_states = (residual + x).map_err(ModelError::Candle)?;

        let residual = hidden_states.clone();
        let x = crate::ops::rms_norm(&hidden_states, &self.norm2).map_err(ModelError::Candle)?;
        // SwiGLU MLP: silu(gate_proj(x)) * up_proj(x) → down_proj
        let gate = self.gate_proj.forward(&x).map_err(ModelError::Candle)?;
        let gate = gate.silu().map_err(ModelError::Candle)?;
        let up = self.up_proj.forward(&x).map_err(ModelError::Candle)?;
        let x = (gate * up).map_err(ModelError::Candle)?;
        let x = self.down_proj.forward(&x).map_err(ModelError::Candle)?;
        (residual + x).map_err(ModelError::Candle)
    }
}

// ---------------------------------------------------------------------------
// Qwen2VisionPatchMerger
// ---------------------------------------------------------------------------

/// Spatial 2x2 patch merger: groups 2x2 patches → MLP projection.
///
/// Weights: `visual.merger.ln_q`, `visual.merger.mlp.0`, `visual.merger.mlp.2`
struct Qwen2VisionPatchMerger {
    ln_q: LayerNorm,
    mlp_fc1: Linear,
    mlp_fc2: Linear,
    spatial_merge_size: usize,
}

impl Qwen2VisionPatchMerger {
    fn load(
        weights: &ModelWeights,
        prefix: &str,
        config: &Qwen2VisionConfig,
        text_hidden_size: usize,
        dtype: DType,
    ) -> ModelResult<Self> {
        let ln_q = LayerNorm::load(weights, &format!("{prefix}.ln_q"), 1e-6, dtype)?;
        let mlp_fc1 = Linear::load(weights, &format!("{prefix}.mlp.0"), dtype)?;
        let mlp_fc2 = Linear::load(weights, &format!("{prefix}.mlp.2"), dtype)?;
        let _ = text_hidden_size;

        Ok(Self {
            ln_q,
            mlp_fc1,
            mlp_fc2,
            spatial_merge_size: config.spatial_merge_size,
        })
    }

    /// Forward: patches `[L, embed_dim]` with grid `(grid_h, grid_w)` → `[L', text_hidden]`.
    ///
    /// Groups 2x2 patches, concatenates, applies LayerNorm + MLP.
    fn forward(&self, x: &Tensor, grid_h: usize, grid_w: usize) -> ModelResult<Tensor> {
        let embed_dim = x.dim(1).map_err(ModelError::Candle)?;
        let m = self.spatial_merge_size;
        let out_h = grid_h / m;
        let out_w = grid_w / m;

        // Reshape [grid_h * grid_w, embed_dim] → [out_h, m, out_w, m, embed_dim]
        let x = x
            .reshape((out_h, m, out_w, m, embed_dim))
            .map_err(ModelError::Candle)?;
        // Permute to [out_h, out_w, m, m, embed_dim]
        let x = x.permute([0, 2, 1, 3, 4]).map_err(ModelError::Candle)?;
        // Flatten: [out_h * out_w, m * m * embed_dim]
        let merged_tokens = out_h * out_w;
        let merged_dim = m * m * embed_dim;
        let x = x
            .reshape((merged_tokens, merged_dim))
            .map_err(ModelError::Candle)?;

        // LayerNorm on embed_dim — reshape to apply per-channel.
        // Actually, ln_q operates on embed_dim, so reshape to [merged_tokens * m * m, embed_dim].
        let x_for_norm = x
            .reshape((merged_tokens * m * m, embed_dim))
            .map_err(ModelError::Candle)?;
        let x_normed = self.ln_q.forward(&x_for_norm).map_err(ModelError::Candle)?;
        let x = x_normed
            .reshape((merged_tokens, merged_dim))
            .map_err(ModelError::Candle)?;

        // MLP: Linear → GELU → Linear
        let x = self.mlp_fc1.forward(&x).map_err(ModelError::Candle)?;
        let x = x.gelu().map_err(ModelError::Candle)?;
        self.mlp_fc2.forward(&x).map_err(ModelError::Candle)
    }
}

// ---------------------------------------------------------------------------
// VisionBlock enum (dispatches between Qwen2 and Qwen2.5 variants)
// ---------------------------------------------------------------------------

enum VisionBlock {
    Qwen2(Qwen2VisionBlock),
    Qwen25(Qwen25VisionBlock),
}

impl VisionBlock {
    fn forward(
        &self,
        hidden_states: &Tensor,
        rotary_pos_emb: Option<(&Tensor, &Tensor)>,
    ) -> ModelResult<Tensor> {
        match self {
            VisionBlock::Qwen2(b) => b.forward(hidden_states, rotary_pos_emb),
            VisionBlock::Qwen25(b) => b.forward(hidden_states, rotary_pos_emb),
        }
    }
}

// ---------------------------------------------------------------------------
// Qwen2VisionTransformer
// ---------------------------------------------------------------------------

/// Full vision encoder: PatchEmbed → blocks with 2D RoPE → PatchMerger.
struct Qwen2VisionTransformer {
    patch_embed: Qwen2VisionPatchEmbed,
    blocks: Vec<VisionBlock>,
    merger: Qwen2VisionPatchMerger,
    /// Precomputed 2D RoPE cos/sin.
    rotary_cos: Tensor,
    rotary_sin: Tensor,
}

impl Qwen2VisionTransformer {
    fn load(
        weights: &ModelWeights,
        prefix: &str,
        config: &Qwen2VisionConfig,
        text_hidden_size: usize,
        dtype: DType,
        device: &Device,
    ) -> ModelResult<Self> {
        let patch_embed =
            Qwen2VisionPatchEmbed::load(weights, &format!("{prefix}.patch_embed"), dtype)?;

        let mut blocks = Vec::with_capacity(config.depth);
        for i in 0..config.depth {
            let block = if config.is_qwen25 {
                VisionBlock::Qwen25(Qwen25VisionBlock::load(
                    weights,
                    &format!("{prefix}.blocks.{i}"),
                    config,
                    dtype,
                )?)
            } else {
                VisionBlock::Qwen2(Qwen2VisionBlock::load(
                    weights,
                    &format!("{prefix}.blocks.{i}"),
                    config,
                    dtype,
                )?)
            };
            blocks.push(block);
        }

        let merger = Qwen2VisionPatchMerger::load(
            weights,
            &format!("{prefix}.merger"),
            config,
            text_hidden_size,
            dtype,
        )?;

        // Precompute 2D RoPE for vision encoder.
        // Uses half of head_dim for rotation (partial_rotary_factor = 0.5).
        let rot_dim = config.head_dim() / 2;
        let max_grid = 4096; // generous upper bound for grid positions
        let (rotary_cos, rotary_sin) =
            Self::precompute_2d_rope(rot_dim, max_grid, 10000.0, dtype, device)?;

        Ok(Self {
            patch_embed,
            blocks,
            merger,
            rotary_cos,
            rotary_sin,
        })
    }

    /// Precompute 2D RoPE cos/sin tables.
    ///
    /// Returns `(cos, sin)` each of shape `[max_grid, rot_dim]`.
    fn precompute_2d_rope(
        rot_dim: usize,
        max_grid: usize,
        base: f64,
        dtype: DType,
        device: &Device,
    ) -> ModelResult<(Tensor, Tensor)> {
        let half_rot = rot_dim / 2;
        let inv_freq: Vec<f32> = (0..half_rot)
            .map(|i| (1.0 / base.powf(2.0 * i as f64 / rot_dim as f64)) as f32)
            .collect();

        let inv_freq_tensor =
            Tensor::from_slice(&inv_freq, half_rot, device).map_err(ModelError::Candle)?;

        let positions: Vec<f32> = (0..max_grid).map(|p| p as f32).collect();
        let pos_tensor =
            Tensor::from_slice(&positions, max_grid, device).map_err(ModelError::Candle)?;

        let pos_2d = pos_tensor
            .reshape((max_grid, 1))
            .map_err(ModelError::Candle)?;
        let inv_freq_2d = inv_freq_tensor
            .reshape((1, half_rot))
            .map_err(ModelError::Candle)?;
        let freqs = pos_2d.matmul(&inv_freq_2d).map_err(ModelError::Candle)?;

        // Duplicate for full rot_dim.
        let freqs_full = Tensor::cat(&[&freqs, &freqs], 1).map_err(ModelError::Candle)?;

        let cos = freqs_full
            .cos()
            .map_err(ModelError::Candle)?
            .to_dtype(dtype)
            .map_err(ModelError::Candle)?;
        let sin = freqs_full
            .sin()
            .map_err(ModelError::Candle)?
            .to_dtype(dtype)
            .map_err(ModelError::Candle)?;

        Ok((cos, sin))
    }

    /// Encode pixels → merged patch embeddings.
    ///
    /// * `pixel_values` — flattened patches `[L, patch_dim]`
    /// * `grid_h`, `grid_w` — spatial grid dimensions
    fn forward(&self, pixel_values: &Tensor, grid_h: usize, grid_w: usize) -> ModelResult<Tensor> {
        // Patch embedding.
        let mut hidden_states = self.patch_embed.forward(pixel_values)?;

        // Compute 2D position IDs for the grid and gather cos/sin.
        let seq_len = hidden_states.dim(0).map_err(ModelError::Candle)?;
        let device = hidden_states.device().clone();

        // Build 2D position IDs: for position (y, x), interleave y and x positions.
        let mut pos_ids = Vec::with_capacity(seq_len);
        for y in 0..grid_h {
            for x in 0..grid_w {
                pos_ids.push(y as u32);
                pos_ids.push(x as u32);
            }
        }
        // If seq_len doesn't match grid_h * grid_w * 2 (it shouldn't — the grid is grid_h * grid_w),
        // just use sequential positions as fallback.
        let pos_tensor = if seq_len == grid_h * grid_w {
            // For each patch at (y, x), we interleave y-position RoPE and x-position RoPE
            // across the rotation dimension. Gather both and interleave.
            let y_positions: Vec<u32> = (0..grid_h)
                .flat_map(|y| (0..grid_w).map(move |_| y as u32))
                .collect();
            let x_positions: Vec<u32> = (0..grid_h)
                .flat_map(|_| (0..grid_w).map(|x| x as u32))
                .collect();
            let y_pos =
                Tensor::from_slice(&y_positions, seq_len, &device).map_err(ModelError::Candle)?;
            let x_pos =
                Tensor::from_slice(&x_positions, seq_len, &device).map_err(ModelError::Candle)?;

            // Gather cos/sin for y and x positions separately.
            let cos_y = self
                .rotary_cos
                .index_select(&y_pos, 0)
                .map_err(ModelError::Candle)?;
            let sin_y = self
                .rotary_sin
                .index_select(&y_pos, 0)
                .map_err(ModelError::Candle)?;
            let cos_x = self
                .rotary_cos
                .index_select(&x_pos, 0)
                .map_err(ModelError::Candle)?;
            let sin_x = self
                .rotary_sin
                .index_select(&x_pos, 0)
                .map_err(ModelError::Candle)?;

            // Interleave: first half from y, second half from x (each half is rot_dim/2).
            let rot_dim = self.rotary_cos.dim(1).map_err(ModelError::Candle)?;
            let half = rot_dim / 2;
            let cos = Tensor::cat(
                &[
                    &cos_y.narrow(1, 0, half).map_err(ModelError::Candle)?,
                    &cos_x.narrow(1, 0, half).map_err(ModelError::Candle)?,
                    &cos_y.narrow(1, half, half).map_err(ModelError::Candle)?,
                    &cos_x.narrow(1, half, half).map_err(ModelError::Candle)?,
                ],
                1,
            )
            .map_err(ModelError::Candle)?;
            let sin = Tensor::cat(
                &[
                    &sin_y.narrow(1, 0, half).map_err(ModelError::Candle)?,
                    &sin_x.narrow(1, 0, half).map_err(ModelError::Candle)?,
                    &sin_y.narrow(1, half, half).map_err(ModelError::Candle)?,
                    &sin_x.narrow(1, half, half).map_err(ModelError::Candle)?,
                ],
                1,
            )
            .map_err(ModelError::Candle)?;

            // Run through all blocks with 2D RoPE.
            for block in &self.blocks {
                hidden_states = block.forward(&hidden_states, Some((&cos, &sin)))?;
            }

            // Merge patches.
            return self.merger.forward(&hidden_states, grid_h, grid_w);
        } else {
            // Fallback: sequential positions.
            let positions: Vec<u32> = (0..seq_len).map(|i| i as u32).collect();
            Tensor::from_slice(&positions, seq_len, &device).map_err(ModelError::Candle)?
        };

        let cos = self
            .rotary_cos
            .index_select(&pos_tensor, 0)
            .map_err(ModelError::Candle)?;
        let sin = self
            .rotary_sin
            .index_select(&pos_tensor, 0)
            .map_err(ModelError::Candle)?;

        for block in &self.blocks {
            hidden_states = block.forward(&hidden_states, Some((&cos, &sin)))?;
        }

        self.merger.forward(&hidden_states, grid_h, grid_w)
    }
}

// ---------------------------------------------------------------------------
// Qwen2VLForConditionalGeneration
// ---------------------------------------------------------------------------

/// Qwen2-VL / Qwen2.5-VL multimodal model: vision encoder + Qwen2 LM backbone.
pub struct Qwen2VLForConditionalGeneration {
    visual: Qwen2VisionTransformer,
    language_model: LlamaForCausalLM,
    /// M-RoPE section sizes [s0, s1, s2].
    #[allow(dead_code)]
    mrope_sections: [usize; 3],
    /// Spatial merge size for computing tokens per image.
    #[allow(dead_code)]
    spatial_merge_size: usize,
    /// Stashed multimodal data for the next forward pass.
    stashed_mm_data: Option<MultimodalData>,
    /// Vision config for patch dimension computation.
    vision_config: Qwen2VisionConfig,
    dtype: DType,
}

impl Qwen2VLForConditionalGeneration {
    fn load(
        weights: &ModelWeights,
        config: &Qwen2VLConfig,
        dtype: DType,
        device: &Device,
    ) -> ModelResult<Self> {
        let visual = Qwen2VisionTransformer::load(
            weights,
            "visual",
            &config.vision_config,
            config.text_config.hidden_size,
            dtype,
            device,
        )?;

        // Load text backbone: weights under "model.*" prefix.
        let model = LlamaModel::load(weights, "model", &config.text_config, dtype, device, 0, 1)?;
        let lm_head = if config.text_config.tie_word_embeddings {
            Linear::new(model.embed_tokens.weight().clone(), None)
        } else {
            Linear::load(weights, "lm_head", dtype)?
        };
        let language_model = LlamaForCausalLM { model, lm_head };

        Ok(Self {
            visual,
            language_model,
            mrope_sections: config.mrope_sections,
            spatial_merge_size: config.spatial_merge_size,
            stashed_mm_data: None,
            vision_config: config.vision_config.clone(),
            dtype,
        })
    }

    /// Merge vision embeddings with text embeddings at placeholder positions.
    fn merge_vision_embeddings(
        &self,
        input_ids: &Tensor,
        mm_data: &MultimodalData,
    ) -> ModelResult<Tensor> {
        let text_embeds = self.language_model.model().embed(input_ids)?;
        let device = text_embeds.device().clone();

        if mm_data.images.is_empty() {
            return Ok(text_embeds);
        }

        // Process each image through the vision encoder.
        let vc = &self.vision_config;
        let p = vc.patch_size;
        let tp = vc.temporal_patch_size;
        let c = vc.in_channels;
        let patch_dim = c * tp * p * p;

        let mut all_image_embeds = Vec::new();
        for img in &mm_data.images {
            // Compute grid dimensions for this image.
            let grid_h = img.height / p;
            let grid_w = img.width / p;
            let num_patches = grid_h * grid_w;

            // Build flattened patches: [num_patches, patch_dim].
            // For static images, temporal_patch_size frames are just duplicated.
            // pixels are in [C, H, W] CHW layout.
            let pixel_tensor =
                Tensor::from_vec(img.pixels.clone(), (c, img.height, img.width), &device)
                    .map_err(ModelError::Candle)?
                    .to_dtype(self.dtype)
                    .map_err(ModelError::Candle)?;

            // Reshape to patches: [C, grid_h, P, grid_w, P] → [grid_h, grid_w, C*P*P]
            let x = pixel_tensor
                .reshape((c, grid_h, p, grid_w, p))
                .map_err(ModelError::Candle)?;
            let x = x.permute([1, 3, 0, 2, 4]).map_err(ModelError::Candle)?;
            let single_frame_dim = c * p * p;
            let x = x
                .reshape((num_patches, single_frame_dim))
                .map_err(ModelError::Candle)?;

            // Duplicate for temporal_patch_size to match expected patch_dim.
            let x = if tp > 1 {
                // Tile along last dim: [num_patches, c*p*p] → [num_patches, c*tp*p*p]
                let parts: Vec<Tensor> = (0..tp).map(|_| x.clone()).collect();
                let refs: Vec<&Tensor> = parts.iter().collect();
                Tensor::cat(&refs, 1).map_err(ModelError::Candle)?
            } else {
                x
            };

            debug_assert_eq!(x.dim(1).unwrap_or(0), patch_dim, "patch dim mismatch");

            // Encode: [num_patches, patch_dim] → [merged_patches, text_hidden]
            let image_embeds = self.visual.forward(&x, grid_h, grid_w)?;
            all_image_embeds.push(image_embeds);
        }

        // Scatter image embeddings into text embeddings.
        let mut merged = text_embeds;
        for (img_idx, placeholder) in mm_data.image_placeholders.iter().enumerate() {
            if img_idx >= all_image_embeds.len() {
                break;
            }
            let image_embeds = &all_image_embeds[img_idx];
            let n_embed_tokens = image_embeds.dim(0).map_err(ModelError::Candle)?;

            let num_tokens = merged.dim(0).map_err(ModelError::Candle)?;
            let end = (placeholder.offset + placeholder.length).min(num_tokens);
            let actual_len = end.saturating_sub(placeholder.offset);
            if actual_len == 0 {
                continue;
            }

            let use_len = actual_len.min(n_embed_tokens);
            let image_embeds = image_embeds
                .narrow(0, 0, use_len)
                .map_err(ModelError::Candle)?;

            let mut parts = Vec::new();
            if placeholder.offset > 0 {
                parts.push(
                    merged
                        .narrow(0, 0, placeholder.offset)
                        .map_err(ModelError::Candle)?,
                );
            }
            parts.push(image_embeds);
            if end < num_tokens {
                parts.push(
                    merged
                        .narrow(0, end, num_tokens - end)
                        .map_err(ModelError::Candle)?,
                );
            }
            merged = Tensor::cat(&parts, 0).map_err(ModelError::Candle)?;
        }

        Ok(merged)
    }
}

impl crate::Model for Qwen2VLForConditionalGeneration {
    fn forward(
        &self,
        input_ids: &Tensor,
        positions: &Tensor,
        kv_cache: Option<&mut crate::KvCacheStorage<'_>>,
    ) -> ModelResult<Tensor> {
        if let Some(mm_data) = &self.stashed_mm_data {
            // Multimodal forward: merge vision + text, use forward_embeds.
            let merged_embeds = self.merge_vision_embeddings(input_ids, mm_data)?;
            self.language_model
                .forward_embeds(&merged_embeds, positions, kv_cache)
        } else {
            // Text-only forward.
            self.language_model.forward(input_ids, positions, kv_cache)
        }
    }

    fn forward_embeds(
        &self,
        inputs_embeds: &Tensor,
        positions: &Tensor,
        kv_cache: Option<&mut crate::KvCacheStorage<'_>>,
    ) -> ModelResult<Tensor> {
        self.language_model
            .forward_embeds(inputs_embeds, positions, kv_cache)
    }

    fn set_mm_data(&mut self, mm_data: Option<MultimodalData>) {
        self.stashed_mm_data = mm_data;
    }

    fn num_layers(&self) -> usize {
        self.language_model.num_layers()
    }
}

// ---------------------------------------------------------------------------
// Factory functions
// ---------------------------------------------------------------------------

/// Factory function for `Qwen2VLForConditionalGeneration`.
pub fn create_qwen2_vl(
    weights: &ModelWeights,
    config: &HfModelConfig,
    dtype: DType,
    device: &Device,
    rank: usize,
    world_size: usize,
) -> ModelResult<Box<dyn crate::Model>> {
    let _ = (rank, world_size);
    let mm_config = Qwen2VLConfig::from_hf_config(config, false)?;
    let model = Qwen2VLForConditionalGeneration::load(weights, &mm_config, dtype, device)?;
    Ok(Box::new(model))
}

/// Factory function for `Qwen2_5_VLForConditionalGeneration`.
pub fn create_qwen25_vl(
    weights: &ModelWeights,
    config: &HfModelConfig,
    dtype: DType,
    device: &Device,
    rank: usize,
    world_size: usize,
) -> ModelResult<Box<dyn crate::Model>> {
    let _ = (rank, world_size);
    let mm_config = Qwen2VLConfig::from_hf_config(config, true)?;
    let model = Qwen2VLForConditionalGeneration::load(weights, &mm_config, dtype, device)?;
    Ok(Box::new(model))
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn test_vision_config() -> Qwen2VisionConfig {
        Qwen2VisionConfig {
            hidden_size: 32,
            num_heads: 4,
            depth: 2,
            patch_size: 14,
            temporal_patch_size: 2,
            in_channels: 3,
            mlp_ratio: 4.0,
            spatial_merge_size: 2,
            is_qwen25: false,
            fullatt_block_indexes: vec![],
        }
    }

    #[test]
    fn test_vision_config_from_json() {
        let json: serde_json::Value = serde_json::from_str(
            r#"{
                "embed_dim": 1280,
                "num_heads": 16,
                "depth": 32,
                "patch_size": 14,
                "temporal_patch_size": 2,
                "in_channels": 3,
                "mlp_ratio": 4.0,
                "spatial_merge_size": 2
            }"#,
        )
        .unwrap();
        let config = Qwen2VisionConfig::from_json(&json, false).unwrap();
        assert_eq!(config.hidden_size, 1280);
        assert_eq!(config.num_heads, 16);
        assert_eq!(config.depth, 32);
        assert_eq!(config.head_dim(), 80);
        assert_eq!(config.intermediate_size(), 5120);
    }

    #[test]
    fn test_qwen2_vl_config_from_hf() {
        let json: HfModelConfig = serde_json::from_str(
            r#"{
                "architectures": ["Qwen2VLForConditionalGeneration"],
                "model_type": "qwen2_vl",
                "hidden_size": 1536,
                "num_attention_heads": 12,
                "num_key_value_heads": 2,
                "num_hidden_layers": 28,
                "intermediate_size": 8960,
                "vocab_size": 151936,
                "rope_theta": 1000000.0,
                "vision_config": {
                    "embed_dim": 1280,
                    "num_heads": 16,
                    "depth": 32,
                    "patch_size": 14,
                    "temporal_patch_size": 2,
                    "in_channels": 3,
                    "mlp_ratio": 4.0,
                    "spatial_merge_size": 2
                },
                "rope_scaling": {
                    "type": "mrope",
                    "mrope_section": [16, 24, 24]
                }
            }"#,
        )
        .unwrap();

        let config = Qwen2VLConfig::from_hf_config(&json, false).unwrap();
        assert_eq!(config.vision_config.hidden_size, 1280);
        assert_eq!(config.text_config.hidden_size, 1536);
        assert_eq!(config.mrope_sections, [16, 24, 24]);
        assert_eq!(config.spatial_merge_size, 2);
    }

    #[test]
    fn test_quick_gelu() {
        let x = Tensor::new(&[0.0f32, 1.0, -1.0], &Device::Cpu).unwrap();
        let result = quick_gelu(&x).unwrap();
        let vals = result.to_vec1::<f32>().unwrap();
        // At x=0: 0 * sigmoid(0) = 0
        assert!(vals[0].abs() < 1e-5);
        // At x=1: positive
        assert!(vals[1] > 0.0);
        // At x=-1: negative but close to 0
        assert!(vals[2] < 0.0);
    }

    #[test]
    fn test_2d_rope_precompute() {
        let (cos, sin) =
            Qwen2VisionTransformer::precompute_2d_rope(16, 100, 10000.0, DType::F32, &Device::Cpu)
                .unwrap();
        assert_eq!(cos.dims(), &[100, 16]);
        assert_eq!(sin.dims(), &[100, 16]);
    }
}
